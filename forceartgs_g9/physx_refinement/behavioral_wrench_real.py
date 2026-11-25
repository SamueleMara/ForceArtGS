"""G10 wrench-driven behavioral PhysX calibration on a REAL ForceArtGS run (append-only
<run>/dynamics/g9/physx_behavioral_wrench/vNNN).

Experimental facts (given, never re-estimated): E1 known sensor-to-global transform, E2 wrench already tared at the
start, E3 contact always on, E4 qdot(t0) = 0, E5 q(t0) = q_visual(t0), E6 recorded EE pose / wrench / visual q on the
shared source-frame clock. Flow: full 6D wrench -> joint-screw effort Q_full(t) (no contact point, no classifier, no
extra tare) -> ONE uninterrupted SAPIEN PhysX rollout from (q0, 0) -> bounded multistart fit of (rho_scale, c, Td) on
the full visual trajectory -> best candidate + near-optimal family + baselines + engine readback.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .behavioral_contracts import load_wrench_config, wrench_to_fit_config


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if p and Path(p).is_file() else None


PASSIVE_KEYS = ("joint_friction", "joint_damping", "joint_stiffness", "link_linear_damping", "link_angular_damping",
                "sleep_threshold", "drive_target", "drive_velocity_target", "armature", "timestep", "gravity",
                "solver_iterations")


def passive_readback(res):
    rb = ((res.diagnostics or {}).get("readback") or {})
    return {k: rb.get(k) for k in PASSIVE_KEYS}


def model_description(cfg) -> str:
    """Human label of the simulated model. Full model: inertia + viscous damping + Coulomb friction + gravity."""
    ph = cfg["physics"]
    terms, fitted = ["inertia"], ["rho_scale"]
    if ph["friction_mode"] != "frictionless_physx":
        terms.append("viscous damping")
        fitted.append("viscous")
        if ph["coulomb_friction"] == "fitted":
            terms.append("Coulomb friction")
            fitted.append("dynamic_effort")
    if ph["gravity"] == "geometry":
        terms.append("gravity")
    off = [t for t in ("viscous damping", "Coulomb friction", "gravity") if t not in terms]
    return (f"{' + '.join(terms)}" + (f" (OFF: {', '.join(off)})" if off else " (full model)") +
            f"; fitted: {', '.join(fitted)}")


def run(run_root, scene, *, behavioral_config=None, robot_archive=None, contact_segments=None, pulse_rows=None,
        out_root=None, declared_joint_type=None, declaration_note=None, quick=False, python=None, io_response=None,
        log=print) -> dict:
    from .. import physics_lib
    from ..adapters.forceartgs_g9 import _q_noise, accepted_kinematics, build_inputs
    from ..config import load_dynamics_config
    from ..forcetwin.pipeline import _clean, _git, _mass_model, _write, next_version
    from ..wrench_input import full_record as FR
    from ..wrench_input.full_wrench_projection import WrenchContractError
    from . import behavioral_adapter as BA
    from . import behavioral_fit as BF
    from . import behavioral_report as BR
    from .behavioral_real import latest_io_response, starts_for
    from .real_data import reference_urdf
    from .sapien_diagnostic import IsolatedSapienCpu, IsolatedSapienGpu, scale_urdf_mass_inertia
    t_start = time.time()
    physics_lib.load()
    cfg = load_wrench_config(behavioral_config, quick=quick)
    if python:
        cfg["physics"]["python"] = python
    if not cfg["enabled"]:
        raise SystemExit("behavioral wrench config has enabled: false")
    fcfg = wrench_to_fit_config(cfg)
    out = next_version(Path(out_root) if out_root else Path(run_root) / "dynamics" / "g9" / "physx_behavioral_wrench")
    git = _git()
    status = dict(schema="forceartgs.g10.behavioral_wrench_status/v2", scene=Path(scene).name, out_dir=str(out),
                  git=git, quick=bool(quick), owner="ForceArtGS (estimator side; no PArtTwin1D import)",
                  claim="DEMO_REPLAY_FIT (never physical identification / transfer)")
    import yaml
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(_clean(dict(
        behavioral_wrench={k: v for k, v in cfg.items() if not k.startswith("_")},
        behavioral_config=str(behavioral_config) if behavioral_config else None, run=str(run_root), scene=str(scene),
        robot_archive=robot_archive, contact_segments=str(contact_segments) if contact_segments else None,
        pulse_rows=list(pulse_rows) if pulse_rows else None,
        declared_joint_type=declared_joint_type, declaration_note=declaration_note, quick=bool(quick), git=git)),
        sort_keys=False))
    hashes = dict(files={})

    def finish(st, reason, summary_kw=None, **extra):
        status.update(status=st, reason=reason, runtime_s=time.time() - t_start, **extra)
        _write(out / "source_hashes.json", hashes)
        _write(out / "STATUS.json", status)
        (out / "summary.md").write_text(BR.wrench_summary_md(Path(scene).name, status, **(summary_kw or {})))
        log(f"[g10-wrench] {Path(scene).name}: {st} ({reason})")
        return status
    dcfg_p = cfg["inputs"]["dynamics_config"]
    dcfg = load_dynamics_config(dcfg_p)
    dcfg["pipeline"]["allow_diagnostic_kinematics"] = True
    kin = accepted_kinematics(run_root, True, declared_joint_type, declaration_note)
    if kin.get("kind") is None:
        return finish("DATA_UNAVAILABLE", f"no joint authority under {run_root} ({kin.get('reason')}; "
                      f"selection.json present: {(Path(run_root) / 'selection.json').is_file()})")
    auth = kin["authority"]
    arc = robot_archive
    inp = build_inputs(run_root, scene, dcfg, robot_archive=arc, contact_segments=None,
                       declared_joint_type=declared_joint_type, declaration_note=declaration_note, log=log)
    rc = inp.record_context or {}
    if arc is None:
        arc = ((rc.get("rec") or {}).get("sources") or {}).get("robot_archive")
    if arc is None:
        raw = sorted(str(p) for p in (Path(scene) / "metadata" / "raw").glob("episode_*.parquet"))
        return finish("DATA_UNAVAILABLE", "no robot archive (TCP pose + six-axis wrench): none passed with "
                      "--robot-archive and the run has no canonical robot record; raw recording "
                      f"{raw or 'absent under ' + str(Path(scene) / 'metadata' / 'raw')}")
    fx = Path(run_root).parent.parent / "contact_annotations" / Path(scene).name / "contact_annotations.json"
    try:
        contract = FR.load_contract(scene, arc, fx if fx.is_file() else None)
    except (WrenchContractError, FileNotFoundError) as e:
        return finish("DATA_UNAVAILABLE", f"{e}")
    hashes.update(input_hashes=inp.input_hashes, files=dict(
        robot_archive=dict(path=str(arc), sha256=contract["robot_archive_sha256"]),
        robot_tool_json=dict(path=contract["robot_tool_json"], sha256=contract["robot_tool_sha256"]),
        sign_convention=dict(sha256=(contract["sign_provenance"] or {}).get("metadata_sha256")),
        behavioral_config=dict(path=str(behavioral_config), sha256=_sha(behavioral_config)) if behavioral_config
        else None, dynamics_config=dict(path=str(dcfg_p), sha256=_sha(dcfg_p)),
        joint_authority=dict(path=str(kin.get("authority_dir")), id=kin.get("authority_id"))))
    m = np.asarray(auth.q_valid, bool)
    qs = auth.q_sigma[m] if auth.q_sigma is not None else _q_noise(auth, m)
    qs = np.asarray(qs, float).copy()
    pos = qs[np.isfinite(qs) & (qs > 0)]
    qs[~np.isfinite(qs) | (qs <= 0)] = float(np.median(pos)) if len(pos) else 1e-3   # gauge frame (q = 0 by construction)
    jt = auth.joint_type
    fr = FR.build(contract, scene=Path(scene).name, joint_type=jt, axis=auth.axis_world, origin=auth.origin_world,
                  q_src=np.asarray(auth.source_frame_index)[m], q=np.asarray(auth.q, float)[m], q_sigma=qs,
                  legacy=(rc.get("rec") if cfg["input"]["compare_legacy_tcp_force_only"] else None),
                  baseline_s=float(cfg["input"]["noise_baseline_s"]),
                  compare_alternative_origin=bool(cfg["input"]["compare_alternative_moment_origin"]))
    hashes["full_wrench_record_sha256"] = fr.sha256()
    fr.save_npz(out / "full_wrench_joint_effort.npz")
    contract_out = dict(contract, experimental_facts=cfg["experimental_facts"], noise=fr.noise,
                        diagnostics=fr.diagnostics, joint=dict(joint_type=jt, axis_B=fr.axis_B.tolist(),
                                                               origin_B=None if fr.origin_B is None else
                                                               np.asarray(fr.origin_B).tolist(),
                                                               kinematic_status=inp.kinematic_status,
                                                               export_ready=bool(inp.kinematics_export_ready),
                                                               declaration_note=declaration_note),
                        legacy_comparison=("canonical record Q_object (TCP point force, software baseline tare) on the "
                                           "same rows" if fr.Q_legacy_tcp_force_only is not None else "not available"))
    _write(out / "input_contract.json", contract_out)
    seg_p = Path(contact_segments) if contact_segments else None
    if seg_p is None or not seg_p.is_file():
        return finish("DATA_UNAVAILABLE", f"no interaction segments file (existing force-peak + visual contact segments) "
                      f"at {seg_p}; pass --contact-segments configs/contact_intervals/<scene>.yaml",
                      summary_kw=dict(fr=fr))
    segments = BA.load_interaction_segments(seg_p)
    hashes["files"]["interaction_segments"] = dict(path=str(seg_p), sha256=_sha(seg_p), segments=segments)
    built = BA.build_wrench_episode(fr, cfg, segments)
    ev = built["evidence"]
    _write(out / "full_demo_episode.json", dict(status=built["status"], reason=built["reason"], evidence=ev,
                                                episode=built["episode"].describe() if built["episode"] else None))
    if built["status"] != "OK":
        return finish(built["status"], built["reason"], summary_kw=dict(evidence=ev, fr=fr))
    ep = built["episode"]
    log(f"[g10-wrench] {ep.scene}: {built['reason']}; record {fr.sha256()[:12]}")
    zero_g = cfg["physics"]["gravity"] == "zero"
    if inp.gravity is None and not zero_g:
        return finish("GRAVITY_UNAVAILABLE", f"{dcfg_p}: model.gravity.vector_world_mps2 is null (PhysX always "
                      "applies gravity)", summary_kw=dict(evidence=ev, fr=fr))
    mm = _mass_model(inp, cfg["physics"]["mass_mode"])
    urdf = out / "reference.urdf"
    info = reference_urdf(inp, SimpleNamespace(joint_type=jt, axis_B=fr.axis_B), mm, urdf)
    if info is None:
        return finish("MASS_GEOMETRY_UNAVAILABLE", "revolute joint without a validated mass geometry / reference pose "
                      f"(reference pose {(inp.reference_pose or {}).get('status')})", summary_kw=dict(evidence=ev, fr=fr))
    _write(out / "urdf_info.json", info)
    hashes["files"]["urdf"] = dict(path=str(urdf), sha256=_sha(urdf))
    bounds = BF.bounds_for(fcfg["model"], info["axis_inertia"], float(np.max(np.abs(ep.effort_object))))
    io_init = None
    if cfg["fitting"]["use_optional_g9_io_warmstart"]:
        io_path = Path(io_response) if io_response else latest_io_response(run_root)
        if io_path is not None and io_path.is_file():
            from ..system_identification.physx_bridge import initialize_physx_refinement
            try:
                ini = initialize_physx_refinement(json.loads(io_path.read_text()), dict(
                    fitting=dict(bounds=bounds), axis_inertia=info["axis_inertia"],
                    prior_rho_range=info["prior_rho_range"]), [], require_same_windows=False)
            except Exception as e:
                ini = dict(status="FAILED", reason=f"{type(e).__name__}: {e}")
            io_init = ini.get("x0") if ini.get("status") == "OK" else None
            hashes["files"]["io_response"] = dict(path=str(io_path), sha256=_sha(io_path), warm_start=ini.get("status"))
    starts = starts_for(bounds, info["axis_inertia"], io_init)
    g = [0.0, 0.0, 0.0] if zero_g else list(map(float, inp.gravity.g_world))
    model_label = model_description(cfg)
    log(f"[g10-wrench] model: {model_label}")
    sim_kw = dict(moving_link=info["moving_link"], joint_name=info["joint_name"], dt=float(cfg["physics"]["timestep_s"]),
                  friction_mode=cfg["physics"]["friction_mode"], python=cfg["physics"]["python"])
    gpu = cfg["physics"]["engine"] == "sapien_gpu_physx"
    Engine = IsolatedSapienGpu if gpu else IsolatedSapienCpu
    if gpu:
        sim_kw["device"] = cfg["physics"]["device"]
    try:
        sim = Engine(urdf, gravity=g, **sim_kw)
    except Exception as e:
        return finish("ENGINE_FAILED", f"{type(e).__name__}: {e}", summary_kw=dict(evidence=ev, fr=fr))
    roll = BF.CachingRollout(sim)
    alt, alt_q = None, None
    try:
        fit = BF.fit_full_demo(roll, ep, fcfg, bounds, starts, log=log)
        p = fit["_params"]
        fam = (BF.profile_family(roll, fit, fcfg, bounds, log=log) if fcfg["fitting"]["profile_near_optimal_family"]
               else dict(rule="disabled", profiles={}, members=[dict(fit["best"], label="best")], not_unique=True,
                         n_members=1, tolerance_sigma=fcfg["fitting"]["family_tolerance_sigma"], bound_limited=[]))
        hyps, skipped = BA.wrench_input_hypotheses(fr, ep, cfg, seed=int(cfg["fitting"]["seed"]))
        sens = BF.input_sensitivity(roll, ep, hyps, p, fcfg)
        sens["skipped"] = skipped
        work = BF.work_audit(ep, potential=BF.gravity_potential(info, p.rho_scale, g))
        policy = BF.policy_spread(roll, ep, fam["members"], fcfg)
        caps = sim.capabilities()
        pe = ep.to_physx_episode()
        res_best = fit["_result"]
        base_p = BF.PhysicalParameters(**fit["baselines"]["prior_reference"]["params"])
        res_base = roll(base_p, pe)
        rb_best, rb_base = passive_readback(res_best), passive_readback(res_base)
        if cfg["diagnostics"]["zero_gravity_alternative"] and not zero_g:
            sim0 = Engine(urdf, gravity=[0.0, 0.0, 0.0], **sim_kw)
            try:
                roll0 = BF.CachingRollout(sim0)
                alt = BF.zero_gravity_alternative(roll0, ep, fcfg, bounds, p, log=log)
                if alt.get("status") == "FITTED":
                    alt_q = np.asarray(roll0(BF.PhysicalParameters(**alt["best"]["params"]),
                                             ep.to_physx_episode(name="full_demo_zero_gravity")).q)
            finally:
                sim0.close()
        dg = res_best.diagnostics or {}
        readback = dict(engine=dict(backend=sim.backend_process, timestep=sim.dt, gravity=g,
                                    gravity_source=cfg["physics"]["gravity"], model=model_label,
                                    friction_mode=cfg["physics"]["friction_mode"],
                                    runtime_semantics=res_best.runtime_semantics,
                                    sapien_version=res_best.engine_version,
                                    pipeline=("PhysX GPU (PhysxGpuSystem, batched sub-scenes)" if gpu else
                                              "PhysX CPU (PhysxCpuSystem)"),
                                    device=(cfg["physics"]["device"] if gpu else "cpu"),
                                    n_gpu_batches=roll.n_batches),
                        passive_terms_at_best=rb_best, passive_terms_at_prior_reference=rb_base,
                        passive_terms_invariant_across_runs=bool(json.dumps(rb_best, sort_keys=True, default=str) ==
                                                                 json.dumps(rb_base, sort_keys=True, default=str)),
                        readback_at_best=dg.get("readback"), engine_defaults_as_loaded=dg.get("engine_defaults_as_loaded"),
                        axis_inertia_engine=dg.get("axis_inertia_engine"),
                        axis_inertia_expected=dg.get("axis_inertia_expected"), n_contacts=dg.get("n_contacts"),
                        sleeping_at_end=dg.get("sleeping_at_end"), engine_initial_state=fit["engine_initial_state"],
                        capabilities=caps, urdf_sha256=_sha(urdf), n_engine_calls=roll.n_engine_calls,
                        n_rollouts=sim.n_rollouts)
    except RuntimeError as e:
        sim.close()
        return finish("ENGINE_FAILED", str(e), summary_kw=dict(evidence=ev, fr=fr))
    sim.close()
    pulse = BR.pulse_comparison(fr, pulse_rows)
    cls = BR.classify_wrench(fit, fcfg)
    cand_urdf = None
    if cls["status"] == "BEHAVIORAL_FIT_OK" and cfg["output"]["write_candidate_urdf"]:
        cu = out / "behavioral_candidate.urdf"
        scale_urdf_mass_inertia(urdf, cu, info["moving_link"], p.rho_scale)
        cand_urdf = dict(path=str(cu), sha256=_sha(cu), note="reference geometry with mass / inertia scaled by "
                         "rho_scale; joint <dynamics> are 0: friction is the explicit surrogate given in "
                         "behavioral_simulator_candidate.json")
    prov = dict(kinematic_status=inp.kinematic_status, export_ready=bool(inp.kinematics_export_ready),
                model=model_label,
                gravity=("ZERO (simplified model, declared in the behavioral config; NOT a measured gravity)" if zero_g
                         else inp.gravity.provenance),
                gravity_verified=(False if zero_g else bool(inp.gravity.verified)),
                declaration_note=declaration_note)
    cand = BR.wrench_candidate(scene=Path(scene).name, run=run_root, ep=ep, fr=fr, fit=fit, family=fam, cls=cls,
                               info=info, readback=readback, hashes=hashes, input_sens=sens, work=work, policy=policy,
                               alternative=alt, pulse=pulse, provenance=prov, candidate_urdf=cand_urdf)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    _write(out / "baseline_replay.json", dict(baselines=fit["baselines"], warm_start_used=fit["warm_start_used"],
                                              improvement=fit["improvement"],
                                              prior_reference_rollout=dict(t=ep.t_q_s.tolist(),
                                                                           q=np.asarray(res_base.q).tolist())))
    _write(out / "best_behavioral_replay.json", dict(
        strip(fit), input_sensitivity=sens, work_audit=work,
        policy_probe={k: v for k, v in policy.items() if k != "q_members"}, zero_gravity_alternative=alt,
        pulse_comparison=pulse,
        best_rollout=dict(t=ep.t_q_s.tolist(), q=np.asarray(res_best.q).tolist(),
                          qdot=None if res_best.qdot is None else np.asarray(res_best.qdot).tolist(),
                          q_obs=ep.q_observed.tolist(), q_sigma=ep.q_sigma.tolist())))
    _write(out / "near_optimal_family.json", fam)
    _write(out / "engine_readback.json", readback)
    _write(out / "behavioral_simulator_candidate.json", cand)
    try:
        BR.wrench_plots(out / "plots", ep, fr, fit, fam, roll, res_base, pulse_rows, alternative_q=alt_q)
    except Exception as e:                                       # plotting never hides the numbers
        log(f"[g10-wrench] plotting failed: {type(e).__name__}: {e}")
    lab = cand["labels"]
    return finish(cls["status"], f"rmse/sigma {fit['best']['metrics']['rmse_over_sigma']:.3g}; preview "
                  f"{cand['status']['behavioral_preview_allowed']}",
                  summary_kw=dict(evidence=ev, fr=fr, ep=ep, fit=fit, family=fam, cand=cand, input_sens=sens,
                                  work=work, policy=policy, alternative=alt, pulse=pulse),
                  labels=lab, parameters=cand["candidate"]["parameters"],
                  training_rmse=fit["best"]["metrics"]["rmse"],
                  training_rmse_over_sigma=fit["best"]["metrics"]["rmse_over_sigma"],
                  n_rollouts=readback["n_rollouts"], simulation_export_allowed=False)
