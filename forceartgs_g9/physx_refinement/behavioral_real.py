"""G10-A behavioral full-demo PhysX replay on a REAL ForceArtGS run (append-only <run>/dynamics/g9/physx_behavioral/vNNN).

Data owner: the existing G9 inputs (joint authority, canonical robot record, force audit, mass geometry, gravity) --
the same as the G10 refinement. Nothing is recalibrated, re-projected, re-tared, sign-flipped or cropped. The URDF is
the existing G10 reference URDF (``real_data.reference_urdf``); the engine is the existing ``IsolatedSapienCpu``.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .behavioral_contracts import load_config


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if p and Path(p).is_file() else None


def latest_io_response(run_root):
    vs = sorted((Path(run_root) / "dynamics" / "g9" / "io_identification").glob("v[0-9][0-9][0-9]/io_response.json"))
    return vs[-1] if vs else None


def starts_for(bounds, I_ref, io_init=None) -> dict:
    """Deterministic, physically admissible baselines / warm starts: the prior reference (rho_s = 1, frictionless),
    low / high inertia x low / high viscous corners, and the io warm start when available."""
    from .optimizer import from_vector
    lo, hi = bounds["rho_scale"]
    cmax = bounds["viscous"][1]
    arm = bounds["armature"][0]
    s = dict(prior_reference=from_vector([min(max(1.0, lo), hi), arm, 0.0, 0.0, 0.0]))
    for rl, r in (("low_inertia", max(lo, 0.3)), ("high_inertia", min(hi, 3.0))):
        for cl, c in (("low_viscous", 0.02 * cmax), ("high_viscous", 0.3 * cmax)):
            s[f"{rl}_{cl}"] = from_vector([r, arm, 0.0, 0.0, c])
    if io_init is not None:
        x = io_init
        s["io_warm_start"] = from_vector([x["rho_scale"], x["armature"], x["dynamic_effort"], x["static_excess"],
                                          x["viscous"]])
    return s


def run(run_root, scene, *, behavioral_config=None, robot_archive=None, contact_segments=None, pulse_rows=None,
        out_root=None, declared_joint_type=None, declaration_note=None, quick=False, python=None, io_response=None,
        log=print) -> dict:
    from ..adapters.forceartgs_g9 import build_inputs
    from ..config import load_dynamics_config
    from ..forcetwin import record as RC
    from ..forcetwin.pipeline import _clean, _git, _mass_model, _write, next_version
    from ..free_forced import segments as SG
    from ..free_forced.pipeline import _part_points, load_options as ff_options
    from . import behavioral_adapter as BA
    from . import behavioral_fit as BF
    from . import behavioral_report as BR
    from .real_data import reference_urdf
    from .robot_replay import detect_backends
    from .sapien_diagnostic import IsolatedSapienCpu, scale_urdf_mass_inertia
    t_start = time.time()
    cfg = load_config(behavioral_config, quick=quick)
    if python:
        cfg["simulation"]["python"] = python
    if not cfg["enabled"]:
        raise SystemExit("behavioral config has enabled: false")
    ffo, cfg_ff = ff_options(cfg["inputs"]["free_forced_config"])
    out = next_version(Path(out_root) if out_root else Path(run_root) / "dynamics" / "g9" / "physx_behavioral")
    git = _git()
    status = dict(schema="forceartgs.g10.behavioral_status/v1", scene=Path(scene).name, out_dir=str(out), git=git,
                  quick=bool(quick), owner="ForceArtGS (estimator side; no PArtTwin1D import)",
                  claim="BEHAVIORAL_FIT_TO_DEMONSTRATION (never physical identification)")
    import yaml
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(_clean(dict(
        behavioral={k: v for k, v in cfg.items() if not k.startswith("_")},
        behavioral_config=str(behavioral_config) if behavioral_config else None, dynamics_config=str(cfg_ff),
        run=str(run_root), scene=str(scene), robot_archive=robot_archive, contact_segments=contact_segments,
        pulse_rows=list(pulse_rows) if pulse_rows else None, declared_joint_type=declared_joint_type,
        declaration_note=declaration_note, quick=bool(quick), git=git)), sort_keys=False))
    dcfg = load_dynamics_config(cfg_ff)
    dcfg["pipeline"]["allow_diagnostic_kinematics"] = True
    inp = build_inputs(run_root, scene, dcfg, robot_archive=robot_archive, contact_segments=contact_segments,
                       declared_joint_type=declared_joint_type, declaration_note=declaration_note, log=log)
    kp = inp.kinematics_provenance or {}
    hashes = dict(input_hashes=inp.input_hashes, files={k: dict(path=str(v), sha256=_sha(v)) for k, v in dict(
        behavioral_config=behavioral_config, dynamics_config=cfg_ff, free_forced_config=cfg["inputs"]["free_forced_config"],
        robot_archive=robot_archive, contact_segments=contact_segments,
        joint_authority=kp.get("authority_path") or kp.get("path") or kp.get("authority_dir")).items() if v})

    def finish(st, reason, summary_kw=None, **extra):
        status.update(status=st, reason=reason, runtime_s=time.time() - t_start, **extra)
        _write(out / "source_hashes.json", hashes)
        _write(out / "STATUS.json", status)
        (out / "summary.md").write_text(BR.summary_md(Path(scene).name, status, **(summary_kw or {})))
        log(f"[g10-behavioral] {Path(scene).name}: {st} ({reason})")
        return status
    try:
        rec = RC.from_g9_inputs(inp, contact_segments=contact_segments)
    except RC.RecordError as e:
        fa = inp.force_audit or {}
        if fa.get("status") == "NOT_EVALUATED" and inp.kinematic_status not in ("IDENTIFIED", "OPERATOR_DECLARED"):
            return finish("KINEMATICS_NOT_ACCEPTED", f"upstream joint status {inp.kinematic_status}: no joint "
                          "authority -- nothing simulated (nothing fabricated)")
        return finish("DATA_UNAVAILABLE", str(e))
    hashes["record_sha256"] = rec.sha256()
    fa_status = (inp.force_audit or {}).get("status")
    kin_ev = dict(kinematic_status=inp.kinematic_status, export_ready=bool(inp.kinematics_export_ready),
                  axis_B=np.asarray(rec.axis_B).tolist(), axis_std_rad=inp.axis_std_rad,
                  origin_std_m=inp.origin_std_m, q_reference=rec.q_reference.get("q_ref"),
                  declaration_note=declaration_note, provenance=kp)
    cal_ev = dict(force_audit_status=fa_status, gates=rec.force_calibration_assumptions,
                  object_side_multiplier=rec.provenance.get("object_side_multiplier"))
    built = BA.build_full_demo_episode(rec, cfg, kinematics_evidence=kin_ev, calibration_evidence=cal_ev,
                                       provenance=dict(run=str(run_root), scene=str(scene)))
    ev = built["evidence"]
    # classifier regimes: EVIDENCE only (never crop / reset / zero the replay)
    so = ffo["segments"]
    try:
        q_ref = float((inp.reference_pose or {}).get("q_ref", 0.0) or 0.0)
        cl = SG.classify(rec, z_sig=so["z_sig"], win_rows=so["win_rows"], v_k=so["v_k"],
                         sep_margin_m=so["sep_margin_m"], part_points_ref=_part_points(inp, int(so["n_part_points"])),
                         q_ref=q_ref, min_run=so["min_run"], pulse_z=so["pulse_z"], z_J=so["z_J"])
        segs = SG.build_segments(rec, cl, free_hypotheses=ffo["free_hypotheses"], min_q=so["min_q"], v_k=so["v_k"])
        regimes = dict(interval_summary=cl.get("summary"), segments=[s.describe() for s in segs],
                       use="evidence only: the behavioral replay applies the recorded input on every row")
    except Exception as e:                                       # evidence; never blocks the replay
        regimes = dict(status="CLASSIFIER_FAILED", reason=f"{type(e).__name__}: {e}")
    _write(out / "input_coverage_and_contact_evidence.json", dict(
        start_and_coverage=ev, g9_classifier_regimes=regimes,
        contact_evidence=(built["episode"].contact_evidence if built["episode"] is not None else None)))
    _write(out / "full_demo_episode.json", dict(status=built["status"], reason=built["reason"], evidence=ev,
                                                episode=(built["episode"].describe() if built["episode"] else None)))
    if built["status"] != "OK":
        return finish(built["status"], built["reason"], summary_kw=dict(evidence=ev))
    ep = built["episode"]
    log(f"[g10-behavioral] {Path(scene).name}: t0 {ep.t0_s:.4g} s (frame {ep.source_start_frame}), q0 "
        f"{ep.q0_observed:.6g}, v0 = 0 ({ep.start_basis}); {len(ep.t_q_s)} q samples over {ep.duration_s:.3g} s, "
        f"{len(ep.effort_t_s)} effort rows; record {rec.sha256()[:12]}")
    if inp.gravity is None:
        return finish("GRAVITY_UNAVAILABLE", "PhysX always applies gravity; no registered gravity direction",
                      summary_kw=dict(evidence=ev))
    mm = _mass_model(inp, cfg["model"]["mass_mode"])
    urdf = out / "reference.urdf"
    info = reference_urdf(inp, rec, mm, urdf)
    if info is None:
        return finish("MASS_GEOMETRY_UNAVAILABLE", "revolute joint without a validated mass geometry / reference pose",
                      summary_kw=dict(evidence=ev))
    _write(out / "urdf_info.json", info)
    hashes["files"]["urdf"] = dict(path=str(urdf), sha256=_sha(urdf))
    bounds = BF.bounds_for(cfg["model"], info["axis_inertia"], float(np.max(np.abs(ep.effort_object))))
    io_init = None
    if cfg["fitting"]["initialize_from_io"]:
        io_path = Path(io_response) if io_response else latest_io_response(run_root)
        if io_path is not None and io_path.is_file():
            from ..system_identification.physx_bridge import initialize_physx_refinement
            try:
                ini = initialize_physx_refinement(json.loads(io_path.read_text()), dict(
                    fitting=dict(bounds=bounds), axis_inertia=info["axis_inertia"],
                    prior_rho_range=info["prior_rho_range"]), [], require_same_windows=False)
            except Exception as e:
                ini = dict(status="FAILED", reason=f"{type(e).__name__}: {e}")
            ini["io_response"] = dict(path=str(io_path), sha256=_sha(io_path))
            _write(out / "io_initialization.json", ini)
            io_init = ini.get("x0") if ini.get("status") == "OK" else None
            log(f"[g10-behavioral] io warm start: {ini.get('status')}")
    starts = starts_for(bounds, info["axis_inertia"], io_init)
    g = list(map(float, inp.gravity.g_world))
    sim_kw = dict(moving_link=info["moving_link"], joint_name=info["joint_name"], dt=float(cfg["simulation"]["timestep"]),
                  friction_mode=cfg["model"]["friction_mode"], python=cfg["simulation"]["python"])
    sim = IsolatedSapienCpu(urdf, gravity=g, **sim_kw)
    roll = BF.CachingRollout(sim)
    alt = None
    alt_q = None
    try:
        fit = BF.fit_full_demo(roll, ep, cfg, bounds, starts, log=log)
        p = fit["_params"]
        fam = (BF.profile_family(roll, fit, cfg, bounds, log=log) if cfg["fitting"]["profile_near_optimal_family"]
               else dict(rule="disabled", profiles={}, members=[dict(fit["best"], label="best")], not_unique=True,
                         n_members=1, tolerance_sigma=cfg["fitting"]["family_tolerance_sigma"]))
        hyps, skipped = BA.input_hypotheses(rec, ep, cfg, seed=int(cfg["fitting"]["seed"]))
        sens = BF.input_sensitivity(roll, ep, hyps, p, cfg)
        sens["skipped"] = skipped
        work = BF.work_audit(ep, potential=BF.gravity_potential(info, p.rho_scale, g))
        policy = BF.policy_spread(roll, ep, fam["members"], cfg)
        caps = sim.capabilities()
        rb = (fit["_result"].diagnostics or {})
        readback = dict(engine=dict(backend=sim.backend_process, timestep=sim.dt, gravity=g,
                                    friction_mode=cfg["model"]["friction_mode"],
                                    runtime_semantics=fit["_result"].runtime_semantics,
                                    engine_version=fit["_result"].engine_version),
                        readback_at_best=rb.get("readback"), engine_defaults_as_loaded=rb.get("engine_defaults_as_loaded"),
                        axis_inertia_engine=rb.get("axis_inertia_engine"),
                        axis_inertia_expected=rb.get("axis_inertia_expected"), n_contacts=rb.get("n_contacts"),
                        sleeping_at_end=rb.get("sleeping_at_end"), engine_initial_state=fit["engine_initial_state"],
                        capabilities=caps, urdf_sha256=_sha(urdf), n_engine_calls=roll.n_engine_calls)
        if cfg["alternatives"]["zero_gravity_load"]:
            sim0 = IsolatedSapienCpu(urdf, gravity=[0.0, 0.0, 0.0], **sim_kw)
            try:
                roll0 = BF.CachingRollout(sim0)
                alt = BF.zero_gravity_alternative(roll0, ep, cfg, bounds, p, log=log)
                if alt.get("status") == "FITTED":
                    from .objective import PhysicalParameters
                    alt_q = np.asarray(roll0(PhysicalParameters(**alt["best"]["params"]),
                                             ep.to_physx_episode(name="full_demo_zero_gravity")).q)
                    readback["n_engine_calls_alternative"] = roll0.n_engine_calls
            finally:
                sim0.close()
        backends = detect_backends(inp.record_context)
        n_roll = sim.n_rollouts
        sim_settings = dict(engine="SAPIEN 3 PhysX CPU (isolated one-joint articulation)",
                            sapien_version=fit["_result"].engine_version, timestep=sim.dt, gravity=g,
                            friction_mode=cfg["model"]["friction_mode"],
                            runtime_semantics=fit["_result"].runtime_semantics, process=sim.backend_process,
                            passive_defaults_zeroed_and_read_back=True, contacts_allowed=False)
        try:
            BR.plots(out / "plots", ep, fit, fam, policy, roll, alternative_q=alt_q)
        except Exception as e:                                   # plotting never hides the numbers
            log(f"[g10-behavioral] plotting failed: {type(e).__name__}: {e}")
    except RuntimeError as e:
        return finish("BEHAVIORAL_FIT_FAILED", str(e), summary_kw=dict(evidence=ev))
    finally:
        try:
            sim.close()
        except Exception:
            pass
    # pulse audit (fridge): a DIAGNOSTIC of the conditional input model, never a sign change
    pulse = None
    if pulse_rows:
        from ..validation.impulse_balance import audit
        try:
            ia = audit(rec, pulse_rows, observability=(inp.record_context or {}).get("observability"),
                       robot_record=(inp.record_context or {})["rec"])
            pulse = dict(rows=list(pulse_rows), decision=ia.get("decision"), primary=ia["decision"]["primary"])
        except Exception as e:
            pulse = dict(rows=list(pulse_rows), primary="AUDIT_FAILED", error=f"{type(e).__name__}: {e}")
    lab = ep.contact_evidence["counts"]
    cond = []
    if inp.kinematic_status != "IDENTIFIED" or not inp.kinematics_export_ready:
        cond.append(f"joint kinematics {inp.kinematic_status} (export_ready {inp.kinematics_export_ready}): the "
                    "axis / type are a working authority")
    if fa_status != "VERIFIED":
        cond.append(f"force input {fa_status}: canonical object-side effort with unverified calibration gates "
                    f"{sorted(k for k, v in (rec.force_calibration_assumptions or {}).items() if v != 'VERIFIED')}")
    if not inp.gravity.verified:
        cond.append("gravity direction DECLARED, not registered")
    if ep.contact_evidence["conditional_rows"]:
        cond.append(f"{ep.contact_evidence['conditional_rows']} of {len(ep.effort_t_s)} input rows are not "
                    f"confirmed contact {lab}: their recorded effort is applied as a CONDITIONAL measured input")
    if not ep.start_is_full_demo:
        cond.append(f"rest start verified at a CROPPED authority start (frames {ev['uncovered_source_frames']} not "
                    "covered by the joint authority)")
    cond.append("friction realized by the explicit-effort surrogate (no stiction, not native PhysX joint friction)")
    if pulse and pulse["primary"] != "IMPULSE_CONSISTENT":
        cond.append(f"declared pulse rows {pulse_rows[0]}-{pulse_rows[1]}: {pulse['primary']} (input-model "
                    "inconsistency; no sign change)")
    if work["verdict"] != "PASSIVE_ENERGY_CONSISTENT":
        cond.append(f"work audit {work['verdict']}: the recorded input cannot passively explain the motion")
    if alt and alt.get("status") == "FITTED" and alt["best"]["metrics"]["rmse_over_sigma"] + \
            float(cfg["fitting"]["family_tolerance_sigma"]) < fit["best"]["metrics"]["rmse_over_sigma"]:
        cond.append(f"the separately labelled zero-gravity hypothesis replays better (rmse/sigma "
                    f"{alt['best']['metrics']['rmse_over_sigma']:.3g} vs {fit['best']['metrics']['rmse_over_sigma']:.3g})"
                    ": gravity geometry unresolved")
    conditional = dict(input=fa_status != "VERIFIED" or bool(ep.contact_evidence["conditional_rows"]), list=cond)
    cls = BR.classify(fit, cfg, input_verified=fa_status == "VERIFIED", conditional=conditional["input"])
    cand_urdf = None
    preview = cls["adequate"] and cfg["output"]["allow_preview_without_unique_physical_id"]
    if preview and cfg["output"]["write_behavioral_candidate"]:
        cu = out / "behavioral_candidate.urdf"
        scale_urdf_mass_inertia(urdf, cu, info["moving_link"], p.rho_scale)
        cand_urdf = dict(path=str(cu), sha256=_sha(cu), note="reference geometry with mass / inertia scaled by "
                         "rho_scale; joint <dynamics> are 0: friction is the explicit surrogate in "
                         "behavioral_simulator_candidate.json (SAPIEN URDF tags cannot express it)")
    cand = BR.candidate(scene=Path(scene).name, run=run_root, ep=ep, fit=fit, family=fam, cls=cls, info=info,
                        sim_settings=sim_settings, hashes=hashes, conditional=conditional, input_sens=sens, work=work,
                        policy=policy, alternative=alt, capabilities=caps, backends=backends, candidate_urdf=cand_urdf,
                        cfg=cfg)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    _write(out / "baseline_replay.json", dict(baselines=fit["baselines"], warm_start_used=fit["warm_start_used"],
                                              improvement=fit["improvement"]))
    _write(out / "behavioral_fit.json", dict(strip(fit), input_sensitivity=sens, work_audit=work,
                                             policy_probe={k: v for k, v in policy.items() if k != "q_members"},
                                             zero_gravity_alternative=alt, pulse_audit=pulse,
                                             robot_replay_backends=backends,
                                             best_rollout=dict(t=ep.t_q_s.tolist(), q=np.asarray(fit["_result"].q).tolist(),
                                                               qdot=None if fit["_result"].qdot is None else
                                                               np.asarray(fit["_result"].qdot).tolist(),
                                                               q_obs=ep.q_observed.tolist(), q_sigma=ep.q_sigma.tolist())))
    _write(out / "near_optimal_family.json", fam)
    _write(out / "simulator_readback.json", readback)
    if cfg["output"]["write_behavioral_candidate"]:
        _write(out / "behavioral_simulator_candidate.json", cand)
    st = cand["status"]
    return finish(st["behavioral_status"], f"{st['replay_label']}; preview {st['behavioral_preview_allowed']}",
                  summary_kw=dict(ep=ep, fit=fit, family=fam, cand=cand, input_sens=sens, work=work, policy=policy,
                                  alternative=alt, evidence=ev),
                  replay_label=st["replay_label"], behavioral_preview_allowed=st["behavioral_preview_allowed"],
                  training_rmse=fit["best"]["metrics"]["rmse"],
                  training_rmse_over_sigma=fit["best"]["metrics"]["rmse_over_sigma"],
                  parameters=cand["candidate"]["parameters"], not_unique=st["not_unique"],
                  start_basis=ep.start_basis, n_rollouts=n_roll, simulation_export_allowed=False,
                  sim_to_real_validated=False)
