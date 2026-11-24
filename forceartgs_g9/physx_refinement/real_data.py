"""G10 PhysX refinement on a real ForceArtGS run (estimator side; append-only <run>/dynamics/g9/physx_refinement/vNNN).

Data owner: the existing G9 inputs (joint authority, canonical robot record, force audit, mass geometry, gravity).
Nothing is recalibrated, re-projected, re-tared or sign-flipped. Regimes come from the SAME classifier as the G9
free/forced stage (forceartgs_g9.free_forced.segments) with the SAME options file; only FREE_CONFIRMED and
FORCED_MEASURED intervals become episodes, UNKNOWN / FREE_HYPOTHESIS intervals are listed as excluded.

Engine: SAPIEN 3 / PhysX CPU (the diagnostic runner), default friction mode ``external_effort_surrogate`` ->
every result is DIAGNOSTIC_ONLY_NOT_IDENTIFIED and ``simulation_export_allowed`` is False.

Scale: the URDF carries the G9 hull geometry at the reference density rho_0 = sqrt(I_lo I_hi) / M_1 (geometric mean
of the existing size prior ``_prior_for_sim``); ``rho_scale`` is relative to it. Free-only fits hold rho_scale = 1
and armature = 0 (gauge) and report the prior-conditioned correlated family.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

DEFAULTS = dict(
    free_forced_config="configs/g9_real/free_forced.yaml",
    mass_mode="uniform_solid",
    v0_prefix_samples=4, min_samples=4,
    heldout="last_if_multiple",                    # last_if_multiple | none
    physics=dict(engine="sapien_cpu_diagnostic", timestep=0.001, friction_mode="external_effort_surrogate",
                 python=None),
    loss=dict(huber_delta=2.0, numeric_floor_q=3e-4),   # sigma_num: PhysX float32 q accumulation (measured)
    fitting=dict(rho_bounds=[0.1, 10.0], lam_max=20.0, gam_max=20.0, effort_bound_factor=3.0, armature=0.0,
                 seed=0, maxiter=30, popsize=8, polish=True, tol=1e-3,
                 v0_nuisance_bound_sd=0.0),               # > 0 (with loss.fit_initial_velocity) fits v0 offsets
    profile=dict(n=9, inner_maxiter=15, inner_popsize=6, inner_polish=True),
)
QUICK = dict(fitting=dict(maxiter=3, popsize=4, polish=False),
             profile=dict(n=5, inner_maxiter=2, inner_popsize=4, inner_polish=False))


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_options(path=None, quick=False):
    import yaml
    o = _merge(DEFAULTS, yaml.safe_load(Path(path).read_text()) if path else {})
    return _merge(o, QUICK) if quick else o


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if p and Path(p).is_file() else None


def _bounds(o, I_ref, Q_max, free_only):
    f = o["fitting"]
    rlo, rhi = (1.0, 1.0) if free_only else map(float, f["rho_bounds"])
    I_hi = I_ref * rhi
    Td = max(f["gam_max"] * I_hi, f["effort_bound_factor"] * Q_max)
    return dict(rho_scale=(rlo, rhi), armature=(float(f["armature"]),) * 2, dynamic_effort=(0.0, float(Td)),
                static_excess=(0.0, 0.0), viscous=(0.0, float(f["lam_max"] * I_hi)))


def reference_urdf(inp, rec, mm, path):
    """Single-joint URDF of the G9 geometry at the PRIOR reference density rho_0 = sqrt(I_lo I_hi) / M_1 (shared by
    the G10 refinement and the input-output identification reconciliation). None: revolute without geometry."""
    from ..forcetwin.pipeline import _prior_for_sim
    from . import urdf_builder as UB
    prior = _prior_for_sim(inp, mm)
    pr_rng = prior.get("I_eff_range") or [2.0, 20.0]
    I_ref = math.sqrt(pr_rng[0] * pr_rng[1])
    if mm is not None:
        info = UB.from_mass_model(mm, I_ref / mm.M1, path)
    elif rec.joint_type == "prismatic":
        info = UB.write_single_joint_urdf(path, joint_type="prismatic", axis=rec.axis_B, origin=np.zeros(3),
                                          mass=I_ref, com_ref=np.zeros(3), inertia_com_ref=np.eye(3) * 1e-3 * I_ref)
        info["note"] = "no mass geometry: point mass (prismatic dynamics depend on m only)"
    else:
        return None
    info.update(prior=prior, I_ref=I_ref, prior_rho_range=[pr_rng[0] / I_ref, pr_rng[1] / I_ref],
                scale_provenance="PRIOR_ONLY: reference density = geometric mean of the size prior, NOT measured")
    return info


def run(run_root, scene, *, g10_config=None, cfg_path=None, robot_archive=None, contact_segments=None,
        pulse_rows=None, out_root=None, declared_joint_type=None, declaration_note=None, quick=False, python=None,
        io_init=None, log=print) -> dict:
    from .. import physics_lib
    from ..adapters.forceartgs_g9 import build_inputs
    from ..config import load_dynamics_config
    from ..forcetwin import record as RC
    from ..forcetwin.pipeline import _clean, _git, _mass_model, _write, next_version
    from ..free_forced import segments as SG
    from ..free_forced.pipeline import _part_points, load_options as ff_options
    from . import g9_adapter as GA
    from .api import estimate
    from .sapien_diagnostic import IsolatedSapienCpu
    t0 = time.time()
    physics_lib.load()
    o = load_options(g10_config, quick=quick)
    if python:
        o["physics"]["python"] = python
    ffo, cfg_ff = ff_options(o["free_forced_config"])
    cfg_path = cfg_path or cfg_ff
    out = next_version(Path(out_root) if out_root else Path(run_root) / "dynamics" / "g9" / "physx_refinement")
    git = _git()
    status = dict(schema="forceartgs.g10.physx_refinement_status/v1", scene=Path(scene).name, out_dir=str(out),
                  git=git, quick=bool(quick), owner="ForceArtGS (estimator side; no PArtTwin1D import)")
    cfg = load_dynamics_config(cfg_path)
    cfg["pipeline"]["allow_diagnostic_kinematics"] = True
    inp = build_inputs(run_root, scene, cfg, robot_archive=robot_archive, contact_segments=contact_segments,
                       declared_joint_type=declared_joint_type, declaration_note=declaration_note, log=log)
    import yaml
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(_clean(dict(
        g10=o, g10_config=str(g10_config) if g10_config else None, dynamics_config=str(cfg_path),
        free_forced_options=ffo, run=str(run_root), scene=str(scene), robot_archive=robot_archive,
        contact_segments=contact_segments, pulse_rows=list(pulse_rows) if pulse_rows else None,
        declared_joint_type=declared_joint_type, declaration_note=declaration_note, git=git)), sort_keys=False))
    kp = inp.kinematics_provenance or {}
    hashes = dict(input_hashes=inp.input_hashes, files={k: dict(path=str(v), sha256=_sha(v)) for k, v in dict(
        g10_config=g10_config, dynamics_config=cfg_path, free_forced_config=o["free_forced_config"],
        robot_archive=robot_archive, contact_segments=contact_segments,
        joint_authority=kp.get("authority_path") or kp.get("path")).items() if v})

    def finish(st, reason, **extra):
        status.update(status=st, reason=reason, runtime_s=time.time() - t0, **extra)
        _write(out / "source_hashes.json", hashes)
        _write(out / "STATUS.json", status)
        log(f"[g10] {Path(scene).name}: {st} ({reason})")
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
    jt = rec.joint_type
    if inp.gravity is None:
        return finish("GRAVITY_UNAVAILABLE", "PhysX always applies gravity; no registered gravity direction")
    # ------------------------------------------------------------------ regimes (same classifier + options as G9)
    so = ffo["segments"]
    q_ref = float((inp.reference_pose or {}).get("q_ref", 0.0) or 0.0)
    ev = SG.classify(rec, z_sig=so["z_sig"], win_rows=so["win_rows"], v_k=so["v_k"], sep_margin_m=so["sep_margin_m"],
                     part_points_ref=_part_points(inp, int(so["n_part_points"])), q_ref=q_ref,
                     min_run=so["min_run"], pulse_z=so["pulse_z"], z_J=so["z_J"])
    segs = SG.build_segments(rec, ev, free_hypotheses=ffo["free_hypotheses"], min_q=so["min_q"], v_k=so["v_k"])
    fa_status = (inp.force_audit or {}).get("status")
    inter, regime, v0s, cal, excluded = GA.interactions_from_segments(
        segs, force_audit_status=fa_status, v0_prefix_samples=o["v0_prefix_samples"], min_samples=o["min_samples"])
    regime_summary = dict(interval_summary=ev.get("summary"), force_audit=fa_status,
                          segments=[s.describe() for s in segs], admissible=list(regime), excluded=excluded)
    _write(out / "regime_evidence.json", regime_summary)
    if not inter:
        return finish("NO_ADMISSIBLE_EPISODE", "no FREE_CONFIRMED / FORCED_MEASURED interval long enough; UNKNOWN "
                      "and hypothesis-only intervals are excluded (spec G10), nothing converted to force-free",
                      excluded=excluded)
    names = [it.name for it in inter]
    order = np.argsort([it.t_obs[0] for it in inter])
    held = {names[order[-1]]} if (o["heldout"] == "last_if_multiple" and len(inter) > 1) else set()
    for n in names:
        regime[n]["train"] = n not in held
    # ------------------------------------------------------------------ URDF at the prior reference density
    mm = _mass_model(inp, o["mass_mode"])
    urdf = out / "model_prior_reference.urdf"
    info = reference_urdf(inp, rec, mm, urdf)
    if info is None:
        return finish("MASS_GEOMETRY_UNAVAILABLE", "revolute joint without a validated mass geometry / reference pose")
    _write(out / "urdf_info.json", info)
    hashes["files"]["urdf"] = dict(path=str(urdf), sha256=_sha(urdf))
    # ------------------------------------------------------------------ episodes + fit
    episodes = GA.episodes_from_g9_interactions(inter, regime, v0s, calibration_by_name=cal)
    GA.write_g9_physx_episodes(episodes, {k: v["evidence"] for k, v in regime.items()}, out / "physx_episodes.json")
    free_only = all(e.mode == "free" for e in episodes if e.used_for_training)
    Q_max = max([float(np.max(np.abs(e.input_effort))) for e in episodes] + [0.0])
    f = o["fitting"]
    fitting = dict(bounds=_bounds(o, info["axis_inertia"], Q_max, free_only), seed=f["seed"], maxiter=f["maxiter"],
                   popsize=f["popsize"], polish=f["polish"], tol=f["tol"],
                   v0_nuisance_bound_sd=float(f.get("v0_nuisance_bound_sd", 0.0)))
    # optional warm start from the analytical input-output identification (same windows; NOT a loss term)
    x0 = x0_prov = None
    if io_init:
        from ..system_identification.physx_bridge import initialize_physx_refinement
        io_path = Path(io_init)
        try:
            io_res = json.loads(io_path.read_text())
            ini = initialize_physx_refinement(io_res, dict(fitting=fitting, axis_inertia=info["axis_inertia"],
                                                           prior_rho_range=info["prior_rho_range"]), episodes)
        except Exception as e:                                   # reported; the G10 fit proceeds uninitialized
            ini = dict(status="FAILED", reason=f"{type(e).__name__}: {e}")
        ini["io_response"] = dict(path=str(io_path), sha256=_sha(io_path))
        _write(out / "io_initialization.json", ini)
        if ini.get("status") == "OK":
            x0, x0_prov = ini["x0"], f"{ini['x0_provenance']} [{io_path}]"
        log(f"[g10] io initialization: {ini.get('status')} {ini.get('reasons') or ini.get('reason') or ''}")
    ph = dict(o["physics"], joint_type=jt, moving_link=info["moving_link"], joint_name=info["joint_name"],
              gravity=list(map(float, inp.gravity.g_world)))
    sim = IsolatedSapienCpu(urdf, moving_link=ph["moving_link"], joint_name=ph["joint_name"], dt=ph["timestep"],
                            gravity=ph["gravity"], friction_mode=ph["friction_mode"], python=ph.get("python"))
    try:
        rep = estimate(sim, episodes, physics=ph, loss=o["loss"], fitting=fitting, profile=o["profile"],
                       prior_rho_range=info["prior_rho_range"], log=log, x0=x0, x0_provenance=x0_prov)
        from .objective import PhysicalParameters
        p = PhysicalParameters(**rep["params"])
        rolls = {e.name: sim(p, e) for e in episodes}
        fam_roll = {}
        for m in (rep.get("correlated_family") or {}).get("members", [])[:: max(1, len(rep["correlated_family"]
                                                                                      ["members"]) // 3)]:
            pm = PhysicalParameters(m["rho_scale"], p.armature, m["static_effort"], m["dynamic_effort"], m["viscous"])
            fam_roll[m["rho_scale"]] = {e.name: sim(pm, e).q.tolist() for e in episodes}
        n_roll, backend = sim.n_rollouts, sim.backend_process
    finally:
        sim.close()
    blockers = []
    if pulse_rows:
        from ..validation.impulse_balance import audit
        try:
            ia = audit(rec, pulse_rows, observability=(inp.record_context or {}).get("observability"),
                       robot_record=(inp.record_context or {})["rec"])
            dec = ia["decision"]["primary"]
        except Exception as e:                                  # the audit is reported, never silently skipped
            ia, dec = dict(error=f"{type(e).__name__}: {e}"), "AUDIT_FAILED"
        rep["impulse_audit"] = dict(rows=list(pulse_rows), decision=ia.get("decision"), primary=dec)
        if dec != "IMPULSE_CONSISTENT":
            blockers.append(f"declared pulse rows {pulse_rows[0]}-{pulse_rows[1]}: {dec} -- BLOCKER for any "
                            "absolute-inertia claim until independently resolved (no torque / displacement sign flip)")
    if not any(not e.used_for_training for e in episodes):
        blockers.append("NO_INDEPENDENT_EPISODE: no held-out / independent demonstration")
    if fa_status != "VERIFIED" and not free_only:
        blockers.append(f"force audit {fa_status}: exploratory forced input, not certified")
    if not inp.gravity.verified:
        blockers.append("gravity direction DECLARED, not verified")
    rep["blockers"] = blockers
    rep["provenance"] = dict(owner="ForceArtGS", source="G9 inputs -> free_forced segments -> G9Interaction",
                             urdf_sha256=_sha(urdf), engine_process=backend, n_rollouts=n_roll,
                             run_mode=ph["engine"], friction_semantics=ph["friction_mode"],
                             force_source="canonical robot record object-side Q (native rows, ZOH); no retare / "
                                          "sign change", git=git)
    rep["episodes_used"] = {e.name: dict(mode=e.mode, train=e.used_for_training, n=len(e.t_s),
                                         t=[float(e.t_s[0]), float(e.t_s[-1])], v0=e.v0, sigma_v0=e.sigma_v0,
                                         calibration=e.calibration_status) for e in episodes}
    rep["excluded_intervals"] = excluded
    _write(out / "physx_refinement.json", rep)
    try:
        _plots(out / "plots", episodes, rolls, fam_roll, rep, UB_units(jt))
    except Exception as e:                                       # plotting never hides the numbers
        log(f"[g10] plotting failed: {type(e).__name__}: {e}")
    (out / "summary.md").write_text(summary_md(Path(scene).name, jt, rep, info, excluded))
    st = rep["status_table"]
    return finish("OK", f"overall {st['overall']}; absolute scale {st['absolute_scale']}",
                  overall=st["overall"], absolute_scale=st["absolute_scale"], scale_status=rep["scale_status"],
                  n_train=sum(e.used_for_training for e in episodes),
                  n_heldout=sum(not e.used_for_training for e in episodes), n_excluded=len(excluded),
                  blockers=blockers, simulation_export_allowed=False)


def UB_units(jt):
    return dict(q="rad" if jt == "revolute" else "m", I="kg m^2" if jt == "revolute" else "kg",
                c="N m s/rad" if jt == "revolute" else "N s/m", T="N m" if jt == "revolute" else "N")


def _plots(d, episodes, rolls, fam_roll, rep, u):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d.mkdir(parents=True, exist_ok=True)
    n = len(episodes)
    fig, ax = plt.subplots(n, 1, figsize=(8, 3 * n), squeeze=False)
    for a, e in zip(ax[:, 0], episodes):
        a.errorbar(e.t_s, e.q, yerr=e.sigma_q, fmt=".", ms=3, color="k", label="observed q (G9 record)")
        a.plot(e.t_s, rolls[e.name].q, "C0-", label="PhysX rollout at the fit")
        for k, (rho, fr) in enumerate(fam_roll.items()):
            a.plot(e.t_s, fr[e.name], "--", lw=0.8, color=f"C{k + 1}", label=f"family rho_s={rho:.3g}")
        a.set_title(f"{e.name} ({e.mode}, {'train' if e.used_for_training else 'held-out'})", fontsize=9)
        a.set_ylabel(f"q [{u['q']}]")
        a.legend(fontsize=7)
    ax[-1, 0].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(d / "rollouts.png", dpi=120)
    plt.close(fig)
    pr = rep.get("scale_profile") or {}
    pts = [p for p in pr.get("points", []) if "summed_loss" in p]
    if pts:
        fig, a = plt.subplots(figsize=(6, 4))
        a.semilogx(pr["rho_evaluated"], pr["delta_L"], "o-")
        a.axhline(pr["threshold"], color="r", ls="--", label=f"95% threshold ({pr['threshold']:.3g})")
        a.set_xlabel("rho_scale (relative to the prior reference density)")
        a.set_ylabel("profile delta L")
        a.set_title(pr["status"], fontsize=9)
        a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(d / "scale_profile.png", dpi=120)
        plt.close(fig)


def _g(x, n=4):
    return "n/a" if x is None else (f"{x:.{n}g}" if isinstance(x, (int, float)) else str(x))


def summary_md(scene, jt, rep, info, excluded):
    u = UB_units(jt)
    st = rep["status_table"]
    L = [f"# G10 PhysX refinement -- {scene}", "",
         f"- overall: **{st['overall']}**; absolute scale: **{st['absolute_scale']}**; scale status "
         f"`{rep['scale_status']}`; runtime semantics `{st['runtime_semantics']}`",
         f"- simulation_export_allowed: **False**; downstream point estimate: "
         f"{rep['downstream']['point_estimate_allowed']} ({rep['downstream']['reason']})",
         f"- URDF: G9 {info.get('mass_mode', 'point-mass')} geometry at the PRIOR reference density "
         f"(I_ref {_g(info['I_ref'])} {u['I']}); rho_scale is relative to it", "",
         "## Episodes", "", "| name | mode | train | n | rmse | rmse/sigma | rmse/range | adequate |", "|---|---|---|---|---|---|---|---|"]
    for n_, a in rep["model_adequacy"]["episodes"].items():
        L.append(f"| {n_} | {a['mode']} | {a['train']} | {rep['episodes_used'][n_]['n']} | {_g(a['rmse'])} {u['q']} | "
                 f"{_g(a['rmse_over_sigma'])} | {_g(a['rmse_over_range'])} | {a['adequate']} |")
    L += ["", f"Excluded intervals: {len(excluded)}"] + [f"- `{x['name']}` {x['label']}: {x['reason']}" for x in excluded]
    L += ["", "## Parameters (estimate_not_for_sim)", "", "| parameter | estimate | status |", "|---|---|---|"]
    for k, v in st["parameters"].items():
        L.append(f"| {k} | {_g(v['estimate_not_for_sim'])} | {v['status']} |")
    pr = rep.get("scale_profile") or {}
    L += ["", f"## Scale profile: {pr.get('status')}", ""]
    if pr.get("ci95"):
        L.append(f"best rho_scale {_g(pr['rho_best'])}, 95% set [{_g(pr['ci95'][0])}, {_g(pr['ci95'][1])}] "
                 f"(open low {pr['open_low']}, "
                 f"open high {pr['open_high']}), chi2_red {_g(pr['chi2_red_at_best'])}")
    L += ["", "## Blockers", ""] + [f"- {b}" for b in rep.get("blockers", [])] + [
        "", "Every number above is an engine-conditional diagnostic (SAPIEN PhysX CPU, explicit friction surrogate). "
            "No native PhysX friction, no stiction, no identifiability claim."]
    return "\n".join(L) + "\n"
