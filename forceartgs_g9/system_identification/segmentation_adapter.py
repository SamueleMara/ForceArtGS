"""Admissible episodes from the EXISTING G9 free/forced classifier (``forceartgs_g9.free_forced.segments``) through the
EXISTING G10 bridge (``physx_refinement.g9_adapter.interactions_from_segments``) -- one canonical path, so the
analytical identification and the PhysX refinement see identical windows, initial states, effort rows and clock.

No new classification: FREE_CONFIRMED -> free, FORCED_MEASURED (+ admissible force audit) -> forced; every other
interval is excluded with a reason code (spec 3.3) and its evidence; a near-zero wrench in contact is never free."""
from __future__ import annotations

import numpy as np

from .contracts import IOEpisode

REASON_CODE = {
    "LOW_MEASURED_EFFORT_IN_ANNOTATED_CONTACT": "NEAR_ZERO_FORCE_IN_CONTACT",
    "STICK_INTERVAL": "STICK_INTERVAL",
    "EFFORT_WITHOUT_ANNOTATED_CONTACT": "CONTACT_UNKNOWN",
    "AMBIGUOUS_CONTACT": "CONTACT_UNKNOWN",
    "ABSENT_WRENCH": "CONTACT_UNKNOWN",
}


def _code(x):
    lab, why = x.get("label", ""), x.get("reason", "")
    if lab.startswith("FREE_HYPOTHESIS"):
        return "HYPOTHESIS_ONLY"
    if "not admissible" in why or "force rows start" in why:
        return "INPUT_NOT_ADMISSIBLE"
    if "q samples: fewer than" in why:
        return "TOO_SHORT"
    return "CONTACT_UNKNOWN"


def from_segments(segments, *, force_audit_status, v0_prefix_samples=4, min_samples=4, heldout="last_if_multiple",
                  ev=None):
    """-> (episodes [IOEpisode], g10_episodes, excluded [dict], regime). ``ev``: the classifier output, whose
    UNKNOWN moving intervals are added to the exclusion log (they never became segments)."""
    from ..physx_refinement import g9_adapter as GA
    inter, regime, v0s, cal, excl = GA.interactions_from_segments(
        segments, force_audit_status=force_audit_status, v0_prefix_samples=v0_prefix_samples, min_samples=min_samples)
    excluded = [dict(x, code=_code(x)) for x in excl]
    by_name = {s.name: s for s in segments}
    hyp_rows = set()
    for x in excluded:
        s = by_name.get(x["name"])
        if s is not None and s.evidence.get("interval_rows") is not None:
            x["rows"] = list(s.evidence["interval_rows"])
            x["underlying_reason"] = s.evidence.get("reason")
            x["underlying_code"] = REASON_CODE.get(s.evidence.get("reason"), "CONTACT_UNKNOWN") \
                if x["code"] == "HYPOTHESIS_ONLY" else None
            if x["code"] == "HYPOTHESIS_ONLY":
                hyp_rows.add(tuple(s.evidence["interval_rows"]))
    if ev is not None:
        for iv in ev.get("intervals", []):
            if tuple(iv["rows"]) in hyp_rows:
                continue                                   # already listed as the hypothesis-only segment
            if iv["label"] == "UNKNOWN_OR_UNMEASURED" and (iv.get("moving_fraction") or 0) > 0 and iv.get("n_q", 0):
                excluded.append(dict(name=f"interval_rows_{iv['rows'][0]}-{iv['rows'][1]}", label=iv["label"],
                                     reason=iv["reason"], code=REASON_CODE.get(iv["reason"], "CONTACT_UNKNOWN"),
                                     rows=iv["rows"], t=iv["t"], n_q=iv["n_q"],
                                     moving_fraction=iv["moving_fraction"], effort=iv.get("effort"),
                                     note="moving interval not admitted (never relabelled free)"))
    names = [it.name for it in inter]
    order = np.argsort([it.t_obs[0] for it in inter]) if inter else []
    held = {names[order[-1]]} if (heldout == "last_if_multiple" and len(inter) > 1) else set()
    for n in names:
        regime[n]["train"] = n not in held
    g10 = GA.episodes_from_g9_interactions(inter, regime, v0s, calibration_by_name=cal) if inter else []
    seg_by = {s.name: s for s in segments}
    eps = []
    for it, ge in zip(inter, g10):
        s = seg_by[it.name]
        ist = it.initial_state
        mode = regime[it.name]["mode"]
        ev_ = dict(regime[it.name]["evidence"], initial_state_source=ist.get("source"))
        eff_t = eff = eff_s = None
        if mode == "forced":
            eff_t, eff = np.asarray(it.effort.t_s, float), np.asarray(it.effort.levels, float)
            eff_s = None if s.Q_sigma is None else np.asarray(s.Q_sigma, float)
        eps.append(IOEpisode(name=it.name, mode=mode, joint_type=it.joint_type, t=it.t_obs, q=it.q_obs,
                             q_sigma=it.q_std, t0=float(ist["t_s"]), q0=float(ist["q"]), v0=float(ist["qdot"]),
                             sigma_q0=max(float(ist.get("q_sd") or 0.0), 1e-9),
                             sigma_v0=max(float(ist.get("qdot_sd") or 0.0), 1e-9),
                             sign=float(s.sign), effort_t=eff_t, effort=eff, effort_sigma=eff_s,
                             train=regime[it.name]["train"],
                             calibration=(cal.get(it.name) or {}).get("status", "NOT_APPLICABLE") if mode == "forced"
                             else "NOT_APPLICABLE", evidence=ev_, segment=s, g10=ge))
    return eps, g10, excluded, regime


def prepare_run(run_root, scene, *, ff_config, cfg_path=None, robot_archive=None, contact_segments=None,
                declared_joint_type=None, declaration_note=None, allow_diagnostic_kinematics=True, log=print):
    """G9 inputs -> canonical record -> SAME classifier + options as the free/forced stage. Returns a context dict;
    ``status`` is OK, KINEMATICS_NOT_ACCEPTED or DATA_UNAVAILABLE (nothing fabricated)."""
    from ..adapters.forceartgs_g9 import build_inputs
    from ..config import load_dynamics_config
    from ..forcetwin import record as RC
    from ..free_forced import segments as SG
    from ..free_forced.pipeline import _part_points, load_options as ff_options
    ffo, cfg_ff = ff_options(ff_config)
    cfg_path = cfg_path or cfg_ff
    cfg = load_dynamics_config(cfg_path)
    cfg["pipeline"]["allow_diagnostic_kinematics"] = bool(allow_diagnostic_kinematics)
    inp = build_inputs(run_root, scene, cfg, robot_archive=robot_archive, contact_segments=contact_segments,
                       declared_joint_type=declared_joint_type, declaration_note=declaration_note, log=log)
    ctx = dict(inp=inp, ffo=ffo, cfg_path=cfg_path, cfg=cfg)
    try:
        rec = RC.from_g9_inputs(inp, contact_segments=contact_segments)
    except RC.RecordError as e:
        fa = inp.force_audit or {}
        if fa.get("status") == "NOT_EVALUATED" and inp.kinematic_status not in ("IDENTIFIED", "OPERATOR_DECLARED"):
            ctx.update(status="KINEMATICS_NOT_ACCEPTED", reason=f"upstream joint status {inp.kinematic_status}: no "
                       "joint authority -- nothing identified (nothing fabricated)")
        else:
            ctx.update(status="DATA_UNAVAILABLE", reason=str(e))
        return ctx
    so = ffo["segments"]
    q_ref = float((inp.reference_pose or {}).get("q_ref", 0.0) or 0.0)
    ev = SG.classify(rec, z_sig=so["z_sig"], win_rows=so["win_rows"], v_k=so["v_k"], sep_margin_m=so["sep_margin_m"],
                     part_points_ref=_part_points(inp, int(so["n_part_points"])), q_ref=q_ref,
                     min_run=so["min_run"], pulse_z=so["pulse_z"], z_J=so["z_J"])
    segs = SG.build_segments(rec, ev, free_hypotheses=ffo["free_hypotheses"], min_q=so["min_q"], v_k=so["v_k"])
    ctx.update(status="OK", rec=rec, ev=ev, segments=segs, q_ref=q_ref,
               force_audit_status=(inp.force_audit or {}).get("status"))
    return ctx
