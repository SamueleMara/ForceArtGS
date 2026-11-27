"""G9 generalized free + forced dynamics on an existing real run (append-only <run>/dynamics/g9_free_forced/vNNN/).

    resolved_config.yaml, source_hashes.json, segment_evidence.json, normalized_model_spec.json,
    free_weak_fit.json, free_pinn_fit.json, free_ekf_fit.json, free_rollout_fit.json, forced_scale_fit.json,
    identifiability_report.json, model_adequacy.json, geometry_gravity_hypotheses.json,
    simulator_handoff_preview.json, plots/*.png, comparison_summary.md, comparison_rows.csv, STATUS.json

Inputs are read-only (the canonical robot record, the accepted joint authority, the RGB-D tracks); nothing is
re-tared, re-signed, re-calibrated or re-trained, and the accepted axis is never silently replaced (the vertical-hinge
refit is a reported ALTERNATIVE hypothesis).
"""
from __future__ import annotations

import copy
import csv
import hashlib
import math
import time
from pathlib import Path

import numpy as np

from ..estimators import free_weak_form as FW
from ..forcetwin import record as RC
from ..forcetwin.pipeline import _clean, _git, _mass_model, _prior_for_sim, _vertical_record, _write, next_version
from . import replay as RP
from . import segments as SG
from . import status as ST
from .normalized import GravityModel, NormalizedDynamicsSpec, UNITS

DEFAULTS = dict(
    segments=dict(z_sig=3.0, win_rows=5, v_k=3.0, sep_margin_m=0.03, n_part_points=6000, min_q=5, min_run=10,
                  pulse_z=10.0, z_J=5.0),
    free_hypotheses=["measured_near_zero_effort_in_contact"],
    sigma_floor=dict(revolute=1e-3, prismatic=1e-3),
    replay=dict(prefix_frac=0.3, heldout_frac=0.6),
    bootstrap=200, negligible_factor=0.2,
    pinn=dict(epochs=2500, epochs_warm=600, seeds=[0, 1, 2], lbfgs_iters=300, phase_b_epochs=1500),
)


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_options(path):
    if not path:
        return copy.deepcopy(DEFAULTS), None
    import yaml
    doc = yaml.safe_load(Path(path).read_text()) or {}
    return _merge(DEFAULTS, doc.get("free_forced") or {}), doc.get("dynamics_config")


def _sha(p):
    if not p or not Path(p).is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _part_points(inp, n, seed=0):
    if inp.mass_mesh is None:
        return None
    from ..synthetic import _surface_samples
    return _surface_samples(inp.mass_mesh, n, np.random.default_rng(seed))


# ================================================================================================ per-method status
def method_entry(name, res, rp, weak, ns, *, caps, nis=None):
    est = (res or {}).get("estimate") or {}
    ok = str((res or {}).get("status")).lower() == "ok"
    agg = (rp or {}).get("aggregate") or {}
    structural = (weak or {}).get("per_parameter") or {}
    out = {}
    for n in ns.names:
        s0 = structural.get(n, {}).get("status", (weak or {}).get("status", "FREE_UNINFORMATIVE"))
        st, why = s0, []
        if not ok:
            st, why = "NOT_IDENTIFIED", [f"estimator status {(res or {}).get('status')}"]
        elif agg.get("adequate") is False:
            st = "FREE_MODEL_INADEQUATE"
            why.append(f"free replay rmse/range {agg.get('rmse_over_range'):.3g} fails the adequacy rule")
        if nis is not None and nis.get("consistent") is False and st == "RATIO_IDENTIFIED":
            st = "MODEL_INADEQUATE"
            why.append(f"EKF mean NIS {nis.get('mean'):.3g} outside [0.5, 2]")
        if "ENSEMBLE" in str((res or {}).get("uncertainty_label", "")).upper():
            why.append("seed spread only (not a posterior): status inherited from the SHARED weak-form "
                       "identifiability, never from the PINN's own convergence")
        ci = ((weak or {}).get("profile") or {}).get(n, {}).get("ci95")
        e = est.get(n)
        agrees = None if (e is None or not ci or ci[0] is None) else bool(ci[0] - 1e-12 <= e <= ci[1] + 1e-12)
        out[n] = dict(estimate=e, sd=((res or {}).get("sd") or {}).get(n), unit=ns.units[n], status=st, caps=caps,
                      reasons=why, structural_status=s0, agrees_with_weak_profile_ci95=agrees)
    return dict(method=name, status=(res or {}).get("status"), parameters=out, replay=agg or None,
                uncertainty_label=(res or {}).get("uncertainty_label"))


def _theta(ns, res):
    e = (res or {}).get("estimate") or {}
    if not e:
        return None
    return ns.pack(e.get("lam", 0.0), e.get("gam", 0.0), e.get("beta") if ns.forced else None)


def _heldout(fit_fn, segs, ns, opts, floor):
    heads, tails = [], []
    for s in segs:
        h, t = s.split(opts["replay"]["heldout_frac"])
        heads.append(h)
        if t is not None:
            tails.append(t)
    if not tails:
        return dict(status="NOT_EVALUATED", reason="segments too short to split")
    r = fit_fn(heads)
    th = _theta(ns, r)
    if th is None:
        return dict(status="FIT_FAILED", fit=r)
    rp = RP.replay(tails, ns, th, prefix_frac=opts["replay"]["prefix_frac"], sigma_floor=floor)
    return dict(status="OK", fit_on_heads=dict(estimate=r.get("estimate"), status=r.get("status")),
                heldout_replay=rp, split=f"first {opts['replay']['heldout_frac']:.0%} of each segment -> fit; "
                                         "remainder -> replay (state from ITS prefix only)")


def passive(segs, ns, min_q):
    kept, aud = [], []
    for s in segs:
        k, a = FW.passivity_split(s, ns, min_q=min_q)
        aud.append(a)
        if k is not None:
            kept.append(k)
    return kept, aud


def _passivity_failure(raw, kept, aud):
    """All free data removed by an energy gain: the free model under THIS gravity hypothesis is contradicted
    (FREE_MODEL_INADEQUATE), which is different from 'too little data' (FREE_UNINFORMATIVE)."""
    if raw and not kept and any(a.get("verdict") == "ENERGY_GAIN_WITHOUT_MEASURED_INPUT" for a in aud):
        g = max(aud, key=lambda a: a.get("gain_over_sd") or 0)
        return dict(status="FREE_MODEL_INADEQUATE", names=["lam", "gam"],
                    reasons=[f"passivity violated under gravity '{g['gravity']}': normalized energy rises by "
                             f"{g['energy_gain']:.3g} {g['unit']} ({g['gain_over_sd']:.0f} sd) with no measured input "
                             "-- this gravity hypothesis (or the free assumption) is inconsistent with the motion"],
                    per_parameter={n: dict(status="NOT_IDENTIFIED", reason="free model contradicted")
                                   for n in ("lam", "gam")}, passivity=aud)
    return None


# ================================================================================================ run
def run(run_root, scene, *, cfg_path, ff_config=None, robot_archive=None, contact_segments=None, discovery_dir=None,
        pulse_rows=None, pinn=False, out_root=None, declared_joint_type=None, declaration_note=None,
        log=print) -> dict:
    from ..adapters.forceartgs_g9 import build_inputs
    from ..config import load_dynamics_config
    t0 = time.time()
    opts, cfg_from_opts = load_options(ff_config)
    cfg_path = cfg_path or cfg_from_opts
    out = next_version(Path(out_root) if out_root else Path(run_root) / "dynamics" / "g9_free_forced")
    (out / "plots").mkdir(exist_ok=True)
    cfg = load_dynamics_config(cfg_path)
    cfg["pipeline"]["allow_diagnostic_kinematics"] = True
    git = _git()
    status = dict(schema="forceartgs.g9.free_forced_status/v1", scene=Path(scene).name, out_dir=str(out), git=git,
                  flags=dict(pinn=bool(pinn), free_hypotheses=opts["free_hypotheses"]))
    inp = build_inputs(run_root, scene, cfg, robot_archive=robot_archive, contact_segments=contact_segments,
                       declared_joint_type=declared_joint_type, declaration_note=declaration_note, log=log)
    import yaml
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(_clean(dict(
        dynamics_config=str(cfg_path), free_forced_config=str(ff_config) if ff_config else None, free_forced=opts,
        run=str(run_root), scene=str(scene), robot_archive=robot_archive, contact_segments=contact_segments,
        discovery_dir=discovery_dir, pulse_rows=list(pulse_rows) if pulse_rows else None, pinn=bool(pinn),
        declared_joint_type=declared_joint_type, declaration_note=declaration_note,
        gravity=None if inp.gravity is None else dict(g_B=list(inp.gravity.g_world), verified=inp.gravity.verified,
                                                      provenance=inp.gravity.provenance),
        reference_pose=inp.reference_pose, kinematic_status=inp.kinematic_status,
        force_convention="object_side_multiplier from the canonical robot record (applied exactly once upstream)",
        git=git)), sort_keys=False))
    kp = inp.kinematics_provenance or {}
    hashes = dict(input_hashes=inp.input_hashes, files={k: dict(path=str(v), sha256=_sha(v)) for k, v in dict(
        dynamics_config=cfg_path, free_forced_config=ff_config, robot_archive=robot_archive,
        contact_segments=contact_segments, joint_authority=kp.get("authority_path") or kp.get("path")).items() if v},
        tool_versions=_versions())
    # ------------------------------------------------------------------ record (or readiness audit)
    try:
        rec = RC.from_g9_inputs(inp, contact_segments=contact_segments)
    except RC.RecordError as e:
        fa = inp.force_audit or {}
        if fa.get("status") == "NOT_EVALUATED" and inp.kinematic_status not in ("IDENTIFIED", "OPERATOR_DECLARED"):
            status.update(status="KINEMATICS_NOT_ACCEPTED", reason=f"upstream joint status {inp.kinematic_status} "
                          f"({kp.get('selection_status')}): no joint to project the force on; no q(t) authority -- "
                          "free dynamics NOT run (nothing fabricated)")
        else:
            status.update(status="DATA_UNAVAILABLE", reason=str(e))
        _write(out / "segment_evidence.json", dict(status=status["status"], reason=status["reason"],
                                                   readiness_audit=dict(kinematic_status=inp.kinematic_status,
                                                                        force_audit=fa.get("status"),
                                                                        n_windows=len(inp.train or []))))
        _write(out / "source_hashes.json", hashes)
        _write(out / "STATUS.json", status)
        log(f"[free_forced] {Path(scene).name}: {status['status']} ({status['reason']})")
        return status
    hashes["record_sha256"] = rec.sha256()
    _write(out / "source_hashes.json", hashes)
    jt = rec.joint_type
    u = UNITS[jt]
    floor = float(opts["sigma_floor"][jt])
    kin_auth = ST.kinematic_authority(inp.kinematic_status)
    # ------------------------------------------------------------------ joint-geometry + gravity hypotheses
    kin = None
    if jt == "revolute" and discovery_dir and (Path(discovery_dir) / "tracks_observations.npz").is_file():
        from ..forcetwin.kinematics_hypotheses import hypotheses
        kin = hypotheses(discovery_dir, inp.axis, inp.origin)
    recs = {"accepted_axis": rec}
    if kin and kin.get("status") != "NOT_RUN":
        try:
            recs["vertical_hinge_refit"] = _vertical_record(inp, kin, contact_segments)
        except RC.RecordError as e:
            kin["vertical_record_error"] = str(e)
    mm_solid, mm_shell = _mass_model(inp, "uniform_solid"), _mass_model(inp, "uniform_shell")
    prior = _prior_for_sim(inp, mm_solid)
    g = None if inp.gravity is None else inp.gravity.g_world
    ghyp = {}
    if g is not None:
        if jt == "revolute":
            for nm, mm in (("known_shape_solid_hull", mm_solid), ("known_shape_shell_hull", mm_shell)):
                if mm is not None:
                    ghyp[nm] = GravityModel.from_mass_model(mm, g, f"gravity_geometry_{nm}")
        else:
            ghyp["prismatic_ideal"] = GravityModel.prismatic_ideal(rec.axis_B, g)
    ghyp["negligible_test"] = GravityModel.zero(jt, "gravity_negligible_test", "negligible",
                                                "a_g := 0 (valid only if the negligible test supports it)")
    primary_g = next(iter(k for k in ghyp if k != "negligible_test"), "negligible_test")
    if primary_g == "negligible_test":
        ghyp["negligible_test"].provenance += "; NO gravity / mass geometry available -> gravity_unknown"
    # ------------------------------------------------------------------ segment evidence
    so = opts["segments"]
    q_ref = float((inp.reference_pose or {}).get("q_ref", 0.0) or 0.0)
    pts = _part_points(inp, int(so["n_part_points"]))
    seg_ev, seg_sets = {}, {}
    for rn, r in recs.items():
        ev = SG.classify(r, z_sig=so["z_sig"], win_rows=so["win_rows"], v_k=so["v_k"], sep_margin_m=so["sep_margin_m"],
                         part_points_ref=pts if rn == "accepted_axis" else None, q_ref=q_ref,
                         min_run=so["min_run"], pulse_z=so["pulse_z"], z_J=so["z_J"])
        segs = SG.build_segments(r, ev, free_hypotheses=opts["free_hypotheses"], min_q=so["min_q"], v_k=so["v_k"])
        conf = [s for s in segs if s.label == "FREE_CONFIRMED"]
        hyp = [s for s in segs if s.label.startswith("FREE_HYPOTHESIS")]
        forced = [s for s in segs if s.label == "FORCED_MEASURED"]
        seg_sets[rn] = dict(free=conf or hyp, hypothesis_only=(not conf) and bool(hyp), forced=forced, all=segs)
        ev = dict(ev, segments=[s.describe() for s in segs],
                  used_as_free=[s.name for s in (conf or hyp)], used_as_forced=[s.name for s in forced],
                  free_basis=("FREE_CONFIRMED" if conf else ("FREE_HYPOTHESIS (explicitly named; results capped "
                                                             "CONDITIONAL_ON_FREE_HYPOTHESIS)" if hyp else "NONE")))
        seg_ev[rn] = ev
    _write(out / "segment_evidence.json", dict(schema="forceartgs.g9.segment_evidence_set/v1", per_geometry=seg_ev))
    P = seg_sets["accepted_axis"]
    free, forced, hyp_only = P["free"], P["forced"], P["hypothesis_only"]
    caps = (["CONDITIONAL_ON_FREE_HYPOTHESIS"] if hyp_only else []) + \
           ([f"CONDITIONAL_ON_{kin_auth}_KINEMATICS"] if kin_auth != "ACCEPTED" else [])
    ns = NormalizedDynamicsSpec(jt, ghyp[primary_g], provenance=dict(gravity_hypothesis=primary_g,
                                                                    geometry="accepted_axis"))
    _write(out / "normalized_model_spec.json", dict(ns.describe(), primary_gravity_hypothesis=primary_g,
                                                    forced_stage=ns.with_forced().describe()))
    # ------------------------------------------------------------------ every geometry / gravity hypothesis:
    #                                                                      passivity audit + weak fit + replay
    B = int(opts["bootstrap"])
    hyps = {}
    for gname, gm in ghyp.items():
        hyps[f"accepted_axis/{gname}"] = dict(geometry="accepted_axis", ns=ns.with_gravity(gm), raw=P["free"],
                                              forced=P["forced"], rec=rec, hyp_only=P["hypothesis_only"])
    if "vertical_hinge_refit" in recs:
        V = seg_sets["vertical_hinge_refit"]
        hyps["vertical_hinge_refit/vertical_exact"] = dict(
            geometry="vertical_hinge_refit", rec=recs["vertical_hinge_refit"], raw=V["free"], forced=V["forced"],
            hyp_only=V["hypothesis_only"],
            ns=NormalizedDynamicsSpec(jt, GravityModel.zero(jt, "vertical_exact", "vertical_exact",
                                                            "exactly vertical hinge: gravity torque identically 0"),
                                      provenance=dict(geometry="vertical_hinge_refit", gravity_hypothesis="vertical_exact")))
    primary_key = f"accepted_axis/{primary_g}"
    for k, h in hyps.items():
        h["free"], h["passivity"] = passive(h["raw"], h["ns"], so["min_q"])
        h["weak"] = (FW.fit(h["free"], h["ns"], B=B if k == primary_key else 50, sensitivity=(k == primary_key))
                     if h["free"] else (_passivity_failure(h["raw"], h["free"], h["passivity"]) or
                                        dict(status="FREE_UNINFORMATIVE", reasons=["no free segment"],
                                             names=h["ns"].names)))
        th = _theta(h["ns"], h["weak"])
        h["replay"] = RP.replay(h["free"], h["ns"], th, prefix_frac=opts["replay"]["prefix_frac"],
                                sigma_floor=floor, cov=h["weak"].get("cov")) if th is not None else None
    seg_ev["accepted_axis"]["passivity_audit"] = hyps[primary_key]["passivity"]
    _write(out / "segment_evidence.json", dict(schema="forceartgs.g9.segment_evidence_set/v1", per_geometry=seg_ev))
    # analysis hypothesis: the primary if its free data survive the passivity audit, else the passive alternative
    # with the lowest integral-equation chi2/dof (data-driven, recorded; the accepted axis is NOT replaced)
    # gravity_negligible_test is admissible only when the bound supports it against every known shape
    neg_ok, neg_tests = True, {}
    zk = "accepted_axis/negligible_test"
    if zk in hyps and hyps[zk]["free"]:
        for k, gm in ghyp.items():
            if gm.kind in ("known_shape", "prismatic_ideal"):
                neg_tests[k] = FW.gravity_negligible_test(hyps[zk]["free"], ns.with_gravity(gm),
                                                          opts["negligible_factor"])
                neg_ok &= neg_tests[k]["status"] == "NEGLIGIBLE_SUPPORTED"
    cand = [k for k, h in hyps.items() if h["free"] and "estimate" in h["weak"] and (k != zk or neg_ok)]

    def _rank(k):
        ag = (hyps[k]["replay"] or {}).get("aggregate") or {}
        return (not ag.get("adequate", False), not ag.get("beats_null", False),
                hyps[k]["weak"].get("chi2_per_dof", math.inf))
    if hyps[primary_key]["free"]:
        akey, why_a = primary_key, "primary hypothesis (free data pass the passivity audit)"
    elif cand:
        akey = min(cand, key=_rank)
        why_a = (f"primary {primary_key} is {hyps[primary_key]['weak'].get('status')}; analysis on the admissible "
                 "passive alternative ranked by (free replay adequate, beats the null, lowest chi2/dof)"
                 + ("" if neg_ok else "; negligible_test EXCLUDED (the known-shape gravity is NOT negligible)"))
    else:
        akey, why_a = primary_key, "no hypothesis has usable free data"
    H = hyps[akey]
    ns_a, free, forced, hyp_only, rec_a = H["ns"], H["free"], H["forced"], H["hyp_only"], H["rec"]
    caps = (["CONDITIONAL_ON_FREE_HYPOTHESIS"] if hyp_only else []) + \
           ([f"CONDITIONAL_ON_{kin_auth}_KINEMATICS"] if kin_auth != "ACCEPTED" else []) + \
           ([f"CONDITIONAL_ON_ALTERNATIVE_HYPOTHESIS:{akey}"] if akey != primary_key else [])
    if akey != primary_key and H["free"]:
        H["weak"] = FW.fit(H["free"], ns_a, B=B)                 # full uncertainty for the analysis hypothesis
        th = _theta(ns_a, H["weak"])
        H["replay"] = RP.replay(H["free"], ns_a, th, prefix_frac=opts["replay"]["prefix_frac"], sigma_floor=floor,
                                cov=H["weak"].get("cov")) if th is not None else None
    log(f"[free_forced] analysis hypothesis {akey}: {len(free)} free segment(s) "
        f"({'hypothesis' if hyp_only else 'confirmed'}), {len(forced)} forced")
    # ------------------------------------------------------------------ free fits (same segments, same model)
    weak, rp_w = H["weak"], H["replay"]
    ho_w = _heldout(lambda s_: FW.fit(s_, ns_a, B=0, sensitivity=False), free, ns_a, opts, floor) if free else None
    weak.update(replay=rp_w, heldout=ho_w, caps=caps, analysis_hypothesis=akey, analysis_rule=why_a)

    def _norm(w, rp, ho, hyp_o, extra_caps=()):
        nm = ST.normalized_status(w, (rp or {}).get("aggregate", {}).get("adequate"),
                                  (rp or {}).get("aggregate", {}).get("beats_null"), hypothesis_only=hyp_o,
                                  kin_auth=kin_auth)
        nm["caps"] += list(extra_caps)
        if ho and ho.get("status") == "OK":
            ha = ho["heldout_replay"].get("aggregate") or {}
            if ha.get("adequate") is False and nm["status"] in ("FREE_RATIOS_IDENTIFIED", "FREE_COMBINATION_IDENTIFIED"):
                nm["status"] = "FREE_MODEL_INADEQUATE"
                nm["reasons"].append("held-out (rates fitted on segment heads) replay fails the adequacy rule")
        return nm
    norm = _norm(weak, rp_w, ho_w, hyp_only, [c for c in caps if c.startswith("CONDITIONAL_ON_ALTERNATIVE")])
    norm_primary = norm if akey == primary_key else _norm(hyps[primary_key]["weak"], hyps[primary_key]["replay"],
                                                          None, hyps[primary_key]["hyp_only"])
    weak["normalized_status"] = norm
    _write(out / "free_weak_fit.json", dict(weak, primary_hypothesis=primary_key,
                                            primary_hypothesis_fit=None if akey == primary_key else
                                            hyps[primary_key]["weak"], primary_normalized_status=norm_primary))
    from ..estimators import normalized_ekf as NE
    from ..estimators import normalized_rollout as NR
    methods = {"weak_form": method_entry("weak_form", dict(weak, status="ok" if "estimate" in weak else "none",
                                                           uncertainty_label="profile likelihood + covariance + "
                                                                             "window bootstrap"),
                                         rp_w, weak, ns_a, caps=caps)}
    results = {}
    if free:
        log("[free_forced] rollout ...")
        ro = NR.fit(free, ns_a, sigma_floor=floor)
        th = _theta(ns_a, ro)
        ro["replay"] = RP.replay(free, ns_a, th, prefix_frac=opts["replay"]["prefix_frac"], sigma_floor=floor,
                                 cov=ro.get("cov")) if th is not None else None
        ro["heldout"] = _heldout(lambda s_: NR.fit(s_, ns_a, sigma_floor=floor), free, ns_a, opts, floor)
        results["rollout"] = ro
        log("[free_forced] augmented EKF ...")
        ek = NE.fit(free, ns_a)
        th = _theta(ns_a, ek)
        ek["replay"] = RP.replay(free, ns_a, th, prefix_frac=opts["replay"]["prefix_frac"], sigma_floor=floor,
                                 cov=ek.get("cov")) if th is not None else None
        ek["heldout"] = _heldout(lambda s_: NE.fit(s_, ns_a), free, ns_a, opts, floor)
        results["ekf"] = ek
        if pinn:
            from ..estimators import normalized_pinn as NP
            pc = NP.NPINNConfig.from_dict(dict(opts["pinn"], sigma_floor=floor))
            log("[free_forced] normalized PINN phase A ...")
            pn = NP.fit_free(free, ns_a, pc, init=weak.get("estimate"), log=log)
            th = _theta(ns_a, pn)
            pn["replay"] = RP.replay(free, ns_a, th, prefix_frac=opts["replay"]["prefix_frac"], sigma_floor=floor) \
                if th is not None else None
            pc1 = NP.NPINNConfig.from_dict(dict(opts["pinn"], sigma_floor=floor, seeds=[pn.get("selected_seed", 0)]))
            pn["heldout"] = _heldout(lambda s_: NP.fit_free(s_, ns_a, pc1, init=weak.get("estimate")), free, ns_a,
                                     opts, floor)
            results["pinn"] = pn
        else:
            results["pinn"] = dict(status="NOT_RUN", reason="--pinn not given (run the printed command)")
    for nm, fn in (("rollout", "free_rollout_fit.json"), ("ekf", "free_ekf_fit.json"), ("pinn", "free_pinn_fit.json")):
        r = results.get(nm) or dict(status="NOT_RUN", reason="no free segment")
        _write(out / fn, dict(r, analysis_hypothesis=akey))
        if r.get("estimate"):
            methods[nm] = method_entry(nm, r, r.get("replay"), weak, ns_a, caps=caps,
                                       nis=r.get("nis") if nm == "ekf" else None)
    # ------------------------------------------------------------------ hypotheses report
    gh = dict(schema="forceartgs.g9.geometry_gravity_hypotheses/v1", primary=dict(geometry="accepted_axis",
                                                                                 gravity=primary_g, key=primary_key),
              analysis_hypothesis=dict(key=akey, rule=why_a),
              rule="the accepted axis stays primary (never silently replaced); the vertical-hinge refit and every "
                   "gravity hypothesis are reported alternatives with their own passivity audit, weak fit and replay",
              gravity={k: v.describe() for k, v in ghyp.items()},
              gravity_unknown=dict(label="gravity_unknown", note="no fit: a free trajectory cannot separate friction "
                                   "from an unknown configuration load -- see effective_diagnostic rank / correlation"),
              kinematics=kin or dict(status="NOT_RUN"), fits={})
    for k, h in hyps.items():
        w = h["weak"]
        gh["fits"][k] = dict(status=w.get("status", "NO_FREE_DATA"), estimate=w.get("estimate"),
                             per_parameter=w.get("per_parameter"), chi2_per_dof=w.get("chi2_per_dof"),
                             combination=w.get("combination"), replay=(h["replay"] or {}).get("aggregate"),
                             passivity=h["passivity"], n_free_segments=len(h["free"]), hypothesis_only=h["hyp_only"],
                             reasons=w.get("reasons"))
    zfree = hyps[zk]["free"] if zk in hyps else []
    gh["negligible_test"] = neg_tests
    if zfree:
        gh["effective_diagnostic"] = FW.effective_load_fit(zfree, hyps[zk]["ns"], jt)
        # how much of the hull-tied gravity does the free motion tolerate?  scale a_g by f in [0, 1]
        for k, gm in ghyp.items():
            if gm.kind != "known_shape" or jt != "revolute":
                continue
            scan = []
            for f in np.linspace(0.0, 1.0, 21):
                gf = GravityModel(f"{gm.name}_x{f:.2f}", "scaled", jt, f * gm.A, f * gm.B)
                verdicts = [FW.passivity_split(x, ns.with_gravity(gf), min_q=so["min_q"])[1]["verdict"]
                            for x in hyps[zk]["raw"]]
                scan.append(dict(f=float(f), passive=all(v == "PASSIVE_CONSISTENT" for v in verdicts)))
            fmax = max([x["f"] for x in scan if x["passive"]], default=None)
            a_ = np.asarray(inp.axis, float) / np.linalg.norm(inp.axis)
            tilt = float(np.degrees(np.arccos(min(1.0, abs(a_[2])))))
            gh.setdefault("gravity_scale_passivity_scan", {})[k] = dict(
                scan=scan, max_passive_fraction=fmax, accepted_tilt_deg=tilt,
                equivalent_max_tilt_deg=None if fmax is None else
                float(np.degrees(np.arcsin(min(1.0, fmax * math.sin(math.radians(tilt)))))),
                note="a_g scaled by f (the hull COM / hinge-tilt gravity at fraction f); the largest f whose free "
                     "data stay passive. f ~ sin(tilt_eff)/sin(tilt_accepted) for a fixed COM -- a DIAGNOSTIC, not "
                     "an axis estimate")
    _write(out / "geometry_gravity_hypotheses.json", gh)
    free_raw = H["raw"]
    ns = ns_a
    # ------------------------------------------------------------------ forced scale (+ declared pulse audit)
    imp_dec, imp_aud = None, None
    if pulse_rows:
        from ..validation.impulse_balance import audit
        imp_aud = audit(rec, pulse_rows, kin_hyp=kin, observability=(inp.record_context or {}).get("observability"),
                        robot_record=(inp.record_context or {})["rec"])
        imp_dec = imp_aud["decision"]["primary"]
    imp_windows = []
    for iv in seg_ev[H["geometry"]]["intervals"]:
        if iv["label"] == "FORCED_MEASURED" and iv["n_q"] < so["min_q"]:
            w = FW.impulse_window(rec_a, ns, iv["rows"])
            if w is not None:
                imp_windows.append(w)
    if pulse_rows and not any(w["seg"] == f"pulse_{pulse_rows[0]}-{pulse_rows[1]}" for w in imp_windows):
        w = FW.impulse_window(rec_a, ns, pulse_rows)
        if w is not None:
            imp_windows.append(w)
    fsc = FW.forced_scale(free, forced, ns, free_fit=weak if "estimate" in weak else None,
                          impulse_windows=imp_windows)
    if fsc.get("estimate") and forced:
        fsn = ns.with_forced()
        thf = _theta(fsn, fsc)
        fsc["replay"] = RP.replay(forced, fsn, thf, prefix_frac=opts["replay"]["prefix_frac"], sigma_floor=floor)
        fsc["replay_ok"] = bool((fsc["replay"].get("aggregate") or {}).get("adequate"))
        log("[free_forced] forced rollout / EKF ...")
        fsc["rollout"] = NR.fit_staged(free, forced, ns, sigma_floor=floor).get("forced")
        fsc["ekf"] = {k: v for k, v in NE.fit(list(free) + list(forced), fsn,
                                             init=fsc.get("estimate")).items() if k != "per_segment"}
        if pinn:
            from ..estimators import normalized_pinn as NP
            fsc["pinn"] = NP.fit_forced(free, forced, ns, results.get("pinn"),
                                        NP.NPINNConfig.from_dict(dict(opts["pinn"], sigma_floor=floor)), log=log)
    # sensitivity of the scale to the gravity hypothesis (same geometry, same forced data): a scale that exists only
    # under one gravity model is carried by that load, not by inertia (one-direction pushes: M a_g dt and Fc s dt
    # are collinear)
    gsens = {}
    for k, h in hyps.items():
        if h["geometry"] != H["geometry"] or not h["forced"]:
            continue
        fk = FW.forced_scale(h["free"], h["forced"], h["ns"], free_fit=h["weak"] if "estimate" in h["weak"] else None)
        gsens[k] = dict(status=fk.get("status"), beta=(fk.get("estimate") or {}).get("beta"),
                        beta_ci95=(fk.get("profile_beta") or {}).get("ci95"),
                        M=((fk.get("dimensional") or {}).get("M") or {}).get("value"),
                        M_ci95=((fk.get("dimensional") or {}).get("M") or {}).get("ci95"),
                        lam=(fk.get("estimate") or {}).get("lam"), gam=(fk.get("estimate") or {}).get("gam"))
    if gsens:
        obs = {k: v["status"] in ("FORCE_IDENTIFIED_CANDIDATE", "ABSOLUTE_SCALE_WEAK") for k, v in gsens.items()}
        carried = any(obs.values()) and not all(obs.values())
        fsc["gravity_sensitivity"] = dict(per_hypothesis=gsens, scale_carried_by_gravity_hypothesis=carried,
                                          note=("the beta interval excludes 0 under some gravity hypotheses only: the "
                                                "apparent scale is carried by the assumed gravity load (collinear "
                                                "with Coulomb friction for a one-direction push), NOT by inertia"
                                                if carried else "scale status does not depend on the gravity "
                                                                "hypothesis"))
        if carried and fsc.get("status") in ("FORCE_IDENTIFIED_CANDIDATE", "ABSOLUTE_SCALE_WEAK"):
            fsc["status_before_gravity_sensitivity"] = fsc["status"]
            fsc["status"] = "ABSOLUTE_SCALE_NOT_OBSERVABLE"
            fsc["reason"] = ("scale exists only under some gravity hypotheses (carried by the gravity load, not "
                             "inertia); " + str(fsc.get("reason")))
    if hyp_only and free_raw:
        # the SAME low-effort intervals with their MEASURED (small) effort instead of Q := 0 (sensitivity of the
        # free hypothesis; borderline impulses are reported both ways)
        alt = [SG.MotionSegment(x.name + "_measured_Q", "FORCED_MEASURED", x.joint_type, x.t, x.q, x.q_sigma, x.sign,
                                x.force_t, x.Q, x.Q_sigma, dict(x.evidence), x.q_sigma_reported) for x in free_raw]
        fa = FW.forced_scale([], alt, ns)
        fsc["low_effort_intervals_with_measured_Q"] = {k: fa.get(k) for k in ("status", "reason", "estimate", "sd",
                                                                              "profile_beta", "dimensional",
                                                                              "window_audit", "chi2_per_dof")}
        fsc["low_effort_intervals_with_measured_Q"]["gravity"] = ns.gravity.name
        fsc["low_effort_intervals_with_measured_Q"]["note"] = (
            "alternative to the free hypothesis: the measured near-noise effort enters as beta Q")
    if imp_dec not in (None, "IMPULSE_CONSISTENT") and fsc.get("status") == "FORCE_IDENTIFIED_CANDIDATE":
        fsc["status_note"] = f"declared pulse audit {imp_dec}: the pulse is excluded from the scale"
    fsc["impulse_audit"] = None if imp_aud is None else dict(rows=list(pulse_rows), decision=imp_aud["decision"])
    fsc["impulse_verdict"] = _pulse_verdict(fsc, imp_dec, pulse_rows)
    _write(out / "forced_scale_fit.json", fsc)
    # ------------------------------------------------------------------ factorized status + handoff
    contradicted = fsc.get("status") == "SCALE_CONTRADICTED" or imp_dec in (
        "NO_POSITIVE_INERTIA_EXPLANATION", "UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE")
    scale = ST.absolute_scale(fsc, imp_dec)
    ready = ST.simulator_readiness(norm, scale, kin_auth)
    axes = dict(analysis_hypothesis=akey, primary_hypothesis_normalized=norm_primary["status"],
                kinematic_authority=kin_auth, input_evidence=ST.input_evidence(free + forced, (inp.force_audit or {})
                                                                               .get("status"), contradicted),
                normalized_dynamics=norm, absolute_scale=scale, simulator_readiness=ready,
                normalized_dynamics_alternatives={k: dict(status=f.get("status"),
                                                          estimate=f.get("estimate"),
                                                          replay_rmse_over_range=(f.get("replay") or {})
                                                          .get("rmse_over_range"))
                                                  for k, f in gh["fits"].items()})
    handoff = ST.handoff_preview(jt, weak, norm, scale, ready, prior, fsc, mm_solid)
    _write(out / "simulator_handoff_preview.json", handoff)
    ident = dict(schema="forceartgs.g9.free_forced_identifiability/v1", factorized_status=axes, methods=methods,
                 structural=dict(svd=weak.get("svd"), correlation=weak.get("correlation"),
                                 profile_ci95={n: (weak.get("profile") or {}).get(n, {}).get("ci95") for n in ns.names},
                                 bootstrap=weak.get("bootstrap"), smoothing_sensitivity=weak.get("smoothing_sensitivity"),
                                 combination=weak.get("combination")),
                 absolute_scale=dict(status=scale, reason=("no trusted forced anchor: M is structurally absent from "
                                                           "the free model" if scale == "NOT_OBSERVABLE" else
                                                           fsc.get("reason")),
                                     forced=fsc.get("status")))
    _write(out / "identifiability_report.json", ident)
    adeq = dict(schema="forceartgs.g9.free_forced_model_adequacy/v1", sigma_floor=floor, unit=u["q"],
                rule=f"rmse/range <= {RP.ADEQ_RANGE} or rmse/sigma_eff <= {RP.ADEQ_SIGMA}",
                methods={m: dict(in_sample_prefix_replay=(results.get(m) or weak).get("replay", {}) and
                                 ((results.get(m) or weak).get("replay") or {}).get("aggregate"),
                                 heldout=((results.get(m) or weak).get("heldout") or {}).get("heldout_replay", {})
                                 .get("aggregate") if (results.get(m) or weak).get("heldout") else None,
                                 normalized_equation_chi2_per_dof=weak.get("chi2_per_dof") if m == "weak_form" else None)
                         for m in ["weak_form"] + [k for k in ("rollout", "ekf", "pinn") if (results.get(k) or {})
                                                   .get("estimate")]},
                forced_replay=(fsc.get("replay") or {}).get("aggregate"),
                note="a near-zero effort RMS on Q ~ 0 is NOT a dynamics-quality metric; q replay in physical units is")
    _write(out / "model_adequacy.json", adeq)
    try:
        _plots(out / "plots", rec_a, seg_ev[H["geometry"]], free, ns, weak, rp_w, results, fsc, imp_windows)
    except Exception as e:                                        # plotting never hides the numbers
        log(f"[free_forced] plotting failed: {type(e).__name__}: {e}")
    rows = comparison_rows(Path(scene).name, methods, weak, fsc, axes, gh, git)
    with open(out / "comparison_rows.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    (out / "comparison_summary.md").write_text(summary_md(Path(scene).name, jt, seg_ev["accepted_axis"], free, forced,
                                                          hyp_only, methods, gh, fsc, axes, handoff, out))
    status.update(status="OK", factorized_status=axes, primary=dict(geometry="accepted_axis", gravity=primary_g),
                  analysis_hypothesis=akey, primary_normalized_status=norm_primary,
                  n_free_segments=len(free), free_basis=seg_ev["accepted_axis"]["free_basis"],
                  n_forced_segments=len(forced), impulse=imp_dec, runtime_s=time.time() - t0,
                  production="NOT_PRODUCTION: within-record checks only; no independent episode")
    _write(out / "STATUS.json", status)
    log(f"[free_forced] {Path(scene).name}: {out} normalized {norm['status']} {norm['caps']}; scale {scale}; "
        f"sim {ready}")
    return status


def _pulse_verdict(fsc, imp_dec, pulse_rows):
    if not pulse_rows:
        return None
    tag = f"pulse_{pulse_rows[0]}-{pulse_rows[1]}"
    a = [x for x in fsc.get("window_audit") or [] if x.get("seg") == tag]
    if not a:
        return dict(verdict="NOT_EVALUATED", reason="no bracket velocities around the pulse")
    a = a[0]
    pos = a["D_over_J"] is not None and a["D_over_J"] > 0 and a["J_significant"] and a["D_significant"]
    return dict(rows=list(pulse_rows), D=a["D"], sd_D=a["sd_D"], J=a["J"], sd_J=a["sd_J"], D_over_J=a["D_over_J"],
                supports_positive_D_over_J=bool(pos), contradiction=a["contradiction"], impulse_audit=imp_dec,
                verdict=("PULSE_SUPPORTS_SCALE" if pos and imp_dec in (None, "IMPULSE_CONSISTENT") else
                         "PULSE_NOT_USED_FOR_SCALE"),
                rates_used=a["rates_used"],
                note="D = dv + lam dq + gam int sgn + int a_g with the FREE rates; no sign is ever flipped")


def _versions():
    import platform

    import scipy
    import torch
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                torch=torch.__version__, machine=platform.machine(), processor=platform.processor())


# ================================================================================================ reports
def _f(x, n=3):
    if x is None:
        return "—"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_f(v, n) for v in x) + "]"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{n}g}"
    return str(x)


def comparison_rows(scene, methods, weak, fsc, axes, gh, git):
    rows = []
    ak = gh["analysis_hypothesis"]["key"]
    for m, e in methods.items():
        p = e["parameters"]
        rp = e.get("replay") or {}
        rows.append(dict(scene=scene, geometry=ak.split("/")[0], gravity=ak.split("/")[1], method=m,
                         status=e.get("status"), lam=p.get("lam", {}).get("estimate"),
                         lam_status=p.get("lam", {}).get("status"), gam=p.get("gam", {}).get("estimate"),
                         gam_status=p.get("gam", {}).get("status"), caps="|".join(p.get("lam", {}).get("caps") or []),
                         replay_rmse=rp.get("rmse"), replay_rmse_over_range=rp.get("rmse_over_range"),
                         replay_beats_null=rp.get("beats_null"), normalized_dynamics=axes["normalized_dynamics"]["status"],
                         absolute_scale=axes["absolute_scale"], simulator_readiness=axes["simulator_readiness"],
                         code_revision=f"{(git.get('sha') or '')[:12]}+{git.get('dirty_entries')}dirty"))
    for k, f in gh["fits"].items():
        geo, gr = k.split("/")
        if k == ak:
            continue
        e = f.get("estimate") or {}
        rows.append(dict(scene=scene, geometry=geo, gravity=gr, method="weak_form", status=f.get("status"),
                         lam=e.get("lam"), lam_status=(f.get("per_parameter") or {}).get("lam", {}).get("status"),
                         gam=e.get("gam"), gam_status=(f.get("per_parameter") or {}).get("gam", {}).get("status"),
                         caps="", replay_rmse=(f.get("replay") or {}).get("rmse"),
                         replay_rmse_over_range=(f.get("replay") or {}).get("rmse_over_range"),
                         replay_beats_null=(f.get("replay") or {}).get("beats_null"),
                         normalized_dynamics="(alternative)", absolute_scale="", simulator_readiness="",
                         code_revision=f"{(git.get('sha') or '')[:12]}+{git.get('dirty_entries')}dirty"))
    return rows


def summary_md(scene, jt, ev, free, forced, hyp_only, methods, gh, fsc, axes, handoff, out):
    u = UNITS[jt]
    L = [f"# G9 free + forced normalized dynamics — {scene}", "", f"Output: `{out}`", "",
         "## Factorized status", "",
         f"- kinematic authority: **{axes['kinematic_authority']}**",
         f"- input evidence: {', '.join(axes['input_evidence'])}",
         f"- normalized dynamics: **{axes['normalized_dynamics']['status']}** "
         f"{' '.join(axes['normalized_dynamics']['caps'])}",
         f"- absolute scale: **{axes['absolute_scale']}**",
         f"- simulator readiness: **{axes['simulator_readiness']}**",
         f"- primary hypothesis ({gh['primary']['key']}): **{axes['primary_hypothesis_normalized']}**; analysis "
         f"hypothesis: **{gh['analysis_hypothesis']['key']}** — {gh['analysis_hypothesis']['rule']}", "",
         "## Segment evidence (accepted axis)", "",
         "| label | reason | rows | n_q | moving frac | J [" + u["J"] + "] | J/σ | TCP→carried point max [cm] |",
         "|---|---|---|---|---|---|---|---|"]
    for iv in ev["intervals"]:
        if iv["n_rows"] < 3 and iv["label"] != "FORCED_MEASURED":
            continue
        tt = iv.get("tcp_tracking") or {}
        L.append(f"| {iv['label']} | {iv['reason']} | {iv['rows'][0]}–{iv['rows'][1]} | {iv['n_q']} | "
                 f"{iv['moving_fraction']:.2f} | {_f(iv['effort']['J'])} | {_f(iv['effort']['J_over_sigma'])} | "
                 f"{_f(100 * tt['max_tcp_to_carried_point_m']) if tt.get('max_tcp_to_carried_point_m') is not None else '—'} |")
    L += ["", f"Free basis: **{ev['free_basis']}**; free segments used: {', '.join(s.name for s in free) or 'none'}; "
          f"forced segments: {', '.join(s.name for s in forced) or 'none'}.", ""]
    for a in ev.get("passivity_audit") or []:
        L.append(f"- passivity ({a['seg']}, gravity {a['gravity']}): **{a['verdict']}** — energy gain "
                 f"{_f(a['energy_gain'])} {a['unit']} ({_f(a['gain_over_sd'])} sd); excluded {a.get('excluded')}, "
                 f"kept {a.get('kept')}")
    L += ["",
          f"## Normalized free rates — analysis hypothesis {gh['analysis_hypothesis']['key']} (same segments, same "
          "model)", "",
          f"| method | λ = c/M [{u['lam']}] | λ status | γ = Fc/M [{u['gam']}] | γ status | replay RMSE [{u['q']}] | "
          "RMSE/range | beats null |", "|---|---|---|---|---|---|---|---|"]
    for m, e in methods.items():
        p, rp = e["parameters"], e.get("replay") or {}
        L.append(f"| {m} | {_f(p.get('lam', {}).get('estimate'))} | {p.get('lam', {}).get('status')} | "
                 f"{_f(p.get('gam', {}).get('estimate'))} | {p.get('gam', {}).get('status')} | {_f(rp.get('rmse'))} | "
                 f"{_f(rp.get('rmse_over_range'))} | {_f(rp.get('beats_null'))} |")
    L += ["", "## Geometry / gravity hypotheses (weak form + replay)", "",
          f"| hypothesis | status | λ | γ | χ²/dof | replay RMSE [{u['q']}] | RMSE/range | beats null | passivity |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, f in gh["fits"].items():
        e, rp = f.get("estimate") or {}, f.get("replay") or {}
        pv = ", ".join(sorted({a["verdict"] for a in f.get("passivity") or []})) or "—"
        L.append(f"| {k} | {f.get('status')} | {_f(e.get('lam'))} | {_f(e.get('gam'))} | {_f(f.get('chi2_per_dof'))} | "
                 f"{_f(rp.get('rmse'))} | {_f(rp.get('rmse_over_range'))} | {_f(rp.get('beats_null'))} | {pv} |")
    for k, t in (gh.get("negligible_test") or {}).items():
        L.append(f"\nGravity negligible test ({k}): **{t.get('status')}** — max |a_g| {_f(t.get('max_abs_a_g'))} "
                 f"{t.get('unit')} vs acceleration resolution {_f(t.get('acceleration_resolution'))}.")
    for k, t in (gh.get("gravity_scale_passivity_scan") or {}).items():
        L.append(f"\nGravity-scale passivity scan ({k}): the free motion stays passive up to f = "
                 f"{_f(t.get('max_passive_fraction'))} of the hull gravity (≈ hinge tilt "
                 f"{_f(t.get('equivalent_max_tilt_deg'))}° vs accepted {_f(t.get('accepted_tilt_deg'))}°).")
    ed = gh.get("effective_diagnostic") or {}
    if ed.get("svd"):
        L.append(f"\nEffective-load diagnostic: condition {_f(ed['svd']['condition_number'])} "
                 f"({ed.get('label')}).")
    L += ["", "## Forced scale / pulse audit", "", f"- forced-scale status: **{fsc.get('status')}** — "
          f"{fsc.get('reason')}"]
    pv = fsc.get("impulse_verdict")
    if pv:
        L.append(f"- declared pulse rows {pv.get('rows')}: J = {_f(pv.get('J'))} ± {_f(pv.get('sd_J'))} {u['J']}, "
                 f"D = {_f(pv.get('D'))} ± {_f(pv.get('sd_D'))} {u['v']}, D/J = {_f(pv.get('D_over_J'))}; "
                 f"impulse audit {pv.get('impulse_audit')} → **{pv.get('verdict')}**")
    d = fsc.get("dimensional")
    if d:
        L.append(f"- M = {_f(d['M']['value'])} {u['M']} (95 % {_f(d['M'].get('ci95'))}; {d.get('provenance')})")
    for k, v in ((fsc.get("gravity_sensitivity") or {}).get("per_hypothesis") or {}).items():
        L.append(f"  - under {k}: {v['status']}, β CI {_f(v['beta_ci95'])}, M {_f(v['M'])} {u['M']}, "
                 f"λ {_f(v['lam'])}, γ {_f(v['gam'])}")
    if (fsc.get("gravity_sensitivity") or {}).get("scale_carried_by_gravity_hypothesis"):
        L.append("  - **scale carried by the gravity hypothesis, not by inertia** → NOT_OBSERVABLE")
    mq = fsc.get("low_effort_intervals_with_measured_Q")
    if mq:
        L.append(f"- low-effort intervals with their MEASURED effort (instead of Q := 0): {mq.get('status')}; "
                 f"rates {_f((mq.get('estimate') or {}).get('lam'))} / {_f((mq.get('estimate') or {}).get('gam'))}, "
                 f"β {_f((mq.get('estimate') or {}).get('beta'))} ± {_f((mq.get('sd') or {}).get('beta'))}")
    L += ["", "## Simulator handoff", "", f"- readiness **{handoff['simulator']['readiness']}**, source "
          f"{handoff['simulator']['parameter_set_source']}; M estimate (for sim): "
          f"{_f(handoff['absolute_scale']['M']['estimate'])}",
          "", "Within-record checks only (no independent episode): NOT production.", ""]
    return "\n".join(L)


def _plots(d, rec, ev, free, ns, weak, rp_w, results, fsc, imp_windows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = ns.units
    # 1. q observed vs free replay
    per = (rp_w or {}).get("per_segment") or []
    n = max(len(per), 1)
    fig, axs = plt.subplots(n, 1, figsize=(8, 3.2 * n), squeeze=False)
    for ax, p, s in zip(axs[:, 0], per, free):
        ax.plot(s.t, s.q, "k.", label="q observed")
        if p.get("status") == "OK":
            ax.plot(p["t_test"], p["q_pred"], "C0-", label="weak-form replay (prefix state)")
            ax.plot(p["t_test"], p["q_null"], "C3--", label="zero-dissipation null")
            if p.get("band"):
                ax.fill_between(p["t_test"], p["band"]["lo"], p["band"]["hi"], color="C0", alpha=0.15,
                                label="95 % band")
            ax.axvline(p["state0"]["t"], color="0.5", lw=0.8)
            for m, c in (("rollout", "C2"), ("ekf", "C4"), ("pinn", "C1")):
                rr = ((results.get(m) or {}).get("replay") or {}).get("per_segment") or []
                for x in rr:
                    if x.get("seg") == s.name and x.get("status") == "OK":
                        ax.plot(x["t_test"], x["q_pred"], color=c, lw=0.8, label=f"{m} replay")
        ax.set_title(f"{s.name} [{s.label}]", fontsize=9)
        ax.set_ylabel(f"q [{u['q']}]")
        ax.legend(fontsize=7)
    axs[-1, 0].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(d / "q_observed_vs_free_replay.png", dpi=120)
    plt.close(fig)
    # 2. qdot + decay uncertainty, 3. phase space
    from ..estimators.free_weak_form import _smooth
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    e = weak.get("estimate") or {}
    for k, s in enumerate(free):
        tg, dd = _smooth(s)
        axs[0].plot(tg, dd["qd"], f"C{k}", label=s.name)
        axs[0].fill_between(tg, dd["qd"] - 2 * dd["qd_sd"], dd["qd"] + 2 * dd["qd_sd"], color=f"C{k}", alpha=0.2)
        axs[1].plot(dd["qd"], dd["qdd"], f"C{k}.", ms=2, label=f"{s.name} spline")
        if e:
            vv = np.linspace(min(dd["qd"]), max(dd["qd"]), 50)
            qq = np.interp(vv, dd["qd"][np.argsort(dd["qd"])], dd["q"][np.argsort(dd["qd"])])
            axs[1].plot(vv, ns.accel(qq, vv, s.sign, ns.pack(e.get("lam", 0), e.get("gam", 0))), f"C{k}-", lw=1,
                        label=f"{s.name} model a(v)")
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel(f"qdot [{u['v']}] (±2σ)")
    axs[1].set_xlabel(f"qdot [{u['v']}]")
    axs[1].set_ylabel(f"qddot [{u['a']}]")
    for a in axs:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "qdot_and_decay_uncertainty.png", dpi=120)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for k, s in enumerate(free):
        tg, dd = _smooth(s)
        ax2.plot(dd["q"], dd["qd"], f"C{k}-", label=s.name)
    ax2.set_xlabel(f"q [{u['q']}]")
    ax2.set_ylabel(f"qdot [{u['v']}]")
    ax2.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(d / "phase_space_diagnostics.png", dpi=120)
    plt.close(fig)
    plt.close(fig2)
    # 4. force and generalized impulse
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    t, Q, Qs = rec.force_t_s, rec.Q_object, rec.Q_sigma
    axs[0].plot(t, Q, "k-", lw=0.8)
    axs[0].fill_between(t, -3 * Qs, 3 * Qs, color="0.8", label="±3σ")
    cols = {"FREE_CONFIRMED": "C2", "FORCED_MEASURED": "C3", "UNKNOWN_OR_UNMEASURED": "C7"}
    for iv in ev["intervals"]:
        axs[0].axvspan(iv["t"][0], iv["t"][1], color=cols[iv["label"]], alpha=0.15, lw=0)
    axs[0].set_ylabel(f"Q [{u['Q']}]")
    axs[0].set_title("green FREE_CONFIRMED / red FORCED_MEASURED / grey UNKNOWN", fontsize=8)
    J = np.cumsum(np.nan_to_num(Q) * 0.02)
    sJ = np.sqrt(np.cumsum((Qs * 0.02) ** 2))
    axs[1].plot(t, J, "C0-")
    axs[1].fill_between(t, J - 2 * sJ, J + 2 * sJ, color="C0", alpha=0.2)
    axs[1].set_ylabel(f"J_Q(t) [{u['J']}]")
    for w in imp_windows:
        axs[1].axvspan(w["ta"], w["tb"], color="C3", alpha=0.3)
    axs[2].plot(rec.q_t_s, rec.q, "k.-", ms=3)
    axs[2].set_ylabel(f"q [{u['q']}]")
    axs[2].set_xlabel("t [s] (source row / 50)")
    fig.tight_layout()
    fig.savefig(d / "force_and_generalized_impulse.png", dpi=120)
    plt.close(fig)
    # 5. profiles
    prof = weak.get("profile") or {}
    pb = fsc.get("profile_beta")
    k = len(prof) + (1 if pb else 0)
    if k:
        fig, axs = plt.subplots(1, k, figsize=(4 * k, 3.4), squeeze=False)
        items = list(prof.items()) + ([("beta", pb)] if pb else [])
        for ax, (nm, p) in zip(axs[0], items):
            ax.plot(p["grid"], p["dchi2"], "k-")
            ax.axhline(p["threshold"], color="C3", lw=0.8)
            ax.set_ylim(0, max(4 * p["threshold"], 1))
            ax.set_xlabel(f"{nm} [{u.get(nm, '')}]")
            ax.set_ylabel("Δχ² (profile)")
        fig.tight_layout()
        fig.savefig(d / "normalized_parameter_profiles.png", dpi=120)
        plt.close(fig)
