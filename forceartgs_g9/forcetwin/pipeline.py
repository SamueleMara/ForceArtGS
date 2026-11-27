"""ForceTwin-adaptation orchestration: identifiability-first effective joint dynamics on existing real runs.

Output (append-only): <run>/dynamics/forcetwin_adaptation/vNNN/
    input_manifest.json, joint_interaction_record.{npz,json}, kinematics_hypotheses.json,
    impulse_balance.{json,png}, identification/{nnls_direct,nnls_weak,forcetwin_linear_<hyp>,general_*}.json,
    identifiability.json, model_adequacy.json, mechanism_residual/ (opt-in), simulation_handoff.json,
    comparison.{csv,md}, STATUS.json
Steps: audit (record + kinematic hypotheses + signed impulse audit), identify (NNLS direct / weak, nested variants,
per joint-geometry hypothesis), general (integral LS / rollout / EKF [/ PINN] on the SAME minimal model and data),
residual (structured mechanism residual on a frozen baseline).
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from ..estimators import forcetwin_linear as FL
from . import record as RC

STEPS = ("audit", "identify", "general", "residual")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return x if math.isfinite(x) else str(x)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_clean(obj), indent=1, default=str))


def _git():
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                               text=True).stdout.count("\n")
    except OSError:
        sha, dirty = None, None
    return dict(sha=sha, dirty_entries=dirty)


def next_version(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    vs = sorted(p for p in root.glob("v[0-9][0-9][0-9]") if p.is_dir())
    out = root / f"v{len(vs):03d}"
    out.mkdir()
    return out


def _vertical_record(inp, kin, contact_segments):
    h = kin["hypotheses"]["vertical_hinge"]
    fm = np.asarray(kin["tracks"]["fm_source"], int)
    qv = np.asarray([np.nan if x is None else x for x in h["q_rgbd_frames"]], float)
    rc = inp.record_context
    ok = np.isfinite(qv)
    qs = np.interp(fm[ok], np.asarray(rc["q_src"], float), np.asarray(rc["q_std"], float))
    return RC.from_g9_inputs(inp, hypothesis_id="vertical_hinge_refit", axis=h["axis_B"], origin=h["origin_B"],
                             q_override=(fm[ok], qv[ok], qs), contact_segments=contact_segments)


def _mass_model(inp, mode):
    from ..dynamics.geometry_inertia import PASSIVE, GeometryMassModel
    if inp.mass_mesh is None or inp.gravity is None:
        return None
    rp = inp.reference_pose or {}
    if rp.get("status") not in ("PASS", "DECLARED"):
        return None
    return GeometryMassModel.from_validated_mesh(inp.mass_mesh, mode, inp.joint_type, inp.axis,
                                                 None if inp.joint_type == "prismatic" else inp.origin,
                                                 source_provenance=inp.mass_mesh_provenance or dict(source="proxy"),
                                                 reflected_actuator_inertia=PASSIVE, q_ref=float(rp["q_ref"]))


def _prior_for_sim(inp, mm):
    """Defensible PRIOR (not a measurement) on I_eff from the part's size: I = m w^2 / 3 (plate about its edge),
    m in [2, 20] kg (revolute) / I = m (prismatic)."""
    if mm is None:
        return dict(status="NOT_AVAILABLE", reason="no mass geometry")
    V = inp.mass_mesh.vertices
    if inp.joint_type == "revolute":
        a = mm.axis
        r = V - mm.origin
        w = float(np.max(np.linalg.norm(r - np.outer(r @ a, a), axis=1)))
        lo, hi = 2.0 * w * w / 3, 20.0 * w * w / 3
        return dict(status="PRIOR_FOR_SIM", I_eff_range=[lo, hi], unit="kg m^2", basis=f"plate of width {w:.3f} m "
                    "about its edge, mass 2-20 kg (assumed range, NOT measured)", distribution="log-uniform")
    return dict(status="PRIOR_FOR_SIM", I_eff_range=[2.0, 20.0], unit="kg", basis="moving mass 2-20 kg (assumed)",
                distribution="log-uniform")


def run(run_root, scene, *, cfg_path, robot_archive=None, contact_segments=None, discovery_dir=None,
        pulse_rows=None, steps=("audit", "identify", "general"), pinn=False, out_root=None,
        declared_joint_type=None, declaration_note=None, bootstrap=200, log=print) -> dict:
    from ..adapters.forceartgs_g9 import build_inputs
    from ..config import load_dynamics_config
    t0 = time.time()
    bad = sorted(set(steps) - set(STEPS))
    if bad:
        raise ValueError(f"unknown steps {bad}")
    out = next_version(Path(out_root) if out_root else Path(run_root) / "dynamics" / "forcetwin_adaptation")
    cfg = load_dynamics_config(cfg_path)
    cfg["pipeline"]["allow_diagnostic_kinematics"] = True
    status = dict(schema="forceartgs.g9.forcetwin_status/v1", scene=Path(scene).name, out_dir=str(out),
                  steps=list(steps), git=_git(),
                  states=dict(KINEMATICS_ACCEPTED=None, FORCE_CALIBRATION=None, JOINT_DYNAMICS_IDENTIFIED=False,
                              MECHANISM_VALIDATED=False, MASS_ALLOCATION_ASSUMED=False, ASSET_COMPILES="NOT_RUN",
                              ENGINE_REPLAY_VALIDATED="NOT_RUN"))
    inp = build_inputs(run_root, scene, cfg, robot_archive=robot_archive, contact_segments=contact_segments,
                       declared_joint_type=declared_joint_type, declaration_note=declaration_note, log=log)
    status["states"]["KINEMATICS_ACCEPTED"] = inp.kinematic_status
    status["states"]["FORCE_CALIBRATION"] = (inp.force_audit or {}).get("status")
    manifest = dict(schema="forceartgs.g9.forcetwin_input_manifest/v1", run=str(run_root), scene=str(scene),
                    config=str(cfg_path), robot_archive=str(robot_archive) if robot_archive else None,
                    contact_segments=str(contact_segments) if contact_segments else None,
                    discovery_dir=str(discovery_dir) if discovery_dir else None, input_hashes=inp.input_hashes,
                    kinematic_status=inp.kinematic_status, kinematics_provenance=inp.kinematics_provenance,
                    gravity=None if inp.gravity is None else dict(g_B=inp.gravity.g_world,
                                                                  provenance=inp.gravity.provenance),
                    reference_pose=inp.reference_pose, git=status["git"])
    _write(out / "input_manifest.json", manifest)
    try:
        rec = RC.from_g9_inputs(inp, contact_segments=contact_segments)
    except RC.RecordError as e:
        fa = inp.force_audit or {}
        if fa.get("status") == "NOT_EVALUATED" and inp.kinematic_status not in ("IDENTIFIED", "OPERATOR_DECLARED"):
            status.update(status="KINEMATICS_NOT_ACCEPTED", reason=f"upstream joint status {inp.kinematic_status} "
                          f"({(inp.kinematics_provenance or {}).get('selection_status')}): no joint to project the "
                          "force on; diagnostics only")
        else:
            status.update(status="DATA_UNAVAILABLE", reason=str(e))
        _write(out / "STATUS.json", status)
        log(f"[forcetwin] {Path(scene).name}: {status['status']} ({status['reason']})")
        return status
    rec.save(out)
    rc = inp.record_context
    # ------------------------------------------------------------------ kinematic hypotheses
    kin = None
    if inp.joint_type == "revolute" and discovery_dir and (Path(discovery_dir) / "tracks_observations.npz").is_file():
        from .kinematics_hypotheses import hypotheses
        kin = hypotheses(discovery_dir, inp.axis, inp.origin)
    _write(out / "kinematics_hypotheses.json", kin or dict(status="NOT_RUN", reason=(
        "prismatic working model: no vertical-hinge question" if inp.joint_type == "prismatic" else
        "no discovery track observations")))
    recs = {"accepted_axis": rec}
    if kin and kin.get("status") != "NOT_RUN":
        try:
            recs["vertical_hinge_refit"] = _vertical_record(inp, kin, contact_segments)
        except RC.RecordError as e:
            kin["vertical_record_error"] = str(e)
    # ------------------------------------------------------------------ signed impulse audit
    imp_state = None
    if "audit" in steps:
        if pulse_rows:
            from ..validation.impulse_balance import audit, plot
            aud = audit(rec, pulse_rows, kin_hyp=kin, observability=rc.get("observability"),
                        robot_record=rc["rec"])
            imp_state = aud["decision"]["primary"]
            _write(out / "impulse_balance.json", aud)
            try:
                plot(rec, aud, rc["rec"], out / "impulse_balance.png")
            except Exception as e:                           # a plotting failure never hides the numbers
                log(f"[forcetwin] plot failed: {e}")
            status["impulse"] = dict(rows=list(pulse_rows), decision=aud["decision"]["primary"],
                                     states=aud["decision"]["states"],
                                     absolute_inertia=aud["decision"]["absolute_inertia"])
        else:
            _write(out / "impulse_balance.json", dict(status="NOT_RUN", reason="no declared pulse rows"))
    # ------------------------------------------------------------------ minimal NNLS / weak identification
    hyps = {}
    mm_solid = _mass_model(inp, "uniform_solid")
    mm_shell = _mass_model(inp, "uniform_shell")
    g = None if inp.gravity is None else inp.gravity.g_world
    hyps["accepted_axis_no_gravity_term"] = dict(rec=rec, grav=None, grav_mm=None, label=(
        "DIAGNOSTIC: gravity term omitted on the accepted axis" if inp.joint_type == "revolute" else
        "DIAGNOSTIC: slider gravity omitted"))
    for nm, mm in (("solid", mm_solid), ("shell", mm_shell)):
        if mm is not None:
            hyps[f"accepted_axis_known_gravity_{nm}_hull"] = dict(
                rec=rec, grav=FL.GravityHypothesis.from_mass_model(mm, g, f"known_gravity_{nm}_hull"), grav_mm=mm,
                label=f"KNOWN GRAVITY tied to the {nm}-hull mass geometry (tau_grav = I_eff G_1/I_axis,1; "
                      "TENTATIVE geometry, DECLARED vertical)")
    if "vertical_hinge_refit" in recs:
        hyps["vertical_hinge_refit"] = dict(rec=recs["vertical_hinge_refit"], grav=None, grav_mm=None, label=(
            "HYPOTHESIS: exactly vertical hinge re-estimated from the RGB-D tracks (own q, own effort); gravity "
            "torque identically zero; NOT the accepted joint"))
    tilt = float(np.degrees(np.arccos(min(1.0, abs(np.asarray(inp.axis, float)[2] / np.linalg.norm(inp.axis))))))
    vert_ok = ("vertical_hinge_refit" in hyps and tilt < 10.0 and
               (kin or {}).get("comparison", {}).get("verdict") != "VERTICAL_WORSE_GEOMETRICALLY")
    primary = ("vertical_hinge_refit" if vert_ok else
               "accepted_axis_known_gravity_solid_hull" if "accepted_axis_known_gravity_solid_hull" in hyps else
               "accepted_axis_no_gravity_term")
    status["primary_rule"] = ("vertical-hinge refit is primary only for a nearly vertical accepted hinge (tilt < 10 "
                              "deg) that the RGB-D tracks do not reject; otherwise the accepted axis with the known "
                              f"(hull-tied) gravity. Accepted-axis tilt from vertical: {tilt:.1f} deg -> "
                              f"{'vertical refit eligible' if vert_ok else 'accepted axis used'}")
    kin_gate = inp.kinematic_status == "IDENTIFIED"
    studies = {}
    if "identify" in steps or "general" in steps or "residual" in steps:
        pr = _prior_for_sim(inp, mm_solid)
        I_prior = pr.get("I_eff_range")
        for hn, h in hyps.items():
            st = FL.study(h["rec"], grav=h["grav"], grav_mm=h["grav_mm"], gravity=inp.gravity,
                          impulse_state=imp_state, B=bootstrap, I_prior=I_prior)
            st["hypothesis"] = hn
            st["hypothesis_label"] = h["label"]
            if not kin_gate:
                for vv in st["variants"].values():
                    for e in (vv.get("identifiability") or {}).values():
                        e["kinematics_gate"] = f"KINEMATICS_NOT_ACCEPTED ({inp.kinematic_status})"
                        if e["identifiability_status"] == "ESTIMATED_CONDITIONAL":
                            e["identifiability_status"] = "CONDITIONAL_ON_UNACCEPTED_KINEMATICS"
            studies[hn] = st
            _write(out / "identification" / f"forcetwin_linear_{hn}.json", st)
        P = studies[primary]["variants"]["inertia_viscous_coulomb"]
        _write(out / "identification" / "nnls_direct.json", dict(hypothesis=primary, fit=P["primary"],
                                                                 identifiability=P["identifiability"],
                                                                 replay=P["replay"]))
        _write(out / "identification" / "nnls_weak.json", dict(hypothesis=primary, fit=P["weak_primary"]))
    # ------------------------------------------------------------------ general estimators, same model / data
    general = {}
    if "general" in steps and studies:
        general = _general(hyps[primary], studies[primary], inp, imp_state, pinn, out, log,
                           I_prior=_prior_for_sim(inp, mm_solid).get("I_eff_range"), kin_gate=kin_gate)
    # ------------------------------------------------------------------ residual
    if "residual" in steps and studies:
        from ..estimators.forcetwin_residual import ablations
        e = studies[primary]["variants"]["inertia_viscous_coulomb"]["primary"].get("estimates") or {}
        beta = [e.get(k, {}).get("estimate", 0.0) for k in ("I_eff", "mu", "b")]
        if beta[0] <= 1e-9:
            res = dict(status="NOT_RUN", reason="baseline I_eff at its bound: no defensible frozen baseline")
        else:
            res = ablations(hyps[primary]["rec"], beta, grav=hyps[primary]["grav"])
        _write(out / "mechanism_residual" / "residual_ablations.json", res)
        status["states"]["MECHANISM_VALIDATED"] = False
    # ------------------------------------------------------------------ aggregate reports
    ident = {hn: st["variants"]["inertia_viscous_coulomb"]["identifiability"] for hn, st in studies.items()}
    for m, r in general.items():
        ident.setdefault(primary + "__general", {})[m] = r.get("identifiability")
    _write(out / "identifiability.json", dict(schema="forceartgs.g9.forcetwin_identifiability/v1", primary=primary,
                                              impulse_state=imp_state, per_hypothesis=ident,
                                              coast=studies.get(primary, {}).get("coast")))
    adeq = {hn: {v: dict(direct=x["primary"].get("whitened_residual_rms"),
                         weak=(x["weak_primary"] or {}).get("whitened_residual_rms"),
                         replay=(x["replay"] or {}).get("mean_rmse_over_sigma"),
                         rank=x["primary"].get("rank"), cond=x["primary"].get("condition_number"))
                 for v, x in st["variants"].items()} | {"diagnostic_ranking": st["model_comparison"]}
            for hn, st in studies.items()}
    _write(out / "model_adequacy.json", dict(schema="forceartgs.g9.forcetwin_model_adequacy/v1", per_hypothesis=adeq))
    rows = comparison_rows(Path(scene).name, studies, general, imp_state, inp, status["git"])
    with open(out / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["scene"])
        w.writeheader()
        w.writerows(rows)
    (out / "comparison.md").write_text(comparison_md(rows))
    handoff = simulation_handoff(inp, studies.get(primary), general, mm_solid, mm_shell, imp_state, primary)
    _write(out / "simulation_handoff.json", handoff)
    ok_I = ((ident.get(primary) or {}).get("I_eff") or {}).get("identifiability_status") == "ESTIMATED_CONDITIONAL"
    status["states"]["JOINT_DYNAMICS_IDENTIFIED"] = bool(ok_I) and imp_state in (None, "IMPULSE_CONSISTENT")
    status["states"]["MASS_ALLOCATION_ASSUMED"] = handoff.get("mass_allocation", {}).get("status") == "CONDITIONAL"
    status.update(status="OK", primary_hypothesis=primary, hypotheses={k: v["label"] for k, v in hyps.items()},
                  parameters=ident.get(primary), runtime_s=time.time() - t0,
                  production="NOT_PRODUCTION: no independent held-out episode; exploratory calibration")
    _write(out / "STATUS.json", status)
    log(f"[forcetwin] {Path(scene).name}: {out} primary {primary}; impulse {imp_state}")
    return status


def _general(h, st, inp, imp_state, pinn, out, log, I_prior=None, kin_gate=True):
    from ..estimators.general_constrained import general_rollout_fit
    from ..estimators.general_ekf import general_akf_fit
    from ..estimators.general_integral_ls import general_integral_ls, theta0_from_ls
    variant = "inertia_viscous_coulomb"
    rec = h["rec"]
    spec = FL.spec_for(rec.joint_type, variant, h["grav_mm"], inp.gravity)
    its = rec.interactions()
    e = st["variants"][variant]["primary"].get("estimates") or {}
    est = {k: v["estimate"] for k, v in e.items()}
    eff = float(np.max(np.abs(rec.Q_object)))
    ls = general_integral_ls(its, spec)
    th_ls = theta0_from_ls(ls, spec, dict(effort=eff))
    th0 = FL.theta_for(spec, est, h["grav_mm"]) if est.get("I_eff", 0) > 1e-9 else th_ls
    init = "NNLS (direct, spline)" if est.get("I_eff", 0) > 1e-9 else "integral-LS initializer (NNLS I_eff at bound)"
    res = {}

    def pack(name, r, th):
        if r is None:
            return
        x = np.asarray(r.get("x", th), float)[:len(spec.names)]
        ph = spec.physical(x)
        I = float(ph["M"])
        pe = {"I_eff": I, "mu": float(ph.get("Fc") or 0.0), "b": float(ph.get("c") or 0.0)}
        rp = FL.replay(rec, spec, x)
        stds, labels = {}, {}
        for n, k in (("I_eff", "alpha" if "alpha" in spec.names else "M"), ("mu", "Fc"), ("b", "c")):
            v = (r.get("estimates") or {}).get(k, {})
            sd = v.get("std")
            if k == "alpha" and sd is not None and h["grav_mm"] is not None:
                sd = sd * h["grav_mm"].M1
            stds[n] = sd
            labels[n] = v.get("uncertainty_label", "")
        idf = {}
        nis = r.get("nis") if isinstance(r.get("nis"), dict) else None
        for n, v in pe.items():
            sd = stds.get(n)
            status, why = FL.gate(n, v, sd, uncertainty_label=labels[n], estimator_status=r.get("status"),
                                  I_prior=I_prior, impulse_state=imp_state, nis=nis,
                                  replay_rmse=rp.get("mean_rmse_over_sigma"), kin_ok=kin_gate,
                                  kin_status=inp.kinematic_status)
            idf[n] = dict(estimate=v, unit=FL.units(rec.joint_type)[n], method=name, model_variant=variant,
                          own_uncertainty=sd, uncertainty_label=labels[n], identifiability_status=status,
                          reasons=why,
                          open_loop_replay=rp.get("mean_rmse_over_sigma"), production=False)
        res[name] = dict(status=r.get("status"), estimates=pe, identifiability=idf, replay=rp, initializer=init,
                         runtime_s=r.get("runtime_s"), nis=r.get("nis"))
        _write(out / "identification" / f"{name}.json", dict(res[name], raw={k: v for k, v in r.items()
                                                                              if k not in ("timelines", "problem",
                                                                                           "jacobian")}))
    pack("general_integral_ls", dict(status="OK", x=th_ls, estimates=ls.get("estimates")), th_ls)
    log("[forcetwin] rollout ...")
    pack("general_rollout", general_rollout_fit(its, spec, th0, n_starts=4, seed=0), th0)
    log("[forcetwin] augmented EKF ...")
    pack("general_ekf", general_akf_fit(its, spec, th0, s_Q=float(np.median(rec.Q_sigma))), th0)
    if pinn:
        from ..estimators.general_pinn import PINNConfig, general_pinn_fit
        log("[forcetwin] GeneralPINN (long) ...")
        pack("general_pinn", general_pinn_fit(its, spec, th0, PINNConfig()), th0)
    else:
        _write(out / "identification" / "general_pinn.json", dict(status="NOT_RUN",
                                                                  reason="--pinn not given (long run)"))
    return res


def comparison_rows(scene, studies, general, imp_state, inp, git):
    rows = []
    for hn, st in studies.items():
        for v, x in st["variants"].items():
            for est_name, f in (("NNLS_direct", x["primary"]), ("NNLS_weak", x["weak_primary"] or {})):
                if f.get("status") != "OK":
                    rows.append(dict(scene=scene, hypothesis=hn, variant=v, estimator=est_name, status=f.get("status")))
                    continue
                e = f["estimates"]
                idf = x["identifiability"] or {}
                rows.append(dict(
                    scene=scene, hypothesis=hn, variant=v, estimator=est_name, residual="none", status="OK",
                    **{f"{p}": e.get(p, {}).get("estimate") for p in FL.PARAMS},
                    **{f"{p}_ci95": e.get(p, {}).get("interval95_statistical") for p in FL.PARAMS},
                    **{f"{p}_status": (idf.get(p, {}) if est_name == "NNLS_direct" else {}).get(
                        "identifiability_status") for p in FL.PARAMS},
                    rank=f["rank"], condition_number=f["condition_number"],
                    whitened_effort_rms=f["whitened_residual_rms"],
                    open_loop_rmse_over_sigma=(x["replay"] or {}).get("mean_rmse_over_sigma")
                    if est_name == "NNLS_direct" else None,
                    impulse_state=imp_state, n_train_episodes=len(inp.train), n_heldout_episodes=0,
                    geometry_hypothesis=st.get("hypothesis"), calibration="EXPLORATORY (declared)",
                    code_revision=f"{(git.get('sha') or '')[:12]}+{git.get('dirty_entries')}dirty"))
    for m, r in general.items():
        rows.append(dict(scene=scene, hypothesis="primary", variant="inertia_viscous_coulomb", estimator=m,
                         residual="none", status=r.get("status"),
                         **{p: r["estimates"].get(p) for p in FL.PARAMS},
                         **{f"{p}_status": r["identifiability"][p]["identifiability_status"] for p in FL.PARAMS},
                         open_loop_rmse_over_sigma=(r["replay"] or {}).get("mean_rmse_over_sigma"),
                         nis=(r.get("nis") or {}).get("mean") if isinstance(r.get("nis"), dict) else None,
                         impulse_state=imp_state, n_train_episodes=len(inp.train), n_heldout_episodes=0,
                         runtime_s=r.get("runtime_s"),
                         code_revision=f"{(git.get('sha') or '')[:12]}+{git.get('dirty_entries')}dirty"))
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    return [{k: r.get(k) for k in keys} for r in rows]


def _f(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3g}"
    if isinstance(x, list):
        return "[" + ", ".join(_f(v) for v in x) + "]"
    return str(x)


def comparison_md(rows):
    if not rows:
        return "no rows\n"
    cols = ["hypothesis", "variant", "estimator", "status", "I_eff", "I_eff_ci95", "I_eff_status", "mu", "mu_status", "b",
            "b_status", "rank", "condition_number", "whitened_effort_rms", "open_loop_rmse_over_sigma", "nis"]
    cols = [c for c in cols if any(c in r for r in rows)]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_f(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def simulation_handoff(inp, st, general, mm_solid, mm_shell, imp_state, primary) -> dict:
    out = dict(schema="forceartgs.g9.forcetwin_simulation_handoff/v1", primary_hypothesis=primary,
               priority=["I_eff", "mu", "b"], impulse_state=imp_state)
    v = (st or {}).get("variants", {}).get("inertia_viscous_coulomb", {})
    idf = v.get("identifiability") or {}
    terms = {}
    for p in FL.PARAMS:
        e = idf.get(p) or {}
        ok = e.get("identifiability_status") == "ESTIMATED_CONDITIONAL" and imp_state in (None, "IMPULSE_CONSISTENT")
        # a non-identified point estimate is NEVER placed in a field a simulator would read
        terms[p] = dict(estimate=e.get("estimate") if ok else None,
                        estimate_not_for_sim=None if ok else e.get("estimate"), unit=e.get("unit"),
                        status=e.get("identifiability_status"),
                        use_in_sim="CONDITIONAL_ESTIMATE" if ok else "PRIOR_ONLY (sample the prior; do not use "
                                                                    "estimate_not_for_sim)")
    out["joint_terms"] = terms
    I_ok = terms["I_eff"]["status"] == "ESTIMATED_CONDITIONAL" and imp_state in (None, "IMPULSE_CONSISTENT")
    if I_ok and mm_solid is not None:
        I = terms["I_eff"]["estimate"]
        alloc = {}
        for nm, mm in (("solid_hull", mm_solid), ("shell_hull", mm_shell)):
            if mm is None:
                continue
            rho = I / mm.M1
            alloc[nm] = dict(density=rho, density_units=mm.props.density_units, mass_kg=rho * mm.props.m1,
                             com_B=mm.props.c1.tolist(), inertia_com_kg_m2=(rho * mm.props.J1).tolist())
        out["mass_allocation"] = dict(status="CONDITIONAL", label="MASS_ALLOCATION_ASSUMED (rho = (I_eff - I_ref) / "
                                      "I_axis,1; uniform tentative hull; NOT a measured mass)", templates=alloc)
    else:
        out["mass_allocation"] = dict(status="NOT_ALLOCATED", reason=(
            "I_eff not identified / impulse audit not consistent: the geometry equation is not inverted"),
            prior=_prior_for_sim(inp, mm_solid))
    out["engines"] = dict(
        urdf=dict(joint="<dynamics damping=b friction=mu/> (friction semantics are engine-specific)",
                  inertia="link <inertial> needs m, COM, full tensor: only from a mass allocation (conditional) -- a "
                          "scalar hinge inertia does not fit a URDF joint field",
                  residual="RUNTIME_PLUGIN_REQUIRED (a bare URDF cannot carry the neural residual)"),
        mjcf=dict(joint="damping=b, frictionloss=mu; armature can carry a reflected / effective joint inertia",
                  note="verify runtime readback"),
        sapien_maniskill=dict(joint="joint friction / damping via the articulation API",
                              note="the URDF loader's friction handling was observed to differ from the requested "
                                   "value in an earlier G9 check: verify at runtime for the installed version"),
        isaac_usd=dict(note="PhysX joint friction / damping attributes; runtime parity NOT_RUN"),
        status="ENGINE_REPLAY_VALIDATED: NOT_RUN (no engine was executed here)")
    out["export_readiness"] = dict(KINEMATICS_ACCEPTED=inp.kinematic_status == "IDENTIFIED",
                                   JOINT_DYNAMICS_IDENTIFIED=bool(I_ok), MECHANISM_VALIDATED=False,
                                   MASS_ALLOCATION_ASSUMED=out["mass_allocation"]["status"] == "CONDITIONAL",
                                   ASSET_COMPILES="NOT_RUN", ENGINE_REPLAY_VALIDATED="NOT_RUN",
                                   label="assumption / prior preview only")
    return out
