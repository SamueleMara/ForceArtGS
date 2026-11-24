"""Public API of the input-output identification layer (schema forceartgs.g9.io_response/v1).

    identify_io_response(episodes, accepted_joint, contact_classification, gravity_model, config, output_dir)
    initialize_physx_refinement(io_result, existing_g10_config, existing_g10_episodes)      (physx_bridge)
    validate_response_consistency(io_result, physx_rollouts, config)                        (physx_bridge)
    reconcile_physx(io_result, sim, episodes, ...)  -> actual PhysX rollouts of the conditional candidates

Order (spec 6.4): input / kinematics audit (the caller's admissible episodes + exclusion log) -> experimental
response identification (weak form warm start, output-error variants, nested selection, profiles, block bootstrap,
sensitivities) -> physical mapping (point or ratio family) -> work / impulse audit -> method comparison -> status
dimensions -> fail-closed handoff.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import energy_audit as EA
from . import estimator_bridge as EB
from . import identifiability as IDF
from . import output_error as OE
from . import selection as SEL
from . import transfer_features as TF
from . import weak_form as WF
from .contracts import SCHEMA, UNITS, VARIANT_BY_NAME, assert_no_value_field, load_config
from .physx_bridge import (g10_parameters, initialize_physx_refinement, physx_sensitivity,  # noqa: F401
                           validate_response_consistency)


def _clean(o):
    from ..forcetwin.pipeline import _clean as c
    return c(o)


def _write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_clean(obj), indent=1, default=str))


def _M_grid(oe, prior_M_range, cfg, n=11):
    Ms = cfg["fit"]["M_search"]
    if oe.get("forced_in_training") and oe["estimate"].get("g_u"):
        Mc = 1.0 / oe["estimate"]["g_u"]
        lo, hi = Mc / 30, Mc * 30
        if prior_M_range:                       # always span the prior too, so prior-vs-data overlap is testable
            lo, hi = min(lo, prior_M_range[0] / 3), max(hi, prior_M_range[1] * 3)
        lo, hi = max(Ms[0], lo), min(Ms[1], hi)
        n = max(n, int(math.ceil(4 * math.log10(hi / lo))) + 1)
    elif prior_M_range:
        lo, hi = prior_M_range[0] / 3, prior_M_range[1] * 3
    else:
        lo, hi = 1e-2, 1e2
    return np.geomspace(lo, hi, n)


def identify_io_response(episodes, accepted_joint: dict, contact_classification: dict, gravity_model, config,
                         output_dir=None, log=print) -> dict:
    """``accepted_joint``: {joint_type, kinematic_status, export_ready, axis_std_rad, origin_std_m, joint_source,
    note}; ``contact_classification``: {excluded, force_audit_status, pulse_audit, source}; ``gravity_model``:
    {primary: GravityModel, alternatives: {name: GravityModel}, rigid_unit: RigidBodyGravity | None, M1, verified,
    prior_M_range, prior_provenance} or None (zero gravity load, labelled)."""
    from ..free_forced.normalized import GravityModel
    t_start = time.time()
    cfg = load_config(config) if not (isinstance(config, dict) and config.get("_resolved")) else config
    fc = cfg["fit"]
    jt = accepted_joint["joint_type"]
    gm = dict(gravity_model or {})
    grav = gm.get("primary") or GravityModel.zero(jt, provenance="no gravity model supplied")
    zero_g = GravityModel.zero(jt, "zero_gravity_test", provenance="audit alternative")
    eps = list(episodes)
    tr = [e for e in eps if e.train]
    kin = SEL.kinematics_validity(accepted_joint.get("kinematic_status"), accepted_joint.get("export_ready"),
                                  cfg["joint_source"])
    cond = []
    if kin != "ACCEPTED":
        cond.append(f"kinematics {kin} ({accepted_joint.get('kinematic_status')})")
    fa = contact_classification.get("force_audit_status")
    if any(e.forced for e in tr) and fa != "VERIFIED":
        cond.append(f"force input {fa} (exploratory)")
    if gm.get("verified") is False:
        cond.append("gravity direction declared, not verified")
    pulse = EA.summarize_pulse_audit(contact_classification.get("pulse_audit"))
    pulse_valid = EA.pulse_input_validity((pulse or {}).get("primary"))
    res = dict(schema=SCHEMA, joint_type=jt, units=UNITS[jt], accepted_joint=accepted_joint,
               contact_policy=dict(policy=cfg["contact_policy"], source=contact_classification.get("source"),
                                   force_audit_status=fa),
               episodes={e.name: e.describe() for e in eps},
               excluded_intervals=contact_classification.get("excluded") or [],
               gravity=dict(primary=grav.describe(), alternatives={k: g.describe() for k, g in
                                                                   (gm.get("alternatives") or {}).items()}),
               conditional_on=cond, declared_pulse_audit=pulse)
    out = Path(output_dir) if output_dir else None
    if not tr:
        dims = SEL.dimensions(n_episodes=0, kin_status=kin,
                              input_validity=SEL.input_validity(eps, None, pulse_valid, fa),
                              structural="NOT_TESTABLE", practical="NOT_TESTABLE", adequacy="NOT_RUN",
                              heldout="NOT_AVAILABLE",
                              notes=dict(STRUCTURAL_IDENTIFIABILITY="no admissible interval: nothing is claimed "
                                         "unobservable on the basis of a classifier exclusion alone"))
        res.update(status="NO_ADMISSIBLE_EPISODE", status_dimensions=dims, features=[],
                   reason="no FREE_CONFIRMED / FORCED_MEASURED window long enough (UNKNOWN / hypothesis-only "
                          "intervals are excluded, never converted to force-free)",
                   handoff=dict(schema="forceartgs.g9.io_dynamics_handoff/v1", simulator_readiness="BLOCKED",
                                simulation_export_allowed=False, estimate_not_for_sim=None,
                                simulator_prior_family=None))
        res["runtime_s"] = time.time() - t_start
        assert_no_value_field(_clean(res))
        if out:
            _write_outputs(out, res, {}, None, None, eps, None)
        return res
    variants = [VARIANT_BY_NAME[v] for v in cfg["model_variants"]]
    wvar = "L1" if "L1" in variants else "L0"
    log(f"[io] weak form ({wvar}) on {len(tr)} training window(s)")
    weak = WF.fit(eps, grav, wvar)
    init = WF.init_from(weak)
    log(f"[io] output-error variants {variants} (init {init or 'default'})")
    var = OE.fit_variants(eps, grav, fc, init=init, variants=cfg["model_variants"])
    sel = SEL.select(var)
    oe = var[sel["selected"]]
    log(f"[io] selected {sel['selected']}: {sel['reason']}")
    grid = _M_grid(oe, gm.get("prior_M_range"), cfg)
    sprof = OE.scale_profile(eps, sel["selected"], grav, fc, grid, sigma_input=(oe["_model"].sigma_input, oe["_model"].chol))
    boot = OE.block_bootstrap(oe, B=int(fc["feature_bootstrap_seeds"]))
    sens = IDF.sensitivity(oe)
    # geometry uncertainty of the gravity stiffness at each window's operating point
    gunc = {}
    rbg = gm.get("rigid_unit")
    if rbg is not None:
        from .physical_response import gravity_stiffness_uncertainty
        for e in eps:
            gunc[e.name] = gravity_stiffness_uncertainty(
                rbg, float(np.mean(e.q)), axis_sd_rad=float(accepted_joint.get("axis_std_rad") or 0.0),
                origin_sd_m=float(accepted_joint.get("origin_std_m") or 0.0), inertia=gm.get("M1"), n=300)
    ft = TF.features(oe, eps, gravity=grav, gravity_unc=gunc, conditional_reasons=cond, scale_profile=sprof)
    feats = ft["_features"]
    mapping = TF.physical_mapping(oe, feats, joint_type=jt, prior_M_range=gm.get("prior_M_range"),
                                  prior_provenance=gm.get("prior_provenance", "ASSUMED"), scale_profile=sprof)
    rates = dict(lam=oe["estimate"]["lam"], gam=oe["estimate"].get("gam") or 0.0)
    energy = EA.audit(eps, grav, rates, oe["estimate"].get("g_u"), zero_gravity=None if grav.is_zero else zero_g,
                      cfg=fc, excluded=res["excluded_intervals"], pulse_audit=pulse)
    methods = EB.run_methods(eps, grav, sel["selected"], oe, weak, fc["compare_methods"], log=log)
    comp = EB.compare(methods)
    alt = {}
    for name, g in (gm.get("alternatives") or {}).items():
        try:
            r = OE.fit(eps, sel["selected"], g, fc, init=init, profiles=False)
            alt[name] = dict(gravity=g.describe(), estimate=r["estimate"], cost=r["cost"], chi2_red=r["chi2_red"],
                             nll=r["nll"], adequate_train=SEL._adequate(r), episodes=r["episodes"])
        except Exception as ex:
            alt[name] = dict(status="FAILED", reason=f"{type(ex).__name__}: {ex}")
    gcheck = SEL.gravity_check(oe, alt)
    struct, s_why = IDF.structural(oe, sens, len(tr))
    pract, p_why = IDF.practical(oe, feats, sprof)
    adeq = "ADEQUATE" if SEL._adequate(oe) else "INADEQUATE"
    adeq_why = None
    if any(v.get("misfit_only_within_propagated_input_noise") for v in oe["episodes"].values() if v.get("train")):
        adeq_why = ("misfit exceeds the whole observed motion and the measurement noise (only the propagated input "
                    "noise of the fitted gain covers it)")
    if gcheck["status"] == "CONTRADICTED":
        adeq = "INADEQUATE"
        adeq_why = gcheck["reason"]
    dims = SEL.dimensions(n_episodes=len(tr), kin_status=kin,
                          input_validity=SEL.input_validity(eps, energy, pulse_valid, fa),
                          structural=struct, practical=pract, adequacy=adeq, heldout=SEL.heldout_status(oe),
                          notes=dict(STRUCTURAL_IDENTIFIABILITY=s_why, PRACTICAL_IDENTIFIABILITY=p_why,
                                     MODEL_ADEQUACY=adeq_why,
                                     INDEPENDENT_VALIDATION="held-out windows replayed with theta frozen from their "
                                                            "prefix initial state" if any(not e.train for e in eps)
                                     else "no held-out / independent episode: NOT_AVAILABLE (not 'unobservable')"))
    handoff = SEL.handoff(oe=oe, mapping=mapping, dims=dims, feats=feats,
                          engine_semantics="analytical hybrid ODE (exact ZOH steps); PhysX parity in "
                                           "physx_reconciliation.json")
    res.update(status="OK", selection=sel,
               variants={k: _variant_summary(v) for k, v in var.items()},
               selected_fit=_variant_summary(oe, full=True), weak_form=weak, scale_profile=sprof,
               block_bootstrap=boot, sensitivity=sens, features=ft["features"], physical_mapping=mapping,
               method_comparison=dict(methods=methods, comparison=comp),
               gravity_alternatives=alt, gravity_check=gcheck, gravity_stiffness_uncertainty=gunc,
               status_dimensions=dims, handoff=handoff,
               classification_chain=_chain(oe, feats, mapping, dims),
               runtime_s=time.time() - t_start)
    res["work_energy_audit_summary"] = dict(contradicting=energy["contradicting_windows"],
                                            nonpassive=energy["nonpassive_windows"],
                                            verdicts={w["name"]: w["verdict"] for w in energy["windows"]})
    assert_no_value_field(_clean(res))
    if out:
        _write_outputs(out, res, ft["transfer_functions"], energy, sprof, eps, oe)
    res["_oe"], res["_feats"], res["_grav"] = oe, feats, grav
    return res


def _variant_summary(v, full=False):
    if v.get("status") != "FITTED":
        return v
    keys = ["variant", "variant_name", "status", "estimate", "sd", "g_u_sd", "at_bound", "bounds", "cost", "chi2", "nll", "half_logdet_sigma",
            "chi2_red", "n_obs", "n_params", "n_nuisance", "aic", "bic", "episodes", "multistart", "optimizer",
            "n_eval", "n_failed", "forced_in_training", "uncertainty_label", "names"]
    if full:
        keys += ["profiles", "cov", "theta_raw", "gravity", "runtime_s"]
    d = {k: v.get(k) for k in keys if k in v}
    d["estimate_not_for_sim"] = d.pop("estimate", None)
    return d


def _chain(oe, feats, mapping, dims):
    fs = {f.name: f.status for f in feats}
    return dict(observed="q(t) samples + canonical effort u(t) on admitted windows",
                inferred_normalized_dynamics=dict(lam=fs.get("lambda"), gam=fs.get("gamma"), gain=fs.get("gain")),
                recoverable_physical_coefficients=mapping.get("status"),
                physx_matched_conditional_candidates="see physx_reconciliation.json (diagnostic surrogate)",
                simulation_export_readiness=dims["SIMULATOR_READINESS"])


def _write_outputs(out, res, tfs, energy, sprof, eps, oe):
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "io_response.json", res)
    _write(out / "transfer_functions.json", dict(schema="forceartgs.g9.io_transfer_functions/v1",
                                                 joint_type=res["joint_type"], windows=tfs,
                                                 note="local sliding-regime TFs only; no global TF across stick-slip"))
    _write(out / "work_energy_audit.json", energy or dict(status="NOT_RUN", reason=res.get("reason")))
    _write(out / "parameter_profiles.json", dict(
        schema="forceartgs.g9.io_parameter_profiles/v1", scale_profile_log_M=sprof,
        parameter_profiles=((res.get("selected_fit") or {}).get("profiles")), sensitivity=res.get("sensitivity"),
        family=(res.get("physical_mapping") or {}).get("family")))
    if not (out / "physx_reconciliation.json").exists():
        _write(out / "physx_reconciliation.json", dict(status="NOT_RUN", reason="no PhysX reconciliation requested "
                                                       "or no admissible episode"))
    if oe is not None:
        write_replays(out, eps, oe)


def replay_arrays(eps, oe, physx=None):
    model = oe["_model"]
    th = model.theta(oe["_z"])
    tr, ho = {}, {}
    for e in eps:
        ic = (oe["episodes"].get(e.name) or {})
        r = model.simulate(th, e, ic.get("q0_fit", e.q0), ic.get("v0_fit", e.v0))
        d = {f"{e.name}/t": e.t, f"{e.name}/q_obs": e.q, f"{e.name}/q_sigma": e.q_sigma,
             f"{e.name}/q_analytic": r.q, f"{e.name}/u_at_q_times_zoh": e.u_at(e.t)}
        if physx and e.name in physx:
            d[f"{e.name}/q_physx"] = physx[e.name]
        (tr if e.train else ho).update(d)
    return tr, ho


def write_replays(out, eps, oe, physx=None):
    tr, ho = replay_arrays(eps, oe, physx)
    np.savez(out / "training_replay.npz", **tr, timeline="native q clock; effort forward-ZOH at the q times")
    if ho:
        np.savez(out / "heldout_replay.npz", **ho, timeline="native q clock; theta frozen")


# ============================================================================================ PhysX reconciliation
def reconcile_physx(io_result, sim, episodes, *, axis_inertia_unit, cfg, log=print) -> dict:
    """Actual PhysX rollouts (the G10 worker) of the conditional candidates on the G10 episodes (same windows), the
    analytical replay at the SAME physical parameters and initial state, parity + feature errors, engine readback and
    capabilities, PhysX sensitivities. Diagnostic only (explicit-effort friction surrogate)."""
    from . import physx_bridge as PB
    oe, grav = io_result["_oe"], io_result["_grav"]
    est = oe["estimate"]
    lam, gam = est["lam"], est.get("gam") or 0.0
    # initial state: the output-error (q0, v0) nuisance for TRAINING windows (identified with the rates), the prefix
    # estimate for held-out windows (theta frozen, nothing refitted) -- identical for PhysX and the analytical replay
    from dataclasses import replace as _rep
    g10 = []
    for e in episodes:
        if e.g10 is None:
            continue
        ic = oe["episodes"].get(e.name) or {}
        g10.append(_rep(e.g10, q0=float(ic.get("q0_fit", e.g10.q0)), v0=float(ic.get("v0_fit", e.g10.v0)))
                   if e.train else e.g10)
    g10_by = {g.name: g for g in g10}
    cand = {}                                    # label -> (M, lam, gam)
    mp = io_result["physical_mapping"]
    if mp.get("M"):
        M = mp["M"]["estimate_not_for_sim"]
        ciM = mp["M"].get("ci95") or []
        cand["M_from_gain"] = (M, lam, gam)
        pts = (io_result.get("scale_profile") or {}).get("points") or []
        for k, v in zip(("ci95_low", "ci95_high"), ciM):
            if v and math.isfinite(v):
                # at a CI edge the rates are the PROFILED ones at that M (a measured input: M lam* is not equivalent)
                p = min(pts, key=lambda p: abs(math.log(p["M"] / v))) if pts else None
                cand[f"M_{k}"] = (v, p["c"] / p["M"], p["Fc"] / p["M"]) if p else (v, lam, gam)
    else:
        fam = (mp.get("family") or {}).get("members") or []
        k = int(cfg["physx"]["reconcile_family_members"])
        if fam and k > 0:
            idx = sorted({int(round(i)) for i in np.linspace(0, len(fam) - 1, min(k, len(fam)))})
            for m in (fam[i] for i in idx):
                # profile members carry their own re-optimised c, Fc; prior (free-only) members are exact M lam*
                cand[f"family_M={m['M']:.4g}"] = (m["M"], m["c"] / m["M"], m["Fc"] / m["M"])
        if not cand:
            cand["unit_scale"] = (axis_inertia_unit, lam, gam)
    params = {lab: PB.g10_parameters(M, l_, g_, axis_inertia_unit) for lab, (M, l_, g_) in cand.items()}
    log(f"[io] PhysX reconciliation: {len(params)} candidate(s) x {len(g10)} episode(s)")
    rolls = PB.candidate_rollouts(sim, g10, params)
    model = oe["_model"]
    pr = {}
    for lab, eps_r in rolls.items():
        p = params[lab]
        I = sim.effective_inertia(p)
        th = dict(lam=p.viscous / I, gam=p.dynamic_effort / I, g_u=1.0 / I, kappa=est.get("kappa", 0.0),
                  q_eq=est.get("q_eq", 0.0))
        pr[lab] = {}
        for e in episodes:
            r = eps_r.get(e.name)
            if isinstance(r, Exception) or r is None:
                pr[lab][e.name] = dict(error=f"{type(r).__name__}: {r}")
                continue
            ge = g10_by[e.name]
            qa = model.simulate(th, e, ge.q0, ge.v0, t_query=ge.t_s).q
            d = r.diagnostics or {}
            pr[lab][e.name] = dict(t=ge.t_s, q_physx=r.q, q_analytic=qa, q_obs=ge.q, sigma=ge.sigma_q,
                                   params=p.__dict__, I_eff=I, q0=ge.q0, v0=ge.v0,
                                   initial_state=("output-error fitted (q0, v0) nuisance" if e.train else
                                                  "observed-prefix estimate (held-out, nothing refitted)"),
                                   diagnostics=dict(axis_inertia_engine=d.get("axis_inertia_engine"),
                                                    axis_inertia_expected=d.get("axis_inertia_expected"),
                                                    n_contacts=d.get("n_contacts"),
                                                    sleeping_at_end=d.get("sleeping_at_end"),
                                                    readback={k: v for k, v in (d.get("readback") or {}).items()
                                                              if k not in ("moving_inertia_tensor_cmass_link_frame",)},
                                                    engine_defaults_as_loaded=d.get("engine_defaults_as_loaded"),
                                                    runtime_semantics=r.runtime_semantics,
                                                    engine_version=r.engine_version))
    rec = PB.validate_response_consistency(io_result, pr, dict(parity_tol_sigma=1.0))
    rec["engine_capabilities"] = PB.engine_capabilities(sim)
    rec["engine_settings"] = dict(timestep=sim.dt, gravity=sim.gravity.tolist(), friction_mode=sim.friction_mode,
                                  process=sim.backend_process, urdf=str(sim.urdf))
    # free-only: the PhysX family members must coincide (scale gauge in the actual engine)
    if not oe.get("forced_in_training") and len(pr) > 1:
        spreads = {}
        for e in episodes:
            qs = [np.asarray(v[e.name]["q_physx"]) for v in pr.values() if "q_physx" in v.get(e.name, {})]
            if len(qs) > 1:
                spreads[e.name] = float(np.max(np.ptp(np.vstack(qs), axis=0)))
        rec["family_trajectory_spread"] = dict(max_abs=spreads, note="free-only family members (M, M lam, M gam) "
                                               "must give the same PhysX trajectory (scale gauge)")
    if cfg["physx"]["sensitivity"] and params:
        first = next(iter(params.values()))
        try:
            rec["physx_sensitivity"] = PB.physx_sensitivity(sim, g10, first)
        except Exception as ex:
            rec["physx_sensitivity"] = dict(status="FAILED", reason=f"{type(ex).__name__}: {ex}")
    rec["status"] = "DIAGNOSTIC_ONLY_NOT_IDENTIFIED"
    rec["note"] = ("conditional candidates of the analytical identification replayed in actual PhysX (explicit-effort "
                   "friction surrogate, not native axis friction); not a G10 refit")
    physx_q = {e: next(iter(pr.values()))[e]["q_physx"] for e in next(iter(pr.values()))
               if "q_physx" in next(iter(pr.values()))[e]} if pr else {}
    return rec, physx_q
