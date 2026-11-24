"""Factorized status (spec section 7) and the TYPED simulator handoff preview (no generic ``value`` field).

Axes:
  kinematic_authority  : ACCEPTED | OPERATOR_DECLARED | WORKING_ONLY | REJECTED
  input_evidence       : FREE_CONFIRMED | FORCE_CALIBRATED | EXPLORATORY_DECLARED | FREE_HYPOTHESIS | UNKNOWN |
                         CONTRADICTED   (a set, per segment kind)
  normalized_dynamics  : FREE_RATIOS_IDENTIFIED | FREE_COMBINATION_IDENTIFIED | FREE_UNINFORMATIVE |
                         FREE_MODEL_INADEQUATE  (+ CONDITIONAL_ON_FREE_HYPOTHESIS / _ON_UNACCEPTED_KINEMATICS caps)
  absolute_scale       : FORCE_IDENTIFIED | MEASURED_MASS_DERIVED | GEOMETRY_PRIOR_DERIVED | NOT_OBSERVABLE |
                         CONTRADICTED
  simulator_readiness  : PHYSICAL_IDENTIFIED | PRIOR_SCALED_PREVIEW | INSUFFICIENT_PHYSICAL_DATA | BLOCKED
"""
from __future__ import annotations

import math

import numpy as np

from .normalized import UNITS


def kinematic_authority(kin_status):
    return {"IDENTIFIED": "ACCEPTED", "OPERATOR_DECLARED": "OPERATOR_DECLARED", "REJECTED": "REJECTED"}.get(
        str(kin_status), "WORKING_ONLY")


def input_evidence(segments, force_cal_status, contradicted):
    ev = set()
    for s in segments:
        if s.label == "FREE_CONFIRMED":
            ev.add("FREE_CONFIRMED")
        elif s.label.startswith("FREE_HYPOTHESIS"):
            ev.add("FREE_HYPOTHESIS")
        elif s.label == "FORCED_MEASURED":
            ev.add("FORCE_CALIBRATED" if str(force_cal_status) in ("PASS", "CALIBRATED") else "EXPLORATORY_DECLARED")
    if contradicted:
        ev.add("CONTRADICTED")
    return sorted(ev) or ["UNKNOWN"]


def normalized_status(weak, replay_ok, replay_beats_null, *, hypothesis_only, kin_auth):
    st = (weak or {}).get("status", "FREE_UNINFORMATIVE")
    caps, why = [], list((weak or {}).get("reasons") or [])
    if st in ("FREE_RATIOS_IDENTIFIED", "FREE_COMBINATION_IDENTIFIED"):
        if replay_ok is False:
            st = "FREE_MODEL_INADEQUATE"
            why.append("free forward replay fails at a physically meaningful scale")
        elif replay_beats_null is False:
            why.append("fitted dissipation does not beat the zero-dissipation null replay")
    if hypothesis_only:
        caps.append("CONDITIONAL_ON_FREE_HYPOTHESIS")
    if kin_auth != "ACCEPTED":
        caps.append(f"CONDITIONAL_ON_{kin_auth}_KINEMATICS")
    return dict(status=st, caps=caps, reasons=why)


def absolute_scale(forced, impulse_decision, mass_measured=False, geometry_prior=False):
    if forced and forced.get("status") == "SCALE_CONTRADICTED":
        return "CONTRADICTED"
    if impulse_decision not in (None, "IMPULSE_CONSISTENT") and not (forced and forced.get("status") ==
                                                                     "FORCE_IDENTIFIED_CANDIDATE"):
        return "CONTRADICTED" if impulse_decision in ("NO_POSITIVE_INERTIA_EXPLANATION",
                                                      "UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE") else "NOT_OBSERVABLE"
    if forced and forced.get("status") == "FORCE_IDENTIFIED_CANDIDATE" and forced.get("replay_ok", False):
        return "FORCE_IDENTIFIED"
    if mass_measured:
        return "MEASURED_MASS_DERIVED"
    if geometry_prior:
        return "GEOMETRY_PRIOR_DERIVED"
    return "NOT_OBSERVABLE"


def simulator_readiness(norm, scale, kin_auth):
    if kin_auth == "REJECTED":
        return "BLOCKED"
    if scale == "FORCE_IDENTIFIED" and norm["status"] == "FREE_RATIOS_IDENTIFIED" and not norm["caps"]:
        return "PHYSICAL_IDENTIFIED"
    if norm["status"] in ("FREE_RATIOS_IDENTIFIED", "FREE_COMBINATION_IDENTIFIED"):
        return "PRIOR_SCALED_PREVIEW"
    return "INSUFFICIENT_PHYSICAL_DATA"


def handoff_preview(joint_type, weak, norm, scale, readiness, prior, forced=None, mass_model=None,
                    preview_scenarios=(0.0, 0.5, 1.0)) -> dict:
    """Typed handoff. Normalized estimates only when the normalized status supports them; M only as
    estimate_not_for_sim unless FORCE_IDENTIFIED; a positive prior-scaled preview only as a labelled preview."""
    u = UNITS[joint_type]
    est = (weak or {}).get("estimate") or {}
    per = (weak or {}).get("per_parameter") or {}
    ok_norm = norm["status"] in ("FREE_RATIOS_IDENTIFIED", "FREE_COMBINATION_IDENTIFIED")
    npar = {}
    for n, key in (("lam", "lam_c_over_M"), ("gam", "gam_Fc_over_M")):
        p = per.get(n, {})
        supported = ok_norm and p.get("status") in ("RATIO_IDENTIFIED",)
        npar[key] = dict(estimate=est.get(n) if supported else None, unit=u[n], status=p.get("status", norm["status"]),
                         ci95=p.get("ci95"), upper95=p.get("upper95"), caps=norm["caps"],
                         estimate_not_for_sim=None if supported else est.get(n))
    comb = (weak or {}).get("combination")
    out = dict(schema="forceartgs.g9.simulator_handoff_preview/v1", joint_type=joint_type,
               normalized_parameters=npar,
               normalized_combination=comb if norm["status"] == "FREE_COMBINATION_IDENTIFIED" else None,
               normalized_dynamics_status=norm)
    M_est = ((forced or {}).get("dimensional") or {}).get("M")
    out["absolute_scale"] = dict(M=dict(estimate=M_est["value"] if (M_est and scale == "FORCE_IDENTIFIED") else None,
                                        estimate_not_for_sim=None if (not M_est or scale == "FORCE_IDENTIFIED")
                                        else M_est.get("value"),
                                        unit=u["M"], status=scale))
    sim = dict(readiness=readiness, parameter_set_source=(
        "FORCE_IDENTIFIED" if readiness == "PHYSICAL_IDENTIFIED" else "EXPLICIT_PRIOR_NOT_MEASURED"),
        recommended_prior_range=(prior or {}).get("I_eff_range"), prior=prior)
    # prior-scaled preview: positive, follows the identified free decay, cannot claim a forced response
    if readiness == "PRIOR_SCALED_PREVIEW" and prior and prior.get("I_eff_range"):
        lo, hi = prior["I_eff_range"]
        lam = est.get("lam", 0.0) if per.get("lam", {}).get("status") == "RATIO_IDENTIFIED" else None
        gam = est.get("gam", 0.0) if per.get("gam", {}).get("status") == "RATIO_IDENTIFIED" else None
        if comb and lam is None and gam is None:
            lam, gam = est.get("lam", 0.0), est.get("gam", 0.0)        # the fitted point ON the identified combination
        scen = []
        for f in preview_scenarios:
            M = float(math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo))))
            sc = dict(quantile_log_uniform=f, M=M, c=None if lam is None else lam * M,
                      Fc=None if gam is None else gam * M, units=dict(M=u["M"], c=u["c"], Fc=u["Fc"]))
            if mass_model is not None and joint_type == "revolute" and mass_model.M1 > 0:
                rho = M / mass_model.M1
                sc["mass_distribution"] = dict(density=rho, density_units=mass_model.props.density_units,
                                               mass_kg=rho * mass_model.props.m1,
                                               com_B=np.asarray(mass_model.props.c1).tolist(),
                                               inertia_com_kg_m2=(rho * np.asarray(mass_model.props.J1)).tolist(),
                                               note="ONE uniform tentative hull for inertia AND gravity")
            elif joint_type == "prismatic":
                sc["mass_distribution"] = dict(mass_kg=M, note="single rigid translating mass (M = m assumed)")
            scen.append(sc)
        sim["preview"] = dict(label="PRIOR_SCALED_PREVIEW -- NOT IDENTIFIED (free decay identified; absolute scale "
                                    "from an explicit prior; forced response NOT calibrated)", scenarios=scen)
    out["simulator"] = sim
    out["engines"] = dict(
        urdf="<dynamics damping=c friction=Fc/> + link <inertial> from ONE positive mass distribution; friction "
             "semantics engine-specific; Fs not identified",
        mjcf="damping=c, frictionloss=Fc, armature may carry reflected inertia; verify runtime readback",
        sapien_maniskill="articulation joint friction / damping API (URDF-loader friction differed from the request "
                         "in an earlier G9 check: verify at runtime)",
        isaac_usd="PhysX joint friction / damping attributes; parity NOT_RUN",
        gravity="the engine's own gravity with the SAME mass distribution (never a second gravitational mass)",
        status="ENGINE_REPLAY_VALIDATED: NOT_RUN")
    return out
