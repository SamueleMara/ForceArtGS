"""Nested model selection on identical windows, the seven SEPARATE status dimensions, and the fail-closed simulator
handoff (``estimate_not_for_sim`` + ``simulator_prior_family``; never a ``value`` field)."""
from __future__ import annotations

import math

from .contracts import STATUS_DIMENSIONS, assert_no_value_field

LR_CRIT = {1: 3.841, 2: 5.991}


def _adequate(fit):
    eps = [v for v in (fit.get("episodes") or {}).values() if v.get("train")]
    return bool(eps) and all(v.get("adequate") for v in eps)


GRAVITY_DNLL = 3.0          # likelihood ratio e^3 ~ 20 between same-dimension hypotheses


def gravity_check(primary: dict, alternatives: dict) -> dict:
    """Primary gravity / geometry hypothesis vs the separately labelled alternatives, by the full Gaussian NLL
    (including log det of the parameter-dependent covariance; same variant, same number of parameters). The
    alternatives are hypotheses, never ground truth: a decisive loss CONTRADICTS the primary as modelled."""
    rows = {}
    for name, a in (alternatives or {}).items():
        if a.get("nll") is None or primary.get("nll") is None:
            rows[name] = dict(status="NOT_TESTED", reason=a.get("reason", "no likelihood"))
            continue
        d = float(primary["nll"] - a["nll"])
        rows[name] = dict(delta_nll_primary_minus_alt=d, alt_adequate=bool(a.get("adequate_train")),
                          primary_adequate=_adequate(primary),
                          status="ALTERNATIVE_DECISIVE" if d > GRAVITY_DNLL and a.get("adequate_train") else
                          ("ALTERNATIVE_PREFERRED_WEAKLY" if d > 0 else "PRIMARY_PREFERRED"))
    dec = [n for n, r in rows.items() if r.get("status") == "ALTERNATIVE_DECISIVE"]
    if dec:
        best = max(dec, key=lambda n: rows[n]["delta_nll_primary_minus_alt"])
        return dict(status="CONTRADICTED", alternatives=rows, preferred=best,
                    reason=f"primary gravity / geometry hypothesis contradicted: '{best}' is better by delta NLL "
                           f"{rows[best]['delta_nll_primary_minus_alt']:.3g} > {GRAVITY_DNLL} (INCONSISTENT_GRAVITY_"
                           "GEOMETRY: axis direction, gravity load or an unmeasured support -- not resolved here)")
    return dict(status="CONSISTENT" if rows else "NOT_TESTED", alternatives=rows,
                reason=None if rows else "no labelled alternative")


def select(variants: dict) -> dict:
    """Minimal adequate model: L0 first; a larger nested variant only when it improves the robust cost beyond the
    likelihood-ratio threshold (scaled by max(1, chi2_red)) on the SAME windows. Nothing is turned off silently --
    every candidate is tabulated."""
    rows = []
    for k in ("L0", "L1", "L2", "L3", "L4"):
        v = variants.get(k) or {}
        if v.get("status") != "FITTED":
            rows.append(dict(variant=k, status=v.get("status", "NOT_RUN"), reason=v.get("reason")))
            continue
        rows.append(dict(variant=k, status="FITTED", cost=v["cost"], chi2_red=v["chi2_red"], n_params=v["n_params"],
                         aic=v["aic"], bic=v["bic"], adequate_train=_adequate(v)))
    fitted = {r["variant"]: variants[r["variant"]] for r in rows if r["status"] == "FITTED"}
    if not fitted:
        return dict(selected=None, table=rows, reason="no fitted variant")
    sel, why = "L0", ["L0 is the minimal model"]
    for big, small in (("L1", "L0"), ("L3", "L1")):
        if big not in fitted or small not in fitted or sel != small:
            continue
        a, b = fitted[small], fitted[big]
        dp = b["n_params"] - a["n_params"]
        stat = 2 * (a["cost"] - b["cost"])
        thr = LR_CRIT.get(dp, 3.841 * dp) * max(1.0, b["chi2_red"])
        better = stat > thr
        if better or (not _adequate(a) and _adequate(b)):
            sel = big
            why.append(f"{big} over {small}: 2 delta cost {stat:.3g} > {thr:.3g}" if better else
                       f"{big} adequate where {small} is not")
        else:
            why.append(f"{big} not supported over {small}: 2 delta cost {stat:.3g} <= {thr:.3g}")
    if sel == "L3":
        why.append("L3 selected: a CONFIGURATION-DEPENDENT LOAD candidate (spring, drive, or model mismatch) -- not "
                   "called a spring without independent mechanism evidence")
    return dict(selected=sel, table=rows, reason="; ".join(why), adequate=_adequate(fitted[sel]))


def dimensions(*, n_episodes, kin_status, input_validity, structural, practical, adequacy, heldout,
               engine_native=False, notes=None) -> dict:
    d = dict(INPUT_VALIDITY=input_validity, KINEMATICS_VALIDITY=kin_status, STRUCTURAL_IDENTIFIABILITY=structural,
             PRACTICAL_IDENTIFIABILITY=practical, MODEL_ADEQUACY=adequacy, INDEPENDENT_VALIDATION=heldout)
    ready = (input_validity == "VERIFIED" and kin_status == "ACCEPTED" and structural == "FULL"
             and practical == "INFORMATIVE" and adequacy == "ADEQUATE" and heldout == "PASS" and engine_native)
    if ready:
        sr = "PRODUCTION_READY"
    elif n_episodes and adequacy == "ADEQUATE" and input_validity != "CONTRADICTED" and kin_status != "UNACCEPTED" \
            and practical not in ("UNSTABLE", "NOT_TESTABLE"):
        sr = "CONDITIONAL_PREVIEW"
    else:
        sr = "BLOCKED"
    d["SIMULATOR_READINESS"] = sr
    for k, v in d.items():
        if v not in STATUS_DIMENSIONS[k]:
            raise ValueError(f"{k}={v!r} not in {STATUS_DIMENSIONS[k]}")
    d["_notes"] = notes or {}
    return d


def kinematics_validity(kinematic_status, export_ready, joint_source="accepted_authority"):
    if kinematic_status == "IDENTIFIED" and export_ready:
        return "ACCEPTED"
    if kinematic_status in ("IDENTIFIED", "OPERATOR_DECLARED") or joint_source == "diagnostic_working_authority":
        return "CONDITIONAL"
    return "UNACCEPTED"


def input_validity(episodes, energy, pulse_validity=None, force_audit_status=None):
    forced = [e for e in episodes if e.forced]
    if (energy or {}).get("contradicting_windows") or pulse_validity == "CONTRADICTED":
        return "CONTRADICTED"
    if not forced:
        return "UNKNOWN"
    if all(e.calibration == "VERIFIED" for e in forced) and force_audit_status == "VERIFIED":
        return "VERIFIED"
    return "CONDITIONAL"


def heldout_status(oe):
    held = [v for v in (oe.get("episodes") or {}).values() if not v.get("train")] if oe else []
    if not held:
        return "NOT_AVAILABLE"
    return "PASS" if all(v.get("adequate") for v in held) else "FAIL"


def handoff(*, oe, mapping, dims, feats, engine_semantics, prior_family=None) -> dict:
    """Versioned handoff skeleton (no PArtTwin import): point values only as estimate_not_for_sim."""
    est = oe["estimate"] if oe else {}
    h = dict(schema="forceartgs.g9.io_dynamics_handoff/v1", simulator_readiness=dims["SIMULATOR_READINESS"],
             simulation_export_allowed=dims["SIMULATOR_READINESS"] == "PRODUCTION_READY",
             estimate_not_for_sim=dict(normalized=dict(lam=est.get("lam"), gam=est.get("gam"), g_u=est.get("g_u")),
                                       physical=mapping),
             simulator_prior_family=prior_family or (mapping or {}).get("family"),
             feature_status={f.name: f.status for f in feats},
             friction_implementation=engine_semantics,
             rule="non-identified point estimates appear only as estimate_not_for_sim; a production property needs "
                  "accepted kinematics, verified input, FULL + INFORMATIVE identifiability, adequacy, a passing "
                  "independent episode and native engine parity")
    assert_no_value_field(h)
    return h


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)
