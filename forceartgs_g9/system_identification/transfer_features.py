"""Response features (lambda, gamma, gain, kappa, poles, time constant, v_inf, Coulomb offset, stop time), the LOCAL
sliding-regime transfer functions, and the Stage-B algebraic mapping to physical parameters (Eq. 13-15).

    M = 1/g_u,  c = M lam,  Fc = M gam,  K_eff = M kappa                                                (13)
    g_u not identified  ->  family {(M, M lam, M gam): M in [M_min, M_max]_prior}  (ASSUMED bounds)       (14)

Every feature carries status (IDENTIFIED | CONDITIONAL | RATIO_ONLY | NOT_IDENTIFIED | NOT_APPLICABLE), uncertainty,
evidence and the source windows. A gain is never reported from free-only motion. CONDITIONAL marks a feature whose
interval is closed but that is conditioned on unaccepted kinematics, an exploratory force input, or declared
gravity. A transfer function is emitted per window ONLY for a moving, fixed-sign, sliding regime.
"""
from __future__ import annotations

import math

import numpy as np

from .contracts import UNITS, ResponseFeature
from .physical_response import LocalLinearModel, classify_poles, poles

REL_OK = 0.5


def _interval_status(est, ci, open_lo, open_hi, reaches_zero=False, conditional=False):
    if est is None or ci is None or ci[0] is None:
        return "NOT_IDENTIFIED", "no estimate / empty profile interval"
    if reaches_zero:
        return "NOT_IDENTIFIED", f"95 % profile interval reaches 0 (upper bound {ci[1]:.4g} only)"
    if open_lo or open_hi:
        return "NOT_IDENTIFIED", "95 % profile interval open at the search bound"
    if est > 0 and (ci[1] - ci[0]) / 2 > REL_OK * est:
        return "NOT_IDENTIFIED", f"95 % half-width > {REL_OK:.0%} of the estimate (weak)"
    return ("CONDITIONAL" if conditional else "IDENTIFIED"), "closed 95 % profile interval"


def features(oe, episodes, *, gravity, gravity_unc=None, conditional_reasons=(), scale_profile=None) -> dict:
    """``oe``: the selected output-error fit; ``gravity_unc``: {episode: physical_response.gravity_stiffness_
    uncertainty in normalized units} (kappa_g = a_g'(q_b) with geometry uncertainty)."""
    jt = episodes[0].joint_type
    u = UNITS[jt]
    cond = bool(conditional_reasons)
    tr = [e.name for e in episodes if e.train]
    est = oe["estimate"]
    prof = oe.get("profiles") or {}
    feats = []
    ev_common = [f"output-error fit {oe['variant']} ({oe['variant_name']})", f"gravity: {gravity.name}"] + \
        [f"conditional: {r}" for r in conditional_reasons]

    def rate(name, key, unit):
        p = prof.get(key) or {}
        st, why = _interval_status(est.get(key), p.get("ci95"), p.get("open_low"), p.get("open_high"),
                                   p.get("reaches_zero", False), cond)
        return ResponseFeature(name, est.get(key), unit, st,
                               dict(ci95=p.get("ci95"), sd_local=(oe.get("sd") or {}).get(key), reason=why,
                                    profile_threshold=p.get("threshold")), ev_common, tr)
    feats.append(rate("lambda", "lam", u["lam"]))
    if est.get("gam") is not None:
        feats.append(rate("gamma", "gam", u["gam"]))
        g = feats[-1]
        feats.append(ResponseFeature("coulomb_offset", g.estimate, u["gam"], g.status,
                                     dict(g.uncertainty, meaning="fixed signed offset s*gam in a moving regime; its "
                                          "velocity derivative is zero (not a viscous term)"), g.evidence, tr))
    else:
        feats.append(ResponseFeature("gamma", None, u["gam"], "NOT_APPLICABLE", dict(reason="Coulomb term not in "
                                     "the selected variant"), ev_common, tr))
    # gain
    if not oe.get("forced_in_training"):
        feats.append(ResponseFeature("gain", None, u["g_u"], "NOT_IDENTIFIED",
                                     dict(reason="free-only training data: g_u = 1/M is absent from the model "
                                          "(scale symmetry) -- never labelled measured from free motion"),
                                     ev_common, tr))
    else:
        p = prof.get("log_g_u") or {}
        ci = p.get("ci95_g_u")
        st, why = _interval_status(est.get("g_u"), ci, p.get("open_low"), p.get("open_high"), False, cond)
        if scale_profile and scale_profile.get("status") == "FLAT_SCALE_SYMMETRY":
            st, why = "NOT_IDENTIFIED", "dimensional scale profile is flat"
        feats.append(ResponseFeature("gain", est.get("g_u"), u["g_u"], st,
                                     dict(ci95=ci, ci95_M=p.get("ci95_M"), reason=why,
                                          sd_local=oe.get("g_u_sd")), ev_common, tr))
    lam_f = feats[0]
    lam = est.get("lam") or 0.0
    lam_sd = (oe.get("sd") or {}).get("lam") or 0.0
    if lam > 0 and lam_f.status in ("IDENTIFIED", "CONDITIONAL"):
        ci = lam_f.uncertainty.get("ci95")
        feats.append(ResponseFeature("time_constant", 1.0 / lam, "s", lam_f.status,
                                     dict(ci95=[1 / ci[1], 1 / ci[0]] if ci and ci[0] and ci[0] > 0 else None,
                                          meaning="T = M/c = 1/lam"), ev_common, tr))
    else:
        feats.append(ResponseFeature("time_constant", None, "s", "NOT_IDENTIFIED",
                                     dict(reason="lambda not identified (or zero)"), ev_common, tr))
    # per-window kappa / poles / v_inf / stop time
    tfs = {}
    model = oe.get("_model")
    th = model.theta(oe["_z"]) if model is not None else None
    rng = np.random.default_rng(0)
    for e in episodes:
        epm = (oe.get("episodes") or {}).get(e.name, {})
        qb = float(np.mean(e.q))
        r = None
        if model is not None:
            try:
                r = model.simulate(th, e, epm.get("q0_fit", e.q0), epm.get("v0_fit", e.v0), record_events=True)
            except (ValueError, FloatingPointError):
                r = None
        stops = [ev for ev in (r.events if r is not None else []) if ev.get("kind") in ("stop", "breakaway")]
        vb = float(np.mean(r.qdot)) if r is not None else e.sign * abs(e.v0)
        sliding = (not stops) and r is not None and bool(np.all(np.sign(r.qdot[np.abs(r.qdot) > 0]) == e.sign))
        kg = float(gravity.deriv(qb))
        gu_ = (gravity_unc or {}).get(e.name) or {}
        kg_sd = float(gu_.get("kappa_g_sd") or 0.0)
        kap = kg + (est.get("kappa") or 0.0)
        kap_src = "geometry gravity a_g'(q_b)" + (" + fitted configuration load" if est.get("kappa") else "")
        if gravity.is_zero and not est.get("kappa"):
            k_status, k_reason = ("CONDITIONAL", "zero gravity stiffness by the declared/accepted geometry "
                                  "(vertical hinge / horizontal slide): kappa = 0, one integrator pole at 0")
        else:
            k_status, k_reason = "CONDITIONAL", "geometry-derived (accepted axis / COM; uncertainty propagated)"
        feats.append(ResponseFeature(f"kappa[{e.name}]", kap, u["kappa"], k_status,
                                     dict(sd_geometry=kg_sd, source=kap_src, reason=k_reason, q_bar=qb),
                                     ev_common, [e.name]))
        if not sliding:
            feats.append(ResponseFeature(f"poles[{e.name}]", None, "1/s", "NOT_APPLICABLE",
                                         dict(reason="window not a single fixed-sign sliding regime (stop / "
                                              "breakaway / reversal): no single transfer function"),
                                         ev_common, [e.name]))
            tfs[e.name] = dict(status="NOT_APPLICABLE", reason="stick / stop / reversal inside the window",
                               events=stops)
        else:
            p1, p2 = poles(lam, kap)
            samp = []
            for _ in range(400):
                lm = max(0.0, lam + lam_sd * rng.standard_normal())
                kk = kap + kg_sd * rng.standard_normal()
                a, b = poles(lm, kk)
                samp.append([complex(a).real, complex(b).real])
            samp = np.asarray(samp)
            pst = "NOT_IDENTIFIED" if lam_f.status == "NOT_IDENTIFIED" else "CONDITIONAL"
            feats.append(ResponseFeature(f"poles[{e.name}]", None, "1/s", pst,
                                         dict(p1=_c(p1), p2=_c(p2), **classify_poles(p1, p2),
                                              p1_real_interval90=np.quantile(samp[:, 0], [0.05, 0.95]).tolist(),
                                              p2_real_interval90=np.quantile(samp[:, 1], [0.05, 0.95]).tolist(),
                                              propagation="MC over lambda (local sd) and kappa (geometry sd)"),
                                         ev_common, [e.name]))
            ub = float(np.mean(e.u_at(e.t))) if e.forced else 0.0
            nm = [est.get("g_u") or 0.0]
            tfs[e.name] = dict(status="LOCAL_SLIDING_ONLY", sliding_only=True, mode=e.mode,
                               operating_point=dict(q_bar=qb, v_bar=vb, u_bar=ub, sign=e.sign),
                               normalized=dict(H_q=dict(num=nm, den=[1.0, lam, kap]),
                                               H_v=dict(num=nm + [0.0], den=[1.0, lam, kap]),
                                               form="g_u / (s^2 + lam s + kappa)",
                                               g_u_status=next(f.status for f in feats if f.name == "gain")),
                               coulomb_offset=dict(estimate_not_for_sim=(est.get("gam") or 0.0) * e.sign,
                                                   meaning="signed offset s*gam (removed from the linear part)"),
                               nominal_load_normalized=float(gravity(qb)),
                               poles=[_c(p1), _c(p2)], **classify_poles(p1, p2),
                               validity="continuous-time, local, sliding regime; not the static-friction / solver "
                                        "model; no single TF across stick-slip")
            if e.forced and est.get("g_u") and lam > 0:
                vinf = (est["g_u"] * ub - float(gravity(qb)) - e.sign * (est.get("gam") or 0.0)) / lam
                feats.append(ResponseFeature(f"v_inf[{e.name}]", vinf, u["v"],
                                             "CONDITIONAL" if lam_f.status != "NOT_IDENTIFIED" else "NOT_IDENTIFIED",
                                             dict(u_bar=ub, formula="(g_u u_bar - a_g(q_b) - s gam) / lam"),
                                             ev_common, [e.name]))
        if not e.forced:
            st_m, st_o = epm.get("stop_time_model"), epm.get("stop_time_observed")
            feats.append(ResponseFeature(f"stop_time[{e.name}]", st_m, "s",
                                         "CONDITIONAL" if (st_m is not None and st_o is not None) else "NOT_APPLICABLE",
                                         dict(observed=st_o, error=epm.get("stop_time_error"),
                                              reason=None if st_o is not None else "no stop inside the window"),
                                         ev_common, [e.name]))
    return dict(features=[f.to_dict() for f in feats], _features=feats, transfer_functions=tfs)


def _c(p):
    p = complex(p)
    return dict(re=p.real, im=p.imag)


def physical_mapping(oe, feats, *, joint_type, prior_M_range=None, prior_provenance="ASSUMED",
                     scale_profile=None) -> dict:
    """Stage B. Point mapping only when the gain is IDENTIFIED / CONDITIONAL. Free-only data: an equivalence family
    over the EXPLICIT prior range (marked ASSUMED; exact symmetry). Forced data with a weak gain: the members are the
    DATA-CONSISTENT points of the absolute-scale profile (each with its own re-optimised c, Fc -- (M lam*, M gam*) is
    NOT equivalent when a measured input is present), plus a prior / data overlap check. Revolute: I_h only."""
    u = UNITS[joint_type]
    est = oe["estimate"]
    lam, gam = est.get("lam") or 0.0, est.get("gam") or 0.0
    gain = next(f for f in feats if f.name == "gain")
    out = dict(equation="M = 1/g_u, c = M lam, Fc = M gam, K_eff = M kappa", units=dict(M=u["M"], c=u["c"],
                                                                                         Fc=u["Fc"]),
               inferred=("hinge inertia I_h only (not the full tensor)" if joint_type == "revolute" else
                         "effective moving mass m_eff"))
    if gain.status in ("IDENTIFIED", "CONDITIONAL") and gain.estimate:
        M = 1.0 / gain.estimate
        ciM = gain.uncertainty.get("ci95_M")
        out.update(status=gain.status, M=dict(estimate_not_for_sim=M, ci95=ciM),
                   c=dict(estimate_not_for_sim=M * lam, ci95_conditional=None if not ciM else [ciM[0] * lam,
                                                                                            ciM[1] * lam]),
                   Fc=dict(estimate_not_for_sim=M * gam, ci95_conditional=None if not ciM else [ciM[0] * gam,
                                                                                             ciM[1] * gam]),
                   provenance="FORCE_SCALED (measured input gain); conditional on the input / kinematics gates")
    elif oe.get("forced_in_training") and scale_profile and scale_profile.get("status") in ("OPEN", "CLOSED"):
        thr = float(scale_profile["threshold"])
        pts = [dict(M=float(p["M"]), c=float(p["c"]), Fc=float(p["Fc"]), delta_cost=float(d))
               for p, d in zip(scale_profile["points"], scale_profile["delta_cost"]) if d <= thr]
        pts.sort(key=lambda p: p["M"])
        ci = scale_profile.get("ci95_M") or [None, None]
        overlap = None
        if prior_M_range and ci[0] is not None and ci[1] is not None:
            dl = 0.0 if scale_profile.get("open_low") else float(ci[0])        # open side: unbounded by the data
            dh = math.inf if scale_profile.get("open_high") else float(ci[1])
            lo, hi = max(prior_M_range[0], dl), min(prior_M_range[1], dh)
            overlap = dict(prior_M_range=list(prior_M_range), data_ci95_M=list(ci),
                           open_low=bool(scale_profile.get("open_low")),
                           open_high=bool(scale_profile.get("open_high")),
                           status="OVERLAP" if lo <= hi else "PRIOR_DATA_CONFLICT",
                           overlap=[lo, hi] if lo <= hi else None)
        out.update(status="DATA_CONSISTENT_FAMILY", reason=f"gain {gain.status} with a measured input: the absolute "
                   "scale is constrained by the force but not closed; members are the profile points inside the 95 % "
                   "set, each with its own profiled c, Fc", ratios=dict(lam=lam, gam=gam),
                   prior_overlap=overlap,
                   family=dict(M_range=[pts[0]["M"], pts[-1]["M"]] if pts else None,
                               M_range_provenance="absolute-scale profile (data), not the prior",
                               members=pts))
    else:
        if prior_M_range is None:
            out.update(status="RATIO_ONLY", reason="gain not identified and no explicit prior range: ratios only",
                       ratios=dict(lam=lam, gam=gam), family=None)
        else:
            Ms = np.geomspace(prior_M_range[0], prior_M_range[1], 7)
            out.update(status="RATIO_ONLY", reason=f"gain {gain.status}: equivalence family over an {prior_provenance} "
                       "prior range (not a measured mass)", ratios=dict(lam=lam, gam=gam),
                       family=dict(M_range=list(prior_M_range), M_range_provenance=prior_provenance,
                                   members=[dict(M=float(m), c=float(m * lam), Fc=float(m * gam)) for m in Ms]))
    return out


def local_model_at(oe, e, gravity):
    """LocalLinearModel at the window operating point (conditional on M = 1/g_u)."""
    est = oe["estimate"]
    if not est.get("g_u"):
        return None
    M = 1.0 / est["g_u"]
    qb = float(np.mean(e.q))
    return LocalLinearModel(M, M * (est.get("lam") or 0.0), M * (float(gravity.deriv(qb)) + (est.get("kappa") or 0.0)),
                            e.joint_type, M * ((est.get("gam") or 0.0) * e.sign + float(gravity(qb))))


def finite(x):
    return x is not None and math.isfinite(x)
