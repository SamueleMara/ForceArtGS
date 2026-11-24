"""Diagnostic work / impulse / passivity audit (spec 5.1, Eq. 16) -- AUDIT ONLY, never an extra loss on the same samples.

    int u qd dt = Delta(1/2 M qd^2 + U(q)) + int c qd^2 dt + int Fc |qd| dt + W_other                        (16)

Per window, with the canonical effort u (forward ZOH, native rows, unchanged sign) and an uncertainty-aware smoothing
spline of q (Monte Carlo over q AND effort noise -- correlations kept):

    W_in  = int u dq                     (exact per ZOH row: u_k [q(t_k+1) - q(t_k)])           [J]
    J     = int u dt                     (exact ZOH impulse)                                    [N m s | N s]
    e_n   = Delta(1/2 v^2) + Delta U_n,  U_n = int a_g dq    (mechanical energy change per unit M)
    D_n   = lam int v^2 dt + gam int |v| dt                   (modelled dissipation per unit M, >= 0)
    D_imp = Delta v + lam Delta q + gam s Delta t + int a_g dt   (J = M D_imp for the minimal model)

Scale-free contradictions (no M needed):
  * forced: W_in < 0 significantly while e_n > 0 significantly -> no M > 0 is passive -> INPUT_OUTPUT_SIGN_INCONSISTENT
  * forced: J and D_imp both significant with opposite signs -> M = J / D_imp < 0 -> INPUT_OUTPUT_SIGN_INCONSISTENT
  * free  : e_n > 0 significantly (energy gain without input) -> NONPASSIVE_UNMODELED_FORCE, or
            INCONSISTENT_GRAVITY_GEOMETRY when the gain exists only under the assumed gravity load
Passivity bound on the scale: W_in >= M e_n  ->  M <= W_in / e_n (e_n > 0) or M >= W_in / e_n (e_n < 0, W_in < 0).
With a fitted g_u: w_other_n = g_u W_in - e_n - D_n; significantly < 0 -> POSSIBLE_UNMODELED_DRIVE, > 0 ->
UNMODELED_DISSIPATION. Sensitivity: effort time offset +-lag, lever / projection scale (1 +- rel). The force SIGN is
the canonical one and is not varied (no second sign switch; a contradiction is reported, never optimised away).
"""
from __future__ import annotations

import math

import numpy as np

Z = 3.0


def _segment_of(ep):
    if ep.segment is not None:
        return ep.segment
    from ..free_forced.segments import MotionSegment
    return MotionSegment(ep.name, "FORCED_MEASURED" if ep.forced else "FREE_CONFIRMED", ep.joint_type, ep.t, ep.q,
                         ep.q_sigma, ep.sign, ep.effort_t, ep.effort,
                         ep.effort_sigma if ep.effort_sigma is not None else
                         (None if ep.effort is None else np.zeros_like(ep.effort)))


def _quantities(tg, qg, vg, tf, Qf, ag, rates, sign):
    """Window quantities on the fine grid for ONE sample of (q, v, u)."""
    lam, gam = rates.get("lam") or 0.0, rates.get("gam") or 0.0
    dq = float(qg[-1] - qg[0])
    dv = float(vg[-1] - vg[0])
    de_k = 0.5 * float(vg[-1] ** 2 - vg[0] ** 2)
    a = ag(qg)
    dU = float(np.trapz(a, qg))
    int_ag_dt = float(np.trapz(a, tg))
    D_n = lam * float(np.trapz(vg * vg, tg)) + gam * float(np.trapz(np.abs(vg), tg))
    T = float(tg[-1] - tg[0])
    W = J = 0.0
    if tf is not None and len(tf):
        tb = np.r_[tf[1:], np.inf]
        for k in range(len(tf)):
            a_, b_ = max(tf[k], tg[0]), min(tb[k], tg[-1])
            if b_ <= a_:
                continue
            qa_, qb_ = np.interp([a_, b_], tg, qg)
            W += Qf[k] * (qb_ - qa_)
            J += Qf[k] * (b_ - a_)
        # rows before the first effort timestamp: forward ZOH holds the first level (no extrapolation beyond)
    D_imp = dv + lam * dq + gam * sign * T + int_ag_dt
    return dict(dq=dq, dv=dv, de_kin_n=de_k, dU_n=dU, e_n=de_k + dU, D_n=D_n, int_ag_dt=int_ag_dt, W_in=W, J=J,
                D_imp=D_imp, T=T)


def window_audit(ep, gravity, rates=None, g_u=None, *, n_mc=200, seed=0, lag_s=0.02, lever_rel=0.1,
                 zero_gravity=None) -> dict:
    from scipy.interpolate import make_smoothing_spline
    from ..estimators.free_weak_form import gcv_lambda
    rates = dict(rates or {})
    seg = _segment_of(ep)
    t, q, s = np.asarray(seg.t, float), np.asarray(seg.q, float), np.asarray(seg.q_sigma, float)
    n = len(t)
    out = dict(name=ep.name, mode=ep.mode, train=ep.train, n=n, window=[float(t[0]), float(t[-1])],
               sign_convention="canonical object-side effort (G9 record), not varied", rates_used=rates,
               g_u_used=g_u, gravity=gravity.name)
    if n < 6:
        out.update(verdict="SHORT_UNINFORMATIVE_WINDOW", codes=["SHORT_UNINFORMATIVE_WINDOW"],
                   reason=f"{n} q samples")
        return out
    lam_s = gcv_lambda(t, q, s)
    w = 1.0 / s ** 2
    w = w / w.mean()
    edge = float(np.median(np.diff(t)))                   # spline end derivatives are biased: trim one sample
    tg = np.linspace(t[0] + edge, t[-1] - edge, max(200, 8 * n))
    ag = (lambda x: np.zeros_like(x)) if gravity.is_zero else gravity
    tf = None if not ep.forced else np.asarray(seg.force_t, float)
    Qf = None if not ep.forced else np.asarray(seg.Q, float)
    Qs = None if not ep.forced else (np.zeros_like(Qf) if seg.Q_sigma is None else np.asarray(seg.Q_sigma, float))
    if ep.forced:
        keep = tf <= t[-1]
        tf, Qf, Qs = tf[keep], Qf[keep], Qs[keep]
    rng = np.random.default_rng(seed)

    def sample(qq, QQ, tff, fac=1.0):
        sp = make_smoothing_spline(t, qq, w=w, lam=None if lam_s is None else lam_s * fac)
        return _quantities(tg, sp(tg), sp.derivative(1)(tg), tff, QQ, ag, rates, seg.sign)
    nom = sample(q, Qf, tf)
    mc = [sample(q + rng.standard_normal(n) * s, None if Qf is None else Qf + rng.standard_normal(len(Qf)) * Qs, tf)
          for _ in range(int(n_mc))]
    keys = list(nom)
    sd_mc = {k: float(np.std([m[k] for m in mc], ddof=1)) for k in keys}
    # velocity-inference (smoothing bandwidth) sensitivity, added in quadrature: the MC covers noise only
    bw = [sample(q, Qf, tf, f) for f in (0.3, 3.0)]
    sd_bw = {k: 0.5 * float(max(abs(b[k] - nom[k]) for b in bw)) * 2 for k in keys}
    sd = {k: float(np.hypot(sd_mc[k], sd_bw[k])) for k in keys}
    out.update(nominal=nom, sd=sd, sd_noise_mc=sd_mc, sd_smoothing_bandwidth=sd_bw, n_mc=int(n_mc),
               spline_lambda=lam_s, window_trimmed_s=edge,
               units=dict(W_in="J", J="N m s" if ep.joint_type == "revolute" else "N s", e_n="(q/s)^2 per unit M"))
    codes = []
    sig = lambda k: abs(nom[k]) > Z * max(sd[k], 1e-300)              # noqa: E731
    noise_q = float(np.median(s))
    if abs(nom["dq"]) < Z * noise_q:
        codes.append("MOTION_NOISE_DOMINATED")
    if ep.forced:
        if not sig("J"):
            codes.append("SHORT_UNINFORMATIVE_WINDOW")
        if sig("W_in") and sig("e_n") and nom["W_in"] < 0 < nom["e_n"]:
            codes.append("INPUT_OUTPUT_SIGN_INCONSISTENT")
        if rates and sig("J") and sig("D_imp") and np.sign(nom["J"]) != np.sign(nom["D_imp"]):
            if "INPUT_OUTPUT_SIGN_INCONSISTENT" not in codes:
                codes.append("INPUT_OUTPUT_SIGN_INCONSISTENT")
            out["impulse_implied_M"] = "NEGATIVE (no physical mass explains the impulse)"
        # passivity bound on M (dissipation >= 0)
        if sig("e_n"):
            r = nom["W_in"] / nom["e_n"]
            if nom["e_n"] > 0:
                out["passivity_M_bound"] = dict(M_max=r if nom["W_in"] > 0 else None,
                                                rule="W_in >= M e_n with e_n > 0 -> M <= W_in / e_n")
            elif nom["W_in"] < 0:
                out["passivity_M_bound"] = dict(M_min=r, rule="W_in >= M e_n with e_n < 0 -> M >= W_in / e_n")
        if rates and sig("D_imp") and nom["D_imp"] != 0:
            out["impulse_M_not_for_sim"] = nom["J"] / nom["D_imp"]
        if g_u:
            wo = g_u * nom["W_in"] - nom["e_n"] - nom["D_n"]
            wos = float(np.hypot(np.std([g_u * m["W_in"] - m["e_n"] - m["D_n"] for m in mc], ddof=1),
                                 max(abs(g_u * b["W_in"] - b["e_n"] - b["D_n"] - wo) for b in bw)))
            out["w_other_n"] = dict(estimate=wo, sd=wos, at_g_u=g_u)
            if abs(wo) > Z * wos:
                codes.append("POSSIBLE_UNMODELED_DRIVE" if wo < 0 else "UNMODELED_DISSIPATION")
        # sensitivity: time offset and lever / projection scale (sign never varied)
        sens = {}
        for tag, tff, scale in ((f"lag_+{lag_s}", tf + lag_s, 1.0), (f"lag_-{lag_s}", tf - lag_s, 1.0),
                                (f"lever_x{1 + lever_rel:g}", tf, 1 + lever_rel),
                                (f"lever_x{1 - lever_rel:g}", tf, 1 - lever_rel)):
            x = sample(q, Qf * scale, tff)
            sens[tag] = dict(W_in=x["W_in"], J=x["J"], sign_consistent=bool(
                not (x["W_in"] < -Z * sd["W_in"] and nom["e_n"] > Z * sd["e_n"])
                and not (rates and abs(x["J"]) > Z * sd["J"] and abs(nom["D_imp"]) > Z * sd["D_imp"]
                         and np.sign(x["J"]) != np.sign(nom["D_imp"]))))
        out["sensitivity"] = sens
        base_ok = "INPUT_OUTPUT_SIGN_INCONSISTENT" not in codes
        out["verdict_robust_to_lag_and_lever"] = all(v["sign_consistent"] == base_ok for v in sens.values())
    else:
        if sig("e_n") and nom["e_n"] > 0:
            gain_zero_g = None
            if zero_gravity is not None and not gravity.is_zero:
                z0 = window_audit(ep, zero_gravity, rates, None, n_mc=max(50, n_mc // 4), seed=seed + 1,
                                  zero_gravity=None)
                gain_zero_g = z0["nominal"]["e_n"] > Z * z0["sd"]["e_n"]
                out["zero_gravity_e_n"] = dict(estimate=z0["nominal"]["e_n"], sd=z0["sd"]["e_n"])
            codes.append("INCONSISTENT_GRAVITY_GEOMETRY" if gain_zero_g is False else "NONPASSIVE_UNMODELED_FORCE")
        if rates:
            res = nom["e_n"] + nom["D_n"]
            ress = float(np.hypot(np.std([m["e_n"] + m["D_n"] for m in mc], ddof=1),
                                  max(abs(b["e_n"] + b["D_n"] - res) for b in bw)))
            out["free_balance_residual_n"] = dict(estimate=res, sd=ress,
                                                  meaning="Delta E_n + D_n (0 for the fitted free model)")
    out["codes"] = codes
    out["verdict"] = next((c for c in ("INPUT_OUTPUT_SIGN_INCONSISTENT", "NONPASSIVE_UNMODELED_FORCE",
                                       "INCONSISTENT_GRAVITY_GEOMETRY", "POSSIBLE_UNMODELED_DRIVE",
                                       "UNMODELED_DISSIPATION", "MOTION_NOISE_DOMINATED",
                                       "SHORT_UNINFORMATIVE_WINDOW") if c in codes), "CONSISTENT")
    return out


def audit(episodes, gravity, rates=None, g_u=None, *, zero_gravity=None, cfg=None, excluded=None,
          pulse_audit=None) -> dict:
    cfg = dict(cfg or {})
    win = [window_audit(e, gravity, rates, g_u if e.forced else None, n_mc=int(cfg.get("energy_mc", 200)),
                        lag_s=float(cfg.get("energy_lag_s", 0.02)), lever_rel=float(cfg.get("energy_lever_rel", 0.1)),
                        zero_gravity=zero_gravity) for e in episodes]
    contra = [w["name"] for w in win if w["verdict"] == "INPUT_OUTPUT_SIGN_INCONSISTENT"]
    nonpassive = [w["name"] for w in win if w["verdict"] in ("NONPASSIVE_UNMODELED_FORCE",
                                                            "INCONSISTENT_GRAVITY_GEOMETRY")]
    return dict(schema="forceartgs.g9.io_work_energy_audit/v1", role="AUDIT_ONLY (not a loss term)",
                equation="int u qd dt = Delta(1/2 M qd^2 + U) + int c qd^2 + int Fc |qd| + W_other",
                windows=win, contradicting_windows=contra, nonpassive_windows=nonpassive,
                excluded_intervals=[dict(name=x.get("name"), code=x.get("code"), reason=x.get("reason"))
                                    for x in (excluded or [])],
                declared_pulse_audit=pulse_audit,
                note="every contradiction is reported with its window; the force sign is never flipped")


def summarize_pulse_audit(ia):
    """Compact view of validation.impulse_balance.audit (the fridge rows 229-232 regression)."""
    if not ia:
        return None
    dec = ia.get("decision") or {}
    return dict(primary=dec.get("primary"), reasons=dec.get("reasons"), decision=dec,
                note="declared pulse rows: impulse / momentum sign audit (validation.impulse_balance); no flip")


def pulse_input_validity(primary):
    if primary is None:
        return None
    p = str(primary).upper()
    if p == "IMPULSE_CONSISTENT":
        return "CONSISTENT"
    if any(k in p for k in ("CONTRADICT", "INCONSISTENT", "MISMATCH", "SIGN")):
        return "CONTRADICTED"
    return "UNRESOLVED"


def finite_or_none(x):
    return x if (x is not None and isinstance(x, (int, float)) and math.isfinite(x)) else None
