"""Free-motion forward validation (spec section 4.2): roll the NORMALIZED hybrid ODE out from (q0, v0) estimated on
the initial PREFIX only, compare with the held-out remainder of the segment.

Metrics in physical units (rad or m), range-normalised, sigma-normalised with a TRANSPARENT floor (sigma_eff =
max(q_sigma, floor); the measured sigma is reported alongside), final drift, and stop-time error. A null model with
zero dissipation (same gravity) is replayed from the same state: a fitted model that does not beat it does not
support the fitted dissipation. A Monte Carlo band over (q0, v0) [+ rate covariance] gives the coverage of the
held-out samples. The tail is NOT an independent episode -- it is a within-record predictive check.
"""
from __future__ import annotations

import math

import numpy as np

ADEQ_RANGE = 0.05      # held-out RMSE / observed range below which the replay is adequate ...
ADEQ_SIGMA = 3.0       # ... or held-out RMSE / sigma_eff below this


def prefix_state(seg, n_prefix):
    """(t_c, q, v, cov) at the last prefix sample from a weighted quadratic (linear if < 4 samples)."""
    t, q, s = seg.t[:n_prefix], seg.q[:n_prefix], seg.q_sigma[:n_prefix]
    tc = float(t[-1])
    deg = 2 if n_prefix >= 4 else 1
    x = t - tc
    A = np.vander(x, deg + 1, increasing=True)
    w = 1.0 / s ** 2
    C = np.linalg.inv((A.T * w) @ A)
    b = C @ (A.T * w) @ q
    r = q - A @ b
    chi2 = float(np.sum(w * r ** 2) / max(n_prefix - deg - 1, 1))
    C = C * max(1.0, chi2)
    return tc, float(b[0]), float(b[1]), C[:2, :2], dict(n_prefix=n_prefix, degree=deg, chi2_dof=chi2,
                                                         t_prefix=[float(t[0]), tc])


def _stop_time(t, q, v_tol):
    v = np.gradient(q, t)
    k = np.flatnonzero(np.abs(v) < v_tol)
    return float(t[k[0]]) if len(k) else None


def segment_replay(seg, nspec, theta, *, prefix_frac=0.3, min_prefix=4, sigma_floor=1e-3, cov=None, n_mc=60,
                   seed=0, effort=None):
    n = len(seg.t)
    npf = int(min(max(min_prefix, round(prefix_frac * n)), n - 2))
    if npf < 3:
        return dict(seg=seg.name, status="TOO_SHORT", n=n)
    tc, q0, v0, C0, pinfo = prefix_state(seg, npf)
    th, tq, sq = seg.t[npf:], seg.q[npf:], seg.q_sigma[npf:]
    try:
        r = nspec.simulate(theta, th, tc, q0, v0, effort=effort)
    except Exception as e:
        return dict(seg=seg.name, status="NOT_SIMULABLE", reason=f"{type(e).__name__}: {e}")
    null = nspec.simulate(nspec.pack(0.0, 0.0, None if not nspec.forced else nspec.unpack(theta)["beta"]),
                          th, tc, q0, v0, effort=effort)
    rng_q = float(np.ptp(seg.q)) or 1e-12
    se = np.maximum(sq, sigma_floor)

    def met(pred):
        e = pred - tq
        return dict(rmse=float(np.sqrt(np.mean(e ** 2))), rmse_over_range=float(np.sqrt(np.mean(e ** 2)) / rng_q),
                    rmse_over_sigma_eff=float(np.sqrt(np.mean((e / se) ** 2))),
                    rmse_over_measured_sigma=float(np.sqrt(np.mean((e / sq) ** 2))), final_drift=float(e[-1]),
                    max_abs=float(np.max(np.abs(e))))
    m, m0 = met(r.q), met(null.q)
    # Monte Carlo band over the initial state (+ rates if a covariance is given)
    rng = np.random.default_rng(seed)
    sims = []
    L0 = np.linalg.cholesky(C0 + 1e-18 * np.eye(2))
    Lt = None
    if cov is not None:
        Cv = np.asarray(cov, float)
        try:
            Lt = np.linalg.cholesky(Cv + 1e-18 * np.eye(len(Cv)))
        except np.linalg.LinAlgError:
            Lt = None
    for _ in range(n_mc):
        z0 = L0 @ rng.standard_normal(2)
        thk = np.asarray(theta, float).copy()
        if Lt is not None:
            thk = np.maximum(thk + Lt @ rng.standard_normal(len(thk)), 0.0)
        try:
            sims.append(nspec.simulate(thk, th, tc, q0 + z0[0], v0 + z0[1], effort=effort).q)
        except Exception:
            continue
    band = None
    if len(sims) >= 10:
        S = np.asarray(sims)
        lo, hi = np.percentile(S, 2.5, 0), np.percentile(S, 97.5, 0)
        lo, hi = lo - 2 * sq, hi + 2 * sq
        band = dict(coverage95=float(np.mean((tq >= lo) & (tq <= hi))), lo=lo.tolist(), hi=hi.tolist(),
                    n_mc=len(sims), includes="initial-state covariance" + (" + rate covariance" if Lt is not None
                                                                            else ""))
    vt = 0.02 * max(abs(v0), 1e-6)
    stop_obs, stop_pred = _stop_time(seg.t, seg.q, vt), (float(th[np.flatnonzero(r.stuck)[0]]) if r.stuck.any()
                                                         else None)
    adequate = bool(m["rmse_over_range"] <= ADEQ_RANGE or m["rmse_over_sigma_eff"] <= ADEQ_SIGMA)
    return dict(seg=seg.name, label=seg.label, status="OK", prefix=pinfo, state0=dict(t=tc, q=q0, v=v0, cov=C0.tolist()),
                t_test=th.tolist(), q_test=tq.tolist(), q_pred=r.q.tolist(), v_pred=r.qdot.tolist(),
                q_null=null.q.tolist(), metrics=m, null_metrics=m0,
                beats_null=bool(m["rmse"] < m0["rmse"]), band=band, stop_time_observed=stop_obs,
                stop_time_predicted=stop_pred, sigma_floor=sigma_floor, adequate=adequate,
                adequacy_rule=f"rmse/range <= {ADEQ_RANGE} or rmse/sigma_eff <= {ADEQ_SIGMA}",
                unit="rad" if seg.joint_type == "revolute" else "m",
                note="within-record predictive check (NOT an independent episode)")


def replay(segments, nspec, theta, **kw) -> dict:
    per = [segment_replay(s, nspec, theta, effort=s.effort if (nspec.forced and not s.is_free) else None, **kw)
           for s in segments]
    ok = [p for p in per if p.get("status") == "OK"]
    if not ok:
        return dict(status="NOT_EVALUATED", per_segment=per)
    agg = dict(rmse=float(np.sqrt(np.mean([p["metrics"]["rmse"] ** 2 for p in ok]))),
               rmse_over_range=float(np.mean([p["metrics"]["rmse_over_range"] for p in ok])),
               rmse_over_sigma_eff=float(np.mean([p["metrics"]["rmse_over_sigma_eff"] for p in ok])),
               null_rmse=float(np.sqrt(np.mean([p["null_metrics"]["rmse"] ** 2 for p in ok]))),
               beats_null=all(p["beats_null"] for p in ok), adequate=all(p["adequate"] for p in ok),
               coverage95=float(np.mean([p["band"]["coverage95"] for p in ok if p.get("band")]))
               if any(p.get("band") for p in ok) else None)
    return dict(status="OK", aggregate=agg, per_segment=per)


def heldout_split_fit(fit_fn, segments, frac=0.6):
    """Rates fitted on the first ``frac`` of every segment; returns (theta, head_segments, tail_segments)."""
    heads, tails = [], []
    for s in segments:
        h, t = s.split(frac)
        heads.append(h)
        if t is not None:
            tails.append(t)
    return fit_fn(heads), heads, tails
