"""ANALYTIC_CONVOLUTION_1D: exact output-error baseline for  M q'' + c q' = Q  from (q0, v0 = 0).

The model is first order in v. With lambda = c / M and beta = 1 / M,

    q(t) - q0 = beta h_lambda(t),     h_lambda(t) = int_{t0}^{t} (1 - exp(-lambda (t - s))) / lambda  Q(s) ds

(limit kernel t - s for lambda -> 0). For the forward-ZOH native effort h is EXACT per constant interval:

    int_u^w (1 - e^{-lambda x}) / lambda dx = F(w) - F(u),  F(x) = (lambda x + expm1(-lambda x)) / lambda^2
                                                                 = x^2/2 - lambda x^3/6 + lambda^2 x^4/24 - ...  (small)

For each lambda on a bounded profile grid, beta >= 0 is a ONE-dimensional robust fit of the SAME raw-q position loss as
the PhysX objective (mean Huber(delta) of (q_model - q_obs) / sqrt(sigma_q^2 + sigma_num^2)); the profile, its valley and
the near-best family are kept. Never estimates q0 or v0. A fast PhysX warm start and a diagnostic of the minimal
dynamics -- not the final simulator and not in the final PhysX scoreboard as such.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import minimize_scalar

from .rest_to_rest import RestToRestEpisode


def _F(x, lam):
    x = np.asarray(x, float)
    if lam == 0.0:
        return 0.5 * x * x
    z = lam * x
    small = np.abs(z) < 1e-3
    out = np.empty_like(x)
    zs, xs = z[small], x[small]
    out[small] = xs * xs * (0.5 - zs / 6.0 + zs * zs / 24.0 - zs ** 3 / 120.0)
    zb = z[~small]
    out[~small] = (zb + np.expm1(-zb)) / (lam * lam)
    return out


def h_lambda(force_t, levels, t0, t_eval, lam) -> np.ndarray:
    """Exact h_lambda(t_i) for a forward-ZOH effort (rows hold until the next row; the last row holds on)."""
    ft = np.asarray(force_t, float)
    lev = np.asarray(levels, float)
    a = np.maximum(ft, t0)
    b = np.maximum(np.r_[ft[1:], np.inf], t0)
    t = np.asarray(t_eval, float)[:, None]
    e = np.minimum(b[None, :], t)
    act = (t > a[None, :]) & (e > a[None, :])
    w = np.where(act, t - a[None, :], 0.0)                  # x at the interval start
    u = np.where(act, t - e, 0.0)                           # x at the interval end
    contrib = np.where(act, _F(w, lam) - _F(u, lam), 0.0)
    return contrib @ lev


def _huber(z, d):
    a = np.abs(z)
    return np.where(a <= d, 0.5 * z * z, d * (a - 0.5 * d))


def fit_beta(h, dq, sig, delta, iters=50) -> tuple:
    """min_{beta >= 0} mean Huber((beta h - dq) / sig): IRLS from the weighted LS solution (convex in beta)."""
    w = 1.0 / sig ** 2
    hh = float(np.sum(w * h * h))
    if hh <= 0:
        return 0.0, float(np.mean(_huber(dq / sig, delta)))
    b = max(float(np.sum(w * h * dq)) / hh, 0.0)
    for _ in range(iters):
        z = (b * h - dq) / sig
        ww = w * np.minimum(1.0, delta / np.maximum(np.abs(z), 1e-300))
        den = float(np.sum(ww * h * h))
        bn = max(float(np.sum(ww * h * dq)) / den, 0.0) if den > 0 else 0.0
        if abs(bn - b) <= 1e-12 * max(abs(b), 1e-300):
            b = bn
            break
        b = bn
    return b, float(np.mean(_huber((b * h - dq) / sig, delta)))


def simulate_two_term(force_t, levels, t0, q0, M, c, t_eval) -> dict:
    """EXACT solution of M q'' + c q' = Q from (q0, v0 = 0) at t0 under a forward-ZOH effort: q and v at t_eval."""
    t = np.asarray(t_eval, float)
    lam = float(c) / float(M)
    h = h_lambda(force_t, levels, t0, t, lam)
    ft = np.asarray(force_t, float)
    a = np.maximum(ft, t0)
    b = np.maximum(np.r_[ft[1:], np.inf], t0)
    tt = t[:, None]
    e = np.minimum(b[None, :], tt)
    act = (tt > a[None, :]) & (e > a[None, :])
    if lam == 0.0:
        g = np.where(act, e - a[None, :], 0.0)
    else:
        g = np.where(act, (np.exp(-lam * (tt - e)) - np.exp(-lam * (tt - a[None, :]))) / lam, 0.0)
    return dict(t=t, q=float(q0) + h / float(M), v=(g @ np.asarray(levels, float)) / float(M))


def replay(ep: RestToRestEpisode, M, c, t_eval=None) -> dict:
    """Exact open-loop trajectory of the two-term model from (q0, v0 = 0) under the episode's native ZOH effort."""
    return simulate_two_term(ep.force_t_s, ep.force_Q, ep.t0, ep.q0, M, c, ep.t_q_s if t_eval is None else t_eval)


def convolution_1d(ep: RestToRestEpisode, *, lam_max=50.0, n_grid=61, huber_delta=2.0, numeric_floor_q=3e-4,
                   family_tol_sigma=0.5) -> dict:
    """Profile lambda in [0, lam_max] (0 + geometric grid), beta fitted per lambda, then a bounded scalar refinement in
    log lambda around the best grid point. Returns (M, c) = (1 / beta, lambda / beta), the profile and the family."""
    t0_ = time.time()
    t = ep.t_q_s
    dq = ep.q_obs - ep.q0
    sig = np.hypot(ep.q_sigma, numeric_floor_q)
    lam_min = max(lam_max * 1e-5, 1e-4)
    grid = np.r_[0.0, np.geomspace(lam_min, lam_max, max(int(n_grid) - 1, 2))]
    prof = []
    for lam in grid:
        h = h_lambda(ep.force_t_s, ep.force_Q, ep.t0, t, float(lam))
        b, L = fit_beta(h, dq, sig, huber_delta)
        r = (b * h - dq)
        prof.append(dict(lam=float(lam), beta=b, objective=L, rmse=float(np.sqrt(np.mean(r ** 2))),
                         rmse_over_sigma=float(np.sqrt(np.mean((r / sig) ** 2)))))
    feas = [p for p in prof if p["beta"] > 0]
    if not feas:
        return dict(method="ANALYTIC_CONVOLUTION_1D", status="NO_POSITIVE_INERTIA",
                    reason="beta = 1/M is 0 at every lambda: the recorded effort cannot drive the observed displacement "
                           "in the two-term model (sign / input inconsistency)", profile=prof,
                    runtime_s=time.time() - t0_)
    k = min(range(len(prof)), key=lambda i: prof[i]["objective"] if prof[i]["beta"] > 0 else math.inf)
    best = dict(prof[k])
    lo_i, hi_i = max(k - 1, 0), min(k + 1, len(grid) - 1)
    if grid[hi_i] > 0 and hi_i > lo_i:
        llo = math.log(max(grid[lo_i], lam_min))
        lhi = math.log(grid[hi_i])

        def f(x):
            lam = math.exp(x)
            h = h_lambda(ep.force_t_s, ep.force_Q, ep.t0, t, lam)
            b, L = fit_beta(h, dq, sig, huber_delta)
            return L if b > 0 else 1e300
        r = minimize_scalar(f, bounds=(llo, lhi), method="bounded", options=dict(xatol=1e-4))
        if r.fun < best["objective"]:
            lam = math.exp(r.x)
            h = h_lambda(ep.force_t_s, ep.force_Q, ep.t0, t, lam)
            b, L = fit_beta(h, dq, sig, huber_delta)
            rr = b * h - dq
            best = dict(lam=lam, beta=b, objective=L, rmse=float(np.sqrt(np.mean(rr ** 2))),
                        rmse_over_sigma=float(np.sqrt(np.mean((rr / sig) ** 2))))
    M = 1.0 / best["beta"]
    c = best["lam"] / best["beta"]
    fam = [p for p in feas if p["rmse_over_sigma"] <= best["rmse_over_sigma"] + family_tol_sigma]
    lam_rng = [min(p["lam"] for p in fam), max(p["lam"] for p in fam)] if fam else [best["lam"]] * 2
    flat = bool(fam and (lam_rng[1] / max(lam_rng[0], lam_min) > 10 or lam_rng[0] == 0.0 or
                         lam_rng[1] >= grid[-1]))
    rp = replay(ep, M, c)
    return dict(method="ANALYTIC_CONVOLUTION_1D", status="OK", M=M, c=c, lam=best["lam"], beta=best["beta"],
                objective=best["objective"], rmse=best["rmse"], rmse_over_sigma=best["rmse_over_sigma"],
                v_T=float(rp["v"][-1]), q_T_error=float(rp["q"][-1] - ep.qT), profile=prof,
                valley=dict(rule=f"profile points with rmse/sigma <= best + {family_tol_sigma} (engineering tolerance)",
                            lam_range=lam_rng, M_range=[min(1 / p["beta"] for p in fam), max(1 / p["beta"] for p in fam)]
                            if fam else None, c_range=[min(p["lam"] / p["beta"] for p in fam),
                                                       max(p["lam"] / p["beta"] for p in fam)] if fam else None,
                            flat_lambda_valley=flat, n_members=len(fam)),
                at_lambda_bound=bool(best["lam"] >= grid[-1] * (1 - 1e-9) or best["lam"] == 0.0),
                grid=dict(lam_max=float(lam_max), n=len(grid), lam_min_positive=float(lam_min)),
                loss=f"mean Huber(delta={huber_delta}) of (q_model - q_obs) / sqrt(sigma_q^2 + {numeric_floor_q}^2): "
                     "the same raw-q loss as the PhysX objective", runtime_s=time.time() - t0_,
                note="exact ZOH convolution of the minimal model from (q0, 0); never estimates q0 or v0")
