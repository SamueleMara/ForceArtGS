"""Two INDEPENDENT reconstructions of qdot, qddot from the native RGB-D q samples (never a single 2-point
gradient), each with propagated uncertainty and its regularisation recorded.

    local_poly : weighted local polynomial (degree 2) in a +/- h window, derivatives and covariance from the fit
    spline     : penalised smoothing spline (scipy make_smoothing_spline, GCV-selected lambda), uncertainty by Monte
                 Carlo re-fits with q noise (fixed lambda, fixed seed)
"""
from __future__ import annotations

import numpy as np


def local_poly(t, q, q_std, t_eval=None, h=0.12, degree=2, min_pts=5):
    t, q, s = (np.asarray(x, float) for x in (t, q, q_std))
    te = t if t_eval is None else np.asarray(t_eval, float)
    out = {k: np.full(len(te), np.nan) for k in ("q", "qd", "qdd", "q_sd", "qd_sd", "qdd_sd")}
    for i, tc in enumerate(te):
        hh = h
        m = np.abs(t - tc) <= hh
        while m.sum() < max(min_pts, degree + 2) and hh < 8 * h:
            hh *= 1.5
            m = np.abs(t - tc) <= hh
        if m.sum() < degree + 2:
            continue
        x = t[m] - tc
        w = 1.0 / s[m] ** 2 * (1 - (np.abs(x) / (hh * 1.0001)) ** 3) ** 3          # tricube x inverse variance
        A = np.vander(x, degree + 1, increasing=True)
        AtW = A.T * w
        C = np.linalg.pinv(AtW @ A)
        beta = C @ AtW @ q[m]
        # covariance with the measurement sigma (sandwich): C A^T W Sigma W A C, Sigma = diag(s^2)
        Cov = C @ (AtW * s[m] ** 2) @ AtW.T @ C
        out["q"][i], out["qd"][i] = beta[0], beta[1]
        out["q_sd"][i], out["qd_sd"][i] = np.sqrt(Cov[0, 0]), np.sqrt(Cov[1, 1])
        if degree >= 2:
            out["qdd"][i], out["qdd_sd"][i] = 2 * beta[2], 2 * np.sqrt(Cov[2, 2])
    out["method"] = dict(name="local_poly", degree=degree, half_window_s=h, weights="tricube x 1/q_std^2",
                         uncertainty="sandwich covariance of the local fit")
    return out


def spline(t, q, q_std, t_eval=None, n_mc=40, seed=0, lam=None):
    from scipy.interpolate import make_smoothing_spline
    t, q, s = (np.asarray(x, float) for x in (t, q, q_std))
    te = t if t_eval is None else np.asarray(t_eval, float)
    w = 1.0 / s ** 2
    w = w / w.mean()
    sp = make_smoothing_spline(t, q, w=w, lam=lam)
    lam_used = lam
    if lam_used is None:                      # recover the GCV lambda for the Monte Carlo refits
        lam_used = getattr(sp, "lam", None)
    d1, d2 = sp.derivative(1), sp.derivative(2)
    out = dict(q=sp(te), qd=d1(te), qdd=d2(te))
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_mc):
        qs = q + rng.standard_normal(len(q)) * s
        spk = make_smoothing_spline(t, qs, w=w, lam=lam_used)
        sims.append([spk(te), spk.derivative(1)(te), spk.derivative(2)(te)])
    sims = np.asarray(sims)
    out["q_sd"], out["qd_sd"], out["qdd_sd"] = sims.std(0, ddof=1)
    out["method"] = dict(name="spline", penalty="integrated squared 2nd derivative",
                         lambda_selection="GCV" if lam is None else "fixed", lam=lam_used, n_mc=n_mc, seed=seed,
                         uncertainty="Monte Carlo re-fits with q noise")
    return out


def reconstruct(t, q, q_std, t_eval=None):
    """Both reconstructions + their disagreement (a derivative-uncertainty diagnostic)."""
    a = local_poly(t, q, q_std, t_eval)
    b = spline(t, q, q_std, t_eval)
    dis = dict(qd_rms_diff=float(np.sqrt(np.nanmean((a["qd"] - b["qd"]) ** 2))),
               qdd_rms_diff=float(np.sqrt(np.nanmean((a["qdd"] - b["qdd"]) ** 2))))
    return dict(local_poly=a, spline=b, disagreement=dis)
