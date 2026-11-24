"""Normalized rollout optimizer (spec 6.4): fit (lam, gam) on FREE trajectories through the validated hybrid
integrator (normalized: M = 1), per-segment nuisance (q0, v0); then -- only with trusted forced segments -- beta on
the forced trajectories with the free rates as a starting point and an optional joint refinement. M never appears in
the free stage; no M = 0 boundary exists."""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import least_squares


def _init_state(seg):
    k = min(4, len(seg.t))
    A = np.vander(seg.t[:k] - seg.t[0], 2, increasing=True)
    b = np.linalg.lstsq(A, seg.q[:k], rcond=None)[0]
    return float(b[0]), float(b[1])


def fit(segments, nspec, *, sigma_floor=1e-3, starts=None, max_nfev=400, theta_fixed=None) -> dict:
    """Least squares on q-replay residuals (q_sim - q_obs) / sigma_eff; rates >= 0; nuisance (q0, v0) per segment."""
    t0 = time.time()
    names = nspec.names
    npar = len(names)
    segs = [s for s in segments if (s.is_free or nspec.forced)]
    if not segs:
        return dict(status="NOT_RUN", reason="no segments")
    from ..free_forced.normalized import a_scale
    A = a_scale(segs)
    x_states = []
    for s in segs:
        x_states += list(_init_state(s))

    def unpack(z):
        return z[:npar], z[npar:]

    def resid(z):
        th, st = unpack(z)
        out = []
        for k, s in enumerate(segs):
            q0, v0 = st[2 * k], st[2 * k + 1]
            try:
                r = nspec.simulate(th, s.t, s.t[0], q0, v0, effort=s.effort if (nspec.forced and not s.is_free)
                                   else None)
                out.append((r.q - s.q) / np.maximum(s.q_sigma, sigma_floor))
            except Exception:
                out.append(np.full(len(s.t), 1e6))
        return np.concatenate(out)

    if starts is None:
        lam0 = [0.1, 1.0, 5.0] if "lam" in names else [0.0]
        gam0 = [0.0, 0.3 * A, 3.0 * A] if "gam" in names else [0.0]
        starts = [nspec.pack(l, g, 1.0 if nspec.forced else None) for l in lam0 for g in gam0]
    lo = np.r_[np.zeros(npar), np.full(len(x_states), -np.inf)]
    best = None
    for th0 in starts:
        z0 = np.r_[np.asarray(th0, float), x_states]
        try:
            r = least_squares(resid, z0, bounds=(lo, np.inf), x_scale="jac", max_nfev=max_nfev)
        except Exception as e:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return dict(status="FAILED", reason="no start converged")
    th, st = unpack(best.x)
    J = best.jac
    dof = max(len(best.fun) - len(best.x), 1)
    chi2 = float(2 * best.cost)
    try:
        C = np.linalg.inv(J.T @ J) * max(1.0, chi2 / dof)
        Cth = C[:npar, :npar]
        sd = np.sqrt(np.clip(np.diag(Cth), 0, None))
        cov_status = "ok"
    except np.linalg.LinAlgError:
        Cth, sd, cov_status = None, np.full(npar, np.nan), "singular"
    at_bound = [names[i] for i in range(npar) if th[i] <= 1e-10]
    return dict(schema="forceartgs.g9.free_rollout_fit/v1", status="ok" if best.success else "not_converged",
                estimate=dict(zip(names, th.tolist())), sd=dict(zip(names, sd.tolist())),
                cov=None if Cth is None else Cth.tolist(), cov_status=cov_status,
                uncertainty_label="Gauss-Newton covariance (chi2-scaled), nuisance states marginalised",
                states={s.name: dict(q0=float(st[2 * k]), v0=float(st[2 * k + 1])) for k, s in enumerate(segs)},
                chi2_per_dof=chi2 / dof, n_obs=int(len(best.fun)), at_bound=at_bound, n_starts=len(starts),
                nfev=int(best.nfev), sigma_floor=sigma_floor, runtime_s=time.time() - t0,
                note="nuisance (q0, v0) fitted on the same segment: this is a fit, not the prefix-only validation")


def fit_staged(free_segments, forced_segments, nspec, **kw) -> dict:
    """Free stage (lam, gam) -> forced stage (beta with rates started from the free fit) -> optional joint."""
    out = dict(free=fit(free_segments, nspec.free(), **kw) if free_segments else dict(status="NOT_RUN",
                                                                                     reason="no free segments"))
    if not forced_segments:
        out["forced"] = dict(status="NOT_RUN", reason="no trusted forced segment")
        return out
    fs = nspec.with_forced()
    e = (out["free"].get("estimate") or {})
    starts = [fs.pack(e.get("lam", 0.5), e.get("gam", 0.0), b) for b in (0.1, 1.0, 10.0)]
    out["forced"] = fit(list(free_segments) + list(forced_segments), fs, starts=starts, **kw)
    return out
