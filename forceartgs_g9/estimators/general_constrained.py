"""general_rollout_fit: constrained maximum-likelihood fit of the general hybrid model (G9 spec 4.2).

Residuals are (q_sim(t_obs) - q_obs) / q_std on the RAW observation timestamps, with q_sim from the complete
stick/slip rollout (``simulate_general``) driven by the measured generalized effort. Physical parameters are SHARED
by all interactions of the same object/joint; each interaction has its own nuisance initial state (q0, v0) unless a
DECLARED initial state is given. No per-time latent force or contact wrench exists in the model, so nothing can
absorb the residual except the declared terms. Bounds keep M > 0, c, Fc, dFs >= 0 (so Fs >= Fc). Robust loss
(Huber, in sigma units), deterministic multi-start from the integral-LS initializer.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import least_squares

from ..dynamics.general_1dof import ModelConfigError, simulate_general
from .model_spec import G9Interaction, ModelSpec

NUISANCE = ("q0", "v0")


def _layout(spec: ModelSpec, interactions):
    names = list(spec.names)
    nuis = []
    for it in interactions:
        if it.initial_state is None:
            nuis += [(it.name, "q0"), (it.name, "v0")]
    return names, nuis


def simulate_interaction(spec: ModelSpec, theta_phys, it: G9Interaction, q0=None, v0=None, max_step=2e-3,
                         t_query=None):
    p = spec.sim(theta_phys)
    if it.initial_state is not None:
        t0, q0, v0 = float(it.initial_state["t_s"]), float(it.initial_state["q"]), float(it.initial_state["qdot"])
    else:
        t0 = float(it.t_obs[0])
    tq = it.t_obs if t_query is None else t_query
    return simulate_general(p, it.effort, tq, t0=t0, q0=q0, v0=v0, max_step=max_step)


class RolloutProblem:
    def __init__(self, spec: ModelSpec, interactions, max_step=2e-3, robust="huber"):
        self.spec, self.its, self.max_step, self.robust = spec, list(interactions), max_step, robust
        self.names, self.nuis = _layout(spec, interactions)
        self.n_phys = len(self.names)

    def split(self, x):
        th = x[:self.n_phys]
        st = {}
        k = self.n_phys
        for it in self.its:
            if it.initial_state is None:
                st[it.name] = (float(x[k]), float(x[k + 1]))
                k += 2
        return th, st

    def residuals(self, x, effort_override=None):
        th, st = self.split(x)
        out = []
        for i, it in enumerate(self.its):
            q0, v0 = st.get(it.name, (None, None))
            itx = it if effort_override is None else _with_effort(it, effort_override[i])
            try:
                r = simulate_interaction(self.spec, th, itx, q0, v0, self.max_step)
                out.append((r.q - it.q_obs) / it.q_std)
            except ModelConfigError:
                out.append(np.full(len(it.t_obs), 1e6))
        return np.concatenate(out)

    def bounds(self):
        """Physical bounds; the nuisance initial state is bounded by the data (|v0| <= 5 x max observed |q'|,
        q0 within the observed range +/- 5 sigma): a numerical safeguard against degenerate M -> 0 solutions in
        which v0 becomes unconstrained, not a prior on the physical parameters."""
        lo = list(self.spec.lower_bounds())
        hi = [np.inf] * self.n_phys
        for it in self.its:
            if it.initial_state is None:
                vmax = 5.0 * float(np.max(np.abs(np.gradient(it.q_obs, it.t_obs)))) + 1e-9
                s5 = 5.0 * float(np.median(it.q_std))
                lo += [float(it.q_obs.min()) - s5, -vmax]
                hi += [float(it.q_obs.max()) + s5, vmax]
        return np.array(lo), np.array(hi)

    def x0(self, theta_phys):
        x = list(theta_phys)
        for it in self.its:
            if it.initial_state is None:
                v = np.gradient(it.q_obs[:5], it.t_obs[:5])[0]
                x += [float(it.q_obs[0]), float(v)]
        return np.asarray(x, float)


def _with_effort(it, effort):
    return G9Interaction(it.name, it.joint_type, it.t_obs, it.q_obs, it.q_std, effort, it.initial_state, it.meta)


def general_rollout_fit(interactions, spec: ModelSpec, theta0, *, n_starts=4, seed=0, max_step=2e-3,
                        robust="huber", max_nfev=300, scale_hint=None) -> dict:
    """Fit; returns estimates, covariance (Gauss-Newton at the optimum), per-start record and the problem."""
    t0 = time.time()
    P = RolloutProblem(spec, interactions, max_step, robust)
    lo, hi = P.bounds()
    rng = np.random.default_rng(seed)
    base = np.asarray(theta0, float)
    starts = [base]
    for _ in range(max(n_starts - 1, 0)):
        pert = base.copy()
        for j, n in enumerate(P.names):
            if n in ("alpha", "M", "c", "Fc", "dFs", "Fs"):
                mag = abs(pert[j]) if pert[j] != 0 else (scale_hint or {}).get(n, 1.0)
                pert[j] = max(mag * math.exp(rng.normal(0, 0.7)), 1e-9 if n in ("alpha", "M") else 0.0)
            else:
                scale = abs(pert[j]) if pert[j] != 0 else (scale_hint or {}).get("effort", 1.0) * 0.1
                pert[j] = pert[j] + rng.normal(0, 0.5) * scale
        starts.append(pert)
    runs, best = [], None
    js0 = [j for j, n in enumerate(P.names) if n in ("dFs", "Fs")]
    for k, s in enumerate(starts):
        x0 = P.x0(s)
        fin = np.isfinite(lo)
        x0[fin] = np.maximum(x0[fin], lo[fin] + 1e-10)
        fh = np.isfinite(hi)
        x0[fh] = np.minimum(x0[fh], hi[fh] - 1e-10)
        if js0:                      # place the breakaway level first: smooth terms would otherwise bend around it
            x0[js0[0]] = max(profile_scan(P, x0, js0[0], scale_hint, n=40)["best"], lo[js0[0]] + 1e-10)
        try:
            r = least_squares(P.residuals, x0, bounds=(lo, hi), loss="huber" if robust == "huber" else "linear",
                              f_scale=1.0, x_scale="jac", max_nfev=max_nfev, diff_step=1e-5)
        except (ValueError, FloatingPointError) as e:
            runs.append(dict(start=k, status="numerical_failure", error=str(e)))
            continue
        cost = float(0.5 * np.sum(P.residuals(r.x) ** 2))
        runs.append(dict(start=k, status=r.status, cost=cost, nfev=int(r.nfev), message=r.message))
        if best is None or cost < best[0]:
            best = (cost, r)
    if best is None:
        return dict(status="numerical_failure", runs=runs, problem=P)
    cost, r = best
    profile = None
    js = [j for j, n in enumerate(P.names) if n in ("dFs", "Fs")]
    if js:
        # the likelihood is piecewise CONSTANT in the breakaway level (it only changes which ZOH sample crosses it):
        # gradients are zero, so Fs is found by a 1-D profile scan, alternating with the smooth parameters
        j = js[0]
        x = r.x.copy()
        for _ in range(2):
            profile = profile_scan(P, x, j, scale_hint)
            x[j] = profile["best"]
            lo2, hi2 = lo.copy(), hi.copy()
            lo2[j], hi2[j] = x[j] - 1e-12 - 1e-9 * abs(x[j]), x[j] + 1e-12 + 1e-9 * abs(x[j])
            x0 = np.clip(x, lo2, hi2)
            r2 = least_squares(P.residuals, x0, bounds=(lo2, hi2), loss="huber" if robust == "huber" else "linear",
                               f_scale=1.0, x_scale="jac", max_nfev=max_nfev, diff_step=1e-5)
            x = r2.x
        profile = profile_scan(P, x, j, scale_hint)
        x[j] = profile["best"]
        r = r2
        r.x = x
        runs.append(dict(start="static_profile", cost=float(0.5 * np.sum(P.residuals(x) ** 2)),
                         interval=profile["interval95"]))
    res = P.residuals(r.x)
    J = r.jac
    n, p = len(res), len(r.x)
    s2 = float(res @ res) / max(n - p, 1)
    keep = [k for k in range(p) if not (js and k == js[0])]     # the profiled breakaway column is identically 0
    Jk = J[:, keep]
    cov = np.full((p, p), np.nan)
    try:
        ck = s2 * np.linalg.inv(Jk.T @ Jk)
        cov_status = "gauss_newton"
    except np.linalg.LinAlgError:
        ck = s2 * np.linalg.pinv(Jk.T @ Jk)
        cov_status = "pseudo_inverse_singular_fisher"
    cov[np.ix_(keep, keep)] = ck
    if js:
        cov[js[0], js[0]] = 0.0                                   # replaced by the profile interval below
    th, st = P.split(r.x)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    est = {nm: dict(value=float(th[j]), std=float(se[j]), at_lower_bound=bool(np.isfinite(lo[j]) and
                                                                            th[j] <= lo[j] + 1e-9 * max(1, abs(lo[j]))))
           for j, nm in enumerate(P.names)}
    if profile is not None:
        nm = P.names[js[0]]
        lo95, hi95 = profile["interval95"]
        est[nm].update(std=float((hi95 - lo95) / (2 * 1.96)) if np.isfinite(hi95) else math.inf,
                       interval95=[lo95, hi95], uncertainty_method="profile likelihood (delta chi2 <= 3.84); "
                       "the likelihood is piecewise constant in the breakaway level", profile=profile)
    rms = float(np.sqrt(np.mean(res ** 2)))
    return dict(status="ok" if r.success else "not_converged", method="general_rollout_fit", x=r.x.tolist(),
                estimates=est, nuisance={k: dict(q0=v[0], v0=v[1]) for k, v in st.items()},
                physical=spec.physical(th), cost=cost, residual_rms_sigma=rms, reduced_chi2=s2, n_obs=n,
                n_params=p, covariance=cov.tolist(), covariance_status=cov_status, jacobian=J,
                runs=runs, max_step=max_step, robust_loss=robust, runtime_s=time.time() - t0, problem=P,
                converged=bool(r.success), optimality=float(r.optimality))


def profile_scan(P: RolloutProblem, x, j, scale_hint=None, n=80) -> dict:
    """chi2(param_j) with every other parameter held at x: grid over [0, effort scale], refined around the
    minimum; 95% profile interval = {chi2 <= min + 3.84} (unbounded above if the grid edge is inside)."""
    eff = max(float(np.max(np.abs(np.concatenate([it.effort.levels for it in P.its])))), 1e-9)
    grids = [np.linspace(0.0, eff, n)]
    xs = []
    for g in grids:
        for v in g:
            y = x.copy()
            y[j] = v
            xs.append((float(v), float(np.sum(P.residuals(y) ** 2))))
    xs.sort()
    vals = np.array([a for a, _ in xs])
    chi = np.array([b for _, b in xs])
    k = int(np.argmin(chi))
    lo_r, hi_r = vals[max(k - 2, 0)], vals[min(k + 2, len(vals) - 1)]
    for v in np.linspace(lo_r, hi_r, 41):
        y = x.copy()
        y[j] = v
        xs.append((float(v), float(np.sum(P.residuals(y) ** 2))))
    xs.sort()
    vals = np.array([a for a, _ in xs])
    chi = np.array([b for _, b in xs])
    cmin = float(chi.min())
    inside = chi <= cmin + 3.84
    best_set = vals[chi <= cmin + 1e-9 * max(cmin, 1)]
    lo95 = float(vals[inside].min())
    hi95 = float(vals[inside].max()) if not inside[-1] else math.inf
    return dict(best=float(np.median(best_set)), chi2_min=cmin, interval95=[lo95, hi95],
                flat_minimum_width=float(best_set.max() - best_set.min()), n_evaluations=len(vals),
                grid_max=float(vals.max()), unbounded_above=bool(inside[-1]))


def step_convergence(spec, theta, it: G9Interaction, steps=(4e-3, 2e-3, 1e-3, 5e-4)) -> dict:
    """q differences between successive integrator step sizes (numerical convergence of the rollout)."""
    qs = [simulate_interaction(spec, theta, it, it.q_obs[0], 0.0, h).q for h in steps]
    diffs = [float(np.max(np.abs(qs[i] - qs[-1]))) for i in range(len(steps) - 1)]
    return dict(steps=list(steps), max_abs_diff_to_finest=diffs,
                converged=bool(diffs[-1] <= max(1e-9, 0.1 * float(np.median(it.q_std)))))
