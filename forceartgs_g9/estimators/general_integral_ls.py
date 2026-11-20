"""general_integral_ls: weak (double-integral) least-squares initializer for the general model (G9 spec 4.2).

On every MOVING segment [t_a, t] (|q'| > v_eps; dry friction is set-valued at rest, so sticking intervals are not
used) the equation of motion integrated twice from t_a needs no differentiation of noisy q:

    M (q - q_a) - (M v_a)(t - t_a) + c int(q - q_a) + Fc iint s + iint G(q) + b (t - t_a)^2 / 2 + e_a = iint Q

with s = sign(q') on the segment (from a smoothed derivative), iint Q EXACT for the forward-ZOH effort, e_a a
per-segment offset absorbing the noise of q_a, and (M v_a) a per-segment nuisance. Geometry-coupled: the alpha
column is M1 (q - q_a) + iint G_1(q) (inertia AND gravity share the one density scale).
Static friction Fs is not linear in the moving-segment equations: it is not estimated here (rollout fit only).
Returns estimates, column-scaled singular values, rank, parameter correlations and physical-sign consistency.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter

from .model_spec import G9Interaction, ModelSpec


def zoh_double_integral(effort, t_a: float, t) -> tuple:
    """(int_{t_a}^t Q, iint Q) exactly for a forward-ZOH effort, at sorted times t >= t_a."""
    tf, lev = np.asarray(effort.t_s, float), np.asarray(effort.levels, float)
    t = np.asarray(t, float)
    knots = np.r_[t_a, tf[(tf > t_a) & (tf < t[-1])], np.inf]
    I1 = np.empty(len(t))
    I2 = np.empty(len(t))
    j, a1, a2, now = 0, 0.0, 0.0, t_a
    for i, tt in enumerate(t):
        while knots[j + 1] <= tt:
            h = knots[j + 1] - now
            Q = float(effort.at(now))
            a2 += a1 * h + 0.5 * Q * h * h
            a1 += Q * h
            now = knots[j + 1]
            j += 1
        h = tt - now
        Q = float(effort.at(now))
        I1[i] = a1 + Q * h
        I2[i] = a2 + a1 * h + 0.5 * Q * h * h
    return I1, I2


def velocity_estimate(t, q, window=7, poly=2):
    """Smoothed derivative (Savitzky-Golay on a locally uniform grid; falls back to gradient)."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    w = min(window if window % 2 else window + 1, len(q) - (1 - len(q) % 2))
    if w < poly + 2:
        return np.gradient(q, t)
    dt = float(np.median(np.diff(t)))
    if np.ptp(np.diff(t)) > 0.2 * dt:
        return np.gradient(q, t)
    return savgol_filter(q, w, poly, deriv=1, delta=dt)


def moving_segments(t, q, q_std, v_eps=None, min_len=5):
    """Contiguous runs with |v| > v_eps. v_eps (computational default) = 4 x the derivative noise implied by q_std
    through the smoothing window, documented in the result."""
    v = velocity_estimate(t, q)
    dt = float(np.median(np.diff(t)))
    if v_eps is None:
        v_eps = 4.0 * float(np.median(q_std)) / (dt * 3.0)
    mov = np.abs(v) > v_eps
    segs, start = [], None
    for i, m in enumerate(list(mov) + [False]):
        if m and start is None:
            start = i
        if not m and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    return segs, v, float(v_eps)


def general_integral_ls(interactions, spec: ModelSpec, v_eps=None) -> dict:
    t_ = spec.terms
    names = [n for n in spec.names if n not in ("dFs", "Fs")]          # Fs is not linear here
    rows, rhs, seg_info = [], [], []
    nuis_cols = []
    g_unit = (spec.mass_model.gravity_unit(spec.gravity.g_world) if (t_.gravity and spec.parameter_source ==
                                                                       "geometry_coupled") else None)
    for it in interactions:
        segs, v, eps = moving_segments(it.t_obs, it.q_obs, it.q_std, v_eps)
        for (a, z) in segs:
            tt, qq = it.t_obs[a:z], it.q_obs[a:z]
            T = tt - tt[0]
            dq = qq - qq[0]
            _, J = zoh_double_integral(it.effort, tt[0], tt)
            cols = {}
            if t_.inertia:
                if spec.parameter_source == "geometry_coupled":
                    col = spec.mass_model.M1 * dq
                    if g_unit is not None:
                        col = col + cumulative_trapezoid(cumulative_trapezoid(g_unit.G(qq), tt, initial=0), tt,
                                                         initial=0)
                    cols["alpha"] = col
                    J = J - spec.mass_model.reflected() * dq            # declared reflected inertia is known
                else:
                    cols["M"] = dq
                    if spec.parameter_source == "joint_space_gravity_coupled" and t_.gravity and \
                            spec.joint_type == "prismatic":
                        ga = float(np.asarray(spec.gravity.g_world, float) @ (np.asarray(spec.joint_axis, float) /
                                                                              np.linalg.norm(spec.joint_axis)))
                        cols["M"] = dq - ga * 0.5 * T ** 2         # G_P = -M g.a shares the mass column
            if t_.gravity and spec.parameter_source == "joint_space_gravity_coupled" and spec.joint_type == "revolute":
                A1, B1 = spec._g_js_coeffs({"mu": 1.0})
                cols["mu"] = cumulative_trapezoid(cumulative_trapezoid(A1 * np.cos(qq) + B1 * np.sin(qq), tt, initial=0),
                                                  tt, initial=0)
            if t_.viscous:
                cols["c"] = cumulative_trapezoid(dq, tt, initial=0)
            if t_.coulomb:
                s = np.sign(v[a:z])
                cols["Fc"] = cumulative_trapezoid(cumulative_trapezoid(s, tt, initial=0), tt, initial=0)
            if t_.gravity and spec.parameter_source == "joint_space_only":
                ii = lambda f: cumulative_trapezoid(cumulative_trapezoid(f, tt, initial=0), tt, initial=0)
                if spec.joint_type == "revolute":
                    cols["gA"], cols["gB"] = ii(np.cos(qq)), ii(np.sin(qq))
                else:
                    cols["gP"] = 0.5 * T ** 2
            if t_.bias:
                cols["b"] = 0.5 * T ** 2
            for k, p in spec.fixed_priors.items():                    # known priors move to the right-hand side
                if k in cols:
                    J = J - float(p["value"]) * cols.pop(k)
            rows.append((cols, T, len(seg_info)))
            rhs.append(J)
            seg_info.append(dict(interaction=it.name, i0=int(a), i1=int(z), t0=float(tt[0]), t1=float(tt[-1]),
                                 direction=int(np.sign(np.median(v[a:z]))), v_eps=eps))
            nuis_cols.append(None)
    if not rows:
        return dict(status="NO_MOVING_SEGMENTS", reason="no interval with |q'| above the velocity tolerance",
                    estimates={}, segments=[])
    nseg = len(seg_info)
    blocks, Y = [], []
    for cols, T, k in rows:
        n = len(T)
        B = np.zeros((n, len(names) + 2 * nseg))
        for j, nm in enumerate(names):
            if nm in cols:
                B[:, j] = cols[nm]
        B[:, len(names) + 2 * k] = -T                                 # (M v_a)
        B[:, len(names) + 2 * k + 1] = 1.0                            # offset e_a
        blocks.append(B)
    X = np.vstack(blocks)
    y = np.concatenate(rhs)
    sc = np.linalg.norm(X, axis=0)
    sc[sc == 0] = 1.0
    Xn = X / sc
    sv = np.linalg.svd(Xn, compute_uv=False)
    rank = int((sv > sv[0] * 1e-10).sum())
    coef_u, *_ = np.linalg.lstsq(Xn, y, rcond=None)
    coef_u = coef_u / sc
    lo = np.full(X.shape[1], -np.inf)
    nonneg = {"alpha", "M", "c", "Fc", "mu"}
    for j, nm in enumerate(names):
        if nm in nonneg:
            lo[j] = 0.0
    r = lsq_linear(Xn, y, bounds=(lo * sc, np.full(X.shape[1], np.inf)), method="bvls")
    coef = r.x / sc
    res = y - X @ coef
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(res @ res) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    phys = slice(0, len(names))
    Cp = cov[phys, phys]
    d = np.sqrt(np.clip(np.diag(Cp), 1e-300, None))
    corr = Cp / np.outer(d, d)
    violations = [nm for j, nm in enumerate(names) if nm in nonneg and coef_u[j] < 0]
    est = {nm: dict(value=float(coef[j]), std=float(se[j]), unconstrained=float(coef_u[j]),
                    rel_std=float(se[j] / abs(coef[j])) if coef[j] != 0 else math.inf) for j, nm in enumerate(names)}
    sv_phys = np.linalg.svd(Xn[:, phys], compute_uv=False) if len(names) else np.array([1.0])
    return dict(status="OK", method="general_integral_ls (weak double-integral form, exact ZOH force integral)",
                estimates=est, names=names, not_estimated=[n for n in spec.names if n in ("dFs", "Fs")],
                correlation={f"{a}~{b}": float(corr[i, j]) for i, a in enumerate(names) for j, b in
                             enumerate(names) if j > i},
                singular_values_scaled=sv.tolist(), rank=rank, n_columns=int(X.shape[1]),
                physical_singular_values_scaled=sv_phys.tolist(),
                physical_condition_number=float(sv_phys[0] / sv_phys[-1]) if sv_phys[-1] > 0 else math.inf,
                residual_rms=float(math.sqrt(s2)), physically_consistent=not violations,
                sign_violations_unconstrained=violations, segments=seg_info, n_rows=int(len(y)),
                directions=sorted({s["direction"] for s in seg_info}))


def theta0_from_ls(ls: dict, spec: ModelSpec, scale_hint: dict | None = None) -> np.ndarray:
    """Initial theta for the rollout fit (falls back to scale hints / small positive values)."""
    est = ls.get("estimates", {})
    hint = scale_hint or {}
    out = []
    for n in spec.names:
        if n in est and np.isfinite(est[n]["value"]):
            v = est[n]["value"]
            if n in ("alpha", "M") and v <= 0:
                v = hint.get(n, 1.0)
            out.append(v)
        elif n in ("dFs", "Fs"):
            out.append(hint.get(n, 0.1 * max(est.get("Fc", {}).get("value", 0.0), hint.get("effort", 1.0))))
        else:
            out.append(hint.get(n, 1.0 if n in ("alpha", "M") else 0.0))
    return np.asarray(out, float)
