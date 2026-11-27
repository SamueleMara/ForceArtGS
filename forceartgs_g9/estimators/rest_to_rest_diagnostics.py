"""Rest-to-rest diagnostics of the two-term model (computed from the SAME data; never extra training targets).

* Orthogonality (Eq. 7): for the true continuous v on an exact rest-to-rest window, int v vdot dt = (v(T)^2 - v(t0)^2)/2
  = 0, so the UNWEIGHTED strong-form columns qdd and qd are orthogonal there. Nothing more is claimed: a weighted,
  discretized or noisy derivative is generally NOT orthogonal, and orthogonality is not rank / unbiasedness / a
  diagonal covariance. (Not a Hilbert transform.)
* Damping energy (Eq. 8): int Q v dt = c int v^2 dt for the minimal model -> c_energy = int Q v / int v^2 from a SMOOTH
  trajectory (smoothing spline of the observed q, or a constrained PINN trajectory), with a bootstrap of the declared q /
  wrench noise as its honest derivative uncertainty. Purely viscous dynamics need int Q v >= 0; a significantly
  negative external work is a model / input inconsistency.
* Model-free work W_in = sum_k Qbar_k (q_{k+1} - q_k) on the visual clock (no derivative at all).
"""
from __future__ import annotations

import math

import numpy as np

from .rest_to_rest import RestToRestEpisode


def _trapz(y, x):
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def orthogonality(t, v, a) -> dict:
    t, v, a = (np.asarray(x, float) for x in (t, v, a))
    I = _trapz(v * a, t)
    nv, na = math.sqrt(max(_trapz(v * v, t), 0.0)), math.sqrt(max(_trapz(a * a, t), 0.0))
    return dict(int_v_vdot=I, exact_identity=0.5 * (v[-1] ** 2 - v[0] ** 2), normalized=I / max(nv * na, 1e-300),
                claim="exact only for the true continuous v with v(t0) = v(T) = 0 and the UNWEIGHTED inner product; a "
                      "weighted / discrete / noisy derivative is not claimed orthogonal; says nothing about rank")


def _fine_grid(ep, dt=1e-3):
    t = np.arange(ep.t0, ep.T + 1e-12, dt)
    return np.unique(np.r_[t, ep.T])


def _spline_v(t, q, s):
    from scipy.interpolate import make_smoothing_spline
    w = 1.0 / np.asarray(s, float) ** 2
    sp = make_smoothing_spline(t, q, w=w / w.mean())
    return sp, sp.derivative()


def energy_audit(ep: RestToRestEpisode, *, n_boot=50, seed=0, k=3.0, v_fn=None, label="smoothing_spline") -> dict:
    """c_energy = int Q v / int v^2 on a 1 ms grid (ZOH Q exact on the grid), v from a smoothing spline of q (GCV,
    weights 1/sigma^2) or ``v_fn``. Bootstrap: q noise (+ anchors) and wrench-row noise, spline refitted per draw."""
    tg = _fine_grid(ep)
    idx = np.clip(np.searchsorted(ep.force_t_s, tg, side="right") - 1, 0, len(ep.force_Q) - 1)
    Qg = ep.force_Q[idx]

    def one(q, Q_rows):
        if v_fn is not None:
            vg = np.asarray(v_fn(tg), float)
        else:
            _, dsp = _spline_v(ep.t_q_s, q, ep.q_sigma)
            vg = dsp(tg)
        Qv = Qg if Q_rows is None else np.asarray(Q_rows, float)[idx]
        W = _trapz(Qv * vg, tg)
        D = _trapz(vg * vg, tg)
        return W, D, vg
    W, D, vg = one(ep.q_obs, None)
    out = dict(method=f"damping energy int Q v / int v^2 ({label})", W_ext=W, int_v2=D,
               c_energy=W / D if D > 0 else None, v_start=float(vg[0]), v_end=float(vg[-1]))
    if v_fn is None and n_boot:
        rng = np.random.default_rng(seed)
        sQ = np.zeros(len(ep.force_Q)) if ep.force_sigma is None else np.asarray(ep.force_sigma, float)
        Ws, cs = [], []
        for _ in range(int(n_boot)):
            qd = ep.q_obs + rng.standard_normal(len(ep.q_obs)) * ep.q_sigma
            Qd = ep.force_Q + rng.standard_normal(len(sQ)) * sQ
            Wd, Dd, _ = one(qd, Qd)
            Ws.append(Wd)
            if Dd > 0:
                cs.append(Wd / Dd)
        out["bootstrap"] = dict(n=len(Ws), W_sd=float(np.std(Ws)), c_energy_p2_5_p97_5=np.percentile(cs, [2.5, 97.5]).tolist()
                                if cs else None, c_energy_sd=float(np.std(cs)) if cs else None,
                                note="declared q / wrench noise; the spline is refitted per draw (derivative "
                                     "uncertainty included)")
        sdW = out["bootstrap"]["W_sd"]
    else:
        sdW = 0.0
    out["verdict"] = ("NEGATIVE_EXTERNAL_WORK_MODEL_INPUT_INCONSISTENT" if W < -k * max(sdW, 1e-300) and sdW > 0
                      else "NONNEGATIVE_WORK_CONSISTENT" if W >= 0 else "NEGATIVE_WITHIN_NOISE")
    out["rule"] = f"purely viscous dynamics dissipate: int Q v >= 0; < -{k} sd(bootstrap) is inconsistent"
    return out


def work_model_free(ep: RestToRestEpisode, k=3.0) -> dict:
    t, q, s = ep.t_q_s, ep.q_obs, ep.q_sigma
    ft, Q = ep.force_t_s, ep.force_Q
    Qs = ep.force_sigma if ep.force_sigma is not None else np.zeros_like(Q)
    Qbar, Qbar_sd = np.empty(len(t) - 1), np.empty(len(t) - 1)
    for i in range(len(t) - 1):
        a, b = t[i], t[i + 1]
        edges = np.r_[a, ft[(ft > a) & (ft < b)], b]
        idx = np.clip(np.searchsorted(ft, edges[:-1] + 1e-12, side="right") - 1, 0, len(Q) - 1)
        w = np.diff(edges) / (b - a)
        Qbar[i] = float(w @ Q[idx])
        Qbar_sd[i] = float(np.sqrt((w ** 2) @ (Qs[idx] ** 2)))
    dq = np.diff(q)
    sd_dq = np.hypot(s[1:], s[:-1])
    W = float(Qbar @ dq)
    W_sd = float(np.sqrt(np.sum((dq * Qbar_sd) ** 2) + np.sum((Qbar * sd_dq) ** 2)))
    return dict(W_in=W, W_in_sd=W_sd, units="J (revolute: N m rad; prismatic: N m)",
                verdict=("INPUT_OUTPUT_SIGN_INCONSISTENT" if W < -k * W_sd else "PASSIVE_ENERGY_CONSISTENT"),
                rule=f"from rest to rest, a passive viscous object needs W_in >= -{k} sd (no derivative used)")


def strong_form_orthogonality_check(ep: RestToRestEpisode, M, c) -> dict:
    """Orthogonality of v and vdot for the EXACT two-term replay at (M, c) on a fine grid (the minimal model's own
    v(t), not a noisy derivative). Its v(T) is generally NOT zero (viscous decay stops only asymptotically)."""
    from .rest_to_rest_convolution import replay
    tg = _fine_grid(ep)
    r = replay(ep, M, c, tg)
    a = (ep.effort.at(tg) - c * r["v"]) / M
    return dict(orthogonality(tg, r["v"], a), v_T=float(r["v"][-1]),
                note="evaluated on the minimal model's exact replay; non-zero int v vdot = v(T)^2 / 2 when v(T) != 0")
