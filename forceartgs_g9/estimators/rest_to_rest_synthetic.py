"""Exact synthetic rest-to-rest demonstrations for the two-term model (tests + the synthetic CLI benchmark).

``exact_rest_to_rest``: rest -> push(es) -> an exact braking row block that brings v to 0 at T (constant ZOH force over
d seconds solved in closed form) -> rest. Viscous coasting alone stops only asymptotically, so it is never used as an
exact rest-to-rest truth (``coast_only`` exists to TEST that). Truth comes from the EXACT two-term solution.
"""
from __future__ import annotations

import math

import numpy as np

from .rest_to_rest import build_rest_to_rest_episode
from .rest_to_rest_convolution import simulate_two_term

ROW_HZ = 50.0


def brake_level(M, c, v_b, d):
    """Constant effort that takes v from v_b to exactly 0 in d seconds (M v' + c v = Q_b)."""
    if c == 0.0:
        return -M * v_b / d
    lam = c / M
    e = math.exp(-lam * d)
    return -c * v_b * e / (1.0 - e)


def exact_rest_to_rest(M=0.8, c=1.5, *, q0=0.37, rest0=0.6, pushes=((0.3, 2.0),), brake_s=0.2, rest1=0.8,
                       q_every=2, jitter=0.0, noise=0.0, q_sigma=2e-4, seed=0, bias=0.0, Q_sigma=0.02):
    """Rows at 50 Hz; pushes = [(duration_s, level)] back to back; then the exact brake; then rest. Returns
    (force_t, Q_rows, t_q, q_obs, q_true, v_true, info). ``bias``: constant extra measured effort (an unmodelled
    load: the motion is simulated with Q_true, the RECORD carries Q_true + bias)."""
    for x in (rest0, brake_s, rest1, *(d for d, _ in pushes)):
        if abs(x * ROW_HZ - round(x * ROW_HZ)) > 1e-9:
            raise ValueError(f"{x} s is not on the {ROW_HZ:g} Hz row grid: the exact brake needs row-aligned phases")
    t_push_end = rest0 + sum(d for d, _ in pushes)
    t_b_end = t_push_end + brake_s
    t_end = t_b_end + rest1
    rows = np.arange(0, int(round(t_end * ROW_HZ)) + 1)
    ft = rows / ROW_HZ
    Q = np.zeros(len(rows))
    t = rest0
    for d, lev in pushes:
        Q[(ft >= t - 1e-9) & (ft < t + d - 1e-9)] = lev
        t += d
    v_b = simulate_two_term(ft, Q, 0.0, q0, M, c, [t_push_end])["v"][0]
    Q[(ft >= t_push_end - 1e-9) & (ft < t_b_end - 1e-9)] = brake_level(M, c, v_b, brake_s)
    rng = np.random.default_rng(seed)
    frames = np.arange(0, len(rows), q_every).astype(float)
    if jitter:
        frames[1:-1] = frames[1:-1] + rng.uniform(-jitter, jitter, len(frames) - 2)
    tq = frames / ROW_HZ
    sim = simulate_two_term(ft, Q, 0.0, q0, M, c, tq)
    q_obs = sim["q"] + rng.standard_normal(len(tq)) * noise
    return dict(force_t=ft, Q=Q + bias, Q_true=Q, t_q=tq, q=q_obs, q_true=sim["q"], v_true=sim["v"],
                frames=np.round(frames).astype(int), rows=rows, M=M, c=c, q0=q0, v_b=v_b, T_true=t_b_end,
                q_sigma=q_sigma, Q_sigma=Q_sigma)


def coast_only(M=0.8, c=1.5, *, q0=0.0, rest0=0.6, push=(0.3, 2.0), coast=1.6, q_every=2):
    """Push then a free viscous coast: v decays exponentially, never exactly zero."""
    t_end = rest0 + push[0] + coast
    rows = np.arange(0, int(round(t_end * ROW_HZ)) + 1)
    ft = rows / ROW_HZ
    Q = np.where((ft >= rest0 - 1e-9) & (ft < rest0 + push[0] - 1e-9), push[1], 0.0)
    tq = rows[::q_every] / ROW_HZ
    sim = simulate_two_term(ft, Q, 0.0, q0, M, c, tq)
    return dict(force_t=ft, Q=Q, t_q=tq, q=sim["q"], q_true=sim["q"], v_true=sim["v"], frames=rows[::q_every],
                rows=rows, M=M, c=c, q0=q0, q_sigma=2e-4, Q_sigma=0.02)


def episode(sy, **kw):
    """The rest-to-rest episode of a synthetic record (visual endpoint selection, unmasked rows)."""
    n = len(sy["t_q"])
    return build_rest_to_rest_episode(scene="synthetic", joint_type=kw.pop("joint_type", "prismatic"), t_q=sy["t_q"],
                                      q=sy["q"], q_sigma=np.full(n, sy["q_sigma"]), q_frames=sy["frames"],
                                      force_t=sy["force_t"], force_Q=sy["Q"],
                                      force_sigma=np.full(len(sy["Q"]), sy["Q_sigma"]), force_rows=sy["rows"], **kw)


def out_and_back(M=0.8, c=1.5, *, q0=-0.12, A=2.0, d=0.3, brake_s=0.2, gap=0.3, rest0=0.6, rest1=0.8, q_every=2,
                 q_sigma=1e-9, Q_sigma=0.02):
    """Push +A, exact brake, rest, push -A, mirrored brake, rest: the second half mirrors the first, so the object
    returns EXACTLY to q0 (delta q = 0, nontrivial motion in between)."""
    t1 = rest0 + d
    t2 = t1 + brake_s
    t3 = t2 + gap
    t4 = t3 + d
    t5 = t4 + brake_s
    t_end = t5 + rest1
    rows = np.arange(0, int(round(t_end * ROW_HZ)) + 1)
    ft = rows / ROW_HZ
    Q = np.zeros(len(rows))
    Q[(ft >= rest0 - 1e-9) & (ft < t1 - 1e-9)] = A
    v_b = simulate_two_term(ft, Q, 0.0, q0, M, c, [t1])["v"][0]
    Qb = brake_level(M, c, v_b, brake_s)
    Q[(ft >= t1 - 1e-9) & (ft < t2 - 1e-9)] = Qb
    Q[(ft >= t3 - 1e-9) & (ft < t4 - 1e-9)] = -A
    Q[(ft >= t4 - 1e-9) & (ft < t5 - 1e-9)] = -Qb
    tq = rows[::q_every] / ROW_HZ
    sim = simulate_two_term(ft, Q, 0.0, q0, M, c, tq)
    return dict(force_t=ft, Q=Q, Q_true=Q, t_q=tq, q=sim["q"], q_true=sim["q"], v_true=sim["v"], frames=rows[::q_every],
                rows=rows, M=M, c=c, q0=q0, q_sigma=q_sigma, Q_sigma=Q_sigma, T_true=t5)
