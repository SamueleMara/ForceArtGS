"""Competing revolute kinematic hypotheses re-estimated from the ORIGINAL RGB-D 3-D track observations (spec 3.1.7):
the accepted free-axis hinge vs an exactly vertical hinge. Each hypothesis gets its OWN q(t) (and, downstream, its
own projected effort); neither replaces the accepted joint authority -- this is an ablation.

Model per moving track k observed at frames t: x_tk = o + R_a(q_t)(x_k,ref - o), x_k,ref = its position at the
reference frame. Alternating least squares: q_t closed form per frame (rotation about a fixed axis), o linear
(perpendicular to a); the free-axis variant also optimises the axis direction. Held-out: the hinge (axis, origin) is
fitted on one temporal half of the frames and evaluated (q_t re-solved per frame) on the other half, both ways.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ..dynamics.general_1dof import rotation


def _unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def load_moving_tracks(obs_dir, p_min=0.9, min_frames=8, max_tracks=600, seed=0):
    d = Path(obs_dir)
    z = np.load(d / "tracks_observations.npz", allow_pickle=False)
    sup = np.load(d / "part_support" / "track_support.npz", allow_pickle=False)
    pm = sup["p_moving_R"] if "p_moving_R" in sup.files else sup["p_moving_P"]
    moving = (pm >= p_min) & sup["eligible"]
    ok = z["depth_valid"] & z["uv_valid"] & ~z["depth_edge"] & z["mask_valid"]
    tr, ti, X = z["track"][ok], z["time"][ok], z["x_w"][ok]
    sel = moving[tr]
    tr, ti, X = tr[sel], ti[sel], X[sel]
    T = int(z["time"].max()) + 1
    tracks = {}
    for k in np.unique(tr):
        m = tr == k
        if m.sum() >= min_frames and 0 in set(ti[m].tolist()):
            P = np.full((T, 3), np.nan)
            P[ti[m]] = X[m]
            tracks[int(k)] = P
    keys = sorted(tracks)
    if len(keys) > max_tracks:
        keys = sorted(np.random.default_rng(seed).choice(keys, max_tracks, replace=False).tolist())
    A = np.stack([tracks[k] for k in keys]) if keys else np.zeros((0, T, 3))
    return A, dict(n_moving_tracks_total=int(moving.sum()), n_tracks_used=len(keys), p_moving_min=p_min,
                   frames=T, fm_rgbd=z["fm_rgbd"].tolist(), fm_source=z["fm_source"].tolist(),
                   source=str(d / "tracks_observations.npz"))


def _angle(a, o, ref, cur):
    """Closed-form rotation angle about axis a through o mapping ref -> cur (least squares, per frame)."""
    m = np.isfinite(cur).all(1) & np.isfinite(ref).all(1)
    if m.sum() < 3:
        return np.nan
    u = ref[m] - o
    v = cur[m] - o
    up = u - np.outer(u @ a, a)
    vp = v - np.outer(v @ a, a)
    s = np.sum(np.einsum("ij,ij->i", np.cross(up, vp), np.tile(a, (len(up), 1))))
    c = np.sum(np.einsum("ij,ij->i", up, vp))
    return math.atan2(s, c)


def _origin(a, qs, A, frames):
    """o minimising sum ||(x_t - R x_ref) - (I - R) o||^2 (only the component perpendicular to a is defined)."""
    ref = A[:, 0]
    M, r = np.zeros((3, 3)), np.zeros(3)
    for t in frames:
        if not np.isfinite(qs[t]):
            continue
        R = rotation(a, qs[t])
        K = np.eye(3) - R
        m = np.isfinite(A[:, t]).all(1) & np.isfinite(ref).all(1)
        d = A[m, t] - ref[m] @ R.T
        M += m.sum() * K.T @ K
        r += K.T @ d.sum(0)
    P = np.eye(3) - np.outer(a, a)
    o = np.linalg.lstsq(M + 1e-9 * np.outer(a, a), r, rcond=None)[0]
    return P @ o


def _residuals(a, o, qs, A, frames):
    ref = A[:, 0]
    res = []
    for t in frames:
        if not np.isfinite(qs[t]):
            continue
        R = rotation(a, qs[t])
        pred = o + (ref - o) @ R.T
        m = np.isfinite(A[:, t]).all(1) & np.isfinite(ref).all(1)
        res.append(np.linalg.norm(A[m, t] - pred[m], axis=1))
    return np.concatenate(res) if res else np.zeros(0)


def fit_hinge(A, frames, a0, o0, free_axis, iters=8):
    a = _unit(a0)
    o = np.asarray(o0, float)

    def inner(a):
        o_ = o
        for _ in range(iters):
            qs = np.full(A.shape[1], np.nan)
            for t in frames:
                qs[t] = _angle(a, o_, A[:, 0], A[:, t])
            o_ = _origin(a, qs, A, frames)
        return o_, qs

    if free_axis:
        def obj(ang):
            aa = _unit([math.sin(ang[0]) * math.cos(ang[1]), math.sin(ang[0]) * math.sin(ang[1]), math.cos(ang[0])])
            o_, qs = inner(aa)
            r = _residuals(aa, o_, qs, A, frames)
            return float(np.mean(np.minimum(r, 0.05) ** 2))          # truncated (robust) squared residual
        th0 = [math.acos(np.clip(a[2], -1, 1)), math.atan2(a[1], a[0])]
        best = minimize(obj, th0, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-12, maxiter=200))
        a = _unit([math.sin(best.x[0]) * math.cos(best.x[1]), math.sin(best.x[0]) * math.sin(best.x[1]),
                   math.cos(best.x[0])])
    o, qs = inner(a)
    return a, o, qs


def evaluate(A, a, o, frames):
    qs = np.full(A.shape[1], np.nan)
    for t in frames:
        qs[t] = _angle(a, o, A[:, 0], A[:, t])
    r = _residuals(a, o, qs, A, frames)
    return qs, dict(rms_m=float(np.sqrt(np.mean(r ** 2))) if len(r) else None,
                    median_m=float(np.median(r)) if len(r) else None,
                    p90_m=float(np.percentile(r, 90)) if len(r) else None, n=int(len(r)))


def hypotheses(obs_dir, accepted_axis, accepted_origin) -> dict:
    A, info = load_moving_tracks(obs_dir)
    if A.shape[0] < 10:
        return dict(status="NOT_RUN", reason=f"{A.shape[0]} moving tracks observed at the reference frame", **info)
    T = A.shape[1]
    allf = list(range(1, T))
    # track-level outlier rejection decided ONCE on the accepted-axis fit; the same inlier set serves every
    # hypothesis (a fair comparison)
    a_acc0 = _unit(accepted_axis)
    a_, o_, qs_ = fit_hinge(A, allf, a_acc0, accepted_origin, False)
    per = []
    for k in range(A.shape[0]):
        r = _residuals(a_, o_, qs_, A[k:k + 1], allf)
        per.append(np.sqrt(np.mean(r ** 2)) if len(r) else np.inf)
    per = np.asarray(per)
    keep = per <= 2.0 * np.median(per)
    A = A[keep]
    info.update(n_tracks_inliers=int(keep.sum()), inlier_rule="per-track RMS <= 2 x median (accepted-axis fit)",
                per_track_rms_median_m=float(np.median(per)))
    halves = (list(range(1, T // 2)), list(range(T // 2, T)))
    a_acc = a_acc0
    vz = np.array([0, 0, 1.0]) * (1 if a_acc[2] >= 0 else -1)
    out = dict(schema="forceartgs.g9.kinematics_hypotheses/v1", tracks=info, hypotheses={},
               note="re-estimated from the original RGB-D 3-D tracks of the moving part; an ABLATION -- the accepted "
                    "joint authority is unchanged and remains the upstream decision")
    for name, a0, free in (("accepted_axis_fixed", a_acc, False), ("free_axis_refit", a_acc, True),
                           ("vertical_hinge", vz, False)):
        a, o, qs = fit_hinge(A, allf, a0, accepted_origin, free)
        _, tr = evaluate(A, a, o, allf)
        held = []
        for tr_f, te_f in (halves, halves[::-1]):
            a_h, o_h, _ = fit_hinge(A, tr_f, a0, accepted_origin, free)
            _, ev = evaluate(A, a_h, o_h, te_f)
            held.append(ev)
        out["hypotheses"][name] = dict(axis_B=a.tolist(), origin_B=o.tolist(),
                                       tilt_from_vertical_deg=float(math.degrees(math.acos(min(1, abs(a[2]))))),
                                       q_rgbd_frames=[None if not np.isfinite(x) else float(x) for x in qs],
                                       train_residual=tr, heldout_residual=held,
                                       heldout_rms_mean_m=float(np.mean([h["rms_m"] for h in held])),
                                       axis_free=free)
    hv, ha = out["hypotheses"]["vertical_hinge"], out["hypotheses"]["accepted_axis_fixed"]
    out["comparison"] = dict(
        heldout_rms_vertical_minus_accepted_m=hv["heldout_rms_mean_m"] - ha["heldout_rms_mean_m"],
        verdict=("VERTICAL_WORSE_GEOMETRICALLY" if hv["heldout_rms_mean_m"] > 1.05 * ha["heldout_rms_mean_m"] else
                 "VERTICAL_NOT_DISTINGUISHABLE" if hv["heldout_rms_mean_m"] > 0.95 * ha["heldout_rms_mean_m"] else
                 "VERTICAL_BETTER_GEOMETRICALLY"),
        rule="geometric (RGB-D) held-out residuals only; a better dynamics replay is never kinematic evidence")
    return out
