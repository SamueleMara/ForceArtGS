"""Calibrated null for rotation evidence, rank/conditioning, detectability (spec 2.3).

The null is *empirical*: the same robust interval estimator is applied to patches of the candidate-independent
static core, each patch matched to the moving support of that interval in track count and spatial extent (nearest
neighbours around a random static seed).  Static patches carry the real depth bias, correlation, occlusion and
calibration error of the scene.  They do not carry motion-induced tracker bias, so each interval's null is inflated
by the ratio of the moving support's robust residual scale to the static patches' scale (never deflated).
"""
from __future__ import annotations

import numpy as np

from .motion_field import PointTable, estimate_interval_rigid_evidence, support_principal


def support_configuration(X, w=None, rank_tol_m=0.005) -> dict:
    """Principal stds, rank and condition of a point configuration: a strip/line constrains one rotation poorly."""
    X = np.asarray(X, float)
    if len(X) < 3:
        return dict(status="INSUFFICIENT_POINTS", n=len(X))
    sd, V, m = support_principal(X, w)
    rank = int(np.sum(sd > rank_tol_m))
    return dict(status="OK", n=len(X), principal_std_m=sd.round(5).tolist(), principal_axes=V.T.round(5).tolist(),
                rank=rank, condition=float(sd[0] / max(sd[2], 1e-6)), baseline_m=float(2 * sd[1]),
                weakly_constrained_rotation_axis=V[:, 0].round(5).tolist() if rank >= 1 else None,
                note="rotation about the first principal axis is constrained only by the 2nd/3rd stds")


def _patch(static_idx, Xs, seed_i, k):
    d = np.linalg.norm(Xs - Xs[seed_i], axis=1)
    return static_idx[np.argsort(d)[:k]]


def calibrate_rotation_null(tab: PointTable, static_core, intervals, axis=None, n_patches=40, seed=0,
                            calibs=None, min_tracks=8) -> dict:
    """Null distribution of the net rotation (projected on ``axis`` if given, else vector-summed norm) for a support
    that does not rotate.  ``intervals``: the moving-support RigidIntervalEvidence list (defines frames, counts)."""
    rng = np.random.default_rng(seed)
    static_core = np.asarray(static_core, bool)
    per_interval, rows, trans_null = [], [], []
    for iv in intervals:
        a, b = iv.frame_a, iv.frame_b
        idx, P, Q, _ = tab.pair(a, b, static_core)
        n_mov = max(iv.n_tracks, min_tracks)
        if len(idx) < max(min_tracks, n_mov // 2) or iv.status != "OK":
            per_interval.append(None)
            rows.append(dict(frame_a=a, frame_b=b, status="NO_NULL", n_static=len(idx)))
            continue
        k = min(n_mov, len(idx))
        ws, scales, cds = [], [], []
        for _ in range(n_patches):
            sel = _patch(idx, P, int(rng.integers(len(idx))), k)
            m = np.zeros(tab.N)
            m[sel] = 1.0
            ev = estimate_interval_rigid_evidence(tab, m, a, b, seed=int(rng.integers(1 << 30)), refine_2d=False,
                                                  calibs=calibs, min_tracks=min_tracks)
            if ev.status in ("OK", "INSUFFICIENT_CLUSTERS") and ev.n_tracks >= min_tracks:
                ws.append(ev.rotation_vector_rad)
                scales.append(ev.diagnostics.get("robust_scale_m", np.nan))
                cds.append(ev.centroid_b - ev.centroid_a)
        if len(ws) < 5:
            per_interval.append(None)
            rows.append(dict(frame_a=a, frame_b=b, status="NO_NULL", n_patches=len(ws)))
            continue
        W = np.stack(ws)
        s_static = float(np.nanmedian(scales))
        s_moving = float(iv.diagnostics.get("robust_scale_m", np.nan))
        infl = float(max(1.0, s_moving / s_static)) if np.isfinite(s_moving) and s_static > 0 else 1.0
        per_interval.append(W * infl)
        trans_null.append(np.stack(cds) * infl)
        rows.append(dict(frame_a=a, frame_b=b, status="OK", n_patches=len(ws), patch_tracks=k,
                         static_scale_mm=1000 * s_static, moving_scale_mm=1000 * s_moving, inflation=infl,
                         null_angle_q95_deg=float(np.degrees(np.percentile(np.linalg.norm(W * infl, axis=1), 95)))))
    valid = [w for w in per_interval if w is not None]
    if not valid:
        return dict(status="NO_NULL", per_interval=rows)
    # net-rotation null: one patch draw per interval, summed (intervals are disjoint time blocks)
    B = 4000
    tot = np.zeros(B)
    vec = np.zeros((B, 3))
    for W in valid:
        pick = W[rng.integers(0, len(W), B)]
        vec += pick
        if axis is not None:
            tot += pick @ np.asarray(axis, float)
    net = np.abs(tot) if axis is not None else np.linalg.norm(vec, axis=1)
    tvec = np.zeros((B, 3))
    for Dn in trans_null:
        tvec += Dn[rng.integers(0, len(Dn), B)]
    return dict(status="OK", projection="shared axis" if axis is not None else "vector norm",
                net_abs_rotation_q95_deg=float(np.degrees(np.percentile(net, 95))),
                net_abs_rotation_q99_deg=float(np.degrees(np.percentile(net, 99))),
                minimum_detectable_rotation_deg=float(np.degrees(np.percentile(net, 95))),
                net_translation_q95_m=float(np.percentile(np.linalg.norm(tvec, axis=1), 95)),
                n_intervals_with_null=len(valid), per_interval=rows,
                caveat="static patches carry depth/calibration/occlusion error but not motion-induced tracker bias; "
                       "each interval is inflated by moving/static robust residual scale (>= 1)")


def rotation_observability_report(intervals, shared_axis, null, support_X) -> dict:
    """Everything spec 2.3 asks to export, in one place."""
    good = [iv for iv in intervals if iv.status == "OK"]
    rows = [iv.to_row() for iv in intervals]
    conf = support_configuration(support_X) if support_X is not None and len(support_X) >= 3 else None
    ratio = [iv.diagnostics.get("robust_scale_m") for iv in good]
    return dict(per_interval=rows, n_valid_intervals=len(good), shared_axis=shared_axis, null=null,
                support_configuration=conf,
                moving_robust_residual_mm=[None if r is None else round(1000 * r, 3) for r in ratio])
