"""Fixed-camera visual evidence (spec section 6, level V1) and a point-splat render proxy.

All predictions go through each camera's own ``K_c`` and fixed ``T_WC_c``; there is no
cross-camera track identity. Residuals are whitened by the stated noise model so 2D (px) and
depth (m) terms are dimensionless and comparable. The 2D pixel and its depth are treated as one
measurement with independent image-plane and depth-axis noise (no double counting of the pixel as
a separate 3D point).
"""
from __future__ import annotations

import numpy as np

from .observations import TrackObservations, depth_sigma
from .rigid_kinematics import apply_joint, transform_points


def camera_stacks(obs: TrackObservations, calibs: dict):
    K = np.stack([np.asarray(calibs[c].K, float) for c in obs.camera_ids])
    T_CW = np.stack([calibs[c].T_CW for c in obs.camera_ids])
    return K, T_CW


def project_stacked(x_w, cam, K, T_CW):
    x_c = transform_points(T_CW[cam], x_w)
    z = x_c[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    Kc = K[cam]
    u = Kc[:, 0, 0] * x_c[:, 0] / zs + Kc[:, 0, 1] * x_c[:, 1] / zs + Kc[:, 0, 2]
    v = Kc[:, 1, 1] * x_c[:, 1] / zs + Kc[:, 1, 2]
    return np.stack([u, v], 1), z, x_c


def predict_world(kind, axis, origin, q_t, X):
    """Moving prediction ``A_H(q_t) X``; ``kind='static'`` returns X."""
    if kind == "static":
        return np.asarray(X, float)
    return apply_joint(kind, axis, origin, np.asarray(q_t, float), np.asarray(X, float))


def whitened_residuals(kind, axis, origin, q, anchors, obs: TrackObservations, K, T_CW, noise, obs_mask=None,
                       use_depth=True):
    """Per-observation whitened residuals.

    Returns dict(r_uv [M,2], r_z [M] NaN where depth is unusable, uv_pred, z_pred, mask)."""
    m = np.ones(obs.n_obs, bool) if obs_mask is None else np.asarray(obs_mask, bool)
    idx = np.flatnonzero(m)
    X = anchors[obs.track[idx]]
    xw = predict_world(kind, axis, origin, q[obs.time[idx]], X)
    uv_pred, z_pred, _ = project_stacked(xw, obs.cam[idx], K, T_CW)
    r_uv = np.full((obs.n_obs, 2), np.nan)
    r_z = np.full(obs.n_obs, np.nan)
    uvv = obs.uv_valid[idx]
    r_uv[idx[uvv]] = (uv_pred[uvv] - obs.uv[idx[uvv]]) / noise.sigma_px
    if use_depth:
        dz = obs.depth_valid[idx] & obs.mask_valid[idx] & np.isfinite(obs.z[idx])
        sz = depth_sigma(obs.z[idx[dz]], noise, obs.depth_edge[idx[dz]], obs.depth_cos[idx[dz]])
        r_z[idx[dz]] = (z_pred[dz] - obs.z[idx[dz]]) / sz
    full_uv = np.full((obs.n_obs, 2), np.nan)
    full_z = np.full(obs.n_obs, np.nan)
    full_uv[idx], full_z[idx] = uv_pred, z_pred
    return dict(r_uv=r_uv, r_z=r_z, uv_pred=full_uv, z_pred=full_z)


def residual_report(res, obs: TrackObservations, track_weight=None, noise=None) -> dict:
    """Per camera / per time robust residual statistics (px and m when ``noise`` given)."""
    w = np.ones(obs.n_tracks) if track_weight is None else np.asarray(track_weight)
    out = {}
    for ci, cid in enumerate(obs.camera_ids):
        per_t = []
        for t in range(obs.frame_map.T):
            sel = (obs.cam == ci) & (obs.time == t) & (w[obs.track] > 0.5)
            ruv = res["r_uv"][sel]
            ruv = ruv[np.isfinite(ruv).all(1)]
            rz = res["r_z"][sel]
            rz = rz[np.isfinite(rz)]
            per_t.append(dict(t=t, timestamp_s=float(obs.frame_map.timestamp_s[t]), n_uv=int(len(ruv)), n_depth=int(len(rz)),
                              median_uv_whitened=float(np.median(np.linalg.norm(ruv, axis=1))) if len(ruv) else None,
                              median_abs_depth_whitened=float(np.median(np.abs(rz))) if len(rz) else None))
        sel = (obs.cam == ci) & (w[obs.track] > 0.5)
        ruv = res["r_uv"][sel]
        ruv = ruv[np.isfinite(ruv).all(1)]
        rz = res["r_z"][sel]
        rz = rz[np.isfinite(rz)]
        entry = dict(n_moving_tracks=int(((obs.track_camera == ci) & (w > 0.5)).sum()), per_time=per_t,
                     median_uv_whitened=float(np.median(np.linalg.norm(ruv, axis=1))) if len(ruv) else None,
                     median_abs_depth_whitened=float(np.median(np.abs(rz))) if len(rz) else None)
        if noise is not None and entry["median_uv_whitened"] is not None:
            entry["median_uv_px"] = entry["median_uv_whitened"] * noise.sigma_px
        out[cid] = entry
    return out


def silhouette_support(points_w, calib, mask, depth=None, occlusion_margin_m=0.02) -> dict:
    """Fraction of projected points inside the whole-object mask among points predicted visible.

    A point is skipped (not counted as an error) if it projects outside the image or if valid
    measured depth is in front of it by more than ``occlusion_margin_m`` (occluded)."""
    uv, z = None, None
    x_c = transform_points(calib.T_CW, np.asarray(points_w, float))
    z = x_c[:, 2]
    K = np.asarray(calib.K, float)
    uv = np.stack([K[0, 0] * x_c[:, 0] / z + K[0, 2], K[1, 1] * x_c[:, 1] / z + K[1, 2]], 1)
    H, W = mask.shape[:2]
    px = np.rint(uv).astype(int)
    inside = (z > 0) & (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < W) & (px[:, 1] < H)
    visible = inside.copy()
    if depth is not None:
        d = np.full(len(z), np.nan)
        d[inside] = depth[px[inside, 1], px[inside, 0]]
        occluded = np.isfinite(d) & (d > 0) & (d < z - occlusion_margin_m)
        visible &= ~occluded
    in_mask = np.zeros(len(z), bool)
    in_mask[visible] = mask[px[visible, 1], px[visible, 0]] > 0
    n = int(visible.sum())
    return dict(n_visible=n, n_in_mask=int(in_mask.sum()), fraction=float(in_mask.sum() / n) if n else None,
                n_occluded_or_outside=int(len(z) - n))


def render_point_proxy(points_w, colors, calib, T_world_points=None, splat_radius_px=0):
    """Minimal z-buffer point-splat renderer (RGB, metric camera-Z depth, alpha).

    ``T_world_points`` optionally maps the point set into the world (object motion). Used to prove
    that moving the object under a fixed camera equals rendering the canonical object from the
    derived virtual camera (spec 11 / test T20). Not a Gaussian renderer."""
    X = np.asarray(points_w, float)
    if T_world_points is not None:
        X = transform_points(np.asarray(T_world_points, float), X)
    x_c = transform_points(calib.T_CW, X)
    z = x_c[:, 2]
    K = np.asarray(calib.K, float)
    ok = z > 1e-6
    u = K[0, 0] * x_c[ok, 0] / z[ok] + K[0, 2]
    v = K[1, 1] * x_c[ok, 1] / z[ok] + K[1, 2]
    H, W = calib.height, calib.width
    rgb = np.zeros((H, W, 3))
    depth = np.full((H, W), np.inf)
    alpha = np.zeros((H, W))
    cols = np.asarray(colors, float)[ok]
    zz = z[ok]
    order = np.argsort(-zz)  # far first, near overwrites
    r = int(splat_radius_px)
    for i in order:
        cx, cy = int(np.rint(u[i])), int(np.rint(v[i]))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < W and 0 <= y < H and zz[i] <= depth[y, x]:
                    depth[y, x] = zz[i]
                    rgb[y, x] = cols[i]
                    alpha[y, x] = 1.0
    depth[~np.isfinite(depth)] = 0.0
    return rgb, depth, alpha


def patch_ncc(image_a, uv_a, image_b, uv_b, radius=3):
    """Normalized cross-correlation of grey patches around matched predicted pixels (appearance
    diagnostic; callers must pass only predicted-visible observations)."""
    def grab(img, uv):
        g = img.mean(-1) if img.ndim == 3 else img
        H, W = g.shape
        out = np.full((len(uv), (2 * radius + 1) ** 2), np.nan)
        for k, (u, v) in enumerate(np.rint(uv).astype(int)):
            if radius <= u < W - radius and radius <= v < H - radius:
                out[k] = g[v - radius:v + radius + 1, u - radius:u + radius + 1].ravel()
        return out
    a, b = grab(image_a, uv_a), grab(image_b, uv_b)
    a = a - np.nanmean(a, 1, keepdims=True)
    b = b - np.nanmean(b, 1, keepdims=True)
    den = np.sqrt(np.nansum(a * a, 1) * np.nansum(b * b, 1))
    return np.where(den > 1e-9, np.nansum(a * b, 1) / np.where(den > 1e-9, den, 1), np.nan)
