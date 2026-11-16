"""Analytic synthetic scenes for tests and the ``synthetic`` CLI (independent of the optimizer).

Ground truth produced here is for *oracle diagnostics only*; the estimator never reads it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observations import CameraCalib, FrameMap, TrackObservations
from .rigid_kinematics import apply_joint, normalize


def look_at_camera(camera_id, center, target, up=(0, 0, 1), f=300.0, width=320, height=240) -> CameraCalib:
    """OpenCV camera (x right, y down, z forward) at ``center`` looking at ``target``."""
    c, tgt = np.asarray(center, float), np.asarray(target, float)
    z = normalize(tgt - c)
    x = normalize(np.cross(z, np.asarray(up, float)))
    if not np.isfinite(x).all() or np.linalg.norm(np.cross(z, up)) < 1e-6:
        x = normalize(np.cross(z, [0, 1, 0]))
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = c
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    return CameraCalib(camera_id, K, T, width, height, calibration_source="synthetic look_at")


@dataclass
class SyntheticScene:
    obs: TrackObservations
    calibs: dict
    gt_kind: str
    gt_axis: np.ndarray
    gt_origin: np.ndarray | None
    gt_q: np.ndarray
    gt_moving: np.ndarray          # [N] oracle part label (diagnostic only)
    gt_anchor_ref: np.ndarray      # [N,3] canonical (q_ref=0) positions
    seed: int


def box_surface(rng, center, size, n):
    """Points on the visible faces of an axis-aligned box (top and four sides)."""
    c, s = np.asarray(center, float), np.asarray(size, float)
    pts = []
    for _ in range(n):
        face = rng.integers(0, 5)
        p = rng.uniform(-0.5, 0.5, 3) * s
        if face == 0:
            p[2] = s[2] / 2
        else:
            ax = (face - 1) // 2
            p[ax] = (1 if face % 2 else -1) * s[ax] / 2
        pts.append(c + p)
    return np.array(pts)


def make_scene(kind="revolute", seed=0, T=12, n_static=140, n_moving=60, sigma_px=0.5, sigma_z=0.002,
               outlier_fraction=0.0, depth_invalid_fraction=0.0, dropout=0.0, max_angle_deg=40.0, max_disp_m=0.12,
               hinge_offset=None, axis=None, cameras=("cam_001", "cam_002"), disjoint_moving_cameras=False,
               base_only_camera=None, late_tracks_fraction=0.0, fps=10.0) -> SyntheticScene:
    """Static base + one moving part observed by fixed calibrated cameras.

    ``disjoint_moving_cameras`` gives each camera a different half of the moving points;
    ``base_only_camera`` names a camera that sees only base points (zero moving tracks)."""
    rng = np.random.default_rng(seed)
    base = box_surface(rng, [0, 0, 0.05], [0.3, 0.2, 0.1], n_static)
    if kind == "revolute":
        lid = box_surface(rng, [0, 0, 0.115], [0.3, 0.2, 0.03], n_moving)
        u = normalize(np.array([1.0, 0.0, 0.0]) if axis is None else axis)
        o = np.array([0.0, 0.1, 0.13]) if hinge_offset is None else np.asarray(hinge_offset, float)
        qmax = -np.radians(max_angle_deg)
    else:
        lid = box_surface(rng, [0, -0.02, 0.05], [0.26, 0.16, 0.06], n_moving)
        lid[:, 1] -= 0.11  # drawer front sticking out of the base
        u = normalize(np.array([0.0, -1.0, 0.0]) if axis is None else axis)
        o = None
        qmax = max_disp_m
    s = np.linspace(0, 1, T)
    q = qmax * (3 * s ** 2 - 2 * s ** 3)  # smooth start/stop
    X = np.concatenate([base, lid])
    moving = np.r_[np.zeros(len(base), bool), np.ones(len(lid), bool)]
    calibs = {}
    centers = {"cam_000": [0.9, 0.1, 0.5], "cam_001": [-0.1, -0.9, 0.6], "cam_002": [0.5, -0.7, 0.5],
               "cam_003": [-0.6, -0.6, 0.6]}
    for cid in cameras:
        calibs[cid] = look_at_camera(cid, centers.get(cid, [0.5, -0.8, 0.6]), [0, 0, 0.08])
    ts = np.arange(T) / fps
    fm = FrameMap(np.arange(T), np.arange(T) * 2, ts, provenance="synthetic")
    rows = {k: [] for k in ("cam", "track", "time", "uv", "uv_conf", "uv_valid", "z", "depth_valid", "depth_edge",
                            "mask_valid", "x_w")}
    tc, tl, tq = [], [], []
    gt_m, gt_x = [], []
    nt = 0
    mov_idx = np.flatnonzero(moving)
    for ci, cid in enumerate(cameras):
        cal = calibs[cid]
        pts = np.arange(len(X))
        if disjoint_moving_cameras:
            half = mov_idx[ci::len(cameras)]
            pts = np.r_[np.flatnonzero(~moving), half]
        if base_only_camera == cid:
            pts = np.flatnonzero(~moving)
        for j in pts:
            qstart = int(rng.integers(1, max(2, T // 2))) if rng.random() < late_tracks_fraction else 0
            Xt = apply_joint(kind, u, o, q, np.repeat(X[j][None], T, 0)) if moving[j] else np.repeat(X[j][None], T, 0)
            xc = (Xt - cal.T_WC[:3, 3]) @ cal.T_WC[:3, :3]
            z = xc[:, 2]
            uv = np.stack([cal.K[0, 0] * xc[:, 0] / z + cal.K[0, 2], cal.K[1, 1] * xc[:, 1] / z + cal.K[1, 2]], 1)
            inside = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < cal.width) & (uv[:, 1] < cal.height)
            present = inside & (np.arange(T) >= qstart) & (rng.random(T) >= dropout)
            present[qstart] = inside[qstart]
            if present.sum() < 3:
                continue
            tt = np.flatnonzero(present)
            uv_n = uv[tt] + rng.normal(0, sigma_px, (len(tt), 2))
            z_n = z[tt] + rng.normal(0, sigma_z, len(tt))
            if outlier_fraction > 0:
                bad = rng.random(len(tt)) < outlier_fraction
                uv_n[bad] += rng.normal(0, 25, (bad.sum(), 2))
                z_n[bad] += rng.normal(0, 0.08, bad.sum())
            dvalid = rng.random(len(tt)) >= depth_invalid_fraction
            xw = np.full((len(tt), 3), np.nan)
            rays = np.c_[(uv_n - cal.K[:2, 2]) / [cal.K[0, 0], cal.K[1, 1]], np.ones(len(tt))]
            xw[dvalid] = (rays[dvalid] * z_n[dvalid, None]) @ cal.T_WC[:3, :3].T + cal.T_WC[:3, 3]
            m = len(tt)
            rows["cam"].append(np.full(m, ci))
            rows["track"].append(np.full(m, nt))
            rows["time"].append(tt)
            rows["uv"].append(uv_n)
            rows["uv_conf"].append(np.ones(m))
            rows["uv_valid"].append(np.ones(m, bool))
            rows["z"].append(np.where(dvalid, z_n, np.nan))
            rows["depth_valid"].append(dvalid)
            rows["depth_edge"].append(np.zeros(m, bool))
            rows["mask_valid"].append(np.ones(m, bool))
            rows["x_w"].append(xw)
            tc.append(ci)
            tl.append(j)
            tq.append(qstart)
            gt_m.append(moving[j])
            gt_x.append(X[j])
            nt += 1
    cat = {k: np.concatenate(v) for k, v in rows.items()}
    obs = TrackObservations(list(cameras), fm, track_camera=np.array(tc), track_local_id=np.array(tl),
                            track_query_time=np.array(tq), provenance=dict(source="synthetic", seed=seed), **cat)
    return SyntheticScene(obs, calibs, kind, u, o, q, np.array(gt_m), np.array(gt_x), seed)
