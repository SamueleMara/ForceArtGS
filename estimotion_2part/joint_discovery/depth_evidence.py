"""Candidate-independent depth evidence with provenance codes (Stage B1).

A depth sample contributes to the likelihood only when independent evidence says it belongs to
the tracked material point's *own, front* surface. Nothing here looks at a joint hypothesis; all
tests use only the observation table, the raw depth maps and the measured robot kinematics. The
2D observation always survives (only ``depth_valid`` is cleared), and every decision is recorded.

Rules (thresholds are physical tolerances declared up front, never tuned on a scene reference):

``invalid_depth``                 no finite depth / sampler found no valid neighbour
``depth_edge_ambiguous``          2x2 sampler mixed surfaces (existing edge flag)
``outside_object_mask``           pixel outside the whole-object mask
``behind_depth_discontinuity``    a surface >= ``front_gap_m`` nearer the camera lies within
                                  ``front_window_px``: the tracked pixel may sit on the far side
                                  of a silhouette (background / wrong-surface association)
``depth_jump``                    |z_t - median(z_{t-1}, z_{t+1})| > ``jump_m`` along the track
                                  (a surface switch, not rigid motion between adjacent frames)
``inconsistent_local_rigidity``   the 3D distance to same-camera neighbour tracks deviates from its
                                  temporal median by > ``rigidity_tol_m`` for most neighbours
                                  (material points on one rigid part keep their distances)
``grazing_incidence``             |cos(incidence)| < ``min_incidence_cos`` (off unless set)
``robot_contamination``           measured point within ``tool_margin_m`` of the measured tool
``unverified``                    too few neighbours to test rigidity (policy: drop or keep)
``valid_surface``                 passed every applicable test
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CODES = ("valid_surface", "invalid_depth", "depth_edge_ambiguous", "outside_object_mask", "behind_depth_discontinuity",
         "depth_jump", "inconsistent_local_rigidity", "grazing_incidence", "robot_contamination", "unverified")
CODE = {c: i for i, c in enumerate(CODES)}


@dataclass
class DepthEvidenceConfig:
    front_gap_m: float = 0.03
    front_window_px: int = 2
    jump_m: float = 0.03
    rigidity_tol_m: float = 0.01
    rigidity_sigma_k: float = 3.0
    neighbour_radius_m: float = 0.04
    max_neighbours: int = 10
    min_neighbours: int = 3
    min_incidence_cos: float = 0.0
    tool_margin_m: float = 0.05
    unverified_policy: str = "drop"      # 'drop' (never promote unverified depth) | 'keep'

    def validate(self):
        if self.unverified_policy not in ("drop", "keep"):
            raise ValueError("unverified_policy must be 'drop' or 'keep'")
        return self


def _front_gap(scene, obs, ci, cid, window):
    from scipy.ndimage import minimum_filter
    gap = np.full(obs.n_obs, np.nan)
    for t, f in enumerate(obs.frame_map.rgbd_frame_index):
        m = (obs.cam == ci) & (obs.time == t)
        if not m.any():
            continue
        p = Path(scene) / "depth_npy" / cid / f"frame_{int(f):06d}.npy"
        if not p.is_file():
            raise FileNotFoundError(f"missing depth map {p}; depth evidence cannot be assessed")
        d = np.load(p).astype(float)
        d[~np.isfinite(d) | (d <= 0)] = np.inf
        dmin = minimum_filter(d, size=2 * window + 1)
        H, W = d.shape
        px = np.clip(np.rint(obs.uv[m]).astype(int), 0, [W - 1, H - 1])
        g = obs.z[m] - dmin[px[:, 1], px[:, 0]]
        gap[np.flatnonzero(m)] = np.where(np.isfinite(g), g, np.nan)
    return gap


def _dense(obs, ci):
    tracks = np.flatnonzero(obs.track_camera == ci)
    loc = -np.ones(obs.n_tracks, int)
    loc[tracks] = np.arange(len(tracks))
    T = obs.frame_map.T
    X = np.full((len(tracks), T, 3), np.nan)
    row = -np.ones((len(tracks), T), int)
    m = (obs.cam == ci) & obs.has_3d
    X[loc[obs.track[m]], obs.time[m]] = obs.x_w[m]
    row[loc[obs.track[m]], obs.time[m]] = np.flatnonzero(m)
    return tracks, X, row


def local_rigidity(obs, ci, cfg: DepthEvidenceConfig, sigma_z):
    """Per-observation median neighbour-distance deviation [m] and neighbour count (same camera)."""
    from scipy.spatial import cKDTree
    tracks, X, row = _dense(obs, ci)
    dev = np.full(obs.n_obs, np.nan)
    cnt = np.zeros(obs.n_obs, int)
    if len(tracks) < 2:
        return dev, cnt
    with np.errstate(all="ignore"):
        ok_rows = np.isfinite(X).all(2).any(1)
        ref = np.full((len(tracks), 3), np.nan)
        ref[ok_rows] = np.nanmedian(X[ok_rows], 1)     # robust per-track position for neighbour search
    ok = np.isfinite(ref).all(1)
    if ok.sum() < 2:
        return dev, cnt
    idx_ok = np.flatnonzero(ok)
    tree = cKDTree(ref[ok])
    dd, nb = tree.query(ref[ok], k=min(cfg.max_neighbours + 1, ok.sum()), distance_upper_bound=cfg.neighbour_radius_m)
    nb = np.where(np.isfinite(dd), nb, -1)[:, 1:]
    for a, i in enumerate(idx_ok):
        js = idx_ok[nb[a][nb[a] >= 0]]
        if len(js) == 0:
            continue
        D = np.linalg.norm(X[i][None] - X[js], axis=2)      # [k, T]
        med = np.nanmedian(D, 1, keepdims=True)
        E = np.abs(D - med)
        valid = np.isfinite(E)
        n = valid.sum(0)
        has = np.isfinite(X[i]).all(1)
        e = np.where(n > 0, np.nanmedian(np.where(valid, E, np.nan), 0), np.nan) if valid.any() else np.full(X.shape[1], np.nan)
        r = row[i][has]
        dev[r] = e[has]
        cnt[r] = n[has]
    return dev, cnt


def assess_depth_evidence(obs, cfg: DepthEvidenceConfig, scene=None, sigma_z=None, tool_distance_m=None):
    """Return (code [M] int8, details dict). ``sigma_z`` [M] nominal depth std (for the rigidity tol)."""
    cfg.validate()
    code = np.full(obs.n_obs, CODE["valid_surface"], np.int8)
    z = obs.z
    code[~np.isfinite(z) | ~(obs.depth_valid | obs.depth_edge)] = CODE["invalid_depth"]
    code[(code == 0) & obs.depth_edge & ~obs.depth_valid] = CODE["depth_edge_ambiguous"]
    code[(code == 0) & ~obs.mask_valid] = CODE["outside_object_mask"]
    det = dict(config=asdict(cfg))
    if scene is not None:
        gap = np.full(obs.n_obs, np.nan)
        for ci, cid in enumerate(obs.camera_ids):
            g = _front_gap(scene, obs, ci, cid, cfg.front_window_px)
            gap = np.where(obs.cam == ci, g, gap)
        code[(code == 0) & (gap > cfg.front_gap_m)] = CODE["behind_depth_discontinuity"]
        det["front_gap_assessed"] = True
    else:
        det["front_gap_assessed"] = False
    # temporal jumps along each track
    o = np.lexsort((obs.time, obs.track))
    tr, zz = obs.track[o], np.where(code[o] == 0, z[o], np.nan)
    prev = np.r_[np.nan, np.where(tr[1:] == tr[:-1], zz[:-1], np.nan)]
    nxt = np.r_[np.where(tr[:-1] == tr[1:], zz[1:], np.nan), np.nan]
    ref = np.nanmedian(np.stack([prev, nxt]), 0) if len(zz) else zz
    jump = np.isfinite(zz) & np.isfinite(ref) & (np.abs(zz - ref) > cfg.jump_m)
    jj = np.zeros(obs.n_obs, bool)
    jj[o] = jump
    code[(code == 0) & jj] = CODE["depth_jump"]
    if cfg.min_incidence_cos > 0:
        c = obs.depth_cos
        code[(code == 0) & np.isfinite(c) & (c < cfg.min_incidence_cos)] = CODE["grazing_incidence"]
    if tool_distance_m is not None:
        code[(code == 0) & (np.asarray(tool_distance_m) < cfg.tool_margin_m)] = CODE["robot_contamination"]
    # local rigidity on the still-valid samples
    obs_v = obs.select_obs(np.ones(obs.n_obs, bool))
    obs_v.depth_valid = obs.depth_valid & (code == 0)
    sz = np.full(obs.n_obs, 0.0) if sigma_z is None else np.asarray(sigma_z, float)
    dev = np.full(obs.n_obs, np.nan)
    cnt = np.zeros(obs.n_obs, int)
    for ci in range(len(obs.camera_ids)):
        d, n = local_rigidity(obs_v, ci, cfg, sz)
        m = obs.cam == ci
        dev[m], cnt[m] = d[m], n[m]
    tol = np.maximum(cfg.rigidity_tol_m, cfg.rigidity_sigma_k * np.sqrt(2) * sz)
    code[(code == 0) & (cnt >= cfg.min_neighbours) & (dev > tol)] = CODE["inconsistent_local_rigidity"]
    code[(code == 0) & (cnt < cfg.min_neighbours)] = CODE["unverified"]
    det.update(counts=counts(code, obs), rigidity_median_dev_mm=float(np.nanmedian(dev) * 1000) if np.isfinite(dev).any() else None)
    return code, det


def counts(code, obs) -> dict:
    out = {}
    for ci, cid in enumerate(obs.camera_ids):
        m = obs.cam == ci
        out[cid] = {c: int((code[m] == i).sum()) for i, c in enumerate(CODES) if (code[m] == i).any()}
    return out


def apply_depth_evidence(obs, code, policy="drop"):
    """New observation table whose ``depth_valid`` keeps only supported depth (2D untouched)."""
    keep = code == CODE["valid_surface"]
    if policy == "keep":
        keep |= code == CODE["unverified"]
    out = obs.select_obs(np.ones(obs.n_obs, bool))
    out.depth_valid = obs.depth_valid & keep
    out.x_w = np.where(out.depth_valid[:, None], obs.x_w, np.nan)
    out.provenance = dict(obs.provenance, depth_evidence=dict(policy=policy, n_depth_kept=int(out.depth_valid.sum()),
                                                              n_depth_before=int(obs.depth_valid.sum())))
    return out
