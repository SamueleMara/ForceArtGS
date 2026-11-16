"""Sparse partial-track observation tables, calibrated cameras and timestamp maps.

Design rules (spec sections 2 and 4.3):

* One row per *observation* (camera, track, time). Missing observations are absent rows, never
  zero-filled entries of a padded ``(N, T, 3)`` array.
* Track IDs are scoped per camera (and per tracker query window). No cross-camera identity is
  ever assumed; two cameras contribute independent material points to one shared joint fit.
* A 2D observation survives invalid depth. Depth is interpolated only from valid samples on the
  same surface; mixed/invalid neighbourhoods are flagged as depth edges.
* Physical cameras are fixed: ``T_WC`` is constant and is never optimized.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .rigid_kinematics import se3_inverse, transform_points


# --------------------------------------------------------------------------- cameras
@dataclass(frozen=True)
class CameraCalib:
    camera_id: str
    K: np.ndarray          # [3,3] pixels, this camera's own intrinsics
    T_WC: np.ndarray       # [4,4] OpenCV camera-to-world, constant
    width: int
    height: int
    calibration_source: str = "unspecified"

    @property
    def T_CW(self) -> np.ndarray:
        return se3_inverse(np.asarray(self.T_WC, float))

    @property
    def center_w(self) -> np.ndarray:
        return np.asarray(self.T_WC, float)[:3, 3]

    @property
    def calib_sha(self) -> str:
        payload = np.concatenate([np.asarray(self.K, float).ravel(), np.asarray(self.T_WC, float).ravel(),
                                  [self.width, self.height]])
        return hashlib.sha256(payload.tobytes()).hexdigest()[:16]

    def scaled(self, sx: float, sy: float | None = None) -> "CameraCalib":
        """Intrinsics for a resized image (pixel-corner convention used by the tracker resize)."""
        sy = sx if sy is None else sy
        K = np.asarray(self.K, float).copy()
        K[0] *= sx
        K[1] *= sy
        return CameraCalib(self.camera_id, K, self.T_WC, int(round(self.width * sx)), int(round(self.height * sy)),
                           self.calibration_source + f" (scaled {sx:g},{sy:g})")

    def to_json(self) -> dict:
        return dict(camera_id=self.camera_id, K=np.asarray(self.K).tolist(), T_WC=np.asarray(self.T_WC).tolist(),
                    width=self.width, height=self.height, calibration_source=self.calibration_source,
                    calib_sha=self.calib_sha, pose_convention="opencv_optical_camera_to_world")


def project(x_w, K, T_CW):
    """World points [...,3] -> (uv [...,2], z_cam [...]) with the OpenCV pinhole model."""
    x_c = transform_points(np.asarray(T_CW, float), np.asarray(x_w, float))
    z = x_c[..., 2]
    zs = np.where(np.abs(z) < 1e-12, 1e-12, z)
    K = np.asarray(K, float)
    u = K[0, 0] * x_c[..., 0] / zs + K[0, 1] * x_c[..., 1] / zs + K[0, 2]
    v = K[1, 1] * x_c[..., 1] / zs + K[1, 2]
    return np.stack([u, v], -1), z


def unproject(uv, z, K, T_WC):
    """``x_W = T_WC (z K^-1 [u, v, 1])``; z is camera-Z depth in metres."""
    uv = np.asarray(uv, float)
    z = np.asarray(z, float)
    Kinv = np.linalg.inv(np.asarray(K, float))
    rays = np.concatenate([uv, np.ones(uv.shape[:-1] + (1,))], -1) @ Kinv.T
    return transform_points(np.asarray(T_WC, float), rays * z[..., None])


def depth_sigma(z, noise, edge=None, cos_incidence=None):
    """Axial depth std: ``(m0 + m2 z^2)``, inflated at depth edges and, when the incidence model
    is enabled, divided by ``max(cos(incidence), floor)`` (RGB-D depth degrades at grazing angles).
    Unknown incidence (NaN) is not inflated."""
    s = noise.sigma_depth_m0 + noise.sigma_depth_m2 * np.asarray(z, float) ** 2
    if edge is not None:
        s = np.where(edge, s * noise.edge_depth_inflation, s)
    if cos_incidence is not None and getattr(noise, "incidence_model", False):
        c = np.nan_to_num(np.asarray(cos_incidence, float), nan=1.0)
        s = s / np.clip(c, noise.incidence_cos_floor, 1.0)
    return s


def incidence_cos_map(depth, K, step=2):
    """|cos| between the viewing ray and the depth-map surface normal (central differences over
    ``step`` px; NaN where any needed sample is missing)."""
    d = np.asarray(depth, float).copy()
    d[~np.isfinite(d) | (d <= 0)] = np.nan
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W].astype(float)
    K = np.asarray(K, float)
    P = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
    du = np.full_like(P, np.nan)
    dv = np.full_like(P, np.nan)
    du[:, step:-step] = P[:, 2 * step:] - P[:, :-2 * step]
    dv[step:-step] = P[2 * step:] - P[:-2 * step]
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    ray = P / np.linalg.norm(P, axis=-1, keepdims=True)
    return np.abs(np.sum(n * ray, -1))


def sample_depth_valid(depth, uv, same_surface_abs_m=0.01, same_surface_rel=0.02):
    """Bilinear depth from *valid same-surface* neighbours only.

    ``depth`` [H,W] metres, 0/NaN/inf = missing; ``uv`` [N,2] in pixel-centre coordinates.
    Returns (z [N] NaN when unavailable, valid [N], edge [N]).  ``edge`` is set when any of the
    four neighbours is missing or lies on a different surface; those samples use only the
    same-surface subset (renormalized weights) and are flagged for inflated uncertainty."""
    depth = np.asarray(depth, float)
    H, W = depth.shape
    uv = np.asarray(uv, float)
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (v >= 0) & (u <= W - 1) & (v <= H - 1) & np.isfinite(u) & np.isfinite(v)
    uc, vc = np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
    x0 = np.clip(np.floor(uc).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(vc).astype(int), 0, H - 2)
    fx, fy = uc - x0, vc - y0
    corners = [(y0, x0, (1 - fx) * (1 - fy)), (y0, x0 + 1, fx * (1 - fy)),
               (y0 + 1, x0, (1 - fx) * fy), (y0 + 1, x0 + 1, fx * fy)]
    vals = np.stack([depth[y, x] for y, x, _ in corners], 1)
    wts = np.stack([w for _, _, w in corners], 1)
    ok = np.isfinite(vals) & (vals > 0)
    # reference: the nearest valid corner (largest weight among valid)
    ref_idx = np.argmax(np.where(ok, wts + 1e-9, -1.0), 1)
    ref = vals[np.arange(len(vals)), ref_idx]
    tol = np.maximum(same_surface_abs_m, same_surface_rel * np.where(np.isfinite(ref), ref, 0.0))
    same = ok & (np.abs(vals - ref[:, None]) <= tol[:, None])
    wsum = np.sum(np.where(same, wts, 0.0), 1)
    z = np.sum(np.where(same, wts * np.where(ok, vals, 0.0), 0.0), 1) / np.where(wsum > 0, wsum, 1.0)
    any_ok = ok.any(1) & inside & (wsum > 1e-6)
    edge = any_ok & ~same.all(1)
    z = np.where(any_ok, z, np.nan)
    return z, any_ok, edge


def sample_mask_nearest(mask, uv):
    mask = np.asarray(mask)
    H, W = mask.shape[:2]
    uv = np.asarray(uv, float)
    x = np.rint(uv[:, 0]).astype(int)
    y = np.rint(uv[:, 1]).astype(int)
    inside = (x >= 0) & (y >= 0) & (x < W) & (y < H)
    out = np.zeros(len(uv), bool)
    out[inside] = mask[y[inside], x[inside]] > 0
    return out


# --------------------------------------------------------------------------- frames / time
@dataclass
class FrameMap:
    """Explicit time map: model time index -> RGB-D output frame -> original capture frame -> stamp."""
    rgbd_frame_index: np.ndarray       # [T] output RGB-D frame id (frame_XXXXXX in rgb/, depth_npy/)
    source_frame_index: np.ndarray     # [T] original capture index (joins the robot Parquet)
    timestamp_s: np.ndarray            # [T] recorded timestamp
    provenance: str = "unspecified"

    def __post_init__(self):
        self.rgbd_frame_index = np.asarray(self.rgbd_frame_index, int)
        self.source_frame_index = np.asarray(self.source_frame_index, int)
        self.timestamp_s = np.asarray(self.timestamp_s, float)

    @property
    def T(self) -> int:
        return len(self.rgbd_frame_index)

    def validate(self) -> "FrameMap":
        n = {len(self.rgbd_frame_index), len(self.source_frame_index), len(self.timestamp_s)}
        if len(n) != 1:
            raise ValueError("frame map arrays have different lengths")
        if self.T and (np.any(np.diff(self.timestamp_s) <= 0) or np.any(np.diff(self.rgbd_frame_index) <= 0)
                       or np.any(np.diff(self.source_frame_index) <= 0)):
            raise ValueError("frame map must be strictly increasing in time, RGB-D index and source index")
        return self

    def time_of_rgbd(self, rgbd_index: int) -> int:
        hit = np.flatnonzero(self.rgbd_frame_index == rgbd_index)
        if len(hit) != 1:
            raise KeyError(f"RGB-D frame {rgbd_index} not in frame map")
        return int(hit[0])

    def to_json(self) -> dict:
        return dict(rgbd_frame_index=self.rgbd_frame_index.tolist(), source_frame_index=self.source_frame_index.tolist(),
                    timestamp_s=self.timestamp_s.tolist(), provenance=self.provenance)


def load_scene_frame_map(scene_root, rgbd_indices) -> FrameMap:
    """Read ``metadata/frame_timestamps.csv`` (frame_index, source_frame_index, timestamp_s)."""
    path = Path(scene_root) / "metadata" / "frame_timestamps.csv"
    rows = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows[int(row["frame_index"])] = (int(row["source_frame_index"]), float(row["timestamp_s"]))
    missing = [i for i in rgbd_indices if i not in rows]
    if missing:
        raise KeyError(f"RGB-D frames {missing[:5]} missing from {path}")
    return FrameMap([int(i) for i in rgbd_indices], [rows[i][0] for i in rgbd_indices],
                    [rows[i][1] for i in rgbd_indices], provenance=str(path)).validate()


def parse_frame_range(text: str) -> list[int]:
    """``"70:128:2"`` -> [70, 72, ..., 128] (inclusive end, matching the prior diagnostic windows)."""
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 2:
        parts.append(1)
    start, end, step = parts
    return list(range(start, end + 1, step))


# --------------------------------------------------------------------------- calibration loading
def _read_colmap_poses(images_txt: Path) -> dict:
    """camera name -> list of c2w from COLMAP images.txt (w2c qvec/tvec, Hamilton wxyz)."""
    poses: dict[str, list] = {}
    lines = [l for l in images_txt.read_text().splitlines() if l and not l.startswith("#")]
    for line in lines[::2]:
        f = line.split()
        qw, qx, qy, qz = map(float, f[1:5])
        t = np.array(list(map(float, f[5:8])))
        R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                      [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                      [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = R, t
        poses.setdefault(Path(f[9]).parts[0], []).append(np.linalg.inv(w2c))
    return poses


def load_scene_cameras(scene_root, cameras) -> tuple[dict, dict]:
    """Per-camera ``CameraCalib`` from ``metadata/cameras.json`` cross-checked against COLMAP.

    Returns (calibs, audit). Raises if a camera moves in COLMAP or disagrees with cameras.json."""
    scene_root = Path(scene_root)
    meta = json.loads((scene_root / "metadata" / "cameras.json").read_text())
    by_name = {c["name"]: c for c in meta["cameras"]}
    colmap_images = scene_root / "colmap" / "sparse" / "0" / "images.txt"
    colmap = _read_colmap_poses(colmap_images) if colmap_images.is_file() else {}
    calibs, audit = {}, {}
    for cam in cameras:
        if cam not in by_name:
            raise KeyError(f"{cam} not in {scene_root / 'metadata/cameras.json'}")
        c = by_name[cam]
        if c.get("pose_convention") != "opencv_optical_camera_to_world":
            raise ValueError(f"{cam}: unsupported pose convention {c.get('pose_convention')}")
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
        T = np.asarray(c["pose_matrix_4x4"], float)
        entry = dict(coordinate_frame=c.get("coordinate_frame"), colmap_checked=False)
        if cam in colmap:
            stack = np.stack(colmap[cam])
            if not np.allclose(stack, stack[0], atol=1e-6):
                raise ValueError(f"{cam} moves in COLMAP images.txt; fixed calibrated cameras are required")
            dR = np.degrees(np.arccos(np.clip((np.trace(stack[0][:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
            dt = float(np.linalg.norm(stack[0][:3, 3] - T[:3, 3]))
            entry.update(colmap_checked=True, colmap_rot_diff_deg=float(dR), colmap_trans_diff_m=dt)
            if dR > 0.05 or dt > 1e-3:
                raise ValueError(f"{cam}: cameras.json and COLMAP disagree ({dR:.3f} deg, {dt * 1000:.2f} mm)")
        calibs[cam] = CameraCalib(cam, K, T, int(c["width"]), int(c["height"]),
                                  calibration_source=f"{scene_root}/metadata/cameras.json[{cam}]")
        audit[cam] = entry
    return calibs, audit


# --------------------------------------------------------------------------- observation table
OBS_FIELDS = ("cam", "track", "time", "uv", "uv_conf", "uv_valid", "z", "depth_valid", "depth_edge",
              "mask_valid", "x_w", "depth_cos")


@dataclass
class TrackObservations:
    camera_ids: list
    frame_map: FrameMap
    cam: np.ndarray            # [M] index into camera_ids
    track: np.ndarray          # [M] global track index 0..N-1
    time: np.ndarray           # [M] model time index
    uv: np.ndarray             # [M,2] pixels at the camera's full calibrated resolution
    uv_conf: np.ndarray        # [M]
    uv_valid: np.ndarray       # [M]
    z: np.ndarray              # [M] camera-Z metres, NaN if unavailable
    depth_valid: np.ndarray    # [M]
    depth_edge: np.ndarray     # [M]
    mask_valid: np.ndarray     # [M] inside the whole-object mask (NOT a part label)
    x_w: np.ndarray            # [M,3] world point, NaN unless depth_valid
    track_camera: np.ndarray   # [N]
    track_local_id: np.ndarray  # [N] per-camera scope id (query window * stride + pixel id)
    track_query_time: np.ndarray  # [N]
    provenance: dict = field(default_factory=dict)
    depth_cos: np.ndarray | None = None  # [M] |cos| of depth incidence angle, NaN unknown

    def __post_init__(self):
        M = len(self.track)
        if self.depth_cos is None:
            self.depth_cos = np.full(M, np.nan)
        for name in OBS_FIELDS:
            arr = getattr(self, name)
            if len(arr) != M:
                raise ValueError(f"observation field {name} has length {len(arr)} != {M}")
        self.cam = np.asarray(self.cam, int)
        self.track = np.asarray(self.track, int)
        self.time = np.asarray(self.time, int)

    @property
    def n_tracks(self) -> int:
        return len(self.track_camera)

    @property
    def n_obs(self) -> int:
        return len(self.track)

    @property
    def has_3d(self) -> np.ndarray:
        return self.depth_valid & self.mask_valid & np.isfinite(self.x_w).all(1)

    def select_obs(self, keep: np.ndarray) -> "TrackObservations":
        keep = np.asarray(keep, bool)
        kw = {name: getattr(self, name)[keep] for name in OBS_FIELDS}
        return TrackObservations(self.camera_ids, self.frame_map, track_camera=self.track_camera,
                                 track_local_id=self.track_local_id, track_query_time=self.track_query_time,
                                 provenance=dict(self.provenance), **kw)

    def select_tracks(self, track_keep: np.ndarray) -> tuple["TrackObservations", np.ndarray]:
        """Keep a subset of tracks and re-index them; returns (obs, old_index_of_new)."""
        track_keep = np.asarray(track_keep, bool)
        old = np.flatnonzero(track_keep)
        new_of_old = -np.ones(self.n_tracks, int)
        new_of_old[old] = np.arange(len(old))
        obs_keep = track_keep[self.track]
        kw = {name: getattr(self, name)[obs_keep] for name in OBS_FIELDS}
        kw["track"] = new_of_old[kw["track"]]
        return TrackObservations(self.camera_ids, self.frame_map, track_camera=self.track_camera[old],
                                 track_local_id=self.track_local_id[old], track_query_time=self.track_query_time[old],
                                 provenance=dict(self.provenance), **kw), old

    def per_track_counts(self):
        n2d = np.bincount(self.track[self.uv_valid], minlength=self.n_tracks)
        n3d = np.bincount(self.track[self.has_3d], minlength=self.n_tracks)
        return n2d, n3d

    def summary(self) -> dict:
        n2d, n3d = self.per_track_counts()
        out = dict(n_tracks=int(self.n_tracks), n_obs=int(self.n_obs), T=int(self.frame_map.T),
                   cameras=list(self.camera_ids), per_camera={})
        for ci, cid in enumerate(self.camera_ids):
            tm = self.track_camera == ci
            om = self.cam == ci
            out["per_camera"][cid] = dict(
                tracks=int(tm.sum()), obs=int(om.sum()), obs_uv_valid=int((om & self.uv_valid).sum()),
                obs_depth_valid=int((om & self.depth_valid).sum()), obs_depth_edge=int((om & self.depth_edge).sum()),
                obs_3d=int((om & self.has_3d).sum()), tracks_with_3d_anchor=int((tm & (n3d > 0)).sum()),
                tracks_2d_only=int((tm & (n3d == 0) & (n2d > 0)).sum()),
                tracks_absent_at_t0=int((tm & (self.track_query_time > 0)).sum()))
        return out

    # ----------------------------------------------------------------- IO
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self, name) for name in OBS_FIELDS}
        arrays.update(track_camera=self.track_camera, track_local_id=self.track_local_id,
                      track_query_time=self.track_query_time,
                      fm_rgbd=self.frame_map.rgbd_frame_index, fm_source=self.frame_map.source_frame_index,
                      fm_time=self.frame_map.timestamp_s)
        np.savez_compressed(path, **arrays)
        meta = dict(camera_ids=list(self.camera_ids), frame_map_provenance=self.frame_map.provenance,
                    provenance=self.provenance, units=dict(uv="px (full calibrated resolution)", z="m camera-Z",
                                                           x_w="m world", time="model time index"))
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))

    @classmethod
    def load(cls, path) -> "TrackObservations":
        path = Path(path)
        z = np.load(path, allow_pickle=False)
        meta = json.loads(path.with_suffix(".json").read_text())
        fm = FrameMap(z["fm_rgbd"], z["fm_source"], z["fm_time"], provenance=meta.get("frame_map_provenance", ""))
        return cls(meta["camera_ids"], fm, track_camera=z["track_camera"], track_local_id=z["track_local_id"],
                   track_query_time=z["track_query_time"], provenance=meta.get("provenance", {}),
                   **{name: z[name] for name in OBS_FIELDS if name in z.files})


def concatenate_observations(parts: list) -> TrackObservations:
    """Merge per-camera tables into one multiview table (track IDs stay disjoint per camera)."""
    if not parts:
        raise ValueError("no observation tables")
    fm = parts[0].frame_map
    for p in parts[1:]:
        if not (np.array_equal(p.frame_map.rgbd_frame_index, fm.rgbd_frame_index)
                and np.allclose(p.frame_map.timestamp_s, fm.timestamp_s)):
            raise ValueError("observation tables use different frame maps; refuse to merge unsynchronized cameras")
    cams = []
    for p in parts:
        for c in p.camera_ids:
            if c in cams:
                raise ValueError(f"camera {c} appears in two tables")
            cams.append(c)
    out = {name: [] for name in OBS_FIELDS}
    tc, tl, tq, offset = [], [], [], 0
    for p in parts:
        cam_offset = cams.index(p.camera_ids[0])
        for name in OBS_FIELDS:
            arr = getattr(p, name)
            if name == "track":
                arr = arr + offset
            elif name == "cam":
                arr = arr + cam_offset
            out[name].append(arr)
        tc.append(p.track_camera + cam_offset)
        tl.append(p.track_local_id)
        tq.append(p.track_query_time)
        offset += p.n_tracks
    prov = {cid: p.provenance for p in parts for cid in p.camera_ids}
    return TrackObservations(cams, fm, track_camera=np.concatenate(tc), track_local_id=np.concatenate(tl),
                             track_query_time=np.concatenate(tq), provenance=prov,
                             **{k: np.concatenate(v) for k, v in out.items()})


def build_from_dense_tracks(camera: CameraCalib, frame_map: FrameMap, uv, vis, depths, masks, query_time: int,
                            track_ids=None, vis_threshold=0.5, require_query_in_mask=True,
                            reject_edge_depth=True, provenance=None) -> TrackObservations:
    """Convert tracker output for one camera into an observation table.

    ``uv`` [T,N,2] full-resolution pixels for model times ``query_time..query_time+T-1``,
    ``vis`` [T,N] confidence, ``depths``/``masks`` lists of [H,W] for the same times.
    Observations below ``vis_threshold`` are dropped (missing), not zero-filled. Tracks whose query
    pixel is outside the whole-object mask are not created (the robot/background is never tracked
    as object). Depth edges keep their 2D observation; with ``reject_edge_depth`` their depth is
    marked invalid but the sampled value is retained for audit."""
    uv = np.asarray(uv, float)
    vis = np.asarray(vis, float)
    Tw, N = vis.shape
    keep_track = np.ones(N, bool)
    if require_query_in_mask:
        keep_track &= sample_mask_nearest(masks[0], uv[0])
    ids = np.arange(N) if track_ids is None else np.asarray(track_ids)
    ids = ids[keep_track]
    uv, vis = uv[:, keep_track], vis[:, keep_track]
    rows = dict(cam=[], track=[], time=[], uv=[], uv_conf=[], uv_valid=[], z=[], depth_valid=[], depth_edge=[],
                mask_valid=[], x_w=[], depth_cos=[])
    n = len(ids)
    for k in range(Tw):
        ok = (vis[k] >= vis_threshold) & np.isfinite(uv[k]).all(1)
        idx = np.flatnonzero(ok)
        z, dv, edge = sample_depth_valid(depths[k], uv[k, idx])
        mv = sample_mask_nearest(masks[k], uv[k, idx])
        dvalid = dv & ~(edge & reject_edge_depth)
        x = np.full((len(idx), 3), np.nan)
        good = dvalid & mv
        x[good] = unproject(uv[k, idx[good]], z[good], camera.K, camera.T_WC)
        rows["cam"].append(np.zeros(len(idx), int))
        rows["track"].append(idx)
        rows["time"].append(np.full(len(idx), query_time + k))
        rows["uv"].append(uv[k, idx])
        rows["uv_conf"].append(vis[k, idx])
        rows["uv_valid"].append(np.ones(len(idx), bool))
        rows["z"].append(z)
        rows["depth_valid"].append(dvalid)
        rows["depth_edge"].append(edge)
        rows["mask_valid"].append(mv)
        rows["x_w"].append(x)
        cmap = incidence_cos_map(depths[k], camera.K)
        px = np.clip(np.rint(uv[k, idx]).astype(int), 0, [cmap.shape[1] - 1, cmap.shape[0] - 1])
        rows["depth_cos"].append(cmap[px[:, 1], px[:, 0]])
    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in rows.items()}
    cat["x_w"] = cat["x_w"].reshape(-1, 3)
    cat["uv"] = cat["uv"].reshape(-1, 2)
    return TrackObservations([camera.camera_id], frame_map, track_camera=np.zeros(n, int),
                             track_local_id=np.asarray(ids, int), track_query_time=np.full(n, query_time, int),
                             provenance=provenance or {}, **cat)


def _segment_distance(X, a, b):
    """Distance of points X [M,3] to segments a [M,3] -> b [M,3]."""
    ab = b - a
    s = np.clip(np.einsum("mi,mi->m", X - a, ab) / np.maximum(np.einsum("mi,mi->m", ab, ab), 1e-12), 0, 1)
    return np.linalg.norm(X - (a + s[:, None] * ab), axis=1)


def exclude_tool_observations(obs: TrackObservations, calibs: dict, p_tcp_w, p_ee_w, radius_m=0.03, extend_m=0.10,
                              tip_offset_m=0.015):
    """Remove the measured robot tool from the object observations (spec 16.9).

    The tool is the segment from ``tip_offset_m`` behind the TCP to ``extend_m`` beyond the EE,
    inflated by ``radius_m``. The offset keeps the object surface the tip touches (a pointed tip
    has no volume at the contact; 1.5 cm = documented tip uncertainty). Tracks whose *query* observation lies on the tool are robot tracks and are
    dropped; later observations that come within the tool zone keep their 2D position only if they
    are not on the tool in 3D (depth-valid) or in the image (projected segment, 2D-only)."""
    p_tcp_w, p_ee_w = np.asarray(p_tcp_w, float), np.asarray(p_ee_w, float)
    d = p_ee_w - p_tcp_w
    dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    p_far = p_ee_w + extend_m * dn
    p_tcp_w = p_tcp_w + tip_offset_m * dn
    t = obs.time
    on_tool = np.zeros(obs.n_obs, bool)
    h = np.isfinite(obs.x_w).all(1)
    on_tool[h] = _segment_distance(obs.x_w[h], p_tcp_w[t[h]], p_far[t[h]]) < radius_m
    # 2D test for depth-less observations: distance to the projected tool segment
    for ci, cid in enumerate(obs.camera_ids):
        cal = calibs[cid]
        m = (obs.cam == ci) & ~h
        if not m.any():
            continue
        a_uv, za = project(p_tcp_w[t[m]], cal.K, cal.T_CW)
        b_uv, zb = project(p_far[t[m]], cal.K, cal.T_CW)
        r_px = cal.K[0, 0] * radius_m / np.maximum(np.minimum(za, zb), 0.05)
        A = np.c_[a_uv, np.zeros(len(a_uv))]
        B = np.c_[b_uv, np.zeros(len(b_uv))]
        on_tool[m] = _segment_distance(np.c_[obs.uv[m], np.zeros(m.sum())], A, B) < r_px
    query_obs = obs.time == obs.track_query_time[obs.track]
    robot_tracks = np.zeros(obs.n_tracks, bool)
    robot_tracks[obs.track[query_obs & on_tool]] = True
    keep_obs = ~on_tool & ~robot_tracks[obs.track]
    out = obs.select_obs(keep_obs)
    out, _ = out.select_tracks(~robot_tracks)
    report = dict(radius_m=radius_m, extend_m=extend_m, tip_offset_m=tip_offset_m, robot_tracks_dropped=int(robot_tracks.sum()),
                  tool_observations_dropped=int((on_tool & ~robot_tracks[obs.track]).sum()),
                  source="measured TCP/EE kinematics (not an oracle object label)")
    out.provenance = dict(obs.provenance, tool_exclusion=report)
    return out, report


def from_window_track_dumps(dump_dir, camera: CameraCalib, frame_map: FrameMap, clip_offset: int = 0,
                            use_prefilter_keep: bool = False) -> TrackObservations:
    """Adapter for ``samuele/dump_window_tracks.py`` archives (read-only).

    Each window's tracks are identified by their frame-0 tracking-resolution pixel; observations at
    window frame k exist only where the dump kept the pair (0,k) (visible + in mask in both). The
    dump's depth comes from the legacy ``-inf`` bilinear sampler, so it is labelled as such and
    flagged in provenance. ``clip_offset`` maps dump clip frame c to model time c - clip_offset."""
    dump_dir = Path(dump_dir)
    files = sorted(dump_dir.glob("tracks_w*.npz"))
    if not files:
        raise FileNotFoundError(f"no tracks_w*.npz in {dump_dir}")
    rows = {k: [] for k in OBS_FIELDS}
    tl, tq, n_off = [], [], 0
    for f in files:
        z = np.load(f, allow_pickle=False)
        K = np.asarray(z["K"], float)
        scale = float(z["scale"])
        frames = np.asarray(z["frames"], int) - clip_offset
        ks = sorted(int(k.split("_")[-1]) for k in z.files if k.startswith("pre_xyz_"))
        pix = {}
        for k in ks:
            yx = z[f"pre_yx_{k}"]
            for key in map(tuple, yx):
                pix.setdefault(key, len(pix))
        n = len(pix)
        seen0 = np.zeros(n, bool)
        for k in ks:
            xyz = np.asarray(z[f"pre_xyz_{k}"], float)
            yx = z[f"pre_yx_{k}"]
            keep = np.asarray(z[f"keep_{k}"], bool) if use_prefilter_keep else np.ones(len(xyz), bool)
            tid = np.array([pix[tuple(p)] for p in yx], int)[keep]
            xyz = xyz[keep]
            for slot, frame in ((0, frames[0]), (1, frames[k])):
                if slot == 0:
                    sel = ~seen0[tid]
                    seen0[tid] = True
                else:
                    sel = np.ones(len(tid), bool)
                pc = xyz[sel, slot]
                finite = np.isfinite(pc).all(1) & (pc[:, 2] > 0)
                uv_t = np.where(finite[:, None], (pc[:, :2] / np.where(finite, pc[:, 2], 1)[:, None]) * [K[0, 0], K[1, 1]]
                                + [K[0, 2], K[1, 2]], np.nan)
                # tracking-resolution pixel centres -> full resolution (resize by `scale`)
                uv_full = (uv_t + 0.5) / scale - 0.5
                xw = np.full_like(pc, np.nan)
                xw[finite] = transform_points(np.asarray(camera.T_WC, float), pc[finite])
                m = len(pc)
                rows["cam"].append(np.zeros(m, int))
                rows["track"].append(tid[sel] + n_off)
                rows["time"].append(np.full(m, frame))
                rows["uv"].append(uv_full)
                rows["uv_conf"].append(np.ones(m))
                rows["uv_valid"].append(finite)
                rows["z"].append(np.where(finite, pc[:, 2], np.nan))
                rows["depth_valid"].append(finite)
                rows["depth_edge"].append(np.zeros(m, bool))
                rows["mask_valid"].append(np.ones(m, bool))
                rows["x_w"].append(xw)
                rows["depth_cos"].append(np.full(m, np.nan))
        tl.append(np.arange(n) + 100000 * int(frames[0] + clip_offset))
        tq.append(np.full(n, frames[0]))
        n_off += n
    cat = {k: np.concatenate(v) for k, v in rows.items()}
    valid_time = (cat["time"] >= 0) & (cat["time"] < frame_map.T)
    cat = {k: v[valid_time] for k, v in cat.items()}
    obs = TrackObservations([camera.camera_id], frame_map, track_camera=np.zeros(n_off, int),
                            track_local_id=np.concatenate(tl), track_query_time=np.concatenate(tq),
                            provenance=dict(source=str(dump_dir), adapter="from_window_track_dumps",
                                            depth_sampler="legacy bilinear with -inf invalid (tracker_toolkit."
                                                          "get_4d_traj_maps); edges NOT flagged",
                                            prefilter_keep_applied=use_prefilter_keep), **cat)
    return obs
