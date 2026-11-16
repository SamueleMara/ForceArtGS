#!/usr/bin/env python3
"""Shared TCP / contact evidence adapter for FreeArtGS tracking (CPU only).

Joins one converted FreeArt camera sequence to the measured robot TCP of its
source real scene, projects the calibrated TCP (and an optional calibrated
tool-contact point) into that fixed camera and records explicit contact state.

Join chain (never nominal FPS multiplication):

    FreeArt metadata/freeart_conversion.json frame_map[i].source_stem  ("frame_000123")
      -> real scene output frame 123 (metadata/real_conversion.json output_frames)
      -> real_conversion source_frames / poses_rgbd.npz source_frame_index
      -> raw Parquet frame_index row.

Conventions:
    * cameras.json ``pose_matrix_4x4`` is OpenCV optical camera -> base (T_base_camera).
    * T_base_tcp = T_base_ee @ T_ee_tcp is applied exactly once.  An archive that already
      holds ``tcp_transform`` is used as is.
    * Image coordinates are (u=x, v=y) with pixel centres on integers (OpenCV/K convention).
    * Invalid values are NaN; booleans say why.  Nothing is filled with zeros.

This module is not a physical prior: it only aligns and reports evidence.  The TCP is a
robot point, never an object material point.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SAMUELE_DIR = Path(__file__).resolve().parent
ARTIFACT_NPZ = "tcp_contact.npz"
ARTIFACT_MANIFEST = "manifest.json"
SCHEMA_VERSION = 1
STEM_PATTERN = re.compile(r"^frame_(\d+)$")


# ----------------------------------------------------------------------------- helpers


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rigid(matrix: Any, label: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{label}: expected a finite 4x4 transform")
    rotation = value[:3, :3]
    if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
        raise ValueError(f"{label}: invalid rigid transform")
    return value


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply (N,4,4) or (4,4) transforms to (N,3) or (3,) points."""
    T = np.asarray(T, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if T.ndim == 2:
        return points @ T[:3, :3].T + T[:3, 3]
    return np.einsum("nij,nj->ni", T[:, :3, :3], points) + T[:, :3, 3]


def pinhole(K: np.ndarray, points_cam: np.ndarray, image_size_wh: tuple[int, int] | None = None):
    """Project camera points.  Returns (uv, valid) with NaN for invalid entries.

    Valid requires finite input, positive camera Z and, if a size is given, a pixel
    inside [0, W) x [0, H).
    """
    K = np.asarray(K, dtype=np.float64)
    points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    z = points_cam[:, 2]
    finite = np.isfinite(points_cam).all(axis=1)
    positive = finite & (z > 0)
    uv = np.full((len(points_cam), 2), np.nan)
    safe_z = np.where(positive, z, 1.0)
    uv_all = np.stack([K[0, 0] * points_cam[:, 0] / safe_z + K[0, 2],
                       K[1, 1] * points_cam[:, 1] / safe_z + K[1, 2]], axis=1)
    uv[positive] = uv_all[positive]
    valid = positive.copy()
    if image_size_wh is not None:
        width, height = image_size_wh
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        valid &= np.where(np.isfinite(uv).all(axis=1), inside, False)
    uv[~valid] = np.nan
    return uv, valid


def project_tcp(T_base_tcp: np.ndarray, T_base_camera: np.ndarray, K: np.ndarray,
                image_size_wh: tuple[int, int] | None = None,
                point_local_tool: np.ndarray | None = None) -> np.ndarray:
    """Project the TCP origin (or a TCP-local point) into a fixed camera.

    ``T_base_camera`` is camera-to-base; it is inverted here.  Returns (N,2) pixel
    coordinates (u=x, v=y) with NaN where Z <= 0 or outside the optional image size.
    """
    T_base_tcp = np.asarray(T_base_tcp, dtype=np.float64).reshape(-1, 4, 4)
    T_camera_base = np.linalg.inv(rigid(T_base_camera, "T_base_camera"))
    local = np.zeros(3) if point_local_tool is None else np.asarray(point_local_tool, dtype=np.float64)
    p_base = transform_points(T_base_tcp, np.repeat(local[None], len(T_base_tcp), axis=0))
    p_cam = transform_points(T_camera_base, p_base)
    uv, _ = pinhole(K, p_cam, image_size_wh)
    return uv


# ----------------------------------------------------------------------------- frame mapping


def _stem_frame(stem: str) -> int:
    match = STEM_PATTERN.match(str(stem))
    if not match:
        raise ValueError(f"FreeArt source_stem {stem!r} is not a real-scene frame_NNNNNN stem")
    return int(match.group(1))


def map_freeart_frames(frame_map: list, pose_archive: dict, real_conversion: dict | None = None) -> list:
    """Map every FreeArt local frame to its pose-archive row and raw source frame index.

    ``pose_archive`` must contain ``frame_index`` (real-scene output frame IDs, RGB-D aligned)
    and ``source_frame_index`` (raw capture/Parquet frame IDs).  When ``real_conversion`` is
    supplied, its ``output_frames -> source_frames`` mapping is cross-checked.

    Returns one dict per FreeArt frame: local_index, output_stem, source_stem,
    output_frame_index, source_frame_index, archive_row.
    """
    if not frame_map:
        raise ValueError("FreeArt frame_map is empty")
    archive_frames = np.asarray(pose_archive["frame_index"]).astype(np.int64)
    archive_sources = np.asarray(pose_archive["source_frame_index"]).astype(np.int64)
    if archive_frames.ndim != 1 or archive_frames.shape != archive_sources.shape:
        raise ValueError("Pose archive frame_index/source_frame_index must be equal-length vectors")
    if len(np.unique(archive_frames)) != len(archive_frames):
        raise ValueError("Pose archive frame_index values must be unique")
    if np.any(np.diff(archive_sources) <= 0):
        raise ValueError("Pose archive source_frame_index must be strictly increasing (chronological)")
    row_of = {int(frame): row for row, frame in enumerate(archive_frames)}
    conversion_lookup = None
    if real_conversion is not None:
        sources = list(real_conversion["source_frames"])
        outputs = list(real_conversion.get("output_frames", range(len(sources))))
        if len(sources) != len(outputs):
            raise ValueError("real_conversion output_frames/source_frames lengths differ")
        conversion_lookup = {int(o): int(s) for o, s in zip(outputs, sources)}

    mapped = []
    seen_outputs = set()
    for local_index, row in enumerate(frame_map):
        expected_stem = f"{local_index:05d}"
        if str(row.get("output_stem")) != expected_stem:
            raise ValueError(f"frame_map[{local_index}] output_stem {row.get('output_stem')!r} != {expected_stem!r}")
        output_frame = _stem_frame(row["source_stem"])
        if output_frame in seen_outputs:
            raise ValueError(f"Duplicate source frame {row['source_stem']} in FreeArt frame_map")
        seen_outputs.add(output_frame)
        if output_frame not in row_of:
            raise ValueError(f"Source frame {row['source_stem']} absent from pose archive")
        archive_row = row_of[output_frame]
        source_index = int(archive_sources[archive_row])
        if conversion_lookup is not None:
            if output_frame not in conversion_lookup:
                raise ValueError(f"Source frame {row['source_stem']} absent from real_conversion output_frames")
            if conversion_lookup[output_frame] != source_index:
                raise ValueError(
                    f"real_conversion maps output frame {output_frame} to source {conversion_lookup[output_frame]}, "
                    f"pose archive to {source_index}")
        mapped.append(dict(local_index=local_index, output_stem=expected_stem, source_stem=row["source_stem"],
                           output_frame_index=output_frame, source_frame_index=source_index,
                           archive_row=archive_row))
    sources = np.array([m["source_frame_index"] for m in mapped])
    if np.any(np.diff(sources) <= 0):
        raise ValueError("Selected FreeArt frames are not in strictly increasing source chronology")
    return mapped


# ----------------------------------------------------------------------------- contact state


CONTACT_STATE_CODES = {"contact": 1, "absent": 0, "uncertain": -1}


def load_contact_state(scene_dir: Path, source_indices: np.ndarray, source: Path | None):
    """Return (contact_active bool[N], contact_confidence float[N], provenance dict, contact_state int8[N]).

    ``contact_state`` has three values: 1 confirmed contact, 0 confirmed absence, -1 uncertain.
    ``contact_active`` is exactly ``contact_state == 1``; uncertain never becomes contact.

    Supported explicit sources (JSON or CSV), indexed by **raw source_frame_index**:

    * per-frame JSON: ``{"index": "source_frame_index", "source_frame_index": [...],
      "contact_state": ["contact"|"absent"|"uncertain", ...]`` (or legacy boolean
      ``"contact_active"``, where False means confirmed absence), ``"contact_confidence"``
      (optional), ``"provenance": "..."}`` - e.g. ``export_force_contact_annotations.py``.
    * confirmed-interval JSON: ``{"index": "source_frame_index", "mode": "confirmed_intervals",
      "intervals": [[start, end], ...] (inclusive), "provenance": "...",
      "outside_intervals": "uncertain"|"absent"}`` (default uncertain).
    * CSV with columns ``source_frame_index`` and ``contact_state`` or ``contact_active``
      (``[,contact_confidence]``).

    ``source=None`` yields no confirmed contact and an *uncertain* state everywhere
    (never all-contact, never confirmed absence).  Per-frame sources must cover every frame.
    """
    source_indices = np.asarray(source_indices, dtype=np.int64)
    count = len(source_indices)
    confidence = np.full(count, np.nan)
    if source is None:
        return np.zeros(count, dtype=bool), confidence, dict(
            contact_source="none", status="unavailable",
            note="No explicit contact source supplied; contact state is uncertain for every frame."), \
            np.full(count, -1, dtype=np.int8)
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Contact source not found: {source}")
    provenance = dict(contact_source=str(source), contact_source_sha256=sha256_file(source))
    if source.suffix.lower() == ".csv":
        with source.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "source_frame_index" not in rows[0] or not ({"contact_active", "contact_state"} & set(rows[0])):
            raise ValueError("Contact CSV requires source_frame_index and contact_state or contact_active columns")
        data = {"index": "source_frame_index", "mode": "per_frame",
                "source_frame_index": [int(r["source_frame_index"]) for r in rows],
                "provenance": f"CSV {source.name}"}
        if "contact_state" in rows[0]:
            data["contact_state"] = [r["contact_state"].strip().lower() for r in rows]
        else:
            data["contact_active"] = [r["contact_active"].strip().lower() in ("1", "true", "yes") for r in rows]
        if "contact_confidence" in rows[0]:
            data["contact_confidence"] = [float(r["contact_confidence"]) if r["contact_confidence"] != "" else np.nan
                                          for r in rows]
    else:
        data = json.loads(source.read_text())
    if data.get("index", "source_frame_index") != "source_frame_index":
        raise ValueError("Contact source must be indexed by raw source_frame_index")
    if not str(data.get("provenance", "")).strip():
        raise ValueError("Contact source must state its provenance (who/what confirmed contact)")
    provenance["declared_provenance"] = data["provenance"]
    mode = data.get("mode", "confirmed_intervals" if "intervals" in data else "per_frame")
    provenance["mode"] = mode
    if mode == "confirmed_intervals":
        intervals = np.asarray(data["intervals"], dtype=np.int64).reshape(-1, 2)
        if np.any(intervals[:, 0] > intervals[:, 1]) or np.any(intervals < 0):
            raise ValueError("Contact intervals must be [start, end] with 0 <= start <= end")
        active = np.zeros(count, dtype=bool)
        for start, end in intervals:
            active |= (source_indices >= start) & (source_indices <= end)
        outside = data.get("outside_intervals", "uncertain")
        if outside not in ("uncertain", "absent"):
            raise ValueError("outside_intervals must be 'uncertain' or 'absent'")
        state = np.where(active, 1, -1 if outside == "uncertain" else 0).astype(np.int8)
        provenance.update(status="available", intervals=intervals.tolist(), outside_intervals=outside)
        return active, confidence, provenance, state
    if mode != "per_frame":
        raise ValueError(f"Unknown contact source mode {mode!r}")
    frames = np.asarray(data["source_frame_index"], dtype=np.int64)
    if "contact_state" in data:
        names = list(data["contact_state"])
        unknown = sorted(set(names) - set(CONTACT_STATE_CODES))
        if unknown:
            raise ValueError(f"Unknown contact_state values {unknown}; use contact/absent/uncertain")
        states = np.array([CONTACT_STATE_CODES[n] for n in names], dtype=np.int8)
        if "contact_active" in data and not np.array_equal(np.asarray(data["contact_active"], bool), states == 1):
            raise ValueError("contact_active disagrees with contact_state")
    else:
        states = np.where(np.asarray(data["contact_active"], dtype=bool), 1, 0).astype(np.int8)
    flags = states == 1
    if frames.shape != flags.shape or len(np.unique(frames)) != len(frames):
        raise ValueError("Per-frame contact source needs unique source_frame_index and matching contact states")
    lookup = {int(f): i for i, f in enumerate(frames)}
    missing = [int(f) for f in source_indices if int(f) not in lookup]
    if missing:
        raise ValueError(f"Contact source does not cover source frames {missing[:10]}; refusing to guess")
    rows = np.array([lookup[int(f)] for f in source_indices], dtype=np.int64)
    if "contact_confidence" in data:
        conf = np.asarray(data["contact_confidence"], dtype=np.float64)
        if conf.shape != frames.shape:
            raise ValueError("contact_confidence length mismatch")
        confidence = conf[rows]
    provenance["status"] = "available"
    provenance["state_counts"] = {name: int((states[rows] == code).sum()) for name, code in CONTACT_STATE_CODES.items()}
    return flags[rows], confidence, provenance, states[rows]


# ----------------------------------------------------------------------------- tool geometry


def load_tool_geometry(path: Path | None, tool_id: str | None) -> dict:
    """Explicit calibrated contact point in TCP-local coordinates, or an approximate anchor.

    JSON schema::

        {"tool_geometry_id": "pointed_tip", "contact_point_local_tool_m": [0, 0, 0],
         "contact_point_calibrated": true, "contact_uncertainty_m": 0.004,
         "shape": {"type": "point"} | {"type": "sphere", "radius_m": ...},
         "secondary_feature_local_tool_m": [x, y, z] (optional, rigid on the tool),
         "provenance": "..."}

    Without a file the TCP origin is an approximate anchor: ``contact_point_calibrated`` is
    False and ``contact_uncertainty_m`` is None (consumers must supply an explicit sigma).
    """
    if path is None:
        return dict(tool_geometry_id=tool_id or "unknown", contact_point_local_tool_m=[0.0, 0.0, 0.0],
                    contact_point_calibrated=False, contact_uncertainty_m=None, shape={"type": "point"},
                    secondary_feature_local_tool_m=None, source="none",
                    note="TCP origin used as approximate anchor; physical contact point not calibrated.")
    path = Path(path)
    data = json.loads(path.read_text())
    point = np.asarray(data["contact_point_local_tool_m"], dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("contact_point_local_tool_m must be three finite metres")
    uncertainty = data.get("contact_uncertainty_m")
    if uncertainty is not None and not (np.isfinite(uncertainty) and uncertainty > 0):
        raise ValueError("contact_uncertainty_m must be positive when given")
    if data.get("tool_geometry_id") and tool_id and data["tool_geometry_id"] != tool_id:
        raise ValueError(f"Tool geometry {data['tool_geometry_id']!r} does not match scene tool {tool_id!r}")
    secondary = data.get("secondary_feature_local_tool_m")
    if secondary is not None:
        secondary = np.asarray(secondary, dtype=np.float64)
        if secondary.shape != (3,) or not np.isfinite(secondary).all():
            raise ValueError("secondary_feature_local_tool_m must be three finite metres")
        secondary = secondary.tolist()
    shape = data.get("shape", {"type": "point"})
    if shape.get("type") not in ("point", "sphere"):
        raise ValueError("Tool shape type must be point or sphere")
    if shape["type"] == "sphere" and not float(shape.get("radius_m", 0)) > 0:
        raise ValueError("Sphere tool shape needs positive radius_m")
    return dict(tool_geometry_id=data.get("tool_geometry_id", tool_id or "unknown"),
                contact_point_local_tool_m=point.tolist(),
                contact_point_calibrated=bool(data.get("contact_point_calibrated", False)),
                contact_uncertainty_m=uncertainty, shape=shape,
                secondary_feature_local_tool_m=secondary, source=str(path), source_sha256=sha256_file(path),
                provenance=data.get("provenance"))


# ----------------------------------------------------------------------------- loading


@dataclass
class TCPSequence:
    camera_id: str
    image_size_wh: tuple[int, int]
    K_fullres: np.ndarray
    T_base_camera: np.ndarray
    local_frame_index: np.ndarray
    output_frame_index: np.ndarray
    source_frame_index: np.ndarray
    timestamp_s: np.ndarray
    T_base_ee: np.ndarray
    T_base_tcp: np.ndarray
    p_tcp_base_m: np.ndarray
    p_tcp_cam_m: np.ndarray
    uv_tcp_fullres_xy: np.ndarray
    valid_projection: np.ndarray
    p_contact_base_m: np.ndarray
    p_contact_cam_m: np.ndarray
    uv_contact_fullres_xy: np.ndarray
    valid_contact_projection: np.ndarray
    contact_active: np.ndarray
    contact_confidence: np.ndarray
    tool_geometry: dict
    provenance: dict = field(default_factory=dict)
    contact_state: np.ndarray | None = None
    p_feature2_cam_m: np.ndarray | None = None
    uv_feature2_fullres_xy: np.ndarray | None = None
    valid_feature2_projection: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.local_frame_index)

    def arrays(self) -> dict[str, np.ndarray]:
        out = dict(
            local_frame_index=self.local_frame_index, output_frame_index=self.output_frame_index,
            source_frame_index=self.source_frame_index, timestamp_s=self.timestamp_s,
            T_base_ee=self.T_base_ee, T_base_tcp=self.T_base_tcp, p_tcp_base_m=self.p_tcp_base_m,
            p_tcp_cam_m=self.p_tcp_cam_m, uv_tcp_fullres_xy=self.uv_tcp_fullres_xy,
            valid_projection=self.valid_projection, p_contact_base_m=self.p_contact_base_m,
            p_contact_cam_m=self.p_contact_cam_m, uv_contact_fullres_xy=self.uv_contact_fullres_xy,
            valid_contact_projection=self.valid_contact_projection, contact_active=self.contact_active,
            contact_confidence=self.contact_confidence, K_fullres=self.K_fullres,
            contact_state=(self.contact_state if self.contact_state is not None
                           else np.where(self.contact_active, 1, -1).astype(np.int8)),
            T_base_camera=self.T_base_camera, image_size_wh=np.asarray(self.image_size_wh, dtype=np.int64),
            contact_point_local_tool_m=np.asarray(self.tool_geometry["contact_point_local_tool_m"]),
        )
        if self.p_feature2_cam_m is not None:
            out.update(p_feature2_cam_m=self.p_feature2_cam_m, uv_feature2_fullres_xy=self.uv_feature2_fullres_xy,
                       valid_feature2_projection=self.valid_feature2_projection)
        return out


def _load_pose_archive(scene_dir: Path, pose_archive: Path | None, parquet_python: Path | None,
                       calibration_map: Path | None) -> tuple[dict, dict]:
    """Return RGB-D aligned pose arrays and provenance; T_ee_tcp applied at most once."""
    scene_dir = Path(scene_dir)
    archive = Path(pose_archive) if pose_archive else scene_dir / "metadata/robot_poses/poses_rgbd.npz"
    if archive.is_file():
        with np.load(archive, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        if "tcp_transform" not in arrays:
            raise ValueError(f"{archive} has no calibrated tcp_transform; re-export with an explicit T_ee_tcp")
        manifest_path = archive.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        return arrays, dict(pose_source="poses_rgbd.npz", pose_archive=str(archive),
                            pose_archive_sha256=sha256_file(archive),
                            T_ee_tcp=manifest.get("ee_to_tcp"), recorded_frame=manifest.get("recorded_frame"),
                            tcp_derivation="archive tcp_transform used as is (T_ee_tcp not re-applied)")
    if pose_archive is not None:
        raise FileNotFoundError(f"Pose archive not found: {archive}")
    if parquet_python is None:
        raise FileNotFoundError(
            f"No {archive}. Export it with samuele/export_real_robot_poses.py or pass --parquet-python "
            "to derive TCP poses from the raw Parquet with the scene's explicit tool calibration.")
    if str(SAMUELE_DIR) not in sys.path:
        sys.path.insert(0, str(SAMUELE_DIR))
    from export_real_robot_poses import build_arrays, calibration_for_scene  # noqa: E402
    from robot_mask_guidance import read_parquet_rows  # noqa: E402

    conversion = json.loads((scene_dir / "metadata/real_conversion.json").read_text())
    raw = scene_dir / "metadata/raw"
    parquet = raw / f"episode_{conversion['episode']:06d}.parquet"
    info = json.loads((raw / "info.json").read_text())
    if calibration_map is not None:
        frame, ee_to_tcp = calibration_for_scene(scene_dir, json.loads(Path(calibration_map).read_text()), "unknown")
        calibration_source = dict(calibration_map=str(calibration_map),
                                  calibration_map_sha256=sha256_file(Path(calibration_map)))
    else:
        tool_file = scene_dir / "metadata/robot_tool.json"
        if not tool_file.is_file():
            raise FileNotFoundError(f"No explicit tool calibration: {tool_file}; pass --calibration-map")
        tool = json.loads(tool_file.read_text())
        if tool.get("T_ee_tcp") is None:
            raise ValueError(f"{tool_file} declares no usable T_ee_tcp for tool {tool.get('tool')!r}")
        frame, ee_to_tcp = tool.get("recorded_pose_frame", "unknown"), rigid(tool["T_ee_tcp"], "T_ee_tcp")
        calibration_source = dict(robot_tool=str(tool_file), robot_tool_sha256=sha256_file(tool_file),
                                  tool=tool.get("tool"))
    if frame != "ee":
        raise ValueError(f"Recorded pose frame must be explicitly 'ee' to derive TCP; got {frame!r}")
    rows = read_parquet_rows(parquet, parquet_python, ["timestamp", "frame_index", "episode_index", "observation.state"])
    _, aligned = build_arrays(rows, info, conversion, frame, ee_to_tcp)
    return aligned, dict(pose_source="raw_parquet_via_export_real_robot_poses.build_arrays",
                         parquet=str(parquet), parquet_sha256=sha256_file(parquet),
                         T_ee_tcp=np.asarray(ee_to_tcp).tolist(), recorded_frame=frame,
                         tcp_derivation="T_base_tcp = T_base_ee @ T_ee_tcp (applied once)", **calibration_source)


def _scene_tool_id(scene_dir: Path) -> str | None:
    tool_file = Path(scene_dir) / "metadata/robot_tool.json"
    if tool_file.is_file():
        return json.loads(tool_file.read_text()).get("tool")
    return None


def _camera_entry(scene_dir: Path, camera: str) -> dict:
    cameras = json.loads((Path(scene_dir) / "metadata/cameras.json").read_text())
    for entry in cameras["cameras"]:
        if entry["name"] == camera:
            convention = entry.get("pose_convention", "opencv_optical_camera_to_world")
            if convention != "opencv_optical_camera_to_world":
                raise ValueError(f"Unsupported camera pose convention {convention!r} for {camera}")
            return entry
    raise ValueError(f"Camera {camera} absent from {scene_dir}/metadata/cameras.json")


def load_aligned_tcp(scene_dir: Path, camera_freeart_dir: Path, *, contact_source: Path | None = None,
                     tool_geometry: Path | None = None, pose_archive: Path | None = None,
                     parquet_python: Path | None = None, calibration_map: Path | None = None) -> TCPSequence:
    """Synchronise one FreeArt camera sequence with measured, calibrated TCP evidence."""
    scene_dir = Path(scene_dir).expanduser().resolve()
    camera_freeart_dir = Path(camera_freeart_dir).expanduser().resolve()
    conversion_path = camera_freeart_dir / "metadata/freeart_conversion.json"
    freeart = json.loads(conversion_path.read_text())
    if Path(freeart["scene_root"]).resolve() != scene_dir:
        raise ValueError(f"FreeArt data {camera_freeart_dir} was converted from {freeart['scene_root']}, not {scene_dir}")
    camera = str(freeart["selected_camera"])
    real_conversion_path = scene_dir / "metadata/real_conversion.json"
    real_conversion = json.loads(real_conversion_path.read_text())
    entry = _camera_entry(scene_dir, camera)
    image_size = (int(entry["width"]), int(entry["height"]))

    K = np.loadtxt(camera_freeart_dir / "cam_K.txt").reshape(3, 3)
    K_scene = np.array([[entry["fx"], 0, entry["cx"]], [0, entry["fy"], entry["cy"]], [0, 0, 1]], dtype=np.float64)
    if not np.allclose(K, K_scene, rtol=1e-6, atol=1e-6):
        raise ValueError(f"FreeArt cam_K.txt differs from scene cameras.json intrinsics for {camera}")
    if "rectified_K" in real_conversion and not np.allclose(np.asarray(real_conversion["rectified_K"]), K, atol=1e-4):
        raise ValueError(f"FreeArt cam_K.txt differs from real_conversion rectified_K for {camera}")
    T_base_camera = rigid(entry["pose_matrix_4x4"], f"{camera} pose_matrix_4x4")

    arrays, pose_provenance = _load_pose_archive(scene_dir, pose_archive, parquet_python, calibration_map)
    mapping = map_freeart_frames(freeart["frame_map"], arrays, real_conversion)
    rows = np.array([m["archive_row"] for m in mapping], dtype=np.int64)
    count = len(mapping)

    T_base_tcp = np.stack([rigid(T, "tcp_transform") for T in arrays["tcp_transform"][rows]])
    T_base_ee = arrays["ee_transform"][rows] if "ee_transform" in arrays else np.full((count, 4, 4), np.nan)
    timestamps = arrays["timestamp_s"][rows].astype(np.float64) if "timestamp_s" in arrays else np.full(count, np.nan)
    source_indices = np.array([m["source_frame_index"] for m in mapping], dtype=np.int64)

    tool_id = _scene_tool_id(scene_dir)
    geometry = load_tool_geometry(tool_geometry, tool_id)
    T_camera_base = np.linalg.inv(T_base_camera)
    p_tcp_base = T_base_tcp[:, :3, 3].copy()
    p_tcp_cam = transform_points(T_camera_base, p_tcp_base)
    uv_tcp, valid_tcp = pinhole(K, p_tcp_cam, image_size)
    contact_local = np.asarray(geometry["contact_point_local_tool_m"], dtype=np.float64)
    p_contact_base = transform_points(T_base_tcp, np.repeat(contact_local[None], count, axis=0))
    p_contact_cam = transform_points(T_camera_base, p_contact_base)
    uv_contact, valid_contact = pinhole(K, p_contact_cam, image_size)
    feature2 = dict()
    if geometry.get("secondary_feature_local_tool_m") is not None:
        local2 = np.asarray(geometry["secondary_feature_local_tool_m"], dtype=np.float64)
        p2_cam = transform_points(T_camera_base, transform_points(T_base_tcp, np.repeat(local2[None], count, axis=0)))
        uv2, valid2 = pinhole(K, p2_cam, image_size)
        feature2 = dict(p_feature2_cam_m=p2_cam, uv_feature2_fullres_xy=uv2, valid_feature2_projection=valid2)

    active, confidence, contact_provenance, contact_state = load_contact_state(scene_dir, source_indices, contact_source)
    provenance = dict(
        schema_version=SCHEMA_VERSION, scene_dir=str(scene_dir), camera_freeart_dir=str(camera_freeart_dir),
        camera=camera, source_camera_name=entry.get("source_camera_name"),
        freeart_conversion_sha256=sha256_file(conversion_path),
        real_conversion_sha256=sha256_file(real_conversion_path),
        cameras_json_sha256=sha256_file(scene_dir / "metadata/cameras.json"),
        camera_pose_convention="pose_matrix_4x4 = T_base_camera (OpenCV optical camera -> base_link)",
        intrinsics_check="cam_K.txt == cameras.json == real_conversion.rectified_K",
        frame_join="freeart frame_map.source_stem -> real output frame -> real_conversion/poses source_frame_index",
        frame_stride=freeart.get("frame_stride"), max_frames=freeart.get("max_frames"),
        frame_intervals=freeart.get("frame_intervals"),
        source_stems=[m["source_stem"] for m in mapping],
        pose=pose_provenance, contact=contact_provenance, tool_geometry=geometry,
        units=dict(position="m", pixels="full-resolution rectified image pixels, (u=x, v=y), integer pixel centres"),
        tcp_semantics="Measured robot TCP point; NOT an object material point.",
    )
    return TCPSequence(
        camera_id=camera, image_size_wh=image_size, K_fullres=K, T_base_camera=T_base_camera,
        local_frame_index=np.arange(count, dtype=np.int64),
        output_frame_index=np.array([m["output_frame_index"] for m in mapping], dtype=np.int64),
        source_frame_index=source_indices, timestamp_s=timestamps, T_base_ee=T_base_ee, T_base_tcp=T_base_tcp,
        p_tcp_base_m=p_tcp_base, p_tcp_cam_m=p_tcp_cam, uv_tcp_fullres_xy=uv_tcp, valid_projection=valid_tcp,
        p_contact_base_m=p_contact_base, p_contact_cam_m=p_contact_cam, uv_contact_fullres_xy=uv_contact,
        valid_contact_projection=valid_contact, contact_active=active, contact_confidence=confidence, contact_state=contact_state,
        tool_geometry=geometry, provenance=provenance, **feature2)


# ----------------------------------------------------------------------------- artifacts


def save_tcp_contact(sequence: TCPSequence, output_dir: Path, *, forbidden_roots: list[Path] = ()) -> dict:
    output_dir = Path(output_dir).expanduser().resolve()
    for root in forbidden_roots:
        root = Path(root).resolve()
        if output_dir == root or output_dir.is_relative_to(root):
            raise ValueError(f"Refusing to write TCP artifacts inside read-only source {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = sequence.arrays()
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise TypeError(f"Object array {key} cannot be saved without pickle")
    np.savez_compressed(output_dir / ARTIFACT_NPZ, **arrays)
    manifest = dict(sequence.provenance)
    manifest.update(
        artifact=ARTIFACT_NPZ, arrays=sorted(arrays), num_frames=len(sequence),
        status="available",
        summary=dict(valid_tcp_projections=int(sequence.valid_projection.sum()),
                     valid_contact_projections=int(sequence.valid_contact_projection.sum()),
                     contact_active_frames=int(sequence.contact_active.sum()),
                     contact_state_counts={n: int((sequence.arrays()["contact_state"] == c).sum())
                                           for n, c in CONTACT_STATE_CODES.items()}),
        artifact_sha256=sha256_file(output_dir / ARTIFACT_NPZ),
    )
    (output_dir / ARTIFACT_MANIFEST).write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def load_tcp_contact_artifact(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load ``tcp_contact.npz`` + manifest from a directory or the npz path itself."""
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    npz = directory / ARTIFACT_NPZ
    if not npz.is_file():
        raise FileNotFoundError(f"TCP contact artifact not found: {npz}")
    with np.load(npz, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    manifest_path = directory / ARTIFACT_MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return arrays, manifest


def write_overlays(sequence: TCPSequence, camera_freeart_dir: Path, output_dir: Path, count: int) -> list[str]:
    import cv2

    written = []
    if count <= 0:
        return written
    indices = np.unique(np.linspace(0, len(sequence) - 1, min(count, len(sequence))).astype(int))
    overlay_dir = Path(output_dir) / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        stem = f"{index:05d}"
        rgb_path = next((p for p in [Path(camera_freeart_dir) / "rgb" / f"{stem}.png",
                                     Path(camera_freeart_dir) / "images_ori" / f"{stem}.png"] if p.is_file()), None)
        if rgb_path is None:
            continue
        image = cv2.imread(str(rgb_path))
        for uv, valid, color in [(sequence.uv_tcp_fullres_xy[index], sequence.valid_projection[index], (0, 0, 255)),
                                 (sequence.uv_contact_fullres_xy[index], sequence.valid_contact_projection[index],
                                  (0, 255, 0))]:
            if valid:
                cv2.circle(image, (int(round(uv[0])), int(round(uv[1]))), 5, color, 1, cv2.LINE_AA)
        label = f"src {sequence.source_frame_index[index]} contact={bool(sequence.contact_active[index])}"
        cv2.putText(image, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out = overlay_dir / f"{stem}.png"
        cv2.imwrite(str(out), image)
        written.append(str(out))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True, help="Converted real scene (read only)")
    parser.add_argument("--freeart-camera-data", type=Path, required=True,
                        help="One converted FreeArt camera folder with metadata/freeart_conversion.json")
    parser.add_argument("--contact-file", type=Path, help="Explicit contact JSON/CSV indexed by source_frame_index")
    parser.add_argument("--tool-geometry", type=Path, help="Calibrated tool contact point/shape JSON")
    parser.add_argument("--pose-archive", type=Path, help="poses_rgbd.npz (default: scene metadata/robot_poses)")
    parser.add_argument("--parquet-python", type=Path,
                        help="Python with polars; derive TCP from the raw Parquet when no pose archive exists")
    parser.add_argument("--calibration-map", type=Path,
                        help="Calibration map (configs/real_robot_pose_calibration.json) instead of robot_tool.json")
    parser.add_argument("--output", type=Path, help="Output directory (default: <freeart-camera-data>/metadata/tcp_contact)")
    parser.add_argument("--overlays", type=int, default=0, help="Number of overlay PNG snapshots to write")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarise without writing anything")
    args = parser.parse_args(argv)
    sequence = load_aligned_tcp(args.scene, args.freeart_camera_data, contact_source=args.contact_file,
                                tool_geometry=args.tool_geometry, pose_archive=args.pose_archive,
                                parquet_python=args.parquet_python, calibration_map=args.calibration_map)
    output = args.output or Path(args.freeart_camera_data) / "metadata/tcp_contact"
    summary = dict(camera=sequence.camera_id, frames=len(sequence),
                   source_frames=[int(sequence.source_frame_index[0]), int(sequence.source_frame_index[-1])],
                   valid_tcp_projections=int(sequence.valid_projection.sum()),
                   valid_contact_projections=int(sequence.valid_contact_projection.sum()),
                   contact_active_frames=int(sequence.contact_active.sum()),
                   contact_source=sequence.provenance["contact"].get("contact_source"),
                   contact_point_calibrated=sequence.tool_geometry["contact_point_calibrated"],
                   pose_source=sequence.provenance["pose"]["pose_source"], output=str(output))
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, **summary), indent=2))
        return 0
    save_tcp_contact(sequence, output, forbidden_roots=[Path(args.scene)])
    overlays = write_overlays(sequence, args.freeart_camera_data, output, args.overlays)
    print(json.dumps(dict(dry_run=False, overlays=len(overlays), **summary), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
