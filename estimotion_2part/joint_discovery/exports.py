"""Artifacts, provenance and the explicit new-to-old FreeArtGS adapter (spec 11, 14).

The adapter writes *new* files next to (never over) legacy outputs. Physical cameras keep their
calibrated ``T_WC``; the moving part's virtual cameras are derived exactly as
``T_M0,C(t) = A_H(q_t)^-1 T_WC`` from the shared world-frame joint trajectory. Prismatic joints
export ``joint_origin_world: null`` with ``origin_convention: N/A``.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from .rigid_kinematics import A_joint, canonicalize_sign, se3_inverse


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float):
        return None if not np.isfinite(x) else x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=1))


# --------------------------------------------------------------------------- candidates
def candidate_to_json(c, frame_map) -> dict:
    axis, q = canonicalize_sign(c.axis_w, c.q)
    moving = np.nan_to_num(c.moving_probability) > 0.5
    return dict(
        joint_type=c.kind,
        joint_axis_world=axis,
        joint_origin_world=c.origin_w if c.kind == "revolute" else None,
        origin_convention=c.origin_convention,
        sign_convention="(u, q) ~ (-u, -q); exported axis has a positive largest-magnitude component",
        q_units="rad" if c.kind == "revolute" else "m",
        q=q, q_valid=c.q_valid,
        q_uncertainty=None if c.q_uncertainty is None else c.q_uncertainty,
        reference_time_index=c.reference_time_index,
        timestamps_s=frame_map.timestamp_s, rgbd_frame_index=frame_map.rgbd_frame_index,
        source_frame_index=frame_map.source_frame_index,
        n_moving_tracks=int(moving.sum()),
        world_frame="calibrated reconstruction world (T_WC of each physical camera)",
        transform_convention="A_H(q) maps canonical (q_ref=0) world points to world points at q",
        diagnostics={k: v for k, v in c.diagnostics.items() if k != "seed"},
    )


def write_joint_states(path, candidates: dict, frame_map):
    arrays = dict(timestamp_s=frame_map.timestamp_s, rgbd_frame_index=frame_map.rgbd_frame_index,
                  source_frame_index=frame_map.source_frame_index)
    for kind, c in candidates.items():
        key = "R" if kind == "revolute" else "P"
        arrays[f"q_{key}"] = c.q
        arrays[f"q_valid_{key}"] = c.q_valid
        arrays[f"q_sigma_{key}"] = c.q_uncertainty if c.q_uncertainty is not None else np.full(len(c.q), np.nan)
        from .tcp_contact import joint_rate
        arrays[f"qdot_{key}"] = joint_rate(c.q, frame_map.timestamp_s)
    np.savez(path, **arrays)


# --------------------------------------------------------------------------- virtual cameras
def virtual_camera_poses(c, T_WC) -> np.ndarray:
    """``T_M0,C(t) = A_H(q_t)^-1 T_WC`` [T,4,4] (OpenCV c2w in the moving part's reference frame)."""
    A = A_joint(c.kind, c.axis_w, c.origin_w, np.asarray(c.q, float))
    return se3_inverse(A) @ np.asarray(T_WC, float)


def export_for_existing_freeart(candidate, cam_calibs: dict, frame_map, output_dir, stems=None,
                                visible=None) -> dict:
    """Write per-camera ``transforms_0.json`` (static part, physical camera) and
    ``transforms_1.json`` (moving part, virtual part camera) plus ``joint_candidate.json``.

    Every frame row carries camera name, time index, source stem, intrinsics, paths and an
    explicit ``pose_kind``. Rows are never dropped silently: invisible frames are kept with
    ``valid: false``. Poses are OpenCV camera-to-world (``pose_convention``); legacy readers that
    expect another convention must convert explicitly."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    T = frame_map.T
    stems = stems or [f"frame_{int(i):06d}" for i in frame_map.rgbd_frame_index]
    manifest = dict(files=[], provenance="estimated_joint_world_frame", pose_convention="opencv_c2w",
                    joint_type=candidate.kind)
    for cid, cal in cam_calibs.items():
        Tv = virtual_camera_poses(candidate, cal.T_WC)
        vis = np.ones(T, bool) if visible is None else np.asarray(visible.get(cid, np.ones(T, bool)))
        for part, poses, kind in ((0, np.repeat(np.asarray(cal.T_WC)[None], T, 0), "physical_camera"),
                                  (1, Tv, "virtual_part_camera")):
            frames = []
            for t in range(T):
                frames.append(dict(camera_name=cid, time_index=t, timestamp_s=float(frame_map.timestamp_s[t]),
                                   source_stem=stems[t], source_frame_index=int(frame_map.source_frame_index[t]),
                                   file_path=f"{cid}/images/{stems[t]}.png", mask_path=f"{cid}/mask_{part}/{stems[t]}.png",
                                   depth_path=f"{cid}/depth/{stems[t]}.npz", transform_matrix=poses[t],
                                   pose_kind=kind, valid=bool(vis[t]),
                                   fl_x=float(cal.K[0, 0]), fl_y=float(cal.K[1, 1]), cx=float(cal.K[0, 2]),
                                   cy=float(cal.K[1, 2]), w=int(cal.width), h=int(cal.height)))
            path = out / cid / f"transforms_{part}.json"
            write_json(path, dict(frames=frames, pose_convention="opencv_c2w", part=part,
                                  part_role="static" if part == 0 else "moving",
                                  camera_optimizer="off (fixed calibrated physical cameras)"))
            manifest["files"].append(str(path.relative_to(out)))
    joint = candidate_to_json(candidate, frame_map)
    write_json(out / "joint_candidate.json", joint)
    manifest["files"].append("joint_candidate.json")
    write_json(out / "adapter_manifest.json", manifest)
    return manifest


def load_joint_candidate(path) -> dict:
    """Downstream import (e.g. instead of re-running ``preprocessing/estimate_joint.py``'s
    rotational-span classifier, which must not silently re-classify a validated candidate)."""
    data = json.loads(Path(path).read_text())
    if data["joint_type"] == "prismatic" and data.get("joint_origin_world") is not None:
        raise ValueError("prismatic candidate must not carry a pivot")
    return data


# --------------------------------------------------------------------------- provenance
def git_provenance(repo) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30).stdout
        except Exception as exc:  # pragma: no cover
            return f"unavailable: {exc}"
    diff = run("diff")
    return dict(head=run("rev-parse", "HEAD").strip(), status_short=run("status", "--short").splitlines()[:200],
                diff_sha256=hashlib.sha256(diff.encode()).hexdigest(), diff_stat=run("diff", "--stat").splitlines()[-1:])


def package_versions() -> dict:
    out = dict(python=sys.version.split()[0], platform=platform.platform())
    for name in ("numpy", "scipy", "torch"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = None
    try:
        import torch
        out["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return out
