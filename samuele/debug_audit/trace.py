"""Opt-in, behaviour-neutral diagnostic tracing for the root-cause audit.

Off unless ``FREEARTGS_TRACE_ROOT`` is set in the environment, so hooks placed in production
code cost one dictionary lookup and change nothing.  When on, every ``event()`` appends one
JSON line to ``$FREEARTGS_TRACE_ROOT/events.jsonl`` with the audit schema, and any arrays are
written losslessly to ``$FREEARTGS_TRACE_ROOT/arrays/<seq>_<event>.npz`` (referenced from the
event's ``artifact_refs``).

Guarantees (tested in tests/test_debug_audit_trace.py):
  * tracing never mutates its inputs: tensors are detached and copied before conversion;
  * tracing never consumes random numbers (no torch/numpy/python RNG calls);
  * with tracing off, ``event()`` returns immediately and ``enabled()`` is False.

Context: ``FREEARTGS_TRACE_RUN_ID``, ``FREEARTGS_TRACE_SCENE``, ``FREEARTGS_TRACE_CAMERA`` and
``FREEARTGS_TRACE_CONFIG`` (path of the effective config; its sha256 becomes ``config_hash``)
are read from the environment; code boundaries add per-call context with ``context(...)``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

ENV_ROOT = "FREEARTGS_TRACE_ROOT"
SCHEMA_KEYS = ("run_id", "event", "scene", "camera", "source_rgbd_frame", "raw_source_frame", "frame_map_entry",
               "pingpong_iteration", "window_id", "part_id", "local_file", "function", "source_revision",
               "tensor_shape", "coordinate_frame", "transform_direction", "units", "config_hash", "seed",
               "value", "artifact_refs")
REPO = Path(__file__).resolve().parents[2]

_lock = threading.Lock()
_state = {"seq": 0, "revision": None, "config_hash": None, "frame_maps": None}
_ctx = threading.local()


def enabled() -> bool:
    return bool(os.environ.get(ENV_ROOT))


def root() -> Path:
    return Path(os.environ[ENV_ROOT])


def source_revision() -> str:
    """``<HEAD sha>+<sha256 of `git diff HEAD` and untracked file hashes>`` (computed once)."""
    if _state["revision"] is None:
        def git(*args):
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=False).stdout
        head = git("rev-parse", "HEAD").decode().strip() or "UNVERIFIED"
        digest = hashlib.sha256(git("diff", "HEAD"))
        for name in sorted(git("ls-files", "--others", "--exclude-standard").decode().splitlines()):
            path = REPO / name
            if path.is_file():
                digest.update(name.encode())
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        _state["revision"] = f"{head}+dirty:{digest.hexdigest()[:16]}"
    return _state["revision"]


def config_hash():
    if _state["config_hash"] is None:
        path = os.environ.get("FREEARTGS_TRACE_CONFIG")
        _state["config_hash"] = (hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 if path and Path(path).is_file() else "MISSING")
    return _state["config_hash"]


def _current():
    if not hasattr(_ctx, "stack"):
        _ctx.stack = [{}]
    return _ctx.stack[-1]


@contextlib.contextmanager
def context(**fields):
    """Add schema fields (window_id, part_id, pingpong_iteration, ...) for nested events."""
    if not enabled():
        yield
        return
    merged = dict(_current())
    merged.update(fields)
    _ctx.stack.append(merged)
    try:
        yield
    finally:
        _ctx.stack.pop()


def to_numpy(x):
    """Detached, copied numpy view of a tensor/array/scalar (never shares memory with ``x``)."""
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().to("cpu", copy=True).numpy()
    except ImportError:  # pragma: no cover - torch is always present in the pipeline env
        pass
    return np.array(x, copy=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else repr(v)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    try:
        import torch
        if torch.is_tensor(value):
            return _jsonable(to_numpy(value))
    except ImportError:  # pragma: no cover
        pass
    return value


def event(name, *, local_file, function, value=None, arrays=None, **fields):
    """Persist one event. ``arrays``: {name: tensor/ndarray}, stored losslessly in one npz."""
    if not enabled():
        return None
    out = root()
    with _lock:
        _state["seq"] += 1
        seq = _state["seq"]
    record = {key: None for key in SCHEMA_KEYS}
    record.update(run_id=os.environ.get("FREEARTGS_TRACE_RUN_ID", "UNSET"),
                  scene=os.environ.get("FREEARTGS_TRACE_SCENE", "UNSET"),
                  camera=os.environ.get("FREEARTGS_TRACE_CAMERA", "UNSET"),
                  source_revision=source_revision(), config_hash=config_hash(), units="m")
    record.update(_current())
    record.update(fields)
    record.update(event=name, local_file=local_file, function=function, value=_jsonable(value or {}),
                  artifact_refs=list(record.get("artifact_refs") or []), seq=seq, wall_time=time.time())
    if arrays:
        (out / "arrays").mkdir(parents=True, exist_ok=True)
        rel = f"arrays/{seq:07d}_{name}.npz"
        np.savez_compressed(out / rel, **{k: to_numpy(v) for k, v in arrays.items()})
        record["artifact_refs"].append(rel)
        record["tensor_shape"] = {k: list(to_numpy(v).shape) for k, v in arrays.items()}
    out.mkdir(parents=True, exist_ok=True)
    with _lock, open(out / "events.jsonl", "a") as handle:
        handle.write(json.dumps(_jsonable(record)) + "\n")
    return record


def matrix_rigidity(T):
    """Non-rigidity of a raw 4x4: ‖RᵀR−I‖_F, det R, ‖last row − e4‖, singular values, polar rotation angle."""
    T = np.asarray(to_numpy(T), dtype=np.float64)
    R = T[:3, :3]
    U, S, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1
        Rp = U @ Vt
    angle = float(np.degrees(np.arccos(np.clip((np.trace(Rp) - 1) / 2, -1, 1))))
    return dict(orth_err_fro=float(np.linalg.norm(R.T @ R - np.eye(3))), det=float(np.linalg.det(R)),
                homog_err=float(np.linalg.norm(T[3] - np.array([0, 0, 0, 1.0]))),
                singular_values=[float(s) for s in S], max_stretch=float(np.max(np.abs(S - 1))),
                polar_rotation_deg=angle, translation=[float(x) for x in T[:3, 3]])


def read_events(path):
    """Load an events.jsonl into a list of dicts (helper for the analysis scripts)."""
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]
