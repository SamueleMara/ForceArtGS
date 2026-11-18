"""Authoritative joint contract for joint-aware reconstruction (end-to-end spec section 2).

A ``JointAuthority`` is the single source of truth for one branch H (revolute / prismatic / static):
joint type, world axis u, hinge point o (revolute only; ``None`` for prismatic), one signed q per
acquisition time with ``q[reference_time_index] == 0`` exactly, the frame map, the calibrated cameras
and the hashes of everything it was derived from. Downstream builders and training workers only
read it; legacy-compatible files (``estimated_joint.json``) are derived adapters checked against it.

Conventions: column vectors, ``T_AB`` maps B to A, world W = the calibrated ``metadata/cameras.json``
frame. ``A_R(t) = [R_u(q_t), (I - R_u(q_t)) o; 0 1]``, ``A_P(t) = [I, q_t u; 0 1]``, ``A_S = I``.
Canonical material coordinates are W at ``reference_time_index``.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .rigid_kinematics import A_joint

SCHEMA_VERSION = "forceartgs.joint_authority/1"
KINDS = ("revolute", "prismatic", "static")
DECISION_STATUS = ("candidate", "provisional", "identified", "ambiguous")


class AuthorityError(ValueError):
    """Raised for any authority that fails validation (fail closed)."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_jsonable)


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not serialisable: {type(x)}")


def atomic_write_text(path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_jsonable))


def atomic_save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".npz")
    os.close(fd)
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


@dataclass
class CameraEntry:
    camera_id: str
    K: np.ndarray            # [3,3]
    T_WC: np.ndarray         # [4,4] OpenCV camera-to-world
    width: int
    height: int
    source: str = ""

    def to_json(self) -> dict:
        return dict(camera_id=self.camera_id, K=np.asarray(self.K, float).tolist(),
                    T_WC=np.asarray(self.T_WC, float).tolist(), width=int(self.width), height=int(self.height),
                    source=self.source, pose_convention="opencv_c2w")

    @staticmethod
    def from_json(d) -> "CameraEntry":
        return CameraEntry(d["camera_id"], np.asarray(d["K"], float), np.asarray(d["T_WC"], float),
                           int(d["width"]), int(d["height"]), d.get("source", ""))

    def sha(self) -> str:
        return sha256_bytes(canonical_json(self.to_json()).encode())


@dataclass
class JointAuthority:
    joint_type: str
    axis_world: Optional[np.ndarray]          # [3] unit (None for static)
    origin_world: Optional[np.ndarray]        # [3] revolute only; None for prismatic/static
    q: np.ndarray                             # [T] signed, rad (R) or m (P); zeros for static
    q_valid: np.ndarray                       # [T] bool
    reference_time_index: int
    rgbd_frame_index: np.ndarray              # [T]
    source_frame_index: np.ndarray            # [T]
    timestamp_s: np.ndarray                   # [T]
    cameras: dict                             # camera_id -> CameraEntry
    decision_status: str = "candidate"
    q_sigma: Optional[np.ndarray] = None      # [T] or None (never turned into zeros)
    fit_covariance_status: str = "UNAVAILABLE"
    support_centroid_world: Optional[np.ndarray] = None
    source_hashes: dict = field(default_factory=dict)
    derived_from: dict = field(default_factory=dict)
    parent_authority_id: Optional[str] = None
    version: int = 0
    world_frame: str = "metadata/cameras.json world (OpenCV c2w), metres"

    # ------------------------------------------------------------------ geometry
    @property
    def T(self) -> int:
        return int(len(self.q))

    @property
    def q_units(self) -> str:
        return {"revolute": "rad", "prismatic": "m", "static": "none"}[self.joint_type]

    def transform(self, t: int) -> np.ndarray:
        """``A_H(q_t)`` in W, float64 [4,4]."""
        if self.joint_type == "static":
            return np.eye(4)
        o = self.origin_world if self.joint_type == "revolute" else np.zeros(3)
        return np.asarray(A_joint(self.joint_type, self.axis_world, o, np.array([self.q[t]], float))[0], float)

    def transforms(self) -> np.ndarray:
        return np.stack([self.transform(t) for t in range(self.T)])

    def relative(self, s: int, t: int) -> np.ndarray:
        """Exact finite relative motion ``A(s) A(t)^-1`` (maps a moving point seen at t to time s)."""
        return self.transform(s) @ np.linalg.inv(self.transform(t))

    def virtual_camera(self, camera_id: str, t: int) -> np.ndarray:
        """``C_M(c,t) = A_H(q_t)^-1 T_WC``: the camera pose in the moving part's canonical frame."""
        return np.linalg.inv(self.transform(t)) @ np.asarray(self.cameras[camera_id].T_WC, float)

    # ------------------------------------------------------------------ validation / hashing
    def validate(self) -> "JointAuthority":
        if self.joint_type not in KINDS:
            raise AuthorityError(f"unknown joint type {self.joint_type!r}")
        if self.decision_status not in DECISION_STATUS:
            raise AuthorityError(f"unknown decision status {self.decision_status!r}")
        T = self.T
        for name in ("q_valid", "rgbd_frame_index", "source_frame_index", "timestamp_s"):
            if len(getattr(self, name)) != T:
                raise AuthorityError(f"{name} has length {len(getattr(self, name))} != T={T}")
        if not 0 <= self.reference_time_index < T:
            raise AuthorityError("reference_time_index out of range")
        q = np.asarray(self.q, float)
        if not np.isfinite(q).all():
            raise AuthorityError("q must be finite (mark unobservable times with q_valid=False, not NaN)")
        if q[self.reference_time_index] != 0.0:
            raise AuthorityError("q at the reference time must be exactly 0")
        for name in ("timestamp_s", "rgbd_frame_index", "source_frame_index"):
            if T > 1 and np.any(np.diff(np.asarray(getattr(self, name), float)) <= 0):
                raise AuthorityError(f"{name} must be strictly increasing")
        if self.joint_type == "static":
            if np.any(q != 0):
                raise AuthorityError("static authority must have q == 0")
        else:
            u = np.asarray(self.axis_world, float)
            if u.shape != (3,) or not np.isfinite(u).all() or abs(np.linalg.norm(u) - 1) > 1e-9:
                raise AuthorityError("axis must be a finite unit 3-vector")
        if self.joint_type == "revolute":
            o = np.asarray(self.origin_world, float)
            if o.shape != (3,) or not np.isfinite(o).all():
                raise AuthorityError("revolute authority needs a finite hinge point")
            if self.support_centroid_world is not None:
                gauge = float((o - np.asarray(self.support_centroid_world, float)) @ np.asarray(self.axis_world))
                if abs(gauge) > 1e-6:
                    raise AuthorityError(f"hinge-point gauge violated: (o-a).u = {gauge:.3g}")
        if self.joint_type == "prismatic" and self.origin_world is not None:
            raise AuthorityError("prismatic authority must carry origin_world = None")
        if self.q_sigma is not None and len(self.q_sigma) != T:
            raise AuthorityError("q_sigma length mismatch")
        if not self.cameras:
            raise AuthorityError("authority needs at least one calibrated camera")
        for cid, cam in self.cameras.items():
            R = np.asarray(cam.T_WC, float)[:3, :3]
            if abs(np.linalg.det(R) - 1) > 1e-6 or np.abs(R.T @ R - np.eye(3)).max() > 1e-6:
                raise AuthorityError(f"{cid}: camera rotation is not a proper rotation")
        for t in range(T):
            A = self.transform(t)
            if abs(np.linalg.det(A[:3, :3]) - 1) > 1e-9:
                raise AuthorityError("joint transform is not SE(3)")
        if np.abs(self.transform(self.reference_time_index) - np.eye(4)).max() > 1e-12:
            raise AuthorityError("A(t_ref) must be the identity")
        return self

    def to_json(self) -> dict:
        return dict(
            schema_version=SCHEMA_VERSION, version=int(self.version), joint_type=self.joint_type,
            axis_world=None if self.axis_world is None else np.asarray(self.axis_world, float).tolist(),
            origin_world=None if self.origin_world is None else np.asarray(self.origin_world, float).tolist(),
            reference_time_index=int(self.reference_time_index), q_units=self.q_units, world_frame=self.world_frame,
            decision_status=self.decision_status, fit_covariance_status=self.fit_covariance_status,
            support_centroid_world=None if self.support_centroid_world is None
            else np.asarray(self.support_centroid_world, float).tolist(),
            cameras={cid: cam.to_json() for cid, cam in sorted(self.cameras.items())},
            calibration_hash=sha256_bytes(canonical_json({c: v.to_json() for c, v in sorted(self.cameras.items())}).encode()),
            source_hashes=self.source_hashes, derived_from=self.derived_from,
            parent_authority_id=self.parent_authority_id)

    def arrays(self) -> dict:
        out = dict(q=np.asarray(self.q, float), q_valid=np.asarray(self.q_valid, bool),
                   rgbd_frame_index=np.asarray(self.rgbd_frame_index, np.int64),
                   source_frame_index=np.asarray(self.source_frame_index, np.int64),
                   timestamp_s=np.asarray(self.timestamp_s, float))
        if self.q_sigma is not None:
            out["q_sigma"] = np.asarray(self.q_sigma, float)
        return out

    def frame_map_hash(self) -> str:
        a = self.arrays()
        return sha256_bytes(b"".join(a[k].tobytes() for k in ("rgbd_frame_index", "source_frame_index", "timestamp_s")))

    def authority_id(self) -> str:
        a = self.arrays()
        payload = canonical_json(self.to_json()).encode() + b"".join(a[k].tobytes() for k in sorted(a))
        return sha256_bytes(payload)

    # ------------------------------------------------------------------ io
    def write(self, directory) -> str:
        """Atomic write of ``authority.json`` + ``joint_states.npz``; refuses to overwrite a different authority."""
        self.validate()
        d = Path(directory)
        aid = self.authority_id()
        if (d / "authority.json").exists():
            old = json.loads((d / "authority.json").read_text())
            if old.get("authority_id") != aid:
                raise AuthorityError(f"refusing to overwrite authority {old.get('authority_id')} in {d} with {aid}; "
                                     "bump the version into a new directory")
            return aid
        atomic_save_npz(d / "joint_states.npz", **self.arrays())
        meta = self.to_json()
        meta.update(authority_id=aid, frame_map_hash=self.frame_map_hash(),
                    joint_states_sha256=sha256_file(d / "joint_states.npz"))
        atomic_write_json(d / "authority.json", meta)
        return aid

    @staticmethod
    def load(directory) -> "JointAuthority":
        """Load and verify: schema, joint_states hash and authority id must all match (fail closed)."""
        d = Path(directory)
        meta = json.loads((d / "authority.json").read_text())
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise AuthorityError(f"unsupported authority schema {meta.get('schema_version')}")
        if sha256_file(d / "joint_states.npz") != meta["joint_states_sha256"]:
            raise AuthorityError(f"{d}/joint_states.npz was modified after the authority was written")
        z = np.load(d / "joint_states.npz")
        a = JointAuthority(
            joint_type=meta["joint_type"],
            axis_world=None if meta["axis_world"] is None else np.asarray(meta["axis_world"], float),
            origin_world=None if meta["origin_world"] is None else np.asarray(meta["origin_world"], float),
            q=z["q"], q_valid=z["q_valid"], reference_time_index=int(meta["reference_time_index"]),
            rgbd_frame_index=z["rgbd_frame_index"], source_frame_index=z["source_frame_index"],
            timestamp_s=z["timestamp_s"], cameras={c: CameraEntry.from_json(v) for c, v in meta["cameras"].items()},
            decision_status=meta["decision_status"], q_sigma=z["q_sigma"] if "q_sigma" in z.files else None,
            fit_covariance_status=meta["fit_covariance_status"],
            support_centroid_world=None if meta.get("support_centroid_world") is None
            else np.asarray(meta["support_centroid_world"], float),
            source_hashes=meta["source_hashes"], derived_from=meta["derived_from"],
            parent_authority_id=meta.get("parent_authority_id"), version=int(meta["version"]),
            world_frame=meta["world_frame"])
        a.validate()
        if a.authority_id() != meta["authority_id"]:
            raise AuthorityError(f"authority id mismatch in {d}: content was edited")
        return a


def regauge(a: JointAuthority, new_ref: int) -> JointAuthority:
    """Same physical motion with the canonical frame moved to W at ``new_ref`` (exact for 1-DoF joints:
    R(q_s) R(q_r)^-1 = R(q_s - q_r) about the same line; translations subtract)."""
    q = np.asarray(a.q, float) - float(a.q[new_ref])
    q[new_ref] = 0.0
    b = JointAuthority(**{**a.__dict__, "q": q, "reference_time_index": int(new_ref)})
    return b


def authority_from_discovery(run_dir, kind: str, scene_root, cameras, *, decision_status: str,
                             reference_time_index: int = 0, observations=None) -> JointAuthority:
    """Build a branch authority from a finished ``joint_discovery fit`` run (read-only).

    Uses ``joint_candidates.json`` as the single source for (type, u, o, q) — its exported (u, q) pair is
    self-consistent — and ``joint_states.npz`` only for validity flags and per-time sigma. The discovery's own
    gauge (q = 0 at its reference) is re-gauged exactly to ``reference_time_index``. The hinge point is moved
    along u to the gauge nearest the moving-support centroid."""
    from .observations import load_scene_cameras, load_scene_frame_map
    run = Path(run_dir)
    cands = json.loads((run / "joint_candidates.json").read_text())
    if kind == "static":
        states = np.load(run / "joint_states.npz")
        T = len(states["rgbd_frame_index"])
        c = dict(joint_type="static", q=[0.0] * T)
    else:
        if kind not in cands["candidates"]:
            raise AuthorityError(f"{run} has no {kind} candidate")
        c = cands["candidates"][kind]
    states = np.load(run / "joint_states.npz")
    rgbd = np.asarray(states["rgbd_frame_index"], int)
    fm = load_scene_frame_map(scene_root, rgbd.tolist())
    if not (np.array_equal(fm.source_frame_index, states["source_frame_index"])
            and np.allclose(fm.timestamp_s, states["timestamp_s"], atol=1e-9)):
        raise AuthorityError("discovery frame map disagrees with the scene's metadata/frame_timestamps.csv")
    calibs, audit = load_scene_cameras(scene_root, cameras)
    cams = {cid: CameraEntry(cid, cal.K, cal.T_WC, cal.width, cal.height, cal.calibration_source)
            for cid, cal in calibs.items()}
    q = np.asarray(c["q"], float)
    if len(q) != len(rgbd):
        raise AuthorityError("candidate q length != discovery frame map length")
    suffix = {"revolute": "R", "prismatic": "P"}.get(kind)
    q_valid = np.asarray(states[f"q_valid_{suffix}"], bool) if suffix else np.ones(len(q), bool)
    q_sigma = np.asarray(states[f"q_sigma_{suffix}"], float) if suffix and f"q_sigma_{suffix}" in states.files else None
    if q_sigma is not None and not np.isfinite(q_sigma).all():
        q_sigma = None
    axis = None if kind == "static" else np.asarray(c["joint_axis_world"], float)
    if axis is not None:
        axis = axis / np.linalg.norm(axis)
    origin, centroid = None, None
    support = run / "part_support" / "track_support.npz"
    if kind == "revolute":
        o = np.asarray(c["joint_origin_world"], float)
        if support.is_file():
            s = np.load(support)
            pm, anc = s[f"p_moving_{suffix}"], s[f"anchor_w_{suffix}"]
            ok = s["eligible"] & (pm > 0.5) & np.isfinite(anc).all(1)
            if ok.sum() >= 3:
                centroid = np.average(anc[ok], axis=0, weights=pm[ok])
        if centroid is not None:
            o = o + float((centroid - o) @ axis) * axis    # gauge: (o - a).u = 0
            centroid = centroid  # stored so the gauge is checkable
        origin = o
    hashes = {name: sha256_file(run / name) for name in ("joint_candidates.json", "joint_states.npz", "manifest.json",
                                                          "model_selection.json") if (run / name).is_file()}
    if support.is_file():
        hashes["part_support/track_support.npz"] = sha256_file(support)
    disc_ref = int(np.argmin(np.abs(q))) if kind != "static" else 0
    if abs(q[disc_ref]) > 1e-9:
        raise AuthorityError(f"discovery q has no exact reference time (min |q| = {abs(q[disc_ref]):.3g})")
    q = q.copy()
    q[disc_ref] = 0.0
    a = JointAuthority(joint_type=kind, axis_world=axis, origin_world=origin, q=q, q_valid=q_valid,
                       reference_time_index=disc_ref,
                       rgbd_frame_index=rgbd, source_frame_index=fm.source_frame_index, timestamp_s=fm.timestamp_s,
                       cameras=cams, decision_status=decision_status, q_sigma=q_sigma,
                       fit_covariance_status="per_time_marginal_sigma" if q_sigma is not None else "UNAVAILABLE",
                       support_centroid_world=centroid, source_hashes=hashes,
                       derived_from=dict(discovery_run=str(run), candidate=kind, scene=str(scene_root),
                                         calibration_audit=audit, discovery_decision=cands.get("decision")))
    a.derived_from["discovery_reference_time_index"] = disc_ref
    if kind != "static":
        a = regauge(a.validate(), reference_time_index)
    return a.validate()


def derive_estimated_joint(a: JointAuthority, N_BW: np.ndarray, authority_id: str) -> dict:
    """Legacy ``estimated_joint.json`` for splatfacto-art, expressed in the blend frame B = N_BW W.

    Valid only because A_B(t_ref) = I and both parts are fused in the same canonical frame (checked by the
    caller); hence ``global_transform = I``. Prismatic origin is the legacy zero vector (adapter only)."""
    N = np.asarray(N_BW, float)
    if np.abs(N[:3, :3].T @ N[:3, :3] - np.eye(3)).max() > 1e-6 or abs(np.linalg.det(N[:3, :3]) - 1) > 1e-6:
        raise AuthorityError("N_BW must be rigid (no hidden scale) for this adapter")
    if a.reference_time_index != 0:
        raise AuthorityError("the legacy splatfacto-art frame indexing needs reference_time_index == 0")
    axis_B = N[:3, :3] @ a.axis_world
    origin_B = (N @ np.r_[a.origin_world, 1.0])[:3] if a.joint_type == "revolute" else np.zeros(3)
    return dict(joint_type=a.joint_type, axis=axis_B.tolist(), origin=origin_B.tolist(),
                joint_params=np.asarray(a.q, float).tolist(), global_transform=np.eye(4).tolist(),
                num_frames=a.T, authority_id=authority_id, reference_time_index=a.reference_time_index,
                q_units=a.q_units, frame_convention="blend_frame_B = N_BW * W (rigid, metres)",
                N_BW=N.tolist(), derived_adapter=True)


def frame_alignment(C_ref: np.ndarray, C_other: np.ndarray, tol_rot=1e-5, tol_t=1e-5) -> np.ndarray:
    """Single rigid N with ``C_other[i] = N C_ref[i]`` for all i; raises unless one N explains every pose."""
    C_ref, C_other = np.asarray(C_ref, float), np.asarray(C_other, float)
    Ns = C_other @ np.linalg.inv(C_ref)
    N = Ns[0]
    err = np.abs(Ns - N[None]).max()
    if err > max(tol_rot, tol_t):
        raise AuthorityError(f"camera sets are not related by one rigid transform (max deviation {err:.2e})")
    return N
