"""Known TCP as a *sliding* contact point (spec section 7).

The calibrated TCP is the instantaneous contact location during independently verified contact.
It is not a material point: ``x0_H(t) = A_H(q_t)^-1 p_tcp(t)`` changes with time when the tip
slides. Maintained contact constrains only the normal component ``n^T (v_tcp - v_M) = 0`` and only
when a contact surface is independently supported by observed moving-part geometry. Without such
support every surface term is disabled with an explicit reason; no plane is fabricated from the
TCP curve itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .rigid_kinematics import (apply_joint, apply_joint_inverse, body_velocity, canonical_contact_point,
                               exp_so3_axis_angle, normalize, relative_slip)

UNAVAILABLE = "UNAVAILABLE"


# --------------------------------------------------------------------------- records
@dataclass
class ContactRecord:
    """Robot TCP in the reconstruction world frame, aligned to model time indices."""
    timestamp_s: np.ndarray        # [T] RGB-D timestamps of the model times
    p_tcp_w: np.ndarray            # [T,3] metres, NaN when unavailable
    v_tcp_w: np.ndarray            # [T,3] m/s from irregular-time differences, NaN at gaps
    contact_valid: np.ndarray      # [T] independently verified contact (never inferred from the fit)
    wrench_sensor: np.ndarray | None = None   # [T,6] raw recorded (F, tau) in sensor axes
    R_WS: np.ndarray | None = None            # [T,3,3] sensor axes -> world
    T_WB: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_WB_source: str = "unspecified"
    contact_source: str = "unspecified"
    sync: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    p_ee_w: np.ndarray | None = None          # [T,3] measured EE (used only for robot-tool exclusion)


def tcp_velocity(p, timestamps):
    """Non-uniform 3-point derivative; NaN where a neighbour is missing."""
    p = np.asarray(p, float)
    t = np.asarray(timestamps, float)
    v = np.full_like(p, np.nan)
    T = len(t)
    for i in range(T):
        if 0 < i < T - 1 and np.isfinite(p[[i - 1, i, i + 1]]).all():
            h1, h2 = t[i] - t[i - 1], t[i + 1] - t[i]
            v[i] = (-h2 / (h1 * (h1 + h2)) * p[i - 1] + (h2 - h1) / (h1 * h2) * p[i]
                    + h1 / (h2 * (h1 + h2)) * p[i + 1])
        elif i == 0 and T > 1 and np.isfinite(p[[0, 1]]).all():
            v[i] = (p[1] - p[0]) / (t[1] - t[0])
        elif i == T - 1 and T > 1 and np.isfinite(p[[T - 2, T - 1]]).all():
            v[i] = (p[-1] - p[-2]) / (t[-1] - t[-2])
    return v


def joint_rate(q, timestamps):
    return tcp_velocity(np.asarray(q, float)[:, None], timestamps)[:, 0]


def load_robot_contact_record(scene_root, frame_map, T_WB=None, tcp_npz=None, contact_flags=None, cfg=None) -> ContactRecord:
    """Load measured TCP (and raw wrench, if present) for the model times.

    Sources, in order: ``tcp_npz`` (needs ``T_base_tcp`` [N,4,4] and ``source_row`` [N]) or
    ``metadata/robot_poses/poses_rgbd.npz`` (``tcp_transform``, ``source_frame_index``). Rows are
    joined on the *source* frame index; any mismatch raises. ``T_WB`` must be given explicitly or
    proven from metadata: here it is accepted as identity only if every camera in
    ``metadata/cameras.json`` declares ``coordinate_frame == base_link``. Contact flags are never
    derived from the fit; without them ``contact_valid`` is all False."""
    scene_root = Path(scene_root)
    reasons = []
    if T_WB is None:
        meta = json.loads((scene_root / "metadata" / "cameras.json").read_text())
        frames = {c.get("coordinate_frame") for c in meta["cameras"]}
        if frames != {"base_link"}:
            raise ValueError(f"cannot assume T_WB = I: camera frames {frames}; pass T_WB explicitly")
        T_WB, src = np.eye(4), "identity: metadata/cameras.json poses are expressed in base_link"
    else:
        T_WB, src = np.asarray(T_WB, float), "explicit argument"
    wrench, R_WS = None, None
    if tcp_npz is not None:
        z = np.load(tcp_npz, allow_pickle=False)
        T_tcp = z["T_base_tcp"]
        rows = np.asarray(z["source_row"], int)
        T_ee = z["T_base_ee"] if "T_base_ee" in z.files else None
        wrench = z["wrench"] if "wrench" in z.files else None
        source = str(tcp_npz)
    else:
        path = scene_root / "metadata" / "robot_poses" / "poses_rgbd.npz"
        z = np.load(path, allow_pickle=False)
        T_tcp, rows = z["tcp_transform"], np.asarray(z["source_frame_index"], int)
        T_ee = z["ee_transform"] if "ee_transform" in z.files else None
        source = str(path)
    index = {int(r): i for i, r in enumerate(rows)}
    missing = [int(s) for s in frame_map.source_frame_index if int(s) not in index]
    if missing:
        raise KeyError(f"robot record lacks source frames {missing[:5]} (no interpolation is performed)")
    sel = np.array([index[int(s)] for s in frame_map.source_frame_index])
    p_b = T_tcp[sel, :3, 3]
    p_w = p_b @ T_WB[:3, :3].T + T_WB[:3, 3]
    v_w = tcp_velocity(p_w, frame_map.timestamp_s)
    if wrench is not None:
        wrench = np.asarray(wrench, float)[sel]
        if T_ee is not None:
            # wrench axes = EE axes for the recorded datasets (robot_tool.json/wrench_frame); a
            # different R_ee_sensor must be supplied by the caller via ``with_sensor_rotation``.
            R_WS = T_WB[None, :3, :3] @ T_ee[sel, :3, :3]
            reasons.append("wrench axes assumed equal to EE axes (robot_tool.json wrench_frame.R_ee_force_sensor = I)")
    flags = np.zeros(frame_map.T, bool) if contact_flags is None else np.asarray(contact_flags, bool)
    if contact_flags is None:
        reasons.append("no independently verified contact flags supplied: contact terms disabled")
    p_ee = None if T_ee is None else T_ee[sel, :3, 3] @ T_WB[:3, :3].T + T_WB[:3, 3]
    return ContactRecord(frame_map.timestamp_s, p_w, v_w, flags, wrench, R_WS, T_WB, src, source,
                         sync=dict(join="source_frame_index exact match", interpolation=False,
                                   hardware_sync_verified=False), reasons=reasons, p_ee_w=p_ee)


# --------------------------------------------------------------------------- kinematics of the contact
def contact_kinematics(cand, rec: ContactRecord) -> dict:
    """Canonical contact path, material velocity at the TCP, slip, per time (both hypotheses)."""
    q = np.asarray(cand.q, float)
    qd = joint_rate(q, rec.timestamp_s)
    x0 = canonical_contact_point(cand.kind, cand.axis_w, cand.origin_w, q, rec.p_tcp_w)
    v_m = body_velocity(cand.kind, cand.axis_w, cand.origin_w, qd, rec.p_tcp_w)
    s = relative_slip(cand.kind, cand.axis_w, cand.origin_w, qd, rec.p_tcp_w, rec.v_tcp_w)
    return dict(qdot=qd, canonical_contact=x0, v_material=v_m, slip=s,
                canonical_path_length_m=float(np.nansum(np.linalg.norm(np.diff(x0, axis=0), axis=1))))


# --------------------------------------------------------------------------- surface support
@dataclass
class SurfaceSupport:
    status: str                    # 'OK' or UNAVAILABLE
    reason: str
    n0: np.ndarray | None = None   # canonical outward normal (unit)
    a0: np.ndarray | None = None   # canonical plane point
    rms_m: float | None = None
    n_points: int = 0
    source: str = ""
    points: np.ndarray | None = None   # observed canonical points of the supporting face

    def support_count(self, x0, radius):
        """Observed face points within ``radius`` of each canonical contact point (geometry support,
        independent of contact residuals)."""
        x0 = np.atleast_2d(np.asarray(x0, float))
        if self.points is None or not len(self.points):
            return np.zeros(len(x0), int)
        d = np.linalg.norm(x0[:, None, :] - self.points[None], axis=2)
        return (d < radius).sum(1)


def surface_from_observed_patch(canonical_points, contact_point0, cfg, camera_centers0=None,
                                source="moving-support canonical anchors") -> SurfaceSupport:
    """Plane through *observed* moving-part points near the canonical contact point(s).

    ``contact_point0`` is one canonical point [3] or the canonical contact path [K,3] (the contact
    slides, so the face is gathered along the whole path).

    Rejected (UNAVAILABLE) when: too few observed points in the patch; the patch is not planar;
    the contact point is not on the fitted plane (e.g. the contact is on an unobserved underside
    while only the rim/top face is visible); or no camera sees the face's front side. The robot
    and the TCP curve never contribute points."""
    P = np.asarray(canonical_points, float)
    P = P[np.isfinite(P).all(1)]
    C = np.atleast_2d(np.asarray(contact_point0, float))
    C = C[np.isfinite(C).all(1)]
    if len(P) == 0 or len(C) == 0:
        return SurfaceSupport(UNAVAILABLE, "no observed moving-part points" if len(P) == 0 else "no contact points")
    d = np.min(np.linalg.norm(P[:, None, :] - C[None], axis=2), axis=1)
    near = P[d < cfg.surface_patch_radius_m]
    if len(near) < cfg.surface_min_points:
        return SurfaceSupport(UNAVAILABLE, f"only {len(near)} observed moving points within "
                                           f"{cfg.surface_patch_radius_m} m of the contact (hidden or unobserved face)",
                              n_points=len(near))
    m = near.mean(0)
    _, S, Vt = np.linalg.svd(near - m)
    n = Vt[2]
    rms = float(S[2] / np.sqrt(len(near)))
    if rms > cfg.surface_max_rms_m:
        return SurfaceSupport(UNAVAILABLE, f"patch not planar (rms {rms * 1000:.1f} mm)", rms_m=rms, n_points=len(near))
    off = float(np.median(np.abs((C - m) @ n)))
    if off > cfg.surface_max_distance_m:
        return SurfaceSupport(UNAVAILABLE, f"contact point is {off * 1000:.1f} mm off the observed face "
                                           f"(e.g. underside contact); nearest visible geometry rejected",
                              rms_m=rms, n_points=len(near))
    if camera_centers0 is not None:
        facing = [(n @ (np.asarray(cc) - m)) for cc in camera_centers0]
        if all(f < 0 for f in facing):
            n = -n
    return SurfaceSupport("OK", "observed planar moving-part patch", n, m, rms, len(near), source, points=near)


def surface_normal_world(kind, axis, q, n0):
    """``n_W(t) = R_H(q_t) n0`` (prismatic: constant)."""
    n0 = np.asarray(n0, float)
    if kind == "prismatic":
        return np.repeat(n0[None], len(q), 0)
    R = exp_so3_axis_angle(normalize(axis), np.asarray(q, float))
    return np.einsum("tij,j->ti", R, n0)


def contact_residuals(cand, rec: ContactRecord, surf: SurfaceSupport, cfg, common_mask=None) -> dict:
    """Point-on-surface (primary) and normal-slip (diagnostic, correlated) residuals.

    A contact sample is active only if it is verified contact AND observed face points lie within
    the patch radius of its canonical contact point (``surface_supported``). Pass ``common_mask``
    so every hypothesis is scored on the same samples."""
    T = len(rec.timestamp_s)
    out = dict(active=np.zeros(T, bool), reason=surf.reason if surf.status != "OK" else "")
    if surf.status != "OK":
        out["status"] = UNAVAILABLE
        return out
    kin = contact_kinematics(cand, rec)
    supported = surf.support_count(kin["canonical_contact"], cfg.surface_patch_radius_m) >= 3
    act = rec.contact_valid & np.isfinite(rec.p_tcp_w).all(1) & supported
    if common_mask is not None:
        act &= np.asarray(common_mask, bool)
    d = (kin["canonical_contact"] - surf.a0) @ surf.n0
    nW = surface_normal_world(cand.kind, cand.axis_w, cand.q, surf.n0)
    ns = np.einsum("ti,ti->t", nW, kin["slip"])
    out.update(status="OK", active=act, point_to_plane_m=np.where(act, d, np.nan),
               normal_slip_mps=np.where(act & np.isfinite(ns), ns, np.nan),
               L_contact=float(np.nansum((np.where(act, d, np.nan) / cfg.sigma_contact_m) ** 2)),
               L_normal_vel_diagnostic=float(np.nansum((np.where(act, ns, np.nan) / cfg.sigma_normal_velocity_mps) ** 2)),
               n_active=int(act.sum()), surface_supported=supported, normal_world=nW, kinematics=kin)
    return out


# --------------------------------------------------------------------------- degeneracy checks
def free_plane_degeneracy(canonical_path, tol_m=1e-3) -> dict:
    """Does *some* plane fit the canonical TCP path trivially? (spec 7.3; T15)

    A path of rank <= 2 (collinear or coplanar after centring) is always fitted by a free plane,
    so zero residual is not evidence for either hypothesis."""
    P = np.asarray(canonical_path, float)
    P = P[np.isfinite(P).all(1)]
    if len(P) < 3:
        return dict(degenerate=True, reason="fewer than 3 contact samples", singular_values=[])
    S = np.linalg.svd(P - P.mean(0), compute_uv=False)
    rms_plane = float(S[2] / np.sqrt(len(P)))
    return dict(degenerate=bool(rms_plane < tol_m), plane_rms_m=rms_plane, singular_values=S.tolist(),
                reason="free plane fits the inverse-transformed TCP path; not a joint-type classifier"
                if rms_plane < tol_m else "path is not planar")


def pin_in_slot(h=0.1, x_start=0.35, x_end=0.12, T=40, duration=4.0):
    """Analytic 2-D lid (hinge at origin, axis +z) pushed by a TCP moving on y = h (spec 7.5)."""
    t = np.linspace(0, duration, T)
    x = x_start + (x_end - x_start) * t / duration
    xd = np.full(T, (x_end - x_start) / duration)
    theta = np.arctan2(h, x)
    thetad = -h * xd / (x ** 2 + h ** 2)
    p = np.stack([x, np.full(T, h), np.zeros(T)], 1)
    v = np.stack([xd, np.zeros(T), np.zeros(T)], 1)
    return dict(t=t, p_tcp=p, v_tcp=v, theta=theta, thetadot=thetad, axis=np.array([0, 0, 1.0]),
                origin=np.zeros(3), radius=np.sqrt(x ** 2 + h ** 2))
