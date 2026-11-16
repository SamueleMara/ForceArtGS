"""Exact SO(3)/SE(3) kinematics for 1-DOF revolute and prismatic joints.

Conventions (asserted by tests/test_joint_discovery_se3.py and _twist.py):

* ``T_AB`` maps coordinates in frame B to frame A, column vectors, ``x_A = T_AB @ [x_B; 1]``.
* Twists are ordered ``xi = [omega; v]`` (rotation first).
* Angles are radians, lengths are metres.
* Revolute: ``A_R(theta) x = o + R_u(theta) (x - o)`` = ``Exp_SE3(theta [u; -u x o])``.
* Prismatic: ``A_P(d) x = x + d u``.
* Every function accepts NumPy arrays or torch tensors (same backend in, same backend out) so that
  torch autograd can differentiate the same code that NumPy fitting evaluates.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

try:  # torch is optional for the NumPy path
    import torch
except ImportError:  # pragma: no cover
    torch = None

JointKind = Literal["revolute", "prismatic"]
_SMALL = 1e-8


# --------------------------------------------------------------------------- backend helpers
def _is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _asarray(x, like=None):
    if like is not None and _is_torch(like):
        return x if _is_torch(x) else torch.as_tensor(x, dtype=like.dtype, device=like.device)
    if _is_torch(x):
        return x
    return np.asarray(x, dtype=float)


def _cross(a, b):
    if _is_torch(a) or _is_torch(b):
        a, b = _asarray(a, b if _is_torch(b) else a), _asarray(b, a if _is_torch(a) else b)
        a, b = torch.broadcast_tensors(a, b)
        return torch.linalg.cross(a, b, dim=-1)
    return np.cross(a, b)


def _norm(x, keepdims=False):
    if _is_torch(x):
        return torch.linalg.norm(x, dim=-1, keepdim=keepdims)
    return np.linalg.norm(x, axis=-1, keepdims=keepdims)


def _sum(x, axis=-1, keepdims=False):
    if _is_torch(x):
        return x.sum(dim=axis, keepdim=keepdims)
    return x.sum(axis=axis, keepdims=keepdims)


def _eye(n, like):
    if _is_torch(like):
        return torch.eye(n, dtype=like.dtype, device=like.device)
    return np.eye(n)


def _stack(xs, axis):
    if any(_is_torch(x) for x in xs):
        like = next(x for x in xs if _is_torch(x))
        return torch.stack([_asarray(x, like) for x in xs], dim=axis)
    return np.stack(xs, axis=axis)


def _where(c, a, b):
    if _is_torch(c) or _is_torch(a) or _is_torch(b):
        return torch.where(c, a, b)
    return np.where(c, a, b)


def _sin(x):
    return torch.sin(x) if _is_torch(x) else np.sin(x)


def _cos(x):
    return torch.cos(x) if _is_torch(x) else np.cos(x)


def _matvec(M, x):
    """``M @ x`` for stacks: M [...,3,3], x [...,3]."""
    if _is_torch(M) or _is_torch(x):
        like = M if _is_torch(M) else x
        return (_asarray(M, like) @ _asarray(x, like).unsqueeze(-1)).squeeze(-1)
    return np.einsum("...ij,...j->...i", M, x)


def _transpose(M):
    return M.transpose(-1, -2) if _is_torch(M) else np.swapaxes(M, -1, -2)


# --------------------------------------------------------------------------- SO(3)
def normalize(v):
    v = _asarray(v)
    return v / _norm(v, keepdims=True)


def skew(w):
    """[...,3] -> [...,3,3] with ``skew(w) @ x == w x x``."""
    w = _asarray(w)
    z = w[..., 0] * 0
    return _stack([
        _stack([z, -w[..., 2], w[..., 1]], -1),
        _stack([w[..., 2], z, -w[..., 0]], -1),
        _stack([-w[..., 1], w[..., 0], z], -1),
    ], -2)


def exp_so3(rotvec):
    """Rodrigues' formula with a Taylor expansion near zero (differentiable at 0)."""
    rotvec = _asarray(rotvec)
    theta2 = _sum(rotvec * rotvec)
    theta = (theta2 + 1e-30) ** 0.5
    small = theta2 < 1e-12
    safe_t = _where(small, theta * 0 + 1.0, theta)
    a = _where(small, 1 - theta2 / 6 + theta2 ** 2 / 120, _sin(safe_t) / safe_t)
    b = _where(small, 0.5 - theta2 / 24 + theta2 ** 2 / 720, (1 - _cos(safe_t)) / (safe_t * safe_t))
    K = skew(rotvec)
    I = _eye(3, rotvec)
    return I + a[..., None, None] * K + b[..., None, None] * (K @ K)


def exp_so3_axis_angle(axis, theta):
    axis = _asarray(axis)
    theta = _asarray(theta, axis)
    return exp_so3(axis * theta[..., None])


def log_so3(R):
    """Rotation vector of R (NumPy only; used for initialization/diagnostics)."""
    R = np.asarray(R, dtype=float)
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(theta)
    out = np.where((theta < 1e-6)[..., None], 0.5 * w, (theta / np.where(np.abs(s) < 1e-12, 1.0, 2 * s))[..., None] * w)
    near_pi = theta > np.pi - 1e-4
    if np.any(near_pi):  # axis from the symmetric part
        Rs = R[near_pi]
        B = (Rs + np.swapaxes(Rs, -1, -2)) / 2 - np.eye(3) * np.cos(theta[near_pi])[..., None, None]
        idx = np.argmax(np.diagonal(B, axis1=-2, axis2=-1), axis=-1)
        ax = normalize(B[np.arange(len(idx)), idx])
        sign = np.sign(np.sum(ax * w[near_pi], -1) + 1e-30)
        out[near_pi] = ax * (theta[near_pi] * sign)[..., None]
    return out


# --------------------------------------------------------------------------- SE(3)
def make_se3(R, t):
    R = _asarray(R)
    t = _asarray(t, R)
    top = _stack([R[..., 0, :], R[..., 1, :], R[..., 2, :]], -2)
    top = torch.cat([top, t[..., None]], -1) if _is_torch(top) else np.concatenate([top, t[..., None]], -1)
    bottom = top[..., :1, :] * 0
    if _is_torch(bottom):
        bottom = bottom.clone()
        bottom[..., 0, 3] = 1.0
        return torch.cat([top, bottom], -2)
    bottom = bottom.copy()
    bottom[..., 0, 3] = 1.0
    return np.concatenate([top, bottom], -2)


def se3_inverse(T):
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = _transpose(R)
    return make_se3(Rt, -_matvec(Rt, t))


def transform_points(T, X):
    """Apply [...,4,4] to points [...,3]."""
    return _matvec(T[..., :3, :3], X) + T[..., :3, 3]


def exp_se3(xi, theta=1.0):
    """``Exp([omega; v] * theta)`` for any twist (not only unit / zero-pitch)."""
    xi = _asarray(xi)
    theta = _asarray(theta, xi)
    phi = xi[..., :3] * theta[..., None]
    rho = xi[..., 3:] * theta[..., None]
    R = exp_so3(phi)
    a2 = _sum(phi * phi)
    a = (a2 + 1e-30) ** 0.5
    small = a2 < 1e-12
    safe = _where(small, a * 0 + 1.0, a)
    c1 = _where(small, 0.5 - a2 / 24, (1 - _cos(safe)) / (safe * safe))
    c2 = _where(small, 1.0 / 6 - a2 / 120, (safe - _sin(safe)) / (safe * safe * safe))
    K = skew(phi)
    V = _eye(3, phi) + c1[..., None, None] * K + c2[..., None, None] * (K @ K)
    return make_se3(R, _matvec(V, rho))


def revolute_twist(axis, origin):
    """Unit zero-pitch twist ``[u; -u x o]`` of the hinge line (u, o)."""
    u = normalize(axis)
    o = _asarray(origin, u)
    v = -_cross(u, o)
    return torch.cat([u, v], -1) if _is_torch(u) else np.concatenate([u, v], -1)


def A_revolute(axis_w, origin_w, theta_rad):
    """Exact hinge transform ``[R, (I-R) o; 0, 1]``; theta [...] -> [...,4,4]."""
    u = normalize(axis_w)
    o = _asarray(origin_w, u)
    theta = _asarray(theta_rad, u)
    R = exp_so3_axis_angle(u, theta)
    I = _eye(3, R)
    t = _matvec(I - R, o * (theta[..., None] * 0 + 1))
    return make_se3(R, t)


def A_prismatic(axis_w, displacement_m):
    u = normalize(axis_w)
    d = _asarray(displacement_m, u)
    t = u * d[..., None]
    R = _eye(3, t) + t[..., :1, None] * 0
    return make_se3(R, t)


def A_joint(kind: JointKind, axis_w, origin_w, q):
    if kind == "revolute":
        return A_revolute(axis_w, origin_w, q)
    if kind == "prismatic":
        return A_prismatic(axis_w, q)
    raise ValueError(f"unknown joint kind {kind!r}")


def apply_joint(kind: JointKind, axis_w, origin_w, q, X):
    """``A_H(q) X`` without forming 4x4 matrices (q [...], X [...,3])."""
    u = normalize(axis_w)
    q = _asarray(q, u)
    X = _asarray(X, u)
    if kind == "prismatic":
        return X + u * q[..., None]
    o = _asarray(origin_w, u)
    R = exp_so3_axis_angle(u, q)
    return o + _matvec(R, X - o)


def apply_joint_inverse(kind: JointKind, axis_w, origin_w, q, X):
    q = _asarray(q, _asarray(axis_w))
    return apply_joint(kind, axis_w, origin_w, -q, X)


# --------------------------------------------------------------------------- velocities and contact
def body_velocity(kind: JointKind, axis_w, origin_w, qdot, point_w):
    """Material velocity of the moving part at the world point ``point_w`` (instantaneous)."""
    u = normalize(axis_w)
    qdot = _asarray(qdot, u)
    p = _asarray(point_w, u)
    if kind == "prismatic":
        return u * qdot[..., None] + p * 0
    o = _asarray(origin_w, u)
    return _cross(u * qdot[..., None], p - o)


def canonical_contact_point(kind: JointKind, axis_w, origin_w, q, tcp_point_w):
    """``x0 = A_H(q)^-1 p_tcp``: the material coordinate currently under the TCP (slides over time)."""
    return apply_joint_inverse(kind, axis_w, origin_w, q, tcp_point_w)


def relative_slip(kind: JointKind, axis_w, origin_w, qdot, tcp_point_w, tcp_velocity_w):
    """``s_H = v_tcp - v_M,H(p_tcp)`` in the world frame."""
    v_m = body_velocity(kind, axis_w, origin_w, qdot, tcp_point_w)
    return _asarray(tcp_velocity_w, v_m) - v_m


def contact_jacobian(kind: JointKind, axis_w, origin_w, point_w):
    """``J_c,H(p)`` with ``v_M = J qdot``: ``u x (p-o)`` (R) or ``u`` (P)."""
    u = normalize(axis_w)
    p = _asarray(point_w, u)
    if kind == "prismatic":
        return u + p * 0
    return _cross(u, p - _asarray(origin_w, u))


def decompose_revolute_velocity(axis_w, origin_w, points_w, velocities_w):
    """Velocity components in the local cylindrical basis (e_r, e_theta, u).

    Returns dict(radial, tangential, axial, radius, valid). Points within 1e-9 m of the hinge
    have no defined e_r and are flagged invalid (unobservable angular rate)."""
    u = normalize(np.asarray(axis_w, float))
    o = np.asarray(origin_w, float)
    X = np.asarray(points_w, float)
    v = np.asarray(velocities_w, float)
    r_vec = (X - o) - np.sum((X - o) * u, -1, keepdims=True) * u
    r = np.linalg.norm(r_vec, axis=-1)
    valid = r > 1e-9
    e_r = r_vec / np.where(valid, r, 1.0)[..., None]
    e_t = np.cross(u, e_r)
    return dict(radial=np.sum(v * e_r, -1), tangential=np.sum(v * e_t, -1), axial=v @ u,
                radius=r, valid=valid)


def decompose_prismatic_velocity(axis_w, velocities_w):
    u = normalize(np.asarray(axis_w, float))
    v = np.asarray(velocities_w, float)
    par = v @ u
    perp = v - par[..., None] * u
    return dict(parallel=par, perpendicular=perp, perpendicular_norm=np.linalg.norm(perp, axis=-1))


# --------------------------------------------------------------------------- gauge / comparison
def tangent_basis(axis):
    """Deterministic orthonormal (e1, e2) perpendicular to ``axis`` (NumPy)."""
    a = normalize(np.asarray(axis, float))
    ref = np.eye(3)[int(np.argmin(np.abs(a)))]
    e1 = normalize(np.cross(a, ref))
    e2 = np.cross(a, e1)
    return e1, e2


def hinge_gauge(axis_w, origin_w, center_w):
    """Point of the hinge line closest to ``center_w`` (removes the along-axis gauge)."""
    u = normalize(np.asarray(axis_w, float))
    o = np.asarray(origin_w, float)
    c = np.asarray(center_w, float)
    return o + np.dot(c - o, u) * u


def axis_angle_deg(a, b):
    """Sign-invariant angle between two axes in degrees."""
    a, b = normalize(np.asarray(a, float)), normalize(np.asarray(b, float))
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def line_to_line_distance(u1, o1, u2, o2):
    """Distance between two (possibly skew) lines; for near-parallel lines the point-line distance."""
    u1, u2 = normalize(np.asarray(u1, float)), normalize(np.asarray(u2, float))
    o1, o2 = np.asarray(o1, float), np.asarray(o2, float)
    n = np.cross(u1, u2)
    nn = np.linalg.norm(n)
    d = o2 - o1
    if nn < 1e-6:
        return float(np.linalg.norm(d - np.dot(d, u1) * u1))
    return float(abs(np.dot(d, n)) / nn)


def canonicalize_sign(axis, q, reference=None):
    """Apply the documented sign convention ``(u, q) ~ (-u, -q)``.

    Without a reference the axis is flipped so that its largest-magnitude component is positive.
    This is a display convention only; it never uses ground truth."""
    a = np.asarray(axis, float)
    ref = np.asarray(reference, float) if reference is not None else np.eye(3)[int(np.argmax(np.abs(a)))]
    if np.dot(a, ref) < 0:
        return -a, -np.asarray(q, float)
    return a, np.asarray(q, float)


def is_rigid(T, atol=1e-9) -> bool:
    T = np.asarray(T, float)
    R = T[..., :3, :3]
    ok = np.allclose(np.swapaxes(R, -1, -2) @ R, np.eye(3), atol=atol)
    ok &= np.allclose(np.linalg.det(R), 1.0, atol=atol)
    ok &= np.allclose(T[..., 3, :], [0, 0, 0, 1], atol=atol)
    return bool(ok)
