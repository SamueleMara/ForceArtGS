"""Generalized effort of a measured force on a 1-DoF joint (ForceArtGS's own implementation).

    prismatic : Q = a . F                      [N]
    revolute  : Q = a . ((p - o) x F)          [N m]

a = unit joint axis, o = a point on the axis, p = force application point, F = force ON the moving part, all in the
same frame. Positive Q drives positive q (same axis convention as the joint kinematics).
"""
from __future__ import annotations

import numpy as np


def generalized_effort(joint_type: str, axis, origin, force, point=None) -> np.ndarray:
    a = np.asarray(axis, float).reshape(3)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n <= 0:
        raise ValueError("joint axis must be a finite non-zero vector")
    a = a / n
    F = np.asarray(force, float).reshape(-1, 3)
    if joint_type == "prismatic":
        return F @ a
    if joint_type != "revolute":
        raise ValueError(f"joint_type must be revolute or prismatic, got {joint_type!r}")
    if point is None:
        raise ValueError("revolute effort needs the force application point")
    P = np.asarray(point, float).reshape(-1, 3)
    if len(P) != len(F):
        raise ValueError("one application point per force sample")
    o = np.zeros(3) if origin is None else np.asarray(origin, float).reshape(3)
    return np.cross(P - o, F) @ a
