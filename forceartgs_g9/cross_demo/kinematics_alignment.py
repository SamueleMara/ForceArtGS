"""Common joint coordinate between a TRAINING episode i and a TARGET episode j (spec section 4), checked BEFORE replay.

Both joints are expressed in the robot base frame B (the per-episode registration of the capture world; the wrench
record stores axis_B / origin_B in the same frame as the end-effector poses). Nothing here looks at the target's
future motion or at a simulation residual: only the two fixed, independently reconstructed joints.

  type        joint_type_i == joint_type_j, else KINEMATICS_TYPE_MISMATCH (never kg against kg m^2, m against rad)
  prismatic   axes are unoriented lines: sign = sign(a_i . a_j); |angle| <= axis_tol_deg
  revolute    axis angle (sign-resolved) <= axis_tol_deg AND hinge-line distance <= hinge_tol_m
  gauge       q_ref_in_train(t) = sign * (q_j(t) - q0_j): displacement from the target's own initial rest plateau;
              the simulation starts at 0 in the same displacement coordinate. No fitted scale, offset or time lag.
Tolerances are PREDECLARED in configs/g9_real/cross_demo_validation.yaml (not tuned on transfer errors).
"""
from __future__ import annotations

import math

import numpy as np


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n <= 0:
        raise ValueError("joint axis must be a finite nonzero vector")
    return v / n


def line_distance(a1, o1, a2, o2) -> float:
    """Distance between the two (infinite) hinge lines."""
    a1, a2 = _unit(a1), _unit(a2)
    o1, o2 = np.asarray(o1, float), np.asarray(o2, float)
    n = np.cross(a1, a2)
    if np.linalg.norm(n) < 1e-9:                                  # parallel: point-to-line distance
        d = o2 - o1
        return float(np.linalg.norm(d - (d @ a1) * a1))
    return float(abs((o2 - o1) @ n) / np.linalg.norm(n))


def align(train: dict, test: dict, *, axis_tol_deg: float, hinge_tol_m: float) -> dict:
    """train / test: dict(joint_type, axis_B, origin_B). Returns status OK or a refusal, with the evidence."""
    out = dict(train_joint_type=train["joint_type"], test_joint_type=test["joint_type"],
               axis_tol_deg=float(axis_tol_deg), hinge_tol_m=float(hinge_tol_m), frame="robot base B")
    if train["joint_type"] != test["joint_type"]:
        return dict(out, status="KINEMATICS_TYPE_MISMATCH", sign=None,
                    reason=f"train {train['joint_type']} vs test {test['joint_type']}: not comparable")
    ai, aj = _unit(train["axis_B"]), _unit(test["axis_B"])
    dot = float(np.clip(ai @ aj, -1.0, 1.0))
    sign = 1.0 if dot >= 0 else -1.0
    ang = math.degrees(math.acos(abs(dot)))
    out.update(sign=sign, axis_angle_deg=ang, axis_dot=dot)
    if ang > axis_tol_deg:
        return dict(out, status="AXIS_MISMATCH", reason=f"axis angle {ang:.2f} deg > {axis_tol_deg:g} deg")
    if train["joint_type"] == "revolute":
        if train.get("origin_B") is None or test.get("origin_B") is None:
            return dict(out, status="HINGE_UNAVAILABLE", reason="revolute without a hinge point")
        d = line_distance(ai, train["origin_B"], aj, test["origin_B"])
        out["hinge_line_distance_m"] = d
        if d > hinge_tol_m:
            return dict(out, status="HINGE_MISMATCH", reason=f"hinge lines {d:.4f} m apart > {hinge_tol_m:g} m")
    return dict(out, status="OK", gauge="q_ref_in_train = sign * (q_test - q0_test); simulation from 0",
                reason="same joint type, axis (and hinge) within the predeclared tolerances")


def reference_in_train_gauge(q_test, q0_test: float, sign: float) -> np.ndarray:
    """Target visual q as a displacement in the TRAINING joint coordinate (only the first-rest value enters)."""
    return float(sign) * (np.asarray(q_test, float) - float(q0_test))
