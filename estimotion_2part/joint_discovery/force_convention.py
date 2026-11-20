"""The ONE force sign / rotation contract (G9 corrections section 1), shared by the joint frontend
(``wrench_physics``, ``wrench_calibration``) and G9 (``robot_record``).

The sensors record the force of the OBJECT ON THE ROBOT. The force acting on the object is its opposite:

    F_robot->object^B(t) = s R_B,EE(t) R_EE,S F_sensor_corr^S(t),     s = -1 for the recorded datasets

``F_sensor_corr`` already has the initial tare removed (``export_force_contact_annotations.py``); it is never
re-tared and never negated before this function. The sign comes ONLY from explicit scene metadata
(``object_side_force_multiplier`` in {-1, +1}, agreeing with ``recorded_quantity`` / ``force_acting_on`` when those
are given); never from a fit, a positive-work score or a joint preference. Missing or malformed metadata raises.

Force stages are typed so a converted force cannot be converted again:
    F_sensor_raw (archive)  ->  SensorCorrForce (export, tare removed)  ->  ObjectBaseForce (signed, rotated)  ->  Q
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONVERSION_VERSION = "forceartgs.force_convention/1"
QUANTITY_SIGN = {"object_on_robot": -1.0, "robot_on_object": 1.0}


class ForceConventionError(ValueError):
    """Missing / malformed / contradictory force convention (fail closed, never a fallback)."""


def sign_from_recorded_quantity(q: str) -> float:
    if q not in QUANTITY_SIGN:
        raise ForceConventionError(f"unknown recorded wrench quantity {q!r}; refusing to guess a sign")
    return QUANTITY_SIGN[q]


def _quantity_of(conv: dict):
    """Normalised recorded quantity from the convention text fields (None if none is given)."""
    q = str(conv.get("recorded_quantity") or "")
    on = conv.get("force_acting_on")
    if "on_robot" in q or on == "robot":
        rq = "object_on_robot"
    elif "on_object" in q or on == "object":
        rq = "robot_on_object"
    else:
        rq = None
    if ("on_robot" in q and on == "object") or ("on_object" in q and on == "robot"):
        raise ForceConventionError(f"contradictory convention: recorded_quantity {q!r} vs force_acting_on {on!r}")
    return rq


def object_side_sign(conv: dict, *, source: str = "metadata/contact_force_convention.json") -> tuple:
    """(sign, provenance) from an explicit ``object_side_force_multiplier``; raises on anything else."""
    if not isinstance(conv, dict) or "object_side_force_multiplier" not in conv:
        raise ForceConventionError(f"{source}: object_side_force_multiplier is absent (no implicit default)")
    m = conv["object_side_force_multiplier"]
    if isinstance(m, bool) or not isinstance(m, (int, float)) or not math.isfinite(float(m)) or float(m) not in (
            -1.0, 1.0):
        raise ForceConventionError(f"{source}: object_side_force_multiplier must be -1 or +1, got {m!r}")
    rq = _quantity_of(conv)
    if rq is not None and QUANTITY_SIGN[rq] != float(m):
        raise ForceConventionError(f"{source}: multiplier {m} contradicts the recorded quantity {rq}")
    return float(m), dict(sign=float(m), sign_source=source, recorded_quantity=rq or "not stated",
                          confirmation=conv.get("confirmation_source"), conversion_version=CONVERSION_VERSION)


def load_scene_convention(meta_dir) -> tuple:
    p = Path(meta_dir) / "contact_force_convention.json"
    if not p.is_file():
        raise ForceConventionError(f"{p}: missing (the force sign is never assumed)")
    sign, prov = object_side_sign(json.loads(p.read_text()), source=str(p))
    prov["metadata_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return sign, prov


def validate_rotation(R, name="rotation", tol=1e-5) -> np.ndarray:
    """3x3 (or [N,3,3]) finite proper rotations; raises otherwise."""
    R = np.asarray(R, float)
    if R.shape[-2:] != (3, 3) or not np.isfinite(R).all():
        raise ForceConventionError(f"{name}: expected finite 3x3 rotation(s), got shape {R.shape}")
    E = np.einsum("...ji,...jk->...ik", R, R) - np.eye(3)
    if np.abs(E).max() > tol:
        raise ForceConventionError(f"{name}: not orthogonal (max |R^T R - I| = {np.abs(E).max():.2e})")
    if np.min(np.linalg.det(R)) < 1 - 1e-4:
        raise ForceConventionError(f"{name}: det(R) != +1 (reflection)")
    return R


@dataclass(frozen=True)
class SensorCorrForce:
    """Tare-corrected force in SENSOR axes (from the existing export). The only accepted conversion input."""
    values: np.ndarray
    source: str = ""
    tare: str = "initial baseline removed by export_force_contact_annotations.py"


@dataclass(frozen=True)
class ObjectBaseForce:
    """Force BY THE ROBOT ON THE OBJECT in the robot base frame. Never a conversion input."""
    values: np.ndarray
    provenance: dict = field(default_factory=dict)


def sensor_corr_to_object_base(f: SensorCorrForce, R_base_ee, R_ee_sensor, sign: float, *,
                               rotation_source: str = "declared", sign_prov: dict | None = None) -> ObjectBaseForce:
    """F_obj^B = s R_B,EE R_EE,S F_sensor_corr (applied exactly once)."""
    if not isinstance(f, SensorCorrForce):
        raise TypeError(f"sensor_corr_to_object_base accepts only SensorCorrForce, got {type(f).__name__} "
                        "(an object-side force must never be converted again)")
    if float(sign) not in (-1.0, 1.0):
        raise ForceConventionError(f"sign must be -1 or +1, got {sign!r}")
    R_b = validate_rotation(R_base_ee, "R_base_ee")
    R_s = validate_rotation(R_ee_sensor, "R_ee_sensor")
    F = np.asarray(f.values, float)
    out = float(sign) * np.einsum("nij,jk,nk->ni", R_b, R_s, F)
    prov = dict(sign_prov or {}, sign=float(sign), rotation_source=rotation_source,
                R_ee_sensor=R_s.tolist(), conversion_version=CONVERSION_VERSION, input_stage="F_sensor_corr",
                output_stage="F_object_base", source=f.source)
    return ObjectBaseForce(out, prov)


def generalized_effort(joint_type, axis, origin, F_obj: ObjectBaseForce, p_tcp=None) -> np.ndarray:
    """Q_P = a.F (contact point irrelevant), Q_R = a.((p - o) x F). Signed, SI, from the object-side force."""
    if not isinstance(F_obj, ObjectBaseForce):
        raise TypeError("generalized_effort needs an ObjectBaseForce")
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    F = np.asarray(F_obj.values, float)
    if joint_type == "prismatic":
        return F @ a
    if p_tcp is None or origin is None:
        raise ValueError("revolute effort needs the contact point and the hinge origin")
    return np.einsum("j,nj->n", a, np.cross(np.asarray(p_tcp, float) - np.asarray(origin, float), F))
