"""G10-B: physical robot-action / contact replay -- explicit backend CONTRACT, no backend implemented here.

G10-B replays the recorded robot COMMAND (its true control mode, timing and gains) in an interacting PhysX scene and
lets the engine compute contact, slip and compliance; the object's simulated q(t) is then compared with the observed
one. It is NOT: prescribing the TCP pose (teleportation), replaying the measured joint torque (that is G10-A), or
inventing a controller / contact force. Native Isaac Sim / Isaac Lab per-axis friction needs a version-checked
runtime with readback; none is installed on this machine.

``detect_backends(record_context)`` reports what is actually available. Every unavailable route is
NOT_IMPLEMENTED_BACKEND with the missing ingredients listed; G10-A stays the working baseline.
"""
from __future__ import annotations

import abc
import importlib.util


class RobotInteractionReplayBackend(abc.ABC):
    """Required inputs: recorded robot command + control mode + timing; validated robot geometry and controller gains;
    calibrated tool / TCP; object collision model, friction and compliance assumptions; canonical object initial
    configuration with v0 = 0; the candidate PhysX object parameters.
    Required outputs: simulated object q(t), robot state, contacts, applied joint effort, collision events and the
    physical controller provenance."""

    name: str = "abstract"

    @abc.abstractmethod
    def capabilities(self) -> dict: ...

    @abc.abstractmethod
    def replay(self, *, robot_command, control_mode, controller_gains, robot_model, tool_model, object_asset,
               object_parameters, q0, v0=0.0) -> dict: ...


def _has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def detect_backends(record_context=None) -> dict:
    rec = (record_context or {}).get("rec") or {}
    keys = set(rec.keys()) if hasattr(rec, "keys") else set()
    have_pose = {"p_tcp_base_m", "R_base_ee"} <= keys
    missing_cmd = [k for k in ("robot_command", "control_mode", "controller_gains", "robot_urdf")
                   if k not in keys]
    isaac = {m: _has(m) for m in ("isaacsim", "omni.isaac.core", "isaaclab")}
    rows = dict(
        robot_action_contact_replay=dict(
            status="NOT_IMPLEMENTED_BACKEND",
            missing=missing_cmd + ["object collision mesh (3DGS point clouds are not collision geometry)",
                                   "contact friction / compliance model"],
            note="the canonical record has no robot command / controller gains; replaying TCP poses would be a "
                 "kinematic-path replay, not a physical robot-action replay"),
        kinematic_path_replay=dict(
            status="NOT_IMPLEMENTED_BACKEND" if have_pose else "DATA_UNAVAILABLE",
            available_inputs=["p_tcp_base_m", "R_base_ee"] if have_pose else [],
            note="possible future DIAGNOSTIC only (prescribed TCP path, engine contacts); never sim-to-real "
                 "validation"),
        native_isaac_axis_friction=dict(
            status="NOT_IMPLEMENTED_BACKEND", modules_found=isaac,
            note="needs a native Isaac runtime with staticFrictionEffort / dynamicFrictionEffort / "
                 "viscousFrictionCoefficient set AND read back, verified by tests; not added to freeartgs"),
        measured_effort_replay=dict(status="AVAILABLE", backend="G10-A IsolatedSapienCpu (SAPIEN 3 PhysX CPU, "
                                    "explicit friction surrogate)"))
    return dict(schema="forceartgs.g10.robot_replay_backends/v1", backends=rows,
                g10b_status="NOT_IMPLEMENTED_BACKEND")
