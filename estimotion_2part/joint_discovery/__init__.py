"""Calibrated, joint-aware Phase-1 motion discovery (opt-in; ``joint_discovery.mode: legacy`` is the default).

See ``estimotion_2part/joint_discovery/README.md`` and
``FREEARTGS_PHASE1_JOINT_AWARE_MOTION_TCP_FORCE_CODEX_SPEC.md``.
"""
from .config import JointDiscoveryConfig, load_joint_discovery_config, resolve_joint_discovery
from .geometry_fit import JointCandidate
from .model_selection import ModelDecision, fit_joint_hypotheses, score_candidates_on_heldout
from .observations import CameraCalib, FrameMap, TrackObservations
from .rigid_kinematics import (A_prismatic, A_revolute, body_velocity, canonical_contact_point,
                               relative_slip)
from .exports import export_for_existing_freeart
from .tcp_contact import load_robot_contact_record

__all__ = [
    "JointDiscoveryConfig", "load_joint_discovery_config", "resolve_joint_discovery", "JointCandidate",
    "ModelDecision", "fit_joint_hypotheses", "score_candidates_on_heldout", "CameraCalib", "FrameMap",
    "TrackObservations", "A_prismatic", "A_revolute", "body_velocity", "canonical_contact_point", "relative_slip",
    "export_for_existing_freeart", "load_robot_contact_record",
]
