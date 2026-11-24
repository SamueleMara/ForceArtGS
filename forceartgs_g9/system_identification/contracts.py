"""Typed contracts of the input-output identification layer: ONE canonical episode schema, response features, the
seven separate status dimensions, the strict configuration (unknown keys rejected) and the no-``value`` rule."""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

SCHEMA = "forceartgs.g9.io_response/v1"
CONFIG_SCHEMA = "forceartgs.g9.io_identification/v1"

FEATURE_STATUS = ("IDENTIFIED", "CONDITIONAL", "RATIO_ONLY", "NOT_IDENTIFIED", "NOT_APPLICABLE")

STATUS_DIMENSIONS = {
    "INPUT_VALIDITY": ("VERIFIED", "CONDITIONAL", "CONTRADICTED", "UNKNOWN"),
    "KINEMATICS_VALIDITY": ("ACCEPTED", "CONDITIONAL", "UNACCEPTED"),
    "STRUCTURAL_IDENTIFIABILITY": ("FULL", "RATIO_ONLY", "RANK_DEFICIENT", "NOT_TESTABLE"),
    "PRACTICAL_IDENTIFIABILITY": ("INFORMATIVE", "WEAK", "BOUNDARY_ONLY", "UNSTABLE", "NOT_TESTABLE"),
    "MODEL_ADEQUACY": ("ADEQUATE", "INADEQUATE", "NOT_RUN"),
    "INDEPENDENT_VALIDATION": ("PASS", "FAIL", "NOT_AVAILABLE"),
    "SIMULATOR_READINESS": ("PRODUCTION_READY", "CONDITIONAL_PREVIEW", "BLOCKED"),
}

# exclusion reason codes (spec 3.3); the G9 classifier labels are mapped, never reinterpreted
EXCLUSION_CODES = ("CONTACT_UNKNOWN", "NEAR_ZERO_FORCE_IN_CONTACT", "HYPOTHESIS_ONLY", "TRANSITION", "LIMIT_CONTACT",
                   "UNSUPPORTED_KINEMATICS", "STICK_INTERVAL", "INPUT_NOT_ADMISSIBLE", "TOO_SHORT")

MODEL_VARIANTS = {
    "L0": "inertial_viscous",
    "L1": "inertial_viscous_coulomb",
    "L2": "inertial_viscous_coulomb_static",
    "L3": "inertial_viscous_coulomb_spring",
    "L4": "structured_residual",
}
VARIANT_BY_NAME = {v: k for k, v in MODEL_VARIANTS.items()}

UNITS = {
    "revolute": dict(q="rad", v="rad/s", a="rad/s^2", lam="1/s", gam="rad/s^2", g_u="1/(kg m^2)", kappa="1/s^2",
                     M="kg m^2", c="N m s/rad", Fc="N m", K="N m/rad", u="N m", W="J", J="N m s"),
    "prismatic": dict(q="m", v="m/s", a="m/s^2", lam="1/s", gam="m/s^2", g_u="1/kg", kappa="1/s^2",
                      M="kg", c="N s/m", Fc="N", K="N/m", u="N", W="J", J="N s"),
}


class IOContractError(ValueError):
    pass


@dataclass(frozen=True)
class ResponseFeature:
    name: str                     # lambda, gamma, gain, kappa, pole1, pole2, ...
    estimate: Optional[float]
    units: str
    status: str                   # IDENTIFIED | CONDITIONAL | RATIO_ONLY | NOT_IDENTIFIED | NOT_APPLICABLE
    uncertainty: dict
    evidence: list
    source_windows: list

    def __post_init__(self):
        if self.status not in FEATURE_STATUS:
            raise IOContractError(f"feature {self.name}: status {self.status!r} not in {FEATURE_STATUS}")
        if self.estimate is not None and not isinstance(self.estimate, complex) and not math.isfinite(self.estimate):
            raise IOContractError(f"feature {self.name}: non-finite estimate")

    def to_dict(self):
        d = asdict(self)
        e = d.pop("estimate")
        if isinstance(e, complex):
            d["estimate_real"], d["estimate_imag"] = e.real, e.imag
        else:
            d["estimate"] = e
        return d


@dataclass
class IOEpisode:
    """One admissible, MONOTONIC moving window from the G9 free/forced classifier, shared by every method.

    ``t, q, q_sigma`` are the native joint samples (from the start state on); ``t0, q0, v0`` the initial state
    (estimated on the observed prefix -- the same trajectory, NOT independent -- with sd ``sigma_q0, sigma_v0``);
    ``effort_t, effort`` the canonical object-side generalized effort on its native rows (forward ZOH, unchanged
    sign) -- None for a FREE window; ``segment`` is the full source MotionSegment (weak form / energy audit);
    ``g10`` the G10 PhysX Episode built from the SAME interaction (identical window and clock)."""

    name: str
    mode: str                      # free | forced
    joint_type: str
    t: np.ndarray
    q: np.ndarray
    q_sigma: np.ndarray
    t0: float
    q0: float
    v0: float
    sigma_q0: float
    sigma_v0: float
    sign: float
    effort_t: Optional[np.ndarray] = None
    effort: Optional[np.ndarray] = None
    effort_sigma: Optional[np.ndarray] = None
    train: bool = True
    calibration: str = "NOT_APPLICABLE"
    evidence: dict = field(default_factory=dict)
    segment: object = None
    g10: object = None

    def __post_init__(self):
        if self.mode not in ("free", "forced"):
            raise IOContractError(f"{self.name}: mode must be free or forced (UNKNOWN contact is never fit)")
        for k in ("t", "q", "q_sigma"):
            setattr(self, k, np.asarray(getattr(self, k), float))
        if len(self.t) < 3 or len(self.q) != len(self.t) or len(self.q_sigma) != len(self.t):
            raise IOContractError(f"{self.name}: q samples / sigmas must have equal length >= 3")
        if np.any(np.diff(self.t) <= 0) or np.any(self.q_sigma <= 0) or self.t[0] < self.t0 - 1e-12:
            raise IOContractError(f"{self.name}: increasing times >= t0 and positive sigmas required")
        if not (self.sigma_v0 > 0 and self.sigma_q0 > 0):
            raise IOContractError(f"{self.name}: the initial-state prior needs positive sd")
        if self.mode == "forced":
            if self.effort is None or self.effort_t is None:
                raise IOContractError(f"{self.name}: forced window without measured effort")
            self.effort_t = np.asarray(self.effort_t, float)
            self.effort = np.asarray(self.effort, float)
            if len(self.effort_t) != len(self.effort) or self.effort_t[0] > self.t0 + 1e-9:
                raise IOContractError(f"{self.name}: effort rows must start no later than the initial state")
            if not self.evidence.get("contact_confirmed"):
                raise IOContractError(f"{self.name}: FORCED needs confirmed contact evidence")
            if self.effort_sigma is not None:
                self.effort_sigma = np.asarray(self.effort_sigma, float)
        else:
            if self.effort is not None and np.max(np.abs(self.effort)) > 0:
                raise IOContractError(f"{self.name}: a FREE window carries zero applied effort by regime")
            if not self.evidence.get("contact_independently_absent"):
                raise IOContractError(f"{self.name}: FREE needs independent evidence of contact absence "
                                      "(near-zero wrench in contact is NOT free)")

    @property
    def forced(self):
        return self.mode == "forced"

    def zoh(self, scale=1.0, t_shift=0.0):
        """Forward-ZOH effort object (``scale`` x levels, timeline shifted by ``t_shift`` for audits only)."""
        from ..synthetic import ZOH
        if not self.forced:
            return ZOH(np.array([self.t0]), np.array([0.0]), model="free: zero applied effort")
        return ZOH(self.effort_t + t_shift, scale * self.effort, model="canonical object-side effort (ZOH)")

    def u_at(self, t):
        if not self.forced:
            return np.zeros(np.shape(t))
        i = np.clip(np.searchsorted(self.effort_t, np.asarray(t, float), side="right") - 1, 0, len(self.effort) - 1)
        return self.effort[i]

    def describe(self):
        return dict(name=self.name, mode=self.mode, joint_type=self.joint_type, train=self.train, n=len(self.t),
                    t=[float(self.t0), float(self.t[-1])], q_range=float(np.ptp(self.q)),
                    q0=self.q0, v0=self.v0, sigma_q0=self.sigma_q0, sigma_v0=self.sigma_v0, sign=self.sign,
                    calibration=self.calibration, n_effort_rows=0 if self.effort is None else len(self.effort),
                    effort_range=None if self.effort is None else [float(np.min(self.effort)),
                                                                  float(np.max(self.effort))],
                    initial_state_source=self.evidence.get("initial_state_source"),
                    regime_source=self.evidence.get("source"))


# ======================================================================================================== config
DEFAULT_CONFIG = {
    "schema": CONFIG_SCHEMA,
    "enabled": False,
    "joint_source": "accepted_authority",        # accepted_authority | diagnostic_working_authority (opt-in)
    "contact_policy": "inherit_g9_g10",
    "model_variants": ["inertial_viscous", "inertial_viscous_coulomb"],
    "gravitation": "inherited_geometry_and_world_gravity",
    "free_forced_config": "configs/g9_real/free_forced.yaml",
    "mass_mode": "uniform_solid",
    "heldout": "last_if_multiple",
    "fit": {
        "method": "output_error_and_weak_form",
        "initial_velocity": "uncertainty_aware_fit",
        "input_interpolation": "forward_zoh",
        "robust_loss": "huber",
        "huber_delta": 2.0,
        "feature_bootstrap_seeds": 8,
        "local_linearization": "moving_only",
        "v0_prefix_samples": 4,
        "min_samples": 4,
        "sigma_floor_q": 0.0,
        "lam_max": 50.0,
        "gam_max_factor": 50.0,
        "M_search": [1e-4, 1e4],
        "profile_points": 15,
        "n_starts": 4,
        "compare_methods": ["output_error", "weak_form", "normalized_ekf"],
        "spring_min_q_coverage": 0.3,
        "energy_lag_s": 0.02,
        "energy_lever_rel": 0.1,
        "energy_mc": 200,
    },
    "physx": {
        "inherit_g10_worker": True,
        "g10_config": "configs/g9_real/physx_refinement.yaml",
        "use_response_initialization": True,
        "response_regularizer_weight": 0.0,
        "energy_penalty_weight": 0.0,
        "disable_unexpected_link_damping": True,
        "disable_unexpected_joint_drive": True,
        "require_engine_property_readback": True,
        "mass_scale_profile": True,
        "reconcile": True,
        "reconcile_family_members": 3,
        "sensitivity": True,
        "engine_python": None,
    },
    "validation": {
        "synthetic_ground_truth_required": True,
        "real_heldout_required_for_production_export": True,
        "forbid_point_export_without_identifiability": True,
        "report_absolute_and_normalized_error": True,
    },
}

_FIXED_TRUE = ("disable_unexpected_link_damping", "disable_unexpected_joint_drive", "require_engine_property_readback")
_VALIDATION_FIXED = ("real_heldout_required_for_production_export", "forbid_point_export_without_identifiability")


def _check(d, ref, path=""):
    for k, v in d.items():
        if k not in ref:
            raise IOContractError(f"unknown io_identification config key {path + k!r}")
        if isinstance(ref[k], dict):
            if not isinstance(v, dict):
                raise IOContractError(f"{path + k} must be a mapping")
            _check(v, ref[k], path + k + ".")


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(src=None) -> dict:
    """Default config merged with ``src`` (path or mapping). Unknown keys, a mechanism head / spring that is not
    listed explicitly, a nonzero response / energy penalty weight, or a disabled engine-cleanliness check are
    rejected (they are not supported claims)."""
    user = {}
    if isinstance(src, (str, Path)):
        import yaml
        user = yaml.safe_load(Path(src).read_text()) or {}
    elif isinstance(src, dict):
        user = src
    _check(user, DEFAULT_CONFIG)
    if user.get("schema", CONFIG_SCHEMA) != CONFIG_SCHEMA:
        raise IOContractError(f"config schema {user.get('schema')!r} != {CONFIG_SCHEMA}")
    c = _merge(DEFAULT_CONFIG, user)
    for v in c["model_variants"]:
        if v not in VARIANT_BY_NAME:
            raise IOContractError(f"model variant {v!r} not in {list(VARIANT_BY_NAME)}")
    if "inertial_viscous" not in c["model_variants"]:
        raise IOContractError("the minimal model inertial_viscous (L0) is always fitted first")
    if c["joint_source"] not in ("accepted_authority", "diagnostic_working_authority"):
        raise IOContractError(f"joint_source {c['joint_source']!r}")
    if c["contact_policy"] != "inherit_g9_g10":
        raise IOContractError("contact_policy must inherit the G9/G10 classifier (no new free/noncontact labels)")
    if c["fit"]["input_interpolation"] != "forward_zoh":
        raise IOContractError("measured effort is applied by the source-backed forward ZOH only")
    if c["fit"]["robust_loss"] not in ("huber", "l2"):
        raise IOContractError("robust_loss: huber | l2")
    if c["fit"]["local_linearization"] != "moving_only":
        raise IOContractError("a transfer function exists only in a moving, fixed-sign regime")
    for k in ("response_regularizer_weight", "energy_penalty_weight"):
        if float(c["physx"][k]) != 0.0:
            raise IOContractError(f"physx.{k} > 0: the feature / energy term is derived from the SAME q, u samples; "
                                  "without a joint covariance model with L_q it double counts (diagnostic / "
                                  "initialization use only)")
    for k in _FIXED_TRUE:
        if c["physx"][k] is not True:
            raise IOContractError(f"physx.{k} cannot be disabled (passive baseline fitting)")
    for k in _VALIDATION_FIXED:
        if c["validation"][k] is not True:
            raise IOContractError(f"validation.{k} cannot be disabled")
    return c


# ======================================================================================================== no value
def assert_no_value_field(obj, path="$"):
    """Fail-closed handoff: a non-identified point estimate never appears under a key called ``value``."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "value":
                raise IOContractError(f"forbidden 'value' field at {path}")
            assert_no_value_field(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_value_field(v, f"{path}[{i}]")


def rename_value_fields(obj):
    """Copy of ``obj`` with inherited ``value`` keys renamed to ``estimate_not_for_sim`` (legacy G9 dicts)."""
    if isinstance(obj, dict):
        return {("estimate_not_for_sim" if k == "value" else k): rename_value_fields(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rename_value_fields(v) for v in obj]
    return obj
