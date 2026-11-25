"""G10-A behavioral full-demonstration replay: data contracts (estimator side, ForceArtGS only).

The behavioral task is NOT physical identification: choose simulator parameters whose PhysX rollout, driven by the
recorded canonical generalized effort from the demonstration's initial REST state, best reproduces the observed joint
trajectory. Four claims are kept separate and never implied by each other:

    BEHAVIORAL_FIT_TO_DEMONSTRATION      -- what this stage produces (training score on the one demo)
    PHYSICAL_PARAMETERS_IDENTIFIED       -- never claimed here
    INDEPENDENT_REAL_PREDICTION_VALIDATED-- needs another real action (NOT_EVALUATED otherwise)
    NATIVE_TARGET_PHYSX_PARITY           -- needs the native target engine (SAPIEN 3.0.3: surrogate friction)

Initial-state contract (user, 2026-11-25): the physical joint starts at REST, qdot(t0) = 0 exactly; the initial
POSITION is q_authority(t0) in the authority's own coordinate (never forced to 0). v0 is a constant 0.0, never a
prefix estimate, nuisance, free or regularized parameter.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

EPISODE_SCHEMA = "forceartgs.g10.behavioral_episode/v1"
CONFIG_SCHEMA = "forceartgs.g10.behavioral_fit/v1"
CANDIDATE_SCHEMA = "forceartgs.g10.behavioral_simulator_candidate/v1"
REGIME_SEMANTICS = ("behavioral_full_demo: the recorded canonical generalized effort is applied over the whole "
                    "demonstration; mode 'forced' means 'recorded input applied', NOT 'contact proven at every instant'")

# start-state bases (how v0 = 0 is justified)
START_FULL_RECORDING = "USER_DECLARED_FULL_DEMONSTRATION_INITIAL_REST"
START_REST_VERIFIED_CROP = "USER_DECLARED_REST_VERIFIED_AT_CROPPED_AUTHORITY_START"
# wrench-driven path (FORCEARTGS_G10_WRENCH_DRIVEN_BEHAVIORAL_PHYSX_CODEX_UPDATED, fact E4): the initial joint speed is
# an experimental fact, never tested or estimated
START_EXPERIMENT_KNOWN_ZERO = "EXPERIMENT_KNOWN_ZERO"
START_BASES = (START_FULL_RECORDING, START_REST_VERIFIED_CROP, START_EXPERIMENT_KNOWN_ZERO)

# contact-evidence labels (evidence for claims / hypotheses; NEVER a substitution of the recorded input)
CONTACT_LABELS = ("CONTACT_CONFIRMED", "MEASURED_NEAR_ZERO_CONTACT", "CONTACT_UNKNOWN", "RELEASE_CONFIRMED",
                  "NO_CONTACT_MEASURED", "UNMEASURED_INPUT")
HYPOTHESIS_KINDS = ("recorded", "time_offset", "noise_realization", "contact_interpretation")


class BehavioralContractError(ValueError):
    pass


def _a(x):
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True)
class FullDemoBehavioralEpisode:
    scene: str
    joint_type: str
    t0_s: float                        # physical initial frame time on the canonical clock (source_frame / 50)
    source_start_frame: int            # original source frame of t0 (NOT an array index)
    recording_start_frame: int         # first source frame of the robot recording
    t_q_s: np.ndarray                  # every valid authority observation time from t0
    source_q_frame: np.ndarray
    q_observed: np.ndarray             # authority coordinate, never re-zeroed
    q_sigma: np.ndarray
    q0_observed: float                 # q(t0), same coordinate
    v0_fixed: float                    # MUST be exactly 0.0
    effort_t_s: np.ndarray             # native force-row clock, rows from the last row <= t0 to the end
    effort_source_row: np.ndarray
    effort_object: np.ndarray          # canonical Q_object, sign unchanged
    effort_sigma: Optional[np.ndarray]
    contact_evidence: dict
    kinematics_evidence: dict
    calibration_evidence: dict
    start_basis: str
    full_demo_start_verified: bool
    start_is_full_demo: bool           # True only when the authority covers the recording start itself
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        for k in ("t_q_s", "source_q_frame", "q_observed", "q_sigma", "effort_t_s", "effort_source_row",
                  "effort_object"):
            object.__setattr__(self, k, _a(getattr(self, k)))
        if self.effort_sigma is not None:
            object.__setattr__(self, "effort_sigma", _a(self.effort_sigma))
        if self.joint_type not in ("revolute", "prismatic"):
            raise BehavioralContractError(f"joint_type {self.joint_type!r}")
        # the contract: exact zero, bit for bit (also rejects -0.0)
        v = self.v0_fixed
        if not (isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0.0 and math.copysign(1.0, v) > 0):
            raise BehavioralContractError(f"v0_fixed must be exactly +0.0 in full-demo mode, got {self.v0_fixed!r}")
        if not self.full_demo_start_verified:
            raise BehavioralContractError("full-demo episode requires a verified rest start (FULL_START_UNAVAILABLE)")
        if self.start_basis not in START_BASES:
            raise BehavioralContractError(f"start_basis {self.start_basis!r}")
        if self.start_is_full_demo != (self.start_basis != START_REST_VERIFIED_CROP):
            raise BehavioralContractError("start_is_full_demo inconsistent with start_basis")
        n = len(self.t_q_s)
        if n < 3 or len(self.q_observed) != n or len(self.q_sigma) != n or len(self.source_q_frame) != n:
            raise BehavioralContractError("q times / values / sigmas / frames must have equal length >= 3")
        if np.any(np.diff(self.t_q_s) <= 0) or np.any(self.q_sigma <= 0):
            raise BehavioralContractError("strictly increasing q times and positive q sigmas required")
        if abs(float(self.t_q_s[0]) - float(self.t0_s)) > 1e-12:
            raise BehavioralContractError("the first observation must be at t0 (no prefix crop)")
        if float(self.q0_observed) != float(self.q_observed[0]):
            raise BehavioralContractError("q0_observed must equal the authority q at t0 (never imposed)")
        m = len(self.effort_t_s)
        if m == 0 or len(self.effort_object) != m or len(self.effort_source_row) != m:
            raise BehavioralContractError("effort timeline / values / rows mismatch")
        if self.effort_sigma is not None and len(self.effort_sigma) != m:
            raise BehavioralContractError("effort sigma length mismatch")
        if np.any(np.diff(self.effort_t_s) <= 0):
            raise BehavioralContractError("effort times must be strictly increasing")
        if self.effort_t_s[0] > self.t0_s + 1e-12:
            raise BehavioralContractError("effort must be defined at t0 (no extrapolation)")
        if not all(np.isfinite(x).all() for x in (self.t_q_s, self.q_observed, self.q_sigma, self.effort_t_s,
                                                   self.effort_object)):
            raise BehavioralContractError("non-finite episode data")

    @property
    def duration_s(self):
        return float(self.t_q_s[-1] - self.t0_s)

    def to_physx_episode(self, name="full_demo", effort=None):
        """ONE G10 Episode over the whole demonstration: q0 = q_authority(t0), v0 = 0.0, the recorded effort (or an
        explicitly labelled input hypothesis ``effort`` on the SAME timeline) applied throughout. No reset anywhere."""
        from .objective import Episode
        lev = self.effort_object if effort is None else _a(effort)
        if lev.shape != self.effort_object.shape:
            raise BehavioralContractError("input hypothesis must live on the recorded effort timeline")
        return Episode(name=name, t_s=self.t_q_s, q=self.q_observed, sigma_q=self.q_sigma, q0=float(self.q0_observed),
                       v0=0.0, input_t_s=self.effort_t_s, input_effort=lev, mode="forced",
                       calibration_status=str(self.calibration_evidence.get("force_audit_status") or "UNVERIFIED"),
                       used_for_training=True, exploratory_assumptions_accepted=True, sigma_v0=None,
                       regime_semantics=REGIME_SEMANTICS)

    def describe(self) -> dict:
        return dict(schema=EPISODE_SCHEMA, scene=self.scene, joint_type=self.joint_type, t0_s=self.t0_s,
                    source_start_frame=int(self.source_start_frame),
                    recording_start_frame=int(self.recording_start_frame),
                    joint_authority_start_frame=int(self.source_q_frame[0]),
                    start_is_full_demo=bool(self.start_is_full_demo), start_basis=self.start_basis,
                    full_demo_start_verified=bool(self.full_demo_start_verified),
                    q0_observed=float(self.q0_observed), v0_fixed=self.v0_fixed,
                    v0_source=self.start_basis, n_q=int(len(self.t_q_s)),
                    t_q=[float(self.t_q_s[0]), float(self.t_q_s[-1])], duration_s=self.duration_s,
                    q_range=float(np.ptp(self.q_observed)), median_q_sigma=float(np.median(self.q_sigma)),
                    n_effort_rows=int(len(self.effort_t_s)),
                    effort_rows=[int(self.effort_source_row[0]), int(self.effort_source_row[-1])],
                    effort_t=[float(self.effort_t_s[0]), float(self.effort_t_s[-1])],
                    effort_abs_max=float(np.max(np.abs(self.effort_object))),
                    median_effort_sigma=None if self.effort_sigma is None else float(np.median(self.effort_sigma)),
                    regime_semantics=REGIME_SEMANTICS, kinematics=self.kinematics_evidence,
                    calibration=self.calibration_evidence, provenance=self.provenance)


@dataclass(frozen=True)
class InputHypothesis:
    """A declared, GROUNDED alternative of the recorded input on the same timeline (time offset, noise realization,
    contact interpretation). A sign flip is never a hypothesis. Not an independent demonstration."""
    name: str
    kind: str
    effort_object: np.ndarray
    basis: str

    def __post_init__(self):
        object.__setattr__(self, "effort_object", _a(self.effort_object))
        if self.kind not in HYPOTHESIS_KINDS:
            raise BehavioralContractError(f"hypothesis kind {self.kind!r} not in {HYPOTHESIS_KINDS}")
        if "sign" in self.name.lower() or "flip" in self.name.lower():
            raise BehavioralContractError("a sign reversal is never an input hypothesis")
        if not np.isfinite(self.effort_object).all():
            raise BehavioralContractError("non-finite input hypothesis")


# --------------------------------------------------------------------------------------------------- config
DEFAULT_CONFIG = {
    "schema": CONFIG_SCHEMA,
    "enabled": True,
    "inputs": {
        # the SAME G9 dynamics config + contact classifier options as the free/forced / G10 stages (the classifier
        # regimes are reported as EVIDENCE only; they never crop, reset or zero the behavioral replay)
        "free_forced_config": "configs/g9_real/free_forced.yaml",
    },
    "initial_state": {
        "mode": "full_demonstration_rest",          # | rest_verified_cropped_start (explicit opt-in)
        "require_complete_demo_start": True,
        "initial_q": "visual_authority",
        "initial_velocity_rad_or_m_per_s": 0.0,
        "optimize_q0": False,
        "optimize_v0": False,
        "rest_check_window_s": 0.4,                 # H0: v(t0) = 0 consistency GATE (never a v0 estimate)
        "rest_check_min_samples": 4,
        "rest_check_k_sigma": 3.0,
        "rest_check_input_onset_k_sigma": 3.0,      # the test only uses samples before the measured input onset
        "rest_check_input_onset_rows": 3,
    },
    "input": {
        "mode": "canonical_measured_generalized_effort",
        "include_unknown_contact_conditionally": True,
        "preserve_original_force_sign": True,
        "missing_effort_policy": "refuse_full_replay",
        "max_row_gap_factor": 1.5,
        "uncertainty_sensitivity": True,
        "time_offset_rows": [-1, 1],
        "noise_realizations": 2,
        "contact_interpretation_hypotheses": True,
        "near_zero_k_sigma": 2.0,
    },
    "model": {
        "mass_mode": "uniform_solid",
        "armature": 0.0,
        "friction_mode": "external_effort_surrogate",
        "physical_bounds_source": "existing_g10_and_geometry_prior",
        "rho_bounds": [0.1, 10.0],
        "lam_max": 20.0,
        "gam_max": 20.0,
        "effort_bound_factor": 3.0,
    },
    "simulation": {
        "engine": "sapien_cpu_diagnostic",
        "timestep": 0.001,
        "require_clean_passive_readback": True,
        "full_continuous_rollout": True,
        "python": None,
    },
    "loss": {
        "primary": "robust_position_output_error",
        "huber_delta": 2.0,
        "numeric_floor_q": 3e-4,
        "increment_weight": 0.0,
        "event_weight": 0.0,
        "optimize_initial_velocity": False,
        "optimize_initial_position": False,
        "adequacy_sigma": 3.0,
    },
    "fitting": {
        "seed": 0,
        "maxiter": 40,
        "popsize": 8,
        "polish": True,
        "tol": 1e-3,
        "initialize_from_io": True,
        "profile_near_optimal_family": True,
        "profile_parameters": ["rho_scale", "viscous", "dynamic_effort"],
        # local refinement after the global DE: bounded robust TRF on the same PhysX residuals (Huber delta, same
        # sigma floor). The finite-difference step must exceed the engine's float32 joint-state resolution: SciPy's
        # default (~1.5e-8 relative) gives a zero Jacobian (a no-op polish) -- measured, see the report.
        "local_refine": True,
        "local_diff_step": 1e-3,
        "local_max_nfev": 200,                    # per start, shared by its restarts
        "local_restarts": 3,                      # TRF restarts from its own end point (trust-region reset)
        "local_starts": 3,                        # global best + the best (local_starts - 1) scored baselines
        "profile_points": 7,
        "profile_inner_max_nfev": 60,             # local TRF with the profiled parameter fixed, from the best point
        "family_tolerance_sigma": 0.5,            # ENGINEERING tolerance on rmse/sigma, NOT a confidence interval
    },
    "alternatives": {
        "zero_gravity_load": True,                # separately labelled hypothesis, never the selected candidate
        "maxiter": 15,
        "popsize": 6,
        "local_max_nfev": 100,
    },
    "policy_probe": {
        "enabled": True,                          # T15: near-equal demo fits under a NEW (synthetic) force profile
        "duration_s": 2.0,
        "push_fraction": 0.5,
        "amplitude_factor": 1.0,                  # x median |Q| over confirmed-contact rows
    },
    "output": {
        "append_only": True,
        "write_behavioral_candidate": True,
        "allow_preview_without_unique_physical_id": True,
        "require_independent_episode_for_validated_transfer": True,
        "do_not_modify_certified_g9_handoff": True,
    },
}
QUICK = {"fitting": {"maxiter": 2, "popsize": 4, "polish": False, "local_max_nfev": 40, "local_starts": 2,
                     "profile_points": 3, "profile_inner_max_nfev": 12, "profile_parameters": ["rho_scale"]},
         "alternatives": {"maxiter": 1, "popsize": 4, "local_max_nfev": 20}, "input": {"noise_realizations": 1}}


def _merge_strict(base, over, path=""):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise BehavioralContractError(f"unknown config key {path + k!r}")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise BehavioralContractError(f"{path + k} must be a mapping")
            out[k] = _merge_strict(base[k], v, path + k + ".")
        else:
            out[k] = v
    return out


def load_config(src=None, quick=False) -> dict:
    """Strict: unknown keys are errors; every option below is wired (no decorative fields)."""
    import yaml
    doc = {} if src is None else (copy.deepcopy(src) if isinstance(src, dict) else
                                  (yaml.safe_load(Path(src).read_text()) or {}))
    cfg = _merge_strict(DEFAULT_CONFIG, doc)
    if quick:
        cfg = _merge_strict(cfg, QUICK)
    validate_config(cfg)
    cfg["_resolved"] = True
    return cfg


def validate_config(cfg):
    if cfg["schema"] != CONFIG_SCHEMA:
        raise BehavioralContractError(f"schema {cfg['schema']!r} != {CONFIG_SCHEMA}")
    ini, inp, lo, sim = cfg["initial_state"], cfg["input"], cfg["loss"], cfg["simulation"]
    if ini["mode"] not in ("full_demonstration_rest", "rest_verified_cropped_start"):
        raise BehavioralContractError(f"initial_state.mode {ini['mode']!r}")
    if ini["mode"] == "full_demonstration_rest" and not ini["require_complete_demo_start"]:
        raise BehavioralContractError("full_demonstration_rest requires require_complete_demo_start: true")
    if ini["mode"] == "rest_verified_cropped_start" and ini["require_complete_demo_start"]:
        raise BehavioralContractError("rest_verified_cropped_start is the explicit opt-in for a cropped authority "
                                      "start: set require_complete_demo_start: false")
    if ini["initial_q"] != "visual_authority":
        raise BehavioralContractError("initial_q must be visual_authority (q0 is never imposed)")
    v = ini["initial_velocity_rad_or_m_per_s"]
    if not (isinstance(v, (int, float)) and float(v) == 0.0):
        raise BehavioralContractError("full-demo mode: initial velocity is exactly 0 (user rest contract)")
    for k in ("optimize_q0", "optimize_v0"):
        if ini[k]:
            raise BehavioralContractError(f"initial_state.{k} must be false in full-demo mode")
    for k in ("optimize_initial_velocity", "optimize_initial_position"):
        if lo[k]:
            raise BehavioralContractError(f"loss.{k} must be false in full-demo mode")
    if lo["primary"] != "robust_position_output_error":
        raise BehavioralContractError("loss.primary must be robust_position_output_error")
    if inp["mode"] != "canonical_measured_generalized_effort":
        raise BehavioralContractError("input.mode must be canonical_measured_generalized_effort")
    if not inp["preserve_original_force_sign"]:
        raise BehavioralContractError("the canonical force sign is never changed")
    if inp["missing_effort_policy"] != "refuse_full_replay":
        raise BehavioralContractError("missing effort is refused (no hold / zero fill)")
    if not inp["include_unknown_contact_conditionally"]:
        raise BehavioralContractError("behavioral mode keeps UNKNOWN-contact rows with their recorded input "
                                      "(conditional); excluding them is the legacy segmented path")
    if any(int(s) == 0 for s in inp["time_offset_rows"]):
        raise BehavioralContractError("time_offset_rows must be nonzero row shifts")
    if sim["engine"] != "sapien_cpu_diagnostic":
        raise BehavioralContractError("native Isaac PhysX backend is not implemented (NOT_IMPLEMENTED_BACKEND)")
    if not sim["full_continuous_rollout"] or not sim["require_clean_passive_readback"]:
        raise BehavioralContractError("full continuous rollout with clean passive readback is required")
    if cfg["model"]["friction_mode"] not in ("external_effort_surrogate", "frictionless_physx"):
        raise BehavioralContractError("friction_mode external_effort_surrogate | frictionless_physx")
    if cfg["model"]["physical_bounds_source"] != "existing_g10_and_geometry_prior":
        raise BehavioralContractError("physical_bounds_source must be existing_g10_and_geometry_prior")
    if lo["event_weight"] != 0.0:
        raise BehavioralContractError("event_weight > 0 is not implemented (events are reported as diagnostics)")
    if lo["increment_weight"] < 0:
        raise BehavioralContractError("increment_weight >= 0")
    if not float(cfg["fitting"]["local_diff_step"]) >= 5e-4:
        raise BehavioralContractError("local_diff_step must exceed the float32 engine-state resolution (>= 5e-4; "
                                      "1e-4 measured to give a useless Jacobian)")
    bad = set(cfg["fitting"]["profile_parameters"]) - {"rho_scale", "viscous", "dynamic_effort"}
    if bad:
        raise BehavioralContractError(f"profile_parameters {sorted(bad)} (rho_scale | viscous | dynamic_effort)")
    out = cfg["output"]
    if not out["append_only"] or not out["do_not_modify_certified_g9_handoff"]:
        raise BehavioralContractError("append-only output that never modifies the certified G9 handoff is required")
    if not out["require_independent_episode_for_validated_transfer"]:
        raise BehavioralContractError("validated transfer always requires an independent episode")
    return cfg


# =============================================================================== wrench-driven path (schema v2)
WRENCH_CONFIG_SCHEMA = "forceartgs.g10.behavioral_wrench/v2"
WRENCH_CANDIDATE_SCHEMA = "forceartgs.g10.behavioral_wrench_candidate/v2"
DEFAULT_WRENCH_CONFIG = {
    "schema": WRENCH_CONFIG_SCHEMA,
    "enabled": True,
    "experimental_facts": {                       # GIVEN by the experiment owner; validated to be exactly these
        "known_sensor_to_global_transform": True,
        "wrench_already_tared_at_start": True,
        # contact is ON during the interaction -- delimited by the EXISTING force-peak + visual contact segments
        # (configs/contact_intervals/<scene>.yaml) -- and OFF (zero object-side effort) outside it
        "contact_on_during_interaction": True,
        "initial_joint_velocity": 0.0,              # at the interaction start (object at rest before contact)
    },
    "inputs": {
        "dynamics_config": "configs/g9_real/forcetwin_effective.yaml",   # base G9 config (gravity, mass mesh, authority)
    },
    "initial_state": {
        "q0_source": "first_visual_joint_observation",
        "require_full_demonstration_start": True,
        "optimize_q0": False,
        "optimize_v0": False,
        "v0_nuisance_bound_sd": 0.0,
    },
    "input": {
        "source": "native_full_six_axis_tared_wrench",
        "coordinate_transform": "existing_known_global_robot_transform",
        "projection": "full_wrench_joint_screw",
        "moment_reference": "from_contract",          # EE frame origin per robot_tool.json wrench_frame
        "apply_transform_once": True,
        "additional_tare": False,
        "contact_classifier": "disabled",             # no NEW classification: the existing segments are only read
        "interaction_segments": "existing_force_peak_visual_segments",
        "interaction_segments_required": True,
        "outside_interaction_effort": "zero_contact_off",
        "include_post_interaction_coast": True,       # free motion after release, zero effort, to the end of the visual q
        "use_every_valid_measured_sample": True,
        "input_interpolation": "forward_zoh",
        "missing_input": "report_input_coverage_error",
        "max_row_gap_factor": 1.5,
        "compare_legacy_tcp_force_only": True,
        "compare_alternative_moment_origin": True,    # TCP-origin reading: diagnostic only
        "noise_baseline_s": 0.5,
        "uncertainty_sensitivity": True,              # same theta under +-row clock offsets / noise realizations
        "time_offset_rows": [-1, 1],
        "noise_realizations": 2,
    },
    "physics": {
        "engine": "sapien_gpu_physx",                 # PhysX GPU (batched); sapien_cpu_diagnostic for parity checks
        "device": "cuda",
        # 'geometry': the G9 base config's gravity on the reference mass geometry; 'zero': NO gravity load in the fit
        # (simplified model, labelled -- e.g. inertia-only I qdd = Q with friction_mode frictionless_physx)
        "gravity": "geometry",
        # full model = inertia + viscous damping + Coulomb friction + gravity. 'off': Coulomb friction Td fixed at 0
        # (not searched); viscous damping c is still fitted (friction_mode frictionless_physx removes both)
        "coulomb_friction": "fitted",
        "friction_mode": "external_effort_surrogate",
        "timestep_s": 0.001,
        "fixed_armature": 0.0,
        "static_excess": 0.0,
        "zero_and_readback_hidden_passive_terms": True,
        "mass_mode": "uniform_solid",
        "python": None,
    },
    "loss": {
        "objective": "full_trajectory_physx_position_output_error",
        "huber_delta": 2.0,
        "numeric_floor_q": 3e-4,
        "fit_initial_velocity": False,
        "increment_weight": 0.0,
        "adequacy_sigma": 3.0,                        # BEHAVIORAL_FIT_OK: rmse / sqrt(sigma_q^2 + sigma_num^2) <= this
    },
    "fitting": {
        "bounds": "existing_g10_geometry_constrained",
        "rho_bounds": [0.1, 10.0],
        "lam_max": 20.0,
        "gam_max": 20.0,
        "effort_bound_factor": 3.0,
        "use_optional_g9_io_warmstart": True,
        "seed": 0,
        "maxiter": 40,
        # PhysX GPU is latency-bound (~1.5 s per batched call almost regardless of width; break-even vs CPU ~40
        # candidates, 5.6x at 300, 14x at 1000 -- measured): a wide population costs no extra wall time
        "popsize": 100,
        "tol": 1e-3,
        # 'polish' = the local refinement on the PhysX residuals after DE: GPU -> batched lockstep Levenberg-Marquardt
        # (all starts in one batch per step); CPU -> bounded robust TRF. (DE's own L-BFGS-B polish is a no-op on the
        # float32 engine state with SciPy's default finite-difference step -- measured)
        "polish": True,
        "local_diff_step": 1e-3,
        "local_max_nfev": 200,
        "local_restarts": 3,
        "local_starts": 3,
        "collect_near_optimal_candidates": True,
        "profile_parameters": ["rho_scale", "viscous", "dynamic_effort"],
        "profile_points": 7,
        "profile_inner_max_nfev": 60,
        "family_tolerance_sigma": 0.5,                # engineering tolerance on rmse/sigma, NOT a confidence interval
    },
    "diagnostics": {
        "zero_gravity_alternative": True,
        "alternative_maxiter": 15,
        "alternative_popsize": 6,
        "alternative_local_max_nfev": 100,
        "policy_probe": True,
        "probe_duration_s": 2.0,
        "probe_push_fraction": 0.5,
        "probe_amplitude_factor": 1.0,
    },
    "output": {
        "append_only": True,
        "save_best_numeric_candidate_even_if_inadequate": True,
        "require_heldout_for_demo_fit": False,
        "retain_certified_export_rules": True,
        "write_candidate_urdf": True,
    },
}
WRENCH_QUICK = {"fitting": {"maxiter": 3, "popsize": 30, "local_max_nfev": 40, "local_starts": 2, "profile_points": 3,
                            "profile_inner_max_nfev": 12, "profile_parameters": ["rho_scale"]},
                "diagnostics": {"alternative_maxiter": 1, "alternative_popsize": 4,
                                "alternative_local_max_nfev": 20},
                "input": {"noise_realizations": 1}}
BANNED_WRENCH_KEYS = ("recalibrate_sensor", "estimate_tare", "detect_contact", "infer_contact_point",
                      "contact_segments", "free_hypotheses", "prefix_velocity", "v0_prefix_samples")


def load_wrench_config(src=None, quick=False) -> dict:
    """Strict v2 loader: unknown keys (including the banned calibration / tare / contact options) are errors."""
    import yaml
    doc = {} if src is None else (copy.deepcopy(src) if isinstance(src, dict) else
                                  (yaml.safe_load(Path(src).read_text()) or {}))

    def banned(d, path=""):
        for k, v in (d or {}).items():
            if k in BANNED_WRENCH_KEYS:
                raise BehavioralContractError(f"{path + k}: banned in the wrench-driven path (the experiment supplies "
                                              "the transform, the tare and continuous contact)")
            if isinstance(v, dict):
                banned(v, path + k + ".")
    banned(doc)
    cfg = _merge_strict(DEFAULT_WRENCH_CONFIG, doc)
    if quick:
        cfg = _merge_strict(cfg, WRENCH_QUICK)
    validate_wrench_config(cfg)
    cfg["_resolved"] = True
    return cfg


def validate_wrench_config(cfg):
    if cfg["schema"] != WRENCH_CONFIG_SCHEMA:
        raise BehavioralContractError(f"schema {cfg['schema']!r} != {WRENCH_CONFIG_SCHEMA}")
    ef = cfg["experimental_facts"]
    if not (ef["known_sensor_to_global_transform"] is True and ef["wrench_already_tared_at_start"] is True and
            ef["contact_on_during_interaction"] is True and float(ef["initial_joint_velocity"]) == 0.0):
        raise BehavioralContractError("experimental_facts are given (known transform, tared, contact on during the "
                                      "interaction, initial joint velocity 0.0) and cannot be changed here")
    ini, inp, ph, lo, f = cfg["initial_state"], cfg["input"], cfg["physics"], cfg["loss"], cfg["fitting"]
    expect = {("initial_state", "q0_source"): "first_visual_joint_observation",
              ("initial_state", "require_full_demonstration_start"): True,
              ("initial_state", "optimize_q0"): False, ("initial_state", "optimize_v0"): False,
              ("input", "source"): "native_full_six_axis_tared_wrench",
              ("input", "coordinate_transform"): "existing_known_global_robot_transform",
              ("input", "projection"): "full_wrench_joint_screw", ("input", "moment_reference"): "from_contract",
              ("input", "apply_transform_once"): True, ("input", "additional_tare"): False,
              ("input", "contact_classifier"): "disabled",
              ("input", "interaction_segments"): "existing_force_peak_visual_segments",
              ("input", "interaction_segments_required"): True,
              ("input", "outside_interaction_effort"): "zero_contact_off",
              ("input", "use_every_valid_measured_sample"): True, ("input", "input_interpolation"): "forward_zoh",
              ("input", "missing_input"): "report_input_coverage_error",
              ("physics", "zero_and_readback_hidden_passive_terms"): True,
              ("loss", "objective"): "full_trajectory_physx_position_output_error",
              ("loss", "fit_initial_velocity"): False,
              ("fitting", "bounds"): "existing_g10_geometry_constrained",
              ("output", "append_only"): True, ("output", "retain_certified_export_rules"): True,
              ("output", "save_best_numeric_candidate_even_if_inadequate"): True,
              ("output", "require_heldout_for_demo_fit"): False}
    for (sec, k), v in expect.items():
        if cfg[sec][k] != v:
            raise BehavioralContractError(f"{sec}.{k} must be {v!r} in the wrench-driven path, got {cfg[sec][k]!r}")
    if float(ini["v0_nuisance_bound_sd"]) != 0.0:
        raise BehavioralContractError("initial_state.v0_nuisance_bound_sd must be 0 (v0 = 0 exactly, E4)")
    if ph["engine"] not in ("sapien_gpu_physx", "sapien_cpu_diagnostic"):
        raise BehavioralContractError("physics.engine sapien_gpu_physx | sapien_cpu_diagnostic (native Isaac: "
                                      "NOT_IMPLEMENTED_BACKEND)")
    if ph["engine"] == "sapien_gpu_physx" and not str(ph["device"]).startswith("cuda"):
        raise BehavioralContractError("sapien_gpu_physx needs a CUDA device")
    if ph["friction_mode"] not in ("external_effort_surrogate", "frictionless_physx"):
        raise BehavioralContractError("physics.friction_mode external_effort_surrogate | frictionless_physx")
    if ph["gravity"] not in ("geometry", "zero"):
        raise BehavioralContractError("physics.gravity geometry | zero")
    if ph["gravity"] == "zero" and cfg["diagnostics"]["zero_gravity_alternative"]:
        raise BehavioralContractError("physics.gravity zero: the zero-gravity alternative IS the main fit; set "
                                      "diagnostics.zero_gravity_alternative false")
    if ph["coulomb_friction"] not in ("fitted", "off"):
        raise BehavioralContractError(f"physics.coulomb_friction fitted | \"off\" (quote it: bare YAML off is a boolean), got {ph['coulomb_friction']!r}")
    if ph["friction_mode"] == "frictionless_physx" and set(f["profile_parameters"]) - {"rho_scale"}:
        raise BehavioralContractError("friction_mode frictionless_physx: viscous / dynamic_effort are fixed at 0, "
                                      "profile only rho_scale")
    if ph["coulomb_friction"] == "off" and "dynamic_effort" in f["profile_parameters"]:
        raise BehavioralContractError("physics.coulomb_friction off: dynamic_effort is fixed at 0, do not profile it")
    if float(ph["static_excess"]) != 0.0:
        raise BehavioralContractError("physics.static_excess must be 0: the SAPIEN surrogate has no stiction")
    if float(ph["fixed_armature"]) < 0 or not float(ph["timestep_s"]) > 0:
        raise BehavioralContractError("physics.fixed_armature >= 0 and timestep_s > 0")
    if float(lo["increment_weight"]) < 0:
        raise BehavioralContractError("loss.increment_weight >= 0")
    if any(int(s) == 0 for s in inp["time_offset_rows"]):
        raise BehavioralContractError("input.time_offset_rows must be nonzero row shifts")
    if not float(f["local_diff_step"]) >= 5e-4:
        raise BehavioralContractError("fitting.local_diff_step >= 5e-4 (float32 engine-state resolution)")
    bad = set(f["profile_parameters"]) - {"rho_scale", "viscous", "dynamic_effort"}
    if bad:
        raise BehavioralContractError(f"profile_parameters {sorted(bad)}")
    return cfg


def wrench_to_fit_config(cfg) -> dict:
    """Map the v2 config onto the shared fit core (``behavioral_fit``) WITHOUT any contact / rest / calibration option:
    the core reads only loss, fitting, model, alternatives and policy_probe."""
    f, d, ph, lo = cfg["fitting"], cfg["diagnostics"], cfg["physics"], cfg["loss"]
    return {
        "loss": dict(primary="robust_position_output_error", huber_delta=lo["huber_delta"],
                     numeric_floor_q=lo["numeric_floor_q"], increment_weight=lo["increment_weight"], event_weight=0.0,
                     optimize_initial_velocity=False, optimize_initial_position=False,
                     adequacy_sigma=lo["adequacy_sigma"]),
        "fitting": dict(seed=f["seed"], maxiter=f["maxiter"], popsize=f["popsize"], polish=False, tol=f["tol"],
                        initialize_from_io=f["use_optional_g9_io_warmstart"],
                        profile_near_optimal_family=f["collect_near_optimal_candidates"],
                        profile_parameters=f["profile_parameters"], local_refine=f["polish"],
                        local_diff_step=f["local_diff_step"], local_max_nfev=f["local_max_nfev"],
                        local_restarts=f["local_restarts"], local_starts=f["local_starts"],
                        profile_points=f["profile_points"], profile_inner_max_nfev=f["profile_inner_max_nfev"],
                        family_tolerance_sigma=f["family_tolerance_sigma"]),
        "model": dict(mass_mode=ph["mass_mode"], armature=ph["fixed_armature"], friction_mode=ph["friction_mode"],
                      gravity=ph["gravity"], coulomb_friction=ph["coulomb_friction"],
                      physical_bounds_source="existing_g10_and_geometry_prior", rho_bounds=f["rho_bounds"],
                      lam_max=f["lam_max"], gam_max=f["gam_max"], effort_bound_factor=f["effort_bound_factor"]),
        "alternatives": dict(zero_gravity_load=d["zero_gravity_alternative"], maxiter=d["alternative_maxiter"],
                             popsize=d["alternative_popsize"], local_max_nfev=d["alternative_local_max_nfev"]),
        "policy_probe": dict(enabled=d["policy_probe"], duration_s=d["probe_duration_s"],
                             push_fraction=d["probe_push_fraction"], amplitude_factor=d["probe_amplitude_factor"]),
        "input": dict(uncertainty_sensitivity=cfg["input"]["uncertainty_sensitivity"],
                      time_offset_rows=cfg["input"]["time_offset_rows"],
                      noise_realizations=cfg["input"]["noise_realizations"],
                      contact_interpretation_hypotheses=False),
    }
