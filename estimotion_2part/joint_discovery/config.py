"""Validated configuration for the ``joint_discovery`` section.

``mode: legacy`` (the default) means the new estimator is not used anywhere and the existing
FreeArtGS Phase 1 is unchanged. Unknown keys are an error, never silently reinterpreted.
Every numeric noise value is logged in the run manifest together with its provenance.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

MODES = ("legacy", "calibrated_joint_aware")
HYPOTHESES = ("prismatic", "revolute")
SELECTIONS = ("heldout_predictive", "observable_physical_v2")


@dataclass
class NoiseConfig:
    """Measurement model.  ``calibrate_from_static_core`` re-estimates the scale factors on the
    initial candidate-independent static core (never per candidate)."""
    sigma_px: float = 1.0                 # 2D track std [px]
    sigma_depth_m0: float = 0.002         # depth std = m0 + m2 * z^2  [m]
    sigma_depth_m2: float = 0.004         # [1/m]
    edge_depth_inflation: float = 4.0     # multiply depth std at flagged depth edges
    incidence_model: bool = True          # depth std / max(cos(incidence), floor)
    incidence_cos_floor: float = 0.15
    student_t_dof: float = 4.0            # robust per-observation likelihood
    outlier_px_range: float = 640.0       # broad uniform outlier support per image axis [px]
    outlier_depth_range_m: float = 2.0    # broad uniform outlier support in depth [m]
    calibrate_from_static_core: bool = True
    # Correlated depth errors (Stage B2). None = legacy iid per-observation depth noise. When set,
    # depth std is inflated by sqrt((1+rho)/(1-rho)), the AR(1) effective-sample-size correction for
    # temporally correlated residuals along a track. ``calibrate_depth_ar1`` estimates rho on the
    # candidate-independent static core of the TRAINING observations (never on held-out data).
    depth_ar1_rho: float | None = None
    calibrate_depth_ar1: bool = False
    provenance: str = "default prior (RGB-D datasheet-order values); see manifest for calibrated values"


@dataclass
class FitConfig:
    max_tracks_per_camera: int = 1500
    min_track_observations: int = 3
    em_iterations: int = 4
    least_squares_max_nfev: int = 200    # per EM round; GPU solver: every round converges (fridge: <= 133 used, 500 identical)
    robust_loss: str = "soft_l1"
    robust_f_scale: float = 3.0           # in whitened units
    temporal_sigma_qacc: float = 50.0     # rad/s^2 or m/s^2 scale for the weak acceleration prior
    enable_temporal_prior: bool = True
    n_seeds: int = 5
    min_moving_tracks: int = 8
    moving_chi2_threshold: float = 16.0   # per-track displacement chi2 to propose motion support
    ransac_iterations: int = 300
    ransac_threshold_sigma: float = 3.0
    prior_moving: float = 0.3
    prior_outlier: float = 0.05
    responsibility_floor: float = 0.05    # tracks below this moving posterior are excluded from M-step
    label_confidence: float = 0.9         # posterior needed to label a track static/moving
    seed: int = 0


@dataclass
class ComparisonConfig:
    heldout_time_fraction: float = 0.20
    selection: str = "heldout_predictive"
    allow_ambiguous: bool = True
    z_threshold: float = 2.0              # predictive difference / its standard error
    min_motion_sigma: float = 3.0         # joint span must exceed this many sigma to leave static
    remote_hinge_ratio: float = 5.0       # hinge radius / moving-support extent above which a revolute
                                          # fit is in its translation limit (scale-free degeneracy gate)
    # Stage C (opt-in; legacy defaults keep the original behaviour bit-for-bit)
    uncertainty_method: str = "track_paired"   # 'track_paired' (naive, legacy) | 'cluster_robust'
    group_voxel_m: float = 0.05                # surface-group voxel for cluster-robust inference
    remote_hinge_policy: str = "ratio"         # 'ratio' (legacy) | 'observability'
    min_translation_distinguishability_sigma: float = 3.0  # observability: R vs best translation at large |q|,
                                                           # p90 image difference in units of sigma_px
    hinge_support_margin: float = 1.0          # observability: hinge line within margin*extent of support


@dataclass
class ContactConfig:
    surface_max_distance_m: float = 0.01
    surface_min_points: int = 12
    surface_max_rms_m: float = 0.004
    surface_patch_radius_m: float = 0.03
    sigma_contact_m: float = 0.005
    sigma_normal_velocity_mps: float = 0.01
    min_tcp_speed_mps: float = 0.002
    min_force_n: float = 1.0
    mu_max: float = 1.5
    sync_offset_s: float = 0.0


@dataclass
class DepthEvidenceSection:
    """Stage B1 candidate-independent depth evidence (``depth_evidence.py``); 'raw' = legacy."""
    mode: str = "raw"                      # 'raw' | 'evidence'
    front_gap_m: float = 0.03
    front_window_px: int = 2
    jump_m: float = 0.03
    rigidity_tol_m: float = 0.01
    rigidity_sigma_k: float = 3.0
    neighbour_radius_m: float = 0.04
    max_neighbours: int = 10
    min_neighbours: int = 3
    min_incidence_cos: float = 0.0
    tool_margin_m: float = 0.05
    unverified_policy: str = "drop"


@dataclass
class SelectionV2Config:
    """``comparison.selection: observable_physical_v2`` (spec 10.1, flattened to the local typed config).

    Every numeric threshold below is a declared default whose calibration provenance is recorded in
    ``threshold_provenance`` and in every run manifest; none encodes a scene or object class."""
    heldout_fraction: float = 0.35          # fraction of time blocks withheld (non-adjacent interior blocks)
    block_acf_threshold: float = 0.2        # block length = first lag with pooled static residual ACF below this
    min_block_length: int = 2
    min_independent_blocks: int = 4         # held-out block preference needs this many blocks, else INSUFFICIENT
    render_min_independent_blocks: int = 6  # render evidence claim (spec 8.3)
    render_repeated_seeds: int = 2          # render veto needs >= this many paired training seeds (spec 8.2)
    null_patches: int = 40                  # static patches per interval for the rotation null
    axis_min_sigma: float = 3.0             # an interval constrains the shared axis only above this |w|/sd
    max_support_noise_ratio: float = 2.0    # moving/static robust residual ratio above which support is unreliable
    prismatic_margin: float = 3.0           # implied hinge rotation must exceed margin * max(upper bound, MDR)
    interface_gap_m: float = 0.03
    interface_min_cameras: int = 2
    sigma_calib_m: float = 0.01
    separation_k_sigma: float = 2.0
    max_axis_disagreement_deg: float = 15.0
    max_camera_axis_disagreement_deg: float = 25.0
    forbid_bbox_hard_gate: bool = True
    never_certify_type_from_occam_alone: bool = True
    threshold_provenance: str = ("declared defaults, checked on the synthetic Gate A-C controls of "
                                 "tests/test_joint_physical_selection_v2.py; not fitted to any real scene")


SLIP_MODES = ("sliding", "stick_soft", "stick_hard")
MECHANICAL_MODELS = ("none", "min_effort_tangent")


@dataclass
class ContactModeConfig:
    """Unified TCP contact: ``sliding`` (parent model) with ``stick_soft`` / ``stick_hard`` specialisations.

    Uncertainties are calibration placeholders: ``None`` means not calibrated. A diagnostic fallback is then used
    and every result is flagged UNVERIFIED_PRIOR_SCALE (never physics-export ready)."""
    enabled: bool = True
    default_mode: str = "sliding"                 # sliding | stick_soft | stick_hard   (rigid_grasp: not implemented)
    require_operator_mode_confirmation: bool = True
    sigma_tcp_position_m: float | None = None
    sigma_contact_surface_m: float | None = None
    sigma_stick_soft_m: float | None = None
    fallback_sigma_tcp_position_m: float = 0.015  # user-stated acceptable TCP error 1-2 cm (unverified scale)
    fallback_sigma_contact_surface_m: float = 0.01
    fallback_sigma_stick_soft_m: float = 0.02
    intervals_file: str | None = None
    min_segment_samples: int = 3
    contradiction_chi2_per_dof: float = 9.0       # mean whitened^2 above this on a confirmed segment -> contradicted
    surface_patch_radius_m: float = 0.03
    surface_min_points: int = 8


@dataclass
class TcpSupportConfig:
    enabled: bool = True
    tool_geometry_source: str = "metadata/robot_tool.json"
    seed_contact_object_surface: bool = True
    exclude_occluding_robot_geometry: bool = True
    tool_radius_m: float = 0.02                   # crude capsule TCP->EE (no measured tool mesh): APPROXIMATE
    tool_ambiguity_margin_m: float = 0.015
    tool_extend_m: float = 0.05
    seed_radius_m: float = 0.04
    proposal_k_sigma: float = 5.0                 # unconfirmed-contact proposals: relaxed motion test (vs 7 sigma)


@dataclass
class ForceModeConfig:
    enabled: bool = True
    require_verified_wrench: bool = True
    use_signed_alignment_diagnostic: bool = True
    mode_aware_friction: bool = True
    mechanical_model: str = "none"                # none | min_effort_tangent (explicitly assumed actuation model)
    min_force_n: float = 1.0
    eps_slip_mps: float = 0.003
    mu_max: float = 1.5
    mu_static_max: float = 1.0
    min_direction_separation_deg: float = 10.0
    sigma_direction_deg: float = 15.0
    n_null: int = 200


@dataclass
class FrontendValidationConfig:
    use_frozen_temporal_folds: bool = True
    compare_on_common_observations: bool = True
    forbid_test_time_anchor_refit: bool = True
    min_relative_improvement: float = 0.002       # held-out NLL must improve by this fraction to accept a round
    max_support_loss_fraction: float = 0.2        # a round may not drop more than this fraction of moving support


@dataclass
class MultimodalFrontendConfig:
    enabled: bool = False
    max_iterations: int = 4
    track_refinement: bool = True
    ncc_min: float = 0.85
    max_recover_per_round: int = 4000
    dense_masks: bool = True
    contact: ContactModeConfig = field(default_factory=ContactModeConfig)
    tcp_support: TcpSupportConfig = field(default_factory=TcpSupportConfig)
    force: ForceModeConfig = field(default_factory=ForceModeConfig)
    validation: FrontendValidationConfig = field(default_factory=FrontendValidationConfig)


@dataclass
class JointDiscoveryConfig:
    mode: str = "legacy"
    cameras: list = field(default_factory=lambda: ["cam_001", "cam_002"])
    fixed_calibration: bool = True
    hypotheses: list = field(default_factory=lambda: list(HYPOTHESES))
    keep_static_and_unknown: bool = True
    allow_partial_tracks: bool = True
    enable_visual_2d: bool = True
    enable_depth: bool = True
    enable_appearance: bool = False
    enable_coarse_renderer: bool = False
    enable_tcp_contact: bool = False
    enable_force_selection: bool = False
    enable_contact_surface_fit: bool = False
    require_calibrated_wrench: bool = True
    no_oracle_labels: bool = True
    output_subdir: str = "joint_discovery"
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    depth_evidence: DepthEvidenceSection = field(default_factory=DepthEvidenceSection)
    enable_physical_selection: bool = False   # Stage E: contact/force terms in candidate scoring (gated)
    selection_v2: SelectionV2Config = field(default_factory=SelectionV2Config)
    multimodal_frontend: MultimodalFrontendConfig = field(default_factory=MultimodalFrontendConfig)
    output_schema_version: int = 1             # 2 when comparison.selection == observable_physical_v2

    # ----------------------------------------------------------------- validation
    def validate(self) -> "JointDiscoveryConfig":
        if self.mode not in MODES:
            raise ValueError(f"joint_discovery.mode must be one of {MODES}, got {self.mode!r}")
        if not self.fixed_calibration:
            raise ValueError("joint_discovery requires fixed_calibration: true; physical camera poses are never optimized")
        bad = [h for h in self.hypotheses if h not in HYPOTHESES]
        if bad or not self.hypotheses:
            raise ValueError(f"hypotheses must be a non-empty subset of {HYPOTHESES}, got {self.hypotheses}")
        if not self.cameras:
            raise ValueError("at least one camera is required")
        if not self.no_oracle_labels:
            raise ValueError("no_oracle_labels: false is not supported in production inference; "
                             "oracle inputs are only allowed in clearly named diagnostics")
        if not (0.0 < self.comparison.heldout_time_fraction < 0.5):
            raise ValueError("comparison.heldout_time_fraction must be in (0, 0.5)")
        if self.comparison.selection not in SELECTIONS:
            raise ValueError(f"comparison.selection must be one of {SELECTIONS}")
        if self.comparison.selection == "observable_physical_v2":
            if not self.selection_v2.forbid_bbox_hard_gate or not self.selection_v2.never_certify_type_from_occam_alone:
                raise ValueError("selection_v2 safety switches (forbid_bbox_hard_gate, "
                                 "never_certify_type_from_occam_alone) cannot be disabled")
            if not (0.1 <= self.selection_v2.heldout_fraction <= 0.5):
                raise ValueError("selection_v2.heldout_fraction must be in [0.1, 0.5]")
            self.output_schema_version = 2
        mf = self.multimodal_frontend
        if mf.contact.default_mode == "rigid_grasp":
            raise ValueError("contact.default_mode rigid_grasp: NOT_IMPLEMENTED (needs an independently verified 6-DoF "
                             "grasp transform and an SE(3) solver); it is never reinterpreted as stick_hard")
        if mf.contact.default_mode not in SLIP_MODES:
            raise ValueError(f"multimodal_frontend.contact.default_mode must be one of {SLIP_MODES}")
        if mf.force.mechanical_model not in MECHANICAL_MODELS:
            raise ValueError(f"multimodal_frontend.force.mechanical_model must be one of {MECHANICAL_MODELS}")
        if not mf.force.require_verified_wrench:
            raise ValueError("multimodal_frontend.force.require_verified_wrench cannot be disabled")
        v = mf.validation
        if not (v.use_frozen_temporal_folds and v.compare_on_common_observations and v.forbid_test_time_anchor_refit):
            raise ValueError("multimodal_frontend.validation leakage guards cannot be disabled")
        for name in ("sigma_tcp_position_m", "sigma_contact_surface_m", "sigma_stick_soft_m"):
            val = getattr(mf.contact, name)
            if val is not None and not (val > 0):
                raise ValueError(f"multimodal_frontend.contact.{name} must be > 0 (a structural hard stick is not "
                                 "sigma = 0)")
        if mf.enabled and self.comparison.selection != "observable_physical_v2":
            raise ValueError("multimodal_frontend requires comparison.selection: observable_physical_v2")
        if mf.max_iterations < 1:
            raise ValueError("multimodal_frontend.max_iterations must be >= 1")
        if self.enable_appearance or self.enable_coarse_renderer:
            # Not silently ignored: these paths are diagnostics only in this version.
            pass
        if self.enable_force_selection and not self.enable_tcp_contact:
            raise ValueError("enable_force_selection requires enable_tcp_contact (force needs the contact point)")
        if self.comparison.uncertainty_method not in ("track_paired", "cluster_robust"):
            raise ValueError("comparison.uncertainty_method must be 'track_paired' or 'cluster_robust'")
        if self.comparison.remote_hinge_policy not in ("ratio", "observability"):
            raise ValueError("comparison.remote_hinge_policy must be 'ratio' or 'observability'")
        if self.depth_evidence.mode not in ("raw", "evidence"):
            raise ValueError("depth_evidence.mode must be 'raw' or 'evidence'")
        if self.depth_evidence.unverified_policy not in ("drop", "keep"):
            raise ValueError("depth_evidence.unverified_policy must be 'drop' or 'keep'")
        r = self.noise.depth_ar1_rho
        if r is not None and not (0.0 <= r < 1.0):
            raise ValueError("noise.depth_ar1_rho must be in [0, 1)")
        if self.enable_physical_selection and not self.enable_tcp_contact:
            raise ValueError("enable_physical_selection requires enable_tcp_contact")
        if self.fit.robust_loss not in ("linear", "soft_l1", "huber", "cauchy", "arctan"):
            raise ValueError(f"unsupported robust loss {self.fit.robust_loss}")
        return self

    @property
    def is_new_mode(self) -> bool:
        return self.mode == "calibrated_joint_aware"

    @property
    def uses_cam_000(self) -> bool:
        return "cam_000" in self.cameras

    def to_dict(self) -> dict:
        d = asdict(self)
        if not self.multimodal_frontend.enabled:                # keep legacy / v2 config hashes bit-identical
            d.pop("multimodal_frontend", None)
        if self.comparison.selection == "heldout_predictive":   # legacy: keep config hashes bit-identical
            d.pop("selection_v2", None)
            d.pop("output_schema_version", None)
        return d

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def _from_dict(cls, data: dict, path: str):
    if not isinstance(data, dict):
        raise TypeError(f"{path} must be a mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise KeyError(f"unknown key(s) in {path}: {unknown}")
    kwargs = {}
    for name, value in data.items():
        f = known[name]
        default = f.default_factory() if f.default_factory is not MISSING else f.default  # type: ignore[misc]
        if is_dataclass(default):
            kwargs[name] = _from_dict(type(default), value, f"{path}.{name}")
        else:
            kwargs[name] = copy.deepcopy(value)
    return cls(**kwargs)


def load_joint_discovery_config(source: Any = None) -> JointDiscoveryConfig:
    """Accepts None, a dict (either the section or a full pipeline config), or a YAML/JSON path."""
    if source is None:
        return JointDiscoveryConfig().validate()
    if isinstance(source, (str, Path)):
        text = Path(source).read_text()
        if str(source).endswith(".json"):
            data = json.loads(text)
        else:
            import yaml
            data = yaml.safe_load(text) or {}
    else:
        data = dict(source)
    if "joint_discovery" in data:
        data = data["joint_discovery"] or {}
    return _from_dict(JointDiscoveryConfig, data, "joint_discovery").validate()


def resolve_joint_discovery(pipeline_config: dict | None) -> JointDiscoveryConfig:
    """Pipeline hook: absent section -> legacy mode (existing behaviour, bit-for-bit)."""
    section = (pipeline_config or {}).get("joint_discovery")
    return load_joint_discovery_config({} if section is None else section)
