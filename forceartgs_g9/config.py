"""G9 dynamics configuration, schema ``forceartgs.dynamics_config/v1`` (G9 spec 6.3). Unknown keys are errors.

Defaults are the FULL scientific model (inertia, viscous, Coulomb, static, gravity ON; bias OFF) and fail-closed
policies. No numerical physical value is defaulted: gravity vector, mass-model mesh, mechanical limits and static
link inertials come from explicit evidence files or stay null (then the matching readiness is BLOCKED_*).
"""
from __future__ import annotations

import copy
from pathlib import Path

from .dynamics.general_1dof import GeneralTerms, ModelConfigError

SCHEMA = "forceartgs.dynamics_config/v1"

DEFAULTS = {
    "schema": SCHEMA,
    "pipeline": {"enabled": True, "only_after_final_articulated_reconstruction": True,
                 "visual_handoff_input": "auto_verified", "allow_diagnostic_kinematics": False,
                 "on_missing_calibration": "abstain", "on_missing_geometry": "joint_space_diagnostic",
                 # certified: every force gate must be VERIFIED (production). exploratory: the canonical robot record
                 # is used with its DECLARED / ESTIMATED assumptions listed; estimates are conditional, never exported
                 "force_policy": "certified"},
    "model": {"kind": "general_rigid_1dof",
              "terms": {"inertia": True, "viscous": True, "coulomb": True, "static": True, "gravity": True,
                        "bias": False},
              "parameter_source": "geometry_coupled", "automatic_term_selection": False, "quasistatic": False,
              "static_friction": {"solver": "karnopp_hybrid", "velocity_band_magnitude": None},
              "gravity": {"source": "verified_world_gravity", "vector_world_mps2": None, "calibration_file": None},
              "integrator_max_step_s": 2e-3},
    "mass_model": {"mode": "uniform_shell", "mesh": None, "mesh_units": "m", "geometry_frame": "W",
                   "shell_thickness_m": None, "source_provenance": None,
                   "allow_collision_proxy_as_mass_geometry": False, "mesh_is_collision_proxy": False,
                   "reflected_actuator_inertia": None, "user_link_inertial": None},
    "fitting": {"methods": ["general_integral_ls", "general_rollout_fit"],
                "compare_frozen_viscous_baselines": ["IntegralLS", "Backprop", "AKF", "PINN", "MLP"],
                "use_multiple_interactions": True, "robust_loss": "huber", "seed": 0, "n_starts": 4,
                "measured_force_uncertainty_required": True, "holdout_mode": "independent_episodes",
                "auto_lag_fit": False, "model_variants": None, "load_model_comparison": False},
    "contact": {"inherit_upstream_slip_mode": True, "require_contact_provenance": True, "force_on_object_only": True,
                "use_pure_sensor_torque": False},
    "identifiability": {"propagate_joint_geometry_uncertainty": True, "propagate_calibration_uncertainty": True,
                        "require_excitations": True, "on_unidentifiable_enabled_parameter": "block_identified_export",
                        "force_lag_sweep_s": 0.02, "force_scale_sweep_rel": 0.05, "force_bias_sweep": None,
                        "contact_point_sweep_m": None, "use_second_window_tare_sweep": True},
    "export": {"write_joint_space_result": True, "reconstruct_link_inertia_when_supported": True,
               "production_requires_heldout_real_prediction": True, "allow_assumption_preview": False,
               "canonical_builder": "parttwin1d", "prefer_imported_single_exporter": True, "write_urdf": True,
               "write_usd": True, "write_mjcf": True, "mechanical_spec": None},
    "episodes": None,
}

CHOICES = {
    ("pipeline", "on_missing_calibration"): ("abstain", "diagnostic_only"),
    ("pipeline", "on_missing_geometry"): ("joint_space_diagnostic", "abstain"),
    ("pipeline", "force_policy"): ("certified", "exploratory"),
    ("pipeline", "visual_handoff_input"): ("auto_verified",),
    ("model", "kind"): ("general_rigid_1dof",),
    ("model", "parameter_source"): ("geometry_coupled", "joint_space_only", "joint_space_gravity_coupled",
                                    "user_link_inertial"),
    ("mass_model", "mode"): ("uniform_solid", "uniform_shell", "segmented_material", "user_link_inertial"),
    ("fitting", "robust_loss"): ("huber", "linear"),
    ("fitting", "holdout_mode"): ("independent_episodes", "none"),
    ("identifiability", "on_unidentifiable_enabled_parameter"): ("block_identified_export",),
    ("export", "canonical_builder"): ("parttwin1d",),
}


def _merge(base, over, where):
    if not isinstance(over, dict):
        raise ModelConfigError(f"{where}: expected a mapping")
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in base:
            raise ModelConfigError(f"{where}.{k}: unknown key (allowed {sorted(base)})")
        if isinstance(base[k], dict) and base[k] and k != "user_link_inertial":
            out[k] = _merge(base[k], v or {}, f"{where}.{k}")
        else:
            out[k] = v
    return out


def load_dynamics_config(src=None) -> dict:
    """``src``: None (defaults), a dict, or a YAML path."""
    import yaml
    if src is None:
        d = {}
        base_dir = None
    elif isinstance(src, dict):
        d, base_dir = src, None
    else:
        p = Path(src)
        d = yaml.safe_load(p.read_text()) or {}
        base_dir = p.resolve().parent
    if d.get("schema", SCHEMA) != SCHEMA:
        raise ModelConfigError(f"dynamics config schema {d.get('schema')!r} != {SCHEMA}")
    cfg = _merge(DEFAULTS, d, "dynamics")
    for (a, b), allowed in CHOICES.items():
        if cfg[a][b] not in allowed:
            raise ModelConfigError(f"{a}.{b} = {cfg[a][b]!r} not in {list(allowed)}")
    if cfg["model"]["automatic_term_selection"]:
        raise ModelConfigError("model.automatic_term_selection: never silently disable requested effects; use "
                               "fitting.model_variants (opt-in, scored on withheld episodes)")
    t = dict(cfg["model"]["terms"])
    GeneralTerms(**t, quasistatic=bool(cfg["model"]["quasistatic"]))          # validates the combination
    if cfg["fitting"]["auto_lag_fit"]:
        raise ModelConfigError("fitting.auto_lag_fit: lag is never tuned on the evaluation data (declare it)")
    g = cfg["model"]["gravity"]
    if g["vector_world_mps2"] is not None and (not isinstance(g["vector_world_mps2"], list) or
                                               len(g["vector_world_mps2"]) != 3):
        raise ModelConfigError("model.gravity.vector_world_mps2 must be [gx, gy, gz] or null")
    cfg["_base_dir"] = None if base_dir is None else str(base_dir)
    return cfg


def resolve_path(cfg, p):
    if p is None:
        return None
    p = Path(p).expanduser()
    if not p.is_absolute() and cfg.get("_base_dir"):
        p = Path(cfg["_base_dir"]) / p
    return p


def terms_of(cfg) -> GeneralTerms:
    return GeneralTerms(**cfg["model"]["terms"], quasistatic=bool(cfg["model"]["quasistatic"]))
