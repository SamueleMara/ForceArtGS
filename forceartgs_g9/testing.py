"""Shared synthetic end-to-end fixture for the G9 gate tests (synthetic truth only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from .config import load_dynamics_config
from .core import G9Inputs, run_g9
from .dynamics.general_1dof import GeneralTerms
from .synthetic import box_mesh, episode, make_object, write_visual_handoff


def mechanical_spec(obj, out: Path, static_mesh, anchor_frame, *, with_static=True, with_limits=True,
                    with_collision=True):
    out.mkdir(parents=True, exist_ok=True)
    static_mesh.write_obj(out / "static_collision.obj", ["synthetic static base"])
    obj.mesh.write_obj(out / "moving_collision.obj", ["synthetic moving part at q = 0"])
    from .dynamics.mass_properties import unit_density_properties
    sp = unit_density_properties(static_mesh, "uniform_solid")
    rho = 500.0
    d = {"schema": "forceartgs.g9.mechanical_spec/v1",
         "registration": {"T_world_capture": None, "gauge": "capture_world", "gravity_alignment": "declared",
                          "provenance": "synthetic world: z up"}}
    if with_collision:
        d["collision"] = {"static": {"source": "static_collision.obj", "units": "m", "T_capture_mesh": np.eye(4).tolist(),
                                     "provenance": "measured"},
                          "moving": {"source": "moving_collision.obj", "units": "m", "T_capture_mesh": np.eye(4).tolist(),
                                     "provenance": "measured", "at_original_frame": int(anchor_frame)}}
    if with_static:
        d["static_link_inertial"] = {"mass_kg": rho * sp.m1, "com_capture_m": sp.c1.tolist(),
                                     "inertia_com_capture_kg_m2": (rho * sp.J1).tolist(), "provenance": "measured"}
    if with_limits:
        lim = (-1.6, 1.6) if obj.joint_type == "revolute" else (-0.6, 0.9)
        d["joint_limits"] = {"lower": lim[0], "upper": lim[1], "velocity_limit": 10.0, "effort_limit": 50.0,
                             "provenance": "measured"}
    (out / "mechanical_spec.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    from .export.dynamics_v2 import load_mechanical_spec
    return load_mechanical_spec(out / "mechanical_spec.yaml")


def static_base_for(obj):
    if obj.joint_type == "revolute":
        return box_mesh([-0.3, -0.25, 0.5], [0.3, -0.05, 0.6])
    return box_mesh([-0.3, -0.6, 0.2], [0.3, -0.32, 0.5])


def synthetic_inputs(kind, out_dir, *, terms=None, n_train=2, n_heldout=2, seed=0, mech=True, force_status="VERIFIED",
                     hinge="horizontal", **obj_kw):
    out = Path(out_dir)
    obj = make_object(kind, terms=terms, hinge=hinge, g=(0, 0, -9.81), **obj_kw)
    train = [episode(obj, variant=v, seed=seed + v, name=f"train{v}")[0] for v in range(n_train)]
    held = [episode(obj, variant=n_train + v, seed=seed + 100 + v, declared_initial=True, name=f"heldout{v}")
            for v in range(n_heldout)]
    held_its = [h[0] for h in held]
    q_vis = held[0][1]["q"] if held else None
    base = static_base_for(obj)
    tr0, truth0 = episode(obj, variant=0, seed=seed)
    hand, anchor = write_visual_handoff(obj, out / "visual_handoff", static_mesh=base, q_traj=truth0["q"])
    ms = mechanical_spec(obj, out / "mech", base, anchor) if mech else None
    inp = G9Inputs(scene=f"synthetic_{kind}", joint_type=kind, axis=obj.axis, origin=obj.origin if kind == "revolute"
                   else None, kinematic_status="IDENTIFIED", kinematics_export_ready=True,
                   kinematics_provenance=dict(synthetic=True), train=train, heldout=held_its,
                   force_audit=dict(status=force_status, gates={}, failing=[] if force_status == "VERIFIED" else
                                    ["sensor_rotation"]),
                   gravity=obj.gravity, mass_mesh=obj.mesh, mass_mesh_provenance=dict(source="synthetic truth mesh"),
                   anchor_frame=anchor, visual_handoff=str(hand), mechanical_spec=ms,
                   input_hashes=dict(synthetic=kind, visual_handoff_sha256=__import__("hashlib").sha256(
                       Path(hand).read_bytes()).hexdigest()), synthetic=True,
                   moving_gaussian_xyz=_ply_xyz(hand.parent / "moving_canonical.ply"))
    return obj, inp, hand


def _ply_xyz(p):
    from plyfile import PlyData
    v = PlyData.read(str(p))["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(float)


def synthetic_cfg(terms: GeneralTerms | None = None, **over):
    t = (terms or GeneralTerms()).to_dict()
    t.pop("quasistatic")
    d = {"model": {"terms": t, "parameter_source": "geometry_coupled", "gravity": {"vector_world_mps2": [0, 0, -9.81]}},
         "mass_model": {"mode": "uniform_solid", "reflected_actuator_inertia": "none_passive_mechanism"},
         "fitting": {"compare_frozen_viscous_baselines": [], "n_starts": 2},
         # synthetic clock and force scale are exact: declared sweeps are the small residual uncertainty
         "identifiability": {"force_lag_sweep_s": 0.002, "force_scale_sweep_rel": 0.01}}
    for k, v in over.items():
        d.setdefault(k, {}).update(v)
    return load_dynamics_config(d)
