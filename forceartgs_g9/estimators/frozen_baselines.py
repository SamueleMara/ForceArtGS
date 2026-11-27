"""Frozen ForceArtGS-Physics viscous estimators as BASELINES (never as general-model estimators).

``api.identify`` is called unchanged on a forceartgs_physics ``Interaction`` built from the G9 interaction: the
generalized effort Q(t) is re-expressed EXACTLY as a point force (prismatic: F = Q a at any point; revolute: F = Q
(a x r) / |r|^2 at p = o + r, r a fixed unit lever perpendicular to the axis), so the library's own signed
projection returns Q. The results are labelled ``viscous_1dof baseline``: they fit M q'' + c q' = Q only.
"""
from __future__ import annotations

import time

import numpy as np

from .. import physics_lib


def to_physics_interaction(it, axis, origin, frame="W"):
    physics_lib.load()            # on demand: only this opt-in comparison runs ForceArtGS-Physics
    from forceartgs_physics.schemas import ForceRecord, Interaction, JointKinematics, KinematicStatus, Trajectory
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    o = np.zeros(3) if origin is None else np.asarray(origin, float)
    t, Q = np.asarray(it.effort.t_s, float), np.asarray(it.effort.levels, float)
    if it.joint_type == "prismatic":
        F = Q[:, None] * a
        P = np.tile(o, (len(t), 1))
    else:
        r = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        r /= np.linalg.norm(r)
        F = Q[:, None] * np.cross(a, r)
        P = np.tile(o + r, (len(t), 1))
    joint = JointKinematics(it.joint_type, a, o, frame, KinematicStatus.IDENTIFIED, source=dict(g9=it.name))
    traj = Trajectory(it.t_obs, it.q_obs, it.q_std, source=dict(g9=it.name))
    fr = ForceRecord(t_s=t, force_N=F, point_m=P, frame=frame,
                     calibration=dict(g9="exact re-expression of the G9 generalized effort as a point force"))
    return Interaction(it.name, joint, traj, fr)


def run_baselines(interactions, axis, origin, methods, **config) -> dict:
    from forceartgs_physics import api
    out = {}
    for m in methods:
        rows = []
        for it in interactions:
            t0 = time.time()
            try:
                r = api.identify(to_physics_interaction(it, axis, origin), m, **config)
                p = r.parameters
                rows.append(dict(interaction=it.name, status=r.status, inertia=None if p is None else p.inertia,
                                 damping=None if p is None else p.damping,
                                 identifiable=(r.identifiability or {}).get("identifiable"),
                                 verdict=(r.identifiability or {}).get("verdict"), runtime_s=time.time() - t0))
            except Exception as e:                     # a frozen method failing is a baseline result, not a crash
                rows.append(dict(interaction=it.name, status="failed", error=f"{type(e).__name__}: {e}"))
        out[m] = rows
    return dict(model="viscous_1dof baseline (M q'' + c q' = Q; ForceSAPIEN model, frozen methods unchanged)",
                results=out, physics_library=physics_lib.provenance())
