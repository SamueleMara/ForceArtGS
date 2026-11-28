"""Synthetic multi-push benchmark (spec section 10): ONE unchanged object per joint type, KNOWN M and c, several distinct
pushes (amplitude, sign, timing, two-stage, initial q0). Each push is trained with the EXISTING estimator
(rest_to_rest_comparison.run_synthetic: analytics, PINNs, G10 PhysX fits on actual PhysX); the cross-demo loop then
replays every other push's input with the frozen parameters and scores against the noisy observation (metrics) and
the exact noise-free truth (TRUE_GT). A misspecified variant adds a constant unmodelled effort to the RECORD (the
minimal 2-term model cannot represent it): its transfer must degrade, not be "recovered".
"""
from __future__ import annotations

import numpy as np

from . import heldout_episode as H
from .replay import TrainModel

PUSHES = {                                   # [(duration_s, level / M)], q0  (row grid 50 Hz)
    "p0": ([(0.3, 3.0)], None),
    "p1": ([(0.4, -2.0)], 0.05),
    "p2": ([(0.2, 4.0), (0.2, 1.0)], -0.02),
    "p3": ([(0.5, 1.5)], 0.10),
    "p4_unseen": ([(0.3, -3.5)], 0.0),
}


def train_one(out_root, joint_type, name, pushes, q0, *, bias=0.0, quick=True, seed=0, log=print):
    from ..physx_refinement.rest_to_rest_comparison import run_synthetic
    return run_synthetic(out_root, joint_type=joint_type, quick=quick, seed=seed, pushes=pushes, q0=q0, bias=bias,
                         log=log)


def model_from_result(res, name, method) -> TrainModel:
    px, info = res["_px"], res["_info"]
    f = px["fits"][method]
    if f.get("status") != "FITTED":
        raise ValueError(f"{name}: {method} {f.get('status')}: {f.get('error')}")
    p = dict(f["params"])
    jt = res["truth"]["joint_type"]
    return TrainModel(name=name, r2r_dir="", method=method, chosen_method=method, params=p,
                      M=float(p["rho_scale"]) * float(info["axis_inertia"]), c=float(p["viscous"]),
                      rho_scale=float(p["rho_scale"]), axis_inertia=float(info["axis_inertia"]), joint_type=jt,
                      axis_B=np.asarray(info["axis"], float),
                      origin_B=np.asarray(info["origin"], float) if jt == "revolute" else None, urdf=res["_urdf"],
                      urdf_sha256="synthetic", moving_link=info["moving_link"], joint_name=info["joint_name"],
                      timestep=float(px["engine"]["timestep"]), friction_mode=px["engine"]["friction_mode"],
                      training_objective=float(f["objective"]), at_bound=f.get("at_bound") or {},
                      episode_sha256=res["_ep"].sha256())


def record_from_result(res, name):
    sy, info = res["_sy"], res["_info"]
    jt = res["truth"]["joint_type"]
    return H.synthetic_record(name, jt, info["axis"], info["origin"] if jt == "revolute" else None, sy["t_q"], sy["q"],
                              np.full(len(sy["t_q"]), sy["q_sigma"]), sy["force_t"], sy["Q"], res["_evidence"],
                              q_true=sy["q_true"])


def build_object(out_root, joint_type, *, misspecified=False, quick=True, log=print):
    """Train every push of one synthetic object; returns (object_id, results by push, truth)."""
    oid = f"synthetic_{joint_type}" + ("_misspecified_bias" if misspecified else "")
    res = {}
    for k, (pushes, q0) in PUSHES.items():
        res[f"{oid}_episode_{list(PUSHES).index(k):06d}"] = train_one(
            f"{out_root}/{oid}/{k}", joint_type, k, pushes, q0, bias=0.3 if misspecified else 0.0, quick=quick, log=log)
    truth = next(iter(res.values()))["truth"]
    return oid, res, truth
