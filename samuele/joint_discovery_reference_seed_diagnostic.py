#!/usr/bin/env python3
"""ORACLE DIAGNOSTIC (never part of inference): optimizer failure vs evidence failure.

Fits the revolute hypothesis twice on the same observations: once from the production seeds and
once seeded at an independent reference hinge (e.g. the tracker-free 1-DOF reference in
``hinge_reference.json`` from the black-box root-cause study). If both converge to the same
solution and likelihood, the result is determined by the tracks, not by initialization.

    python samuele/joint_discovery_reference_seed_diagnostic.py --scene <scene> \
        --observations obs_cam_001.npz obs_cam_002.npz --hinge-reference hinge_reference.json \
        --reference-camera cam_000 --output diag.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.candidates import Seed  # noqa: E402
from estimotion_2part.joint_discovery.config import load_joint_discovery_config  # noqa: E402
from estimotion_2part.joint_discovery.exports import write_json  # noqa: E402
from estimotion_2part.joint_discovery.geometry_fit import fit_hypothesis  # noqa: E402
from estimotion_2part.joint_discovery.model_selection import fit_joint_hypotheses, observability_report  # noqa: E402
from estimotion_2part.joint_discovery.observations import (TrackObservations, concatenate_observations,  # noqa: E402
                                                           load_scene_cameras)
from estimotion_2part.joint_discovery.rigid_kinematics import axis_angle_deg, line_to_line_distance  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--observations", nargs="+", required=True, type=Path)
    ap.add_argument("--hinge-reference", required=True, type=Path)
    ap.add_argument("--reference-camera", default="cam_000", help="camera whose reference angle series seeds q")
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args(argv)
    ref = json.loads(a.hinge_reference.read_text())
    obs = concatenate_observations([TrackObservations.load(p) for p in a.observations])
    calibs, _ = load_scene_cameras(a.scene, obs.camera_ids)
    cfg = load_joint_discovery_config({"mode": "calibrated_joint_aware", "cameras": obs.camera_ids})
    fit = fit_joint_hypotheses(obs, calibs, cfg, ref=0)
    u_ref, o_ref = np.array(ref["hinge_direction"]), np.array(ref["hinge_point_nominal"])
    frames = ref["cameras"][a.reference_camera]["frames"]
    src = np.array([int(k) for k in frames])
    th = np.array([frames[k]["theta_deg"] for k in frames])
    q0 = np.radians(np.interp(obs.frame_map.rgbd_frame_index, src, th))   # keys are RGB-D frame ids
    q0 -= q0[0]
    seeded, nll_seeded = fit_hypothesis(fit.problem, Seed("revolute", u_ref, o_ref, q0, np.ones(len(q0), bool),
                                                          dict(oracle_reference_seed=True)))
    out = dict(oracle_diagnostic=True, cameras=obs.camera_ids, reference=str(a.hinge_reference),
               reference_q_deg=np.degrees(q0).tolist())
    for name, c in (("production", fit.candidates["revolute"]), ("reference_seeded", seeded)):
        out[name] = dict(training_nll=c.diagnostics["training_nll"], n_moving=c.diagnostics["n_moving"],
                         arc_deg=float(np.degrees(np.ptp(c.q))), q_deg=np.degrees(c.q).tolist(),
                         axis_vs_reference_deg=axis_angle_deg(c.axis_w, u_ref),
                         line_vs_reference_m=line_to_line_distance(c.axis_w, c.origin_w, u_ref, o_ref),
                         observability={k: v for k, v in observability_report(c, fit.problem).items()
                                        if k != "moving_obs_per_camera_time"})
    write_json(a.output, out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("q_deg", "observability")}
                      for k, v in out.items() if isinstance(v, dict)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
