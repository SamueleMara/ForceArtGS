#!/usr/bin/env python3
"""Full-episode contact-window and wrench-readiness audit for joint discovery (Stage D1-D3).

Read-only on the scene. For every robot sample (50 Hz) and every RGB-D frame it exports aligned
traces and a reviewable contact timeline with explicit state and provenance:

* robot rate: timestamp, source frame, TCP xyz / velocity, raw force + torque (sensor axes),
  |F - tare|, noise-normalised force z and the existing force-event state
  (``samuele/export_force_contact_annotations.detect_contact_states``), and the force on the object
  in world axes under the DECLARED conventions with its signed projection on the TCP motion;
* RGB-D rate: TCP-to-observed-object distance per camera (measured masked depth, robot tool
  removed by its measured kinematics), TCP visibility, a candidate-independent visual-motion
  signal (masked depth change w.r.t. the first frame), the fused timeline
  (``contact_annotations.fuse_sensor_cues``) and heuristic flags with their agreement.

Nothing here is an independent contact annotation: without operator/video labels the best state
is ``contact_probable``. The wrench calibration contract reports which force-selection gates pass.

    python samuele/audit_joint_contact_windows.py --scene <scene> --clip-frames 70:128:2 --output <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "samuele"))

from estimotion_2part.joint_discovery.contact_annotations import (evaluate_heuristic, fuse_sensor_cues,  # noqa: E402
                                                                  tcp_proximity_to_observed_object)
from estimotion_2part.joint_discovery.observations import (_segment_distance, load_scene_cameras,  # noqa: E402
                                                           parse_frame_range, unproject)
from estimotion_2part.joint_discovery.wrench_calibration import WrenchCalibration  # noqa: E402
from export_force_contact_annotations import (baseline_mask_from, detect_contact_states,  # noqa: E402
                                              locate_inputs, read_episode, rotvec_to_matrix)


def tool_segment(p_tcp, p_ee, tip_offset_m=0.015, extend_m=0.10):
    d = p_ee - p_tcp
    dn = d / max(np.linalg.norm(d), 1e-9)
    return p_tcp + tip_offset_m * dn, p_ee + extend_m * dn


def proximity(scene, calib, frame, p_tcp, p_ee, mask_dir="mask", window_px=60, tool_radius_m=0.03):
    import cv2
    depth = np.load(scene / "depth_npy" / calib.camera_id / f"frame_{frame:06d}.npy").astype(float)
    mask = cv2.imread(str(scene / mask_dir / calib.camera_id / f"frame_{frame:06d}.png"), cv2.IMREAD_GRAYSCALE) > 127
    a, b = tool_segment(p_tcp, p_ee)
    # remove the measured tool from the object mask in 3D (same rule as observations.exclude_tool_observations)
    v, u = np.nonzero(mask & (depth > 0))
    if len(u):
        P = unproject(np.c_[u, v].astype(float), depth[v, u], calib.K, calib.T_WC)
        on_tool = _segment_distance(P, np.repeat(a[None], len(P), 0), np.repeat(b[None], len(P), 0)) < tool_radius_m
        tm = np.zeros_like(mask)
        tm[v[on_tool], u[on_tool]] = True
    else:
        tm = None
    return tcp_proximity_to_observed_object(p_tcp, calib, depth, mask, window_px, tool_mask=tm)


def visual_motion(scene, cam, frames, mask_dir="mask"):
    """Median |z_t - z_0| [m] and fraction > 1 cm over pixels in both masks with valid depth."""
    import cv2
    z0 = np.load(scene / "depth_npy" / cam / f"frame_{frames[0]:06d}.npy").astype(float)
    m0 = cv2.imread(str(scene / mask_dir / cam / f"frame_{frames[0]:06d}.png"), cv2.IMREAD_GRAYSCALE) > 127
    med, frac = [], []
    for f in frames:
        z = np.load(scene / "depth_npy" / cam / f"frame_{f:06d}.npy").astype(float)
        m = cv2.imread(str(scene / mask_dir / cam / f"frame_{f:06d}.png"), cv2.IMREAD_GRAYSCALE) > 127
        ok = m & m0 & (z > 0) & (z0 > 0)
        d = np.abs(z - z0)[ok]
        med.append(float(np.median(d)) if len(d) else np.nan)
        frac.append(float(np.mean(d > 0.01)) if len(d) else np.nan)
    return np.array(med), np.array(frac)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--clip-frames", default=None, help="visual clip start:end:step (RGB-D frames), shown on plots")
    ap.add_argument("--cameras", nargs="+", default=["cam_001", "cam_002"])
    ap.add_argument("--near-m", type=float, default=0.025, help="TCP-to-object distance for 'near' (TCP error 1-2 cm)")
    ap.add_argument("--far-m", type=float, default=0.06, help="distance beyond which every camera says 'far'")
    ap.add_argument("--baseline-s", type=float, default=0.5)
    ap.add_argument("--contact-z", type=float, default=6.0)
    ap.add_argument("--absent-z", type=float, default=3.0)
    ap.add_argument("--min-contact-s", type=float, default=0.2)
    ap.add_argument("--heuristic-n", type=float, nargs="+", default=[0.3, 1.0])
    ap.add_argument("--tool-mass-upper-bound-kg", type=float, default=None,
                    help="declared upper bound for the gravity-constancy gate (None: gate stays UNVERIFIED)")
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args(argv)
    scene = a.scene
    out = a.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    conv, parquet, info = locate_inputs(scene)
    ep = read_episode(parquet)
    tool = json.loads((scene / "metadata" / "robot_tool.json").read_text())
    if tool.get("tcp_calibration_valid") is False:
        raise SystemExit("TCP calibration marked invalid; refusing")
    t, fi, W, st = ep["timestamp"], ep["frame_index"], ep["wrench"], ep["state"]
    R_ee = rotvec_to_matrix(st[:, 3:6])
    T_ee_tcp = np.asarray(tool["T_ee_tcp"], float)
    p_ee = st[:, :3]
    p_tcp = p_ee + np.einsum("nij,j->ni", R_ee, T_ee_tcp[:3, 3])
    v_tcp = np.gradient(p_tcp, t, axis=0)
    F = W[:, :3]
    # existing force-event detector (tare + MAD noise on the first baseline_s seconds)
    base = baseline_mask_from(t, fi, duration_s=a.baseline_s)
    det = detect_contact_states(t, F, base, contact_z=a.contact_z, absent_z=a.absent_z,
                                min_contact_duration_s=a.min_contact_s)
    # declared-convention world force (DECLARED, not verified)
    cal = WrenchCalibration.from_scene_metadata(scene)
    cal.tare_S.value = det["tare_N"].tolist()
    cal.tare_S.status = "ESTIMATED"
    cal.tare_S.method = f"mean over first {a.baseline_s} s (force-event detector baseline)"
    conv_f = cal.force_on_object_world(F, R_ee)
    f_obj = conv_f["f_object_w"]
    vn = np.linalg.norm(v_tcp, axis=1)
    tang = v_tcp / np.where(vn > 1e-9, vn, 1)[:, None]
    f_par = np.einsum("ni,ni->n", f_obj, tang)
    f_par[vn < 0.002] = np.nan
    # ---------------------------------------------------------------- RGB-D rate
    rows = list(csv.DictReader(open(scene / "metadata" / "frame_timestamps.csv")))
    rgbd = np.array([int(r["frame_index"]) for r in rows])
    src = np.array([int(r["source_frame_index"]) for r in rows])
    t_rgbd = np.array([float(r["timestamp_s"]) for r in rows])
    row_of = {int(s): i for i, s in enumerate(fi)}
    missing = [int(s) for s in src if int(s) not in row_of]
    if missing:
        raise SystemExit(f"robot Parquet lacks source frames {missing[:5]}")
    ri = np.array([row_of[int(s)] for s in src])
    calibs, _ = load_scene_cameras(scene, a.cameras)
    dist = np.full((len(rgbd), len(a.cameras)), np.nan)
    why = np.full((len(rgbd), len(a.cameras)), "", dtype=object)
    for k, f in enumerate(rgbd):
        for c, cam in enumerate(a.cameras):
            dist[k, c], why[k, c] = proximity(scene, calibs[cam], int(f), p_tcp[ri[k]], p_ee[ri[k]])
    dmin = np.nanmin(np.where(np.isfinite(dist), dist, np.inf), 1)
    seen = np.isfinite(dist).any(1)
    near = seen & (dmin <= a.near_m)
    far = seen & np.all(np.where(np.isfinite(dist), dist > a.far_m, True), 1)
    fstate = det["state"][ri]
    timeline = fuse_sensor_cues(src, fstate, near, far)
    vm_med, vm_frac = {}, {}
    for cam in a.cameras:
        vm_med[cam], vm_frac[cam] = visual_motion(scene, cam, [int(f) for f in rgbd])
    # heuristics as in the joint-discovery CLI: |F - tare| > N with tare = first 3 clip frames
    heur = {}
    clip = parse_frame_range(a.clip_frames) if a.clip_frames else list(rgbd)
    in_clip = np.isin(rgbd, clip)
    Fr = F[ri]
    for thr in a.heuristic_n:
        clip_idx = np.flatnonzero(in_clip)
        tare = np.median(Fr[clip_idx[:3]], 0)
        flags = np.linalg.norm(Fr - tare, axis=1) > thr
        heur[f"abs_dF_gt_{thr}N_clip_tare"] = dict(full_episode=evaluate_heuristic(flags, timeline),
                                                   clip=evaluate_heuristic(flags[in_clip], timeline.select(src[in_clip])))
    # ---------------------------------------------------------------- wrench readiness on geometric no-contact
    free_rgbd = timeline.state == "no_contact"
    free_rows = np.zeros(len(t), bool)
    for k in np.flatnonzero(free_rgbd):
        free_rows[max(ri[k] - 1, 0):ri[k] + 2] = True
    cal2 = WrenchCalibration.from_scene_metadata(scene)
    if a.tool_mass_upper_bound_kg is not None:
        cal2.tool_mass_upper_bound_kg.value = a.tool_mass_upper_bound_kg
        cal2.tool_mass_upper_bound_kg.status = "DECLARED"
    sustained = [e for e in det["episodes"] if e["sustained"]]
    absent = det["state"] == 0
    if free_rows.sum() >= 10:
        cal2.estimate_tare(F, free_rows, label="robot samples at fused no_contact RGB-D frames")
    elif sustained:
        pre = absent & (np.arange(len(t)) < sustained[0]["first_index"])
        post = absent & (np.arange(len(t)) > sustained[-1]["last_index"])
        near_note = "TCP within the near tolerance of object geometry" if near[:max(1, int(np.argmax(fstate == 1)))].all() else ""
        cal2.estimate_tare(F, pre, label=f"force-quiet samples before the first sustained force event "
                                         f"(sensor cue only; {near_note or 'geometry not checked'})",
                           second_mask=post if post.sum() >= 10 else None,
                           second_label="force-quiet samples after the last force event")
    else:
        cal2.estimate_tare(F, absent, label="force-quiet samples (no sustained force event in the episode)")
    clip_rows = np.zeros(len(t), bool)
    if in_clip.any():
        clip_rows[ri[in_clip].min():ri[in_clip].max() + 1] = True
    cal2.check_orientation_constancy(R_ee, clip_rows)
    cal2.check_static_multipose_gravity(F, R_ee, free_rows)
    ready = cal2.readiness()
    # ---------------------------------------------------------------- exports
    with open(out / "robot_rate_trace.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp_s", "source_frame_index", "tcp_x", "tcp_y", "tcp_z", "tcp_vx", "tcp_vy", "tcp_vz",
                    "Fx_raw_N", "Fy_raw_N", "Fz_raw_N", "Tx_raw_Nm", "Ty_raw_Nm", "Tz_raw_Nm", "abs_F_minus_tare_N",
                    "force_z", "force_event_state", "f_obj_world_declared_x", "f_obj_world_declared_y",
                    "f_obj_world_declared_z", "F_parallel_declared_N"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.6f}", int(fi[i]), *np.round(p_tcp[i], 6), *np.round(v_tcp[i], 6), *np.round(W[i], 5),
                        round(float(np.linalg.norm(det["force_corrected"][i])), 5), round(float(det["z"][i]), 3),
                        int(det["state"][i]), *np.round(f_obj[i], 5), "" if not np.isfinite(f_par[i]) else round(float(f_par[i]), 5)])
    with open(out / "rgbd_rate_timeline.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["rgbd_frame", "source_frame_index", "timestamp_s", "in_visual_clip", "force_event_state", "abs_F_minus_tare_N"]
        head += [f"tcp_obj_dist_{c}_m" for c in a.cameras] + [f"tcp_obj_status_{c}" for c in a.cameras]
        head += [f"visual_motion_median_{c}_m" for c in a.cameras] + ["near", "far", "contact_state", "provenance"]
        w.writerow(head)
        for k in range(len(rgbd)):
            w.writerow([int(rgbd[k]), int(src[k]), f"{t_rgbd[k]:.4f}", int(in_clip[k]), int(fstate[k]),
                        round(float(np.linalg.norm(det["force_corrected"][ri[k]])), 4)]
                       + ["" if not np.isfinite(d) else round(float(d), 4) for d in dist[k]] + list(why[k])
                       + [round(float(vm_med[c][k]), 4) for c in a.cameras]
                       + [int(near[k]), int(far[k]), timeline.state[k], timeline.provenance[k]])
    ev = det["episodes"]
    summary = dict(scene=str(scene), parquet=str(parquet), n_robot_samples=int(len(t)), n_rgbd_frames=int(len(rgbd)),
                   clip_frames=a.clip_frames, cameras=a.cameras, thresholds=dict(near_m=a.near_m, far_m=a.far_m,
                   contact_z=a.contact_z, absent_z=a.absent_z, min_contact_s=a.min_contact_s, baseline_s=a.baseline_s),
                   force_event_tare_N=det["tare_N"].tolist(), force_event_sigma_N=det["sigma_N"].tolist(),
                   force_events=[dict(e, t_start_s=float(t[e["first_index"]]), t_end_s=float(t[e["last_index"]]),
                                      source_frames=[int(fi[e["first_index"]]), int(fi[e["last_index"]])]) for e in ev],
                   peak_abs_dF_N=float(np.nanmax(np.linalg.norm(det["force_corrected"], axis=1))),
                   peak_abs_dF_in_clip_N=float(np.nanmax(np.linalg.norm(det["force_corrected"][ri[in_clip]], axis=1)))
                   if in_clip.any() else None,
                   timeline_counts_full=timeline.counts(), timeline_counts_clip=timeline.select(src[in_clip]).counts(),
                   rgbd_frames_by_state={s: rgbd[timeline.state == s].tolist() for s in
                                         ("contact_probable", "no_contact")},
                   tcp_near_frames=rgbd[near].tolist(), force_event_contact_frames=rgbd[fstate == 1].tolist(),
                   visual_motion_onset={c: _onset(rgbd, vm_frac[c]) for c in a.cameras},
                   heuristics=heur, wrench_calibration=cal2.to_json(), wrench_readiness=ready,
                   independent_annotations="none: no operator/video labels; best attainable state is contact_probable",
                   time_sync="exact source_frame_index join; hardware clock offset NOT measured")
    (out / "contact_window_summary.json").write_text(json.dumps(summary, indent=1, default=_np))
    _plot(out / "contact_timeline.png", t, fi, det, f_par, rgbd, t_rgbd, dist, a, timeline, vm_med, in_clip)
    print(json.dumps(dict(timeline_full=summary["timeline_counts_full"], timeline_clip=summary["timeline_counts_clip"],
                          force_events=[(e["source_frames"], round(e["peak_force_N"], 2), e["sustained"])
                                        for e in summary["force_events"]],
                          readiness=ready["failing"]), indent=1))
    return 0


def _onset(frames, frac, thr=0.05):
    hit = np.flatnonzero(np.nan_to_num(frac) > thr)
    return dict(first_frame_with_5pct_mask_depth_change_gt_1cm=int(frames[hit[0]]) if len(hit) else None)


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _plot(path, t, fi, det, f_par, rgbd, t_rgbd, dist, a, timeline, vm_med, in_clip):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    Fc = det["force_corrected"]
    for k, n in enumerate("xyz"):
        ax[0].plot(t, Fc[:, k], lw=0.8, label=f"F{n} - tare (sensor)")
    ax[0].plot(t, np.linalg.norm(Fc, axis=1), "k", lw=1.2, label="|F - tare|")
    ax[0].plot(t, f_par, "m--", lw=0.8, label="F_parallel on object (declared conv.)")
    ax[0].set_ylabel("N")
    ax[0].legend(fontsize=7, ncol=3)
    ax[1].plot(t, det["z"], lw=0.8)
    ax[1].axhline(a.contact_z, color="r", ls="--")
    ax[1].axhline(a.absent_z, color="g", ls="--")
    ax[1].set_ylabel("force z (noise-norm.)")
    for c, cam in enumerate(a.cameras):
        ax[2].plot(t_rgbd, dist[:, c], ".-", ms=3, label=f"TCP-object {cam}")
    ax[2].axhline(a.near_m, color="r", ls="--")
    ax[2].axhline(a.far_m, color="g", ls="--")
    ax[2].set_ylabel("m")
    ax[2].set_ylim(0, 0.2)
    ax[2].legend(fontsize=7)
    for cam, v in vm_med.items():
        ax[3].plot(t_rgbd, v, label=f"median |z_t - z_0| in mask {cam}")
    colors = dict(contact_probable="tab:red", no_contact="tab:green", contact_unknown="tab:gray",
                  contact_confirmed="tab:purple")
    for k in range(len(t_rgbd)):
        ax[3].axvspan(t_rgbd[k] - 0.02, t_rgbd[k] + 0.02, color=colors[timeline.state[k]], alpha=0.25, lw=0)
    ax[3].set_ylabel("m / state")
    ax[3].legend(fontsize=7)
    if in_clip.any():
        for x in ax:
            x.axvspan(t_rgbd[in_clip].min(), t_rgbd[in_clip].max(), color="yellow", alpha=0.08)
    ax[3].set_xlabel("time [s] (yellow: visual clip; bottom bands: red probable, green no_contact, grey unknown)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
