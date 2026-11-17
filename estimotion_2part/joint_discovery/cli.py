"""Standalone diagnostics / short optimizations for calibrated joint-aware Phase-1 discovery.

    python -m estimotion_2part.joint_discovery.cli --help
    python -m estimotion_2part.joint_discovery.cli synthetic --kind revolute --output OUT
    python -m estimotion_2part.joint_discovery.cli extract --scene S --camera cam_001 --frames 70:128:2 --output obs_cam_001.npz
    python -m estimotion_2part.joint_discovery.cli from-dumps --scene S --camera cam_002 --dump-dir D --frames 70:128:2 --output obs.npz
    python -m estimotion_2part.joint_discovery.cli fit --scene S --observations a.npz b.npz --output RUN_DIR [--tcp-npz F]

Nothing here modifies a dataset or a legacy output; runs go to a fresh directory.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import assignments as asg
from .config import load_joint_discovery_config
from .exports import (candidate_to_json, export_for_existing_freeart, git_provenance, package_versions,
                      virtual_camera_poses, write_joint_states, write_json)
from .model_selection import (DECISIONS, bic_report, choose_reference_time, compare_to_reference,
                              fit_joint_hypotheses, make_heldout_split, observability_report,
                              score_candidates_on_heldout)
from .observations import (TrackObservations, concatenate_observations, load_scene_cameras, load_scene_frame_map,
                           parse_frame_range)
from .rigid_kinematics import apply_joint, decompose_prismatic_velocity, decompose_revolute_velocity
from .visual_losses import camera_stacks, render_point_proxy, residual_report, whitened_residuals

REPO = Path(__file__).resolve().parents[2]


def _log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- velocity diagnostics
def velocity_field_diagnostics(cands: dict, obs: TrackObservations) -> dict:
    """Finite-difference 3D velocities of tracked points vs each hypothesis' exact field."""
    from .tcp_contact import joint_rate
    h = np.flatnonzero(obs.has_3d)
    order = h[np.lexsort((obs.time[h], obs.track[h]))]
    tr, t, x = obs.track[order], obs.time[order], obs.x_w[order]
    nxt = (tr[1:] == tr[:-1]) & (t[1:] == t[:-1] + 1)
    i0 = np.flatnonzero(nxt)
    ts = obs.frame_map.timestamp_s
    dt = ts[t[i0 + 1]] - ts[t[i0]]
    v = (x[i0 + 1] - x[i0]) / dt[:, None]
    mid = 0.5 * (x[i0 + 1] + x[i0])
    out = dict(track=tr[i0], t0=t[i0], timestamp_mid_s=0.5 * (ts[t[i0]] + ts[t[i0 + 1]]), velocity_w=v, point_w=mid)
    for kind, c in cands.items():
        key = "R" if kind == "revolute" else "P"
        qd = joint_rate(c.q, ts)
        # chord-consistent rate between the two samples (not the instantaneous tangent of a large arc)
        rate = (c.q[t[i0 + 1]] - c.q[t[i0]]) / dt
        if kind == "revolute":
            from .rigid_kinematics import body_velocity
            pred = body_velocity(kind, c.axis_w, c.origin_w, rate, mid)
            dec = decompose_revolute_velocity(c.axis_w, c.origin_w, mid, v - pred)
            out.update({f"pred_{key}": pred, f"radial_resid_{key}": dec["radial"], f"axial_resid_{key}": dec["axial"],
                        f"tangential_resid_{key}": dec["tangential"], f"radius_{key}": dec["radius"]})
        else:
            dec = decompose_prismatic_velocity(c.axis_w, v)
            out.update({f"perp_speed_{key}": dec["perpendicular_norm"],
                        f"common_speed_resid_{key}": dec["parallel"] - rate})
        out[f"moving_prob_{key}"] = c.moving_probability[tr[i0]]
        out[f"qdot_{key}"] = qd
    return out


def summarize_velocity(vd: dict) -> dict:
    out = {}
    for key in ("R", "P"):
        if f"moving_prob_{key}" not in vd:
            continue
        m = np.nan_to_num(vd[f"moving_prob_{key}"]) > 0.5
        if key == "R":
            out["revolute"] = {k: float(np.median(np.abs(vd[f"{k}_resid_R"][m]))) if m.any() else None
                               for k in ("radial", "axial", "tangential")}
        else:
            out["prismatic"] = dict(perp_speed=float(np.median(vd["perp_speed_P"][m])) if m.any() else None,
                                    common_speed_resid=float(np.median(np.abs(vd["common_speed_resid_P"][m]))) if m.any() else None)
        out[f"n_moving_velocity_samples_{key}"] = int(m.sum())
    return out


# --------------------------------------------------------------------------- run orchestration
def run_discovery(obs: TrackObservations, calibs: dict, cfg, out_dir, scene=None, contact=None, wrench_convention=None,
                  oracle=None, extra_manifest=None) -> dict:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    T = obs.frame_map.T
    ref = choose_reference_time(obs)
    heldout = make_heldout_split(T, ref, cfg.comparison.heldout_time_fraction, cfg.fit.seed)
    train = obs.select_obs(~np.isin(obs.time, heldout))
    test = obs.select_obs(np.isin(obs.time, heldout))
    write_json(out / "input_audit.json", dict(summary=obs.summary(), reference_time_index=ref,
                                              heldout_times=heldout, cameras={c: calibs[c].to_json() for c in obs.camera_ids},
                                              frame_map=obs.frame_map.to_json(), provenance=obs.provenance,
                                              contact_available=contact is not None,
                                              wrench_available=bool(contact is not None and contact.wrench_sensor is not None)))
    obs.save(out / "tracks_observations.npz")
    _log(f"[S1/S2] fitting on {len(np.unique(train.time))} training times, held out {heldout.tolist()}")
    fit_tr = fit_joint_hypotheses(train, calibs, cfg, ref=ref, log=_log)
    decision = score_candidates_on_heldout(fit_tr, test, cfg)
    decision.heldout_times = heldout.tolist()
    _log(f"[selection] {decision.label}: {decision.reasons}")
    # depth ablation on the same split (B2/B3 style: 2D-only vs 2D+depth evidence)
    _log("[final] refitting both hypotheses on all times (selection already made on the held-out split)")
    fit_all = fit_joint_hypotheses(obs, calibs, cfg, ref=ref, log=_log)
    cands = fit_all.candidates
    problem = fit_all.problem
    # ---- artifacts: candidates, states, support
    cj = {k: candidate_to_json(c, obs.frame_map) for k, c in cands.items()}
    for k in cj:
        cj[k]["observability"] = observability_report(cands[k], problem)
        if oracle is not None:
            cj[k]["oracle_diagnostic_only"] = compare_to_reference(cands[k], oracle.get("axis"), oracle.get("origin"))
    write_json(out / "joint_candidates.json", dict(candidates=cj, decision=decision.label,
                                                   coordinate_conventions=dict(T_AB="maps B to A, column vectors",
                                                                               twist="[omega; v]", units="m, rad, s")))
    write_joint_states(out / "joint_states.npz", cands, obs.frame_map)
    (out / "part_support").mkdir(exist_ok=True)
    arrays = dict(track_camera=obs.track_camera, track_local_id=obs.track_local_id, eligible=problem.eligible,
                  static_anchor_w=problem.static_anchors)
    for k, c in cands.items():
        key = "R" if k == "revolute" else "P"
        arrays.update({f"p_static_{key}": c.static_probability, f"p_moving_{key}": c.moving_probability,
                       f"p_outlier_{key}": c.outlier_probability, f"anchor_w_{key}": c.canonical_anchors_w})
    np.savez(out / "part_support" / "track_support.npz", **arrays)
    _write_track_fate(out / "tracks_fate.csv", obs, problem, cands)
    vd = velocity_field_diagnostics(cands, obs)
    np.savez(out / "velocity_field_diagnostics.npz", **{k: np.asarray(v) for k, v in vd.items()})
    # ---- per-camera projection residuals
    K, T_CW = camera_stacks(obs, calibs)
    proj = {}
    for k, c in cands.items():
        res = whitened_residuals(k, c.axis_w, c.origin_w, c.q, np.nan_to_num(c.canonical_anchors_w), obs, K, T_CW,
                                 problem.noise, use_depth=problem.use_depth)
        proj[k] = residual_report(res, obs, np.nan_to_num(c.moving_probability), problem.noise)
    write_json(out / "camera_projection" / "residuals.json", proj)
    # ---- contact / force (S3/S4)
    cf = None
    if contact is not None:
        cf = contact_force_stage(cands, contact, cfg, out, calibs, wrench_convention, problem)
    sel = dict(decision=decision.label, allowed=DECISIONS, reasons=decision.reasons, scores=decision.scores,
               heldout_times=decision.heldout_times, n_scored_tracks=decision.n_scored_tracks,
               bic_training_diagnostic=bic_report(fit_tr), seed_training_nll=fit_tr.seed_scores,
               training_split_candidates={k: dict(axis=c.axis_w, origin=c.origin_w, q=c.q) for k, c in fit_tr.candidates.items()},
               noise_calibration=fit_tr.noise_report, robot_terms_used_in_selection=False,
               note="selection is vision-only (S2); contact/force evidence is reported separately (S3/S4)")
    write_json(out / "model_selection.json", sel)
    integ = integration_checks(cands, calibs, obs)
    write_json(out / "integration_checks.json", integ)
    best = cands.get(decision.label)
    if best is not None:
        export_for_existing_freeart(best, {c: calibs[c] for c in obs.camera_ids}, obs.frame_map, out / "freeart_adapter")
    manifest = dict(created_unix=time.time(), runtime_s=round(time.time() - t_start, 1), config=cfg.to_dict(),
                    config_hash=cfg.config_hash(), git=git_provenance(REPO), packages=package_versions(), scene=str(scene),
                    cameras=obs.camera_ids, camera_sources={c: calibs[c].calibration_source for c in obs.camera_ids},
                    frame_map=obs.frame_map.to_json(), robot_available=contact is not None,
                    oracle_used_for_fitting=False, oracle_reported=oracle is not None, extra=extra_manifest or {})
    write_json(out / "manifest.json", manifest)
    summary = dict(decision=decision.label, reasons=decision.reasons,
                   candidates={k: dict(axis=cj[k]["joint_axis_world"], origin=cj[k]["joint_origin_world"],
                                       q_span=cj[k]["observability"]["q_span"], q_units=cj[k]["q_units"],
                                       n_moving=cj[k]["n_moving_tracks"], flags=cj[k]["observability"]["degeneracy_flags"],
                                       oracle=cj[k].get("oracle_diagnostic_only"))
                               for k in cj},
                   velocity=summarize_velocity(vd), projection={k: {cam: dict(median_uv_px=v.get("median_uv_px"),
                                                                              n_moving=v["n_moving_tracks"])
                                                                   for cam, v in proj[k].items()} for k in proj},
                   heldout_scores={k: v for k, v in decision.scores.items() if "vs" in k},
                   contact_force=None if cf is None else cf.get("summary"))
    write_json(out / "summary.json", summary)
    _write_report(out / "report.md", summary, sel, obs, integ)
    _log(f"[done] {out} ({manifest['runtime_s']} s)")
    return summary


def _write_track_fate(path, obs, problem, cands):
    n2d, n3d = obs.per_track_counts()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["track", "camera", "local_id", "query_time", "n_obs_2d", "n_obs_3d", "eligible", "ineligible_reason"]
        for k in cands:
            head += [f"label_{k}", f"p_moving_{k}"]
        w.writerow(head)
        conf = problem.cfg.fit.label_confidence
        labs = {k: asg.labels(np.stack([c.static_probability, c.moving_probability, c.outlier_probability], 1),
                              problem.eligible, conf) for k, c in cands.items()}
        for i in range(obs.n_tracks):
            row = [i, obs.camera_ids[obs.track_camera[i]], int(obs.track_local_id[i]), int(obs.track_query_time[i]),
                   int(n2d[i]), int(n3d[i]), bool(problem.eligible[i]), problem.ineligible_reason[i]]
            for k, c in cands.items():
                row += [labs[k][i], "" if not np.isfinite(c.moving_probability[i]) else round(float(c.moving_probability[i]), 4)]
            w.writerow(row)


def integration_checks(cands, calibs, obs) -> dict:
    """Camera consistency and virtual-camera equivalence on the fitted candidates."""
    out = dict(physical_camera_poses_optimized=False, legacy_mode_untouched=True,
               quaternion_boundary="not exercised: joint_discovery uses rotation matrices only; GSplat xyzw/wxyz "
                                   "must be checked before any Gaussian training (not run here)")
    for kind, c in cands.items():
        moving = np.nan_to_num(c.moving_probability) > 0.5
        X = c.canonical_anchors_w[moving]
        X = X[np.isfinite(X).all(1)][:400]
        if len(X) == 0:
            continue
        rows = {}
        for cid in obs.camera_ids:
            cal = calibs[cid]
            Tv = virtual_camera_poses(c, cal.T_WC)
            t = int(np.argmax(np.abs(c.q)))
            from .rigid_kinematics import A_joint
            from .observations import CameraCalib
            A = A_joint(kind, c.axis_w, c.origin_w, c.q[t])
            rgb1, d1, a1 = render_point_proxy(X, np.ones((len(X), 3)), cal, T_world_points=A)
            vcam = CameraCalib(cid, cal.K, Tv[t], cal.width, cal.height, "virtual")
            rgb2, d2, a2 = render_point_proxy(X, np.ones((len(X), 3)), vcam)
            rows[cid] = dict(time_index=t, max_depth_diff_m=float(np.max(np.abs(d1 - d2))),
                             alpha_mismatch_px=int(np.sum(a1 != a2)))
        out[f"virtual_camera_equivalence_{kind}"] = rows
    return out


def _write_report(path, summary, sel, obs, integ):
    lines = ["# Joint discovery run report", "",
             "Generated by `estimotion_2part/joint_discovery/cli.py`. Selection is vision-only and uses held-out",
             "timestamps; ground truth (if any) is reported as an oracle diagnostic only.", "",
             f"**Decision:** `{summary['decision']}`", ""]
    lines += [f"- {r}" for r in summary["reasons"]]
    lines += ["", "## Candidates", "", "| kind | axis (world) | hinge point | q span | moving tracks | flags |",
              "|---|---|---|---|---|---|"]
    for k, v in summary["candidates"].items():
        ax = np.round(v["axis"], 4).tolist()
        o = None if v["origin"] is None else np.round(v["origin"], 4).tolist()
        span = v["q_span"]
        span_s = f"{np.degrees(span):.2f} deg" if v["q_units"] == "rad" else f"{span * 100:.2f} cm"
        lines.append(f"| {k} | {ax} | {o if o is not None else 'N/A'} | {span_s} | {v['n_moving']} | {', '.join(v['flags']) or '-'} |")
    lines += ["", "## Held-out predictive comparison", "", "```", json.dumps(summary["heldout_scores"], indent=1), "```",
              "", "## Velocity-field residuals (median |.| over moving samples, m/s)", "", "```",
              json.dumps(summary["velocity"], indent=1), "```", "", "## Projection", "", "```",
              json.dumps(summary["projection"], indent=1), "```", "", "## Inputs", "", "```",
              json.dumps(obs.summary()["per_camera"], indent=1), "```"]
    if summary.get("contact_force"):
        lines += ["", "## Contact / force (S3/S4, reported separately from selection)", "", "```",
                  json.dumps(summary["contact_force"], indent=1, default=str), "```"]
    lines += ["", "## Not run / limitations", "",
              "- Level-V2 Gaussian rendering feedback, appearance loss and downstream 3DGS training were not run.",
              "- Held-out *camera* evaluation was not run (time hold-out only).",
              f"- Integration: {json.dumps({k: v for k, v in integ.items() if k.startswith('virtual')})}"]
    Path(path).write_text("\n".join(lines) + "\n")


def contact_force_stage(cands, rec, cfg, out, calibs, convention, problem) -> dict:
    """S3/S4: TCP sliding-contact and wrench evidence on identical samples for both hypotheses."""
    from .tcp_contact import (contact_kinematics, contact_residuals, free_plane_degeneracy,
                              surface_from_observed_patch)
    from .wrench_physics import (decompose_force, force_direction_evidence, force_on_object_world, null_controls,
                                 passivity_penalty, power_terms)
    T = len(rec.timestamp_s)
    rows = dict(t=np.arange(T), timestamp_s=rec.timestamp_s, contact_valid=rec.contact_valid,
                p_tcp_x=rec.p_tcp_w[:, 0], p_tcp_y=rec.p_tcp_w[:, 1], p_tcp_z=rec.p_tcp_w[:, 2],
                v_tcp_x=rec.v_tcp_w[:, 0], v_tcp_y=rec.v_tcp_w[:, 1], v_tcp_z=rec.v_tcp_w[:, 2])
    summary = dict(contact_samples=int(rec.contact_valid.sum()), reasons=list(rec.reasons), T_WB_source=rec.T_WB_source)
    kin, surf, cres = {}, {}, {}
    centers = [calibs[c].center_w for c in problem.obs.camera_ids]
    for kind, c in cands.items():
        kin[kind] = contact_kinematics(c, rec)
        moving = np.nan_to_num(c.moving_probability) > 0.5
        pts = c.canonical_anchors_w[moving]
        act = rec.contact_valid
        if act.any():
            x0 = kin[kind]["canonical_contact"][act]
            from .rigid_kinematics import apply_joint_inverse
            cam0 = [apply_joint_inverse(kind, c.axis_w, c.origin_w, 0.0, cc) for cc in centers]
            surf[kind] = surface_from_observed_patch(pts, x0, cfg.contact, cam0)
        else:
            from .tcp_contact import SurfaceSupport, UNAVAILABLE
            surf[kind] = SurfaceSupport(UNAVAILABLE, "no verified contact samples")
        deg = free_plane_degeneracy(kin[kind]["canonical_contact"][rec.contact_valid])
        summary.setdefault("_deg", {})[kind] = deg
    # identical contact samples for every hypothesis: intersect per-hypothesis surface support
    pre = {k: contact_residuals(c, rec, surf[k], cfg.contact) for k, c in cands.items()}
    common = np.ones(T, bool)
    for k, r in pre.items():
        common &= r.get("surface_supported", np.zeros(T, bool)) if r.get("status") == "OK" else np.zeros(T, bool)
    for kind, c in cands.items():
        cres[kind] = contact_residuals(c, rec, surf[kind], cfg.contact, common_mask=common)
        deg = summary["_deg"][kind]
        key = "R" if kind == "revolute" else "P"
        x0 = kin[kind]["canonical_contact"]
        rows.update({f"x0_{key}_x": x0[:, 0], f"x0_{key}_y": x0[:, 1], f"x0_{key}_z": x0[:, 2],
                     f"slip_{key}_norm": np.linalg.norm(kin[kind]["slip"], axis=1), f"qdot_{key}": kin[kind]["qdot"]})
        if cres[kind].get("status") == "OK":
            rows[f"point_to_plane_{key}_m"] = cres[kind]["point_to_plane_m"]
            rows[f"normal_slip_{key}_mps"] = cres[kind]["normal_slip_mps"]
        summary[kind] = dict(surface=dict(status=surf[kind].status, reason=surf[kind].reason, n_points=surf[kind].n_points),
                             contact=dict(status=cres[kind].get("status"), n_active=cres[kind].get("n_active"),
                                          L_contact=cres[kind].get("L_contact")),
                             free_plane_degeneracy=deg,
                             canonical_contact_path_length_m=kin[kind]["canonical_path_length_m"])
        write_json(out / "contact_surface" / f"{kind}.json",
                   dict(status=surf[kind].status, reason=surf[kind].reason, n0=surf[kind].n0, a0=surf[kind].a0,
                        rms_m=surf[kind].rms_m, n_points=surf[kind].n_points, source=surf[kind].source))
    summary.pop("_deg", None)
    summary["common_supported_contact_samples"] = int((common & rec.contact_valid).sum())
    if rec.wrench_sensor is not None and rec.R_WS is not None and convention is not None:
        f_obj = force_on_object_world(rec.wrench_sensor[:, :3], rec.R_WS, convention)
        dec = decompose_force(f_obj, rec.v_tcp_w, cfg.contact.min_tcp_speed_mps, cfg.contact.min_force_n)
        rows.update(f_obj_x=f_obj[:, 0], f_obj_y=f_obj[:, 1], f_obj_z=f_obj[:, 2], force_norm=dec["force_norm"],
                    F_parallel=dec["F_parallel"], F_perp=dec["F_perp"], rho_parallel=dec["rho_parallel"],
                    rho_perp=dec["rho_perp"], cos_phi=dec["cos_phi"], force_valid=dec["valid"], force_reason=dec["reason"])
        active = dec["valid"] & rec.contact_valid
        active_surface = active & common
        fs = dict(convention=convention.__dict__, calibrated=convention.calibrated, n_valid_force_samples=int(dec["valid"].sum()),
                  n_valid_and_contact=int(active.sum()),
                  exclusion_counts={r: int((dec["reason"] == r).sum()) for r in set(dec["reason"].tolist()) if r})
        if cfg.require_calibrated_wrench and not convention.calibrated:
            fs["physics_terms"] = "DISABLED: wrench not tare/gravity compensated (require_calibrated_wrench: true)"
        rng = np.random.default_rng(cfg.fit.seed)
        nulls = null_controls(f_obj, rng)
        for kind, c in cands.items():
            key = "R" if kind == "revolute" else "P"
            pw = power_terms(kind, c.axis_w, c.origin_w, rec.p_tcp_w, rec.v_tcp_w, f_obj, kin[kind]["qdot"])
            rows.update({f"Q_{key}": pw["Q"], f"P_joint_{key}": pw["P_joint"], f"D_{key}": pw["D"]})
            sigP = 0.01
            fs[f"passivity_{kind}"] = passivity_penalty(pw["D"], sigP, active=active)
            fs[f"passivity_{kind}_nulls"] = {n: passivity_penalty(
                power_terms(kind, c.axis_w, c.origin_w, rec.p_tcp_w, rec.v_tcp_w, f, kin[kind]["qdot"])["D"], sigP,
                active=active) for n, f in nulls.items()}
        fs["n_valid_contact_and_surface_supported"] = int(active_surface.sum())
        fs["force_direction"] = force_direction_evidence(f_obj, kin, cands, surf, active_surface, cfg.contact)
        summary["force"] = fs
    _write_csv(out / "contact_force_diagnostics.csv", rows)
    return dict(summary=summary)


def _write_csv(path, cols: dict):
    keys = list(cols)
    n = len(next(iter(cols.values())))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for i in range(n):
            w.writerow([_fmt(cols[k][i]) for k in keys])


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.6g}"
    if isinstance(v, (np.bool_, bool)):
        return int(v)
    return v


# --------------------------------------------------------------------------- extraction (GPU, AllTracker)
def extract_alltracker(scene, camera, rgbd_indices, output, query_every=10, grid_stride=4, vis_threshold=0.5,
                       downscale=1, mask_dir="mask", tcp_npz=None, tool_radius_m=0.03):
    """Run the repository's AllTracker on raw RGB and build a partial-track observation table.

    Queries start every ``query_every`` frames (tracks may begin after time 0). Depth is sampled
    from raw ``depth_npy`` with ``sample_depth_valid``; nothing is written into the scene."""
    import cv2
    import torch
    from omegaconf import OmegaConf
    from tracking.tracker_toolkit import Alltracker
    from .observations import build_from_dense_tracks
    scene = Path(scene)
    calibs, audit = load_scene_cameras(scene, [camera])
    cal = calibs[camera]
    fm = load_scene_frame_map(scene, rgbd_indices)
    rgbs, depths, masks = [], [], []
    for i in rgbd_indices:
        stem = f"frame_{i:06d}"
        img = cv2.imread(str(scene / "rgb" / camera / f"{stem}.png"))[:, :, ::-1]
        if img.shape[:2] != (cal.height, cal.width):
            raise ValueError(f"{camera} image size {img.shape[:2]} != calibration {(cal.height, cal.width)}")
        rgbs.append(img)
        depths.append(np.load(scene / "depth_npy" / camera / f"{stem}.npy").astype(float))
        m = cv2.imread(str(scene / mask_dir / camera / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        masks.append(m > 127)
    cfg = OmegaConf.load(REPO / "tracking" / "configs" / "alltracker.yaml")
    tracker = Alltracker(cfg)
    parts = []
    H, W = cal.height // downscale, cal.width // downscale
    for q0 in range(0, len(rgbd_indices) - 2, query_every):
        clip = np.stack([cv2.resize(r, (W, H), interpolation=cv2.INTER_AREA) for r in rgbs[q0:]])
        x = torch.from_numpy(clip).permute(0, 3, 1, 2)[None].float().cuda()
        with torch.inference_mode():
            traj, vis = tracker.get_track_maps(dict(rgbs_tensor=x))
        traj = traj[0, :, :, ::grid_stride, ::grid_stride].reshape(traj.shape[1], 2, -1).permute(0, 2, 1).cpu().numpy()
        visc = vis[0, :, 0, ::grid_stride, ::grid_stride].reshape(vis.shape[1], -1).cpu().numpy()
        uv_full = (traj + 0.5) * downscale - 0.5
        n_grid = uv_full.shape[1]
        part = build_from_dense_tracks(cal, fm, uv_full, visc, depths[q0:], masks[q0:], q0,
                                       track_ids=np.arange(n_grid) + 1_000_000 * q0, vis_threshold=vis_threshold,
                                       provenance=dict(tracker="AllTracker (tracking/tracker_toolkit.Alltracker)",
                                                       query_time=q0, grid_stride=grid_stride, downscale=downscale,
                                                       depth_sampler="sample_depth_valid (valid same-surface only)",
                                                       mask_dir=mask_dir, scene=str(scene)))
        parts.append(part)
        del traj, vis, x
        torch.cuda.empty_cache()
    obs = _merge_same_camera(parts)
    obs.provenance = dict(parts=[p.provenance for p in parts], calibration_audit=audit)
    if tcp_npz is not None:
        from .observations import exclude_tool_observations
        from .tcp_contact import load_robot_contact_record
        rec = load_robot_contact_record(scene, fm, tcp_npz=tcp_npz)
        obs, rep = exclude_tool_observations(obs, calibs, rec.p_tcp_w, rec.p_ee_w, radius_m=tool_radius_m)
        _log(f"tool exclusion: {rep}")
    obs.save(output)
    _log(f"extracted {obs.n_tracks} tracks / {obs.n_obs} observations for {camera} -> {output}")
    return obs


def _merge_same_camera(parts):
    from .observations import OBS_FIELDS
    kw = {k: [] for k in OBS_FIELDS}
    tc, tl, tq, off = [], [], [], 0
    for p in parts:
        for k in OBS_FIELDS:
            arr = getattr(p, k)
            kw[k].append(arr + off if k == "track" else arr)
        tc.append(p.track_camera)
        tl.append(p.track_local_id)
        tq.append(p.track_query_time)
        off += p.n_tracks
    return TrackObservations(parts[0].camera_ids, parts[0].frame_map, track_camera=np.concatenate(tc),
                             track_local_id=np.concatenate(tl), track_query_time=np.concatenate(tq),
                             **{k: np.concatenate(v) for k, v in kw.items()})


def build_robot_archive(scene, output):
    """Write ``T_base_ee``, ``T_base_tcp = T_base_ee @ T_ee_tcp``, ``source_row`` and raw ``wrench``.

    Reads ``metadata/raw/*.parquet`` (``observation.state`` = EE position + rotation vector,
    ``observation.wrench`` = recorded wrench in EE axes) and ``metadata/robot_tool.json``.
    Refuses scenes whose TCP calibration is marked invalid. Nothing is interpolated."""
    import glob
    import polars as pl
    from scipy.spatial.transform import Rotation
    scene = Path(scene)
    tool = json.loads((scene / "metadata" / "robot_tool.json").read_text())
    if tool.get("tcp_calibration_valid") is False:
        raise SystemExit(f"{scene.name}: robot_tool.json marks the TCP calibration invalid; refusing to build a TCP archive")
    if tool.get("recorded_pose_frame", "ee") != "ee":
        raise SystemExit(f"{scene.name}: recorded pose frame {tool.get('recorded_pose_frame')!r} is not 'ee'")
    files = sorted(glob.glob(str(scene / "metadata" / "raw" / "*.parquet")))
    if len(files) != 1:
        raise SystemExit(f"{scene.name}: expected one raw Parquet, found {len(files)}")
    df = pl.read_parquet(files[0])
    st = np.stack(df["observation.state"].to_list()).astype(float)
    wr = np.stack(df["observation.wrench"].to_list()).astype(float)
    T_ee = np.tile(np.eye(4), (len(st), 1, 1))
    T_ee[:, :3, :3] = Rotation.from_rotvec(st[:, 3:6]).as_matrix()
    T_ee[:, :3, 3] = st[:, :3]
    T_ee_tcp = np.asarray(tool["T_ee_tcp"], float)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, T_base_ee=T_ee, T_base_tcp=T_ee @ T_ee_tcp, source_row=df["frame_index"].to_numpy(),
             wrench=wr[:, :6])
    _log(f"robot archive: {len(st)} rows, tool {tool.get('tool')}, T_ee_tcp z {T_ee_tcp[2, 3]:.4f} m -> {output}")


def summarize_runs(runs, ref_axis=None, ref_point=None) -> str:
    from .rigid_kinematics import axis_angle_deg, line_to_line_distance
    head = ("| run | cameras | depth | decision | z(R vs P) | z(R vs static) | z(P vs static) | R arc [deg] | "
            "R hinge radius [m] | P span [cm] | moving tracks R/P | R axis vs ref [deg] | R line vs ref [cm] |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in runs:
        s = json.loads((r / "summary.json").read_text())
        man = json.loads((r / "manifest.json").read_text())
        jc = json.loads((r / "joint_candidates.json").read_text())["candidates"]
        sc = s["heldout_scores"]
        R, P = jc.get("revolute"), jc.get("prismatic")
        z = lambda k: f"{sc[k]['z']:.1f}" if k in sc else "-"
        ax = ln = "-"
        if R is not None and ref_axis is not None:
            ax = f"{axis_angle_deg(R['joint_axis_world'], ref_axis):.1f}"
            if ref_point is not None:
                ln = f"{100 * line_to_line_distance(R['joint_axis_world'], R['joint_origin_world'], ref_axis, ref_point):.1f}"
        rad = R["observability"].get("hinge_radius_m", {}).get("median") if R else None
        lines.append(f"| {r.name} | {'+'.join(man['cameras'])} | {man['config']['enable_depth']} | {s['decision']} | "
                     f"{z('revolute_vs_prismatic')} | {z('revolute_vs_static')} | {z('prismatic_vs_static')} | "
                     f"{np.degrees(R['observability']['q_span']):.1f} | {rad if rad is None else round(rad, 3)} | "
                     f"{100 * P['observability']['q_span']:.1f} | {R['n_moving_tracks'] if R else '-'}/"
                     f"{P['n_moving_tracks'] if P else '-'} | {ax} | {ln} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m estimotion_2part.joint_discovery.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic", help="analytic synthetic scene end to end (oracle reported as diagnostic only)")
    s.add_argument("--kind", choices=["revolute", "prismatic"], default="revolute")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-angle-deg", type=float, default=40.0)
    s.add_argument("--max-disp-m", type=float, default=0.12)
    s.add_argument("--config", default=None)
    s.add_argument("--output", required=True, type=Path)
    e = sub.add_parser("extract", help="run AllTracker on raw RGB and save a partial-track observation table (GPU)")
    e.add_argument("--scene", required=True, type=Path)
    e.add_argument("--camera", required=True)
    e.add_argument("--frames", required=True, help="start:end:step RGB-D frame indices, inclusive end")
    e.add_argument("--query-every", type=int, default=10)
    e.add_argument("--grid-stride", type=int, default=4)
    e.add_argument("--downscale", type=int, default=1)
    e.add_argument("--mask-dir", default="mask")
    e.add_argument("--tcp-npz", type=Path, default=None, help="exclude the measured robot tool (TCP->EE shaft)")
    e.add_argument("--tool-radius-m", type=float, default=0.03)
    e.add_argument("--output", required=True, type=Path)
    d = sub.add_parser("from-dumps", help="adapter for samuele/dump_window_tracks.py archives")
    d.add_argument("--scene", required=True, type=Path)
    d.add_argument("--camera", required=True)
    d.add_argument("--dump-dir", required=True, type=Path)
    d.add_argument("--frames", required=True, help="RGB-D frames of the dumped clip, start:end:step")
    d.add_argument("--output", required=True, type=Path)
    f = sub.add_parser("fit", help="fit static/P/R with held-out selection and write a run directory")
    f.add_argument("--scene", required=True, type=Path)
    f.add_argument("--observations", nargs="+", required=True, type=Path)
    f.add_argument("--config", default=None, help="YAML/JSON with a joint_discovery section")
    f.add_argument("--cameras", nargs="+", default=None, help="override config cameras (cam_000 = explicit stress test)")
    f.add_argument("--no-depth", action="store_true", help="2D reprojection only (ablation)")
    f.add_argument("--tcp-npz", type=Path, default=None, help="robot TCP/wrench archive joined on source frames")
    f.add_argument("--contact-force-threshold-n", type=float, default=None,
                   help="label contact where |F - tare| exceeds this (heuristic, recorded in the manifest)")
    f.add_argument("--tare-frames", default=None, help="model-time range a:b used as wrench tare (pre-contact)")
    f.add_argument("--contact-annotations-csv", type=Path, default=None,
                   help="independent contact annotations (source_frame_index, contact_active); joined exactly")
    f.add_argument("--output", required=True, type=Path)
    a = sub.add_parser("robot-archive", help="TCP/EE/wrench archive from the raw Parquet (measured EE x T_ee_tcp)")
    a.add_argument("--scene", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path)
    m = sub.add_parser("summarize", help="markdown table over run directories (reference = oracle diagnostic only)")
    m.add_argument("--runs", nargs="+", required=True, type=Path)
    m.add_argument("--reference-axis", type=float, nargs=3, default=None, help="diagnostic reference hinge direction")
    m.add_argument("--reference-point", type=float, nargs=3, default=None, help="diagnostic reference hinge point")
    args = ap.parse_args(argv)

    if args.cmd == "summarize":
        print(summarize_runs(args.runs, args.reference_axis, args.reference_point))
        return 0
    if args.cmd == "robot-archive":
        build_robot_archive(args.scene, args.output)
        return 0
    if args.cmd == "synthetic":
        from .synthetic import make_scene
        cfg = load_joint_discovery_config(args.config or {"mode": "calibrated_joint_aware"})
        sc = make_scene(args.kind, seed=args.seed, max_angle_deg=args.max_angle_deg, max_disp_m=args.max_disp_m,
                        late_tracks_fraction=0.2, dropout=0.1)
        run_discovery(sc.obs, sc.calibs, cfg, args.output, scene="synthetic",
                      oracle=dict(axis=sc.gt_axis, origin=sc.gt_origin))
        return 0
    if args.cmd == "extract":
        extract_alltracker(args.scene, args.camera, parse_frame_range(args.frames), args.output,
                           args.query_every, args.grid_stride, downscale=args.downscale, mask_dir=args.mask_dir,
                           tcp_npz=args.tcp_npz, tool_radius_m=args.tool_radius_m)
        return 0
    if args.cmd == "from-dumps":
        from .observations import from_window_track_dumps
        idx = parse_frame_range(args.frames)
        calibs, _ = load_scene_cameras(args.scene, [args.camera])
        fm = load_scene_frame_map(args.scene, idx)
        obs = from_window_track_dumps(args.dump_dir, calibs[args.camera], fm)
        obs.save(args.output)
        _log(json.dumps(obs.summary(), indent=1))
        return 0
    if args.cmd == "fit":
        cfg = load_joint_discovery_config(args.config or {"mode": "calibrated_joint_aware"})
        if args.cameras:
            cfg.cameras = list(args.cameras)
        if args.no_depth:
            cfg.enable_depth = False
        if not cfg.is_new_mode:
            raise SystemExit("joint_discovery.mode is 'legacy'; nothing to do (use calibrated_joint_aware)")
        parts = [TrackObservations.load(p) for p in args.observations]
        parts = [p for p in parts if p.camera_ids[0] in cfg.cameras]
        missing = set(cfg.cameras) - {p.camera_ids[0] for p in parts}
        if missing:
            raise SystemExit(f"no observation table for cameras {sorted(missing)}")
        obs = concatenate_observations(parts)
        calibs, audit = load_scene_cameras(args.scene, obs.camera_ids)
        contact, conv = None, None
        extra = dict(calibration_audit=audit, observation_files=[str(p) for p in args.observations])
        if args.tcp_npz is not None:
            from .tcp_contact import load_robot_contact_record
            from .wrench_physics import WrenchConvention
            meta = args.scene / "metadata"
            conv = WrenchConvention.from_scene_metadata(json.loads((meta / "contact_force_convention.json").read_text()),
                                                        json.loads((meta / "robot_tool.json").read_text()))
            flags = None
            contact = load_robot_contact_record(args.scene, obs.frame_map, tcp_npz=args.tcp_npz)
            if args.contact_force_threshold_n is not None and contact.wrench_sensor is not None:
                F = contact.wrench_sensor[:, :3]
                tare = np.zeros(3)
                if args.tare_frames:
                    a, b = map(int, args.tare_frames.split(":"))
                    tare = np.median(F[a:b], 0)
                flags = np.linalg.norm(F - tare, axis=1) > args.contact_force_threshold_n
                contact.contact_valid = flags
                contact.reasons = [r for r in contact.reasons if not r.startswith("no independently verified")]
                contact.contact_source = (f"heuristic: |F_recorded - tare| > {args.contact_force_threshold_n} N "
                                          f"(tare frames {args.tare_frames}); not an independent contact annotation")
                contact.reasons.append(contact.contact_source)
                extra["contact_flags"] = contact.contact_source
                extra["wrench_tare_N"] = tare.tolist()
                contact.wrench_sensor = contact.wrench_sensor.copy()
                contact.wrench_sensor[:, :3] -= tare
            if args.contact_annotations_csv is not None:
                with open(args.contact_annotations_csv, newline="") as fh:
                    ann = {int(r["source_frame_index"]): int(r["contact_active"]) for r in csv.DictReader(fh)}
                missing = [int(s) for s in obs.frame_map.source_frame_index if int(s) not in ann]
                if missing:
                    raise SystemExit(f"contact annotations lack source frames {missing[:5]}")
                contact.contact_valid = np.array([bool(ann[int(s)]) for s in obs.frame_map.source_frame_index])
                contact.contact_source = f"annotations: {args.contact_annotations_csv}"
                contact.reasons = [r for r in contact.reasons if not r.startswith("no independently verified")]
                contact.reasons.append(contact.contact_source)
                extra["contact_flags"] = contact.contact_source
        run_discovery(obs, calibs, cfg, args.output, scene=args.scene, contact=contact, wrench_convention=conv,
                      extra_manifest=extra)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
