#!/usr/bin/env python3
"""Evaluate an A/B/C ablation produced by ``run_tcp_contact_ablation.py`` (CPU).

Independent of the optimisation objectives where possible:

* masks       accuracy of the final per-frame part masks (mask_0/1) on reference regions
              drawn by visual inspection (``--reference-regions``), moving vs static.
* rigid fit   frame-0 part points (mask + depth) moved by the ICP part motion; median
              distance to the observed object depth at frame t (one-sided, robust).
* contact     distance of the confirmed contact point to the moved moving-part surface.
              Phase C optimises a related quantity per window, so this is not independent
              for configurations with C enabled (reported, flagged).
* multiview   moving-part motion of each camera expressed in base_link; agreement of the
              per-frame rotation angle and of the motion of a base point between cameras,
              and of the screw axis over the clip (a diagnostic, not the trained joint).
* Phase A     raw-tracker robot-point errors, visibility, requery events, object tracks.
* priors      per-window status, effective samples and loss contributions of B/C.

Writes ``ablation_metrics.json`` and ``ablation_table.md`` into ``--output``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def screw_axis(T):
    """Rotation axis (unit) and angle of a rigid transform; axis None for tiny rotations."""
    R = T[:3, :3]
    angle = rotation_angle_deg(R)
    if angle < 1e-3:
        return None, angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / np.linalg.norm(w), angle


def motion_score(poses):
    """Same scoring as prepare_multiview: translation path + rotation path (rad)."""
    t = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()
    r = sum(np.radians(rotation_angle_deg(a[:3, :3].T @ b[:3, :3])) for a, b in zip(poses[:-1], poses[1:]))
    return float(t + r)


class CameraRun:
    def __init__(self, root):
        self.root = Path(root)
        self.K = np.loadtxt(self.root / "cam_K.txt")
        self.stems = sorted(p.stem for p in (self.root / "images").glob("*.png"))
        self.G = [np.load(self.root / f"global_poses_{k}.npy") for k in (0, 1)]
        self.valid = [np.load(self.root / f"pose_valid_{k}.npy") if (self.root / f"pose_valid_{k}.npy").is_file()
                      else np.ones(len(self.stems), bool) for k in (0, 1)]
        # ICP global pose G[t] maps frame-t points into frame-0 coordinates; motion 0 -> t:
        self.M = [np.linalg.inv(G) @ G[0] for G in self.G]
        scores = [motion_score(G) for G in self.G]
        self.moving = int(np.argmax(scores))
        self.motion_scores = scores
        conversion = json.loads((self.root / "metadata/freeart_conversion.json").read_text())
        self.camera = conversion["selected_camera"]
        self.scene = Path(conversion["scene_root"])
        tcp = self.root / "metadata/tcp_contact/tcp_contact.npz"
        self.tcp = dict(np.load(tcp)) if tcp.is_file() else None

    def depth(self, i):
        with np.load(self.root / "depth" / f"{self.stems[i]}.npz") as data:
            return data["depth"].astype(np.float64)

    def mask(self, folder, i):
        m = cv2.imread(str(self.root / folder / f"{self.stems[i]}.png"), cv2.IMREAD_GRAYSCALE)
        return m > 127

    def points(self, mask, depth, max_points=4000, seed=0):
        ys, xs = np.nonzero(mask & (depth > 0) & np.isfinite(depth))
        if len(xs) > max_points:
            pick = np.random.default_rng(seed).choice(len(xs), max_points, replace=False)
            xs, ys = xs[pick], ys[pick]
        z = depth[ys, xs]
        return np.stack([(xs - self.K[0, 2]) / self.K[0, 0] * z, (ys - self.K[1, 2]) / self.K[1, 1] * z, z], 1)


def transform(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def rigid_fit(run):
    """Median distance of moved frame-0 part points to the observed object at each frame."""
    d0 = run.depth(0)
    out = {}
    for name, part in (("moving", run.moving), ("static", 1 - run.moving)):
        P0 = run.points(run.mask(f"mask_{part}", 0), d0)
        per_frame, identity = [], []
        for t in range(1, len(run.stems)):
            if not run.valid[part][t] or len(P0) == 0:
                continue
            obs = run.points(run.mask("masks", t), run.depth(t), max_points=20000, seed=t)
            if len(obs) == 0:
                continue
            tree = cKDTree(obs)
            per_frame.append(float(np.median(tree.query(transform(run.M[part][t], P0))[0])))
            identity.append(float(np.median(tree.query(P0)[0])))
        out[name] = dict(points=int(len(P0)), frames=len(per_frame),
                         median_m=float(np.median(per_frame)) if per_frame else None,
                         p90_over_frames_m=float(np.percentile(per_frame, 90)) if per_frame else None,
                         last_frame_m=per_frame[-1] if per_frame else None,
                         no_motion_median_m=float(np.median(identity)) if identity else None)
    return out


def contact_residual(run):
    if run.tcp is None:
        return dict(status="no_tcp_artifact")
    P0 = run.points(run.mask(f"mask_{run.moving}", 0), run.depth(0), max_points=8000)
    active = run.tcp["contact_active"].astype(bool)
    dists = []
    for t in range(len(run.stems)):
        if not active[t] or not run.valid[run.moving][t]:
            continue
        p = run.tcp["p_contact_cam_m"][t]
        dists.append(float(np.min(np.linalg.norm(transform(run.M[run.moving][t], P0) - p, axis=1))))
    if not dists:
        return dict(status="no_confirmed_contact_frames")
    return dict(status="available", frames=len(dists), median_m=float(np.median(dists)),
                p90_m=float(np.percentile(dists, 90)), first_m=dists[0], last_m=dists[-1])


def reference_accuracy(run, reference):
    cams = (reference or {}).get("cameras", {})
    if run.camera not in cams:
        return dict(status="no_reference_for_camera")
    rows, totals = [], dict(moving_hit=0, moving_n=0, static_hit=0, static_n=0)
    for frame, regions in cams[run.camera].items():
        i = int(frame)
        moving_mask = run.mask(f"mask_{run.moving}", i)
        static_mask = run.mask(f"mask_{1 - run.moving}", i)
        row = dict(frame=i)
        for label, predicted in (("moving", moving_mask), ("static", static_mask)):
            region = np.zeros(moving_mask.shape, np.uint8)
            for poly in regions.get(label, []):
                cv2.fillPoly(region, [np.asarray(poly, np.int32)], 1)
            region = region.astype(bool)
            n = int(region.sum())
            hit = int((predicted & region).sum())
            row[f"{label}_accuracy"] = hit / n if n else None
            row[f"{label}_unassigned"] = float((~moving_mask & ~static_mask & region).sum() / n) if n else None
            totals[f"{label}_hit"] += hit
            totals[f"{label}_n"] += n
        rows.append(row)
    moving_acc = totals["moving_hit"] / max(1, totals["moving_n"])
    static_acc = totals["static_hit"] / max(1, totals["static_n"])
    return dict(status="available", frames=rows, moving_accuracy=moving_acc, static_accuracy=static_acc,
                balanced_accuracy=(moving_acc + static_acc) / 2)


def base_motion(run, T_base_cam):
    return [T_base_cam @ M @ np.linalg.inv(T_base_cam) for M in run.M[run.moving]]


def camera_pose(scene, camera):
    data = json.loads((Path(scene) / "metadata/cameras.json").read_text())
    for cam in data["cameras"]:
        if cam["name"] == camera:
            return np.asarray(cam["pose_matrix_4x4"], dtype=np.float64)
    raise KeyError(camera)


def multiview_agreement(runs):
    if len(runs) < 2:
        return dict(status="single_camera")
    names = sorted(runs)
    base = {c: base_motion(runs[c], camera_pose(runs[c].scene, c)) for c in names}
    a, b = names[0], names[1]
    probe = None
    tcp = runs[a].tcp
    if tcp is not None and "p_tcp_base_m" in tcp:
        probe = np.asarray(tcp["p_tcp_base_m"][0], dtype=np.float64)  # a base point near the contact
    rows = []
    for t in range(1, len(runs[a].stems)):
        if not (runs[a].valid[runs[a].moving][t] and runs[b].valid[runs[b].moving][t]):
            continue
        Ba, Bb = base[a][t], base[b][t]
        row = dict(t=t, angle_a_deg=rotation_angle_deg(Ba[:3, :3]), angle_b_deg=rotation_angle_deg(Bb[:3, :3]),
                   relative_rotation_deg=rotation_angle_deg(Ba[:3, :3].T @ Bb[:3, :3]))
        if probe is not None:
            row["probe_displacement_difference_m"] = float(np.linalg.norm(transform(Ba, probe) - transform(Bb, probe)))
            row["probe_displacement_a_m"] = float(np.linalg.norm(transform(Ba, probe) - probe))
        rows.append(row)
    axes = {c: screw_axis(base[c][-1]) for c in names}
    axis_angle = None
    if all(axes[c][0] is not None for c in names):
        axis_angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(axes[a][0], axes[b][0])), -1, 1))))
    rel = [r["relative_rotation_deg"] for r in rows]
    probe_diff = [r["probe_displacement_difference_m"] for r in rows if "probe_displacement_difference_m" in r]
    return dict(status="available" if rows else "no_common_valid_frames", cameras=names, frames=len(rows),
                relative_rotation_median_deg=float(np.median(rel)) if rel else None,
                relative_rotation_last_deg=rel[-1] if rel else None,
                probe_difference_median_m=float(np.median(probe_diff)) if probe_diff else None,
                probe_difference_last_m=probe_diff[-1] if probe_diff else None,
                final_angle_deg={c: rotation_angle_deg(base[c][-1][:3, :3]) for c in names},
                final_axis_base={c: None if axes[c][0] is None else axes[c][0].round(4).tolist() for c in names},
                axis_disagreement_deg=axis_angle, per_frame=rows)


def phase_a(run):
    summary_path = run.root / "tracking/tcp_track_summary.json"
    if not summary_path.is_file():
        return dict(status="not_run")
    summary = json.loads(summary_path.read_text())
    residuals, windows, attempts = [], [], []
    for path in sorted((run.root / "tracking").glob("window_*/tcp_track/tcp_track.json")):
        data = json.loads(path.read_text())
        npz = np.load(path.with_suffix(".npz"))
        valid = npz["valid_comparison"].astype(bool)
        residuals += npz["residual_px"][valid].tolist()
        windows.append(dict(window=path.parts[-3], status=data["status"], valid=int(valid.sum()),
                            visibility=data.get("visibility_status_counts"),
                            span=data.get("accepted_window_span"), original_span=data.get("original_window_span")))
        attempts += [dict(a, window=path.parts[-3]) for a in data.get("requery_attempts", [])]
    r = np.asarray(residuals)
    return dict(status=summary.get("status"), valid_comparisons=int(len(r)),
                median_px=float(np.median(r)) if len(r) else None, p95_px=float(np.percentile(r, 95)) if len(r) else None,
                requery_proposed=sum(a.get("proposed", False) for a in attempts),
                requery_accepted=sum(bool(a.get("accepted")) for a in attempts), requery_attempts=attempts,
                windows=windows)


def priors(run):
    rows = []
    for path in sorted((run.root / "tracking").glob("window_*/priors/priors.json")):
        data = json.loads(path.read_text())
        joint = data.get("joint") or {}
        contact, seed = joint.get("contact") or {}, joint.get("seed") or {}
        rows.append(dict(window=path.parts[-3], seed_pixels=data.get("seed_pixels"),
                         seed_status=seed.get("status"), seed_branch=seed.get("moving_branch"),
                         seed_tracks=seed.get("seed_tracks"), L_seed_weighted=seed.get("L_seed"),
                         contact_status=contact.get("status"), contact_reason=contact.get("reason"),
                         N_contact=contact.get("N_contact"),
                         L_contact_weighted_final=contact.get("L_contact_weighted_final"),
                         L_visual_final=contact.get("L_visual_final"),
                         contact_distance_before_median_m=float(np.median(contact["distance_before_m"]))
                         if contact.get("distance_before_m") else None,
                         contact_distance_after_median_m=float(np.median(contact["distance_after_m"]))
                         if contact.get("distance_after_m") else None,
                         pair_contact_active=sum((p.get("contact") or {}).get("status") == "active" for p in data["pairs"]),
                         pair_seed_active=sum((p.get("seed") or {}).get("status") == "active" for p in data["pairs"]),
                         pairs=len(data["pairs"])))
    report = run.root / "tracking/tracking_priors.json"
    return dict(windows=rows, context=json.loads(report.read_text()) if report.is_file() else None)


def windows(run):
    spans = []
    for path in sorted((run.root / "tracking").glob("window_*/window_info.json")):
        spans.append(json.loads(path.read_text())["frame_indices"])
    return dict(count=len(spans), lengths=[len(s) for s in spans])


def evaluate(work, configs, cameras, reference):
    work = Path(work)
    results = {}
    for cfg in configs:
        runs = {}
        for cam in cameras:
            path = work / "datasets" / f"{cfg}_{cam}"
            if (path / "global_poses_0.npy").is_file():
                runs[cam] = CameraRun(path)
        entry = dict(cameras={})
        for cam, run in runs.items():
            entry["cameras"][cam] = dict(moving_part=run.moving, motion_scores=run.motion_scores,
                                         masks=reference_accuracy(run, reference), rigid_fit=rigid_fit(run),
                                         contact=contact_residual(run), phase_a=phase_a(run), priors=priors(run),
                                         windows=windows(run))
        entry["multiview"] = multiview_agreement(runs)
        mv = work / "datasets" / f"{cfg}_mv/multiview.json"
        if mv.is_file():
            data = json.loads(mv.read_text())
            entry["multiview_preparation"] = dict(reference_camera=data["reference_camera"],
                                                  part_assignments=data["part_assignments"])
        results[cfg] = entry
    return results


def fmt(v, scale=1.0, digits=1):
    return "–" if v is None else f"{v * scale:.{digits}f}"


def table(results, cameras):
    lines = ["| Config | cam | mask moving acc | mask static acc | rigid fit moving (mm) | rigid fit static (mm) "
             "| contact dist median (mm) | A valid / median px | A requery acc. | B joint seed status | C joint N / status |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg, entry in results.items():
        for cam in cameras:
            c = entry["cameras"].get(cam)
            if c is None:
                continue
            m, rf, ct, a = c["masks"], c["rigid_fit"], c["contact"], c["phase_a"]
            seeds = sorted({str(w["seed_status"]) for w in c["priors"]["windows"]}) or ["–"]
            contacts = [f"{w['N_contact']}/{w['contact_status']}" for w in c["priors"]["windows"]] or ["–"]
            lines.append(f"| {cfg} | {cam} | {fmt(m.get('moving_accuracy'), 1, 3)} | {fmt(m.get('static_accuracy'), 1, 3)} "
                         f"| {fmt(rf['moving']['median_m'], 1000)} | {fmt(rf['static']['median_m'], 1000)} "
                         f"| {fmt(ct.get('median_m'), 1000)} | {a.get('valid_comparisons', '–')} / {fmt(a.get('median_px'))} "
                         f"| {a.get('requery_accepted', '–')} | {','.join(seeds)} | {' '.join(contacts[:4])} |")
    lines += ["", "| Config | cross-camera rel. rotation median (deg) | probe diff median (mm) | final angle per camera (deg) | axis disagreement (deg) |",
              "|---|---|---|---|---|"]
    for cfg, entry in results.items():
        mv = entry["multiview"]
        angles = ", ".join(f"{k}: {v:.1f}" for k, v in (mv.get("final_angle_deg") or {}).items())
        lines.append(f"| {cfg} | {fmt(mv.get('relative_rotation_median_deg'), 1, 2)} | {fmt(mv.get('probe_difference_median_m'), 1000)} "
                     f"| {angles or '–'} | {fmt(mv.get('axis_disagreement_deg'), 1, 1)} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--cameras", nargs="+", required=True)
    parser.add_argument("--reference-regions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reference = json.loads(args.reference_regions.read_text()) if args.reference_regions else None
    results = evaluate(args.work_root, args.configs, args.cameras, reference)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ablation_metrics.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    text = table(results, args.cameras)
    (args.output / "ablation_table.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
