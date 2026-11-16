#!/usr/bin/env python3
"""Multi-camera audit of the calibrated TCP projection against observed RGB-D depth.

Distinguishes TCP occlusion from calibration error without refitting anything:

* The calibrated TCP and points along the tool axis (tip -> EE) are projected into every
  camera with each *existing* calibration (active ``metadata/cameras.json`` and optional
  alternatives such as the pre-registration backup).
* Observed depth at each projected pixel gives a signed residual
  ``r = z_predicted - z_observed`` (positive: something is nearer than the predicted point,
  e.g. occlusion or a calibration offset towards the camera).
* Visible tool-axis points should show r of about +tool radius (the axis lies behind the
  visible shaft surface).  A residual that is the same along the whole visible shaft, and in
  all cameras, points to calibration or TCP error; a residual confined to the tip region,
  with a consistent shaft, points to occlusion.
* For confirmed contact, the tip must lie on an observed surface when that surface is
  visible: the nearest back-projected depth point to the predicted tip is reported.

Visual identifiability is not inferred from depth: ``--visual-labels`` merges labels
written after inspecting the saved crops.  Run with a Python that has polars, OpenCV
and matplotlib (e.g. the yolo26 env).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "samuele"))
from export_force_contact_annotations import read_episode, rotvec_to_matrix  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cameras(path):
    data = json.loads(Path(path).read_text())
    cams = {}
    for cam in data["cameras"]:
        if cam.get("pose_convention") != "opencv_optical_camera_to_world" or cam.get("coordinate_frame", data.get("coordinate_frame")) != "base_link":
            raise ValueError(f"{path}: {cam['name']} is not an OpenCV camera->base_link pose")
        K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
        cams[cam["name"]] = dict(K=K, T_base_cam=np.asarray(cam["pose_matrix_4x4"], dtype=np.float64),
                                 size=(int(cam["width"]), int(cam["height"])))
    return dict(cameras=cams, status=data.get("calibration_status"), path=str(path), sha256=sha256(path))


def observed_depth(depth, u, v, half=2):
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return np.nan
    win = depth[max(0, vi - half):vi + half + 1, max(0, ui - half):ui + half + 1]
    win = win[np.isfinite(win) & (win > 0)]
    return float(np.median(win)) if win.size else np.nan


def nearest_surface(depth, K, p_cam, u, v, radius):
    """Nearest back-projected depth point to ``p_cam`` within ``radius`` px of (u, v)."""
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    y0, y1, x0, x1 = max(0, vi - radius), min(h, vi + radius + 1), max(0, ui - radius), min(w, ui + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    z = depth[y0:y1, x0:x1]
    ok = np.isfinite(z) & (z > 0)
    if not ok.any():
        return None
    xs, ys, z = xs[ok], ys[ok], z[ok]
    pts = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], axis=1)
    d = np.linalg.norm(pts - p_cam, axis=1)
    k = int(np.argmin(d))
    return dict(distance_m=float(d[k]), pixel=[int(xs[k]), int(ys[k])], offset_px=float(np.hypot(xs[k] - u, ys[k] - v)),
                delta_cam_m=(pts[k] - p_cam).tolist())


def classify(residual, tol):
    if not np.isfinite(residual):
        return "no_depth"
    if residual > tol:
        return "nearer_surface"
    if residual < -tol:
        return "farther_surface"
    return "depth_consistent"


def select_frames(conversion, annotations, count):
    out = np.asarray(conversion["output_frames"])
    if annotations is None:
        return out[np.linspace(0, len(out) - 1, count).round().astype(int)].tolist()
    states = {r["rgbd_frame_index"]: r["state"] for r in annotations["rgbd_alignment"]}
    contact = [f for f in out if states.get(int(f)) == "contact"]
    before = [f for f in out if states.get(int(f)) == "absent"]
    picks = list(np.asarray(before)[np.linspace(0, len(before) - 1, 2).round().astype(int)]) if before else []
    if contact:
        picks += list(np.asarray(contact)[np.linspace(0, len(contact) - 1, max(1, count - len(picks))).round().astype(int)])
    return sorted({int(f) for f in picks})


def check_tool(scene, tool):
    """Compare the scene's T_ee_tcp with the repository registry entry for its tool."""
    report = dict(scene_tool=tool.get("tool"), scene_T_ee_tcp=tool["T_ee_tcp"], recorded_pose_frame=tool.get("recorded_pose_frame"),
                  coordinate_frame=tool.get("coordinate_frame"), reference_assumption=tool.get("reference_assumption"))
    for label, path in (("repo_registry", REPO / "configs/real_robot_tools.json"), ("scene_registry", tool.get("registry_path"))):
        if not path or not Path(path).is_file():
            report[label] = dict(status="not_found", path=str(path))
            continue
        reg = json.loads(Path(path).read_text())
        tools = reg.get("tools", {})
        entry = tools.get(tool.get("tool"), {})
        ds = reg.get("datasets", {}).get(scene.name, {})
        T = np.asarray(entry.get("T_ee_tcp", np.full((4, 4), np.nan)), dtype=np.float64)
        report[label] = dict(path=str(path), sha256=sha256(path), dataset_tool=ds.get("tool"),
                             tcp_calibration_valid=entry.get("tcp_calibration_valid", ds.get("tcp_calibration_valid")),
                             max_abs_difference=float(np.nanmax(np.abs(T - np.asarray(tool["T_ee_tcp"])))) if np.isfinite(T).all() else None,
                             matches=bool(np.isfinite(T).all() and np.allclose(T, tool["T_ee_tcp"]) and ds.get("tool") == tool.get("tool")))
    return report


def crop_panel(rgb, uv_axis, residuals, tol, label, size=160):
    import cv2
    img = rgb.copy()
    for (u, v), r in zip(uv_axis[::-1], residuals[::-1]):
        if not np.isfinite(u):
            continue
        color = (128, 128, 128) if not np.isfinite(r) else ((0, 200, 0) if abs(r) <= tol else ((0, 0, 255) if r > 0 else (255, 128, 0)))
        cv2.circle(img, (int(round(u)), int(round(v))), 2, color, -1)
    u0, v0 = uv_axis[0]
    if np.isfinite(u0):
        cv2.drawMarker(img, (int(round(u0)), int(round(v0))), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
    h, w = img.shape[:2]
    cx = int(np.clip(np.nan_to_num(u0, nan=w / 2), size // 2, w - size // 2))
    cy = int(np.clip(np.nan_to_num(v0, nan=h / 2), size // 2, h - size // 2))
    crop = cv2.resize(img[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2], (size * 2, size * 2),
                      interpolation=cv2.INTER_NEAREST)
    cv2.putText(crop, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    cv2.putText(crop, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return crop


def audit(scene, output, *, frames=None, count=8, annotations=None, alt_cameras=(), tol=0.02,
          axis_step=0.01, axis_length=None, nearest_radius_px=25, visual_labels=None):
    import cv2
    scene, output = Path(scene).resolve(), Path(output).resolve()
    if scene in output.parents or output == scene:
        raise ValueError("Refusing to write inside the scene")
    output.mkdir(parents=True, exist_ok=True)
    conversion = json.loads((scene / "metadata/real_conversion.json").read_text())
    tool = json.loads((scene / "metadata/robot_tool.json").read_text())
    if tool.get("recorded_pose_frame") != "ee" or tool.get("coordinate_frame") != "base_link":
        raise ValueError("robot_tool.json must declare recorded_pose_frame=ee in base_link")
    T_ee_tcp = np.asarray(tool["T_ee_tcp"], dtype=np.float64)
    annotations = json.loads(Path(annotations).read_text()) if annotations else None
    contact_state = {r["rgbd_frame_index"]: r["state"] for r in annotations["rgbd_alignment"]} if annotations else {}
    episode = int(conversion["episode"])
    parquet = scene / "metadata/raw" / f"episode_{episode:06d}.parquet"
    ep = read_episode(parquet)
    row = {int(f): i for i, f in enumerate(ep["frame_index"])}
    src_of = dict(zip(conversion["output_frames"], conversion["source_frames"]))
    frames = frames or select_frames(conversion, annotations, count)
    calibrations = {"active": load_cameras(scene / "metadata/cameras.json")}
    for item in alt_cameras:
        name, path = item.split("=", 1)
        calibrations[name] = load_cameras(path)
    tcp_length = float(np.linalg.norm(T_ee_tcp[:3, 3]))
    axis_length = tcp_length if axis_length is None else axis_length
    s_values = np.round(np.arange(0.0, axis_length + 1e-9, axis_step), 4)
    labels = json.loads(Path(visual_labels).read_text()) if visual_labels else {}

    samples = []
    panels = {}
    for frame in frames:
        src = int(src_of[frame])
        k = row[src]
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = rotvec_to_matrix(ep["state"][k, 3:6])[0]
        T_base_ee[:3, 3] = ep["state"][k, :3]
        T_base_tcp = T_base_ee @ T_ee_tcp
        axis_tcp = np.stack([np.zeros_like(s_values), np.zeros_like(s_values), -s_values, np.ones_like(s_values)], axis=1)
        axis_base = (T_base_tcp @ axis_tcp.T).T[:, :3]
        for cal_name, cal in calibrations.items():
            for cam, info in cal["cameras"].items():
                depth = np.load(scene / "depth_npy" / cam / f"frame_{frame:06d}.npy")
                mask_path = scene / "mask" / cam / f"frame_{frame:06d}.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
                T_cam_base = np.linalg.inv(info["T_base_cam"])
                p_cam = (T_cam_base[:3, :3] @ axis_base.T).T + T_cam_base[:3, 3]
                z = p_cam[:, 2]
                uv = (info["K"] @ p_cam.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                w, h = info["size"]
                inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
                uv[~inside] = np.nan
                obs = np.array([observed_depth(depth, *uv[i]) if inside[i] else np.nan for i in range(len(s_values))])
                res = z - obs
                tip_near = nearest_surface(depth, info["K"], p_cam[0], *uv[0], nearest_radius_px) if inside[0] else None
                on_mask = None
                if mask is not None and inside[0]:
                    on_mask = bool(mask[int(round(uv[0, 1])), int(round(uv[0, 0]))] > 0)
                visual = labels.get(f"{cam}/{frame}")
                samples.append(dict(
                    calibration=cal_name, camera=cam, rgbd_frame_index=int(frame), source_frame_index=src,
                    contact_state=contact_state.get(int(frame)),
                    tip_uv=None if not inside[0] else uv[0].round(2).tolist(), tip_z_pred_m=float(z[0]),
                    tip_z_obs_m=None if not np.isfinite(obs[0]) else float(obs[0]),
                    tip_residual_m=None if not np.isfinite(res[0]) else float(res[0]),
                    tip_depth_class=classify(res[0], tol), tip_on_object_mask=on_mask,
                    tip_nearest_observed_surface=tip_near, visual_label=visual,
                    axis_s_m=s_values.tolist(),
                    axis_residual_m=[None if not np.isfinite(v) else float(v) for v in res],
                    axis_class=[classify(v, tol) for v in res]))
                if cal_name == "active" or len(calibrations) == 1 or True:
                    rgb = cv2.imread(str(scene / "rgb" / cam / f"frame_{frame:06d}.png"))
                    lab = f"{cal_name} {cam} f{frame} s{src} {contact_state.get(int(frame), '')} r_tip=" + \
                          ("nan" if not np.isfinite(res[0]) else f"{res[0]*100:+.1f}cm")
                    panels.setdefault((cal_name, cam), []).append(crop_panel(rgb, uv, res, tol, lab))
    for (cal_name, cam), crops in panels.items():
        rows = [np.hstack(crops[i:i + 4] + [np.zeros_like(crops[0])] * (4 - len(crops[i:i + 4]))) for i in range(0, len(crops), 4)]
        cv2.imwrite(str(output / f"crops_{cal_name}_{cam}.jpg"), np.vstack(rows))

    summary = summarise(samples, calibrations, s_values, tol)
    plot_axis_profiles(output / "axis_residual_profiles.png", samples, calibrations, s_values)
    report = dict(
        scene=str(scene), frames=frames, depth_tolerance_m=tol, tolerance_note="smoke-test value; not increased",
        residual_convention="r = z_predicted - z_observed (camera Z, m); r > 0: observed surface nearer than prediction",
        axis_note="axis samples run from the TCP (s=0) towards the EE origin along -z_tcp; visible shaft surface is "
                  "nearer than its axis by about the shaft radius, so a small positive r is expected there",
        calibrations={k: dict(path=v["path"], sha256=v["sha256"], status=v["status"]) for k, v in calibrations.items()},
        tool_check=check_tool(scene, tool), timing=dict(frame_matching=conversion.get("frame_matching"),
                                                       timestamps=conversion.get("timestamps")),
        contact_annotations=None if annotations is None else annotations.get("sources", {}).get("parquet"),
        summary=summary, samples=samples,
        visual_labels_source=None if not visual_labels else dict(path=str(visual_labels), sha256=sha256(visual_labels)))
    (output / "tcp_projection_audit.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def _stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)])
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), median_m=float(np.median(v)), p05_m=float(np.percentile(v, 5)),
                p95_m=float(np.percentile(v, 95)), mean_m=float(v.mean()))


def summarise(samples, calibrations, s_values, tol):
    out = {}
    for cal in calibrations:
        out[cal] = {}
        for cam in calibrations[cal]["cameras"]:
            rows = [r for r in samples if r["calibration"] == cal and r["camera"] == cam]
            tips = [r["tip_residual_m"] for r in rows]
            vis = [r["tip_residual_m"] for r in rows if (r["visual_label"] or {}).get("tip") == "visible"]
            occ = [r["tip_residual_m"] for r in rows if (r["visual_label"] or {}).get("tip") == "occluded"]
            per_s = {}
            for j, s in enumerate(s_values):
                vals = [r["axis_residual_m"][j] for r in rows]
                consistent = [v for v in vals if v is not None and abs(v) <= tol]
                per_s[f"{s:.3f}"] = dict(_stats(vals), depth_consistent=len(consistent))
            near = [r["tip_nearest_observed_surface"]["distance_m"] for r in rows if r["tip_nearest_observed_surface"]
                    and r["contact_state"] == "contact"]
            out[cal][cam] = dict(
                tip_residual_all=_stats(tips), tip_residual_visual_visible=_stats(vis), tip_residual_visual_occluded=_stats(occ),
                tip_depth_class_counts={c: sum(r["tip_depth_class"] == c for r in rows) for c in
                                        ("depth_consistent", "nearer_surface", "farther_surface", "no_depth")},
                tip_nearest_surface_distance_contact=_stats(near), axis_residual_by_s=per_s)
    return out


def plot_axis_profiles(path, samples, calibrations, s_values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cams = sorted({r["camera"] for r in samples})
    fig, axes = plt.subplots(len(calibrations), len(cams), figsize=(5 * len(cams), 3.6 * len(calibrations)), squeeze=False, sharey=True)
    for i, cal in enumerate(calibrations):
        for j, cam in enumerate(cams):
            ax = axes[i, j]
            for r in samples:
                if r["calibration"] == cal and r["camera"] == cam:
                    y = np.array([np.nan if v is None else v for v in r["axis_residual_m"]]) * 100
                    ax.plot(s_values * 100, y, ".-", lw=0.6, ms=3, alpha=0.7, label=str(r["rgbd_frame_index"]))
            ax.axhspan(-2, 2, color="green", alpha=0.08)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylim(-15, 15)
            ax.set_title(f"{cal} {cam}")
            ax.set_xlabel("distance from TCP along tool axis (cm)")
            ax.set_ylabel("z_pred - z_obs (cm)")
            ax.grid(alpha=0.3)
    axes[0, -1].legend(fontsize=6, ncol=2, title="rgbd frame")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", help="RGB-D output frame indices (default: auto)")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--contact-annotations", type=Path, help="contact_annotations.json from the force exporter")
    parser.add_argument("--alt-cameras", nargs="*", default=[], metavar="NAME=CAMERAS_JSON",
                        help="Additional existing calibrations, e.g. before=<backup>/metadata/cameras.json")
    parser.add_argument("--depth-tol-m", type=float, default=0.02)
    parser.add_argument("--axis-step-m", type=float, default=0.01)
    parser.add_argument("--nearest-radius-px", type=int, default=25)
    parser.add_argument("--visual-labels", type=Path, help='JSON {"cam_000/<frame>": {"tip": "visible|occluded|unclear", ...}}')
    args = parser.parse_args(argv)
    report = audit(args.scene, args.output, frames=args.frames, count=args.count, annotations=args.contact_annotations,
                   alt_cameras=args.alt_cameras, tol=args.depth_tol_m, axis_step=args.axis_step_m,
                   nearest_radius_px=args.nearest_radius_px, visual_labels=args.visual_labels)
    for cal, cams in report["summary"].items():
        for cam, s in cams.items():
            t = s["tip_residual_all"]
            print(f"{cal:8s} {cam}: tip r median {t.get('median_m')} n={t.get('n')} classes {s['tip_depth_class_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
