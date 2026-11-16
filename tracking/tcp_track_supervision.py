"""Phase A: compare the raw AllTracker track of the robot TCP with its measured projection.

Pure NumPy helpers (no model initialisation at import time).  The TCP is a robot point:
results here are diagnostics or window-split proposals.  Nothing here moves object tracks,
inserts the TCP into object correspondences, or trains AllTracker (its weights are frozen
and it runs under ``torch.inference_mode``).

Map layout (from ``Alltracker.get_track_maps``):
    traj_maps  (T, 2, H, W): channel 0 = u (x), channel 1 = v (y) of the point queried at
                             integer pixel (x, y) of window frame ``query_frame`` (= 0).
    vis_maps   (T, C, H, W): channel 0 = visibility, channel 1 = confidence (when present).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

STATUS_VISIBLE = "visible"
STATUS_OCCLUDED = "occluded_by_nearer_surface"
STATUS_NO_DEPTH = "no_valid_depth"
STATUS_DEPTH_BEHIND = "tip_not_observed_in_depth"
STATUS_IN_OBJECT_MASK = "ambiguous_inside_object_mask"
STATUS_NO_PROJECTION = "no_valid_projection"
STATUS_ARTIFICIAL = "artificial_duplicate_frame"


# ----------------------------------------------------------------------------- sampling


def _as_numpy(array):
    if hasattr(array, "detach"):
        return array.detach().float().cpu().numpy()
    return np.asarray(array, dtype=np.float64)


def bilinear_sample(maps, uv) -> np.ndarray:
    """Bilinearly sample (T, C, H, W) maps at one sub-pixel location ``uv=(u, v)``.

    Only the 2x2 neighbourhood is transferred, so torch CUDA maps are fine.  Returns
    (T, C); NaN when the location is outside the map.
    """
    u, v = float(uv[0]), float(uv[1])
    T, C, H, W = maps.shape
    if not (np.isfinite(u) and np.isfinite(v)) or u < 0 or v < 0 or u > W - 1 or v > H - 1:
        return np.full((T, C), np.nan)
    x0, y0 = int(np.floor(u)), int(np.floor(v))
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    ax, ay = u - x0, v - y0
    patch = _as_numpy(maps[:, :, y0:y1 + 1, x0:x1 + 1]).astype(np.float64)
    # Pad degenerate border patches to 2x2.
    if patch.shape[2] == 1:
        patch = np.concatenate([patch, patch], axis=2)
    if patch.shape[3] == 1:
        patch = np.concatenate([patch, patch], axis=3)
    top = patch[:, :, 0, 0] * (1 - ax) + patch[:, :, 0, 1] * ax
    bottom = patch[:, :, 1, 0] * (1 - ax) + patch[:, :, 1, 1] * ax
    return top * (1 - ay) + bottom * ay


def extract_seeded_track(traj_maps, vis_maps, seed_uv):
    """Raw tracker prediction for the image feature at ``seed_uv`` in window frame 0.

    Returns (pred_uv (T,2), visibility (T,), confidence (T,) or NaN).
    """
    sampled = bilinear_sample(traj_maps, seed_uv)
    vis = bilinear_sample(vis_maps, seed_uv)
    visibility = vis[:, 0]
    confidence = vis[:, 1] if vis.shape[1] > 1 else np.full(len(vis), np.nan)
    return sampled[:, :2], visibility, confidence


# ----------------------------------------------------------------------------- visibility


def tcp_visibility(uv, z_point_m, depth_map, object_mask, depth_tolerance_m):
    """Physical observability of a robot point from depth ordering and image evidence.

    ``visible`` only if the observed depth at the nearest pixel agrees with the point depth
    within ``depth_tolerance_m`` and the pixel is not labelled object foreground (robot pixels
    are excluded from object masks, so a tip pixel inside the mask is ambiguous contact).
    """
    if depth_tolerance_m is None or not np.isfinite(depth_tolerance_m) or depth_tolerance_m <= 0:
        raise ValueError("tcp_visibility_depth_tol_m must be explicitly configured and positive")
    uv = np.asarray(uv, dtype=np.float64)
    if not np.isfinite(uv).all() or not np.isfinite(z_point_m) or z_point_m <= 0:
        return STATUS_NO_PROJECTION, np.nan
    H, W = depth_map.shape[:2]
    x, y = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= x < W and 0 <= y < H):
        return STATUS_NO_PROJECTION, np.nan
    observed = float(depth_map[y, x])
    if not np.isfinite(observed) or observed <= 0:
        return STATUS_NO_DEPTH, observed
    if observed < z_point_m - depth_tolerance_m:
        return STATUS_OCCLUDED, observed
    if observed > z_point_m + depth_tolerance_m:
        return STATUS_DEPTH_BEHIND, observed
    if object_mask is not None and bool(np.asarray(object_mask)[y, x]):
        return STATUS_IN_OBJECT_MASK, observed
    return STATUS_VISIBLE, observed


# ----------------------------------------------------------------------------- residuals


def huber_from_normalized(e):
    e = np.abs(e)
    return np.where(e <= 1.0, e * e, 2.0 * e - 1.0)


def compute_residuals(pred_uv, gt_uv, valid, sigma_px=None):
    """Per-frame residual ``r = u_track - u_GT`` and robust aggregate.

    Returns dict with residual_xy (T,2, NaN where invalid), residual_px (T,), and summary
    (count, median, p95, max, E_tcp or None when sigma is not configured).
    """
    pred_uv = np.asarray(pred_uv, dtype=np.float64)
    gt_uv = np.asarray(gt_uv, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(pred_uv).all(axis=1) & np.isfinite(gt_uv).all(axis=1)
    residual = np.full_like(pred_uv, np.nan)
    residual[valid] = pred_uv[valid] - gt_uv[valid]
    norm = np.linalg.norm(residual, axis=1)
    values = norm[valid]
    summary = dict(valid_count=int(valid.sum()))
    if len(values):
        summary.update(median_px=float(np.median(values)), p95_px=float(np.percentile(values, 95)),
                       max_px=float(values.max()), status="available")
        summary["E_tcp"] = (float(np.mean(huber_from_normalized(values / float(sigma_px))))
                            if sigma_px else None)
    else:
        summary.update(median_px=None, p95_px=None, max_px=None, E_tcp=None, status="unavailable")
    return dict(residual_xy=residual, residual_px=norm, valid=valid, summary=summary)


def propose_requery_split(residual_px, valid, tolerance_px, persistence=1):
    """First persistent, visibility-valid TCP failure -> last reliable window frame.

    A failure is ``valid & residual > tolerance``.  ``persistence`` consecutive valid failing
    frames (invalid frames neither count nor reset) trigger a split ending one frame before
    the run started.  Returns (end_index or None, reason).
    """
    if tolerance_px is None or not np.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("tcp_requery_error_px must be explicitly configured and positive")
    persistence = max(1, int(persistence))
    residual_px = np.asarray(residual_px, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    run_start, run = None, 0
    for t in range(1, len(residual_px)):
        if not valid[t]:
            continue
        if residual_px[t] > tolerance_px:
            run_start = t if run == 0 else run_start
            run += 1
            if run >= persistence:
                end = run_start - 1
                if end < 1:
                    return None, "failure_at_first_tracked_step"
                return end, f"tcp_residual>{tolerance_px}px_from_window_frame_{run_start}"
        else:
            run_start, run = None, 0
    return None, "no_persistent_failure"


# ----------------------------------------------------------------------------- per-window evaluation


def evaluate_window(traj_maps, vis_maps, window_tcp, depths, masks, *, depth_tolerance_m,
                    visibility_threshold, sigma_px=None):
    """Compare the raw track seeded at the frame-0 GT TCP pixel with GT at every frame.

    ``window_tcp`` holds per-window-frame arrays at tracking resolution: uv (T,2), z (T,),
    usable (T,) and source_frame_index / sequence_index (T,).
    Returns a records dict suitable for :func:`write_window_report`.
    """
    T = len(window_tcp["uv"])
    statuses, observed = [], []
    for t in range(T):
        if not window_tcp["usable"][t]:
            statuses.append(STATUS_ARTIFICIAL)
            observed.append(np.nan)
            continue
        status, depth = tcp_visibility(window_tcp["uv"][t], window_tcp["z"][t], depths[t], masks[t],
                                       depth_tolerance_m)
        statuses.append(status)
        observed.append(depth)
    visible = np.array([s == STATUS_VISIBLE for s in statuses])
    pred_uv = np.full((T, 2), np.nan)
    tracker_vis = np.full(T, np.nan)
    tracker_conf = np.full(T, np.nan)
    if visible[0]:
        pred_uv, tracker_vis, tracker_conf = extract_seeded_track(traj_maps, vis_maps, window_tcp["uv"][0])
        window_status = "available"
    else:
        window_status = f"unavailable_seed_{statuses[0]}"
    tracker_ok = np.isfinite(tracker_vis) & (tracker_vis > visibility_threshold)
    valid_comparison = visible & tracker_ok
    valid_comparison[0] = False  # frame 0 is the seed itself, not a measurement
    result = compute_residuals(pred_uv, window_tcp["uv"], valid_comparison, sigma_px)
    return dict(tcp_pred_uv=pred_uv, tcp_gt_uv=np.asarray(window_tcp["uv"], dtype=np.float64),
                visible=visible, valid_comparison=result["valid"], residual_px=result["residual_px"],
                residual_xy=result["residual_xy"], tracker_visibility=tracker_vis, tracker_confidence=tracker_conf,
                observed_depth_m=np.asarray(observed, dtype=np.float64), tcp_depth_m=np.asarray(window_tcp["z"]),
                visibility_status=statuses, source_frame_index=np.asarray(window_tcp["source_frame_index"]),
                sequence_index=np.asarray(window_tcp["sequence_index"]),
                window_frame_index=np.arange(T), window_status=window_status, summary=result["summary"])


def write_window_report(output_dir, records, camera, extra=None, rgbs=None):
    """CSV + NPZ + JSON (+ optional overlays) under ``tracking/window_XXXXX/tcp_track/``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    T = len(records["window_frame_index"])
    with (output_dir / "tcp_track.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["camera", "window_frame_index", "sequence_index", "source_frame_index",
                         "tcp_gt_u", "tcp_gt_v", "tcp_pred_u", "tcp_pred_v", "visible", "visibility_status",
                         "valid_comparison", "residual_px"])
        for t in range(T):
            writer.writerow([camera, t, int(records["sequence_index"][t]), int(records["source_frame_index"][t]),
                             *records["tcp_gt_uv"][t], *records["tcp_pred_uv"][t], bool(records["visible"][t]),
                             records["visibility_status"][t], bool(records["valid_comparison"][t]),
                             records["residual_px"][t]])
    arrays = {k: np.asarray(v) for k, v in records.items()
              if k not in ("visibility_status", "summary", "window_status")}
    np.savez_compressed(output_dir / "tcp_track.npz", **arrays)
    report = dict(camera=camera, status=records["window_status"], summary=records["summary"],
                  visibility_status_counts={s: records["visibility_status"].count(s)
                                            for s in sorted(set(records["visibility_status"]))},
                  semantics="Raw AllTracker map sampled at the frame-0 projected TCP; robot point only.")
    if extra:
        report.update(extra)
    (output_dir / "tcp_track.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    if rgbs is not None:
        write_overlays(output_dir, records, rgbs)
    return report


def write_overlays(output_dir, records, rgbs):
    import cv2

    overlay_dir = Path(output_dir) / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for t, image in enumerate(rgbs):
        canvas = np.ascontiguousarray(np.asarray(image).copy())
        for uv, color in [(records["tcp_gt_uv"][t], (0, 255, 0)), (records["tcp_pred_uv"][t], (0, 0, 255))]:
            if np.isfinite(uv).all():
                cv2.circle(canvas, (int(round(uv[0])), int(round(uv[1]))), 4, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(overlay_dir / f"{t:03d}.png"), canvas)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def aggregate_reports(tracking_dir):
    """Aggregate per-window ``tcp_track.npz`` into one sequence-level report."""
    tracking_dir = Path(tracking_dir)
    residuals, visible, windows = [], 0, 0
    for npz in sorted(tracking_dir.glob("window_*/tcp_track/tcp_track.npz")):
        with np.load(npz) as data:
            valid = data["valid_comparison"].astype(bool)
            residuals.extend(data["residual_px"][valid].tolist())
            visible += int(data["visible"].sum())
        windows += 1
    values = np.asarray(residuals)
    report = dict(windows=windows, visible_tcp_samples=visible, valid_comparisons=int(len(values)),
                  status="available" if len(values) else ("skipped" if windows else "not_run"))
    if len(values):
        report.update(median_px=float(np.median(values)), p95_px=float(np.percentile(values, 95)),
                      max_px=float(values.max()))
    (tracking_dir / "tcp_track_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
