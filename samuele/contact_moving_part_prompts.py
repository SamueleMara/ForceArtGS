#!/usr/bin/env python3
"""Phase B: per-frame contact-guided moving-part prompts (CPU only).

For each confirmed-contact frame of one camera, select *object-surface* pixels next to
the calibrated tool contact point as positive moving-part prompts, and depth-confirmed
robot/tool pixels as negatives.  The TCP pixel itself is never a positive object prompt.
Object foreground masks are read only; outputs are separate artifacts:

    <out>/contact_guidance/contact_prompts.json
    <out>/contact_guidance/proposed_moving_positive/cam_NNN/frame_XXXXXX.png
    <out>/contact_guidance/prompt_report.json
    <out>/moving_prompts/moving_seed_maps.npz (+ .json)   # per FreeArt local frame, for tracking

where ``<out>`` defaults to ``<freeart-camera-data>/metadata``.  Frames are keyed by
(camera, source_frame_index); the seed artifact remaps them to FreeArt local frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from samuele.tcp_contact_evidence import load_tcp_contact_artifact, pinhole, sha256_file, transform_points  # noqa: E402

STATUS_VALID = "valid"
STATUS_NO_CONTACT = "no_contact"
STATUS_NO_PROJECTION = "no_valid_projection"
STATUS_NO_POSITIVE = "no_valid_positive_prompt"
TOOL_AXIS_SAMPLES = 9  # sampling resolution along the measured contact->EE segment, not a threshold


def unproject(K, xs, ys, depth):
    z = depth
    return np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], axis=-1)


def robot_negative_pixels(K, depth, object_mask, tool_points_cam, depth_margin_m):
    """Pixels where the measured tool is *observed*: projected tool point, not object
    foreground, and observed depth within ``depth_margin_m`` of the tool point depth."""
    H, W = depth.shape
    uv, valid = pinhole(K, tool_points_cam, (W, H))
    negatives = []
    for (u, v), ok, point in zip(uv, valid, tool_points_cam):
        if not ok:
            continue
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < W and 0 <= y < H) or object_mask[y, x]:
            continue
        observed = depth[y, x]
        if observed > 0 and abs(observed - point[2]) <= depth_margin_m:
            negatives.append((x, y))
    return sorted(set(negatives))


def tool_volume_mask(points, p_contact_cam, tool_points_cam, radius_m, start_m):
    """Observed 3D points inside the tool's volume behind its tip (they cannot be object).

    The volume is a cylinder of ``radius_m`` around the measured contact->EE axis, from
    ``start_m`` along the axis (positive: behind the contact, towards the EE; negative: the
    cylinder also covers the uncertain tip region in front of the calibrated contact) to the EE.
    """
    p0 = np.asarray(p_contact_cam, dtype=np.float64)
    end = np.asarray(tool_points_cam, dtype=np.float64).reshape(-1, 3)[-1]
    axis = end - p0
    length = float(np.linalg.norm(axis))
    if length <= 0:
        return np.zeros(points.shape[:-1], dtype=bool)
    axis /= length
    rel = points - p0
    s = rel @ axis
    radial = np.linalg.norm(rel - s[..., None] * axis, axis=-1)
    return (s >= start_m) & (s <= length) & (radial <= radius_m)


def select_contact_prompts(K, depth, object_mask, p_contact_cam, tool_points_cam, *, search_radius_m,
                           depth_margin_m, max_positive, tool_exclusion_radius_m=None, tool_exclusion_start_m=None):
    """Positive object-surface pixels near the contact and confirmed robot negatives for one frame.

    With ``tool_exclusion_radius_m``/``tool_exclusion_start_m`` set, object-mask pixels whose
    3D point lies inside the tool volume are rejected (object masks can include a dark tool).
    """
    if (tool_exclusion_radius_m is None) != (tool_exclusion_start_m is None):
        raise ValueError("Set both tool_exclusion_radius_m and tool_exclusion_start_m, or neither")
    if tool_exclusion_radius_m is not None and not (tool_exclusion_radius_m > 0 and np.isfinite(tool_exclusion_start_m)):
        raise ValueError("tool_exclusion_radius_m must be > 0 and tool_exclusion_start_m finite")
    if not (search_radius_m and search_radius_m > 0 and depth_margin_m and depth_margin_m > 0):
        raise ValueError("search_radius_m and depth_margin_m must be explicitly configured and positive")
    H, W = depth.shape
    object_mask = np.asarray(object_mask, dtype=bool)
    uv, valid = pinhole(K, np.asarray(p_contact_cam)[None], (W, H))
    result = dict(positive_xy=[], negative_robot_xy=[], contact_candidate_3d=None, depth_evidence={})
    if not valid[0]:
        return dict(result, status=STATUS_NO_PROJECTION, reason="contact point behind camera or outside image")
    z_contact = float(p_contact_cam[2])
    u, v = uv[0]
    negatives = robot_negative_pixels(K, depth, object_mask, tool_points_cam, depth_margin_m)
    radius_px = int(np.ceil(max(K[0, 0], K[1, 1]) * search_radius_m / z_contact)) + 1
    x0, x1 = max(0, int(np.floor(u)) - radius_px), min(W, int(np.ceil(u)) + radius_px + 1)
    y0, y1 = max(0, int(np.floor(v)) - radius_px), min(H, int(np.ceil(v)) + radius_px + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = depth[y0:y1, x0:x1]
    inside = object_mask[y0:y1, x0:x1] & (d > 0) & np.isfinite(d)
    points = unproject(K, xs.astype(np.float64), ys.astype(np.float64), d.astype(np.float64))
    distance = np.linalg.norm(points - np.asarray(p_contact_cam)[None, None], axis=-1)
    # Object surface the tool touches lies at/behind the contact depth; nearer pixels are occluders.
    candidate = inside & (distance <= search_radius_m) & (d >= z_contact - depth_margin_m)
    if tool_exclusion_radius_m is not None:
        in_tool = inside & tool_volume_mask(points, p_contact_cam, tool_points_cam, tool_exclusion_radius_m,
                                            tool_exclusion_start_m)
        result["tool_volume"] = dict(object_mask_pixels_inside_tool_volume=int(in_tool.sum()),
                                     rejected_candidates=int((candidate & in_tool).sum()),
                                     radius_m=tool_exclusion_radius_m, start_m=tool_exclusion_start_m)
        candidate &= ~in_tool
    else:
        result["tool_volume"] = dict(status="not_configured")
    negative_set = set(negatives)
    if negative_set:
        for (nx, ny) in negative_set:
            if y0 <= ny < y1 and x0 <= nx < x1:
                candidate[ny - y0, nx - x0] = False
    cx, cy = int(round(u)), int(round(v))
    observed_at_contact = float(depth[cy, cx]) if 0 <= cy < H and 0 <= cx < W else float("nan")
    result["depth_evidence"] = dict(contact_depth_m=z_contact, observed_depth_at_contact_px_m=observed_at_contact,
                                    object_pixels_in_window=int(inside.sum()), candidates=int(candidate.sum()),
                                    search_radius_px=radius_px)
    result["negative_robot_xy"] = [list(p) for p in negatives]
    if not candidate.any():
        occluded = np.isfinite(observed_at_contact) and 0 < observed_at_contact < z_contact - depth_margin_m
        reason = "contact_surface_occluded_by_nearer_surface" if occluded else "no_object_depth_within_search_radius"
        return dict(result, status=STATUS_NO_POSITIVE, reason=reason)
    order = np.argsort(distance[candidate])[:int(max_positive)]
    cand_x, cand_y = xs[candidate][order], ys[candidate][order]
    positives = [[int(x), int(y)] for x, y in zip(cand_x, cand_y)]
    selected = points[candidate][order]
    result.update(positive_xy=positives, contact_candidate_3d=selected.mean(axis=0).tolist(),
                  status=STATUS_VALID, reason=None)
    result["depth_evidence"]["nearest_object_distance_m"] = float(distance[candidate][order][0])
    return result


def _load_frame(camera_dir, stem):
    import cv2

    with np.load(camera_dir / "depth" / f"{stem}.npz") as data:
        depth = data["depth"].astype(np.float64)
    mask = cv2.imread(str(camera_dir / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != depth.shape:
        raise ValueError(f"Missing or misaligned object mask for frame {stem} in {camera_dir}")
    return depth, mask > 127


def tool_axis_points_cam(arrays, index):
    """Measured contact -> EE segment in camera coordinates (empty if EE unavailable)."""
    p_contact = arrays["p_contact_cam_m"][index]
    T_ee = arrays["T_base_ee"][index]
    if not np.isfinite(T_ee).all():
        return p_contact[None]
    T_camera_base = np.linalg.inv(arrays["T_base_camera"])
    p_ee = transform_points(T_camera_base, T_ee[:3, 3])
    s = np.linspace(0.0, 1.0, TOOL_AXIS_SAMPLES)[:, None]
    return p_contact[None] * (1 - s) + p_ee[None] * s


def generate(camera_dir, tcp_artifact, *, search_radius_m, depth_margin_m, max_positive,
             tool_exclusion_radius_m=None, tool_exclusion_start_m=None):
    camera_dir = Path(camera_dir)
    arrays, manifest = load_tcp_contact_artifact(tcp_artifact)
    conversion = json.loads((camera_dir / "metadata/freeart_conversion.json").read_text())
    if manifest.get("freeart_conversion_sha256") and \
            manifest["freeart_conversion_sha256"] != sha256_file(camera_dir / "metadata/freeart_conversion.json"):
        raise ValueError("TCP artifact belongs to a different FreeArt conversion")
    camera = conversion["selected_camera"]
    K = arrays["K_fullres"]
    frames = []
    for local, row in enumerate(conversion["frame_map"]):
        stem = row["output_stem"]
        entry = dict(camera=camera, local_frame_index=local, output_stem=stem, source_stem=row["source_stem"],
                     source_frame_index=int(arrays["source_frame_index"][local]),
                     contact_active=bool(arrays["contact_active"][local]))
        if not entry["contact_active"]:
            frames.append(dict(entry, status=STATUS_NO_CONTACT, reason="contact not confirmed",
                               positive_xy=[], negative_robot_xy=[]))
            continue
        depth, mask = _load_frame(camera_dir, stem)
        selection = select_contact_prompts(K, depth, mask, arrays["p_contact_cam_m"][local],
                                           tool_axis_points_cam(arrays, local), search_radius_m=search_radius_m,
                                           depth_margin_m=depth_margin_m, max_positive=max_positive,
                                           tool_exclusion_radius_m=tool_exclusion_radius_m,
                                           tool_exclusion_start_m=tool_exclusion_start_m)
        frames.append(dict(entry, **selection))
    statuses = [f["status"] for f in frames]
    report = dict(camera=camera, frames=len(frames), status_counts={s: statuses.count(s) for s in sorted(set(statuses))},
                  valid_positive_frames=statuses.count(STATUS_VALID),
                  robot_negative_frames=sum(bool(f["negative_robot_xy"]) for f in frames),
                  # Positives come from the object mask; if that mask contains tool pixels, only the
                  # tool-volume test can reject them (count of mask pixels found inside the tool).
                  tool_volume_exclusion=tool_exclusion_radius_m is not None,
                  object_mask_pixels_inside_tool_volume=(
                      sum(f.get("tool_volume", {}).get("object_mask_pixels_inside_tool_volume", 0) for f in frames)
                      if tool_exclusion_radius_m is not None else None),
                  positives_rejected_inside_tool_volume=(
                      sum(f.get("tool_volume", {}).get("rejected_candidates", 0) for f in frames)
                      if tool_exclusion_radius_m is not None else None),
                  contact_point_calibrated=manifest.get("tool_geometry", {}).get("contact_point_calibrated"))
    prompts = dict(schema_version=1, coordinate_system="image_pixels_xy", resolution="full-res rectified",
                   image_size=arrays["image_size_wh"].tolist(), index=["camera", "source_frame_index"],
                   parameters=dict(search_radius_m=search_radius_m, depth_margin_m=depth_margin_m,
                                   max_positive=max_positive, tool_exclusion_radius_m=tool_exclusion_radius_m,
                                   tool_exclusion_start_m=tool_exclusion_start_m),
                   provenance=dict(tcp_artifact=str(tcp_artifact), freeart_conversion_sha256=manifest.get(
                       "freeart_conversion_sha256"), tcp_artifact_sha256=manifest.get("artifact_sha256"),
                                   object_masks="existing FreeArt masks (read only)"),
                   semantics="positive_xy: visible object-surface pixels adjacent to the calibrated contact point; "
                             "negative_robot_xy: depth-confirmed tool pixels; TCP pixel is never positive.",
                   frames=frames)
    return prompts, report


def write_outputs(prompts, report, out_root, camera_dir, seed_source="prompts", sam2_dir=None):
    import cv2

    out_root = Path(out_root)
    guidance = out_root / "contact_guidance"
    camera = report["camera"]
    width, height = prompts["image_size"]
    positive_dir = guidance / "proposed_moving_positive" / camera
    positive_dir.mkdir(parents=True, exist_ok=True)
    points_all, offsets, valid, sources = [], [0], [], []
    for frame in prompts["frames"]:
        source_png = f"{frame['source_stem']}.png"
        seed_points = np.zeros((0, 2), dtype=np.int32)
        if frame["status"] == STATUS_VALID:
            canvas = np.zeros((height, width), dtype=np.uint8)
            for x, y in frame["positive_xy"]:
                canvas[y, x] = 255
            cv2.imwrite(str(positive_dir / source_png), canvas)
            seed_points = np.asarray(frame["positive_xy"], dtype=np.int32).reshape(-1, 2)
            if seed_source == "sam2":
                proposal = Path(sam2_dir) / camera / source_png
                if proposal.is_file():
                    mask = cv2.imread(str(proposal), cv2.IMREAD_GRAYSCALE) > 127
                    ys, xs = np.nonzero(mask)
                    seed_points = np.stack([xs, ys], axis=1).astype(np.int32)
                else:
                    seed_points = np.zeros((0, 2), dtype=np.int32)
        points_all.append(seed_points)
        offsets.append(offsets[-1] + len(seed_points))
        valid.append(len(seed_points) > 0)
        sources.append(frame["source_frame_index"])
    (guidance / "contact_prompts.json").write_text(json.dumps(prompts, indent=1) + "\n")
    (guidance / "prompt_report.json").write_text(json.dumps(dict(report, status="passed"), indent=2) + "\n")
    seeds_dir = out_root / "moving_prompts"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    arrays, _ = load_tcp_contact_artifact(prompts["provenance"]["tcp_artifact"])
    np.savez_compressed(seeds_dir / "moving_seed_maps.npz",
                        points_xy=np.concatenate(points_all) if points_all else np.zeros((0, 2), np.int32),
                        frame_offsets=np.asarray(offsets, dtype=np.int64), valid=np.asarray(valid, dtype=bool),
                        source_frame_index=np.asarray(sources, dtype=np.int64), K_fullres=arrays["K_fullres"],
                        image_size_wh=np.asarray([width, height], dtype=np.int64))
    seed_manifest = dict(seed_source=seed_source, camera=camera, frames=len(valid), frames_with_seeds=int(sum(valid)),
                         freeart_conversion_sha256=prompts["provenance"]["freeart_conversion_sha256"],
                         semantics="Positive moving-object seed pixels per FreeArt local frame (full resolution). "
                                   "Robot pixels are never seeds. Branch identity is resolved from motion in tracking.")
    (seeds_dir / "moving_seed_maps.json").write_text(json.dumps(seed_manifest, indent=2) + "\n")
    return seed_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeart-camera-data", type=Path, required=True)
    parser.add_argument("--tcp-contact", type=Path, help="tcp_contact artifact dir (default <camera>/metadata/tcp_contact)")
    parser.add_argument("--search-radius-m", type=float, required=True, help="3D radius around the contact point")
    parser.add_argument("--depth-margin-m", type=float, required=True,
                        help="Depth agreement tolerance for tool pixels and the occluder test")
    parser.add_argument("--max-positive", type=int, default=5, help="Max positive prompt pixels per frame (count)")
    parser.add_argument("--tool-exclusion-radius-m", type=float,
                        help="Reject object-mask pixels within this radius of the tool axis (tool radius + uncertainty)")
    parser.add_argument("--tool-exclusion-start-m", type=float,
                        help="Tool-volume exclusion starts this far behind the contact point (negative: in front)")
    parser.add_argument("--seed-source", choices=["prompts", "sam2"], default="prompts")
    parser.add_argument("--sam2-proposals", type=Path, help="sam2_moving_proposals dir (for --seed-source sam2)")
    parser.add_argument("--output", type=Path, help="Output metadata root (default <camera>/metadata)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    camera_dir = args.freeart_camera_data.resolve()
    tcp = args.tcp_contact or camera_dir / "metadata/tcp_contact"
    prompts, report = generate(camera_dir, tcp, search_radius_m=args.search_radius_m,
                               depth_margin_m=args.depth_margin_m, max_positive=args.max_positive,
                               tool_exclusion_radius_m=args.tool_exclusion_radius_m,
                               tool_exclusion_start_m=args.tool_exclusion_start_m)
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return 0
    if args.seed_source == "sam2" and args.sam2_proposals is None:
        parser.error("--seed-source sam2 requires --sam2-proposals")
    manifest = write_outputs(prompts, report, args.output or camera_dir / "metadata", camera_dir,
                             args.seed_source, args.sam2_proposals)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
