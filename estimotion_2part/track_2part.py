import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")
import sys
import argparse
from pathlib import Path
import torch    
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from estimotion_2part.solve_transformation import solve_window_transformations, joint_optimize_segmentation_and_transforms
from estimotion_2part.process_part_segment import initialize_part_segmentation
import json
from termcolor import cprint
from estimotion_2part.process_part_segment import pass_segmentation_map
from utils.utils import load_config, ImageLoader
from utils.pipeline_config import (
    apply_section_to_args,
    load_pipeline_config,
    resolve_pipeline_paths,
    resolve_tracking,
    set_arg_defaults,
    set_path_defaults,
)
from tracking.tracker_toolkit import Alltracker, Tracker
from estimotion_2part.process_part_segment import convert_trajs_to_segmentation_map, _resize_feature_to_hw_torch
from estimotion_2part.utils.vis import visualize_seg_map_list
from estimotion_2part.utils.eval import check_segmentation_consistency
from estimotion_2part.tracking_prior_hooks import WindowPriors, run_phase_a
from samuele.debug_audit import trace as _trace


def _resize_feature_to_rgb(first_frame, device):
    rgb = first_frame["rgbs"]
    h, w = rgb.shape[:2]
    return _resize_feature_to_hw_torch(first_frame["features"], h, w, device)


def _blend_segmap_inplace(first_frame_seg_map_torch, filled_frame_seg_map, valid_pixel_masks, device):
    valid_pixel_masks_torch = valid_pixel_masks.to(device=device, dtype=torch.bool)
    filled_frame_seg_map_torch = filled_frame_seg_map.to(
        device=device, dtype=first_frame_seg_map_torch.dtype
    )

    first_frame_seg_map_torch[valid_pixel_masks_torch] = (
        first_frame_seg_map_torch[valid_pixel_masks_torch] + filled_frame_seg_map_torch[valid_pixel_masks_torch]
    ) * 0.5


def merge_trajectories(args, pair_trajectories, pair_point_seg_weights, pair_point_indices, t1_map, t2_map, first_frame, first_frame_seg_map=None, factor=1.0, global_progress: float = 1.0,
                       prior_builder=None):
    """
    Merge trajectories from multiple windows with joint optimization of segmentation weights.
    
    Args:
        pair_trajectories: List of trajectory arrays with segmentation, each with shape (N_i, 2, 3)
        pair_point_seg_weights: List of segmentation weights for each trajectory, shape (N_i, 1)
        pair_point_indices: List of point indices tuples, each containing (y_indices, x_indices)
        transform_matrices_t1: Transform matrices T1 for each window pair
        transform_matrices_t2: Transform matrices T2 for each window pair
        args: Arguments containing optimization parameters
    
    Returns:
        merged_trajectories: numpy array of merged trajectories with shape (N, total_T, 4)
        merged_point_indices: numpy array of point indices with shape (N, 2)
        optimized_transform_t1: Optimized T1 transformation matrix
        optimized_transform_t2: Optimized T2 transformation matrix
    """
    cam_K = first_frame["cam_K"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.inference_mode():
        feature_tensor = _resize_feature_to_rgb(first_frame, device)
        feature_map = feature_tensor.permute(1, 2, 0).contiguous()

        point_trajectory_dict = {}
        for pair_idx, (trajectories, point_seg_weights, point_indices) in enumerate(
            zip(pair_trajectories, pair_point_seg_weights, pair_point_indices)
        ):
            trajectories_tensor = trajectories
            point_seg_weights_tensor = point_seg_weights.reshape(-1)

            y_indices, x_indices = point_indices
            point_features = feature_map[y_indices, x_indices]
            y_list = y_indices.reshape(-1).tolist()
            x_list = x_indices.reshape(-1).tolist()

            for trajectory_idx, (y, x) in enumerate(zip(y_list, x_list)):
                point_key = (int(y), int(x))
                if point_key not in point_trajectory_dict:
                    point_trajectory_dict[point_key] = {
                        'trajectories': [trajectories_tensor[trajectory_idx]],
                        'pair_ids': [pair_idx],
                        'segmentation_weights': [point_seg_weights_tensor[trajectory_idx]],
                        'feature': point_features[trajectory_idx],
                    }
                else:
                    point_trajectory_dict[point_key]['trajectories'].append(trajectories_tensor[trajectory_idx])
                    point_trajectory_dict[point_key]['pair_ids'].append(pair_idx)
                    point_trajectory_dict[point_key]['segmentation_weights'].append(point_seg_weights_tensor[trajectory_idx])
    
    point_keys = list(point_trajectory_dict.keys())

    first_frame_weights = None
    if first_frame_seg_map is not None:
        if point_keys:
            point_key_tensor = torch.tensor(point_keys, dtype=torch.long, device=first_frame_seg_map.device)
            first_frame_weights = first_frame_seg_map[point_key_tensor[:, 0], point_key_tensor[:, 1]]
        else:
            first_frame_weights = torch.empty((0,), dtype=first_frame_seg_map.dtype, device=first_frame_seg_map.device)

    seed_term, contact_term = (None, None) if prior_builder is None else prior_builder(point_keys, device)

    # Joint optimization of all point segmentation weights and transformation matrices
    optimized_weights, optimized_transform_t1, optimized_transform_t2 = joint_optimize_segmentation_and_transforms(
        args,
        point_trajectory_dict,
        t1_map,
        t2_map,
        device,
        cam_K,
        first_frame_weights,
        factor,
        global_progress=global_progress,
        seed_term=seed_term,
        contact_term=contact_term,
    )
    if prior_builder is not None:
        prior_builder.joint_infos = dict(seed=None if seed_term is None else seed_term.info,
                                         contact=None if contact_term is None else contact_term.info)
    
    # Update point weights
    for point_idx, point_key in enumerate(point_keys):
        # Use optimized average weight
        point_trajectory_dict[point_key]['segmentation_weights'] = (point_trajectory_dict[point_key]['segmentation_weights'][-1] + optimized_weights[point_idx]) / 2.0

    return point_trajectory_dict, optimized_transform_t1, optimized_transform_t2


def process_single_window(args, tracker: Tracker, window_image_loader, max_window_size, window_index, current_frame, segmentation_maps, valid_seg_masks, total_frames,
                          tcp_context=None):
    window_output_dir = os.path.join(args.output_dir, f"window_{window_index:05d}")
    os.makedirs(window_output_dir, exist_ok=True)
    first_frame = dict(window_image_loader[0])

    with torch.inference_mode():
        trajectory_maps, visibility_maps = tracker.get_track_maps(window_image_loader.data)
    duplicated_last = len(window_image_loader) > max_window_size
    if tcp_context is not None and tcp_context.phase_a:
        # Phase A sees the RAW dense maps before object masking; the TCP never enters object tracks.
        window_image_loader, max_window_size, trajectory_maps, visibility_maps = run_phase_a(
            args, tracker, tcp_context, window_image_loader, max_window_size, current_frame, duplicated_last,
            trajectory_maps, visibility_maps, window_output_dir)
        duplicated_last = len(window_image_loader) > max_window_size
    priors = None
    if tcp_context is not None and (tcp_context.phase_b or tcp_context.phase_c):
        priors = WindowPriors(args, tcp_context, current_frame, max_window_size, duplicated_last)

    t1_map = np.eye(4)[None, None, :, :].repeat(max_window_size, axis=0).repeat(max_window_size, axis=1)
    t2_map = t1_map.copy()
    with torch.inference_mode():
        coords_4d = tracker.get_4d_traj_maps(window_image_loader.data, trajectory_maps, visibility_maps)
    _trace_window = dict(window_id=int(window_index), window_start_seq=int(current_frame),
                         window_seq_frames=[int(current_frame + i) for i in range(int(coords_4d.shape[1]))],
                         duplicated_last=bool(duplicated_last)) if _trace.enabled() else None
    if _trace_window is not None:
        # Frozen 2D tracks of every frame-0 object pixel (eroded mask), for offline depth-lifting
        # counterfactuals; dense flow of background pixels is not stored.
        with torch.inference_mode():
            m0 = torch.from_numpy(np.asarray(window_image_loader.data["masks_eroded"][0])).to(trajectory_maps.device) > 0
            ys, xs = torch.where(m0)
            _trace.event("window_2d_tracks", local_file="estimotion_2part/track_2part.py", function="process_single_window",
                         coordinate_frame="image_pixels (u right, v down) of the resized tracking images", units="px",
                         value=dict(_trace_window, cam_K=np.asarray(window_image_loader.data["cam_K"]).tolist(),
                                    image_hw=[int(trajectory_maps.shape[-2]), int(trajectory_maps.shape[-1])]),
                         arrays=dict(y0=ys, x0=xs, uv=trajectory_maps[0][:, :, ys, xs].permute(2, 0, 1),
                                     visconf=visibility_maps[0][:, :, ys, xs].permute(2, 0, 1),
                                     z_bilinear=coords_4d[0][:, 2, ys, xs].permute(1, 0),
                                     in_mask=coords_4d[0][:, 4, ys, xs].permute(1, 0)))

    with torch.no_grad():
        if first_frame.get("features", None) is not None:
            first_frame["features"] = _resize_feature_to_rgb(first_frame, coords_4d.device)

    T_effective = int(coords_4d.shape[1])

    pair_trajectories = []
    pair_point_seg_weights = []
    pair_point_indices = []

    filter_ratio_threshold = getattr(args, 'filter_ratio_threshold', 0.15)
    segmentation_maps_torch = segmentation_maps.to(device=coords_4d.device, dtype=torch.float32)
    valid_seg_masks_torch = valid_seg_masks.to(device=coords_4d.device, dtype=torch.float32)
    first_frame_seg_map_torch = segmentation_maps_torch.clamp(0.001, 0.999)

    max_valid_window_size = min(max_window_size, T_effective)
    factor = 1.0
    for window_size in range(2, max_valid_window_size + 1):
        save_dir = os.path.join(window_output_dir, f"{window_size-1}")
        last_idx = min(window_size - 1, max(0, T_effective - 1))
        coords_subset = coords_4d[:, [0, last_idx], :, :, :]
        _pair_ctx = _trace.context(**(dict(_trace_window or {}, pair_k=int(last_idx), seq_frame_a=int(current_frame),
                                           seq_frame_b=int(current_frame + last_idx))))
        _pair_ctx.__enter__()
        trajectories, point_indices, _, filter_ratio, filtered_count = tracker.get_visible_trajectories(args, coords_subset)

        if filter_ratio > filter_ratio_threshold:
            cprint(f"    High filter ratio detected: {filter_ratio:.4f} > {filter_ratio_threshold:.3f}", "red")
        elif filtered_count < 4000:
            cprint(f"    Low trajectory count detected: {filtered_count} < 4000 ", "red")

        y_indices, x_indices = point_indices
        init_point_seg_weights = first_frame_seg_map_torch[y_indices, x_indices].unsqueeze(1)

        global_frame_idx = current_frame + last_idx
        global_progress = float(global_frame_idx + 1) / max(1, total_frames)
        seed_term, contact_term = (None, None) if priors is None else priors.pair_terms(point_indices, last_idx, coords_4d.device)
        transform_t1, transform_t2, final_point_seg_weights, _ = solve_window_transformations(
            args,
            trajectories,
            init_point_seg_weights,
            point_indices,
            save_dir,
            first_frame,
            valid_seg_masks_torch,
            factor,
            global_progress=global_progress,
            seed_term=seed_term,
            contact_term=contact_term,
        )

        solved_t2 = transform_t2
        _solved = (transform_t1, transform_t2, final_point_seg_weights) if _trace_window is not None else None
        final_point_seg_weights, transform_t1, transform_t2 = check_segmentation_consistency(
            final_point_seg_weights, point_indices, segmentation_maps_torch, valid_seg_masks_torch, transform_t1, transform_t2
        )
        if _solved is not None:
            _trace.event("pair_solution", local_file="estimotion_2part/track_2part.py", function="process_single_window",
                         coordinate_frame="camera_optical",
                         transform_direction="maps frame seq_frame_a points to seq_frame_b points",
                         value=dict(labels_swapped=bool(transform_t1 is solved_t2),
                                    T1_solved=_trace.to_numpy(_solved[0]).tolist(), T2_solved=_trace.to_numpy(_solved[1]).tolist(),
                                    T1=_trace.to_numpy(transform_t1).tolist(), T2=_trace.to_numpy(transform_t2).tolist(),
                                    T1_rigidity=_trace.matrix_rigidity(transform_t1), T2_rigidity=_trace.matrix_rigidity(transform_t2),
                                    num_points=int(trajectories.shape[0])),
                         arrays=dict(y=point_indices[0], x=point_indices[1], xyz=trajectories,
                                     init_weights=init_point_seg_weights.reshape(-1),
                                     solved_weights=_solved[2].reshape(-1), final_weights=final_point_seg_weights.reshape(-1)))
        _pair_ctx.__exit__(None, None, None)
        if priors is not None:
            if transform_t1 is solved_t2:  # labels reversed: keep the physical moving branch
                for term in (seed_term, contact_term):
                    if term is not None:
                        term.swap_labels()
            priors.record_pair()

        pair_trajectories.append(trajectories)
        pair_point_seg_weights.append(final_point_seg_weights.reshape(-1))
        pair_point_indices.append(point_indices)
        t1_map[0, window_size - 1] = transform_t1
        t2_map[0, window_size - 1] = transform_t2

        cprint(f"  Generating dense segmentation maps for all frames using {args.seg_fill_method} method", "cyan")
        filled_frame_seg_map, valid_pixel_masks = convert_trajs_to_segmentation_map(
            point_seg_weight=final_point_seg_weights,
            point_indices=point_indices,
            first_image_dict=first_frame,
            fill_method=args.seg_fill_method,
        )
        _blend_segmap_inplace(first_frame_seg_map_torch, filled_frame_seg_map, valid_pixel_masks, coords_4d.device)

    actual_window_size = len(pair_trajectories) + 1
    try_window_end = current_frame + len(pair_trajectories)
    window_frame_indices = list(range(current_frame, try_window_end + 1))

    cprint(f"  Saving window size {actual_window_size} (frames {current_frame} - {try_window_end})", "blue")

    window_info = {
        "window_idx": window_index,
        "frame_indices": window_frame_indices,
        "actual_window_size": actual_window_size,
    }
    window_info_path = os.path.join(window_output_dir, "window_info.json")
    with open(window_info_path, "w") as f:
        json.dump(window_info, f, indent=4)

    window_global_progress = float(try_window_end + 1) / max(1, total_frames)
    point_trajectory_dict, optimized_t1, optimized_t2 = merge_trajectories(
        args,
        pair_trajectories,
        pair_point_seg_weights,
        pair_point_indices,
        t1_map,
        t2_map,
        first_frame,
        first_frame_seg_map=first_frame_seg_map_torch,
        factor=factor,
        global_progress=window_global_progress,
        prior_builder=priors,
    )
    if priors is not None:
        priors.write(window_output_dir)

    cprint(f"  Merged {len(point_trajectory_dict)} consistent trajectories across {len(pair_trajectories) + 1} windows", "blue")

    np.savez(os.path.join(window_output_dir, 'optimized_transformations.npz'), T1=optimized_t1, T2=optimized_t2)
    if _trace_window is not None:
        _trace.event("window_merged", local_file="estimotion_2part/track_2part.py", function="process_single_window",
                     coordinate_frame="camera_optical",
                     transform_direction="T[k-1] maps window frame 0 points to window frame k points (k = 1..)",
                     value=dict(_trace_window, frame_indices=window_frame_indices,
                                T1_rigidity=[_trace.matrix_rigidity(m) for m in optimized_t1],
                                T2_rigidity=[_trace.matrix_rigidity(m) for m in optimized_t2]),
                     arrays=dict(T1=optimized_t1, T2=optimized_t2))

    seg_map_list_t, trajectory_mask_list_t, next_seg_map, next_valid_seg_mask = pass_segmentation_map(
        args,
        trajectory_maps,
        point_trajectory_dict,
        window_image_loader.data,
        actual_window_size,
    )
    seg_map_list = [seg_map_t.detach().cpu().numpy() for seg_map_t in seg_map_list_t]
    trajectory_mask_list = [mask_t.detach().cpu().numpy().astype(bool) for mask_t in trajectory_mask_list_t]
    visualize_seg_map_list(
        seg_map_list,
        trajectory_mask_list,
        rgb_list=window_image_loader.data["rgbs"],
        mask_list=window_image_loader.data["masks"],
        save_dir=window_output_dir,
        save_visualizations=getattr(args, "debug", False),
    )

    return len(pair_trajectories) + 1, next_seg_map, next_valid_seg_mask


def estimate_global_geometry_stats(image_loader: ImageLoader):
    first_frame = image_loader[0]
    depth0 = first_frame["depths"]
    K = first_frame["cam_K"]
    mask0 = first_frame["masks"]
    mask_valid = (mask0 > 0) & np.isfinite(depth0) & (depth0 > 1e-6)
    ys, xs = np.where(mask_valid)
    z = depth0[ys, xs].astype(np.float32)
    u = xs.astype(np.float32)
    v = ys.astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    X = (u - cx) / max(fx, 1e-6) * z
    Y = (v - cy) / max(fy, 1e-6) * z
    pts = np.stack([X, Y, z], axis=1)
    pts_min = np.min(pts, axis=0)
    pts_max = np.max(pts, axis=0)
    scale_obj = float(np.max(pts_max - pts_min))
    cprint(f"Global scale object: {scale_obj:.4f}", "green")

    height, width = first_frame["rgbs"].shape[:2]
    projection_normalizer = float(max(height, width))
    cprint(f"Global projection normalizer: {projection_normalizer:.4f}", "green")
    return scale_obj, projection_normalizer


def slide_window_tracking(args, cfg, tracker: Tracker, image_loader: ImageLoader):
    config_log_path = os.path.join(args.output_dir, "config.yaml")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(config_log_path, "w") as f:
        yaml.dump({"args": vars(args), "tracker_config": dict(cfg) if hasattr(cfg, "keys") else cfg}, f)

    max_resolution = max(image_loader.data["rgbs"][0].shape[0], image_loader.data["rgbs"][0].shape[1]) // args.downscale_factor
    image_loader, _, _ = image_loader.resize(max_resolution)

    # Optional TCP/contact priors (Phases A/B/C).  Loaded once, aligned to the pingpong sequence.
    tcp_context = None
    if any(getattr(args, flag, False) for flag in ("tcp_supervision_enabled", "moving_prompt_enabled",
                                                   "contact_surface_enabled", "contact_multiview_refine_enabled")):
        from estimotion_2part.prior_context import TrackingPriorContext
        num_original = len([f for f in os.listdir(os.path.join(args.input_dir, "images_ori"))
                            if f.endswith((".png", ".jpg"))])
        tcp_context = TrackingPriorContext(args, image_loader, args.input_dir, num_original)
        tcp_context.write_report(args.output_dir)
        cprint(f"Tracking priors enabled: A={tcp_context.phase_a} B={tcp_context.phase_b} C={tcp_context.phase_c}", "magenta")

    cprint(f"Resized image shape: {image_loader.rgbs[0].shape}", "green")
    scale_obj, projection_normalizer = estimate_global_geometry_stats(image_loader)
    args.scale_obj = scale_obj
    args.projection_normalizer = projection_normalizer

    window_index = 0
    current_frame = 0
    total_frames = len(image_loader)
    
    cprint(f"Starting dynamic window tracking with max_window_size={args.window_size}", "cyan")
    cprint(f"Quality thresholds: min_inlier_ratio={args.min_inlier_ratio}, min_trajectory_count={args.min_trajectory_count}", "cyan")
    segmentation_maps = None
    valid_seg_mask = None

    if segmentation_maps is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seed_map0 = tcp_context.seed_map(0) if tcp_context is not None and tcp_context.phase_b else None
        seed_init_report = {}
        segmentation_maps_np = initialize_part_segmentation(image_loader[0], method=args.init_method,
                                                            seed_map=seed_map0, seed_report=seed_init_report)
        if tcp_context is not None and tcp_context.phase_b:
            tcp_context.report["seed_initialisation"] = seed_init_report or dict(status="skipped", reason="no_seed_at_frame_0")
            tcp_context.write_report(args.output_dir)
        segmentation_maps = torch.from_numpy(segmentation_maps_np).to(device=device, dtype=torch.float32)
        valid_seg_mask = (segmentation_maps > 0).to(dtype=torch.float32)

    while current_frame < total_frames - 1:

        max_window_end = min(current_frame + args.window_size - 1, total_frames - 1)
        max_window_size = max_window_end - current_frame + 1

        print("\n")
        cprint(f"--- Processing Window {window_index} (frames {current_frame}-{max_window_end}, max_size={max_window_size}) ---", "cyan")

        window_image_loader = image_loader[current_frame:max_window_end + 1]
        if len(window_image_loader) == 2:
            window_image_loader = window_image_loader.duplicate_last_frame()

        len_window, next_seg_map, next_valid_seg_mask = process_single_window(args, tracker, window_image_loader, max_window_size, window_index, current_frame, segmentation_maps, valid_seg_mask, total_frames,
                                                                              tcp_context=tcp_context)

        segmentation_maps = next_seg_map
        valid_seg_mask = next_valid_seg_mask
        window_index += 1
        if len_window <= 1:
            current_frame = max_window_end
        else:
            current_frame += len_window - 1

    if tcp_context is not None and tcp_context.phase_a:
        from tracking.tcp_track_supervision import aggregate_reports
        cprint(f"Phase A TCP summary: {aggregate_reports(args.output_dir)}", "magenta")


if __name__ == "__main__":
    import os as _ja_os
    if _ja_os.environ.get("FORCEARTGS_JOINT_AWARE_TRIPWIRE") == "1":
        raise RuntimeError("legacy stage estimotion_2part/track_2part.py invoked inside a joint-aware run")
    parser = argparse.ArgumentParser(description="Dynamic Tracking with Transformation Solving")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline config YAML")
    parser.add_argument("--object_name", type=str, default=None, help="Object name override used with --config")
    parser.add_argument("--input_dir", type=str, default=None, help="input dir")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir")
    parser.add_argument('-c', "--tracker_config", type=str, default=None, help="path to alltracker config file")
    parser.add_argument('--include_turn_frame', action='store_true', help='Include turning frame when appending reversed (duplicates last frame)')
    args = parser.parse_args()
    set_arg_defaults(args, resolve_tracking({}))

    if args.config:
        config = load_pipeline_config(args.config, object_name_override=args.object_name)
        runtime_paths = resolve_pipeline_paths(config, object_name_override=args.object_name)
        tracking_cfg = resolve_tracking(config)
        set_path_defaults(
            args,
            {
                "input_dir": runtime_paths["dataset_dir"],
                "output_dir": runtime_paths["tracking_dir"],
            },
        )
        apply_section_to_args(args, tracking_cfg, preserve_existing={"tracker_config"})

    if args.input_dir is None or args.output_dir is None:
        parser.error("--input_dir and --output_dir are required when --config is not provided")
    
    cprint("="*60, "cyan")
    cprint("Starting Dynamic 3D Tracking with Transformation Solving", "green")
    
    # Load data
    image_loader = ImageLoader(args.input_dir, append_reversed=args.pingpong, exclude_boundary=not args.include_turn_frame)

    tracker_config = load_config(args.tracker_config)
    cprint("Initializing alltracker models...", "cyan")
    tracker = Alltracker(tracker_config)

    # Run tracking
    slide_window_tracking(args, tracker_config, tracker, image_loader)
