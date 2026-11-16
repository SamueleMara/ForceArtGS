#!/usr/bin/env python3
"""Dump the 3D object tracks the two-part tracker sees, for diagnosis (read-only on the dataset).

Re-runs exactly the tracker front end used by ``estimotion_2part/track_2part.py``
(``Alltracker.get_track_maps`` -> ``get_4d_traj_maps`` -> ``get_visible_trajectories``) on the
forward windows of a prepared FreeArt dataset, with the same tracking configuration, and writes
for each window and each (0, k) pair the trajectories *before* and *after* the tracker's
displacement pre-filter.  Nothing is optimised and no dataset file is modified.

    python samuele/dump_window_tracks.py --config <job config.yaml> --output <dir> [--windows 0 7 14 21]

Output: ``<output>/tracks_w<start>.npz`` with, per pair k (window frame k = 1..W-1):
    pre_xyz_k  (N,2,3)  camera-frame xyz at window frames 0 and k, visible+in-mask in both
    pre_yx_k   (N,2)    pixel (row, col) at frame 0 at tracking resolution
    keep_k     (N,)     True if the pipeline's pre-filter (configured track_filter_mode) keeps it
    keep_global_k / keep_part_aware_k   the same for each filter mode, on identical candidates
    weight_part_aware_k  1 = explained by one rigid motion, 0.5 = ambiguous, 0 = rejected
and ``K`` (tracking-resolution intrinsics), ``scale`` (tracking / full resolution), ``frames``
(clip frame indices of the window).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "estimotion_2part"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=None,
                        help="window start frames (forward, original clip indices); default: every window_size-1")
    args_cli = parser.parse_args(argv)

    from utils.pipeline_config import (apply_section_to_args, load_pipeline_config, resolve_pipeline_paths,
                                       resolve_tracking, set_arg_defaults, set_path_defaults)
    from utils.utils import ImageLoader, load_config
    from tracking.tracker_toolkit import Alltracker

    args = argparse.Namespace(config=args_cli.config, object_name=None, input_dir=None, output_dir=None,
                              tracker_config=None, include_turn_frame=False)
    set_arg_defaults(args, resolve_tracking({}))
    config = load_pipeline_config(args.config)
    paths = resolve_pipeline_paths(config)
    set_path_defaults(args, {"input_dir": paths["dataset_dir"], "output_dir": paths["tracking_dir"]})
    apply_section_to_args(args, resolve_tracking(config), preserve_existing={"tracker_config"})

    # Forward frames only: tracks are a property of the frames, ping-pong replays add no new observations.
    loader = ImageLoader(args.input_dir, append_reversed=False, exclude_boundary=True)
    full_h = loader.data["rgbs"][0].shape[0]
    max_res = max(loader.data["rgbs"][0].shape[:2]) // args.downscale_factor
    loader, _, _ = loader.resize(max_res)
    scale = loader.data["rgbs"][0].shape[0] / full_h
    tracker = Alltracker(load_config(args.tracker_config))
    n = len(loader)
    starts = args_cli.windows or list(range(0, n - 1, args.window_size - 1))
    args_cli.output.mkdir(parents=True, exist_ok=True)
    for s in starts:
        e = min(s + args.window_size - 1, n - 1)
        window = loader[s:e + 1]
        with torch.inference_mode():
            traj, vis = tracker.get_track_maps(window.data)
            coords = tracker.get_4d_traj_maps(window.data, traj, vis)
        out = dict(K=np.asarray(window.data["cam_K"][0] if isinstance(window.data["cam_K"], list) else window[0]["cam_K"]),
                   scale=scale, frames=np.arange(s, e + 1))
        T = coords.shape[1]
        for k in range(1, T):
            sub = coords[:, [0, k]]
            vis_k = (sub[0, :, 3] > args.visibility_threshold) & (sub[0, :, 4] > 0.5)
            ys, xs = torch.where(vis_k.all(0))
            pts = sub[0][:, :3, ys, xs].permute(2, 0, 1)
            depth_ok = ((pts[:, :, 2] >= args.min_depth) & (pts[:, :, 2] <= args.max_depth)).all(1)
            ys, xs, pts = ys[depth_ok], xs[depth_ok], pts[depth_ok]
            out[f"pre_xyz_{k}"] = pts.cpu().numpy()
            out[f"pre_yx_{k}"] = torch.stack([ys, xs], 1).cpu().numpy()
            configured = getattr(args, "track_filter_mode", "global")
            for mode in ("global", "part_aware"):
                args.track_filter_mode = mode
                kept, (ky, kx), aux, *_ = tracker.get_visible_trajectories(args, sub)
                kept_keys = set((ky * 100000 + kx).tolist())
                keep = np.array([int(k_) in kept_keys for k_ in (ys * 100000 + xs).tolist()], bool)
                out[f"keep_{mode}_{k}"] = keep
                if mode == "part_aware":
                    w = np.zeros(len(keep), np.float32)
                    w[keep] = aux[:, 0].cpu().numpy()
                    out[f"weight_part_aware_{k}"] = w
            args.track_filter_mode = configured
            out[f"keep_{k}"] = out[f"keep_{configured}_{k}"]
        # 2D pixel tracks of a sparse grid for visualisation (tracking resolution)
        tr = traj[0].cpu().numpy()  # (T, 2, H, W)
        out["grid_tracks_uv"] = tr[:, :, ::6, ::6]
        out["grid_vis"] = vis[0].cpu().numpy()[:, ..., ::6, ::6] if vis.dim() >= 4 else vis.cpu().numpy()
        np.savez_compressed(args_cli.output / f"tracks_w{s:03d}.npz", **out)
        print(json.dumps(dict(window=[s, e], pairs={k: int(len(out[f'keep_{k}'])) for k in range(1, T)},
                              dropped={k: int((~out[f'keep_{k}']).sum()) for k in range(1, T)})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
