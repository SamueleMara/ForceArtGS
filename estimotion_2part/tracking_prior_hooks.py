"""Per-window hooks connecting :mod:`prior_context` to ``track_2part`` (all opt-in).

Phase A  run_phase_a           raw-map TCP diagnostic, optional adaptive requery split
Phase B  seed masks            -> SeedTerm in pairwise and joint optimisation
Phase C  contact samples       -> ContactTerm in pairwise and joint optimisation
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from termcolor import cprint

from estimotion_2part.window_priors import ContactTerm, SeedTerm
from tracking.tcp_track_supervision import (
    _jsonable,
    evaluate_window,
    propose_requery_split,
    write_window_report,
)


def _require(args, name):
    value = getattr(args, name, None)
    if value is None:
        raise ValueError(f"{name} must be explicitly configured when its phase is enabled")
    return value


# ----------------------------------------------------------------------------- Phase A


def _count_object_tracks(args, tracker, loader, traj, vis, end):
    with torch.inference_mode():
        coords = tracker.get_4d_traj_maps(loader.data, traj, vis)
        _, _, _, _, count = tracker.get_visible_trajectories(args, coords[:, [0, end]])
    return int(count)


def _slice_window(loader, end):
    short = loader[:end + 1]
    duplicated = len(short) == 2
    if duplicated:
        short = short.duplicate_last_frame()
    return short, duplicated


def run_phase_a(args, tracker, ctx, loader, max_window_size, current_frame, duplicated, traj, vis, window_output_dir):
    """Evaluate the TCP track; in ``adaptive_requery`` mode possibly shorten and requery the window.

    Object tracks are only ever recomputed from the actual RGB images by AllTracker; no
    displacement is copied from the robot.  Returns (loader, max_window_size, traj, vis).
    """
    mode = getattr(args, "tcp_track_correction_mode", "diagnostic_only")
    if mode not in ("diagnostic_only", "adaptive_requery"):
        raise ValueError(f"Unknown tcp_track_correction_mode {mode!r}")
    if int(getattr(tracker.cfg, "query_frame", 0)) != 0:
        raise ValueError("Phase A requires AllTracker query_frame=0 (maps seeded at window frame 0)")
    depth_tol = _require(args, "tcp_visibility_depth_tol_m")
    sigma = getattr(args, "tcp_sigma_px", None)

    def evaluate(loader_, size_, dup_, traj_, vis_):
        window_tcp = ctx.window_tcp(current_frame, size_, dup_)
        return evaluate_window(traj_[0], vis_[0], window_tcp, loader_.data["depths"], loader_.data["masks"],
                               depth_tolerance_m=depth_tol, visibility_threshold=args.visibility_threshold,
                               sigma_px=sigma)

    records = evaluate(loader, max_window_size, duplicated, traj, vis)
    original_span = [current_frame, current_frame + max_window_size - 1]
    attempts = []
    if mode == "adaptive_requery":
        tolerance = _require(args, "tcp_requery_error_px")
        max_retries = int(getattr(args, "tcp_requery_max_retries", 2))
        persistence = int(getattr(args, "tcp_requery_persistence_frames", 1))
        for retry in range(max_retries):
            end, reason = propose_requery_split(records["residual_px"], records["valid_comparison"],
                                                tolerance, persistence)
            if end is None or end >= max_window_size - 1:
                attempts.append(dict(retry=retry, proposed=False, reason=reason))
                break
            short, short_dup = _slice_window(loader, end)
            with torch.inference_mode():
                new_traj, new_vis = tracker.get_track_maps(short.data)
            original_count = _count_object_tracks(args, tracker, loader, traj, vis, end)
            new_count = _count_object_tracks(args, tracker, short, new_traj, new_vis, end)
            new_records = evaluate(short, end + 1, short_dup, new_traj, new_vis)
            before = records["residual_px"][:end + 1][records["valid_comparison"][:end + 1]]
            after = new_records["residual_px"][new_records["valid_comparison"]]
            accepted = new_count >= original_count and end >= 1
            attempts.append(dict(
                retry=retry, proposed=True, split_reason=reason, accepted=accepted,
                proposed_span=[current_frame, current_frame + end],
                object_tracks_original_window=original_count, object_tracks_requeried=new_count,
                tcp_error_before_px=float(np.median(before)) if len(before) else None,
                tcp_error_after_px=float(np.median(after)) if len(after) else None,
                rejection_reason=None if accepted else "requery_reduces_valid_object_tracks"))
            if not accepted:
                break
            cprint(f"  Phase A: requery accepted, window shortened to frames {current_frame}-{current_frame + end}", "yellow")
            loader, max_window_size, duplicated, traj, vis, records = short, end + 1, short_dup, new_traj, new_vis, new_records
    report = write_window_report(
        Path(window_output_dir) / "tcp_track", records, camera=str(ctx.input_dir.name),
        extra=dict(mode=mode, track_point=(getattr(ctx, "tcp", None) or {}).get("track_point", "tcp"), original_window_span=original_span,
                   accepted_window_span=[current_frame, current_frame + max_window_size - 1],
                   requery_attempts=attempts, depth_tolerance_m=depth_tol, sigma_px=sigma,
                   visibility_threshold=args.visibility_threshold),
        rgbs=loader.data["rgbs"] if getattr(args, "debug", False) else None)
    summary = report["summary"]
    cprint(f"  Phase A TCP track: {report['status']}, valid={summary['valid_count']}, "
           f"median={summary.get('median_px')} px", "magenta")
    return loader, max_window_size, traj, vis


# ----------------------------------------------------------------------------- Phase B / C


def seed_mask_for_pixels(seed_map, y, x):
    if seed_map is None:
        return None
    seed = torch.from_numpy(seed_map).to(device=y.device)
    return seed[y.long(), x.long()]


def _contact_term(args, contact_xyz, valid, tool_axis=None):
    return ContactTerm(torch.as_tensor(contact_xyz, dtype=torch.float32), torch.as_tensor(valid, dtype=torch.bool),
                       weight=getattr(args, "contact_surface_weight", 0.0), sigma_m=getattr(args, "contact_sigma_m", None),
                       robust_delta_m=getattr(args, "contact_robust_delta_m", None),
                       support_max_distance_m=getattr(args, "contact_support_max_distance_m", None),
                       min_depth=getattr(args, "min_depth", None), max_depth=getattr(args, "max_depth", None),
                       tool_axis=tool_axis, tool_radius_m=getattr(args, "contact_exclude_tool_radius_m", None))


def _seed_term(args, mask):
    return SeedTerm(mask, getattr(args, "moving_prompt_weight", 0.0),
                    min_seeds=int(getattr(args, "moving_prompt_min_seeds", 1)))


class WindowPriors:
    """Builds per-pair and joint prior terms for one tracking window and records their reports."""

    def __init__(self, args, ctx, current_frame, window_length, duplicated):
        self.args, self.ctx = args, ctx
        self.current_frame, self.window_length, self.duplicated = current_frame, window_length, duplicated
        self.seed_map = ctx.seed_map(current_frame) if ctx.phase_b else None
        self.tool_axis = None
        if ctx.phase_c:
            self.contact_xyz, self.contact_valid, _ = ctx.window_contact(current_frame, window_length, duplicated)
            if getattr(args, "contact_exclude_tool_radius_m", None) is not None:
                # canonical support lives in window frame 0: the tool axis at that frame
                self.tool_axis = (ctx.tcp["tool_tip_cam"][current_frame], ctx.tcp["tool_ee_cam"][current_frame])
        self.pair_infos = []
        self.joint_infos = None
        self.num_pairs = 0

    def pair_terms(self, point_indices, last_idx, device):
        y, x = point_indices
        seed_term = contact_term = None
        if self.ctx.phase_b and self.seed_map is not None:
            mask = seed_mask_for_pixels(self.seed_map, y, x)
            if mask is not None and bool(mask.any()):
                seed_term = _seed_term(self.args, mask.to(device))
        if self.ctx.phase_c:
            contact_term = _contact_term(self.args, self.contact_xyz[[last_idx]], self.contact_valid[[last_idx]],
                                         tool_axis=self.tool_axis)
        self._pending = (last_idx, seed_term, contact_term)
        return seed_term, contact_term

    def record_pair(self):
        last_idx, seed_term, contact_term = self._pending
        self.pair_infos.append(dict(window_frame=int(last_idx),
                                    seed=None if seed_term is None else seed_term.info,
                                    contact=None if contact_term is None else contact_term.info))
        self.num_pairs += 1

    def __call__(self, point_keys, device):
        """Joint terms: one transform row per window frame 1..num_pairs."""
        seed_term = contact_term = None
        if self.ctx.phase_b and self.seed_map is not None and point_keys:
            keys = torch.as_tensor(point_keys, dtype=torch.long)
            mask = torch.from_numpy(self.seed_map)[keys[:, 0], keys[:, 1]]
            if bool(mask.any()):
                seed_term = _seed_term(self.args, mask.to(device))
        if self.ctx.phase_c:
            rows = self.window_length - 1  # t1_map rows 1..max_window_size-1 -> window frames 1..
            xyz = self.contact_xyz[1:1 + rows]
            valid = self.contact_valid[1:1 + rows].copy()
            valid[self.num_pairs:] = False  # rows without tracked pairs have no visual constraint
            contact_term = _contact_term(self.args, xyz, valid, tool_axis=self.tool_axis)
        return seed_term, contact_term

    def write(self, window_output_dir, seed_init_info=None):
        report = dict(window_first_frame=self.current_frame, window_length=self.window_length,
                      seed_map_available=self.seed_map is not None,
                      seed_pixels=int(self.seed_map.sum()) if self.seed_map is not None else 0,
                      pairs=self.pair_infos, joint=self.joint_infos, seed_initialisation=seed_init_info)
        path = Path(window_output_dir) / "priors"
        path.mkdir(parents=True, exist_ok=True)
        (path / "priors.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
