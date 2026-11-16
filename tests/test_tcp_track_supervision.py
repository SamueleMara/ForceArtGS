import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tracking.tcp_track_supervision import (  # noqa: E402
    STATUS_IN_OBJECT_MASK, STATUS_OCCLUDED, STATUS_VISIBLE, aggregate_reports, bilinear_sample,
    evaluate_window, extract_seeded_track, propose_requery_split, tcp_visibility,
)
from estimotion_2part.prior_context import build_sequence_index_map, verify_sequence_map  # noqa: E402

H, W = 40, 60


def grid(T, shift=None):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    maps = np.zeros((T, 2, H, W), dtype=np.float32)
    for t in range(T):
        dx, dy = (0.0, 0.0) if shift is None else shift(t)
        maps[t, 0] = xs + dx
        maps[t, 1] = ys + dy
    return maps


def window_tcp(uv, z=1.0):
    T = len(uv)
    return dict(uv=np.asarray(uv, float), z=np.full(T, z), usable=np.ones(T, bool),
                source_frame_index=np.arange(T) * 2, sequence_index=np.arange(T))


def depths_at(uv, z=1.0, T=None):
    out = []
    for p in uv:
        d = np.full((H, W), 2.0, dtype=np.float32)  # background behind the tip
        d[int(round(p[1])), int(round(p[0]))] = z
        out.append(d)
    return out


NO_MASK = [np.zeros((H, W), bool)] * 8


class SamplingTests(unittest.TestCase):
    def test_bilinear_subpixel(self):
        maps = grid(2)
        sampled = bilinear_sample(maps, (10.25, 7.5))
        np.testing.assert_allclose(sampled[0], [10.25, 7.5], atol=1e-6)
        self.assertTrue(np.isnan(bilinear_sample(maps, (W + 3, 2))).all())

    def test_torch_maps_supported(self):
        maps = torch.from_numpy(grid(3, lambda t: (t, 0)))
        pred, _, _ = extract_seeded_track(maps, torch.ones(3, 2, H, W), (5.5, 5.0))
        np.testing.assert_allclose(pred[:, 0], [5.5, 6.5, 7.5], atol=1e-6)


class EvaluationTests(unittest.TestCase):
    def test_identity_flow_zero_error(self):
        uv = [(20.3, 15.6)] * 5
        rec = evaluate_window(grid(5), np.ones((5, 2, H, W)), window_tcp(uv), depths_at(uv), NO_MASK,
                              depth_tolerance_m=0.01, visibility_threshold=0.5, sigma_px=1.0)
        self.assertEqual(rec["summary"]["valid_count"], 4)
        self.assertLess(rec["summary"]["max_px"], 1e-4)
        self.assertLess(rec["summary"]["E_tcp"], 1e-8)

    def test_moving_tcp_tracked_and_injected_offset_reported(self):
        motion = lambda t: (2.0 * t, 1.0 * t)
        uv = [(20 + 2 * t, 15 + t) for t in range(5)]
        rec = evaluate_window(grid(5, motion), np.ones((5, 2, H, W)), window_tcp(uv), depths_at(uv), NO_MASK,
                              depth_tolerance_m=0.01, visibility_threshold=0.5)
        self.assertLess(rec["summary"]["max_px"], 1e-4)
        offset = lambda t: (2.0 * t + (3.0 if t >= 2 else 0.0), 1.0 * t - (4.0 if t >= 2 else 0.0))
        rec = evaluate_window(grid(5, offset), np.ones((5, 2, H, W)), window_tcp(uv), depths_at(uv), NO_MASK,
                              depth_tolerance_m=0.01, visibility_threshold=0.5)
        np.testing.assert_allclose(rec["residual_px"][2:], 5.0, atol=1e-4)
        np.testing.assert_allclose(rec["residual_xy"][3], [3.0, -4.0], atol=1e-4)
        self.assertIsNone(rec["summary"]["E_tcp"])  # sigma not configured -> no invented robust score

    def test_occluded_and_masked_tip_invalid_not_zero(self):
        uv = [(20.0, 15.0)] * 4
        depths = depths_at(uv)
        depths[2][15, 20] = 0.5  # a nearer surface hides the tip
        masks = [np.zeros((H, W), bool) for _ in range(4)]
        masks[3][15, 20] = True  # tip pixel labelled object -> ambiguous contact
        rec = evaluate_window(grid(4, lambda t: (9.0, 0.0)), np.ones((4, 2, H, W)), window_tcp(uv), depths, masks,
                              depth_tolerance_m=0.01, visibility_threshold=0.5)
        self.assertEqual(rec["visibility_status"][2], STATUS_OCCLUDED)
        self.assertEqual(rec["visibility_status"][3], STATUS_IN_OBJECT_MASK)
        self.assertEqual(rec["valid_comparison"].tolist(), [False, True, False, False])
        self.assertTrue(np.isnan(rec["residual_px"][2]) and np.isnan(rec["residual_px"][3]))

    def test_hidden_seed_makes_window_unavailable(self):
        uv = [(20.0, 15.0)] * 3
        depths = depths_at(uv)
        depths[0][15, 20] = 0.3
        rec = evaluate_window(grid(3), np.ones((3, 2, H, W)), window_tcp(uv), depths, NO_MASK,
                              depth_tolerance_m=0.01, visibility_threshold=0.5)
        self.assertTrue(rec["window_status"].startswith("unavailable_seed"))
        self.assertEqual(rec["summary"]["valid_count"], 0)
        self.assertIsNone(rec["summary"]["median_px"])

    def test_low_tracker_visibility_excluded(self):
        uv = [(20.0, 15.0)] * 3
        vis = np.ones((3, 2, H, W))
        vis[2, 0] = 0.1
        rec = evaluate_window(grid(3), vis, window_tcp(uv), depths_at(uv), NO_MASK,
                              depth_tolerance_m=0.01, visibility_threshold=0.5)
        self.assertEqual(rec["valid_comparison"].tolist(), [False, True, False])

    def test_visibility_requires_explicit_tolerance(self):
        with self.assertRaises(ValueError):
            tcp_visibility((1, 1), 1.0, np.ones((4, 4)), None, None)
        self.assertEqual(tcp_visibility((1, 1), 1.0, np.ones((4, 4)), None, 0.01)[0], STATUS_VISIBLE)


class RequeryTests(unittest.TestCase):
    def test_split_proposal(self):
        residual = np.array([0, 0.5, 0.4, 9, 9, 9])
        valid = np.array([False, True, True, True, True, True])
        self.assertEqual(propose_requery_split(residual, valid, 2.0)[0], 2)
        self.assertEqual(propose_requery_split(residual, valid, 2.0, persistence=3)[0], 2)
        self.assertIsNone(propose_requery_split(residual, valid, 2.0, persistence=4)[0])
        self.assertEqual(propose_requery_split(np.array([0, 9.0]), np.array([False, True]), 2.0)[1],
                         "failure_at_first_tracked_step")
        invalid_only = propose_requery_split(residual, np.zeros(6, bool), 2.0)
        self.assertIsNone(invalid_only[0])
        with self.assertRaises(ValueError):
            propose_requery_split(residual, valid, None)


class FakeTracker:
    """Drifts on long windows (TCP track off by 6 px from frame 3); exact on short ones."""

    def __init__(self, short_tracks=500, long_tracks=500):
        self.cfg = SimpleNamespace(query_frame=0)
        self.calls = []
        self.short_tracks, self.long_tracks = short_tracks, long_tracks

    def get_track_maps(self, data):
        T = len(data["rgbs"])
        self.calls.append(T)
        drift = (lambda t: (6.0 if t >= 3 else 0.0, 0.0)) if T > 4 else None
        return torch.from_numpy(grid(T, drift))[None], torch.ones(1, T, 2, H, W)

    def get_4d_traj_maps(self, data, traj, vis):
        T = len(data["rgbs"])
        return torch.full((1, T, 1, 1, 1), float(T))  # encodes which window produced the tracks

    def get_visible_trajectories(self, args, coords):
        window_length = int(coords[0, 0, 0, 0, 0])
        return None, None, None, 0.0, (self.long_tracks if window_length > 4 else self.short_tracks)


class PhaseAWindowTests(unittest.TestCase):
    def make(self, T):
        from utils.utils import ImageLoader
        uv = [(20.0, 15.0)] * T
        rgbs = [np.full((H, W, 3), t, np.uint8) for t in range(T)]
        data = dict(rgbs=rgbs, rgbs_tensor=torch.zeros(1, T, 3, H, W), depths=depths_at(uv),
                    masks=[np.zeros((H, W), bool)] * T, masks_eroded=[np.zeros((H, W), bool)] * T,
                    cam_K=np.eye(3), features=None)
        loader = ImageLoader(images_dict=data)
        ctx = SimpleNamespace(window_tcp=lambda cf, n, dup: window_tcp(uv[:n] + ([uv[0]] if dup else [])),
                              input_dir=Path("cam_000"))
        return loader, ctx

    def run_phase(self, tracker, mode="adaptive_requery"):
        from estimotion_2part.tracking_prior_hooks import run_phase_a
        loader, ctx = self.make(6)
        args = SimpleNamespace(tcp_track_correction_mode=mode, tcp_visibility_depth_tol_m=0.01, tcp_sigma_px=None,
                               tcp_requery_error_px=2.0, tcp_requery_max_retries=2,
                               tcp_requery_persistence_frames=1, visibility_threshold=0.5, debug=False)
        traj, vis = tracker.get_track_maps(loader.data)
        with tempfile.TemporaryDirectory() as tmp:
            window_dir = Path(tmp) / "window_00000"
            out = run_phase_a(args, tracker, ctx, loader, 6, 0, False, traj, vis, window_dir)
            report = json.loads((window_dir / "tcp_track/tcp_track.json").read_text())
            summary = aggregate_reports(tmp)
        return out, report, summary

    def test_drift_triggers_bounded_requery_preserving_tracks(self):
        tracker = FakeTracker()
        (loader, size, traj, vis), report, _ = self.run_phase(tracker)
        self.assertEqual(size, 3)
        self.assertEqual(len(loader), 3)
        self.assertEqual(report["accepted_window_span"], [0, 2])
        attempt = report["requery_attempts"][0]
        self.assertTrue(attempt["accepted"])
        self.assertEqual(attempt["proposed_span"], [0, 2])
        self.assertAlmostEqual(attempt["tcp_error_after_px"], 0.0, places=4)
        self.assertEqual(report["summary"]["max_px"], 0.0)  # accepted window ends before the drift
        self.assertLessEqual(len(tracker.calls), 1 + 2)  # original + bounded retries

    def test_requery_rejected_when_object_tracks_drop(self):
        tracker = FakeTracker(short_tracks=10, long_tracks=500)
        (loader, size, _, _), report, _ = self.run_phase(tracker)
        self.assertEqual(size, 6)
        self.assertFalse(report["requery_attempts"][0]["accepted"])
        self.assertEqual(report["requery_attempts"][0]["rejection_reason"], "requery_reduces_valid_object_tracks")

    def test_diagnostic_mode_never_changes_window(self):
        tracker = FakeTracker()
        (loader, size, _, _), report, summary = self.run_phase(tracker, "diagnostic_only")
        self.assertEqual(size, 6)
        self.assertEqual(report["requery_attempts"], [])
        self.assertEqual(len(tracker.calls), 1)
        self.assertGreater(summary["max_px"], 5.9)


class SequenceMapTests(unittest.TestCase):
    def test_pingpong_mapping_and_verification(self):
        m = build_sequence_index_map(4, True, True)
        self.assertEqual(m["original"].tolist(), [0, 1, 2, 3, 2, 1, 0])
        m2 = build_sequence_index_map(3, True, False)
        self.assertEqual(m2["original"].tolist(), [0, 1, 2, 2, 1, 0])
        self.assertEqual(m2["kind"][3], "duplicate")
        frames = [np.full((2, 2), i) for i in [0, 1, 2, 3, 2, 1, 0]]
        self.assertTrue(verify_sequence_map(frames, m).all())
        frames[5] = np.full((2, 2), 9)
        self.assertFalse(verify_sequence_map(frames, m)[5])
        self.assertEqual(build_sequence_index_map(4, False, True)["original"].tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
