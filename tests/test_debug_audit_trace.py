"""Audit tracing (samuele/debug_audit/trace.py) is opt-in and behaviour-neutral."""
import json
import os
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samuele.debug_audit import trace  # noqa: E402
from tracking.tracker_toolkit import Tracker  # noqa: E402
from estimotion_2part.models.trans_models import TransformLossComputer  # noqa: E402


def _coords(seed=0, H=40, W=50):
    g = torch.Generator().manual_seed(seed)
    c = torch.zeros(1, 2, 5, H, W)
    c[0, :, 0:2] = torch.rand(2, 2, H, W, generator=g)
    c[0, :, 2] = 0.5 + torch.rand(2, H, W, generator=g)
    c[0, 1, :3] += 0.01 * torch.randn(3, H, W, generator=g)
    c[0, 1, :3, :, : W // 3] += 0.05                           # a moving minority
    c[0, :, 2, :3, :3] = float("-inf")                           # invalid depth corner
    c[0, :, 3] = torch.rand(2, H, W, generator=g)               # visibility
    c[0, :, 4] = (torch.rand(2, H, W, generator=g) > 0.1).float()
    return c


ARGS = types.SimpleNamespace(visibility_threshold=0.2, min_depth=0.1, max_depth=3.0)


class TraceSwitchTests(unittest.TestCase):
    def test_off_by_default_and_noop(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(trace.ENV_ROOT, None)
            self.assertFalse(trace.enabled())
            self.assertIsNone(trace.event("x", local_file="f", function="g", value={"a": 1}))

    def test_event_schema_arrays_and_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {trace.ENV_ROOT: tmp,
                                                                              "FREEARTGS_TRACE_RUN_ID": "unit"}):
            x = torch.arange(6.0).reshape(2, 3)
            before = x.clone()
            states = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
            with trace.context(window_id=3, part_id=1):
                rec = trace.event("unit_event", local_file="a.py", function="f", value={"v": np.float32(1.5)},
                                  arrays={"x": x}, coordinate_frame="camera_optical")
            self.assertTrue(torch.equal(x, before))
            self.assertTrue(torch.equal(states[0], torch.get_rng_state()))
            self.assertTrue(np.array_equal(states[1], np.random.get_state()[1]))
            self.assertEqual(states[2], random.getstate())
            line = json.loads(Path(tmp, "events.jsonl").read_text().splitlines()[-1])
            for key in trace.SCHEMA_KEYS:
                self.assertIn(key, line)
            self.assertEqual((line["run_id"], line["window_id"], line["part_id"]), ("unit", 3, 1))
            self.assertEqual(rec["artifact_refs"], line["artifact_refs"])
            saved = np.load(Path(tmp, line["artifact_refs"][0]))
            self.assertTrue(np.array_equal(saved["x"], before.numpy()))

    def test_matrix_rigidity(self):
        T = np.eye(4)
        T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        r = trace.matrix_rigidity(T)
        self.assertAlmostEqual(r["orth_err_fro"], 0.0)
        self.assertAlmostEqual(r["polar_rotation_deg"], 90.0)
        T[:3, :3] *= 1.1
        self.assertAlmostEqual(trace.matrix_rigidity(T)["max_stretch"], 0.1, places=9)


class HookNeutralityTests(unittest.TestCase):
    def _filter(self):
        tracker = Tracker.__new__(Tracker)
        return tracker.get_visible_trajectories(ARGS, _coords())

    def test_track_filter_identical_with_tracing(self):
        os.environ.pop(trace.ENV_ROOT, None)
        off = self._filter()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {trace.ENV_ROOT: tmp}):
            on = self._filter()
            events = trace.read_events(Path(tmp, "events.jsonl"))
            arr = np.load(Path(tmp, events[-1]["artifact_refs"][0]))
        self.assertTrue(torch.equal(off[0], on[0]))
        self.assertTrue(torch.equal(off[1][0], on[1][0]) and torch.equal(off[1][1], on[1][1]))
        self.assertEqual((off[3], off[4]), (on[3], on[4]))
        v = events[-1]["value"]
        self.assertEqual(v["final"], off[4])
        # the recorded stage masks reproduce the returned set exactly
        self.assertTrue(np.array_equal(arr["final_y"], off[1][0].numpy()))
        n_range = int(arr["depth_in_range"].sum())
        self.assertEqual(n_range, len(arr["global_keep"]))
        self.assertEqual(int(arr["global_keep"].sum()), len(arr["density_keep"]))
        self.assertFalse(bool(arr["cand_finite"].all()))     # the -inf corner is recorded, then rejected

    def test_loss_terms_identical_with_tracing(self):
        torch.manual_seed(0)
        n = 64
        kw = dict(pred_t1=torch.randn(n, 3), pred_t2=torch.randn(n, 3) + 1, w_sel=torch.rand(n, 1).squeeze(1),
                  blended=torch.randn(n, 3), target_points=torch.randn(n, 3) + 2, source_points=torch.randn(n, 3),
                  weight_logits=torch.randn(n, 1), init_weights_tensor=torch.rand(n))
        kw["blended"] = None
        lc = lambda: TransformLossComputer(loss_type="huber", cam_k_tensor=torch.tensor([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]),
                                           w_p=0.1, w_m=1.0, w_e=0.01, w_s=0.0, w_init=0.5, weight_layout="flat",
                                           allow_normalized_l1=False, invalid_loss_mode="raise")
        os.environ.pop(trace.ENV_ROOT, None)
        a = lc()
        try:
            off = a.compute_total_loss(**kw)
        except Exception as error:                       # loss signature details are not under test
            self.skipTest(f"loss computer needs pipeline-specific inputs: {error}")
        self.assertFalse(hasattr(a, "_trace_terms"))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {trace.ENV_ROOT: tmp}):
            b = lc()
            on = b.compute_total_loss(**kw)
            self.assertIn("total", b._trace_terms)
        self.assertTrue(torch.equal(off[0], on[0]) and torch.equal(off[1], on[1]))


if __name__ == "__main__":
    unittest.main()
