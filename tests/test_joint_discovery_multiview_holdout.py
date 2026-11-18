"""Stage C2: held-out camera and blocked-time folds with explicit q rules (§10 items 5, 16)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.config import JointDiscoveryConfig  # noqa: E402
from estimotion_2part.joint_discovery.synthetic import make_scene  # noqa: E402
from estimotion_2part.joint_discovery.validation_folds import (FoldSpec, camera_folds, fold_label_stability,  # noqa: E402
                                                               run_fold, time_folds)


def cfg():
    c = JointDiscoveryConfig(mode="calibrated_joint_aware")
    c.fit.em_iterations, c.fit.n_seeds = 2, 2
    return c.validate()


class TestFolds(unittest.TestCase):
    def test_fold_construction(self):
        T, ref = 20, 4
        folds = time_folds(T, ref, ["cam_001", "cam_002"], interleaved_times=[3, 9, 15])
        kinds = [f.kind for f in folds]
        self.assertEqual(kinds, ["interleaved_time", "contiguous_block", "forward_block"])
        for f in folds:
            self.assertFalse(f.test_time_mask[ref])
            self.assertFalse(np.any(f.train_time_mask & f.test_time_mask))
        blk = np.flatnonzero(folds[1].test_time_mask)
        self.assertTrue(np.all(np.diff(blk) == 1))
        self.assertTrue(folds[2].test_time_mask[-1])
        cf = camera_folds(T, ["cam_001", "cam_002"])
        self.assertEqual([f.test_cameras for f in cf], [["cam_001"], ["cam_002"]])
        with self.assertRaises(ValueError):
            FoldSpec("x", "interleaved_time", np.ones(T, bool), np.ones(T, bool), [], [])

    def test_heldout_camera_predicts_with_q_from_the_other_camera(self):
        sc = make_scene("revolute", seed=6, T=12, n_moving=120, max_angle_deg=25)
        fold = camera_folds(12, sc.obs.camera_ids)[1]          # fit cam_001, score cam_002
        r = run_fold(sc.obs, sc.calibs, cfg(), fold, ref=0)
        self.assertEqual(r["status"], "OK")
        self.assertEqual(set(r["q_at_test"]["revolute"]), {"constrained_by_other_camera"})
        self.assertGreater(r["comparisons"]["revolute_vs_static"]["z_cluster"], 2.0)
        self.assertGreater(r["comparisons"]["revolute_vs_prismatic"]["delta"], 0.0)
        self.assertGreater(r["test_camera_coverage"]["cam_002"], 0)

    def test_block_fold_reports_interpolated_q_and_view_without_moving_part(self):
        sc = make_scene("revolute", seed=7, T=12, n_moving=120, max_angle_deg=25, base_only_camera="cam_002")
        folds = time_folds(12, 0, sc.obs.camera_ids)
        r = run_fold(sc.obs, sc.calibs, cfg(), folds[0], ref=0)
        self.assertEqual(r["status"], "OK")
        self.assertIn("interpolated", r["q_at_test"]["revolute"])
        self.assertNotIn("observed_in_training(leak)", r["q_at_test"]["revolute"])
        # the base-only camera is scored too, and contributes no spurious moving support
        fc = camera_folds(12, sc.obs.camera_ids)[1]
        rc = run_fold(sc.obs, sc.calibs, cfg(), fc, ref=0)
        self.assertLess(rc["comparisons"]["revolute_vs_static"]["z_cluster"], 2.0)
        st = fold_label_stability([r, rc])
        self.assertIn("per_fold", st)


if __name__ == "__main__":
    unittest.main()
