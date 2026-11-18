"""Stage A: frozen-hypothesis depth-conflict audit (spec next-steps §3, §10 items 3, 8, 18)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.config import JointDiscoveryConfig  # noqa: E402
from estimotion_2part.joint_discovery.depth_conflict_audit import (AuditProblem, FrozenJoint,  # noqa: E402
                                                                   candidate_independent_patches, classify_depth,
                                                                   pair_attribution, run_depth_audit)
from estimotion_2part.joint_discovery.synthetic import make_scene  # noqa: E402

FIELDS = ("r_uv_px", "r_z_m", "sigma_z_m", "depth_edge", "incidence_cos", "depth_valid", "mask_valid", "track_id",
          "moving_posterior", "original_pixel", "predicted_pixel", "measured_z", "predicted_z", "patch_id",
          "dist_to_depth_discontinuity_px", "front_surface_gap_m", "valid_measured_depth", "residual_direction",
          "depth_class", "reason")


def scene(**kw):
    return make_scene("revolute", seed=5, T=12, n_moving=120, max_angle_deg=20, **kw)


def joints(sc, wrong_scale=0.2):
    true = FrozenJoint("TRUE_R", "revolute", sc.gt_axis, sc.gt_origin, sc.gt_q, "diagnostic")
    small = FrozenJoint("SMALL_R", "revolute", sc.gt_axis, sc.gt_origin + np.array([0, 0.6, 0.4]), sc.gt_q * wrong_scale,
                        "fitted")
    static = FrozenJoint("STATIC_null", "static", None, None, np.zeros(len(sc.gt_q)), "null")
    return dict(STATIC_null=static, TRUE_R=true, SMALL_R=small)


class TestDepthConflictAudit(unittest.TestCase):
    def test_true_joint_preferred_on_clean_data_and_table_contract(self):
        sc = scene()
        with tempfile.TemporaryDirectory() as td:
            s = run_depth_audit(sc.obs, sc.calibs, JointDiscoveryConfig().noise, joints(sc), Path(td) / "a",
                                pairs=[("SMALL_R", "TRUE_R")])
            t = np.load(Path(td) / "a" / "observations_TRUE_R_rgbd.npz")
            for f in FIELDS:
                self.assertIn(f, t.files)
                self.assertEqual(len(t[f]), sc.obs.n_obs)  # every observation kept, rejected ones with a reason
            with self.assertRaises(FileExistsError):
                run_depth_audit(sc.obs, sc.calibs, JointDiscoveryConfig().noise, joints(sc), Path(td) / "a")
        p = [x for x in s["pairs"] if x["anchors_mode"] == "rgbd" and x["support"] == "union"][0]
        self.assertGreater(p["weighted_gap_total"], 0)  # positive: the true joint explains better
        self.assertGreater(p["weighted_gap_uv"], 0)
        self.assertGreater(p["weighted_gap_depth"], 0)

    def test_wrong_surface_depth_is_localized_to_the_corrupted_camera(self):
        sc = scene(wrong_surface_fraction={"cam_002": 0.5}, wrong_surface_offset_m=(0.03, 0.08))
        noise = JointDiscoveryConfig().noise
        prob = AuditProblem(sc.obs, sc.calibs, noise)
        J = joints(sc)
        ev_t = prob.evaluate(J["TRUE_R"], "2d")
        patch, _, _ = candidate_independent_patches(sc.obs)
        cls, _ = classify_depth(ev_t, prob, None, patch)
        corrupt = sc.obs_corrupted & prob.depth_scored
        clean_moving = ~sc.obs_corrupted & prob.depth_scored & sc.gt_moving[sc.obs.track]
        # injected wrong-surface samples are not classified as random noise; clean ones mostly are
        self.assertGreater(np.mean(cls[corrupt] != "valid_surface_random"), 0.8)
        self.assertGreater(np.mean(cls[clean_moving] == "valid_surface_random"), 0.8)
        # and the depth disagreement under the TRUE joint sits in cam_002, not cam_001
        r = np.abs(ev_t.r_z_m)
        c2 = (sc.obs.cam == 1) & np.isfinite(r) & sc.gt_moving[sc.obs.track]
        c1 = (sc.obs.cam == 0) & np.isfinite(r) & sc.gt_moving[sc.obs.track]
        self.assertGreater(np.median(r[c2]), 3 * np.median(r[c1]))
        ev_s = prob.evaluate(J["SMALL_R"], "2d")
        pa = pair_attribution(ev_s, ev_t, prob, patch, "all")
        self.assertIn("cam_002", pa["by_camera"])
        self.assertTrue(np.isfinite(pa["weighted_gap_depth"]))

    def test_reference_role_is_declared_and_never_fitted(self):
        with self.assertRaises(ValueError):
            FrozenJoint("x", "revolute", [1, 0, 0], [0, 0, 0], np.zeros(3), role="ground_truth")
        j = FrozenJoint("REF", "revolute", [2, 0, 0], [0, 0, 0], np.zeros(3), role="reference_posthoc")
        self.assertAlmostEqual(float(np.linalg.norm(j.axis)), 1.0)
        self.assertEqual(j.describe()["role"], "reference_posthoc")


if __name__ == "__main__":
    unittest.main()
