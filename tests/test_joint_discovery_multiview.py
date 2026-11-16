"""T05, T06, T08, T10, T11, T20: recovery from noisy calibrated multiview tracks."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.config import JointDiscoveryConfig, load_joint_discovery_config  # noqa: E402
from estimotion_2part.joint_discovery.exports import virtual_camera_poses  # noqa: E402
from estimotion_2part.joint_discovery.geometry_fit import JointCandidate  # noqa: E402
from estimotion_2part.joint_discovery.model_selection import fit_joint_hypotheses  # noqa: E402
from estimotion_2part.joint_discovery.observations import CameraCalib  # noqa: E402
from estimotion_2part.joint_discovery.rigid_kinematics import (A_revolute, axis_angle_deg,  # noqa: E402
                                                               line_to_line_distance)
from estimotion_2part.joint_discovery.synthetic import look_at_camera, make_scene  # noqa: E402
from estimotion_2part.joint_discovery.visual_losses import camera_stacks, render_point_proxy, residual_report, whitened_residuals  # noqa: E402


def cfg(**fit):
    c = JointDiscoveryConfig(mode="calibrated_joint_aware")
    c.fit.em_iterations = 3
    c.fit.n_seeds = 3
    for k, v in fit.items():
        setattr(c.fit, k, v)
    return c.validate()


class TestMultiview(unittest.TestCase):
    def test_T05_revolute_recovery_with_noise_motivated_tolerance(self):
        sig_px, sig_z = 0.7, 0.003
        sc = make_scene("revolute", seed=11, sigma_px=sig_px, sigma_z=sig_z, outlier_fraction=0.03, dropout=0.1)
        c = fit_joint_hypotheses(sc.obs, sc.calibs, cfg()).candidates["revolute"]
        # tolerance: a 0.2 m lid observed with ~3 mm depth noise and ~1500 observations -> well under 1 deg / 5 mm
        self.assertLess(axis_angle_deg(c.axis_w, sc.gt_axis), 1.0)
        self.assertLess(line_to_line_distance(c.axis_w, c.origin_w, sc.gt_axis, sc.gt_origin), 0.005)
        sign = np.sign(c.axis_w @ sc.gt_axis)
        self.assertLess(np.max(np.abs(sign * c.q - sc.gt_q)), np.radians(1.0))
        self.assertIsNotNone(c.diagnostics["final_ls"].get("axis_sigma_deg"))

    def test_T06_prismatic_recovery_and_no_pivot(self):
        sc = make_scene("prismatic", seed=12, sigma_px=0.7, sigma_z=0.003, dropout=0.1)
        c = fit_joint_hypotheses(sc.obs, sc.calibs, cfg()).candidates["prismatic"]
        self.assertLess(axis_angle_deg(c.axis_w, sc.gt_axis), 1.0)
        sign = np.sign(c.axis_w @ sc.gt_axis)
        self.assertLess(np.max(np.abs(sign * c.q - sc.gt_q)), 0.003)
        self.assertIsNone(c.origin_w)

    def test_T08_minority_moving_support_without_quotas(self):
        sc = make_scene("revolute", seed=13, n_static=210, n_moving=90, outlier_fraction=0.02)
        fit = fit_joint_hypotheses(sc.obs, sc.calibs, cfg())
        c = fit.candidates["revolute"]
        mv = np.nan_to_num(c.moving_probability) > 0.5
        gt = sc.gt_moving
        recall = (mv & gt).sum() / gt.sum()
        leak = (mv & ~gt).sum() / (~gt).sum()
        self.assertGreater(recall, 0.9)   # fast-moving tracks are not dropped as outliers
        self.assertLess(leak, 0.05)

    def test_T10_disjoint_views_and_base_only_camera(self):
        sc = make_scene("revolute", seed=14, disjoint_moving_cameras=True)
        c = fit_joint_hypotheses(sc.obs, sc.calibs, cfg()).candidates["revolute"]
        self.assertLess(axis_angle_deg(c.axis_w, sc.gt_axis), 1.5)
        sc2 = make_scene("revolute", seed=15, base_only_camera="cam_002")
        c2 = fit_joint_hypotheses(sc2.obs, sc2.calibs, cfg()).candidates["revolute"]
        mv = np.nan_to_num(c2.moving_probability) > 0.5
        cam2 = sc2.obs.track_camera == sc2.obs.camera_ids.index("cam_002")
        self.assertLessEqual((mv & cam2).sum(), 2)   # zero moving support is allowed, not fabricated
        self.assertLess(axis_angle_deg(c2.axis_w, sc2.gt_axis), 2.0)

    def test_T11_default_cameras_and_cam000_stress(self):
        self.assertEqual(load_joint_discovery_config({"mode": "calibrated_joint_aware"}).cameras, ["cam_001", "cam_002"])
        sc = make_scene("revolute", seed=16, cameras=("cam_000", "cam_001", "cam_002"))
        calibs = dict(sc.calibs)
        T = calibs["cam_000"].T_WC.copy()
        T[:3, 3] += [0.0, 0.03, 0.0]  # 3 cm extrinsic error, as observed on the real rig
        calibs["cam_000"] = CameraCalib("cam_000", calibs["cam_000"].K, T, 320, 240, "perturbed")
        fit = fit_joint_hypotheses(sc.obs, calibs, cfg())
        c = fit.candidates["revolute"]
        # physical poses are not optimized: the perturbed calibration object is used unchanged
        np.testing.assert_array_equal(fit.problem.T_CW[0], calibs["cam_000"].T_CW)
        K, T_CW = camera_stacks(sc.obs, calibs)
        res = whitened_residuals("revolute", c.axis_w, c.origin_w, c.q, np.nan_to_num(c.canonical_anchors_w),
                                 sc.obs, K, T_CW, fit.problem.noise)
        rep = residual_report(res, sc.obs, np.nan_to_num(c.moving_probability))
        # the calibration conflict is reported as a larger moving-track residual of the stressed camera
        others = max(rep["cam_001"]["median_uv_whitened"], rep["cam_002"]["median_uv_whitened"])
        self.assertGreater(rep["cam_000"]["median_uv_whitened"], 1.5 * others)
        sc_def = make_scene("revolute", seed=16, cameras=("cam_001", "cam_002"))
        c_def = fit_joint_hypotheses(sc_def.obs, sc_def.calibs, cfg()).candidates["revolute"]
        self.assertLess(axis_angle_deg(c_def.axis_w, sc_def.gt_axis), 1.0)
        self.assertTrue(np.isfinite(c.axis_w).all())

    def test_T20_virtual_camera_equals_moving_object(self):
        cal = look_at_camera("cam_001", [0.2, -0.7, 0.5], [0, 0, 0.1], width=96, height=72, f=90)
        rng = np.random.default_rng(0)
        X = rng.uniform([-0.1, -0.05, 0.1], [0.1, 0.05, 0.14], (300, 3))
        cols = rng.uniform(size=(300, 3))
        u, o, q = np.array([1.0, 0, 0]), np.array([0, 0.05, 0.12]), np.array([0.0, -0.5])
        cand = JointCandidate("revolute", u, o, q, np.ones(2, bool), None, 0, X, np.ones(300), np.zeros(300), np.zeros(300))
        Tv = virtual_camera_poses(cand, cal.T_WC)
        A = A_revolute(u, o, q[1])
        rgb1, d1, a1 = render_point_proxy(X, cols, cal, T_world_points=A)
        vcam = CameraCalib("cam_001", cal.K, Tv[1], cal.width, cal.height, "virtual")
        rgb2, d2, a2 = render_point_proxy(X, cols, vcam)
        np.testing.assert_allclose(d1, d2, atol=1e-12)    # metric depth
        np.testing.assert_array_equal(a1, a2)              # silhouette
        np.testing.assert_allclose(rgb1, rgb2)             # colour
        self.assertGreater(a1.sum(), 50)
        np.testing.assert_allclose(Tv[0], cal.T_WC)        # q_ref = 0: virtual = physical


if __name__ == "__main__":
    unittest.main()
