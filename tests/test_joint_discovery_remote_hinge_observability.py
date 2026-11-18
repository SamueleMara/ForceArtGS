"""Stage C3 / Gate C: far-hinge translation limit vs a weak but real near hinge (§10 items 4, 17)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.cli import run_discovery  # noqa: E402
from estimotion_2part.joint_discovery.config import JointDiscoveryConfig  # noqa: E402
from estimotion_2part.joint_discovery.observability import best_translation, hinge_support_check  # noqa: E402
from estimotion_2part.joint_discovery.synthetic import make_scene  # noqa: E402


def cfg():
    c = JointDiscoveryConfig(mode="calibrated_joint_aware")
    c.fit.em_iterations, c.fit.n_seeds = 3, 3
    c.comparison.uncertainty_method = "cluster_robust"
    c.comparison.remote_hinge_policy = "observability"
    return c.validate()


def run(sc):
    with tempfile.TemporaryDirectory() as td:
        return run_discovery(sc.obs, sc.calibs, cfg(), Path(td) / "r", scene="synthetic")


class TestRemoteHingeObservability(unittest.TestCase):
    def test_drawer_with_joint_play_is_not_a_confident_revolute(self):
        # a 10 cm drawer-like motion along an arc of 3 m radius (1.9 deg of rotational play)
        sc = make_scene("revolute", seed=8, T=12, n_moving=150, axis=[0, 0, 1], hinge_offset=[3.0, 0.0, 0.05],
                        max_angle_deg=1.9)
        s = run(sc)
        self.assertNotEqual(s["decision"], "revolute")
        ro = s["revolute_observability"]
        self.assertFalse(ro["geometry_identifiable"])
        self.assertFalse(ro["report_hinge_origin"])

    def test_weak_but_real_near_hinge_is_not_rejected_by_a_ratio(self):
        sc = make_scene("revolute", seed=9, T=12, n_moving=150, max_angle_deg=8.0)
        s = run(sc)
        self.assertEqual(s["decision"], "revolute")
        self.assertTrue(s["revolute_observability"]["geometry_identifiable"])
        self.assertIn(s["type_confidence"]["level"], ("moderate", "high"))

    def test_helpers(self):
        X = np.random.default_rng(0).normal(size=(20, 3))
        Xt = X[None] + np.array([[0, 0, 0], [0.1, 0, 0]])[:, None]
        np.testing.assert_allclose(best_translation(X, Xt), [[0, 0, 0], [0.1, 0, 0]], atol=1e-12)

        class C:
            kind = "revolute"
            axis_w = np.array([0, 0, 1.0])
            origin_w = np.array([5.0, 0, 0])
            moving_probability = np.ones(20)
            canonical_anchors_w = X * 0.05
        self.assertFalse(hinge_support_check(C(), 1.0)["hinge_within_support"])
        C.origin_w = np.array([0.02, 0, 0])
        self.assertTrue(hinge_support_check(C(), 1.0)["hinge_within_support"])


if __name__ == "__main__":
    unittest.main()
