"""Stage B2/C1: correlated errors, cluster-robust z and AR(1) depth calibration (§10 items 1, 2, 16)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.cli import run_discovery  # noqa: E402
from estimotion_2part.joint_discovery.config import JointDiscoveryConfig  # noqa: E402
from estimotion_2part.joint_discovery.model_selection import static_core_depth_ar1  # noqa: E402
from estimotion_2part.joint_discovery.observations import depth_sigma  # noqa: E402
from estimotion_2part.joint_discovery.synthetic import make_scene  # noqa: E402
from estimotion_2part.joint_discovery.validation_folds import grouped_paired_statistics  # noqa: E402


def ar1(rng, n, rho, sigma):
    e = np.zeros(n)
    e[0] = rng.normal(0, sigma)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(0, sigma * np.sqrt(1 - rho ** 2))
    return e


class TestCorrelatedNoise(unittest.TestCase):
    def test_naive_z_overstates_correlated_tracks_cluster_z_does_not(self):
        rng = np.random.default_rng(0)
        G, ng = 20, 50
        groups = np.repeat(np.arange(G), ng)
        d = 0.1 + rng.normal(0, 1.0, G)[groups] + rng.normal(0, 0.1, G * ng)  # shared surface effect
        s = grouped_paired_statistics(d, groups)
        self.assertGreater(abs(s["z_naive"]), 3 * abs(s["z_cluster"]))   # naive overstates significance
        self.assertGreater(abs(s["z_naive"]), 2.0)
        self.assertGreater(s["design_effect"], 10)
        self.assertLess(s["n_eff"], G * 2)
        self.assertLessEqual(s["bootstrap_ci95"][0], 0.0)
        # independent tracks: both agree
        d2 = 0.1 + rng.normal(0, 1.0, G * ng)
        s2 = grouped_paired_statistics(d2, groups)
        self.assertLess(abs(s2["z_naive"] - s2["z_cluster"]), 1.5)

    def test_ar1_rho_estimated_on_static_core_and_inflates_sigma(self):
        sc = make_scene("revolute", seed=0, T=20, n_moving=1, n_static=150, sigma_z=0.0, max_angle_deg=0.0,
                        cameras=("cam_001",))
        rng = np.random.default_rng(1)
        obs = sc.obs
        for k in np.unique(obs.track):
            m = np.flatnonzero(obs.track == k)
            m = m[np.argsort(obs.time[m])]
            obs.z[m] = obs.z[m] + ar1(rng, len(m), 0.8, 0.004)
        rho = static_core_depth_ar1(obs, np.ones(obs.n_tracks, bool))
        self.assertAlmostEqual(rho, 0.8, delta=0.1)
        n = JointDiscoveryConfig().noise
        s0 = depth_sigma(np.array([1.0]), n)
        n.depth_ar1_rho = 0.8
        np.testing.assert_allclose(depth_sigma(np.array([1.0]), n), s0 * 3.0)  # sqrt(1.8/0.2) = 3
        with self.assertRaises(ValueError):
            JointDiscoveryConfig(mode="calibrated_joint_aware", noise=type(n)(depth_ar1_rho=1.0)).validate()

    def test_static_object_with_camera_correlated_depth_bias_is_not_moving(self):
        sc = make_scene("revolute", seed=4, T=12, max_angle_deg=0.0, camera_depth_bias_m={"cam_002": 0.01})
        rng = np.random.default_rng(2)
        off = rng.normal(0, 0.005, sc.obs.n_tracks)       # persistent per-track depth offsets
        sc.obs.z = sc.obs.z + off[sc.obs.track]
        cfg = JointDiscoveryConfig(mode="calibrated_joint_aware")
        cfg.fit.em_iterations, cfg.fit.n_seeds = 2, 2
        cfg.comparison.uncertainty_method = "cluster_robust"
        cfg.comparison.remote_hinge_policy = "observability"
        cfg.noise.calibrate_depth_ar1 = True
        with tempfile.TemporaryDirectory() as td:
            s = run_discovery(sc.obs, sc.calibs, cfg, Path(td) / "r", scene="synthetic")
        self.assertIn(s["decision"], ("static/no_motion", "ambiguous/unobservable"))


if __name__ == "__main__":
    unittest.main()
