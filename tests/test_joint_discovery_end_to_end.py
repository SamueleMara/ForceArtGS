"""Stage G / §10 item 20: legacy untouched, stale inputs refused, explicit abstention, output contract."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.cli import _refuse_stale_observations, run_discovery  # noqa: E402
from estimotion_2part.joint_discovery.config import (JointDiscoveryConfig, load_joint_discovery_config,  # noqa: E402
                                                     resolve_joint_discovery)
from estimotion_2part.joint_discovery.synthetic import make_scene  # noqa: E402

CONTRACT = ("used_for_visual_fit", "used_for_contact_fit", "used_for_force_fit", "used_for_model_selection",
            "calibration_verified", "contact_annotation_provenance", "holdout_camera_ids", "heldout_time_block_indices",
            "uncertainty_method", "renderer_evaluation_kind")


def fast(c):
    c.fit.em_iterations, c.fit.n_seeds = 2, 2
    return c.validate()


class TestEndToEnd(unittest.TestCase):
    def test_legacy_defaults_unchanged(self):
        c = resolve_joint_discovery(None)
        self.assertEqual(c.mode, "legacy")
        self.assertEqual(c.comparison.uncertainty_method, "track_paired")
        self.assertEqual(c.comparison.remote_hinge_policy, "ratio")
        self.assertEqual(c.depth_evidence.mode, "raw")
        self.assertIsNone(c.noise.depth_ar1_rho)
        self.assertFalse(c.noise.calibrate_depth_ar1)
        self.assertFalse(c.enable_physical_selection)
        self.assertTrue(c.require_calibrated_wrench)
        for bad in ({"comparison": {"uncertainty_method": "bayes"}}, {"comparison": {"remote_hinge_policy": "x"}},
                    {"depth_evidence": {"mode": "magic"}}, {"enable_physical_selection": True}):
            with self.assertRaises(ValueError):
                load_joint_discovery_config(dict(mode="calibrated_joint_aware", **bad))

    def test_stale_observation_table_is_refused(self):
        sc = make_scene("revolute", seed=0, T=6)
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "metadata").mkdir()
            rows = ["frame_index,source_frame_index,timestamp_s"] + [f"{i},{2 * i + 1},{0.1 * i:.2f}" for i in range(10)]
            (Path(td) / "metadata" / "frame_timestamps.csv").write_text("\n".join(rows))
            with self.assertRaises(SystemExit):          # source frames differ from the cached table
                _refuse_stale_observations([sc.obs], td)

    def test_contract_fingerprints_and_explicit_abstention(self):
        c = fast(JointDiscoveryConfig(mode="calibrated_joint_aware"))
        sc = make_scene("revolute", seed=1, T=10, n_moving=100, max_angle_deg=30)
        with tempfile.TemporaryDirectory() as td:
            s = run_discovery(sc.obs, sc.calibs, c, Path(td) / "r", scene="synthetic", experiment_id="NEXT_TEST")
            sel = json.loads((Path(td) / "r" / "model_selection.json").read_text())
            man = json.loads((Path(td) / "r" / "manifest.json").read_text())
            self.assertEqual(s["decision"], "revolute")
            self.assertTrue((Path(td) / "r" / "freeart_adapter").is_dir())
            self.assertFalse((Path(td) / "r" / "freeart_adapter" / "ABSTAINED.json").exists())
        for k in CONTRACT:
            self.assertIn(k, sel["output_contract"])
        self.assertEqual(sel["output_contract"]["renderer_evaluation_kind"], "point_proxy")
        self.assertEqual(sel["experiment_id"], "NEXT_TEST")
        self.assertEqual(sel["scores"]["revolute_vs_prismatic"]["z_method"], "track_paired")   # legacy z
        self.assertIn("z_cluster", sel["grouped_statistics"]["revolute_vs_prismatic"])        # always reported
        for k in ("config_sha", "calibration_sha", "time_map_sha"):
            self.assertIn(k, man["fingerprints"])
        # an unobservable case must not export a silently chosen joint
        sc2 = make_scene("revolute", seed=2, T=10, max_angle_deg=0.0)
        with tempfile.TemporaryDirectory() as td:
            s2 = run_discovery(sc2.obs, sc2.calibs, c, Path(td) / "r", scene="synthetic")
            self.assertNotIn(s2["decision"], ("revolute", "prismatic"))
            ab = json.loads((Path(td) / "r" / "freeart_adapter" / "ABSTAINED.json").read_text())
            self.assertEqual(ab["status"], "ABSTAINED")
            self.assertFalse(list((Path(td) / "r" / "freeart_adapter").glob("**/transforms_*.json")))

    def test_identical_fold_override(self):
        c = fast(JointDiscoveryConfig(mode="calibrated_joint_aware"))
        sc = make_scene("prismatic", seed=3, T=10)
        with tempfile.TemporaryDirectory() as td:
            run_discovery(sc.obs, sc.calibs, c, Path(td) / "a", scene="synthetic",
                          fold=dict(reference_time_index=1, heldout_times=[4, 7]))
            ia = json.loads((Path(td) / "a" / "input_audit.json").read_text())
            self.assertEqual((ia["reference_time_index"], ia["heldout_times"]), (1, [4, 7]))
            with self.assertRaises(ValueError):
                run_discovery(sc.obs, sc.calibs, c, Path(td) / "b", scene="synthetic",
                              fold=dict(reference_time_index=4, heldout_times=[4, 7]))


if __name__ == "__main__":
    unittest.main()
