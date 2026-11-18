"""Stage B1: candidate-independent depth evidence (spec next-steps §4 B1; §10 items 6, 7, 8)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.depth_evidence import (CODE, DepthEvidenceConfig, apply_depth_evidence,  # noqa: E402
                                                             assess_depth_evidence)
from estimotion_2part.joint_discovery.observations import FrameMap, build_from_dense_tracks  # noqa: E402
from estimotion_2part.joint_discovery.synthetic import look_at_camera, make_scene  # noqa: E402


class TestDepthEvidence(unittest.TestCase):
    def test_clean_scene_keeps_depth_and_never_touches_2d(self):
        sc = make_scene("revolute", seed=1, T=10, n_moving=120, late_tracks_fraction=0.4, dropout=0.1)
        code, det = assess_depth_evidence(sc.obs, DepthEvidenceConfig())
        dv = sc.obs.depth_valid
        # clean depth is never flagged as wrong; sparse synthetic points may only be 'unverified'
        self.assertLess(np.isin(code[dv], [CODE["inconsistent_local_rigidity"], CODE["depth_jump"]]).mean(), 0.02)
        self.assertTrue(np.all(np.isin(code[dv], [CODE["valid_surface"], CODE["unverified"],
                                                   CODE["inconsistent_local_rigidity"], CODE["depth_jump"]])))
        out = apply_depth_evidence(sc.obs, code)
        self.assertEqual(out.n_obs, sc.obs.n_obs)                         # 2D rows all survive
        np.testing.assert_array_equal(out.uv_valid, sc.obs.uv_valid)
        self.assertTrue(np.all(out.depth_valid <= sc.obs.depth_valid))   # never promotes depth
        self.assertTrue(np.isnan(out.x_w[~out.depth_valid]).all())

    def test_wrong_surface_track_fails_local_rigidity_and_jump(self):
        sc = make_scene("revolute", seed=2, T=10, n_moving=150, wrong_surface_fraction={"cam_001": 0.3},
                        wrong_surface_offset_m=(0.04, 0.08))
        code, _ = assess_depth_evidence(sc.obs, DepthEvidenceConfig())
        bad = sc.obs_corrupted & sc.obs.depth_valid
        flagged = np.isin(code, [CODE["inconsistent_local_rigidity"], CODE["depth_jump"]])
        self.assertGreater(flagged[bad].mean(), 0.7)
        clean = ~sc.obs_corrupted & sc.obs.depth_valid
        self.assertLess(flagged[clean].mean(), 0.1)

    def test_coherent_patch_bias_is_not_detectable_by_local_rules(self):
        """Documented limitation (black-box finding): a bias shared by a whole patch preserves local
        distances, so per-observation rules cannot remove it; it must surface in Stage A/C instead."""
        sc = make_scene("revolute", seed=3, T=10, n_moving=150, moving_depth_bias_m={"cam_002": 0.02})
        code, _ = assess_depth_evidence(sc.obs, DepthEvidenceConfig())
        bad = sc.obs_corrupted & sc.obs.depth_valid
        self.assertGreater((code[bad] == CODE["valid_surface"]).mean(), 0.8)

    def test_behind_silhouette_depth_and_tool_and_unverified_policy(self):
        cal = look_at_camera("cam_001", [0, -1, 0.5], [0, 0, 0], width=40, height=30, f=40)
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "depth_npy" / "cam_001"
            d.mkdir(parents=True)
            depth = np.full((30, 40), 2.0)
            depth[:, :20] = 1.0                   # near surface on the left, background on the right
            for f in (0, 1):
                np.save(d / f"frame_{f:06d}.npy", depth)
            fm = FrameMap([0, 1], [0, 2], [0.0, 0.1])
            uv = np.array([[[10.0, 15.0], [21.0, 15.0], [30.0, 15.0]]] * 2)
            obs = build_from_dense_tracks(cal, fm, uv, np.ones((2, 3)), [depth, depth], [np.ones((30, 40), bool)] * 2, 0)
            code, det = assess_depth_evidence(obs, DepthEvidenceConfig(), scene=td)
            self.assertTrue(det["front_gap_assessed"])
            # track 1 is 1 px right of the silhouette on the far surface: its depth is not promoted
            self.assertTrue(np.all(code[obs.track == 1] == CODE["behind_depth_discontinuity"]))
            self.assertTrue(np.all(code[obs.track == 0] != CODE["behind_depth_discontinuity"]))
            # too few neighbours for the rigidity test -> unverified, dropped by default, kept on request
            self.assertTrue(np.all(code[obs.track == 0] == CODE["unverified"]))
            self.assertFalse(apply_depth_evidence(obs, code, "drop").depth_valid[obs.track == 0].any())
            self.assertTrue(apply_depth_evidence(obs, code, "keep").depth_valid[obs.track == 0].all())
            code2, _ = assess_depth_evidence(obs, DepthEvidenceConfig(), scene=td, tool_distance_m=np.full(obs.n_obs, 0.01))
            self.assertTrue(np.all(code2[obs.track == 0] == CODE["robot_contamination"]))
        with self.assertRaises(ValueError):
            DepthEvidenceConfig(unverified_policy="promote").validate()


if __name__ == "__main__":
    unittest.main()
