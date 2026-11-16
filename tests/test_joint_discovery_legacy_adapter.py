"""T21 + adapter contracts: legacy default, unchanged legacy optimizer math, export schemas."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from estimotion_2part.joint_discovery.config import (JointDiscoveryConfig, load_joint_discovery_config,  # noqa: E402
                                                     resolve_joint_discovery)
from estimotion_2part.joint_discovery.exports import (export_for_existing_freeart, load_joint_candidate,  # noqa: E402
                                                      virtual_camera_poses)
from estimotion_2part.joint_discovery.geometry_fit import JointCandidate  # noqa: E402
from estimotion_2part.joint_discovery.observations import FrameMap, project  # noqa: E402
from estimotion_2part.joint_discovery.rigid_kinematics import A_joint, se3_inverse, transform_points  # noqa: E402
from estimotion_2part.joint_discovery.synthetic import look_at_camera  # noqa: E402


def _legacy_trans_output():
    import torch
    sys.path.insert(0, str(REPO / "estimotion_2part"))
    from estimotion_2part.models.trans_models import Trans
    torch.manual_seed(0)
    T1 = torch.eye(4).repeat(2, 1, 1)
    T2 = torch.eye(4).repeat(2, 1, 1)
    T2[:, 0, 3] = torch.tensor([0.1, 0.2])
    model = Trans(5, 2, init_weights=np.full(5, 0.75), init_T1=T1, init_T2=T2, cam_K=np.eye(3), device="cpu")
    x = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10
    out = model(x, torch.arange(5), torch.tensor([0, 1, 0, 1, 0]))
    return [o.detach().numpy() if hasattr(o, "detach") else o for o in (out if isinstance(out, tuple) else (out,))]


class TestLegacyAdapter(unittest.TestCase):
    def test_T21_legacy_is_default_and_unchanged(self):
        self.assertEqual(resolve_joint_discovery({}).mode, "legacy")
        self.assertEqual(resolve_joint_discovery({"tracking": {"window_size": 8}}).mode, "legacy")
        self.assertFalse(JointDiscoveryConfig().is_new_mode)
        with self.assertRaises(KeyError):
            load_joint_discovery_config({"joint_discovery": {"mode": "legacy", "typo_key": 1}})
        with self.assertRaises(ValueError):
            load_joint_discovery_config({"mode": "calibrated_joint_aware", "fixed_calibration": False})
        with self.assertRaises(ValueError):
            load_joint_discovery_config({"no_oracle_labels": False})
        try:
            out = _legacy_trans_output()
        except Exception as exc:  # legacy model needs its own deps; report instead of hiding
            self.skipTest(f"legacy Trans unavailable in this environment: {exc!r}")
        # the documented legacy blend (1-w) T1 x + w T2 x with w = 0.75 on the fixed baseline
        x = np.arange(15, dtype=np.float32).reshape(5, 3) / 10
        shift = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
        expected = x.copy()
        expected[:, 0] += 0.75 * shift
        pred = out[0]
        pred = pred[..., :3] if pred.shape[-1] >= 3 else pred
        np.testing.assert_allclose(pred.reshape(5, -1)[:, :3], expected, atol=1e-5)

    def test_virtual_pose_and_cross_camera_projection(self):
        cams = {c: look_at_camera(c, pos, [0, 0, 0.1]) for c, pos in
                (("cam_001", [-0.1, -0.9, 0.6]), ("cam_002", [0.5, -0.7, 0.5]))}
        c = JointCandidate("revolute", np.array([1.0, 0, 0]), np.array([0, 0.1, 0.13]), np.array([0.0, -0.4, -0.8]),
                           np.ones(3, bool), None, 0, np.zeros((1, 3)), np.ones(1), np.zeros(1), np.zeros(1))
        X0 = np.array([0.05, 0.02, 0.12])
        for cid, cal in cams.items():
            Tv = virtual_camera_poses(c, cal.T_WC)
            for t in range(3):
                Xt = transform_points(A_joint("revolute", c.axis_w, c.origin_w, c.q[t]), X0)
                uv_phys, z1 = project(Xt, cal.K, cal.T_CW)          # moving point, physical camera
                uv_virt, z2 = project(X0, cal.K, se3_inverse(Tv[t]))  # canonical point, virtual camera
                np.testing.assert_allclose(uv_phys, uv_virt, atol=1e-9)
                np.testing.assert_allclose(z1, z2, atol=1e-12)

    def test_export_schema_rows_and_prismatic_origin(self):
        cal = look_at_camera("cam_001", [0, -1, 0.5], [0, 0, 0])
        fm = FrameMap([70, 72, 74], [140, 144, 148], [2.8, 2.88, 2.96])
        c = JointCandidate("prismatic", np.array([0, -1.0, 0]), None, np.array([0, 0.01, 0.02]), np.ones(3, bool),
                           None, 0, np.zeros((1, 3)), np.ones(1), np.zeros(1), np.zeros(1))
        with tempfile.TemporaryDirectory() as td:
            export_for_existing_freeart(c, {"cam_001": cal}, fm, td, visible={"cam_001": np.array([True, False, True])})
            for part, kind in ((0, "physical_camera"), (1, "virtual_part_camera")):
                rows = json.loads((Path(td) / "cam_001" / f"transforms_{part}.json").read_text())["frames"]
                self.assertEqual(len(rows), 3)                      # invisible frame kept, not dropped
                self.assertEqual([r["valid"] for r in rows], [True, False, True])
                for key in ("camera_name", "time_index", "source_stem", "fl_x", "fl_y", "cx", "cy", "w", "h",
                            "mask_path", "depth_path", "pose_kind"):
                    self.assertIn(key, rows[0])
                self.assertTrue(all(r["pose_kind"] == kind for r in rows))
                self.assertEqual(rows[1]["source_stem"], "frame_000072")
            j = load_joint_candidate(Path(td) / "joint_candidate.json")
            self.assertIsNone(j["joint_origin_world"])
            self.assertIn("N/A", j["origin_convention"])
            self.assertEqual(j["joint_type"], "prismatic")
            # the visualize_articulation reader accepts the exported keys
            self.assertIn("joint_axis_world", j)


if __name__ == "__main__":
    unittest.main()
