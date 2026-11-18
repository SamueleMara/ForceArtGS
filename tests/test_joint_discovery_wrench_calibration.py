"""Stage D2/D3: wrench calibration contract and readiness gates (§10 item 11)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.rigid_kinematics import exp_so3  # noqa: E402
from estimotion_2part.joint_discovery.wrench_calibration import G_BASE, Provenanced, WrenchCalibration  # noqa: E402


def verified(R_EE_S, tare=(0.0, 0.0, 0.0)):
    c = WrenchCalibration()
    c.R_EE_S = Provenanced(np.asarray(R_EE_S).tolist(), "VERIFIED", "test")
    c.recorded_quantity = Provenanced("object_on_robot", "VERIFIED", "test")
    c.tare_S = Provenanced(list(tare), "VERIFIED", "test")
    return c


class TestWrenchCalibration(unittest.TestCase):
    def test_rotated_sensor_sign_and_tare(self):
        rng = np.random.default_rng(0)
        R_EE_S = exp_so3(np.array([0.2, -0.5, 1.1]))
        R_W_EE = exp_so3(rng.normal(size=(5, 3)))
        f_obj = rng.normal(size=(5, 3))
        tare = np.array([0.3, -0.1, 0.2])
        F_rec = -np.einsum("ji,nj->ni", R_EE_S, np.einsum("nji,nj->ni", R_W_EE, f_obj)) + tare
        out = verified(R_EE_S, tare).force_on_object_world(F_rec, R_W_EE)
        np.testing.assert_allclose(out["f_object_w"], f_obj, atol=1e-12)
        self.assertFalse(np.allclose(out["f_object_w_raw_uncompensated"], f_obj))   # raw kept separately
        # wrong (identity) rotation gives a wrong force: the rotation is not optional
        bad = verified(np.eye(3), tare).force_on_object_world(F_rec, R_W_EE)["f_object_w"]
        self.assertGreater(np.abs(bad - f_obj).max(), 0.1)

    def test_unknown_rotation_or_sign_refuses(self):
        c = WrenchCalibration()
        with self.assertRaises(ValueError):
            c.force_on_object_world(np.zeros((1, 3)), np.eye(3)[None])
        c.R_EE_S = Provenanced(np.eye(3).tolist(), "DECLARED", "user confirmation")
        c.tare_S = Provenanced([0, 0, 0], "ESTIMATED")
        with self.assertRaises(ValueError):
            c.force_on_object_world(np.zeros((1, 3)), np.eye(3)[None])
        r = c.readiness()
        self.assertFalse(r["force_selection_ready"])
        self.assertEqual(r["gates"]["frame_rotation"]["status"], "UNVERIFIED")
        self.assertEqual(r["gates"]["sign"]["status"], "FAIL")
        self.assertEqual(r["gates"]["time_sync"]["status"], "UNVERIFIED")

    def test_scene_metadata_is_declared_not_verified(self):
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "metadata"
            m.mkdir()
            (m / "robot_tool.json").write_text(json.dumps(dict(dataset="x", tool="tip", wrench_frame=dict(
                axes="ee", R_ee_force_sensor=np.eye(3).tolist(), alignment_source="User confirmation",
                torque_origin="recorded_origin_unspecified; not assumed"))))
            (m / "contact_force_convention.json").write_text(json.dumps(dict(force_acting_on="robot",
                                                                              confirmation_source="operator")))
            c = WrenchCalibration.from_scene_metadata(td)
        self.assertEqual(c.R_EE_S.status, "DECLARED")
        self.assertEqual(c.recorded_quantity.value, "object_on_robot")
        r = c.readiness(use_torque=True)
        self.assertFalse(r["force_selection_ready"])
        self.assertEqual(set(r["failing"]) >= {"frame_rotation", "sign", "tare", "gravity", "time_sync",
                                               "torque_transport"}, True)

    def test_tare_drift_needs_two_windows(self):
        rng = np.random.default_rng(1)
        F = rng.normal(0, 0.02, (200, 3)) + [0.5, 0, 0]
        c = WrenchCalibration()
        a = np.zeros(200, bool)
        a[:50] = True
        c.estimate_tare(F, a)
        self.assertEqual(c.tests["tare"]["status"], "UNVERIFIED")
        b = np.zeros(200, bool)
        b[150:] = True
        c.estimate_tare(F, a, second_mask=b)
        self.assertEqual(c.tests["tare"]["status"], "PASS")
        F2 = F.copy()
        F2[150:] += [0.4, 0, 0]                                    # zero drifted by 0.4 N
        c.estimate_tare(F2, a, second_mask=b)
        self.assertEqual(c.tests["tare"]["status"], "FAIL")

    def test_gravity_multipose_and_orientation_bound(self):
        rng = np.random.default_rng(2)
        R = exp_so3(rng.normal(size=(60, 3)) * 0.6)
        m = 0.4
        F = np.array([0.1, 0.2, -0.3]) + m * np.einsum("nji,j->ni", R, G_BASE) + rng.normal(0, 0.01, (60, 3))
        c = WrenchCalibration()
        c.R_EE_S = Provenanced(np.eye(3).tolist(), "DECLARED")
        c.check_static_multipose_gravity(F, R, np.ones(60, bool))
        self.assertEqual(c.tests["multipose_gravity"]["status"], "PASS")
        self.assertAlmostEqual(c.tests["multipose_gravity"]["mass_kg"], m, delta=0.02)
        c2 = WrenchCalibration()
        Rc = np.tile(np.eye(3), (30, 1, 1))
        c2.check_static_multipose_gravity(np.zeros((30, 3)), Rc, np.ones(30, bool))
        self.assertEqual(c2.tests["multipose_gravity"]["status"], "UNVERIFIED")
        c2.tool_mass_upper_bound_kg = Provenanced(0.5, "DECLARED")
        c2.estimate_tare(rng.normal(0, 0.05, (60, 3)), np.r_[np.ones(30, bool), np.zeros(30, bool)],
                         second_mask=np.r_[np.zeros(30, bool), np.ones(30, bool)])
        c2.check_orientation_constancy(Rc, np.ones(30, bool))
        self.assertEqual(c2.readiness()["gates"]["gravity"]["status"], "PASS")   # constant pose bound

    def test_torque_transport_lever_arm(self):
        c = verified(np.eye(3))
        c.p_S_EE = Provenanced([0.0, 0.0, 0.05], "VERIFIED")
        f_obj = np.array([[1.0, 0.0, 0.0]])
        F_rec = -f_obj                                   # identity rotations, zero tare
        tau = c.torque_on_object_about_tcp_world(F_rec, np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3)),
                                                 np.array([[0.0, 0.0, 0.2]]))
        # (p_S - p_TCP) x f = (0,0,-0.15) x (1,0,0) = (0,-0.15,0)
        np.testing.assert_allclose(tau, [[0.0, -0.15, 0.0]], atol=1e-12)
        c.p_S_EE = Provenanced(None, "UNKNOWN")
        with self.assertRaises(ValueError):
            c.torque_on_object_about_tcp_world(F_rec, np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3)), np.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main()
