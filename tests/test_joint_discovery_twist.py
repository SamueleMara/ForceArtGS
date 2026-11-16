"""T03, T04: instantaneous velocity fields, finite-motion consistency, rigid-flow diagnostic."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery import rigid_kinematics as rk  # noqa: E402
from estimotion_2part.joint_discovery.candidates import kabsch, rigid_flow_omega  # noqa: E402


class TestVelocityFields(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.u = rk.normalize([0.1, 0.2, 1.0])
        self.o = np.array([0.2, -0.1, 0.05])

    def test_T03_revolute_tangential_only_and_heterogeneous(self):
        X = self.rng.normal(size=(40, 3)) * 0.2
        thd = 0.8
        v = rk.body_velocity("revolute", self.u, self.o, thd, X)
        dec = rk.decompose_revolute_velocity(self.u, self.o, X, v)
        np.testing.assert_allclose(dec["radial"], 0, atol=1e-12)
        np.testing.assert_allclose(dec["axial"], 0, atol=1e-12)
        np.testing.assert_allclose(dec["tangential"], dec["radius"] * thd, atol=1e-12)
        # tangent directions differ between points; speed grows with radius
        dirs = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.assertGreater(np.ptp(dirs @ dirs[0]), 0.5)

    def test_T03_exact_finite_motion_for_60_degrees(self):
        X0 = self.rng.normal(size=(30, 3)) * 0.2
        qa, qb = 0.1, 0.1 + np.radians(60)
        Xa = rk.apply_joint("revolute", self.u, self.o, qa, X0)
        Xb = rk.apply_joint("revolute", self.u, self.o, qb, X0)
        Ab, Aa = rk.A_revolute(self.u, self.o, qb), rk.A_revolute(self.u, self.o, qa)
        np.testing.assert_allclose(rk.transform_points(Ab @ rk.se3_inverse(Aa), Xa), Xb, atol=1e-12)
        # the chord is NOT the instantaneous tangent: it has a clear radial component
        chord = Xb - Xa
        dec = rk.decompose_revolute_velocity(self.u, self.o, Xa, chord)
        far = dec["radius"] > 0.05
        self.assertGreater(np.median(np.abs(dec["radial"][far]) / dec["radius"][far]), 0.3)

    def test_T04_prismatic_common_velocity(self):
        u = rk.normalize([1.0, -1.0, 0.2])
        X = self.rng.normal(size=(25, 3))
        v = rk.body_velocity("prismatic", u, None, 0.05, X)
        dec = rk.decompose_prismatic_velocity(u, v)
        np.testing.assert_allclose(dec["perpendicular_norm"], 0, atol=1e-14)
        np.testing.assert_allclose(dec["parallel"], 0.05)
        # a perturbed point has a measurable perpendicular residual
        v2 = v.copy()
        v2[0] += 0.01 * rk.tangent_basis(u)[0]
        self.assertAlmostEqual(rk.decompose_prismatic_velocity(u, v2)["perpendicular_norm"][0], 0.01, places=12)

    def test_pairwise_identity_and_rigid_flow(self):
        X = self.rng.normal(size=(60, 3)) * 0.3
        thd = 0.5
        v = rk.body_velocity("revolute", self.u, self.o, thd, X)
        i, j = 3, 17
        np.testing.assert_allclose(v[i] - v[j], thd * np.cross(self.u, X[i] - X[j]), atol=1e-12)
        r = rigid_flow_omega(X, v)
        np.testing.assert_allclose(r["omega"], thd * self.u, atol=1e-10)
        self.assertEqual(r["rank"], 3)
        # pure translation -> omega ~ 0 (no rotation is inferred from noise-free translation)
        vt = rk.body_velocity("prismatic", self.u, None, 0.2, X)
        self.assertLess(np.linalg.norm(rigid_flow_omega(X, vt)["omega"]), 1e-10)
        # collinear points cannot determine rotation about their own line: rank 2
        L = np.outer(np.linspace(0, 1, 20), [1, 0, 0])
        self.assertLess(rigid_flow_omega(L, np.zeros_like(L))["rank"], 3)

    def test_kabsch_recovers_hinge_motion(self):
        X = self.rng.normal(size=(20, 3))
        A = rk.A_revolute(self.u, self.o, 0.4)
        R, t = kabsch(X, rk.transform_points(A, X))
        np.testing.assert_allclose(R, A[:3, :3], atol=1e-10)
        np.testing.assert_allclose(t, A[:3, 3], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
