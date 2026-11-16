"""T01, T02, T23: exact SE(3) joint transforms, units/gauge, and autograd gradient checks."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery import rigid_kinematics as rk  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class TestSE3(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_T01_revolute_equals_twist_exponential(self):
        for _ in range(20):
            u = rk.normalize(self.rng.normal(size=3))
            o = self.rng.normal(size=3) * 0.5  # arbitrary non-origin hinge
            theta = self.rng.uniform(-3, 3, 7)
            A = rk.A_revolute(u, o, theta)
            xi = np.repeat(rk.revolute_twist(u, o)[None], len(theta), 0)
            self.assertLess(np.abs(A - rk.exp_se3(xi, theta)).max(), 1e-12)
            self.assertTrue(rk.is_rigid(A))
            # zero pitch: omega . v == 0, twist order [omega; v]
            tw = rk.revolute_twist(u, o)
            self.assertAlmostEqual(float(tw[:3] @ tw[3:]), 0.0, places=12)
            np.testing.assert_allclose(tw[:3], u)
            # points on the hinge line are fixed; distance to the line preserved
            for s in (-1.0, 0.3, 2.0):
                np.testing.assert_allclose(rk.transform_points(A, o + s * u), np.broadcast_to(o + s * u, (7, 3)), atol=1e-12)

    def test_T01_identity_inverse_and_axis_flip(self):
        u = rk.normalize([0.2, -0.4, 0.9])
        o = np.array([0.1, 0.5, -0.3])
        np.testing.assert_allclose(rk.A_revolute(u, o, 0.0), np.eye(4), atol=1e-15)
        A = rk.A_revolute(u, o, 0.7)
        np.testing.assert_allclose(rk.se3_inverse(A) @ A, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(rk.A_revolute(u, o, -0.7), rk.se3_inverse(A), atol=1e-12)
        # (u, q) ~ (-u, -q): identical motion
        np.testing.assert_allclose(rk.A_revolute(-u, o, -0.7), A, atol=1e-12)
        # hinge origin moves along the axis without changing the transform (gauge)
        np.testing.assert_allclose(rk.A_revolute(u, o + 3.0 * u, 0.7), A, atol=1e-12)
        self.assertLess(rk.axis_angle_deg(u, -u), 1e-6)

    def test_T01_derivative_matches_finite_difference(self):
        u = rk.normalize([1.0, 2.0, -0.5])
        o = np.array([0.3, -0.2, 0.1])
        X = np.array([0.4, 0.4, 0.4])
        th, h = 0.9, 1e-6
        fd = (rk.apply_joint("revolute", u, o, th + h, X) - rk.apply_joint("revolute", u, o, th - h, X)) / (2 * h)
        xt = rk.apply_joint("revolute", u, o, th, X)
        np.testing.assert_allclose(fd, rk.body_velocity("revolute", u, o, 1.0, xt), atol=1e-8)

    def test_T02_prismatic_exact_translation_units(self):
        u = rk.normalize([0.0, -2.0, 0.0])
        d = np.array([0.0, 0.05, 0.12])  # metres
        A = rk.A_prismatic(u, d)
        self.assertTrue(rk.is_rigid(A))
        np.testing.assert_allclose(A[:, :3, :3], np.broadcast_to(np.eye(3), (3, 3, 3)))
        np.testing.assert_allclose(A[:, :3, 3], d[:, None] * u)
        np.testing.assert_allclose(A[0], np.eye(4))  # d_ref = 0
        np.testing.assert_allclose(np.linalg.norm(u), 1.0)
        # omega = 0 everywhere: every point has the same velocity
        P = self.rng.normal(size=(5, 3))
        v = rk.body_velocity("prismatic", u, None, 0.3, P)
        np.testing.assert_allclose(v, np.broadcast_to(0.3 * u, (5, 3)))

    def test_exp_log_so3_roundtrip_including_pi(self):
        rv = self.rng.normal(size=(50, 3))
        rv = rv / np.linalg.norm(rv, axis=1, keepdims=True) * self.rng.uniform(0, np.pi - 1e-3, (50, 1))
        np.testing.assert_allclose(rk.log_so3(rk.exp_so3(rv)), rv, atol=1e-9)
        R = rk.exp_so3(np.array([0, 0, np.pi]))
        np.testing.assert_allclose(rk.exp_so3(rk.log_so3(R)), R, atol=1e-9)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_T23_gradcheck(self):
        dt = torch.float64
        u = torch.tensor([0.3, -0.5, 0.8], dtype=dt, requires_grad=True)
        o = torch.tensor([0.1, 0.2, -0.1], dtype=dt, requires_grad=True)
        th = torch.tensor([0.4, -1.1], dtype=dt, requires_grad=True)
        d = torch.tensor([0.05, 0.2], dtype=dt, requires_grad=True)
        p = torch.tensor([0.5, 0.1, 0.3], dtype=dt, requires_grad=True)
        v = torch.tensor([0.02, -0.01, 0.03], dtype=dt, requires_grad=True)
        n = torch.tensor([0.0, 0.6, 0.8], dtype=dt)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b, c: rk.A_revolute(a, b, c), (u, o, th)))
        self.assertTrue(torch.autograd.gradcheck(lambda a, c: rk.A_prismatic(a, c), (u, d)))
        # contact normal-slip n . (v_tcp - v_M(p))
        qd = torch.tensor(0.7, dtype=dt, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, b, q_, p_, v_: (n * rk.relative_slip("revolute", a, b, q_, p_, v_)).sum(-1), (u, o, qd, p, v)))
        # reprojection of a joint-moved point through a fixed pinhole camera
        K = torch.tensor([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]], dtype=dt)
        X = torch.tensor([0.1, 0.05, 1.0], dtype=dt, requires_grad=True)

        def proj(a, b, q_, x_):
            y = rk.apply_joint("revolute", a, b, q_, x_)
            z = y[..., 2:3]
            return (K[:2, :2] @ (y[..., :2] / z).unsqueeze(-1)).squeeze(-1) + K[:2, 2]
        self.assertTrue(torch.autograd.gradcheck(proj, (u, o, torch.tensor(0.2, dtype=dt, requires_grad=True), X)))
        # gradient at theta = 0 is finite (Taylor branch)
        t0 = torch.tensor(0.0, dtype=dt, requires_grad=True)
        rk.A_revolute(u, o, t0).sum().backward()
        self.assertTrue(torch.isfinite(t0.grad))


if __name__ == "__main__":
    unittest.main()
