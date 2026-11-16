"""Synthetic physical validation of Phase C (sliding contact to a fixed canonical moving surface).

Revolute lid lifted by a nearly straight TCP path and a prismatic drawer pushed while the
TCP slides sideways over its face.  For both: finite non-zero gradients to the moving
transform, descent towards the true motion, a worse residual for wrong motion, no
gradient on the static branch, no force along the sliding directions (no rigid grasp),
and a canonical surface that does not depend on the candidate transform.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from estimotion_2part.contact_surface_loss import BRANCH_T2, moving_surface_contact_loss  # noqa: E402
from estimotion_2part.window_priors import ContactTerm  # noqa: E402

D = torch.float64


def rot_x(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    one, zero = torch.ones_like(theta), torch.zeros_like(theta)
    R = torch.stack([torch.stack([one, zero, zero]), torch.stack([zero, c, -s]), torch.stack([zero, s, c])])
    return R.permute(2, 0, 1) if R.ndim == 3 else R


def hinge_T(theta, hinge=torch.tensor([0.0, 0.0, 0.0], dtype=D)):
    """Rotation about the X axis through ``hinge`` (lid hinge), one 4x4 per angle."""
    theta = torch.atleast_1d(theta)
    R = rot_x(theta)
    T = torch.eye(4, dtype=D).repeat(len(theta), 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = hinge - torch.einsum("kij,j->ki", R, hinge)
    return T


def lid_surface(n_x=41, n_y=81):
    """Lid underside in its closed pose: plane z=0, x in [-0.1, 0.1] (hinge axis), y in [0, 0.2]."""
    xs, ys = np.meshgrid(np.linspace(-0.1, 0.1, n_x), np.linspace(0.0, 0.2, n_y))
    return torch.tensor(np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1), dtype=D)


def revolute_scene():
    """The TCP moves on a nearly straight upward line y = 0.15 + small drift and slides on the lid."""
    theta = torch.deg2rad(torch.tensor([5.0, 12.0, 20.0, 28.0, 36.0, 44.0], dtype=D))
    y_line = 0.12 + 0.002 * torch.arange(len(theta), dtype=D)  # nearly linear TCP path, r <= 0.18 m on the lid
    # Lid point at distance r from the hinge: (x, r cos, r sin); the TCP at y=y_line touches it.
    r = y_line / torch.cos(theta)
    x = torch.linspace(-0.03, 0.03, len(theta), dtype=D)  # also slides along the hinge direction
    contact = torch.stack([x, r * torch.cos(theta), r * torch.sin(theta)], 1)
    return lid_surface(), theta, contact


def drawer_face(n_y=61, n_z=31):
    """Drawer front face in the reference pose: plane x=0, y in [-0.15, 0.15], z in [0, 0.1]."""
    ys, zs = np.meshgrid(np.linspace(-0.15, 0.15, n_y), np.linspace(0.0, 0.1, n_z))
    return torch.tensor(np.stack([np.zeros(ys.size), ys.ravel(), zs.ravel()], 1), dtype=D)


def prismatic_scene():
    """The drawer translates along -X; the TCP pushes the face while sliding sideways in Y."""
    d = torch.tensor([0.0, -0.02, -0.04, -0.06, -0.08], dtype=D)
    # Off-grid contact points (face samples every 5 mm in Y, 3.3 mm in Z).
    contact = torch.stack([d, torch.linspace(-0.061, 0.058, len(d), dtype=D), torch.full_like(d, 0.0517)], 1)
    T = torch.eye(4, dtype=D).repeat(len(d), 1, 1)
    T[:, 0, 3] = d
    return drawer_face(), T, contact


def contact_loss(canonical, T, contact, sigma=0.01, delta=0.01):
    return moving_surface_contact_loss(canonical_moving_xyz=canonical, moving_T_ref_to_t=T,
                                       tool_contact_camera_xyz=contact, valid_contact_t=torch.ones(len(contact), dtype=torch.bool),
                                       sigma_m=sigma, robust_delta_m=delta)


class RevoluteSlidingContactTests(unittest.TestCase):
    def setUp(self):
        self.canonical, self.theta, self.contact = revolute_scene()

    def test_true_motion_fits_with_sliding_material_point(self):
        value, info = contact_loss(self.canonical, hinge_T(self.theta), self.contact)
        self.assertLess(info["p95_m"], 0.003)  # within surface sampling (5 x 2.5 mm grid)
        self.assertEqual(info["distinct_contact_samples"], len(self.theta))  # a different lid point every frame
        # A rigid grasp would need one canonical point: here its back-projection moves by > 4 cm.
        x0 = torch.einsum("kij,kj->ki", torch.linalg.inv(hinge_T(self.theta)),
                          torch.cat([self.contact, torch.ones(len(self.contact), 1, dtype=D)], 1))[:, :3]
        self.assertGreater(float(torch.linalg.norm(x0 - x0[0], dim=1).max()), 0.04)

    def test_gradient_and_descent_towards_true_angle(self):
        theta = (self.theta + torch.deg2rad(torch.tensor(8.0, dtype=D))).clone().requires_grad_(True)
        start = None
        optimizer = torch.optim.Adam([theta], lr=0.01)
        for step in range(300):
            optimizer.zero_grad()
            value, _ = contact_loss(self.canonical, hinge_T(theta), self.contact)
            value.backward()
            if step == 0:
                start = float(value)
                self.assertTrue(torch.isfinite(theta.grad).all() and float(theta.grad.abs().min()) > 0)
            optimizer.step()
        final, _ = contact_loss(self.canonical, hinge_T(theta.detach()), self.contact)
        self.assertLess(float(final), 0.05 * start)
        error = torch.rad2deg((theta.detach() - self.theta).abs()).max()
        self.assertLess(float(error), 1.0)

    def test_wrong_motion_increases_residual(self):
        _, good = contact_loss(self.canonical, hinge_T(self.theta), self.contact)
        for offset in (-10.0, 10.0):
            _, bad = contact_loss(self.canonical, hinge_T(self.theta + torch.deg2rad(torch.tensor(offset, dtype=D))),
                                  self.contact)
            self.assertGreater(bad["median_m"], good["median_m"] + 0.01)

    def test_no_rigid_grasp_along_hinge(self):
        """Jointly optimise the angle and a shift along the hinge axis (a sliding direction).

        A rigid grasp would pull the shift back to 0; sliding contact leaves it where it
        started up to the canonical sample spacing (5 mm in X), the only tangential effect
        of a point-sampled surface.  Plain gradient descent measures the force field itself
        (Adam would rescale the sampling ripple to full-size steps; in the pipeline the
        visual term constrains these directions).
        """
        theta = (self.theta + torch.deg2rad(torch.tensor(6.0, dtype=D))).clone().requires_grad_(True)
        shift = torch.full((len(self.theta),), 0.02, dtype=D, requires_grad=True)
        optimizer = torch.optim.SGD([theta, shift], lr=2e-4)
        for _ in range(1500):
            optimizer.zero_grad()
            T = hinge_T(theta).clone()
            T[:, 0, 3] = T[:, 0, 3] + shift
            contact_loss(self.canonical, T, self.contact)[0].backward()
            optimizer.step()
        self.assertLess(float(torch.rad2deg((theta.detach() - self.theta).abs()).max()), 1.0)
        self.assertLess(float((shift.detach() - 0.02).abs().max()), 0.005)


class PrismaticSlidingContactTests(unittest.TestCase):
    def setUp(self):
        self.canonical, self.T, self.contact = prismatic_scene()

    def test_true_motion_and_sliding(self):
        _, info = contact_loss(self.canonical, self.T, self.contact)
        self.assertLess(info["p95_m"], 0.003)  # within face sampling
        self.assertEqual(info["distinct_contact_samples"], len(self.contact))

    def test_gradient_descent_on_push_axis_only(self):
        d = (self.T[:, :3, 3] + torch.tensor([0.02, 0.03, -0.01], dtype=D)).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([d], lr=0.002)
        start = None
        for step in range(400):
            optimizer.zero_grad()
            T = torch.eye(4, dtype=D).repeat(len(d), 1, 1)
            T[:, :3, 3] = d
            value, _ = contact_loss(self.canonical, T, self.contact)
            value.backward()
            if step == 0:
                start = float(value)
                self.assertTrue(torch.isfinite(d.grad).all())
                self.assertGreater(float(d.grad[:, 0].abs().min()), 0)
            optimizer.step()
        self.assertLess(float(value), 0.05 * start)
        np.testing.assert_allclose(d.detach()[:, 0].numpy(), self.T[:, 0, 3].numpy(), atol=1e-3)
        # Sliding directions are not pinned (no rigid grasp): the lateral offsets keep their
        # start values up to the face sample spacing.
        np.testing.assert_allclose(d.detach()[:, 1].numpy(), 0.03, atol=0.005)
        np.testing.assert_allclose(d.detach()[:, 2].numpy(), -0.01, atol=0.0034)

    def test_wrong_motion_increases_residual(self):
        _, good = contact_loss(self.canonical, self.T, self.contact)
        wrong = self.T.clone()
        wrong[:, 0, 3] = 0.0  # static hypothesis
        _, bad = contact_loss(self.canonical, wrong, self.contact)
        self.assertGreater(bad["median_m"], good["median_m"] + 0.03)


class ContactTermIsolationTests(unittest.TestCase):
    """Full ContactTerm: moving branch only, static transform untouched, canonical frozen."""

    OFFSET = torch.tensor([0.0, 0.0, 1.0], dtype=D)  # scene 1 m in front of the camera (positive depth)

    def to_camera(self, T):
        shift = torch.eye(4, dtype=D)
        shift[:3, 3] = self.OFFSET
        return (shift @ T.to(D) @ torch.linalg.inv(shift)).float()

    def setUp(self):
        canonical, self.theta, contact = revolute_scene()
        base = drawer_face() + torch.tensor([0.0, 0.0, -0.3], dtype=D)  # static body, away from the contact
        self.source = (torch.cat([canonical, base]) + self.OFFSET).float()
        self.weights = torch.cat([torch.ones(len(canonical)), torch.zeros(len(base))])  # >0.5 -> T2 moving
        self.contact = (contact + self.OFFSET).float()

    def make(self):
        term = ContactTerm(self.contact, torch.ones(len(self.contact), dtype=torch.bool), weight=1.0, sigma_m=0.01,
                           robust_delta_m=0.01, support_max_distance_m=0.05)
        T_static = torch.eye(4).repeat(len(self.theta), 1, 1)
        T_moving = self.to_camera(hinge_T(self.theta + 0.05))
        self.assertEqual(term.prepare(T_static, T_moving, self.source, self.weights), BRANCH_T2)
        return term, T_static, T_moving

    def test_static_transform_gets_no_gradient(self):
        term, T_static, T_moving = self.make()
        T1 = T_static.clone().requires_grad_(True)
        T2 = T_moving.clone().requires_grad_(True)
        value, _ = term.loss(T1, T2)
        value.backward()
        self.assertTrue(T1.grad is None or float(T1.grad.abs().max()) == 0.0)
        self.assertGreater(float(T2.grad.abs().max()), 0.0)

    def test_canonical_does_not_follow_candidate_transform(self):
        term, T_static, T_moving = self.make()
        before = term.canonical.clone()
        term.rebuild(self.source, self.weights)  # rebuild depends on observations and weights only
        self.assertTrue(torch.equal(before, term.canonical))
        # The residual stays non-trivial for a wrong candidate after rebuilding.
        wrong = self.to_camera(hinge_T(self.theta + 0.3))
        _, info = term.loss(T_static, wrong)
        self.assertGreater(info["median_m"], 0.02)

    def test_weighted_contribution_reported(self):
        term, T_static, T_moving = self.make()
        info = term.finalize(T_static, T_moving)
        self.assertEqual(info["contact_weight"], 1.0)
        self.assertAlmostEqual(info["L_contact_weighted_final"], info["L_contact_final"])
        self.assertEqual(info["N_contact_final"], len(self.theta))


if __name__ == "__main__":
    unittest.main()
