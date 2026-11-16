import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from estimotion_2part.contact_surface_loss import (  # noqa: E402
    BRANCH_T1, BRANCH_T2, moving_surface_contact_loss, resolve_motion_branch,
)
from estimotion_2part.window_priors import ContactTerm  # noqa: E402
from estimotion_2part.models.trans_models import TransformLossComputer  # noqa: E402


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def door_panel(n_len=201, n_h=5):
    """Panel along +X from a hinge at the origin (z-axis hinge), thin in Y."""
    xs = np.linspace(0.0, 1.0, n_len)
    zs = np.linspace(-0.1, 0.1, n_h)
    X, Z = np.meshgrid(xs, zs)
    return np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)


def revolute_case():
    """Door opens by theta_t; the TCP moves on the straight line y = 0.3 and slides on it."""
    thetas = np.deg2rad([20, 30, 40, 50, 60])
    T = np.stack([rot_z(t) for t in thetas])
    contact = np.stack([[0.3 / np.tan(t), 0.3, 0.0] for t in thetas])  # line/panel intersection
    return door_panel(), T, contact


def loss(canonical, T, contact, valid=None, sigma=0.01, delta=0.02, dtype=torch.float64):
    T = torch.as_tensor(T, dtype=dtype)
    valid = np.ones(len(contact), bool) if valid is None else valid
    return moving_surface_contact_loss(canonical_moving_xyz=torch.as_tensor(canonical, dtype=dtype),
                                       moving_T_ref_to_t=T, tool_contact_camera_xyz=torch.as_tensor(contact, dtype=dtype),
                                       valid_contact_t=torch.as_tensor(valid), sigma_m=sigma, robust_delta_m=delta)


class ContactLossTests(unittest.TestCase):
    def test_revolute_sliding_straight_line_contact(self):
        canonical, T, contact = revolute_case()
        self.assertTrue(np.allclose(contact[:, 1], 0.3))  # TCP path is a straight line
        value, info = loss(canonical, T, contact)
        self.assertLess(max(info["distances_m"]), 0.003)  # within panel sampling (5 mm spacing)
        self.assertLess(float(value), 0.05)
        # The canonical contact sample changes over time: no fixed material point.
        self.assertGreaterEqual(info["distinct_contact_samples"], 4)
        x_canonical = canonical[info["argmin_canonical"], 0]
        self.assertTrue(np.all(np.diff(x_canonical) < 0))

    def test_fixed_point_hypothesis_is_not_imposed(self):
        canonical, T, contact = revolute_case()
        # A rigid-grasp model p(t) = T(t) x0 would need constant x0; here it is not constant.
        x0 = np.linalg.inv(T) @ np.c_[contact, np.ones(len(contact))][..., None]
        self.assertGreater(np.ptp(x0[:, 0, 0]), 0.2)

    def test_prismatic_near_zero(self):
        canonical = door_panel()
        T = np.repeat(np.eye(4)[None], 4, axis=0)
        T[:, 1, 3] = [0.0, 0.05, 0.1, 0.15]  # drawer slides along +Y
        contact = np.stack([[0.5, y, 0.0] for y in T[:, 1, 3]])
        _, info = loss(canonical, T, contact)
        self.assertLess(max(info["distances_m"]), 1e-9)

    def test_wrong_hypothesis_scores_worse(self):
        canonical, T, contact = revolute_case()
        _, good = loss(canonical, T, contact)
        wrong = np.repeat(np.eye(4)[None], len(T), axis=0)  # static-part hypothesis
        _, bad = loss(canonical, wrong, contact)
        self.assertGreater(bad["median_m"], 10 * good["median_m"] + 0.05)

    def test_gradient_matches_finite_differences_and_descends(self):
        canonical, T, contact = revolute_case()
        T = T.copy()
        T[:, 0, 3] += 0.03  # perturb
        Tt = torch.tensor(T, dtype=torch.float64, requires_grad=True)
        value, _ = loss(canonical, Tt, contact)
        value.backward()
        grad = Tt.grad.numpy()
        self.assertTrue(np.isfinite(grad).all() and np.abs(grad).sum() > 0)
        eps = 1e-6
        for k, i, j in [(0, 0, 3), (2, 1, 3), (4, 0, 1)]:
            Tp, Tm = T.copy(), T.copy()
            Tp[k, i, j] += eps
            Tm[k, i, j] -= eps
            fd = (float(loss(canonical, Tp, contact)[0]) - float(loss(canonical, Tm, contact)[0])) / (2 * eps)
            self.assertAlmostEqual(fd, grad[k, i, j], delta=1e-4 * max(1.0, abs(fd)))
        stepped = torch.tensor(T - 1e-5 * grad, dtype=torch.float64)
        self.assertLess(float(loss(canonical, stepped, contact)[0]), float(value))

    def test_invalid_inputs_skipped_never_false_evidence(self):
        canonical, T, contact = revolute_case()
        noisy = canonical.copy()
        noisy[::7] = np.nan  # invalid depth samples
        bad_contact = contact.copy()
        bad_contact[1] = np.nan
        _, info = loss(noisy, T, bad_contact)
        self.assertEqual(info["N_contact"], 4)
        self.assertGreater(info["rejected_nonfinite_canonical"], 0)
        self.assertEqual(info["rejected_nonfinite_contact"], 1)
        Tt = torch.tensor(T, requires_grad=True)
        value, info = loss(canonical, Tt, contact, valid=np.zeros(len(T), bool))
        value.backward()
        self.assertEqual(info["status"], "skipped")
        self.assertEqual(float(value), 0.0)
        self.assertEqual(float(Tt.grad.abs().sum()), 0.0)
        value, info = loss(np.zeros((0, 3)), T, contact)
        self.assertEqual(info["status"], "skipped")

    def test_explicit_parameters_required(self):
        canonical, T, contact = revolute_case()
        with self.assertRaises(ValueError):
            loss(canonical, T, contact, sigma=None)
        with self.assertRaises(ValueError):
            ContactTerm(torch.zeros(1, 3), torch.ones(1, dtype=torch.bool), weight=1.0, sigma_m=None,
                        robust_delta_m=0.01, support_max_distance_m=0.05)


class ContactTermTests(unittest.TestCase):
    def setUp(self):
        canonical, T, contact = revolute_case()
        static = canonical + np.array([0.0, -0.5, 0.0])  # static frame, away from the contact
        self.source = torch.tensor(np.concatenate([canonical, static]), dtype=torch.float32)
        self.weights = torch.tensor(np.r_[np.ones(len(canonical)), np.zeros(len(static))], dtype=torch.float32)
        self.T_moving = torch.tensor(T, dtype=torch.float32)
        self.T_static = torch.eye(4).repeat(len(T), 1, 1)
        self.contact = torch.tensor(contact, dtype=torch.float32)

    def term(self):
        return ContactTerm(self.contact, torch.ones(len(self.contact), dtype=torch.bool), weight=1.0, sigma_m=0.01,
                           robust_delta_m=0.02, support_max_distance_m=0.05)

    def test_branch_resolved_from_motion_and_label_swap_invariant(self):
        term = self.term()
        self.assertEqual(term.prepare(self.T_static, self.T_moving, self.source, self.weights), BRANCH_T2)
        loss_a, _ = term.loss(self.T_static, self.T_moving)
        swapped = self.term()
        self.assertEqual(swapped.prepare(self.T_moving, self.T_static, self.source, 1 - self.weights), BRANCH_T1)
        loss_b, _ = swapped.loss(self.T_moving, self.T_static)
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=6)
        term.swap_labels()
        self.assertEqual(term.branch, BRANCH_T1)

    def test_unsupported_contact_surface_skipped(self):
        far = ContactTerm(self.contact + 1.0, torch.ones(len(self.contact), dtype=torch.bool), weight=1.0,
                          sigma_m=0.01, robust_delta_m=0.02, support_max_distance_m=0.05,
                          inherited_branch=BRANCH_T2)
        self.assertIsNone(far.prepare(self.T_static, self.T_moving, self.source, self.weights))
        self.assertEqual(far.info["status"], "skipped")
        self.assertIsNone(far.loss(self.T_static, self.T_moving)[0])

    def test_motion_contact_disagreement_is_ambiguous(self):
        # Contact located on the static frame while the other branch moves: evidence disagrees.
        static_contact = self.source[len(self.source) // 2 + 100:len(self.source) // 2 + 105]
        branch, info = resolve_motion_branch(self.T_static, self.T_moving, self.source, self.weights,
                                             static_contact, torch.ones(5, dtype=torch.bool))
        self.assertIsNone(branch)
        self.assertEqual(info["reason"], "motion_and_contact_evidence_disagree")

    def test_disabled_path_matches_baseline(self):
        computer = TransformLossComputer(loss_type="l1", cam_k_tensor=torch.eye(3), w_p=0.0, w_m=1.0, w_e=0.0,
                                         w_s=0.0, w_init=1.0)
        args = dict(pred_t1=torch.rand(8, 3), pred_t2=torch.rand(8, 3), w_sel=torch.rand(8), blended=torch.rand(8, 3),
                    target_points=torch.rand(8, 3) + 1, source_points=torch.rand(8, 3) + 1,
                    weight_logits=torch.randn(8, 1), init_weights_tensor=torch.rand(8))
        base, main = computer.compute_total_loss(**args)
        same, main2 = computer.compute_total_loss(**args, extra_loss=None)
        self.assertEqual(float(base), float(same))
        self.assertEqual(float(main), float(main2))


if __name__ == "__main__":
    unittest.main()
