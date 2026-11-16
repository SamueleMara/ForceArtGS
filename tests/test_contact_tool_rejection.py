"""Phase C: canonical contact support rejects robot-tool geometry when configured (opt-in)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from estimotion_2part.contact_surface_loss import BRANCH_T2  # noqa: E402
from estimotion_2part.window_priors import ContactTerm  # noqa: E402


def make_term(**kw):
    return ContactTerm(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([True]), weight=1.0, sigma_m=0.015,
                       robust_delta_m=0.015, support_max_distance_m=0.03, inherited_branch=BRANCH_T2, **kw)


class ContactToolRejectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # moving surface: plane z = 1.0; the tool pushes on it along its normal (tip at z = 0.995, shaft
        # towards the camera), so the surface lies in front of the calibrated tip
        surface = np.c_[rng.uniform(-0.05, 0.05, (400, 2)), np.full(400, 1.0)]
        tool = np.c_[rng.uniform(-0.004, 0.004, (200, 2)), rng.uniform(0.80, 0.99, 200)]
        self.points = torch.tensor(np.vstack([surface, tool]), dtype=torch.float32)
        self.weights = torch.ones(len(self.points))           # all assigned to the moving branch (T2)
        self.n_tool = 200
        self.axis = (np.array([0.0, 0.0, 0.995]), np.array([0.0, 0.0, 0.805]))   # tip -> EE

    def prepare(self, term):
        I = torch.eye(4).unsqueeze(0)
        return term.prepare(I, I, self.points, self.weights)

    def test_default_keeps_legacy_support(self):
        term = make_term()
        self.prepare(term)
        self.assertEqual(term.canonical.shape[0], len(self.points))
        self.assertNotIn("canonical_tool_points_rejected", term.info)

    def test_tool_points_are_removed_and_surface_kept(self):
        term = make_term(tool_axis=self.axis, tool_radius_m=0.015)
        self.prepare(term)
        self.assertGreaterEqual(term.info["canonical_tool_points_rejected"], int(0.95 * self.n_tool))
        z = term.canonical[:, 2].numpy()
        self.assertEqual(int((z == 1.0).sum()), 400)           # surface in front of the tip is never removed
        self.assertLess(int((z < 0.995).sum()), 10)

    def test_invalid_configuration_is_refused(self):
        with self.assertRaises(ValueError):
            make_term(tool_axis=None, tool_radius_m=0.015)
        with self.assertRaises(ValueError):
            make_term(tool_axis=self.axis, tool_radius_m=0.0)


if __name__ == "__main__":
    unittest.main()
