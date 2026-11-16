"""T13-T16: sliding TCP contact kinematics, surface support and degeneracies."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery import rigid_kinematics as rk  # noqa: E402
from estimotion_2part.joint_discovery.config import ContactConfig  # noqa: E402
from estimotion_2part.joint_discovery.geometry_fit import JointCandidate  # noqa: E402
from estimotion_2part.joint_discovery.tcp_contact import (UNAVAILABLE, ContactRecord, contact_kinematics,  # noqa: E402
                                                          contact_residuals, free_plane_degeneracy, pin_in_slot,
                                                          surface_from_observed_patch, surface_normal_world,
                                                          tcp_velocity)


def cand(kind, axis, origin, q):
    n = 1
    return JointCandidate(kind, rk.normalize(axis), origin, np.asarray(q, float), np.ones(len(q), bool), None, 0,
                          np.zeros((n, 3)), np.ones(n), np.zeros(n), np.zeros(n))


class TestContact(unittest.TestCase):
    def test_T13_pin_in_slot(self):
        ps = pin_in_slot()
        u, o, th, thd, p, v = ps["axis"], ps["origin"], ps["theta"], ps["thetadot"], ps["p_tcp"], ps["v_tcp"]
        # TCP path is a straight line
        self.assertLess(np.linalg.svd(p - p.mean(0), compute_uv=False)[1], 1e-12)
        # canonical contact points lie on the lid's contact line (canonical x-axis) at sliding radii
        x0 = rk.canonical_contact_point("revolute", u, o, th, p)
        np.testing.assert_allclose(x0[:, 1:], 0, atol=1e-12)
        np.testing.assert_allclose(x0[:, 0], ps["radius"], atol=1e-12)
        self.assertGreater(np.ptp(ps["radius"]), 0.05)               # the contact slides
        # material velocity at the TCP is circular (perpendicular to the radius)
        vM = rk.body_velocity("revolute", u, o, thd, p)
        np.testing.assert_allclose(np.einsum("ti,ti->t", vM, p - o), 0, atol=1e-12)
        s = rk.relative_slip("revolute", u, o, thd, p, v)
        self.assertGreater(np.min(np.linalg.norm(s, axis=1)), 1e-3)  # full-vector equality (rigid grasp) fails
        n = surface_normal_world("revolute", u, th, np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(np.einsum("ti,ti->t", n, s), 0, atol=1e-12)  # normal-velocity equality holds
        # the record-based pipeline gives the same result with numerically differentiated TCP / q
        rec = ContactRecord(ps["t"], p, tcp_velocity(p, ps["t"]), np.ones(len(th), bool))
        kin = contact_kinematics(cand("revolute", u, o, th), rec)
        inner = slice(2, -2)
        ns = np.einsum("ti,ti->t", n, kin["slip"])[inner]
        self.assertLess(np.max(np.abs(ns)), 2e-3 * np.max(np.linalg.norm(kin["slip"], axis=1)))

    def test_T14_prismatic_sliding_on_face(self):
        u = np.array([1.0, 0, 0])
        t = np.linspace(0, 2, 30)
        d = 0.05 * t
        p = np.stack([0.3 + d, 0.03 * t, np.full_like(t, 0.1)], 1)   # TCP not parallel to the slider
        c = cand("prismatic", u, None, d)
        rec = ContactRecord(t, p, tcp_velocity(p, t), np.ones(len(t), bool))
        kin = contact_kinematics(c, rec)
        n0 = np.array([1.0, 0, 0])  # face perpendicular to the slide direction
        ns = kin["slip"] @ n0
        np.testing.assert_allclose(ns, 0, atol=1e-12)
        self.assertGreater(np.min(np.linalg.norm(kin["slip"], axis=1)), 0.02)  # tangential sliding allowed
        np.testing.assert_allclose(kin["canonical_contact"][:, 1], 0.03 * t)   # slides along the face

    def test_T15_straight_tcp_free_plane_is_not_evidence(self):
        t = np.linspace(0, 1, 25)
        p = np.stack([0.4 - 0.2 * t, np.full_like(t, 0.1), np.zeros_like(t)], 1)
        cP = cand("prismatic", [0.3, 1.0, 0.2], None, 0.08 * t)
        cR = cand("revolute", [0, 0, 1.0], np.zeros(3), np.arctan2(0.1, p[:, 0]))
        for c in (cP, cR):
            x0 = rk.canonical_contact_point(c.kind, c.axis_w, c.origin_w, c.q, p)
            self.assertTrue(free_plane_degeneracy(x0)["degenerate"])
        # with no independently observed surface, contact terms are disabled for both
        cfg = ContactConfig()
        rec = ContactRecord(t, p, tcp_velocity(p, t), np.ones(len(t), bool))
        for c in (cP, cR):
            surf = surface_from_observed_patch(np.zeros((0, 3)), p[0], cfg)
            self.assertEqual(surf.status, UNAVAILABLE)
            self.assertEqual(contact_residuals(c, rec, surf, cfg)["status"], UNAVAILABLE)

    def test_T16_underside_contact_rejected(self):
        rng = np.random.default_rng(0)
        cfg = ContactConfig()
        top = np.c_[rng.uniform(-0.1, 0.1, (400, 2)), np.full(400, 0.10)]   # only the top face is observed
        s = surface_from_observed_patch(top, np.array([0.0, 0.0, 0.07]), cfg)   # underside of a 3 cm lid
        self.assertEqual(s.status, UNAVAILABLE)
        self.assertIn("hidden or unobserved", s.reason)
        s = surface_from_observed_patch(top, np.array([0.0, 0.0, 0.085]), cfg)  # 1.5 cm lid: top face is near
        self.assertEqual(s.status, UNAVAILABLE)                                 # but the contact is not on it
        self.assertIn("off the observed face", s.reason)
        # a contact on the observed face is supported, with a camera-facing outward normal
        s2 = surface_from_observed_patch(top, np.array([0.0, 0.0, 0.10]), cfg, camera_centers0=[[0, 0, 1.0]])
        self.assertEqual(s2.status, "OK")
        np.testing.assert_allclose(s2.n0, [0, 0, 1], atol=1e-6)
        # too few nearby observed points (rim only) -> unavailable, never a nearest-neighbour normal
        rim = np.c_[np.linspace(-0.1, 0.1, 5), np.full(5, 0.1), np.full(5, 0.1)]
        self.assertEqual(surface_from_observed_patch(rim, np.array([0.0, 0.1, 0.1]), cfg).status, UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
