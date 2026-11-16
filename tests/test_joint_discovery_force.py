"""T17-T19: wrench frames/signs, decomposition, passivity, friction direction and abstention."""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery import rigid_kinematics as rk  # noqa: E402
from estimotion_2part.joint_discovery import wrench_physics as wp  # noqa: E402
from estimotion_2part.joint_discovery.config import ContactConfig  # noqa: E402
from estimotion_2part.joint_discovery.geometry_fit import JointCandidate  # noqa: E402
from estimotion_2part.joint_discovery.tcp_contact import (ContactRecord, SurfaceSupport, contact_kinematics,  # noqa: E402
                                                          pin_in_slot, tcp_velocity)


def cand(kind, axis, origin, q):
    return JointCandidate(kind, rk.normalize(axis), origin, np.asarray(q, float), np.ones(len(q), bool), None, 0,
                          np.zeros((1, 3)), np.ones(1), np.zeros(1), np.zeros(1))


class TestForce(unittest.TestCase):
    def test_T17_frames_signs_and_decomposition(self):
        rng = np.random.default_rng(0)
        f_reaction_w = np.array([[3.0, -1.0, 2.0]] * 4)
        R = rk.exp_so3(rng.normal(size=(4, 3)))           # rotating sensor orientation
        f_s = np.einsum("tji,tj->ti", R, f_reaction_w)    # R^T f
        conv = wp.WrenchConvention("object_on_robot")
        f_obj = wp.force_on_object_world(f_s, R, conv)
        np.testing.assert_allclose(f_obj, -f_reaction_w, atol=1e-12)  # invariant world force, sign reversed
        np.testing.assert_allclose(wp.force_on_object_world(f_s, R, wp.WrenchConvention("robot_on_object")),
                                   f_reaction_w, atol=1e-12)
        with self.assertRaises(ValueError):
            wp.WrenchConvention("unknown").object_side_multiplier
        with self.assertRaises(ValueError):  # unknown torque origin -> no transport to the TCP
            wp.torque_about_tcp_world(f_s, f_s, R, np.zeros(3), np.ones(3), conv)
        f = np.array([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0], [0.1, 0.0, 0.0], [np.nan, 0, 0]])
        v = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0, 0], [0.1, 0, 0]])
        d = wp.decompose_force(f, v, eps_v=0.002, eps_f=0.5)
        self.assertTrue(d["valid"][0])
        self.assertAlmostEqual(d["F_parallel"][0], 2.0)       # N, signed along the TCP tangent
        self.assertAlmostEqual(d["F_perp"][0], 1.0)
        self.assertAlmostEqual(d["rho_parallel"][0] ** 2 + d["rho_perp"][0] ** 2, 1.0)
        d2 = wp.decompose_force(-f, v, 0.002, 0.5)
        self.assertAlmostEqual(d2["F_parallel"][0], -2.0)     # sign reversal flips F_par, keeps the ratios
        self.assertAlmostEqual(d2["rho_parallel"][0], d["rho_parallel"][0])
        self.assertEqual(list(d["reason"][1:]), ["tcp_speed_below_threshold", "force_below_threshold", "force_missing"])

    def test_T18_passivity_and_sticking(self):
        n_in = np.array([[0, 0, -1.0]] * 3)
        s = np.array([[0.05, 0, 0], [0.02, 0.01, 0], [0, 0, 0]])
        f_hat, stick = wp.coulomb_direction(n_in, s, mu=0.4)
        self.assertTrue(stick[2] and np.isnan(f_hat[2]).all())  # sticking: direction undefined, no division by 0
        f = 10 * f_hat[:2]
        D = np.einsum("ti,ti->t", f, s[:2])
        self.assertTrue(np.all(D >= 0))
        self.assertEqual(wp.passivity_penalty(D, 0.01)["n_violations"], 0)
        bad = wp.passivity_penalty(-D, 0.01)                     # wrench sign error is detected
        self.assertEqual(bad["n_violations"], 2)
        self.assertGreater(bad["penalty"], 0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wp.coulomb_direction(n_in, np.zeros((3, 3)), 0.3)

    def _lid_scene(self, mu=0.3):
        ps = pin_in_slot(h=0.1, x_start=0.35, x_end=0.10, T=40)
        u, o, th = ps["axis"], ps["origin"], ps["theta"]
        cR = cand("revolute", u, o, th)
        rec = ContactRecord(ps["t"], ps["p_tcp"], tcp_velocity(ps["p_tcp"], ps["t"]), np.ones(40, bool))
        kinR = contact_kinematics(cR, rec)
        n0_out = np.array([0, -1.0, 0])                      # TCP pushes the lid underside
        pred, stick = wp.predicted_force_directions("revolute", u, th, n0_out, kinR["slip"], mu)
        f_obj = 12.0 * pred
        # prismatic competitor: translation along the mean contact velocity (same TCP data)
        vM = kinR["v_material"][~np.isnan(kinR["v_material"]).any(1)]
        uP = rk.normalize(vM.mean(0))
        dP = np.cumsum(np.r_[0, np.linalg.norm(np.diff(ps["p_tcp"] @ np.eye(3), axis=0), axis=1)]) * 0 + \
            np.r_[0, np.cumsum(np.diff(ps["t"]) * (kinR["v_material"][:-1] @ uP))]
        cP = cand("prismatic", uP, None, np.nan_to_num(dP))
        kinP = contact_kinematics(cP, rec)
        return dict(cR=cR, cP=cP, rec=rec, kin=dict(revolute=kinR, prismatic=kinP), f_obj=f_obj, n0=n0_out,
                    active=~stick & np.isfinite(f_obj).all(1))

    def test_T19_force_direction_supports_correct_hypothesis(self):
        sc = self._lid_scene()
        surf = {k: SurfaceSupport("OK", "observed", sc["n0"], np.zeros(3), 0.001, 50) for k in ("revolute", "prismatic")}
        ev = wp.force_direction_evidence(np.nan_to_num(sc["f_obj"]), sc["kin"], dict(revolute=sc["cR"], prismatic=sc["cP"]),
                                         surf, sc["active"], ContactConfig())
        self.assertEqual(ev["status"], "OK", ev)
        r, p = ev["per_kind"]["revolute"], ev["per_kind"]["prismatic"]
        self.assertLess(r["mean_angle_deg"], 1.0)
        self.assertAlmostEqual(r["mu"], 0.3, places=1)          # shared, bounded mu recovered
        self.assertGreater(p["mean_angle_deg"], 5.0)            # constant prismatic normal cannot follow the lid
        # measured ratios are reported for both, but they are not the classifier
        dec = wp.decompose_force(np.nan_to_num(sc["f_obj"]), sc["rec"].v_tcp_w)
        self.assertTrue(np.all(np.isfinite(dec["rho_parallel"][dec["valid"]])))
        # no independently supported surface -> abstain
        surf_none = dict(surf, prismatic=SurfaceSupport("UNAVAILABLE", "underside unobserved"))
        ev2 = wp.force_direction_evidence(np.nan_to_num(sc["f_obj"]), sc["kin"], dict(revolute=sc["cR"], prismatic=sc["cP"]),
                                          surf_none, sc["active"], ContactConfig())
        self.assertEqual(ev2["status"], "ABSTAIN")

    def test_generalized_force_units_and_power(self):
        sc = self._lid_scene()
        pw = wp.power_terms("revolute", sc["cR"].axis_w, sc["cR"].origin_w, sc["rec"].p_tcp_w, sc["rec"].v_tcp_w,
                            np.nan_to_num(sc["f_obj"]), sc["kin"]["revolute"]["qdot"])
        self.assertEqual(pw["Q_units"], "N*m")
        m = sc["active"] & np.isfinite(pw["D"])
        self.assertTrue(np.all(pw["D"][m] >= -1e-6))  # friction-consistent synthetic force is passive
        nulls = wp.null_controls(np.nan_to_num(sc["f_obj"]), np.random.default_rng(0))
        Dn = wp.power_terms("revolute", sc["cR"].axis_w, sc["cR"].origin_w, sc["rec"].p_tcp_w, sc["rec"].v_tcp_w,
                            nulls["sign_reversed"], sc["kin"]["revolute"]["qdot"])["D"]
        self.assertGreater(wp.passivity_penalty(Dn, 0.01, active=m)["n_violations"], 0)


if __name__ == "__main__":
    unittest.main()
