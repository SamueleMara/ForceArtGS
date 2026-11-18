"""Stage E / Gate E: TCP contact and calibrated force inside refinement and selection (§10 items 12-15, 18)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.cli import run_discovery  # noqa: E402
from estimotion_2part.joint_discovery.config import JointDiscoveryConfig  # noqa: E402
from estimotion_2part.joint_discovery.contact_annotations import ContactTimeline  # noqa: E402
from estimotion_2part.joint_discovery.rigid_kinematics import exp_so3  # noqa: E402
from estimotion_2part.joint_discovery.synthetic import make_scene, make_sliding_contact  # noqa: E402
from estimotion_2part.joint_discovery.wrench_calibration import Provenanced  # noqa: E402

R_EE_S = exp_so3(np.array([0.0, 0.3, 0.7]))


def cfg():
    c = JointDiscoveryConfig(mode="calibrated_joint_aware", enable_tcp_contact=True, enable_physical_selection=True)
    c.fit.em_iterations, c.fit.n_seeds = 2, 2
    c.comparison.uncertainty_method = "cluster_robust"
    c.contact.surface_max_distance_m = 0.01
    return c.validate()


def scene(kind):
    return make_scene(kind, seed=3, T=14, n_moving=300, n_static=150, sigma_px=1.5, cameras=("cam_001",),
                      max_angle_deg=20, max_disp_m=0.08)


def run(sc, rec, cal, tl):
    with tempfile.TemporaryDirectory() as td:
        run_discovery(sc.obs, sc.calibs, cfg(), Path(td) / "r", scene="synthetic", contact=rec,
                      physical=dict(contact=rec, calibration=cal, calibs=sc.calibs, timeline=tl))
        return json.loads((Path(td) / "r" / "model_selection.json").read_text())


class TestForceSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc_r = scene("revolute")
        cls.rec_r, cls.tl_r, cls.cal_r, cls.gt_r = make_sliding_contact(cls.sc_r, R_EE_S=R_EE_S)
        cls.sel_r = run(cls.sc_r, cls.rec_r, cls.cal_r, cls.tl_r)

    def test_revolute_sliding_contact_and_force_enter_refinement_and_selection(self):
        ph = self.sel_r["physical_selection"]
        self.assertEqual(ph["status"], "OK")
        self.assertEqual(ph["contact_gate"], "PASS")
        self.assertEqual(ph["force_gate"], "PASS")
        self.assertIn("force_direction", ph["selection_terms"])
        self.assertTrue(ph["integration_changed_parameters"])              # S3/S4 are not S2 by design
        self.assertTrue(self.sel_r["robot_terms_used_in_selection"])
        self.assertTrue(self.sel_r["output_contract"]["used_for_force_fit"])
        mu = ph["S4"]["mu"]
        self.assertLess(mu["revolute"]["mean_angle_deg"], mu["prismatic"]["mean_angle_deg"])
        self.assertAlmostEqual(mu["revolute"]["mu"], self.gt_r["mu"], delta=0.1)
        self.assertGreater(ph["S4"]["heldout_physical_nll"]["prismatic"], ph["S4"]["heldout_physical_nll"]["revolute"])
        self.assertEqual(ph["S4"]["decision"], "revolute")
        nulls = ph["S4"]["null_controls"]
        self.assertTrue(nulls["sign_reversed"]["would_abstain_goodness_of_fit"])

    def test_sign_reversed_wrench_abstains_force_but_keeps_contact(self):
        rec = self.rec_r
        import copy
        r2 = copy.copy(rec)
        r2.wrench_sensor = -rec.wrench_sensor
        sel = run(self.sc_r, r2, self.cal_r, self.tl_r)
        ph = sel["physical_selection"]
        self.assertEqual(ph["S4"]["status"], "ABSTAIN")
        self.assertIn("goodness of fit", ph["force_gate"])
        self.assertNotIn("force_direction", ph["selection_terms"])
        self.assertIn("S3", ph)

    def test_time_shifted_wrench_fits_worse(self):
        sc = self.sc_r
        rec, tl, cal, gt = make_sliding_contact(sc, R_EE_S=R_EE_S, stick_fraction=0.4)
        a = run(sc, rec, cal, tl)["physical_selection"]
        import copy
        r2 = copy.copy(rec)
        r2.wrench_sensor = np.roll(rec.wrench_sensor, 5, axis=0)
        b = run(sc, r2, cal, tl)["physical_selection"]
        ang = lambda ph: ph["S4"]["mu"]["revolute"]["mean_angle_deg"] if "mu" in ph.get("S4", {}) else np.inf
        self.assertGreater(ang(b), ang(a))

    def test_uncalibrated_wrench_runs_contact_only(self):
        import copy
        cal = copy.deepcopy(self.cal_r)
        cal.R_EE_S = Provenanced(cal.R_EE_S.value, "DECLARED", "user confirmation")
        ph = run(self.sc_r, self.rec_r, cal, self.tl_r)["physical_selection"]
        self.assertEqual(ph["status"], "OK")
        self.assertTrue(ph["force_gate"].startswith("FAIL"))
        self.assertNotIn("S4", ph)
        self.assertEqual(ph["selection_terms"], ["contact"])

    def test_heuristic_or_missing_contact_abstains_and_leaves_vision_unchanged(self):
        sel_none = run(self.sc_r, self.rec_r, self.cal_r, None)
        self.assertEqual(sel_none["physical_selection"]["status"], "ABSTAIN")
        self.assertFalse(sel_none["robot_terms_used_in_selection"])
        self.assertEqual(sel_none["decision"], self.sel_r["decision"])
        T = len(self.tl_r.state)
        heur = ContactTimeline(self.tl_r.source_frame_index, np.array(["contact_unknown"] * T, dtype=object),
                               np.array(["heuristic_force_threshold"] * T, dtype=object))
        self.assertEqual(run(self.sc_r, self.rec_r, self.cal_r, heur)["physical_selection"]["status"], "ABSTAIN")

    def test_hidden_contact_face_is_unavailable(self):
        sc = self.sc_r
        rec, tl, cal, gt = make_sliding_contact(sc, R_EE_S=R_EE_S)
        # push the canonical contact 3 cm below the observed top face (unobserved underside)
        from estimotion_2part.joint_discovery.rigid_kinematics import apply_joint
        x0 = gt["x0"] - np.array([0, 0, 0.03])
        rec.p_tcp_w = np.stack([apply_joint("revolute", sc.gt_axis, sc.gt_origin, sc.gt_q[t], x0[t])
                                for t in range(len(x0))])
        ph = run(sc, rec, cal, tl)["physical_selection"]
        self.assertEqual(ph["status"], "ABSTAIN")
        self.assertIn("CONTACT_NORMAL_UNAVAILABLE", ph["reason"])

    def test_prismatic_sliding_contact(self):
        sc = scene("prismatic")
        rec, tl, cal, gt = make_sliding_contact(sc, R_EE_S=R_EE_S)
        ph = run(sc, rec, cal, tl)["physical_selection"]
        if ph["status"] != "OK":
            self.assertIn("UNAVAILABLE", ph["reason"])   # allowed: abstain, never a fabricated surface
            return
        mu = ph["S4"]["mu"]
        self.assertLessEqual(mu["prismatic"]["mean_angle_deg"], mu["revolute"]["mean_angle_deg"] + 1.0)
        self.assertNotEqual(ph["S4"]["decision"], "revolute")


if __name__ == "__main__":
    unittest.main()
