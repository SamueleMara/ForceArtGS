"""Gate B of the multimodal TCP spec: physically correct, mode-aware force behaviour."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.config import ForceModeConfig  # noqa: E402
from estimotion_2part.joint_discovery.contact_modes import (ContactRows, MovingSurface,  # noqa: E402
                                                            load_contact_segments)
from estimotion_2part.joint_discovery.rigid_kinematics import (body_velocity, exp_so3_axis_angle,  # noqa: E402
                                                               normalize)
from estimotion_2part.joint_discovery.synthetic import make_tcp_contact  # noqa: E402
from estimotion_2part.joint_discovery.tcp_contact import joint_rate, tcp_velocity  # noqa: E402
from estimotion_2part.joint_discovery.wrench_calibration import WrenchCalibration  # noqa: E402
from estimotion_2part.joint_discovery.wrench_physics import (force_alignment_diagnostics, force_gate,  # noqa: E402
                                                             kinetic_direction_evidence, mechanical_model_evidence,
                                                             static_contact_feasibility)


def cand(sc, kind=None, axis=None, origin=None, q=None):
    return dict(kind=kind or sc["kind"], axis=sc["axis"] if axis is None else axis,
                origin=sc["origin"] if origin is None else origin, q=sc["q"] if q is None else q)


def coulomb_force(sc, mu=0.3, fmag=6.0, noise=0.03, seed=0):
    """Robot-on-object force for a sliding point contact on the observed face (ground truth kinematics)."""
    rng = np.random.default_rng(seed)
    ts = sc["frame_map"].timestamp_s
    p = sc["p_tcp"]
    v = tcp_velocity(p, ts)
    qd = joint_rate(sc["q"], ts)
    n0 = np.array([0, 0, 1.0]) if sc["kind"] == "revolute" else np.array([0, -1.0, 0])
    nW = (np.einsum("tij,j->ti", exp_so3_axis_angle(sc["axis"], sc["q"]), n0) if sc["kind"] == "revolute"
          else np.tile(n0, (len(ts), 1)))
    vm = body_velocity(sc["kind"], sc["axis"], sc["origin"], qd, p)
    s = v - vm
    sT = s - np.einsum("ti,ti->t", s, nW)[:, None] * nW
    d = -nW + mu * sT / np.maximum(np.linalg.norm(sT, axis=1, keepdims=True), 1e-12)
    return fmag * d / np.linalg.norm(d, axis=1, keepdims=True) + rng.normal(0, noise, d.shape), v


def surface(sc):
    return MovingSurface.from_points(sc["surface_points"], sc["camera_centers"])


TRAIN = np.r_[np.ones(20, bool), np.zeros(10, bool)]


class GateB(unittest.TestCase):
    def test_b01_sliding_kinetic_direction_forward_and_backward(self):
        cfg = ForceModeConfig()
        for slide in ((-0.08, 0.08), (0.08, -0.08)):
            sc = make_tcp_contact("revolute", "sliding", sigma_tcp=0.0, slide=slide)
            f, v = coulomb_force(sc, mu=0.3)
            ev = kinetic_direction_evidence(f, sc["p_tcp"], v, sc["frame_map"].timestamp_s, dict(revolute=cand(sc)),
                                            surface(sc), np.ones(30, bool), TRAIN, cfg)
            self.assertEqual(ev["status"], "OK", ev.get("reason"))
            self.assertAlmostEqual(ev["per_kind"]["revolute"]["mu"], 0.3, delta=0.05)
            self.assertLess(ev["per_kind"]["revolute"]["train_mean_angle_deg"], 2.0)

    def test_b02_b09_zero_slip_has_no_kinetic_direction(self):
        cfg = ForceModeConfig()
        sc = make_tcp_contact("revolute", "stick_hard", sigma_tcp=0.0)
        segs, _ = load_contact_segments(sc["segments_doc"], sc["frame_map"], sc["p_tcp"])
        f, v = coulomb_force(sc, mu=0.0)                         # pure normal push: real contact force
        ev = kinetic_direction_evidence(f, sc["p_tcp"], v, sc["frame_map"].timestamp_s, dict(revolute=cand(sc)),
                                        surface(sc), np.ones(30, bool), TRAIN, cfg)
        self.assertEqual(ev["status"], "ABSTAIN")                 # undefined, never a zero-residual "perfect fit"
        st = static_contact_feasibility(f, sc["p_tcp"], dict(revolute=cand(sc)), surface(sc), segs, cfg)
        self.assertEqual(st["status"], "OK")
        seg = st["per_segment"][0]
        self.assertIn("ABSTAIN", seg["kinetic_direction"])
        self.assertGreater(seg["per_kind"]["revolute"]["feasible_fraction"], 0.95)
        pulled = static_contact_feasibility(-f, sc["p_tcp"], dict(revolute=cand(sc)), surface(sc), segs, cfg)
        self.assertLess(pulled["per_segment"][0]["per_kind"]["revolute"]["feasible_fraction"], 0.05)  # tension

    def test_b03_grasp_abstains_from_unilateral_cone(self):
        cfg = ForceModeConfig()
        sc = make_tcp_contact("revolute", "stick_hard", sigma_tcp=0.0)
        doc = copy.deepcopy(sc["segments_doc"])
        doc["segments"][0]["contact_model"] = "grasp"
        segs, _ = load_contact_segments(doc, sc["frame_map"], sc["p_tcp"])
        f, _ = coulomb_force(sc, mu=0.0)
        st = static_contact_feasibility(-f, sc["p_tcp"], dict(revolute=cand(sc)), surface(sc), segs, cfg)
        self.assertEqual(st["status"], "ABSTAIN")                 # tension through a grasp is not "infeasible"

    def _pushed(self, kind):
        sc = make_tcp_contact(kind, "stick_hard", sigma_tcp=0.0)
        ts = sc["frame_map"].timestamp_s
        vm = body_velocity(kind, sc["axis"], sc["origin"], joint_rate(sc["q"], ts), sc["p_tcp"])
        rng = np.random.default_rng(1)
        f = 5 * vm / np.maximum(np.linalg.norm(vm, axis=1, keepdims=True), 1e-9) + rng.normal(0, 0.2, vm.shape)
        return sc, f, tcp_velocity(sc["p_tcp"], ts)

    def test_b04_high_alignment_never_certifies_prismatic(self):
        cfg = ForceModeConfig()
        for kind in ("revolute", "prismatic"):
            sc, f, v = self._pushed(kind)
            d = force_alignment_diagnostics(f, sc["p_tcp"], v, sc["frame_map"].timestamp_s, {kind: cand(sc)})
            self.assertGreater(d["median_rho_parallel_signed"], 0.9, kind)     # both have high alignment
            other = "prismatic" if kind == "revolute" else "revolute"
            m = mechanical_model_evidence(f, sc["p_tcp"], sc["frame_map"].timestamp_s,
                                          {kind: cand(sc), other: cand(sc, kind=other, axis=normalize(sc["p_tcp"][-1] - sc["p_tcp"][0]),
                                                                       origin=None if other == "prismatic" else np.zeros(3))},
                                          np.ones(30, bool), TRAIN, cfg)
            self.assertEqual(m["status"], "FORCE_NONDISCRIMINATIVE")             # no model declared

    def _slider_and_hinge(self):
        sc, f, v = self._pushed("prismatic")
        hinge = sc["p_tcp"][0] + np.array([0.25, 0.0, 0.0])
        qR = np.linalg.norm(sc["p_tcp"] - sc["p_tcp"][0], axis=1) / 0.25         # same path length about z, same sense
        R = cand(sc, kind="revolute", axis=np.array([0, 0, 1.0]), origin=hinge, q=qR)
        return sc, f, dict(prismatic=cand(sc), revolute=R)

    def test_b05_declared_model_gives_grounded_prismatic_evidence(self):
        sc, f, cands = self._slider_and_hinge()
        cfg = ForceModeConfig(mechanical_model="min_effort_tangent")
        m = mechanical_model_evidence(f, sc["p_tcp"], sc["frame_map"].timestamp_s, cands, np.ones(30, bool), TRAIN, cfg)
        self.assertEqual(m["status"], "OK", m.get("reason"))
        self.assertEqual(m["favours"], "prismatic")
        none = mechanical_model_evidence(f, sc["p_tcp"], sc["frame_map"].timestamp_s, cands, np.ones(30, bool), TRAIN,
                                         ForceModeConfig())
        self.assertEqual(none["status"], "FORCE_NONDISCRIMINATIVE")

    def test_b06_generalized_force_units(self):
        sc, f, cands = self._slider_and_hinge()
        d = force_alignment_diagnostics(f, sc["p_tcp"], tcp_velocity(sc["p_tcp"], sc["frame_map"].timestamp_s),
                                        sc["frame_map"].timestamp_s, cands)
        self.assertEqual(d["per_hypothesis"]["prismatic"]["Q_units"], "N")
        self.assertEqual(d["per_hypothesis"]["revolute"]["Q_units"], "N*m")
        self.assertIn("P_joint_prismatic_W", d["per_frame"][10])               # compared as powers [W]

    def test_b07_unverified_wrench_abstains_without_touching_tcp_rows(self):
        sc = make_tcp_contact("revolute", "stick_hard")
        cal = WrenchCalibration(dataset_id="x", tool_id="y")
        g = force_gate(cal, np.ones((30, 3)))
        self.assertEqual(g["status"], "ABSTAIN")
        self.assertEqual(force_gate(None, None)["status"], "ABSTAIN")
        segs, _ = load_contact_segments(sc["segments_doc"], sc["frame_map"], sc["p_tcp"])
        self.assertGreater(ContactRows(segs, sc["p_tcp"], np.ones(30, bool)).n_rows, 0)

    def test_b08_force_nulls_do_not_create_evidence(self):
        sc, f, cands = self._slider_and_hinge()
        cfg = ForceModeConfig(mechanical_model="min_effort_tangent")
        m = mechanical_model_evidence(f, sc["p_tcp"], sc["frame_map"].timestamp_s, cands, np.ones(30, bool), TRAIN, cfg)
        self.assertGreater(abs(m["delta_ll"]["value"]), m["null_threshold"])
        nc = m["null_controls"]
        for k in ("sign_reversed", "gravity_only", "random_constant_direction_q99", "random_per_frame_direction_q99"):
            self.assertLess(abs(nc[k]), abs(m["delta_ll"]["value"]), k)
        rng = np.random.default_rng(3)
        noise = rng.normal(size=f.shape)
        r = mechanical_model_evidence(noise, sc["p_tcp"], sc["frame_map"].timestamp_s, cands, np.ones(30, bool), TRAIN,
                                      cfg)
        self.assertNotEqual(r["status"], "OK")

    def test_b10_same_samples_and_train_only_parameters(self):
        cfg = ForceModeConfig()
        sc = make_tcp_contact("revolute", "sliding", sigma_tcp=0.0)
        f, v = coulomb_force(sc, mu=0.4)
        P = cand(sc, kind="prismatic", axis=normalize(sc["p_tcp"][-1] - sc["p_tcp"][0]), origin=None,
                 q=np.linalg.norm(sc["p_tcp"] - sc["p_tcp"][0], axis=1))
        args = (sc["p_tcp"], v, sc["frame_map"].timestamp_s, dict(revolute=cand(sc), prismatic=P), surface(sc),
                np.ones(30, bool), TRAIN, cfg)
        e1 = kinetic_direction_evidence(f, *args)
        f2 = f.copy()
        f2[~TRAIN] = -f2[~TRAIN]                                 # change held-out forces only
        e2 = kinetic_direction_evidence(f2, *args)
        self.assertEqual(e1["per_kind"]["revolute"]["n_train"], e1["per_kind"]["prismatic"]["n_train"])
        for k in ("revolute", "prismatic"):
            self.assertEqual(e1["per_kind"][k]["mu"], e2["per_kind"][k]["mu"])
        self.assertNotEqual(e1["per_kind"]["revolute"]["heldout_mean_angle_deg"],
                            e2["per_kind"]["revolute"]["heldout_mean_angle_deg"])


if __name__ == "__main__":
    unittest.main()
