"""Method parity (LS / rollout / PINN / EKF on identical inputs), unified mass geometry (one density for inertia
and gravity, analytic r_perp), tentative geometry proxy, no held-out leakage. Synthetic truth only."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9 import physics_lib  # noqa: E402
from forceartgs_g9.core import run_g9  # noqa: E402
from forceartgs_g9.dynamics.general_1dof import GeneralTerms, RigidBodyGravity, rotation  # noqa: E402
from forceartgs_g9.dynamics.geometry_inertia import PASSIVE, GeometryMassModel  # noqa: E402
from forceartgs_g9.dynamics.mass_properties import TriMesh, unit_density_properties  # noqa: E402
from forceartgs_g9.synthetic import box_mesh  # noqa: E402
from forceartgs_g9.testing import synthetic_cfg, synthetic_inputs  # noqa: E402

physics_lib.load()
PROV = dict(source="test")


def l_shape():
    """Asymmetric off-axis rigid part: union of two boxes as one watertight solid is complex; use two disjoint
    closed boxes in one mesh (still a valid closed volume for the polyhedral integrals)."""
    a = box_mesh([0.0, 0.0, 0.0], [0.4, 0.05, 0.3])
    b = box_mesh([0.3, 0.05, 0.0], [0.4, 0.25, 0.1])
    V = np.r_[a.vertices, b.vertices]
    F = np.r_[a.faces, b.faces + len(a.vertices)]
    return TriMesh.from_arrays(V, F)


class UnifiedMass(unittest.TestCase):
    def test_rperp_inertia_and_gravity_share_one_density(self):
        mesh = l_shape()
        a = np.array([0.0, 0.0, 1.0])
        o = np.array([-0.05, 0.02, 0.0])
        mm = GeometryMassModel.from_validated_mesh(mesh, "uniform_solid", "revolute", a, o, source_provenance=PROV,
                                                   reflected_actuator_inertia=PASSIVE)
        p = mm.props
        r = p.c1 - o
        rp = r - (r @ a) * a
        I1 = float(a @ p.J1 @ a + p.m1 * rp @ rp)
        self.assertAlmostEqual(mm.M1, I1, places=12)                         # I_axis,1 = a'J a + m1 |r_perp|^2
        o2 = o + 0.7 * a                                                      # origin moved along the axis
        mm2 = GeometryMassModel.from_validated_mesh(mesh, "uniform_solid", "revolute", a, o2, source_provenance=PROV,
                                                    reflected_actuator_inertia=PASSIVE)
        self.assertAlmostEqual(mm2.M1, mm.M1, places=12)
        rho = 650.0
        g = np.array([0.0, 0.2, -9.79])                                       # slightly tilted gravity vs axis
        gu = mm.gravity_unit(g)
        for q in (-0.6, 0.0, 0.9):
            c = o + rotation(a, q) @ (p.c1 - o)
            direct = -(rho * p.m1) * g @ np.cross(a, c - o)                    # rigid-body torque form
            self.assertAlmostEqual(rho * gu.G(q), direct, places=9)
            h = 1e-6
            U = lambda x: -(rho * p.m1) * g @ (o + rotation(a, x) @ (p.c1 - o))
            self.assertAlmostEqual(rho * gu.G(q), (U(q + h) - U(q - h)) / (2 * h), places=5)
        rep = mm.reconstruct(rho, None, fitted_M_eff=rho * mm.M1)
        self.assertAlmostEqual(rep["mass_kg"], rho * p.m1)                   # inertial mass == gravitational mass
        vert = RigidBodyGravity("revolute", (0, 0, 1), tuple(o), tuple(p.c1), p.m1, (0, 0, -9.81))
        self.assertLess(max(abs(vert.G(q)) for q in np.linspace(-3, 3, 9)), 1e-12)   # eccentric COM, vertical hinge

    def test_prismatic_shared_mass_and_reflected_inertia(self):
        mesh = box_mesh([0, 0, 0], [0.3, 0.2, 0.1])
        for axis, expect_zero in (([1.0, 0, 0], True), ([0.0, 0.6, 0.8], False)):
            mm = GeometryMassModel.from_validated_mesh(mesh, "uniform_solid", "prismatic", axis, None,
                                                       source_provenance=PROV, reflected_actuator_inertia=0.5)
            gu = mm.gravity_unit((0, 0, -9.81))
            rho = 200.0
            G = rho * gu.G(0.0)
            if expect_zero:
                self.assertAlmostEqual(G, 0.0)
            else:
                self.assertAlmostEqual(G, -(rho * mm.props.m1) * (-9.81 * 0.8))   # resists upward motion
            self.assertAlmostEqual(mm.alpha_from_effective(rho * mm.M1 + 0.5), rho)   # reflected != gravitational

    def test_shell_vs_solid_units_and_tentative_proxy(self):
        from forceartgs_g9.dynamics.geometry_proxy import build_proxy
        mesh = box_mesh([0, 0, 0], [0.5, 0.02, 0.6])
        s = unit_density_properties(mesh, "uniform_solid")
        h = unit_density_properties(mesh, "uniform_shell")
        self.assertIn("kg/m^3", s.density_units)
        self.assertIn("kg/m^2", h.density_units)
        self.assertTrue(np.all(np.linalg.eigvalsh(s.J1) > 0) and np.all(np.linalg.eigvalsh(h.J1) > 0))
        with tempfile.TemporaryDirectory() as td:
            obj, inp, hand = synthetic_inputs("revolute", td, terms=GeneralTerms(static=False))
            px = build_proxy(hand, "shell_hull", td)
            self.assertEqual(px["provenance"]["status"], "TENTATIVE_GEOMETRIC_PREVIEW")
            from forceartgs_g9.dynamics.mass_properties import topology
            self.assertTrue(topology(px["mesh"])["watertight"])
            hc = unit_density_properties(px["mesh"], "uniform_solid").c1
            naive = px["points"].mean(0)
            self.assertTrue(np.all(np.isfinite(hc)))
            self.assertGreater(np.linalg.norm(px["mesh"].vertices.mean(0) - hc) + np.linalg.norm(naive - hc), 0.0)


class MethodParity(unittest.TestCase):
    def test_all_methods_identical_inputs_and_no_leakage(self):
        t = GeneralTerms(static=False)
        with tempfile.TemporaryDirectory() as td:
            obj, inp, _ = synthetic_inputs("prismatic", td, terms=t, n_heldout=1)
            cfg = synthetic_cfg(t, fitting={"methods": ["general_integral_ls", "general_rollout_fit", "general_pinn",
                                                        "general_akf"],
                                            "pinn": {"n_seeds": 1, "steps_data": 1500, "steps_joint": 1500},
                                            "ekf": {"reverse_order_check": False}})
            r = run_g9(inp, cfg, Path(td) / "g9", log=lambda m: None)
            mc = json.loads((Path(td) / "g9" / "estimator_results" / "method_comparison.json").read_text())
            for f in ("general_rollout_fit.json", "general_pinn.json", "general_akf.json"):
                self.assertTrue((Path(td) / "g9" / "estimator_results" / f).is_file(), f)
            names = {m: sorted(v["parameters"]) for m, v in mc["methods"].items() if "parameters" in v}
            self.assertEqual(len(names), 3)
            self.assertEqual(len({tuple(v) for v in names.values()}), 1)            # same parameter vector
            self.assertEqual(names["general_pinn"], ["Fc", "alpha", "c"])            # one density, no M / mu
            for m, v in mc["methods"].items():
                self.assertIn("heldout", v)
            # held-out predictions use frozen theta + declared state only: perturbing future test q changes nothing
            from forceartgs_g9.validation.heldout_prediction import predict_heldout
            from forceartgs_g9.estimators.model_spec import G9Interaction
            h = inp.heldout[0]
            th = np.asarray(json.loads((Path(td) / "g9" / "estimator_results" / "general_akf.json").read_text())["x"])
            a = predict_heldout(r["spec"], th, h)
            h2 = G9Interaction(h.name, h.joint_type, h.t_obs, h.q_obs + 0.1 * (h.t_obs > 0.3), h.q_std, h.effort,
                               h.initial_state)
            self.assertEqual(a["q_pred"], predict_heldout(r["spec"], th, h2)["q_pred"])

    def test_no_mesh_joint_space_still_fits_without_tensor(self):
        t = GeneralTerms(static=False, gravity=False)
        with tempfile.TemporaryDirectory() as td:
            obj, inp, _ = synthetic_inputs("prismatic", td, terms=t, n_heldout=0)
            inp.mass_mesh = None
            cfg = synthetic_cfg(t, fitting={"methods": ["general_integral_ls", "general_rollout_fit", "general_akf"],
                                            "ekf": {"reverse_order_check": False}})
            r = run_g9(inp, cfg, Path(td) / "g9", log=lambda m: None)
            self.assertTrue(r["manifest"]["readiness"]["DYNAMICS_DIAGNOSTIC"])
            self.assertIsNone(r["physics_spec_v1"])
            rep = json.loads((Path(td) / "g9" / "inertia_reconstruction_report.json").read_text())
            self.assertIsNone(rep.get("full_tensor"))


if __name__ == "__main__":
    unittest.main()
