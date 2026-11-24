"""One physical contract across ModelSpec / TorchDynamics (PINN) / EKF / rollout / integral LS (section 8C), and
bounded read-only real-fridge diagnostics (section 8D; skipped when the dataset is not mounted)."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9 import physics_lib  # noqa: E402
from forceartgs_g9.dynamics.general_1dof import GeneralTerms, GravityField  # noqa: E402
from forceartgs_g9.dynamics.geometry_inertia import PASSIVE, GeometryMassModel  # noqa: E402
from forceartgs_g9.estimators.model_spec import G9Interaction, ModelSpec  # noqa: E402
from forceartgs_g9.estimators.torch_model import DT, TorchDynamics, zoh_integral  # noqa: E402
from forceartgs_g9.synthetic import ZOH, box_mesh, episode, make_object  # noqa: E402
from forceartgs_g9.validation import gravity_geometry_audit as GA  # noqa: E402

physics_lib.load()
G = (0.0, 0.0, -9.81)
GF = GravityField(G, True, "test")
rng = np.random.default_rng(3)


def geo(jt="revolute", axis=(0.12, -0.08, 1.0), q_ref=0.0, refl=PASSIVE, mode="uniform_solid"):
    mesh = box_mesh([0.05, -0.02, 0.0], [0.55, 0.03, 0.7])
    m = GeometryMassModel.from_validated_mesh(mesh, mode, jt, axis, None if jt == "prismatic" else (0, 0, 0),
                                              source_provenance=dict(source="t"), reflected_actuator_inertia=refl,
                                              q_ref=q_ref)
    return m, mesh


def spec_of(m, terms=GeneralTerms(static=False)):
    return ModelSpec(terms, m.joint_type, "geometry_coupled", m, GF)


def fake_record(jt, axis, origin, n=200, seed=0):
    r = np.random.default_rng(seed)
    src = np.arange(1000, 1000 + n)
    F = r.normal(0, 3, (n, 3))
    P = np.c_[0.5 + 0.01 * np.arange(n) / n, np.full(n, 0.1), np.full(n, 0.4)]
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    Q = F @ a if jt == "prismatic" else np.einsum("j,nj->n", a, np.cross(P - origin, F))
    rec = dict(source_frame_index=src, F_object_base_N=F, p_tcp_base_m=P, in_contact_segment=np.ones(n, bool),
               M_sensor_Nm=r.normal(0, 5, (n, 3)), object_side_multiplier=-1.0)
    t = np.arange(0, n, 4) / 50.0 + src[0] / 50.0
    it = G9Interaction("w", jt, t, 0.001 * np.arange(len(t)), np.full(len(t), 1e-3), ZOH(src / 50.0, Q))
    it.meta["_record_slice"] = (0, n)
    return rec, it


# ===================================================================================== C. method parity
class MethodParity(unittest.TestCase):
    def test_C01_revolute_all_representations_agree(self):
        m, _ = geo()
        rep = GA.method_gravity_parity(spec_of(m), np.array([420.0, 0.3, 0.07]))
        self.assertEqual(rep["status"], "PASS", rep["checks"])
        self.assertEqual(rep["dr_drho_form"], "I_axis,1 qdd + G1(q)")

    def test_C02_prismatic(self):
        m, _ = geo("prismatic", axis=(0.3, 0.0, 0.95))
        rep = GA.method_gravity_parity(spec_of(m), np.array([150.0, 2.0, 0.5]))
        self.assertEqual(rep["status"], "PASS", rep["checks"])
        self.assertEqual(rep["dr_drho_form"], "m1 qdd - m1 g.a")

    def test_C03_nonzero_q_ref_and_reflected(self):
        m, _ = geo(q_ref=-0.41, refl=0.02)
        rep = GA.method_gravity_parity(spec_of(m), np.array([300.0, 0.2, 0.05]))
        self.assertEqual(rep["status"], "PASS", rep["checks"])
        self.assertEqual(rep["one_density"]["q_ref"], -0.41)
        self.assertEqual(rep["one_density"]["reflected"], 0.02)

    def test_C04_pinn_physics_gradient_through_one_density(self):
        m, _ = geo()
        spec = spec_of(m, GeneralTerms(coulomb=False, static=False))
        dyn = TorchDynamics(spec)
        eta = {k: v.detach().clone().requires_grad_(True) for k, v in dyn.eta0([500.0, 0.4]).items()}
        t = torch.linspace(0, 1, 50, dtype=DT)
        q, qd, qdd = 0.4 * torch.sin(3 * t), 1.2 * torch.cos(3 * t), -3.6 * torch.sin(3 * t)
        Q = torch.as_tensor(rng.normal(0, 1, 50), dtype=DT)
        r = dyn.residual(dyn.values(eta), q, qd, qdd, Q)
        L = 0.5 * (r ** 2).sum()
        g = torch.autograd.grad(L, eta["alpha"])[0] / 500.0                    # d/d rho (log parameterisation)
        gu = m.gravity_unit(G)
        dr = m.M1 * qdd.numpy() + gu.G(q.numpy())
        self.assertAlmostEqual(float(g), float(np.sum(r.detach().numpy() * dr)), places=8)

    def test_C05_impulse_integrals_vs_quadrature(self):
        e = ZOH(np.sort(rng.uniform(0, 2, 30)), rng.normal(0, 5, 30))
        for _ in range(5):
            a, b = np.sort(rng.uniform(-0.2, 2.2, 2))
            tt = np.linspace(a, b, 400001)
            lev, ts = np.asarray(e.levels), np.asarray(e.t_s)
            f = lev[np.clip(np.searchsorted(ts, tt, side="right") - 1, 0, len(lev) - 1)]
            self.assertAlmostEqual(zoh_integral(e, a, b), float(np.trapz(f, tt)), delta=1e-3 * max(1, abs(b - a)))

    def test_C06_force_sign_applied_once(self):
        from estimotion_2part.joint_discovery.force_convention import (ObjectBaseForce, SensorCorrForce,
                                                                       generalized_effort, sensor_corr_to_object_base)
        F = rng.normal(0, 2, (5, 3))
        R = np.repeat(np.eye(3)[None], 5, 0)
        Fo = sensor_corr_to_object_base(SensorCorrForce(F), R, np.eye(3), -1.0)
        np.testing.assert_allclose(Fo.values, -F)
        with self.assertRaises(TypeError):
            sensor_corr_to_object_base(Fo, R, np.eye(3), -1.0)             # an object-side force is never re-converted
        np.testing.assert_allclose(generalized_effort("prismatic", (1, 0, 0), None, Fo), -F[:, 0])

    def test_C07_reprojection_identity_and_same_timeline(self):
        a, o = np.array([0.1, 0.0, 1.0]) / np.linalg.norm([0.1, 0.0, 1.0]), np.zeros(3)
        rec, it = fake_record("revolute", a, o)
        same = GA._scenario_interactions([it], rec, "revolute", a, o)[0]
        np.testing.assert_allclose(same.effort.levels, it.effort.levels, atol=1e-12)
        np.testing.assert_array_equal(same.effort.t_s, it.effort.t_s)
        tilted = GA._scenario_interactions([it], rec, "revolute", GA._unit(a + [0.05, 0.02, 0]), o)[0]
        np.testing.assert_array_equal(tilted.effort.t_s, it.effort.t_s)    # same native force clock
        self.assertGreater(np.max(np.abs(np.asarray(tilted.effort.levels) - it.effort.levels)), 1e-4)

    def test_C08_unknown_sensor_moment_excluded(self):
        a, o = np.array([0.0, 0.0, 1.0]), np.array([0.2, 0.0, 0.0])
        rec, it = fake_record("revolute", a, o)
        Q = GA._effort_for(rec, "revolute", a, o, (0, len(rec["source_frame_index"])))
        np.testing.assert_allclose(Q, np.einsum("j,nj->n", a, np.cross(rec["p_tcp_base_m"] - o, rec["F_object_base_N"])))
        rec2 = dict(rec, M_sensor_Nm=rec["M_sensor_Nm"] * 100)
        np.testing.assert_allclose(GA._effort_for(rec2, "revolute", a, o, (0, len(Q))), Q)

    def test_C09_at_bound_coefficients_not_resolved(self):
        from forceartgs_g9.core import method_comparison
        obj = make_object("prismatic", terms=GeneralTerms(static=False, gravity=False))
        it, _ = episode(obj, variant=0, seed=0)
        spec = ModelSpec(GeneralTerms(static=False, gravity=False), "prismatic", "geometry_coupled", obj.mass_model, None)
        th = np.array([obj.density, 1e-13, obj.Fc])
        fake = dict(status="ok", x=th.tolist(), estimates={
            "alpha": dict(value=obj.density, std=1.0), "c": dict(value=1e-13, std=1e-18, at_lower_bound=True),
            "Fc": dict(value=obj.Fc, std=0.01)})
        mc = method_comparison(spec, None, {"general_pinn": fake}, None, None, [], "t", train=[it])
        c = mc["methods"]["general_pinn"]["parameters"]["c"]
        self.assertTrue(c["at_bound"])
        self.assertIsNone(c["std"])
        self.assertFalse(c["resolved_by_own_uncertainty"])
        self.assertIn("AT_NUMERICAL_FLOOR", c["uncertainty_label"])
        self.assertIn("replay_rmse_over_sigma_at_zero", c["profile_at_zero"])

    def test_C10_frozen_physics_library_untouched(self):
        p = physics_lib.provenance()
        self.assertTrue(p["pinned"], p)
        self.assertEqual(p["dirty_files"], 0)
        self.assertEqual(p["commit"], physics_lib.PINNED_COMMIT)

    def test_C11_fixed_theta_open_loop_replay_synthetic(self):
        from forceartgs_g9.estimators.general_ekf import _replay
        for jt in ("prismatic", "revolute"):
            obj = make_object(jt, terms=GeneralTerms(static=False), g=G)
            spec = ModelSpec(obj.terms, jt, "geometry_coupled", obj.mass_model, obj.gravity)
            its = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
            truth = np.array([obj.density, obj.c, obj.Fc])
            rr = [x["rmse_over_sigma"] for x in _replay(spec, truth, its)["per_interaction"]]
            self.assertLess(max(rr), 5.0, (jt, rr))
            off = truth * [1.5, 1.0, 1.0]
            self.assertGreater(min(x["rmse_over_sigma"] for x in _replay(spec, off, its)["per_interaction"]),
                               max(rr))

    def test_C12_sensitivity_scenarios_consistent(self):
        obj = make_object("revolute", terms=GeneralTerms(static=False), g=G)
        m, mesh = geo(axis=(0.1, 0.0, 1.0))
        spec = spec_of(m)
        rec, it = fake_record("revolute", m.axis, m.origin, n=120)
        rep = GA.gravity_joint_sensitivity(spec, mesh, "uniform_solid", {"x": np.array([200.0, 0.3, 0.05])}, [it], rec,
                                           observability=dict(axis_sigma_deg=1.5, hinge_line_sigma_m=0.01), n_random=3)
        by = {s["name"]: s for s in rep["scenarios"]}
        self.assertTrue(rep["vertical_alternative"]["G_identically_zero"])
        self.assertIn("ALTERNATIVE_KINEMATIC_HYPOTHESIS", by["exactly_vertical_hinge"]["label"])
        self.assertIn("NOT_RUN", rep["vertical_alternative"]["upstream_kinematic_residual"])
        tilt = by["axis_tilt_u1_+1sigma"]
        self.assertGreater(tilt["effort_rms_change"], 0)                  # effort reprojected
        self.assertNotAlmostEqual(tilt["G1_amplitude"], by["accepted"]["G1_amplitude"], places=6)   # gravity changes
        pm = tilt["per_method"]["x"]
        self.assertIn("replay_rmse_over_sigma_reprojected_effort", pm)
        self.assertIn("replay_rmse_over_sigma_GRAVITY_ONLY_COUNTERFACTUAL", pm)
        sh = by["mass_mode_uniform_shell"]["per_method"]["x"]
        self.assertAlmostEqual(sh["M"], by["accepted"]["per_method"]["x"]["M"], places=10)   # density re-matched


# ===================================================================================== D. real diagnostics
SSD = Path("/media/samuelemara/Extreme SSD/Samuele")
RUNS = SSD / "results_19_11_26" / "forceartgs_joint_aware"
SC = "final_red_fridge_episode_000000"
SCENE = SSD / "Backup_09_10_26" / "Sapien" / "output" / "real" / SC
ARCH = SSD / "results_17_11_26" / "overnight_real" / "joint" / SC / "robot_archive.npz"
FRIDGE = sorted(RUNS.glob(f"{SC}_multiview_*_joint_aware_full")) if RUNS.is_dir() else []


@unittest.skipUnless(FRIDGE and SCENE.is_dir() and ARCH.is_file(), "real fridge data not mounted")
class RealFridgeDiagnostics(unittest.TestCase):
    """Read-only: builds inputs in memory and writes only to a temporary directory."""

    @classmethod
    def setUpClass(cls):
        from forceartgs_g9.adapters.forceartgs_g9 import build_inputs
        from forceartgs_g9.config import load_dynamics_config
        root = Path(__file__).resolve().parents[1]
        cfg = load_dynamics_config(str(root / "configs/g9_real/exploratory_geo_solid_viscous_gravity.yaml"))
        cfg["pipeline"]["allow_diagnostic_kinematics"] = True
        cfg["identifiability"]["declared_pulse_rows"] = [[229, 232]]
        cfg["fitting"]["methods"] = ["general_integral_ls", "general_rollout_fit"]
        cls.cfg = cfg
        cls.inp = build_inputs(FRIDGE[0], SCENE, cfg, robot_archive=ARCH,
                               contact_segments=root / f"configs/contact_intervals/{SC}.yaml", log=lambda m: None)

    def test_D01_provenance_and_reference_pose(self):
        r = self.inp.reference_pose
        self.assertEqual(r["status"], "PASS", r.get("reasons"))
        self.assertEqual(r["anchor_original_frame"], 90)
        self.assertEqual(r["authority_source_row_at_anchor"], 180)
        self.assertAlmostEqual(r["q_ref"], 0.0, places=9)
        self.assertIn("UNVERIFIED", r["frames"]["T_B_Wc"])                  # applied registration, not validated
        self.assertEqual(self.inp.gravity.g_world, (0.0, 0.0, -9.81))
        self.assertIn("DECLARED", self.inp.gravity.provenance)
        self.assertEqual(self.inp.mass_mesh_provenance["status"], "TENTATIVE_GEOMETRIC_PREVIEW")

    def test_D02_solid_shell_unit_audit_and_implied_mass(self):
        i = self.inp
        rep = GA.mass_com_inertia_audit(i.mass_mesh, i.joint_type, i.axis, i.origin, i.gravity.g_world, 0.0,
                                        densities={"pinn_solid_reported": 1.44}, fitted_mode="uniform_solid")
        s, h = rep["modes"]["uniform_solid"], rep["modes"]["uniform_shell"]
        self.assertAlmostEqual(s["m1"], 0.0609, delta=0.002)
        self.assertAlmostEqual(h["m1"], 0.941, delta=0.02)
        self.assertAlmostEqual(s["I_axis1"], 0.00351, delta=2e-4)
        self.assertAlmostEqual(rep["implied_by_method"]["pinn_solid_reported"]["mass_kg"], 1.44 * s["m1"], places=12)
        self.assertAlmostEqual(0.0878 * h["m1"], 0.083, delta=0.003)       # the transcript's shell figure
        self.assertTrue(rep["hull"]["watertight"])
        self.assertAlmostEqual(s["G1_amplitude"], s["G1_amplitude_analytic"], places=10)

    def test_D03_tilt_vs_vertical_alternative(self):
        a = self.inp.axis / np.linalg.norm(self.inp.axis)
        self.assertAlmostEqual(math.degrees(math.acos(abs(a[2]))), 6.03, delta=0.05)   # never rounded to 0
        m = GeometryMassModel.from_validated_mesh(self.inp.mass_mesh, "uniform_solid", "revolute", a, self.inp.origin,
                                                  source_provenance=dict(source="t"),
                                                  reflected_actuator_inertia=PASSIVE)
        self.assertGreater(m.gravity_unit(self.inp.gravity.g_world).amplitude_bound(), 1e-3)
        v = GeometryMassModel.from_validated_mesh(self.inp.mass_mesh, "uniform_solid", "revolute", (0, 0, 1),
                                                  self.inp.origin, source_provenance=dict(source="t"),
                                                  reflected_actuator_inertia=PASSIVE)
        self.assertLess(v.gravity_unit(self.inp.gravity.g_world).amplitude_bound(), 1e-12)

    def test_D04_axis_perturbation_changes_effort_and_gravity(self):
        i = self.inp
        rec = i.record_context["rec"]
        a = i.axis / np.linalg.norm(i.axis)
        u, _ = GA._perp_basis(a)
        a2 = GA._unit(a + math.tan(math.radians(1.6)) * u)
        base, pert = i.train[0], GA._scenario_interactions(i.train, rec, "revolute", a2, i.origin)[0]
        self.assertGreater(np.max(np.abs(np.asarray(pert.effort.levels) - base.effort.levels)), 1e-5)
        rb = lambda ax: GeometryMassModel.from_validated_mesh(i.mass_mesh, "uniform_solid", "revolute", ax, i.origin,
                                                              source_provenance=dict(source="t"),
                                                              reflected_actuator_inertia=PASSIVE).gravity_unit(
            i.gravity.g_world).amplitude_bound()
        self.assertNotAlmostEqual(rb(a), rb(a2), places=5)

    def test_D05_impulse_sign_discrepancy_preserved(self):
        from forceartgs_g9.validation.impulse_coast import analyze
        rc = self.inp.record_context
        it = next(x for x in self.inp.train if x.meta["source_rows"][0] <= 229 <= x.meta["source_rows"][1])
        an = analyze(rc["rec"], rc["joint_type"], rc["axis"], rc["origin"], rc["q_src"], rc["q"], rc["q_std"],
                     it.meta["source_rows"], observability=rc["observability"], declared_pulse_rows=[[229, 232]])
        dp = an["declared_pulses"][0]
        self.assertGreater(dp["pulse"]["J_Q"], 0)
        self.assertLess(dp["velocity"]["dv"], 0)
        self.assertTrue(dp["J_over_dqdot_CONDITIONAL_M"]["sign_contradiction"])
        self.assertEqual(float(rc["rec"]["object_side_multiplier"]), -1.0)  # declared sign untouched
        self.assertNotEqual(an["verdict"], "IMPULSE_INFORMATIVE")

    def test_D06_no_physical_identification_from_tentative_geometry(self):
        from forceartgs_g9.core import run_g9
        with tempfile.TemporaryDirectory() as td:
            r = run_g9(self.inp, self.cfg, Path(td) / "g9", log=lambda m: None)
            man = r["manifest"]
            self.assertNotEqual(man["status"], "PRODUCTION_PHYSICS_SPEC_READY")
            self.assertIsNone(r["physics_spec_v1"])
            ga = json.loads((Path(td) / "g9" / "gravity_geometry_audit.json").read_text())
            self.assertIn("TENTATIVE_GEOMETRIC_PREVIEW", ga["flags"])
            self.assertEqual(ga["gravity"]["provenance_class"], "DECLARED_OPERATOR_VERTICAL")
            ib = json.loads((Path(td) / "g9" / "impulse_balance_audit.json").read_text())
            self.assertIn(ib["pulses"][0]["verdict"], ("SIGN_OR_TIMING_INCONCLUSIVE",
                                                       "IMPULSE_DYNAMIC_MODEL_CONTRADICTED"))
            for n in ("reference_pose_alignment", "mass_com_inertia_audit", "gravity_joint_sensitivity",
                      "method_gravity_parity"):
                self.assertTrue((Path(td) / "g9" / f"{n}.json").is_file(), n)
            self.assertEqual(json.loads((Path(td) / "g9" / "method_gravity_parity.json").read_text())["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
