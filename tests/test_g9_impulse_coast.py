"""G9 corrections: force sign contract, impulse vs coast, effective load vs bias, scenarios vs statistics, verdicts."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.force_convention import (ForceConventionError, ObjectBaseForce,  # noqa: E402
                                                               SensorCorrForce, generalized_effort, object_side_sign,
                                                               sensor_corr_to_object_base)
from forceartgs_g9 import physics_lib  # noqa: E402
from forceartgs_g9.dynamics.general_1dof import GeneralTerms, rotation, sim_params, simulate_general  # noqa: E402
from forceartgs_g9.synthetic import ZOH  # noqa: E402
from forceartgs_g9.validation.impulse_coast import analyze  # noqa: E402

physics_lib.load()
CONV = dict(object_side_force_multiplier=-1, recorded_quantity="contact_reaction_force_on_robot",
            force_acting_on="robot")


def pulse_coast_record(M=2.0, c=4.0, F_pulse=20.0, n_pulse=4, T=3.0, t_pulse=1.0, q_hz=25.0, q_noise=5e-4, seed=0,
                       sign=1.0):
    """Prismatic slider along +x: rest, a 4-sample (80 ms) push, free coast. Record in the canonical format."""
    rng = np.random.default_rng(seed)
    src = np.arange(int(T * 50))
    t = src / 50.0
    F = np.zeros((len(src), 3))
    k0 = int(t_pulse * 50)
    F[k0:k0 + n_pulse, 0] = F_pulse
    p = sim_params(GeneralTerms(coulomb=False, static=False, gravity=False), "prismatic", M=M, c=c)
    tq_src = np.arange(0, len(src), int(50 / q_hz))
    r = simulate_general(p, ZOH(t, F[:, 0]), tq_src / 50.0, max_step=5e-4)
    q = r.q + q_noise * rng.standard_normal(len(tq_src))
    state = np.where(np.linalg.norm(F, axis=1) > 0, "contact", "absent")
    seg = np.zeros(len(src), bool)
    seg[k0 - 5:k0 + n_pulse + 40] = True
    rec = dict(source_frame_index=src, timestamp_s=t + 2e-4, F_object_base_N=sign * F + 0.02 * rng.standard_normal(F.shape),
               p_tcp_base_m=np.zeros((len(src), 3)), contact_state=state, in_contact_segment=seg,
               force_sigma_N=np.array([0.02, 0.02, 0.02]), force_conversion={})
    return rec, tq_src, q, np.full(len(q), q_noise), dict(M=M, c=c, J=F_pulse * n_pulse / 50.0)


class SignContract(unittest.TestCase):
    def test_reaction_on_robot_opposite_on_object_exactly_once(self):
        s, prov = object_side_sign(CONV)
        self.assertEqual(s, -1.0)
        f = sensor_corr_to_object_base(SensorCorrForce(np.array([[-2.0, 0, 0]])), np.eye(3)[None], np.eye(3), s)
        np.testing.assert_allclose(f.values, [[2.0, 0, 0]])
        self.assertAlmostEqual(float(generalized_effort("prismatic", [1, 0, 0], None, f)[0]), 2.0)
        r = 0.4
        fy = ObjectBaseForce(np.array([[0.0, 2.0, 0.0]]))
        Q = generalized_effort("revolute", [0, 0, 1], [0, 0, 0], fy, np.array([[r, 0, 0]]))
        self.assertAlmostEqual(float(Q[0]), 2 * r)
        with self.assertRaises(TypeError):                         # a converted force is never converted again
            sensor_corr_to_object_base(f, np.eye(3)[None], np.eye(3), s)

    def test_rotation_power_invariance_and_rejections(self):
        R = rotation([0.3, -0.5, 0.8], 0.9)[None]
        Fs = np.array([[0.4, -1.1, 2.0]])
        f = sensor_corr_to_object_base(SensorCorrForce(Fs), R, np.eye(3), -1.0)
        np.testing.assert_allclose(f.values, -(R[0] @ Fs[0])[None])
        v = np.array([0.2, 0.1, -0.3])
        self.assertAlmostEqual(float(f.values[0] @ (R[0] @ v)), float(-Fs[0] @ v))    # power is frame-invariant
        for bad in ({}, dict(object_side_force_multiplier=0), dict(object_side_force_multiplier=None),
                    dict(object_side_force_multiplier=float("nan")), dict(object_side_force_multiplier=-2),
                    dict(object_side_force_multiplier=1, recorded_quantity="contact_reaction_force_on_robot")):
            with self.assertRaises(ForceConventionError):
                object_side_sign(bad)
        with self.assertRaises(ForceConventionError):
            sensor_corr_to_object_base(SensorCorrForce(Fs), np.diag([1.0, 1.0, -1.0])[None], np.eye(3), -1.0)


class ImpulseCoast(unittest.TestCase):
    def test_known_pulse_and_coast(self):
        rec, qs, q, sq, tr = pulse_coast_record()
        an = analyze(rec, "prismatic", [1, 0, 0], None, qs, q, sq, (0, 149))
        main = max(an["pulses"], key=lambda p: abs(p["pulse"]["J_Q"]))
        self.assertEqual(main["pulse"]["n_force_samples"], 4)                    # 80 ms stays 4 native samples
        self.assertAlmostEqual(main["pulse"]["J_Q"], tr["J"], delta=0.02)
        self.assertEqual(main["verdict"], "IMPULSE_INFORMATIVE")
        self.assertGreater(main["pulse"]["SNR_pulse_peak"], 100)
        self.assertLess(main["coast"]["SNR_coast"], 3)                          # high pulse, low-force coast
        tau = main["coast"]["fit"]["decay_time_M_over_c_s"]["estimate"]
        self.assertAlmostEqual(tau, tr["M"] / tr["c"], delta=0.1 * tr["M"] / tr["c"])
        self.assertAlmostEqual(main["impulse_only_conditional_M"]["estimate"], tr["M"], delta=0.15 * tr["M"])
        self.assertEqual(main["impulse_only_conditional_M"]["label"], "CONDITIONAL")
        self.assertIn("lag_+10ms", main["sensitivity"])

    def test_sparse_q_around_pulse_is_unresolved(self):
        rec, qs, q, sq, _ = pulse_coast_record()
        keep = (qs < 40) | (qs > 75)                                            # q missing around the pulse
        an = analyze(rec, "prismatic", [1, 0, 0], None, qs[keep], q[keep], sq[keep], (0, 149))
        main = max(an["pulses"], key=lambda p: abs(p["pulse"]["J_Q"]))
        self.assertEqual(main["verdict"], "IMPULSE_UNRESOLVED")

    def test_negative_work_reports_contradiction_never_flips(self):
        rec, qs, q, sq, _ = pulse_coast_record()
        rec["F_object_base_N"] = -rec["F_object_base_N"]                        # a sign error in the input
        an = analyze(rec, "prismatic", [1, 0, 0], None, qs, q, sq, (0, 149))
        main = max(an["pulses"], key=lambda p: abs(p["pulse"]["J_Q"]))
        self.assertTrue(main["sign_consistency"]["contradiction"])
        self.assertNotEqual(main["verdict"], "IMPULSE_INFORMATIVE")
        self.assertLess(main["pulse"]["J_Q"], 0)                                # the input sign was NOT changed


class LoadAndUncertainty(unittest.TestCase):
    def test_effective_load_vs_bias_short_range(self):
        from forceartgs_g9.estimators.model_spec import ModelSpec
        from forceartgs_g9.synthetic import episode, make_object
        from forceartgs_g9.validation.load_models import load_model_comparison
        obj = make_object("revolute", hinge="vertical", terms=GeneralTerms(static=False, gravity=False, bias=True),
                          b=0.05)
        its = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
        lm = load_model_comparison(its, GeneralTerms(static=False), "revolute", n_starts=1, lags_s=(0.01,))
        self.assertIn("C_effective_load", lm["variants"])
        self.assertGreater(lm["basis_condition_number_D"], lm["basis_condition_number_C"])
        self.assertIn("never physical gravity", lm["label_note"])
        self.assertEqual(lm["variants"]["C_effective_load"]["effective_configuration_load"]["label"], "CONDITIONAL")
        js = ModelSpec(GeneralTerms(static=False), "revolute", "joint_space_only")
        g = js.physical(np.array([0.1, 0.1, 0.1, 0.2, 0.3]))["gravity"]
        self.assertEqual(g["name"], "effective_configuration_load")
        self.assertFalse(g["physical_gravity"])
        gc = ModelSpec(obj.terms if obj.terms.gravity else GeneralTerms(), "revolute", "geometry_coupled",
                       obj.mass_model, obj.gravity).physical(np.array([obj.density, 0.1, 0.1, 0.1]))["gravity"]
        self.assertEqual(gc["kind"], "geometry_coupled")                       # physical gravity path unchanged

    def test_scenarios_do_not_change_statistical_ci(self):
        from forceartgs_g9.estimators.general_constrained import general_rollout_fit
        from forceartgs_g9.estimators.general_integral_ls import general_integral_ls, theta0_from_ls
        from forceartgs_g9.estimators.model_spec import ModelSpec
        from forceartgs_g9.synthetic import episode, make_object
        from forceartgs_g9.validation.general_identifiability import diagnose_general, standard_perturbations
        obj = make_object("prismatic", terms=GeneralTerms(static=False, gravity=False))
        its = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
        spec = ModelSpec(obj.terms, "prismatic", "geometry_coupled", obj.mass_model, obj.gravity)
        ls = general_integral_ls(its, spec)
        f = general_rollout_fit(its, spec, theta0_from_ls(ls, spec, dict(effort=5.0)), n_starts=1)
        d0 = diagnose_general(f, its, spec, perturb={})
        d1 = diagnose_general(f, its, spec, perturb=standard_perturbations(scale_rel=0.05, lag_s=0.01))
        for n in ("alpha", "c", "Fc"):
            self.assertEqual(d0["parameters"][n]["ci95_statistical"], d1["parameters"][n]["ci95_statistical"])
            self.assertEqual(d1["parameters"][n]["std_total"], d1["parameters"][n]["std_statistical"])   # no quadrature
            self.assertGreater(d1["parameters"][n]["scenario_worst_single_family"], 0)
        fam = d1["parameters"]["c"]["sensitivity_scenarios"]
        self.assertTrue(any(k.startswith("force_scale") and set(v["shifts"]) == {"+", "-"} for k, v in fam.items()))
        self.assertIn("sensitivity_scenarios", d1["derived_combinations"]["decay_time_M_over_c_s"])

    def test_lever_scenarios_only_for_revolute(self):
        from types import SimpleNamespace
        from forceartgs_g9.adapters.forceartgs_g9 import record_perturbations
        from forceartgs_g9.config import load_dynamics_config
        rec, qs, q, sq, _ = pulse_coast_record()
        rec["object_side_multiplier"] = -1.0
        rec["R_base_ee"] = np.tile(np.eye(3), (len(rec["source_frame_index"]), 1, 1))
        rec["evidence"] = dict(tare_second_window=dict(status="NOT_ESTABLISHED"))
        cfg = load_dynamics_config({"identifiability": {"contact_point_sweep_m": 0.015}})
        for kind, n_lever in (("prismatic", 0), ("revolute", 6)):
            auth = SimpleNamespace(joint_type=kind, axis_world=np.array([0, 0, 1.0]),
                                   origin_world=None if kind == "prismatic" else np.array([0.3, 0, 0]))
            pert = record_perturbations(rec, auth, cfg, [], dict(axis_sigma_deg=1.0, hinge_line_sigma_m=0.02))
            self.assertEqual(sum(k.startswith("tcp_contact") for k in pert), n_lever, kind)
            self.assertTrue(any(k.startswith("hinge_axis_tilt") for k in pert))
            self.assertEqual(any(k.startswith("hinge_line_offset") for k in pert), kind == "revolute")

    def test_rejected_model_never_physically_interpretable(self):
        from forceartgs_g9.core import verdicts
        table = {"viscous": dict(parameters={"c": dict(fitted_or_fixed="fitted",
                                                       parameter_statistically_resolved_in_this_model=True,
                                                       scenario_sensitive=False)})}
        verdicts(table, None, dict(status="FAIL"), kin_ok=False, force_ok=False, held_ok=False, struct={})
        c = table["viscous"]["parameters"]["c"]
        self.assertFalse(c["physically_interpretable_validated"])
        self.assertFalse(c["conditionally_interpretable"])
        self.assertIn("model adequacy FAIL", c["interpretation_blockers"])
        self.assertIn("joint type not accepted (working model)", c["interpretation_blockers"])


if __name__ == "__main__":
    unittest.main()


class DeclaredGravityCoupled(unittest.TestCase):
    """Declared vertical (dataset z is true vertical): joint-space gravity tied to the physics without a mesh."""

    def _fit(self, obj, its, **kw):
        from forceartgs_g9.estimators.general_constrained import general_rollout_fit
        from forceartgs_g9.estimators.general_integral_ls import general_integral_ls, theta0_from_ls
        from forceartgs_g9.estimators.model_spec import ModelSpec
        from forceartgs_g9.dynamics.general_1dof import GravityField
        g = GravityField((0, 0, -9.81), False, "DECLARED by the operator: test")
        spec = ModelSpec(obj.terms, obj.joint_type, "joint_space_gravity_coupled", None, g,
                         allow_unverified_gravity=True, joint_axis=obj.axis, **kw)
        ls = general_integral_ls(its, spec)
        f = general_rollout_fit(its, spec, theta0_from_ls(ls, spec, dict(effort=5.0)), n_starts=2)
        return spec, f

    def test_inclined_slider_mass_couples_gravity(self):
        from forceartgs_g9.synthetic import episode, make_object
        obj = make_object("prismatic", axis=[0.0, 0.99985, 0.0175], terms=GeneralTerms(static=False),
                          g=(0, 0, -9.81))
        its = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
        spec, f = self._fit(obj, its)
        self.assertNotIn("gP", spec.names)                             # no free gravity parameter
        self.assertAlmostEqual(f["physical"]["M"] / obj.M_true, 1.0, delta=0.03)
        self.assertAlmostEqual(f["physical"]["c"] / obj.c, 1.0, delta=0.08)
        self.assertTrue(f["physical"]["gravity"]["physical_gravity"])
        self.assertIn("DECLARED", f["physical"]["gravity"]["gravity_status"])

    def test_tilted_hinge_mass_moment(self):
        from forceartgs_g9.synthetic import episode, make_object
        obj = make_object("revolute", terms=GeneralTerms(static=False), g=(0, 0, -9.81))   # hanging flap
        its = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
        c1 = obj.mass_model.props.c1
        r = c1 - obj.origin
        r = r - (r @ obj.axis) * obj.axis
        u, lev = r / np.linalg.norm(r), float(np.linalg.norm(r))
        spec, f = self._fit(obj, its, joint_origin=obj.origin, com_direction=u, com_lever_m=lev)
        mu_true = obj.density * obj.mass_model.props.m1 * lev
        self.assertAlmostEqual(f["estimates"]["mu"]["value"] / mu_true, 1.0, delta=0.03)
        self.assertTrue(f["physical"]["gravity"]["consistency_M_ge_mu_r"]["ok"])
        with self.assertRaises(Exception):                             # declared gravity needs the explicit opt-in
            from forceartgs_g9.estimators.model_spec import ModelSpec
            from forceartgs_g9.dynamics.general_1dof import GravityField
            ModelSpec(obj.terms, "revolute", "joint_space_gravity_coupled", None,
                      GravityField((0, 0, -9.81), False, "DECLARED"), joint_axis=obj.axis, joint_origin=obj.origin,
                      com_direction=u)
