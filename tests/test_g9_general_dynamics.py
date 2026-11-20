"""G9 Gate A: physics equations and switches (synthetic truth only)."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9 import physics_lib  # noqa: E402
from forceartgs_g9.dynamics.general_1dof import (GeneralTerms, ModelConfigError, RigidBodyGravity,  # noqa: E402
                                                 mechanical_energy, quasistatic_equilibrium, sim_params,
                                                 simulate_general)
from forceartgs_g9.estimators.model_spec import GravityRegistrationError, ModelSpec  # noqa: E402
from forceartgs_g9.synthetic import ZOH, force_record_for, make_object, profile  # noqa: E402
from forceartgs_g9.validation.simulator_parity import term_matrix  # noqa: E402

physics_lib.load()


def eff(segs, hz=100.0):
    t, Q = profile(segs, hz)
    return ZOH(t, Q)


class GateA(unittest.TestCase):
    def test_a01_generalized_force_power_identity(self):
        from forceartgs_physics.forces.projection import generalized_effort
        from forceartgs_physics.schemas import ForceRecord, JointKinematics
        rng = np.random.default_rng(0)
        for kind in ("revolute", "prismatic"):
            obj = make_object(kind)
            t = np.linspace(0, 1, 50)
            q = 0.3 * np.sin(2 * t)
            qd = 0.6 * np.cos(2 * t)
            P = np.array([obj.contact_point(x) for x in q])
            F = rng.standard_normal((50, 3)) * 5.0                       # arbitrary direction (not along motion)
            j = JointKinematics(kind, obj.axis, obj.origin, "W")
            Q = generalized_effort(j, ForceRecord(t_s=t, force_N=F, point_m=P, frame="W")).levels
            v_mat = (np.cross(obj.axis, P - obj.origin) if kind == "revolute" else np.tile(obj.axis, (50, 1))) * qd[:, None]
            np.testing.assert_allclose(Q * qd, np.sum(F * v_mat, 1), atol=1e-10)
            # and the synthetic force builder reproduces a requested effort exactly
            P2, F2 = force_record_for(obj, t, Q, t, q)
            Q2 = generalized_effort(j, ForceRecord(t_s=t, force_N=F2, point_m=P2, frame="W")).levels
            np.testing.assert_allclose(Q2, Q, atol=1e-9)

    def test_a02_gravity_is_dU_dq_and_signs(self):
        g = (0.0, 0.0, -9.81)
        for kind, axis in (("revolute", (1.0, 0, 0)), ("prismatic", (0, 0.6, 0.8))):
            rb = RigidBodyGravity(kind, axis, (0, 0, 0.5), (0.03, 0.2, 0.3), 2.5, g)
            for q in np.linspace(-1.2, 1.2, 9):
                h = 1e-6
                fd = (rb.potential(q + h) - rb.potential(q - h)) / (2 * h)
                self.assertAlmostEqual(rb.G(q), fd, places=6)
            # no applied effort, no friction: q'' = -G/M (sign)
            M = 0.7
            q0 = 0.4
            p = sim_params(GeneralTerms(coulomb=False, static=False, viscous=False), kind, M=M, G=rb.G)
            r = simulate_general(p, eff([(1.0, 0.0)]), [1e-4], q0=q0, max_step=1e-5)
            acc = 2 * (r.q[0] - q0) / 1e-8
            self.assertAlmostEqual(acc, -rb.G(q0) / M, delta=1e-3 * max(1, abs(rb.G(q0) / M)))

    def test_a03_vertical_horizontal_and_prismatic_gravity(self):
        g = (0.0, 0.0, -9.81)
        vert = RigidBodyGravity("revolute", (0, 0, 1), (0, 0, 0), (0.2, 0.3, 0.4), 3.0, g)
        self.assertLess(max(abs(vert.G(q)) for q in np.linspace(-3, 3, 13)), 1e-12)
        hor = RigidBodyGravity("revolute", (1, 0, 0), (0, 0, 0), (0, 0.3, 0.0), 3.0, g)
        self.assertGreater(abs(hor.G(0.0)), 1.0)
        up = RigidBodyGravity("prismatic", (0, 0, 1), (0, 0, 0), (0, 0, 0), 3.0, g)
        self.assertAlmostEqual(up.G(0.1), up.G(0.7))
        self.assertAlmostEqual(up.G(0.0), 3.0 * 9.81)                  # gravity resists upward motion
        flat = RigidBodyGravity("prismatic", (1, 0, 0), (0, 0, 0), (0, 0, 0), 3.0, g)
        self.assertAlmostEqual(flat.G(0.0), 0.0)

    def test_a04_viscous_only_reproduces_frozen_exact_zoh_and_integral_ls(self):
        from forceartgs_physics.dynamics.viscous import propagate_from_state
        from forceartgs_physics.forces.projection import GeneralizedEffort
        t, Q = profile([(0.4, 0.0), (0.5, 2.0), (0.6, -1.0), (0.5, 0.5)], 100)
        e = GeneralizedEffort(t, Q, "revolute", "test")
        tq = np.linspace(0.0, 1.9, 77)
        ref = propagate_from_state(0.3, 0.7, e, tq, 0.0, 0.1, -0.2)
        p = sim_params(GeneralTerms(coulomb=False, static=False, gravity=False), "revolute", M=0.3, c=0.7)
        got = simulate_general(p, e, tq, q0=0.1, v0=-0.2, max_step=7e-3).q
        np.testing.assert_allclose(got, ref, atol=1e-12)
        # frozen IntegralLS on the same data vs the general integral LS restricted to the viscous model
        from forceartgs_g9.estimators.frozen_baselines import run_baselines
        from forceartgs_g9.estimators.general_integral_ls import general_integral_ls
        from forceartgs_g9.estimators.model_spec import G9Interaction
        rng = np.random.default_rng(1)
        tobs = np.arange(0, 1.95, 1 / 60)
        qobs = propagate_from_state(0.3, 0.7, e, tobs, 0.0, 0.0, 0.0) + 1e-4 * rng.standard_normal(len(tobs))
        it = G9Interaction("v", "revolute", tobs, qobs, 1e-4, ZOH(t, Q))
        b = run_baselines([it], [0, 0, 1.0], [0, 0, 0], ["IntegralLS"])["results"]["IntegralLS"][0]
        spec = ModelSpec(GeneralTerms(coulomb=False, static=False, gravity=False), "revolute", "joint_space_only")
        ls = general_integral_ls([it], spec)
        self.assertAlmostEqual(b["inertia"], 0.3, delta=0.01)
        self.assertAlmostEqual(ls["estimates"]["M"]["value"], 0.3, delta=0.01)
        self.assertAlmostEqual(ls["estimates"]["c"]["value"], b["damping"], delta=0.03)

    def test_a05_disabled_terms_are_exactly_zero_and_null(self):
        e = eff([(0.3, 0.0), (0.6, 3.0), (0.6, -3.0), (0.5, 0.0)])
        tq = np.linspace(0, 1.9, 40)
        base = dict(M=0.5, c=0.2, Fc=0.3, Fs=0.5, b=0.1, G=lambda q: 0.4 * math.sin(q))
        for t, key in (("viscous", "c"), ("coulomb", "Fc"), ("static", "Fs"), ("gravity", "G"), ("bias", "b")):
            on = GeneralTerms(bias=True)
            off = GeneralTerms(**{**on.to_dict(), t: False})
            with self.assertRaises(ModelConfigError):
                sim_params(off, "revolute", **base)                     # a value for a disabled term is refused
            kw = {k: v for k, v in base.items() if k != key}
            q_off = simulate_general(sim_params(off, "revolute", **kw), e, tq).q
            zero = dict(base, **{key: (lambda q: 0.0) if key == "G" else 0.0})
            if t == "static":
                zero["Fs"] = base["Fc"]                    # static OFF == no breakaway level above the Coulomb set
            q_zero = simulate_general(sim_params(on, "revolute", **zero), e, tq).q
            np.testing.assert_array_equal(q_off, q_zero)
            self.assertEqual(off.status()[t], "disabled_by_user")
            spec = ModelSpec(off, "revolute", "joint_space_only")
            self.assertFalse({"c": "c", "Fc": "Fc", "Fs": "dFs", "G": "gA", "b": "b"}[key] in spec.names)

    def test_a06_inertia_off_only_quasistatic(self):
        with self.assertRaises(ModelConfigError):
            GeneralTerms(inertia=False)
        qs = GeneralTerms(inertia=False, quasistatic=True)
        with self.assertRaises(ModelConfigError):
            sim_params(qs, "revolute", c=0.1, Fc=0.1, Fs=0.2, G=lambda q: 0.0)
        with self.assertRaises(ModelConfigError):
            sim_params(GeneralTerms(), "revolute", M=0.0, c=0.1, Fc=0.1, Fs=0.2, G=lambda q: 0.0)
        r = quasistatic_equilibrium(qs, [0.1, 0.5], Fc=0.2, Fs=0.3)
        self.assertEqual(r["dynamic_rollout"], "NOT_APPLICABLE_QUASISTATIC")

    def test_a07_coulomb_resists_both_signs_no_energy_creation(self):
        T = GeneralTerms(static=False, gravity=False)
        for v0 in (1.0, -1.0):
            p = sim_params(T, "prismatic", M=2.0, c=0.5, Fc=1.0)
            tq = np.linspace(0, 3, 300)
            r = simulate_general(p, eff([(3.0, 0.0)]), tq, v0=v0)
            E = mechanical_energy(p, r.q, r.qdot)
            self.assertTrue(np.all(np.diff(E) <= 1e-12))
            self.assertTrue(np.all(r.qdot * v0 >= 0))                 # friction never reverses the motion
            self.assertEqual(r.qdot[-1], 0.0)
            self.assertGreater(r.q[-1] * v0, 0)

    def test_a08_static_breakaway_stick_and_step_convergence(self):
        T = GeneralTerms(gravity=False)
        p = sim_params(T, "prismatic", M=1.0, c=0.2, Fc=1.0, Fs=2.0)
        tq = np.linspace(0, 2, 100)
        sub = simulate_general(p, eff([(2.0, 1.9)]), tq)
        self.assertTrue(np.all(sub.q == 0.0))                          # sub-breakaway: stays exactly at rest
        sup = simulate_general(p, eff([(2.0, 2.1)]), tq)
        self.assertGreater(sup.q[-1], 0.0)
        rel = simulate_general(p, eff([(0.5, 3.0), (1.5, 0.0)]), tq, record_events=True)
        self.assertEqual(rel.qdot[-1], 0.0)                             # released -> stops -> re-sticks
        self.assertTrue(any(e["kind"] == "stop" for e in rel.events))
        e = eff([(0.3, 0.0), (0.8, 0.0, 3.0), (0.6, 0.0)])
        qs = [simulate_general(p, e, [1.7], max_step=h).q[0] for h in (4e-3, 1e-3, 2.5e-4)]
        self.assertLess(abs(qs[0] - qs[-1]), 1e-9)                    # exact between events (no G(q) here)
        pg = sim_params(GeneralTerms(), "revolute", M=0.15, c=0.15, Fc=0.25, Fs=0.45,
                        G=lambda q: 5.0 * math.sin(q))
        qg = [simulate_general(pg, e, [1.7], max_step=h).q[0] for h in (4e-3, 1e-3, 2.5e-4, 6e-5)]
        self.assertLess(abs(qg[2] - qg[3]), abs(qg[0] - qg[3]) + 1e-15)
        self.assertLess(abs(qg[1] - qg[3]), 1e-5)                     # 2nd-order (Heun) configuration load

    def test_a09_no_sign0_hack_nan_or_invalid_parameters(self):
        T = GeneralTerms(gravity=False)
        with self.assertRaises(ModelConfigError):
            sim_params(T, "prismatic", M=1.0, c=0.1, Fc=2.0, Fs=1.0)    # Fs < Fc
        for bad in (dict(M=-1.0), dict(M=float("nan")), dict(c=-0.1), dict(Fc=float("inf"))):
            kw = {**dict(M=1.0, c=0.1, Fc=1.0, Fs=2.0), **bad}
            with self.assertRaises(ModelConfigError):
                sim_params(T, "prismatic", **kw)
        p = sim_params(T, "prismatic", M=1.0, c=0.1, Fc=1.0, Fs=2.0)
        r = simulate_general(p, eff([(3.0, 1.999999)]), np.linspace(0, 3, 600))
        self.assertTrue(np.all(r.qdot == 0.0) and np.all(np.isfinite(r.q)))   # no chatter at the threshold
        z = simulate_general(p, eff([(1.0, 0.0)]), [0.5, 1.0])
        self.assertTrue(np.all(z.q == 0.0))                               # sign(0) never drives motion

    def test_a10_gravity_registration_gate(self):
        obj = make_object("revolute", gravity_verified=False)
        with self.assertRaises(GravityRegistrationError):
            ModelSpec(GeneralTerms(), "revolute", "geometry_coupled", obj.mass_model, obj.gravity)
        with self.assertRaises(GravityRegistrationError):
            ModelSpec(GeneralTerms(), "revolute", "geometry_coupled", obj.mass_model, None)
        ok = make_object("revolute")
        ModelSpec(GeneralTerms(), "revolute", "geometry_coupled", ok.mass_model, ok.gravity)
        ModelSpec(GeneralTerms(gravity=False), "revolute", "geometry_coupled", obj.mass_model, None)

    def test_a11_paired_axis_sign_invariance(self):
        from forceartgs_physics.forces.projection import generalized_effort
        from forceartgs_physics.schemas import ForceRecord, JointKinematics
        obj = make_object("revolute", terms=GeneralTerms(gravity=False))
        t = np.linspace(0, 2, 200)
        P = np.tile(obj.contact_point(0.0), (200, 1))
        F = np.tile([0.0, 0.0, 1.0], (200, 1)) * (t[:, None] < 1.0)
        out = []
        for s in (1.0, -1.0):
            j = JointKinematics("revolute", s * obj.axis, obj.origin, "W")
            Q = generalized_effort(j, ForceRecord(t_s=t, force_N=F, point_m=P, frame="W"))
            p = sim_params(GeneralTerms(gravity=False), "revolute", M=0.15, c=0.15, Fc=0.05, Fs=0.08)
            out.append(s * simulate_general(p, Q, t).q)               # q measured along s*axis -> physical angle
        np.testing.assert_allclose(out[0], out[1], atol=1e-14)

    def test_a12_irregular_sampling_and_zoh(self):
        rng = np.random.default_rng(3)
        tf = np.sort(np.r_[0.0, rng.uniform(0, 2, 60)])
        lev = rng.normal(0, 2, len(tf))
        e = ZOH(tf, lev)
        dense_t = np.arange(0, 2.0, 2e-5)
        dense = ZOH(dense_t, e.at(dense_t))                           # resampled high-frequency reference
        tq = np.sort(rng.uniform(0, 1.999, 37))
        p = sim_params(GeneralTerms(static=False, gravity=False), "prismatic", M=1.0, c=0.5, Fc=0.3)
        a = simulate_general(p, e, tq, max_step=5e-4).q
        b = simulate_general(p, dense, tq, max_step=5e-4).q
        np.testing.assert_allclose(a, b, atol=1e-3 * max(np.ptp(a), 1e-6))

    def test_a13_gravity_off_model_vs_gravity_on_engine(self):
        off = GeneralTerms(gravity=False)
        tm = term_matrix(off, 0.2, 0.4, gravity_bound=5.0, effort_scale=2.0)
        self.assertEqual(tm["mujoco"]["gravity"]["status"], "GRAVITY_CONFIGURATION_MISMATCH")
        self.assertFalse(tm["mujoco"]["full_model_representable"])
        vert = RigidBodyGravity("revolute", (0, 0, 1), (0, 0, 0), (0.2, 0.3, 0.4), 3.0, (0, 0, -9.81))
        tm2 = term_matrix(off, 0.2, 0.4, gravity_bound=vert.amplitude_bound(), effort_scale=2.0)
        self.assertEqual(tm2["mujoco"]["gravity"]["status"], "NEGLIGIBLE")       # tested bound: vertical hinge


if __name__ == "__main__":
    unittest.main()
