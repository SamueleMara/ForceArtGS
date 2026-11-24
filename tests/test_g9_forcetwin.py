"""ForceTwin-adaptation: record contract, constrained NNLS / weak identification, signed impulse audit, kinematic
hypotheses, structured residual, EKF / PINN on the minimal model (spec sections 8.3 and 10). Synthetic truth from
forceartgs_g9.synthetic / simulate_general: a PASS is a software check, not real-world validity."""
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
from forceartgs_g9.dynamics.general_1dof import GeneralTerms, rotation, sim_params, simulate_general  # noqa: E402
from forceartgs_g9.estimators import forcetwin_linear as FL  # noqa: E402
from forceartgs_g9.estimators.model_spec import G9Interaction  # noqa: E402
from forceartgs_g9.forcetwin import record as RC  # noqa: E402
from forceartgs_g9.synthetic import ZOH, episode, make_object, profile  # noqa: E402
from forceartgs_g9.validation import impulse_balance as IB  # noqa: E402

physics_lib.load()
torch.set_num_threads(4)
NT = GeneralTerms(static=False, gravity=False)


def door():
    return make_object("revolute", hinge="vertical", terms=NT)


def run_episode(obj, segs, *, rate=50.0, cam_hz=25.0, hidden=None, seed=0, qn=2e-4, terms=None, G=None):
    """Simulate with the TRUE effort (measured effort + optional hidden torque); record only the measured one."""
    tf, Q = profile(segs, rate_hz=rate)
    Qtrue = Q + (hidden(tf) if hidden is not None else 0.0)
    t = np.arange(0.0, tf[-1] + 0.01, 1.0 / cam_hz)
    sp = obj.sim() if G is None else sim_params(terms, obj.joint_type, M=obj.M_true, c=obj.c, Fc=obj.Fc, G=G)
    roll = simulate_general(sp, ZOH(tf, Qtrue), t, max_step=2.5e-4)
    q = roll.q + qn * np.random.default_rng(seed).standard_normal(len(t))
    it = G9Interaction(f"ep{seed}", obj.joint_type, t, q, qn, ZOH(tf, Q))
    return it, dict(q=roll.q, qdot=roll.qdot)


def rec_of(obj, eps):
    return RC.from_synthetic(obj, [e[0] for e in eps], [e[1] for e in eps])


PUSHES = [[(0.3, 0.0), (0.5, 2.5), (0.4, -0.3), (0.6, -2.2), (0.5, 0.4), (0.4, 1.8), (0.8, 0.0)],
          [(0.2, 0.0), (0.4, -2.8), (0.5, 0.6), (0.5, 2.0), (0.3, -1.5), (0.6, 0.0)]]


class RecordContract(unittest.TestCase):
    def test_sign_once_reversal_and_invalid_metadata(self):
        obj = door()
        rec = rec_of(obj, [run_episode(obj, PUSHES[0])])
        r2 = rec.reversed_coordinate()
        np.testing.assert_allclose(r2.Q_object, -rec.Q_object)
        a = FL.fit(rec, B=30)
        b = FL.fit(r2, B=30)
        for n in ("I_eff", "mu", "b"):
            self.assertAlmostEqual(a["estimates"][n]["estimate"], b["estimates"][n]["estimate"], places=9)
        qd = np.gradient(rec.q, rec.q_t_s)
        qd2 = np.gradient(r2.q, r2.q_t_s)
        Qa = np.interp(rec.q_t_s, rec.force_t_s, rec.Q_object)
        Qb = np.interp(r2.q_t_s, r2.force_t_s, r2.Q_object)
        np.testing.assert_allclose(Qa * qd, Qb * qd2)                        # work tau qdot is invariant
        np.testing.assert_allclose(rec.effort_for(rec.axis_B, rec.origin_B), rec.Q_object, atol=1e-9)
        rec.provenance["object_side_multiplier"] = None
        with self.assertRaises(RC.RecordError):
            rec.validate()
        rec.provenance["object_side_multiplier"] = -1.0
        rec.q_sigma = rec.q_sigma * 0
        with self.assertRaises(RC.RecordError):
            rec.validate()

    def test_native_zoh_pulse_preserved(self):
        t = np.arange(0, 1, 0.02)
        Q = np.zeros(len(t))
        Q[25:29] = [0.4, 0.1, -0.1, -0.1]
        J, sJ = IB._zoh_J(t, Q, np.full(len(t), 0.01), 0.5, 0.58)
        self.assertAlmostEqual(J, 0.02 * 0.3, places=12)
        self.assertAlmostEqual(sJ, 0.01 * 0.02 * 2, places=12)


class Identification(unittest.TestCase):
    def test_01_vertical_panel_recovery_and_coast_ratio_only(self):
        obj = door()
        rec = rec_of(obj, [run_episode(obj, PUSHES[k], seed=k) for k in range(2)])
        for form in ("direct", "weak"):
            r = FL.fit(rec, form=form, L=0.2, B=60)
            e = r["estimates"]
            self.assertAlmostEqual(e["I_eff"]["estimate"] / obj.M_true, 1, delta=0.12, msg=form)
            self.assertAlmostEqual(e["mu"]["estimate"] / obj.Fc, 1, delta=0.25, msg=form)
        idf = FL.identifiability(FL.fit(rec, B=60), "revolute")
        self.assertEqual(idf["I_eff"]["identifiability_status"], "ESTIMATED_CONDITIONAL")
        coast = rec_of(obj, [run_episode(obj, [(0.2, 0.0), (0.4, 3.0), (2.0, 0.0)], seed=5)])
        cr = FL.coast_ratios(coast)
        self.assertEqual(cr["status"], "RATIO_ONLY")
        self.assertAlmostEqual(cr["b_over_I"] + cr["mu_over_I"] / 1.0, (obj.c + obj.Fc) / obj.M_true,
                               delta=0.5 * (obj.c + obj.Fc) / obj.M_true)
        # coast only (push removed): the absolute scale is NOT identified -- never a fabricated value
        only = rec_of(obj, [run_episode(obj, [(0.4, 3.0), (2.0, 0.0)], seed=6)])
        only.windows = [(n, slice(fs.start + 20, fs.stop), qs, s) for n, fs, qs, s in only.windows]
        only.Q_object[:] = np.where(np.arange(len(only.Q_object)) < 20, only.Q_object, 0.0)
        r0 = FL.fit(only, B=30)
        if r0["status"] == "OK":
            self.assertNotEqual(FL.identifiability(r0, "revolute")["I_eff"]["identifiability_status"],
                                "ESTIMATED_CONDITIONAL")

    def test_02_oblique_axis_known_gravity(self):
        obj = make_object("revolute", terms=GeneralTerms(static=False))            # horizontal hinge, gravity on
        eps = [episode(obj, variant=v, seed=v) for v in range(2)]
        rec = rec_of(obj, eps)
        gh = FL.GravityHypothesis.from_mass_model(obj.mass_model, obj.gravity.g_world, "truth_geometry")
        r = FL.fit(rec, grav=gh, B=40)
        self.assertAlmostEqual(r["estimates"]["I_eff"]["estimate"] / obj.M_true, 1, delta=0.12)
        q = np.linspace(-0.5, 0.5, 7)
        np.testing.assert_allclose(obj.M_true * gh.k(q), obj.density * obj.mass_model.gravity_unit(
            obj.gravity.g_world).G(q), atol=1e-10)                                 # one density for both

    def test_03_prismatic_horizontal_and_inclined(self):
        # a mild 2.3 deg incline: pushed both ways (a steep slider without stops slides one way only, where the
        # constant gravity load is collinear with sgn(qdot) -- genuinely unidentifiable)
        for axis, grav in (([0.0, 1.0, 0.0], False), ([0.0, 0.9992, 0.04], True)):
            obj = make_object("prismatic", axis=axis, terms=GeneralTerms(static=False, gravity=grav))
            rec = rec_of(obj, [episode(obj, variant=v, seed=v) for v in range(2)])
            gh = FL.GravityHypothesis.from_mass_model(obj.mass_model, obj.gravity.g_world, "g") if grav else None
            r = FL.fit(rec, grav=gh, B=40)
            self.assertAlmostEqual(r["estimates"]["I_eff"]["estimate"] / obj.M_true, 1, delta=0.12, msg=str(axis))
            self.assertEqual(FL.units("prismatic")["I_eff"], "kg")

    def test_04_no_excitation_rank_failure(self):
        obj = door()
        v, T = 0.3, 3.0
        t = np.arange(0, T, 0.04)
        q = v * t + 2e-4 * np.random.default_rng(0).standard_normal(len(t))
        tf = np.arange(0, T, 0.02)
        it = G9Interaction("steady", "revolute", t, q, 2e-4, ZOH(tf, np.full(len(tf), obj.c * v + obj.Fc)))
        rec = rec_of(obj, [(it, dict(q=v * t))])
        r = FL.fit(rec, B=30)
        self.assertLess(r["rank"], 3)
        idf = FL.identifiability(r, "revolute")
        self.assertEqual(idf["I_eff"]["identifiability_status"], "UNOBSERVABLE")

    def test_05_short_80ms_impulse_native_rates(self):
        obj = door()
        it, tr = run_episode(obj, [(0.5, 0.0), (0.08, 3.0), (1.5, 0.0)], rate=50.0, cam_hz=25.0)
        rec = rec_of(obj, [(it, tr)])
        self.assertEqual(len(rec.q_t_s), len(it.t_obs))                         # q stays at 25 Hz
        self.assertAlmostEqual(np.median(np.diff(rec.force_t_s)), 0.02)          # force stays at 50 Hz
        aud = IB.audit(rec, (25, 28))
        self.assertTrue(aud["cases"]["accepted"]["positive_inertia_feasible"], aud["cases"]["accepted"])
        self.assertEqual(aud["decision"]["primary"], "IMPULSE_CONSISTENT")
        self.assertGreater(aud["cases"]["accepted"]["J_over_dv"], 0)

    def test_06_wrong_sign_caught_not_repaired(self):
        obj = door()
        it, tr = run_episode(obj, [(0.5, 0.0), (0.08, 3.0), (1.5, 0.0)])
        it.effort = ZOH(it.effort.t_s, -np.asarray(it.effort.levels))           # broken sign
        rec = rec_of(obj, [(it, tr)])
        aud = IB.audit(rec, (25, 28))
        self.assertIn("NO_POSITIVE_INERTIA_EXPLANATION", aud["decision"]["states"])
        self.assertTrue(aud["cases"]["COUNTERFACTUAL_force_sign_flipped"]["positive_inertia_feasible"])
        bad = rec_of(obj, [(G9Interaction("b", "revolute", e[0].t_obs, e[0].q_obs, e[0].q_std,
                                          ZOH(e[0].effort.t_s, -np.asarray(e[0].effort.levels))), e[1])
                           for e in [run_episode(obj, PUSHES[0])]])
        r = FL.fit(bad, B=30)
        self.assertGreaterEqual(r["estimates"]["I_eff"]["estimate"], 0.0)       # never a negative inertia
        self.assertLess(r["estimates"]["I_eff"]["unconstrained_ols"], 0.0)       # the contradiction is visible
        self.assertNotEqual(FL.identifiability(r, "revolute")["I_eff"]["identifiability_status"],
                            "ESTIMATED_CONDITIONAL")

    def test_07_motion_without_measured_effort(self):
        obj = door()
        hidden = lambda t: np.where((t >= 0.62) & (t < 0.9), -1.5, 0.0)         # unmeasured latch / seal torque
        it, tr = run_episode(obj, [(0.5, 0.0), (0.08, 0.6), (1.4, 0.0)], hidden=hidden)
        rec = rec_of(obj, [(it, tr)])
        aud = IB.audit(rec, (25, 28))
        self.assertIn("UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE", aud["decision"]["states"])
        self.assertFalse(aud["onset_bracket"]["measured_effort_explains_motion"])

    def test_08_effective_mode_independent_of_hull(self):
        obj = door()
        rec = rec_of(obj, [run_episode(obj, PUSHES[0])])
        r0 = FL.fit(rec, B=30)
        from forceartgs_g9.dynamics.geometry_inertia import PASSIVE, GeometryMassModel
        from forceartgs_g9.synthetic import box_mesh
        huge = GeometryMassModel.from_validated_mesh(box_mesh([-1, 0, 0], [1, 3, 2]), "uniform_solid", "revolute",
                                                     [0, 0, 1], obj.origin, source_provenance=dict(source="wrong"),
                                                     reflected_actuator_inertia=PASSIVE)
        gh = FL.GravityHypothesis.from_mass_model(huge, obj.gravity.g_world, "vertical: G = 0 for any hull")
        r1 = FL.fit(rec, grav=gh, B=30)
        self.assertAlmostEqual(r0["estimates"]["I_eff"]["estimate"], r1["estimates"]["I_eff"]["estimate"], places=9)
        from forceartgs_g9.forcetwin.pipeline import simulation_handoff
        import types
        st = dict(variants=dict(inertia_viscous_coulomb=dict(identifiability=dict(
            I_eff=dict(estimate=1e-3, unit="kg m^2", identifiability_status="NOT_IDENTIFIED")))))
        inp = types.SimpleNamespace(joint_type="revolute", kinematic_status="IDENTIFIED", mass_mesh=huge.mesh
                                    if hasattr(huge, "mesh") else box_mesh([-1, 0, 0], [1, 3, 2]))
        h = simulation_handoff(inp, st, {}, huge, None, None, "x")
        self.assertEqual(h["mass_allocation"]["status"], "NOT_ALLOCATED")
        self.assertEqual(h["mass_allocation"]["prior"]["status"], "PRIOR_FOR_SIM")

    def test_09_ekf_and_pinn_on_the_minimal_model(self):
        from forceartgs_g9.estimators.general_ekf import general_akf_fit
        from forceartgs_g9.estimators.torch_model import DT, TorchDynamics
        obj = door()
        rec = rec_of(obj, [run_episode(obj, PUSHES[k], seed=k, qn=1e-3) for k in range(2)])
        spec = FL.spec_for("revolute", "inertia_viscous_coulomb")
        self.assertEqual(spec.names, ["M", "c", "Fc"])                           # I_eff, b, mu (no density)
        th0 = np.array([obj.M_true * 0.7, obj.c * 1.5, obj.Fc * 0.7])
        r = general_akf_fit(rec.interactions(), spec, th0, s_Q=0.02)
        self.assertAlmostEqual(r["estimates"]["M"]["value"] / obj.M_true, 1, delta=0.15)
        self.assertGreaterEqual(r["min_state_cov_eigenvalue"], -1e-12)
        self.assertTrue(0.3 < r["nis"]["mean"] < 2.5, r["nis"])
        dyn = TorchDynamics(spec)
        eta = {k: v.detach().clone().requires_grad_(True) for k, v in dyn.eta0([0.4, 0.15, 0.25]).items()}
        t = torch.linspace(0, 1, 20, dtype=DT)
        q, qd, qdd = 0.2 * torch.sin(2 * t), 0.4 * torch.cos(2 * t), -0.8 * torch.sin(2 * t)
        res = dyn.residual(dyn.values(eta), q, qd, qdd, torch.zeros_like(t))
        for name, val, reg in (("M", 0.4, qdd), ("c", 0.15, qd), ("Fc", 0.25, torch.sign(qd))):
            g = torch.autograd.grad(res.sum(), eta[name], retain_graph=True)[0]
            self.assertAlmostEqual(float(g) / val, float(reg.sum()), places=9)    # d r / d theta, log param

    def test_10_residual_door_closer(self):
        from forceartgs_g9.estimators.forcetwin_residual import ablations
        obj = door()
        k = 1.2
        tm = GeneralTerms(static=False, gravity=True)
        eps = [run_episode(obj, PUSHES[j], seed=j, terms=tm, G=lambda q: k * q) for j in range(2)]
        rec = rec_of(obj, eps)
        r = FL.fit(rec, B=30)
        beta = [r["estimates"][n]["estimate"] for n in ("I_eff", "mu", "b")]
        ab = ablations(rec, beta, epochs=500, patience=80)
        self.assertEqual(ab["validation_label"], "EXPLORATORY_TRAIN_ONLY")
        self.assertIn("NOT_EXPORTED", ab["export"])
        g = ab["ablations"]["g"]
        self.assertGreater(g["val_improvement_vs_none"], 0.05)                  # a configuration load is learnt
        d = ab["ablations"]["damping"]
        self.assertGreaterEqual(d["dissipation"]["min_c"], 0.0)                  # never energy injection
        self.assertLessEqual(d["dissipation"]["dissipated_power_max"], 0.0)


class GateRules(unittest.TestCase):
    """Identifiability gate for the general estimators (regression of the PINN mislabels in the 2026-11-21 batch)."""

    def test_clean_estimate_passes(self):
        st, why = FL.gate("I_eff", 0.4, 0.02, uncertainty_label="STATISTICAL", I_prior=(0.1, 1.0), replay_rmse=1.3)
        self.assertEqual(st, "ESTIMATED_CONDITIONAL", why)

    def test_seed_ensemble_spread_is_not_an_uncertainty(self):
        st, why = FL.gate("mu", 0.0055, 2.5e-4, uncertainty_label="ENSEMBLE (seeds; not a calibrated posterior)",
                          replay_rmse=1.0)
        self.assertEqual(st, "NOT_IDENTIFIED")
        self.assertIn("seed-ensemble", why[0])

    def test_failed_replay_is_model_inadequate(self):
        for rr in (45000.0, 11.0, None, float("nan")):
            st, why = FL.gate("mu", 0.9, 0.01, uncertainty_label="EKF_POSTERIOR", replay_rmse=rr)
            self.assertEqual(st, "MODEL_INADEQUATE", (rr, why))
        st, _ = FL.gate("mu", 0.9, 0.01, uncertainty_label="EKF_POSTERIOR", replay_rmse=9.5)
        self.assertEqual(st, "ESTIMATED_CONDITIONAL")

    def test_suitcase_pinn_case_and_other_gates(self):
        # the mislabelled suitcase PINN: I = 8.31 kg, ensemble spread, replay 159 sigma, unaccepted kinematics
        st, why = FL.gate("I_eff", 8.31, 0.006, uncertainty_label="ENSEMBLE (seeds)", I_prior=(2, 20),
                          replay_rmse=159.0, kin_ok=False, kin_status="UNOBSERVABLE")
        self.assertEqual(st, "NOT_IDENTIFIED")
        self.assertTrue(any("replay" in w for w in why) and any("KINEMATICS" in w for w in why))
        self.assertEqual(FL.gate("I_eff", 0.4, 0.02, estimator_status="not_converged", replay_rmse=1)[0],
                         "NOT_IDENTIFIED")
        self.assertEqual(FL.gate("I_eff", 0.4, 0.02, impulse_state="UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE",
                                 replay_rmse=1)[0], "DATA_INCONSISTENT")
        self.assertEqual(FL.gate("I_eff", 0.4, 0.02, nis=dict(consistent=False, mean=4.7, mean_upper95=1.2),
                                 replay_rmse=1)[0], "MODEL_INADEQUATE")
        self.assertEqual(FL.gate("I_eff", 0.4, 0.02, replay_rmse=1, kin_ok=False, kin_status="X")[0],
                         "CONDITIONAL_ON_UNACCEPTED_KINEMATICS")
        self.assertEqual(FL.gate("I_eff", 0.4, 0.0, replay_rmse=1)[0], "NOT_IDENTIFIED")     # degenerate covariance

    def test_nnls_identifiability_replay_gate(self):
        obj = door()
        rec = rec_of(obj, [run_episode(obj, PUSHES[0])])
        r = FL.fit(rec, B=30)
        bad = FL.identifiability(r, "revolute", replay_res=dict(mean_rmse_over_sigma=500.0))
        self.assertNotEqual(bad["I_eff"]["identifiability_status"], "ESTIMATED_CONDITIONAL")


class KinematicHypotheses(unittest.TestCase):
    def _tracks(self, td, tilt_deg, T=20, N=80):
        from forceartgs_g9.forcetwin.kinematics_hypotheses import hypotheses
        rng = np.random.default_rng(1)
        b = math.radians(tilt_deg)
        a = np.array([math.sin(b), 0.0, math.cos(b)])
        o = np.array([0.1, 0.2, 0.0])
        ref = o + np.c_[rng.uniform(0.05, 0.5, N), rng.uniform(-0.02, 0.02, N), rng.uniform(0.1, 0.8, N)]
        qs = np.linspace(0, -0.6, T)
        rows = []
        for tt in range(T):
            X = o + (ref - o) @ rotation(a, qs[tt]).T + 1e-3 * rng.standard_normal((N, 3))
            for k in range(N):
                rows.append((k, tt, *X[k]))
        rows = np.array(rows)
        n = len(rows)
        d = Path(td)
        (d / "part_support").mkdir(parents=True, exist_ok=True)
        np.savez(d / "tracks_observations.npz", track=rows[:, 0].astype(int), time=rows[:, 1].astype(int),
                 x_w=rows[:, 2:5], depth_valid=np.ones(n, bool), uv_valid=np.ones(n, bool),
                 depth_edge=np.zeros(n, bool), mask_valid=np.ones(n, bool), fm_rgbd=np.arange(T) * 2,
                 fm_source=np.arange(T) * 4)
        np.savez(d / "part_support" / "track_support.npz", p_moving_R=np.ones(N), eligible=np.ones(N, bool))
        return hypotheses(d, a, o), a

    def test_tilted_vs_vertical_from_tracks(self):
        with tempfile.TemporaryDirectory() as td:
            r, a = self._tracks(td, 15.0)
            fr = np.asarray(r["hypotheses"]["free_axis_refit"]["axis_B"])
            self.assertLess(math.degrees(math.acos(min(1, abs(fr @ a)))), 1.0)
            self.assertEqual(r["comparison"]["verdict"], "VERTICAL_WORSE_GEOMETRICALLY")
        with tempfile.TemporaryDirectory() as td:
            r, _ = self._tracks(td, 0.3)
            self.assertEqual(r["comparison"]["verdict"], "VERTICAL_NOT_DISTINGUISHABLE")


if __name__ == "__main__":
    unittest.main()
