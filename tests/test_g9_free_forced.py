"""Generalized free + forced normalized dynamics (FORCEARTGS_G9_GENERALIZED_FREE_AND_FORCED_DYNAMICS spec section 9).

Synthetic truth from the validated hybrid integrator (simulate_general). Tolerances are DECLARED here, before any
result was seen (section 9.1), and are not relaxed after failures:
  noiseless equivalence 1e-7; weak-form rate recovery 10 %; rollout 5 %; EKF 15 %; PINN 20 %;
  forced-scale M 10 %, c 15 %, Fc 20 %.
A PASS is a software check of the estimators against their own model, not real-world validity."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9 import physics_lib  # noqa: E402
from forceartgs_g9.dynamics.general_1dof import GeneralTerms, sim_params, simulate_general  # noqa: E402
from forceartgs_g9.dynamics.geometry_inertia import PASSIVE, GeometryMassModel  # noqa: E402
from forceartgs_g9.estimators import free_weak_form as FW  # noqa: E402
from forceartgs_g9.estimators import normalized_ekf as NE  # noqa: E402
from forceartgs_g9.estimators import normalized_pinn as NP  # noqa: E402
from forceartgs_g9.estimators import normalized_rollout as NR  # noqa: E402
from forceartgs_g9.free_forced import replay as RP  # noqa: E402
from forceartgs_g9.free_forced import segments as SG  # noqa: E402
from forceartgs_g9.free_forced import status as ST  # noqa: E402
from forceartgs_g9.free_forced.normalized import (GravityModel, NormalizedDynamicsSpec,  # noqa: E402
                                                  NormalizedModelError)
from forceartgs_g9.forcetwin import record as RC  # noqa: E402
from forceartgs_g9.synthetic import ZOH, box_mesh, profile  # noqa: E402

physics_lib.load()
torch.set_num_threads(4)
TOL = dict(equiv=1e-7, weak=0.10, rollout=0.05, ekf=0.15, pinn=0.20, M=0.10, c=0.15, Fc=0.20)
G = (0.0, 0.0, -9.81)


def simulate(M, c, Fc, *, jt="revolute", v0=1.0, T=3.0, cam=25.0, qn=1e-4, Gfun=None, Qprof=None, seed=0, q0=0.0):
    """Dimensional truth -> (t, q_obs, q_true, v_true, force_t, Q)."""
    terms = GeneralTerms(static=False, gravity=Gfun is not None, bias=False)
    sp = sim_params(terms, jt, M=M, c=c, Fc=Fc, G=Gfun)
    if Qprof is None:
        tf = np.arange(0.0, T + 0.02, 0.02)
        Q = np.zeros(len(tf))
    else:
        tf, Q = profile(Qprof, rate_hz=50.0)
    t = np.arange(0.0, T, 1.0 / cam)
    r = simulate_general(sp, ZOH(tf, Q), t, t0=0.0, q0=q0, v0=v0, max_step=2.5e-4)
    qo = r.q + qn * np.random.default_rng(seed).standard_normal(len(t))
    return t, qo, r.q, r.qdot, tf, Q


def seg_from(t, q, qn, sign, *, label="FREE_CONFIRMED", tf=None, Q=None, name="s", jt="revolute", Qs=0.01):
    return SG.MotionSegment(name, label, jt, np.asarray(t), np.asarray(q), np.full(len(t), qn), float(sign),
                            None if tf is None else np.asarray(tf), None if Q is None else np.asarray(Q),
                            None if Q is None else np.full(len(Q), Qs))


def free_pair(M=0.1, c=0.04, Fc=0.01, jt="revolute", qn=1e-4, Gfun=None):
    out = []
    for k, v0 in enumerate((1.0, -0.8)):
        t, qo, *_ = simulate(M, c, Fc, jt=jt, v0=v0, T=3.0 if v0 > 0 else 2.4, qn=qn, Gfun=Gfun, seed=k)
        out.append(seg_from(t, qo, qn, np.sign(v0), name=f"free{k}", jt=jt))
    return out


def spec(jt="revolute", g=None):
    return NormalizedDynamicsSpec(jt, g or GravityModel.zero(jt))


def rel(a, b):
    return abs(a - b) / abs(b)


class StructuralScale(unittest.TestCase):
    def test_1_scale_equivalence_and_shared_ratios(self):
        a = simulate(0.1, 0.04, 0.01, qn=0.0)
        b = simulate(1.0, 0.4, 0.1, qn=0.0)
        self.assertLess(np.max(np.abs(a[2] - b[2])), TOL["equiv"])            # identical free trajectories
        segs = free_pair()
        ns = spec()
        w = FW.fit(segs, ns, B=50, sensitivity=False)
        self.assertEqual(w["names"], ["lam", "gam"])
        self.assertEqual(w["absolute_scale"], "ABSOLUTE_SCALE_NOT_OBSERVABLE")
        self.assertLess(rel(w["estimate"]["lam"], 0.4), TOL["weak"])
        self.assertLess(rel(w["estimate"]["gam"], 0.1), TOL["weak"])
        r = NR.fit(segs, ns)
        self.assertLess(rel(r["estimate"]["lam"], 0.4), TOL["rollout"])
        self.assertLess(rel(r["estimate"]["gam"], 0.1), TOL["rollout"])
        e = NE.fit(segs, ns)
        self.assertLess(rel(e["estimate"]["lam"], 0.4), TOL["ekf"])
        self.assertLess(rel(e["estimate"]["gam"], 0.1), TOL["ekf"])
        for res in (w, r, e):
            self.assertFalse({"M", "I", "beta", "I_eff"} & set(res["estimate"]))

    def test_2_no_zero_inertia_in_free_mode(self):
        ns = spec()
        self.assertNotIn("beta", ns.names)
        self.assertEqual(ns.free().names, ["lam", "gam"])
        with self.assertRaises(NormalizedModelError):
            ns.physical(ns.pack(0.4, 0.1), 0.0)                              # M = 0 never reconstructed
        # legacy handoff: a non-identified I_eff is never exposed as a simulator value
        from forceartgs_g9.forcetwin.pipeline import simulation_handoff

        class _Inp:
            joint_type = "revolute"
            kinematic_status = "IDENTIFIED"
            mass_mesh = None
        st = dict(variants=dict(inertia_viscous_coulomb=dict(identifiability={
            "I_eff": dict(estimate=0.0, unit="kg m^2", identifiability_status="NOT_IDENTIFIED"),
            "mu": dict(estimate=0.01, unit="N m", identifiability_status="NOT_IDENTIFIED"),
            "b": dict(estimate=0.0, unit="N m s/rad", identifiability_status="NOT_IDENTIFIED")})))
        h = simulation_handoff(_Inp(), st, {}, None, None, None, "p")
        for term in h["joint_terms"].values():
            self.assertNotIn("value", term)
            self.assertIsNone(term["estimate"])


class Gravity(unittest.TestCase):
    def test_3_prismatic_free_damping_units(self):
        segs = free_pair(M=2.0, c=1.6, Fc=0.1, jt="prismatic", qn=5e-5)
        ns = spec("prismatic")
        w = FW.fit(segs, ns, B=30, sensitivity=False)
        self.assertEqual(w["units"], {"lam": "1/s", "gam": "m/s^2"})
        self.assertLess(rel(w["estimate"]["lam"], 0.8), TOL["weak"])
        self.assertLess(rel(w["estimate"]["gam"], 0.05), TOL["weak"])
        self.assertNotIn("M", w["estimate"])

    def test_4_inclined_prismatic_gravity(self):
        a = np.array([0.6, 0.0, -0.8])
        g = GravityModel.prismatic_ideal(a, G)
        self.assertAlmostEqual(g(0.3), float(-np.asarray(G) @ a), places=12)
        m = 3.0                                                     # energy: U = -m g.(x0 + q a) -> dU/dq = m a_g
        U = lambda q: -m * np.asarray(G) @ (np.array([0.1, 0.2, 0.3]) + q * a)
        self.assertAlmostEqual((U(1e-4) - U(-1e-4)) / 2e-4, m * g(0.0), places=6)
        ns = NormalizedDynamicsSpec("prismatic", g)
        r = ns.simulate(ns.pack(0.5, 1.0), np.array([0.5, 1.0]), 0.0, 0.0, 0.0)  # |a_g| = 7.85 > gam: slides
        # +q is DOWNHILL (a_z < 0): the released slider accelerates towards +q although Q = 0
        self.assertLess(g(0.0), 0.0)
        self.assertGreater(r.q[-1], 0.1)
        self.assertGreater(float(np.asarray(G) @ a), 0.0)

    def _mm(self, axis, origin=(0.0, 0.0, 0.5), refl=PASSIVE, q_ref=0.0):
        mesh = box_mesh([-0.01, 0.0, 0.2], [0.01, 0.5, 0.9])
        return GeometryMassModel.from_validated_mesh(mesh, "uniform_solid", "revolute", np.asarray(axis, float),
                                                     np.asarray(origin, float),
                                                     source_provenance=dict(source="synthetic"),
                                                     reflected_actuator_inertia=refl, q_ref=q_ref)

    def test_5_revolute_vertical_and_tilted(self):
        g = GravityModel.from_mass_model(self._mm([0, 0, 1]), G)
        self.assertLess(g.amplitude(), 1e-12)                       # exactly vertical -> identically zero
        tilt = np.array([math.sin(0.1), 0.0, math.cos(0.1)])
        for q_ref in (0.0, 0.7):
            mm = self._mm(tilt, q_ref=q_ref)
            gm = GravityModel.from_mass_model(mm, G)
            gu = mm.gravity_unit(G)
            for q in (-0.5, 0.2, 1.1):
                fd = (gu.potential(q + 1e-6) - gu.potential(q - 1e-6)) / 2e-6
                self.assertAlmostEqual(gm(q), fd / mm.M1, places=6)
                self.assertAlmostEqual(gm.deriv(q), (gm(q + 1e-6) - gm(q - 1e-6)) / 2e-6, places=5)
            self.assertGreater(gm.amplitude(), 0.1)

    def test_6_reflected_inertia_breaks_density_cancellation(self):
        tilt = np.array([math.sin(0.1), 0.0, math.cos(0.1)])
        mm = self._mm(tilt, refl=0.05)
        with self.assertRaises(NormalizedModelError):
            GravityModel.from_mass_model(mm, G)
        a1 = GravityModel.from_mass_model(mm, G, rho=100.0).amplitude()
        a2 = GravityModel.from_mass_model(mm, G, rho=1000.0).amplitude()
        self.assertGreater(abs(a1 - a2) / a2, 0.01)
        self.assertTrue(GravityModel.from_mass_model(mm, G, rho=100.0).depends_on_scale)


def forced_set(M=0.5, c=0.2, Fc=0.05, qn=1e-4, flip_measured=False):
    free = free_pair(M, c, Fc, qn=qn)
    forced = []
    for k, prof in enumerate(([(0.8, 0.6), (0.8, 0.1), (0.8, 0.9)], [(0.6, 1.2), (0.9, 0.2), (0.9, 0.7)])):
        t, qo, qt, vt, tf, Q = simulate(M, c, Fc, v0=0.3, T=sum(p[0] for p in prof) - 0.02, Qprof=prof, qn=qn,
                                        seed=10 + k)
        assert np.all(vt > 0)
        forced.append(seg_from(t, qo, qn, 1.0, label="FORCED_MEASURED", tf=tf, Q=-Q if flip_measured else Q,
                               name=f"forced{k}"))
    return free, forced


class ForcedScale(unittest.TestCase):
    def test_7_forced_scale_recoverability(self):
        free, forced = forced_set()
        ns = spec()
        wf = FW.fit(free, ns, B=30, sensitivity=False)
        fs = FW.forced_scale(free, forced, ns, free_fit=wf)
        self.assertEqual(fs["status"], "FORCE_IDENTIFIED_CANDIDATE")
        d = fs["dimensional"]
        self.assertLess(rel(d["M"]["value"], 0.5), TOL["M"])
        self.assertLess(rel(d["c"]["value"], 0.2), TOL["c"])
        self.assertLess(rel(d["Fc"]["value"], 0.05), TOL["Fc"])
        self.assertGreater(d["M"]["sd"], 0.0)

    def test_8_pulse_sign_contradiction(self):
        free, forced = forced_set(flip_measured=True)
        ns = spec()
        wf = FW.fit(free, ns, B=0, sensitivity=False)
        fs = FW.forced_scale(free, forced, ns, free_fit=wf)
        self.assertEqual(fs["status"], "SCALE_CONTRADICTED")
        self.assertNotIn("dimensional", fs)                         # no negative / zero M is produced
        self.assertGreater(fs["n_contradicting_windows"], 0)


def toy_record(Q, in_contact, modes, q=None, valid=True):
    n = len(Q)
    tf = np.arange(n) * 0.02
    tq = tf[::2]
    qq = np.linspace(0, 0.2, len(tq)) if q is None else q
    r = RC.JointInteractionRecord(
        scene="toy", joint_hypothesis_id="toy", joint_type="revolute", axis_B=np.array([0, 0, 1.0]),
        origin_B=np.zeros(3), q_reference={}, force_t_s=tf, source_force_row=np.arange(n),
        Q_object=np.asarray(Q, float), Q_sigma=np.full(n, 0.02), force_object_B=np.zeros((n, 3)),
        contact_point_B=np.tile([0.3, 0.0, 0.5], (n, 1)), contact_mode=np.asarray(modes),
        in_contact=np.asarray(in_contact, bool), q_t_s=tq, source_q_frame=np.arange(0, n, 2), q=qq,
        q_sigma=np.full(len(tq), 1e-4), windows=[], provenance=dict(object_side_multiplier=-1.0))
    return r


class SegmentEvidence(unittest.TestCase):
    def test_9_unknown_force_is_not_zero_force(self):
        n = 120
        rng = np.random.default_rng(0)
        Q = 0.005 * rng.standard_normal(n)
        # in an annotated sliding contact with noise-floor effort -> UNKNOWN, never FREE_CONFIRMED
        r = toy_record(Q, np.ones(n, bool), np.array(["sliding"] * n))
        ev = SG.classify(r)
        self.assertEqual(ev["summary"]["FREE_CONFIRMED"]["n_rows"], 0)
        self.assertIn("LOW_MEASURED_EFFORT_IN_ANNOTATED_CONTACT", ev["summary"]["reasons"])
        self.assertEqual(SG.build_segments(r, ev), [])
        hyp = SG.build_segments(r, ev, free_hypotheses=("measured_near_zero_effort_in_contact",))
        self.assertTrue(hyp and all(s.label.startswith("FREE_HYPOTHESIS") for s in hyp))
        # absent wrench -> UNKNOWN (ABSENT_WRENCH)
        Q2 = Q.copy()
        Q2[30:60] = np.nan
        ev2 = SG.classify(toy_record(Q2, np.zeros(n, bool), np.array(["absent"] * n)))
        self.assertEqual(ev2["summary"]["reasons"].get("ABSENT_WRENCH"), 30)
        # released + quiet -> FREE_CONFIRMED; released + significant effort -> UNKNOWN
        ev3 = SG.classify(toy_record(Q, np.zeros(n, bool), np.array(["absent"] * n)))
        self.assertEqual(ev3["summary"]["FREE_CONFIRMED"]["n_rows"], n)
        Q4 = Q.copy()
        Q4[40:70] = 0.5
        ev4 = SG.classify(toy_record(Q4, np.zeros(n, bool), np.array(["absent"] * n)))
        self.assertIn("EFFORT_WITHOUT_ANNOTATED_CONTACT", ev4["summary"]["reasons"])
        # a sticking contact is not moving-mode data
        ev5 = SG.classify(toy_record(Q4, np.ones(n, bool), np.array(["sticking"] * n)))
        self.assertIn("STICK_INTERVAL", ev5["summary"]["reasons"])

    def test_10_insufficient_velocity_spread(self):
        t = np.arange(0, 2.0, 0.04)
        q = 0.5 * t + 1e-4 * np.random.default_rng(1).standard_normal(len(t))   # constant velocity, no decay
        w = FW.fit([seg_from(t, q, 1e-4, 1.0)], spec(), B=30, sensitivity=False)
        self.assertEqual(w["status"], "FREE_UNINFORMATIVE")
        self.assertTrue(all(p["status"] == "UPPER_BOUND_ONLY" for p in w["per_parameter"].values()))
        # one direction, tiny velocity variation: rates not separable (combination at most)
        t2, q2, *_ = simulate(1.0, 0.02, 0.02, v0=0.5, T=1.5, qn=1e-4, seed=3)
        w2 = FW.fit([seg_from(t2, q2, 1e-4, 1.0)], spec(), B=30, sensitivity=False)
        self.assertNotEqual(w2["status"], "FREE_RATIOS_IDENTIFIED")
        self.assertFalse(all(p["status"] == "RATIO_IDENTIFIED" for p in w2["per_parameter"].values()))


class RealDataGuards(unittest.TestCase):
    """Behaviours added after the real-data audit (passivity, q-noise floor, fragmented weak pushes)."""

    def test_passivity_split_detects_unmeasured_input(self):
        # a hidden push accelerates the part during the first second of a 'free' segment
        prof = [(1.0, 0.6), (2.0, 0.0)]
        t, qo, qt, vt, tf, Q = simulate(0.5, 0.1, 0.02, v0=0.2, T=2.98, Qprof=prof, qn=1e-4)
        seg = seg_from(t, qo, 1e-4, 1.0, label="FREE_HYPOTHESIS:measured_near_zero_effort_in_contact")
        kept, a = FW.passivity_split(seg, spec())
        self.assertEqual(a["verdict"], "ENERGY_GAIN_WITHOUT_MEASURED_INPUT")
        self.assertGreater(kept.t[0], 0.8)                          # the energised part is excluded
        w = FW.fit([kept], spec(), B=0, sensitivity=False)
        self.assertLess(rel(w["estimate"]["lam"], 0.2), 0.25)       # c/M = 0.2 recovered from the passive tail
        clean = free_pair()[0]
        k2, a2 = FW.passivity_split(clean, spec())
        self.assertEqual(a2["verdict"], "PASSIVE_CONSISTENT")
        self.assertIs(k2, clean)

    def test_empirical_noise_floor_never_shrinks(self):
        t, qo, *_ = simulate(0.1, 0.04, 0.01, qn=2e-3, seed=4)
        seg = seg_from(t, qo, 2e-4, 1.0)                             # reported sigma 10x too small
        SG.inflate_noise(seg)
        self.assertTrue(seg.evidence["q_noise"]["inflated"])
        self.assertGreater(np.median(seg.q_sigma), 1e-3)
        self.assertTrue(np.all(seg.q_sigma >= seg.q_sigma_reported))
        seg2 = seg_from(t, simulate(0.1, 0.04, 0.01, qn=1e-5, seed=4)[1], 1e-3, 1.0)
        SG.inflate_noise(seg2)
        self.assertTrue(np.all(seg2.q_sigma == 1e-3))                # over-reported sigma is kept, never reduced

    def test_weak_sustained_push_is_forced_and_pulse_kept(self):
        n = 200
        rng = np.random.default_rng(2)
        Q = -0.05 + 0.02 * rng.standard_normal(n)                     # ~2.5 sigma per row, sustained
        Q[150:154] = 0.6                                              # a sharp pulse inside the contact
        r = toy_record(Q, np.ones(n, bool), np.array(["sliding"] * n))
        ev = SG.classify(r)
        forced_rows = ev["summary"]["FORCED_MEASURED"]["n_rows"]
        self.assertGreater(forced_rows, 0.8 * n)
        self.assertTrue(any(d["pulse"] for d in ev["contact_run_decisions"]))


class LearnedEstimators(unittest.TestCase):
    def test_11_pinn_differentiable_physics(self):
        segs = free_pair()
        cfg = NP.NPINNConfig(epochs=300, epochs_warm=100, ramp=50, seeds=(0,), n_colloc=60)
        r = NP.fit_free(segs, spec(), cfg)
        self.assertEqual(r["parameter_names"], ["lam", "gam"])
        self.assertFalse(r["beta_trainable"])
        g = r["seeds"][0]["max_abs_eta_grad"]
        self.assertGreater(g["lam"], 0.0)
        self.assertGreater(g["gam"], 0.0)
        # analytic time derivatives of the trajectory net
        from forceartgs_g9.estimators.general_pinn import _derivs, _Net
        net = _Net(0.0, 1.0, 0.0, 1.0, 8, 2, 0)
        t = torch.linspace(0.1, 0.9, 5, dtype=torch.float64)
        q, qd, _ = _derivs(net, t)
        h = 1e-6
        fd = (net(t + h) - net(t - h)) / (2 * h)
        self.assertLess(float(torch.max(torch.abs(fd - qd))), 1e-6)
        _, forced = forced_set()
        rb = NP.fit_forced(segs, forced, spec(), r, NP.NPINNConfig(phase_b_epochs=200, epochs_warm=50, ramp=50,
                                                                    seeds=(0,), n_colloc=60))
        self.assertIn("beta", rb["parameter_names"])
        self.assertGreater(rb["seeds"][0]["max_abs_eta_grad"]["beta"], 0.0)

    def test_11b_pinn_recovers_free_ratios(self):
        segs = free_pair()
        r = NP.fit_free(segs, spec(), NP.NPINNConfig(epochs=2500, seeds=(0,)))
        self.assertLess(rel(r["estimate"]["lam"], 0.4), TOL["pinn"])
        self.assertLess(rel(r["estimate"]["gam"], 0.1), TOL["pinn"])

    def test_12_ekf_jacobian_psd_nis_events(self):
        tilt = np.array([math.sin(0.2), 0.0, math.cos(0.2)])
        ns = NormalizedDynamicsSpec("revolute", GravityModel("g", "known_shape", "revolute", 0.3, -0.2))
        f = NE.NormalizedAEKF(ns.with_forced())
        x = np.array([0.3, 0.7, NE.inv_softplus(0.4), NE.inv_softplus(0.1), NE.inv_softplus(2.0)])
        y, Fa, _ = f.substep(x, 0.25, 1e-3, False)
        Ffd = f.jacobian_fd(x, 0.25, 1e-3)
        self.assertLess(np.max(np.abs(Fa - Ffd)), 1e-6)
        e = NE.fit(free_pair(), spec())
        self.assertEqual(e["status"], "ok")
        self.assertTrue(0.3 < e["nis"]["mean"] < 3.0)
        self.assertNotIn("beta", e["estimate"])
        # a segment that stops inside (zero crossing) stays finite and PSD
        t, qo, qt, vt, *_ = simulate(1.0, 0.0, 0.5, v0=0.6, T=2.0, qn=1e-4)
        self.assertTrue(np.any(vt == 0.0))
        e2 = NE.fit([seg_from(t, qo, 1e-4, 1.0)], spec())
        self.assertTrue(np.isfinite(list(e2["estimate"].values())).all())
        self.assertEqual(e2["status"], "ok")


class Residual(unittest.TestCase):
    def test_13_residual_neutrality(self):
        from forceartgs_g9.estimators.forcetwin_residual import fit_residual
        from forceartgs_g9.synthetic import make_object
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_g9_forcetwin import PUSHES, door, rec_of, run_episode
        obj = door()
        rec = rec_of(obj, [run_episode(obj, p, seed=k) for k, p in enumerate(PUSHES)])
        beta = [obj.M_true, obj.Fc, obj.c]
        for mode in ("g", "damping", "both"):
            r = fit_residual(rec, beta, mode, epochs=300, patience=60)
            self.assertLessEqual(r["val_whitened_rms"], r["baseline_val_whitened_rms"] + 1e-12)
            self.assertLess(r["max_abs_cosine_with_physical_regressors"], 1e-6)
            if r["dissipation"]:
                self.assertGreaterEqual(r["dissipation"]["min_c"], 0.0)
        z = fit_residual(rec, beta, "both", epochs=1)
        self.assertEqual(z["selected"], "baseline")            # zero-initialised: starts at the physical baseline


class Invariance(unittest.TestCase):
    def test_14_coordinate_reversal(self):
        a = np.array([0.6, 0.0, -0.08])
        a = a / np.linalg.norm(a)
        g = GravityModel.prismatic_ideal(a, G)
        M = 2.0
        Gf = lambda q: M * g(q)
        segs, rsegs = [], []
        for k, (v0, prof) in enumerate(((0.4, [(0.6, 4.0), (0.6, 1.0), (0.6, 3.0)]),
                                        (0.5, [(0.6, 2.0), (0.6, 5.0), (0.5, 1.5)]))):
            t, qo, qt, vt, tf, Q = simulate(M, 1.2, 0.3, jt="prismatic", v0=v0, T=sum(p[0] for p in prof) - 0.02,
                                            Qprof=prof, Gfun=Gf, qn=5e-5, seed=k)
            assert np.all(vt > 0)
            segs.append(seg_from(t, qo, 5e-5, 1.0, label="FORCED_MEASURED", tf=tf, Q=Q, jt="prismatic", name=f"f{k}"))
            rsegs.append(seg_from(t, -qo, 5e-5, -1.0, label="FORCED_MEASURED", tf=tf, Q=-Q, jt="prismatic",
                                  name=f"r{k}"))
            self.assertTrue(np.allclose(Q * np.gradient(qt, t)[0], (-Q) * np.gradient(-qt, t)[0]))   # virtual work
        ns, nr = NormalizedDynamicsSpec("prismatic", g), NormalizedDynamicsSpec("prismatic", g.reversed())
        f1 = FW.forced_scale([], segs, ns)
        f2 = FW.forced_scale([], rsegs, nr)
        for n in ("lam", "gam", "beta"):
            self.assertAlmostEqual(f1["estimate"][n], f2["estimate"][n], places=6)
        self.assertGreater(f1["estimate"]["beta"], 0.0)
        self.assertLess(rel(f1["dimensional"]["M"]["value"], M), TOL["M"])


class Handoff(unittest.TestCase):
    def test_15_handoff_never_exposes_unverified_scale(self):
        segs = free_pair()
        ns = spec()
        w = FW.fit(segs, ns, B=30, sensitivity=False)
        rp = RP.replay(segs, ns, ns.pack(w["estimate"]["lam"], w["estimate"]["gam"]))
        norm = ST.normalized_status(w, rp["aggregate"]["adequate"], rp["aggregate"]["beats_null"],
                                    hypothesis_only=False, kin_auth="ACCEPTED")
        self.assertEqual(norm["status"], "FREE_RATIOS_IDENTIFIED")
        scale = ST.absolute_scale(None, None)
        self.assertEqual(scale, "NOT_OBSERVABLE")
        ready = ST.simulator_readiness(norm, scale, "ACCEPTED")
        self.assertEqual(ready, "PRIOR_SCALED_PREVIEW")
        prior = dict(status="PRIOR_FOR_SIM", I_eff_range=[0.1, 1.0], unit="kg m^2")
        h = ST.handoff_preview("revolute", w, norm, scale, ready, prior)
        self.assertIsNone(h["absolute_scale"]["M"]["estimate"])
        self.assertEqual(h["simulator"]["parameter_set_source"], "EXPLICIT_PRIOR_NOT_MEASURED")
        for sc in h["simulator"]["preview"]["scenarios"]:
            self.assertGreater(sc["M"], 0.0)
            self.assertAlmostEqual(sc["c"], w["estimate"]["lam"] * sc["M"])
            self.assertGreater(sc["Fc"], 0.0)
        self.assertIn("NOT IDENTIFIED", h["simulator"]["preview"]["label"])
        self.assertTrue(rp["aggregate"]["adequate"])
        self.assertTrue(rp["aggregate"]["beats_null"])


class Bridge(unittest.TestCase):
    def test_model_spec_bridge_and_simulator_reuse(self):
        from forceartgs_g9.estimators.model_spec import ModelSpec
        ms = ModelSpec(GeneralTerms(static=False, gravity=False), "revolute", "joint_space_only")
        ns = NormalizedDynamicsSpec.from_model_spec(ms)
        self.assertEqual(ns.names, ["lam", "gam"])
        t, _, qt, *_ = simulate(0.25, 0.1, 0.025, qn=0.0)
        r = ns.simulate(ns.pack(0.4, 0.1), t, 0.0, 0.0, 1.0)
        self.assertLess(np.max(np.abs(r.q - qt)), TOL["equiv"])


if __name__ == "__main__":
    unittest.main()
