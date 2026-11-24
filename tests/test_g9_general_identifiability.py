"""G9 Gate C: estimation and scientific identifiability (synthetic truth only; ~3-5 min)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9 import physics_lib  # noqa: E402
from forceartgs_g9.core import run_g9  # noqa: E402
from forceartgs_g9.dynamics.general_1dof import GeneralTerms, sim_params, simulate_general  # noqa: E402
from forceartgs_g9.estimators.general_constrained import general_rollout_fit  # noqa: E402
from forceartgs_g9.estimators.general_integral_ls import general_integral_ls, theta0_from_ls  # noqa: E402
from forceartgs_g9.estimators.model_spec import G9Interaction, ModelSpec  # noqa: E402
from forceartgs_g9.synthetic import ZOH, episode, make_object, profile  # noqa: E402
from forceartgs_g9.testing import synthetic_cfg, synthetic_inputs  # noqa: E402
from forceartgs_g9.validation.general_identifiability import (diagnose_general,  # noqa: E402
                                                              standard_perturbations, structural_checks)
from forceartgs_g9.validation.heldout_prediction import heldout_report, leakage, predict_heldout  # noqa: E402
from forceartgs_g9.validation.model_adequacy import adequacy  # noqa: E402

physics_lib.load()


def fit(its, spec, n_starts=2, pert=None):
    ls = general_integral_ls(its, spec)
    eff = float(np.max(np.abs(np.concatenate([i.effort.levels for i in its]))))
    f = general_rollout_fit(its, spec, theta0_from_ls(ls, spec, dict(effort=eff)), n_starts=n_starts,
                            scale_hint=dict(effort=eff))
    return ls, f, diagnose_general(f, its, spec, perturb=pert or {})


def spec_of(obj, **kw):
    return ModelSpec(kw.pop("terms", obj.terms), obj.joint_type, kw.pop("source", "geometry_coupled"),
                     obj.mass_model, obj.gravity, **kw)


def constant_speed(obj, v=0.05, T=3.0, cam=30, noise=5e-4, seed=0):
    """Steady push at constant speed (no acceleration, one direction), started before observation."""
    rng = np.random.default_rng(seed)
    Qc = obj.c * v + obj.Fc + (obj.b if obj.terms.bias else 0.0)
    t, Q = profile([(T, Qc)])
    tq = np.arange(0, T, 1 / cam)
    q = v * tq + noise * rng.standard_normal(len(tq))
    return G9Interaction("steady", obj.joint_type, tq, q, noise, ZOH(t, Q))


class GateC(unittest.TestCase):
    def test_c01_full_model_recovery_r_and_p(self):
        for kind in ("revolute", "prismatic"):
            obj = make_object(kind)
            its = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
            _, f, d = fit(its, spec_of(obj))
            e = f["estimates"]
            self.assertAlmostEqual(e["alpha"]["value"] / obj.density, 1.0, delta=0.02, msg=kind)
            self.assertAlmostEqual(e["c"]["value"] / obj.c, 1.0, delta=0.06, msg=kind)
            self.assertAlmostEqual(e["Fc"]["value"] / obj.Fc, 1.0, delta=0.06, msg=kind)
            lo, hi = e["dFs"]["interval95"]
            self.assertTrue(lo - 0.02 <= obj.Fs - obj.Fc <= hi + 0.02, msg=(kind, lo, hi))
            self.assertEqual(d["status"], "PASS", msg={k: v["reasons"] for k, v in d["parameters"].items()})

    def test_c02_no_false_extra_terms_and_heldout_model_choice(self):
        obj = make_object("prismatic", Fc=0.0, Fs=0.0)                 # truth: viscous only (+ inertia)
        its = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
        held = [episode(obj, variant=3, seed=50, declared_initial=True)[0]]
        _, f, d = fit(its, spec_of(obj, terms=GeneralTerms(static=False)))
        self.assertFalse(d["parameters"]["Fc"]["identifiable"])        # zero Coulomb is not "identified"
        vis = spec_of(obj, terms=GeneralTerms(coulomb=False, static=False))
        _, fv, dv = fit(its, vis)
        self.assertEqual(dv["status"], "PASS")
        hf = heldout_report(spec_of(obj, terms=GeneralTerms(static=False)), np.asarray(f["x"][:3]), held)
        hv = heldout_report(vis, np.asarray(fv["x"][:2]), held)
        self.assertLessEqual(hv["summary"]["mean_rmse_over_sigma"], hf["summary"]["mean_rmse_over_sigma"] * 1.05)

    def test_c03_constant_speed_and_one_direction_confounds(self):
        obj = make_object("prismatic", terms=GeneralTerms(static=False, bias=True, gravity=False), b=0.3)
        it = constant_speed(obj)
        spec = spec_of(obj)
        st = structural_checks([it], spec)
        self.assertIn("alpha", [p for p, _ in st["unidentifiable"]])
        self.assertTrue(any(c[:2] == ("Fc", "b") for c in st["confounds"]))
        _, f, d = fit([it], spec)
        for p in ("alpha", "Fc", "b"):
            self.assertFalse(d["parameters"][p]["identifiable"], p)

    def test_c04_prismatic_gravity_vs_bias_confound(self):
        obj = make_object("prismatic", axis=[0.0, 0.99985, 0.0175], terms=GeneralTerms(static=False, bias=True),
                          b=0.5)                                  # 1 deg incline: G_P ~ 0.9 N < Fc
        its = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
        st = structural_checks(its, spec_of(obj))
        self.assertTrue(any("PRISMATIC_GRAVITY_IS_CONSTANT" in c[2] for c in st["confounds"]))
        _, f, d = fit(its, spec_of(obj))
        self.assertLess(f["residual_rms_sigma"], 1.5)                    # tiny training residual ...
        self.assertFalse(d["parameters"]["b"]["identifiable"])           # ... still confounded

    def test_c05_no_breakaway_static_unidentifiable(self):
        obj = make_object("prismatic", terms=GeneralTerms(gravity=False))
        segs = [(0.6, 3.0), (0.6, 4.5), (0.6, 2.8), (0.6, 5.0), (0.6, 3.5)]   # pushing, never released to rest
        it, tr = episode(obj, segs, seed=1)
        it = G9Interaction(it.name, it.joint_type, it.t_obs[6:], it.q_obs[6:], it.q_std[6:], it.effort)
        self.assertGreater(np.min(np.abs(np.gradient(tr["q"][6:], it.t_obs))), 0.0)
        _, f, d = fit([it], spec_of(obj))
        self.assertFalse(d["parameters"]["dFs"]["identifiable"])
        self.assertTrue(any("NO_BREAKAWAY" in r for r in d["parameters"]["dFs"]["reasons"]))
        self.assertTrue(d["parameters"]["c"]["identifiable"] and d["parameters"]["Fc"]["identifiable"])

    def test_c06_reversals_and_speeds_separate_c_and_fc(self):
        obj = make_object("prismatic", terms=GeneralTerms(static=False, gravity=False))
        spec = spec_of(obj)
        one = constant_speed(obj, v=0.08)
        ls1 = general_integral_ls([one], spec)
        rich = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
        ls2 = general_integral_ls(rich, spec)
        self.assertGreater(abs(ls1["correlation"]["c~Fc"]), 0.95)
        self.assertLess(abs(ls2["correlation"]["c~Fc"]), abs(ls1["correlation"]["c~Fc"]))
        self.assertEqual(ls2["directions"], [-1, 1])
        self.assertLess(ls2["estimates"]["c"]["rel_std"], 0.05)

    def test_c07_hinge_gravity_basis_needs_range(self):
        obj = make_object("revolute", terms=GeneralTerms(static=False))
        js = spec_of(obj, source="joint_space_only")
        its = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
        _, f, d = fit(its, js)
        self.assertTrue(d["parameters"]["gB"]["identifiable"], d["parameters"]["gB"]["reasons"])
        small = make_object("revolute", terms=GeneralTerms(static=False))
        tiny = [episode(small, [(0.3, 0.0), (0.6, 0.0, 0.3), (0.6, 0.0)], seed=3)[0]]
        _, f2, d2 = fit(tiny, js)
        self.assertFalse(d2["status"] == "PASS")
        self.assertFalse(d2["parameters"]["gB"]["identifiable"] and d2["parameters"]["gA"]["identifiable"])

    def test_c08_shared_parameters_over_interactions(self):
        obj = make_object("prismatic", terms=GeneralTerms(static=False, gravity=False))
        its = [episode(obj, variant=v, seed=10 + v)[0] for v in range(3)]
        spec = spec_of(obj)
        _, f1, _ = fit(its[:1], spec)
        _, f3, _ = fit(its, spec)
        self.assertLess(f3["estimates"]["alpha"]["std"], f1["estimates"]["alpha"]["std"])
        self.assertEqual(sorted(f3["estimates"]), sorted(spec.names))       # only shared physical parameters
        self.assertEqual(len(f3["nuisance"]), 3)                           # one (q0, v0) per interaction

    def test_c09_wrong_sign_and_lag_fail_gates(self):
        with tempfile.TemporaryDirectory() as td:
            obj, inp, _ = synthetic_inputs("prismatic", td, terms=GeneralTerms(static=False), n_heldout=1)
            for it in inp.train:
                it.effort = ZOH(it.effort.t_s, -np.asarray(it.effort.levels))   # sign error
            r = run_g9(inp, synthetic_cfg(GeneralTerms(static=False)), Path(td) / "g9", log=lambda m: None)
            self.assertFalse(r["manifest"]["readiness"]["PARAMETERS_IDENTIFIED"]
                             and r["manifest"]["readiness"]["MODEL_VALIDATED"])
            self.assertIsNone(r["physics_spec_v1"])
            obj, inp, _ = synthetic_inputs("prismatic", Path(td) / "lag", terms=GeneralTerms(static=False),
                                           n_heldout=1, force_status="FORCE_CALIBRATION_UNVERIFIED")
            inp.force_audit["failing"] = ["time_sync"]
            r2 = run_g9(inp, synthetic_cfg(GeneralTerms(static=False)), Path(td) / "g9lag", log=lambda m: None)
            self.assertEqual(r2["manifest"]["status"], "BLOCKED")
            self.assertTrue(any("time_sync" in b for b in r2["manifest"]["blocked"]))
        o = make_object("revolute", terms=GeneralTerms(coulomb=False, static=False))
        its = [episode(o, variant=v, seed=v)[0] for v in range(2)]
        _, _, d20 = fit(its, spec_of(o), pert=standard_perturbations(lag_s=0.02))
        self.assertGreater(d20["parameters"]["c"]["std_systematic"], 3 * d20["parameters"]["c"]["std_statistical"])

    def test_c10_contact_point_uncertainty_propagates_and_prismatic_gauge(self):
        from forceartgs_physics.forces.projection import generalized_effort
        from forceartgs_physics.schemas import ForceRecord, JointKinematics
        t = np.linspace(0, 1, 40)
        F = np.tile([0.0, 2.0, 0.5], (40, 1))
        P = np.tile([0.1, 0.0, 0.3], (40, 1))
        q1 = generalized_effort(JointKinematics("prismatic", [0, 1, 0], [0, 0, 0], "W"),
                                ForceRecord(t_s=t, force_N=F, point_m=P, frame="W")).levels
        q2 = generalized_effort(JointKinematics("prismatic", [0, 1, 0], [5, -3, 2], "W"),
                                ForceRecord(t_s=t, force_N=F, point_m=P, frame="W")).levels
        np.testing.assert_array_equal(q1, q2)
        obj = make_object("revolute", terms=GeneralTerms(static=False))
        its = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
        lever = {"contact_point_+1cm": lambda it: ZOH(it.effort.t_s, np.asarray(it.effort.levels) * (0.36 / 0.35))}
        _, _, d = fit(its, spec_of(obj), pert=lever)
        self.assertGreater(abs(d["systematic_shifts"]["contact_point_+1cm"]["alpha"]), 0.0)
        self.assertGreater(d["parameters"]["alpha"]["std_systematic"], 0.0)

    def test_c11_unknown_contact_is_excluded_not_zero(self):
        from forceartgs_g9.adapters.forceartgs_g9 import contact_efforts, force_audit
        n = 200
        rows = np.arange(100, 100 + n)
        T = np.tile(np.eye(4), (n, 1, 1))
        T[:, :3, 3] = [0.3, 0.0, 0.4]
        W = np.zeros((n, 6))
        W[:, 1] = -3.0
        auth = SimpleNamespace(joint_type="prismatic", axis_world=np.array([0, 1.0, 0]), origin_world=None,
                               source_frame_index=rows[::2], q_valid=np.ones(n // 2, bool),
                               timestamp_s=rows[::2] / 50.0, q=np.linspace(0, 0.1, n // 2), q_sigma=None,
                               q_units="m", rgbd_frame_index=rows[::2], reference_time_index=0)
        with tempfile.TemporaryDirectory() as td:
            arc = Path(td) / "robot_archive.npz"
            np.savez(arc, T_base_ee=T, T_base_tcp=T, wrench=W, source_row=rows)
            seg = Path(td) / "seg.yaml"
            seg.write_text("schema: forceartgs.contact_segments/v1\nsource: test\nsegments:\n"
                           "  - {segment_id: c0, start_source_frame: 140, end_source_frame: 200, "
                           "contact_state: contact_confirmed, slip_mode: sliding, provenance: operator_label}\n")
            (Path(td) / "metadata").mkdir()
            fa = force_audit(td, auth, arc, seg)
            self.assertEqual(fa["status"], "FORCE_CALIBRATION_UNVERIFIED")
            self.assertEqual(fa["rejections"]["unknown_contact_excluded"], n - 61)
            its = contact_efforts(auth, arc, fa["segments"], multiplier=-1.0)     # explicit (test declares it)
            self.assertEqual(len(its), 1)
            self.assertGreaterEqual(its[0].t_obs.min(), 140 / 50.0)
            self.assertLessEqual(its[0].t_obs.max(), 200 / 50.0)
            self.assertTrue(np.all(np.asarray(its[0].effort.levels) == 3.0))   # object side: -1 x recorded

    def test_c12_no_calibration_blocks_and_diagnostic_is_labelled(self):
        with tempfile.TemporaryDirectory() as td:
            obj, inp, _ = synthetic_inputs("prismatic", td, terms=GeneralTerms(static=False), n_heldout=0,
                                           force_status="FORCE_CALIBRATION_UNVERIFIED")
            r = run_g9(inp, synthetic_cfg(GeneralTerms(static=False)), Path(td) / "g9", log=lambda m: None)
            self.assertEqual(r["manifest"]["status"], "BLOCKED")
            self.assertIsNone(r["fit"])
            self.assertIsNone(r["physics_spec_v1"])
            cfg = synthetic_cfg(GeneralTerms(static=False), pipeline={"on_missing_calibration": "diagnostic_only"})
            r2 = run_g9(inp, cfg, Path(td) / "g9d", log=lambda m: None)
            self.assertEqual(r2["manifest"]["status"], "DYNAMICS_DIAGNOSTIC")
            self.assertTrue(r2["manifest"]["label"].startswith("NOT_PHYSICAL"))
            p = json.loads((Path(td) / "g9d" / "physics_parameters.json").read_text())
            self.assertFalse(p["production_fields_valid"])
            self.assertEqual(p["terms"]["viscous"]["status"], "diagnostic_only")

    def test_c13_heldout_uses_frozen_parameters_and_declared_state(self):
        obj = make_object("prismatic", terms=GeneralTerms(static=False, gravity=False))
        spec = spec_of(obj)
        th = np.array([obj.density, obj.c, obj.Fc])
        it, _ = episode(obj, variant=2, seed=7, declared_initial=True)
        a = predict_heldout(spec, th, it)
        it2 = G9Interaction(it.name, it.joint_type, it.t_obs, it.q_obs + 0.05 * (it.t_obs > 0.5), it.q_std,
                            it.effort, it.initial_state)
        b = predict_heldout(spec, th, it2)
        np.testing.assert_array_equal(a["q_pred"], b["q_pred"])           # future q never enters the forecast
        self.assertLess(a["rmse_over_sigma"], 2.0)
        nd = G9Interaction(it.name, it.joint_type, it.t_obs, it.q_obs, it.q_std, it.effort, None)
        self.assertEqual(predict_heldout(spec, th, nd)["status"], "NOT_RUN")

    def test_c14_force_refined_q_is_not_an_independent_target(self):
        obj = make_object("prismatic", terms=GeneralTerms(static=False, gravity=False))
        it, _ = episode(obj, variant=2, seed=7, declared_initial=True)
        it.meta["frontend_force_used"] = True
        self.assertEqual(leakage(it)["status"], "CONTAMINATED")
        rep = heldout_report(spec_of(obj), np.array([obj.density, obj.c, obj.Fc]), [it])
        self.assertEqual(rep["status"], "NOT_RUN")
        self.assertEqual(rep["n_contaminated"], 1)
        it.meta.update(frontend_force_used=False, frontend_contact_used=True)
        self.assertEqual(leakage(it)["status"], "SHARED_TCP_GEOMETRY")

    def test_c15_frozen_library_untouched_and_deterministic(self):
        import subprocess
        prov = physics_lib.provenance()
        root = prov["root"]
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True, text=True).stdout
        self.assertEqual(head, physics_lib.PINNED_COMMIT)
        self.assertEqual(dirty.strip(), "")
        from forceartgs_physics import estimators
        self.assertEqual(tuple(estimators.VALIDATED), ("PINN", "Backprop", "AKF", "MLP"))
        from forceartgs_g9.estimators.frozen_baselines import run_baselines
        obj = make_object("prismatic", terms=GeneralTerms(coulomb=False, static=False, gravity=False))
        it, _ = episode(obj, variant=0, seed=0)
        a = run_baselines([it], obj.axis, None, ["IntegralLS"])["results"]["IntegralLS"][0]
        b = run_baselines([it], obj.axis, None, ["IntegralLS"])["results"]["IntegralLS"][0]
        self.assertEqual((a["inertia"], a["damping"]), (b["inertia"], b["damping"]))


if __name__ == "__main__":
    unittest.main()
