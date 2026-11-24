"""GeneralAugmentedEKF: transition / Jacobian, Joseph update, native-force propagation with sparse q,
synthetic recovery + coverage, non-identification, sign errors. Synthetic truth only."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9 import physics_lib  # noqa: E402
from forceartgs_g9.dynamics.general_1dof import GeneralTerms, _phi  # noqa: E402
from forceartgs_g9.estimators.general_ekf import EKFConfig, GeneralAugmentedEKF, general_akf_fit  # noqa: E402
from forceartgs_g9.estimators.general_integral_ls import general_integral_ls, theta0_from_ls  # noqa: E402
from forceartgs_g9.estimators.model_spec import G9Interaction, ModelSpec  # noqa: E402
from forceartgs_g9.synthetic import ZOH, episode, make_object  # noqa: E402

physics_lib.load()


def spec_of(obj):
    return ModelSpec(obj.terms, obj.joint_type, "geometry_coupled", obj.mass_model, obj.gravity)


class EKFCore(unittest.TestCase):
    def test_jacobian_matches_analytic_viscous_step(self):
        spec = ModelSpec(GeneralTerms(coulomb=False, static=False, gravity=False), "prismatic", "joint_space_only")
        ekf = GeneralAugmentedEKF(spec)
        M, c, Q, h = 2.0, 3.0, 1.5, 0.02
        x = np.array([0.1, 0.3, math.log(M), math.log(c)])
        F, sw = ekf.jacobian(x, Q, h)
        k = c / M
        p1, p2 = _phi(k * h)
        self.assertFalse(sw)
        self.assertAlmostEqual(F[0, 1], h * p1, places=6)                 # dq'/dv
        self.assertAlmostEqual(F[1, 1], math.exp(-k * h), places=6)       # dv'/dv
        self.assertEqual(F[2, 2], 1.0)
        self.assertEqual(F[3, 3], 1.0)                                    # eta' = 0
        self.assertNotEqual(F[1, 2], 0.0)                                 # parameter sensitivity enters

    def test_joseph_psd_native_force_and_sparse_q(self):
        obj = make_object("prismatic", terms=GeneralTerms(static=False, gravity=False))
        it, _ = episode(obj, variant=0, seed=0, cam_hz=12.0)               # sparse q (12 Hz) vs 100 Hz force
        spec = spec_of(obj)
        ekf = GeneralAugmentedEKF(spec)
        m, C, lg, mineig = ekf.run_episode(it, np.log([obj.density, obj.c, obj.Fc]), np.eye(3) * 0.25, 0.05,
                                           obj.M_true)
        self.assertGreaterEqual(mineig, -1e-12)                          # Joseph form keeps P PSD
        n_force = int(np.sum((it.effort.t_s > it.t_obs[0]) & (it.effort.t_s < it.t_obs[-1])))
        self.assertGreaterEqual(len(lg["t"]), n_force)                    # every native force step propagated
        self.assertEqual(len(lg["nis"]), len(it.t_obs) - 1)               # updates only at camera times
        # an 80 ms impulse between two camera frames still moves the predicted state
        e = ZOH([0.0, 0.50, 0.58], [0.0, 30.0, 0.0])
        t = np.arange(0.0, 1.2, 1 / 12.0)
        it2 = G9Interaction("p", "prismatic", t, np.zeros(len(t)), 5e-4, e)
        _, _, lg2, _ = ekf.run_episode(it2, np.log([obj.density, obj.c, obj.Fc]), np.eye(3) * 1e-6, 0.05, obj.M_true)
        vmax = max(abs(v) for v in lg2["v"])
        self.assertGreater(vmax, 0.05)


class EKFRecovery(unittest.TestCase):
    def test_recovery_and_coverage_prismatic(self):
        obj = make_object("prismatic", terms=GeneralTerms(static=False), g=(0, 0, -9.81))
        its = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
        spec = spec_of(obj)
        th0 = theta0_from_ls(general_integral_ls(its, spec), spec, dict(effort=5.0))
        r = general_akf_fit(its, spec, th0)
        for n, tv in (("alpha", obj.density), ("c", obj.c), ("Fc", obj.Fc)):
            e = r["estimates"][n]
            self.assertLess(abs(e["value"] - tv), 4 * e["std"] + 0.03 * tv, n)    # coverage
        self.assertTrue(0.3 < r["nis"]["mean"] < 2.0, r["nis"])
        self.assertIn("eta_alpha", r["augmented_state"])                         # ONE density state

    def test_constant_speed_not_identified_and_sign_error(self):
        from forceartgs_g9.validation.general_identifiability import structural_checks
        obj = make_object("prismatic", terms=GeneralTerms(static=False, gravity=False))
        v, T = 0.05, 3.0
        t = np.arange(0, T, 1 / 30)
        q = v * t + 5e-4 * np.random.default_rng(0).standard_normal(len(t))
        e = ZOH(np.arange(0, T, 0.01), np.full(int(T * 100), obj.c * v + obj.Fc))
        it = G9Interaction("steady", "prismatic", t, q, 5e-4, e)
        spec = spec_of(obj)
        st = structural_checks([it], spec)
        self.assertIn("alpha", [p for p, _ in st["unidentifiable"]])            # independent of the EKF covariance
        its = [episode(obj, variant=k, seed=k)[0] for k in range(2)]
        for x in its:
            x.effort = ZOH(x.effort.t_s, -np.asarray(x.effort.levels))           # inverted sign
        th0 = np.array([obj.density, obj.c, obj.Fc])
        bad = general_akf_fit(its, spec, th0)
        good = general_akf_fit([episode(obj, variant=k, seed=k)[0] for k in range(2)], spec, th0)
        self.assertGreater(bad["nis"]["mean"], 100 * good["nis"]["mean"])        # physically inconsistent
        self.assertFalse(bad["nis"]["consistent"])


if __name__ == "__main__":
    unittest.main()
