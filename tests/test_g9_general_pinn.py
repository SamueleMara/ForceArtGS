"""GeneralPINN: autograd derivatives, full-equation residual, gradients through ONE density, weak / impulse /
stick losses, synthetic recovery with open-loop (frozen-theta) held-out replay. Synthetic truth only."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9 import physics_lib  # noqa: E402
from forceartgs_g9.dynamics.general_1dof import GeneralTerms  # noqa: E402
from forceartgs_g9.estimators import general_pinn as GP  # noqa: E402
from forceartgs_g9.estimators.model_spec import ModelSpec  # noqa: E402
from forceartgs_g9.estimators.torch_model import DT, TorchDynamics, zoh_integral, zoh_torch  # noqa: E402
from forceartgs_g9.synthetic import ZOH, episode, make_object  # noqa: E402

physics_lib.load()
torch.set_num_threads(4)


class PINNCore(unittest.TestCase):
    def test_autograd_derivatives_in_physical_seconds(self):
        A, w = 0.3, 2.7
        f = lambda t: A * torch.sin(w * t)
        t = torch.linspace(0.1, 2.0, 50, dtype=DT)
        q, qd, qdd = GP._derivs(f, t)
        np.testing.assert_allclose(qd.detach(), A * w * np.cos(w * t.numpy()), atol=1e-12)
        np.testing.assert_allclose(qdd.detach(), -A * w * w * np.sin(w * t.numpy()), atol=1e-11)
        net = GP._Net(5.0, 9.0, 0.0, 1.0, 8, 2, 0)                    # normalised input, chain rule automatic
        tt = torch.linspace(5.5, 8.5, 7, dtype=DT)
        _, d1, _ = GP._derivs(net, tt)
        h = 1e-6
        fd = (net(tt + h) - net(tt - h)) / (2 * h)
        np.testing.assert_allclose(d1.detach(), fd.detach(), rtol=1e-6, atol=1e-9)

    def test_full_residual_zero_and_switches(self):
        spec = ModelSpec(GeneralTerms(static=False, bias=True), "revolute", "joint_space_only")
        self.assertEqual(spec.names, ["M", "c", "Fc", "gA", "gB", "b"])
        dyn = TorchDynamics(spec)
        th = dict(M=0.4, c=0.2, Fc=0.05, gA=0.3, gB=-0.2, b=0.07)
        eta = {k: torch.tensor(math.log(v) if k in ("M", "c", "Fc") else v, dtype=DT) for k, v in th.items()}
        t = torch.linspace(0.05, 1.4, 40, dtype=DT)
        q = 0.5 * torch.sin(1.1 * t) + 0.2
        qd, qdd = 0.55 * torch.cos(1.1 * t), -0.605 * torch.sin(1.1 * t)
        Q = (th["M"] * qdd + th["c"] * qd + th["Fc"] * torch.sign(qd) + th["gA"] * torch.cos(q) +
             th["gB"] * torch.sin(q) + th["b"])
        r = dyn.residual(dyn.values(eta), q, qd, qdd, Q)
        self.assertLess(float(r.abs().max()), 1e-12)
        off = ModelSpec(GeneralTerms(static=False, coulomb=False, gravity=False), "revolute", "joint_space_only")
        self.assertEqual(off.names, ["M", "c"])                     # disabled terms leave the trainable vector
        d2 = TorchDynamics(off)
        e2 = dict(M=eta["M"], c=eta["c"])
        r2 = d2.residual(d2.values(e2), q, qd, qdd, th["M"] * qdd + th["c"] * qd)
        self.assertLess(float(r2.abs().max()), 1e-12)

    def test_physics_gradient_reaches_one_density_through_inertia_and_gravity(self):
        obj = make_object("revolute", terms=GeneralTerms(static=False), g=(0, 0, -9.81))
        spec = ModelSpec(obj.terms, "revolute", "geometry_coupled", obj.mass_model, obj.gravity)
        self.assertEqual(spec.names, ["alpha", "c", "Fc"])           # no separate M / mu / gA / gB
        dyn = TorchDynamics(spec)
        eta = dyn.eta0([obj.density, obj.c, obj.Fc])
        t = torch.linspace(0.1, 1.0, 30, dtype=DT)
        q, qd, qdd = 0.3 * torch.sin(2 * t), 0.6 * torch.cos(2 * t), -1.2 * torch.sin(2 * t)
        r = dyn.residual(dyn.values(eta), q, qd, qdd, torch.zeros_like(t))
        g = torch.autograd.grad(r.sum(), eta["alpha"])[0]
        A1, B1 = dyn.gA1, dyn.gB1
        inertia_part = float(dyn.M1 * qdd.sum())
        gravity_part = float((A1 * torch.cos(q) + B1 * torch.sin(q)).sum())
        self.assertNotAlmostEqual(inertia_part, 0.0, places=6)
        self.assertNotAlmostEqual(gravity_part, 0.0, places=6)
        self.assertAlmostEqual(float(g) / obj.density, inertia_part + gravity_part, places=8)   # d/dlog rho

    def test_losses_and_gradients_on_episode(self):
        obj = make_object("prismatic", terms=GeneralTerms(), g=(0, 0, -9.81))
        it, _ = episode(obj, variant=0, seed=0)
        spec = ModelSpec(obj.terms, "prismatic", "geometry_coupled", obj.mass_model, obj.gravity)
        cfg = GP.PINNConfig(steps_data=50, steps_joint=0, n_seeds=1)
        ep = GP._Episode(it, cfg, obj.M_true, obj.c)
        net = GP._Net(ep.t0, ep.t1, float(it.q_obs[0]), ep.q_scale, 16, 2, 0)
        dyn = TorchDynamics(spec)
        eta = dyn.eta0([obj.density, obj.c, obj.Fc, obj.Fs - obj.Fc])
        tc, mov, rest, ri = ep.colloc(200, 0.4, np.random.default_rng(0))
        if len(ep.knots):
            self.assertGreater(float(np.min(np.abs(tc.numpy()[:, None] - ep.knots[None, :]))), 0)   # no knot hits
        L = GP.pinn_losses(dyn, eta, ep, net, cfg, tc, mov, rest, ri)
        for k in ("data", "phys", "stick", "breakaway", "weak"):
            self.assertTrue(torch.isfinite(L[k]))
        L["phys"].backward(retain_graph=True)
        for n in ("alpha", "c", "Fc"):
            self.assertIsNotNone(eta[n].grad)
            self.assertNotEqual(float(eta[n].grad), 0.0)                 # physics -> physical parameters
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.parameters()))
        self.assertTrue(ep.rest_runs and ep.breakaway_t)                 # sticking + breakaway constraints active

    def test_weak_impulse_balance_and_sliding_only_fs(self):
        e = ZOH([0.0, 1.0, 1.08, 2.0], [0.0, 10.0, 0.0, 0.0])             # 80 ms, 10 N pulse
        self.assertAlmostEqual(zoh_integral(e, 0.5, 1.5), 0.8, places=12)
        self.assertAlmostEqual(zoh_integral(e, 1.02, 1.05), 0.3, places=12)
        tt = torch.tensor([0.5, 1.04, 1.5], dtype=DT)
        np.testing.assert_allclose(zoh_torch(e, tt).numpy(), [0.0, 10.0, 0.0])
        # frictionless free mass: M dv = J exactly -> the weak impulse residual of the true trajectory is zero
        M = 2.0
        v = lambda t: torch.where(t < 1.0, torch.zeros_like(t), torch.where(t < 1.08, 5.0 * (t - 1.0),
                                                                            torch.full_like(t, 0.4)))
        self.assertAlmostEqual(float(M * (v(torch.tensor(1.5, dtype=DT)) - v(torch.tensor(0.9, dtype=DT)))),
                               zoh_integral(e, 0.9, 1.5), places=12)
        # sliding-only data: no rest / breakaway samples -> the breakaway loss carries no information on Fs
        obj = make_object("prismatic", terms=GeneralTerms(gravity=False))
        it, _ = episode(obj, [(0.6, 3.0), (0.6, 4.5), (0.6, 2.8)], seed=1)
        from forceartgs_g9.estimators.model_spec import G9Interaction
        it = G9Interaction(it.name, it.joint_type, it.t_obs[6:], it.q_obs[6:], it.q_std[6:], it.effort)
        cfg = GP.PINNConfig(n_seeds=1)
        ep = GP._Episode(it, cfg, obj.M_true, obj.c)
        self.assertEqual(ep.breakaway_t, [])


class PINNRecovery(unittest.TestCase):
    def test_full_model_recovery_and_open_loop_heldout(self):
        from forceartgs_g9.estimators.general_integral_ls import general_integral_ls, theta0_from_ls
        from forceartgs_g9.validation.heldout_prediction import heldout_report
        obj = make_object("prismatic", terms=GeneralTerms(static=False), g=(0, 0, -9.81))
        its = [episode(obj, variant=v, seed=v)[0] for v in range(2)]
        held = [episode(obj, variant=3, seed=40, declared_initial=True)[0]]
        spec = ModelSpec(obj.terms, "prismatic", "geometry_coupled", obj.mass_model, obj.gravity)
        th0 = theta0_from_ls(general_integral_ls(its, spec), spec, dict(effort=5.0))
        r = GP.general_pinn_fit(its, spec, th0, GP.PINNConfig(n_seeds=1, steps_data=3000, steps_joint=3000))
        e = r["estimates"]
        self.assertAlmostEqual(e["alpha"]["value"] / obj.density, 1.0, delta=0.05)
        self.assertAlmostEqual(e["c"]["value"] / obj.c, 1.0, delta=0.15)
        self.assertAlmostEqual(e["Fc"]["value"] / obj.Fc, 1.0, delta=0.15)
        self.assertIn("autograd", r["differentiation"])
        h = heldout_report(spec, np.asarray(r["x"]), held)                 # frozen theta, declared state only
        self.assertEqual(h["n_independent"], 1)
        self.assertLess(h["summary"]["mean_rmse_over_sigma"], 15.0)


if __name__ == "__main__":
    unittest.main()
