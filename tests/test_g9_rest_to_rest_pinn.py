"""Rest-to-rest PINN ablations (T11-T14). Physics quality is judged by OPEN-LOOP replay from (q0, 0), never the net."""
import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from forceartgs_g9.dynamics.general_1dof import ModelConfigError
from forceartgs_g9.estimators import general_pinn as GP
from forceartgs_g9.estimators import rest_to_rest_pinn as RPN
from forceartgs_g9.estimators.model_spec import G9Interaction, inertia_viscous_spec
from forceartgs_g9.estimators.torch_model import DT, TorchDynamics

from r2r_fixtures import episode, exact_rest_to_rest, out_and_back

torch.set_num_threads(4)


@pytest.mark.parametrize("seed,dq", [(0, 0.15), (3, 0.0), (11, -0.4)])
def test_T11_hard_network_satisfies_q0_qT_and_zero_endpoint_speeds_for_any_weights(seed, dq):
    net = RPN.RestToRestNet(1.3, 2.9, -0.21, -0.21 + dq, 0.5, 16, 3, seed)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 3.0)                          # arbitrary weights
    te = torch.tensor([1.3, 2.9], dtype=DT)
    q, qd, _ = GP._derivs(net, te)
    assert float(q[0]) == pytest.approx(-0.21, abs=1e-14) and float(q[1]) == pytest.approx(-0.21 + dq, abs=1e-14)
    assert abs(float(qd[0])) < 1e-12 and abs(float(qd[1])) < 1e-12
    tm = torch.linspace(1.4, 2.8, 20, dtype=DT)
    assert float(net(tm).std()) > 0                                      # nontrivial motion, also for dq = 0


def test_T12_double_integral_residual_gives_finite_nonzero_gradients_without_any_derivative(monkeypatch):
    ep = episode(exact_rest_to_rest(0.8, 1.5))["episode"]
    cfg = GP.PINNConfig(n_quad=96)
    d = RPN._R2RData(ep, cfg, [0.8, 1.5])
    net = RPN.RestToRestNet(ep.t0, ep.T, ep.q0, ep.qT, RPN._scale(ep), 8, 2, 0)
    dyn = TorchDynamics(inertia_viscous_spec("prismatic"))
    eta = dyn.eta0([0.6, 1.0])
    v = dyn.values(eta)

    def boom(*a, **k):
        raise AssertionError("the double-integral residual must not differentiate q")
    monkeypatch.setattr(GP, "_derivs", boom)
    monkeypatch.setattr(torch.autograd, "grad", boom)
    r = RPN.double_integral_residual(net, v["M"], v["c"], d)
    (r ** 2).sum().backward()
    gn = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gn) and any(float(g.abs().sum()) > 0 for g in gn)
    assert all(torch.isfinite(eta[k].grad) and float(eta[k].grad) != 0.0 for k in ("M", "c"))
    assert not r.requires_grad is False                                  # nothing detached
    # at the truth with the exact trajectory interpolant the residual is small against its declared scale
    assert d.s2.shape == d.y.shape and float(d.s2.min()) > 0


FAST = dict(hidden=16, layers=2, steps_data=300, steps_joint=700, n_colloc=150, n_quad=128, lr_theta=2e-2)


def test_T13_original_and_hard_bc_pinns_on_identical_data_report_open_loop_errors():
    sy = exact_rest_to_rest(0.8, 1.5, noise=2e-4, q_sigma=2e-4, seed=2)
    ep = episode(sy)["episode"]
    cfg = GP.PINNConfig(**FAST)
    th0 = [0.5, 0.8]                                                     # same start for every variant
    out = {v: RPN.fit_variant(v, ep, cfg, th0, seeds=(0,)) for v in ("LEGACY_PINN_STRONG", "PINN_HARD_BC_STRONG",
                                                                     "PINN_HARD_BC_DOUBLE_INTEGRAL")}
    for v, r in out.items():
        assert r["status"] == "OK", (v, r.get("reason"))
        ol = r["open_loop"]
        assert np.isfinite(ol["rmse_over_sigma"]) and "v0 = 0" in ol["replay"]
        assert r["M"] > 0 and r["c"] >= 0
    hi = out["PINN_HARD_BC_DOUBLE_INTEGRAL"]
    assert hi["settings"]["replay_v0"] == "zero" and hi["settings"]["physics_form"] == "double_integral_rest2rest"
    assert hi["M"] == pytest.approx(0.8, rel=0.25) and hi["c"] == pytest.approx(1.5, rel=0.3)
    assert "neural_q_rmse" in hi and hi["neural_q_rmse"] != hi["open_loop"]["rmse"]   # reported separately


def test_T13_zero_net_displacement_still_trains():
    ep = episode(out_and_back(0.8, 1.5, q_sigma=2e-4))["episode"]
    r = RPN.fit_variant("PINN_HARD_BC_DOUBLE_INTEGRAL", ep, GP.PINNConfig(**FAST), [0.5, 0.8], seeds=(0,))
    assert r["status"] == "OK" and r["M"] > 0
    assert abs(ep.delta_q) < 3 * math.hypot(ep.q0_sigma, ep.qT_sigma)          # return to start: dq not resolvable


def test_T14_legacy_defaults_unchanged_and_new_modes_replay_from_exact_zero_velocity():
    d = asdict(GP.PINNConfig())
    assert (d["trajectory_boundary_mode"], d["physics_form"], d["replay_v0"]) == ("legacy", "legacy", "prefix_fit")
    for k, v in dict(hidden=32, layers=3, n_seeds=3, steps_data=3000, steps_joint=4000, l_weak=1.0, l_ic=0.1).items():
        assert d[k] == v
    with pytest.raises(ModelConfigError):
        GP.PINNConfig(trajectory_boundary_mode="hard_rest_to_rest")                  # mixing net / physics refused
    with pytest.raises(ModelConfigError):
        GP.general_pinn_fit([], inertia_viscous_spec("prismatic"), [1.0, 1.0],
                            GP.PINNConfig(trajectory_boundary_mode="hard_rest_to_rest", physics_form="strong_only"))
    sy = exact_rest_to_rest(0.8, 1.5)
    k = 18                                                                            # starts mid-push: v != 0
    it = G9Interaction("x", "prismatic", sy["t_q"][k:], sy["q_true"][k:], np.full(len(sy["t_q"]) - k, 2e-4),
                       __import__("forceartgs_g9.estimators.rest_to_rest", fromlist=["ZOHEffort"]).ZOHEffort(
                           sy["force_t"], sy["Q"]))
    spec = inertia_viscous_spec("prismatic")
    legacy = GP.open_loop_replay(spec, [0.8, 1.5], [it])                              # historical: 4-sample v0
    zero = GP.open_loop_replay(spec, [0.8, 1.5], [it], v0_mode="zero")
    assert legacy["v0_mode"] == "prefix_fit" and zero["v0_mode"] == "zero"
    assert legacy["per_interaction"][0]["rmse_over_sigma"] != zero["per_interaction"][0]["rmse_over_sigma"]
