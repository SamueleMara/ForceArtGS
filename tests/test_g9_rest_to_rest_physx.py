"""Rest-to-rest on ACTUAL SAPIEN PhysX (T15, T16, T19, T20, T22, T25). CPU PhysX for determinism; GPU parity when
CUDA is available in the engine interpreter."""
import importlib.util
import subprocess

import numpy as np
import pytest

from forceartgs_g9.estimators import rest_to_rest as RR
from forceartgs_g9.estimators import rest_to_rest_convolution as RC
from forceartgs_g9.estimators import rest_to_rest_integral as RI
from forceartgs_g9.physx_refinement import behavioral_fit as BF
from forceartgs_g9.physx_refinement.behavioral_contracts import load_wrench_config, wrench_to_fit_config
from forceartgs_g9.physx_refinement.behavioral_real import starts_for
from forceartgs_g9.physx_refinement.objective import RefinementLoss, _huber
from forceartgs_g9.physx_refinement.rest_to_rest_objective import TerminalVelocityLoss, factory
from forceartgs_g9.physx_refinement.sapien_diagnostic import IsolatedSapienCpu, default_engine_python
from forceartgs_g9.physx_refinement.urdf_builder import write_single_joint_urdf

from r2r_fixtures import episode, exact_rest_to_rest

HAVE = importlib.util.find_spec("sapien") is not None or default_engine_python() is not None
pytestmark = pytest.mark.skipif(not HAVE, reason="no SAPIEN interpreter")
SIMPLE = "configs/g9_real/physx_behavioral_inertia_damping.yaml"


def _fcfg():
    import yaml
    raw = yaml.safe_load(open(SIMPLE).read())
    raw["fitting"].update(maxiter=3, popsize=6, local_max_nfev=30, local_starts=1, collect_near_optimal_candidates=False)
    raw["diagnostics"]["policy_probe"] = False
    return wrench_to_fit_config(load_wrench_config(raw))


@pytest.fixture(scope="module", params=["prismatic", "revolute"])
def setup(request, tmp_path_factory):
    jt = request.param
    d = tmp_path_factory.mktemp(f"r2r_{jt}")
    axis = [1.0, 0.0, 0.0] if jt == "prismatic" else [0.0, 0.0, 1.0]
    info = write_single_joint_urdf(d / "ref.urdf", joint_type=jt, axis=axis, origin=[0, 0, 0], mass=2.0,
                                   com_ref=[0.4, 0.1, 0.0], inertia_com_ref=np.diag([0.02, 0.03, 0.04]))
    rho, M = 1.5, 1.5 * info["axis_inertia"]
    c = 1.8 * M
    sy = exact_rest_to_rest(M, c, q0=0.12 if jt == "prismatic" else -0.3, pushes=((0.3, 3.0 * M),), noise=2e-4,
                            q_sigma=2e-4, seed=1)
    ep = episode(sy, joint_type=jt)["episode"]
    sim = IsolatedSapienCpu(info["path"], moving_link=info["moving_link"], joint_name=info["joint_name"], dt=1e-3,
                            gravity=[0.0, 0.0, 0.0], friction_mode="external_effort_surrogate")
    yield dict(jt=jt, info=info, rho=rho, M=M, c=c, sy=sy, ep=ep, sim=sim)
    sim.close()


def _bounds(s):
    return BF.bounds_for(_fcfg()["model"], s["info"]["axis_inertia"], float(np.max(np.abs(s["ep"].force_Q))))


def test_T15_physx_starts_at_q0_zero_velocity_with_the_unmasked_native_input_and_reports_qdot_at_T(setup):
    s = setup
    ep, pe = s["ep"], s["ep"].to_physx_episode()
    p = BF.params_from(dict(rho_scale=s["rho"], armature=0.0, dynamic_effort=0.0, static_excess=0.0, viscous=s["c"]))
    req = s["sim"].request(p, pe)
    np.testing.assert_array_equal(np.asarray(req["effort_levels"]), ep.force_Q)        # no gating, every row
    assert req["v0"] == 0.0 and req["q0"] == ep.q0
    r = BF.CachingRollout(s["sim"])(p, pe)
    st = BF.verify_engine_start(r, pe)
    assert st["exact"] and st["v0_engine"] == 0.0
    qd = np.asarray(r.qdot)
    ex = RC.replay(ep, s["M"], s["c"])
    assert len(qd) == len(ep.t_q_s)
    # qdot(T) is the ENGINE's own emergent value (never overwritten): it equals the exact model's v(T) -- which is not
    # zero here, because T was located at the q resolution 20 ms before the exact stop (see the endpoint evidence)
    assert qd[-1] == pytest.approx(ex["v"][-1], abs=0.01 * np.max(np.abs(qd)))
    assert abs(ex["v"][-1]) > 0 and ep.T < s["sy"]["T_true"]
    assert np.max(np.abs(np.asarray(r.q) - ex["q"])) < 0.01 * np.ptp(ex["q"])      # PhysX ~ exact (O(dt) offset)


def test_T16_T25_terminal_term_is_exactly_additive_and_V0_V1_report_the_same_metrics(setup):
    s = setup
    pe = s["ep"].to_physx_episode()
    roll = BF.CachingRollout(s["sim"])
    fc = _fcfg()
    st = BF.loss_settings(fc)
    p = BF.params_from(dict(rho_scale=1.2, armature=0.0, dynamic_effort=0.0, static_excess=0.0, viscous=s["c"] * 0.7))
    base = RefinementLoss(roll, [pe], st).evaluate(p)
    sv = RR.terminal_speed_sigma(s["ep"])["sigma_vT"]
    v0 = TerminalVelocityLoss(roll, [pe], st, weight=0.0, sigma_vT=sv).evaluate(p)
    assert v0["objective"] == base["objective"]                                       # lambda_T = 0: bitwise V0
    w = 0.7
    v1 = TerminalVelocityLoss(roll, [pe], st, weight=w, sigma_vT=sv).evaluate(p)
    vT = float(np.asarray(roll(p, pe).qdot)[-1])
    assert v1["objective"] - base["objective"] == pytest.approx(w * float(_huber(np.asarray(vT / sv), st.huber_delta)),
                                                                rel=1e-12)
    b = _bounds(s)
    starts = starts_for(b, s["info"]["axis_inertia"], None)
    f0 = BF.fit_full_demo(BF.CachingRollout(s["sim"]), s["ep"], fc, b, starts)
    f1 = BF.fit_full_demo(BF.CachingRollout(s["sim"]), s["ep"], fc, b, starts, objective_factory=factory(1.0, sv),
                          terminal=dict(weight=1.0, sigma=sv))
    assert f0["optimizer"]["nit"] == f1["optimizer"]["nit"] <= 3                      # identical DE budget
    for f in (f0, f1):                                                                # Pareto inputs reported for both
        r = BF.CachingRollout(s["sim"])(f["_params"], pe)
        assert np.isfinite(f["best"]["metrics"]["rmse_over_sigma"]) and np.isfinite(np.asarray(r.qdot)[-1])
    assert isinstance(f1["_objective"], TerminalVelocityLoss) and not isinstance(f0["_objective"], TerminalVelocityLoss)


def test_T19_analytic_M_maps_to_rho_and_the_engine_reads_back_exactly_M(setup):
    s = setup
    b = _bounds(s)
    mp = RI.map_to_physx(s["M"], s["c"], s["info"], b, label="truth")
    assert mp["status"] == "OK" and mp["params"]["rho_scale"] == pytest.approx(s["rho"], rel=1e-12)
    assert mp["params"]["viscous"] == s["c"] and mp["params"]["dynamic_effort"] == 0.0
    r = BF.CachingRollout(s["sim"])(mp["physical"], s["ep"].to_physx_episode())
    assert r.diagnostics["axis_inertia_engine"] == pytest.approx(s["M"], rel=1e-4)    # PhysX mass / inertia readback
    assert RI.map_to_physx(-1.0, 1.0, s["info"], b, label="x")["status"] == "REJECTED"
    big = RI.map_to_physx(1e6 * s["M"], 0.0, s["info"], b, label="y")
    assert big["status"] == "OK_CLIPPED" and big["clip_log"]["rho_scale"][0] > b["rho_scale"][1]


def test_T20_minimal_runs_use_the_explicit_viscous_surrogate_without_coulomb_gravity_or_armature(setup):
    s = setup
    b = _bounds(s)
    assert b["dynamic_effort"] == (0.0, 0.0) and b["static_excess"] == (0.0, 0.0) and b["armature"] == (0.0, 0.0)
    p = BF.params_from(dict(rho_scale=1.0, armature=0.0, dynamic_effort=0.0, static_excess=0.0, viscous=1.0))
    r = BF.CachingRollout(s["sim"])(p, s["ep"].to_physx_episode())
    assert r.runtime_semantics == "EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE"
    assert np.allclose(r.diagnostics["readback"]["gravity"], 0.0)


def test_T22_initializers_are_plausible_and_the_same_budget_fit_is_not_worse_than_its_start(setup):
    s = setup
    ep = s["ep"]
    svd = RI.integral_svd_r2r(ep)
    cv = RC.convolution_1d(ep)
    for M, c in ((svd["estimate"]["M"], svd["estimate"]["c"]), (cv["M"], cv["c"])):
        assert M == pytest.approx(s["M"], rel=0.1) and c == pytest.approx(s["c"], rel=0.15)
    b = _bounds(s)
    start = RI.map_to_physx(cv["M"], cv["c"], s["info"], b, label="conv")["physical"]
    f = BF.fit_full_demo(BF.CachingRollout(s["sim"]), ep, _fcfg(), b, {"conv": start})
    assert f["best"]["objective"] <= f["baselines"]["conv"]["objective"] + 1e-12


def _gpu():
    py = default_engine_python()
    return py is not None and subprocess.run([py, "-c", "import torch; print(torch.cuda.is_available())"],
                                             capture_output=True, text=True).stdout.strip() == "True"


@pytest.mark.skipif(not _gpu(), reason="no CUDA in the engine interpreter")
def test_T15_gpu_parity_on_the_same_rest_to_rest_episode(setup):
    from forceartgs_g9.physx_refinement.sapien_diagnostic import IsolatedSapienGpu
    s = setup
    info = s["info"]
    g = IsolatedSapienGpu(info["path"], moving_link=info["moving_link"], joint_name=info["joint_name"], dt=1e-3,
                          gravity=[0.0, 0.0, 0.0], friction_mode="external_effort_surrogate")
    try:
        p = BF.params_from(dict(rho_scale=s["rho"], armature=0.0, dynamic_effort=0.0, static_excess=0.0,
                                viscous=s["c"]))
        pe = s["ep"].to_physx_episode()
        rg, rc = g.batch([p], pe)[0], s["sim"](p, pe)
        assert np.max(np.abs(np.asarray(rg.q) - np.asarray(rc.q))) < 1e-4 * max(1.0, np.ptp(rc.q))
        assert float(np.asarray(rg.qdot)[0]) == 0.0
    finally:
        g.close()
