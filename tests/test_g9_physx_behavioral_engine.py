"""G10-A behavioral replay WITH the actual engine (SAPIEN 3 / PhysX CPU via the worker; skipped without SAPIEN):
T1 engine readback of (q0 != 0, v0 = 0 exactly), T2 q0 is never zeroed (tilted hinge), T5 data-consistent PhysX fit
(prismatic + revolute) from a wrong prior with a near-optimal family, T6 zero-input rest demo -> best candidate flagged
non-unique, T7 short pulse from rest matches the discrete engine law and a sign reversal fails the audit, T8 gravity
sensitivity (vertical / tilted hinge, horizontal / tilted slider) + labelled zero-gravity alternative, T10 surrogate
semantics / limit refusal, T15 near-equal demo fits under a new input."""
import importlib.util
import math

import numpy as np
import pytest

from forceartgs_g9.dynamics.general_1dof import RigidBodyGravity
from forceartgs_g9.physx_refinement import behavioral_adapter as BA
from forceartgs_g9.physx_refinement import behavioral_fit as BF
from forceartgs_g9.physx_refinement.behavioral_contracts import load_config
from forceartgs_g9.physx_refinement.behavioral_real import starts_for
from forceartgs_g9.physx_refinement.objective import Episode
from forceartgs_g9.physx_refinement.sapien_diagnostic import IsolatedSapienCpu, default_engine_python
from forceartgs_g9.physx_refinement.urdf_builder import write_single_joint_urdf

from g10_behavioral_fixtures import synthetic_demo

HAVE_ENGINE = importlib.util.find_spec("sapien") is not None or default_engine_python() is not None
pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="no SAPIEN interpreter")

G = [0.0, 0.0, -9.81]
J_COM = np.array([[0.05, 0.01, 0.0], [0.01, 0.04, 0.002], [0.0, 0.002, 0.03]])
O = np.array([0.1, 0.2, 0.3])
COM = np.array([0.5, 0.3, 0.4])
PH = [(10, 0.0, "absent"), (40, 3.0, "sliding"), (15, 0.02, "contact"), (85, 0.0, "absent")]
ROWS = 150
CFG = load_config({"fitting": {"maxiter": 4, "popsize": 4, "polish": False, "local_max_nfev": 150, "local_starts": 2,
                               "profile_points": 3, "profile_inner_max_nfev": 25, "profile_parameters": ["rho_scale"],
                               "family_tolerance_sigma": 1.0},
                   "alternatives": {"maxiter": 2, "popsize": 4, "local_max_nfev": 40}})


def tilted(deg):
    t = math.radians(deg)
    return np.array([math.sin(t), 0.0, math.cos(t)])


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("g10_behavioral_engine")


def make(tmp, name, *, axis, jt="revolute", mass=2.0, mode="external_effort_surrogate", g=G, limit=100.0):
    info = write_single_joint_urdf(tmp / f"{name}.urdf", joint_type=jt, axis=axis, origin=O, mass=mass, com_ref=COM,
                                   inertia_com_ref=J_COM, q_ref=0.0, prismatic_limit_m=limit)
    sim = IsolatedSapienCpu(info["path"], moving_link="moving_link", joint_name="joint", dt=1e-3, gravity=g,
                            friction_mode=mode)
    return info, sim


def demo(sim, truth, *, jt, q0, noise=2e-4, seed=0, phases=PH, rows=ROWS):
    rec, q_true = synthetic_demo(sim, truth, q0=q0, rows=rows, phases=phases, noise=noise, seed=seed)
    rec.joint_type = jt
    out = BA.build_full_demo_episode(rec, CFG)
    assert out["status"] == "OK", out["reason"]
    return rec, out["episode"], q_true


def P(rho, c=0.0, Td=0.0):
    return BF.params_from(dict(rho_scale=rho, armature=0.0, dynamic_effort=Td, static_excess=0.0, viscous=c))


# ------------------------------------------------------------------------------------------------ T1 / T2
def test_engine_starts_at_observed_q0_with_exactly_zero_velocity(tmp):
    for jt, ax, q0 in (("revolute", tilted(30), 0.4), ("prismatic", tilted(10), 0.05)):
        info, sim = make(tmp, f"t1_{jt}", axis=ax, jt=jt)
        try:
            _, ep, _ = demo(sim, P(1.0, 0.5), jt=jt, q0=q0)
            pe = ep.to_physx_episode()
            assert sim.request(P(1.3, 0.2), pe)["q0"] == q0 and sim.request(P(1.3, 0.2), pe)["v0"] == 0.0
            st = BF.verify_engine_start(sim(P(1.3, 0.2), pe), pe)
            assert st["exact"] and st["v0_engine"] == 0.0 and st["q0_engine"] == float(np.float32(q0))
        finally:
            sim.close()


def test_nonzero_q0_matters_on_a_tilted_hinge_and_is_kept(tmp):
    info, sim = make(tmp, "t2", axis=tilted(30))
    try:
        rec, ep, _ = demo(sim, P(1.0, 0.3), jt="revolute", q0=0.4)
        pe = ep.to_physx_episode()
        q_kept = np.asarray(sim(P(1.0, 0.3), pe).q)
        zeroed = Episode("z", pe.t_s, pe.q, pe.sigma_q, 0.0, 0.0, pe.input_t_s, pe.input_effort, "forced",
                         "VERIFIED")
        q_zero = np.asarray(sim(P(1.0, 0.3), zeroed).q)
        d_kept, d_zero = q_kept - q_kept[0], q_zero - q_zero[0]
        assert np.max(np.abs(d_kept - d_zero)) > 50 * 2e-4                 # gravity torque depends on q
        assert ep.q0_observed == 0.4 and pe.q0 == 0.4
    finally:
        sim.close()


# ------------------------------------------------------------------------------------------------ T5 / T15
@pytest.fixture(scope="module", params=[("prismatic", 88.0), ("revolute", 25.0)], ids=["prismatic", "revolute"])
def fitted(tmp, request):
    jt, deg = request.param
    info, sim = make(tmp, f"t5_{jt}", axis=tilted(deg), jt=jt)
    truth = P(1.5, 2.0, 0.3)
    rec, ep, q_true = demo(sim, truth, jt=jt, q0=0.1 if jt == "revolute" else 0.02, seed=4)
    roll = BF.CachingRollout(sim)
    b = BF.bounds_for(CFG["model"], info["axis_inertia"], 3.0)
    fit = BF.fit_full_demo(roll, ep, CFG, b, starts_for(b, info["axis_inertia"]))
    fam = BF.profile_family(roll, fit, CFG, b)
    yield dict(jt=jt, sim=sim, roll=roll, ep=ep, truth=truth, fit=fit, fam=fam, q_true=q_true, rec=rec)
    sim.close()


def test_physx_fit_recovers_the_demo_from_a_wrong_prior(fitted):
    fit, ep = fitted["fit"], fitted["ep"]
    pe = ep.to_physx_episode()
    truth_m = BF.replay_metrics(pe, fitted["roll"](fitted["truth"], pe), 3e-4)
    best = fit["best"]["metrics"]
    assert fit["best"]["runtime_semantics"] == "EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE"
    assert fit["improvement"]["rmse_ratio"] < 0.2                          # vs the frictionless prior reference
    assert best["rmse_over_sigma"] <= max(3.0, 1.5 * truth_m["rmse_over_sigma"])
    assert fitted["fam"]["n_members"] >= 1 and "rho_scale" in fitted["fam"]["profiles"]


def test_near_equal_demo_fits_diverge_under_a_new_input(fitted):
    pts = [p for p in fitted["fam"]["profiles"]["rho_scale"]["points"] if "metrics" in p]
    members = [dict(fitted["fit"]["best"], label="best")] + sorted(pts, key=lambda p: p["objective"])[:2]
    sp = BF.policy_spread(fitted["roll"], fitted["ep"], members, CFG)
    assert sp["status"] == "EVALUATED" and sp["max_q_spread"] > 0
    assert sp["probe"]["semantics"] == "SYNTHETIC_PROBE_NOT_A_REAL_DEMONSTRATION"


# ------------------------------------------------------------------------------------------------ T6
def test_zero_input_rest_demo_gives_a_candidate_flagged_non_unique(tmp):
    info, sim = make(tmp, "t6", axis=(0, 0, 1))                            # vertical hinge: zero gravity torque
    try:
        rec, ep, _ = demo(sim, P(1.0), jt="revolute", q0=0.2, phases=[(ROWS, 0.0, "absent")])
        roll = BF.CachingRollout(sim)
        b = BF.bounds_for(CFG["model"], info["axis_inertia"], 1.0)
        fit = BF.fit_full_demo(roll, ep, CFG, b, starts_for(b, info["axis_inertia"]))
        fam = BF.profile_family(roll, fit, CFG, b)
        assert fam["not_unique"] and fam["profiles"]["rho_scale"]["wide"]
        rb = fit["_result"].diagnostics
        assert rb["axis_inertia_engine"] > 0 and rb["readback"]["moving_mass"] > 0
    finally:
        sim.close()


# ------------------------------------------------------------------------------------------------ T7
def test_short_pulse_from_rest_matches_the_discrete_engine_law(tmp):
    info, sim = make(tmp, "t7", axis=(0, 0, 1), mode="frictionless_physx")
    try:
        A, k = 0.8, 5                                                      # 5 rows x 20 ms
        rec, ep, q_true = demo(sim, P(1.0), jt="revolute", q0=0.3, noise=1e-5,
                               phases=[(10, 0.0, "absent"), (k, A, "sliding"), (ROWS - 10 - k, 0.0, "absent")])
        I = info["axis_inertia"]
        res = sim(P(1.0), ep.to_physx_episode())
        v_end = float(np.asarray(res.qdot)[-1])
        assert v_end == pytest.approx(A * k * 0.02 / I, rel=2e-3)          # delta v = J / I, from v0 = 0
        assert float(np.asarray(res.qdot)[0]) == 0.0
        assert BF.work_audit(ep)["verdict"] == "PASSIVE_ENERGY_CONSISTENT"
        from forceartgs_g9.physx_refinement.behavioral_contracts import FullDemoBehavioralEpisode
        d = {f: getattr(ep, f) for f in ep.__dataclass_fields__}
        rev = FullDemoBehavioralEpisode(**dict(d, effort_object=-ep.effort_object))
        assert BF.work_audit(rev)["verdict"] == "INPUT_OUTPUT_SIGN_INCONSISTENT"
    finally:
        sim.close()


# ------------------------------------------------------------------------------------------------ T8
def _semi_implicit(q0, acc, t, dt=1e-3):
    n = np.rint((t - t[0]) / dt).astype(int)
    q, v, out, j = float(q0), 0.0, np.empty(len(t)), 0
    for i in range(n[-1] + 1):
        while j < len(t) and n[j] == i:
            out[j] = q
            j += 1
        if i == n[-1]:
            break
        v += acc(q) * dt
        q += v * dt
    return out


@pytest.mark.parametrize("case", ["vertical_hinge", "tilted_hinge", "horizontal_slider", "tilted_slider"])
def test_gravity_sensitivity_from_observed_q0_and_rest(tmp, case):
    jt = "revolute" if "hinge" in case else "prismatic"
    ax = {"vertical_hinge": (0, 0, 1), "tilted_hinge": tilted(35), "horizontal_slider": (1, 0, 0),
          "tilted_slider": tilted(80)}[case]
    info, sim = make(tmp, f"t8_{case}", axis=ax, jt=jt, mode="frictionless_physx")
    try:
        q0 = 0.35 if jt == "revolute" else 0.04
        t = np.arange(0, 60) / 50.0 * 2
        ep = Episode("g", t, np.zeros_like(t), np.ones_like(t), q0, 0.0, np.array([0.0]), np.array([0.0]), "free")
        r = sim(P(1.0), ep)
        I = info["axis_inertia"]
        rg = RigidBodyGravity(jt, tuple(np.asarray(ax, float) / np.linalg.norm(ax)), tuple(O), tuple(COM), 2.0,
                              tuple(G), 0.0)
        ref = _semi_implicit(q0, lambda q: -rg.G(q) / I, t)
        np.testing.assert_allclose(r.q, ref, atol=5e-5)
        assert float(np.asarray(r.qdot)[0]) == 0.0
        moved = float(np.ptp(r.q))
        if case in ("vertical_hinge", "horizontal_slider"):
            assert moved < 1e-6                                            # no gravity load: stays at q0
        else:
            assert moved > 1e-3
    finally:
        sim.close()


def test_zero_gravity_alternative_is_separately_labelled(tmp):
    info, sim = make(tmp, "t8_alt", axis=tilted(15), jt="prismatic")
    _, sim0 = make(tmp, "t8_alt0", axis=tilted(15), jt="prismatic", g=[0.0, 0.0, 0.0])
    try:
        rec, ep, _ = demo(sim, P(1.2, 2.0, 0.2), jt="prismatic", q0=0.0, seed=2)
        b = BF.bounds_for(CFG["model"], info["axis_inertia"], 3.0)
        alt = BF.zero_gravity_alternative(BF.CachingRollout(sim0), ep, CFG, b, P(1.0, 1.0))
        assert alt["status"] == "FITTED" and alt["label"] == "SEPARATELY_LABELLED_ALTERNATIVE_NOT_SELECTED"
    finally:
        sim.close()
        sim0.close()


# ------------------------------------------------------------------------------------------------ T10
def test_surrogate_semantics_capabilities_and_limit_refusal(tmp):
    info, sim = make(tmp, "t10", axis=(1, 0, 0), jt="prismatic", limit=0.01)
    try:
        cap = sim.capabilities()
        assert cap["native_axis_friction_api"] is False
        t = np.arange(0, 40) / 25.0
        ep = Episode("lim", t, np.zeros_like(t), np.ones_like(t), 0.0, 0.0, np.arange(0, 80) / 50.0,
                     np.full(80, 5.0), "forced", "VERIFIED")
        with pytest.raises(RuntimeError, match="joint limit"):
            sim(P(1.0, 0.5), ep)
        ok = Episode("ok", t, np.zeros_like(t), np.ones_like(t), 0.0, 0.0, np.arange(0, 80) / 50.0,
                     np.full(80, 1e-4), "forced", "VERIFIED")
        assert sim(P(1.0, 0.5, 0.1), ok).runtime_semantics == "EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE"
    finally:
        sim.close()


# ================================================================================ wrench-driven path (actual engine)
def _wrench_scene(tmp, name, n_rows, W):
    import json as _json
    meta = tmp / name / "scene" / "metadata"
    meta.mkdir(parents=True)
    (meta / "robot_tool.json").write_text(_json.dumps(dict(recorded_pose_frame="ee", wrench_frame=dict(
        axes="ee", R_ee_force_sensor=np.eye(3).tolist(), alignment_source="test"))))
    (meta / "contact_force_convention.json").write_text(_json.dumps(dict(
        object_side_force_multiplier=-1, recorded_quantity="contact_reaction_force_on_robot")))
    T = np.repeat(np.eye(4)[None], n_rows, 0)
    T[:, :3, 3] = [0.7, 0.4, 0.5]                                              # sensor origin, off the hinge line
    arc = tmp / name / "robot_archive.npz"
    np.savez(arc, wrench=W, T_base_ee=T, T_base_tcp=T, source_row=np.arange(n_rows))
    return tmp / name / "scene", arc


@pytest.mark.parametrize("jt,deg", [("revolute", 25.0), ("prismatic", 88.0)])
def test_wrench_driven_fit_improves_with_actual_physx(tmp, jt, deg):
    from forceartgs_g9.physx_refinement.behavioral_contracts import load_wrench_config, wrench_to_fit_config
    from forceartgs_g9.wrench_input import full_record as FR
    cfg = load_wrench_config({"fitting": {"maxiter": 4, "popsize": 4, "local_max_nfev": 150, "local_starts": 2,
                                          "profile_points": 3, "profile_inner_max_nfev": 20,
                                          "profile_parameters": ["rho_scale"]},
                              "physics": {"engine": "sapien_cpu_diagnostic"}})
    fc = wrench_to_fit_config(cfg)
    ax = tilted(deg)
    info, sim = make(tmp, f"wr_{jt}", axis=ax, jt=jt)
    try:
        n = 150
        W = np.zeros((n, 6))
        W[10:50, 3:] = -np.array([0.0, 0.9, 0.3])                             # moment (revolute only feels it)
        W[10:50, :3] = -np.array([2.5, 0.0, 0.4])                             # and a force (object side = -recorded)
        sc, arc = _wrench_scene(tmp, f"wr_{jt}", n, W)
        c = FR.load_contract(sc, arc)
        qf = np.arange(0, n - 1, 4)
        o = None if jt == "prismatic" else O
        fr0 = FR.build(c, scene="w", joint_type=jt, axis=ax, origin=o, q_src=qf, q=np.zeros(len(qf)),
                       q_sigma=np.full(len(qf), 2e-4))
        truth = P(1.5, 2.0, 0.3)
        q0 = 0.1 if jt == "revolute" else 0.02
        e = Episode("truth", qf / 50.0, np.zeros(len(qf)), np.ones(len(qf)), q0, 0.0, fr0.t_s, fr0.Q_full, "forced",
                    "VERIFIED")
        q_true = np.asarray(sim(truth, e).q)
        q_obs = q_true + np.random.default_rng(1).standard_normal(len(qf)) * 2e-4
        q_obs[0] = q0
        fr = FR.build(c, scene="w", joint_type=jt, axis=ax, origin=o, q_src=qf, q=q_obs, q_sigma=np.full(len(qf), 2e-4))
        assert np.max(np.abs(fr.Q_full[10:50])) > 0 and np.all(fr.Q_full[60:] == 0)
        out = BA.build_wrench_episode(fr, cfg, [(0, 60)])          # the interaction: contact from row 0, off after 60
        assert out["status"] == "OK"
        ep = out["episode"]
        roll = BF.CachingRollout(sim)
        b = BF.bounds_for(fc["model"], info["axis_inertia"], float(np.max(np.abs(ep.effort_object))))
        fit = BF.fit_full_demo(roll, ep, fc, b, starts_for(b, info["axis_inertia"]))
        pe = ep.to_physx_episode()
        truth_m = BF.replay_metrics(pe, roll(truth, pe), 3e-4)
        assert fit["improvement"]["rmse_ratio"] < 0.2
        assert fit["best"]["metrics"]["rmse_over_sigma"] <= max(3.0, 1.5 * truth_m["rmse_over_sigma"])
        assert fit["engine_initial_state"]["v0_engine"] == 0.0 and fit["engine_initial_state"]["q0_engine"] == \
            float(np.float32(q0))
    finally:
        sim.close()


def test_passive_terms_read_back_and_invariant_across_runs(tmp):
    from forceartgs_g9.physx_refinement.behavioral_wrench_real import passive_readback
    info, sim = make(tmp, "t21", axis=tilted(20))
    try:
        t = np.arange(0, 30) / 25.0
        e = Episode("r", t, np.zeros_like(t), np.ones_like(t), 0.2, 0.0, np.arange(0, 60) / 50.0, np.full(60, 0.3),
                    "forced", "VERIFIED")
        rbs = [passive_readback(sim(p, e)) for p in (P(1.0, 0.5, 0.1), P(3.0, 2.0, 0.4), P(1.0, 0.5, 0.1))]
        for rb in rbs:
            assert rb["joint_friction"] == 0.0 and rb["joint_damping"] == 0.0 and rb["sleep_threshold"] == 0.0
            assert all(v == 0.0 for v in rb["link_linear_damping"].values())
            assert all(v == 0.0 for v in rb["link_angular_damping"].values())
        import json as _json
        assert len({_json.dumps(rb, sort_keys=True) for rb in rbs}) == 1
    finally:
        sim.close()


# ================================================================================ PhysX GPU (batched)
def _have_gpu():
    import subprocess
    py = default_engine_python()
    if py is None:
        return False
    r = subprocess.run([py, "-c", "import torch; print(torch.cuda.is_available())"], capture_output=True, text=True)
    return r.stdout.strip() == "True"


gpu = pytest.mark.skipif(not _have_gpu(), reason="no CUDA in the engine interpreter")


def make_gpu(tmp, name, *, axis, jt="revolute", limit=100.0, mode="external_effort_surrogate"):
    from forceartgs_g9.physx_refinement.sapien_diagnostic import IsolatedSapienGpu
    info = write_single_joint_urdf(tmp / f"{name}.urdf", joint_type=jt, axis=axis, origin=O, mass=2.0, com_ref=COM,
                                   inertia_com_ref=J_COM, q_ref=0.0, prismatic_limit_m=limit)
    kw = dict(moving_link="moving_link", joint_name="joint", dt=1e-3, gravity=G, friction_mode=mode)
    return info, IsolatedSapienGpu(info["path"], **kw), IsolatedSapienCpu(info["path"], **kw)


@gpu
def test_gpu_batch_matches_cpu_physx(tmp):
    info, g, c = make_gpu(tmp, "gpu_par", axis=tilted(25))
    try:
        t = np.arange(0, 60) / 25.0
        ft = np.arange(0, 120) / 50.0
        e = Episode("p", t, np.zeros_like(t), np.ones_like(t), 0.3, 0.0, ft, np.where(ft < 1.0, 0.6, 0.0), "forced",
                    "VERIFIED")
        ths = [P(1.0, 0.5, 0.1), P(2.0, 1.0, 0.2), P(0.5, 0.2, 0.0), P(3.0, 0.0, 0.4)]
        rg = g.batch(ths, e)
        for th, r in zip(ths, rg):
            rc = c(th, e)
            assert np.max(np.abs(r.q - rc.q)) < 5e-6 and np.max(np.abs(r.qdot - rc.qdot)) < 2e-5
            assert r.runtime_semantics == "EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE" and r.backend == "sapien_physx_gpu"
            assert r.qdot[0] == 0.0 and r.q[0] == float(np.float32(0.3))
            assert r.diagnostics["readback"]["n_collision_shapes"] == 0
        single = g(ths[1], e)
        np.testing.assert_allclose(single.q, rg[1].q, atol=1e-9)              # batch of one == member of a batch
    finally:
        g.close()
        c.close()


@gpu
def test_gpu_batch_isolates_a_failing_candidate(tmp):
    info, g, c = make_gpu(tmp, "gpu_lim", axis=(1, 0, 0), jt="prismatic", limit=0.01)
    try:
        t = np.arange(0, 40) / 25.0
        ft = np.arange(0, 80) / 50.0
        e = Episode("l", t, np.zeros_like(t), np.ones_like(t), 0.0, 0.0, ft, np.full(80, 2.0), "forced", "VERIFIED")
        res = g.batch([P(1.0, 500.0, 0.0), P(1.0, 0.0, 0.0)], e)              # damped: stays inside; free: hits limit
        assert not isinstance(res[0], Exception)
        assert isinstance(res[1], RuntimeError) and "joint limit" in str(res[1])
    finally:
        g.close()
        c.close()


@gpu
def test_gpu_vectorized_fit_improves_and_reads_back_clean(tmp):
    from forceartgs_g9.physx_refinement.behavioral_wrench_real import passive_readback
    info, g, c = make_gpu(tmp, "gpu_fit", axis=tilted(25))
    try:
        truth = P(1.5, 2.0, 0.3)
        rec, ep, q_true = demo(c, truth, jt="revolute", q0=0.1, seed=4)
        roll = BF.CachingRollout(g)
        b = BF.bounds_for(CFG["model"], info["axis_inertia"], 3.0)
        fit = BF.fit_full_demo(roll, ep, CFG, b, starts_for(b, info["axis_inertia"]))
        pe = ep.to_physx_episode()
        truth_m = BF.replay_metrics(pe, roll(truth, pe), 3e-4)
        assert fit["improvement"]["rmse_ratio"] < 0.2
        assert fit["best"]["metrics"]["rmse_over_sigma"] <= max(3.0, 1.5 * truth_m["rmse_over_sigma"])
        assert fit["best"]["runtime_semantics"] == "EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE"
        assert roll.n_batches > 0 and fit["engine_initial_state"]["v0_engine"] == 0.0
        rbs = [passive_readback(roll(p, pe)) for p in (fit["_params"], truth)]
        assert rbs[0] == rbs[1] and rbs[0]["joint_friction"] == 0.0
    finally:
        g.close()
        c.close()
