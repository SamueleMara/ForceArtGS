"""G10 acceptance tests WITH the actual engine (SAPIEN 3 / PhysX CPU, via the worker in the engine interpreter).

Skipped when no SAPIEN interpreter exists (this interpreter or $PARTTWIN1D_PYTHON / parttwin_sim). CPU only.
Covers the spec's engine list that the diagnostic runner can honour: frictionless inertia under known pulses, input
sign / time / impulse, vertical hinge (zero gravity torque, nonzero inertia), nonvertical gravity projection against
the G9 RigidBodyGravity law (q_ref != 0), prismatic gravity, free decay at several normalized ratios, the free-motion
scale gauge, inertia readback under scaling, repeatability, refusals, and end-to-end recovery + scale profile.
NOT covered (no runtime): native PhysX friction, static breakaway, CPU/GPU agreement.
"""
import importlib.util
import json
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from forceartgs_g9.dynamics.general_1dof import RigidBodyGravity
from forceartgs_g9.physx_refinement import identifiability as ID
from forceartgs_g9.physx_refinement.api import estimate, run_forceartgs_physx_estimator
from forceartgs_g9.physx_refinement.objective import Episode, PhysicalParameters, effective_joint_inertia
from forceartgs_g9.physx_refinement.sapien_diagnostic import IsolatedSapienCpu, default_engine_python
from forceartgs_g9.physx_refinement.urdf_builder import write_single_joint_urdf

HAVE_ENGINE = importlib.util.find_spec("sapien") is not None or default_engine_python() is not None
pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="no SAPIEN interpreter")

G = [0.0, 0.0, -9.81]
J_COM = np.array([[0.05, 0.01, 0.0], [0.01, 0.04, 0.002], [0.0, 0.002, 0.03]])
O = np.array([0.1, 0.2, 0.3])
COM = np.array([0.5, 0.3, 0.4])


def tilted(deg):
    t = math.radians(deg)
    return np.array([math.sin(t), 0.0, math.cos(t)])


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("g10_engine")


DT = 1e-3


def symplectic(q0, v0, acc, t, dt=DT):
    """PhysX articulation time stepping (semi-implicit Euler: v += a dt, then q += v dt), float64."""
    n = np.rint((t - t[0]) / dt).astype(int)
    q, v, out = float(q0), float(v0), np.empty(len(t))
    k = 0
    for i in range(n[-1] + 1):
        while k < len(t) and n[k] == i:
            out[k] = q
            k += 1
        if i == n[-1]:
            break
        v += acc(t[0] + i * dt, q) * dt
        q += v * dt
    return out


def make(tmp, name, *, axis=(0, 0, 1), jt="revolute", q_ref=0.7, mass=2.0, mode="frictionless_physx", dt=1e-3):
    info = write_single_joint_urdf(tmp / f"{name}.urdf", joint_type=jt, axis=axis, origin=O, mass=mass, com_ref=COM,
                                   inertia_com_ref=J_COM, q_ref=q_ref)
    sim = IsolatedSapienCpu(info["path"], moving_link="moving_link", joint_name="joint", dt=dt, gravity=G,
                            friction_mode=mode)
    return info, sim


def ep(t, *, q0=0.0, v0=0.0, mode="free", ft=None, fq=None, q=None, train=True, name="e"):
    q = np.zeros_like(t) if q is None else q
    if mode == "free":
        return Episode(name, t, q, np.full_like(t, 1e-3), q0, v0, np.array([t[0]]), np.array([0.0]), "free",
                       used_for_training=train)
    return Episode(name, t, q, np.full_like(t, 1e-3), q0, v0, ft, fq, "forced", "VERIFIED", used_for_training=train)


P0 = PhysicalParameters(1.0, 0.0, 0.0, 0.0, 0.0)


def test_hidden_engine_damping_is_zeroed_and_read_back(tmp):
    info, sim = make(tmp, "vert")
    try:
        t = np.linspace(0, 3, 31)
        r = sim(P0, ep(t, v0=1.0))
        np.testing.assert_allclose(r.qdot, 1.0, atol=1e-5)          # vertical hinge: no gravity torque, no loss
        # PhysX keeps q in float32: 3000 increments of ~1e-3 rad at |q| ~ 3 accumulate ~3e-4 relative (measured 8e-4
        # rad) -- the reason the real-data loss carries sigma_num (numeric_floor_q)
        np.testing.assert_allclose(r.q, t, atol=1.5e-3)
        d = r.diagnostics["engine_defaults_as_loaded"]
        assert d["link_angular_damping"]["moving_link"] == pytest.approx(0.05)   # the trap this runner removes
        assert d["joint_friction"] == pytest.approx(0.05)
        rb = r.diagnostics["readback"]
        assert rb["joint_friction"] == 0 and rb["sleep_threshold"] == 0
        assert all(v == 0 for v in rb["link_angular_damping"].values())
        I_spec = effective_joint_inertia(joint_type="revolute", axis=[0, 0, 1], hinge_origin=O, com=COM, unit_mass=2.0,
                                         unit_inertia_com=J_COM, rho_scale=1.0, armature=0.0)
        assert info["axis_inertia"] == pytest.approx(I_spec, rel=1e-12)
        assert r.diagnostics["axis_inertia_engine"] == pytest.approx(I_spec, rel=1e-5)
    finally:
        sim.close()


@pytest.mark.parametrize("rho,arm", [(1.0, 0.0), (2.5, 0.0), (1.0, 0.2)])
def test_known_pulse_gives_impulse_over_inertia(tmp, rho, arm):
    info, sim = make(tmp, "pulse")
    try:
        t = np.linspace(0, 1, 21)
        r = sim(PhysicalParameters(rho, arm, 0, 0, 0), ep(t, mode="forced", ft=np.array([0.0, 0.2]),
                                                          fq=np.array([0.5, 0.0])))
        I = rho * info["axis_inertia"] + arm
        assert r.diagnostics["external_impulse"][-1] == pytest.approx(0.1, rel=1e-9)
        assert r.qdot[-1] == pytest.approx(0.1 / I, rel=2e-4)
        assert r.diagnostics["axis_inertia_engine"] == pytest.approx(rho * info["axis_inertia"], rel=1e-5)
    finally:
        sim.close()


@pytest.mark.parametrize("sgn", [1.0, -1.0])
def test_input_sign_and_zoh_timing(tmp, sgn):
    info, sim = make(tmp, "step")
    try:
        t = np.round(np.arange(0, 0.801, 0.05), 6)
        t1, Q = 0.3, 0.4 * sgn
        r = sim(P0, ep(t, q0=0.2, mode="forced", ft=np.array([0.0, t1]), fq=np.array([0.0, Q])))
        I = info["axis_inertia"]
        before = t <= t1 + 1e-12
        np.testing.assert_allclose(r.q[before], 0.2, atol=1e-7)       # nothing moves before the input starts
        disc = symplectic(0.2, 0.0, lambda tt, _: Q / I if tt >= t1 - 1e-12 else 0.0, t)
        np.testing.assert_allclose(r.q, disc, atol=2e-6)              # the engine's own discrete law
        cont = 0.2 + 0.5 * Q / I * np.clip(t - t1, 0, None) ** 2      # continuous law: O(a dt t) offset
        assert np.max(np.abs(r.q - cont)) <= 0.5 * abs(Q / I) * DT * (t[-1] - t1) * 1.05 + 1e-6
        assert np.sign(r.qdot[-1]) == sgn
    finally:
        sim.close()


@pytest.mark.parametrize("deg,q_ref", [(20.0, 0.7), (35.0, -0.4)])
def test_nonvertical_gravity_matches_g9_rigid_body_gravity(tmp, deg, q_ref):
    a = tilted(deg)
    info, sim = make(tmp, f"tilt{int(deg)}", axis=a, q_ref=q_ref)
    try:
        t = np.linspace(0, 1.5, 61)
        q0 = 0.3
        r = sim(P0, ep(t, q0=q0))
        rg = RigidBodyGravity("revolute", tuple(a), tuple(O), tuple(COM), 2.0, tuple(G), q_ref)
        I = info["axis_inertia"]
        sol = solve_ivp(lambda _, y: [y[1], -rg.G(y[0]) / I], (0, 1.5), [q0, 0.0], t_eval=t, rtol=1e-10, atol=1e-12,
                        max_step=1e-3)
        assert np.ptp(r.q) > 0.1                                        # gravity actually drives it
        np.testing.assert_allclose(r.q, symplectic(q0, 0.0, lambda _, qq: -rg.G(qq) / I, t), atol=5e-5)
        np.testing.assert_allclose(r.q, sol.y[0], atol=1e-2)            # continuous ODE: O(dt) discretisation
    finally:
        sim.close()


def test_vertical_hinge_zero_gravity_torque_nonzero_inertia(tmp):
    info, sim = make(tmp, "vert2", q_ref=1.3)
    try:
        t = np.linspace(0, 1, 11)
        r = sim(P0, ep(t, q0=1.0))
        np.testing.assert_allclose(r.q, 1.0, atol=1e-6)                 # no gravity torque about a vertical axis
        r2 = sim(P0, ep(t, q0=1.0, mode="forced", ft=np.array([0.0]), fq=np.array([0.2])))
        assert r2.qdot[-1] == pytest.approx(0.2 / info["axis_inertia"], rel=2e-4)   # ... but inertia is there
    finally:
        sim.close()


def test_prismatic_gravity_projection(tmp):
    a = tilted(30.0)
    info, sim = make(tmp, "pri", axis=a, jt="prismatic")
    try:
        t = np.linspace(0, 0.8, 17)
        r = sim(P0, ep(t, q0=0.05))
        g_a = float(np.dot(G, a))
        n = np.rint(t / DT)
        np.testing.assert_allclose(r.q, 0.05 + 0.5 * g_a * DT ** 2 * n * (n + 1), atol=2e-5)   # discrete law
        assert np.max(np.abs(r.q - (0.05 + 0.5 * g_a * t ** 2))) <= 0.5 * abs(g_a) * DT * t[-1] * 1.05
        assert r.diagnostics["axis_inertia_engine"] == pytest.approx(2.0, rel=1e-6)
    finally:
        sim.close()


def _decay_ode(lam, gam, v0, t):
    def f(_, y):
        v = y[1]
        return [v, -lam * v - (gam * np.sign(v) if abs(v) > 1e-9 else 0.0)]
    stop = lambda _, y: y[1]
    stop.terminal, stop.direction = True, -1
    sol = solve_ivp(f, (t[0], t[-1]), [0.0, v0], events=stop, dense_output=True, rtol=1e-10, atol=1e-12, max_step=1e-3)
    ts = sol.t[-1]
    return np.where(t <= ts, sol.sol(np.minimum(t, ts))[0], sol.y[0][-1])


@pytest.mark.parametrize("lam,gam", [(0.0, 0.5), (1.5, 0.0), (0.8, 0.3), (3.0, 1.0)])
def test_free_decay_ratios_and_scale_gauge(tmp, lam, gam):
    info, sim = make(tmp, "decay", mode="external_effort_surrogate")
    try:
        t = np.linspace(0, 3, 61)
        I = info["axis_inertia"]
        e = ep(t, v0=0.6)
        r1 = sim(PhysicalParameters(1.0, 0, gam * I, gam * I, lam * I), e)
        np.testing.assert_allclose(r1.q, _decay_ode(lam, gam, 0.6, t), atol=2e-3)
        # gauge: (rho, c, Td) and (k rho, k c, k Td) give the same free motion in the ENGINE
        r2 = sim(PhysicalParameters(2.7, 0, 2.7 * gam * I, 2.7 * gam * I, 2.7 * lam * I), e)
        np.testing.assert_allclose(r1.q, r2.q, atol=1e-5)
    finally:
        sim.close()


def test_repeatable_and_refusals(tmp):
    info, sim = make(tmp, "rep", mode="external_effort_surrogate")
    try:
        t = np.linspace(0, 1, 21)
        e = ep(t, v0=0.4)
        p = PhysicalParameters(1.0, 0, 0.05, 0.05, 0.1)
        assert np.array_equal(sim(p, e).q, sim(p, e).q)
        with pytest.raises(ValueError, match="static breakaway"):
            sim(PhysicalParameters(1.0, 0, 0.1, 0.05, 0.1), e)
    finally:
        sim.close()
    _, sim2 = make(tmp, "fl")
    try:
        with pytest.raises(ValueError, match="frictionless"):
            sim2(PhysicalParameters(1.0, 0, 0.05, 0.05, 0.0), ep(np.linspace(0, 1, 5)))
    finally:
        sim2.close()


TRUE = dict(rho=1.6, lam=0.9, gam=0.35)
B = dict(rho_scale=(0.4, 4.0), armature=(0.0, 0.0), dynamic_effort=(0.0, 1.0), static_excess=(0.0, 0.0),
         viscous=(0.0, 2.0))
PHYS = dict(engine="sapien_cpu_diagnostic", joint_type="revolute", moving_link="moving_link", joint_name="joint",
            timestep=1e-3, gravity=G, friction_mode="external_effort_surrogate")


def _truth(info):
    I = TRUE["rho"] * info["axis_inertia"]
    return PhysicalParameters(TRUE["rho"], 0.0, TRUE["gam"] * I, TRUE["gam"] * I, TRUE["lam"] * I)


def test_end_to_end_forced_recovery_profile_and_heldout(tmp):
    info, sim = make(tmp, "e2e", mode="external_effort_surrogate")
    try:
        t = np.round(np.arange(0, 2.0001, 0.05), 6)
        ft = np.array([0.0, 0.25, 0.6, 1.0, 1.3])
        fq = np.array([0.9, 0.0, -1.1, 0.0, 0.6])
        rng = np.random.default_rng(0)
        eps = []
        for k, (scale, train) in enumerate(((1.0, True), (0.7, False))):
            proto = ep(t, mode="forced", ft=ft, fq=scale * fq, name=f"push{k}", train=train)
            q = sim(_truth(info), proto).q + rng.normal(0, 1e-3, len(t))
            eps.append(ep(t, mode="forced", ft=ft, fq=scale * fq, q=q, name=f"push{k}", train=train))
        rep = estimate(sim, eps, physics=PHYS, loss=dict(huber_delta=2.0), prior_rho_range=(0.5, 3.0),
                       fitting=dict(bounds=B, seed=0, maxiter=25, popsize=6, polish=True),
                       profile=dict(n=5, inner_maxiter=8, inner_popsize=5))
        p, I = rep["params"], rep["normalized"]["I_eff"]
        assert p["rho_scale"] == pytest.approx(TRUE["rho"], rel=0.10)
        assert p["viscous"] / I == pytest.approx(TRUE["lam"], rel=0.15)
        assert p["dynamic_effort"] / I == pytest.approx(TRUE["gam"], rel=0.15)
        assert rep["model_adequacy"]["train_adequate"] and rep["model_adequacy"]["heldout_adequate"]
        assert rep["heldout"]["status"] == "EVALUATED_WITH_FROZEN_THETA"
        pr = rep["scale_profile"]
        assert pr["status"] == "SCALE_PROFILE_BOUNDED_CANDIDATE"
        assert pr["ci95"][0] <= TRUE["rho"] <= pr["ci95"][1]
        assert pr["ci95"][1] / pr["ci95"][0] < 2.0
        st = rep["status_table"]
        assert st["absolute_scale"] == "PROFILE_BOUNDED_CANDIDATE_NOT_CERTIFIED"
        assert st["overall"] == "DIAGNOSTIC_ONLY_NOT_IDENTIFIED"            # surrogate: never more
        assert rep["simulation_export_allowed"] is False and rep["downstream"]["point_estimate_allowed"] is False
    finally:
        sim.close()


def _no_value_key(x):
    if isinstance(x, dict):
        return "value" not in x and all(_no_value_key(v) for v in x.values())
    if isinstance(x, list):
        return all(_no_value_key(v) for v in x)
    return True


def test_free_only_gauge_family_and_api_outputs(tmp):
    from types import SimpleNamespace
    info, sim = make(tmp, "free", mode="external_effort_surrogate")
    t = np.round(np.arange(0, 2.0001, 0.05), 6)
    try:
        q = sim(_truth(info), ep(t, v0=0.8)).q + np.random.default_rng(1).normal(0, 1e-3, len(t))
    finally:
        sim.close()
    it = SimpleNamespace(name="coast", t_obs=t, q_obs=q, q_std=np.full(len(t), 1e-3),
                         effort=SimpleNamespace(t_s=t[:1], levels=np.zeros(1)), initial_state={"qdot": 0.8})
    bounds = dict(B, rho_scale=(1.0, 1.0))
    rep = run_forceartgs_physx_estimator(
        [it], {"coast": {"mode": "free", "evidence": {"source": "synthetic: no contact by construction",
                                                      "contact_independently_absent": True}}},
        {"coast": 0.8}, calibration_by_name={}, urdf=info["path"], physics=PHYS, loss=dict(huber_delta=2.0),
        fitting=dict(bounds=bounds, seed=0, maxiter=20, popsize=6), out_dir=tmp / "api_free",
        prior_rho_range=(0.5, 3.0), profile=dict(n=5))
    nz = rep["normalized"]
    assert nz["lam"] == pytest.approx(TRUE["lam"], rel=0.15) and nz["gam"] == pytest.approx(TRUE["gam"], rel=0.15)
    assert rep["scale_status"] == "ABSOLUTE_SCALE_UNIDENTIFIED_FREE_ONLY"
    assert rep["scale_profile"]["status"] == "NOT_RUN_FREE_ONLY_GAUGE"
    st = rep["status_table"]
    assert st["absolute_scale"] == "NOT_IDENTIFIED" and st["parameters"]["rho_scale"]["status"] == "FIXED_PRIOR"
    fam = rep["correlated_family"]["members"]
    assert len(fam) == 7
    for m in fam:                                                     # c/I and Td/I held across the family
        assert m["viscous"] / m["I_eff"] == pytest.approx(nz["lam"], rel=1e-9)
        assert m["dynamic_effort"] / m["I_eff"] == pytest.approx(nz["gam"], rel=1e-9)
    ok, why = ID.downstream_point_allowed(rep)
    assert not ok and "refused" in why
    assert ID.downstream_point_allowed(rep, explicit_prior_preview=True)[0]
    doc = json.loads((tmp / "api_free" / "physx_refinement.json").read_text())
    assert _no_value_key(doc) and doc["simulation_export_allowed"] is False
    assert json.loads((tmp / "api_free" / "physx_episodes.json").read_text())["schema"] == \
        "forceartgs.g9.physx_episodes/v1"
