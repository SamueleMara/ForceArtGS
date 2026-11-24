"""Input-output identification -- analytical backbone (no engine): T1 transfer response, T2 scale law, T4 Coulomb stop,
T5 gravity / geometry and coordinate reversal."""
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from forceartgs_g9.dynamics.general_1dof import GeneralTerms, RigidBodyGravity, sim_params, simulate_general
from forceartgs_g9.synthetic import ZOH
from forceartgs_g9.system_identification import physical_response as PR


# ------------------------------------------------------------------------------------------------- T1
@pytest.mark.parametrize("jt,M,c,K", [("revolute", 0.7, 0.4, 0.0), ("prismatic", 3.0, 2.0, 0.0),
                                      ("revolute", 0.7, 0.4, 5.0), ("prismatic", 3.0, 0.5, 40.0),
                                      ("revolute", 0.7, 0.4, -2.0)])
def test_local_transfer_function_matches_ode(jt, M, c, K):
    m = PR.LocalLinearModel(M, c, K, jt)
    num, den = m.tf_q()
    assert num == [pytest.approx(1 / M)] and den == pytest.approx([1.0, c / M, K / M])
    p1, p2 = m.poles()
    # poles are the roots of the characteristic polynomial
    for p in (p1, p2):
        assert abs(M * p * p + c * p + K) < 1e-9
    if K == 0:
        assert {round(complex(p).real, 12) for p in (p1, p2)} == {0.0, round(-c / M, 12)}
        assert "MARGINAL" in PR.classify_poles(p1, p2)["stability"]
    if K < 0:
        assert max(complex(p).real for p in (p1, p2)) > 0           # unstable, not clamped
        assert "UNSTABLE" in PR.classify_poles(p1, p2)["stability"]
    if c * c < 4 * M * K:
        assert PR.classify_poles(p1, p2)["kind"] == "UNDERDAMPED_COMPLEX_PAIR"
    # frequency response at w equals the steady-state amplitude / phase of the ODE driven by cos(w t)
    if K > 0:
        w = 1.7
        T = max(80.0, 30.0 / (0.5 * c / M))                       # the transient has decayed by e^-30
        sol = solve_ivp(lambda t, x: [x[1], (math.cos(w * t) - c * x[1] - K * x[0]) / M], (0, T), [0, 0],
                        rtol=1e-10, atol=1e-12, dense_output=True)
        tt = np.linspace(T - 10, T, 2000)
        H = m.H(w)
        np.testing.assert_allclose(sol.sol(tt)[0], np.real(H * np.exp(1j * w * tt)), atol=2e-5 * abs(H) + 1e-7)
        Hv = m.H(w, "v")
        np.testing.assert_allclose(sol.sol(tt)[1], np.real(Hv * np.exp(1j * w * tt)), atol=2e-5 * abs(Hv) + 1e-7)


def test_coulomb_is_an_offset_not_a_derivative():
    G = lambda q: 0.3 * math.sin(q)                                 # noqa: E731
    dG = lambda q: 0.3 * math.cos(q)                                # noqa: E731
    a = PR.linearize(0.4, 0.8, M=0.5, c=0.2, Fc=0.7, G=G, dG=dG)
    b = PR.linearize(0.4, 0.8, M=0.5, c=0.2, Fc=0.0, G=G, dG=dG)
    assert a.B == b.B == 0.2 and a.K == b.K == pytest.approx(0.3 * math.cos(0.4))
    assert a.offset - b.offset == pytest.approx(0.7)
    assert PR.linearize(0.4, -0.8, M=0.5, c=0.2, Fc=0.7).offset == pytest.approx(-0.7)
    with pytest.raises(PR.PhysicalResponseError):
        PR.linearize(0.4, 0.0, M=0.5, c=0.2, Fc=0.7)                  # no TF at rest / stick


@pytest.mark.parametrize("c", [0.8, 1e-12, 0.0])
def test_constant_effort_transient_and_c_to_zero_limit(c):
    M, Fc, u0, G0, v0, q0 = 0.6, 0.2, 1.1, 0.3, 0.5, 0.1
    t = np.linspace(0, 2, 101)
    r = PR.sliding_transient(t, M=M, c=c, Fc=Fc, u0=u0, G0=G0, s_v=1, q0=q0, v0=v0)
    sol = solve_ivp(lambda tt, x: [x[1], (u0 - G0 - Fc - c * x[1]) / M], (0, 2), [q0, v0], t_eval=t, rtol=1e-11,
                    atol=1e-13)
    np.testing.assert_allclose(r["q"], sol.y[0], atol=1e-9)
    np.testing.assert_allclose(r["v"], sol.y[1], atol=1e-9)
    assert r["t_stop"] is None
    # the integrator used by every estimator agrees
    terms = GeneralTerms(inertia=True, viscous=True, coulomb=True, static=False, gravity=False, bias=False)
    ro = simulate_general(sim_params(terms, "revolute", M=M, c=c, Fc=Fc), ZOH([0.0], [u0 - G0]), t, q0=q0, v0=v0)
    np.testing.assert_allclose(ro.q, sol.y[0], atol=1e-9)


# ------------------------------------------------------------------------------------------------- T2 (law)
def test_free_motion_scale_invariance_of_the_dimensional_law():
    t = np.linspace(0, 1.5, 61)
    rbg = RigidBodyGravity("revolute", (0.2, 0.0, 0.98), (0, 0, 0), (0.4, 0.1, 0.0), 1.0, (0, 0, -9.81), 0.0)
    terms = GeneralTerms(inertia=True, viscous=True, coulomb=True, static=False, gravity=True, bias=False)
    base = None
    for k in (0.1, 1.0, 7.0):
        M, c, Fc = 0.5 * k, 0.3 * k, 0.1 * k
        ro = simulate_general(sim_params(terms, "revolute", M=M, c=c, Fc=Fc, G=lambda q, k=k: k * rbg.G(q)),
                              ZOH([0.0], [0.0]), t, q0=0.2, v0=1.4)
        base = ro.q if base is None else base
        np.testing.assert_allclose(ro.q, base, atol=1e-10)


# ------------------------------------------------------------------------------------------------- T4
def test_zero_velocity_crossing_terminates_without_chatter():
    t = np.linspace(0, 3, 301)
    r = PR.sliding_transient(t, M=0.5, c=0.3, Fc=0.4, u0=0.0, s_v=1, q0=0.0, v0=1.0)
    ts = r["t_stop"]
    assert ts is not None and 0 < ts < 3 and r["stuck_after_stop"]
    after = t >= ts
    assert np.all(r["v"][after] == 0.0) and np.ptp(r["q"][after]) == 0.0
    assert np.all(r["v"][~after] > 0)                              # no negative-velocity excursion
    # the hybrid integrator stops at the same time and stays stuck (|u| <= Fc)
    terms = GeneralTerms(inertia=True, viscous=True, coulomb=True, static=False, gravity=False, bias=False)
    ro = simulate_general(sim_params(terms, "revolute", M=0.5, c=0.3, Fc=0.4), ZOH([0.0], [0.0]), t, q0=0.0, v0=1.0,
                          record_events=True)
    stops = [e["t_s"] for e in ro.events if e["kind"] == "stop"]
    assert len(stops) == 1 and stops[0] == pytest.approx(ts, rel=1e-9)
    assert np.all(ro.qdot[after] == 0.0) and np.all(ro.qdot >= 0)
    # with an effort that would reverse the motion the analytic continuation is refused (NaN), not extrapolated
    r2 = PR.sliding_transient(t, M=0.5, c=0.3, Fc=0.1, u0=-1.0, s_v=1, q0=0.0, v0=1.0)
    assert r2["t_stop"] is not None and not r2["continues_after_stop"] and np.isnan(r2["q"][-1])


# ------------------------------------------------------------------------------------------------- T5
G = (0.0, 0.0, -9.81)


def test_vertical_revolute_zero_gravity_nonzero_inertia():
    rbg = PR.rigid_gravity("revolute", (0, 0, 1), (0.1, 0.2, 0.0), (0.5, 0.3, 0.4), 2.0, G, 0.3)
    Gq, dG = PR.gravity_load(rbg, np.linspace(-3, 3, 13))
    assert np.max(np.abs(Gq)) < 1e-12 and np.max(np.abs(dG)) < 1e-12
    J = np.diag([0.05, 0.04, 0.03])
    assert PR.hinge_inertia((0, 0, 1), (0.1, 0.2, 0.0), (0.5, 0.3, 0.4), 2.0, J) == pytest.approx(
        0.03 + 2.0 * (0.4 ** 2 + 0.1 ** 2))


@pytest.mark.parametrize("deg,q_ref", [(10, 0.0), (35, 0.8)])
def test_tilted_revolute_gravity_and_stiffness_match_potential(deg, q_ref):
    t = math.radians(deg)
    rbg = PR.rigid_gravity("revolute", (math.sin(t), 0, math.cos(t)), (0.1, 0.2, 0.3), (0.5, 0.3, 0.4), 2.0, G, q_ref)
    h = 1e-6
    for q in np.linspace(-2, 2, 9):
        Gq, dG = PR.gravity_load(rbg, q)
        assert Gq == pytest.approx((rbg.potential(q + h) - rbg.potential(q - h)) / (2 * h), abs=1e-6)
        h2 = 1e-4                                                  # second difference: larger step (round-off)
        assert dG == pytest.approx((rbg.potential(q + h2) - 2 * rbg.potential(q) + rbg.potential(q - h2)) / h2 ** 2,
                                   abs=1e-5)
    # geometry uncertainty is propagated (not absorbed into density) and vanishes with zero sd
    u0 = PR.gravity_stiffness_uncertainty(rbg, 0.4, n=50)
    u1 = PR.gravity_stiffness_uncertainty(rbg, 0.4, axis_sd_rad=math.radians(2), origin_sd_m=0.01, n=400)
    assert u0["dG_sd"] < 1e-12 and u1["dG_sd"] > 1e-3 and u1["G_sd"] > 1e-3


def test_prismatic_gravity_constant_and_stiffness_free():
    flat = PR.rigid_gravity("prismatic", (1, 0, 0), None, (0, 0, 0), 3.0, G)
    tilt = PR.rigid_gravity("prismatic", (math.cos(0.3), 0, math.sin(0.3)), None, (0, 0, 0), 3.0, G)
    q = np.linspace(-0.5, 0.5, 7)
    Gf, dGf = PR.gravity_load(flat, q)
    Gt, dGt = PR.gravity_load(tilt, q)
    assert np.all(Gf == 0) and np.all(dGf == 0)
    assert np.allclose(Gt, 3.0 * 9.81 * math.sin(0.3)) and np.all(dGt == 0)


def test_coordinate_reversal_preserves_work_and_flips_effort():
    """q' = -q, a' = -a: u' = -u, the same physical work, the same normalized rates and gain."""
    from forceartgs_g9.free_forced.normalized import GravityModel
    from forceartgs_g9.system_identification import energy_audit as EA
    from forceartgs_g9.system_identification import output_error as OE
    from forceartgs_g9.system_identification import synthetic_io as S
    u = S.pulse_train([1.5, 0.5], 0.3)
    s = S.segment("f", label="FORCED_MEASURED", M=0.8, c=0.5, Fc=0.2, v0=0.6, T=1.2, effort_levels=u, seed=5)
    from forceartgs_g9.free_forced.segments import MotionSegment
    r = MotionSegment("f", s.label, s.joint_type, s.t, -s.q, s.q_sigma, -s.sign, s.force_t, -s.Q, s.Q_sigma,
                      dict(s.evidence))
    e, er = S.episodes([s])[0], S.episodes([r])[0]
    g = GravityModel.zero("revolute")
    wa = EA.window_audit(e, g, n_mc=20)
    wb = EA.window_audit(er, g, n_mc=20)
    assert wb["nominal"]["W_in"] == pytest.approx(wa["nominal"]["W_in"], rel=1e-9)
    assert wb["nominal"]["J"] == pytest.approx(-wa["nominal"]["J"], rel=1e-9)
    fa = OE.fit([e], "L1", g, dict(n_starts=2), profiles=False)
    fb = OE.fit([er], "L1", g, dict(n_starts=2), profiles=False)
    for k in ("lam", "gam", "g_u"):
        assert fb["estimate"][k] == pytest.approx(fa["estimate"][k], rel=1e-4)
