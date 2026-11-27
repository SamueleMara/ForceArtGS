"""ANALYTIC_CONVOLUTION_1D (T10): exact ZOH kernel, stable lambda -> 0 limit, recovery and the flat valley."""
import numpy as np
import pytest
from scipy.integrate import quad

from forceartgs_g9.estimators import rest_to_rest_convolution as RC
from forceartgs_g9.estimators.rest_to_rest import ZOHEffort

from r2r_fixtures import episode, exact_rest_to_rest


def test_T10_kernel_is_exact_per_zoh_interval_and_continuous_at_lambda_zero():
    rng = np.random.default_rng(2)
    ft = np.cumsum(rng.uniform(0.01, 0.04, 40)) + 1.0
    lev = rng.normal(0, 1.0, 40)
    eff = ZOHEffort(ft, lev)
    t0 = ft[2] + 0.003
    t = np.sort(rng.uniform(t0, ft[-1] + 0.2, 12))
    for lam in (0.0, 1e-9, 1e-4, 0.37, 8.0, 60.0):
        h = RC.h_lambda(ft, lev, t0, t, lam)
        for ti, hi in zip(t, h):
            k = (lambda x: x) if lam == 0 else (lambda x: -np.expm1(-lam * x) / lam)
            ref = quad(lambda s: k(ti - s) * float(eff.at(s)), t0, ti, points=list(ft[(ft > t0) & (ft < ti)]),
                       limit=500, epsabs=1e-13)[0]
            assert hi == pytest.approx(ref, rel=1e-8, abs=1e-11)
    h0 = RC.h_lambda(ft, lev, t0, t, 0.0)
    np.testing.assert_allclose(RC.h_lambda(ft, lev, t0, t, 1e-12), h0, rtol=1e-9, atol=1e-14)
    x = np.array([0.999e-3, 1.001e-3]) / 0.5                               # both sides of the series switch
    F = RC._F(x, 0.5)
    np.testing.assert_allclose(F, x * x / 2 - 0.5 * x ** 3 / 6 + 0.25 * x ** 4 / 24, rtol=1e-9)


def test_T10_recovers_M_c_on_informative_forcing_and_never_estimates_q0_v0():
    sy = exact_rest_to_rest(1.7, 3.1, q0=0.44, pushes=((0.34, 6.0),), noise=2e-4, q_sigma=2e-4, seed=7)
    ep = episode(sy)["episode"]
    r = RC.convolution_1d(ep)
    assert r["status"] == "OK" and r["M"] == pytest.approx(1.7, rel=0.03) and r["c"] == pytest.approx(3.1, rel=0.05)
    assert "q0" not in r and "v0" not in r and "never estimates q0 or v0" in r["note"]
    rp = RC.replay(ep, r["M"], r["c"])
    assert rp["q"][0] == ep.q0 and rp["v"][0] == 0.0                         # starts at (q0, 0) exactly


def test_T10_weak_excitation_exposes_a_flat_lambda_valley():
    sy = exact_rest_to_rest(0.8, 0.02, pushes=((0.06, 1.0),), brake_s=0.06, rest1=0.4, noise=1e-3, q_sigma=1e-3, seed=8)
    r = RC.convolution_1d(episode(sy)["episode"])
    assert r["status"] == "OK" and r["valley"]["flat_lambda_valley"]
    assert r["valley"]["n_members"] > 3


def test_sign_inconsistent_input_cannot_produce_positive_inertia():
    sy = exact_rest_to_rest(0.8, 1.5)
    sy["Q"] = -sy["Q"]
    r = RC.convolution_1d(episode(sy)["episode"])
    assert r["status"] == "NO_POSITIVE_INERTIA"
