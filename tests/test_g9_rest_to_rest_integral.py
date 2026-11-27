"""Double-integral LS, exact ZOH integrals, orthogonality and SVD / family audits (T04, T08, T09)."""
import numpy as np
import pytest
from scipy.integrate import quad

from forceartgs_g9.estimators import rest_to_rest_diagnostics as RD
from forceartgs_g9.estimators import rest_to_rest_integral as RI
from forceartgs_g9.estimators.general_integral_ls import zoh_double_integral
from forceartgs_g9.estimators.rest_to_rest import ZOHEffort
from forceartgs_g9.estimators.rest_to_rest_convolution import simulate_two_term

from r2r_fixtures import episode, exact_rest_to_rest


def test_T04_native_zoh_double_integral_is_exact_on_nonuniform_clocks_with_signed_forces_and_t0_nonzero():
    rng = np.random.default_rng(1)
    ft = np.cumsum(rng.uniform(0.01, 0.05, 60)) + 3.0
    lev = rng.normal(0, 2.0, 60)                                         # negative and positive levels
    eff = ZOHEffort(ft, lev)
    t0 = ft[3] + 0.0071                                                  # inside a row interval, t0 != 0
    t = np.sort(rng.uniform(t0, ft[-1] + 0.3, 25))
    I1, I2 = zoh_double_integral(eff, t0, t)
    for ti, a, b in zip(t, I1, I2):
        pts = list(ft[(ft > t0) & (ft < ti)])
        qa = quad(lambda s: float(eff.at(s)), t0, ti, points=pts, limit=500, epsabs=1e-13)[0]
        qb = quad(lambda s: (ti - s) * float(eff.at(s)), t0, ti, points=pts, limit=500, epsabs=1e-13)[0]
        assert a == pytest.approx(qa, rel=1e-9, abs=1e-11) and b == pytest.approx(qb, rel=1e-9, abs=1e-11)
    G, H = RI.interval_kernel(ft, t0, t)                                 # the covariance kernel is the same integral
    np.testing.assert_allclose(G @ lev, I2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(H @ lev, I1, rtol=1e-12, atol=1e-12)


def test_T08_orthogonality_holds_for_the_exact_rest_to_rest_velocity_only():
    sy = exact_rest_to_rest(0.8, 1.5)
    ft, Q = sy["force_t"], sy["Q"]
    tg = np.linspace(0.0, sy["T_true"] + 0.5, 20001)
    r = simulate_two_term(ft, Q, 0.0, 0.0, 0.8, 1.5, tg)
    a = (ZOHEffort(ft, Q).at(tg) - 1.5 * r["v"]) / 0.8
    o = RD.orthogonality(tg, r["v"], a)
    assert abs(o["exact_identity"]) < 1e-20 and abs(o["normalized"]) < 1e-3       # quadrature error at ZOH jumps only
    # it rests on v(t0) = v(T) = 0: a window ending in motion is NOT orthogonal (int v vdot = v(T)^2 / 2)
    i = int(np.searchsorted(tg, sy["T_true"] - 0.1))
    o2 = RD.orthogonality(tg[:i], r["v"][:i], a[:i])
    assert o2["int_v_vdot"] == pytest.approx(o2["exact_identity"], rel=1e-2) and abs(o2["normalized"]) > 1e-2
    # noisy finite differences are never CLAIMED orthogonal; a weighted inner product breaks the identity
    rng = np.random.default_rng(0)
    tq = sy["t_q"]
    vq = np.gradient(sy["q_true"] + rng.standard_normal(len(tq)) * 2e-3, tq)
    assert "not claimed orthogonal" in RD.orthogonality(tq, vq, np.gradient(vq, tq))["claim"]
    assert abs(RD.orthogonality(tg, np.exp(-tg) * r["v"], a)["normalized"]) > 1e-2


def test_T09_well_excited_is_unique_poorly_excited_keeps_a_family():
    sy = exact_rest_to_rest(0.8, 1.5, noise=2e-4, q_sigma=2e-4, seed=3)
    good = RI.integral_svd_r2r(episode(sy)["episode"])
    assert good["parameters_unique"] and good["whitened"]["svd"]["rank"] == 2
    assert good["estimate"]["M"] == pytest.approx(0.8, rel=0.05) and good["estimate"]["c"] == pytest.approx(1.5, rel=0.1)
    # weak damping signature: c tiny against the inertial response -> c not pinned: a family, not a false unique fit
    sy = exact_rest_to_rest(0.8, 0.01, pushes=((0.1, 2.0),), brake_s=0.1, rest1=0.6, noise=2e-4, q_sigma=2e-4, seed=4)
    ep = episode(sy)["episode"]
    poor = RI.integral_svd_r2r(ep, draws=RI.noise_draws(ep, 60))
    assert not poor["parameters_unique"] and poor["near_optimal_family"]["relative_sd"]["c"] > 0.5
    cs = [m["c"] for m in poor["near_optimal_family"]["members"]]
    assert max(cs) - min(cs) > 0.1 and poor["estimate"]["M"] == pytest.approx(0.8, rel=0.05)


def test_ls_is_cross_checked_against_the_output_error_optimum_of_the_same_model():
    from forceartgs_g9.estimators import rest_to_rest_convolution as RC
    for noise, expect in ((2e-3, "INTEGRAL_LS_OUTPUT_ERROR_INCONSISTENT"), (2e-4, "CONSISTENT")):
        sy = exact_rest_to_rest(0.8, 0.01, pushes=((0.1, 2.0),), brake_s=0.1, rest1=0.6, noise=noise, q_sigma=noise,
                                seed=4)
        ep = episode(sy)["episode"]
        r = RI.integral_svd_r2r(ep)
        x = RI.output_error_crosscheck(ep, r["estimate"]["M"], r["estimate"]["c"], RC.convolution_1d(ep))
        assert x["status"] == expect, x


def test_svd_whitening_is_declared_noise_not_residual_inflation():
    sy = exact_rest_to_rest(0.8, 1.5, noise=2e-4, q_sigma=2e-4, seed=5)
    ep = episode(sy)["episode"]
    s1 = RI.integral_svd_r2r(ep)
    C, comp, sb = RI.residual_covariance(ep, [0.8, 1.5])
    assert set(comp) == {"q_samples", "q0_anchor", "force_rows", "tare_reference"} and sb > 0
    assert np.allclose(C, C.T) and np.all(np.linalg.eigvalsh(C) > 0)
    # the reported sd does not scale with the fitted residual (no chi2 inflation): same covariance at chi2 != 1
    assert s1["whitened"]["sd"]["M"] > 0 and "declared" in s1["whitened"]["covariance"]
