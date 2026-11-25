"""G10-A behavioral replay -- synthetic pipeline tests on the TEST DOUBLE of the worker's step law (fast; the real
engine is exercised in test_g9_physx_behavioral_engine.py): T3 one continuous rollout vs deliberate segmented resets,
T5-lite optimizer improvement + near-optimal family, T9 a wrong input is labelled (cannot hide behind a widened
covariance), T14 reproducibility, T15 near-equal demo fits diverge under a new input."""
import numpy as np
import pytest

from forceartgs_g9.physx_refinement import behavioral_adapter as BA
from forceartgs_g9.physx_refinement import behavioral_fit as BF
from forceartgs_g9.physx_refinement import behavioral_report as BR
from forceartgs_g9.physx_refinement.behavioral_contracts import load_config
from forceartgs_g9.physx_refinement.objective import Episode, PhysicalParameters

from g10_behavioral_fixtures import TestDoubleEngine, synthetic_demo

TRUTH = BF.params_from(dict(rho_scale=1.6, armature=0.0, dynamic_effort=0.4, static_excess=0.0, viscous=3.0))
SMALL = {"fitting": {"maxiter": 6, "popsize": 5, "polish": False, "local_max_nfev": 150, "profile_points": 4,
                     "profile_inner_max_nfev": 30, "profile_parameters": ["rho_scale", "viscous"]},
         "alternatives": {"maxiter": 2, "popsize": 4}}
CFG = load_config(SMALL)


def _bounds():
    return BF.bounds_for(CFG["model"], 1.0, 2.5)


def _starts(b):
    from forceartgs_g9.physx_refinement.behavioral_real import starts_for
    return starts_for(b, 1.0)


@pytest.fixture(scope="module")
def demo():
    eng = TestDoubleEngine()
    rec, q_true = synthetic_demo(eng, TRUTH, q0=0.15, noise=2e-4, seed=3)
    ep = BA.build_full_demo_episode(rec, CFG)["episode"]
    return eng, rec, ep, q_true


@pytest.fixture(scope="module")
def fitted(demo):
    eng, rec, ep, _ = demo
    roll = BF.CachingRollout(eng)
    b = _bounds()
    fit = BF.fit_full_demo(roll, ep, CFG, b, _starts(b))
    fam = BF.profile_family(roll, fit, CFG, b)
    return roll, fit, fam


# -------------------------------------------------------------------------------------------- T3
def test_scored_rollout_is_one_continuous_run_from_rest(demo, fitted):
    eng, rec, ep, q_true = demo
    roll, fit, _ = fitted
    starts = {(r["t0"], r["q0"], r["v0"]) for r in eng.requests}
    assert starts == {(ep.t0_s, ep.q0_observed, 0.0)}                     # every rollout: same t0, q0, v0 = 0
    assert all(r["n"] == len(ep.t_q_s) for r in eng.requests)              # the whole demo, never a window
    # a deliberate SEGMENTED replay (reset at the phase boundaries with a prefix-velocity estimate) differs from the
    # continuous truth; the continuous one is what is scored
    pe = ep.to_physx_episode()
    cont = np.asarray(eng(TRUTH, pe).q)
    assert np.sqrt(np.mean((cont - q_true) ** 2)) < 1e-6
    k = int(np.searchsorted(ep.t_q_s, 70 / 50.0))                          # push end
    v_pref = (ep.q_observed[k] - ep.q_observed[k - 3]) / (ep.t_q_s[k] - ep.t_q_s[k - 3])
    seg = Episode("seg", ep.t_q_s[k:], ep.q_observed[k:], ep.q_sigma[k:], float(ep.q_observed[k]), float(v_pref),
                  ep.effort_t_s[ep.effort_t_s <= ep.t_q_s[k] + 1e-9][-1:].tolist() +
                  ep.effort_t_s[ep.effort_t_s > ep.t_q_s[k] + 1e-9].tolist(),
                  np.r_[ep.effort_object[ep.effort_t_s <= ep.t_q_s[k] + 1e-9][-1:],
                        ep.effort_object[ep.effort_t_s > ep.t_q_s[k] + 1e-9]], "forced", "VERIFIED")
    q_seg = np.r_[cont[:k], np.asarray(eng(TRUTH, seg).q)]
    assert np.max(np.abs(q_seg - q_true)) > np.max(np.abs(cont - q_true))


# -------------------------------------------------------------------------------------------- T5-lite
def test_fit_improves_over_a_wrong_prior_and_reports_a_family(fitted):
    _, fit, fam = fitted
    base = fit["baselines"]["prior_reference"]["metrics"]["rmse"]
    best = fit["best"]["metrics"]
    assert best["rmse"] < 0.2 * base
    assert best["rmse_over_sigma"] < 3.0
    assert fit["engine_initial_state"]["v0_engine"] == 0.0
    assert fam["n_members"] >= 1 and "rho_scale" in fam["profiles"]
    assert fam["rule"].count("NOT a frequentist") == 1


# -------------------------------------------------------------------------------------------- T9
def test_wrong_input_scale_is_inadequate_under_gravity_and_metrics_are_separate():
    eng = TestDoubleEngine(G1=2.0)                                         # constant gravity load: scale not absorbable
    rec, _ = synthetic_demo(eng, TRUTH, Q_scale=0.4, phases=[(10, 0.0, "absent"), (60, 6.0, "sliding"),
                                                             (220, 0.0, "absent")], seed=1)
    ep = BA.build_full_demo_episode(rec, CFG)["episode"]
    roll = BF.CachingRollout(eng)
    b = BF.bounds_for(CFG["model"], 1.0, 6.0)
    fit = BF.fit_full_demo(roll, ep, CFG, b, _starts(b))
    m = fit["best"]["metrics"]
    for k in ("rmse", "rmse_over_sigma", "rmse_over_sigma_q_measurement", "rmse_over_range"):
        assert k in m
    assert m["rmse_over_sigma"] <= m["rmse_over_sigma_q_measurement"]      # the floor can only reduce it
    cls = BR.classify(fit, CFG, input_verified=False, conditional=True)
    assert cls["status"] == "BEHAVIORAL_FIT_INADEQUATE" and cls["replay_label"] == "DEMO_REPLAY_INADEQUATE"


# -------------------------------------------------------------------------------------------- T14
def test_reproducible_with_the_same_seed(demo):
    eng, rec, ep, _ = demo
    b = _bounds()
    r = [BF.fit_full_demo(BF.CachingRollout(TestDoubleEngine()), ep, CFG, b, _starts(b)) for _ in range(2)]
    assert r[0]["best"]["params"] == r[1]["best"]["params"]
    assert r[0]["best"]["objective"] == r[1]["best"]["objective"]


# -------------------------------------------------------------------------------------------- T15
def test_near_equal_demo_fits_diverge_under_a_new_input(demo):
    eng, rec, ep, _ = demo
    roll = BF.CachingRollout(eng)
    pe = ep.to_physx_episode()
    # two different parameter sets that fit THIS demo almost equally (heavier + more damped)
    a = TRUTH
    b = BF.params_from(dict(rho_scale=2.2, armature=0.0, dynamic_effort=0.4, static_excess=0.0, viscous=3.1))
    ma, mb = (BF.replay_metrics(pe, roll(p, pe), 3e-4) for p in (a, b))
    members = [dict(label="a", params=a.__dict__, metrics=ma), dict(label="b", params=b.__dict__, metrics=mb)]
    sp = BF.policy_spread(roll, ep, members, CFG)
    assert sp["status"] == "EVALUATED" and sp["probe"]["semantics"] == "SYNTHETIC_PROBE_NOT_A_REAL_DEMONSTRATION"
    assert sp["max_q_spread"] > 0


def test_input_sensitivity_uses_the_same_theta(demo, fitted):
    eng, rec, ep, _ = demo
    _, fit, _ = fitted
    hyps, _ = BA.input_hypotheses(rec, ep, CFG)
    s = BF.input_sensitivity(BF.CachingRollout(eng), ep, hyps, fit["_params"], CFG)
    assert s["status"] in ("INPUT_ROBUST_AT_THETA", "INPUT_CONDITIONED_FIT")
    assert set(s["hypotheses"]) >= {"recorded", "noise_realization_0"}
