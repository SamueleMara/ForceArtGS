"""Input-output identification -- estimators, audits, contracts (no engine): T2 free scale invariance, T3 known-force
absolute scale (also small high-SNR effort), T4 static-friction refusal, T6 work / impulse consistency, T7 short / noisy
windows and Coulomb-vs-viscous weak separation, T10 contracts / status / no value / append-only."""
import json

import numpy as np
import pytest

from forceartgs_g9.free_forced.normalized import GravityModel
from forceartgs_g9.system_identification import energy_audit as EA
from forceartgs_g9.system_identification import output_error as OE
from forceartgs_g9.system_identification import synthetic_io as S
from forceartgs_g9.system_identification.api import identify_io_response
from forceartgs_g9.system_identification.contracts import (STATUS_DIMENSIONS, IOContractError, assert_no_value_field,
                                                           load_config)
from forceartgs_g9.system_identification.physx_bridge import initialize_physx_refinement
from forceartgs_g9.system_identification.segmentation_adapter import from_segments

FAST = {"fit": {"compare_methods": ["output_error", "weak_form"], "profile_points": 9, "n_starts": 2,
                "feature_bootstrap_seeds": 2, "energy_mc": 60}}
JOINT = dict(joint_type="revolute", kinematic_status="IDENTIFIED", export_ready=True)
M, C, FC = 0.8, 0.6, 0.3
ZERO = GravityModel.zero("revolute")


def _feat(r, name):
    return next(f for f in r["features"] if f["name"] == name)


def _forced_set(scale=1.0, noise=1e-3, effort_sigma=0.02):
    u1 = S.pulse_train([2.0 * scale, 0.6 * scale, 1.4 * scale, 0.2 * scale], period=0.3)
    u2 = S.pulse_train([1.0 * scale, 2.5 * scale], 0.4)
    return [S.segment("f1", label="FORCED_MEASURED", M=M, c=C, Fc=FC, v0=0.4, T=1.6, effort_levels=u1, seed=1,
                      q_noise=noise, effort_sigma=effort_sigma),
            S.segment("f2", label="FORCED_MEASURED", M=M, c=C, Fc=FC, v0=0.8, T=1.6, effort_levels=u2, seed=2,
                      t_start=3.0, q_noise=noise, effort_sigma=effort_sigma),
            S.segment("c1", label="FREE_CONFIRMED", M=M, c=C, Fc=FC, v0=1.2, T=1.0, seed=3, t_start=6.0,
                      q_noise=noise)]


@pytest.fixture(scope="module")
def forced_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("io_forced")
    eps = S.episodes(_forced_set())
    return identify_io_response(eps, JOINT, dict(force_audit_status="VERIFIED"), None, load_config(FAST), out,
                                log=lambda *a: None), out


@pytest.fixture(scope="module")
def free_result():
    segs = [S.segment("c1", label="FREE_CONFIRMED", M=M, c=C, Fc=FC, v0=1.5, T=1.2, seed=3),
            S.segment("c2", label="FREE_CONFIRMED", M=M, c=C, Fc=FC, v0=1.1, T=1.0, seed=4, t_start=4)]
    eps = S.episodes(segs)
    return identify_io_response(eps, JOINT, dict(force_audit_status="VERIFIED"), dict(prior_M_range=[0.2, 5.0]),
                                load_config(FAST), None, log=lambda *a: None)


# ------------------------------------------------------------------------------------------------- T2
def test_free_only_is_ratio_only_flat_scale_and_never_massless(free_result):
    r = free_result
    assert _feat(r, "gain")["status"] == "NOT_IDENTIFIED" and _feat(r, "gain")["estimate"] is None
    for name, truth in (("lambda", C / M), ("gamma", FC / M)):
        f = _feat(r, name)
        lo, hi = f["uncertainty"]["ci95"]
        assert lo <= truth <= hi, (name, lo, hi)
    sp = r["scale_profile"]
    assert sp["status"] == "FLAT_SCALE_SYMMETRY" and sp["ci95_M"] is None
    assert sp["flat_range"] < 1e-6                      # identical fits at every M (scale symmetry)
    assert min(p["M"] for p in sp["points"]) > 0
    assert r["status_dimensions"]["STRUCTURAL_IDENTIFIABILITY"] == "RATIO_ONLY"
    assert r["physical_mapping"]["status"] == "RATIO_ONLY"
    fam = r["physical_mapping"]["family"]
    assert fam["M_range_provenance"].startswith("ASSUMED")
    for m in fam["members"]:                            # (M, M lam, M gam): ratios fixed
        assert m["c"] / m["M"] == pytest.approx(r["handoff"]["estimate_not_for_sim"]["normalized"]["lam"])
    assert "g_u" not in r["selected_fit"]["names"] and "log_g_u" not in r["selected_fit"]["names"]


def test_free_fit_identical_normalized_parameters_for_scaled_truth():
    est = []
    for k in (0.2, 1.0, 6.0):
        segs = [S.segment("c1", label="FREE_CONFIRMED", M=M * k, c=C * k, Fc=FC * k, v0=1.5, T=1.2, seed=3)]
        f = OE.fit(S.episodes(segs), "L1", ZERO, dict(n_starts=2), profiles=False)
        est.append((f["estimate"]["lam"], f["estimate"]["gam"]))
    np.testing.assert_allclose(est[0], est[1], rtol=1e-5)
    np.testing.assert_allclose(est[2], est[1], rtol=1e-5)


# ------------------------------------------------------------------------------------------------- T3
def test_known_force_recovers_absolute_scale(forced_result):
    r, out = forced_result
    g = _feat(r, "gain")
    assert g["status"] == "IDENTIFIED"
    lo, hi = g["uncertainty"]["ci95_M"]
    assert lo <= M <= hi and (hi - lo) / M < 0.05
    assert _feat(r, "lambda")["estimate"] == pytest.approx(C / M, rel=0.05)
    assert _feat(r, "gamma")["estimate"] == pytest.approx(FC / M, rel=0.08)
    sp = r["scale_profile"]
    assert sp["status"] == "CLOSED" and sp["ci95_M"][0] <= M * 1.01 and sp["ci95_M"][1] >= M * 0.99
    mp = r["physical_mapping"]
    assert mp["M"]["estimate_not_for_sim"] == pytest.approx(M, rel=0.02)
    assert mp["c"]["estimate_not_for_sim"] == pytest.approx(C, rel=0.06)
    d = r["status_dimensions"]
    assert d["STRUCTURAL_IDENTIFIABILITY"] == "FULL" and d["PRACTICAL_IDENTIFIABILITY"] == "INFORMATIVE"
    assert d["INDEPENDENT_VALIDATION"] == "NOT_AVAILABLE"            # no held-out window here
    assert d["SIMULATOR_READINESS"] != "PRODUCTION_READY"             # no native engine / held-out
    assert r["selection"]["selected"] == "L1"


def test_small_but_high_snr_effort_still_identifies():
    """Efforts ~0.1-0.4 N m (vs 0.6-2.5 above) but sigma_u = 2 mN m and sigma_q = 0.2 mrad: informative."""
    c, fc = 0.15, 0.05
    segs = [S.segment(f"s{k}", label="FORCED_MEASURED", M=M, c=c, Fc=fc, v0=0.3 + 0.2 * k, T=1.5,
                      effort_levels=S.pulse_train([0.4, 0.1, 0.3][k:] + [0.15], 0.3), seed=10 + k, t_start=3 * k,
                      q_noise=2e-4, effort_sigma=0.002) for k in range(2)]
    eps = S.episodes(segs)
    assert max(np.max(np.abs(e.effort)) for e in eps) < 0.45
    f = OE.fit(eps, "L1", ZERO, dict(n_starts=2, profile_points=9))
    p = f["profiles"]["log_g_u"]
    lo, hi = p["ci95_M"]
    assert lo <= M <= hi and not p["open_low"] and not p["open_high"]


# ------------------------------------------------------------------------------------------------- T4
def test_static_friction_never_inferred_and_variants_recorded():
    eps = S.episodes(_forced_set()[:1])
    v = OE.fit_variants(eps, ZERO, dict(n_starts=1, profile_points=5),
                        variants=["inertial_viscous", "inertial_viscous_coulomb", "inertial_viscous_coulomb_static",
                                  "inertial_viscous_coulomb_spring"])
    assert v["L2"]["status"] == "NOT_RUN" and "never inferred from non-motion" in v["L2"]["reason"]
    assert v["L3"]["status"] == "NOT_RUN"                            # one window: no configuration coverage
    assert v["L4"]["status"] == "NOT_RUN" and v["L0"]["status"] == v["L1"]["status"] == "FITTED"


# ------------------------------------------------------------------------------------------------- T6
def test_work_impulse_audit_detects_sign_lag_and_lever_errors():
    u = S.pulse_train([1.0, 2.5], 0.4)
    good = S.segment("g", label="FORCED_MEASURED", M=M, c=C, Fc=FC, v0=0.8, T=1.6, effort_levels=u, seed=2)
    sign = S.segment("s", label="FORCED_MEASURED", M=M, c=C, Fc=FC, v0=0.8, T=1.6, effort_levels=u, seed=2,
                     effort_override=lambda t: -u(t))
    lag = S.segment("l", label="FORCED_MEASURED", M=M, c=C, Fc=FC, v0=0.8, T=1.6, effort_levels=u, seed=2,
                    effort_override=lambda t: u(t - 0.2))
    lever = S.segment("v", label="FORCED_MEASURED", M=M, c=C, Fc=FC, v0=0.8, T=1.6, effort_levels=u, seed=2,
                      effort_override=lambda t: 2.0 * u(t))
    rates = dict(lam=C / M, gam=FC / M)
    eg, es, el, ev = S.episodes([good, sign, lag, lever])
    wg = EA.window_audit(eg, ZERO, rates, 1 / M, n_mc=80)
    assert wg["verdict"] == "CONSISTENT" and wg["nominal"]["W_in"] > 0
    assert EA.window_audit(es, ZERO, rates, 1 / M, n_mc=80)["verdict"] == "INPUT_OUTPUT_SIGN_INCONSISTENT"
    assert set(EA.window_audit(el, ZERO, rates, 1 / M, n_mc=80)["codes"]) & {"POSSIBLE_UNMODELED_DRIVE",
                                                                             "UNMODELED_DISSIPATION"}
    # a lever (projection scale) error is visible only against an independent mass ...
    assert "UNMODELED_DISSIPATION" in EA.window_audit(ev, ZERO, rates, 1 / M, n_mc=80)["codes"]
    # ... a fitted gain absorbs it as a scale error (M doubles) -- reported as such, not "corrected"
    f = OE.fit([ev], "L1", ZERO, dict(n_starts=2), profiles=False)
    assert 1 / f["estimate"]["g_u"] == pytest.approx(2 * M, rel=0.05)


def test_wrong_sign_cannot_be_fitted_away():
    u = S.pulse_train([1.0, 2.5], 0.4)
    bad = S.segment("s", label="FORCED_MEASURED", M=M, c=C, Fc=FC, v0=0.8, T=1.6, effort_levels=u, seed=2,
                    effort_override=lambda t: -u(t))
    eps = S.episodes([bad])
    r = identify_io_response(eps, JOINT, dict(force_audit_status="VERIFIED"), None, load_config(FAST), None,
                             log=lambda *a: None)
    est = r["handoff"]["estimate_not_for_sim"]["normalized"]
    assert est["g_u"] > 0 and est["lam"] >= 0 and (est["gam"] or 0) >= 0     # no negative mass / friction
    assert r["status_dimensions"]["INPUT_VALIDITY"] == "CONTRADICTED"
    assert r["status_dimensions"]["SIMULATOR_READINESS"] == "BLOCKED"


def test_free_passive_coast_is_passive():
    s = S.segment("c", label="FREE_CONFIRMED", M=M, c=C, Fc=FC, v0=1.5, T=1.2, seed=3)
    w = EA.window_audit(S.episodes([s])[0], ZERO, dict(lam=C / M, gam=FC / M), n_mc=60)
    assert w["verdict"] == "CONSISTENT" and w["nominal"]["e_n"] < 0
    res = w["free_balance_residual_n"]
    assert abs(res["estimate"]) <= 3 * res["sd"]


# ------------------------------------------------------------------------------------------------- T7
def test_short_noisy_window_and_no_fabricated_frequency_response():
    u = S.pulse_train([2.0, 0.5], 0.15)
    s = S.segment("short", label="FORCED_MEASURED", M=M, c=C, Fc=FC, v0=0.6, T=0.45, effort_levels=u, seed=7,
                  q_noise=5e-3)
    eps = S.episodes([s])
    r = identify_io_response(eps, JOINT, dict(force_audit_status="VERIFIED"), None, load_config(FAST), None,
                             log=lambda *a: None)
    assert r["status"] == "OK"
    for name in ("lambda", "gain"):
        f = _feat(r, name)
        ci = f["uncertainty"].get("ci95")
        if f["status"] in ("IDENTIFIED", "CONDITIONAL"):
            truth = C / M if name == "lambda" else 1 / M
            assert ci[0] <= truth <= ci[1]
    txt = json.dumps(r, default=str)
    assert "frequency_response" not in txt and "fft" not in txt.lower()


def test_flat_velocity_coulomb_viscous_weakly_separated():
    s = S.segment("flat", label="FREE_CONFIRMED", M=M, c=0.05, Fc=0.02, v0=1.0, T=0.6, seed=8, q_noise=2e-3)
    eps = S.episodes([s])
    r = identify_io_response(eps, JOINT, dict(force_audit_status="VERIFIED"), None, load_config(FAST), None,
                             log=lambda *a: None)
    sens = r["sensitivity"]
    labs = sens["theta_labels"]
    if "gam" in labs:
        i, j = labs.index("lam"), labs.index("gam")
        assert abs(sens["theta_correlation_with_nuisance"][i][j]) > 0.9
    assert r["status_dimensions"]["PRACTICAL_IDENTIFIABILITY"] in ("WEAK", "BOUNDARY_ONLY")
    assert any(_feat(r, n)["status"] == "NOT_IDENTIFIED" for n in ("lambda", "gamma") if
               _feat(r, n)["status"] != "NOT_APPLICABLE")


# ------------------------------------------------------------------------------------------------- T10
def test_outputs_status_dimensions_and_no_value(forced_result):
    r, out = forced_result
    for f in ("io_response.json", "transfer_functions.json", "work_energy_audit.json", "parameter_profiles.json",
              "physx_reconciliation.json", "training_replay.npz"):
        assert (out / f).is_file(), f
    for f in ("io_response.json", "transfer_functions.json", "work_energy_audit.json", "parameter_profiles.json"):
        assert_no_value_field(json.loads((out / f).read_text()))
    io = json.loads((out / "io_response.json").read_text())
    assert io["schema"] == "forceartgs.g9.io_response/v1"
    dims = {k: v for k, v in io["status_dimensions"].items() if not k.startswith("_")}
    assert set(dims) == set(STATUS_DIMENSIONS) and all(v in STATUS_DIMENSIONS[k] for k, v in dims.items())
    tf = json.loads((out / "transfer_functions.json").read_text())["windows"]
    assert all(w["status"] in ("LOCAL_SLIDING_ONLY", "NOT_APPLICABLE") for w in tf.values())
    assert io["handoff"]["simulation_export_allowed"] is False


def test_config_is_strict():
    with pytest.raises(IOContractError):
        load_config({"fit": {"no_such_key": 1}})
    with pytest.raises(IOContractError):
        load_config({"physx": {"response_regularizer_weight": 0.1}})
    with pytest.raises(IOContractError):
        load_config({"physx": {"disable_unexpected_link_damping": False}})
    with pytest.raises(IOContractError):
        load_config({"model_variants": ["inertial_viscous_coulomb"]})       # L0 must be fitted first
    with pytest.raises(IOContractError):
        load_config({"contact_policy": "threshold_free"})
    c = load_config("configs/g9_real/io_identification.example.yaml")
    assert c["enabled"] is False and c["physx"]["response_regularizer_weight"] == 0.0


def test_near_zero_force_in_contact_is_excluded_never_free():
    u = S.pulse_train([1.0], 1.0)
    hyp = S.segment("hyp", label="FREE_HYPOTHESIS:measured_near_zero_effort_in_contact", M=M, c=C, Fc=FC, v0=1.0,
                    T=1.0, seed=1)
    hyp.evidence.update(interval_rows=[10, 60], reason="LOW_MEASURED_EFFORT_IN_ANNOTATED_CONTACT")
    f = S.segment("f", label="FORCED_MEASURED", M=M, c=C, Fc=FC, v0=0.5, T=1.0, effort_levels=u, seed=2, t_start=3)
    ev = dict(intervals=[dict(label="UNKNOWN_OR_UNMEASURED", reason="LOW_MEASURED_EFFORT_IN_ANNOTATED_CONTACT",
                              rows=[10, 60], t=[0, 1], n_q=30, moving_fraction=0.9),
                         dict(label="UNKNOWN_OR_UNMEASURED", reason="STICK_INTERVAL", rows=[80, 90], t=[2, 2.2],
                              n_q=5, moving_fraction=0.2)])
    eps, g10, excl, reg = from_segments([hyp, f], force_audit_status="VERIFIED", ev=ev)
    assert [e.name for e in eps] == ["f"] and all(e.mode == "forced" for e in eps)
    codes = {x["name"]: x["code"] for x in excl}
    assert codes["hyp"] == "HYPOTHESIS_ONLY"
    assert next(x for x in excl if x["name"] == "hyp")["underlying_code"] == "NEAR_ZERO_FORCE_IN_CONTACT"
    assert codes["interval_rows_80-90"] == "STICK_INTERVAL"
    assert "interval_rows_10-60" not in codes                         # listed once (as the hypothesis segment)
    # the shared episode is the SAME window / clock / effort as the G10 episode
    assert eps[0].g10.t_s[0] == eps[0].t0 and np.array_equal(eps[0].g10.input_effort, eps[0].effort)


def test_no_admissible_episode_reports_not_testable(tmp_path):
    r = identify_io_response([], JOINT, dict(force_audit_status="NOT_EVALUATED", excluded=[
        dict(name="x", code="NEAR_ZERO_FORCE_IN_CONTACT")]), None, load_config(FAST), tmp_path, log=lambda *a: None)
    assert r["status"] == "NO_ADMISSIBLE_EPISODE"
    d = r["status_dimensions"]
    assert d["STRUCTURAL_IDENTIFIABILITY"] == "NOT_TESTABLE" and d["INPUT_VALIDITY"] == "UNKNOWN"
    assert d["INDEPENDENT_VALIDATION"] == "NOT_AVAILABLE" and d["SIMULATOR_READINESS"] == "BLOCKED"
    assert_no_value_field(json.loads((tmp_path / "io_response.json").read_text()))


def test_heldout_frozen_theta_status():
    segs = _forced_set()[:2]
    eps = S.episodes(segs, heldout="last_if_multiple")
    assert [e.train for e in eps] == [True, False]
    f = OE.fit(eps, "L1", ZERO, dict(n_starts=2), profiles=False)
    assert f["episodes"]["f2"]["ic_fitted"] is False and f["episodes"]["f2"]["v0_fit"] == eps[1].v0


def test_append_only_versions(tmp_path):
    from forceartgs_g9.forcetwin.pipeline import next_version
    a = next_version(tmp_path / "io_identification")
    (a / "io_response.json").write_text("{}")
    b = next_version(tmp_path / "io_identification")
    assert a.name == "v000" and b.name == "v001" and (a / "io_response.json").read_text() == "{}"


def test_physx_initialization_maps_and_refuses_window_mismatch(forced_result):
    r, out = forced_result
    io = json.loads((out / "io_response.json").read_text())
    eps = S.episodes(_forced_set())
    g10 = [e.g10 for e in eps]
    b = dict(rho_scale=(0.1, 10.0), armature=(0.0, 0.0), dynamic_effort=(0.0, 50.0), static_excess=(0.0, 0.0),
             viscous=(0.0, 50.0))
    ini = initialize_physx_refinement(io, dict(fitting=dict(bounds=b), axis_inertia=0.4), g10)
    assert ini["status"] == "OK" and ini["response_regularizer_weight"] == 0.0
    assert ini["x0"]["rho_scale"] == pytest.approx(M / 0.4, rel=0.03)
    assert ini["x0"]["viscous"] == pytest.approx(C, rel=0.06)
    from dataclasses import replace
    bad = [replace(g10[0], t_s=g10[0].t_s + 0.5)] + g10[1:]
    assert initialize_physx_refinement(io, dict(fitting=dict(bounds=b), axis_inertia=0.4), bad)["status"] == \
        "REFUSED_WINDOW_MISMATCH"


def test_g10_v0_nuisance_opt_in_recovers_initial_velocity():
    """G10 optimizer extension (opt-in): a wrong prefix v0 is corrected inside +-k sigma_v0, with the prior."""
    from forceartgs_g9.physx_refinement.objective import Episode, LossSettings, RefinementLoss, SimulationResult
    from forceartgs_g9.physx_refinement.optimizer import FitConfiguration, fit_physx

    def sim(p, e):
        t = e.t_s - e.t_s[0]
        I = p.rho_scale * 0.5
        return SimulationResult(q=e.q0 + e.v0 * t + 0.5 * (2.0 - p.dynamic_effort) / I * t ** 2)
    t = np.linspace(0, 1, 26)
    truth_v0 = 0.30
    q = 0.0 + truth_v0 * t + 0.5 * (2.0 - 0.4) / 0.5 * t ** 2
    e = Episode("f", t, q, np.full_like(t, 1e-3), 0.0, 0.25, np.array([0.0]), np.array([2.0]), "forced", "VERIFIED",
                sigma_v0=0.03)
    b = dict(rho_scale=(1.0, 1.0), armature=(0.0, 0.0), dynamic_effort=(0.0, 1.0), static_excess=(0.0, 0.0),
             viscous=(0.0, 0.0))
    with pytest.raises(ValueError):                                   # needs the loss opt-in
        fit_physx(RefinementLoss(sim, [e]), FitConfiguration(b, v0_nuisance_bound_sd=3.0, maxiter=5))
    obj = RefinementLoss(sim, [e], LossSettings(fit_initial_velocity=True))
    r = fit_physx(obj, FitConfiguration(b, v0_nuisance_bound_sd=3.0, maxiter=60, popsize=10))
    assert r["optimizer"]["v0_offsets"]["f"] == pytest.approx(truth_v0 - 0.25, abs=3e-3)
    assert r["params"]["dynamic_effort"] == pytest.approx(0.4, abs=0.01)
    r0 = fit_physx(RefinementLoss(sim, [e]), FitConfiguration(b, maxiter=60, popsize=10))
    assert r0["optimizer"]["v0_offsets"] == {} and r0["loss"]["objective"] > r["loss"]["objective"]


# ------------------------------------------------------------------------------- real-data regressions (2026-11-24)
def test_rate_at_zero_bound_finite_sensitivity_and_features():
    """Undamped truth: lambda lands on its zero bound. It must be exactly 0 (not a denormal), the sensitivity must be
    finite (no 0/0 log step), and v_inf must not be emitted as inf (real suitcase crash)."""
    segs = [S.segment("f1", label="FORCED_MEASURED", M=M, c=0.0, Fc=0.0, v0=0.3, T=1.4,
                      effort_levels=S.pulse_train([1.0, 0.3, 0.8], period=0.4), seed=5, effort_sigma=0.02)]
    r = identify_io_response(S.episodes(segs), JOINT, dict(force_audit_status="VERIFIED"), None, load_config(FAST),
                             None, log=lambda *a: None)
    assert r["status"] == "OK"
    assert r["sensitivity"]["nonfinite_columns"] == []
    lam = r["selected_fit"]["estimate_not_for_sim"]["lam"]
    assert lam == 0.0 or lam > 1e-6
    assert all(f["estimate"] is None or np.isfinite(f["estimate"]) for f in r["features"]
               if not isinstance(f["estimate"], (list, dict)))


def test_nested_variant_never_above_minimal_and_shares_weighting():
    eps = S.episodes(_forced_set())
    cfg = load_config(FAST)["fit"]
    var = OE.fit_variants(eps, ZERO, cfg, variants=["inertial_viscous", "inertial_viscous_coulomb"])
    assert var["L1"]["cost"] <= var["L0"]["cost"] + 1e-6
    assert var["L1"]["input_noise"]["status"] == "INHERITED"


def test_wrong_gravity_geometry_is_contradicted_not_absorbed():
    """Truth has no gravity load; a primary geometry with a large load must lose decisively (full NLL) to the labelled
    zero-gravity alternative -> INADEQUATE / BLOCKED, never 'adequate' through an inflated gain. Control: correct
    primary stays CONSISTENT."""
    eps = S.episodes(_forced_set())
    wrong = GravityModel("wrong_tilt", "known_shape", "revolute", A=-6.0, B=1.0, provenance="test: wrong geometry")
    r = identify_io_response(eps, JOINT, dict(force_audit_status="VERIFIED"),
                             dict(primary=wrong, alternatives=dict(vertical_axis_zero_gravity=ZERO)),
                             load_config(FAST), None, log=lambda *a: None)
    assert r["gravity_check"]["status"] == "CONTRADICTED"
    assert r["status_dimensions"]["MODEL_ADEQUACY"] == "INADEQUATE"
    assert r["status_dimensions"]["SIMULATOR_READINESS"] == "BLOCKED"
    ok = identify_io_response(eps, JOINT, dict(force_audit_status="VERIFIED"),
                              dict(primary=ZERO, alternatives=dict(wrong_tilt=wrong)), load_config(FAST), None,
                              log=lambda *a: None)
    assert ok["gravity_check"]["status"] == "CONSISTENT"
    assert ok["status_dimensions"]["MODEL_ADEQUACY"] == "ADEQUATE"


def test_misfit_larger_than_motion_is_never_adequate():
    from types import SimpleNamespace
    ep = SimpleNamespace(q=np.linspace(0, 0.01, 7), q_sigma=np.full(7, 3e-4), t=np.linspace(0, 0.5, 7))
    big_sigma = np.full(7, 0.02)                       # propagated input noise of a large fitted gain
    m = OE.replay_metrics(ep, ep.q + 0.04, big_sigma)
    assert m["rmse_over_sigma"] <= 3.0 and m["misfit_only_within_propagated_input_noise"] and not m["adequate"]


def test_forced_weak_gain_family_comes_from_the_profile_not_the_prior():
    from forceartgs_g9.system_identification.contracts import ResponseFeature
    from forceartgs_g9.system_identification.transfer_features import physical_mapping
    gain = ResponseFeature("gain", 2.0, "1/(kg m^2)", "NOT_IDENTIFIED", {}, [], ["f"])
    oe = dict(estimate=dict(lam=1.0, gam=0.0), forced_in_training=True)
    sp = dict(status="OPEN", threshold=1.92, ci95_M=[0.1, 2.0], open_low=False, open_high=True,
              points=[dict(M=m, c=c, Fc=0.0) for m, c in ((0.05, 9.0), (0.1, 3.0), (0.5, 1.0), (2.0, 0.4))],
              delta_cost=[10.0, 1.9, 0.0, 1.0])
    mp = physical_mapping(oe, [gain], joint_type="revolute", prior_M_range=[5.0, 50.0], scale_profile=sp)
    assert mp["status"] == "DATA_CONSISTENT_FAMILY"
    assert [p["M"] for p in mp["family"]["members"]] == [0.1, 0.5, 2.0]
    assert mp["family"]["members"][0]["c"] == 3.0                      # its own profiled c, not M * lam*
    assert mp["prior_overlap"]["status"] == "OVERLAP"                  # open above: the prior is not excluded
    sp2 = dict(sp, open_high=False)
    assert physical_mapping(oe, [gain], joint_type="revolute", prior_M_range=[5.0, 50.0],
                            scale_profile=sp2)["prior_overlap"]["status"] == "PRIOR_DATA_CONFLICT"
