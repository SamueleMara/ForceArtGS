"""Cross-demo pure metrics (spec Section 8): T23, T24, T33-T47 with the hand-computed fixtures of Section 8.6."""
import json
import math

import numpy as np
import pytest

from forceartgs_g9.cross_demo import metrics as CM

A = dict(abs=1e-12)
META = {"status", "metrics_version", "joint_type", "q_units", "weighting", "quantile_rule", "segments", "quartiles",
        "identity_checks", "null_reasons", "units", "sigma_source", "error_slope_se_assumption", "travel_threshold",
        "sigma_floor", "frac_frames_floor_dominated", "t_start_s", "t_end_s"}


def cell(obj, i, j, rmse, jt="prismatic", method="B0", status="OK", **m):
    return {"object": obj, "train_id": i, "test_id": j, "method": method, "seed": 0, "status": status,
            "joint_type": jt, "metrics": dict(rmse=rmse, **m)}


def score(t, e, jt="prismatic", **kw):
    t = np.asarray(t, float)
    q_ref = 0.3 + 0.05 * np.sin(t)                                 # arbitrary reference; only e matters
    return CM.score_trajectory(t, q_ref, q_ref + np.asarray(e, float), joint_type=jt, **kw)


# ----------------------------------------------------------------------------------------------- primitives
def test_trapezoid_weights_irregular_and_rejections():
    w = CM.trapezoid_weights([0.0, 1.0, 3.0])
    assert w.tolist() == [0.5, 1.5, 1.0] and w.sum() == 3.0
    for bad in ([0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, np.nan, 1.0]):
        with pytest.raises(ValueError):
            CM.trapezoid_weights(bad)


def test_weighted_quantile_convention_is_nearest_rank_with_equal_weights():
    rng = np.random.default_rng(3)
    for n in (1, 2, 5, 17):
        x = rng.normal(size=n)
        for p in (0.0, 0.25, 0.5, 0.9, 0.95, 1.0):
            assert CM.weighted_quantile(x, np.ones(n), p) == np.quantile(x, p, method="inverted_cdf")
    # weighted: a=[1,0,1], w=[.5,1,.5] -> C=[1,1.5,2] over sorted [0,1,1]; median = 0 (C_0 = 1 >= 0.5*2)
    assert CM.weighted_quantile([1.0, 0.0, 1.0], [0.5, 1.0, 0.5], 0.5) == 0.0
    assert CM.weighted_quantile([1.0, 0.0, 1.0], [0.5, 1.0, 0.5], 0.51) == 1.0


def test_scoring_failed_on_invalid_data_and_strict_joint_type():
    assert score([0, 1], [0.1, 0.1])["status"] == "SCORING_FAILED"
    assert score([0, 1, 2], [0.1, np.nan, 0.1])["status"] == "SCORING_FAILED"
    assert score([0, 2, 1], [0.1, 0.1, 0.1])["status"] == "SCORING_FAILED"
    with pytest.raises(ValueError):
        score([0, 1, 2], [0, 0, 0], jt="spherical")


# ----------------------------------------------------------------------------------------------- T33-T37
@pytest.mark.parametrize("c", [0.1, -0.1])
def test_T33_constant_signed_residual(c):
    r = score([0.0, 0.4, 1.7, 2.0, 3.1], [c] * 5)
    assert r["status"] == "OK"
    assert r["signed_mean_error"] == pytest.approx(c, **A) and r["bias"] == r["signed_mean_error"]
    assert r["mean_abs_error"] == pytest.approx(0.1, **A) and r["rmse"] == pytest.approx(0.1, **A)
    assert r["variance_error_time"] == pytest.approx(0.0, abs=1e-15)
    assert r["weighted_deviance"] == pytest.approx(0.0, abs=1e-15)
    assert r["sample_deviance"] == pytest.approx(0.0, abs=1e-15)
    assert r["endpoint_error_signed"] == pytest.approx(c, **A) and r["initial_q_error_signed"] == pytest.approx(c, **A)
    assert r["drift_end_minus_start"] == pytest.approx(0.0, **A) and r["error_slope_per_s"] == pytest.approx(0.0, **A)


def test_T34_alternating_residual_bias_vs_dispersion_and_uniform_minus1_0_plus1():
    r = score([0, 1, 2, 3, 4], [0.5, 0.1, 0.5, 0.1, 0.5])           # w = [.5,1,1,1,.5], W = 4
    assert r["signed_mean_error"] == pytest.approx(1.2 / 4, **A)       # 0.3, not the 0.34 frame mean
    assert r["variance_error_time"] == pytest.approx(0.04, **A) and r["std_error_time"] == pytest.approx(0.2, **A)
    assert r["mse"] == pytest.approx(0.13, **A)
    assert r["rmse"] ** 2 == pytest.approx(r["bias"] ** 2 + r["variance_error_time"], rel=1e-12)
    assert r["samplewt_bias"] == pytest.approx(0.34, **A)
    # e = [-1, 0, +1] on t = [0, 1, 2]: w = [.5, 1, .5]
    r = score([0, 1, 2], [-1.0, 0.0, 1.0])
    assert r["signed_mean_error"] == pytest.approx(0.0, **A)
    assert r["mse"] == pytest.approx(0.5, **A) and r["rmse"] == pytest.approx(math.sqrt(0.5), **A)
    assert r["variance_error_time"] == pytest.approx(0.5, **A) and r["weighted_deviance"] == pytest.approx(1.0, **A)
    assert r["mean_abs_error"] == pytest.approx(0.5, **A)
    assert r["iae"] == pytest.approx(1.0, **A) and r["ise"] == pytest.approx(1.0, **A)
    assert r["median_abs_error"] == pytest.approx(0.0, **A) and r["p90_abs_error"] == pytest.approx(1.0, **A)
    assert r["max_abs_error"] == pytest.approx(1.0, **A)
    assert r["peak_error_frame"] == 0 and r["peak_error_time_s"] == 0.0           # tie -> smallest index
    assert r["sample_deviance"] == pytest.approx(2.0, **A)
    assert r["sample_variance"] == pytest.approx(1.0, **A) and r["sample_std"] == pytest.approx(1.0, **A)


def test_T35_irregular_timestamps_trapezoid_vs_frame_weighted():
    t = np.array([0.0, 1.0, 3.0])
    r = CM.score_trajectory(t, np.zeros(3), t.copy(), joint_type="prismatic")      # e(t) = t, piecewise linear
    assert r["iae"] == pytest.approx(4.5, **A)                                       # exact int_0^3 t dt
    assert r["signed_mean_error"] == pytest.approx(1.5, **A)                         # exact time mean of t on [0, 3]
    assert r["ise"] == pytest.approx(0.5 * 1 * (0 + 1) + 0.5 * 2 * (1 + 9), **A)     # trapezoid of e^2 = 10.5
    assert r["mse"] == pytest.approx(3.5, **A)
    assert r["samplewt_bias"] == pytest.approx(4 / 3, **A) and r["samplewt_mse"] == pytest.approx(10 / 3, **A)
    assert r["samplewt_bias"] != pytest.approx(r["signed_mean_error"])
    assert r["error_slope_per_s"] == pytest.approx(1.0, **A)                         # e = t exactly
    assert r["drift_end_minus_start"] == pytest.approx(2.85 - 0.15, **A)              # mean_[2.7,3] t - mean_[0,.3] t


def test_T36_sample_deviance_is_not_time_weighted_population_variance():
    r = score([0, 1, 2], [-1.0, 0.0, 1.0])
    assert r["sample_variance"] == pytest.approx(1.0, **A)                     # D/(K-1) = 2/2
    assert r["samplewt_variance_population"] == pytest.approx(2 / 3, **A)       # D/K
    assert r["variance_error_time"] == pytest.approx(0.5, **A)                  # time-weighted population
    assert r["units"]["sample_deviance"] == "m^2" and r["units"]["weighted_deviance"] == "m^2*s"


@pytest.mark.parametrize("jt,u", [("prismatic", "m"), ("revolute", "rad")])
def test_T37_iae_ise_identities_and_units(jt, u):
    rng = np.random.default_rng(7)
    t = np.cumsum(rng.uniform(0.01, 0.2, 200))
    e = rng.normal(0.02, 0.05, 200)
    r = score(t, e, jt=jt)
    d = r["duration_s"]
    assert d == pytest.approx(t[-1] - t[0], rel=1e-15)
    assert r["mse"] * d == pytest.approx(r["ise"], rel=1e-10)
    assert r["mean_abs_error"] * d == pytest.approx(r["iae"], rel=1e-10)
    assert r["weighted_deviance"] / d == pytest.approx(r["variance_error_time"], rel=1e-10)
    assert r["rmse"] ** 2 == pytest.approx(r["bias"] ** 2 + r["variance_error_time"], rel=1e-10)
    assert set(r["identity_checks"]) == {"rmse2_eq_bias2_plus_variance", "mse_times_duration_eq_ise",
                                         "mae_times_duration_eq_iae", "weighted_deviance_over_duration_eq_variance"}
    U = r["units"]
    assert (r["q_units"], U["rmse"], U["iae"], U["ise"], U["mse"]) == (u, u, f"{u}*s", f"{u}^2*s", f"{u}^2")
    assert U["error_slope_per_s"] == f"{u}/s" and U["rmse_over_sigma"] == "1"
    with pytest.raises(CM.MetricIdentityError):
        CM._check("bad", 1.0, 1.0 + 1e-6, 1.0, {})


def test_every_scalar_field_has_a_definition_and_null_has_a_reason():
    r = score(np.linspace(0, 2, 21), np.linspace(0, 0.1, 21), q_sigma=0.01, qdot_sim=np.zeros(21),
              travel_threshold=1.0)
    for k, v in r.items():
        if k in META:
            continue
        assert k in CM.METRIC_DEFINITIONS, k
        if v is None:
            assert k in r["null_reasons"], k
    json.dumps(CM.METRIC_DEFINITIONS)
    assert all({"formula", "unit", "dispersion"} <= set(d) for d in CM.METRIC_DEFINITIONS.values())


def test_sigma_normalised_uses_q_sigma_and_floor():
    r = score([0, 1, 2], [0.3, 0.3, 0.3], q_sigma=[0.3, 0.4, 0.0], sigma_floor=0.4)
    s = np.sqrt(np.array([0.3, 0.4, 0.0]) ** 2 + 0.16)                  # [0.5, 0.5657, 0.4]
    z = 0.3 / s
    assert r["bias_over_sigma"] == pytest.approx((0.5 * z[0] + z[1] + 0.5 * z[2]) / 2, rel=1e-12)
    assert r["sigma_source"] == "q_sigma+floor" and r["frac_frames_floor_dominated"] == 1.0     # 0.4 >= every q_sigma
    r = score([0, 1, 2], [0.3, 0.3, 0.3])
    assert r["rmse_over_sigma"] is None and r["null_reasons"]["rmse_over_sigma"] == "NO_SIGMA"


def test_zero_motion_target_has_no_finite_nrmse():
    t = np.linspace(0, 3, 31)
    r = CM.score_trajectory(t, np.full(31, 0.2), np.full(31, 0.2) + 0.01 * t, joint_type="prismatic",
                            travel_threshold=0.0)
    assert r["travel_span"] == 0.0
    assert r["normalized_rmse_travel"] is None
    assert r["null_reasons"]["normalized_rmse_travel"] == "NOT_DEFINED_LOW_TRAVEL"
    assert r["displacement_relative_error"] is None
    r = CM.score_trajectory(t, np.full(31, 0.2), np.full(31, 0.2), joint_type="prismatic")
    assert r["null_reasons"]["normalized_rmse_travel"] == "NOT_DEFINED_NO_TRAVEL_THRESHOLD"
    r = CM.score_trajectory(t, 0.1 * t, 0.1 * t + 0.03, joint_type="prismatic", travel_threshold=0.05)
    assert r["normalized_rmse_travel"] == pytest.approx(0.03 / 0.3, rel=1e-10)
    assert r["displacement_relative_error"] == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------------------------------- T38 / T47
def _run(K=61):
    t = np.linspace(0.0, 6.0, K) ** 1.1                                  # irregular clock
    q_ref = np.interp(t, [0, 1.5, 4.0, t[-1]], [0.2, 0.2, 0.5, 0.5])
    q_sim = np.interp(t, [0, 1.6, 4.3, t[-1]], [0.2, 0.2, 0.53, 0.55])
    qd = np.gradient(q_sim, t)
    return t, q_ref, q_sim, qd


def test_T38_segments_and_quartiles_never_change_full_run_score():
    t, qr, qs, qd = _run()
    qs0 = qs.copy()
    a = CM.score_trajectory(t, qr, qs, qdot_sim=qd, joint_type="prismatic")
    b = CM.score_trajectory(t, qr, qs, qdot_sim=qd, joint_type="prismatic",
                            segments={"t_motion_start": 1.5, "t_stop": 4.0}, qdot_stop_threshold=1e-6)
    assert np.array_equal(qs, qs0)
    for k, v in a.items():
        if k not in ("segments", "quartiles", "null_reasons", "units") and not k.startswith("terminal_") \
                and k != "settling_time_error_s":
            assert b[k] == v, k
    assert all(s["status"] == "SEGMENT_NOT_ESTABLISHED" for s in a["segments"].values())
    segs = b["segments"]
    assert all(segs[s]["status"] == "OK" for s in CM.SEGMENTS)
    W = b["duration_s"]
    assert sum(segs[s]["duration_s"] for s in CM.SEGMENTS) == pytest.approx(W, rel=1e-12)
    assert sum(segs[s]["n_samples"] for s in CM.SEGMENTS) == len(t)
    assert sum(q["duration_s"] for q in b["quartiles"]) == pytest.approx(W, rel=1e-12)
    ise_bins = sum(q["rmse"] ** 2 * q["duration_s"] for q in b["quartiles"])
    assert ise_bins == pytest.approx(b["ise"], rel=1e-10)                 # bins partition the integral exactly
    assert [q["t_start_s"] for q in b["quartiles"]] == pytest.approx(list(t[0] + W * np.array([0, .25, .5, .75])))
    assert segs["INITIAL_REST"]["bias"] == pytest.approx(0.0, abs=1e-12)  # both flat at 0.2 before t = 1.5
    assert b["terminal_plateau_bias"] == segs["TERMINAL_REST"]["bias"]
    assert b["terminal_qdot_sim_max_abs"] == pytest.approx(np.abs(qd[t >= 4.0]).max())
    assert b["settling_time_error_s"] is None                             # sim still creeping 0.53 -> 0.55


def test_T47_stationary_tail_scored_through_T_eval_end_and_missing_tail_flagged():
    t, qr, qs, qd = _run()
    qs0 = qs.copy()
    sc = CM.score_scopes(t, qr, qs, t_stop=4.0, tail_input_complete=True, qdot_sim=qd, joint_type="prismatic")
    stop, full = sc["T_STOP"], sc["T_EVAL_END"]
    assert sc["primary"] == "T_EVAL_END" and sc["coverage"] == "FULL"
    assert full["n_valid"] == len(t) and full["t_end_s"] == t[-1]
    assert stop["n_valid"] == int(np.sum(t <= 4.0)) < len(t) and stop["t_end_s"] <= 4.0
    assert full["segments"]["TERMINAL_REST"]["status"] == "OK"
    assert stop["segments"]["TERMINAL_REST"]["status"] == "SEGMENT_NOT_ESTABLISHED"   # never from the T_stop frame
    assert full["rmse"] != stop["rmse"]                                                 # tail actually scored
    whole = CM.score_trajectory(t, qr, qs, qdot_sim=qd, joint_type="prismatic")
    assert full["rmse"] == whole["rmse"] and np.array_equal(qs, qs0)
    sc = CM.score_scopes(t, qr, qs, t_stop=4.0, tail_input_complete=False, qdot_sim=qd, joint_type="prismatic")
    assert sc["T_EVAL_END"] == {"status": "TAIL_INPUT_INCOMPLETE"}
    assert sc["primary"] == "T_STOP" and sc["coverage"] == "REDUCED_TAIL_INPUT_INCOMPLETE"
    assert sc["T_STOP"]["rmse"] == stop["rmse"]


# ----------------------------------------------------------------------------------------------- aggregates
def test_aggregate_scalars_conventions():
    s = CM.aggregate_scalars([1.0, 2.0, 3.0, 4.0, None, float("nan")])
    assert (s["n"], s["n_invalid"]) == (4, 2)
    assert s["mean"] == 2.5 and s["deviance"] == pytest.approx(5.0)
    assert s["var_sample"] == pytest.approx(5 / 3) and s["var_population"] == pytest.approx(1.25)
    assert s["sd_sample"] == pytest.approx(math.sqrt(5 / 3))
    assert (s["min"], s["p50"], s["p90"], s["p95"], s["max"]) == (1.0, 2.0, 4.0, 4.0, 4.0)
    s = CM.aggregate_scalars([0.7])
    assert s["var_sample"] is None and s["sd_sample"] is None and s["null_reasons"]["sd_sample"] == "INSUFFICIENT_N"
    assert s["var_population"] == 0.0 and s["status"] == "OK"
    s = CM.aggregate_scalars([None, float("nan")])
    assert s["status"] == "NO_VALID_RUNS" and s["mean"] is None and s["n_invalid"] == 2


R3 = np.array([[0.10, 0.40, 0.70],
               [0.20, 0.15, 0.90],
               [0.50, 0.30, 0.12]])                     # rows = train i, cols = test j (asymmetric)


def _matrix_cells(R, obj="obj", jt="prismatic", ids=None):
    ids = ids or [f"e{k}" for k in range(len(R))]
    return [cell(obj, ids[i], ids[j], float(R[i, j]), jt=jt, signed_mean_error=float(R[i, j]) * (-1) ** (i + j))
            for i in range(len(R)) for j in range(len(R))]


def test_T40_asymmetric_3x3_row_column_object_summaries():
    m = CM.aggregate_matrix(_matrix_cells(R3))
    assert m["status"] == "OK" and m["joint_type"] == "prismatic" and m["q_units"] == "m"
    assert m["diagonal"]["counts"]["n_success"] == 3 and m["offdiag"]["counts"]["n_success"] == 6
    assert len(m["rows"]) == 3 and len(m["columns"]) == 3
    rows = {"e0": (0.55, 0.045, 0.10), "e1": (0.55, 0.245, 0.15), "e2": (0.40, 0.020, 0.12)}
    for i, (mu, D, d) in rows.items():
        s = m["rows"][i]["metrics"]["rmse"]
        assert (s["n"], s["mean"], s["deviance"], s["var_sample"]) == (2, pytest.approx(mu), pytest.approx(D),
                                                                       pytest.approx(D))
        assert s["sd_sample"] == pytest.approx(math.sqrt(D))
        assert m["rows"][i]["train_diagonal"]["rmse"] == d
        assert m["rows"][i]["generalization_gap_rmse"] == pytest.approx(mu - d)
    cols = {"e0": (0.35, 0.045), "e1": (0.35, 0.005), "e2": (0.80, 0.020)}
    for j, (mu, D) in cols.items():
        s = m["columns"][j]["metrics"]["rmse"]
        assert (s["mean"], s["deviance"]) == (pytest.approx(mu), pytest.approx(D))
        assert m["columns"][j]["n_distinct_train"] == 2
    off = m["offdiag"]["metrics"]["rmse"]
    assert (off["n"], off["mean"], off["deviance"], off["var_sample"]) == (6, pytest.approx(0.5), pytest.approx(0.34),
                                                                           pytest.approx(0.068))
    assert m["diagonal"]["metrics"]["rmse"]["mean"] == pytest.approx(0.37 / 3)    # never blended with offdiag
    assert m["generalization_gap_rmse"] == pytest.approx(0.5 - 0.37 / 3)
    ab = m["rows"]["e0"]["metrics"]["abs_signed_mean_error"]
    assert ab["mean"] == pytest.approx(0.55) and m["rows"]["e0"]["metrics"]["signed_mean_error"]["mean"] == \
        pytest.approx((-0.40 + 0.70) / 2)


def test_T23_four_episode_matrix_has_16_entries():
    R = np.arange(16, dtype=float).reshape(4, 4) / 10 + 0.1
    m = CM.aggregate_matrix(_matrix_cells(R))
    assert m["diagonal"]["counts"]["n_expected"] == 4 and m["offdiag"]["counts"]["n_expected"] == 12
    assert m["diagonal"]["counts"]["n_success"] + m["offdiag"]["counts"]["n_success"] == 16


def test_T24_single_episode_object_is_insufficient_and_excluded():
    one = [cell("solo", "e0", "e0", 0.1)]
    m = CM.aggregate_matrix(one)
    assert m["status"] == "INSUFFICIENT_EPISODES" and "offdiag" not in m and "rows" not in m
    sw = CM.aggregate_sweep(one + _matrix_cells(R3))
    assert sw["insufficient_episodes"] == ["solo"]
    assert sw["by_joint_type"]["prismatic"]["n_objects"] == 1
    assert sw["by_joint_type"]["prismatic"]["metrics"]["rmse"]["micro_pair"]["n"] == 6


def test_T39_micro_time_is_duration_weighted_not_mean_of_run_rmse():
    runs = []
    for (i, j), (T, c) in {("a", "b"): (1.0, 1.0), ("b", "a"): (3.0, 2.0)}.items():
        t = np.linspace(0.0, T, 11)
        r = CM.score_trajectory(t, np.zeros(11), np.full(11, c), joint_type="revolute")
        runs.append({"object": "o", "train_id": i, "test_id": j, "method": "B0", "seed": 0, "status": "OK",
                     "joint_type": "revolute", "metrics": r})
    g = CM.aggregate_sweep(runs)["by_joint_type"]["revolute"]
    assert g["metrics"]["rmse"]["micro_pair"]["mean"] == pytest.approx(1.5)          # mean of run RMSEs
    assert g["micro_time"]["rmse_time_weighted"] == pytest.approx(math.sqrt((1 + 4 * 3) / 4))
    assert g["micro_time"]["mae_time_weighted"] == pytest.approx((1 + 6) / 4)
    assert "DURATION-WEIGHTED" in g["micro_time"]["label"]
    assert g["pooled_residual_deviance"]["rmse_sweep"] == pytest.approx(g["micro_time"]["rmse_time_weighted"])


def test_T41_unequal_object_counts_macro_vs_micro_weights():
    A2 = np.array([[0.0, 1.0], [3.0, 0.0]])
    B3 = np.array([[0.0, 2.0, 2.0], [5.0, 0.0, 5.0], [8.0, 8.0, 0.0]])
    cells = [c for c in _matrix_cells(A2, "A") + _matrix_cells(B3, "B") if c["train_id"] != c["test_id"]]
    g = CM.aggregate_sweep(cells)["by_joint_type"]["prismatic"]
    r = g["metrics"]["rmse"]
    assert r["micro_pair"]["mean"] == pytest.approx(34 / 8)
    assert r["macro_object"]["mean"] == pytest.approx((2.0 + 5.0) / 2) and r["macro_object"]["n_objects"] == 2
    assert r["macro_train"]["mean"] == pytest.approx((1 + 3 + 2 + 5 + 8) / 5) and r["macro_train"]["n_train_demos"] == 5
    w = g["pair_weights"]
    assert w["micro_pair"][("A", "e0", "e1")] == pytest.approx(1 / 8)
    assert w["macro_object"][("A", "e0", "e1")] == pytest.approx(1 / 4)
    assert w["macro_object"][("B", "e0", "e1")] == pytest.approx(1 / 12)
    assert w["macro_train"][("A", "e1", "e0")] == pytest.approx(1 / 5)
    assert w["macro_train"][("B", "e2", "e0")] == pytest.approx(1 / 10)
    val = {(c["object"], c["train_id"], c["test_id"]): c["metrics"]["rmse"] for c in cells}
    for scheme in ("micro_pair", "macro_object", "macro_train"):
        assert sum(w[scheme].values()) == pytest.approx(1.0)
        assert sum(w[scheme][k] * val[k] for k in val) == pytest.approx(r[scheme]["mean"])


def test_T42_no_raw_unit_cross_type_pooling():
    pr = _matrix_cells(R3, "drawer", "prismatic")
    rv = _matrix_cells(R3 * 2, "fridge", "revolute")
    for c in pr + rv:
        c["metrics"]["normalized_rmse_travel"] = c["metrics"]["rmse"] / 2
    pr[1]["metrics"]["normalized_rmse_travel"] = None                    # NOT_DEFINED_LOW_TRAVEL target
    sw = CM.aggregate_sweep(pr + rv)
    assert set(sw["by_joint_type"]) == {"prismatic", "revolute"} and sw["raw_cross_type"] == "REFUSED_MIXED_UNITS"
    assert sw["by_joint_type"]["prismatic"]["metrics"]["rmse"]["unit"] == "m"
    assert sw["by_joint_type"]["revolute"]["metrics"]["rmse"]["unit"] == "rad"
    assert "rmse" not in sw["normalized_cross_type"] and "mean_abs_error" not in sw["normalized_cross_type"]
    n = sw["normalized_cross_type"]["normalized_rmse_travel"]
    assert n["joint_types_included"] == ["prismatic", "revolute"] and "travel_threshold" in n["normalization"]
    assert (n["n_success_pairs"], n["n_eligible_values"]) == (12, 11)
    with pytest.raises(ValueError, match="units"):
        CM.pooled_residual_deviance([dict(joint_type="prismatic", duration_s=1, signed_mean_error=0,
                                          weighted_deviance=0),
                                     dict(joint_type="revolute", duration_s=1, signed_mean_error=0,
                                          weighted_deviance=0)])
    mixed = _matrix_cells(R3, "x", "prismatic")
    mixed[1]["joint_type"] = "revolute"
    with pytest.raises(ValueError, match="joint types"):
        CM.aggregate_matrix(mixed)


def test_T43_failure_inclusive_coverage():
    cells = _matrix_cells(R3)
    by = {(c["train_id"], c["test_id"]): c for c in cells}
    del by[("e0", "e2")]                                                  # missing
    by[("e1", "e0")]["status"] = "REPLAY_FAILED"
    by[("e1", "e2")]["metrics"]["rmse"] = float("nan")                    # corrupt OK row
    by[("e2", "e0")]["status"] = "KINEMATICS_TYPE_MISMATCH"               # rejected before replay
    by[("e1", "e1")]["status"] = "FIT_FAILED"                             # finite rmse but never a success
    m = CM.aggregate_matrix(list(by.values()), episodes=["e0", "e1", "e2"])
    c = m["offdiag"]["counts"]
    assert (c["n_expected"], c["n_listed"], c["n_missing"], c["n_ineligible"], c["n_eligible"], c["n_attempted"],
            c["n_success"], c["n_failed"]) == (6, 5, 1, 1, 5, 4, 2, 2)
    assert c["coverage_fraction"] == pytest.approx(2 / 6) and c["coverage_of_eligible"] == pytest.approx(2 / 5)
    assert c["failure_categories"] == {"CORRUPT_METRICS": 1, "INELIGIBLE:KINEMATICS_TYPE_MISMATCH": 1, "MISSING": 1,
                                       "REPLAY_FAILED": 1}
    s = m["offdiag"]["metrics"]["rmse"]
    assert s["n"] == 2 and s["mean"] == pytest.approx((0.40 + 0.30) / 2)
    assert m["diagonal"]["counts"]["n_success"] == 2
    assert m["diagonal"]["counts"]["failure_categories"] == {"FIT_FAILED": 1}
    assert m["rows"]["e1"]["generalization_gap_rmse"] is None
    assert m["rows"]["e1"]["null_reasons"]["generalization_gap_rmse"] == "NO_SUCCESSFUL_HELDOUT"
    assert m["rows"]["e0"]["counts"]["n_missing"] == 1


def test_T44_paired_comparison_uses_identical_ordered_pairs():
    X = [cell("o", "e0", "e1", 0.1, method="X"), cell("o", "e1", "e0", 0.3, method="X"),
         cell("o", "e0", "e2", 0.8, method="X"), cell("o", "e2", "e0", 9.9, method="X", status="REPLAY_FAILED")]
    B = [cell("o", "e0", "e1", 0.2, method="B0"), cell("o", "e1", "e0", 0.2, method="B0"),
         cell("o", "e2", "e0", 0.05, method="B0"), cell("o", "e1", "e1", 0.01, method="B0")]
    for c in X + B:
        c["metrics"]["mean_abs_error"] = c["metrics"]["rmse"] / 2
    with pytest.raises(ValueError, match="mismatched"):
        CM.paired_method_comparison(X + B, "X", "B0", on_mismatch="raise")
    p = CM.paired_method_comparison(X + B, "X", "B0")
    assert p["status"] == "INTERSECTED_MISMATCHED_SETS"
    cov = p["coverage"]
    assert (cov["n_method_valid"], cov["n_baseline_valid"], cov["n_union"], cov["n_paired"]) == (3, 3, 4, 2)
    assert cov["coverage_fraction"] == 0.5
    d = p["by_joint_type"]["prismatic"]["delta_rmse"]
    assert d["n"] == 2 and d["mean"] == pytest.approx(0.0, abs=1e-15)
    assert d["sd_sample"] == pytest.approx(math.sqrt(0.02))
    assert d["median"] == pytest.approx(-0.1) and (d["n_improved"], d["n_worse"]) == (1, 1)
    assert p["by_joint_type"]["prismatic"]["delta_mae"]["mean"] == pytest.approx(0.0, abs=1e-15)
    assert d["percent_improvement"]["mean"] == pytest.approx((50.0 - 50.0) / 2)
    unrelated = np.mean([0.1, 0.3, 0.8]) - np.mean([0.2, 0.2, 0.05])
    assert unrelated == pytest.approx(0.25) and d["mean"] != pytest.approx(unrelated)
    m = CM.paired_method_comparison([c for c in X + B if c["test_id"] != "e2" and c["train_id"] != "e2"
                                     and c["status"] == "OK"], "X", "B0", on_mismatch="raise")
    assert m["status"] == "MATCHED" and m["coverage"]["n_paired"] == 2


def test_T45_reaggregation_from_residuals_reproduces_summaries(tmp_path):
    rng = np.random.default_rng(11)
    cells = []
    for i, j in [("e0", "e1"), ("e1", "e0"), ("e0", "e0")]:
        t = np.cumsum(rng.uniform(0.02, 0.1, 50))
        qr = np.cumsum(rng.normal(0, 0.01, 50))
        qs = qr + rng.normal(0.01, 0.02, 50)
        arr = CM.residual_arrays(t, qr, qs, q_sigma=0.01, segments={"t_motion_start": t[5], "t_stop": t[40]})
        f = tmp_path / f"{i}_{j}.npz"
        np.savez(f, **arr)
        r = CM.score_trajectory(t, qr, qs, q_sigma=0.01, joint_type="prismatic")
        cells.append((cell("o", i, j, None), f, r))
    orig, again = [], []
    for c, f, r in cells:
        z = np.load(f)
        assert z["w_time"] @ z["e_abs"] == pytest.approx(r["iae"], rel=1e-12)
        assert np.sqrt(z["w_time"] @ z["z_normalized"] ** 2 / z["w_time"].sum()) == pytest.approx(r["rmse_over_sigma"])
        r2 = CM.score_trajectory(z["t_s"], z["q_reference"], z["q_sim"], q_sigma=0.01, joint_type="prismatic")
        assert r2["rmse"] == r["rmse"] and r2["p95_abs_error"] == r["p95_abs_error"]
        orig.append(dict(c, metrics=r))
        again.append(dict(c, metrics=r2))
    a, b = CM.aggregate_matrix(orig), CM.aggregate_matrix(again)
    assert a["offdiag"]["metrics"] == b["offdiag"]["metrics"] and a["offdiag"]["counts"] == b["offdiag"]["counts"]


def test_T46_law_of_total_weighted_deviance():
    rng = np.random.default_rng(5)
    runs, T_all, E_all = [], [], []
    for k, bias in enumerate([0.05, -0.02, 0.10]):
        t = np.cumsum(rng.uniform(0.01, 0.05, 40 + 10 * k))
        e = bias + rng.normal(0, 0.01, t.size)
        r = CM.score_trajectory(t, np.zeros_like(t), e, joint_type="prismatic")
        runs.append(r)
        T_all.append(CM.trapezoid_weights(t))
        E_all.append(e)
    p = CM.pooled_residual_deviance(runs)
    w, e = np.concatenate(T_all), np.concatenate(E_all)
    b = w @ e / w.sum()
    assert p["b_sweep"] == pytest.approx(b, rel=1e-12)
    assert p["D_sweep"] == pytest.approx(w @ (e - b) ** 2, rel=1e-10)              # direct pooled timelines
    assert p["D_sweep"] == pytest.approx(p["deviance_within_runs"] + p["deviance_between_run_biases"], rel=1e-14)
    assert p["rmse_sweep"] ** 2 == pytest.approx(p["b_sweep"] ** 2 + p["var_sweep"], rel=1e-12)
    assert p["rmse_sweep"] ** 2 == pytest.approx(sum(r["ise"] for r in runs) / p["W_tot_s"], rel=1e-10)
    assert p["fraction_within"] + p["fraction_between"] == pytest.approx(1.0)
    same = [{k: r[k] for k in ("joint_type", "duration_s", "weighted_deviance")} | {"signed_mean_error": 0.04}
            for r in runs]                                                          # equal biases -> no between
    assert CM.pooled_residual_deviance(same)["deviance_between_run_biases"] == pytest.approx(0.0, abs=1e-18)
    assert p["deviance_between_run_biases"] > 0


def test_parameter_variation_cv_and_rejections():
    models = [{"object": "o", "train_id": f"e{k}", "joint_type": "prismatic", "M": M, "c": 0.0}
              for k, M in enumerate([2.0, 4.0, 6.0])]
    models.append({"object": "o", "train_id": "e9", "joint_type": "prismatic", "status": "FIT_FAILED", "M": 99.0,
                   "c": 1.0})
    models.append({"object": "o", "train_id": "e8", "joint_type": "prismatic", "M": float("nan"), "c": 1.0})
    p = CM.parameter_variation(models)
    assert p["n_valid"] == 3 and p["rejected"] == {"FIT_FAILED": 1, "NONFINITE_OR_OUT_OF_BOUNDS": 1}
    assert (p["M"]["mean"], p["M"]["sd_sample"], p["M"]["cv"], p["M"]["range"]) == (4.0, 2.0, 0.5, 4.0)
    assert p["c"]["cv"] is None and p["null_reasons"]["c.cv"] == "CV_UNDEFINED_NEAR_ZERO_MEAN"
    assert p["units"] == {"M": "kg", "c": "N*s/m"} and p["null_reasons"]["corr_M_c"] == "ZERO_VARIANCE"
    one = CM.parameter_variation(models[:1])
    assert one["M"]["cv"] is None and one["null_reasons"]["M.cv"] == "INSUFFICIENT_N"
