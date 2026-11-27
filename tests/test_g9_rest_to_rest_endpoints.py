"""Rest-to-rest episode + endpoint formulas (T01-T03, T05-T07, T17, T18). Truth from the EXACT two-term solution."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from forceartgs_g9.estimators import rest_to_rest as RR
from forceartgs_g9.estimators import rest_to_rest_integral as RI
from forceartgs_g9.estimators.rest_to_rest_convolution import replay, simulate_two_term

from r2r_fixtures import coast_only, episode, exact_rest_to_rest, out_and_back

STRICT = dict(sigma_floor=0.0)          # noise-free truth: the plateau test at (near) machine resolution


@pytest.mark.parametrize("jt,M,c,q0", [("prismatic", 2.4, 6.0, 0.37), ("revolute", 0.35, 0.9, -1.1)])
def test_T01_endpoint_formulas_recover_truth_with_nonzero_q0_and_nonuniform_clock(jt, M, c, q0):
    sy = exact_rest_to_rest(M, c, q0=q0, pushes=((0.3, 4.0 * M),), jitter=0.35, q_sigma=1e-9)
    b = episode(sy, joint_type=jt, **STRICT)
    assert b["status"] == "OK", b["reason"]
    ep = b["episode"]
    assert ep.q0 == pytest.approx(q0, abs=1e-12) and ep.q0 != 0.0          # never re-zeroed
    assert ep.T >= sy["T_true"] - 1e-9                                       # T after the exact stop
    assert np.ptp(np.diff(ep.t_q_s)) > 0                                     # nonuniform visual clock
    e = RI.endpoint_r2r(ep)
    assert e["status"] == "OK" and e["admissible"]
    assert e["c_end"] == pytest.approx(c, rel=1e-8)
    assert e["M_end"] == pytest.approx(M, rel=2e-3)                          # trapezoid int (q - q0) on the clock
    assert e["JQ"] == pytest.approx(c * (sy["q_true"][-1] - q0), rel=1e-8)   # Eq. (2) identity with v(t0) = v(T) = 0


def test_T02_viscous_coast_is_not_exact_rest_and_the_endpoint_error_is_M_vT_over_dq():
    sy = coast_only(M=0.8, c=1.5, coast=4.0)
    assert sy["v_true"][-1] > 0                                              # exponential decay: never exactly zero
    b = episode(sy)
    assert b["status"] == "OK", b["reason"]
    ep = b["episode"]
    vT = replay(ep, 0.8, 1.5)["v"][-1]
    assert vT != 0.0                                                         # the code never asserts vT = 0 here
    e = RI.endpoint_r2r(ep)
    # exact first integral: M [v(T) - 0] + c dq = J_Q  ->  c_end - c = M v(T) / dq (no hidden rest assumption)
    dq_true = simulate_two_term(sy["force_t"], sy["Q"], 0.0, 0.0, 0.8, 1.5, [ep.T])["q"][0] - 0.0
    J = 0.8 * vT + 1.5 * dq_true
    assert e["JQ"] == pytest.approx(J, rel=1e-9)
    # with the plateau-mean anchor qT (the object still creeps inside the plateau): c_end - c = [M v(T) + c (q(T) - qT)]
    # / (qT - q0) -- an explicit, tracked approximation error, not a hidden vT = 0
    expected = (0.8 * vT + 1.5 * (dq_true - (ep.qT - ep.q0))) / ep.delta_q
    assert e["c_end"] - 1.5 == pytest.approx(expected, rel=1e-6, abs=1e-9)
    assert abs(e["c_end"] - 1.5) > 1e-4                                      # visibly not exact


def test_T03_exact_braked_rest_to_rest_closed_form_conditions():
    sy = exact_rest_to_rest(1.3, 2.2, q0=0.05, pushes=((0.24, 3.0), (0.2, 1.0)), q_sigma=1e-9)
    assert abs(sy["v_true"][-1]) < 1e-12                                    # exact stop by the braking rows
    ep = episode(sy, **STRICT)["episode"]
    e = RI.endpoint_r2r(ep)
    assert e["c_end"] == pytest.approx(2.2, rel=1e-8) and e["M_end"] == pytest.approx(1.3, rel=2e-3)
    s = RI.integral_svd_r2r(ep)
    # the double-integral LS integrates the observed q with a trapezoid on the 25 Hz clock (quadrature error only)
    assert s["estimate"]["M"] == pytest.approx(1.3, rel=1e-2) and s["estimate"]["c"] == pytest.approx(2.2, rel=1e-2)


def test_T05_return_to_start_skips_the_endpoint_formula_but_the_multitime_fit_works():
    sy = out_and_back(0.8, 1.5)
    assert abs(sy["q_true"][-1] - sy["q0"]) < 1e-9 and np.ptp(sy["q_true"]) > 0.05
    b = episode(sy, **STRICT)
    assert b["status"] == "OK", b["reason"]
    ep = b["episode"]
    e = RI.endpoint_r2r(ep)
    assert e["status"] == RI.ENDPOINT_ILL and e["c_end"] is None and e["M_end"] is None
    assert "not distinguishable" in e["skipped_reason"]
    s = RI.integral_svd_r2r(ep)
    assert s["estimate"]["M"] == pytest.approx(0.8, rel=2e-2) and s["estimate"]["c"] == pytest.approx(1.5, rel=2e-2)


def test_T06_wrong_sign_or_constant_residual_gives_a_signed_contradiction_never_clipped():
    sy = exact_rest_to_rest(0.8, 1.5, q_sigma=1e-9)
    sy["Q"] = -sy["Q"]                                                       # deliberately wrong force sign
    e = RI.endpoint_r2r(episode(sy, **STRICT)["episode"])
    assert e["c_end"] < 0 and "NEGATIVE_ENDPOINT_DAMPING" in e["flags"] and e["status"] == "MODEL_INPUT_INCONSISTENT"
    sy = exact_rest_to_rest(0.8, 1.5, q_sigma=1e-9, bias=-0.5)              # constant residual wrench
    ep = episode(sy, **STRICT)["episode"]
    e = RI.endpoint_r2r(ep)
    assert e["c_end"] == pytest.approx(1.5 + (-0.5) * (ep.T - ep.t0) / ep.delta_q, rel=1e-6)   # signed, not clipped
    assert e["c_end"] < 0 and e["status"] == "MODEL_INPUT_INCONSISTENT"


def test_T07_terminal_load_is_flagged_and_never_retared():
    sy = exact_rest_to_rest(0.8, 1.5, bias=0.3)
    b = episode(sy)
    assert "UNMODELED_STATIONARY_LOAD" in b["flags"]
    ld = b["evidence"].stationary_load["terminal"]
    assert ld["status"] == RR.UNMODELED_STATIONARY_LOAD and ld["mean_Q"] == pytest.approx(0.3, abs=1e-9)
    ep = b["episode"]
    rows = np.isin(sy["rows"], ep.source_rows_force)
    np.testing.assert_array_equal(ep.force_Q, sy["Q"][rows])                # the recorded values, no second tare
    assert "contact" not in ld["status"].lower()


def test_synthetic_generator_refuses_phases_off_the_row_grid():
    with pytest.raises(ValueError, match="row grid"):
        exact_rest_to_rest(pushes=((0.25, 1.0),))


def test_T17_crops_without_a_plateau_return_exact_not_covered_statuses():
    sy = exact_rest_to_rest(0.8, 1.5)
    n = len(sy["t_q"])
    i_move = int(np.argmax(np.abs(sy["q"] - sy["q"][0]) > 1e-3))
    cut = {k: (v[i_move:] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in sy.items()}
    b = episode(cut)
    assert b["status"] == RR.REST_START_NOT_COVERED and b["episode"] is None
    assert f"source frame {int(cut['frames'][0])}" in b["reason"]
    i_mid = int(np.argmax(sy["t_q"] > sy["T_true"] - 0.15))
    cut = {k: (v[:i_mid] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in sy.items()}
    b = episode(cut)
    assert b["status"] == RR.REST_END_NOT_COVERED and b["episode"] is None
    assert b["evidence"].terminal["missing_frames"].startswith("after source frame")


def _fr(sy, **extra):
    qs = np.full(len(sy["t_q"]), sy["q_sigma"])
    ns = SimpleNamespace(scene="synthetic", joint_type="prismatic", q_t_s=sy["t_q"], q=sy["q"], q_sigma=qs,
                         source_q_frame=sy["frames"], t_s=sy["force_t"], Q_full=sy["Q"],
                         Q_sigma=np.full(len(sy["Q"]), 0.02), source_row=sy["rows"], contract=dict(extra),
                         sha256=lambda: "x")
    for k, v in extra.items():
        setattr(ns, k, v)
    return ns


def test_T18_contact_labels_segments_or_thresholds_cannot_change_the_new_input():
    sy = exact_rest_to_rest(0.8, 1.5)
    a = RR.from_full_wrench_record(_fr(sy))["episode"]
    b = RR.from_full_wrench_record(_fr(sy, contact_segments=[(30, 40)], contact_mode=np.array(["absent"] * len(sy["Q"])),
                                       force_threshold_N=5.0))["episode"]
    np.testing.assert_array_equal(a.force_Q, b.force_Q)
    assert a.sha256() == b.sha256() and (a.t0, a.T) == (b.t0, b.T)
    assert np.count_nonzero(a.force_Q) == np.count_nonzero(sy["Q"][np.isin(sy["rows"], a.source_rows_force)])
    with pytest.raises(TypeError):                                           # the builder has no segment argument
        RR.build_rest_to_rest_episode(scene="s", joint_type="prismatic", t_q=sy["t_q"], q=sy["q"],
                                      q_sigma=np.full(len(sy["t_q"]), 2e-4), q_frames=sy["frames"],
                                      force_t=sy["force_t"], force_Q=sy["Q"], force_sigma=None, force_rows=sy["rows"],
                                      contact_segments=[(1, 2)])


def test_episode_is_immutable_and_hash_checked(tmp_path):
    ep = episode(exact_rest_to_rest(0.8, 1.5))["episode"]
    with pytest.raises(ValueError):
        ep.force_Q[0] = 1.0
    with pytest.raises(Exception):
        ep.q0 = 2.0
    p = tmp_path / "e.npz"
    ep.save_npz(p)
    assert RR.RestToRestEpisode.load_npz(p).sha256() == ep.sha256()
    with pytest.raises(ValueError, match="v0"):
        RR.RestToRestEpisode(**{**{k: getattr(ep, k) for k in ("scene", "joint_type", "t_q_s", "q_obs", "q_sigma",
                                                              "q0", "qT", "t0", "T", "force_t_s", "force_Q")}, "v0": 1e-9})
    pe = ep.to_physx_episode()
    assert pe.v0 == 0.0 and math.copysign(1.0, pe.v0) > 0 and pe.q0 == ep.q0
    np.testing.assert_array_equal(pe.input_effort, ep.force_Q)
