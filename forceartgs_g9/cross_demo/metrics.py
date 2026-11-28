"""Cross-demo trajectory metrics and failure-inclusive aggregation (spec Section 8). PURE numpy, no PhysX/optimizer.

Per replay r = (object, train i, test j, method, seed) on the target's immutable visual grid t_0 < ... < t_(K-1):

* signed error e_k = q_sim,k - q_ref,k (never abs-valued before the bias), a_k = |e_k|; units m (prismatic) or rad
  (revolute), never pooled;
* trapezoid / Voronoi time weights w_k (``trapezoid_weights``), W = sum w = t_(K-1) - t_0, mean_w(f) = sum w f / W.
  Time-weighted metrics are PRIMARY; sample-weighted legacy metrics carry the ``samplewt_`` prefix;
* quantiles (QUANTILE_RULE): deterministic weighted inverse-CDF ("weighted nearest rank"): sort x ascending (stable),
  C_k = cumulative weight, Q(p) = the smallest sorted x_(k) with C_k >= p * sum(w) (1e-12 relative slack against
  float round-off). Always an observed value; with equal weights it IS nearest-rank = numpy method="inverted_cdf".
  Trajectory quantiles use w_k; scalar run-metric aggregates use equal weights (the same rule);
* undefined quantities are ``None`` with the reason in ``null_reasons[field]`` (never an epsilon division, never NaN
  rendered as a value). Invalid data -> ``status = SCORING_FAILED``.

Aggregation (``aggregate_scalars``): mean = sum x / n, deviance D = sum (x - mean)^2 (devianza), var_sample = D/(n-1)
(null + INSUFFICIENT_N on n = 1), sd_sample = sqrt(var_sample), var_population = D/n; n = 0 -> NO_VALID_RUNS.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

METRICS_VERSION = "forceartgs.cross_demo.metrics/v1"
QUANTILE_RULE = "weighted_inverted_cdf"   # smallest x_(k) with cumulative weight >= p*total; equal w = nearest rank
UNITS = {"prismatic": "m", "revolute": "rad"}
PARAM_UNITS = {"prismatic": {"M": "kg", "c": "N*s/m"}, "revolute": {"M": "kg*m^2", "c": "N*m*s/rad"}}
SUCCESS_STATUSES = frozenset({"OK", "VALID_TRANSFER"})
INELIGIBLE_STATUSES = frozenset({"KINEMATICS_TYPE_MISMATCH", "Q_REFERENCE_ALIGNMENT_UNAVAILABLE",
                                 "INPUT_COVERAGE_INADEQUATE", "INITIAL_REST_NOT_COVERED"})   # + any "INELIGIBLE_*"
DIMENSIONLESS = ("normalized_rmse_travel", "normalized_mae_travel", "rmse_over_sigma", "mae_over_sigma",
                 "bias_over_sigma")
NORMALIZATION = {
    "normalized_rmse_travel": "rmse / travel_span of the TARGET reference; valid only where travel_span > the "
                              "predeclared travel_threshold (else NOT_DEFINED_LOW_TRAVEL, excluded)",
    "normalized_mae_travel": "mean_abs_error / travel_span of the TARGET reference; same eligibility as NRMSE",
    "rmse_over_sigma": "sqrt(mean_w(z^2)), z = e / sqrt(q_sigma^2 + sigma_floor^2); caveat: q_sigma is a visual "
                       "uncertainty estimate, not a certified encoder noise",
    "mae_over_sigma": "mean_w(|z|); same sigma caveat", "bias_over_sigma": "mean_w(z); same sigma caveat"}
DEFAULT_AGG_METRICS = ("rmse", "mean_abs_error", "signed_mean_error", "abs_signed_mean_error", "std_error_time",
                       "endpoint_error_abs", "p95_abs_error", "normalized_rmse_travel", "rmse_over_sigma")
MIN_CI_OBJECTS = 5
_REL_TOL = 1e-10


class MetricIdentityError(ArithmeticError):
    """An exact algebraic identity of Section 8.2 / 8.4 failed beyond float tolerance (a metric bug, never data)."""


# ----------------------------------------------------------------------------------------------- definitions
def _d(formula, unit, dispersion=None, weighting="time"):
    return {"formula": formula, "unit": unit, "dispersion": dispersion, "weighting": weighting}


_TIME, _SAMP = "time_within_trajectory", "frames_within_trajectory"
METRIC_DEFINITIONS = {
    # ---- per replay, time-weighted (primary); unit templates use {u} = m | rad
    "signed_mean_error": _d("b_e = mean_w(e), e = q_sim - q_ref", "{u}"),
    "bias": _d("alias of signed_mean_error", "{u}"),
    "mean_abs_error": _d("mean_w(|e|) (NOT a signed mean)", "{u}"),
    "mse": _d("mean_w(e^2)", "{u}^2"),
    "rmse": _d("sqrt(mean_w(e^2))", "{u}"),
    "variance_error_time": _d("mean_w((e - b_e)^2); temporal POPULATION variance", "{u}^2", _TIME),
    "std_error_time": _d("sqrt(variance_error_time); temporal population SD", "{u}", _TIME),
    "weighted_deviance": _d("sum_k w_k (e_k - b_e)^2; time-integrated centred sum of squares", "{u}^2*s", _TIME),
    "iae": _d("sum_k w_k |e_k|", "{u}*s"),
    "ise": _d("sum_k w_k e_k^2", "{u}^2*s"),
    "median_abs_error": _d("weighted quantile p=0.50 of |e| with w (QUANTILE_RULE)", "{u}"),
    "p90_abs_error": _d("weighted quantile p=0.90 of |e| with w (QUANTILE_RULE)", "{u}"),
    "p95_abs_error": _d("weighted quantile p=0.95 of |e| with w (QUANTILE_RULE)", "{u}"),
    "p99_abs_error": _d("weighted quantile p=0.99 of |e| with w (QUANTILE_RULE)", "{u}"),
    "max_abs_error": _d("max_k |e_k| (sample maximum)", "{u}", weighting="none"),
    "peak_error_frame": _d("smallest k with |e_k| = max |e|", "frame index", weighting="none"),
    "peak_error_time_s": _d("t_k at peak_error_frame", "s", weighting="none"),
    "rmse_over_sigma": _d("sqrt(mean_w(z^2)), z = e / sqrt(q_sigma^2 + sigma_floor^2)", "1"),
    "mae_over_sigma": _d("mean_w(|z|)", "1"),
    "bias_over_sigma": _d("mean_w(z)", "1"),
    "travel_span": _d("max(q_ref) - min(q_ref) of the TARGET", "{u}", weighting="none"),
    "net_displacement": _d("q_ref(T) - q_ref(t0)", "{u}", weighting="none"),
    "normalized_rmse_travel": _d("rmse / travel_span iff travel_span > travel_threshold", "1"),
    "normalized_mae_travel": _d("mean_abs_error / travel_span iff travel_span > travel_threshold", "1"),
    "delta_q_pred": _d("q_sim(T) - q_sim(t0)", "{u}", weighting="none"),
    "delta_q_ref": _d("q_ref(T) - q_ref(t0) (observed values, common gauge)", "{u}", weighting="none"),
    "displacement_error_signed": _d("delta_q_pred - delta_q_ref", "{u}", weighting="none"),
    "displacement_error_abs": _d("|displacement_error_signed|", "{u}", weighting="none"),
    "displacement_relative_error": _d("(delta_q_pred - delta_q_ref) / delta_q_ref iff |delta_q_ref| > "
                                      "travel_threshold; <0 undershoot, >0 overshoot", "1", weighting="none"),
    "endpoint_error_signed": _d("e(T) = q_sim(T) - q_ref(T)", "{u}", weighting="none"),
    "endpoint_error_abs": _d("|e(T)|", "{u}", weighting="none"),
    "initial_q_error_signed": _d("e(t0)", "{u}", weighting="none"),
    "qdot_sim_T": _d("qdot_sim at the last grid time", "{u}/s", weighting="none"),
    "abs_qdot_sim_T": _d("|qdot_sim(T)|", "{u}/s", weighting="none"),
    "settling_time_error_s": _d("t_settle_sim - t_stop_ref; t_settle_sim = first t_k after which |qdot_sim| <= "
                                "qdot_stop_threshold for all later frames; requires an established TERMINAL_REST",
                                "s", weighting="none"),
    "error_slope_per_s": _d("time-weighted LS slope of e vs (t - t0)", "{u}/s"),
    "error_slope_se": _d("DESCRIPTIVE cluster-robust (CR1) SE, clusters = the 4 physical-time quartiles, "
                         "sqrt(G/(G-1) sum_g (sum_{k in g} w_k (x_k - xbar) u_k)^2) / Sxx; not an IID-frame CI",
                         "{u}/s", _TIME),
    "drift_end_minus_start": _d("time mean of the piecewise-linear e over the final 10% of the duration minus the "
                                "same over the first 10% (time-based partitions)", "{u}"),
    "sample_deviance": _d("sum_k (e_k - mean(e))^2 (classical devianza over frames)", "{u}^2", _SAMP, "sample"),
    "sample_variance": _d("sample_deviance / (K - 1)", "{u}^2", _SAMP, "sample"),
    "sample_std": _d("sqrt(sample_variance)", "{u}", _SAMP, "sample"),
    "samplewt_bias": _d("mean_k e_k (legacy, frame-weighted)", "{u}", weighting="sample"),
    "samplewt_mae": _d("mean_k |e_k| (legacy, frame-weighted)", "{u}", weighting="sample"),
    "samplewt_mse": _d("mean_k e_k^2 (legacy, frame-weighted)", "{u}^2", weighting="sample"),
    "samplewt_rmse": _d("sqrt(samplewt_mse) (legacy, frame-weighted)", "{u}", weighting="sample"),
    "samplewt_variance_population": _d("sample_deviance / K (legacy)", "{u}^2", _SAMP, "sample"),
    "duration_s": _d("t_(K-1) - t_0 = W", "s", weighting="none"),
    "n_valid": _d("K frames scored", "count", weighting="none"),
    # ---- segments / quartiles (reporting bins only; full-grid Voronoi weights restricted to the bin)
    "segment.duration_s": _d("sum of full-grid w_k over the bin's frames (bins partition W exactly)", "s"),
    "terminal_plateau_bias": _d("TERMINAL_REST bias", "{u}"),
    "terminal_plateau_rmse": _d("TERMINAL_REST rmse", "{u}"),
    "terminal_plateau_std_error": _d("TERMINAL_REST std_error_time", "{u}", _TIME),
    "terminal_qdot_sim_rms": _d("sqrt(mean_w(qdot_sim^2)) over TERMINAL_REST", "{u}/s"),
    "terminal_qdot_sim_max_abs": _d("max |qdot_sim| over TERMINAL_REST", "{u}/s", weighting="none"),
    # ---- aggregates across runs (aggregate_scalars over the stated units of analysis)
    "agg.mean": _d("sum x / n", "metric unit", weighting="equal per unit of analysis"),
    "agg.deviance": _d("sum (x - mean)^2", "metric unit^2", "runs_or_units_of_analysis", "equal"),
    "agg.var_sample": _d("deviance / (n - 1), null (INSUFFICIENT_N) on n = 1", "metric unit^2",
                         "runs_or_units_of_analysis", "equal"),
    "agg.sd_sample": _d("sqrt(var_sample)", "metric unit", "runs_or_units_of_analysis", "equal"),
    "agg.var_population": _d("deviance / n", "metric unit^2", "runs_or_units_of_analysis", "equal"),
    "agg.p50|p90|p95": _d("QUANTILE_RULE with equal weights (nearest rank)", "metric unit", weighting="equal"),
    "matrix.row": _d("summary over successful held-out targets j != i of training demo i", "metric unit",
                     "different_target_runs", "equal"),
    "matrix.column": _d("summary over training demos i != j for target j", "metric unit",
                        "different_training_episodes", "equal"),
    "matrix.generalization_gap_rmse": _d("mean_j(rmse_ij, j != i) - rmse_ii", "{u}", weighting="equal"),
    "sweep.macro_object": _d("mean_O(mean over off-diagonal pairs within O)", "metric unit", "objects", "equal"),
    "sweep.macro_train": _d("mean_(O,i)(mean_j x_ij)", "metric unit", "training_demos", "equal"),
    "sweep.micro_pair": _d("mean over all valid off-diagonal ordered pairs", "metric unit", "pairs", "equal"),
    "sweep.micro_time_mae": _d("sum IAE_r / sum duration_r (DURATION-WEIGHTED, not a mean of run MAEs)", "{u}"),
    "sweep.micro_time_rmse": _d("sqrt(sum ISE_r / sum duration_r) (DURATION-WEIGHTED, not a mean of run RMSEs)",
                                "{u}"),
    "sweep.b_sweep": _d("sum_r W_r b_r / W_tot", "{u}", weighting="duration"),
    "sweep.D_sweep": _d("sum_r D_r + sum_r W_r (b_r - b_sweep)^2 (law of total weighted deviance)", "{u}^2*s",
                        "pooled_timelines", "duration"),
    "sweep.var_sweep": _d("D_sweep / W_tot (descriptive pooled residual variance, not independent experiments)",
                          "{u}^2", "pooled_timelines", "duration"),
    "sweep.rmse_sweep": _d("sqrt(b_sweep^2 + var_sweep) = sqrt(sum ISE_r / W_tot)", "{u}", weighting="duration"),
    "paired.delta": _d("metric_method - metric_baseline on the SAME ordered pair; negative = improvement",
                       "metric unit", "paired_differences", "equal"),
    "param.cv": _d("sd_sample / |mean|; null if n < 2 or |mean| <= near_zero", "1", "training_episodes", "equal"),
}


def q_units(joint_type):
    if joint_type not in UNITS:
        raise ValueError(f"joint_type must be 'prismatic' or 'revolute', got {joint_type!r}")
    return UNITS[joint_type]


def _unit(field, u):
    d = METRIC_DEFINITIONS.get(field)
    if field == "abs_signed_mean_error":
        return u
    return None if d is None else d["unit"].replace("{u}", u)


# ----------------------------------------------------------------------------------------------- primitives
def trapezoid_weights(t):
    """w_0 = (t_1 - t_0)/2, w_k = (t_(k+1) - t_(k-1))/2, w_(K-1) = (t_(K-1) - t_(K-2))/2; sum w = t_(K-1) - t_0.
    Requires K >= 3, finite strictly increasing t, all w > 0 (ValueError otherwise)."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 3:
        raise ValueError(f"time grid needs K >= 3 samples, got shape {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("time grid contains non-finite values")
    dt = np.diff(t)
    if not np.all(dt > 0):
        raise ValueError("time grid must be strictly increasing")
    w = np.empty_like(t)
    w[0], w[-1] = dt[0] / 2.0, dt[-1] / 2.0
    w[1:-1] = (t[2:] - t[:-2]) / 2.0
    if not np.all(w > 0):
        raise ValueError("non-positive trapezoid weight")
    return w


def weighted_quantile(x, w, p):
    """QUANTILE_RULE: smallest sorted x_(k) whose cumulative weight C_k >= p * sum(w) (stable sort, 1e-12 relative
    slack). Equal weights -> nearest rank (numpy 'inverted_cdf')."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    if x.size == 0 or x.shape != w.shape or not 0.0 <= p <= 1.0 or np.any(w < 0) or not w.sum() > 0:
        raise ValueError("weighted_quantile needs matching non-empty x/w, w >= 0, sum w > 0, p in [0, 1]")
    o = np.argsort(x, kind="stable")
    c = np.cumsum(w[o])
    k = int(np.searchsorted(c, p * c[-1] - 1e-12 * c[-1], side="left"))
    return float(x[o][min(k, x.size - 1)])


def _interval_mean(t, e, a, b):
    """Exact time mean over [a, b] of the piecewise-linear interpolant of e (what trapezoid quadrature integrates)."""
    pts = np.concatenate([[a], t[(t > a) & (t < b)], [b]])
    v = np.interp(pts, t, e)
    return float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(pts)) / (b - a))


def _check(name, lhs, rhs, ref, out):
    """|lhs - rhs| <= 1e-10 max(|lhs|, |rhs|) + 1e-14 ref (ref = data scale, the safe absolute floor near zero)."""
    diff = abs(lhs - rhs)
    out[name] = diff
    if not diff <= _REL_TOL * max(abs(lhs), abs(rhs)) + 1e-14 * ref:
        raise MetricIdentityError(f"{name}: {lhs!r} vs {rhs!r} (|diff| = {diff:.3e})")


def _bins(t, edges):
    """Frame -> bin index for consecutive half-open bins [edges_b, edges_(b+1)); the last bin is closed."""
    idx = np.searchsorted(edges, t, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def _wstats(t, e, w, mask, qdot=None):
    """Bin metrics with the FULL-grid Voronoi weights restricted to ``mask`` (bins partition W and ISE exactly)."""
    n = int(mask.sum())
    if n == 0:
        return {"status": "NO_SAMPLES", "n_samples": 0}
    ww, ee = w[mask], e[mask]
    W = float(ww.sum())
    b = float(ww @ ee / W)
    var = float(ww @ (ee - b) ** 2 / W)
    out = {"status": "OK", "n_samples": n, "duration_s": W, "bias": b, "mean_abs_error": float(ww @ np.abs(ee) / W),
           "rmse": math.sqrt(float(ww @ ee ** 2 / W)), "variance_error_time": var, "std_error_time": math.sqrt(var),
           "p95_abs_error": weighted_quantile(np.abs(ee), ww, 0.95), "max_abs_error": float(np.abs(ee).max())}
    if qdot is not None:
        qd = qdot[mask]
        out["qdot_sim_rms"] = math.sqrt(float(ww @ qd ** 2 / W))
        out["qdot_sim_max_abs"] = float(np.abs(qd).max())
    return out


def _vec(name, x, K):
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        x = np.full(K, float(x))
    if x.shape != (K,):
        raise ValueError(f"{name} shape {x.shape} != ({K},)")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    return x


# ----------------------------------------------------------------------------------------------- segments
SEGMENTS = ("INITIAL_REST", "MOVING_INTERVAL", "TERMINAL_REST")


def segment_labels(t_s, t_motion_start=None, t_stop=None, min_rest_samples=2):
    """Per-frame label (INITIAL_REST [t0, t_motion_start), MOVING_INTERVAL [t_motion_start, t_stop), TERMINAL_REST
    [t_stop, T_eval_end]) plus {segment: reason} for segments the given boundaries cannot support
    (SEGMENT_NOT_ESTABLISHED: boundary missing / outside the grid / too few frames). Unlabeled frames: 'UNLABELED'."""
    t = np.asarray(t_s, dtype=float)
    lab = np.full(t.size, "UNLABELED", dtype=object)
    why = {}
    t0, tend = t[0], t[-1]
    ms = t_motion_start if t_motion_start is not None and t0 < t_motion_start <= tend else None
    st = t_stop if t_stop is not None and t0 < t_stop <= tend else None
    if ms is not None and st is not None and not ms < st:
        return lab, {s: "SEGMENT_NOT_ESTABLISHED: t_motion_start >= t_stop" for s in SEGMENTS}
    masks = {"INITIAL_REST": (t < ms) if ms is not None else None,
             "MOVING_INTERVAL": ((t >= ms) & (t < st)) if ms is not None and st is not None else None,
             "TERMINAL_REST": (t >= st) if st is not None else None}
    need = {"INITIAL_REST": min_rest_samples, "MOVING_INTERVAL": 1, "TERMINAL_REST": min_rest_samples}
    for s in SEGMENTS:
        m = masks[s]
        if m is None:
            why[s] = "SEGMENT_NOT_ESTABLISHED: boundary not given or outside the evaluation grid"
        elif int(m.sum()) < need[s]:
            why[s] = f"SEGMENT_NOT_ESTABLISHED: {int(m.sum())} frame(s) < {need[s]} required"
        else:
            lab[m] = s
    return lab, why


def score_segments(t_s, q_ref, q_sim, t_motion_start=None, t_stop=None, qdot_sim=None, min_rest_samples=2):
    """Evaluation-only breakdown; NEVER changes the prediction, the grid or the full-run score.

    Segments: INITIAL_REST / MOVING_INTERVAL / TERMINAL_REST from the given (independently frozen) boundary times;
    TERMINAL_REST requires T_eval_end > t_stop with >= ``min_rest_samples`` frames, else SEGMENT_NOT_ESTABLISHED.
    Quartiles: 4 consecutive physical-time bins of [t0, T_eval_end] (edges t0 + W*{0,.25,.5,.75,1}).
    Each bin: n_samples, duration_s (= sum of full-grid w_k of its frames, so bins partition W), bias,
    mean_abs_error, rmse, variance_error_time, std_error_time, p95_abs_error, max_abs_error."""
    t = np.asarray(t_s, dtype=float)
    w = trapezoid_weights(t)
    K = t.size
    e = _vec("q_sim", q_sim, K) - _vec("q_ref", q_ref, K)
    qd = None if qdot_sim is None else _vec("qdot_sim", qdot_sim, K)
    lab, why = segment_labels(t, t_motion_start, t_stop, min_rest_samples)
    segs = {}
    for s in SEGMENTS:
        if s in why:
            segs[s] = {"status": "SEGMENT_NOT_ESTABLISHED", "reason": why[s]}
        else:
            m = lab == s
            segs[s] = dict(_wstats(t, e, w, m, qd if s == "TERMINAL_REST" else None),
                           t_start_s=float(t[m][0]), t_end_s=float(t[m][-1]))
    edges = t[0] + (t[-1] - t[0]) * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    qb = _bins(t, edges)
    quart = [dict(_wstats(t, e, w, qb == b), quartile=b + 1, t_start_s=float(edges[b]), t_end_s=float(edges[b + 1]))
             for b in range(4)]
    out = {"segments": segs, "quartiles": quart, "segment_label": lab, "boundaries":
           {"t_motion_start": t_motion_start, "t_stop": t_stop, "t_eval_end": float(t[-1])}, "null_reasons": {}}
    ter = segs["TERMINAL_REST"]
    for k, src in (("terminal_plateau_bias", "bias"), ("terminal_plateau_rmse", "rmse"),
                   ("terminal_plateau_std_error", "std_error_time"), ("terminal_qdot_sim_rms", "qdot_sim_rms"),
                   ("terminal_qdot_sim_max_abs", "qdot_sim_max_abs")):
        if ter.get("status") != "OK":
            out[k], out["null_reasons"][k] = None, "SEGMENT_NOT_ESTABLISHED"
        elif src not in ter:
            out[k], out["null_reasons"][k] = None, "NO_QDOT_SIM"
        else:
            out[k] = ter[src]
    return out


# ----------------------------------------------------------------------------------------------- one replay
def _failed(reason, joint_type):
    return {"status": "SCORING_FAILED", "reason": reason, "joint_type": joint_type, "q_units": UNITS[joint_type]}


def score_trajectory(t_s, q_ref, q_sim, q_sigma=None, sigma_floor=0.0, qdot_sim=None, joint_type=None,
                     travel_threshold=None, segments=None, qdot_stop_threshold=None):
    """Full-run metrics of ONE replay on the complete immutable grid (Section 8.2) + segment/quartile breakdown.

    ``segments``: optional {'t_motion_start': s | None, 't_stop': s | None} from the target's frozen plateau evidence
    (reporting only). ``travel_threshold`` [m|rad]: predeclared observability threshold for travel / displacement
    normalisations (None -> those normalisations are NOT_DEFINED_NO_TRAVEL_THRESHOLD). ``sigma_k = sqrt(q_sigma_k^2 +
    sigma_floor^2)``. Identities rmse^2 = bias^2 + variance_error_time, mse*duration = ise, mae*duration = iae,
    weighted_deviance/duration = variance_error_time are asserted (MetricIdentityError) with rel 1e-10 + 1e-14*scale.
    Bad data (shape, K < 3, non-increasing t, non-finite) -> {'status': 'SCORING_FAILED', 'reason': ...}."""
    u = q_units(joint_type)
    if travel_threshold is not None and not (math.isfinite(travel_threshold) and travel_threshold >= 0):
        raise ValueError("travel_threshold must be finite and >= 0 (predeclared)")
    if not (math.isfinite(sigma_floor) and sigma_floor >= 0):
        raise ValueError("sigma_floor must be finite and >= 0")
    try:
        t = np.asarray(t_s, dtype=float)
        w = trapezoid_weights(t)
        K = t.size
        qr, qs = _vec("q_ref", q_ref, K), _vec("q_sim", q_sim, K)
        qd = None if qdot_sim is None else _vec("qdot_sim", qdot_sim, K)
        qsig = None if q_sigma is None else _vec("q_sigma", q_sigma, K)
        if qsig is not None and np.any(qsig < 0):
            raise ValueError("q_sigma must be >= 0")
    except ValueError as exc:
        return _failed(str(exc), joint_type)
    nr = {}
    e = qs - qr
    a = np.abs(e)
    W = float(w.sum())
    duration = float(t[-1] - t[0])
    mw = lambda f: float(w @ f / W)                                     # noqa: E731
    bias, mae, mse = mw(e), mw(a), mw(e * e)
    r = e - bias
    var_t = mw(r * r)
    wdev, iae, ise = float(w @ (r * r)), float(w @ a), float(w @ (e * e))
    emean = float(e.mean())
    sdev = float(np.sum((e - emean) ** 2))
    peak = int(np.argmax(a))
    out = {"status": "OK", "metrics_version": METRICS_VERSION, "joint_type": joint_type, "q_units": u,
           "weighting": "trapezoid_time", "quantile_rule": QUANTILE_RULE, "n_valid": K, "duration_s": duration,
           "t_start_s": float(t[0]), "t_end_s": float(t[-1]),
           "signed_mean_error": bias, "bias": bias, "mean_abs_error": mae, "rmse": math.sqrt(mse), "mse": mse,
           "std_error_time": math.sqrt(var_t), "variance_error_time": var_t, "weighted_deviance": wdev,
           "sample_deviance": sdev, "sample_variance": sdev / (K - 1), "sample_std": math.sqrt(sdev / (K - 1)),
           "iae": iae, "ise": ise,
           "median_abs_error": weighted_quantile(a, w, 0.50), "p90_abs_error": weighted_quantile(a, w, 0.90),
           "p95_abs_error": weighted_quantile(a, w, 0.95), "p99_abs_error": weighted_quantile(a, w, 0.99),
           "max_abs_error": float(a[peak]), "peak_error_frame": peak, "peak_error_time_s": float(t[peak]),
           "samplewt_bias": emean, "samplewt_mae": float(a.mean()), "samplewt_mse": float(np.mean(e * e)),
           "samplewt_rmse": math.sqrt(float(np.mean(e * e))), "samplewt_variance_population": sdev / K}
    # ---- uncertainty-normalised (actual q_sigma vs imposed floor recorded separately)
    if qsig is None and sigma_floor == 0.0:
        sig, why = None, "NO_SIGMA"
    else:
        sig = np.sqrt((qsig if qsig is not None else 0.0) ** 2 + sigma_floor ** 2) * np.ones(K)
        why = None if np.all(sig > 0) else "NONPOSITIVE_SIGMA"
    out["sigma_source"] = ("none" if qsig is None and sigma_floor == 0 else "floor_only" if qsig is None
                           else "q_sigma" if sigma_floor == 0 else "q_sigma+floor")
    out["sigma_floor"] = float(sigma_floor)
    out["frac_frames_floor_dominated"] = (None if qsig is None else float(np.mean(sigma_floor >= qsig)))
    for k in ("rmse_over_sigma", "mae_over_sigma", "bias_over_sigma"):
        out[k] = None
    if why is None:
        z = e / sig
        out.update(rmse_over_sigma=math.sqrt(mw(z * z)), mae_over_sigma=mw(np.abs(z)), bias_over_sigma=mw(z))
    else:
        nr.update(rmse_over_sigma=why, mae_over_sigma=why, bias_over_sigma=why)
    # ---- travel / displacement / endpoints (target-only definitions)
    span, dref, dpred = float(np.ptp(qr)), float(qr[-1] - qr[0]), float(qs[-1] - qs[0])
    out.update(travel_span=span, net_displacement=dref, delta_q_pred=dpred, delta_q_ref=dref,
               displacement_error_signed=dpred - dref, displacement_error_abs=abs(dpred - dref),
               endpoint_error_signed=float(e[-1]), endpoint_error_abs=float(a[-1]),
               initial_q_error_signed=float(e[0]), travel_threshold=travel_threshold)
    if travel_threshold is None:
        nr.update(normalized_rmse_travel="NOT_DEFINED_NO_TRAVEL_THRESHOLD",
                  normalized_mae_travel="NOT_DEFINED_NO_TRAVEL_THRESHOLD",
                  displacement_relative_error="NOT_DEFINED_NO_TRAVEL_THRESHOLD")
        out.update(normalized_rmse_travel=None, normalized_mae_travel=None, displacement_relative_error=None)
    else:
        if span > travel_threshold:
            out.update(normalized_rmse_travel=out["rmse"] / span, normalized_mae_travel=mae / span)
        else:
            out.update(normalized_rmse_travel=None, normalized_mae_travel=None)
            nr.update(normalized_rmse_travel="NOT_DEFINED_LOW_TRAVEL", normalized_mae_travel="NOT_DEFINED_LOW_TRAVEL")
        if abs(dref) > travel_threshold:
            out["displacement_relative_error"] = (dpred - dref) / dref
        else:
            out["displacement_relative_error"], nr["displacement_relative_error"] = None, "NOT_DEFINED_LOW_DISPLACEMENT"
    if qd is None:
        out.update(qdot_sim_T=None, abs_qdot_sim_T=None)
        nr.update(qdot_sim_T="NO_QDOT_SIM", abs_qdot_sim_T="NO_QDOT_SIM")
    else:
        out.update(qdot_sim_T=float(qd[-1]), abs_qdot_sim_T=abs(float(qd[-1])))
    # ---- trend: weighted LS slope of the signed residual, CR1 SE over 4 time-quartile clusters; 10 % drift
    x = t - t[0]
    xc = x - mw(x)
    sxx = float(w @ xc ** 2)
    slope = float(w @ (xc * r)) / sxx
    res = r - slope * xc
    qb = _bins(t, t[0] + duration * np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    scores = np.array([float(np.sum((w * xc * res)[qb == g])) for g in range(4) if np.any(qb == g)])
    G = scores.size
    out["error_slope_per_s"] = slope
    out["error_slope_se_assumption"] = "descriptive CR1 cluster-robust SE, G = nonempty physical-time quartiles"
    if G >= 2:
        out["error_slope_se"] = math.sqrt(G / (G - 1) * float(scores @ scores)) / sxx
    else:
        out["error_slope_se"], nr["error_slope_se"] = None, "INSUFFICIENT_CLUSTERS"
    out["drift_end_minus_start"] = (_interval_mean(t, e, t[-1] - 0.1 * duration, t[-1])
                                    - _interval_mean(t, e, t[0], t[0] + 0.1 * duration))
    # ---- breakdown (reporting only) and settling diagnostic
    seg = segments or {}
    br = score_segments(t, qr, qs, seg.get("t_motion_start"), seg.get("t_stop"), qd)
    out["segments"], out["quartiles"] = br["segments"], br["quartiles"]
    for k in ("terminal_plateau_bias", "terminal_plateau_rmse", "terminal_plateau_std_error",
              "terminal_qdot_sim_rms", "terminal_qdot_sim_max_abs"):
        out[k] = br[k]
        if k in br["null_reasons"]:
            nr[k] = br["null_reasons"][k]
    out["settling_time_error_s"] = None
    if br["segments"]["TERMINAL_REST"].get("status") != "OK":
        nr["settling_time_error_s"] = "NOT_EVALUATED: reference stop not established by a terminal rest"
    elif qd is None or qdot_stop_threshold is None:
        nr["settling_time_error_s"] = "NOT_EVALUATED: needs qdot_sim and a predeclared qdot_stop_threshold"
    else:
        moving = np.abs(qd) > qdot_stop_threshold
        if moving[-1]:
            nr["settling_time_error_s"] = "NOT_EVALUATED: SIM_NOT_SETTLED"
        else:
            k_settle = int(np.flatnonzero(moving)[-1] + 1) if moving.any() else 0
            out["settling_time_error_s"] = float(t[k_settle] - seg["t_stop"])
    # ---- exact identities (Section 8.2)
    s2 = float(a.max()) ** 2
    chk = {}
    _check("rmse2_eq_bias2_plus_variance", out["rmse"] ** 2, bias ** 2 + var_t, s2, chk)
    _check("mse_times_duration_eq_ise", mse * duration, ise, s2 * duration, chk)
    _check("mae_times_duration_eq_iae", mae * duration, iae, float(a.max()) * duration, chk)
    _check("weighted_deviance_over_duration_eq_variance", wdev / duration, var_t, s2, chk)
    out["identity_checks"] = chk
    out["null_reasons"] = nr
    out["units"] = {k: _unit(k, u) for k in out if _unit(k, u) is not None}
    return out


def residual_arrays(t_s, q_ref, q_sim, q_sigma=None, sigma_floor=0.0, qdot_sim=None, segments=None,
                    source_frame=None):
    """Arrays for residuals.npz (t_s, source_frame, q_reference, q_sim, qdot_sim, e_signed, e_abs, z_normalized,
    w_time, segment_label, valid): every reported metric can be recomputed from them alone."""
    t = np.asarray(t_s, dtype=float)
    K = t.size
    w = trapezoid_weights(t)
    qr, qs = _vec("q_ref", q_ref, K), _vec("q_sim", q_sim, K)
    e = qs - qr
    if q_sigma is None and sigma_floor == 0:
        z = np.full(K, np.nan)
    else:
        z = e / np.sqrt((0.0 if q_sigma is None else _vec("q_sigma", q_sigma, K)) ** 2 + sigma_floor ** 2)
    seg = segments or {}
    lab, _ = segment_labels(t, seg.get("t_motion_start"), seg.get("t_stop"))
    return {"t_s": t, "source_frame": np.arange(K) if source_frame is None else np.asarray(source_frame),
            "q_reference": qr, "q_sim": qs,
            "qdot_sim": np.full(K, np.nan) if qdot_sim is None else _vec("qdot_sim", qdot_sim, K),
            "e_signed": e, "e_abs": np.abs(e), "z_normalized": z, "w_time": w,
            "segment_label": lab.astype(str), "valid": np.ones(K, dtype=bool)}


def score_scopes(t_s, q_ref, q_sim, *, t_stop, tail_input_complete, q_sigma=None, qdot_sim=None, **kw):
    """Two DISTINCT scopes (Section 8.1): T_STOP = [t0, T_stop] and T_EVAL_END = [t0, T_eval_end] (whole grid,
    stationary tail included). Primary = T_EVAL_END when the force input covers the tail; otherwise the tail scope
    is TAIL_INPUT_INCOMPLETE (never extrapolated) and the primary falls back to T_STOP with reduced coverage."""
    t = np.asarray(t_s, dtype=float)
    m = t <= t_stop
    sl = lambda x: x if x is None or np.ndim(x) == 0 else np.asarray(x)[m]    # noqa: E731
    seg = dict(kw.pop("segments", None) or {}, t_stop=t_stop)
    stop = score_trajectory(t[m], sl(q_ref), sl(q_sim), q_sigma=sl(q_sigma), qdot_sim=sl(qdot_sim), segments=seg,
                            **kw)
    has_tail = bool(t[-1] > t_stop)
    if not has_tail:
        full = {"status": "NO_TAIL_AFTER_T_STOP"}
    elif not tail_input_complete:
        full = {"status": "TAIL_INPUT_INCOMPLETE"}
    else:
        full = score_trajectory(t, q_ref, q_sim, q_sigma=q_sigma, qdot_sim=qdot_sim, segments=seg, **kw)
    primary = "T_EVAL_END" if full.get("status") == "OK" else "T_STOP"
    return {"T_STOP": stop, "T_EVAL_END": full, "primary": primary,
            "coverage": "FULL" if primary == "T_EVAL_END" or not has_tail else "REDUCED_TAIL_INPUT_INCOMPLETE",
            "t_stop": float(t_stop), "t_eval_end": float(t[-1])}


# ----------------------------------------------------------------------------------------------- aggregation
def aggregate_scalars(x):
    """n, mean, deviance (sum (x - mean)^2), var_sample (D/(n-1); null INSUFFICIENT_N on n = 1), sd_sample,
    var_population (D/n), min, p50, p90, p95 (QUANTILE_RULE, equal weights = nearest rank), max.
    None / non-finite inputs are excluded and counted in n_invalid. n = 0 -> status NO_VALID_RUNS (all values None)."""
    vals = [] if x is None else list(x)
    good = [float(v) for v in vals if v is not None and isinstance(v, (int, float, np.number)) and math.isfinite(v)]
    out = {"n": len(good), "n_invalid": len(vals) - len(good), "null_reasons": {}}
    keys = ("mean", "deviance", "var_sample", "sd_sample", "var_population", "min", "p50", "p90", "p95", "max")
    if not good:
        out.update({k: None for k in keys}, status="NO_VALID_RUNS")
        out["null_reasons"] = {k: "NO_VALID_RUNS" for k in keys}
        return out
    v = np.array(good)
    mu = float(v.mean())
    D = float(np.sum((v - mu) ** 2))
    ones = np.ones_like(v)
    out.update(status="OK", mean=mu, deviance=D, var_population=D / v.size, min=float(v.min()), max=float(v.max()),
               p50=weighted_quantile(v, ones, 0.5), p90=weighted_quantile(v, ones, 0.9),
               p95=weighted_quantile(v, ones, 0.95))
    if v.size >= 2:
        out.update(var_sample=D / (v.size - 1), sd_sample=math.sqrt(D / (v.size - 1)))
    else:
        out.update(var_sample=None, sd_sample=None)
        out["null_reasons"] = {"var_sample": "INSUFFICIENT_N", "sd_sample": "INSUFFICIENT_N"}
    return out


def _value(cell, name):
    src = cell["metrics"] if isinstance(cell.get("metrics"), dict) else cell
    if name == "abs_signed_mean_error":
        v = _value(cell, "signed_mean_error")
        return None if v is None else abs(v)
    v = src.get(name)
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float, np.number)) or not math.isfinite(v):
        return None
    return float(v)


def _classify(cell):
    """'success' | 'ineligible' | failure category (status string, or CORRUPT_METRICS for a success status whose
    rmse is missing/non-finite)."""
    st = str(cell.get("status"))
    if st in SUCCESS_STATUSES:
        return "success" if _value(cell, "rmse") is not None else "CORRUPT_METRICS"
    if st in INELIGIBLE_STATUSES or st.startswith("INELIGIBLE") or cell.get("eligible") is False:
        return "ineligible"
    return st


def _counts(by_key, expected):
    """Failure-inclusive accounting over the EXPECTED ordered cells (missing = expected but absent)."""
    cat = Counter()
    n_succ = n_inel = 0
    for k in expected:
        c = by_key.get(k)
        cls = "MISSING" if c is None else _classify(c)
        if cls == "success":
            n_succ += 1
        elif cls == "ineligible":
            n_inel += 1
            cat["INELIGIBLE:" + str(c.get("status"))] += 1
        else:
            cat[cls] += 1
    n_exp, n_miss = len(expected), cat.get("MISSING", 0)
    n_att = n_exp - n_miss - n_inel
    return {"n_expected": n_exp, "n_listed": n_exp - n_miss, "n_missing": n_miss, "n_ineligible": n_inel,
            "n_eligible": n_exp - n_inel, "n_attempted": n_att, "n_success": n_succ, "n_failed": n_att - n_succ,
            "coverage_fraction": n_succ / n_exp if n_exp else None,
            "coverage_of_eligible": n_succ / (n_exp - n_inel) if n_exp - n_inel else None,
            "failure_categories": dict(sorted(cat.items()))}


def _sum_counts(cs):
    keys = ("n_expected", "n_listed", "n_missing", "n_ineligible", "n_eligible", "n_attempted", "n_success",
            "n_failed")
    out = {k: sum(c[k] for c in cs) for k in keys}
    out["coverage_fraction"] = out["n_success"] / out["n_expected"] if out["n_expected"] else None
    out["coverage_of_eligible"] = out["n_success"] / out["n_eligible"] if out["n_eligible"] else None
    cat = Counter()
    for c in cs:
        cat.update(c["failure_categories"])
    out["failure_categories"] = dict(sorted(cat.items()))
    return out


def _single(cells, field, what):
    vals = {c.get(field) for c in cells}
    if len(vals) > 1:
        raise ValueError(f"{what}: cells mix {field} values {sorted(map(str, vals))}; aggregate them separately")
    return next(iter(vals)) if vals else None


def _joint_type(cells):
    """Unique joint type of the successful cells; refuses to pool m and rad."""
    jts = {c.get("joint_type") for c in cells}
    if None in jts:
        raise ValueError("a successful cell lacks joint_type; units unknown")
    if len(jts) > 1:
        raise ValueError(f"refusing to pool raw errors across joint types {sorted(jts)} (m vs rad)")
    jt = next(iter(jts)) if jts else None
    if jt is not None:
        q_units(jt)
    return jt


def _diag(c):
    d = c["train_id"] == c["test_id"]
    if "is_train_diagonal" in c and bool(c["is_train_diagonal"]) != d:
        raise ValueError(f"is_train_diagonal inconsistent with train/test ids in {c}")
    return d


def aggregate_matrix(cells, episodes=None, metrics=DEFAULT_AGG_METRICS):
    """One (object, method, seed, protocol) ordered N x N block. Cells: {object, train_id, test_id, method, seed,
    status, joint_type, metrics: {...} (or flat)}; success = status in SUCCESS_STATUSES with finite rmse.

    Diagonal (i = j, TRAIN) and off-diagonal (i != j, HELD-OUT) are summarised SEPARATELY; rows = per training demo
    over targets j != i, columns = per target over trainers i != j; generalization_gap_rmse/_mae per row =
    mean_j x_ij - x_ii and object-level = offdiag mean - diag mean. Counts are failure-inclusive over all N^2
    expected cells (``episodes`` defaults to every id seen). N < 2 -> INSUFFICIENT_EPISODES, no transfer metric."""
    cells = list(cells)
    obj = _single(cells, "object", "aggregate_matrix")
    method, seed = _single(cells, "method", "aggregate_matrix"), _single(cells, "seed", "aggregate_matrix")
    protocol = _single(cells, "protocol", "aggregate_matrix")
    by = {}
    for c in cells:
        k = (c["train_id"], c["test_id"])
        if k in by:
            raise ValueError(f"duplicate cell {k} for object {obj}")
        _diag(c)
        by[k] = c
    eps = sorted({k[0] for k in by} | {k[1] for k in by}) if episodes is None else list(episodes)
    extra = set(by) - {(i, j) for i in eps for j in eps}
    if extra:
        raise ValueError(f"cells outside the declared episodes: {sorted(extra)}")
    head = {"object": obj, "method": method, "seed": seed, "protocol": protocol, "n_episodes": len(eps),
            "episodes": eps}
    diag_keys = [(e, e) for e in eps]
    off_keys = [(i, j) for i in eps for j in eps if i != j]
    if len(eps) < 2:
        return dict(head, status="INSUFFICIENT_EPISODES", reason=f"N = {len(eps)} < 2: no cross-demo transfer",
                    diagonal={"counts": _counts(by, diag_keys)})
    ok = {k: c for k, c in by.items() if _classify(c) == "success"}
    jt = _joint_type(ok.values())
    u = UNITS.get(jt)

    def block(keys):
        return {"counts": _counts(by, keys),
                "metrics": {m: aggregate_scalars([_value(ok[k], m) for k in keys if k in ok]) for m in metrics}}

    def gap(m, keys, dk):
        offm = aggregate_scalars([_value(ok[k], m) for k in keys if k in ok])["mean"]
        d = _value(ok[dk], m) if dk in ok else None
        if offm is None:
            return None, "NO_SUCCESSFUL_HELDOUT"
        if d is None:
            return None, "TRAIN_DIAGONAL_UNAVAILABLE"
        return offm - d, None

    rows, cols = {}, {}
    for i in eps:
        keys = [(i, j) for j in eps if j != i]
        r = block(keys)
        r["train_diagonal"] = {m: (_value(ok[(i, i)], m) if (i, i) in ok else None) for m in metrics}
        r["train_diagonal_status"] = by[(i, i)].get("status") if (i, i) in by else "MISSING"
        r["null_reasons"] = {}
        for m, name in (("rmse", "generalization_gap_rmse"), ("mean_abs_error", "generalization_gap_mae")):
            r[name], why = gap(m, keys, (i, i))
            if why:
                r["null_reasons"][name] = why
        rows[i] = r
    for j in eps:
        keys = [(i, j) for i in eps if i != j]
        cl = block(keys)
        okc = [ok[k] for k in keys if k in ok]
        cl["n_distinct_train"] = len({c["train_id"] for c in okc})
        for name, src in (("target_duration_s", "duration_s"), ("target_travel_span", "travel_span"),
                          ("target_impulse", "target_impulse")):
            vals = [_value(c, src) for c in okc if _value(c, src) is not None]
            cl[name] = vals[0] if vals else None
        cols[j] = cl
    diag, off = block(diag_keys), block(off_keys)
    okoff = [k for k in off_keys if k in ok]
    om, dm = off["metrics"].get("rmse", {}).get("mean"), diag["metrics"].get("rmse", {}).get("mean")
    return dict(head, status="OK", joint_type=jt, q_units=u, diagonal=diag, offdiag=off, rows=rows, columns=cols,
                n_valid_train=len({k[0] for k in okoff}), n_valid_test=len({k[1] for k in okoff}),
                generalization_gap_rmse=None if om is None or dm is None else om - dm,
                units={m: _unit(m, u) if u else None for m in metrics})


def pooled_residual_deviance(runs):
    """Law of total weighted deviance over replay timelines of ONE unit (Section 8.4 item 4). Per run:
    W_r = duration_s, b_r = signed_mean_error, D_r = weighted_deviance. b_sweep = sum W_r b_r / W_tot,
    D_sweep = sum D_r + sum W_r (b_r - b_sweep)^2, var_sweep = D_sweep / W_tot, rmse_sweep = sqrt(b_sweep^2 +
    var_sweep) (= sqrt(sum ISE_r / W_tot), asserted when ise is present). Descriptive, duration-weighted; runs share
    training/target demos and are NOT independent experiments. Mixed m / rad -> ValueError."""
    runs = list(runs)
    units = {r.get("q_units") or UNITS.get(r.get("joint_type")) for r in runs}
    if None in units or len(units) > 1:
        raise ValueError(f"refusing pooled residual deviance across units {sorted(map(str, units))}")
    if not runs:
        return {"status": "NO_VALID_RUNS", "n_runs": 0}
    vals = [(_value(r, "duration_s"), _value(r, "signed_mean_error"), _value(r, "weighted_deviance")) for r in runs]
    if any(v is None for t in vals for v in t):
        raise ValueError("every run needs finite duration_s, signed_mean_error and weighted_deviance")
    W, b, D = (np.array(x) for x in zip(*vals))
    Wt = float(W.sum())
    bs = float(W @ b / Wt)
    within, between = float(D.sum()), float(W @ (b - bs) ** 2)
    Ds = within + between
    var = Ds / Wt
    u = next(iter(units))
    out = {"status": "OK", "n_runs": len(runs), "q_units": u, "W_tot_s": Wt, "b_sweep": bs,
           "deviance_within_runs": within, "deviance_between_run_biases": between, "D_sweep": Ds,
           "fraction_within": within / Ds if Ds > 0 else None, "fraction_between": between / Ds if Ds > 0 else None,
           "var_sweep": var, "std_sweep": math.sqrt(var), "rmse_sweep": math.sqrt(bs ** 2 + var),
           "note": "descriptive duration-weighted pooled residual statistics; not independent experiments",
           "units": {"b_sweep": u, "D_sweep": f"{u}^2*s", "var_sweep": f"{u}^2", "std_sweep": u, "rmse_sweep": u}}
    if Ds == 0:
        out["null_reasons"] = {"fraction_within": "ZERO_TOTAL_DEVIANCE", "fraction_between": "ZERO_TOTAL_DEVIANCE"}
    ise = [_value(r, "ise") for r in runs]
    if all(v is not None for v in ise):
        chk = {}
        _check("rmse_sweep2_eq_sum_ise_over_W", out["rmse_sweep"] ** 2 * Wt, float(sum(ise)),
               float(sum(ise)) + Wt * bs ** 2, chk)
        out["identity_checks"] = chk
    return out


def _macro_micro(ok_cells, metric):
    """micro_pair over all pairs; macro_train over (object, train) row means; macro_object over object means."""
    vals = [(c["object"], c["train_id"], _value(c, metric)) for c in ok_cells]
    vals = [v for v in vals if v[2] is not None]
    rows, objs = defaultdict(list), defaultdict(list)
    for o, i, x in vals:
        rows[(o, i)].append(x)
        objs[o].append(x)
    return {"micro_pair": aggregate_scalars([v[2] for v in vals]),
            "macro_train": dict(aggregate_scalars([float(np.mean(v)) for v in rows.values()]),
                                n_train_demos=len(rows)),
            "macro_object": dict(aggregate_scalars([float(np.mean(v)) for v in objs.values()]), n_objects=len(objs)),
            "n_eligible_values": len(vals), "n_success_pairs": len(ok_cells)}


def _pair_weights(ok_cells):
    """Weight of every (object, train, test) pair in each scheme's final mean (each sums to 1)."""
    n = len(ok_cells)
    rows = Counter((c["object"], c["train_id"]) for c in ok_cells)
    objs = Counter(c["object"] for c in ok_cells)
    out = {"micro_pair": {}, "macro_train": {}, "macro_object": {}}
    for c in ok_cells:
        k = (c["object"], c["train_id"], c["test_id"])
        out["micro_pair"][k] = 1.0 / n
        out["macro_train"][k] = 1.0 / (len(rows) * rows[k[:2]])
        out["macro_object"][k] = 1.0 / (len(objs) * objs[k[0]])
    return out


def _pooled_or_reason(ok):
    if not ok:
        return {"status": "NO_VALID_RUNS", "n_runs": 0}
    if any(_value(c, k) is None for c in ok for k in ("duration_s", "signed_mean_error", "weighted_deviance")):
        return {"status": "NOT_DEFINED", "reason": "duration_s/signed_mean_error/weighted_deviance missing on some "
                                                   "successful pair"}
    return pooled_residual_deviance(ok)


def aggregate_sweep(cells, episodes_by_object=None, metrics=DEFAULT_AGG_METRICS):
    """Whole sweep for ONE (method, seed, protocol): per-object ``aggregate_matrix`` blocks, then off-diagonal
    successful pairs SEPARATELY PER JOINT TYPE (raw SI units never pooled): macro_object, macro_train, micro_pair
    (each an ``aggregate_scalars`` over its units of analysis, with exact ``pair_weights``), micro_time
    (mae = sum IAE / sum duration, rmse = sqrt(sum ISE / sum duration); DURATION-WEIGHTED, not a mean of run
    errors) and ``pooled_residual_deviance``. The cross-type table holds DIMENSIONLESS metrics only, with their
    normalisation and eligibility. Objects with N < 2 -> INSUFFICIENT_EPISODES, excluded from transfer averages."""
    cells = list(cells)
    method, seed = _single(cells, "method", "aggregate_sweep"), _single(cells, "seed", "aggregate_sweep")
    protocol = _single(cells, "protocol", "aggregate_sweep")
    byobj = defaultdict(list)
    for c in cells:
        byobj[c["object"]].append(c)
    objects, insufficient, per_jt_cells, per_jt_counts = {}, [], defaultdict(list), defaultdict(list)
    no_success = []
    for o in sorted(byobj, key=str):
        blk = aggregate_matrix(byobj[o], (episodes_by_object or {}).get(o), metrics)
        objects[o] = blk
        if blk["status"] == "INSUFFICIENT_EPISODES":
            insufficient.append(o)
            continue
        if blk["joint_type"] is None:
            no_success.append(o)
            continue
        per_jt_counts[blk["joint_type"]].append(blk["offdiag"]["counts"])
        per_jt_cells[blk["joint_type"]].extend(c for c in byobj[o] if not _diag(c) and _classify(c) == "success")
    by_jt = {}
    for jt, ok in per_jt_cells.items():
        u = UNITS[jt]
        raw = [m for m in metrics if m not in DIMENSIONLESS]
        mt = {"status": "OK"}
        tri = [(_value(c, "iae"), _value(c, "ise"), _value(c, "duration_s")) for c in ok]
        if ok and all(v is not None for t in tri for v in t):
            I, S, T = (sum(x) for x in zip(*tri))
            mt.update(mae_time_weighted=I / T, rmse_time_weighted=math.sqrt(S / T), total_duration_s=T, n_runs=len(ok),
                      label="DURATION-WEIGHTED (sum IAE / sum duration, sqrt(sum ISE / sum duration))")
        else:
            mt = {"status": "NOT_DEFINED", "reason": "iae/ise/duration_s missing on some successful pair"}
        demos = {(c["object"], c["train_id"]) for c in ok} | {(c["object"], c["test_id"]) for c in ok}
        n_obj = len({c["object"] for c in ok})
        by_jt[jt] = {"q_units": u, "n_objects": n_obj, "n_unique_train_demos": len({(c["object"], c["train_id"])
                                                                                     for c in ok}),
                     "n_unique_demos": len(demos), "counts": _sum_counts(per_jt_counts[jt]),
                     "metrics": {m: dict(_macro_micro(ok, m), unit=_unit(m, u)) for m in metrics},
                     "raw_metrics": raw, "micro_time": mt,
                     "pooled_residual_deviance": _pooled_or_reason(ok),
                     "pair_weights": _pair_weights(ok) if ok else {},
                     "ci_status": "CI_NOT_RELIABLE_SMALL_N" if n_obj < MIN_CI_OBJECTS else "CI_NOT_COMPUTED"}
    all_ok = [c for v in per_jt_cells.values() for c in v]
    norm = {m: dict(_macro_micro(all_ok, m), normalization=NORMALIZATION[m], joint_types_included=sorted(per_jt_cells))
            for m in DIMENSIONLESS if m in metrics or m == "normalized_rmse_travel"}
    allc = [b["offdiag"]["counts"] for b in objects.values() if b["status"] == "OK"]
    return {"method": method, "seed": seed, "protocol": protocol, "metrics_version": METRICS_VERSION,
            "quantile_rule": QUANTILE_RULE, "objects": objects, "insufficient_episodes": insufficient,
            "objects_without_success": no_success, "by_joint_type": by_jt,
            "raw_cross_type": "REFUSED_MIXED_UNITS" if len(by_jt) > 1 else "SINGLE_JOINT_TYPE",
            "normalized_cross_type": norm,
            "failure_inventory": {"offdiag": _sum_counts(allc) if allc else None,
                                  "by_object": {o: b["offdiag"]["counts"] for o, b in objects.items()
                                                if b["status"] == "OK"},
                                  "insufficient_episodes": insufficient}}


def paired_method_comparison(cells, method, baseline, metrics=("rmse", "mean_abs_error"), on_mismatch="intersect",
                             include_diagonal=False, zero_floor=0.0):
    """Paired differences delta_x = x_method - x_baseline on the INTERSECTION of valid ordered (object, train, test)
    pairs (negative = improvement); per joint type (never pooled m/rad). Reports coverage (n_method_valid,
    n_baseline_valid, n_union, n_paired, unmatched pairs). Mismatched sets: on_mismatch='raise' -> ValueError,
    'intersect' -> status INTERSECTED_MISMATCHED_SETS. Percent improvement 100 (base - method) / base only where
    base > zero_floor."""
    if on_mismatch not in ("intersect", "raise"):
        raise ValueError("on_mismatch must be 'intersect' or 'raise'")

    def valid(m):
        out = {}
        for c in cells:
            if c.get("method") != m or (not include_diagonal and _diag(c)):
                continue
            k = (c["object"], c["train_id"], c["test_id"])
            if k in out:
                raise ValueError(f"method {m}: several cells for pair {k} (seed ambiguity); filter seeds first")
            out[k] = c
        return {k: c for k, c in out.items() if _classify(c) == "success"}

    A, B = valid(method), valid(baseline)
    inter = sorted(set(A) & set(B), key=str)
    union = set(A) | set(B)
    only_a, only_b = sorted(set(A) - set(B), key=str), sorted(set(B) - set(A), key=str)
    if (only_a or only_b) and on_mismatch == "raise":
        raise ValueError(f"mismatched pair sets: {len(only_a)} only in {method}, {len(only_b)} only in {baseline}")
    cov = {"n_method_valid": len(A), "n_baseline_valid": len(B), "n_union": len(union), "n_paired": len(inter),
           "coverage_fraction": len(inter) / len(union) if union else None,
           "pairs_only_method": only_a, "pairs_only_baseline": only_b}
    status = "NO_VALID_RUNS" if not inter else "MATCHED" if not (only_a or only_b) else "INTERSECTED_MISMATCHED_SETS"
    by_jt = defaultdict(list)
    for k in inter:
        if A[k].get("joint_type") != B[k].get("joint_type"):
            raise ValueError(f"pair {k}: joint types differ between methods")
        by_jt[A[k]["joint_type"]].append(k)
    res = {}
    for jt, keys in by_jt.items():
        u = q_units(jt)
        r = {"q_units": u, "n_paired": len(keys)}
        for m in metrics:
            d, pct, n_pct_excl = [], [], 0
            for k in keys:
                a, b = _value(A[k], m), _value(B[k], m)
                if a is None or b is None:
                    continue
                d.append(a - b)
                if b > zero_floor:
                    pct.append(100.0 * (b - a) / b)
                else:
                    n_pct_excl += 1
            agg = aggregate_scalars(d)
            r["delta_" + {"mean_abs_error": "mae"}.get(m, m)] = dict(
                agg, median=agg["p50"], n_improved=sum(x < 0 for x in d), n_worse=sum(x > 0 for x in d),
                unit=_unit(m, u), percent_improvement=aggregate_scalars(pct), n_percent_excluded_zero_floor=n_pct_excl)
        res[jt] = r
    return {"method": method, "baseline": baseline, "status": status, "coverage": cov, "by_joint_type": res,
            "include_diagonal": include_diagonal, "sign_convention": "delta = method - baseline; negative = better"}


def parameter_variation(models, near_zero=1e-12):
    """Between-training-episode variation of M and c for ONE object/method: n, mean, sd_sample, var_sample,
    deviance, cv = sd_sample / |mean| (null CV_UNDEFINED_NEAR_ZERO_MEAN when |mean| <= near_zero, INSUFFICIENT_N on
    n < 2), min, max, range; Pearson corr(M, c) when n >= 3. Rejected fits (non-OK status, non-finite, M <= 0,
    c < 0) are excluded and counted. Units: prismatic M [kg], c [N*s/m]; revolute M [kg*m^2], c [N*m*s/rad]."""
    models = list(models)
    _single(models, "object", "parameter_variation")
    good, rejected = [], Counter()
    for m in models:
        st = m.get("status", "OK")
        M, c = _value(m, "M"), _value(m, "c")
        if st not in SUCCESS_STATUSES:
            rejected[str(st)] += 1
        elif M is None or c is None or M <= 0 or c < 0:
            rejected["NONFINITE_OR_OUT_OF_BOUNDS"] += 1
        else:
            good.append(m)
    jt = _joint_type(good) if good else None
    out = {"joint_type": jt, "n_models": len(models), "n_valid": len(good), "rejected": dict(rejected),
           "units": PARAM_UNITS.get(jt), "null_reasons": {}}
    for p in ("M", "c"):
        agg = aggregate_scalars([_value(m, p) for m in good])
        s = {k: agg[k] for k in ("n", "mean", "deviance", "var_sample", "sd_sample", "min", "max")}
        s["range"] = None if agg["n"] == 0 else agg["max"] - agg["min"]
        if agg["n"] < 2:
            s["cv"], out["null_reasons"][p + ".cv"] = None, "INSUFFICIENT_N"
        elif abs(agg["mean"]) <= near_zero:
            s["cv"], out["null_reasons"][p + ".cv"] = None, "CV_UNDEFINED_NEAR_ZERO_MEAN"
        else:
            s["cv"] = agg["sd_sample"] / abs(agg["mean"])
        out[p] = s
    out["corr_M_c"] = None
    if len(good) >= 3:
        M = np.array([_value(m, "M") for m in good])
        c = np.array([_value(m, "c") for m in good])
        if M.std() > 0 and c.std() > 0:
            out["corr_M_c"] = float(np.corrcoef(M, c)[0, 1])
        else:
            out["null_reasons"]["corr_M_c"] = "ZERO_VARIANCE"
    else:
        out["null_reasons"]["corr_M_c"] = "INSUFFICIENT_N"
    return out
