"""Fair, held-out model selection between static / prismatic / revolute (spec 9.2, 9.3).

* The time split is made *before* initialization; every training step (noise calibration,
  proposals, seeds, EM) sees only training timestamps (no leakage, test T24).
* Both hypotheses are fit on exactly the same observations with the same budget and noise model.
* Selection uses the held-out per-track predictive likelihood ``sum_k p(k | train) p(y_ho | k)``
  with a paired standard error over tracks; differences below ``z_threshold`` standard errors give
  ``ambiguous/unobservable``. Ground truth is never read here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from . import assignments as asg
from .candidates import (calibrate_noise, propose_finite_motions, seeds_from_proposals,
                         track_static_statistics)
from .geometry_fit import JointCandidate, Problem, fit_hypothesis
from .observations import TrackObservations
from .rigid_kinematics import axis_angle_deg, line_to_line_distance
from .visual_losses import whitened_residuals

DECISIONS = ("revolute", "prismatic", "static/no_motion", "ambiguous/unobservable")


def choose_reference_time(obs: TrackObservations) -> int:
    counts = np.bincount(obs.time[obs.has_3d], minlength=obs.frame_map.T)
    if counts.max() == 0:
        raise ValueError("no depth-anchored observations at all")
    return int(np.flatnonzero(counts >= 0.5 * counts.max())[0])


def make_heldout_split(T: int, ref: int, fraction: float, seed: int = 0) -> np.ndarray:
    """Evenly spaced interior held-out times (never the reference, first or last time)."""
    interior = np.array([t for t in range(1, T - 1) if t != ref])
    n = int(round(fraction * T))
    if n == 0 or len(interior) == 0:
        return np.zeros(0, int)
    step = len(interior) / n
    offset = (seed % max(int(step), 1)) if step >= 1 else 0
    pick = np.unique(np.clip((np.arange(n) * step + offset + step / 2).astype(int), 0, len(interior) - 1))
    return np.sort(interior[pick])


def calibrated_noise(obs, cfg, log=None):
    """Candidate-independent noise calibration on the static core; returns (noise, report)."""
    noise = copy.deepcopy(cfg.noise)
    report = dict(applied=False, prior=dict(sigma_px=noise.sigma_px, sigma_depth_m0=noise.sigma_depth_m0,
                                            sigma_depth_m2=noise.sigma_depth_m2))
    if not noise.calibrate_from_static_core:
        return noise, report
    st = track_static_statistics(obs, noise, None)
    n2d, n3d = obs.per_track_counts()
    core = (st["chi2_per_dof"] < 4.0) & (n3d >= 3)
    rep = calibrate_noise(obs, noise, core)
    report.update(rep)
    s = rep.get("sigma_px_measured")
    if s is not None and 0.2 <= s <= 10:
        noise.sigma_px = float(s)
        report["applied"] = True
    d = rep.get("depth_scale_measured")
    if d is not None and 0.2 <= d <= 10:
        noise.sigma_depth_m0 *= d
        noise.sigma_depth_m2 *= d
        report["applied"] = True
    if getattr(noise, "calibrate_depth_ar1", False):
        rho = static_core_depth_ar1(obs, core)
        report["depth_ar1_rho_measured"] = rho
        if rho is not None:
            noise.depth_ar1_rho = float(np.clip(rho, 0.0, 0.99))
            report["applied"] = True
    report["calibrated"] = dict(sigma_px=noise.sigma_px, sigma_depth_m0=noise.sigma_depth_m0,
                                sigma_depth_m2=noise.sigma_depth_m2, depth_ar1_rho=noise.depth_ar1_rho)
    if log:
        log(f"  noise calibration on {rep.get('n_static_tracks')} static-core tracks: {report['calibrated']}")
    return noise, report


def static_core_depth_ar1(obs, core_tracks, min_pairs=50, return_raw=False):
    """Lag-1 autocorrelation of de-meaned depth along static-core tracks (consecutive model times)."""
    h = obs.has_3d & core_tracks[obs.track]
    if h.sum() < min_pairs:
        return None
    idx = np.flatnonzero(h)
    o = idx[np.lexsort((obs.time[idx], obs.track[idx]))]
    tr, t, z = obs.track[o], obs.time[o], obs.z[o]
    med = np.zeros(obs.n_tracks)
    for k in np.unique(tr):
        med[k] = np.median(z[tr == k])
    r = z - med[tr]
    pair = (tr[1:] == tr[:-1]) & (t[1:] == t[:-1] + 1)
    if pair.sum() < min_pairs:
        return None
    a, b = r[:-1][pair], r[1:][pair]
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if not den > 0:
        return None
    raw = float(np.sum(a * b) / den)
    # de-meaning each short track biases the lag-1 estimate low (Marriott-Pope: E[r] ~ rho - (1+3 rho)/L)
    L = float(np.mean(np.bincount(tr)[np.unique(tr)]))
    corr = (L * raw + 1.0) / (L - 3.0) if L > 4 else raw
    return float(np.clip(corr, raw, 0.99)) if return_raw is False else (float(np.clip(corr, raw, 0.99)), raw, L)


@dataclass
class HypothesisFit:
    candidates: dict
    problem: Problem
    proposals: object
    noise_report: dict
    seed_scores: dict = field(default_factory=dict)


def fit_joint_hypotheses(obs: TrackObservations, calibs: dict, cfg, ref=None, log=None, use_depth=None) -> HypothesisFit:
    """Fit every configured hypothesis on the same observations (spec 10 API)."""
    use_depth = cfg.enable_depth if use_depth is None else use_depth
    ref = choose_reference_time(obs) if ref is None else ref
    noise, noise_report = calibrated_noise(obs, cfg, log)
    cfg_run = copy.deepcopy(cfg)
    cfg_run.noise = noise
    problem = Problem(obs, calibs, cfg_run, ref, use_depth=use_depth)
    proposals = propose_finite_motions(obs, noise, cfg_run.fit)
    if log:
        ok = [p for p in proposals.diagnostics["pairs"] if p.get("status") == "ok"]
        log(f"  finite-motion proposals: {len(ok)} ok of {len(proposals.diagnostics['pairs'])} pairs; "
            f"{int(problem.eligible.sum())}/{obs.n_tracks} tracks depth-anchored")
    out, seed_scores = {}, {}
    for kind in cfg.hypotheses:
        seeds = seeds_from_proposals(proposals, obs, ref, kind)[: cfg.fit.n_seeds]
        if not seeds:
            seed_scores[kind] = "no_motion_proposals"
            continue
        trials = []
        for s in seeds:
            c, nll = fit_hypothesis(problem, s, em_iterations=1, max_nfev=max(10, cfg.fit.least_squares_max_nfev // 3))
            trials.append((nll, s, c))
        seed_scores[kind] = [float(t[0]) for t in trials]
        best = min(trials, key=lambda t: t[0])
        b = best[2]
        from .candidates import Seed
        cont = Seed(kind, b.axis_w, b.origin_w, b.q, b.q_valid, dict(best[1].diagnostics, continued_from_best_seed=True))
        cand, nll = fit_hypothesis(problem, cont, log=log)
        cand.diagnostics["seed_training_nll"] = seed_scores[kind]
        out[kind] = cand
    return HypothesisFit(out, problem, proposals, noise_report, seed_scores)


# --------------------------------------------------------------------------- held-out scoring
@dataclass
class ModelDecision:
    label: str
    reasons: list
    scores: dict
    heldout_times: list
    n_scored_tracks: int
    track_pred_nll: dict = field(default_factory=dict)   # per-track held-out predictive NLL per model
    scored_mask: np.ndarray | None = None


def _heldout_track_ll(problem: Problem, obs_ho: TrackObservations, kind, axis, origin, q, anchors):
    from .observations import depth_sigma
    from .visual_losses import camera_stacks
    K, T_CW = camera_stacks(obs_ho, problem.calibs)
    ok = problem.eligible[obs_ho.track] & np.isfinite(anchors[obs_ho.track]).all(1)
    res = whitened_residuals(kind, axis, origin, q, np.nan_to_num(anchors), obs_ho, K, T_CW, problem.noise,
                             obs_mask=ok, use_depth=problem.use_depth)
    sz = depth_sigma(np.nan_to_num(obs_ho.z, nan=1.0), problem.noise, obs_ho.depth_edge, obs_ho.depth_cos)
    ll = asg.observation_loglik(res, problem.noise.sigma_px, sz, problem.noise)
    return asg.track_sum(obs_ho, ll, ok), asg.track_sum(obs_ho, asg.outlier_loglik(obs_ho, problem.noise, problem.use_depth), ok)


def score_candidates_on_heldout(fit: HypothesisFit, obs_ho: TrackObservations, cfg) -> ModelDecision:
    problem = fit.problem
    T = obs_ho.frame_map.T
    n_ho = np.bincount(obs_ho.track[obs_ho.uv_valid], minlength=obs_ho.n_tracks)
    scored = problem.eligible & (n_ho > 0)
    ll_s, ll_o = _heldout_track_ll(problem, obs_ho, "static", None, None, np.zeros(T), problem.static_anchors)
    # null model: static + outlier, responsibilities from training only
    pri0 = np.array([1 - cfg.fit.prior_outlier, cfg.fit.prior_outlier])
    Lt = np.stack([problem.ll_static, problem.ll_outlier], 1) + np.log(pri0)
    r0 = np.exp(Lt - logsumexp(Lt, axis=1, keepdims=True))
    pred = {"static": -logsumexp(np.stack([ll_s, ll_o], 1), b=r0, axis=1)}
    reasons, scores = [], {}
    for kind, c in fit.candidates.items():
        ll_m, _ = _heldout_track_ll(problem, obs_ho, kind, c.axis_w, c.origin_w, c.q, c.canonical_anchors_w)
        r = np.stack([c.static_probability, c.moving_probability, c.outlier_probability], 1)
        r = np.where(np.isfinite(r), r, np.array([1.0, 0.0, 0.0]))
        pred[kind] = -logsumexp(np.stack([ll_s, ll_m, ll_o], 1), b=np.clip(r, 1e-300, 1), axis=1)
    n = int(scored.sum())
    cluster = cfg.comparison.uncertainty_method == "cluster_robust"
    groups = None
    if cluster:
        from .validation_folds import surface_groups
        groups = surface_groups(problem.obs, cfg.comparison.group_voxel_m)

    def paired(a, b):
        d = pred[a][scored] - pred[b][scored]  # positive: b predicts better
        if n < 2:
            return dict(delta_nll=float(d.sum()) if n else 0.0, z=0.0, n=n)
        se = float(d.std(ddof=1) * np.sqrt(n))
        z_naive = float(d.sum() / se) if se > 0 else 0.0
        out = dict(delta_nll=float(d.sum()), z=z_naive, z_naive=z_naive, n=n, z_method="track_paired")
        if cluster:
            from .validation_folds import grouped_paired_statistics
            g = grouped_paired_statistics(d, groups[scored])
            out.update(z=g["z_cluster"], z_method="cluster_robust", n_groups=g["n_groups"],
                       design_effect=g.get("design_effect"), n_eff=g.get("n_eff"),
                       bootstrap_ci95=g.get("bootstrap_ci95"), bootstrap_p_second_better=g.get("bootstrap_p_second_better"),
                       largest_group_share=g.get("largest_group_share"))
        return out

    for k in pred:
        scores[f"heldout_nll_{k}"] = float(pred[k][scored].sum())
        scores[f"heldout_nll_per_track_{k}"] = float(pred[k][scored].mean()) if n else None
    kinds = [k for k in ("prismatic", "revolute") if k in fit.candidates]
    for k in kinds:
        scores[f"{k}_vs_static"] = paired("static", k)
    if len(kinds) == 2:
        scores["revolute_vs_prismatic"] = paired("prismatic", "revolute")
    z = cfg.comparison.z_threshold
    better_than_null = [k for k in kinds if scores[f"{k}_vs_static"]["z"] > z]
    if n < 2:
        label = "ambiguous/unobservable"
        reasons.append(f"only {n} tracks have held-out observations")
    elif not kinds:
        label = "static/no_motion"
        reasons.append("no finite-motion proposals above the noise threshold for any hypothesis")
    elif not better_than_null:
        spans = {k: _span_sigma(fit.candidates[k]) for k in kinds}
        if all(s is not None and s < cfg.comparison.min_motion_sigma for s in spans.values()):
            label = "static/no_motion"
            reasons.append(f"no hypothesis beats the static null on held-out data; joint span/sigma {spans}")
        else:
            label = "ambiguous/unobservable"
            reasons.append(f"no hypothesis beats the static null by z>{z}; span/sigma {spans}")
    elif len(better_than_null) == 1:
        other = [k for k in kinds if k not in better_than_null]
        if other and abs(scores["revolute_vs_prismatic"]["z"]) < z:
            label = "ambiguous/unobservable"
            reasons.append("only one hypothesis beats the null, but the P-vs-R difference is below threshold")
        else:
            label = better_than_null[0]
            reasons.append(f"{label} is the only hypothesis that beats the static null by z>{z}")
    else:
        zz = scores["revolute_vs_prismatic"]["z"]
        if abs(zz) < z:
            label = "ambiguous/unobservable"
            reasons.append(f"both beat the null; revolute-vs-prismatic held-out z={zz:.2f} below {z}")
        else:
            label = "revolute" if zz > 0 else "prismatic"
            reasons.append(f"held-out revolute-vs-prismatic z={zz:.2f}")
    if cfg.comparison.remote_hinge_policy == "observability" and "revolute" in fit.candidates:
        from .observability import observability_assessment
        oa = observability_assessment(fit.candidates["revolute"], problem, cfg)
        scores["revolute_observability"] = oa
        if label == "revolute" and not oa["geometry_identifiable"]:
            label = "ambiguous/unobservable"
            if "prismatic" in better_than_null:
                scores["occam_preference"] = "prismatic"
            reasons.append("revolute wins on held-out data but its geometry is not identifiable: " + "; ".join(oa["reasons"]))
        elif label == "ambiguous/unobservable" and not oa["geometry_identifiable"] and "prismatic" in better_than_null:
            scores["occam_preference"] = "prismatic"
            reasons.append("revolute geometry not identifiable: " + "; ".join(oa["reasons"]))
        scores["type_confidence"] = _type_confidence(label, scores)
        return ModelDecision(label, reasons, scores, [], n, track_pred_nll=pred, scored_mask=scored)
    if label == "revolute" and "prismatic" in better_than_null:
        rep = observability_report(fit.candidates["revolute"], problem, cfg.comparison.remote_hinge_ratio)
        if "remote_hinge_approximates_translation" in rep["degeneracy_flags"]:
            r = rep.get("hinge_radius_m", {}).get("median")
            label = "ambiguous/unobservable"
            scores["occam_preference"] = "prismatic"
            reasons.append(f"revolute wins on held-out data, but its hinge is {r:.2f} m from a moving support of extent "
                           f"{max(rep.get('moving_support_extent_m', [0])):.2f} m (arc {np.degrees(rep['q_span']):.2f} deg): "
                           "the remote-hinge degeneracy makes the 1-DOF type unidentifiable; the gain indicates a small "
                           "rotation component (e.g. joint play or calibration) on top of a translation; "
                           "occam_preference=prismatic")
    elif label == "ambiguous/unobservable" and "revolute" in fit.candidates and "prismatic" in better_than_null:
        rep = observability_report(fit.candidates["revolute"], problem, cfg.comparison.remote_hinge_ratio)
        if "remote_hinge_approximates_translation" in rep["degeneracy_flags"]:
            r = rep.get("hinge_radius_m", {}).get("median")
            scores["occam_preference"] = "prismatic"
            reasons.append(f"the revolute fit collapsed to its prismatic limit (hinge radius {r:.2f} m, arc "
                           f"{np.degrees(rep['q_span']):.2f} deg): the data are a translation that a remote hinge can "
                           "mimic; label kept ambiguous, occam_preference=prismatic")
    return ModelDecision(label, reasons, scores, [], n, track_pred_nll=pred, scored_mask=scored)


def _type_confidence(label, scores) -> dict:
    """Descriptive confidence of the joint TYPE (not of its geometry)."""
    rp = scores.get("revolute_vs_prismatic", {})
    z = rp.get("z")
    p = rp.get("bootstrap_p_second_better")
    if label not in ("revolute", "prismatic"):
        return dict(level="none", z=z, bootstrap_p_revolute_better=p)
    zz = abs(z or 0.0)
    return dict(level="high" if zz >= 4 else ("moderate" if zz >= 2 else "low"), z=z, z_method=rp.get("z_method"),
                bootstrap_p_revolute_better=p)


def _span_sigma(c: JointCandidate):
    if c.q_uncertainty is None:
        return None
    unc = np.nanmedian(c.q_uncertainty[c.q_valid & (c.q_uncertainty > 0)]) if np.any(c.q_valid & (c.q_uncertainty > 0)) else np.nan
    span = float(np.nanmax(c.q) - np.nanmin(c.q))
    return float(span / unc) if np.isfinite(unc) and unc > 0 else None


def bic_report(fit: HypothesisFit) -> dict:
    """BIC-type diagnostics on the *training* likelihood (reported, not used for selection)."""
    p = fit.problem
    n = int(p.obs.uv_valid.sum() + (p.obs.depth_valid & p.obs.mask_valid).sum())
    out = {}
    for kind, c in fit.candidates.items():
        k = (4 if kind == "revolute" else 2) + int(c.q_valid.sum()) - 1
        nll = c.diagnostics["training_nll"]
        out[kind] = dict(training_nll=nll, joint_params=k, bic=float(2 * nll + k * np.log(max(n, 1))))
    return out


def observability_report(c: JointCandidate, problem: Problem, remote_hinge_ratio: float | None = None) -> dict:
    """Spec 9.3: span, radii, support counts, conditioning, degeneracy flags."""
    obs = problem.obs
    moving = np.nan_to_num(c.moving_probability) > 0.5
    X = c.canonical_anchors_w[moving]
    rep = dict(kind=c.kind, n_moving_tracks=int(moving.sum()),
               q_span=float(np.nanmax(c.q) - np.nanmin(c.q)), q_units="rad" if c.kind == "revolute" else "m",
               span_over_sigma=_span_sigma(c),
               schur_condition=c.diagnostics.get("final_ls", {}).get("schur_condition"),
               axis_sigma_deg=c.diagnostics.get("final_ls", {}).get("axis_sigma_deg"),
               hinge_line_sigma_m=c.diagnostics.get("final_ls", {}).get("hinge_line_sigma_m"))
    per_t = np.zeros((len(obs.camera_ids), obs.frame_map.T), int)
    sel = moving[obs.track] & obs.uv_valid
    np.add.at(per_t, (obs.cam[sel], obs.time[sel]), 1)
    rep["moving_obs_per_camera_time"] = {cid: per_t[i].tolist() for i, cid in enumerate(obs.camera_ids)}
    flags = []
    if len(X) >= 2:
        span = np.nanpercentile(X, 95, axis=0) - np.nanpercentile(X, 5, axis=0)  # robust to stray anchors
        rep["moving_support_extent_m"] = span.tolist()
        rep["moving_support_extent_convention"] = "5-95 percentile range per world axis"
        if np.max(span) < 0.02:
            flags.append("tracks_from_tiny_patch")
    if c.kind == "revolute" and len(X):
        u, o = c.axis_w, c.origin_w
        r = np.linalg.norm((X - o) - np.outer((X - o) @ u, u), axis=1)
        rep["hinge_radius_m"] = dict(min=float(r.min()), median=float(np.median(r)), max=float(r.max()))
        if np.degrees(rep["q_span"]) < 2.0:
            flags.append("small_arc_lt_2deg")
        ratio = problem.cfg.comparison.remote_hinge_ratio if remote_hinge_ratio is None else remote_hinge_ratio
        rep["hinge_radius_over_extent"] = float(np.median(r) / max(np.max(rep.get("moving_support_extent_m", [0.0])), 1e-3))
        if rep["hinge_radius_over_extent"] > ratio:
            flags.append("remote_hinge_approximates_translation")
    if rep["span_over_sigma"] is not None and rep["span_over_sigma"] < 3:
        flags.append("q_span_within_3_sigma")
    rep["degeneracy_flags"] = flags
    return rep


def compare_to_reference(c: JointCandidate, axis_ref, origin_ref=None) -> dict:
    """DIAGNOSTIC ONLY (oracle comparison after fitting). Never used for fitting or selection."""
    out = dict(oracle_diagnostic=True, axis_error_deg=axis_angle_deg(c.axis_w, axis_ref))
    if c.kind == "revolute" and origin_ref is not None:
        out["hinge_line_distance_m"] = line_to_line_distance(c.axis_w, c.origin_w, axis_ref, origin_ref)
    return out
