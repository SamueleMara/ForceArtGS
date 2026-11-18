"""Grouped / camera / blocked-time held-out evaluation (Stage C1/C2).

Why: the baseline z of ``model_selection`` divides the summed per-track held-out difference by
``std * sqrt(n_tracks)``, treating tracks as independent. Tracks on one surface patch, seen by one
camera and queried in one window, share depth bias, tracker drift and the interpolated ``q``; the
naive z then overstates significance. Here differences are aggregated over *candidate-independent
spatiotemporal surface groups* and the uncertainty is estimated over groups (cluster-robust SE and
a group bootstrap), with the grouping sensitivity disclosed.

Fold types and the rule that gives ``q`` at test timestamps:

``interleaved_time``     evenly spaced interior times held out (baseline); q is *interpolated*
                         between neighbouring training times.
``contiguous_block``     an interior block held out; q is interpolated *across the gap*.
``forward_block``        the last block held out; q is *held constant* after the last training
                         time (extrapolation): a weak predictive test, reported, not used to decide.
``heldout_camera``       the joint (axis, hinge, q(t) at every time) is fit on the other camera(s);
                         q at test times is *constrained by another camera*. Anchors of the test
                         camera's tracks are nuisance parameters estimated with the joint frozen on
                         that camera's even times; scoring uses its odd times.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from . import assignments as asg
from .geometry_fit import refine_anchors
from .observations import TrackObservations, depth_sigma
from .visual_losses import camera_stacks, whitened_residuals

FOLD_KINDS = ("interleaved_time", "contiguous_block", "forward_block", "heldout_camera")


@dataclass
class FoldSpec:
    name: str
    kind: str
    train_time_mask: np.ndarray
    test_time_mask: np.ndarray
    train_cameras: list
    test_cameras: list
    q_rule: str = ""
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in FOLD_KINDS:
            raise ValueError(f"fold kind must be one of {FOLD_KINDS}")
        self.train_time_mask = np.asarray(self.train_time_mask, bool)
        self.test_time_mask = np.asarray(self.test_time_mask, bool)
        if self.kind != "heldout_camera" and np.any(self.train_time_mask & self.test_time_mask):
            raise ValueError("train and test times overlap in a time fold")

    def to_json(self) -> dict:
        return dict(name=self.name, kind=self.kind, train_times=np.flatnonzero(self.train_time_mask).tolist(),
                    test_times=np.flatnonzero(self.test_time_mask).tolist(), train_cameras=self.train_cameras,
                    test_cameras=self.test_cameras, q_rule=self.q_rule, notes=self.notes)


def time_folds(T, ref, cameras, block_fraction=0.25, interleaved_times=None) -> list:
    """Interleaved (given), one interior contiguous block, and one forward block. ``ref`` never tested."""
    folds = []
    if interleaved_times is not None:
        te = np.zeros(T, bool)
        te[np.asarray(interleaved_times, int)] = True
        folds.append(FoldSpec("interleaved", "interleaved_time", ~te, te, list(cameras), list(cameras),
                              "interpolated between neighbouring training times"))
    L = max(2, int(round(block_fraction * T)))
    start = (T - L) // 2
    te = np.zeros(T, bool)
    te[start:start + L] = True
    te[ref] = False
    folds.append(FoldSpec("block_mid", "contiguous_block", ~te & (np.arange(T) != -1), te, list(cameras), list(cameras),
                          "interpolated across the held-out gap (no observation constrains q inside it)"))
    te = np.zeros(T, bool)
    te[T - L:] = True
    te[ref] = False
    folds.append(FoldSpec("block_forward", "forward_block", ~te, te, list(cameras), list(cameras),
                          "held constant after the last training time (extrapolation; diagnostic only)"))
    return folds


def camera_folds(T, cameras) -> list:
    out = []
    even = (np.arange(T) % 2) == 0
    for c in cameras:
        others = [o for o in cameras if o != c]
        if not others:
            continue
        out.append(FoldSpec(f"camera_{'+'.join(others)}_to_{c}", "heldout_camera", np.ones(T, bool), ~even,
                            others, [c], "constrained by another camera at every test time",
                            dict(test_camera_anchor_times="even model times", test_camera_scored_times="odd model times")))
    return out


# --------------------------------------------------------------------------- candidate-independent groups
def surface_groups(obs: TrackObservations, voxel_m=0.05, by_query_window=True):
    """Group id per track: (camera, 3D voxel of the first measured point[, query window])."""
    first = np.full((obs.n_tracks, 3), np.nan)
    h = np.flatnonzero(obs.has_3d)
    order = h[np.lexsort((obs.time[h], obs.track[h]))]
    tr, i = np.unique(obs.track[order], return_index=True)
    first[tr] = obs.x_w[order[i]]
    ok = np.isfinite(first).all(1)
    key = np.full((obs.n_tracks, 5), -1, np.int64)
    key[:, 0] = obs.track_camera
    key[ok, 1:4] = np.floor(first[ok] / voxel_m).astype(np.int64)
    if by_query_window:
        key[:, 4] = obs.track_query_time
    _, g = np.unique(key, axis=0, return_inverse=True)
    return g.ravel()


def grouped_paired_statistics(d_track, groups, n_boot=2000, seed=0) -> dict:
    """Cluster-robust inference for ``D = sum_i d_i`` (positive: the second model is better).

    ``z_naive`` = D / (std(d) sqrt(n)) (baseline); ``z_cluster`` uses group sums ``S_g``:
    ``Var(D) ~ G/(G-1) sum_g (S_g - mean S)^2``. ``n_eff`` = G * (sum_i d_i^2 ... ) is reported as
    the design-effect-corrected track count. The group bootstrap resamples whole groups."""
    d = np.asarray(d_track, float)
    g = np.asarray(groups)
    ok = np.isfinite(d)
    d, g = d[ok], g[ok]
    n = len(d)
    if n < 2:
        return dict(n_tracks=n, n_groups=0, delta=float(d.sum()) if n else 0.0, z_naive=0.0, z_cluster=0.0)
    D = float(d.sum())
    z_naive = D / (d.std(ddof=1) * np.sqrt(n)) if d.std(ddof=1) > 0 else 0.0
    ug, gi = np.unique(g, return_inverse=True)
    S = np.bincount(gi, d)
    G = len(S)
    var = G / max(G - 1, 1) * np.sum((S - S.mean()) ** 2)
    z_cl = D / np.sqrt(var) if var > 0 else 0.0
    deff = (var / (d.var(ddof=1) * n)) if d.var(ddof=1) > 0 else np.nan
    rng = np.random.default_rng(seed)
    boots = S[rng.integers(0, G, (n_boot, G))].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n_tracks=n, n_groups=G, delta=D, z_naive=float(z_naive), z_cluster=float(z_cl),
                design_effect=float(deff) if np.isfinite(deff) else None,
                n_eff=float(n / deff) if np.isfinite(deff) and deff > 0 else None,
                bootstrap_ci95=[float(lo), float(hi)], bootstrap_p_second_better=float(np.mean(boots > 0)),
                largest_group_share=float(np.max(np.abs(S)) / max(np.sum(np.abs(S)), 1e-12)))


def grouping_sensitivity(d_track, obs: TrackObservations, voxels=(0.03, 0.05, 0.10)) -> dict:
    out = {}
    for v in voxels:
        for qw in (True, False):
            out[f"voxel{int(v * 100)}cm_{'qwin' if qw else 'noqwin'}"] = grouped_paired_statistics(
                d_track, surface_groups(obs, v, qw))
    out["per_camera_only"] = grouped_paired_statistics(d_track, obs.track_camera)
    return out


def decision_statistics(decision, obs: TrackObservations, voxel_m=0.05) -> dict:
    """Grouped versions of every pairwise comparison in a ``ModelDecision``."""
    pred, sc = decision.track_pred_nll, decision.scored_mask
    if not pred or sc is None:
        return {}
    g = surface_groups(obs, voxel_m)
    out = {}
    pairs = [(a, b) for a, b in (("static", "prismatic"), ("static", "revolute"), ("prismatic", "revolute"))
             if a in pred and b in pred]
    for a, b in pairs:
        d = np.where(sc, pred[a] - pred[b], np.nan)
        key = "revolute_vs_prismatic" if (a, b) == ("prismatic", "revolute") else f"{b}_vs_static"
        out[key] = grouped_paired_statistics(d, g)
        out[key]["grouping_sensitivity"] = {k: dict(z_cluster=v["z_cluster"], n_groups=v["n_groups"])
                                            for k, v in grouping_sensitivity(d, obs).items()}
    return out


# --------------------------------------------------------------------------- frozen-joint predictive scoring
def frozen_predictive_nll(joints: dict, obs_train: TrackObservations, obs_test: TrackObservations, calibs, noise,
                          priors=(0.65, 0.30, 0.05), min_train_obs=2):
    """Per-track predictive NLL on ``obs_test`` for frozen joints (and the static null).

    Anchors and mixture responsibilities come from ``obs_train`` only (same tracks). ``joints``:
    name -> object with kind/axis/origin/q (q defined at every model time)."""
    K, T_CW = camera_stacks(obs_train, calibs)
    n2d, n3d = obs_train.per_track_counts()
    elig = (n3d >= 1) & (n2d >= min_train_obs)
    X0 = np.full((obs_train.n_tracks, 3), np.nan)
    tf = np.full(obs_train.n_tracks, 0)
    h = np.flatnonzero(obs_train.has_3d)
    order = h[np.lexsort((obs_train.time[h], obs_train.track[h]))]
    tr, i = np.unique(obs_train.track[order], return_index=True)
    X0[tr] = obs_train.x_w[order[i]]
    tf[tr] = obs_train.time[order[i]]
    from .rigid_kinematics import apply_joint_inverse

    def track_ll(kind, axis, origin, q, X, ob):
        ok = elig[ob.track] & np.isfinite(X[ob.track]).all(1)
        res = whitened_residuals(kind, axis, origin, q, np.nan_to_num(X), ob, K, T_CW, noise, obs_mask=ok)
        sz = depth_sigma(np.nan_to_num(ob.z, nan=1.0), noise, ob.depth_edge, ob.depth_cos)
        return asg.track_sum(ob, asg.observation_loglik(res, noise.sigma_px, sz, noise), ok)

    Xs = refine_anchors("static", None, None, np.zeros(obs_train.frame_map.T), X0, obs_train, K, T_CW, noise, elig)
    tr_s = track_ll("static", None, None, np.zeros(obs_train.frame_map.T), Xs, obs_train)
    te_s = track_ll("static", None, None, np.zeros(obs_train.frame_map.T), Xs, obs_test)
    okt = elig[obs_train.track]
    tr_o = asg.track_sum(obs_train, asg.outlier_loglik(obs_train, noise), okt)
    te_o = asg.track_sum(obs_test, asg.outlier_loglik(obs_test, noise), elig[obs_test.track])
    pri0 = np.array([1 - priors[2], priors[2]])
    L0 = np.stack([tr_s, tr_o], 1) + np.log(pri0)
    r0 = np.exp(L0 - logsumexp(L0, 1, keepdims=True))
    pred = {"static": -logsumexp(np.stack([te_s, te_o], 1), b=r0, axis=1)}
    post = {}
    for name, j in joints.items():
        X = X0.copy()
        ok = np.isfinite(X).all(1)
        X[ok] = apply_joint_inverse(j.kind, j.axis, j.origin, np.asarray(j.q)[tf[ok]], X[ok])
        X = refine_anchors(j.kind, j.axis, j.origin, np.asarray(j.q), X, obs_train, K, T_CW, noise, elig)
        tr_m = track_ll(j.kind, j.axis, j.origin, np.asarray(j.q), X, obs_train)
        te_m = track_ll(j.kind, j.axis, j.origin, np.asarray(j.q), X, obs_test)
        r = asg.responsibilities(tr_s, tr_m, tr_o, np.asarray(priors), elig)
        r = np.where(np.isfinite(r), r, np.array([1.0, 0.0, 0.0]))
        pred[name] = -logsumexp(np.stack([te_s, te_m, te_o], 1), b=np.clip(r, 1e-300, 1), axis=1)
        post[name] = r
    n_te = np.bincount(obs_test.track[obs_test.uv_valid], minlength=obs_test.n_tracks)
    scored = elig & (n_te > 0)
    return pred, scored, post


def run_fold(obs: TrackObservations, calibs, cfg, fold: FoldSpec, ref, log=None) -> dict:
    """Fit both hypotheses on the fold's training data and score the fold's test data."""
    from .depth_conflict_audit import FrozenJoint
    from .model_selection import calibrated_noise, fit_joint_hypotheses
    cams = obs.camera_ids
    tr_cam = np.isin(np.array(cams)[obs.cam], fold.train_cameras)
    te_cam = np.isin(np.array(cams)[obs.cam], fold.test_cameras)
    if fold.kind == "heldout_camera":
        fit_obs = obs.select_obs(tr_cam)
        anchor_obs = obs.select_obs(te_cam & fold.train_time_mask[obs.time] & ((obs.time % 2) == 0))
        test_obs = obs.select_obs(te_cam & fold.test_time_mask[obs.time])
    else:
        fit_obs = obs.select_obs(tr_cam & fold.train_time_mask[obs.time])
        anchor_obs = fit_obs
        test_obs = obs.select_obs(te_cam & fold.test_time_mask[obs.time])
    if fit_obs.n_obs == 0 or test_obs.n_obs == 0:
        return dict(fold=fold.to_json(), status="EMPTY", reason="no training or test observations")
    fit = fit_joint_hypotheses(fit_obs, calibs, cfg, ref=ref, log=log)
    noise = fit.problem.noise
    joints = {k: FrozenJoint(k, c.kind, c.axis_w, c.origin_w, c.q, "fitted", f"fold {fold.name}")
              for k, c in fit.candidates.items()}
    pred, scored, post = frozen_predictive_nll(joints, anchor_obs, test_obs, calibs, noise)
    g = surface_groups(obs)
    res = dict(fold=fold.to_json(), status="OK", n_test_tracks=int(scored.sum()),
               test_camera_coverage={c: int((scored & (obs.track_camera == i)).sum()) for i, c in enumerate(cams)},
               moving_obs_in_test=int(sum((np.nan_to_num(post[k][:, 1]) > 0.5)[test_obs.track].sum() for k in post)),
               q_at_test={k: _q_status(fit.candidates[k], fold) for k in fit.candidates},
               candidates={k: dict(axis=c.axis_w.tolist(), origin=None if c.origin_w is None else c.origin_w.tolist(),
                                   q_span=float(np.nanmax(c.q) - np.nanmin(c.q)),
                                   n_moving=int((np.nan_to_num(c.moving_probability) > 0.5).sum()))
                           for k, c in fit.candidates.items()}, comparisons={})
    for a, b in (("static", "prismatic"), ("static", "revolute"), ("prismatic", "revolute")):
        if a in pred and b in pred:
            d = np.where(scored, pred[a] - pred[b], np.nan)
            res["comparisons"][f"{b}_vs_{a}"] = grouped_paired_statistics(d, g)
    return res


def _q_status(c, fold: FoldSpec) -> dict:
    te = np.flatnonzero(fold.test_time_mask)
    obs_t = np.asarray(c.q_valid, bool)
    if fold.kind == "heldout_camera":
        status = ["constrained_by_other_camera" if obs_t[t] else "interpolated" for t in te]
    else:
        lo = np.flatnonzero(obs_t).min() if obs_t.any() else 0
        hi = np.flatnonzero(obs_t).max() if obs_t.any() else 0
        status = ["observed_in_training(leak)" if obs_t[t] else ("interpolated" if lo < t < hi else "extrapolated_constant")
                  for t in te]
    return {s: status.count(s) for s in set(status)}


def fold_label_stability(fold_results: list, z=2.0) -> dict:
    """Per-fold labels from cluster-robust z (revolute_vs_prismatic and each vs static)."""
    labels = []
    for r in fold_results:
        if r.get("status") != "OK":
            labels.append((r["fold"]["name"], "EMPTY"))
            continue
        c = r["comparisons"]
        zs = {k: v["z_cluster"] for k, v in c.items()}
        beat = [k for k in ("prismatic", "revolute") if zs.get(f"{k}_vs_static", 0) > z]
        rp = zs.get("revolute_vs_prismatic", 0)
        if not beat:
            lab = "no_motion_or_ambiguous"
        elif abs(rp) < z:
            lab = "ambiguous"
        else:
            lab = "revolute" if rp > 0 else "prismatic"
        labels.append((r["fold"]["name"], lab))
    vals = [l for _, l in labels if l != "EMPTY"]
    return dict(per_fold=dict(labels), stable=len(set(vals)) == 1 if vals else False)
