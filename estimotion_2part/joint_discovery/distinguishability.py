"""R/P distinguishability in observation space and blocked held-out preference (spec sections 5 and 6).

Two different quantities, never merged:

1. ``prediction_separation`` -- could *this* camera/track setup tell the two fitted hypotheses apart?  Both
   candidates were fit on training data with their own anchors and q (nuisance parameters refit, never a frozen P
   against an optimised R).  delta_k = h_R(k) - h_P(k) on IDENTICAL material ids, cameras and times, in pixels
   and (separately) metric depth, compared with the *empirical* residual scale of the fits on the same tracks
   (not a nominal 0.5 px).  The common translation is removed per time (spec 5.1) before the spatially varying
   separation is measured.  Per-time-block aggregation; a diagonal I^2 is reported as descriptive only.

2. ``heldout_block_preference`` -- which hypothesis better predicts observations of withheld *time blocks*, on one
   candidate-independent common track set, bootstrapped over blocks.  q at held-out times is the training-only
   interpolation of the fit (documented trajectory-interpolation benchmark); no held-out measurement enters the fit.
"""
from __future__ import annotations

import numpy as np

from . import assignments as asg
from .blocked_statistics import block_bootstrap
from .observations import TrackObservations, depth_sigma
from .visual_losses import camera_stacks, predict_world, project_stacked, whitened_residuals


def _predict(c, kind, obs, idx, K, T_CW):
    X = c.canonical_anchors_w[obs.track[idx]]
    xw = predict_world(kind, c.axis_w, c.origin_w, c.q[obs.time[idx]], X)
    uv, z, _ = project_stacked(xw, obs.cam[idx], K, T_CW)
    return xw, uv, z


def prediction_separation(cands: dict, obs: TrackObservations, calibs, track_set, partition=None, k_sigma=2.0,
                          motion_fraction=0.5) -> dict:
    if not {"revolute", "prismatic"} <= set(cands):
        return dict(status="NOT_APPLICABLE", reason="both hypotheses are required")
    cR, cP = cands["revolute"], cands["prismatic"]
    K, T_CW = camera_stacks(obs, calibs)
    ok = (track_set[obs.track] & obs.uv_valid & np.isfinite(cR.canonical_anchors_w[obs.track]).all(1)
          & np.isfinite(cP.canonical_anchors_w[obs.track]).all(1))
    idx = np.flatnonzero(ok)
    if len(idx) < 30:
        return dict(status="INSUFFICIENT_COMMON_SUPPORT", n_obs=int(len(idx)))
    xR, uvR, zR = _predict(cR, "revolute", obs, idx, K, T_CW)
    xP, uvP, zP = _predict(cP, "prismatic", obs, idx, K, T_CW)
    uv_obs = obs.uv[idx]
    # empirical per-model residual scale on these very observations (robust)
    sR = 1.4826 * np.median(np.linalg.norm(uvR - uv_obs, axis=1))
    sP = 1.4826 * np.median(np.linalg.norm(uvP - uv_obs, axis=1))
    s_emp = float(max(min(sR, sP), 0.25))
    d_uv = np.linalg.norm(uvR - uvP, axis=1)
    d3 = xR - xP
    t = obs.time[idx]
    # remove the best common translation per time (spatially uniform component; spec 5.1)
    d3_var = d3.copy()
    for tt in np.unique(t):
        m = t == tt
        d3_var[m] -= np.median(d3[m], axis=0)
    d3n, d3v = np.linalg.norm(d3, axis=1), np.linalg.norm(d3_var, axis=1)
    dz = np.abs(zR - zP)
    dz_valid = obs.depth_valid[idx] & np.isfinite(obs.z[idx])
    # motion phase: times where either model has moved >= motion_fraction of its span
    qR = np.abs(cR.q - cR.q[np.nanargmin(np.abs(cR.q))])
    big_t = np.flatnonzero(qR >= motion_fraction * np.nanmax(qR)) if np.nanmax(qR) > 0 else np.arange(len(cR.q))
    big = np.isin(t, big_t)
    blocks = partition.block_of() if partition is not None else np.arange(obs.frame_map.T)
    per_block = {}
    for b in np.unique(blocks[t]):
        m = (blocks[t] == b) & big
        if m.sum() >= 5:
            per_block[int(b)] = dict(rms_uv_px=float(np.sqrt(np.mean(d_uv[m] ** 2))),
                                     sep_over_sigma=float(np.sqrt(np.mean(d_uv[m] ** 2)) / s_emp),
                                     rms_3d_varying_mm=float(1000 * np.sqrt(np.mean(d3v[m] ** 2))))
    seps = [v["sep_over_sigma"] for v in per_block.values()]
    med = float(np.median(seps)) if seps else 0.0
    status = "DISTINGUISHABLE" if seps and med >= k_sigma else "INDISTINGUISHABLE"

    def q(x, p):
        return float(np.percentile(x, p)) if len(x) else None

    return dict(status=status, k_sigma=k_sigma, empirical_sigma_px=s_emp, sigma_R_px=float(sR), sigma_P_px=float(sP),
                n_obs=int(len(idx)), n_tracks=int(len(np.unique(obs.track[idx]))),
                median_block_separation_over_sigma=med, per_block=per_block,
                uv_px=dict(max=q(d_uv[big], 100), p90=q(d_uv[big], 90), median=q(d_uv[big], 50)),
                metric_3d_mm=dict(max=q(1000 * d3n[big], 100), p90=q(1000 * d3n[big], 90)),
                metric_3d_spatially_varying_mm=dict(max=q(1000 * d3v[big], 100), p90=q(1000 * d3v[big], 90),
                                                    median=q(1000 * d3v[big], 50)),
                depth_mm=dict(p90=q(1000 * dz[big & dz_valid], 90)),
                descriptive_I2_diagonal=float(np.sum((d_uv / s_emp) ** 2)),
                note="I2 with a diagonal covariance is descriptive only (correlated tracks); decisions use the "
                     "per-time-block median separation against the empirical residual scale")


def heldout_block_preference(fit, test_obs: TrackObservations, partition, track_set, *, min_blocks=4, n_boot=2000,
                             seed=0) -> dict:
    """Per-observation held-out log-likelihood difference R - P on a common candidate-independent track set,
    summed per held-out time block and bootstrapped over blocks (positive = revolute predicts better)."""
    cands = fit.candidates
    if not {"revolute", "prismatic"} <= set(cands):
        return dict(status="NOT_APPLICABLE")
    problem = fit.problem
    K, T_CW = camera_stacks(test_obs, problem.calibs)
    common = (track_set & problem.eligible & np.isfinite(cands["revolute"].canonical_anchors_w).all(1)
              & np.isfinite(cands["prismatic"].canonical_anchors_w).all(1))
    ok = common[test_obs.track]
    sz = depth_sigma(np.nan_to_num(test_obs.z, nan=1.0), problem.noise, test_obs.depth_edge, test_obs.depth_cos)
    ll = {}
    for k, c in cands.items():
        res = whitened_residuals(k, c.axis_w, c.origin_w, c.q, np.nan_to_num(c.canonical_anchors_w), test_obs, K, T_CW,
                                 problem.noise, obs_mask=ok, use_depth=problem.use_depth)
        ll[k] = asg.observation_loglik(res, problem.noise.sigma_px, sz, problem.noise)
    d = (ll["revolute"] - ll["prismatic"])
    blocks = partition.block_of()[test_obs.time]
    held = [b for b, h in enumerate(partition.heldout) if h]
    per_block = [d[ok & (blocks == b)].sum() for b in held if (ok & (blocks == b)).any()]
    n_tracks = int(len(np.unique(test_obs.track[ok])))
    bb = block_bootstrap(per_block, n_boot=n_boot, seed=seed)
    out = dict(n_heldout_blocks=len(per_block), per_block_delta_ll=[float(x) for x in per_block],
               common_tracks=n_tracks, n_obs=int(ok.sum()), bootstrap=bb, min_blocks=min_blocks,
               q_heldout="training-only interpolation of each fit (trajectory-interpolation benchmark)")
    if len(per_block) < min_blocks or bb["ci"] is None:
        out.update(status="INSUFFICIENT_INDEPENDENT_EVIDENCE", preference=None)
    elif bb["ci"][0] > 0:
        out.update(status="OK", preference="revolute")
    elif bb["ci"][1] < 0:
        out.update(status="OK", preference="prismatic")
    else:
        out.update(status="OK", preference="none")
    return out


def complexity_report(fit) -> dict:
    """Declared parameter counts; raw-pixel BIC is diagnostic only (correlated observations)."""
    out = {}
    for k, c in fit.candidates.items():
        nq = int(np.sum(np.isfinite(c.q)))
        out[k] = dict(joint_params=(2 + 2 if k == "revolute" else 2) + nq,
                      note="axis(2) + hinge line(2, revolute only) + one q per time")
    out["extra_revolute_params"] = 2
    out["policy"] = ("held-out block prediction already penalises over-fitting of the 2 extra hinge parameters; "
                     "no raw-pixel-count BIC is used for decisions")
    return out
