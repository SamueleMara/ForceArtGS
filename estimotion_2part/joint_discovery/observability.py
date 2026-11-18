"""Physical observability of a revolute hypothesis vs its translation limit (Stage C3).

A far-hinge small-angle rotation approaches a translation (theta -> 0, r -> inf, r*theta finite);
a marginally better NLL does not identify a physical hinge. Instead of a fixed
``hinge_radius / extent > 5`` cutoff, three measured quantities are reported:

1. translation distinguishability: on the observed moving support, the image distance between the
   revolute prediction ``A_R(q_t) X`` and the *best translation* of the same support (least-squares
   ``X + d_t``), projected through each observing camera, at large motion (|q| >= 2/3 max), as the
   p90 in units of the calibrated ``sigma_px``. At the noise level no camera can tell the finite
   hinge from a translation.
2. ``hinge_within_support``: the hinge line passes within ``margin * extent`` of the observed moving
   support (real hinges lie at the part boundary). A line far outside it is not a supported
   physical hinge, and its numerically determined origin is not reported as identified.
3. Fisher information of (axis, hinge line) after marginalizing anchors and q (Schur covariance):
   singular / unbounded -> not identifiable.

``geometry_identifiable`` requires all three. ``threshold_sensitivity`` re-evaluates the outcome on
a grid of thresholds as a *robustness disclosure*, never to choose the answer.
"""
from __future__ import annotations

import numpy as np

from .rigid_kinematics import apply_joint
from .visual_losses import project_stacked


def best_translation(X, X_t):
    """Least-squares translation per time mapping support X [N,3] to X_t [T,N,3]: d_t = mean(X_t - X)."""
    return np.nanmean(X_t - X[None], axis=1)


def translation_distinguishability(c, problem, max_tracks=3000, seed=0) -> dict:
    obs = problem.obs
    if c.kind != "revolute":
        return dict(applicable=False)
    mv = (np.nan_to_num(c.moving_probability) > 0.5) & np.isfinite(c.canonical_anchors_w).all(1)
    tr_ids = np.flatnonzero(mv)
    if len(tr_ids) < 3:
        return dict(applicable=True, status="insufficient_support", n_tracks=int(len(tr_ids)))
    rng = np.random.default_rng(seed)
    if len(tr_ids) > max_tracks:
        tr_ids = np.sort(rng.choice(tr_ids, max_tracks, replace=False))
    X = c.canonical_anchors_w[tr_ids]
    T = obs.frame_map.T
    Xt = np.stack([apply_joint("revolute", c.axis_w, c.origin_w, c.q[t], X) for t in range(T)])
    d = best_translation(X, Xt)
    sel = np.isin(obs.track, tr_ids) & obs.uv_valid
    loc = -np.ones(obs.n_tracks, int)
    loc[tr_ids] = np.arange(len(tr_ids))
    i = loc[obs.track[sel]]
    t = obs.time[sel]
    cam = obs.cam[sel]
    uv_r, _, _ = project_stacked(Xt[t, i], cam, problem.K, problem.T_CW)
    uv_p, _, _ = project_stacked(X[i] + d[t], cam, problem.K, problem.T_CW)
    dpx = np.linalg.norm(uv_r - uv_p, axis=1)
    qa = np.abs(np.asarray(c.q, float) - c.q[problem.ref])
    large = qa[t] >= 0.67 * np.nanmax(qa)          # where the motion is large; early frames carry no signal
    sig = float(problem.noise.sigma_px)
    per_cam = {}
    for ci, cid in enumerate(obs.camera_ids):
        m = (cam == ci) & large
        per_cam[cid] = dict(n_large_motion=int(m.sum()),
                            p90_px=float(np.percentile(dpx[m], 90)) if m.any() else None,
                            median_px=float(np.median(dpx[m])) if m.any() else None,
                            p90_over_sigma=float(np.percentile(dpx[m], 90) / sig) if m.any() else None)
    best = max((v["p90_over_sigma"] or 0.0) for v in per_cam.values())
    return dict(applicable=True, status="OK", n_tracks=int(len(tr_ids)), n_obs=int(sel.sum()),
                median_px_all=float(np.median(dpx)), large_motion_definition="|q - q_ref| >= 0.67 max",
                best_camera_p90_over_sigma=best, per_camera=per_cam,
                best_translation_span_m=float(np.linalg.norm(d[-1] - d[0])), sigma_px=sig)


def hinge_support_check(c, margin=1.0) -> dict:
    if c.kind != "revolute":
        return dict(applicable=False)
    mv = (np.nan_to_num(c.moving_probability) > 0.5) & np.isfinite(c.canonical_anchors_w).all(1)
    X = c.canonical_anchors_w[mv]
    if len(X) < 3:
        return dict(applicable=True, status="insufficient_support")
    lo, hi = np.percentile(X, 5, 0), np.percentile(X, 95, 0)
    extent = float(np.max(hi - lo))
    inner = X[np.all((X >= lo) & (X <= hi), 1)]
    u, o = c.axis_w, c.origin_w
    rel = inner - o
    dist = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
    dmin = float(np.min(dist))
    return dict(applicable=True, status="OK", support_extent_m=extent, hinge_to_support_min_m=dmin,
                hinge_to_support_median_m=float(np.median(dist)), margin=margin,
                hinge_within_support=bool(dmin <= margin * extent))


def fisher_check(c, max_axis_sigma_deg=10.0, max_line_sigma_rel=0.5) -> dict:
    ls = c.diagnostics.get("final_ls", {}) if c.diagnostics else {}
    ax, hl, cond = ls.get("axis_sigma_deg"), ls.get("hinge_line_sigma_m"), ls.get("schur_condition")
    out = dict(axis_sigma_deg=ax, hinge_line_sigma_m=hl, schur_condition=cond,
               note="Gauss-Newton covariance assumes independent residuals; correlated tracks make it optimistic")
    if ax is None or not np.isfinite(ax) or cond is None or not np.isfinite(cond) or cond > 1e14:
        out.update(fisher_ok=False, reason="covariance unavailable or singular")
        return out
    ok = ax <= max_axis_sigma_deg
    if c.kind == "revolute":
        ok &= hl is not None and np.isfinite(hl)
    out.update(fisher_ok=bool(ok), reason="" if ok else "axis/hinge covariance too large")
    return out


def q_span_uncertainty(c) -> dict:
    q = np.asarray(c.q, float)
    span = float(np.nanmax(q) - np.nanmin(q))
    unc = c.q_uncertainty
    if unc is None:
        return dict(q_span=span, q_span_sigma=None)
    i, j = int(np.nanargmax(q)), int(np.nanargmin(q))
    s = float(np.sqrt(np.nansum([np.nan_to_num(unc[i]) ** 2, np.nan_to_num(unc[j]) ** 2])))
    return dict(q_span=span, q_span_sigma=s, q_span_over_sigma=float(span / s) if s > 0 else None)


def observability_assessment(c, problem, cfg) -> dict:
    """Combined, physically stated identifiability report for one candidate."""
    comp = cfg.comparison
    out = dict(kind=c.kind, q_span=q_span_uncertainty(c))
    if c.kind != "revolute":
        out.update(geometry_identifiable=True, finite_hinge_distinguishable_from_translation=None,
                   reasons=["prismatic: axis direction only (no hinge origin)"])
        return out
    tdist = translation_distinguishability(c, problem)
    hs = hinge_support_check(c, comp.hinge_support_margin)
    fi = fisher_check(c)
    k = comp.min_translation_distinguishability_sigma
    dist_ok = tdist.get("status") == "OK" and tdist["best_camera_p90_over_sigma"] >= k
    hinge_ok = hs.get("status") == "OK" and hs["hinge_within_support"]
    reasons = []
    if not dist_ok:
        reasons.append(f"at large |q| the revolute and its best translation differ by only "
                       f"{tdist.get('best_camera_p90_over_sigma', 0):.2f} sigma_px (p90, best camera) < {k}")
    if not hinge_ok:
        reasons.append(f"hinge line {hs.get('hinge_to_support_min_m', np.nan):.3f} m from the observed moving support "
                       f"(extent {hs.get('support_extent_m', np.nan):.3f} m, margin {comp.hinge_support_margin})")
    if not fi["fisher_ok"]:
        reasons.append(f"Fisher information: {fi['reason']}")
    out.update(translation_distinguishability=tdist, hinge_support=hs, fisher=fi,
               finite_hinge_distinguishable_from_translation=bool(dist_ok),
               geometry_identifiable=bool(dist_ok and hinge_ok and fi["fisher_ok"]), reasons=reasons,
               report_hinge_origin=bool(dist_ok and hinge_ok and fi["fisher_ok"]))
    out["threshold_sensitivity"] = {
        f"sigma{p}_margin{m}": bool(tdist.get("status") == "OK" and tdist["best_camera_p90_over_sigma"] >= p
                                 and hs.get("status") == "OK" and hs["hinge_to_support_min_m"] <= m * hs["support_extent_m"]
                                 and fi["fisher_ok"])
        for p in (2.0, 3.0, 5.0) for m in (0.5, 1.0, 2.0)}
    return out
