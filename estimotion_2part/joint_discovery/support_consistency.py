"""Candidate-independent support and structured consistency checks (spec section 3).

* ``S_core``      confident fixed-world tracks (whitened spread small, >= ``min_3d`` depth anchors);
* ``M_proposal``  tracks that demonstrably move (whitened spread large AND metric displacement above the noise),
                  decided *before* and independently of any R/P hypothesis;
* ``U``           everything else -- never silently static.

Only training-time observations are used.  The consistency report flags evidence that one 1-DoF rigid field cannot
explain the moving support (second rigid group, moving base, low inlier coverage, tool contamination).
"""
from __future__ import annotations

import numpy as np

from .blocked_statistics import spatial_clusters
from .candidates import kabsch, ransac_rigid, track_static_statistics
from .motion_field import PointTable
from .observations import TrackObservations
from .rigid_kinematics import log_so3


def robust_track_displacement(obs: TrackObservations, noise) -> tuple:
    """Per track: p90 of |X(t) - median_t X(t)| [m] and the median 3D measurement sigma [m] (depth + lateral)."""
    from .observations import depth_sigma
    N = obs.n_tracks
    h = obs.has_3d & ~obs.depth_edge
    tr, X = obs.track[h], obs.x_w[h]
    sz = depth_sigma(obs.z[h], noise, obs.depth_edge[h], obs.depth_cos[h])
    disp, sig = np.full(N, np.nan), np.full(N, np.nan)
    order = np.argsort(tr, kind="stable")
    tr, X, sz = tr[order], X[order], sz[order]
    bounds = np.flatnonzero(np.diff(tr)) + 1
    for g, x, s in zip(np.split(tr, bounds), np.split(X, bounds), np.split(sz, bounds)):
        if len(g):
            d = np.linalg.norm(x - np.median(x, 0), axis=1)
            disp[g[0]] = np.percentile(d, 90)
            sig[g[0]] = np.median(s)
    return disp, sig


def candidate_independent_support(obs: TrackObservations, noise, *, static_chi2=4.0, min_disp_m=0.008,
                                  k_sigma=7.0, min_3d=3) -> dict:
    """S/M/U from per-track motion significance, decided before (and independently of) any R/P hypothesis.

    span = 2 * p90 |X(t) - median X| (~ the full motion range of a monotonic track, robust to single outliers).
    M: span >= max(min_disp_m, k_sigma * sigma_track).  For pure noise span ~ 5 sigma, so k_sigma = 7 keeps static
    tracks out.  S: chi2/dof of a fixed point < static_chi2 AND span < max(2.5 * sigma, 5 mm).  Everything else: U.
    No per-frame whitened chi2 threshold decides M -- it averages motion over stationary frames and loses short arcs."""
    st = track_static_statistics(obs, noise, None)
    n2d, n3d = obs.per_track_counts()
    disp, sig = robust_track_displacement(obs, noise)
    span = 2 * disp
    sig = np.nan_to_num(sig, nan=np.inf)
    ok3 = (n3d >= min_3d) & np.isfinite(span)
    m_prop = ok3 & (span >= np.maximum(min_disp_m, k_sigma * sig))
    s_core = ok3 & ~m_prop & (st["chi2_per_dof"] < static_chi2) & (span < np.maximum(2.5 * sig * 2, 0.005))
    unknown = ~(s_core | m_prop)
    return dict(S_core=s_core, M_proposal=m_prop, U=unknown, chi2_per_dof=st["chi2_per_dof"],
                max_disp_m=st["max_disp_m"], robust_disp_m=span, n3d=n3d,
                counts=dict(S_core=int(s_core.sum()), M_proposal=int(m_prop.sum()), U=int(unknown.sum()),
                            total=int(obs.n_tracks)),
                rule=dict(moving_span_ge="max(min_disp_m, k_sigma * sigma_track)", min_disp_m=min_disp_m,
                          k_sigma=k_sigma, static="chi2/dof < static_chi2 and span < max(5 sigma, 5 mm)",
                          static_chi2_per_dof_lt=static_chi2, min_depth_anchors=min_3d,
                          span="2 * p90 |X(t) - median X(t)| over depth-anchored, non-edge observations",
                          source="candidate independent, training times only"))


def per_camera_frame_coverage(obs: TrackObservations, support: dict) -> list:
    rows = []
    for ci, cid in enumerate(obs.camera_ids):
        for t in range(obs.frame_map.T):
            m = (obs.cam == ci) & (obs.time == t) & obs.has_3d
            tr = obs.track[m]
            rows.append(dict(camera=cid, time=t, n_obs_3d=int(m.sum()),
                             n_static_core=int(support["S_core"][tr].sum()),
                             n_moving_proposal=int(support["M_proposal"][tr].sum()),
                             n_unknown=int(support["U"][tr].sum()),
                             depth_edge_fraction=float(obs.depth_edge[(obs.cam == ci) & (obs.time == t)].mean())
                             if ((obs.cam == ci) & (obs.time == t)).any() else None))
    return rows


def second_rigid_group(tab: PointTable, moving, iv, min_fraction=0.25, min_tracks=8, seed=0) -> dict:
    """RANSAC on the outliers of the main interval fit: a second coherent rigid motion with substantial support
    indicates two moving parts, a moving base or tracks on the tool."""
    a, b = iv.frame_a, iv.frame_b
    idx, P, Q, sig = tab.pair(a, b, moving)
    if len(idx) < 2 * min_tracks or iv.inlier_tracks is None:
        return dict(status="NOT_TESTABLE")
    out_mask = ~np.isin(idx, iv.inlier_tracks)
    if out_mask.sum() < max(min_tracks, min_fraction * len(idx)):
        return dict(status="OK", second_group=False, n_outliers=int(out_mask.sum()), n=len(idx))
    rng = np.random.default_rng(seed)
    thr = max(0.01, 4 * float(np.median(sig)))
    rs = ransac_rigid(P[out_mask], Q[out_mask], thr, 300, rng, min_inliers=min_tracks)
    if rs is None:
        return dict(status="OK", second_group=False, n_outliers=int(out_mask.sum()), n=len(idx))
    R2, t2, inl2 = rs
    frac = float(inl2.sum() / len(idx))
    ncl = len(np.unique(spatial_clusters(P[out_mask][inl2])))
    d_rot = float(np.degrees(np.linalg.norm(log_so3(R2 @ iv.rotation_matrix.T))))
    Xc = P[out_mask][inl2].mean(0)
    d_tr = float(np.linalg.norm((R2 @ Xc + t2) - (iv.rotation_matrix @ Xc + iv.translation_vec_m)))
    found = frac >= min_fraction and ncl >= 3 and (d_tr > 3 * thr or d_rot > 3.0)
    return dict(status="OK", second_group=bool(found), fraction=frac, n_clusters=ncl,
                rotation_diff_deg=d_rot, motion_diff_at_group_m=d_tr, n=len(idx))


def base_motion(tab: PointTable, s_core, iv) -> dict:
    """Rigid motion of the static core over the same interval (should be identity)."""
    idx, P, Q, sig = tab.pair(iv.frame_a, iv.frame_b, s_core)
    if len(idx) < 8:
        return dict(status="NOT_TESTABLE", n=len(idx))
    R, t = kabsch(P, Q)
    c = P.mean(0)
    return dict(status="OK", n=len(idx), rotation_deg=float(np.degrees(np.linalg.norm(log_so3(R)))),
                centroid_motion_m=float(np.linalg.norm(R @ c + t - c)))


def tool_proximity(tab: PointTable, moving, robot: dict | None, radius_m=0.05) -> dict:
    """Fraction of moving-proposal 3D observations within ``radius_m`` of the measured tool shaft (TCP->EE)."""
    if not robot or robot.get("p_tcp_w") is None:
        return dict(status="NO_ROBOT_DATA")
    p_tcp, p_ee = np.asarray(robot["p_tcp_w"], float), np.asarray(robot.get("p_ee_w", robot["p_tcp_w"]), float)
    near, total = 0, 0
    for t in range(min(tab.T, len(p_tcp))):
        if not np.isfinite(p_tcp[t]).all():
            continue
        X = tab.X[t, moving]
        X = X[np.isfinite(X).all(1)]
        if not len(X):
            continue
        a, b = p_tcp[t], p_ee[t]
        ab = b - a
        s = np.clip(((X - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.linalg.norm(X - (a + s[:, None] * ab), axis=1)
        near += int((d < radius_m).sum())
        total += len(X)
    return dict(status="OK", radius_m=radius_m, fraction_near_tool=float(near / max(total, 1)), n_obs=total)


def consistency_report(obs, tab: PointTable, support: dict, intervals, robot=None, thresholds=None) -> dict:
    cfg = dict(min_inlier_fraction=0.5, second_group_fraction=0.15, min_second_group_intervals=2,
               base_rotation_deg=1.0, base_motion_m=0.01, tool_fraction=0.2)
    cfg.update(thresholds or {})
    moving = support["M_proposal"]
    groups = [second_rigid_group(tab, moving, iv, cfg["second_group_fraction"]) if iv.status == "OK"
              else dict(status="NOT_TESTABLE") for iv in intervals]
    base = [base_motion(tab, support["S_core"], iv) for iv in intervals]
    inl = [iv.diagnostics.get("inlier_fraction") for iv in intervals if iv.status == "OK"]
    tool = tool_proximity(tab, moving, robot)
    n_second = sum(bool(g.get("second_group")) for g in groups)
    moving_base = [b for b in base if b.get("status") == "OK" and
                   (b["rotation_deg"] > cfg["base_rotation_deg"] or b["centroid_motion_m"] > cfg["base_motion_m"])]
    # spatial coverage of the moving proposal by interval inliers
    all_inl = np.unique(np.concatenate([iv.inlier_tracks for iv in intervals
                                        if iv.status == "OK" and iv.inlier_tracks is not None] or [np.zeros(0, int)]))
    cov = float(len(all_inl) / max(moving.sum(), 1))
    flags = []
    if inl and np.median(inl) < cfg["min_inlier_fraction"]:
        flags.append(f"median interval inlier fraction {np.median(inl):.2f} < {cfg['min_inlier_fraction']}: no single "
                     "rigid motion explains most of the moving support")
    if n_second >= cfg["min_second_group_intervals"]:
        flags.append(f"a second coherent rigid group (>= {cfg['second_group_fraction']:.0%} of moving tracks) "
                     f"in {n_second} intervals")
    if len(moving_base) >= 2:
        flags.append(f"static core moves in {len(moving_base)} intervals (base motion / calibration drift)")
    if tool.get("status") == "OK" and tool["fraction_near_tool"] > cfg["tool_fraction"]:
        flags.append(f"{tool['fraction_near_tool']:.0%} of moving observations within {tool['radius_m']} m of the tool")
    return dict(second_group_per_interval=groups, base_motion_per_interval=base, tool_proximity=tool,
                median_inlier_fraction=float(np.median(inl)) if inl else None,
                moving_proposal_inlier_coverage=cov, inconsistency_flags=flags,
                inconsistent=bool(flags), thresholds=cfg)
