"""Frozen-joint q(t) extension: the ACCEPTED joint (type, axis, hinge point, gauge) is kept exactly; only the joint
coordinate q is measured on additional RGB-D frames (e.g. the whole recording instead of the reconstruction clip).

Inputs: the existing JointAuthority (clip) and a TrackObservations table extracted (AllTracker + depth) over a frame
range that CONTAINS the clip frames. Nothing is re-estimated about the joint; no 3DGS is involved.

1. Moving-part tracks are selected on the OVERLAP frames: a track is moving when its world points pulled back through
   the frozen joint, A(q_auth(t))^-1 x_w(t), collapse onto one canonical point (spread <= tau) while the raw points
   themselves move (spread_moving < ratio x spread_static). Static / ambiguous tracks (e.g. near the hinge) are unused.
2. Each moving track gets a canonical point X_c (median of the pulled-back points).
3. For every frame, q is solved with the frozen axis / hinge from all moving tracks seen in 3D there: a weighted median
   of the per-point angles (revolute) / axial displacements (prismatic), refined by a bounded Huber(tau) least squares
   on the 3D residuals || A(q) X_c - x_w ||. sigma_q = 1.4826 weighted MAD of the per-point values / sqrt(n_eff),
   then inflated by the factor calibrated on the overlap (declared, reported).
4. Validation on the overlap: the new q must reproduce the existing q (rms difference <= max(k x median sigma_auth,
   rel_tol x q range)); otherwise INCONSISTENT_WITH_AUTHORITY and nothing is written.
5. The extended authority keeps the parent's q / sigma UNCHANGED on the parent's frames and adds the measured frames
   outside them (same gauge: q(ref) = 0 at the parent's reference frame). It is a NEW versioned authority with
   parent_authority_id; the parent is never modified.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .integration_contract import AuthorityError, JointAuthority
from .rigid_kinematics import A_joint

SCHEMA = "estimotion.joint_discovery.frozen_joint_q_extension/v1"


def _wmedian(x, w):
    o = np.argsort(x)
    x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def _A(auth, q):
    o = auth.origin_world if auth.joint_type == "revolute" else np.zeros(3)
    return np.asarray(A_joint(auth.joint_type, auth.axis_world, o, np.atleast_1d(np.asarray(q, float))), float)


def _pull_back(auth, q, x):
    """A(q)^-1 x for points x [n,3] with per-point q [n]."""
    A = _A(auth, q)
    R, t = A[:, :3, :3], A[:, :3, 3]
    return np.einsum("nji,nj->ni", R, x - t)


def _push(auth, q, X):
    A = _A(auth, np.full(len(X), q))
    return np.einsum("nij,nj->ni", A[:, :3, :3], X) + A[:, :3, 3]


def _per_point_q(auth, Xc, x):
    """Per-point joint coordinate with the frozen axis (and hinge) + its weight (lever^2 for revolute)."""
    u = np.asarray(auth.axis_world, float)
    if auth.joint_type == "prismatic":
        return (x - Xc) @ u, np.ones(len(x))
    o = np.asarray(auth.origin_world, float)
    p, r = Xc - o, x - o
    p = p - np.outer(p @ u, u)
    r = r - np.outer(r @ u, u)
    ang = np.arctan2(np.cross(p, r) @ u, np.einsum("ij,ij->i", p, r))
    lever = np.linalg.norm(p, axis=1)
    return ang, lever ** 2


def _solve(auth, Xc, x, tau):
    qi, w = _per_point_q(auth, Xc, x)
    q0 = _wmedian(qi, w)
    span = 0.2 if auth.joint_type == "revolute" else 0.05

    def cost(q):
        d = np.linalg.norm(_push(auth, q, Xc) - x, axis=1) / tau
        return float(np.sum(np.where(d <= 1.0, 0.5 * d * d, d - 0.5)))
    r = minimize_scalar(cost, bounds=(q0 - span, q0 + span), method="bounded", options=dict(xatol=1e-7))
    q = float(r.x) if r.fun <= cost(q0) else q0
    dev = np.abs(qi - _wmedian(qi, w))
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
    sd = 1.4826 * _wmedian(dev, w) / math.sqrt(max(n_eff, 1.0))
    return q, sd, len(x), n_eff


def extend_authority(auth: JointAuthority, obs, *, tau_m=0.006, moving_ratio=0.3, min_overlap_obs=3,
                     min_points=8, k_sigma=3.0, rel_tol=0.02, sources=None) -> dict:
    """Returns dict(status, report, authority | None). ``obs``: TrackObservations over a frame range containing the
    parent's frames (same scene / cameras / world frame)."""
    if auth.joint_type not in ("revolute", "prismatic"):
        raise AuthorityError("only revolute / prismatic authorities can be extended")
    fm = obs.frame_map
    src_full = np.asarray(fm.source_frame_index, int)
    src_auth = np.asarray(auth.source_frame_index, int)
    pos_auth = {int(s): i for i, s in enumerate(src_auth)}
    auth_idx_of_t = np.array([pos_auth.get(int(s), -1) for s in src_full])
    qv = np.asarray(auth.q_valid, bool)
    overlap_t = np.flatnonzero((auth_idx_of_t >= 0) & qv[np.clip(auth_idx_of_t, 0, None)])
    missing = sorted(set(src_auth[qv].tolist()) - set(src_full.tolist()))
    rep = dict(schema=SCHEMA, parent_authority_id=auth.authority_id(), joint_type=auth.joint_type,
               axis_world=np.asarray(auth.axis_world).tolist(),
               origin_world=None if auth.origin_world is None else np.asarray(auth.origin_world).tolist(),
               frames=dict(extended=[int(src_full[0]), int(src_full[-1]), int(len(src_full))],
                           parent=[int(src_auth[0]), int(src_auth[-1]), int(len(src_auth))],
                           overlap=int(len(overlap_t)), parent_frames_missing_from_extension=missing[:20]),
               settings=dict(tau_m=tau_m, moving_ratio=moving_ratio, min_overlap_obs=min_overlap_obs,
                             min_points=min_points, k_sigma=k_sigma, rel_tol=rel_tol),
               frozen="joint type, axis, hinge point, reference frame and the parent's q / sigma are NOT changed")
    if len(overlap_t) < 3:
        return dict(status="NO_OVERLAP", report=dict(rep, reason="fewer than 3 parent frames in the extension range"),
                    authority=None)
    m3 = obs.has_3d
    trk, tim, X = obs.track[m3], obs.time[m3], obs.x_w[m3]
    # ---- 1. moving tracks on the overlap
    ov = np.isin(tim, overlap_t)
    q_ov = np.asarray(auth.q, float)[auth_idx_of_t[np.where(ov, tim, 0)]]
    sel = []
    sm_all, ss_all = [], []
    canon = {}
    iov = np.flatnonzero(ov)                       # group the overlap observations by track ONCE (stable: each track
    order = iov[np.argsort(trk[iov], kind="stable")]  # keeps its observations in their original order)
    utr, starts, counts = np.unique(trk[order], return_index=True, return_counts=True)
    for tr, s0, cnt in zip(utr, starts, counts):
        if cnt < min_overlap_obs:
            continue
        mk = order[s0:s0 + cnt]
        x = X[mk]
        Xb = _pull_back(auth, q_ov[mk], x)
        cm = np.median(Xb, axis=0)
        sm = float(np.median(np.linalg.norm(Xb - cm, axis=1)))
        ss = float(np.median(np.linalg.norm(x - np.median(x, axis=0), axis=1)))
        sm_all.append(sm)
        ss_all.append(ss)
        if sm <= tau_m and sm < moving_ratio * ss:
            sel.append(int(tr))
            canon[int(tr)] = cm
    rep["tracks"] = dict(with_overlap=len(sm_all), moving_selected=len(sel),
                         spread_moving_median_m=float(np.median(sm_all)) if sm_all else None,
                         spread_static_median_m=float(np.median(ss_all)) if ss_all else None,
                         rule=f"moving: median |A^-1 x - X_c| <= {tau_m} m and < {moving_ratio} x raw spread")
    if len(sel) < min_points:
        return dict(status="TOO_FEW_MOVING_TRACKS", report=dict(rep, reason=f"{len(sel)} moving tracks < {min_points}"),
                    authority=None)
    # ---- 2-3. per-frame solve with the frozen joint
    selset = np.isin(trk, sel)
    T = len(src_full)
    q_ext, s_ext, n_pts = np.full(T, np.nan), np.full(T, np.nan), np.zeros(T, int)
    for t in range(T):
        mk = selset & (tim == t)
        if mk.sum() < min_points:
            n_pts[t] = int(mk.sum())
            continue
        Xc = np.stack([canon[int(i)] for i in trk[mk]])
        q, sd, n, _ = _solve(auth, Xc, X[mk], tau_m)
        q_ext[t], s_ext[t], n_pts[t] = q, sd, n
    # ---- 4. validation + sigma calibration on the overlap
    ok_ov = overlap_t[np.isfinite(q_ext[overlap_t])]
    if len(ok_ov) < 3:
        return dict(status="NO_OVERLAP", report=dict(rep, reason="the extension resolves q on fewer than 3 overlap frames"),
                    authority=None)
    ia = auth_idx_of_t[ok_ov]
    qa = np.asarray(auth.q, float)[ia]
    sa = (np.asarray(auth.q_sigma, float)[ia] if auth.q_sigma is not None else np.zeros(len(ia)))
    diff = q_ext[ok_ov] - qa
    offset = float(np.median(diff))
    d0 = diff - offset
    z_raw = d0 / np.maximum(np.hypot(s_ext[ok_ov], sa), 1e-12)
    infl = max(1.0, float(np.sqrt(np.mean(z_raw ** 2))))
    rms = float(np.sqrt(np.mean(d0 ** 2)))
    rng = float(np.ptp(np.asarray(auth.q, float)[qv]))
    tol = max(k_sigma * float(np.median(sa)) if np.any(sa > 0) else 0.0, rel_tol * rng)
    val = dict(n_overlap=int(len(ok_ov)), gauge_offset=offset, rms_difference=rms, max_abs_difference=float(np.max(np.abs(d0))),
               parent_q_range=rng, tolerance=tol, sigma_inflation_from_overlap=infl,
               z_raw_rms=float(np.sqrt(np.mean(z_raw ** 2))),
               rule=f"rms(q_ext - offset - q_parent) <= max({k_sigma} x median sigma_parent, {rel_tol} x parent q range)",
               per_frame=[dict(source_frame=int(src_full[t]), q_parent=float(a), q_extension=float(q_ext[t] - offset),
                               sigma_extension=float(s_ext[t] * infl), n_points=int(n_pts[t]))
                          for t, a in zip(ok_ov, qa)])
    rep["validation"] = val
    if rms > tol:
        return dict(status="INCONSISTENT_WITH_AUTHORITY", report=dict(rep, reason=f"rms difference {rms:.4g} > {tol:.4g} "
                    "on the overlap: the extension does not reproduce the frozen joint's q"), authority=None)
    # ---- 5. extended authority: parent values on parent frames, measured values elsewhere
    keep = np.isfinite(q_ext) | (auth_idx_of_t >= 0)
    q_new = np.where(np.isfinite(q_ext), q_ext - offset, 0.0)
    s_new = np.where(np.isfinite(s_ext), s_ext * infl, np.nan)
    valid = np.isfinite(q_ext)
    source = np.full(T, "extension", dtype="U16")
    par = auth_idx_of_t >= 0
    q_new[par] = np.asarray(auth.q, float)[auth_idx_of_t[par]]
    valid[par] = qv[auth_idx_of_t[par]]
    if auth.q_sigma is not None:
        s_new[par] = np.asarray(auth.q_sigma, float)[auth_idx_of_t[par]]
    source[par] = "parent"
    ref_src = int(src_auth[auth.reference_time_index])
    sel_t = np.flatnonzero(keep)
    ref_new = int(np.flatnonzero(src_full[sel_t] == ref_src)[0])
    s_fill = float(np.nanmedian(s_new[sel_t])) if np.isfinite(s_new[sel_t]).any() else 1e-3
    q_sigma = np.where(np.isfinite(s_new[sel_t]), s_new[sel_t], s_fill)
    new = JointAuthority(
        joint_type=auth.joint_type, axis_world=auth.axis_world, origin_world=auth.origin_world, q=q_new[sel_t],
        q_valid=valid[sel_t], reference_time_index=ref_new,
        rgbd_frame_index=np.asarray(fm.rgbd_frame_index, int)[sel_t], source_frame_index=src_full[sel_t],
        timestamp_s=np.asarray(fm.timestamp_s, float)[sel_t], cameras=auth.cameras,
        decision_status=auth.decision_status, q_sigma=q_sigma, fit_covariance_status=auth.fit_covariance_status,
        support_centroid_world=auth.support_centroid_world, source_hashes=dict(auth.source_hashes),
        derived_from=dict(kind="FROZEN_JOINT_Q_EXTENSION", parent_authority_id=rep["parent_authority_id"],
                          observations=sources,
                          n_parent_frames=int(par.sum()), n_extension_frames=int((keep & ~par).sum()),
                          sigma_inflation_from_overlap=infl, gauge_offset=offset),
        parent_authority_id=rep["parent_authority_id"], version=int(auth.version) + 1, world_frame=auth.world_frame)
    new.validate()
    rep["result"] = dict(n_frames=int(len(sel_t)), n_valid=int(new.q_valid.sum()),
                         q_source=dict(zip(map(int, src_full[sel_t]), source[sel_t].tolist())),
                         n_points_per_frame=dict(zip(map(int, src_full[sel_t]), n_pts[sel_t].tolist())))
    return dict(status="EXTENDED", report=rep, authority=new)


def write_extension(result, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = dict(result["report"], status=result["status"])
    if result["authority"] is not None:
        rep["authority_id"] = result["authority"].write(out)
    (out / "extension_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
