"""Full-sequence exact-joint fitting with persistent support (spec 4.4 B-E, 9.3).

Parameters per hypothesis (gauge fixed): unit axis on S^2 via a 2-D local chart, revolute hinge
line position via a 2-D chart perpendicular to the axis, joint values ``q_t`` with
``q_ref = 0``, and one canonical anchor per material track. Physical cameras are constants.
The objective is the robust, whitened 2D-reprojection + depth-axis likelihood of
``visual_losses.whitened_residuals``, weighted by each track's moving responsibility.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from . import assignments as asg
from .candidates import Seed, interpolate_missing
from .observations import TrackObservations, depth_sigma
from .rigid_kinematics import (apply_joint, apply_joint_inverse, exp_so3, hinge_gauge, normalize,
                               tangent_basis)
from .visual_losses import camera_stacks, project_stacked, whitened_residuals


@dataclass
class JointCandidate:
    kind: str                      # 'revolute' | 'prismatic'
    axis_w: np.ndarray             # [3] unit
    origin_w: np.ndarray | None    # [3] hinge-line point (closest to the moving support); None for prismatic
    q: np.ndarray                  # [T] rad (R) or m (P)
    q_valid: np.ndarray            # [T] True where q_t is constrained by observations (else interpolated)
    q_uncertainty: np.ndarray | None
    reference_time_index: int
    canonical_anchors_w: np.ndarray    # [N,3] NaN where not anchored
    moving_probability: np.ndarray     # [N] NaN = unknown/ineligible
    static_probability: np.ndarray
    outlier_probability: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    @property
    def origin_convention(self) -> str:
        return ("point on hinge line closest to the moving-support centroid; location along the axis is a gauge"
                if self.kind == "revolute" else "N/A: a prismatic joint has no physical pivot")


# --------------------------------------------------------------------------- problem state
class Problem:
    """Candidate-independent quantities shared by every hypothesis (same observations, same noise)."""

    def __init__(self, obs: TrackObservations, calibs: dict, cfg, ref: int, use_depth=True):
        self.obs, self.calibs, self.cfg, self.ref = obs, calibs, cfg, ref
        self.noise = cfg.noise
        self.use_depth = use_depth
        self.K, self.T_CW = camera_stacks(obs, calibs)
        n2d, n3d = obs.per_track_counts()
        self.eligible = (n3d >= 1) & (n2d >= cfg.fit.min_track_observations)
        self.ineligible_reason = np.where(n3d < 1, "no_depth_anchor",
                                          np.where(n2d < cfg.fit.min_track_observations, "too_few_observations", ""))
        self.sigma_z = depth_sigma(np.nan_to_num(obs.z, nan=1.0), self.noise, obs.depth_edge, obs.depth_cos)
        self.ll_outlier = asg.track_sum(obs, asg.outlier_loglik(obs, self.noise, use_depth))
        X0 = self.first_3d_points()
        self.static_anchors = refine_anchors("static", None, None, np.zeros(obs.frame_map.T), X0, obs, self.K,
                                             self.T_CW, self.noise, self.eligible, use_depth=use_depth)
        self.ll_static = self.track_ll("static", None, None, np.zeros(obs.frame_map.T), self.static_anchors)

    def first_3d_points(self):
        obs = self.obs
        X = np.full((obs.n_tracks, 3), np.nan)
        h = np.flatnonzero(obs.has_3d)
        order = h[np.lexsort((obs.time[h], obs.track[h]))]
        first = np.unique(obs.track[order], return_index=True)
        X[first[0]] = obs.x_w[order[first[1]]]
        tfirst = np.full(obs.n_tracks, -1)
        tfirst[first[0]] = obs.time[order[first[1]]]
        self.first_time = tfirst
        return X

    def track_ll(self, kind, axis, origin, q, anchors):
        ok = self.eligible[self.obs.track]
        res = whitened_residuals(kind, axis, origin, q, np.nan_to_num(anchors), self.obs, self.K, self.T_CW, self.noise,
                                 obs_mask=ok, use_depth=self.use_depth)
        ll = asg.observation_loglik(res, self.noise.sigma_px, self.sigma_z, self.noise)
        return asg.track_sum(self.obs, ll, ok)


# --------------------------------------------------------------------------- anchors (per-track GN)
def _joint_rotation(kind, axis, q_obs):
    if kind in ("static", "prismatic"):
        return np.broadcast_to(np.eye(3), (len(q_obs), 3, 3))
    return exp_so3(normalize(axis)[None] * q_obs[:, None])


def refine_anchors(kind, axis, origin, q, X0, obs: TrackObservations, K, T_CW, noise, track_mask, iters=6,
                   use_depth=True):
    """Robust Gauss-Newton (Student-t IRLS) for each track's canonical anchor, joint fixed."""
    X = np.array(X0, float)
    sel = track_mask[obs.track] & obs.uv_valid & np.isfinite(X[obs.track]).all(1)
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        return X
    tr, t, cam = obs.track[idx], obs.time[idx], obs.cam[idx]
    RA = _joint_rotation(kind, axis, q[t])
    Rcw = T_CW[cam, :3, :3]
    M = Rcw @ RA
    dz_ok = (obs.depth_valid[idx] & obs.mask_valid[idx] & np.isfinite(obs.z[idx])) if use_depth else np.zeros(len(idx), bool)
    sz = depth_sigma(np.nan_to_num(obs.z[idx], nan=1.0), noise, obs.depth_edge[idx], obs.depth_cos[idx])
    dof = noise.student_t_dof
    N = len(X)
    for _ in range(iters):
        xw = (X[tr] if kind == "static" else apply_joint(kind, axis, origin, q[t], X[tr]))
        uv, z, xc = project_stacked(xw, cam, K, T_CW)
        zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
        fx, fy = K[cam, 0, 0], K[cam, 1, 1]
        D = np.zeros((len(idx), 2, 3))
        D[:, 0, 0], D[:, 0, 2] = fx / zz, -fx * xc[:, 0] / zz ** 2
        D[:, 1, 1], D[:, 1, 2] = fy / zz, -fy * xc[:, 1] / zz ** 2
        Juv = D @ M / noise.sigma_px
        ruv = (uv - obs.uv[idx]) / noise.sigma_px
        wuv = (dof + 1) / (dof + np.sum(ruv ** 2, 1) / 2)
        H = np.einsum("mki,mkj->mij", Juv, Juv) * wuv[:, None, None]
        g = np.einsum("mki,mk->mi", Juv, ruv) * wuv[:, None]
        if dz_ok.any():
            Jz = M[:, 2, :] / sz[:, None]
            rz = np.where(dz_ok, (z - np.nan_to_num(obs.z[idx])) / sz, 0.0)
            wz = np.where(dz_ok, (dof + 1) / (dof + rz ** 2), 0.0)
            H += np.einsum("mi,mj->mij", Jz, Jz) * wz[:, None, None]
            g += Jz * (rz * wz)[:, None]
        Ht = np.zeros((N, 3, 3))
        gt = np.zeros((N, 3))
        np.add.at(Ht, tr, H)
        np.add.at(gt, tr, g)
        active = np.unique(tr)
        Ha = Ht[active] + np.eye(3)[None] * (1e-6 + 1e-9 * np.trace(Ht[active], axis1=1, axis2=2))[:, None, None]
        step = np.linalg.solve(Ha, gt[active][..., None])[..., 0]
        step = np.clip(step, -0.05, 0.05)
        X[active] -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return X


def initial_moving_anchors(problem: Problem, kind, axis, origin, q):
    X = problem.first_3d_points()
    ok = np.isfinite(X).all(1)
    tf = problem.first_time
    X[ok] = apply_joint_inverse(kind, axis, origin, q[tf[ok]], X[ok])
    return X


# --------------------------------------------------------------------------- global least squares
class _Layout:
    def __init__(self, kind, axis0, origin0, q, free_q, fit_tracks, center):
        self.kind = kind
        self.axis0 = normalize(axis0)
        self.e1, self.e2 = tangent_basis(self.axis0)
        self.center = center
        self.origin0 = hinge_gauge(self.axis0, origin0, center) if kind == "revolute" else None
        self.q0 = q.copy()
        self.free_q = np.flatnonzero(free_q)
        self.fit_tracks = fit_tracks
        self.n_joint = 4 if kind == "revolute" else 2
        self.n_q = len(self.free_q)
        self.n_global = self.n_joint + self.n_q
        self.track_col = -np.ones(int(fit_tracks.max()) + 1 if len(fit_tracks) else 1, int)
        self.track_col[fit_tracks] = self.n_global + 3 * np.arange(len(fit_tracks))

    def pack(self, X):
        return np.concatenate([np.zeros(self.n_joint), self.q0[self.free_q], X[self.fit_tracks].ravel()])

    def unpack(self, p):
        a = normalize(self.axis0 + p[0] * self.e1 + p[1] * self.e2)
        o = None
        if self.kind == "revolute":
            o = self.origin0 + p[2] * self.e1 + p[3] * self.e2
        q = self.q0.copy()
        q[self.free_q] = p[self.n_joint:self.n_global]
        X = p[self.n_global:].reshape(-1, 3)
        return a, o, q, X


def fit_geometry(kind, axis0, origin0, q0, anchors, fit_tracks, weights, problem: Problem, max_nfev=None, extra=None):
    """Robust sparse least squares over (axis, hinge, q, anchors of ``fit_tracks``).

    ``extra`` (Stage E, optional): object with ``residuals(kind, axis, origin, q) -> [k]`` already
    whitened, and ``row_times`` [k] (model time each row depends on through q; -1 = joint params
    only). Rows are appended to the vision residuals; their Jacobian pattern covers the joint
    parameters and the q column of their time. Vision rows are unchanged."""
    obs, cfg, noise = problem.obs, problem.cfg, problem.noise
    T = obs.frame_map.T
    in_fit = np.zeros(obs.n_tracks, bool)
    in_fit[fit_tracks] = True
    sel = np.flatnonzero(in_fit[obs.track] & obs.uv_valid)
    observed_t = np.zeros(T, bool)
    observed_t[np.unique(obs.time[sel])] = True
    free_q = observed_t.copy()
    free_q[problem.ref] = False
    q_init = q0.copy()
    q_init[problem.ref] = 0.0
    center = np.nanmedian(anchors[fit_tracks], 0)
    L = _Layout(kind, axis0, origin0 if origin0 is not None else center, q_init, free_q, fit_tracks, center)
    tr, t, cam = obs.track[sel], obs.time[sel], obs.cam[sel]
    dz = (obs.depth_valid[sel] & obs.mask_valid[sel] & np.isfinite(obs.z[sel])) if problem.use_depth else np.zeros(len(sel), bool)
    sz = problem.sigma_z[sel]
    sw = np.sqrt(np.clip(weights[tr], 0, 1))
    uv_obs = obs.uv[sel]
    z_obs = np.nan_to_num(obs.z[sel])
    ts = obs.frame_map.timestamp_s
    use_prior = cfg.fit.enable_temporal_prior and T >= 3
    n_prior = T - 2 if use_prior else 0

    def fun(p):
        a, o, q, X = L.unpack(p)
        Xi = X[(L.track_col[tr] - L.n_global) // 3]
        xw = apply_joint(kind, a, o, q[t], Xi)
        uv, z, _ = project_stacked(xw, cam, problem.K, problem.T_CW)
        r_uv = ((uv - uv_obs) / noise.sigma_px) * sw[:, None]
        r_z = (((z - z_obs) / sz) * sw)[dz]
        parts = [r_uv.ravel(), r_z]
        if use_prior:
            v = np.diff(q) / np.diff(ts)
            acc = np.diff(v) / ((ts[2:] - ts[:-2]) / 2)
            parts.append(acc / cfg.fit.temporal_sigma_qacc)
        if extra is not None:
            parts.append(np.nan_to_num(np.asarray(extra.residuals(kind, a, o, q), float)))
        return np.concatenate(parts)

    # sparsity pattern
    n_uv, n_z = 2 * len(sel), int(dz.sum())
    rows, cols = [], []
    obs_rows_uv = np.arange(n_uv).reshape(-1, 2)
    obs_rows_z = n_uv + np.arange(n_z)
    qcol = -np.ones(T, int)
    qcol[L.free_q] = L.n_joint + np.arange(L.n_q)

    def add(rr, cc):
        rows.append(np.asarray(rr).ravel())
        cols.append(np.asarray(cc).ravel())

    tcol = L.track_col[tr]
    for rr, mask in ((obs_rows_uv[:, 0], None), (obs_rows_uv[:, 1], None), (obs_rows_z, dz)):
        sub = slice(None) if mask is None else mask
        r_ = rr
        for j in range(L.n_joint):
            add(r_, np.full(len(r_), j))
        qc = qcol[t[sub]]
        m = qc >= 0
        add(r_[m], qc[m])
        for d in range(3):
            add(r_, tcol[sub] + d)
    if use_prior:
        pr = n_uv + n_z + np.arange(n_prior)
        for k in range(3):
            qc = qcol[np.arange(n_prior) + k]
            m = qc >= 0
            add(pr[m], qc[m])
    n_extra = 0
    if extra is not None:
        rt = np.asarray(extra.row_times, int)
        n_extra = len(rt)
        er = n_uv + n_z + n_prior + np.arange(n_extra)
        for j in range(L.n_joint):
            add(er, np.full(n_extra, j))
        qc = np.where(rt >= 0, qcol[np.clip(rt, 0, T - 1)], -1)
        m = qc >= 0
        add(er[m], qc[m])
        # finite-difference velocities use neighbouring q values too
        for sh in (-1, 1):
            qn = np.where(rt >= 0, qcol[np.clip(rt + sh, 0, T - 1)], -1)
            m = qn >= 0
            add(er[m], qn[m])
    n_rows = n_uv + n_z + n_prior + n_extra
    n_cols = L.n_global + 3 * len(fit_tracks)
    S = sparse.coo_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n_rows, n_cols)).tocsr()
    S.data[:] = 1
    p0 = L.pack(anchors)
    t0 = time.time()
    res = least_squares(fun, p0, jac_sparsity=S, method="trf", loss=cfg.fit.robust_loss,
                        f_scale=cfg.fit.robust_f_scale, max_nfev=max_nfev or cfg.fit.least_squares_max_nfev,
                        x_scale="jac", tr_solver="lsmr", verbose=0)
    a, o, q, X = L.unpack(res.x)
    if kind == "revolute":
        o = hinge_gauge(a, o, np.nanmedian(X, 0))
    anchors_out = anchors.copy()
    anchors_out[fit_tracks] = X
    cov = marginal_covariance(res.jac, L)
    q_unc = np.full(T, np.nan)
    q_unc[problem.ref] = 0.0
    if cov.get("cov") is not None:
        q_unc[L.free_q] = np.sqrt(np.clip(np.diag(cov["cov"])[L.n_joint:L.n_global], 0, None))
    q = interpolate_missing(q, observed_t | (np.arange(T) == problem.ref), ts)
    info = dict(cost=float(res.cost), nfev=int(res.nfev), status=int(res.status), message=str(res.message),
                n_residuals=int(n_rows), n_params=int(n_cols), n_fit_tracks=int(len(fit_tracks)), n_extra_rows=int(n_extra),
                runtime_s=round(time.time() - t0, 3), observed_times=observed_t.tolist())
    info.update({k: v for k, v in cov.items() if k != "cov"})
    return a, o, q, anchors_out, q_unc, observed_t, info


def marginal_covariance(J, L: _Layout):
    """Gauss-Newton covariance of (axis chart, hinge chart, q) after eliminating anchors (Schur)."""
    try:
        J = sparse.csr_matrix(J)
        g = L.n_global
        Jg, Jx = J[:, :g], J[:, g:]
        A = (Jg.T @ Jg).toarray()
        B = (Jg.T @ Jx).toarray()
        D = (Jx.T @ Jx).tocsr()
        n = Jx.shape[1] // 3
        Dinv_blocks = np.zeros((n, 3, 3))
        Dd = D.toarray() if n <= 400 else None
        for i in range(n):
            blk = Dd[3 * i:3 * i + 3, 3 * i:3 * i + 3] if Dd is not None else D[3 * i:3 * i + 3, 3 * i:3 * i + 3].toarray()
            Dinv_blocks[i] = np.linalg.pinv(blk + np.eye(3) * 1e-12)
        Bb = B.reshape(g, n, 3)
        BDinv = np.einsum("gni,nij->gnj", Bb, Dinv_blocks).reshape(g, -1)
        Sm = A - BDinv @ B.T
        sv = np.linalg.svd(Sm, compute_uv=False)
        cov = np.linalg.pinv(Sm, rcond=1e-12)
        out = dict(cov=cov, schur_singular_values=sv.tolist(),
                   schur_condition=float(sv[0] / max(sv[-1], 1e-300)),
                   axis_sigma_deg=float(np.degrees(np.sqrt(max(cov[0, 0] + cov[1, 1], 0)))))
        if L.kind == "revolute":
            out["hinge_line_sigma_m"] = float(np.sqrt(max(cov[2, 2] + cov[3, 3], 0)))
        return out
    except Exception as exc:  # covariance is a diagnostic; never abort a fit for it
        return dict(cov=None, covariance_error=repr(exc))


# --------------------------------------------------------------------------- EM driver
def _select_fit_tracks(problem: Problem, post_m, rng):
    cfg = problem.cfg.fit
    obs = problem.obs
    cand = np.flatnonzero(problem.eligible & (np.nan_to_num(post_m) >= cfg.responsibility_floor))
    chosen = []
    for ci in range(len(obs.camera_ids)):
        c = cand[obs.track_camera[cand] == ci]
        if len(c) > cfg.max_tracks_per_camera:
            p = post_m[c] / post_m[c].sum()
            c = np.sort(rng.choice(c, cfg.max_tracks_per_camera, replace=False, p=p))
        chosen.append(c)
    return np.concatenate(chosen) if chosen else np.zeros(0, int)


def fit_hypothesis(problem: Problem, seed: Seed, em_iterations=None, max_nfev=None, log=None) -> tuple:
    """EM-like alternation for one seed; returns (JointCandidate, training mixture NLL)."""
    cfg = problem.cfg
    obs = problem.obs
    rng = np.random.default_rng(cfg.fit.seed)
    kind = seed.kind
    a, o, q = normalize(seed.axis), seed.origin, seed.q.copy()
    q[problem.ref] = 0.0
    X = initial_moving_anchors(problem, kind, a, o, q)
    priors = np.array([1 - cfg.fit.prior_moving - cfg.fit.prior_outlier, cfg.fit.prior_moving, cfg.fit.prior_outlier])
    history = []
    q_unc, observed_t, info = None, np.ones(len(q), bool), {}
    initial_moving = None
    for it in range(em_iterations or cfg.fit.em_iterations):
        X = refine_anchors(kind, a, o, q, X, obs, problem.K, problem.T_CW, problem.noise, problem.eligible,
                           use_depth=problem.use_depth)
        ll_m = problem.track_ll(kind, a, o, q, X)
        post = asg.responsibilities(problem.ll_static, ll_m, problem.ll_outlier, priors, problem.eligible)
        priors = asg.update_priors(post, problem.eligible)
        if initial_moving is None:
            initial_moving = post[:, 1] > 0.5
        fit_tracks = _select_fit_tracks(problem, post[:, 1], rng)
        if len(fit_tracks) < cfg.fit.min_moving_tracks:
            history.append(dict(iteration=it, status="insufficient_moving_support", n_fit=int(len(fit_tracks))))
            break
        a, o, q, X, q_unc, observed_t, info = fit_geometry(kind, a, o, q, X, fit_tracks, post[:, 1], problem,
                                                           max_nfev=max_nfev)
        nll = asg.mixture_nll(problem.ll_static, ll_m, problem.ll_outlier, priors, problem.eligible).sum()
        history.append(dict(iteration=it, n_fit=int(len(fit_tracks)), priors=priors.tolist(), nll_before_m=float(nll),
                            ls_cost=info["cost"], nfev=info["nfev"]))
        if log:
            log(f"    [{kind}] EM {it}: fit {len(fit_tracks)} tracks, cost {info['cost']:.1f}, priors {np.round(priors, 3)}")
    X = refine_anchors(kind, a, o, q, X, obs, problem.K, problem.T_CW, problem.noise, problem.eligible,
                       use_depth=problem.use_depth)
    ll_m = problem.track_ll(kind, a, o, q, X)
    post = asg.responsibilities(problem.ll_static, ll_m, problem.ll_outlier, priors, problem.eligible)
    nll = float(asg.mixture_nll(problem.ll_static, ll_m, problem.ll_outlier, priors, problem.eligible).sum())
    final_moving = np.nan_to_num(post[:, 1]) > 0.5
    X_out = np.where(problem.eligible[:, None], X, np.nan)
    cand = JointCandidate(kind, a, o, q, observed_t, q_unc, problem.ref, X_out, post[:, 1], post[:, 0], post[:, 2],
                          diagnostics=dict(seed=seed.diagnostics, em_history=history, final_ls=info,
                                           priors=priors.tolist(), training_nll=nll,
                                           n_moving=int(final_moving.sum()),
                                           n_moving_gained_vs_first_estep=int((final_moving & ~initial_moving).sum()),
                                           n_moving_lost_vs_first_estep=int((~final_moving & initial_moving).sum())))
    cand._ll_moving = ll_m  # cached for model selection (not serialized)
    return cand, nll
