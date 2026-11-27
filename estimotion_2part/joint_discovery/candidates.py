"""Type-agnostic initialization: static evidence, finite rigid-motion proposals, R/P seeds.

Nothing here decides the joint type. Both the prismatic and the revolute seed are derived from
the same robust finite-motion proposals (spec 4.3); ``rigid_flow_omega`` is the unconstrained
instantaneous diagnostic of spec 3.3 and is never used as the final articulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .observations import TrackObservations, depth_sigma
from .rigid_kinematics import log_so3, normalize, tangent_basis


# --------------------------------------------------------------------------- rigid fits
def kabsch(P, Q, w=None):
    """Weighted least-squares ``Q ~ R P + t`` (proper rotation)."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    mp, mq = w @ P, w @ Q
    H = ((P - mp) * w[:, None]).T @ (Q - mq)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def ransac_rigid(P, Q, threshold, iterations, rng, min_inliers=6):
    n = len(P)
    if n < max(3, min_inliers):
        return None
    best, best_count = None, -1
    for _ in range(iterations):
        idx = rng.choice(n, 3, replace=False)
        if np.linalg.matrix_rank(P[idx] - P[idx].mean(0), tol=1e-4) < 2:
            continue
        R, t = kabsch(P[idx], Q[idx])
        inl = np.linalg.norm(P @ R.T + t - Q, axis=1) < threshold
        c = int(inl.sum())
        if c > best_count:
            best, best_count = inl, c
    if best is None or best_count < min_inliers:
        return None
    for _ in range(2):  # refit on inliers
        R, t = kabsch(P[best], Q[best])
        best = np.linalg.norm(P @ R.T + t - Q, axis=1) < threshold
        if best.sum() < 3:
            return None
    return R, t, best


def rigid_flow_omega(points, velocities, weights=None, max_pairs=20000, rng=None):
    """Instantaneous ``v_i - v_j = omega x (X_i - X_j)`` least squares (translation eliminated).

    Returns dict(omega, a, singular_values, condition, rank). ``a`` follows from
    ``v_i = a + omega x X_i``. Diagnostic only."""
    X, V = np.asarray(points, float), np.asarray(velocities, float)
    n = len(X)
    rng = np.random.default_rng(0) if rng is None else rng
    if n < 3:
        return dict(omega=np.full(3, np.nan), a=np.full(3, np.nan), singular_values=np.zeros(3), condition=np.inf, rank=0)
    i = rng.integers(0, n, min(max_pairs, n * (n - 1) // 2))
    j = rng.integers(0, n, len(i))
    keep = i != j
    i, j = i[keep], j[keep]
    dX = X[i] - X[j]
    dV = V[i] - V[j]
    # omega x dX = -[dX]x omega
    A = np.zeros((len(i), 3, 3))
    A[:, 0, 1], A[:, 0, 2] = dX[:, 2], -dX[:, 1]
    A[:, 1, 0], A[:, 1, 2] = -dX[:, 2], dX[:, 0]
    A[:, 2, 0], A[:, 2, 1] = dX[:, 1], -dX[:, 0]
    w = np.ones(len(i)) if weights is None else np.sqrt(np.asarray(weights)[i] * np.asarray(weights)[j])
    A = (A * w[:, None, None]).reshape(-1, 3)
    b = (dV * w[:, None]).reshape(-1)
    omega, *_ = np.linalg.lstsq(A, b, rcond=None)
    s = np.linalg.svd(A, compute_uv=False)
    a = np.median(V - np.cross(omega, X), axis=0)
    return dict(omega=omega, a=a, singular_values=s, condition=float(s[0] / max(s[-1], 1e-30)),
                rank=int(np.sum(s > s[0] * 1e-6)))


# --------------------------------------------------------------------------- static evidence
def track_static_statistics(obs: TrackObservations, noise, calibs) -> dict:
    """Per-track whitened spread around a constant camera-fixed position (candidate independent)."""
    N = obs.n_tracks
    chi2 = np.zeros(N)
    dof = np.zeros(N)
    disp_m = np.zeros(N)
    # 2D: pixel deviation from the per-track median pixel
    for arr_name in ("uv",):
        valid = obs.uv_valid
        tr = obs.track[valid]
        uv = obs.uv[valid]
        med = np.zeros((N, 2))
        for d in range(2):
            med[:, d] = _group_median(tr, uv[:, d], N)
        dev = (uv - med[tr]) / noise.sigma_px
        chi2 += np.bincount(tr, np.sum(dev ** 2, 1), minlength=N)
        dof += 2 * np.bincount(tr, minlength=N)
    h = obs.has_3d
    tr = obs.track[h]
    if len(tr):
        x = obs.x_w[h]
        med = np.stack([_group_median(tr, x[:, d], N) for d in range(3)], 1)
        dist = np.linalg.norm(x - med[tr], axis=1)
        disp_m = np.maximum(disp_m, _group_max(tr, dist, N))
        zs = (obs.z[h] - _group_median(tr, obs.z[h], N)[tr]) / depth_sigma(obs.z[h], noise, obs.depth_edge[h], obs.depth_cos[h])
        chi2 += np.bincount(tr, zs ** 2, minlength=N)
        dof += np.bincount(tr, minlength=N)
    return dict(chi2=chi2, dof=dof, chi2_per_dof=chi2 / np.maximum(dof, 1), max_disp_m=disp_m)


def _group_median(groups, values, n):
    out = np.full(n, np.nan)
    order = np.argsort(groups, kind="stable")
    g, v = groups[order], values[order]
    bounds = np.flatnonzero(np.diff(g)) + 1
    for seg_g, seg_v in zip(np.split(g, bounds), np.split(v, bounds)):
        if len(seg_g):
            out[seg_g[0]] = np.median(seg_v)
    return out


def _group_max(groups, values, n):
    out = np.zeros(n)
    np.maximum.at(out, groups, values)
    return out


def calibrate_noise(obs: TrackObservations, noise, static_mask) -> dict:
    """Robust (MAD) scale of residuals on the static core -> multiplicative noise corrections.

    Candidate-independent and computed once; returns the factors applied, for the manifest."""
    tr_ok = static_mask[obs.track]
    out = dict(n_static_tracks=int(static_mask.sum()))
    v = obs.uv_valid & tr_ok
    if v.sum() > 50:
        N = obs.n_tracks
        med = np.stack([_group_median(obs.track[v], obs.uv[v, d], N) for d in range(2)], 1)
        dev = (obs.uv[v] - med[obs.track[v]]).ravel()
        sigma = 1.4826 * np.median(np.abs(dev))
        out["sigma_px_measured"] = float(sigma)
    h = obs.has_3d & tr_ok
    if h.sum() > 50:
        N = obs.n_tracks
        medz = _group_median(obs.track[h], obs.z[h], N)
        r = (obs.z[h] - medz[obs.track[h]]) / depth_sigma(obs.z[h], noise, obs.depth_edge[h], obs.depth_cos[h])
        out["depth_scale_measured"] = float(1.4826 * np.median(np.abs(r)))
    return out


# --------------------------------------------------------------------------- finite-motion proposals
@dataclass
class MotionProposal:
    t_a: int
    t_b: int
    camera: int
    R: np.ndarray
    t: np.ndarray
    n_candidates: int
    n_inliers: int
    inlier_tracks: np.ndarray
    rms_m: float
    centroid_a: np.ndarray | None = None   # inlier centroid at t_a (world)

    @property
    def support_displacement(self) -> np.ndarray:
        """Mean displacement of the moving support points (not the motion of the world origin)."""
        c = np.zeros(3) if self.centroid_a is None else self.centroid_a
        return self.R @ c + self.t - c


@dataclass
class ProposalSet:
    proposals: list = field(default_factory=list)
    static_core: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)


def propose_finite_motions(obs: TrackObservations, noise, fit_cfg, rng=None, max_pairs_per_group=None) -> ProposalSet:
    """Robust rigid fits of the *non-static* tracks between pairs of timestamps.

    Groups are (camera, query time); within a group the pair (query, t) uses tracks with valid 3D
    observations at both times. Pairs whose candidate set is too small are reported as
    ``no_motion`` rather than forced."""
    rng = np.random.default_rng(fit_cfg.seed) if rng is None else rng
    h = obs.has_3d
    tr, tm, x, zh = obs.track[h], obs.time[h], obs.x_w[h], obs.z[h]
    lookup = {}
    for i, (a, b) in enumerate(zip(tr, tm)):
        lookup[(int(a), int(b))] = i
    out = ProposalSet()
    pairs_report = []
    groups = {}
    for i in range(obs.n_tracks):
        groups.setdefault((int(obs.track_camera[i]), int(obs.track_query_time[i])), []).append(i)
    for (cam, tq), tracks in sorted(groups.items()):
        tracks = np.asarray(tracks)
        times = sorted({int(t) for t in tm[np.isin(tr, tracks)]} - {tq})
        for tb in times[:max_pairs_per_group] if max_pairs_per_group else times:
            ia = [lookup.get((int(i), tq)) for i in tracks]
            ib = [lookup.get((int(i), tb)) for i in tracks]
            both = np.array([a is not None and b is not None for a, b in zip(ia, ib)])
            if both.sum() < fit_cfg.min_moving_tracks:
                continue
            ids = tracks[both]
            P = x[[ia[k] for k in np.flatnonzero(both)]]
            Q = x[[ib[k] for k in np.flatnonzero(both)]]
            z = zh[[ia[k] for k in np.flatnonzero(both)]]
            sig = depth_sigma(z, noise) * np.sqrt(2)
            chi2 = np.sum((Q - P) ** 2, 1) / sig ** 2
            moving = chi2 > fit_cfg.moving_chi2_threshold
            rep = dict(camera=cam, t_a=tq, t_b=tb, n_pairs=int(len(ids)), n_candidates=int(moving.sum()))
            if moving.sum() < fit_cfg.min_moving_tracks:
                rep["status"] = "no_motion_candidates"
                pairs_report.append(rep)
                continue
            thr = fit_cfg.ransac_threshold_sigma * float(np.median(sig[moving]))
            res = ransac_rigid(P[moving], Q[moving], thr, fit_cfg.ransac_iterations, rng,
                               min_inliers=fit_cfg.min_moving_tracks)
            if res is None:
                rep["status"] = "ransac_failed"
                pairs_report.append(rep)
                continue
            R, t, inl = res
            rms = float(np.sqrt(np.mean(np.sum((P[moving][inl] @ R.T + t - Q[moving][inl]) ** 2, 1))))
            out.proposals.append(MotionProposal(tq, tb, cam, R, t, int(moving.sum()), int(inl.sum()),
                                                ids[moving][inl], rms, centroid_a=P[moving][inl].mean(0)))
            rep.update(status="ok", n_inliers=int(inl.sum()), rms_m=rms,
                       rot_deg=float(np.degrees(np.linalg.norm(log_so3(R)))), trans_m=float(np.linalg.norm(t)))
            pairs_report.append(rep)
    out.diagnostics["pairs"] = pairs_report
    return out


def _graph_solve(T, ref, edges):
    """Solve q_b - q_a = val (weighted) with q_ref = 0; unconnected times -> NaN."""
    if not edges:
        q = np.full(T, np.nan)
        q[ref] = 0.0
        return q, np.zeros(T, bool)
    # connectivity from ref
    adj = {t: set() for t in range(T)}
    for a, b, _, _ in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, stack = {ref}, [ref]
    while stack:
        n = stack.pop()
        for m in adj[n]:
            if m not in seen:
                seen.add(m)
                stack.append(m)
    idx = sorted(seen - {ref})
    col = {t: i for i, t in enumerate(idx)}
    rows, rhs = [], []
    for a, b, val, w in edges:
        if a not in seen or b not in seen:
            continue
        r = np.zeros(len(idx))
        if b != ref:
            r[col[b]] += 1
        if a != ref:
            r[col[a]] -= 1
        rows.append(r * w)
        rhs.append(val * w)
    q = np.full(T, np.nan)
    q[ref] = 0.0
    if idx:
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        q[idx] = sol
    valid = np.zeros(T, bool)
    valid[list(seen)] = True
    return q, valid


def interpolate_missing(q, valid, timestamps):
    """Fill unobserved joint values by interpolation in time; returned mask marks them as estimated."""
    q = q.copy()
    if valid.sum() >= 2:
        q[~valid] = np.interp(timestamps[~valid], timestamps[valid], q[valid])
    elif valid.sum() == 1:
        q[~valid] = q[valid][0]
    else:
        q[:] = 0.0
    return q


@dataclass
class Seed:
    kind: str
    axis: np.ndarray
    origin: np.ndarray | None
    q: np.ndarray
    q_observed: np.ndarray
    diagnostics: dict


def seeds_from_proposals(props: ProposalSet, obs: TrackObservations, ref: int, kind: str) -> list:
    """One seed per proposal subset; both kinds read the same proposals."""
    T = obs.frame_map.T
    ts = obs.frame_map.timestamp_s
    good = [p for p in props.proposals if p.n_inliers >= 3]
    if not good:
        return []
    out = []
    if kind == "prismatic":
        tvec = np.array([p.support_displacement for p in good])
        w = np.array([p.n_inliers for p in good], float)
        U, S, Vt = np.linalg.svd(tvec * np.sqrt(w)[:, None], full_matrices=False)
        u = normalize(Vt[0])
        edges = [(p.t_a, p.t_b, float(u @ p.support_displacement), np.sqrt(p.n_inliers)) for p in good]
        q, valid = _graph_solve(T, ref, edges)
        out.append(Seed("prismatic", u, None, interpolate_missing(q, valid, ts), valid,
                        dict(singular_values=S.tolist(), n_proposals=len(good))))
        return out
    rv = np.array([log_so3(p.R) for p in good])
    w = np.array([p.n_inliers for p in good], float)
    U, S, Vt = np.linalg.svd(rv * np.sqrt(w)[:, None], full_matrices=False)
    u = normalize(Vt[0])
    edges = [(p.t_a, p.t_b, float(u @ r), np.sqrt(p.n_inliers)) for p, r in zip(good, rv)]
    q, valid = _graph_solve(T, ref, edges)
    # hinge: t_ab = (I - R_ab) o for a pure hinge; gauge u.(o - c) = 0
    pts = obs.x_w[obs.has_3d & np.isin(obs.track, np.concatenate([p.inlier_tracks for p in good]))]
    c = np.median(pts, 0) if len(pts) else np.zeros(3)
    A = [np.eye(3) - p.R for p in good] + [u[None, :] * 10.0]
    b = [p.t for p in good] + [np.array([10.0 * (u @ c)])]
    A, b = np.concatenate(A), np.concatenate(b)
    o, *_ = np.linalg.lstsq(A, b, rcond=None)
    sv = np.linalg.svd(A, compute_uv=False)
    diag = dict(singular_values=S.tolist(), n_proposals=len(good), hinge_ls_condition=float(sv[0] / max(sv[-1], 1e-30)),
                max_abs_angle_deg=float(np.degrees(np.nanmax(np.abs(q)))))
    out.append(Seed("revolute", u, o, interpolate_missing(q, valid, ts), valid, diag))
    # remote-hinge seeds: a revolute joint far away approximates the observed translation, so the
    # revolute hypothesis gets a fair chance to explain prismatic-looking data (and vice versa).
    tvec = np.array([p.support_displacement for p in good])
    _, _, Vt_t = np.linalg.svd(tvec * np.sqrt(w)[:, None], full_matrices=False)
    d = normalize(Vt_t[0])
    rho = 1.0
    for e in tangent_basis(d):
        o_far = c - rho * np.cross(d, e)          # u x (c - o) = rho * d at the support centroid
        edges_t = [(p.t_a, p.t_b, float(d @ p.support_displacement) / rho, np.sqrt(p.n_inliers)) for p in good]
        qt, vt = _graph_solve(T, ref, edges_t)
        out.append(Seed("revolute", e, o_far, interpolate_missing(qt, vt, ts), vt,
                        dict(diag, remote_hinge_seed=True, remote_radius_m=rho)))
    # small-arc alternative: hinge at the far side of the moving region (multiple plausible seeds)
    if len(pts):
        e1, e2 = tangent_basis(u)
        spread = float(np.max(np.linalg.norm(pts - c, axis=1))) if len(pts) > 1 else 0.1
        for sgn in (+1, -1):
            for e in (e1, e2):
                out.append(Seed("revolute", u, c + sgn * spread * e, interpolate_missing(q, valid, ts), valid,
                                dict(diag, alternative_hinge_seed=True)))
    return out
