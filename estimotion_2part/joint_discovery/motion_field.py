"""Hinge-free rigid-motion evidence (spec section 2).

For material points i, j of one rigid part observed at frames a and b,

    X_i(b) - X_j(b) = R_ba (X_i(a) - X_j(a))            (translation cancels exactly)

so a centred weighted Kabsch fit of tracked correspondences measures the finite rotation R_ba of the support without
any hinge location.  ``estimate_interval_rigid_evidence`` does that robustly (RANSAC + Cauchy IRLS) with a
delete-one-surface-cluster jackknife covariance (pairs/tracks are never treated as independent).  An image-space
refinement re-estimates (R, t) per camera from 3D points at frame a and *2D* tracks at frame b, so a camera with
unreliable depth at b cannot dictate the rotation.

``estimate_shared_angular_axis`` fits ONE fixed axis u with signed per-interval angles (spec 2.4);
``hinge_from_intervals`` estimates the hinge line second, from t_k = (I - R_k) o, and reports its covariance
(small angles -> huge covariance -> HINGE_UNOBSERVABLE, never a precise remote pivot).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .blocked_statistics import block_bootstrap, cluster_jackknife, effective_n_from_clusters, spatial_clusters
from .candidates import kabsch, ransac_rigid
from .observations import TrackObservations, depth_sigma
from .rigid_kinematics import exp_so3, log_so3, normalize


@dataclass
class RigidIntervalEvidence:
    frame_a: int
    frame_b: int
    camera_ids: tuple
    n_tracks: int
    n_effective_tracks: float          # Kish-effective number of surface clusters (NOT the track count)
    spatial_baseline_m: float          # 2 * second principal std of the support at frame a
    rotation_matrix: np.ndarray
    translation_vec_m: np.ndarray
    rotation_vector_rad: np.ndarray
    rotation_cov_rad2: np.ndarray | None
    translation_cov_m2: np.ndarray | None
    singular_values: np.ndarray        # principal stds of the centred support [m]
    rank: int
    condition: float
    per_camera_residuals: dict
    diagnostics: dict = field(default_factory=dict)
    inlier_tracks: np.ndarray | None = None
    centroid_a: np.ndarray | None = None
    centroid_b: np.ndarray | None = None
    status: str = "OK"

    @property
    def angle_rad(self) -> float:
        return float(np.linalg.norm(self.rotation_vector_rad))

    def to_row(self) -> dict:
        w = self.rotation_vector_rad
        sd = np.sqrt(np.trace(self.rotation_cov_rad2)) if self.rotation_cov_rad2 is not None else np.nan
        return dict(frame_a=self.frame_a, frame_b=self.frame_b, status=self.status, cameras="+".join(self.camera_ids),
                    n_tracks=self.n_tracks, n_eff_clusters=round(self.n_effective_tracks, 2),
                    baseline_m=round(self.spatial_baseline_m, 4), rank=self.rank, condition=round(self.condition, 2),
                    angle_deg=round(float(np.degrees(np.linalg.norm(w))), 4),
                    omega_x=float(w[0]), omega_y=float(w[1]), omega_z=float(w[2]),
                    omega_sd_deg=round(float(np.degrees(sd)), 4) if np.isfinite(sd) else None,
                    t_x=float(self.translation_vec_m[0]), t_y=float(self.translation_vec_m[1]),
                    t_z=float(self.translation_vec_m[2]),
                    centroid_disp_m=round(float(np.linalg.norm(self.centroid_b - self.centroid_a)), 5)
                    if self.centroid_a is not None else None,
                    robust_residual_mm=round(1000 * self.diagnostics.get("robust_scale_m", np.nan), 3),
                    inlier_fraction=round(self.diagnostics.get("inlier_fraction", np.nan), 3))


# --------------------------------------------------------------------------- point access
class PointTable:
    """Depth-anchored 3D positions per (track, time) with a scalar 3D sigma, built once per observation table."""

    def __init__(self, obs: TrackObservations, noise, calibs=None, exclude_depth_edges=True):
        ok = obs.has_3d & obs.uv_valid
        if exclude_depth_edges:
            ok &= ~obs.depth_edge
        self.obs = obs
        self.T, self.N = obs.frame_map.T, obs.n_tracks
        self.X = np.full((self.T, self.N, 3), np.nan)
        self.S = np.full((self.T, self.N), np.nan)
        self.UV = np.full((self.T, self.N, 2), np.nan)
        z = obs.z[ok]
        sz = depth_sigma(z, noise, obs.depth_edge[ok], obs.depth_cos[ok])
        f = np.array([calibs[c].K[0, 0] for c in obs.camera_ids])[obs.cam[ok]] if calibs else np.full(ok.sum(), 500.0)
        lat = noise.sigma_px * z / f
        self.X[obs.time[ok], obs.track[ok]] = obs.x_w[ok]
        self.S[obs.time[ok], obs.track[ok]] = np.sqrt(sz ** 2 + 2 * lat ** 2)
        uvok = obs.uv_valid
        self.UV[obs.time[uvok], obs.track[uvok]] = obs.uv[uvok]
        self.track_camera = obs.track_camera

    def pair(self, a, b, track_mask=None, cameras=None):
        k = np.isfinite(self.X[a]).all(1) & np.isfinite(self.X[b]).all(1)
        if track_mask is not None:
            k &= track_mask
        if cameras is not None:
            k &= np.isin(self.track_camera, cameras)
        idx = np.flatnonzero(k)
        return idx, self.X[a, idx], self.X[b, idx], np.sqrt(self.S[a, idx] ** 2 + self.S[b, idx] ** 2)


def support_principal(X, w=None):
    X = np.asarray(X, float)
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    m = w @ X
    C = ((X - m) * w[:, None]).T @ (X - m)
    ev, V = np.linalg.eigh(C)
    sd = np.sqrt(np.clip(ev[::-1], 0, None))
    return sd, V[:, ::-1], m


# --------------------------------------------------------------------------- robust interval fit
def _irls_kabsch(P, Q, sig, w0, iters=8):
    """Cauchy IRLS on 3D residual / (sigma * scale); scale re-estimated robustly each iteration."""
    w = w0.copy()
    R, t = kabsch(P, Q, w)
    scale = None
    for _ in range(iters):
        r = np.linalg.norm(P @ R.T + t - Q, axis=1)
        rn = r / sig
        scale = max(1.4826 * np.median(rn[w > 0.5 * w.max()]) if (w > 0.5 * w.max()).sum() >= 3 else np.median(rn), 1e-6)
        c = 2.385 * scale
        w = w0 / (1 + (rn / c) ** 2) / sig ** 2
        R, t = kabsch(P, Q, w)
    r = np.linalg.norm(P @ R.T + t - Q, axis=1)
    return R, t, w, r, scale


def estimate_interval_rigid_evidence(obs_or_table, support_weights, frame_a, frame_b, noise=None, calibs=None, *,
                                     seed=0, cameras=None, voxel_m=0.05, min_tracks=8, min_clusters=4,
                                     ransac_threshold_m=None, refine_2d=True) -> RigidIntervalEvidence:
    """Robust finite rigid motion of the weighted support between two frames (spec 2.2).

    ``support_weights``: per-track probability of belonging to the moving support (candidate independent); tracks
    with weight < 0.5 are ignored.  Only tracks with 3D at *both* endpoints enter (partial tracks are fine)."""
    tab = obs_or_table if isinstance(obs_or_table, PointTable) else PointTable(obs_or_table, noise, calibs)
    sw = np.asarray(support_weights, float)
    cam_idx = None
    obs = tab.obs
    if cameras is not None:
        cam_idx = [obs.camera_ids.index(c) for c in cameras]
    idx, P, Q, sig = tab.pair(frame_a, frame_b, sw >= 0.5, cam_idx)
    cam_names = tuple(cameras) if cameras is not None else tuple(obs.camera_ids)
    empty = RigidIntervalEvidence(frame_a, frame_b, cam_names, len(idx), 0.0, 0.0, np.eye(3), np.zeros(3), np.zeros(3),
                                  None, None, np.zeros(3), 0, np.inf, {}, {}, status="INSUFFICIENT_SUPPORT")
    if len(idx) < min_tracks:
        empty.diagnostics = dict(reason=f"{len(idx)} tracks observed in 3D at both frames (< {min_tracks})")
        return empty
    rng = np.random.default_rng(seed)
    thr = ransac_threshold_m or max(0.01, 4 * float(np.median(sig)))
    rs = ransac_rigid(P, Q, thr, 300, rng, min_inliers=max(6, min_tracks // 2))
    w0 = sw[idx].copy()
    if rs is not None:
        w0 = w0 * rs[2]           # RANSAC outliers (e.g. a second rigid group) never enter the fit
    R, t, w, r, scale = _irls_kabsch(P, Q, sig, w0)
    # inliers against the absolute, noise-based RANSAC threshold: an IRLS scale inflated by a mixed second group must
    # not turn every track into an inlier
    inl = (r < thr) & (w0 > 0)
    clusters = spatial_clusters(P, voxel_m)
    n_cl = len(np.unique(clusters[inl])) if inl.any() else 0
    sd, V, m = support_principal(P[inl] if inl.sum() >= 3 else P, None)
    rank = int(np.sum(sd > 0.005))

    def est(keep):
        kk = keep & (w > 0)
        if kk.sum() < 3:
            return None
        Rk, _ = kabsch(P[kk], Q[kk], w[kk])
        return log_so3(Rk)

    def est_t(keep):
        kk = keep & (w > 0)
        if kk.sum() < 3:
            return None
        return kabsch(P[kk], Q[kk], w[kk])[1]

    cov_w, G = cluster_jackknife(est, clusters, n_min=min_clusters)
    cov_t, _ = cluster_jackknife(est_t, clusters, n_min=min_clusters)
    per_cam = {}
    for ci, cid in enumerate(obs.camera_ids):
        k = tab.track_camera[idx] == ci
        if k.sum() == 0:
            continue
        entry = dict(n_tracks=int(k.sum()), n_inliers=int((k & inl).sum()),
                     median_residual_mm=float(1000 * np.median(r[k & inl])) if (k & inl).any() else None)
        kk = k & inl
        if kk.sum() >= min_tracks and len(np.unique(clusters[kk])) >= min_clusters:
            Rc, tc = kabsch(P[kk], Q[kk], w[kk])
            entry["rotation_vector_3d_rad"] = log_so3(Rc).tolist()
            if refine_2d and calibs is not None:
                r2 = refine_interval_2d(tab, idx[kk], frame_a, frame_b, calibs[cid], Rc, tc)
                if r2 is not None:
                    entry.update(r2)
        per_cam[cid] = entry
    ev = RigidIntervalEvidence(
        frame_a, frame_b, cam_names, int(len(idx)), effective_n_from_clusters(clusters[inl], w[inl]) if inl.any() else 0.0,
        float(2 * sd[1]) if len(sd) > 1 else 0.0, R, t, log_so3(R), cov_w, cov_t, sd, rank,
        float(sd[0] / max(sd[2], 1e-6)), per_cam,
        dict(inlier_fraction=float(inl.mean()), robust_scale_m=float(scale * np.median(sig)),
             robust_scale_whitened=float(scale), ransac_threshold_m=float(thr), n_clusters=int(n_cl),
             n_clusters_total=int(G), principal_axes=V.T.tolist(), ransac_ok=rs is not None),
        inlier_tracks=idx[inl], centroid_a=(w @ P) / w.sum(), centroid_b=(w @ Q) / w.sum())
    if n_cl < min_clusters or cov_w is None:
        ev.status = "INSUFFICIENT_CLUSTERS"
    return ev


def refine_interval_2d(tab: PointTable, tracks, a, b, calib, R0, t0, f_scale_px=2.0):
    """Re-estimate (R, t) from 3D points at frame a and the 2D tracks at frame b of ONE camera (no depth at b)."""
    tracks = np.asarray(tracks)
    uvb = tab.UV[b, tracks]
    ok = np.isfinite(uvb).all(1)
    if ok.sum() < 8:
        return None
    Xa, uvb = tab.X[a, tracks[ok]], uvb[ok]
    T_CW = np.linalg.inv(calib.T_WC)
    K = calib.K

    def res(p):
        Rm = exp_so3(p[:3])
        Xc = (Xa @ Rm.T + p[3:]) @ T_CW[:3, :3].T + T_CW[:3, 3]
        uv = np.stack([K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]], 1)
        return (uv - uvb).ravel()

    sol = least_squares(res, np.r_[log_so3(R0), t0], loss="cauchy", f_scale=f_scale_px, max_nfev=200)
    J = sol.jac
    rr = sol.fun.reshape(-1, 2)
    s = 1.4826 * np.median(np.linalg.norm(rr, axis=1))
    try:
        cov = np.linalg.inv(J.T @ J) * max(s, 1e-3) ** 2
        sd = np.sqrt(np.clip(np.diag(cov)[:3], 0, None))
    except np.linalg.LinAlgError:
        sd = np.full(3, np.nan)
    # the per-observation covariance is optimistic (correlated tracks); the jackknife on 3D is the decision input
    return dict(rotation_vector_2d_rad=sol.x[:3].tolist(), translation_2d_m=sol.x[3:].tolist(),
                reprojection_median_px=float(np.median(np.linalg.norm(rr, axis=1))),
                rotation_2d_sd_deg_optimistic=np.degrees(sd).tolist())


# --------------------------------------------------------------------------- chained intervals
def chained_intervals(tab, support, boundaries, noise=None, calibs=None, cameras=None, seed=0, **kw) -> list:
    """Evidence on consecutive non-overlapping intervals [b_k, b_{k+1}]; increments about a fixed axis add exactly."""
    out = []
    for k in range(len(boundaries) - 1):
        out.append(estimate_interval_rigid_evidence(tab, support, int(boundaries[k]), int(boundaries[k + 1]), noise,
                                                    calibs, seed=seed + k, cameras=cameras, **kw))
    return out


# --------------------------------------------------------------------------- shared axis
def estimate_shared_angular_axis(intervals, timestamps=None, *, robust_config=None, min_sigma=3.0) -> dict:
    """min_{|u|=1, dtheta_k} sum_k rho(|log R_k - dtheta_k u|^2_{Sigma_k^-1}) with a fixed sign gauge (spec 2.4).

    Intervals whose rotation is not distinguishable from zero (|w|/sd < ``min_sigma``) do not constrain u but still
    receive a signed angle afterwards. Sign gauge: the net signed rotation over the clip is >= 0."""
    rc = dict(loss="soft_l1", f_scale=2.0, n_boot=500, seed=0)
    rc.update(robust_config or {})
    good = [iv for iv in intervals if iv.status == "OK" and iv.rotation_cov_rad2 is not None]
    if not good:
        return dict(status="NO_VALID_INTERVALS", axis=None)
    Ws, ws = [], []
    for iv in good:
        C = iv.rotation_cov_rad2 + np.eye(3) * (np.radians(0.05) ** 2)
        Ws.append(np.linalg.inv(C))
        ws.append(iv.rotation_vector_rad)
    Ws, ws = np.stack(Ws), np.stack(ws)
    sig = np.array([np.linalg.norm(w) / np.sqrt(max(w @ np.linalg.inv(W) @ w / max(w @ w, 1e-30), 1e-30))
                    for w, W in zip(ws, Ws)])
    informative = sig >= min_sigma
    if informative.sum() == 0:
        return dict(status="NO_INFORMATIVE_INTERVAL", axis=None, interval_sigma=sig.tolist(),
                    reason=f"no interval rotation exceeds {min_sigma} of its own cluster-jackknife sd")

    def angles(u, W, w):
        return (u @ W @ w) / max(u @ W @ u, 1e-30)

    def resid(p, W_, w_):
        th, ph = p
        u = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        out = []
        for W, w in zip(W_, w_):
            L = np.linalg.cholesky(W)
            r = w - angles(u, W, w) * u
            out.append(L.T @ r)
        return np.concatenate(out)

    def fit(W_, w_):
        M = sum(np.outer(w, w) for w in w_)
        starts = [np.linalg.eigh(M)[1][:, -1], *np.eye(3)]
        best = None
        for u0 in starts:
            u0 = normalize(u0)
            p0 = [np.arccos(np.clip(u0[2], -1, 1)), np.arctan2(u0[1], u0[0])]
            s = least_squares(resid, p0, args=(W_, w_), loss=rc["loss"], f_scale=rc["f_scale"])
            if best is None or s.cost < best.cost:
                best = s
        th, ph = best.x
        return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]), best

    u, sol = fit(Ws[informative], ws[informative])
    dth = np.array([angles(u, W, w) for W, w in zip(Ws, ws)])
    if dth.sum() < 0:
        u, dth = -u, -dth
    out_of_axis = np.array([np.degrees(np.linalg.norm(w - a * u)) for w, a in zip(ws, dth)])
    # axis uncertainty: bootstrap over informative intervals (each interval = one independent time block)
    rng = np.random.default_rng(rc["seed"])
    inf_idx = np.flatnonzero(informative)
    cone = None
    boot_axes = []
    if len(inf_idx) >= 2:
        for _ in range(rc["n_boot"] if len(inf_idx) >= 3 else 0):
            pick = rng.choice(inf_idx, len(inf_idx))
            ub, _ = fit(Ws[pick], ws[pick])
            boot_axes.append(ub if ub @ u >= 0 else -ub)
        if boot_axes:
            ang = np.degrees(np.arccos(np.clip(np.stack(boot_axes) @ u, -1, 1)))
            cone = float(np.percentile(ang, 95))
    return dict(status="OK", axis=u.tolist(), interval_frames=[(iv.frame_a, iv.frame_b) for iv in good],
                signed_angles_rad=dth.tolist(), net_angle_rad=float(dth.sum()),
                interval_sigma=sig.round(3).tolist(), informative=informative.tolist(),
                n_informative=int(informative.sum()), out_of_axis_deg=out_of_axis.round(4).tolist(),
                axis_cone95_deg=cone, axis_identifiable=bool(cone is not None and cone < 20.0),
                optimizer_cost=float(sol.cost), sign_gauge="net signed rotation over the clip >= 0")


# --------------------------------------------------------------------------- rotation vs translation
def rotation_vs_translation_evidence(intervals, shared_axis: dict, *, null: dict | None = None,
                                     camera_intervals: dict | None = None, bootstrap_config=None,
                                     min_blocks=3) -> dict:
    """Net rotation about the shared axis and net centroid translation, bootstrapped over interval blocks; compared
    with a calibrated no-rotation null (``rotation_observability.calibrate_rotation_null``)."""
    bc = dict(n_boot=2000, seed=0)
    bc.update(bootstrap_config or {})
    good = [iv for iv in intervals if iv.status == "OK"]
    out = dict(n_intervals=len(intervals), n_valid_intervals=len(good))
    if shared_axis.get("axis") is None or len(good) < min_blocks:
        # rotation magnitude without an axis: norm of each interval's rotation (upper bound only)
        mags = [np.degrees(iv.angle_rad) for iv in good]
        out.update(status="INSUFFICIENT_INDEPENDENT_EVIDENCE" if len(good) < min_blocks else "NO_SHARED_AXIS",
                   rotation_detected=False, per_interval_angle_deg=mags,
                   rotation_upper_bound_deg=float(np.sum(mags)) if mags else None)
    else:
        u = np.asarray(shared_axis["axis"])
        frames = [tuple(f) for f in shared_axis["interval_frames"]]
        dth = dict(zip(frames, shared_axis["signed_angles_rad"]))
        per = [np.degrees(dth[(iv.frame_a, iv.frame_b)]) for iv in good if (iv.frame_a, iv.frame_b) in dth]
        bb = block_bootstrap(per, bc["n_boot"], bc["seed"])
        out.update(status=bb["status"], net_rotation_deg=bb["estimate"], net_rotation_ci95_deg=bb["ci"],
                   per_interval_deg=np.round(per, 4).tolist(), n_blocks=bb["n_blocks"])
        lo_hi = bb["ci"] or [np.nan, np.nan]
        out["rotation_upper_bound_deg"] = float(max(abs(lo_hi[0]), abs(lo_hi[1])))
        nq = None if null is None else null.get("net_abs_rotation_q95_deg")
        out["null_q95_deg"] = nq
        ci_excludes_zero = bb["ci"] is not None and (lo_hi[0] > 0 or lo_hi[1] < 0)
        exceeds_null = nq is not None and abs(bb["estimate"]) > nq
        out["ci_excludes_zero"] = bool(ci_excludes_zero)
        out["exceeds_calibrated_null"] = bool(exceeds_null) if nq is not None else None
        out["rotation_detected"] = bool(ci_excludes_zero and (exceeds_null if nq is not None else False))
        if nq is None:
            out["note"] = "no calibrated null available: rotation is never declared detected without one"
    # translation of the support centroid (spatially uniform component)
    disp = [iv.centroid_b - iv.centroid_a for iv in good]
    if disp:
        D = np.stack(disp)
        out["net_centroid_translation_m"] = D.sum(0).tolist()
        out["net_centroid_translation_norm_m"] = float(np.linalg.norm(D.sum(0)))
        out["path_length_m"] = float(np.linalg.norm(D, axis=1).sum())
        if shared_axis.get("axis") is not None:
            u = np.asarray(shared_axis["axis"])
            out["translation_along_axis_m"] = float(D.sum(0) @ u)
    # camera sensitivity: each camera alone (never pooled into a 2-sample bootstrap)
    if camera_intervals:
        sens = {}
        for cid, ivs in camera_intervals.items():
            sa = estimate_shared_angular_axis(ivs, robust_config=dict(n_boot=0))
            sens[cid] = dict(status=sa["status"], net_angle_deg=None if sa.get("axis") is None
                             else float(np.degrees(sa["net_angle_rad"])),
                             axis=sa.get("axis"),
                             axis_vs_pooled_deg=None if sa.get("axis") is None or shared_axis.get("axis") is None
                             else float(np.degrees(np.arccos(min(1, abs(np.dot(sa["axis"], shared_axis["axis"])))))))
        out["per_camera"] = sens
    return out


# --------------------------------------------------------------------------- hinge second
def hinge_from_intervals(intervals, axis, min_angle_deg=0.5, n_boot=500, seed=0) -> dict:
    """Hinge line (u, o) from the finite motions: t_k = (I - R_k) o + h_k u (h_k = pitch, 0 for a revolute).

    Only the two coordinates of o perpendicular to u are estimable (along-axis gauge).  Reports the screw pitch and
    the line covariance; a tiny (I - R) makes the covariance explode and the status HINGE_UNOBSERVABLE."""
    if axis is None:
        return dict(status="NO_AXIS")
    u = normalize(np.asarray(axis, float))
    e1 = normalize(np.cross(u, [1, 0, 0] if abs(u[0]) < 0.9 else [0, 1, 0]))
    e2 = np.cross(u, e1)
    B = np.stack([e1, e2], 1)
    rows = [iv for iv in intervals if iv.status == "OK" and np.degrees(iv.angle_rad) >= min_angle_deg]
    if len(rows) < 2:
        return dict(status="HINGE_UNOBSERVABLE", reason=f"{len(rows)} intervals rotate >= {min_angle_deg} deg")

    def solve(rs):
        A = np.concatenate([(np.eye(3) - iv.rotation_matrix) @ B for iv in rs])
        b = np.concatenate([iv.translation_vec_m - (iv.translation_vec_m @ u) * u for iv in rs])
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x, A

    x, A = solve(rows)
    o = B @ x
    pitch = [float(iv.translation_vec_m @ u) for iv in rows]
    rng = np.random.default_rng(seed)
    boots = np.stack([solve([rows[i] for i in rng.integers(0, len(rows), len(rows))])[0] for _ in range(n_boot)])
    cov = np.cov(boots.T)
    sd = float(np.sqrt(max(np.linalg.eigvalsh(cov).max(), 0)))
    s = np.linalg.svd(A, compute_uv=False)
    status = "OK" if sd < 0.05 else "HINGE_UNOBSERVABLE"
    return dict(status=status, axis=u.tolist(), origin=o.tolist(), line_cov_m2=cov.tolist(), line_sd_m=sd,
                design_singular_values=s.tolist(), screw_pitch_m=pitch, n_intervals=len(rows),
                gauge="origin is the point of the line closest to the world origin within span(e1, e2)",
                note="bootstrap over rotating intervals (time blocks)")
