"""Stage A: counterfactual RGB-vs-depth likelihood audit of frozen joint hypotheses (no refit).

Question answered: *which observations, and why,* make a physically plausible joint less likely
than the fitted one under the current likelihood. Every frozen hypothesis (axis, hinge, q(t)) is
evaluated on the *identical* observation table. The joint is never re-optimized; only the
per-track canonical anchors (nuisance parameters) are re-estimated with the joint held fixed, by
the same robust Gauss-Newton used in fitting (``geometry_fit.refine_anchors``), for every
hypothesis alike. Anchors are estimated twice, from RGB-D and from 2D only, so the depth terms
can also be scored against geometry that the depth did not shape.

Reference hinges (e.g. the tracker-free study) may be evaluated with ``role='reference_posthoc'``.
They are reported post hoc only and never enter fitting, thresholds or support selection.

Per-observation exports (one row per observation, rejected rows kept with a reason code):
``r_uv_px, r_z_m, sigma_z_m, depth_edge, incidence_cos, depth_valid, mask_valid, track_id,
moving_posterior, original_pixel, predicted_pixel, measured_z, predicted_z, patch_id,
image_cell_id, dist_to_depth_discontinuity_px, front_surface_gap_m, valid_measured_depth,
residual_direction, depth_class, reason``.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import assignments as asg
from .geometry_fit import refine_anchors
from .observations import TrackObservations, depth_sigma
from .rigid_kinematics import apply_joint_inverse, axis_angle_deg, line_to_line_distance, normalize
from .visual_losses import camera_stacks, whitened_residuals

ROLES = ("fitted", "diagnostic", "null", "reference_posthoc")
DEPTH_CLASSES = ("not_depth_scored", "valid_surface_random", "bias_valid_surface", "wrong_surface_or_occlusion",
                 "robot_contamination", "tracking_2d_error", "unexplained_outlier")
REASONS = ("used", "ineligible_track", "uv_invalid", "outside_object_mask_depth_unused", "depth_invalid",
           "depth_edge_rejected")


@dataclass
class FrozenJoint:
    name: str
    kind: str                    # 'static' | 'revolute' | 'prismatic'
    axis: np.ndarray | None
    origin: np.ndarray | None
    q: np.ndarray
    role: str = "fitted"
    source: str = ""

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        self.q = np.asarray(self.q, float)
        if self.axis is not None:
            self.axis = normalize(np.asarray(self.axis, float))
        if self.origin is not None:
            self.origin = np.asarray(self.origin, float)

    def describe(self) -> dict:
        d = dict(name=self.name, kind=self.kind, role=self.role, source=self.source)
        if self.kind != "static":
            span = float(np.nanmax(self.q) - np.nanmin(self.q))
            d.update(axis=self.axis.tolist(), origin=None if self.origin is None else self.origin.tolist(),
                     q_span=span, q_span_readable=(f"{np.degrees(span):.2f} deg" if self.kind == "revolute"
                                                   else f"{100 * span:.2f} cm"))
        return d


def load_frozen_from_run(run_dir, prefix, role="fitted", kinds=("revolute", "prismatic")) -> dict:
    """Frozen candidates from a ``fit`` run directory (``joint_candidates.json``)."""
    run_dir = Path(run_dir)
    jc = json.loads((run_dir / "joint_candidates.json").read_text())["candidates"]
    out = {}
    for k in kinds:
        if k in jc:
            c = jc[k]
            out[f"{prefix}_{k[0].upper()}"] = FrozenJoint(f"{prefix}_{k[0].upper()}", k, c["joint_axis_world"],
                                                          c["joint_origin_world"], c["q"], role, str(run_dir))
    return out


def reference_joint_posthoc(ref_json, frame_map, camera) -> FrozenJoint:
    """Tracker-free reference hinge as a frozen revolute (POST HOC ONLY; never used for fitting)."""
    ref = json.loads(Path(ref_json).read_text())
    frames = ref["cameras"][camera]["frames"]
    theta = []
    for f in frame_map.rgbd_frame_index:
        e = frames.get(str(int(f)))
        theta.append(np.nan if e is None else np.radians(e["theta_deg"]))
    theta = np.asarray(theta)
    ok = np.isfinite(theta)
    if ok.sum() < 2:
        raise ValueError(f"reference has < 2 angles for {camera} in this window")
    theta = np.interp(np.arange(len(theta)), np.flatnonzero(ok), theta[ok])
    piv = ref["cameras"][camera].get("pivot", ref.get("hinge_point_nominal"))
    return FrozenJoint(f"REF_{camera}_posthoc", "revolute", ref["hinge_direction"], piv, theta - theta[0],
                       "reference_posthoc", f"{ref_json} [{camera}] (diagnostic only)")


# --------------------------------------------------------------------------- per-observation geometry cues
def depth_discontinuity_features(depth, uv, abs_m=0.01, rel=0.02, window=3):
    """Distance [px] to the nearest depth discontinuity/missing pixel and the gap to the nearest
    front surface in a (2*window+1)^2 neighbourhood (``z_obs - min z``; > 0 means something nearer
    the camera is close by in the image: a potential occluder)."""
    from scipy.ndimage import distance_transform_edt, minimum_filter
    d = np.asarray(depth, float).copy()
    bad = ~np.isfinite(d) | (d <= 0)
    d[bad] = np.nan
    tol = np.maximum(abs_m, rel * np.nan_to_num(d, nan=0.0))
    disc = bad.copy()
    for axis in (0, 1):
        diff = np.abs(np.diff(d, axis=axis))
        jump = ~(diff <= np.minimum(np.take(tol, range(diff.shape[axis]), axis=axis),
                                    np.take(tol, range(1, diff.shape[axis] + 1), axis=axis)))
        if axis == 0:
            disc[:-1] |= jump
            disc[1:] |= jump
        else:
            disc[:, :-1] |= jump
            disc[:, 1:] |= jump
    dist = distance_transform_edt(~disc)
    dmin = minimum_filter(np.where(bad, np.inf, d), size=2 * window + 1)
    H, W = d.shape
    px = np.clip(np.rint(uv).astype(int), 0, [W - 1, H - 1])
    zc = d[px[:, 1], px[:, 0]]
    gap = zc - dmin[px[:, 1], px[:, 0]]
    return dist[px[:, 1], px[:, 0]], np.where(np.isfinite(gap), gap, np.nan)


def load_depth_cues(scene, obs: TrackObservations, depth_dir="depth_npy") -> dict:
    """Per-observation depth-discontinuity distance and front-surface gap from the raw depth maps."""
    scene = Path(scene)
    dist = np.full(obs.n_obs, np.nan)
    gap = np.full(obs.n_obs, np.nan)
    for ci, cid in enumerate(obs.camera_ids):
        for t, f in enumerate(obs.frame_map.rgbd_frame_index):
            m = (obs.cam == ci) & (obs.time == t)
            if not m.any():
                continue
            p = scene / depth_dir / cid / f"frame_{int(f):06d}.npy"
            if not p.is_file():
                raise FileNotFoundError(f"missing depth map {p}; refusing to audit with partial evidence")
            dd, gg = depth_discontinuity_features(np.load(p), obs.uv[m])
            dist[m], gap[m] = dd, gg
    return dict(dist_px=dist, front_gap_m=gap, source=f"{scene}/{depth_dir}")


def candidate_independent_patches(obs: TrackObservations, voxel_m=0.05, cell_px=64):
    """Patch ids that do not depend on any hypothesis: 3D voxel of a track's first measured point
    (``-1`` if the track never has depth) and the image cell of its query pixel."""
    first3d = np.full((obs.n_tracks, 3), np.nan)
    h = np.flatnonzero(obs.has_3d)
    order = h[np.lexsort((obs.time[h], obs.track[h]))]
    tr, idx = np.unique(obs.track[order], return_index=True)
    first3d[tr] = obs.x_w[order[idx]]
    vox = np.full(obs.n_tracks, -1, int)
    ok = np.isfinite(first3d).all(1)
    key = np.floor(first3d[ok] / voxel_m).astype(int)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    vox[ok] = inv.ravel()
    q_obs = obs.time == obs.track_query_time[obs.track]
    cell = np.full(obs.n_tracks, -1, int)
    qi = np.flatnonzero(q_obs)
    cid = (obs.cam[qi] * 10000 + (obs.uv[qi, 1] // cell_px).astype(int) * 100 + (obs.uv[qi, 0] // cell_px).astype(int))
    cell[obs.track[qi]] = cid
    return vox, cell, first3d


# --------------------------------------------------------------------------- evaluation of one frozen joint
def _split_ll(res, sigma_px, sz, noise):
    ok = np.isfinite(res["r_uv"])
    ll_uv = np.where(ok, asg.student_t_logpdf(np.where(ok, res["r_uv"], 0), noise.student_t_dof) - np.log(sigma_px), 0).sum(1)
    okz = np.isfinite(res["r_z"])
    ll_z = np.where(okz, asg.student_t_logpdf(np.where(okz, res["r_z"], 0), noise.student_t_dof) - np.log(sz), 0)
    return ll_uv, ll_z


@dataclass
class FrozenEvaluation:
    joint: FrozenJoint
    anchors_mode: str              # 'rgbd' | '2d'
    anchors: np.ndarray
    r_uv_px: np.ndarray            # [M,2]
    r_z_m: np.ndarray              # [M] predicted - measured (NaN when not depth-scored)
    uv_pred: np.ndarray
    z_pred: np.ndarray
    nll_uv: np.ndarray             # [M] per-observation Student-t NLL (whitened, with log sigma)
    nll_z: np.ndarray
    track_ll: np.ndarray           # [N]
    posterior: np.ndarray          # [N,3] static / moving / outlier, fixed priors
    extra: dict = field(default_factory=dict)


class AuditProblem:
    """Candidate-independent pieces: eligibility, noise model, static null, depth usability."""

    def __init__(self, obs: TrackObservations, calibs: dict, noise, priors=(0.65, 0.30, 0.05), min_obs=3):
        self.obs, self.calibs, self.noise = obs, calibs, noise
        self.K, self.T_CW = camera_stacks(obs, calibs)
        n2d, n3d = obs.per_track_counts()
        self.eligible = (n3d >= 1) & (n2d >= min_obs)
        self.sz = depth_sigma(np.nan_to_num(obs.z, nan=1.0), noise, obs.depth_edge, obs.depth_cos)
        self.depth_scored = obs.depth_valid & obs.mask_valid & np.isfinite(obs.z) & self.eligible[obs.track]
        self.priors = np.asarray(priors, float)
        self.reason = np.full(obs.n_obs, "used", dtype=object)
        self.reason[~obs.uv_valid] = "uv_invalid"
        nd = obs.uv_valid & ~self.depth_scored
        self.reason[nd & ~np.isfinite(obs.z)] = "depth_invalid"
        self.reason[nd & np.isfinite(obs.z) & ~obs.depth_valid & obs.depth_edge] = "depth_edge_rejected"
        self.reason[nd & np.isfinite(obs.z) & ~obs.depth_valid & ~obs.depth_edge] = "depth_invalid"
        self.reason[nd & obs.depth_valid & ~obs.mask_valid] = "outside_object_mask_depth_unused"
        self.reason[~self.eligible[obs.track]] = "ineligible_track"
        X0 = self._first_points()
        self.first_points = X0
        self.static = self.evaluate(FrozenJoint("STATIC_null", "static", None, None, np.zeros(obs.frame_map.T), "null",
                                                "static null"), "rgbd", posterior=False)

    def _first_points(self):
        obs = self.obs
        X = np.full((obs.n_tracks, 3), np.nan)
        tf = np.full(obs.n_tracks, -1)
        h = np.flatnonzero(obs.has_3d)
        order = h[np.lexsort((obs.time[h], obs.track[h]))]
        tr, i = np.unique(obs.track[order], return_index=True)
        X[tr] = obs.x_w[order[i]]
        tf[tr] = obs.time[order[i]]
        self.first_time = tf
        return X

    def evaluate(self, j: FrozenJoint, anchors_mode="rgbd", posterior=True, iters=10) -> FrozenEvaluation:
        obs = self.obs
        X0 = self.first_points.copy()
        ok = np.isfinite(X0).all(1)
        if j.kind != "static":
            X0[ok] = apply_joint_inverse(j.kind, j.axis, j.origin, j.q[self.first_time[ok]], X0[ok])
        X = refine_anchors(j.kind, j.axis, j.origin, j.q, X0, obs, self.K, self.T_CW, self.noise, self.eligible,
                           iters=iters, use_depth=(anchors_mode == "rgbd"))
        ok_obs = self.eligible[obs.track]
        res = whitened_residuals(j.kind, j.axis, j.origin, j.q, np.nan_to_num(X), obs, self.K, self.T_CW, self.noise,
                                 obs_mask=ok_obs, use_depth=True)
        res["r_z"] = np.where(self.depth_scored, res["r_z"], np.nan)
        ll_uv, ll_z = _split_ll(res, self.noise.sigma_px, self.sz, self.noise)
        tll = asg.track_sum(obs, ll_uv + ll_z, ok_obs)
        post = None
        if posterior:
            ll_o = asg.track_sum(obs, asg.outlier_loglik(obs, self.noise, True), ok_obs)
            post = asg.responsibilities(self.static.track_ll, tll, ll_o, self.priors, self.eligible)
        r_uv_px = res["r_uv"] * self.noise.sigma_px
        r_z_m = res["r_z"] * self.sz
        return FrozenEvaluation(j, anchors_mode, X, r_uv_px, r_z_m, res["uv_pred"], res["z_pred"], -ll_uv, -ll_z, tll,
                                post)


# --------------------------------------------------------------------------- classification and attribution
def classify_depth(ev: FrozenEvaluation, prob: AuditProblem, cues: dict | None, patch_id, time_block=5,
                   tool_dist_m=None, k_sigma=3.0, edge_px=3.0, front_gap_m=0.02, tool_margin_m=0.05):
    """Per-observation depth class relative to one frozen hypothesis (rules in DEPTH_CLASSES order).

    * ``robot_contamination``: measured point within ``tool_margin_m`` of the measured tool segment.
    * ``wrong_surface_or_occlusion``: residual > k sigma and the pixel is within ``edge_px`` of a
      depth discontinuity, or a surface ``front_gap_m`` nearer the camera is in its neighbourhood.
    * ``bias_valid_surface``: residual > k sigma, but within k sigma of its (camera, patch,
      time-block) median, which itself is > k sigma: a shared systematic offset.
    * ``tracking_2d_error``: 2D residual > k sigma_px while depth is consistent.
    """
    obs = prob.obs
    s = prob.sz
    r = ev.r_z_m
    scored = np.isfinite(r)
    big = scored & (np.abs(r) > k_sigma * s)
    cls = np.full(obs.n_obs, "not_depth_scored", dtype=object)
    cls[scored] = "valid_surface_random"
    uvbig = np.isfinite(ev.r_uv_px).all(1) & (np.linalg.norm(ev.r_uv_px, axis=1) > k_sigma * prob.noise.sigma_px)
    cls[scored & ~big & uvbig] = "tracking_2d_error"
    # shared offsets per (camera, patch, time block)
    grp = (obs.cam * 10 ** 7 + np.maximum(patch_id[obs.track], 0) * 100 + obs.time // time_block)
    med = np.full(obs.n_obs, np.nan)
    if big.any() or scored.any():
        g_sc = grp[scored]
        uniq, inv = np.unique(g_sc, return_inverse=True)
        meds = np.array([np.median(r[scored][inv == k]) for k in range(len(uniq))]) if len(uniq) < 20000 else None
        if meds is not None:
            med[scored] = meds[inv]
    bias = big & np.isfinite(med) & (np.abs(med) > k_sigma * s) & (np.abs(r - med) <= k_sigma * s)
    cls[bias] = "bias_valid_surface"
    if cues is not None:
        near = (cues["dist_px"] <= edge_px) | (cues["front_gap_m"] > front_gap_m)
        cls[big & ~bias & near] = "wrong_surface_or_occlusion"
    cls[big & (cls == "valid_surface_random")] = "unexplained_outlier"
    if tool_dist_m is not None:
        cls[scored & (tool_dist_m < tool_margin_m) & big] = "robot_contamination"
    return cls, med


def tool_distance(obs: TrackObservations, p_tcp_w, p_ee_w, extend_m=0.10):
    """3D distance of each measured point to the measured tool segment (TCP -> EE + extension)."""
    from .observations import _segment_distance
    p_tcp_w, p_ee_w = np.asarray(p_tcp_w, float), np.asarray(p_ee_w, float)
    d = p_ee_w - p_tcp_w
    far = p_ee_w + extend_m * d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    out = np.full(obs.n_obs, np.inf)
    h = np.isfinite(obs.x_w).all(1)
    out[h] = _segment_distance(obs.x_w[h], p_tcp_w[obs.time[h]], far[obs.time[h]])
    return out


def _agg(keys, values, mask):
    out = {}
    if not mask.any():
        return out
    k = keys[mask]
    v = values[mask]
    for u in np.unique(k):
        out[int(u) if np.issubdtype(type(u), np.integer) else str(u)] = float(v[k == u].sum())
    return out


def pair_attribution(a: FrozenEvaluation, b: FrozenEvaluation, prob: AuditProblem, patch_id, support="union",
                     top_k=12) -> dict:
    """Objective gap NLL(a) - NLL(b) (positive: b explains the data better) on a common support.

    ``support``: 'union' (tracks moving under a or b), 'intersection', or 'all' (every eligible
    track, moving likelihood for both). Reports weighted (Student-t NLL) and unweighted (sum of
    squared physical residuals) gaps split into 2D and depth, by camera, time and patch."""
    obs = prob.obs
    ma = np.nan_to_num(a.posterior[:, 1]) > 0.5
    mb = np.nan_to_num(b.posterior[:, 1]) > 0.5
    S = {"union": ma | mb, "intersection": ma & mb, "all": prob.eligible.copy()}[support] & prob.eligible
    row = S[obs.track]
    d_uv = a.nll_uv - b.nll_uv
    d_z = np.where(np.isfinite(a.r_z_m) & np.isfinite(b.r_z_m), a.nll_z - b.nll_z, 0.0)
    dz_ok = row & np.isfinite(a.r_z_m) & np.isfinite(b.r_z_m)
    uv_ok = row & np.isfinite(a.r_uv_px).all(1) & np.isfinite(b.r_uv_px).all(1)
    sq_uv = np.where(uv_ok, np.sum(a.r_uv_px ** 2, 1) - np.sum(b.r_uv_px ** 2, 1), 0.0)
    sq_z = np.where(dz_ok, a.r_z_m ** 2 - b.r_z_m ** 2, 0.0)
    out = dict(a=a.joint.name, b=b.joint.name, anchors_mode=a.anchors_mode, support=support,
               n_tracks=int(S.sum()), n_obs_uv=int(uv_ok.sum()), n_obs_depth=int(dz_ok.sum()),
               weighted_gap_uv=float(d_uv[uv_ok].sum()), weighted_gap_depth=float(d_z[dz_ok].sum()),
               unweighted_sq_gap_uv_px2=float(sq_uv.sum()), unweighted_sq_gap_depth_m2=float(sq_z.sum()),
               convention="positive gap = hypothesis b explains these observations better than a")
    out["weighted_gap_total"] = out["weighted_gap_uv"] + out["weighted_gap_depth"]
    for name, key in (("by_camera", obs.cam), ("by_time", obs.time)):
        out[name] = dict(uv=_agg(key, d_uv, uv_ok), depth=_agg(key, d_z, dz_ok))
    if "by_camera" in out:
        out["by_camera"] = {prob.obs.camera_ids[int(k)]: dict(uv=out["by_camera"]["uv"].get(k, 0.0),
                                                                depth=out["by_camera"]["depth"].get(k, 0.0))
                            for k in set(out["by_camera"]["uv"]) | set(out["by_camera"]["depth"])}
    pid = patch_id[obs.track]
    pz = _agg(pid, d_z, dz_ok & (pid >= 0))
    puv = _agg(pid, d_uv, uv_ok & (pid >= 0))
    tot_depth = sum(pz.values())
    order = sorted(pz, key=lambda k: pz[k])  # most negative first: patches where a beats b in depth
    out["patches_where_depth_favours_a"] = [dict(patch=k, depth_gap=pz[k], uv_gap=puv.get(k, 0.0)) for k in order[:top_k]]
    out["patches_where_depth_favours_b"] = [dict(patch=k, depth_gap=pz[k], uv_gap=puv.get(k, 0.0))
                                           for k in order[::-1][:top_k]]
    neg = sorted([v for v in pz.values() if v < 0])
    out["depth_gap_concentration"] = dict(
        n_patches=len(pz), total=tot_depth,
        share_of_negative_from_top5=(float(sum(neg[:5]) / sum(neg)) if neg and sum(neg) != 0 else None))
    return out


def systematic_report(ev: FrozenEvaluation, prob: AuditProblem, patch_id, cues=None, tool_dist=None, k_sigma=3.0):
    """Stage A2: static-core and moving-support residual statistics under one hypothesis."""
    obs = prob.obs
    post = ev.posterior if ev.posterior is not None else np.tile([1.0, 0, 0], (obs.n_tracks, 1))
    groups = dict(static_core=np.nan_to_num(post[:, 0]) > 0.9, moving=np.nan_to_num(post[:, 1]) > 0.5)
    cls, _ = classify_depth(ev, prob, cues, patch_id, tool_dist_m=tool_dist, k_sigma=k_sigma)
    out = dict(hypothesis=ev.joint.name, anchors_mode=ev.anchors_mode)
    for gname, gm in groups.items():
        per_cam = {}
        for ci, cid in enumerate(obs.camera_ids):
            m = gm[obs.track] & (obs.cam == ci)
            rz = ev.r_z_m[m]
            okz = np.isfinite(rz)
            ruv = ev.r_uv_px[m]
            okuv = np.isfinite(ruv).all(1)
            # lag-1 autocorrelation of the signed depth residual along each track
            ac = _lag1_autocorr(obs.track[m][okz], obs.time[m][okz], rz[okz])
            # temporal drift of 2D residual: median |r_uv| early vs late third
            t = obs.time[m][okuv]
            nuv = np.linalg.norm(ruv[okuv], axis=1)
            T = obs.frame_map.T
            early, late = nuv[t < T / 3], nuv[t >= 2 * T / 3]
            entry = dict(n_tracks=int((gm & (obs.track_camera == ci)).sum()), n_depth=int(okz.sum()),
                         depth_median_signed_mm=_q(rz[okz], 50, 1000), depth_p10_mm=_q(rz[okz], 10, 1000),
                         depth_p90_mm=_q(rz[okz], 90, 1000), depth_mad_mm=_mad(rz[okz], 1000),
                         depth_trimmed_mean_mm=_trim(rz[okz], 1000),
                         depth_frac_beyond_k_sigma=float(np.mean(np.abs(rz[okz]) > k_sigma * prob.sz[m][okz]))
                         if okz.any() else None,
                         depth_lag1_autocorr=ac, uv_median_px=_q(nuv, 50), uv_median_px_early=_q(early, 50),
                         uv_median_px_late=_q(late, 50),
                         median_incidence_cos=_q(obs.depth_cos[m][okz & np.isfinite(obs.depth_cos[m])], 50),
                         depth_class_counts={c: int((cls[m] == c).sum()) for c in DEPTH_CLASSES if (cls[m] == c).any()})
            if cues is not None:
                entry["median_dist_to_discontinuity_px"] = _q(cues["dist_px"][m][okz], 50)
            per_cam[cid] = entry
        out[gname] = per_cam
    # cross-camera disagreement: patches seen (with depth) by >1 camera
    pid = patch_id[obs.track]
    mv = groups["moving"][obs.track] & np.isfinite(ev.r_z_m) & (pid >= 0)
    dis = []
    for p in np.unique(pid[mv]):
        meds = [np.median(ev.r_z_m[mv & (pid == p) & (obs.cam == ci)]) for ci in range(len(obs.camera_ids))
                if (mv & (pid == p) & (obs.cam == ci)).sum() >= 10]
        if len(meds) >= 2:
            dis.append(max(meds) - min(meds))
    out["moving_patch_cross_camera_median_disagreement_mm"] = _q(np.array(dis), 50, 1000) if dis else None
    out["n_patches_seen_by_multiple_cameras"] = len(dis)
    return out, cls


def _q(x, p, scale=1.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p) * scale) if len(x) else None


def _mad(x, scale=1.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))) * scale) if len(x) else None


def _trim(x, scale=1.0, frac=0.1):
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    k = int(len(x) * frac)
    return float(x[k:len(x) - k].mean() * scale) if len(x) > 2 * k else None


def _lag1_autocorr(track, time, r):
    if len(r) < 10:
        return None
    o = np.lexsort((time, track))
    tr, t, rr = track[o], time[o], r[o]
    pair = (tr[1:] == tr[:-1]) & (t[1:] == t[:-1] + 1)
    if pair.sum() < 10:
        return None
    a, b = rr[:-1][pair], rr[1:][pair]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else None


# --------------------------------------------------------------------------- export
def export_observation_table(path, ev: FrozenEvaluation, prob: AuditProblem, patch_id, cell_id, cues, cls):
    obs = prob.obs
    rd = np.where(np.isfinite(ev.r_z_m), np.where(ev.r_z_m > 0, "measured_nearer_than_predicted",
                                                    "measured_farther_than_predicted"), "none")
    np.savez_compressed(
        path, hypothesis=ev.joint.name, anchors_mode=ev.anchors_mode, camera=np.array(obs.camera_ids)[obs.cam],
        time=obs.time, track_id=obs.track, track_local_id=obs.track_local_id[obs.track],
        r_uv_px=ev.r_uv_px, r_z_m=ev.r_z_m, sigma_z_m=prob.sz, depth_edge=obs.depth_edge, incidence_cos=obs.depth_cos,
        depth_valid=obs.depth_valid, mask_valid=obs.mask_valid,
        moving_posterior=ev.posterior[obs.track, 1] if ev.posterior is not None else np.full(obs.n_obs, np.nan),
        original_pixel=obs.uv, predicted_pixel=ev.uv_pred, measured_z=obs.z, predicted_z=ev.z_pred,
        patch_id=patch_id[obs.track], image_cell_id=cell_id[obs.track],
        dist_to_depth_discontinuity_px=cues["dist_px"] if cues else np.full(obs.n_obs, np.nan),
        front_surface_gap_m=cues["front_gap_m"] if cues else np.full(obs.n_obs, np.nan),
        valid_measured_depth=prob.depth_scored, residual_direction=rd.astype(str), depth_class=cls.astype(str),
        reason=prob.reason.astype(str), nll_uv=ev.nll_uv, nll_z=ev.nll_z,
        conventions=json.dumps(dict(r_z_m="predicted_z - measured_z (camera-Z, m)",
                                    r_uv_px="predicted - measured pixel",
                                    nll="Student-t negative log-likelihood per observation, with log sigma")))


def plot_heatmaps(out_dir, scene, prob: AuditProblem, a: FrozenEvaluation, b: FrozenEvaluation, times=None):
    """Depth-term gap NLL(a) - NLL(b) per observation drawn on raw RGB and depth (blue: depth
    favours a; red: depth favours b)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    obs = prob.obs
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gap = np.where(np.isfinite(a.r_z_m) & np.isfinite(b.r_z_m), a.nll_z - b.nll_z, np.nan)
    sup = (np.nan_to_num(a.posterior[:, 1]) > 0.5) | (np.nan_to_num(b.posterior[:, 1]) > 0.5)
    T = obs.frame_map.T
    times = times or [T // 2, T - 1]
    files = []
    for ci, cid in enumerate(obs.camera_ids):
        for t in times:
            f = int(obs.frame_map.rgbd_frame_index[t])
            rgb = cv2.imread(str(Path(scene) / "rgb" / cid / f"frame_{f:06d}.png"))
            dep = np.load(Path(scene) / "depth_npy" / cid / f"frame_{f:06d}.npy")
            m = (obs.cam == ci) & (obs.time == t) & np.isfinite(gap) & sup[obs.track]
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            ax[0].imshow(rgb[:, :, ::-1])
            ax[1].imshow(np.where(dep > 0, dep, np.nan), cmap="gray")
            lim = np.nanpercentile(np.abs(gap[m]), 95) if m.any() else 1.0
            for k in range(2):
                sc = ax[k].scatter(obs.uv[m, 0], obs.uv[m, 1], c=gap[m], cmap="coolwarm", vmin=-lim, vmax=lim, s=6)
                ax[k].set_axis_off()
            fig.colorbar(sc, ax=ax, fraction=0.02, label=f"depth NLL({a.joint.name}) - NLL({b.joint.name})")
            ax[0].set_title(f"{cid} frame {f} (t={t}): blue = depth favours {a.joint.name}, red = favours {b.joint.name}")
            ax[1].set_title("raw depth_npy")
            p = out_dir / f"depth_gap_{a.joint.name}_vs_{b.joint.name}_{cid}_t{t:02d}.png"
            fig.savefig(p, dpi=80, bbox_inches="tight")
            plt.close(fig)
            files.append(str(p))
    return files


def run_depth_audit(obs: TrackObservations, calibs: dict, noise, joints: dict, out_dir, scene=None, pairs=None,
                    tcp=None, heatmap_pairs=None) -> dict:
    """Evaluate every frozen joint on identical observations; write tables, attribution and report."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    prob = AuditProblem(obs, calibs, noise)
    patch_id, cell_id, _ = candidate_independent_patches(obs)
    cues = load_depth_cues(scene, obs) if scene is not None else None
    tdist = tool_distance(obs, tcp["p_tcp_w"], tcp["p_ee_w"]) if tcp is not None else None
    evs = {}
    for name, j in joints.items():
        for mode in ("rgbd", "2d"):
            evs[(name, mode)] = prob.evaluate(j, mode)
    summary = dict(joints={n: j.describe() for n, j in joints.items()},
                   n_obs=int(obs.n_obs), n_tracks=int(obs.n_tracks), cameras=list(obs.camera_ids),
                   reason_counts={r: int((prob.reason == r).sum()) for r in REASONS},
                   noise=dict(sigma_px=noise.sigma_px, sigma_depth_m0=noise.sigma_depth_m0,
                              sigma_depth_m2=noise.sigma_depth_m2, incidence_model=noise.incidence_model,
                              edge_depth_inflation=noise.edge_depth_inflation),
                   method="frozen joints; per-track anchors re-estimated with the joint fixed (same GN for all); "
                          "no joint refit", hypotheses={}, pairs=[], systematic={})
    static = prob.static
    for (name, mode), ev in evs.items():
        nll = -ev.track_ll[prob.eligible].sum()
        mv = np.nan_to_num(ev.posterior[:, 1]) > 0.5
        summary["hypotheses"][f"{name}|{mode}"] = dict(
            role=ev.joint.role, moving_tracks=int(mv.sum()),
            moving_tracks_per_camera={cid: int((mv & (obs.track_camera == ci)).sum()) for ci, cid in enumerate(obs.camera_ids)},
            moving_nll_all_eligible=float(nll),
            mixture_nll=float(asg.mixture_nll(static.track_ll, ev.track_ll,
                                              asg.track_sum(obs, asg.outlier_loglik(obs, noise, True), prob.eligible[obs.track]),
                                              prob.priors, prob.eligible).sum()),
            moving_depth_nll=float(ev.nll_z[mv[obs.track] & np.isfinite(ev.r_z_m)].sum()),
            moving_uv_nll=float(ev.nll_uv[mv[obs.track] & np.isfinite(ev.r_uv_px).all(1)].sum()))
    for a, b in (pairs or []):
        for mode in ("rgbd", "2d"):
            for support in ("union", "all"):
                summary["pairs"].append(pair_attribution(evs[(a, mode)], evs[(b, mode)], prob, patch_id, support))
    for (name, mode), ev in evs.items():
        rep, cls = systematic_report(ev, prob, patch_id, cues, tdist)
        summary["systematic"][f"{name}|{mode}"] = rep
        export_observation_table(out / f"observations_{name}_{mode}.npz", ev, prob, patch_id, cell_id,
                                 cues or {"dist_px": np.full(obs.n_obs, np.nan), "front_gap_m": np.full(obs.n_obs, np.nan)},
                                 cls)
    # static-core camera bias: consensus static core = static (p > 0.9) under EVERY evaluated
    # RGB-D hypothesis, evaluated with the static null's anchors (no single candidate defines it)
    cons = prob.eligible.copy()
    for (name, mode), ev in evs.items():
        if mode == "rgbd" and ev.joint.kind != "static":   # the null against itself is uninformative
            cons &= np.nan_to_num(ev.posterior[:, 0]) > 0.9
    summary["consensus_static_core_tracks"] = int(cons.sum())
    st_rep, _ = systematic_report(_with_posterior(static, prob, cons), prob, patch_id, cues, tdist)
    summary["static_null_core"] = st_rep
    if scene is not None and heatmap_pairs:
        summary["heatmaps"] = []
        for a, b in heatmap_pairs:
            summary["heatmaps"] += plot_heatmaps(out / "heatmaps", scene, prob, evs[(a, "rgbd")], evs[(b, "rgbd")])
    # geometry comparison between frozen joints (diagnostic)
    geo = {}
    names = list(joints)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ja, jb = joints[a], joints[b]
            if ja.kind == "static" or jb.kind == "static":
                continue
            e = dict(axis_angle_deg=axis_angle_deg(ja.axis, jb.axis))
            if ja.kind == jb.kind == "revolute":
                e["hinge_line_distance_m"] = line_to_line_distance(ja.axis, ja.origin, jb.axis, jb.origin)
            geo[f"{a}~{b}"] = e
    summary["joint_geometry_comparison"] = geo
    from .exports import write_json
    write_json(out / "depth_conflict_summary.json", summary)
    (out / "depth_conflict_report.md").write_text(render_report(summary))
    return summary


def _with_posterior(ev, prob, static_core):
    e = copy.copy(ev)
    e.posterior = np.where(static_core[:, None], np.array([1.0, 0.0, 0.0])[None],
                           np.where(prob.eligible[:, None], np.array([0.0, 0.0, 1.0])[None], np.nan))
    return e


def render_report(s: dict) -> str:
    L = ["# Stage A depth-conflict audit (frozen hypotheses, identical observations)", "",
         f"Method: {s['method']}.", "",
         "Reference hypotheses (role `reference_posthoc`) are diagnostics only.", "",
         "## Hypotheses", "", "| name | role | kind | span | axis | origin |", "|---|---|---|---|---|---|"]
    for n, j in s["joints"].items():
        L.append(f"| {n} | {j['role']} | {j['kind']} | {j.get('q_span_readable', '-')} | "
                 f"{np.round(j.get('axis'), 3).tolist() if j.get('axis') is not None else '-'} | "
                 f"{np.round(j['origin'], 3).tolist() if j.get('origin') is not None else '-'} |")
    L += ["", "## Likelihood per hypothesis (anchors re-estimated; joint frozen)", "",
          "| hypothesis / anchors | moving tracks | mixture NLL | moving depth NLL | moving 2D NLL |", "|---|---|---|---|---|"]
    for k, v in s["hypotheses"].items():
        L.append(f"| {k} | {v['moving_tracks']} {v['moving_tracks_per_camera']} | {v['mixture_nll']:.1f} | "
                 f"{v['moving_depth_nll']:.1f} | {v['moving_uv_nll']:.1f} |")
    L += ["", "## Objective-gap attribution (positive = b explains better)", "",
          "| a | b | anchors | support | tracks | 2D gap | depth gap | total | sq gap 2D [px²] | sq gap depth [m²] | per camera (2D / depth) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in s["pairs"]:
        pc = "; ".join(f"{c}: {v['uv']:.0f} / {v['depth']:.0f}" for c, v in sorted(p["by_camera"].items()))
        L.append(f"| {p['a']} | {p['b']} | {p['anchors_mode']} | {p['support']} | {p['n_tracks']} | "
                 f"{p['weighted_gap_uv']:.1f} | {p['weighted_gap_depth']:.1f} | {p['weighted_gap_total']:.1f} | "
                 f"{p['unweighted_sq_gap_uv_px2']:.0f} | {p['unweighted_sq_gap_depth_m2']:.4f} | {pc} |")
    L += ["", "## Static-null core (candidate independent)", "", "```",
          json.dumps(s["static_null_core"], indent=1), "```", "",
          "## Systematic-error statistics per hypothesis", "", "```",
          json.dumps(s["systematic"], indent=1)[:60000], "```"]
    return "\n".join(L) + "\n"
