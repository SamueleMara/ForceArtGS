"""RGB material-track recovery guided by the current R AND P predictions (multimodal spec 5.2).

Raw ``TrackObservations`` are never modified: recovered observations are appended to a copy and flagged in a
``recovered`` mask (provenance). For a moving track lacking an observation at a TRAINING time, the canonical anchor of
each hypothesis is moved by that hypothesis' A_H(q_t) and projected; the NCC search window covers BOTH predictions plus
calibration/tracking uncertainty (never a universal radius, never biased to one hypothesis). The template is the
track's own patch at its nearest observed time. A match is accepted only with NCC >= ``ncc_min``, a unique peak, inside
the object mask. By default a recovered observation is 2D ONLY: its depth association is not independently verified
(a ~1 cm depth tolerance is far looser than the sensor noise and corrupts anchors); ``attach_depth=True`` attaches depth
only off depth edges and within tolerance of SOME hypothesis' predicted depth. Held-out times are never touched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .observations import OBS_FIELDS, TrackObservations
from .rigid_kinematics import apply_joint


class SceneImageSource:
    """rgb/<cam>/frame_XXXXXX.png, depth_npy/<cam>/frame_XXXXXX.npy, mask/<cam>/frame_XXXXXX.png (lazy, cached)."""

    def __init__(self, scene_root, frame_map, max_cache=64):
        self.root = Path(scene_root)
        self.fm = frame_map
        self.cache = {}
        self.max_cache = max_cache

    def _get(self, kind, cam, t):
        key = (kind, cam, t)
        if key not in self.cache:
            import cv2
            stem = f"frame_{int(self.fm.rgbd_frame_index[t]):06d}"
            if kind == "gray":
                im = cv2.imread(str(self.root / "rgb" / cam / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
                val = None if im is None else im.astype(np.float32)
            elif kind == "depth":
                f = self.root / "depth_npy" / cam / f"{stem}.npy"
                val = np.load(f).astype(np.float64) if f.is_file() else None
            else:
                im = cv2.imread(str(self.root / "mask" / cam / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
                val = None if im is None else im > 127
            if len(self.cache) >= self.max_cache:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = val
        return self.cache[key]

    def gray(self, cam, t):
        return self._get("gray", cam, t)

    def depth(self, cam, t):
        return self._get("depth", cam, t)

    def mask(self, cam, t):
        return self._get("mask", cam, t)


def _ncc_search(img, tmpl, cu, cv, radius):
    import cv2
    h, w = tmpl.shape
    hh, hw = h // 2, w // 2
    x0, y0 = int(round(cu)) - radius - hw, int(round(cv)) - radius - hh
    x1, y1 = int(round(cu)) + radius + hw + 1, int(round(cv)) + radius + hh + 1
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        return None
    win = img[y0:y1, x0:x1]
    if win.shape[0] < h or win.shape[1] < w or tmpl.std() < 2.0:
        return None
    res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
    _, best, _, loc = cv2.minMaxLoc(res)
    r2 = res.copy()
    yy, xx = loc[1], loc[0]
    r2[max(yy - 3, 0):yy + 4, max(xx - 3, 0):xx + 4] = -1
    second = float(r2.max()) if r2.size else -1
    return float(best), second, x0 + loc[0] + hw, y0 + loc[1] + hh


def recover_tracks(obs: TrackObservations, images, calibs, cands: dict, moving_tracks, train_times, *, ncc_min=0.85,
                   patch=11, sigma_px=2.0, max_radius_px=25, max_recover=4000, uniqueness=0.05,
                   depth_tol_m=0.015, attach_depth=False) -> tuple:
    """Returns (augmented observations, recovered mask [M_new], report)."""
    T = obs.frame_map.T
    train = np.zeros(T, bool)
    train[np.asarray(train_times, int)] = True
    has = np.zeros((T, obs.n_tracks), bool)
    has[obs.time[obs.uv_valid], obs.track[obs.uv_valid]] = True
    obs_idx = {}
    for k in np.flatnonzero(obs.uv_valid):
        obs_idx.setdefault(int(obs.track[k]), []).append(k)
    new = {f: [] for f in OBS_FIELDS}
    rep = dict(attempted=0, accepted=0, rejected=dict(no_prediction=0, no_template=0, out_of_view=0, low_ncc=0,
                                                      not_unique=0, outside_mask=0), per_camera={})
    hp = patch // 2
    for i in np.flatnonzero(np.asarray(moving_tracks, bool)):
        if rep["accepted"] >= max_recover:
            break
        ci = int(obs.track_camera[i])
        cid = obs.camera_ids[ci]
        cal = calibs[cid]
        T_CW = np.linalg.inv(cal.T_WC)
        ks = obs_idx.get(int(i), [])
        if not ks:
            continue
        t_obs = obs.time[ks]
        for t in np.flatnonzero(train & ~has[:, i]):
            rep["attempted"] += 1
            preds, zpred = [], []
            for kind, c in cands.items():
                X = c.canonical_anchors_w[i]
                if not np.isfinite(X).all() or not np.isfinite(c.q[t]):
                    continue
                xw = apply_joint(kind, c.axis_w, c.origin_w, np.array([c.q[t]]), X[None])[0]
                xc = T_CW[:3, :3] @ xw + T_CW[:3, 3]
                if xc[2] > 0.05:
                    zpred.append(xc[2])
                    preds.append(np.array([cal.K[0, 0] * xc[0] / xc[2] + cal.K[0, 2], cal.K[1, 1] * xc[1] / xc[2] + cal.K[1, 2]]))
            if not preds:
                rep["rejected"]["no_prediction"] += 1
                continue
            P = np.stack(preds)
            centre = P.mean(0)
            radius = int(min(max_radius_px, np.ceil(np.linalg.norm(P - centre, axis=1).max() + 3 * sigma_px + 2)))
            k0 = ks[int(np.argmin(np.abs(t_obs - t)))]
            img0 = images.gray(cid, int(obs.time[k0]))
            img = images.gray(cid, int(t))
            if img0 is None or img is None:
                rep["rejected"]["no_template"] += 1
                continue
            u0, v0 = obs.uv[k0]
            iu, iv = int(round(u0)), int(round(v0))
            if iu - hp < 0 or iv - hp < 0 or iu + hp + 1 > img0.shape[1] or iv + hp + 1 > img0.shape[0]:
                rep["rejected"]["no_template"] += 1
                continue
            tmpl = img0[iv - hp:iv + hp + 1, iu - hp:iu + hp + 1]
            m = _ncc_search(img, tmpl, centre[0], centre[1], radius)
            if m is None:
                rep["rejected"]["out_of_view"] += 1
                continue
            score, second, u, v = m
            if score < ncc_min:
                rep["rejected"]["low_ncc"] += 1
                continue
            if second > score - uniqueness:
                rep["rejected"]["not_unique"] += 1
                continue
            mk = images.mask(cid, int(t))
            if mk is not None and not mk[int(v), int(u)]:
                rep["rejected"]["outside_mask"] += 1
                continue
            dep = images.depth(cid, int(t))
            z = np.nan
            dvalid, edge = False, False
            if dep is not None and attach_depth:
                zz = dep[int(v), int(u)]
                win = dep[max(int(v) - 1, 0):int(v) + 2, max(int(u) - 1, 0):int(u) + 2]
                edge = bool((win > 0).all() and np.ptp(win) > 0.02)
                # depth only on a consistent surface: within tolerance of SOME hypothesis' predicted depth of this
                # material point (never biased to one candidate); otherwise the observation keeps its 2D match only
                consistent = any(abs(zz - zp) < max(depth_tol_m, 0.01 * zp) for zp in zpred)
                if zz > 0 and not edge and consistent:
                    z, dvalid = float(zz), True
                elif zz > 0 and not edge:
                    rep["depth_rejected_inconsistent"] = rep.get("depth_rejected_inconsistent", 0) + 1
            xw = np.full(3, np.nan)
            if dvalid:
                xc = np.array([(u - cal.K[0, 2]) * z / cal.K[0, 0], (v - cal.K[1, 2]) * z / cal.K[1, 1], z])
                xw = cal.T_WC[:3, :3] @ xc + cal.T_WC[:3, 3]
            for f, val in (("cam", ci), ("track", i), ("time", t), ("uv", [u, v]), ("uv_conf", score),
                           ("uv_valid", True), ("z", z), ("depth_valid", dvalid), ("depth_edge", edge),
                           ("mask_valid", True), ("x_w", xw), ("depth_cos", np.nan)):
                new[f].append(val)
            rep["accepted"] += 1
            pc = rep["per_camera"].setdefault(cid, 0)
            rep["per_camera"][cid] = pc + 1
    rep["matcher_calibration"] = calibrate_matcher(obs, images, calibs, cands, moving_tracks, train_times, ncc_min=ncc_min,
                                                   patch=patch, sigma_px=sigma_px, max_radius_px=max_radius_px,
                                                   uniqueness=uniqueness)
    n_new = rep["accepted"]
    if n_new == 0:
        return obs, np.zeros(obs.n_obs, bool), rep
    kw = {f: np.concatenate([getattr(obs, f), np.asarray(new[f])]) for f in OBS_FIELDS}
    out = TrackObservations(obs.camera_ids, obs.frame_map, track_camera=obs.track_camera,
                            track_local_id=obs.track_local_id, track_query_time=obs.track_query_time,
                            provenance=dict(obs.provenance, track_refinement=f"{n_new} NCC-recovered observations"), **kw)
    recovered = np.r_[np.zeros(obs.n_obs, bool), np.ones(n_new, bool)]
    if np.isin(out.time[recovered], np.flatnonzero(~train)).any():
        raise AssertionError("track recovery touched a held-out time")
    return out, recovered, rep


def calibrate_matcher(obs, images, calibs, cands, moving_tracks, train_times, *, ncc_min=0.85, patch=11, sigma_px=2.0,
                      max_radius_px=25, uniqueness=0.05, n_max=300, seed=0) -> dict:
    """Independent localisation uncertainty of the recovery matcher: re-find EXISTING raw observations of moving tracks
    with exactly the recovery procedure (template from another time of the same track, search window covering both
    hypotheses' predictions) and measure the spread of (match - raw tracker position). Not circular: the raw positions
    were measured by the tracker, not predicted by a candidate."""
    rng = np.random.default_rng(seed)
    train = set(int(t) for t in train_times)
    mv = np.asarray(moving_tracks, bool)
    cand_obs = np.flatnonzero(obs.uv_valid & mv[obs.track] & np.isin(obs.time, list(train)))
    if len(cand_obs) == 0:
        return dict(status="NO_SAMPLES")
    pick = rng.choice(cand_obs, min(n_max, len(cand_obs)), replace=False)
    by_track = {}
    for k in np.flatnonzero(obs.uv_valid):
        by_track.setdefault(int(obs.track[k]), []).append(k)
    hp = patch // 2
    errs = []
    for k in pick:
        i, t = int(obs.track[k]), int(obs.time[k])
        others = [j for j in by_track.get(i, []) if j != k]
        if not others:
            continue
        cid = obs.camera_ids[int(obs.track_camera[i])]
        cal = calibs[cid]
        T_CW = np.linalg.inv(cal.T_WC)
        preds = []
        for kind, c in cands.items():
            X = c.canonical_anchors_w[i]
            if not np.isfinite(X).all():
                continue
            xw = apply_joint(kind, c.axis_w, c.origin_w, np.array([c.q[t]]), X[None])[0]
            xc = T_CW[:3, :3] @ xw + T_CW[:3, 3]
            if xc[2] > 0.05:
                preds.append([cal.K[0, 0] * xc[0] / xc[2] + cal.K[0, 2], cal.K[1, 1] * xc[1] / xc[2] + cal.K[1, 2]])
        if not preds:
            continue
        P = np.asarray(preds)
        centre = P.mean(0)
        radius = int(min(max_radius_px, np.ceil(np.linalg.norm(P - centre, axis=1).max() + 3 * sigma_px + 2)))
        k0 = others[int(np.argmin(np.abs(obs.time[others] - t)))]
        img0, img = images.gray(cid, int(obs.time[k0])), images.gray(cid, t)
        if img0 is None or img is None:
            continue
        iu, iv = int(round(obs.uv[k0, 0])), int(round(obs.uv[k0, 1]))
        if iu - hp < 0 or iv - hp < 0 or iu + hp + 1 > img0.shape[1] or iv + hp + 1 > img0.shape[0]:
            continue
        m = _ncc_search(img, img0[iv - hp:iv + hp + 1, iu - hp:iu + hp + 1], centre[0], centre[1], radius)
        if m is None or m[0] < ncc_min or m[1] > m[0] - uniqueness:
            continue
        errs.append(np.hypot(m[2] - obs.uv[k, 0], m[3] - obs.uv[k, 1]))
    if len(errs) < 10:
        return dict(status="TOO_FEW", n=len(errs))
    e = np.asarray(errs)
    return dict(status="OK", n=int(len(e)), median_err_px=float(np.median(e)), p90_err_px=float(np.percentile(e, 90)),
                sigma_per_axis_px=float(np.median(e) / np.sqrt(2 * np.log(2))), rayleigh_note="per-axis sigma from the "
                "median of the 2D error (Rayleigh); includes the raw tracker noise, so it is conservative")
