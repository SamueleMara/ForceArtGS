"""Candidate-specific dense static / moving / unknown segmentation (end-to-end spec G1, section 3).

For one branch H (a validated ``JointAuthority``) every foreground RGB-D pixel a = (c, t, p) with valid
depth is unprojected, ``x_a = T_WC [z K^-1 p]``, and carried to training target images b = (c', s) at a
different time under the two exact membership hypotheses

    static:  x_{a->b}^S = x_a          moving:  x_{a->b}^M = A_H(s) A_H(t)^-1 x_a .

Each prediction is classified against the target's measured depth with the *same* rules for S and M:
SUPPORTED_VISIBLE (front-visible, same surface: depth + colour residuals), OCCLUDED (a nearer measured
surface: no residual, censored cost), FREE_SPACE_CONTRADICTION (prediction in front of a measured opaque
surface or outside the verified silhouette: penalised) and UNKNOWN (out of image, invalid / mixed depth,
robot-owned pixel). A target is scored only when it is *discriminative* (S and M predictions differ by
>= ``min_disc_px`` pixels or ``min_disc_sigma`` depth sigmas) and *common-observable* (neither prediction
UNKNOWN), so a hypothesis cannot win by making its points invalid. Per-target cost difference
``delta = c_S - c_M`` (> 0 favours moving) is pooled over a local window, combined with a bounded seed prior
from confident discovery tracks, and thresholded with fixed, candidate-independent rules; everything else is
UNKNOWN. Same-time targets never count (no temporal baseline). Held-out images are never targets.

Noise scales are shared across hypotheses: depth sigma from the discovery's static-core calibration, colour
sigma from the MAD of residuals on non-discriminative supported targets (both hypotheses agree there).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .integration_contract import JointAuthority, atomic_write_json, sha256_file

SUPPORTED, OCCLUDED, FREE, UNKNOWN = 0, 1, 2, 3
LABEL_S, LABEL_M, LABEL_U = 0, 1, 2


@dataclass
class DenseConfig:
    work_downscale: int = 2              # evidence resolution = full / work_downscale (labels upsampled nearest)
    max_targets: int = 8                 # temporally spread training targets per source image
    min_time_gap: int = 2                # targets need |s - t| >= this (temporal baseline)
    sigma_depth_m0: float = 0.002        # sigma_D(z) = m0 + m2 z^2 (discovery static-core calibration)
    sigma_depth_m2: float = 0.004
    sigma_px: float = 1.0
    occlusion_k: float = 3.0             # same-surface band = k * sigma_eff
    edge_k: float = 4.0                  # 3x3 depth range > edge_k * sigma -> mixed depth -> UNKNOWN
    min_disc_px: float = 2.0
    min_disc_sigma: float = 3.0
    free_space_cost: float = 4.0         # = Cauchy cost of a ~7-sigma residual, per modality pair
    pool_radius: int = 2                 # (2r+1)^2 window at evidence resolution
    decide_mean_delta: float = 0.5       # |mean delta| over the window needed for a label
    decide_min_targets: int = 6          # discriminative common-observable targets in the window
    seed_prior: float = 1.5              # bounded log-odds added at confident track pixels
    seed_min_prob: float = 0.9
    seed_radius: int = 2
    tool_radius_m: float = 0.03
    tool_extend_m: float = 0.10
    tool_tip_offset_m: float = 0.015
    device: str = "cuda"
    chunk: int = 400_000

    @staticmethod
    def from_dict(d: dict) -> "DenseConfig":
        unknown = set(d) - set(DenseConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown dense config keys {sorted(unknown)}")
        return DenseConfig(**d)


@dataclass
class ImageStack:
    """All selected images of a scene at evidence resolution (torch tensors on ``device``)."""
    cams: list
    times: np.ndarray            # [N] time index of each image
    cam_idx: np.ndarray          # [N]
    rgb: object                  # [N,h,w,3] float
    depth: object                # [N,h,w] metres, 0 = invalid
    obj: object                  # [N,h,w] bool
    ignore: object               # [N,h,w] bool (robot / ownership ambiguity)
    edge: object                 # [N,h,w] bool mixed depth
    grad_rgb: object             # [N,h,w] |grad I| per pixel (colour units / px)
    grad_d: object               # [N,h,w] |grad D| per pixel (m / px)
    K: object                    # [C,3,3] at evidence resolution
    T_WC: object                 # [C,4,4]
    T_CW: object
    full_hw: tuple
    stems: list
    full_obj: list               # full-resolution bool object masks
    full_ignore: list


def _sigma_d(z, cfg):
    return cfg.sigma_depth_m0 + cfg.sigma_depth_m2 * z * z


def tool_ignore_mask(depth, K, T_WC, p_tcp, p_ee, cfg: DenseConfig):
    """Robot pixels: measured depth within ``tool_radius_m`` of the tool segment (TCP+tip offset -> EE+extend),
    plus depth-less pixels inside the projected tool capsule. The surface the tip touches is kept."""
    from .observations import _segment_distance, project
    h, w = depth.shape
    d = p_ee - p_tcp
    dn = d / max(np.linalg.norm(d), 1e-9)
    a, b = p_tcp + cfg.tool_tip_offset_m * dn, p_ee + cfg.tool_extend_m * dn
    v, u = np.mgrid[0:h, 0:w]
    z = depth
    ok = z > 0
    x_c = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)
    x_w = x_c @ T_WC[:3, :3].T + T_WC[:3, 3]
    out = np.zeros((h, w), bool)
    out[ok] = _segment_distance(x_w[ok], a[None], b[None]) < cfg.tool_radius_m
    T_CW = np.linalg.inv(T_WC)
    ab_uv, ab_z = project(np.stack([a, b]), K, T_CW)
    if (ab_z > 0.05).all():
        r_px = K[0, 0] * cfg.tool_radius_m / max(min(ab_z), 0.05)
        P = np.stack([u[~ok], v[~ok], np.zeros((~ok).sum())], -1).astype(float)
        A = np.r_[ab_uv[0], 0.0][None]
        B = np.r_[ab_uv[1], 0.0][None]
        out[~ok] = _segment_distance(P, A, B) < r_px
    return out


def load_image_stack(scene_root, authority: JointAuthority, cameras, cfg: DenseConfig, robot=None) -> ImageStack:
    """Raw ``rgb/``, ``depth_npy/`` and ``mask/`` of every (camera, time) of the authority's frame map."""
    import cv2
    import torch
    scene = Path(scene_root)
    dev = torch.device(cfg.device)
    ds = int(cfg.work_downscale)
    rgbs, deps, objs, igns, edges, times, cidx, stems, f_obj, f_ign = [], [], [], [], [], [], [], [], [], []
    g_rgb, g_d = [], []
    Ks, Ts = [], []
    for ci, cid in enumerate(cameras):
        cam = authority.cameras[cid]
        K = np.asarray(cam.K, float).copy()
        K[:2] /= ds
        K[0, 2] = (cam.K[0, 2] + 0.5) / ds - 0.5
        K[1, 2] = (cam.K[1, 2] + 0.5) / ds - 0.5
        Ks.append(K)
        Ts.append(np.asarray(cam.T_WC, float))
    for t in range(authority.T):
        for ci, cid in enumerate(cameras):
            cam = authority.cameras[cid]
            stem = f"frame_{int(authority.rgbd_frame_index[t]):06d}"
            img = cv2.imread(str(scene / "rgb" / cid / f"{stem}.png"))
            if img is None:
                raise FileNotFoundError(scene / "rgb" / cid / f"{stem}.png")
            img = img[:, :, ::-1]
            if img.shape[:2] != (cam.height, cam.width):
                raise ValueError(f"{cid} {stem}: image {img.shape[:2]} != calibration {(cam.height, cam.width)}")
            dep = np.load(scene / "depth_npy" / cid / f"{stem}.npy").astype(np.float64)
            if dep.shape != img.shape[:2]:
                raise ValueError(f"{cid} {stem}: depth/image size mismatch")
            dep[~np.isfinite(dep) | (dep <= 0)] = 0.0
            m = cv2.imread(str(scene / "mask" / cid / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(scene / "mask" / cid / f"{stem}.png")
            obj = m > 127
            ign = np.zeros_like(obj)
            if robot is not None:
                ign = tool_ignore_mask(dep, cam.K, cam.T_WC, robot["p_tcp_w"][t], robot["p_ee_w"][t], cfg)
            f_obj.append(obj)
            f_ign.append(ign)
            h, w = cam.height // ds, cam.width // ds
            rgbs.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
            d_s = dep[::ds, ::ds][:h, :w]                      # point-sample depth (never average across edges)
            deps.append(d_s.astype(np.float32))
            objs.append(obj[::ds, ::ds][:h, :w])
            igns.append(cv2.resize(ign.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
            # mixed-depth pixels: second difference (zero on any plane, large across a discontinuity) or an
            # invalid neighbour; a plain 3x3 range would flag every slanted surface
            pad = np.pad(d_s, 1, mode="edge")
            c0 = pad[1:-1, 1:-1]
            lap = np.maximum(np.abs(pad[1:-1, 2:] + pad[1:-1, :-2] - 2 * c0), np.abs(pad[2:, 1:-1] + pad[:-2, 1:-1] - 2 * c0))
            win = np.stack([pad[i:i + h, j:j + w] for i in range(3) for j in range(3)])
            edges.append(~(win > 0).all(0) | (lap > cfg.edge_k * _sigma_d(np.maximum(d_s, 0.1), cfg)))
            gx = 0.5 * (pad[1:-1, 2:] - pad[1:-1, :-2])
            gy = 0.5 * (pad[2:, 1:-1] - pad[:-2, 1:-1])
            g_d.append(np.sqrt(gx ** 2 + gy ** 2).astype(np.float32))
            rr = rgbs[-1]
            rp = np.pad(rr, ((1, 1), (1, 1), (0, 0)), mode="edge")
            gxr = 0.5 * (rp[1:-1, 2:] - rp[1:-1, :-2])
            gyr = 0.5 * (rp[2:, 1:-1] - rp[:-2, 1:-1])
            g_rgb.append(np.sqrt((gxr ** 2 + gyr ** 2).sum(-1)).astype(np.float32))
            times.append(t)
            cidx.append(ci)
            stems.append(stem)
    tt = lambda x, dt=torch.float32: torch.as_tensor(np.stack(x), dtype=dt, device=dev)
    Kt = torch.as_tensor(np.stack(Ks), dtype=torch.float32, device=dev)
    Tt = torch.as_tensor(np.stack(Ts), dtype=torch.float64, device=dev)
    return ImageStack(list(cameras), np.asarray(times), np.asarray(cidx), tt(rgbs), tt(deps), tt(objs, torch.bool),
                      tt(igns, torch.bool), tt(edges, torch.bool), tt(g_rgb), tt(g_d), Kt, Tt.float(), torch.linalg.inv(Tt).float(),
                      (authority.cameras[cameras[0]].height, authority.cameras[cameras[0]].width), stems, f_obj, f_ign)


def choose_targets(n_images, times, train_mask, src, cfg: DenseConfig):
    """Up to ``max_targets`` training images with |s - t| >= min_time_gap, spread evenly in time."""
    cand = [b for b in range(n_images) if train_mask[b] and abs(int(times[b]) - int(times[src])) >= cfg.min_time_gap]
    if len(cand) <= cfg.max_targets:
        return cand
    order = sorted(cand, key=lambda b: (times[b], b))
    pick = np.linspace(0, len(order) - 1, cfg.max_targets).round().astype(int)
    return [order[i] for i in sorted(set(pick))]


def _cauchy(r):
    import torch
    return torch.log1p(0.5 * r * r)


def _sample_nearest(img, u, v):
    import torch
    h, w = img.shape[:2]
    ui, vi = torch.round(u).long(), torch.round(v).long()
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi = ui.clamp(0, w - 1), vi.clamp(0, h - 1)
    return img[vi, ui], inside


def _sample_bilinear_rgb(img, u, v):
    import torch
    h, w = img.shape[:2]
    grid = torch.stack([(u + 0.5) / w * 2 - 1, (v + 0.5) / h * 2 - 1], -1)[None, None]
    out = torch.nn.functional.grid_sample(img.permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=False)
    return out[0, :, 0].T


def classify_and_cost(stack: ImageStack, b: int, x_w, rgb_src, sigma_c, cfg: DenseConfig):
    """Visibility class and robust cost of world points ``x_w`` [n,3] in target image ``b``."""
    import torch
    c = int(stack.cam_idx[b])
    K, T_CW = stack.K[c], stack.T_CW[c]
    y = x_w @ T_CW[:3, :3].T + T_CW[:3, 3]
    z = y[:, 2]
    zc = z.clamp(min=1e-6)
    u = K[0, 0] * y[:, 0] / zc + K[0, 2]
    v = K[1, 1] * y[:, 1] / zc + K[1, 2]
    D, inside = _sample_nearest(stack.depth[b], u, v)
    obj, _ = _sample_nearest(stack.obj[b], u, v)
    ign, _ = _sample_nearest(stack.ignore[b], u, v)
    edge, _ = _sample_nearest(stack.edge[b], u, v)
    g_d, _ = _sample_nearest(stack.grad_d[b], u, v)
    g_c, _ = _sample_nearest(stack.grad_rgb[b], u, v)
    # effective sigmas: sensor noise of both measurements + projected pixel uncertainty times the local gradient
    sig = torch.sqrt(_sigma_d(z, cfg) ** 2 + _sigma_d(D.clamp(min=0.05), cfg) ** 2 + (g_d * cfg.sigma_px) ** 2)
    status = torch.full_like(z, UNKNOWN, dtype=torch.long)
    known = inside & (z > 0.05) & (D > 0) & ~ign & ~edge
    band = cfg.occlusion_k * sig
    same = known & ((z - D).abs() <= band)
    occl = known & (z > D + band)
    free = known & (z < D - band)
    status[same & obj] = SUPPORTED
    status[same & ~obj] = FREE          # lands on a measured non-object surface while visible
    status[occl] = OCCLUDED
    status[free] = FREE
    cost = torch.zeros_like(z)
    r_d = (z - D) / sig
    rgb_t = _sample_bilinear_rgb(stack.rgb[b], u, v)
    r_c = (rgb_t - rgb_src).norm(dim=-1) / torch.sqrt(sigma_c ** 2 + (g_c * cfg.sigma_px) ** 2)
    sup = status == SUPPORTED
    cost[sup] = _cauchy(r_d[sup]) + _cauchy(r_c[sup])
    cost[status == FREE] = cfg.free_space_cost
    cost[status == OCCLUDED] = float(_cauchy(torch.tensor(1.0)) * 2)   # censored: expected inlier cost
    return status, cost, u, v, z, sig, r_c


def _box_sum(x, r):
    import torch
    k = 2 * r + 1
    return torch.nn.functional.avg_pool2d(x[None, None], k, stride=1, padding=r, count_include_pad=False)[0, 0] * k * k


def seed_prior_maps(stack: ImageStack, seeds, cfg: DenseConfig):
    """Bounded log-odds maps from confident discovery tracks: +prior moving, -prior static.

    ``seeds``: dict (cam_id, time) -> list of (u, v, label) at full resolution."""
    import torch
    N, h, w = stack.depth.shape
    prior = torch.zeros((N, h, w), device=stack.depth.device)
    counts = np.zeros(2, int)
    if not seeds:
        return prior, counts
    index = {(stack.cams[int(stack.cam_idx[i])], int(stack.times[i])): i for i in range(N)}
    r = cfg.seed_radius
    for key, pts in seeds.items():
        if key not in index:
            continue
        i = index[key]
        for (u, v, lab) in pts:
            uu, vv = int(round((u + 0.5) / cfg.work_downscale - 0.5)), int(round((v + 0.5) / cfg.work_downscale - 0.5))
            if 0 <= uu < w and 0 <= vv < h:
                s = cfg.seed_prior if lab == LABEL_M else -cfg.seed_prior
                prior[i, max(vv - r, 0):vv + r + 1, max(uu - r, 0):uu + r + 1] = s
                counts[int(lab == LABEL_M)] += 1
    return prior.clamp(-cfg.seed_prior, cfg.seed_prior), counts


def seeds_from_discovery(run_dir, kind: str, observation_files, cfg: DenseConfig):
    """Confident persistent track labels of branch ``kind`` (eligible, not outlier, posterior >= seed_min_prob)."""
    from .observations import TrackObservations
    s = np.load(Path(run_dir) / "part_support" / "track_support.npz")
    suf = {"revolute": "R", "prismatic": "P"}[kind]
    pm, ps, po = s[f"p_moving_{suf}"], s[f"p_static_{suf}"], s[f"p_outlier_{suf}"]
    lab = np.full(len(pm), -1)
    good = s["eligible"] & (po < 0.5)
    lab[good & (pm >= cfg.seed_min_prob)] = LABEL_M
    lab[good & (ps >= cfg.seed_min_prob)] = LABEL_S
    key = {(int(c), int(l)): i for i, (c, l) in enumerate(zip(s["track_camera"], s["track_local_id"]))}
    seeds = {}
    n_used = 0
    for f in observation_files:
        obs = TrackObservations.load(f)
        cam_ids = obs.camera_ids
        for k in range(obs.n_obs):
            tr = int(obs.track[k])
            gi = key.get((int(obs.track_camera[tr]), int(obs.track_local_id[tr])))
            if gi is None or lab[gi] < 0 or not obs.uv_valid[k]:
                continue
            seeds.setdefault((cam_ids[int(obs.cam[k])], int(obs.time[k])), []).append(
                (float(obs.uv[k, 0]), float(obs.uv[k, 1]), int(lab[gi])))
            n_used += 1
    return seeds, dict(tracks_moving=int((lab == LABEL_M).sum()), tracks_static=int((lab == LABEL_S).sum()),
                       tracks_unlabelled=int((lab < 0).sum()), seed_observations=n_used,
                       rule=f"eligible & p_outlier<0.5 & posterior>={cfg.seed_min_prob}")


def dense_evidence(stack: ImageStack, authority: JointAuthority, train_mask, cfg: DenseConfig, prior=None):
    """Per-image evidence maps: summed delta (c_S - c_M) and discriminative target counts at evidence res."""
    import torch
    dev = stack.depth.device
    N, h, w = stack.depth.shape
    A = torch.as_tensor(authority.transforms(), dtype=torch.float64, device=dev)
    A_inv = torch.linalg.inv(A)
    delta = torch.zeros((N, h, w), device=dev)
    ndisc = torch.zeros((N, h, w), device=dev)
    nfree = torch.zeros((2, N, h, w), device=dev)
    nsup = torch.zeros((2, N, h, w), device=dev)
    v, u = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    # colour sigma from non-discriminative supported targets (shared by S and M)
    sigma_c = torch.tensor(0.05, device=dev)
    samples = []
    for phase in ("calibrate", "score"):
        for a in range(N):
            m = stack.obj[a] & (stack.depth[a] > 0) & ~stack.ignore[a] & ~stack.edge[a]
            if not m.any():
                continue
            ci, t = int(stack.cam_idx[a]), int(stack.times[a])
            K, T_WC = stack.K[ci], stack.T_WC[ci]
            z = stack.depth[a][m]
            pu, pv = u[m].float(), v[m].float()
            x_c = torch.stack([(pu - K[0, 2]) * z / K[0, 0], (pv - K[1, 2]) * z / K[1, 1], z], -1)
            x_w = x_c @ T_WC[:3, :3].T + T_WC[:3, 3]
            rgb_src = stack.rgb[a][m]
            d_acc = torch.zeros_like(z)
            n_acc = torch.zeros_like(z)
            for b in choose_targets(N, stack.times, train_mask, a, cfg):
                s = int(stack.times[b])
                rel = (A[s] @ A_inv[t]).float()
                x_m = x_w @ rel[:3, :3].T + rel[:3, 3]
                st_s, c_s, us, vs, zs, sig_s, rc_s = classify_and_cost(stack, b, x_w, rgb_src, sigma_c, cfg)
                st_m, c_m, um, vm, zm, sig_m, _ = classify_and_cost(stack, b, x_m, rgb_src, sigma_c, cfg)
                sep_px = torch.sqrt((us - um) ** 2 + (vs - vm) ** 2)
                disc = (sep_px >= cfg.min_disc_px) | ((zs - zm).abs() >= cfg.min_disc_sigma * sig_s)
                # common visible claims only: an occluded/unknown prediction makes the target non-comparable,
                # so neither hypothesis can gain by hiding its points
                vis = lambda st: (st == SUPPORTED) | (st == FREE)
                common = vis(st_s) & vis(st_m)
                if phase == "calibrate":
                    agree = ~disc & (st_s == SUPPORTED)
                    if agree.any():
                        samples.append(rc_s[agree])
                    continue
                use = disc & common
                d_acc[use] += (c_s - c_m)[use]
                n_acc[use] += 1
                for k, st in ((0, st_s), (1, st_m)):
                    nfree[k, a][m] += (use & (st == FREE)).float()
                    nsup[k, a][m] += (use & (st == SUPPORTED)).float()
            if phase == "score":
                delta[a][m] = d_acc
                ndisc[a][m] = n_acc
        if phase == "calibrate":
            if samples:
                allr = torch.cat(samples)
                # rc_s was computed with sigma_c = 0.05: rescale its robust spread to colour units
                sigma_c = (0.05 * 1.4826 * (allr - allr.median()).abs().median()).clamp(min=0.01)
    return dict(delta=delta, ndisc=ndisc, nfree=nfree, nsup=nsup, sigma_c=float(sigma_c))


def decide_labels(stack: ImageStack, ev: dict, cfg: DenseConfig, prior=None):
    """Pool evidence over a local window, add the bounded seed prior, threshold -> S/M/U at evidence res."""
    import torch
    N, h, w = stack.depth.shape
    labels = torch.full((N, h, w), LABEL_U, dtype=torch.long, device=stack.depth.device)
    score = torch.zeros((N, h, w), device=stack.depth.device)
    for a in range(N):
        fg = stack.obj[a] & ~stack.ignore[a]
        sd = _box_sum(ev["delta"][a], cfg.pool_radius)
        sn = _box_sum(ev["ndisc"][a], cfg.pool_radius)
        mean = torch.where(sn > 0, sd / sn.clamp(min=1), torch.zeros_like(sd))
        enough = sn >= cfg.decide_min_targets
        data_m = enough & (mean > cfg.decide_mean_delta)
        data_s = enough & (mean < -cfg.decide_mean_delta)
        if prior is not None:
            # a confident track seed labels its pixel unless dense evidence clearly contradicts it (-> unknown)
            p = prior[a]
            seed_m, seed_s = p > 0, p < 0
            mov = fg & ((data_m & ~seed_s) | (seed_m & ~data_s))
            sta = fg & ((data_s & ~seed_m) | (seed_s & ~data_m))
            score[a] = mean + p / max(cfg.decide_min_targets, 1)
        else:
            mov, sta = fg & data_m, fg & data_s
            score[a] = mean
        lab = labels[a]
        lab[mov & ~sta] = LABEL_M
        lab[sta & ~mov] = LABEL_S
        lab[~fg] = -1
    return labels, score


def export_dense(stack: ImageStack, labels, score, ev, out_dir, authority_id: str, cfg: DenseConfig,
                 train_mask, extra: dict) -> dict:
    """Full-resolution PNG masks + compact per-image arrays + label manifest (atomic)."""
    import cv2
    out = Path(out_dir)
    H, W = stack.full_hw
    per_image = []
    tot = np.zeros(3, int)
    for a in range(len(stack.stems)):
        cid = stack.cams[int(stack.cam_idx[a])]
        stem = stack.stems[a]
        lab = labels[a].cpu().numpy()
        lab_full = cv2.resize(lab.astype(np.int16), (W, H), interpolation=cv2.INTER_NEAREST)
        obj_full, ign_full = stack.full_obj[a], stack.full_ignore[a]
        fg = obj_full & ~ign_full
        S = fg & (lab_full == LABEL_S)
        M = fg & (lab_full == LABEL_M)
        U = fg & ~S & ~M
        assert not (S & M).any() and ((S | M | U) == fg).all()
        d = out / cid
        for name, arr in (("mask_0", S), ("mask_1", M), ("mask_unknown", U), ("masks", obj_full), ("ignore_mask", ign_full)):
            (d / name).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / name / f"{stem}.png"), (arr.astype(np.uint8) * 255))
        sc = cv2.resize(score[a].cpu().numpy().astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        nd = cv2.resize(ev["ndisc"][a].cpu().numpy().astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        (d / "part_probability").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "part_probability" / f"{stem}.npz", score_moving_minus_static=sc.astype(np.float16),
                            discriminative_targets=nd.astype(np.uint16),
                            note="score = pooled mean (c_S - c_M) + seed prior; a calibrated score, NOT a probability")
        counts = [int(S.sum()), int(M.sum()), int(U.sum())]
        tot += counts
        per_image.append(dict(camera_id=cid, time_index=int(stack.times[a]), stem=stem, train=bool(train_mask[a]),
                              static_px=counts[0], moving_px=counts[1], unknown_px=counts[2],
                              ignore_px=int(ign_full.sum()), object_px=int(obj_full.sum())))
    fg_total = max(tot.sum(), 1)
    manifest = dict(schema="forceartgs.dense_support/1", authority_id=authority_id, config=asdict(cfg),
                    resampling=f"evidence at 1/{cfg.work_downscale} resolution, labels upsampled nearest and "
                               "intersected with the full-resolution object mask; unknown preserved",
                    sigma_colour=ev["sigma_c"], totals=dict(static=int(tot[0]), moving=int(tot[1]), unknown=int(tot[2]),
                                                            static_frac=tot[0] / fg_total, moving_frac=tot[1] / fg_total,
                                                            unknown_frac=tot[2] / fg_total),
                    images=per_image, **extra)
    atomic_write_json(out / "label_manifest.json", manifest)
    return manifest


def run_dense_support(scene_root, authority: JointAuthority, authority_id: str, cameras, out_dir, train_mask,
                      cfg: DenseConfig = DenseConfig(), seeds=None, seed_report=None, robot=None,
                      extra_seed_fn=None) -> dict:
    """G1 entry point. ``train_mask`` [N images, time-major (t, camera)] marks training targets."""
    import torch
    t0 = time.time()
    if authority.joint_type == "static":
        raise ValueError("dense S/M support is undefined for the static branch (everything is static)")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stack = load_image_stack(scene_root, authority, cameras, cfg, robot=robot)
    train_mask = np.asarray(train_mask, bool)
    if len(train_mask) != len(stack.stems):
        raise ValueError("train_mask must cover every (time, camera) image")
    if extra_seed_fn is not None:                  # e.g. TCP contact pixel seeds (multimodal frontend)
        extra, extra_rep = extra_seed_fn(stack)
        seeds = dict(seeds or {})
        for key, pts in extra.items():
            seeds.setdefault(key, []).extend(pts)
        seed_report = dict(seed_report or {}, tcp_pixel_seeds=extra_rep)
    prior, seed_counts = seed_prior_maps(stack, seeds, cfg)
    with torch.no_grad():
        ev = dense_evidence(stack, authority, train_mask, cfg)
        labels, score = decide_labels(stack, ev, cfg, prior=prior if seeds else None)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
    extra = dict(seed_report=seed_report or {}, seed_pixels_used=seed_counts.tolist(),
                 visibility_totals={h: dict(supported=int(ev["nsup"][k].sum()), free_space=int(ev["nfree"][k].sum()))
                                    for k, h in enumerate(("static", "moving"))},
                 robot_ignore="measured TCP/EE tool capsule" if robot is not None else "none (no robot archive)",
                 walltime_s=time.time() - t0, peak_vram_mib=peak, cameras=list(cameras),
                 joint_type=authority.joint_type)
    return export_dense(stack, labels, score, ev, out_dir, authority_id, cfg, train_mask, extra)
