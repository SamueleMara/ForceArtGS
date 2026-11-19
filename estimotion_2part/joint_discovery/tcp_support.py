"""TCP-guided moving-support evidence (multimodal spec section 5.1).

The TCP is an object-contact PROPOSAL, not a gripper-mask label:

* Tool occupancy uses the measured tool geometry (``metadata/robot_tool.json``: T_ee_tcp) as a capsule along the
  TCP -> EE line, starting one tool radius above the tip, so the surface the tip touches is NOT tool. Points inside the
  capsule are TOOL, a margin band around it is AMBIGUOUS (kept as unknown, never deleted as background, never claimed
  as object), everything else is CLEAR. Without a measured tool mesh the capsule radius is an approximation and is
  flagged ``APPROXIMATE`` -- it replaces the blanket "within 5 cm of the tool" exclusion that removed object points.
* Contact seeds: CLEAR tracks within an uncertainty-scaled radius of the TCP at a contact frame. Independently
  confirmed contact -> positive moving-object seeds; unknown contact -> low-confidence proposals that still need real
  motion evidence (relaxed motion test), never a residual or a selection statistic.
* Sticking segments: the fitted canonical anchor predicts where the contacted surface is at EVERY contact frame
  (``anchor_seeds``), including frames where the tip region was occluded earlier.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .motion_field import PointTable

TOOL, AMBIGUOUS, CLEAR = "TOOL", "AMBIGUOUS", "CLEAR"


@dataclass
class ToolModel:
    radius_m: float
    ambiguity_margin_m: float
    extend_m: float
    tip_offset_m: float
    status: str
    source: str
    tool_name: str = "unknown"

    @classmethod
    def from_scene(cls, scene_root, cfg):
        path = Path(scene_root) / cfg.tool_geometry_source if scene_root else None
        name, src = "unknown", "config defaults"
        if path is not None and path.is_file():
            meta = json.loads(path.read_text())
            name = meta.get("tool", "unknown")
            src = f"{path} (T_ee_tcp; no tool mesh/radius)"
        return cls(cfg.tool_radius_m, cfg.tool_ambiguity_margin_m, cfg.tool_extend_m, cfg.tool_radius_m,
                   "APPROXIMATE", src, name)

    def to_json(self):
        return asdict(self)


def _segment_distance(X, a, b):
    ab = b - a
    s = np.clip(((X - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(X - (a + s[:, None] * ab), axis=1)


def tool_distance(points, p_tcp, p_ee, tool: ToolModel, return_height=False):
    """Distance of world points to the tool capsule axis (tip + radius -> EE + extend); optionally the height of each
    point along the shaft above the tip (negative = below the tip, i.e. on the contacted object's side)."""
    d = np.asarray(p_ee, float) - np.asarray(p_tcp, float)
    dn = d / max(np.linalg.norm(d), 1e-9)
    a = np.asarray(p_tcp, float) + tool.tip_offset_m * dn
    b = np.asarray(p_ee, float) + tool.extend_m * dn
    X = np.atleast_2d(points)
    dist = _segment_distance(X, a, b)
    if return_height:
        return dist, (X - np.asarray(p_tcp, float)) @ dn
    return dist


def classify_tracks_tool(tab: PointTable, p_tcp, p_ee, tool: ToolModel, times=None) -> dict:
    """Per-track TOOL / AMBIGUOUS / CLEAR from 3D occupancy at the frames the track is observed."""
    T, N = tab.T, tab.N
    n = np.zeros(N)
    n_tool = np.zeros(N)
    n_amb = np.zeros(N)
    times = range(T) if times is None else times
    for t in times:
        if t >= len(p_tcp) or not np.isfinite(p_tcp[t]).all() or p_ee is None or not np.isfinite(p_ee[t]).all():
            continue
        ok = np.isfinite(tab.X[t]).all(1)
        if not ok.any():
            continue
        d, h = tool_distance(tab.X[t, ok], p_tcp[t], p_ee[t], tool, return_height=True)
        idx = np.flatnonzero(ok)
        n[idx] += 1
        n_tool[idx] += d < tool.radius_m
        # the ambiguity band runs alongside the shaft only: beside/below the tip lies the contacted object
        n_amb[idx] += (d >= tool.radius_m) & (d < tool.radius_m + tool.ambiguity_margin_m) & (h > tool.tip_offset_m)
    frac_t = n_tool / np.maximum(n, 1)
    frac_a = n_amb / np.maximum(n, 1)
    label = np.full(N, CLEAR, dtype=object)
    label[(frac_t + frac_a) >= 0.5] = AMBIGUOUS
    label[frac_t >= 0.5] = TOOL
    return dict(label=label, frac_tool=frac_t, frac_ambiguous=frac_a, n_checked=n,
                counts={k: int((label == k).sum()) for k in (TOOL, AMBIGUOUS, CLEAR)}, tool=tool.to_json())


def contact_seeds(tab: PointTable, p_tcp, contact_state, tool_label, cfg, sigma_tcp_m=0.015) -> dict:
    """Tracks near the TCP at contact frames. ``contact_state``: [T] str (contact_confirmed / contact_probable /
    contact_unknown / no_contact). Confirmed -> positive seeds; probable/unknown -> proposals; no_contact -> nothing."""
    r = cfg.seed_radius_m + 2 * sigma_tcp_m
    pos, prop = np.zeros(tab.N, int), np.zeros(tab.N, int)
    clear = np.asarray(tool_label) == CLEAR
    per_t = []
    for t in range(min(tab.T, len(p_tcp))):
        st = contact_state[t]
        if st == "no_contact" or not np.isfinite(p_tcp[t]).all():
            per_t.append(dict(t=t, state=st, n=0))
            continue
        ok = np.isfinite(tab.X[t]).all(1) & clear
        d = np.linalg.norm(tab.X[t] - p_tcp[t], axis=1)
        near = ok & (d < r)
        (pos if st == "contact_confirmed" else prop)[near] += 1
        per_t.append(dict(t=t, state=st, n=int(near.sum())))
    return dict(positive=pos > 0, proposal=(prop > 0) & ~(pos > 0), radius_m=float(r), per_time=per_t,
                n_positive=int((pos > 0).sum()), n_proposal=int(((prop > 0) & ~(pos > 0)).sum()))


def anchor_seeds(tab: PointTable, kind, axis, origin, q, anchor, times, radius_m, tool_label) -> np.ndarray:
    """Tracks at the predicted contacted material point A_H(q_t) c*_k over every contact frame of a sticking segment."""
    from .contact_modes import predict_tcp
    pred = predict_tcp(kind, axis, origin, q, anchor, times)
    hit = np.zeros(tab.N, bool)
    clear = np.asarray(tool_label) == CLEAR
    for t, p in zip(times, pred):
        ok = np.isfinite(tab.X[t]).all(1) & clear
        hit |= ok & (np.linalg.norm(tab.X[t] - p, axis=1) < radius_m)
    return hit


def update_support(base: dict, tool: dict, seeds: dict, sigma_track, cfg, anchor_hits=None) -> dict:
    """Candidate-independent S/M/U + tool, updated with TCP evidence (provenance recorded per track)."""
    span = base["robust_disp_m"]
    S, M, U = base["S_core"].copy(), base["M_proposal"].copy(), base["U"].copy()
    lab = np.asarray(tool["label"])
    is_tool = lab == TOOL
    amb = lab == AMBIGUOUS
    added_pos = seeds["positive"] & ~M & ~is_tool & (span >= 0.008)
    if anchor_hits is not None:
        added_pos |= anchor_hits & ~M & ~is_tool & (span >= 0.008)
    added_prop = seeds["proposal"] & ~M & ~is_tool & ~amb & (span >= np.maximum(0.008, cfg.proposal_k_sigma * 2 *
                                                                                 np.nan_to_num(sigma_track, nan=np.inf)))
    M = (M | added_pos | added_prop) & ~is_tool & ~amb
    S &= ~is_tool & ~amb & ~M
    U = ~(S | M) & ~is_tool
    out = dict(base, S_core=S, M_proposal=M, U=U, TOOL=is_tool)
    out["counts"] = dict(S_core=int(S.sum()), M_proposal=int(M.sum()), U=int(U.sum()), TOOL=int(is_tool.sum()),
                         total=int(len(S)))
    out["tcp_provenance"] = dict(removed_as_tool=int((base["M_proposal"] & is_tool).sum() + (base["S_core"] & is_tool).sum()),
                                 moved_to_unknown_ambiguous=int(((base["M_proposal"] | base["S_core"]) & amb).sum()),
                                 added_from_confirmed_contact=int(added_pos.sum()),
                                 added_from_unconfirmed_proposals=int(added_prop.sum()),
                                 rule="TOOL: >=50% of observations inside the tool capsule; AMBIGUOUS -> U; "
                                      "confirmed-contact seeds need span >= 8 mm; proposals need the relaxed "
                                      f"{cfg.proposal_k_sigma}-sigma motion test")
    return out


def sigma_track(tab: PointTable) -> np.ndarray:
    return np.nanmedian(tab.S, axis=0)


# --------------------------------------------------------------------------- pixel seeds for the dense pass
def tcp_pixel_seeds(stack, p_tcp, p_ee, confirmed_times, cfg_dense, tool: ToolModel, radius_m=0.04, depth_band_m=0.02):
    """(cam, time) -> [(u, v, LABEL_M)] at FULL resolution: object pixels around the projected TCP whose depth is
    within ``depth_band_m`` of the TCP depth, outside the tool capsule, inside the object mask. Confirmed contact only."""
    from .dense_support import LABEL_M
    from .observations import project
    seeds = {}
    n = 0
    for i in range(len(stack.stems)):
        t = int(stack.times[i])
        if t not in confirmed_times or not np.isfinite(p_tcp[t]).all():
            continue
        cam = stack.cams[int(stack.cam_idx[i])]
        tonp = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        K, T_WC = tonp(stack.K[int(stack.cam_idx[i])]).astype(float), tonp(stack.T_WC[int(stack.cam_idx[i])]).astype(float)
        uv, z = project(p_tcp[t][None], K, np.linalg.inv(T_WC))          # K is at evidence resolution
        if z[0] <= 0.05:
            continue
        ds = cfg_dense.work_downscale
        dep = tonp(stack.depth[i])
        obj = tonp(stack.obj[i]).astype(bool)
        h, w = dep.shape
        r_px = max(2, int(K[0, 0] * radius_m / z[0]))
        uc, vc = int(round(uv[0, 0])), int(round(uv[0, 1]))
        pts = []
        for vv in range(max(vc - r_px, 0), min(vc + r_px + 1, h)):
            for uu in range(max(uc - r_px, 0), min(uc + r_px + 1, w)):
                zz = dep[vv, uu]
                if zz <= 0 or not obj[vv, uu] or abs(zz - z[0]) > depth_band_m:
                    continue
                u_f, v_f = (uu + 0.5) * ds - 0.5, (vv + 0.5) * ds - 0.5      # full-resolution seed coordinates
                x_c = np.array([(uu - K[0, 2]) * zz / K[0, 0], (vv - K[1, 2]) * zz / K[1, 1], zz])
                x_w = T_WC[:3, :3] @ x_c + T_WC[:3, 3]
                if tool_distance(x_w, p_tcp[t], p_ee[t], tool)[0] < tool.radius_m + tool.ambiguity_margin_m:
                    continue
                pts.append((u_f, v_f, LABEL_M))
        if pts:
            seeds[(cam, t)] = pts
            n += len(pts)
    return seeds, dict(n_seed_pixels=n, n_images=len(seeds), rule="confirmed contact only; depth band and tool capsule")


def write_overlays(scene_root, frame_map, calibs, p_tcp, p_ee, tool: ToolModel, tab: PointTable, labels: dict,
                   out_dir, times=None, max_frames=6):
    """Debug overlays per view: TCP tip, tool axis, track labels (green M, blue S, yellow U, red TOOL)."""
    import cv2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T = frame_map.T
    times = list(range(0, T, max(1, T // max_frames)))[:max_frames] if times is None else times
    written = []
    colors = dict(M_proposal=(0, 200, 0), S_core=(255, 120, 0), U=(0, 220, 220), TOOL=(0, 0, 255))
    for ci, cid in enumerate(tab.obs.camera_ids):
        cal = calibs[cid]
        T_CW = np.linalg.inv(cal.T_WC)
        for t in times:
            img = cv2.imread(str(Path(scene_root) / "rgb" / cid / f"frame_{int(frame_map.rgbd_frame_index[t]):06d}.png"))
            if img is None:
                continue
            for name, col in colors.items():
                m = labels.get(name)
                if m is None:
                    continue
                sel = np.asarray(m, bool) & (tab.track_camera == ci) & np.isfinite(tab.UV[t]).all(1)
                for u, v in tab.UV[t, sel][::3]:
                    cv2.circle(img, (int(u), int(v)), 2, col, -1)
            if np.isfinite(p_tcp[t]).all():
                def proj(x):
                    xc = T_CW[:3, :3] @ x + T_CW[:3, 3]
                    return (int(cal.K[0, 0] * xc[0] / xc[2] + cal.K[0, 2]), int(cal.K[1, 1] * xc[1] / xc[2] + cal.K[1, 2])), xc[2]
                (a, za), (b, zb) = proj(p_tcp[t]), proj(p_ee[t]) if p_ee is not None else (proj(p_tcp[t]))
                if za > 0 and zb > 0:
                    cv2.line(img, a, b, (0, 0, 255), 1)
                    cv2.circle(img, a, 6, (255, 0, 255), 2)
            cv2.putText(img, f"{cid} t={t} green=M blue=S yellow=U red=TOOL magenta=TCP", (5, 15), 0, 0.4, (255, 255, 255), 1)
            f = out / f"{cid}_t{t:03d}.png"
            cv2.imwrite(str(f), img)
            written.append(str(f))
    return written
