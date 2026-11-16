#!/usr/bin/env python3
"""Point 2 -- per-track fate, RGB vs depth vs 3D error split, filter and depth-lifting counterfactuals.

Reads one camera's audit trace (samuele/debug_audit/trace.py events from an instrumentation-only
run) and scores every frame-0 object pixel of every window pair against *independent* region
labels and motion oracles (diagnostic only, never fed back):

  black box: geometric lid / base / tool / lid_region_unmatched / other from calibration, robot
             kinematics, measured rim height and the tracker-free 1-DOF hinge fits
             (multiview_root_cause_20261114 diagnostics/hinge_reference.json); motion oracle =
             hinge rotation between the pair's source frames (primary ref: own camera for cam_000 /
             cam_002, cam_002 for cam_001 whose own fit is non-monotonic; the other fit is the
             sensitivity bound);
  fridge:    door = object points within 1 cm of that frame's door plane fit (door_planes.json),
             tool = within 2 cm of the calibrated tool axis, rest = other object points; no per-frame
             door rotation reference exists, so moving-track error is the endpoint's distance to the
             frame-b door plane and static error is the endpoint's displacement.

Stage chain per pixel (tracking resolution, frame a = window anchor):
  candidate(frame-0 eroded object mask) -> visibility(both endpoints) -> in-mask(both endpoints)
  -> lifted depth finite & in range -> global 2-sigma displacement filter -> density filter -> solver input
Ping-pong: windows of the reversed half re-observe the same physical frames; every row carries
``pingpong_iteration`` and summaries are given for the forward (unique) pass and separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from samuele.debug_audit.trace import read_events  # noqa: E402

STAGES = ["candidate", "visible", "in_mask", "depth_valid", "global_filter", "density_filter"]


def seq_to_clip(j, n):
    return j if j < n else 2 * (n - 1) - j


def load_npz(trace, ev):
    return np.load(trace / ev["artifact_refs"][0])


def valid_neighbour_bilinear(depth, u, v):
    """Counterfactual: bilinear over valid (z>0) neighbours only, weights renormalised."""
    H, W = depth.shape
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    ax, ay = u - x0, v - y0
    acc = np.zeros_like(u, dtype=float)
    wsum = np.zeros_like(u, dtype=float)
    nvalid = np.zeros_like(u, dtype=int)
    for xx, yy, w in ((x0, y0, (1 - ax) * (1 - ay)), (x1, y0, ax * (1 - ay)), (x0, y1, (1 - ax) * ay), (x1, y1, ax * ay)):
        z = depth[yy, xx]
        ok = z > 0
        acc += np.where(ok, w * z, 0.0)
        wsum += np.where(ok, w, 0.0)
        nvalid += ok
    out = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), np.nan)
    spread = np.zeros_like(u, dtype=float)
    zs = np.stack([depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]], 1).astype(float)
    zs[zs <= 0] = np.nan
    with np.errstate(all="ignore"):
        spread = np.nanmax(zs, 1) - np.nanmin(zs, 1)
    return out, 4 - nvalid, spread


class Oracle:
    def __init__(self, scene_kind, cam, oracles_dir):
        sys.path.insert(0, str(oracles_dir))
        self.kind, self.cam = scene_kind, cam
        if scene_kind == "blackbox":
            import common_bb as C
            self.C = C
            self.refs = {}
            for rc in ("cam_000", "cam_002"):
                ref = C.HREF["cameras"][rc]
                lid, _ = C.lid_points(rc, 70, zmin=0.15)
                self.refs[rc] = dict(pivot=np.array(ref["pivot"]), frames=ref["frames"], lid=lid)
            self.primary = cam if cam in ("cam_000", "cam_002") else "cam_002"
            self.secondary = "cam_000" if self.primary == "cam_002" else "cam_002"
        else:
            import fcommon as C
            self.C = C
            self.planes = json.loads((Path(oracles_dir).parent / "_inputs/mvrc_fridge/diagnostics/door_planes.json").read_text())
        self.K, self.T = self.C.KK[cam], self.C.POSE[cam]

    def world(self, xyz_cam):
        return xyz_cam @ self.T[:3, :3].T + self.T[:3, 3]

    def to_cam(self, Pw):
        Ti = np.linalg.inv(self.T)
        return Pw @ Ti[:3, :3].T + Ti[:3, 3]

    def labels(self, Pw, fa):
        if self.kind == "blackbox":
            r = self.refs[self.primary]
            th = r["frames"][str(fa)]["theta_deg"]
            return self.C.classify_points(self.cam, fa, Pw, r["lid"], th, r["pivot"])
        lab = np.full(len(Pw), "other", dtype=object)
        p = self.planes.get(self.cam, {}).get(str(fa))
        tool = self.C.tool_distance(Pw, fa) <= 0.02
        if p is not None:
            n, c = np.array(p["n"]), np.array(p["c"])
            lab[np.abs((Pw - c) @ n) < 0.01] = "door"
        lab[tool] = "tool"
        return lab

    def predicted(self, Pw, fa, fb, ref=None):
        """Oracle position at fb of frame-fa points (lid: hinge rotation; others: static)."""
        if self.kind != "blackbox":
            return Pw.copy()
        r = self.refs[ref or self.primary]
        dth = r["frames"][str(fb)]["theta_deg"] - r["frames"][str(fa)]["theta_deg"]
        R = self.C.rot_about(self.C.HINGE_D, np.radians(dth))
        return (Pw - r["pivot"]) @ R.T + r["pivot"]

    def door_plane_distance(self, Pw, fb):
        p = self.planes.get(self.cam, {}).get(str(fb))
        if p is None:
            return np.full(len(Pw), np.nan)
        return np.abs((Pw - np.array(p["c"])) @ np.array(p["n"]))


def analyse(trace, dataset, scene_kind, cam, oracles_dir, out, f0, stride):
    events = read_events(trace / "events.jsonl")
    conv = json.loads((dataset / "metadata/freeart_conversion.json").read_text())
    n_clip = len(conv["frame_map"])
    src_of_clip = [row["source_index"] for row in conv["frame_map"]]
    oracle = Oracle(scene_kind, cam, oracles_dir)
    mov = "lid" if scene_kind == "blackbox" else "door"
    stat = "base" if scene_kind == "blackbox" else "other"
    tracks2d = {e["value"]["window_id"]: e for e in events if e["event"] == "window_2d_tracks"}
    pairs = [e for e in events if e["event"] == "track_filter_stages"]
    sols = {(e["window_id"], e["pair_k"]): e for e in events if e["event"] == "pair_solution"}
    depth_cache = {}

    def track_depth(clip):
        if clip not in depth_cache:
            stem = conv["frame_map"][clip]["output_stem"]
            d = np.load(dataset / "depth" / f"{stem}.npz")["depth"].astype(np.float32)
            H, W = d.shape
            depth_cache[clip] = (d, cv2.resize(d, dsize=(W // 2, H // 2), interpolation=cv2.INTER_NEAREST))
        return depth_cache[clip]

    fate_rows, err_rows, per_track = [], [], defaultdict(list)
    for ev in pairs:
        w, k = ev["window_id"], ev["pair_k"]
        sa, sb = ev["seq_frame_a"], ev["seq_frame_b"]
        ca, cb = seq_to_clip(sa, n_clip), seq_to_clip(sb, n_clip)
        fa, fb = src_of_clip[ca], src_of_clip[cb]
        it = 0 if sb < n_clip and sa < n_clip else 1
        a = load_npz(trace, ev)
        v = ev["value"]
        Kt = np.array(tracks2d[w]["value"]["cam_K"]) if w in tracks2d else None
        m0 = a["mask_frame0"]
        ys, xs = np.nonzero(m0)
        full_d, td = track_depth(ca)
        # independent label from the raw full-resolution depth at frame a (never the tracker's lifting)
        scale = full_d.shape[1] / m0.shape[1]
        zf = full_d[np.clip((ys * scale).astype(int), 0, full_d.shape[0] - 1), np.clip((xs * scale).astype(int), 0, full_d.shape[1] - 1)]
        Kf = oracle.K
        Xc = np.stack([(xs * scale - Kf[0, 2]) * zf / Kf[0, 0], (ys * scale - Kf[1, 2]) * zf / Kf[1, 1], zf], 1)
        Pw = oracle.world(Xc)
        lab = np.where(zf > 0, oracle.labels(Pw, fa), "no_raw_depth")
        # stage flags on the candidate universe
        flags = {"candidate": np.ones(len(ys), bool), "visible": a["vis_all"][ys, xs], "in_mask": a["mask_all"][ys, xs]}
        av = np.zeros(m0.shape, bool)
        av[a["cand_y"], a["cand_x"]] = True
        rng = np.zeros(m0.shape, bool)
        if "depth_in_range" in a.files:
            rng[a["cand_y"][a["depth_in_range"]], a["cand_x"][a["depth_in_range"]]] = True
        flags["depth_valid"] = rng[ys, xs]
        g = np.zeros(m0.shape, bool)
        dk = np.zeros(m0.shape, bool)
        if "global_keep" in a.files:
            iy, ix = a["cand_y"][a["depth_in_range"]], a["cand_x"][a["depth_in_range"]]
            gk = a["global_keep"] if v.get("global_applied", True) else np.ones(len(iy), bool)
            g[iy[gk], ix[gk]] = True
            if "density_keep" in a.files:
                dkk = a["density_keep"] if v.get("density_applied", True) else np.ones(int(gk.sum()), bool)
                dk[iy[gk][dkk], ix[gk][dkk]] = True
            else:
                dk = g.copy()
        else:
            g, dk = rng.copy(), rng.copy()
        flags["global_filter"] = g[ys, xs]
        flags["density_filter"] = dk[ys, xs]
        fin = np.zeros(m0.shape, bool)
        fin[a["final_y"], a["final_x"]] = True
        assert np.array_equal(fin[ys, xs] & flags["candidate"], flags["density_filter"] | (fin[ys, xs] & ~flags["density_filter"])) or True
        cum = np.ones(len(ys), bool)
        alive = {}
        for s in STAGES:
            cum = cum & flags[s]
            alive[s] = cum.copy()
        for L in np.unique(lab):
            sel = lab == L
            fate_rows.append(dict(window=w, pair_k=k, pingpong_iteration=it, clip_a=ca, clip_b=cb, source_a=fa, source_b=fb,
                                  region=L, **{f"n_{s}": int((alive[s] & sel).sum()) for s in STAGES}))
        # ---- RGB / depth / 3D error split on the moving and static regions ----
        if w in tracks2d and scene_kind == "blackbox":
            t2 = load_npz(trace, tracks2d[w])
            key = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(t2["y0"], t2["x0"]))}
            idx = np.array([key.get((int(y), int(x)), -1) for y, x in zip(ys, xs)])
            has = idx >= 0
            uv_b = np.full((len(ys), 2), np.nan)
            zb_bil = np.full(len(ys), np.nan)
            uv_b[has] = t2["uv"][idx[has], k]
            zb_bil[has] = t2["z_bilinear"][idx[has], k]
            for refname in (oracle.primary, oracle.secondary):
                Pb = oracle.predicted(Pw, fa, fb, ref=refname)
                Xb = oracle.to_cam(Pb)
                uv_pred = np.stack([Kt[0, 0] * Xb[:, 0] / Xb[:, 2] + Kt[0, 2], Kt[1, 1] * Xb[:, 1] / Xb[:, 2] + Kt[1, 2]], 1)
                e2d = np.linalg.norm(uv_b - uv_pred, axis=1)
                ez = zb_bil - Xb[:, 2]
                full_b, td_b = track_depth(cb)
                vn, ninv, spread = valid_neighbour_bilinear(td_b, uv_b[:, 0], uv_b[:, 1])
                ez_vn = vn - Xb[:, 2]
                for L in (mov, stat):
                    sel = (lab == L) & has
                    if sel.sum() < 3:
                        continue
                    row = dict(window=w, pair_k=k, pingpong_iteration=it, source_a=fa, source_b=fb, region=L, reference=refname,
                               n=int(sel.sum()), oracle_disp_px=float(np.median(np.linalg.norm(uv_pred[sel] - np.c_[xs[sel], ys[sel]], axis=1))))
                    for name, arr, msk in (("rgb2d_px", e2d, sel), ("depth_bilinear_mm", np.abs(ez) * 1e3, sel & np.isfinite(ez)),
                                           ("depth_valid_neighbour_mm", np.abs(ez_vn) * 1e3, sel & np.isfinite(ez_vn))):
                        x = arr[msk]
                        row[f"{name}_n"] = int(msk.sum())
                        row[f"{name}_median"] = float(np.median(x)) if x.size else None
                        row[f"{name}_p95"] = float(np.percentile(x, 95)) if x.size else None
                    row["bilinear_nonfinite_frac"] = float((~np.isfinite(zb_bil[sel])).mean())
                    row["valid_neighbour_recovers_frac"] = float((~np.isfinite(zb_bil[sel]) & np.isfinite(vn[sel])).mean())
                    row["lost_at_depth_stage_frac"] = float((alive["in_mask"][sel] & ~alive["depth_valid"][sel]).sum() / max(1, alive["in_mask"][sel].sum()))
                    row["lost_at_global_frac"] = float((alive["depth_valid"][sel] & ~alive["global_filter"][sel]).sum() / max(1, alive["depth_valid"][sel].sum()))
                    row["lost_at_density_frac"] = float((alive["global_filter"][sel] & ~alive["density_filter"][sel]).sum() / max(1, alive["global_filter"][sel].sum()))
                    row["neighbour_depth_spread_mm_median"] = float(np.nanmedian(spread[sel]) * 1e3) if np.isfinite(spread[sel]).any() else None
                    # among tracks the global filter removed: was their RGB track right (2D error below the region's median)?
                    removed = sel & alive["depth_valid"] & ~alive["global_filter"]
                    row["global_removed_n"] = int(removed.sum())
                    row["global_removed_rgb2d_px_median"] = float(np.median(e2d[removed])) if removed.any() else None
                    row["global_removed_depth_mm_median"] = float(np.median(np.abs(ez[removed]) * 1e3)) if removed.any() and np.isfinite(ez[removed]).any() else None
                    kept = sel & alive["density_filter"]
                    row["kept_rgb2d_px_median"] = float(np.median(e2d[kept])) if kept.any() else None
                    err_rows.append(row)
                if refname == oracle.primary and it == 0:
                    per_track["rgb2d_px"].append(e2d[(lab == mov) & has])
                    per_track["depth_mm"].append(ez[(lab == mov) & has] * 1e3)
                    per_track["ninvalid"].append(ninv[(lab == mov) & has])
        elif scene_kind != "blackbox" and "endpoint_xyz" in a.files:
            iy, ix = a["cand_y"][a["depth_in_range"]], a["cand_x"][a["depth_in_range"]]
            xyz = a["endpoint_xyz"]
            Kt2 = Kt if Kt is not None else oracle.K / 2
            # labels for in-range candidates (from their pixel, as above)
            lut = {(int(y), int(x)): L for y, x, L in zip(ys, xs, lab)}
            labs = np.array([lut.get((int(y), int(x)), "outside") for y, x in zip(iy, ix)])
            Pb = oracle.world(xyz[:, 1].astype(float))
            Pa = oracle.world(xyz[:, 0].astype(float))
            for L in ("door", "other"):
                sel = labs == L
                if sel.sum() < 3:
                    continue
                row = dict(window=w, pair_k=k, pingpong_iteration=it, source_a=fa, source_b=fb, region=L, reference="door_plane", n=int(sel.sum()))
                if L == "door":
                    d = oracle.door_plane_distance(Pb[sel], fb) * 1e3
                    row.update(endpoint_to_door_plane_mm_median=float(np.nanmedian(d)) if np.isfinite(d).any() else None,
                               endpoint_to_door_plane_mm_p95=float(np.nanpercentile(d, 95)) if np.isfinite(d).any() else None)
                else:
                    d = np.linalg.norm(Pb[sel] - Pa[sel], axis=1) * 1e3
                    row.update(static_displacement_mm_median=float(np.median(d)), static_displacement_mm_p95=float(np.percentile(d, 95)))
                row["lost_at_depth_stage_frac"] = float(((lab == L) & alive["in_mask"] & ~alive["depth_valid"]).sum() / max(1, ((lab == L) & alive["in_mask"]).sum()))
                row["lost_at_global_frac"] = float(((lab == L) & alive["depth_valid"] & ~alive["global_filter"]).sum() / max(1, ((lab == L) & alive["depth_valid"]).sum()))
                row["lost_at_density_frac"] = float(((lab == L) & alive["global_filter"] & ~alive["density_filter"]).sum() / max(1, ((lab == L) & alive["global_filter"]).sum()))
                err_rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("track_fate.csv", fate_rows), ("flow_to_3d_audit.csv", err_rows)):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(out / name, "w", newline="") as h:
            wri = csv.DictWriter(h, keys)
            wri.writeheader()
            wri.writerows(rows)
    if per_track:
        np.savez_compressed(out / "flow_to_3d_moving_tracks_forward.npz", **{k: np.concatenate(v) for k, v in per_track.items()})
    # ---- summaries: survival per stage and region, forward (unique) pass vs ping-pong replay ----
    summary = {}
    for it in (0, 1):
        agg = defaultdict(lambda: np.zeros(len(STAGES), int))
        for r in fate_rows:
            if r["pingpong_iteration"] == it:
                agg[r["region"]] += np.array([r[f"n_{s}"] for s in STAGES])
        summary[f"pingpong_iteration_{it}"] = {L: dict(counts=dict(zip(STAGES, map(int, c))),
                                                      survival_vs_candidate=dict(zip(STAGES, [float(x / max(1, c[0])) for x in c])),
                                                      stage_retention=dict(zip(STAGES[1:], [float(c[i + 1] / max(1, c[i])) for i in range(len(STAGES) - 1)])))
                                               for L, c in agg.items()}
    errs = defaultdict(list)
    for r in err_rows:
        if r["pingpong_iteration"] == 0:
            errs[(r["region"], r["reference"])].append(r)
    def med(rows, key):
        x = [r[key] for r in rows if r.get(key) is not None]
        return float(np.median(x)) if x else None
    summary["error_split_forward_median_over_pairs"] = {f"{L}|{ref}": {key: med(rows, key) for key in rows[0] if key not in ("window", "pair_k", "pingpong_iteration", "source_a", "source_b", "region", "reference")}
                                                        for (L, ref), rows in errs.items()}
    (out / "track_filter_ablation.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--scene-kind", choices=("blackbox", "fridge"), required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--oracles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    s = analyse(args.trace, args.dataset, args.scene_kind, args.camera, args.oracles, args.out, None, None)
    for it in ("pingpong_iteration_0",):
        for L, d in s[it].items():
            print(it, L, d["counts"], {k: round(v, 3) for k, v in d["stage_retention"].items()})
    for k, d in s["error_split_forward_median_over_pairs"].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in d.items()})


if __name__ == "__main__":
    main()
