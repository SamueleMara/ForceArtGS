#!/usr/bin/env python3
"""B3 -- frozen 2D tracks, original bilinear depth lifting vs valid-neighbour lifting (diagnostic fork).

For every forward window pair of the black box, take the frozen AllTracker 2D tracks of frame-0
object pixels (window_2d_tracks trace), label them with the geometry oracle, lift both endpoints
(a) exactly as the pipeline does (bilinear on the nearest-resized depth with 0 -> -inf, so any
invalid neighbour kills the track) and (b) with bilinear over valid neighbours only, then fit a
rigid motion to the oracle-lid tracks and measure the rotation about the reference hinge.
Masks, filters and the solver are untouched: this isolates whether lifting removes rotation
information, not whether the pipeline would use it.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from samuele.debug_audit.trace import read_events  # noqa: E402
from samuele.debug_audit.point2_tracks import Oracle, seq_to_clip, valid_neighbour_bilinear  # noqa: E402
from samuele.debug_audit.point3_multiview import about, kabsch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--oracles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ev = read_events(args.trace / "events.jsonl")
    conv = json.loads((args.dataset / "metadata/freeart_conversion.json").read_text())
    N = len(conv["frame_map"])
    src = [r["source_index"] for r in conv["frame_map"]]
    orc = Oracle("blackbox", args.camera, args.oracles)
    B = orc.T
    depth = {}

    def td(clip):
        if clip not in depth:
            d = np.load(args.dataset / "depth" / f"{conv['frame_map'][clip]['output_stem']}.npz")["depth"].astype(np.float32)
            depth[clip] = cv2.resize(d, (d.shape[1] // 2, d.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        return depth[clip]

    rows = []
    for e in ev:
        if e["event"] != "window_2d_tracks":
            continue
        v = e["value"]
        seq = v["window_seq_frames"]
        if seq[-1] >= N:
            continue
        a = np.load(args.trace / e["artifact_refs"][0])
        K = np.array(v["cam_K"])
        uv, zb = a["uv"], a["z_bilinear"]
        ca = seq_to_clip(seq[0], N)
        fa = src[ca]
        u0, v0 = a["x0"].astype(float), a["y0"].astype(float)
        z0 = td(ca)[a["y0"], a["x0"]].astype(float)
        X0 = np.stack([(u0 - K[0, 2]) * z0 / K[0, 0], (v0 - K[1, 2]) * z0 / K[1, 1], z0], 1)
        lab = np.where(z0 > 0, orc.labels(X0 @ B[:3, :3].T + B[:3, 3], fa), "no_raw_depth")
        lid = lab == "lid"
        for k in range(1, uv.shape[1]):
            cb = seq_to_clip(seq[k], N)
            fb = src[cb]
            ref = orc.refs[orc.primary]["frames"]
            dref = ref[str(fb)]["theta_deg"] - ref[str(fa)]["theta_deg"]
            if abs(dref) < 2:
                continue
            ub, vb = uv[:, k, 0].astype(float), uv[:, k, 1].astype(float)
            zvn, _, _ = valid_neighbour_bilinear(td(cb), ub, vb)
            r = dict(window=v["window_id"], k=k, source_a=fa, source_b=fb, reference_deg=dref, lid_tracks=int(lid.sum()))
            for name, z in (("pipeline_bilinear", zb[:, k].astype(float)), ("valid_neighbour", zvn)):
                ok = lid & np.isfinite(z) & (z > 0) & (z0 > 0) & np.isfinite(zb[:, 0])
                Xb = np.stack([(ub - K[0, 2]) * z / K[0, 0], (vb - K[1, 2]) * z / K[1, 1], z], 1)
                r[f"{name}_n"] = int(ok.sum())
                if ok.sum() >= 6:
                    T = kabsch(X0[ok], Xb[ok])
                    r[f"{name}_deg"] = about((B @ T @ np.linalg.inv(B))[:3, :3], orc.C.HINGE_D)
            rows.append(r)
    args.out.mkdir(parents=True, exist_ok=True)
    summ = {}
    for name in ("pipeline_bilinear", "valid_neighbour"):
        f = [r[f"{name}_deg"] / r["reference_deg"] for r in rows if f"{name}_deg" in r]
        n = [r[f"{name}_n"] for r in rows]
        summ[name] = dict(pairs=len(f), median_recovered_fraction=float(np.median(f)) if f else None,
                          wrong_sign=int(sum(x < 0 for x in f)), median_lid_tracks=float(np.median(n)) if n else None)
    (args.out / "depth_lifting_counterfactual.json").write_text(json.dumps(dict(summary=summ, pairs=rows), indent=1))
    print(args.camera, json.dumps(summ))


if __name__ == "__main__":
    main()
