#!/usr/bin/env python3
"""Contact segments for the multimodal frontend from FORCE PEAKS + VISUAL evidence (operator-validated estimator).

Per RGB-D frame of the reconstruction window:
  force   : ``contact`` state of samuele/export_force_contact_annotations.py (FreeArtGS fork; 6-sigma sustained force
            above a baseline-noise tare, >= 0.2 s), joined on source_frame_index;
  visual  : the observed object is moving (cumulative masked-depth motion grows by >= ``motion_step_m`` w.r.t. the
            previous frame in any camera) while the TCP is within ``near_m`` of the observed object depth
            (D1 audit ``rgbd_rate_timeline.csv``).
  contact = force OR visual; gaps of <= ``max_gap`` frames between contact frames are filled; runs of >= ``min_run``
  frames become ``contact_confirmed`` sliding segments (operator statement 2026-11-19: every real demo is a sliding
  contact with an almost linear TCP path; the operator validated this force + visual estimate). Everything else is
  written per frame to an audit CSV (never as a residual).

    python samuele/estimate_contact_segments.py --scene NAME --frames a:b:k --force-csv F --visual-csv V --output YAML
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--force-csv", type=Path, required=True)
    ap.add_argument("--visual-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--audit-csv", type=Path, required=True)
    ap.add_argument("--near-m", type=float, default=0.05)
    ap.add_argument("--motion-step-m", type=float, default=0.001)
    ap.add_argument("--max-gap", type=int, default=2)
    ap.add_argument("--min-run", type=int, default=3)
    a = ap.parse_args(argv)
    lo, hi, k = map(int, a.frames.split(":"))
    frames = list(range(lo, hi + 1, k))
    force = {}
    with open(a.force_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            force[int(r["source_frame_index"])] = r["contact_state"]
    vis = {}
    with open(a.visual_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            vis[int(r["rgbd_frame"])] = r
    cams = [c.replace("visual_motion_median_", "").replace("_m", "") for c in next(iter(vis.values())) if
            c.startswith("visual_motion_median_")]

    def f(r, key):
        v = r.get(key, "")
        return float(v) if v not in ("", None) else np.nan

    rows, contact = [], []
    for i, fr in enumerate(frames):
        r = vis[fr]
        src = int(r["source_frame_index"])
        fstate = force.get(src, "missing")
        prev = vis.get(frames[i - 1] if i else fr - k, r)
        step = max((f(r, f"visual_motion_median_{c}_m") - f(prev, f"visual_motion_median_{c}_m") for c in cams),
                   default=np.nan)
        dist = np.nanmin([f(r, f"tcp_obj_dist_{c}_m") for c in cams]) if cams else np.nan
        moving = bool(np.isfinite(step) and step >= a.motion_step_m)
        near = bool(np.isfinite(dist) and dist <= a.near_m)
        c = fstate == "contact" or (moving and near)
        contact.append(c)
        rows.append(dict(rgbd_frame=fr, source_frame_index=src, force_state=fstate, visual_motion_step_m=step,
                         tcp_obj_dist_m=dist, moving=moving, near=near, contact_raw=c))
    contact = np.array(contact)
    idx = np.flatnonzero(contact)
    filled = contact.copy()
    for x, y in zip(idx[:-1], idx[1:]):
        if 1 < y - x <= a.max_gap + 1:
            filled[x:y] = True
    segs, start = [], None
    for i, c in enumerate(list(filled) + [False]):
        if c and start is None:
            start = i
        if not c and start is not None:
            if i - start >= a.min_run:
                segs.append((start, i - 1))
            start = None
    for i, r in enumerate(rows):
        r["contact_filled"] = bool(filled[i])
        r["in_segment"] = any(s <= i <= e for s, e in segs)
    a.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.audit_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"# Contact segments estimated from FORCE PEAKS + VISUAL motion/proximity (operator-validated estimator,",
             f"# samuele/estimate_contact_segments.py). Operator statement 2026-11-19: all real demos are SLIDING contacts",
             f"# with an almost linear TCP path. Window {a.frames}. force: {a.force_csv.name} sha256 {sha(a.force_csv)[:16]};",
             f"# visual: {a.visual_csv.name} sha256 {sha(a.visual_csv)[:16]}; per-frame audit: {a.audit_csv}",
             "schema: forceartgs.contact_segments/v1", "frame_id: base_link", "frame_index_type: source_frame_index",
             f"scene_id: {a.scene}", "source: operator_validated_force_peak_plus_visual_contact_estimate", "segments:"]
    for n, (s, e) in enumerate(segs):
        lines += [f"  - segment_id: contact_{n}", f"    start_source_frame: {rows[s]['source_frame_index']}",
                  f"    end_source_frame: {rows[e]['source_frame_index']}", "    contact_state: contact_confirmed",
                  "    slip_mode: sliding", "    provenance: operator_label", "    contact_model: unilateral_point"]
    if not segs:
        lines.append("  []")
    a.output.write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(scene=a.scene, segments=[(rows[s]["rgbd_frame"], rows[e]["rgbd_frame"]) for s, e in segs],
                          n_force=int(sum(r["force_state"] == "contact" for r in rows)),
                          n_visual=int(sum(r["moving"] and r["near"] for r in rows)), n_frames=len(rows))))


if __name__ == "__main__":
    main()
