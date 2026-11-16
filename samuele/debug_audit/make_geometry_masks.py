#!/usr/bin/env python3
"""DIAGNOSTIC ORACLE (black box only): tracker-independent part masks for ICP counterfactuals.

lid    = object mask & base-frame height > 0.165 m & > 2 cm from the calibrated tool axis
static = object mask & base-frame height < 0.150 m
Uses only calibration, robot kinematics and the measured rim height (same definition as
multiview_root_cause_20261114 scripts/make_geo_masks.py).  Never used by any learned method;
outputs go to a *copy* of a prepared dataset, never to the source scene.

    python samuele/debug_audit/make_geometry_masks.py --oracles <audit>/_oracles --camera cam_002 \
        --src <dataset> --dst <new dataset> --first-source-frame 70 --stride 2
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracles", type=Path, required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.oracles))
    import common_bb as C
    if args.dst.exists():
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst, ignore=shutil.ignore_patterns(
        "tracking", "point_clouds*", "registered_*", "global_poses_*", "pose_valid_*", "relative_transforms_*",
        "transforms_*.json", "trajectory_vis_*"))
    conv = json.loads((args.dst / "metadata/freeart_conversion.json").read_text())
    stats = []
    for row in conv["frame_map"]:
        stem, f = row["output_stem"], int(row["source_index"])
        obj = cv2.imread(str(args.dst / "masks" / f"{stem}.png"), 0) > 127
        P, uv = C.backproject(args.camera, f, obj, erode=0)
        lid = (P[:, 2] > 0.165) & (C.tool_distance(P, f) > 0.02)
        sta = P[:, 2] < 0.150
        m1 = np.zeros(obj.shape, np.uint8)
        m0 = np.zeros(obj.shape, np.uint8)
        m1[uv[lid, 1], uv[lid, 0]] = 255
        m0[uv[sta, 1], uv[sta, 0]] = 255
        cv2.imwrite(str(args.dst / "mask_1" / f"{stem}.png"), m1)
        cv2.imwrite(str(args.dst / "mask_0" / f"{stem}.png"), m0)
        stats.append(dict(stem=stem, source_frame=f, lid_px=int(lid.sum()), static_px=int(sta.sum())))
    (args.dst / "metadata/geometry_oracle_masks.json").write_text(json.dumps(dict(
        oracle="DIAGNOSTIC ONLY: geometry-derived part masks (not training supervision)", camera=args.camera,
        definition="lid: z_base>0.165 & tool distance>0.02 m; static: z_base<0.150", frames=stats), indent=1))
    print(args.camera, "lid/static px first/last:", stats[0]["lid_px"], stats[0]["static_px"], "/", stats[-1]["lid_px"], stats[-1]["static_px"])


if __name__ == "__main__":
    main()
