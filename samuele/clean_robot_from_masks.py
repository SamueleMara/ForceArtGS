#!/usr/bin/env python3
"""Remove robot-tool pixels from the object masks of a prepared FreeArt dataset, upstream of tracking.

Object masks of real sequences can include a dark tool in contact with the object.  Those
pixels enter tracking (tracks are only kept inside ``masks/``), the two-part segmentation and
the per-part ICP clouds.  This tool removes, per frame, every object-mask pixel whose observed
3D point (``depth/``) lies inside the tool volume: a cylinder of ``--radius-m`` around the
recorded TCP -> EE axis (``metadata/tcp_contact/tcp_contact.npz``), from ``--start-m`` behind the
calibrated tip up to the EE.  It never removes pixels in front of ``--start-m`` (default 0: the
calibrated tip), so object surface touching the tip is kept; the contact neighbourhood stays
ambiguous and is reported, not guessed.

The dataset is copied (``--output``); the input is never modified.  ``masks/`` and the masked
``images/`` are rewritten; everything else is copied unchanged.  Run it on a *prepared* dataset
(after preprocessing, before tracking) and then track the copy.

    python samuele/clean_robot_from_masks.py --dataset <prepared dataset> --output <copy> --radius-m 0.015

Report: ``<output>/metadata/robot_mask_cleaning.json`` with removed pixels per frame.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samuele.contact_moving_part_prompts import tool_volume_mask  # noqa: E402


def backproject(depth, K):
    h, w = depth.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    return np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], axis=-1)


def clean_frame(mask, depth, K, p_tcp_cam, p_ee_cam, radius_m, start_m):
    """Return (cleaned mask, removed boolean map).  Pixels without valid depth are kept."""
    points = backproject(depth, K)
    valid = np.isfinite(depth) & (depth > 0)
    inside = tool_volume_mask(points, p_tcp_cam, p_ee_cam, radius_m, start_m) & valid & mask
    return mask & ~inside, inside


def clean_dataset(dataset, output, radius_m, start_m, overwrite=False):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError("output must be a separate directory (the input dataset is never modified)")
    if (dataset / "tracking").exists():
        raise ValueError(f"{dataset} is already tracked; clean a prepared dataset before tracking")
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    tcp_path = dataset / "metadata/tcp_contact/tcp_contact.npz"
    if not tcp_path.is_file():
        raise FileNotFoundError(f"{tcp_path}: build the TCP artifact first (tcp_contact_evidence)")
    tcp = np.load(tcp_path)
    K = np.loadtxt(dataset / "cam_K.txt")
    stems = sorted(p.stem for p in (dataset / "masks").glob("*.png"))
    if len(stems) != len(tcp["T_base_tcp"]):
        raise ValueError(f"TCP artifact has {len(tcp['T_base_tcp'])} frames, dataset has {len(stems)}")
    T_cam_base = np.linalg.inv(np.asarray(tcp["T_base_camera"], dtype=np.float64))
    shutil.copytree(dataset, output)
    rows = []
    for i, stem in enumerate(stems):
        mask = cv2.imread(str(dataset / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
        with np.load(dataset / "depth" / f"{stem}.npz") as data:
            depth = data["depth"].astype(np.float64)
        p_tcp = (T_cam_base @ tcp["T_base_tcp"][i])[:3, 3]
        p_ee = (T_cam_base @ tcp["T_base_ee"][i])[:3, 3]
        cleaned, removed = clean_frame(mask, depth, K, p_tcp, p_ee, radius_m, start_m)
        cv2.imwrite(str(output / "masks" / f"{stem}.png"), cleaned.astype(np.uint8) * 255)
        image_path = output / "images" / f"{stem}.png"
        if image_path.is_file():
            image = cv2.imread(str(dataset / "images_ori" / f"{stem}.png"))
            image[~cleaned] = 255
            cv2.imwrite(str(image_path), image)
        rows.append(dict(stem=stem, mask_pixels=int(mask.sum()), removed_pixels=int(removed.sum())))
    report = dict(input=str(dataset), radius_m=radius_m, start_m=start_m, tcp_artifact=str(tcp_path),
                  rule="object-mask pixel removed iff its depth point lies in the cylinder of radius_m around "
                       "the TCP->EE axis, from start_m behind the calibrated tip to the EE",
                  removed_total=int(sum(r["removed_pixels"] for r in rows)), frames=rows)
    (output / "metadata/robot_mask_cleaning.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, required=True,
                        help="tool-volume radius; use the calibrated tool uncertainty, not a tuned value")
    parser.add_argument("--start-m", type=float, default=0.0,
                        help="start of the volume along the tip->EE axis (0 = calibrated tip; never negative here)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.start_m < 0:
        parser.error("--start-m must be >= 0: removing pixels in front of the calibrated tip would erase object surface")
    report = clean_dataset(args.dataset, args.output, args.radius_m, args.start_m, args.overwrite)
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
