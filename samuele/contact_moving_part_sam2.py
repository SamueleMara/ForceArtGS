#!/usr/bin/env python3
"""Phase B: optional SAM2 moving-part proposals from per-frame contact prompts.

Thin adapter over ``sam2_point_guidance.predict_guided_mask`` (which enforces every
positive/negative point).  Prompts come from ``contact_prompts.json`` keyed by
(camera, source_frame_index), not from the per-camera ``read_guidance`` schema.
Proposals are clipped to the existing object foreground (robot pixels are outside it) and
written to ``contact_guidance/sam2_moving_proposals/cam_NNN/frame_XXXXXX.png``; object
masks are never modified.  A SAM2 proposal is only an *initial* motion-part proposal: it
may cover the whole articulated object and is validated by motion during tracking.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SAMUELE_DIR = Path(__file__).resolve().parent
if str(SAMUELE_DIR) not in sys.path:
    sys.path.insert(0, str(SAMUELE_DIR))

from sam2_point_guidance import predict_guided_mask  # noqa: E402


def frame_prompt(prompts, camera, source_frame_index):
    for frame in prompts["frames"]:
        if frame["camera"] == camera and int(frame["source_frame_index"]) == int(source_frame_index):
            return frame
    raise KeyError(f"No contact prompt for ({camera}, {source_frame_index})")


def read_contact_guidance(frame, object_mask):
    """(box, points, labels) for one frame; raises ValueError when no valid positive exists."""
    if frame["status"] != "valid" or not frame["positive_xy"]:
        raise ValueError(f"no_valid_positive_prompt: {frame.get('reason')}")
    object_mask = np.asarray(object_mask, dtype=bool)
    ys, xs = np.nonzero(object_mask)
    if not len(xs):
        raise ValueError("empty object foreground")
    H, W = object_mask.shape
    box = np.array([xs.min(), ys.min(), min(xs.max() + 1, W), min(ys.max() + 1, H)], dtype=np.float32)
    positive = np.asarray(frame["positive_xy"], dtype=np.float32).reshape(-1, 2)
    negative = np.asarray(frame.get("negative_robot_xy", []), dtype=np.float32).reshape(-1, 2)
    if not object_mask[positive[:, 1].astype(int), positive[:, 0].astype(int)].all():
        raise ValueError("positive prompt outside object foreground")
    points = np.concatenate([positive, negative]).astype(np.float32)
    labels = np.r_[np.ones(len(positive), dtype=np.int32), np.zeros(len(negative), dtype=np.int32)]
    return box, points, labels


def constrain_proposal(mask, object_mask, negative_xy=()):
    """Clip to object foreground; any robot negative remaining inside is an error."""
    out = (np.asarray(mask) > 0) & np.asarray(object_mask, dtype=bool)
    for x, y in negative_xy:
        if out[int(y), int(x)]:
            raise ValueError(f"SAM2 proposal includes robot pixel {(x, y)}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeart-camera-data", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, help="contact_prompts.json (default <camera>/metadata/contact_guidance)")
    parser.add_argument("--model-root", type=Path, required=True, help="Local SAM2 repository with checkpoints/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, help="Default <camera>/metadata/contact_guidance/sam2_moving_proposals")
    args = parser.parse_args(argv)
    camera_dir = args.freeart_camera_data.resolve()
    guidance = camera_dir / "metadata/contact_guidance"
    prompts = json.loads((args.prompts or guidance / "contact_prompts.json").read_text())
    out = args.output or guidance / "sam2_moving_proposals"
    report_path = (args.output.parent if args.output else guidance) / "sam2_report.json"
    try:
        import cv2
        from PIL import Image
        import torch
        sys.path.insert(0, str(args.model_root.expanduser().resolve()))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        checkpoint = args.model_root.expanduser().resolve() / "checkpoints/sam2.1_hiera_base_plus.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", str(checkpoint),
                                                  device=args.device))
    except Exception as error:  # clean opt-out: never fall back to all-foreground
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(dict(status="not_run", reason=f"SAM2 unavailable: {error}"), indent=2) + "\n")
        print(f"SAM2 unavailable, no proposals written: {error}", file=sys.stderr)
        return 2
    results = []
    for frame in prompts["frames"]:
        stem = frame["output_stem"]
        mask = cv2.imread(str(camera_dir / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
        try:
            box, points, labels = read_contact_guidance(frame, mask)
        except ValueError as error:
            results.append(dict(source_frame_index=frame["source_frame_index"], status="skipped", reason=str(error)))
            continue
        rgb = np.array(Image.open(camera_dir / "rgb" / f"{stem}.png").convert("RGB"))
        try:
            with torch.inference_mode():
                predictor.set_image(rgb)
                raw, score, _ = predict_guided_mask(predictor, box, points, labels)
            proposal = constrain_proposal(raw, mask, frame.get("negative_robot_xy", []))
        except ValueError as error:
            results.append(dict(source_frame_index=frame["source_frame_index"], status="failed", reason=str(error)))
            continue
        target = Path(out) / frame["camera"]
        target.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target / f"{frame['source_stem']}.png"), proposal.astype(np.uint8) * 255)
        results.append(dict(source_frame_index=frame["source_frame_index"], status="passed", sam_score=float(score),
                            proposal_pixels=int(proposal.sum()), object_pixels=int(mask.sum()),
                            proposal_object_fraction=float(proposal.sum() / max(1, mask.sum()))))
    report = dict(status="passed", frames=results, note="Proposals are initial moving-part hypotheses only.")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
