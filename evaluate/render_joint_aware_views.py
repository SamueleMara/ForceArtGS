#!/usr/bin/env python3
"""Render, with the trained splatfacto-art model itself, every dataset view that is missing from
``<results>/eval_results`` (the held-out views of a joint-aware run), using the same call and file naming as
``SplatfactoArtModel.visualize_all_train_frames_after_train``. This makes the FreeArtGS comparison script
(evaluate/visualize_local_comparison.py) usable unchanged.

Cameras are the fixed physical calibration expressed in the art/blend frame B (``N_BW`` from
``twopart_blend/JOINT_AWARE_AUTHORITY.json``), OpenCV -> OpenGL on the right; the joint state is selected by
``metadata.frame_index = time_index`` exactly as during training. Nothing is optimised.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="splatfacto-art config.yml")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--blend", type=Path, required=True, help="twopart_blend dir with JOINT_AWARE_AUTHORITY.json")
    args = ap.parse_args()
    import torch
    from PIL import Image
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nerfstudio.utils.eval_utils import eval_setup

    N = np.asarray(json.loads((args.blend / "JOINT_AWARE_AUTHORITY.json").read_text())["N_BW"], float)
    cams = {c["camera_id"]: c for c in json.loads((args.dataset / "joint_authority" / "authority.json").read_text())
            ["cameras"].values()}
    _, pipeline, _, _ = eval_setup(args.config, test_mode="inference")
    model = pipeline.model
    model.eval()
    out = Path(args.config).parent / "eval_results"
    out.mkdir(exist_ok=True)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    done = 0
    for f in json.loads((args.dataset / "multiview.json").read_text())["frames"]:
        stem = Path(f["file_path"]).stem
        if (out / f"rgb_{stem}.png").is_file():
            continue
        c = cams[f["camera_name"]]
        c2w = (N @ np.asarray(c["T_WC"], float) @ flip)[:3]
        cam = Cameras(camera_to_worlds=torch.tensor(c2w, dtype=torch.float32)[None],
                      fx=torch.tensor([[f["fl_x"]]], dtype=torch.float32), fy=torch.tensor([[f["fl_y"]]], dtype=torch.float32),
                      cx=torch.tensor([[f["cx"]]], dtype=torch.float32), cy=torch.tensor([[f["cy"]]], dtype=torch.float32),
                      width=torch.tensor([[f["w"]]]), height=torch.tensor([[f["h"]]]),
                      camera_type=CameraType.PERSPECTIVE,
                      metadata={"frame_index": torch.tensor([[int(f["time_index"])]], dtype=torch.long)})
        with torch.no_grad():
            rgb = model.get_outputs_for_camera(camera=cam.to(model.device))["rgb"].detach().cpu().numpy().clip(0, 1)
        Image.fromarray((rgb * 255).astype("uint8")).save(out / f"rgb_{stem}.png")
        done += 1
    print(f"rendered {done} missing views into {out}")


if __name__ == "__main__":
    main()
