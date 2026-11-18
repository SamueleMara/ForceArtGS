"""Candidate-specific FreeArtGS dataset with exact joint-derived virtual cameras (end-to-end spec G2, section 4).

Independent of ``prepare_multiview``: no ICP poses, no reference-camera selection, no part-ID swapping.
part 0 is always the static part (physical cameras ``C_S = T_WC``), part 1 always the moving part (virtual
cameras ``C_M = A_H(q_t)^-1 T_WC``). Masks come only from a dense-support export whose authority id matches.
Held-out images stay in every transforms file (so time indices stay contiguous and every row keeps its q)
but are listed in ``eval_filenames`` and never in ``train_filenames``.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimotion_2part.joint_discovery.integration_contract import (  # noqa: E402
    AuthorityError, JointAuthority, atomic_write_json, sha256_file)

SUPPORT_NAMES = ("mask_0", "mask_1", "mask_unknown", "masks", "ignore_mask")


def _read_mask(path):
    import cv2
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def build_joint_aware_dataset(scene_root, authority_dir, dense_dir, output, cameras, heldout_times=(), *,
                              overwrite=False, tol=1e-9) -> dict:
    import cv2
    scene = Path(scene_root)
    out = Path(output)
    a = JointAuthority.load(authority_dir)
    aid = json.loads((Path(authority_dir) / "authority.json").read_text())["authority_id"]
    lab = json.loads((Path(dense_dir) / "label_manifest.json").read_text())
    if lab["authority_id"] != aid:
        raise AuthorityError(f"dense support {dense_dir} was built from authority {lab['authority_id']}, not {aid}")
    if a.reference_time_index != 0:
        raise AuthorityError("datasets for the legacy part/blend stages need reference_time_index == 0")
    if out.exists():
        if not overwrite:
            prev = out / "joint_aware_manifest.json"
            if prev.is_file() and json.loads(prev.read_text()).get("authority_id") == aid:
                return json.loads(prev.read_text())
            raise FileExistsError(f"{out} exists with different content; use a new version directory")
        shutil.rmtree(out)
    heldout = set(int(t) for t in heldout_times)
    if a.T - 1 in heldout:
        raise ValueError("the last time index must stay in training (legacy art parser sizes q by max train time)")
    for d in ("images", "images_ori", "depth", "masks", "mask_0", "mask_1", "mask_unknown", "ignore_mask",
              "supervision_0", "supervision_1", "part_probability"):
        (out / d).mkdir(parents=True, exist_ok=True)
    frames, rows = [], [[], []]
    train_names, eval_names = [], []
    max_err = 0.0
    for t in range(a.T):
        A = a.transform(t)
        for cid in cameras:
            cam = a.cameras[cid]
            stem = f"frame_{int(a.rgbd_frame_index[t]):06d}"
            name = f"{len(frames):05d}"
            img = cv2.imread(str(scene / "rgb" / cid / f"{stem}.png"))
            dep = np.load(scene / "depth_npy" / cid / f"{stem}.npy").astype(np.float32)
            if img is None or img.shape[:2] != (cam.height, cam.width) or dep.shape != (cam.height, cam.width):
                raise ValueError(f"{cid}/{stem}: missing or size-mismatched RGB/depth")
            dep[~np.isfinite(dep) | (dep < 0)] = 0.0
            sup = {k: _read_mask(Path(dense_dir) / cid / k / f"{stem}.png") for k in SUPPORT_NAMES}
            if any(m.shape != (cam.height, cam.width) for m in sup.values()):
                raise ValueError(f"{cid}/{stem}: dense support resolution mismatch")
            S, M, U, obj, ign = sup["mask_0"], sup["mask_1"], sup["mask_unknown"], sup["masks"], sup["ignore_mask"]
            fg = obj & ~ign
            if (S & M).any() or ((S | M | U) != fg).any():
                raise ValueError(f"{cid}/{stem}: S/M/U do not partition the valid object pixels")
            negative = ~obj & ~ign                       # verified visible absence of the object
            whitened = img.copy()
            whitened[~fg] = 255
            cv2.imwrite(str(out / "images" / f"{name}.png"), whitened)
            cv2.imwrite(str(out / "images_ori" / f"{name}.png"), img)
            np.savez_compressed(out / "depth" / f"{name}.npz", depth=dep)
            for k, m in (("masks", fg), ("mask_0", S), ("mask_1", M), ("mask_unknown", U), ("ignore_mask", ign),
                         ("supervision_0", S | negative), ("supervision_1", M | negative)):
                cv2.imwrite(str(out / k / f"{name}.png"), m.astype(np.uint8) * 255)
            shutil.copy2(Path(dense_dir) / cid / "part_probability" / f"{stem}.npz", out / "part_probability" / f"{name}.npz")
            C_S = np.asarray(cam.T_WC, float)
            C_M = np.linalg.inv(A) @ C_S
            max_err = max(max_err, float(np.abs(C_S @ np.linalg.inv(C_M) - A).max()))
            row = dict(file_path=f"images/{name}.png", time_index=t, camera_name=cid, camera_id=cid,
                       image_id=name, source_stem=stem, rgbd_frame_index=int(a.rgbd_frame_index[t]),
                       source_frame_index=int(a.source_frame_index[t]), timestamp_s=float(a.timestamp_s[t]),
                       fl_x=float(cam.K[0, 0]), fl_y=float(cam.K[1, 1]), cx=float(cam.K[0, 2]), cy=float(cam.K[1, 2]),
                       w=int(cam.width), h=int(cam.height), physical_camera_sha=cam.sha(),
                       valid_rgb=True, valid_depth=bool((dep > 0).any()), split="eval" if t in heldout else "train")
            frames.append(row)
            for part, pose in ((0, C_S), (1, C_M)):
                rows[part].append(dict(row, transform_matrix=pose.tolist(), mask_path=f"mask_{part}/{name}.png",
                                       depth_path=f"depth/{name}.npz",
                                       supervision_path=f"supervision_{part}/{name}.png",
                                       pose_kind="physical_camera" if part == 0 else "virtual_part_camera",
                                       part_pixels=int((S if part == 0 else M).sum())))
            (eval_names if t in heldout else train_names).append(f"images/{name}.png")
    if max_err > tol:
        raise AuthorityError(f"C_S C_M^-1 != A(t) (max error {max_err:.2e})")
    for part in (0, 1):
        doc = dict(camera_model="OPENCV", pose_convention="opencv_c2w", part=part,
                   part_role="static" if part == 0 else "moving", frames=rows[part],
                   train_filenames=train_names, eval_filenames=eval_names, val_filenames=eval_names,
                   test_filenames=eval_names, authority_id=aid,
                   camera_optimizer="off (fixed calibrated physical cameras)")
        atomic_write_json(out / f"transforms_{part}.json", doc)
    first = a.cameras[cameras[0]].K
    np.savetxt(out / "cam_K.txt", np.asarray(first))     # legacy compatibility only; rows carry per-view K
    (out / "joint_authority").mkdir(exist_ok=True)
    for f in ("authority.json", "joint_states.npz"):
        shutil.copy2(Path(authority_dir) / f, out / "joint_authority" / f)
    manifest = dict(schema="forceartgs.joint_aware_dataset/1", source_scene=str(scene), cameras=list(cameras),
                    authority_id=aid, dense_support=str(dense_dir),
                    dense_manifest_sha256=sha256_file(Path(dense_dir) / "label_manifest.json"),
                    joint_type=a.joint_type, num_timestamps=a.T, heldout_times=sorted(heldout),
                    n_train_images=len(train_names), n_eval_images=len(eval_names),
                    virtual_camera_max_error=max_err, motion_part=1, reference_time_index=a.reference_time_index,
                    initialization_method="joint_aware_authority (no ICP, no reference-camera propagation)",
                    frames=frames)
    atomic_write_json(out / "multiview.json", dict(manifest, part_assignments="part0=static, part1=moving (fixed)"))
    atomic_write_json(out / "frame_map.json", dict(rgbd_frame_index=a.rgbd_frame_index.tolist(),
                                                   source_frame_index=a.source_frame_index.tolist(),
                                                   timestamp_s=a.timestamp_s.tolist(),
                                                   reference_time_index=a.reference_time_index))
    atomic_write_json(out / "joint_aware_manifest.json", {k: v for k, v in manifest.items() if k != "frames"})
    return manifest
