"""Synthetic converted-real-scene fixtures for TCP/contact tests (no real data needed)."""
import json
from pathlib import Path

import numpy as np

K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
SIZE = (64, 48)


def look_at_camera(position, target, up=(0, 0, 1)):
    """OpenCV optical camera->base: +Z forward, +X right, +Y down."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    z = target - position
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = position
    return T


def make_scene(root, *, num_output=12, stride_source=2, first_source=10, cameras=("cam_000", "cam_001"),
               tcp_positions=None, with_ee=True, T_ee_tcp=None):
    """Write a minimal converted real scene + robot_poses/poses_rgbd.npz.  Returns metadata."""
    root = Path(root)
    (root / "metadata/robot_poses").mkdir(parents=True, exist_ok=True)
    outputs = list(range(num_output))
    sources = [first_source + stride_source * i for i in outputs]
    poses = {"cam_000": look_at_camera([0.0, -1.0, 0.0], [0.0, 0.0, 0.0]),
             "cam_001": look_at_camera([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])}
    entries = []
    for i, name in enumerate(cameras):
        entries.append(dict(id=i + 1, name=name, source_camera_name=f"cam_{i}", width=SIZE[0], height=SIZE[1],
                            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], pose_matrix_4x4=poses[name].tolist(),
                            pose_convention="opencv_optical_camera_to_world", coordinate_frame="base_link"))
    (root / "metadata/cameras.json").write_text(json.dumps(dict(cameras=entries)))
    (root / "metadata/real_conversion.json").write_text(json.dumps(dict(
        episode=0, source_frames=sources, output_frames=outputs, rectified_K=K.tolist())))
    if tcp_positions is None:
        tcp_positions = np.stack([np.linspace(-0.1, 0.1, num_output), np.zeros(num_output), np.zeros(num_output)], 1)
    T_ee_tcp = np.eye(4) if T_ee_tcp is None else np.asarray(T_ee_tcp)
    tcp = np.repeat(np.eye(4)[None], num_output, axis=0)
    tcp[:, :3, 3] = tcp_positions
    arrays = dict(frame_index=np.asarray(outputs), source_frame_index=np.asarray(sources),
                  timestamp_s=np.asarray(sources) / 25.0, tcp_transform=tcp, tcp_position_m=tcp[:, :3, 3])
    if with_ee:
        arrays["ee_transform"] = tcp @ np.linalg.inv(T_ee_tcp)
    np.savez_compressed(root / "metadata/robot_poses/poses_rgbd.npz", **arrays)
    (root / "metadata/robot_poses/manifest.json").write_text(json.dumps(dict(
        recorded_frame="ee", ee_to_tcp=T_ee_tcp.tolist())))
    (root / "metadata/robot_tool.json").write_text(json.dumps(dict(tool="pointed_tip", recorded_pose_frame="ee",
                                                                   T_ee_tcp=T_ee_tcp.tolist())))
    return dict(sources=sources, outputs=outputs, poses=poses, tcp=tcp)


def make_freeart_camera(root, scene, camera, selected_outputs, depth=None, mask=None):
    """Minimal converted FreeArt camera folder referencing ``scene``."""
    root = Path(root)
    for sub in ("metadata", "rgb", "depth", "masks"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    np.savetxt(root / "cam_K.txt", K)
    frame_map = []
    for i, output in enumerate(selected_outputs):
        stem = f"{i:05d}"
        frame_map.append(dict(output_stem=stem, source_stem=f"frame_{output:06d}", source_index=output))
        if depth is not None:
            np.savez(root / "depth" / f"{stem}.npz", depth=np.asarray(depth(i) if callable(depth) else depth,
                                                                    dtype=np.float32))
        if mask is not None:
            import cv2
            m = mask(i) if callable(mask) else mask
            cv2.imwrite(str(root / "masks" / f"{stem}.png"), np.asarray(m, dtype=np.uint8) * 255)
    (root / "metadata/freeart_conversion.json").write_text(json.dumps(dict(
        scene_root=str(Path(scene).resolve()), selected_camera=camera, frame_map=frame_map, frame_stride=1,
        max_frames=None, frame_intervals=[])))
    return root
