"""Point 1 conventions: fixed physical cameras, virtual part cameras, and the ICP pose direction.

* ``calibration_poses`` returns OpenCV camera-to-world from COLMAP world-to-camera (qvec, tvec).
* ``camera_in_part_frame`` gives a *virtual* camera in the moving part's frame-0 canonical frame
  that reproduces every physical camera's observations, provided ``global_poses[t]`` maps
  frame-t camera points to frame-0 camera points (H1).  Nonzero calibrations and translations.
* The real ICP front end (``icp/color_icp_gpu.build_pose_graph``) produces H1 on a synthetic
  moving rigid cloud with a known motion (CUDA required; skipped otherwise).
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from preprocessing.prepare_multiview import calibration_poses, camera_in_part_frame  # noqa: E402


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    S = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * S + (1 - np.cos(t)) * S @ S


def se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def hinge_motion(deg, axis=(0.3, 1.0, 0.2), pivot=(0.4, -0.2, 1.1), extra=(0.01, -0.02, 0.005)):
    R = rot(axis, deg)
    p = np.array(pivot)
    return se3(R, p - R @ p + np.array(extra) * deg / 10.0)     # rotation plus translation


def rotmat2qvec(R):
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


CAMS = {"cam_a": se3(rot([1, 0.2, 0], 160), [0.3, -0.4, 0.2]),
        "cam_b": se3(rot([0.2, 1, 0.3], 140), [-0.5, 0.1, 0.7])}


class VirtualCameraConventionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X0 = rng.uniform([-0.1, -0.1, 0.9], [0.1, 0.1, 1.2], (200, 3))   # world points at t = 0
        self.A = [hinge_motion(d) for d in np.linspace(0, 25, 6)]           # world motion, frame 0 -> t

    def observe(self, C, t):
        """Camera-frame coordinates of the moved points seen by physical camera C (c2w)."""
        Xt = self.X0 @ self.A[t][:3, :3].T + self.A[t][:3, 3]
        Cinv = np.linalg.inv(C)
        return Xt @ Cinv[:3, :3].T + Cinv[:3, 3]

    def h1_poses(self, C):
        """H1: G_t maps frame-t camera points of camera C to its frame-0 camera points."""
        return np.stack([np.linalg.inv(C) @ np.linalg.inv(A) @ C for A in self.A])

    def test_virtual_camera_reproduces_all_physical_observations(self):
        for ref in CAMS:
            G = self.h1_poses(CAMS[ref])
            for cam, C in CAMS.items():
                V = camera_in_part_frame(C, CAMS[ref], G)
                for t in range(len(self.A)):
                    Vinv = np.linalg.inv(V[t])
                    seen_virtual = self.X0 @ Vinv[:3, :3].T + Vinv[:3, 3]
                    np.testing.assert_allclose(seen_virtual, self.observe(C, t), atol=1e-10)
                np.testing.assert_allclose(V[0], C, atol=1e-12)               # physical pose at frame 0

    def test_inverse_convention_would_be_wrong(self):
        G_inverse = np.stack([np.linalg.inv(g) for g in self.h1_poses(CAMS["cam_a"])])
        V = camera_in_part_frame(CAMS["cam_b"], CAMS["cam_a"], G_inverse)
        Vinv = np.linalg.inv(V[-1])
        err = np.abs(self.X0 @ Vinv[:3, :3].T + Vinv[:3, 3] - self.observe(CAMS["cam_b"], len(self.A) - 1)).max()
        self.assertGreater(err, 0.05)

    def test_calibration_poses_are_opencv_camera_to_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)
            (scene / "colmap/sparse/0").mkdir(parents=True)
            lines = []
            for i, (name, C) in enumerate(CAMS.items(), 1):
                w2c = np.linalg.inv(C)
                q, t = rotmat2qvec(w2c[:3, :3]), w2c[:3, 3]
                for frame in (0, 5):
                    lines += [f"{i * 10 + frame} {' '.join(map(repr, q))} {' '.join(map(repr, t))} {i} {name}/frame_{frame:06d}.png", ""]
            (scene / "colmap/sparse/0/images.txt").write_text("\n".join(lines) + "\n")
            poses = calibration_poses(scene)
            for name, C in CAMS.items():
                np.testing.assert_allclose(poses[name], C, atol=1e-9)


class IcpPoseDirectionTest(unittest.TestCase):
    def test_build_pose_graph_global_poses_map_frame_t_to_frame_0(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("ICP front end is CUDA-only")
        import open3d as o3d
        import icp.color_icp_gpu as icp
        from utils.pipeline_config import resolve_icp
        icp.ICP_T_DEVICE, icp.TORCH_DEVICE = icp.resolve_gpu_devices("cuda")
        rng = np.random.default_rng(1)
        # asymmetric textured "box corner" so the registration is well conditioned
        faces = [np.c_[rng.uniform(0, .2, 3000), rng.uniform(0, .15, 3000), np.zeros(3000)],
                 np.c_[np.zeros(2500), rng.uniform(0, .15, 2500), rng.uniform(0, .1, 2500)],
                 np.c_[rng.uniform(0, .2, 2000), np.zeros(2000), rng.uniform(0, .1, 2000)]]
        P0 = np.vstack(faces) + np.array([-0.1, -0.05, 0.8])
        colors = np.c_[0.5 + 0.5 * np.sin(40 * P0[:, 0]), 0.5 + 0.5 * np.cos(30 * P0[:, 1]), 0.5 + 0.5 * np.sin(50 * P0[:, 2])]
        motions = [se3(rot([0.2, 1, 0.1], 2.0 * k), [0.003 * k, -0.002 * k, 0.001 * k]) for k in range(8)]  # camera frame, 0 -> t
        clouds = []
        for M in motions:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(P0 @ M[:3, :3].T + M[:3, 3])
            pc.colors = o3d.utility.Vector3dVector(colors)
            clouds.append(pc)
        args = types.SimpleNamespace(**resolve_icp({}))
        args.voxel_size = float(icp.compute_adaptive_voxel_size(clouds))
        args.local_map_window = max(1, int(args.local_map_window))
        tensor = [icp._to_tensor_pcd(c, icp.ICP_T_DEVICE) for c in clouds]
        _, poses, valid = icp.build_pose_graph(args, clouds, tensor_point_clouds=tensor)
        self.assertTrue(all(valid))
        for M, G in zip(motions, poses):
            # H1: G maps frame-t points back to frame 0, i.e. G = inv(M)
            Pt = P0 @ M[:3, :3].T + M[:3, 3]
            back = Pt @ np.asarray(G)[:3, :3].T + np.asarray(G)[:3, 3]
            self.assertLess(np.median(np.linalg.norm(back - P0, axis=1)), 2e-3)


if __name__ == "__main__":
    unittest.main()
