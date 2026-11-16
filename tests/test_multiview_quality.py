"""Quality-based multiview reference selection (preprocessing/multiview_quality.py)."""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from preprocessing.multiview_quality import camera_quality, select_reference, sign_test_p  # noqa: E402

K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
H, W, N_T = 240, 320, 12


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    S = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * S + (1 - math.cos(t)) * S @ S


def look_at(eye, target):
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    C = np.eye(4)
    C[:3, :3] = np.stack([x, y, z], 1)
    C[:3, 3] = eye
    return C  # camera-to-world (OpenCV)


def lid_motion(deg):
    """Lid hinged along world x at y=0, z=0.2."""
    T = np.eye(4)
    R = rot([1, 0, 0], deg)
    p = np.array([0, 0, 0.2])
    T[:3, :3] = R
    T[:3, 3] = p - R @ p
    return T


def scene_points():
    g = np.linspace(-0.15, 0.15, 120)
    X, Y = np.meshgrid(g, np.linspace(-0.3, 0.0, 120))
    base_top = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 0.2)], 1)
    base_front = np.stack([X.ravel(), np.full(X.size, -0.3), 0.2 - (Y.ravel() + 0.3) * 0.6], 1)  # vertical face
    X2, Y2 = np.meshgrid(g, np.linspace(0.0, 0.3, 120))
    lid = np.stack([X2.ravel(), Y2.ravel(), np.full(X2.size, 0.2)], 1)
    return np.vstack([base_top, base_front]), lid


def render(C, pts_by_label):
    """Z-buffered depth and label maps (2x2 splats) of labelled world points."""
    Tinv = np.linalg.inv(C)
    us, vs, zs, labs = [], [], [], []
    for lab, P in pts_by_label.items():
        pc = P @ Tinv[:3, :3].T + Tinv[:3, 3]
        pc = pc[pc[:, 2] > 0.05]
        u = np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
        for du in (0, 1):
            for dv in (0, 1):
                us.append(u + du), vs.append(v + dv), zs.append(pc[:, 2]), labs.append(np.full(len(u), lab))
    u, v, z, lab = map(np.concatenate, (us, vs, zs, labs))
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, lab = u[ok], v[ok], z[ok], lab[ok]
    depth = np.full((H, W), np.inf)
    np.minimum.at(depth, (v, u), z)
    label = np.zeros((H, W), np.uint8)
    front = z <= depth[v, u]
    label[v[front], u[front]] = lab[front]
    depth[~np.isfinite(depth)] = 0
    return depth, label


def write_camera(root, C, motions_cam, angles):
    base, lid = scene_points()
    for sub in ("depth", "masks", "mask_0", "mask_1"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    np.savetxt(root / "cam_K.txt", K)
    for t, deg in enumerate(angles):
        T = lid_motion(deg)
        depth, label = render(C, {1: base, 2: lid @ T[:3, :3].T + T[:3, 3]})
        np.savez(root / "depth" / f"{t:05d}.npz", depth=depth.astype(np.float32))
        cv2.imwrite(str(root / "masks" / f"{t:05d}.png"), (label > 0).astype(np.uint8) * 255)
        cv2.imwrite(str(root / "mask_0" / f"{t:05d}.png"), (label == 1).astype(np.uint8) * 255)
        cv2.imwrite(str(root / "mask_1" / f"{t:05d}.png"), (label == 2).astype(np.uint8) * 255)
    # global poses such that inv(G_t) @ G_0 equals the given camera-frame motion
    G = np.stack([np.linalg.inv(M) for M in motions_cam])
    np.save(root / "global_poses_1.npy", G)
    np.save(root / "global_poses_0.npy", np.stack([np.eye(4)] * len(angles)))


class MultiviewQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.angles = np.linspace(0, -30, N_T)
        cls.cal = {"cam_a": look_at(np.array([0.0, -0.9, 0.9]), [0, 0, 0.2]),
                   "cam_b": look_at(np.array([0.8, 0.2, 0.8]), [0, 0, 0.2])}
        true_world = [lid_motion(a) for a in cls.angles]
        spurious_world = [np.eye(4) for _ in cls.angles]
        for M, a in zip(spurious_world, cls.angles):       # translation instead of rotation
            M[:3, 3] = [0, 0, -0.002 * a]
        cls.paths = {}
        for name, C in cls.cal.items():
            for kind, world in (("true", true_world), ("spurious", spurious_world)):
                motions = [np.linalg.inv(C) @ Mw @ C for Mw in world]
                path = root / f"{name}_{kind}"
                write_camera(path, C, motions, cls.angles)
                cls.paths[(name, kind)] = path
        cls.stems = [f"{t:05d}" for t in range(N_T)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def quality(self, kinds):
        data = {name: self.paths[(name, kinds[name])] for name in self.cal}
        return camera_quality(data, self.stems, self.cal, {name: 1 for name in self.cal})

    def test_correct_motion_is_selected_over_a_larger_spurious_motion(self):
        report = self.quality({"cam_a": "true", "cam_b": "spurious"})
        ref, status, reasons = select_reference(report)
        self.assertEqual(status, "resolved")
        self.assertEqual(ref, "cam_a")
        self.assertEqual(report["cam_a"]["held_out"]["cam_b"]["verdict"], "support")
        self.assertNotEqual(report["cam_b"]["held_out"]["cam_a"]["verdict"], "support")
        self.assertAlmostEqual(report["cam_a"]["final_rotation_deg"], 30.0, delta=0.1)

    def test_unresolved_when_no_camera_is_supported(self):
        report = self.quality({"cam_a": "spurious", "cam_b": "spurious"})
        ref, status, _ = select_reference(report)
        self.assertIsNone(ref)
        self.assertEqual(status, "unresolved")

    def test_both_correct_cameras_agree(self):
        report = self.quality({"cam_a": "true", "cam_b": "true"})
        ref, status, _ = select_reference(report)
        self.assertEqual(status, "resolved")
        self.assertLess(report["cam_a"]["agreement"]["cam_b"]["relative_rotation_deg"], 1e-6)

    def test_contest_floor_excludes_the_views_own_points(self):
        # A view's own frame-0 points have zero residual in that view by construction; the floor must
        # come from the other camera's points, so frames where the two motions coincide are not counted.
        report = self.quality({"cam_a": "true", "cam_b": "true"})
        for view, res in report["cam_a"]["contests"]["cam_b"].items():
            self.assertGreater(res["frame0_floor_m"], 1e-4, view)
            self.assertEqual(res["frames"], 0, view)
            self.assertEqual(res["verdict"], "tie", view)

    def test_corrupted_extrinsic_detection_is_direction_dependent(self):
        # Point 3/F characterisation.  A 3 cm extrinsic error of cam_b along its optical axis raises the
        # frame-0 cross-view depth floor (a calibration symptom measured before anything moves) without
        # making the correct motion look wrong in cam_a's own depth.  The same error *in the image plane*
        # is nearly invisible to depth residuals on smooth surfaces: a documented blind spot.
        good = self.quality({"cam_a": "true", "cam_b": "true"})
        floor_good = good["cam_a"]["held_out"]["cam_b"]["frame0_floor_m"]
        data = {name: self.paths[(name, "true")] for name in self.cal}
        floors = {}
        for label, column in (("optical_axis", 2), ("image_plane_x", 0)):
            cal = dict(self.cal)
            bad = cal["cam_b"].copy()
            bad[:3, 3] += 0.03 * bad[:3, column]
            cal["cam_b"] = bad
            rep = camera_quality(data, self.stems, cal, {name: 1 for name in cal})
            floors[label] = rep["cam_a"]["held_out"]["cam_b"]["frame0_floor_m"]
            self.assertEqual(rep["cam_a"]["in_sample"]["verdict"], "support")
        self.assertGreater(floors["optical_axis"], floor_good + 0.02)
        self.assertLess(floors["image_plane_x"], floor_good + 0.01)

    def test_sign_test(self):
        self.assertAlmostEqual(sign_test_p(0, 10), 1.0)
        self.assertAlmostEqual(sign_test_p(10, 10), 1 / 1024)
        self.assertLess(sign_test_p(9, 11), 0.05)
        self.assertGreater(sign_test_p(8, 11), 0.05)


if __name__ == "__main__":
    unittest.main()
