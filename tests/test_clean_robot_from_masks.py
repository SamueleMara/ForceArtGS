"""Upstream robot-tool removal from object masks (samuele/clean_robot_from_masks.py)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from samuele.clean_robot_from_masks import clean_dataset, clean_frame  # noqa: E402

K = np.array([[500.0, 0, 64], [0, 500.0, 48], [0, 0, 1]])
H, W = 96, 128


def scene():
    """Object plane at z=1.0 m; a tool along -x whose tip touches the plane region at (0, 0, 0.98)."""
    depth = np.full((H, W), 1.0)
    ys, xs = np.mgrid[0:H, 0:W]
    x = (xs - K[0, 2]) / K[0, 0]
    y = (ys - K[1, 2]) / K[1, 1]
    # tool: cylinder radius 8 mm around the segment from tip (0,0,0.98) to EE (-0.19,0,0.98), nearer than the plane
    tool = (np.abs(y * 0.98) < 0.008) & (x * 0.98 < 0.0) & (x * 0.98 > -0.19)
    depth[tool] = 0.98
    mask = np.ones((H, W), bool)  # object mask wrongly includes the tool
    return depth, mask, tool, np.array([0.0, 0.0, 0.98]), np.array([-0.19, 0.0, 0.98])


class CleanFrameTests(unittest.TestCase):
    def test_tool_pixels_removed_object_kept(self):
        depth, mask, tool, tip, ee = scene()
        cleaned, removed = clean_frame(mask, depth, K, tip, ee, radius_m=0.015, start_m=0.0)
        self.assertTrue(removed[tool].mean() > 0.95)          # the tool is removed
        self.assertFalse(removed[~tool].any())                 # the plane (2 cm behind the tool) is kept
        self.assertTrue(cleaned[~tool].all())

    def test_nothing_removed_in_front_of_the_calibrated_tip(self):
        depth, mask, tool, tip, ee = scene()
        # object surface touching the tip, *in front of* it (x > 0) at the tool's depth
        ys, xs = np.mgrid[0:H, 0:W]
        front = (np.abs((ys - K[1, 2]) / K[1, 1] * 0.98) < 0.008) & ((xs - K[0, 2]) / K[0, 0] * 0.98 > 0.001) \
            & ((xs - K[0, 2]) / K[0, 0] * 0.98 < 0.01)
        depth[front] = 0.98
        _, removed = clean_frame(mask, depth, K, tip, ee, radius_m=0.015, start_m=0.0)
        self.assertFalse(removed[front].any())

    def test_pixels_without_depth_are_kept(self):
        depth, mask, tool, tip, ee = scene()
        depth[tool] = 0.0
        cleaned, removed = clean_frame(mask, depth, K, tip, ee, radius_m=0.015, start_m=0.0)
        self.assertFalse(removed.any())
        self.assertTrue(cleaned.all())


class CleanDatasetTests(unittest.TestCase):
    def _dataset(self, root):
        depth, mask, tool, tip, ee = scene()
        for sub in ("masks", "images", "images_ori", "depth", "metadata/tcp_contact"):
            (root / sub).mkdir(parents=True)
        np.savetxt(root / "cam_K.txt", K)
        T_base_camera = np.eye(4)
        T_tcp = np.eye(4)
        T_tcp[:3, 3] = tip
        T_ee = np.eye(4)
        T_ee[:3, 3] = ee
        for i in range(2):
            cv2.imwrite(str(root / "masks" / f"{i:05d}.png"), mask.astype(np.uint8) * 255)
            cv2.imwrite(str(root / "images_ori" / f"{i:05d}.png"), np.full((H, W, 3), 100, np.uint8))
            cv2.imwrite(str(root / "images" / f"{i:05d}.png"), np.full((H, W, 3), 100, np.uint8))
            np.savez(root / "depth" / f"{i:05d}.npz", depth=depth.astype(np.float32))
        np.savez(root / "metadata/tcp_contact/tcp_contact.npz", T_base_camera=T_base_camera,
                 T_base_tcp=np.stack([T_tcp, T_tcp]), T_base_ee=np.stack([T_ee, T_ee]))
        return tool

    def test_copy_is_cleaned_and_input_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "prepared", Path(tmp) / "cleaned"
            tool = self._dataset(src)
            before = cv2.imread(str(src / "masks/00000.png"), 0).copy()
            report = clean_dataset(src, out, radius_m=0.015, start_m=0.0)
            self.assertTrue(np.array_equal(cv2.imread(str(src / "masks/00000.png"), 0), before))
            cleaned = cv2.imread(str(out / "masks/00000.png"), 0) > 127
            self.assertFalse(cleaned[tool].mean() > 0.05)
            self.assertTrue(cleaned[~tool].all())
            self.assertTrue((cv2.imread(str(out / "images/00000.png"))[tool] == 255).all())
            saved = json.loads((out / "metadata/robot_mask_cleaning.json").read_text())
            self.assertEqual(saved["removed_total"], report["removed_total"])
            self.assertGreater(report["removed_total"], 0)

    def test_refuses_tracked_dataset_and_in_place_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "prepared"
            self._dataset(src)
            with self.assertRaises(ValueError):
                clean_dataset(src, src / "inside", radius_m=0.015, start_m=0.0)
            (src / "tracking").mkdir()
            with self.assertRaises(ValueError):
                clean_dataset(src, Path(tmp) / "out", radius_m=0.015, start_m=0.0)


if __name__ == "__main__":
    unittest.main()
