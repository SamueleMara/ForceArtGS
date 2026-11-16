import importlib.util
from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / 'samuele/audits/robot_registration/rerun_multicam_pages.py'
if not SCRIPT.exists():
    SCRIPT = ROOT / 'scripts/audits/robot_registration/rerun_multicam_pages.py'
spec = importlib.util.spec_from_file_location('multicam_pose_comparison', SCRIPT)
comparison = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparison)


@dataclass
class Camera:
    name: str
    width: int
    height: int
    intrinsics: np.ndarray
    world_to_camera: np.ndarray


class PoseComparisonTests(unittest.TestCase):
    def fixture(self):
        G = np.eye(4); G[:3, :3] = Rotation.from_euler('xyz', [1, -2, 3], degrees=True).as_matrix()
        G[:3, 3] = [-.02, -.01, .035]
        cameras = []
        for i in range(3):
            W = np.eye(4); W[0, 3] = i * .01
            cameras.append(Camera(f'cam_{i:03d}', 12, 10, np.array([20., 20, 6, 5]), W))
        ob = SimpleNamespace(depth=np.ones((10, 12)), mask=np.ones((10, 12), bool), raw_valid=np.ones((10, 12), bool), rgb=np.ones((10, 12, 3)) * .5)
        data = SimpleNamespace(cameras=cameras, observation=lambda k, c: ob)
        return G, data

    def test_fresh_backprojection_equals_common_world_transform(self):
        G, data = self.fixture(); writer = comparison.exporter(comparison.DEFAULT_EXPORTER)
        new = SimpleNamespace(cameras=comparison.transformed_cameras(data.cameras, G), observation=data.observation)
        a, b = writer.frame_cloud(data, 0), writer.frame_cloud(new, 0)
        np.testing.assert_array_equal(a['_pid'], b['_pid'])
        np.testing.assert_allclose(b['_X'], a['_X'] @ G[:3, :3].T + G[:3, 3], atol=1e-7)
        self.assertLess(comparison.relative_pose_delta(data.cameras, new.cameras), 1e-12)
        self.assertEqual(a['rgb'], b['rgb'])

    def test_nearest_multiview_distances_unchanged_by_common_correction(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(120, 3)); cam = np.arange(120) % 3
        G, _ = self.fixture(); Y = X @ G[:3, :3].T + G[:3, 3]
        result = comparison.nearest_metrics(X, Y, cam)
        self.assertEqual(len(result), 6)
        self.assertLess(max(r['max_distance_change_m'] for r in result), 1e-12)

    def test_individual_camera_perturbation_is_detectable(self):
        X = np.r_[np.zeros((20, 3)), np.ones((20, 3)) * .005]
        cam = np.r_[np.zeros(20, int), np.ones(20, int)]
        Y = X.copy(); Y[cam == 1, 0] += .02
        result = comparison.nearest_metrics(X, Y, cam)
        self.assertGreater(max(r['max_distance_change_m'] for r in result), .01)

    def test_nonrigid_transform_rejected(self):
        G, data = self.fixture(); G[0, 0] *= 1.1
        with self.assertRaisesRegex(ValueError, 'proper rigid'):
            comparison.transformed_cameras(data.cameras, G)

    def test_page_retains_original_template_and_embeds_new_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = dict(frames=[0, 1], dataset='example')
            template = ('<body><h3 id="title"></h3><a href="index.html">all datasets</a><script>', ';document.getElementById(id);</script></body>')
            comparison.write_page(root, 'example', meta, template, 'corrected')
            loaded, _ = comparison.read_page(root / 'example.html')
            self.assertEqual(loaded, meta)
            self.assertIn('comparison-frame', (root / 'example.html').read_text())
            self.assertIn('href="../index.html"', (root / 'example.html').read_text())


if __name__ == '__main__':
    unittest.main()
