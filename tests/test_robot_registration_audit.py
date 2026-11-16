"""Synthetic known-transform, ambiguity, holdout and observability controls."""
import importlib.util
from pathlib import Path
import unittest
import json
import tempfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

root = Path(__file__).parents[1]
script = root / 'samuele/audits/robot_registration/audit_robot_registration.py'
if not script.exists():
    script = root / 'scripts/audits/robot_registration/audit_robot_registration.py'
spec = importlib.util.spec_from_file_location('robot_registration_audit', script)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
from background_stability import compare as compare_background
from refit_acquisition_group import load_group


def fixture(constant_axis=False):
    rng = np.random.default_rng(8)
    K = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
    cameras = []
    for c in range(3):
        C = np.eye(4); C[:3, 3] = [.12 * c, -.12 * c, -.8]
        cameras.append(dict(name=f'cam_{c:03d}', pose_matrix_4x4=C.tolist(), fx=500, fy=500, cx=320, cy=240))
    infos = {f'episode_{e}': dict(cameras=cameras) for e in range(3)}
    x = np.array([.01, -.02, .025, -.15, .06, .30])
    G = audit.transform(x)
    rows = []
    for e in range(3):
        for frame in range(48):
            p = rng.uniform(-.1, .1, 3)
            axis = np.array([1., 0, 0]) if constant_axis else Rotation.from_rotvec(rng.normal(0, .7, 3)).apply([0, 0, 1.])
            for c, cam in enumerate(cameras):
                uv = audit.project(p + np.outer([.075, .15], axis), G @ np.array(cam['pose_matrix_4x4']), K)
                line = np.cross(np.r_[uv[0], 1.], np.r_[uv[1], 1.]); line /= np.linalg.norm(line[:2])
                rows.append(dict(scene=f'episode_{e}', camera=c, camera_name=cam['name'], frame=frame,
                                 p=p.tolist(), axis=axis.tolist(), line=line.tolist()))
    return rows, infos, x


class RegistrationAuditTest(unittest.TestCase):
    def test_projection_and_left_transform_agree(self):
        rows, infos, x = fixture()
        self.assertLess(np.max(abs(audit.residuals(x, audit.arrays(rows, infos)))), 1e-10)

    def test_known_six_dof_correction_recovered(self):
        rows, infos, expected = fixture()
        fitted, ident = audit.solve(audit.balanced(rows), infos)
        np.testing.assert_allclose(fitted, expected, atol=1e-6)
        self.assertEqual(ident['rank'], 6)
        self.assertLess(audit.errors(fitted, rows, infos)['p95_px'], 1e-5)

    def test_robust_to_outlier_lines(self):
        rows, infos, expected = fixture()
        for row in rows[::15]:
            row['line'][2] += 30
        fitted, _ = audit.solve(rows, infos)
        self.assertLess(np.linalg.norm(fitted[:3] - expected[:3]), .003)
        self.assertLess(np.linalg.norm(fitted[3:] - expected[3:]) * .1, .002)

    def test_parallel_axes_leave_translation_unidentifiable(self):
        rows, infos, _ = fixture(constant_axis=True)
        _, ident = audit.solve(rows, infos)
        self.assertLess(ident['rank'], 6)

    def test_whole_episode_holdout(self):
        rows, infos, _ = fixture()
        result, _ = audit.fit_recording(rows, infos, bootstrap=0)
        self.assertEqual(result['heldout_episodes'], ['episode_1'])
        self.assertEqual(result['heldout']['after']['count'], 48 * 3)
        self.assertNotEqual(result['status'], 'CANDIDATE_PASSES_IMAGE_CHECKS')  # no uncertainty check

    def test_isolated_shaft_and_ambiguous_parallel_edges(self):
        image = np.full((200, 200, 3), 220, np.uint8)
        cv2.rectangle(image, (50, 96), (150, 104), (10, 10, 10), -1)
        fit, status = audit.detect_shaft(image, np.array([[60., 80.], [140., 80.]]))
        self.assertEqual(status, 'accepted')
        self.assertAlmostEqual(fit['center'][1], 100, delta=.5)
        cv2.rectangle(image, (50, 65), (150, 73), (10, 10, 10), -1)
        fit, status = audit.detect_shaft(image, np.array([[60., 80.], [140., 80.]]))
        self.assertIsNone(fit)
        self.assertEqual(status, 'too_few_unambiguous_cross_sections')

    def test_short_projections_rejected(self):
        _, status = audit.detect_shaft(np.zeros((20, 20, 3), np.uint8), np.array([[5., 5.], [8., 5.]]))
        self.assertEqual(status, 'projected_shaft_too_short_or_long')

    def test_pooled_holds_out_entire_recordings(self):
        rows, infos, _ = fixture()
        for name, info in infos.items():
            info['recording'] = name
        result, _ = audit.fit_pooled(rows, infos, bootstrap=0)
        self.assertEqual(len(result['leave_recording_out']), 3)
        for fold in result['leave_recording_out']:
            self.assertEqual(fold['after']['count'], 48 * 3)
            self.assertLess(fold['after']['p95_px'], 1e-4)
        self.assertNotEqual(result['status'], 'CANDIDATE_PASSES_IMAGE_CHECKS')

    def test_background_identity_and_camera_change_control(self):
        rng = np.random.default_rng(3)
        xy = rng.uniform(40, 400, (32, 2))
        keys = [cv2.KeyPoint(float(x), float(y), 3) for x, y in xy]
        moved = [cv2.KeyPoint(float(x + 12), float(y - 7), 3) for x, y in xy]
        desc = rng.uniform(0, 1, (32, 128)).astype(np.float32)
        result = compare_background(keys, desc, keys, desc)
        self.assertEqual(result['status'], 'CONSISTENT_STATIC_IMAGE_FEATURES')
        self.assertAlmostEqual(result['median_identity_displacement_px'], 0)
        result = compare_background(keys, desc, moved, desc)
        self.assertEqual(result['status'], 'BACKGROUND_CHANGE_OR_AMBIGUOUS_MATCHES')
        self.assertLess(result['heldout_homography_median_px'], 1e-3)

    def test_background_with_no_descriptors_has_no_false_success(self):
        self.assertEqual(compare_background([], None, [], None)['status'], 'INSUFFICIENT_DISTINCTIVE_BACKGROUND')

    def test_acquisition_group_excludes_earlier_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['final_red_fridge', 'final_black_laptop', 'old_fridge']:
                scene = root / (name + '_episode_000000'); scene.mkdir()
                (scene / 'inventory.json').write_text(json.dumps(dict(scene=scene.name, recording=name,
                    calibration_hash='shared', cameras=[dict(name='cam_000')])) )
                (scene / 'observations.jsonl').write_text(json.dumps(dict(scene=scene.name,
                    camera_name='cam_000', status='accepted')) + '\n')
            infos, rows, coverage, _ = load_group(root, 'final_')
            self.assertEqual(len(infos), 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(coverage), 2)
            self.assertTrue(all(i['recording'].startswith('final_') for i in infos.values()))
            with self.assertRaisesRegex(ValueError, 'No recordings'):
                load_group(root, 'missing_')

    def test_acquisition_group_rejects_mixed_calibration_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for idx in range(2):
                scene = root / f'final_{idx}'; scene.mkdir()
                (scene / 'inventory.json').write_text(json.dumps(dict(scene=scene.name, recording=scene.name,
                    calibration_hash=str(idx), cameras=[])))
                (scene / 'observations.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError, 'different supplied calibration'):
                load_group(root, 'final_')


if __name__ == '__main__':
    unittest.main()
