import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

root = Path(__file__).parents[1]
script = root / 'samuele/audits/robot_registration/apply_robot_base_registration.py'
if not script.exists():
    script = root / 'scripts/audits/robot_registration/apply_robot_base_registration.py'
spec = importlib.util.spec_from_file_location('apply_registration', script)
app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
sys.path.insert(0, '/home/samuelemara/Joint3DGS/Sapien/scripts')
import read_write_model as model
sys.path.insert(0, str(script.parent))
from verify_applied_registration import project


def make_ply(path, xyz, ascii=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.empty(len(xyz), dtype=[(n, '<f4') for n in ['x', 'y', 'z', 'nx', 'ny', 'nz']] + [('red', 'u1')])
    for k, name in enumerate(['x', 'y', 'z']):
        array[name] = np.asarray(xyz)[:, k]
    array['nx'] = 1; array['ny'] = 0; array['nz'] = 0; array['red'] = 127
    with path.open('wb') as f:
        f.write(('ply\nformat ' + ('ascii' if ascii else 'binary_little_endian') + ' 1.0\nelement vertex ' + str(len(xyz)) + '\n' + ''.join(f'property float {n}\n' for n in ['x', 'y', 'z', 'nx', 'ny', 'nz']) + 'property uchar red\nend_header\n').encode())
        if ascii:
            np.savetxt(f, np.column_stack([array[n] for n in array.dtype.names]), fmt=['%.9g'] * 6 + ['%d'])
        else:
            array.tofile(f)


def load_ply(path):
    with path.open('rb') as f:
        _, fmt, n, dtype = app.ply_header(f)
        return np.atleast_1d(np.loadtxt(f, dtype=dtype, max_rows=n)) if fmt == 'ascii' else np.fromfile(f, dtype=dtype, count=n)


class ApplyRegistrationTest(unittest.TestCase):
    def test_camera_projection_moves_without_changing_robot_positions(self):
        P = np.array([[.1, 0., 1.]])
        original = P.copy(); C = np.eye(4); C[:3, 3] = [.02, 0, .1]
        K = np.array([[100., 0, 32], [0, 100., 24], [0, 0, 1]])
        np.testing.assert_allclose(project(P, np.eye(4), K), [[42, 24]], atol=1e-12)
        np.testing.assert_allclose(project(P, C, K), [[32 + 100 * .08 / .9, 24]], atol=1e-12)
        np.testing.assert_array_equal(P, original)

    def fixture(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name); scene = root / 'real/final_test_episode_000000'; meta = scene / 'metadata'; (meta / 'raw').mkdir(parents=True)
        G = np.eye(4); G[:3, :3] = Rotation.from_euler('xyz', [1, -2, 3], degrees=True).as_matrix(); G[:3, 3] = [.01, -.02, .035]
        cams, metadata = {}, []
        for c in range(1, 4):
            C = np.eye(4); C[:3, 3] = [c * .1, -.1, .2]
            cams[c] = model.Camera(c, 'PINHOLE', 640, 480, np.array([500., 500, 320, 240]))
            metadata.append(dict(name=f'cam_{c - 1:03d}', pose_matrix_4x4=C.tolist()))
        app.write_json(meta / 'cameras.json', dict(cameras=metadata, calibration_status='original'))
        app.write_json(meta / 'real_conversion.json', dict(output_frames=[0, 1], calibration_status='original'))
        app.write_json(meta / 'raw/camera_calibration.json', dict(status='original', cameras={f'cam_{c}': dict(T_base_color_optical=m['pose_matrix_4x4']) for c, m in enumerate(metadata)}))
        (meta / 'raw/episode_000000.parquet').write_bytes(b'robot native measurements: never modified')
        (meta / 'robot_tool.json').write_text('{"tool":"pointed", "local_z":0.1876174}')
        xyz = np.array([[0., 0, 1], [.1, .2, 1.2]])
        for prefix, frame in [('colmap', 0), ('colmap_final', 1)]:
            for sparse in ['sparse', 'sparse_unmasked']:
                folder = scene / prefix / sparse / '0'; folder.mkdir(parents=True)
                images = {}
                for c, cam in cams.items():
                    C = np.array(metadata[c - 1]['pose_matrix_4x4']); local = xyz - C[:3, 3]
                    uv = local[:, :2] / local[:, 2:] * 500 + [320, 240]
                    images[c] = model.Image(c, np.array([1., 0, 0, 0]), -C[:3, 3], c, f'cam_{c - 1:03d}/frame_{frame:06d}.png', uv, np.array([1, 2]))
                pts = {i + 1: model.Point3D(i + 1, p, np.array([127, 100, 80]), 0., np.array([1, 2, 3]), np.array([i, i, i])) for i, p in enumerate(xyz)}
                model.write_model(cams, images, pts, str(folder), ext='.bin'); model.write_model(cams, images, pts, str(folder), ext='.txt')
                make_ply(folder / 'pointcloud.ply', xyz)
        make_ply(scene / 'pointcloud/views/start/cam_000.ply', xyz, ascii=True)
        report = root / 'report'; report.mkdir()
        info = dict(scene=scene.name, cameras=metadata)
        originals = {str(p.relative_to(scene)): app.sha(p) for p in scene.glob('**/*') if p.is_file()}
        return scene, report, info, G, originals

    def test_apply_models_metadata_ply_normals_and_restore_exactly(self):
        scene, report, info, G, before = self.fixture()
        result = app.apply_scene((str(scene), str(report), info, G, dict(test=True)))
        self.assertEqual(result['status'], 'applied')
        self.assertEqual(len(result['model_checks']), 4)
        current = json.loads((scene / 'metadata/cameras.json').read_text())
        for a, b in zip(current['cameras'], info['cameras']):
            np.testing.assert_allclose(a['pose_matrix_4x4'], G @ np.array(b['pose_matrix_4x4']), atol=1e-12)
        cloud = load_ply(scene / 'pointcloud/views/start/cam_000.ply')
        np.testing.assert_allclose([cloud['nx'][0], cloud['ny'][0], cloud['nz'][0]], G[:3, 0], atol=1e-7)
        self.assertTrue((cloud['red'] == 127).all())
        self.assertEqual(app.sha(scene / 'metadata/raw/episode_000000.parquet'), before['metadata/raw/episode_000000.parquet'])
        marker = json.loads((scene / app.MARKER).read_text())
        app.restore_scene(scene, report / 'backups' / scene.name, marker['after_sha256'])
        for rel, digest in before.items():
            self.assertEqual(app.sha(scene / rel), digest)
        self.assertFalse((scene / app.MARKER).exists())
        self.assertFalse((scene / app.ACTIVE_NATIVE).exists())

    def test_idempotent_and_modified_file_guard(self):
        scene, report, info, G, _ = self.fixture(); job = (str(scene), str(report), info, G, {})
        app.apply_scene(job); marker_before = (scene / app.MARKER).read_bytes()
        self.assertEqual(app.apply_scene(job)['status'], 'already_applied_verified')
        self.assertEqual((scene / app.MARKER).read_bytes(), marker_before)
        (scene / 'pointcloud/views/start/cam_000.ply').write_bytes(b'changed externally')
        with self.assertRaisesRegex(ValueError, 'changed since registration'):
            app.apply_scene(job)

    def test_interrupted_restore_preserves_unrelated_edits(self):
        scene, report, info, G, _ = self.fixture()
        app.apply_scene((str(scene), str(report), info, G, {}))
        (scene / app.MARKER).unlink()
        target = scene / 'pointcloud/views/start/cam_000.ply'
        target.write_bytes(b'external user edit')
        with self.assertRaisesRegex(ValueError, 'unrelated edits'):
            app.restore_scene(scene, report / 'backups' / scene.name)
        self.assertEqual(target.read_bytes(), b'external user edit')

    def test_untracked_native_calibration_is_not_overwritten(self):
        scene, report, info, G, _ = self.fixture()
        (scene / app.ACTIVE_NATIVE).write_text('existing independent calibration')
        with self.assertRaisesRegex(ValueError, 'Untracked active'):
            app.apply_scene((str(scene), str(report), info, G, {}))
        self.assertEqual((scene / app.ACTIVE_NATIVE).read_text(), 'existing independent calibration')

    def test_failed_publish_rolls_back_all_files(self):
        scene, report, info, G, before = self.fixture()
        original = app.publish; counter = [0]
        def fail_once(source, target):
            counter[0] += 1
            if counter[0] == 3:
                raise OSError('injected publication failure')
            return original(source, target)
        with patch.object(app, 'publish', side_effect=fail_once):
            with self.assertRaisesRegex(OSError, 'injected'):
                app.apply_scene((str(scene), str(report), info, G, {}))
        for rel, digest in before.items():
            self.assertEqual(app.sha(scene / rel), digest)
        self.assertFalse((scene / app.BUSY).exists())

    def test_database_pose_priors_and_relative_matches(self):
        _, report, _, G, _ = self.fixture(); dbfile = report / 'database.db'
        with sqlite3.connect(dbfile) as db:
            db.execute('CREATE TABLE images(image_id INTEGER,prior_qw REAL,prior_qx REAL,prior_qy REAL,prior_qz REAL,prior_tx REAL,prior_ty REAL,prior_tz REAL)')
            db.execute('INSERT INTO images VALUES(1,1,0,0,0,0,0,0)')
            db.execute('CREATE TABLE matches(data BLOB)'); db.execute("INSERT INTO matches VALUES(X'010203')")
        self.assertEqual(app.transform_database(dbfile, G), 1)
        with sqlite3.connect(dbfile) as db:
            values = db.execute('SELECT prior_qw,prior_qx,prior_qy,prior_qz,prior_tx,prior_ty,prior_tz FROM images').fetchone()
            self.assertEqual(db.execute('SELECT data FROM matches').fetchone()[0], b'\x01\x02\x03')
        q, t = app.change_pose([1, 0, 0, 0], [0, 0, 0], G)
        np.testing.assert_allclose(values, q + t, atol=1e-12)

    def test_bad_endpoint_rejected_without_changing_source(self):
        scene, report, info, G, before = self.fixture()
        p = scene / 'metadata/real_conversion.json'; app.write_json(p, dict(output_frames=[0, 2]))
        before['metadata/real_conversion.json'] = app.sha(p)
        with self.assertRaisesRegex(ValueError, 'endpoint membership'):
            app.apply_scene((str(scene), str(report), info, G, {}))
        for rel, digest in before.items():
            self.assertEqual(app.sha(scene / rel), digest)

    def test_modern_rgb_model_frames_yaml_and_priors_stay_consistent(self):
        scene, report, info, G, _ = self.fixture()
        rgb = scene / 'colmap_rgb'; folder = rgb / 'sparse/0'; folder.mkdir(parents=True)
        rec = pycolmap.Reconstruction(str(scene / 'colmap/sparse/0'))
        rec.write_binary(str(folder)); rec.write_text(str(folder))
        rig_hash, frame_hash = app.sha(folder / 'rigs.bin'), app.sha(folder / 'frames.bin')
        image = rec.images[1]; W = image.cam_from_world().matrix()
        q = Rotation.from_matrix(W[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist()
        (rgb / 'cameras.yaml').write_text(yaml.safe_dump(dict(extrinsics={image.name: dict(qvec=q, tvec=W[:3, 3].tolist())})))
        with sqlite3.connect(rgb / 'database.db') as db:
            db.execute('CREATE TABLE images(image_id INTEGER,prior_qw REAL,prior_qx REAL,prior_qy REAL,prior_qz REAL,prior_tx REAL,prior_ty REAL,prior_tz REAL)')
            db.execute('INSERT INTO images VALUES(?,?,?,?,?,?,?,?)', (1, *q, *W[:3, 3]))
        result = app.apply_scene((str(scene), str(report), info, G, {}))
        self.assertEqual(len(result['model_checks']), 5)
        self.assertEqual(app.sha(folder / 'rigs.bin'), rig_hash)
        self.assertNotEqual(app.sha(folder / 'frames.bin'), frame_hash)
        check = pycolmap.Reconstruction(str(folder))
        np.testing.assert_allclose(check.images[1].cam_from_world().matrix(), (np.vstack([W, [0, 0, 0, 1]]) @ np.linalg.inv(G))[:3], atol=1e-12)
        values = yaml.safe_load((rgb / 'cameras.yaml').read_text())['extrinsics'][image.name]
        expected_q, expected_t = app.change_pose(q, W[:3, 3], G)
        np.testing.assert_allclose(values['qvec'], expected_q, atol=1e-12)
        np.testing.assert_allclose(values['tvec'], expected_t, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
