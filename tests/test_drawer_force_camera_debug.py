from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/('samuele' if (BASE/'samuele/debug_drawer_forces_camera.py').is_file() else 'scripts')))
from debug_drawer_forces_camera import project_force,arrow,main


class CameraForceDebug(unittest.TestCase):
    def test_perspective_projection_rotates_force_without_translating_it(self):
        K=np.array([[100.,0,50],[0,100,40],[0,0,1]])
        T=np.eye(4);pixels,force=project_force([0,0,1],np.array([1.,0,0]),T,K,.1)
        np.testing.assert_allclose(pixels,[[50,40],[60,40]])
        T[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix();T[:3,3]=[.1,.2,.3]
        pixels,force=project_force([0,0,1],np.array([1.,0,0]),T,K,.1)
        np.testing.assert_allclose(force,[0,1,0],atol=1e-12)
        np.testing.assert_allclose(pixels,[[50+10/1.3,40+20/1.3],[50+10/1.3,40+30/1.3]])
        pixels,_=project_force([0,0,1],np.array([0.,0,2]),np.eye(4),K,.1)
        self.assertEqual(arrow(np.zeros((100,100,3),np.uint8),pixels,(0,255,0)),'projected length <2 px')

    def test_behind_camera_outside_origin_and_clipped_head_are_explicit(self):
        K=np.eye(3);pixels,_=project_force([0,0,1],np.array([0.,0,-2]),np.eye(4),K,1)
        self.assertTrue(np.isnan(pixels[1]).all())
        image=np.zeros((50,50,3),np.uint8)
        self.assertIn('outside image',arrow(image,np.array([[-1,20],[30,20]]),(0,255,0)))
        self.assertIn('head outside image',arrow(image,np.array([[20,20],[100,20]]),(0,255,0)))

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg not installed')
    def test_video_csv_and_exact_rgbd_mapping_without_source_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);scene=root/'real/drawer';meta=scene/'metadata';meta.mkdir(parents=True)
            rgb=scene/'rgb/cam_002';rgb.mkdir(parents=True)
            measurements=root/'measurements/drawer';measurements.mkdir(parents=True)
            camera=dict(name='cam_002',width=64,height=48,fx=100,fy=100,cx=32,cy=24,
                pose_matrix_4x4=np.eye(4).tolist(),pose_convention='opencv_optical_camera_to_world',coordinate_frame='base_link')
            (meta/'cameras.json').write_text(json.dumps(dict(cameras=[camera])))
            (meta/'real_conversion.json').write_text(json.dumps(dict(source_frames=[0,2,4],output_frames=[0,1,2])))
            (meta/'frame_timestamps.csv').write_text('frame_index,source_frame_index,timestamp_s\n0,0,0\n1,2,0.04\n2,4,0.08\n')
            for i in range(3):cv2.imwrite(str(rgb/f'frame_{i:06d}.png'),np.full((48,64,3),50+i,np.uint8))
            data=dict(frame_index=np.arange(5),rgbd_frame_index=np.array([0,-1,1,-1,2]),timestamp_s=np.arange(5)*.02,
                ee_position_m=np.c_[np.arange(5)*.002,np.zeros(5),np.full(5,.98)],tcp_position_m=np.c_[np.zeros(5),np.arange(5)*.002,np.ones(5)],
                ee_rotvec_rad=np.zeros((5,3)),force_N=np.tile([1.,0,1.],(5,1)),torque_Nm=np.tile([.1,.2,.3],(5,1)))
            np.savez_compressed(measurements/'measurements.npz',**data)
            (measurements/'summary.json').write_text(json.dumps(dict(tcp_calibration=dict(wrench_frame=dict(axes='ee')))))
            protected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            args=['--converted-root',str(scene.parent),'--measurements-root',str(measurements.parent),'--output-root',str(root/'debug'),'--scene','drawer']
            self.assertEqual(main(args),0)
            out=root/'debug/drawer';manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['frames'],3);self.assertAlmostEqual(manifest['fps'],25)
            with (out/'projected_forces.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual([int(r['native_frame']) for r in rows],[0,0,2,2,4,4])
            np.testing.assert_allclose([float(rows[1][f'force_ee_N_{a}']) for a in 'xyz'],[0,1,1],atol=1e-12)
            capture=cv2.VideoCapture(str(out/'force_cam_002.mp4'))
            try:
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),3)
                ok,frame=capture.read();self.assertTrue(ok);self.assertEqual(frame.shape,(224,128,3))
            finally:capture.release()
            self.assertTrue((out/'peak_force.jpg').is_file());self.assertTrue((root/'debug/index.html').is_file())
            camera['name']='cam_001'
            (meta/'cameras.json').write_text(json.dumps(dict(cameras=[camera,dict(camera,name='cam_002')])))
            shutil.copytree(rgb,scene/'rgb/cam_001')
            args2=['--converted-root',str(scene.parent),'--measurements-root',str(measurements.parent),'--output-root',str(root/'decomposed'),
                '--all-scenes','--all-cameras','--decompose','--interaction-root',str(root/'interactions'),'--torque-3d-images']
            interaction=root/'interactions/drawer';interaction.mkdir(parents=True)
            (interaction/'result.json').write_text(json.dumps(dict(scene='drawer',provenance=dict(tcp_archive_sha256=hashlib.sha256((measurements/'measurements.npz').read_bytes()).hexdigest()),
                interval=dict(selected_window_s=[.02,.06],selected_interval_source='test_bounds',status='TEST'))))
            self.assertEqual(main(args2),0)
            shared=root/'decomposed/drawer'
            with np.load(shared/'force_decomposition.npz',allow_pickle=False) as values:
                self.assertEqual(values['force_tangent_signed_N'].shape,(1,5))
                np.testing.assert_allclose(values['force_tangent_signed_N'][:,1:-1],1,atol=1e-12)
                np.testing.assert_allclose(values['force_perpendicular_norm_N'][:,1:-1],1,atol=1e-12)
                np.testing.assert_array_equal(values['frame_index'],np.arange(5))
            self.assertTrue((shared/'force_decomposition.png').is_file())
            for cam in ['cam_001','cam_002']:
                self.assertTrue((shared/cam/f'force_{cam}.mp4').is_file())
                self.assertTrue((shared/cam/'torque_camera.csv').is_file())
                self.assertTrue((shared/cam/'torque_camera.png').is_file())
            with np.load(shared/'wrench_analysis.npz',allow_pickle=False) as values:
                np.testing.assert_allclose(values['force_tangential_pct'][:,1:-1],100/(np.sqrt(2)+1e-8),atol=1e-10)
                self.assertTrue(np.isnan(values['force_tangential_pct'][:,[0,-1]]).all())
                np.testing.assert_allclose(values['torque_base_Nm'],np.tile([.1,.2,.3],(1,5,1)),atol=1e-12)
            self.assertTrue((root/'decomposed/force_torque_summary.html').is_file())
            self.assertEqual(len(list((shared/'torque_3d_frames').glob('frame_*.png'))),5)
            self.assertTrue((shared/'torque_3d_frames/index.html').is_file())
            summary=json.loads((root/'decomposed/force_torque_summary.json').read_text())
            self.assertAlmostEqual(summary['per_dataset'][0]['interaction_temporal_mean_pct'],100/(np.sqrt(2)+1e-8))
            # A refreshed legacy run must ignore the previously cached TCP direction.
            cached=shared/'force_decomposition.npz'
            with np.load(cached,allow_pickle=False) as z:stale={key:z[key].copy() for key in z.files}
            stale['tangent'][1:-1]=[0,1,0];stale['force_tangent_signed_N'][:]=0
            np.savez_compressed(cached,**stale)
            from analyze_force_share_torques import update_analysis
            refreshed=update_analysis(root/'decomposed',measurements.parent,0.,root/'interactions',False,1e-6)
            self.assertFalse(refreshed['failed'])
            with np.load(shared/'wrench_analysis.npz',allow_pickle=False) as values:
                np.testing.assert_allclose(values['tangent'][1:-1],np.tile([1,0,0],(3,1)),atol=1e-12)
                np.testing.assert_allclose(values['eta'][:,1:-1],1/(np.sqrt(2)+1e-6),atol=1e-12)
            # Camera-fixture changes above are deliberate; every other source stays read-only.
            protected.pop(meta/'cameras.json')
            for p,digest in protected.items():self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),digest)
            with self.assertRaises(SystemExit):main(args)


if __name__=='__main__':unittest.main()
