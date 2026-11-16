from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/('samuele' if (BASE/'samuele/plot_tcp_force_directions.py').is_file() else 'scripts')))
from plot_tcp_force_directions import force_to_base,select_frames,prepare,main,arrow_geometry,initial_camera,interactive_figure,process,refresh_html
from types import SimpleNamespace


def example():
    return dict(frame_index=np.arange(8),rgbd_frame_index=np.array([0,-1,1,-1,2,-1,3,-1]),timestamp_s=np.arange(8)*.02,
        tcp_position_m=np.c_[np.arange(8)*.01,np.zeros(8),np.full(8,.4)],
        ee_rotvec_rad=np.tile([0,0,np.pi/2],(8,1)),force_N=np.tile([2.,0,0],(8,1)))


class ForceDirections(unittest.TestCase):
    def test_uncalibrated_mounting_and_explicit_unknown_axes_cannot_reuse_cached_claims(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);scene=root/'measurements/drawer';scene.mkdir(parents=True)
            target=root/'plots/drawer';target.mkdir(parents=True)
            manifest=target/'manifest.json';manifest.write_text(json.dumps(dict(tcp_calibration_valid=False)))
            html=target/'force_directions.html';html.write_text('historical plot')
            summary=scene/'summary.json'
            summary.write_text(json.dumps(dict(wrench_coordinate_frame='EE_ALIGNED_AXES_USER_CONFIRMED',tcp_calibration=dict(tcp_calibration_valid=False,wrench_frame=dict(axes='ee')))))
            with self.assertRaisesRegex(ValueError,'Uncalibrated sensor mounting'):process(scene,target.parent,SimpleNamespace())
            with self.assertRaisesRegex(ValueError,'Uncalibrated sensor mounting'):refresh_html(scene,target.parent,SimpleNamespace(overwrite=True))
            summary.write_text(json.dumps(dict(wrench_coordinate_frame='EE_ALIGNED_AXES_USER_CONFIRMED',tcp_calibration=dict(wrench_frame=dict(axes='unspecified')))))
            with self.assertRaisesRegex(ValueError,'EE-aligned force axes'):process(scene,target.parent,SimpleNamespace())
            self.assertEqual(html.read_text(),'historical plot')

    def test_small_arrowhead_preserves_root_tip_and_vector_direction(self):
        root=np.array([.2,-.3,.4]);vector=np.array([.01,.02,-.01])
        shaft,vertices,faces=arrow_geometry(root,vector,.15)
        np.testing.assert_allclose(shaft[0],root)
        np.testing.assert_allclose(vertices[0],root+vector)
        direction=vector/np.linalg.norm(vector);offset=vertices[1:]-vertices[0]
        np.testing.assert_allclose(offset@direction,-.15*np.linalg.norm(vector),atol=1e-12)
        width=np.linalg.norm(vertices[1:]-shaft[1],axis=1)
        np.testing.assert_allclose(width,.35*.15*np.linalg.norm(vector))
        self.assertEqual(faces.shape,(8,3))
        camera=initial_camera(np.array([[0.,0,0],[1.,0,0]]),np.array([[0.,1.,0]]))
        eye=np.array(list(camera['eye'].values()))
        self.assertLess(abs(eye[0]/np.linalg.norm(eye)),.5)
        self.assertEqual(camera['projection']['type'],'orthographic')
        data=example();p=prepare(data,'native',None,None,2,.05,'direction',.025,.01)
        figure=interactive_figure('control',data,p,'control','direction',.025,.01)
        self.assertNotIn('cone',[trace.type for trace in figure.data])
        scene=figure.layout.scene
        spans=np.array([np.ptp(axis.range) for axis in [scene.xaxis,scene.yaxis,scene.zaxis]])
        sizes=np.array([scene.aspectratio.x,scene.aspectratio.y,scene.aspectratio.z])
        np.testing.assert_allclose(sizes/spans,np.full(3,sizes[0]/spans[0]))

    def test_rotates_force_to_common_axes_and_preserves_norm(self):
        f=np.array([[2.,0,0],[0,0,3.]])
        result=force_to_base([[0,0,np.pi/2],[0,np.pi/2,0]],f)
        np.testing.assert_allclose(result,[[0,2,0],[3,0,0]],atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(result,axis=1),np.linalg.norm(f,axis=1))

    def test_rgbd_arrow_spacing_and_inclusive_native_time_interval(self):
        d=example();selected,roots=select_frames(d,'rgbd',[1,3],every=2)
        np.testing.assert_array_equal(np.flatnonzero(selected),[2,3,4,5,6])
        np.testing.assert_array_equal(np.flatnonzero(roots),[2,6])
        with self.assertRaises(ValueError):select_frames(d,'rgbd',[0,4])
        d['timestamp_s'][:3]=0.
        selected,_=select_frames(d,'native',[1,3],every=1)
        np.testing.assert_array_equal(np.flatnonzero(selected),[1,2,3])

    def test_direction_and_magnitude_scales_and_zero_force(self):
        d=example();p=prepare(d,'native',[1,6],None,2,.05,'direction',.025,.01)
        np.testing.assert_array_equal(np.flatnonzero(p['drawn']),[1,3,5])
        np.testing.assert_allclose(p['display_vectors_m'][p['drawn']],np.tile([0,.025,0],(3,1)),atol=1e-12)
        q=prepare(d,'native',[1,6],None,2,.05,'magnitude',.025,.01)
        np.testing.assert_allclose(q['display_vectors_m'][q['drawn']],np.tile([0,.02,0],(3,1)),atol=1e-12)
        d['force_N'][3]=0;d['ee_rotvec_rad'][5]=np.nan
        r=prepare(d,'native',[1,6],None,2,.05,'direction',.025,.01)
        np.testing.assert_array_equal(np.flatnonzero(r['drawn']),[1])

    def test_batch_outputs_and_original_data_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);scene=root/'measurements/object_episode_000000';scene.mkdir(parents=True)
            np.savez_compressed(scene/'measurements.npz',**example())
            (scene/'summary.json').write_text(json.dumps(dict(wrench_coordinate_frame='EE_ALIGNED_AXES_USER_CONFIRMED',tcp_calibration=dict(wrench_frame=dict(axes='ee')))))
            original={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in scene.iterdir()}
            command=['--measurements-root',str(scene.parent),'--output-root',str(root/'plots'),'--frame-range','1','6','--arrow-every','2']
            self.assertEqual(main(command),0)
            output=root/'plots'/scene.name
            for filename in ['force_directions.png','force_directions.html','force_vectors.npz','force_vectors.csv','manifest.json']:self.assertTrue((output/filename).is_file())
            with np.load(output/'force_vectors.npz',allow_pickle=False) as data:
                np.testing.assert_array_equal(data['source_frame'],np.arange(1,7))
                np.testing.assert_allclose(data['force_base_N'],np.tile([0,2,0],(6,1)),atol=1e-12)
                np.testing.assert_allclose(data['tcp_position_m'],example()['tcp_position_m'][1:7])
            html=(output/'force_directions.html').read_text();self.assertIn('"type":"mesh3d"',html);self.assertIn('"name":"Force arrows"',html)
            preserved={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [output/'force_vectors.csv',output/'force_vectors.npz',output/'force_directions.png']}
            self.assertEqual(main(command+['--html-only','--overwrite','--head-fraction','.1']),0)
            for path,digest in preserved.items():self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
            self.assertEqual(json.loads((output/'manifest.json').read_text())['html_head_fraction'],.1)
            for path,digest in original.items():self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main()
