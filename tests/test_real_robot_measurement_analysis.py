from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/('samuele' if (BASE/'samuele/analyze_real_robot_measurements.py').is_file() else 'scripts')))
from analyze_real_robot_measurements import POSE_FIELDS,WRENCH_FIELDS,discover,main,parse_measurements,add_tcp,tool_for_job,rigid_transform,require_tcp_calibration,analyze


def fixture_rows():
    rows=[]
    for i in range(4):
        rows.append({'timestamp':float(i),'frame_index':i,'episode_index':0,
            'observation.state':[float(i),0.,0.,0.,0.,float(i)*.1],
            'observation.wrench':[3.,4.,0.,0.,0.,2.]})
    return rows


class RobotMeasurements(unittest.TestCase):
    def test_early_mounting_overrides_global_axes_and_blocks_tcp_generation(self):
        registry=json.loads((BASE/'configs/real_robot_tools.json').read_text())
        invalid=[]
        for name,entry in registry['datasets'].items():
            job=dict(name=name,episode=entry['episode'],recording_root=Path('/'+entry['source_recording']),conversion=None)
            metadata=tool_for_job(job,registry)
            if metadata.get('tcp_calibration_valid') is False:
                invalid.append(name)
                self.assertEqual(metadata['wrench_frame']['axes'],'unspecified')
                self.assertIsNone(metadata['wrench_frame']['R_ee_force_sensor'])
                with self.assertRaisesRegex(ValueError,'Uncalibrated sensor mounting'):
                    analyze(dict(tool_metadata=metadata),Path('/unused'))
            else:require_tcp_calibration(metadata)
        self.assertEqual(set(invalid),{f'white_drawer_20261021_161154_episode_{i:06d}' for i in range(3)})

    def test_tcp_rotates_local_offset_and_composes_tool_rotation(self):
        rows=fixture_rows()
        for row in rows:row['observation.state'][3:]=[0.,np.pi/2,0.]
        for z in [.03487365,.18761740]:
            data=parse_measurements(rows,dict(state_names=POSE_FIELDS,wrench_names=WRENCH_FIELDS),0)
            T=np.eye(4);T[2,3]=z;add_tcp(data,T)
            np.testing.assert_allclose(data['tcp_position_m']-data['ee_position_m'],np.tile([z,0,0],(4,1)),atol=1e-12)
            np.testing.assert_allclose(data['tcp_rotvec_rad'],data['ee_rotvec_rad'],atol=1e-12)
        from scipy.spatial.transform import Rotation
        T[:3,:3]=Rotation.from_euler('x',.4).as_matrix();add_tcp(data,T)
        expected=Rotation.from_rotvec(data['ee_rotvec_rad']).as_matrix()@T[:3,:3]
        np.testing.assert_allclose(Rotation.from_rotvec(data['tcp_rotvec_rad']).as_matrix(),expected,atol=1e-12)

    def test_tcp_speed_includes_rotation_and_missing_orientation_stays_missing(self):
        rows=fixture_rows()
        for i,row in enumerate(rows):row['observation.state']=[0.,0.,0.,0.,i*np.pi/2,0.]
        data=parse_measurements(rows,dict(state_names=POSE_FIELDS,wrench_names=WRENCH_FIELDS),0)
        T=np.eye(4);T[2,3]=.18761740;add_tcp(data,T)
        np.testing.assert_allclose(data['linear_speed_m_s'][1:],0.)
        np.testing.assert_allclose(data['tcp_linear_speed_m_s'][1:],np.sqrt(2)*T[2,3],atol=1e-12)
        data['ee_rotvec_rad'][1]=np.nan;add_tcp(data,T)
        self.assertTrue(np.isnan(data['tcp_position_m'][1]).all())
        self.assertTrue(np.isnan(data['tcp_linear_speed_m_s'][1:3]).all())
        invalid=T.copy();invalid[0,0]=-1
        with self.assertRaises(ValueError):rigid_transform(invalid)

    def test_registry_has_exactly_three_bare_tool_exceptions_and_no_defaults(self):
        registry=json.loads((BASE/'configs/real_robot_tools.json').read_text())
        bare={name for name,e in registry['datasets'].items() if e['tool']=='bare_force_sensor'}
        self.assertEqual(bare,{f'white_drawer_20261021_161154_episode_{i:06d}' for i in range(3)})
        self.assertEqual(len(registry['datasets']),118)
        self.assertNotIn('defaults',registry)
        job=dict(name='unknown_episode_000000',episode=0,recording_root=Path('/unknown'),conversion=None)
        with self.assertRaises(ValueError):tool_for_job(job,registry)
        for name,entry in registry['datasets'].items():
            job=dict(name=name,episode=entry['episode'],recording_root=Path('/'+entry['source_recording']),conversion=None)
            metadata=tool_for_job(job,registry)
            self.assertEqual(metadata['tool'],entry['tool'])
            self.assertEqual(metadata['T_ee_tcp'][2][3],.03487365 if name in bare else .18761740)
            job['name']=f'{entry["source_recording"]}_episode_{entry["episode"]:06d}'
            self.assertEqual(tool_for_job(job,registry)['tool'],entry['tool'])

    def test_named_fields_units_velocity_and_rgbd_alignment(self):
        rows=fixture_rows();info=dict(state_names=list(POSE_FIELDS)[::-1],wrench_names=list(WRENCH_FIELDS)[::-1])
        for row in rows:
            row['observation.state']=row['observation.state'][::-1];row['observation.wrench']=row['observation.wrench'][::-1]
        data=parse_measurements(rows[::-1],info,0,dict(episode=0,source_frames=[0,2],output_frames=[0,1]))
        np.testing.assert_allclose(data['force_norm_N'],5.)
        np.testing.assert_allclose(data['torque_norm_Nm'],2.)
        np.testing.assert_allclose(data['linear_speed_m_s'][1:],1.)
        np.testing.assert_allclose(data['angular_speed_rad_s'][1:],.1)
        np.testing.assert_equal(data['rgbd_frame_index'],[0,-1,1,-1])
        self.assertTrue(np.isnan(data['linear_speed_m_s'][0]))

    def test_rotation_vector_wrap_uses_relative_rotation(self):
        rows=fixture_rows()[:2];rows[0]['observation.state'][5]=np.pi-.01;rows[1]['observation.state'][5]=-np.pi+.01
        d=parse_measurements(rows,dict(state_names=POSE_FIELDS,wrench_names=WRENCH_FIELDS),0)
        self.assertAlmostEqual(d['angular_speed_rad_s'][1],.02)

    def test_missing_measurements_and_duplicate_time_are_not_filled(self):
        rows=fixture_rows();rows[1]['timestamp']=0.;rows[2]['observation.wrench']=None;rows[2]['observation.state'][0]=None
        d=parse_measurements(rows,dict(state_names=POSE_FIELDS,wrench_names=WRENCH_FIELDS),0)
        self.assertTrue(np.isnan(d['linear_speed_m_s'][1:]).all())
        self.assertTrue(np.isnan(d['force_norm_N'][2]))
        d=parse_measurements(rows,dict(state_names=POSE_FIELDS),0,wrench_available=False)
        self.assertTrue(np.isnan(d['force_N']).all())

    def test_decreasing_time_and_wrong_episode_are_rejected(self):
        rows=fixture_rows();rows[2]['timestamp']=-1.
        with self.assertRaises(ValueError):parse_measurements(rows,dict(state_names=POSE_FIELDS,wrench_names=WRENCH_FIELDS),0)
        with self.assertRaises(ValueError):parse_measurements(fixture_rows(),dict(state_names=POSE_FIELDS,wrench_names=WRENCH_FIELDS),1)

    def test_native_and_converted_discovery_and_full_plot_smoke(self):
        with tempfile.TemporaryDirectory(prefix='robot measurements ') as td:
            root=Path(td);native=root/'assets/final_box';(native/'meta').mkdir(parents=True);(native/'data/chunk-000').mkdir(parents=True)
            info=dict(state_names=POSE_FIELDS,wrench_names=WRENCH_FIELDS)
            (native/'meta/info.json').write_text(json.dumps(info));pq=native/'data/chunk-000/episode_000000.parquet';pl.DataFrame(fixture_rows()).write_parquet(pq)
            scene=root/'real/final_box_episode_000000';raw=scene/'metadata/raw';raw.mkdir(parents=True)
            (raw/'info.json').write_bytes((native/'meta/info.json').read_bytes());(raw/pq.name).write_bytes(pq.read_bytes())
            (scene/'metadata/real_conversion.json').write_text(json.dumps(dict(source=str(native),episode=0,source_frames=[0,2],output_frames=[0,1])))
            registry=json.loads((BASE/'configs/real_robot_tools.json').read_text());registry['datasets']={scene.name:dict(source_recording='final_box',episode=0,tool='pointed_tip',recorded_pose_frame='ee',coordinate_frame='base_link')}
            registry_file=root/'tools.json';registry_file.write_text(json.dumps(registry))
            self.assertEqual(len(discover([root/'assets'])),1);self.assertEqual(len(discover([root/'real'])),1)
            protected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            args=['--input',str(root/'real'),'--output-root',str(root/'plots'),'--tool-metadata',str(registry_file)]
            self.assertEqual(main(args+['--dry-run']),0);self.assertFalse((root/'plots').exists())
            self.assertEqual(main(args),0)
            out=root/'plots'/scene.name
            for name in ['measurements.npz','measurements.csv','summary.json','measurements.png','ee_trajectory.png','tcp_measurements.png','tcp_trajectory.png','robot_tool.json']:self.assertTrue((out/name).is_file())
            self.assertEqual(json.loads((out/'summary.json').read_text())['force']['max'],5.)
            np.testing.assert_equal(np.load(out/'measurements.npz')['rgbd_frame_index'],[0,-1,1,-1])
            np.testing.assert_allclose(np.load(out/'measurements.npz')['tcp_position_m'][:,2],.18761740)
            self.assertEqual(json.loads((out/'robot_tool.json').read_text())['tool'],'pointed_tip')
            self.assertEqual(main(args),2)
            for p,h in protected.items():self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),h)
            self.assertEqual(main(['--input',str(pq),'--output-root',str(native/'analysis'),'--tool-metadata',str(registry_file)]),2)
            metadata=tool_for_job(discover([scene])[0],registry);metadata['tool']='bare_force_sensor'
            (scene/'metadata/robot_tool.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):tool_for_job(discover([scene])[0],registry)


if __name__=='__main__':unittest.main()
