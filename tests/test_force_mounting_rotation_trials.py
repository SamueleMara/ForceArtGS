from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/('samuele' if (BASE/'samuele/compare_force_mounting_rotations.py').is_file() else 'scripts')))
from compare_force_mounting_rotations import rotate_sensor_forces,main
from plot_tcp_force_directions import force_to_base


class MountingTrials(unittest.TestCase):
    def test_active_z_rotation_sign_equivalence_and_order(self):
        data=dict(force_N=np.eye(3),tcp_position_m=np.array([[.1,.2,.3]]))
        trial,matrix=rotate_sensor_forces(data,90)
        np.testing.assert_allclose(trial['force_N'],[[0,1,0],[-1,0,0],[0,0,1]],atol=1e-12)
        equivalent,other=rotate_sensor_forces(data,-270)
        np.testing.assert_allclose(matrix,other,atol=1e-12)
        np.testing.assert_allclose(equivalent['force_N'],trial['force_N'],atol=1e-12)
        np.testing.assert_array_equal(data['force_N'],np.eye(3))
        np.testing.assert_array_equal(trial['tcp_position_m'],data['tcp_position_m'])
        base=force_to_base([[np.pi/2,0,0]],trial['force_N'][:1])
        np.testing.assert_allclose(base,[[0,0,1]],atol=1e-12)

    def test_separate_labeled_trials_preserve_sources_and_nominal_positions(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);scene=root/'measurements/drawer';scene.mkdir(parents=True)
            fit=root/'fits/drawer';fit.mkdir(parents=True)
            d=dict(frame_index=np.arange(5),rgbd_frame_index=np.arange(5),timestamp_s=np.arange(5)*.02,
                tcp_position_m=np.c_[np.arange(5)*.01,np.zeros(5),np.ones(5)],
                ee_rotvec_rad=np.zeros((5,3)),force_N=np.tile([2.,3.,4.],(5,1)))
            archive=scene/'measurements.npz';np.savez_compressed(archive,**d)
            (scene/'summary.json').write_text(json.dumps(dict(tcp_calibration=dict(tcp_calibration_valid=False))))
            (fit/'result.json').write_text(json.dumps(dict(scene='drawer',interval=dict(selected_window_s=[0,.08]),
                provenance=dict(tcp_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))))
            original={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            args=['--measurements-root',str(scene.parent),'--interaction-root',str(fit.parent),'--output-root',str(root/'trial'),'--scene','drawer']
            self.assertEqual(main(args),0)
            output=root/'trial/drawer/sensor_z_plus90'
            with np.load(output/'force_vectors.npz') as z:
                np.testing.assert_array_equal(z['tcp_position_m'],d['tcp_position_m'])
                np.testing.assert_array_equal(z['force_sensor_N'],d['force_N'])
                np.testing.assert_allclose(z['force_ee_N'],np.tile([-3,2,4],(5,1)),atol=1e-12)
                np.testing.assert_allclose(z['force_base_N'],z['force_ee_N'],atol=1e-12)
            m=json.loads((output/'manifest.json').read_text());self.assertFalse(m['tcp_calibration_valid'])
            self.assertEqual(m['status'],'UNVALIDATED_MOUNTING_TRIAL')
            self.assertTrue((root/'trial/index.html').is_file())
            self.assertIn('UNVALIDATED',(output/'force_directions.html').read_text())
            for p,digest in original.items():self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),digest)
            with self.assertRaises(SystemExit):main(args)


if __name__=='__main__':unittest.main()
