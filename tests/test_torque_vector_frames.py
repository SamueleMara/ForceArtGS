from pathlib import Path
import json
import sys
import tempfile
import unittest

import cv2
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/('samuele' if (BASE/'samuele/torque_vector_frames.py').is_file() else 'scripts')))
from torque_vector_frames import vector_segments,save_torque_frames


class TorqueVectorFrames(unittest.TestCase):
    def test_vector_geometry_and_separate_all_camera_images(self):
        vector=np.array([.1,-.2,.3]);segments=vector_segments(vector)
        np.testing.assert_array_equal(segments[0,0],np.zeros(3))
        np.testing.assert_array_equal(segments[0,1],vector)
        self.assertEqual(vector_segments(np.zeros(3)).shape,(0,2,3))
        self.assertEqual(vector_segments([np.nan,0,1]).shape,(0,2,3))
        records=[dict(camera=f'cam_{i:03d}',camera_to_base=np.eye(4).tolist()) for i in range(3)]
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);torques=[dict(torque_base_Nm=np.array([vector,-2*vector]))]
            result=save_torque_frames(folder,[0,1],[0,-1],[0,.02],torques,['ee_aligned'],records)
            self.assertEqual(result['images'],2);self.assertEqual(len(result['panels']),4)
            self.assertAlmostEqual(result['axis_limit_Nm'],2*np.linalg.norm(vector)*1.15)
            first=cv2.imread(str(folder/'frame_000000.png'));second=cv2.imread(str(folder/'frame_000001.png'))
            self.assertEqual(first.shape,second.shape);self.assertFalse(np.array_equal(first,second))
            self.assertTrue((folder/'index.html').is_file())
            self.assertEqual(json.loads((folder/'manifest.json').read_text())['rgbd_frames'],[0,-1])


if __name__=='__main__':unittest.main()
