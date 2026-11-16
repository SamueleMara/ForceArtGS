"""Automatic follow-up oracles, immutable-input and resumability checks."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
import numpy as np

BASE=Path(__file__).resolve().parents[1]/'samuele/audits/real_rgbd'
if not BASE.is_dir():BASE=Path('/home/samuelemara/Joint3DGS/Sapien/scripts/audits/real_rgbd')
sys.path.insert(0,str(BASE))
# Import sibling test helpers from this repository's tests/ (not the Joint3DGS checkout), so unittest
# discovery does not resolve later test modules such as test_robot_registration_audit elsewhere.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from followup.cli import main
from followup.common import rows_write,iter_rows
from followup.synthetic import run_controls
from followup.surfaces import extract,associate
from followup.identifiability import residual_and_jacobian
from audit_registration import load_converter
from audit_common import sha
from followup.phase1_worker import endpoint_ply,parity_qualification
from test_real_rgbd_audit import make_real_fixture,main as original_audit

CONVERTER=load_converter('/home/samuelemara/FreeArtGS/samuele/convert_real_recording.py')


class AutomaticIdentifiability(unittest.TestCase):
    def test_nondeterministic_floor_does_not_relax_parity(self):
        zero=dict(mu=0.,v=0.,omega=0.,R=0.,P=0.)
        self.assertEqual(parity_qualification(zero,zero,True,[0.])['status'],'PASS')
        large=dict(zero,mu=.012)
        self.assertEqual(parity_qualification(large,large,True,[.008])['status'],'FAIL')
        self.assertEqual(parity_qualification(zero,zero,False,[0.])['status'],'FAIL')

    def test_endpoint_iteration_is_numeric_not_lexicographic(self):
        with tempfile.TemporaryDirectory() as td:
            for i in [0,250,500,1000]:
                p=Path(td)/f'point_cloud/iteration_{i}/point_cloud.ply';p.parent.mkdir(parents=True);p.write_bytes(b'control')
            self.assertEqual(endpoint_ply(td).parent.name,'iteration_1000')

    def test_scientific_positive_and_negative_controls(self):
        with tempfile.TemporaryDirectory() as td:
            r=run_controls(Path(td),CONVERTER)
            self.assertEqual(r['failed'],[]);self.assertGreaterEqual(r['passed'],15);self.assertTrue(r['hard_negative_identity'])

    def test_end_to_end_same_plane_and_aliased_parallel_planes(self):
        K=np.array([[80.,0,32],[0,80,24],[0,0,1.]])
        d=np.ones((48,64),np.float32);rgb=np.full((48,64,3),127,np.uint8)
        a=extract(d,rgb,None,K,np.eye(4));a=extract(d,rgb,None,K,np.eye(4),a)
        features=[dict(u_i_px=x,v_i_px=y,u_j_px=x,v_j_px=y) for x,y in [(12,12),(45,12),(12,35),(45,35),(16,16),(40,16),(16,30),(40,30),(22,20),(35,20),(22,27),(35,27)]]
        for bias,expected in [(.005,'AUTO_MULTICUE_CONSISTENT_CONDITIONAL'),(.15,'UNIDENTIFIABLE_LAYER_OR_CALIBRATION')]:
            b=extract(d+bias,rgb,None,K,np.eye(4));b=extract(d+bias,rgb,None,K,np.eye(4),b)
            rows=associate(a,b,d,d+bias,K,np.eye(4),np.eye(4),features)
            self.assertTrue(rows);self.assertTrue(any(r['status']==expected for r in rows),rows)
            if bias==.005:
                good=next(r for r in rows if r['status']==expected)
                self.assertAlmostEqual(abs(good['signed_world_plane_offset_m']),bias,places=6)

    def test_late_added_measurement_fields_are_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            p=rows_write(Path(td)/'x.parquet',[dict(frame=0,value=None),dict(frame=1,value=2.,newly_missing=7)])
            rows=list(iter_rows(p));self.assertEqual(rows[1]['newly_missing'],7);self.assertIsNone(rows[0]['newly_missing'])

    def test_full_fixture_scope_replay_resume_and_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);raw,scene=make_real_fixture(root)
            args=['--converted-roots',str(root/'converted'),'--raw-roots',str(root/'assets'),'--output-root',str(root/'audit'),'--mode','full','--all-scenes','--all-frames','--all-cameras']
            self.assertEqual(original_audit(args),0)
            protected={p:sha(p) for parent in [raw,scene,root/'audit'] for p in parent.rglob('*') if p.is_file()}
            command=['--audit-root',str(root/'audit'),'--out',str(root/'followup'),'--all','--workers','1','--stage','inventory,synthetic,holes,coverage,verified-geometry,pose-vs-bias,timing']
            self.assertEqual(main(command+['--dry-run']),0);self.assertFalse((root/'followup').exists())
            self.assertEqual(main(command),0)
            receipt=json.loads((root/'followup/execution_receipt.json').read_text());r=next(iter(receipt['scene_results'].values()))
            self.assertEqual(r['counts']['frame_camera_observations'],8);self.assertEqual(r['counts']['registration_replays'],8);self.assertEqual(r['counts']['replay_different_pixels'],0)
            for p,value in protected.items():self.assertEqual(sha(p),value)
            self.assertEqual(main(command+['--resume']),0)
            data=scene/'depth_npy/cam_000/frame_000000.npy';np.save(data,np.zeros((24,32),np.float32))
            self.assertNotEqual(main(command+['--resume']),0)


if __name__=='__main__':unittest.main()
