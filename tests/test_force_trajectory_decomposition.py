import sys
from pathlib import Path
import unittest

import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/('samuele' if (BASE/'samuele/force_trajectory_decomposition.py').is_file() else 'scripts')))
from force_trajectory_decomposition import trajectory_tangent,decompose_force


class TrajectoryForces(unittest.TestCase):
    def test_signed_line_components_reconstruct_force_and_power(self):
        time=np.arange(101)*.02;position=np.c_[.1*time,np.zeros((101,2))]
        info=trajectory_tangent(time,position)
        force=np.tile([-2.,3.,4.],(101,1));case=decompose_force(force,info)
        self.assertTrue(info['tangent_valid'][1:-1].all());self.assertFalse(info['tangent_valid'][[0,-1]].any())
        np.testing.assert_allclose(info['tangent'][1:-1],np.tile([1,0,0],(99,1)),atol=1e-12)
        np.testing.assert_allclose(case['force_tangent_signed_N'][1:-1],-2,atol=1e-12)
        np.testing.assert_allclose(case['force_perpendicular_norm_N'][1:-1],5,atol=1e-12)
        np.testing.assert_allclose(case['force_dot_ee_velocity_W'][1:-1],-.2,atol=1e-12)
        np.testing.assert_allclose(case['force_tangent_base_N'][1:-1]+case['force_perpendicular_base_N'][1:-1],force[1:-1])
        np.testing.assert_allclose(case['eta'][1:-1],2/(np.sqrt(29)+1e-8))

    def test_curve_uses_local_tangent_not_one_global_axis(self):
        time=np.arange(201)*.01;angle=time*1.3
        position=np.c_[.2*np.cos(angle),.2*np.sin(angle),np.zeros(len(time))]
        info=trajectory_tangent(time,position)
        expected=np.c_[-np.sin(angle),np.cos(angle),np.zeros(len(time))]
        np.testing.assert_allclose(info['tangent'][10:-10],expected[10:-10],atol=1e-6)
        case=decompose_force(2*expected,info)
        np.testing.assert_allclose(case['force_tangent_signed_N'][10:-10],2,atol=1e-8)
        np.testing.assert_allclose(case['force_perpendicular_norm_N'][10:-10],0,atol=1e-6)

    def test_stationary_reversal_missing_data_and_gaps_remain_explicit(self):
        time=np.arange(101)*.02
        fixed=trajectory_tangent(time,np.zeros((101,3)))
        self.assertFalse(fixed['tangent_valid'].any())
        case=decompose_force(np.ones((101,3)),fixed)
        self.assertTrue(np.isnan(case['force_tangent_signed_N']).all())
        self.assertTrue(np.isnan(case['force_perpendicular_norm_N']).all())
        position=np.c_[(time-1)**2,np.zeros((101,2))]
        info=trajectory_tangent(time,position)
        self.assertFalse(info['tangent_valid'][50]);self.assertLess(info['tangent'][20,0],0);self.assertGreater(info['tangent'][80,0],0)
        broken=time.copy();broken[50]=broken[49]
        info=trajectory_tangent(broken,position)
        self.assertEqual(info['tangent_status'][49],'DUPLICATE_TIMESTAMP')
        self.assertEqual(info['tangent_status'][50],'DUPLICATE_TIMESTAMP')
        position[49:52]=np.nan
        info=trajectory_tangent(time,position)
        self.assertFalse(info['tangent_valid'][49:52].any())
        gap_time=np.r_[np.arange(6)*.02,10+np.arange(6)*.02]
        p=np.c_[np.r_[np.zeros(6),np.ones(6)],np.zeros((12,2))]
        frames=np.r_[np.arange(6),500+np.arange(6)]
        self.assertFalse(trajectory_tangent(gap_time,p,frames)['tangent_valid'].any())

    def test_exact_neighbors_without_smoothing_or_speed_cutoff(self):
        time=np.arange(5,dtype=float)
        position=np.array([[0,0,0],[1,2,3],[2,0,0],[2,4,0],[6,0,0]],float)*1e-10
        info=trajectory_tangent(time,position)
        delta=position[2:]-position[:-2]
        np.testing.assert_allclose(info['tangent'][1:-1],delta/np.linalg.norm(delta,axis=1)[:,None],atol=1e-15)
        self.assertTrue(info['tangent_valid'][1:-1].all())
        self.assertTrue(np.isnan(info['tangent'][[0,-1]]).all())
        zero=decompose_force(np.zeros((5,3)),info)
        np.testing.assert_array_equal(zero['eta'][1:-1],0)

    def test_no_force_does_not_remove_a_valid_tangent(self):
        t=np.arange(11)*.02;p=np.c_[t,np.zeros((11,2))]
        info=trajectory_tangent(t,p);f=np.ones((11,3));f[5]=np.nan
        case=decompose_force(f,info)
        self.assertTrue(info['tangent_valid'][5]);self.assertFalse(case['decomposition_valid'][5])
        self.assertTrue(np.isnan(case['force_tangent_signed_N'][5]))


if __name__=='__main__':unittest.main()
