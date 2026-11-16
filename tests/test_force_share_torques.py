from pathlib import Path
import sys
import unittest

import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/('samuele' if (BASE/'samuele/analyze_force_share_torques.py').is_file() else 'scripts')))
from analyze_force_share_torques import force_shares,torque_components,temporal_mean


class ForceShareTorques(unittest.TestCase):
    def test_force_magnitude_percent_signed_zero_and_stationary(self):
        signed=np.array([3.,-3.,0.,3.]);perp=np.array([4.,4.,0.,4.]);norm=np.array([5.,5.,0.,5.])
        result=force_shares(signed,perp,norm,np.array([True,True,True,False]))
        np.testing.assert_allclose(result['force_tangential_pct'][:2],60)
        np.testing.assert_allclose(result['force_tangential_signed_pct'][:2],[60,-60])
        np.testing.assert_allclose(result['force_perpendicular_pct'][:2],80)
        self.assertEqual(result['force_tangential_pct'][2],0)
        self.assertTrue(result['force_percentage_valid'][2]);self.assertFalse(result['force_percentage_valid'][3])
        self.assertTrue(np.isnan(result['force_tangential_pct'][3]))
        regularized=force_shares(signed,perp,norm,np.array([True,True,True,False]),force_epsilon_n=1.)
        np.testing.assert_allclose(regularized['eta'][:2],.5)
        np.testing.assert_allclose(regularized['force_tangential_pct'][:2],50)

    def test_temporal_mean_uses_actual_time_and_interaction_only(self):
        summary=temporal_mean([0,1,3],[0,100,100],[0,3])
        self.assertAlmostEqual(summary['interaction_temporal_mean_pct'],250/3)
        self.assertAlmostEqual(summary['interaction_coverage_pct'],100)
        clipped=temporal_mean([0,1,3],[0,100,100],[.5,2])
        self.assertAlmostEqual(clipped['interaction_temporal_mean_pct'],137.5/1.5)
        restricted=temporal_mean([0,1,2,3,4],[0,20,40,60,0],[1,3])
        self.assertAlmostEqual(restricted['interaction_temporal_mean_pct'],40)

    def test_undefined_time_is_not_zero_filled_or_bridged(self):
        summary=temporal_mean([0,1,2,3],[20,np.nan,40,60],[0,3])
        self.assertIsNone(summary['interaction_temporal_mean_pct'])
        self.assertAlmostEqual(summary['valid_time_temporal_mean_pct'],50)
        self.assertAlmostEqual(summary['interaction_coverage_pct'],100/3)
        self.assertEqual(temporal_mean([0,1],[20,40],None)['temporal_mean_status'],'NO_INTERACTION_WINDOW')

    def test_absolute_percentage_is_integrated_without_force_weighting(self):
        result=force_shares(np.array([1.,0.]),np.array([0.,9.]),np.array([1.,9.]),np.ones(2,bool))
        summary=temporal_mean([0,1],result['force_tangential_pct'],[0,1])
        self.assertAlmostEqual(summary['interaction_temporal_mean_pct'],50/(1+1e-8))
        absolute=force_shares(np.array([-1.,1.]),np.zeros(2),np.ones(2),np.ones(2,bool))
        self.assertAlmostEqual(temporal_mean([0,1],absolute['force_tangential_pct'],[0,1])['interaction_temporal_mean_pct'],100/(1+1e-8))

    def test_moment_rotation_preserves_origin_norm_and_stationary_measurement(self):
        sensor=np.tile([.1,.2,.3],(2,1));rv=np.tile([np.pi/2,0,0],(2,1))
        mount=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        tangent=np.array([[1.,0,0],[np.nan,np.nan,np.nan]])
        result=torque_components(sensor,rv,mount,tangent,np.array([True,False]))
        np.testing.assert_allclose(result['torque_ee_Nm'],np.tile([-.2,.1,.3],(2,1)),atol=1e-12)
        np.testing.assert_allclose(result['torque_base_Nm'],np.tile([-.2,-.3,.1],(2,1)),atol=1e-12)
        np.testing.assert_allclose(result['torque_norm_Nm'],np.linalg.norm(sensor,axis=1),atol=1e-12)
        self.assertAlmostEqual(result['torque_tangent_signed_Nm'][0],-.2)
        np.testing.assert_allclose(result['torque_tangent_base_Nm'][0]+result['torque_perpendicular_base_Nm'][0],result['torque_base_Nm'][0])
        self.assertTrue(np.isnan(result['torque_tangent_signed_Nm'][1]))
        self.assertTrue(np.isfinite(result['torque_base_Nm'][1]).all())
        np.testing.assert_array_equal(result['torque_sensor_Nm'],sensor)


if __name__=='__main__':unittest.main()
