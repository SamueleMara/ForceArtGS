from pathlib import Path
import sys
import unittest
import json
import tempfile

import cv2
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/('samuele' if (BASE/'samuele/estimate_real_tcp_joints.py').is_file() else 'scripts')))
from estimate_real_tcp_joints import spike_bounds,compare_joints,load_fitter,tracked_pair,compare_intervals,robot_exclusion,run_scene


class RealTCPJoints(unittest.TestCase):
    def test_invalid_mounting_rejects_cached_tcp_before_replacing_old_fit(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);measurement=root/'measurements/drawer';measurement.mkdir(parents=True)
            (measurement/'summary.json').write_text(json.dumps(dict(tcp_calibration=dict(tcp_calibration_valid=False))))
            output=root/'fits/drawer';output.mkdir(parents=True);old=output/'result.json';old.write_text('historical result')
            result=run_scene(dict(name='drawer'),dict(output_root=str(output.parent),measurements_root=str(measurement.parent)),[])
            self.assertEqual(result['status'],'FAILED')
            self.assertIn('Uncalibrated sensor mounting',result['error'])
            self.assertEqual(old.read_text(),'historical result')

    @classmethod
    def setUpClass(cls):
        cls.backend=load_fitter('/home/samuelemara/joint_estimation')

    def test_first_last_force_events_and_missing_data(self):
        t=np.arange(300)*.02;f=np.zeros(len(t));f[50]=2;f[53]=1;f[180]=4;f[220]=3;f[100:120]=np.nan
        result=spike_bounds(dict(timestamp_s=t,force_norm_N=f,frame_index=np.arange(len(t))))
        self.assertEqual(result['window_s'],[1.,4.4]);self.assertEqual(len(result['events']),3)
        result=spike_bounds(dict(timestamp_s=t,force_norm_N=np.zeros_like(t),frame_index=np.arange(len(t))))
        self.assertIsNone(result['window_s'])

    def test_existing_circle_fit_and_metric_model_selection(self):
        rng=np.random.default_rng(5);theta=np.linspace(-.6,.6,180);p=np.c_[.2*np.cos(theta),.2*np.sin(theta),np.zeros(len(theta))]+[.4,.3,.8]
        p+=rng.normal(0,.0002,p.shape);fit=compare_joints(p,self.backend)
        self.assertEqual(fit['best_scored_model'],'revolute');self.assertEqual(fit['heldout_best_model'],'revolute')
        self.assertLess(abs(fit['models']['revolute']['radius_m']-.2),.003)
        self.assertLess(fit['models']['revolute']['heldout_rmse_m'],.001)
        self.assertAlmostEqual(abs(fit['models']['revolute']['axis_direction'][2]),1.,places=3)

    def test_line_and_static_degeneracy(self):
        rng=np.random.default_rng(2);p=np.c_[np.linspace(0,.2,180),np.zeros(180),np.zeros(180)]+rng.normal(0,.0002,(180,3))
        fit=compare_joints(p,self.backend)
        self.assertEqual(fit['best_scored_model'],'prismatic');self.assertFalse(fit['models']['prismatic']['axis_location_identifiable'])
        self.assertEqual(compare_joints(np.zeros((100,3)),self.backend)['status'],'UNOBSERVABLE')

    def test_real_optical_flow_static_and_moving_control(self):
        rng=np.random.default_rng(7);im=rng.integers(30,180,(120,160),np.uint8);mask=np.zeros_like(im,bool);mask[35:85,50:110]=True
        other=im.copy();other[35:85,53:113]=im[35:85,50:110];next_mask=np.zeros_like(mask);next_mask[35:85,53:113]=True
        static,_=tracked_pair(im,im,mask,mask);motion,_=tracked_pair(im,other,mask,next_mask)
        self.assertFalse(static['active']);self.assertTrue(motion['active'],motion)
        self.assertGreater(motion['moving_object'],4)
        noisy=np.clip(im.astype(float)+rng.normal(0,.5,im.shape),0,255).astype(np.uint8)
        stationary,_=tracked_pair(im,noisy,mask,mask);self.assertFalse(stationary['active'],stationary)

    def test_source_frame_mapping_training_comparison_and_visual_conflict(self):
        timeline=dict(times=np.arange(10)*.04,frames=np.arange(10),motion=np.array([0,0,0,1,1,1,1,0,0,0],bool))
        force=dict(window_s=[.1,.28]);visual=dict(window_s=[.12,.24])
        training=[dict(start_frame=2,end_frame=9,path='control',direction='forward')]
        result=compare_intervals(force,visual,timeline,training)
        self.assertEqual(result['status'],'VISUALLY_CORROBORATED');self.assertEqual(result['selected_converted_frames'],[3,7])
        self.assertEqual(result['training_windows'][0]['window_s'],[.08,.36])
        self.assertAlmostEqual(result['training_windows'][0]['selected_interval_fraction_covered'],1.)
        result=compare_intervals(dict(window_s=[0,.04]),visual,timeline,[])
        self.assertEqual(result['status'],'FORCE_VISUAL_INTERVAL_CONFLICT')
        result=compare_intervals(dict(window_s=None),visual,timeline,[])
        self.assertEqual(result['selected_interval_source'],'visual_only_no_two_force_spikes')
        visual['status']='SINGLE_VIEW_IMAGE_MOTION_CONDITIONAL'
        self.assertEqual(compare_intervals(force,visual,timeline,[])['status'],'SINGLE_VIEW_CONDITIONAL_CORROBORATION')

    def test_projected_tool_and_robot_occlusion_are_not_object_motion(self):
        K=np.array([[100.,0,80],[0,100,60],[0,0,1.]])
        data=dict(ee_position_m=np.array([[0.,0.,1.]]),tcp_position_m=np.array([[0.,0.,.8]]))
        exclusion=robot_exclusion(data,0,dict(pose_matrix_4x4=np.eye(4)),K,(120,160))
        self.assertTrue(exclusion[60,80]);self.assertFalse(exclusion[0,0])
        rng=np.random.default_rng(9);im=rng.integers(30,180,(120,160),np.uint8);mask=np.zeros_like(im,bool);mask[25:95,40:120]=True
        other=im.copy();other[45:65,68:88]=im[45:65,65:85]
        ignored=np.zeros_like(mask);ignored[35:75,55:100]=True
        filtered,_=tracked_pair(im,other,mask,mask,ignored,ignored)
        self.assertFalse(filtered['active'],filtered)


if __name__=='__main__':unittest.main()
