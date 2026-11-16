"""Independent numerical oracles and complete read-only/resume integration tests."""
from pathlib import Path
import sys
import tempfile
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import polars as pl
from scipy.spatial.transform import Rotation

AUDIT=Path(__file__).resolve().parents[1]/'samuele/audits/real_rgbd'
if not AUDIT.is_dir():AUDIT=Path(__file__).resolve().parents[1]/'scripts/audits/real_rgbd'
sys.path.insert(0,str(AUDIT))
from audit_common import Settings,distribution,cluster_interval,sha
from audit_registration import load_converter,camera_context,trace_registration,raster_counterfactual
from audit_geometry import project,transform_points,triangulate,triangulation_sigma,dense_pair,fundamental,epipolar,patch_planes
from audit_rgb import lag_rgb
from audit_temporal import NativeSeries
from audit_real_rgbd import main
from audit_common import table,stable,join_partitions
from audit_reporting import heldout_models,patch_track_intervals
from finalize_real_rgbd import main as finalize_main
from audit_measurements import attribution_rows,coverage_regions

CONVERTER=Path('/home/samuelemara/FreeArtGS/samuele/convert_real_recording.py')
converter=load_converter(CONVERTER)


class GeometryOracles(unittest.TestCase):
    def test_optical_transform_6_degrees_32_mm_and_inverse(self):
        T=np.eye(4);T[:3,:3]=Rotation.from_euler('y',6,degrees=True).as_matrix();T[0,3]=.032
        point=np.array([[.1,.2,1.]])
        c,s=np.cos(np.deg2rad(6)),np.sin(np.deg2rad(6))
        expected=np.array([[c*.1+s+.032,.2,-s*.1+c]])
        np.testing.assert_allclose(transform_points(point,T),expected,atol=1e-12)
        np.testing.assert_allclose(transform_points(expected,np.linalg.inv(T)),point,atol=1e-12)
        self.assertGreater(expected[0,2],0)

    def test_all_eight_rational_distortion_coefficients_at_corners(self):
        K=np.array([[18.,0,8],[0,19,6],[0,0,1.]])
        D=np.array([.15,.02,.002,-.001,.01,.1,.03,.002])
        camera=dict(width=16,height=12,K=K.ravel().tolist(),D=D.tolist(),distortion_model='rational_polynomial')
        rays,valid=converter.depth_rays(camera);x=rays[:,0];y=rays[:,1];r=x*x+y*y
        radial=(1+D[0]*r+D[1]*r*r+D[4]*r*r*r)/(1+D[5]*r+D[6]*r*r+D[7]*r*r*r)
        u=K[0,0]*(x*radial+2*D[2]*x*y+D[3]*(r+2*x*x))+K[0,2]
        v=K[1,1]*(y*radial+D[2]*(r+2*y*y)+2*D[3]*x*y)+K[1,2]
        yy,xx=np.indices((12,16));np.testing.assert_allclose(u,xx.ravel(),atol=1e-7);np.testing.assert_allclose(v,yy.ravel(),atol=1e-7)
        self.assertTrue(valid.all())
        truncated=dict(camera,D=D[:5].tolist());wrong,_=converter.depth_rays(truncated)
        self.assertGreater(np.max(np.abs(wrong-rays)),.001)

    def test_nearest_color_z_winner_and_valid_map(self):
        raw=np.array([[2000,1000,0]],np.uint16)
        context=dict(rays=np.array([[0,0,1],[0,0,1],[0,0,1.]]),ray_valid=np.array([True,True,True]),
            T_color_depth=np.eye(4),K=np.eye(3),size=(2,1),color_valid=np.ones((1,2),bool))
        depth,stats,maps=trace_registration(raw,context,converter)
        np.testing.assert_array_equal(depth,[[1.,0.]])
        np.testing.assert_array_equal(maps['winner_native_index'],[[1,-1]])
        self.assertEqual(stats['collision_samples'],1);self.assertEqual(stats['native_zero_count'],1)
        context['color_valid'][0,0]=False
        depth,_,maps=trace_registration(raw,context,converter)
        self.assertEqual(depth.sum(),0);self.assertTrue((maps['winner_native_index']==-1).all())

    def test_measured_point_counterfactual_fills_holes_without_changing_baseline(self):
        raw=np.full((4,4),1000,np.uint16);yy,xx=np.indices(raw.shape)
        context=dict(rays=np.c_[xx.ravel(),yy.ravel(),np.ones(16)],ray_valid=np.ones(16,bool),T_color_depth=np.eye(4),K=np.diag([2.,2.,1.]),size=(8,8),color_valid=np.ones((8,8),bool))
        depth,_,maps=trace_registration(raw,context,converter);original=depth.copy()
        rows,fill=raster_counterfactual(raw,context,maps,np.zeros((8,8,3),np.uint8),Settings())
        self.assertGreater(fill.sum(),0);np.testing.assert_array_equal(depth,original)
        self.assertEqual(rows[0]['measured_source_support'],16)

    def test_tilted_plane_registration_against_independent_pixel_loop(self):
        yy,xx=np.indices((12,16));rays=np.c_[(xx.ravel()-8)/30,(yy.ravel()-6)/30,np.ones(192)]
        z=1.2/(1+.2*rays[:,0]-.1*rays[:,1]);raw=np.rint(z.reshape(12,16)*1000).astype(np.uint16)
        T=np.eye(4);T[:3,:3]=Rotation.from_euler('y',6,degrees=True).as_matrix();T[0,3]=.032
        K=np.array([[80.,0,24],[0,80,20],[0,0,1.]])
        context=dict(rays=rays,ray_valid=np.ones(192,bool),T_color_depth=T,K=K,size=(48,40),color_valid=np.ones((40,48),bool))
        actual,_,_=trace_registration(raw,context,converter);expected=np.zeros((40,48),np.float32)
        for ray,mm in zip(rays,raw.ravel()):
            point=T[:3,:3]@(ray*float(mm)*.001)+T[:3,3]
            u=int(np.rint(80*point[0]/point[2]+24));v=int(np.rint(80*point[1]/point[2]+20))
            if 0<=u<48 and 0<=v<40:
                value=np.float32(point[2]);expected[v,u]=value if expected[v,u]==0 else min(value,expected[v,u])
        np.testing.assert_array_equal(actual,expected)
        patches=patch_planes(raw.astype(float)*.001,rays,Settings())
        self.assertEqual(len(patches),1);self.assertLess(patches[0]['mae_m'],.0006)

    def test_empty_native_neighborhoods_have_no_counterfactual_support(self):
        raw=np.zeros((4,4),np.uint16);yy,xx=np.indices(raw.shape)
        context=dict(rays=np.c_[xx.ravel(),yy.ravel(),np.ones(16)],ray_valid=np.ones(16,bool),T_color_depth=np.eye(4),K=np.eye(3),size=(4,4),color_valid=np.ones((4,4),bool))
        with np.errstate(over='raise',invalid='raise'):
            _,_,maps=trace_registration(raw,context,converter)
            rows,fill=raster_counterfactual(raw,context,maps,np.zeros((4,4,3),np.uint8),Settings())
        self.assertFalse(fill.any())
        self.assertTrue(all(row['measured_source_support']==0 for row in rows))

    def test_dense_plane_zero_residual_and_occlusion_not_certified_as_bias(self):
        K=np.eye(3);a=np.ones((4,5),np.float32)
        row,raster=dense_pair(a,a,K,np.eye(4),np.eye(4))
        self.assertEqual(row['support_count'],20);self.assertEqual(row['signed_median_m'],0)
        row,_=dense_pair(a*2,a,K,np.eye(4),np.eye(4))
        self.assertEqual(row['signed_median_m'],1)
        self.assertIn('not_certified',row['population'])
        self.assertNotIn('sensor_bias',row)

    def test_rgb_triangulation_exact_and_feature_uncertainty(self):
        K=np.array([[500.,0,320],[0,500,240],[0,0,1.]])
        T1=np.eye(4);T2=np.eye(4);T2[0,3]=.1
        X=np.array([[.02,.03,1.],[.13,-.04,1.5],[.1,.1,2.]])
        p=project(X,K);q=project(X-[.1,0,0],K)
        recovered,angle,error,z=triangulate(p,q,K,K,T1,T2)
        np.testing.assert_allclose(recovered,X,atol=1e-10);self.assertTrue((angle>1).all());self.assertLess(error.max(),1e-8)
        np.testing.assert_allclose(epipolar(p,q,fundamental(K,K,T1,T2)),0,atol=1e-10)
        _,sigma=triangulation_sigma(p,q,K,K,T1,T2,.2)
        rng=np.random.default_rng(172);n=2000
        pnoise=np.repeat(p[:1],n,axis=0)+rng.normal(0,.2,(n,2));qnoise=np.repeat(q[:1],n,axis=0)+rng.normal(0,.2,(n,2))
        samples=triangulate(pnoise,qnoise,K,K,T1,T2)[0][:,2]
        self.assertLess(abs(np.std(samples)/sigma[0,0]-1),.1)
        weak=T2.copy();weak[0,3]=1e-5
        _,angle,_,_=triangulate(p,project(X-[1e-5,0,0],K),K,K,T1,weak)
        self.assertTrue((angle<.01).all())

    def test_native_repeatability_does_not_measure_constant_bias(self):
        with tempfile.TemporaryDirectory() as tmp:
            series=NativeSeries(tmp,'cam_000',list(range(8)),(2,2))
            for i in range(8):series.put(i,np.full((2,2),1030,np.uint16),np.ones((2,2),bool))
            report=series.finalize()
            self.assertEqual(report['temporal_repeatability_statistics']['mae_m'],0)
            self.assertIn('absolute depth accuracy',report['inference_limit'])
            series.release_scratch()

    def test_lag_direction_with_static_negative_control(self):
        settings=Settings();K=np.array([[200.,0,60],[0,200,60],[0,0,1.]])
        T1=np.eye(4);T2=np.eye(4);T2[0,3]=.1
        X=np.array([[.01,.01,1.],[.03,.04,1.],[.05,.07,1.],[.09,-.06,1.]])
        p=project(X,K);static=np.zeros((120,120),bool);mask=np.ones_like(static);motion=np.ones((120,120))*8
        xy=np.rint(p[-1]).astype(int);static[xy[1],xy[0]]=True;motion[xy[1],xy[0]]=0
        a=dict(depth=np.ones((120,120)),mask=mask,stationary=static,motion=motion,features=dict(keypoints=p))
        values={}
        with patch('audit_rgb.match_rgb',return_value=(np.c_[np.arange(4),np.arange(4)],np.zeros(4),dict(status='measured'))):
            for lag in range(-3,4):
                moved=X.copy();moved[:3,1]+=(lag-2)*.005
                b=dict(depth=np.ones((120,120)),features=dict(keypoints=project(moved-[.1,0,0],K)))
                result=lag_rgb(a,b,K,T1,T2,settings);values[lag]=result['moving_epipolar_median_px'];self.assertAlmostEqual(result['static_epipolar_median_px'],0)
            self.assertEqual(min(values,key=values.get),2)
            a['motion'][:]=0;result=lag_rgb(a,b,K,T1,T2,settings)
            self.assertIsNone(result['moving_epipolar_median_px'])

    def test_signed_tails_missingness_and_equal_scene_uncertainty(self):
        row=distribution([-2.,-.01,0,.01,3.,np.nan]);self.assertEqual(row['count'],5);self.assertEqual(row['nonfinite_count'],1)
        self.assertEqual(sum(row['signed_histogram_counts']),5);self.assertGreater(row['rmse_m'],1)
        self.assertIsNone(distribution([])['mae_m'])
        ci=cluster_interval([0.]*100+[1.],['a']*100+['b'],draws=50)
        self.assertEqual(ci['estimate'],.5);self.assertIsNone(cluster_interval([.01]*100,['a']*100)['ci95'])

    def test_injected_scale_pose_and_wrong_rgb_depth_direction_remain_distinct(self):
        K=np.array([[300.,0,160],[0,300,120],[0,0,1.]])
        a=np.ones((6,8),np.float32)
        statistics,_=dense_pair(a*1.03,a,K,np.eye(4),np.eye(4))
        self.assertAlmostEqual(statistics['signed_median_m'],.03,places=6)
        X=np.array([[.05,.04,1.],[.1,-.03,1.5],[.2,.1,2.]])
        T1=np.eye(4);T2=np.eye(4);T2[0,3]=.1
        p=project(X,K);q=project(X-[.1,0,0],K)
        wrong=T2.copy();wrong[:3,:3]=Rotation.from_euler('x',6,degrees=True).as_matrix();wrong[1,3]=.032
        self.assertGreater(np.median(np.abs(epipolar(p,q,fundamental(K,K,T1,wrong)))),1.)
        raw=np.array([[1000]],np.uint16);T=np.eye(4);T[0,3]=.1
        context=dict(rays=np.array([[0.,0.,1.]]),ray_valid=np.array([True]),T_color_depth=T,K=np.diag([10.,10.,1.]),size=(3,1),color_valid=np.ones((1,3),bool))
        registered,_,_=trace_registration(raw,context,converter)
        self.assertEqual(registered[0,1],1.)
        context['T_color_depth']=np.linalg.inv(T)
        wrong,_,_=trace_registration(raw,context,converter);self.assertEqual(wrong.sum(),0.)

    def test_false_or_missing_mask_never_establishes_static_background_or_bias(self):
        depth=np.ones((8,8));mask=np.zeros((8,8),bool);mask[:2,:2]=True
        regions=coverage_regions(depth,mask,np.ones((8,8),bool))
        self.assertEqual(regions['outside_object_unknown_not_assumed_static']['pixels'],60)
        scene=dict(id='x',name='control',source_by_frame={0:0},calibration_sha256='c')
        evidence=attribution_rows(scene,0,'cam_000',dict(native_status='not_measurable'))
        self.assertTrue(all(r['evidence_grade']=='UNOBSERVABLE' for r in evidence))

    def test_heldout_models_recover_injected_arms_without_claiming_causality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[];rng=np.random.default_rng(9)
            names=[f'control_{i}' for i in range(30)]
            selected=[n for n in names if int(stable(n)[:8],16)%5==0][:3]+[n for n in names if int(stable(n)[:8],16)%5!=0][:4]
            delta=np.array([0.,-.002,.001,.001,-.0005,.0002])
            for name in selected:
                for frame in range(12):
                    derivative=rng.normal(size=6)*[100,100,100,200,200,200];z=.5+frame*.1
                    rows.append(dict(dataset_id=name,dataset=name,camera='cam_000',target_camera='cam_001',calibration_sha256='rig',frame_index=frame,status='measured',
                        region_i='conditional_stationary_rgb',region_j='conditional_stationary_rgb',geometry_accepted=True,edge_safe_i=True,
                        signed_sampson_px=-float(derivative@delta),epipolar_derivative_px_per_m=derivative[:3].tolist(),epipolar_derivative_px_per_rad=derivative[3:].tolist(),fixed_baseline_world_m=[.1,0,0],
                        rgb_depth_i_m=z,sensor_minus_rgb_i_m=.03+.01*z))
            table(root/'correspondences.parquet',rows);models=heldout_models([root/'correspondences.parquet'],Settings(),root)
            self.assertEqual(len(models),2)
            for model in models:
                self.assertTrue(model['heldout_scenes']);self.assertGreater(model['heldout_improvement']['estimate'],0)
                self.assertEqual(model['evidence_grade'],'POSSIBLE');self.assertFalse(model['production_parameters_changed'])
                expected=delta[1:] if model['arm'].startswith('A4') else [.03,.01]
                np.testing.assert_allclose(model['coefficients'],expected,atol=1e-7)


def make_real_fixture(root):
    raw=root/'assets/recording';(raw/'meta').mkdir(parents=True);(raw/'data/chunk-000').mkdir(parents=True)
    native=dict(width=32,height=24,K=[30,0,16,0,30,12,0,0,1],D=[0]*8,distortion_model='rational_polynomial')
    cameras={}
    rng=np.random.default_rng(9);texture=rng.integers(0,256,(24,32,3),dtype=np.uint8)
    for i in range(2):
        name=f'cam_{i}';pose=np.eye(4);pose[0,3]=i*.03
        cameras[name]=dict(color=native,depth=dict(native,units='uint16_mm'),T_color_depth=np.eye(4).tolist(),T_base_color_optical=pose.tolist())
        videos=raw/f'videos/chunk-000/observation.images.{name}';videos.mkdir(parents=True)
        depths=raw/f'depth/chunk-000/observation.depth.{name}/episode_000000';depths.mkdir(parents=True)
        writer=cv2.VideoWriter(str(videos/'episode_000000.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25.,(32,24));assert writer.isOpened()
        for f in range(8):
            writer.write(texture)
            if f%2==0:cv2.imwrite(str(depths/f'frame_{f:06d}.png'),np.full((24,32),1000,np.uint16))
        writer.release()
    (raw/'meta/camera_calibration.json').write_text(json.dumps(dict(cameras=cameras,base_frame='base_link',status='UNVERIFIED_SYNTHETIC_TEST_CONTROL',extrinsic_convention='T_base_color_optical; p_base = T_base_color_optical * p_color_optical')))
    (raw/'meta/info.json').write_text(json.dumps(dict(fps=25)))
    pl.DataFrame(dict(frame_index=list(range(8)),episode_index=[0]*8,timestamp=[i/25 for i in range(8)])).write_parquet(raw/'data/chunk-000/episode_000000.parquet')
    scene=root/'converted/control_episode_000000'
    converter.main(['--input',str(raw),'--output',str(scene),'--size','32','24','--point-stride','2'])
    for cam in ['cam_000','cam_001']:
        folder=scene/'mask'/cam;folder.mkdir(parents=True)
        for f in range(4):cv2.imwrite(str(folder/f'frame_{f:06d}.png'),np.full((24,32),255,np.uint8))
    return raw,scene


class AuditIntegration(unittest.TestCase):
    def test_bounded_merge_preserves_all_rows_and_nullable_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for i in range(3):
                p=root/f'{i}.parquet';table(p,[dict(dataset='scene',frame_index=i,support_count=None,large_diagnostic='x'*10000)]);paths.append(p)
            with patch.object(pl.LazyFrame,'sink_parquet',side_effect=AssertionError('multi-file sink must not run')):
                join_partitions(paths,root/'merged.parquet')
            result=pl.read_parquet(root/'merged.parquet')
            self.assertEqual(result.height,3);self.assertEqual(result['frame_index'].to_list(),[0,1,2])
            self.assertEqual(result['support_count'].null_count(),3)
            self.assertTrue(all(json.loads(r)['large_diagnostic']=='x'*10000 for r in result['metrics_json']))
            import pyarrow.parquet as pq
            metadata=pq.read_metadata(root/'merged.parquet')
            self.assertTrue(all(metadata.row_group(i).num_rows<=1024 for i in range(metadata.num_row_groups)))

    def test_patch_track_compaction_preserves_disjoint_frame_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for scene in ['a','b']:
                for start in [0,4]:
                    p=root/scene/f'chunk_{start:06d}'/'patches.parquet'
                    table(p,[dict(dataset=scene,camera='cam_000',frame_index=f,patch_x_px=0,patch_y_px=0,
                        population='native_local_patch',stationary_rgb_evidence_fraction=.9,
                        plane_normal=[0,0,1],centroid_m=[0,0,1+f*.001],unused_histogram=list(range(1000))) for f in range(start,start+4)])
                    paths.append(p)
            rows=patch_track_intervals(paths,Settings(),root)
            self.assertEqual(len(rows),2)
            for row in rows:
                self.assertEqual(row['train_frames'],[0,2,4,6]);self.assertEqual(row['heldout_frames'],[1,3,5,7])
                np.testing.assert_allclose(row['heldout_centroid_signed_m'],[-.002,0,.002,.004],atol=1e-12)

    def test_report_recovery_keeps_measurements_and_refuses_corrupt_partitions(self):
        import audit_real_rgbd as driver
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw,scene=make_real_fixture(root)
            args=['--converted-roots',str(root/'converted'),'--raw-roots',str(root/'assets'),'--output-root',str(root/'audit'),'--mode','full','--all-scenes','--all-frames','--all-cameras','--chunk-frames','2']
            with patch.object(driver,'summarize',side_effect=RuntimeError('interrupted reporting')):
                with self.assertRaisesRegex(RuntimeError,'interrupted reporting'):main(args)
            manifest_sha=sha(root/'audit/manifest.json')
            original={p:sha(p) for p in (root/'audit/partitions').rglob('*') if p.is_file()}
            with patch.object(driver,'scene_audit',side_effect=AssertionError('must not repeat measurements')):
                self.assertEqual(finalize_main(['--audit-root',str(root/'audit')]),0)
            self.assertEqual(sha(root/'audit/manifest.json'),manifest_sha)
            self.assertEqual(pl.read_parquet(root/'audit/frames.parquet').height,8)
            for p,digest in original.items():self.assertEqual(sha(p),digest)
            bad=next((root/'audit/partitions').rglob('patches.parquet'));bad.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash changed'):finalize_main(['--audit-root',str(root/'audit')])

    def test_partial_chunk_resume_has_no_duplicate_observations(self):
        import audit_real_rgbd as driver
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw,scene=make_real_fixture(root)
            args=['--converted-roots',str(root/'converted'),'--raw-roots',str(root/'assets'),'--output-root',str(root/'audit'),'--mode','full','--all-scenes','--all-frames','--all-cameras','--chunk-frames','2']
            original=driver.table
            def interrupted(path,rows):
                if Path(path).parent.name=='chunk_000001' and Path(path).name=='frames.parquet':raise RuntimeError('simulated power interruption')
                return original(path,rows)
            with patch.object(driver,'table',side_effect=interrupted):self.assertNotEqual(main(args),0)
            self.assertEqual(pl.read_parquet(root/'audit/frames.parquet').height,4)
            self.assertEqual(main(args+['--resume']),0)
            result=pl.read_parquet(root/'audit/frames.parquet')
            self.assertEqual(result.height,8);self.assertEqual(result.select(['dataset_id','frame_index','camera']).unique().height,8)

    def test_exhaustive_row_receipts_read_only_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw,scene=make_real_fixture(root)
            before={p:sha(p) for base in [raw,scene] for p in base.rglob('*') if p.is_file()}
            args=['--converted-roots',str(root/'converted'),'--raw-roots',str(root/'assets'),'--output-root',str(root/'audit'),'--mode','full','--all-scenes','--all-frames','--all-cameras','--chunk-frames','2']
            self.assertEqual(main(args),0)
            frames=pl.read_parquet(root/'audit/frames.parquet');pairs=pl.read_parquet(root/'audit/pairs.parquet')
            self.assertEqual(frames.height,8);self.assertEqual(pairs.height,8)
            self.assertEqual(frames.select(['dataset_id','frame_index','camera']).unique().height,8)
            self.assertEqual(frames.get_column('depth_replay_different_pixels').sum(),0)
            self.assertEqual(main(args+['--resume']),0)
            self.assertEqual(pl.read_parquet(root/'audit/frames.parquet').height,8)
            for p,digest in before.items():self.assertEqual(sha(p),digest)
            cv2.imwrite(str(scene/'mask/cam_000/frame_000000.png'),np.zeros((24,32),np.uint8))
            self.assertNotEqual(main(args+['--resume']),0)

    def test_reject_output_inside_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw,scene=make_real_fixture(root)
            with self.assertRaises(SystemExit):main(['--converted-roots',str(root/'converted'),'--raw-roots',str(root/'assets'),'--output-root',str(scene/'audit')])


if __name__=='__main__':unittest.main()
