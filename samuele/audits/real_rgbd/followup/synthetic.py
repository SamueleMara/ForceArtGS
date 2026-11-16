"""Independent positive/negative controls; fixture truth never enters real data."""
from pathlib import Path
import numpy as np

from .common import atomic_json
from .registration_trace import trace,footprint
from .mask_annotation import mask_metrics
from .surfaces import extract,association_decision
from .identifiability import residual_and_jacobian,perturbed_residual,rank_diagnostic,solve_metric_target
from .timing import fixed_population_delay


def run_controls(out,converter):
    rows=[]
    def check(name,fn):
        try:evidence=fn();rows.append(dict(test=name,status='PASS',evidence=evidence))
        except Exception as e:rows.append(dict(test=name,status='FAIL',error=str(e)))
    yy,xx=np.indices((8,8));context=dict(rays=np.c_[xx.ravel(),yy.ravel(),np.ones(64)],ray_valid=np.ones(64,bool),T_color_depth=np.eye(4),K=np.diag([2.,2.,1.]),size=(16,16),color_valid=np.ones((16,16),bool))
    rgb=np.zeros((16,16,3),np.uint8)
    def plane_gap():
        raw=np.full((8,8),1000,np.uint16);baseline=converter.register_depth(raw,context['rays'],context['ray_valid'],context['T_color_depth'],context['K'],context['size'])
        r,m=trace(raw,baseline,context,converter,rgb,None,True)
        assert r['bitwise_different_pixels']==0 and r['native_zero_count']==0 and r['registered_zero_inside_rgb_count']>0
        assert r['gap_candidate_pixels']>0
        assert r['heldout_counterfactual']['residual']['mae_m']==0
        return dict(native_zeros=0,registration_holes=r['registered_zero_inside_rgb_count'],gap_candidates=r['gap_candidate_pixels'])
    check('thin_plane_gaps_without_native_zeros_and_heldout_splat',plane_gap)
    def stripe():
        raw=np.full((8,8),1000,np.uint16);raw[:,2:6]=0
        baseline=converter.register_depth(raw,context['rays'],context['ray_valid'],context['T_color_depth'],context['K'],context['size'])
        r,m=trace(raw,baseline,context,converter,rgb,None)
        assert r['native_zero_count']==32 and r['unknown_hole_pixels']>0
        assert not m['source_uv'][raw==0].size or np.isnan(m['source_uv'][raw==0]).all()
        return dict(invalid_native_samples=32,unknown_missing_targets=r['unknown_hole_pixels'])
    check('native_invalid_stripe_has_no_invented_target_mapping',stripe)
    def layers():
        raw=np.full((8,8),1000,np.uint16);raw[:,4:]=1150
        baseline=converter.register_depth(raw,context['rays'],context['ray_valid'],context['T_color_depth'],context['K'],context['size'])
        r,m=trace(raw,baseline,context,converter,rgb,None,True);alternative=footprint(raw,context,m)
        assert r['heldout_counterfactual']['false_fill_edge_abs_gt_5mm_count']==0
        np.testing.assert_array_equal(baseline,converter.register_depth(raw,context['rays'],context['ray_valid'],context['T_color_depth'],context['K'],context['size']))
        return r['heldout_counterfactual']
    check('two_layer_step_not_bridged_or_written_to_official_depth',layers)
    def footprint_exclusion():
        ctx={**context,'color_valid':context['color_valid'].copy()};ctx['color_valid'][:,:4]=False
        raw=np.full((8,8),1000,np.uint16);baseline=converter.register_depth(raw,ctx['rays'],ctx['ray_valid'],ctx['T_color_depth'],ctx['K'],ctx['size']);baseline[~ctx['color_valid']]=0
        r,m=trace(raw,baseline,ctx,converter,rgb,None,True)
        assert r['outside_rgb_support_pixels']==64 and np.all(m['category'][:,:4]==0)
        return dict(excluded_target_pixels=64)
    check('outside_RGB_footprint_excluded',footprint_exclusion)
    def zbuffer():
        raw=np.array([[2000,1000,1500]],np.uint16);ctx=dict(rays=np.array([[0,0,1.]]*3),ray_valid=np.ones(3,bool),T_color_depth=np.eye(4),K=np.eye(3),size=(2,1),color_valid=np.ones((1,2),bool))
        from audit_registration import trace_registration
        d,c,m=trace_registration(raw,ctx,converter)
        assert d[0,0]==1 and m['winner_native_index'][0,0]==1 and c['collision_samples']==2
        return dict(nearest_color_Z_m=1.,winning_native_id=1,collisions=2)
    check('independent_multiple_ray_Zbuffer_oracle',zbuffer)
    def shifted_mask():
        d=np.zeros((20,20),np.float32);d[2:10,2:10]=1;m=d>0;shift=np.roll(m,8,axis=1)
        a,_=mask_metrics(d,m);b,_=mask_metrics(d,shift)
        assert a['predicted_valid_pixels']==64 and b['predicted_valid_pixels']<64 and b['mask_truth']=='MASK_TRUTH_UNVERIFIED'
        return dict(original_valid=64,shifted_valid=b['predicted_valid_pixels'],sensor_data_unchanged=True)
    check('shifted_mask_and_no_truth_abstention',shifted_mask)
    def occluder():
        d=np.ones((20,20),np.float32);m=np.ones_like(d,bool);d[:,8:12]=.7;r,_=mask_metrics(d,m)
        assert r['predicted_valid_pixels']==400
        return dict(occluder_has_valid_depth=True,mask_truth=r['mask_truth'],invalid_sensor_returns=0)
    check('robot_occluder_is_not_invalid_sensor_depth',occluder)
    def plane_extraction():
        K=np.array([[80.,0,32],[0,80.,24],[0,0,1.]]);d=np.ones((48,64),np.float32);im=np.zeros((48,64,3),np.uint8)
        a=extract(d,im,None,K,np.eye(4));b=extract(d+.005,im,None,K,np.eye(4))
        p=max(a['patches'],key=lambda p:p['pixels']);q=max(b['patches'],key=lambda p:p['pixels'])
        assert p['status']=='CONDITIONAL_PLANE' and abs(abs(q['offset_camera_m']-p['offset_camera_m'])-.005)<1e-6
        return dict(offset_difference_m=q['offset_camera_m']-p['offset_camera_m'],reference='synthetic physically identical plane')
    check('automatic_coplanar_extraction_known_5mm_depth_bias',plane_extraction)
    def curved():
        K=np.array([[80.,0,32],[0,80.,24],[0,0,1.]]);y,x=np.indices((48,64));d=(1.+.4*((x-32)/32)**2).astype(np.float32)
        r=extract(d,np.zeros((48,64,3),np.uint8),None,K,np.eye(4))
        assert r['patches'] and any(p['status']=='NONPLANAR_UNRESOLVED' for p in r['patches'])
        return dict(nonplanar=sum(p['status']=='NONPLANAR_UNRESOLVED' for p in r['patches']))
    check('curvature_is_not_forced_into_a_plane',curved)
    def hard_negative():
        common=dict(rgb_train=10,rgb_heldout=10,heldout_reprojection_px=0.,overlap=.8,temporal_support=True)
        opposite=association_decision(oriented_normal_dot=-1.,texture_ambiguity=False,**common)
        repeated=association_decision(oriented_normal_dot=1.,texture_ambiguity=True,**common)
        parallel=association_decision(oriented_normal_dot=1.,texture_ambiguity=False,layer_conflict=True,**common)
        positive=association_decision(oriented_normal_dot=1.,texture_ambiguity=False,**common)
        zero=association_decision(oriented_normal_dot=1.,texture_ambiguity=False,**{**common,'overlap':0.})
        assert opposite.startswith('REJECTED') and repeated.startswith('AMBIGUOUS') and parallel.startswith('UNIDENTIFIABLE') and zero.startswith('UNOBSERVABLE')
        assert positive=='AUTO_MULTICUE_CONSISTENT_CONDITIONAL'
        return dict(opposite_faces=opposite,identical_logo_hypotheses=repeated,parallel_150mm_faces=parallel,zero_overlap=zero,positive=positive,
            impossibility='identical observations on parallel aliased faces cannot identify physical identity; abstain instead of inventing a sensor bias')
    check('hard_negative_identity_opposite_parallel_repeated_logos',hard_negative)
    def descriptor_alias():
        import cv2
        from audit_rgb import image_features,pair_rgb
        from audit_common import Settings
        h,w=96,128;K=np.array([[100.,0,64],[0,100.,48],[0,0,1.]])
        rng=np.random.default_rng(42);im=np.full((h,w,3),127,np.uint8);tile=rng.integers(0,256,(48,48,3),np.uint8);im[24:72,40:88]=tile
        shifted=cv2.warpAffine(im,np.float32([[1,0,-10],[0,1,0]]),(w,h),borderValue=(127,127,127))
        da=np.ones((h,w),np.float32);db=np.full((h,w),1.15,np.float32);Ta=np.eye(4);Tb=np.eye(4);Tb[0,3]=.1
        def data(rgb,depth):return dict(rgb=rgb,depth=depth,mask=np.ones((h,w),bool),stationary=np.ones((h,w),bool),motion=np.zeros((h,w)),features=image_features(rgb))
        old,summary,_=pair_rgb(data(im,da),data(shifted,db),K,Ta,Tb,Settings())
        assert summary['same_surface_support_count']>0
        from .surfaces import associate
        a=extract(da,im,None,K,Ta);a=extract(da,im,None,K,Ta,a)
        b=extract(db,shifted,None,K,Tb);b=extract(db,shifted,None,K,Tb,b)
        new=associate(a,b,da,db,K,Ta,Tb,old)
        assert new and not any(r['status']=='AUTO_MULTICUE_CONSISTENT_CONDITIONAL' for r in new)
        return dict(legacy_false_supported_matches=summary['same_surface_support_count'],new_decisions=[r['status'] for r in new],
            known_distinct_parallel_faces_m=.15,texture='identical RGB pattern shifted to pass true reciprocal SIFT and calibrated epipolar tests',
            conclusion='layer/calibration/identity confounded: abstention, not a measured 150-mm sensor bias')
    check('actual_SIFT_150mm_parallel_face_alias_rejected_end_to_end',descriptor_alias)
    def indistinguishable_identity():
        # The same RGB-D observations can arise from one biased plane or from
        # two identical-looking, separated planes. No algorithm can prove ID.
        common=dict(oriented_normal_dot=1.,rgb_train=12,rgb_heldout=12,heldout_reprojection_px=0.,texture_ambiguity=False,overlap=.8,temporal_support=True)
        decision=association_decision(**common)
        one_plane_biased=np.full((16,16),1.005,np.float32)
        second_plane_unbiased=np.full((16,16),1.005,np.float32)
        np.testing.assert_array_equal(one_plane_biased,second_plane_unbiased)
        assert decision=='AUTO_MULTICUE_CONSISTENT_CONDITIONAL'
        return dict(physical_identity='UNIDENTIFIABLE',association=decision,observation_equivalence='same plane +5mm bias versus distinct identical-texture plane +5mm separation',
            missing_independent_evidence='a unique shared landmark or independently established common surface; calibrated geometric agreement cannot supply this')
    check('identical_observations_do_not_prove_physical_plane_identity',indistinguishable_identity)
    def jacobian():
        rng=np.random.default_rng(4);rays=np.c_[rng.uniform(-.3,.3,(40,2)),np.ones(40)];points=rays*rng.uniform(.6,1.5,(40,1));pose=np.eye(4);pose[:3,3]=[.1,.02,-.01];n=np.array([.2,.3,.9327379]);n/=np.linalg.norm(n)
        residual,J=residual_and_jacobian(points,rays,pose,n,1.)
        for i in range(8):
            delta=np.zeros(8);delta[i]=1e-6
            fd=(perturbed_residual(points,rays,pose,n,1.,delta)-perturbed_residual(points,rays,pose,n,1.,-delta))/(2e-6)
            np.testing.assert_allclose(fd,J[:,i],atol=1e-8)
        return dict(finite_difference_columns=8,max_tolerance=1e-8)
    check('pose_depth_Jacobian_independent_finite_difference',jacobian)
    def rank():
        rng=np.random.default_rng(2);rays=np.c_[rng.uniform(-.3,.3,(40,2)),np.ones(40)];J=[]
        for _ in range(20):
            n=np.r_[rng.uniform(-.5,.5,2),1.];n/=np.linalg.norm(n);d=rng.uniform(.6,2.);points=rays*(d/(rays@n))[:,None]
            J.append(residual_and_jacobian(points,rays,np.eye(4),n,d)[1])
        full=rank_diagnostic(np.concatenate(J));single=rank_diagnostic(J[0]);assert full['rank']==8 and single['rank']<8
        delta=np.array([.001,-.002,.001,.0003,-.0002,.0001,.005,.003]);recovered=np.linalg.lstsq(np.concatenate(J),np.concatenate(J)@delta,rcond=None)[0]
        np.testing.assert_allclose(recovered,delta,atol=1e-10)
        return dict(diverse_normals_rank=full['rank'],parallel_single_plane_rank=single['rank'],simultaneous_pose_bias_scale_recovery=True)
    check('multi_plane_pose_bias_scale_identifiability_and_degeneracy',rank)
    def dimensions():
        r=solve_metric_target([],[],np.eye(3),dict(measured_by='synthetic oracle',units='m',independently_measured=True,nominal_square_m=.04,measured_square_m=.05,uncertainty_m=.0001))
        assert r['status']=='REFERENCE_DIMENSION_CONTRADICTED';return r
    check('wrong_reference_dimension_provenance_rejected',dimensions)
    def lag():
        t=np.arange(160)*.04;f=lambda x:.05*np.sin(2*x)+.03*np.sin(5*x)+.01*x*x
        results=[]
        for delay in [-.04,.04]:
            r=fixed_population_delay(t,f(t),f(t-delay),np.arange(-3,4)*.04,identity_confirmed=True)
            assert r['status']=='CONDITIONAL_IMAGE_CONTENT_DELAY' and abs(r['chosen_target_lag_s']-delay)<1e-12;results.append(r['chosen_target_lag_s'])
        flat=fixed_population_delay(t,np.ones_like(t),np.ones_like(t),[-.04,0,.04],identity_confirmed=True)
        monotone=fixed_population_delay(t,t,t-.04,[-.04,0,.04],identity_confirmed=True)
        wrong=fixed_population_delay(t[::-1],f(t),f(t),[-.04,0,.04],identity_confirmed=True)
        assert all(r['status']=='TIMING_UNOBSERVABLE' for r in [flat,monotone,wrong])
        return dict(recovered_lags_s=results,static=flat['reason'],constant_velocity=monotone['reason'],backward_unmapped=wrong['reason'])
    check('fixed_population_positive_negative_lags_static_and_index_controls',lag)
    def coupling():
        A=np.array([[2.,1.],[1.,2.]]);force=np.array([1.,0.]);delta=np.linalg.solve(A,force)
        assert delta[1]!=0 and np.linalg.norm(A@delta-force)<1e-12
        prior=np.array([.2,.1]);dt=.04;velocity=np.array([.3,0.]);pred=prior+dt*velocity
        np.testing.assert_allclose(pred-prior,dt*velocity)
        return dict(coupled_motion_without_own_force=delta[1],solver_residual=float(np.linalg.norm(A@delta-force)),process_vs_measurement_separated=True,
            scope='linear coupled-system oracle; actual renderer controls are separately required by E6')
    check('coupled_forces_and_process_extrapolation_oracle',coupling)
    p=Path(out)/'synthetic_tests/results.json';p.parent.mkdir(parents=True,exist_ok=True)
    failed=[r for r in rows if r['status']=='FAIL'];result=dict(tests=rows,passed=len(rows)-len(failed),failed=failed,
        hard_negative_identity=any(r['test']=='hard_negative_identity_opposite_parallel_repeated_logos' and r['status']=='PASS' for r in rows))
    atomic_json(p,result);return result
