"""Depth-independent SIFT correspondence, calibration controls, and timing evidence."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from audit_common import distribution
from audit_geometry import fundamental,epipolar,triangulate,triangulation_sigma,depth_sample,transform_points,fit_plane


def image_features(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    points,descriptors=cv2.SIFT_create(nfeatures=0).detectAndCompute(gray,None)
    return dict(gray=gray,keypoints=np.array([p.pt for p in points],np.float32).reshape(-1,2),descriptors=descriptors,
                keypoint_sizes=np.array([p.size for p in points]),localization_note='SIFT scale is not a calibrated localization uncertainty')


def temporal_rgb(rgb,previous,mask,settings):
    gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY);h,w=gray.shape
    gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    if previous is None:return np.zeros((h,w),bool),np.full((h,w),np.nan,np.float32),dict(status='not_measurable_first_frame',stationary_evidence_pixels=0)
    old=cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY)
    flow=cv2.calcOpticalFlowFarneback(old,gray,None,.5,3,15,3,5,1.2,0)
    magnitude=cv2.magnitude(flow[:,:,0],flow[:,:,1]);difference=np.abs(gray.astype(float)-old)
    repeated=np.array_equal(gray,old)
    evidence=(magnitude<=settings.stationary_flow_px)&(difference<=settings.stationary_rgb_difference_dn)&(gradient>=settings.min_texture_dn)
    if mask is not None:evidence&=~mask
    if repeated:evidence[:]=False
    return evidence,magnitude,dict(status='conditional_rgb_stationarity_not_assumed_from_mask',
        stationary_evidence_pixels=int(evidence.sum()),rgb_exact_repeat=repeated,
        near_duplicate_rgb_mae_dn=float(difference.mean()),flow_median_px=float(np.median(magnitude)),flow_p95_px=float(np.percentile(magnitude,95)),
        flow_sensitivity_counts={str(t):int(((magnitude<=t)&(difference<=settings.stationary_rgb_difference_dn)&(gradient>=settings.min_texture_dn)).sum()) for t in [.125,.25,.5,1.]})


def match_rgb(a,b,settings):
    p=a['features'];q=b['features'];d1=p['descriptors'];d2=q['descriptors']
    if d1 is None or d2 is None or min(len(d1),len(d2))<2:return np.empty((0,2),int),np.empty(0),dict(status='not_measurable_insufficient_rgb_features',candidates=0)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    first=matcher.knnMatch(d1,d2,k=2);second=matcher.knnMatch(d2,d1,k=2)
    reverse={m.queryIdx:m.trainIdx for row in second if len(row)==2 for m,n in [row] if m.distance<settings.descriptor_ratio*n.distance}
    accepted=[(m.queryIdx,m.trainIdx,m.distance/max(n.distance,1e-12)) for row in first if len(row)==2 for m,n in [row] if m.distance<settings.descriptor_ratio*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
    ids=np.array([(i,j) for i,j,_ in accepted],int).reshape(-1,2)
    return ids,np.array([r for _,_,r in accepted]),dict(status='measured' if len(ids) else 'not_measurable_no_reciprocal_descriptor_matches',
        first_features=len(d1),second_features=len(d2),candidates=len(first),reciprocal_matches=len(ids),ratio_gate=settings.descriptor_ratio,depth_used_for_selection=False)


def labels_at(image,points):
    xy=np.rint(points).astype(int);h,w=image['depth'].shape;xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
    mask=image.get('mask');stationary=image.get('stationary');motion=image.get('motion')
    fg=mask[xy[:,1],xy[:,0]] if mask is not None else np.zeros(len(points),bool)
    stat=stationary[xy[:,1],xy[:,0]] if stationary is not None else np.zeros(len(points),bool)
    speed=motion[xy[:,1],xy[:,0]] if motion is not None else np.full(len(points),np.nan)
    return np.where(stat,'conditional_stationary_rgb',np.where(fg,'predicted_object','unknown')),speed


def pair_rgb(a,b,K,Ta,Tb,settings):
    ids,ratios,summary=match_rgb(a,b,settings)
    if not len(ids):return [],dict(summary,same_surface_support_count=0,absolute_metric_accuracy='unverified'),[]
    p=a['features']['keypoints'][ids[:,0]].astype(float);q=b['features']['keypoints'][ids[:,1]].astype(float)
    epi=epipolar(p,q,fundamental(K,K,Ta,Tb));X,angle,error,zrgb=triangulate(p,q,K,K,Ta,Tb)
    gate=settings.feature_sigma_px*settings.pixel_gate_sigma
    geometry=np.isfinite(X).all(axis=1)&(zrgb>0).all(axis=1)&(angle>=settings.min_parallax_deg)&(error<=gate)
    covariance=np.full((len(p),3,3),np.nan);sigma=np.full((len(p),2),np.nan)
    good=np.flatnonzero(geometry)
    if len(good):covariance[good],sigma[good]=triangulation_sigma(p[good],q[good],K,K,Ta,Tb,settings.feature_sigma_px)
    d1,safe1,spread1=depth_sample(a['depth'],p,settings.depth_edge_m);d2,safe2,spread2=depth_sample(b['depth'],q,settings.depth_edge_m)
    local=transform_points(np.c_[((np.c_[p,np.ones(len(p))]@np.linalg.inv(K).T)*d1[:,None])],np.linalg.inv(Tb)@Ta)
    supported=geometry&np.isfinite(d1)&np.isfinite(d2)&safe1&safe2
    signed=local[:,2]-d2
    label1,motion1=labels_at(a,p);label2,motion2=labels_at(b,q)
    photo=[]
    for image,points in [(a,p),(b,q)]:
        gray=image['features']['gray'];xy=np.rint(points).astype(int)
        mean=cv2.blur(gray.astype(np.float32),(5,5));texture=np.sqrt(np.maximum(0,cv2.blur(gray.astype(np.float32)**2,(5,5))-mean**2))
        hsv=cv2.cvtColor(image['rgb'],cv2.COLOR_BGR2HSV)
        photo.append((gray[xy[:,1],xy[:,0]],texture[xy[:,1],xy[:,0]],hsv[xy[:,1],xy[:,0],1]))
    # Perturbations are assumed sensitivity experiments, not uncertainty estimates.
    sensitivity={}
    for family,step in [('translation',.001),('rotation',np.deg2rad(.1))]:
        columns=[]
        for axis in range(3):
            plus=Tb.copy();minus=Tb.copy();v=np.zeros(3);v[axis]=step
            if family=='translation':plus[:3,3]+=v;minus[:3,3]-=v
            else:plus[:3,:3]=Rotation.from_rotvec(v).as_matrix()@Tb[:3,:3];minus[:3,:3]=Rotation.from_rotvec(-v).as_matrix()@Tb[:3,:3]
            columns.append((epipolar(p,q,fundamental(K,K,Ta,plus))-epipolar(p,q,fundamental(K,K,Ta,minus)))/(2*step))
        sensitivity[family]=np.stack(columns,axis=1)
    rows=[]
    for i in range(len(p)):
        reason='RGB_geometry_qualified_not_GT' if geometry[i] else 'weak_parallax' if angle[i]<settings.min_parallax_deg else 'cheirality_or_reprojection_failure'
        rows.append(dict(match_index=i,keypoint_i=int(ids[i,0]),keypoint_j=int(ids[i,1]),u_i_px=p[i,0],v_i_px=p[i,1],u_j_px=q[i,0],v_j_px=q[i,1],
            descriptor_ratio=ratios[i],signed_sampson_px=epi[i],triangulation_angle_deg=angle[i],reprojection_px=error[i],
            geometry_accepted=bool(geometry[i]),geometry_status=reason,world_xyz_m=X[i].tolist(),
            depth_i_m=d1[i],depth_j_m=d2[i],rgb_depth_i_m=zrgb[i,0],rgb_depth_j_m=zrgb[i,1],
            sensor_minus_rgb_i_m=d1[i]-zrgb[i,0],sensor_minus_rgb_j_m=d2[i]-zrgb[i,1],
            rgb_localization_sigma_px=settings.feature_sigma_px,rgb_z_sigma_i_m=sigma[i,0],rgb_z_sigma_j_m=sigma[i,1],
            world_xyz_localization_covariance_m2=covariance[i].tolist(),
            feature_sigma_sensitivity_z_i_m={str(settings.feature_sigma_px*scale):float(sigma[i,0]*scale) for scale in [.5,1,2]},
            calibration_covariance_status='unavailable; localization-only sigma is not total metric uncertainty',
            edge_safe_i=bool(safe1[i]),edge_safe_j=bool(safe2[i]),depth_patch_range_i_m=spread1[i],depth_patch_range_j_m=spread2[i],
            signed_projected_source_minus_target_m=signed[i],same_surface_supported=bool(supported[i]),region_i=label1[i],region_j=label2[i],
            rgb_motion_i_px=motion1[i],rgb_motion_j_px=motion2[i],
            brightness_i_dn=float(photo[0][0][i]),brightness_j_dn=float(photo[1][0][i]),texture_i_dn=float(photo[0][1][i]),texture_j_dn=float(photo[1][1][i]),saturation_i_dn=float(photo[0][2][i]),saturation_j_dn=float(photo[1][2][i]),
            epipolar_derivative_px_per_m=sensitivity['translation'][i].tolist(),epipolar_derivative_px_per_rad=sensitivity['rotation'][i].tolist()))
        rows[-1]['fixed_baseline_world_m']=(Tb[:3,3]-Ta[:3,3]).tolist()
    summary.update(rgb_matches=len(ids),geometry_accepted_count=int(geometry.sum()),same_surface_support_count=int(supported.sum()),
        sampson_abs_median_px=float(np.median(np.abs(epi))),sampson_abs_p95_px=float(np.percentile(np.abs(epi),95)),
        rgb_field_coverage_i_fraction=float(np.prod(np.ptp(p,axis=0))/np.prod(a['depth'].shape[::-1])),
        triangulation_parallax_sensitivity={str(t):int((np.isfinite(X).all(axis=1)&(zrgb>0).all(axis=1)&(angle>=t)&(error<=gate)).sum()) for t in [.5,1.,2.,5.]},
        same_surface_statistics=distribution(signed[supported]),
        sensor_minus_rgb_i_statistics=distribution((d1-zrgb[:,0])[geometry&safe1&np.isfinite(d1)]),
        sensor_minus_rgb_j_statistics=distribution((d2-zrgb[:,1])[geometry&safe2&np.isfinite(d2)]),
        absolute_metric_accuracy='unverified_without_independent_baseline_calibration_uncertainty',
        independent_selection='reciprocal SIFT + RGB reprojection/parallax; never depth residual magnitude')
    summary['same_surface_by_region']={label:distribution(signed[supported&(label1==label)&(label2==label)]) for label in ['conditional_stationary_rgb','predicted_object','unknown']}
    summary['epipolar_status']='measured_reciprocal_RGB_candidates'
    summary['triangulation_status']='measured_conditionally' if geometry.any() else 'not_measurable_weak_parallax_or_inconsistent_geometry'
    summary['same_surface_depth_status']='measured_conditionally' if supported.any() else 'not_measurable_no_edge_safe_RGB_depth_support'
    summary['assumed_calibration_sensitivity']=[]
    for family,levels in [('translation',settings.calibration_translation_sweep_m),('rotation',settings.calibration_rotation_sweep_deg)]:
        for level in levels:
            amount=level if family=='translation' else np.deg2rad(level)
            for axis in range(3):
                delta=sensitivity[family][:,axis]*amount
                summary['assumed_calibration_sensitivity'].append(dict(family=family,assumed_amplitude=level,unit='m' if family=='translation' else 'deg',axis=axis,
                    baseline_abs_sampson_median_px=float(np.median(np.abs(epi))),positive_perturbation_abs_median_px=float(np.median(np.abs(epi+delta))),negative_perturbation_abs_median_px=float(np.median(np.abs(epi-delta))),
                    interpretation='first-order fixed-geometry sensitivity, not measured calibration uncertainty or a chosen correction'))
    planes=matched_planes(a,b,p,q,geometry,K,Ta,Tb,settings)
    return rows,summary,planes


def matched_planes(a,b,p,q,geometry,K,Ta,Tb,settings):
    rows=[];tiles=np.floor(p/settings.patch_pixels).astype(int)
    for tile in np.unique(tiles,axis=0):
        candidates=np.flatnonzero(geometry&(tiles==tile).all(axis=1))
        if len(candidates)<8:continue
        H,inliers=cv2.findHomography(p[candidates],q[candidates],cv2.RANSAC,settings.feature_sigma_px*settings.pixel_gate_sigma)
        if H is None or inliers is None:continue
        support=candidates[inliers.ravel()>0]
        if len(support)<8 or min(np.linalg.matrix_rank(p[support]-p[support].mean(0)),np.linalg.matrix_rank(q[support]-q[support].mean(0)))<2:continue
        d1,s1,_=depth_sample(a['depth'],p[support],settings.depth_edge_m);d2,s2,_=depth_sample(b['depth'],q[support],settings.depth_edge_m)
        valid=s1&s2&np.isfinite(d1)&np.isfinite(d2)
        if valid.sum()<8:continue
        uv1=p[support][valid];uv2=q[support][valid]
        X1=(np.c_[uv1,np.ones(len(uv1))]@np.linalg.inv(K).T)*d1[valid,None]
        X2=(np.c_[uv2,np.ones(len(uv2))]@np.linalg.inv(K).T)*d2[valid,None]
        # Split RGB feature IDs before fitting; evaluation anchors are held out.
        train=np.arange(len(X1))%2==0;held=~train
        if min(train.sum(),held.sum())<6:continue
        model1=fit_plane(X1[train]);model2=fit_plane(X2[train])
        if model1 is None or model2 is None:continue
        n1,d1p=model1;n2,d2p=model2
        world1=transform_points(X1[held],Ta);world2=transform_points(X2[held],Tb)
        nw1=Ta[:3,:3]@n1;dw1=d1p-nw1@Ta[:3,3];nw2=Tb[:3,:3]@n2;dw2=d2p-nw2@Tb[:3,3]
        if nw1@nw2<0:nw2=-nw2;dw2=-dw2
        values=world1@nw2+dw2;statistics=distribution(values)
        rows.append(dict(population='RGB_homography_supported_local_plane_candidates',patch_x_px=int(tile[0]*settings.patch_pixels),patch_y_px=int(tile[1]*settings.patch_pixels),
            support_count=int(held.sum()),normal_angle_deg=float(np.degrees(np.arccos(np.clip(nw1@nw2,-1,1)))),
            normal_i_world=nw1.tolist(),normal_j_world=nw2.tolist(),plane_i_world_offset_m=dw1,plane_j_world_offset_m=dw2,
            signed_common_support_statistics=statistics,signed_median_m=statistics['signed_median_m'],mae_m=statistics['mae_m'],
            camera_i_self_shape_heldout=distribution(X1[held]@n1+d1p),camera_j_self_shape_heldout=distribution(X2[held]@n2+d2p),
            physical_identity_evidence='reciprocal RGB features with homography; finite spatial support; plane fits on disjoint features',
            absolute_bias_status='unverified; normals and physical anchors reported, offsets not blindly subtracted'))
    return rows


def lag_rgb(a,b,K,Ta,Tb,settings):
    ids,_,quality=match_rgb(a,b,settings)
    if not len(ids):return dict(quality,static_support=0,moving_support=0,static_epipolar_median_px=None,moving_epipolar_median_px=None)
    p=a['features']['keypoints'][ids[:,0]];q=b['features']['keypoints'][ids[:,1]]
    values=np.abs(epipolar(p,q,fundamental(K,K,Ta,Tb)));labels,speed=labels_at(a,p)
    stationary=labels=='conditional_stationary_rgb';moving=(labels=='predicted_object')&np.isfinite(speed)&(speed>settings.pixel_gate_sigma*settings.feature_sigma_px)
    return dict(quality,static_support=int(stationary.sum()),moving_support=int(moving.sum()),
        static_epipolar_median_px=float(np.median(values[stationary])) if stationary.any() else None,
        moving_epipolar_median_px=float(np.median(values[moving])) if moving.any() else None,
        interpretation='image-content relative lag evidence; not hardware clock offset')
