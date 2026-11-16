"""Per-observation metrics with explicit denominators and non-exclusive evidence."""
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from audit_common import distribution,record
from audit_geometry import patch_planes,project,transform_points,depth_sample
from audit_registration import trace_registration,raster_counterfactual
from audit_temporal import array_hash


def coverage_regions(depth,mask,color_valid):
    valid=np.isfinite(depth)&(depth>0);regions={'all_pixels':np.ones(depth.shape,bool),'valid_rectified_rgb':color_valid}
    if mask is not None:
        distance=distance_transform_edt(mask)
        regions['predicted_object']=mask;regions['outside_object_unknown_not_assumed_static']=~mask
        for lower,upper in zip([0,1,2,4,8,16],[1,2,4,8,16,32]):regions[f'object_boundary_distance_{lower}_{upper}_px']=mask&(distance>=lower)&(distance<upper)
        regions['object_interior_ge_32_px']=distance>=32
    return {name:dict(pixels=int(region.sum()),valid_depth_pixels=int((region&valid).sum()),valid_fraction=float(valid[region].mean()) if region.any() else None) for name,region in regions.items()}


def frame_metrics(scene,camera,frame,data,context,converter,raw,raw_rgb,settings):
    depth=data['depth'];rgb=data['rgb'];valid=np.isfinite(depth)&(depth>0);metadata=scene['conversion'];name=camera['name']
    row=record(scene,frame,name,support_count=int(valid.sum()),depth_valid_fraction=float(valid.mean()),
        depth_zero_fraction=float(np.mean(depth==0)),depth_nonfinite_fraction=float(np.mean(~np.isfinite(depth))),depth_negative_count=int((depth<0).sum()),
        rgb_decoded_sha256=array_hash(rgb),registered_depth_array_sha256=array_hash(depth),
        absolute_metric_accuracy='unverified_no_independent_metric_target',temporal_quality=data.get('temporal_quality'),
        mask_quality='predicted_not_ground_truth' if data['mask'] is not None else 'not_measurable_missing_mask')
    gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY);rgb_edge=cv2.Canny(gray,50,150)>0
    # Discontinuity sensitivity is reported; these are not correspondence gates.
    low=cv2.erode(np.where(valid,depth,np.inf),np.ones((3,3),np.uint8));high=cv2.dilate(np.where(valid,depth,-np.inf),np.ones((3,3),np.uint8));complete=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8))>0
    span=np.subtract(high,low,out=np.zeros_like(high),where=complete)
    row['depth_edge_counts_by_level_mm']={str(mm):int((complete&(span>mm*.001)).sum()) for mm in [2,5,10,20,50]}
    depth_edges=complete&(span>settings.depth_edge_m)
    distance=distance_transform_edt(~rgb_edge)
    row['depth_edge_to_rgb_edge_median_px']=float(np.median(distance[depth_edges])) if depth_edges.any() and rgb_edge.any() else None
    row['brightness_dn_mean']=float(gray.mean());row['rgb_edge_pixels']=int(rgb_edge.sum())
    row['depth_range_statistics']=distribution(depth[valid]);row['regional_coverage']=coverage_regions(depth,data['mask'],context['color_valid'] if context else np.ones(depth.shape,bool))
    png=data['depth_png']
    expected=np.rint(np.where(np.isfinite(depth),depth,0)*1000).clip(0,65535).astype(np.uint16)
    row['png_quantization_different_pixels']=int((png!=expected).sum()) if png is not None else None
    row['png_status']='measured' if png is not None else 'not_measurable_missing_or_invalid_depth_png'
    patches=[];maps=None;native_stat=None
    if raw is None or context is None:
        row.update(native_status='not_measurable_raw_depth_or_native_calibration_missing',depth_replay_status='not_measurable',rgb_replay_status='not_measurable')
    else:
        reproduced,native_stat,maps=trace_registration(raw,context,converter)
        counterfactuals,fill=raster_counterfactual(raw,context,maps,rgb,settings)
        maps['hole_candidates']&=fill
        native_stat['sampling_gap_bracket_candidates']=int(maps['hole_candidates'].sum())
        native_stat['unresolved_holes']=native_stat['registered_zero_inside_rgb_count']-native_stat['sampling_gap_bracket_candidates']
        difference=np.abs(reproduced.astype(float)-depth);neq=(reproduced.view(np.uint32)!=depth.view(np.uint32))
        row.update(native_status='measured',native_statistics=native_stat,native_invalid_fraction=native_stat['native_invalid_fraction'],
            registered_sampling_gap_candidate_fraction=float(maps['hole_candidates'].sum()/max(1,context['color_valid'].sum())),
            depth_replay_different_pixels=int(neq.sum()),depth_replay_max_abs_m=float(np.max(difference)),
            depth_replay_median_abs_m=float(np.median(difference)),depth_replay_status='bitwise_equal' if not neq.any() else 'failed_bitwise_reproduction',
            native_array_sha256=array_hash(raw),raster_counterfactuals=counterfactuals)
        uv=maps['source_uv'];xy=np.rint(np.nan_to_num(uv,nan=-1)).astype(int)
        h,w=depth.shape;inside=np.isfinite(uv).all(axis=2)&(xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<h)
        native_stationary=np.zeros(raw.shape,bool);native_stationary[inside]=data['stationary'][xy[inside,1],xy[inside,0]]
        maps['native_stationary']=native_stationary
        patches=patch_planes(raw.astype(float)*.001,context['rays'],settings,native_stationary)
        row['native_plane_patch_candidates']=len(patches)
        if raw_rgb is not None:
            replay_rgb=cv2.remap(raw_rgb,*context['color_maps'],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            if replay_rgb.shape!=rgb.shape:row.update(rgb_replay_status='failed_shape',rgb_replay_different_pixels=None)
            else:
                rgb_difference=np.any(replay_rgb!=rgb,axis=2);row.update(rgb_replay_status='bitwise_equal' if not rgb_difference.any() else 'failed_bitwise_reproduction',rgb_replay_different_pixels=int(rgb_difference.sum()),rgb_replay_max_abs_dn=int(np.max(np.abs(replay_rgb.astype(int)-rgb))))
        else:row.update(rgb_replay_status='not_measurable_raw_video_missing_or_decode_failed',rgb_replay_different_pixels=None)
    timestamp=scene['parquet_by_source_frame'].get(scene['source_by_frame'][frame],{})
    row.update(recorded_timestamp_s=timestamp.get('timestamp'),nominal_timestamp_s=scene['source_by_frame'][frame]/scene['info'].get('fps',1) if scene['info'].get('fps',0)>0 else None,
        timestamp_quality='recorder/nominal_only; no hardware synchronization proof',source_timestamp_fields={k:timestamp.get(k) for k in scene['sensor_timestamp_candidate_fields']})
    csv_row=scene.get('timestamp_csv_by_frame',{}).get(frame)
    row['nominal_timestamp_mapping_equal']=bool(int(csv_row['source_frame_index'])==scene['source_by_frame'][frame] and row['nominal_timestamp_s'] is not None and np.isclose(float(csv_row['timestamp_s']),row['nominal_timestamp_s'],atol=1e-9)) if csv_row else None
    row['recorded_minus_nominal_timestamp_s']=float(row['recorded_timestamp_s']-row['nominal_timestamp_s']) if row['recorded_timestamp_s'] is not None and row['nominal_timestamp_s'] is not None else None
    return row,patches,maps


def attribution_rows(scene,frame,camera,metrics):
    rows=[];native=metrics.get('native_statistics',{});replay=metrics.get('depth_replay_status')
    evidence=[
      ('native_missing','CONFIRMED' if native.get('native_zero_count',0)>0 else 'REJECTED_FOR_TESTED_SCOPE' if native else 'UNOBSERVABLE',native.get('native_zero_count'),native.get('native_pixels'),'native zero values directly observed before registration'),
      ('registration_holes','POSSIBLE' if native.get('sampling_gap_bracket_candidates',0)>0 else 'UNOBSERVABLE',native.get('sampling_gap_bracket_candidates'),native.get('color_valid_pixels'),'measured continuous-neighborhood footprint brackets; correctness of unmeasured holes not independently known'),
      ('conversion_reproduction','CONFIRMED' if replay=='failed_bitwise_reproduction' else 'REJECTED_FOR_TESTED_SCOPE' if replay=='bitwise_equal' else 'UNOBSERVABLE',metrics.get('depth_replay_different_pixels'),None,'exact reuse of production registration plus independent winner accounting'),
      ('native_random_noise','UNOBSERVABLE',None,None,'requires episode temporal stationary-support aggregation; per-frame smoothness is not precision'),
      ('structured_native_shape_error','POSSIBLE' if metrics.get('native_plane_patch_candidates',0) else 'UNOBSERVABLE',metrics.get('native_plane_patch_candidates'),None,'held-out native patch shape residuals; true panel geometry is not independently established'),
      ('cross_view_depth_disagreement','UNOBSERVABLE',None,None,'see ordered pairs and separately RGB-supported populations; no guilty camera assigned'),
      ('rgb_depth_metric_disagreement','UNOBSERVABLE',None,None,'see independent RGB correspondences with localization uncertainty and unverified calibration covariance'),
      ('calibration_residual','UNOBSERVABLE',None,None,'see RGB-only epipolar candidates and held-out perturbation arms'),
      ('temporal_mismatch','UNOBSERVABLE',None,None,'see repeated decoded frames, image-content lag and static negative controls'),
      ('occlusion_or_correspondence','UNOBSERVABLE',None,None,'dense residual sign alone is not physical visibility/correspondence certification'),
      ('unresolved','UNOBSERVABLE',native.get('unresolved_holes'),native.get('color_valid_pixels'),'absolute bias and unknown missing depths are not identifiable without independent observations')]
    for mechanism,grade,n,d,note in evidence:rows.append(record(scene,frame,camera,mechanism=mechanism,evidence_grade=grade,numerator=n,denominator=d,evidence=note,scope='this_observation; non-exclusive causes'))
    return rows


def leave_one_out(rows_by_pair,images,cameras,K,settings):
    output=[];lookup={}
    for (i,j),rows in rows_by_pair.items():
        lookup[i,j]={r['keypoint_i']:r['keypoint_j'] for r in rows if r['geometry_accepted']}
        lookup[j,i]={r['keypoint_j']:r['keypoint_i'] for r in rows if r['geometry_accepted']}
    for (i,j),rows in rows_by_pair.items():
        for k in range(len(cameras)):
            if k in (i,j) or not images[k]['usable']:continue
            inv=np.linalg.inv(np.asarray(cameras[k]['pose_matrix_4x4']))
            for r in rows:
                if not r['geometry_accepted']:continue
                ik=lookup.get((i,k),{}).get(r['keypoint_i']);jk=lookup.get((j,k),{}).get(r['keypoint_j'])
                if ik is None or jk is None or ik!=jk:continue
                X=np.asarray(r['world_xyz_m']);local=transform_points(X[None],inv)[0]
                if local[2]<=0:continue
                point=images[k]['features']['keypoints'][ik]
                error=float(np.linalg.norm(project(local[None],K)[0]-point))
                if error>settings.feature_sigma_px*settings.pixel_gate_sigma:continue
                depth,safe,_=depth_sample(images[k]['depth'],point[None],settings.depth_edge_m)
                output.append(dict(population='leave_one_camera_out_RGB_prediction',camera=cameras[k]['name'],target_camera=f'{cameras[i]["name"]}+{cameras[j]["name"]}',
                    u_i_px=float(point[0]),v_i_px=float(point[1]),depth_i_m=depth[0],rgb_depth_i_m=float(local[2]),sensor_minus_rgb_i_m=float(depth[0]-local[2]),
                    geometry_accepted=True,edge_safe_i=bool(safe[0]),reprojection_px=error,
                    independent_support='two other RGB cameras + third-camera reciprocal feature cycle; tested depth not used for triangulation',
                    calibration_covariance_status='unknown; absolute metric bias unverified'))
    return output
