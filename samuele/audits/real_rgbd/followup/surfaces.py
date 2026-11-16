"""Automatic connected depth planes and deliberately conditional association.

RGB correspondences are proposals, not physical IDs. The tested calibration is
used for visibility/overlap but never described as independent identity proof.
"""
from collections import Counter
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .common import population


def fit_plane(points):
    points=np.asarray(points,float)
    if len(points)<12:return None
    weights=np.ones(len(points));normal=None
    for _ in range(3):
        center=np.average(points,axis=0,weights=weights);delta=points-center
        values,vectors=np.linalg.eigh((delta*weights[:,None]).T@delta/max(weights.sum(),1))
        normal=vectors[:,0];residual=delta@normal
        weights=np.minimum(1.,.003/np.maximum(np.abs(residual),1e-12))
    if values[1]<1e-8:return None
    return normal,float(normal@center),center,values


def extract(depth,rgb,mask,K,pose,previous=None,stride=4):
    """Coarse connected segmentation; independent fit uses all valid fine pixels.

    Normal estimates use local valid-depth averages. Averaged values are never
    substituted for measured depth in fitting or downstream comparisons.
    """
    h,w=depth.shape;valid=np.isfinite(depth)&(depth>0)
    small_size=(max(1,w//stride),max(1,h//stride))
    coverage=cv2.resize(valid.astype(np.float32),small_size,interpolation=cv2.INTER_AREA)
    mean=cv2.resize(np.where(valid,depth,0),small_size,interpolation=cv2.INTER_AREA)/np.maximum(coverage,1e-6)
    yy,xx=np.indices(mean.shape);u=(xx+.5)*w/mean.shape[1]-.5;v=(yy+.5)*h/mean.shape[0]-.5
    rays=np.stack([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones_like(u)],2);points=rays*mean[:,:,None]
    du=np.zeros_like(points);dv=np.zeros_like(points);du[:,1:-1]=points[:,2:]-points[:,:-2];dv[1:-1]=points[2:]-points[:-2]
    normals=np.cross(du,dv);length=np.linalg.norm(normals,axis=2);normals/=np.maximum(length[:,:,None],1e-12)
    # Orient towards the observing camera. Opposite visible sides stay opposite.
    normals[np.sum(normals*points,axis=2)>0]*=-1
    ok=(coverage>.3)&(length>1e-8)
    ok[[0,-1],:]=False;ok[:,[0,-1]]=False
    cls=cv2.resize(mask.astype(np.uint8),small_size,interpolation=cv2.INTER_NEAREST) if mask is not None else np.full(mean.shape,2,np.uint8)
    ids=np.arange(mean.size).reshape(mean.shape);source=[];target=[]
    for a,b in [((slice(None),slice(None,-1)),(slice(None),slice(1,None))),((slice(None,-1),slice(None)),(slice(1,None),slice(None)))]:
        diff=points[b]-points[a]
        compatible=ok[a]&ok[b]&(cls[a]==cls[b])&(np.sum(normals[a]*normals[b],axis=2)>np.cos(np.deg2rad(12)))
        compatible&=(np.abs(np.sum(normals[a]*diff,axis=2))<.008)&(np.abs(np.sum(normals[b]*diff,axis=2))<.008)
        source.extend(ids[a][compatible]);target.extend(ids[b][compatible])
    graph=coo_matrix((np.ones(len(source)),(source,target)),shape=(mean.size,mean.size))
    _,labels=connected_components(graph,directed=False);labels=labels.reshape(mean.shape);labels[~ok]=-1
    fine=cv2.resize((labels+1).astype(np.float32),(w,h),interpolation=cv2.INTER_NEAREST).astype(np.int32)-1
    yyf,xxf=np.indices(depth.shape);all_rays=np.stack([(xxf-K[0,2])/K[0,0],(yyf-K[1,2])/K[1,1],np.ones_like(depth)],2)
    fine_xyz=all_rays*depth[:,:,None];patches=[];output_labels=np.full(depth.shape,-1,np.int32)
    sizes=Counter(labels[ok].ravel().tolist());R=pose[:3,:3];t=pose[:3,3]
    gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    prev_patches=previous['patches'] if previous else []
    for label,size in sizes.items():
        if size<12:continue
        region=(fine==label)&valid
        # Independently split measured pixels before fitting; keep all held-out tails.
        train=region&((xxf+3*yyf)%3!=0);held=region&~train
        if min(train.sum(),held.sum())<24:continue
        fit=fit_plane(fine_xyz[train])
        if fit is None:continue
        normal,offset,center,eigenvalues=fit
        if normal@center>0:normal=-normal;offset=-offset
        residual=fine_xyz[held]@normal-offset
        planar=np.percentile(np.abs(residual),95)<=.01
        # Patches remain explicit NONPLANAR candidates rather than being forced flat.
        pid=len(patches);output_labels[region]=pid
        world_normal=R@normal;world_center=R@center;world_center+=t;world_offset=offset+world_normal@t
        y,x=np.nonzero(region);bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
        identity=None;best=0
        if previous:
            old_ids=previous['labels'][region];counts=Counter(old_ids[old_ids>=0].tolist())
            for old_pid,intersection in counts.items():
                old=prev_patches[old_pid];union=int(region.sum())+old['pixels']-intersection;iou=intersection/max(1,union)
                if iou>best and iou>.4 and np.dot(normal,old['normal_camera'])>.98:
                    best=iou;identity=old['track_id']
        track_id=identity if identity is not None else f'{previous["frame"]+1 if previous else 0}:{pid}'
        # Spatial tile medians quantify conditional fit variation, not pixel-IID CI.
        tiles=(yf:=y//32)*(w//32+1)+x//32;tile_values=[]
        all_res=fine_xyz[region]@normal-offset
        order=np.argsort(tiles);splits=np.flatnonzero(np.diff(tiles[order]))+1
        tile_values=[float(np.median(v)) for v in np.split(all_res[order],splits)]
        fit_sigma=float(1.4826*np.median(np.abs(np.asarray(tile_values)-np.median(tile_values)))) if len(tile_values)>1 else None
        rgb_mean=float(gray[region].mean())
        stable=False
        if identity is not None:
            old=next(p for p in prev_patches if p['track_id']==identity)
            stable=abs(old['offset_camera_m']-offset)<.003 and abs(old['gray_mean_dn']-rgb_mean)<3 and np.dot(normal,old['normal_camera'])>.995
        patches.append(dict(patch_id=pid,track_id=track_id,status='CONDITIONAL_PLANE' if planar else 'NONPLANAR_UNRESOLVED',
            normal_camera=normal.tolist(),offset_camera_m=offset,centroid_camera_m=center.tolist(),normal_world=world_normal.tolist(),offset_world_m=float(world_offset),
            centroid_world_m=world_center.tolist(),pixels=int(region.sum()),fitting_pixels=int(train.sum()),heldout_pixels=int(held.sum()),
            heldout=population(residual),inlier_pixels=int((np.abs(all_res)<=.009).sum()),bbox_xyxy=bbox,
            depth_min_m=float(depth[region].min()),depth_max_m=float(depth[region].max()),
            spatial_extent_camera_m=np.ptp(fine_xyz[region],axis=0).tolist(),centroid_range_m=float(np.linalg.norm(center)),
            incidence_deg=float(np.rad2deg(np.arccos(np.clip(-normal@center/max(np.linalg.norm(center),1e-12),-1,1)))),
            region='predicted_object' if mask is not None and np.mean(mask[region])>.5 else 'outside_mask_unknown' if mask is not None else 'unknown',
            mask_truth='UNVERIFIED',gray_mean_dn=rgb_mean,spatial_tiles=len(tile_values),fit_tile_sigma_m=fit_sigma,
            conditional_fit_offset_sd_m=fit_sigma/np.sqrt(len(tile_values)) if fit_sigma is not None else None,
            conditional_fit_normal_sd_deg=float(np.rad2deg(fit_sigma/np.sqrt(max(eigenvalues[1],1e-12)*len(tile_values)))) if fit_sigma is not None else None,
            uncertainty='spatial-tile median dispersion; conditional planar fit, not absolute accuracy or calibrated extrinsic covariance',
            consecutive_overlap=best,temporal_persistent=bool(stable),static_evidence='CONDITIONAL_FIXED_CAMERA_DEPTH_AND_RGB' if stable else 'UNVERIFIED',
            surface_identity='AUTOMATIC_PROPOSAL_NOT_PHYSICAL_GT'))
    return dict(patches=patches,labels=output_labels,frame=(previous['frame']+1 if previous else 0),rays=all_rays)


def association_decision(*,oriented_normal_dot,rgb_train,rgb_heldout,heldout_reprojection_px,texture_ambiguity,overlap,temporal_support,layer_conflict=False):
    """Hard negatives abstain; fixed-pose overlap never independently proves ID."""
    if oriented_normal_dot<-.5:return 'REJECTED_OPPOSITE_VISIBLE_FACES'
    if texture_ambiguity:return 'AMBIGUOUS_REPEATED_TEXTURE'
    if layer_conflict:return 'UNIDENTIFIABLE_LAYER_OR_CALIBRATION'
    if oriented_normal_dot<np.cos(np.deg2rad(20)):return 'REJECTED_INCOMPATIBLE_NORMALS_UNDER_FIXED_POSE'
    if overlap<=0:return 'UNOBSERVABLE_NO_PROJECTED_OVERLAP'
    if rgb_train<4 or rgb_heldout<4:return 'CONDITIONAL_GEOMETRY_ONLY_IDENTITY_UNVERIFIED'
    if heldout_reprojection_px is None or heldout_reprojection_px>3:return 'REJECTED_RGB_HELDOUT_IDENTITY'
    if not temporal_support:return 'CONDITIONAL_SINGLE_FRAME_IDENTITY_UNVERIFIED'
    return 'AUTO_MULTICUE_CONSISTENT_CONDITIONAL'


def associate(a,b,depth_a,depth_b,K,Ta,Tb,matches):
    """Preserve competing patch hypotheses and their unfiltered signed residuals."""
    labels_a=a['labels'];labels_b=b['labels'];h,w=labels_a.shape
    pair_features={};competition_a=Counter();competition_b=Counter()
    for row in matches:
        ua,va=int(round(row['u_i_px'])),int(round(row['v_i_px']));ub,vb=int(round(row['u_j_px'])),int(round(row['v_j_px']))
        if not (0<=ua<w and 0<=ub<w and 0<=va<h and 0<=vb<h):continue
        ia,ib=int(labels_a[va,ua]),int(labels_b[vb,ub])
        if min(ia,ib)<0:continue
        pair_features.setdefault((ia,ib),[]).append(row)
    for (ia,ib),rows in pair_features.items():competition_a[ia]+=len(rows);competition_b[ib]+=len(rows)
    result=[];inv_b=np.linalg.inv(Tb);rays=a['rays'];valid_a=(depth_a>0)&(labels_a>=0)
    # Project all measured source points once. No residual-size selection.
    X=rays[valid_a]*depth_a[valid_a,None];world=X@Ta[:3,:3].T+Ta[:3,3];local=world@inv_b[:3,:3].T+inv_b[:3,3]
    uv=local@K.T;xy=np.rint(uv[:,:2]/np.maximum(uv[:,2:],1e-9)).astype(int)
    inside=(local[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    aid=labels_a[valid_a];bid=np.full(len(X),-1);bid[inside]=labels_b[xy[inside,1],xy[inside,0]]
    yy,xx=np.indices(depth_a.shape);heldout_source=((xx+3*yy)%3==0)[valid_a]
    overlap_counts=Counter(zip(aid[inside].tolist(),bid[inside].tolist()))
    pairs=set(pair_features)|{(i,j) for (i,j),n in overlap_counts.items() if j>=0 and n>=12}
    for ia,ib in sorted(pairs):
        p,q=a['patches'][ia],b['patches'][ib]
        dot=float(np.dot(p['normal_world'],q['normal_world']));rows=pair_features.get((ia,ib),[])
        source=inside&(aid==ia)&(bid==ib);overlap=int(source.sum())/max(1,min(p['pixels'],q['pixels']))
        source_depth=depth_b[xy[source,1],xy[source,0]];projected=local[source,2];depth_delta=projected-source_depth
        hypotheses_a=[len(v) for (i,j),v in pair_features.items() if i==ia and j!=ib]
        hypotheses_b=[len(v) for (i,j),v in pair_features.items() if j==ib and i!=ia]
        ambiguous=bool(rows and max(hypotheses_a+hypotheses_b+[0])>=.8*len(rows))
        train=rows[::2];held=rows[1::2];error=None
        if len(train)>=4 and len(held)>=4:
            first=np.array([[r['u_i_px'],r['v_i_px']] for r in train]);second=np.array([[r['u_j_px'],r['v_j_px']] for r in train])
            H,inliers=cv2.findHomography(first,second,cv2.RANSAC,3.)
            if H is not None:
                hp=np.c_[[[r['u_i_px'],r['v_i_px']] for r in held],np.ones(len(held))]@H.T
                predicted=hp[:,:2]/hp[:,2:];target=np.array([[r['u_j_px'],r['v_j_px']] for r in held]);error=float(np.median(np.linalg.norm(predicted-target,axis=1)))
        # Large layer conflict is retained as unidentifiable, not interpreted as
        # a bias correction and not silently removed from candidate statistics.
        layer_conflict=bool(len(depth_delta) and abs(np.median(depth_delta))>.05)
        decision=association_decision(oriented_normal_dot=dot,rgb_train=len(train),rgb_heldout=len(held),heldout_reprojection_px=error,
            texture_ambiguity=ambiguous,overlap=overlap,temporal_support=p['consecutive_overlap']>.4 and q['consecutive_overlap']>.4,layer_conflict=layer_conflict)
        if p['status']!='CONDITIONAL_PLANE' or q['status']!='CONDITIONAL_PLANE':decision='NONPLANAR_UNRESOLVED'
        n=np.array(q['normal_world']);signed_offset=p['offset_world_m']-q['offset_world_m'] if dot>0 else None
        # Fit each view independently. Other-plane residuals are conditional
        # candidates until independent physical identity is actually available.
        point_residual=world[source&heldout_source]@n-q['offset_world_m']
        offset_sd=[p['conditional_fit_offset_sd_m'],q['conditional_fit_offset_sd_m']]
        normal_sd=[p['conditional_fit_normal_sd_deg'],q['conditional_fit_normal_sd_deg']]
        result.append(dict(source_patch=ia,target_patch=ib,source_track=p['track_id'],target_track=q['track_id'],status=decision,
            physical_identity='AUTOMATIC_CONDITIONAL_NOT_INDEPENDENT_GT',calibration_used_for_overlap=True,
            normal_angle_deg=float(np.rad2deg(np.arccos(np.clip(dot,-1,1)))),oriented_normal_dot=dot,
            signed_world_plane_offset_m=signed_offset,source_to_target_plane=population(point_residual),projected_depth_residual=population(depth_delta),
            projected_overlap_fraction=overlap,projected_overlap_pixels=int(source.sum()),rgb_candidates=len(rows),rgb_heldout_reprojection_px=error,
            texture_ambiguous=ambiguous,source_region=p['region'],target_region=q['region'],
            plane_fit_uncertainty_m=[p['fit_tile_sigma_m'],q['fit_tile_sigma_m']],
            propagated_conditional_offset_sd_m=float(np.linalg.norm(offset_sd)) if all(x is not None for x in offset_sd) else None,
            propagated_conditional_normal_sd_deg=float(np.linalg.norm(normal_sd)) if all(x is not None for x in normal_sd) else None,
            uncertainty_assumption='independent spatial-tile fit variation in each camera; excludes calibration covariance and identity uncertainty',
            cross_view_residual_population='source held-out modulo-three pixels; target plane fitted independently',
            source_normal_world=p['normal_world'],target_normal_world=q['normal_world'],
            source_centroid_world_m=p['centroid_world_m'],target_centroid_world_m=q['centroid_world_m'],
            source_static_conditional=p['temporal_persistent'],target_static_conditional=q['temporal_persistent'],
            source_offset_world_m=p['offset_world_m'],target_offset_world_m=q['offset_world_m'],
            visibility_conflict_pixels=int((depth_delta>.02).sum()),
            evidence='oriented measured normals, fixed-pose projected layer, disjoint RGB homography, temporal persistence; calibration-conditioned, not an independent pose validation'))
        if decision=='AUTO_MULTICUE_CONSISTENT_CONDITIONAL':
            bins=np.r_[-np.inf,np.linspace(-.5,.5,1001),np.inf]
            result[-1]['qualified_residual_histogram_counts']=np.histogram(point_residual,bins=bins)[0].tolist()
            result[-1]['qualified_histogram_definition']='1-mm signed bins [-0.5,0.5] m plus two overflow bins; quantiles approximate to 1 mm'
    return result
