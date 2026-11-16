"""Optical +Z geometry, signed projective populations, and held-out plane fits."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from audit_common import distribution


def project(points,K):
    homogeneous=points@np.asarray(K).T
    return homogeneous[:,:2]/homogeneous[:,2:]


def backproject(depth,K):
    y,x=np.nonzero(np.isfinite(depth)&(depth>0));z=depth[y,x]
    rays=np.c_[x,y,np.ones(len(x))]@np.linalg.inv(K).T
    return rays*z[:,None],np.c_[x,y]


def transform_points(points,T):return points@T[:3,:3].T+T[:3,3]


def fundamental(K1,K2,T1,T2):
    T=np.linalg.inv(T2)@T1;t=T[:3,3]
    cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    return np.linalg.inv(K2).T@cross@T[:3,:3]@np.linalg.inv(K1)


def epipolar(p,q,F):
    a=np.c_[p,np.ones(len(p))];b=np.c_[q,np.ones(len(q))]
    fa=a@F.T;fb=b@F;value=np.sum(b*fa,axis=1)
    denominator=fa[:,:2]**2+fb[:,:2]**2
    return value/np.sqrt(np.maximum(denominator.sum(axis=1),1e-30))


def triangulate(p,q,K1,K2,T1,T2):
    W1=np.linalg.inv(T1);W2=np.linalg.inv(T2)
    P1=K1@W1[:3];P2=K2@W2[:3]
    if not len(p):return np.empty((0,3)),np.empty(0),np.empty(0),np.empty((0,2))
    h=cv2.triangulatePoints(P1,P2,np.asarray(p,float).T,np.asarray(q,float).T).T
    X=h[:,:3]/np.where(np.abs(h[:,3:])>1e-20,h[:,3:],np.nan)
    a=transform_points(X,W1);b=transform_points(X,W2)
    ray1=X-T1[:3,3];ray2=X-T2[:3,3]
    cosine=np.sum(ray1*ray2,axis=1)/np.maximum(np.linalg.norm(ray1,axis=1)*np.linalg.norm(ray2,axis=1),1e-30)
    angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    errors=np.c_[np.linalg.norm(project(a,K1)-p,axis=1),np.linalg.norm(project(b,K2)-q,axis=1)]
    return X,angle,np.maximum(errors[:,0],errors[:,1]),np.c_[a[:,2],b[:,2]]


def triangulation_sigma(p,q,K1,K2,T1,T2,sigma_px):
    n=len(p);jac=np.zeros((n,3,4));step=.01
    for k in range(4):
        dp=np.zeros_like(p);dq=np.zeros_like(q)
        (dp if k<2 else dq)[:,k%2]=step
        plus=triangulate(p+dp,q+dq,K1,K2,T1,T2)[0]
        minus=triangulate(p-dp,q-dq,K1,K2,T1,T2)[0]
        jac[:,:,k]=(plus-minus)/(2*step)
    covariance=np.einsum('nik,njk->nij',jac,jac)*sigma_px**2
    W1=np.linalg.inv(T1);W2=np.linalg.inv(T2)
    zsig=[]
    for W in [W1,W2]:zsig.append(np.sqrt(np.maximum(0,np.einsum('i,nij,j->n',W[2,:3],covariance,W[2,:3]))))
    return covariance,np.column_stack(zsig)


def depth_sample(depth,points,edge_m=.005):
    h,w=depth.shape;xy=np.rint(points).astype(int)
    inside=np.isfinite(points).all(axis=1)&(xy[:,0]>=1)&(xy[:,0]<w-1)&(xy[:,1]>=1)&(xy[:,1]<h-1)
    values=np.full(len(points),np.nan);safe=np.zeros(len(points),bool);spread=np.full(len(points),np.nan)
    for i in np.flatnonzero(inside):
        x,y=xy[i];patch=depth[y-1:y+2,x-1:x+2];valid=np.isfinite(patch)&(patch>0)
        if np.isfinite(depth[y,x]) and depth[y,x]>0:values[i]=depth[y,x]
        if valid.all():spread[i]=float(np.ptp(patch));safe[i]=spread[i]<=edge_m
    return values,safe,spread


def dense_pair(a,b,K,Ta,Tb):
    points,source_xy=backproject(a,K);T=np.linalg.inv(Tb)@Ta
    local=transform_points(points,T);positive=np.isfinite(local).all(axis=1)&(local[:,2]>0)
    uv=np.full((len(points),2),np.nan);uv[positive]=project(local[positive],K)
    pix=np.rint(np.nan_to_num(uv,nan=-1)).astype(int);h,w=b.shape
    inside=positive&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    ids=np.flatnonzero(inside);target=pix[ids,1]*w+pix[ids,0]
    zbuffer=np.full(h*w,np.inf);np.minimum.at(zbuffer,target,local[ids,2])
    visible=local[ids,2]==zbuffer[target]
    # Keep one nearest source per target pixel, with deterministic exact ties.
    nearest=ids[visible];targets=pix[nearest,1]*w+pix[nearest,0]
    _,first=np.unique(targets,return_index=True);nearest=nearest[first]
    y,x=pix[nearest,1],pix[nearest,0];observed=b[y,x];valid=np.isfinite(observed)&(observed>0)
    all_values=local[nearest[valid],2]-observed[valid]
    stats=distribution(all_values)
    stats.update(source_valid_count=len(points),behind_target_count=int((~positive).sum()),
        outside_target_count=int((positive&~inside).sum()),projected_in_bounds_count=int(inside.sum()),
        source_self_occluded_or_collided_count=int(inside.sum()-len(nearest)),target_occupied_count=len(nearest),
        target_invalid_count=int((~valid).sum()),support_count=len(all_values),
        population='unfiltered_projective_candidates_not_certified_physical_correspondences',
        sign_convention='projected_source_Z_in_target - measured_target_Z; positive is behind target observation')
    raster=np.full(b.shape,np.nan,np.float32);raster[y[valid],x[valid]]=all_values
    return stats,raster


def fit_plane(points):
    if len(points)<6:return None
    center=np.median(points,axis=0);weights=np.ones(len(points))
    for _ in range(4):
        center=np.average(points,axis=0,weights=weights)
        _,_,v=np.linalg.svd((points-center)*np.sqrt(weights[:,None]),full_matrices=False)
        n=v[-1];r=(points-center)@n;scale=max(1e-8,1.4826*np.median(np.abs(r-np.median(r))))
        weights=np.minimum(1,1.5*scale/np.maximum(np.abs(r),1e-20))
    if n[2]>0:n=-n
    return n,-float(n@center)


def patch_planes(depth,rays,settings,stationary=None,region=None):
    h,w=depth.shape;xyz=rays.reshape(h,w,3)*depth[:,:,None];rows=[]
    for y in range(0,h,settings.patch_pixels):
        for x in range(0,w,settings.patch_pixels):
            block=depth[y:y+settings.patch_pixels,x:x+settings.patch_pixels];valid=np.isfinite(block)&(block>0)
            yy,xx=np.indices(block.shape);train=valid&((xx+yy)%2==0);held=valid&~train
            if min(train.sum(),held.sum())<settings.min_plane_points:continue
            points=xyz[y:y+settings.patch_pixels,x:x+settings.patch_pixels];model=fit_plane(points[train])
            if model is None:continue
            n,d=model;residual=points[held]@n+d
            # Test systematic curvature on pixels never used for the plane/curvature fit.
            uv=np.c_[xx.ravel()/max(1,block.shape[1])-0.5,yy.ravel()/max(1,block.shape[0])-0.5]
            basis=np.c_[np.ones(len(uv)),uv,uv[:,0]**2,uv[:,0]*uv[:,1],uv[:,1]**2]
            fit_residual=points[train]@n+d
            coefficient=np.linalg.lstsq(basis[train.ravel()],fit_residual,rcond=None)[0]
            curvature=basis[held.ravel()]@coefficient
            stats=distribution(residual);stat_fraction=float(stationary[y:y+block.shape[0],x:x+block.shape[1]][held].mean()) if stationary is not None else None
            centroid=np.median(points[held],axis=0);incidence=np.degrees(np.arccos(np.clip(abs(n@centroid)/max(1e-12,np.linalg.norm(centroid)),0,1)))
            rows.append(dict(patch_x_px=x,patch_y_px=y,patch_width_px=block.shape[1],patch_height_px=block.shape[0],
                support_count=int(held.sum()),plane_normal=n.tolist(),plane_offset_m=d,centroid_m=centroid.tolist(),
                stationary_rgb_evidence_fraction=stat_fraction,physical_patch_identity='conditional_stationary_native_rays' if stat_fraction is not None and stat_fraction>.8 else 'uncertified_local_surface_candidate',
                incidence_deg=float(incidence),range_m=float(np.median(block[held])),
                curvature_heldout_mae_m=float(np.mean(np.abs(residual-curvature))),curvature_baseline_heldout_mae_m=float(np.mean(np.abs(residual))),
                curvature_coefficients_m=coefficient.tolist(),plane_statistics=stats,
                signed_median_m=stats['signed_median_m'],mae_m=stats['mae_m'],
                interpretation='held-out local shape self-consistency, not absolute depth accuracy'))
    return rows
