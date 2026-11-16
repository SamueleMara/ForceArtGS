"""E1: exact replay, exhaustive availability, conservative held-out splats."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from audit_registration import trace_registration


def neighborhoods(depth):
    valid=np.isfinite(depth)&(depth>0);kernel=np.ones((3,3),np.uint8)
    lo=cv2.erode(np.where(valid,depth,np.inf),kernel)
    hi=cv2.dilate(np.where(valid,depth,-np.inf),kernel)
    span=np.subtract(hi,lo,out=np.zeros_like(hi),where=(hi>0)&np.isfinite(lo))
    return valid,span


def footprint(raw,context,maps,keep=None,bound_m=.005):
    """Only measured locally continuous native samples; diagnostic, never data."""
    z=raw.astype(float)*.001
    training=(raw>0)&keep if keep is not None else raw>0
    valid,span=neighborhoods(np.where(training,z,0))
    if keep is None:complete=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8))>0
    else:complete=cv2.filter2D(valid.astype(np.float32),-1,np.ones((3,3),np.float32))>=5
    uv=maps['source_uv'];ok=valid&complete&(span<=bound_m)&np.isfinite(uv).all(axis=2)
    T=context['T_color_depth'];rays=context['rays'].reshape(*raw.shape,3)
    zc=(rays*z[:,:,None])@T[2,:3]+T[2,3]
    xy=np.floor(np.where(np.isfinite(uv),uv,-1)).astype(int)
    h,w=context['color_valid'].shape;buffer=np.full(h*w,np.inf,np.float32)
    spacing=np.linalg.norm(np.diff(uv,axis=1),axis=2);spacing=spacing[np.isfinite(spacing)&(spacing>0)&training[:,1:]&training[:,:-1]]
    radius=min(3,max(1,int(np.ceil(np.median(spacing))) if len(spacing) else 1))
    for dx,dy in [(x,y) for x in range(-radius,radius+1) for y in range(-radius,radius+1)]:
        p=xy+[dx,dy];inside=ok&(p[:,:,0]>=0)&(p[:,:,0]<w)&(p[:,:,1]>=0)&(p[:,:,1]<h)&(zc>0)
        np.minimum.at(buffer,p[inside,1]*w+p[inside,0],zc[inside].astype(np.float32))
    out=buffer.reshape(h,w);out[~np.isfinite(out)]=0;out[~context['color_valid']]=0
    return out


def heldout_rasterization(raw,context,maps):
    """The withheld measured depths are not used to generate the splat values."""
    yy,xx=np.indices(raw.shape);held=(xx+3*yy)%5==0
    # Selection uses spatial/native layer continuity; no cross-view residual gate.
    predicted=footprint(raw,context,maps,keep=~held)
    uv=maps['source_uv'];xy=np.rint(np.where(np.isfinite(uv),uv,-1)).astype(int);h,w=predicted.shape
    rays=context['rays'].reshape(*raw.shape,3);T=context['T_color_depth']
    truth=(rays*(raw.astype(float)*.001)[:,:,None])@T[2,:3]+T[2,3]
    use=held&(raw>0)&np.isfinite(uv).all(axis=2)&(xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<h)
    values=predicted[xy[use,1],xy[use,0]];reference=truth[use];supported=values>0
    residual=values[supported]-reference[supported]
    _,span=neighborhoods(raw.astype(float)*.001);edge=span[use][supported]>.005
    from .common import population
    return dict(status='CONDITIONAL' if len(residual) else 'UNOBSERVABLE',heldout_native_samples=int(use.sum()),supported_heldout_samples=len(residual),
        false_fill_abs_gt_5mm_count=int((np.abs(residual)>.005).sum()),edge_heldout_samples=int(edge.sum()),
        false_fill_edge_abs_gt_5mm_count=int((edge&(np.abs(residual)>.005)).sum()),
        residual=population(residual),reference='withheld measured native samples; not independent absolute GT; 20% spatial holdout')


def trace(raw,stored,context,converter,rgb,mask,save_detail=False):
    replay,counts,maps=trace_registration(raw,context,converter)
    bitwise=int((replay.view(np.uint32)!=stored.view(np.uint32)).sum())
    occupied=replay>0;inside=context['color_valid'];holes=inside&~occupied
    projected_distance=distance_transform_edt(~occupied)
    conservative=footprint(raw,context,maps,bound_m=.002)
    _,span=neighborhoods(replay)
    bracket=maps['hole_candidates'];gap=holes&bracket&(conservative>0)&(span<=.002)
    discontinuity=holes&~gap&(projected_distance<=2)&(span>.005)
    categories=np.full(stored.shape,4,np.uint8);categories[~inside]=0;categories[occupied&inside]=1;categories[gap]=2;categories[discontinuity]=3
    gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY);edge=cv2.Canny(gray,50,150)>0;edge_distance=distance_transform_edt(~edge)
    regions={'all':np.ones(stored.shape,bool),'valid_rgb':inside,'rgb_edge_le_1px':edge_distance<=1,'rgb_edge_1_4px':(edge_distance>1)&(edge_distance<=4),'rgb_edge_gt_4px':edge_distance>4}
    if mask is not None:
        d=distance_transform_edt(mask);regions['predicted_object']=mask;regions['outside_mask_unknown']=~mask
        for low,high in [(0,1),(1,2),(2,4),(4,8),(8,16),(16,32)]:regions[f'mask_inside_{low}_{high}px']=mask&(d>=low)&(d<high)
        regions['mask_interior_ge32px']=mask&(d>=32)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV);regions['bright_low_saturation_candidate_not_material_GT']=(hsv[:,:,2]>230)&(hsv[:,:,1]<20)
    regional={name:dict(pixels=int(r.sum()),valid_depth=int((r&occupied).sum()),outside_rgb=int((r&~inside).sum()),holes_inside_rgb=int((r&holes).sum()),gap_candidates=int((r&gap).sum()),discontinuity_candidates=int((r&discontinuity).sum()),unknown_holes=int((r&holes&~gap&~discontinuity).sum())) for name,r in regions.items()}
    ray_valid=context['ray_valid'].reshape(raw.shape)
    row=dict(status='MEASURED' if bitwise==0 else 'CONTRADICTED',bitwise_different_pixels=bitwise,**counts,
        native_calibrated_ray_pixels=int(ray_valid.sum()),native_zero_in_calibrated_rays=int(((raw==0)&ray_valid).sum()),
        outside_rgb_support_pixels=int((~inside).sum()),gap_candidate_pixels=int(gap.sum()),discontinuity_candidate_pixels=int(discontinuity.sum()),
        unknown_hole_pixels=int((holes&~gap&~discontinuity).sum()),regions=regional,
        hard_categories=['outside_RGB_remap_support','no_projected_valid_native_ray'],conditional_categories=['sampling_gap_candidate','discontinuity_candidate'],
        unknown_native_zero_target_mapping='UNOBSERVABLE_without_depth_hypothesis')
    if save_detail:
        T=context['T_color_depth'];z=raw.astype(float)*.001;rays=context['rays'].reshape(*raw.shape,3)
        zc=(rays*z[:,:,None])@T[2,:3]+T[2,3]
        maps.update(category=categories,distance_to_projected_samples_px=projected_distance.astype(np.float32),transformed_color_z_m=zc.astype(np.float32),
            diagnostic_splat_z_m=conservative)
        row['heldout_counterfactual']=heldout_rasterization(raw,context,maps)
    return row,maps
