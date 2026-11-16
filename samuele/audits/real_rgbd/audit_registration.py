"""Exact production replay and independent source-to-target occupancy accounting."""
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np


def load_converter(path):
    path=Path(path).resolve();sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('audit_original_real_converter',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def camera_context(camera,native,K,converter):
    width,height=int(camera['width']),int(camera['height'])
    matrix,distortion=converter.intrinsics(native['color'])
    maps=cv2.initUndistortRectifyMap(matrix,distortion,None,K,(width,height),cv2.CV_32FC1)
    valid=(maps[0]>=0)&(maps[0]<native['color']['width']-1)&(maps[1]>=0)&(maps[1]<native['color']['height']-1)
    rays,ray_valid=converter.depth_rays(native['depth'])
    return dict(rays=rays,ray_valid=ray_valid,color_maps=maps,color_valid=valid,
                T_color_depth=converter.transform(native['T_color_depth'],'T_color_depth'),size=(width,height),K=K)


def trace_registration(raw,context,converter):
    rays=context['rays'];ray_ok=context['ray_valid'];T=context['T_color_depth'];K=context['K'];width,height=context['size']
    if raw.dtype!=np.uint16 or raw.size!=len(rays):raise ValueError('Native depth dtype/shape differs from recorded calibration')
    z=raw.ravel().astype(float)*.001
    measured=(z>0)&ray_ok
    source=np.flatnonzero(measured)
    local=rays[source]*z[source,None];color=local@T[:3,:3].T+T[:3,3]
    finite=np.isfinite(color).all(axis=1);positive=finite&(color[:,2]>0)&(color[:,2]<=65.535)
    pixels=np.full((len(color),2),-1,dtype=np.int64);projected=color[positive]@K.T
    pixels[positive]=np.rint(projected[:,:2]/projected[:,2:]).astype(np.int64)
    inside=positive&(pixels[:,0]>=0)&(pixels[:,0]<width)&(pixels[:,1]>=0)&(pixels[:,1]<height)
    lin=pixels[inside,1]*width+pixels[inside,0];samples=color[inside,2].astype(np.float32);ids=source[inside]
    density=np.bincount(lin,minlength=width*height).reshape(height,width)
    buffer=np.full(width*height,np.inf,dtype=np.float32);np.minimum.at(buffer,lin,samples)
    winner=np.full(width*height,raw.size,dtype=np.int64);wins=samples==buffer[lin]
    np.minimum.at(winner,lin[wins],ids[wins]);winner[winner==raw.size]=-1
    oracle=buffer.reshape(height,width).copy();oracle[~np.isfinite(oracle)]=0;oracle[~context['color_valid']]=0
    replay=converter.register_depth(raw,rays,ray_ok,T,K,(width,height));replay[~context['color_valid']]=0
    if not np.array_equal(oracle,replay):raise AssertionError('Independent provenance rasterization disagrees with production converter')
    winner=winner.reshape(height,width);winner[~context['color_valid']]=-1
    inside_rgb=inside.copy();inside_rgb[inside]=context['color_valid'][pixels[inside,1],pixels[inside,0]]
    # Four-sided occupancy brackets are evidence for a sampling-gap candidate,
    # not proof that an unmeasured native depth would land at that target.
    occupied=(density>0)&context['color_valid'];holes=(~occupied)&context['color_valid']
    left=np.zeros_like(occupied);right=left.copy();up=left.copy();down=left.copy()
    for step in [1,2]:
        left[:,step:]|=occupied[:,:-step];right[:,:-step]|=occupied[:,step:]
        up[step:]|=occupied[:-step];down[:-step]|=occupied[step:]
    bracket=holes&((left&right)|(up&down))
    stats=dict(native_pixels=raw.size,native_zero_count=int((z==0).sum()),native_valid_count=int((z>0).sum()),
        native_invalid_fraction=float(np.mean(z==0)),failed_inverse_distortion_count=int(((z>0)&~ray_ok).sum()),
        transformed_nonfinite_count=int((~finite).sum()),behind_or_beyond_color_range_count=int((finite&~positive).sum()),
        outside_color_frame_count=int((positive&~inside).sum()),outside_color_valid_count=int((inside&~inside_rgb).sum()),
        projected_samples=int(inside.sum()),projected_in_rgb_valid_samples=int(inside_rgb.sum()),
        occupied_target_pixels=int(occupied.sum()),collision_samples=int(np.maximum(density-1,0).sum()),
        collision_rate=float(np.maximum(density-1,0).sum()/max(1,len(lin))),
        registered_zero_inside_rgb_count=int(holes.sum()),sampling_gap_bracket_candidates=int(bracket.sum()),
        unresolved_holes=int((holes&~bracket).sum()),color_valid_pixels=int(context['color_valid'].sum()),
        registered_valid_fraction=float(np.mean(replay>0)))
    full_uv=np.full((raw.size,2),np.nan);pp=color[positive]@K.T;full_uv[source[positive]]=pp[:,:2]/pp[:,2:]
    return replay,stats,dict(density=density,winner_native_index=winner,hole_candidates=bracket,
        color_valid=context['color_valid'],source_uv=full_uv.reshape(*raw.shape,2),raw_valid=raw>0,native_depth_m=raw.astype(np.float32)*.001)


def raster_counterfactual(raw,context,maps,rgb,settings):
    """Scatter the same measured samples; no filled values enter baseline errors."""
    z=raw.astype(float)*.001;valid=raw>0
    low=cv2.erode(np.where(valid,z,np.inf),np.ones((3,3),np.uint8))
    high=cv2.dilate(np.where(valid,z,-np.inf),np.ones((3,3),np.uint8))
    all_valid=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8))>0
    span=np.subtract(high,low,out=np.zeros_like(high),where=all_valid)
    uv=maps['source_uv'];finite=np.isfinite(uv).all(axis=2)
    pixel=np.rint(np.nan_to_num(uv,nan=-1)).astype(int)
    h,w=rgb.shape[:2];inside=finite&(pixel[:,:,0]>=0)&(pixel[:,:,0]<w)&(pixel[:,:,1]>=0)&(pixel[:,:,1]<h)
    gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY);gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    rgb_edge=np.zeros(raw.shape,bool);rgb_edge[inside]=gradient[pixel[inside,1],pixel[inside,0]]>4*settings.min_texture_dn
    baseline=(maps['density']>0)&context['color_valid'];rows=[];strict_fill=np.zeros((h,w),bool)
    for bound_mm in [2,5,10]:
        continuous=inside&all_valid&(span<=bound_mm*.001)&~rgb_edge
        xy=uv[continuous];density=np.zeros((h,w),bool)
        for dx,dy in [(0,0),(0,1),(1,0),(1,1)]:
            target=np.floor(xy).astype(int)+[dx,dy];ok=(target[:,0]>=0)&(target[:,0]<w)&(target[:,1]>=0)&(target[:,1]<h)
            density[target[ok,1],target[ok,0]]=True
        fill=density&~baseline&context['color_valid']
        # Veto image discontinuities rather than pretending they certify geometry.
        fill&=gradient<4*settings.min_texture_dn
        if bound_mm==2:strict_fill=fill
        rows.append(dict(arm='A1_measured_point_footprint',continuity_bound_mm=bound_mm,
            measured_source_support=int(continuous.sum()),filled_target_candidates=int(fill.sum()),
            bracketed_fill_candidates=int((fill&maps['hole_candidates']).sum()),
            fill_fraction_of_baseline_holes=float(fill.sum()/max(1,((~baseline)&context['color_valid']).sum())),
            correctness_status='not_independently_observable; counterfactual candidates, never sensor truth'))
    return rows,strict_fill
