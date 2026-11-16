"""Exact native temporal maps and content-based lag evidence, never hardware claims."""
from pathlib import Path
import hashlib
import warnings

import cv2
import numpy as np

from audit_common import file_record,distribution,table,sha,block_interval,block_length


def array_hash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class NativeSeries:
    def __init__(self,folder,camera,frames,shape,resume=False):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.frames=frames;self.shape=shape
        self.path=self.folder/(camera+'_native_m.dat');self.support_path=self.folder/(camera+'_rgb_stationary.dat')
        shape=(len(frames),*shape);expected=int(np.prod(shape))
        if resume and self.path.is_file():
            if self.path.stat().st_size!=expected*4 or self.support_path.stat().st_size!=expected:raise ValueError('Native temporal cache size differs on resume')
            mode='r+'
        else:mode='w+'
        self.depth=np.memmap(self.path,mode=mode,dtype=np.float32,shape=shape)
        self.stationary=np.memmap(self.support_path,mode=mode,dtype=np.uint8,shape=shape)
        # New mmap files contain zero; missing/unwritten observations are converted
        # to NaN per tile below, without faulting the entire episode into RAM.
    def put(self,index,raw,stationary):
        self.depth[index]=np.where(raw>0,raw.astype(np.float32)*.001,np.nan);self.stationary[index]=stationary.astype(np.uint8)
    def put_meters(self,index,depth,stationary):
        self.depth[index]=np.where(np.isfinite(depth)&(depth>0),depth,np.nan);self.stationary[index]=stationary.astype(np.uint8)
    def release_scratch(self):
        for value,path in [(self.depth,self.path),(self.stationary,self.support_path)]:
            if not path.resolve().is_relative_to(self.folder.resolve()):raise ValueError('Audit scratch escapes its new output folder')
            value.flush();value._mmap.close();path.unlink()
    def flush(self):self.depth.flush();self.stationary.flush()
    def chunk_hash(self,indices):
        return dict(depth_sha256=array_hash(self.depth[indices]),stationarity_sha256=array_hash(self.stationary[indices]))
    def finalize(self,seed=1731,draws=200):
        n,h,w=self.depth.shape
        median=np.full((h,w),np.nan,np.float32);sigma=median.copy();drift=median.copy();transitions=np.zeros((h,w),np.uint32);count=np.zeros((h,w),np.uint32);support=np.zeros((h,w),np.float32)
        times=np.array(self.frames,float);times-=times.mean();intervals=[];delta_count=0
        # Every native pixel and every time is included, tiled only to bound RAM.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            for y in range(0,h,32):
                block=np.where(self.depth[:,y:y+32,:]>0,self.depth[:,y:y+32,:],np.nan);evidence=np.asarray(self.stationary[:,y:y+32,:],bool)
                valid=np.isfinite(block);med=np.nanmedian(block,axis=0)
                median[y:y+32]=med;sigma[y:y+32]=1.4826*np.nanmedian(np.abs(block-med),axis=0)
                count[y:y+32]=valid.sum(axis=0);support[y:y+32]=evidence.sum(axis=0)/max(1,n-1)
                transitions[y:y+32]=np.count_nonzero(valid[1:]!=valid[:-1],axis=0)
                t=np.broadcast_to(times[:,None,None],block.shape);c=valid.sum(axis=0)
                mt=np.sum(np.where(valid,t,0),axis=0)/np.maximum(c,1);mz=np.nansum(block,axis=0)/np.maximum(c,1)
                numerator=np.sum(np.where(valid,(t-mt)*(block-mz),0),axis=0)
                denominator=np.sum(np.where(valid,(t-mt)**2,0),axis=0)
                drift[y:y+32]=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
                conditional=(evidence.sum(axis=0)/max(1,n-1)>=.8)&(valid.sum(axis=0)>=5)&(n>=6)
                deltas=np.abs(block[1:]-block[:-1]);deltas[:,~conditional]=np.nan
                delta_count+=int(np.isfinite(deltas).sum());intervals.append(float(np.nanmedian(deltas)) if np.isfinite(deltas).any() else None)
        certified=(support>=.8)&(count>=5)&(n>=6)
        np.savez_compressed(self.folder/(self.path.stem+'_temporal_maps.npz'),median_depth_m=median,robust_sigma_m=sigma,
            slope_m_per_converted_frame=drift,valid_count=count,valid_invalid_transitions=transitions,
            rgb_stationary_evidence_fraction=support,conditional_stationary_mask=certified)
        result=dict(conditional_stationary_pixels=int(certified.sum()),sensor_geometry_pixels=h*w,frames=n,
            geometry='registered_rectified_color_Z' if '_registered' in self.path.stem else 'native_depth_Z',
            stationary_precision_status='measured_conditionally' if certified.any() else 'not_measurable_no_adequate_RGB_stationary_support',
            temporal_repeatability_statistics=distribution(sigma[certified]),native_sigma_all_rays_statistics=distribution(sigma[np.isfinite(sigma)]),
            drift_statistics_m_per_converted_frame=distribution(drift[certified]),
            median_of_tile_median_abs_consecutive_change_m=float(np.median([v for v in intervals if v is not None])) if any(v is not None for v in intervals) else None,
            consecutive_change_support=delta_count,
            stationary_status='conditional_on_RGB_texture_motion_and_appearance; unknown pixels not treated as static',
            inference_limit='Repeatability/plane self-consistency cannot establish absolute depth accuracy; constant metric bias is unobservable here')
        residual_time=[];residual_support=[]
        for frame in range(n):
            current=np.asarray(self.depth[frame]);values=np.where(current>0,current,np.nan)[certified]-median[certified]
            finite=np.isfinite(values);residual_time.append(float(np.median(values[finite])) if finite.any() else None);residual_support.append(int(finite.sum()))
        block=block_length([v if v is not None else np.nan for v in residual_time])
        result['stationary_frame_median_residual_m']=residual_time;result['stationary_frame_support']=residual_support
        result['block_uncertainty_sensitivity']=[block_interval([v if v is not None else np.nan for v in residual_time],b,seed,draws) for b in sorted({max(1,block//2),block,max(1,block*2)})]
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(13,8))
        for ax,title,image in zip(axes.ravel(),['Temporal median camera Z (m)','Temporal robust sigma (mm)','Slope (mm / converted frame)','Valid sample count','RGB stationary evidence fraction','Valid/invalid transitions'],[median,sigma*1000,drift*1000,count,support,transitions]):
            im=ax.imshow(image);fig.colorbar(im,ax=ax,shrink=.6);ax.set_title(title);ax.set_xlabel('sensor-domain u (px)');ax.set_ylabel('sensor-domain v (px)')
        fig.suptitle(f'{self.path.stem}: N={n}, conditional stationary rays={certified.sum()}; repeatability is not accuracy')
        fig.tight_layout();fig.savefig(self.folder/(self.path.stem+'_temporal_maps.png'),dpi=100);plt.close(fig)
        self.flush();return result


def audit_raw_series(scene,camera,source,out):
    """Every available raw depth and every decoded RGB video frame, not stride-2 only."""
    episode=scene['conversion']['episode'];name=camera['source_camera_name'];root=Path(source);rows=[];inputs=[]
    depth_dir=root/f'depth/chunk-000/observation.depth.{name}/episode_{episode:06d}'
    previous_hash=None;previous_rgb=None
    for path in sorted(depth_dir.glob('frame_*.png')):
        raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);inputs.append(file_record(path))
        index=int(path.stem.removeprefix('frame_'));valid=raw is not None and raw.ndim==2 and raw.dtype==np.uint16
        digest=array_hash(raw) if valid else None
        rows.append(dict(dataset_id=scene['id'],dataset=scene['name'],camera=camera['name'],frame_index=None,source_frame_index=index,
            status='measured' if valid else 'failed',series='raw_depth',path=str(path),decoded_sha256=digest,
            exact_repeat=bool(digest is not None and digest==previous_hash),invalid_fraction=float(np.mean(raw==0)) if valid else None,
            timestamp_quality='index_only; sensor hardware exposure time unavailable',source_file_sha256=inputs[-1].get('sha256')))
        previous_hash=digest
    video=root/f'videos/chunk-000/observation.images.{name}/episode_{episode:06d}.mp4';inputs.append(file_record(video));previous_hash=None
    if video.is_file():
        cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);index=0
        try:
            while True:
                ok,rgb=cap.read()
                if not ok:break
                digest=array_hash(rgb);gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
                rows.append(dict(dataset_id=scene['id'],dataset=scene['name'],camera=camera['name'],frame_index=None,source_frame_index=index,
                    status='measured',series='decoded_raw_rgb',path=str(video),decoded_sha256=digest,exact_repeat=digest==previous_hash,
                    rgb_previous_mae_dn=float(np.mean(np.abs(gray.astype(float)-previous_rgb))) if previous_rgb is not None else None,
                    video_metadata_fps=fps,nominal_video_time_s=index/fps if fps>0 else None,
                    timestamp_quality='video frame counter / declared FPS, not hardware acquisition timestamp'))
                previous_hash=digest;previous_rgb=gray;index+=1
        finally:cap.release()
    table(out,rows)
    rgb_rows=[r for r in rows if r['series']=='decoded_raw_rgb'];depth_rows=[r for r in rows if r['series']=='raw_depth']
    repeats=sum(r.get('exact_repeat',False) for r in rgb_rows);fps=rgb_rows[0].get('video_metadata_fps') if rgb_rows else None
    return dict(rows=len(rows),inputs=inputs,decoded_rgb_frames=len(rgb_rows),native_depth_frames=len(depth_rows),exact_rgb_repeats=repeats,
        exact_depth_repeats=sum(r.get('exact_repeat',False) for r in depth_rows),unique_decoded_rgb_contents=len({r.get('decoded_sha256') for r in rgb_rows}),
        rgb_content_change_rate_nominal_hz=(len(rgb_rows)-repeats)*fps/len(rgb_rows) if rgb_rows and fps and fps>0 else None,
        rate_interpretation='decoded content changes per nominal video second, not measured distinct sensor exposures or hardware FPS')
