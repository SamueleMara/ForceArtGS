#!/usr/bin/env python3
"""Force-bounded TCP joint fits, checked against image tracks and saved training windows."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
import cv2
import numpy as np
from scipy.ndimage import binary_closing,binary_opening
from scipy.signal import find_peaks,peak_widths
from scipy.spatial.transform import Rotation
from analyze_real_robot_measurements import discover,require_tcp_calibration

cv2.setNumThreads(1)


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(bool,np.bool_)):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,Path):return str(x)
    return x


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path,rows):
    import csv
    path.parent.mkdir(parents=True,exist_ok=True)
    names=sorted({k for row in rows for k in row}) or ['status']
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(clean(v)) if isinstance(v,(list,dict,np.ndarray)) else clean(v) for k,v in row.items()})


def load_fitter(root):
    path=Path(root)/'trajectory_constraint_fit.py'
    spec=importlib.util.spec_from_file_location('existing_tcp_joint_fitter',path);module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def spike_bounds(data,min_prominence=.5,sigma_factor=6.,separation_s=.15):
    f=data['force_norm_N'];t=data['timestamp_s'];finite=np.isfinite(f)&np.isfinite(t)
    baseline=finite&(t<=t[0]+.5);quiet=f[baseline]
    noise=float(1.4826*np.median(np.abs(np.diff(quiet)-np.median(np.diff(quiet))))/np.sqrt(2)) if len(quiet)>2 else 0.
    threshold=max(min_prominence,sigma_factor*noise);candidates=[]
    # Process finite monotone segments independently; do not interpolate holes.
    ids=np.flatnonzero(finite);segments=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    for segment in segments:
        if len(segment)<3:continue
        dt=np.diff(t[segment]);positive=dt[dt>0]
        if not len(positive):continue
        step=float(np.median(positive));peaks,properties=find_peaks(f[segment],prominence=threshold)
        widths=peak_widths(f[segment],peaks)[0]*step if len(peaks) else []
        for i,prom,width in zip(segment[peaks],properties['prominences'],widths):
            candidates.append(dict(sample=int(i),source_frame=int(data['frame_index'][i]),timestamp_s=float(t[i]),force_N=float(f[i]),prominence_N=float(prom),half_prominence_width_s=float(width)))
    # Treat nearby peaks as one force event; preserve every candidate in output.
    groups=[]
    for p in candidates:
        if groups and p['timestamp_s']-groups[-1][-1]['timestamp_s']<separation_s:groups[-1].append(p)
        else:groups.append([p])
    events=[max(g,key=lambda p:p['prominence_N']) for g in groups]
    window=[events[0]['timestamp_s'],events[-1]['timestamp_s']] if len(events)>=2 else None
    return dict(status='PROPOSED_FORCE_BOUNDS' if window else 'NO_TWO_DISTINCT_FORCE_SPIKES',window_s=window,
        baseline_force_N=float(np.median(quiet)) if len(quiet) else None,baseline_difference_sigma_N=noise,
        prominence_threshold_N=threshold,candidates=candidates,events=events,
        boundary_caveats=['A peak at the first/last recorded sample cannot be identified by an interior local maximum test',
            'First/last prominent force events propose contact bounds; visual correspondence must corroborate them'],
        broad_boundary_peak=bool(window and max(events[0]['half_prominence_width_s'],events[-1]['half_prominence_width_s'])>.5))


def fit_model(points,kind,fitter):
    center=points.mean(0);q=points-center;directions=fitter._principal_directions(q)
    if kind=='prismatic':
        u=fitter.normalize(directions[0]);u*=1 if u[np.argmax(np.abs(u))]>=0 else -1
        return dict(type=kind,axis_direction=u,axis_point=center-np.dot(center,u)*u,line_point=center,
            axis_location_identifiable=False,axis_location_note='A point translating along a line identifies direction; the object joint may be any parallel line')
    axis=fitter.fit_axis_with_direction(q,directions[2]);u=axis['axis_direction'];u*=1 if u[np.argmax(np.abs(u))]>=0 else -1
    c=axis['axis_point']+center;c-=np.dot(c,u)*u
    rho,phi,h,_=fitter.cylindrical_coordinates(points,u,c);circle=fitter.fit_circle_model(rho,h)
    return dict(type=kind,axis_direction=u,axis_point=c,circle_center=c+circle['h0']*u,radius_m=circle['radius'],
        angular_extent_rad=float(np.ptp(phi)),axis_location_identifiable=True)


def project_model(points,model):
    u=model['axis_direction']
    if model['type']=='prismatic':
        center=model['line_point'];return center+np.outer((points-center)@u,u)
    center=model['circle_center'];delta=points-center;radial=delta-np.outer(delta@u,u);length=np.linalg.norm(radial,axis=1)
    return center+radial/np.maximum(length[:,None],1e-15)*model['radius_m']


def compare_joints(points,fitter):
    points=np.asarray(points,float);points=points[np.isfinite(points).all(axis=1)]
    if len(points)<15:return dict(status='UNOBSERVABLE',reason='FEWER_THAN_15_FINITE_TCP_SAMPLES',samples=len(points))
    extent=float(np.linalg.norm(np.ptp(points,axis=0)))
    if extent<.015:return dict(status='UNOBSERVABLE',reason='TCP_MOTION_EXTENT_LESS_THAN_15MM',extent_m=extent,samples=len(points))
    models={};folds=np.array_split(np.arange(len(points)),5)
    for kind,k in [('prismatic',4),('revolute',6)]:
        try:
            model=fit_model(points,kind,fitter);pred=project_model(points,model);error=np.linalg.norm(points-pred,axis=1)
            cv_error=[];axes=[];cv_failures=[]
            for i,held in enumerate(folds):
                train=np.concatenate([v for j,v in enumerate(folds) if i!=j])
                try:
                    fit=fit_model(points[train],kind,fitter);cv_error.extend(np.linalg.norm(points[held]-project_model(points[held],fit),axis=1));axes.append(fit['axis_direction'])
                except Exception as e:cv_failures.append(str(e))
            count=2*len(points);bic=float(count*np.log(max(float(np.sum(error**2))/count,1e-18))+k*np.log(count))
            model.update(rmse_m=float(np.sqrt(np.mean(error**2))),mae_m=float(error.mean()),p95_m=float(np.percentile(error,95)),
                bic=bic,parameter_count=k,heldout_rmse_m=float(np.sqrt(np.mean(np.square(cv_error)))) if len(cv_error)==len(points) else None,
                heldout_fold_failures=cv_failures,heldout_folds=5,
                fold_axis_deviation_deg=[float(np.degrees(np.arccos(np.clip(abs(np.dot(a,model['axis_direction'])),0,1)))) for a in axes])
            if kind=='revolute':model['axis_location_identifiable']=model['angular_extent_rad']>=.15 and model['radius_m']<10*extent
            models[kind]=model
        except Exception as e:models[kind]=dict(type=kind,status='FIT_FAILED',error=str(e),bic=None)
    available=[v for v in models.values() if v.get('bic') is not None]
    if len(available)!=2:return dict(status='UNOBSERVABLE',reason='BOTH_MODELS_MUST_FIT',models=models,samples=len(points))
    order=sorted(available,key=lambda m:m['bic']);raw_winner=order[0]['type'];delta=order[1]['bic']-order[0]['bic']
    for model in available:
        model['selection_eligible']=model['type']=='prismatic' or model['axis_location_identifiable']
        if not model['selection_eligible']:model['selection_exclusion']='UNIDENTIFIABLE_REVOLUTE_CURVATURE_OR_RADIUS'
    eligible=sorted([m for m in available if m['selection_eligible']],key=lambda m:m['bic']);winner=eligible[0]['type']
    cv=[v for v in available if v['heldout_rmse_m'] is not None];cv_winner=min(cv,key=lambda m:m['heldout_rmse_m'])['type'] if len(cv)==2 else None
    ratio=max(v['heldout_rmse_m'] for v in cv)/max(min(v['heldout_rmse_m'] for v in cv),1e-9) if len(cv)==2 else None
    ambiguous=delta<6 or cv_winner!=winner or ratio is None or ratio<1.1
    poor=eligible[0]['rmse_m']>max(.005,.05*extent)
    status='POOR_FIXED_JOINT_FIT' if poor else 'AMBIGUOUS' if ambiguous else 'CONDITIONAL_SUPPORTED'
    if winner=='prismatic' and not models['revolute'].get('axis_location_identifiable',False) and not poor:status='LINE_LIKE_CURVATURE_UNRESOLVED'
    if eligible[0]['fold_axis_deviation_deg'] and max(eligible[0]['fold_axis_deviation_deg'])>15:status='AXIS_UNSTABLE_ACROSS_HELDOUT_FOLDS'
    return dict(status=status,best_scored_model=winner,raw_bic_best_model=raw_winner,heldout_best_model=cv_winner,delta_bic=float(delta),heldout_rmse_ratio=ratio,
        models=models,samples=len(points),extent_m=extent,
        score_convention='Same 3D orthogonal distance in meters; BIC uses 2N residual coordinates and 4/6 line/circle parameters; five contiguous temporal holdouts. Best scored model excludes unidentifiable revolute curvature/radius; raw BIC winner is also retained.',
        interpretation='TCP path constraint conditional on sustained no-slip contact with one rigid moving part; not independent object-joint GT')


def tracked_pair(previous,current,previous_mask,current_mask,ignore=None,current_ignore=None):
    """Forward/backward LK matches; static-background translation is diagnostic only."""
    h,w=previous.shape;all_p=[];counts=[]
    ignore=np.zeros_like(previous_mask) if ignore is None else ignore
    current_ignore=np.zeros_like(current_mask) if current_ignore is None else current_ignore
    for region in [previous_mask&~ignore,(~cv2.dilate(previous_mask.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool))&~ignore]:
        mask=cv2.erode(region.astype(np.uint8),np.ones((3,3),np.uint8))*255
        p=cv2.goodFeaturesToTrack(previous,250,.015,4,mask=mask)
        counts.append(0 if p is None else len(p))
        if p is not None:all_p.append(p)
    if not all_p:return dict(status='NO_FEATURES',tracked_object=0,tracked_background=0,moving_object=0,active=False),None
    p=np.concatenate(all_p);q,status,_=cv2.calcOpticalFlowPyrLK(previous,current,p,None,winSize=(21,21),maxLevel=3)
    back,reverse,_=cv2.calcOpticalFlowPyrLK(current,previous,q,None,winSize=(21,21),maxLevel=3)
    p=p[:,0];q=q[:,0];back=back[:,0];good=status[:,0].astype(bool)&reverse[:,0].astype(bool)&(np.linalg.norm(back-p,axis=1)<.75)
    xy=np.rint(q).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h);good&=inside
    obj=np.arange(len(p))<counts[0];object_keep=good&obj
    object_keep[inside]&=(current_mask&~current_ignore)[xy[inside,1],xy[inside,0]]
    bg=good&~obj;bg[inside]&=~current_ignore[xy[inside,1],xy[inside,0]]
    flow=(q-p)*2.;background=np.median(flow[bg],axis=0) if bg.sum()>=8 else np.zeros(2)
    bg_vectors=flow[bg]
    # Cameras are fixed. Robot features outside the object mask must not set
    # the static-room noise floor: retain the slower half as a conditional control.
    if len(bg_vectors)>=16:
        bg_vectors=bg_vectors[np.argsort(np.linalg.norm(bg_vectors,axis=1))[:len(bg_vectors)//2]]
        background=np.median(bg_vectors,axis=0)
    bg_distance=np.linalg.norm(bg_vectors-background,axis=1)
    noise=float(np.median(bg_distance)+4*1.4826*np.median(np.abs(bg_distance-np.median(bg_distance)))) if len(bg_distance) else 0.
    threshold=max(.35,noise);distance=np.linalg.norm(flow[object_keep]-background,axis=1);moving=distance>threshold
    reliable=object_keep.sum()>=6 and bg.sum()>=8
    return dict(status='MEASURED' if reliable else 'LOW_FEATURE_SUPPORT',tracked_object=int(object_keep.sum()),tracked_background=int(bg.sum()),
        moving_object=int(moving.sum()),moving_fraction=float(moving.mean()) if len(moving) else None,
        object_flow_p90_px=float(np.percentile(distance,90)) if len(distance) else None,background_motion_px=background.tolist(),
        static_background_control_tracks=len(bg_vectors),motion_threshold_px=threshold,active=bool(reliable and moving.sum()>=max(3,.03*object_keep.sum()))),dict(p=p[object_keep]*2,q=q[object_keep]*2,moving=moving)


def robot_exclusion(data,index,camera,K,shape):
    """Analysis-only projected wrist/shaft envelope; original masks stay intact."""
    excluded=np.zeros(shape,np.uint8);ee=data['ee_position_m'][index];tcp=data['tcp_position_m'][index]
    if not np.isfinite(np.r_[ee,tcp]).all():return excluded.astype(bool)
    length=float(np.linalg.norm(tcp-ee))
    if length<1e-6:return excluded.astype(bool)
    direction=(tcp-ee)/length;T=np.linalg.inv(np.array(camera['pose_matrix_4x4']))
    # Stop the thin shaft model 25 mm before the TCP. The larger wrist may
    # still obscure nearby object pixels, which reduces support, not certainty.
    offsets=np.r_[[-.06,-.03,0.,.015],np.arange(.03,max(.03,length-.025),.01)]
    for distance in offsets:
        point=ee+distance*direction;local=T[:3,:3]@point+T[:3,3]
        if local[2]<=0:continue
        uv=K@local;uv=uv[:2]/uv[2];radius=(.04 if distance<=.015 else .012)*max(K[0,0],K[1,1])/local[2]
        if np.isfinite(uv).all():cv2.circle(excluded,tuple(np.rint(uv).astype(int)),max(1,int(np.ceil(radius))),1,-1)
    return excluded.astype(bool)


def visual_motion(scene,conversion,data,cameras):
    root=Path(scene);frames=conversion.get('output_frames',conversion['source_frames']);source=conversion['source_frames'];lookup={int(f):i for i,f in enumerate(data['frame_index'])}
    times=np.array([data['timestamp_s'][lookup[f]] for f in source]);rows=[];activity=[];reliable=[]
    camera_meta={c['name']:c for c in json.loads((root/'metadata/cameras.json').read_text())['cameras']};K=np.array(conversion['rectified_K'])
    for camera in cameras:
        previous=previous_mask=previous_ignore=None;active=np.zeros(len(frames),bool);valid=active.copy()
        for i,frame in enumerate(frames):
            rgb=root/'rgb'/camera/f'frame_{frame:06d}.png';mask=root/'mask'/camera/f'frame_{frame:06d}.png'
            gray=cv2.imread(str(rgb),cv2.IMREAD_GRAYSCALE);m=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
            if gray is None or m is None:
                rows.append(dict(camera=camera,processed_frame=frame,source_frame=source[i],timestamp_s=times[i],status='RGB_OR_MASK_MISSING',active=False));previous=previous_mask=None;continue
            ignored=robot_exclusion(data,lookup[source[i]],camera_meta[camera],K,gray.shape)
            excluded_object=int((ignored&(m>0)).sum())
            gray=cv2.resize(gray,(gray.shape[1]//2,gray.shape[0]//2),interpolation=cv2.INTER_AREA);m=cv2.resize(m,gray.shape[::-1],interpolation=cv2.INTER_NEAREST)>0
            ignored=cv2.resize(ignored.astype(np.uint8),gray.shape[::-1],interpolation=cv2.INTER_NEAREST)>0
            if previous is not None:
                result,_=tracked_pair(previous,gray,previous_mask,m,previous_ignore,ignored);rows.append(dict(camera=camera,processed_frame=frame,source_frame=source[i],timestamp_s=times[i],robot_excluded_object_pixels=excluded_object,**result))
                active[i]=result['active'];valid[i]=result['status']=='MEASURED'
            previous,previous_mask,previous_ignore=gray,m,ignored
        activity.append(active);reliable.append(valid)
    votes=np.sum(activity,axis=0);support=np.sum(reliable,axis=0);step=float(np.median(np.diff(times))) if len(times)>1 else 0.
    width=max(2,int(np.ceil(.12/max(step,.001))));motion=binary_opening(binary_closing(votes>=2,structure=np.ones(3)),structure=np.ones(width))
    multiview=bool(motion.any())
    if not multiview:motion=binary_opening(binary_closing(votes>=1,structure=np.ones(3)),structure=np.ones(width))
    ids=np.flatnonzero(motion);window=[float(times[ids[0]]),float(times[ids[-1]])] if len(ids) else None
    summary=dict(status='MULTIVIEW_IMAGE_MOTION' if multiview else 'SINGLE_VIEW_IMAGE_MOTION_CONDITIONAL' if window else 'UNOBSERVABLE_NO_PERSISTENT_IMAGE_MOTION',window_s=window,
        cameras=cameras,frames=len(frames),two_view_supported_frames=int((support>=2).sum()),moving_frames=int(motion.sum()),
        correspondence='Forward/backward Lucas-Kanade tracks inside both object masks; stationary-background median flow and noise control; no applied calibration or synchronization change',
        robot_occlusion_control='Analysis-only projected recorded-EE/TCP wrist and shaft envelope: 40-mm wrist radius, 12-mm shaft radius, shaft stops 25 mm before TCP; excluded from object/background tracks; not a replacement object mask',
        caveat='Image motion supports an interval, not physical contact or object-part identity; robot leakage or missed mask regions remain possible')
    edges=np.diff(np.r_[False,motion,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
    summary['motion_segments']=[dict(start_s=float(times[a]),end_s=float(times[b]),start_frame=int(frames[a]),end_frame=int(frames[b])) for a,b in zip(starts,ends)]
    return summary,rows,dict(frames=np.array(frames),source=np.array(source),times=times,votes=votes,support=support,motion=motion)


def training_windows(roots):
    result={}
    for root in roots:
        for path in sorted(Path(root).glob('*/*/pipeline/*/bidirectional/*/stage_manifest.json')):
            r=json.loads(path.read_text());s=r.get('signature',{});config=s.get('config',{})
            if r.get('status')!='complete' or s.get('stage')!='phase1' or 'MSTART' not in config or 'MEND' not in config:continue
            source=s['source'];name=Path(source['dir']).name
            result.setdefault(name,[]).append(dict(path=str(path),sha256=sha(path),source_dir=source['dir'],start_frame=int(config['MSTART']),end_frame=int(config['MEND']),direction=s.get('direction'),cameras=source['cameras']))
    return result


def compare_intervals(force,visual,timeline,training):
    fw=force['window_s'];vw=visual['window_s'];t=timeline['times'];active=timeline['motion'];selected=fw or vw
    result=dict(selected_window_s=selected,selected_interval_source='first_last_force_spikes' if fw else 'visual_only_no_two_force_spikes' if vw else None,
        status='UNOBSERVABLE',training_windows=[])
    if fw and vw:
        covered=active&(t>=fw[0]-.15)&(t<=fw[1]+.15);fraction=float(covered.sum()/active.sum())
        agrees=fraction>=.9 and vw[0]>=fw[0]-.2 and vw[1]<=fw[1]+.2
        result.update(status=('VISUALLY_CORROBORATED' if visual.get('status','MULTIVIEW_IMAGE_MOTION')=='MULTIVIEW_IMAGE_MOTION' else 'SINGLE_VIEW_CONDITIONAL_CORROBORATION') if agrees else 'FORCE_VISUAL_INTERVAL_CONFLICT',
            visual_motion_inside_force_fraction=fraction,visual_start_minus_first_spike_s=vw[0]-fw[0],last_spike_minus_visual_end_s=fw[1]-vw[1])
    elif fw:result['status']='FORCE_BOUNDS_VISUAL_UNOBSERVABLE'
    elif vw:result['status']='VISUAL_FALLBACK_FORCE_BOUNDS_UNOBSERVABLE'
    if selected:
        ids=np.flatnonzero((t>=selected[0])&(t<=selected[1]));result['selected_converted_frames']=[int(timeline['frames'][ids[0]]),int(timeline['frames'][ids[-1]])] if len(ids) else None
    for item in training:
        item=dict(item);indices={int(f):i for i,f in enumerate(timeline['frames'])}
        if item['start_frame'] not in indices or item['end_frame'] not in indices:item['comparison_status']='TRAINING_FRAMES_NOT_IN_CONVERSION'
        else:
            w=sorted([float(t[indices[item['start_frame']]]),float(t[indices[item['end_frame']]])]);item['window_s']=w
            item['comparison_status']='MEASURED_SAVED_TRAINING_WINDOW'
            if selected:
                overlap=max(0.,min(w[1],selected[1])-max(w[0],selected[0]));union=max(w[1],selected[1])-min(w[0],selected[0])
                item.update(interval_iou=overlap/max(union,1e-12),selected_interval_fraction_covered=overlap/max(selected[1]-selected[0],1e-12),training_preinteraction_s=max(0.,selected[0]-w[0]),training_postinteraction_s=max(0.,w[1]-selected[1]))
        result['training_windows'].append(item)
    result['training_interval_status']='RECORDED_WINDOWS_COMPARED' if training else 'NO_SAVED_TRAINING_WINDOW_FOUND; dataset coverage direction is not a training motion interval'
    return result


def plots(folder,scene,data,force,visual,timeline,interval,fit,secondary,training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True);t=data['timestamp_s']
    axes[0].plot(t,data['force_norm_N'],lw=.8,label='Recorded force norm (N)')
    events=force['events']
    if events:axes[0].scatter([p['timestamp_s'] for p in events],[p['force_N'] for p in events],c='red',s=22,label='Detected force events')
    for ax in axes:
        if force['window_s']:ax.axvspan(*force['window_s'],alpha=.15,color='orange',label='First–last force spikes')
        if visual['window_s']:ax.axvspan(*visual['window_s'],alpha=.12,color='green',label='Multiview image motion')
        for i,w in enumerate(interval['training_windows']):
            if w.get('window_s') and i==0:ax.axvspan(*w['window_s'],alpha=.08,color='blue',label='Saved training window (first run)')
        ax.grid(alpha=.2)
    axes[1].plot(timeline['times'],timeline['votes'],label='Views with tracked object motion');axes[1].plot(timeline['times'],timeline['support'],alpha=.5,label='Views with reliable tracks')
    axes[2].plot(t,data['tcp_linear_speed_m_s'],label='Calibrated TCP speed (m/s)')
    axes[0].set(ylabel='N');axes[1].set(ylabel='Camera count');axes[2].set(ylabel='m/s',xlabel='Recorded episode timestamp (s)')
    for ax in axes:ax.legend(fontsize=8,loc='upper right')
    fig.suptitle(scene+'\n'+interval['status']);fig.tight_layout();fig.savefig(folder/'interaction_bounds.png',dpi=130);plt.close(fig)
    window=interval['selected_window_s']
    if window and 'models' in fit:
        inside=(t>=window[0])&(t<=window[1])&np.isfinite(data['tcp_position_m']).all(axis=1);p=data['tcp_position_m'][inside]
        fig=plt.figure(figsize=(14,6))
        for i,kind in enumerate(['prismatic','revolute']):
            ax=fig.add_subplot(1,2,i+1,projection='3d');m=fit['models'][kind];ax.plot(*p.T,lw=.7,color='black',label='TCP inside selected interval')
            if m.get('bic') is not None:
                curve=project_model(p,m);ax.plot(*curve.T,label=kind+' projection',color='orange')
                center=m.get('circle_center',m.get('line_point'));span=max(np.linalg.norm(np.ptp(p,axis=0)),.01);axis=center+np.outer([-span/2,span/2],m['axis_direction']);ax.plot(*axis.T,color='red',label='Estimated axis/direction')
                title=f'{kind}: RMS {1000*m["rmse_m"]:.2f} mm; CV {1000*m["heldout_rmse_m"]:.2f} mm' if m['heldout_rmse_m'] is not None else kind+'; heldout failed'
                if kind=='revolute':title+=f'\nR={m["radius_m"]:.3g} m; arc={np.degrees(m["angular_extent_rad"]):.2f} deg'+(' — EXCLUDED' if not m['selection_eligible'] else '')
                else:title+='\nObject-axis location is not identifiable from this point trace'
                ax.set_title(title,fontsize=9)
                values=np.concatenate([p,curve]);c=(values.max(0)+values.min(0))/2;r=max(np.ptp(values,axis=0).max()/2,.01)
                ax.set(xlim=(c[0]-r,c[0]+r),ylim=(c[1]-r,c[1]+r),zlim=(c[2]-r,c[2]+r));ax.set_box_aspect((1,1,1))
            ax.set(xlabel='base_link x (m)',ylabel='y (m)',zlabel='z (m)');ax.legend(fontsize=7)
        fig.suptitle(scene+'\nBest scored: '+str(fit.get('best_scored_model'))+'; '+fit['status']+'; '+interval['status']);fig.tight_layout();fig.savefig(folder/'joint_models.png',dpi=130);plt.close(fig)


def contact_sheet(folder,root,cameras,timeline,force,visual,training,data,conversion):
    import matplotlib.pyplot as plt
    times=timeline['times'];frames=timeline['frames'];ids={0,len(frames)-1}
    lookup={int(f):i for i,f in enumerate(data['frame_index'])};calibration=json.loads((root/'metadata/cameras.json').read_text())['cameras'];camera_by_name={c['name']:c for c in calibration}
    K=np.array(conversion['rectified_K']);correspondences={}
    for window in [force['window_s'],visual['window_s']]:
        if window:
            for t in window:
                i=int(np.argmin(np.abs(times-t)));ids.update([max(0,i-1),i,min(len(frames)-1,i+1)])
    for record in training[:1]:
        for f in [record['start_frame'],record['end_frame']]:
            where=np.flatnonzero(frames==f)
            if len(where):ids.add(int(where[0]))
    ids=sorted(ids);fig,axes=plt.subplots(len(cameras),len(ids),figsize=(2.7*len(ids),2.3*len(cameras)),squeeze=False)
    for row,camera in enumerate(cameras):
        for col,i in enumerate(ids):
            p=root/'rgb'/camera/f'frame_{frames[i]:06d}.png';im=cv2.imread(str(p));ax=axes[row,col]
            if im is not None:
                ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                j=lookup[int(timeline['source'][i])];point=data['tcp_position_m'][j]
                if np.isfinite(point).all():
                    T=np.linalg.inv(np.array(camera_by_name[camera]['pose_matrix_4x4']));xyz=T[:3,:3]@point+T[:3,3];uv=K@xyz
                    if xyz[2]>0:
                        uv=uv[:2]/uv[2]
                        if 0<=uv[0]<im.shape[1] and 0<=uv[1]<im.shape[0]:ax.plot(uv[0],uv[1],marker='+',color='yellow',markersize=10,mew=1.5)
                if i>0:
                    before=cv2.imread(str(root/'rgb'/camera/f'frame_{frames[i-1]:06d}.png'),cv2.IMREAD_GRAYSCALE)
                    before_mask=cv2.imread(str(root/'mask'/camera/f'frame_{frames[i-1]:06d}.png'),cv2.IMREAD_GRAYSCALE)
                    mask=cv2.imread(str(root/'mask'/camera/f'frame_{frames[i]:06d}.png'),cv2.IMREAD_GRAYSCALE)
                    if before is not None and before_mask is not None and mask is not None:
                        size=(im.shape[1]//2,im.shape[0]//2)
                        gray=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),size,interpolation=cv2.INTER_AREA)
                        current_exclusion=robot_exclusion(data,j,camera_by_name[camera],K,im.shape[:2])
                        previous_exclusion=robot_exclusion(data,lookup[int(timeline['source'][i-1])],camera_by_name[camera],K,im.shape[:2])
                        _,tracks=tracked_pair(cv2.resize(before,size,interpolation=cv2.INTER_AREA),gray,cv2.resize(before_mask,size,interpolation=cv2.INTER_NEAREST)>0,cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)>0,
                            cv2.resize(previous_exclusion.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0,cv2.resize(current_exclusion.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0)
                        contours,_=cv2.findContours(current_exclusion.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if len(contour)>2:ax.plot(contour[:,0,0],contour[:,0,1],color='red',alpha=.4,lw=.6)
                        correspondences[f'{camera}_{frames[i]:06d}_robot_exclusion']=current_exclusion
                        if tracks is not None:
                            for key,value in tracks.items():correspondences[f'{camera}_{frames[i]:06d}_{key}']=value
                            moving=tracks['moving'];p=tracks['p'][moving][:80];delta=(tracks['q']-tracks['p'])[moving][:80]*4
                            if len(p):ax.quiver(p[:,0],p[:,1],delta[:,0],delta[:,1],color='cyan',angles='xy',scale_units='xy',scale=1,width=.004)
            ax.set_title(f'{camera}; RGBD {frames[i]}\nraw {timeline["source"][i]}; {times[i]:.2f}s',fontsize=8);ax.axis('off')
    fig.suptitle(root.name+' — cyan: object tracks ×4; yellow: TCP projection; red: robot exclusion control',fontsize=11);fig.tight_layout();fig.savefig(folder/'boundary_contact_sheet.jpg',dpi=100);plt.close(fig)
    np.savez_compressed(folder/'boundary_correspondences.npz',**correspondences)


def run_scene(job,config,training):
    name=job['name'];folder=Path(config['output_root'])/name;folder.mkdir(parents=True,exist_ok=True)
    try:
        measurement=Path(config['measurements_root'])/name;report=json.loads((measurement/'summary.json').read_text())
        require_tcp_calibration(report.get('tcp_calibration',{}))
        local_tool=job['conversion'].parent/'robot_tool.json'
        if local_tool.is_file():require_tcp_calibration(json.loads(local_tool.read_text()))
        for filename in ['result.json','error.log','force_events.csv','visual_correspondences.csv','interaction_tcp.npz','interaction_bounds.png','joint_models.png','boundary_contact_sheet.jpg','boundary_correspondences.npz']:
            if not (folder/filename).resolve().is_relative_to(folder.resolve()):raise ValueError('Episode output file escapes analysis folder')
        for filename in ['error.log','joint_models.png','result.json']:
            if (folder/filename).is_file():(folder/filename).unlink()
        measurement=Path(config['measurements_root'])/name;report=json.loads((measurement/'summary.json').read_text())
        if report['source_parquet_sha256']!=sha(job['parquet']):raise ValueError('TCP measurements refer to different Parquet bytes')
        archive=measurement/'measurements.npz'
        with np.load(archive,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        for key in ['tcp_position_m','tcp_linear_speed_m_s','force_norm_N','timestamp_s','frame_index']:
            if key not in data:raise ValueError('Missing calibrated measurement array: '+key)
        local_tool=job['conversion'].parent/'robot_tool.json'
        if local_tool.is_file():
            tool=json.loads(local_tool.read_text())
            if tool['tool']!=report['tool'] or not np.array_equal(tool['T_ee_tcp'],report['tcp_calibration']['T_ee_tcp']):
                raise ValueError('Cached TCP calibration differs from current scene metadata; regenerate robot measurements')
        conversion=json.loads(job['conversion'].read_text());cameras=sorted(p.name for p in (job['recording_root']/'rgb').iterdir() if p.is_dir())
        force=spike_bounds(data,config['min_prominence_N'],config['force_sigma'],config['spike_separation_s'])
        visual,rows,timeline=visual_motion(job['recording_root'],conversion,data,cameras)
        comparison=compare_intervals(force,visual,timeline,training);window=comparison['selected_window_s'];fitter=load_fitter(config['joint_code'])
        t=data['timestamp_s'];inside=(t>=window[0])&(t<=window[1]) if window else np.zeros(len(t),bool)
        fit=compare_joints(data['tcp_position_m'][inside],fitter) if window else dict(status='NOT_RUN_NO_INTERACTION_INTERVAL')
        secondary=None
        if visual['window_s'] and comparison['status']=='FORCE_VISUAL_INTERVAL_CONFLICT':
            a,b=visual['window_s'];secondary=compare_joints(data['tcp_position_m'][(t>=a)&(t<=b)],fitter)
        orientations=data['ee_rotvec_rad'][inside];orientations=orientations[np.isfinite(orientations).all(axis=1)]
        orientation_extent=float(np.degrees((Rotation.from_rotvec(orientations[:1]).inv()*Rotation.from_rotvec(orientations)).magnitude().max())) if len(orientations) else None
        result=dict(scene=name,status='ANALYZED',force=force,visual=visual,interval=comparison,joint_fit=fit,visual_interval_diagnostic_fit=secondary,
            ee_orientation_extent_deg=orientation_extent,
            tool=report['tool'],tcp_calibration=report['tcp_calibration'],coordinate_frame='base_link',
            production_training_modified=False,original_data_modified=False,
            provenance=dict(parquet=str(job['parquet']),parquet_sha256=report['source_parquet_sha256'],tcp_archive=str(archive),tcp_archive_sha256=sha(archive),
                conversion_sha256=sha(job['conversion']),fitter_source_sha256=sha(Path(config['joint_code'])/'trajectory_constraint_fit.py'),analysis_source_sha256=sha(__file__)),
            limitations=['A contacting TCP can slip or change contact point; a TCP curve is not automatically an object-fixed landmark',
                'Absolute TCP/joint coordinates inherit the user-supplied calibration reference assumption',
                'Prismatic joint-axis location is not identifiable from one point trajectory; only direction is constrained'])
        csv_rows(folder/'force_events.csv',force['candidates']);csv_rows(folder/'visual_correspondences.csv',rows)
        np.savez_compressed(folder/'interaction_tcp.npz',source_frame=data['frame_index'][inside],timestamp_s=t[inside],tcp_position_m=data['tcp_position_m'][inside],
            rgbd_frame_index=data['rgbd_frame_index'][inside],force_norm_N=data['force_norm_N'][inside])
        plots(folder,name,data,force,visual,timeline,comparison,fit,secondary,training)
        contact_sheet(folder,job['recording_root'],cameras,timeline,force,visual,training,data,conversion)
        save_json(folder/'result.json',result)
        selected=np.flatnonzero(inside);rgbd=comparison.get('selected_converted_frames');best=fit.get('models',{}).get(fit.get('best_scored_model'),{})
        first_training=comparison['training_windows'][0] if comparison['training_windows'] else {}
        return dict(scene=name,status='ANALYZED',interval_status=comparison['status'],interval_source=comparison['selected_interval_source'],
            start_s=window[0] if window else None,end_s=window[1] if window else None,first_force_spike_s=force['window_s'][0] if force['window_s'] else None,
            last_force_spike_s=force['window_s'][1] if force['window_s'] else None,events=len(force['events']),
            selected_source_frame_start=int(data['frame_index'][selected[0]]) if len(selected) else None,selected_source_frame_end=int(data['frame_index'][selected[-1]]) if len(selected) else None,
            selected_rgbd_frame_start=rgbd[0] if rgbd else None,selected_rgbd_frame_end=rgbd[1] if rgbd else None,
            first_spike_source_frame=force['events'][0]['source_frame'] if force['events'] else None,last_spike_source_frame=force['events'][-1]['source_frame'] if force['events'] else None,
            visual_window_s=visual['window_s'],visual_check_status=visual['status'],
            training_start_frame=first_training.get('start_frame'),training_end_frame=first_training.get('end_frame'),training_interval_iou=first_training.get('interval_iou'),
            best_scored_model=fit.get('best_scored_model'),joint_fit_status=fit['status'],delta_bic=fit.get('delta_bic'),
            ee_orientation_extent_deg=orientation_extent,
            raw_bic_best_model=fit.get('raw_bic_best_model'),axis_direction_base_link=best.get('axis_direction'),axis_point_base_link_m=best.get('axis_point'),axis_location_identifiable=best.get('axis_location_identifiable'),radius_m=best.get('radius_m'),
            prismatic_bic=fit.get('models',{}).get('prismatic',{}).get('bic'),revolute_bic=fit.get('models',{}).get('revolute',{}).get('bic'),
            prismatic_cv_rmse_mm=1000*fit['models']['prismatic']['heldout_rmse_m'] if fit.get('models',{}).get('prismatic',{}).get('heldout_rmse_m') is not None else None,
            revolute_cv_rmse_mm=1000*fit['models']['revolute']['heldout_rmse_m'] if fit.get('models',{}).get('revolute',{}).get('heldout_rmse_m') is not None else None,
            prismatic_rmse_mm=1000*fit['models']['prismatic']['rmse_m'] if fit.get('models',{}).get('prismatic',{}).get('rmse_m') is not None else None,
            revolute_rmse_mm=1000*fit['models']['revolute']['rmse_m'] if fit.get('models',{}).get('revolute',{}).get('rmse_m') is not None else None,
            saved_training_runs=len(training),result_path=str(folder/'result.json'))
    except Exception as e:
        import traceback
        (folder/'error.log').write_text(traceback.format_exc());return dict(scene=name,status='FAILED',error=str(e))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real-root',type=Path,required=True);p.add_argument('--measurements-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--scene',nargs='+',help='Optional exact scene names; default every converted real scene')
    p.add_argument('--joint-code',type=Path,default=Path('/home/samuelemara/joint_estimation'))
    p.add_argument('--training-results',type=Path,nargs='*',default=[Path('/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours_v6_two_cams')])
    p.add_argument('--min-prominence-N',type=float,default=.5);p.add_argument('--force-sigma',type=float,default=6.);p.add_argument('--spike-separation-s',type=float,default=.15)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--dry-run',action='store_true');p.add_argument('--overwrite',action='store_true')
    a=p.parse_args(argv);jobs=discover([a.real_root]);root=a.output_root.expanduser().resolve()
    if a.workers<1 or min(a.min_prominence_N,a.force_sigma,a.spike_separation_s)<=0:p.error('Workers and detector parameters must be positive')
    if a.scene:
        jobs=[j for j in jobs if j['name'] in a.scene]
        if {j['name'] for j in jobs}!=set(a.scene):p.error('Some selected scenes were not discovered')
    for source in [a.real_root,a.measurements_root,a.joint_code,*a.training_results]:
        source=source.expanduser().resolve()
        if root==source or root.is_relative_to(source) or source.is_relative_to(root):p.error('Analysis output must be separate from all inputs')
    if not (a.joint_code/'trajectory_constraint_fit.py').is_file():p.error('Existing joint fitting module not found')
    for j in jobs:
        if j['conversion'] is None:p.error('This stage requires converted RGB-D scenes with exact source-frame mappings')
        if not (a.measurements_root/j['name']/'measurements.npz').is_file():p.error('Missing calibrated measurements: '+j['name'])
        if not (root/j['name']).resolve().is_relative_to(root):p.error('Scene output escapes output root')
    training=training_windows(a.training_results)
    config=dict(real_root=str(a.real_root.resolve()),measurements_root=str(a.measurements_root.resolve()),output_root=str(root),joint_code=str(a.joint_code.resolve()),
        min_prominence_N=a.min_prominence_N,force_sigma=a.force_sigma,spike_separation_s=a.spike_separation_s,workers=a.workers,
        selected_scenes=[j['name'] for j in jobs],analysis_source_sha256=sha(__file__),existing_fitter_sha256=sha(a.joint_code/'trajectory_constraint_fit.py'))
    revision=subprocess.run(['git','-C',str(a.joint_code),'rev-parse','HEAD'],capture_output=True,text=True)
    config['existing_fitter_commit']=revision.stdout.strip() if revision.returncode==0 else None
    config['existing_fitter_functions']=['_principal_directions','normalize','fit_axis_with_direction','cylindrical_coordinates','fit_circle_model']
    if a.dry_run:print(json.dumps(dict(**config,scenes=len(jobs),scenes_with_training_windows=sum(j['name'] in training for j in jobs)),indent=2));return 0
    if root.exists() and any(root.iterdir()) and not a.overwrite:p.error('Existing analysis output requires --overwrite or a fresh root')
    for filename in ['.matplotlib','manifest.json','summary.csv','summary.json','REPORT.md']:
        if not (root/filename).resolve().is_relative_to(root):p.error('Output file escapes analysis root')
    root.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(root/'.matplotlib');os.environ['MPLBACKEND']='Agg'
    save_json(root/'manifest.json',config);results=[]
    if a.workers==1:
        for job in jobs:
            results.append(run_scene(job,config,training.get(job['name'],[])));print(f'{len(results)}/{len(jobs)}: {job["name"]}: {results[-1]["status"]}',flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(run_scene,j,config,training.get(j['name'],[])) for j in jobs]
            for future in concurrent.futures.as_completed(futures):
                result=future.result();results.append(result);print(f'{len(results)}/{len(jobs)}: {result["scene"]}: {result["status"]}',flush=True)
    results.sort(key=lambda r:r['scene']);csv_rows(root/'summary.csv',results)
    from collections import Counter
    summary=dict(scenes=len(jobs),completed=sum(r['status']=='ANALYZED' for r in results),failed=[r for r in results if r['status']=='FAILED'],
        interval_status_counts=dict(Counter(r.get('interval_status','FAILED') for r in results)),best_scored_models=dict(Counter(r.get('best_scored_model') or 'NOT_FIT' for r in results)),
        fit_status_counts=dict(Counter(r.get('joint_fit_status','FAILED') for r in results)),scenes_with_saved_training_windows=sum(j['name'] in training for j in jobs))
    save_json(root/'summary.json',summary)
    report=['# Force-bounded TCP joint estimation','',json.dumps(summary,indent=2),'',
        'Both candidate joint constraints are fitted by the unchanged functions in the supplied joint_estimation repository. Scores use the same 3D orthogonal distance in meters, with BIC complexity and contiguous temporal holdouts. The prototype classifier’s mixed angle/metre score is not used.',
        '', 'The first and last noise-adaptive force peaks propose the interval. Missing force bounds are explicitly replaced by a labeled two-view visual-only fallback where available. Force/visual conflicts remain flagged; a separate visual-interval fit is retained for diagnosis. Training windows are read from existing completed Phase-1 manifests, never inferred from depth-coverage direction recommendations or changed.',
        '', 'Each result.json contains both fits, axis/direction, center/radius where applicable, held-out scores, interval evidence, exact frame mapping and training comparisons. interaction_bounds.png overlays the evidence; boundary_contact_sheet.jpg overlays tracked motion (cyan, displayed at x4 scale) and calibrated TCP projection (yellow); boundary_correspondences.npz preserves the actual unscaled image matches; joint_models.png shows both constraints.',
        '', 'A circle with tiny observed arc and enormous radius is excluded as unidentifiable even when it slightly improves raw BIC. Both raw scores and the excluded raw winner remain available. A line-like TCP trace does not establish a prismatic object joint. With nearly constant EE orientation, a constant TCP calibration offset changes the trajectory location but not its shape.',
        '', 'These are conditional TCP contact-trajectory constraints. Slip, varying contact, multiple moving parts and robot motion can violate a fixed-joint model. A prismatic axis location is not observable from one tracked point. Absolute coordinates inherit the recorded-EE/calibration-reference assumption. No GT or production calibration/training data is changed.']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n');print(json.dumps(summary,indent=2),flush=True);return int(bool(summary['failed']))


if __name__=='__main__':sys.exit(main())
