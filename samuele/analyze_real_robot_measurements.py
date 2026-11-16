#!/usr/bin/env python3
"""Plot recorded EE, calibrated TCP and six-axis wrench for real episodes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import polars as pl
from scipy.spatial.transform import Rotation

POSE_FIELDS=('pos_x','pos_y','pos_z','rotvec_x','rotvec_y','rotvec_z')
WRENCH_FIELDS=('Fx_N','Fy_N','Fz_N','Tx_Nm','Ty_Nm','Tz_Nm')
FILES=('measurements.npz','measurements.csv','summary.json','measurements.png','ee_trajectory.png',
    'tcp_measurements.png','tcp_trajectory.png','robot_tool.json')


def discover(inputs,episodes=None):
    """Read only known metadata/data directories; never scan videos or depths."""
    jobs={}
    def recording(path):
        conversion=path/'metadata/real_conversion.json'
        if conversion.is_file():
            if (path/'conversion_in_progress.json').exists():raise ValueError(f'Incomplete conversion: {path}')
            meta=json.loads(conversion.read_text());episode=int(meta['episode'])
            add(path/'metadata/raw'/f'episode_{episode:06d}.parquet',path/'metadata/raw/info.json',path.name,conversion)
            return True
        if (path/'meta/info.json').is_file() and (path/'data').is_dir():
            for p in sorted((path/'data').glob('chunk-*/episode_*.parquet')):
                add(p,path/'meta/info.json',f'{path.name}_{p.stem}',None)
            return True
        return False
    def add(parquet,info,name,conversion):
        match=re.fullmatch(r'episode_(\d+)',parquet.stem)
        if not match:raise ValueError(f'Expected episode_NNNNNN.parquet: {parquet}')
        episode=int(match[1])
        if episodes is not None and episode not in episodes:return
        if not parquet.is_file() or not info.is_file():raise FileNotFoundError(f'Missing Parquet or info.json: {parquet}, {info}')
        job=dict(name=name,episode=episode,parquet=parquet.resolve(),info=info.resolve(),conversion=conversion,
            recording_root=info.parent.parent if info.parent.name=='meta' else info.parents[2])
        if name in jobs and jobs[name]!=job:raise ValueError(f'Conflicting episode output name: {name}')
        jobs[name]=job
    for source in inputs:
        source=source.expanduser().resolve()
        if source.is_file():
            for parent in source.parents:
                info=parent/'meta/info.json'
                if info.is_file():add(source,info,f'{parent.name}_{source.stem}',None);break
                info=parent/'metadata/raw/info.json'
                if info.is_file():add(source,info,parent.name,parent/'metadata/real_conversion.json');break
            else:raise ValueError(f'No recording info.json found for {source}')
        elif source.is_dir():
            if not recording(source):
                for child in sorted(source.iterdir()):
                    if child.is_dir():recording(child)
        else:raise FileNotFoundError(source)
    if not jobs:raise ValueError('No matching real episodes found')
    return list(jobs.values())


def vector_column(rows,column,names,required,available=True):
    if not available:return np.full((len(rows),len(required)),np.nan)
    if not all(n in names for n in required):raise ValueError(f'info.json does not define {column}: {required}')
    indices=[names.index(n) for n in required];values=[]
    for row in rows:
        value=row.get(column)
        if value is None:values.append([np.nan]*len(required));continue
        if len(value)!=len(names):raise ValueError(f'{column} length does not match info.json')
        values.append([value[i] if value[i] is not None else np.nan for i in indices])
    result=np.array(values,dtype=float);result[~np.isfinite(result)]=np.nan
    return result


def parse_measurements(rows,info,episode,conversion=None,wrench_available=True):
    if not rows:raise ValueError('Empty Parquet episode')
    frames=np.asarray([r['frame_index'] for r in rows])
    if not np.issubdtype(frames.dtype,np.integer) or np.any(frames<0) or len(set(frames))!=len(frames):
        raise ValueError('Frame indices must be unique nonnegative integers')
    order=np.argsort(frames);frames=frames[order];rows=[rows[int(i)] for i in order]
    if any(r['episode_index']!=episode for r in rows):raise ValueError('Parquet episode_index differs from filename/metadata')
    t=np.asarray([r['timestamp'] for r in rows],float)
    if not np.isfinite(t).all() or np.any(np.diff(t)<0):raise ValueError('Recorded timestamps must be finite and nondecreasing in frame order')
    pose=vector_column(rows,'observation.state',info.get('state_names',[]),POSE_FIELDS)
    wrench=vector_column(rows,'observation.wrench',info.get('wrench_names',[]),WRENCH_FIELDS,wrench_available)
    dt=np.diff(t);dp=np.diff(pose[:,:3],axis=0)
    linear=np.full(len(t),np.nan);angular=linear.copy()
    valid=(dt>0)&np.isfinite(dp).all(axis=1)
    linear[1:]=np.divide(np.linalg.norm(dp,axis=1),dt,out=np.full(len(dt),np.nan),where=valid)
    valid_rotation=np.isfinite(pose[:,3:]).all(axis=1)
    ids=np.flatnonzero((dt>0)&valid_rotation[:-1]&valid_rotation[1:])
    if len(ids):
        relative=Rotation.from_rotvec(pose[ids,3:]).inv()*Rotation.from_rotvec(pose[ids+1,3:])
        angular[ids+1]=relative.magnitude()/dt[ids]
    mapping=np.full(len(t),-1,np.int64)
    if conversion:
        source=conversion['source_frames'];output=conversion.get('output_frames',source)
        valid_indices=all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in source+output)
        if not source or not valid_indices or conversion['episode']!=episode or len(source)!=len(output) or len(set(source))!=len(source) or len(set(output))!=len(output):
            raise ValueError('Invalid converted source/output frame mapping')
        lookup={int(f):i for i,f in enumerate(frames)}
        for a,b in zip(source,output):
            if a not in lookup:raise ValueError(f'RGB-D source frame {a} absent from Parquet')
            mapping[lookup[a]]=b
    return dict(frame_index=frames,timestamp_s=t,time_from_start_s=t-t[0],rgbd_frame_index=mapping,
        ee_position_m=pose[:,:3],ee_rotvec_rad=pose[:,3:],force_N=wrench[:,:3],torque_Nm=wrench[:,3:],
        force_norm_N=np.linalg.norm(wrench[:,:3],axis=1),torque_norm_Nm=np.linalg.norm(wrench[:,3:],axis=1),
        linear_speed_m_s=linear,angular_speed_rad_s=angular,
        displacement_from_first_ee_m=np.linalg.norm(pose[:,:3]-pose[0,:3],axis=1))


def rigid_transform(value):
    T=np.asarray(value,dtype=float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-12,rtol=0):
        raise ValueError('T_ee_tcp must be a finite 4x4 homogeneous transform')
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8,rtol=0) or not np.isclose(np.linalg.det(T[:3,:3]),1.,atol=1e-8,rtol=0):
        raise ValueError('T_ee_tcp rotation must be proper and orthonormal')
    return T


def tool_for_job(job,registry):
    """Explicit scene/recording matches only; an unknown tool never defaults."""
    if registry.get('schema_version')!=1:raise ValueError('Unsupported tool metadata schema')
    conversion=json.loads(job['conversion'].read_text()) if job['conversion'] else None
    source=Path(conversion['source']).name if conversion else job['recording_root'].name
    matches=[(name,e) for name,e in registry['datasets'].items() if e['source_recording']==source and e['episode']==job['episode']]
    if job['name'] in registry['datasets']:
        name=job['name'];entry=registry['datasets'][name]
        if entry['source_recording']!=source or entry['episode']!=job['episode']:raise ValueError(f'Tool assignment source/episode mismatch: {name}')
    elif matches and len({e['tool'] for _,e in matches})==1:name,entry=matches[0]
    else:raise ValueError(f'No unambiguous tool assignment for {job["name"]}; update the tool metadata explicitly')
    if entry.get('recorded_pose_frame')!='ee' or entry.get('coordinate_frame')!='base_link':raise ValueError('Tool assignment must declare recorded EE in base_link')
    tool=registry['tools'][entry['tool']]
    if tool.get('translation_units')!='m':raise ValueError('Tool calibration translations must be meters')
    T=rigid_transform(tool['T_ee_tcp'])
    result=dict(schema_version=1,dataset=job['name'],assignment_dataset=name,source_recording=source,episode=job['episode'],
        tool=entry['tool'],recorded_pose_frame='ee',coordinate_frame='base_link',T_ee_tcp=T.tolist(),translation_units='m',
        transform_convention=registry['transform_convention'],reference_assumption=registry['reference_assumption'],
        calibration_source=tool['calibration_source'],table_height_shift_applied=False)
    for field in ['mounting_id','tcp_calibration_valid','calibration_status']:
        if field in entry:result[field]=entry[field]
    result['wrench_frame']=entry.get('wrench_frame',registry.get('wrench_frame',dict(axes='unspecified',torque_origin='recorded_origin_unspecified')))
    if conversion:
        local=job['conversion'].parent/'robot_tool.json'
        if local.is_file():
            stored=json.loads(local.read_text())
            if stored.get('schema_version')!=1:raise ValueError(f'Unsupported scene tool metadata schema: {local}')
            for field in ['dataset','source_recording','episode','tool','recorded_pose_frame','coordinate_frame','translation_units']:
                if stored.get(field)!=result[field]:raise ValueError(f'Scene tool metadata disagrees with registry: {local}: {field}')
            if not np.array_equal(rigid_transform(stored['T_ee_tcp']),T):raise ValueError(f'Scene tool transform disagrees with registry: {local}')
            if stored.get('tcp_calibration_valid') is False:result['tcp_calibration_valid']=False
            result.update(source_scene_metadata=str(local),source_scene_metadata_sha256=hashlib.sha256(local.read_bytes()).hexdigest())
    return result


def require_tcp_calibration(metadata):
    if metadata.get('tcp_calibration_valid') is False:
        raise ValueError('Uncalibrated sensor mounting: establish this recording\'s EE-to-TCP transform before generating TCP results')


def add_tcp(data,transform):
    """Right-compose a constant local tool transform with every recorded EE pose."""
    T=rigid_transform(transform);p=data['ee_position_m'];rv=data['ee_rotvec_rad'];t=data['timestamp_s']
    good=np.isfinite(rv).all(axis=1);tcp=np.full_like(p,np.nan);rotvec=np.full_like(rv,np.nan)
    R=Rotation.from_rotvec(rv[good]).as_matrix() if good.any() else np.empty((0,3,3))
    if good.any():
        tcp[good]=p[good]+R@T[:3,3]
        rotvec[good]=Rotation.from_matrix(R@T[:3,:3]).as_rotvec()
    dt=np.diff(t);dp=np.diff(tcp,axis=0);linear=np.full(len(t),np.nan);angular=linear.copy()
    valid=(dt>0)&np.isfinite(dp).all(axis=1)
    linear[1:]=np.divide(np.linalg.norm(dp,axis=1),dt,out=np.full(len(dt),np.nan),where=valid)
    ids=np.flatnonzero((dt>0)&good[:-1]&good[1:])
    if len(ids):angular[ids+1]=(Rotation.from_rotvec(rotvec[ids]).inv()*Rotation.from_rotvec(rotvec[ids+1])).magnitude()/dt[ids]
    data.update(tcp_position_m=tcp,tcp_rotvec_rad=rotvec,tcp_linear_speed_m_s=linear,tcp_angular_speed_rad_s=angular,
        displacement_from_first_tcp_m=np.linalg.norm(tcp-tcp[0],axis=1))
    return data


def stats(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    return dict(valid_samples=len(x),missing_samples=int(np.size(values)-len(x)),min=float(x.min()) if len(x) else None,
        max=float(x.max()) if len(x) else None,mean=float(x.mean()) if len(x) else None,
        rms=float(np.sqrt(np.mean(x*x))) if len(x) else None,p95=float(np.percentile(x,95)) if len(x) else None)


def summarize(data):
    t=data['timestamp_s'];dt=np.diff(t);positive=dt[dt>0];p=data['ee_position_m'];steps=np.linalg.norm(np.diff(p,axis=0),axis=1)
    result=dict(samples=len(t),duration_s=float(t[-1]-t[0]),duplicate_timestamps=int((dt==0).sum()),
        median_sample_interval_s=float(np.median(positive)) if len(positive) else None,
        measured_sampling_hz=float(1/np.median(positive)) if len(positive) else None,
        timestamp_gaps_gt_5x_median=int((dt>5*np.median(positive)).sum()) if len(positive) else 0,
        rgbd_selected_samples=int((data['rgbd_frame_index']>=0).sum()),
        ee_valid_position_samples=int(np.isfinite(p).all(axis=1).sum()),
        observed_adjacent_path_length_m=float(np.nansum(steps)),path_length_excludes_invalid_adjacent_pairs=True,
        force=stats(data['force_norm_N']),torque=stats(data['torque_norm_Nm']),
        linear_speed=stats(data['linear_speed_m_s']),angular_speed=stats(data['angular_speed_rad_s']))
    result['wrench_components']={name:stats(data['force_N' if i<3 else 'torque_Nm'][:,i%3]) for i,name in enumerate(WRENCH_FIELDS)}
    valid=np.flatnonzero(np.isfinite(data['force_norm_N']))
    result['peak_force_timestamp_s']=float(t[valid[np.argmax(data['force_norm_N'][valid])]]) if len(valid) else None
    return result


def write_csv(path,data):
    cols={}
    for key,value in data.items():
        if value.ndim==1:cols[key]=value
        else:
            stem,unit=key.rsplit('_',1)
            for i,axis in enumerate('xyz'):cols[f'{stem}_{axis}_{unit}']=value[:,i]
    with path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(cols)
        for values in zip(*cols.values()):writer.writerow([v if np.isfinite(v) else '' for v in values])


def plot(data,name,folder,frame='ee'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    label=frame.upper();linear='linear_speed_m_s' if frame=='ee' else 'tcp_linear_speed_m_s'
    angular='angular_speed_rad_s' if frame=='ee' else 'tcp_angular_speed_rad_s'
    t=data['time_from_start_s'];fig,axes=plt.subplots(4,2,figsize=(14,15))
    for ax,key,title,unit in [(axes[0,0],'force_N','Recorded force','N'),(axes[0,1],'torque_Nm','Recorded torque','N m'),
        (axes[1,0],frame+'_position_m',label+' position in base_link','m'),(axes[1,1],frame+'_rotvec_rad',label+' rotation vector','rad')]:
        for i,axis in enumerate('xyz'):ax.plot(t,data[key][:,i],label=axis,linewidth=.8)
        if key in {'force_N','torque_Nm'}:ax.plot(t,data['force_norm_N' if key=='force_N' else 'torque_norm_Nm'],label='norm',color='black',linewidth=1.)
        ax.set(title=title,ylabel=unit,xlabel='Recorded elapsed time (s)');ax.legend()
    for ax,key,title,unit in [(axes[2,0],linear,label+' linear speed (adjacent samples)','m/s'),(axes[2,1],angular,label+' angular speed (relative rotations)','rad/s')]:
        ax.plot(t,data[key],linewidth=.8);ax.set(title=title,ylabel=unit,xlabel='Recorded elapsed time (s)')
    for ax,key,xlabel in [(axes[3,0],'displacement_from_first_'+frame+'_m','Distance from first '+label+' position (m)'),(axes[3,1],linear,label+' speed (m/s)')]:
        good=np.isfinite(data[key])&np.isfinite(data['force_norm_N'])
        if good.any():
            points=ax.scatter(data[key][good],data['force_norm_N'][good],c=t[good],s=5,cmap='viridis')
            fig.colorbar(points,ax=ax,label='Recorded elapsed time (s)')
        ax.set(xlabel=xlabel,ylabel='Recorded force norm (N)',title='Force vs '+('displacement' if 'displacement' in key else 'speed'))
    for ax in axes.flat:ax.grid(alpha=.25)
    convention='Recorded EE' if frame=='ee' else 'Derived TCP using tool calibration and recorded-EE reference assumption'
    fig.suptitle(name+'\n'+convention+'; wrench axes as recorded; no filtering or baseline subtraction',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(folder/('measurements.png' if frame=='ee' else 'tcp_measurements.png'),dpi=140);plt.close(fig)
    fig=plt.figure(figsize=(10,8));ax=fig.add_subplot(111,projection='3d');p=data[frame+'_position_m']
    ax.plot(*p.T,color='gray',alpha=.5,linewidth=.8)
    if frame=='tcp':ax.plot(*data['ee_position_m'].T,color='gray',linestyle='--',alpha=.5,label='Recorded EE reference');ax.legend()
    good=np.isfinite(p).all(axis=1)&np.isfinite(data['force_norm_N'])
    if good.any():
        pts=ax.scatter(*p[good].T,c=data['force_norm_N'][good],s=8,cmap='viridis');fig.colorbar(pts,ax=ax,shrink=.6,label='Recorded force norm (N)')
    points=np.concatenate([p,data['ee_position_m']]) if frame=='tcp' else p
    valid=points[np.isfinite(points).all(axis=1)]
    if len(valid):
        center=(valid.max(0)+valid.min(0))/2;radius=max(float(np.ptp(valid,axis=0).max())/2,1e-4)
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
        ax.set_box_aspect((1,1,1))
    title='Recorded EE trajectory in base_link (equal axis scales)' if frame=='ee' else 'Derived TCP in base_link (EE-reference assumption; equal axis scales)'
    ax.set(xlabel=label+' x (m)',ylabel=label+' y (m)',zlabel=label+' z (m)');ax.set_title(name+'\n'+title,fontsize=10)
    fig.tight_layout();fig.savefig(folder/(frame+'_trajectory.png'),dpi=140);plt.close(fig)


def analyze(job,output,overwrite=False):
    require_tcp_calibration(job['tool_metadata'])
    dest=output/job['name']
    if not dest.resolve().is_relative_to(output):raise ValueError('Episode output escapes output root')
    if not overwrite and any((dest/f).exists() for f in FILES):raise FileExistsError(f'Existing episode outputs; use --overwrite: {dest}')
    schema=pl.read_parquet_schema(job['parquet']);required=['timestamp','frame_index','episode_index','observation.state']
    if not set(required)<=set(schema):raise ValueError('Missing required Parquet columns')
    has_wrench='observation.wrench' in schema
    df=pl.read_parquet(job['parquet'],columns=required+(['observation.wrench'] if has_wrench else []))
    info=json.loads(job['info'].read_text());conversion=json.loads(job['conversion'].read_text()) if job['conversion'] else None
    data=parse_measurements(df.to_dicts(),info,job['episode'],conversion,has_wrench)
    calibration=job['tool_metadata'];add_tcp(data,calibration['T_ee_tcp'])
    summary=dict(name=job['name'],episode=job['episode'],status='MEASURED' if np.isfinite(data['force_norm_N']).any() else 'POSE_ONLY_NO_FINITE_FORCE',**summarize(data),
        source_parquet=str(job['parquet']),source_parquet_sha256=hashlib.sha256(job['parquet'].read_bytes()).hexdigest(),
        source_info=str(job['info']),source_info_sha256=hashlib.sha256(job['info'].read_bytes()).hexdigest(),
        conversion=str(job['conversion']) if conversion else None,
        conversion_sha256=hashlib.sha256(job['conversion'].read_bytes()).hexdigest() if conversion else None,
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tool=calibration['tool'],tcp_calibration=calibration,
        tcp_valid_position_samples=int(np.isfinite(data['tcp_position_m']).all(axis=1).sum()),
        tcp_path_length_m=float(np.nansum(np.linalg.norm(np.diff(data['tcp_position_m'],axis=0),axis=1))),
        tcp_linear_speed=stats(data['tcp_linear_speed_m_s']),tcp_angular_speed=stats(data['tcp_angular_speed_rad_s']),
        ee_convention='observation.state measured EE in base_link; position meters, rotation-vector radians; user-confirmed',
        wrench_coordinate_frame='EE_ALIGNED_AXES_USER_CONFIRMED; recorded moment origin retained; no TCP torque transport' if calibration['wrench_frame']['axes']=='ee' else 'UNSPECIFIED_IN_RECORDING_METADATA; stored components are not rotated or transported to TCP',
        timestamp_provenance='Recorded episode timestamp; no claim of hardware synchronization',
        velocity_convention='Adjacent finite measurements divided by positive recorded dt, assigned to interval end; SO(3) relative rotation for angular speed',
        processing='All Parquet rows; missing values retained as NaN; no smoothing, interpolation, tare or gravity compensation applied')
    dest.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.analysis-',dir=dest) as tmp:
        stage=Path(tmp);np.savez_compressed(stage/'measurements.npz',**data);write_csv(stage/'measurements.csv',data)
        (stage/'robot_tool.json').write_text(json.dumps(calibration,indent=2,allow_nan=False)+'\n')
        (stage/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');plot(data,job['name'],stage);plot(data,job['name'],stage,'tcp')
        for f in FILES:(stage/f).replace(dest/f)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,nargs='+',required=True,help='Converted real root/scene, native assets root/recording, or episode Parquet')
    p.add_argument('--output-root',type=Path,required=True,help='Separate analysis output root; inputs are read-only')
    p.add_argument('--episodes',type=int,nargs='+',help='Optional episode indices; default all discovered episodes')
    p.add_argument('--tool-metadata',type=Path,default=Path(__file__).resolve().parents[1]/'configs/real_robot_tools.json',help='Explicit scene/recording-to-tool registry and calibrated T_ee_tcp transforms')
    p.add_argument('--overwrite',action='store_true',help='Replace only this script\'s known analysis files')
    p.add_argument('--dry-run',action='store_true',help='List episode sources and output paths without writing')
    args=p.parse_args(argv);out=args.output_root.expanduser().resolve()
    try:
        jobs=discover(args.input,args.episodes)
        registry_path=args.tool_metadata.expanduser().resolve();registry_bytes=registry_path.read_bytes();registry=json.loads(registry_bytes)
        for job in jobs:
            job['tool_metadata']=tool_for_job(job,registry)
            job['tool_metadata'].update(registry_path=str(registry_path),registry_sha256=hashlib.sha256(registry_bytes).hexdigest())
        for raw in args.input:
            raw=raw.expanduser().resolve()
            if out==raw or out.is_relative_to(raw) or raw.is_relative_to(out):raise ValueError('Analysis output must be separate from input paths')
        for job in jobs:
            recording=job['recording_root'].resolve()
            if out==recording or out.is_relative_to(recording) or recording.is_relative_to(out):raise ValueError('Output overlaps source recording')
            for source in [job['parquet'],job['info']]+([job['conversion']] if job['conversion'] else []):
                if source.resolve().is_relative_to(out) or out.is_relative_to(source.parent.resolve()):raise ValueError('Output overlaps source recording')
        if registry_path.is_relative_to(out):raise ValueError('Tool registry source must be outside the analysis output root')
        for name in ['summary.csv','run_summary.json','tool_registry.json','tool_assignments.csv']:
            if not (out/name).resolve().is_relative_to(out):raise ValueError('Batch output file escapes output root')
        if args.dry_run:
            for j in jobs:print(f'{j["parquet"]} -> {out/j["name"]}; tool={j["tool_metadata"]["tool"]}, local Z={j["tool_metadata"]["T_ee_tcp"][2][3]:.8f} m; calibration_valid={j["tool_metadata"].get("tcp_calibration_valid", True)}')
            print(f'{len(jobs)} episodes; no files written');return 0
        if (out/'summary.csv').exists() and not args.overwrite:raise FileExistsError('Existing batch summary; use --overwrite or a fresh output root')
        out.mkdir(parents=True,exist_ok=True)
        cache=out/'.matplotlib'
        if not cache.resolve().is_relative_to(out):raise ValueError('Matplotlib cache escapes output root')
        os.environ['MPLCONFIGDIR']=str(cache)
        (out/'tool_registry.json').write_bytes(registry_bytes)
        with (out/'tool_assignments.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['dataset','tool','episode','source_recording','local_z_offset_m','tcp_calibration_valid','mounting_id'])
            for j in jobs:
                m=j['tool_metadata'];writer.writerow([j['name'],m['tool'],m['episode'],m['source_recording'],m['T_ee_tcp'][2][3],m.get('tcp_calibration_valid',True),m.get('mounting_id','')])
        results=[];failures=[]
        for job in jobs:
            try:
                r=analyze(job,out,args.overwrite);results.append(r)
                print(f'{job["name"]}: {r["samples"]} rows, {r["duration_s"]:.2f}s; force max={r["force"]["max"]} N',flush=True)
            except Exception as error:
                failures.append(dict(name=job['name'],error=str(error)));print(f'FAILED {job["name"]}: {error}',file=sys.stderr,flush=True)
        with (out/'summary.csv').open('w',newline='') as f:
            names=['name','episode','tool','status','samples','duration_s','measured_sampling_hz','duplicate_timestamps','ee_valid_position_samples','observed_adjacent_path_length_m','tcp_valid_position_samples','tcp_path_length_m','force_max_N','force_rms_N','force_p95_N','torque_max_Nm','linear_speed_max_m_s','angular_speed_max_rad_s','tcp_linear_speed_max_m_s']
            writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
            for r in results:writer.writerow({**{k:r[k] for k in names if k in r},'force_max_N':r['force']['max'],'force_rms_N':r['force']['rms'],'force_p95_N':r['force']['p95'],
                'torque_max_Nm':r['torque']['max'],'linear_speed_max_m_s':r['linear_speed']['max'],'angular_speed_max_rad_s':r['angular_speed']['max'],'tcp_linear_speed_max_m_s':r['tcp_linear_speed']['max']})
        (out/'run_summary.json').write_text(json.dumps(dict(requested=len(jobs),completed=len(results),failures=failures),indent=2)+'\n')
        print(f'{len(results)}/{len(jobs)} episodes complete: {out}',flush=True);return int(bool(failures))
    except Exception as error:print(f'ERROR: {error}',file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
