#!/usr/bin/env python3
"""Add force-share percentages and recorded-origin torque analysis to completed debug runs."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from force_trajectory_decomposition import trajectory_tangent,decompose_force


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def force_shares(signed,perpendicular,norm,valid,force_epsilon_n=1e-8):
    if not np.isfinite(force_epsilon_n) or force_epsilon_n<=0:raise ValueError('Force epsilon must be finite and positive')
    good=np.asarray(valid,bool)&np.isfinite(norm)&(norm>=0)&np.isfinite(signed)&np.isfinite(perpendicular)
    outputs={key:np.full(len(norm),np.nan) for key in ['force_tangential_pct','force_tangential_signed_pct','force_perpendicular_pct']}
    eta=np.full(len(norm),np.nan);eta[good]=abs(signed[good])/(norm[good]+force_epsilon_n)
    outputs['force_tangential_pct'][good]=100*eta[good]
    outputs['force_tangential_signed_pct'][good]=100*signed[good]/(norm[good]+force_epsilon_n)
    outputs['force_perpendicular_pct'][good]=100*perpendicular[good]/(norm[good]+force_epsilon_n)
    return dict(force_percentage_valid=good,eta=eta,**outputs)


def torque_components(sensor,rotvec,mounting,tangent,tangent_valid):
    """Rotate a moment about its recorded origin; no wrench-origin translation."""
    sensor=np.asarray(sensor,float);ee=sensor@np.asarray(mounting).T;base=np.full_like(sensor,np.nan)
    good=np.isfinite(rotvec).all(axis=1)&np.isfinite(sensor).all(axis=1)
    if good.any():base[good]=Rotation.from_rotvec(rotvec[good]).apply(ee[good])
    valid=good&tangent_valid;scalar=np.full(len(sensor),np.nan)
    scalar[valid]=np.sum(base[valid]*tangent[valid],axis=1)
    parallel=scalar[:,None]*tangent;perpendicular=np.full_like(base,np.nan);perpendicular[valid]=base[valid]-parallel[valid]
    return dict(torque_sensor_Nm=sensor,torque_ee_Nm=ee,torque_base_Nm=base,torque_norm_Nm=np.linalg.norm(base,axis=1),
        torque_decomposition_valid=valid,torque_tangent_signed_Nm=scalar,torque_tangent_base_Nm=parallel,
        torque_perpendicular_base_Nm=perpendicular,torque_perpendicular_norm_Nm=np.linalg.norm(perpendicular,axis=1))


def temporal_mean(time,percentage,window,valid=None):
    """Trapezoidal time mean; clip to bounds and never bridge undefined samples/gaps."""
    t=np.asarray(time,float);y=np.asarray(percentage,float)
    if t.ndim!=1 or y.shape!=t.shape or not np.isfinite(t).all() or np.any(np.diff(t)<0):raise ValueError('Invalid timestamps/percentages')
    if window is None:
        return dict(interaction_start_s=None,interaction_end_s=None,interaction_duration_s=None,
            valid_interaction_duration_s=0.,interaction_coverage_pct=0.,interaction_temporal_mean_pct=None,
            valid_time_temporal_mean_pct=None,temporal_mean_status='NO_INTERACTION_WINDOW')
    start,end=map(float,window)
    if not np.isfinite([start,end]).all() or end<=start:raise ValueError('Interaction window must have positive duration')
    good=np.isfinite(y) if valid is None else np.isfinite(y)&np.asarray(valid,bool)
    if good.shape!=t.shape:raise ValueError('Validity mask must match timestamps')
    dt=np.diff(t);positive=dt[dt>0];gap=5*np.median(positive) if len(positive) else np.inf
    left=np.maximum(t[:-1],start);right=np.minimum(t[1:],end)
    pairs=good[:-1]&good[1:]&(dt>0)&(dt<=gap)&(right>left)
    durations=right[pairs]-left[pairs]
    slope=(y[1:][pairs]-y[:-1][pairs])/dt[pairs]
    yl=y[:-1][pairs]+slope*(left[pairs]-t[:-1][pairs]);yr=y[:-1][pairs]+slope*(right[pairs]-t[:-1][pairs])
    observed=float(durations.sum());integral=float(np.sum(.5*(yl+yr)*durations));duration=end-start
    complete=abs(observed-duration)<=max(1e-8,1e-8*duration)
    return dict(interaction_start_s=start,interaction_end_s=end,interaction_duration_s=duration,
        valid_interaction_duration_s=observed,interaction_coverage_pct=min(100.,100*observed/duration),
        interaction_temporal_mean_pct=integral/duration if complete else None,
        valid_time_temporal_mean_pct=integral/observed if observed>0 else None,
        temporal_mean_status='COMPLETE_INTERACTION_WINDOW' if complete else 'INCOMPLETE_VALID_TIME')


def save_csv(path,columns):
    keys=list(columns);length=len(columns[keys[0]])
    if any(len(value)!=length for value in columns.values()):raise ValueError('CSV column length mismatch')
    with path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(keys)
        for i in range(length):
            row=[columns[k][i] for k in keys]
            writer.writerow(['' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for v in row])


def vector_columns(columns,name,values):
    for i,axis in enumerate('xyz'):columns[name+'_'+axis]=values[:,i]


def plot_scene(path,time,labels,force_cases,torques,sensor,min_force,window=None):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(13,12),sharex=True,constrained_layout=True)
    for label,force,torque in zip(labels,force_cases,torques):
        pct=force['force_tangential_pct'].copy();pct[force['force_norm_N']<min_force]=np.nan
        axes[0].plot(time,pct,label=label,lw=.9)
        for i,axis in enumerate('xyz'):axes[2].plot(time,torque['torque_base_Nm'][:,i],label=label+' '+axis,lw=.8)
        axes[3].plot(time,torque['torque_tangent_signed_Nm'],label=label+' signed tangent',lw=.9)
        axes[3].plot(time,torque['torque_perpendicular_norm_Nm'],'--',label=label+' perpendicular norm',lw=.9)
    for i,axis in enumerate('xyz'):axes[1].plot(time,sensor[:,i],label='Sensor '+axis,lw=.8)
    axes[1].plot(time,np.linalg.norm(sensor,axis=1),color='black',alpha=.5,label='Norm',lw=.8)
    axes[0].set(ylabel='100 eta (%)',ylim=(0,100));axes[1].set_ylabel('Recorded torque (N m)')
    axes[2].set_ylabel('Base-axis torque (N m)');axes[3].set(ylabel='Torque spatial split (N m)',xlabel='Native Parquet timestamp (s)')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
        if window is not None:ax.axvspan(*window,color='orange',alpha=.12)
    fig.suptitle(path.parent.name+f'\nOrange: saved interaction window; time mean of absolute percentage. Recorded-origin moments; no TCP shift.',fontsize=10)
    fig.savefig(path,dpi=115);plt.close(fig)


def analyze_scene(root,measurement_root,name,records,min_force,interaction_root,torque_3d_images=False,force_epsilon_n=1e-8):
    folder=root/name;archive=folder/'force_decomposition.npz';source=measurement_root/name/'measurements.npz'
    digest=sha(source)
    with np.load(archive,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(source,allow_pickle=False) as z:raw={k:z[k].copy() for k in ['frame_index','timestamp_s','ee_position_m','ee_rotvec_rad','torque_Nm']}
    if digest!=records[0]['measurement_archive_sha256']:raise ValueError('Source measurements differ from force analysis: '+name)
    interval_path=interaction_root/name/'result.json';interaction=json.loads(interval_path.read_text())
    if interaction['scene']!=name or interaction['provenance']['tcp_archive_sha256']!=digest:raise ValueError('Interaction bounds refer to different measurements')
    interval=interaction['interval'];window=interval['selected_window_s']
    np.testing.assert_array_equal(raw['frame_index'],data['frame_index']);np.testing.assert_array_equal(raw['ee_rotvec_rad'],data['ee_rotvec_rad'])
    np.testing.assert_array_equal(raw['timestamp_s'],data['timestamp_s']);np.testing.assert_array_equal(raw['ee_position_m'],data['ee_position_m'])
    # Old archives used a smoothed TCP tangent. Always recompute the requested EE formula.
    data.update(trajectory_tangent(raw['timestamp_s'],raw['ee_position_m'],raw['frame_index']))
    labels=list(data['case_labels']);matrices=records[0]['R_ee_sensor']
    if len(matrices)!=len(labels):raise ValueError('Mounting case mismatch')
    force_cases=[];torques=[];summaries=[];n=len(raw['frame_index'])
    for c,(label,matrix) in enumerate(zip(labels,matrices)):
        force=decompose_force(data['force_base_N'][c],data,force_epsilon_n)
        force.update(force_shares(force['force_tangent_signed_N'],force['force_perpendicular_norm_N'],force['force_norm_N'],force['decomposition_valid'],force_epsilon_n))
        torque=torque_components(raw['torque_Nm'],raw['ee_rotvec_rad'],matrix,data['tangent'],data['tangent_valid'])
        force_cases.append(force);torques.append(torque)
        eligible=force['force_percentage_valid']&(force['force_norm_N']>=min_force)
        summaries.append(dict(scene=name,case=str(label),tcp_calibration_valid=records[0]['tcp_calibration_valid'],native_samples=n,min_force_N=min_force,force_epsilon_N=force_epsilon_n,
            interaction_source=interval.get('selected_interval_source'),interaction_status=interval.get('status'),
            **temporal_mean(data['timestamp_s'],force['force_tangential_pct'],window,eligible)))
    numerical={key:data[key] for key in ['frame_index','rgbd_frame_index','timestamp_s','ee_position_m','tangent','tangent_valid','tangent_status','speed_m_s']}
    inside=(data['timestamp_s']>=window[0])&(data['timestamp_s']<=window[1]) if window is not None else np.zeros(n,bool)
    numerical.update(case_labels=data['case_labels'],force_base_N=data['force_base_N'],torque_sensor_Nm=raw['torque_Nm'],interaction_sample=inside)
    for key in force_cases[0]:numerical[key]=np.array([case[key] for case in force_cases])
    for key in torques[0]:
        if key!='torque_sensor_Nm':numerical[key]=np.array([case[key] for case in torques])
    np.savez_compressed(folder/'wrench_analysis.npz',**numerical)
    columns={key:np.tile(data[key],len(labels)) for key in ['frame_index','rgbd_frame_index','timestamp_s','tangent_status','tangent_valid']}
    columns['case']=np.repeat(labels,n)
    for key,value in numerical.items():
        if key in columns or key in ['case_labels','torque_sensor_Nm','tangent','speed_m_s']:continue
        if value.shape==(len(labels),n):columns[key]=value.reshape(-1)
        elif value.shape==(len(labels),n,3):vector_columns(columns,key,value.reshape(-1,3))
    vector_columns(columns,'torque_sensor_Nm',np.tile(raw['torque_Nm'],(len(labels),1)))
    vector_columns(columns,'ee_position_m',np.tile(raw['ee_position_m'],(len(labels),1)))
    vector_columns(columns,'tangent',np.tile(data['tangent'],(len(labels),1)))
    columns['interaction_sample']=np.tile(inside,len(labels))
    save_csv(folder/'wrench_analysis.csv',columns)
    plot_scene(folder/'force_share_torque.png',data['timestamp_s'],labels,force_cases,torques,raw['torque_Nm'],min_force,window)
    if torque_3d_images:
        from torque_vector_frames import save_torque_frames
        save_torque_frames(folder/'torque_3d_frames',data['frame_index'],data['rgbd_frame_index'],data['timestamp_s'],torques,labels,records)
    import matplotlib.pyplot as plt
    lookup={int(frame):i for i,frame in enumerate(data['frame_index'])}
    for record in records:
        camera_folder=root/record['relative_folder'];indices=np.flatnonzero(data['rgbd_frame_index']>=0)
        with (camera_folder/'projected_forces.csv').open() as f:projected=list(csv.DictReader(f))
        indices_by_case=[]
        for c in range(len(labels)):
            angle=0 if c==0 else 90
            indices_by_case.append(np.array([lookup[int(row['native_frame'])] for row in projected if float(row['rotation_z_deg'])==angle],int))
        if any(not np.array_equal(indices_by_case[0],ids) for ids in indices_by_case[1:]):raise ValueError('Camera mounting cases use different frames')
        indices=indices_by_case[0]
        if len(indices)!=record['frames']:raise ValueError('Camera frame count mismatch')
        T=np.linalg.inv(np.array(record['camera_to_base']));cases=[]
        fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True)
        for label,torque in zip(labels,torques):
            camera=torque['torque_base_Nm'][indices]@T[:3,:3].T;cases.append(camera)
            for i,axis in enumerate('xyz'):ax.plot(data['timestamp_s'][indices],camera[:,i],label=label+' '+axis,lw=.8)
        ax.plot(data['timestamp_s'][indices],np.linalg.norm(cases[0],axis=1),color='black',alpha=.5,label='Norm',lw=.8)
        ax.set(title=name+' '+record['camera']+' — moment about recorded origin',xlabel='Native timestamp (s)',ylabel='Camera-axis torque (N m)');ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.savefig(camera_folder/'torque_camera.png',dpi=110);plt.close(fig)
        camera_columns=dict(frame_index=np.tile(data['frame_index'][indices],len(labels)),rgbd_frame_index=np.tile(data['rgbd_frame_index'][indices],len(labels)),
            timestamp_s=np.tile(data['timestamp_s'][indices],len(labels)),case=np.repeat(labels,len(indices)))
        for key in ['force_tangential_pct','force_tangential_signed_pct','force_perpendicular_pct']:
            camera_columns[key]=np.concatenate([force[key][indices] for force in force_cases])
        for key in ['torque_tangent_signed_Nm','torque_perpendicular_norm_Nm','torque_norm_Nm']:
            camera_columns[key]=np.concatenate([torque[key][indices] for torque in torques])
        for key in ['torque_ee_Nm','torque_base_Nm']:
            vector_columns(camera_columns,key,np.concatenate([torque[key][indices] for torque in torques]))
        vector_columns(camera_columns,'torque_camera_Nm',np.concatenate(cases));vector_columns(camera_columns,'torque_sensor_Nm',np.tile(raw['torque_Nm'][indices],(len(labels),1)))
        save_csv(camera_folder/'torque_camera.csv',camera_columns)
    meta=dict(scene=name,summary=summaries,percentage_definition='100 * eta; eta = abs(F dot t)/(norm(F)+epsilon)',force_epsilon_N=force_epsilon_n,
        tangent_definition='t[i]=(p_EE[i+1]-p_EE[i-1])/norm(p_EE[i+1]-p_EE[i-1]); no smoothing or speed threshold; endpoints undefined; cached TCP tangent is not used',
        torque_origin='Recorded moment origin is unspecified; no reference-point shift or TCP torque transport',
        torque_transform='tau_base = R_base_EE @ R_EE_sensor @ tau_sensor; tau_camera = R_camera_base @ tau_base',
        torque_tangent_interpretation='Spatial moment component along centered observed EE direction, not an identified revolute-joint torque',
        min_force_summary_N=min_force,scope='Temporal mean of absolute percentage over saved interaction window; no force-magnitude weighting',
        interaction=interval,interaction_result_sha256=sha(interval_path),
        source_measurement_sha256=digest,source_force_decomposition_sha256=sha(archive),source_script_sha256=sha(Path(__file__)),
        mounting_cases=matrices,cameras=[r['camera'] for r in records])
    (folder/'wrench_analysis.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    assert sha(source)==digest
    return summaries,force_cases,torques


def update_analysis(root,measurements,min_force=0.,interaction_root=None,torque_3d_images=False,force_epsilon_n=1e-8):
    import matplotlib
    if interaction_root is None:raise ValueError('An interaction-root is required; whole-episode statistics are not used')
    interaction_root=Path(interaction_root).resolve()
    os.environ['MPLCONFIGDIR']=str(root/'.matplotlib');matplotlib.use('Agg')
    records=json.loads((root/'summary.json').read_text());grouped={}
    for record in records:
        if record['status']!='FAILED':grouped.setdefault(record['scene'],[]).append(record)
    summaries=[];failures=[]
    for number,(name,cameras) in enumerate(sorted(grouped.items()),1):
        try:
            rows,_,_=analyze_scene(root,measurements,name,cameras,min_force,interaction_root,torque_3d_images,force_epsilon_n)
            summaries.extend(rows)
            print(f'{number}/{len(grouped)} {name}: interaction time mean and torque analysis saved for {len(cameras)} cameras',flush=True)
        except Exception as error:
            failures.append(dict(scene=name,error=str(error)));print('FAILED '+name+': '+str(error),flush=True)
    if summaries:save_csv(root/'force_torque_summary.csv',{key:[r[key] for r in summaries] for key in summaries[0]})
    result=dict(scenes=len(grouped),completed_scenes=len({r['scene'] for r in summaries}),camera_runs=sum(len(v) for v in grouped.values()),failed=failures,
        per_dataset=summaries,min_force_N=min_force,source_script_sha256=sha(Path(__file__)),interaction_root=str(interaction_root),
        metric='100/(b-a) * integral_a^b abs(F dot t)/(norm(F)+epsilon) dt; trapezoidal integration at actual native timestamps',
        tangent_source='Centered observed EE difference, recomputed from raw EE positions; no cached tangent, smoothing or speed cutoff',force_epsilon_N=force_epsilon_n,
        scope='Saved interaction windows only; no force weighting, no whole-episode averaging',
        undefined_policy='Full-window mean is null unless full valid-time coverage. Separate valid-time mean and coverage are reported; no zero fill or bridging undefined samples.',
        torque_reference='Recorded origin unchanged; zero in torque images is component space, not TCP/base spatial origin',
        torque_3d_images=torque_3d_images)
    (root/'force_torque_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>Interaction time mean and torque vectors</title><style>body{font:15px sans-serif;margin:24px}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}a{color:#1565c0}table{border-collapse:collapse}p{max-width:1100px}</style>',
        '<h1>Interaction-window temporal mean of absolute tangential percentage</h1>',
        '<p>Requested metric: 100/(b-a) integral abs(F dot t)/(norm(F)+epsilon) dt, within the saved interaction bounds only. No force weighting. Default: all finite-force samples; zero force gives eta=0. Zero displacement, unavailable neighbors and invalid data have undefined tangents. A full-window mean is shown only at 100% valid-time coverage; the conditional valid-time mean and coverage are shown separately.</p>',
        '<p>Bounds retain their original force/visual agreement flags; they are not independently verified contact ground truth. Torque PNGs show vectors in base and every camera basis in one separate image for each native frame. Moment origin remains unchanged.</p>',
        '<p><a href="force_torque_summary.csv">CSV</a> | <a href="force_torque_summary.json">JSON</a> | <a href="index.html">All camera debug</a></p>',
        '<table><tr><th>Dataset (case)</th><th>Window start s</th><th>End s</th><th>Temporal mean %</th><th>Valid-time mean %</th><th>Coverage %</th><th>Window quality</th><th>3D torque frames</th></tr>']
    def fmt(value):return 'undefined' if value is None else f'{value:.3f}'
    for row in summaries:
        name=html.escape(row['scene'],quote=True)
        label=html.escape(row['scene']+' ('+row['case']+')'+(' [UNVERIFIED MOUNT]' if not row['tcp_calibration_valid'] else ''))
        gallery=f'<a href="{name}/torque_3d_frames/index.html">Open frames</a>' if (root/row['scene']/'torque_3d_frames/index.html').is_file() else 'not generated'
        values=[row[k] for k in ['interaction_start_s','interaction_end_s','interaction_temporal_mean_pct','valid_time_temporal_mean_pct','interaction_coverage_pct']]
        page.append('<tr><td>'+label+'</td>'+''.join('<td>'+fmt(v)+'</td>' for v in values)+'<td>'+html.escape(str(row['interaction_status']))+'</td><td>'+gallery+'</td></tr>')
    page+=['</table><p>The temporal mean integrates 100 eta using the centered observed EE difference, without smoothing or a speed threshold. Epsilon is the configured positive force-denominator regularizer. Missing-neighbor and zero-displacement periods are reported through coverage.</p>']
    (root/'force_torque_summary.html').write_text('\n'.join(page)+'\n')
    report=['# Interaction-window time mean and per-frame 3D torque vectors','',
        'The previous whole-episode mean, median and force-weighted percentage have been replaced.',
        '`mean = 100/(b-a) * integral_a^b abs(F dot t)/(norm(F)+epsilon) dt`','',
        f'Completed scenes: {result["completed_scenes"]}/{result["scenes"]}; failed scenes: {len(failures)}.',
        'Use force_torque_summary.html / CSV / JSON for each dataset and mounting hypothesis. The bounds are read from the selected interaction window, with original source/quality flags retained.',
        'Full-window means require complete valid-time coverage. Otherwise the requested mean is undefined; the separately labeled valid-time mean and coverage show available evidence. No stationary/zero-force sample is zero-filled, and undefined gaps are not bridged.',
        'Centered observed EE differences are recomputed from raw EE positions; no cached tangent, smoothing, or speed cutoff is used. Positive epsilon regularizes the force denominator; zero force yields eta=0 when the tangent exists. No force weighting or whole-episode fallback is used.','',
        'Each torque_3d_frames/frame_<native>.png is a separate torque-only 3D image, containing base and every camera-axis panel. Frame index.html provides a slider. Scales are fixed in N m across each scene. The displayed origin is component-space zero; the recorded physical moment reference is unchanged. No TCP moment-arm shift is applied.',
        f'Per-frame 3D image generation enabled: {torque_3d_images}.',
        'Original measurements, force-decomposition files and force videos are preserved. Historical validation artifacts describe their original generation; the new temporal integration is covered by the updated implementation tests and reported coverage.']
    (root/'FORCE_SHARE_TORQUE_REPORT.md').write_text('\n'.join(report)+'\n')
    from debug_drawer_forces_camera import write_index
    write_index(root,records,True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis-root',type=Path,required=True);p.add_argument('--measurements-root',type=Path,required=True)
    p.add_argument('--interaction-root',type=Path,required=True,help='Per-scene result.json containing selected interaction timestamps')
    p.add_argument('--min-force-n',type=float,default=0.,help='Optional explicit threshold; default includes all finite forces')
    p.add_argument('--force-epsilon-n',type=float,default=1e-8,help='Positive denominator regularizer in N, default 1e-8')
    p.add_argument('--torque-3d-images',action='store_true',help='Write one torque-only 3D PNG per native frame, showing base and all camera axes')
    args=p.parse_args(argv)
    if not np.isfinite(args.min_force_n) or args.min_force_n<0:p.error('Force threshold must be finite and nonnegative')
    if not np.isfinite(args.force_epsilon_n) or args.force_epsilon_n<=0:p.error('Force epsilon must be finite and positive')
    root=args.analysis_root.expanduser().resolve();measurements=args.measurements_root.expanduser().resolve();interaction=args.interaction_root.expanduser().resolve()
    for source in [measurements,interaction]:
        if root==source or root.is_relative_to(source) or source.is_relative_to(root):p.error('Analysis must be separate from all inputs')
    plan=json.loads((root/'run.json').read_text())
    if not plan['completed'] or not plan['decompose']:p.error('Complete the force-decomposition run first')
    result=update_analysis(root,measurements,args.min_force_n,interaction,args.torque_3d_images,args.force_epsilon_n)
    print(json.dumps({k:v for k,v in result.items() if k!='per_dataset'},indent=2));return int(bool(result['failed']))


if __name__=='__main__':raise SystemExit(main())
