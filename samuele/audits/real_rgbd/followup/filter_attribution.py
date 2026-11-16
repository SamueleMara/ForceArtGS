"""E6 discovery/dispatch; never edits a production solver or saved run."""
from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np

from .common import atomic_json,csv_write,file_record,json_read,sha,stable


def automatic_static_labels(run,scene,out):
    """Fixed analysis labels from whole-window depth visibility, never training GT."""
    from .measurements import read_data
    s=json_read(run['stage_manifest'])['signature'];root=Path(run['stage_manifest']).parent
    canonical=root/'phase1_centers/canonical_xyz.npy'
    if not canonical.is_file():return None
    mu=np.load(canonical,allow_pickle=False);seen=np.zeros(len(mu),int);free=np.zeros(len(mu),int);foreground=np.zeros(len(mu),int)
    K=np.array(scene['K']);start=int(s['config']['MSTART']);end=int(s['config']['MEND'])
    for c in scene['cameras']:
        if c['name'] not in s['source']['cameras']:continue
        T=np.linalg.inv(np.array(c['pose_matrix_4x4']));p=mu@T[:3,:3].T+T[:3,3];uv=p@K.T;xy=np.rint(uv[:,:2]/np.maximum(uv[:,2:],1e-9)).astype(int)
        inside=(p[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<c['width'])&(xy[:,1]>=0)&(xy[:,1]<c['height']);ids=np.flatnonzero(inside)
        for frame in range(start,end+1):
            data,_=read_data(scene,c,frame)
            if data is None:continue
            d=data['depth'][xy[ids,1],xy[ids,0]];valid=np.isfinite(d)&(d>0);delta=d-p[ids,2]
            seen[ids]+=valid&(np.abs(delta)<.015);free[ids]+=valid&(delta>.015)
            if data['mask'] is not None:foreground[ids]+=data['mask'][xy[ids,1],xy[ids,0]]&valid&(np.abs(delta)<.015)
    static=(seen>=max(5,int((end-start+1)*.05)))&(free/(seen+free+1)<.05)&(foreground>0)
    path=out/'automatic_static_labels.npz';np.savez_compressed(path,conditional_static=static,surface_observations=seen,free_space_contradictions=free,foreground_observations=foreground)
    atomic_json(out/'automatic_static_label_provenance.json',dict(status='CONDITIONAL',gaussians=len(mu),static_candidates=int(static.sum()),
        source='saved canonical Gaussian centers and whole-window recorded depth/masks',visibility='behind measured foreground is occluded, not a contradiction; measured free space contradicts static presence',
        thresholds=dict(surface_agreement_m=.015,max_free_space_fraction=.05,min_observations=max(5,int((end-start+1)*.05))),
        physical_static_truth='UNVERIFIED',labels_used_for_training=False))
    return path


def certified_solve(A,b,tolerance=1e-10):
    x=np.linalg.solve(A,b);res=float(np.linalg.norm(A@x-b)/max(np.linalg.norm(b),1e-30))
    return x,dict(converged=res<=tolerance,true_relative_residual=res,tolerance=tolerance)


def summarize_run(rd):
    """Causal increments and true composited visibility; no physical-bias claim."""
    frame_path=rd/'first_false_motion_steps.json';force_path=rd/'per_camera_channel_forces.json'
    if not frame_path.is_file():return [],[],[]
    frames=json_read(frame_path);forces=json_read(force_path) if force_path.is_file() else [];cases=[]
    replay_qualified=json_read(rd/'status.json').get('parity')=='PASS'
    static=np.load(rd/'automatic_static_labels.npz')['conditional_static'] if (rd/'automatic_static_labels.npz').is_file() else None
    for state_path in sorted(rd.glob('actual_states_*.npz')):
        index=int(state_path.stem.split('_')[-1]);folder=rd/f'frozen_{index:04d}'
        if not (folder/'selected_gaussians.json').is_file():continue
        ids=json_read(folder/'selected_gaussians.json')['selected_ids'];state=np.load(state_path);linear=np.load(folder/'forces_and_prior.npz')
        qualifiers={r['arm']:r for r in json_read(folder/'counterfactual_qualifications.json') if r['processing_index']==index}
        traces=json_read(folder/'actual_composited_contributors.json');group=linear['supernode_id'].astype(int)
        channels=[k for k in linear.files if k.startswith('cam_')];total=sum(linear[k][:,:3] for k in channels);own=np.linalg.norm(total,axis=1)
        all_delta=np.load(folder/'all_delta.npz')['delta'];context=next((r for r in frames if r['processing_index']==index),{})
        for g in ids:
            update=float(np.linalg.norm(state['measurement_update'][g]));process=float(np.linalg.norm(state['process_increment'][g]));tags=[]
            if process>2*update and process>1e-5:tags.append('PROCESS_EXTRAPOLATION_CONDITIONAL')
            mates=(group==group[g])&(np.arange(len(group))!=g)
            if own[g]<1e-10 and update>1e-5 and np.any(own[mates]>0):tags.append('COUPLING_CONDITIONAL')
            for key in channels:
                camera,ch=key.rsplit('_',1);rows=[t for t in traces if t['gaussian_id']==g and t['camera']==camera]
                arm='minus_'+key;qualification=qualifiers.get(arm,{});influence=None
                delta=folder/(arm+'_delta.npz')
                if replay_qualified and delta.is_file() and qualification.get('status')=='CONDITIONAL_CERTIFIED_FROZEN_RESPONSE':influence=float(np.linalg.norm(all_delta[g,:3]-np.load(delta)['delta'][g,:3]))
                visible=[t for t in rows if t['q']>0];classes=sorted({t['class'] for t in visible})
                record=dict(processing_index=index,processed_frame=context.get('processed_frame'),source_frame=context.get('source_frame'),gaussian_id=g,supernode_id=int(group[g]),camera=camera,channel=ch,
                    conditional_static=bool(static[g]) if static is not None else None,own_force_norm=float(np.linalg.norm(linear[key][g,:3])),groupmate_total_force_norm=float(own[mates].sum()),
                    actual_measurement_displacement_m=update,actual_process_displacement_m=process,coupled_frozen_channel_influence_m=influence,
                    replay_parity='PASS' if replay_qualified else 'FAIL',solver_status=qualification.get('status','NOT_RUN'),true_solver_residual=(qualification.get('linear_solver') or {}).get('true_rel_residual'),
                    nonlinear_common_objective_accepted=qualification.get('nonlinear_accepted'),visible_contributor_records=len(visible),visible_pixels=len({(t['u'],t['v']) for t in visible}),
                    sum_transmittance_alpha=float(sum(t['q'] for t in visible)),layer_classes=classes,mechanisms=tags+['UNRESOLVED_CANONICAL_MODEL_VS_OBSERVATION'],
                    physical_sensor_bias='UNVERIFIED',source_view_owner='UNAVAILABLE_IN_RETAINED_ENDPOINT',static_truth='AUTOMATIC_CONDITIONAL_NOT_GT')
                cases.append(record)
    return frames,forces,cases


def run_phase1(out,results,code,python,scenes,resume=False):
    dest=out/'phase1';dest.mkdir(parents=True,exist_ok=True);selected={s['name'] for s in scenes};runs=[]
    if results and Path(results).is_dir():
        for p in sorted(Path(results).rglob('stage_manifest.json')):
            r=json_read(p);sig=r.get('signature',{})
            if r.get('status')=='complete' and sig.get('stage')=='phase1' and sig.get('scene') in selected:
                runs.append(dict(stage_manifest=str(p),scene=sig['scene'],direction=sig['direction'],run_id=stable(str(p))[:16]))
    csv_write(dest/'available_runs.csv',runs)
    if not runs:
        atomic_json(dest/'status.json',dict(status='NOT_RUN',reason='NO_COMPATIBLE_SAVED_PHASE1_RUNS',expected_scenes=len(scenes),available_runs=0));return dict(status='NOT_RUN',available_runs=0)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TORCH_EXTENSIONS_DIR=str(out/'.runtime/torch_extensions'),CUDA_CACHE_PATH=str(out/'.runtime/cuda'),MPLCONFIGDIR=str(out/'.runtime/matplotlib'))
    probe=subprocess.run([str(python),'-B','-c','import torch,json; print(json.dumps({"cuda":torch.cuda.is_available()}))'],capture_output=True,text=True,env=env)
    try:gpu=json.loads(probe.stdout.strip().splitlines()[-1])['cuda']
    except Exception:gpu=False
    if not gpu:
        rows=[dict(**r,status='NOT_RUN',reason='CUDA_RENDERER_UNAVAILABLE',detail=probe.stderr[-1000:]) for r in runs]
        csv_write(dest/'solver_qualifications.csv',rows);atomic_json(dest/'status.json',dict(status='NOT_RUN',available_runs=len(runs),reason='CUDA_RENDERER_UNAVAILABLE'))
        return dict(status='NOT_RUN',available_runs=len(runs),reason='CUDA_RENDERER_UNAVAILABLE')
    control=out/'synthetic_tests/cuda_actual_renderer.json';control.parent.mkdir(parents=True,exist_ok=True)
    control_signature=stable(dict(diagnostic_sha256=sha(Path(__file__).parent/'phase1_cuda_controls.py'),production_python_sha256={str(p.relative_to(code)):sha(p) for folder in ['estimator','tests'] for p in (Path(code)/folder).glob('*.py')}))
    control_receipt=control.with_suffix('.receipt.json')
    if not (resume and control.is_file() and control_receipt.is_file() and json_read(control_receipt)==dict(signature=control_signature,result_sha256=sha(control)) and json_read(control).get('status')=='PASS'):
        with (control.parent/'cuda_actual_renderer.log').open('w') as log:
            process=subprocess.run([str(python),'-B',str(Path(__file__).parent/'phase1_cuda_controls.py'),'--code',str(code),'--out',str(control)],stdout=log,stderr=subprocess.STDOUT,env=env)
        if process.returncode or not control.is_file() or json_read(control).get('status')!='PASS':
            return dict(status='FAIL',reason='ACTUAL_RENDERER_SYNTHETIC_CONTROLS_FAILED',log=str(control.parent/'cuda_actual_renderer.log'))
        atomic_json(control_receipt,dict(signature=control_signature,result_sha256=sha(control)))
    summaries=[];all_frames=[];all_forces=[];all_cases=[]
    gpu_code=stable({p.name:sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in {'filter_attribution.py','phase1_worker.py','phase1_cuda_controls.py','common.py'}})[:12]
    for run in runs:
        rd=dest/(run['run_id']+'_'+gpu_code);status=rd/'status.json'
        if resume and status.is_file() and (rd/'completion.json').is_file():
            completion=json_read(rd/'completion.json')
            if completion['source_manifest_sha256']!=sha(run['stage_manifest']):raise ValueError('Saved Phase1 source manifest changed')
            from .common import validate_records
            validate_records(completion['outputs']);summaries.append(json_read(status))
            f,c,g=summarize_run(rd);all_frames.extend(dict(run_id=run['run_id'],scene=run['scene'],**r) for r in f);all_forces.extend(dict(run_id=run['run_id'],scene=run['scene'],**r) for r in c);all_cases.extend(dict(run_id=run['run_id'],scene=run['scene'],**r) for r in g)
            continue
        rd.mkdir(parents=True,exist_ok=True)
        command=[str(python),'-B',str(Path(__file__).parent/'phase1_worker.py'),'--manifest',run['stage_manifest'],'--code',str(code),'--out',str(rd)]
        scene=next(s for s in scenes if s['name']==run['scene']);labels=automatic_static_labels(run,scene,rd)
        if labels:command+=['--static-labels',str(labels)]
        print(f'Phase1 actual-renderer audit: {run["scene"]} {run["run_id"]}',flush=True)
        with (rd/'execution.log').open('w') as log:process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        if status.is_file():result=json_read(status)
        else:result=dict(status='FAIL' if process.returncode else 'UNOBSERVABLE',reason='WORKER_DID_NOT_PRODUCE_QUALIFIED_RESULT',returncode=process.returncode)
        summaries.append({**run,**result,'diagnostic_code_id':gpu_code,'diagnostic_dir':str(rd)});atomic_json(rd/'status.json',summaries[-1])
        f,c,g=summarize_run(rd);all_frames.extend(dict(run_id=run['run_id'],scene=run['scene'],**r) for r in f);all_forces.extend(dict(run_id=run['run_id'],scene=run['scene'],**r) for r in c);all_cases.extend(dict(run_id=run['run_id'],scene=run['scene'],**r) for r in g)
        if result.get('status') in {'CONDITIONAL','UNOBSERVABLE','NOT_RUN'}:
            atomic_json(rd/'completion.json',dict(source_manifest_sha256=sha(run['stage_manifest']),outputs=[file_record(p) for p in rd.rglob('*') if p.is_file() and p.name!='completion.json']))
    csv_write(dest/'solver_qualifications.csv',summaries)
    csv_write(dest/'first_false_motion_steps.csv',all_frames);csv_write(dest/'per_camera_channel_forces.csv',all_forces);csv_write(dest/'per_gaussian_attribution.csv',all_cases)
    if all_frames:
        import matplotlib.pyplot as plt
        p=dest/'residual_and_contribution_maps';p.mkdir(exist_ok=True)
        for run in runs:
            rows=[r for r in all_frames if r['run_id']==run['run_id']]
            if not rows:continue
            fig,ax=plt.subplots(figsize=(10,4));x=[r['processed_frame'] for r in rows]
            for key,label in [('conditional_static_measurement_median_m','measurement correction'),('conditional_static_process_median_m','process extrapolation'),('conditional_static_total_from_canonical_median_m','total from canonical')]:
                ax.plot(x,[1000*r[key] if r.get(key) is not None else np.nan for r in rows],label=label)
            ax.set(xlabel='processed dataset frame',ylabel='conditional-static median (mm)',title=run['scene']+' '+run['run_id']);ax.legend();fig.tight_layout();fig.savefig(p/(run['run_id']+'_onset.png'),dpi=110);plt.close(fig)
    qualified=sum(r.get('status')=='CONDITIONAL' for r in summaries)
    result=dict(status='CONDITIONAL' if qualified else 'UNOBSERVABLE',available_runs=len(runs),qualified_runs=qualified,
        scope='saved compatible Phase1 runs only; actual renderer, conditional static labels; not 118 trained runs',physical_depth_bias='UNVERIFIED')
    atomic_json(dest/'status.json',result);return result
