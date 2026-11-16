#!/usr/bin/env python3
"""Isolated CUDA process using unchanged v6 setup, hooks and rendering kernels."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess

sys.dont_write_bytecode=True
import numpy as np


def write_json(path,value):
    def sanitize(x):
        if isinstance(x,dict):return {str(k):sanitize(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [sanitize(v) for v in x]
        if isinstance(x,np.ndarray):return sanitize(x.tolist())
        if isinstance(x,np.generic):return sanitize(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    Path(path).write_text(json.dumps(sanitize(value),indent=2,allow_nan=False)+'\n')


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def endpoint_ply(endpoint):
    return max((Path(endpoint)/'point_cloud').glob('iteration_*/point_cloud.ply'),key=lambda p:int(p.parent.name.removeprefix('iteration_')))


def parity_qualification(off_floor,on_difference,same_accept,saved_prefix):
    # A large nondeterministic floor cannot license a large observer effect.
    tolerance=dict(mu=1e-6,v=1e-5,omega=1e-5,R=1e-6,P=1e-5)
    repeatable=all(off_floor[k]<=tolerance[k] for k in tolerance)
    saved_ok=bool(saved_prefix) and all(v is not None and v<=tolerance['mu'] for v in saved_prefix)
    passed=repeatable and same_accept and saved_ok and all(on_difference[k]<=tolerance[k] for k in tolerance)
    return dict(status='PASS' if passed else 'FAIL',off_off_max_abs=off_floor,off_on_max_abs=on_difference,
        same_accept_sequence=same_accept,saved_run_prefix_matches=saved_ok,saved_run_prefix_differences_m=saved_prefix,
        tolerance=tolerance,baseline_repeatable=repeatable)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True,type=Path);parser.add_argument('--code',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);parser.add_argument('--static-labels',type=Path);args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);repo=args.code.resolve();sys.path.insert(0,str(repo))
    import torch
    from tools import stage_signature as ss
    original=json.loads(args.manifest.read_text());s=original['signature'];cfg=s['config']
    if s['observation'].get('object_mask_constraint','off')!='off' or cfg.get('P1FR','off')!='off':
        write_json(out/'status.json',dict(status='NOT_RUN',reason='UNSUPPORTED_SAVED_MEASUREMENT_VARIANT'));return 0
    current_code=ss._code_version();compatibility=dict(saved=s['code'],current=current_code,status='EXACT_SIGNATURE')
    if current_code!=s['code']:
        diff=subprocess.run(['git','diff','--diff-filter=MRTD',s['code']['commit'],'--','*.py','*.sh','*.cu','*.cpp','*.h','*.cuh'],cwd=repo,capture_output=True,text=True)
        equivalent=diff.returncode==0 and hashlib.sha256(diff.stdout.strip().encode()).hexdigest()==s['code'].get('uncommitted_diff_sha256')
        if not equivalent:
            write_json(out/'status.json',dict(status='NOT_RUN',reason='SAVED_RUN_CODE_SIGNATURE_UNAVAILABLE',saved=s['code'],current=current_code));return 0
        compatibility.update(status='COMMITTED_TRACKED_PATCH_MATCH',limitation='Original signature omitted then-untracked helpers; saved-state replay qualification is mandatory')
    write_json(out/'code_compatibility.json',compatibility)
    endpoint=Path(s['endpoint']['model_dir']);ply=endpoint_ply(endpoint)
    if digest(ply)!=s['endpoint']['ply_sha256']:raise ValueError('Endpoint bytes changed')
    sensor_path=Path(json.loads((args.manifest.parent/'sensor_manifest.json').read_text())['sensor_model_path'])
    if digest(sensor_path)!=cfg['SENSOR_SHA']:raise ValueError('Sensor model bytes changed')
    p0=endpoint/'geometry_information.npz'
    if cfg.get('BOOTP0')=='1' and digest(p0)!=cfg['P0_SHA']:raise ValueError('Bootstrap prior bytes changed')
    source=Path(s['source']['dir']);cams=s['source']['cameras']
    for folder,ext,key in [('rgb',ss.EXT,'rgb_content_sha256'),('depth_npy',('.npy',),'depth_content_sha256')]:
        if ss._content_digest(source/folder,cams,ext,s['source']['frames'],s['source']['frame_start'])!=s['source'][key]:raise ValueError(f'Original phase input bytes changed: {folder}')
    if ss._content_hash(Path(s['observation']['mask_root']),cams)!=s['observation']['mask_content_sha256']:raise ValueError('Original phase mask bytes changed')
    from tools.audit_calibrated_phase1 import setup_window
    from tools.kpart_pipeline import bootstrap_positions
    from estimator import wrong_camera_audit as wa
    from estimator.sparse_iekf import measurement_operators,evaluate_true_map_energy,group_precond
    from estimator.renderer_bridge import embed_pose_to_full_state
    from estimator.state import boxplus
    from estimator.process import process_predict
    from estimator.covariance import covariance_predict
    a=argparse.Namespace(model=str(endpoint),source=s['source']['dir'],cameras=','.join(s['source']['cameras']),direction=s['direction'],
        motion_start_frame=int(cfg['MSTART']),motion_end_frame=int(cfg['MEND']),frame_timestamps=cfg['TSTAMPS'],sensor_model=str(sensor_path),sensor_policy=cfg['SPOL'],
        rgb_model_sigma=float(cfg['RGBMS']),depth_model_sigma=float(cfg['DEPMS']),velocity_tau=float(cfg.get('VTAU',0)),
        bootstrap_iters=int(cfg['BOOT']),iekf_supernode_size=int(cfg['IEKF_SN']),phase1_partition=cfg.get('P1PART','canonical'),phase1_covariance_transport=cfg.get('P1TRANS')=='1',
        p0_geometry=str(p0) if cfg.get('BOOTP0')=='1' else None,p0_velocity_sigma=[float(x) for x in cfg['P0V'].split()],
        observation_mode=s['observation']['mode'],rgb_background_distribution=s['observation'].get('rgb_background_distribution','uniform01'),rgb_background_constant=s['observation'].get('rgb_background_constant'),
        object_mask_constraint='off',mask_source=s['observation']['mask_root'],mask_subdir='mask',first_return_mode='off',first_return_prior=None)
    a.alpha_precision=cfg.get('P1ALPHA','coupled');a.alpha_sd=float(cfg['P1ALPHASD']) if cfg.get('P1ALPHASD') else None;a.rgb_support=cfg.get('P1RGBSUP','full')
    if cfg.get('P1DET')=='1':
        from estimator.deterministic_ops import enable
        enable()
    truth,cameras,window,config,ns,frame_ids=setup_window(a)
    conversion=json.loads((source/'metadata/real_conversion.json').read_text());raw_mapping=dict(zip(conversion['output_frames'],conversion['source_frames']))
    # Preserve every measured frame in the original motion interval.
    snapshots=[];forces=[];qualifications=[];current={'ti':None,'fid':None};pending_traces={};original_frame_count=len(window)
    static=np.load(args.static_labels)['conditional_static'] if args.static_labels else np.zeros(truth.num_gaussians,bool)
    def array(t):return t.detach().float().cpu().numpy()
    prefix_replays=[]
    for _ in range(2):
        capture=[]
        def collect(**kw):capture.append({key:array(value) for key,value in dict(mu=kw['state'].mu,v=kw['state'].v,omega=kw['state'].omega,R=kw['state'].R,P=kw['P_post']).items()}|{'accept_sequence':[r['event'] for r in kw['diag'].rows if r.get('event') in {'accepted','rejected'}]})
        baseline=argparse.Namespace(**vars(ns));baseline.phase1_audit=collect;baseline.phase1_iteration_observer=None
        if baseline.masked_observation is not None:baseline.masked_observation=dataclasses.replace(baseline.masked_observation,report=[])
        torch.manual_seed(0);p,i=bootstrap_positions(truth,cameras,window[:3],config,None,baseline,verbose=False)
        prefix_replays.append(capture);del p,i
    off_floor={key:max((float(np.max(np.abs(a[key]-b[key]))) for a,b in zip(*prefix_replays)),default=0.) for key in ['mu','v','omega','R','P']};on_prefix=[]
    baseline_repeatable=parity_qualification(off_floor,off_floor,True,[0.])['baseline_repeatable']
    if not baseline_repeatable:
        # Record actual initial forces, but do not replay a long trajectory whose
        # counterfactual attribution already fails the no-diagnostics control.
        window=window[:3]
    selected={1,len(window)-1}
    def frame_audit(**kw):
        state=kw['state'];prior=kw['prior_state'];dt=kw['dt'];cfg_=kw['cfg'];pred=process_predict(prior,dt,tau=cfg_.velocity_tau)
        update=array(state.mu-pred.mu);prop=array(pred.mu-prior.mu);velocity=array(state.v-pred.v)
        row=dict(processing_index=kw['ti'],frame=kw['fid'],dt_s=float(dt),median_measurement_displacement_m=float(np.median(np.linalg.norm(update,axis=1))),
            median_process_displacement_m=float(np.median(np.linalg.norm(prop,axis=1))),median_velocity_update_mps=float(np.median(np.linalg.norm(velocity,axis=1))),
            actual_solver_rows=[{k:v for k,v in r.items() if isinstance(v,(int,float,str,bool)) or v is None} for r in kw['diag'].rows],
            static_truth='UNVERIFIED_AUTOMATIC_STATIC_PATCH_EVIDENCE_SEPARATE',physical_sensor_bias='UNVERIFIED')
        row.update(conditional_static_gaussians=int(static.sum()),conditional_static_measurement_median_m=float(np.median(np.linalg.norm(update[static],axis=1))) if static.any() else None,
            conditional_static_process_median_m=float(np.median(np.linalg.norm(prop[static],axis=1))) if static.any() else None,
            conditional_static_total_from_canonical_median_m=float(np.median(np.linalg.norm(array(state.mu-truth.mu)[static],axis=1))) if static.any() else None)
        fid=kw['fid'];frame=int(fid.get('original_frame_index',fid.get('dataset_frame_index'))) if isinstance(fid,dict) else int(fid)
        saved=args.manifest.parent/'phase1_centers'/f'centers_frame_{frame:04d}.npy'
        row.update(processed_frame=frame,source_frame=raw_mapping.get(frame),saved_run_position_max_abs_difference_m=float(np.max(np.abs(array(state.mu)-np.load(saved)))) if saved.is_file() else None)
        if kw['ti']<3:on_prefix.append({key:array(value) for key,value in dict(mu=state.mu,v=state.v,omega=state.omega,R=state.R,P=kw['P_post']).items()}|{'accept_sequence':[r['event'] for r in kw['diag'].rows if r.get('event') in {'accepted','rejected'}]})
        snapshots.append(row)
        if kw['ti'] in selected:
            np.savez_compressed(out/f'actual_states_{kw["ti"]:04d}.npz',mu_prior=array(prior.mu),mu_pred=array(pred.mu),mu_post=array(state.mu),v_prior=array(prior.v),v_pred=array(pred.v),v_post=array(state.v),
                R_prior=array(prior.R),R_pred=array(pred.R),R_post=array(state.R),omega_prior=array(prior.omega),omega_pred=array(pred.omega),omega_post=array(state.omega),
                P_prev=array(kw['P_prev']),P_post=array(kw['P_post']),measurement_update=update,process_increment=prop)
            frozen=pending_traces.pop(kw['ti'],None)
            if frozen:
                lin0,obs0,cfg0=frozen;length=np.linalg.norm(update,axis=1);candidate=np.flatnonzero(static)
                ids=list(candidate[np.argsort(length[candidate])[-10:]]) if len(candidate) else []
                ids+=list(candidate[np.argsort(length[candidate])[:3]]) if len(candidate) else []
                ids+=list(np.argsort(length)[-5:])
                from tools.audit_phase1_wrong_camera import make_sigma_fn
                sigma=make_sigma_fn();traces=wa.contributor_traces(lin0,obs0,cfg0,ids,list(config.camera_names),lambda c,D,v:sigma(c,D,v,cfg0))
                write_json(out/f'frozen_{kw["ti"]:04d}'/'actual_composited_contributors.json',traces)
                write_json(out/f'frozen_{kw["ti"]:04d}'/'selected_gaussians.json',dict(selected_ids=[int(g) for g in sorted(set(ids))],
                    selection='largest actual conditional-static measurement corrections, static stable controls, largest unlabelled controls',static_truth='UNVERIFIED_AUTOMATIC'))
    class Observer:
        def begin_frame(self,ti,fid):current.update(ti=ti,fid=fid)
        @torch.no_grad()
        def __call__(self,**kw):
            if kw['iteration']!=0:return
            ti=current['ti'];lin=kw['lin'];terms=kw['terms'];bc=wa.camera_channel_rhs(lin,terms);names=list(config.camera_names)
            for (ci,ch),f in bc.items():
                norms=f[:,:3].norm(dim=1);forces.append(dict(processing_index=ti,frame=current['fid'],camera=names[ci],channel=ch,gaussians=len(norms),
                    nonzero_force_gaussians=int((norms>0).sum()),sum_force_norm=float(norms.sum()),median_force_norm=float(norms.median()),
                    conditional_static_gaussians=int(static.sum()),conditional_static_force_sum=float(norms[torch.as_tensor(static,device=norms.device)].sum()) if static.any() else None,
                    direct_force_is_not_coupled_influence=True,depth_bias='UNVERIFIED'))
            if ti not in selected:return
            folder=out/f'frozen_{ti:04d}';folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/'forces_and_prior.npz',**{f'{names[c]}_{ch}':array(v) for (c,ch),v in bc.items()},Lambda_prior=array(kw['Lambda_prior']),P_pred=array(kw['P_prior']),rhs=array(kw['b']),mu_linearization=array(kw['x'].mu),R_linearization=array(kw['x'].R),mu_pred=array(kw['x_prior'].mu),supernode_id=array(kw['grouped'].part.supernode_id) if kw['grouped'] is not None else np.arange(truth.num_gaussians))
            checks,_=wa.additivity_and_symmetry(lin,terms,kw['rhs_pose'],kw['meas_normal'],n_probe=2)
            write_json(folder/'operator_checks.json',checks)
            fisher_arrays={}
            for ci,(view,t) in enumerate(zip(lin.views,terms)):
                np.savez_compressed(folder/f'{names[ci]}_pixel_terms.npz',rendered_normalized_Z=array(view.depth),observed_Z=array(view.depth+t.r_d),
                    depth_residual=array(t.r_d),rgb_residual=array(t.r_rgb),rendered_alpha=array(view.alpha),
                    depth_effective_precision=array(t.p_d),rgb_effective_precision=array(t.p_rgb),
                    alpha_precision=array(t.kappa) if t.kappa is not None else np.zeros_like(array(t.r_d)))
            for c,t in enumerate(terms):
                for ch in wa.CHANNELS:
                    prec=[(torch.zeros_like(z.p_rgb),torch.zeros_like(z.p_d)) for z in terms]
                    if ch=='rgb':prec[c]=(t.p_rgb,torch.zeros_like(t.p_d))
                    elif ch=='depth':prec[c]=(torch.zeros_like(t.p_rgb),t.p_d)
                    F=lin.build_block_fisher(prec)
                    if ch=='alpha' and t.kappa is not None:
                        ka=[torch.zeros_like(z.r_d) for z in terms];ka[c]=t.kappa;F=F+lin.build_alpha_fisher(ka)
                    fisher_arrays[f'{names[c]}_{ch}']=array(F)
            total_F=sum(fisher_arrays.values());ev,evec=np.linalg.eigh(.5*(total_F+total_F.transpose(0,2,1)))
            np.savez_compressed(folder/'per_camera_channel_fisher.npz',**fisher_arrays,total_eigenvalues=ev,total_eigenvectors=evec)
            grouped=kw['grouped'];pre=None
            if grouped is not None:
                G=lambda v:torch.cat([grouped.expand(v[:,:6].float()),grouped.expand(v[:,6:].float())],1)
                GT=lambda v:torch.cat([grouped.reduce(v[:,:6].float().contiguous()),grouped.reduce(v[:,6:].float().contiguous())],1)
                pre=group_precond(kw['M_blocks'],grouped,G,GT)
            base=wa.FrozenSystem(lin,terms,kw['Lambda_prior'],kw['b'],kw['meas_normal'],float(kw['lm']),grouped,pre)
            frozen=wa.leave_one_out(base,len(names),identity_check=True,tol=1e-6,max_iters=800)
            arms={'all':dict(delta=frozen['delta_all'],solver=frozen['solver_all'])}
            for key,r in frozen['arms'].items():
                index=int(key.split('cam')[1].split('_')[0]);physical=key.replace('cam'+str(index),names[index])
                arms[physical]=dict(delta=frozen['delta_all']-r['influence'],solver=r['solver'])
            # Additional boundary/layer and declared-model-variance arms on the
            # same cached render and prior. Production terms are never modified.
            for label in ['no_depth_boundary_mixed_layers','double_declared_depth_model_sd']:
                new=[]
                if label=='double_declared_depth_model_sd':
                    from estimator.sparse_iekf import _frame_terms
                    new=_frame_terms(lin.views,kw['rgbd_gt'],dataclasses.replace(kw['cfg'],depth_model_sigma=2*kw['cfg'].depth_model_sigma),kw['visibility'],kw['observation_model'])
                else:
                    import cv2
                    for t in terms:
                        D=array(t.r_d+lin.views[len(new)].depth);valid=np.isfinite(D)&(D>0)
                        mask=(array(t.p_d)>0).astype(np.uint8);edge=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((7,7),np.uint8))>0
                        lo=cv2.erode(np.where(valid,D,np.inf),np.ones((3,3),np.uint8));hi=cv2.dilate(np.where(valid,D,-np.inf),np.ones((3,3),np.uint8));span=np.subtract(hi,lo,out=np.zeros_like(hi),where=np.isfinite(lo)&(hi>0))
                        keep=torch.as_tensor(~edge&(span<.02),device=t.p_d.device);new.append(dataclasses.replace(t,p_d=t.p_d*keep))
                rhs,normal,fisher=measurement_operators(lin,[t.as_prec4() for t in new],[(t.kappa,t.r_a) for t in new] if new[0].kappa is not None else None)
                b=kw['b']-embed_pose_to_full_state(kw['rhs_pose'])+embed_pose_to_full_state(rhs)
                system=wa.FrozenSystem(lin,new,kw['Lambda_prior'],b,normal,float(kw['lm']),grouped,pre);delta,_,info=system.solve(tol=1e-6,max_iters=800)
                arms[label]=dict(delta=delta,solver=info)
            before,_,_=evaluate_true_map_energy(kw['x'],kw['x_prior'],kw['Lambda_prior'],kw['rgbd_gt'],kw['cameras'],kw['cfg'],background=kw['background'],visibility=kw['visibility'],observation_model=kw['observation_model'])
            for label,r in arms.items():
                accepted=False;after=None;factor=None
                if r['solver']['converged']:
                    for k in range(12):
                        factor=.5**k;trial=boxplus(kw['x'],(r['delta']*factor).to(kw['x'].mu.dtype))
                        after,_,_=evaluate_true_map_energy(trial,kw['x_prior'],kw['Lambda_prior'],kw['rgbd_gt'],kw['cameras'],kw['cfg'],background=kw['background'],visibility=kw['visibility'],observation_model=kw['observation_model'])
                        if np.isfinite(after) and after<before:accepted=True;break
                record=dict(processing_index=ti,frame=current['fid'],arm=label,linear_solver=r['solver'],complete_original_nonlinear_energy_before=before,
                    complete_original_nonlinear_energy_after=after,nonlinear_accepted=accepted,line_search_factor=factor if accepted else None,
                    status='CONDITIONAL_CERTIFIED_FROZEN_RESPONSE' if r['solver']['converged'] and accepted else 'UNOBSERVABLE_UNCONVERGED_OR_NONLINEAR_REJECTED',
                    scope='common complete production objective acceptance; frozen counterfactual is not the original trust-region step',physical_depth_bias='UNVERIFIED')
                qualifications.append(record)
                np.savez_compressed(folder/(label+'_delta.npz'),delta=array(r['delta']))
            write_json(folder/'counterfactual_qualifications.json',qualifications)
            pending_traces[ti]=(lin,kw['rgbd_gt'],kw['cfg'])
    started=time.monotonic();ns.phase1_audit=frame_audit;ns.phase1_iteration_observer=Observer();torch.manual_seed(0)
    positions,info=bootstrap_positions(truth,cameras,window,config,None,ns,verbose=False)
    write_json(out/'first_false_motion_steps.json',snapshots);write_json(out/'per_camera_channel_forces.json',forces);write_json(out/'solver_qualifications.json',qualifications)
    on_difference={key:max((float(np.max(np.abs(a[key]-b[key]))) for a,b in zip(prefix_replays[0],on_prefix)),default=float('inf')) for key in off_floor}
    same_accept=all(a['accept_sequence']==b['accept_sequence'] for a,b in zip(prefix_replays[0],on_prefix))
    saved_prefix=[r['saved_run_position_max_abs_difference_m'] for r in snapshots if r['processing_index']<3]
    parity=parity_qualification(off_floor,on_difference,same_accept,saved_prefix);parity_ok=parity['status']=='PASS'
    write_json(out/'baseline_parity.json',dict(prefix_frames=3,**parity))
    usable=sum(r['status']=='CONDITIONAL_CERTIFIED_FROZEN_RESPONSE' for r in qualifications) if parity_ok else 0
    write_json(out/'status.json',dict(status='CONDITIONAL' if usable else 'UNOBSERVABLE',scene=s['scene'],measured_frames=len(window),frozen_case_indices=sorted(selected),
        qualified_counterfactuals=usable,elapsed_s=time.monotonic()-started,actual_renderer=True,production_code_changed=False,
        original_window_frames=original_frame_count,trajectory_scope='full original interval' if baseline_repeatable else 'first three frames only: no-diagnostics replay is not repeatable',
        physical_sensor_bias='UNVERIFIED',static_label='AUTOMATIC_CONDITIONAL_ONLY',parity='PASS' if parity_ok else 'FAIL',
        cameras=list(config.camera_names),camera_000_ablation='NOT_APPLICABLE_ALREADY_EXCLUDED' if 'cam_000' not in config.camera_names else 'TESTED',code_compatibility=compatibility))
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        import traceback
        traceback.print_exc();sys.exit(1)
