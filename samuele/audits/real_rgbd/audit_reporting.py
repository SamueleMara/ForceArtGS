"""Streaming aggregation, conditional held-out diagnostics, and honest conclusions."""
from collections import defaultdict
from itertools import groupby
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from scipy.optimize import lsq_linear

from audit_common import atomic_json,block_interval,block_length,cluster_interval,dumps,stable,table


def iter_rows(paths):
    for path in paths:
        for text in pl.read_parquet(path,columns=['metrics_json']).get_column('metrics_json'):
            yield json.loads(text)


def write_csv(path,rows):
    columns=sorted({k for row in rows for k in row}) if rows else ['status']
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({k:dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def diagnostic(out,scene,camera,frame,data,context,maps,pair_maps=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,5,figsize=(20,8));rgb=cv2.cvtColor(data['rgb'],cv2.COLOR_BGR2RGB)
    items=[('Rectified RGB',rgb),('Registered color-Z (m)',np.where(data['depth']>0,data['depth'],np.nan)),
           ('Predicted mask, not GT',data['mask'] if data['mask'] is not None else np.zeros(data['depth'].shape)),
           ('RGB motion (px)',data.get('motion'))]
    if maps:
        winner=maps['winner_native_index'];items += [('Native depth-camera Z (m)',maps['native_depth_m']),('Native depth valid pixels',maps['raw_valid']),('Rectified RGB valid map',maps['color_valid']),('Measured native projected density',maps['density']),('Winning native pixel index',np.where(winner>=0,winner,np.nan)),('Sampling-gap candidates, not truth',maps['hole_candidates'])]
        np.savez_compressed(out/f'{camera}_frame_{frame:06d}_registration_maps.npz',**{k:maps[k] for k in ['density','winner_native_index','hole_candidates','color_valid','raw_valid','native_depth_m']})
    else:items += [('No native registration evidence',np.zeros(data['depth'].shape))]*6
    for ax,(title,image) in zip(axes.ravel(),items):
        if image is not None:
            display=ax.imshow(image,cmap='viridis' if np.ndim(image)==2 else None)
            if np.ndim(image)==2:fig.colorbar(display,ax=ax,shrink=.6)
        ax.set_title(title);ax.set_xlabel('u (px)');ax.set_ylabel('v (px)')
    fig.suptitle(f'{scene["name"]}, {camera}, output {frame}, raw {scene["source_by_frame"][frame]}\ncalibration {scene.get("calibration_sha256")}; no absolute accuracy GT')
    fig.tight_layout();fig.savefig(out/f'{camera}_frame_{frame:06d}_provenance.png',dpi=100);plt.close(fig)
    if pair_maps:
        fig,axes=plt.subplots(1,len(pair_maps),figsize=(6*len(pair_maps),5));axes=np.atleast_1d(axes)
        for ax,(target,residual) in zip(axes,pair_maps.items()):
            display=ax.imshow(residual*1000,cmap='coolwarm',vmin=-50,vmax=50);fig.colorbar(display,ax=ax,label='projected source Z - target D (mm)')
            ax.set_title(f'{camera} -> {target}, N={np.isfinite(residual).sum()}\nall projective candidates, not certified matches')
        fig.suptitle(f'{scene["name"]}, output {frame}; display ±50 mm, full tails retained in Parquet')
        fig.tight_layout();fig.savefig(out/f'{camera}_frame_{frame:06d}_projective_residuals.png',dpi=100);plt.close(fig)


def correspondence_plot(path,a,b,rows):
    if not rows:return
    # Visualization-only deterministic sample; all matches and counts are saved.
    h=max(a['rgb'].shape[0],b['rgb'].shape[0]);w=a['rgb'].shape[1]
    canvas=np.zeros((h,w+b['rgb'].shape[1],3),np.uint8);canvas[:a['rgb'].shape[0],:w]=a['rgb'];canvas[:b['rgb'].shape[0],w:]=b['rgb']
    selected=np.linspace(0,len(rows)-1,min(100,len(rows)),dtype=int)
    for k in selected:
        row=rows[k];p=(round(row['u_i_px']),round(row['v_i_px']));q=(w+round(row['u_j_px']),round(row['v_j_px']))
        color=(0,200,0) if row['geometry_accepted'] else (0,0,255)
        cv2.circle(canvas,p,2,color,1);cv2.circle(canvas,q,2,color,1);cv2.line(canvas,p,q,color,1)
    cv2.putText(canvas,f'RGB-only reciprocal matches: N={len(rows)}, plotted={len(selected)}; green geometry-qualified, red rejected',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    cv2.imwrite(str(path),canvas)


def heldout_models(paths,settings,out,exclude_datasets=()):
    """Constrained arms are fitted separately; all accepted rows enter normal sums."""
    groups={};unobservable={};evaluations=defaultdict(lambda:defaultdict(lambda:np.zeros(3)))
    def observations():
        for row in iter_rows(paths):
            if row['dataset'] in exclude_datasets:continue
            if row.get('population')=='leave_one_camera_out_RGB_prediction':continue
            stationary=row.get('region_i')=='conditional_stationary_rgb' and row.get('region_j')=='conditional_stationary_rgb'
            if not stationary:continue
            group=row.get('split_group') or row['dataset']
            held_scene=int(stable(group)[:8],16)%5==0
            patch=None
            if row.get('u_i_px') is not None and row.get('v_i_px') is not None:
                patch=stable([row.get('calibration_sha256'),row['camera'],int(row['u_i_px']//settings.patch_pixels),int(row['v_i_px']//settings.patch_pixels)])
            held_patch=bool(patch and int(patch[:8],16)%5==0)
            held=held_scene or held_patch
            row=dict(row,holdout_cluster=group,holdout_kind='scene_and_stationary_tile' if held_scene and held_patch else 'scene' if held_scene else 'stationary_tile')
            key=(row.get('calibration_sha256'),row['camera'],row.get('target_camera'))
            epi=row.get('signed_sampson_px');derivatives=(row.get('epipolar_derivative_px_per_m') or [])+(row.get('epipolar_derivative_px_per_rad') or [])
            if epi is not None and len(derivatives)==6 and np.isfinite(derivatives).all():
                baseline=np.asarray(row.get('fixed_baseline_world_m',[]),float)
                if baseline.shape==(3,) and np.linalg.norm(baseline)>1e-8:
                    # Epipolar geometry cannot determine baseline length. Keep it
                    # fixed and fit only two transverse translation directions.
                    _,_,basis=np.linalg.svd((baseline/np.linalg.norm(baseline))[None],full_matrices=True)
                    transverse=basis[1:].T
                    row=dict(row,translation_basis_world=transverse.tolist())
                    x=np.r_[np.asarray(derivatives[:3])@transverse,derivatives[3:]]
                    yield ('A4_fixed_relative_pose_first_order',key,held,x,-float(epi),row)
            if row.get('geometry_accepted') and row.get('edge_safe_i') and row.get('sensor_minus_rgb_i_m') is not None:
                z=row.get('rgb_depth_i_m')
                if z is not None and z>0:yield ('A3_constant_bias_and_scale_fixed_pose',key,held,np.array([1.,z]),float(row['sensor_minus_rgb_i_m']),row)
            if row.get('geometry_accepted') and row.get('edge_safe_j') and row.get('sensor_minus_rgb_j_m') is not None:
                z=row.get('rgb_depth_j_m')
                if z is not None and z>0:
                    yield ('A3_constant_bias_and_scale_fixed_pose',(key[0],key[2],key[1]),held,np.array([1.,z]),float(row['sensor_minus_rgb_j_m']),row)
    for pass_index in range(3):
        accum={}
        for arm,key,held,x,y,row in observations():
            identity=(arm,key)
            if held:
                if pass_index==2 and identity in groups:
                    model=groups[identity];prediction=float(x@model['beta']);after=y-prediction
                    evaluations[identity][row['holdout_cluster'],row['frame_index'],row['holdout_kind']]+=np.array([1.,abs(y),abs(after)])
                continue
            n=len(x);entry=accum.setdefault(identity,dict(A=np.zeros((n,n)),b=np.zeros(n),count=0,scenes=set(),before=0.,after=0.,translation_basis_world=row.get('translation_basis_world')))
            beta=groups.get(identity,{}).get('beta',np.zeros(n));residual=y-x@beta
            # Declared robust influence caps, not acceptance filters; no rows discarded.
            scale=settings.feature_sigma_px*settings.pixel_gate_sigma if arm.startswith('A4') else .005
            weight=min(1.,scale/max(abs(residual),1e-12))
            entry['A']+=weight*np.outer(x,x);entry['b']+=weight*x*y;entry['count']+=1;entry['scenes'].add(row.get('split_group') or row['dataset'])
        if pass_index==2:break
        for identity,entry in accum.items():
            arm,_=identity;A=entry['A'];b=entry['b'];n=len(b)
            if entry['count']<n*3:
                unobservable[identity]=dict(train_rows=entry['count'],reason='insufficient_training_correspondences');continue
            values,vectors=np.linalg.eigh(A);threshold=max(1e-12,float(values.max())*1e-10)
            rank=int((values>threshold).sum())
            if rank<n:
                unobservable[identity]=dict(train_rows=entry['count'],rank=rank,required_rank=n,reason='rank_deficient_fixed_gauge_or_scene_support');continue
            root=np.sqrt(np.maximum(values,threshold))[:,None]*vectors.T
            target=np.linalg.solve(root.T,b)
            if arm.startswith('A4'):
                bound=np.r_[np.full(n-3,.01),np.full(3,np.deg2rad(.3))]
                beta=lsq_linear(root,target,bounds=(-bound,bound)).x
            else:beta=np.linalg.solve(A,b)
            groups[identity]=dict(beta=beta,train_rows=entry['count'],train_scenes=sorted(entry['scenes']),rank=rank,translation_basis_world=entry.get('translation_basis_world'))
    result=[]
    for identity,model in groups.items():
        arm,key=identity;validation=evaluations.get(identity,{});by_scene=defaultdict(list);by_tile=defaultdict(list)
        for (dataset,frame,kind),(count,before,after) in validation.items():
            if kind.startswith('scene'):by_scene[dataset].append((before-after)/count)
            if kind in {'stationary_tile','scene_and_stationary_tile'}:by_tile[dataset].append((before-after)/count)
        effects=[float(np.median(v)) for v in by_scene.values()];labels=list(by_scene)
        result.append(dict(arm=arm,calibration_sha256=key[0],camera=key[1],target_camera=key[2],coefficients=model['beta'].tolist(),
            coefficient_units='two_transverse_translation_m, rotation_world_xyz_rad; baseline length fixed' if arm.startswith('A4') else 'offset_m, dimensionless_scale',translation_basis_world=model.get('translation_basis_world'),
            train_rows=model['train_rows'],train_scenes=model['train_scenes'],heldout_rows=int(sum(v[0] for v in validation.values())),heldout_frame_clusters=len(validation),heldout_scenes=labels,
            holdout_support_by_kind={kind:int(sum(value[0] for key,value in validation.items() if key[2]==kind)) for kind in ['scene','stationary_tile','scene_and_stationary_tile']},
            heldout_improvement=cluster_interval(effects,labels,settings.seed,settings.bootstrap_draws),
            heldout_stationary_tile_improvement=cluster_interval([float(np.median(v)) for v in by_tile.values()],list(by_tile),settings.seed,settings.bootstrap_draws),
            evidence_grade='POSSIBLE' if len(labels)>=2 and effects and np.median(effects)>0 else 'UNOBSERVABLE' if not effects else 'REJECTED_FOR_TESTED_SCOPE',
            identifiability='supported_but_confounded_if_predictive; conditional stationary image tiles are physical-patch proxies; unknown calibration covariance prevents absolute bias attribution',
            production_parameters_changed=False))
    for identity,evidence in unobservable.items():
        if identity not in groups:
            arm,key=identity;result.append(dict(arm=arm,calibration_sha256=key[0],camera=key[1],target_camera=key[2],evidence_grade='UNOBSERVABLE',coefficients=None,**evidence))
    atomic_json(Path(out)/'heldout_counterfactual_models.json',result)
    return result


def summarize(root,manifest,receipts,settings):
    root=Path(root);parts=root/'partitions';tables={name:sorted(parts.glob(f'*/*/{name}.parquet')) for name in ['frames','pairs','unordered_pairs','correspondences','patches','timesync','attribution']}
    from audit_common import join_partitions
    for name,paths in tables.items():
        print(f'Finalization: merging {name}, {len(paths)} partitions (sequential, 128-row read batches)',flush=True)
        join_partitions(paths,root/(name+'.parquet'))
    frame_groups=defaultdict(list);frame_rows=[];camera_rows=[];pair_groups=defaultdict(list);by_dataset=defaultdict(list)
    for row in iter_rows(tables['frames']):
        compact={k:row.get(k) for k in ['dataset','camera','frame_index','status','depth_valid_fraction','native_invalid_fraction','depth_replay_status','rgb_replay_status']}
        frame_groups[row['dataset'],row['camera']].append(compact);by_dataset[row['dataset']].append(compact)
    for row in iter_rows(tables['pairs']):
        compact={k:row.get(k) for k in ['frame_index','source_frame_index','signed_median_m','mae_m','support_count']}
        compact['rgb_supported']={k:row.get('rgb_supported',{}).get(k) for k in ['signed_median_m','count']}
        pair_groups[row['dataset'],row['camera'],row.get('target_camera')].append(compact)
    for (dataset,camera),rows in frame_groups.items():
        measured=[r for r in rows if r.get('depth_valid_fraction') is not None]
        native=[r for r in rows if r.get('native_invalid_fraction') is not None]
        camera_rows.append(dict(dataset=dataset,camera=camera,expected_observations=len(rows),measured_observations=len(measured),
            native_measured_observations=len(native),mean_registered_valid_fraction=float(np.mean([r['depth_valid_fraction'] for r in measured])) if measured else None,
            mean_native_invalid_fraction=float(np.mean([r['native_invalid_fraction'] for r in native])) if native else None,
            depth_exact_replay_equal=sum(r.get('depth_replay_status')=='bitwise_equal' for r in rows),
            depth_replay_failed=sum(r.get('depth_replay_status')=='failed_bitwise_reproduction' for r in rows),
            rgb_exact_replay_equal=sum(r.get('rgb_replay_status')=='bitwise_equal' for r in rows)))
    for row in iter_rows(tables['frames']):
        frame_rows.append({k:row.get(k) for k in ['dataset','camera','frame_index','source_frame_index','status','depth_valid_fraction','native_invalid_fraction','depth_replay_different_pixels','rgb_replay_different_pixels','depth_edge_to_rgb_edge_median_px']})
    dataset_rows=[];seen_contents={};duplicate_datasets=[]
    for scene in manifest['scenes']:
        rows=by_dataset.get(scene['name'],[]);r=receipts.get(scene['id'],{})
        fingerprint=r.get('processed_content_fingerprint');alias=seen_contents.get(fingerprint) if fingerprint else None
        if alias:duplicate_datasets.append(scene['name'])
        elif fingerprint:seen_contents[fingerprint]=scene['name']
        dataset_rows.append(dict(dataset=scene['name'],calibration_sha256=scene.get('calibration_sha256'),raw_source=scene.get('source'),
            expected_frame_camera_rows=scene.get('expected_frame_camera_rows'),processed_frame_camera_rows=len(rows),
            failed_frame_camera_rows=sum(x.get('status')=='failed' or x.get('depth_replay_status')=='failed_bitwise_reproduction' or x.get('rgb_replay_status')=='failed_bitwise_reproduction' for x in rows),
            expected_ordered_pair_rows=scene.get('expected_ordered_pair_rows'),processed_ordered_pair_rows=r.get('pair_rows',0),
            status=r.get('status','not_scheduled_pilot' if manifest['mode']=='pilot' else 'failed'),
            native_available=bool(scene.get('source') and scene.get('calibration')),verified_content_alias_of=alias,absolute_metric_accuracy='unverified'))
    write_csv(root/'summary_by_dataset.csv',dataset_rows);write_csv(root/'summary_by_camera.csv',camera_rows);write_csv(root/'summary_by_frame.csv',frame_rows)
    write_csv(root/'coverage_receipt.csv',dataset_rows)
    print('Finalization: summaries and held-out models',flush=True)
    models=heldout_models(tables['correspondences'],settings,root,duplicate_datasets)
    write_pair_plots(root,pair_groups,settings)
    write_inventory_plot(root,dataset_rows,camera_rows)
    dependence_curves(tables['correspondences'],settings,root)
    timing_models(tables['timesync'],settings,root)
    print('Finalization: patch intervals, one scene at a time',flush=True)
    patch_track_intervals(tables['patches'],settings,root)
    failure_gallery(root,manifest,frame_rows,pair_groups)
    quality_assurance(root,manifest,receipts)
    calibration_groups=defaultdict(list)
    for s in manifest['scenes']:calibration_groups[s.get('calibration_sha256')].append(s['name'])
    total_expected=sum(s.get('expected_frame_camera_rows',0) for s in manifest['scenes']);total_processed=sum(len(v) for v in by_dataset.values())
    replay_measured=sum(r['depth_exact_replay_equal']+r['depth_replay_failed'] for r in camera_rows);replay_failed=sum(r['depth_replay_failed'] for r in camera_rows)
    incomplete=manifest['mode']!='full' or total_processed!=total_expected or any(r['status']!='complete' for r in dataset_rows)
    report=['# ALL REAL DATASET DEPTH ERROR ROOT CAUSE REPORT','', '**PILOT / NON-CONCLUSIVE**' if manifest['mode']!='full' else '**EXHAUSTIVE RUN — inspect coverage receipts before interpreting**','',
        '## Executive technical verdict','',
        'Absolute metric bias is unverified: no independent metric target or independently validated baseline covariance was supplied. Cross-view disagreements do not identify a guilty camera.',
        f'Processed {total_processed} of {total_expected} expected converted frame/camera observations. Exact native registration replay was measurable for {replay_measured}, with {replay_failed} failures. No global causal conclusion is issued for incomplete coverage.' if incomplete else f'Complete observation enumeration: {total_processed}/{total_expected}. Exact native registration replay: {replay_measured} measurable, {replay_failed} failures.',
        '', '## Inventory, provenance and calibration strata','',f'{len(manifest["scenes"])} scenes; {len(calibration_groups)} calibration-hash strata. Differing rigs/calibrations are not pooled as one calibration. See coverage_receipt.csv, manifest.json and skips_and_failures.jsonl.',
        '', '## Native missingness versus registration coverage','',
        'Native zero samples are directly observed missing measurements. Forward-rasterization gap candidates require occupancy brackets and continuous measured-neighborhood support. Their unknown depths prevent a unique numerical causal partition. Conservative footprint fill is a read-only counterfactual, never sensor truth.',
        '', '## Native sensor repeatability and shape','',
        'Per-camera temporal maps report median, MAD-derived sigma, drift and dropout transitions at the same native rays. Only textured RGB-stationary background evidence supports conditional repeatability. Other pixels remain unknown. A constant 30 mm bias can have zero temporal noise; local plane residual is not accuracy.',
        '', '## RGB-only geometry','',
        'All reciprocal SIFT candidates are retained. Initial correspondence selection and triangulation use RGB and fixed camera geometry, never depth-residual magnitude. Feature-localization covariance is propagated by numerical Jacobians. Baseline/calibration covariance is unknown; pose perturbation ranges are assumed sensitivities rather than calibrated CIs. No-match and weak-parallax observations are explicitly unavailable.',
        '', '## Dense and independently supported cross-view populations','',
        'pairs.parquet keeps all projective candidates, signed histograms, broad tails, invalid overlap and source z-buffer visibility. RGB-anchored same-surface residuals are reported separately. Positive projected Z minus target D is compatible with occlusion, timing, pose/depth error or wrong surface; it is not automatically called sensor bias. Homography-supported plane candidates use independent RGB identities and held-out feature anchors. Three-view RGB cycles provide leave-one-camera-out depth predictions when observable.',
        '', '## Synchronization','',
        'Nominal/index/recorder timestamps do not certify sensor synchronization. Raw decoded-RGB/depth repeat receipts and image-content lag sweeps are separate from physical clock measurements. Dynamic epipolar improvements require stationary negative controls and held-out support before interpretation. RGB-depth edge-lag overlap is a confounded proxy, not a measured exposure offset.',
        '', '## Counterfactual and held-out evidence','',
        f'{sum(m.get("coefficients") is not None for m in models)} constrained diagnostic models fitted where support permitted; see heldout_counterfactual_models.json. A3 bias/scale and A4 relative-pose first-order arms are fitted separately with fixed gauge and held-out scenes. No ICP, production refinement or simultaneous unrestricted bias/pose fitting occurs. Correlation or better fit alone is not a causal attribution.',
        '', '## Ranked cases and figures','',
        'summary_by_frame.csv and per-scene diagnostics retain actual frame/source IDs. Pair figures show all frames with support and both all-candidate and RGB-supported populations. Diagnostic provenance/feature overlays are linked under scenes/. Inspect native holes, image edges, unknown regions, textureless/specular candidates and actual mask boundaries before attributing residuals.',
        '', '## Joint3DGS / RigEuler3DGS implications','',
        'Missing observations should remain distinguishable from biased but apparently precise depth. A depth zero is not a physical background measurement. Suggestions only: no loaders, masks, calibration, reconstruction or input files were changed.',
        '', '## Unobservable quantities and missing evidence','',
        'Absolute sensor bias, true missing native depths, unknown sensor exposure ages, physical material identities, perfect mask/patch identity and calibrated baseline uncertainty remain unverified unless independent evidence is supplied. Per-test unavailable reasons and exact missing paths are retained; unavailable is not zero error.',
        '', '## Reproducibility','',
        f'Code/settings/source hashes and software versions are in manifest.json. Random seed={settings.seed}. Scene-cluster and temporal moving-block CIs report actual independent support; a single cluster never produces a narrow CI. Baseline input files are read-only. The full run must pass expected/processed/failed accounting, not merely emit a plot.',
        '', '## Known evidence restrictions','',
        'Conditional RGB stationarity and planar/homography candidates are fallible physical-identity evidence. No hardware recorder source code was available unless explicitly recorded in the manifest. Calibration covariance is absent in these input contracts; only localization CIs and declared sensitivity arms are available. These limitations constrain attribution grades and prevent absolute-accuracy claims.']
    (root/'ALL_REAL_DATASET_DEPTH_ERROR_ROOT_CAUSE_REPORT.md').write_text('\n'.join(report)+'\n')
    (root/'report.md').write_text('\n'.join(report)+'\n')
    return dataset_rows


def write_pair_plots(root,groups,settings):
    import matplotlib.pyplot as plt
    folder=root/'figures';folder.mkdir(exist_ok=True)
    for (dataset,camera,target),rows in groups.items():
        rows=sorted(rows,key=lambda r:r['frame_index']);frames=[r.get('source_frame_index',r['frame_index']) for r in rows]
        values=[r.get('signed_median_m') for r in rows];support=[r.get('support_count',0) for r in rows]
        rgb=[r.get('rgb_supported',{}).get('signed_median_m') for r in rows]
        fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
        axes[0].plot(frames,[v*1000 if v is not None else np.nan for v in values],label='all projective candidates')
        axes[0].plot(frames,[v*1000 if v is not None else np.nan for v in rgb],label='independent RGB support')
        axes[0].set_ylabel('signed target-camera Z residual (mm)');axes[0].legend();axes[0].grid(alpha=.2)
        axes[1].plot(frames,support);axes[1].set_ylabel('candidate support (pixels)');axes[1].set_xlabel('original recorded frame index')
        axes[0].set_title(f'{dataset}: {camera} -> {target}; no absolute accuracy GT')
        fig.tight_layout();fig.savefig(folder/f'{dataset}_{camera}_to_{target}.png',dpi=100);plt.close(fig)
        finite=np.array([v if v is not None else np.nan for v in values]);block=block_length(finite)
        atomic_json(folder/f'{dataset}_{camera}_to_{target}_block_ci.json',dict(baseline_block_frames=block,sensitivity=[block_interval(finite,max(1,b),settings.seed,settings.bootstrap_draws) for b in sorted({max(1,block//2),block,block*2})],interpretation='frame-median distribution with temporal block resampling, not pixel-IID accuracy'))


def write_inventory_plot(root,datasets,cameras):
    import matplotlib.pyplot as plt
    if not datasets:return
    fig,ax=plt.subplots(figsize=(12,max(5,len(datasets)*.16)))
    y=np.arange(len(datasets));ax.barh(y,[r['expected_frame_camera_rows'] or 0 for r in datasets],label='expected',alpha=.35)
    ax.barh(y,[r['processed_frame_camera_rows'] for r in datasets],label='processed')
    ax.set_yticks(y,[r['dataset'] for r in datasets],fontsize=5);ax.set_xlabel('frame x camera observations');ax.legend();fig.tight_layout()
    fig.savefig(root/'figures/inventory_coverage.png',dpi=120);plt.close(fig)


def cloud_diagnostic(out,scene,frame,images,cameras,K,matches):
    """Save all calibrated points; deterministic decimation is display-only."""
    import matplotlib.pyplot as plt
    from audit_geometry import backproject,transform_points
    out=Path(out);arrays={};colors=['tab:red','tab:green','tab:blue','tab:orange','tab:purple']
    fig=plt.figure(figsize=(14,6));ax=fig.add_subplot(121,projection='3d');rgbax=fig.add_subplot(122,projection='3d')
    for i,(image,camera) in enumerate(zip(images,cameras)):
        if not image['usable']:continue
        points,xy=backproject(image['depth'],K);world=transform_points(points,np.asarray(camera['pose_matrix_4x4']))
        arrays[camera['name']+'_world_xyz_m']=world.astype(np.float32);arrays[camera['name']+'_rgb']=image['rgb'][xy[:,1],xy[:,0],::-1]
        select=np.linspace(0,len(world)-1,min(4000,len(world)),dtype=int) if len(world) else []
        if len(select):ax.scatter(*world[select].T,s=1,c=colors[i%len(colors)],label=f'{camera["name"]}, N={len(world)}')
    certified=[]
    for rows in matches.values():certified += [r['world_xyz_m'] for r in rows if r['geometry_accepted']]
    if certified:
        values=np.asarray(certified);rgbax.scatter(*values.T,s=3,c='black');arrays['RGB_triangulated_world_xyz_m']=values
    for axis,title in [(ax,'Actual calibrated RGB-D clouds: no ICP'),(rgbax,f'Independent RGB triangulation support: N={len(certified)}')]:
        axis.set_title(title);axis.set_xlabel('base X (m)');axis.set_ylabel('base Y (m)');axis.set_zlabel('base Z (m)')
    ax.legend(fontsize=7);fig.suptitle(f'{scene["name"]}, frame {frame}; visual samples only, all points in NPZ; not GT')
    fig.tight_layout();fig.savefig(out/f'frame_{frame:06d}_calibrated_3D.png',dpi=110);plt.close(fig)
    np.savez_compressed(out/f'frame_{frame:06d}_calibrated_clouds.npz',**arrays)


def dependence_curves(paths,settings,root):
    """All geometry-qualified RGB points, pooled and equal-scene views separated."""
    definitions={'range_m':np.array([0,.25,.5,.75,1,1.5,2,3,5,10,100]),'u_px':np.arange(0,2049,64),
        'brightness_dn':np.arange(0,257,32),'texture_dn':np.array([0,2,5,10,20,40,80,256]),'motion_px':np.array([0,.25,.5,1,2,4,8,16,64,10000])}
    groups=defaultdict(lambda:defaultdict(lambda:np.zeros(3)))
    for row in iter_rows(paths):
        if not row.get('geometry_accepted'):continue
        for role,camera in [('i',row.get('camera')),('j',row.get('target_camera'))]:
            y=row.get(f'sensor_minus_rgb_{role}_m')
            if camera is None or y is None or not row.get('edge_safe_'+role):continue
            factors={'range_m':row.get('rgb_depth_'+role+'_m'),'u_px':row.get('u_'+role+'_px'),
                'brightness_dn':row.get('brightness_'+role+'_dn'),'texture_dn':row.get('texture_'+role+'_dn'),'motion_px':row.get('rgb_motion_'+role+'_px')}
            for factor,value in factors.items():
                if value is None or not np.isfinite(value):continue
                edges=definitions[factor];index=int(np.searchsorted(edges,value,side='right')-1)
                if not 0<=index<len(edges)-1:continue
                key=(row.get('calibration_sha256'),camera,factor,index)
                groups[key][row['dataset']]+=np.array([1.,y,abs(y)])
    rows=[]
    for (calibration,camera,factor,index),scenes in groups.items():
        values=[s[1]/s[0] for s in scenes.values()];ids=list(scenes)
        pooled=np.sum(list(scenes.values()),axis=0);ci=cluster_interval(values,ids,settings.seed,settings.bootstrap_draws)
        rows.append(dict(calibration_sha256=calibration,camera=camera,factor=factor,bin_left=definitions[factor][index],bin_right=definitions[factor][index+1],
            matched_point_count=int(pooled[0]),scene_count=len(ids),pooled_signed_mean_m=float(pooled[1]/pooled[0]),pooled_mae_m=float(pooled[2]/pooled[0]),
            equal_scene_median_of_scene_means_m=ci['estimate'],cluster_ci95_m=ci['ci95'],calibration_uncertainty='not established; conditional relative RGB-depth discrepancy'))
    write_csv(root/'dependence_curves.csv',rows)
    import matplotlib.pyplot as plt
    keys=sorted({(r['calibration_sha256'],r['camera'],r['factor']) for r in rows})
    for calibration,camera,factor in keys:
        selected=sorted([r for r in rows if (r['calibration_sha256'],r['camera'],r['factor'])==(calibration,camera,factor)],key=lambda r:r['bin_left'])
        fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
        x=[(r['bin_left']+r['bin_right'])/2 for r in selected];y=np.array([r['equal_scene_median_of_scene_means_m']*1000 for r in selected])
        axes[0].plot(x,y,'o-');axes[0].set_ylabel('equal-scene signed discrepancy (mm)');axes[0].axhline(0,c='gray',lw=.5)
        for xx,r in zip(x,selected):
            if r['cluster_ci95_m']:axes[0].plot([xx,xx],np.array(r['cluster_ci95_m'])*1000,c='black')
        axes[1].plot(x,[r['matched_point_count'] for r in selected],'o-');axes[1].set_ylabel('RGB-point support');axes[1].set_xlabel(factor)
        axes[0].set_title(f'{camera}, rig {str(calibration)[:12]}: conditional RGB-depth evidence, not absolute bias')
        fig.tight_layout();fig.savefig(root/'figures'/f'dependence_{str(calibration)[:12]}_{camera}_{factor}.png',dpi=110);plt.close(fig)
    return rows


def timing_models(paths,settings,root):
    groups=defaultdict(lambda:defaultdict(dict))
    for row in iter_rows(paths):
        if row.get('experiment') not in {'camera_camera_rgb_lag','within_camera_RGB_depth_triangulated_track_lag'}:continue
        if row['experiment']=='within_camera_RGB_depth_triangulated_track_lag':
            if row.get('moving_depth_mae_m') is None:continue
            row=dict(row,moving_epipolar_median_px=row['moving_depth_mae_m']*1000,static_epipolar_median_px=row['static_depth_mae_m']*1000 if row.get('static_depth_mae_m') is not None else None)
        elif row.get('moving_epipolar_median_px') is None:continue
        key=(row.get('calibration_sha256'),row['camera'],row.get('target_camera'),row['experiment'])
        # Histograms and provenance are already in Parquet; lag fitting needs
        # only these four scalars, not the complete per-observation JSON.
        compact={k:row.get(k) for k in ['moving_epipolar_median_px','static_epipolar_median_px','moving_support','static_support']}
        cluster=row.get('split_group') or row['dataset']
        groups[key][cluster,row['frame_index']][row['lag_converted_frames']]=compact
    results=[]
    for key,frames in groups.items():
        unit='px' if key[3]=='camera_camera_rgb_lag' else 'mm'
        costs=defaultdict(list);static_costs=defaultdict(list);eligible_train=0
        for (cluster,frame),lags in frames.items():
            if int(stable(cluster)[:8],16)%5==0:continue
            if 0 not in lags:continue
            eligible_train+=1
            for lag,row in lags.items():
                if row.get('moving_support',0)>=3:costs[lag].append((cluster,row['moving_epipolar_median_px']))
                if row.get('static_support',0)>=3 and row.get('static_epipolar_median_px') is not None:static_costs[lag].append((cluster,row['static_epipolar_median_px']))
        def equal_scene(rows):
            scenes=defaultdict(list)
            for scene,value in rows:scenes[scene].append(value)
            return float(np.median([np.median(v) for v in scenes.values()])) if scenes else None
        curve={lag:equal_scene(rows) for lag,rows in costs.items()}
        if not curve or 0 not in curve:
            results.append(dict(calibration_sha256=key[0],camera=key[1],target_camera=key[2],experiment=key[3],loss_unit=unit,evidence_grade='UNOBSERVABLE',reason='No adequate moving RGB-feature baseline/training support'));continue
        chosen=min(curve,key=lambda lag:(curve[lag],abs(lag)));effects=defaultdict(list);controls=defaultdict(list)
        for (cluster,frame),lags in frames.items():
            if int(stable(cluster)[:8],16)%5!=0 or 0 not in lags or chosen not in lags:continue
            a,b=lags[0],lags[chosen]
            if min(a.get('moving_support',0),b.get('moving_support',0))>=3:effects[cluster].append(a['moving_epipolar_median_px']-b['moving_epipolar_median_px'])
            if min(a.get('static_support',0),b.get('static_support',0))>=3 and a.get('static_epipolar_median_px') is not None and b.get('static_epipolar_median_px') is not None:controls[cluster].append(a['static_epipolar_median_px']-b['static_epipolar_median_px'])
        ci=cluster_interval([float(np.median(v)) for v in effects.values()],list(effects),settings.seed,settings.bootstrap_draws)
        control=cluster_interval([float(np.median(v)) for v in controls.values()],list(controls),settings.seed,settings.bootstrap_draws)
        supported=unit=='px' and chosen!=0 and ci['ci95'] is not None and ci['ci95'][0]>0 and control['estimate'] is not None and abs(control['estimate'])<=settings.feature_sigma_px
        results.append(dict(calibration_sha256=key[0],camera=key[1],target_camera=key[2],experiment=key[3],loss_unit=unit,chosen_target_lag_converted_frames=chosen,
            sign_convention='Compare source frame t with target stored frame t+lag',training_equal_scene_loss=curve,
            training_equal_scene_static_loss={lag:equal_scene(values) for lag,values in static_costs.items()},train_frame_groups=eligible_train,
            heldout_moving_improvement=ci,heldout_static_change=control,
            evidence_grade='SUPPORTED' if supported else 'POSSIBLE' if effects else 'UNOBSERVABLE',
            claim_scope='relative image-content lag only; match populations may change; not hardware sensor/exposure-clock latency',
            dropped_endpoints='all unavailable shifted endpoints recorded in timesync.parquet'))
    atomic_json(root/'heldout_timing_models.json',results)
    import matplotlib.pyplot as plt
    for index,row in enumerate(results):
        if 'training_equal_scene_loss' not in row:continue
        curve=row['training_equal_scene_loss'];control=row['training_equal_scene_static_loss'];x=sorted(curve)
        fig,ax=plt.subplots(figsize=(9,4));ax.plot(x,[curve[v] for v in x],'o-',label='moving, equal scene medians')
        ax.plot(x,[control.get(v,np.nan) for v in x],'o-',label='conditional stationary negative control')
        ax.set_xlabel('target lag (converted frame intervals)');ax.set_ylabel('RGB epipolar distance (px)' if row['loss_unit']=='px' else 'sensor minus RGB-track depth MAE (mm)');ax.legend()
        ax.set_title(f'{row["camera"]} -> {row["target_camera"]}, train groups={row["train_frame_groups"]}; image-content lag only')
        fig.tight_layout();fig.savefig(root/'figures'/f'lag_sensitivity_{index:03d}.png',dpi=110);plt.close(fig)
    return results


def patch_track_intervals(paths,settings,root):
    rows=[]
    # Group one scene at a time; never retain millions of patch histograms.
    for _,scene_paths in groupby(sorted(paths,key=str),key=lambda p:Path(p).parents[1]):
        groups=defaultdict(list)
        for row in iter_rows(scene_paths):
            if row.get('population')!='native_local_patch' or row.get('stationary_rgb_evidence_fraction') is None or row['stationary_rgb_evidence_fraction']<.8:continue
            compact={k:row[k] for k in ['frame_index','plane_normal','centroid_m']}
            groups[row['dataset'],row['camera'],row['patch_x_px'],row['patch_y_px']].append(compact)
        rows.extend(_patch_intervals(groups,settings))
    atomic_json(root/'patch_track_uncertainty.json',rows)
    return rows


def _patch_intervals(groups,settings):
    rows=[]
    for (dataset,camera,x,y),samples in groups.items():
        samples=sorted(samples,key=lambda r:r['frame_index']);train=samples[::2];held=samples[1::2]
        if len(train)<3 or len(held)<3:
            rows.append(dict(dataset=dataset,camera=camera,patch_x_px=x,patch_y_px=y,status='not_measurable',reason='fewer than six conditional stationary patch observations'));continue
        normal=np.median([r['plane_normal'] for r in train],axis=0);normal/=np.linalg.norm(normal)
        centroid=np.median([r['centroid_m'] for r in train],axis=0);offset=-normal@centroid
        residuals=[float(normal@np.asarray(r['centroid_m'])+offset) for r in held]
        block=block_length(residuals)
        rows.append(dict(dataset=dataset,camera=camera,patch_x_px=x,patch_y_px=y,status='conditional_empirical_patch_track',train_frames=[r['frame_index'] for r in train],heldout_frames=[r['frame_index'] for r in held],
            normal_reference=normal.tolist(),offset_reference_m=float(offset),heldout_centroid_signed_m=residuals,
            reference='temporal self-reference on disjoint frames, not external plane ground truth',
            uncertainty=block_interval(residuals,block,settings.seed,settings.bootstrap_draws)))
    return rows


def failure_gallery(root,manifest,frame_rows,pair_groups):
    import matplotlib.pyplot as plt
    scenes={s['name']:s for s in manifest['scenes']};candidates=[]
    for row in frame_rows:
        if row.get('native_invalid_fraction') is not None:candidates.append((row['native_invalid_fraction'],'native_invalid_fraction',row['dataset'],row['camera'],row['frame_index']))
    for (dataset,camera,target),rows in pair_groups.items():
        for row in rows:
            if row.get('mae_m') is not None:candidates.append((row['mae_m'],'all_projective_mae_m_not_bias',dataset,camera,row['frame_index']))
    chosen=[]
    for criterion in sorted({r[1] for r in candidates}):chosen+=sorted([r for r in candidates if r[1]==criterion],reverse=True)[:15]
    gallery=root/'failure_gallery';gallery.mkdir(exist_ok=True);registry=[]
    for score,criterion,dataset,camera,frame in chosen:
        scene=scenes[dataset];base=Path(scene['path']);name=f'frame_{frame:06d}'
        rgb_path=base/'rgb'/camera/(name+'.png');depth_path=base/'depth_npy'/camera/(name+'.npy');mask_path=base/'mask'/camera/(name+'.png')
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        rgb=cv2.imread(str(rgb_path));depth=np.load(depth_path,allow_pickle=False) if depth_path.is_file() else None;mask=cv2.imread(str(mask_path),cv2.IMREAD_UNCHANGED)
        for ax,title,image in zip(axes,['RGB','Registered camera-Z (m)','Predicted mask, not GT'],[cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB) if rgb is not None else None,depth,mask]):
            if image is not None:
                shown=ax.imshow(image,cmap='viridis' if image.ndim==2 else None)
                if image.ndim==2:fig.colorbar(shown,ax=ax,shrink=.6)
            else:ax.text(.1,.5,'not measurable: source unavailable')
            ax.set_title(title)
        fig.suptitle(f'{dataset}, {camera}, converted {frame}; {criterion}={score:.6g}\ncalibration {scene.get("calibration_sha256")}; no faulty-camera inference')
        fig.tight_layout();path=gallery/f'{dataset}_{camera}_{name}_{criterion}.png';fig.savefig(path,dpi=100);plt.close(fig)
        registry.append(dict(dataset=dataset,camera=camera,frame_index=frame,source_frame_index=scene['source_by_frame'].get(frame),criterion=criterion,value=score,figure=str(path.relative_to(root)),rgb_path=str(rgb_path),depth_path=str(depth_path)))
    write_csv(root/'worst_frames.csv',registry)
    return registry


def quality_assurance(root,manifest,receipts):
    frames=pl.scan_parquet(root/'frames.parquet');pairs=pl.scan_parquet(root/'pairs.parquet')
    checks={}
    for name,data,keys in [('frames',frames,['dataset_id','frame_index','camera']),('pairs',pairs,['dataset_id','frame_index','camera','target_camera'])]:
        count=data.select(pl.len()).collect().item();unique=data.select(keys).unique().select(pl.len()).collect().item()
        checks[name]=dict(rows=count,unique_keys=unique,duplicates=count-unique)
        if count!=unique:raise AssertionError(f'Duplicated {name} rows in exhaustive output')
    targets=[s for s in manifest['scenes'] if manifest['mode']=='full' or s['id'] in receipts]
    expected_frames=sum(s.get('expected_frame_camera_rows',0) if manifest['mode']=='full' else min(3,len(s.get('frames',[])))*len(s.get('cameras',[])) for s in targets)
    expected_pairs=sum(s.get('expected_ordered_pair_rows',0) if manifest['mode']=='full' else min(3,len(s.get('frames',[])))*len(s.get('cameras',[]))*max(0,len(s.get('cameras',[]))-1) for s in targets)
    checks['selected_scope_expected_rows']=dict(frames=expected_frames,pairs=expected_pairs)
    checks['accounting_complete']=checks['frames']['rows']==expected_frames and checks['pairs']['rows']==expected_pairs and all(receipts.get(s['id'],{}).get('processing_complete',False) for s in targets)
    checks['measurements_without_required_failures']=checks['accounting_complete'] and all(r.get('status')=='complete' for r in receipts.values())
    checks['full_scope']=manifest['mode']=='full';checks['absolute_accuracy_established']=False
    atomic_json(root/'quality_assurance.json',checks)
    return checks
