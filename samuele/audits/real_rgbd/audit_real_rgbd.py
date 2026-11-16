#!/usr/bin/env python3
"""Exhaustive, read-only real RGB-D depth-error attribution audit.

Full mode processes every discovered real scene, converted frame, camera and
ordered pair. Pilot mode is explicitly non-conclusive. Outputs must be a new
directory outside all source/converted roots; --resume requires matching hashes.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import sys
import time
import resource
import traceback

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
# Bound streaming-reader concurrency before Polars initializes its thread pool.
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('POLARS_ROW_GROUP_PREFETCH_SIZE','1')
os.environ.setdefault('POLARS_RECORD_BATCH_PREFETCH_SIZE','1')
import cv2
import numpy as np
import polars as pl

from audit_common import VERSION,Settings,TABLES,atomic_json,clean,dumps,sha,stable,file_record,validate_records,revision,record,table,distribution
from audit_provenance import discover,inspect_scene,Images
from audit_registration import load_converter,camera_context
from audit_rgb import pair_rgb,lag_rgb
from audit_geometry import dense_pair,transform_points,depth_sample
from audit_measurements import frame_metrics,attribution_rows,leave_one_out
from audit_temporal import NativeSeries,audit_raw_series,array_hash
from audit_reporting import summarize,diagnostic,correspondence_plot,cloud_diagnostic


def fail(rows,scene,frame,camera,test,reason,status='not_measurable'):
    rows.append(dict(scene=scene['name'],frame=frame,camera=camera,test=test,status=status,reason=str(reason)))


def flush_failures(path,rows):
    if not rows:return
    with Path(path).open('a') as stream:
        for row in rows:stream.write(dumps(row)+'\n')


def selected_frames(scene,mode):
    frames=scene['frames']
    return frames if mode=='full' else [frames[i] for i in sorted({0,len(frames)//2,len(frames)-1})]


def inputs_changed_in_inventory(scene):validate_records(scene['metadata_records'])


def scene_audit(scene,root,converter,settings,mode,resume):
    scene_dir=root/'scenes'/scene['id'];scene_dir.mkdir(parents=True,exist_ok=True)
    final_receipt=scene_dir/'receipt.json';parts=root/'partitions'/scene['id']
    if final_receipt.is_file():
        old=json.loads(final_receipt.read_text())
        if not resume:raise FileExistsError(final_receipt)
        inputs_changed_in_inventory(scene)
        validate_records(old.get('derived_outputs',[]))
        for receipt_path in sorted(parts.glob('*/receipt.json')):
            receipt=json.loads(receipt_path.read_text());validate_records(receipt['inputs']);validate_records(receipt['outputs'])
        if old.get('processing_complete'):print(f'RESUME completed accounting: {scene["name"]}',flush=True);return old
    cameras=scene['cameras'];frames=selected_frames(scene,mode);K=np.asarray(scene['K']);failure_rows=[]
    images=Images(scene,settings,failure_rows,max_items=48)
    contexts={};series={};registered_series={};videos={};video_failures=set();raw_inputs=[]
    source=Path(scene['source']) if scene.get('source') else None
    for camera in cameras:
        name=camera['name'];native=scene.get('calibration',{}).get('cameras',{}).get(camera['source_camera_name']) if scene.get('calibration') else None
        if native:
            try:
                contexts[name]=camera_context(camera,native,K,converter)
                shape=(native['depth']['height'],native['depth']['width'])
                series[name]=NativeSeries(scene_dir/'temporal',name,frames,shape,resume)
            except Exception as error:fail(failure_rows,scene,None,name,'native_calibration_context',error)
        registered_series[name]=NativeSeries(scene_dir/'temporal',name+'_registered',frames,(camera['height'],camera['width']),resume)
        if source:
            video=source/f'videos/chunk-000/observation.images.{camera["source_camera_name"]}/episode_{scene["conversion"]["episode"]:06d}.mp4'
            raw_inputs.append(file_record(video))
            if video.is_file():videos[name]=iter(converter.selected_rgb(video,[scene['source_by_frame'][f] for f in frames]))
    camera_by_name={c['name']:c for c in cameras};counts=defaultdict(int);all_chunk_inputs={};raw_receipts={}
    try:
        for chunk_number,start in enumerate(range(0,len(frames),settings.chunk_frames)):
            chunk_started=time.perf_counter()
            current=frames[start:start+settings.chunk_frames];part=parts/f'chunk_{chunk_number:06d}';receipt_path=part/'receipt.json'
            chunk_indices=list(range(start,start+len(current)));old=None
            if receipt_path.is_file():
                if not resume:raise FileExistsError(receipt_path)
                old=json.loads(receipt_path.read_text());validate_records(old['inputs']);validate_records(old['outputs'])
                for name,cache in {**{f'native:{n}':s for n,s in series.items()},**{f'registered:{n}':s for n,s in registered_series.items()}}.items():
                    if cache.chunk_hash(chunk_indices)!=old['temporal_cache_hashes'][name]:raise ValueError('Temporal cache checksum changed on resume')
            raw_rgb_by_frame={}
            # Sequential decoding exactly follows the existing converter, including
            # already completed chunks, without rerunning their geometry/features.
            for frame in current:
                raw_rgb_by_frame[frame]={}
                for name,video in videos.items():
                    if name in video_failures:continue
                    try:
                        index,rgb=next(video)
                        if index!=scene['source_by_frame'][frame]:raise ValueError('Raw RGB source index differs from recorded mapping')
                        raw_rgb_by_frame[frame][name]=rgb
                    except Exception as error:
                        video_failures.add(name);fail(failure_rows,scene,frame,name,'RGB_exact_replay',error)
            if old:
                for key,value in old['counts'].items():counts[key]+=value
                for item in old['inputs']:all_chunk_inputs[item['path']]=item
                print(f'RESUME {scene["name"]}: chunk {chunk_number}',flush=True);continue
            rows={name:[] for name in TABLES};images.inputs={};part.mkdir(parents=True,exist_ok=True)
            for offset,frame in enumerate(current):
                index=start+offset;observations=[];maps_by_camera={};pair_maps=defaultdict(dict);rows_by_pair={}
                for camera in cameras:
                    name=camera['name'];data=images.get(frame,camera,features=True);observations.append(data)
                    if not data['usable']:
                        rows['frames'].append(record(scene,frame,name,status='failed',reason='missing_or_invalid_registered_rgbd'))
                        rows['timesync'].append(record(scene,frame,name,status='not_measurable',reason='missing_or_invalid_rgbd'))
                        rows['attribution'].append(record(scene,frame,name,status='not_measurable',mechanism='unresolved',evidence_grade='UNOBSERVABLE',reason='required observation unavailable'))
                        continue
                    raw=None;context=contexts.get(name)
                    if source and context:
                        raw_path=source/f'depth/chunk-000/observation.depth.{camera["source_camera_name"]}/episode_{scene["conversion"]["episode"]:06d}/frame_{scene["source_by_frame"][frame]:06d}.png'
                        images.add_input(raw_path)
                        if raw_path.is_file():raw=cv2.imread(str(raw_path),cv2.IMREAD_UNCHANGED)
                        if raw is None or raw.dtype!=np.uint16 or raw.shape!=(scene['calibration']['cameras'][camera['source_camera_name']]['depth']['height'],scene['calibration']['cameras'][camera['source_camera_name']]['depth']['width']):
                            fail(failure_rows,scene,frame,name,'native_depth',f'missing_or_invalid {raw_path}');raw=None
                    try:metrics,patches,maps=frame_metrics(scene,camera,frame,data,context,converter,raw,raw_rgb_by_frame[frame].get(name),settings)
                    except Exception as error:
                        fail(failure_rows,scene,frame,name,'frame_metrics',error,'failed')
                        metrics=record(scene,frame,name,status='failed',reason=str(error));patches=[];maps=None
                    rows['frames'].append(metrics);rows['attribution']+=attribution_rows(scene,frame,name,metrics)
                    for patch in patches:rows['patches'].append(record(scene,frame,name,population='native_local_patch',**patch))
                    if maps is not None:
                        maps_by_camera[name]=maps;series[name].put(index,raw,maps['native_stationary'])
                    registered_series[name].put_meters(index,data['depth'],data['stationary'])
                    rows['timesync'].append(record(scene,frame,name,**data.get('temporal_quality',{}),decoded_rgb_sha256=array_hash(data['rgb']),
                        source_timestamp_fields=metrics.get('source_timestamp_fields'),recorded_timestamp_s=metrics.get('recorded_timestamp_s'),
                        nominal_timestamp_s=metrics.get('nominal_timestamp_s'),clock_claim='no hardware synchronization certification'))
                    if metrics.get('depth_replay_status')=='failed_bitwise_reproduction' or metrics.get('rgb_replay_status')=='failed_bitwise_reproduction':
                        fail(failure_rows,scene,frame,name,'E0_exact_replay','Nonzero bitwise difference; inspect source mapping/version, not automatically sensor geometry','failed')
                    if metrics.get('png_quantization_different_pixels'):
                        fail(failure_rows,scene,frame,name,'E0_png_quantization','depth_png differs from quantized registered depth_npy','failed')
                pair_rows={}
                for i,j in itertools.permutations(range(len(cameras)),2):
                    a,b=observations[i],observations[j];name,target=cameras[i]['name'],cameras[j]['name']
                    if not a['usable'] or not b['usable']:
                        row=record(scene,frame,name,status='not_measurable',target_camera=target,reason='source_or_target_rgbd_missing',support_count=0)
                    else:
                        statistics,raster=dense_pair(a['depth'],b['depth'],K,np.asarray(cameras[i]['pose_matrix_4x4']),np.asarray(cameras[j]['pose_matrix_4x4']))
                        row=record(scene,frame,name,target_camera=target,**statistics);pair_maps[name][target]=raster
                    rows['pairs'].append(row);pair_rows[i,j]=row
                for i,j in itertools.combinations(range(len(cameras)),2):
                    a,b=observations[i],observations[j];name,target=cameras[i]['name'],cameras[j]['name']
                    rgb_rows=[];rgb_summary={};planes=[]
                    if a['usable'] and b['usable']:
                        cv2.setRNGSeed(int(stable([settings.seed,scene['split_group'],frame,i,j])[:8],16)%(2**31-1))
                        try:rgb_rows,rgb_summary,planes=pair_rgb(a,b,K,np.asarray(cameras[i]['pose_matrix_4x4']),np.asarray(cameras[j]['pose_matrix_4x4']),settings)
                        except Exception as error:fail(failure_rows,scene,frame,name,'RGB_geometry',error,'failed')
                    if not rgb_rows:fail(failure_rows,scene,frame,f'{name}/{target}','RGB_geometry','No independent reliable RGB geometry; not a zero-error pass')
                    if not rgb_summary.get('geometry_accepted_count',0):fail(failure_rows,scene,frame,f'{name}/{target}','RGB_triangulation','No cheiral, informative-parallax, low-reprojection RGB geometry')
                    if not rgb_summary.get('same_surface_support_count',0):fail(failure_rows,scene,frame,f'{name}/{target}','RGB_supported_depth','No geometry-qualified edge-safe depth samples; not zero error')
                    rows_by_pair[i,j]=rgb_rows
                    for row in rgb_rows:rows['correspondences'].append(record(scene,frame,name,target_camera=target,**row))
                    for patch in planes:rows['patches'].append(record(scene,frame,name,target_camera=target,**patch))
                    forward=distribution([r['signed_projected_source_minus_target_m'] for r in rgb_rows if r['same_surface_supported']])
                    reverse_values=[]
                    for r in rgb_rows:
                        if r['same_surface_supported']:
                            ray=np.linalg.inv(K)@np.array([r['u_j_px'],r['v_j_px'],1.]);point=ray*r['depth_j_m']
                            local=transform_points(point[None],np.linalg.inv(np.asarray(cameras[i]['pose_matrix_4x4']))@np.asarray(cameras[j]['pose_matrix_4x4']))[0]
                            reverse_values.append(local[2]-r['depth_i_m'])
                    pair_rows[i,j]['rgb_supported']=forward;pair_rows[i,j]['rgb_geometry']=rgb_summary
                    pair_rows[i,j]['rgb_supported_by_region']=rgb_summary.get('same_surface_by_region',{})
                    pair_rows[j,i]['rgb_supported']=distribution(reverse_values)
                    rows['unordered_pairs'].append(record(scene,frame,name,target_camera=target,pair_uniqueness='camera_i_index < camera_j_index; signed populations remain directional',
                        i_to_j=pair_rows[i,j],j_to_i=pair_rows[j,i],rgb_geometry=rgb_summary))
                    rows['attribution'].append(record(scene,frame,name,target_camera=target,mechanism='cross_view_depth_disagreement',evidence_grade='SUPPORTED' if forward['count'] else 'UNOBSERVABLE',
                        support_count=forward['count'],signed_median_m=forward['signed_median_m'],mae_m=forward['mae_m'],
                        interpretation='RGB-supported relative discrepancy; not attribution to a faulty camera or absolute bias'))
                    # All base frames and camera pairs; missing shifted endpoints are receipts.
                    full_index=scene['frames'].index(frame)
                    for lag in settings.lags:
                        shifted=full_index+lag
                        if not 0<=shifted<len(scene['frames']):
                            rows['timesync'].append(record(scene,frame,name,status='not_measurable',target_camera=target,experiment='camera_camera_rgb_lag',lag_converted_frames=lag,reason='shifted_endpoint_outside_recording'));continue
                        other=images.get(scene['frames'][shifted],cameras[j],features=True)
                        if not a['usable'] or not other['usable']:
                            rows['timesync'].append(record(scene,frame,name,status='not_measurable',target_camera=target,experiment='camera_camera_rgb_lag',lag_converted_frames=lag,reason='missing_shifted_observation'));continue
                        lag_stats=lag_rgb(a,other,K,np.asarray(cameras[i]['pose_matrix_4x4']),np.asarray(cameras[j]['pose_matrix_4x4']),settings)
                        rows['timesync'].append(record(scene,frame,name,target_camera=target,experiment='camera_camera_rgb_lag',lag_converted_frames=lag,
                            shifted_source_frame=scene['source_by_frame'][scene['frames'][shifted]],lag_nominal_recorder_s=(scene['source_by_frame'][scene['frames'][shifted]]-scene['source_by_frame'][frame])/scene['info']['fps'] if scene['info'].get('fps',0)>0 else None,**lag_stats))
                    for camera_index,role,reference in [(i,'i',target),(j,'j',name)]:
                        qualified=[r for r in rgb_rows if r['geometry_accepted']]
                        points=np.array([[r[f'u_{role}_px'],r[f'v_{role}_px']] for r in qualified],float).reshape(-1,2)
                        for lag in settings.lags:
                            shifted=full_index+lag
                            base=record(scene,frame,cameras[camera_index]['name'],target_camera=reference,experiment='within_camera_RGB_depth_triangulated_track_lag',lag_converted_frames=lag,
                                interpretation='sensor depth(t+lag) at RGB feature(t) minus independent RGB-triangulated Z(t); physical sample correspondence across lag is a hypothesis, not hardware latency proof')
                            if not qualified or not 0<=shifted<len(scene['frames']):
                                rows['timesync'].append(dict(base,status='not_measurable',reason='no_qualified_RGB_tracks_or_shift_outside_recording'));continue
                            shifted_image=images.get(scene['frames'][shifted],cameras[camera_index])
                            if not shifted_image['usable']:
                                rows['timesync'].append(dict(base,status='not_measurable',reason='shifted_depth_missing'));continue
                            measured,safe,_=depth_sample(shifted_image['depth'],points,settings.depth_edge_m)
                            residual=measured-np.array([r[f'rgb_depth_{role}_m'] for r in qualified])
                            static=np.array([r[f'region_{role}']=='conditional_stationary_rgb' for r in qualified])&safe&np.isfinite(residual)
                            moving=np.array([r[f'region_{role}']=='predicted_object' and r.get(f'rgb_motion_{role}_px') is not None and r[f'rgb_motion_{role}_px']>settings.pixel_gate_sigma*settings.feature_sigma_px for r in qualified])&safe&np.isfinite(residual)
                            rows['timesync'].append(dict(base,static_support=int(static.sum()),moving_support=int(moving.sum()),
                                static_sensor_minus_RGB=distribution(residual[static]),moving_sensor_minus_RGB=distribution(residual[moving]),all_candidate_sensor_minus_RGB=distribution(residual[np.isfinite(residual)]),
                                static_depth_mae_m=float(np.mean(np.abs(residual[static]))) if static.any() else None,moving_depth_mae_m=float(np.mean(np.abs(residual[moving]))) if moving.any() else None))
                    if frame in {frames[0],frames[len(frames)//2],frames[-1]} and a['usable'] and b['usable']:
                        correspondence_plot(scene_dir/f'{name}_{target}_frame_{frame:06d}_RGB_matches.png',a,b,rgb_rows)
                for row in leave_one_out(rows_by_pair,observations,cameras,K,settings):
                    camera=row.pop('camera');rows['correspondences'].append(record(scene,frame,camera,**row))
                # RGB-depth edge-lag is explicitly only a confounded alignment proxy.
                for i,camera in enumerate(cameras):
                    a=observations[i]
                    if not a['usable']:continue
                    rgb_edges=cv2.Canny(cv2.cvtColor(a['rgb'],cv2.COLOR_BGR2GRAY),50,150)>0
                    full_index=scene['frames'].index(frame)
                    for lag in settings.lags:
                        shifted=full_index+lag
                        if not 0<=shifted<len(scene['frames']):
                            rows['timesync'].append(record(scene,frame,camera['name'],status='not_measurable',experiment='within_camera_rgb_depth_edge_lag_proxy',lag_converted_frames=lag,reason='shifted_endpoint_outside_recording'));continue
                        b=images.get(scene['frames'][shifted],camera)
                        if not b['usable']:continue
                        d=b['depth'];valid=np.isfinite(d)&(d>0);low=cv2.erode(np.where(valid,d,np.inf),np.ones((3,3),np.uint8));high=cv2.dilate(np.where(valid,d,-np.inf),np.ones((3,3),np.uint8))
                        complete=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8))>0
                        span=np.subtract(high,low,out=np.zeros_like(high),where=complete)
                        edges=complete&(span>settings.depth_edge_m)
                        rows['timesync'].append(record(scene,frame,camera['name'],experiment='within_camera_rgb_depth_edge_lag_proxy',lag_converted_frames=lag,
                            support_count=int(edges.sum()),rgb_depth_edge_overlap_fraction=float((edges&rgb_edges).sum()/edges.sum()) if edges.any() else None,
                            interpretation='RGB texture edges need not be depth edges; no exposure-offset or same-surface certification'))
                if frame in {frames[0],frames[len(frames)//2],frames[-1]}:
                    cloud_diagnostic(scene_dir,scene,frame,observations,cameras,K,rows_by_pair)
                    for i,camera in enumerate(cameras):
                        if observations[i]['usable']:diagnostic(scene_dir,scene,camera['name'],frame,observations[i],contexts.get(camera['name']),maps_by_camera.get(camera['name']),pair_maps[camera['name']])
            failed_observations=sum(r.get('status')=='failed' or r.get('depth_replay_status')=='failed_bitwise_reproduction' or r.get('rgb_replay_status')=='failed_bitwise_reproduction' or bool(r.get('png_quantization_different_pixels',0)) for r in rows['frames'])
            outputs=[]
            for name,records in rows.items():table(part/(name+'.parquet'),records);outputs.append(file_record(part/(name+'.parquet')));counts[name]+=len(records)
            for cache in [*series.values(),*registered_series.values()]:cache.flush()
            cache_hash={**{f'native:{name}':cache.chunk_hash(chunk_indices) for name,cache in series.items()},**{f'registered:{name}':cache.chunk_hash(chunk_indices) for name,cache in registered_series.items()}}
            inputs=list(images.inputs.values())+raw_inputs+scene['metadata_records']
            validate_records(inputs)
            for item in inputs:all_chunk_inputs[item['path']]=item
            chunk_counts={name:len(v) for name,v in rows.items()};chunk_counts['failed_frame_observations']=failed_observations;counts['failed_frame_observations']+=failed_observations
            atomic_json(receipt_path,dict(frames=current,counts=chunk_counts,inputs=inputs,outputs=outputs,temporal_cache_hashes=cache_hash,processing_elapsed_s=time.perf_counter()-chunk_started,process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
            flush_failures(root/'skips_and_failures.jsonl',failure_rows);failure_rows[:]=[]
            print(f'{scene["name"]}: {min(start+len(current),len(frames))}/{len(frames)} frames, {len(cameras)} cameras, all {len(cameras)*(len(cameras)-1)} ordered pairs',flush=True)
        temporal={}
        for camera in cameras:
            name=camera['name'];temporal[name]=dict(registered=registered_series[name].finalize(settings.seed,settings.bootstrap_draws))
            if name in series:temporal[name]['native']=series[name].finalize(settings.seed,settings.bootstrap_draws)
            if source:
                raw_path=scene_dir/f'{name}_raw_series.parquet';raw_receipt=scene_dir/f'{name}_raw_series_receipt.json'
                if resume and raw_receipt.is_file():
                    receipt=json.loads(raw_receipt.read_text());validate_records(receipt['inputs']);raw_receipts[name]=receipt
                else:
                    raw_receipts[name]=audit_raw_series(scene,camera,source,raw_path);atomic_json(raw_receipt,raw_receipts[name])
        atomic_json(scene_dir/'temporal_quality.json',temporal)
        source_records=list(all_chunk_inputs.values())
        content=stable(dict(calibration=scene['calibration_sha256'],K=scene['K'],poses=[c['pose_matrix_4x4'] for c in cameras],files=sorted((str(Path(r['path']).relative_to(scene['path'])),r.get('sha256')) for r in source_records if Path(r['path']).is_relative_to(scene['path']) and any(k in Path(r['path']).parts for k in ['rgb','depth_npy','depth_png','mask']))))
        receipt=dict(status='complete' if counts['failed_frame_observations']==0 else 'complete_with_failures',processing_complete=True,dataset=scene['name'],mode=mode,frame_rows=counts['frames'],pair_rows=counts['pairs'],
            expected_selected_frame_rows=len(frames)*len(cameras),expected_selected_pair_rows=len(frames)*len(cameras)*(len(cameras)-1),
            counts=dict(counts),temporal=temporal,raw_series=raw_receipts,processed_content_fingerprint=content,
            derived_outputs=[file_record(p) for p in scene_dir.rglob('*') if p.is_file() and p.name!='receipt.json' and p.suffix!='.dat'])
        if counts['frames']!=receipt['expected_selected_frame_rows'] or counts['pairs']!=receipt['expected_selected_pair_rows']:raise AssertionError('Coverage row-count mismatch')
        atomic_json(final_receipt,receipt)
        # Only scratch created by this audit is released after final receipts/maps
        # are committed. No source or converted dataset file is ever deleted.
        for cache in [*series.values(),*registered_series.values()]:cache.release_scratch()
        return receipt
    finally:
        for video in videos.values():
            if hasattr(video,'close'):video.close()
        flush_failures(root/'skips_and_failures.jsonl',failure_rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--converted-roots',nargs='+',type=Path,required=True)
    parser.add_argument('--raw-roots',nargs='+',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--converter',type=Path,default=Path('/home/samuelemara/FreeArtGS/samuele/convert_real_recording.py'))
    parser.add_argument('--mode',choices=['inventory','pilot','full'],default='inventory')
    parser.add_argument('--all-scenes',action='store_true');parser.add_argument('--all-frames',action='store_true');parser.add_argument('--all-cameras',action='store_true')
    parser.add_argument('--pilot-scenes',nargs='+',help='Pilot only: explicit scene names; default three different recordings')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--chunk-frames',type=int,default=16)
    parser.add_argument('--seed',type=int,default=1731);parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args(argv)
    if args.mode=='full' and (not all([args.all_scenes,args.all_frames,args.all_cameras]) or args.pilot_scenes):parser.error('Full mode requires --all-scenes --all-frames --all-cameras and forbids pilot selection')
    if args.chunk_frames<1 or args.threads<1:parser.error('Chunk size and threads must be positive')
    roots=[p.expanduser().resolve() for p in args.converted_roots];raw_roots=[p.expanduser().resolve() for p in args.raw_roots];out=args.output_root.expanduser().resolve()
    for source in [*roots,*raw_roots]:
        if not source.is_dir():parser.error(f'Input root unavailable: {source}')
        if out==source or out.is_relative_to(source) or source.is_relative_to(out):parser.error('Audit output must be outside and non-nested with all input roots')
    if out.exists() and not args.resume:parser.error('Audit directory exists: choose a NEW folder, or --resume the exact same audit')
    if args.resume and not (out/'manifest.json').is_file():parser.error('--resume requires an existing audit manifest')
    for name in ['scenes','partitions','figures','.runtime']:
        if not (out/name).resolve().is_relative_to(out):parser.error(f'Audit output subfolder escapes selected root: {name}')
    os.environ['MPLCONFIGDIR']=str(out/'.runtime/matplotlib');os.environ['MPLBACKEND']='Agg';cv2.setNumThreads(args.threads)
    settings=Settings(chunk_frames=args.chunk_frames,seed=args.seed)
    converter=load_converter(args.converter)
    module_root=Path(__file__).resolve().parent;code={p.name:sha(p) for p in sorted(module_root.glob('*.py'))}
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','polars','matplotlib','pyarrow']};versions['opencv']=cv2.__version__
    config=dict(version=VERSION,settings=asdict(settings),threads=args.threads,mode=args.mode,converted_roots=list(map(str,roots)),raw_roots=list(map(str,raw_roots)),converter_sha256=sha(args.converter),code_sha256=code,versions=versions,pilot_scenes=args.pilot_scenes)
    scenes=[];inventory_failures=[]
    for path in discover(roots):
        try:scene=inspect_scene(path,raw_roots,converter)
        except Exception as error:
            inventory_failures.append(dict(scene=path.name,test='inventory',status='failed',reason=str(error),path=str(path)))
            scene=dict(id=stable(str(path))[:16],name=path.name,path=str(path),inventory_error=str(error),expected_frame_camera_rows=0,expected_ordered_pair_rows=0)
        scenes.append(scene)
    for scene in scenes:
        if scene.get('source'):
            raw=Path(scene['source']).resolve()
            if out==raw or out.is_relative_to(raw) or raw.is_relative_to(out):parser.error('Audit directory overlaps an actual resolved raw source')
    manifest=dict(config=config,config_sha256=stable(config),mode=args.mode,scenes=scenes,
        source_revisions=[revision(Path('/home/samuelemara/FreeArtGS')),revision(Path('/home/samuelemara/Joint3DGS/Sapien'))],
        scope='all real conversions; synthetic controls are separate unit tests',absolute_metric_ground_truth='not supplied or independently established')
    if args.resume:
        old=json.loads((out/'manifest.json').read_text())
        if old['config_sha256']!=manifest['config_sha256']:parser.error('Resume refused: code/settings/environment/config hash changed')
        if stable(old['scenes'])!=stable(scenes):parser.error('Resume refused: scene inventory/provenance hash changed')
    else:out.mkdir(parents=True);atomic_json(out/'manifest.json',manifest)
    flush_failures(out/'skips_and_failures.jsonl',inventory_failures)
    eligible=[s for s in scenes if 'inventory_error' not in s]
    print(f'Inventory: {len(scenes)} real scenes, {sum(s.get("expected_frame_camera_rows",0) for s in scenes)} frame/camera observations, {sum(s.get("expected_ordered_pair_rows",0) for s in scenes)} ordered pair observations; {len(inventory_failures)} inventory failures',flush=True)
    if args.mode=='inventory':
        from audit_reporting import write_csv
        write_csv(out/'coverage_receipt.csv',[dict(dataset=s['name'],status='inventory_only' if 'inventory_error' not in s else 'failed_inventory',expected_frame_camera_rows=s.get('expected_frame_camera_rows'),expected_ordered_pair_rows=s.get('expected_ordered_pair_rows'),processed_frame_camera_rows=0,processed_ordered_pair_rows=0,raw_source=s.get('source'),calibration_sha256=s.get('calibration_sha256')) for s in scenes])
        atomic_json(out/'inventory.json',manifest);return int(bool(inventory_failures))
    if args.mode=='pilot':
        wanted=args.pilot_scenes or ['final_black_box_episode_000000','final_orange_suitecase_episode_000000','fridge_plastic_20261016_163314']
        selected=[s for s in eligible if s['name'] in wanted]
        if len(selected)!=len(set(wanted)):parser.error('Some requested pilot scenes were not discovered')
    else:selected=eligible
    receipts={};failed=bool(inventory_failures)
    for scene in selected:
        try:
            receipt=scene_audit(scene,out,converter,settings,args.mode,args.resume)
            receipts[scene['id']]={k:receipt.get(k) for k in ['status','processing_complete','dataset','mode','frame_rows','pair_rows','expected_selected_frame_rows','expected_selected_pair_rows','counts','processed_content_fingerprint']}
            receipts[scene['id']]['detail_receipt']=str(out/'scenes'/scene['id']/'receipt.json')
            del receipt
            failed|=receipts[scene['id']]['status']!='complete'
        except Exception as error:
            failed=True;receipts[scene['id']]=dict(status='failed',error=str(error))
            flush_failures(out/'skips_and_failures.jsonl',[dict(scene=scene['name'],status='failed',test='scene_execution',reason=str(error),traceback=traceback.format_exc())]);print(f'FAILED {scene["name"]}: {error}',file=sys.stderr,flush=True)
    summaries=summarize(out,manifest,receipts,settings)
    atomic_json(out/'run_receipt.json',dict(mode=args.mode,scenes_discovered=len(scenes),scenes_selected=len(selected),scenes_complete=sum(r.get('status')=='complete' for r in receipts.values()),scenes_failed=sum(r.get('status')!='complete' for r in receipts.values()),results=receipts))
    if args.mode=='full':failed|=any(r['status']!='complete' or r['processed_frame_camera_rows']!=r['expected_frame_camera_rows'] or r['processed_ordered_pair_rows']!=r['expected_ordered_pair_rows'] for r in summaries)
    print(f'Audit {args.mode} reports: {out}; failures={failed}; no input dataset modified',flush=True)
    return int(failed)


if __name__=='__main__':sys.exit(main())
