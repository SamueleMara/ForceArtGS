"""Exhaustive E1/E2/automatic E3, bounded frame chunks and read-only inputs."""
from collections import Counter,defaultdict
from pathlib import Path
import itertools
import json

import cv2
import numpy as np
import polars as pl

from audit_registration import camera_context
from .common import atomic_json,file_record,family,rows_write,sha
from .mask_annotation import mask_metrics
from .registration_trace import trace
from .surfaces import extract,associate
from .identifiability import residual_and_jacobian


def read_data(scene,camera,frame):
    root=Path(scene['path']);name=f'frame_{frame:06d}';c=camera['name']
    paths=[root/'rgb'/c/(name+'.png'),root/'depth_npy'/c/(name+'.npy'),root/'mask'/c/(name+'.png')]
    if any(not p.is_file() for p in paths[:2]):return None,paths
    rgb=cv2.imread(str(paths[0]));depth=np.load(paths[1],allow_pickle=False);mask=cv2.imread(str(paths[2]),cv2.IMREAD_UNCHANGED) if paths[2].exists() else None
    shape=(camera['height'],camera['width'])
    if rgb is None or rgb.shape!=(*shape,3) or depth.shape!=shape or depth.dtype!=np.float32:raise ValueError(f'Invalid RGB-D geometry: {paths}')
    if mask is not None and (mask.shape!=shape or mask.dtype!=np.uint8 or not np.isin(mask,[0,255]).all()):raise ValueError(f'Invalid mask: {paths[2]}')
    return dict(rgb=rgb,depth=depth,mask=mask>0 if mask is not None else None),paths


def original_hashes(audit_root,scene):
    """Source byte identities already recorded by the exhaustive audit."""
    hashes={}
    for path in sorted((audit_root/'partitions'/scene['id']).glob('*/receipt.json')):
        receipt=json.loads(path.read_text())
        for r in receipt['inputs']:
            if r.get('exists'):hashes[r['path']]=r['sha256']
    return hashes


def debug_frame(path,scene,camera,frame,data,surface,maps):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    items=[('Rectified RGB',cv2.cvtColor(data['rgb'],cv2.COLOR_BGR2RGB)),('Measured registered color Z (m)',np.where(data['depth']>0,data['depth'],np.nan)),
           ('Predicted mask, not GT',data['mask']),('Automatic patch IDs, not physical GT',surface['labels'] if surface else None),
           ('Native measured Z (m)',maps.get('native_depth_m') if maps else None),('Hole category: outside/observed/gap/discontinuity/unknown',maps.get('category') if maps else None)]
    for ax,(name,image) in zip(axes.flat,items):
        if image is not None:im=ax.imshow(image);fig.colorbar(im,ax=ax,shrink=.6) if image.ndim==2 else None
        else:ax.text(.1,.5,'UNOBSERVABLE')
        ax.set_title(name,fontsize=9)
    fig.suptitle(f'{scene["name"]} {camera["name"]}: processed {frame}, source {scene["source_by_frame"][frame]}; automatic conditional evidence')
    fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=110);plt.close(fig)


def scene_measurements(scene,audit_root,out,converter,stages):
    K=np.array(scene['K']);cameras=scene['cameras'];contexts={};previous={};previous_missing={};previous_masks={};old_missing={};outputs=[];counts=Counter();inputs={};all_decisions=Counter()
    expected_hashes=original_hashes(audit_root,scene);validated=set()
    def check(paths):
        for p in paths:
            p=Path(p);key=str(p)
            if key in validated:continue
            rec=file_record(p)
            if key in expected_hashes and rec.get('sha256')!=expected_hashes[key]:raise ValueError(f'Original audited input bytes changed: {p}')
            inputs[key]=rec;validated.add(key)
    check([Path(r['path']) for r in scene.get('metadata_records',[])])
    for c in cameras:
        native=(scene.get('calibration') or {}).get('cameras',{}).get(c['source_camera_name'])
        if native:contexts[c['name']]=camera_context(c,native,K,converter)
        if scene.get('source'):
            check([Path(scene['source'])/f'videos/chunk-000/observation.images.{c["source_camera_name"]}/episode_{scene["conversion"]["episode"]:06d}.mp4'])
    match_rows=defaultdict(list)
    if 'verified-geometry' in stages:
        cols=['frame_index','camera','target_camera','u_i_px','v_i_px','u_j_px','v_j_px','keypoint_i','keypoint_j']
        df=pl.scan_parquet(audit_root/'correspondences.parquet').filter((pl.col('dataset')==scene['name'])&pl.col('population').is_null()).select(cols).collect(engine='streaming')
        for r in df.iter_rows(named=True):match_rows[r['frame_index'],r['camera'],r['target_camera']].append(r)
    representatives={scene['frames'][0],scene['frames'][len(scene['frames'])//2],scene['frames'][-1]}
    buffers=defaultdict(list);pose_rows=[];stage_key='_'.join(sorted(stages))
    for index,frame in enumerate(scene['frames']):
        data_by_camera={};surface_by_camera={}
        for c in cameras:
            name=c['name'];data,paths=read_data(scene,c,frame);check(paths)
            base=dict(scene=scene['name'],family=family(scene),calibration_sha256=scene.get('calibration_sha256'),camera=name,processed_frame=frame,source_frame=scene['source_by_frame'][frame])
            if data is None:
                buffers['coverage'].append(dict(base,status='SOURCE_UNAVAILABLE'));counts['source_unavailable']+=1;continue
            data_by_camera[name]=data;counts['frame_camera_observations']+=1;maps=None
            if 'holes' in stages:
                raw_path=Path(scene['source'])/f'depth/chunk-000/observation.depth.{c["source_camera_name"]}/episode_{scene["conversion"]["episode"]:06d}/frame_{scene["source_by_frame"][frame]:06d}.png' if scene.get('source') else None
                if raw_path is not None and raw_path.is_file() and name in contexts:
                    check([raw_path]);raw=cv2.imread(str(raw_path),cv2.IMREAD_UNCHANGED)
                    r,maps=trace(raw,data['depth'],contexts[name],converter,data['rgb'],data['mask'],save_detail=frame in representatives)
                    counts['registration_replays']+=1;counts['replay_different_pixels']+=r['bitwise_different_pixels']
                    region=r.pop('regions');r['regions_json']=json.dumps(region);buffers['registration'].append(dict(base,**r))
                    if frame in representatives:
                        p=out/'registration'/scene['id']/f'{name}_{frame:06d}_maps.npz';p.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(p,**{k:v for k,v in maps.items() if isinstance(v,np.ndarray)});outputs.append(p)
                else:buffers['registration'].append(dict(base,status='SOURCE_UNAVAILABLE',reason='native depth/calibration unavailable; no invented projection'))
            if 'coverage' in stages:
                r,missing=mask_metrics(data['depth'],data['mask'],previous_missing.get(name));previous_missing[name]=missing
                if name in previous_masks and data['mask'] is not None:
                    before=previous_masks[name];now=data['mask'];r.update(mask_entering_pixels=int((now&~before).sum()),mask_leaving_pixels=int((before&~now).sum()),
                        newly_missing_inside_persistent_mask=int((now&before&missing&~old_missing[name]).sum()))
                if data['mask'] is not None:previous_masks[name]=data['mask']
                old_missing[name]=missing.copy()
                r['mask_boundary_bands_json']=json.dumps(r.pop('mask_boundary_bands',[]));r['manual_regions_json']=json.dumps(r.pop('manual_regions',[]))
                buffers['coverage'].append(dict(base,status='CONDITIONAL',**r));counts['coverage_rows']+=1
            surface=None
            if 'verified-geometry' in stages:
                cv2.setRNGSeed((frame+1731)%2147483647)
                surface=extract(data['depth'],data['rgb'],data['mask'],K,np.array(c['pose_matrix_4x4']),previous.get(name));previous[name]=surface;surface_by_camera[name]=surface
                for p in surface['patches']:buffers['patches'].append(dict(base,**p));counts['patches']+=1
            if frame in representatives:
                p=out/'annotation_review'/scene['id']/f'{stage_key}_{name}_{frame:06d}_automatic.png';debug_frame(p,scene,c,frame,data,surface,maps);outputs.append(p)
        if 'verified-geometry' in stages:
            for ca,cb in itertools.combinations(cameras,2):
                na,nb=ca['name'],cb['name'];base=dict(scene=scene['name'],family=family(scene),calibration_sha256=scene.get('calibration_sha256'),camera=na,target_camera=nb,processed_frame=frame,source_frame=scene['source_by_frame'][frame])
                if na not in surface_by_camera or nb not in surface_by_camera:
                    buffers['scope'].append(dict(base,status='SOURCE_UNAVAILABLE',candidate_pairs=0,automatic_consistent_pairs=0));continue
                a,b=surface_by_camera[na],surface_by_camera[nb];da,db=data_by_camera[na],data_by_camera[nb]
                matches=match_rows.get((frame,na,nb),[])
                rows=associate(a,b,da['depth'],db['depth'],K,np.array(ca['pose_matrix_4x4']),np.array(cb['pose_matrix_4x4']),matches)
                qualified=[r for r in rows if r['status']=='AUTO_MULTICUE_CONSISTENT_CONDITIONAL']
                for r in rows:
                    buffers['associations'].append(dict(base,**r));all_decisions[r['status']]+=1
                buffers['scope'].append(dict(base,status='CONDITIONAL' if qualified else 'UNIDENTIFIABLE',candidate_pairs=len(rows),automatic_consistent_pairs=len(qualified)))
                counts['unordered_pair_observations']+=1
                for r in qualified:
                    p,q=a['patches'][r['source_patch']],b['patches'][r['target_patch']]
                    if not (p['temporal_persistent'] and q['temporal_persistent']):continue
                    yy,xx=np.indices(db['depth'].shape)
                    ys,xs=np.nonzero((b['labels']==q['patch_id'])&(db['depth']>0)&((xx+3*yy)%3==0));selection=np.arange(len(xs))[::max(1,len(xs)//64)]
                    xs,ys=xs[selection],ys[selection];rays=b['rays'][ys,xs];local=rays*db['depth'][ys,xs,None]
                    residual,J=residual_and_jacobian(local,rays,np.array(cb['pose_matrix_4x4']),np.array(p['normal_world']),p['offset_world_m'])
                    for value,j in zip(residual,J):pose_rows.append(dict(base,surface_key=f'{scene["id"]}:{na}:{p["track_id"]}:{nb}:{q["track_id"]}',residual_m=float(value),jacobian=j.tolist()))
        if (index+1)%16==0 or index==len(scene['frames'])-1:
            chunk=index//16
            for kind,rows in buffers.items():
                folder={'registration':'registration','coverage':'coverage','patches':'geometry/patches','associations':'geometry/associations','scope':'geometry/scope'}[kind]
                outputs.append(rows_write(out/folder/scene['id']/f'{chunk:06d}.parquet',rows))
            if pose_rows:
                outputs.append(rows_write(out/'geometry/pose_support'/scene['id']/f'{chunk:06d}.parquet',pose_rows));pose_rows.clear()
            buffers.clear();print(f'{scene["name"]}: {index+1}/{len(scene["frames"])} frames, {counts["patches"]} patches',flush=True)
    inp=out/'inputs'/f'{scene["id"]}_{stage_key}.json';atomic_json(inp,list(inputs.values()));outputs.append(inp)
    result=dict(status='CONTRADICTED' if counts['replay_different_pixels'] else 'CONDITIONAL',counts=dict(counts),association_decisions=dict(all_decisions),
        absolute_accuracy='UNVERIFIED',human_verified_images=0,physical_identity='AUTOMATIC_CONDITIONAL',source_status=scene['raw_status'])
    return result,outputs
