"""Inventory all real conversions; resolve original sources with recorded evidence."""
from collections import OrderedDict
import csv
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np
import polars as pl

from audit_common import sha,stable,file_record,dumps

SKIP={'rgb','mask','depth_npy','depth_png','colmap','colmap_final','metadata','pointcloud','video_rgb','.runtime'}


def discover(roots):
    found={}
    for root in roots:
        root=Path(root).expanduser().resolve()
        if not root.is_dir():raise FileNotFoundError(root)
        for directory,children,_ in os.walk(root,followlinks=False):
            path=Path(directory);marker=path/'metadata/real_conversion.json'
            if marker.is_file():
                found.setdefault(path.resolve(),[]).append(str(path));children[:]=[]
            else:children[:]=[c for c in children if c not in SKIP and not c.startswith('.')]
    if not found:raise ValueError('No real scenes with metadata/real_conversion.json discovered')
    return sorted(found)


def resolve_source(scene,conversion,raw_roots):
    pointer=Path(conversion['source']).expanduser();episode=int(conversion['episode'])
    raw=scene/'metadata/raw';copied=raw/f'episode_{episode:06d}.parquet'
    candidates=[pointer] if pointer.is_dir() else []
    if not candidates:
        for root in raw_roots:
            root=Path(root).expanduser().resolve()
            candidates.extend([root] if (root/'meta/info.json').is_file() else [p for p in root.iterdir() if p.is_dir() and (p/'meta/info.json').is_file()])
    accepted=[]
    for candidate in dict.fromkeys(p.resolve() for p in candidates):
        parquet=candidate/f'data/chunk-000/episode_{episode:06d}.parquet'
        if copied.is_file() and parquet.is_file():
            if sha(copied)==sha(parquet):accepted.append((candidate,'exact_copied_episode_parquet_sha256'))
            continue
        # A still-valid recorded absolute pointer is provenance; relocation needs
        # positive content identity, not a similar name/calibration shared by a rig.
        if candidate==pointer.resolve() and (candidate/'meta/camera_calibration.json').is_file():accepted.append((candidate,'recorded_absolute_source_pointer; episode identity not independently hash-certified'))
    if len(accepted)==1:return str(accepted[0][0]),accepted[0][1]
    return None,'raw_source_unavailable_or_ambiguous_content_identity' if not accepted else 'ambiguous_multiple_raw_recordings_with_identical_episode_hash'


def inspect_scene(path,raw_roots,converter):
    path=Path(path).resolve();meta=path/'metadata';conversion=json.loads((meta/'real_conversion.json').read_text())
    optional_issues=[]
    def optional_json(path):
        if not path.is_file():return None
        try:
            value=json.loads(path.read_text())
            if not isinstance(value,dict):raise ValueError('expected JSON object')
            return value
        except Exception as error:
            optional_issues.append(dict(path=str(path),status='not_measurable',reason=str(error)));return None
    cameras=json.loads((meta/'cameras.json').read_text())['cameras']
    if not cameras:raise ValueError('No recorded cameras')
    source_frames=conversion['source_frames'];frames=conversion.get('output_frames',source_frames)
    if len(frames)!=len(source_frames) or not frames:raise ValueError('Invalid converted/source index mapping')
    for values in [frames,source_frames]:
        array=np.asarray(values)
        if array.ndim!=1 or not np.issubdtype(array.dtype,np.integer) or len(set(values))!=len(values) or min(values)<0:raise ValueError('Frame IDs must be unique nonnegative integers')
    if not (path/'rgb').is_dir() or not (path/'depth_npy').is_dir():raise ValueError('Real conversion lacks RGB/depth_npy directories')
    K=np.asarray(conversion['rectified_K'],float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0 or not np.allclose(K[2],[0,0,1]):raise ValueError('Invalid recorded rectified K')
    calibration_path=meta/'raw/camera_calibration.json';calibration=optional_json(calibration_path)
    info_path=meta/'raw/info.json';info=optional_json(info_path) or {}
    seen=set()
    for c in cameras:
        if not re.fullmatch(r'cam_\d+',c['name']) or not re.fullmatch(r'cam_\d+',c['source_camera_name']):raise ValueError('Unsafe or unknown recorded camera name')
        if c['name'] in seen:raise ValueError('Duplicate camera ID')
        seen.add(c['name']);converter.transform(c['pose_matrix_4x4'],'recorded camera-to-world')
        if c.get('pose_convention')!='opencv_optical_camera_to_world':raise ValueError('Unknown camera pose convention')
        if not np.allclose([c['fx'],c['fy'],c['cx'],c['cy']],[K[0,0],K[1,1],K[0,2],K[1,2]]):raise ValueError('Camera K differs from real_conversion.json')
        native=calibration.get('cameras',{}).get(c['source_camera_name']) if calibration else None
        c['source_serial_metadata']={k:native[k] for k in ['serial','serial_number','device_serial','device_uid'] if native and k in native}
        c['native_to_processed_pose_equal']=bool(np.allclose(c['pose_matrix_4x4'],native['T_base_color_optical'],atol=1e-9)) if native and 'T_base_color_optical' in native else None
    source,mapping=resolve_source(path,conversion,raw_roots)
    masks=meta/'mask_generation.json';mask_meta=optional_json(masks)
    parquet=meta/f'raw/episode_{conversion["episode"]:06d}.parquet'
    schema={};parquet_rows={};sensor_fields=[]
    if parquet.is_file():
        try:
            schema={k:str(v) for k,v in pl.read_parquet_schema(parquet).items()}
            columns=[name for name in ['frame_index','episode_index','timestamp'] if name in schema]
            columns += [name for name in schema if re.search(r'(sensor|header|capture|hardware).*time|time.*(sensor|hardware)',name,re.I)]
            sensor_fields=[name for name in columns if name not in ['frame_index','episode_index','timestamp']]
            if 'frame_index' in columns:
                rows=pl.read_parquet(parquet,columns=columns).to_dicts()
                if len({r['frame_index'] for r in rows})!=len(rows):raise ValueError('Duplicate Parquet frame indices')
                parquet_rows={int(r['frame_index']):r for r in rows}
        except Exception as error:optional_issues.append(dict(path=str(parquet),status='not_measurable',reason=str(error)));parquet_rows={}
    scene=dict(id=stable(str(path))[:16],name=path.name,path=str(path),conversion=conversion,cameras=cameras,
        frames=frames,source_by_frame=dict(zip(frames,source_frames)),K=K.tolist(),calibration=calibration,
        calibration_sha256=sha(calibration_path) if calibration else None,source=source,source_mapping_evidence=mapping,
        info=info,mask_metadata_present=mask_meta is not None,mask_manifest_completed=mask_meta.get('completed') if mask_meta else None,
        mask_manifest_empty=mask_meta.get('empty') if mask_meta else None,
        parquet_schema=schema,parquet_by_source_frame=parquet_rows,sensor_timestamp_candidate_fields=sensor_fields,
        timestamps_note='frame_timestamps.csv and Parquet timestamp are recorder/nominal time, not hardware synchronization certification',
        expected_frame_camera_rows=len(frames)*len(cameras),expected_ordered_pair_rows=len(frames)*len(cameras)*(len(cameras)-1),
        calibration_status=conversion.get('calibration_status'),calibration_warning=conversion.get('calibration_warning'))
    scene['registered_units_confirmed']=conversion.get('depth_output_unit')=='float32 meters, rectified color-camera Z; zero means missing'
    scene['split_group']=sha(parquet) if parquet.is_file() else stable([source or conversion['source'],conversion['episode']])
    # Metadata availability is inventoried without silently dropping any frame.
    scene['metadata_records']=[file_record(p) for p in [meta/'real_conversion.json',meta/'cameras.json',calibration_path,info_path,meta/'frame_timestamps.csv',masks,parquet]]
    if source:
        scene['metadata_records'] += [file_record(Path(source)/p) for p in ['meta/info.json','meta/camera_calibration.json',f'data/chunk-000/episode_{conversion["episode"]:06d}.parquet']]
    scene['input_signature']=stable(scene['metadata_records'])
    scene['inventory_issues']=optional_issues
    timestamp_file=meta/'frame_timestamps.csv'
    scene['timestamp_csv_by_frame']={}
    if timestamp_file.is_file():
        try:
            with timestamp_file.open(newline='') as stream:
                for row in csv.DictReader(stream):
                    frame=int(row['frame_index'])
                    if frame in scene['timestamp_csv_by_frame']:scene['inventory_issues'].append(f'duplicate_nominal_timestamp_frame:{frame}')
                    scene['timestamp_csv_by_frame'][frame]=row
        except Exception as error:scene['inventory_issues'].append(dict(path=str(timestamp_file),reason=str(error),status='not_measurable'));scene['timestamp_csv_by_frame']={}
    for kind in ['rgb','depth_npy','depth_png','mask']:
        folder=path/kind
        available_cameras=[p.name for p in folder.iterdir() if p.is_dir() and re.fullmatch(r'cam_\d+',p.name)] if folder.is_dir() else []
        if set(available_cameras)!=seen:scene['inventory_issues'].append(f'{kind}: camera set differs: {available_cameras}')
        for camera in available_cameras:
            ids={int(p.stem.removeprefix('frame_')) for p in (folder/camera).iterdir() if re.fullmatch(r'frame_\d+\.(png|npy)',p.name)}
            missing=sorted(set(frames)-ids);extra=sorted(ids-set(frames))
            if missing or extra:scene['inventory_issues'].append(dict(folder=kind,camera=camera,missing_frames=missing,extra_unmapped_frames=extra))
    if (path/'conversion_in_progress.json').exists():scene['inventory_issues'].append('conversion_in_progress.json exists')
    if source is None:scene['inventory_issues'].append(mapping)
    if calibration is None:scene['inventory_issues'].append('native_calibration_missing: native/replay tests not measurable')
    if not schema:scene['inventory_issues'].append('raw_parquet_missing: recorder timestamp audit unavailable')
    if not sensor_fields:scene['inventory_issues'].append('no recognized sensor timestamp fields; hardware age is unobservable')
    if any(c['native_to_processed_pose_equal'] is False for c in cameras):scene['inventory_issues'].append('processed_pose_differs_from_copied_native_T_base_color_optical; no automatic refinement performed')
    if conversion.get('depth_output_unit')!='float32 meters, rectified color-camera Z; zero means missing':scene['inventory_issues'].append('registered_depth_unit_not_confirmed')
    return scene


class Images:
    """Bounded read-only frame/feature cache; access retains exact output indices."""
    def __init__(self,scene,settings,failures,max_items=40):
        self.scene=scene;self.settings=settings;self.failures=failures;self.max_items=max_items;self.cache=OrderedDict();self.inputs={};self.known_inputs={}
    def add_input(self,path):
        path=Path(path)
        key=str(path)
        if key not in self.known_inputs:self.known_inputs[key]=file_record(path)
        self.inputs[key]=self.known_inputs[key]
        return path
    def image(self,path,kind):
        path=self.add_input(path)
        if not path.is_file():return None
        if kind=='npy':return np.load(path,allow_pickle=False)
        return cv2.imread(str(path),cv2.IMREAD_COLOR if kind=='rgb' else cv2.IMREAD_UNCHANGED)
    def get(self,frame,camera,features=False):
        from audit_rgb import image_features,temporal_rgb
        key=(frame,camera['name']);root=Path(self.scene['path']);name=f'frame_{frame:06d}'
        if key not in self.cache:
            input_paths=[str(root/k/camera['name']/(name+suffix)) for k,suffix in [('rgb','.png'),('depth_npy','.npy'),('mask','.png'),('depth_png','.png')]]
            rgb=self.image(root/'rgb'/camera['name']/(name+'.png'),'rgb');depth=self.image(root/'depth_npy'/camera['name']/(name+'.npy'),'npy')
            mask=self.image(root/'mask'/camera['name']/(name+'.png'),'mask');png=self.image(root/'depth_png'/camera['name']/(name+'.png'),'mask')
            expected=(camera['height'],camera['width']);errors=[]
            if rgb is None or rgb.shape!=(*expected,3) or rgb.dtype!=np.uint8:errors.append('missing_or_invalid_rgb')
            if depth is None or depth.shape!=expected or depth.dtype!=np.float32:errors.append('missing_or_invalid_registered_float32_depth')
            if mask is not None and (mask.shape!=expected or mask.dtype!=np.uint8 or not np.isin(mask,[0,255]).all()):errors.append('invalid_binary_mask_or_geometry');mask=None
            if png is not None and (png.shape!=expected or png.dtype!=np.uint16):errors.append('invalid_uint16_depth_png');png=None
            if mask is None:self.failures.append(dict(scene=self.scene['name'],camera=camera['name'],frame=frame,test='mask_regions',status='not_measurable',path=str(root/'mask'/camera['name']/(name+'.png')),reason='mask missing or invalid; no mask is created'))
            for reason in errors:self.failures.append(dict(scene=self.scene['name'],camera=camera['name'],frame=frame,test='E0',status='failed',reason=reason,path_candidates=input_paths))
            usable=rgb is not None and rgb.shape==(*expected,3) and depth is not None and depth.shape==expected and depth.dtype==np.float32 and self.scene.get('registered_units_confirmed',False)
            data=dict(rgb=rgb,depth=depth,mask=mask>0 if mask is not None else None,depth_png=png,errors=errors,usable=usable,input_paths=input_paths)
            if usable:
                index=self.scene['frames'].index(frame);previous=None
                if index>0:
                    old=root/'rgb'/camera['name']/f'frame_{self.scene["frames"][index-1]:06d}.png'
                    previous=self.image(old,'rgb')
                    input_paths.append(str(old))
                    if previous is not None and previous.shape!=rgb.shape:previous=None
                data['stationary'],data['motion'],data['temporal_quality']=temporal_rgb(rgb,previous,data['mask'],self.settings)
            self.cache[key]=data
        data=self.cache.pop(key);self.cache[key]=data
        for path in data['input_paths']:self.inputs[path]=self.known_inputs[path]
        if features and data['usable'] and 'features' not in data:data['features']=image_features(data['rgb'])
        while len(self.cache)>self.max_items:self.cache.popitem(last=False)
        return data
