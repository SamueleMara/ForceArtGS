from pathlib import Path
from collections import Counter

from audit_common import revision
from audit_registration import load_converter
from .common import atomic_json,code_hashes,file_record,json_read,sha,stable


def load_inventory(audit_root):
    root=Path(audit_root).expanduser().resolve();old=json_read(root/'manifest.json')
    if stable(old['config'])!=old['config_sha256']:raise ValueError('Original manifest config hash is invalid')
    scenes=old['scenes']
    for s in scenes:
        if 'inventory_error' in s:continue
        s['source_by_frame']={int(k):int(v) for k,v in s['source_by_frame'].items()}
        s['frames']=[int(f) for f in s['frames']]
        p=Path(s['path']).resolve()
        if not p.is_dir():s['source_status']='CONVERTED_SOURCE_UNAVAILABLE'
        else:s['source_status']='MEASURED'
        s['raw_status']='MEASURED' if s.get('source') and Path(s['source']).is_dir() else 'SOURCE_UNAVAILABLE'
        s['family']=s['name'].split('_episode_')[0] if s['name'].startswith('final_') else 'legacy:'+s['name']
    return root,old,scenes


def inventory(root,old,scenes):
    return dict(original_manifest=file_record(root/'manifest.json'),original_config_sha256=old['config_sha256'],
        original_code_hashes=old['config']['code_sha256'],converter_sha256=old['config']['converter_sha256'],
        scenes=scenes,scene_count=len(scenes),final_episode_count=sum(s['name'].startswith('final_') for s in scenes),
        legacy_scene_count=sum(not s['name'].startswith('final_') for s in scenes),
        expected_frame_camera_rows=sum(s.get('expected_frame_camera_rows',0) for s in scenes),
        expected_ordered_pair_rows=sum(s.get('expected_ordered_pair_rows',0) for s in scenes),
        calibration_strata=dict(Counter(s.get('calibration_sha256') for s in scenes)),
        followup_code=code_hashes(),revisions=[revision(Path('/home/samuelemara/Joint3DGS/RigEuler3DGS')),revision(Path('/home/samuelemara/Joint3DGS/Sapien'))],
        units='registered color-camera Z meters; native depth-camera Z uint16 mm',hardware_sync='UNVERIFIED')


def scene_inputs(scene):
    """Per-scene metadata and original committed partition receipts anchor stages."""
    return [file_record(p) for p in sorted((Path(scene['path'])/'metadata').rglob('*')) if p.is_file() and p.suffix in {'.json','.csv','.parquet'}]
