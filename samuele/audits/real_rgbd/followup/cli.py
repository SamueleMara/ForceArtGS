#!/usr/bin/env python3
"""Automatic all-frame RGB-D identifiability follow-up; all inputs read-only."""
import argparse
import concurrent.futures
import contextlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import shutil

for k,v in {'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','POLARS_MAX_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}.items():os.environ[k]=v
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import cv2
cv2.setNumThreads(1)
from audit_registration import load_converter
from .common import atomic_json,code_hashes,file_record,json_read,sha,stable,stage_run,validate_records
from .manifests import inventory,load_inventory,scene_inputs
from .measurements import scene_measurements
from .report import calibration_report,generate,timing_report
from .synthetic import run_controls

STAGES={'inventory','holes','coverage','export-review-patches','verified-geometry','pose-vs-bias','timing','phase1-cause','synthetic'}


def worker(job):
    scene,audit_root,out,converter_path,core,signature,resume=job;out=Path(out)
    logfile=out/'logs'/('_'.join(sorted(core)))/(scene['id']+'.log');logfile.parent.mkdir(parents=True,exist_ok=True)
    with logfile.open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            converter=load_converter(converter_path)
            result=stage_run(out,scene,'_'.join(sorted(core)),signature,resume,
                lambda:scene_measurements(scene,Path(audit_root),out,converter,set(core)))
            return scene['id'],result
        except Exception as e:
            traceback.print_exc();return scene['id'],dict(status='FAIL',reason=str(e),scene=scene['name'],log=str(logfile))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-root',type=Path,required=True,help='Completed original exhaustive audit, read-only')
    p.add_argument('--out',type=Path,required=True,help='New isolated follow-up output; never the original audit directory')
    selection=p.add_mutually_exclusive_group(required=True);selection.add_argument('--all',action='store_true');selection.add_argument('--scene',nargs='+',help='Exact scene names; every frame/camera in selected scenes')
    p.add_argument('--stage',default='inventory,synthetic,holes,coverage,verified-geometry,pose-vs-bias,timing,phase1-cause',help='Comma-separated: '+','.join(sorted(STAGES))+'; verified-geometry is automatic, no manual prerequisites')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--rigeuler-results',type=Path,default=Path('/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours_v6_two_cams'))
    p.add_argument('--rigeuler-code',type=Path,default=Path('/home/samuelemara/Joint3DGS/RigEuler3DGS-v6'))
    p.add_argument('--rigeuler-python',type=Path,default=Path('/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python'))
    args=p.parse_args(argv);stages=set(args.stage.split(','))
    if not stages<=STAGES:p.error('Unknown stage(s): '+str(stages-STAGES))
    if args.workers<1:p.error('--workers must be positive')
    root,old,scenes=load_inventory(args.audit_root);out=args.out.expanduser().resolve()
    if out==root or root.is_relative_to(out):p.error('Output must be isolated; it cannot contain or replace the original audit')
    for raw in old['config']['converted_roots']+old['config']['raw_roots']:
        raw=Path(raw).resolve()
        if out==raw or out.is_relative_to(raw) or raw.is_relative_to(out):p.error('Output overlaps an original dataset')
    for name in ['registration','coverage','geometry','calibration','timing','phase1','receipts','inputs','logs','annotation_review','.runtime']:
        if not (out/name).resolve().is_relative_to(out):p.error('Output subdirectory escapes isolated output')
    selected=scenes if args.all else [s for s in scenes if s['name'] in args.scene]
    if args.scene and {s['name'] for s in selected}!=set(args.scene):p.error('Some scene names were not discovered')
    if not selected:p.error('No scenes selected')
    inv=inventory(root,old,scenes)
    plan=dict(command=sys.argv if argv is None else argv,audit_root=str(root),out=str(out),selected_scenes=[s['name'] for s in selected],
        selected_frame_camera_rows=sum(s.get('expected_frame_camera_rows',0) for s in selected),stages=sorted(stages),workers=args.workers,
        execution='CPU surface/provenance stages; unchanged CUDA renderer for compatible Phase1 runs',manual_annotations_required=False)
    if args.dry_run:print(json.dumps(plan,indent=2));return 0
    converter_path=Path('/home/samuelemara/FreeArtGS/samuele/convert_real_recording.py')
    if sha(converter_path)!=old['config']['converter_sha256']:p.error('Exact original converter version unavailable; no silent replacement')
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','polars','pyarrow','matplotlib']}
    versions.update(opencv=cv2.__version__,python=sys.version)
    all_code=code_hashes()
    core_names={'__init__.py','cli.py','common.py','manifests.py','measurements.py','mask_annotation.py','registration_trace.py','surfaces.py','identifiability.py'}
    config=dict(core_code={k:v for k,v in all_code.items() if k in core_names},base_audit_code={f.name:sha(f) for f in Path(__file__).resolve().parents[1].glob('*.py')},versions=versions,
        original_manifest_sha256=sha(root/'manifest.json'),converter_sha256=sha(converter_path),seed=1731,
        method='automatic planar connected components; fixed-calibration conditional association; no production corrections',
        rigeuler_code=str(args.rigeuler_code.resolve()),rigeuler_results=str(args.rigeuler_results.resolve()),rigeuler_python=str(args.rigeuler_python.resolve()))
    manifest=out/'manifest.json'
    if out.exists():
        if not args.resume:p.error('Existing output requires --resume; no overwrite')
        if not manifest.is_file() or json_read(manifest)['signature']!=stable(config):p.error('Resume refused: follow-up code/environment/original audit changed')
    else:out.mkdir(parents=True)
    os.environ['MPLCONFIGDIR']=str(out/'.runtime/matplotlib');os.environ['MPLBACKEND']='Agg'
    for name in ['registration','coverage','geometry','calibration','timing','phase1','receipts','inputs','logs','annotation_review','.runtime']:(out/name).mkdir(parents=True,exist_ok=True)
    if not manifest.exists():atomic_json(manifest,dict(signature=stable(config),config=config))
    snapshot=out/'.runtime/source_snapshot'/stable(all_code)[:16];snapshot.mkdir(parents=True,exist_ok=True)
    for source in Path(__file__).parent.glob('*'):
        if source.suffix in {'.py','.html'} and not (snapshot/source.name).exists():shutil.copyfile(source,snapshot/source.name)
    plan.update(executed_code_sha256=all_code,source_snapshot=str(snapshot))
    atomic_json(out/'inventory.json',inv);atomic_json(out/'execution_plan.json',plan)
    print(f'Inventory: {len(scenes)} scenes; selected {len(selected)}, {plan["selected_frame_camera_rows"]} observations; stages {sorted(stages)}',flush=True)
    started=time.monotonic();synthetic=None;receipts={};phase1=None;failed=False
    atomic_json(out/'execution_receipt.json',dict(status='RUNNING',plan=plan,scene_results={}))
    if 'synthetic' in stages:
        synthetic=run_controls(out,load_converter(converter_path));print(f'Synthetic controls: {synthetic["passed"]} passed, {len(synthetic["failed"])} failed',flush=True)
        if synthetic['failed']:atomic_json(out/'execution_receipt.json',dict(status='FAIL',synthetic=synthetic));return 1
    core=stages&{'holes','coverage','verified-geometry'}
    if 'export-review-patches' in stages:core.add('verified-geometry')
    gpu_pool=None;gpu_future=None
    if 'phase1-cause' in stages:
        from .filter_attribution import run_phase1
        gpu_pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
        gpu_future=gpu_pool.submit(run_phase1,out,args.rigeuler_results,args.rigeuler_code,args.rigeuler_python,selected,args.resume)
    if core:
        jobs=[]
        for s in selected:
            signature=stable(dict(config=stable(config),scene=s['id'],metadata=scene_inputs(s),stages=sorted(core)))
            jobs.append((s,str(root),str(out),str(converter_path),sorted(core),signature,args.resume))
        if args.workers==1:
            for job in jobs:
                key,result=worker(job);receipts[key]=result;print(f'Completed {len(receipts)}/{len(jobs)}: {result.get("status")}',flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(worker,job) for job in jobs]
                for future in concurrent.futures.as_completed(futures):
                    key,result=future.result();receipts[key]=result;print(f'Completed {len(receipts)}/{len(jobs)}: {result.get("status")}',flush=True)
                    atomic_json(out/'execution_receipt.json',dict(status='RUNNING',plan=plan,completed_selected_scenes=len(receipts),scene_results=receipts))
        failed=any(r['status'] in {'FAIL','CONTRADICTED'} for r in receipts.values())
    # Additional stages consume committed surface outputs; no hidden human gate.
    if 'pose-vs-bias' in stages:calibration_report(out)
    if 'timing' in stages:
        timing=timing_report(out,selected);print('Timing:',timing,flush=True)
    if gpu_future is not None:
        phase1=gpu_future.result();gpu_pool.shutdown()
        failed|=phase1.get('status')=='FAIL'
    # Include earlier committed scene results when extending a selected pilot.
    for path in (out/'receipts').glob('*/*.json'):
        r=json_read(path);sid=path.stem
        if sid not in receipts:receipts[sid]=r['result']
    gates=generate(out,inv,receipts,stages,synthetic,phase1)
    record=dict(status='FAIL' if failed else 'EXECUTION_COMPLETE_SCIENTIFIC_IDENTIFICATION_PARTIAL',plan=plan,scene_results=receipts,gates=gates,
        elapsed_s=time.monotonic()-started,peak_parent_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        calibration_changed=False,production_changed=False,original_audit_manifest_sha256=sha(root/'manifest.json'))
    atomic_json(out/'execution_receipt.json',record);print(f'Report: {out/"ROOT_CAUSE_IDENTIFIABILITY_REPORT.md"}; failed={failed}',flush=True)
    return int(failed)


if __name__=='__main__':sys.exit(main())
