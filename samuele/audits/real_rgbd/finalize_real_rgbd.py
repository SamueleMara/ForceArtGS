#!/usr/bin/env python3
"""Recover reports from a completed audit without replaying any measurements."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys

sys.dont_write_bytecode=True
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('POLARS_ROW_GROUP_PREFETCH_SIZE','1')
os.environ.setdefault('POLARS_RECORD_BATCH_PREFETCH_SIZE','1')
sys.path.insert(0,str(Path(__file__).resolve().parent))

from audit_common import Settings,TABLES,atomic_json,sha,stable,validate_records
from audit_reporting import summarize


def completed_receipts(root,manifest):
    """Accept only complete, hash-verified partitions; inputs remain untouched."""
    receipts={}
    for scene in manifest['scenes']:
        if manifest['mode']=='pilot' and not (root/'scenes'/scene['id']/'receipt.json').is_file():continue
        path=root/'scenes'/scene['id']/'receipt.json'
        detail=json.loads(path.read_text())
        if not detail.get('processing_complete') or detail.get('dataset')!=scene['name']:
            raise ValueError(f'Incomplete or mismatched scene receipt: {path}')
        counts={name:0 for name in TABLES};expected=set()
        chunks=sorted((root/'partitions'/scene['id']).glob('*/receipt.json'))
        if not chunks:raise ValueError(f'No committed chunks: {scene["name"]}')
        for chunk in chunks:
            receipt=json.loads(chunk.read_text())
            outputs=receipt['outputs']
            required={chunk.parent/(name+'.parquet') for name in TABLES}
            actual={Path(r['path']) for r in outputs}
            if actual!=required or any(not p.resolve().is_relative_to(root) for p in actual):
                raise ValueError(f'Unexpected partition paths: {chunk}')
            validate_records(outputs)
            expected.update(actual)
            for name in TABLES:counts[name]+=receipt['counts'][name]
        actual=set((root/'partitions'/scene['id']).glob('*/*.parquet'))
        if actual!=expected:raise ValueError(f'Uncommitted or missing partitions: {scene["name"]}')
        if any(counts[k]!=detail['counts'][k] for k in TABLES):
            raise ValueError(f'Partition/scene counts disagree: {scene["name"]}')
        for name in ['frames','pairs']:
            target=detail['expected_selected_frame_rows' if name=='frames' else 'expected_selected_pair_rows']
            if counts[name]!=target:raise ValueError(f'Incomplete {name} accounting: {scene["name"]}')
        receipts[scene['id']]={k:detail.get(k) for k in ['status','processing_complete','dataset','mode','frame_rows','pair_rows','counts','processed_content_fingerprint']}
        receipts[scene['id']]['detail_receipt']=str(path)
        receipts[scene['id']]['detail_receipt_sha256']=sha(path)
        print(f'Verified {scene["name"]}: {counts["frames"]} frame/camera rows, {len(chunks)} chunks',flush=True)
    return receipts


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root',required=True,type=Path)
    args=parser.parse_args(argv);root=args.audit_root.expanduser().resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('mode') not in {'full','pilot'}:raise ValueError('Only completed measurements can be finalized')
    if stable(manifest['config'])!=manifest['config_sha256']:raise ValueError('Invalid original config hash')
    for scene in manifest['scenes']:
        scene['source_by_frame']={int(k):v for k,v in scene.get('source_by_frame',{}).items()}
    for source in manifest['config']['converted_roots']+manifest['config']['raw_roots']:
        source=Path(source).resolve()
        if root==source or root.is_relative_to(source) or source.is_relative_to(root):raise ValueError('Audit overlaps input trees')
    for name in ['scenes','partitions','figures','.runtime','failure_gallery']:
        if not (root/name).resolve().is_relative_to(root):raise ValueError('Audit subfolder escapes selected root')
    os.environ['MPLCONFIGDIR']=str(root/'.runtime/matplotlib');os.environ['MPLBACKEND']='Agg'
    receipts=completed_receipts(root,manifest)
    settings=Settings(**manifest['config']['settings'])
    evidence=dict(status='running',original_manifest_sha256=sha(root/'manifest.json'),
        analysis_config_sha256=manifest['config_sha256'],new_measurements=False,
        input_scope='saved committed measurement partitions; original raw/converted inputs not revalidated',
        reporting_code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        reporting_versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','polars','matplotlib','pyarrow']})
    atomic_json(root/'finalization_receipt.json',evidence)
    summaries=summarize(root,manifest,receipts,settings)
    failed=any(r['status']!='complete' for r in summaries)
    atomic_json(root/'run_receipt.json',dict(mode=manifest['mode'],scenes_discovered=len(manifest['scenes']),
        scenes_selected=len(receipts),scenes_complete=sum(r['status']=='complete' for r in receipts.values()),
        scenes_failed=sum(r['status']!='complete' for r in receipts.values()),results=receipts,report_recovery=True))
    evidence.update(status='complete',measurements_have_failures=failed,process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    atomic_json(root/'finalization_receipt.json',evidence)
    print(f'Finalized existing {manifest["mode"]} audit: {root}; measurement failures={failed}',flush=True)
    return int(failed)


if __name__=='__main__':sys.exit(main())
