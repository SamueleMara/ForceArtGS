import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from audit_common import atomic_json,clean,file_record,sha,stable,validate_records

SEED=1731


def json_read(path):return json.loads(Path(path).read_text())


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in rows for k in r}) or ['status']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v),allow_nan=False) if isinstance(v,(list,dict,np.ndarray)) else clean(v) for k,v in r.items()})


def rows_write(path,rows):
    """One bounded scene/frame chunk; nested diagnostics preserve null semantics."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        empty=path.with_suffix('.empty.json');atomic_json(empty,{'rows':0});return empty
    fields=sorted({k for r in rows for k in r})
    data=pa.Table.from_pylist([{k:clean(r.get(k)) for k in fields} for r in rows]);temp=Path(str(path)+'.tmp')
    pq.write_table(data,temp,compression='zstd',row_group_size=128)
    if path.exists():
        if sha(path)!=sha(temp):raise ValueError(f'Regenerated committed output differs: {path}')
        temp.unlink()
    else:temp.replace(path)
    return path


def iter_rows(path):
    with pq.ParquetFile(path) as f:
        for batch in f.iter_batches(batch_size=128,use_threads=False):yield from batch.to_pylist()


def intervals(values,clusters,seed=SEED):
    """Equal independent-cluster median; never pixel-IID confidence intervals."""
    from audit_common import cluster_interval
    return cluster_interval(values,clusters,seed,200)


def population(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if not len(x):return dict(count=0,signed_median_m=None,mae_m=None,abs_p95_m=None)
    return dict(count=len(x),signed_median_m=float(np.median(x)),signed_mean_m=float(x.mean()),mae_m=float(np.mean(np.abs(x))),abs_p95_m=float(np.percentile(np.abs(x),95)),
        sum_signed_m=float(x.sum()),sum_abs_m=float(np.abs(x).sum()),sum_sq_m2=float(np.sum(x*x)))


def family(scene):
    name=scene['name'];return name.split('_episode_')[0] if name.startswith('final_') else 'legacy:'+name


def point_cloud(depth,K,pose):
    yy,xx=np.indices(depth.shape);ok=np.isfinite(depth)&(depth>0)
    p=np.c_[xx[ok],yy[ok],np.ones(ok.sum())]@np.linalg.inv(K).T*depth[ok,None]
    return p@pose[:3,:3].T+pose[:3,3],ok


def path_inside(root,path):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError(f'Path escapes isolated output: {path}')
    return path


def code_hashes():
    return {p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*')) if p.suffix in {'.py','.html'}}


def stage_run(out,scene,stage,signature,resume,fn):
    dest=Path(out)/'receipts'/stage/(scene['id']+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_file():
        old=json_read(dest)
        if not resume:raise FileExistsError(f'Use --resume for committed stage: {dest}')
        if old['signature']!=signature:raise ValueError(f'Stage inputs/code changed: {dest}')
        validate_records(old['outputs'])
        for r in old['outputs']:
            if Path(r['path']).parent==Path(out)/'inputs':validate_records(json_read(r['path']))
        print(f'RESUME {stage}: {scene["name"]}',flush=True);return old['result']
    started=time.monotonic()
    result,outputs=fn()
    for p in outputs:path_inside(out,p)
    atomic_json(dest,dict(signature=signature,result=result,outputs=[file_record(p) for p in outputs],elapsed_s=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    print(f'{stage}: {scene["name"]}: {result.get("status")}',flush=True)
    return result
