"""E5: real timestamp provenance and fixed-population image-content delays."""
from pathlib import Path
import re
import numpy as np
import polars as pl

from .common import csv_write,json_read,population


def fixed_population_delay(times,a,b,lags,*,identity_confirmed=False,occluded=None):
    """Delay sign: compare a(t) with b(t+lag), fixed identities and population.

    Demeaned position removes constant spatial bias. Constant-velocity motion
    cannot separate that bias from delay and must abstain.
    """
    t=np.asarray(times,float);a=np.asarray(a,float);b=np.asarray(b,float)
    if not identity_confirmed:return dict(status='TIMING_UNOBSERVABLE',reason='NO_FIXED_PHYSICAL_TRACK_POPULATION')
    if len(t)<20 or np.any(np.diff(t)<=0):return dict(status='TIMING_UNOBSERVABLE',reason='INSUFFICIENT_OR_NONMONOTONE_TIME_SUPPORT')
    good=np.isfinite(a)&np.isfinite(b)
    if occluded is not None:good&=~np.asarray(occluded,bool)
    margin=max(abs(float(x)) for x in lags)
    common=good&(t>=t[0]+margin)&(t<=t[-1]-margin)
    for lag in lags:
        k=np.searchsorted(t,t+lag).clip(1,len(t)-1);common&=good[k]&good[k-1]
    ids=np.flatnonzero(common)
    if len(ids)<16:return dict(status='TIMING_UNOBSERVABLE',reason='INSUFFICIENT_FIXED_UNOCCLUDED_SUPPORT')
    velocity=np.diff(a[ids])/np.diff(t[ids])
    if np.std(velocity)<1e-3 or np.ptp(a[ids])<.02 or np.ptp(b[ids])<.02:return dict(status='TIMING_UNOBSERVABLE',reason='STATIC_OR_CONSTANT_VELOCITY_DELAY_BIAS_DEGENERACY')
    train=ids[:len(ids)//2];held=ids[len(ids)//2:]
    def cost(index,lag):
        r=a[index]-np.interp(t[index]+lag,t,b);r-=np.median(r)
        return float(np.mean(np.abs(r)))
    curves={float(lag):cost(train,lag) for lag in lags};chosen=min(curves,key=lambda l:curves[l])
    held_curve={float(lag):cost(held,lag) for lag in lags};held_best=min(held_curve,key=lambda l:held_curve[l])
    ordered=sorted(curves.values());unique=len(ordered)>1 and ordered[1]-ordered[0]>1e-5
    supported=chosen==held_best and unique and 0. in curves and curves[0.]>curves[chosen]+1e-5 and held_curve[0.]>held_curve[chosen]+1e-5
    return dict(status='CONDITIONAL_IMAGE_CONTENT_DELAY' if supported else 'TIMING_UNOBSERVABLE',
        chosen_target_lag_s=chosen if supported else None,training_cost=curves,heldout_cost=held_curve,
        fixed_population_count=len(ids),train_count=len(train),heldout_count=len(held),
        confidence='independent contiguous motion holdout; no hardware-clock inference; not a production shift',
        applied_shift=False,reason='heldout_unique_motion_support' if supported else 'NO_UNIQUE_HELDOUT_DELAY')


def timestamp_inventory(scene):
    files=[Path(scene['path'])/'metadata/raw'/f'episode_{scene["conversion"]["episode"]:06d}.parquet']
    if scene.get('source'):files.append(Path(scene['source'])/'data/chunk-000'/f'episode_{scene["conversion"]["episode"]:06d}.parquet')
    rows=[]
    for p in files:
        if not p.is_file():rows.append(dict(scene=scene['name'],path=str(p),status='SOURCE_UNAVAILABLE',field=None));continue
        schema=pl.read_parquet_schema(p)
        for name,dtype in schema.items():
            if not re.search(r'time|stamp|frame|index|callback|receive|cache',name,re.I):continue
            if name in {'timestamp','frame_index','episode_index','index','task_index'}:kind='NOMINAL_OR_RECORDER_NOT_SENSOR_CLOCK'
            elif re.search(r'hardware|sensor|header|exposure|capture',name,re.I):kind='SENSOR_TIMESTAMP_CANDIDATE_CLOCK_DOMAIN_UNVERIFIED'
            else:kind='OTHER_TIME_OR_INDEX_CANDIDATE'
            values=None
            if str(dtype).startswith(('Float','Int','UInt')):
                data=pl.read_parquet(p,columns=[name])[name].drop_nulls().to_numpy()
                if len(data):values=dict(count=len(data),min=float(data.min()),max=float(data.max()),nonmonotone_steps=int((np.diff(data)<0).sum()),repeat_steps=int((np.diff(data)==0).sum()))
            rows.append(dict(scene=scene['name'],path=str(p),field=name,dtype=str(dtype),status=kind,statistics=values,
                clock_domain='UNVERIFIED',units='UNVERIFIED unless explicitly recorded; never guessed from magnitude'))
    frames=scene['source_by_frame'];stride=sorted(set(np.diff([frames[f] for f in scene['frames']]).tolist()))
    rows.append(dict(scene=scene['name'],field='processed_to_source_frame_mapping',status='MEASURED_INDEX_MAPPING_NOT_SYNCHRONIZATION',source_index_steps=stride))
    return rows
