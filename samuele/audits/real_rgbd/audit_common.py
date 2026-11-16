"""Read-only audit primitives, explicit units, deterministic statistics and receipts."""
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import polars as pl

VERSION = 1
TABLES = ['frames','pairs','unordered_pairs','correspondences','patches','timesync','attribution']


@dataclass(frozen=True)
class Settings:
    chunk_frames: int = 16
    seed: int = 1731
    feature_sigma_px: float = .7
    descriptor_ratio: float = .75
    pixel_gate_sigma: float = 3.
    min_parallax_deg: float = 1.
    native_quantum_m: float = .001
    stationary_flow_px: float = .25
    stationary_rgb_difference_dn: float = 3.
    min_texture_dn: float = 5.
    patch_pixels: int = 64
    min_plane_points: int = 32
    depth_edge_m: float = .005
    lags: tuple = (-6,-3,-2,-1,0,1,2,3,6)
    calibration_translation_sweep_m: tuple = (.001,.003,.01)
    calibration_rotation_sweep_deg: tuple = (.05,.1,.3)
    bootstrap_draws: int = 200


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,Path):return str(value)
    if isinstance(value,(np.bool_,)):return bool(value)
    return value


def dumps(value):return json.dumps(clean(value),sort_keys=True,allow_nan=False)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def stable(value):return hashlib.sha256(dumps(value).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');temp.replace(path)


def file_record(path):
    path=Path(path)
    if not path.is_file():return dict(path=str(path),exists=False)
    stat=path.stat();return dict(path=str(path),exists=True,size_bytes=stat.st_size,sha256=sha(path))


def validate_records(records):
    for row in records:
        path=Path(row['path'])
        if path.is_file()!=row['exists']:raise ValueError(f'Resume input availability changed: {path}')
        if row['exists'] and (path.stat().st_size!=row['size_bytes'] or sha(path)!=row['sha256']):
            raise ValueError(f'Resume input hash changed: {path}')


def revision(root):
    def run(args):
        result=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True)
        return result.stdout.strip() if result.returncode==0 else None
    return dict(root=str(root),commit=run(['rev-parse','HEAD']),status=run(['status','--porcelain']))


def distribution(values):
    x=np.asarray(values,dtype=float).ravel();finite=np.isfinite(x);x=x[finite]
    result=dict(count=int(len(x)),nonfinite_count=int((~finite).sum()))
    for name in ['signed_mean_m','signed_median_m','mae_m','rmse_m','robust_sigma_m','abs_p50_m','abs_p90_m','abs_p95_m','abs_p99_m','min_m','max_m']:result[name]=None
    edges=np.r_[-np.inf,np.linspace(-1.,1.,401),np.inf]
    hist=np.histogram(x,bins=edges)[0]
    result.update(signed_histogram_edges_m=clean(edges),signed_histogram_counts=hist.tolist(),histogram_tail_convention='null first/last edges mean negative/positive infinity')
    if len(x):
        absolute=np.abs(x);median=np.median(x)
        result.update(signed_mean_m=float(x.mean()),signed_median_m=float(median),mae_m=float(absolute.mean()),rmse_m=float(np.sqrt(np.mean(x*x))),robust_sigma_m=float(1.4826*np.median(np.abs(x-median))),min_m=float(x.min()),max_m=float(x.max()))
        result.update({f'abs_p{q}_m':float(np.percentile(absolute,q)) for q in [50,90,95,99]})
    for mm in [2,5,10,20,50]:result[f'fraction_abs_gt_{mm}_mm']=float(np.mean(np.abs(x)>mm*.001)) if len(x) else None
    return result


def record(scene,frame,camera,status='measured',**values):
    return dict(dataset_id=scene['id'],dataset=scene['name'],split_group=scene.get('split_group'),frame_index=int(frame),source_frame_index=scene['source_by_frame'].get(int(frame)),camera=camera,calibration_sha256=scene.get('calibration_sha256'),status=status,**values)


def table(path,rows):
    """Stable nullable envelope; every record preserves typed metric JSON as well."""
    schema={'dataset_id':pl.String,'dataset':pl.String,'frame_index':pl.Int64,'source_frame_index':pl.Int64,'camera':pl.String,'target_camera':pl.String,'status':pl.String,'calibration_sha256':pl.String,'support_count':pl.Int64,'signed_median_m':pl.Float64,'mae_m':pl.Float64,'metrics_json':pl.String}
    schema['split_group']=pl.String
    for name in ['depth_valid_fraction','depth_zero_fraction','depth_nonfinite_fraction','native_invalid_fraction','registered_sampling_gap_candidate_fraction','depth_replay_max_abs_m','depth_replay_median_abs_m','depth_edge_to_rgb_edge_median_px','brightness_dn_mean','recorded_timestamp_s','nominal_timestamp_s','signed_mean_m','rmse_m','robust_sigma_m','abs_p50_m','abs_p90_m','abs_p95_m','abs_p99_m','u_i_px','v_i_px','u_j_px','v_j_px','signed_sampson_px','triangulation_angle_deg','reprojection_px','depth_i_m','depth_j_m','rgb_depth_i_m','rgb_depth_j_m','sensor_minus_rgb_i_m','sensor_minus_rgb_j_m','rgb_z_sigma_i_m','rgb_z_sigma_j_m','signed_projected_source_minus_target_m','rgb_motion_i_px','rgb_motion_j_px','plane_offset_m','incidence_deg','range_m','stationary_rgb_evidence_fraction','normal_angle_deg','static_epipolar_median_px','moving_epipolar_median_px','rgb_depth_edge_overlap_fraction']:
        schema[name]=pl.Float64
    for name in ['depth_replay_different_pixels','rgb_replay_different_pixels','png_quantization_different_pixels','source_valid_count','target_invalid_count','projected_in_bounds_count','native_plane_patch_candidates','lag_converted_frames','static_support','moving_support','patch_x_px','patch_y_px','patch_width_px','patch_height_px','keypoint_i','keypoint_j','match_index']:
        schema[name]=pl.Int64
    for name in ['native_status','depth_replay_status','rgb_replay_status','mask_quality','mechanism','evidence_grade','population','experiment','geometry_status','region_i','region_j','physical_patch_identity']:
        schema[name]=pl.String
    for name in ['geometry_accepted','same_surface_supported','edge_safe_i','edge_safe_j']:schema[name]=pl.Boolean
    envelope=[]
    for row in rows:
        envelope.append({**{k:row.get(k) for k in schema if k!='metrics_json'},'metrics_json':dumps(row)})
    df=pl.DataFrame(envelope,schema=schema,strict=False)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.parquet.tmp');df.write_parquet(tmp,compression='zstd');tmp.replace(path)
    return len(rows)


def join_partitions(paths,out):
    if not paths:table(out,[]);return
    import pyarrow as pa
    import pyarrow.parquet as pq
    # A Polars multi-file sink buffers across partitions. Read only one file
    # and write one small batch at a time, including the large metric JSON.
    temp=Path(str(out)+'.tmp')
    schema=pq.read_schema(paths[0])
    with pq.ParquetWriter(temp,schema,compression='zstd',use_dictionary=False,data_page_size=65536) as writer:
        pending=[];pending_bytes=0;pending_rows=0
        for path in paths:
            with pq.ParquetFile(path,pre_buffer=False) as source:
                if not source.schema_arrow.equals(schema,check_metadata=False):raise ValueError(f'Partition schema differs: {path}')
                for batch in source.iter_batches(batch_size=128,use_threads=False):
                    pending.append(batch);pending_bytes+=batch.nbytes;pending_rows+=batch.num_rows
                    # Bound data buffers without accumulating a huge footer of
                    # tiny row groups for the millions of small patch records.
                    if pending_bytes>=8*1024**2 or pending_rows>=1024:
                        writer.write_table(pa.Table.from_batches(pending),row_group_size=pending_rows)
                        pending.clear();pending_bytes=0;pending_rows=0
                    del batch
        if pending:writer.write_table(pa.Table.from_batches(pending),row_group_size=pending_rows)
    temp.replace(out)


def cluster_interval(values,clusters,seed=1731,draws=200):
    x=np.asarray(values,float);clusters=np.asarray(clusters);valid=np.isfinite(x);x=x[valid];clusters=clusters[valid]
    groups=[x[clusters==c] for c in sorted(set(clusters.tolist()))]
    if len(groups)<2:return dict(estimate=float(np.median(x)) if len(x) else None,ci95=None,clusters=len(groups),status='not_measurable_independent_cluster_ci')
    rng=np.random.default_rng(seed);medians=np.array([np.median(g) for g in groups])
    estimates=[float(np.median(medians[rng.integers(0,len(groups),len(groups))])) for _ in range(draws)]
    return dict(estimate=float(np.median(medians)),ci95=np.percentile(estimates,[2.5,97.5]).tolist(),clusters=len(groups),status='measured_scene_cluster_bootstrap')


def block_interval(values,block,seed=1731,draws=200):
    values=np.asarray(values,float);n=len(values)
    if n<max(6,block*2):return dict(ci95=None,status='insufficient_independent_temporal_blocks',block_frames=block)
    rng=np.random.default_rng(seed);result=[]
    for _ in range(draws):
        starts=rng.integers(0,max(1,n-block+1),int(np.ceil(n/block)))
        sample=np.concatenate([values[i:i+block] for i in starts])[:n]
        if np.isfinite(sample).any():result.append(float(np.nanmedian(sample)))
    return dict(ci95=np.percentile(result,[2.5,97.5]).tolist() if result else None,status='temporal_moving_block_bootstrap',block_frames=block)


def block_length(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if len(x)<6:return 1
    x=x-x.mean();variance=np.dot(x,x)
    if variance==0:return max(1,int(np.sqrt(len(x))))
    for lag in range(1,max(2,min(len(x)//3,100))):
        if np.dot(x[:-lag],x[lag:])/variance<np.exp(-1):return lag
    return max(1,min(len(x)//3,100))
