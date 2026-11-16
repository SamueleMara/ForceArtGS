"""Centered observed EE direction, force projections and epsilon-regularized eta."""
import csv
from pathlib import Path

import numpy as np


def trajectory_tangent(time,position,frame_index=None):
    """Exact observed EE direction (p[i+1]-p[i-1])/norm(...), no smoothing.

    No one-sided endpoint replacement or speed cutoff. A zero displacement,
    missing neighbor, or invalid/nonincreasing neighbor time is undefined.
    """
    t=np.asarray(time,float);p=np.asarray(position,float)
    if t.ndim!=1 or p.shape!=(len(t),3):raise ValueError('Expected time (N,) and positions (N,3)')
    n=len(t);frames=np.arange(n) if frame_index is None else np.asarray(frame_index)
    if frames.shape!=(n,) or not np.issubdtype(frames.dtype,np.integer):raise ValueError('Frame indices must be integers with shape (N,)')
    finite_dt=np.diff(t);finite_dt=finite_dt[np.isfinite(finite_dt)]
    if np.any(finite_dt<0):raise ValueError('Timestamps must be nondecreasing')
    tangent=np.full_like(p,np.nan);velocity=np.full_like(p,np.nan)
    status=np.full(n,'NO_CENTERED_NEIGHBORS',dtype='U40');support=np.zeros(n,int);span=np.zeros(n)
    if n>=3:
        interior=np.arange(1,n-1);delta=p[2:]-p[:-2];length=np.linalg.norm(delta,axis=1)
        time_good=np.isfinite(t[:-2])&np.isfinite(t[1:-1])&np.isfinite(t[2:])&(t[1:-1]>t[:-2])&(t[2:]>t[1:-1])
        adjacent=(frames[1:-1]-frames[:-2]==1)&(frames[2:]-frames[1:-1]==1)
        good=np.isfinite(delta).all(axis=1)&time_good&adjacent
        status[interior]='INVALID_CENTERED_NEIGHBORS'
        status[interior[~adjacent]]='MISSING_ADJACENT_FRAME'
        duplicate=(t[1:-1]==t[:-2])|(t[2:]==t[1:-1]);status[interior[duplicate]]='DUPLICATE_TIMESTAMP'
        ids=interior[good];support[ids]=3;span[ids]=t[ids+1]-t[ids-1]
        velocity[ids]=delta[good]/span[ids,None]
        status[ids]='ZERO_CENTERED_DISPLACEMENT'
        moving=good&(length>0);ids=interior[moving]
        tangent[ids]=delta[moving]/length[moving,None];status[ids]='MOVING'
    valid=status=='MOVING'
    return dict(velocity_m_s=velocity,speed_m_s=np.linalg.norm(velocity,axis=1),tangent=tangent,tangent_valid=valid,
        tangent_status=status,local_support_samples=support,local_support_span_s=span)


def decompose_force(force,tangent_info,force_epsilon_n=1e-8):
    if not np.isfinite(force_epsilon_n) or force_epsilon_n<=0:raise ValueError('Force epsilon must be finite and positive')
    force=np.asarray(force,float);t=tangent_info['tangent']
    if force.shape!=t.shape:raise ValueError('Force and tangent shapes must match')
    valid=tangent_info['tangent_valid']&np.isfinite(force).all(axis=1)
    signed=np.full(len(force),np.nan);signed[valid]=np.sum(force[valid]*t[valid],axis=1)
    parallel=signed[:,None]*t;perpendicular=np.full_like(force,np.nan);perpendicular[valid]=force[valid]-parallel[valid]
    norm=np.linalg.norm(force,axis=1);normal=np.linalg.norm(perpendicular,axis=1)
    angle=np.full(len(force),np.nan);nonzero=valid&(norm>1e-12)
    angle[nonzero]=np.degrees(np.arccos(np.clip(abs(signed[nonzero])/norm[nonzero],0,1)))
    eta=np.full(len(force),np.nan);eta[valid]=abs(signed[valid])/(norm[valid]+force_epsilon_n)
    power=np.full(len(force),np.nan);power[valid]=np.sum(force[valid]*tangent_info['velocity_m_s'][valid],axis=1)
    if valid.any():
        np.testing.assert_allclose(parallel[valid]+perpendicular[valid],force[valid],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(np.sum(perpendicular[valid]*t[valid],axis=1),0,atol=1e-10)
    return dict(decomposition_valid=valid,force_tangent_signed_N=signed,force_tangent_base_N=parallel,
        force_perpendicular_base_N=perpendicular,force_perpendicular_norm_N=normal,
        force_norm_N=norm,eta=eta,force_angle_to_unoriented_tangent_deg=angle,force_dot_ee_velocity_W=power)


def save_decomposition(folder,data,tangent,forces,labels,force_epsilon_n=1e-8):
    """Save every native sample, including ones with no RGB-D frame."""
    cases=[decompose_force(force,tangent,force_epsilon_n) for force in forces]
    shared={k:data[k] for k in ['frame_index','rgbd_frame_index','timestamp_s','tcp_position_m','ee_position_m','ee_rotvec_rad']}
    archive=dict(**shared,force_sensor_N=data['force_N'],**tangent,case_labels=np.array(labels),force_base_N=np.array(forces))
    for key in cases[0]:archive[key]=np.array([case[key] for case in cases])
    np.savez_compressed(folder/'force_decomposition.npz',**archive)
    fields=['native_frame','rgbd_frame','timestamp_s','case','tangent_status','decomposition_valid','speed_m_s','local_support_samples','local_support_span_s',
        'force_tangent_signed_N','force_perpendicular_norm_N','force_norm_N','force_angle_to_unoriented_tangent_deg','force_dot_ee_velocity_W','eta']
    vectors=['ee_position_m','tcp_position_m','force_sensor_N','force_base_N','velocity_m_s','tangent','force_tangent_base_N','force_perpendicular_base_N']
    fields+=[name+'_'+axis for name in vectors for axis in 'xyz']
    with (folder/'force_decomposition.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c,(label,case) in enumerate(zip(labels,cases)):
            for i in range(len(data['timestamp_s'])):
                row=dict(native_frame=data['frame_index'][i],rgbd_frame=data['rgbd_frame_index'][i],timestamp_s=data['timestamp_s'][i],case=label,
                    tangent_status=tangent['tangent_status'][i],decomposition_valid=case['decomposition_valid'][i])
                for name in fields[6:15]:row[name]=(tangent if name in tangent else case)[name][i]
                for name in vectors:
                    value=forces[c][i] if name=='force_base_N' else (data['force_N'][i] if name=='force_sensor_N' else (data if name in data else tangent if name in tangent else case)[name][i])
                    row.update({name+'_'+axis:v for axis,v in zip('xyz',value)})
                writer.writerow({k:('' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in row.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,constrained_layout=True)
    time=data['timestamp_s']
    for label,case in zip(labels,cases):
        axes[0].plot(time,case['force_tangent_signed_N'],lw=1,label=label+' signed tangential')
        axes[0].plot(time,case['force_perpendicular_norm_N'],'--',lw=.9,label=label+' perpendicular norm')
        axes[1].plot(time,case['force_angle_to_unoriented_tangent_deg'],lw=.8,label=label)
    axes[0].plot(time,np.linalg.norm(data['force_N'],axis=1),color='gray',alpha=.4,lw=.8,label='Total force norm')
    axes[0].axhline(0,color='black',lw=.5);axes[0].set_ylabel('Force (N)');axes[0].legend(fontsize=8)
    axes[1].set(ylabel='Angle to tangent (deg)',ylim=(0,90));axes[1].legend(fontsize=8)
    axes[2].plot(time,tangent['speed_m_s'],lw=.9,label='EE centered-difference speed')
    axes[2].fill_between(time,0,1,where=~tangent['tangent_valid'],color='gray',alpha=.15,transform=axes[2].get_xaxis_transform(),label='Tangent undefined')
    axes[2].set(xlabel='Native Parquet time (s)',ylabel='EE speed (m/s)');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(folder.name+'\nExact centered EE direction; positive tangential force follows motion. Not object-joint GT.',fontsize=10)
    fig.savefig(folder/'force_decomposition.png',dpi=130);plt.close(fig)
    return cases
