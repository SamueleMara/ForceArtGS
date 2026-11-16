#!/usr/bin/env python3
"""Plot EE-aligned measured forces in base_link, rooted at calibrated TCP positions."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def force_to_base(rotvec,force):
    rotvec=np.asarray(rotvec,float);force=np.asarray(force,float)
    if rotvec.shape!=force.shape or force.ndim!=2 or force.shape[1]!=3:raise ValueError('Rotation vectors and forces must both have shape (N,3)')
    out=np.full_like(force,np.nan);good=np.isfinite(rotvec).all(axis=1)&np.isfinite(force).all(axis=1)
    if good.any():out[good]=Rotation.from_rotvec(rotvec[good]).apply(force[good])
    return out


def select_frames(data,frame_space,frame_range=None,window_s=None,every=10):
    """Select a continuous native-time interval; origins use exact chosen frame IDs."""
    if every<1:raise ValueError('Arrow spacing must be a positive number of frames')
    key='frame_index' if frame_space=='native' else 'rgbd_frame_index';ids=data[key];t=data['timestamp_s']
    registered=ids>=0
    if frame_range is not None:
        start,end=frame_range
        if start<0 or end<start or start not in ids[registered] or end not in ids[registered]:raise ValueError('Frame-range endpoints must exist and be in increasing order')
        first=int(np.flatnonzero(ids==start)[0]);last=int(np.flatnonzero(ids==end)[0])
        selected=(np.arange(len(t))>=first)&(np.arange(len(t))<=last)
    elif window_s is not None:
        lo,hi=window_s
        if not np.isfinite([lo,hi]).all() or hi<lo:raise ValueError('Invalid interaction timestamps')
    else:lo,hi=float(t[0]),float(t[-1])
    if frame_range is None:selected=(t>=lo)&(t<=hi)
    available=np.flatnonzero(selected&registered)
    if not selected.any() or not len(available):raise ValueError('No samples in the selected frame/time interval')
    origin=int(ids[available[0]]);requested=selected&registered&((ids-origin)%every==0)
    return selected,requested


def prepare(data,frame_space,frame_range,window_s,every,minimum_force,mode,length,scale):
    selected,requested=select_frames(data,frame_space,frame_range,window_s,every)
    force=force_to_base(data['ee_rotvec_rad'],data['force_N']);magnitude=np.linalg.norm(force,axis=1)
    valid=np.isfinite(data['tcp_position_m']).all(axis=1)&np.isfinite(force).all(axis=1)
    drawn=requested&valid&(magnitude>=minimum_force)&(magnitude>1e-12)
    vectors=np.full_like(force,np.nan)
    if mode=='direction':vectors[drawn]=force[drawn]/magnitude[drawn,None]*length
    else:vectors[drawn]=force[drawn]*scale
    return dict(selected=selected,requested=requested,drawn=drawn,force_base_N=force,force_norm_N=magnitude,display_vectors_m=vectors)


def save_vectors(folder,data,prepared):
    selected=prepared['selected'];drawn=prepared['drawn']
    np.savez_compressed(folder/'force_vectors.npz',source_frame=data['frame_index'][selected],rgbd_frame=data['rgbd_frame_index'][selected],
        timestamp_s=data['timestamp_s'][selected],tcp_position_m=data['tcp_position_m'][selected],force_ee_N=data['force_N'][selected],
        force_base_N=prepared['force_base_N'][selected],force_norm_N=prepared['force_norm_N'][selected],
        requested_arrow_origin=prepared['requested'][selected],drawn_arrow_origin=drawn[selected],display_vectors_m=prepared['display_vectors_m'][selected])
    with (folder/'force_vectors.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['native_frame','rgbd_frame','timestamp_s','tcp_x_m','tcp_y_m','tcp_z_m',
            'force_ee_x_N','force_ee_y_N','force_ee_z_N','force_base_x_N','force_base_y_N','force_base_z_N','force_norm_N','requested_arrow_origin','drawn_arrow_origin'])
        for i in np.flatnonzero(selected):
            values=[data['frame_index'][i],data['rgbd_frame_index'][i],data['timestamp_s'][i],*data['tcp_position_m'][i],*data['force_N'][i],
                *prepared['force_base_N'][i],prepared['force_norm_N'][i],bool(prepared['requested'][i]),bool(drawn[i])]
            writer.writerow([v if np.isfinite(v) else '' for v in values])


def arrow_geometry(root,vector,head_fraction=.15):
    """Thin shaft plus an explicit small tip, independent of Plotly cone sizing."""
    root=np.asarray(root,float);vector=np.asarray(vector,float);length=float(np.linalg.norm(vector))
    if length<=0 or not 0<head_fraction<1:raise ValueError('Arrow length and head fraction must be positive; head fraction must be less than one')
    direction=vector/length;reference=np.eye(3)[np.argmin(np.abs(direction))]
    first=np.cross(direction,reference);first/=np.linalg.norm(first);second=np.cross(direction,first)
    head_length=length*head_fraction;tip=root+vector;neck=tip-head_length*direction
    theta=np.arange(8)*2*np.pi/8
    ring=neck+.35*head_length*(np.cos(theta)[:,None]*first+np.sin(theta)[:,None]*second)
    vertices=np.vstack([tip,ring]);faces=np.array([[0,1+i,1+(i+1)%8] for i in range(8)])
    return np.vstack([root,neck]),vertices,faces


def initial_camera(points,vectors):
    centered=points-points.mean(0)
    direction=np.linalg.svd(centered,full_matrices=False)[2][0] if np.linalg.norm(centered)>1e-12 else np.array([1.,0,0])
    force=np.mean(vectors/np.maximum(np.linalg.norm(vectors,axis=1)[:,None],1e-12),axis=0) if len(vectors) else np.array([0.,0,1.])
    normal=np.cross(direction,force)
    if np.linalg.norm(normal)<1e-3:normal=np.cross(direction,np.eye(3)[np.argmin(np.abs(direction))])
    normal/=np.linalg.norm(normal)
    if normal[2]<0:normal=-normal
    eye=1.35*normal+.49*np.cross(normal,direction)+.15*direction
    up=np.array([0.,1,0]) if abs(eye[2]/np.linalg.norm(eye))>.9 else np.array([0.,0,1.])
    return dict(eye=dict(zip('xyz',eye)),up=dict(zip('xyz',up)),projection=dict(type='orthographic'))


def interactive_figure(name,data,p,description,mode,length,scale,head_fraction=.15):
    import plotly.graph_objects as go
    from matplotlib.colors import Normalize,to_hex
    from matplotlib import colormaps
    selected=p['selected'];origins=np.flatnonzero(p['drawn']);coords=data['tcp_position_m'][selected]
    roots=data['tcp_position_m'][origins];vectors=p['display_vectors_m'][origins];magnitude=p['force_norm_N']
    finite=magnitude[selected];finite=finite[np.isfinite(finite)];vmax=max(float(finite.max()) if len(finite) else 0.,.001)
    colors=colormaps['viridis'](Normalize(0,vmax)(magnitude[origins]))
    valid=coords[np.isfinite(coords).all(axis=1)];extent=np.concatenate([valid,roots+vectors]) if len(roots) else valid
    if not len(valid):raise ValueError('No finite TCP positions to display')
    padding=max(float(np.ptp(extent,axis=0).max())*.08,.005)
    extra=np.c_[data['frame_index'][selected],data['rgbd_frame_index'][selected],data['timestamp_s'][selected],p['force_base_N'][selected]]
    figure=go.Figure()
    figure.add_trace(go.Scatter3d(x=coords[:,0],y=coords[:,1],z=coords[:,2],mode='lines+markers',name='TCP trajectory',
        line=dict(color='#1e3a5f',width=5),marker=dict(size=2.5,color=magnitude[selected],colorscale='Viridis',cmin=0,cmax=vmax,colorbar=dict(title='Force (N)',thickness=16,len=.7)),
        customdata=extra,hovertemplate='Native %{customdata[0]:.0f}; RGB-D %{customdata[1]:.0f}<br>t=%{customdata[2]:.3f}s<br>TCP=(%{x:.4f}, %{y:.4f}, %{z:.4f}) m<br>F_base=(%{customdata[3]:.3f}, %{customdata[4]:.3f}, %{customdata[5]:.3f}) N<extra></extra>'))
    for index,(i,root,vector,color) in enumerate(zip(origins,roots,vectors,colors)):
        shaft,head,faces=arrow_geometry(root,vector,head_fraction);c=to_hex(color)
        hover=f'Native {data["frame_index"][i]}; RGB-D {data["rgbd_frame_index"][i]}<br>|F|={magnitude[i]:.3f} N<extra></extra>'
        figure.add_trace(go.Scatter3d(x=shaft[:,0],y=shaft[:,1],z=shaft[:,2],mode='lines',line=dict(color=c,width=3),
            name='Force arrows',legendgroup='forces',showlegend=index==0,hovertemplate=hover))
        figure.add_trace(go.Mesh3d(x=head[:,0],y=head[:,1],z=head[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color=c,
            flatshading=True,lighting=dict(ambient=1,diffuse=0,specular=0),name='Force tip',legendgroup='forces',showlegend=False,hovertemplate=hover))
    for index,color,label in [(0,'#16a34a','Start'),(-1,'#dc2626','End')]:
        point=coords[index]
        figure.add_trace(go.Scatter3d(x=[point[0]],y=[point[1]],z=[point[2]],mode='markers',marker=dict(size=5,color=color),name=label,hoverinfo='name'))
    caption=f'Direction arrows: {length:g} m; color = force norm (N)' if mode=='direction' else f'Arrow scale: {scale:g} m/N; color = force norm (N)'
    axes=[dict(title=title,range=[extent[:,j].min()-padding,extent[:,j].max()+padding],showbackground=False,gridcolor='#e2e8f0',zeroline=False) for j,title in enumerate(['base_link x (m)','y (m)','z (m)'])]
    ranges=np.ptp(extent,axis=0)+2*padding
    # Equal metres on every axis, with a bounded box diagonal so thin, long
    # trajectories do not overflow Plotly's orthographic viewport.
    aspect=dict(zip('xyz',2.4*ranges/np.linalg.norm(ranges)))
    camera=initial_camera(valid,vectors)
    views=[dict(label='Side view',method='relayout',args=[{'scene.camera':camera}])]
    for label,eye,up in [('XY',[0,0,1.5],[0,1,0]),('XZ',[0,-1.5,0],[0,0,1]),('YZ',[1.5,0,0],[0,0,1])]:
        views.append(dict(label=label,method='relayout',args=[{'scene.camera':dict(eye=dict(zip('xyz',eye)),up=dict(zip('xyz',up)),projection=dict(type='orthographic'))}]))
    figure.update_layout(template='plotly_white',title=dict(text=name+'<br>'+description+'<br>'+caption,font=dict(size=13)),
        scene=dict(xaxis=axes[0],yaxis=axes[1],zaxis=axes[2],aspectmode='manual',aspectratio=aspect,bgcolor='white',camera=camera),
        updatemenus=[dict(buttons=views,x=.98,y=1.08,xanchor='right',yanchor='top')],
        legend=dict(orientation='h',x=0,y=1.04),margin=dict(t=125,b=30,l=15,r=90))
    return figure


def write_interactive(folder,name,data,p,description,mode,length,scale,head_fraction=.15):
    figure=interactive_figure(name,data,p,description,mode,length,scale,head_fraction)
    figure.write_html(str(folder/'force_directions.html'),include_plotlyjs=True,auto_open=False,default_height='100vh',config=dict(responsive=True,displaylogo=False,scrollZoom=True))


def draw(folder,name,data,p,description,mode,length,scale,write_html=True,head_fraction=.15):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,to_hex
    selected=p['selected'];indices=np.flatnonzero(selected);origins=np.flatnonzero(p['drawn'])
    points=data['tcp_position_m'][selected];roots=data['tcp_position_m'][origins];vectors=p['display_vectors_m'][origins]
    magnitude=p['force_norm_N'];finite=magnitude[selected];finite=finite[np.isfinite(finite)];vmax=max(float(finite.max()) if len(finite) else 0.,.001)
    norm=Normalize(0,vmax);cmap=plt.get_cmap('viridis');colors=cmap(norm(magnitude[origins]))
    valid_points=points[np.isfinite(points).all(axis=1)]
    if not len(valid_points):raise ValueError('No finite TCP positions to display')
    extent=np.concatenate([valid_points,roots+vectors]) if len(roots) else valid_points
    center=(extent.max(0)+extent.min(0))/2;radius=max(float(np.ptp(extent,axis=0).max())*.6,.01)
    caption=f'Equal-length direction arrows: {length:g} m; color = measured force norm (N)' if mode=='direction' else f'Arrow length scale: {scale:g} m/N; color = measured force norm (N)'
    caption+='; recorded force sign; F_base = R_base_EE F_EE'
    fig=plt.figure(figsize=(14,11));ax=fig.add_subplot(2,2,1,projection='3d');ax.plot(*points.T,color='steelblue',lw=1.,label='TCP trajectory')
    for root,vector,color in zip(roots,vectors,colors):ax.quiver(*root,*vector,pivot='tail',length=1,normalize=False,color=color,arrow_length_ratio=.25,lw=1.2)
    ax.scatter(*points[[0,-1]].T,color=['green','red'],s=30)
    ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),xlabel='base_link x (m)',ylabel='y (m)',zlabel='z (m)',title='3D force arrows rooted at TCP')
    ax.set_box_aspect((1,1,1))
    for panel,(a,b) in enumerate([(0,1),(0,2),(1,2)],start=2):
        ax2=fig.add_subplot(2,2,panel);ax2.plot(points[:,a],points[:,b],color='steelblue',lw=1.)
        if len(roots):ax2.quiver(roots[:,a],roots[:,b],vectors[:,a],vectors[:,b],color=colors,angles='xy',scale_units='xy',scale=1,pivot='tail',width=.004)
        ax2.scatter(points[[0,-1],a],points[[0,-1],b],color=['green','red'],s=25)
        ax2.set(xlim=(center[a]-radius,center[a]+radius),ylim=(center[b]-radius,center[b]+radius),xlabel='base_link '+'xyz'[a]+' (m)',ylabel='base_link '+'xyz'[b]+' (m)',title='xyz'[a].upper()+'–'+'xyz'[b].upper()+' projection')
        ax2.set_aspect('equal');ax2.grid(alpha=.25)
    fig.suptitle(name+'\n'+description+'\n'+caption,fontsize=10)
    fig.subplots_adjust(top=.86,bottom=.08,left=.06,right=.88,wspace=.25,hspace=.28)
    bar=fig.add_axes([.92,.2,.02,.55]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=bar,label='Measured force norm (N)')
    if not len(origins):fig.text(.1,.015,'No arrows meet the selected force/validity threshold; trajectory is retained.')
    fig.savefig(folder/'force_directions.png',dpi=150);plt.close(fig)
    if write_html:write_interactive(folder,name,data,p,description,mode,length,scale,head_fraction)


def refresh_html(folder,output,args):
    """Refresh presentation from saved vectors; preserve sampling and numerical outputs."""
    target=output/folder.name;manifest=target/'manifest.json';archive=target/'force_vectors.npz'
    for file in [manifest,archive,target/'force_directions.html']:
        if not file.resolve().is_relative_to(output):raise ValueError('Existing plot file escapes output root')
    if not args.overwrite:raise ValueError('--html-only requires --overwrite')
    meta=json.loads(manifest.read_text())
    summary=json.loads((folder/'summary.json').read_text())
    if meta.get('tcp_calibration_valid') is False or summary.get('tcp_calibration',{}).get('tcp_calibration_valid') is False:
        raise ValueError('Uncalibrated sensor mounting: cached force/TCP vectors cannot be republished')
    with np.load(archive,allow_pickle=False) as z:
        data=dict(frame_index=z['source_frame'],rgbd_frame_index=z['rgbd_frame'],timestamp_s=z['timestamp_s'],tcp_position_m=z['tcp_position_m'])
        p=dict(selected=np.ones(len(z['source_frame']),bool),drawn=z['drawn_arrow_origin'],force_base_N=z['force_base_N'],force_norm_N=z['force_norm_N'],display_vectors_m=z['display_vectors_m'])
    registered=data['frame_index' if meta['frame_space']=='native' else 'rgbd_frame_index'];registered=registered[registered>=0]
    description=f'{meta["frame_space"]} frames {registered[0]}–{registered[-1]}; arrow every {meta["arrow_every"]} frames; {meta["interval_source"]}'
    if meta.get('interval_status'):description+='; '+meta['interval_status']
    write_interactive(target,folder.name,data,p,description,meta['arrow_mode'],meta['arrow_length_m'],meta['meters_per_N'],args.head_fraction)
    meta.update(html=True,html_style='thin_shafts_small_mesh_tips',html_head_fraction=args.head_fraction,html_renderer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest.write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    return {k:meta[k] for k in ['scene','status','interval_source','interval_status','selected_native_frames','selected_samples','requested_origins','drawn_arrows']}


def process(folder,output,args):
    archive=folder/'measurements.npz';summary_path=folder/'summary.json';summary=json.loads(summary_path.read_text())
    if summary.get('tcp_calibration',{}).get('tcp_calibration_valid') is False:
        raise ValueError('Uncalibrated sensor mounting: establish EE-to-TCP and sensor-to-EE transforms first')
    axes=summary.get('tcp_calibration',{}).get('wrench_frame',{}).get('axes')
    if axes is None and summary.get('wrench_coordinate_frame','').startswith('EE_ALIGNED_AXES_USER_CONFIRMED'):axes='ee'
    if axes!='ee':
        raise ValueError('EE-aligned force axes must be confirmed in measurement metadata')
    with np.load(archive,allow_pickle=False) as z:data={k:z[k] for k in ['frame_index','rgbd_frame_index','timestamp_s','tcp_position_m','ee_rotvec_rad','force_N']}
    window=None;interval_source='whole_episode';interval_status=None;source_result=None
    if args.interaction_root:
        source_result=args.interaction_root/folder.name/'result.json';result=json.loads(source_result.read_text())
        if result['scene']!=folder.name:raise ValueError('Interaction result refers to a different scene')
        window=result['interval']['selected_window_s'];interval_source=result['interval']['selected_interval_source'];interval_status=result['interval']['status']
        if window is None:return dict(scene=folder.name,status='SKIPPED_NO_INTERACTION_INTERVAL')
        expected=result['provenance']['tcp_archive_sha256']
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:raise ValueError('TCP archive differs from the interval analysis; regenerate interaction bounds')
    elif args.frame_range:interval_source='explicit_'+args.frame_space+'_frames'
    p=prepare(data,args.frame_space,args.frame_range,window,args.arrow_every,args.min_force_N,args.arrow_mode,args.arrow_length_m,args.meters_per_N)
    target=output/folder.name
    if not target.resolve().is_relative_to(output):raise ValueError('Scene output escapes output root')
    files=['force_directions.png','force_directions.html','force_vectors.npz','force_vectors.csv','manifest.json']
    for name in files:
        if not (target/name).resolve().is_relative_to(output):raise ValueError('Plot output escapes output root')
    if not args.overwrite and any((target/f).exists() for f in files):raise FileExistsError('Existing force plot; use --overwrite')
    target.mkdir(parents=True,exist_ok=True)
    selected=np.flatnonzero(p['selected']);ids=data['frame_index' if args.frame_space=='native' else 'rgbd_frame_index'];registered=selected[ids[selected]>=0]
    description=f'{args.frame_space} frames {ids[registered[0]]}–{ids[registered[-1]]}; arrow every {args.arrow_every} frames; {interval_source}'
    if interval_status:description+='; '+interval_status
    save_vectors(target,data,p);draw(target,folder.name,data,p,description,args.arrow_mode,args.arrow_length_m,args.meters_per_N,not args.no_html,args.head_fraction)
    if args.no_html and (target/'force_directions.html').exists():(target/'force_directions.html').unlink()
    meta=dict(scene=folder.name,status='PLOTTED',interval_source=interval_source,interval_status=interval_status,frame_space=args.frame_space,arrow_every=args.arrow_every,
        selected_native_frames=[int(data['frame_index'][selected[0]]),int(data['frame_index'][selected[-1]])],selected_samples=len(selected),
        requested_origins=int(p['requested'].sum()),drawn_arrows=int(p['drawn'].sum()),min_force_N=args.min_force_N,arrow_mode=args.arrow_mode,arrow_length_m=args.arrow_length_m,meters_per_N=args.meters_per_N,
        force_transform='F_base = R_base_EE @ F_recorded; force sensor axes are EE-aligned; no additional sensor/TCP rotation or force translation',
        origin='Each force arrow tail is the corresponding calibrated TCP position; translation does not change the force vector',
        sign='As recorded; no action/reaction sign reversal',
        measurement_archive=str(archive),measurement_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        measurement_metadata_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interaction_result_sha256=hashlib.sha256(source_result.read_bytes()).hexdigest() if source_result else None,
        tcp_reference_assumption=summary.get('tcp_calibration',{}).get('reference_assumption'),html=not args.no_html,
        html_style='thin_shafts_small_mesh_tips',html_head_fraction=args.head_fraction)
    (target/'manifest.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    return {k:meta[k] for k in ['scene','status','interval_source','interval_status','selected_native_frames','selected_samples','requested_origins','drawn_arrows']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements-root',type=Path,required=True);parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--scene',nargs='+',help='Optional scene selection; default all calibrated episodes')
    interval=parser.add_mutually_exclusive_group();interval.add_argument('--interaction-root',type=Path,help='Use each scene\'s detected interaction interval');interval.add_argument('--frame-range',nargs=2,type=int,metavar=('FIRST','LAST'),help='Inclusive range applied to selected scenes; default whole episode')
    parser.add_argument('--frame-space',choices=['native','rgbd'],default='native');parser.add_argument('--arrow-every',type=int,default=10)
    parser.add_argument('--arrow-mode',choices=['direction','magnitude'],default='direction');parser.add_argument('--arrow-length-m',type=float,default=.025)
    parser.add_argument('--meters-per-N',type=float,default=.01);parser.add_argument('--min-force-N',type=float,default=.05)
    parser.add_argument('--no-html',action='store_true',help='PNG/CSV/NumPy only; otherwise requires Plotly')
    parser.add_argument('--html-only',action='store_true',help='Refresh existing HTML from saved vectors; preserve each plot\'s spacing, scales, PNG and numerical exports')
    parser.add_argument('--head-fraction',type=float,default=.15,help='HTML arrowhead length as a fraction of total arrow length (default 0.15)')
    parser.add_argument('--overwrite',action='store_true');args=parser.parse_args(argv)
    if args.arrow_every<1 or min(args.arrow_length_m,args.meters_per_N)<=0 or args.min_force_N<0:parser.error('Arrow spacing/scales must be positive; force threshold must be nonnegative')
    if not 0<args.head_fraction<1:parser.error('--head-fraction must be between zero and one')
    if args.html_only and args.no_html:parser.error('--html-only and --no-html cannot be combined')
    root=args.measurements_root.expanduser().resolve();output=args.output_root.expanduser().resolve()
    if args.interaction_root:args.interaction_root=args.interaction_root.expanduser().resolve()
    for source in [root]+([args.interaction_root] if args.interaction_root else []):
        if output==source or output.is_relative_to(source) or source.is_relative_to(output):parser.error('Output must be separate from all inputs')
    scenes=sorted(p.parent for p in root.glob('*/measurements.npz'))
    if args.scene:
        scenes=[p for p in scenes if p.name in args.scene]
        if {p.name for p in scenes}!=set(args.scene):parser.error('Some requested scenes were not discovered')
    if not scenes:parser.error('No calibrated measurement archives found')
    if not args.no_html:
        try:
            import plotly.graph_objects as go
            go.Mesh3d(x=[0],y=[0],z=[0])
        except (ImportError,ValueError):parser.error('Interactive plots need Plotly; use the existing miniconda3/bin/python or --no-html')
    for filename in ['.matplotlib','summary.json','summary.csv']:
        if not (output/filename).resolve().is_relative_to(output):parser.error('Output file escapes output root')
    if (output/'summary.json').exists() and not args.overwrite:parser.error('Existing batch results require --overwrite or a fresh root')
    output.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(output/'.matplotlib');rows=[]
    for folder in scenes:
        try:row=refresh_html(folder,output,args) if args.html_only else process(folder,output,args)
        except Exception as e:row=dict(scene=folder.name,status='FAILED',error=str(e))
        rows.append(row);print(f'{len(rows)}/{len(scenes)} {folder.name}: {row["status"]}; arrows={row.get("drawn_arrows",0)}',flush=True)
    fields=sorted({k for r in rows for k in r})
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    result=dict(requested=len(rows),plotted=sum(r['status']=='PLOTTED' for r in rows),skipped=[r for r in rows if r['status'].startswith('SKIPPED')],failed=[r for r in rows if r['status']=='FAILED'])
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return int(bool(result['failed']))


if __name__=='__main__':sys.exit(main())
