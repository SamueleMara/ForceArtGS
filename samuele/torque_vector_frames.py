"""One standalone 3D torque-vector image per native sample, including all cameras."""
import json
from pathlib import Path

import cv2
import numpy as np


def vector_segments(vector):
    vector=np.asarray(vector,float)
    if not np.isfinite(vector).all():return np.empty((0,2,3))
    length=np.linalg.norm(vector)
    if length<=1e-15:return np.empty((0,2,3))
    direction=vector/length;side=np.cross(direction,np.eye(3)[np.argmin(abs(direction))]);side/=np.linalg.norm(side)
    neck=vector-.15*length*direction;wing=.055*length*side
    return np.array([[np.zeros(3),vector],[vector,neck+wing],[vector,neck-wing]])


def save_torque_frames(folder,frame_ids,rgbd_ids,time,torques,labels,records):
    """Moments are shown from zero in component space (N m), not at a TCP position."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    base=np.array([case['torque_base_Nm'] for case in torques])
    frames=[('Base axes',base)]
    for record in records:
        R=np.linalg.inv(np.array(record['camera_to_base']))[:3,:3]
        frames.append((record['camera']+' axes',base@R.T))
    finite_norm=np.linalg.norm(base,axis=2);finite_norm=finite_norm[np.isfinite(finite_norm)]
    limit=max(float(finite_norm.max())*1.15 if len(finite_norm) else 0.,.001)
    columns=2;rows=(len(frames)+columns-1)//columns
    fig=plt.figure(figsize=(10,4*rows),dpi=90)
    colors=['#138a36','#ee8500'];artists=[];texts=[]
    for panel,(name,values) in enumerate(frames):
        ax=fig.add_subplot(rows,columns,panel+1,projection='3d')
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit),xlabel='tau x (N m)',ylabel='tau y (N m)',zlabel='tau z (N m)',title=name)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-55);ax.scatter([0],[0],[0],color='black',s=8)
        for case,label in enumerate(labels):
            collection=Line3DCollection(np.zeros((1,2,3)),colors=colors[case%len(colors)],linewidths=2.2,label=str(label))
            ax.add_collection3d(collection,autolim=False);artists.append((ax,collection,values,case))
        text=ax.text2D(.01,.99,'',transform=ax.transAxes,va='top',fontsize=7);texts.append((ax,text,values))
        ax.legend(loc='lower left',fontsize=7)
    fig.subplots_adjust(top=.88,bottom=.06,wspace=.16,hspace=.18)
    title=fig.text(.02,.965,'',fontsize=11,va='top')
    fig.text(.02,.015,'Recorded moment origin unchanged. Zero is the component-space origin; fixed scale across all frames.',fontsize=8)
    fig.canvas.draw();background=fig.canvas.copy_from_bbox(fig.bbox)
    for i,frame in enumerate(frame_ids):
        fig.canvas.restore_region(background)
        for ax,collection,values,case in artists:
            segments=vector_segments(values[case,i]);collection.set_segments(segments if len(segments) else np.zeros((1,2,3)))
            collection.set_alpha(1. if len(segments) else 0.);collection.do_3d_projection();ax.draw_artist(collection)
        for ax,text,values in texts:
            lines=[]
            for c,label in enumerate(labels):
                vector=values[c,i]
                lines.append(f'{label}: ({vector[0]:+.5f}, {vector[1]:+.5f}, {vector[2]:+.5f}) N m; norm {np.linalg.norm(vector):.5f}')
            text.set_text('\n'.join(lines));ax.draw_artist(text)
        warning='; UNVERIFIED mounting trials' if len(labels)>1 else ''
        title.set_text(f'Torque vectors — native frame {int(frame)}; RGB-D {int(rgbd_ids[i])}; t={time[i]:.6f} s{warning}')
        fig.draw_artist(title)
        rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
        target=folder/f'frame_{int(frame):06d}.png'
        if not cv2.imwrite(str(target),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise OSError('Cannot write torque image '+str(target))
        if i%250==0:print(f'{folder.parent.name}: 3D torque images {i+1}/{len(frame_ids)}',flush=True)
    plt.close(fig)
    manifest=dict(native_frames=[int(f) for f in frame_ids],rgbd_frames=[int(f) for f in rgbd_ids],time_s=[float(t) for t in time],
        images=len(frame_ids),panels=[name for name,_ in frames],cases=[str(label) for label in labels],axis_limit_Nm=limit,
        origin='Zero in moment-component space; recorded physical moment origin unchanged; no TCP transport',all_native_frames=True)
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    first=int(frame_ids[0]) if len(frame_ids) else 0
    page='''<!doctype html><meta charset="utf-8"><title>Per-frame 3D torque vectors</title>
<style>body{font:16px sans-serif;margin:20px}img{max-width:100%}input{width:70%}</style>
<h1>Per-frame 3D torque vectors</h1><p>Fixed N m scale. Every panel shows the same recorded moment in different axes. No TCP-origin shift.</p>
<button id="prev">Previous</button><input id="frame" type="range" min="0" value="0"><button id="next">Next</button><span id="label"></span>
<p><img id="image" src="frame_'''+f'{first:06d}'+'''.png" alt="Standalone 3D torque vectors"></p>
<script>const data='''+json.dumps(manifest)+''';const slider=document.getElementById('frame');slider.max=Math.max(0,data.images-1);
function show(){const i=Number(slider.value);document.getElementById('image').src='frame_'+String(data.native_frames[i]).padStart(6,'0')+'.png';document.getElementById('label').textContent=' Native '+data.native_frames[i]+' / RGB-D '+data.rgbd_frames[i]+' / '+data.time_s[i].toFixed(4)+' s';}
document.getElementById('prev').onclick=()=>{slider.value=Math.max(0,Number(slider.value)-1);show();};document.getElementById('next').onclick=()=>{slider.value=Math.min(data.images-1,Number(slider.value)+1);show();};slider.oninput=show;show();</script>'''
    (folder/'index.html').write_text(page)
    return manifest
