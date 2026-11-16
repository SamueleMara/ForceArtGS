#!/usr/bin/env python3
"""Project identity/+90-Z trial forces onto calibrated RGB frames, without changing data."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from compare_force_mounting_rotations import rotate_sensor_forces
from plot_tcp_force_directions import force_to_base
from force_trajectory_decomposition import trajectory_tangent,save_decomposition,decompose_force


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(points,world_to_camera,K):
    points=np.asarray(points,float).reshape(-1,3)
    local=points@world_to_camera[:3,:3].T+world_to_camera[:3,3]
    good=np.isfinite(local).all(axis=1)&(local[:,2]>1e-6)
    pixels=np.full((len(points),2),np.nan)
    q=local[good]@K.T;pixels[good]=q[:,:2]/q[:,2:]
    return pixels


def project_force(origin,force_base,world_to_camera,K,scale):
    # Translation applies to the origin, never to a force vector.
    pixels=project([origin,np.asarray(origin)+scale*np.asarray(force_base)],world_to_camera,K)
    return pixels,world_to_camera[:3,:3]@force_base


def arrow(image,pixels,color,thickness=3,label=None):
    height,width=image.shape[:2]
    if not np.isfinite(pixels[0]).all():return 'origin behind camera or invalid'
    root=tuple(np.rint(pixels[0]).astype(int))
    if not (0<=root[0]<width and 0<=root[1]<height):return 'origin outside image'
    if not np.isfinite(pixels[1]).all():return 'tip behind camera or invalid'
    if np.linalg.norm(pixels[1]-pixels[0])<2:return 'projected length <2 px'
    tip=tuple(np.rint(np.clip(pixels[1],-1000000,1000000)).astype(int))
    visible,start,end=cv2.clipLine((0,0,width,height),root,tip)
    if not visible:return 'not visible'
    inside=0<=tip[0]<width and 0<=tip[1]<height
    if inside:cv2.arrowedLine(image,start,end,color,thickness,cv2.LINE_AA,tipLength=.16)
    else:cv2.line(image,start,end,color,thickness,cv2.LINE_AA)
    if label:cv2.putText(image,label,end,cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    return 'drawn' if inside else 'shaft clipped; head outside image'


def panel(rgb,pixels,ee_pixel,z_pixels,force_sensor,force_ee,force_camera,rotation,minimum,scale,header,decomposition=None,calibration_valid=False):
    image=rgb.copy();height,width=rgb.shape[:2]
    arrow(image,z_pixels,(255,100,60),1,'t' if decomposition is not None else 'EE Z')
    for point,color in [(ee_pixel,(255,255,0)),(pixels[0],(0,255,255))]:
        if np.isfinite(point).all() and 0<=point[0]<width and 0<=point[1]<height:
            cv2.drawMarker(image,tuple(np.rint(point).astype(int)),color,cv2.MARKER_CROSS,12,2)
    norm=float(np.linalg.norm(force_sensor))
    state=arrow(image,pixels,(60,220,60) if rotation==0 else (0,165,255)) if np.isfinite(norm) and norm>=minimum else 'below threshold or invalid'
    if decomposition is not None:
        for key,color,label in [('tangent',(255,220,0),'Ft'),('perpendicular',(255,60,230),'Fp')]:
            decomposition[key+'_projection_status']=arrow(image,decomposition[key+'_pixels'],color,2,label) if decomposition['valid'] and norm>=minimum else 'undefined tangent or below display threshold'
    canvas=np.full((height+(220 if decomposition is not None else 176),width,3),24,np.uint8);canvas[66:66+height]=image
    lines=[(header,20),(f'{rotation:+g} deg Z sensor-to-EE trial; NOMINAL TCP ORIGIN',44),
        (f'|F|={norm:.3f} N; sensor ({force_sensor[0]:+.2f}, {force_sensor[1]:+.2f}, {force_sensor[2]:+.2f}) N',height+88),
        (f'EE ({force_ee[0]:+.2f}, {force_ee[1]:+.2f}, {force_ee[2]:+.2f}); cam ({force_camera[0]:+.2f}, {force_camera[1]:+.2f}, {force_camera[2]:+.2f}) N',height+109),
        (f'|axial Z|={abs(force_sensor[2]):.3f} N; transverse XY={np.linalg.norm(force_sensor[:2]):.3f} N',height+130),
        (f'{scale:g} m/N display scale; min {minimum:g} N; {state}',height+151),
        ('Yellow: nominal TCP; cyan: EE; blue: EE Z (reference)',height+170)]
    if decomposition is not None:
        origin_label='stored TCP' if calibration_valid else 'NOMINAL TCP ORIGIN; UNVERIFIED'
        lines=[(header,20),(f'{rotation:+g} deg Z mounting; {origin_label}',44),
            (f'|F|={norm:.3f} N; sensor ({force_sensor[0]:+.2f}, {force_sensor[1]:+.2f}, {force_sensor[2]:+.2f}) N',height+88),
            ((f'Ft={decomposition["signed"]:+.3f} N; |Fperp|={decomposition["normal"]:.3f} N' if decomposition['valid'] else 'Ft and Fperp undefined: '+decomposition['status']),height+109),
            (f'EE speed={decomposition["speed"]:.4f} m/s; {decomposition["status"]}',height+130),
            (f'{scale:g} m/N; min {minimum:g} N; {state}',height+151),
            ('Green/orange: resultant; cyan: Ft; magenta: Fperp',height+172),
            ('Blue: centered EE tangent (4 cm); yellow cross: TCP',height+193),
            ('Positive Ft follows travel; negative Ft opposes travel',height+214)]
    for text,y in lines:cv2.putText(canvas,text,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.43,(235,235,235),1,cv2.LINE_AA)
    return canvas,state


def render_scene(scene,measurements,output,camera_name,scale,minimum,ffmpeg,decompose=False,force_epsilon_n=1e-8,camera_subfolder=False):
    archive=measurements/scene.name/'measurements.npz';digest=sha(archive)
    conversion_path=scene/'metadata/real_conversion.json';cameras_path=scene/'metadata/cameras.json'
    conversion=json.loads(conversion_path.read_text());metadata=json.loads(cameras_path.read_text())
    cameras=[c for c in metadata['cameras'] if c['name']==camera_name]
    if len(cameras)!=1:raise ValueError('Expected exactly one camera '+camera_name)
    camera=cameras[0]
    if camera['pose_convention']!='opencv_optical_camera_to_world' or camera['coordinate_frame']!='base_link':raise ValueError('Unsupported camera convention')
    T=np.asarray(camera['pose_matrix_4x4'],float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(T[:3,:3]),1):raise ValueError('Invalid camera transform')
    world_to_camera=np.linalg.inv(T);K=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1.]])
    if not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:raise ValueError('Invalid intrinsics')
    with np.load(archive,allow_pickle=False) as z:
        data={k:z[k].copy() for k in ['frame_index','rgbd_frame_index','timestamp_s','ee_position_m','ee_rotvec_rad','tcp_position_m','force_N']}
    ids=data['frame_index'];lookup={int(f):i for i,f in enumerate(ids)}
    if len(lookup)!=len(ids):raise ValueError('Duplicate measurement frame indices')
    source=conversion['source_frames'];frames=conversion['output_frames']
    if len(source)!=len(frames) or len(frames)<2 or len(set(frames))!=len(frames):raise ValueError('Invalid frame mapping')
    indices=np.array([lookup[f] for f in source]);frames=np.array(frames)
    if not np.array_equal(data['rgbd_frame_index'][indices],frames):raise ValueError('Cached frame mapping differs from conversion')
    timing_path=scene/'metadata/frame_timestamps.csv'
    with timing_path.open(newline='') as f:timing={int(row['frame_index']):row for row in csv.DictReader(f)}
    if any(int(timing[int(frame)]['source_frame_index'])!=native for frame,native in zip(frames,source)):raise ValueError('Video timestamps use a different source mapping')
    times=np.array([float(timing[int(frame)]['timestamp_s']) for frame in frames]);dt=np.diff(times)
    if not np.all(dt>0) or not np.allclose(dt,np.median(dt),atol=1e-7,rtol=1e-6):raise ValueError('Debug MP4 requires uniform converted frame cadence')
    fps=float(1/np.median(dt));height,width=camera['height'],camera['width']
    video_height=height+(220 if decompose else 176)
    if width%2 or video_height%2:raise ValueError('H264 dimensions must be even')
    calibration_valid=False;angles=[0,90]
    if decompose:
        summary=json.loads((measurements/scene.name/'summary.json').read_text());calibration=summary.get('tcp_calibration',{})
        calibration_valid=calibration.get('tcp_calibration_valid') is not False
        if calibration_valid:
            axes=calibration.get('wrench_frame',{}).get('axes')
            if axes!='ee':raise ValueError('EE-aligned force axes are required for calibrated scenes')
            angles=[0]
        elif calibration.get('mounting_id')!='white_drawer_early_mounting':raise ValueError('Unknown mounting: no supported diagnostic trial')
    forces=[];matrices=[]
    for angle in angles:
        trial,matrix=rotate_sensor_forces(data,angle);matrices.append(matrix)
        forces.append(force_to_base(data['ee_rotvec_rad'],trial['force_N']))
    if len(forces)==2:np.testing.assert_allclose(np.linalg.norm(forces[0],axis=1),np.linalg.norm(forces[1],axis=1),atol=1e-12,equal_nan=True)
    shared_out=output/scene.name;shared_out.mkdir(exist_ok=True)
    out=shared_out/camera_name if decompose or camera_subfolder else shared_out
    if out!=shared_out:out.mkdir()
    tangent=None;decompositions=None;labels=['ee_aligned'] if calibration_valid else ['identity_reference','sensor_z_plus90']
    if decompose:
        tangent=trajectory_tangent(data['timestamp_s'],data['ee_position_m'],data['frame_index'])
        if not (shared_out/'force_decomposition.npz').exists():decompositions=save_decomposition(shared_out,data,tangent,forces,labels,force_epsilon_n)
        else:decompositions=[decompose_force(force,tangent,force_epsilon_n) for force in forces]
    video=out/f'force_{camera_name}.mp4'
    command=[ffmpeg,'-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{len(angles)*width}x{video_height}',
        '-framerate',f'{fps:.10g}','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    native_norm=np.linalg.norm(data['force_N'],axis=1);sampled_norm=native_norm[indices]
    peak=int(np.nanargmax(sampled_norm)) if np.isfinite(sampled_norm).any() else 0
    snapshot_ids=set([0,len(frames)-1,peak,*np.linspace(0,len(frames)-1,5,dtype=int)]);snapshots=[];rows=[]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for number,(frame,i) in enumerate(zip(frames,indices)):
            rgb=cv2.imread(str(scene/f'rgb/{camera_name}/frame_{frame:06d}.png'))
            if rgb is None or rgb.shape!=(height,width,3):raise ValueError('Missing or invalid RGB frame '+str(frame))
            origin=data['tcp_position_m'][i];ee=data['ee_position_m'][i];rv=data['ee_rotvec_rad'][i]
            z_axis=Rotation.from_rotvec(rv).apply([0,0,1]) if np.isfinite(rv).all() else np.full(3,np.nan)
            ee_pixel=project([ee],world_to_camera,K)[0];z_pixels=project([origin,origin+.04*z_axis],world_to_camera,K)
            if decompose:z_pixels=project([origin,origin+.04*tangent['tangent'][i]],world_to_camera,K)
            header=f'{camera_name} RGB-D {frame}; native {ids[i]}; t={data["timestamp_s"][i]:.3f}s'
            panels=[]
            for case,angle in enumerate(angles):
                pixels,fcam=project_force(origin,forces[case][i],world_to_camera,K,scale)
                fee=matrices[case]@data['force_N'][i]
                parts=None
                if decompose:
                    component=decompositions[case]
                    parts=dict(valid=bool(component['decomposition_valid'][i]),signed=component['force_tangent_signed_N'][i],normal=component['force_perpendicular_norm_N'][i],
                        speed=tangent['speed_m_s'][i],status=tangent['tangent_status'][i],
                        tangent_pixels=project_force(origin,component['force_tangent_base_N'][i],world_to_camera,K,scale)[0],
                        perpendicular_pixels=project_force(origin,component['force_perpendicular_base_N'][i],world_to_camera,K,scale)[0])
                view,state=panel(rgb,pixels,ee_pixel,z_pixels,data['force_N'][i],fee,fcam,angle,minimum,scale,header,parts,calibration_valid)
                panels.append(view)
                row=dict(rgbd_frame=int(frame),native_frame=int(ids[i]),timestamp_s=float(data['timestamp_s'][i]),rotation_z_deg=angle,
                    force_norm_N=float(np.linalg.norm(data['force_N'][i])),projection_status=state)
                for label,values in [('force_sensor_N',data['force_N'][i]),('force_ee_N',fee),('force_base_N',forces[case][i]),('force_camera_N',fcam),('nominal_tcp_m',origin)]:
                    row.update({label+'_'+axis:float(value) for axis,value in zip('xyz',values)})
                row.update(root_u=float(pixels[0,0]),root_v=float(pixels[0,1]),tip_u=float(pixels[1,0]),tip_v=float(pixels[1,1]));rows.append(row)
                if parts is not None:
                    row.update(tangent_status=parts['status'],tangent_valid=parts['valid'],speed_m_s=parts['speed'],force_tangent_signed_N=parts['signed'],force_perpendicular_norm_N=parts['normal'])
                    for kind in ['tangent','perpendicular']:
                        row.update({kind+'_tip_'+axis:float(value) for axis,value in zip('uv',parts[kind+'_pixels'][1])})
                        row[kind+'_projection_status']=parts[kind+'_projection_status']
            composite=np.hstack(panels);process.stdin.write(composite.tobytes())
            if number in snapshot_ids:
                name=f'frame_{frame:06d}.jpg'
                if not cv2.imwrite(str(out/name),composite):raise OSError('Cannot write snapshot')
                snapshots.append(name)
                if number==peak:shutil.copyfile(out/name,out/'peak_force.jpg')
            if number%100==0:print(f'{scene.name}: {number+1}/{len(frames)} frames',flush=True)
        process.stdin.close();error=process.stderr.read().decode();returncode=process.wait()
        if returncode:raise RuntimeError('ffmpeg failed: '+error)
    except BaseException:
        if process.poll() is None:process.kill();process.wait()
        raise
    finally:
        if not process.stdin.closed:process.stdin.close()
        process.stderr.close()
    with (out/'projected_forces.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=dict(scene=scene.name,camera=camera_name,relative_folder=str(out.relative_to(output)),status='UNVALIDATED_MOUNTING_AND_NOMINAL_TCP_DEBUG',frames=len(frames),fps=fps,
        video=video.name,snapshots=snapshots,force_scale_m_per_N=scale,min_force_N=minimum,
        R_ee_sensor=[m.tolist() for m in matrices],camera_to_base=T.tolist(),rectified_K=K.tolist(),
        projection='pinhole(T_camera_base @ p_nominal_tcp) to pinhole(T_camera_base @ (p_nominal_tcp + scale * R_base_EE @ R_EE_sensor @ F_sensor))',
        reference_arrow='EE local Z at nominal TCP, length 0.04 m; not independently verified drawer axis',
        force_sign='As recorded; no action/reaction reversal or baseline subtraction',
        timing='Exact converted RGB-D to native Parquet frame mapping; no interpolation. Hardware synchronization unverified.',
        peak_force_all_native_N=float(np.nanmax(native_norm)) if np.isfinite(native_norm).any() else None,
        peak_force_rgbd_samples_N=float(np.nanmax(sampled_norm)) if np.isfinite(sampled_norm).any() else None,
        limitations=['EE-to-contact transform remains unknown; projected arrow origin uses existing nominal TCP',
            'An out-of-plane force can project to a short arrow; image angles are not 3D force/axis angles',
            'Odd native force samples without an RGB-D mapping are not displayed; narrow spikes can be missed',
            'Episode 2 contains different drawer interactions; this full-episode video is not a single-joint fit'],
        measurement_archive_sha256=digest,conversion_sha256=sha(conversion_path),camera_metadata_sha256=sha(cameras_path),
        frame_timestamps_sha256=sha(timing_path),source_script_sha256=sha(Path(__file__)))
    if decompose:
        from collections import Counter
        manifest.update(status='TRAJECTORY_FORCE_DEBUG' if calibration_valid else manifest['status'],tcp_calibration_valid=calibration_valid,
            reference_arrow='Centered EE direction p[i+1]-p[i-1], normalized; 4 cm display reference',
            measurement_metadata_sha256=sha(measurements/scene.name/'summary.json'),
            decomposition=dict(native_samples=len(ids),force_epsilon_N=force_epsilon_n,
                method='Observed EE centered difference: t[i]=(p_EE[i+1]-p_EE[i-1])/norm(...); no smoothing, no speed cutoff, no endpoint fallback',
                tangent_status_counts=dict(Counter(tangent['tangent_status'])),cases=[]))
        manifest['limitations']=['Tangent follows centered observed EE motion; contact, slip and object motion are not established',
            'Zero centered displacement, missing neighbors and endpoints have undefined tangents; no smoothing or speed cutoff',
            'eta=abs(F_parallel)/(norm(F)+epsilon); zero force gives eta=0 when the tangent is defined',
            'Pixel angles are perspective projections; unmapped native samples are saved numerically but absent from video',
            'The three early drawers retain nominal TCP origins and two unvalidated force-mounting trials']
        for label,component,force in zip(labels,decompositions,forces):
            valid=component['decomposition_valid'];strong=valid&(component['force_norm_N']>=minimum)
            error=np.linalg.norm(component['force_tangent_base_N'][valid]+component['force_perpendicular_base_N'][valid]-force[valid],axis=1)
            def median(values):
                values=values[np.isfinite(values)];return float(np.median(values)) if len(values) else None
            manifest['decomposition']['cases'].append(dict(case=label,valid_native_samples=int(valid.sum()),moving_samples_force_above_display_threshold=int(strong.sum()),
                median_signed_tangent_N=median(component['force_tangent_signed_N'][strong]),median_perpendicular_N=median(component['force_perpendicular_norm_N'][strong]),
                median_angle_to_unoriented_tangent_deg=median(component['force_angle_to_unoriented_tangent_deg'][strong]),max_reconstruction_error_N=float(error.max()) if len(error) else None))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    assert sha(archive)==digest
    print(f'{scene.name}: saved {video}',flush=True)
    return manifest


def render_dataset(job):
    import traceback
    cv2.setNumThreads(1)
    results=[]
    for camera in job['cameras']:
        try:
            results.append(render_scene(Path(job['scene']),Path(job['measurements']),Path(job['output']),camera,
                job['scale'],job['minimum'],job['ffmpeg'],job['decompose'],job['epsilon'],job['camera_subfolder']))
        except Exception as error:
            relative=Path(job['scene']).name+'/'+camera
            folder=Path(job['output'])/relative;folder.mkdir(parents=True,exist_ok=True)
            (folder/'error.log').write_text(traceback.format_exc())
            results.append(dict(scene=Path(job['scene']).name,camera=camera,relative_folder=relative,status='FAILED',error=str(error)))
    return results


def write_index(output,results,decompose):
    import html
    from itertools import groupby
    page=['<!doctype html><meta charset="utf-8"><title>Trajectory force debug</title><style>body{font:16px sans-serif;margin:24px}video,img{max-width:100%}p{max-width:1100px}details{margin:18px 0}summary{cursor:pointer}</style>',
        '<h1>Trajectory force diagnostics</h1>']
    if decompose:
        if (output/'force_torque_summary.html').is_file():page.append('<p><b><a href="force_torque_summary.html">Interaction-window mean eta and torque summary</a></b></p><p>Current wrench-analysis files recompute the centered EE formula. Existing force videos/force-decomposition plots retain their generation method in each camera manifest; a fresh debug run regenerates those overlays.</p>')
        page.extend(['<p>Green/orange: resultant force. Cyan: tangential force. Magenta: perpendicular force. Blue: centered observed EE direction. Positive tangential force follows travel; negative opposes it. Tangents are undefined at rest or without sufficient support.</p>',
            '<p>Numerical CSV/NPZ contain every native Parquet frame, including samples without RGB-D. Videos use exact RGB-D matches and each camera calibration. Three old drawers show identity and +90 Z trials at nominal TCP positions; other scenes use their stored EE-aligned mounting. EE motion is not independent object-joint or contact ground truth.</p>'])
    else:page.append('<p>Left green: identity mounting. Right orange: +90 Z trial. Yellow: nominal TCP; cyan: EE; blue: EE Z reference. Both mounting trials are unvalidated.</p>')
    for name,records in groupby(sorted(results,key=lambda r:(r['scene'],r['camera'])),key=lambda r:r['scene']):
        name=html.escape(name,quote=True);page.append('<details><summary>'+name+'</summary>')
        if decompose:
            page.append(f'<p><a href="{name}/force_decomposition.csv">All native frames (CSV)</a> | <a href="{name}/force_decomposition.npz">NumPy</a> | <a href="{name}/force_decomposition.png">Time plots</a></p>')
            page.append(f'<img loading="lazy" src="{name}/force_decomposition.png" alt="Signed tangential and perpendicular force time series">')
            if (output/html.unescape(name)/'force_share_torque.png').is_file():
                page.append(f'<p><a href="{name}/wrench_analysis.csv">Percentages and torques CSV</a> | <a href="{name}/wrench_analysis.npz">NumPy</a></p><img loading="lazy" src="{name}/force_share_torque.png" alt="Tangential force percentage and recorded-origin torque analysis">')
            if (output/html.unescape(name)/'torque_3d_frames/index.html').is_file():
                page.append(f'<p><b><a href="{name}/torque_3d_frames/index.html">Separate 3D torque image for every native frame: base and all camera axes</a></b></p>')
        for record in records:
            camera=html.escape(record['camera']);page.append('<h3>'+camera+'</h3>')
            if record['status']=='FAILED':page.append('<p>FAILED: '+html.escape(record['error'])+'</p>');continue
            rel=html.escape(record['relative_folder'],quote=True);video=html.escape(record['video'],quote=True)
            page.extend([f'<video controls preload="none" poster="{rel}/peak_force.jpg" src="{rel}/{video}"></video>',
                f'<p><a href="{rel}/projected_forces.csv">Camera projections CSV</a> | <a href="{rel}/manifest.json">Metadata</a></p>'])
            if (output/record['relative_folder']/'torque_camera.png').is_file():
                page.append(f'<p><a href="{rel}/torque_camera.csv">Camera-frame torque and percentage CSV</a></p><img loading="lazy" src="{rel}/torque_camera.png" alt="Camera-frame torque components at recorded origin">')
        page.append('</details>')
    (output/'index.html').write_text('\n'.join(page)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--converted-root',type=Path,required=True);p.add_argument('--measurements-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    selection=p.add_mutually_exclusive_group(required=True);selection.add_argument('--scene',nargs='+');selection.add_argument('--all-scenes',action='store_true')
    cameras=p.add_mutually_exclusive_group();cameras.add_argument('--camera',default='cam_002');cameras.add_argument('--all-cameras',action='store_true')
    p.add_argument('--force-scale-m-per-n',type=float,default=.02);p.add_argument('--min-force-n',type=float,default=.2)
    p.add_argument('--decompose',action='store_true',help='Save all native tangential/perpendicular forces and overlay components')
    p.add_argument('--interaction-root',type=Path,help='Required with --decompose: saved per-scene interaction bounds')
    p.add_argument('--torque-3d-images',action='store_true',help='Save a separate 3D torque image for every native frame')
    p.add_argument('--force-epsilon-n',type=float,default=1e-8,help='Positive epsilon in eta=abs(F_parallel)/(norm(F)+epsilon)')
    p.add_argument('--workers',type=int,default=1,help='Parallel datasets; each uses two encoder threads')
    args=p.parse_args(argv)
    if args.decompose and args.interaction_root is None:p.error('--decompose requires --interaction-root for window-only temporal metrics')
    if args.torque_3d_images and not args.decompose:p.error('--torque-3d-images requires --decompose')
    if args.interaction_root:args.interaction_root=args.interaction_root.expanduser().resolve()
    parameters=[args.force_scale_m_per_n,args.min_force_n,args.force_epsilon_n]
    if not np.isfinite(parameters).all() or min(parameters[0],parameters[2])<=0 or parameters[1]<0 or args.workers<1:p.error('Scale, force epsilon and workers must be positive; force threshold nonnegative')
    if Path(args.camera).name!=args.camera or args.camera in ['.','..']:p.error('Camera must be a directory name')
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:p.error('ffmpeg is required for H264 videos')
    source=args.converted_root.expanduser().resolve();measurements=args.measurements_root.expanduser().resolve();out=args.output_root.expanduser().resolve()
    for root in [source,measurements]+([args.interaction_root.expanduser().resolve()] if args.interaction_root else []):
        if out==root or out.is_relative_to(root) or root.is_relative_to(out):p.error('Debug output must be separate from inputs')
    if out.exists():p.error('Choose a fresh output directory; existing results are preserved')
    names=sorted(set(args.scene if args.scene else [q.parent.name for q in measurements.glob('*/measurements.npz')]))
    if not names:p.error('No datasets discovered')
    jobs=[]
    for name in names:
        if Path(name).name!=name or name in ['.','..']:p.error('Scene must be a directory name')
        for file in [source/name/'metadata/real_conversion.json',source/name/'metadata/cameras.json',source/name/'metadata/frame_timestamps.csv',measurements/name/'measurements.npz']+([measurements/name/'summary.json',args.interaction_root/name/'result.json'] if args.decompose else []):
            if not file.is_file():p.error('Missing input: '+str(file))
        available=[c['name'] for c in json.loads((source/name/'metadata/cameras.json').read_text())['cameras']]
        selected=sorted(available) if args.all_cameras else [args.camera]
        if not selected or len(set(available))!=len(available) or any(c not in available or Path(c).name!=c or c in ['.','..'] for c in selected):p.error('Invalid or missing cameras in '+name)
        jobs.append(dict(scene=str(source/name),measurements=str(measurements),output=str(out),cameras=selected,
            scale=args.force_scale_m_per_n,minimum=args.min_force_n,ffmpeg=ffmpeg,decompose=args.decompose,
            epsilon=args.force_epsilon_n,camera_subfolder=args.all_cameras))
    out.mkdir(parents=True);os.environ['MPLCONFIGDIR']=str(out/'.matplotlib');cv2.setNumThreads(1)
    plan=dict(datasets=len(jobs),cameras=sum(len(j['cameras']) for j in jobs),decompose=args.decompose,workers=args.workers,completed=False)
    (out/'run.json').write_text(json.dumps(plan,indent=2)+'\n')
    results=[]
    def record(batch):
        results.extend(batch)
        temporary=out/'summary.json.tmp';temporary.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n');temporary.replace(out/'summary.json')
        write_index(out,results,args.decompose)
        print(f'Completed camera runs: {len(results)}/{plan["cameras"]}; failures: {sum(r["status"]=="FAILED" for r in results)}',flush=True)
    if args.workers==1:
        for job in jobs:record(render_dataset(job))
    else:
        import concurrent.futures,multiprocessing
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending=[pool.submit(render_dataset,job) for job in jobs]
            for future in concurrent.futures.as_completed(pending):record(future.result())
    if args.decompose:
        numerical=[];seen=set()
        for result in results:
            if result['status']=='FAILED' or result['scene'] in seen:continue
            seen.add(result['scene'])
            for case in result['decomposition']['cases']:
                numerical.append(dict(scene=result['scene'],tcp_calibration_valid=result['tcp_calibration_valid'],
                    native_samples=result['decomposition']['native_samples'],**case))
        if numerical:
            with (out/'decomposition_summary.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(numerical[0]));writer.writeheader();writer.writerows(numerical)
    plan.update(completed=True,successful_camera_runs=sum(r['status']!='FAILED' for r in results),failed_camera_runs=sum(r['status']=='FAILED' for r in results))
    (out/'run.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.decompose:
        from analyze_force_share_torques import update_analysis
        wrench=update_analysis(out,measurements,0.,args.interaction_root,args.torque_3d_images,args.force_epsilon_n)
        plan.update(wrench_analysis_completed_scenes=wrench['completed_scenes'],wrench_analysis_failed_scenes=len(wrench['failed']))
        (out/'run.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('Open '+str(out/'index.html'))
    return int(bool(plan['failed_camera_runs'] or plan.get('wrench_analysis_failed_scenes',0)))


if __name__=='__main__':raise SystemExit(main())
