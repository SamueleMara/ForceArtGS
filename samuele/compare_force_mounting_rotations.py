#!/usr/bin/env python3
"""Compare identity and a user-proposed +90 degree sensor-to-EE Z rotation.

These are explicitly unvalidated trials at the existing nominal TCP positions.
Neither the source measurements nor the production calibration are changed.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from plot_tcp_force_directions import prepare,save_vectors,draw


def rotate_sensor_forces(data,degrees):
    """R_EE_sensor maps sensor components to EE components, then to base."""
    matrix=Rotation.from_euler('z',degrees,degrees=True).as_matrix()
    result=dict(data)
    result['force_N']=np.asarray(data['force_N'])@matrix.T
    return result,matrix


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_scene(folder,interaction,output):
    summary_path=folder/'summary.json';archive=folder/'measurements.npz'
    summary=json.loads(summary_path.read_text());interval_path=interaction/folder.name/'result.json'
    interval=json.loads(interval_path.read_text())
    if interval['scene']!=folder.name:raise ValueError('Interaction result belongs to a different scene')
    digest=sha(archive)
    if interval['provenance']['tcp_archive_sha256']!=digest:raise ValueError('Interaction interval refers to different measurements')
    window=interval['interval']['selected_window_s']
    if window is None:raise ValueError('No interaction window to compare')
    with np.load(archive,allow_pickle=False) as z:
        data={k:z[k].copy() for k in ['frame_index','rgbd_frame_index','timestamp_s','tcp_position_m','ee_rotvec_rad','force_N']}
    reference=None;rows=[]
    for label,degrees in [('identity_reference',0),('sensor_z_plus90',90)]:
        trial,matrix=rotate_sensor_forces(data,degrees)
        p=prepare(trial,'native',None,window,10,.05,'direction',.025,.01)
        if reference is None:reference=p
        else:
            np.testing.assert_allclose(p['force_norm_N'],reference['force_norm_N'],atol=1e-12,rtol=1e-12)
            np.testing.assert_array_equal(p['selected'],reference['selected'])
            np.testing.assert_array_equal(p['drawn'],reference['drawn'])
        target=output/folder.name/label;target.mkdir(parents=True,exist_ok=False)
        selected=p['selected'];ids=data['frame_index'][selected]
        description=f'UNVALIDATED {label}; nominal TCP origin; native {ids[0]}–{ids[-1]}'
        description+='; original force interval (multiple drawers)' if folder.name=='white_drawer_20261021_161154_episode_000002' else '; original force interval'
        save_vectors(target,trial,p)
        # Keep sensor measurements alongside candidate EE/base components.
        with np.load(target/'force_vectors.npz',allow_pickle=False) as z:vectors={k:z[k].copy() for k in z.files}
        vectors['force_sensor_N']=data['force_N'][selected]
        np.savez_compressed(target/'force_vectors.npz',**vectors)
        csv_path=target/'force_vectors.csv'
        with csv_path.open(newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;values=list(reader)
        fields+=['force_sensor_x_N','force_sensor_y_N','force_sensor_z_N']
        for row,force in zip(values,data['force_N'][selected]):row.update(dict(zip(fields[-3:],force)))
        with csv_path.open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(values)
        draw(target,folder.name,trial,p,description,'direction',.025,.01)
        strong=selected&(p['force_norm_N']>=.5)
        a=p['force_base_N'][strong];b=reference['force_base_N'][strong]
        angles=np.degrees(np.arccos(np.clip(np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-15),-1,1)))
        meta=dict(scene=folder.name,case=label,status='UNVALIDATED_MOUNTING_TRIAL',tcp_calibration_valid=False,
            R_ee_force_sensor=matrix.tolist(),sensor_z_rotation_deg=degrees,equivalent_rotation_deg=degrees-360,
            force_transform='F_base = R_base_EE @ R_EE_sensor @ F_sensor; active right-handed rotation',
            force_component_mapping='Fx_EE=-Fy_sensor; Fy_EE=Fx_sensor; Fz_EE=Fz_sensor' if degrees else 'identity reference',
            tcp_origin='Existing nominal TCP copied unchanged; old mounting EE-to-contact transform remains unknown',
            nominal_T_ee_tcp=summary.get('tcp_calibration',{}).get('T_ee_tcp'),
            torque_transport_applied=False,force_sign='As recorded; no action/reaction reversal',
            window_s=window,native_frames=[int(ids[0]),int(ids[-1])],arrow_every=10,min_force_N=.05,arrow_length_m=.025,
            strong_force_threshold_N=.5,strong_samples=len(angles),
            direction_change_median_deg=float(np.median(angles)) if len(angles) else None,
            direction_change_p95_deg=float(np.percentile(angles,95)) if len(angles) else None,
            measurement_archive=str(archive),measurement_archive_sha256=digest,
            measurement_metadata_sha256=sha(summary_path),interaction_result_sha256=sha(interval_path),
            source_script_sha256=sha(Path(__file__)),production_calibration_changed=False)
        (target/'manifest.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n');rows.append(meta)
    assert sha(archive)==digest
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements-root',type=Path,required=True)
    parser.add_argument('--interaction-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--scene',nargs='+',required=True)
    args=parser.parse_args(argv)
    roots=[p.expanduser().resolve() for p in [args.measurements_root,args.interaction_root]]
    output=args.output_root.expanduser().resolve()
    for root in roots:
        if output==root or output.is_relative_to(root) or root.is_relative_to(output):parser.error('Trial output must be separate from inputs')
    for name in args.scene:
        if Path(name).name!=name or name in ['.','..']:parser.error('Scene must be a directory name')
        for path in [roots[0]/name/'measurements.npz',roots[0]/name/'summary.json',roots[1]/name/'result.json']:
            if not path.is_file():parser.error('Missing input: '+str(path))
    if output.exists():parser.error('Choose a fresh trial output directory; existing results are preserved')
    output.mkdir(parents=True);os.environ['MPLCONFIGDIR']=str(output/'.matplotlib')
    rows=[]
    for name in args.scene:
        rows.extend(compare_scene(roots[0]/name,roots[1],output))
        print(name+': identity and +90 deg Z trials saved',flush=True)
    (output/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    report=['# White-drawer force-mounting rotation trial','',
        '+90 degrees and -270 degrees about EE Z are equivalent. The candidate matrix maps sensor X to EE +Y, sensor Y to EE -X, and preserves Z.',
        '`F_base = R_base_EE @ [[0,-1,0],[1,0,0],[0,0,1]] @ F_sensor`','',
        'These trials retain the previous frames, force threshold, arrow spacing, magnitudes and nominal TCP positions. Both the identity reference and rotated candidate are unvalidated; no production calibration is changed. The EE-to-contact transform remains unknown. Rotation about Z does not change the dominant axial Fz component.',
        'Episode 2 still combines upper/lower drawer interactions; a force rotation cannot correct that interval or make one joint fit meaningful.','',
        '| Scene | Median force-direction change, force >= 0.5 N | 95th percentile |','|---|---|---|']
    for row in rows:
        if row['sensor_z_rotation_deg']==90:
            angles=[f'{row[k]:.2f} deg' if row[k] is not None else 'No forces >= 0.5 N' for k in ['direction_change_median_deg','direction_change_p95_deg']]
            report.append(f"| {row['scene']} | {angles[0]} | {angles[1]} |")
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>Drawer sensor rotation trial</title><style>body{font:16px sans-serif;margin:24px}table{width:100%}td{width:50%;vertical-align:top}img{max-width:100%}p{max-width:1100px}.warning{padding:16px;background:#fee2e2;color:#991b1b}</style>',
        '<h1>White drawers: +90° Z (= −270° Z) force trial</h1><p class="warning">Unvalidated sensor mounting. TCP arrow origins retain the old nominal offset. These plots do not establish contact-point calibration or force-on-object sign. Episode 2 includes multiple drawers.</p>',
        '<p>Left: previous identity rotation. Right: proposed +90° rotation about EE Z. Frame intervals, positions, arrow lengths and colors are preserved. Click a picture for the interactive plot.</p>']
    for name in args.scene:
        page.append('<h2>'+html.escape(name)+'</h2><table><tr>')
        for case in ['identity_reference','sensor_z_plus90']:
            rel=html.escape(name+'/'+case,quote=True)
            page.append(f'<td><h3>{case}</h3><a href="{rel}/force_directions.html"><img src="{rel}/force_directions.png" alt="{case}"></a></td>')
        page.append('</tr></table>')
    (output/'index.html').write_text('\n'.join(page)+'\n')
    print('Open '+str(output/'index.html'))
    return 0


if __name__=='__main__':raise SystemExit(main())
