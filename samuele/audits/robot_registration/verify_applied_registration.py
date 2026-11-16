#!/usr/bin/env python3
"""Verify the installed dataset registration and write EE/TCP projection previews."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np

from apply_robot_base_registration import MARKER, sha, verify_applied


def project(P, C, K):
    q = (P - C[:3, 3]) @ C[:3, :3]
    uv = np.full((len(P), 2), np.nan)
    valid = np.isfinite(q).all(axis=1) & (q[:, 2] > .02)
    h = q[valid] @ K.T
    uv[valid] = h[:, :2] / h[:, 2:]
    return uv


def preview(scene, before, measurements, output):
    old = json.loads((before / 'metadata/cameras.json').read_text())['cameras']
    new = json.loads((scene / 'metadata/cameras.json').read_text())['cameras']
    conversion = json.loads((scene / 'metadata/real_conversion.json').read_text())
    with np.load(measurements / scene.name / 'measurements.npz', allow_pickle=False) as d:
        ids = np.flatnonzero(d['rgbd_frame_index'] >= 0)
        chosen = ids[np.unique(np.linspace(0, len(ids) - 1, 3).astype(int))]
        frames = d['rgbd_frame_index'][chosen].astype(int)
        native = d['frame_index'][chosen].astype(int)
        ee = d['ee_position_m'][chosen]
        tcp = d['tcp_position_m'][chosen]
    rows, receipt = [], []
    for j, frame in enumerate(frames):
        columns = []
        for a, b in zip(old, new):
            if a['name'] != b['name']:
                raise ValueError('Camera order changed')
            image = cv2.imread(str(scene / 'rgb' / a['name'] / f'frame_{frame:06d}.png'))
            if image is None:
                raise FileNotFoundError('Missing RGB preview frame')
            K = np.array([[a['fx'], 0, a['cx']], [0, a['fy'], a['cy']], [0, 0, 1.]])
            uv_before = project(np.array([ee[j], tcp[j]]), np.array(a['pose_matrix_4x4']), K)
            uv_after = project(np.array([ee[j], tcp[j]]), np.array(b['pose_matrix_4x4']), K)
            for uv, color in [(uv_before, (255, 255, 0)), (uv_after, (0, 255, 255))]:
                if np.isfinite(uv).all() and abs(uv).max() < 10000:
                    points = np.rint(uv).astype(int)
                    cv2.line(image, tuple(points[0]), tuple(points[1]), color, 2)
                    cv2.drawMarker(image, tuple(points[0]), color, cv2.MARKER_CROSS, 12, 2)
                    cv2.circle(image, tuple(points[1]), 4, color, 1)
            cv2.putText(image, f'{a["name"]}: RGB-D {frame}, native {native[j]}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            columns.append(image)
            receipt.append(dict(camera=a['name'], rgbd_frame=int(frame), native_frame=int(native[j]),
                ee_base_m=ee[j].tolist(), tcp_base_m=tcp[j].tolist(),
                before_uv_px=np.where(np.isfinite(uv_before), uv_before, None).tolist(),
                after_uv_px=np.where(np.isfinite(uv_after), uv_after, None).tolist()))
        rows.append(np.hstack(columns))
    image = np.vstack(rows)
    cv2.putText(image, 'Cyan: original EE-to-TCP axis; yellow: registered. Cross=link6, circle=TCP. Applied candidate, not ground truth.',
                (12, image.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, .63, (255, 255, 255), 2)
    cv2.imwrite(str(output / f'{scene.name}.jpg'), image)
    (output / f'{scene.name}.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--converted-root', type=Path, required=True)
    p.add_argument('--group-root', type=Path, required=True)
    p.add_argument('--report-root', type=Path, required=True)
    p.add_argument('--measurements-root', type=Path, required=True)
    a = p.parse_args()
    group = json.loads((a.group_root / 'group.json').read_text())
    candidate = json.loads((a.group_root / 'candidate.json').read_text())
    G = np.array(candidate['G_left_multiply_camera_to_base'])
    receipts, previews = [], {}
    cv2.setNumThreads(1)
    preview_dir = a.report_root / 'ee_tracking_previews'; preview_dir.mkdir(exist_ok=True)
    for name in group['scenes']:
        scene = a.converted_root / name
        marker = verify_applied(scene, G)
        for rel, digest in marker['untouched_sha256'].items():
            if sha(scene / rel) != digest:
                raise AssertionError(f'Original robot/raw metadata changed: {scene / rel}')
        job = Path(marker['backup']).parent
        state = json.loads((job / 'state.json').read_text())
        if state['status'] != 'complete':
            raise AssertionError('Incomplete registration')
        for rel, digest in state['before_sha256'].items():
            if sha(job / 'before' / rel) != digest:
                raise AssertionError(f'Original backup hash mismatch: {job / "before" / rel}')
        metadata = json.loads((scene / 'metadata/cameras.json').read_text())
        original = json.loads((job / 'before/metadata/cameras.json').read_text())
        for old, new in zip(original['cameras'], metadata['cameras']):
            if not np.allclose(new['pose_matrix_4x4'], G @ np.array(old['pose_matrix_4x4']), atol=1e-12, rtol=0):
                raise AssertionError('Camera registration was not applied exactly once')
        receipts.append(dict(scene=name, files_verified=len(marker['after_sha256']),
            backup_files_verified=len(state['before_sha256']),
            unchanged_robot_and_raw_files=len(marker['untouched_sha256']),
            model_count=len(state['checks']['model_checks']), ply_count=len(state['checks']['ply_checks']),
            database_pose_priors=sum(state['checks']['database_priors'].values()), backup=marker['backup']))
        family = name.rsplit('_episode_', 1)[0]
        if family not in previews:
            preview(scene, job / 'before', a.measurements_root, preview_dir); previews[family] = name
        print(f'Verified {len(receipts)}/{len(group["scenes"])} {name}', flush=True)
    report = dict(status='PASS', scenes=len(receipts), modified_file_hashes_verified=sum(r['files_verified'] for r in receipts),
        backup_file_hashes_verified=sum(r['backup_files_verified'] for r in receipts),
        models=sum(r['model_count'] for r in receipts), ply_files=sum(r['ply_count'] for r in receipts),
        database_pose_priors=sum(r['database_pose_priors'] for r in receipts),
        unchanged_robot_raw_files=sum(r['unchanged_robot_and_raw_files'] for r in receipts),
        candidate_status=candidate['status'], G_left_multiply_camera_to_base=G.tolist(), previews=list(previews.values()), records=receipts)
    (a.report_root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    text = ['# Installed robot-base registration for final-session datasets', '',
        f'Verified {len(receipts)} scenes, {report["models"]} COLMAP models, {report["ply_files"]} PLY files and {report["database_pose_priors"]} database pose priors.', '',
        'The active metadata/cameras.json files and all saved camera-derived geometry are now transformed into the recorded robot base frame by the same shared final-session G. '
        'Both initial/final states, masked/unmasked models, per-view clouds, RGB-triangulated models, rig/frame poses and YAML/database pose priors are included.', '',
        f'{report["modified_file_hashes_verified"]} active file hashes and {report["unchanged_robot_raw_files"]} untouched robot/raw metadata hashes passed verification. '
        'Each camera matrix equals G times its backed-up original exactly once. Existing intrinsics, IDs, image membership, tracks and 2D observations are preserved.', '',
        'The Parquet link6 measurements, fixed EE-to-TCP tool transforms, timestamps, RGB-D, masks, raw native calibration and base-frame forces were not transformed. '
        'metadata/camera_calibration_robot_base.json is the active native-camera counterpart; metadata/raw/camera_calibration.json remains original acquisition provenance.', '',
        'The applied candidate remains unverified. Application was requested by the operator; it is not a new independent calibration certification. '
        'This operation changes robot/scene registration and preserves relative multiview geometry. It does not establish exact TCP contact accuracy.', '',
        'Backups are in backups/<scene>/before. The updater refuses a second application or edits to registered files; use its restore phase for a hash-checked rollback. '
        'Earlier real recordings and synthetic datasets are excluded. Previously rendered videos/viewers retain their original transforms; regenerate them into a new output root to see the installed registration.', '',
        '## EE/TCP projection previews', '',
        'One episode per object, three time samples and all three cameras. Cyan is the original projected EE-to-TCP axis; yellow uses installed camera metadata. Cross = link6; circle = TCP. These are diagnostic projections, not independent ground truth.', '']
    for name in previews.values():
        text.append(f'- [{name}](ee_tracking_previews/{name}.jpg)')
    (a.report_root / 'REPORT.md').write_text('\n'.join(text) + '\n')
    links = ''.join(f'<h2>{html.escape(n)}</h2><a href="ee_tracking_previews/{n}.jpg"><img loading="lazy" style="max-width:100%" src="ee_tracking_previews/{n}.jpg"></a>' for n in previews.values())
    (a.report_root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Installed robot registration</title>'
        '<h1>Installed robot registration: EE/TCP projections</h1><p>Cyan: original; yellow: installed candidate. '
        'Robot-native link6 and TCP positions are unchanged. The applied candidate remains unverified.</p><a href="REPORT.md">Report and backup details</a>' + links)


if __name__ == '__main__':
    main()
