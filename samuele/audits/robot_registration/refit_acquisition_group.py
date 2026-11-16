#!/usr/bin/env python3
"""Refit a user-confirmed acquisition group from saved observations, without rescanning RGB."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from audit_robot_registration import (background_control, camera_arrays, errors,
    fit_pooled, overlay, project, write_json)


def load_group(audit_root, recording_prefix):
    infos, rows, coverage, hashes = {}, [], [], {}
    for path in sorted(audit_root.glob('*/inventory.json')):
        info = json.loads(path.read_text())
        if not info['recording'].startswith(recording_prefix):
            continue
        infos[info['scene']] = info
        observations = path.parent / 'observations.jsonl'
        for source in [path, observations]:
            hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        records = [json.loads(line) for line in observations.read_text().splitlines()]
        rows.extend(r for r in records if r['status'] == 'accepted')
        for cam in info['cameras']:
            subset = [r for r in records if r['camera_name'] == cam['name']]
            counts = Counter(r['status'] for r in subset)
            coverage.append(dict(scene=info['scene'], camera=cam['name'], attempted=len(subset),
                                 accepted=counts['accepted'], statuses=dict(counts)))
    if not infos:
        raise ValueError(f'No recordings match prefix {recording_prefix!r}')
    if len({i['calibration_hash'] for i in infos.values()}) != 1:
        raise ValueError('Group contains different supplied calibration models; explicitly separate configurations')
    return infos, rows, coverage, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--recording-prefix', required=True)
    parser.add_argument('--group-id', required=True)
    parser.add_argument('--group-evidence', required=True, help='Source of acquisition grouping, e.g. operator statement')
    parser.add_argument('--bootstrap', type=int, default=24)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error('Bootstrap must be nonnegative')
    audit_root, output = args.audit_root.resolve(), args.output_root.resolve()
    if output == audit_root or output in audit_root.parents or audit_root in output.parents:
        parser.error('Use a separate new output root to preserve the previous audit')
    if output.exists() and any(output.iterdir()):
        parser.error('Output root is not empty; choose a fresh folder')
    infos, rows, coverage, hashes = load_group(audit_root, args.recording_prefix)
    for info in infos.values():
        scene = Path(info['scene_path']).resolve()
        if output == scene or output in scene.parents or scene in output.parents:
            parser.error('Output must be separate from original scene trees')
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    write_json(output / 'group.json', dict(group_id=args.group_id, evidence=args.group_evidence,
        grouping_changes='Only the selected acquisition group is pooled; earlier recordings excluded',
        recordings=sorted({i['recording'] for i in infos.values()}), scenes=sorted(infos),
        source_audit=str(audit_root), production_modified=False, input_sha256=hashes,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
    write_json(output / 'coverage.json', coverage)
    print(f'{len(infos)} scenes; {len(rows)} accepted observations. Fitting {args.group_id}', flush=True)
    result, x = fit_pooled(rows, infos, args.bootstrap)
    result.update(group_id=args.group_id, evidence=args.group_evidence,
        validation_unit='withhold one complete object/native recording at a time')
    write_json(output / 'candidate.json', result)
    print('Background checks against an anchor inside this acquisition group', flush=True)
    background = background_control(audit_root, infos, camera_arrays, project)
    write_json(output / 'background_stability.json', background)
    scene_metrics = []
    if x is not None:
        for scene, info in infos.items():
            subset = [r for r in rows if r['scene'] == scene]
            for camera in info['cameras']:
                camera_rows = [r for r in subset if r['camera_name'] == camera['name']]
                scene_metrics.append(dict(scene=scene, camera=camera['name'], before=errors(np.zeros(6), camera_rows, infos),
                    after=errors(x, camera_rows, infos), scope='includes training; use heldout-object folds for validation'))
            if subset:
                (output / scene).mkdir(exist_ok=True)
                overlay(scene, infos, subset, x, output)
    write_json(output / 'scene_camera_metrics.json', scene_metrics)
    report = [f'# Acquisition group: {args.group_id}', '', f'Grouping evidence: {args.group_evidence}', '',
        f'{len(infos)} scenes; {len(coverage)} scene/camera pairs; {sum(c["attempted"] for c in coverage)} images attempted in the source audit; {len(rows)} accepted observations.', '',
        f'Status: **{result["status"]}**. No production calibration or dataset was changed.', '',
        'This supersedes the previous calibration-hash pool for these recordings: older sessions are excluded. '
        'An unsuccessful common fit is not evidence of separate acquisition sessions. Detector errors, insufficient geometry, '
        'per-camera calibration errors, tool-reference and timing discrepancies remain possible.', '',
        '| Withheld object/recording | Median before / after (px) | p95 before / after (px) |', '|---|---:|---:|']
    for fold in result.get('leave_recording_out', []):
        report.append(f'| {fold["withheld_recording"]} | {fold["before"]["median_px"]:.2f} / {fold["after"]["median_px"]:.2f} | '
                      f'{fold["before"]["p95_px"]:.2f} / {fold["after"]["p95_px"]:.2f} |')
    report += ['', 'Background check counts: ' + json.dumps(dict(Counter(r['status'] for r in background['results']))), '',
        'candidate.json preserves the matrix, checks, whole-object holdout scores and recording-cluster bootstrap uncertainty. '
        'Per-scene candidate_overlay.jpg files show cyan nominal / yellow fitted / green detected shaft. '
        'They include training observations and heuristic detections; these are not certified TCP/contact-point accuracy.', '',
        'The acquisition grouping is supplied by the operator. Common acquisition does not by itself prove an error-free calibration or correct image-line identities. '
        'The fitter and detector are unchanged from the source audit; only the grouping and background anchor change.']
    (output / 'REPORT.md').write_text('\n'.join(report) + '\n')
    links = ''.join(f'<li><a href="{scene}/candidate_overlay.jpg">{scene}</a></li>' for scene in infos if (output / scene / 'candidate_overlay.jpg').exists())
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Acquisition group audit</title>'
        '<h1>Candidate overlays — uninstalled calibration</h1><p>See REPORT.md and candidate.json for held-out validation and status.</p><ul>' + links + '</ul>')
    print(f'{result["status"]}; report: {output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
