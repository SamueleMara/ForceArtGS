#!/usr/bin/env python3
"""Read-only, all-recording robot/camera registration experiment. Never installs a calibration."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from background_stability import run as background_control

VERSION = 1


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def camera_arrays(cameras):
    C = np.array([c['pose_matrix_4x4'] for c in cameras], float)
    K = np.array([[[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1]] for c in cameras])
    if not np.isfinite(C).all() or not np.allclose(np.linalg.det(C[:, :3, :3]), 1, atol=1e-5):
        raise ValueError('Invalid camera-to-base matrices')
    return C, K


def project(points, C, K):
    q = (np.asarray(points) - C[:3, 3]) @ C[:3, :3]
    if (q[:, 2] <= .02).any():
        return None
    uv = q @ K.T
    return uv[:, :2] / uv[:, 2:]


def detect_shaft(image, uv):
    """Isolated dark shaft cross-sections; ambiguous/poor fits stay rejected.

    shortcut: proposal uses nominal calibration; this is not an independent tip detector.
    Large registration errors and non-black/occluded tools need another detector.
    """
    if uv is None or not np.isfinite(uv).all():
        return None, 'behind_camera_or_invalid_pose'
    length = np.linalg.norm(uv[1] - uv[0])
    if length < 14 or length > 160:
        return None, 'projected_shaft_too_short_or_long'
    u = (uv[1] - uv[0]) / length
    v = np.array([-u[1], u[0]])
    centers = uv[0] + np.arange(0, length)[:, None] * u
    offsets = np.arange(-45, 46)
    xy = np.rint(centers[:, None] + offsets[None, :, None] * v).astype(int)
    valid = (xy[:, :, 0] >= 0) & (xy[:, :, 0] < image.shape[1]) & (xy[:, :, 1] >= 0) & (xy[:, :, 1] < image.shape[0])
    dark = np.zeros(valid.shape, bool)
    dark[valid] = np.max(image[xy[:, :, 1][valid], xy[:, :, 0][valid]], axis=1) < 65
    points, widths = [], []
    for center, values in zip(centers, dark):
        starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
        ends = np.flatnonzero(values & ~np.r_[values[1:], False])
        runs = [(a, b) for a, b in zip(starts, ends) if 3 <= b - a + 1 <= 18]
        if len(runs) == 1:
            a, b = runs[0]
            points.append(center + offsets[a:b + 1].mean() * v)
            widths.append(b - a + 1)
    if len(points) < 12:
        return None, 'too_few_unambiguous_cross_sections'
    points = np.array(points)
    line_fit = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
    direction, center = line_fit[:2].astype(float), line_fit[2:].astype(float)
    direction /= np.linalg.norm(direction)
    if direction @ u < 0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    residual = (points - center) @ normal
    mad = float(np.median(abs(residual - np.median(residual))))
    if mad > .75:
        return None, 'nonlinear_or_mixed_cross_sections'
    if direction @ u < np.cos(np.deg2rad(20)):
        return None, 'direction_incompatible_with_proposal'
    if np.ptp(points @ direction) < 12:
        return None, 'insufficient_line_span'
    return dict(line=np.r_[normal, -normal @ center].tolist(), center=center.tolist(),
                direction=direction.tolist(), mad_px=mad, cross_sections=len(points),
                width_px=float(np.median(widths))), 'accepted'


def scene_info(scene, measurements):
    conversion = json.loads((scene / 'metadata/real_conversion.json').read_text())
    cameras = json.loads((scene / 'metadata/cameras.json').read_text())['cameras']
    raw = json.loads((scene / 'metadata/raw/camera_calibration.json').read_text())
    C, K = camera_arrays(cameras)
    calibration_hash = hashlib.sha256(json.dumps(dict(C=C.tolist(), K=K.tolist(),
        serials={k: v.get('serial') for k, v in raw['cameras'].items()}), sort_keys=True).encode()).hexdigest()
    summary = json.loads((measurements / scene.name / 'summary.json').read_text())
    return dict(scene=scene.name, scene_path=str(scene), recording=Path(conversion['source']).name,
                source=conversion['source'], episode=conversion['episode'], cameras=cameras,
                calibration_hash=calibration_hash, calibration_session=raw.get('source', {}).get('session'),
                calibration_status=raw.get('status'), tool=summary.get('tool'),
                source_parquet_sha256=summary.get('source_parquet_sha256'))


def extract_scene(job):
    scene_path, measurements, output, stride = job
    scene, measurements, output = Path(scene_path), Path(measurements), Path(output)
    cv2.setNumThreads(1)
    dest = output / scene.name
    dest.mkdir(parents=True, exist_ok=True)
    info = scene_info(scene, measurements)
    source_hash = hashlib.sha256(Path(__file__).read_bytes())
    for path in [scene / 'metadata/cameras.json', measurements / scene.name / 'measurements.npz']:
        source_hash.update(path.read_bytes())
    images = sorted(scene.glob('rgb/cam_*/frame_*.png'))
    for p in images:
        st = p.stat()
        source_hash.update(f'{p.relative_to(scene)}:{st.st_size}:{st.st_mtime_ns}'.encode())
    fingerprint = source_hash.hexdigest() + f':stride={stride}'
    marker = dest / 'extraction.json'
    if marker.exists() and (dest / 'observations.jsonl').exists():
        previous = json.loads(marker.read_text())
        if previous.get('fingerprint') == fingerprint:
            return dict(scene=scene.name, reused=True, **previous['counts'])
        raise ValueError(f'{scene.name}: cached inputs/code changed; use a fresh audit output root')
    d = np.load(measurements / scene.name / 'measurements.npz', allow_pickle=False)
    indices = np.flatnonzero(d['rgbd_frame_index'] >= 0)[::stride]
    frames = d['rgbd_frame_index'][indices].astype(int)
    if len(set(frames)) != len(frames):
        raise ValueError('Duplicate RGB-D/native pose mapping')
    C, K = camera_arrays(info['cameras'])
    axes = Rotation.from_rotvec(d['ee_rotvec_rad'][indices]).as_matrix()[:, :, 2]
    positions = d['ee_position_m'][indices]
    counts = Counter()
    temp = dest / 'observations.jsonl.tmp'
    with temp.open('w') as f:
        for j, (idx, frame) in enumerate(zip(indices, frames)):
            p, axis = positions[j], axes[j]
            for c, camera in enumerate(info['cameras']):
                record = dict(scene=scene.name, camera=c, camera_name=camera['name'], frame=int(frame),
                    native_frame=int(d['frame_index'][idx]), timestamp_s=float(d['timestamp_s'][idx]),
                    p=p.tolist(), axis=axis.tolist(), speed_m_s=float(d['linear_speed_m_s'][idx]) if np.isfinite(d['linear_speed_m_s'][idx]) else None)
                # The early bare force-sensor assembly has no pointed black shaft.
                if scene.name.startswith('white_drawer_20261021_161154'):
                    fit, status = None, 'unsupported_bare_sensor_geometry'
                else:
                    path = scene / 'rgb' / camera['name'] / f'frame_{frame:06d}.png'
                    image = cv2.imread(str(path))
                    if image is None:
                        fit, status = None, 'missing_or_unreadable_rgb'
                    else:
                        # Include the distal shaft: the laptop tool's proximal section is white-wrapped.
                        uv = project(p + np.outer([.10, .18], axis), C[c], K[c])
                        fit, status = detect_shaft(image, uv)
                record['status'] = status
                if fit:
                    record.update(fit)
                counts[status] += 1
                f.write(json.dumps(record, allow_nan=False) + '\n')
    temp.replace(dest / 'observations.jsonl')
    write_json(dest / 'inventory.json', info)
    result = dict(fingerprint=fingerprint, counts=dict(counts), attempted=len(indices) * len(C),
                  source_rgb_images=len(images), frame_stride=stride)
    write_json(marker, result)
    return dict(scene=scene.name, reused=False, **counts)


def transform(x):
    G = np.eye(4)
    G[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    G[:3, 3] = np.asarray(x[3:]) * .1
    return G


def arrays(rows, infos):
    C, K = [], []
    for row in rows:
        ci, ki = camera_arrays(infos[row['scene']]['cameras'])
        C.append(ci[row['camera']]); K.append(ki[row['camera']])
    return dict(p=np.array([r['p'] for r in rows]), axis=np.array([r['axis'] for r in rows]),
                line=np.array([r['line'] for r in rows]), C=np.array(C), K=np.array(K))


def residuals(x, a):
    G = transform(x)
    P = a['p'][:, None] + np.array([.075, .15])[None, :, None] * a['axis'][:, None]
    Q = (P - G[:3, 3]) @ G[:3, :3]
    Q = np.einsum('nsi,nij->nsj', Q - a['C'][:, None, :3, 3], a['C'][:, :3, :3])
    U = np.einsum('nsi,nji->nsj', Q, a['K'])
    uv = U[:, :, :2] / np.maximum(U[:, :, 2:], .02)
    return np.sum(uv * a['line'][:, None, :2], axis=2) + a['line'][:, None, 2]


def balanced(rows, limit=24):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['scene'], r['camera'])].append(r)
    result = []
    for group in groups.values():
        group = sorted(group, key=lambda r: r['frame'])
        result.extend(group[i] for i in np.unique(np.linspace(0, len(group) - 1, min(limit, len(group))).astype(int)))
    return result


def solve(rows, infos, start=None):
    a = arrays(rows, infos)
    result = least_squares(lambda x: residuals(x, a).ravel(), np.zeros(6) if start is None else start,
        bounds=([- .15] * 3 + [-1.5] * 3, [.15] * 3 + [1.5] * 3),
        loss='soft_l1', f_scale=1., max_nfev=120, xtol=1e-9, ftol=1e-9, gtol=1e-9)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-4))
    condition = float(singular[0] / max(singular[-1], 1e-12))
    return result.x, dict(success=bool(result.success), rank=rank, scaled_jacobian_condition=condition,
        singular_values=singular.tolist(), scaling='rotation radians; translation / 0.1 m',
        at_parameter_bound=bool(np.any(np.abs(result.active_mask) > 0)))


def errors(x, rows, infos):
    if not rows:
        return dict(count=0, median_px=None, p95_px=None)
    e = np.mean(abs(residuals(x, arrays(rows, infos))), axis=1)
    return dict(count=len(e), median_px=float(np.median(e)), p95_px=float(np.percentile(e, 95)))


def fit_recording(rows, infos, bootstrap):
    scenes = sorted({r['scene'] for r in rows})
    if len(rows) < 60 or len({r['camera'] for r in rows}) < 3:
        return dict(status='UNVERIFIED_INSUFFICIENT_OBSERVATIONS', observations=len(rows)), None
    # Whole episodes are withheld. A lone episode instead withholds contiguous time blocks.
    heldout_scenes = set(scenes[1::3]) if len(scenes) > 1 else set()
    def test(r):
        return r['scene'] in heldout_scenes if heldout_scenes else (r['frame'] // 40) % 3 == 1
    train = balanced([r for r in rows if not test(r)])
    heldout = [r for r in rows if test(r)]
    if len(train) < 30 or len(heldout) < 30:
        return dict(status='UNVERIFIED_INSUFFICIENT_HOLDOUT', observations=len(rows)), None
    x, ident = solve(train, infos)
    metrics = dict(before=errors(np.zeros(6), heldout, infos), after=errors(x, heldout, infos))
    cams = {}
    for c in sorted({r['camera_name'] for r in rows}):
        subset = [r for r in heldout if r['camera_name'] == c]
        cams[c] = dict(before=errors(np.zeros(6), subset, infos), after=errors(x, subset, infos))
    # Cluster bootstrap preserves neighboring-frame dependence instead of resampling pixels.
    clusters = defaultdict(list)
    for r in train:
        clusters[(r['scene'], r['frame'] // 40)].append(r)
    keys = list(clusters)
    samples, rng = [], np.random.default_rng(42)
    if len(keys) >= 6 and ident['success'] and ident['rank'] == 6 and ident['scaled_jacobian_condition'] < 1000 and not ident['at_parameter_bound']:
        for _ in range(bootstrap):
            sample = [r for k in rng.integers(0, len(keys), len(keys)) for r in clusters[keys[k]]]
            fitted, diagnostic = solve(sample, infos, x)
            if diagnostic['success'] and diagnostic['rank'] == 6 and not diagnostic['at_parameter_bound']:
                samples.append(fitted)
    uncertainty = dict(requested=bootstrap, successful=len(samples), clusters=len(keys),
        skip_reason=None if samples else 'no successful bootstrap or initial fit insufficiently constrained')
    if len(samples) >= max(8, bootstrap // 2):
        sample = np.array(samples)
        uncertainty.update(rotation_vector_p025_p975_deg=np.rad2deg(np.percentile(sample[:, :3], [2.5, 97.5], axis=0)).tolist(),
            translation_p025_p975_mm=(np.percentile(sample[:, 3:], [2.5, 97.5], axis=0) * 100).tolist(),
            translation_std_mm=(sample[:, 3:].std(axis=0) * 100).tolist(),
            rotation_vector_std_deg=np.rad2deg(sample[:, :3].std(axis=0)).tolist())
    checks = dict(observable=ident['rank'] == 6 and ident['scaled_jacobian_condition'] < 1000,
        converged=ident['success'] and not ident['at_parameter_bound'],
        improves_heldout=metrics['after']['median_px'] < metrics['before']['median_px'],
        good_heldout_fit=metrics['after']['median_px'] <= 3 and metrics['after']['p95_px'] <= 8,
        all_cameras_covered=all(v['after']['count'] >= 30 for v in cams.values()),
        no_degraded_camera=all(v['after']['count'] and v['after']['median_px'] <= v['before']['median_px'] + .25
                               and v['after']['p95_px'] <= v['before']['p95_px'] + .5 for v in cams.values()),
        bootstrap_stable='translation_std_mm' in uncertainty and max(uncertainty['translation_std_mm']) < 5
                         and max(uncertainty['rotation_vector_std_deg']) < .5)
    status = 'CANDIDATE_PASSES_IMAGE_CHECKS' if all(checks.values()) else 'UNVERIFIED'
    if not checks['observable']:
        status = 'UNIDENTIFIABLE'
    return dict(status=status, observations=len(rows), training_observations=len(train),
        heldout_episodes=sorted(heldout_scenes), holdout_method='whole episodes' if heldout_scenes else 'contiguous 40-frame blocks',
        G_left_multiply_camera_to_base=transform(x).tolist(), rotation_vector_deg=np.rad2deg(x[:3]).tolist(),
        translation_mm=(x[3:] * 100).tolist(), identifiability=ident, uncertainty=uncertainty,
        checks=checks, heldout=metrics, heldout_by_camera=cams), x


def fit_pooled(rows, infos, bootstrap):
    """Explicit common-mount hypothesis, challenged by leaving whole recordings out."""
    recordings = sorted({infos[r['scene']]['recording'] for r in rows})
    if len(recordings) < 3:
        return dict(status='UNVERIFIED_TOO_FEW_RECORDINGS'), None
    selected = balanced(rows, limit=12)
    x, ident = solve(selected, infos)
    folds = []
    for recording in recordings:
        train = [r for r in selected if infos[r['scene']]['recording'] != recording]
        test = [r for r in rows if infos[r['scene']]['recording'] == recording]
        fx, fi = solve(train, infos)
        folds.append(dict(withheld_recording=recording, training_identifiability=fi,
            G=transform(fx).tolist(), before=errors(np.zeros(6), test, infos), after=errors(fx, test, infos),
            cameras={c: dict(before=errors(np.zeros(6), [r for r in test if r['camera_name'] == c], infos),
                            after=errors(fx, [r for r in test if r['camera_name'] == c], infos))
                     for c in sorted({r['camera_name'] for r in rows})}))
    clusters = [[r for r in selected if infos[r['scene']]['recording'] == name] for name in recordings]
    samples, rng = [], np.random.default_rng(42)
    for _ in range(bootstrap):
        sample = [r for k in rng.integers(0, len(clusters), len(clusters)) for r in clusters[k]]
        fx, fi = solve(sample, infos, x)
        if fi['success'] and fi['rank'] == 6 and fi['scaled_jacobian_condition'] < 1000 and not fi['at_parameter_bound']:
            samples.append(fx)
    uncertainty = dict(requested=bootstrap, successful=len(samples), clusters=len(clusters), unit='native recording')
    if len(samples) >= 8:
        s = np.array(samples)
        uncertainty.update(rotation_vector_std_deg=np.rad2deg(s[:, :3].std(axis=0)).tolist(),
            translation_std_mm=(s[:, 3:].std(axis=0) * 100).tolist(),
            rotation_vector_p025_p975_deg=np.rad2deg(np.percentile(s[:, :3], [2.5, 97.5], axis=0)).tolist(),
            translation_p025_p975_mm=(np.percentile(s[:, 3:], [2.5, 97.5], axis=0) * 100).tolist())
    checks = dict(observable=ident['rank'] == 6 and ident['scaled_jacobian_condition'] < 1000,
        converged=ident['success'] and not ident['at_parameter_bound'],
        all_folds_identifiable=all(f['training_identifiability']['rank'] == 6 and f['training_identifiability']['scaled_jacobian_condition'] < 1000 and f['training_identifiability']['success'] for f in folds),
        improves_every_recording=all(f['after']['median_px'] <= f['before']['median_px'] for f in folds),
        good_every_recording=all(f['after']['median_px'] <= 3 and f['after']['p95_px'] <= 8 for f in folds),
        no_degraded_camera=all(v['after']['count'] >= 30 and v['after']['median_px'] <= v['before']['median_px'] + .25
            and v['after']['p95_px'] <= v['before']['p95_px'] + .5 for f in folds for v in f['cameras'].values()),
        bootstrap_stable=len(samples) >= max(8, int(.8 * bootstrap)) and 'translation_std_mm' in uncertainty
            and max(uncertainty['translation_std_mm']) < 5 and max(uncertainty['rotation_vector_std_deg']) < .5)
    status = 'CANDIDATE_PASSES_IMAGE_CHECKS' if all(checks.values()) else 'UNVERIFIED_COMMON_MOUNT_HYPOTHESIS'
    if not checks['observable']:
        status = 'UNIDENTIFIABLE'
    return dict(status=status, recordings=recordings, observations=len(rows), training_observations=len(selected),
        G_left_multiply_camera_to_base=transform(x).tolist(), translation_mm=(x[3:] * 100).tolist(),
        rotation_vector_deg=np.rad2deg(x[:3]).tolist(), identifiability=ident, checks=checks,
        leave_recording_out=folds, uncertainty=uncertainty,
        all_observations_before=errors(np.zeros(6), rows, infos), all_observations_after=errors(x, rows, infos)), x


def overlay(scene, infos, rows, x, output):
    info = infos[scene]
    C, K = camera_arrays(info['cameras'])
    tiles = []
    for c, camera in enumerate(info['cameras']):
        selected = sorted([r for r in rows if r['camera'] == c], key=lambda r: r['frame'])
        panels = []
        if selected:
            for i in np.linspace(0, len(selected) - 1, 3).astype(int):
                r = selected[i]
                image = cv2.imread(str(Path(info['scene_path']) / 'rgb' / camera['name'] / f'frame_{r["frame"]:06d}.png'))
                p, axis = np.array(r['p']), np.array(r['axis'])
                for G, color in [(np.eye(4), (255, 255, 0)), (transform(x), (0, 255, 255))]:
                    uv = project(p + np.outer([0, .1876174], axis), G @ C[c], K[c])
                    if uv is not None and np.isfinite(uv).all() and np.max(abs(uv)) < 10000:
                        cv2.line(image, tuple(np.rint(uv[0]).astype(int)), tuple(np.rint(uv[1]).astype(int)), color, 2)
                center, direction = np.array(r['center']), np.array(r['direction'])
                ends = np.rint(center + np.outer([-35, 35], direction)).astype(int)
                cv2.line(image, tuple(ends[0]), tuple(ends[1]), (0, 255, 0), 1)
                cv2.putText(image, f'{camera["name"]} frame {r["frame"]}', (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                panels.append(image)
        while len(panels) < 3:
            panels.append(np.zeros((480, 640, 3), np.uint8))
        tiles.append(np.hstack(panels))
    image = np.vstack(tiles)
    cv2.putText(image, 'Cyan: nominal   Yellow: fitted candidate   Green: detected shaft (heuristic)',
                (12, image.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    cv2.imwrite(str(output / scene / 'candidate_overlay.jpg'), image)


def fit_all(output, bootstrap):
    infos, rows, coverage = {}, [], []
    for path in sorted(output.glob('*/inventory.json')):
        info = json.loads(path.read_text()); infos[info['scene']] = info
        observations = [json.loads(line) for line in (path.parent / 'observations.jsonl').read_text().splitlines()]
        rows.extend(r for r in observations if r['status'] == 'accepted')
        for camera in info['cameras']:
            attempted = [r for r in observations if r['camera_name'] == camera['name']]
            counts = Counter(r['status'] for r in attempted)
            coverage.append(dict(scene=info['scene'], camera=camera['name'], attempted=len(attempted),
                accepted=counts['accepted'], acceptance_fraction=counts['accepted'] / max(1, len(attempted)), rejection_counts=dict(counts)))
    print('Checking independent background feature stability across native recordings', flush=True)
    write_json(output / 'background_stability.json', background_control(output, infos, camera_arrays, project))
    groups = defaultdict(list)
    for r in rows:
        info = infos[r['scene']]
        groups[(info['recording'], info['calibration_hash'])].append(r)
    results, fits = {}, {}
    all_groups = sorted({(i['recording'], i['calibration_hash']) for i in infos.values()})
    for recording, calibration in all_groups:
        key = recording + '__' + calibration[:8]
        group_rows = groups[(recording, calibration)]
        print(f'Fit {recording}: {len(group_rows)} accepted observations', flush=True)
        result, x = fit_recording(group_rows, infos, bootstrap)
        result.update(recording=recording, calibration_hash=calibration,
            scenes=sorted(i['scene'] for i in infos.values() if i['recording'] == recording and i['calibration_hash'] == calibration))
        results[key] = result
        if x is not None:
            fits[key] = x
    pooled, pooled_fits = {}, {}
    for calibration in sorted({i['calibration_hash'] for i in infos.values()}):
        selected = [r for r in rows if infos[r['scene']]['calibration_hash'] == calibration]
        print(f'Pooled common-mount hypothesis {calibration[:8]}: {len(selected)} observations', flush=True)
        result, x = fit_pooled(selected, infos, bootstrap)
        pooled[calibration] = result
        if x is not None:
            pooled_fits[calibration] = x
    # Out-of-recording transfer is a control, not evidence that equal calibration files imply a fixed rig.
    transfer = []
    for source, x in fits.items():
        cal = results[source]['calibration_hash']
        for target, result in results.items():
            if source == target or result['calibration_hash'] != cal:
                continue
            test = [r for r in rows if r['scene'] in result['scenes']]
            transfer.append(dict(source=source, target=target, before=errors(np.zeros(6), test, infos), after=errors(x, test, infos)))
    scene_metrics = []
    for scene, info in infos.items():
        key = info['recording'] + '__' + info['calibration_hash'][:8]
        subset = [r for r in rows if r['scene'] == scene]
        x = fits.get(key)
        for camera in info['cameras']:
            rr = [r for r in subset if r['camera_name'] == camera['name']]
            m = dict(scene=scene, camera=camera['name'], candidate_status=results[key]['status'] if rr else 'UNVERIFIED_NO_USABLE_SHAFT',
                before=errors(np.zeros(6), rr, infos), after=errors(x, rr, infos) if x is not None else None,
                validation_scope='all accepted observations; includes training, use recording heldout for validation')
            scene_metrics.append(m)
        if x is not None:
            overlay(scene, infos, subset, x, output)
        if subset and info['calibration_hash'] in pooled_fits:
            pooled_x = pooled_fits[info['calibration_hash']]
            overlay(scene, infos, subset, pooled_x, output)
            (output / scene / 'candidate_overlay.jpg').replace(output / scene / 'pooled_candidate_overlay.jpg')
            if x is not None:
                overlay(scene, infos, subset, x, output)
    write_json(output / 'candidate_corrections.json', results)
    write_json(output / 'coverage.json', coverage)
    write_json(output / 'scene_camera_metrics.json', scene_metrics)
    write_json(output / 'cross_recording_transfer.json', transfer)
    write_json(output / 'pooled_common_mount_hypothesis.json', pooled)
    with (output / 'scene_camera_metrics.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['scene', 'camera', 'candidate_status', 'accepted', 'before_median_px', 'after_median_px', 'before_p95_px', 'after_p95_px'])
        for m in scene_metrics:
            a = m['after'] or {}
            writer.writerow([m['scene'], m['camera'], m['candidate_status'], m['before']['count'], m['before']['median_px'], a.get('median_px'), m['before']['p95_px'], a.get('p95_px')])
    attempted = sum(c['attempted'] for c in coverage)
    accepted = sum(c['accepted'] for c in coverage)
    lines = ['# All-recording robot/camera registration audit', '',
        f'{len(infos)} scenes; {len(coverage)} scene/camera pairs; {attempted} attempted images; {accepted} accepted shaft observations ({accepted / max(1, attempted):.1%}).', '',
        'Original calibration, RGB-D, masks, clouds, robot poses and production outputs were not changed. All transforms below are candidates only.', '',
        '| Native recording | Status | Held-out median before / after (px) | Held-out p95 before / after (px) |',
        '|---|---|---|---|']
    for result in results.values():
        m = result.get('heldout')
        med = f"{m['before']['median_px']:.2f} / {m['after']['median_px']:.2f}" if m else 'unavailable'
        tail = f"{m['before']['p95_px']:.2f} / {m['after']['p95_px']:.2f}" if m else 'unavailable'
        lines.append(f"| {result['recording']} | {result['status']} | {med} | {tail} |")
    lines += ['', '## Pooled common-mount hypothesis', '']
    for calibration, p in pooled.items():
        lines.append(f"- Calibration {calibration[:8]}: {p['status']}")
        for f in p.get('leave_recording_out', []):
            lines.append(f"  - Withheld {f['withheld_recording']}: median {f['before']['median_px']:.2f} → {f['after']['median_px']:.2f} px; p95 {f['before']['p95_px']:.2f} → {f['after']['p95_px']:.2f} px.")
    lines += ['', '## Interpretation and limits', '',
        'Each native recording/calibration hash is fitted separately. This does not establish physical stability within a recording. ' 
        'Equal calibration files do not establish equal physical mounts. Cross-recording transfer is saved separately; no shared correction is installed.', '',
        'The detector searches within 45 pixels of the nominal projected black shaft and can confuse dark edges with the tool. ' 
        'All rejected frames and cameras remain in coverage.json and observations.jsonl. The three early drawer scenes have no compatible pointed shaft and are explicitly unsupported. ' 
        'Image checks and cluster bootstrap measure consistency conditional on this detector, not absolute calibration truth.', '',
        'Candidates use link6 local Z lines with zero local X/Y tool offset, one common six-DOF camera-to-base correction per recording, and fixed relative camera transforms. ' 
        'Each measured image line constrains two projected points along an infinite 3D line; it does not locate the actual TCP along that line. ' 
        'A constant EE orientation can leave translation along the shaft unidentifiable. Rank/condition and bounded optimization checks are saved.', '',
        'Held-out episodes or contiguous time blocks are selected before fitting; thresholds are fixed. Training is balanced across scene/camera pairs. ' 
        'Bootstrap clusters are episode/time blocks, not individual pixels. Per-scene overlays and metrics also include training observations and must not be presented as independent accuracy.', '',
        'Background RGB feature stability is tested separately at three frames in the first episode of each native recording, with the split before homography filtering (background_stability.json). It does not establish physical feature identity or a base reference. '
        'No independent flange/tip detector, room-feature identity verification, sensor synchronization correction, depth-bias calibration or physical mounting verification is performed here. ' 
        'These candidates cannot disambiguate every tool-reference, timing and camera-error hypothesis. Production acceptance requires independent checks.', '',
        'A common rigid correction improves only robot/scene registration; it preserves relative-camera rendering geometry and does not change base-frame force/trajectory percentages.', '',
        'Files: candidate_corrections.json (matrices, heldout, uncertainty, checks), pooled_common_mount_hypothesis.json (explicit common-mount hypothesis and leave-recording-out validation), coverage.json, scene_camera_metrics.csv, cross_recording_transfer.json, per-scene observations.jsonl, candidate_overlay.jpg and pooled_candidate_overlay.jpg.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    table = ''.join(f'<tr><td>{html.escape(s)}</td><td>{html.escape(infos[s]["recording"])}</td><td>' +
        ' '.join(f'<a href="{html.escape(s)}/{file}">{label}</a>' for file, label in [('candidate_overlay.jpg', 'Recording fit'), ('pooled_candidate_overlay.jpg', 'Pooled hypothesis')] if (output / s / file).exists()) +
        '</td></tr>' for s in infos)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Registration audit</title>'
        '<h1>Candidate registration overlays</h1><p>Cyan: nominal; yellow: fitted; green: detected shaft. '
        'Heuristic observations; these are not certified calibration corrections. See REPORT.md.</p><table>' + table + '</table>')
    print(f'Completed {len(infos)} scenes; {accepted}/{attempted} accepted observations. Report: {output / "REPORT.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--converted-root', type=Path, required=True)
    parser.add_argument('--measurements-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--bootstrap', type=int, default=24)
    parser.add_argument('--phase', choices=['extract', 'fit', 'all'], default='all')
    args = parser.parse_args()
    if args.workers < 1 or args.frame_stride < 1 or args.bootstrap < 0:
        parser.error('Workers/stride must be positive and bootstrap nonnegative')
    source, measurements, output = [p.resolve() for p in (args.converted_root, args.measurements_root, args.output_root)]
    if output == source or source in output.parents or output == measurements or measurements in output.parents or output in source.parents or output in measurements.parents:
        parser.error('Audit output must be separate from dataset and measurement trees')
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    scenes = sorted(p.parent.parent for p in source.glob('*/metadata/cameras.json'))
    if args.scenes:
        scenes = [p for p in scenes if p.name in args.scenes]
        missing = set(args.scenes) - {p.name for p in scenes}
        if missing:
            parser.error(f'Unknown scenes: {sorted(missing)}')
    if not scenes:
        parser.error('No converted real scenes found')
    write_json(output / 'run.json', dict(version=VERSION, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        converted_root=str(source), measurements_root=str(measurements), output_root=str(output),
        requested_scenes=[s.name for s in scenes], frame_stride=args.frame_stride, workers=args.workers,
        bootstrap=args.bootstrap, phase=args.phase, production_modified=False))
    failures = []
    if args.phase != 'fit':
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(extract_scene, (str(s), str(measurements), str(output), args.frame_stride)): s.name for s in scenes}
            for future in as_completed(futures):
                try:
                    print(json.dumps(future.result()), flush=True)
                except Exception as exc:
                    failures.append(dict(scene=futures[future], error=str(exc)))
                    print(f'FAILED {futures[future]}: {exc}', flush=True)
        write_json(output / 'failures.json', failures)
    if args.phase != 'extract':
        fit_all(output, args.bootstrap)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
