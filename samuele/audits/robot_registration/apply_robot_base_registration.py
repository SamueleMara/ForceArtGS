#!/usr/bin/env python3
"""Back up and register converted final-session geometry to the recorded robot base."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation
import yaml

STATUS = 'OPERATOR_APPLIED_ROBOT_REGISTRATION_UNVERIFIED'
MARKER = Path('metadata/robot_base_registration.json')
BUSY = Path('metadata/robot_base_registration_in_progress.json')
ACTIVE_NATIVE = Path('metadata/camera_calibration_robot_base.json')
PLY_TYPES = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
             'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
             'short': '<i2', 'ushort': '<u2', 'int': '<i4', 'uint': '<u4'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def rigid(G):
    G = np.asarray(G, float)
    if G.shape != (4, 4) or not np.isfinite(G).all() or not np.allclose(G[3], [0, 0, 0, 1]) or not np.allclose(G[:3, :3].T @ G[:3, :3], np.eye(3), atol=1e-10) or not np.isclose(np.linalg.det(G[:3, :3]), 1):
        raise ValueError('Expected a finite proper rigid transform')
    return G


def ply_header(stream):
    header, props, count, fmt, element = [], [], None, None, None
    for _ in range(10000):
        line = stream.readline()
        if not line:
            raise ValueError('Truncated PLY header')
        header.append(line); fields = line.decode('ascii').strip().split()
        if fields[:1] == ['format']:
            fmt = fields[1]
        elif fields[:1] == ['element']:
            element = fields[1]
            if element == 'vertex':
                count = int(fields[2])
            elif int(fields[2]):
                raise ValueError('PLY with non-vertex elements requires an explicit transform implementation')
        elif fields[:1] == ['property'] and element == 'vertex':
            if fields[1] not in PLY_TYPES:
                raise ValueError(f'Unsupported PLY property: {fields}')
            props.append((fields[2], PLY_TYPES[fields[1]]))
        if fields == ['end_header']:
            break
    else:
        raise ValueError('Oversized PLY header')
    names = [p[0] for p in props]
    if fmt not in ['ascii', 'binary_little_endian'] or count is None or not {'x', 'y', 'z'} <= set(names):
        raise ValueError('Unsupported PLY geometry')
    if any(n.startswith(('rot_', 'f_rest_', 'scale_')) for n in names):
        raise ValueError('Gaussian attributes require rotation-aware SH/covariance handling')
    return b''.join(header), fmt, count, np.dtype(props)


def transform_ply(source, target, G):
    with source.open('rb') as f:
        header, fmt, count, dtype = ply_header(f)
        data = np.atleast_1d(np.loadtxt(f, dtype=dtype, max_rows=count)) if fmt == 'ascii' and count else np.fromfile(f, dtype=dtype, count=count)
        tail = f.read()
    if len(data) != count:
        raise ValueError(f'Truncated PLY vertices: {source}')
    xyz = np.column_stack([data[n] for n in ['x', 'y', 'z']]).astype(float)
    if not np.isfinite(xyz).all():
        raise ValueError(f'Nonfinite PLY geometry: {source}')
    new = xyz @ G[:3, :3].T + G[:3, 3]
    for k, n in enumerate(['x', 'y', 'z']):
        data[n] = new[:, k]
    if {'nx', 'ny', 'nz'} <= set(dtype.names):
        normals = np.column_stack([data[n] for n in ['nx', 'ny', 'nz']]).astype(float) @ G[:3, :3].T
        for k, n in enumerate(['nx', 'ny', 'nz']):
            data[n] = normals[:, k]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as f:
        f.write(header)
        if fmt == 'ascii':
            np.savetxt(f, np.column_stack([data[n] for n in dtype.names]),
                       fmt=['%d' if dtype[n].kind in 'iu' else ('%.9g' if dtype[n].itemsize == 4 else '%.17g') for n in dtype.names])
        else:
            data.tofile(f)
        f.write(tail)
    return dict(vertices=count, format=fmt, max_stored_xyz_rounding_m=float(np.max(abs(np.column_stack([data[n] for n in ['x', 'y', 'z']]) - new))) if count else 0.)


def change_pose(qvec, tvec, G):
    W = np.eye(4)
    W[:3, :3] = Rotation.from_quat(np.asarray(qvec)[[1, 2, 3, 0]]).as_matrix()
    W[:3, 3] = tvec
    W = W @ np.linalg.inv(G)
    return Rotation.from_matrix(W[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist(), W[:3, 3].tolist()


def geometry_files(scene):
    """Restrict discovery to geometry; never walk RGB/depth/mask sequences."""
    roots = sorted(p for p in scene.glob('colmap*') if p.is_dir())
    models, files = [], set()
    for root in roots:
        folders = [p for sparse in root.glob('sparse*') if sparse.is_dir() for p in sparse.iterdir() if p.is_dir()]
        for folder in sorted(folders):
            formats = [ext for ext in ['bin', 'txt'] if all((folder / f'{n}.{ext}').is_file() for n in ['cameras', 'images', 'points3D'])]
            if not formats:
                if any(folder.iterdir()):
                    raise ValueError(f'Incomplete COLMAP model: {folder}')
                continue
            models.append((folder.relative_to(scene), formats))
            files.update(p.relative_to(scene) for p in folder.iterdir() if p.is_file() and p.suffix in ['.bin', '.txt', '.ply'])
        files.update(p.relative_to(scene) for p in root.glob('**/*.ply'))
        files.update(p.relative_to(scene) for p in [root / 'cameras.yaml', root / 'database.db', root / 'model_summary.json', root / 'export_summary.json'] if p.is_file())
    files.update(p.relative_to(scene) for p in (scene / 'pointcloud').glob('**/*.ply'))
    files.update(Path('metadata') / n for n in ['cameras.json', 'real_conversion.json', 'colmap_export.json', 'colmap_export_final.json'] if (scene / 'metadata' / n).is_file())
    for rel in files:
        if (scene / rel).is_symlink() or not (scene / rel).resolve().is_relative_to(scene):
            raise ValueError(f'Geometry symlink or path escape: {rel}')
        if rel.suffix == '.ply':
            with (scene / rel).open('rb') as f:
                ply_header(f)
        if rel.name == 'database.db' and any(Path(str(scene / rel) + suffix).exists() for suffix in ['-wal', '-journal']):
            raise ValueError(f'COLMAP database has an active journal: {scene / rel}')
    return models, sorted(files)


def transform_model(before, staged, relative, formats, G, cameras, endpoint):
    rec = pycolmap.Reconstruction(str(before / relative))
    names = {i.name for i in rec.images.values()}
    expected = {f'{name}/frame_{endpoint:06d}.png' for name in cameras}
    if names != expected or rec.num_images() != len(cameras):
        raise ValueError(f'{relative}: endpoint membership differs from {expected}: {names}')
    old_poses = {i.image_id: i.cam_from_world().matrix().copy() for i in rec.images.values()}
    ids = list(rec.points3D.keys())
    old_xyz = np.array([rec.points3D[pid].xyz for pid in ids]).reshape(-1, 3)
    for image in rec.images.values():
        if relative.parts[0] in ['colmap', 'colmap_final'] or relative.name == 'known_poses':
            C = np.linalg.inv(np.vstack([old_poses[image.image_id], [0, 0, 0, 1]]))
            if not np.allclose(C, cameras[image.name.split('/')[0]], atol=1e-8, rtol=0):
                raise ValueError(f'{relative}: model and camera metadata were already inconsistent')
    rec.transform(pycolmap.Sim3d(1., pycolmap.Rotation3d(G[:3, :3]), G[:3, 3]))
    destination = staged / relative
    # Native COLMAP handles rig/frame poses as well as legacy images/points.
    with tempfile.TemporaryDirectory(dir=staged) as temp:
        temp = Path(temp)
        for ext in formats:
            (rec.write_binary if ext == 'bin' else rec.write_text)(str(temp))
            for name in ['images', 'points3D', 'frames']:
                if (before / relative / f'{name}.{ext}').is_file():
                    shutil.copyfile(temp / f'{name}.{ext}', destination / f'{name}.{ext}')
    check = pycolmap.Reconstruction(str(destination))
    if set(check.points3D.keys()) != set(ids) or {i.name for i in check.images.values()} != names:
        raise AssertionError('COLMAP identities changed')
    new_xyz = np.array([check.points3D[pid].xyz for pid in ids]).reshape(-1, 3)
    error = float(np.max(abs(new_xyz - (old_xyz @ G[:3, :3].T + G[:3, 3])))) if len(ids) else 0.
    if error > 1e-9:
        raise AssertionError('COLMAP point transform failed')
    for iid, W in old_poses.items():
        expected_W = np.vstack([W, [0, 0, 0, 1]]) @ np.linalg.inv(G)
        image = check.images[iid]
        if not np.allclose(image.cam_from_world().matrix(), expected_W[:3], atol=1e-10, rtol=0):
            raise AssertionError('COLMAP pose transform failed')
        a, b = rec.images[iid].points2D, image.points2D
        if len(a) != len(b) or any(x.point3D_id != y.point3D_id or not np.array_equal(x.xy, y.xy) for x, y in zip(a, b)):
            raise AssertionError('COLMAP image observations changed')
    # Check the text representation independently when both formats exist.
    if len(formats) == 2:
        text = pycolmap.Reconstruction(); text.read_text(str(destination))
        if text.num_points3D() != len(ids) or any(not np.allclose(text.images[i].cam_from_world().matrix(), check.images[i].cam_from_world().matrix(), atol=1e-10, rtol=0) for i in old_poses):
            raise AssertionError('COLMAP text/binary disagreement')
    return dict(model=str(relative), images=len(names), points=len(ids), endpoint=endpoint, max_point_error_m=error)


def transform_database(path, G):
    if any(Path(str(path) + s).exists() for s in ['-wal', '-journal']):
        raise ValueError('Database has an active journal')
    with sqlite3.connect(path) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Invalid COLMAP database')
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'pose_priors' in tables and db.execute('SELECT COUNT(*) FROM pose_priors').fetchone()[0]:
            raise ValueError('Nonempty position/covariance priors require explicit coordinate-system handling')
        columns = {r[1] for r in db.execute('PRAGMA table_info(images)')}
        if 'prior_qw' not in columns:
            return 0
        rows = db.execute('SELECT image_id,prior_qw,prior_qx,prior_qy,prior_qz,prior_tx,prior_ty,prior_tz FROM images').fetchall()
        count = 0
        for iid, *pose in rows:
            if all(v is None for v in pose):
                continue
            if any(v is None for v in pose) or not np.isfinite(pose).all():
                raise ValueError('Incomplete/nonfinite database pose prior')
            q, t = change_pose(pose[:4], pose[4:], G)
            db.execute('UPDATE images SET prior_qw=?,prior_qx=?,prior_qy=?,prior_qz=?,prior_tx=?,prior_ty=?,prior_tz=? WHERE image_id=?', (*q, *t, iid)); count += 1
        return count


def verify_applied(scene, G):
    marker = json.loads((scene / MARKER).read_text())
    if not np.array_equal(marker['G_left_multiply_camera_to_base'], G):
        raise ValueError('A different registration is already applied; restore it first')
    for rel, digest in marker['after_sha256'].items():
        if sha(scene / rel) != digest:
            raise ValueError(f'Applied file changed since registration: {scene / rel}')
    return marker


def publish(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.robot-registration-', dir=target.parent, delete=False) as f:
        temporary = Path(f.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def restore_scene(scene, job, expected=None):
    state = json.loads((job / 'state.json').read_text())
    if expected is not None:
        for rel, digest in expected.items():
            if sha(scene / rel) != digest:
                raise ValueError(f'Refusing to overwrite edits after registration: {scene / rel}')
    for rel, digest in state['before_sha256'].items():
        if sha(job / 'before' / rel) != digest:
            raise ValueError('Backup hash mismatch')
    # An interrupted publication may mix before/after files; preserve unrelated edits.
    for rel in set(state['before_sha256']) | set(state.get('after_sha256', {})):
        path = scene / rel
        allowed = {m[rel] for m in [state['before_sha256'], state.get('after_sha256', {})] if rel in m}
        if (path.exists() and sha(path) not in allowed) or (not path.exists() and rel in state['before_sha256']):
            raise ValueError(f'Restore would overwrite unrelated edits: {path}')
    for rel in state['before_sha256']:
        publish(job / 'before' / rel, scene / rel)
    for rel in [MARKER, ACTIVE_NATIVE, BUSY]:
        (scene / rel).unlink(missing_ok=True)
    for rel, digest in state['before_sha256'].items():
        if sha(scene / rel) != digest:
            raise AssertionError('Restore verification failed')
    state['status'] = 'restored'; write_json(job / 'state.json', state)


def apply_scene(job):
    scene_path, report_path, info, G, provenance = job
    scene, jobdir = Path(scene_path), Path(report_path) / 'backups' / info['scene']
    G = rigid(G)
    if (scene / BUSY).exists():
        raise ValueError(f'Interrupted or active registration: {scene / BUSY}; use restore before continuing')
    if (scene / MARKER).exists():
        verify_applied(scene, G)
        return dict(scene=scene.name, status='already_applied_verified')
    if (scene / ACTIVE_NATIVE).exists():
        raise ValueError('Untracked active native calibration already exists; refusing to replace it')
    if jobdir.exists():
        raise ValueError(f'Backup already exists: {jobdir}; inspect it or choose a fresh report root')
    models, files = geometry_files(scene)
    current = json.loads((scene / 'metadata/cameras.json').read_text())
    original = {c['name']: np.array(c['pose_matrix_4x4']) for c in info['cameras']}
    if {c['name'] for c in current['cameras']} != set(original) or any(not np.allclose(c['pose_matrix_4x4'], original[c['name']], atol=1e-10, rtol=0) for c in current['cameras']):
        raise ValueError('Current cameras differ from the recorded pre-registration baseline')
    before, staged = jobdir / 'before', jobdir / 'staged'
    before.mkdir(parents=True); staged.mkdir()
    hashes = {}
    for rel in files:
        target = before / rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scene / rel, target); hashes[str(rel)] = sha(target)
        if sha(scene / rel) != hashes[str(rel)]:
            raise ValueError('Source changed during backup')
        stage = staged / rel; stage.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(target, stage)
    untouched = [scene / 'metadata/robot_tool.json', scene / 'metadata/frame_timestamps.csv', scene / 'metadata/raw/camera_calibration.json'] + list((scene / 'metadata/raw').glob('*.parquet'))
    unchanged_hashes = {str(p.relative_to(scene)): sha(p) for p in untouched if p.is_file()}
    state = dict(status='staging', before_sha256=hashes, scene=str(scene), provenance=provenance)
    write_json(jobdir / 'state.json', state)
    conversion = json.loads((before / 'metadata/real_conversion.json').read_text())
    final = int(conversion['output_frames'][-1])
    record = dict(scene=scene.name, status='applied', model_checks=[], ply_checks=[], database_priors={})
    for relative, formats in models:
        endpoint = final if 'final' in relative.parts[0] else 0
        record['model_checks'].append(transform_model(before, staged, relative, formats, G, original, endpoint))
    for rel in files:
        if rel.suffix == '.ply':
            record['ply_checks'].append(dict(path=str(rel), **transform_ply(before / rel, staged / rel, G)))
        elif rel.name == 'database.db':
            record['database_priors'][str(rel)] = transform_database(staged / rel, G)
        elif rel.name == 'cameras.yaml':
            content = yaml.safe_load((before / rel).read_text())
            for values in content.get('extrinsics', {}).values():
                values['qvec'], values['tvec'] = change_pose(values['qvec'], values['tvec'], G)
            (staged / rel).write_text(yaml.safe_dump(content, sort_keys=False))
    registration = dict(status=STATUS, G_left_multiply_camera_to_base=G.tolist(), backup=str(before),
        provenance=provenance, coordinate_frame='base_link', robot_link6_and_tcp_unchanged=True,
        raw_calibration_preserved=True, active_native_calibration=str(ACTIVE_NATIVE), candidate_independently_validated=False)
    for cam in current['cameras']:
        cam['pose_matrix_4x4'] = (G @ original[cam['name']]).tolist()
    current['source_calibration_status'] = current.get('calibration_status')
    current.update(calibration_status=STATUS, robot_base_registration=registration)
    write_json(staged / 'metadata/cameras.json', current)
    for rel in files:
        if rel.suffix == '.json' and rel != Path('metadata/cameras.json'):
            content = json.loads((staged / rel).read_text())
            content['robot_base_registration'] = registration
            if 'calibration_status' in content:
                content['source_calibration_status'] = content['calibration_status']; content['calibration_status'] = STATUS
            write_json(staged / rel, content)
    native = json.loads((scene / 'metadata/raw/camera_calibration.json').read_text())
    for cam in native['cameras'].values():
        cam['T_base_color_optical'] = (G @ np.array(cam['T_base_color_optical'])).tolist()
    native['source_calibration_status'] = native.get('status'); native['status'] = STATUS
    native['robot_base_registration'] = registration
    write_json(staged / ACTIVE_NATIVE, native)
    active_files = files + [ACTIVE_NATIVE]
    after = {str(rel): sha(staged / rel) for rel in active_files}
    for rel, digest in hashes.items():
        if sha(scene / rel) != digest:
            raise ValueError('Source geometry changed before publishing')
    for rel, digest in unchanged_hashes.items():
        if sha(scene / rel) != digest:
            raise ValueError('Robot/raw metadata changed during staging')
    state.update(status='ready', after_sha256=after, checks=record, untouched_sha256=unchanged_hashes)
    write_json(jobdir / 'state.json', state)
    owned = False
    try:
        with (scene / BUSY).open('x') as f:
            json.dump(dict(backup=str(jobdir), report=str(Path(report_path))), f)
        owned = True
        for rel in active_files:
            publish(staged / rel, scene / rel)
        for rel, digest in after.items():
            if sha(scene / rel) != digest:
                raise AssertionError('Published file hash mismatch')
        for rel, digest in unchanged_hashes.items():
            if sha(scene / rel) != digest:
                raise AssertionError('Robot/raw metadata was changed')
        write_json(scene / MARKER, dict(**registration, after_sha256=after, untouched_sha256=unchanged_hashes))
        (scene / BUSY).unlink()
        state['status'] = 'complete'; write_json(jobdir / 'state.json', state)
    except BaseException:
        if owned:
            restore_scene(scene, jobdir)
        raise
    shutil.rmtree(staged)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--converted-root', type=Path, required=True)
    parser.add_argument('--group-root', type=Path, required=True)
    parser.add_argument('--report-root', type=Path, required=True)
    parser.add_argument('--phase', choices=['plan', 'apply', 'restore'], default='plan')
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    scene_root, group_root, report = [p.resolve() for p in [args.converted_root, args.group_root, args.report_root]]
    if args.workers < 1 or report == scene_root or scene_root in report.parents or report in scene_root.parents:
        parser.error('Use positive workers and a report root separate from the datasets')
    group = json.loads((group_root / 'group.json').read_text()); candidate = json.loads((group_root / 'candidate.json').read_text())
    G = rigid(candidate['G_left_multiply_camera_to_base'])
    names = args.scenes or group['scenes']
    if set(names) - set(group['scenes']) or any(Path(n).name != n for n in names):
        parser.error('Only explicitly listed acquisition-group scenes are supported')
    infos = {n: json.loads((Path(group['source_audit']) / n / 'inventory.json').read_text()) for n in names}
    report.mkdir(parents=True, exist_ok=True)
    with (report / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = dict(group_root=str(group_root), group_id=group['group_id'], grouping_evidence=group['evidence'],
            candidate_status=candidate['status'], candidate_sha256=sha(group_root / 'candidate.json'),
            script_sha256=sha(Path(__file__)), pycolmap_version=pycolmap.__version__, operator_requested_application=True)
        write_json(report / f'{args.phase}_invocation.json', dict(**provenance, scenes=names, G=G.tolist()))
        if args.phase == 'plan':
            plans = []
            for name in names:
                scene = scene_root / name
                if (scene / MARKER).is_file():
                    verify_applied(scene, G); plans.append(dict(scene=name, status='already_applied_verified')); continue
                cameras = json.loads((scene / 'metadata/cameras.json').read_text())['cameras']
                if any(not np.allclose(c['pose_matrix_4x4'], b['pose_matrix_4x4'], atol=1e-10, rtol=0) for c, b in zip(cameras, infos[name]['cameras'])):
                    raise ValueError(f'Pre-registration cameras differ: {name}')
                models, files = geometry_files(scene)
                plans.append(dict(scene=name, files=[str(p) for p in files], model_count=len(models), bytes=sum((scene / p).stat().st_size for p in files)))
            write_json(report / 'plan.json', plans)
            print(f'Planned {len(plans)} scenes; {sum(p.get("bytes", 0) for p in plans) / 1e9:.2f} GB of originals to back up.', flush=True)
            return
        results, failures = [], []
        if args.phase == 'restore':
            for name in names:
                scene = scene_root / name; job = report / 'backups' / name
                marker = json.loads((scene / MARKER).read_text()) if (scene / MARKER).exists() else None
                restore_scene(scene, job, marker['after_sha256'] if marker else None)
                results.append(dict(scene=name, status='restored'))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(apply_scene, (str(scene_root / n), str(report), infos[n], G, provenance)): n for n in names}
                for future in as_completed(futures):
                    try:
                        rec = future.result(); results.append(rec); print(f'{len(results)}/{len(names)} {rec["scene"]}: {rec["status"]}', flush=True)
                    except Exception as exc:
                        failures.append(dict(scene=futures[future], error=str(exc))); print(f'FAILED {futures[future]}: {exc}', flush=True)
                    write_json(report / 'progress.json', dict(results=results, failures=failures))
        write_json(report / f'{args.phase}_results.json', dict(results=results, failures=failures))
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
