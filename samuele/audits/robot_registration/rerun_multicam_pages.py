#!/usr/bin/env python3
"""Repeat an archived masked RGB-D viewer export with a controlled camera-pose comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_EXPORTER = Path('/home/samuelemara/Joint3DGS/RGBDekf')


def exporter(repo):
    sys.path.insert(0, str(repo))
    from tools import write_multicam_rgbd_pages
    return write_multicam_rgbd_pages


def read_page(path):
    before, rest = path.read_text().split('const M=', 1)
    encoded, after = rest.split(',$=id', 1)
    return json.loads(encoded), (before, after)


def read_frame(path):
    return json.loads(path.read_text().split('=', 2)[2].rstrip(';\n'))


def transformed_cameras(cameras, G):
    if G.shape != (4, 4) or not np.isfinite(G).all() or not np.allclose(G[3], [0, 0, 0, 1]):
        raise ValueError('Invalid homogeneous correction')
    if not np.allclose(G[:3, :3].T @ G[:3, :3], np.eye(3), atol=1e-8) or np.linalg.det(G[:3, :3]) < .999999:
        raise ValueError('Correction must be a proper rigid transform')
    inverse = np.linalg.inv(G)
    return [replace(c, world_to_camera=c.world_to_camera @ inverse) for c in cameras]


def relative_pose_delta(before, after):
    return max(float(np.max(abs(np.asarray(after[j].world_to_camera) @ np.linalg.inv(after[i].world_to_camera) -
                                np.asarray(before[j].world_to_camera) @ np.linalg.inv(before[i].world_to_camera))))
               for i in range(len(before)) for j in range(len(before)) if i != j)


def nearest_metrics(X, Y, camera_ids, limit=2048):
    """Compare the same source indices, using every target point; no correspondence claim."""
    records = []
    for i in np.unique(camera_ids):
        source = np.flatnonzero(camera_ids == i)
        selected = source[np.unique(np.linspace(0, len(source) - 1, min(limit, len(source))).astype(int))]
        for j in np.unique(camera_ids):
            if i == j:
                continue
            target = camera_ids == j
            a = cKDTree(X[target].astype(float)).query(X[selected].astype(float))[0]
            b = cKDTree(Y[target].astype(float)).query(Y[selected].astype(float))[0]
            records.append(dict(source_camera=int(i), target_camera=int(j), source_samples=len(selected),
                target_points=int(target.sum()), baseline_median_m=float(np.median(a)), corrected_median_m=float(np.median(b)),
                baseline_p95_m=float(np.percentile(a, 95)), corrected_p95_m=float(np.percentile(b, 95)),
                max_distance_change_m=float(np.max(abs(a - b)))))
    return records


MESSAGE_SCRIPT = '''<script>
window.__comparisonSnapshot=()=>({frame:M.frames[idx],ready:!!current&&current===window.MC[M.frames[idx]],points:current?current.n:0,
 errors:document.getElementById('errors').textContent,cameras:M.cameras});
addEventListener('message',e=>{if(e.source!==parent||!e.data)return;
 if(e.data.type==='comparison-frame'){const n=Number(e.data.index);if(!Number.isInteger(n)||n<0||n>=M.frames.length)return;
  const s=document.getElementById('frame');s.value=n;s.oninput();}
 if(e.data.type==='comparison-request-view'){const st=view.state;parent.postMessage({type:'comparison-view',
  state:{target:st.target.toArray(),th:st.th,ph:st.ph,r:st.r,radius:st.radius}},'*');}
 if(e.data.type==='comparison-set-view'){const v=e.data.state;if(!v||!Array.isArray(v.target)||![...v.target,v.th,v.ph,v.r,v.radius].every(Number.isFinite))return;
  const st=view.state;st.target.fromArray(v.target);for(const k of ['th','ph','r','radius'])st[k]=v[k];
  view.camera.position.set(st.target.x+st.r*Math.sin(st.ph)*Math.cos(st.th),st.target.y+st.r*Math.sin(st.ph)*Math.sin(st.th),st.target.z+st.r*Math.cos(st.ph));
  view.camera.lookAt(st.target);view.camera.updateMatrixWorld();}
});
</script>'''


def write_page(dest, name, meta, template, variant):
    before, after = template
    page = before + 'const M=' + json.dumps(meta) + ',$=id' + after
    label = f'<p style="color:#f6c970">{html.escape(variant)}. Candidate comparison; see the comparison report.</p>'
    page = page.replace('<h3 id="title"></h3>', '<h3 id="title"></h3>' + label)
    page = page.replace('<a href="index.html">all datasets</a>', '<a href="../index.html">all datasets</a>')
    page = page.replace('</body>', MESSAGE_SCRIPT + '</body>')
    (dest / f'{name}.html').write_text(page)


def rerun_scene(job):
    baseline_root, output, repo, name, G, frame_limit = job
    baseline_root, output = Path(baseline_root), Path(output)
    writer = exporter(Path(repo))
    from scene.dataset import Dataset
    meta, template = read_page(baseline_root / f'{name}.html')
    if meta.get('knn'):
        raise ValueError('This controlled rerun expects the original no-kNN exporter settings')
    data = Dataset(Path(meta['source']))
    old_frames = meta['frames']
    frames = old_frames[:frame_limit] if frame_limit else old_frames
    for k, fid in zip(old_frames, meta['original_frame_ids']):
        if k >= len(data.frame_ids) or int(data.frame_ids[k]) != fid:
            raise ValueError('Current frame IDs differ from the archived export')
    if len(meta['cameras']) != len(data.cameras):
        raise ValueError('Camera count differs from the archived export')
    for old, camera in zip(meta['cameras'], data.cameras):
        if old['name'] != camera.name or not np.allclose(old['T_wc'], np.linalg.inv(camera.world_to_camera), atol=1e-10, rtol=0) or not np.array_equal(old['K'], camera.intrinsics):
            raise ValueError('Source calibration differs from the archived original; cannot isolate this correction')
    corrected = SimpleNamespace(cameras=transformed_cameras(data.cameras, G), observation=data.observation)
    camera_delta = relative_pose_delta(data.cameras, corrected.cameras)
    if camera_delta > 1e-10:
        raise AssertionError('Common rigid correction changed relative camera poses')
    for variant in ['baseline', 'corrected']:
        (output / variant / f'{name}_data').mkdir(parents=True, exist_ok=True)
    receipts, metrics, unchanged_archive, max_error = [], [], 0, 0.
    selected_metrics = {frames[0], frames[len(frames) // 2], frames[-1]}
    for k in frames:
        a = writer.frame_cloud(data, k, meta['pixel_grid_step'])
        b = writer.frame_cloud(corrected, k, meta['pixel_grid_step'])
        expected = a['_X'].astype(float) @ G[:3, :3].T + G[:3, 3]
        error = float(np.max(abs(b['_X'] - expected))) if a['n'] else 0.
        if error > 1e-6 or not np.array_equal(a['_pid'], b['_pid']) or a['rgb'] != b['rgb']:
            raise AssertionError('Updated cameras did not produce the expected rigidly transformed identical observations')
        max_error = max(max_error, error)
        archived_path = baseline_root / f'{name}_data/f_{k:06d}.js'
        archived = read_frame(archived_path)
        same = all(a[field] == archived[field] for field in ['n', 'xyz', 'rgb', 'cam'])
        unchanged_archive += int(same)
        receipt = dict(frame=k, points=a['n'], archived_points=archived['n'], archived_export_exactly_matches_current=same,
            corrected_vs_G_baseline_max_component_error_m=error,
            source_fingerprints=[data.base.observation_fingerprints[f'{k:06d}_{c:03d}'] for c in range(len(data.cameras))])
        receipts.append(receipt)
        if k in selected_metrics:
            for metric in nearest_metrics(a['_X'], b['_X'], a['_cam']):
                metric['frame'] = k; metrics.append(metric)
        for variant, cloud in [('baseline', a), ('corrected', b)]:
            payload = {key: value for key, value in cloud.items() if not key.startswith('_')}
            (output / variant / f'{name}_data/f_{k:06d}.js').write_text(f'window.MC=window.MC||{{}};window.MC[{k}]={json.dumps(payload)};')
    for variant, dataset in [('baseline', data), ('corrected', corrected)]:
        updated = dict(meta, cameras=writer.cameras_spec(dataset), frames=frames,
            original_frame_ids=meta['original_frame_ids'][:len(frames)], times_s=meta['times_s'][:len(frames)],
            masked_valid_points_per_frame=[r['points'] for r in receipts], status='COMPLETE', error=None,
            comparison_variant=variant, comparison_source=str(baseline_root), camera_correction=G.tolist() if variant == 'corrected' else np.eye(4).tolist())
        write_page(output / variant, name, updated, template, variant)
    record = dict(dataset=name, frames=len(frames), points=sum(r['points'] for r in receipts),
        archived_matching_frames=unchanged_archive, archived_changed_frames=len(frames) - unchanged_archive,
        max_rigid_transform_error_m=max_error, relative_camera_matrix_max_delta=camera_delta,
        nearest_metrics=metrics, metric_sampling='first/middle/last frame; at most 2048 evenly spaced source points per camera; every target point',
        metric_interpretation='nearest measured surface proximity, not certified physical correspondences or absolute sensor accuracy',
        input_settings=dict(pixel_grid_step=meta['pixel_grid_step'], frame_stride=meta['frame_stride'], max_points_per_frame=meta['max_points_per_frame']))
    (output / 'receipts' / f'{name}.json').write_text(json.dumps(dict(summary=record, frames=receipts), indent=1) + '\n')
    return record


def write_index(output, records, controls, archived_root):
    links = ''.join(f'<tr><td>{html.escape(r["dataset"])}</td><td>{r["frames"]}</td><td>{r["archived_changed_frames"]}</td><td>'
        f'<a href="baseline/{r["dataset"]}.html">current inputs / original poses</a> · '
        f'<a href="corrected/{r["dataset"]}.html">current inputs / corrected poses</a> · '
        f'<a href="{html.escape((archived_root / r["dataset"]).as_uri())}.html">November 3 export</a></td></tr>' for r in records)
    options = ''.join(f'<option value="{html.escape(r["dataset"])}" data-frames="{r["frames"]}">{html.escape(r["dataset"])}</option>' for r in records)
    control_links = ''.join(f'<li><a href="{html.escape((archived_root / r["html"]).as_uri())}">{html.escape(r["dataset"])}</a> — no applicable final-session correction</li>' for r in controls)
    (output / 'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Multicam RGB-D pose comparison</title>
<style>body{background:#15191e;color:#ddd;font:15px sans-serif;margin:18px}a{color:#89caff}td{padding:5px}iframe{width:49%;height:720px;border:1px solid #59616a}select{max-width:70%}</style>
<h1>Multicam RGB-D: original versus candidate camera poses</h1>
<p>Both sides use the same current RGB-D and masks. The archived November 3 export is a separate reference because some masks have since changed.
The candidate is unverified. A common rigid transform changes robot/world registration while preserving inter-camera geometry.</p>
<p><a href="REPORT.md">Report</a> · <a href="comparison.json">Numerical comparison</a></p>
<select id="scene">''' + options + '''</select> Frame index <input id="frame" type="number" min="0" value="0"><button id="set">Set both frames</button><button id="syncview">Match right view to left</button>
<p>Left: original poses. Right: candidate poses. Every frame and every masked valid point are exported.</p>
<iframe id="left"></iframe><iframe id="right"></iframe>
<table><tr><th>Dataset</th><th>Frames</th><th>Changed archived frames</th><th>Open full page</th></tr>''' + links + '''</table>
<details><summary>Unchanged controls from the original collection</summary><ul>''' + control_links + '''</ul></details>
<script>const s=document.getElementById('scene'),f=document.getElementById('frame'),l=document.getElementById('left'),r=document.getElementById('right');
function select(){if(!s.value)return;l.src='baseline/'+s.value+'.html';r.src='corrected/'+s.value+'.html';f.max=Number(s.selectedOptions[0].dataset.frames)-1;f.value=0;}
s.onchange=select;document.getElementById('set').onclick=()=>{for(const w of [l,r])w.contentWindow.postMessage({type:'comparison-frame',index:Number(f.value)},'*')};
document.getElementById('syncview').onclick=()=>l.contentWindow.postMessage({type:'comparison-request-view'},'*');
addEventListener('message',e=>{if(e.source===l.contentWindow&&e.data&&e.data.type==='comparison-view')r.contentWindow.postMessage({type:'comparison-set-view',state:e.data.state},'*')});select();</script>''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--group-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--exporter-repo', type=Path, default=DEFAULT_EXPORTER)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--test-frame-limit', type=int, default=0, help='Pilot only; zero processes every archived frame')
    args = parser.parse_args()
    if args.workers < 1 or args.test_frame_limit < 0:
        parser.error('Positive workers and nonnegative test frame limit required')
    original, group_root, output = args.original_root.resolve(), args.group_root.resolve(), args.output_root.resolve()
    group = json.loads((group_root / 'group.json').read_text())
    candidate = json.loads((group_root / 'candidate.json').read_text())
    G = np.array(candidate['G_left_multiply_camera_to_base'], float)
    records = json.loads((original / 'summary.json').read_text())
    selected = [r for r in records if r['dataset'] in group['scenes'] and (not args.scenes or r['dataset'] in args.scenes)]
    if not selected or (args.scenes and set(args.scenes) - {r['dataset'] for r in selected}):
        parser.error('Missing requested scenes or no applicable acquisition-group scenes')
    for root in [original, group_root] + [Path(r['source']).resolve() for r in selected]:
        if output == root or root in output.parents or output in root.parents:
            parser.error('Output must be separate from all inputs')
    if output.exists() and any(output.iterdir()):
        parser.error('Use an empty output folder; prior exports are preserved')
    required = 2 * sum(r.get('bytes', 0) for r in selected) if not args.test_frame_limit else 0
    if required and shutil.disk_usage(output.parent).free < required * 1.1:
        parser.error('Insufficient free disk space for two full-resolution exports')
    output.mkdir(parents=True, exist_ok=True); (output / 'receipts').mkdir()
    for variant in ['baseline', 'corrected']:
        (output / variant).mkdir()
        for asset in ['three.min.js', 'projective_common.js', 'projective_theme.css', 'THREE_LICENSE.txt']:
            shutil.copyfile(original / asset, output / variant / asset)
    manifest = dict(original_root=str(original), group_root=str(group_root), original_scenes=len(records), selected_scenes=[r['dataset'] for r in selected],
        G_left_multiply_camera_to_base=G.tolist(), candidate_status=candidate['status'], test_frame_limit=args.test_frame_limit,
        new_input_baseline=True, no_knn_graph=True, source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256((group_root / 'candidate.json').read_bytes()).hexdigest(),
        exporter_sha256=hashlib.sha256((args.exporter_repo / 'tools/write_multicam_rgbd_pages.py').read_bytes()).hexdigest(),
        production_modified=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    completed, failures = [], []
    controls = [r for r in records if r['dataset'] not in group['scenes']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(rerun_scene, (str(original), str(output), str(args.exporter_repo), r['dataset'], G, args.test_frame_limit)): r['dataset'] for r in selected}
        for future in as_completed(jobs):
            try:
                rec = future.result(); completed.append(rec)
                print(f'{len(completed)}/{len(selected)} {rec["dataset"]}: {rec["frames"]} frames; changed since archive {rec["archived_changed_frames"]}', flush=True)
            except Exception as exc:
                failures.append(dict(scene=jobs[future], error=str(exc)))
                print(f'FAILED {jobs[future]}: {exc}', flush=True)
            write_index(output, sorted(completed, key=lambda r: r['dataset']), controls, original)
    maximum = lambda key: max((r[key] for r in completed), default=None)
    metrics = [m for r in completed for m in r['nearest_metrics']]
    result = dict(scenes=len(completed), expected_scenes=len(selected), frames=sum(r['frames'] for r in completed),
        archived_changed_frames=sum(r['archived_changed_frames'] for r in completed),
        max_rigid_transform_error_m=maximum('max_rigid_transform_error_m'),
        relative_camera_matrix_max_delta=maximum('relative_camera_matrix_max_delta'),
        nearest_neighbor_pair_checks=len(metrics), max_nearest_distance_change_m=max((m['max_distance_change_m'] for m in metrics), default=None),
        records=sorted(completed, key=lambda r: r['dataset']), failures=failures)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Controlled rerun of the November 3 multicamera RGB-D pages', '',
        f'Completed {result["scenes"]}/{result["expected_scenes"]} final-session datasets, {result["frames"]} frames, {len(failures)} failures.', '',
        'Two new exports reconstruct the same current RGB-D observations and masks using the original camera poses and the shared final-session candidate, respectively. '
        'The original November 3 HTML template, assets, pixel grid, frame selection and no-kNN settings are preserved. '
        'Earlier real sessions and synthetic scenes have no applicable final-session correction and are linked as unchanged controls.', '',
        f'Frames whose archived cloud differs from the current-input baseline: {result["archived_changed_frames"]}. '
        'Those differences must not be attributed to the camera correction.', '',
        f'Maximum change in a relative camera matrix: {result["relative_camera_matrix_max_delta"]}.',
        f'Maximum component error between fresh corrected-pose backprojection and G applied to baseline cloud: {result["max_rigid_transform_error_m"]} m.',
        f'Nearest-neighbor pair checks: {len(metrics)}; maximum distance change: {result["max_nearest_distance_change_m"]} m.', '',
        'Nearest-neighbor checks use first/middle/last frames per dataset, at most 2048 deterministic source samples per camera, and all target points. '
        'Cloud exports and rigid-transform consistency checks cover every selected frame and every masked valid depth pixel. '
        'Nearest neighbors measure surface proximity, not physical correspondence or absolute depth accuracy.', '',
        'For C_i camera-to-world and one common rigid G, (G C_j)^-1 (G C_i) = C_j^-1 C_i. '
        'All inter-camera projective coordinates and rigid-invariant distances therefore remain unchanged in exact arithmetic. '
        'Tiny stored-cloud differences reflect float32 quantization. This candidate can change robot/world registration but cannot repair relative multiview overlap.', '',
        f'The applied candidate status remains {candidate["status"]}. No source dataset, original export, mask or production camera model was edited.', '',
        'Open index.html for the paired viewer, or baseline/<scene>.html and corrected/<scene>.html. '
        'comparison.json and receipts/<scene>.json preserve numeric metrics, archive comparisons and current input hashes.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(output / 'index.html', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
