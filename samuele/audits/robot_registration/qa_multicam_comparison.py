#!/usr/bin/env python3
"""Use the existing RGBDekf headless browser to verify paired full-resolution viewer pages."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--exporter-repo', type=Path, default=Path('/home/samuelemara/Joint3DGS/RGBDekf'))
    parser.add_argument('--scenes', nargs='+')
    args = parser.parse_args()
    output = args.output_root.resolve()
    manifest = json.loads((output / 'manifest.json').read_text())
    results = json.loads((output / 'comparison.json').read_text())
    records = {r['dataset']: r for r in results['records']}
    selected = args.scenes
    if selected is None:
        objects = {}
        for name in sorted(records):
            objects.setdefault(name.rsplit('_episode_', 1)[0], name)
        selected = list(objects.values())
    if not selected or set(selected) - records.keys():
        parser.error('Requested scenes are missing from the completed comparison')
    sys.path.insert(0, str(args.exporter_repo))
    from tools.qa_projective_debug import Browser
    qa = output / 'qa'; qa.mkdir(exist_ok=True)
    checks, screenshots = [], []

    def check(name, value, detail=None):
        checks.append(dict(check=name, passed=bool(value), detail=detail))
        print(('PASS ' if value else 'FAIL ') + name, flush=True)

    browser = Browser(output)
    try:
        session = browser.open('index.html')
        def js(expression):
            return browser.js(session, expression)
        def wait(expression):
            until = time.monotonic() + 45
            while time.monotonic() < until:
                if js(expression):
                    return True
                time.sleep(.2)
            return False
        child = lambda side, expression: f'document.getElementById({json.dumps(side)}).contentWindow.eval({json.dumps(expression)})'
        available = lambda side: f"document.getElementById('{side}').contentWindow.__comparisonSnapshot"
        ready = ' && '.join(f'({available(s)} && document.getElementById("{s}").contentWindow.__comparisonSnapshot().ready)' for s in ['left', 'right'])
        if not wait("!!document.getElementById('scene')"):
            raise RuntimeError('Comparison index did not load')
        for name in selected:
            js(f"document.getElementById('scene').value={json.dumps(name)};document.getElementById('scene').onchange()")
            loaded = wait(ready)
            check(f'{name}: both viewers load', loaded)
            if not loaded:
                continue
            frames = records[name]['frames']
            for index in sorted({0, frames // 2, frames - 1}):
                js(f"document.getElementById('frame').value={index};document.getElementById('set').click()")
                wanted = f'{ready} && ' + ' && '.join(f"document.getElementById('{s}').contentWindow.__comparisonSnapshot().frame==={index}" for s in ['left', 'right'])
                loaded = wait(wanted)
                check(f'{name} frame {index}: synchronized frame selection', loaded)
                if not loaded:
                    continue
                left = js("document.getElementById('left').contentWindow.__comparisonSnapshot()")
                right = js("document.getElementById('right').contentWindow.__comparisonSnapshot()")
                check(f'{name} frame {index}: same displayed point count', left['points'] == right['points'] and left['points'] > 0, [left['points'], right['points']])
                check(f'{name} frame {index}: no displayed load error', not left['errors'] and not right['errors'])
                X = np.array(js(child('left', 'Array.from(current._xyz.slice(0,300))'))).reshape(-1, 3)
                Y = np.array(js(child('right', 'Array.from(current._xyz.slice(0,300))'))).reshape(-1, 3)
                G = np.array(manifest['G_left_multiply_camera_to_base'])
                err = float(np.max(abs(Y - (X @ G[:3, :3].T + G[:3, 3]))))
                check(f'{name} frame {index}: displayed vertices use corrected poses', err < 1e-6, err)
                check(f'{name} frame {index}: displayed frusta use corrected poses',
                      all(np.allclose(np.array(b['T_wc']), G @ np.array(a['T_wc']), atol=1e-10, rtol=0) for a, b in zip(left['cameras'], right['cameras'])))
            js("document.getElementById('syncview').click()")
            time.sleep(.3)
            pa = js(child('left', 'view.camera.position.toArray()'))
            pb = js(child('right', 'view.camera.position.toArray()'))
            check(f'{name}: matching camera-view button', np.allclose(pa, pb, atol=1e-10, rtol=0))
            png = qa / f'{name}.png'; browser.shot(session, png); screenshots.append(str(png))
        check('no parent-page JavaScript errors', not js('window.__errors||[]'), js('window.__errors||[]'))
    finally:
        browser.close()
    passed = all(c['passed'] for c in checks)
    (qa / 'report.json').write_text(json.dumps(dict(status='PASS' if passed else 'FAIL', checks=checks,
        screenshots=screenshots, scenes=selected, scope='first/middle/last exported frames of one episode per object by default; displayed vertices, frusta, frame controls and matched view'), indent=2) + '\n')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
