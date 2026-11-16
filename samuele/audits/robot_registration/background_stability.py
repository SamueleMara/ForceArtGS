"""Separate RGB background control; never used to fit the robot registration."""
import json
from pathlib import Path

import cv2
import numpy as np


def features(info, row, camera_arrays, project):
    camera = info['cameras'][row['camera']]
    path = Path(info['scene_path'])
    image = cv2.imread(str(path / 'rgb' / camera['name'] / f'frame_{row["frame"]:06d}.png'))
    if image is None:
        return [], None
    mask = np.full(image.shape[:2], 255, np.uint8)
    mask[:25] = 0; mask[-20:] = 0; mask[:, :20] = 0; mask[:, -20:] = 0
    obj = cv2.imread(str(path / 'mask' / camera['name'] / f'frame_{row["frame"]:06d}.png'), cv2.IMREAD_GRAYSCALE)
    if obj is None:
        return [], None  # no silent inclusion of the appliance
    mask[cv2.dilate(obj, np.ones((25, 25), np.uint8)) > 0] = 0
    C, K = camera_arrays(info['cameras'])
    c = row['camera']; p = np.array(row['p']); axis = np.array(row['axis'])
    for s in np.arange(-.55, .25, .025):
        point = p + s * axis
        uv = project([point], C[c], K[c])
        if uv is None or not np.isfinite(uv).all() or abs(uv).max() > 10000:
            continue
        z = ((point - C[c, :3, 3]) @ C[c, :3, :3])[2]
        radius = int(min(200, K[c, 0, 0] * (.12 if s < .03 else .07) / max(.1, z)))
        cv2.circle(mask, tuple(np.rint(uv[0]).astype(int)), radius, 0, -1)
    return cv2.SIFT_create(nfeatures=5000, contrastThreshold=.008).detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)


def compare(key_a, desc_a, key_b, desc_b):
    if desc_a is None or desc_b is None or min(len(desc_a), len(desc_b)) < 2:
        return dict(status='INSUFFICIENT_DISTINCTIVE_BACKGROUND', matches=0)
    matcher = cv2.BFMatcher()
    forward = {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(desc_a, desc_b, k=2)
               if len(pair) == 2 for m, n in [pair] if m.distance < .7 * n.distance}
    reverse = {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(desc_b, desc_a, k=2)
               if len(pair) == 2 for m, n in [pair] if m.distance < .7 * n.distance}
    pairs = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
    if len(pairs) < 12:
        return dict(status='INSUFFICIENT_DISTINCTIVE_BACKGROUND', matches=len(pairs))
    a = np.array([key_a[i].pt for i, _ in pairs]); b = np.array([key_b[j].pt for _, j in pairs])
    delta = np.linalg.norm(a - b, axis=1)
    stats = dict(matches=len(pairs), median_identity_displacement_px=float(np.median(delta)),
        p95_identity_displacement_px=float(np.percentile(delta, 95)))
    # Split before geometric filtering. H tests image consistency, not physical identity.
    ids = np.random.default_rng(42).permutation(len(a)); cut = int(.75 * len(a))
    H, inliers = cv2.findHomography(a[ids[:cut]], b[ids[:cut]], cv2.RANSAC, 2.)
    if H is not None:
        pred = cv2.perspectiveTransform(a[ids[cut:]].astype(np.float32)[None], H)[0]
        stats.update(training_homography_inliers=int(inliers.sum()), heldout_matches=len(ids[cut:]),
            heldout_homography_median_px=float(np.median(np.linalg.norm(pred - b[ids[cut:]], axis=1))))
    for label, selector in [('upper_image', (a[:, 1] < 240) & (b[:, 1] < 240)),
                            ('lower_image', (a[:, 1] >= 240) & (b[:, 1] >= 240))]:
        stats[label] = dict(matches=int(selector.sum()), median_identity_displacement_px=float(np.median(delta[selector])) if selector.any() else None)
    stats['status'] = 'CONSISTENT_STATIC_IMAGE_FEATURES' if np.median(delta) < 1.5 else 'BACKGROUND_CHANGE_OR_AMBIGUOUS_MATCHES'
    return stats


def run(output, infos, camera_arrays, project):
    cohorts = {}
    for scene, info in sorted(infos.items()):
        cohorts.setdefault(info['calibration_hash'], {}).setdefault(info['recording'], scene)
    results = []
    for calibration, recordings in cohorts.items():
        if len(recordings) < 2:
            continue
        anchor = next(iter(recordings.values()))
        observations = [json.loads(line) for line in (output / anchor / 'observations.jsonl').read_text().splitlines()]
        reference = {}
        for c in range(len(infos[anchor]['cameras'])):
            row = next(r for r in observations if r['camera'] == c)
            reference[c] = features(infos[anchor], row, camera_arrays, project)
        for recording, scene in recordings.items():
            if scene == anchor:
                continue
            rows = [json.loads(line) for line in (output / scene / 'observations.jsonl').read_text().splitlines()]
            for c in range(len(infos[scene]['cameras'])):
                selected = [r for r in rows if r['camera'] == c]
                for i in np.unique(np.linspace(0, len(selected) - 1, 3).astype(int)):
                    row = selected[i]
                    key, desc = features(infos[scene], row, camera_arrays, project)
                    result = compare(*reference[c], key, desc)
                    result.update(calibration_hash=calibration, anchor_scene=anchor, anchor_frame=0,
                        target_recording=recording, target_scene=scene, frame=row['frame'], camera=infos[scene]['cameras'][c]['name'])
                    results.append(result)
    return dict(method='Three evenly spaced frames from first episode per native recording, same-calibration cohort; mutual SIFT ratio .7, robot/appliance excluded; no robot-fit residuals used.',
        limits='RGB control only. Repeated features can survive matching. Upper/lower image bins are not verified wall/table identities. Furniture motion is not proof of camera motion. No depth or absolute robot-base reference is inferred.', results=results)
