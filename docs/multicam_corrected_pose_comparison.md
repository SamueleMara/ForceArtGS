# Rerunning the November 3 multicamera point-cloud pages

The original result collection is:

```
/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/multicam_rgbd_20261103_122136_ee67f0c0
```

Its exporter is `RGBDekf/tools/write_multicam_rgbd_pages.py`. The archived pages
contain every masked valid RGB-D pixel in every frame, with camera frusta, frame
slider/playback and per-camera colours/toggles. They have no kNN graph. The
controlled rerun keeps that original HTML template and the original local viewer
assets, pixel grid, frame indices and point density.

## Controlled comparison

`scripts/audits/robot_registration/rerun_multicam_pages.py` in Sapien imports the
existing exporter rather than replacing its backprojection code. It writes two
new exports from the **same current RGB, depth and masks**:

- `baseline/`: original supplied camera poses.
- `corrected/`: the final-session candidate, with `C_new = G @ C_old` for every
  camera. The corresponding world-to-camera model is `W_new = W_old @ inverse(G)`.

The correction is the explicit candidate from
`ee_tcp_registration_final_session_20261112/candidate.json`, whose status remains
`UNVERIFIED_COMMON_MOUNT_HYPOTHESIS`. It is a diagnostic comparison, not an
installation of a validated calibration.

All 110 `final_*` episodes belong to the operator-confirmed common acquisition and
receive this candidate. Their archived frame selection contains 19,823 frames.
The original collection's 18 earlier-real/synthetic controls do not receive the
final-session correction; the new index links to their original pages.

The black-box and orange-suitcase masks have changed since the archived export.
For example, frame 0 camera 000 of final black box episode 0 had 37,400 mask pixels
in the archived page and has 19,910 in the current mask. Therefore comparing the
new corrected cloud directly with the old export would mix mask changes with
camera-pose changes. Every rerun frame also records whether its fresh baseline
matches the archived XYZ/RGB/camera arrays exactly.

## Commands

This reuses the existing `yolo26` Python environment and RGBDekf repository. It is
CPU image/geometry processing, with bounded workers and no GPU model inference.
The full pair of exports needs roughly 70 GB of additional disk space; the script
checks free space and refuses to overwrite an existing result folder.

The completed comparison root created on 2026-11-12 is
`results_11_10_26/multicam_rgbd_corrected_pose_comparison_20261112`.
Choose a fresh root for a new run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/audits/robot_registration/rerun_multicam_pages.py \
  --original-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/multicam_rgbd_20261103_122136_ee67f0c0" \
  --group-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ee_tcp_registration_final_session_20261112" \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/multicam_rgbd_corrected_pose_comparison_review_01" \
  --workers 4
```

Optional pilots use `--scenes NAME... --test-frame-limit 3`. A zero frame limit,
the default, exports every archived frame. This option does not change the pixel
grid or point density.

Open the new `index.html` directly in a browser. Select an episode, enter a frame
index and click **Set both frames**. **Match right view to left** aligns the orbit
camera so both clouds can be inspected from the same world viewpoint. Each side
also links to its full-size standalone page. The original November 3 export is a
separate reference in the table.

After the export finishes, validate the interactive pages in headless Chrome:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/audits/robot_registration/qa_multicam_comparison.py \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/multicam_rgbd_corrected_pose_comparison_20261112"
```

By default browser QA checks one episode of each of the eight objects at its
first/middle/last frame. It verifies synchronized frame controls, displayed point
counts, displayed corrected XYZ, transformed frusta, matched viewing position and
load errors. Screenshots and the report are saved in `qa/`.

## What can improve

A common rigid transform preserves all relative camera models:

```
inverse(G @ C_j) @ (G @ C_i) = inverse(C_j) @ C_i
```

Consequently it also preserves cross-camera projections and distances, including
inter-camera nearest-neighbor distances, in exact arithmetic. It moves the whole
camera-derived cloud and its frusta relative to the robot base; it cannot repair
misalignment between camera-derived surfaces. Such an improvement requires a
relative-camera calibration change, which is a separate estimation problem.

The rerun checks every exported point against `G @ X_baseline` and every relative
camera matrix. It also measures cross-camera nearest-neighbor distances before
and after on first/middle/last frames per dataset, using at most 2,048 deterministic
source samples per camera and every target point. These metrics measure proximity,
not certified same-surface correspondence or absolute sensor accuracy. Tiny changes
in float32 stored clouds are numerical rounding, not calibration improvement.

## Completed results on 2026-11-12

All 110 final-session datasets and 19,823 frames completed with zero failures.
Each variant contains 1,570,044,928 backprojected points across its frames. The
maximum relative-camera matrix change was `4.44e-16`; the largest sampled
nearest-neighbor distance change was `1.87e-7 m` (`0.000187 mm`). Multiview overlap
did not improve under this common rigid candidate.

The current-input baseline differs from the November 3 archive in 3,651 frames
across the 12 black-box and 12 orange-suitcase episodes. All remaining archived
frames match the current-input baseline XYZ/RGB/camera arrays exactly.

Five numerical tests pass. Headless Chrome passed 137 checks over all eight object
families, including first/middle/last frames, and saved eight screenshots in `qa/`.
The exact export code used for the full run is retained in `execution_code/` and
matches the source hash in `manifest.json`. The subsequent repository navigation
fix points standalone pages directly back to the comparison index; the completed
export has equivalent redirect pages for that link.

## Files and provenance

- `index.html`: paired viewer, original-export links and unchanged controls.
- `baseline/<scene>.html`, `corrected/<scene>.html`: standalone viewers with exact
  camera-to-world matrices embedded in their metadata.
- `<variant>/<scene>_data/f_NNNNNN.js`: full-resolution backprojected XYZ, RGB,
  source-camera identities and per-camera point/depth counts.
- `REPORT.md`, `comparison.json`: counts, failures and measured before/after changes.
- `receipts/<scene>.json`: every-frame archive comparison and current RGB/depth/mask
  hashes, plus the sampled nearest-neighbor metrics.
- `manifest.json`: original collection, selected acquisition group, candidate
  transform/status and code/candidate hashes.
- `qa/`: browser verification and screenshots.

The original result collection and production datasets/calibration are read-only.
FreeArtGS retains copies of the scripts and tests; Sapien is the documented entry
point. Five numerical tests cover rigid backprojection, invariance, sensitivity to
an individual-camera perturbation, invalid transforms and metadata/template handling:

```bash
cd /home/samuelemara/Joint3DGS/Sapien
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  -m unittest discover -s tests -p test_multicam_pose_comparison.py -v
```
