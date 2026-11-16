# Robot/camera registration across all real recordings

This is a read-only experiment, added on 2026-11-12. It estimates and tests candidate
camera-to-robot-base corrections; it never edits converted datasets, camera
calibration, masks, point clouds, cached robot measurements or production training.
The implementation is in `scripts/audits/robot_registration/` in Sapien, with a copy
under `samuele/audits/robot_registration/` in FreeArtGS.

## Corrected acquisition grouping: the final datasets

The operator confirmed that all `final_*` datasets were recorded in one go.
Treat their 110 episodes across eight objects as one acquisition group. The
calibration-hash pool from the first audit also included earlier acquisitions and
must not be used as the final-session-only result. Its failed fit did not establish
separate acquisition sessions or camera movement among the final datasets.

`refit_acquisition_group.py` reuses the saved all-frame observations without
rescanning RGB. It excludes earlier sessions, withholds an entire object/native
recording in each fold, bootstraps whole objects and repeats the background control
with its anchor inside the final session. The completed rerun is in
`results_11_10_26/ee_tcp_registration_final_session_20261112`.

```bash
OPENBLAS_NUM_THREADS=1 \
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/audits/robot_registration/refit_acquisition_group.py \
  --audit-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ee_tcp_registration_all_20261112" \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ee_tcp_registration_final_session_review_01" \
  --recording-prefix final_ \
  --group-id final_continuous_acquisition \
  --group-evidence "Operator confirmed all final_* recordings were acquired in one go; earlier sessions excluded." \
  --bootstrap 24
```

Choose an empty output directory; this script refuses to overwrite previous results.
`group.json` records the grouping statement, exact scene membership, source hashes
and input observation hashes. `candidate.json`, `REPORT.md`, `index.html` and
`background_stability.json` contain the revised scores and images.

The final-only rerun contains 27,003 accepted shaft observations. It remains
`UNVERIFIED_COMMON_MOUNT_HYPOTHESIS`: the withheld laptop median worsens from
4.74 to 5.51 pixels, several tails worsen, and the cluster-bootstrap translation
standard deviations are approximately 14.34, 18.05 and 5.37 mm. Background checks
are more consistent with the correct in-session anchor (31 consistent, 11 changed
or ambiguous, 21 insufficient). These outcomes require debugging observations and
pose/calibration modelling within the shared acquisition; they do not establish
multiple final-session camera configurations. No production correction was installed.

To repeat the original November 3 multicamera point-cloud viewers with original
versus candidate poses on identical current inputs, see
[the controlled viewer comparison](multicam_corrected_pose_comparison.md).

## Original all-scene audit

Run from any directory using the existing `yolo26` Python environment:

```bash
OPENBLAS_NUM_THREADS=1 \
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/audits/robot_registration/audit_robot_registration.py \
  --converted-root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --measurements-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_robot_measurements" \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ee_tcp_registration_all_20261112" \
  --frame-stride 1 --workers 2 --bootstrap 24
```

This uses bounded CPU workers for image processing and rigid geometry. No GPU or
new model download is required. Re-running with the same code and unchanged inputs
reuses completed scene extractions. Partial scenes are regenerated inside the audit
folder. Changed code/input fingerprints require a fresh output root. Use
`--phase fit` to repeat only fitting/report generation from saved observations;
the background image checks are repeated. Do not edit the implementation during a
run. `--scenes NAME...` selects a pilot; fit/report generation includes all completed
scene extractions already present in that output root.

## What is measured

1. Inventory every scene's native recording, episode, camera serials, calibration
   session/hash and measurement provenance. Equal matrices are not proof of equal
   physical camera placement.
2. Attempt a shaft observation in every mapped RGB-D frame/camera. A fixed nominal
   EE-local segment `[0.10, 0.18] m` supplies a generous 45-pixel search band.
   Isolated dark cross-sections define an observed image line. Ambiguous, occluded,
   short and nonlinear detections are rejected and counted. The distal segment
   includes the black section below the laptop tool's white wrapping.
3. Keep the user-confirmed Parquet link6 pose and centered tool axis fixed:
   `L(s) = p_link6 + s * R_link6[:,2]`. Fit one six-DOF rigid correction `G`
   per native recording/calibration hash, with `C_new = G @ C_old` for every
   camera. Relative camera transforms remain fixed. This uses image-line geometry,
   not depth, force direction, a fabricated TCP X/Y offset, or the older candidate
   calibration as ground truth.
4. Hold out whole episodes before fitting; a single episode instead uses contiguous
   40-frame blocks. Limit training to 24 evenly distributed observations per
   scene/camera so long recordings do not dominate. Evaluate all accepted held-out
   observations. Bootstrap episode/time blocks when the initial fit is observable.
5. Separately fit an explicit **shared-mount hypothesis** per calibration hash,
   balancing to 12 training observations per scene/camera. Leave an entire native
   recording out for each validation fold. Bootstrap whole native recordings.
   This tests whether combining different tool orientations can resolve weak
   directions without assuming physical camera stability.
6. Check static RGB feature consistency separately between native recordings
   sharing a calibration hash. Compare three evenly spaced frames in the first
   episode with one anchor recording. Exclude appliance masks and a generous robot
   envelope; require mutual SIFT matches with a 0.7 descriptor ratio. Split before
   homography filtering. Save raw identity displacement and held-out homography
   error, including upper/lower image summaries. These are controls, not physical
   identity proof or a second absolute calibration.

The early `white_drawer_20261021_161154_episode_00000{0,1,2}` recordings use the bare
sensor assembly, so the pointed-shaft detector explicitly marks all of their frames
`unsupported_bare_sensor_geometry`. They remain in inventory and coverage; no
shaft-based correction is verified for them. A shared-mount candidate fitted from
other episodes cannot certify their different setup.

## Acceptance and uncertainty

Every candidate is diagnostic. `CANDIDATE_PASSES_IMAGE_CHECKS` requires a successful
interior solution, full Jacobian rank with scaled condition number below 1000,
held-out median error at most 3 pixels and p95 at most 8 pixels, improvement and
adequate coverage in every camera, and stable bootstrap results (maximum component
standard deviation below 5 mm and 0.5 degrees). The pooled hypothesis additionally
checks every withheld native recording and camera. These thresholds are fixed
screening rules, not manufacturer calibration tolerances or absolute accuracy.

The Jacobian uses rotation in radians and translation divided by 0.1 m, with rank
threshold `largest singular value * 1e-4`. Its condition depends on this documented
scale. Bootstrap intervals describe variation conditional on the detector and
accepted observations; they do not include systematic image-line identity errors.

Almost constant tool orientation leaves displacement along the infinite shaft
line weakly or completely unobservable. More frames of the same orientation do
not resolve that problem. A tiny image residual can coexist with an incorrect TCP
position along the shaft. Such fits are `UNIDENTIFIABLE` or `UNVERIFIED`, including
solutions that hit parameter bounds. Fitting all data together is useful only if
the physical configurations are compatible and held-out checks support it.

## Outputs

- `REPORT.md`: coverage, recording candidates, pooled hypothesis and held-out results.
- `index.html`: links to before/after diagnostic images for every scene with a fit.
- `candidate_corrections.json`: per-recording candidate matrices, checks, uncertainty,
  held-out episodes/blocks and per-camera scores.
- `pooled_common_mount_hypothesis.json`: shared candidates, leave-recording-out
  errors, observability, recording bootstrap intervals and explicit acceptance checks.
- `background_stability.json`: independent image-feature controls and insufficient
  or ambiguous cases.
- `cross_recording_transfer.json`: per-recording candidate transfer tests.
- `coverage.json`, `scene_camera_metrics.csv` and `.json`: all cameras, rejection
  counts, before/after residuals. Per-scene metrics include training observations;
  use the held-out results for validation.
- Per scene: `inventory.json`, `extraction.json`, `observations.jsonl`,
  `candidate_overlay.jpg` and, where available, `pooled_candidate_overlay.jpg`.
  Cyan denotes nominal axis, yellow candidate axis and green observed image line.
  Overlay positions are fixed samples of accepted detections, not certified TCP tips.
- `run.json`, `failures.json`: invocation/provenance and extraction failures.

No correction is installed automatically. The line detector is prediction-guided
and can confuse other dark edges with the tool. Matching room texture does not
prove physical identity, especially on movable furniture. No flange/tip landmark
detector, depth-bias correction, timing correction or independent physical mounting
verification is included. Unresolved cases remain explicit rather than forcing a
transform onto every scene.

A common rigid correction changes absolute robot/scene registration. It does not
change internal multiview rendering geometry when camera-derived clouds and cameras
are transformed consistently, or robot-base force/trajectory percentages. If a
candidate is subsequently validated, all masked/unmasked initial/final camera-derived
clouds and COLMAP metadata must be transformed together in a new versioned output;
the source dataset should be preserved.

Tests (known-transform recovery, outlier robustness, parallel-axis ambiguity,
whole-recording holdout, dark-edge ambiguity and background controls):

```bash
cd /home/samuelemara/Joint3DGS/Sapien
OPENBLAS_NUM_THREADS=1 /home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  -m unittest discover -s tests -p test_robot_registration_audit.py -v
```
