# Exhaustive real RGB-D depth-error audit

The standalone audit implements the investigation in
`RGBD_ALL_DATASETS_DEPTH_ERROR_AUDIT_CODEX.md`. It reads original and converted
recordings; it never runs conversion, masking, ICP, recalibration or training.
Existing preparation/model code and all input files stay unchanged.

Canonical Sapien entry point:
`/home/samuelemara/Joint3DGS/Sapien/scripts/audits/real_rgbd/audit_real_rgbd.py`.
An identical audit-code copy is retained under
`/home/samuelemara/FreeArtGS/samuele/audits/real_rgbd/`.

## Confirmed local scope

The discovered real root contains **118 scenes, 71,154 frame/camera observations,
and 142,308 ordered camera-pair observations**, including older real recordings
as well as all `final_*` episodes. There are two copied-calibration hash strata.
All 118 recorded raw-source paths were resolved from copied episode Parquet
hashes or explicit recorded source pointers. The inventory is rebuilt at launch;
new eligible real scenes are included. Synthetic Sapien datasets are excluded
from real aggregates; synthetic numerical controls live in the test suite.

## Run the full debug audit

```bash
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/audits/real_rgbd/audit_real_rgbd.py \
  --converted-roots "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --raw-roots "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/assets" \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/rgbd_depth_audit_all_real_2026_11_07" \
  --all-scenes --all-frames --all-cameras \
  --mode full
```

The existing `yolo26` Python provides NumPy, SciPy, OpenCV/SIFT, Polars and
Matplotlib. PyArrow was added on 2026-11-08 for sequential Parquet merging.
On another installation, install it with that interpreter's `-m pip install
pyarrow`. This is CPU scientific analysis; no YOLO/model inference, GPU,
weights or downloads are involved. `--threads` controls OpenCV threads (default
1). `--chunk-frames` defaults to 16 and changes write/cache granularity, never
which frames/pixels are analyzed. Full mode requires all three exhaustive flags
and has no frame-stride or maximum-frame option.

The output directory must be **new** and outside both converted and raw trees.
Do not create it beforehand for logging; the script creates it. Use a new name
if that audit folder already belongs to another run. Audit outputs can be large:
all dense observations and RGB correspondences are processed, partitioned
Parquet tables are retained, and temporary per-scene temporal arrays use disk
rather than keeping an entire recording in RAM. The temporary arrays are
released only after that scene's final maps/receipt have been committed.

After an interruption, run the identical command with `--resume` appended.
Matching completed chunks/scenes are validated and reused; no duplicate rows are
appended. Changed input bytes, code, settings or dependency versions cause
resume refusal. A changed dataset needs a new audit directory. A failed scene
is visible in receipts and does not silently remove other scenes. Nonzero exit
status indicates a failure or failed required integrity check; inspect the
report/JSONL to distinguish input findings from software errors.

## Inventory and pilot gates

Use a separate NEW `--output-root` with `--mode inventory` to write just the
manifest, inventory and coverage receipt; do not interpret it as analyzed data.
For example, retain the full command's roots and use
`--output-root /tmp/rgbd_audit_inventory_new --mode inventory`, omitting the
three exhaustive flags.

## Recover the completed run killed during report merging

The 2026-11-07 full run finished all 118 scene receipts (71,154 frame/camera
and 142,308 ordered-pair observations), then was OOM-killed during the final
patch-table merge. Linux recorded 41.3 GiB resident memory and no free swap.
`patches.parquet.tmp` opened at 16:46:47; the kill occurred at 16:47:15.
The old Polars multi-file sink buffered too much data. Reducing row-group size
alone also failed a guarded stress test, so the merge now uses a sequential
Arrow reader/writer: one file, 128-row read batches, and an approximately 8 MiB
write buffer (plus the current batch), capped at 1,024 rows. Parquet footer
metadata still grows with row-group count. Reports retain compact patch/lag
statistics, group patches one scene at a time, and keep detailed scene receipts
on disk instead of duplicating their contents in memory.

To finish this existing run without rerunning measurements:

```bash
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/audits/real_rgbd/finalize_real_rgbd.py \
  --audit-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/rgbd_depth_audit_all_real_2026_11_07"
```

The finalizer requires completed scene/chunk receipts, verifies every partition
hash and recorded row-count accounting, then rebuilds derived tables/reports
in the same audit folder. Incomplete or corrupt measurements cause refusal.
It never rewrites measurement partitions or `manifest.json`. A separate
`finalization_receipt.json` records the original manifest hash, reporting code
and dependency versions. Original raw/converted inputs are not revalidated;
these reports summarize the already recorded measurements. The failure gallery
reads current RGB/depth/mask images for display only.

Use this dedicated finalizer for recovery, **not `--resume` on the measurement
driver**: normal resume correctly rejects changes to its code/environment hash.
No blanket hash bypass was added. The finalizer can be rerun after interruption;
derived tables are written to temporary files and replaced after completion.
Do not run two finalizers against the same output folder at once.

A pilot is explicitly non-conclusive:

```bash
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/audits/real_rgbd/audit_real_rgbd.py \
  --converted-roots "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --raw-roots "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/assets" \
  --output-root /tmp/rgbd_audit_pilot_new \
  --mode pilot \
  --pilot-scenes final_black_box_episode_000000 final_orange_suitecase_episode_000000 fridge_plastic_20261016_163314
```

This processes early/middle/late base frames, including black-box output frames
0, 114 and 228, with neighborhood frames used for lag controls. Pilot output
is labelled PILOT / NON-CONCLUSIVE; all other scenes remain explicitly
unscheduled in its coverage receipt. Pilot and full runs use separate output
directories because their coverage/settings differ.

## Measurements and their limits

- E0: validates mappings, units, dtype/shape, binary masks, timestamp provenance,
  copied calibration and file hashes; reproduces production depth registration
  and RGB remapping in RAM. Bitwise differences are failures, not rounded-away
  "near enough" equality. Source/native evidence may be unavailable while
  converted-domain tests still run.
- E1: records every native sample's projection/rejection, collisions, occupancy
  and deterministic z-buffer winners. Regional coverage includes mask-distance
  bands. Conservative measured-point footprint experiments only produce
  candidate hole support; no filled depth enters official residuals. A target
  zero is not uniquely attributable to a native zero with unknown depth.
- E2: same-native-ray and registered-domain temporal median/MAD, transitions,
  drift and conditional RGB-stationary support; checkerboard held-out plane and
  curvature fits, patch tracking, temporal block CIs. Areas without textured,
  repeatable RGB stationarity evidence stay unknown. Outside a mask is not
  automatically static. Constant bias is invisible to temporal precision.
- E3: uncapped SIFT detection, reciprocal descriptor matches, fixed-calibration
  Sampson residuals, RGB-only triangulation, cheirality/parallax/reprojection
  controls and localization covariance. No depth residual enters initial match
  selection. Calibration covariance is absent: localization uncertainty and
  declared pose-sensitivity sweeps do not certify absolute depth accuracy.
- E4: every valid registered pixel is reprojected for every ordered camera pair,
  with source z-buffer visibility. All-candidate signed distributions/histograms
  retain broad tails. Independent RGB-supported residuals, homography-supported
  local planes and three-camera RGB-cycle leave-one-out checks are separate.
  Occlusion-compatible residual signs alone never identify sensor bias.
- E5: all original RGB-video frames and native depth files get content-repeat
  receipts. Camera-camera RGB lag sweeps and within-camera RGB/depth lag tests
  use moving features and stationary negative controls. Edge-overlap lag is a
  separately labelled confounded proxy. Nominal/recorder timestamps and decoded
  content-change rates are not hardware timestamps or distinct-exposure FPS.
- E6/E7: separately constrained robust bias/scale and epipolar pose diagnostics,
  episode/scene and stationary-image-tile holdouts, lag holdouts, scene-cluster
  and temporal-block uncertainty, region/dependence curves and complete
  observation receipts. Epipolar pose fitting keeps baseline length fixed;
  its unobservable scale is never fitted as independent metric calibration.
  Image tiles are conditional physical-patch proxies, not externally annotated
  reference surfaces. Diagnostic coefficients never enter production poses.

Every observation has evidence/status even if a test has no support. Missing
matches/weak parallax/absent timestamps are `not_measurable` or `UNOBSERVABLE`,
not zero error. Cross-view residual sign is projected source Z minus target D;
RGB-reference residual sign is measured sensor D minus triangulated RGB Z.
Do not combine these residual types in one statistic.

## Outputs to inspect

```text
manifest.json                         # roots, scenes, schemas, hashes, versions
frames.parquet                        # one row / scene / output frame / camera
pairs.parquet                         # one row / frame / ordered camera pair
unordered_pairs.parquet               # i<j; both signed directions preserved
correspondences.parquet               # RGB matches, 3D points, uncertainty
patches.parquet                        # native and RGB-supported plane candidates
timesync.parquet                       # repeats/quality/lag observations
attribution.parquet                    # non-exclusive grades and evidence
summary_by_dataset.csv
summary_by_camera.csv
summary_by_frame.csv
coverage_receipt.csv
quality_assurance.json                # unique-key and expected-count checks
skips_and_failures.jsonl               # exact missing/failed paths and reasons
heldout_counterfactual_models.json
heldout_timing_models.json
patch_track_uncertainty.json
dependence_curves.csv
worst_frames.csv
ALL_REAL_DATASET_DEPTH_ERROR_ROOT_CAUSE_REPORT.md
report.md
figures/                              # inventory, pair time series, dependence
failure_gallery/                      # ranked cases; not a guilty-camera ranking
scenes/<scene_id>/                     # provenance, matches and calibrated 3D
  receipt.json
  temporal_quality.json
  temporal/*_temporal_maps.{npz,png}
  cam_*_raw_series.parquet
partitions/<scene_id>/chunk_*/         # atomic tables and resume receipts
.runtime/matplotlib/                  # audit-local plot cache
```

Parquet tables expose common typed scalar columns and retain the complete
per-observation record in `metrics_json` (including histograms, covariance,
thresholds, provenance and unavailable reasons). Thus nested populations with
different scientific meaning remain separate. Full RGB matches are stored;
plot thumbnails alone use a clearly labelled deterministic display sample.
All calibrated cloud points for diagnostic frames are saved as NPZ; 3D scatter
plots use a display sample, never an analysis subsample.

For example:

```python
import polars as pl

bad_replays = (
    pl.scan_parquet("/path/to/audit/frames.parquet")
    .filter(pl.col("depth_replay_different_pixels") > 0)
    .select("dataset", "frame_index", "source_frame_index", "camera",
            "depth_replay_different_pixels", "depth_replay_max_abs_m")
    .collect()
)
```

Start with `coverage_receipt.csv` and `quality_assurance.json`, then the report
and worst-frame figures. A full result must account for every expected
observation, including failed and unavailable tests. Absolute metric sensor
bias, material properties, true exposure age and perfect physical surface
identity remain unverified without independent measurements. The audit does
not claim they become identifiable just because a regression fits well.

## Tests

```bash
cd /home/samuelemara/Joint3DGS/Sapien
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  -m unittest discover -s tests -p 'test_real_rgbd_audit.py'
```

The tests include independent transform/distortion/registration/triangulation
oracles, known injected scale/pose/lag controls, missing evidence, held-out
model checks, immutable input hashes, and interruption/resume without duplicate
records. Numerical controls are synthetic and do not constitute real accuracy GT.

## Local validation on 2026-11-07

The non-conclusive pilot completed on `final_black_box_episode_000000`,
`final_orange_suitecase_episode_000000` and `fridge_plastic_20261016_163314`.
It emitted exactly 27 distinct frame/camera rows and 54 distinct ordered-pair
rows, with no failed observations. All 27 sampled RGB and registered-depth
images reproduced the existing converter bit-for-bit. The pilot also reported
40 unavailable geometry tests; those retain null errors and explicit reasons.
This verifies the sampled conversion replay, not absolute depth accuracy or
complete coverage of all scenes.

Pilot evidence is in `/tmp/rgbd_audit_pilot_2026_11_07_v1/`, with its log at
`/tmp/rgbd_audit_pilot_2026_11_07_v1.log`. Temporary files may be removed by the
system. The pilot exposed overflow warnings when subtracting morphology
sentinels in empty neighborhoods. The audit now subtracts only in valid
neighborhoods; a regression test rejects invented support from empty inputs.
At that pilot-validation gate the full audit had not been launched. The later
full run completed its measurements but was killed during reporting; use the
recovery instructions above for that existing output folder.
