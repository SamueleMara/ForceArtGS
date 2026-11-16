# Interaction bounds and joint constraints from calibrated TCP trajectories

**Early drawer mounting:** the three `white_drawer_20261021_161154_episode_00000[0-2]` recordings used a different force-sensor mounting (user confirmation, 2026-11-12). Their EE-to-TCP offset and sensor-axis rotation are currently uncalibrated. The registry marks `tcp_calibration_valid: false`; TCP generation, joint estimation, force plotting and HTML refresh refuse these recordings until their own transforms are established. The retained +34.87365 mm matrix is historical provenance, not a valid calibration for this mounting. Raw EE and wrench measurements remain available in Parquet and cached measurement archives.

The script `scripts/estimate_real_tcp_joints.py` uses calibrated measurements from `analyze_real_robot_measurements.py`, identifies first/last force events, checks image correspondences and saved dynamic-training windows, and fits **both revolute and prismatic** TCP trajectory constraints.

Run all 118 converted real scenes on CPU:

```bash
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/estimate_real_tcp_joints.py \
  --real-root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --measurements-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_robot_measurements" \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_tcp_joint_estimation" \
  --joint-code /home/samuelemara/joint_estimation \
  --workers 4
```

Add `--dry-run` to check discovery without writes, or `--overwrite` to rerun into an existing analysis root. For a pilot, use a separate output root and add:

```bash
--scene final_red_fridge_episode_000000 final_light_blue_box_episode_000000 \
  final_black_laptop_episode_000000 final_wooden_box_episode_000000
```

Input measurements, datasets, calibration, fitter source and production training are read-only. The script verifies each measurement archive's source Parquet hash. Outputs and Matplotlib cache stay under the separate analysis root. Processed RGB-D frame indices are mapped through the actual conversion metadata and recorded Parquet frame/timestamp arrays; no `raw_frame == RGBD_frame` or nominal-FPS assumption is used.

## Force and visual intervals

The detector uses recorded force magnitude. Its prominence threshold is the larger of **0.5 N** and **six baseline noise standard deviations**, where noise is a robust estimate from first differences during the first 0.5 seconds. Peaks within 0.15 s are grouped as one event; the strongest peak represents each group. The first and last distinct events propose the inclusive interaction bounds. All candidate peaks, event heights, widths and exact source frames/timestamps remain available. A broad load peak is flagged; it is not guaranteed to mark the precise moment of contact. No force samples are filtered, filled or modified.

Use `--min-prominence-N 0.2` to explicitly examine weaker force events. `--force-sigma` and `--spike-separation-s` are also available. Changing a threshold can change the proposed interval; it is not evidence that a physical contact boundary was established.

Every consecutive RGB-D image pair in every camera is examined using forward/backward Lucas–Kanade correspondences within both object masks. Noise is controlled using slower background tracks from the fixed cameras, limiting contamination from the robot outside the object mask. A projected EE-to-TCP wrist/shaft envelope excludes likely robot occluders from both object and background tracking. This is an **analysis-only geometric control**, using a 40-mm wrist radius, a 12-mm shaft radius and a shaft endpoint 25 mm short of the calibrated tip; it is not an independently segmented robot mask. It can remove nearby object features and reduce observability. Original dataset masks are unchanged, and excluded pixel counts are reported. Persistent image motion supported by at least two views is preferred. When only one view has persistent support, the result is explicitly conditional. Missing features, bad masks and remaining robot leakage remain limitations. The analysis neither changes calibration nor applies synchronization shifts.

Force/visual agreement and disagreement are recorded separately. If two force events are unavailable but a persistent visual interval exists, the fit uses a clearly labeled **visual-only fallback**. If neither is measurable, fitting is `NOT_RUN`. When force and visual intervals conflict, the requested force-bounded fit is kept, flagged, and accompanied by a separate visual-interval diagnostic fit. The code does not choose whichever interval produces the preferred joint label.

## Actual dynamic-training interval

`--training-results` defaults to the existing `ours_v6_two_cams` results. It reads completed Phase-1 `stage_manifest.json` files, including `MSTART`, `MEND`, source directory, cameras and processing direction. For each available run it reports exact training frames/times, interval overlap and training time before/after the proposed interaction.

Only eight representative episodes have saved runs in that default root. The other 110 scenes remain explicitly `NO_SAVED_TRAINING_WINDOW_FOUND`. A dataset's forward/backward **coverage recommendation is not a training motion interval**. Specify other existing result roots with `--training-results ROOT1 ROOT2` when needed. No training manifest or input sequence is modified.

## Existing fitter and scores

The supplied `/home/samuelemara/joint_estimation/trajectory_constraint_fit.py` is imported directly and remains unchanged. Its PCA directions, normalization, fixed-direction axis fit, cylindrical conversion and circle fit are reused. Its source SHA and Git revision are recorded. No screw model is fitted because this analysis compares the requested two joint families.

Both candidates use the same **3D orthogonal distance in metres**, avoiding the prototype classifier's mixed angular/metric translation score. Scores include RMS, MAE, 95th-percentile error, BIC with four line or six circle parameters, and five contiguous temporal holdouts. Fold axis deviations, radius and observed angular span are retained.

`best_scored_model` selects among identifiable candidate constraints. A circle spanning less than 0.15 rad, or with radius exceeding ten times the observed trajectory extent, is excluded as unidentifiable. Its numeric fit and `raw_bic_best_model` remain visible. Tiny motion, insufficient samples, poor fits, weak model separation and unstable axes are flagged rather than reported as confident object-joint estimates.

A line-like TCP trace does **not establish a prismatic object joint**: a pointed tool may push a revolute part while sliding or changing contact point. A valid object-joint estimate requires the TCP to remain attached to one object-fixed point throughout the fitted interval. The exact prismatic axis location is not identifiable from one point; only its direction is constrained. Absolute coordinates retain the supplied EE/calibration-reference assumption. With fixed EE orientation, changing the constant TCP offset moves the trajectory but cannot turn a line into a circular arc.

## Outputs

The root contains `summary.csv`, `summary.json`, `REPORT.md` and `manifest.json`. The CSV includes raw and RGB-D interval frames, force events, visual/training comparisons, both model scores, selected constraint, axis/direction and quality flags.

Each scene contains:

- `result.json`: bounds, both complete fits, exact frame mapping, score/identifiability status, saved training-window comparisons and input/source hashes.
- `force_events.csv`: every candidate force spike.
- `visual_correspondences.csv`: per-camera/frame match support, motion counts and noise thresholds.
- `interaction_tcp.npz`: precisely the selected source frames/timestamps, RGB-D mapping and calibrated TCP points used for fitting.
- `interaction_bounds.png`: force peaks, visual motion, TCP speed and the recorded training interval on one time axis.
- `boundary_contact_sheet.jpg`: boundary images with tracked-motion arrows (cyan, displayed at four times their measured length), projected calibrated TCP (yellow), and the robot exclusion control (red outline).
- `boundary_correspondences.npz`: unscaled image-track endpoints and moving-point flags used for the contact sheet.
- `joint_models.png`: both fitted constraints, residuals, radius/arc diagnostics and excluded-fit labels; absent if no fit is possible.

`ANALYZED` means the analysis completed. It does not imply corroborated force boundaries, a uniquely identified object joint, or a calibrated GT axis.
