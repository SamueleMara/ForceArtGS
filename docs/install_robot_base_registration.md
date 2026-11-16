# Installed robot-base registration for EE tracking

**Contact force sign:** the recorded/plotted vectors are contact reactions acting
on the robot. The robot's force on the object is their negative in the same frame.
See [contact force sign metadata](contact_force_convention.md). This clarification
does not change any force arrays, arrows or camera registration.

The operator requested applying the shared final-session registration after the
multicamera comparison. The operation targets only the 110 `final_*` episodes in
the operator-confirmed continuous acquisition. Earlier real recordings, including
the three old drawers, and synthetic scenes are excluded.

The transformation comes from
`results_11_10_26/ee_tcp_registration_final_session_20261112/candidate.json`:

```
C_camera_to_base_new = G @ C_camera_to_base_old
P_camera_derived_new = R_G @ P_camera_derived_old + t_G
W_colmap_new        = W_colmap_old @ inverse(G)
```

It is the same candidate used in the paired viewer comparison, with translation
part approximately `[-19.577, -23.039, +37.931] mm`. Its image-fit status is still
`UNVERIFIED_COMMON_MOUNT_HYPOTHESIS`. Applied dataset metadata explicitly says
`OPERATOR_APPLIED_ROBOT_REGISTRATION_UNVERIFIED`; the operation does not turn it
into a certified calibration or establish exact TCP/contact accuracy.

## What changes

- `metadata/cameras.json`: active camera-to-base poses, registration provenance
  and the explicit applied/unverified status.
- All initial/final, masked/unmasked COLMAP image poses and points, in both text
  and binary formats. Intrinsics, IDs, endpoint image membership, tracks and 2D
  observations are preserved.
- Existing `colmap_rgb` / `colmap_rgb_final` models, including `known_poses`, modern
  rig/frame world poses, `cameras.yaml` extrinsics and SQLite image pose priors.
  Relative rig transforms, matches and feature coordinates remain unchanged.
- Every PLY in the geometry folders, including `pointcloud/views/start` and
  `pointcloud/views/end`. World positions move by G; normals rotate by R_G;
  colours and other scalar attributes remain unchanged.
- Export/conversion summaries receive the registration record.
- `metadata/camera_calibration_robot_base.json`: active native-camera calibration
  counterpart, preserving the original intrinsics/depth-to-colour calibration.
- `metadata/robot_base_registration.json`: completion marker, exact transform,
  source, backup location and active/untouched file hashes.

The original `metadata/raw/camera_calibration.json` remains acquisition provenance.
Raw Parquet link6 poses, the tool assignment and local EE-to-TCP Z transform,
timestamps, RGB, camera-frame depth, masks and robot-base force measurements are
not transformed. EE tracking should read active `metadata/cameras.json` and use:

```
p_camera = inverse(C_camera_to_base_new) @ p_robot_base
pixel = project(K, p_camera)
```

The supplied centered link6/tool geometry is retained. No local X/Y TCP patch or
table-height adjustment is introduced. Robot-only tangential force statistics
remain unchanged; their camera projections use the new camera poses on regeneration.

## Application and backups

The updater uses the existing `pycolmap_env`, because native COLMAP 3.12.5 handles
both legacy models and modern rig/frame poses. It needs no GPU, new dependency
or model download. All geometry is backed up before staging, checked before
publication, and replaced atomically per file. A publication failure rolls the
scene back. Reapplying the same registration verifies the installed hashes and
skips it; another transform or externally edited registered files are rejected.

There is a per-report lock and a per-scene in-progress marker. Do not regenerate
geometry while this operation is running. RGB/depth/mask sequences are never walked
or rewritten by the updater. The application plan found 867 COLMAP models and
approximately 21.99 GB of original geometry to preserve.

Backup/report location for the application on 2026-11-12:

```
/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/robot_base_registration_applied_20261112
```

Original affected files are under `backups/<scene>/before/`. Each scene has a
`state.json` with before/after hashes and model/PLY checks. The source RGB/depth,
raw robot recordings and archival calibration remain in their original locations.

The completed application verified all 110 scenes, 867 COLMAP models, 1,755 PLY
files and 660 database pose priors. All 8,542 active-file hashes, 8,432 original
backup hashes and 440 untouched robot/raw metadata hashes passed. Nine tests pass,
and the existing production RGB-D loader successfully loads registered datasets.
`index.html` in the report folder contains EE/TCP previews for all eight objects;
`verification.json` contains the complete receipts. The eight earlier real scenes
retain their original camera matrices and have no registration marker.

Repeat verification / apply to still-unregistered scenes with the same command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/samuelemara/miniconda3/envs/pycolmap_env/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/audits/robot_registration/apply_robot_base_registration.py \
  --converted-root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --group-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ee_tcp_registration_final_session_20261112" \
  --report-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/robot_base_registration_applied_20261112" \
  --phase apply --workers 2
```

Use `--phase plan` to inventory without changing datasets. `--scenes NAME...`
selects a subset. A failed staging attempt keeps the original active scene intact
and retains its backup for inspection; it is not silently reused.

To deliberately undo this registration, run the same command with `--phase restore`.
Completed-scene restoration first checks that registered files have not changed
since application, then restores the original bytes and removes the registration
markers/active native-camera counterpart. Backups are retained.

## Verify and preview EE tracking

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/samuelemara/miniconda3/envs/pycolmap_env/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/audits/robot_registration/verify_applied_registration.py \
  --converted-root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --group-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ee_tcp_registration_final_session_20261112" \
  --report-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/robot_base_registration_applied_20261112" \
  --measurements-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_robot_measurements"
```

This checks every installed file hash, untouched robot/raw metadata and exact
one-time camera registration. It writes `verification.json`, `REPORT.md` and an
`index.html` gallery with original/registered EE-to-TCP projections on RGB: one
episode per object, three times, all three cameras. Crosses mark link6, circles
mark TCP. These are diagnostic projections, not independently annotated contact.

Previously generated tracking videos and HTML exports embed their old camera
poses. They do not update automatically. Regenerate into a fresh output directory
using the existing tracking command; it reads the active camera metadata. For
example, one episode with all three cameras and force decomposition:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/debug_drawer_forces_camera.py \
  --converted-root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --measurements-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_robot_measurements" \
  --interaction-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_tcp_joint_estimation" \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ee_tracking_registered_fridge_review_01" \
  --scene final_red_fridge_episode_000000 --all-cameras --decompose
```

Existing trained 3DGS/Gaussian outputs also retain their original world frame. Do
not pair those old coordinates with the newly registered cameras; regenerate them
or consistently re-express the complete model separately. This dataset operation
does not modify trained outputs or the production estimation pipeline.

## Force/EE visualizations regenerated on 2026-11-13

All 110 registered final-session datasets were rerun through the existing force/EE
visualizer, on all three cameras: 330 successful H.264 videos, 59,469 RGB-D frames,
and 110 completed interaction-window force/torque analyses, with zero failures.
Results are in:

```
/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_force_ee_registered_20261113
```

Open `index.html` for the videos, EE/TCP markers, resultant/tangential/perpendicular
force overlays and snapshots. `force_torque_summary.html` contains the saved-window
temporal mean of absolute tangential force percentage and torque plots. The same
0.02 m/N scale, 0.2 N display threshold, epsilon 1e-8 N, centered EE differences and
interaction windows were retained. Separate per-native-frame 3D torque PNGs were
not requested for this rerun.

`calibration_provenance.json` records the exact command, selected scenes, source
hashes and installed camera matrices/status. `calibration_usage_validation.json`
checks every video frame count from container metadata, every camera matrix/hash,
camera-force rotation and every saved TCP projection. All 110 scenes retain the
same robot-base EE/TCP measurements, forces, tangents and eta arrays as the previous
run; only their camera projection changes. This is verification of calibration use,
not independent certification of the applied camera registration.

The previous visualizations remain in `real_force_trajectory_ee_centered_20261112`.
The new root includes execution source snapshots and the validation helper;
`real_force_ee_registered_20261113.log` is beside it in the results parent folder.

The updater is standalone: a fresh native-to-standard reconversion must apply the
registration again as its final geometry step. Mask/point-cloud regeneration on
an already registered scene should use its active camera poses; do not manually
apply G again. The hash guard will flag regenerated files if the application command
is rerun; inspect that change rather than forcing another transform.

Tests cover both PLY encodings, normals/colours, complete masked/unmasked endpoints,
text/binary models, modern rig/frame models, YAML/database priors, exact restoration,
idempotence, edit guards and rollback after an injected publication failure:

```bash
cd /home/samuelemara/Joint3DGS/Sapien
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/samuelemara/miniconda3/envs/pycolmap_env/bin/python -B \
  -m unittest discover -s tests -p test_apply_robot_registration.py -v
```
