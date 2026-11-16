# Force, end-effector and calibrated TCP measurements

**Force sign:** recorded forces are object-on-robot contact reactions. The force
produced by the robot on the object is their negative in the same coordinate
frame: `F_robot_on_object = -F_recorded_reaction`. Existing data and plots retain
the recorded reaction sign. See [the contact force convention](contact_force_convention.md)
and `metadata/contact_force_convention.json` in each converted real scene.

**Early drawer mounting:** the three `white_drawer_20261021_161154_episode_00000[0-2]` recordings used a different force-sensor mounting (user confirmation, 2026-11-12). Their EE-to-TCP offset and sensor-axis rotation are currently uncalibrated. The registry marks `tcp_calibration_valid: false`; TCP generation, joint estimation, force plotting and HTML refresh refuse these recordings until their own transforms are established. The retained +34.87365 mm matrix is historical provenance, not a valid calibration for this mounting. Raw EE and wrench measurements remain available in Parquet and cached measurement archives.

`analyze_real_robot_measurements.py` reads the real recordings' Parquet files, applies the dataset's calibrated EE-to-TCP transform, and saves EE/TCP plots and numerical measurements for every episode. Run it with the existing `yolo26` environment (NumPy, SciPy, Polars and Matplotlib); it uses the CPU.

Process all 118 converted real scenes:

```bash
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/analyze_real_robot_measurements.py \
  --input "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_robot_measurements" \
  --overwrite
```

Add `--dry-run` to list sources without writing. Existing analysis files require `--overwrite`; only this script's output files are replaced. Original recordings, converted data and calibration are read-only. All output, including the Matplotlib cache, stays under `--output-root`.

Use one converted scene instead of the real root to inspect one episode. To process only the 110 final episodes, expand the scene names in the shell:

```bash
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/analyze_real_robot_measurements.py \
  --input "$REAL"/final_* \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/final_robot_measurements"
```

Native input is also supported, without converting RGB-D or creating masks:

```bash
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/analyze_real_robot_measurements.py \
  --input "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/assets/final_black_laptop" \
  --episodes 0 \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/laptop_robot_measurements"
```

`--input` accepts multiple converted scenes, native recording directories, a root containing those directories, or individual `episode_NNNNNN.parquet` files with their recording metadata. `--episodes` optionally selects episode indices. Robot measurements are exported once per episode, not duplicated for each RGB-D camera.

## Outputs

Each `<output-root>/<scene>/` contains:

- `measurements.png`: force XYZ and magnitude (N), torque XYZ and magnitude (N m), EE position (m), EE rotation vector (rad), linear/angular speed, and force versus displacement/speed.
- `ee_trajectory.png`: EE trajectory in base_link, with equal spatial axis scales and recorded force magnitude as color.
- `tcp_measurements.png`: force/torque alongside calibrated TCP position, orientation, speed, and force versus TCP displacement/speed.
- `tcp_trajectory.png`: calibrated TCP trajectory colored by recorded force, with the recorded EE trajectory shown dashed for comparison.
- `robot_tool.json`: exact tool assignment, full local transform, source identity, registry/metadata hashes and calibration-reference assumption used for these results.
- `measurements.npz`: recorded EE and derived TCP arrays and magnitudes/speeds, loadable with `numpy.load(..., allow_pickle=False)`. TCP keys are `tcp_position_m`, `tcp_rotvec_rad`, `tcp_linear_speed_m_s`, `tcp_angular_speed_rad_s`, and `displacement_from_first_tcp_m`.
- `measurements.csv`: the same data in columns; unavailable values are blank.
- `summary.json`: duration, actual recorded sampling intervals, duplicate timestamps, gaps, valid/missing counts, force/torque statistics, path length, source hashes and conventions.

The root contains `summary.csv` for comparing episodes, `tool_assignments.csv`, a snapshot `tool_registry.json`, and `run_summary.json` listing completed and failed episodes. A failed episode makes the command exit nonzero; other episodes continue. An unknown or conflicting tool assignment stops preflight before analysis outputs are written.

## Dataset-to-tool metadata

`configs/real_robot_tools.json` explicitly matches all 118 existing converted scenes to the tool used, recording name and native episode index. There is no default tool for unknown datasets. Use `--tool-metadata PATH` for a different explicitly reviewed registry. The script finds the repository registry automatically.

| Scenes | Tool | EE-local +Z translation |
|---|---|---:|
| `white_drawer_20261021_161154_episode_000000`, `000001`, `000002` | `bare_force_sensor` | 0.03487365 m |
| Other 115 currently registered real scenes | `pointed_tip` | 0.18761740 m |

Each existing converted scene also receives `metadata/robot_tool.json` with its explicit tool, full `T_ee_tcp`, source recording/episode and reference assumption. This addition records the user-confirmed assignment; original Parquet files, camera calibration and RGB-D data are preserved. The analyzer checks local scene metadata against the registry and does not rewrite input metadata while plotting. Native-recording input matches the registry by recording name and episode, so the three white-drawer exceptions also apply when reading their native Parquet files.

The complete rigid composition is:

```text
T_base_tcp = T_base_recorded_ee @ T_ee_tcp
p_base_tcp = p_base_ee + R_base_ee @ t_ee_tcp
R_base_tcp = R_base_ee @ R_ee_tcp
```

The force-sensor axes are aligned with the EE axes, as confirmed by the user: `R_ee_force_sensor = I`. Both tool transforms also have identity rotation. The listed Z offsets are the **total EE-to-TCP offsets** for the selected tool; they are not added together. The translation is rotated by each measured EE orientation; it is not added directly to base-frame Z. TCP speed is recomputed from the TCP positions, including motion caused by tool rotation. No table-height shift is applied.

The supplied contact calibration is relative to the bare contact reference. Applying it here assumes that reference coincides with the recorded EE reference. Absolute `link_6`-to-tip validity additionally requires the bare contact point to coincide with the `link_6` origin; the metadata preserves this limitation rather than claiming an independently verified `link_6` pose.

## Conventions

Pose fields come from `state_names` in `info.json`: `pos_x/y/z` and `rotvec_x/y/z` in `observation.state`. These are the measured EE pose, as confirmed for these recordings. TCP is derived from this pose and the selected tool calibration; it is not a second measured pose or a forward-kinematics reconstruction of `link_6`.

Wrench fields come from `wrench_names`: `Fx_N`, `Fy_N`, `Fz_N`, `Tx_Nm`, `Ty_Nm`, `Tz_Nm` in `observation.wrench`. The user confirms that these axes are EE-aligned, recorded in `wrench_frame` metadata. No additional sensor-to-EE rotation is required. Components are retained exactly as recorded; the script does not rotate them into base_link or shift the torque reference to the TCP. Axes alignment does not establish that the measurement origin is the TCP origin. Expressing the force in base_link would use `F_base = R_base_ee @ F_recorded`; force norms and their detected peaks are rotation-invariant. The existing scalar force-versus-speed plots remain associations, not mechanical-work estimates. No tare subtraction, gravity compensation or smoothing is applied.

All Parquet rows are retained, including frames not selected by RGB-D depth sampling. `frame_index` stays the native recording index. For converted scenes, `rgbd_frame_index` uses `metadata/real_conversion.json`; `-1` marks a native row with no converted RGB-D frame. Thus the robot sample count can exceed the RGB-D frame count.

Elapsed time uses the recorded `timestamp` column. Linear speed uses adjacent EE positions and positive recorded time differences. Angular speed uses the magnitude of the relative SO(3) rotation, so a rotation-vector wrap at pi does not create a false spike. Speeds are assigned to interval ends; the first sample, zero-duration intervals and invalid adjacent measurements remain NaN. Timestamps are not presented as synchronized hardware exposure times. Missing force data remain missing, never zero-filled.
