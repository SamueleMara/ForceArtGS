# Force direction along the TCP trajectory

The arrows show **object-on-robot contact reactions**, with the recorded sign.
The robot's force on the object is the negative of that vector in the same frame.
Existing plots and arrays are not negated; see [the contact force convention](contact_force_convention.md)
for the JSON sign rule and its effect on signed versus absolute tangential forces.

**Early drawer mounting:** the three `white_drawer_20261021_161154_episode_00000[0-2]` recordings used a different force-sensor mounting (user confirmation, 2026-11-12). Their EE-to-TCP offset and sensor-axis rotation are currently uncalibrated. The registry marks `tcp_calibration_valid: false`; TCP generation, joint estimation, force plotting and HTML refresh refuse these recordings until their own transforms are established. The retained +34.87365 mm matrix is historical provenance, not a valid calibration for this mounting. Raw EE and wrench measurements remain available in Parquet and cached measurement archives.

Run for **all 118 datasets**, using each scene's previously detected interaction interval:

```bash
/home/samuelemara/miniconda3/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/plot_tcp_force_directions.py \
  --measurements-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_robot_measurements" \
  --interaction-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_tcp_joint_estimation" \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_tcp_force_directions" \
  --frame-space native \
  --arrow-every 10
```

This uses the existing base Python, which already has NumPy, SciPy, Matplotlib and Plotly 6.3. It runs on CPU. Add `--overwrite` when regenerating existing output. Input archives, datasets and joint estimates are read-only.

To refresh **only the existing HTML plots** with slender arrows and smaller heads, preserving each scene's original frame selection, arrow spacing, force values, CSV/NumPy exports and PNG:

```bash
/home/samuelemara/miniconda3/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/plot_tcp_force_directions.py \
  --measurements-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_robot_measurements" \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_tcp_force_directions" \
  --html-only --overwrite --head-fraction 0.15
```

HTML arrows use thin line shafts with explicit small mesh tips. The head occupies 15% of the total arrow length; use `--head-fraction 0.1` for smaller tips. The initial orthographic camera is chosen to show the trajectory from the side. Tight axis ranges preserve equal spatial units without wasting space on a large empty cube. The view menu also offers XY, XZ and YZ projections. Force colors, endpoints, lengths and TCP roots are preserved. The dark trajectory and green/red start/end markers remain visible; the force-arrow legend entry toggles the arrows as a group.

`--arrow-every 10` places an arrow at every tenth native Parquet frame within the selected interval, beginning at its first available frame. Change it to control density. Use `--frame-space rgbd` instead to count converted RGB-D frames; only native samples with an exact RGB-D mapping can then be arrow origins. Trajectories still retain every native sample within the selected time interval. No force interpolation or temporal averaging is applied, so narrow spikes may fall between spaced arrows; use spacing 1 to inspect every eligible sample.

The interaction interval is taken exactly from each scene's `result.json`. Force/visual conflicts and visual-only fallback labels are retained in the plot title and manifest. This visualization does not turn a proposed interval into a verified contact interval. Scenes without a selected interval are explicitly skipped.

To show an entire episode, omit `--interaction-root`. To use a manual inclusive range, replace it with `--frame-range FIRST LAST`, interpreted in `--frame-space`. Endpoints must exist. Add `--scene SCENE_NAME` to select one or more episodes instead of the full batch.

## Arrow interpretation

- Each arrow's **tail is the calibrated TCP position at that same sample**.
- The measured force is expressed in the common plotting frame using `F_base = R_base_ee @ F_recorded`. The user-confirmed sensor-to-EE rotation is identity.
- The local TCP Z offset affects the arrow's position through the existing TCP trajectory. Translating a force vector's origin does not rotate or change that vector. Torque transport is not performed here.
- Default `--arrow-mode direction` uses equal-length arrows of 0.025 m for readability; arrow color shows measured force magnitude in N. Change `--arrow-length-m` to adjust the display length.
- For lengths proportional to magnitude, use `--arrow-mode magnitude --meters-per-N 0.01`: one N is drawn as one centimetre. The display scale is written on the figures.
- The original measured force sign is preserved; no action/reaction sign flip is assumed.
- `--min-force-N 0.05` suppresses arrows whose direction is dominated by near-zero readings. Missing positions/orientations/forces are not filled. CSV/NumPy files retain selected measurements and distinguish requested from actually drawn origins.

## Output

Each dataset folder contains:

- `force_directions.html`: offline interactive 3D plot; open it in a browser to rotate, zoom, and hover for frame IDs, timestamps, TCP coordinates and force components. It embeds Plotly and needs no network connection.
- `force_directions.png`: a 3D plot plus XY, XZ and YZ projections, all in metres with equal spatial scaling. Green/red dots mark the start/end; arrow colors show force magnitude.
- `force_vectors.csv` and `force_vectors.npz`: selected source/RGB-D frame IDs, positions, EE/base-frame forces, norms and arrow-origin flags.
- `manifest.json`: interval, spacing, scaling, coordinate/sign conventions and source hashes.

The root contains `summary.csv` and `summary.json` with successful, skipped and failed scenes. Each figure's colorbar is scaled to that interval's measured force range. Use numeric magnitudes when comparing different episodes.

Use `--no-html` for static outputs in an environment without Plotly; the existing `yolo26` Python supports that mode. The requested all-dataset command above creates both interactive and static plots. Absolute TCP coordinates retain the tool calibration's recorded-EE reference assumption.

## Early drawer +90° Z mounting trial

The proposed `R_EE_sensor = Rz(+90°) = Rz(-270°)` maps `(Fx,Fy,Fz)` to `(-Fy,Fx,Fz)` before applying the recorded EE-to-base rotation. This **does not change the axial Fz component**. The dedicated comparison script creates an identity reference and a rotated candidate in separate folders, retaining the original frame intervals, nominal TCP positions and plotting settings. Both are labeled unvalidated. It does not alter the production mounting metadata or lift the calibration guards; the older EE-to-contact transform is still unknown.

```bash
/home/samuelemara/miniconda3/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/compare_force_mounting_rotations.py \
  --measurements-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_robot_measurements" \
  --interaction-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_tcp_joint_estimation" \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/white_drawer_force_rotation_trials_20261112" \
  --scene white_drawer_20261021_161154_episode_000000 \
    white_drawer_20261021_161154_episode_000001 \
    white_drawer_20261021_161154_episode_000002
```

Open the resulting `index.html` for side-by-side comparisons; click each preview for the interactive plot. `REPORT.md` gives direction-change statistics for forces at least 0.5 N. Each candidate includes raw sensor force, candidate EE/base forces, the trial matrix and provenance. The script requires a fresh output directory; choose another root for a rerun. Episode 2's existing combined interval still spans different drawers and remains unsuitable for a single joint fit.

## Camera-002 force debug videos for the drawers

`debug_drawer_forces_camera.py` overlays forces on each exactly mapped RGB frame from cam_002. It uses the rectified intrinsics and inverse camera-to-base transform, projecting the nominal TCP and the endpoint `p_tcp + display_scale * F_base`. Camera translation applies to points; force components use only camera rotation. Both force-mounting assumptions are shown side by side: identity on the left (green force arrow), +90° Z on the right (orange).

```bash
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/debug_drawer_forces_camera.py \
  --converted-root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --measurements-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_robot_measurements" \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/white_drawer_cam002_force_debug_20261112" \
  --camera cam_002 --force-scale-m-per-n 0.02 --min-force-n 0.2 \
  --scene white_drawer_20261021_161154_episode_000000 \
    white_drawer_20261021_161154_episode_000001 \
    white_drawer_20261021_161154_episode_000002
```

Requires the already installed ffmpeg, OpenCV, NumPy and SciPy; runs on CPU with two encoding threads. Open `index.html` for all three videos and peak-force snapshots. Each scene also contains additional sampled snapshots, `projected_forces.csv` with exact frame IDs, sensor/EE/base/camera force components and projected endpoints, and `manifest.json` with calibration/projection provenance. The output directory must be fresh; use a different root to repeat a run.

Yellow cross = **nominal** TCP origin; cyan cross = recorded EE; blue arrow = recorded EE-local Z reference (4 cm), not independent drawer-axis ground truth. Force display length is proportional to magnitude using 0.02 m/N; it is not normalized. The default 0.2 N cutoff exceeds the observed quiet-start 95th-percentile norms in these recordings. Small image projections, clipping and out-of-view origins are labeled. No force sign flip, tare subtraction or TCP recalibration is applied. A prismatic constraint can produce lateral contact reactions, and a 2D projected arrow angle is not the 3D force/axis angle. +90° around Z preserves Fz and the XY force norm, so this change alone cannot alter the axial/transverse force ratio.

The videos cover the complete episodes, including repositioning and both drawer interactions in episode 2. They use the exact existing RGB-D/native mapping, without pose/force interpolation. Hardware synchronization remains unverified. Native force samples without a corresponding RGB-D frame are omitted; the manifest compares the full native and displayed peak forces so missed narrow spikes are explicit. The old contact-point transform and the proposed mounting rotation remain unvalidated; production calibration and original data are untouched.

## Tangential/perpendicular force decomposition: every dataset, every camera

The same debug code now supports `--decompose --all-scenes --all-cameras`. Each native Parquet frame uses a local TCP trajectory tangent `t`, then decomposes the base-frame force as:

```text
Ft_signed = dot(F_base, t)
Ft_vector = Ft_signed * t
Fperp_vector = F_base - Ft_vector
Fperp_norm = norm(Fperp_vector)
```

Positive `Ft_signed` follows the instantaneous direction of travel; negative opposes it. The perpendicular component contains all directions orthogonal to the trajectory, not a known surface-normal contact force. The recorded force sign is preserved. The centered observed EE direction follows curved trajectories and reversals; there is no global line/circle axis assumption. Observed EE motion does not establish contact, no-slip motion or an object joint axis.

The tangent now follows the user's exact observed **EE** formula: `t[i] = (p_EE[i+1] - p_EE[i-1]) / norm(p_EE[i+1] - p_EE[i-1])`. It uses the recorded EE positions in base_link, without a local fit, smoothing, or speed threshold. The first/last samples, zero centered displacement and missing/invalid adjacent samples are undefined; there is no one-sided replacement. A centered EE velocity is saved as `(p[i+1]-p[i-1])/(time[i+1]-time[i-1])` for the auxiliary speed/power diagnostics.

For the corrected force already expressed in robot-base axes, `F_parallel = dot(F_base,t)` and `F_perp = norm(F_base - F_parallel*t)`. The signed projection is preserved. `eta = abs(F_parallel)/(norm(F_base)+epsilon)` is exported in dimensionless units; the percentage is `100*eta`. Default `epsilon=1e-8 N` is configurable with `--force-epsilon-n`. Zero force yields eta=0 when the tangent is defined. No new force bias, tare or sign correction is invented.

```bash
OPENBLAS_NUM_THREADS=1 \
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/debug_drawer_forces_camera.py \
  --converted-root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --measurements-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_robot_measurements" \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_force_trajectory_decomposition_20261112" \
  --interaction-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_tcp_joint_estimation" \
  --all-scenes --all-cameras --decompose --workers 2 \
  --force-epsilon-n 1e-8 \
  --force-scale-m-per-n 0.02 --min-force-n 0.2
```

The discovered inputs currently contain **118 scenes, 354 cameras, 47,382 native measurements and 71,154 camera frames**. Two datasets can render in parallel, each with two CPU encoding threads; GPU is not required. Every scene is retained, including all final episodes and the older recordings. Original data, masks, calibration and cached measurements are read-only. All new files, including Matplotlib cache, stay inside the new output root. An existing output root is refused; choose a fresh root to repeat the run.

Per scene:

- `force_decomposition.npz` and `force_decomposition.csv`: **every native sample**, including unmapped RGB-D samples; raw sensor force, base-frame force, centered EE tangent, EE velocity/speed, signed tangential force, both 3D component vectors, perpendicular magnitude, angle to unoriented tangent, `F dot EE velocity` in W, validity flags and fit-support counts. No force threshold filters numerical samples. The NPZ has shared native-frame arrays and a leading mounting-case dimension for force/decomposition arrays.
- `force_decomposition.png`: signed tangential/perpendicular force, angle and speed plots over the complete episode, with undefined-tangent periods marked.
- `cam_000/`, `cam_001/`, `cam_002/`: an MP4, sampled JPEGs, per-camera `projected_forces.csv` and `manifest.json`. Resultant = green/orange; tangential = cyan; perpendicular = magenta; centered EE direction reference = blue. Force vectors use the same m/N scale, not normalized lengths. Each vector is projected separately in 3D through that camera's calibration; the projected image arrows need not add in pixel coordinates because of perspective. The 0.2 N display threshold applies to arrows only.

The three early drawers retain two explicitly unvalidated mounting cases, identity and +90° Z, and nominal TCP origins. Other scenes use their user-confirmed EE-aligned force axes and supplied TCP transform. The source calibration validity is recorded in each camera manifest. Numerical outputs are shared across cameras, avoiding duplicate native measurements.

The root `index.html` groups all camera videos and numerical links by scene. `summary.json` is updated as dataset jobs finish; `run.json` records planned/completed camera runs and failures. `decomposition_summary.csv` contains one row per scene/mounting case, with all-camera-independent force statistics computed over moving samples above the display threshold. Statistics are over whole episodes, including free-space motion and repositioning; they are not contact-only joint metrics. Every valid sample is checked for `F = Ft + Fperp` and `dot(Fperp,t) = 0`. Failed camera jobs are reported and other cameras/datasets continue.

## Interaction-window temporal mean and per-frame 3D torque images

The requested summary is now exactly the **temporal mean of the instantaneous absolute tangential percentage within the saved interaction window**:

```text
p(t) = 100 * abs(F_base(t) dot unit_tangent(t)) / (norm(F_base(t)) + epsilon)
mean = integral_a^b p(t) dt / (b - a)
```

This replaces the earlier whole-episode frame mean, median and force-magnitude-weighted ratio. Integration uses trapezoids at actual native Parquet timestamps, clipped to `[a,b]`. There is no force weighting and no default 0.2 N summary cutoff: every finite force is included, including zero (eta=0). The direction is recomputed from centered raw EE differences. Endpoint, zero-displacement and invalid-neighbor tangents are undefined; zero force has eta=0 wherever the tangent exists. They are never filled with zero or bridged across missing samples/gaps. The requested **full-window mean** is reported only when valid-time coverage is 100%; otherwise it is blank with `INCOMPLETE_VALID_TIME`, and a separately labeled `valid_time_temporal_mean_pct` and coverage are provided. No interval falls back to a whole episode.

Bounds come from `--interaction-root/<scene>/result.json`, field `interval.selected_window_s`. Their original source and force/visual conflict status are retained. These saved bounds are not newly asserted as contact GT. The earlier 88/30 coverage count used the superseded smoothed TCP tangent; coverage is recomputed using centered EE differences.

Run this command to update the existing results for all datasets and all camera bases, and generate the torque images:

```bash
OPENBLAS_NUM_THREADS=1 \
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/analyze_force_share_torques.py \
  --analysis-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_force_trajectory_decomposition_20261112" \
  --measurements-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_robot_measurements" \
  --interaction-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/real_tcp_joint_estimation" \
  --min-force-n 0 --force-epsilon-n 1e-8 \
  --torque-3d-images
```

This command is prepared for the user to launch; the full image batch has not been run by the assistant. It writes one separate **torque-only 3D PNG per native frame**: 47,382 PNGs for the current datasets. Each PNG contains a base-axis panel and **all three camera-axis panels**. Old-drawer identity/+90° mounting hypotheses appear as separate colored vectors. Images are stored as `<scene>/torque_3d_frames/frame_<native_index:06d>.png`; open that folder's `index.html` for a frame slider, including native samples without RGB-D images. Axes are in **N m**, equal in all directions, and their bounds are fixed across all frames of a scene. Vector lengths preserve the recorded moment magnitudes. Every native frame is exported without subsampling; progress is printed every 250 images.

The displayed zero is the origin in torque-component space, not a claimed physical TCP position. The recorded physical moment origin remains unspecified; no `r cross F` transport or TCP shift is performed. Proper sensor-to-EE, EE-to-base and base-to-camera rotations are applied to the moments. Norms remain invariant. This is a 3D moment vector image, separate from force overlays and torque-vs-time plots.

The command refreshes only its own companion analysis files, percentage summary and browser index. Source measurements, force-decomposition files and videos remain untouched. Numerical outputs retain all native-frame percentages and torques; `interaction_sample` marks samples inside the saved bounds. Root `force_torque_summary.html`, `.csv`, `.json` now contain the requested window time mean, exact bounds, coverage and validity status. Per-scene JSON preserves interval provenance and mounting validity. Three old drawers remain unverified mounting hypotheses; no corrected physical calibration is claimed.

For a new complete debug run, pass `--interaction-root PATH` with `--decompose`, and add `--torque-3d-images` to export the separate images. The force display threshold remains independent from the zero-default summary threshold.


## Regenerate force overlays with the exact EE formula

The refresh command above **recomputes the new EE projections from raw EE samples**, even when the input force archive contains an older smoothed TCP tangent. It writes the new tangent, signed F_parallel, F_perp, eta and percentages into `wrench_analysis.csv`/`.npz`; the old force archives and videos remain historical. To regenerate all force overlays and force-decomposition files as well, run a fresh complete pipeline:

```bash
FORCE_RESULTS="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"
OPENBLAS_NUM_THREADS=1 \
/home/samuelemara/miniconda3/envs/yolo26/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/debug_drawer_forces_camera.py \
  --converted-root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --measurements-root "$FORCE_RESULTS/real_robot_measurements" \
  --output-root "$FORCE_RESULTS/real_force_trajectory_ee_centered_20261112" \
  --interaction-root "$FORCE_RESULTS/real_tcp_joint_estimation" \
  --all-scenes --all-cameras --decompose --torque-3d-images \
  --force-epsilon-n 1e-8 --workers 2
```

The output root must be fresh. No full batch was launched while updating this code. All 118 scenes and all cameras are retained, with the old drawers' mounting hypotheses still explicit. The interaction-window metric is the temporal mean of `100*eta`; its numerator is the instantaneous absolute signed projection, not an absolute mean force or a force-weighted ratio. Undefined tangent time is reported by coverage. Endpoints use actual adjacent native samples, including positions just outside an interaction bound when available.
