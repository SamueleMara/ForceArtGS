# Robot-TCP-guided tracking, moving-part prompts and sliding-contact constraints

Three **independent, opt-in** improvements to the moving-part trajectories that FreeArtGS
hands to the existing joint estimator, plus one shared alignment adapter. All are OFF by
default; with every flag off the tracking, ICP, `prepare_multiview`, `twopart-blend`,
`estimate_joint.py` and `splatfacto-art` paths are unchanged.

| Phase | What it does | Changes object tracking? |
|---|---|---|
| Adapter | Aligns measured, calibrated TCP + explicit contact state to each FreeArt camera frame | No (artifact only) |
| A | Compares the raw AllTracker track of the TCP pixel with its measured projection; optional `adaptive_requery` window splitting | Only in `adaptive_requery`, and only by re-running AllTracker on the real RGB images |
| B | Contact-guided *object-surface* positive prompts (+ depth-confirmed robot negatives) → moving-part seed loss | Yes, segmentation weights (seed BCE) |
| C | Sliding contact-to-moving-surface distance in the transform optimiser | Yes, moving transform |

Physical assumptions enforced throughout: the TCP is a **robot point**, never an object
material point; contact does **not** imply a fixed material contact point
(`p_TCP(t) = T(t) x0` is never imposed); robot pixels never enter object masks, object
point clouds or object correspondences; tracking labels 0/1 are arbitrary until
`prepare_multiview` reindexes them; AllTracker is frozen and never trained.

Not in scope and not changed: joint classification, force-ratio classifier, dynamics,
`preprocessing/estimate_joint.py`, `splatfacto-art`, `nerfstudio-local`, AllTracker.

## 1. Data, frames and conventions

Source real scene (read only) and per-camera FreeArt data:

```text
SCENE/metadata/{real_conversion.json, cameras.json, robot_tool.json, raw/episode_*.parquet}
SCENE/metadata/robot_poses/poses_rgbd.npz          # optional, from export_real_robot_poses.py
CAMERA_DIR/metadata/freeart_conversion.json        # frame_map, from convert_standard_to_freeart.py
CAMERA_DIR/{rgb,depth,masks}/00000.*, cam_K.txt
```

**Frame join** (no FPS arithmetic):
`freeart_conversion.frame_map[i].source_stem` (`frame_000123`) → real output frame 123 →
`real_conversion.source_frames` / `poses_rgbd.source_frame_index` → Parquet `frame_index`.
The adapter checks one-to-one order, unique IDs, strictly increasing source chronology and
agreement between `real_conversion` and the pose archive, so `frame_stride`,
`frame_intervals`, `max_frames` and non-zero raw indices are handled. Example (verified on
`final_black_box_episode_000000`, interval 100:139, stride 2): local frames 0..19 → output
100..138 → source 200..276.

**TCP pose**: `poses_rgbd.npz:tcp_transform` is used as is. If no archive exists (true for
all real scenes as of 2026-11-13) and `--parquet-python` is given, the adapter calls
`export_real_robot_poses.build_arrays()` with the scene's `robot_tool.json` (or a
`--calibration-map`), applying `T_base_tcp = T_base_ee @ T_ee_tcp` exactly once and
requiring the recorded pose frame to be explicitly `ee`.

**Cameras**: `cameras.json:pose_matrix_4x4` is `T_base_camera` (OpenCV optical
camera → `base_link`); `p_cam = inv(T_base_camera) @ p_base`. `cam_K.txt` must equal
`cameras.json` and `real_conversion.rectified_K`.

**Pixels**: `(u = x, v = y)`, integer pixel centres (OpenCV/K convention); NumPy indexing
is `[y, x]`. Tracking resolution uses the resized intrinsics: `u_track = K_track @ p_cam`
(`ImageLoader.resize` scales K by the same factor).

**Pingpong and duplicates**: tracking may run on `[0..N-1] + [N-2..0]`. Because
`bundle_preprocess.split_seg_maps_to_masks` builds the final masks from the *reversed*
half, reversed copies are mapped explicitly to their original frame (and image equality is
verified) and stay usable. True duplicates (the turning frame with `include_turn_frame`, and
`duplicate_last_frame` for 2-frame windows) are never used for any prior.

**Contact state** has three values per raw `source_frame_index`: `contact` (confirmed),
`absent` (confirmed absence) and `uncertain`. Only `contact` drives Phases B/C; `uncertain` is
never turned into absence or contact. The normal source is the automatic force exporter
(`samuele/export_force_contact_annotations.py`, see below); explicit files are also accepted:

```json
{"index": "source_frame_index", "mode": "per_frame", "source_frame_index": [...],
 "contact_state": ["absent", "contact", "uncertain", ...], "provenance": "..."}
{"index": "source_frame_index", "mode": "confirmed_intervals", "intervals": [[200, 280]],
 "outside_intervals": "uncertain", "provenance": "operator confirmation, 2026-11-13"}
```

Legacy per-frame boolean `contact_active` files (False = confirmed absence) and CSVs with
`source_frame_index` and `contact_state` or `contact_active` columns still work. A missing
source gives `uncertain` everywhere; per-frame sources that do not cover a frame are an error.
**Phase B or C enabled without any confirmed contact fails** (`ValueError`), instead of
reporting a successful no-op; Phase A does not need contact.

**Force-based contact annotations** (`export_force_contact_annotations.py`, Python with
polars/pyarrow, e.g. `yolo26`). Inputs: the episode Parquet (`observation.wrench` =
`Fx_N, Fy_N, Fz_N, …`, checked against `info.json:wrench_names`), the EE pose
(`observation.state`) and `robot_tool.json:wrench_frame` (axes must be declared `ee`). The
sensor bias is re-estimated as the mean force in a contact-free baseline (first
`baseline_duration_s`, default 0.5 s, or `--baseline-frames`); per-axis noise is
`1.4826·MAD`. `z = ||(F − tare)/σ||`: `z ≥ contact_z` (default 6, the same six-sigma rule
as the existing force-event detector) inside a run without confirmed absence whose
above-threshold samples span `≥ min_contact_duration_s` (default 0.2 s) is `contact`;
`z ≤ absent_z` (default 3) is `absent`; anything else, short spikes and non-finite samples
are `uncertain`. Tool gravity changes with EE orientation are subtracted when
`--tool-mass-kg` is given; without it the run refuses EE rotations above
`--max-uncompensated-rotation-deg` (default 1°). Manual corrections are optional
(`--corrections`, JSON with `provenance` and per-range `state` + `reason`). Outputs:
`contact_annotations.{json,csv}` (per native sample, plus the exact RGB-D alignment through
`real_conversion.source_frames`), `contact_summary.json` (thresholds, tare, noise, gravity
handling, counts) and `force_contact.png`. Force decides *when* the tool interacts; which
part moves is not inferred, and no joint-type or dynamics estimate is made.

**Tool contact geometry** (optional JSON, `tool_contact_geometry`):

```json
{"tool_geometry_id": "pointed_tip", "contact_point_local_tool_m": [0, 0, 0],
 "contact_point_calibrated": true, "contact_uncertainty_m": 0.004,
 "shape": {"type": "point"}, "secondary_feature_local_tool_m": null, "provenance": "..."}
```

Without it the TCP origin is an *approximate* anchor (`contact_point_calibrated: false`);
set `contact_sigma_m` accordingly larger. No zero-distance contact is ever enforced.
`secondary_feature_local_tool_m` (a calibrated tool-fixed point, e.g. `[0, 0, -0.10]` on the
shaft) can be tracked by Phase A instead of the TCP (`tcp_track_point: secondary_feature`)
when the tip itself is not visible.

## 2. Configuration (`tracking:` section, `utils/pipeline_config.py`)

| Key | Default | Meaning |
|---|---|---|
| `tcp_supervision_enabled` | `false` | Phase A |
| `moving_prompt_enabled` | `false` | Phase B |
| `contact_surface_enabled` | `false` | Phase C |
| `contact_multiview_refine_enabled` | `false` | Optional post-ICP stage; **not implemented**, raises if `true` |
| `tcp_contact_artifact` | `null` | default `<dataset>/metadata/tcp_contact` |
| `moving_prompt_artifact` | `null` | default `<dataset>/metadata/moving_prompts/moving_seed_maps.npz` |
| `tool_contact_geometry` | `null` | read by the adapter (driver) |
| `tcp_track_correction_mode` | `diagnostic_only` | or `adaptive_requery` |
| `tcp_track_point` | `tcp` | or `secondary_feature` (calibrated tool-fixed point; needs it in the tool geometry) |
| `tcp_visibility_depth_tol_m` | `null` | **required for A** |
| `tcp_sigma_px` | `null` | optional robust score `E_tcp` |
| `tcp_requery_error_px` | `null` | **required for `adaptive_requery`** |
| `tcp_requery_max_retries` | `2` | bounded retry count |
| `tcp_requery_persistence_frames` | `1` | consecutive valid failures before a split |
| `moving_prompt_weight` | `0.0` | **must be > 0 for B** |
| `moving_prompt_min_seeds` | `1` | seed tracks needed to resolve the branch |
| `contact_surface_weight` | `0.0` | **must be > 0 for C** |
| `contact_sigma_m`, `contact_robust_delta_m`, `contact_support_max_distance_m` | `null` | **required for C** |

No physical tolerance or weight has an invented default; an enabled phase with a missing
value fails with a clear error. Copy [`configs/tcp_contact_tracking.example.yaml`](../configs/tcp_contact_tracking.example.yaml).

## 3. Commands

All CPU commands below only write under the FreeArt camera folder (never the scene).

```bash
PY=/home/samuelemara/miniconda3/envs/freeartgs/bin/python
SCENE="<real scene>"; CAM="<converted FreeArt camera folder>"

# Shared adapter (CPU). --dry-run validates and writes nothing.
$PY -m samuele.tcp_contact_evidence --scene "$SCENE" --freeart-camera-data "$CAM" \
    --contact-file contact.json [--tool-geometry tool.json] \
    [--parquet-python /home/samuelemara/miniconda3/envs/yolo26/bin/python] \
    [--overlays 8] [--dry-run]
# -> $CAM/metadata/tcp_contact/{tcp_contact.npz, manifest.json, overlays/}

# Contact annotations from the recorded wrench (CPU, needs polars), written outside the scene
/home/samuelemara/miniconda3/envs/yolo26/bin/python samuele/export_force_contact_annotations.py \
    --scene "$SCENE" --output <dir> [--corrections fixes.json] [--tool-mass-kg <kg>]
# -> <dir>/{contact_annotations.json, contact_annotations.csv, contact_summary.json, force_contact.png}

# Multi-camera TCP projection audit (CPU, needs polars + OpenCV)
/home/samuelemara/miniconda3/envs/yolo26/bin/python samuele/audit_tcp_projection.py --scene "$SCENE" \
    --output <dir> --contact-annotations <dir>/contact_annotations.json \
    [--alt-cameras before=<registration backup>/metadata/cameras.json] [--visual-labels labels.json]
# -> tcp_projection_audit.json, axis_residual_profiles.png, crops_<calibration>_<camera>.jpg

# Phase B prompts + seed artifact (CPU)
$PY -m samuele.contact_moving_part_prompts --freeart-camera-data "$CAM" \
    --search-radius-m <m> --depth-margin-m <m> [--max-positive 5] \
    [--tool-exclusion-radius-m <m> --tool-exclusion-start-m <m>]
# Optional SAM2 proposals (GPU, grounded-sam2 env), then seeds from proposals
<sam2-python> samuele/contact_moving_part_sam2.py --freeart-camera-data "$CAM" --model-root <sam2-repo>
$PY -m samuele.contact_moving_part_prompts --freeart-camera-data "$CAM" --search-radius-m <m> \
    --depth-margin-m <m> --seed-source sam2 \
    --sam2-proposals "$CAM/metadata/contact_guidance/sam2_moving_proposals"
```

Artifacts must exist in the converted input **before** `--preprocess_only`, which copies
`metadata/` into the tracked dataset. Tracking then reads them when the flags are set in the
per-camera config (`bash pipelines/freeartgs.sh --config cfg.yaml --tracking_only`).

End-to-end multiview driver (builds artifacts per camera, merges `tracking:` overrides,
suffixes job names with the override file stem):

```bash
$PY pipelines/train_local_datasets.py --datasets "<real root>" --object <scene> \
    --cameras cam_000 cam_001 cam_002 --frame-intervals 100:180 --frame-stride 2 \
    --outputs-root "<outputs>" --tracking-overrides phaseA_only.yaml [--dry-run]
```

In the override file, `tcp_contact_evidence.contact_file: force` derives contact from the
wrench at preparation time (with `parquet_python` pointing to a polars environment; optional
`force_contact:` section with `contact_z`, `absent_z`, `min_contact_duration_s`,
`baseline_duration_s`, `baseline_frames`, `tool_mass_kg`, `corrections`).

Controlled ablation up to ICP and `prepare_multiview` (no Gaussian training). Every
configuration shares one conversion, one artifact build and one preprocessing per camera;
only tracking (and optionally Phase B prompts) differ:

```bash
$PY samuele/run_tcp_contact_ablation.py --scene "$SCENE" --cameras cam_000 cam_002 \
    --frame-intervals 70:129 --frame-stride 2 --work-root <work> --artifacts artifacts.yaml \
    --config baseline=none --config phaseA=phaseA.yaml --config phaseB=phaseB.yaml ... \
    [--stages track icp multiview]
$PY samuele/evaluate_tcp_contact_ablation.py --work-root <work> --configs baseline phaseA ... \
    --cameras cam_000 cam_002 --reference-regions reference_regions.json --output <eval>
```

## 4. Phase details and outputs

### Shared adapter — `samuele/tcp_contact_evidence.py`
`load_aligned_tcp`, `project_tcp`, `map_freeart_frames`, `load_contact_state`. NPZ arrays
(no pickle): `local/output/source_frame_index`, `timestamp_s`, `T_base_ee`, `T_base_tcp`,
`p_tcp_{base,cam}_m`, `uv_tcp_fullres_xy`, `valid_projection`,
`p_contact_{base,cam}_m`, `uv_contact_fullres_xy`, `valid_contact_projection`,
`contact_active`, `contact_confidence`, `K_fullres`, `T_base_camera`, `image_size_wh`.
The manifest records hashes of every input, calibration, frame map and contact provenance.

### Phase A — `tracking/tcp_track_supervision.py`, `estimotion_2part/tracking_prior_hooks.py::run_phase_a`
Inserted in `track_2part.process_single_window` right after `get_track_maps` and before
`get_4d_traj_maps`. The raw dense map is bilinearly sampled at the window-frame-0 projected
TCP (AllTracker `query_frame` must be 0). A frame is a valid comparison only if the TCP is
physically visible — observed depth within `tcp_visibility_depth_tol_m` of the TCP depth
and the pixel is not object foreground — **and** AllTracker visibility exceeds
`visibility_threshold`. An invisible seed makes the whole window `unavailable`; occlusion is
never reported as tracker error.

`adaptive_requery`: the first persistent visible failure (`> tcp_requery_error_px`) proposes
a shorter window ending one frame earlier; AllTracker is re-run on that window's RGB; the
split is accepted only if the requeried window has at least as many valid object tracks over
the same span. Bounded by `tcp_requery_max_retries`; windows of 2 frames are never split.
The next window starts at the last accepted frame (existing overlap logic).

Outputs: `tracking/window_XXXXX/tcp_track/{tcp_track.csv,npz,json, overlays/ (debug)}`
with `tcp_pred_uv`, `tcp_gt_uv`, `visible`, `visibility_status`, `valid_comparison`,
`residual_px`, `source_frame_index`, `window_frame_index`, split attempts
(`split_reason`, spans, TCP error before/after, object track counts); aggregate
`tracking/tcp_track_summary.json` (median/p95/max, visible samples, status).

### Phase B — prompts, SAM2 adapter, seed loss
`samuele/contact_moving_part_prompts.py::select_contact_prompts`: positives are object-mask
pixels with valid depth whose 3D point lies within `search_radius_m` of the contact point and
not in front of it by more than `depth_margin_m` (nearer surfaces are occluders). Object masks
can contain a dark tool: with `tool_exclusion_radius_m`/`tool_exclusion_start_m`, mask pixels
whose 3D point lies in a cylinder around the measured contact→EE axis are rejected (an object
cannot occupy the tool's volume); a negative start also removes the uncertain tip region, so
only pixels clearly beside the tool remain. Counts are reported per frame. Robot
negatives are projected samples of the measured contact→EE segment that are outside the
object mask and whose observed depth matches the tool depth. The TCP pixel itself is never
a positive. Hidden contact → `no_valid_positive_prompt` with a reason; nothing is synthesized.

Artifacts under `CAM/metadata/`: `contact_guidance/contact_prompts.json` (keyed by
`(camera, source_frame_index)`), `contact_guidance/proposed_moving_positive/cam_NNN/frame_*.png`,
`contact_guidance/prompt_report.json`, optional `contact_guidance/sam2_moving_proposals/…`
(clipped to the object foreground; SAM2 unavailable → `not_run`, never all-foreground), and
`moving_prompts/moving_seed_maps.npz` (per FreeArt local frame).

In tracking: `initialize_part_segmentation(seed_map=…)` only consolidates seed pixels into the
K-means cluster holding most of them. The moving **branch** (T1 or T2) is resolved per
optimisation from motion (`window_priors.resolve_seed_branch`): the branch must both fit the
seed tracks better *and* move them more; otherwise the window is `skipped`. Only then is
`L_seed = BCE(p_m, 1)` added (stage 2 of `solve_window_transformations` and in
`joint_optimize_segmentation_and_transforms`), with `p_m = sigmoid(w)` for T2 and
`1 - sigmoid(w)` for T1. Branch identity is flipped whenever `check_segmentation_consistency`
or the joint first-frame alignment swaps T1/T2. Reports: `window_XXXXX/priors/priors.json`.

### Phase C — `estimotion_2part/contact_surface_loss.py`, `window_priors.ContactTerm`
`L_contact = mean_t Huber(min_x ||p_contact(t) - T_m(t) x|| / sigma)` over confirmed and
supported contact times; the `min` is recomputed at every `t` (sliding contact), only the
selected moving transform is used (never the T1/T2 blend). Canonical support = window
frame-0 tracked RGB-D points (camera metres, same frame as `source_points`) hard-assigned to
the moving branch; frozen during inner updates and rebuilt at the existing re-initialisation
epochs. The branch is inherited from Phase B or resolved from motion and cross-checked
against the contact geometry (ambiguous → `skipped`). Contact samples whose initial distance
exceeds `contact_support_max_distance_m` are rejected as unsupported. Added only to the
transform-unfrozen stage and the window-global joint fit; joint rows without tracked pairs
get no contact term. Reports `L_visual`, `L_contact` (unweighted) and
`L_contact_weighted_final` (actual contribution), `contact_weight`, `N_contact`,
`N_contact_final`, `moving_branch`, `sigma_contact_m`, distances before/after, rejected
counts, distinct canonical contact samples (sliding diagnostic). The canonical surface is
rebuilt only from observed frame-0 points and segmentation weights, never from a candidate
transform. With a point-sampled surface the only tangential effect is a ripple bounded by the
sample spacing; Adam can amplify such tiny gradients along sliding directions when nothing
else constrains them (in the pipeline the visual term does).

## 5. Status vocabulary

Reports use `available`, `skipped`, `not_run`, `failed`, `passed`. Zero valid samples are
always `skipped`/`unavailable`, never a zero error.

## 6. Tests

```bash
PY=/home/samuelemara/miniconda3/envs/freeartgs/bin/python
$PY -m unittest tests.test_tcp_contact_evidence -v     # alignment, projection, contact sources
$PY -m unittest tests.test_tcp_track_supervision -v    # Phase A
$PY -m unittest tests.test_contact_moving_prompts -v   # Phase B (uses the solver; GPU if present)
$PY -m unittest tests.test_contact_surface_loss -v     # Phase C
$PY -m unittest tests.test_tcp_tracking_pipeline -v    # config + real slide_window_tracking loop
$PY -m unittest tests.test_contact_phase_c_validation -v  # synthetic revolute/prismatic sliding contact
$PY -m unittest tests.test_force_contact_annotations -v   # force exporter, 3-state contract, B/C failure
# Whole suite (run from tests/ so sibling helpers resolve to this repository):
cd tests && $PY -m unittest discover
```

Modules needing polars/pyarrow (`test_real_rgbd_audit`, `test_real_robot_measurement_analysis`,
`test_real_tcp_joint_estimation`, `test_rgbd_identifiability_followup`) run in `yolo26`;
`test_apply_robot_registration` needs pycolmap (base env). In `freeartgs` they fail at import
(environment, not regressions). `test_dataset_loading` has an unrelated nerfstudio
CPU/GPU device error that predates this work.

## 7. Known limitations

* Contact comes from the recorded wrench (three states); it says *when* the tool interacts,
  not where on the object. A sustained static load after the robot stops is labelled contact
  if it exceeds the threshold — check images before trusting such tails.
* `T_ee_tcp` gives the nominal tip; the physical contact point is uncalibrated
  (`contact_point_calibrated: false`), so `contact_sigma_m` must absorb that offset.
* Robot-base registration of the `final_*` scenes is `OPERATOR_APPLIED_ROBOT_REGISTRATION_UNVERIFIED`
  (see section 9 for the audit of `final_black_box_episode_000000`).
* Object masks can contain a dark tool; the tool then joins the moving part in tracking.
  Phase B can exclude it from prompts (tool volume), but tracking itself is unchanged.
* Phase A visibility needs the thin tip to appear in depth; tips missing from depth are
  `tip_not_observed_in_depth`, so many real frames may be unavailable.
* `adaptive_requery` judges improvement by valid object-track count over the same span; it
  does not prove better object motion. Demonstrate gains with ablations (section 8).
* Seed and contact priors cannot override strong contradicting visual evidence at modest
  weights (by design); they decide genuinely ambiguous tracks.
* `pass_segmentation_map` is unchanged; seeds act through the per-window optimisers.
* The optional post-ICP multiview refinement (`refine_multiview_moving_poses`) is not
  implemented; enabling it raises `NotImplementedError`.

## 8. Ablations to run

Baseline, A only, B only, C only, A+B+C (and A+B, A+C, B+C if possible) on the same scene,
cameras and frame subset; one override file per configuration (job names are suffixed with
its stem). Report Phase A median/p95/max and visible samples; Phase B valid prompts, robot
contamination, branch decisions; Phase C contact distance median/p95, `N_contact`, distinct
contact samples; downstream `global_poses_{0,1}`, `pose_valid`, `prepare_multiview` part
assignments, and joint-axis error only where GT exists. Do not claim reconstruction gains
from Phase A `diagnostic_only`.

## 9. Validation on `final_black_box_episode_000000` (2026-11-14)

Clip: RGB-D frames 70–128 (stride 2, 30 frames, source 140–256), cam_000 and cam_002
(cam_001 sees the tool only behind the wrist). Lid lifted by the pointed tool from below.

* **Contact (force)**: tare (0.029, −0.033, −0.009) N, noise σ ≈ (0.039, 0.043, 0.031) N;
  one sustained episode, source 122–456 (to the end; static ≈0.55 N load after the robot
  stops); 166 contact / 53 absent / 10 uncertain RGB-D frames; EE rotation 0.04° (no gravity
  compensation needed). All 30 clip frames are confirmed contact.
* **TCP projection audit** (all cameras, active and pre-registration calibrations): white shaft
  in cam_000 within 0.0…+1.0 cm of observed depth in every frame; the dark tip segment reads
  +1.3…+3.7 cm nearer even before contact; the projected TCP pixel lands on the lid lip once the
  lid lifts (+5…+6 cm). Static frames: predicted TCP 0.2–0.5 cm from an observed surface point
  in all three cameras. The pre-registration calibration misses the shaft in cam_000. The
  physical TCP is not unambiguously visible in any camera; Phase A on the TCP is
  `occluded_by_nearer_surface` (cam_000) / `tip_not_observed_in_depth` (cam_002) in every sample.
* **Phase A** on a calibrated shaft point (`secondary_feature`, 10 cm behind the TCP), cam_000:
  58 valid comparisons (30 independent clip frames; the rest are ping-pong replays), raw
  AllTracker vs projection median 0.25 px, p95 0.50 px (320×240);
  `adaptive_requery` never triggered (tolerance 10 px), so its output equals diagnostic mode.
  cam_002: shaft hidden by the arm, 0 valid. Tracking is bit-identical to baseline.
* **Controls**: Phase B/C with weight 1e-6 reproduce baseline (segmentation |Δ| ≤ 9e-5);
  ICP alone is not bit-repeatable (pose |Δ| ≤ 0.003 between identical tracking inputs).
* **Phase B / C / combined**: effects are real (controls) but inconsistent between cameras
  (lid / base reference-region accuracy; baseline cam_000 0.79/1.00, cam_002 0.80/0.93):
  B 0.72/1.00 and 0.61/0.60; B with tool-volume exclusion 0.59/0.99 and 0.61/0.41;
  C 0.87/1.00 and 0.49/0.91; A+B+C 0.89/0.99 and 0.39/0.44. Phase C moved the window
  contact distances by < 1 mm (lr_T = 1e-4); its weighted contact loss (0.21–1.24) was 1–10 %
  of the *weighted* visual term (`L_visual_final` is the unweighted main loss; the optimiser
  uses 200 × main loss). Correction: an earlier version of this section compared it with the
  unweighted main loss and called it "2–20× the visual loss". 4/9 cam_002 windows were
  skipped (motion and contact evidence disagree). Counts on independent frames (tracking is
  ping-pong; windows 4–8 replay frames 28…0): Phase C contact samples cam_000 58 = 30
  independent frames, cam_002 30 = 29; Phase B seed-active windows cam_000 3 → 1 independent
  (with tool exclusion 7 → 4), cam_002 1 → 0 (2 with tool exclusion).
* **Conclusion for this recording**: no phase gives a consistent improvement; keep A in
  `diagnostic_only` and B/C disabled by default. Multiview motion disagrees between cameras
  already at baseline (final lid angle 14.6° vs 8.0°). Root cause: see
  [multiview_root_cause_black_box.md](multiview_root_cause_black_box.md) (cam_002/cam_001
  per-camera failures: unstable two-part segmentation plus grazing, noisy lid geometry; not
  coordinate frames).
