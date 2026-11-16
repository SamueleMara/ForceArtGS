# Calibrated joint-aware Phase 1: implementation and first real-data evidence (2026-11-16)

Implements [`FREEARTGS_PHASE1_JOINT_AWARE_MOTION_TCP_FORCE_CODEX_SPEC.md`](../FREEARTGS_PHASE1_JOINT_AWARE_MOTION_TCP_FORCE_CODEX_SPEC.md)
as the opt-in package [`estimotion_2part/joint_discovery/`](../estimotion_2part/joint_discovery/README.md).
**This does not claim that the real-data failures are fixed.** Legacy Phase 1 is unchanged and remains
the default (`joint_discovery.mode: legacy`). No Gaussian training was run. The "reference" values
below come from the earlier tracker-free study ([multiview_root_cause_black_box.md](multiview_root_cause_black_box.md))
or the legacy ICP. They are **oracle diagnostics only** and are never used for fitting or selection.

## 1. Summary

| Question (spec §15) | Answer from this run |
|---|---|
| Does the exact static/P/R model work when the tracks are good? | **Yes.** Synthetic: axis < 1°, hinge line < 5 mm, selection correct or abstaining (tests T05–T10, T07). Real cam_000 alone (black box): revolute, **14.9°** arc vs the ≈15° tracker-free reference, hinge line **0.4–0.6 cm** from the reference line, hinge radius 0.11 m. Real red fridge (cam_001+cam_002, default): revolute, **18.4°** about a near-vertical axis **4.6°** from the legacy ICP axis. Motion correctly stays zero until the door starts to move. |
| Which measured information first recovers valid revolute motion on the problematic black box with the default cameras? | **None of the default-camera variants recovers it fully.** With 2D reprojection + depth (B4) the shared fit is a 2.6–2.8° arc about a remote hinge (1.2 m); the tracker-free reference is ≈15°. **Disabling depth (2D reprojection only)** raises it to 8.0–8.8° about a plausible hinge (radius 0.14 m, axis 5.8–6.5° from the reference), so the depth channel drives the translation degeneracy. That run, however, leaks base tracks into the moving support (46% of moving tracks lie below the lid height). The view that does recover the motion is cam_000, which is excluded by default because of calibration disagreement. Adding cam_000 to the joint fit (B9) gives 6.9–7.0° but an axis **52–54°** off, the calibration conflict the spec anticipated. |
| Is the cam_001/cam_002 failure an optimizer problem? | **No.** Seeded at the reference hinge with the reference 0→−15° trajectory, the fit converges to the same ≈2.8° remote-hinge solution with equal or slightly better likelihood (`samuele/joint_discovery_reference_seed_diagnostic.py`). The failure is in the evidence (tracks/depth), as in the earlier root-cause study. |
| Does an incidence-angle depth-noise model fix it? | **No.** Inflating depth σ by 1/cos(incidence) (cam_002 lid median cos 0.39) changes the arc from 2.6° to 2.8°. Negative result; the model is kept and is on by default. |
| Do TCP and then force improve held-out selection? | **Not measurable on these episodes, and the code abstains instead of inventing a benefit.** Black box: the TCP canonical contact path lies 11–28 mm off every observed lid face under both hypotheses (both camera policies). Surface terms are therefore UNAVAILABLE. This is consistent with the documented 1.5 cm tip uncertainty and a possible underside contact. The path is also free-plane degenerate, and all 30 wrench samples are < 1 N with no declared tare/gravity compensation, so physics terms are disabled. Red fridge: no wrench in the archive; independent contact annotations give 5/30 contact frames (§4.3). |
| Prismatic regression control | White drawer, vision only (its TCP/sensor mounting is documented as uncalibrated, so robot terms are disabled). Prismatic candidate: 16.2 cm along an axis 5° from the measured EE push direction. The decision is **`ambiguous/unobservable` with `occam_preference: prismatic`**, because a ~1° remote-hinge revolute (8.9 m) also wins on held-out data. The first selector version labelled this **revolute**, a confident false decision; it was fixed with a remote-hinge gate (§4.4). |

## 2. What was implemented

* **Kernel** (`rigid_kinematics.py`): exact `A_R(θ) = [R, (I−R)o]` = `Exp([u; −u×o]θ)`, `A_P(d)`,
  sliding-contact maps, velocity decompositions, sign/gauge helpers. Works in NumPy and torch;
  gradient-checked (T01–T04, T23).
* **Observations** (`observations.py`): a sparse partial-track table with no zero padding.
  * 2D observations are kept when depth is invalid.
  * Depth is sampled only from valid same-surface pixels, with edge flags (fixes the `−inf`
    bilinear contamination of `tracker_toolkit.get_4d_traj_maps` in the new path, T12).
  * Each camera uses its own `K` and fixed `T_WC`, cross-checked against COLMAP.
  * Time map: RGB-D frame → source frame → timestamp (T22).
  * Robot-tool exclusion uses the measured TCP/EE kinematics. It starts 1.5 cm behind the tip so
    the touched surface is kept.
* **Discovery** (`candidates.py`, `assignments.py`, `geometry_fit.py`):
  * RANSAC finite-motion proposals give both P and R seeds, including remote-hinge seeds so that R
    can mimic P fairly.
  * A per-track static/moving/outlier Student-t mixture over all of a track's observations
    (never an affine blend of positions).
  * Sparse robust least squares over (S² axis chart, hinge-line chart, `q_t`, anchors) with
    `q_ref = 0`, a timestamp-aware weak acceleration prior, and a Schur-complement covariance.
* **Selection** (`model_selection.py`): the time split is made before any fitting (T24). Both
  hypotheses are fit on identical data and compared by held-out predictive likelihood against a
  static null, with a paired standard error over tracks. Possible outputs are `revolute`,
  `prismatic`, `static/no_motion` and `ambiguous/unobservable`. When the revolute fit has
  collapsed to its prismatic limit, the result also carries `occam_preference`.
* **Robot** (`tcp_contact.py`, `wrench_physics.py`):
  * Sliding contact `x0 = A(q)^-1 p_tcp` and slip `s = v_tcp − v_M`.
  * Surface support comes only from observed moving-part geometry along the whole canonical
    contact path, and all hypotheses are scored on the same supported samples.
  * Wrench handling: sign conversion (`f_obj = −f_reaction`), signed F∥/F⊥ with exclusion
    reasons, `Q_H`, power, one-sided passivity, Coulomb direction with one shared bounded μ, and
    null controls.
  * Ratios are reported but never used as a classifier (T13–T19).
* **Exports** (`exports.py`): `joint_candidates.json` (both candidates; prismatic origin
  `null`/N/A), `joint_states.npz`, and per-camera `transforms_{0,1}.json` with `pose_kind`
  (`physical_camera`, or `virtual_part_camera = A(q)^-1 T_WC`). Virtual-camera rendering
  equivalence holds to 1e-15 m on every real run (point-splat proxy, T20).

Tests: `tests/test_joint_discovery_*.py`, **41 tests, all passing** (`python -m unittest`, freeartgs env,
~40 s). Baseline suite before this work: 271 tests, 6 pre-existing errors (missing `pycolmap`/`polars`/
`pyarrow`, one CUDA/CPU device mismatch in `test_dataset_loading`), unchanged by this work.

## 3. Real-data inputs

Scenes are read-only from `/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/`.
Outputs are in `outputs/joint_discovery_runs/<scene>/` (git-ignored); each run has `manifest.json`
(git SHA/diff hash, package versions, config hash, camera sources, frame map).

| Scene | RGB-D frames | Cameras extracted | Tracks (per camera) | Robot data |
|---|---|---|---|---|
| `final_black_box_episode_000000` | 70:128:2 (30) | cam_000/001/002 | 3608 / 1880 / 2375; 1.2–2.4k start after t=0; 91–769 per camera are 2D-only | TCP + raw wrench (`tcp_rgbd.npz` from the root-cause study), joined on source frame |
| `final_red_fridge_episode_000000` | 90:148:2 (30) | cam_001/002 (+cam_000 v1) | 8.7k / 14.0k | TCP, no wrench; contact annotations from the prior study |
| `white_drawer_20261021_161154_episode_000000` | 150:330:6 (31) | cam_001/002 | 12.4k / 18.8k (4-px grid); final runs 3.1k / 4.7k (8-px grid) | EE only; TCP/sensor mounting uncalibrated, robot terms disabled |

`T_WB = I` is accepted because every `cameras.json` pose is declared in `base_link`. Tracks
are AllTracker (repository weights), full resolution, queried every 10 frames on a 4-px grid
inside the whole-object mask.

## 4. Results

### 4.1 Black box (problematic revolute lid)

Final runs (`outputs/joint_discovery_runs/final_black_box_episode_000000/final/`). Inputs:
tool mask starting 1.5 cm behind the tip, incidence-angle depth noise on, remote-hinge gate on.

| run | cameras | depth | decision | z(R vs P) | R arc | R hinge radius (ratio to extent) | P candidate | R axis vs ref | R line vs ref |
|---|---|---|---|---|---|---|---|---|---|
| B4 (default) | cam_001+002 | yes | revolute | 3.2 | 2.8° | 1.21 m (4.7) | 6.3 cm "lift" along (0.00, 0.32, 0.95) | 12.0° | 2.5 cm |
| 2D only (depth off) | cam_001+002 | no | revolute | 7.6 | 8.0° | 0.14 m (0.3) | 2.3 cm | 5.8° | 5.6 cm |
| B9 (stress) | cam_000+001+002 | yes | revolute | 8.2 | 6.9° | 0.42 m (1.6) | 5.6 cm | 52.4° | 0.5 cm |
| cam_000 only (diagnostic) | cam_000 | yes | revolute | 18.4 | **15.0°** | 0.12 m (0.2) | 4.0 cm | 11.9° | **0.6 cm** |

Earlier ablations on the first inputs (tool mask at the tip; `.../final_black_box_episode_000000/*_20261116_1555*`)
gave the same picture:

| run | R arc | R hinge radius | R axis vs ref | R line vs ref |
|---|---|---|---|---|
| B4 | 2.6° | 1.26 m | 13.5° | 6.0 cm |
| B4, incidence noise on | 2.8° | 1.20 m | 13.3° | 3.3 cm |
| 2D only | 8.8° | 0.14 m | 6.5° | 7.2 cm |
| B9 | 7.0° | 0.42 m | 53.7° | 1.1 cm |
| cam_000 only | 14.9° | 0.11 m | 12.9° | 0.4 cm |

The reference is the tracker-free hinge (direction (0.998, −0.060, 0.020), point near y 0.59,
z 0.17) and ≈15° (cam_000) to 21° (cam_002) of opening at the clip end. The reference's own axis
uncertainty is several degrees (it is a plane-intersection estimate), so axis differences around
12° are only partly attributable to the fit.

* **Label vs kinematics.** All default-camera runs say "revolute" because a revolute explains
  held-out data better than a translation (z ≈ 3 for B4). But the arc and hinge are physically wrong.
  The label alone is not a success criterion.
* **Support.** With depth, moving support is coherent: about 7% of moving tracks lie below the
  lid height (z < 0.145 m, a diagnostic split from the prior study), against 54% of all tracks.
  The 2D-only run leaks base tracks (46%), because without depth, points near the hinge are
  ambiguous and fall to the mixing prior.
* **Per-camera residuals** (median, moving tracks): about 0.65–0.72 px in B4, 0.39–0.40 px for
  cam_000 alone, and 1.16 px for cam_000 inside B9. The calibration conflict shows up as that
  camera's residual, as in synthetic test T11.

### 4.2 Contact and wrench (black box)

| | cam_001+002 | cam_000 only |
|---|---|---|
| contact frames (heuristic: \|F − tare\| > 0.3 N, tare frames 0–2; **not** an independent annotation) | 18/30 | 18/30 |
| canonical TCP contact path length | 2.2 cm | 1.7–2.5 cm |
| observed-face support | UNAVAILABLE: contact 11–13 mm off the observed face | UNAVAILABLE: 24–28 mm off |
| free-plane degeneracy of the contact path | degenerate (both P and R) | degenerate |
| valid force samples (\|f\| > 1 N, moving TCP) | 0/30 | 0/30 |
| wrench physics terms | disabled: no tare/gravity compensation declared | disabled |
| force-direction selection | ABSTAIN | ABSTAIN |

### 4.3 Red fridge (revolute control)

B4 (cam_001+cam_002, default): **revolute**. Held-out z = 28.5 over prismatic and 76 over static.
The arc is 18.4° about axis (−0.046, −0.057, 0.997), 4.6° from the legacy cam_001 ICP axis (that
ICP reports 22.0°). `q_t` stays at 0 for t = 0–13 and then opens steadily. Hinge radius of the
moving support is 0.32 m; 6.7k moving tracks.

S3 used the prior study's independent force-z-score contact annotations (joined on source frame),
with re-extracted tracks (tool mask starting 1.5 cm behind the tip). The vision decision was again
revolute (held-out z = 24.4 over prismatic). Only **5/30** clip frames are annotated as contact.
Under both hypotheses the canonical TCP path lies **32–40 mm** off the observed door face
(63–82 supporting points), so contact terms are **UNAVAILABLE**. The TCP most likely pushes a door
edge that is not observed as a face. The contact path is free-plane degenerate. The archive holds
no wrench, so S4 was not possible.

### 4.4 White drawer (prismatic control, vision only)

Tracks: grid stride 8 (3.1k / 4.7k tracks) to keep a run under 10 min. A first stride-4 pass
(12k / 19k tracks, 40 min per run) gave the same candidates. Robot terms are disabled because the
TCP/sensor mounting is uncalibrated.

| run | decision | z(R vs P) | prismatic axis | P span | angle to measured EE push direction (diagnostic) | revolute fit |
|---|---|---|---|---|---|---|
| B4 (2D + depth) | **ambiguous/unobservable**, `occam_preference: prismatic` | 11.9 | (−0.306, 0.952, −0.009) | 16.2 cm | 5.1° | 1.0° about a hinge 8.9 m away (24× the support extent) |
| 2D only | **ambiguous/unobservable**, `occam_preference: prismatic` | 14.0 | (−0.310, 0.951, −0.006) | 16.0 cm | 4.9° | 1.1° about a hinge 3.5 m away (11.6×) |

The joint value stays at zero until the drawer starts moving. The EE moved 22.4 cm over the
same window; the gap is consistent with an approach phase or the tool sliding on the face.

**This control caught a confident false decision in the first version of the selector.** The
revolute fit wins on held-out data (z ≈ 12–14) with a ~1° arc about a hinge many metres away.
That indicates a small real rotation component (drawer play, or the documented ~1.5° cam_001/cam_002
calibration disagreement) on top of the translation. A 1-DOF revolute with such a remote hinge is
the degeneracy named in spec §9.2/§9.3, not an identified hinge. The selector now gates any
revolute win on `hinge radius / robust support extent > remote_hinge_ratio` (5, scale-free,
configurable; extent = 5–95 percentile range). Such a win becomes `ambiguous/unobservable` with
`occam_preference: prismatic`. A synthetic test covers it (real revolute, hinge 4 m away, 3° arc).
Under the same gate the black-box default run (ratio 4.7) stays `revolute`, close to the boundary.
The fridge (ratio 0.74) and cam_000 (ratio 0.2) are unaffected.

## 5. Failure table (first measured failure stage)

| Stage | Location | Finding |
|---|---|---|
| Tracks / depth (cam_001, cam_002, black box) | evidence, not code | The joint fit with depth prefers a 2.6–2.8° remote-hinge motion. A reference-seeded fit converges to the same optimum. Removing depth increases the arc to 8.0–8.8°. |
| Legacy depth lifting | `tracking/tracker_toolkit.py::get_4d_traj_maps` (`depth_map_t[depth_map_t == 0] = -inf`, bilinear `grid_sample`) | Mixes invalid/other-surface depth at edges. Replaced in the new path by `observations.sample_depth_valid` (T12). Legacy is untouched. |
| Legacy pre-filter | `tracking/tracker_toolkit.py::get_visible_trajectories` (`torch.all(combined_visibility, dim=0)`, global 2σ displacement filter) | Requires visibility in every window frame and removes minority moving tracks. Not used by the new path, which keeps partial tracks: 1.2–2.4k black-box tracks per camera start after t = 0. |
| Calibration | cam_000 extrinsics (2–5 cm offset, prior study) | Fusing cam_000 rotates the shared axis by 52–54° (B9). Default stays cam_001+cam_002. |
| Model selection (fixed) | `model_selection.score_candidates_on_heldout` | The first version accepted a held-out revolute win in the remote-hinge regime (white drawer: 1.4° about 6.5 m). It is now gated (§4.4). |
| TCP contact geometry | robot record (`tcp_rgbd.npz`) vs observed lid | The contact point is not on any observed face (11–28 mm off). Contact terms are disabled. |
| Wrench | recorded wrench, no tare/gravity metadata | Below 1.1 N for the whole clip. Physics terms are disabled. |

## 6. Not run / limitations

* B0 (legacy Phase 1 re-run) and B1 (MoMa-SG reference) were **not re-run** here. B0 numbers
  are in the earlier study. MoMa-SG is not vendored. The spec's revolute-transform caveat is
  covered by independent tests.
* Level-V2 Gaussian feedback, the appearance loss, held-out-*camera* evaluation and downstream
  3DGS training were not run. The FreeArtGS adapter was exercised only through its schema and
  virtual-camera equivalence checks.
* Red fridge: the B3/B9/cam_000 runs were stopped for time (about 20 min each at 22k tracks); only
  B4 and S3 completed.
* Pixel-noise calibration: the static-core MAD is about 0.1 px. That is below the configured
  floor of 0.2 px, so the 1 px prior is used. Moving-lid track errors measured in the prior study
  (1–5 px) make 0.1 px unrealistic for the moving support.
* Black-box contact flags come from a force-threshold heuristic and are labelled as such.

## 7. Reproduce

```bash
cd ~/ForceArtGS   # freeartgs environment
python -m unittest tests.test_joint_discovery_se3 tests.test_joint_discovery_twist \
  tests.test_joint_discovery_partial_tracks tests.test_joint_discovery_multiview \
  tests.test_joint_discovery_contact tests.test_joint_discovery_force \
  tests.test_joint_discovery_observability tests.test_joint_discovery_legacy_adapter
S="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/final_black_box_episode_000000"
P="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/multiview_root_cause_20261114/black_box_ep0/diagnostics"
O=outputs/joint_discovery_runs/final_black_box_episode_000000
mkdir -p $O && cp "$P/tcp_rgbd.npz" $O/ && cp "$P/hinge_reference.json" $O/hinge_reference_tracker_free.json
bash pipelines/joint_discovery_real.sh "$S" 70:128:2 $O/tcp_rgbd.npz $O extract
# drawer prismatic control (vision only):
D="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/white_drawer_20261021_161154_episode_000000"
W=outputs/joint_discovery_runs/white_drawer_20261021_161154_episode_000000
for c in cam_001 cam_002; do python -m estimotion_2part.joint_discovery.cli extract --scene "$D" --camera $c \
  --frames 150:330:6 --grid-stride 8 --output $W/observations_stride8/obs_$c.npz; done
python -m estimotion_2part.joint_discovery.cli fit --scene "$D" --output $W/final/B4 \
  --observations $W/observations_stride8/obs_cam_001.npz $W/observations_stride8/obs_cam_002.npz
python samuele/joint_discovery_reference_seed_diagnostic.py --scene "$S" \
  --observations $O/observations/obs_cam_001.npz $O/observations/obs_cam_002.npz \
  --hinge-reference $O/hinge_reference_tracker_free.json --output $O/oracle_diagnostic_reference_seed.json
python -m estimotion_2part.joint_discovery.cli summarize --runs $O/B4_* $O/diag_* \
  --reference-axis 0.99799 -0.06006 0.02002 --reference-point -0.56 0.59 0.172
```
