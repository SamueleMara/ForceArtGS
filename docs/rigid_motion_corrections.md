# Rigid-motion corrections: implementation and validation

This follows [multiview_root_cause_black_box.md](multiview_root_cause_black_box.md). It implements and tests the
proposed rigid-motion corrections on the black box (episode 0, source frames 70–128, stride 2) and on the red
fridge (episode 0, source frames 90–148, stride 2). Both clips use the same calibration and preprocessing as
the earlier report.

All new behaviour is opt-in. With the defaults, tracking is bit-identical to the previous code: T1/T2 match in
all 9 windows and all 60 part masks match. No Gaussian training was run. Contact Phase A stays
`diagnostic_only`, and Phases B and C stay off.

Results, figures and configs are under
`/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/rigid_motion_corrections_20261116/`.

## 1. Summary

| Stage | Outcome |
|---|---|
| 1. SE(3) transforms | Implemented as T = C·Exp(ξ)·C⁻¹·T_init, opt-in. The tracker's transforms are now exactly rigid during optimisation, not projected at export. Black-box cam_000 improves: RMS 0.7° → 0.4° against its reference, final −15.3° against −15.0°. It does not repair the cameras whose failure is upstream (black-box cam_001/cam_002). On the fridge it leaves cam_001 correct, but cam_002 falls from −15.9° to −11.4° with a 1.7° backward step, and cam_001's static-part drift rises from 0.45° to 1.33°. **Physically valid, but not sufficient on its own, and not uniformly better.** |
| — legacy defect | `reinit_part_trans` has never applied its re-initialisation: it writes into a temporary view. The defect is documented and tested; the fix is opt-in (`transform_reinit_apply`) and stays off for bit-identity. |
| 2. Part-aware track filter | Implemented as two LMedS rigid hypotheses, provisional assignment, robust refit and 0.5 weights for ambiguous tracks; a track is rejected only if neither motion explains it. **The premise did not hold:** the legacy 2σ filter already keeps 99–100% of genuine moving tracks. The new filter improves precision (static precision 1.00, p95 error halved) but loses 12–37% of genuine lid tracks. Downstream it flips black-box cam_002's part labels (+9.3°, closing through the base) and degrades fridge cam_002. **Not adopted; default off.** |
| 3. Multiview reference selection | Implemented as coverage, in-sample and held-out RGB-D tests against the static hypothesis, data-derived informativeness floors, head-to-head contests judged by independent views, a stability record and an explicit "unresolved" status. Two defects in its first version were found and fixed in validation (§6.1). **Final result:** it never selects a camera the references show to be wrong. The legacy rule does so in 3/4 black-box variants and 1/5 fridge variants. But it is **unresolved in 9 of 10** scene-variant cases, because these rigs rarely provide independent evidence. |
| 4. ICP failure, black-box cam_002 | **Not** initialisation and not rank deficiency. The 6-DOF truncated ICP objective has its minimum away from the true motion, because about half of the moved lid leaves the visible surface (overlap 0.47–0.57) and the black material's depth is biased at grazing angles (lid-plane tilt lags by up to 6.5°, then jumps 10.4°). RGB has no usable correspondences: 4 SIFT keypoints. The silhouette rejects the pipeline motion but cannot confirm the reference. The depth contains the information (CRB 0.1–0.3°, and a hinge-constrained fit recovers it); the implemented estimator cannot extract it. |
| 5. Contact priors | Unchanged defaults: A diagnostic-only, B/C off, no weights tuned, no seeds activated. Phase C gained opt-in rejection of robot-tool geometry. The black box still has no independently observed contact surface, so Phase C stays gated off. |
| 6. Validation | Five variants × two scenes, identical settings (§8). The only robust gain is SE(3) on black-box cam_000. No variant makes the black-box cameras agree. On the fridge, every change worsens cross-camera agreement (6.3° → 11–13°). Several "plausible" final angles come from label flips or jumps and are counted as failures. |
| 7. Fridge contact | The 6σ force threshold is not "too conservative". During free arm motion z is 3–7 regardless of distance to the door, and 4 of the 5 force-confirmed frames have the TCP 19–43 mm from the door in both cameras. The force channel cannot separate this push from free motion; the counts are reported separately (§10). |

**Bottom line.** Physically valid rigid motion is now enforced in the tracker. Consistency across calibrated
views is not achieved on either clip. On the black box the cause is upstream: segmentation, view geometry, ICP
on a dark, edge-on surface. On the fridge, the baseline was close (6°), and the tracking changes move cam_002
away. The multiview selector now says so rather than picking a camera.

## 2. Code changes

| File | Change | Option (default = legacy) |
|---|---|---|
| `estimotion_2part/models/trans_models.py` | `RigidTrans`: T = C·Exp(ξ)·C⁻¹·T_init with ξ ∈ R⁶ (`se3_exp` = matrix exponential of the twist). C is a translation to the window's source-point centroid and only conditions the rotation coordinates. `T1`/`T2` are read-only properties with the legacy (num_T, 4, 4) layout, so the losses, the Phase B/C priors and the exports are unchanged. Also adds `make_trans`, `project_to_se3` (used only on initialisations) and parameterization-neutral accessors (`transform_parameters`, `set_transforms_trainable`, `assign_transform`). The legacy `Trans` class is untouched, apart from those accessors, which do exactly what the old call sites did. | `transform_parameterization: matrix` (`se3`) |
| `estimotion_2part/solve_transformation.py` | Builds the model through `make_trans`, sets the optimiser parameter groups and the frozen first stage through the accessors, and `reinit_part_trans`/`joint_reinit_part_trans` can now actually write a re-initialisation (§3.1). | `transform_reinit_apply: false` |
| `tracking/part_aware_track_filter.py` (new), `tracking/tracker_toolkit.py` | Two-hypothesis track pre-filter (LMedS motion A on all candidates, LMedS motion B on what A leaves, alternating refits, 50/50 weights for ambiguous tracks). A track is rejected only when **both** rigid motions fail to explain it. The global 2σ rule is skipped in this mode; the density filter is unchanged. Per-track weights (1 / 0.5) are returned in the previously unused auxiliary output. | `track_filter_mode: global` (`part_aware`) |
| `preprocessing/multiview_quality.py` (new), `preprocessing/prepare_multiview.py` | Motion-quality evaluation and reference selection (§6). With no independently supported camera the run stops and writes `multiview_quality.json` (`UnresolvedReferenceError`), or with `on_unresolved="flag"` it records `reference_status: unresolved`. Legacy manifests are unchanged. | `reference_selection="motion"` (`quality`) |
| `pipelines/train_local_datasets.py`, `samuele/run_tcp_contact_ablation.py` | `--multiview-reference-selection {motion,quality}` | `motion` |
| `estimotion_2part/window_priors.py`, `tracking_prior_hooks.py`, `prior_context.py` | Phase C: canonical contact support rejects points inside the tool cylinder. The cylinder runs from the calibrated TCP back to the EE, at the window's reference frame. Nothing in front of the tip is removed. | `contact_exclude_tool_radius_m: null` |
| `samuele/dump_window_tracks.py` | Saves keep masks and weights for both filter modes on identical candidates. | — |
| `utils/pipeline_config.py` | New defaults listed above. | — |

## 3. Tests

New: `tests/test_rigid_transform_tracking.py` (14), `tests/test_multiview_quality.py` (5) and
`tests/test_contact_tool_rejection.py` (3).

- **Exp map.** `se3_exp` is orthonormal with det = +1 and matches Rodrigues to 1e-12.
- **Rigidity during optimisation.** Through 200 Adam steps of the real loss, ‖RᵀR − I‖ < 1e-5 and det > 0 at
  every step. The control case — the legacy matrix parameterization under the same optimisation — leaves SO(3).
- **Gradients.** Analytic gradients match central finite differences for all 3 translation and 3 rotation
  coordinates, and `gradcheck` passes.
- **Synthetic recovery.** Two parts, 2 mm per-axis noise, 10% gross outliers, the pipeline's lr_T = 1e-4,
  confident part weights:

  | Start | SE(3) rotation error | SE(3) median lid-point error | Matrix (projected) rotation error |
  |---|---|---|---|
  | From the true motion | 0.20° | 0.35 mm | 0.26° |
  | 3° / 5 mm away | 0.49° | 0.65 mm | 1.63° |

  With the soft 0.2/0.8 initial weights, **both** parameterizations end 1–2° from the truth even at 0.5 mm
  noise. The legacy mixture loss lets low-weight static points pull T2 towards identity. This is a property of
  the objective and is not changed here.
- **Legacy re-initialisation is a no-op.** A test documents this (§3.1). The opt-in fix writes the
  re-initialisation, and the rigid model resets ξ.
- **Part-aware filter.** On synthetic scenes it keeps over 90% of the moving and static inliers, rejects over
  80% of gross outliers, and is independent of which part is the majority. With no outliers, the legacy 2σ rule
  removes over 10% of the moving part and under 1% of the static part.
- **Multiview quality.** On rendered two-camera scenes:
  - the true motion is selected over a larger spurious translation;
  - two spurious cameras give "unresolved";
  - two correct cameras agree;
  - in a head-to-head, a contestant's own view gets a non-zero resolution floor, so frames where the two
    motions coincide are not counted (regression test for the defect in §6.1).
- **Phase C tool rejection.**
  - The default keeps all support.
  - With the option set, at least 95% of tool points are removed and the surface in front of the tip is kept.
  - Invalid configurations are refused.
- **Legacy behaviour.**
  - Global mode of `get_visible_trajectories` matches `HEAD` bit-for-bit in 20/20 random trials.
  - Re-running black-box cam_000 with defaults reproduces the earlier baseline exactly.
  - ICP is not bit-reproducible run to run: up to 2 mm on the static part from identical masks. That is the
    ICP noise floor, independent of these changes.
- **Full suite.** 261 tests pass on CPU. Five modules cannot be imported in the `freeartgs` environment because
  `polars`, `pyarrow` and `pycolmap` are missing; they are unrelated to this work. One nerfstudio dataparser
  test fails only when CUDA is visible (a device mismatch inside nerfstudio, also unrelated); it passes on CPU.

### 3.1 Legacy defect found: transform re-initialisation never takes effect

`reinit_part_trans` assigns `transform_net.T1[0].data = …`. Indexing creates a temporary view, and assigning
`.data` on it does not write into the parameter (verified with torch 2.5.1). So the re-initialisation every 50
epochs is computed and then discarded. `joint_reinit_part_trans` is called without `save_dir`, so it computes
nothing at all.

Legacy results therefore never used re-initialisation. The default keeps that, for bit-identity.
`transform_reinit_apply: true` makes it effective; it was not used in the ablation, so the SE(3) experiment
changes the parameterization only.

## 4. Stage 1 — SE(3) parameterization on the real scenes

The variant `se3` sets only `transform_parameterization: se3`; tracking inputs, losses and ICP are unchanged.

- **Rigidity.** The tracker's transforms are now exactly rigid: the maximum singular-value deviation is 0.0000,
  against 0.024–0.030 for the legacy matrices.
- **Black box.** The physically valid camera improves:
  - cam_000 RMS error against its own reference falls from 0.7° to 0.4°; the final angle is −15.3° against a
    reference of −15.0°;
  - the axis moves from 19° to 17° off the hinge;
  - lid recall of the moving mask goes from 0.62 to 0.85 at the last frame.

  The rigid model does **not** fix cam_001 or cam_002. They stay at +1.5° and −3.2° about the hinge, with axes
  78° and 75° off it, because their failures are upstream (part labels, view geometry; §7).
- **Fridge:** see §8.

Per-window motion and the final ICP rotations for every variant are in `diagnostics/ablation_*.json`.

## 5. Stage 2 — moving-part track preservation

### Method

- **Candidates.** Both filters were applied to the same candidates: every forward window pair (0, k), 28 pairs
  per black-box camera and 21 per fridge camera, re-extracted by `samuele/dump_window_tracks.py`. Its counts
  match the pipeline logs.
- **Regions** (evaluation only, never seen by the filters):
  - black box: geometric lid, base and tool, from calibration, robot kinematics and the measured rim height;
  - fridge: the door, meaning object points within 1 cm of the door plane, excluding the tool.
- **Genuine moving track.** A lid track whose 3D endpoint error, against the tracker-free hinge reference, is
  within the 95th percentile of base-track errors against zero motion in the same pair.
- **Static precision.** The fraction of kept base tracks within that same tolerance.
- **Recovered SE(3).** An LMedS rigid fit on the kept lid tracks, compared with the reference.

### Results (mean over pairs; errors are medians over pairs)

| Black box | Filter | Lid recall | Genuine-lid recall | Base recall | Base precision | Lid error, median / p95 (mm) | Base error, median / p95 (mm) | Lid SE(3): rotation error / displacement error |
|---|---|---|---|---|---|---|---|---|
| cam_000 | legacy 2σ | 0.98 | **1.00** | 1.00 | 0.95 | 2.3 / 10.7 | 2.4 / 8.7 | **1.4° / 1.2 mm** |
| | part-aware | 0.79 | 0.88 | 0.82 | **1.00** | **1.9 / 4.9** | **2.0 / 4.7** | 1.7° / 1.3 mm |
| cam_001 | legacy 2σ | 0.95 | **0.99** | 0.98 | 0.97 | 5.5 / 12.0 | 3.8 / 13.3 | **3.7° / 8.4 mm** |
| | part-aware | 0.82 | 0.88 | 0.80 | **1.00** | **4.9 / 9.4** | **3.1 / 9.0** | 3.7° / 8.6 mm |
| cam_002 | legacy 2σ | 0.90 | **1.00** | 0.99 | 0.96 | 7.7 / 19.6 | 3.5 / 11.7 | **2.7° / 4.2 mm** |
| | part-aware | 0.49 | 0.63 | 0.84 | **1.00** | **4.7 / 8.9** | **2.9 / 7.9** | 3.9° / 6.9 mm |

| Fridge | Filter | Door recall | Rest recall |
|---|---|---|---|
| cam_001 | legacy 2σ | 1.00 | 0.97 |
| | part-aware | 0.84 | 0.72 |
| cam_002 | legacy 2σ | 0.99 | 0.98 |
| | part-aware | 0.80 | 0.80 |

### Interpretation

- **The premise does not hold on these clips.** The legacy filter rejects 2–10% of lid-region tracks and under
  1% of base tracks; the earlier cam_002 window-0 figure was 12%. But it keeps **99–100% of the genuine lid
  tracks**. The lid tracks it rejects have errors beyond the static-track tolerance: they are bad
  correspondences, mostly the depth errors on the black lid documented earlier, not genuine motion. The
  per-window lid displacements (≤ ~2 cm) are small enough that genuine lid tracks stay inside the global 2σ band.
  The fridge door is also kept at 99–100%.
- **The part-aware filter does what it was designed to do:**
  - it never rejects a track that a rigid motion explains;
  - it removes non-rigid outliers — base precision rises to 1.00, and the p95 errors halve;
  - it is not circular.
- **But it is stricter than the data warrant.** Its inlier bound comes from the legacy 4.55% Gaussian tail (see
  the module docstring). The real residuals are heavy-tailed, so it drops 12–37% of genuine lid tracks and 16–20%
  of base tracks.
- **The recovered lid SE(3) does not improve** (1.4° → 1.7° and 2.7° → 3.9°).
- **Not adopted.** It stays opt-in; its downstream effect is in §8. The bound was not retuned, because that would
  be an invented threshold.

## 6. Stage 3 — multiview reference selection

`preprocessing/multiview_quality.py` scores each camera's moving-part ICP motion in the calibrated world frame at
the same source timestamps. The per-camera reports are in `diagnostics/mvq_*.json` and in the per-variant
ablation JSONs. The components are:

| Component | Measure |
|---|---|
| Coverage | Fraction of valid ICP poses, and the fraction of the frame-0 moving surface that still projects onto observed object depth |
| In-sample residual | Own-camera depth residual of the predicted moving surface versus the static hypothesis (frame-wise sign test, p < 0.05) |
| Improvement over static | The same comparison, as a residual ratio |
| Held-out consistency | The same test in every *other* camera's depth (not used to estimate this camera's motion) |
| Informativeness | A frame counts only if the two hypotheses' predicted surfaces are further apart than the observing view's resolution floor. For points another camera measured, the floor is their residual in this view at frame 0, before anything moves, so it contains the calibration offset and depth bias. For points this camera measured itself, the frame-0 residual is zero by construction, so the floor is its temporal depth repeatability: the smallest mean absolute static-hypothesis residual of its own frame-0 points over the clip. Nothing is tuned. |
| Agreement / contests | Pairwise final relative rotation and axis angle. Head-to-head: each pair of motions on the same points, judged by third cameras; a camera's own view counts only when it prefers the other's motion. |
| Stability | Largest single-step share of the rotation path |

**Selection rule.** A camera is eligible when:
- its own depth supports its motion;
- another camera supports it;
- no camera contradicts it.

The reference must then win or tie every head-to-head contest against cameras that are not discredited.
- A *tie* means no view distinguishes the two motions.
- If a camera's own depth prefers its own motion and no independent view decides, the contest is a
  *disagreement*.
- The held-out residual ratio only breaks true ties.

If no camera qualifies, the result is **unresolved**: `prepare_multiview` stops by default, after writing
`multiview_quality.json`.

**Black box, baseline.**
- The legacy rule would choose **cam_001** when all three cameras are used. It is the camera with the largest
  motion-score gap, yet its rotation is −0.7° about the hinge with an axis 82° off.
- With cam_000 and cam_002 only, as in the earlier report, it chose **cam_002**.
- The quality rule returns **unresolved**:
  - cam_002's motion is contradicted by cam_001's depth (19 of 27 informative frames worse than static);
  - cam_001's motion is "supported" by cam_002 against static. Its moving points include surfaces whose real
    motion any motion explains better than none, so beating the static hypothesis is not sufficient;
  - cam_000 and cam_001 each prefer their own motion (23/23 and 24/24 informative frames), and cam_002's view
    cannot separate them (floor 27 mm against a 17 mm separation);
  - cam_000's motion, the correct one, cannot be verified from other views. The frame-0 floors of its lid points
    are 109 mm in cam_001 and 55 mm in cam_002 — mostly surfaces those cameras do not see, plus cam_000's 2–5 cm
    calibration offset — against a 17–32 mm motion separation.

So the evaluation **refuses a wrong reference instead of picking the right one**. On this scene no RGB-D
evidence outside cam_000 can confirm cam_000.

**Fridge, baseline.**
- Both cameras' motions are supported by their own depth and by the other camera's depth against the static
  hypothesis: 12/14 and 16/16 frames.
- Head-to-head, each camera's depth prefers its own motion on every informative frame: 17/17 and 18/18. There
  is no third view, so the result is **unresolved**.
- The legacy rule chose cam_001 (−22.0°), which is inside the 17–24° plane-based reference. cam_002 (−15.9°) is
  1° below its lower end.

The selector says "unresolved" here because two cameras cannot settle a disagreement between them, even when
an evaluation-only reference favours one.

### 6.1 Defect found and fixed during validation

The first version measured every view's resolution floor at frame 0 on the union of both contestants' points.
In a contestant's own view, its own points have zero residual at frame 0 by construction, so the floor was 0
and every frame counted. Before the fridge door moves (frames 1–13) both motions are ≈ identity, so those frames
entered the sign test as coin flips.

The fridge outcome then depended on ICP run-to-run noise:
- baseline: cam_002's view preferred cam_002 in 19/29 frames, p = 0.068, a "tie", and cam_002 was **selected**
  on the held-out ratio;
- the bit-identical-tracking rerun: 20/29, p = 0.031, a "disagreement", **unresolved**.

On the motion frames alone, both runs give 15–16/16.

Two corrections, both making the code do what its docstring already stated:
1. **Own-view floor.** A contestant's own view, and the in-sample test, use the camera's temporal depth
   repeatability as their floor (see the table above). It uses the mean, not the median, because with ≈1 mm
   depth quantisation the median residual of an unchanged surface is exactly 0 in many frames.

   An intermediate version took the floor from the *other* camera's points instead. That raised cam_000's
   self-view floor on the black box to the cross-view calibration offset, discarded its valid self-evidence,
   and selected the wrong cam_001. It was rejected for that reason, not kept.
2. **Ties.** A contest is a "tie" only when no view distinguishes the motions. Previously one self-preferring
   view plus one undecided view counted as a tie. That let the held-out ratio, which is not a correctness
   measure, select fridge cam_002 at −9.4° in the "track filter only" variant.

Effect: baseline and rerun now agree on both scenes, and no variant selects a camera that the references show
is wrong. The black-box outcomes, which were all unresolved, are unchanged. The motion results are unaffected:
every non-selection field of the ablation JSONs is identical before and after.

**Fridge, per variant** (final rotation about the vertical; reference 17–24°):

| Variant | cam_001 / cam_002 | Legacy | Quality | Deciding evidence |
|---|---|---|---|---|
| baseline | −22.0° / −15.9° | cam_001 | unresolved | Each camera's depth prefers its own motion (17/17, 18/18); no third view |
| rerun | −22.1° / −16.0° | cam_001 | unresolved | Same (17/17, 18/18) |
| SE(3) only | −22.5° / −11.4° | cam_001 | unresolved | Same (14/14, 16/17) |
| track filter only | −21.3° / −9.4° | cam_001 | unresolved | cam_001's depth prefers its own motion (16/16); cam_002's is undecided (10/17 for its own motion, not significant) |
| both | −21.9° / −9.3° | cam_002 ✗ | **cam_001** | cam_001's depth contradicts cam_002's motion: only 2 of 13 frames are better than static |

Per the instructions, no global multi-camera optimiser was implemented. The current reference-camera architecture
is unchanged.

## 7. Stage 4 — why ICP fails on black-box cam_002

### Setup

- **Estimator.** The pipeline's own `refine_registration_tensor` (coloured ICP, then hybrid Gauss-Newton at 3
  voxels and 1 voxel, point-to-plane weight 1.0, point-to-point weight 0.2), run on CPU.
- **Input.** The exact per-frame lid clouds from the geometry-derived masks (`geomask_cam_002/point_clouds_1`),
  voxel 12.5 mm.
- **Reference (evaluation only).** cam_002's own tracker-free 1-DOF hinge fit.
- **Error measure.** Errors are lid point displacements, not translation norms at the camera origin. At the clip
  end the reference has a 392 mm "translation" at the origin, but the pipeline estimate's 74 mm is the wrong one.

Figure: `figures/fig_icp_failure.png`.

| Hypothesis | Test | Result |
|---|---|---|
| Inadequate rotation initialisation | Direct frame 0→t ICP, started from identity and from the reference motion | **Ruled out.** Both starts converge to the same angle (to 3 decimals) for every frame from 1 to 27; at src 102 that is −3.5° against a reference of −11.2°. |
| Local minima of sequential registration | Direct 0→t versus chained (t−1)→t versus the pipeline's local map | **Not the primary cause.** Direct ICP also fails (−0.9° to −4.0°). Sequential ICP is worse (≈0°). Only at the last two frames does the objective become multimodal: direct-from-identity jumps to −56° at src 126. |
| Objective prefers the wrong motion | Truncated hybrid objective at the reference vs at the estimates, frame 0→t | **Confirmed.** For t ≥ 2, J(direct ICP) < J(pipeline) < J(reference), e.g. at src 102 38 < 78 < 95 (×10⁻⁶ m²). The global minimum the estimator finds is not the physical motion. |
| Insufficient observed moving surface | Overlap at the reference: fraction of the moved frame-0 lid with an observed point within 1 voxel | **Confirmed.** Overlap is ≥ 0.98 up to src 80, then 0.54–0.57 from src 102 and 0.47 at the end. The lid turns edge-on (cam_002's elevation over the lid plane goes from 12.5° to 0° around src 106–110). Lid points drop from 1689 to 805 (src 102), then rise to 2137 as the underside becomes visible. Half the moved surface has no counterpart, and the truncated objective is minimised by sliding the remaining overlap rather than rotating it. |
| Systematic RGB-D error on the black material | Lid-top plane tilt from each camera's depth (no tracking); residual at the reference | **Confirmed.** cam_002's plane tilt lags the reference by up to 6.5° before the crossing (src 102: −4.7° vs −11.2°), then jumps 10.4° between src 102 and 106. The residual at the reference motion is 2.6–10 mm for cam_002, against 0.8–1.2 mm for cam_000. |
| Objective degeneracy (rank) | 6×6 Gauss-Newton Hessian at the reference, along the hinge twist | **Not rank-deficient.** The white-noise Cramér–Rao bound on the hinge angle is 0.09–0.34° (marginal over the other 5 DOF). The weakest direction is not the hinge twist (cosine 0.09–0.54). The failure is a biased minimum from model mismatch, not missing curvature. |
| RGB correspondences where depth is unreliable | SIFT/ORB/corners on the frame-0 lid; matching with Lowe's 0.75 ratio test; PnP | **Not available.** The lid is nearly black (mean intensity 23/255). It has 4 SIFT keypoints at the default threshold and 14 at contrastThreshold 0.003, giving at most 4 matches per frame, so PnP is impossible. **The silhouette is informative:** after the pipeline motion only 57–67% of the frame-0 lid projects inside the observed SAM object mask (src 90–120), against 94–100% for the reference. The silhouette rejects the rotation-to-translation substitution but cannot tell the reference from no motion in this view. |

**Contrast with cam_000** (same estimator, its lid clouds):
- direct ICP is within 0.2° of the reference from either start;
- J(reference) ≈ J(estimate);
- overlap is 0.96–1.0;
- the CRB is 0.03–0.04°;
- the pipeline's local-map chain adds ≈1.1° of drift (−16.1° against −14.9° direct).

**Theoretical versus implemented.** cam_002's depth is not information-free. Under white noise the hinge angle
is bounded to 0.1–0.3°, and constraining the motion to the hinge recovers −21° from the same depth; that
constrained fit is how the reference was built. The unconstrained 6-DOF truncated ICP cannot use that
information, because half the moved surface leaves the visible set and the grazing-angle depth is biased. The
estimator fails here; the camera is not unobservable in principle.

On the pipeline motion, the lid point-displacement error is a median of 11–20 mm and a p90 of 42–70 mm from
src 98 onward, compared with 2.5 / 3.2 mm for cam_000.

## 8. Stage 6 — ablations

The variants share frames, calibration, preprocessing (features, masks) and ICP settings. `rerun` is the new code
with defaults: its tracking is bit-identical to `baseline`, and its ICP differs only by run-to-run noise (cam_001
total rotation 4.7° vs 4.8°).

The angle is the signed rotation about the reference axis: the hinge on the black box, the vertical on the
fridge. Black-box references are tracker-free 1-DOF hinge fits from cam_000's depth (−15.0° at the clip end) and
cam_002's depth (−21.1°); their spread is the reference uncertainty. They are measurements, not ground truth.

### Black box (source frames 70–128)

| Variant | Camera | Moving part | Final angle | RMS vs ref000 / ref002 | Axis off hinge | Lid displacement error (mm, median) | Largest step | Moving mask at last frame: lid / base / tool, lid recall | Tracks kept | Tracker rigidity (max \|σ−1\|) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | cam_000 | 1 | −13.8° | 0.7 / 4.0 | 19° | 2 | 0.9° | 0.85/0.06/0.08, 0.62 | 0.99 | 0.027 |
| | cam_001 | 1 | −0.7° | 7.6 / 10.9 | 82° | 34 | 1.1° | 0.93/0.00/0.07, 0.73 | 0.97 | 0.030 |
| | cam_002 | 1 | −0.4° | 10.1 / 13.3 | 87° | 37 | 1.3° | 0.64/0.15/0.00, 0.74 | 0.97 | 0.024 |
| SE(3) only | cam_000 | 1 | **−15.3°** | **0.4** / 3.0 | 17° | 3 | 1.6° | 0.75/0.12/0.05, **0.85** | 0.99 | **0.000** |
| | cam_001 | 1 | +1.5° | 9.3 / 12.6 | 78° | 34 | 1.0° | 0.92/0.00/0.08, 0.68 | 0.97 | 0.000 |
| | cam_002 | 1 | −3.2° | 8.4 / 11.6 | 75° | 37 | 1.7° | 0.49/0.30/0.01, 0.74 | 0.97 | 0.000 |
| track filter only | cam_000 | 1 | −13.7° | 0.9 / 4.2 | 17° | 3 | 0.8° | 0.86/0.06/0.07, 0.76 | 0.79 | 0.022 |
| | cam_001 | 1 | +7.6° ✗ | 12.6 / 15.8 | 21° | 46 | 2.3° | 0.89/0.04/0.05, 0.92 | 0.80 | 0.029 |
| | cam_002 | **0** ✗ | −6.7° | 8.6 / 11.6 | 38° | 52 | **12.0° jump at src 118** ✗ | **0.15/0.73**/0.01, 0.24 | 0.76 | 0.017 |
| both tracking fixes | cam_000 | 1 | **−14.9°** | **0.3** / 3.1 | **15°** | 3 | 0.9° | 0.86/0.06/0.07, 0.75 | 0.79 | 0.000 |
| | cam_001 | 1 | +0.0° | 10.2 / 13.4 | 88° | 70 | 5.3° | 0.89/0.04/0.06, 0.87 | 0.80 | 0.000 |
| | cam_002 | **0** ✗ | **+9.3°** ✗ (closes through the base) | 12.7 / 15.9 | 7° | 61 | 4.4° | **0.21/0.69**/0.00, 0.18 | 0.76 | 0.000 |

Cross-camera agreement at the clip end (relative rotation / axis angle):

| Variant | cam_000 vs cam_001 | cam_000 vs cam_002 | cam_001 vs cam_002 |
|---|---|---|---|
| baseline | 14.9° / 85° | 16.0° / 85° | 3.6° / 14° |
| SE(3) only | 19.1° / 75° | 18.4° / 80° | 6.7° / 28° |
| track filter only | 21.7° / 32° | 11.2° / 52° | 15.2° / 49° |
| both | 15.4° / 87° | 24.7° / 9° | 9.4° / 89° |

Reference camera (legacy rule vs quality rule):

| Variant | Legacy (largest motion) | Quality |
|---|---|---|
| baseline | cam_001 (wrong) | **unresolved** |
| SE(3) only | cam_001 (wrong) | **unresolved** |
| track filter only | cam_002 (wrong, label flip) | **unresolved** |
| both | cam_000 | **unresolved** |
| **combined (both + quality reference)** | — | **unresolved: flagged, no reference chosen** |

Reading:
- **SE(3)** is the only change that helps without harming another camera. It makes the tracker physically valid
  and brings cam_000 to the reference: RMS 0.4°, 0.3° with both fixes.
- **The part-aware filter harms cam_002 and cam_001.** On cam_002 it flips which label moves: part 0 becomes 73%
  base. It also produces a 12° single-step jump at src 118. With both fixes, cam_002 rotates **+9.3°**, i.e. the
  lid closing through the base, which is non-physical. On cam_001 the rotation reverses sign (+7.6°).

  These are segmentation failures caused by removing 20–24% of the tracks (§5). Several final numbers look
  plausible — for example, cam_002's axis is only 7° off the hinge with both fixes — but they come from
  label-flipped static parts or jumps. They are failures, not improvements.
- **No variant makes the three cameras agree.** cam_002's failure is in ICP (§7) and is unchanged by tracking
  fixes. cam_001 sees too little of the lid (675 lid pixels at frame 0).
- **The quality rule never picks a wrong camera; the legacy rule picks one in 3 of 4 variants.**
  - The legacy rule chooses cam_001 or cam_002 in three variants. It chooses cam_000 under "both" only by
    accident: the other cameras' motions collapsed.
  - Under "both", only cam_000's motion is supported by its own depth: cam_002's own depth contradicts its
    motion, and cam_001's is uninformative. But no other view can check cam_000. Its frame-0 cross-view floors
    are 60–66 mm (cam_000 sits 2–5 cm off the others' calibration, plus occlusion), larger than the lid
    displacement. The honest outcome is "unresolved, single-view evidence only".

### Red fridge (source frames 90–148)

**Reference.** The fit is the plane of door points within 10 cm of the TCP, with the tool excluded; it comes
from the earlier report and is used for evaluation only. Each camera fits a different face of the door, so only
changes are compared:
- the plane is static until src ≈ 116–118 (clip frame 13–14);
- it then turns by ≈ 17° in cam_001's view, ≈ 20–24° in cam_002's view (to src 144) and ≈ 24° in cam_000's
  view.

The door closes monotonically, so any backward step in a recovered rotation series is non-physical. No
per-frame reference series exists for this clip, so the comparison is the final angle against 17–24°, motion
onset, and temporal consistency.

**Mask fractions.** These use the door region: object points within 1 cm of the door plane. That region
excludes the door's edges, gaskets and handle, so even a correct moving mask contains a large "rest" share.
Compare these columns across variants, not to 1.0.

| Variant | Camera | Moving part | Final angle | Error vs 17–24° | Axis off vertical | Static-part drift | Largest step | Backward steps (frame: °) | Moving mask door / rest / tool at src 120; at src 148; door recall at 148 | Tracks kept | Tracker rigidity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | cam_001 | 0 | −22.0° | inside | 2.6° | 0.45° | 2.2° | 7: 0.8 (pre-motion) | 0.41/0.57/0.02; 0.41/0.59/0.00; 0.99 | 0.99 | 0.034 |
| | cam_002 | 1 | −15.9° | −1.1° | 6.2° | 0.52° | **3.9° at src 124** | 21: 1.0 | 0.69/0.28/0.04; 0.38/0.62/0.00; 0.81 | 0.99 | 0.029 |
| SE(3) only | cam_001 | 0 | −22.5° | inside | 2.4° | **1.33°** | 2.2° | none | 0.40/0.58/0.02; 0.43/0.56/0.00; 0.97 | 0.99 | **0.000** |
| | cam_002 | 1 | −11.4° ✗ | −5.6° | 7.7° | 0.38° | 2.7° | **17: 1.7** | 0.66/0.30/0.04; 0.38/0.61/0.00; 0.89 | 0.99 | 0.000 |
| track filter only | cam_001 | 0 | −21.3° | inside | 2.1° | 0.35° | 2.4° | 7: 0.7 (pre-motion) | 0.44/0.54/0.03; 0.41/0.59/0.00; 0.98 | 0.76 | 0.030 |
| | cam_002 | 1 | −9.4° ✗ | −7.6° | 10.9° | 0.29° | 2.5° | **17: 0.9, 19: 0.6, 21: 2.4** | **0.40/0.57**/0.02; 0.40/0.60/0.00; 1.00 | 0.80 | 0.028 |
| both | cam_001 | 0 | −21.9° | inside | 1.8° | **1.32°** | 2.2° | none | 0.39/0.59/0.02; 0.42/0.58/0.00; 0.98 | 0.76 | 0.000 |
| | cam_002 | 1 | −9.3° ✗ | −7.7° | 10.7° | 0.34° | **4.9° backward at src 132** | **17: 0.7, 21: 4.9** | **0.40/0.58**/0.02; 0.40/0.60/0.00; 1.00 | 0.80 | 0.000 |

The rerun with defaults reproduces the baseline: −22.1° / −16.0°, with tracking bit-identical, the same
backward steps, and the largest cam_002 step 4.2° instead of 3.9° (ICP noise).

Cross-camera agreement at the clip end (relative rotation / axis angle):

| Variant | cam_001 vs cam_002 |
|---|---|
| baseline | 6.3° / 5.9° |
| SE(3) only | 11.2° / 7.2° |
| track filter only | 12.0° / 9.8° |
| both | 12.7° / 9.9° |

Reading:
- **cam_001 is robust to every variant.**
  - Its final angle is −21.3° to −22.5°, inside the reference, and its axis is within 1.8–2.6° of vertical.
  - Its motion starts at clip frame 14–15 (src 118–120), matching the reference onset.
  - With SE(3), its only backward step, a 0.8° pre-motion wobble, disappears. But the static part's drift
    rises from 0.45° to 1.33°. Exactly rigid transforms are not automatically more accurate ones: the legacy
    matrices were absorbing some error that now goes into the static part.
- **cam_002 is the fragile camera, and every change makes it worse.**
  - Its baseline already has a 3.9° single step at src 124, when the TCP reaches the door in cam_002's view,
    followed by a 1° reversal.
  - With SE(3), it rotates 1.7° backwards at the same frame and ends at −11.4°.
  - With the track filter, its moving mask at src 120 drops from 0.69 to 0.40 door: base points enter the
    moving part. It ends at −9.4° to −9.3°, with backward steps of up to 4.9°.
  - So its total rotation is under-estimated, and the reversals show it is not a smooth under-estimate but
    discontinuous tracking.
- **Agreement gets worse** with every change: 6.3° → 11–13°. Unlike the black box, the fridge baseline was
  already nearly consistent across views, and neither correction preserves that.
- **The part-aware filter costs 20–24% of tracks** here as on the black box. The fridge's large frontal door
  does not need it.
- **Reference selection.** The legacy rule picks cam_001, the correct camera, in four of five variants, and
  cam_002 (−9.3°) under "both". The quality rule is unresolved in four and picks cam_001 under "both" (§6). So
  on the fridge the legacy rule happens to be right more often. The quality rule never picks the wrong
  camera, but it refuses to decide when the only evidence is the two cameras' own depth.

## 9. Stage 5 — contact priors

- **Defaults are unchanged.** Phase A is `diagnostic_only`, `moving_prompt_enabled` and
  `contact_surface_enabled` are false, and the B/C weights are 0. No weight was tuned and no B/C run was made.
- **Phase B.** The seed artifacts were not activated. As shown earlier, the accepted seeds sit inside the
  tool/contact uncertainty. The opt-in mask cleaner (`samuele/clean_robot_from_masks.py`) is unchanged. In the
  earlier controlled test, removing 1–2% of mask pixels reshuffled the part labels and did not improve the motion.
- **Phase C.** The robot-geometry rejection (`contact_exclude_tool_radius_m`) is now available and tested. It
  only removes tool points from the canonical support; it does not create the missing contacted surface.
  - On the black box, the contacted underside of the lid lip is still not observed, so Phase C would still skip
    for lack of support. The earlier 53/56 matches were tool points.
  - Re-enable Phase C only when the remaining canonical support contains an independently observed moving
    surface within the TCP uncertainty.
  - The sliding-contact model is unchanged: the material point is re-matched over time, and no fixed
    TCP–object transform is imposed.

## 10. Stage 7 — red-fridge contact detection

### Evidence used

- **Force.** The exporter's z = ‖(F − tare)/σ‖, with the per-axis σ (0.035–0.059 N) taken from the first 0.5 s
  of the episode, while the arm is still. "Contact" means z ≥ 6 (≈ 0.29 N) sustained for ≥ 0.2 s; "absent" means
  z ≤ 3.
- **Visual.** The door plane from each camera's depth, the TCP-to-plane distance, and the documented 1.5 cm TCP
  uncertainty.

Figure: `figures/force_audit.png`.

### Findings

- **The arm moves continuously** at ≈ 0.105 m/s from about frame 88 to the end of the record. At rest (frames
  0–85) z ≈ 1–3. **During free motion z ≈ 3–7 regardless of distance to the door**, so the noise model measured
  at rest does not hold while the arm moves.
- **TCP more than 25 mm from the door in both cameras** (37 frames, so no contact is possible): z median 3.8
  (p75 5.9), and 6 of these frames are labelled force-"contact".
- **TCP within 15 mm of the door in at least one camera** (12 frames): z median 5.1 (p25 4.5, p75 6.2). This is
  barely above free motion.
- **Of the 5 force-confirmed clip frames** (110, 114, 116, 142, 144), four have the TCP 19–43 mm from the door
  plane in **both** cameras. Only frame 144 has a camera at the door (cam_002, 3.9 mm).
- **Plausible pushing interval.** cam_002 puts the TCP within 15 mm of the door for frames ≈ 120–144 (minimum
  2.7 mm). Over the same frames cam_001 says 25–42 mm. The two cameras disagree by 20–35 mm, more than the TCP
  uncertainty, so visual contact there is *plausible*, not confirmed.
- **The door keeps swinging after release.** cam_001's door angle changes monotonically from about frame 116 to
  188, while the TCP has withdrawn to 35–60 mm (frames 150–198). Door motion alone is therefore not evidence of
  contact. cam_002's door-plane angle is unusable: the plane fit switches surfaces.

### Separate counts, clip frames 92–148 (29 frames)

Force state is taken over native samples within ±1 of each RGB-D frame. "Visual contact" means the TCP is near
the door and the door is moving.

| Force \\ Visual | Contact | Uncertain | Absent |
|---|---|---|---|
| Contact (z ≥ 6, sustained) | 1 | 1 | **6** |
| Uncertain (3 < z < 6) | **4** | 6 | 9 |
| Absent (z ≤ 3) | 0 | 0 | 2 |

### Verdict

The 6σ threshold is not "too conservative". The force channel does not separate this low-force push from free
arm motion: it misses the push (z 3–6.5 while cam_002 puts the TCP at the door) and fires when the tool is
visibly 32–43 mm away. Lowering the threshold would label free motion as contact. **The sensor signal is
inconclusive for this interaction.**

Distinguishing it would need a noise model measured during matched free motion, or a dynamic model of the
wrench, and an agreed visual contact criterion. Neither was invented here.

The contact states are left as exported. This was not mixed with joint-type classification or dynamics.

## 11. Remaining limitations

- **References are measurements, not ground truth.**
  - The black-box references (cam_000 and cam_002 hinge fits) differ by 6° at the clip end.
  - The fridge reference is a 17–24° range from per-camera plane fits, with no per-frame series. So fridge
    errors are against a range, and the judgement on cam_002 rests on its departure from the range plus its
    non-physical backward steps.
- **Two clips, one episode each, one run per variant.** ICP is not bit-reproducible run to run: up to 2 mm on
  static parts, and a 3.9° vs 4.2° largest step on fridge cam_002. Differences of that size between variants
  are noise.
- **SE(3) changes the parameterization only.** Nothing compensates for the legacy objective's soft-weight bias
  towards identity (§3), or for ICP. The higher static-part drift on fridge cam_001 under SE(3) is not
  explained.
- **The part-aware filter's inlier bound** is the χ²₃ equivalent of the legacy 2σ tail and assumes Gaussian
  residuals. The real residuals are heavy-tailed. It was deliberately not retuned. A data-driven bound, such as
  one derived from the static-track residual distribution, would be the next thing to try, but only together
  with a change that stops the label flips.
- **Reference selection is decisive only with an independent view.** With two cameras there is none, so any
  disagreement is "unresolved".
  - The black-box cam_000 calibration offset (2–5 cm) makes cross-view floors of 44–109 mm. So the most
    accurate camera there cannot be verified by the others.
  - The own-view repeatability floor is an upper bound when the surface moves from frame 1 (8 mm for black-box
    cam_000).
  - The selector was revised once after seeing validation results (§6.1). Both revisions correct the code to
    its stated definitions, and the revision that would have selected a wrong camera was rejected. Still, the
    rule has now been validated on the same two clips it was debugged on. It needs a third scene before it can
    be trusted.
- **No global multi-camera optimisation**, as instructed. Joint estimation with calibration refinement, or a
  hinge-constrained ICP, is what the Stage 4 analysis points to for cam_002-type views. Neither was implemented.
- **The ICP diagnosis is for one camera and one scene.** The overlap and grazing-angle mechanism is plausible
  for fridge cam_002 at src 124 (the TCP enters its view of the door), but that was not diagnosed.
- **The force-contact audit has no matched free-motion noise model.** Distinguishing the push needs one, or a
  dynamic wrench model; none was fitted.
- **Not done:** no Gaussian training, no Phase B/C activation, no commits, no extrinsic refitting.

## 12. Reproduction

Environment: conda `freeartgs` (torch, CUDA, open3d). Tests use `unittest`.

```bash
# tests
cd tests && PYTHONPATH=..:. python -m unittest test_rigid_transform_tracking test_multiview_quality test_contact_tool_rejection

# variant configs (only the listed key differs from the defaults)
#   se3.yaml:  tracking: {transform_parameterization: se3}
#   trk.yaml:  tracking: {track_filter_mode: part_aware}
#   both.yaml: tracking: {transform_parameterization: se3, track_filter_mode: part_aware}

# black box: track + ICP + multiview for all variants ("rerun" = new code, defaults)
python samuele/run_tcp_contact_ablation.py \
  --scene ".../real/final_black_box_episode_000000" --cameras cam_000 cam_001 cam_002 \
  --frame-intervals 70:129 --frame-stride 2 --work-root <work> --artifacts <artifacts.yaml> \
  --stages track icp multiview --config rerun=none --config se3=se3.yaml --config trk=trk.yaml --config both=both.yaml
# fridge: same with --cameras cam_001 cam_002 --frame-intervals 90:149
# quality-based selection in the pipeline itself: add --multiview-reference-selection quality

# Stage 2 candidate tracks (both filter modes on identical candidates)
python samuele/dump_window_tracks.py --config <work>/configs/baseline_<cam>.yaml --output <tracks>/<scene>_<cam>
```

The evaluation scripts (`ablation_eval.py`, `eval_tracks.py`, `mv_quality.py`, `icp_diag.py`,
`silhouette_test.py`, `rgb_lid_texture.py`, `force_audit.py`, and the fridge contest checks) are archived in
`scripts.tar.gz` in the results directory. They are evaluation-only and are not part of the pipeline.

Results directory, `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/rigid_motion_corrections_20261116/`:

| Path | Content |
|---|---|
| `REPORT.md` | Copy of this document |
| `configs/` | Variant configs |
| `diagnostics/ablation_{blackbox,fridge}_*.json` | Per-variant, per-camera series, references, masks, tracks, rigidity, agreement, selection |
| `diagnostics/ablation_*.prefloorfix.json` | The same, before the §6.1 correction. Only the fridge `multiview` entries differ; the black-box file is identical. |
| `diagnostics/mvq_*.json` | Full quality reports, baseline |
| `diagnostics/fridge_contest_frames.json` | Pre-motion vs motion-frame contest split (§6.1) |
| `diagnostics/tracks_*.json` | Stage 2 per-pair track metrics |
| `diagnostics/icp_diag_*.json`, `silhouette_*.json`, `lid_texture.json` | Stage 4 |
| `diagnostics/force_audit.json` | Stage 7 |
| `figures/` | `fig_icp_failure.png`, `force_audit.png` |

The ablation datasets themselves (ICP poses, masks, point clouds) were not copied. They are reproducible with
the commands above.
