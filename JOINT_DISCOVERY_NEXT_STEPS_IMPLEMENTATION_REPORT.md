# Joint discovery next steps: implementation report

Contract: `FORCEARTGS_JOINT_DISCOVERY_NEXT_STEPS_CODEX.md` (2026-11-17).
Per-scene results: [`docs/joint_discovery_next_steps_results.md`](docs/joint_discovery_next_steps_results.md).

## 1. Baseline and changes

* **Baseline.** `main` at `468071f`, clean, in sync with `origin/main` (SamueleMara/ForceArtGS).
  * `git diff --check` was clean.
  * The 41 joint-discovery tests passed.
  * Full suite at baseline (run from a worktree pinned at `468071f`): 346 tests, **4 pre-existing
    errors**: `test_apply_robot_registration` and `test_debug_audit_trace` (missing `pycolmap`),
    `test_dataset_loading` (CUDA/CPU device mismatch), and
    `test_tcp_tracking_pipeline.SlideWindowIntegrationTests`. Recorded in
    `outputs/joint_discovery_next/NEXT_A0_full_suite_baseline.txt`.
* **Nothing was committed or pushed.** The separate `SamueleMara/FreeArtGS` remote, the datasets
  and the historical runs were not modified. Superseded sprint outputs were moved, not deleted,
  to `outputs/joint_discovery_next/final_black_box_episode_000000/_superseded_mixed_code/`.

Modified (all new behaviour opt-in; legacy defaults unchanged and tested):

| file | change |
|---|---|
| `estimotion_2part/joint_discovery/cli.py` | `run_discovery`: fold override, grouped statistics, Stage-E hook, output contract, fingerprints, explicit `ABSTAINED.json`; `fit` flags; new `audit-depth` and `folds` commands; stale-cache refusal |
| `.../config.py` | `DepthEvidenceSection`, AR(1) noise fields, `uncertainty_method`, `remote_hinge_policy`, observability thresholds, `enable_physical_selection` (all validated) |
| `.../model_selection.py` | per-track held-out predictions exposed, cluster-robust z, observability policy, `type_confidence`, AR(1) calibration with small-sample correction |
| `.../geometry_fit.py` | optional `extra` residual hook (Stage E) with correct Jacobian sparsity |
| `.../observations.py` | AR(1) inflation in `depth_sigma` (only when ρ is set) |
| `.../synthetic.py` | unflagged depth corruptions; `make_sliding_contact` (TCP + Coulomb wrench + verified calibration) |
| `.../README.md` | module table, commands |

New:

| file | purpose |
|---|---|
| `depth_conflict_audit.py` | Stage A frozen-hypothesis audit |
| `depth_evidence.py` | Stage B1 depth support codes |
| `validation_folds.py` | Stage C1/C2 grouped stats, folds |
| `observability.py` | Stage C3 |
| `contact_annotations.py` | Stage D1 |
| `wrench_calibration.py` | Stage D2/D3 |
| `physical_selection.py` | Stage E |
| `gaussian_feedback.py` | Stage F (actual gsplat 0.1.11 path) |
| `samuele/audit_joint_contact_windows.py` | full-episode contact/wrench audit |
| `samuele/stage_f_frozen_asset_score.py` | real frozen-asset scoring |
| `samuele/summarize_joint_next_steps.py` | `comparison.csv` / `summary.md` |
| `pipelines/joint_discovery_next_steps.sh` | staged A/B pipeline |
| `configs/joint_discovery_next_steps.example.yaml` | example config for the new options |
| `tests/test_joint_discovery_{depth_conflict,depth_validity,correlated_noise,multiview_holdout,remote_hinge_observability,contact_windows,wrench_calibration,force_selection,gaussian_renderer,end_to_end}.py` | the ten §10 test files |

## 2. What is implemented, and what is not

**Implemented**

* **A1.** `audit-depth` evaluates frozen hypotheses on the identical table.
  * Only per-track anchors are re-estimated (joint fixed), with RGB-D and with 2D-only anchors.
  * One row per observation carries every field in §3 A1, plus a reason code for rejected rows.
  * Gaps are attributed by camera, time and patch, weighted and unweighted.
  * Heatmaps are drawn on raw RGB and depth.
  * Reference hinges are evaluated only with role `reference_posthoc`.
* **A2.** Static/moving statistics: signed bias, MAD, trimmed mean, lag-1 autocorrelation, 2D drift
  early vs late, incidence, distance to depth discontinuities, cross-camera patch disagreement,
  six-way depth classification, and a consensus static core.
* **A3.** Ablation matrix on identical tracks and folds (`--fold-from`).
* **B1.** Candidate-independent depth codes: front-surface/silhouette, temporal jump, local
  rigidity, incidence, robot proximity, `unverified`. 2D always survives.
* **B2.** AR(1) correlated-depth model, `σ_eff = σ √((1+ρ)/(1−ρ))`. ρ comes from the training
  static core with Marriott–Pope correction. Ablated, off by default.
* **C1.** Candidate-independent surface groups (camera × 5 cm voxel × query window), cluster-robust
  z, group bootstrap, design effect / n_eff, and grouping sensitivity (3/5/10 cm, ± query window,
  per camera).
* **C2.** Contiguous and forward time blocks and held-out-camera folds, each with an explicit
  q-at-test rule (`interpolated`, `extrapolated_constant`, `constrained_by_other_camera`) and
  coverage.
* **C3.** The revolute is labelled identifiable only if all of these hold:
  * R vs best translation, p90 at |q| ≥ ⅔ max, ≥ 3 σ_px in some camera;
  * hinge line within one support extent of the observed moving support;
  * Schur-complement covariance (anchors marginalized) finite and non-singular.

  Otherwise the label is `ambiguous/unobservable` with `occam_preference`, and the hinge origin is
  not reported as identified. Threshold sensitivity is disclosed. The legacy ratio gate remains
  the default.
* **D1.** Contact timeline states and provenance; only `operator_label` or
  `independent_video_annotation` can confirm contact. Fusion of the force event with TCP–object
  proximity, and heuristic evaluation, using the existing `export_force_contact_annotations`
  detector.
* **D2/D3.** `WrenchCalibration` (R_EE_S, p_S^EE, sign, tare, gravity, tool-mass bound, clock),
  each with a DECLARED / ESTIMATED / VERIFIED status.
  * Data tests: tare with a two-window drift test, orientation constancy, multipose gravity fit.
  * Readiness gates; conversion `F_obj^W = s R_WEE R_EE_S (F − F_bias)` with the raw channel kept.
  * Torque transport only with a known sensor origin.
* **E.** Confirmed sliding contact (point-to-observed-plane rows, plane fixed from the original
  observed geometry) and calibrated force, inside `fit_geometry`:
  * force rows: Coulomb direction with μ fitted once on training samples (bounded grid), plus
    one-sided passivity;
  * held-out physical NLL added to the cluster-robust vision difference;
  * S2/S3/S4 kept separately, with the parameter change versus S2 and whether the decision changed;
  * force goodness-of-fit gate (best hypothesis ≤ 30° mean angle);
  * nulls: shuffled-time, sign-reversed, gravity-only.

  Every gate failure is an ABSTAIN that leaves the vision decision untouched.
* **F.** Frozen-asset scoring through the existing gsplat 0.1.11 `project_gaussians` +
  `rasterize_gaussians` path:
  * explicit WXYZ/XYZW layout; metric camera-Z depth (alpha-normalized);
  * pixel provenance with z-buffer ownership; no depth loss behind robot or occluders;
  * L_RGB / L_depth / L_alpha on identical pixels; asset fingerprint checked;
  * fail-closed asset loading (`MISSING_` / `INCOMPATIBLE_CANONICAL_ASSET`);
  * a real-data scorer that maps an asset across calibration versions only when all cameras
    agree on one rigid map.
* **G (partial).** The adapter now writes an explicit `freeart_adapter/ABSTAINED.json` (no
  transforms) for non-identified decisions. The existing adapter is unchanged for identified ones.

**NOT implemented / only diagnostic**

* **B2 per-camera/patch/block bias nuisance `b_{c,p,t}`: not implemented.** The A2 evidence shows
  the dominant error is a *time-growing* patch-coherent drift. A smooth bias model with enough
  freedom to absorb it would also absorb the hinge motion, and no test distinguishes the two.
  Only the AR(1) covariance alternative was implemented.
* **Stage F joint ↔ Gaussian alternation (bounded Gaussian optimization): not implemented.** Only
  frozen-asset scoring exists. Black-box overlays and fridge overlay images were not produced.
* **Stage G downstream FreeArtGS training (legacy vs new, matched budget): NOT RUN.** Gates A, B
  and F are not passed on the primary scene, and §12 forbids training a Stage-E model that never
  used force.
* **MoMa-SG reference (B1 in §11): not run, not vendored.**
* **Physical selection on real scenes: not exercised.** No scene has independently confirmed
  contact or a verified wrench calibration, so it would abstain by construction.
* **Black-box renders: none**, because there is no canonical asset.

## 3. Tests

Command (freeartgs env; unittest only):

```
/home/samuelemara/miniconda3/envs/freeartgs/bin/python -m unittest discover -s tests -p "test_joint_discovery_*.py"
```

**New test files: 51 tests, all PASS.**

| file | tests |
|---|---|
| depth_conflict | 3 |
| depth_validity | 4 |
| correlated_noise | 3 |
| multiview_holdout | 3 |
| remote_hinge_observability | 3 |
| contact_windows | 4 |
| wrench_calibration | 6 |
| force_selection | 7 (~105 s) |
| gaussian_renderer | 14 (GPU, ~2.4 s) |
| end_to_end | 4 |

The 41 earlier joint-discovery tests still pass.

**Full suite** (`python -m unittest discover -s tests`, main tree): 401 tests, 367.6 s, **2 errors,
both pre-existing**:
* `test_apply_robot_registration` (missing `pycolmap`);
* `test_dataset_loading.test_art_parser_uses_per_view_intrinsics_and_shared_joint` (CUDA/CPU device
  mismatch).

The other two errors in the baseline record (`test_debug_audit_trace`,
`test_tcp_tracking_pipeline`) do not occur in the main tree. They were most likely caused by the
pinned worktree lacking untracked local files, not fixed by this work. The `FAILED scene_0 …`
lines in the log are messages printed by tests, not test failures. Output:
`outputs/joint_discovery_next/NEXT_full_suite_after.txt`.

* Environment: NVIDIA GeForce RTX 4070 Laptop (8 GB), driver 580.178.04, torch 2.5.1+cu124,
  gsplat 0.1.11 (prebuilt CUDA extension from the torch-extension cache; nothing compiled).
* Synthetic seeds are fixed in every test, and synthetic ground truth is never passed to a solver.
* Tolerances: float64 for kinematics. float32 for the rasterizer (depth 2 mm; covariance rtol 1e-4;
  dL/dq within 5% of central FD at h = 0.01 rad; dL/d(axis) within 10%).

## 4. Black-box depth conflict (Stage A)

See results §1.1–1.3.

**Source.** On the lid only, in cam_001 and cam_002:
* 2D tracks lag the material motion, with ⊥-ray error growing 3 → 31 mm in cam_001;
* cam_002's depth drifts progressively far (+21.5 mm) at grazing incidence (cos 0.45 → 0.37).

**Effect size.**
* Only 4% (cam_001) and 21% (cam_002) of lid observations follow the rigid reference motion within
  ±30%, against 94% in cam_000.
* With 2D-placed anchors, the depth terms favour the 2.8° fit over the 15° fit by 13.7k NLL, of
  which 13.5k comes from cam_001.
* Lid depth residuals have lag-1 autocorrelation 0.96–0.98, so the independent per-observation
  likelihood treats about 30 correlated samples per track as independent evidence.

**Ruled out.**
* A camera-level geometry mismatch for cam_001/002 (consensus static core: median bias
  −0.02/−0.04 mm, 0.1–0.2 px).
* An optimizer cause (the earlier reference-seeded fit returns to the same optimum).

## 5. Candidates, support, observability and fold confidence

See the results document.
* **Default cameras:** 2.8° revolute about a 1.2 m remote hinge, now **ambiguous/unobservable**
  (cluster z 1.7; hinge 1.16 m outside the support; 1.7 σ_px from a translation). Fold labels are
  unstable (held-out camera favours P).
* **cam_000 diagnostic:** 15.0°, hinge radius 0.12 m, identifiable.
* **2D-only:** 8.0° but 74% base leakage (post hoc).

## 6. Calibration evidence

| item | status |
|---|---|
| `T_WB` | identity, accepted because every `cameras.json` pose declares `base_link` |
| camera poses | current registered `metadata/cameras.json` (COLMAP-cross-checked by the loader) |
| `R_EE_S` | identity, **DECLARED** (user confirmation), not experimentally verified (no multi-pose or reference-push data: EE rotation over clips is 0.01–0.54°) |
| force sign | object-on-robot, **DECLARED** (operator, 2026-11-13); no reference push |
| tare | estimated on force-quiet windows; drift **verified only for the suitcase** (0.028 N); black box / fridge / dog lack a second contact-free window |
| gravity | constant over each clip (orientation change 0.012–0.544°); no tool-mass bound declared, so UNVERIFIED |
| time sync | exact `source_frame_index` join only; **hardware clock offset not measured** |
| torque origin | unknown, so torque transport disabled |

Not verified: sensor rotation, sign, gravity model, clock offset, torque origin.

## 7. TCP contact and force in training and selection

* **Real scenes.**
  * Contact timelines: 0 `contact_confirmed` anywhere (no independent annotations). Clip-level
    `contact_probable` counts: black box 30/30, dog 22/31, suitcase 9/36, fridge 0/30.
  * No scene passes the wrench gates.
  * S3 and S4 therefore ABSTAIN on every real scene, and the selections are vision-only (S2). The
    output contract records `used_for_contact_fit = used_for_force_fit = false`.
* **Synthetic (Gate E).**
  * S3/S4 change the fitted parameters (`integration_changed_parameters = true`) and select the
    true type.
  * μ recovered: 0.30.
  * Sign-reversed wrench: S4 abstains (mean angle 180°) while contact remains.
  * Time-shifted wrench: worse fit.
  * Uncalibrated wrench: S3 only.

## 8. Renderer

* **Synthetic.** Actual-rasterizer tests pass: rotation/covariance, quaternion boundary, metric
  depth, occlusion and robot exclusion, gradients vs FD, virtual-camera equivalence for
  RGB/alpha/depth, and frozen-scoring fairness.
* **Black box:** `MISSING_CANONICAL_ASSET`.
* **Fridge.** The existing twopart_blend asset was mapped (one rigid old→new calibration map,
  consistent to 0.014°). On held-out times: revolute 5.34 < prismatic 5.75 ≈ static 5.79,
  consistent in both cameras, render coverage 98%.

  Caveats: the asset was co-trained with the legacy (prismatic) joint, robust depth residuals are
  5–8 cm, and no overlays were produced.
* **Gate F: PARTIAL.**

## 9. End-to-end FreeArtGS

**Not run** (see §2). Legacy mode is unchanged (tested), and the adapter abstains explicitly when
the joint is not identified.

## 10. Gates and blockers

| gate | status |
|---|---|
| A | **PASS.** Source-localized, with effect size. |
| B | **FAIL / UNRESOLVED** on the black box with default cameras. The candidate-independent evidence rules and the AR(1) model do not recover the motion. The system now abstains instead of reporting a wrong axis. Synthetic PASS. |
| C | **PASS.** Synthetic drawer-with-play and near-hinge cases. Real: drawer not called revolute; fridge stays revolute and identifiable (cluster z 5.5, 17.7 σ_px from a translation); the dog's old prismatic label is downgraded to ambiguous (cluster z −1.2). |
| D | **FAIL** (calibration not verifiable from these recordings). Contact windows are reviewable but heuristic/probable only. |
| E | **PASS** synthetic; **ABSTAIN** on real data, as required. |
| F | **PARTIAL.** Numerical gates pass; one real frozen-asset ranking (fridge); no black-box asset; no overlays; no alternation. |
| G | **NOT RUN.** |

**Measurable next actions**

1. **Re-track the black-box lid in cam_001/002.** Try shorter query windows (re-query every 3–5
   frames, short tracks only) or bidirectional tracking with a forward–backward consistency gate.
   Accept only when the reference-free cross-camera fold (§1.3) becomes stable and the
   held-out-camera R-vs-P z turns positive.
2. **Independent contact annotations** (operator or video) for the black box and fridge clips, in
   `source_frame_index, state, provenance` CSV form, to enable S3.
3. **Wrench calibration session:** a static multi-pose sequence (≥ 10° rotations) for R_EE_S and
   gravity, one reference push of known direction (sign), and an event-based clock-offset
   measurement.
4. **Independently verify the cam_000 registration** (its +28 mm TCP residual) before it may move
   from diagnostic to production.
5. **Train a hypothesis-neutral static canonical asset for the black box** under the current
   calibration, then rerun `samuele/stage_f_frozen_asset_score.py` and produce overlays.
