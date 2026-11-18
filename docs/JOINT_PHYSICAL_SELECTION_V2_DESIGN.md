# Joint selection v2: observable physical model selection

`joint_discovery.comparison.selection: observable_physical_v2`. This is the default for the joint-aware path (`pipelines/joint_aware_reconstruction.py --selection v2`). Legacy `heldout_predictive` stays reproducible and is bit-identical, but it is never used as a fallback.

## 1. What is decided

A single label used to mix three different questions. v2 answers them separately:

| question | output |
|---|---|
| What rigid motion does the data show? | rotation and translation evidence: `motion_field/*`, `rotation_vs_translation` |
| Which physical 1-DoF joint is *identified*? | `physical_joint_status` ∈ `IDENTIFIED_REVOLUTE`, `IDENTIFIED_PRISMATIC`, `TYPE_UNOBSERVABLE`, `MODEL_INCONSISTENT`, `NO_SIGNIFICANT_MOTION` |
| What motion model should render this clip? | `working_motion_model` ∈ `revolute`, `prismatic`, `static`, `none`, plus `working_model_is_approximation` |

`physics_joint_export_ready` can be true only for an `IDENTIFIED_*` status. This is enforced in `PhysicalDecision.__post_init__`. The G8 producer (`tools/export_parttwin_handoff.py::physical_provenance`) refuses a production asset otherwise. A legacy render-based "IDENTIFIED" needs an explicit `--assumed-physical-joint "<approval note>"`, recorded with `assumed` provenance.

## 2. Algorithm, in order

```
all times ─ static-core depth residual ACF ─> block length L ─> contiguous blocks, ~35% interior blocks held out
            (no joint fitted; partition written to model_comparison/block_partition.json BEFORE fitting)
training blocks only:
  R and P fitted (unchanged EM / geometry_fit)                     -> fits are nuisance-refit, both models
  candidate-independent support S_core / M_proposal / U            (support_consistency.py)
  chained intervals [b_k, b_k+1] over training-block starts:
     robust finite rigid motion of M_proposal                      (motion_field.py: RANSAC + Cauchy IRLS Kabsch,
                                                                    delete-one-surface-cluster jackknife covariance)
     per-camera 3D estimate + per-camera 2D re-projection refinement (3D at a, 2D tracks at b)
  shared axis u, one signed Δθ_k per interval, gauge: net Δθ >= 0  (estimate_shared_angular_axis)
  calibrated null: same estimator on static-core patches matched in count/extent per interval,
     inflated by the moving/static robust residual ratio             (rotation_observability.py)
  rotation evidence: net Σ Δθ_k, block bootstrap over intervals, vs null q95
  hinge second: t_k = (I - R_k) o + h_k u (bootstrap over intervals) -> line sd, screw pitch
  consistency: second rigid group, moving base, inlier coverage, tool proximity
  assembly: static/moving interface at t_ref (≥ 2 cameras, ≥ 3 clusters, else ABSTAIN) ->
     soft hinge plausibility, prismatic guide alignment, implied minimum hinge rotation
  R/P prediction separation: δ = h_R - h_P on identical observations, common translation removed,
     per-block median vs empirical residual σ
held-out blocks:
  per-observation log-likelihood R − P on the common candidate-independent track set, summed per block,
  bootstrap over blocks (≥ 4 blocks, else INSUFFICIENT_INDEPENDENT_EVIDENCE)
select_physical_joint (physical_selection.py)
```

Key equation (spec §2). For one rigid part, `X_i(b) − X_j(b) = R_ba (X_i(a) − X_j(a))`, so the centred Kabsch fit measures the finite rotation without any hinge. Translation cancels exactly, and finite rotations are used: there is no small-angle approximation.

Statistics. Every interval, CI and p-value resamples *whole* units: time blocks (intervals), or surface clusters (5 cm voxels) for the per-interval covariance. Duplicating tracks, pixels or pairs cannot shrink any interval (tests A6, C23). The pairwise `rigid_flow_omega` is exported as a diagnostic only (`spatial_flow_diagnostics.npz`). Its pair count is never used as statistical power.

## 3. Decision flowchart

```
                     M_proposal ≥ 20 and (rotation detected or net translation > static-patch null q95)?
                       │no ─────────────────────────────────────────────> NO_SIGNIFICANT_MOTION   (working static)
                       │yes
       consistency flags (second rigid group in ≥ 2 intervals, moving base, inlier coverage < 50 %, tool)?
                       │yes ────────────────────────────────────────────> MODEL_INCONSISTENT      (working none)
                       │no
 rotation CI excludes 0 AND |net| > calibrated null q95 AND axis cone95 < 20°?
   │yes                                                                │no
   ├ hinge line sd ≥ 5 cm ────────> TYPE_UNOBSERVABLE (working revolute) │
   ├ hinge far from observed interface (CONTRADICTED) ─> MODEL_INCONSISTENT
   ├ fitted R axis ≠ field axis (> 15°) ───────────────> MODEL_INCONSISTENT
   ├ held-out blocks prefer P ─────────────────────────> MODEL_INCONSISTENT
   └ else ─────────> IDENTIFIED_REVOLUTE (export ready unless cameras disagree on the axis)
                                                                       │
            every hinge on the observed interface would need a rotation > 3·max(upper bound, MDR)
            AND observed guide aligned with the slide AND moving/static residual ≤ 2 AND cameras agree?
                       │yes ───────────────────────────────────────────> IDENTIFIED_PRISMATIC (export ready)
                       │no ────────────────────────────────────────────> TYPE_UNOBSERVABLE    (working prismatic,
                                                                             approximation, not export ready)
```

Every decision carries `reasons`, `failure_codes` and a `counterfactual` ("why is the rejected model not also acceptable").

What is deliberately *not* a rule (spec §7.1):
- A remote hinge or a short arc does not imply prismatic.
- A hinge outside the bounding box is a graded prior. It is never a veto.
- A large angle does not imply revolute; the consistency checks and the assembly check run first.
- A lower RGB render error does not imply the correct joint.

## 4. Render evidence (joint-aware stage `compare`)

Render evidence can only **veto**, through `physical_selection.render_veto_allowed`. A veto needs:
- at least 6 independent held-out blocks (`count_independent_blocks`, so two runs of three frames count as 2, not 6);
- at least 2 paired training seeds;
- a difference larger than twice the between-seed standard deviation.

The current pipeline trains one seed. The current held-out design has 2 blocks. So render evidence is reported as `NOT_IDENTIFIABLE_FROM_AVAILABLE_BLOCKS` and never decides.

The controlled counterfactual (`evaluate_branch(..., motion_kind=...)`) keeps one branch's Gaussians, cameras, support and pixels fixed and swaps only `A(q_t)`. It is recorded in `joint_selection_v2/render_comparison/controlled_counterfactual.json`.

## 5. Integration

| stage | v2 behaviour |
|---|---|
| `select` (new) | `cli select-v2 --from-run <discovery run>` writes `<run>/joint_selection_v2/` |
| `authority` | `IDENTIFIED_X`: only branch X (decision status `identified`). `TYPE_UNOBSERVABLE`: both (`ambiguous`). `MODEL_INCONSISTENT`: stop, unless `--allow-inconsistent-diagnostics`. `NO_SIGNIFICANT_MOTION`: stop. |
| `compare` | render evidence + controlled counterfactual; may only veto |
| `final` | `PASSED` only for `IDENTIFIED_*`. `TYPE_UNOBSERVABLE`: `ABSTAIN` with the working branch named. `MODEL_INCONSISTENT`: `BLOCKED`. |
| `export` (G8) | production only if `physics_joint_export_ready`; diagnostics are flagged NOT_FOR_RESULTS |

## 6. Configuration

`joint_discovery.selection_v2` is the flattened local equivalent of the spec's `comparison.*` sketch. Unknown keys are rejected. The two safety switches cannot be disabled: `forbid_bbox_hard_gate` and `never_certify_type_from_occam_alone`.

All thresholds are declared defaults, with provenance recorded in `threshold_provenance`. They were checked on synthetic controls only, never fitted to a real scene. No scene names or classes appear anywhere in the code or configs.

## 7. Commands

```bash
# post-process an existing discovery run (writes a NEW directory)
python -m estimotion_2part.joint_discovery.cli select-v2 --scene <scene> --from-run <discovery run> \
    --config configs/joint_discovery.real_tcp2cm.yaml --tcp-npz <robot_archive.npz> --output <new dir>
# all five real scenes + camera ablations
bash pipelines/joint_selection_v2_pilot.sh
# tests
python -m unittest tests.test_joint_physical_selection_v2
```
