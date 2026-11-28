# All-episode runs and cross-demonstration PhysX validation: implementation report (28 Nov 2026)

Base: `multimodal-tcp-contact` at `dda4cc1`, plus the changes below. The spec is
`FORCEARTGS_CODEX_MULTI_EPISODE_CROSS_DEMO_PHYSX_VALIDATION_DETAILED_METRICS.md`.

## 1. Actual inventory

The source is `Backup_09_10_26/Sapien/output/real`, scanned on 28 Nov.

- **Totals:** 117 episodes in 13 candidate objects. Every episode has the RGB-D, depth, masks, cameras, timestamps,
  force convention and robot recording files. No two robot recordings are identical (sha256), so there are no
  duplicate pushes.
- **Objects with 2 or more episodes:**

  | object | episodes |
  |---|---:|
  | `final_black_box` | 12 |
  | `final_black_laptop` | 12 |
  | `final_light_blue_box` | 12 |
  | `final_matchbox` | 17 |
  | `final_orange_suitecase` | 12 |
  | `final_red_fridge` | 14 |
  | `final_robot_dog` | 19 |
  | `final_wooden_box` | 12 |
  | `white_drawer_20261021_161154` | 3 |

- **Objects with 1 episode, `INSUFFICIENT_EPISODES`:** `black_box_`, `black_monitor_`, `laptop_20261021_161154` and
  `dog_20261022_185923`.
- **Joint ground truth:** no episode has an independent joint-state file. Every real reference is
  `RGBD_RECONSTRUCTED`.
- **Timestamps:** `frame_timestamps.csv` is a relative 0.04 s grid. Equal timelines are therefore NOT treated as
  duplicates. A first version did, and wrongly flagged 66 episodes.
- **Object identity:** `configs/g9_real/object_group_manifest.yaml` is populated from the scan with
  `verified_same_physical_object: null`. The operator confirms each group. Unverified groups are refused with
  `OBJECT_IDENTITY_UNVERIFIED`.

## 2. Reused vs added

**Reused unchanged.** The per-episode chain:
- automatic force-guided selection, run on the working branch only;
- the q(t) extension;
- the rest-to-rest estimator with its G10 fits;
- `IsolatedSapienCpu` with its axis-inertia read-back;
- `objective.Episode`;
- the rest-plateau evidence.

**Added (`forceartgs_g9/cross_demo/`):**

| module | role |
|---|---|
| `inventory.py` | Scan, statuses, duplicate pushes, candidate and reviewed manifest. |
| `kinematics_alignment.py` | Type, axis and hinge checks in the robot base frame B against predeclared tolerances. Sets the training-gauge displacement `sign·(q − q0_test)`. |
| `heldout_episode.py` | Target record from a finished run (exact path). The target's native wrench is projected onto the TRAINING joint. Revolute: `(τ_ref + (p_ref − o_i) × F)·a_i`. Prismatic: `F·a_i`. Every native row is used; no mask, no fill, no second tare. The interval `[t0, T_eval_end]` includes the confirmed stationary tail. `TAIL_INPUT_INCOMPLETE` falls back to `[t0, T_stop]`. |
| `replay.py` | `TrainModel` frozen from one training run. It fails closed on `FIT_FAILED`, non-finite values, non-physical values or a URDF hash mismatch. One CPU PhysX rollout per pair; the engine reads back the axis inertia. The module has no optimizer call. `diagonal_parity` replays the training episode against the metrics that run saved. |
| `metrics.py` | The pure §8 metrics: time-weighted, identities asserted, segments, quartiles, row / column / object / sweep aggregates, pooled residual deviance, paired method comparison. |
| `cv.py` | Ordered N × N single-demonstration transfer, per-pair files, resume keyed by content hash. Pooled leave-one-out returns `POOLED_LOO_NOT_IMPLEMENTED`; it is not substituted. |
| `report.py` | CSV / JSONL / LaTeX tables, matrices and heatmaps, best and worst overlays, and an independent re-aggregation from `residuals.npz`. |
| `synthetic.py` | Known M and c. Five pushes per object, prismatic and revolute, plus a misspecified variant with a constant unmodelled load. Each push is trained with the existing `run_synthetic` (now with optional `pushes`, `q0` and `bias`; the defaults are unchanged). |

**Added pipelines and configs:**
- `pipelines/inventory_real_object_episodes.py`
- `pipelines/real_all_episodes_cross_demo.py`, with stages inventory, test-synthetic, estimate, transfer, loo and report
- `configs/g9_real/cross_demo_validation.yaml`, the predeclared values
- `slurm/`, which runs one GPU job per episode, then the summary, then the cross-demo job, plus the synthetic job

**Existing code changed:**
- `real_full_to_estimated_assets.py`: environment-variable paths, `--episodes all`, `--list-scenes`, a per-scene
  `row.json`, an atomic `SUMMARY.md`, and `--summarize-only`.
- The three real shell scripts: `FORCEARTGS_REAL_ROOT` and `FORCEARTGS_CONTACT_DIR`.
- `joint_aware_reconstruction.py`: `FORCEARTGS_OUTPUT_ROOT`, and an atomic weight copy for parallel jobs.
- `optimizer.py`: DE warm starts are placed 1e-9 of the range inside the box. scipy rejected a start clipped exactly
  onto a bound, which made the B7 fit fail on a smoke episode.

## 3. Verified

- **Tests:** 27 metrics tests (T23, T24, T33–T47 and the aggregation checks), 18 contract / engine tests (T01–T04,
  T07–T14, T16–T24, T26, T28–T30, T32, T47) and the regression tests. All pass. The two real-data tests skip with
  `REAL_TEST_NOT_RUN` when the SSD is absent.
- **Real fridge, episode 000000:**
  - Re-projecting its own wrench reproduces the training `Q_full` exactly, for revolute and for prismatic (the
    suitcase).
  - Diagonal parity: the CPU replay matches the saved metrics of the GPU training run to within 2e-6 relative.
  - Replacing the target q with random numbers leaves `q_sim` bit-identical.
  - Fridge → suitcase is refused with `KINEMATICS_TYPE_MISMATCH`.
- **SLURM task script:** run locally on `final_red_fridge_episode_000007` (smoke budget), the first episode other than
  000000. All four steps passed, giving the URDF, USD and 3DGS. The contact segments were written under the output
  root, not the repo.
- **Full pipeline report on real outputs:** 196 cells for the fridge, with 13 episodes not yet estimated and counted
  as `TRAIN_UNAVAILABLE` / `TEST_UNAVAILABLE`. The re-aggregation was `VALIDATED`.

## 4. Not done or limitations

- **No real N × N result exists yet.** Only episode 000000 of each object has been estimated. The numbers come from
  the server run.
- **Pooled leave-one-out is not implemented.** It needs a multi-episode G10 fit, which is a new fitting feature.
- **First real observation:** scored through the stationary tail, the fridge's own model drifts to −1.79 rad against
  an observed −0.66 rad. The robot keeps applying effort while the door is at rest, and the inertia + damping model
  integrates it.
- **`force_only_at_tcp_ablation`** is refused rather than substituted. It is not wired into the replay.
