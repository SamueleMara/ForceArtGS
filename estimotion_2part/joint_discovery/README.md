# `estimotion_2part/joint_discovery` — calibrated joint-aware Phase 1

Opt-in replacement for the unconstrained two-transform (`T1`, `T2`) Phase-1 motion estimate.
Static, prismatic and revolute hypotheses are fit to one shared world-frame joint trajectory
from all selected calibrated cameras. The measured TCP is treated as a sliding contact point,
and the wrench as separate physical evidence. Specification:
[`FREEARTGS_PHASE1_JOINT_AWARE_MOTION_TCP_FORCE_CODEX_SPEC.md`](../../FREEARTGS_PHASE1_JOINT_AWARE_MOTION_TCP_FORCE_CODEX_SPEC.md).
Real-data results: [`docs/joint_discovery_phase1.md`](../../docs/joint_discovery_phase1.md).

**The default is `joint_discovery.mode: legacy`.** Nothing in the legacy tracker, ICP, multiview
preparation or training calls this package. Legacy behaviour is unchanged (test T21).

## Modules

| File | Content |
|---|---|
| `rigid_kinematics.py` | Exact `Exp_SO3`, `A_R(θ) = [R, (I−R)o]`, `A_P(d)`, twist `[ω; v]`, velocities, sliding-contact maps, gauge/sign helpers. NumPy and torch (autograd). |
| `observations.py` | `CameraCalib` (per-camera `K`, fixed `T_WC`), `FrameMap` (model time → RGB-D frame → source frame → timestamp), sparse `TrackObservations`, valid-only depth sampling, dump adapter, robot-tool exclusion. |
| `candidates.py` | Static evidence, noise calibration, RANSAC finite-motion proposals, R/P seeds (including remote-hinge seeds), instantaneous rigid-flow diagnostic. |
| `assignments.py` | Per-track static/moving/outlier Student-t mixture over all of a track's observations. |
| `geometry_fit.py` | Per-track anchor Gauss-Newton, sparse robust least squares over (axis chart, hinge chart, `q_t`, anchors), Schur-complement covariance, EM driver. |
| `visual_losses.py` | Whitened 2D reprojection plus depth-axis residuals, per-camera reports, silhouette support, point-splat proxy renderer, patch NCC. |
| `model_selection.py` | Held-out time split (made before any fitting), fair fit of both hypotheses, held-out predictive selection against a static null, observability report, BIC diagnostics. |
| `tcp_contact.py` | Robot record loading (joined on source frame), canonical contact path, slip, surface support from observed moving geometry only, contact residuals, free-plane degeneracy, pin-in-slot generator. |
| `wrench_physics.py` | Sign convention (`f_obj` = robot on object), frame rotation, signed `F_∥`/`F_⊥` and ratios with exclusion reasons, `Q_H`, power, passivity, Coulomb direction with one shared bounded μ, null controls. |
| `exports.py` | `joint_candidates.json`, `joint_states.npz`, virtual part cameras `A(q)^-1 T_WC`, per-camera `transforms_{0,1}.json` with `pose_kind`, provenance. |
| `synthetic.py` | Analytic scenes for tests (ground truth is diagnostic only). |
| `depth_conflict_audit.py` | Stage A: frozen-hypothesis RGB-vs-depth audit on identical observations (anchors re-estimated with the joint fixed; no refit), per-observation export with reason codes, objective-gap attribution by camera/time/patch, systematic-error statistics, heatmaps. Reference hinges only as `reference_posthoc`. |
| `depth_evidence.py` | Stage B1: candidate-independent depth support codes (silhouette/front-surface, temporal jump, local rigidity, incidence, robot proximity, unverified). Clears `depth_valid` only; 2D always survives. |
| `validation_folds.py` | Stage C1/C2: candidate-independent surface groups, cluster-robust z and group bootstrap, grouping sensitivity, contiguous/forward time blocks and held-out-camera folds with an explicit q-at-test rule, frozen-joint predictive scoring. |
| `observability.py` | Stage C3: revolute-vs-best-translation distinguishability in units of σ_px at large motion, hinge-within-support check, Fisher check, threshold sensitivity (robustness disclosure). |
| `contact_annotations.py` | Stage D1: contact timelines (`contact_confirmed/probable/unknown`, `no_contact`) with provenance; only independent annotations confirm contact; heuristic evaluation; TCP-to-observed-object distance. |
| `wrench_calibration.py` | Stage D2/D3: versioned `WrenchCalibration` (R_EE_S, sign, tare, gravity, clock, torque origin) with DECLARED vs VERIFIED status, data checks and force-selection readiness gates. |
| `physical_selection.py` | Stage E: confirmed sliding contact and calibrated force inside candidate refinement (`fit_geometry(extra=...)`) and held-out comparison (S2/S3/S4), force goodness-of-fit gate, wrench null controls. Abstains unless gates pass. |
| `cli.py` | `synthetic`, `extract`, `from-dumps`, `fit` (ablation flags, `--fold-from`, `--experiment-id`, `--physical-selection`), `audit-depth`, `folds`, `robot-archive`, `summarize`. |

## Commands

```bash
cd ~/ForceArtGS          # repository root; use the freeartgs environment
python -m estimotion_2part.joint_discovery.cli --help

# Synthetic end to end
python -m estimotion_2part.joint_discovery.cli synthetic --kind revolute --output /tmp/jd_syn_rev

# 1) Partial-track observations from raw RGB-D (GPU AllTracker; tool removed via measured TCP/EE)
python -m estimotion_2part.joint_discovery.cli extract --scene <scene> --camera cam_001 \
    --frames 70:128:2 --tcp-npz <tcp_rgbd.npz> --output <out>/observations/obs_cam_001.npz

# 2) Vision-only fit, held-out selection, artifacts (defaults: cam_001 + cam_002)
python -m estimotion_2part.joint_discovery.cli fit --scene <scene> \
    --observations <out>/observations/obs_cam_001.npz <out>/observations/obs_cam_002.npz --output <run_dir>

# cam_000 only as an explicit stress test
python -m estimotion_2part.joint_discovery.cli fit ... --cameras cam_000 cam_001 cam_002 --observations ...

# All real-data ablations for one scene
bash pipelines/joint_discovery_real.sh <scene> 70:128:2 <tcp_rgbd.npz> outputs/joint_discovery_runs/<scene> [extract]
```

`--frames start:end:step` uses an **inclusive** end, matching the earlier diagnostic windows.
For example, `70:128:2` gives 30 frames.

## Data contracts

* **Frames.** `T_AB` maps frame B to frame A (column vectors). Twists are `[ω; v]`. Units are
  m, rad, s. Camera poses are OpenCV camera-to-world.
* **World frame.** `T_WB = I` is accepted only when every camera in `metadata/cameras.json`
  declares `coordinate_frame: base_link`; otherwise `T_WB` must be passed explicitly.
* **Observations.** There is one row per (camera, track, time) and missing rows are absent.
  `uv` is in pixels at full calibrated resolution. `z` is camera-Z in metres, NaN if invalid.
  `x_w` is filled only when depth is valid, not on an edge, and inside the whole-object mask.
  Track IDs are scoped per camera and per query window; cross-camera identity is never assumed.
* **Robot.** The robot record is joined on `source_frame_index` exactly, with no interpolation.
  The wrench is recorded as the reaction on the robot and is negated once to give `f_obj`.
  Physical force terms are disabled unless the wrench is declared tare- and gravity-compensated
  (`require_calibrated_wrench`).
* **Outputs** follow spec §14. `tracks_fate.csv` replaces the spec's Parquet because pyarrow is
  not installed in the `freeartgs` environment.

## Next-steps sprint (2026-11-17)

```bash
# staged A/B for one scene (A0 baselines, A1 depth audit, A3 ablations, C2 folds, D1 contact audit)
bash pipelines/joint_discovery_next_steps.sh <scene> 70:128:2 cam_001,cam_002
# full-episode contact windows and wrench readiness
python samuele/audit_joint_contact_windows.py --scene <scene> --clip-frames 70:128:2 --output <dir>
```

New behaviour is opt-in (`configs/joint_discovery_next_steps.example.yaml`); the defaults keep the
previous selection (`track_paired` z, `ratio` remote-hinge gate, raw depth). Results:
[`docs/joint_discovery_next_steps_results.md`](../../docs/joint_discovery_next_steps_results.md).

## What is not implemented

* Level-V2 Gaussian rendering feedback and an appearance loss (`enable_coarse_renderer`,
  `enable_appearance`). Only the patch-NCC helper exists.
* A MoMa-SG (B1) reference run. MoMa-SG was not vendored, and its revolute transform caveat
  (spec §1) is covered by independent tests.
* Downstream 3DGS training on the new candidates (Gate 7 beyond the adapter and the
  virtual-camera equivalence test).
