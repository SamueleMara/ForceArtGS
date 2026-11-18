# Joint-aware end-to-end reconstruction: implementation report (2026-11-18)

Spec: `FORCEARTGS_END_TO_END_JOINT_AWARE_RECONSTRUCTION_CODEX_SPEC.md`. Method and call graph:
`docs/joint_aware_reconstruction.md`.

Parent HEAD is `468071f` with uncommitted work. The nerfstudio submodule is at `b1ac879` on branch
`nerfstudio-local`, with uncommitted local changes. **Nothing has been committed or pushed.**

Stage processes import the submodule through `PYTHONPATH`. The `audit` stage verifies this, because the
installed package points at `~/FreeArtGS/reconstruction/nerfstudio`, an identical commit. Each run stores the
parent and submodule diffs in `<run>/audit/`.

## What is implemented

| Spec | Implementation |
|---|---|
| §2 authority contract | `estimotion_2part/joint_discovery/integration_contract.py`: `JointAuthority`, schema, hash id, atomic write, tamper detection, exact re-gauge to `t_ref = 0`, prismatic origin `None`, legacy adapter in the blend frame, rigid frame-alignment check |
| §3 G1 dense S/M/U | `estimotion_2part/joint_discovery/dense_support.py`: cross-time RGB-D reprojection under both memberships; SUPPORTED/OCCLUDED/FREE/UNKNOWN; common-visible discriminative targets only; gradient-aware depth and colour sigmas; seed prior from confident tracks; robot tool capsule excluded; held-out images never used as targets; full-resolution masks plus scores |
| §3.6 loss semantics | submodule `models/joint_aware_losses.py`, hooked into `splatfacto.py` (`joint_aware_supervision`); supervision masks loaded by `depth_dataset.py` / `part_dataparser.py --joint-aware` |
| §4 G2 dataset | `preprocessing/prepare_joint_aware_multiview.py`: part0 static physical cameras, part1 `A(q)^-1 T_WC`, `C_S C_M^-1 = A` checked (error < 1e-12), train/val/eval lists, per-view intrinsics, no ICP or reference camera |
| §4.1 frames | no nerfstudio normalisation anywhere; the blend re-anchor `N_BW` is measured, required to be one rigid transform, and propagated to the adapter (fridge: N ≠ I, handled) |
| §6 G4 blend | `twopart_blend --joint-aware`: `obj_pose_opt` forbidden; object poses for every time come from the authority and are cross-checked against the training cameras; exported poses verified (fridge max error 6e-9) |
| §7 G5 art | `estimate_joint.py` never runs (tripwire, and it refuses joint-aware directories); `estimated_joint.json` is derived and hash-locked; axis/origin/q/offset frozen; export verified (axis change 0°, hinge shift 2e-8 m, q change 1e-8) |
| §8.3 selection (partial) | held-out interior time blocks, renders of both final branch models, paired time-block bootstrap; `IDENTIFIED` only when all significant metrics agree and there are ≥ 4 held-out times |
| §10 controller | `pipelines/joint_aware_reconstruction.py`: stages, dry run, per-budget directories, stale-resume refusal, per-stage status/time manifest; batch script `pipelines/joint_aware_real_scenes.sh` |
| tripwires | `FORCEARTGS_JOINT_AWARE_TRIPWIRE=1` makes `track_2part.py`, `segment.py`, `color_icp_gpu.py`, `prepare_multiview()` and `estimate_joint.py` raise |

## Not implemented (recorded as NOT_RUN, never PASSED)

* G6 alternating Gaussian/membership/joint refinement with the full renderer. The `refine` stage writes
  `NOT_RUN`, and joints stay at authority v000.
* G7 contact and force inside reconstruction. On real data these would abstain: there is no verified contact
  or wrench calibration.
* Static-branch rendering baseline, held-out-camera folds, SAM2 proposals, transmittance-preserving or hard
  opacity split, Gaussian-group counterfactual assignment, iterative dense re-labelling, Gaussian lineage tables.
* Final assets are exported in the blend frame B, with `N_BW` recorded. They are not re-expressed in W.

## Tests (`tests/test_joint_aware_integration.py`, 11/11 pass; joint-discovery suite 92/92 pass)

| ID | Status |
|---|---|
| I01 schema/hash/tamper | PASS |
| I02 / I07 reference re-gauge, sign flip | PASS |
| I04 virtual cameras | PASS (unit test, dataset build, and blend at runtime) |
| I05 / I06 frame firewall and adapter vs the splatfacto-art formula | PASS (rigid N; non-unit scale rejected) |
| I08 static scene | PASS (moving < 2%) |
| I09 synthetic hinge | PASS (precision M 0.97, S 0.95; recall M 0.47, rest UNKNOWN) |
| I09b wrong hypothesis does not label the static base moving | PASS |
| I13 unknown / other-part pixels get zero gradient | PASS |
| I26 tripwires | PASS |
| G2 dataset partition / cameras / authority mismatch | PASS |
| I18, I24, I38 | exercised on real data by the fridge smoke run (runtime checks) |
| I03, I10–I12, I14–I17, I19–I23, I25, I27–I37, I39, I40 | NOT_RUN |

## Real data: red fridge smoke run (300 it/stage, cam_001+cam_002, frames 90:148:2)

`/media/samuelemara/Extreme SSD/Samuele/results_18_11_26/forceartgs_joint_aware/final_red_fridge_episode_000000/`

* Dense support (revolute branch): 49% static, 33% moving, 18% unknown. 40 s and about 300 MB VRAM for both
  branches.
* Every stage passed: authority, dense, build, parts, blend, art; refine is NOT_RUN.
* Held-out times {9, 10, 11, 19, 20, 21}, rendered with the final models:

| branch | depth MAE (m) | RGB L1 | silhouette IoU |
|---|---|---|---|
| revolute | 0.0267 | 0.0393 | 0.909 |
| prismatic | 0.0339 | 0.0435 | 0.908 |

  Revolute − prismatic loss, 95% CI: depth [−0.0086, −0.0059], RGB [−0.0046, −0.0039], IoU n.s. Result:
  **IDENTIFIED revolute**, agreeing with discovery.
* Caveats: smoke budget only; held-out q comes from the transductive discovery tracks; one scene.

## Reproduction

```bash
bash pipelines/joint_aware_real_scenes.sh coarse   # 5 real scenes, both branches, 2000 it/stage
bash pipelines/joint_aware_real_scenes.sh full     # 30000 it/stage, after reviewing the coarse results
```
