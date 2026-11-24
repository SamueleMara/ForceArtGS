# G9 ForceTwin adaptation: pre-edit audit (2026-11-21)

This audit was recorded before any edit for `FORCEARTGS_G9_FORCETWIN_INSPIRED_IDENTIFICATION_CODEX_SPEC.md`. It is
read-only: nothing here was changed.

## Working tree

| item | value |
|---|---|
| branch | `multimodal-tcp-contact` |
| HEAD | `9298f77ceaa235aeaf77d66b637c831200018c28` (WIP PINN and AEKF) |
| public `main` | `136c462…` (none of the work below is on it) |
| uncommitted | 33 entries: 14 modified, 19 untracked (G9 general PINN / EKF, geometry proxy, gravity-geometry audit, configs, tests, docs). All were preserved: the edits are additive |
| running jobs (not touched) | `pipelines/joint_aware_all_real.sh full …` (pid 285569) with an `ns-train twopart-blend` child (3DGS training) |
| pinned external library | `~/ForceArtGS-Physics` at `f3824316…` (clean, pinned; never modified) |

## Reused modules (sha256 prefix, line count)

| component | file | sha256 | lines |
|---|---|---|---|
| dynamics contract | `forceartgs_g9/dynamics/general_1dof.py` | ddb558c0fffb69da | 382 |
| model / parameter switches | `forceartgs_g9/estimators/model_spec.py` | 9815f4cb75ff4a59 | 260 |
| integral LS | `forceartgs_g9/estimators/general_integral_ls.py` | 7da0fda8dcfb7bb9 | 205 |
| rollout fit | `forceartgs_g9/estimators/general_constrained.py` | a56b7b407eef0602 | 238 |
| differentiable physics | `forceartgs_g9/estimators/torch_model.py` | 54849ee078442ab7 | 135 |
| GeneralPINN | `forceartgs_g9/estimators/general_pinn.py` | 0ab460fc483057cd | 383 |
| GeneralAugmentedEKF | `forceartgs_g9/estimators/general_ekf.py` | feb520503843e382 | 284 |
| mass proxy / geometry | `geometry_proxy.py` / `geometry_inertia.py` / `mass_properties.py` | 7eca5dc0… / da44b21d… / 2a67765e… | 92 / 286 / 245 |
| canonical robot record | `estimotion_2part/joint_discovery/robot_record.py` | e1a0e6445d1d959c | 237 |
| force-sign contract | `estimotion_2part/joint_discovery/force_convention.py` | 8db09adca031640c | 136 |
| wrench physics | `estimotion_2part/joint_discovery/wrench_physics.py` | 14e4564765e49e24 | 472 |
| effort / windows adapter | `forceartgs_g9/adapters/forceartgs_g9.py` | 7936118cb81d8715 | 554 |
| impulse / coast | `forceartgs_g9/validation/impulse_coast.py` | 39b483aa4c5bb8d8 | 447 |
| load models | `forceartgs_g9/validation/load_models.py` | 2e9998ce9423837c | 167 |
| gravity-geometry audit | `forceartgs_g9/validation/gravity_geometry_audit.py` | 4be2cd00a48effe7 | 649 |
| orchestration | `core.py` / `stage.py` / `cli.py` | 0044e792… / 37f10ee8… / b6ca5665… | 681 / 133 / 95 |
| exports | `export/dynamics_v2.py` / `export/final_build.py` | c56847dc… / 8a0dec32… | 235 / 165 |

## Interfaces (unchanged)

- `build_inputs(run_root, scene, cfg, *, robot_archive, contact_segments, force_export, declared_joint_type, declaration_note, log) -> G9Inputs`
  - It produces `train` (`G9Interaction`: `t_obs`, `q_obs`, `q_std`, a ZOH effort on the native 50 Hz rows, and meta
    with `source_rows` and `_record_slice`).
  - It also produces `record_context` (the canonical record, axis, origin, q on the authority clock and the
    observability).
- `ModelSpec(terms, joint_type, parameter_source in {geometry_coupled, joint_space_only, joint_space_gravity_coupled}, mass_model, gravity, …)`.
  - In `joint_space_only` with gravity off, the parameters are M, c, Fc (+ dFs). These are exactly the effective
    I_eff, b and μ.
- Estimators:
  - `general_integral_ls(its, spec)`
  - `general_rollout_fit(its, spec, theta0, …)`
  - `general_pinn_fit(its, spec, theta0, cfg)`
  - `general_akf_fit(its, spec, theta0, cfg, s_Q)`
- Impulse analysis: `impulse_coast.analyze(rec, jt, axis, origin, q_src, q, q_std, window, …, declared_pulse_rows)`.
- Output layout: `<run>/dynamics/g9/vNNN/`, append-only. The reuse signature includes a code fingerprint.

## Data availability (read-only)

| scene | joint-aware run | robot archive | contact intervals | force export |
|---|---|---|---|---|
| final_red_fridge | yes | yes | yes | `results_19_11_26/contact_annotations/…` |
| final_orange_suitecase | yes | yes | yes | yes |
| final_black_box | yes | yes | yes | yes |
| final_robot_dog | yes | yes | yes | yes |
| white_drawer_20261021_161154 | yes | **no** | **no** | **no** (no force-based estimate possible) |

## Existing tests

The `test_g9_*` suite (130 tests) and the `test_joint_*` suite (185 tests) passed on 2026-11-20 after the
gravity-projection corrections.
