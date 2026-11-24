# G9/G10 input-output identification — repository audit (2026-11-23)

Audit of the **local working tree** (HEAD `9298f77` + uncommitted G9/G10 work), done before writing
`forceartgs_g9/system_identification/`. It records the local APIs the new layer depends on, the deviations from the
names the spec assumed, and the files changed.

## Toolchain

| item | value |
|---|---|
| ForceArtGS env | `~/miniconda3/envs/freeartgs` (python 3.10, numpy 1.26.0, scipy 1.15.3, torch 2.5.1+cu124); **no SAPIEN, no pytest** |
| Engine env | `~/miniconda3/envs/parttwin_sim` (SAPIEN **3.0.3**, PhysX CPU). Reached only through the existing G10 worker, never imported into freeartgs |
| Tests | pytest copied from parttwin_sim to a scratch site on `PYTHONPATH`; nothing installed in either env |

## Local APIs reused (actual signatures)

| spec name | local implementation | how it is used |
|---|---|---|
| `G9Interaction` | `estimators/model_spec.py: G9Interaction(name, joint_type, t_obs, q_obs, q_std, effort, initial_state, meta)` | built by the existing G10 bridge; not rebuilt |
| contact / regime classifier | `free_forced/segments.py: classify(rec, ...)`, `build_segments(rec, ev, ...)` → `MotionSegment` | the same options file as G9 free/forced (`configs/g9_real/free_forced.yaml`) |
| G9 → G10 episode bridge | `physx_refinement/g9_adapter.py: interactions_from_segments(...)`, `episodes_from_g9_interactions(...)` | **the** canonical path. `IOEpisode` wraps the resulting interaction, its source segment and the G10 `Episode`, so windows, clock, initial state and effort rows are identical |
| JointAuthority / inputs | `adapters/forceartgs_g9.py: build_inputs(run_root, scene, cfg, ...)` → `G9Inputs` (`axis`, `origin`, `axis_std_rad`, `origin_std_m`, `kinematic_status`, `kinematics_export_ready`, `gravity`, `mass_mesh`, `reference_pose`, `force_audit`, `record_context`) | read only |
| canonical record | `forcetwin/record.py: from_g9_inputs(inp, contact_segments=...)` | read only (`Q_object`, `Q_sigma`, `force_t_s`, `q_t_s`, ...) |
| hybrid 1-DOF integrator | `dynamics/general_1dof.py: simulate_general(p, effort, t_query, t0, q0, v0, max_step, record_events)` | exact linear steps per constant-effort ZOH interval, zero-velocity stop, Heun for G(q). The output-error model rolls the normalized equation out with it (M = 1, c = λ, Fc = γ, u → g_u·u) |
| gravity law | `general_1dof.RigidBodyGravity` (G = dU/dq, the generalized **load**) | Eq. (2); `physical_response.gravity_load` adds the analytic G′ |
| normalized gravity | `free_forced/normalized.py: GravityModel` (a_g = A cos q + B sin q; `from_mass_model`, `prismatic_ideal`, `zero`) | `a_g = G/M` with the density cancelling (no reflected inertia) |
| weak form, Eq. (12) | `estimators/free_weak_form.py: fit(segments, nspec, ...)`, `forced_scale(free, forced, nspec, ...)` | reused unchanged through `system_identification/weak_form.py` |
| normalized AEKF | `estimators/normalized_ekf.py: fit(segments, nspec, init=...)` | run on the same segments, initialized from the output-error features |
| normalized PINN | `estimators/normalized_pinn.py: fit_free(...)`, `fit_forced(...)` | opt-in via `fit.compare_methods: [..., normalized_pinn]` (slow) |
| mass geometry / size prior | `forcetwin/pipeline.py: _mass_model(inp, mode)`, `_prior_for_sim(inp, mm)` | the prior is labelled `ASSUMED` |
| G10 engine runner | `physx_refinement/sapien_diagnostic.py: IsolatedSapienCpu(urdf, *, moving_link, joint_name, dt, gravity, friction_mode, python)`; worker `runtime_scripts/sapien_physx_worker.py` | extended (below); the engine never runs in freeartgs |
| G10 objective / optimizer | `physx_refinement/objective.py: RefinementLoss`, `optimizer.py: fit_physx(objective, config, x0=...)`, `api.py: estimate(...)` | extended (below) |
| pulse audit | `validation/impulse_balance.py: audit(rec, pulse_rows, ...)` | fridge rows 229–232, reported as a blocker; nothing is flipped |
| append-only output | `forcetwin/pipeline.py: next_version(root)` | `<run>/dynamics/g9/io_identification/vNNN` |

## Deviations from the spec's assumed names

- **Exclusion labels.** The spec's `FREE_VERIFIED` / `FORCED_VERIFIED` are the local `FREE_CONFIRMED` /
  `FORCED_MEASURED`. UNKNOWN reasons map to the spec codes in `segmentation_adapter.REASON_CODE`:
  - `LOW_MEASURED_EFFORT_IN_ANNOTATED_CONTACT` → `NEAR_ZERO_FORCE_IN_CONTACT`;
  - `STICK_INTERVAL` → `STICK_INTERVAL`;
  - `EFFORT_WITHOUT_ANNOTATED_CONTACT` and `AMBIGUOUS_CONTACT` → `CONTACT_UNKNOWN`;
  - `FREE_HYPOTHESIS:*` → `HYPOTHESIS_ONLY`.
- **No `general_akf.py`.** The local file is `general_ekf.py` / `normalized_ekf.py`. The bridge uses the
  normalized EKF, which shares the normalized model.
- **Shared episode type.** `IOEpisode` is the one episode schema. It carries the G10 `Episode` built from the same
  interaction, so the two are never reconstructed independently.
- **Window difference.**
  - The output-error and PhysX windows start at the end of the 4-sample prefix used for the initial state.
  - The weak form, EKF and energy audit use the whole source segment.
  - The difference is recorded in the outputs (`window_note`).
- **Spec code layout.** The spec listed `estimator_bridge.py` (Stage C) and `synthetic_io.py` (test truth) under
  other names. They are added as separate modules.

## Hidden-damping check of the earlier parity path

`runtime_scripts/sapien_rollout.py`, the earlier G9 parity script, is **not repaired**. It leaves in place:
- SAPIEN's default link linear/angular damping (0.05);
- the sleep threshold;
- the loader's legacy joint friction (0.05).

Its results are not passive-clean. The new layer never uses it; it uses the G10 worker, which zeroes and reads back
all of these. The fix is to apply the same zero-and-read-back block.

## Changes to existing files (narrow, backward compatible)

| file | change |
|---|---|
| `runtime_scripts/sapien_physx_worker.py` | `kind: capabilities` request: probes the native axis-friction API (**not exposed in SAPIEN 3.0.3**) and the legacy friction semantics. Extra readback and fail-closed checks: timestep, gravity, drive targets, solver iterations, full 3×3 inertia (SPD), per-step contact count, sleeping state |
| `physx_refinement/sapien_diagnostic.py` | refuses a rollout with any contact or a joint-limit hit; adds `capabilities()` |
| `physx_refinement/optimizer.py` | opt-in `FitConfiguration.v0_nuisance_bound_sd`: per-training-episode v0 offset in ±k σ_v0, with the existing Gaussian prior; requires `LossSettings.fit_initial_velocity`. Default 0 leaves behaviour unchanged |
| `physx_refinement/api.py` | `estimate(..., x0=None, x0_provenance=None)` passthrough (warm start only); records `initialization` and the v0-nuisance mode |
| `physx_refinement/real_data.py` | `reference_urdf(...)` factored out of `run()` (same URDF); `run(..., io_init=None)` writes `io_initialization.json`; fitting passes `v0_nuisance_bound_sd` (default 0) |
| `cli.py` | new `io-identification` subcommand; `physx-refinement --use-io-initialization [IO_RESPONSE]` |

No PArtTwin1D file was touched or imported. No raw dataset, robot record, PLY, asset or old `vNNN` was modified.
