# G10 wrench-driven behavioral PhysX calibration — implementation report (2026-11-25)

Spec: `FORCEARTGS_G10_WRENCH_DRIVEN_BEHAVIORAL_PHYSX_CODEX_UPDATED.md`. It supersedes the conflicting parts of the
earlier behavioral replay (`docs/G10_BEHAVIORAL_PHYSX_REPLAY_IMPLEMENTATION_REPORT.md`): the effort comes from the
full six-axis wrench, contact is always on, and v0 = 0 is a given fact rather than a tested hypothesis.

> **Update (same day, from the user):**
> - **Interaction window.** Contact is ON only during the interaction, delimited by the EXISTING force-peak + visual
>   contact segments (`configs/contact_intervals/<scene>.yaml`, built earlier by `estimate_contact_segments.py`); it is
>   OFF outside.
> - **Engine.** The engine is PhysX **GPU**.
>
> Both are implemented (§11), and §2, §5 and §7 below are updated accordingly.

**Nothing was committed or pushed.** Robot archives, force exports, reconstructions, PArtTwin1D and every earlier
`vNNN` result are untouched. The real smoke outputs went to the session scratchpad.

## 1. State and code

Branch `multimodal-tcp-contact`, HEAD `7cedbd5ef613a5698e85403413b52f9f50a52d0d`. The worktree also holds the earlier
(uncommitted) behavioral-replay files.

New files:

| file | content (line numbers) |
|---|---|
| `forceartgs_g9/wrench_input/full_wrench_projection.py` (140) | `SpatialWrench` (:43): F, the moment about `ref`, and a frame / side tag. `sensor_to_base` (:78) applies the known transform and the object-side sign once, and refuses a second application. `transport` (:97). `joint_effort` (:106) implements Eqs. 1 and 2. `effort_sigma` (:122) is the linear propagation of the per-axis noise |
| `forceartgs_g9/wrench_input/full_record.py` (206) | `load_contract` (:34) parses the known contract and raises naming the missing file / array. `FullWrenchRecord` (:87). `build` (:135) produces native Q_full, Q_sigma, the legacy and alternative-origin Q, and the moment-reference diagnostic |
| `forceartgs_g9/physx_refinement/behavioral_wrench_real.py` (265) | `run` (:37): the real-run pipeline. `passive_readback` (:32) |
| `configs/g9_real/physx_behavioral_full_wrench.yaml` | the strict v2 config (`forceartgs.g10.behavioral_wrench/v2`) |
| `pipelines/g9_physx_behavioral_full_wrench_real.sh` | opt-in batch. Env: `SCENES`, `OUT_BASE`, `BEH_CONFIG`, `QUICK` |
| `tests/test_g9_full_wrench_projection.py` (12), `tests/test_g9_physx_behavioral_full_demo.py` (23), `tests/test_g9_physx_behavioral_real_adapter.py` (4) | new tests |

Extended files (all uncommitted work from the same day):
- **`behavioral_contracts.py`.**
  - `START_EXPERIMENT_KNOWN_ZERO` (:37), and the start-basis consistency check (:95).
  - The v2 schema `DEFAULT_WRENCH_CONFIG` (:371 ff.) with `BANNED_WRENCH_KEYS`.
  - `load_wrench_config` (:484) and `validate_wrench_config` (:506).
  - `wrench_to_fit_config` (:559): maps v2 onto the shared fit core, with no contact, rest or calibration option.
- **`behavioral_adapter.py`.** `build_wrench_episode` (:248) and `wrench_input_hypotheses` (:305).
- **`behavioral_report.py`.** `classify_wrench` (:320), `pulse_comparison` (:328), `wrench_candidate` (:364), `wrench_plots` (:462), `wrench_summary_md` (:566).
- **`behavioral_fit.py:427`.** Without contact labels, the probe amplitude uses the top decile of |Q| instead of contact-labelled rows.
- **`tests/g10_behavioral_fixtures.py`.** `wrench_record` builder.
- **`tests/test_g9_physx_behavioral_engine.py`.** Three wrench tests with the real engine.

Committed files changed (`git diff --stat`: +63 −3):
- `cli.py:142-176`: the new `physx-behavioral-wrench` subcommand;
- `objective.py:36`: `Episode.regime_semantics`, with a default;
- `physx_bridge.py:52-63`: `require_same_windows=True` default.

Existing CLI defaults are unchanged.

## 2. Experimental facts (given, used as such)

| fact | how it is used |
|---|---|
| E1 known sensor-to-global transform | R_BS(t) = R_base_ee(t) · R_ee_force_sensor, from `robot_tool.json/wrench_frame` (`axes: ee`, R = I, "User confirmation: force sensor axes are the same as EE axes") and the archive `T_base_ee`. It is applied once and never optimized; `sensor_to_base` refuses a base-frame input |
| E2 wrench already tared | the archive `wrench` is used **as recorded**. The force export's software baseline (`thresholds.tare_N`, 0.0001–0.03 N, re-estimated from the first 0.5 s) is not subtracted, since that would be a second tare. Only its per-axis `sigma_N` is used, as the force noise |
| E3 contact ON during the interaction | the interaction is the EXISTING force-peak + visual segment(s), read only (no new classification; the G9 classifier, labels and rest test are forbidden in tests). Inside it, every native wrench row is applied, low and zero values included. Outside it the object-side effort is exactly 0 (contact off), including the post-release coast |
| E4 qdot(t0) = 0 | `v0_fixed = 0.0`, basis `EXPERIMENT_KNOWN_ZERO`. No prefix estimate and no rest test, since the latter would challenge a given fact |
| E5 q(t0) = q_visual(t0) | q0 is the first valid authority sample in its own coordinate. On the real runs the authority defines q = 0 at its anchor frame, so q0 = 0 is the authority's value, not an imposition (tests use q0 ≠ 0) |
| E6 shared clock | t = source_row / 50 for both wrench rows and visual q; rows must be strictly increasing, with no re-timing |

**Interaction start.** t0 is the last valid visual sample at or before the first segment start. The object is at rest before contact (E4), so v0 = 0 and q0 is the visual q there. If the segment starts before the first visual sample, the status is `FULL_START_NOT_AVAILABLE` with the uncovered frames. The replay ends at the last visual sample, so the free coast after release is included at zero effort.

## 3. Record contract as read from the existing files

- **Archive.** `wrench` [N, 6] = (Fx, Fy, Fz, Tx, Ty, Tz) in N and N·m, in sensor axes (= EE axes), plus `T_base_ee`, `T_base_tcp` and `source_row` (contiguous from 0).
- **Sign.** `contact_force_convention.json`: the recorded quantity is the reaction on the robot, `object_side_force_multiplier = -1`. The object-side wrench is −1 × recorded for force and moment about the same point, applied once. The existing validator also refuses a multiplier that contradicts the recorded quantity.
- **Moment reference.** The EE frame origin p_BE = T_base_ee[:3, 3]: the wrench is in EE axes with `recorded_pose_frame: ee`. The export notes `torque_origin: recorded_origin_unspecified`, so a reading about the TCP (0.1876 m along EE z) is computed side by side as a diagnostic, never fitted.
- **Diagnostic check of that reference** (it never changes the input): regress the recorded moment on r_tcp × F in sensor axes, over rows with |F| > 0.5 N.

  | scene | τ_x slope / corr | τ_y slope / corr | rows |
  |---|---|---|---|
  | suitcase | 0.39 / 0.64 | 0.66 / 0.83 | 105 |
  | black box | 0.90 / 0.85 | 0.77 / 0.87 | 236 |
  | fridge | — | — | fewer than 10 rows |

  A slope near 1 means a moment about a point behind the TCP, i.e. the EE / sensor origin; a moment about the TCP would give a slope near 0. The data support the EE origin.
- **Noise.**
  - force: the export's per-axis σ;
  - moment: estimated as 1.4826·MAD over the first 0.5 s of the archive (a noise level, not a tare; about 1e-3 N·m), because the record has no moment noise;
  - covariance: diagonal in sensor axes, propagated through the linear projection (Monte Carlo agrees within 3 %). This is an approximation; there is no verified 6×6 covariance.

## 4. Numerical projection checks (test outputs)

| check | value |
|---|---|
| pure moment τ_S = −0.7·â, F = 0 | Q_R = **0.700** N·m (object side); the legacy TCP force-only model gives **0.0** |
| moment transport to the point (1, 2, −3) | Q = 0.108780 before and 0.108780 after (identical to 1e-12) |
| sensor-frame vs pre-transformed base-frame record | −0.126879 vs −0.126879 |
| axis and coordinate reversal | Q(−a) = −Q(a), and Q·qdot invariant |
| Q_full vs an independent recomputation on the real fridge archive | equal to 1e-12 |

## 5. Interaction input and visual coverage (real)

| scene | interaction segment (existing) | t0 (visual frame) | q0 | v0 | visual samples (post-release coast / pre-contact not replayed) | wrench rows in contact / contact-off at 0 | coverage |
|---|---|---|---|---|---|---|---|
| red fridge | 220–296 | 4.40 s (220) | visual q at 220 | 0 | 20 (0 / 10) | 77 / 0 | COMPLETE |
| orange suitecase | 124–228 | 2.48 s (124) | visual q at 124 | 0 | 30 (3 / 6) | 105 / 12 | COMPLETE |
| black box | 140–256 | 2.80 s (140) | 0 | 0 | 30 (0 / 0) | 117 / 0 | COMPLETE |
| robot dog | — | — | — | — | — | — | `DATA_UNAVAILABLE`: no joint authority (`selection.json` absent; select stage MODEL_INCONSISTENT) |
| white drawer | — | — | — | — | — | — | `DATA_UNAVAILABLE`: no robot archive (raw `metadata/raw/episode_000000.parquet` exists; the existing `robot-archive` tool could build one, not run here) |

There is one PhysX rollout per candidate from (q0, 0) over the window: every engine request is recorded in tests, and each starts at t0 and covers the full length. `interactions_from_segments`, `classify`, `build_segments`, the contact labels and the rest test are forbidden in tests; the existing segment file is only read.

## 6. Q_full vs legacy TCP force only (real; source numbers only, no sign correction)

| scene | whole-record impulse full / legacy / TCP-origin reading | moment term | row sign agreement |
|---|---|---|---|
| fridge | −0.282 / −0.131 / −0.116 N·m·s | −0.021 | 0.81 |
| suitcase (prismatic) | −4.21 / −4.19 N·s | 0 (no work on a slide) | 0.98 |
| black box | **+0.491 / −0.052** N·m·s | +0.062 | 0.82 |

**Fridge pulse rows 229–232:**

| | row 229 | row 230 | row 231 | row 232 |
|---|---|---|---|---|
| Q_full (N·m) | 0.0009 | 0.409 | −0.238 | −0.238 |
| legacy TCP force only | 0.423 | 0.092 | −0.097 | −0.097 |
| moment term of Q_full | −0.093 | 0.009 | −0.007 | −0.007 |

Pulse impulse: full −0.0013 N·m·s, legacy +0.0064, TCP-origin reading +0.0047. The per-row signs agree, but the magnitudes differ enough that the pulse impulse changes sign under the full wrench. No sign correction is applied or claimed.

For the fridge and the black box, the EE-origin full wrench differs a lot from the TCP point-force model. The recorded moment does not cancel the 0.19 m EE-to-TCP lever the way a pure tip contact would. Which reading is physically right cannot be settled from these records: the fridge has too few high-force rows for the consistency check, and the black box shows a slope below 1. The pipeline follows the contract (EE origin) and keeps the TCP reading as a diagnostic.

## 7. Real smoke results (PhysX GPU, interaction segments; quick budget: DE 3 generations x 90, lockstep LM 40 rollouts per start, rho profile only -- NOT a calibration)

| scene | status | best (rho_s; c; Td) | rmse (x range; sigma) | diagnostics |
|---|---|---|---|---|
| fridge | `BEHAVIORAL_FIT_INADEQUATE` | 0.100 (on the bound); 64.4 N m s/rad; 1.02 N m | 0.173 rad (0.53x; 327 sigma) | work audit `INPUT_OUTPUT_SIGN_INCONSISTENT` (W_in 0.008 J vs Delta U 0.025 J); zero-gravity alternative 324 sigma |
| suitcase | `BEHAVIORAL_FIT_INADEQUATE` | 0.289 (1.83 kg); 24.2 N s/m; 0 N | 2.28 cm (0.13x; 64 sigma) | passive; zero-gravity alternative **19 sigma**; rho family [0.1, 0.29] |
| black box | `BEHAVIORAL_FIT_INADEQUATE` | 0.100 (bound); 905 N m s/rad; 12.9 N m | 0.030 rad (0.62x; 67 sigma) | `INPUT_OUTPUT_SIGN_INCONSISTENT` (W_in 0.0004 J vs Delta U 0.17 J); zero-gravity alternative 64 sigma |

**Saved candidate.** The best numerical candidate is always saved (`behavioral_simulator_candidate.json`, numeric parameters, units, bounds, labels). The preview is not allowed and no candidate URDF is written, because no fit is OK. Labels: `FIT_TO_DEMO`, `PARAMETERS_UNIQUE`, `NATIVE_TARGET_ENGINE_PARITY` and `TRANSFER_VALIDATED` are all false.

**Engine.** SAPIEN 3.0.3 `PhysxGpuSystem` (CUDA, RTX 4070 Laptop) with batched sub-scenes; friction is `EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE`. Passive-term readback is identical across runs.

**Reading.**
- **Fridge and black box.** The recorded wrench, projected with the given transform and sign, cannot passively lift the parts against the declared gravity: the potential-energy gain exceeds the input work. This is a fit result for this input and geometry, not grounds to discard the demonstration.
- **Black box start.** The visual trajectory is already moving at the first frame of its interaction segment, while v0 = 0 is imposed (E4).
- **Suitcase.** It remains limited mainly by the declared 1.4 deg tilt (zero-gravity alternative 19 sigma vs 64 sigma).
- **Budget.** The quick budget understates every fit.

## 8. Tests

The freeartgs env with a scratch pytest; the engine runs through the parttwin_sim worker; nothing was installed.

| suite | result |
|---|---|
| `tests/test_g9_full_wrench_projection.py` | 12 passed (T01–T09, T18, plus noise propagation and named missing files) |
| `tests/test_g9_physx_behavioral_full_demo.py` | 23 passed (T10–T15, T13 with 15 refusal cases, T16-lite, T17, T22) |
| `tests/test_g9_physx_behavioral_engine.py` (wrench part, real SAPIEN) | 3 passed (T16 revolute + prismatic, T21) |
| `tests/test_g9_physx_behavioral_real_adapter.py` (real fridge records) | 4 passed (T01, T02, T15, T19) |
| full `tests/test_g9_*.py` regression (T20: G9 + G10 + IO + both behavioral paths + PhysX GPU, final code after the segment + GPU update) | **335 passed** (16 min 32 s; includes 3 GPU engine tests: parity, failure isolation, vectorized fit) |

Command: `PYTHONPATH=<pytest site> ~/miniconda3/envs/freeartgs/bin/python -m pytest -q tests/test_g9_*.py`.

## 9. Remaining blockers (actual)

- **Robot dog.** No joint authority.
- **White drawer.** No robot archive (one could be built from the existing raw parquet).
- **Moment reference.** The export marks the torque origin as unspecified. The EE-origin reading follows the contract and is supported by the suitcase and black-box regressions, but the fridge cannot be checked.
- **Native target engine.** Native static / dynamic / viscous axis friction needs a native PhysX ≥ 5 / Isaac runtime with readback; none is available.
- **G10-B.** The robot-action / contact backend is not required for this stage and is not implemented.
- **Transfer.** There is no independent real action, so `TRANSFER_VALIDATED` stays false.

## 10. Statement

Nothing was committed or pushed. Original assets and prior results were not overwritten. Outputs are append-only under `<run>/dynamics/g9/physx_behavioral_wrench/vNNN`; the smoke runs used the session scratchpad.

## 11. Update: interaction window and PhysX GPU

### 11.1 Interaction window from the existing segments
`behavioral_adapter.load_interaction_segments` reads `configs/contact_intervals/<scene>.yaml`. `build_wrench_episode(fr, cfg, segments)` then sets:
- **t0:** the last visual sample at or before the segment start (rest, v0 = 0);
- **Q:** Q_full inside the segment(s), exactly 0 outside (contact off);
- **end:** the last visual sample, so the post-release coast is included.

Config v2 changes:
- `experimental_facts.contact_on_during_interaction`;
- `input.interaction_segments: existing_force_peak_visual_segments`, `interaction_segments_required: true`, `outside_interaction_effort: zero_contact_off`, `include_post_interaction_coast`;
- the unused `declared_demo_start_source_frame` key is removed.

The CLI and batch script pass `--contact-segments`. Input hypotheses (clock offsets, noise) act on in-contact rows only.

### 11.2 PhysX GPU
**Worker.** `runtime_scripts/sapien_physx_worker.py: run_batch_gpu` (`kind: batch_gpu`) creates one `PhysxGpuSystem` per call, with one sub-scene per candidate (its own mass-scaled URDF), then `gpu_init`. Each step:
1. fetch qvel;
2. compute the explicit surrogate −c v − Td sat(v / v_eps) plus the recorded effort, as torch on the GPU;
3. `gpu_apply_articulation_qf`;
4. `step`.

q and qdot are fetched at the query steps. Every candidate keeps the CPU runner's passive-term zeroing and readback, plus a check that no link has a collision shape (so the replay is contact-free by construction).

**Client.** `physx_refinement/sapien_diagnostic.py: IsolatedSapienGpu` has `batch(thetas, ep)`, per-candidate inertia, armature and joint-limit checks, and isolates failures: a failing candidate returns its exception, and the rest of the batch is kept.

**Parity with PhysX CPU** (same URDF, same input, 2.4 s): |q_GPU − q_CPU| ≤ 1.1e-6 rad and |qdot| ≤ 2.3e-6 rad/s, with and without the friction surrogate. Test `test_gpu_batch_matches_cpu_physx` uses tolerances of 5e-6 / 2e-5.

**Throughput.** The GPU is launch-latency bound at about 0.3 ms per step whatever the width; the CPU takes about 9 µs per step per candidate. Measured on a 3 s demo:

| candidates per call | GPU | CPU (sequential) | speedup |
|---|---|---|---|
| 1 | 1.64 s | 0.53 s | 0.3x |
| 30 | 1.72 s | 1.30 s | 0.8x |
| 100 | 1.83 s | 3.85 s | 2.1x |
| 300 | 2.05 s | 11.5 s | 5.6x |
| 1000 | 2.79 s | 39.6 s | 14.2x |

**How the fit core uses the GPU.**
- `CachingRollout.prefetch` simulates many candidates in one call; the scoring code is unchanged.
- `RefinementLoss.objective_many` is a single evaluation per candidate after a prefetch.
- `FitConfiguration.vectorized` evaluates each DE generation as ONE batch (SciPy `vectorized=True`, `updating='deferred'`).
- `behavioral_fit.lm_lockstep` is a batched robust Levenberg–Marquardt in lockstep over all starts and all profile points of a parameter. Per iteration, ONE batch holds every Jacobian, and ONE batch holds every damping-ladder trial (7 levels).
- `_grid_starts` evaluates all profile values × a 7×7 grid of the other parameters in one batch.
- The CPU path keeps the robust TRF.

**Synthetic slide.** Truth (1.5, 2.0, 0.3). The GPU path recovers (1.498, 2.021, 0.298) at 0.60 sigma, against 0.58 on CPU, in 89 s, with 50 GPU calls. A 16x wider DE (4292 rollouts) costs the same time.

**Default.** `physics.engine: sapien_gpu_physx` with `fitting.popsize: 100` (300 candidates per GPU generation, i.e. a far wider global search at about the CPU's run time). `sapien_cpu_diagnostic` remains for parity checks.

**Native semantics are unchanged by the GPU.** Friction is still the explicit surrogate: SAPIEN 3.0.3 exposes no native axis friction on either pipeline.

## 12. Update: simplified model (gravity and Coulomb friction off)

**Model hierarchy.** The full model has four terms: inertia, viscous damping, Coulomb friction and gravity. The first simplified step keeps inertia and viscous damping and switches off Coulomb friction and gravity, so PhysX integrates

  I q'' = Q_full − c q'

from (q_visual(t0), 0). Q_full is the full six-axis wrench effort inside the existing force-peak + visual segments and 0 outside. The fitted parameters are rho_scale (axis inertia) and the viscous coefficient c; the Coulomb term Td is fixed at 0 and not searched. The input, contact window, start state, loss, engine (PhysX GPU) and outputs are identical to the full model.

**Config.** `configs/g9_real/physx_behavioral_inertia_damping.yaml`. It differs from `physx_behavioral_full_wrench.yaml` only in:

| key | full model | simplified |
|---|---|---|
| `physics.gravity` | `geometry` (G9 base config) | `zero` |
| `physics.coulomb_friction` | `fitted` | `"off"` (Td = 0) |
| `fitting.rho_bounds` | [0.1, 10] | [0.01, 100] (no gravity or Coulomb term to anchor the scale) |
| `fitting.popsize` | 100 (×3 dims) | 150 (×2 dims), 300 candidates per GPU call |
| `fitting.profile_parameters` | rho_scale, viscous, dynamic_effort | rho_scale, viscous |
| `fitting.use_optional_g9_io_warmstart` | true | false (the io fit includes gravity and Coulomb friction) |
| `diagnostics.zero_gravity_alternative` | true | false (gravity is already zero in the main fit) |

**Contract checks** (`validate_wrench_config`):
- `physics.gravity` must be `geometry` or `zero`.
- `physics.coulomb_friction` must be `fitted` or `"off"`. A bare YAML `off` parses as a boolean and is rejected with a message to quote it.
- `gravity: zero` together with the zero-gravity alternative is rejected as redundant.
- `coulomb_friction: "off"` with `dynamic_effort` in the profile list is rejected.
- `friction_mode: frictionless_physx` also removes damping, so the inertia-only model can be the next step down.

**Labels.** The simulated model is written to `engine_readback.json` (`engine.model`, `engine.gravity_source`), to the candidate (`candidate.model`, with `friction_realization` noting Coulomb friction OFF) and to `summary.md`. Example: "inertia + viscous damping (OFF: Coulomb friction, gravity); fitted: rho_scale, viscous". Provenance records gravity as "ZERO (simplified model, declared in the behavioral config; NOT a measured gravity)". With zero gravity a missing gravity vector in the G9 base config is not a blocker.

**Tests** (`tests/test_g9_physx_behavioral_simplified.py`):
- the shipped config and its quick variant load;
- the default and full configs are still the full model;
- the five inconsistent overrides are rejected;
- frictionless mode also removes damping;
- PhysX fits (CPU, and GPU when CUDA is present) on a tilted hinge with gravity off recover the truth (rho 1.5, c 2.0, Td 0) within 10 % / 15 %, with `fitted_parameters == [rho_scale, viscous]`.

**Run.**

    BEH_CONFIG=configs/g9_real/physx_behavioral_inertia_damping.yaml bash pipelines/g9_physx_behavioral_full_wrench_real.sh
