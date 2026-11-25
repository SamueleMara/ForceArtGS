# G10-A behavioral full-demonstration PhysX replay — implementation report (2026-11-25)

Spec: `FORCEARTGS_G10_BEHAVIORAL_PHYSX_REPLAY_CODEX.md`. Audited branch `multimodal-tcp-contact` at
`7cedbd5ef613a5698e85403413b52f9f50a52d0d`; the working tree held no other uncommitted code. **Nothing was committed or
pushed. No raw data, robot record, PLY, asset, PArtTwin1D file or earlier `vNNN` result was modified.**

## 1. Code paths

New files, all in `forceartgs_g9/physx_refinement/` unless noted:

| file | content |
|---|---|
| `behavioral_contracts.py` | `FullDemoBehavioralEpisode` (:51), with `v0_fixed` required to be exactly `+0.0` (no `-0.0`, bool or nonzero) and `q0_observed == q(t0)`. `.to_physx_episode()` builds ONE G10 `Episode` with `mode='forced'`, `v0=0.0`, `sigma_v0=None` and `regime_semantics='behavioral_full_demo: …'`. `InputHypothesis` (:153) refuses sign flips. The strict loader is `load_config` / `validate_config` (:292, :305): unknown keys are errors, every field is wired, and any q0 / v0 fitting is rejected |
| `behavioral_adapter.py` | `build_full_demo_episode` (:131): start-state evidence, rest test, input coverage, contact labels. `input_onset` (:34), `rest_consistency` (:47), `input_coverage` (:78), `contact_labels` (:103), `input_hypotheses` (:206). Never calls `g9_adapter.interactions_from_segments` |
| `behavioral_fit.py` | `CachingRollout` (:29) memoizes engine results. `assert_full_demo_fit` (:74) checks the contract on the objects actually given to the optimizer. `bounds_for` (:89) applies the existing G10 bounds rule. `verify_engine_start` (:142) checks the engine readback of q(t0) / qdot(t0). `local_refine` (:156) is the bounded robust TRF with restarts. `fit_full_demo` (:221) scores the baselines, runs the global DE (`fit_physx`), then the local TRF, and records the improvement and at-bound flags. Also: `profile_family` (:295), `input_sensitivity` (:348), `work_audit` (:369), `gravity_potential` (:406), `probe_episode` / `policy_spread` (:417 / :438), `zero_gravity_alternative` (:473) |
| `behavioral_report.py` | `classify` (:22), the candidate JSON (`candidate`, :34), `assert_claims_honest` (:110), the plots (:127) and `summary_md` (:248) |
| `behavioral_real.py` | the real-run pipeline `run` (:47). It reuses `real_data.reference_urdf`, `IsolatedSapienCpu`, the G9 inputs / record, the classifier (as evidence only) and the IO warm start |
| `robot_replay.py` | G10-B contract `RobotInteractionReplayBackend` (:18) and `detect_backends` (:42), which reports `NOT_IMPLEMENTED_BACKEND` with the missing ingredients |
| `configs/g9_real/physx_behavioral_full_demo.yaml` | strict start: the authority must cover the recording start |
| `configs/g9_real/physx_behavioral_rest_verified_crop.yaml` | the explicit cropped-start opt-in |
| `pipelines/g9_physx_behavioral_full_demo_real.sh` | opt-in batch. Env: `SCENES`, `OUT_BASE`, `BEH_CONFIG`, `QUICK` |
| `tests/g10_behavioral_fixtures.py` | synthetic record builder and `TestDoubleEngine`, a test double of the worker's step law labelled `TEST_DOUBLE_NOT_PHYSX` and used by unit tests only |
| `tests/test_g9_physx_behavioral_{contracts,synthetic,engine}.py` | 13 + 6 + 13 tests |

Changed files (`git diff --stat`: 3 files, +37 −3):
- **`physx_refinement/objective.py:36`.** `Episode.regime_semantics` gains a backward-compatible default `"segmented_regime_window"`. It is provenance only and does not change legacy behaviour.
- **`system_identification/physx_bridge.py:52-63`.** `initialize_physx_refinement(..., require_same_windows=True)`. The behavioral replay passes `False`, because its IO windows are segmented sub-windows of the same demo and x0 is only a scored start. The default is unchanged.
- **`cli.py:124-152`.** New `physx-behavioral-replay` subcommand. The defaults of `physx-refinement` and `io-identification` are unchanged.

Choice from spec §3.3: option 1. The full-demo episode becomes a G10 `Episode(mode='forced')`, where `forced` means "the recorded input is applied". The episode carries that meaning in `regime_semantics`, and the output says so. No method reclassifies unknown contact as free.

## 2. Initial-state contract and the real data

- **q0 and v0.** `q0 = q_authority(t0)` in the authority coordinate, never re-zeroed. `v0 = 0.0` exactly, never estimated, fitted, nuisanced or regularized. On the real runs the authority defines q = 0 at its anchor frame (`reference_pose.q_ref_source`: "joint authority q at the handoff anchor"), so q0 = 0 there is the authority's own convention, not an imposition. Tests use q0 ≠ 0.
- **Engine check.** Every fit checks the engine readback before the first step: `q(t0) == float32(q0)` and `qdot(t0) == 0.0` bit for bit (`engine_initial_state`).
- **Start basis.** It is recorded and never inferred silently:
  - `USER_DECLARED_FULL_DEMONSTRATION_INITIAL_REST`: the authority's first frame is the recording start (no q-sampling interval missing).
  - `USER_DECLARED_REST_VERIFIED_AT_CROPPED_AUTHORITY_START`: explicit opt-in config. The authority starts later, the force record covers t0, and the first authority samples pass a rest test. Otherwise the status is `FULL_START_UNAVAILABLE`, listing the uncovered source frames.
- **Rest test.** Weighted LS `q = c + v·dt + a·dt²/2`, testing H0: v = 0 at k = 3. It uses only the authority samples **before the measured input onset** (|Q| > 3σ on 3 consecutive rows) and within 0.4 s. From q alone, "already moving" cannot be told apart from "accelerating from rest under an active input", so samples after the onset are never used. Its estimate is never used as v0.
  - A q-only cubic variant was tried and rejected: it passed the real black box, which drifts 30σ in 0.4 s (z = −2.1).
- **Strict mode on an untestable start.** If the rest test is UNCHECKABLE (too few samples before the onset), strict full-recording mode accepts the start under the user contract; the cropped-start opt-in refuses it.

Evidence on the real runs (`full_demo_episode.json`):

| scene | recording start frame | authority start frame | first contact | rest test | strict | cropped-start opt-in |
|---|---|---|---|---|---|---|
| red fridge | 0 | 180 | 166 | CONSISTENT z = −0.30 (6 samples) | `FULL_START_UNAVAILABLE` (frames 0–179) | accepted |
| orange suitcase | 0 | 100 | 122 | CONSISTENT z = −1.59 (6 samples) | `FULL_START_UNAVAILABLE` (0–99) | accepted |
| black box | 0 | 140 | 122 | UNCHECKABLE: input already active at t0 (−0.35 N·m, 6.8σ) | `FULL_START_UNAVAILABLE` | refused |

## 3. Time, input and q contracts

- **Clock.** Force rows and q share the clock `source_frame_index / 50` (the canonical record). The force runs on native 50 Hz rows and q every 4th frame (12.5 Hz). Engine steps are integers from t0, with forward ZOH (the worker's existing semantics).
- **Rows replayed.** From the last row at or before t0 to the last observation. A row gap larger than 1.5 × the row period, a non-finite row, no row at t0, or a record ending early gives `INPUT_COVERAGE_INCOMPLETE` with the intervals. There is no hold, no zero fill and no extrapolation. All three real scenes have complete coverage.
- **Recorded effort.** `Q_object` is applied on every row with its canonical sign. Each row also carries a label: `CONTACT_CONFIRMED`, `MEASURED_NEAR_ZERO_CONTACT`, `CONTACT_UNKNOWN`, `RELEASE_CONFIRMED` or `NO_CONTACT_MEASURED`. Labels feed the claims and input hypotheses, never a zero substitution. The G9 classifier regimes are reported as evidence only.
- **Input hypotheses.** Only grounded ones, evaluated at the fitted θ:
  - ±1-row clock offsets, built from the full record outside the replay;
  - noise realizations from `Q_sigma`;
  - the labelled interpretation "force-absent rows are sensor noise".

  If a hypothesis moves rmse/σ by more than the tolerance, the fit is `INPUT_CONDITIONED_FIT`. Sign flips are refused by construction.

## 4. Objective and parameterization

Spec Eq. 3, implemented with the existing `RefinementLoss` over the existing `IsolatedSapienCpu`, in one continuous rollout with no resets:

    L_q(θ) = mean_i Huber_δ( (q_PX(t_i; θ, q0, 0, Q_rec) − q_obs(t_i)) / sqrt(σ_q,i² + σ_num²) ),   δ = 2, σ_num = 3e-4

- **θ.** `rho_scale` (moving-link mass and full inertia tensor scaled together, COM fixed), `viscous` c (N·m·s/rad or N·s/m) and `dynamic_effort` Td (N·m or N). `static_excess` is 0 because the surrogate has no stiction, and armature is fixed (0).
- **Bounds.** The existing G10 rule: ρ ∈ [0.1, 10] around the prior reference density, c ≤ λ_max·I_hi, Td ≤ max(γ_max·I_hi, 3·|Q|max).
- **Not optimized.** The increment and event weights are 0; events and increments are reported as diagnostics of the same fit. q0, v0, per-window resets and time-varying forces are never optimized; `assert_full_demo_fit` refuses `fit_initial_velocity`, `v0_nuisance_bound_sd ≠ 0`, `static_excess > 0`, a fitted armature and any non-behavioral episode.
- **Adequacy.** `rmse / sqrt(σ_q² + σ_num²) ≤ 3`. There is no range-relative escape and no parameter-dependent covariance to inflate (T9). The candidate status is `BEHAVIORAL_CANDIDATE_UNVALIDATED` when adequate, otherwise `BEHAVIORAL_FIT_INADEQUATE`. The replay label is `GOOD_DEMO_REPLAY[_UNDER_CONDITIONAL_INPUT]` or `DEMO_REPLAY_INADEQUATE`.

### Optimization: two engine facts found while testing

1. **The default local polish is a no-op on PhysX output.**
   - The engine keeps q in float32, and SciPy's default finite-difference step (about 1.5e-8 relative) changes q by less than its resolution. The Jacobian comes out zero, so L-BFGS-B polish, `least_squares` and Powell all stop immediately (measured: `gtol` after 4–15 evaluations).
   - A step of 1e-3 relative recovers the truth from every start, including the frictionless prior. 1e-4 already fails.
   - The new `local_refine` therefore uses a bounded robust TRF (`loss='huber'`, `f_scale = δ`, the same σ floor) with `diff_step ≥ 5e-4`; config default 1e-3, enforced by the validator.
   - This also affects the existing G10 refinement, whose DE `polish=True` uses the default step. Its polish has almost certainly been doing nothing.
2. **TRF stops early on this nonsmooth surface.**
   - On a slide that reverses under gravity (friction changes sign), TRF ended on `xtol` at rmse/σ 4.05. The loss falls monotonically from there to the truth (0.58), so this was a collapsed trust region, not a minimum.
   - Restarting from its own end point reaches 0.58 in 59 evaluations. `local_restarts` (default 3) restarts until the cost stops improving, with the budget counted in SciPy units.
   - Each local search starts strictly inside the box (1e-5 of the range), because TRF's Coleman–Li scaling freezes a variable that starts on a bound.

Sequence: score deterministic baselines → global DE (`fit_physx`, existing) from the best baseline → local TRF from the DE result and the next best baselines → keep the lowest `RefinementLoss`. Recorded: `baselines`, `optimizer` (DE, with counted failed rollouts), `local_refinement` (per start: nfev, failures, restarts), `improvement` against the frictionless prior reference, and `at_bound`.

The baselines are:
- the prior reference (ρ = 1, frictionless);
- four admissible corners (low/high inertia × low/high viscous);
- the IO warm start when an `io_response.json` exists.

The IO warm start uses `require_same_windows=False` and is scored like any other baseline.

**Near-optimal family (Eq. 6).**
- Profiles of ρ, c and Td, with the other parameters re-optimized by the local TRF.
- The family is every evaluated candidate with rmse/σ ≤ best + 0.5. This is an engineering tolerance, labelled as such, not a confidence interval.
- `not_unique` is set when a profile stays inside the tolerance over more than ×1.5 in ρ (or 50 % in a friction), when a profile reaches its grid edge, or when the best ρ, or an upper friction bound, sits on a search bound (`bound_limited`).

## 5. Backend semantics

- **Engine.** SAPIEN 3.0.3 PhysX CPU, an isolated one-joint articulation run through the existing worker in `parttwin_sim`.
- **Passive defaults.** Joint friction, drives, link damping and sleep are zeroed and read back every rollout. Unintended contacts and joint-limit hits are refused.
- **Friction.** `EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE`, applied as `-c·v - Td·sat(v/v_eps)` through `set_qf`. Static friction is not fitted.
- **G10-B.** `NOT_IMPLEMENTED_BACKEND`. The record has TCP poses but no robot command, control mode, gains or robot model, so a pose replay would only be a kinematic-path diagnostic, and none is implemented.
- **Native Isaac.** No Isaac modules are installed, and `PhysXIsaacRollout` still raises. Native target-PhysX parity is a separate milestone and is not complete.

## 6. Synthetic and real-engine results

| test (real SAPIEN unless noted) | result |
|---|---|
| T5 prismatic, 2° slide, reversal under gravity; truth ρ 1.5, c 2.0, Td 0.3 | from the frictionless prior (rmse/σ ≈ 1900): fit ≤ 3σ (truth replays at 0.59σ); rmse ratio vs prior < 0.2 |
| T5 revolute, 25° tilted hinge | same criteria, pass |
| test double, ρ 1.6, c 3, Td 0.4 | TRF recovers ρ 1.599, c 2.994, Td 0.402 (rmse/σ 0.583; truth 0.591) |
| T6 zero-input rest demo (vertical hinge) | best candidate returned, `not_unique` with a wide ρ profile; engine readback mass / inertia positive |
| T7 5-row pulse, frictionless, from rest | Δv = J/I within 0.2 %; qdot(t0) = 0.0; the reversed-sign input gives `INPUT_OUTPUT_SIGN_INCONSISTENT` |
| T8 vertical / tilted hinge, horizontal / tilted slider, q0 ≠ 0 | PhysX = semi-implicit reference within 5e-5; zero-load cases stay at q0 |
| T15 | profile members near-equal on the demo diverge under a new synthetic push (spread > 0) |

## 7. Real scenes (quick budget, outputs in the session scratchpad, NOT the results tree)

`QUICK=1`: DE 2×4, local 40 evaluations × 2 starts, ρ profile only. These are smoke runs, and the numbers are not converged fits.

| scene | FULL_DEMO_START_AVAILABLE | INPUT_COVERAGE | JOINT_AUTHORITY | BEHAVIORAL_FIT (opt-in crop) | CONDITIONAL_ASSUMPTIONS | SIMULATOR_PREVIEW | TRANSFER_VALIDATED |
|---|---|---|---|---|---|---|---|
| red fridge | no (frames 0–179 uncovered); rest-verified crop yes | COMPLETE (rows 180–296) | IDENTIFIED, not export-ready | INADEQUATE 278σ (rmse 0.148 rad / range 0.33); ρ and c bound-limited; work audit `INPUT_OUTPUT_SIGN_INCONSISTENT` | force gates unverified, gravity declared, 112/117 rows not confirmed contact, pulse rows 229–232 | no | NOT_EVALUATED |
| orange suitcase | no (0–99); rest-verified crop yes | COMPLETE (100–240) | UNOBSERVABLE (working prismatic) | INADEQUATE 30σ (rmse 1.0 cm / range 17 cm); `INPUT_CONDITIONED_FIT` (±1-row clock offset moves ±2σ); zero-gravity alternative 8.9σ | force gates unverified, gravity declared, tilt contradicted | no | NOT_EVALUATED |
| black box | no; crop refused (input active at t0) | — | OPERATOR_DECLARED | not run | — | no | NOT_EVALUATED |
| robot dog | — | — | REJECTED | `KINEMATICS_NOT_ACCEPTED` | — | no | — |
| white drawer | — | — | — | `DATA_UNAVAILABLE` (no robot record) | — | no | — |

Notes on the two fitted scenes:
- **Fridge.**
  - The door rises against the declared gravity: ΔU = 0.025 J at the lowest allowed ρ. The recorded input supplies only W_in = 0.003 ± 0.0007 J, so no passive model within the bounds can replay it with this input.
  - This matches the earlier blockers: effort SNR ≈ 1, and pulse rows 229–232 `UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE`.
  - The candidate is reported with `behavioral_preview_allowed = false`, and the sign is not changed.
- **Suitcase.**
  - Under the declared 1.4° tilt, PhysX slides the suitcase +1 cm before the push arrives, while the observed q stays flat until about 2.6 s (`plots/q_replay.png`).
  - The separately labelled zero-gravity hypothesis tracks the start, consistent with the IO `GRAVITY_GEOMETRY_CONTRADICTED` finding.
  - The quick budget understates both fits. Full-budget numbers need the batch run without `QUICK=1`
    (`pipelines/g9_physx_behavioral_full_demo_real.sh`, with
    `BEH_CONFIG=configs/g9_real/physx_behavioral_rest_verified_crop.yaml` for the cropped-start opt-in).

## 8. Tests

Environment: freeartgs (python 3.10), pytest from a scratch site; the engine runs through the parttwin_sim worker. Nothing was installed.

| run | result |
|---|---|
| `tests/test_g9_physx_behavioral_{contracts,synthetic,engine}.py` | **32 passed** (199.98 s) |
| full `tests/test_g9_*.py` regression (G9 + G10 + IO + behavioral) | **287 passed** (18 min 48 s) |
| final code (after the last edit, the at-bound tolerance): behavioral + G10 contract / loss tests | **42 passed** (184 s) |

Command: `PYTHONPATH=<pytest site> ~/miniconda3/envs/freeartgs/bin/python -m pytest -q tests/test_g9_*.py`.

## 9. Remaining blockers

- **Full start of the demonstration.** On every real run the joint authority starts at the reconstruction clip, not the recording start. A strict full-demo replay needs the authority (joint tracking) over the full recording, which means rerunning the visual front end with clip start = recording start. It was not rerun here (no 3DGS retraining).
- **Native target PhysX.** Native per-axis friction (static, dynamic, viscous) and stiction need an Isaac / PhysX ≥ 5 runtime with readback; not available.
- **G10-B.** Robot-action and contact replay needs the recorded robot command and control mode, controller gains, a robot model, object collision geometry and a contact model.
- **Independent validation.** No second real action exists per object, so `independent_real_prediction_validated` and `sim_to_real_validated` stay false.
- **Inputs and geometry.** Force calibration gates and the gravity direction remain unverified. The fridge input contradicts its motion, and the suitcase and black-box gravity geometry is contradicted.

## 10. Statement

Nothing was committed or pushed. The original assets, robot records, reconstructions, PArtTwin1D and every earlier G9 / G10 / IO `vNNN` are untouched. The real smoke outputs went to the session scratchpad.
