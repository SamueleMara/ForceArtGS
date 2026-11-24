# G9/G10 physics-constrained input-output identification — implementation report (2026-11-23)

Spec: `FORCEARTGS_G9_G10_PHYSICS_CONSTRAINED_TRANSFER_FUNCTION_CODEX.md`. Audit: `docs/G9_G10_IO_IDENTIFICATION_AUDIT.md`.
Code: `forceartgs_g9/system_identification/` (new, estimator-owned, opt-in) plus narrow G10 extensions.
Nothing was committed or pushed. PArtTwin1D was not touched.

## 1. What the layer answers

For each admitted window, the layer answers: *does the minimal passive joint equation, driven by the measured
effort, reproduce the measured q(t)?* It then asks which response features are identified, what they constrain
physically, and whether actual PhysX with those parameters reproduces the analytical response.

```
observed           q(t) samples + canonical effort u(t) on admitted FREE_CONFIRMED / FORCED_MEASURED windows
-> normalized      lam = c/M, gam = Fc/M, kappa = K_eff/M  [+ g_u = 1/M only with an informative forced window]
-> physical        M = 1/g_u, c = M lam, Fc = M gam    (point, CONDITIONAL)   |   ratio family over an ASSUMED M range
-> PhysX           conditional candidates replayed in SAPIEN/PhysX (explicit-effort friction surrogate, diagnostic)
-> readiness       SIMULATOR_READINESS (PRODUCTION_READY needs native engine parity -> never reached with SAPIEN 3)
```

## 2. Equations and where they live

| spec eq. | content | module |
|---|---|---|
| (1)–(2) | passive 1-DOF model; revolute gravity, G = dU/dq; I_h = aᵀI_COM a + m\|r⊥\|² | `physical_response.hinge_inertia`, `rigid_gravity`, `gravity_load` (analytic G′), `gravity_stiffness_uncertainty` (axis / hinge / COM Monte Carlo; density not varied) |
| (3)–(6) | local linearization; Coulomb as a signed offset with zero derivative; H_q = g_u/(s² + λs + κ), H_v = s H_q | `physical_response.linearize`, `LocalLinearModel` (refuses v̄ = 0) |
| (7)–(8) | constant-effort sliding transient with the c → 0 limit (expm1 φ-functions); terminated at the zero-velocity crossing (held at rest if \|u0 − G0\| ≤ Fc, otherwise NaN plus "breakaway model required") | `physical_response.sliding_transient` |
| (9), (12) | free normalized equation, integral form | reused: `estimators/free_weak_form` through `weak_form.py` |
| (11) | robust output error on q samples; per-episode (q0, v0) nuisance with the prefix prior; hard physical bounds | `output_error.py` |
| (13)–(14) | algebraic mapping, or an equivalence family over an ASSUMED prior | `transfer_features.physical_mapping` |
| (15) | poles, stability class (an unstable gravitational equilibrium is not clamped) | `physical_response.poles`, `classify_poles` |
| (16) | work / impulse / passivity **audit** (not a loss) | `energy_audit.py` |
| (17)–(20) | PhysX output error; v0 nuisance with prior | existing G10, plus the opt-in `FitConfiguration.v0_nuisance_bound_sd` |
| (21) | response regularizer | **refused** for weight > 0 (same q, u samples; no joint covariance with L_q); features are used as the G10 warm start only |

## 3. Methods

### Output error (primary)

**Forward model.** The normalized model runs on the validated hybrid integrator `simulate_general`. It takes exact
linear steps on every forward-ZOH interval, stops at the zero-velocity crossing, and uses Heun steps only for G(q).

**Fitted parameters.**
- g_u enters as `log_g_u`, and only when a forced window is in training, so no mass of zero or below can occur.
- Search bounds are wide: M ∈ [1e-4, 1e4].
- Each training episode has a (q0, v0) nuisance, bounded to ±10 sd and pulled toward the observed-prefix
  estimate by a quadratic prior.
- Held-out windows are replayed with θ frozen from their prefix state.

**Errors in variables.** The output-error likelihood would otherwise treat the measured effort as exact. Instead,
the recorded effort sigma is propagated by Monte Carlo through the model at the first optimum, giving an input-noise
covariance C_in(t, t′). Residuals are whitened with `chol(diag σ_q² + C_in)`, which keeps the time correlation of the
integrated input noise, and the model is refit once.

**Optimizer.** Huber loss via a C¹ square-root residual transform, solved with scipy `least_squares` (trf). Starts
are deterministic and include the weak-form estimate. Agreement between starts is checked: equivalent optima that
disagree make the result UNSTABLE.

**Profiles.**
- Every parameter is profiled; the others are re-optimised at each grid point.
- The grid is seeded at ±0.5…6 local sd, and each 95 % crossing (Δcost = 1.92·max(1, χ²_red)) is refined by
  5 bisection steps.

**Absolute-scale profile.** A dimensional reparameterization at fixed M (c, Fc and the initial states
re-optimised, G = M·a_g). Free-only data give `FLAT_SCALE_SYMMETRY`; the minimum is never taken at the grid edge or
at M = 0.

**Block bootstrap.** A moving-block residual bootstrap, labelled as a sensitivity rather than a confidence interval.

**Variants.**
- **L0 / L1:** nested, fitted on identical windows. The larger model is accepted only past the likelihood-ratio
  threshold scaled by max(1, χ²_red). Every candidate is tabulated.
- **L3:** only when listed and the configuration coverage supports it. It is reported as a *configuration-dependent
  load candidate*, not a spring.
- **L2 / L4:** recorded as NOT_RUN with the reason. Fs is never inferred from non-motion; L4 is research-only.

### Weak form, EKF and PINN (Stage C)

These run on the same source segments with the same normalized model and gravity. The EKF and PINN are initialized
from the output-error features. Pairwise |a − b| / √(sa² + sb²) > 3 is reported as INCOMPATIBLE and never averaged.
The PINN is opt-in (`compare_methods`); its seed spread stays labelled SEED_ENSEMBLE_SPREAD.

### Work / impulse audit

A joint Monte Carlo over q noise and effort noise, plus a smoothing-bandwidth term (λ_spline × 0.3 / × 3) added in
quadrature. Window edges are trimmed by one sample, because spline end derivatives are biased.

**Scale-free contradictions.**
- W_in < 0 while e_n > 0 (both significant).
- J and D_imp significant with opposite signs.
- Either one gives INPUT_OUTPUT_SIGN_INCONSISTENT.

**Other outputs.**
- A passivity bound on M.
- w_other at the fitted g_u, flagged as POSSIBLE_UNMODELED_DRIVE or UNMODELED_DISSIPATION.
- Free windows: energy gain is NONPASSIVE_UNMODELED_FORCE, or INCONSISTENT_GRAVITY_GEOMETRY when the gain exists
  only under the assumed gravity.

Lag and lever sensitivity is reported. The force sign is the canonical one and is never varied.

### Identifiability and status

**Sensitivities.** Finite-difference d q/d η, with η = log θ for positive parameters, whitened like the residuals.
Singular values, rank (including the nuisance columns) and correlations are reported; θ correlations are given
with the nuisance profiled out.

**Status dimensions** (seven, kept separate):

| dimension | values |
|---|---|
| INPUT_VALIDITY | VERIFIED / CONDITIONAL / CONTRADICTED / UNKNOWN |
| KINEMATICS_VALIDITY | ACCEPTED / CONDITIONAL / UNACCEPTED |
| STRUCTURAL_IDENTIFIABILITY | FULL / RATIO_ONLY / RANK_DEFICIENT / NOT_TESTABLE |
| PRACTICAL_IDENTIFIABILITY | INFORMATIVE / WEAK / BOUNDARY_ONLY / UNSTABLE |
| MODEL_ADEQUACY | ADEQUATE / INADEQUATE / NOT_RUN |
| INDEPENDENT_VALIDATION | PASS / FAIL / NOT_AVAILABLE |
| SIMULATOR_READINESS | PRODUCTION_READY / CONDITIONAL_PREVIEW / BLOCKED |

- "No held-out episode" gives NOT_AVAILABLE, never "unobservable".
- With no admissible window, structural identifiability is NOT_TESTABLE: nothing is claimed unobservable on the
  strength of a classifier exclusion.

### Handoff

`handoff.estimate_not_for_sim` plus `simulator_prior_family`, versioned as `forceartgs.g9.io_dynamics_handoff/v1`.
`assert_no_value_field` is applied to every written JSON, and legacy G9 `value` keys are renamed.

### PhysX bridge

**`engine_capabilities`.** SAPIEN 3.0.3 exposes the legacy `set_friction` (a dimensionless coefficient, not an
effort), `set_armature` and `set_drive_properties`. It does **not** expose
`staticFrictionEffort` / `dynamicFrictionEffort` / `viscousFrictionCoefficient`. Coverage is therefore
`DIAGNOSTIC_EXPLICIT_EFFORT_SURROGATE`.

**`initialize_physx_refinement`.**
- Refuses unless the io windows equal the G10 windows on the same clock.
- When the gain is supported, maps M → ρ_s = (M − A)/I_h(1), c = Mλ, Td = Mγ. Otherwise it uses the prior
  reference scale.
- The result is a warm start only: the G10 bounds are unchanged and no loss term is added.

**`reconcile_physx`.**
- Actual PhysX rollouts of the candidates: M from the gain and its CI ends, or members of the free-only family.
- Same initial state for PhysX and the analytical replay: the output-error fitted (q0, v0) for training windows,
  the prefix estimate for held-out windows.
- Reports the parity gap, PhysX versus observed, feature errors against the io intervals, every readback,
  capabilities, PhysX log-sensitivities, and the family trajectory spread for free-only data.

**Armature.** Held at 0. Link inertia and armature enter only through their sum (shown in the engine test below).

## 4. Engine facts (SAPIEN 3.0.3, PhysX CPU)

Every rollout zeroes and reads back the following, and fails closed if any check fails:
- joint friction, drive stiffness and damping, drive targets;
- link linear and angular damping;
- sleep threshold;
- timestep and gravity;
- moving-link mass and an SPD 3×3 inertia tensor;
- contact count (refused if > 0);
- joint-limit hits (refused).

| test | result |
|---|---|
| frictionless clean room, revolute (vertical) and prismatic (horizontal) | q̇ constant to 2e-6 relative; the loader's default joint friction (0.05) was present and zeroed |
| known prismatic pulse | Δv = J/m to 1e-4, with the predicted sign for ±J |
| tilted hinge | engine gravity torque and its finite-difference stiffness match G(q), G′(q) (0.2 % / 0.5 %) |
| free scale family (prismatic, surrogate) | members (M, Mλ, Mγ) give identical q to 2e-5; a measured force separates them by > 0.05 m |
| link inertia vs armature | vertical hinge: (ρ = 1, A = 0) and (ρ = 0.5, A = 0.5·I₁) agree to 2e-5. Tilted hinge: they differ by > 1e-3 (gravity scales with the link only) |
| io → PhysX reconciliation (prismatic, synthetic) | parity PASS; λ and g_u PhysX-equivalents inside the io 95 % intervals |

## 5. Synthetic validation

**Coverage of the profile intervals.** Revolute, two forced pulse-train windows; M = 0.8, c = 0.6, Fc = 0.3;
σ_q = 1 mrad, σ_u = 0.02 N m; 12 noise seeds.

| interval treatment | M covered | λ covered | γ covered | χ²_red |
|---|---|---|---|---|
| diagonal input-noise sd + linear interpolation on a coarse grid (first implementation) | truth excluded (CI ~0.1 % wide) | — | — | — |
| bisection-refined profile, input noise as a diagonal sd | 8/8 | 8/8 | 6/8 | ≈ 0.5 |
| **bisection-refined profile, full input-noise covariance (final)** | **12/12** | **12/12** | **12/12** | **0.83–1.26** |

The final M half-width is about ±2–3 %. Treating the measured input as exact understated it roughly 30×.

**Synthetic demo** (`python -m forceartgs_g9.system_identification.cli synthetic-demo --out DIR --physx`): two
forced windows plus one held-out free coast.

| feature | estimate | truth | 95 % interval |
|---|---|---|---|
| λ | 0.7525 | 0.75 | [0.716, 0.790] |
| γ | 0.378 | 0.375 | [0.358, 0.398] |
| g_u | 1.256 | 1.25 | [1.231, 1.283] |
| M | — | 0.8 | [0.780, 0.812], profile CLOSED |

- Status dimensions: VERIFIED / ACCEPTED / FULL / INFORMATIVE / ADEQUATE / PASS, readiness CONDITIONAL_PREVIEW.
- PhysX parity gap: 0.17–0.43 σ.
- PhysX versus observed at the identified M: 0.93σ and 1.07σ on the training windows.
- The held-out free coast, replayed from its 4-sample-prefix v0, is 27σ off (0.027 rad). The rmse/range rule still
  passes it. This is the prefix-v0 open-loop error, not a model error.

**Free-only:**
- the absolute-scale profile is flat (spread 1e-12);
- structural identifiability is RATIO_ONLY and the gain NOT_IDENTIFIED;
- λ and γ intervals cover the truth;
- the family carries the `ASSUMED` provenance;
- output error, weak form and EKF are mutually COMPATIBLE.

## 6. Findings that change how G10 should be run

1. **The fixed prefix v0 dominates the open-loop error.** A 4-sample quadratic gives σ_v0 ≈ 0.05 rad/s, which drifts
   0.04–0.08 rad over 1.6 s. G10 fixed v0 at that estimate, a plausible cause of its MODEL_INADEQUATE quick fits on
   real scenes. The opt-in `v0_nuisance_bound_sd` (with `loss.fit_initial_velocity`) fits the offset within ±3σ
   under the prior; `configs/g9_real/physx_refinement_io_init.yaml` enables it.
2. **Input noise must be in the likelihood.** Integrated twice, effort noise dominates late-window position error;
   without it the M interval was too narrow by about 30×.
3. **A lever / projection error is a pure scale error.** A fitted gain absorbs it (M doubles for a 2× lever). The
   audit detects it only against an independent mass, and says so; nothing is "corrected".
4. **Link inertia vs armature** is degenerate in the engine for a vertical hinge (shown above).

## 7. Not implemented (fail-closed)

- **Native Isaac / native PhysX axis friction.** Not exposed in SAPIEN 3.0.3 and not installed.
- **L2 static-breakaway fitting.** Needs confirmed start/stop events; none come from the classifier.
- **L4 structured residual.** Research-only.
- **Multiple shooting.** Optional in the spec; final evaluation is single-shooting open-loop replay.
- **Spectral (FFT) validation.** Not emitted; no frequency-response curve is fabricated.
- **Response-regularizer and energy-penalty weights > 0.** Refused by the config.

## 8. Tests

Run with: `PYTHONPATH=<pytest_site> ~/miniconda3/envs/freeartgs/bin/python -m pytest -q tests/test_g9_io_*.py`.

| file | tests | covers |
|---|---|---|
| `tests/test_g9_io_physics.py` | 16 | T1 TF vs ODE (5 cases, incl. underdamped and unstable); Coulomb offset; c → 0 transient vs solve_ivp and the integrator; T2 dimensional scale law; T4 stop without chatter, refused continuation; T5 vertical / tilted revolute (G, G′ vs potential, geometry MC), prismatic flat / tilted; coordinate reversal |
| `tests/test_g9_io_identification.py` | 18 | T2 RATIO_ONLY / flat / family / identical normalized fits; T3 known force (and small high-SNR effort); T4 L2 / L3 / L4 refusals; T6 sign / lag / lever audit; sign cannot be fitted away; free passivity; T7 short noisy window, no FFT, weak Coulomb/viscous separation; T10 outputs, seven dimensions, no `value`, strict config, near-zero force in contact excluded, NO_ADMISSIBLE_EPISODE, held-out frozen, append-only, G10 init mapping / window refusal; G10 v0 nuisance |
| `tests/test_g9_io_engine.py` | 8 | real engine: capabilities, T8 clean room (revolute, prismatic), prismatic pulse, tilted-hinge stiffness, T9 family / force break, armature degeneracy, io → PhysX reconciliation |

Results: see section 10.

## 9. Real scenes — commands (not run here)

The real-scene runs were **not executed in this session**: a smoke run on the suitcase was blocked by the auto-mode
classifier. Expected behaviour, from the same classifier and windows as the G10 quick run:

| scene | expected |
|---|---|
| red fridge | `NO_ADMISSIBLE_EPISODE`; the pulse audit (rows 229–232) is a blocker; the vertical-axis zero-gravity hypothesis is a separate label |
| orange suitcase | one forced window; conditional prismatic, kinematics CONDITIONAL / UNACCEPTED |
| black box | operator-declared revolute, one forced window |
| robot dog | `KINEMATICS_NOT_ACCEPTED` |
| white drawer | `DATA_UNAVAILABLE` |

```bash
cd ~/ForceArtGS
# input-output identification on every real run (analytical + a few PhysX reconciliation rollouts per scene)
nohup bash pipelines/g9_io_identification_all_real.sh \
  > "/media/samuelemara/Extreme SSD/Samuele/results_23_11_26/g9_io_identification.nohup.log" 2>&1 &
# ... then G10 warm-started from it, with the v0 nuisance (long: ~10-20 min per scene with a forced window)
THEN_G10=1 nohup bash pipelines/g9_io_identification_all_real.sh \
  > "/media/samuelemara/Extreme SSD/Samuele/results_23_11_26/g9_io_identification_then_g10.nohup.log" 2>&1 &
# summaries
cat "/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware/"*/dynamics/g9/io_identification/v*/summary.md
```

A single scene:

```bash
~/miniconda3/envs/freeartgs/bin/python -m forceartgs_g9.cli io-identification --run RUN --scene SCENE \
    --io-config configs/g9_real/io_identification_diagnostic.yaml [--pulse-rows 229:232] [--no-physx]
~/miniconda3/envs/freeartgs/bin/python -m forceartgs_g9.cli physx-refinement --run RUN --scene SCENE \
    --g10-config configs/g9_real/physx_refinement_io_init.yaml --use-io-initialization
```

`io_identification.example.yaml` (the spec's example) has `joint_source: accepted_authority`, so unaccepted
kinematics stop at `KINEMATICS_NOT_ACCEPTED`. The batch uses `io_identification_diagnostic.yaml`, which carries the
explicit `diagnostic_working_authority` opt-in; every result is then conditional.

## 10. Test results (freeartgs env, engine through the parttwin_sim worker, CPU)

| run | result |
|---|---|
| all `tests/test_g9_*.py` (G9 + G10 + io) | **249 passed** (11 min 44 s) |
| new io tests, final code (`test_g9_io_physics/identification/engine.py`) | **42 passed** (16 + 18 + 8, 100 s) |
| existing G10 tests, final code (`test_g9_physx_*.py`) | **40 passed** (17 of them in the engine, 123 s) |

The full-suite run started before the last two edits: the reconciliation's initial state and the G10 v0-nuisance
test. Both io and G10 sets were rerun on the final code (rows 2 and 3).

## 11. First real batch (2026-11-23) and the fixes it forced

The first batch was run by the user with `pipelines/g9_io_identification_all_real.sh`. The outputs are `v000` under
each run's `dynamics/g9/io_identification/`, and they predate the fixes below.

| scene | v000 outcome | problem |
|---|---|---|
| fridge | `NO_ADMISSIBLE_EPISODE`; pulse rows 229–232 blocker | none (expected) |
| suitcase | **crash** (exit 1): `SVD did not converge` | λ came back as 5e-324 at its zero bound; the log step `x(e^h − 1) = 0` gave 0/0 |
| black box | `OK`, `ADEQUATE`, `CONDITIONAL_PREVIEW` | wrong verdict, see below |
| robot dog | `KINEMATICS_NOT_ACCEPTED` | none (expected) |
| white drawer | `DATA_UNAVAILABLE` | none (expected) |

**Black box: a misspecified gravity load reported as adequate.**
- *The contradiction.* The geometry gravity gives a_g = −7.6 rad/s². Unsupported, the lid would fall about 1.2 rad
  in the 0.56 s window. It actually moved 0.0095 rad under |u| ≈ 0.1 N·m.
- *How the fit covered it.*
  - λ went to its bound (50) and g_u to 39.
  - The input-noise covariance, propagated at that large gain, grew to σ ≈ 0.02 rad.
  - A 0.044 rad misfit (4.6 × the whole motion; 147 σ_q of measurement noise) therefore whitened to 2.4 σ and passed
    the adequacy rule.
- *Why the comparisons missed it.*
  - Primary and alternative were compared by whitened cost under two different, parameter-dependent covariances.
  - L1 started neither from L0's optimum nor under its weighting. It ended *above* L0 (46 vs 4.9), which is
    impossible for a nested model.
- *Other effects.*
  - PhysX replayed members of the ASSUMED prior family (M 1.4–6.7), with c = M λ*. With a measured input those
    members are not data-equivalent (the data's 95 % M set was then [0.008, 0.76]).
  - The resulting "PhysX vs observed 153 σ" was therefore an artefact, not an engine error.

**Fixes** (all in `forceartgs_g9/system_identification/`):
1. `output_error._solve` snaps solutions within 1e-10 of a bound onto the bound.
   `identifiability.sensitivity` uses a one-sided linear step for a rate at its zero bound, and zeroes and reports
   any non-finite column (`nonfinite_columns`), which fails closed to `RANK_DEFICIENT`.
2. `fit_variants`: L1/L3 get an extra start at L0's optimum (`nested_start`) and inherit L0's input-noise
   covariance (`noise=`, `input_noise.status = INHERITED`). The likelihood-ratio test then compares the same
   weighting, and the larger variant can never end above L0.
3. Every fit records the full Gaussian NLL, including ½ log det Σ (`nll`, `half_logdet_sigma`).
   `selection.gravity_check` compares the primary gravity / geometry with the labelled alternatives by ΔNLL (same
   variant, same dimension). ΔNLL > 3 with an adequate alternative gives:
   - `CONTRADICTED`;
   - `MODEL_ADEQUACY = INADEQUATE`;
   - the blocker `GRAVITY_GEOMETRY_CONTRADICTED`.
   The alternative is never adopted automatically.
4. `replay_metrics`: a misfit larger than the whole observed motion **and** more than 3 σ_q of measurement noise is
   never adequate (`misfit_only_within_propagated_input_noise`). `rmse_over_sigma_q_measurement` is reported
   alongside.
5. `_M_grid` always spans the prior range too.
   - `physical_mapping` with forced data and a weak gain returns `DATA_CONSISTENT_FAMILY`: the members are the profile
     points inside the 95 % set, each with its own profiled c and Fc.
   - It adds a `prior_overlap` check, and `PRIOR_DATA_CONFLICT` becomes a blocker.
   - The reconciliation replays those members, and at a CI edge it uses the profiled rates at that M.

**Rerun on the fixed code** (scratch output, not written to the results tree):

| scene | dimensions | finding |
|---|---|---|
| suitcase | STRUCTURAL FULL, PRACTICAL WEAK, ADEQUACY **INADEQUATE**, READINESS **BLOCKED** | see below |
| black box | STRUCTURAL FULL, PRACTICAL BOUNDARY_ONLY, ADEQUACY **INADEQUATE**, READINESS **BLOCKED** | see below |

- **Suitcase.**
  - The primary model's 1.4° axis tilt gives a_g = −0.24 m/s². At 9–37 kg that is a 2–9 N load against a 1–3 N
    push.
  - The labelled `horizontal_slide_zero_gravity` alternative wins by ΔNLL 530. Its rates are λ 0.61 /s and
    g_u 0.027 /kg (M ≈ 37 kg, c ≈ 22 N·s/m). Its scale is not profiled.
  - The c is consistent with the earlier exploratory c ≈ 20.7 N·s/m.
- **Black box.**
  - `vertical_axis_zero_gravity` wins by ΔNLL 23: M ≈ 8.4 kg·m², λ ≈ 0, inside the assumed plate prior.
  - The primary 95 % M set is [0.008, 43] (open above) and overlaps the prior.
  - Either the declared geometry or the gravity direction is wrong, or the lid is supported by something unmeasured.
    These data cannot separate the three.

Both scenes have a single forced window and no held-out episode. The alternatives are hypotheses: the axis tilt and
the gravity support need independent evidence before either alternative becomes the primary.

Tests after these fixes: `tests/test_g9_io_*.py` + `tests/test_g9_physx*.py` → **87 passed** (7 min 1 s). This
includes 5 new regressions:
- rate at the zero bound;
- nested variant never above the minimal one;
- wrong gravity geometry contradicted (with a control);
- misfit larger than the motion is never adequate;
- forced weak-gain family taken from the profile, with the prior-overlap check.
