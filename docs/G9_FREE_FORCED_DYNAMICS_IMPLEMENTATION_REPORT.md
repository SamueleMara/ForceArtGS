# G9 generalized free + forced dynamics: implementation report

Spec: `FORCEARTGS_G9_GENERALIZED_FREE_AND_FORCED_DYNAMICS_CODEX.md` (2026-11-22). Implemented 2026-11-23 on branch
`multimodal-tcp-contact` (HEAD `9298f77` + local work). Nothing has been committed or pushed. The 3DGS, calibrations,
sensor sign, joint axes and joint types, raw data and `ForceArtGS-Physics` are untouched.

**Real-data results in this report** come from the PINN-free batch, written to a scratch directory (§6). The real
append-only runs and the PINN are the commands in §9; they were not launched here.

---

## 1. Repository audit and code paths

The full audit is in [G9_FREE_MOTION_NORMALIZED_MODEL_REPO_AUDIT.md](G9_FREE_MOTION_NORMALIZED_MODEL_REPO_AUDIT.md). In
short, the new code is an adapter over the existing G9 stack:
- the canonical record and its single sign conversion;
- `build_inputs`, `ModelSpec`, `RigidBodyGravity` / `GeometryMassModel`, and the trajectory net `_Net`;
- the validated hybrid integrator `simulate_general`, which the normalized rollout calls with M = 1, c = λ, Fc = γ,
  G = a_g and effort = βQ.

The old full-model estimators (`forcetwin_adaptation`, `general_*`) are unchanged and remain the comparative legacy mode.

## 2. Scientific model

Moving-mode effort with constant effective M > 0:

    M q̈ + c q̇ + Fc sgn q̇ + G(q) = Q(t)

Dividing by M (a structural fact, not a fitted bound) gives

    q̈ + λ q̇ + γ sgn q̇ + a_g(q) = β Q(t),   λ = c/M [1/s],  γ = Fc/M [rad/s² | m/s²],  β = 1/M,  a_g = G/M.

**Free intervals:** β is absent from the model and from every estimator's parameter list. (M, c, Fc) → (kM, kc, kFc) is
an exact symmetry; test 1 shows identical trajectories to within 1e-7. A free fit therefore cannot collapse M to 0,
because M does not exist in the free problem.

**Forced intervals:** β > 0 enters only with a trusted measured input. M = 1/β, c = λM and Fc = γM are reconstructed
only after scale anchoring. `NormalizedDynamicsSpec.physical` refuses M ≤ 0.

**Gravity** (g_B = (0, 0, −9.81), operator-declared vertical):

| branch | a_g | notes |
|---|---|---|
| prismatic, single rigid mass | −g·a | |
| revolute, uniform-density shape | G₁(q)/I₁ = A cos q + B sin q | q_ref folded in; density cancels |
| revolute with reflected inertia | ρG₁/(ρI₁ + I_r) | refused unless ρ is given (`depends_on_scale`) |
| exactly vertical hinge | ≡ 0 | for any COM |

Hypotheses evaluated: `known_shape_{solid,shell}_hull`, `negligible_test` (admissible only if the quantitative bound
supports it), `gravity_effective_diagnostic` (signed effective load, never a second mass), `gravity_unknown` (stated,
not fitted), and the RGB-D `vertical_hinge_refit` (an alternative; it never replaces the accepted axis).

**Static breakaway Fs** is not modelled. Free coasting cannot identify it; at rest, the stick threshold is γ.

## 3. Segment evidence

Every interval gets one of three labels: FREE_CONFIRMED, FORCED_MEASURED or UNKNOWN_OR_UNMEASURED. Each carries its
reason, rows, q frames, moving fraction, J and J/σ_J, contact modes, TCP tracking and calibrated geometric separation
(`segments.classify`). The rules are in the module docstring. Two rules came from the real-data audit:

- **Geometric release is calibrated.** The hull-to-TCP distance is 2.1–2.9 cm even *during* the fridge tap. Release
  therefore means separation − separation_during_force-confirmed-contact > 3 cm. With fewer than 3 confirmed contact
  rows the test is only reported.
- **Sustained weak pushes.** Inside an annotated contact, runs shorter than 10 rows are merged (pulses with a peak above
  10σ are kept separate). A merged run is FORCED if |J|/σ_J ≥ 5. Without this, the black-box push, at about 2–3σ per
  row, fragmented into pieces with too few q samples.

**Low effort in an annotated contact is UNKNOWN, never free.** It enters the free model only under the explicitly named
hypothesis `measured_near_zero_effort_in_contact`, and every result is then capped CONDITIONAL_ON_FREE_HYPOTHESIS. The
same intervals are also re-fitted with their *measured* small effort as βQ, to show the sensitivity.

**Passivity audit (new; free motion cannot gain energy).** E = v²/2 + ∫a_g dq must not rise by more than 3σ in free
motion. A rise proves an unmeasured input or a wrong gravity model. The energised part is excluded and reported, and a
hypothesis whose entire free data are excluded is FREE_MODEL_INADEQUATE.

**q noise.** The reported q_sigma is checked against an empirical scatter estimate (pooled residuals of 5-sample local
quadratic fits). The analysis uses σ_eff = max(reported, empirical); both are recorded, and σ is never reduced.
- Fridge: reported 0.33 mrad, empirical 1.41 mrad.
- Spline penalty: chosen by the discrepancy principle (χ²_red = 1 under σ_eff), because GCV on about 15 samples chose
  near-interpolation.

**Fridge, accepted axis:**

| rows | label | reason | n_q | J [N m s] | J/σ |
|---|---|---|---|---|---|
| 180–219 | FREE_CONFIRMED | released (at rest) | 10 | −0.013 | −6.1 |
| 220–226 | UNKNOWN | LOW_MEASURED_EFFORT_IN_ANNOTATED_CONTACT | 2 | −0.0016 | −1.8 |
| 227–234 | FORCED_MEASURED | the tap (pulse, peak 25σ) | 2 | +0.0052 | +5.5 |
| 235–296 | UNKNOWN | LOW_MEASURED_EFFORT_IN_ANNOTATED_CONTACT | 16 | −0.0114 | −4.5 |

During 235–296 the TCP stays within 2.8 cm of the door point it touched while both travel about 9 cm. The robot follows
the door, so this is not confirmed free motion.

## 4. Estimators (same segments, same normalized equation, same σ)

| method | parameters | uncertainty | notes |
|---|---|---|---|
| weak form (`free_weak_form`) | λ, γ ≥ 0 (bvls); + β on forced windows | χ²-scaled covariance, profile likelihood (bisection-refined), window bootstrap, smoothing sensitivity (λ_spline ×0.3 / ×3, local polynomial) | Δv + λΔq + γsΔt + ∫a_g = βJ on non-overlapping windows. Spline end velocities; no double differentiation. SVD on whitened, column-normalised design. |
| rollout (`normalized_rollout`) | λ, γ (+β); nuisance (q0, v0) per segment | Gauss–Newton covariance | `least_squares` through `simulate_general`; 9 starts. |
| augmented EKF + RTS (`normalized_ekf`) | x = [q, v, softplus⁻¹ rates] | posterior / prior sd shrinkage reported | Semi-implicit substeps h ≤ 2 ms, analytic Jacobian (matches finite differences to 1e-6), Joseph update, NIS. Zero crossing sets v = 0, with no Jacobian of sgn at 0; sticks while \|βQ − a_g\| ≤ γ. σ_a chosen by mean NIS over a declared grid. |
| PINN (`normalized_pinn`) | phase A: λ, γ only (no β/M parameter); phase B: + β with trusted forced data | seed spread recorded but never used as an uncertainty | Per-segment trajectory = WLS quadratic baseline + existing `_Net` (3×32 tanh). Loss = data + strong form on interior collocation + integrated weak form. Adam 600 warm-up (data) + 1900 joint + L-BFGS 300. |

**Status is decided by the shared weak-form identifiability plus each method's own replay gate.** A PINN's convergence
never decides it.

**PINN fix.** Without the baseline and L-BFGS polish the PINN did not converge: data loss about 1000 in σ units and
λ off by 41%. With them it recovers λ = 0.404 and γ = 0.098 (truth 0.4 and 0.1).

**Mechanism residual fix (`forcetwin_residual.py`):**
- zero output layer, and damping heads starting at softplus(−8);
- the no-residual baseline is the incumbent in model selection;
- predictions projected orthogonal to the whitened physical regressors, so they cannot absorb q̈, sgn or q̇;
- `accepted` only with a validation improvement of at least 5%.

`legacy_init=True, orthogonalize=False` restores the old behaviour. In the new pipeline a neural normalized residual is
**disabled**: every real scene has one short segment and no independent test interval (spec §6.5). The effective-load
diagnostic is the linear stand-in.

## 5. Identifiability (synthetic, tolerances declared before the runs)

| case | result | tolerance |
|---|---|---|
| scale equivalence: (0.1, 0.04, 0.01) vs (1, 0.4, 0.1) | max \|Δq\| < 1e-7 | 1e-7 |
| weak form, two free segments (±v) | λ 0.389 [0.364, 0.415], γ 0.102 [0.090, 0.114], FREE_RATIOS_IDENTIFIED, cond 5 | 10% |
| rollout | λ 0.3997 ± 0.004, γ 0.1002 ± 0.002 | 5% |
| EKF | λ 0.421 ± 0.003, γ 0.092 ± 0.002, NIS 0.81 | 15% (passes, but sd is ~7× over-confident, see §10) |
| PINN | λ 0.404, γ 0.098 | 20% |
| forced scale, 2 free + 2 forced | M 0.506 [0.500, 0.512] (truth 0.5), c 0.198, Fc 0.051 | 10 / 15 / 20% |
| sign-flipped measured effort | SCALE_CONTRADICTED, 10/10 windows; no M produced | — |
| constant velocity | FREE_UNINFORMATIVE, upper bounds λ < 1.8e-4, γ < 9e-5 | — |
| one direction, small velocity spread | not FREE_RATIOS_IDENTIFIED | — |

## 6. Per-scene results

These come from the PINN-free batch with outputs in the scratchpad (`OUT_BASE`), at code state 2026-11-23 01:24.
Plots are in each `vNNN/plots/`.

### Red fridge (revolute, kinematics ACCEPTED)

- **Primary: accepted axis + known-shape gravity → FREE_MODEL_INADEQUATE.** The tilted (6.03°) hinge with the hull COM
  gives |a_g| up to 2.74 rad/s². Under that gravity the coast's normalized energy rises by 0.736 (rad/s)², 145σ
  (shell hull: 0.583, 115σ), which no passive model allows.
- **Gravity-scale scan.** The coast stays passive only up to 5% of the hull gravity, about 0.30° equivalent tilt for
  a fixed COM. Conditional on the free hypothesis and the hull COM, the free dynamics favour a near-vertical hinge. This
  is independent of the RGB-D track test, which is VERTICAL_NOT_DISTINGUISHABLE (held-out RMS difference 1.2 mm).
- **`negligible_test` is excluded as an analysis basis.** The bound fails (2.74 vs a resolution of 0.070 rad/s²). Its
  fit is reported only: γ 0.134, λ 0, replay 3.9 mrad (1.2% of range), beats the null, FREE_UNINFORMATIVE.
- **Analysis hypothesis: `vertical_hinge_refit/vertical_exact`** (a_g ≡ 0 exactly). It is the only admissible passive
  alternative.

| method | λ [1/s] | γ [rad/s²] | in-sample prefix replay | held-out (heads → tail) replay |
|---|---|---|---|---|
| weak form | 0 (95% ≤ 0.48) | 0.105 (95% ≤ 0.144) | 21.4 mrad, 6.7% of range; null 7.2 mrad → fails | γ 0.052: 1.8 mrad, 1.5%; null 2.4 mrad → adequate |
| rollout | ≈0 | 0.117 ± 0.18 | 23.6 mrad → fails | γ 0.084: 1.6 mrad → adequate |
| EKF | 0.142 ± 0.09 | 0.111 ± 0.10 | 28.9 mrad → fails | — |

- The combination is identified: corr(λ, γ) = −0.995, cond 20.5. Bootstrap 97.5%: λ 0.47, γ 0.126. Smoothing
  sensitivity: γ 0.096–0.120.
- EKF NIS 0.84; posterior/prior sd 0.45 (λ), 0.62 (γ).
- **Status: FREE_MODEL_INADEQUATE**, capped CONDITIONAL_ON_FREE_HYPOTHESIS and CONDITIONAL_ON_ALTERNATIVE_HYPOTHESIS.
  The coast decelerates mostly in its last ~0.3 s, which a constant γ cannot reproduce from the start. Only a
  γ-dominated combination of about 0.05–0.13 rad/s² is supported.
- **Absolute scale: CONTRADICTED** (§7). **Simulator: INSUFFICIENT_PHYSICAL_DATA.**
- Measured-Q alternative (the coast's own −4.5σ effort as βQ): β = 0 ± 6.5, so NOT_OBSERVABLE.

### Orange suitecase (prismatic; kinematics UNOBSERVABLE → WORKING_ONLY)

- No free segment with ≥ 5 q samples; after release the slider moves only 2 mm. Normalized result: FREE_UNINFORMATIVE.
- **Forced push, rows 122–229:** 27 q samples, J = −4.16 N s (591σ).
- **Under the working axis's axial gravity** (a_z = −0.024, a_g = −g·a): β CI [0.071, 0.212], so M = 8.7 kg
  [4.7, 14.1] with λ = γ = 0. Rollout gives 8.3 kg and the EKF 7.2 kg (NIS 1.31).
- **Under negligible gravity:** the β CI reaches 0.
- **Verdict: the scale is carried by the gravity hypothesis.** For a one-direction quasi-static push, M·a_g·Δt and
  Fc·sΔt are collinear, so the apparent ~8 kg comes from the assumed 1.4° tilt, not from inertia. The status is
  downgraded to ABSOLUTE_SCALE_NOT_OBSERVABLE.
- The forced replay is no better than the null (36 mm, 21% of range). Quasi-static view: M 4.3 ± 5.2, c 9.8 ± 36,
  Fc 0 ± 4.2 (cond 29), so nothing is identified.

### Black box (revolute, OPERATOR_DECLARED)

- Forced segment 191–228 (10 q samples); the other forced runs have too few samples. Low-effort segment 229–288
  (7 q samples) is used only under the free hypothesis.
- **Known-shape gravity:** energy gain 0.079 (rad/s)², 2594σ → FREE_MODEL_INADEQUATE. (Under the declared 85°-tilted
  axis, the hull gravity cannot be balanced by the measured near-zero effort.)
- **Negligible gravity:** passive, but FREE_UNINFORMATIVE.
- **Forced:** β CI [0, 146] → NOT_OBSERVABLE. EKF NIS 2501 (inconsistent). Forced replay 0.49 rad against a motion
  range of 0.015 rad → the forced model is inadequate. Quasi-static cond 3e4. The push is under-excited.

### Robot dog / white drawer

- Robot dog: KINEMATICS_NOT_ACCEPTED (REJECTED, MODEL_INCONSISTENT). Readiness audit only; no q(t) or force is
  fabricated.
- White drawer: DATA_UNAVAILABLE (no canonical robot record or interaction window).

## 7. Fridge pulse audit (rows 229–232)

- **Impulse audit** (`impulse_balance.audit`, unchanged): UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE.
- **Bracket row, accepted-axis record:** J = +0.00643 ± 0.00066 N m s (9.8σ), D = −0.578 ± 0.006 rad/s.
- **Bracket row, vertical-refit record:** J = +0.00591 ± 0.00067, D = −0.386 ± 0.018. Here D = Δv + λΔq + γ∫sgn + ∫a_g,
  using the free rates.
- **D/J = −65 to −90.** The signed impulse and the velocity change contradict, and no positive M explains them.
- **Verdict: PULSE_NOT_USED_FOR_SCALE.** No sign was flipped, and no mass or friction is derived from the pulse.

The free-motion analysis is a separate result (§6). It gives only a combination bound, and its in-sample replay fails.

## 8. Simulator export

`simulator_handoff_preview.json` is typed:
- normalized λ and γ carry an `estimate` only when RATIO_IDENTIFIED (otherwise `estimate_not_for_sim`, plus status,
  interval and caps);
- `absolute_scale.M.estimate` is set only when FORCE_IDENTIFIED;
- `simulator.readiness` is one of PHYSICAL_IDENTIFIED / PRIOR_SCALED_PREVIEW / INSUFFICIENT_PHYSICAL_DATA / BLOCKED.

A PRIOR_SCALED_PREVIEW lists positive scenarios from a log-uniform size prior (c = λM, Fc = γM) with one consistent hull
mass distribution, labelled NOT IDENTIFIED. Engine notes cover URDF, MJCF, SAPIEN/ManiSkill and Isaac; ENGINE_REPLAY is
NOT_RUN. **On the real scenes every readiness is INSUFFICIENT_PHYSICAL_DATA, so no physics values are exported.**

The legacy forcetwin `simulation_handoff.json` no longer has a `value` field: `estimate` is present only when
identified, otherwise there is `estimate_not_for_sim`.

## 9. Tests, commands and runtime

**Tests:**
- `tests/test_g9_free_forced.py`: 20 tests, all passing (spec §9 cases 1–15, plus passivity, noise floor, weak-push
  merge and the ModelSpec bridge).
- G9 regression (`tests/test_g9*.py`): 168 tests, all passing, 579 s.
- Joint-discovery suite: not rerun; no joint-discovery code changed in this phase.

```bash
~/miniconda3/envs/freeartgs/bin/python -m unittest tests.test_g9_free_forced -v                    # ~130 s
~/miniconda3/envs/freeartgs/bin/python -m unittest discover -s tests -p "test_g9*.py"             # ~10 min
```

**Real runs** (append-only, `<run>/dynamics/g9_free_forced/vNNN`):

```bash
cd ~/ForceArtGS
# all scenes without the PINN (~1 min)
bash pipelines/g9_free_forced_all_real.sh
# with the normalized PINN (phase A + held-out refit; phase B where trusted forced data exist; several minutes)
PINN=1 nohup bash pipelines/g9_free_forced_all_real.sh \
  > "/media/samuelemara/Extreme SSD/Samuele/results_23_11_26/g9_free_forced_pinn.nohup.log" 2>&1 &
# aggregate (the batch also writes this at the end)
~/miniconda3/envs/freeartgs/bin/python tools/g9_free_forced_comparison.py \
  "/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware" \
  "/media/samuelemara/Extreme SSD/Samuele/results_23_11_26/free_forced_comparison"
```

**Runtime:** about 25 s per scene without the PINN on 22 CPU cores (62 GB RAM). Python 3.10.20, numpy 1.26.0,
scipy 1.15.3, torch 2.5.1 (CPU). The synthetic PINN takes about 35 s per seed for two segments of ~70 samples.

## 10. Limitations and next experiment

**Limitations:**
- **EKF intervals are not calibrated.** In the synthetic test the error is 5–8% at ~7σ. Identifiability therefore never
  comes from the EKF covariance.
- **Every real "free" interval is a hypothesis.** All five real demos keep the robot in annotated sliding contact
  through the motion, so none has confirmed release. The pipeline says so and caps every result.
- **The in-sample replay depends on the prefix state.** With about 15 samples at 12.5 Hz, the prefix fit is short
  (5 samples). The held-out split and the in-sample replay disagree for the fridge; both are reported.
- **The gravity scan assumes the hull COM.** A COM nearer the hinge would allow a larger tilt.

**What would identify the missing quantities:**
- **Confirmed release:** the tool retracts by more than 3 cm beyond the contact offset, or a separate contact sensor
  shows release, followed by a coast of at least 1 s with velocity spread (or two opposite-direction coasts). That
  separates λ from γ under confirmed-free conditions.
- **Several distinct pushes with real acceleration** (Δv well above σ_Δv), timed independently of the visual clock.
  That observes β; the quasi-static pushes recorded so far do not.
- **A gravity-free check of the hinge:** an independent hinge-axis measurement or a level reference, to settle
  accepted 6° versus near-vertical.
- **For the suitcase:** an accepted joint type and a two-direction push, to separate axial gravity from Coulomb
  friction.
