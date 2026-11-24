# G9 general PINN / augmented EKF and geometry-coupled mass: implementation report

This report covers the spec `FORCEARTGS_G9_GENERAL_PINN_EKF_GEOMETRY_COUPLED_MASS_CODEX.md`. It applies on top of
commit 9298f77 (WIP PINN / AEKF). Nothing has been committed or pushed.

## 1. One model for every method

The four estimators all read the same `ModelSpec` (`forceartgs_g9/estimators/model_spec.py`). Their differentiable or
filtered forms (`estimators/torch_model.py`, `general_ekf.py`) are built from that spec, never re-derived. Each method
therefore sees the same:

- interactions, native ZOH effort Q, q and q_std;
- enabled terms and parameter vector;
- gravity field and mass model;
- integral-LS initializer and held-out episodes.

The four estimators are `general_integral_ls`, `general_rollout_fit`, `general_pinn` and `general_akf`.

Equation, while moving:

    M q̈ + c q̇ + Fc sgn(q̇) + G(q) + b = Q

At rest, Karnopp stick holds when |Q − G − b| ≤ Fs.

In `geometry_coupled` mode a single density ρ (`alpha`) drives both inertia and gravity:

    M = ρ I_axis,1 + I_refl,      I_axis,1 = aᵀ J₁ a + m₁ |r⊥|²
    G(q) = −ρ m₁ g · (a × R(q) r⊥)            (revolute)
    G = −ρ m₁ g · a                           (prismatic)

There is no separate M, μ, gA or gB. Reflected actuator inertia is a separate number, never gravitational mass. The
two other parameter sources:

- `joint_space_gravity_coupled` (M, μ) is a conditional diagnostic only. Its COM direction is assumed, not measured.
- `joint_space_only` gravity is labelled `effective_configuration_load`.

## 2. GeneralPINN (`estimators/general_pinn.py`)

**Network.** A tanh MLP q_net(t) with normalised time input. q, q̇ and q̈ come from autograd in physical seconds.

**Physical parameters.** θ is shared by every episode. Positive parameters are trained in log coordinates.

**Losses:**

| loss | what it constrains | scale |
|---|---|---|
| data | q at camera times | q_std |
| IC | first observation | — |
| strong-form residual | collocation on moving segments; never on ZOH knots; excludes points near events | s_r = √(σ_Q² + (M₀σ_q̈)² + (c₀σ_q̇)²), from Savitzky–Golay noise gains |
| weak form | prefix windows from each moving-segment start: M Δq̇ + ∫(c q̇ + Fc sgn q̇ + G + b) dt = ∫Q dt | from window noise |
| impulse | momentum balance over short pulses only (≤ 0.25 s), clipped to moving segments | — |
| stick | trimmed rest runs | inequality |
| breakaway | breakaway times, 0.2 s guard | — |

**Schedule.**
1. Data warm-up: Adam with cosine LR.
2. Joint phase: physics weight ramped over the first 30 %; collocation resampled every 200 steps.

**Outputs.**
- A seed ensemble. The reported θ is the lowest-loss seed; the spread is labelled ENSEMBLE, not a posterior. With one
  seed the std is `null`.
- An open-loop replay with the frozen θ.

**Design history.** Each fix below came from a synthetic failure:
- The strong form alone is flat along c / Fc combinations, so a weak form was added.
- Pulses that included stuck time biased the impulse loss at the truth, so pulses were clipped to moving segments.
- Rest runs with creep biased the stick loss, so the runs were trimmed.

## 3. GeneralAugmentedEKF (`estimators/general_ekf.py`)

**State.** x = [q, v, η], with η in log coordinates for positive parameters and no random walk on the parameters.

**Transition.** `simulate_general` per interval, using the same hybrid Karnopp model as the rollout. Every native
force knot is a propagation step; updates happen only at camera times.

**Jacobian.** Central differences. A flag records intervals whose ± perturbations straddle a stick / slip switch.

**Process noise.** (s_Q / M₀)²·dt on v, and 0 while stuck.

**Update.** Joseph form. The filter reports NIS (mean, the 95 % bound for mean NIS, a `consistent` flag) and a
prediction-error NLL.

**Non-smooth parameter.** The static-friction excess ΔFs is not in the state, because that diverged on synthetic data
(α → 5e7, NIS 469). It is profiled on the summed EKF likelihood instead: a 13-point grid plus a 9-point refinement,
with the interval Δ(NLL) ≤ 3.84.

**Episodes and checks.**
- Episodes run sequentially and share the parameter posterior; q and v are re-initialised per episode.
- A reverse-order run reports the episode-order effect.
- A non-PSD or non-finite covariance gives the status `covariance_not_psd`, not a crash. This occurs, for example,
  with an inverted force sign: NIS 1.6e4 against 0.78 with the correct sign.

## 4. Tentative geometric mass proxy (`dynamics/geometry_proxy.py`)

No validated watertight moving-part mesh exists for the real objects. When `mass_model.mesh` is null and
`mass_model.tentative_proxy` is set, the adapter builds a proxy that is labelled **TENTATIVE_GEOMETRIC_PREVIEW**:

1. Take the moving-part Gaussian means of the frozen G8 handoff, mapped by T_Wc_P (the q = 0 pose in W).
2. Keep points with opacity sigmoid ≥ 0.5, and drop points whose 8-NN distance exceeds 3× the median.
3. Build a convex hull, oriented outward. It is watertight by construction.
4. Integrate the hull:
   - `solid_hull` = uniform volumetric density, kg/m³;
   - `shell_hull` = uniform areal density, kg/m².

Gaussian covariance and opacity are never used as density. The COM is the integrated hull centroid, never the mean
of the Gaussian centres.

The recorded limitations:
- visible surfaces only;
- the hull fills concavities;
- the real part may be hollow or heterogeneous.

The solid / shell spread is a geometry scenario, not a statistical uncertainty.

## 5. Configuration

The example config is `configs/dynamics_general_pinn_akf.example.yaml`. The new keys:

| key | values / meaning |
|---|---|
| `fitting.methods` | validated against the four method names |
| `fitting.pinn`, `fitting.ekf` | per-method options; dataclasses reject unknown keys |
| `mass_model.tentative_proxy` | `solid_hull` / `shell_hull` |
| `identifiability.declared_pulse_rows` | also set by the CLI flag `--declared-pulse-rows A:B` |

Real configs (exploratory policy; declared vertical gravity; `reflected_actuator_inertia: none_passive_mechanism`
recorded as an assumption):

- `configs/g9_real/exploratory_geo_{solid,shell}_viscous_gravity.yaml`;
- `exploratory_geo_solid_full.yaml`;
- the ablation `exploratory_gc_viscous_gravity_allmethods.yaml`.

They are run by `pipelines/g9_real_general_methods.sh` and summarised by `tools/g9_method_comparison_summary.py`.

**Comparison output.** `estimator_results/method_comparison.json` holds, per method:
- values and each method's own uncertainty, with its label;
- `resolved_by_own_uncertainty` (relative std ≤ 10 %, not at a bound, and excited per the structural check);
- an open-loop replay under one common protocol (q0 = first observation, v0 = line fit of the first 4 observations,
  native ZOH, 1 ms step), with each method's own replay kept separately;
- held-out prediction with frozen θ, NIS and the order effect.

No method is selected by training loss.

## 6. Tests

| file | what it checks |
|---|---|
| `test_g9_general_pinn.py` | autograd q̇ / q̈ of A sin ωt, and the chain rule through normalised input; full residual = 0 on an analytic trajectory, with disabled terms removed from θ; gradient of the physics loss w.r.t. one density = inertia part + gravity part; every loss finite, physics gradients reach the physical parameters and the net; no knot collocation; ZOH impulse integral and momentum balance (80 ms pulse); sliding-only data yields no breakaway constraint; recovery (α ±5 %, c / Fc ±15 %) and open-loop held-out on frozen θ |
| `test_g9_general_akf.py` | Jacobian vs the analytic viscous step; Joseph PSD; every native force step propagated with sparse 12 Hz q, and an 80 ms impulse between frames moves the state; recovery with coverage (|err| < 4σ) and NIS in [0.3, 2]; constant speed → α structurally unidentifiable; inverted force sign → NIS ≫ 1 and `consistent = false` |
| `test_g9_general_method_parity.py` | asymmetric off-axis part: I_axis,1 = aᵀJa + m₁\|r⊥\|², invariant to sliding the origin along the axis; ρ G(q) equals the rigid-body torque and −dU/dq under tilted gravity; inertial mass = gravitational mass; vertical hinge → G ≡ 0 despite an eccentric COM; prismatic shared mass (G = 0 for a horizontal axis); reflected inertia ≠ gravitational mass; shell vs solid units; tentative proxy watertight and labelled; all four methods on identical inputs share the same parameter vector `[Fc, alpha, c]`; held-out prediction unchanged when future test q is perturbed (no leakage); missing mesh → joint-space diagnostic, no tensor |

The existing G9 and joint suites are rerun as regression; results are in §8.

## 7. Real data

Filled in below from `results_19_11_26/forceartgs_joint_aware/<run>/dynamics/g9/vNNN`, as new versions; nothing was
overwritten.

The batch is `pipelines/g9_real_general_methods.sh` and the tables come from `tools/g9_method_comparison_summary.py`.

**Common settings.** Every run uses the EXPLORATORY policy: declared vertical gravity, an assumed passive mechanism
and declared calibration. None is production-exported.

**How to read the uncertainties.** Each "±" is the method's own uncertainty, and they are not comparable:
- rollout: statistical covariance;
- PINN: seed-ensemble spread (3 seeds), not a posterior. A tiny spread only means the seeds agree.
- EKF: linearised posterior.

**Replay.** The replay column uses the common open-loop protocol (§5).

### 7.1 Fridge (revolute, IDENTIFIED; one push window, source rows 220–296, 19 camera frames)

Proxy geometry, per unit density:
- solid hull: volume m₁ = 0.061 m³, I_axis,1 = 3.5e-3 m⁵;
- shell hull: area m₁ = 0.94 m², I_axis,1 = 0.069 m⁴.

| variant (version) | method | ρ (or M) | c | M [kg·m²] | gravity amp. [N·m] | replay rmse/σ | NIS |
|---|---|---|---|---|---|---|---|
| geo solid, viscous + gravity (v016) | rollout | 1.9e-12 ± 0.036 (at bound 0) | 0.027 ± 0.025 | ~0 | ~0 | 29 | — |
| | PINN | 1.44 (ens. ± 0.017) | 2.7e-4 | 5.1e-3 | 0.016 | 270 | — |
| | EKF | 2.2 ± 1.2 | 0.031 ± 0.028 | 7.6e-3 | 0.025 | 162 | 0.09 |
| geo shell, viscous + gravity (v017) | rollout | ~0 (± 5e4) | 0.027 | ~0 | ~0 | 29 | — |
| | PINN | 0.088 | 9.2e-4 | 6.1e-3 | 0.015 | 245 | — |
| | EKF | 4.9e-3 ± 4.8e-3 | 0.065 ± 0.027 | 3.4e-4 | 8.5e-4 | 57 | 0.12 |
| geo solid, full (v018) | rollout | 2.4e-7 ± 1.05 | 0.029 ± 0.040 | ~0 | ~0 | 26 | — |
| | PINN | 1.44 | 2.7e-4 | 5.1e-3 | 0.016 | 282 | — |
| | EKF | 3.0 ± 0.84 | 0.034 ± 0.030 | 0.011 | 0.034 | 186 | 0.12 |
| joint-space gravity-coupled, ablation (v019) | rollout | M 0.040 ± 0.017, μ 0.134 ± 0.059 | 0.16 ± 0.059 | 0.040 | 0.138 | 9.9 | — |
| | PINN | M 0.018, μ → 0 | 0.034 | 0.018 | ~0 | 56 | — |
| | EKF | M 0.017 ± 0.010, μ → 0 | 0.031 ± 0.029 | 0.017 | ~0 | 54 | 0.08 |

**Fridge verdict: not identified, under any method.**
- **The methods disagree.** Inertia and density range from the bound 0 (rollout) to ρ ≈ 3 (EKF). In the ablation
  they split between μ ≈ 0.13 (rollout) and μ → 0 (PINN, EKF).
- **No method's own uncertainty resolves ρ or c to within 10 %.** The tight PINN ensemble spreads are seed agreement,
  not identifiability. The PINN's open-loop replay (245–282σ) is the worst of the three.
- **NIS is far below 1 (0.08–0.12).** The declared q_std is conservative relative to the filter's innovations, so the
  EKF intervals are probably too wide rather than too narrow.
- **Geometry coupling is not supported by the data.** The best replay (9.9σ) is the joint-space ablation with a
  free mass moment. With one density the hull proxy fixes the ratio μ / M, and this one window does not support that
  ratio.

**Pulse, source rows 229–232 (declared window):**

| quantity | value |
|---|---|
| duration | 80 ms, 4 force rows, 1 camera frame inside |
| J_Q | 6.43e-3 ± 0.66e-3 N·m·s (9.7σ) |
| peak SNR | 25.6 |
| shape | oscillatory transient (net / abs impulse 0.45, one sign change) |
| q̇ before → after | −0.0017 → −0.235 rad/s |
| Δq̇ | −0.234 ± 0.096 rad/s (2.4σ; post-window χ²/dof 154) |
| sign | J and Δq̇ have opposite signs, so J / Δq̇ = −0.027 ± 0.012: a sign contradiction, reported but not corrected |
| axis tilt ±1σ (1.6°) / ±2σ (3.2°) | J changes by ±1.2 % / ±2.5 % |
| coast | SNR 1.26, decay time not resolved |
| verdict | IMPULSE_UNRESOLVED |

The pulse cannot give an inertia. It needs a cleanly resolved push: higher-rate q around the impact, or a slower,
sustained push.

### 7.2 Suitcase (TYPE_UNOBSERVABLE; working model prismatic, KINEMATICS_NOT_ACCEPTED; window 124–228)

For a prismatic joint the geometry enters only through m₁: M = ρ m₁ and G = −ρ m₁ g·a. Solid and shell therefore give
the same physical mass; only the density differs.

| variant (version) | method | ρ | M [kg] | c | gravity amp. [N] | replay rmse/σ | NIS |
|---|---|---|---|---|---|---|---|
| geo solid, viscous + gravity (v014) | rollout | 591 ± 88 kg/m³ | 8.4 | → 0 | 2.0 | 35 | — |
| | PINN | 589 | 8.3 | → 0 | 2.0 | 34 | — |
| | EKF | 535 ± 6 | 7.6 | → 0 | 1.8 | 8.8 | 8.5 |
| geo shell (v015) | all three | 22–24 kg/m² | 7.6–8.4 | → 0 | 1.8–2.0 | 9–35 | 8.5 |
| geo solid, full (v016) | all three | 535–591 | 7.6–8.4 | → 0 (Fc → 0; ΔFs 0–0.9, unbounded) | 1.8–2.0 | 9–35 | 8.5 |
| joint-space gravity-coupled (v017) | all three | M 7.6–8.4 | — | → 0 | 1.8–2.0 | 9–35 | 8.5 |

**Suitcase verdict: methodologically coherent, physically meaningless.**
- **The methods agree.** M ≈ 8 kg across all of them, with c at its bound.
- **The inputs are not trustworthy:**
  - the joint type is not accepted, so the prismatic axis is only a working model;
  - EKF NIS 8.5 is inconsistent, so the model does not explain the data;
  - adequacy fails, as in earlier runs.
- **M is effectively a gravity / load fit.** About 8 kg against a 2 N gravity amplitude is far from a plausible lid
  inertia. M is therefore a fit to the axial load, not the object's inertia.

## 8. Regression

**Passed:**
- `test_g9_general_pinn` (6 tests, including recovery, 150 s);
- `test_g9_general_method_parity` (5 tests).

**Partly verified: `test_g9_general_akf`.**
- The core tests pass, as does the prismatic recovery test.
- The sign-error test crashed on a negative covariance before the fix. After the fix that case was run by hand
  (NIS 16,000 against 0.78), but the updated test file has not been run as a whole.

**Earlier G9 suite run:** 85 tests OK (478 s).

**Joint suite:** 185 tests OK (1191 s). This suite does not depend on the later `method_comparison` change. It ran before `method_comparison` switched to the common replay protocol, so it needs
a rerun.

**Rerun with:**

    ~/miniconda3/envs/freeartgs/bin/python -m unittest discover -s tests -p "test_g9*.py"
    ~/miniconda3/envs/freeartgs/bin/python -m unittest discover -s tests -p "test_joint*.py"

Note that pytest is not installed in `freeartgs`.

## 9. Gravity geometry projection and reference-pose audit

This section covers the corrections spec `FORCEARTGS_G9_GRAVITY_PROJECTION_PINN_EKF_CORRECTIONS_CODEX.md`. The
gravity law itself was already correct (G = dU/dq), and no new gravity coefficient, mass or sign was added. What
changed is that every quantity feeding that law is now verified and audited.

### 9.1 One gravity implementation, with a proven reference pose

**The law.** `RigidBodyGravity` (`dynamics/general_1dof.py`) now carries `q_ref`, the stored joint coordinate of the
pose in which the mass geometry was measured:

    revolute : c(q) = o + R_a(q − q_ref)(c_ref − o)
               G(q) = −ρ m₁ g·(a × R_a(q − q_ref) r⊥)
    prismatic: c(q) = c_ref + (q − q_ref) a
               G = −ρ m₁ g·a

The applied generalized gravity effort is −G. `basis_coefficients()` folds q_ref into (A, B), and
`GeometryMassModel` passes q_ref to its unit-density gravity.

**Single path.** Every estimator reaches gravity through `GeometryMassModel.gravity_unit`:
- ModelSpec, which feeds the rollout and the EKF transition;
- `TorchDynamics`, for the PINN;
- the integral LS.

There is no second implementation.

**How q_ref is established.** `reference_pose_alignment` (`validation/gravity_geometry_audit.py`) checks the
handoff against the joint authority:
- The anchor is `canonical_anchor_original_frame`, an RGB-D frame id per `frame_map.json`, not a robot source row.
- The authority q at that frame must equal the handoff's `q_anchor` and the `frame_map.json` row, within 3σ.
- The handoff axis and hinge must equal the authority's.
- `T_Wc_P` must be a proper rigid transform.
- The Wc → base_link chain is taken from `cameras.json`.

The check fails closed:
- A missing anchor, missing `q_anchor`, malformed transform or any mismatch raises or returns FAIL.
- On FAIL the adapter refuses the tentative proxy.
- A user mesh needs `mass_model.reference_q`.
- Otherwise core returns `MASS_MODEL_REFERENCE_POSE_UNVERIFIED`.
- Only synthetic inputs default to 0.

**Result.** Fridge and suitcase both PASS with q_ref = 0. For the fridge:
- anchor frame 90 corresponds to source row 180, with q_auth = 0 ± 1.8e-4;
- the axis matches the handoff to 0°, and the hinge line to 0 m;
- T_B_Wc = I as declared by `cameras.json`, whose status is still OPERATOR_APPLIED_ROBOT_REGISTRATION_UNVERIFIED.

### 9.2 New per-version outputs (`dynamics/g9/vNNN/`)

| file | content |
|---|---|
| `reference_pose_alignment.json` | q_ref and its evidence, anchor frame and source row, frame_map row, T_Wc_P check, frame chain, SHA-256 hashes |
| `mass_com_inertia_audit.json` | hull bounds / components / watertightness / winding; per mode (solid m³, shell m²): m₁, c_ref, J_C,1 (3×3), r⊥, \|r⊥\|, axial COM offset, I_axis,1 (two independent computations), G₁ coefficients in stored and local q, phase, amplitude against the analytic m₁\|g×a\|ℓ, G/I ratio, G₁ on the observed q; implied density, mass, tensor, I_axis and gravity load for every method; flags TENTATIVE_GEOMETRIC_PREVIEW and PHYSICAL_MASS_UNVERIFIED |
| `gravity_joint_sensitivity.json` | scenarios with frozen θ per method: accepted joint; ±1σ / ±2σ tilt in two perpendicular directions; 12 random 3-D axis draws from the stored cone; ±1σ hinge-line shifts; an exactly vertical hinge (ALTERNATIVE_KINEMATIC_HYPOTHESIS, with its upstream residual marked NOT_RUN); ±1 cm COM registration shifts; the other mass mode with density re-matched to the same M. Axis and hinge rows REPROJECT the measured 3-D force; GRAVITY_ONLY_COUNTERFACTUAL rows keep the original effort |
| `method_gravity_parity.json` | ModelSpec vs simulator vs Torch vs reference RigidBodyGravity, for M and G at 25 random states; dr/dρ by autograd vs analytic (I_axis,1 q̈ + G₁, or m₁ q̈ − m₁ g·a) vs finite differences |
| `impulse_balance_audit.json` | per declared pulse and method: M Δq̇ + ∫(c q̇ + Fc sgn + G + b)dt − J_Q with σ; q/axis gauge check; force sign; velocity-fit χ²; lag scan of Δq̇ (±10, ±20, ±40 ms); tool-inertia hypothesis |
| `gravity_geometry_audit.json` | summary: law, g_B and its provenance class (DECLARED_OPERATOR_VERTICAL), axis, tilt, \|g×a\|, q_ref, mode, reflected inertia, force sign, flags, parity status |

### 9.3 Other corrections

- **Reuse key.** The stage's reuse signature now includes a G9 source-code fingerprint. Before this, a code change
  silently re-served old versions, so a rerun would not have produced the new audits.
- **Force multiplier.** `contact_efforts` has no default multiplier any more. The legacy raw-archive path loads it
  from `contact_force_convention.json` and fails closed if it is missing.
- **At-bound parameters.** A parameter at its numerical floor (value < 1e-9) is reported as `at_bound` in
  `method_comparison.json`: std and relative std are null, the label is AT_NUMERICAL_FLOOR, and it is never
  "resolved". A `profile_at_zero` entry replays the model with that parameter exactly 0; M = 0 is reported as
  NOT_EVALUABLE.
- **Joint-space ablation.** `joint_space_gravity_coupled` now carries the flag GRAVITY_MASS_NOT_COUPLED and states
  that `physical_gravity=True` is not a reconstructed mass.
- **Initializer label.** The rollout initializer is labelled `integral_ls_initializer`. On the fridge the
  unconstrained integral-LS ρ is −0.19 ± 0.58, which violates the physical sign, so the initializer is the
  documented fallback ρ = 1. It is not an estimate.
- **Warnings fixed.** The unclosed CSV in `robot_record.py`, and the divide-overflow warning in the identifiability
  correlation for at-bound directions.

### 9.4 Fridge (smoke v022, tiny PINN budget; full-budget numbers from v016 in §7.1)

**Geometry.** Axis a_B = (−0.0823, −0.0651, 0.9945), tilted 6.03° from vertical, so |g × a| = 1.03 m/s². The value
is neither rounded to 0 nor assumed exact.

| mode | m₁ | \|r⊥\| | axial COM | I_axis,1 | G₁ amplitude (= analytic) | G/I |
|---|---|---|---|---|---|---|
| solid hull | 0.0609 m³ | 0.180 m | −0.141 m | 3.51e-3 m⁵ | 0.0113 | 3.21 |
| shell hull | 0.941 m² | 0.181 m | −0.148 m | 6.91e-2 m⁴ | 0.175 | 2.54 |

The hull is watertight, one connected component, wound outward, with extent 0.30 × 0.66 × 0.77 m.

**Implied masses.** These are not physical masses:

| source | density | implied mass |
|---|---|---|
| full-budget PINN, solid (v016) | 1.44 kg/m³ | 0.088 kg |
| full-budget PINN, shell (v017) | 0.0878 kg/m² | 0.083 kg |
| EKF, solid | 2.2 kg/m³ | 0.13 kg |
| rollout | ~0 | ~0 (ρ at its bound) |

A real fridge door weighs kilograms. These fits are therefore not physical masses, however closely the seeds agree.

**Geometry sensitivity.** With frozen θ, the axis, hinge and COM scenarios change the gravity amplitude by roughly
−36 % to +51 % (PINN 0.0078–0.018 N·m around 0.012; EKF 0.016–0.037 around 0.025). Tilting the axis by 1σ also
changes the reprojected effort by 1–5 mN·m RMS.

**Vertical-hinge alternative.** With G ≡ 0 it replays at 48–59σ, against 108–162σ for the accepted joint with the
fitted ρ (PINN, EKF, initializer). The rollout is about 29σ under both, because its ρ is already 0. The data prefer
no gravity load at the geometry-implied G/I ratio. This is a statement about the coupled model on this single
window. It is **not** evidence for a vertical hinge: the upstream RGB-D residual of that hypothesis is NOT_RUN.

**Parity.** PASS. M and G agree to ≤ 1e-15 relative across ModelSpec, the simulator, Torch and the reference;
dr/dρ agrees between autograd, analytic and finite differences to 1e-12.

**Impulse balance, rows 229–232.** Verdict SIGN_OR_TIMING_INCONCLUSIVE.
- **Balance:** every method is off by about −10σ. The residual is dominated by J_Q itself: M Δq̇ ≈ −1e-3 and
  ∫(c q̇ + G) dt ≈ −1e-3 cannot offset +6.4e-3.
- **Evidence ruled out:**
  - the gauge is CONSISTENT (TCP tangential velocity has the sign of q̇ on 100 % of 73 contact rows);
  - the force sign is −1 from metadata, applied once;
  - a q-clock shift of ±10 to ±40 ms never flips Δq̇;
  - the tool mass that would cancel J is −24 kg (peak TCP acceleration 0.5 m/s²), so tool inertia is not the cause.
- **What remains:** an oscillatory 50 Hz transient (net / abs 0.45) seen by ≤ 1 camera frame, bracketed by an
  unreliable post-pulse velocity fit (χ²/dof 154).
- **Not done:** no sign was changed.

### 9.5 Suitcase (smoke v018)

**Reference pose.** PASS, q_ref = 0.

**Working axis.** The prismatic working axis is 88.6° from vertical, so the slider is nearly horizontal and
|g·a| = 0.24 m/s².

**Fit.** m₁ = 0.0142 m³ (solid) or 0.347 m² (shell). For a prismatic joint the mode changes only ρ, not M or G.
- **Implied mass:** 7.6–8.4 kg, at ρ 535–591 kg/m³.
- **Gravity load:** 1.8–2.0 N.

**Axis sensitivity.** Within the axis cone the gravity load spans **0.86–17.5 N**. A few degrees of slider tilt
changes the load tenfold. The ~8 kg "mass" is therefore an axial-load fit that depends on an unaccepted axis
(TYPE_UNOBSERVABLE). EKF NIS is 8.5, so the model is inconsistent with the data.

### 9.6 Tests

| file | tests |
|---|---|
| `tests/test_g9_gravity_geometry_contract.py` | 27: A (15) geometry and reference; B (12) gravity physics, including energy conservation and the forced work–energy balance |
| `tests/test_g9_gravity_pinn_ekf_parity.py` | 18: C (12) method parity; D (6) bounded, read-only real-fridge diagnostics (skipped if the SSD is not mounted) |

All 45 pass on the final code.

**Full suites.**
- G9: 130 tests OK.
- Joint: 185 tests OK.

The two warnings seen earlier (ResourceWarning from `robot_record.py`, RuntimeWarning overflow from the identifiability
correlation) no longer appear.

One caveat: the suites started before two small final edits, both in `core.py` (renaming the initializer label and the
NOT_EVALUABLE zero profile). The 45 new tests were rerun after those edits.

Rerun with:

    ~/miniconda3/envs/freeartgs/bin/python -m unittest tests.test_g9_gravity_geometry_contract tests.test_g9_gravity_pinn_ekf_parity

### 9.7 Full-budget real runs (not started here)

`pipelines/g9_real_general_methods.sh` goes through `run_g9`, which now writes all six audits. The new code
fingerprint guarantees new versions instead of reuse.

    cd ~/ForceArtGS
    nohup bash pipelines/g9_real_general_methods.sh \
      > "/media/samuelemara/Extreme SSD/Samuele/results_20_11_26/g9_gravity_geometry_audit.log" 2>&1 &
    ~/miniconda3/envs/freeartgs/bin/python tools/g9_method_comparison_summary.py \
      "/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware"

### 9.8 Full-budget batch with audits (2026-11-20 19:44–20:03; suitcase v019–v022, fridge v023–v026)

**Determinism.** The estimates are bit-identical to the previous batch (v014–v019): all methods use fixed seeds. What
is new is the six audits per version.

**Joint-space versions.** The `joint_space_gravity_coupled` versions (suitcase v022, fridge v026) have no mass
geometry. Their reference pose is NOT_RUN, and the fridge one is flagged GRAVITY_MASS_NOT_COUPLED.

**Fridge: implied moving mass** (TENTATIVE / PHYSICAL_MASS_UNVERIFIED; a real door weighs kilograms):

| version | PINN | EKF | rollout | integral-LS initializer |
|---|---|---|---|---|
| v023 solid, viscous + gravity | ρ 1.44 → **0.088 kg** | ρ 2.18 → 0.133 kg | ρ ~0 (at bound) | ρ 1 (fallback) |
| v024 shell, viscous + gravity | ρ 0.0878 → **0.083 kg** | ρ 0.0049 → 0.0046 kg | ~0 | ρ 0.0046 → 0.0043 kg |
| v025 solid, full | ρ 1.44 → 0.088 kg | ρ 3.03 → 0.185 kg | ~0 | ρ 1 (fallback) |

**Fridge: open-loop replay (rmse/σ), accepted 6.03° hinge vs the vertical-hinge alternative**

The vertical hinge is a labelled alternative only, never substituted. Its upstream residual is NOT_RUN.

| version | init. LS | rollout | PINN | EKF |
|---|---|---|---|---|
| v023 accepted / vertical | 109 / 51 | 29 / 48 | 270 / 80 | 162 / 59 |
| v024 accepted / vertical | 36 / 50 | 29 / 48 | 245 / 72 | 57 / 63 |
| v025 accepted / vertical | 142 / 138 | 26 / 43 | 282 / 186 | 186 / 92 |

**Reading the replay table.**
- **When removing gravity helps.** It helps only where the fitted ρ makes the gravity load significant (solid
  hull, PINN and EKF).
- **When it does not.** In the shell fit the density is so small that the accepted hinge replays as well as or
  better than the vertical one (36 vs 50, 57 vs 63).
- **The best fits are gravity-free.** The best geometry-coupled replay is always the rollout at ρ → 0, i.e. M → 0
  and no gravity: a purely viscous, load-free fit at 25–29σ. Still better is the joint-space ablation with an
  independent μ (v026 rollout, 9.9σ).
- **Conclusion.** On this one push window the data do not support the gravity-to-inertia ratio implied by a
  uniform hull: 3.2 (solid) or 2.5 (shell).

**Fridge: geometry-induced variability of max |G|** (frozen θ, axis / hinge / COM scenarios):

| version | PINN | EKF |
|---|---|---|
| v023 | 0.0162 N·m, range [0.0104, 0.0246] (−36 % / +52 %) | 0.0246, range [0.0157, 0.0372] |
| v025 | — | 0.0342, range [0.0219, 0.0517] |

These ranges are separate from each method's statistical uncertainty.

**Fridge: impulse rows 229–232.** Unchanged from §9.4: SIGN_OR_TIMING_INCONCLUSIVE in every version.

**Suitcase (v019–v021).**
- **Fit:** implied mass 7.6–8.4 kg for every method and mode, with c → 0 (at bound).
- **Gravity load along the near-horizontal slider:** 1.8–2.0 N.
- **Axis sensitivity:** within the axis cone the load spans 0.80–17.8 N.
- **Interpretation:** an axial-load fit on an unaccepted joint (TYPE_UNOBSERVABLE), with EKF NIS 8.5 (inconsistent).
