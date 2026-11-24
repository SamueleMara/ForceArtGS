# ForceArtGS G9: Generalized Free-Motion + Forced-Motion Dynamics Identification

**CODEX IMPLEMENTATION DIRECTIVE — 2026-11-22**  
**Repository:** `~/ForceArtGS`, actual local working tree (including unpublished/local G9 work).  
**Deliverable:** implemented estimators, integrated pipeline, numerical/structural tests, real-scene diagnostic output, append-only run artifacts, and a detailed audit report. **Do the implementation; do not stop at a design memo.**  
**Scientific basis:** ForceTwin (Engelbracht et al., 2026), particularly the effective joint-effort model, separation of simulator-compatible coefficients and optional mechanism residual, and excitation requirements. This specification *adds* an explicit **structural-scale observability separation** and mandatory PINN / augmented EKF, not claimed to be methods in ForceTwin.  
**Primary decision:** treat **free motion as informative about *normalized* joint dynamics**, while recognizing that pure unforced trajectory data cannot identify the **absolute inertia scale**. Never interpret low force as no dynamic information, and never fit zero inertia simply to minimize a homogeneous force residual.

> **Do not retrain 3DGS, rerun calibrations, change sensor sign, silently override joint axes/types, edit the frozen `ForceArtGS-Physics` library, or commit/push without authorization.** Preserve all existing local G9 changes. Do not launch long real-data training as a hidden/background task; provide explicit run commands and report what completed in the active session. Every scientific claim must be supported by machine-readable results.

---

## 0. Why this replacement is necessary

Existing G9 runs report, among other issues:

- Fridge: force after the push is near the sensor noise floor, yet the visually reconstructed `q(t)` shows motion; general estimators drive inertia/density toward zero, give poor open-loop replay (hundreds to tens of thousands of `q_sigma`), and label all physical parameters unidentifiable.
- The fridge's force pulse at native source rows **229–232** has a reported signed generalized impulse of about `0.00643 N m s` (about `9.75 sigma`), while the estimated angular-velocity change has the **opposite sign** and low significance (about `2.4 sigma`). A +/-40 ms timestamp sweep did not resolve it. This observation must remain unresolved until evidence explains it; no automatic sign flip.
- The red fridge is approximately a vertical revolute joint (`a ~= [-0.082, -0.065, 0.994]` from the existing authority, 6.03 degrees off declared vertical). The true hinge may be closer to vertical; gravity is a *hypothesis-dependent* small/large torque, not a reason to zero the mass.
- Suitcase has an unaccepted working joint; black box currently uses an operator-declared revolute hypothesis; do not convert these into certified physical conclusions.
- The new ForceTwin-inspired effort residual may fit `Q ~ 0` even if the predicted motion is completely wrong; a residual-only training score does not establish dynamics.

**The algebraic failure of a raw free-motion fit:**

For free motion of a nearly gravity-neutral revolute joint,

\[
I\ddot q+c\dot q+F_c\operatorname{sgn}(\dot q)=0.
\]

Fitting the homogeneous left side with `I,c,Fc >= 0` admits the **trivial optimum** `(0,0,0)` for any `q(t)`. This is a **parameterization / objective pathology**, not evidence that the moving body is massless. When input torque is zero, division by `I>0` exposes the observable ratios:

\[
\ddot q + \lambda\dot q+\gamma\operatorname{sgn}(\dot q)=0,
\qquad
\lambda=c/I,\quad \gamma=F_c/I.
\]

The observed free trajectory may constrain `lambda`, `gamma`, and other **normalized** loads, if the trajectory contains enough deceleration/variation. It cannot distinguish a family `(I,c,Fc) -> (kI,kc,kFc)`, `k>0`, without an externally known absolute scale.

**Universal rule:** never let missing / insignificant external effort collapse the fitted absolute inertia to zero. Do not assume the ratios themselves are identifiable merely because the optimizer converges: constant speed, extremely short trajectories, almost zero deceleration, and unknown mechanism/gravity can also confound the *ratios*.

## 1. Read and preserve the current local implementation FIRST (mandatory P0)

Inspect `git status --short`, branch/HEAD, current configs and all local modifications before editing. Public `main` may be older than the actual working tree; prior audits found `main` at `136c462...` without the later local PINN/EKF and ForceTwin work. Expected files, **if present**:

| Function | Existing implementation to inspect/extend |
|---|---|
| Canonical robot/force record | `estimotion_2part/joint_discovery/robot_record.py` |
| Contact force / sign contract | `estimotion_2part/joint_discovery/wrench_physics.py`, `force_convention.py` if present |
| G9 input adapter / windows | `forceartgs_g9/adapters/forceartgs_g9.py` |
| Shared dynamics | `forceartgs_g9/dynamics/general_1dof.py` |
| Model terms and gravity | `forceartgs_g9/estimators/model_spec.py` |
| Differentiable model | `forceartgs_g9/estimators/torch_model.py` (local) |
| General PINN | `forceartgs_g9/estimators/general_pinn.py` (local) |
| General augmented EKF | `forceartgs_g9/estimators/general_ekf.py` / current equivalent (local) |
| Integral estimator | `forceartgs_g9/estimators/general_integral_ls.py` |
| Rollout estimator | `forceartgs_g9/estimators/general_constrained.py` |
| Mechanism network | `forceartgs_g9/estimators/forcetwin_residual.py` (local) |
| Geometry to inertials | `forceartgs_g9/dynamics/geometry_inertia.py`, `mass_properties.py`, geometry proxy module if present |
| Identifiability and adequacy | `forceartgs_g9/validation/general_identifiability.py`, `model_adequacy.py`, `impulse_coast.py` if present |
| Orchestration/export | `forceartgs_g9/core.py`, `config.py`, `stage.py`, `cli.py`, `export/dynamics_v2.py`, `export/final_build.py` |
| Reports and runs | `docs/G9_GENERAL_PINN_AKF_IMPLEMENTATION_REPORT.md`, `docs/G9_CORRECTIONS_AUDIT.md`, `results_19_11_26/forceartgs_joint_aware/` on SSD |

Create `docs/G9_FREE_MOTION_NORMALIZED_MODEL_REPO_AUDIT.md` with actual signatures, current behavior, applicable variants, input/output schemas, SHA256 of critical input assets, existing tests, and files you will modify. **Reuse native classes and JSON schemas instead of implementing a parallel physics stack.** All changes additive/config-gated; legacy code paths must preserve old behavior.

## 2. A single generalized dynamics contract for BOTH joint types (P0)

### 2.1 Physical effort model and units

Let `q(t)` be the signed measured joint coordinate. For ideal 1-DoF rigid articulation, moving-mode generalized effort is

\[
M(q)\ddot q + C(q,\dot q) + F_c\operatorname{sgn}(\dot q)
   +G(q)+\tau_{\mathrm{mech}}(q,\dot q)=Q_{\mathrm{external}}(t).
\]

For the minimal G9 model use **constant effective** `M>0`, `C=c qdot` and `Fc>=0`. Revolute: `M=I_h [kg m^2]`, `Q [N m]`, `c [N m s/rad]`, `Fc [N m]`. Prismatic: `M=m_eff [kg]`, `Q [N]`, `c [N s/m]`, `Fc [N]`. Preserve optional terms, but **start with minimal separately identifiable quantities**. For joints with variable inertia or transmissions, explicitly identify the modeled terms; never silently pretend the constant-`M` model covers them.

For a **sticking** interval, use the existing set-valued/hybrid condition `qdot=0`, `qddot=0`, `|Q-G-tau_mech| <= Fs`; friction sign at exactly zero velocity is not defined as `Fc*sign(0)`. In free-moving intervals, use continuous moving-mode equations and remove transition points from strong-form collocation; integrate carefully across event boundaries.

### 2.2 Generalized input is robot-on-object, and only if actually known

Do not change the existing **single negative reaction-to-applied-force sign conversion**:

\[
F^B_{\mathrm{object}}=sR_{B,EE}R_{EE,S}F^S_{\mathrm{tare\ corrected}},\qquad s=-1
\]

for recordings explicitly labeled `object_on_robot`. If sign declaration missing/invalid: stop with data error; no silent `-1` fallback. Do not re-tare/re-align existing exports or silently consider missing wrench as `Q=0`. For a revolute contact point `p` and hinge `(a,o)`, with an independently known contact wrench couple if available,

\[
Q_R=a^T[(p-o)\times F_{\mathrm{obj}}+M_{\mathrm{contact}}],
\qquad Q_P=a^T F_{\mathrm{obj}}.
\]

If using TCP as contact point, **label it as a point-force approximation**; quantify lever-arm uncertainty and sliding contact. Account for tool-assembly dynamic wrench only when estimated from evidence; otherwise bound/report. A high resultant force norm is NOT necessarily a high signed generalized effort. Keep native 50 Hz wrench samples; use synchronized source-frame / q observation associations without inventing timing precision.

### 2.3 Three distinct external-effort states

Implement explicit interval labels and confidence/provenance, not a single `contact` boolean:

1. `FREE_CONFIRMED`: sensor noise-consistent effort, AND contact-free / released according to independent contact annotations (or credible geometric separation), AND no known external support driving the part. `Q` is a **random variable centered near zero**, not a claimed exact zero. Bound any unknown force or tool load. The policy must not automatically infer `FREE_CONFIRMED` solely from low measured force.
2. `FORCED_MEASURED`: active contact with sufficiently established object-side generalized effort/time association for conditional scale identification; ensure sign and lever are auditable. Do not use arbitrary `SNR>3` alone: observability requires the input to be distinguishable from nuisance loads and to generate a measurable change in state.
3. `UNKNOWN_OR_UNMEASURED`: ambiguous contact, absent wrench, unverified input geometry, or unsupported stick/slip. Do not treat unknown intervals as force-free. Retain for position-only diagnostics but **exclude from physical parameter identification** unless an explicitly labeled latent-input model is used (which cannot identify absolute scale without extra evidence).

Intervals are contiguous in actual physical time; separately report overlap with observation q frames and wrench rows, confidence of q and contact state, and geometry hypothesis. Do not artificially connect separated intervals or interpolate through missing data without recording the interpolation.

### 2.4 Shared visual q(t) measurement model

Use the **existing 3DGS/RGB-D-supervised joint coordinate** and its validity/uncertainty/clock; do not jointly re-estimate joint type or overwrite source q. For original RGB-D at ~25 Hz vs force at ~50 Hz, respect the actual timestamps. `q_obs` is a measurement with covariance; it is not perfectly smooth/continuous or necessarily unbiased. Report observed angle/displacement, inferred velocity confidence and finite-difference diagnostics in **physical units**, not only in `sigma`.

Keep a consistent sign convention for `q, a, Q`; under a coordinate reversal, transform `q, v, Q`, and signed configuration-load functions consistently. Test virtual work invariance `Q*qdot` and positive inertial parameter under this transformation. An inconsistent impulse/`Delta v` sign must not be hidden by flipping one quantity alone.

## 3. Normalized/free-motion model — generalized for R/P and gravity (P0)

### 3.1 Canonical normalized equation

For `M>0`, divide the moving equation by effective inertia/mass:

\[
\boxed{\ddot q+\lambda\dot q+\gamma\operatorname{sgn}(\dot q)
  +a_g(q)+a_{\mathrm{mech}}(q,\dot q)=\beta Q(t),}
\]

where

\[
\lambda=c/M\quad [s^{-1}],\qquad
\gamma=F_c/M\quad [q\,s^{-2}],\qquad
\beta=1/M>0,
\]

`a_g(q)=G(q)/M`, `a_mech=tau_mech/M`. The units of `gamma` are `rad/s^2` for revolute or `m/s^2` for prismatic; `beta` has matching physical units and must be reported with them.

For `FREE_CONFIRMED`, the trusted model is

\[
\boxed{\ddot q+\lambda\dot q+\gamma\operatorname{sgn}(\dot q)+a_g(q)
+a_{\mathrm{mech}}(q,\dot q)\approx0.}
\]

Here **`beta`/`M` is absent from the free-only estimator** unless *additional absolute-scale evidence* makes it observable. We can estimate the normalized coefficients and simulate the free response, not an absolute inertial mass. If gravity or a mechanism is unknown, `a_g + a_mech` may itself be non-identifiable from velocity-dependent friction: explicitly state the ambiguity.

### 3.2 Gravity treatment must be correct for any axis

Dataset `base_link` z is **operator-declared true vertical**: `g_B=(0,0,-9.81) m/s^2`. Gravity is **not** a fitted arbitrary load if we have credible mass geometry. For a rigid body with mass `m`, COM `c(q)`, joint axis `a`, hinge origin `o`:

\[
G_P=-m g^Ta,\qquad
G_R(q)=-m g^T[a\times(c(q)-o)],\quad
c(q)=o+R_a(q-q_{ref})[c_{ref}-o].
\]

For an ideal **prismatic single rigid mass** (`M=m` with no reflected or carried extra inertia),

\[
a_g=-g^T a,
\]

which can be known independently of the body's absolute mass. For a revolute uniform-density geometric proxy `m=rho*m1`, `I_h=rho*I1` (without reflected inertia),

\[
a_g(q)=\frac{G_1(q)}{I_1},
\]

and **density cancels**. Thus free-motion gravity may be modeled from shape/COM/hinge geometry without estimating absolute mass. With nonzero known reflected inertia `I_refl`, `a_g=rho G1/(rho I1+I_refl)` and density no longer cancels; handle that dependency in the common physical model, not with an inconsistent independent gravity-mass parameter.

If the mass geometry/COM is *tentative*, label it an assumption and **propagate axis, hinge, COM, reference-angle and mass-shape uncertainty**. Do not substitute the arithmetic mean of 3DGS Gaussian centers for a measured COM, and do not claim the convex hull is a physical mass distribution. Provide clearly separated alternative gravity hypotheses:

- `gravity_geometry_known_shape`: known unit-density COM and inertia ratio (conditional on geometry assumption).
- `gravity_negligible_test`: only when a quantitative torque/acceleration bound relative to measurement sensitivity supports omitting it; for a nearly vertical revolute axis, test vertical-axis hypothesis using original RGB-D evidence first, rather than taking `tilt<10deg` as proof.
- `gravity_effective_diagnostic`: fit a normalized configuration-dependent load if needed for model checking; label it **effective, not verified gravity**. This may confound friction and mechanism effects and must not become a second gravitational mass.
- `gravity_unknown`: do not pretend the observed free trajectory identifies separate physical friction and gravity.

**Never fit a separate inertial mass and gravitational mass in a physically coupled mode.** In a simulator, mass and full inertia tensor must use one consistent mass distribution, even if the scalar hinge inertia was first identified from joint-space behavior.

### 3.3 Free-motion identifiability tests (structural + practical)

For an approximately gravity-neutral monotonic segment,

\[
\ddot q=-\lambda\dot q-\gamma s,\qquad s=\operatorname{sgn}(\dot q).
\]

The data can separate `lambda` and `gamma` only with sufficient variation in `qdot`; for nearly constant `qdot`, their regressor columns are nearly dependent. If there is one motion direction and small velocity variation, only a combination may be determined; if motion is approximately constant velocity, upper bounds / combinations may be all the data support. If free-motion `q(t)` is approximately linear with small noise, **do not fabricate nonzero damping/friction**. More complex `a_g(q)` and residuals further reduce identifiability.

Compute SVD/rank and correlations on **noise-weighted, scale-aware weak-form sensitivities** (not merely the covariance of a failed PINN). Use profile-likelihood or bootstrap confidence ranges, include trajectory noise and geometry hypothesis changes, and test stable fits on withheld temporal subsegments. Distinguish:

- `FREE_RATIOS_IDENTIFIED`: specific normalized rates are supported by data and free replay;
- `FREE_COMBINATION_IDENTIFIED`: only a combination / bound can be supported;
- `FREE_UNINFORMATIVE`: insufficient curvature/velocity excitation or unknown forcing;
- `FREE_MODEL_INADEQUATE`: clear mismatch in free trajectory despite nonzero excitation;
- `ABSOLUTE_SCALE_NOT_OBSERVABLE`: structural consequence of no known external torque/mass prior, **not** failed normalized fit.

## 4. Integral / weak-form identification without noisy second derivatives (P0)

### 4.1 Free-motion weak form

On a smooth monotonic moving segment `[ta,tb]` with known sign `s`, integrate the normalized equation:

\[
\boxed{\Delta v+\lambda\Delta q+\gamma s\Delta t
 +\int_{ta}^{tb}a_g(q)dt+\int a_{mech}(q,v)dt\approx0.}
\]

When no mechanism head is accepted, omit that term. Build windows over (a) short sliding intervals and (b) prefixes from each continuous free segment start, with resampling/uncertainty at the boundaries. Use a robust joint-position spline or existing trajectory uncertainty to infer endpoint `v`, and report `v` sensitivity to smoothing bandwidth. **Do not differentiate twice as the sole source of acceleration.** The free problem is linear in `lambda,gamma` when `a_g` is fixed; estimate nonnegative rates by constrained weighted LS with rank assessment and proper uncertainty. Optionally jointly fit smooth q(t) and rates using integral residuals; do not use hidden zero initial velocity.

If `qdot` reverses sign or transitions into sticking, split intervals or integrate the sign over explicitly modeled subsegments. Do not integrate a single fixed `s` across a reversal.

### 4.2 Free-motion forward validation (mandatory)

Using only free-motion-fitted `lambda,gamma` and the declared gravity hypothesis, solve the **normalized hybrid ODE** from `(q0,v0)` estimated using the **initial prefix only**, with its uncertainty. Compare predicted q(t) and v(t) to held-out q observations; report **rad or meters**, relative range-normalized RMSE, sigma-normalized RMSE, drift, and stop time/error. Never select `(q0,v0)` on held-out future data; document calibration prefix and test interval. A low generalized-force residual with `M,c,Fc->0` does **not** count as free-motion validation.

If normalized free replay fails at physically meaningful scale, do not proceed to report physical scale estimation as successful. Examine visual q errors, contact-state classification, unmodeled inputs, gravity, and mechanism effects first.

## 5. Absolute scale only from independent observable evidence (P0/P1)

### 5.1 Known input model

For trusted forced windows, write

\[
\boxed{\ddot q+\lambda\dot q+\gamma\operatorname{sgn}(\dot q)
+a_g(q)+a_{mech}(q,v)=\beta Q(t),\quad \beta=1/M>0.}
\]

**Estimate `beta` only after selecting a sufficient, physically coherent forced interval.** For an impulse window `[ta,tb]`, the nonparametric scale relation is

\[
D=\Delta v+\lambda\Delta q+\gamma\int_{ta}^{tb}\operatorname{sgn}(v)dt
+\int a_g(q)dt+\int a_{mech}(q,v)dt
=\beta J_Q,
\quad J_Q=\int_{ta}^{tb}Q(t)dt.
\]

If `D` and `J_Q` are observed accurately and consistently, `beta=D/J_Q`, hence `M=J_Q/D`. **Use joint inference of `beta`, `lambda` and `gamma` with noise/uncertainty propagation rather than a brittle ratio point estimate.** Multiple forced windows with distinct timing/acceleration are preferable. Discard or mark an inconsistent impulse as `SCALE_CONTRADICTED` rather than returning negative mass or adjusting sign to fit.

A calibrated known mass measurement or credible unit-density geometry **can also supply a scale**, but label its provenance `GEOMETRY_PRIOR_SCALED` or `MEASURED_MASS_SCALED`, never `FORCE_IDENTIFIED`. For a revolute panel, a known width-based physical mass template can map `I_h` to `m`, but not identify both from one joint axis without geometry assumptions. For prismatic, a direct single rigid translating body gives `m=M` only if unmodeled transmission/reflected mass is absent.

### 5.2 What is and is not identified

- Free q(t) can resolve `c/M`, `Fc/M`, some normalized gravity and normalized mechanism combinations; it **cannot determine `M`** without an absolute scale anchor.
- A noise-floor contact record may be sufficient to establish the absence of driving effort **but not the missing scale**.
- A short force impulse whose sign conflicts with `Delta v` cannot be used to recover `M`; continue using it for a separate measurement-consistency audit.
- When `M` is only prior-supplied, report `c = lambda*M` and `Fc = gamma*M` as **prior-scaled estimates** with intervals propagated from both normalized-rate uncertainty and the mass prior. They are **not** measured absolute friction/damping.
- Static breakaway magnitude `Fs` needs threshold-crossing data or other independent evidence. Free coasting alone cannot identify its absolute value.
- An arbitrary fitted `g(q)` or learned neural residual cannot serve as an absolute scale anchor.

### 5.3 Fridge impulse: mandatory unchanged diagnostic

Revisit source rows `229–232` and neighboring intervals with per-frame native generalized effort, contact-point lever, transform, signed `q`, optical q-confidence, `v` estimates, initial q/v provenance, tool acceleration compensation bounds and timestamp hypotheses. Plot `Q(t), J_Q(t), q(t), v(t)` with uncertainty. Check invariance to a **consistent** joint-coordinate reversal. Report whether corrected physical data support `D/J_Q>0`. If not, **do not generate a friction/mass estimate based on this pulse**. Also show whether the free-motion trajectory alone produces useful normalized rates and an acceptable free replay (a distinct result).

## 6. Mandatory shared estimator architecture (P1)

All estimators use identical `q_obs/q_sigma`, force record, geometry hypotheses, interval labels, normalized equation, and physical parameter naming. Use a single `NormalizedDynamicsSpec` or equivalent adapter **derived from the existing `ModelSpec`**. Do not duplicate differential equations, sign policies, or gravity functions. Preserve the existing full-model estimator as an explicit comparative legacy mode, not the default identification target for unforced real recordings.

### 6.1 New constrained weak-form estimator / initializer

Suggested module `forceartgs_g9/estimators/free_weak_form.py` (reuse equivalent existing code if present). Fit `lambda >= 0`, `gamma >= 0`, and only justified normalized mechanism/load coefficients using weighted integral equations. Output parameter combinations, rank, confidence intervals and per-window residuals. DO NOT output fake absolute `I,c,Fc`. Name any regularized/weakly supported rates `PRIOR_CONDITIONAL`, not identified. Independently fit forced `beta>0` when a trusted input supports it; output `M=1/beta` plus error propagation and correlation with rates.

### 6.2 Full-physics normalized GeneralPINN (MANDATORY, not a baseline)

Keep the existing one-per-episode continuous trajectory MLP `q_hat_phi,k(t)` with **shared** physical parameters across episodes. Time differentiation uses `torch.autograd.grad(..., create_graph=True)` in physical units; use measured `q(t)` with covariance. Do NOT let network q(t) override or silently redefine the underlying joint authority.

**Phase A: train normalized physics on free intervals, WITHOUT trainable M/rho/beta:**

\[
r_{free}=\ddot{\hat q}+\lambda\dot{\hat q}
 +\gamma\operatorname{sgn}(\dot{\hat q})+a_g(\hat q)+a_{mech}(\hat q,\dot{\hat q}).
\]

Use data-consistency loss + normalized strong-form loss on smooth moving collocation + **integrated weak-form loss** + initial conditions + separately implemented stick/breakaway regime losses (where supported). Measured q loss must prevent a degenerate flat trajectory. Do not misuse `torch.sign()` at zero speed; partition regimes. Include force-free classification uncertainty and avoid collocation directly at discontinuities. Use deterministic seeds and record full optimizer settings. **Parameter recovery is not assessed from seed spread alone**.

**Phase B: forced scale training, only with trusted forced inputs:** keep the learned normalized rates and allow controlled refinement, add strictly positive `beta=1/M` to the residual:

\[
r_{forced}=r_{free}-\beta Q(t).
\]

Add exact ZOH impulse integrals, direct forward-rollout fit, and an independent diagnostic for the signed impulse relation. Avoid optimizing `beta` against an untrusted force window or allowing a mechanism network to absorb impulse mismatch. The correct free-only fit must work **even if every forced sample is removed**. If no scale anchor exists, do not instantiate beta as a free parameter in the free stage.

**P1: training and overfitting fixes:** use an independent free-motion replay as an acceptance gate; compare prior and final parameter objective; consider retaining larger frozen PINN-V2 trajectory architecture as an **ablation** without modifying its package. Zero-initialize optional mechanism residual so it starts at physical baseline. Report normalized residual RMSE and open-loop q predictions separately. PINN uncertainty = bootstrap/profile/heldout sensitivity plus optional seed-ensemble spread; seed spread is *not a posterior*.

**Crucial:** the new PINN must still be a **PINN**: the normalized joint dynamics equation is the network's direct differentiable physics residual, not a label built from another estimator's parameters. Its forced phase contains the measured effort as RHS and its free phase explicitly contains zero RHS only for `FREE_CONFIRMED`.

### 6.3 Normalized augmented EKF / smoother (MANDATORY, not a baseline)

**Free stage:** augment the state with normalized parameters, e.g.

\[
x=[q,v,\eta_\lambda,\eta_\gamma,\ldots]^T,
\quad \lambda=\mathrm{softplus}(\eta_\lambda),\ \gamma=\mathrm{softplus}(\eta_\gamma).
\]

Transition in the moving regime:

\[
\dot q=v,\qquad
\dot v=-\lambda v-\gamma\operatorname{sgn}(v)-a_g(q)-a_{mech}(q,v),\qquad
\dot \eta=0
\]

for `FREE_CONFIRMED`. `z_k=q_obs(t_k)+noise`, update with measured `q_sigma`/R, **Joseph covariance form** and calibrated Q. If the data are batch/offline, add RTS smoothing (or explain why existing validated smoother suffices), without pretending the smoothed future-conditioned state is an independent held-out prediction. Model stick/slip events separately; no Jacobian of `sign(v)` at zero. Propagate state and parameter covariances with correct discrete Jacobians, stable numerical method and process-noise/provenance. Report NIS and parameter rank; **bounded/degenerate parameter covariances must not imply identified parameters**.

**Forced stage:** only when forced data pass the audit, introduce `eta_beta` for `beta>0` with `dot v += beta*Q(t)` and use the same data/clock as PINN. Do not include beta as an unobservable free-only state. Cross-check finite-difference vs analytic/autograd Jacobians (excluding discontinuities); propagate forcing noise as input uncertainty and q measurement noise separately. Compare free-only, forced, and smoothed results to common open-loop simulations.

### 6.4 Rollout optimizer (MANDATORY comparative method)

Use the same normalized model and hybrid simulator: first fit free trajectory using only `lambda,gamma,a_g`, then fit `beta` on trusted forced interactions, then optionally jointly refine on training episodes. No optimizing `M` in a homogeneous free residual, no `M=0` boundary interpreted as a physical result. Reconstruct the physical terms from a fitted `M` only after successful scale anchoring. Reuse the validated full-model solver with consistent parameter conversion rather than implementing another incompatible integrator.

### 6.5 Mechanism residual: *late and optional*

Use a ForceTwin-inspired residual only after free rates and baseline replay have been evaluated:

\[
\frac{\tau_{mech}}{M}=h_\theta(q)+d_\phi(q,|v|)v,\qquad d_\phi\ge0.
\]

`h_theta(q)` represents *effective* configuration load (could include model mismatch or unknown gravity), not an automatically verified spring. `d_phi` is nonnegative for dissipativity; use separate directional heads only if data support them. In a free-only fit, this residual is **normalized**, so it cannot supply the missing absolute mass scale. When a force-derived scale is later identified, dimensional mechanism torque can be reconstructed conditionally.

Fix `forcetwin_residual.py` if needed: **zero output-layer initialization**, damping near-zero initialization, **no-residual baseline as a candidate** during model selection, and acceptance only if withheld trajectory/effort prediction improves. Do not introduce both a constant `d_phi` and a free `lambda` without a zero-mean/orthogonality convention: those are non-identifiable. Constrain mechanism load not to soak up acceleration (`qdd`) or sign/timing inconsistency. Avoid full neural residual for a single short interaction with no independent test interval. If optional residual is unsupported, leave it disabled rather than making the primary model fail.

## 7. Validation and estimation status must be factorized (P0/P1)

One categorical `NOT_IDENTIFIED` for every property hides useful free-motion information. Define **separate axes**:

- `kinematic_authority`: accepted / operator-declared / working-only / rejected.
- `input_evidence`: free-confirmed / force-calibrated / exploratory declared / unknown / contradicted.
- `normalized_dynamics`: ratio identified / combination identified / uninformative / inadequate.
- `absolute_scale`: force-identified / measured-mass-derived / geometry-prior-derived / not-observable / contradicted.
- `simulator_readiness`: physical identified / prior-scaled preview / insufficient physical data / blocked.

For each parameter, emit `value` **only if it is approved for a declared consumer purpose**. To avoid unsafe data flow, prefer typed objects:

```json
{
  "normalized_parameters": {
    "lambda_c_over_M": {"estimate": null, "unit": "1/s", "status": "FREE_UNINFORMATIVE"},
    "gamma_Fc_over_M": {"estimate": null, "unit": "rad/s^2", "status": "FREE_UNINFORMATIVE"}
  },
  "absolute_scale": {
    "M": {"estimate_not_for_sim": null, "unit": "kg*m^2", "status": "ABSOLUTE_SCALE_NOT_OBSERVABLE"}
  },
  "simulator": {
    "readiness": "PRIOR_SCALED_PREVIEW_ONLY",
    "parameter_set_source": "EXPLICIT_PRIOR_NOT_MEASURED",
    "recommended_prior_range": null
  }
}
```

This is a schema *example*, **not a default value**. Populate estimates only from actual evidence, and use `null` for unsupported quantities. `M=0`, density=0 or negative inertia must never be accepted physical/production outputs. Never place an invalid point estimate in a generic `value` field used by simulators. A real, positive prior-scaled physics asset may be made **only as an explicitly labeled preview**, with uncertainty/ensemble scenarios and no claim of identification.

### 7.1 Adequacy scores and uncertainty

Report `q` replay in `rad/m`, sigma-normalized RMSE, range-normalized RMSE, test trajectory start state, heldout segment/time indices, as well as normalized-equation residual in physical acceleration units and effort residual where force exists. A near-zero effort RMS on `Q~0` is meaningless as a dynamics-quality metric. Guard against `sigma_q ~ 0` artificially producing enormous normalized errors; preserve the measured variance and supply transparent floors rather than altering the result until it passes.

Split data along **time/episodes**; avoid random per-sample splitting of autocorrelated adjacent samples as an independent test. A withheld suffix within the same demo is a useful *within-record predictive check*, **not** an independent new episode. Certified production still requires independent real force-conditioned validation where applicable.

## 8. Specific dataset experiments and run protocol (P0/P1)

Run lightweight synthetic tests before real runs. On the existing data, **do not require new force measurements before trying the free dynamics tests**.

| Scene | Required experiments | Interpretation restrictions |
|---|---|---|
| Fridge | `FREE_CONFIRMED` coast windows; normalized weak, PINN, EKF, rollout; q-v / v-a diagnostic; tilted vs **independently refitted** vertical-axis hypothesis; native pulse 229–232 sign audit; optional scale fit only if pulse is physically consistent | Can report normalized ratios if validated, **not absolute I** from the contradictory pulse |
| Suitcase | For each explicitly named R/P kinematic candidate, normalized free response, forced model if effort valid, parameter observability, axis/gravity uncertainty | Keep working joint and absolute physics **not certified** |
| Black box | Operator-declared R plus selected alternative, normalized model with geometry-relevant gravity hypotheses | Declaration is not independently accepted joint |
| Robot dog / white drawer | Input/kinematics readiness audit; run free dynamics only if q(t) and time intervals actually support it | Do not fabricate q, force, or a physical joint |

**Real run outputs** (append-only, e.g. `run/dynamics/g9_free_forced/vNNN/`):

```
resolved_config.yaml
source_hashes.json
segment_evidence.json
normalized_model_spec.json
free_weak_fit.json
free_pinn_fit.json
free_ekf_fit.json
free_rollout_fit.json
forced_scale_fit.json
identifiability_report.json
model_adequacy.json
geometry_gravity_hypotheses.json
simulator_handoff_preview.json
plots/
  q_observed_vs_free_replay.png
  qdot_and_decay_uncertainty.png
  phase_space_diagnostics.png
  force_and_generalized_impulse.png
  normalized_parameter_profiles.png
comparison_summary.md
```

Use read-only input paths and versioned output folders. Log tool versions, seed, run flags, input hashes, the actual joint-authority file, exact timestamps, every force and contact convention, and coordinate reference pose. Never claim real replay success based solely on PINN training loss.

## 9. Synthetic and unit tests that must fail before code and pass after (P0/P1)

Make deterministic tests in `tests/test_g9_free_*` and extend existing `test_g9_*`; test all required joint-type and physics branches. At minimum:

1. **Structural scale equivalence**: two R objects `(I,c,Fc)=(0.1,0.04,0.01)` and `(1.0,0.4,0.1)` with identical initial `(q,v)` and no external force/gravity have identical trajectories to numerical precision; all three estimators recognize the same `lambda=0.4 1/s`, `gamma=0.1 rad/s^2` and do **not** claim to resolve absolute `I`.
2. **No zero-inertia collapse**: free-only constant/near-zero `Q` cannot optimize physical `I` because `beta/I` is not a trainable free parameter. Any legacy `M,c,Fc->0` fit is flagged and never exported.
3. **Prismatic free damping**: simulated slider with known `lambda,gamma` (gravity-neutral direction) yields normalized parameters and free replay with correct units, without identifying `m`.
4. **Inclined prismatic gravity**: `g_B=(0,0,-9.81)`, arbitrary unit axis; test `a_g=-g.a` (ideal `M=m`), correct energy derivative, and nonzero acceleration under zero external force. Do not incorrectly assume `Q=0 -> qdd=0`.
5. **Revolute vertical + tilted axes**: exactly vertical -> gravity torque **identically zero** at all q for any COM; tilted -> correct `G(q)/I`, analytic potential-energy derivative and finite-difference agreement; test nonzero `q_ref` and axis/COM frame transform.
6. **Reflected inertia exception**: geometry-coupled `I=rho I1+Irefl` and `G=rho G1`; normalized gravity is NOT constant with respect to rho if `Irefl !=0`.
7. **Forced scale recoverability**: with distinct trusted impulses and same free ratios, estimate absolute `M` and hence `c,Fc` within a predeclared tolerance; test multiple episodes and noise propagation.
8. **Pulse sign contradiction**: a positive input impulse plus negative observed acceleration under insufficient countertorque must give `SCALE_CONTRADICTED` / reject `M`, not a negative or zero `M` or arbitrary sign fix.
9. **Unknown force is not zero force**: missing `Q`, unknown contact, slipping contact with unmeasured lever, or absent force annotations never silently become `FREE_CONFIRMED`.
10. **Insufficient velocity spread**: constant-velocity or 1-way nearly uniform segment reports normalized parameter correlation/rank failure, not spuriously tiny CIs.
11. **PINN differentiability**: `lambda,gamma` gradients pass from the normalized ODE residual to trained scalar parameters; q time derivatives have analytic check. In forced mode, beta gradient is nonzero under trusted force; in free mode beta parameter is absent.
12. **EKF stability**: Joseph covariance PSD, correct Jacobians, NIS, zero-crossing event handling, q-only observation, no spurious confident absolute M in free mode; state covariance not confused with parameter identifiability.
13. **Mechanism residual neutrality**: zero-initialized residual and no-residual candidate cannot yield worse selected validation than baseline; residual stays inactive without accepted baseline or withheld evidence. Test dissipativity `d_phi>=0` and orthogonality with constant viscous damping.
14. **Frame/sign invariance**: consistent q/axis/effort reversal gives equivalent inferred positive `M` and identical physical trajectory; object-side sign converted exactly once.
15. **Simulator handoff**: `ABSOLUTE_SCALE_NOT_OBSERVABLE` never exposes an unverified `value` to export; labeled prior-scaled preview is physically positive and follows identified free decay, but intentionally cannot claim calibrated forced response.
16. **Regression**: existing synthetic general dynamics / G9 / joint-discovery suites pass under their unchanged legacy configs; no 3DGS renders/training required for new unit tests.

### 9.1 Acceptance tolerances

Define and record physical-unit thresholds **before seeing result** in synthetic test fixtures; e.g. numerical equivalence `<1e-7` for equivalent noiseless integrated trajectories, and ratio recovery errors meaningful relative to synthetic noise and excitation (do not silently relax after failures). For real scenes, output *diagnostic evidence* without forcing an arbitrary 1-sigma pass threshold. Tests must include zero-force only, pure viscous, pure Coulomb, combined, stopping, reversal, nonzero gravity, weak excitation, and contradictory force. Never test only training loss/convergence.

## 10. Execution plan for Codex (deliver real code)

**Step 0 — Audit.** Inspect local tree, identify already existing classes and tests, baseline the failures; write file-level plan and preserve uncommitted changes.

**Step 1 — Input/segments.** Implement explicit `FREE_CONFIRMED`/`FORCED_MEASURED`/`UNKNOWN` classification with evidence; keep canonical robot record, no second tare or force conversion.

**Step 2 — Shared model.** Implement normalized R/P gravity-aware model as adapter over existing G9 `ModelSpec`; factor out one math implementation for weak, PINN, EKF, rollout.

**Step 3 — Weak estimator and diagnostic.** Fit normalized free ratios, observability, free forward replay; ensure no M parameter appears in force-free mode. Audit real fridge's coast and its pulse separately.

**Step 4 — Mandatory PINN.** Rewrite/generalize G9 PINN training to normalized free physics + forced-scale physics, preserving its existing network and optional strong/weak losses. Test against known synthetic data and diagnose gradient/derivative bias.

**Step 5 — Mandatory EKF/smoother.** Reparameterize filter as normalized free dynamics, optional beta on measured forced segments, correct event handling, covariance and identifiability.

**Step 6 — Rollout + optional residual.** Same normalized dynamics; ensure independent free replay. Gate optional ForceTwin mechanism and make baseline candidate mandatory.

**Step 7 — Physical/simulator handoff.** Distinguish measured absolute M, geometry/prior scale, normalized-only, and unsupported. Map to URDF/MJCF/USD/SAPIEN/ManiSkill using one consistent positive mass distribution and correct gravity; carry estimation provenance. Do not write physically bogus `value` fields.

**Step 8 — Tests and real comparisons.** All synthetic cases then real fridge, suitcase, black box; report ranks, confidence ranges, selected model, force contradictions, comparison across methods; keep data unchanged.

**Step 9 — Document and stop.** Write `docs/G9_FREE_FORCED_DYNAMICS_IMPLEMENTATION_REPORT.md`, `docs/G9_FREE_MOTION_REPO_AUDIT.md`, and updated method-comparison summaries, including exact commands, run versions, tests, hardware/memory, code changes, failures, and what cannot be identified. **Do not claim completion without passing tests and actual numeric results.** Do not commit or push.

## 11. Final report mandatory structure

1. **Repository audit and actual code paths**; state of uncommitted work and old-vs-new dynamics.
2. **Scientific model**: normalized free equation and physically dimensional forced equation; explicit R/P/gravity branches.
3. **Segment evidence**: which exact frames are free, forced, unknown, and WHY.
4. **Estimator details**: same-model weak, PINN, EKF, rollout; training schedules, network architectures, parameters, state equations, covariance, numerical integration.
5. **Identifiability**: free ratios vs absolute scale vs combinations, profile likelihood/bootstraps, degeneracies.
6. **Per-scene q plots and open-loop replay**, with physical/sigma-normalized errors and honest window boundaries.
7. **Fridge pulse audit**: signed J, signed Delta v, uncertainty and sign/timing/contact hypotheses; explicit verdict.
8. **Simulator export**: only proven physics values or labeled priors; formats, behavior tests, and unsupported features.
9. **Regression tests**: counts, names, commands, failures, runtime and exact input versions.
10. **Limitations / next experiment**: what additional sensing/forced episodes would identify otherwise unobservable absolute quantities.

### Non-negotiable scientific acceptance criterion

> **A zero-force demonstration must be able to produce a valid, predictive *normalized* free-motion dynamics model when its visual q(t) is informative, without ever yielding a spurious zero inertia. The pipeline must explicitly state when and why that data cannot establish absolute inertia. A measured forced episode may then provide the missing scale, but only if its signed torque/impulse and observed motion are consistent.**

