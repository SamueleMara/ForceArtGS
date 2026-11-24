# ForceArtGS G9 — ForceTwin-Inspired, Identifiability-First Joint Dynamics Identification

**IMPLEMENTATION DIRECTIVE FOR CODEX — 2026-11-21**  
**Target working tree:** `~/ForceArtGS` (the active local branch, with existing uncommitted G9/PINN/EKF and diagnostic changes).  
**Deliverable:** working implementation, tests, real-data diagnostic reports, method comparison and simulation handoff. Do not stop after writing an architectural proposal.  
**Source:** Engelbracht et al., *ForceTwin: Physics-informed Digital Twins for Robotic Manipulation from Instrumented Human Interaction*, arXiv:2609.21751 (2026), especially §III-C, Eqs. (4)–(9), §IV-A/B and §V. **Adapt its physical decomposition**; do not claim that its network or optimizer is a PINN/EKF.  
**Important:** At the last public audit, ForceArtGS `main` was `136c462624aa3d4ea4f05666d6e8cc752dc860d9`. Much newer PINN, EKF, geometry, axis-uncertainty and G9 work described in Codex reports is **local/uncommitted and not visible on public main**. Inspect the actual local checkout before touching any file. Never overwrite those changes. Do not modify `~/ForceArtGS-Physics` (pinned external library), retrain 3DGS, overwrite prior results, or commit/push without explicit permission.

## 0. Explicit decision and scientific target

**Replace G9's density-first *identification target* with effective one-degree-of-freedom joint dynamics identification.** Preserve the existing geometric mass/inertia modules as optional **downstream representations, priors and physically coupled counterfactuals**, not the default real-data estimator.

For the nearly vertical revolute fridge, fitting a near-zero density to suppress spurious gravitational torque also zeroes its rotational inertia. This is not evidence of a massless door. Identify the **hinge inertia** `I_h [kg m^2]`, **viscous damping** `b [N m s/rad]`, and **Coulomb friction** `mu [N m]` directly, **only when the measured excitation actually supports them**. Use the same effective description for prismatic objects, where `I_eff [kg]` represents effective translating mass (not automatically a complete physical moving-link mass if unknown actuator/constraint effects exist).

The critical problem in the current fridge recording is a reported **signed generalized impulse / inferred velocity-change conflict** around source frames **229–232**. The previous report found approximately 9.75-sigma impulse, 2.4-sigma velocity change, opposing signs, no sign reversal for ±40 ms lag, and a nonphysical (~−24 kg) tool-inertia explanation. **Resolve or explicitly localize this inconsistency before accepting any absolute inertial estimate.** A PINN, EKF or optimizer must NOT turn this contradiction into a near-zero inertia, false gravity or learned mechanism torque.

**Minimal model (moving regime):**

\[
\boxed{\tau(t)=I_{\rm eff}\ddot q(t)+\mu\,\operatorname{sgn}(\dot q(t))+b\dot q(t)+\tau_{\rm known\,grav}(q)+r_{\rm mech}(q,\dot q)+\epsilon(t).}
\]

- First fit `I_eff >= 0`, `mu >= 0`, `b >= 0` for sliding segments. Require **strictly positive, identifiable** inertia before calling it a physical estimate. `I_eff=0` is an optimization boundary/failed identification, not a valid physical door.
- For a **known exactly vertical hinge**, ideal gravity torque about the hinge is zero, even though `I_h` is nonzero. For the observed fridge **6.03-degree estimated tilt**, gravity is *potentially* nonzero; use a separate **joint-geometry hypothesis test** rather than silently forcing the axis or denying the tilt. The user's `base_link` z is declared physical vertical, so `g_B=[0,0,-9.81] m/s^2` is available as a *declared* datum.
- For horizontal/oblique revolute joints and inclined sliders, include **physically computed gravity** only with a credible mass geometry/COM prior or an identifiable bounded model; do not independently learn an inertial mass and a gravitational mass. If physical gravity is unavailable, keep the unexplained `q`-dependent torque in a **non-physical residual**, not in a fake gravity estimate.
- Treat a learned residual as **optional, regularized, state-dependent mechanism force** and only introduce it **after** robust baseline parameter estimation and observability checks. Residuals must not explain away broken force sign/timing or unmeasured contact moments.
- Retain full-model GeneralPINN, GeneralAugmentedEKF and rollout as first-class methods operating on the **same minimal model and same data**. The ForceTwin-inspired constrained linear estimate is a new primary **interpretable baseline/initializer**, not a substitute for PINN or EKF.
- The objective is **a predictive simulator asset**, not an unverified density, mass or COM number. If absolute inertia cannot be identified, return an explicit unsupported estimate and a defensible prior distribution for simulation, not a point estimate presented as measured physics.

### 0.1 Non-goals

No new calibration from scratch; no change to the accepted joint authority without an explicit hypothesis/ablation run; no silent torque/axis/q sign changes; no arbitrary learned neural acceleration correction; no forced physical mass readout from a Gaussian convex hull; no hidden physical-term selection; no production label without independent validation; no reconstruction training; no direct edits of the pinned Physics library.

## 1. Inspect and preserve the current local implementation

Run and record, **before edits**, commands such as:

```bash
cd ~/ForceArtGS
git branch --show-current
git rev-parse HEAD
git status --short
find forceartgs_g9 estimotion_2part/joint_discovery pipelines tools tests docs -maxdepth 3 -type f \
  \( -iname '*g9*' -o -iname '*pinn*' -o -iname '*ekf*' -o -iname '*force*' -o -iname '*impulse*' \) | sort
```

Read the **actual present files**; the following are expected from previous reports, not guaranteed on public `main`:

| Component | Inspect / extend, never replace blindly |
|---|---|
| Dynamics contract | `forceartgs_g9/dynamics/general_1dof.py` |
| General model / parameter switches | `forceartgs_g9/estimators/model_spec.py`, `forceartgs_g9/config.py` |
| Integral estimator | `forceartgs_g9/estimators/general_integral_ls.py` |
| Existing rollout fit | `forceartgs_g9/estimators/general_constrained.py` |
| Differentiable shared physics | `forceartgs_g9/estimators/torch_model.py` (if present locally) |
| General PINN | `forceartgs_g9/estimators/general_pinn.py` (local report) |
| General augmented EKF | `forceartgs_g9/estimators/general_ekf.py` (local report) |
| Mass proxy / geometry | `forceartgs_g9/dynamics/geometry_proxy.py`, `geometry_inertia.py`, `mass_properties.py` |
| Canonical force/pose input | `estimotion_2part/joint_discovery/robot_record.py` |
| Generalized effort and window creation | `forceartgs_g9/adapters/forceartgs_g9.py` |
| Force-sign contract | `estimotion_2part/joint_discovery/wrench_physics.py`, `force_convention.py` if present |
| Impulse and model audit | `forceartgs_g9/validation/impulse_coast.py`, `load_models.py` (local report) |
| G9 stages and exports | `forceartgs_g9/core.py`, `stage.py`, `cli.py`, `export/dynamics_v2.py`, `export/final_build.py` |
| Existing reports | `docs/G9_GENERAL_PINN_AKF_IMPLEMENTATION_REPORT.md`, `docs/G9_CORRECTIONS_AUDIT.md` |
| Existing runs | `results_19_11_26/forceartgs_joint_aware/` on the external SSD (do not rewrite old versions) |

Record the actual interface signatures, schemas, file hashes and current tests in `docs/G9_FORCETWIN_ADAPTATION_AUDIT.md`. **Extend the existing code** and preserve local staged/unstaged edits. Run read-only audit first and reuse native data and model classes. New proposed module names below are names, not instructions to duplicate an existing equivalent.

## 2. Shared coordinate, measurement and force contract (P0)

### 2.1 Canonical signed object-side wrench

The recorded wrench represents **object-on-robot sensor reaction**, and our physical input is **robot-on-object**:

\[
\mathbf F^B_{\rm obj}=-R_{B,EE}R_{EE,S}\,\mathbf F^S_{\rm tare\ corrected}.
\]

The `−1` must come **exactly once** from an explicit `metadata/contact_force_convention.json` declaration (`object_side_force_multiplier=-1`). If absent/invalid, raise a typed data error; **no silent default −1**. Do not re-tare or independently resynchronize the existing `contact_annotations.csv` / JSON. Preserve `source_frame_index`, timestamps, native 50 Hz force rows, and existing 25 Hz RGB-D trajectory timing. Do not set missing/unknown contact force to zero; only noise-floor-verified absent samples may be treated as approximately zero, with uncertainty.

If sensor **intrinsic moments** are used, the wrench reference point and full spatial transform must be known; otherwise use the point-force approximation and label it. Do not add a moment and `r×F` twice. Tool gravity changes and **dynamic tool acceleration** may bias the short pulse; inspect existing tool mass/COM/inertia/EE acceleration and compute a bound. Do not invent them or claim they have already been dynamically compensated merely because static tare exists.

### 2.2 Generalized effort via virtual work

For joint axis `a_B`, hinge point `o_B`, contact point `p_contact,B`:

\[
\begin{aligned}
\tau_R(t)&=\mathbf a_B^T[(\mathbf p_{\rm contact,B}-\mathbf o_B)\times\mathbf F_{\rm obj,B}+\mathbf M_{\rm obj,contact,B}],\\
\tau_P(t)&=\mathbf a_B^T\mathbf F_{\rm obj,B}.
\end{aligned}
\]

Equivalent to `J_obj(q)^T W_obj`. Compare the current assumption `p_contact = p_TCP` with available pointed-tip calibration, contact normal and geometric surface evidence; sliding contact can shift `p_contact`. Do not force the contact point to a material point. Unknown contact moments must be reported as uncertainty, not modeled as known zero without qualification.

Compute the actual **signed joint torque**, not only Euclidean force magnitude. Preserve a consistent `q`, `a`, `tau`, `qdot` convention; test invariance under joint-coordinate reversal (`q'=-q`, `a'=-a`, `tau'=-tau`) and under hinge-origin shifts along `a`. The observed `tau * qdot` work must not change under an equivalent coordinate convention.

### 2.3 Authoritative timelines

Build a single `JointInteractionRecord` from existing robot record + accepted/hypothesis joint authority, containing at least:

```python
{
  'scene': ..., 'joint_hypothesis_id': ..., 'joint_type': 'revolute|prismatic',
  'axis_B': [3], 'origin_B': [3] or None, 'q_reference': {...},
  'force_t_s': [Nf], 'source_force_row': [Nf], 'Q_object': [Nf],
  'Q_sigma': [Nf] or None, 'force_object_B': [Nf,3],
  'contact_point_B': [Nf,3], 'contact_mode': [Nf],
  'q_t_s': [Nq], 'source_q_frame': [Nq], 'q': [Nq], 'q_sigma': [Nq],
  'contact_intervals': ..., 'force_calibration_assumptions': ...,
  'geometry_hypothesis': ..., 'provenance_sha256': ...
}
```

Use existing data schemas where possible; do not introduce needless copies. Where force is evaluated on a different sampling grid, use **exact native-sample zero-order-hold integration**, not interpolating the impulse across a gap. Record unknown hardware clock offset as a separate scenario variable; do not call exact source-frame joining independent hardware-sync verification. Assertions: SI units, finite tensors, sorted unique timestamps, explicit coordinate frames, signed force provenance.

## 3. Fridge diagnostic first: locate the impulse–motion inconsistency (P0, blocking)

**Do this before fitting or retraining any network.** Use existing red-fridge `final_red_fridge_episode_000000` frames with force pulse in **source rows 229–232**, and a context window e.g. 210–270 as available. Do not infer that those indices are RGB-D frame numbers; use `source_frame_index` mapping.

### 3.1 Required side-by-side plots and machine-readable data

Create `tools/g9_forcetwin_fridge_audit.py` (or equivalent) that writes under a **new versioned** output folder:

1. Corrected 3D force `F_obj,B(t)`, original sensor reaction, tare baseline and sensor noise, plotted in their respective frames with labels.
2. TCP pose, **candidate actual contact point** and lever arm in base frame, joint axis/origin, signed tangential generalized effort `tau(t)`, `||F||`, projected tangential-force component and joint work `tau qdot`.
3. Native 50 Hz effort samples and **cumulative impulse** `J_tau(t)=∫tau dt` with uncertainty, pulse boundary definitions and exact source rows.
4. Measured q(t) at native RGB-D times; at least two explicitly named **independent** velocity reconstructions (regularized spline / local polynomial / state-space smoother) with propagated uncertainty, plus their angular velocity signs before and after pulse.
5. A sensitivity table for: explicit force sign (diagnostic counterfactual only; do not modify metadata), axis sign plus q convention together, lever/contact point uncertainty, hinge axis/origin, frame transforms, tool dynamic wrench, frame-clock offset including existing ±40 ms study, and choice of pre/post velocity windows.
6. Signed impulse–momentum check `J_tau ≈ I_eff Δqdot + ∫(mu sgn(qdot)+b qdot+tau_other)dt`; show the sign of **each** term, plausible bounds, and the residual for candidate positive inertias. Distinguish absolute nonidentifiability from contradictions that **no** positive inertia can satisfy within modeled uncertainties.
7. Counterfactual axes: **accepted estimated tilted hinge** vs **vertical-hinge hypothesis re-estimated using the original RGB-D tracks/TCP**. Reconstruct `q(t)` AND projected effort consistently for each candidate; compare held-out geometric residuals. Never just set the gravity term to zero while keeping the same arbitrary `q(t)` and `tau(t)`.

### 3.2 Decision states

Return at least `IMPULSE_CONSISTENT`, `IMPULSE_SIGN_OR_TIMING_INCONCLUSIVE`, `CONTACT_EFFORT_UNCERTAIN`, `MOTION_DERIVATIVE_UNCERTAIN`, `UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE`, `NO_POSITIVE_INERTIA_EXPLANATION`. Attach *evidence and uncertainty*, not a binary PASS based on curve appearance. **If no consistent explanation emerges, do not permit a fitted absolute inertia to be labeled identified.** Still allow ratio estimation from coasting and a conditional simulator prior.

### 3.3 Data are not magically sufficient

Coasting identifies ratios, e.g. when no external torque or mechanism:

\[
\ddot q+(b/I_{\rm eff})\dot q+(\mu/I_{\rm eff})\operatorname{sgn}(\dot q)=0.
\]

Scaling `(I,b,mu)` by a common factor produces identical unforced trajectories. Only valid measured effort and enough excitation can resolve absolute scale. One pulse with ambiguous velocity change cannot support 3 independent estimates merely because an optimizer returns 3 floats. Never mix `M` = hinge inertia with physical link mass.

## 4. ForceTwin-inspired minimal physical estimator (P0)

### 4.1 Directly estimate simulator-relevant coefficients

Follow ForceTwin §III-C, Eq. (5) and (7)–(9): use generalized joint effort and retain an explicit physical linear component:

\[
\tau_i=\underbrace{I_{\rm eff}\ddot q_i+\mu\operatorname{sgn}(\dot q_i)+b\dot q_i}_{\mathbf z_i^T\boldsymbol\beta}+r_i,
\qquad
\mathbf z_i=[\ddot q_i,\operatorname{sgn}(\dot q_i),\dot q_i],\quad
\boldsymbol\beta=[I_{\rm eff},\mu,b]^T.
\]

Paper order `[sgn(qdot),qdot,qdd]` and `[mu,b,I]` is equivalent; record the chosen order explicitly. For known physical gravity, subtract `tau_grav_known` from the measured generalized effort before regression **using the same assumed mass distribution as inertia**. For fridge primary vertical-hinge *hypothesis*, set `tau_grav=0` as a declared **hypothesis**, not a verified calibration; compare with the observed-axis hypothesis. Do not fit a freely chosen gravitational mass for the fridge.

**NEW module** (unless equivalent exists): `forceartgs_g9/estimators/forcetwin_linear.py`:

- Construct candidate sliding-only intervals with recorded excitation and `q_std`/force uncertainty; exclude uncertain contact, intermittent unknown force, stick/slip switches, hard joint stops and invalid differentiated frames **without using an arbitrary force gate to hide systematic failures**.
- Derive `qdot`, `qdd` with at least two robust smoothers; store bandwidth/regularization and a covariance/noise estimate. The direct regression must **not** rely on one untested numerical gradient.
- Solve **weighted, physically constrained nonnegative LS** (NNLS, e.g. `scipy.optimize.lsq_linear`, bounded `I>=0,mu>=0,b>=0`) for the enabled terms and compute residuals on the **original signed torque**. Weight for heteroscedastic torque/derivative uncertainties, and state the errors-in-variables limitation. Avoid treating strongly correlated consecutive rows as independent.
- Implement a numerically stable **integral / weak-form** counterpart on aligned effort samples, e.g. for window `[a,b]`:

\[
\int_a^b\tau dt=I\,[\dot q(b)-\dot q(a)]+\mu\int_a^b\operatorname{sgn}(\dot q)dt+b\,[q(b)-q(a)]+\int_a^b\tau_{\rm known\,grav}dt.
\]

Use nonoverlapping/blocked windows for conservative uncertainty (overlapping prefix windows are useful as estimation rows but not independent observations). Sweep window length, smoothing method and pulse boundaries. If the full rank is absent, report identifiable linear combinations, e.g. `b/I`, `mu/I`; do **not** invert a rank-deficient matrix and present pseudo-inverse coefficients as measurements.
- Evaluate nested model classes explicitly: `inertia_only`, `inertia_viscous`, `inertia_coulomb`, `inertia_viscous_coulomb`, and `inertia_viscous_coulomb_plus_known_gravity` as supported. Label selection as **diagnostic if no independent episode**; never auto-disable a requested term or declare a physical absence from a single run.
- ForceTwin's paper reports NNLS followed by neural residual training; we are **adapting** that and additionally comparing general PINN, EKF and direct rollout, not pretending the paper used them.
- Derive and report condition number/singular values of whitened `Z`, profile likelihood or bootstrap intervals, confidence in *positive* inertia, parameter-bound occupancy, correlations/confounds, and physically meaningful units.
- The 50 Hz force pulse and 25 Hz q samples must be processed on their native grids. Avoid naïve 2-point finite-difference acceleration at the impulse.

### 4.2 Identifiability, not convergence, is the gate

**Required per parameter:** `estimate`, `unit`, `method`, `model_variant`, `data_window`, `point_or_ratio`, `at_physical_bound`, `interval95_statistical`, `uncertainty_sensitivity`, `identifiability_status`, `model_adequacy`, `open_loop_replay`, `provenance`.

Accept absolute `I_eff` only if (i) independent signed effort and Δvelocity/acceleration are consistent with positive inertia under admissible uncertainty, (ii) effective regressor information/rank supports it and its CI is not dominated by the lower bound, (iii) residual adequacy is reasonable, (iv) robustness to smoothing/lag/contact perturbations, and (v) independent episode validation for production. Otherwise output `I_eff: NOT_IDENTIFIED`, even if a numerical value was fitted; preserve diagnostic numeric output separately.

Distinguish `MEASURED`, `ESTIMATED_CONDITIONAL`, `RATIO_ONLY`, `PRIOR_FOR_SIM`, `MODEL_INADEQUATE`, `DATA_INCONSISTENT`, `UNOBSERVABLE`. Avoid a single global BLOCKED status that loses partial information.

## 5. Structured mechanism residual — only if evidence supports it (P1)

### 5.1 ForceTwin-inspired functional form

ForceTwin §III-C defines a partially linear model with:

\[
\boxed{r_{\rm mech}(q,\dot q)=g_\theta(q)+c_\phi(q,|\dot q|,\operatorname{sgn}\dot q)\,\dot q,\qquad c_\phi\ge0.}
\]

- `g_theta(q)` is a **sign-unconstrained configuration-dependent generalized load**, possibly including spring, closer, latch or unmodeled gravity. **It is not identified as gravity or a specific mechanism without independent evidence.**
- `c_phi` is a nonnegative dissipative coefficient. Use `softplus` output and optionally separate positive-velocity/negative-velocity heads (as ForceTwin does), allowing direction-dependent dissipative behavior while ensuring `(-c_phi*qdot)*qdot <= 0`.
- **Do not include unrestricted acceleration or force-input dependence** in the residual: these could absorb the physical inertia or fix a broken force sign instead of learning a mechanism. `g` depends on `q` only; damping head on `q,abs(v),sign(v)` with sign handled as a discrete branch, not an unrestricted input that learns energy injection.
- Only train residual on valid sliding segments after physically constrained baseline parameters have been fitted and **frozen**. Train a small network; log architecture, inputs, normalization, smoothness, regularization and supported `q`/velocity domain. Use **constant/smoothness/complexity penalties**; nonlinear residual must have an evidence-backed improvement against a no-residual model.
- **Confounding:** `c_phi` constant portion overlaps baseline `b`, and `g_theta` can overlap gravity/bias. Do not claim a unique decomposition. Keep baseline coefficients frozen; use a small-norm/sparse residual penalty and diagnose overlap with the physical regressor space. Do not impose a hard zero-mean constraint on a nonnegative `c_phi` (mathematically impossible except zero). Do not describe the learned load as spring or gravity without supporting tests.
- **No zero-force hiding:** residual fitting must not turn `IMPULSE_SIGN_OR_TIMING_INCONCLUSIVE` into a physical PASS. Maintain a baseline-vs-residual error breakdown on both train and independent validation. Report residual energy/work per episode, and if `g_theta` should represent a conservative potential, optionally check `g_theta = dV/dq` and bounded energy over the observed range; never pretend all config-dependent effort is conservative.

### 5.2 Proposed implementation

`forceartgs_g9/estimators/forcetwin_residual.py` and `forceartgs_g9/validation/residual_adequacy.py` (reuse any equivalent). Use a compact tanh/SiLU MLP for `g_theta(q)` and separate `softplus` heads for dissipative coefficient; choose size based on available training data, not a large default. Record `parameters_count`; do not train large nets on one interaction and then claim physical generalization. ForceTwin's reported residual training is **full-batch AdamW lr=1e-3, weight_decay=1e-3, 1200-epoch maximum, 120-epoch patience, 75/25 random split** (§III-C). Adopt a reproducible baseline while **replacing random splitting with blocked chronological / episode-level validation** for our correlated real data; explicitly report that as a deliberate protocol improvement, not a reproduction of the original paper.

Minimum mandatory ablations: no residual; `g_theta(q)` only; dissipative `c_phi v` only; both; and (when geometry supports it) known projected gravity plus residual. Compare training, blocked validation, and independent episodes. If no held-out episodes, label `EXPLORATORY_TRAIN_ONLY` and do not activate neural-residual production export.

### 5.3 Exportability contract

For simulators, `I_eff`, `mu`, `b` are the **priority**. The residual is exported only as a versioned callable runtime evaluator (with deterministic learned weights and normalization), or a tested approximate backend-supported joint spring/damper profile. For URDF-only, learned nonlinear residual is not natively representable; report `RUNTIME_PLUGIN_REQUIRED` or `APPROXIMATED`, never say a bare URDF contains a neural model. Preserve model domain and safe extrapolation policy (clamp/abstain outside observed q,velocity). ForceTwin explicitly notes that simplified backend approximations can change behavior (§III-D).

## 6. Retain and correct all three full-model estimators (P1)

**All estimators use the same `JointInteractionRecord`, minimal term switches, force projection, gravity hypothesis, units, model-adequacy scores, and evaluation windows.** Do not create per-estimator preprocessing or independent optimized gravity sign. Frozen original ForceArtGS-Physics viscous methods remain frozen baselines, not full-model estimators.

### 6.1 Full-physics GeneralPINN

Keep `forceartgs_g9/estimators/general_pinn.py` and shared `torch_model.py`. Use a continuous trajectory network `q_hat_k(t)` per episode, shared global physical parameters `I_eff, mu, b` (plus explicitly enabled terms), with autodiff `qdot`, `qdd` in **physical seconds**. For the fridge minimal vertical-hinge hypothesis, the mandatory **network physics residual** is:

\[
\boxed{r_{\rm PINN}(t)=I_{\rm eff}\ddot{\hat q}_k(t)+\mu\operatorname{sgn}(\dot{\hat q}_k(t))+b\dot{\hat q}_k(t)-\tau(t).}
\]

For an independently justified known gravity model, **add** `tau_known_grav(q)` to the residual; do not separately fit gravity mass. Preserve data, initial condition, moving physics, weak/integral and impulse losses; include event-aware rest/breakaway only if specifically enabled and supported by observations. **Do not add the structured neural mechanism into the parameter-learning PINN from the beginning.** If residual is later fitted separately, freeze the physical coefficients and evaluate the combined model in replay.

PINN requirements:

1. Initialize physical parameters from robust NNLS / integral LS and preserve a baseline replay. Positive parameters constrained without preventing an exact zero for optional friction; require `I_eff>0` for valid physical dynamics.
2. Use native ZOH effort and oversample collocation near short pushes; avoid differentiating across jumps and incorrect stick/slip discontinuities. Strong-form residual *must* remain a direct ODE residual in `torch.autograd`, not replaced entirely by rollout loss. Use weak-form impulse conservation to carry the meaningful short-push signal without inventing more q samples.
3. Before trusting results, compare `r_PINN` and its gradient against the shared canonical model on synthetic trajectories, including zero/near-zero gravity and coordinate reversal. Test gradient wrt `I_eff`, `b`, `mu` independently.
4. Prevent the observed near-zero-`b` collapse with **predeclared**, train-only choices: data warmup, loss scaling from observation/force noise, physical residual weight scheduling, term-identifiability tests, optional alternate optimization with temporarily fixed physical coefficients, and checkpointing. Do not repeatedly tweak hyperparameters using a held-out scene as a development set. The original Physics PINN-V2's larger 4x64 network, longer schedule and optional LBFGS can be evaluated as a fair architecture ablation, but is not a mandatory clone.
5. Select best seed **by frozen-parameter open-loop replay on training/validation data** as a separate metric; do not report low neural training loss as real predictive accuracy. Production method selection must use independent held-out episodes; without them, no production claim.
6. If `I_eff` unidentifiable or pulse inconsistent, report `NOT_IDENTIFIED` / `MODEL_INCONSISTENT` and preserve the conditional fit in diagnostics. Do not lower `I_eff` to zero to compensate for missing gravity/contact evidence.

### 6.2 Augmented EKF

Use the existing GeneralAugmentedEKF rather than replacing it with a nonlinear optimizer. State for moving model, e.g.:

\[
\mathbf x=[q,v,\eta_I,\eta_\mu,\eta_b]^T,\qquad
\dot v={\tau(t)-\mu\operatorname{sgn}(v)-bv-\tau_{\rm known\,grav}(q)\over I_{\rm eff}},
\]

where transforms from latent `eta` to physical coefficients enforce `I>0`, `mu,b>=0` without falsely ruling out exact zero friction. Use robot effort as an input and vision-derived q with covariance as the observation. State transition integrates over native ZOH force subintervals; the measurement Jacobian, dynamic Jacobian and parameter sensitivities must be verified by finite differences / autodiff. Joseph-form covariance update; meaningful `P0`, `Q`, `R` and process/measurement covariance with evidence or explicit scenarios. Handle sticking/sliding events with explicit hybrid logic; do not apply smooth `sign(v)` Jacobians through an unresolved transition. Optional RTS smoothing may be added but must be separate and reproducible.

Report parameter covariance, normalized innovation squared (NIS) and whiteness, boundary behavior, observability and sensitivity to tool lag/force scale. Do not use an NIS value of ~8.5 on the suitcase to claim a validated model. Use unobserved parameters as `PRIOR_ONLY`, rather than keeping a confident EKF posterior when observation sensitivity is rank deficient.

### 6.3 Constrained rollout fit

Preserve `general_constrained.py` as an independent third estimator of the same **minimal** enabled model. It must support `joint_space_effective` `I,mu,b` without needing mesh/density or a free unmeasured gravity moment. Enforce physically positive inertia, joint-coordinate consistency and stable integrator. Compare with NNLS, PINN and EKF on the same exact force and q timeline; report optimizer boundary hits, NLL/replay quality and differences. Apply model adequacy/identifiability checks equally to all methods.

## 7. Relation to 3DGS geometry, mass and gravitational consistency (P1)

### 7.1 Decouple estimation from physical representation

**Geometry remains essential for rendering, collision and plausible inertials, but must not force the estimated dynamics scale to zero.** Keep `geometry_coupled` as a separate hypothesis/validation mode. The new primary real-data outputs are simulator-compatible **effective joint coefficients** (ForceTwin's choice) independent of exact link mass distribution.

A full URDF/USD/MJCF engine needs valid link mass `m>0`, COM `c` and positive-definite inertia `J_C`, whereas an effective revolute estimate is only `I_eff` about the hinge. Given an **explicitly documented** mass shape and unit-density geometry (`m1`, `J_C,1`, `I_h,1`) and known reflected inertia `I_ref`:

\[
I_{\rm eff}=\rho I_{h,1}+I_{\rm ref},\quad
\rho={I_{\rm eff}-I_{\rm ref}\over I_{h,1}},\quad
m=\rho m_1,\quad J_C=\rho J_{C,1}.
\]

This is a **conditional mass allocation** from an identified hinge inertia, NOT a separately measured mass. If an assumed geometry produces a 0.08 kg door or requires density ~0 to fit, don't call it a real mass. Explore bounded plausible panel templates, thicknesses and mass priors and show a distribution of possible `m,c,J_C`; do not invent a measured mass. If the effective inertia is not identifiable, do not invert this equation; use a labeled prior and propagate uncertainty to the simulator.

For **known exactly vertical hinge**, gravity generalized torque remains identically zero for any consistent mass geometry. For tilted/oblique joints, calculate `G(q)=dU/dq` analytically using the **same** mass and COM allocated to the link and the actually validated joint geometry; `m_gravitational = m_inertial`. No extra independently fitted gravity mass. Compute `r_perp = (I-aa^T)(c-o)` from the chosen mass geometry, with correct q reference; `I_h=a^T J_C a+m||r_perp||²`. Report when geometry/simulator gravity yields a torque incompatible with the observed joint-force model.

### 7.2 Simulator engine details

Do not claim a scalar `I_eff` directly fills a generic URDF `<joint>` inertia field: in URDF, body inertials live on **links**, and any reflected actuator inertia may require backend-specific treatment. Use the canonical PArtTwin1D asset builder; for ManiSkill/SAPIEN, MuJoCo/MJCF and Isaac/USD, inspect how hinge inertia, `mu`, `b` and static friction are represented. Verify actual runtime readback, not just authoring. SAPIEN's observed loader friction behavior may differ from URDF requested friction; test actual version and record it. Residual `g_theta/c_phi` needs a runtime generalized-force callback or a validated spring/damper approximation. Do not claim universal cross-engine parity without Isaac runtime tests.

### 7.3 Export readiness

- `KINEMATICS_ACCEPTED`, force calibration evidence, `JOINT_DYNAMICS_IDENTIFIED`, `MECHANISM_VALIDATED`, `MASS_ALLOCATION_ASSUMED`, `ASSET_COMPILES`, and `ENGINE_REPLAY_VALIDATED` are **separate** states.
- Simulation preview with inferred priors is allowed only with an explicit **assumption/uncertainty label**. A torque model validated on a real held-out push is stronger than a visually plausible density, but neither should be mislabeled.
- Preserve the 2 standard Graphdeco 3DGS PLY files and existing R/P URDF-like transforms; do not change or compress visual representation to solve physics.

## 8. Dataset experiments and evaluation protocol (P0/P1)

### 8.1 Existing real scenes: mandatory results

Use the actual accepted/hypothesis joint authorities and existing robot archives. Do **not** fabricate missing sensor data. Do not claim the suitcase prismatic model is accepted when its joint selection was `TYPE_UNOBSERVABLE`.

| Scene | Minimum evaluation | Reported previous issue |
|---|---|---|
| Red fridge | signed impulse audit; accepted-axis vs refit-vertical hypothesis; minimal `I,mu,b` fits; no-residual vs residual diagnostics; geometry allocation only after I supported | 229–232 impulse/Δv signs conflict; tilted-axis gravity confounds density; near-zero hull masses; geometry-coupled 25–29σ vs freer load ~9.9σ |
| Orange suitcase | both R/P candidates where available, but retain `TYPE_UNOBSERVABLE`; compare effort and q consistency, nested model fits and NIS | horizontal-slider working model with ~8 kg effective mass, damping boundary and load along slider; not physical |
| Black box | declared revolute hypothesis only as clearly operator-declared; full identifiability check | mostly one-direction/constant-speed; underexcited |
| Robot dog | retain no accepted joint status; diagnostics only | model inconsistent |
| White drawer | report no robot force archive; no force-based physical estimate | data unavailable |

**For each scene:** show raw and projected effort, q, qdot/qdd with uncertainty, identified time windows, rank and condition number, physical parameter CI and bound occupancy, `I/b/mu` ratios, signed impulse balance, residual diagnostics, in-sample fit, **frozen-parameter open-loop trajectory replay**, and if present independent withheld episode results. Report each estimator and model-variant separately, never silently choose the lowest training error and call it physics.

### 8.2 Quantitative comparison matrix

Create `tools/g9_forcetwin_comparison.py` producing CSV/JSON/MD and plots:

Rows = scene × axis hypothesis × model variant × estimator (`NNLS_direct`, `NNLS_weak`, `GeneralIntegralLS`, `GeneralPINN`, `GeneralAugmentedEKF`, `GeneralRolloutFit`) × residual option (`none`,`g`,`damping`,`both`).

Columns = `I_eff`, `mu`, `b`, units, confidence intervals, parameter rank, identifiable combinations, `condition_number`, physics adequacy, whitened effort RMS, open-loop q RMSE/σ, peak signed impulse mismatch, NIS (EKF), number of training & heldout episodes, geometry hypothesis, calibration provenance, model status, runtime and code revision.

**No imaginary heldout episode**: blocked temporal splits of one trajectory are diagnostics, not proof of unseen-interaction generalization. Train/validation partitions must not leak overlapping windows, smoothing results, initial states, or force-derived labels. For new episodes when eventually provided, ensure visual q estimation has not used held-out forces to adjust joint or state.

### 8.3 Necessary synthetic experiments

Build targeted synthetic cases rather than only random data:

1. **Vertical revolving panel:** realistic `I_h>0`, `G=0`, `mu>0`, `b>0`, short acceleration push + long coasting; recover `I_h` with rich force and report coasting ratio-only when push removed.
2. **Oblique/horizontal axis:** generate known COM, density, gravity, verify exact `G=dU/dq`, parameter recovery, and force-sign invariance.
3. **Prismatic horizontal/inclined:** force projection, mass term, slider gravity, with and without a known constant load.
4. **No excitation:** constant speed one direction; require rank/identifiability failure for separating inertia, viscous, Coulomb.
5. **Short 80 ms impulse:** native force 50 Hz vs q 25 Hz; recover inertia only where velocity jump is detectable; do not generate spurious acceleration or declare a 4-sample force history as a 4-sample RGB-D history.
6. **Incorrect wrench sign / contact point / lag:** must be caught by forward impulse physics, not repaired with negative inertia or neural residual.
7. **Residual mechanism:** a door closer (configuration torque) + direction-asymmetric dissipative term; test whether residual improves withheld prediction while keeping physical `I` within tolerance; ensure no improvement claims on mismatched input signs.
8. **Geometry conflict:** unrealistic convex hull cannot force a zero mass in primary effective-inertia mode; mass allocation flagged.
9. **EKF consistency:** Joseph covariance symmetry PSD, state parameter positivity, NIS calibration, no false posterior contraction in unobservable data.
10. **PINN gradients and mass coupling:** ODE residual autograd checks for `I,b,mu` and optional `rho`; friction event restrictions and open-loop results; no loss-only success criterion.

Each test should have deterministic seeds, declared noise, the exact simulator used to generate truth, a nontrivial withheld-force trajectory where available, and documented tolerances. Synthetic recovery from the same generator is a necessary software check, **not** independent real-world validity.

## 9. Implementation order, modules and interfaces

Proceed sequentially. Each phase must produce code + runnable tests + a brief result. Do not jump straight to a neural mechanism.

**P0-A — Local audit and shared records:** audit uncommitted changes and running jobs; preserve all existing code and result folders; add strict sign/units/frame validation and canonical `JointInteractionRecord` adapter. Test sign once, coordinate reversal, source-frame joining and invalid metadata.

**P0-B — Fridge impulse audit:** implement `tools/g9_forcetwin_fridge_audit.py` and `validation/impulse_balance.py` (or extend existing `impulse_coast.py`); report force and q data with sign and uncertainty. Refit vertical joint hypothesis with upstream kinematic evidence, not physics-only axis substitution. If inconsistency unresolved, explicitly mark inertia unsupported.

**P0-C — Minimal NNLS + weak estimator:** add `estimators/forcetwin_linear.py`, minimal effective model option in `model_spec.py`, compare direct and integrated formulations, produce nested-model adequacy and rank. Keep current general configuration and geometry-coupled model as options; do not delete or reinterpret stored run results.

**P1-A — PINN/EKF/rollout parity:** extend local GeneralPINN, GeneralAugmentedEKF, rollout to same minimal model and fixed training windows; add fair comparisons and tests. Preserve original pinned Physics baselines untouched.

**P1-B — Mechanism residual:** implement only after a defensible parametric baseline. Network has `g_theta(q)` and a strictly dissipative `c_phi(q,abs(v),direction)*v` head; train with frozen baseline, conservative regularization and blocked validation. Export only if supported and predictive.

**P1-C — Simulator handoff:** use `I_eff` to allocate a plausible mass/inertia *conditionally*, preserve full tensor/COM geometry consistency, map damping/Coulomb correctly per engine; add runtime feedback and heldout force replay. Keep versioned outputs.

**P2 — New interaction protocol:** only after diagnosing existing data, suggest capture requirements (not a prerequisite for P0): one fast acceleration/deceleration push and independently timed force at adequate bandwidth to identify inertia, multiple speed plateaus and reversals to disambiguate viscous/Coulomb, coast/release phases to identify ratios, varied configurations for spring/closer forces. A *slow sustained push alone* does not reliably identify inertia.

### 9.1 Configuration and CLI requirements

Add fields through the current config schema with strict validation (don't break existing YAML):

```yaml
# EXAMPLE ONLY: adapt to current schema rather than blindly replacing it
identification:
  parameterization: joint_space_effective  # alternative: geometry_coupled
  model: inertia_viscous_coulomb            # with explicit toggles
  estimators: [forcetwin_nnls, forcetwin_weak, general_pinn, general_ekf, general_rollout]
  gravity_hypothesis: null_or_validated_axis # per-run explicit hypothesis
  identify_only_supported_parameters: true
  residual:
    enabled: false                           # default OFF
    modes: [configuration, dissipative]
    fit_after_physics: true
    freeze_physical_parameters: true
    require_independent_validation_for_export: true
  validation:
    require_signed_impulse_consistency: true
    report_ratio_only_when_scale_unobservable: true
    allow_assumption_preview: true
```

Expose CLI operations like `fridge-audit`, `fit-minimal-dynamics`, `fit-mechanism-residual`, `compare-dynamics`, preferably as subcommands of the existing `forceartgs_g9.cli`. Provide actual runnable commands using the local run paths and existing env; don't invent flags until implemented and tested. Respect `force_policy: exploratory`; conditional estimates must never be silently promoted to certified.

### 9.2 Output layout (append-only)

```
<run>/dynamics/forcetwin_adaptation/vNNN/
    input_manifest.json                  # file hashes, frames, force, q, geometry hypotheses
    joint_interaction_record.{npz,json} # canonical shared observations and provenance
    impulse_balance.json                # signed impulse, Δv, all sensitivity cases
    impulse_balance.png
    kinematics_hypotheses.json          # RGB-D/TCP refit metrics, not a silent axis override
    identification/nnls_direct.json
    identification/nnls_weak.json
    identification/general_pinn.json
    identification/general_ekf.json
    identification/general_rollout.json
    identifiability.json
    model_adequacy.json
    mechanism_residual/                 # only when enabled; weights, config, domain
    simulation_handoff.json            # only validated terms or explicit prior preview
    comparison.csv
    comparison.md
    STATUS.json
```

Reuse existing G9 versioned artifact conventions if possible. Never modify old `dynamics/g9/vNNN` folders, train data, source archives or simulation PLY files. Report figure/CSV paths and summary in `docs/G9_FORCETWIN_ADAPTATION_IMPLEMENTATION_REPORT.md`.

## 10. Acceptance criteria: code alone is not completion

1. **Shared physics and sign:** Test that one explicit sign flip gives force-on-object; a correct joint coordinate reversal preserves work and parameter magnitudes. Native force pulse is preserved through ZOH effort integration. Wrong metadata is rejected.
2. **Fridge inconsistency localized:** The new report quantifies the 229–232 signed pulse, qdot step, confidence, contact point assumptions, tool dynamic forces, lag, tilt/vertical joint hypotheses. If unresolved, states `NOT_IDENTIFIED_ABSOLUTE_INERTIA` instead of inventing a mass. Must give a specific next experiment based on failure mode.
3. **Identifiability-first effective inertia:** NNLS/weak/PINN/EKF/rollout agree on compatible models *where synthetic data contain enough excitation*; model rank-deficient cases return explicitly unidentifiable quantities and supported ratios.
4. **No mass collapse used as proof:** Geometry-coupled optimizer hitting zero density never becomes evidence of a zero-mass door. Primary `I_eff` estimation is independent of hull density; geometry-to-link mass conversion follows only where physical support exists.
5. **Physics-informed PINN is real:** PINN's loss explicitly contains `I qdd + mu sign(qdot) + b qdot + G - tau`, differentiated in time with autograd. A net trained only to predict q without this residual is not acceptable; neither is substituting the rollout fit.
6. **Augmented EKF is real:** EKF state contains q/v and persistent physics parameters, uses signed Q input and measured q output, Joseph covariance updates and event-consistent dynamics; report innovation and posterior uncertainty.
7. **Residual cannot hide errors:** Neural mechanism is not trained before baseline/force checks, cannot depend on acceleration or raw measured torque, is dissipative where specified, and is only exported with independent predictive evidence (or is clearly an exploratory preview).
8. **Simulation truthful:** A URDF link mass/COM/full inertia are distinguishable from one fitted scalar hinge inertia. The same gravitational/inertial mass is used in link asset; no new independent fitted gravitational mass; per-engine joint friction and runtime residual constraints are verified and labeled.
9. **Real experiments runnable:** Scripts run on existing local fridge/suitcase/black-box data **without requiring new calibration or re-rendering/retraining 3DGS**, gracefully report unavailable robot/joint data, preserve all historical runs, and emit comparison tables/plots.
10. **Tests:** Run the entire relevant `test_g9_*.py` and `test_joint_*.py` suites and the newly introduced targeted tests **after all final edits**. Provide exact pass/fail/skip counts and tracebacks for failures; no stale test claim.
11. **Documentation:** Link comparative evidence from the paper's Eq. (5), (7)–(9) and physical caveats; distinguish paper-specific choices from our extensions (PINN, EKF, weak form, sliding contact, 3DGS export).
12. **Reproducibility:** Add a concise actual-command runbook and output schema; record git SHA and dirty state. **Do not commit or push** unless instructed.

## 11. Mandatory final Codex report

Include these sections, with no fabricated values:

1. Current local tree audit and preserved files/commits.
2. Exactly what changed and which modules were reused.
3. ForceTwin-to-G9 mathematical mapping (what copied in principle, what extended).
4. Fridge effort and impulse balance with signed force plots and competing joint hypotheses.
5. Per-scene parameter table with physical/conditional/prior-only labels and covariance/rank.
6. NNLS vs integral vs PINN vs EKF vs rollout comparison, using identical observations and frozen-parameter replay.
7. Residual network ablation and independent validation status (or explicit NOT_RUN).
8. Geometry and simulator asset mapping, including impossible mass-prior cases.
9. Synthetic tests plus all regression-suite outcomes, including final rerun.
10. Remaining measurement limitations and minimal next acquisition protocol.
11. Exact local commands, input hashes, output paths, current git status, and explicit statement that nothing was committed/pushed.

---

**Central implementation principle:** ForceTwin identifies **effective joint inertia, Coulomb friction and viscous damping directly from measured generalized effort**, then learns only the **repeatable residual mechanism**. In ForceArtGS, keep the PINN and augmented EKF, but make them estimate these **same identifiable joint quantities** instead of trying to backsolve all of mass, density, gravity, friction and hidden mechanisms from one weak push. Use 3DGS geometry **after identification** to construct a physically consistent, explicitly conditional simulation representation. Above all, diagnose the signed fridge impulse inconsistency instead of optimizing around it.
