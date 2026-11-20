# ForceArtGS G9 — Mandatory Full-Physics PINN and Augmented EKF Implementation

**Status:** Revised implementation directive for Codex, including the operator's mandatory unified inertial/gravitational mass and analytic perpendicular COM vector. This version supersedes the earlier PINN/EKF-only directive. Do not claim completion until both new estimators run on the same general rigid 1-DoF model as the current G9 rollout optimizer and pass the tests and acceptance gates below.

**Verified repository basis (2026-11-20):**
- `SamueleMara/ForceArtGS` public `main` commit `136c462624aa3d4ea4f05666d6e8cc752dc860d9` contains `forceartgs_g9/estimators/general_constrained.py`, `general_integral_ls.py`, `model_spec.py`, and `frozen_baselines.py` but no general PINN or general augmented EKF.
- `andrearichichi/ForceArtGS-Physics` pinned `f3824316272d85cb8d788166372c81ac3105ca7c` implements frozen **viscous-only** `PINN` and `AKF` (plus historical baselines); preserve them unchanged for benchmark reproducibility and licensing.
- User's later Codex transcript describes uncommitted experiments with `joint_space_gravity_coupled`, operator-declared vertical and optional declared joint type. **Audit the working tree before editing**; do not overwrite those changes or accept their physical claims without tests.

## 0. Non-negotiable task and architecture

Implement **two first-class general-dynamics estimators**:
1. `GeneralPINN`: a neural trajectory model trained on measured q(t) **AND on the residual of the actual dynamic equation itself**, differentiating the network output with respect to physical time using PyTorch autograd, with static friction treated as a separate hybrid constraint.
2. `GeneralAugmentedEKF`: an augmented-state nonlinear EKF with physical parameters as static latent states, measured signed generalized force as control input, RGB-D-derived joint position q as observation, proper covariance propagation, Joseph update, and an event-aware stick/slip model. Optionally add an offline RTS smoother / iterated EKF as a clearly named extension.

Do **not** use `general_rollout_fit` as a substitute for either. Keep `GeneralIntegralLS` as an initializer and `general_rollout_fit` as a third comparator. Do not rename the frozen ForceArtGS-Physics viscous PINN/AKF as general-model methods. Do not alter the validated Physics source or copy its vendored code.

Architecture stays:

```
RGB-D + TCP + contact/force -> PRE-3DGS joint/support discovery
  -> freeze accepted joint type/axis/origin and measured q(t) + uncertainty
  -> train separate static/moving 3DGS; FreeArtGS articulated refinement
  -> G9 dynamics: canonical robot record -> Q(t) -> [GeneralPINN, GeneralAugmentedEKF,
                                                   GeneralRolloutFit, IntegralLS baseline]
  -> identifiability, model adequacy, independent force-conditioned held-out predictions
  -> optional geometry-coupled mass/COM/full inertia -> canonical PArtTwin physics asset
  -> URDF / USD / MJCF
```

G9 does not reassign a joint type, hinge, or moving support. `--declared-joint-type` remains an explicit, uncertaified **kinematics hypothesis**; neither an EKF fit nor a PINN fit validates that joint geometrically. Preserve all upstream production/export gates.

## 1. Canonical equations, signs, term switches

Read all configuration from `forceartgs_g9/config.py`, model/units from `forceartgs_g9/estimators/model_spec.py` and `forceartgs_g9/dynamics/general_1dof.py`. Reuse these physical definitions, **do not duplicate incompatible gravity equations or change the effective inertia meaning**.

At moving samples, for either revolute (q in rad, Q in N m, M in kg m²) or prismatic (q in m, Q in N, M in kg):

\[
  r(t;\theta, q) = M(\theta)\,\ddot q(t)
                 + c(\theta)\,\dot q(t)
                 + F_c(\theta)\,\operatorname{sgn}(\dot q(t))
                 + G(q(t);\theta,\mathbf g)
                 + b(\theta)-Q_{\rm robot}(t).
\]

Set \(r=0\). All disabled terms are **exactly absent** from this equation and from the optimizable parameter vector. Inertia is normally always enabled; a quasistatic mode must be named separately. Static friction is **not** `+F_s sign(v)` in this moving equation. The stuck regime uses

\[
  v=0,\quad \ddot q=0,\quad |Q-G(q)-b|\le F_s
\]

with appropriate transition/breakaway constraints. Require \(M>0, c\ge0, F_c\ge0, F_s\ge F_c\), with explicit zero-capable parameterization or bounds. Force during contact: `F_object = sign_multiplier * R_base_ee @ R_ee_sensor @ F_sensor_tare_corrected`. **Recorded real reaction on the robot has multiplier −1**, exactly once; metadata must explicitly state the convention (no fallback default). Joint efforts:

\[
 Q_P=\mathbf a^T\mathbf F_{\rm object},\quad
 Q_R=\mathbf a^T[(\mathbf p_{\rm TCP}-\mathbf o)\times\mathbf F_{\rm object}].
\]

No raw sensor torque contribution unless the torque origin and transport are verified.

Gravity: user's declared `base_link` z is physical vertical: \(\mathbf g=(0,0,-9.81)\) m/s². Keep declaration provenance separate from measured calibration. Use \(G=\partial U/\partial q\), \(U=-m\mathbf g\cdot\mathbf c(q)\). For physically geometry-coupled mass shape, one density scale \(\rho\) determines both \(M=\rho M_1+M_{\rm reflected}\) and \(G(q)=\rho G_1(q)\). For prismatic one-rigid-link without reflected inertia \(G_P=-M\mathbf g\cdot\mathbf a\). For revolute `joint_space_gravity_coupled`, \(M\) and \(\mu=mr\) are **independent fitted quantities**, with COM direction assumed; label as conditional, not complete inertia reconstruction. Moving-support centroid is not a measured COM. Gravity/hinge-axis uncertainty must be reported, especially fridge ~6° measured tilt vs physically vertical alternative.


## 1A. Mandatory unified inertial/gravitational mass from moving-part geometry (operator correction)

**This section overrides any implementation that independently estimates hinge inertia `M` and gravitational first mass moment `mu` when the reconstructed moving-part mass geometry is available.** The physical inertial mass and gravitational mass are one and the same. The preferred **physically interpretable** G9 model is constant-density (`geometry_coupled`): a single density variable `rho` determines moving-link mass, COM, full 3x3 inertia, joint inertia and gravitational loading. It is not enough to fit `M` and `mu` separately and check `M >= mu*r` after fitting. A support centroid is not itself a mass centroid.

### A. Reconstruct a tentative mass geometry and COM, independently of force fitting

1. Reuse the **final frozen moving-link geometry and coordinate registration**, never re-train or modify Gaussian PLYs. Prefer an already validated, watertight moving-link mesh; if absent, implement a **separately labeled tentative geometric mass proxy** derived from the masked multiview RGB-D / object moving-part segmentation (optionally aided by 3DGS for surface geometry), with documented completeness, watertightness, thickness, frame and sampling bias. A standard 3DGS Gaussian's covariance, opacity or density of splats is **not** material mass density. Do not compute a physical COM as the naive mean of Gaussian centers. If only a point cloud exists, create a geometric *preview* (not a certified volume), explicitly choose solid-vs-shell, and propagate alternative plausible geometries as scenarios.
2. For a **uniform-volume-density solid** at reference pose, integrate unit-density mass and inertia:

   \[m_1=\int_{V}dV,\qquad \mathbf c_0=\frac{1}{m_1}\int_V\mathbf x\,dV,\qquad
     \mathbf I_{{\rm COM},1}=\int_V\left(\|\mathbf x-\mathbf c_0\|^2\mathbf I_3-(\mathbf x-\mathbf c_0)(\mathbf x-\mathbf c_0)^\top\right)dV.\]

   For an **equal-thickness shell**, integrate area and use one **areal density** parameter `sigma = rho_volume * thickness`; do not mix areal and volumetric density units. Explicitly record which density is fitted. For `segmented_material`, do not assert one density without material priors and sufficient independent dynamics.
3. Ensure `c0`, `I_COM,1`, `axis a`, `origin o`, and the Gaussian/mesh reference pose all lie in the **same rigid coordinate frame**, consistently using the accepted joint's `q=0` or canonical reference configuration. Check transform and q-offset conventions. COM is **fixed in the canonical moving body** and rigidly transformed at any q.
4. **Compute the perpendicular COM vector analytically** for revolute joints, without fitting its direction as a free dynamics parameter:

   \[\boxed{\mathbf r=\mathbf c_0-\mathbf o,\quad
     \mathbf r_\perp=(\mathbf I_3-\mathbf a\mathbf a^\top)\mathbf r,\quad
     r_\perp=\|\mathbf r_\perp\|,\quad
     \hat{\mathbf u}=\mathbf r_\perp/r_\perp.}\]

   Require normalized `a`; handle `r_perp ~ 0` as a degeneracy (COM near axis -> small gravity lever; do not divide by zero). Note that `o` may be any point along the hinge line: `r_perp` must be invariant to changing `o` by `lambda*a`.

### B. One density governs both inertia AND gravity

For **revolute** with `R_a(q)` the rigid body rotation about `a` from canonical q=0:

\[\boxed{I_{\rm axis,1}=\mathbf a^\top\mathbf I_{{\rm COM},1}\mathbf a+m_1\|\mathbf r_\perp\|^2,\qquad
M(\rho)=\rho I_{\rm axis,1}+I_{\rm reflected}.}\]

\[\boxed{G_R(q;\rho)=-\rho m_1\,\mathbf g^\top\left[\mathbf a\times R_{\mathbf a}(q)\mathbf r_\perp\right].}\]

**The mass moment is derived**, never separately optimized: `mu_derived = rho*m1*||r_perp||`. The geometry fixes the COM **direction and lever**; the single density `rho` fixes its mass and gravitation strength, while the same density fixes joint inertia. A known/declared `I_reflected` is separate drivetrain inertia and does **not** enter gravitational mass. If unknown reflected inertia is material, mark `M` ambiguous; do not quietly set it to zero and certify the physical density.

For **prismatic** under the single-rigid-link/no-reflected-inertia assumption:

\[\boxed{m=\rho m_1,\quad M=m,\quad G_P=-m\,\mathbf g^\top\mathbf a.}\]

Full URDF moving-link inertia is `m = rho*m1`, `COM = c0`, `I_COM = rho*I_COM,1`, transformed into the correct link frame. For a solid model `rho` is kg/m^3; for shell model use coherent mass-per-area or thickness conversion.

The declared coordinate convention is `g_base_link=[0,0,-9.81] m/s^2` because the operator identified the dataset z-axis as true vertical. Preserve its provenance (`DECLARED_OPERATOR` unless independently measured), do not block exploratory calculations merely because an obsolete registration gate still says "missing". The sign of `G` is **the derivative of potential** `U(q)=-m*g^T*c(q)`: finite-difference-test `dU/dq` for R and P. When the axis is parallel to gravity, `G_R` must vanish regardless of COM, and small axis-tilt uncertainty must be tested before attributing tiny loads to gravity.

### C. Physics residual shared by GeneralPINN, GeneralAugmentedEKF and rollout

With `rho` the single inertial/gravitational mass parameter and `g(q; rho)` as above, the moving residual is

\[\boxed{r(t)=\left[\rho I_{\rm axis,1}+I_{\rm reflected}\right]\ddot{\hat q}(t)
  +c\dot{\hat q}(t)+F_c\operatorname{sgn}(\dot{\hat q}(t))
  +\rho G_{R,1}(\hat q(t))+b-Q(t)=0}\]

for R, and for P replace the inertia by `rho*m1` and gravity by `-rho*m1*g.a`. Keep the existing static-friction regime and optional switches. **GeneralPINN must backpropagate the strong/weak physics residual through `rho` in both inertia and gravity simultaneously.** No separate trainable `M`, `mu`, `gA`, or `gB` are allowed in this geometry-coupled mode. Its physics loss, data loss, sticking loss and pulse integral loss remain as defined in §2.

**GeneralAugmentedEKF must include `log(rho)` as a single augmented static physical parameter**, not separate `log(M)` and `mu` states. Its transition sensitivities must contain both `d M/d rho` and `d G/d rho` (and `dG/dq`); derive all from the same `ModelSpec` as PINN. EKF observations remain q(t), signed measured Q is the input, and no new independent COM measurement is inferred from q. The multi-episode posterior and held-out policy remain as in §3.

`GeneralRolloutFit` must use the **identical** density coupling to provide a meaningful third estimator; `GeneralIntegralLS` initializes the same `rho` with inertia and gravity contributing to **one column**, rather than independent M/mu columns. For example, the weak second-integral regressor for rho includes `I_axis,1` times integrated acceleration **plus** the double integral of `G_R,1(q)`; sign checked against `M qdd + G = Q`.

### D. Two explicit levels of claims and no accidental fallback

- **`geometry_coupled` (preferred physical model)**: usable for exploratory estimation on a *tentative* geometry with full provenance and geometry uncertainty; **production export** only if geometry quality, density-model assumption, kinematics, force calibration and independent validation satisfy the corresponding gates. For tentative geometry, mass/COM/tensor are **geometry-assumed**, not measured ground truth.
- **`joint_space_gravity_coupled` (conditional diagnostic only)**: if no credible geometric volume/shell exists, use the operator-declared vertical and a support-derived **assumed** COM direction as an explicit ablation, but keep `M` and `mu` marked **independent and not fully physical**. Never call it a unified-mass model; do not export mass/COM/URDF from these independent values.
- **`joint_space_only`**: permits informative dynamics diagnostics without geometry, but any effective gravity basis is not automatically true gravity.
- Do **not** silently downgrade the requested `geometry_coupled` physical run to a different model without a distinct diagnostic version, status and explicit reason.
- **Uncertainty:** evaluate alternative mass geometries (solid vs shell, plausible missing internal surfaces, thickness, segmentation) and accepted hinge/axis uncertainty, separately from q/force statistical uncertainty and from hypothetical force-scale sweeps. For the fridge specifically compare estimated 6-degree tilted axis with independently evaluated vertical-hinge candidate; never force axis vertical by fiat. The gravity moment depends on axis tilt and COM lever, so a good training fit alone cannot certify the mass distribution.

### E. Code and tests that must be added

**Existing modules to extend/reuse:**
- `forceartgs_g9/dynamics/mass_properties.py`: unit-density COM, shell/solid inertia and diagnostics.
- `forceartgs_g9/dynamics/geometry_inertia.py`: geometric mass model with normalized `a`, `r_perp` and true source coordinate frame, closed-form `I_axis,1`, `G_1`, inferred full inertials.
- `forceartgs_g9/estimators/model_spec.py`: one density parameter and one shared `M(rho)` and `G(q;rho)` for all estimator methods, with *no independently fitted mu when geometry_coupled*.
- `forceartgs_g9/estimators/general_integral_ls.py`: correct one-column density initializer.
- `forceartgs_g9/estimators/general_pinn.py`, `general_ekf.py`, `general_constrained.py`: identical density coupling and loss/state derivatives.
- `forceartgs_g9/adapters/forceartgs_g9.py` and `forceartgs_g9/core.py`: geometry sources, true frame mappings and provenance; do not confuse support centroid with COM.
- `forceartgs_g9/export/dynamics_v2.py`: only geometry-consistent mass/COM/inertia, no duplicate gravity effort in simulation.

**Mandatory added tests:**
1. A box and an asymmetric off-axis rigid part: analytic mass/COM/tensor, `r_perp` projection, unit-density inertia, equality of mass inferred from inertia and gravity. Check hinge-origin shift along the axis leaves `r_perp`, I and gravity unchanged.
2. Revolute gravity `G_R(q)` equals finite-difference `dU/dq` at multiple q and matches a direct rigid-body torque calculation. Vertical hinge gives zero gravity torque even with eccentric COM; small tilted hinge scales gravity as expected.
3. Prismatic `G_P=-rho*m1*g.a`; horizontal axis gives zero, tilted axis has the correct sign and shared mass, and known reflected inertia is never treated as gravitational mass.
4. Both a shell and a solid: areal/volume density units, centroid and inertia, mass-properties positive definiteness/semidefiniteness as appropriate, explicit failure/preview for incomplete geometry.
5. PINN autograd gradient of `L_physics` w.r.t. **one** `rho` receives nonzero contributions from **both** `M(rho) qddot` and `G(q;rho)`; verify via finite differences. No independent `mu` or `M` in trainable vector.
6. EKF augmented state includes `eta_rho` and transition Jacobian gets the two shared sensitivities; Joseph PSD and no false precision at static-friction switching.
7. IntegralLS, rollout, PINN, EKF agree on `ModelSpec` parameter names, input units, force sign, geometry provenance and predictions for the same density-coupled synthetic R/P episodes.
8. Synthetic varied-push trials recover one true density, correct COM-derived gravity phase, full inertia, c/Fc/Fs (when excited). Perform a **density-vs-inertia-only-gravity-only** ablation and independent withheld-force replay.
9. Intentional erroneous COM/axis/geometry hypotheses fail or are flagged scenario-sensitive; plausible training fits must not be called measured physics.
10. Missing credible mesh triggers `joint_space_only` or `joint_space_gravity_coupled` **diagnostic** with transparent status, not invented mass and no production simulator asset.

**Definition of done for this operator correction:** `GeneralPINN` physics residual, `GeneralAugmentedEKF` transition, `GeneralRolloutFit` and `IntegralLS` all share **one geometry-derived COM, one density parameter, one joint inertia, one gravity law and one physical link tensor**, with consistent mass and independent validation. The old separate `M`/`mu` mode is clearly relegated to an assumed diagnostic.

## 2. GeneralPINN: direct neural dynamic-equation residual is mandatory

**New file:** `forceartgs_g9/estimators/general_pinn.py` (+ a reusable `pinn_losses.py` if appropriate).

- For each identification episode k define a differentiable neural trajectory \(\hat q_{\phi_k}(t)\) (or one network conditioned on an explicit episode ID). **Physical parameters \(\theta\) are shared across all episodes**, while q0/v0 and trajectory nets are episode-specific.
- Use a small smooth `tanh` MLP with float64 as the simple reproducible starting point. Normalize network input time for conditioning but apply chain rule/autograd so derivatives are in SI seconds. q(t) is a **learned trajectory**, not a network predicting physical parameters from videos.
- PyTorch autograd MUST calculate `qdot = grad(q.sum(), t, create_graph=True)` and `qddot = grad(qdot.sum(), t, create_graph=True)`, with no `.detach()`, `.numpy()` or finite-difference step in the physics residual graph.
- Jointly train \(\phi\) and \(\theta\); initialize \(\theta\) from the existing general integral LS when valid, and train with deterministic multi-start. Optionally warm up the data loss first; report this and all training settings.
- Collocation times cover the full moving trajectory and oversample force-change intervals, especially the fridge's approx. 80 ms impulse. Compute Q(t) from the **native 50 Hz** canonical force record using documented ZOH semantics. Avoid collocation exactly on discontinuities; analyze each side and include interval/weak-form residuals.

Core mandatory loss (not just an optional auxiliary):

\[
 L_{\rm phys}^{(k)} = \frac1{|C_k|}\sum_{t\in C_{k,move}}
     \rho_H\left( \frac{r(t;\theta,\hat q_{\phi_k})}{s_r(t)}\right),
\quad
 L_{\rm data}^{(k)}=\frac1{N_k}\sum_j
     \rho_H\left( \frac{\hat q_{\phi_k}(t_j)-q_{kj}}{\sigma_{q,kj}}\right).
\]

The physics residual is the **full equation with inertia, gravity, viscous, Coulomb, and optional bias**. Physics scaling \(s_r\) must be declared/grounded in force/noise and derivative uncertainty, or use documented nondimensional scales; it must not be tuned on held-out test outcomes. Multiple trajectories share the same physical \(\theta\).

**Sticking:** In independently supported resting intervals add loss on \(\dot{\hat q}\approx0\), \(\ddot{\hat q}\approx0\), and squared positive part of \(|Q-G-b|-F_s\). Do not insert kinetic sign friction at zero velocity. At breakaway transitions use a *one-sided/censored* threshold likelihood; if no trustworthy rest/breakaway samples, report `Fs NOT_IDENTIFIABLE`. Do not classify stuck/moving from fitted \(F_s\) alone. Do not silently smooth away stick/slip events just to enable differentiability.

**Impulse/weak loss:** Add a separate interval constraint including the measured input impulse (native time grid), without numerical differentiation of the noisy observed q:

\[
 L_{\rm impulse}=\sum_{[t_a,t_b]}
 \rho_H\left(\frac{M[\dot{\hat q}(t_b)-\dot{\hat q}(t_a)]
    +\int_{t_a}^{t_b}(c\dot{\hat q}+F_c\operatorname{sgn}(\dot{\hat q})+G(\hat q)+b)dt
    -\int_{t_a}^{t_b}Q(t)dt}{s_J}\right).
\]

Use pulse windows and surrounding coast. Strong-form moving residual remains mandatory; the weak impulse loss supplements it. Include proper uncertainty in impulse integral; four 50Hz samples do not imply the true pulse is fully resolved.

**Training objective:** `L = λ_data L_data + λ_physics L_physics + λ_stick L_stick + λ_impulse L_impulse + λ_ic L_ic`; weights/normalization logged and predeclared, with ablations. Softplus/log parameterizations must respect physical bounds. Gradient tests must prove nonzero finite gradient from **physics residual** to M,c,Fc and mass/gravity parameter(s), not just to neural weights.

PINN outputs: joint trajectory and derivatives at query times; physical parameters with status/uncertainty, training loss decomposition, collocation sampling, optimizer convergence, data-fit/physics-fit diagnostics, input hashes, and open-loop replay using frozen fitted parameters **not** the trained trajectory net.

## 3. GeneralAugmentedEKF: parameters in augmented state

**New file:** `forceartgs_g9/estimators/general_ekf.py` (name `GeneralAugmentedEKF`); use the same `ModelSpec` / `general_1dof` physical dynamics as PINN and rollout. The existing `ForceArtGS-Physics.AKF` is a frozen viscous baseline only.

For a moving interval, define

\[
 x=[q, v, \eta_M,\eta_c,\eta_{Fc},\eta_{Fs-Fc},\eta_g,\eta_b,\ldots]^T,
\]

**only for enabled/independently estimable parameters**; \(\eta\) are positive/constraint-preserving parameter coordinates, and gravity parameters follow the selected source (density \(\rho\); effective \(g_A,g_B\); conditional \(\mu\) if implemented). Constant parameters obey \(\dot\eta=0\) in the process model. Dynamic state obeys

\[
 \dot q=v,\quad
 \dot v = \frac{Q(t)-cv-F_c\operatorname{sgn}(v)-G(q)-b}{M},\quad
 \dot\eta=0.
\]

Use measured generalized Q(t) with its time stamps; `z_k=q_obs(t_k)+ν_k` and \(H=[1,0,\ldots]\). **Do not feed a finite-difference velocity derived from q as an independent sensor measurement** without its correlated covariance. Support irregular RGB-D q update times, more frequent robot-force samples, zero-order-hold force, and exact source-index association.

At every force step: integrate a numerically stable hybrid transition \(x_{k+1}=f(x_k,Q_k,\Delta t)\) with parameter sensitivity Jacobian \(F_k=\partial f/\partial x\) (autodiff or verified analytic/numerical derivative), propagate \(P^- = F_k P F_k^T + Q_k^{process}\), then if q observation is present do innovation and **Joseph covariance update**:

\[
 K=P^-H^T(HP^-H^T+R)^{-1},\quad
 x^+=x^-+K(z-Hx^-),\quad
 P^+=(I-KH)P^-(I-KH)^T+KRK^T.
\]

Include measurement variance from recovered q confidence and propagate uncertainty from force, lever and hinge geometry into transition/process covariance. `theta` is **constant** (parameter process noise zero by default; optional carefully audited random walk for model mismatch must not be called constant-parameter identification). Initialize parameter mean from general IntegralLS / informed priors, and parameter covariance broad but finite, with all priors recorded. Do not select EKF hyperparameters on held-out evaluations.

**Hybrid static friction:** At verified rest, if \(|Q-G-b|\le F_s\), propagate v=0 and q constant. When the threshold is exceeded, transition to sliding using the same Coulomb/viscous/general gravity equation; locate switch events robustly. EKF linearization near switches is problematic: expose event flags, innovation/NIS, covariance diagnostics, and avoid interpreting spurious linearized precision as identifiability. A bank of modes / multiple-model filter is permitted if needed, but remain explicit. \(F_s\) should be unresolved without breakaway information.

Run per-interaction forward filter with **shared object parameter posterior** across training interactions while resetting episode q/v states and preserving parameter/covariance posterior. If episode order affects estimates, report and test this; batch multi-episode estimation or an offline smoother/iterated EKF may be a robust option. RTS smoothing may improve offline state accuracy but MUST NOT access held-out episode data. Preserve positivity, covariance PSD and invariance to joint axis sign/q convention.

EKF outputs: filtered q/v and uncertainty, parameter posterior mean and covariance, innovations and NIS, hybrid regime timeline, per-sample predicted generalized force/acceleration, physics residual diagnostics, and **independent open-loop rollouts with frozen final parameters**.

## 4. Integration into G9

Modify:

- `forceartgs_g9/config.py`: `fitting.methods` supports at least `general_pinn`, `general_akf`, `general_rollout_fit`, `general_integral_ls`; explicit per-method options under schemas, no silent ignored flags.
- `forceartgs_g9/core.py`: run all enabled estimators on **identical** training interactions and the **same resolved ModelSpec**; save separate outputs `estimator_results/general_pinn.json`, `general_akf.json`, `general_rollout_fit.json`, then one comparable table. No default winner based on training loss.
- `forceartgs_g9/estimators/model_spec.py`: ensure the three methods obtain identical enabled/disabled terms and gravity coupling. The user's ongoing `joint_space_gravity_coupled` experiments must be audited/merged carefully, without guessing COM or declaring physical validation.
- `forceartgs_g9/validation/general_identifiability.py`: estimator-specific uncertainty (PINN profile/ensemble/bootstrap, EKF covariance **and independent structural/excitation checks**, rollout covariance); never equate small filtered covariance with physical identifiability. Sensitivity scenarios (force-scale ±5%, lever ±1.5 cm) remain scenario ranges, not measured Gaussian errors.
- `forceartgs_g9/validation/heldout_prediction.py`: frozen-parameter **open-loop** predictions for each estimator, using input Q and declared initial state only. No fitted q trajectory net or EKF q-filter path may be passed off as a new-force prediction.
- `estimotion_2part/joint_discovery/robot_record.py`: single canonical record including measured q, native force, the tare correction already applied, explicit sign −1 from metadata; no re-taring or double negation. Do not quietly coerce unsigned force or imply hardware clock verification from index join.
- `forceartgs_g9/adapters/forceartgs_g9.py`: supply native 50Hz force and q timestamps to both methods; pulse windows and separate coast rows; no SNR computed only over peak q speed as a reason to ignore initial input impulse.
- `docs/G9_GENERAL_DYNAMICS_IMPLEMENTATION_REPORT.md`: precise method comparisons, non-identifiability and force-calibration provenance. No misleading assertion that PINN and EKF were extended merely because viscous baselines ran.

**When geometry is missing** do `joint_space_only` or distinctly labeled, assumed `joint_space_gravity_coupled` conditional diagnostics; do not invent a physical COM or certify a link tensor. If geometry is available do `geometry_coupled`: shared density enforces the complete inertia/gravity relation. No production URDF with invented mass/COM/tensor. No changes to pretrained Gaussians, front-end kinematics, or shared PArtTwin exporter except necessary handoff fields backed by actual validation.

## 5. Mandatory tests and proof

Add `tests/test_g9_general_pinn.py`, `tests/test_g9_general_akf.py`, `tests/test_g9_general_method_parity.py`, and meaningful integration tests. Tests MUST include:

1. Direct autograd derivatives of q(t)=A sin(ωt) vs analytic qdot,qdd in physical seconds.
2. Unit test exact **full** PINN physics residual equals zero on an analytic trajectory and correct Q: gravity ON, damping, Coulomb, optional bias; turning any term OFF removes it from residual and from trainable parameters.
3. Physics loss backprop gradient from `L_physics` to **physical parameters** and neural net (must not detach Q effects inadvertently).
4. Differentiable continuous Q=const and discontinuous ZOH impulse; strong collocation avoids jumps, weak impulse loss equals analytic momentum jump in a frictionless controlled case.
5. Sticking inequality and breakaway both enforced; `Fs` cannot be identified from sliding-only data. No `sign(0)` kinetic-friction artifact.
6. PINN synthetic R/P recovery with FULL gravity+damping+Coulomb+static and multiple varied pushes; compare held-out open-loop trajectories (not trained q-net).
7. Augmented EKF transition/Jacobians against finite-difference or analytic derivatives away from switching; Joseph update PSD/symmetry and valid NIS.
8. EKF propagates native 50Hz force and updates with sparser/irregular q observations; initial 80ms impulse is not dropped.
9. EKF synthetic R/P parameter recovery and uncertainty coverage, both identifiable and explicitly unidentifiable regimes.
10. Zero-force coast, one-direction constant-speed confounds and near-vertical hinge: no false identification of M/c/gravity/constant bias.
11. Correct force-on-object sign once; missing metadata sign raises, inverted sign fails physical consistency; test measured Q vs injected signed Q.
12. Gravity U derivative matches finite differences in R and P; world z is vertical by explicit declaration only. Axis-tilt sensitivity and conditional support-centroid COM-direction assumption reported.
13. All three estimators obtain exactly the same q, Q, contact windows, joint geometry, physics terms and parameter priors; compare frozen `PINN`/`AKF` viscous baselines without editing them.
14. R/P without a mass mesh still fit joint-space parameters, but do not claim full link inertia/URDF readiness.
15. Independent validation episode: fit on A, freeze θ, predict B without assimilating B observations except the declared initial state; fail on leakage.
16. 57 existing G9 tests and 183 existing joint tests continue to pass. Physics original frozen suite unchanged.

**Real-scene outputs:** re-run at least fridge and suitcase exploratory modes with **GeneralPINN, GeneralAugmentedEKF, GeneralRolloutFit side by side**, all full/viscous/gravity variants explicitly configured. For fridge separately report force pulse integral over source rows 229–232, pre-/post-pulse velocities with uncertainty, pulse SNR, coast decay, suspected gravity/axis tilt sensitivity, and per-method parameter identifiability. For suitcase report the failed viscous-only adequacy and model-dependent/uncertified kinematics; never label its damping a validated physical property.

**Definition of done:** the PINN training loss explicitly contains the general dynamic equation residual, the augmented EKF actually estimates dynamic parameters in its state, tests exercise FULL physical terms (not merely viscous), and both are compared against existing rollout/LS on synthetic and real data with identifiability, same inputs and no held-out leakage. Report what remains unidentifiable instead of inventing coefficients.

## 6. Deliverables / operational instructions

- Actual code and tests in the current branch (inspect `git status` and recent Codex gravity-coupling edits first; do not overwrite).
- New config example selecting `general_pinn`, `general_akf`, `general_rollout_fit`; per-method reproducible hyperparameters and seed.
- `docs/G9_GENERAL_PINN_AKF_IMPLEMENTATION_REPORT.md` with architecture, derivations, actual commands, synthetic/real tables, training cost, convergence, NIS, uncertainty/fit/held-out results, experiment dates and hard limitations.
- Report git diff and tests. **No implicit push/merge, no co-author line**, unless explicitly authorized by the operator.

**Do not claim "PINN and EKF are implemented for full G9" if they are only called through the old viscous-only `frozen_baselines.py`.**
