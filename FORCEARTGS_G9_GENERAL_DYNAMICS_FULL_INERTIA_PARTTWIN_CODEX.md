# CODEX IMPLEMENTATION DIRECTIVE — ForceArtGS G9: General Articulated Dynamics, Geometry-Constrained Full Inertia, and PArtTwin1D Physics Export

**Status:** implementation directive, not an implementation report. **Deliver actual working code, tests, CLI, per-scene diagnostic reports, and integrated final-stage hooks; do not stop after creating a design document.**

**Repositories**

- Main orchestration: [`SamueleMara/ForceArtGS`](https://github.com/SamueleMara/ForceArtGS), audited public `main` at `628e3ee47d1c38313a74dcf3cc56e7fae01d0629` on 2026-11-20. Reinspect the actual local HEAD before editing; do **not** reset or overwrite user changes.
- Dynamics library: [`andrearichichi/ForceArtGS-Physics`](https://github.com/andrearichichi/ForceArtGS-Physics), audited `main` at `f3824316272d85cb8d788166372c81ac3105ca7c`. Prefer a pinned external checkout/dependency or, with permission and correct ownership, a Git submodule; do not copy source into ForceArtGS. Its `_validated/` benchmark bodies, frozen method settings, reference outputs, and provenance are **read-only**.
- Asset builder/consumer: [`SamueleMara/PArtTwin1D`](https://github.com/SamueleMara/PArtTwin1D), public `main` at `b67863ae929d09f468c45ffece518f3d9ff2e33d`. The ForceArtGS G8 report describes a **local, not publicly published** `shared-asset-export` branch implementing `parttwin1d.cli.build_asset`. Inspect the user's actual workspace and make this dependency reproducible; **do not claim the published main already has it**.
- Reference exporter: [`SamueleMara/RigEuler3DGS`](https://github.com/SamueleMara/RigEuler3DGS), branch `phase5-sim-asset-export` at `8464d4fd6e377a53ad5ebc0580eb4c5942e13a92`, containing validated physics schema, collision and inertia calculation, and URDF/USD/MJCF writers. **One canonical exporter**, not another independent one inside ForceArtGS.
- Existing specification: `FORCEARTGS_MULTIMODAL_TCP_FORCE_SLIP_MODES_CODEX_IMPLEMENTATION.md`. The newer ForceArtGS code and `docs/MULTIMODAL_TCP_CONTACT_SLIP_IMPLEMENTATION_REPORT.md` take precedence where the implementation has already evolved.

## 0. Non-negotiable objective and architecture

Create the **final dynamics-identification layer** after the accepted joint/support and articulated 3DGS reconstruction. It must use the **final accepted metric joint and recorded applied force** to estimate a physically meaningful **general one-degree-of-freedom dynamics model**, optionally simplify it through switches, reconstruct physically consistent full moving-link mass properties using geometry, and feed them into the **existing** PArtTwin1D canonical asset and URDF/USD/MJCF exporters.

The physical effects selectable by configuration are **inertia, viscous damping, Coulomb friction, static/breakaway friction, gravity, and a clearly labelled nuisance generalized-effort bias**. Support fitting, simulation, diagnostics and export with these switches. The **default scientific configuration is the full model**, but enabling an effect is *not* permission to invent its coefficient: if the observations cannot identify it, report `UNIDENTIFIABLE`, require a verified prior, or block production export. An explicitly selected simplified model must stay simplified in simulation and validation; do not silently compare a gravity-free fit against gravity-enabled rollout.

**Final sequence:**

```text
Calibrated multiview RGB-D + object masks + tracking + robot TCP/wrench
   -> existing iterative pre-3DGS multimodal joint/support frontend
      (sliding / stick_soft / stick_hard; robot contact; force-informed R/P selection)
   -> accepted joint authority in real metric W + q(t) + uncertainty + provenance
   -> train STATIC 3DGS and MOVING 3DGS SEPARATELY
   -> existing two-part blend -> splatfacto-art -> final 3DGS/part handoff
   -> NEW G9 dynamics fitting (ForceArtGS-Physics)
      -> calibrated force/clock/contact audit
      -> fit general dynamics with requested on/off terms
      -> geometry-constrained mass/COM/full 3x3 inertia reconstruction
      -> model/parameter identifiability and independent-force prediction
      -> accepted calibrated physics manifest OR explicit partial/blocked status
   -> existing G8 shared asset builder (or second, explicit G8 finalization step)
      -> unchanged standard visual/static.ply + visual/moving.ply
      -> collision geometries + resolved dynamics.yaml
      -> maniskill/model.urdf + isaac/model.usd + mujoco/model.xml
      -> engine runtime checks, 3DGS visual pose checks, versioned asset
```

**Architectural interpretation:** G8's existing *visual handoff* may run before G9 to supply frozen geometry; G9 then creates physics, and an atomic, final G8 physics build is the **last** step. Do **not** force G9 to wait for an already physics-complete G8 asset or create a circular stage dependency. Name phases unambiguously: `visual_handoff -> dynamics -> final_physics_export`.

**Preserve** current joints/trackers, joint-type selection, frozen-joint training, the two standard Graphdeco PLY files, PArtTwin1D simulator support, and existing legacy/benchmark workflows. **Do not train additional Gaussians inside G9; do not modify final PLY fields or global poses.**

## 1. Audit before writing code: explicit findings required

1. Inspect actual local branches, uncommitted changes, Git submodules and available Python environments. Compare with above pinned commits. Never reset local branches or run `git stash` while jobs are running. If the shared PArtTwin builder exists only in another checkout/branch, record its path/HEAD; if unavailable, implement and test a clean adapter against its documented schema, mark end-to-end export BLOCKED, and do **not** forge successful runtime evidence.
2. Inspect `ForceArtGS/pipelines/joint_aware_reconstruction.py` stages: as of audited HEAD `audit, discover, multimodal_refine, select, authority, dense, build, parts, blend, art, refine, compare, video, final, export`. `multimodal_refine` already handles TCP/force and joint selection; `refine` is marked NOT_RUN. **Do not reimplement this frontend as part of dynamics.**
3. Inspect its G8 `tools/export_parttwin_handoff.py`, `--export-parttwin`, `--dynamics-spec`, `--export-diagnostic`, and current `art`/`final` manifests; new stage must use the accepted branch's `joint_authority/v000` and final handoff, not legacy `estimated_joint.json` or normalised splatfacto frame B.
4. Read ForceArtGS-Physics `README.md`, `docs/QUICKSTART_SAMUELE.md`, `docs/FORCEARTGS_INTERFACE.md`, `docs/REAL_DATA.md`, `docs/PHYSICAL_MODEL_AND_METHODS.md`, `docs/DATA_FORMATS.md`; inspect `api.py`, `adapters/forceartgs.py`, `adapters/real.py`, `schemas.py`, `forces/projection.py`, `dynamics/extensions.py`, `validation/identifiability.py`, `export/parttwin.py`. Determine the exact installer/import and version pin.
5. Inspect PArtTwin's available builder and the RigEuler reference `recovery/sim_asset_export/{schema,physics,collision,urdf_export,usd_export,mjcf_export,validate}.py`. Reuse a **single canonical physics schema and writers**. Identify schema gaps instead of manually postprocessing URDF strings.
6. Identify which real captures have calibrated force axes/sign/tare/gravity/lag and independently supported contact. Existing report (2026-11-19): no real scene had *fully verified* wrench calibration; drawer lacked a robot archive. Do not promote synthetic tests, declared sensor-axis identity, fixed frame indices, or software contact heuristics into verified hardware calibration.
7. Read licensing/provenance docs before any redistribution: ForceArtGS-Physics contains unlicensed collaborative vendored research components. Treat it as a pinned dependency, never vendor `_validated/` into another public repo or publish its source/assets automatically without permission.
8. Write `docs/G9_DYNAMICS_REPO_AUDIT.md` recording exact commits, available files, path conventions, environmental/runtime availability, and unresolved blockers. Proceed with implementation without waiting for the owner when missing *optional* data; fail closed on unsafe physical output.

## 2. Dynamics model and exact sign/unit conventions

### 2.1 Physical scope

Initial supported system: **one ideal fixed-axis revolute OR prismatic joint, one rigid moving body, static base**, positive right-hand revolute or positive translation along the accepted joint axis, all in metric capture/robot frame `W`. No unmodeled parallel mechanisms, flexible bodies, multiple joints, compliance, distributed force actuation, joint motor model, or moving camera. All claims are conditional on these assumptions.

Use a single generalized dynamics core, where **all** physical effects have explicit switches:

\[
\underbrace{M\ddot q}_{\mathrm{inertia}}
+\underbrace{c\dot q}_{\mathrm{viscous}}
+\underbrace{F_c\,\mathrm{sign}(\dot q)}_{\mathrm{kinetic\ dry\ friction}}
+\underbrace{G(q)}_{\mathrm{gravity}}
+\underbrace{b}_{\mathrm{nuisance\ load}}
=Q_{\mathrm{robot}}(t),\qquad |\dot q|>\varepsilon_v.
\]

`M` is an effective scalar: kg m^2 for a hinge, kg for a slider; `c` has N m s/rad or N s/m; `F_c`, `F_s`, `b` have Nm or N; q in rad or m. For the stated 1-DoF rigid mechanism, the generalized Coriolis/centrifugal term is **zero**, and `M` is constant. Do not add unnecessary 6-DoF rigid dynamics or a free, unobserved C(q,dq).

Define gravity as the derivative of potential energy `U` (NOT the applied gravity torque):

\[
U(q)=-m\mathbf g_W^{\top}\mathbf c_W(q),\qquad
G(q)=\frac{dU}{dq};\quad
G_R(q)=-m\mathbf g_W^\top\big(\mathbf a_W\times[\mathbf c_W(q)-\mathbf o_W]\big),\quad
G_P=-m\mathbf g_W^\top\mathbf a_W.
\]

Gravity vector is physical acceleration in calibrated world coordinates, normally `(0,0,-9.80665) m/s²` **only after verified gravity registration**. Sign test: in the absence of robot effort and friction, qdd = `-G/M`. Verify finite differences of `U` numerically for R and P. Do not assume arbitrary COLMAP/world axes are gravity aligned. If the base frame is actually `base_link`, retrieve/verify base orientation relative to gravity.

At velocity near zero, static friction is **not** `F_s sign(0)`. It is a *set-valued reaction* that can hold the joint at rest:

\[
\dot q=0,\quad |Q_\mathrm{robot}-G(q)-b|\le F_s\implies \ddot q=0
\quad (\text{sticking}),
\]

otherwise motion begins in the sign of the net applied effort; during motion kinetic Coulomb `F_c sign(dq)` plus viscous damping apply. Use a documented complementarity/Karnopp event state machine (or carefully validated equivalent) with breakaway/re-stick thresholds, timestep convergence and no force-free energy creation. Require `F_s >= F_c >= 0` when those terms are on, unless a physically justified model explicitly documents otherwise. Distinguish **joint static friction** from **TCP-object sticking/zero-slip contact**: these are different phenomena and different configuration keys.

Optional `bias` is a constant *nuisance generalized load*, not an identified actuator torque and not a quantity inserted into URDF as friction. Never report a fitted `b` as a measured sensor tare: raw sensor tare must be established separately. An unidentifiable bias may still be fitted in diagnostic studies but blocks claims about confounded physical terms.

### 2.2 Parameter switches and model consistency

Switches live under `dynamics.model.terms`: `inertia`, `viscous`, `coulomb`, `static`, `gravity`, `bias`. Full model default is all first five ON, bias OFF unless the user chooses it. Keep inertia ON for dynamic models; OFF allowed **only** under explicitly selected `quasistatic` mode, with separate equilibrium equations and no dynamic rollout certification. Never treat `M=0` as a valid rigid-body link in an ordinary integrator or URDF.

When a term is disabled, it contributes identically ZERO to fit and independent prediction and is recorded as `disabled_by_user`; its parameter is `null`, not an estimated zero. Do not reuse a result fitted with gravity ON for a gravity-OFF simulation (or vice versa). A parameter enabled but unsupported by data is `unidentifiable`; user can choose a simpler model or provide a verified prior. `auto_model_selection` may be offered only as **opt-in**, with explicit nested-validation procedure and the chosen switches recorded.

The fixed benchmark model `M qdd + c qdot = Q` is exactly the configuration: inertia ON, viscous ON, coulomb/static/gravity/bias OFF. Frozen historical benchmark algorithms and results must be unchanged and independently reproducible.

### 2.3 Applied generalized effort: signed and contact-aware

Consume `ForceRecord` with **force on the object**, world-frame force `F_W(t)`, application point `p_W(t)` and optional separately calibrated pure couple `M_W(t)`. For a prismatic joint:

\[ Q_P(t)=\mathbf a_W^\top\mathbf F_W(t).\]

For revolute:

\[ Q_R(t)=\mathbf a_W^\top\{[\mathbf p_W(t)-\mathbf o_W]\times\mathbf F_W(t)+\mathbf M_W(t)\}.\]

No use of absolute magnitudes, sign-blind lever factors, or reusing the legacy ForceSAPIEN magnitude-only benchmark projection on reversing real motions. Preserve the paired sign convention `(axis, q) <-> (-axis, -q)` and frame transforms.

- `p_W(t)` is the calibrated instantaneous contact/application position; under sliding it may move over the object, under verified hard sticking it tracks one material point. **The same generalized force formula works in both contact modes.** Never force TCP to be a static material point in sliding mode.
- Use `contact_valid`, release/regrasp intervals and uncertainty from the multimodal frontend. Contact-confirmed samples may receive force; **no-contact** samples have zero robot effort only if independently known and other force contacts are absent. Unknown contact MUST NOT be silently marked zero or positive.
- A measured 6D wrench must be transformed with its actual sensor frame/torque origin and translated to application point before using its pure couple. If torque origin is unknown, **omit torque channel** and use only `p × F` for a justified point-force contact; do not double-count the moment of the same force.
- Reject frames with missing force, invalid TCP, unverified world registration, unknown sign/sensor axes, uncalibrated hardware clock, excessive gravity/tare uncertainty, or invalid spatial contact evidence. Each rejection gets an exact reason code and count.
- Sampling is irregular where needed; state explicitly whether force is ZOH, interpolation, or bandwidth-limited resampling. Source frames and parquet timestamps are not equivalent to verified hardware clock alignment. Do not tune time lag on evaluation episodes.
- Force during static tool motion may include tool inertia and gravity; documented tare/gravity compensation applies to the sensor/tool, **not automatically to the object's gravity**, which is a separate model term. No parameter should absorb both unnoticed.

## 3. Geometry-constrained full inertial properties (mandatory feature)

**Critical:** a single joint-specific scalar inertia does NOT uniquely identify a general symmetric 3×3 inertia tensor, link mass and COM; however, **a chosen, validated geometry-derived mass-distribution model provides their *shape*, leaving one scalar density scale to identify**. Implement this. The result must be a consistent full rigid-body model, with truthful conditional provenance rather than an arbitrary tensor.

### 3.1 Input geometry and mass-model choices

Use the **canonical moving part** from the accepted reconstruction/handoff, with verified `T_capture_mesh`, canonical `q=0`, and `T_link_ply`. 3DGS Gaussian covariance/opacity is **not a physical density**. Do not integrate mass directly over Gaussian covariance without a separately justified physical-density model.

Supported mass distribution choices:

1. `uniform_solid`: **watertight, consistently oriented 3D moving-part surface mesh**, integration of exact polyhedral volume moments at unit volumetric density (units m, kg/m³). Validate topology, signed volume, orientation, self-intersection plausibility, hole-filling provenance and mesh alignment to the moving Gaussians.
2. `uniform_shell`: validated surface triangulation with unit areal density (kg/m²); integrate triangle area moments for thin shell, optionally finite declared thickness with an explicit error budget. Useful for hollow door/lid/drawer panels. Do not pass open shell into solid volume integrals or invent thickness.
3. `segmented_material` (second phase, opt-in): distinct known/pre-specified material regions with density ratios or measured material priors. A **single** joint inertia scalar cannot identify several free independent densities. Constrain relative densities from external sources or use multiple independent measurements and report rank/identifiability.
4. `user_link_inertial` (override/independent measurement): supplied mass, COM and full inertia with measured/estimated provenance and covariance. Check consistency against joint inertia; do not overwrite it with a single-axis estimate without explicit user approval.

Collision meshes may be suitable for `uniform_solid` only if closed and physically representative. Convex **collision proxies** may severely overestimate volume; do not automatically equate collision proxies with the actual mass distribution. Use a separate `mass_model_geometry` reference in the physics input or explicitly document and flag the proxy approximation. All mesh sources must be checked for unwanted robot/tool points and for canonical link assignment.

### 3.2 Unit-density geometry integrals and axis projection

For unit-density geometry calculate and store, in a clearly named *canonical capture frame* (convert afterward):

- `m_1 > 0`: mass under unit density; for `uniform_shell`, unit is 1 kg/m², for `uniform_solid`, 1 kg/m³.
- `c_1`: center of mass in metres.
- symmetric positive-definite full `J_com,1` in kg m² per density unit (all six independent components), with principal moments/eigenvectors and triangle inequality check.
- `r_1=c_1-o`, `a` normalized. Compute geometry's unit-density hinge inertia:

\[
I_{a,1}=\mathbf a^\top\left[\mathbf J_{\mathrm{COM},1}
+m_1\left((\mathbf r_1^\top\mathbf r_1)\mathbf I_3-\mathbf r_1\mathbf r_1^\top\right)\right]\mathbf a.
\]

For prismatic, `M_1=m_1` under the single rigid moving body assumption. Require finite `I_{a,1}>0`, `m_1>0` and physically valid geometry. For a revolute joint, changing the hinge origin **along the same axis** must leave `I_{a,1}` unchanged (unit test).

When the effective inertia/mass is physically attributable to the moving part (no unmodelled drive inertia/gearing/constraint/compliance):

\[
\alpha=\begin{cases}I_{\mathrm{eff}}/I_{a,1},&R\\M_{\mathrm{eff}}/m_1,&P.\end{cases}
\quad m=\alpha m_1,\quad \mathbf c=\mathbf c_1,\quad
\mathbf J_{\mathrm{COM}}=\alpha\mathbf J_{\mathrm{COM},1}.
\]

This is the *post-fit projection* version. Prefer the **joint geometry-coupled estimation** in §4 when gravity is enabled, which estimates alpha **jointly** from inertial and gravitational response. Then recompute `I_axis=alpha I_a,1`, verify predicted vs fitted effective value, and write an explicit `inertia_reconstruction_report.json` including mass-model assumptions, density units, shape moments, axis projection, uncertainty, and physical consistency checks.

The full 3×3 tensor is defined **about the COM**. Convert to the moving-link frame with rotation `R_L_C`: `J_L=R_L_C J_C R_L_Cᵀ`, `c_L=T_L_C c_C`. Do not apply parallel-axis terms twice: the exporter already writes COM-origin tensor to `<inertial>`.

### 3.3 Coupling inertia and gravity correctly

A fully consistent geometry-first model uses a *single* density scale `alpha`, so for a revolute joint:

\[
M(\alpha)=\alpha I_{a,1},\qquad
G(q;\alpha)=\alpha G_1(q),\quad
G_1(q)=-m_1\mathbf g_W^\top
\left[\mathbf a_W\times(\mathbf c_W(q)-\mathbf o_W)\right].
\]

For a prismatic joint:

\[
M(\alpha)=\alpha m_1,\qquad
G(\alpha)=-\alpha m_1\mathbf g_W^\top\mathbf a_W.
\]

Thus the *moving* equation for an identified rigid body is

\[
\boxed{\alpha\,[M_1\ddot q+G_1(q)]
+c\dot q+F_c\operatorname{sign}(\dot q)+b
=Q_{\rm robot},}
\]

with the gravity term dropped **only when gravity is disabled by configuration**. The above directly estimates a physically consistent full inertia through alpha and prevents fitting an arbitrary gravity amplitude inconsistent with the mass/inertia used in URDF.

If geometry is unavailable or invalid, offer a **joint-space-only diagnostic mode** fitting `M_eff` independently (and, if identifiable, a declared gravity-basis load), but **do not generate a complete link inertia tensor** or a gravity-correct URDF from that mode. A gravity basis that is not tied to geometry remains an *effective fit* and must not be silently converted into COM/mass.

With independent measured `m` or COM, use them as actual constraints with stated covariances, reconcile with the joint-space inertia, and fail/flag inconsistencies. When unmodeled reflected actuator inertia contributes to `M_eff`, fitting `alpha` to `M_eff` would overestimate real link mass; require a declared reflected-inertia term or abstain from full inertial export.

### 3.4 Geometry uncertainty and checks

- Monte Carlo / bootstrap accepted hinge axis/line, camera-to-world registration, contact point, input mesh perturbations and unit-density integrals. Propagate at least 1-sigma/95% intervals to `alpha`, `m`, COM, 6 tensor entries, `I_axis`, gravity profile. Distinguish measurement uncertainty vs mass-model systematic assumption (the latter cannot be made tiny by bootstrapping).
- Test centroid/geometry consistency, mesh units and anchor frame against final Gaussians; reject swapping links, wrong sign, wrong hinge position, millimetre-vs-metre errors and non-watertight `uniform_solid`.
- Require `J_com` symmetric/PD, positive mass, physically realizable triangle inequalities, COM plausible, and `projection_error` below calibrated tolerance. For prismatic, check that reported effective mass is consistent with a single uncoupled moving body.
- One scalar cannot uniquely recover arbitrary material heterogeneity. `estimated_uniform_solid_density` / `estimated_uniform_shell_areal_density` is valid provenance; it is **not** `measured_full_inertia`.
- The visual PLYs are immutable: Gaussian means, SH, rotations, opacities and stored coordinate transforms must not change as part of mass reconstruction.

## 4. Estimator implementation, retaining benchmark integrity

### 4.1 Keep existing algorithms frozen

`forceartgs_physics.api.identify/compare/predict`, `estimators/{pinn,backprop,mlp,akf,integral_ls}.py`, and especially `_validated/` implement the legacy viscous-only benchmark. Do not edit `_validated/`, change algorithm labels, replace frozen hyperparameters, or alter benchmark results. These methods remain **baseline comparisons for the 2-term model** and initializers/diagnostics for the general model; it is scientifically wrong to describe them as full gravity/static-friction estimators without actually extending their physics and validating the extensions separately.

Use the existing `dynamics.extensions` `ExtendedModel`, `simulate`, `identify_extended_ls` as reference/starting point, but create a **new versioned general implementation** (`forceartgs_physics.dynamics.general_1dof` and `estimators.general_constrained`) with its own benchmarks and method names. Do not change the meaning of `model='viscous_1dof'` or `parttwin.dynamics/v1` silently.

### 4.2 Required fitting algorithms

Implement at least:

- `general_integral_ls`: robust *weak/integral* initializer for moving intervals for linear-in-parameters terms. In **geometry-coupled** mode, the regressors for alpha combine inertia and gravity as one column. Avoid directly differentiating noisy q twice; derive proper integral basis functions with the force ZOH integration handled exactly or to bounded tolerance. Return rank, singular values, correlations and coefficient constraints.
- `general_rollout_fit`: bounded multi-start maximum-likelihood/robust least squares fitting by integrating the complete stick/slip ODE or hybrid event model and comparing **q on raw observation timestamps**. Prefer stable semi-implicit or event-aware integration for static friction; provide numerical step-convergence tests. Optimize positive parameters through constrained transforms/bounds; support per-interaction q0, v0 and any declared nuisance initial variables while sharing physical parameters across repeated demonstrations of **the same object/joint**. Do not allow per-time latent force or arbitrary contact wrench to absorb the residual.
- Optional additional general physics estimators may be implemented as separate `general_akf`, `general_pinn`, `general_backprop` only with meaningful integrated physics, tests and own labels. **Do not claim all four frozen methods fit the new model.** Running frozen methods as baselines is still required when requested.
- `general_model_selection` compares user-chosen variants (full, no_gravity, no_static, no_coulomb, viscous_only...) on truly withheld interactions, not training fit alone. This is opt-in; the default respects the requested switches even if the result must abstain.

The parameter vector in geometry-coupled mode is `alpha>0`, `c>=0`, `Fc>=0`, `Fs>=Fc` when enabled, plus optional `b` and registered nuisance initial states. In joint-space-only mode use `M_eff>0`, and explicit *unknown/unsupported* full tensor fields. Do not fit a separate free `M_eff` **and** an independent free density that duplicates inertia.

For each enabled term, store `estimate, units, CI95, fitted_or_fixed, evidence, identifiable, failure_code`. For each disabled term store `null, disabled_by_user`. For known prior values store prior source, uncertainty and an `ASSUMED` vs `MEASURED` label; user priors must not be silently reclassified as identified parameters.

### 4.3 Critical identifiability and excitation

Implement an appropriate general-model identifiability diagnostic, not just the existing viscous CRLB copied onto a higher-dimensional system:

- sensitivity of measured q trajectory to all enabled physical parameters **and nuisance terms**; Jacobian/Fisher information, rank, scaled singular values, correlations and realistic q/force uncertainty;
- block/camera/surface correlated visual-q errors, force bias/tare uncertainty, lag and joint-axis/hinge/contact uncertainty propagated or swept in sensitivity. A local white-noise CRLB is only one diagnostic and not proof of model adequacy;
- multi-interaction information accumulation using **shared parameters with independent initial states and per-interaction nuisance**, not the existing `Estimator.fit_many` median aggregation as a substitute for a joint fit;
- explicit bounds and confounds: near-constant qdot gives poor inertia excitation; one-direction Coulomb and constant bias are collinear; prismatic constant gravity is confounded with constant generalized bias/force offset; vertical hinge may have almost no gravity torque; a short angle range may not distinguish gravity from bias; unobserved breakaway cannot identify Fs; poor reversal data cannot separately identify c vs Fc; force baseline uncertainty may dominate small-magnitude forces;
- a parameter can have a numerical fitted value while `identifiable=false`. Preserve it in *diagnostic* JSON only, not production physics fields.

Prefer real independent excitation: accelerations, speed changes, pauses, start/stop, reversal or several pushes with different force profiles. If one demonstration lacks this, emit a scientifically correct partial result. Do not fabricate dynamics by giving it a realistic value.

### 4.4 Model adequacy and prediction

- Distinguish: numerical convergence, parameter identifiability, physics adequacy, out-of-sample prediction, and simulator parity. Make separate PASS/FAIL/NOT_RUN statuses.
- Parameter- and model-selection gates must include *independent* demonstration holdouts or preregistered contiguous blocks, correct initial state, no refitting q or forces on held-out periods, and per-condition residual patterns (velocity sign, q, acceleration, contact transitions).
- Force was already used by the multimodal frontend to refine the original joint/q on some episodes. A training fit's q/force agreement is **not independent evidence**. For real prediction prefer an independently captured second/third demonstration whose reference q is reconstructed using vision-only or otherwise independently of the test force. Record whether shared evidence contaminated the validation target. Do not certify from a heldout subset whose q was fitted using that exact heldout force.
- When forecasting a new episode, use only pre-declared initial `(q0,qdot0)` and measured force/contact. All future q is **evaluation-only**. Compare time-aligned q using RMSE, range-normalized RMSE (flag tiny range), absolute errors, dynamic event timing, rest positions; run both R/P with proper units only when the kinematic model is independently available.
- Perform signed/lag/tare/frame ablations; a nonphysical fit that predicts well because of a wrong sign or lag is not accepted.
- The chosen model may require gravity and/or dry friction; simulated URDF/MJCF/USD must implement those **once**, not twice. If a destination engine cannot represent a term, that engine's parity status is FAIL/NOT_REPRESENTED until a validated extension/plugin is supplied.

## 5. Source data preparation and provenance

### 5.1 Select the correct final joint input

Use ForceArtGS `joint_authority/v000/authority.json` and `joint_states.npz` from the **accepted final** R/P branch or an explicitly versioned accepted final authority. Validate `authority_id`, `joint_states_sha256`, `world_frame`, `q_units`, `reference_time_index`, valid timestamps, selection status, axis/origin in world W and sign paired with q. If the final 3DGS output uses blend frame B, transform geometry/forces with recorded `N_BW` exactly; never feed a blend-frame axis into a base-frame wrench without conversion.

Distinguish **kinematic `IDENTIFIED_*`** from **dynamics identifiable**. A diagnostic R/P hypothesis can be analyzed with explicit `--diagnostic`, but cannot produce a certified production physics asset. A user-approved assumed joint may yield an `ASSUMED_KINEMATICS` preview, never a measured/identified claim.

### 5.2 Calibrated real-force adapter

In ForceArtGS add a focused bridge using `estimotion_2part/joint_discovery/{tcp_contact,wrench_calibration,wrench_physics,contact_modes}.py` (and the existing robot archive/metadata). Produce measured `ForceRecord` via a versioned adapter. Do not use raw force values with assumed `R_EE_S=I`, sign −1, 50-Hz timestamps or a hardcoded TCP offset as though verified.

Gate separately: `sensor_rotation`, `force_sign`, `contact_point/TCP`, `tare`, `tool_gravity_compensation`, `time_sync`, `torque_origin` (if pure couple used), `camera_robot_registration`, and `contact_intervals`. Verify that the source CSV/parquet rows and timestamps match the accepted authority's frame mapping. Save original raw wrench and calibrated world force separately with recorded transformations and hashes. Calibration declarations are provenance, **not proof**. Mark unavailable channels and let diagnostics proceed where safe.

Do not require *zero TCP slip* for dynamics. The current multimodal `sliding`, `stick_soft`, and `stick_hard` modes remain authoritative; known contact geometry and force application points determine generalized effort. If the gripper applies forces through more than one point, the single-point force model is a limited approximation and must be flagged or replaced by a calibrated net wrench.

### 5.3 Geometry handoff

Use validated canonical moving-body surface meshes and the existing PArtTwin `articulated3dgs.handoff/v1`. If no valid **mass-model geometry** exists, still produce `JOINT_SPACE_ONLY` dynamics diagnostic, not a full tensor/URDF. Implement mesh extraction from the already refined multiview RGB-D/segmented surfaces only as an **explicit separate, tested** feature; do not pretend visual PLYs are watertight solid meshes. Keep collision geometry and mass-model geometry distinct, although one source mesh may be used for both if validated.

Store provenance of geometry source, mesh scale, canonical pose, tool exclusion, cavity/thickness assumptions, world registration and handoff hash. Geometry fitting happens **after** accepted segmentation and static/moving Gaussian training; it must not silently alter the upstream kinematic decision.

## 6. Proposed module APIs and on-disk contracts

### 6.1 ForceArtGS-Physics package (new modules; exact paths may adapt to local repository conventions)

```text
src/forceartgs_physics/
  dynamics/
    general_1dof.py          # enabled terms, gravity, hybrid sticking, deterministic rollout
    mass_properties.py       # unit-density geometry + 3x3 tensor, transforms, axis projection
    geometry_inertia.py      # alpha tie, density interpretation, uncertainty, consistency
  estimators/
    general_integral_ls.py   # initializer, weak integral, constraints
    general_constrained.py   # full-model rollout fit, multi-interaction shared parameters
  validation/
    general_identifiability.py
    model_adequacy.py
    heldout_prediction.py
    simulator_parity.py
  adapters/
    forceartgs_g9.py        # versioned joint/force/mass-model contract; no legacy silent conversion
  export/
    dynamics_v2.py          # structured parameter + mass/inertial evidence; NO duplicate URDF writer
  tests/
    test_general_dynamics_*.py
    test_geometry_inertia_*.py
    test_general_identifiability_*.py
    test_simulator_mapping_*.py
```

Illustrative versioned Python API:

```python
from forceartgs_physics.dynamics.general_1dof import GeneralTerms, simulate_general
from forceartgs_physics.dynamics.geometry_inertia import GeometryMassModel
from forceartgs_physics.estimators.general_constrained import fit_general_many
from forceartgs_physics.validation.general_identifiability import diagnose_general

terms = GeneralTerms(inertia=True, viscous=True, coulomb=True,
                     static=True, gravity=True, bias=False)
geometry = GeometryMassModel.from_validated_mesh(
    moving_mesh, mode="uniform_shell", canonical_frame="W",
    source_provenance=mesh_evidence,
)
result = fit_general_many(interactions, terms=terms, mass_model=geometry,
                          gravity_world=verified_g_world, fixed_priors=priors)
report = diagnose_general(result, interactions)
# save results with real validation/identifiability provenance; never auto-certify
```

The user-facing API must not make a frozen `api.identify(...,'AKF')` appear to support gravity or static friction when it does not. Expose separate explicit names and result schemas. If operating in a single ForceArtGS checkout without edit rights on the Physics repository, implement a **thin callable plugin** in ForceArtGS importing the pinned Physics package; do not fork its frozen methods.

### 6.2 ForceArtGS orchestration

Add an opt-in `dynamics` stage and `final_physics_export` stage to the existing controller. Preserve `export`/legacy behavior and never change default scene training automatically. Proposed CLI additions:

```bash
python pipelines/joint_aware_reconstruction.py \
    --scene /verified/scene \
    --stage dynamics \
    --physics-library /verified/ForceArtGS-Physics \
    --dynamics-config configs/dynamics_general.example.yaml \
    --force-calibration /verified/calibration.json \
    --mass-model-spec /verified/mass_model.yaml \
    --dynamics-episodes /verified/episodes_manifest.yaml

python pipelines/joint_aware_reconstruction.py \
    --scene /verified/scene \
    --stage final_physics_export \
    --export-parttwin \
    --dynamics-config configs/dynamics_general.example.yaml
```

Codex must check the actual `argparse`/stage semantics and adapt these examples rather than introduce duplicate command handlers. Support rerun from an existing completed `final/` and G8 visual handoff **without rerunning 3DGS**. Check source signature/authority IDs/hashes before reusing any dynamics fit. If final is diagnostic/ambiguous, the dynamics stage writes diagnostics but **blocks physics-identification claims**.

### 6.3 Suggested configuration schema

Keep a full model as default, easy individual toggles, explicit contact mode inherited from upstream, and explicit evidence paths (NOT invented numerical defaults):

```yaml
schema: forceartgs.dynamics_config/v1
pipeline:
  enabled: true
  only_after_final_articulated_reconstruction: true
  visual_handoff_input: auto_verified
  allow_diagnostic_kinematics: false
  on_missing_calibration: abstain
  on_missing_geometry: joint_space_diagnostic

model:
  kind: general_rigid_1dof
  terms:
    inertia: true
    viscous: true
    coulomb: true
    static: true
    gravity: true
    bias: false
  parameter_source: geometry_coupled   # geometry_coupled | joint_space_only | user_link_inertial
  automatic_term_selection: false      # never silently disable requested effects
  quasistatic: false
  static_friction:
    solver: karnopp_hybrid
    velocity_band_magnitude: null      # documented calibrated numerical tolerance
  gravity:
    source: verified_world_gravity     # must include frame/uncertainty/provenance
    vector_world_mps2: null            # do not guess registration; supply verified vector
    calibration_file: null

mass_model:
  mode: uniform_shell                 # uniform_solid | uniform_shell | segmented_material | user_link_inertial
  mesh: null                          # independently validated canonical moving surface
  geometry_frame: W
  shell_thickness_m: null             # OPTIONAL measured/assumed finite-thickness model
  source_provenance: null
  allow_collision_proxy_as_mass_geometry: false
  reflected_actuator_inertia: null    # only if measured/declared, no silent zero certification

fitting:
  methods: [general_integral_ls, general_rollout_fit]
  compare_frozen_viscous_baselines: [IntegralLS, Backprop, AKF, PINN, MLP]
  use_multiple_interactions: true
  robust_loss: huber
  seed: 0
  measured_force_uncertainty_required: true
  holdout_mode: independent_episodes
  auto_lag_fit: false

contact:
  inherit_upstream_slip_mode: true
  require_contact_provenance: true
  force_on_object_only: true
  use_pure_sensor_torque: false

identifiability:
  propagate_joint_geometry_uncertainty: true
  propagate_calibration_uncertainty: true
  require_excitations: true
  on_unidentifiable_enabled_parameter: block_identified_export

export:
  write_joint_space_result: true
  reconstruct_link_inertia_when_supported: true
  production_requires_heldout_real_prediction: true
  allow_assumption_preview: false
  canonical_builder: parttwin1d
  prefer_imported_single_exporter: true
  write_urdf: true
  write_usd: true
  write_mjcf: true
```

Numerical solver tolerances, confidence thresholds and masses are **not** to be magically filled with plausible values. Codex may establish safe numerical *computational* defaults with unit-tested convergence and clear provenance; measurement uncertainty requires sensor/scene evidence or an explicit assumption flag.

### 6.4 Inputs/outputs

Version a run directory under the **same ForceArtGS scene and branch**, not under an unrelated new reconstruction root:

```text
<forceartgs_run>/
  final/                                 # existing final 3DGS and joint data; read-only
  parttwin_handoff/                      # existing G8 visual handoff; read-only
  dynamics/g9/v000/
    input_manifest.json                  # source repos/revisions, authority IDs and hashes
    calibrated_forces.json               # PREPROCESSED recorded data, measured only
    force_preprocessing_report.json      # frames/sign/clock/tare/contact audit
    mass_model_geometry_manifest.json    # mesh, frames, assumptions, hashes
    unit_density_inertia.json            # m1,c1,Jcom1,Iaxis1 and units
    general_model_config.yaml            # resolved settings, all toggles
    estimator_results/
      general_integral_ls.json
      general_rollout_fit.json
      frozen_viscous_baselines.json
    general_identifiability.json         # parameter-by-parameter ranks, CI and reasons
    model_adequacy.json                  # observed residual and relevant counter-models
    inertia_reconstruction_report.json   # recovered full 3x3, mass, COM, uncertainty or block reason
    heldout_prediction/                  # q predictions and error curves on independent episodes
    physics_parameters.json              # accepted partial/full result, machine-readable provenance
    dynamics_handoff.yaml                # one canonical handoff to builder; NOT fake link properties
    g9_manifest.json                     # status, readiness, hashes, immutable input signature
  parttwin_assets/<scene>/               # final atomically built canonical simulation asset
    visual/static.ply
    visual/moving.ply
    physics/collision/static.obj
    physics/collision/moving.obj
    physics/dynamics.yaml                # resolved full properties + joint_space_identification provenance
    physics/identification/...
    maniskill/model.urdf
    isaac/model.usd
    mujoco/model.xml
    asset.yaml
    object.yaml
    export_manifest.json
```

Preserve legacy `parttwin.dynamics/v1` for backwards compatibility. Add **versioned** `parttwin.dynamics/v2` or a versioned nested extension for the richer physically consistent model. If `dynamics_handoff.yaml` lacks full inertial properties, it is allowed as a joint-space diagnostic but may **not** be passed as the existing full `--asset-physics-spec` schema. Build an explicit translation layer that produces the full required physics input (collision, link inertia, joint stops, joint damping, friction and source registration), with precise missing fields reported.

## 7. Mapping to actual physics runtimes

### 7.1 One canonical model, no separate simulator writers

Retain the validated RigEuler/PArtTwin shared exporter for all formats. Validate `T_B_J0`, `T_link_ply`, axis+q sign and mechanical limits. The source of masses/inertia is **the reconstructed full-body geometry-constrained tensor**, not an arbitrary scalar placed into the URDF. Do not translate `I_eff` into a URDF `armature` when that inertia is already included in the moving link's full tensor: this double-counts inertia. Only separately identified reflected actuator inertia may map to armature, with engine-specific semantics and independent evidence.

The corrected asset should have one consistent moving-body inertial description:

```xml
<link name="moving_link">
  <inertial>
    <origin xyz="COM_IN_LINK_METRES" rpy="0 0 0"/>
    <mass value="IDENTIFIED_OR_CONDITIONALLY_INFERRED_MASS_KG"/>
    <inertia ixx="..." ixy="..." ixz="..." iyy="..." iyz="..." izz="..."/>
  </inertial>
</link>
<joint name="articulation_joint" type="revolute">
  <parent link="static_link"/>
  <child link="moving_link"/>
  <origin xyz="..." rpy="..."/>
  <axis xyz="0 0 1"/>
  <limit lower="PHYSICAL_STOP" upper="PHYSICAL_STOP" effort="..." velocity="..."/>
  <dynamics damping="IDENTIFIED_C" friction="IDENTIFIED_FC_IF_SUPPORTED"/>
</joint>
```

The XML above is a **mapping illustration**, not a template to paste with placeholders; actual exporter emits numerically valid values only. `static_link` is fixed to the simulator world. Some engines accept omitted root inertials; implement a tested support path where permitted rather than inventing static mass/inertia, but retain a valid inertial record if the existing builder requires it and verified input is available. Never break existing runtime compliance to force this enhancement.

### 7.2 Gravity consistency

Gravity is automatically generated by real simulator engines from `m`, `COM`, gravity vector and link pose. **Do not also inject the fitted `G(q)` as a joint torque** in that case. Verify analytic gravity generalized effort equals simulator gravity projected along the joint for both R and P. When the `gravity` term is disabled by the user, the generator cannot simply make the link massless; choose/validate an engine-specific gravity-disable setting or explicitly use gravity-free simulation scene. URDF has no universal per-joint gravity toggle. Record `GRAVITY_CONFIGURATION_MISMATCH` / mark NOT_PARITY if target simulation would apply gravity despite gravity-OFF identification.

For a truly vertical hinge, `G_R(q)` may be approximately zero even with gravity ON; do not interpret this as proof gravity is disabled. For a horizontal drawer, `G_P` should be near zero; an inclined slider may have significant constant gravitational load.

### 7.3 Joint friction and breakaway

- URDF: standard `<dynamics damping=... friction=...>` can express conventional damping and a generic friction parameter; it **does not faithfully specify a detailed static-vs-kinetic friction law or Stribeck curve** in all engines. Require a per-engine extension or mark static-friction parity unsupported.
- MuJoCo: conventional joint `damping` and `frictionloss`; static breakaway requires verified `frictionloss`/solver behavior or a custom validated friction model; do not promise exact force-displacement equivalence to the custom Karnopp integrator without tests.
- Isaac Sim/USD/PhysX: existing exporter maps revolute damping through angular drive and deliberately **does not map Coulomb torque as if dimensionless PhysX jointFriction were Nm**. Add only a validated PhysX-specific solution, otherwise report `NOT_REPRESENTED` for Coulomb/static friction and do not mark physical parity PASS.
- ManiSkill/SAPIEN/URDF: verify actual simulator interpretation of the friction field empirically; do not assume URDF's generic semantics suffice.

If a term is disabled in the requested simulation mode, its physically corresponding simulator coefficient must be disabled too. Unsupported or mismapped terms must not be silently ignored. Record one per-engine term-parity matrix, and explicitly distinguish **portable kinematics**, **nominal dynamics export**, and **verified numerical dynamics parity**.

### 7.4 Physical limits, collision and G9/G8 readiness

- Joint mechanical stops/limits, velocity and effort limits are **not estimated from q(t)'s observed range or from the inertia fit**; require independently supported physical limits or a clearly labelled preview. Do not infer values from previous demonstration extrema.
- Collision uses validated closed link geometry/proxies, not Gaussians. Allow a distinct geometry for mass integration and for collisions. Check link-frame placement, normals, units, support and output readbacks.
- For engines that require complete links, a valid full moving-link inertia, collision geometry and joint limits are mandatory for production, independent of the successful identification of damping.
- Readiness states at least: `KINEMATICS_ACCEPTED`, `DYNAMICS_DIAGNOSTIC`, `PARAMETERS_IDENTIFIED`, `MASS_MODEL_ASSUMED`, `INERTIA_RECONSTRUCTED`, `MODEL_VALIDATED`, `VISUAL_ASSET_READY`, `COLLISION_READY`, `URDF_COMPILES`, `USD_LOADS`, `MJCF_COMPILES`, `ENGINE_PARITY_PASS`, `FULL_PHYSICS_READY`, `BLOCKED_*`. Ensure no boolean named `physics_joint_export_ready` is misread as proof of calibrated physical dynamics.

## 8. Verification, tests, and acceptance gates

Codex MUST implement deterministic tests with seeded noise and true physical values **only in synthetic tests**. No GT joint/physics labels should leak into the real fit. All existing frozen ForceSAPIEN benchmark tests and ForceArtGS joint-aware/multimodal tests continue to pass. Each proposed gate creates an explicit report with `PASS/FAIL/NOT_RUN`, exact command, input hash and software revisions.

### Gate A — Physics equations and switches (at least 13 tests)

A01. R and P generalized-force projection matches analytic Jacobian power `Q qdot = F·v_material` when contact point velocity is consistent.

A02. R and P gravity `G(q)` matches derivative of potential energy by finite difference; correct sign with no applied force.

A03. Vertical hinge gravity torque ≈ 0 under arbitrary q, horizontal hinge nonzero, prismatic gravity constant and sign correct.

A04. Viscous-only switches reproduce existing exact-ZOH trajectory and frozen `IntegralLS`, `AKF`, `Backprop`, `PINN`, `MLP` outputs within historical tolerances; **do not edit frozen code**.

A05. Toggle each of viscous, Coulomb, static, gravity and bias OFF independently: term contributes exactly zero, result carries `disabled_by_user`, fit/prediction/export agree.

A06. Inertia OFF is only permitted in explicit quasistatic mode; dynamic rollout rejects zero mass.

A07. Positive `Fc` resists motion for both velocity signs, no energy creation; c>=0 dissipates.

A08. Static friction: sub-breakaway input sticks at rest; super-breakaway starts moving; release/re-stick and time-step convergence.

A09. No `sign(0)` hack, NaN, negative mass, invalid `Fs < Fc`, or silent static-friction resampling oscillations.

A10. Invalid/missing gravity registration refuses gravity-dependent physical identification; verified base gravity works.

A11. Force sign and axis-q paired sign invariance: correct remapping leaves physical predictions invariant.

A12. Irregular time sampling and force ZOH yield equivalent predictions to resampled high-frequency reference within tolerance.

A13. A gravity-OFF identified model cannot be advertised as parity-tested under a gravity-ON simulator unless the generalized gravitational effect is provably negligible for that joint (with a tested bound).

### Gate B — Geometry and inertia reconstruction (at least 13 tests)

B01. Solid sphere/box/cylinder unit-density integrals recover known mass, COM, full tensor and hinge-axis inertia; rotated/transformed mesh covariance and parallel-axis theorem.

B02. Open thin shell/box-panel surface area integrals give correct thin-shell mass properties; using volume mode on an open shell FAILS.

B03. For a nonzero-offset hinge, scaling alpha recovers known `m`, `COM`, full tensor, and measured `I_axis`.

B04. Translating hinge origin along its axis changes neither I_axis nor gravity torque.

B05. Prismatic mass from `M_eff` and unit-density shape recovers full tensor only under the declared rigid link mass-distribution model.

B06. Nonuniform density / actuator reflected inertia examples show why scalar joint inertia alone cannot determine real link mass; classify preview/assumed, never assert measured.

B07. Invalid mesh (unclosed solid, inverted topology, nonfinite vertices, degenerate axes, zero area/volume), wrong scale and wrong anchor frame fail with readable reason.

B08. Full inertia stays symmetric PD with triangle inequality and frame-conversion invariance; `T_link_ply` and visual PLY hashes unchanged.

B09. Geometry-coupled `alpha[I1 qdd + G1(q)]` matches direct rigid-body dynamics and differs correctly from independent arbitrary gravity-amplitude fit.

B10. With measured independent mass/COM, inconsistent joint scalar produces `INERTIA_GEOMETRY_CONFLICT`, never silent overwrite.

B11. Bootstrap joint-axis/hinge/registration/mesh perturbations widens full-tensor uncertainty appropriately; systematic model assumptions remain explicitly distinct.

B12. Collision-proxy-only geometry is refused by default as mass geometry, opt-in preview labelled and audited.

B13. Fixed-base mass is not fabricated; each simulator's necessary root inertial fields handled as supported and tested.

### Gate C — Estimation and scientific identifiability (at least 15 tests)

C01. Fit synthetic R and P with independent accelerations and known gravity, viscous, Coulomb, static friction: recover parameters within preregistered tolerance; test with noise.

C02. Fit full model and deliberately simpler models: correct model selected on **independent** data when terms distinguishable; no extra-term false identification.

C03. Constant speed → inertia UNIDENTIFIABLE; direction-only Coulomb vs constant bias → confounded; heldout gating blocks export.

C04. Prismatic gravity offset vs force bias confounding is reported even with tiny training residual.

C05. No rest/breakaway data → static threshold UNIDENTIFIABLE even when viscous and Coulomb fit.

C06. Two directions + multiple speeds improve c/Fc separation; explicit sensitivity rank and confidence intervals.

C07. Non-vertical hinge gravity profile identifiable only with sufficient q variation, frame/gravity calibration and excitation; degenerate short range abstains.

C08. Multi-interaction shared-parameter fit with episode-specific q0/v0 outperforms/confirms single-interaction fit; no per-episode *physical* parameters silently fitted.

C09. Fitted result with bad force sign/frame or 20-ms lag fails input/validation gates, not merely produces plausible parameters.

C10. Force frame and contact point uncertainty propagate into Q_R and physical parameter intervals; prismatic Q projection unaffected by hinge gauge.

C11. Unknown contact intervals are not turned into zero-force intervals and are excluded or treated with explicitly calibrated uncertainty.

C12. No measured wrench calibration → diagnostic can run only on trusted independent inputs, physical export blocked. Never synthesize forces for real scenes.

C13. Heldout prediction uses frozen parameters, force and user-declared initial state, not test q/GT.

C14. q(t) refined with force in frontend does not count as an independent force-based validation target; detect and report leakage.

C15. Existing Physics `api.identify` and ForceSAPIEN A3/B1–B10 retained byte-identical or within existing declared tolerances, and all original package tests pass.

### Gate D — Full export/engine parity (at least 12 tests)

D01. Real asset can be produced from valid synthetic complete physics without modifying standard Gaussians; static/moving PLY SHA-256 matches G8 visual handoff.

D02. Dynamics YAML has schema version, enable flags, joint-space scalar, reconstructed full tensor, density prior/provenance, CI, link-frame transform, and actual mechanical limit provenance.

D03. URDF inertial origin is COM (link frame); tensor is COM tensor, not hinge-origin tensor; projected hinge inertia matches synthetic truth.

D04. No duplicate gravity torque if simulator computes gravity; no duplicated inertia in armature; gravity OFF is explicitly mapped or parity blocked.

D05. MuJoCo parses and uses joint viscous damping, dry-friction semantics recorded, reproduces analytic damped rollout under supported terms.

D06. SAPIEN/ManiSkill URDF loads, `fix_root_link`, prescribed joint axis and link inertials; kinematics match Gaussians for q0, qmid, qend.

D07. Isaac USD loads when runtime available; degrees/radians conversion (limits, velocity, angular damping) correct, inertia principal axes preserved; unsupported friction noted.

D08. Static breakaway parity tested; unsupported engines report NOT_REPRESENTED, never all-PASS.

D09. Withheld external forces produce comparable time-aligned q trajectory in standalone model and each validated simulator; report RMSE and event timing.

D10. Missing mass-model geometry, force verification, physical stops, collision meshes, or runtime returns differentiated `BLOCKED_*`/`NOT_RUN`, not a fabricated complete asset.

D11. Replaying the accepted 3DGS visual poses uses live physics `T_W_L @ T_link_ply` and **does not reapply q** to Gaussians twice.

D12. Real-output source hashes, authority IDs, source commit/submodule revisions, dynamics model ID and export version survive into portable asset manifest; tampering invalidates reuse.

### Gate E — Actual scenes and ablations

1. Fridge (currently one physical R result) — verify final authority and calibrated forces, gravity axis alignment, hinge uncertainty; test both `gravity=true` and `false` with explicit parity claims. If no verified wrench/mesh/force excitation exists, output a complete *diagnostic/blocker report* rather than invented parameters.
2. Drawer — physical P may remain TYPE_UNOBSERVABLE (or missing robot archive) in some runs; do **not** bypass this to obtain a mass estimate. With truly identified drawer and verified force, test prismatic gravity vs bias confounding.
3. Black box — currently camera disagreement; an ambiguous hinge is not sufficient for certified dynamic identification. Keep diagnostics separated from production assets.
4. Dog and suitcase — honor tool/support contamination gates; no automatic dynamics solely because a Gaussian asset exists.
5. Synthetic ForceSAPIEN — run the original A3→B1–B10 benchmark **unchanged**, plus new full-model tests including gravity, reversals, breakaway and calibrated contact.
6. Real experiment design: 1+10 demonstrations/object when available. Fit one identification episode (or declared train group), predict on ten new force episodes; q from those episodes is only evaluation. Report per-episode, per-object and overall distributions, 95% intervals, failure categories, and training-force leakage audit.
7. Ablate each term (`no_gravity`, `no_coulomb`, `no_static`, `viscous_only`) and input uncertainty (nominal vs frame/lag/axis sampled); never silently select one model on the test set.

All comparisons require the same observed q, calibrated forces, coordinate frames, contact validity, initial state and evaluation timestamps. Never assert the real benchmark has passed merely because a synthetic unit test passed.

## 9. Codex implementation sequence and stop rules

**Implement sequentially, producing actual code and tests at each gate:**

1. `AUDIT` — local repo and external submodule/builder/rights review, benchmark baseline test, `docs/G9_DYNAMICS_REPO_AUDIT.md`.
2. `MODEL` — general 1-DoF ODE with on/off terms, exact gravity signs and static-friction hybrid solver. Gate A.
3. `INERTIA` — validated mesh/shell unit-density mass properties, alpha coupling to hinge/prismatic inertia and gravity, frame and uncertainty propagation. Gate B.
4. `ADAPTERS` — accepted ForceArtGS authority, calibrated measured force data, robot contact integration, real clock, missing-input fail-closed gates. Selected C tests.
5. `FITTER` — constrained integral initializer, rollout fit, parameter diagnostics, multiple demonstrations and heldout predictive validation. Gate C.
6. `PIPELINE` — add opt-in G9 stage after art/final and before final physics export; immutable signatures and reruns without 3DGS. Compatibility and smoke tests.
7. `EXPORTER` — versioned G9 physics handoff; resolve geometry-derived full link properties and dynamics through the shared PArtTwin builder, not bespoke format generators. Gate D.
8. `PILOTS` — real scenes; report exactly why identification succeeds/abstains. Gate E.
9. `REPORT` — `docs/G9_GENERAL_DYNAMICS_IMPLEMENTATION_REPORT.md`: source changes; exact invocation; result tables; per-term enabled/identified status; tests; calibration evidence; full simulator readbacks; red flags; outstanding work.

**No committing or pushing unless user authorized in the current workflow.** Preserve existing worktrees, branches, running jobs and generated result directories. Do not run a new long GPU model training by default: initial dynamics fitting and synthetic tests can run independently of 3DGS. Real multi-scene analysis may be split into explicitly logged opt-in runs, with partial results available immediately.

**Mandatory stop rules:**

- Unknown force sign, frame, tare, clock, contact provenance or q/force calibration → `FORCE_CALIBRATION_UNVERIFIED`; no physical identification, no fake numeric fallback.
- Kinematic `TYPE_UNOBSERVABLE` / `MODEL_INCONSISTENT` / uncertified assumed branch → diagnostic only; no automatic production physics.
- Force/trajectory insufficient for one or more active coefficients → identify supported subset only if user explicitly selected a simpler model; otherwise full-model `UNIDENTIFIABLE`.
- Wrong/nonrepresentative mass geometry or missing validated mass model → joint-space diagnostic only, full tensor null.
- Incomplete static/moving collision geometry or physical limits → URDF/USD/MJCF production blocked or explicitly user-approved preview, not fabricated full physics.
- Gravity disabled but runtime would apply nonzero joint gravity → parity FAIL until explicit engine handling or provable zero generalized gravity.
- Unsupported Coulomb/static friction in an engine → report per-term NOT_REPRESENTED; no false three-simulator dynamics parity claim.
- Fit succeeds but fails independent prediction or demonstrates model inconsistency → retain research diagnostic; production acceptance fails.

## 10. Definition of done

This work is **done** only when:

1. The user's chosen general dynamics configuration truly changes the **fitted equations, predictions, stored parameters and per-engine simulation semantics**; turning off any term produces a reproducible simplified variant.
2. The validated original four estimators and benchmark remain intact and available as viscous-only baselines.
3. A full inertial matrix/mass/COM is **computed from validated geometry + a transparent density/mass assumption + identified effective joint inertia/mass**, with a precise projection check and appropriately qualified provenance; it is never simply inferred from one scalar without geometry.
4. ForceArtGS G9 consumes the **final accepted metric joint, q(t), calibrated TCP/force and contact evidence** and cannot silently fit on legacy normalised coordinates or corrupt measurements.
5. Real estimated parameters are tied to identifiability, model adequacy and independent-force prediction, not just fitting accuracy; unsuccessful cases abstain rather than fabricating physically plausible values.
6. G9 hands its physical quantities to the **single shared** PArtTwin canonical exporter. Validated URDF/USD/MJCF agree on link mass/COM/full inertia, joint geometry, damping, gravity and **documented simulator-specific** friction behavior. The two standard 3DGS PLYs remain unchanged.
7. An accepted synthetic fully instrumented R and P asset exports and runs in the available engines; all missing real-world evidence appears as explicit blockers; full results/commands/tests documented.
8. A full real experimental target remains 1 identification demonstration + 10 *independent* validation demonstrations/object, with the exact status of available datasets and force calibration honestly reported.

---

### Required final Codex response

Return: summary of actual implementation vs this directive; exact Git commits/dirty state; changed paths; executed test commands and results; synthetic parameter recovery errors and simulator parity table; real-scene statuses and blockers; output directory paths; precise commands to rerun on the user's workstation; and outstanding limitations. Do not call the system a fully calibrated physical twin when only a joint-space scalar or a geometry-conditioned guess exists.
