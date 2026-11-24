# Codex implementation directive — G9 physically consistent joint-projected gravity, mass geometry and PINN/EKF audit

**Date:** 2026-11-20  
**Repository:** `~/ForceArtGS`  
**Scope:** Correct and validate the existing uncommitted GeneralPINN, GeneralAugmentedEKF and G9 geometry-coupled dynamics. **Do not rewrite working estimators, change the vision-derived joint arbitrarily, or rerun 3DGS training.**  
**Source of audit:** User-provided Codex implementation transcript `Pasted text(20261120-164640).txt`, notably the implementation of `torch_model.py`, `general_pinn.py`, `general_ekf.py`, `geometry_proxy.py`, the real four-method reports, and the final 2026-11-20 summary. Inspect the **actual current working tree first**; the public repo revision is not proof of the later uncommitted implementation. Respect existing edits and running jobs.

## 0. The precise issue and the required result

The gravity equation in G9 is *nominally already correct*: `G(q) = dU/dq`, with `U(q)=-m g·COM(q)`, and with `g_B=[0,0,-9.81] m/s²` because the operator confirms `base_link` z is the real vertical. **Do not introduce another independently fitted gravity force, a second mass, or a second sign reversal.**

What remains unverified in real data is whether all terms are computed with a consistent reference configuration, frame and physical mass geometry, and whether the inferred **nonzero gravitational torque from a nearly vertical fridge hinge** is robust to joint and COM uncertainty. The current geometry-coupled real runs fit poorly; this must be diagnosed, not hidden by adding more free coefficients.

The required deliverables are:

1. A **single checked geometric gravity implementation** used by `ModelSpec`, Torch GeneralPINN, GeneralAugmentedEKF, the integral initializer, and rollout, all with identical units, signs, canonical pose and enabled-term switches.
2. A **unit-density mass/COM/inertia audit** for each geometric source; an analytic perpendicular COM vector and hinged gravity torque, with human-readable numerical per-scene diagnostics.
3. **Finite-difference and analytic verification** of potential-energy derivatives, gravity phase/sign, mass coupling, q-reference offsets, coordinate transformations and reflected inertia.
4. A **fridge axis/COM/geometry counterfactual study** distinguishing a truly tilted hinge from error in the estimated axis; zero-gravity vertical hinge is a *diagnostic alternative*, never a silent replacement.
5. Diagnostics of the **force-impulse vs velocity-change sign mismatch** before making physical-inertia claims; never “fix” the result by silently flipping forces, joint coordinates, or gravity.
6. A fixed-parameter open-loop comparison of **PINN, EKF and rollout** on exactly the same inputs, reporting implied real masses and normalized residuals. No estimator may claim physical identification merely because its own neural trajectory is smooth or the training loss is small.
7. Tests, reports and a reproducible command to run longer real-data experiments **without automatically starting/monitoring them**.

## 1. Existing code to inspect before editing

| Area | Expected current path | Inspect specifically |
|---|---|---|
| Canonical rigid dynamics | `forceartgs_g9/dynamics/general_1dof.py` | `RigidBodyGravity`, rotation conventions, `G=dU/dq`, `simulate_general` |
| Unit-density geometry | `forceartgs_g9/dynamics/mass_properties.py`, `geometry_inertia.py` | solid/shell integrals, inertia about the joint, mesh frame, reflected inertia |
| Tentative real geometry | `forceartgs_g9/dynamics/geometry_proxy.py` | `T_Wc_P`, moving PLY, q anchor, convex hull, opacity filtering, hull volume/area |
| Shared parameterization | `forceartgs_g9/estimators/model_spec.py` | `geometry_coupled`, `joint_space_gravity_coupled`, disabled terms, origin and q-reference |
| Differentiable model | `forceartgs_g9/estimators/torch_model.py` | `TorchDynamics.terms`, `residual`, autograd to density |
| PINN | `forceartgs_g9/estimators/general_pinn.py` | all seven losses, q and qdot scaling, force integration, output validation |
| EKF | `forceartgs_g9/estimators/general_ekf.py` | nonlinear transition/Jacobian, parameter posterior, measurement and force covariance |
| Rollout | `forceartgs_g9/estimators/general_constrained.py` | same scalar `M` and `G(q)` as other estimators |
| Integral initializer | `forceartgs_g9/estimators/general_integral_ls.py` | correct gravity basis and q reference |
| Data adapter | `estimotion_2part/joint_discovery/robot_record.py`, `forceartgs_g9/adapters/forceartgs_g9.py` | force-on-object sign, projected effort, accepted axis/hinge, pose offset and timestamps |
| Orchestration | `forceartgs_g9/core.py`, `stage.py`, `config.py` | provenance and method parity, physical vs exploratory labeling |
| Comparisons | `tools/g9_method_comparison_summary.py`, `pipelines/g9_real_general_methods.sh` | independent normalized replay, source version, commands |
| Documentation | `docs/G9_GENERAL_PINN_AKF_IMPLEMENTATION_REPORT.md`, `docs/G9_CORRECTIONS_AUDIT.md` | update with numerical audit and limitations |

**Preservation:** `ForceArtGS-Physics` frozen PINN/AKF baselines and validation must remain untouched; do not overwrite validated 3DGS PLYs, old G9 versions, or change the joint frontend's accepted authority. Respect the running `joint_aware_all_real.sh`/3DGS tasks.

## 2. Coordinate and sign contract (NON-NEGOTIABLE)

Define `B = base_link` in the original RGB-D/calibrated robot common frame; `g_B=(0,0,-9.81)` m/s² (operator-declared true vertical). Keep a metadata provenance field `DECLARED_OPERATOR_VERTICAL`; do **not** mark it as measured calibration without evidence. `a_B` is a **unit** accepted joint axis, `o_B` is any point on its hinge line, and `q(t)` has the same positive direction and zero/gauge as the stored accepted authority.

The real sensor records the **reaction on the robot**. Convert **once**:

\[
F_{object}^{B}(t)=-R_{B,EE}(t)R_{EE,S}F_{sensor,\,tare\ corrected}^{S}(t).
\]

The `object_side_force_multiplier` must be explicitly present and exactly `-1` for these real recordings. **Fail if absent, NaN or unsupported; do not default to `-1`.** This correction must not accidentally negate force twice in the multimodal frontend or G9.

Generalized input:

\[
Q_P(t)=a_B^TF_{object}^B(t),\qquad
Q_R(t)=a_B^T[(p_{TCP}^B(t)-o_B)\times F_{object}^B(t)].
\]

Sensor intrinsic moment is **not** added unless measured torque-origin / wrench transport are correctly established; avoid double-counting `p × F`. Camera-robot registration and shaft rotation must be checked on the *same* frame; all `T_X_Y` transformations are explicit. No implicit `world=base` unless the input metadata establishes it.

For physical dynamics use `M qdd + c qdot + Fc sign(qdot) + G(q) + b = Q(t)` and **`G(q)=dU/dq`**, not the applied gravity torque. Thus physical applied gravity effort is `-G(q)`. Do not change that sign to make a trajectory fit.

## 3. Canonical mass geometry and the q-reference offset

### 3.1 Mesh/3DGS reference pose is NOT automatically `q=0`

Audit `geometry_proxy.load_moving_points`: it transforms moving Gaussian means by `parts.moving.T_Wc_P` and records `canonical_anchor_original_frame`. The code/documentation currently **assume those points represent `q=0`**. Check this assumption against the actual `JointAuthority` reference frame, stored q trajectory, moving-part PLY coordinate convention, and handoff transform. If the mass model is anchored at `q_ref != 0`, do NOT use `R_a(q)` on its COM directly.

For a fixed revolute axis in B, with reference COM `c_ref` at stored coordinate `q_ref`:

\[
 c_B(q)=o_B+R_{a_B}(q-q_{ref})(c_{ref,B}-o_B).
\]

For prismatic:

\[
 c_B(q)=c_{ref,B}+(q-q_{ref})a_B.
\]

This same offset must be used when the unit-density gravity basis `A1 cos(q)+B1 sin(q)` is built, either by converting the COM to the `q=0` canonical pose or by storing and explicitly applying `q_ref`. The x/y/z axes and the positive q convention **must not change** between `TorchDynamics`, `ModelSpec`, EKF and rollout.

If the reference is truly `q_ref=0`, prove it with an assertion/check derived from the handoff and authority, and record the evidence. Do not hardcode that assumption.

### 3.2 Unit density and analytic perpendicular COM vector

With unit-volume-density solid or unit-areal-density shell **mass geometry**:

\[
 m_1=\int dV \quad\text{or}\quad \int dA,\quad
 c_1={1\over m_1}\int x\,dV \quad\text{or}\quad {1\over m_1}\int x\,dA,
\]

\[
 J_{C,1}=\int (\|x-c_1\|^2 I_3-(x-c_1)(x-c_1)^T)\,dV\quad(\text{or }dA).
\]

Use SI coordinates in meters and a symmetric PSD 3×3 tensor. The `m_1` quantity has **m³** for a solid and **m²** for a shell; the density has respective **kg/m³** or **kg/m²** units. Do not mix density results across modes as though they were the same number.

For a revolute axis:

\[
 r_{ref}=c_{ref,B}-o_B,\qquad
 r_\perp=(I_3-a_Ba_B^T)r_{ref},\qquad
 \ell=\|r_\perp\|.
\]

This is the vector the user requested: **analytic COM displacement perpendicular to the hinge**, calculated from the assumed uniform-density part geometry, not the arithmetic average of Gaussian centers. It must be invariant if the hinge origin moves along the axis (`o -> o + λa`). Handle `\ell≈0` without normalizing by zero. The COM is a property of the moving rigid body and should not vary with density under the uniform-density assumption.

### 3.3 Same density produces inertia and gravitational mass

Let `rho` represent volumetric density (solid) or areal density (shell), unambiguously named and unit-labeled in outputs:

\[
 m=\rho m_1,\quad J_C=\rho J_{C,1},\quad
 I_{axis,1}=a_B^T J_{C,1}a_B + m_1\ell^2.
\]

For revolute:

\[
 \boxed{M_R(\rho)=\rho I_{axis,1} + I_{reflected}}
\]

\[
 \boxed{G_R(q;\rho)=-\rho m_1 g_B^T[a_B\times R_{a_B}(q-q_{ref})r_\perp].}
\]

For prismatic single moving link:

\[
 \boxed{M_P(\rho)=\rho m_1+M_{reflected},\quad G_P(\rho)=-\rho m_1\,g_B^Ta_B.}
\]

A `reflected` actuator inertia/mass, if present and independently identified or declared, enters the kinetic inertia only; **it is not gravitational mass**. Do not use `M_P` to compute `G_P` when `M_reflected` is not zero. Document this explicitly; this was a risk in older joint-space shortcuts.

For a revolute hinge whose axis exactly equals vertical, `a=[0,0,±1]`, `G_R(q)=0` for **any** COM offset and `q`. For a tilted hinge, the projected term is nonzero; gravity amplitude must satisfy

\[
 \max_q |G_R(q)|=m\,\|g_B\times a_B\|\,\ell
             =m\,g\,\sin(\beta)\,\ell
\]

with `beta=acos(|a_B · zhat|)`. In particular, do not project `g` onto `a` for a revolute hinge as though it were a prismatic joint; the appropriate quantity is the **moment arm cross product**.

### 3.4 Potential energy and force/torque sign tests

\[
 U_R(q)=-m\,g_B^T c_B(q),\qquad
 G_R(q)=\frac{dU_R(q)}{dq}.
\]

Numerically verify `G_R(q)` against centered finite differences of `U` at random q, with randomized non-vertical axes and eccentric COMs. Also verify work-energy: in an unforced frictionless trajectory, `0.5*M*qdot² + U(q)` should remain constant up to numerical integration tolerance. In the general forced case, `Δ(K+U) = ∫Q qdot dt − ∫(c qdot² + Fc |qdot|)dt` up to integration error (excluding external hidden contacts). Distinguish `G=dU/dq` from the applied generalized gravity torque `-G` in plot legends and reports.

Implement the same finite-difference verification for prismatic `U(q)`, `G_P`, gravity on/off, and torque sign under axis reversal `a'=-a`, `q'=-q`. In that transformation, physical trajectories and energies must be invariant; signed generalized inputs and gravity load reverse as expected.

## 4. Real geometry validity: do not confuse a watertight convex hull with the object's physical mass

The current tentative proxy is built from opacity-filtered Gaussian means + 8-NN outlier filtering, then a **convex hull**; `solid_hull` fills it, `shell_hull` puts mass on its outer surface. This is an exploratory **geometry scenario**, not a validated physical geometry. For fridge doors and suitcases, non-convex structure, voids, and the difference between visual surface coverage and mass support can dominate the first mass moment and inertia-to-gravity ratio.

Mandatory diagnostics for each proxy:

- inputs: G8 handoff SHA, Gaussian PLY SHA, `T_Wc_P`, `q_ref`, mask/part source, number of retained means, filtering thresholds;
- hull: bounds, dimension, volume (`m³`) or area (`m²`), watertightness, winding, valid facet orientation, number of connected components, solid-vs-shell differences;
- unit density: `m1`, `c_ref`, `J_C,1` 3×3, axis point and direction, `r_perp` vector and magnitude, `I_axis,1`, `G1(q)` coefficients (including phase), gravity-amplitude-to-inertia ratio;
- estimated density and **implied moving mass** `m=rho*m1` (kg), full inertia `rho*J_C,1` (kg m²), `I_axis` (kg m²), and implied gravitational torque (N m) at observed q;
- explicit `TENTATIVE_GEOMETRIC_PREVIEW` and `PHYSICAL_MASS_UNVERIFIED` flags for all such outputs; no certification or full physics export.

**Concrete observed discrepancy to explain:** The latest real transcript gives for the red fridge `solid_hull: m1≈0.0609 m³, I_axis,1≈0.00351`, PINN fitted `rho≈1.44 kg/m³` -> implied mass ≈**0.088 kg**; `shell_hull: m1≈0.941 m², I_axis,1≈0.0691`, PINN fitted areal density ≈0.0878 kg/m² -> implied mass ≈**0.083 kg**. These should be printed prominently. They are not reliable physical masses just because the network/seed ensemble agrees. Do not impose an arbitrary mass prior as truth; report geometry and estimation plausibility and compare against any available independently measured mass only if it exists.

Compute both solid/shell proxies as alternatives and quantify the changes in `c1`, `r_perp`, `I_axis,1`, `rho`, `m`, gravity amplitude, and `M/G` relationship. If a better independent depth/geometry model exists, use it as a **separate** scenario without altering G8 PLYs. Do **not** pretend the Gaussian covariance encodes physical mass density.

## 5. Joint geometry uncertainty: fridge hinge tilt is decisive

The accepted fridge axis in the supplied Codex transcript is approximately `a_B=[-0.082,-0.065,0.994]`, `beta≈6.03°` from the vertical. The observed 6° value must **not** be silently rounded to 0° and must **not** be presumed physically exact. This angle controls the size of joint-projected gravity (`sin(beta)`).

Run a **read-only counterfactual geometry audit**, holding the original observed force and q data fixed, with labeled scenarios:

1. **Accepted original axis/hinge** with its stored uncertainty and mass-geometry proxy.
2. **Axis perturbations within the estimated covariance**, including full 3D perturbations, not only a ±1σ tilt in a single direction; vary hinge-line origin perpendicular to axis according to its uncertainty.
3. **Exactly vertical hinge** (`a_B=±zhat`) as an explicitly alternative kinematic hypothesis; compute `G_R=0` and independently evaluate the RGB-D reprojection/track/contact residual for this hypothesis **in the upstream kinematic model**. It is **NOT** permissible to call the vertical alternative a better hinge solely because G9 replay improves.
4. **COM/mass geometry perturbations** compatible with the observed shell/solid alternatives and registration uncertainty; do not substitute a free mass-moment coefficient in `geometry_coupled` mode.

For each scenario record `|g×a|`, `r_perp`, `G(q)` range on the actual q trajectory, `max_q |G|`, `I_axis,1`, predicted torque from the same fitted density, and **open-loop replay**. Separate geometry-induced variability from statistical parameter uncertainty. Axis and joint type remain upstream decisions. The local `joint_space_gravity_coupled` independent `(M,mu)` fit is only an **ablation** (no true shared mass distribution), never an exportable physics model.

A critical rule: if changing the axis modifies generalized robot effort, **reproject the same measured 3D force into the new axis** and document both the effort and gravitational-load changes; if trying to isolate only the gravity-law sensitivity, explicitly label that as `GRAVITY_ONLY_COUNTERFACTUAL` rather than a physically consistent joint change.

## 6. Shared physics across PINN, EKF, integral LS and rollout

- `TorchDynamics` must **derive**, not reinvent, the same `m1`, `J_C,1`, `I_axis,1`, `G1(q)` and q-reference from `ModelSpec/GeometryMassModel`. The torque and all derivatives w.r.t. density must match the reference scalar implementation for every enabled term combination.
- PINN uses `qhat_phi(t)` with autograd first/second time derivatives. Its strong-form residual on moving intervals remains:

  \[
  r=M(\rho)\ddot{\hat q}_\phi+c\dot{\hat q}_\phi+F_c\operatorname{sign}(\dot{\hat q}_\phi)+G(\hat q_\phi;\rho)+b-Q(t).
  \]

  Preserve weak/impulse/static-friction constraints; do not substitute a neural prediction of gravity for the analytic projection. `rho` is shared by inertia and gravity with gradients from both. Report `d r / d rho = I_axis,1*qdd + G1(q)` (for revolute; with reflected inertia fixed) and verify by PyTorch autograd vs finite differences. For prismatic `d r / d rho = m1*qdd − m1*g·a`.
- EKF uses the **same** `M(rho)` and `G(q;rho)` in the continuous acceleration and in its discrete Jacobian; the derivative of gravity w.r.t. `q`, density, and hinge is accounted for. Sticking/sliding transitions and uncertain force input remain handled explicitly. Use Joseph covariance; non-PSD and inconsistency must produce diagnostics, not false confidence.
- General rollout and integral LS must use the **same** `q_ref`, gravity basis, effort timeline and mass geometry. Fit each method from the same integral-LS initial parameters; compare **frozen-theta forward replay**, not neural trajectory reconstruction error.
- For the R hinge in the local joint-space diagnostic, if independent `M` and `mu` persist, flag `GRAVITY_MASS_NOT_COUPLED` in outputs. Checking `M>=mu*support_centroid_lever` is **not** proof of a full physical inertial model, since support centroid is not verified COM. A report of `physical_gravity=True` must not be interpreted as fully reconstructed mass.
- Zero-capable physical parameters need a truly zero-capable representation for damping/friction; `exp(log c)` can only approach 0 and can exhibit false narrow seed spread at a numerical floor. Do not mark such at-bound coefficients identifiable; report floor/hessian or profile checks.

## 7. Fridge impulse audit: do not misdiagnose a sign mismatch as gravity

The report gives fridge source rows **229–232**, `J_Q≈+0.00643±0.000659 N·m·s` while reported `Δqdot≈−0.234±0.096 rad/s`. These oppose each other. Before attributing any failed PINN/EKF fit to insufficient excitation or to gravity, audit:

1. Whether `q` increases in the same physical sense as `a` and the joint transform. Explicitly check axis sign and q sign as a *paired* gauge transformation, without secretly flipping one.
2. The force sensor convention (`-1` exactly once), `T_B_EE`, `R_EE_S`, `p_TCP`, hinge origin, `Q=a·((p-o)×F)` and force integration over the native rows.
3. Whether the velocity estimates bracketing a **four-sample impulse** are reliable in time, whether the 50 Hz force vs RGB-D observation times are offset by a material fraction of the pulse, and whether the first/last samples represent sticking/sliding transitions.
4. Whether `∫(c qdot + Fc sign + G + b)dt` and any initial contact/reaction can reasonably account for the sign discrepancy; do not assert `M=J_Q/Δqdot` when the equation includes other torques.
5. The corrected **impulse balance** `M Δqdot + ∫(c qdot + Fc sign + G + b)dt = J_Q`, with uncertainty intervals and geometry scenarios, not just a scalar ratio.

If the observed data remain inconsistent, report `IMPULSE_DYNAMIC_MODEL_CONTRADICTED`/`SIGN_OR_TIMING_INCONCLUSIVE` with evidence. **Do not change sign conventions merely to get positive inertia.**

## 8. Numerical tests and mandatory acceptance gates

Write tests in `tests/test_g9_gravity_geometry_contract.py`, `tests/test_g9_gravity_pinn_ekf_parity.py`, and any existing tests whose assumptions change.

**A. Geometry and reference (>=12 tests):** axis normalization; origin shifted along axis invariance; COM on axis; COM offset perpendicular to axis; full asymmetric tensor & parallel-axis theorem; watertight mesh winding; solid and areal shell unit invariants; `T_Wc_P` transform; `q_ref != 0` test; q gauge (`a,q -> -a,-q`); mesh rotation with transformed axis/origin/gravity invariance; hull geometry flagged tentative; malformed transform and missing q-reference fail closed.

**B. Gravity physics (>=12 tests):** `G=dU/dq` at random q; vertical R axis -> zero gravity; tilted R axis analytic `m g sin(beta) ell` amplitude; R torque sign in ±q; P gravity projection and horizontal/vertical sliders; inertia and gravity both linear in same density; full 3×3 COM tensor transforms; reflected inertia affects only kinetic term; gravity on/off switch exact; arbitrary COM along axis does not change R gravity; finite-difference density derivative and q derivative.

**C. All-method parity (>=10 tests):** `ModelSpec`, `TorchDynamics`, EKF and rollout produce same `M`, `G` and derivatives for random valid states/parameters; PINN physics loss gradient w.r.t. density equals inertia-plus-gravity derivative; weak/pulse integrals agree with a direct reference quadrature; no numerical force-sign double negation; same force input timeline; unknown torque origin excluded; zero friction/damping at-bound behavior; old frozen baselines unchanged; fixed-theta open-loop replay of known synthetic R/P trajectories.

**D. Real diagnostics (>=6 bounded tests):** fridge provenance; reported q-reference; implied solid/shell mass/unit audit; 6.03° original vs exactly vertical labeled alternative; axis perturbation changes both effort and gravity; impulse sign discrepancy preserved unless explained by a measured input correction; no real `PHYSICAL_IDENTIFIED` label from tentative geometry and no held-out episode.

No success criterion should depend on matching the old real fitted density. Synthetic PASS means analytic and numerical correctness; real scientific validation requires model adequacy, identifiable parameters and independent prediction. Print actual test counts; do not claim blanket success if a suite was not rerun after final edits.

## 9. Deliverables, report and how to run

Generate new versioned, additive outputs under each selected run's `dynamics/g9/vNNN/`:

```
gravity_geometry_audit.json
mass_com_inertia_audit.json
reference_pose_alignment.json
gravity_joint_sensitivity.json
method_gravity_parity.json
impulse_balance_audit.json
```

The reports must have explicit schema/version, source SHA-256 hashes, unit labels, axis/origin in B, COM and `r_perp` in B at `q_ref`, `q_ref`, gravity source/provenance, solid/shell choice, effective inertia, reflected inertia, force sign and known uncertainties, plus scenario-wise replay metrics.

Update `docs/G9_GENERAL_PINN_AKF_IMPLEMENTATION_REPORT.md` with a dedicated section **“Gravity geometry projection and reference-pose audit”** and `docs/G9_CORRECTIONS_AUDIT.md` with quantitative fridge/suitcase tests. Include a short comparison table per geometric scenario for PINN, EKF, rollout and integral initializer. If an audit cannot be run, mark `NOT_RUN` rather than invent a number.

**Commands for Codex to run immediately (only if short/safe):**

```bash
cd ~/ForceArtGS
~/miniconda3/envs/freeartgs/bin/python -m unittest \
  tests.test_g9_gravity_geometry_contract tests.test_g9_gravity_pinn_ekf_parity
```

For long real-data/G9-PINN batches, **write/update the script and print the exact command; do not start, monitor or wait on it**. Reuse the existing batch root and convention:

```bash
cd ~/ForceArtGS
nohup bash pipelines/g9_real_general_methods.sh \
  > "/media/samuelemara/Extreme SSD/Samuele/results_20_11_26/g9_gravity_geometry_audit.log" 2>&1 &

~/miniconda3/envs/freeartgs/bin/python tools/g9_method_comparison_summary.py \
  "/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware"
```

Only use the batch above if it actually invokes the newly implemented gravity/pose audit; otherwise add a dedicated `pipelines/g9_gravity_geometry_audit.sh` and print its command. Do not overwrite previous `vNNN` outputs. Do not commit/push unless the user explicitly requests it.

## 10. STOP / DONE criteria

Do not declare this task finished until:

- gravitational mass and inertial mass are **one density** in `geometry_coupled` across all methods (with reflected inertia handled separately);
- the perpendicular COM vector is computed analytically from a consistently transformed reference mass geometry;
- nonzero `q_ref` and `T_Wc_P` are explicitly verified, not assumed away;
- projected gravity signs and magnitudes agree with `dU/dq` and with analytic tilted-hinge tests;
- real fridge gravity and general force effort are tested under axis/COM/geometry scenarios, and a vertical-axis alternative is not silently substituted;
- the real convex-hull scenarios report implied moving mass, physical units, and clearly marked uncertainty/assumption status;
- the fridge impulse mismatch has a detailed, quantified audit (or an explicit unresolved reason), not an arbitrary sign flip;
- PINN, EKF, and rollout use one physical contract with common-input open-loop metrics;
- tests and updated documentation are available and precise;
- no upstream PLYs, old results, or active jobs are disturbed; and no new long jobs are launched on behalf of the user.
