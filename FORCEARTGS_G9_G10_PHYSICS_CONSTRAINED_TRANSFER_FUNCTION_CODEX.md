# ForceArtGS — Physics-Constrained Input–Output System Identification and PhysX Refinement

**Codex implementation specification — 2026-11-23**  
**Target repository:** `~/ForceArtGS` only.  
**Pipeline position:** extend existing G9 analytical/integral/PINN/EKF estimators and G10 PhysX refinement on the *estimator side*.  
**Downstream:** PArtTwin1D receives a versioned, validated dynamics handoff; **do not implement estimation, optimization, or new asset-authoring logic in PArtTwin1D**.  
**Scope:** one-DOF revolute and prismatic articulated parts, visually supervised joint displacement, robot contact force/torque when supported, free coasting when contact is genuinely absent, and matching a specific SAPIEN/PhysX or Isaac/PhysX runtime.

## 0. Executive directive

Implement a **joint-specific, physically interpretable input–output identification layer** before the existing G10 PhysX parameter refinement. The identification must reconcile four descriptions of the *same* experiment:

1. **Measured input** — calibrated generalized contact effort `u(t)` from the existing ForceArtGS canonical robot record. A near-zero wrench *during known contact* is **not** evidence that the object is force-free.
2. **Measured output** — the accepted / appropriately labeled working joint-coordinate trajectory `q(t)`, uncertainty, and timestamps from the final 3DGS joint authority.
3. **Expected joint physics** — a minimal passive 1-DOF differential equation with physically meaningful poles, gains, friction, possible gravity and spring loads, and appropriate energy balance.
4. **Actual target simulator response** — real PhysX discrete stepping with version-pinned mass/inertia, armature, friction, gravity, drive and solver settings.

The principal experiment is: **does a PhysX articulation, driven by the *measured* generalized effort, reproduce the measured joint trajectory with parameters that explain its poles, input gain, friction and energy dissipation?**

Do **not** replace G9 or G10, silently relabel uncertain contact as free motion, invent ground-truth mass, set inertia to zero to resolve a bad torque projection, turn a diagnostic result into a production export, or train an expressive mechanism residual before testing the minimal model. Preserve existing tests and original data. No commits/pushes unless explicitly requested.

### Success criteria, in one sentence

The system must (a) estimate the **identified response features** even where absolute inertia is not identifiable, (b) explicitly explain what each feature constrains, (c) reconcile the analytical response with the observed input–output trajectory, (d) refine physically admissible simulation parameters in *actual PhysX*, and (e) abstain precisely where input, contact, geometry, model adequacy, or parameter scale is unresolved.

---

## 1. Mandatory repository audit before editing

**The local Ubuntu checkout has substantial uncommitted G9/G10 work not present in public GitHub `main`.** Inspect the *local working tree*, not only remote files. First execute:

```bash
cd ~/ForceArtGS
git status --short
git diff --stat
find forceartgs_g9 -maxdepth 4 -type f | sort
find tests -maxdepth 1 -type f | grep -E '(g9|g10|physx|free|forced)' | sort
find docs -maxdepth 1 -type f | grep -E '(G9|G10)' | sort
python -m forceartgs_g9.cli --help
```

Read existing current versions of these files **if present**:

- `forceartgs_g9/estimators/model_spec.py`: `G9Interaction`, `ModelSpec`, parameter conventions and ZOH effort interface.
- `forceartgs_g9/estimators/general_integral_ls.py`, `general_pinn.py`, `general_ekf.py` / `general_akf.py`, `general_constrained.py`, and any `free_forced*.py` added locally.
- `forceartgs_g9/dynamics/general_1dof.py`: sign, friction, stick/slip, gravity and rollout conventions.
- `forceartgs_g9/adapters/forceartgs_g9.py`: accepted `JointAuthority`, contact intervals and force/torque projection; **reuse, never rederive input sign**.
- `forceartgs_g9/physx_refinement/`: G10 API, worker, optimizer, adapters, profile and parameter status code as locally installed.
- `forceartgs_g9/runtime_scripts/sapien_rollout.py`: earlier G9 parity path; check for hidden damping and do not assume it is already repaired.
- `forceartgs_g9/validation/general_identifiability.py`, `simulator_parity.py`, real-data adequacy and force audit.
- `forceartgs_g9/export/dynamics_v2.py` and current G10 handoff/export validity logic.
- `docs/G9_GENERAL_PINN_AKF_IMPLEMENTATION_REPORT.md`, `docs/G10_PHYSX_REFINEMENT_IMPLEMENTATION_REPORT.md`, latest G9/G10 corrections, and associated config/batch scripts.

Create `docs/G9_G10_IO_IDENTIFICATION_AUDIT.md` mapping the observed local API signatures to the proposed entry points. **Do not overwrite existing modules with assumed signatures.** New code must integrate through narrow adapters; if names differ, adapt the local implementation and record the deviation. Reuse existing source-index joins, experiment provenance, `force_policy`, release/contact classifications, and simulator environment worker.

Important verified upstream contracts (may have changed locally):

```python
# Existing shape, for orientation only; inspect actual local file.
@dataclass
class G9Interaction:
    name: str
    joint_type: str                     # revolute / prismatic
    t_obs: np.ndarray
    q_obs: np.ndarray
    q_std: np.ndarray
    effort: object                      # forward-ZOH: t_s, levels, at(t)
    initial_state: Optional[dict] = None
    meta: dict = field(default_factory=dict)
```

`forceartgs_g9/cli.py` has a `dynamics` stage; the local G10 changes reportedly added `physx-refinement`. Integrate an **opt-in `io-identification`** action and optional `--use-io-initialization` for G10 without breaking old commands. Preserve append-only results such as `<run>/dynamics/g9/.../vNNN` and input content hashes.

---

## 2. Dynamics and conventions — derive rather than assume

### 2.1 General 1-DOF passive equation

For joint coordinate `q` and **signed force/torque exerted on the object** `u(t)`, write

\[
\boxed{M(q)\ddot q+C(q,\dot q)+c\dot q+F_c\operatorname{sgn}(\dot q)+G(q)+\tau_{\rm mech}(q,\dot q)=u(t).}
\tag{1}
\]

For the initially supported **fixed-base rigid part + ideal single joint**, `M(q)=M` is constant and `C=0`. Keep these terms explicit in the derivation and fail loudly if later scope requires additional DOFs, robot compliance or externally constrained contacts. `M=I_h` in kg m² for revolute and `M=m_eff` in kg for prismatic. `c>=0`, `F_c>=0`, `F_s>=F_c`; static friction governs a separate stationary mode. Do not call a fitted effective `M` a uniquely recovered physical mass distribution.

For a revolute joint of unit axis `a` through origin `o`, moving COM `c0` at reference coordinate `q_ref`, and `g` in the common world frame:

\[
\begin{aligned}
 c(q)&=o+R_a(q-q_{\rm ref})(c_0-o),\\
 Q_g(q)&=\big[a\times(c(q)-o)\big]^\top m g,\\
 G(q)&=-Q_g(q)=\frac{dU_g(q)}{dq},\\
 I_h&=a^\top I_{\rm COM}a+m\| (I-aa^\top)(c_0-o)\|^2.
\end{aligned}\tag{2}
\]

For a prismatic joint of unit direction `a`, `G=-m a^Tg` (constant for uniform gravity); ideal single-link moving mass `M=m`, augmented by documented reflected inertia/actuator mechanisms where appropriate. Respect the actual PhysX inertial representation when computing effective mass.

A **vertical revolute axis** under vertical gravity has `G=0` and `G'(q)=0`, but **nonzero inertia**; a horizontal prismatic slide has `G'(q)=0` (possibly nonzero constant `G`). A tilted hinge may have `G(q)` and `G'(q)`; uncertainty in the axis and COM must be propagated, not absorbed into density.

### 2.2 Local transfer-function model (only in the appropriate regime)

Linearize Eq. (1) around an admissible **moving** operating point `(q_bar, v_bar, u_bar)`, *away from sticking, reversals, impacts, limits, and changing contact mode*, after subtracting the nominal constant generalized load:

\[
M\,\delta\ddot q + B_{\rm eff}\,\delta\dot q + K_{\rm eff}\,\delta q=\delta u,
\tag{3}
\]

where

\[
B_{\rm eff}=c+\left.\partial_{\dot q}\tau_{\rm mech}\right|_{q_b,v_b},\quad
K_{\rm eff}=G'(q_b)+\left.\partial_q\tau_{\rm mech}\right|_{q_b,v_b}.
\tag{4}
\]

Ideal Coulomb friction contributes a **fixed signed offset** in a fixed-velocity-sign moving regime; its local velocity derivative is zero, *not* an invented continuous viscous coefficient. If Stribeck or other smooth nonlinear friction exists, account for its local derivative separately. `K_eff` can be **negative** at an unstable gravitational operating point; do not force all poles stable if a real conservative potential has unstable equilibrium.

The expected **input-effort -> joint displacement** transfer function is

\[
\boxed{H_q(s)=\frac{\delta Q(s)}{\delta U(s)}=\frac{1}{Ms^2+B_{\rm eff}s+K_{\rm eff}}
=\frac{g_u}{s^2+\lambda s+\kappa}.}
\tag{5}
\]

Define

\[
g_u=1/M,\qquad\lambda=B_{\rm eff}/M,\qquad\kappa=K_{\rm eff}/M.
\tag{6}
\]

Velocity-output response is `H_v(s)=s H_q(s)`; with negligible stiffness, `H_v(s)=g_u/(s+lambda)` and `H_q(s)=g_u/[s(s+lambda)]`. These are *continuous-time, local, sliding* response models, **not the full static-friction or solver model**.

Joint-specific expectations:

| Physical joint | Working local physics | Expected response to analyze |
|---|---|---|
| Approximately vertical revolute door, passive | `I_h qdd + c qd + Fc sign(qd) = u` | Decay pole `-c/I_h`, effort gain `1/I_h`; fixed-sign Coulomb offset |
| Horizontal prismatic slider, passive | `m_eff qdd + c qd + Fc sign(qd) = u` | Decay pole `-c/m_eff`, effort gain `1/m_eff` |
| Tilted revolute hinge | Above + verified `G(q)` | Local `K_g=G'(q_bar)`; geometry-dependent conservative load |
| Joint with independently supported spring/closer | Above + `tau_mech(q,qd)` | Configuration-dependent tangent stiffness and possibly direction-dependent damping |
| At rest, stick/slip, limit, collision | Hybrid complementarity/impulse model | **No single transfer function** across these events; use PhysX event and trajectory tests |

### 2.3 Physical transient signatures

For constant signed sliding effort `u0`, approximately constant conservative load `G0`, and velocity sign `s_v` over a window:

\[
M\dot v+c v=u_0-G_0-s_v F_c.
\tag{7}
\]

With `c>0`:

\[
\begin{aligned}
\lambda&=c/M,\quad T=1/\lambda=M/c,\\
v_\infty&=(u_0-G_0-s_v F_c)/c,\\
v(t)&=v_\infty+(v_0-v_\infty)e^{-\lambda(t-t_0)},\\
q(t)&=q_0+v_\infty\Delta t+(v_0-v_\infty)(1-e^{-\lambda\Delta t})/\lambda.
\end{aligned}\tag{8}
\]

Handle `c -> 0` analytically with the finite limit `v=v0+[(u0-G0-s_v Fc)/M] Delta t`, not division by a tiny c. Do not evaluate an exponential continuation **past a zero-velocity crossing**: terminate the sliding segment, then invoke the static-friction or free-breakaway model. General nonconstant input must use exact ZOH effort segments and numerical state-space integration.

Use the response features `lambda`, `g_u`, `kappa`, `v_inf`, Coulomb offset, onset/stop times and signed work where defensible. Record confidence and provenance for each feature.

---

## 3. Free-motion vs forced-motion identifiability: nonnegotiable distinction

### 3.1 Truly force-free moving intervals

Only intervals supported as **noncontact / no applied generalized input** by the existing evidence policy can be called force-free. On these intervals,

\[
\ddot q+\lambda\dot q+\gamma\operatorname{sgn}(\dot q)+\frac{G(q)+\tau_{\rm mech}(q,\dot q)}{M}=0,
\quad\gamma=F_c/M.
\tag{9}
\]

Under zero/negligible conservative/mechanism load, `(lambda, gamma)` can be estimated from sufficiently informative decay. But multiplying `(M,c,Fc)` by the same `k>0` gives **identical free motion**. No estimator (PINN, EKF, transfer fit, PhysX) may claim absolute `M` from this alone.

Qualification: if a **separately measured absolute external load** exists (e.g. known gravitational torque from an independently measured mass/COM), it can supply an absolute scale; do not apply the simple scale-invariance claim blindly when that information is available. If mass and gravity are scaled jointly by the same unknown density, the ambiguity generally remains.

### 3.2 Forced intervals

When the **measured object generalized effort** is valid and informative, its input gain `g_u=1/M` can become identifiable. A small effort is acceptable if its uncertainty and resulting motion are resolvable. A large effort is *not* automatically informative if the contact-point geometry, sign or time sync is wrong.

Keep input and output on their respective original timestamp grids; apply measured effort by the existing source-backed ZOH convention (do not interpolate contact force into fictitious ramps). Treat reference torque from a force-controlled robot as a **measured closed-loop interaction input**, not necessarily an exogenous independent control signal. Report possible input/output noise correlation and use robust output-error / jointly modeled input where needed.

### 3.3 Contact semantics and input provenance

Preserve existing states, adapted to the local classifier naming:

- `FREE_VERIFIED`: no contact / no external object effort; appropriate for normalized free response.
- `FORCED_VERIFIED` or `FORCED_CONDITIONAL`: measured effort is available with explicit verified/declarative calibration status.
- `CONTACT_UNKNOWN`, `NEAR_ZERO_FORCE_IN_CONTACT`, `HYPOTHESIS_ONLY`, `TRANSITION`, `LIMIT_CONTACT`, `UNSUPPORTED_KINEMATICS`: **not** silently reclassified as free; exclude from relevant fits, retain evidence and reason codes.

For valid contact, project the **force on object** to generalized effort only once:

\[
\begin{aligned}
u_R&=a^T\big[(p_{contact}-o)\times F_{object}+M_{contact}\big],\\
u_P&=a^T F_{object}.
\end{aligned}\tag{10}
\]

When the existing point-force model omits `M_contact`, preserve that assumption and uncertainty. Reuse the canonical force sign, sensor transformation and `JointAuthority` q/axis conventions. Do **not** add a second `-1` sign switch or retare/rejoin data on new indices. The diagnostic question is whether signed input work and angular acceleration have a consistent physical explanation, **not** whether sign choice can be optimized away.

---

## 4. New estimator layer: physics-constrained input/output identification

### 4.1 Minimal model family; no automatic extra terms

Fit in this order:

1. `L0`: inertial + viscous; optional fixed/verified geometric gravity.
2. `L1`: L0 + Coulomb sliding friction.
3. `L2`: L1 + identifiable static breakaway, if confirmed starts/stops exist.
4. `L3`: L1/L2 + **physically supported** conservative spring/closer, only when there is repeatable configuration dependence and adequate trajectory coverage.
5. `L4` (research-only): structured ForceTwin-like residual after L0–L3 fail adequacy and independent validation exists. Do not let its position head correlate with qdd and masquerade as inertia.

Compare nested variants using **the same observation windows**, parameter counts, model adequacy, stability/passivity where applicable, and held-out data when available. Never turn fitted terms off quietly; record each candidate. No configuration-dependent gravity/head separation claim without independent mass/COM or mechanism evidence.

### 4.2 Stage A — response-signature extraction from measured q(t), u(t)

Do **not** estimate `H_exp(jw)=FFT(q)/FFT(u)` as the default for the short noisy real pushes. Implement a **continuous-time physics-parameterized output-error fit** against *position samples*:

\[
\boxed{\hat\psi=\arg\min_\psi\sum_{e\in\mathcal E_{train}}\sum_i
\rho_\delta\left(\frac{q_{model}(t_{ei};u_e,\psi,q_{0,e},v_{0,e})-q^{obs}_{ei}}
{\sigma_{q,ei}^{eff}}\right)+\mathcal R_{IC}+\mathcal R_{phys}.}
\tag{11}
\]

Here `psi` parameterizes `lambda`, `gamma`, `kappa`, and **only for informative forced intervals**, `g_u`. For free-only segments either fix `M` to a clearly declared scale to permit forward simulation *or use the normalized equation directly*; do not optimize absolute `M` against homogeneous residual. Allow `q0` and `v0` to be estimated per episode **with explicit observation-driven bounds/priors**. For unaccepted geometry, all outputs are conditional diagnostics only.

Implement two complementary estimators:

- **Exact/ODE transient output error**: solve Eq. (8) on fixed-sign constant-effort subwindows, otherwise integrate the physically minimal hybrid 1-DOF model, preserving ZOH boundaries.
- **Integral/weak-form normalized regression**: on genuine free sliding windows, use

  \[
  \Delta v+\lambda\Delta q+\gamma\int_{t_a}^{t_b}\operatorname{sgn}(v)dt
  +\int_{t_a}^{t_b} a_g(q)dt=0,\tag{12}
  \]

  with a correctly modeled geometric-gravity contribution `a_g=G/M`, covariance from `q` and velocity inference, and no cancellation across stick/reversal/force-transition boundaries. For forced intervals add `-g_u \int u(t)dt` on the left. Integrate input ZOH exactly, while `v` is estimated from the same uncertainty-aware trajectory model (not raw second differences).

Choose feature estimates from the **full joint-space fit**; a local linear transfer only applies where its assumptions are satisfied. For longer excited records, optional spectral validation is permitted, but require input spectral support, coherence/noise estimates, and absence of unmodeled contact switching.

### 4.3 Stage B — reconcile features with joint physics

The estimated signature must map to physical parameters by algebra, with provenance:

\[
M=1/g_u,\qquad c=M\lambda,\qquad F_c=M\gamma,\qquad K_{eff}=M\kappa.
\tag{13}
\]

If `g_u` is not identifiable, return `M,c,Fc` as an **equivalence family**:

\[
\mathcal F=\{(M,M\lambda,M\gamma):M\in[M_{min},M_{max}]_{prior},\ M>0\},\tag{14}
\]

not a unique measured mass. Prior bounds must be explicitly marked `ASSUMED` or `MEASURED`, never hidden in a field called `value`. For a revolute link, infer only `I_h`, not full inertia tensor; conditional geometry mapping can produce physically plausible `m,COM,I_COM` consistent with the observed hinge inertia.

Compute pole locations and predicted response per candidate:

\[
 p_{1,2}=\tfrac12\big(-\lambda\pm\sqrt{\lambda^2-4\kappa}\big),\tag{15}
\]

and annotate unstable/gravity-unsteady cases appropriately. Compare the recovered `K_eff` to `G'(q_bar)` from the accepted geometry (including axis/COM uncertainty). Any excess may indicate a mechanism, a joint drive, or model mismatch; **do not instantly call it a spring**.

### 4.4 Stage C — bridge to existing PINN and augmented EKF

Keep real G9 PINN physics and EKF covariance: do not replace them by a transfer-function-only estimator.

- Initialize the PINN with measured `(lambda,gamma,kappa,g_u)` response features where supported. In a free sliding segment use the **normalized residual** `qdd+lambda qd+gamma sign(qd)+a_g(q)=0`. In a forced segment use the same physical model with the known signed `g_u u(t)`. Preserve data, weak-form, impulse, and start-state losses. Never train a zero-scale `(M,c,Fc)` homogeneous residual.
- EKF state should support a normalized free-mode model `(q,qd,log lambda,log gamma[,kappa])` where meaningful, then include a **separately justified gain/inertia state** for informative forced motion. Preserve Joseph covariance and state retraction conventions if applicable. Parameter covariances must reflect rank deficiency; no fake certainty from a 3-seed spread.
- If an independently established mass/COM is given, fit geometry-coupled gravity consistently rather than creating unrelated inertial and gravitational masses.
- Compare integral LS, PINN, EKF, analytical output-error and the PhysX fit **on exactly the same labeled windows and native clock**. Report incompatible estimates instead of averaging them.

---

## 5. Physical consistency before optimizing PhysX

### 5.1 Input work, kinetic energy, potential and dissipation

For the ideal one-DOF model, with known/assumed `M` and conservative `U(q)`, the mechanical balance is

\[
\boxed{\int_{t_a}^{t_b}u\dot q\,dt
=\Delta\left(\tfrac12M\dot q^2+U(q)\right)
+\int c\dot q^2 dt+\int F_c|\dot q|dt+W_{other}.}
\tag{16}
\]

`W_other` must be explicitly modeled, e.g. an independently supported dissipative mechanism, additional known work, or a reported unexplained residual. Never silently allow negative dissipation to repair a wrong sign. For a purely passive, gravity-consistent coasting segment `E(t)` is non-increasing (apart from quantified numerical error / external energy exchange).

Implement **diagnostic energy and impulse audits**, not automatically another independent loss term from the same samples:

- Signed input impulse and change in generalized momentum; include friction, gravity and mechanism integrals and full uncertainty.
- Signed input work `\int u\,dq`, change in kinetic + potential energy and positive modeled dissipation.
- Sensor sign, lever arm, uncertainty in contact location, time offset, source-frame alignment, q axis and initial velocity uncertainty, with declared bounds.
- Decompose failures: `INPUT_OUTPUT_SIGN_INCONSISTENT`, `MOTION_NOISE_DOMINATED`, `CONTACT_UNKNOWN`, `NONPASSIVE_UNMODELED_FORCE`, `INCONSISTENT_GRAVITY_GEOMETRY`, `SHORT_UNINFORMATIVE_WINDOW`, `POSSIBLE_UNMODELED_DRIVE` etc.

**Fridge regression:** retain its previously reported short-pulse sign/timing contradiction as a known diagnostic. Do not flip the data, fit a negative inertia, or count `near-zero measured effort during contact` as a free coast. If no admissible interval remains, report `NO_ADMISSIBLE_EPISODE` and make a plot/diagnostic rather than inventing identification.

### 5.2 Robust model adequacy

An effort-equation score below sensor noise can be achieved trivially by `(M,c,Fc)->0`. **It is not sufficient**. Report:

- Free-coast position replay RMSE, maximum error, stop-time error, normalized and *absolute* units.
- Forced interval position replay and signed-work diagnostics.
- Parameter profile/rank/uncertainty separately.
- Response-feature agreement (decay pole, gain if identifiable, friction offset); never label `g_u` measured from free-only movement.
- Independent episode results separately from training fit; no heldout means `NOT_VALIDATED`, not necessarily `STRUCTURALLY_UNOBSERVABLE`.

---

## 6. PhysX-native G10 refinement: transfer signatures as initialization and constraints

### 6.1 Simulator forward model and minimum parameterization

Reuse the **working local G10 worker** that executes in `parttwin_sim`; do not move estimation to PArtTwin1D. Define `Phi_PX^S` as the *actual pinned runtime* stepper, with engine config `S` (SDK/version, timestep, substeps, solver iterations, precision, gravity, sleep, link damping, drives, friction API and values, contact model).

\[
 x_{i+1}^{PX}=\Phi_{PX}^{S}(x_i^{PX},u_{[t_i,t_{i+1})},\theta_{PX}),\qquad
 y_i^{PX}=q_i^{PX}.\tag{17}
\]

`theta_PX` is a selected subset of:

- Rigid-link mass/COM/inertia from a physically admissible geometry template and one declared scale `rho` (never negative or zero mass).
- Optional **nonnegative armature**, only when its physical role is justified and separability from link inertia is discussed.
- Native static and dynamic friction efforts and viscous friction **only if that exact installed PhysX runtime exposes and supports them**.
- Otherwise, a labeled **diagnostic explicit-friction torque** in SAPIEN with a tested parity gap; do not advertise this as native PhysX identification.
- Optional spring/closer only after evidence supports it.

**Never fit link inertia and armature freely together from one joint trajectory and then claim both were separately identified.** They often enter only through their sum in the ideal one-DOF equation. Fit effective joint inertia first, then map to a plausible simulator representation with provenance.

### 6.2 Strict runtime cleanliness

Use the corrected G10 worker and assert/read back before each simulation:

- All relevant rigid-link linear and angular damping disabled during baseline passive fitting unless explicitly part of the identified physical model. The previous link angular default `0.05` must not survive unnoticed.
- Zero unintended legacy joint friction; record actual loaded value and updated/new friction mode selection.
- Zero drive stiffness and **zero drive damping**, even where target position is continuously assigned. A drive with `stiffness=0` but `damping>0` is *not passive-free*.
- Disable/explicitly control sleep for tiny coasting velocities; record sleep state where possible.
- Exact integer step counting; exact treatment of ZOH force timestamps and query times, with `q` readback.
- Inertial mass, COM, full 3x3 tensor (symmetric positive definite), axis, gravity, timestep and solver readback; fail when inconsistent with the generated object description.
- No unintended collision or joint-limit impulse in a window assumed to be free sliding.
- Keep CPU vs GPU, SAPIEN vs Isaac, and engine versions separate. Do not silently port parameters across incompatible PhysX API generations.

**Version-specific warning:** The current SAPIEN 3 Python joint interface exposes `set_friction(float)`, `set_armature(array)` and `set_drive_properties(...)`. The newer PhysX/Omni Physics API also has `staticFrictionEffort`, `dynamicFrictionEffort`, `viscousFrictionCoefficient`, and per-axis `armature`. These are **not universally exposed through SAPIEN 3**, and the legacy `joint.friction` coefficient is **not** a Coulomb torque in N m or N. The implementation must run API capability checks and label its engine coverage accurately. Unit conversion is mandatory when mapping revolute radians in ForceArtGS to any USD angle conventions.

### 6.3 Main refinement loss: simulator output error

Fit `theta_PX` and **per-episode nuisance initial velocity** under physically admissible bounds:

\[
\boxed{\begin{aligned}
\mathcal L_{PX}(\theta,\{v_{0,e}\})={}&
\mathcal L_q+\lambda_{IC}\mathcal L_{IC}
+\lambda_{prior}\mathcal L_{prior}
+\lambda_R\mathcal L_{response}+\lambda_{event}\mathcal L_{event}.
\end{aligned}}\tag{18}
\]

Primary data term:

\[
\boxed{\mathcal L_q=\frac{1}{|\mathcal E|}
\sum_{e\in\mathcal E}\frac{1}{N_e}
\sum_i\rho_{Huber}\left(
\frac{q_{ei}^{PX}(\theta,v_{0,e})-q_{ei}^{obs}}
{\sqrt{\sigma_{q,ei}^2+\sigma_{PX,ei}^2+\sigma_{geom,ei}^2}}
\right).}\tag{19}
\]

Use the actual observation times and preserve the source's covariance structure where available. `sigma_PX` comes from **separate numerical precision/time-step convergence checks**, not an arbitrary inflation to excuse model inadequacy. `sigma_geom` is estimated from accepted joint geometry uncertainty; do not double-count correlations. Report effective degrees of freedom / whitened residuals where covariance can be estimated.

Initial state nuisance prior:

\[
\mathcal L_{IC}=\sum_e
\left[\frac{(q_{0,e}^{PX}-q_{0,e}^{obs})^2}{\sigma_{q0,e}^2}
+\frac{(v_{0,e}^{PX}-\hat v_{0,e})^2}{\sigma_{v0,e}^2}\right].\tag{20}
\]

Use a **separately constructed** uncertainty-aware `v0` estimate; do not let a free nuisance velocity absorb an otherwise informative input pulse. No rewriting of `q_obs` to align simulations after each fit.

Physical prior `L_prior`: nonnegative `c,Fc,Fs`, `Fs>=Fc`, positive mass / SPD inertia, geometry-consistent COM and hinge; priors on size/mass only with explicit origin and uncertainty. A hard physical constraint is not an independent observation. Use proper nondimensional scaling for optimizer variables.

Response regularizer (only where statistically supported):

\[
\mathcal L_{response}=
(\hat r_{exp}-r_{model}(\theta))^T
\Sigma_r^{\dagger}(\hat r_{exp}-r_{model}(\theta)),\qquad
r=[\lambda,\gamma,g_u,\kappa,\text{stop features}].\tag{21}
\]

`r_model` comes from analytical linearization of the **same simulator-inertial model**, plus small engine characterization rollouts if discrete friction changes the response. Exclude unidentifiable features from `r` rather than assigning them zero variance. Because these features are derived from the same `q,u`, **avoid double counting**: default to diagnostic/initialization use (`lambda_R=0`) unless joint covariance with `L_q` is modeled or a documented, cross-validated regularizer is explicitly enabled. Never call the feature term an extra independent measurement.

Event loss `L_event`: only for independently supported stop/breakaway/limit events with event-time uncertainty; robust loss on event time and direction. A moving low-force sample is not a confirmed sticking event.

Energy balance `L_energy` is by default **audit-only** for the same input–output samples. Introduce as a penalty solely when its uncertainty/dependence has been documented and weighting justified; it is not a generic independent fourth data source.

For sensitivity to initial conditions, provide optional **multi-shooting** windows: each segment has a locally bounded initial state with a continuity regularizer, while final predictive evaluation uses *single-shooting open-loop replay* with no teacher forcing.

### 6.4 Staged fit and model reconciliation

Perform optimization in this order:

1. **Input/kinematics audit**: select admissible free and forced episodes; build a detailed exclusion log.
2. **Experimental response identification**: fit `lambda`, `gamma`, optional `kappa`, and `g_u` *only* if valid forced excitation supports it. Bootstrap uncertainty by episode/block or a valid time-correlated noise model.
3. **Physical mapping**: initialize `M=1/g_u`, `c=M lambda`, `Fc=M gamma` where possible; otherwise use an `M` prior grid with conditional friction scaling.
4. **PhysX engine sanity and parity**: frictionless drift, free decay, forced impulse, gravity, sticking and stopping; check actual engine fields.
5. **Coarse profile**: evaluate a log-spaced physically plausible mass/inertia scale `rho` grid; optimize conditional `c,Fc` and optional `Fs`; record convergence/failures.
6. **Local PhysX optimization**: optimize only supported parameters with bounded, deterministic black-box methods (e.g. differential evolution then bounded least-squares / trust-region), warm-started by Stage A and per-profile candidates. Lock the engine config, random seed and input source.
7. **Observability and profiles**: compute finite-difference `J_ij=\partial q_PX(t_i)/\partial \eta_j` in **dimensionless transformed parameter coordinates** `eta=log(theta)` for positives; noise-whiten; report singular values, profile likelihoods, confidence regions and correlations. Check near-equivalent families globally.
8. **Selection and heldout**: report training adequacy, separately held-out real predictions if present, and conditional simulator candidate family. One well-excited episode can identify a parameter conditionally, but absent independent episodes **does not** imply independent predictive validation.
9. **Fail-closed handoff**: never publish a non-identified point estimate in an ambiguous `value` field; use `estimate_not_for_sim`, status and explicit prior-family values. A production-ready PhysX property requires the existing data/kinematics/validation gates and engine parity.

Don't let a best-fit `M` sit on a lower bound and call it an identified mass. If the mass profile is flat because of scale symmetry, report a **family** and parameter-ratio information. If PhysX cannot fit even the best admissible candidate, report `MODEL_INADEQUATE` or `INPUT_OUTPUT_INCONSISTENT`, not automatically `NOT_IDENTIFIED` for all quantities.

---

## 7. Required code layout and interfaces

Prefer *additive* modules within existing `forceartgs_g9` paths. Adapt names to the actual local checkout:

```text
ForceArtGS/
  forceartgs_g9/
    system_identification/                    # NEW, estimator-owned
      __init__.py
      contracts.py                            # typed episode/feature/result contracts
      segmentation_adapter.py                 # reuse existing classifier; no reinterpretation
      physical_response.py                    # local M,B,K, signed Coulomb / transient predictions
      output_error.py                         # q(t) analytical ODE transient fitting
      weak_form.py                            # normalized integral identification
      transfer_features.py                    # poles, gains, offsets, feature covariance
      energy_audit.py                         # work, momentum, passivity diagnostics
      identifiability.py                      # rank, profile, bootstrap, scale family
      physx_bridge.py                         # local G10 worker adapter and parameter mapping
      selection.py                            # nested model selection + status semantics
      cli.py                                  # optional standalone entry point
    physx_refinement/                         # EXISTING local G10; extend, don't replace
      ...                                     # worker, optimizer, mapping, profile, handoff
    estimators/                               # EXISTING PINN/EKF/LS adapt optional init
    config.py / core.py / cli.py / stage.py    # minimally integrate, preserve old default
    validation/                               # existing G9 adequacy/parity + new checks
  configs/g9_real/io_identification.example.yaml
  tests/test_g9_io_*.py
  docs/G9_G10_IO_IDENTIFICATION_IMPLEMENTATION_REPORT.md
```

Public Python API (adjust actual local types, keep stable contract):

```python
@dataclass(frozen=True)
class ResponseFeature:
    name: str                     # lambda, gamma, gain, kappa, pole1, pole2, ...
    estimate: float | None
    units: str
    status: str                   # IDENTIFIED | CONDITIONAL | RATIO_ONLY | NOT_IDENTIFIED | NOT_APPLICABLE
    uncertainty: dict
    evidence: list[str]
    source_windows: list[str]


def identify_io_response(
    episodes: list[G9Interaction],
    accepted_joint: dict,             # actual JointAuthority / local adapter
    contact_classification: object,   # reuse real G9/G10 classification
    gravity_model: object | None,
    config: dict,
    output_dir: str,
) -> dict: ...


def initialize_physx_refinement(
    io_result: dict,
    existing_g10_config: dict,
    existing_g10_episodes: object,
) -> dict: ...


def validate_response_consistency(
    io_result: dict,
    physx_rollouts: dict,
    config: dict,
) -> dict: ...
```

Use **one** canonical input-episode schema; wrappers should not independently reconstruct forces or re-filter timestamps. Snapshot hashes of `JointAuthority`, force record, classifier config, mass geometry and engine build. Implement schema version `forceartgs.g9.io_response/v1`; do not overload existing `G9 physics_spec_v1` with newly inferred fields.

Configuration example (the exact schema may differ after the local audit):

```yaml
schema: forceartgs.g9.io_identification/v1
enabled: false                     # opt in; old G9/G10 results reproducible
joint_source: accepted_authority    # diagnostic working authority allowed only with explicit opt-in
contact_policy: inherit_g9_g10     # no new silent free/noncontact classification
model_variants: [inertial_viscous, inertial_viscous_coulomb]
gravitation: inherited_geometry_and_world_gravity
fit:
  method: output_error_and_weak_form
  initial_velocity: uncertainty_aware_fit
  input_interpolation: forward_zoh
  robust_loss: huber
  feature_bootstrap_seeds: 8
  local_linearization: moving_only
physx:
  inherit_g10_worker: true
  use_response_initialization: true
  response_regularizer_weight: 0.0     # derived from same q,u; avoid double counting
  energy_penalty_weight: 0.0          # energy audit by default
  disable_unexpected_link_damping: true
  disable_unexpected_joint_drive: true
  require_engine_property_readback: true
  mass_scale_profile: true
validation:
  synthetic_ground_truth_required: true
  real_heldout_required_for_production_export: true
  forbid_point_export_without_identifiability: true
  report_absolute_and_normalized_error: true
```

Reject unknown keys. Do not auto-enable a mechanism head or spring. Runtime availability check should detect SAPIEN only in `parttwin_sim` and use the current worker; no installing SAPIEN into the ForceArtGS training environment and no hidden Conda env mutation.

### CLI requirements

```bash
# Discover actual local args; these are desired behavior, not assumed existing syntax.
python -m forceartgs_g9.cli io-identification \
  --run /path/to/forceartgs_joint_aware_scene_run \
  --io-config configs/g9_real/io_identification.example.yaml

python -m forceartgs_g9.cli physx-refinement \
  --run /path/to/forceartgs_joint_aware_scene_run \
  --use-io-initialization
```

Append outputs under an existing run, e.g. `dynamics/g9/io_identification/vNNN` and `dynamics/g9/physx_refinement/vNNN`; never overwrite old studies. No default real batch until tests and diagnostic comparison pass.

---

## 8. Reporting and fail-closed classification

Required machine-readable outputs:

- `io_response.json`: per episode/windows, effective model class, input source, q/effort timestamps, poles, `lambda, gamma, kappa, g_u` with confidence and units, inferred physical parameters only when supported.
- `transfer_functions.json`: exact local model family `num/den`, operating point, whether sliding-only, and normalization; no unqualified global TF for stick-slip.
- `work_energy_audit.json`: signed work/impulse, modeled dissipation and uncertainties, contradiction reasons.
- `parameter_profiles.json`: log-M profile, best loss vs M, full family, finite-difference sensitivity singular values/correlations.
- `physx_reconciliation.json`: analytical-vs-PhysX feature errors, numerical/engine settings, property readback and deviations.
- `training_replay.npz`, `heldout_replay.npz` (if any): observed q/uncertainty, analytical q, PhysX q, effort aligned by documented timeline.
- `summary.md`: self-contained scientific report with plots/tables for each real scene; detailed exclusion log.

Status dimensions **must be separate**:

1. `INPUT_VALIDITY`: VERIFIED / CONDITIONAL / CONTRADICTED / UNKNOWN.
2. `KINEMATICS_VALIDITY`: ACCEPTED / CONDITIONAL / UNACCEPTED.
3. `STRUCTURAL_IDENTIFIABILITY`: FULL / RATIO_ONLY / RANK_DEFICIENT / NOT_TESTABLE.
4. `PRACTICAL_IDENTIFIABILITY`: INFORMATIVE / WEAK / BOUNDARY_ONLY / UNSTABLE.
5. `MODEL_ADEQUACY`: ADEQUATE / INADEQUATE / NOT_RUN.
6. `INDEPENDENT_VALIDATION`: PASS / FAIL / NOT_AVAILABLE.
7. `SIMULATOR_READINESS`: PRODUCTION_READY / CONDITIONAL_PREVIEW / BLOCKED.

Example fridge output may be `INPUT_VALIDITY=UNKNOWN`, `STRUCTURAL_IDENTIFIABILITY=NOT_TESTABLE` if no valid interval is admitted. **Do not** claim a specific unobservable physical parameter on the basis of a classifier exclusion alone. If a force-free segment is admitted but absolute inertia scale is unknown, report `RATIO_ONLY` instead. Seed-to-seed spread is not parameter uncertainty; do not use it as a calibrated confidence interval.

Simulation handoff: preserve the G10 rule that non-identifiable point estimates appear **only** as `estimate_not_for_sim`; provide `simulator_prior_family` (parameter constraints and conditional candidates) separately. Do not emit a confusing general `value` field. Mark native/diagnostic friction implementation clearly. Existing validated export requirements remain untouched.

---

## 9. Mandatory tests and numerical experiments

Create dedicated tests for both mathematical properties and **real engine** behavior. Existing 40 G10 tests must still pass; never replace them with mocks or claim native PhysX properties tested when they were not.

### T1 — Exact analytical transfer response

- Revolute and prismatic `(M,c,K)` cases; verify local poles and `H_q, H_v` by direct numerical ODE integration.
- `K=0` must yield one integrator pole at 0 and decay pole `-c/M`.
- Nonzero real spring or geometry-derived gravity stiffness yields appropriate complex/real poles. Negative gravity stiffness may yield instability without being clamped.
- At a fixed velocity sign, verify Coulomb is an offset rather than a linear derivative. Handle `c=0` limit analytically.

### T2 — Free-motion scale invariance

Construct two realizations `(M,c,Fc)` and `(kM,kc,kFc)` with identical `q0,v0`, `u=0`, same consistently scaled conservative loads (or no conservative load). Verify identical trajectories to numerical tolerance. The estimator must return **identical normalized parameters**, a **flat absolute scale profile**, and status `RATIO_ONLY`. It must **not** select `M=0` as a solution.

### T3 — Known-force absolute scale

Generate multiple nontrivial ZOH effort pulses with known positive `M`, `c`, `Fc`, valid contact, varying speeds. Recover `M` and ratios within predeclared tolerances, with confidence/profile covering truth. Repeat for **small but high-SNR effort**; don't gate solely on amplitude.

### T4 — Coulomb/sticking event and reversal

When fixed-sign exponential predicts a zero-velocity crossing, terminate the interval and compare against hybrid stopped state; no negative-velocity chatter and no fitting a single LTI TF through a reversal. Static threshold `Fs` must not be inferred from mere non-motion at a single unknown effort.

### T5 — Gravity and joint geometry

Vertical revolute: `G(q)=0` and `K_g=0`, nonzero inertia. Tilted revolute: analytical `G, G'` agree with finite-difference potential and PhysX gravity torque; propagate axis/COM perturbations. Horizontal prismatic: constant `G`, zero stiffness; tilted prismatic: nonzero `G`, still zero stiffness under uniform gravity. Check coordinate sign reversals preserve physical work and transform effort correctly.

### T6 — Work/impulse consistency

Synthetic passive experiment: measured input work equals kinetic/potential change and nonnegative dissipation within measurement/integration uncertainty. Inject wrong force sign / lag / lever arm; audit must identify contradiction. The fitting optimizer cannot fix sign by learning negative mass/friction.

### T7 — Experimental TF extraction validity

Short transient and high-noise q: output-error fit succeeds in regimes where raw FFT-ratio would be unstable; no fabricated frequency-response curves. Flat velocity range => Coulomb-vs-viscous weakly separated; rank/profile marks it. Do not use finite-difference qdd as uncontested ground truth.

### T8 — Native PhysX engine clean-room

In `parttwin_sim`, check both revolute and prismatic known URDFs (or existing clean hull builder), no external contacts: `qdot` stays constant in frictionless, zero gravity-along-DOF, no-drive scenes; verify zero readbacks for unexpected link angular/linear damping, joint friction, stiffness/damping and sleep. Known impulse changes velocity in the predicted direction. Engine numerical law and float precision documented. Do not infer an engine integration scheme universally from one test.

### T9 — PhysX mass/friction and scale profile

Engine native mode: if supported, verify actual readback of newer static/dynamic/viscous friction fields; otherwise mark diagnostic explicit-effort only. Free-only mass scaling retains identical motion when friction scales; known applied force breaks invariance. Tests check `M` truth profile and physical tensor SPD, and distinguish `I_link` vs armature degeneracy.

### T10 — G9 / G10 nonregression and status

- Original G9/PINN/EKF/G10 methods still run without the new opt-in config.
- No contact => classification follows explicit evidence. Near-zero force **in contact** never relabeled free.
- Existing 40 G10 tests (17 engine) still pass, plus new tests.
- Output report has all seven status dimensions and never emits nonidentified simulator `value`.
- Heldout not available => `NOT_AVAILABLE` independently of whether an individual parameter is observable.
- No overwrite of old `vNNN` results, assets, PLY, or original robot records.

### T11 — Real scene audit and comparable runs

Run low-budget *diagnostic* analyses first on:

- **Red fridge (v005)**: validate known impulse mismatch (previously reported), axis tilt vs vertical hypothesis as a separately labeled geometry test, *no admitted free interval if contact state unverified*; show whether poles/ratios are even estimable. Do not silently use the vertical hypothesis as ground truth.
- **Orange suitcase (v006)**: run conditional prismatic model; report unaccepted joint type, actual force/position prediction and sensitivity profile; do not claim physical 8 kg from an unaccepted kinematic hypothesis.
- **Black box (v005)**: operator-declared hinge, optional conditional fits; audit bound-hitting inertia and geometry.
- **Robot dog / white drawer**: explicit not-tested status if kinematics/data unavailable; don't invent an episode.

Only after green synthetic and engine tests: enable an operator-approved full fitting budget, compare local loss convergence, and save plots/parameter profiles. Do not automatically launch a multi-hour batch on a busy GPU machine. Record all commands; results may be diagnostics only without independent episodes.

---

## 10. Performance, provenance and implementation order

**Phase 0 — Audit (no parameter fitting):** map local APIs and existing tests; record toolchain / SAPIEN / PhysX versions and current files. Preserve uncommitted changes.

**Phase 1 — Analytical backbone:** implement Eq. (1)–(15), output-error fit and normalized weak form with exact transient validation and contact classification adapter. Test without importing SAPIEN.

**Phase 2 — Physical reconciliation:** work/impulse audit Eq. (16), geometry confidence, feature observability and scale family. Output scientific diagnostics.

**Phase 3 — Bridge G9 PINN/EKF:** optional feature initialization, common regimes, parameter and uncertainty reporting. No changes to frozen legacy baseline.

**Phase 4 — PhysX bridge:** use actual existing local G10 worker, clean runtime state, backend-specific native-friction capability checks. Make Eq. (17)–(21) the constrained refinement target. Existing G10 `physics` outputs still compatible.

**Phase 5 — Integration:** CLI, configs, append-only run folders, versioned handoff; tests and report; optional aggregate real diagnostics. No PArtTwin modifications.

**Performance rules:** cache immutable geometry and metadata; avoid rebuilding the URDF for every nearby parameter trial when the worker supports mutable inertials; use batched analytic transient fits to warm start expensive PhysX calls; use integer-counted steps and deterministic seeds; record all failures and abort fitting after a documented fraction of failed rollouts; do not hide failures in robust loss.

### Required final Codex deliverables

1. Source code and tests in `~/ForceArtGS` only.
2. `docs/G9_G10_IO_IDENTIFICATION_AUDIT.md` with **actual local** existing APIs, changed-file inventory and provenance.
3. `docs/G9_G10_IO_IDENTIFICATION_IMPLEMENTATION_REPORT.md`, including exact equations, methods, engine capabilities, synthetic and real-scene tables, observability results and blockers.
4. Reproducible CLI commands and at least one tested small synthetic run; do **not** claim the full real batch ran unless actually executed.
5. Detailed list of new tests and original G9/G10 suite results, with runtime location and test counts.
6. A clear classification of `observed -> inferred normalized dynamics -> recoverable physical coefficients -> PhysX-matched conditional candidates -> simulation export readiness`.
7. No `git commit`, `git push`, asset overwrite, or changes under `PArtTwin1D` without explicit user instruction.

---

## 11. References to use, not silently reimplement

- ForceTwin (Engelbracht et al., 2026), uploaded in the conversation: its **partially linear** one-DOF effort model uses nonnegative inertia, viscous and Coulomb coefficients with a subsequently learned structured mechanism residual; it explicitly cautions that the decomposition is not unique. **Do not mistake its NN residual or its parameter fit for a transfer-function identification method.**
- PhysX 5.7 articulation guide, updated joint friction and legacy coefficient distinction: <https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/Articulations.html>
- Omni Physics PhysxJointAxisAPI (`staticFrictionEffort`, `dynamicFrictionEffort`, `viscousFrictionCoefficient`, `armature`) with API-specific units: <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/joints/physx_joint_schema.html>
- SAPIEN 3 exposed articulation API, `set_qf`, `set_friction`, `set_armature`, `set_drive_properties`, `set_sleep_threshold`: <https://sapien-sim.github.io/docs/api/sapien.pysapien.physx.html>
- Output-error system identification concept and sensitivity to initial conditions: <https://www.mdpi.com/2076-3417/11/9/4303>

**Core principle:** A transfer function is an interpretable *local summary* of an assumed joint physics model; it is not a substitute for correct contact input, hybrid friction, physically consistent geometry or real PhysX execution. The output-error trajectory is primary, expected poles/gains are a strong structural test, and absolute inertia must be justified by actual scale information.
