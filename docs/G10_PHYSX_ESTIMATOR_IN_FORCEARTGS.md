# G10 — PhysX refinement lives in **ForceArtGS**, not PArtTwin1D

**Code location:** `ForceArtGS/forceartgs_g9/physx_refinement/`.

**Data owner:** ForceArtGS `G9Interaction`, joint authority and canonical force record.

**Estimator output:** `g9/physx_refinement/physx_refinement.json` plus prepared episode records. The JSON is a **diagnostic estimator output** and may be consumed by the existing PArtTwin1D export/assembler only after validation. No code is installed in PArtTwin1D; none of these modules import PArtTwin1D.

**Repository version note:** the connected public ForceArtGS GitHub source is at `136c462624aa3d4ea4f05666d6e8cc752dc860d9`; its public tree lacks newer local PINN/EKF/G9 code in the user-reported working directory. This patch is **additive** and does not overwrite `core.py`, `config.py`, `stage.py`, the existing PINN/EKF or user-uncommitted changes. Integration into the newest local stage runner must be verified on the local checkout.

## Physics model and refinement objective

Let the engine run for a fixed solver configuration `S = (engine/version, dt, solver, gravity, limits, contacts, sleep, drives, geometry)`. The optimizer controls only validated simulator-native physical parameters

\[
\theta = (\rho_s, A, T_d, T_s-T_d, c_v),\quad \rho_s>0,\ A,c_v,T_d,T_s-T_d\geq0.
\]

`rho_s` scales the existing rigid-link mass and full COM inertia tensor **together**, preserving the inertial–gravitational mass coupling. `A` is optional reflected/armature inertia, held fixed unless physically justified. Hinge inertia equals

\[
I_h=\rho_s\big(\mathbf a^\top\mathbf I_{\mathrm{COM},0}\mathbf a + m_0\| (I-\mathbf a\mathbf a^\top)(\mathbf c_0-\mathbf o)\|^2 \big)+A.
\]

For prismatic joints `M_eff = rho_s*m_0+A`. Gravity is **always** applied by the actual rigid-body PhysX solver when enabled; for vertical revolute axes the ideal gravitational joint torque is zero, but joint inertia is not.

The continuous engineering reference during sliding is

\[
M_{eff}(\theta)\ddot q+c_v\dot q+T_d\operatorname{sgn}\dot q+G(q)=Q(t),
\]

and at stick: `qdot=0`, `|Q-G|<=T_s`, assuming no additional known mechanism load. This is **not** a substitute for PhysX's discrete reduced-coordinate articulation solver.

For episode `e`, simulate

\[
(q^{PX}_{e,i},v^{PX}_{e,i})=\Phi_{PX,S}^{N_i}(q_{e,0},v_{e,0};\theta,Q_e(\cdot)).
\]

The primary objective is

\[
\boxed{\mathcal L_q=\frac1{|E_{train}|}\sum_e \frac1{N_e}\sum_i
\rho_\delta\left( \frac{q^{PX}_{e,i}-q^{obs}_{e,i}}{\sqrt{\sigma_{q,e,i}^2+\sigma_{num}^2}}\right)}.
\]

Optional (off by default) displacement-increment term:

\[
\mathcal L_{\Delta q} =\frac1{|E_{train}|}\sum_e\frac1{N_e-1}\sum_i
\rho_\delta\!\left(\frac{\Delta q^{PX}_{e,i}-\Delta q^{obs}_{e,i}}
{\sqrt{\sigma_{q,e,i+1}^2+\sigma_{q,e,i}^2+2\sigma_{num}^2}}\right).
\]

If an independent initial angular/linear velocity uncertainty exists, optionally fit per-episode `v0_e` as a nuisance with Gaussian penalty `(v0_e-v0_prior)^2/(2*sigma_v0^2)`; **the current optimizer does not yet optimize `v0`**, although the evaluation API supports explicit offsets. Do not claim that an unimplemented nuisance-variable fit is active. The optional scale prior is `lambda_rho*(log rho_s-log rho_prior)^2`, and an optimizer conditioned on it remains `PRIOR_ONLY` without absolute-scale evidence.

\[
\boxed{\min_{\theta\in\Theta}\quad \mathcal L_q+\lambda_{inc}\mathcal L_{\Delta q}
 + \lambda_{rho}(\log\rho_s-\log\rho_{prior})^2.}
\]

Only *training* episodes contribute to the objective; held-out real episodes are evaluated with frozen theta. `rmse_rad_or_m` and normalized error are recorded separately.

## Free motion vs forced motion

**Free:** requires independent contact absence, never inferred from low force alone. The object can move with nonzero `v0` and zero applied effort. Since `c_v/M_eff` and `T_d/M_eff` govern ideal friction-only free decay, absolute inertia generally cannot be identified from passive motion. The patch refuses to optimize both `rho_scale` and armature in a free-only fit: their bound intervals must be identical endpoints, representing declared nonzero priors. Fit the two friction terms conditional on this scale, report `ABSOLUTE_SCALE_UNIDENTIFIED_FREE_ONLY`. For gravity-dominated motion and physically co-scaled inertial mass, the scale ambiguity remains unless another independently known load fixes it.

**Forced:** requires independently confirmed contact and a canonical generalized effort from the existing `G9Interaction.effort`, plus either verified calibration or explicit exploratory opt-in. No forced data are silently converted into passive input. The native 50Hz force ZOH and established force-on-object sign are preserved. A fitted point still does not establish identifiability; joint impulse/momentum, parameter profile and independent demonstrations are required.

**Unknown-contact:** excluded, not converted to force-free.

The fridge's historical impulse/velocity sign discrepancy (rows 229–232) stays a BLOCKER for claiming absolute inertia until independently resolved; do not flip torque or displacement signs to make the optimizer work.

## Existing ForceArtGS contracts

Current public source audited:

- `forceartgs_g9/estimators/model_spec.py`: `G9Interaction(name, joint_type, t_obs, q_obs, q_std, effort, initial_state, meta)`.
- `forceartgs_g9/adapters/forceartgs_g9.py`: accepted joint authority, `contact_efforts`, canonical record/gates. Reuse these; do not recalibrate or re-project here.
- `forceartgs_g9/core.py`: `G9Inputs.train` / `.heldout` and `run_g9`; the patch does not change its execution or status gates.
- `forceartgs_g9/runtime_scripts/sapien_rollout.py` and `validation/simulator_parity.py`: existing engine parity tests. The new isolated diagnostic runner explicitly zeros drives; it does not replace the current parity gate.
- `forceartgs_g9/export/dynamics_v2.py`: existing handoff contract. The patch does **not** modify an unidentifiable physical point to an exportable `value`. Production export remains blocked unless old/new readiness and native parity pass.

## New files / responsibilities

- `objective.py`: typed PhysX episodes, physical parameters, actual-simulator callable loss, free/forced validation and geometry helper.
- `optimizer.py`: bounded low-dimensional SciPy differential evolution; free-only scale gauge guard.
- `g9_adapter.py`: strict `G9Interaction` conversion; explicit independent regime/force provenance; writes versioned episode JSON.
- `api.py`: ForceArtGS estimator-side direct Python entry point `run_forceartgs_physx_estimator`.
- `sapien_diagnostic.py`: physically integrated *SAPIEN/PhysX CPU* rollout with drives zeroed, mass/inertia scaling and two explicit modes. `frictionless_physx`: native frictionless integration only. `external_effort_surrogate`: manually applied `-c v-Td sign(v)` input **NOT** native static/dynamic/viscous joint friction; cannot identify stiction and cannot certify native PhysX parity.
- `mapping.py`: writes only an unverified suggested property dictionary (SI values plus USD angle-unit conversion), **not** USD assets.
- `cli.py`: separate `python -m forceartgs_g9.physx_refinement.cli`; writes ForceArtGS JSON diagnostics.
- `tests/test_g9_physx*.py`: CPU-only tests of loss, positive inertia, friction ordering, scale gauge, adapter provenance and preservation of source effort.

### Direct API usage (inside existing local ForceArtGS)

```python
from forceartgs_g9.physx_refinement.api import run_forceartgs_physx_estimator

result = run_forceartgs_physx_estimator(
    interactions=g9_inputs.train + g9_inputs.heldout,
    regime_by_name={
        # These must be independently established *for the selected intervals*.
        "free_0": {"mode": "free", "evidence": {
            "source": "operator-confirmed-no-contact",
            "contact_independently_absent": True}, "train": True},
        "forced_1": {"mode": "forced", "evidence": {
            "source": "operator-contact-segment",
            "contact_confirmed": True}, "train": False},
    },
    v0_by_name={"free_0": 0.0, "forced_1": 0.0},  # example only! replace with measured/declared velocity
    calibration_by_name={
        "free_0": {},
        "forced_1": {"status": "EXPLORATORY_ASSUMPTIONS_ACCEPTED", "exploratory_opt_in": True},
    },
    urdf="/actual/path/to/object.urdf",   # baseline physically valid articulated URDF
    physics={"engine":"sapien_cpu_diagnostic", "joint_type":"revolute",
             "moving_link":"moving_link", "joint_name":"hinge",
             "timestep":0.001,"gravity":[0,0,-9.81],
             "friction_mode":"external_effort_surrogate"},
    loss={"huber_delta":2.0,"numeric_floor_q":0.0},
    fitting={"bounds":{
        "rho_scale":(1.0,1.0), "armature":(0.0,0.0),
        "dynamic_effort":(0.0,1.0), "static_excess":(0.0,0.0),
        "viscous":(0.0,1.0)}, "seed":0, "maxiter":30},
    out_dir="/actual/ForceArtGS/run/g9/physx_refinement",
)
```

The names and file paths above are **illustrative**; do not treat `v0=0` or free-contact absence as measurements. Start with actual G9 prepared interactions and their independently established provenance. To avoid a false native-friction claim, the current SAPIEN surrogate mode is `DIAGNOSTIC_ONLY_NOT_IDENTIFIED`, **not simulation-export eligible**.

### CLI usage

Use `write_g9_physx_episodes` or `run_forceartgs_physx_estimator` above to create strictly annotated episode JSON from `G9Interaction`. Or prepare the equivalent documented JSON manually with `schema=forceartgs.g9.physx_episodes/v1` and `regime_provenance` per episode. Then:

```bash
cd ~/ForceArtGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate freeartgs
python -m forceartgs_g9.physx_refinement.cli \
  --urdf /absolute/path/to/object.urdf \
  --episodes-json /absolute/path/to/g9/physx_episodes.json \
  --config-json /absolute/path/to/physx_fit_config.json \
  --out /absolute/path/to/g9/physx_refinement.json
```

`config_json.physics` requires engine `sapien_cpu_diagnostic`, joint_type, moving_link, joint_name, timestep, gravity, friction_mode. `config_json.fitting` requires bounded parameters for `rho_scale,armature,dynamic_effort,static_excess,viscous`. `config_json.loss` is optional.

`SAPIEN` must be installed in the Python environment only when running actual simulations; unit tests need `numpy`, `scipy`, `pytest`.

## PhysX target-runtime integration still required

**This patch does not implement a native Isaac Sim PhysX friction-effort runtime.** Native per-axis `staticFrictionEffort`, `dynamicFrictionEffort`, `viscousFrictionCoefficient`, `armature` mapping needs an Isaac Sim/PhysX version supporting those attributes *and measured runtime readback*. In particular, `sapien_joint.set_friction` in the available SAPIEN API is a legacy, dimensionless load-dependent coefficient, not torque-valued Coulomb friction. Do not silently equate drive damping to physical passive damping.

Native adapter TODO (estimator-side only, no exporter modifications): implement `PhysXIsaacRollout` in this package, instantiate an isolated USD articulation, set native parameters, zero active drives/legacy friction, read back the joint properties, apply the preserved effort ZOH, return `SimulationResult` with `runtime_semantics=NATIVE_PHYSX_VERIFIED` **only after** solver-level tests. Connect it through `api.py` and retain all G9 force/visual provenance.

Acceptance tests with actual engine: frictionless inertia under known pulses, free decay at several normalized ratios, static breakaway vs dynamic sliding, vertical hinge/nonzero inertia/zero gravity torque, nonvertical gravity projection, prismatic gravity, input sign/time equality and impulse, physically valid inertia tensor under scaling, repeatable seeds, CPU/GPU agreement within declared tolerance, and frozen-parameter independent real-episode validation. No `ESTIMATED` claim from parameter optimizer alone.

## Downstream contract

Send only *estimator outputs* (including a confidence/identifiability/status table and PhysX property mapping suggestions) from ForceArtGS to downstream asset construction. **Do not pass `estimate_not_for_sim` as a simulator-ready `value`**; if absolute inertia is unknown, output a prior-conditioned correlated family of `I_eff`, damping, Coulomb effort with `c/I_eff` and `Td/I_eff` held consistent. The downstream exporter should refuse a point estimate whose status is `NOT_IDENTIFIED` or `MODEL_INADEQUATE` unless the user explicitly requests a prior-only simulation preview.

**No modification of `PArtTwin1D` is required or authorized to run the estimator.**
