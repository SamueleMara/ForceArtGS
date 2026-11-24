# G10 PhysX refinement in ForceArtGS — implementation report

Date 2026-11-23. Branch `multimodal-tcp-contact`, HEAD `9298f77` plus uncommitted local work. Nothing has been committed or pushed. The spec is
[G10_PHYSX_ESTIMATOR_IN_FORCEARTGS.md](G10_PHYSX_ESTIMATOR_IN_FORCEARTGS.md).

The following were not touched:
- PArtTwin1D (nothing is imported from it or installed into it);
- ForceArtGS-Physics;
- `core.py`, `config.py`, `stage.py`;
- the existing PINN/EKF code;
- `export/dynamics_v2.py`;
- the existing parity gate (`validation/simulator_parity.py`, `runtime_scripts/sapien_rollout.py`);
- raw data and historical results.

## 1. Starting point: the installed patch

The patch was already installed additively. It comprised `forceartgs_g9/physx_refinement/{objective,optimizer,g9_adapter,api,sapien_diagnostic,mapping,cli}.py`, three test files and `install_into_forceartgs.sh`. Its 15 tests passed as installed, but it could not run on this machine, and on real data it was incomplete:

| Problem | Effect | Fix |
|---|---|---|
| `sapien` is not in the `freeartgs` env (it is only in `parttwin_sim`) | every rollout raised `ImportError` | New standalone `runtime_scripts/sapien_physx_worker.py` (sapien + numpy only). It runs as a persistent JSON-lines worker in the engine interpreter, or in-process if this interpreter has sapien. |
| **PhysX link damping default 0.05** (linear and angular) was not zeroed | The "frictionless" runner was damped: a vertical hinge lost exp(−0.05 t) of its velocity (measured: 1.0 → 0.779 rad/s in 5 s) | Damping is zeroed on every link and read back each rollout; a nonzero readback fails closed |
| Articulation sleeping was left on | a slow coast freezes below the sleep threshold, which looks like stiction | `sleep_threshold = 0`, read back |
| Float time accumulation (`t += dt`) | query samples off by one step over long episodes | integer step counts `round((t_i − t_0)/dt)` |
| Coulomb surrogate `Td·sgn(v)`, explicit | chatters through v = 0 instead of stopping | `Td·sat(v/v_eps)` with `v_eps = Td·dt/I_eff` (the velocity change from one step of friction). Still labelled `EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE`; no stiction branch. |
| No mass/inertia readback | a URDF/engine mismatch would go unseen | The engine's mass, principal inertia and COM pose are read back each rollout and I_h is recomputed; a mismatch with `rho_s·I_h(1)` above 1e-4 relative fails closed |
| DE population sized by all 5 dimensions, including fixed ones; failed rollouts silently scored 1e15 | wasted evaluations; a dead engine gave a "fit" | DE runs only over the free dimensions. Failures are counted and reported; a fit with more than 50% failures raises. Optional warm start `x0`. |
| `q0` = raw first sample | noisy initial state | a declared `initial_state.q` (checked against `t_obs[0]`) takes precedence |
| No identifiability, adequacy, held-out, status table, family or downstream rule | a point fit was the only output | `identifiability.py` (§3) |
| No real-data path: no URDF and no regime evidence for the real scenes | could not run on any scene | `urdf_builder.py`, `g9_adapter.interactions_from_segments`, `real_data.py`, the CLI subcommand, a config and a batch script |

The existing parity runner `runtime_scripts/sapien_rollout.py` also leaves link damping at 0.05. It already reads back joint friction (memory: "SAPIEN parity always FAILs"), but link damping is a second, unreported dissipation term in that gate. It is not modified here (out of scope); fixing it would be a one-line change in that script.

## 2. Engine facts measured (SAPIEN 3.0.3, PhysX CPU)

- **Hidden defaults on load:**
  - joint friction 0.05 (legacy, dimensionless);
  - link linear and angular damping 0.05;
  - sleep threshold nonzero.

  All are zeroed and read back by the worker; `test_hidden_engine_damping_is_zeroed_and_read_back` keeps the trap documented.
- **Time stepping is semi-implicit Euler** (v += a·dt, then q += v·dt). Under constant acceleration q_n = q_0 + a·dt²·n(n+1)/2, which is ½·a·dt·t ahead of the continuous law. The engine matches this discrete law to 2e-5 for:
  - prismatic gravity;
  - a ZOH effort step;
  - tilted-hinge gravity.

  So PhysX is not the continuous ODE; the spec says the same.
- **q is float32.** About 3000 steps at |q| ≈ 3 rad accumulate around 3e-4 relative error (8e-4 rad measured). The real-data loss therefore uses `numeric_floor_q = 3e-4` (σ_num in the spec's loss).
- **Gravity and frames:** the URDF built from the G9 hull pose reproduces the G9 `RigidBodyGravity` law G(q) exactly (unit test). The engine trajectory matches the discrete map of I·q̈ = −G(q) to 5e-5 rad for axis tilts of 20° and 35° with q_ref ≠ 0. This checks the base frame, the q sign convention and the q_ref handling.
- **Speed:** about 40–60 ms per 2 s rollout at dt = 1 ms, including a fresh scene each time (14 µs per step plus about 30–60 ms for scene and URDF loading).

## 3. What was added

| File | Role |
|---|---|
| `runtime_scripts/sapien_physx_worker.py` | One clean scene per request. Zeroes and reads back defaults, uses integer steps, applies the external ZOH effort, and reports the cumulative external and friction impulses. Serve mode or one-shot mode. |
| `physx_refinement/sapien_diagnostic.py` | `IsolatedSapienCpu`: worker or in-process. Mass-scaled URDFs are cached per ρ. Computes `urdf_axis_inertia` (the spec's I_h formula from the URDF) and `readback_axis_inertia` (the same quantity from the engine). `effective_inertia(θ)`. |
| `physx_refinement/urdf_builder.py` | Single-joint URDF in the G9 record frame B (base at identity, so scene gravity is `g_B`). Takes the G9 hull pose at `q_ref` back to q = 0. Inertia only, no collision or visual shapes. Refuses non-physical tensors. `from_mass_model(mm, ρ)`. |
| `physx_refinement/identifiability.py` | `profile_scale`: friction re-optimised at each fixed ρ, warm-started and polished, with the global fit included. The 95% set is ΔL ≤ 1.92·max(1, χ²_red); crossings are interpolated quadratically in log ρ and refined by evaluating at them. Reaching a grid end means **OPEN**. Also `adequacy` (rmse/range ≤ 0.05 or rmse/σ ≤ 3, train and held-out), `status_table` (`estimate_not_for_sim` only), `correlated_family` and `downstream_point_allowed`. |
| `physx_refinement/api.py` | `estimate(sim, episodes, …)`: fit, then frozen-θ held-out evaluation, adequacy, profile (forced data only), status table, family, mapping suggestions, and the downstream verdict. `run_forceartgs_physx_estimator` keeps the spec's signature, adding `python`, `profile` and `prior_rho_range`. `PhysXIsaacRollout` raises `NotImplementedError` (native PhysX is not faked). A scale prior (`scale_prior_weight > 0`) sets the absolute scale to `PRIOR_ONLY`. |
| `physx_refinement/g9_adapter.py` | Adds `interactions_from_segments`, which uses the **same classifier and options** as the G9 free/forced stage. FREE_CONFIRMED maps to free (`contact_independently_absent`); FORCED_MEASURED maps to forced (`contact_confirmed`, the canonical object-side Q on native 50 Hz rows, unchanged). UNKNOWN and FREE_HYPOTHESIS intervals are **excluded and listed**. Forced input requires force-audit status VERIFIED or EXPLORATORY_ASSUMPTIONS_ACCEPTED. v0 comes from a 4-sample prefix fit, labelled "same trajectory, NOT independent", with σ_v0. |
| `physx_refinement/real_data.py` | Real run, append-only under `<run>/dynamics/g9/physx_refinement/vNNN/`. |
| `physx_refinement/cli.py` (patch CLI) | Now goes through `estimate()`; accepts optional `physics.python`, `profile` and `prior_rho_range` in the config JSON. |
| `forceartgs_g9/cli.py` | New `physx-refinement` subcommand (additive). |
| `configs/g9_real/physx_refinement.yaml` | Every option, commented. |
| `pipelines/g9_physx_refinement_all_real.sh` | All five scenes; env `QUICK`, `SCENES`, `OUT_BASE`. |

**Scale convention on real data.** The URDF carries the G9 solid-hull geometry at a reference density ρ₀ = √(I_lo·I_hi)/M₁, the geometric mean of the existing size prior `_prior_for_sim`; `rho_scale` is relative to it. Free-only fits hold ρ = 1 and armature = 0 (the gauge) and report the correlated family over the prior range, with c/I_eff and Td/I_eff held fixed. Forced fits search ρ ∈ [0.1, 10] and are profiled.

**Outputs per scene version:**
- `resolved_config.yaml`, `source_hashes.json` (including the record and URDF hashes);
- `regime_evidence.json`, `physx_episodes.json` (`forceartgs.g9.physx_episodes/v1`);
- `model_prior_reference.urdf`, `urdf_info.json`;
- `physx_refinement.json` with the fit, loss per episode, model adequacy, scale profile, status table, correlated family, native-property suggestions, downstream verdict, blockers, `simulation_export_allowed: false`, and **no `value` field anywhere**;
- `plots/rollouts.png`, `plots/scale_profile.png`;
- `summary.md`, `STATUS.json`.

**Blockers recorded automatically:**
- no independent (held-out) episode;
- exploratory force audit;
- gravity declared rather than verified;
- the declared pulse audit (`--pulse-rows`, e.g. fridge rows 229–232), when it is not IMPULSE_CONSISTENT. Nothing is sign-flipped.

## 4. Tests

| File | Engine | Tests | Covers |
|---|---|---|---|
| `test_g9_physx_loss.py`, `_contracts.py`, `_g9_adapter.py` (patch) | no | 15 | as shipped; unchanged, still passing |
| `test_g9_physx_real_adapter.py` | no | 8 | URDF pose-at-zero reproduces G9 gravity and I_h; prismatic case and refusals; segment bridge (hypothesis/UNKNOWN excluded, Q values, sign and rows untouched, prefix state, contradicted audit excluded); fixed dimensions, warm start, failure accounting; two force levels bound the scale while one stays OPEN; inadequate model labelled and refused; scale prior gives PRIOR_ONLY; native adapter refused |
| `test_g9_physx_engine.py` | **yes** (worker) | 17 | hidden defaults zeroed and read back; pulse Δv = J/I at ρ ∈ {1, 2.5} and armature 0.2; input sign and ZOH timing (discrete law); tilted-hinge gravity against G9 `RigidBodyGravity` (two tilts, q_ref ≠ 0); vertical hinge (zero gravity torque, nonzero inertia); prismatic gravity; free decay at 4 (λ, γ) pairs; **in-engine scale gauge** (ρ, c, Td) ~ k·(ρ, c, Td); repeatability; refusals; end-to-end forced recovery (ρ within 10%, λ and γ within 15%, held-out adequate, profile bounded and containing the truth, width < 2×) with status still DIAGNOSTIC_ONLY; free-only gauge, family consistency, API files and downstream refusal |

All 40 pass. The engine file takes about 170 s, of which the end-to-end test is about 110 s.

Not covered, because no runtime exists for them:
- native PhysX per-axis friction;
- static breakaway versus dynamic sliding;
- CPU/GPU agreement.

## 5. Real data: quick smoke run only

Run: `QUICK=1` (DE maxiter 3, population 4, profile 5×2), with output to the scratchpad. The full run was not executed here; §6 has the command.

| Scene | Result | Why |
|---|---|---|
| red fridge | NO_ADMISSIBLE_EPISODE | Its only motion interval is `FREE_HYPOTHESIS:measured_near_zero_effort_in_contact` (rows 235–296). Under G10 it is UNKNOWN and excluded. There is no FORCED_MEASURED moving interval on the accepted axis. |
| orange suitcase | OK; 1 forced training episode (24 samples), no held-out | At the quick budget the fit is MODEL_INADEQUATE (rmse/σ 158) and the profile is OPEN at ρ = 0.1. The quick budget is too small to judge the fit. |
| black box | OK; 1 forced episode (7 samples) | The free hypothesis is excluded. Quick fit MODEL_INADEQUATE; profile OPEN. |
| robot dog | KINEMATICS_NOT_ACCEPTED | nothing simulated |
| white drawer | DATA_UNAVAILABLE | nothing simulated |

What the full run can and cannot show:
- With one forced episode per scene and no independent demonstration, no real scene can reach better than DIAGNOSTIC_ONLY_NOT_IDENTIFIED, whatever the budget.
- The full run can say whether the PhysX rollout of the measured Q is adequate at all, and whether the ρ profile closes.
- The engine test in which one force level leaves ρ open applies directly to these one-direction pushes.

## 6. Commands (not launched here)

```bash
cd ~/ForceArtGS
# all scenes, full budget, append-only <run>/dynamics/g9/physx_refinement/vNNN (~10-20 min per scene with a forced episode)
nohup bash pipelines/g9_physx_refinement_all_real.sh \
  > "/media/samuelemara/Extreme SSD/Samuele/results_23_11_26/g9_physx_refinement.nohup.log" 2>&1 &
# progress / summary
tail -f "/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware/g9_physx_refinement_batch.log"
cat "/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware/"*/dynamics/g9/physx_refinement/v*/summary.md
# tests (pytest is not in freeartgs: a scratch copy from parttwin_sim on PYTHONPATH)
PYTHONPATH=<pytest_site> ~/miniconda3/envs/freeartgs/bin/python -m pytest -q tests/test_g9_physx*.py
```
