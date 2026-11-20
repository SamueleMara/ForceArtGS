# G9 general dynamics, geometry-constrained inertia and PArtTwin1D physics export: implementation report (2026-11-20)

- **Spec:** `FORCEARTGS_G9_GENERAL_DYNAMICS_FULL_INERTIA_PARTTWIN_CODEX.md`
- **Audit:** `docs/G9_DYNAMICS_REPO_AUDIT.md`

**State:** ForceArtGS `628e3ee`, G9 uncommitted. ForceArtGS-Physics pinned at `f382431` and untouched. PArtTwin1D
builder is local and untracked.

**Claim.** On synthetic, fully instrumented R and P objects, the pipeline:
- identifies the general model;
- reconstructs mass, COM and the full 3×3 tensor;
- validates on withheld episodes;
- builds URDF, USD and MJCF through the single shared builder, after which MuJoCo reproduces withheld-force rollouts.

**No real scene has physical dynamics.** Every real scene ends in an explicit BLOCKED report (§6).

## 1. Code

Package `forceartgs_g9/` is a plugin over the pinned Physics library: no frozen code is edited, and nothing from the
library is copied.

| module | content |
|---|---|
| `physics_lib.py` | locate and pin `~/ForceArtGS-Physics` (refuses another HEAD or a dirty tree) |
| `dynamics/general_1dof.py` | `GeneralTerms` switches; `RigidBodyGravity` (U = −m g·c(q), G = dU/dq); `sim_params` (a disabled term accepts no value); event-driven exponential integrator with Karnopp stick/slip, analytic stop events and a second-order Heun step for G(q) |
| `dynamics/mass_properties.py` | validated `TriMesh`; exact unit-density solid (volume) and shell (area) moments; tensor checks |
| `dynamics/geometry_inertia.py` | α coupling (M = α I_a,1, G = α G_1); reconstruction + Monte Carlo; reflected-inertia declaration; collision-proxy policy; user-inertial conflict; mesh / Gaussian alignment |
| `estimators/model_spec.py` | θ ↔ rollout; gravity-registration gate |
| `estimators/general_integral_ls.py` | weak double-integral initializer (exact ZOH force integral, moving segments only) |
| `estimators/general_constrained.py` | `general_rollout_fit`: robust bounded least squares through the hybrid rollout on raw q timestamps; shared parameters plus per-episode (q0, v0); Fs by profile likelihood |
| `estimators/frozen_baselines.py` | frozen `api.identify` as *viscous_1dof baselines* |
| `validation/*` | general identifiability (Fisher + autocorrelation inflation + systematic sweeps + structural confounds), adequacy, held-out prediction + leakage, engine term matrix + MuJoCo / SAPIEN parity |
| `adapters/forceartgs_g9.py` | accepted authority, 9-gate force audit, contact-segment efforts (unknown contact excluded) |
| `export/dynamics_v2.py`, `export/final_build.py` | `physics_parameters.json`, `dynamics_handoff.yaml` (`parttwin.dynamics/v2` block), translation into the builder's physics spec v1, final build + parity |
| `core.py`, `stage.py`, `cli.py`, `config.py` | fail-closed sequence, versioned `<run>/dynamics/g9/vNNN`, CLI, strict config |

**Controller:**
- **Stages:** `… export, dynamics, final_physics_export`. Both new stages are opt-in through `--dynamics-config`.
- **Flags:** `--physics-library`, `--force-calibration`, `--mass-model-spec`, `--dynamics-episodes`.
- **Batch:** `pipelines/g9_dynamics_real_scenes.sh`.
- **Config:** `configs/dynamics_general.example.yaml`.

**Bugs found by the gates and fixed:**
1. First-order gravity integration was off by 5.5·10⁻³ rad at a 2 ms step; Heun brings it to 4·10⁻⁶.
2. The piecewise-constant likelihood in Fs gave zero gradients and a wrong basin; it is now profiled first.
3. The correlation rule wrongly failed parameters whose marginal std was small; correlation is now a note.
4. The constant-speed detector is now a straight-line test on q.

## 2. Tests

| suite | result |
|---|---|
| `test_g9_*.py` (Gates A 13, B 13, C 15, D 12) | **53 OK**, 186 s |
| existing `test_joint_*.py` | **183 OK**, 820 s (unchanged) |
| ForceArtGS-Physics own suite | 75 passed, 66 skipped (external data); C15 re-checks the pin, a clean tree and a deterministic IntegralLS after the G9 runs |

## 3. Synthetic recovery and parity

Each case uses 2 training and 2 withheld episodes: σ_q 2 mrad (R) / 0.5 mm (P), 30 Hz q, 100 Hz force, ramped
pushes, releases and reversals.

| object | model | α (= mass) err | c err | Fc err | Fs err | held-out RMSE / σ | MuJoCo | SAPIEN as loaded | SAPIEN, PhysX friction zeroed (diag.) |
|---|---|---|---|---|---|---|---|---|---|
| R flap | full | −0.03% | −0.84% | +0.54% | −1.24% | 1.12 | PASS 0.87 mrad | FAIL 98 mrad | 1.7 mrad |
| R flap | no static | +0.00% | +1.20% | −0.15% | — | 1.03 | PASS 0.56 mrad | FAIL 45 mrad | 0.9 mrad |
| R flap | viscous + gravity | +0.05% | +0.20% | — | — | 1.03 | PASS 0.03 mrad | FAIL 81 mrad | 0.9 mrad |
| P drawer | full | −0.00% | +0.04% | −0.04% | +0.66% | 1.01 | PASS 0.29 mm | FAIL 82 mm | 15 mm |
| P drawer | no static | +0.04% | +0.17% | −0.10% | — | 1.02 | PASS 0.15 mm | FAIL 29 mm | 9.1 mm |
| P drawer | viscous + gravity | −0.02% | −0.03% | — | — | 1.03 | PASS 0.01 mm | FAIL 200 mm | 9.1 mm |

- **Gates:** all six cases reach PRODUCTION_PHYSICS_SPEC_READY; identifiability, adequacy and held-out all PASS.
- **Builder:** URDF, MJCF and USD PASS, and the visual PLY payloads are identical to the G8 handoff.
- **Final claim:** NOMINAL_DYNAMICS_EXPORT, never FULL_PHYSICS_READY (§4).

## 4. Engine findings

- **SAPIEN 3.0.3.** Its URDF loader applies a PhysX joint friction coefficient of 0.05 regardless of
  `<dynamics friction>`. That coefficient is dimensionless and proportional to the normal force, not a Coulomb
  torque. With it zeroed (diagnostic only, asset unchanged), inertia, damping and gravity match. As-loaded parity
  therefore FAILs, and this is reported.
- **MuJoCo:** damping and `frictionloss` read back exactly; withheld-force rollouts PASS.
- **Isaac / USD:** USD authoring and readback PASS. Coulomb is NOT_REPRESENTED, and the PhysX runtime is NOT_RUN
  (no Isaac installed).
- **Static Fs > Fc:** NOT_REPRESENTED in every engine. The bias is never exported. Gravity comes from mass and COM
  only; gravity-OFF models give GRAVITY_CONFIGURATION_MISMATCH unless the tested bound is negligible.

**PArtTwin v1 schema gaps** (recorded in the generated files):
1. A static-link inertial is mandatory even for a fixed root, and G9 never fabricates one.
2. There is one joint provenance field, so measured stops get relabelled "estimated".
3. There is no Fs field.
4. One `coulomb_friction` scalar has engine-specific meaning.

## 5. Real scenes (`results_19_11_26`, `<run>/dynamics/g9/v000`)

| scene | status | blockers | excitation (data only) |
|---|---|---|---|
| red fridge | BLOCKED | kinematics not export-ready (UNVERIFIED_PRIOR_SCALE); force gates UNVERIFIED (rotation, sign, TCP, tare, tool gravity, clock, registration); no mass mesh | one direction, 1 breakaway, 0.33 rad; Fc collinear with tare |
| black box | BLOCKED | TYPE_UNOBSERVABLE; force; mesh | constant speed (no inertia), no breakaway (no Fs), c ~ Fc |
| orange suitcase | BLOCKED | TYPE_UNOBSERVABLE; force; mesh | one direction; Fc ~ tare |
| robot dog | BLOCKED | MODEL_INCONSISTENT (no accepted authority) | — |
| white drawer | BLOCKED | TYPE_UNOBSERVABLE; no robot archive | — |

No fit was run on unverified forces (`on_missing_calibration: abstain`). With a ±20 ms clock uncertainty, damping is
unidentifiable even on synthetic data (C09). The current single slow pushes would leave inertia and Fc vs tare weak
even if calibrated. The 1 + 10 protocol needs pauses, speed changes and reversals.

## 6. To unblock a real object (measured inputs, not relaxed settings)

| input | content |
|---|---|
| `--force-calibration` | `{gates: {<gate>: {status: PASS, method, evidence_sha256}}}` for each failing gate |
| `model.gravity` | vector + `calibration_file` with `status: MEASURED` |
| `--mass-model-spec` | validated canonical moving-part mesh, `mode`, `reflected_actuator_inertia` |
| `export.mechanical_spec` | collision meshes, physical stops, velocity / effort limits, measured static inertial, registration |
| `--dynamics-episodes` | independent held-out runs with `initial_state: declared_rest` |

## 7. Not done

- **Geometry:** no watertight mesh extraction from RGB-D; `segmented_material` has the rank check only.
- **Estimators:** no `general_akf`, `general_pinn` or `general_backprop`.
- **Benchmarks:** the ForceSAPIEN A3 → B1–B10 benchmark was not rerun (bundle missing).
- **Engine parity:** no Isaac runtime parity.
- **Reproducibility:** the PArtTwin builder is uncommitted.

## 8. Update (2026-11-20, later): G9 connected to the existing robot preprocessing

The earlier §5 / §6 statement that "every real scene is BLOCKED" conflated two things:
- data that **is** extracted and synchronised;
- the stricter certification gates, which refuse to call those data VERIFIED.

G9 now reuses the existing preprocessing. Nothing is re-tared, re-calibrated or re-synchronised.

**Canonical robot record:** `estimotion_2part/joint_discovery/robot_record.py`, schema `forceartgs.robot_record/v1`,
saved as `<g9 dir>/robot_record/`. It is built from:
- `results_19_11_26/contact_annotations/<scene>/`: the tare-corrected force at the native 50 Hz rows, recorded
  timestamps, baseline tare and σ, per-row contact state, and the EE-orientation bound;
- the robot archive (EE rotation, TCP);
- `contact_force_convention.json` (object-side multiplier −1), `robot_tool.json` (TCP),
  `robot_base_registration.json`, and the contact segments.

The force on the object is `F_obj = −R_base_ee · R_ee_sensor · F_corr`, applied at the TCP, on the q clock
(source_frame_index / 50; the recorded timestamps differ by ≤ 2 ms).

**Interaction windows:** contact-confirmed rows (operator segments) plus rows whose measured force is at the noise
floor (`absent`, i.e. known ≈ zero effort). Uncertain rows outside the segments split a window.

**Policy** (`pipeline.force_policy`):
- `certified` (default): unchanged.
- `exploratory`: every DECLARED / ESTIMATED quantity is used as stated and listed; nothing is marked VERIFIED, and the
  result is `EXPLORATORY_ESTIMATE`, never production-exported.

**Status categories, replacing one generic BLOCKED:**
`DATA_UNAVAILABLE`, `KINEMATICS_NOT_ACCEPTED`, `CALIBRATION_ASSUMPTION_UNVERIFIED`, `CALIBRATION_ESTIMATED_FROM_DATA`,
`DYNAMICS_NOT_IDENTIFIABLE`, `SIMULATION_ASSET_INCOMPLETE`. A missing mesh is reported in the last category and never
blocks joint-space fitting.

**New data checks:**
- **Positive work:** checks the declared sign and axes, since a push does positive work. It supports a declaration
  and never verifies it.
- **Tare stability** in a second, separated contact-free window after the push.
- **Effort SNR:** generalized-effort RMS while moving, divided by the force σ projected on the joint.
- **Scale-free combinations:** decay time M/c and Coulomb deceleration Fc/M, by the delta method.

**Numerical safeguard:** the nuisance initial state is bounded by the data. This removed a degenerate
M → 0 / v0 → 10⁷ solution.

**Configs:** `configs/g9_real/exploratory_{full,no_gravity,no_static,viscous_coulomb,viscous_only}.yaml` (joint-space
only). Driver: `pipelines/g9_real_exploratory.sh`. Summary: `tools/g9_exploratory_summary.py`, giving
`g9_exploratory_summary.{json,md}` in the run root.

**Systematic sweeps:**
- lag ± 10 ms (half a row; exact row join);
- force scale ± 5 % (no calibration evidence);
- contact lever ± 1.5 cm;
- the second-window tare residual.

**Tests:** `tests/test_g9_robot_record.py` (4) — the record reuses the export, certified vs exploratory, positive
work catches a flipped sign, windows, categories, no mesh requirement. G9 total: **57 OK**.

### 8.1 Per-gate translation (fridge; the same structure for every scene)

| gate | existing measurement / preprocessing | status | extra verification for CERTIFIED export | needed for joint-space estimation? |
|---|---|---|---|---|
| sensor rotation | declared R_ee_sensor = I (operator) | DECLARED; positive work 0.71: INCONCLUSIVE (suitcase 1.00: DATA_CONSISTENT) | reference push of known direction or multi-pose test | no — assumption, sign errors are caught by positive work |
| force sign | recorded reaction on robot, × −1 (operator 2026-11-13) | DECLARED, same check | reference push | no |
| tare | baseline 0.5 s: tare (0.0001, 0.0076, 0.0088) N, σ (0.059, 0.049, 0.035) N | ESTIMATED_STABLE; 2nd window (133 rows): (−0.044, −0.005, −0.006) N | documented two-window procedure | no — 2nd-window residual swept as bias |
| tool gravity | EE rotation ≤ 0.055° | ESTIMATED_BOUNDED: ≤ 0.0095 N per kg of tool (suitcase 0.093 N/kg: above σ, so unverified) | tool + sensor mass / COM | no |
| time sync | exact source_frame_index join; recorded timestamps 50 Hz ± 1 ms | ROW_ALIGNED; hardware exposure offset NOT_ESTABLISHED | measured camera-robot offset | no — swept ± 10 ms |
| contact point | TCP 0.1876 m (user contact calibration) | DECLARED | TCP-to-surface verification | no — lever swept ± 1.5 cm |
| camera–robot registration | operator-applied base registration | DECLARED (UNVERIFIED in metadata) | measured registration | no — enters via axis / lever |
| contact intervals | force-peak + visual estimator, operator-validated | DECLARED | independent contact sensing | **yes** (defines the windows) — available |
| torque origin | unspecified | NOT_NEEDED | — | no (point-force model) |

### 8.2 Real exploratory estimates (joint-space; conditional on §8.1; no independent held-out episode exists)

**Fridge — kinematics IDENTIFIED_REVOLUTE.** Window: rows 220–296.
- **The effort is at the sensor noise floor:** SNR 1.15 *(superseded, see §9: this mixed the pulse and the coast; pulse SNR 25.7, coast SNR 1.26)*. The door moves 0.33 rad after an impulsive 2.6 N contact
  at rows 229–232, then coasts with |F| 0.07–0.4 N.
- **Fit:** the full and no-static models fit at 1.3 σ (adequacy PASS); the models without the configuration load
  fail (χ²/dof 47).
- **No parameter is identifiable.** τ = M/c = 0.25 ± 0.07 s (27 % rel. std): **not identified**. The effective configuration load
  (gB 0.13 N·m, CI includes 0) is NOT evidence of gravity: a constant bias explains the same misfit (§9).
- **Frozen viscous baselines** (IntegralLS 0.002 kg·m², 0.027 N·m·s; Backprop 0.012, 0.025) are also
  "NOT separately identifiable".
- **Diagnosis:** this demonstration does not carry enough force information. Calibration is not the limit.

**Orange suitecase — kinematics TYPE_UNOBSERVABLE; prismatic working model, KINEMATICS_NOT_ACCEPTED.** Window: rows
124–228.
- **Data quality:** SNR 62, positive work 1.00.
- **Viscous-only fit:** c = 20.7 [17.9, 23.6] N·s/m (statistically resolved *in this rejected model* only; not physically interpretable), M = 18.1 [6.8, 29.5] kg (not identifiable),
  adequacy FAIL (χ²/dof 4.0).
- **With the effective constant load:** rms 1.2 σ, adequacy PASS, but c, Fc and the constant load are confounded
  (one direction).
- **Frozen baselines agree:** IntegralLS 13.2 kg / 20.8 N·s/m, Backprop 17.6 / 20.5, both CRLB-identifiable.
- **Caveat:** these are effective parameters of a working model whose joint type is unobserved, not the lid's
  physics.

**Black box — TYPE_UNOBSERVABLE (working model prismatic).** Window: rows 140–256. SNR 2.9, constant speed, one
direction: no parameter is identifiable in any variant.

**Robot dog:** DATA_UNAVAILABLE (no accepted joint authority, MODEL_INCONSISTENT).
**White drawer:** DATA_UNAVAILABLE (no robot archive).

**Notes on the outputs:**
- Fridge `dynamics/g9/v001` is the pre-safeguard degenerate fit, superseded by v002–v006.
- The certified-policy runs from earlier (v000) remain as the certification record.

## 9. Corrections (2026-11-20, `FORCEARTGS_G9_CORRECTIONS_CODEX_20261120.md`)

Details, before / after tables and tests: `docs/G9_CORRECTIONS_AUDIT.md`.

**What changed:**
- **Sign:** one contract (`force_convention.py`), used by G9 and the frontend. F_obj = s R F_corr with an EXPLICIT s
  from metadata; missing / malformed → error; applied once (typed stages).
- **Pulse vs coast:** evaluated separately on native rows (`impulse_coast.py`).
- **Load vs gravity:** the joint-space load is `effective_configuration_load`; candidate models A / B / C / D with LR
  tests (`load_models.py`).
- **Uncertainty:** statistical CIs separated from DECLARED scenarios (no quadrature); ±5 % force scale is an
  unvalidated stress test.
- **Verdicts:** two dimensions, `parameter_statistically_resolved_in_this_model` and
  `physically_interpretable_validated`.
- **Frontend:** consumes the same canonical record (opt-in exploratory policy; certified unchanged); R/P ablation
  driver.

**Real verdicts, by category:**

| category | content |
|---|---|
| measured | force samples, TCP poses, recorded timestamps, the q trajectories |
| conditional | every fitted dynamics number (exploratory policy) |
| rejected models | viscous-only fridge and suitcase fits (adequacy FAIL) |
| not identifiable | every fridge, suitcase and black-box physical parameter |

- **Fridge:** the main pulse is an aliased impact transient, so IMPULSE_UNRESOLVED; the coast is at the force noise
  floor.
- **Suitcase:** damping is model-dependent (20.7 vs 135 N·s/m) on an unaccepted joint.
