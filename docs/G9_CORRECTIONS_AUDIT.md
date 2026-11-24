# G9 corrections and multimodal robot-record integration: audit (2026-11-20)

- **Spec:** `FORCEARTGS_G9_CORRECTIONS_CODEX_20261120.md`
- **Base:** `0210948`, published `main` / `multimodal-tcp-contact`. Uncommitted diff described below.
- **Pinned dependency:** ForceArtGS-Physics `f382431`, untouched.
- **PArtTwin1D:** `origin/main` = `eaa2434` (clean history, 0 Claude trailers); pull request #1 still pins `b67863a`.
  The shared builder (`src/parttwin1d/assets/sim_export/`, `cli/build_asset.py`, `tests/test_handoff_build_asset.py`)
  is **still untracked** and local `main` is 8 commits ahead. No parallel builder was written, and no simulator
  writer was edited.

## 1. Changed files and call graph

**New:**

| file | role |
|---|---|
| `estimotion_2part/joint_discovery/force_convention.py` | the ONE sign / rotation contract: `object_side_sign` (explicit multiplier in {−1, +1}, agreeing with `recorded_quantity` / `force_acting_on`, raises otherwise); `validate_rotation`; typed stages `SensorCorrForce → ObjectBaseForce` (`sensor_corr_to_object_base` refuses anything else); `generalized_effort` |
| `estimotion_2part/joint_discovery/frontend_robot_record.py` | exact `source_frame_index` join of the canonical record to the frontend frame map, TCP row-identity check, fingerprint |
| `estimotion_2part/joint_discovery/frontend_ablation.py` | RGB-D / +TCP / +TCP + conditional force runs on identical held-out blocks; never promotes |
| `forceartgs_g9/validation/impulse_coast.py` | pulse vs coast on native 50 Hz rows (ZOH J_Q, velocity fits away from the pulse, conditional M, coast ratios, sensitivities, verdicts, plot) |
| `forceartgs_g9/validation/load_models.py` | A / B / C / D load-model comparison, LR tests, load-amplitude CI, alternatives |
| `tests/test_g9_impulse_coast.py` | 9 tests |

**Modified:**

| file | change |
|---|---|
| `wrench_physics.py`, `wrench_calibration.py` | the sign comes from `force_convention.sign_from_recorded_quantity` (same mapping, one implementation) |
| `robot_record.py` | sign from explicit metadata only (no `-1` default); rotations validated; stages `F_sensor_raw / F_sensor_corr / F_object_base` + `force_conversion` provenance (sign_source, metadata sha256, rotation_source, conversion_version); no-double-tare check (raw − export tare = corr, else error); post-contact window → `THRESHOLD_CONDITIONED_DIAGNOSTIC`; operator `contact_free_windows` → independent tare verification; positive work on contact-confirmed, force-valid rows only (diagnostic) |
| `multimodal_frontend.py`, `cli.py` | opt-in `robot_record` / `force_policy` (constructor + `--force-export --force-policy`), and `multimodal-ablation`. Certified default: no config field added, no hash change, same code path |
| `forceartgs_g9/adapters/forceartgs_g9.py` | tare gate from independent evidence only; symmetric scenarios (TCP lever / tangent / axial for R only; hinge axis tilt ±1σ in two directions; hinge-line offset R only, from `joint_selection_v2.uncertainty`, which was previously not read); `record_context` for the impulse analysis |
| `forceartgs_g9/validation/general_identifiability.py` | `ci95_statistical` (STATISTICAL) separate from `sensitivity_scenarios` (DECLARED_SCENARIO, per family ±, worst single family, linear worst-case sum), never in quadrature; `statistically_resolved_in_this_model`, `scenario_sensitive`; derived ratios with scenario sensitivity |
| `forceartgs_g9/core.py` | impulse / coast and load-model reports; `verdicts()`: `conditionally_interpretable`, `physically_interpretable_validated` + blockers |
| `forceartgs_g9/estimators/model_spec.py`, `export/dynamics_v2.py` | joint-space basis labelled `effective_configuration_load`, `physical_gravity: false`; per-parameter verdict fields |
| `configs/g9_real/*.yaml` | scenarios labelled; ±5 % scale = UNVALIDATED stress test; `load_model_comparison: true` |
| `tools/g9_exploratory_summary.py` | pulse / coast SNR, statistical CI vs scenario envelope, qualified verdict text |

**Call graph (G9 exploratory):**

```
build_robot_record -> object_side_sign + validate_rotation + sensor_corr_to_object_base   (sign once)
  -> record_force_audit (gates) -> record_interactions (contact + known-zero windows)
  -> record_perturbations (DECLARED_SCENARIO families)
  -> run_g9: fit -> diagnose_general (STATISTICAL vs scenarios) -> adequacy
     -> impulse_coast.analyze (native rows) -> load_model_comparison -> verdicts -> documents
```

**Frontend:**

```
multimodal-refine [--force-export --force-policy] -> Frontend(robot_record=, force_policy=)
  -> align_record_to_frame_map -> exploratory: conditional f_obj (selector sees ABSTAIN)
```

## 2. Input semantics and provenance

- **Force path:** `F_sensor_raw` (archive) → `F_sensor_corr` (export, tare removed, reused) →
  `F_object_base = s R_B,EE R_EE,S F_corr`, s = −1 from `metadata/contact_force_convention.json` (all 118 converted
  datasets declare −1 / `contact_reaction_force_on_robot`; the 7 entries without the file now raise).
  Then `Q_P = a·F`, `Q_R = a·((p − o) × F)`.
- **Regression:** the fridge `F_object_base_N` is bitwise identical to `136c462` (max difference 0.0). The no-double-tare
  residual is 5·10⁻⁶ N, the CSV rounding.
- **Clock:** recorded timestamps for the force rows, `source_frame_index / 50` for q (row-exact join; recorded vs row
  clock ≤ 2 ms); the hardware exposure offset is NOT_ESTABLISHED (lag scenarios).

## 3. Before / after on the real scenes (`results_19_11_26`, new versions v006–v011)

| scene | joint | force sign | pulse SNR / coast SNR / impulse | adequacy (best model) | statistical CI (example) | scenario envelope | verdict |
|---|---|---|---|---|---|---|---|
| red fridge | IDENTIFIED_REVOLUTE (not export-ready) | s = −1 declared; positive work INCONCLUSIVE | **25.7 / 1.26** (before: one "SNR 1.15"); main pulse rows 228–234, J_Q = 0.0051 ± 0.0009 N·m·s, Δq′ = −0.25 ± 0.08 rad/s → **IMPULSE_UNRESOLVED**: oscillatory impact transient (net/abs 0.32, sign change, aliased at 50 Hz); J and Δq′ have opposite signs (reported, sign not changed) | PASS with an effective load or a bias (χ²/dof 2.4–2.7); FAIL without (~60) | c 0.16 [0.034, 0.29] (no-static + load) | c ∈ [0.14, 0.18] (worst single family) | not statistically resolved / scenario-sensitive → **not identifiable** |
| orange suitcase | TYPE_UNOBSERVABLE (prismatic working model) | DATA_CONSISTENT (positive work 1.00) | 115 / – / IMPULSE_UNRESOLVED: a sustained push (J = −4.09 N·s); the q clip ends before a post-push velocity | viscous-only **FAIL** (χ²/dof 4.0); + constant load PASS | viscous-only c 20.7 [18.8, 22.7]; with load c 135 [12, 258] | c ∈ [19.7, 21.8] | "viscous-only conditional estimate; model adequacy FAIL; working prismatic joint not certified" → **not physically interpretable** |
| black box | TYPE_UNOBSERVABLE (prismatic working model) | positive work INCONCLUSIVE (0.77) | 7.0 / – / IMPULSE_UNRESOLVED (no q around the push) | PASS (χ²/dof ≈ 1) | all CIs span orders of magnitude | — | **not identifiable** (constant speed, no breakaway) |

**Fridge pulse vs TCP-motion audit.** The force is not explained by tool inertia: corr(F, a_TCP) = 0.02 and the TCP
moves at a near-constant 0.06–0.07 m/s. The door's post-contact speed matches the TCP speed (≈ 0.085 m/s): the robot
drives a light, low-friction door kinematically.

**Fridge load vs bias** (§3 of the spec):

| model | χ²/dof (viscous + Coulomb base) |
|---|---|
| A | 63.7 |
| B (+ bias) | 2.67 |
| C (+ effective load) | 2.58 |
| D (both) | 2.99 |

- LR: A→B p = 2·10⁻⁴⁵; A→C p = 3·10⁻⁴⁴; B→D p = 0.53; C→D p = 1. A constant generalized effort explains the misfit as
  well as the configuration load.
- The load variation over the observed 0.33 rad is 0.043 ± 0.026 N·m (not resolved); basis condition (D) = 570.
- Alternatives for A: time offset −20 ms χ² 955 → 853; linear q trend → 731. Neither explains it alone.
- **No gravity claim** is made (effective load ≠ gravity).
- The full model with static friction gives an unstable B fit (χ²/dof 3010), recorded as a fit failure of that
  variant.

**Suitcase load vs bias:** A 5.2 → B 1.95 (p = 3.5·10⁻⁶). The viscous-only misfit is a constant generalized effort.

**Physical identification.** No real parameter is `physically_interpretable_validated`, and none is
`conditionally_interpretable`. The suitcase c is "statistically resolved in this model" only for the inadequate
viscous models.

## 4. Tests

```
~/miniconda3/envs/freeartgs/bin/python -m unittest discover -s tests -p 'test_g9_*.py'    # 68 OK (57 + 11 new)
~/miniconda3/envs/freeartgs/bin/python -m unittest discover -s tests -p 'test_joint_*.py' # 185 OK (183 + 2 new)
```

**New tests:**
- **Sign contract:** exactly once, R = I and rotated, power invariance, R / P efforts, missing / malformed /
  contradictory metadata errors, reflection rejected, a converted force refused.
- **Tare:** no duplicate subtraction, threshold-conditioned window ≠ verification, independent window accepted,
  export untouched, re-export stable, row mismatch fails.
- **Impulse:** an 80 ms pulse stays 4 samples; known M / τ recovered; sparse q → UNRESOLVED; a sign-reversed input
  is a contradiction, never flipped.
- **Load / uncertainty:** short-range load vs bias diagnostics and labels; the geometry-coupled path is unchanged;
  scenarios do not touch the statistical CI; levers only for R; derived ratios carry scenarios; a rejected model is
  never physically interpretable.
- **Frontend shared record:** same signed force as G9, row / TCP mismatch errors, the certified result is unchanged
  with a record, exploratory (correct, wrong-axis, low force) stays DIAGNOSTIC and is never export-ready.

The real exploratory reruns created new versions; nothing was overwritten.

## 5. Ablation (RGB-D / +TCP / +TCP + conditional force)

The source is `results_20_11_26/frontend_ablation/<scene>/ablation_report.json`. All variants use the same frozen
temporal blocks for the held-out objective. The certified reference is `rgbd_tcp`.

| scene | variant | joint status | export-ready | failure codes | q range | force / contact rows in fit | axis Δ vs certified |
|---|---|---|---|---|---|---|---|
| fridge | rgbd | IDENTIFIED_REVOLUTE | yes | — | 0.325 rad | — | 0.21° |
| fridge | rgbd_tcp | IDENTIFIED_REVOLUTE | no | UNVERIFIED_PRIOR_SCALE | 0.325 rad | 0 / 4 | (reference) |
| fridge | rgbd_tcp_force | IDENTIFIED_REVOLUTE | no | EXPLORATORY_FORCE_DIAGNOSTIC, UNVERIFIED_PRIOR_SCALE | 0.325 rad | 0 / 4 | 0.00° |
| suitcase | rgbd | TYPE_UNOBSERVABLE | no | UNRELIABLE_MOVING_SUPPORT, TYPE_UNOBSERVABLE | 0.172 | — | 0.75° |
| suitcase | rgbd_tcp | TYPE_UNOBSERVABLE | no | UNRELIABLE_MOVING_SUPPORT, TYPE_UNOBSERVABLE | 0.174 | 0 / 0 | (reference) |
| suitcase | rgbd_tcp_force | TYPE_UNOBSERVABLE | no | + EXPLORATORY_FORCE_DIAGNOSTIC | 0.174 | 51 / 0 | 0.03° |
| black box | rgbd | TYPE_UNOBSERVABLE | no | UNRELIABLE_MOVING_SUPPORT, TYPE_UNOBSERVABLE | 0.063 | — | 0.00° |
| black box | rgbd_tcp | TYPE_UNOBSERVABLE | no | UNRELIABLE_MOVING_SUPPORT, TYPE_UNOBSERVABLE | 0.063 | 0 / 11 | (reference) |
| black box | rgbd_tcp_force | TYPE_UNOBSERVABLE | no | + EXPLORATORY_FORCE_DIAGNOSTIC | 0.063 | 3 / 11 | 0.00° |

- **Joint decision.** Force did not change the selected hypothesis on either scene, and the exploratory held-out
  objective did not improve. The certified `rgbd_tcp` authority is kept on both.
- **Fridge.** Adding the TCP costs export readiness (UNVERIFIED_PRIOR_SCALE: the TCP prior's scale is unverified). It
  does not change the joint.
- **Suitcase.** It stays TYPE_UNOBSERVABLE in every variant. The 51 exploratory force rows do not resolve the type.
- **Black box.** It stays TYPE_UNOBSERVABLE in every variant (q range 0.063). It is declared revolute by the operator
  (`--declared-joint-type`), never selected from the data.

## 6. Unresolved

- **Fridge:** inertia needs a cleanly resolved push (higher-rate force or a slower, sustained push).
- **Suitcase / black box:** need an accepted joint type and q beyond the push.
- **Physical claims:** need held-out independent episodes.
- **Gravity:** a real gravity torque needs the geometry-coupled model with a measured gravity registration and a
  mass-model mesh.
- **Isaac:** PhysX parity not run.
- **PArtTwin1D builder:** commit and push.

## 7. Update: declared vertical (operator 2026-11-20: "the dataset z axis is the real z axis")

Gravity g = (0, 0, −9.81) m/s² in base_link is **DECLARED** by the operator
(`configs/g9_real/gravity_base_link_declared.json`). It is not a measured registration, so it is used under the
exploratory policy only.

**New parameter source `joint_space_gravity_coupled`** (no mesh needed):
- **prismatic:** G = −M g·a (single rigid body: effective mass = moving mass), no free gravity parameter;
- **revolute:** G = μ g·(a × R(q) u), with one mass moment μ = m r (kg·m), u = hinge → moving-support centroid, and
  the check M ≥ μ r_c.
- Tests: `test_g9_impulse_coast.DeclaredGravityCoupled`. Synthetic M and μ are recovered within 3 %, and declared
  gravity is refused without the opt-in.

**`--declared-joint-type` / `--declaration-note`:** an operator-declared joint for a TYPE_UNOBSERVABLE scene, labelled
OPERATOR_DECLARED_KINEMATICS (conditional, never certified). Used for the black box, which the operator declared
revolute on 2026-11-19.

**Joint axes relative to the true vertical:**

| scene / branch | tilt from vertical |
|---|---|
| fridge hinge | 6.0° |
| black box revolute (declared) | 85.3° (horizontal hinge) |
| black box prismatic | 18.7° |
| suitcase prismatic working model | 88.6° (horizontal) |

**Results** (`dynamics/g9/v012–v014`):

| scene | model | adequacy | M | c | gravity | verdict |
|---|---|---|---|---|---|---|
| fridge (IDENTIFIED_REVOLUTE) | viscous + declared gravity | PASS, 1.3σ | 0.040 [0.006, 0.073] kg·m² | 0.16 [0.044, 0.28] N·m·s/rad | μ = 0.134 [0.018, 0.25] kg·m (≈ 0.5 kg at 0.28 m); M ≥ μ r_c holds (0.0395 ≥ 0.037) | CIs exclude 0, rel. std 37–44 % > 10 %: physically consistent CONDITIONAL estimate, not identified |
| fridge | + Coulomb + static | FAIL (χ²/dof 3.2) | — | — | — | rejected variant |
| black box (revolute, declared) | all | PASS, 0.84σ | 180 (unbounded) | → 0 | μ → 0 | not identifiable (constant speed, no breakaway) |
| suitcase (prismatic working) | all | FAIL (runs test z = −3.7) | 8.4 [5.9, 10.8] kg | → 0 | G_P = −2.0 N | rejected; working joint not certified |

**Effect on the fridge.** With the declared vertical, the configuration load is physical and has one parameter
instead of two. The previous effective-load coefficients had CIs crossing 0; μ is now positive at about 2.3σ.

## 8. Gravity projection / mass geometry / reference pose (2026-11-20)

The full description is in `G9_GENERAL_PINN_AKF_IMPLEMENTATION_REPORT.md` §9. The quantitative checks:

| check | fridge (smoke v022; full budget v023–v026) | suitcase (smoke v018; full budget v019–v022) |
|---|---|---|
| reference pose | PASS, q_ref = 0 (RGB-D 90 ↔ row 180; axis 0°, hinge 0 m) | PASS, q_ref = 0 |
| joint tilt from vertical / \|g×a\| | 6.03° / 1.03 m/s² (R) | 88.6° / g·a = 0.24 m/s² (P, working model) |
| m₁ solid / shell | 0.0609 m³ / 0.941 m² | 0.0142 m³ / 0.347 m² |
| \|r⊥\| solid / shell | 0.180 / 0.181 m | — (prismatic) |
| G₁ amplitude vs analytic m₁\|g×a\|ℓ | equal (to 1e-15) | equal |
| implied mass, full budget (PINN / EKF) | 0.088 / 0.133 kg (solid v023); 0.083 / 0.0046 kg (shell v024); rollout ρ → 0: **not physical** | 8.3 / 7.6 kg: an axial-load fit |
| gravity amplitude, geometry scenarios | −36 % / +51 % around the accepted value | 0.86–17.5 N (accepted 2.0 N) |
| vertical-hinge alternative | G ≡ 0. Solid v023: replay 51–80σ vs 109–270σ accepted (fitted ρ). Shell v024: no gain (36 / 57 vs 50 / 63). Upstream residual NOT_RUN; never substituted | n/a |
| method parity (M, G, dr/dρ) | PASS (≤ 1e-12) | PASS |
| impulse rows 229–232 | SIGN_OR_TIMING_INCONCLUSIVE: J_Q +6.4e-3 vs Δq̇ −0.234; gauge consistent; ±40 ms lag no flip; tool mass to cancel −24 kg; no sign changed | no declared pulse |

**Status.** Neither scene yields a physical identification:
- the geometry is TENTATIVE;
- the calibration is DECLARED;
- there is no held-out episode;
- the suitcase joint is not accepted.
