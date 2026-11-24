# G9 ForceTwin adaptation: identifiability-first effective joint dynamics

This report covers the spec `FORCEARTGS_G9_FORCETWIN_INSPIRED_IDENTIFICATION_CODEX_SPEC.md` (2026-11-21).

**Source.** Engelbracht et al., *ForceTwin*, arXiv:2609.21751 (2026): §III-C, Eqs. (5) and (7)–(9), §III-D, §IV–V.
We adapt its physical decomposition. Its network and optimiser are not presented as a PINN or EKF.

**Status.** Nothing was committed or pushed.

## 1. Tree audit and what was preserved

See `docs/G9_FORCETWIN_ADAPTATION_AUDIT.md`.

**Working tree.** Branch `multimodal-tcp-contact`, HEAD `9298f77`, with 33 uncommitted entries from the earlier
G9 / PINN / EKF / geometry work.

**Preserved.**
- All earlier work. Every edit is additive.
- `~/ForceArtGS-Physics`, pinned and clean.
- All old `dynamics/g9/vNNN` versions.
- The 3DGS PLYs and the joint authorities.
- The running `joint_aware_all_real.sh` / `ns-train` job.

## 2. What changed

| new / changed | role |
|---|---|
| `forceartgs_g9/forcetwin/record.py` (new) | `JointInteractionRecord` (spec §2.3), built from the existing canonical robot record. The sign is applied once, from metadata; rows are joined on `source_frame_index`; force stays on the native 50 Hz rows (ZOH) and q on the RGB-D clock. Also: effort re-projection to another joint hypothesis, the equivalent-convention reversal (a, q, τ) → (−a, −q, −τ), strict validation, and the npz/json save |
| `forceartgs_g9/forcetwin/derivatives.py` (new) | two independent q̇ / q̈ reconstructions: a weighted local polynomial with sandwich covariance, and a GCV smoothing spline with Monte Carlo uncertainty |
| `forceartgs_g9/estimators/forcetwin_linear.py` (new) | the ForceTwin-inspired bounded NNLS (`lsq_linear`, I, μ, b ≥ 0); a weak form on non-overlapping windows with an exact ZOH force integral; nested variants; a known-gravity variant tied to the same mass geometry; block bootstrap; rank / condition / correlation; coast ratios; per-parameter identifiability records (spec §4.2) |
| `forceartgs_g9/validation/impulse_balance.py` (new) | signed impulse–momentum audit (spec §3): one-sided velocities (polynomial and spline), a sensitivity table, a positive-inertia feasibility box, the onset / bracket analysis, a tool-inertia bound, contact-window work, the six decision states, and a four-panel plot |
| `forceartgs_g9/forcetwin/kinematics_hypotheses.py` (new) | accepted vs free-axis vs exactly vertical hinge, re-estimated from the original RGB-D 3-D tracks. Each hypothesis has its own q(t); held-out residuals use temporal halves |
| `forceartgs_g9/estimators/forcetwin_residual.py` (new) | structured mechanism residual g_θ(q) + c_φ(q, \|q̇\|, sgn q̇) q̇, with c_φ = softplus ≥ 0 and direction-split heads. Fitted on a frozen baseline; blocked chronological validation; ablations none / g / damping / both |
| `forceartgs_g9/forcetwin/pipeline.py`, `compare.py` (new) | orchestration into the append-only `dynamics/forcetwin_adaptation/vNNN/` layout, STATUS with separate states, simulation handoff, comparison CSV / MD |
| `forceartgs_g9/estimators/general_pinn.py` | seed selection by frozen-θ open-loop replay on the training data (`select_seed_by: replay`, the new default); both rankings are recorded |
| `forceartgs_g9/cli.py` | subcommands `fridge-audit`, `fit-minimal-dynamics [--pinn] [--residual]`, `fit-mechanism-residual`, `compare-dynamics` |
| `configs/g9_real/forcetwin_effective.yaml` (new) | exploratory joint-space effective model; the hull proxy is used only for the known-gravity hypothesis and the conditional allocation |
| `tools/g9_forcetwin_fridge_audit.py`, `tools/g9_forcetwin_comparison.py`, `pipelines/g9_forcetwin_all_real.sh` (new) | runbook entry points |
| `tests/test_g9_forcetwin.py` (new) | 13 targeted tests (§9) |

**Reused unchanged:**
- the canonical robot record (`robot_record.py`, explicit sign);
- `build_inputs`, `ModelSpec(joint_space_only)`, where (M, c, Fc) are exactly (I_eff, b, μ);
- `general_integral_ls`, `general_rollout_fit`, `general_akf_fit`, `general_pinn_fit`, `simulate_general`;
- the tentative hull proxy and `GeometryMassModel` with q_ref.

## 3. ForceTwin → G9 mapping

| ForceTwin | here | copied / extended |
|---|---|---|
| Eq. (5): τ = z^T β + r, z = [sgn q̇, q̇, q̈], β = [μ, b, I] | same, in the order β = [I_eff, μ, b] | copied (order recorded) |
| Eqs. (7)–(8): NNLS of the physical part | bounded weighted LS (`lsq_linear` bvls, ≥ 0) with heteroscedastic weights σ_τ² + (I σ_q̈)² | extended (weights; errors-in-variables stated) |
| Eq. (9): residual g(q) + c(q, \|q̇\|) q̇ with c ≥ 0, separate direction heads | same structure; softplus heads; no q̈ or τ input | copied in principle |
| training: AdamW lr 1e-3, wd 1e-3, ≤ 1200 epochs, patience 120, 75/25 **random** split | same optimiser; **blocked chronological** 75/25 split | deliberate protocol change (correlated rows) |
| §III-D: backend approximations change behaviour | engine mapping marked NOT_RUN / RUNTIME_PLUGIN_REQUIRED | copied caveat |
| — | weak (integral) form with exact ZOH, coast ratios, rank / identifiability gate, signed impulse audit, RGB-D kinematic hypotheses, PINN / EKF / rollout on the same record, 3DGS conditional allocation | our extensions |

## 4. Fridge: signed effort, impulse balance and competing joint hypotheses (v003)

**Outputs.** `<fridge run>/dynamics/forcetwin_adaptation/v003/`:
- `impulse_balance.json` and `impulse_balance.png`, with four panels:
  - object-side 3-D force, sensor |F|, tare baseline and ±2σ_F;
  - τ, ‖F‖, tangential F, the power τ q̇ and the lever arm;
  - cumulative impulse J_τ(t) with a 2σ band;
  - q at the RGB-D times, with both q̇ reconstructions and their bands.
- `kinematics_hypotheses.json`.

**Pulse, source rows 229–232.** These are force rows `source_frame_index` 229–232, which is **not** an RGB-D frame
index.
- J_τ = +6.43e-3 ± 0.66e-3 N·m·s (9.7σ).
- The force sign is −1 from metadata, applied once.

**Effort and q around the pulse.** Taken from the `JointInteractionRecord`:

| source rows | τ [N·m] | ‖F‖ [N] | per-row force state | q (RGB-D) |
|---|---|---|---|---|
| 224–228 | −0.03 … +0.002 | ≤ 0.36 | contact | −0.0001 → −0.0004 |
| **229–232** | **+0.42, +0.09, −0.10, −0.10** | 2.6, 2.0, 0.8, 0.8 | contact | −0.0008 at row 232 |
| 233–260 | −0.05 … +0.03 | 0.05–0.34 (noise level) | mostly `absent` / `uncertain` | −0.004 (236) → −0.033 (240) → −0.079 (248) → −0.161 (260) |

**What the data say.**
- **Before the pulse** the door is at rest.
- **During or just after it** the door starts to move. The bracket samples at 4.56 and 4.72 s show Δq = −3.6 mrad
  (16σ). At 12.5 Hz effective q sampling (every other 25 Hz RGB-D frame) the onset cannot be separated from the pulse.
- **After the pulse** it reaches a plateau of **−0.32 rad/s** while the measured effort is at the noise level.
- **Net measured impulse.** From pulse start to the end of the acceleration it is +0.00095 N·m·s: the pulse +0.0064
  and then −0.0055.
- **Required impulse.** Reaching that velocity needs −0.016 N·m·s even for an implausibly light door
  (I = 0.05 kg·m²), and −0.64 for I = 2 kg·m². So the measured effort explains **< 6 %**, and with the wrong sign.
- **Coast ratios.** The 13 force-absent moving rows give μ/I ≈ 0 and b/I ≈ 0 (RATIO_ONLY): the door does not
  decelerate when no force is measured.

**Decision.** `UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE` (primary). It also carries `NO_POSITIVE_INERTIA_EXPLANATION`
and `MOTION_DERIVATIVE_UNCERTAIN`. Absolute inertia: **NOT_IDENTIFIED_ABSOLUTE_INERTIA**.

**Evidence for the decision** (each item is a case in `impulse_balance.json`):
- No admissible case admits a positive inertia within 2σ with μ, b ≥ 0. The cases are:
  - velocity windows (0.2 and 0.45 s, gaps 0 / 40 / 80 ms);
  - polynomial and spline one-sided velocities;
  - q-clock offsets of ±10, ±20 and ±40 ms;
  - contact point ±1.5 cm along the tangent, radial and axial directions;
  - axis ±1σ in two directions, and hinge origin ±1σ;
  - all three RGB-D kinematic refits, each with its own q and its own projected effort.
- Contact point and axis change J by at most ±9 % and never flip its sign.
- The equivalent convention (a, q, τ) → (−a, −q, −τ) gives the identical conclusion.
- A tool-inertia explanation would need **−24 kg** (peak TCP acceleration 0.5 m/s²): not physical.
- A flipped force sign would be "consistent" (I ≈ 0.017 kg·m²). That is a counterfactual only, and still far below
  the size prior. The metadata sign was not changed.

**Why the earlier reports gave Δq̇ = −0.234.** It came from one-sided post-pulse fits that extrapolate the
subsequent, unforced motion back to the pulse boundary. Here it is localised and flagged as
`one_sided_fit_artifact`.

**Most likely mechanism.** It is not identifiable from this data. Candidates:
- the latch or door seal releasing (the 2.6 N tap at row 229 is the unlatching contact);
- a self-opening tendency;
- contact outside the sensed tool.

**Competing hinges, from the RGB-D tracks.** 487 inlier moving tracks out of 600 sampled; median residual 2 mm.

| hypothesis | tilt | held-out RMS |
|---|---|---|
| accepted axis | 6.03° | 2.41 cm |
| free-axis refit | 2.8° | 2.89 cm |
| exactly vertical | 0° | 2.53 cm |

The verdict is **VERTICAL_NOT_DISTINGUISHABLE**: the tracks cannot resolve the tilt. Each hypothesis carries its own
q(t), and the q trajectories agree to < 1 mrad. The vertical hinge is a labelled hypothesis. It is the primary for
identification only because the accepted hinge is within 10° of vertical and the tracks do not reject it. It never
replaces the joint authority.

## 5. Per-scene parameters (inertia_viscous_coulomb; full table in each `comparison.md`)

Labels: MEASURED is never used, because there is no independent episode. All values are exploratory, with declared
calibration.

| scene (version) | primary hypothesis | estimator | I_eff | μ | b | status (I / μ / b) | replay rmse/σ |
|---|---|---|---|---|---|---|---|
| fridge (v003) | vertical-hinge refit | NNLS direct | 4.1e-3 kg·m² | 9.1e-3 N·m | 0 | NOT_ID / NOT_ID / NOT_ID (no interval; impulse audit) | 875 |
| | | NNLS weak | — | — | — | UNOBSERVABLE (too few force-confirmed moving windows) | — |
| | | integral LS | 4.4e-3 | 0 | 0.023 | MODEL_INADEQUATE (I ≪ size prior 0.10–1.0) / NOT_ID / NOT_ID | 365 |
| | | rollout | 9.5e-3 | ~0 | 0.021 | MODEL_INADEQUATE / NOT_ID / NOT_ID | 377 |
| | | EKF | 1.1e-2 | 9.2e-3 | ~0 | MODEL_INADEQUATE / NOT_ID / NOT_ID (NIS 0.063) | 596 |
| suitcase (v004, prismatic working model, TYPE_UNOBSERVABLE) | accepted axis, known gravity (solid hull) | NNLS direct | 1.29 kg | 1.64 N | 0 | NOT_ID (CI [0, 2.5]) / NOT_ID / NOT_ID | 1440 |
| | | NNLS weak | 2.31 | 0.90 | 7.04 | (diagnostic) | — |
| | | integral LS | 8.21 | 0 | 0 | NOT_ID | 129 |
| | | rollout | 4.43 | 1.11 | ~0 | NOT_ID (not converged) | 155 |
| | | EKF | 5.26 | 0.91 | ~0 | MODEL_INADEQUATE (NIS 4.7, inconsistent; unaccepted kinematics) | 187 |
| black box (v003, operator-declared revolute, hinge 85° from vertical) | accepted axis, known gravity | NNLS direct | 0 | 0 | 3.36 | NOT_ID (at bound; condition 667) | — |
| | | integral LS | 8.2e-3 | 0 | 0 | MODEL_INADEQUATE (size prior 1.4–14) | 3.1e4 |
| | | rollout | 1.6e-8 | 0.25 | 0.29 | MODEL_INADEQUATE / NOT_ID (degenerate covariance) / NOT_ID | 23 |
| | | EKF | 1.9e-2 | ~0 | ~0 | MODEL_INADEQUATE (NIS 0.12) | 6.5e3 |
| robot dog (v001) | — | — | — | — | — | KINEMATICS_NOT_ACCEPTED (MODEL_INCONSISTENT): no joint to project the force on | — |
| white drawer (v001) | — | — | — | — | — | DATA_UNAVAILABLE (no robot archive or force export) | — |

**Rank, conditioning and nested variants.** These are in `model_adequacy.json` and `identifiability.json`.
- **Fridge:** whitened rank 3 with condition ≤ 6, so the regressors are informative. The failure is the data
  contradiction, not conditioning.
- **Suitcase:** condition 12.
- **Black box:** condition 667, which is near-collinear: mostly one-direction, constant-speed motion.

**Nested-model ranking.** By BIC, labelled DIAGNOSTIC: fridge `inertia_coulomb`, suitcase and black box
`inertia_only`. These are not claims that a physical term is absent.

**Simulator prior instead of an estimate.** No scene has an identified absolute inertia, so `simulation_handoff.json`
carries a `PRIOR_FOR_SIM` (log-uniform, assumed mass 2–20 kg):

| scene | I_eff prior |
|---|---|
| fridge | 0.10–1.0 kg·m² (plate 0.39 m about its edge) |
| suitcase | 2–20 kg |
| black box | 1.4–14 kg·m² |

## 6. Estimator comparison on identical observations

**Identical inputs.** NNLS, integral LS, rollout and EKF all use the **same** `JointInteractionRecord`, the same
minimal `joint_space_only` model (M, c, Fc) = (I_eff, b, μ), the same native ZOH effort, and the same frozen-θ replay
protocol (q0 = first observation, v0 = line fit of the first 4 observations).

**Fridge.** All methods give I_eff between 4e-3 and 1.1e-2 kg·m², 10–25× below the smallest plausible fridge door.
Every replay is ≥ 365σ. That is the expected signature of motion the measured effort does not drive. No estimator
"rescues" it, and none of their values is labelled identified.

**Suitcase.** The methods disagree (1.3–8.2 kg). The EKF innovations are inconsistent. The joint is unaccepted.

**Black box.** The best replay is the rollout (23σ), at I → 0. The regressors are near-collinear.

**PINN.** `general_pinn.json` is NOT_RUN in this batch, because it is the long run; the command is in §11. Its seed
is now selected by frozen-θ replay.

### 6.1 Gate fix (2026-11-22, after the PINN batch v004 / v005)

**What went wrong.** The PINN batch (fridge v004, suitcase v005, black box v004) exposed two mislabels:
- **Fridge:** the PINN's μ = 0.0055 was labelled ESTIMATED_CONDITIONAL, with a replay of 45,000σ.
- **Suitcase:** the PINN's I = 8.31 kg was labelled CONDITIONAL_ON_UNACCEPTED_KINEMATICS, with a replay of 159σ.

The gate had accepted the seed-ensemble spread (0.06–5 %) as an uncertainty.

**The fix.** There is now one shared gate, `forcetwin_linear.gate`, used by every general estimator:
1. A seed-ensemble spread is **never** an uncertainty: the result is NOT_IDENTIFIED.
2. A frozen-parameter open-loop replay above `REPLAY_MAX` = 10σ, or a model that cannot be simulated, makes the
   method MODEL_INADEQUATE. The NNLS identifiability applies the same rule.
3. The existing rules still apply: estimator status, the physical bound, the size prior, a degenerate or > 25 %
   uncertainty, the impulse audit, EKF NIS consistency, and the kinematics gate.

**Effect.** The physics conclusions are unchanged: nothing was identified before, and nothing is now. Only those two
labels change. Versions written before the fix keep the old labels (append-only); rerun the batch to regenerate them.

## 7. Mechanism residual (ablations on the frozen NNLS baseline; blocked chronological validation)

| scene | none | g(q) | damping | both | label |
|---|---|---|---|---|---|
| fridge | 0.58 | 0.53 (−7 %) | 0.98 (worse) | 3.26 (worse) | EXPLORATORY_TRAIN_ONLY; g overlaps q̈ (corr 0.6): it absorbs inertia-like signal, not a validated mechanism |
| suitcase | 27.1 | 23.4 (−14 %) | 27.9 | 23.5 | EXPLORATORY_TRAIN_ONLY |
| black box | NOT_RUN | | | | baseline I_eff at its bound: no defensible frozen baseline |

The numbers are validation whitened RMS.

**No residual is exported.** None has independent validation, and the baseline itself is not identified, so
MECHANISM_VALIDATED = false everywhere. Where a residual could be supported, it would be RUNTIME_PLUGIN_REQUIRED; a
bare URDF cannot carry it.

## 8. Geometry and simulator mapping

**Primary estimation does not use the hull.** `joint_space_only` I_eff does not depend on hull density. Test 08
checks this: a wrong hull leaves I_eff unchanged.

**Mass allocation.** ρ = (I_eff − I_ref) / I_axis,1, then m = ρ m₁ and J_C = ρ J_C,1. This allocation is computed only
when I_eff is ESTIMATED_CONDITIONAL and the impulse audit is consistent. **That holds for no scene**, so the status is
NOT_ALLOCATED with the prior above.

**Hull masses.** The earlier 0.08 kg "door masses" from hull densities are not physical, and are no longer produced
as an estimate.

**Engine mapping** (`simulation_handoff.json`):

| engine | mapping |
|---|---|
| URDF | `<dynamics damping friction>` (engine-specific semantics). A scalar hinge inertia does **not** fit a URDF joint field: the link `<inertial>` needs a mass allocation |
| MJCF | damping, frictionloss, and armature (which can carry an effective joint inertia) |
| SAPIEN / ManiSkill | articulation-API friction / damping; the URDF loader's friction handling must be verified at runtime |
| Isaac | NOT_RUN |

ASSET_COMPILES and ENGINE_REPLAY_VALIDATED are both **NOT_RUN**: no engine was executed.

## 9. Tests

**`tests/test_g9_forcetwin.py`: 18 tests, OK.** This is 13 original tests plus 5 for the gate rules (§6.1); they were rerun after the gate fix.
- **Record contract:**
  - the sign is applied once;
  - the (a, q, τ) reversal gives identical estimates and invariant τ q̇;
  - invalid metadata and zero sigma are rejected;
  - the native ZOH pulse integral is preserved.
- **Synthetic identification:**
  - vertical panel: I, μ, b recovered (direct and weak); coast-only data identifies ratios only;
  - oblique axis with known gravity: one density for both terms;
  - prismatic, horizontal and mildly inclined;
  - constant-speed motion: rank failure, UNOBSERVABLE;
  - 80 ms impulse with 50 Hz force and 25 Hz q: IMPULSE_CONSISTENT;
  - wrong force sign: caught (NO_POSITIVE_INERTIA_EXPLANATION, negative unconstrained OLS); inertia never negative;
  - motion driven by a hidden torque: UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE;
  - the effective mode does not depend on the hull;
  - EKF on the minimal model, and PINN dr/dθ for I, b and μ;
  - a door-closer residual: g improves validation, and the damping head is strictly dissipative;
  - tilted vs vertical hinge from synthetic RGB-D tracks: free-axis refit within 1°.
- **Synthetic-design note:** a steep slider without end stops is genuinely unidentifiable, because constant gravity
  is collinear with sgn(q̇) in one-way motion, so the test uses a 2.3° incline.

**Full suites, after all non-pipeline edits:**

| suite | tests | result |
|---|---|---|
| G9 (`test_g9*`) | 143 | OK, including the 13 above |
| joint (`test_joint*`) | 185 | OK |

The later edits touched only `forcetwin/pipeline.py`, and `test_g9_forcetwin` was rerun after them.


## 10. Measurement limitations and the minimal next acquisition

**What the current data cannot provide.**
- **Fridge.** The door is driven by a torque the sensor does not see. The measured effort at the noise level cannot
  produce the observed −0.32 rad/s.
- **q sampling.** Effectively 12.5 Hz (stride 2), which cannot resolve an 80 ms pulse.
- **Tilt.** 6° vs vertical is not distinguishable from the RGB-D tracks.
- **Episodes.** There are no held-out episodes.
- **Suitcase / black box.** The joint is unaccepted or declared; the motion is one-directional; the regressors are
  near-collinear.

**Minimal next acquisition (fridge first).** This targets the observed failure mode:
1. **Unlatch first.** Open the latch or seal by a few degrees before recording, so the door is free at the start,
   and record a 2 s rest.
2. **Keep the tool on the door.** Make the instrumented tool the **only** contact, held against the door for the whole
   motion: a pushed swing, not a tap followed by a free swing. Log the per-row contact state.
3. **One fast acceleration and deceleration push** (0.3–0.5 s, at least 5 N at the handle). This is what identifies
   I_eff. A slow sustained push alone does not.
4. **Speed plateaus and reversals.** Push at two or three different speeds, and reverse direction at least twice.
   This separates b from μ.
5. **A free coast after release** (b/I, μ/I), and pushes at several door angles (closer or self-opening torque).
6. **Full-rate cameras.** Record q at the full 25–30 Hz (no stride), and declare the tool mass to bound the tool's
   own inertial force.
7. **At least two episodes,** one held out.


## 11. Commands, hashes, outputs

**Run everything** (append-only; about 5 min without the PINN; the PINN adds about 3 min per scene):

    cd ~/ForceArtGS
    RESIDUAL=1 nohup bash pipelines/g9_forcetwin_all_real.sh \
      > "/media/samuelemara/Extreme SSD/Samuele/results_21_11_26/g9_forcetwin_batch.nohup.log" 2>&1 &
    PINN=1 RESIDUAL=1 nohup bash pipelines/g9_forcetwin_all_real.sh \
      > "/media/samuelemara/Extreme SSD/Samuele/results_21_11_26/g9_forcetwin_pinn.nohup.log" 2>&1 &   # long

**Single steps:**

    ~/miniconda3/envs/freeartgs/bin/python tools/g9_forcetwin_fridge_audit.py          # impulse audit only
    ~/miniconda3/envs/freeartgs/bin/python -m forceartgs_g9.cli fit-minimal-dynamics --run RUN --scene SCENE \
        --robot-archive A.npz --contact-segments C.yaml --discovery-dir D [--pulse-rows 229:232] [--pinn] [--residual]
    ~/miniconda3/envs/freeartgs/bin/python tools/g9_forcetwin_comparison.py            # aggregate table

**Tests:**

    ~/miniconda3/envs/freeartgs/bin/python -m unittest tests.test_g9_forcetwin
    ~/miniconda3/envs/freeartgs/bin/python -m unittest discover -s tests -p "test_g9*.py"
    ~/miniconda3/envs/freeartgs/bin/python -m unittest discover -s tests -p "test_joint*.py"

**Outputs.**
- Per run: `<run>/dynamics/forcetwin_adaptation/vNNN/`. The final versions are fridge v003, suitcase v004, black box
  v003, robot dog v001 and white drawer v001. Earlier versions from the development iterations are kept.
- Aggregate: `/media/samuelemara/Extreme SSD/Samuele/results_21_11_26/forcetwin_comparison/comparison_all.{csv,json,md}`.
- Input hashes: `input_manifest.json` per version (record, archive, segments, handoff, authority).

**Git.** HEAD `9298f77`, 44 dirty entries, recorded in every manifest. **Nothing was committed or pushed.**

