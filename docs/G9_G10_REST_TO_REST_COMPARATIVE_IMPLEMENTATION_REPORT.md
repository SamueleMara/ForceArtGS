# G9 / G10 rest-to-rest comparative implementation report

Spec: `FORCEARTGS_REST_TO_REST_G9_PINN_G10_CODEX.md`. Date 2026-11-26. **No commit / push was made.**

## 1. Code state, changed files, result paths

- Branch `multimodal-tcp-contact`, HEAD `58dc613` (the audited `9bf4802` plus the suspend-resilient run commit); the
  worktree is dirty with the files below only (plus three untracked spec `.md` files).
- **New files**

  | file | lines | content |
  |---|---|---|
  | `forceartgs_g9/estimators/rest_to_rest.py` | 416 | `RestToRestEpisode` (frozen, read-only arrays, sha256, npz round trip, `to_physx_episode`), `RestEndpointEvidence`, visual plateau selection, stationary-load audit, `terminal_speed_sigma` |
  | `forceartgs_g9/estimators/rest_to_rest_integral.py` | 358 | B1 `endpoint_r2r` (Eqs. 2, 5), B2 `integral_svd_r2r` (Eq. 4: unweighted / declared-covariance GLS / Huber; column-scaled SVD; bootstrap; family), `bias_ablation`, `map_to_physx`, `output_error_crosscheck` |
  | `forceartgs_g9/estimators/rest_to_rest_convolution.py` | 166 | B3 exact ZOH convolution (Eq. 6), `simulate_two_term` (exact solution), lambda profile and valley |
  | `forceartgs_g9/estimators/rest_to_rest_diagnostics.py` | 120 | orthogonality (Eq. 7), damping energy (Eq. 8) with bootstrap, model-free work |
  | `forceartgs_g9/estimators/rest_to_rest_pinn.py` | 273 | `RestToRestNet` (Eq. 9), double-integral residual (Eq. 10), B4 / B5 / B6 / soft-BC variants, legacy PINN matched adaptation |
  | `forceartgs_g9/estimators/rest_to_rest_synthetic.py` | 103 | exact braked rest-to-rest generator, viscous coast, out-and-back |
  | `forceartgs_g9/physx_refinement/rest_to_rest_objective.py` | 53 | V1 `TerminalVelocityLoss` (Eq. 13), weight 0 = the existing objective bit for bit |
  | `forceartgs_g9/physx_refinement/rest_to_rest_comparison.py` | 855 | strict config loader, input preparation (same sources as G10), analytics / PINN / legacy / PhysX stages, C0 control, `run`, `episode_only`, `run_synthetic` |
  | `forceartgs_g9/physx_refinement/rest_to_rest_report.py` | 325 | `summary.csv` / `summary.md`, `assert_controlled`, 8 figures |
  | `configs/g9_real/rest_to_rest_compare.yaml` | 93 | strict config (spec Section 8 keys + wired extras) |
  | `pipelines/g9_rest_to_rest_comparison_real.sh` | 38 | batch driver |
  | `tests/test_g9_rest_to_rest_{endpoints,integral,convolution,pinn,physx,comparison}.py`, `tests/r2r_fixtures.py` | 697 | T01-T25 |

- **Modified files (opt-in, defaults unchanged)**
  - `forceartgs_g9/estimators/general_pinn.py` (+37 / -8): `PINNConfig` gains `trajectory_boundary_mode`,
    `physics_form`, `l_double_integral`, `l_soft_bc`, `n_quad`, `replay_v0` (defaults = historical; mixing refused);
    `open_loop_replay(..., v0_mode="prefix_fit" | "zero")`; `general_pinn_fit` refuses the new modes.
  - `forceartgs_g9/estimators/model_spec.py` (+13): `inertia_viscous_spec(joint_type)` (named two-term variant).
  - `forceartgs_g9/physx_refinement/behavioral_fit.py` (+39 / -16): `fit_full_demo(..., objective_factory=None,
    terminal=None)`, `lm_lockstep / local_refine(..., terminal=None)` append the V1 residual only when given.
  - `forceartgs_g9/cli.py` (+32): new `rest-to-rest-compare` command (`--synthetic prismatic|revolute` too).
  - Unchanged: `configs/g9_real/physx_behavioral_inertia_damping.yaml`, `pipelines/g9_physx_behavioral_full_wrench_real.sh`,
    every historical CLI, PArtTwin1D, ForceArtGS-Physics.
- **Results (append-only)**: `/media/samuelemara/Extreme SSD/Samuele/results_26_11_26/rest_to_rest_compare_smoke/`
  (`synthetic_quick/{prismatic,revolute}/v000`, `real_fridge_quick/.../v000`, `regression_suite.log`). Real runs write
  to `<run>/dynamics/g9/rest_to_rest_compare/vNNN`.

## 2. Experimental facts and rest evidence

The facts are validated in the config (cannot be changed): known transform (applied once by the existing `FR.build`),
already tared (no baseline subtraction), continuous contact (input policy C1: every native row in [t0, T], no mask),
v0 = vT = 0 (episode invariants, never fitted). t0 / T come only from visual plateaus: chi2 of a constant-position fit
at the two-sided 3-sigma confidence AND trend |slope| / se <= 3, sigma = hypot(q_sigma, 3e-4), >= 4 samples and 0.12 s,
grown sample by sample with each new sample within 3 sigma of the plateau mean.

| scene (results_19_11_26 authority) | visual frames | status | evidence |
|---|---|---|---|
| red fridge | 180-296 (recording rows 0-576) | **REST_END_NOT_COVERED** | initial plateau 180-232 found; at 296 the door still moves ~0.011 rad / sample (terminal window chi2 / trend z 52) |
| orange suitcase | 100-240 (rows 0-324) | **REST_END_NOT_COVERED** | last samples -0.1723, -0.1741, -0.1743, -0.1729 m (sigma 0.36 mm): still settling at 240 |
| black box (declared revolute) | 140-256 (rows 0-456) | **REST_START_NOT_COVERED** | already moving ~1.5 mrad / sample at frame 140 |
| robot dog | - | DATA_UNAVAILABLE | 19_11 run: no `selection.json` (MODEL_INCONSISTENT); the new 25_11 run is also MODEL_INCONSISTENT (needs `--declared-joint-type`) |
| white drawer | - | DATA_UNAVAILABLE | no robot archive under results_17_11_26 (the new 25_11 run has one) |

So **no clip authority covers both rest plateaus**: the reconstruction clips were chosen around the interaction. On the
clip q the comparison is refused (no fake crop, no imposed vT = 0); the fridge pulse rows 229-232 were not used as a
window. The fix needs NO 3DGS: q is measured on the whole recording with the joint frozen (Section 2b).

### 2b. Frozen-joint q(t) on the whole recording (added 2026-11-26)

The clip limit is NOT a data limit: the full RGB-D videos exist (fridge 289 frames, suitcase 163, black box 229) and q(t)
comes from tracking + the joint, not from 3DGS. `estimotion_2part/joint_discovery/frozen_joint_extension.py`
(`python -m estimotion_2part.joint_discovery.cli extend-states`) keeps the accepted joint EXACTLY (type, axis, hinge,
gauge, the parent's q / sigma on its frames) and measures q on the extra frames: AllTracker + depth over the whole video
(`extract`), moving-part tracks selected on the clip overlap (pulled back through the frozen A(q)^-1 they collapse to
one canonical point), per-frame robust 1-DoF solve, validation against the parent q on the overlap, new versioned
authority (`parent_authority_id`). Batch: `pipelines/extend_joint_states_real.sh`; the comparison takes
`--authority-dir`. Tests: `tests/test_frozen_joint_extension.py` (revolute + prismatic recovery outside the clip, wrong
axis refused).

Real fridge (cam_001, frames 0-288 step 2, 264 s of AllTracker): 4344 moving tracks; overlap (30 clip frames) rms
difference 3.6 mrad <= tolerance 6.5 mrad (2 % of the 0.33 rad clip range), gauge offset 0.45 mrad; per-frame sigma
inflated x11 from the overlap (the per-point MAD ignores per-frame correlated errors). The rest-to-rest episode now
BUILDS: t0 = 0 s (frame 0), T = 7.28 s (frame 364), dq = -0.609 rad (the clip saw only -0.33 rad), 53 confirmation
samples (~4 s) of terminal rest; terminal stationary load -0.034 N m (z 7: UNMODELED_STATIONARY_LOAD).

Real fridge comparison (QUICK budget, `.../rest_to_rest_compare_smoke/real_fridge_extended_quick/compare/v000`):

| row | M [kg m^2] | c | rmse/sigma (PhysX, full episode) | v_sim(T) [rad/s] | rollouts / runtime |
|---|---|---|---|---|---|
| C0 historical masked input + G10 (replayed on C1) | 0.019 | 0.105 | 498.9 | -0.011 | 281 |
| B0 C1 + G10 baseline starts | 0.342 | 0.027 | 61.0 | -0.409 | 280 / 48.5 s |
| B8 C1 + G10 convolution start | 0.365 | 0.036 | 61.2 (lower objective 84.9 vs 86.3) | -0.370 | 259 / 26.6 s |
| B10 terminal loss w 0.1 / 1.0 | 0.221 / 0.042 | 0.50 / 0.77 | 108 / 125 | -0.036 / 0.0003 | 317 |
| B5 PINN hard BC double integral (frozen) | 0.102 | 0.263 | 123 | -0.058 | 1 |
| B2 INTEGRAL_SVD_R2R (frozen) | 0.008 | 0.088 | 658 (flag INTEGRAL_LS_OUTPUT_ERROR_INCONSISTENT) | 0.018 | 1 |

Reading: (1) the input policy dominates: the contact-masked C0 input is ~8x worse on the full record than C1; (2) the
two-term model is INADEQUATE for the fridge (61 sigma; the simulated door is still swinging at -0.4 rad/s at T while
the real one has stopped; terminal stationary load flagged) -- consistent with the 6 deg tilted axis / gravity and
friction found earlier; (3) the convolution start matches the baseline's optimum at fewer rollouts / half the time;
(4) the terminal penalty stops the door only by wrecking the trajectory.

## 3. Derivations and their tests (all run)

| test | what is checked | result |
|---|---|---|
| T01 | Eqs. (2)/(5), prismatic + revolute, q0 != 0, jittered 25 Hz clock | c_end exact to 1e-8, M_end to 2e-3 (trapezoid of q) |
| T02 | viscous coast is not exact rest | c_end - c = [M v(T) + c (q(T) - qT)] / dq, reported, never hidden |
| T03 | exact braked rest-to-rest (two pushes) | v(T) < 1e-12; endpoint and LS recover M, c |
| T04 | ZOH double integral vs adaptive quadrature, nonuniform, signed, t0 != 0 | rel 1e-9 |
| T05 | exact return to start | endpoint skipped (ENDPOINT_ESTIMATE_ILL_CONDITIONED); LS recovers M, c within 2 % |
| T06 | wrong sign / constant residual | negative c_end reported signed, MODEL_INPUT_INCONSISTENT |
| T07 | terminal load | UNMODELED_STATIONARY_LOAD, recorded values unchanged (no re-tare) |
| T08 | orthogonality | exact v: normalized < 1e-3; window ending in motion: int v vdot = v(T)^2/2; weighted: broken |
| T09 | SVD / family | well excited unique (rank 2); weak damping: not unique, c family spans > 0.1 |
| T10 | convolution | kernel exact (rel 1e-8) for lambda in {0, 1e-9 ... 60}, continuous at 0; recovery; flat valley flagged |
| T11-T14 | PINN | hard BC exact for arbitrary weights (incl. dq = 0); double-integral gradients finite, nonzero, no derivative call; variants on identical data; legacy defaults unchanged, new replay v0 = 0 |
| T15, T16, T19, T20, T22, T25 | actual SAPIEN PhysX (CPU; GPU parity) | engine start (q0, 0) bit exact, unmasked input passed, qdot(T) = the exact model's v(T); V1 with w = 0 bitwise V0, term exactly additive; rho = M / axis inertia and engine readback = M (1e-4); explicit surrogate label, Td = static = armature = 0, gravity 0; initializers within 10-15 %; fit not worse than its start |
| T17, T18, T21, T23, T24 | coverage statuses, no annotation influence, controlled-table guard, real coverage, strict config / CLI / append-only | pass |

Commands (the env has no pytest: a scratch pytest site is injected, as for the earlier G9 suites):
`PYTHONPATH=<scratch>/pytest_site ~/miniconda3/envs/freeartgs/bin/python -m pytest -q -p no:cacheprovider tests/test_g9_*.py tests/test_robot_tool_overrides.py`
-> **396 passed in 1020 s** (all existing G9 / G10 suites + the 48 new rest-to-rest tests).

Two findings from building the tests (implemented as reported behaviour, not hidden):
- **T placement is at the q resolution.** With q sigma 0.2 mm the terminal plateau starts ~20 ms before the exact stop
  (v_true(T) ~ 9 % of the peak). Eq. (2) then carries M v(T) / dq (synthetic: c_end 6.37 vs 5.4 true).
- **Equation-error bias of the double-integral LS.** With q noise large against the motion the LS can sit far from the
  trajectory optimum while its declared-noise sd is small (noisy q in the design columns + T inside sub-noise motion).
  The parametric bootstrap did NOT reveal it; the reliable check is `output_error_crosscheck`: replay the LS (M, c)
  exactly and compare with the convolution optimum (flag `INTEGRAL_LS_OUTPUT_ERROR_INCONSISTENT` when > 0.5 sigma worse).

## 4. Historical methods and matched adaptations

- (H) historical outputs are only inventoried (`historical_baseline_inventory.json`, label
  HISTORICAL_DIFFERENT_INPUT_OR_BUDGET); the archival quick full-model G10 numbers in
  `G10_FULL_WRENCH_BEHAVIORAL_IMPLEMENTATION_REPORT.md` are not compared with anything here.
- (M) matched adaptations on the SAME episode with the two-term spec: `general_integral_ls` (its moving-segment +
  per-segment nuisance definition unchanged), `general_rollout_fit` (declared (q0, v0 = 0)), `general_akf` (its
  4-sample v0 unchanged), GeneralPINN (net / terms / segmentation unchanged). All are replayed from (q0, 0) in PhysX.
- C0 (historical contact-masked input, `build_wrench_episode` unchanged, own window) is fitted at the same budget and
  replayed on the C1 episode; it needs `--contact-segments`. It could not run on real data here (no real C1 episode).

## 5. Comparative table (synthetic, actual GPU PhysX, QUICK budget)

Exact braked rest-to-rest, 25 Hz q with 0.2 mm noise, rows 50 Hz; truth prismatic M = 3.0 kg, c = 5.4 N s/m; revolute
M = 0.54 kg m^2, c = 0.972 N m s/rad (rho = 1.5 on the reference URDF). QUICK: DE maxiter 3, popsize 30, local 40 nfev,
PINN 1 seed x (200 + 300) steps. All rows: same episode (sha in `summary.csv`), model, engine, timestep.

| id | method | prismatic M / c | rmse/sigma | v_sim(T) | revolute M / c | rmse/sigma | rollouts (P / R) |
|---|---|---|---|---|---|---|---|
| B0 | G10, existing baseline starts | 3.006 / 5.399 | 0.4375 | 0.0591 | 0.5433 / 0.959 | 0.4684 | 316 / 313 |
| B7 | G10 + SVD start | 3.009 / 5.386 | 0.434 | 0.0598 | 0.5416 / 0.970 | 0.4338 | 295 / 313 |
| B8 | G10 + convolution start | 3.009 / 5.385 | 0.434 | 0.0598 | 0.5416 / 0.970 | 0.4338 | 295 / 313 |
| B9 | G10 + integral-PINN start | 3.009 / 5.386 | 0.434 | 0.0598 | 0.5416 / 0.970 | 0.4338 | 295 / 313 |
| B789 | G10 + union of new starts | 3.009 / 5.385 | 0.434 | 0.0598 | 0.5416 / 0.970 | 0.4338 | 297 / 315 |
| B10 | G10 + terminal loss w 0.1 | 3.821 / 8.864 | 53.3 | 0.0013 | 0.5095 / 1.278 | 7.85 | 290 / 288 |
| B10 | G10 + terminal loss w 1.0 | 5.155 / 12.13 | 92.0 | -0.0009 | 3.352 / 7.845 | 170.8 | 283 / 281 |
| B1 | ENDPOINT_R2R (frozen -> PhysX) | 2.773 / 6.365 | 4.80 | 0.0045 | 0.499 / 1.146 | 4.81 | 1 |
| B2 | INTEGRAL_SVD_R2R | 2.978 / 5.502 | 0.817 | 0.0534 | 0.5375 / 0.991 | 0.582 | 1 |
| B3 | ANALYTIC_CONVOLUTION_1D | 2.988 / 5.460 | 0.652 | 0.0556 | 0.5379 / 0.983 | 0.652 | 1 |
| B4 | PINN hard BC, strong | 2.863 / 6.416 | 3.58 | 0.0101 | 0.513 / 1.171 | 3.95 | 1 |
| B5 | PINN hard BC, double integral | 3.010 / 5.322 | 0.675 | 0.0624 | 0.5439 / 0.950 | 0.614 | 1 |
| H | legacy GeneralPINN (matched data) | 3.735 / 1.590 | 11.0 | 0.227 | 0.645 / 0.318 | 9.48 | 1 |
| M | general_akf (matched) | 2.989 / 5.484 | 0.553 | 0.0547 | 0.538 / 0.987 | 0.550 | 1 |
| M | general_integral_ls (matched) | 2.995 / 5.497 | 0.601 | 0.0545 | 0.539 / 0.990 | 0.601 | 1 |
| M | general_rollout_fit (matched) | 2.987 / 5.463 | 0.660 | 0.0555 | 0.538 / 0.983 | 0.659 | 1 |

Diagnostics (prismatic): SVD whitened M 2.978 +- 0.009, c 5.502 +- 0.038, corr -0.75, scaled singular values
[1.32, 0.50], unique; convolution lambda 1.83 1/s, no flat valley; model-free work 0.46 J (passive-consistent);
c_energy 5.40; bias ablation b0 = 0.001 (z 0.18, none). Revolute: analogous. Runtime 67 s / 72 s per scene (QUICK).
Figures per run: `figures/{trajectory_compare, residuals, terminal_velocity, force_input, svd_spectrum, param_valley,
optimizer_convergence}.png` (+ `rest_plateaus.png` for real runs).

## 6. Answers (synthetic, QUICK budget -- to be re-checked at full budget)

1. **Integral PINN vs strong PINN in open-loop simulation:** yes. Hard-BC double integral 0.68 / 0.61 sigma vs hard-BC
   strong 3.6 / 3.9 sigma vs legacy 11 / 9.5 sigma (PhysX replay of the frozen theta).
2. **Analytic starts at equal budget:** small, real gain at this tiny budget -- every new start reaches 0.434 sigma vs
   0.4375 (prismatic) / 0.468 (revolute) from the existing starts, with equal or fewer engine rollouts. All new starts
   converge to the same point, so here they improve speed / stability, not the final optimum. Re-check at full budget.
3. **Terminal-speed term:** NEGATIVE result. It worsens q tracking badly (53-171 sigma) because T is located at the q
   resolution while the truth still moves (v_true(T) ~ 0.056): forcing v_sim(T) ~ 0 fights the data. Keep weight 0
   unless T is known far more precisely than the plateau test can locate it.

## 7. Limitations

- Minimal model only (inertia + viscous; gravity zero, Coulomb off, no bias); not the real doors' / lids' full physics.
- Contact is treated as continuous per the experiment; a sustained stationary load would be flagged, not corrected.
- Real data: no current authority covers both rest plateaus (Section 2): the real comparison has not produced a
  number yet. Synthetic numbers are QUICK budget.
- One demonstration per scene: FIT_TO_DEMONSTRATION only; INDEPENDENT_REAL_ACTION_VALIDATED = false; no
  sim-to-real transfer claim.

## 8. Run commands

```bash
cd ~/ForceArtGS
# synthetic first (full budget)
~/miniconda3/envs/freeartgs/bin/python -m forceartgs_g9.cli rest-to-rest-compare --synthetic prismatic \
  --out-root "/media/samuelemara/Extreme SSD/Samuele/results_26_11_26/rest_to_rest_compare/synthetic_prismatic"
# real scenes (each prints its coverage status; PhysX runs only where both plateaus are covered)
OUT_BASE="/media/samuelemara/Extreme SSD/Samuele/results_26_11_26/rest_to_rest_compare" \
  bash pipelines/g9_rest_to_rest_comparison_real.sh
```
