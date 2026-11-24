# G9 free-motion normalized model — repository audit (spec §1, P0)

Date 2026-11-23. Branch `multimodal-tcp-contact`, HEAD `9298f77` + local uncommitted G9 work (git status showed 45 entries
before this change, 54 after). Nothing committed or pushed. `ForceArtGS-Physics` is untouched; the 3DGS, calibrations,
sensor sign, joint axes and joint types are untouched.

## 1. Existing code inspected and reused (no parallel physics stack)

| Function | File | What is reused | Changed here? |
|---|---|---|---|
| Canonical robot / force record | `estimotion_2part/joint_discovery/robot_record.py` | signed object-side force, `object_side_multiplier` applied exactly once upstream | no |
| G9 inputs / windows | `forceartgs_g9/adapters/forceartgs_g9.py::build_inputs` | joint authority, q(t), force audit, gravity, mass mesh, reference pose, input hashes | no |
| Joint interaction record | `forceartgs_g9/forcetwin/record.py::from_g9_inputs` | `Q = a·((p−o)×F)` / `a·F`, force rows (50 Hz, ZOH) joined to q on `source_frame_index` | no |
| Hybrid integrator | `forceartgs_g9/dynamics/general_1dof.py::simulate_general`, `sim_params` | the normalized rollout **is** `simulate_general` with M = 1, c = λ, Fc = γ, G = a_g, effort = βQ | no |
| Gravity law | `general_1dof.RigidBodyGravity.basis_coefficients`, `geometry_inertia.GeometryMassModel.gravity_unit / M1 / reflected` | a_g = G₁/M₁ (density cancels), reflected-inertia refusal | no |
| Model spec | `forceartgs_g9/estimators/model_spec.py::ModelSpec` | `NormalizedDynamicsSpec.from_model_spec` derives terms and gravity from it | no |
| Trajectory network | `forceartgs_g9/estimators/general_pinn.py::_Net, _derivs, DT` | per-segment MLP and autograd derivatives | no |
| ZOH helpers | `forceartgs_g9/synthetic.py::ZOH`, `torch_model.py::zoh_torch, zoh_integral` | measured input on native rows | no |
| Derivatives | `forceartgs_g9/forcetwin/derivatives.py::spline, local_poly` | velocity reconstruction (lambda now passed explicitly) | no |
| Kinematic hypotheses | `forceartgs_g9/forcetwin/kinematics_hypotheses.py::hypotheses` | accepted vs vertical-hinge refit from the RGB-D tracks | no |
| Vertical record / mass model / prior | `forceartgs_g9/forcetwin/pipeline.py::_vertical_record, _mass_model, _prior_for_sim, next_version, _write, _git` | reused | **yes** — handoff `value` → `estimate` / `estimate_not_for_sim`; rule text |
| Pulse audit | `forceartgs_g9/validation/impulse_balance.py::audit, momentum_case` | the fridge rows 229–232 audit is unchanged; `momentum_case` gives one bracket row per short pulse | no |
| Gate | `forceartgs_g9/estimators/forcetwin_linear.py::gate` | unchanged rules | **yes** — wording only (zero sd; operator-declared kinematics) |
| Mechanism residual | `forceartgs_g9/estimators/forcetwin_residual.py::fit_residual, ablations` | — | **yes** (spec §6.5 fixes, below) |
| Surface sampling | `forceartgs_g9/synthetic.py::_surface_samples` | hull surface points for the TCP separation test | no |

Legacy paths keep their behaviour. The one change to legacy default behaviour is the requested residual fix (§6.5);
`legacy_init=True, orthogonalize=False` reproduces the old training. The forcetwin handoff renames its field so that a
non-identified estimate can no longer reach a simulator.

## 2. Actual signatures of the new code

```
forceartgs_g9/free_forced/normalized.py
  UNITS[joint_type] -> {q, v, a, lam, gam, beta, M, Q, J, c, Fc}
  GravityModel(name, kind, joint_type, A=0, B=0, provenance="", depends_on_scale=False, extra=None)
      __call__(q), deriv(q), torch(q), fast(), amplitude(), reversed(), describe()
      zero(jt, ...), prismatic_ideal(axis, g_world), from_mass_model(mm, g_world, name, rho=None)
  NormalizedDynamicsSpec(joint_type, gravity, viscous=True, coulomb=True, forced=False, provenance={})
      names -> ["lam","gam"(,"beta")]; free(); with_forced(); with_gravity(g); pack(); unpack()
      accel(q,v,s,theta,Q); residual(...); sim_params(theta); simulate(theta,t_query,t0,q0,v0,effort=None)
      physical(theta, M, M_sd=None, cov=None, provenance=...)   # refuses M <= 0
      from_model_spec(spec, forced=False, gravity=None)
forceartgs_g9/free_forced/segments.py
  MotionSegment(name, label, joint_type, t, q, q_sigma, sign, force_t, Q, Q_sigma, evidence, q_sigma_reported)
      is_free, effort, J(ta,tb), split(frac), describe()
  classify(rec, *, z_sig=3, win_rows=5, v_k=3, sep_margin_m=0.03, part_points_ref=None, q_ref=0, min_rows=3,
           min_run=10, pulse_z=10, z_J=5) -> segment_evidence dict
  build_segments(rec, ev, *, free_hypotheses=(), min_q=5, v_k=3, noise="empirical") -> [MotionSegment]
  empirical_q_sigma(t, q, k=5); inflate_noise(seg)
forceartgs_g9/estimators/free_weak_form.py
  gcv_lambda(t,q,q_sigma, rule="discrepancy"|"gcv"); window_rows(seg, nspec, ...); all_rows; solve; profile; bootstrap
  fit(segments, nspec, *, B=200, sensitivity=True) -> free_weak_fit (status, per_parameter, profile, svd, ...)
  forced_scale(free, forced, nspec, *, free_fit=None, impulse_windows=()) -> forced_scale_fit
  impulse_window(rec, nspec, rows); effective_load_fit(segs, nspec, jt); gravity_negligible_test(segs, nspec_known)
  passivity_split(seg, nspec, k=3, min_q=5) -> (kept_or_None, audit)
forceartgs_g9/estimators/normalized_rollout.py   fit(segments, nspec, *, sigma_floor, starts, max_nfev); fit_staged
forceartgs_g9/estimators/normalized_ekf.py       NormalizedAEKF(nspec, sigma_a, sigma_eta_rw, h_max)
      substep / propagate / jacobian_fd / process_noise / run;  fit(segments, nspec, *, init, eta_sd, sigma_a_grid)
forceartgs_g9/estimators/normalized_pinn.py      NPINNConfig; fit_free(segments, nspec, cfg, init); fit_forced(...)
forceartgs_g9/free_forced/replay.py              prefix_state; segment_replay; replay(segments, nspec, theta, ...)
forceartgs_g9/free_forced/status.py              kinematic_authority; input_evidence; normalized_status;
                                                  absolute_scale; simulator_readiness; handoff_preview
forceartgs_g9/free_forced/pipeline.py            run(run_root, scene, *, cfg_path, ff_config, robot_archive,
                                                  contact_segments, discovery_dir, pulse_rows, pinn, out_root, ...)
CLI  python -m forceartgs_g9.cli free-forced-dynamics --run R --scene S [--pinn] [--pulse-rows A:B] ...
```

## 3. Input / output schemas

- **Input:** the existing G9 inputs (`build_inputs`) plus `configs/g9_real/free_forced.yaml` (`dynamics_config:` points
  at the unchanged `configs/g9_real/forcetwin_effective.yaml`).
- **Output** (append-only `<run>/dynamics/g9_free_forced/vNNN/`):
  - `resolved_config.yaml`, `source_hashes.json`
  - `segment_evidence.json` (`forceartgs.g9.segment_evidence_set/v1`)
  - `normalized_model_spec.json` (`…normalized_model_spec/v1`)
  - `free_weak_fit.json`, `free_rollout_fit.json`, `free_ekf_fit.json`, `free_pinn_fit.json`
  - `forced_scale_fit.json` (`…forced_scale_fit/v1`)
  - `identifiability_report.json`, `model_adequacy.json`, `geometry_gravity_hypotheses.json`
  - `simulator_handoff_preview.json` (`…simulator_handoff_preview/v1`, typed, no `value` field)
  - `plots/` (5 PNGs), `comparison_summary.md`, `comparison_rows.csv`, `STATUS.json`

## 4. SHA256 of critical inputs (from `source_hashes.json`, dry run 2026-11-23)

| scene | robot_archive | force_export | mass_mesh | joint authority id | contact segments yaml |
|---|---|---|---|---|---|
| red fridge | `b8d1aca7c787f337…31cb6b7` | `f96553606bb9d3d6…b0947e0b` | `115bc37801f9aa6a…3f27d5a7` | `2f35c6cba93d6e99…d60b7bb3` | `cd8d75fbb5d63567…a09901c` |
| orange suitcase | `7ea3474ef5e337d1…f48a4d` | `20dccf081bc41cad…d89e` | `b8a8319d0f5a7103…9452` | `5ab24f6431875556…924fd9` | `788ed1464dc88a60…ac2e7d` |
| black box | `494a230968bd48ad…77dd` | `7c15796128ee6b92…15b2fe` | `e02801c474a7b404…0567` | `71637622b86cd645…8de024` | `d2d36cf15b45d315…704291` |

Configs: `forcetwin_effective.yaml` `746f6edd…5b4429c`, `free_forced.yaml` `5380c07a…c49919` (the full hashes are
in each `source_hashes.json`).

## 5. Existing tests (baseline before this change) and files modified

- Baseline: the G9 suite (`tests/test_g9*.py`) passed before this change, including the 18 forcetwin tests.
- New: `tests/test_g9_free_forced.py` (20 tests). After the change the G9 suite has 168 tests, all passing.
- **Modified:** `forceartgs_g9/estimators/forcetwin_residual.py`, `forceartgs_g9/forcetwin/pipeline.py`,
  `forceartgs_g9/estimators/forcetwin_linear.py` (gate wording), `forceartgs_g9/cli.py`.
- **New:**
  - `forceartgs_g9/free_forced/{__init__,normalized,segments,replay,status,pipeline}.py`
  - `forceartgs_g9/estimators/{free_weak_form,normalized_rollout,normalized_ekf,normalized_pinn}.py`
  - `configs/g9_real/free_forced.yaml`, `pipelines/g9_free_forced_all_real.sh`, `tools/g9_free_forced_comparison.py`
  - `tests/test_g9_free_forced.py`, this audit and `docs/G9_FREE_FORCED_DYNAMICS_IMPLEMENTATION_REPORT.md`
