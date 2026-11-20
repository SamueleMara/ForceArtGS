# Codex implementation directive — ForceArtGS G9 corrections and multimodal robot-record integration

**Date:** 2026-11-20  
**Repository:** `~/ForceArtGS` (published `main` and `multimodal-tcp-contact`: `136c462624aa3d4ea4f05666d6e8cc752dc860d9` when audited).  
**Other repositories:** `~/ForceArtGS-Physics` pinned at `f3824316272d85cb8d788166372c81ac3105ca7c` (DO NOT MODIFY); `~/PArtTwin1D` (shared builder still reported untracked; do not change simulator writers as part of this task).  
**Purpose:** Correct identified inconsistencies in force processing, impulse-based dynamics analysis, exploratory uncertainty, real-model interpretation, and use of the same robot evidence by the *pre-3DGS joint frontend* and G9. Do not redesign or retrain Gaussian models. Do not imply that new measurements or new calibration are needed when current records already contain the data.

## 0. Non-negotiable invariants

1. **Single measured force path.** Reuse `samuele/export_force_contact_annotations.py` `contact_annotations.csv/json` (already tare-corrected force and noise at native robot times), the existing `robot_archive.npz` (`T_base_ee`, `T_base_tcp`, `source_row`), scene metadata, and `configs/contact_intervals/<scene>.yaml`. Never re-tare or re-synchronize this input. Preserve exact `source_frame_index` associations and raw recorded timestamps, with a separate declaration of unknown hardware exposure offset.
2. **Correct sign exactly once:** Sensor recording describes the **force of the object on the robot**. Force acting on the object is its opposite. With the declared sign and measured/declared transforms,
   \[
   \mathbf F_{\mathrm{robot\to object}}^{B}(t)
   = s\,\mathbf R_{B,EE}(t)\mathbf R_{EE,S}\mathbf F_{\mathrm{sensor,corr}}^S(t),\qquad s=-1.
   \]
   `F_sensor_corr` already has the initial tare removed. Do not negate it in the annotation exporter and then negate again. Never infer `s` from optimizer fit, positive-work score, or joint preference.
3. **Effort is signed and in SI units:**
   \[
   Q_P(t)=\mathbf a^\top\mathbf F_{\rm obj}^B(t),\qquad
   Q_R(t)=\mathbf a^\top[(\mathbf p_{\rm TCP}^B(t)-\mathbf o^B)\times\mathbf F_{\rm obj}^B(t)].
   \]
   Force-on-object is the *external input* to dynamics. Never use its negative as `Q` after canonical conversion. No pure wrist-torque term unless its origin/frame is independently supported.
4. **Do not promote evidence:** `DECLARED`, `ESTIMATED`, `DATA_CONSISTENT` and `VERIFIED_BY_CONSTRUCTION` are NOT equivalent to hardware-calibrated `VERIFIED`. The exploratory policy must remain non-exportable. The certified policy remains the default.
5. **Keep the two separate optimization layers:** before 3DGS, RGB-D + tracks/masks + TCP/contact + candidate-aware force for joint discovery; after FreeArtGS training, G9 estimates dynamics. No new joint optimization inside G9 and no new 3DGS training.
6. **No automatic R/P decision from positive work or directional alignment.** Both sliders and hinged doors can exhibit positive actuator power. Force is hypothesis-conditioned *additional* evidence only if physically supported and improves held-out comparison. Never silently change production joint authority.
7. **No unsupported parameter identification.** `adequacy=FAIL`, `kinematics_not_accepted`, sensitivity to uncalibrated sign/timing, or confounding imply conditional/diagnostic estimates, not measured physical properties and not certified URDF parameters.
8. **Keep established fixtures and 57 G9 tests, 183 joint tests, and the validated ForceArtGS-Physics estimators unchanged.** Add meaningful regression tests. Never edit `_validated/` or copy its licensed implementation.

## 1. P0 — Fail-closed force sign and rotation contract

**Current source:** `estimotion_2part/joint_discovery/robot_record.py::build_robot_record()` currently contains:
```python
mult = float(conv.get("object_side_force_multiplier", -1))
F_obj = mult * np.einsum("nij,jk,nk->ni", T_ee[:, :3, :3], R_es, Fs)
```
The transformation itself is correct for the existing real data, but `-1` as an implicit default is unsafe.

**Required changes:**

- Require a **present**, finite `object_side_force_multiplier` in the scene's `metadata/contact_force_convention.json`. Restrict to `-1` or `+1`; reject `0`, any other value, NaN, null, absent, and inconsistent metadata. If an alternate explicit string `recorded_quantity` already defines the direction, check agreement with that quantity. Do not silently accept an ambiguous convention. Current real scenes with `-1` must continue working unmodified.
- Validate `R_ee_force_sensor` is explicitly present when the schema declares a non-default frame, is 3×3, finite, orthogonal and det≈+1; validate EE transforms similarly as appropriate. An explicitly declared identity remains valid for exploratory use, but is tagged `DECLARED`, not `VERIFIED`.
- Document the force path at each transformation: `F_sensor_raw` (raw archive), `F_sensor_corr` (from existing export; do not recompute), `F_object_base` (signed/rotated), and `Q_joint` (once projected). Save `sign_source`, the exact metadata SHA, `rotation_source`, and `conversion_version` with the canonical record.
- Refuse paths that pass a field already named `F_object_base_N` through the sign conversion again. Use explicit typed/field-level semantic names or a converter that accepts only `sensor_corr` as source.
- Ensure `wrench_physics.py::force_on_object_world()` and the G9 canonical record use the *same sign-convention helper*; do not create two independent fallback paths. Preserve legacy public function signatures if needed, but do not silently assume signs when source metadata is unavailable.
- Introduce a single round-trip/unit test: with `R=I`, `F_sensor_corr=[-2,0,0]`, declared object-on-robot convention, `F_object_base=[+2,0,0]`; for a prismatic +x slider `Q=+2 N`. With a +z revolute hinge and TCP lever +x, force +y gives `Q=+2r N m`. Apply a nontrivial rotation and verify covariance/physical power. Re-exporting the canonical record must not change the sign.

**Acceptance:** Existing real `-1` inputs produce bitwise-equivalent/within-float-roundoff `F_object_base_N` and `Q` to commit `136c462`. Missing/malformed convention raises an explicit error; not a fallback. `positive_work_check` cannot change the declared sign or certify it.

## 2. P0 — Analyze the fridge's short impulse separately from the coast

**Current source:** `forceartgs_g9/adapters/forceartgs_g9.py::record_interactions()` reports `effort_snr` from `Q` *during the fastest motion*, using a threshold based on `qdot > 0.5 max(qdot)`. This is NOT an impulse SNR. The initial ~2.6 N push lasted ~80 ms (~four samples at 50 Hz) and may precede the largest door velocity. Thus the reported 1.15 SNR describes the coasting/fastest-motion selection, not necessarily the input pulse.

**Implement:** New `forceartgs_g9/validation/impulse_coast.py` and import from G9 analysis/summary. It must:

1. Use **native force samples and their recorded timestamps** to segment independently supported push/contact impulses and following near-zero-input coast. Use `contact_state`, operator contact intervals, contact force relative to measured baseline noise, and positive-work diagnostics for *description*, not a fitted model to select windows. Report ambiguity/threshold sensitivity, never manufacture a contact interval. Preserve 50 Hz pulse resolution; do not downsample forces to q timestamps before integrating.
2. Integrate signed generalized effort:
   \[J_Q=\int_{t_a}^{t_b}Q(t)\,dt\]
   with the **recorded force sampling model** (explicit ZOH or trapezoidal rule, declared), including sample/window endpoint handling. Record `duration`, `peak |F|`, `peak |Q|`, `J_Q`, `sigma_{J_Q}`, and time-offset sensitivity. Include the gap/aliasing between 50 Hz force and RGB-D/joint q samples.
3. Compute pre/post velocity estimates **using only independently measured** q(t), with fit windows separated from the impulse, and propagate q observation uncertainty, hinge/axis uncertainty, sampling/lag, and covariance. Prefer local polynomial/robust fits with explicit time windows. Do not take noisy two-frame finite differences and call them measured acceleration. If there is no valid pre-pulse velocity estimate or sufficient timing resolution, report `IMPULSE_UNRESOLVED`.
4. Integrate the exact joint-space balance over the pulse:
   \[
   M[\dot q(t_b)-\dot q(t_a)]
   =J_Q-\int_{t_a}^{t_b}\{c\dot q+F_c\operatorname{sign}(\dot q)+G(q)+b\}\,dt,
   \]
   using the same general model and switches as the standard estimator. Do not simply take `M=J_Q/Δqdot` when the other terms are not demonstrably negligible. Provide (a) a simplified impulse-only conditional estimate with explicit assumptions, (b) a **joint pulse+coast constrained fit** with parameters shared across both phases, and (c) sensitivity to pulse boundaries, ±lag, force sign, baseline noise, hinge geometry and load model.
5. Coast analysis: for independently force-absent intervals with sufficiently low effort, fit/check the equation `M qdd + c qd + Fc sign(qd)+G(q)+b≈0`. Report scale-free ratios such as `M/c` only with correct uncertainty status. Separate pulse evidence for absolute scale from coast evidence for ratios. **Do not call `0.25±0.07 s` identified** if it fails the configured identifiability criterion.
6. Output `impulse_coast_analysis.json` and a compact plot showing native signed `Q(t)`, measured `q(t)`, estimated velocity with uncertainty, contact state, impulse bounds, and coast intervals. Add to `tools/g9_exploratory_summary.py` with `SNR_pulse`, `SNR_coast`, `J_Q`, `Δqdot` and confidence/abstention.

**Acceptance:** Running on the fridge must yield an explicit verdict: `IMPULSE_INFORMATIVE`, `IMPULSE_UNRESOLVED` or `IMPULSE_NOT_INFORMATIVE`, supported by quantitative evidence. Do not preordain that inertia becomes identifiable. Current conclusion `nothing identifiable` remains if the refined analysis cannot resolve scale.

## 3. P0 — Separate physical gravity from effective load and test bias alternatives

**Current source:** `forceartgs_g9/estimators/model_spec.py::ModelSpec` uses `parameter_source=joint_space_only` and `G(q)=gA cos(q)+gB sin(q)` for R; this is an **unconstrained effective configuration-dependent load**, not an identified mass/COM gravity torque. The fridge spans only ~0.33 rad, so it may be nearly confounded with constant bias or a local linear load.

**Required:**

- Rename labels in report/output to `effective_configuration_load` for `joint_space_only` while preserving backward-compatible raw field names. Only `geometry_coupled` with calibrated gravity and mass/COM can be labeled physical gravitational torque.
- Run explicitly chosen, comparable candidate models on the **same data and windows**: (A) viscous + Coulomb ± static; (B) + constant bias `b`; (C) + one effective `gA cos(q)+gB sin(q)` basis; (D) both bias + basis where structurally resolvable. Keep all other toggles matched, and report that when `b` is omitted, the load basis may absorb it.
- Report the joint/conditional confidence region or profile likelihood of `(gA,gB)`, a model-vs-null test and identifiability of **the amplitude/load over the observed q interval**, not just each coefficient separately. Include basis condition number. Do not label a confidence interval crossing 0 as proof the complete load is unnecessary.
- Preserve `automatic_term_selection: false`; compare variants for diagnostics with training-only scores/information criteria when no independent validation episodes exist, then require held-out episodes for physical claims.
- Test whether `chi²/dof≈47` for no-gravity vs ~1.3 for effective load can instead be explained by a constant bias, time offset, residual force, or a systematic q trend. Do not assert gravity from model adequacy alone.

## 4. P1 — Correct the uncertainty model (do not turn scenarios into fake standard deviations)

**Current sources:** `configs/g9_real/exploratory_*.yaml` use `force_scale_sweep_rel: 0.05` (unsupported ±5% sensor scale), `contact_point_sweep_m: 0.015` (operator 1–2 cm tolerance), `force_lag_sweep_s: 0.01` (nominal 50-Hz half-row), and `forceartgs_g9/validation/general_identifiability.py::diagnose_general()` currently computes
```python
sys_tot = sqrt(sum(shift[param]**2 for shift in sysd.values()))
total_std = hypot(stat_std, sys_tot)
```
This implicitly treats *exploratory deterministic sweeps* as independent 1σ errors; that statistical interpretation is not justified.

**Required:**

- Split `statistical_ci` (from estimator covariance/profile, with correlation/autocorrelation handling) from **`sensitivity_scenarios`** (declared perturbations and change in estimates) and any genuine **`calibrated_uncertainty`** (known distribution or covariance). Aggregate probabilistically only quantities whose distributions and dependence are justified; preserve input covariance and shared nuisances.
- The ±5% force-scale is an explicitly *unvalidated stress-test assumption*. Do **not** report it as a sensor-calibrated 1σ. Allow it to be `null`/disabled; preserve it as an optional labeled scenario for comparability.
- For contact point/hinge, do symmetric ± excursions and multiple physically meaningful directions in 3D: along the lever, tangential to the path, along axis (and/or covariance eigenvectors if covariance available). Account for *joint hinge uncertainty* separately from tool/TCP uncertainty without double counting. For a prismatic joint, contact-point position does not affect `Q_P= a·F`; do not fabricate a lever sensitivity.
- For time lag, support signed ±lag and a bounded nuisance model, distinguishing exact source-row correspondence from unverified sub-row hardware offset. Include residuals/errors over the impulsive interval, where this matters most.
- Second tare check must be based on an **independent contact-free window** (operator-marked or predeclared time interval) for verification. Currently `robot_record.py` picks rows labeled `absent` by a force threshold, biasing the selected window toward low residual force; retain this as a *threshold-conditioned consistency diagnostic* only. Never claim independent drift verification from those selected rows.
- `derived_combinations` (e.g., `M/c`) currently use statistical covariance alone; report scenario sensitivity on these derived quantities too. Do not mark a derived quantity physically identified if its apparent precision relies on an assumed model rejected by the adequacy test.
- Add provenance labels to all uncertainty numbers: `STATISTICAL`, `CALIBRATED`, `DECLARED_SCENARIO`, or `CONDITIONAL`.

## 5. P1 — Integrate shared `robot_record.py` into the PRE-3DGS joint frontend

**Current source:** `estimotion_2part/joint_discovery/multimodal_frontend.py::Frontend.setup()` uses its own `ContactRecord`, raw `r.wrench_sensor` and `force_gate(self.cal, ...)`. It exports TCP but not the canonical, tare-corrected object-side force used by G9. As a result, G9's exploratory fitting and the frontend's physical selection see different evidence/policies.

**Required:**

1. Make `robot_record.py` the one source of force samples and sign/rotation/tare provenance for **both** G9 and the frontend. Add a light, tested adapter to align its 50 Hz native records to `obs.frame_map.source_frame_index` and `timestamp_s` (no unannounced interpolation). Use correct world/base transformation; assert frame compatibility and IDs. TCP, contact labels and force must refer to the same source row; never independently retare or reread uncorrected force if a canonical record was supplied.
2. Preserve `sliding`, `stick_soft`, and `stick_hard` contact modes and the original RGB-D/track updates. Do not conflate *TCP-object slip mode* with *joint static/Coulomb friction*.
3. Implement opt-in `force_policy: exploratory` for the frontend as well as G9. `certified` remains the default and must reproduce existing selections. `exploratory` permits conditional force residuals in candidate fits if explicitly declared source metadata is present and noncontradictory; outputs remain **DIAGNOSTIC**, never `physics_joint_export_ready=true` based on such evidence. If the joint would be identified from visual+TCP evidence alone, retain independent certified evidence without attributing it to exploratory force.
4. Compare identical R/P candidate hypotheses and **identical train/held-out observation blocks** under `RGB-D`, `RGB-D+TCP`, and `RGB-D+TCP+force`. Record joint type, axis, hinge, `q_t`, support/masks, held-out RGB-D error, force/contact consistency and whether a decision changed. Candidate-independent calibration/force recording should be shared; hypothesis-dependent force checks must use held-out samples to avoid self-certification.
5. Explicit accept/rollback: preserve the prior certified result unless the new candidate improves the predetermined held-out joint/geometry/contact objective without breaking physical constraints. It is acceptable for the *exploratory diagnostic candidate* to change; not acceptable to overwrite certified joint authority merely because a conditional force check favored it. Force-work alignment alone is NEVER a prismatic selector.
6. Where `positive_work_check` is used for sign validation, limit its diagnostic to independently contact-confirmed and force-valid rows (not all authority frames); add a candidate-independent sign audit where possible. Report inconclusive for a noise-floor fridge rather than silently flipping the multiplier.
7. Ensure the per-run manifest records canonical record fingerprint, force policy, which residuals actually entered optimization, held-out comparison and whether force changed the selected hypothesis. Do not claim full dense-mask iteration if the current frontend still updates dense masks only at the final stage.

## 6. P1 — Stop promoting inadequate fits to physically identified parameters

**Current report:** the suitcase working prismatic model has `c=20.7 [17.9,23.6] N s/m` from a viscous-only fit, but this very model has **adequacy FAIL** (`χ²/dof≈4`). Adding an effective constant load fits better but induces confounding. The joint type is `TYPE_UNOBSERVABLE`.

**Required:**

- Add two separate verdict dimensions everywhere: `parameter_statistically_resolved_in_this_model` and `physically_interpretable_validated`. Physical interpretation requires at least adequate model fit, sufficient excitation, a sufficiently accepted joint model, and (for production) independent force calibration/held-out validation.
- Update `tools/g9_exploratory_summary.py`: do **not** print `damping identifiable` as an unqualified conclusion for suitcase. Say `viscous-only conditional estimate; model adequacy FAIL; working prismatic joint not certified`. The physical quantity remains unresolved.
- The frozen viscous baselines are comparisons **under the same simplified model**; their agreement is not independent validation of a correct physical model.
- Do not modify upstream model-selection thresholds to get a suitcase joint type from the dynamics fit itself; joint/force re-evaluation belongs to the held-out frontend step (section 5).

## 7. Tests and evaluation gates (MANDATORY)

Create/add tests in `tests/test_g9_robot_record.py`, `tests/test_g9_general_identifiability.py`, `tests/test_joint_multimodal_frontend.py` and dedicated `tests/test_g9_impulse_coast.py`. Minimum scenarios:

1. **Explicit sign contract:** reaction-on-robot -> opposite on object, R=I and rotated R, both R/P generalized efforts, exact once; no metadata => hard error; malformed multiplier => error; sign-reversed input => detectable contradiction but never automatic correction.
2. **Tare handling:** no duplicate subtraction; original export hashes unchanged; threshold-conditioned post-contact absent window cannot become independent tare-verification evidence; independent designated second window accepted.
3. **Time:** exact shared `source_frame_index` joins; mismatched row IDs fail; q clock and force timestamps separate until correctly aligned; synthetic 80 ms pulse remains four 50 Hz samples before integration; ±lag sensitivity.
4. **Impulse/coast:** known simulated pulse + coast with identifiable M/c; high impulse followed by low-force coast cannot be called low-SNR for the entire interaction; missing/sparse q around pulse => `IMPULSE_UNRESOLVED`; signed negative work does not automatically flip convention.
5. **Gravity/bias:** short-range `A cos q+B sin q` vs constant bias gives a confounding diagnostic; cannot label effective load as physical gravity; full geometry-coupled model remains unchanged.
6. **Uncertainty:** scenario-only ±5% affects sensitivity envelope but NOT statistical CI; symmetric ±lever perturbations affect R and not P; derived M/c gets scenario sensitivity; no double counting common uncertainty.
7. **Joint frontend shared-record:** RGB-D-only and certified baseline unchanged; opt-in exploratory force can influence a *diagnostic* candidate and must record ablation/heldout; cannot promote conditional force into `physics_joint_export_ready`; same raw frame and signed force as G9; negative tests for low force and wrong axis.
8. **Model adequacy:** an apparently narrow CI on a rejected model does not produce `physically_interpretable_validated=true`; suitcase-like unaccepted-joint diagnostic remains provisional.
9. **Regression:** existing 57 G9 tests, 183 `test_joint_*.py` tests pass; no changes to pinned ForceArtGS-Physics; two standard 3DGS PLYs / visual output unchanged; `g9_real_exploratory.sh` on real data creates a new version rather than overwriting prior reports.

Run (adapt to the repo's available test runner without changing the validated dependency):
```bash
cd ~/ForceArtGS
python -m unittest discover -s tests -p 'test_g9_*.py'
python -m unittest discover -s tests -p 'test_joint_*.py'
bash pipelines/g9_real_exploratory.sh
python tools/g9_exploratory_summary.py '/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware'
```
Do not run full GPU re-training; this is CPU/metadata + existing trajectory diagnostics unless user requests it.

## 8. Required deliverables

- Code implementing sections 1–6, organized in existing modules; one new `validation/impulse_coast.py` where appropriate.
- Updated `configs/g9_real/*.yaml` with **declared scenario vs calibrated uncertainty** separation, backward-compatible reading where possible.
- Updated `docs/G9_GENERAL_DYNAMICS_IMPLEMENTATION_REPORT.md` with **measured, conditional, rejected, and not identifiable** distinctions and the real fridge pulse/coast findings.
- A new `docs/G9_CORRECTIONS_AUDIT.md`: changed files, exact call graph, input semantics/provenance, before/after measurements (including pulse/ coast SNR), all test commands/results, ablations, and unresolved items.
- A detailed per-scene table for fridge, suitcase and black box: selected/working joint status, force conversion sign, impulse SNR vs coast SNR, model-adequacy verdict, statistical CI, systematic scenario envelope, conditional/physical verdict. Never invent results.
- Reproducibility check: published G9 code may be committed, but `~/PArtTwin1D` shared builder is still reported untracked. Inspect it and state precisely whether it has since been committed/pushed; do **not** silently implement a parallel builder or edit an unrelated simulator.
- **Do not auto-commit or push**; report tested diff and provide minimal commit commands if asked.

## 9. Final scientific/implementation acceptance criteria

The correction is accepted only if it establishes, with tests and real reruns, that:

- `F_object = -R F_sensor_corr` is applied once under an **explicit** reaction-on-robot convention, and both G9 and joint discovery consume it consistently.
- Short high-amplitude input impulses and low-force coasts are **evaluated separately**, with uncertainty and timing rather than a misleading single SNR.
- Model terms `effective_configuration_load` and true geometry-coupled gravity are **not conflated**.
- The 1.5 cm lever and 5% force-scale assumptions are **sensitivity scenarios**, not falsely treated as measured Gaussian uncertainties.
- Joint discovery and G9 share the canonical robot record, with a safe certified/exploratory split and held-out R/P comparisons.
- An inadequate viscous-only suitcase fit cannot produce a physical identified damping claim.
- No new measurements are demanded merely to *run conditional joint-space dynamics*. Physical certification and URDF physics-export readiness remain separately gated.

**Deliver a concrete diff and quantitative outputs. Do not state that any real parameter is identifiable unless its precise model, excitation, uncertainty and adequacy checks support that statement.**
