# Joint selection v2: implementation report (2026-11-18)

**Spec:** `FORCEARTGS_JOINT_OBSERVABILITY_AND_PHYSICAL_MODEL_SELECTION_CODEX.md`
**Design:** `docs/JOINT_PHYSICAL_SELECTION_V2_DESIGN.md` (algorithm, schema, flowchart)
**Repository:** branch `main` at `468071f`. Uncommitted, like all work since that commit; nothing is committed or pushed.
**Submodule:** `reconstruction/nerfstudio` is unchanged by this work.

## 1. Changes and call graph

**New files** (`estimotion_2part/joint_discovery/`):

| file | contents |
|---|---|
| `blocked_statistics.py` | ACF-based block length; `BlockPartition` declared before fitting; block bootstrap; surface-cluster jackknife; `count_independent_blocks` |
| `motion_field.py` | `PointTable`; `RigidIntervalEvidence`; `estimate_interval_rigid_evidence` (RANSAC → Cauchy IRLS Kabsch, cluster-jackknife covariance, per-camera 3D estimate and 2D re-projection refinement); `chained_intervals`; `estimate_shared_angular_axis`; `rotation_vs_translation_evidence`; `hinge_from_intervals` |
| `rotation_observability.py` | static-patch null (`calibrate_rotation_null`, rotation and translation); `support_configuration` (rank, condition, baseline) |
| `support_consistency.py` | candidate-independent S_core / M_proposal / U; second rigid group, moving base, tool proximity, coverage |
| `assembly_geometry.py` | static/moving interface (or ABSTAIN); graded hinge plausibility; guide evidence; implied minimum hinge rotation |
| `distinguishability.py` | R/P prediction separation (common translation removed); held-out block preference; complexity report |
| `selection_v2.py` | orchestration and the output tree of spec §10.3 |

**Modified files:**

| file | change |
|---|---|
| `physical_selection.py` | appended `PhysicalDecision`, `select_physical_joint`, `render_veto_allowed`. The existing Stage-E code is unchanged. |
| `config.py` | `SelectionV2Config` (`joint_discovery.selection_v2`); `comparison.selection` accepts `observable_physical_v2`; safety switches cannot be disabled. The legacy `to_dict` / `config_hash` is unchanged: the stored overnight hash `40c8872c5dcb` reproduces. |
| `cli.py` | `run_discovery` declares the block partition first when v2 is set and calls `run_selection_v2` with the same training fit; the FreeArtGS adapter follows the v2 status only, never the legacy label. New subcommand `select-v2` (`--from-run` / `--observations`), which always writes a new directory. |
| `synthetic.py` | optional `extra_rigid_group` (a second rotating rigid set). The default scene is bit-identical to HEAD (hash `a7dba6378bc0`). |

**Integration:**

- `pipelines/joint_aware_reconstruction.py`:
  - New stage `select` (between `discover` and `authority`), `--selection v2|legacy` (default v2).
  - `authority`: `IDENTIFIED_X` builds X only; `TYPE_UNOBSERVABLE` builds both; `MODEL_INCONSISTENT` and `NO_SIGNIFICANT_MOTION` stop (diagnostics via `--allow-inconsistent-diagnostics`).
  - `compare`: render evidence plus a controlled counterfactual; veto only (`--render-min-blocks 6`).
  - `final`: `PASSED` / `ABSTAIN` (working model named) / `BLOCKED`.
  - `export`: production only if `physics_joint_export_ready`, or a legacy run with `--assume-physical-joint "<note>"`.
- `tools/export_parttwin_handoff.py`: `physical_provenance` gate. `notes.physical_joint` records `identified`, `assumed` or `diagnostic`.
- `pipelines/joint_selection_v2_pilot.sh`: real-scene pilot with camera ablations.

## 2. Equations and covariance as implemented

- **Interval motion.** For tracks with 3D at both a and b (partial tracks allowed), minimise the weighted residual `Σ w_i ‖R X_i(a) + t − X_i(b)‖²`. RANSAC uses the threshold `max(1 cm, 4·median σ_i)`, and RANSAC outliers are excluded. This is followed by 8 Cauchy IRLS steps on `r_i/σ_i` with a robust re-estimated scale. Inliers are tested against the absolute RANSAC threshold. `σ_i² = σ_depth(z)² + 2(σ_px z/f)²` uses the calibrated discovery noise. Finite `log R` is exact, with no small-angle approximation.
- **Covariance of `log R`.** A delete-one-surface-cluster jackknife over 5 cm voxels, with at least 4 clusters. Duplicated tracks share a cluster, so they cannot shrink the covariance (test A6). `n_effective_tracks` is the Kish-effective number of clusters.
- **Shared axis.** `min_{u,Δθ_k} Σ_k ρ_softL1(‖L_kᵀ(ω_k − Δθ_k u)‖²)`, where `L_k L_kᵀ = Σ_k⁻¹`, with the closed-form `Δθ_k = uᵀΣ_k⁻¹ω_k / uᵀΣ_k⁻¹u`. Only intervals with `|ω_k|/sd ≥ 3` constrain u. The gauge is net `ΣΔθ_k ≥ 0`. The axis cone95 comes from a bootstrap over informative intervals.
- **Rotation evidence.** The net `ΣΔθ_k` is block-bootstrapped over intervals (time blocks). It is called *detected* only if the CI excludes 0 **and** `|net|` exceeds the calibrated null q95.
- **Null.** For each interval, 40 static-core patches matched to the moving count, from nearest neighbours of a random seed. Each is inflated by `max(1, moving / static robust residual)`, then one draw per interval is summed. The same applies to centroid translation. Limitation: static patches do not contain motion-induced tracker bias. The inflation is the only proxy for it.
- **Hinge second.** `t_k − (u·t_k)u = (I − R_k) B x` with `B = [e1 e2] ⊥ u`, solved by least squares and bootstrapped over rotating intervals. Line sd ≥ 5 cm gives HINGE_UNOBSERVABLE. The screw pitch `u·t_k` is reported.
- **Support.** `span = 2·p90‖X(t) − median X‖`. M: `span ≥ max(8 mm, 7σ_track)`. S: χ²/dof < 4 and `span < max(5σ, 5 mm)`. U: everything else. All on training times only.
- **Held-out preference.** Per-observation Student-t log-likelihood of R − P on the common M_proposal ∩ eligible set, summed per held-out block, block bootstrap, at least 4 blocks. Held-out `q` is the training-only interpolation, which is documented as a trajectory-interpolation benchmark.
- **Separation.** `δ = h_R − h_P` on identical observations. The median of δ per time removes the common translation in 3D. The per-block RMS uv separation is compared with the empirical residual σ of the fits. A diagonal I² is reported as descriptive only.

## 3. Tests

| suite | result |
|---|---|
| `tests/test_joint_physical_selection_v2.py` (new) | **34/34 OK**, about 6.5 min (5 end-to-end synthetic runs) |
| all joint tests (`test_joint_*.py`, unittest) | **143/143 OK**, against the baseline of 109/109 before the change (`results_18_11_26/joint_selection_v2/baseline/`) |
| `tests/test_joint_aware_integration.py` | 11/11 OK |

**Spec §11 coverage:**

| gate | implemented | not implemented (reason) |
|---|---|---|
| A | A1–A9 | A10 correlated depth-edge bias, A11 informative 2D camera vs grazing-depth camera. The per-camera 2D refinement exists but has no test. |
| B | B12–B16, plus a guide test | B17 tracker-induced spurious rotation on a real P |
| C | C18–C23, C25, C27 (via B12), the `PhysicalDecision` invariant, and three hierarchy-branch tests | C24, C26 as separate monotonicity tests |
| D | D28, D29, D30/D33 (veto logic), and the partition test | D31 and D34 render-side unit tests: the controlled counterfactual ran only in the real pilot below |
| E | E36, E38 (G8 gate), E39 (legacy default, no v2 output), v2 refuses fold overrides | E35, E37 (TCP/slip physics not implemented, see §5); E40 is covered by the existing SE(3) / multiview tests |

**Two estimator bugs found by the tests and fixed:**
1. A per-frame χ² threshold lost short arcs: an 8° lid became NO_SIGNIFICANT_MOTION.
2. An IRLS-scale inlier test absorbed a second rigid group.

## 4. Real-scene pilot

Outputs are in `results_18_11_26/joint_selection_v2/pilot/<scene>/<cams>/`. Inputs are the 2026-11-17 discovery tracks (tool-excluded) with the same config, `real_tcp2cm`. Source runs were not modified.

| scene | legacy | v2 cam001+002 | working | ready | cam001 only | cam002 only | +cam000 (diag) |
|---|---|---|---|---|---|---|---|
| red fridge | revolute | **IDENTIFIED_REVOLUTE** | revolute | **yes** | IDENTIFIED_REVOLUTE | TYPE_UNOBSERVABLE (hinge sd 11 cm) | – |
| white drawer | ambiguous | **TYPE_UNOBSERVABLE** | prismatic | no | TYPE_UNOBSERVABLE | TYPE_UNOBSERVABLE | – |
| black box | ambiguous | **TYPE_UNOBSERVABLE** | prismatic | no | TYPE_UNOBSERVABLE | TYPE_UNOBSERVABLE | TYPE_UNOBSERVABLE |
| robot dog | ambiguous | **MODEL_INCONSISTENT** | none | no | MODEL_INCONSISTENT | MODEL_INCONSISTENT | – |
| orange suitcase | ambiguous | **TYPE_UNOBSERVABLE** | prismatic | no | TYPE_UNOBSERVABLE | TYPE_UNOBSERVABLE | MODEL_INCONSISTENT |

**Evidence (cam_001 + cam_002 unless stated):**

| scene | hinge-free net rotation (95% block CI), null q95 | axis cone95; per-camera | moving / static residual | hinge from rotation | held-out blocks |
|---|---|---|---|---|---|
| fridge | **17.2° [5.9, 29.5]**, null 1.9° | 3.9°; cam1 19.8° (4.2° off the pooled axis), cam2 13.4° (1.8° off) | 2.5 / 0.85 mm | OK, sd 3.9 cm, pitch ≤ 4 mm; interface WEAK (d10 6.8 cm) | R preferred, 5 blocks |
| drawer | 1.3° [0.1, 2.5], null 0.13°: detected but the axis is unstable | **37.5°**; cam1 / cam2 axes 17° / 28° off the pooled axis | 1.35 / 0.92 mm | unobservable | R preferred, 6 blocks |
| black box | no interval rotation reaches 3σ, so no shared axis | –; cam1 4.9°, cam2 5.3° (inconsistent axes) | **6.3 / 2.1 mm (3.0×)** | – | R preferred, 5 blocks |
| black box + cam000 | 4.0° [3.0, 5.0], null 1.9° | **20.8°**; cam_000 alone 13.8°, cam1 5.3°, cam2 5.1° | 5.4 / 2.9 mm | unobservable (sd 8 cm) | R preferred |
| dog | 1.2° [−2.0, 4.5], null 12.5° | –; cam1 axis 73° off | 1.9 / 0.86 mm | – | none |
| suitcase | 8.1° [−20, 52], null 89° | 9.6°; cam1 14.0°, cam2 5.4° | **9.5 / 0.88 mm (10.8×)** | sd 27 cm | R preferred, 6 blocks |

**Per-scene diagnosis:**

- **Fridge.** A genuine rotational field. Rotation-only estimates (17.2°, cam_001 alone 19.8°) agree with the discovery fit (17.8°) and the fitted axis. The hinge line comes out of the rotation itself and lies near the observed static/moving interface. This is the only physics-ready joint. cam_002 alone measures the rotation (13.4°) but cannot place the hinge, so it gets TYPE_UNOBSERVABLE: weak geometry, which is correct behaviour.
- **Drawer.** The motion is a 16 cm translation with 1–1.5° of rotation whose axis wanders across intervals (cone 37°) and between cameras. That is play or wobble, not a hinge. The 9 m remote-hinge revolute is not certified. The working model is prismatic. A physical slider is not certified either: the guide rails are not observable (the interface is the drawer-front gap), so `IDENTIFIED_PRISMATIC` correctly does not fire. The legacy render comparison had wrongly "identified" a revolute.
- **Black box (spec regression case).** The output is **not prismatic**. The lid's moving support carries 3× the static tracking noise (`UNRELIABLE_MOVING_SUPPORT`). No interval rotation reaches 3σ, and the two cameras disagree on the axis.
  - cam_000 alone sees 13.8°, close to the independent 15–21° audit. cam_001 and cam_002 see about 5°.
  - The diagnosis is **"rotation not observed"** (bad lid support and views), not "observably absent rotation". The reason recorded in `joint_selection_v2.json` states exactly this.
  - Held-out blocks prefer the revolute fit in every variant, but without a stable axis or hinge that cannot confer a type.
- **Robot dog.** **MODEL_INCONSISTENT**: 51–60% of the moving-proposal observations lie within 5 cm of the measured tool shaft, in every camera combination. The "moving part" is largely the gripper and its immediate surroundings. This matches the earlier finding that its fitted motion follows the TCP path (3° direction difference). No single articulated part is observed.
- **Orange suitecase.** Only 395 of 27,151 tracks move, and their rigid residual is 10.8× the static noise. The net rotation is not distinguishable from the null (CI −20…52°). With cam_000, 24% of the moving observations are near the tool, giving MODEL_INCONSISTENT. The large legacy "15° rotation" is not supported by a coherent rigid field. Moving-support segmentation and tool contamination have to be fixed before any type claim.

**Render evidence (Gate 5 pilot).** The controller ran `select, compare, final` on the existing coarse runs. The legacy `selection.json` and `run_manifest.json` were backed up as `selection.legacy_render_compare.json` / `run_manifest.pre_v2.json`.

- **Independent held-out blocks:** 2 of the required 6, so the result is `NOT_IDENTIFIABLE_FROM_AVAILABLE_BLOCKS`. There is no veto (1 seed).
- **Controlled counterfactual (fridge):** the revolute-trained Gaussians fit better under the revolute motion (−2.0 mm depth MAE). The prismatic-trained Gaussians fit better under the prismatic motion (+2.1 mm). **Each geometry prefers the motion it was trained with.** So render differences between independently trained branches measure geometry–motion co-adaptation, not joint correctness.
- **Controlled counterfactual (black box):** ±0.3–0.4 mm.
- **Final stage:** fridge → `PASSED` (physics-ready); black box → `ABSTAIN` (working prismatic, approximation).

**Existing assets.** The fridge production asset exported earlier today came from the legacy selection. It is flagged by a sidecar file, `LEGACY_SELECTION_NOT_PHYSICALLY_CERTIFIED.json`. The asset files themselves are unchanged.

## 5. Not implemented / limitations (no fabricated results)

- **TCP/wrench physics (§9).** Only the gate report exists: contact and force both ABSTAIN, because none of the scenes has annotated contact or a verified wrench calibration. The slip / passivity / Coulomb hypothesis comparison is NOT_RUN.
- **Repeated-seed render training (§8.2).** NOT_RUN: one seed per branch. Render evidence therefore cannot veto by design.
- **Extrapolated endpoint held-out blocks (§6.2).** Not added to v2. The existing `cli folds --kinds forward_block` remains available as a separate diagnostic.
- **Null calibration.** Static patches lack motion-induced tracker bias; only the residual-ratio inflation stands in for it. Thresholds are declared defaults checked on synthetic controls only. The synthetic false-positive / false-negative maps of §5.2 were not computed.
- **Support noise.** The real-data static-core depth scale (≈ 0.12–0.23 of the prior) is clamped by the existing calibration. v2 uses the prior σ for the support threshold. The measured σ marks ~2/3 of the black-box tracks as moving, because the static core is biased toward the quietest tracks.
- **Assembly.** The interface comes from sparse tracks at t_ref. Dense-geometry attachment regions and occluded bearings are not modelled; a single camera gives ABSTAIN.
- **Image-space refinement.** Computed per camera and per interval (`per_camera_consistency.json`), but used as a diagnostic only. It does not enter the decision.
- **Dog and suitcase.** Need tool-free moving-support segmentation, and for the suitcase better lid tracking, before the joint type can be observed. The black box needs reliable lid tracks or depth: cam_000 alone is the only view that sees the opening.

## 6. Commands

```bash
python -m unittest tests.test_joint_physical_selection_v2                 # 34 tests, ~6.5 min
bash pipelines/joint_selection_v2_pilot.sh                                # five scenes + camera ablations (~1 h)
# full joint-aware pipeline (v2 is the default selection)
bash pipelines/joint_aware_real_scenes.sh full
```
