# Force-guided joint selection (3D force only) -- implementation report

Spec: `FORCEARTGS_CODEX_FORCE_GUIDED_JOINT_SELECTION.md`, 2026-11-26. **No commit / push.** No torque, no full wrench, no
PhysX / dynamics / fitting in the selector. No running job was killed by this work; no historical file overwritten.

## 1. Code state

Branch `multimodal-tcp-contact`, HEAD `58dc613e6b287394f65ca122c86b573608b9dce1`, dirty worktree (this change plus the
earlier uncommitted rest-to-rest / q-extension work).

| file | change (current line numbers) |
|---|---|
| `estimotion_2part/joint_discovery/physical_selection.py` | `select_physical_joint(..., legacy_r_first=False)` (339-341); `force_guided` channel read (377-379); new `r_failed_assess_p` (381-433): a failed revolute geometry evaluates P on the same evidence; the three early returns replaced (470-495, legacy behaviour kept behind `legacy_r_first=True`); `FAVOURS_PRISMATIC` against an identified revolute withholds export readiness (`FORCE_DISAGREES`, ~524), never relabels |
| `estimotion_2part/joint_discovery/force_guided_selection.py` (new, 437 lines) | 3D-force source (wrench[:, 0:3] only, `force_convention.sensor_corr_to_object_base` once, exact source-row join), Section 3 evidence (reuses `wrench_physics.decompose_force`, `material_velocity_at_tcp`, `tcp_contact.joint_rate / tcp_velocity`), held-out / null tests, cached re-selection B0-B3, artifacts |
| `estimotion_2part/joint_discovery/cli.py` | `force-guided-select` command |
| `pipelines/joint_aware_reconstruction.py` | `--force-guided-selection` (st_select -> `force_guided_selection()`, 335-360, append-only `<job>/joint_selection_force_guided/vNNN`, reused when inputs are identical); `--working-branch-only` (`Run.branches` 147-157, `st_authority` 391-395): only the evidence-based working branch is built; an existing identical authority is kept and its label change recorded (`authority_label_changes`, ~414) instead of being overwritten |
| `configs/joint_discovery.force_guided_selection.yaml` (new) | strict config; torque / extra tare / contact inference cannot be enabled |
| `tests/test_joint_force_guided_selection.py`, `tests/test_joint_force_guided_selector.py`, `tests/force_guided_fixtures.py` (new) | spec tests 1-19 |
| `tests/test_joint_physical_selection_v2.py` | two GateC tests encoded the removed early returns; they now assert the corrected behaviour AND that `legacy_r_first=True` reproduces the old verdict |

A first implementation of the previous spec (full six-axis wrench + dynamic replay) was written in this session and then
REMOVED (never committed): it violates this spec's scope.

## 2. Force path, frames, time

`F_obj^B = s R_base_ee R_ee_sensor F_sensor` with `F_sensor = robot_archive.wrench[:, 0:3]` (already tared at the start,
no new tare / bias), `R_ee_sensor` from `robot_tool.json` (+ `configs/robot_tool_overrides.json` for the white drawer's
bare-sensor settings), `s` from `contact_force_convention.json`. Camera world = `base_link` (asserted from
`metadata/cameras.json`). Rows joined to visual frames by exact `source_row == source_frame_index` (no interpolation).
The canonical robot record's `F_object_base_N` is NOT used (its software baseline would be a second tare). Direction
definedness: |F| > max(0.05 N, 3 x measured noise level) and speed > 2 mm/s -- an undefined direction, never "no contact".

## 3. Why the dog was MODEL_INCONSISTENT and what changed

The hinge-free rotation field detected a rotation (dog 25_11: 46.9 deg, CI [15.9, 82.5], null q95 13.5 deg), sending the
selector into the R-only branch; the fitted revolute hinge lies 2.33 m from the observed interface (object half-diagonal
0.336 m: 6.9x), verdict CONTRADICTED -> immediate `MODEL_INCONSISTENT/none` without ever scoring P. R/P predictions are
INDISTINGUISHABLE (1.57 sigma) and the held-out blocks have no preference. Corrected: the revolute GEOMETRY is rejected
(`HINGE_ASSEMBLY_CONTRADICTED`, `REMOTE_HINGE_DEGENERACY` object-relative), P is evaluated on the same evidence and kept as
the working model (`TYPE_UNOBSERVABLE / prismatic`, export false). `MODEL_INCONSISTENT` remains when the held-out blocks
decisively prefer R with separable predictions (or the 3D force favours R) while R is physically impossible, and for
two rigid groups.

## 4. Real cached comparison (B0 old, B1 corrected visual, B2 + force diagnostics, B3 force-guided)

Outputs: `/media/samuelemara/Extreme SSD/Samuele/results_26_11_26/joint_selection_force_guided_audit/<scene>_<run>/`
(+ `summary_all.md`). The white drawer pairs its 19_11 cached visual evidence with the 25_11 robot archive (same raw
recording, bare-sensor override recorded in `force_source_contract.json`); its 25_11 discovery has not run yet.

| Scene | R hinge distance / half-diag | Rotation vs null | Held-out P vs R | F-EE | F-P | F-R | Force verdict | B0 | B3 (= B1 = B2) | export |
|---|---|---|---|---|---|---|---|---|---|---|
| robot dog (25_11) | 2.33 / 0.336 m CONTRADICTED | 46.9 deg [15.9, 82.5] vs 13.5 | none | 7.6 deg | 9.2 deg | 8.9 deg | NONDISCRIMINATIVE (P/R directions 0.86 vs 1.41 deg unc.) | MODEL_INCONSISTENT/none | **TYPE_UNOBSERVABLE/prismatic** | false |
| robot dog (19_11) | 2.17 / 0.341 m CONTRADICTED | 33.3 deg [10.1, 61.8] vs 11.1 | none | 6.6 | 7.8 | 7.6 | NONDISCRIMINATIVE (0.73 vs 1.35) | MODEL_INCONSISTENT/none | **TYPE_UNOBSERVABLE/prismatic** | false |
| white drawer (19_11) | 8.79 / 0.459 m CONTRADICTED | 1.31 deg [0.14, 2.53] vs 0.13 | revolute | 19.1 | 24.3 | 23.5 | NONDISCRIMINATIVE (0.59 vs 4.14) | TYPE_UNOBSERVABLE/prismatic | TYPE_UNOBSERVABLE/prismatic | false |
| red fridge (19_11) | 0.073 / 0.365 m WEAK | 17.2 deg [5.9, 29.5] vs 1.91 | revolute | 86.0 | 88.4 | 87.4 | NONDISCRIMINATIVE (5.1 vs 14.3) | IDENTIFIED_REVOLUTE | IDENTIFIED_REVOLUTE | false (UNVERIFIED_PRIOR_SCALE, unchanged) |
| suitcase (19_11) | 0.722 / 0.341 m CONTRADICTED | 18.9 deg [-12.3, 66.6] vs 34.9 | revolute | 15.1 | 23.2 | 24.8 | NONDISCRIMINATIVE (held-out diff inside CI / null) | TYPE_UNOBSERVABLE/prismatic | same | false |
| black box (19_11) | 1.21 / 0.209 m CONTRADICTED | n/a vs 5.95 | revolute | 72.0 | 86.5 | 86.6 | NONDISCRIMINATIVE (0.61 vs 6.76) | TYPE_UNOBSERVABLE/prismatic | same | false |

* The force never changed a decision beyond B1 (`force_changed_decision_beyond_B1 = false` everywhere; B2 == B1). The dog
  is fixed by the corrected geometric / visual logic, NOT by force.
* Why force does not discriminate: at the TCP a remote hinge and a slide predict the same linear direction (dog 0.86 deg
  apart vs 1.41 deg force-direction uncertainty). On the fridge the measured force is nearly perpendicular to the EE
  motion (86 deg) -- reported, not tuned (no sign / frame change is allowed here).
* Revolute control preserved: the fridge stays IDENTIFIED_REVOLUTE.

## 5. Tests

`tests/test_joint_force_guided_selection.py` + `tests/test_joint_force_guided_selector.py`: 21 passed (incl. real-dog tests:
single conversion from the force columns, scrambled torque columns change nothing, append-only, FORCE_UNAVAILABLE fallback).
Targeted regressions (`test_joint_physical_selection_v2`, `test_joint_discovery_force_selection`,
`test_joint_discovery_end_to_end`, `test_joint_multimodal_integration`, `test_joint_multimodal_frontend`,
`test_joint_discovery_remote_hinge_observability`, `test_joint_discovery_observability`, `test_robot_tool_overrides`,
`test_frozen_joint_extension`): 94 passed + the 2 updated GateC tests (11/11 GateC pass after the update). The full
G9/G10 suite was not rerun for this change (the selector is not used by G9/G10).

## 6. Real run state

`select` + `authority` were run once on the 25_11 dog run with the earlier (removed) force-informed flag; that wrote
`joint_selection_force/v000` (append-only, now obsolete) and kept the prismatic authority (label change recorded). The
next pipeline invocation with `--force-guided-selection` writes `joint_selection_force_guided/v000` and points the run's
selection at it.
