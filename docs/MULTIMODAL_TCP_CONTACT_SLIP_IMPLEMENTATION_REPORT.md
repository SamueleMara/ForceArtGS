# Multimodal TCP contact-mode frontend: implementation report (2026-11-19)

**Spec:** `FORCEARTGS_MULTIMODAL_TCP_FORCE_SLIP_MODES_CODEX_IMPLEMENTATION.md`
**Design:** `docs/MULTIMODAL_TCP_CONTACT_SLIP_DESIGN.md`
**Repository:** branch `multimodal-tcp-contact`, from `main` at `50bfdb4`. The submodule `reconstruction/nerfstudio`
stays at `e874a4e`, **unchanged** (no Nerfstudio edit was needed).

## Changed call graph

| file | change |
|---|---|
| `contact_modes.py` (new) | segments, `MovingSurface`, `ContactRows` (the `L_TCP` hook), diagnostics, discrimination, contact-only fit |
| `wrench_physics.py` (extended) | force gate, signed alignment, kinetic / static / mechanical-model evidence, `ForceDirectionRows`, `CombinedRows` |
| `tcp_support.py` (new) | tool occupancy, contact / proposal / anchor seeds, support update, pixel seeds, overlays |
| `track_refinement.py` (new) | `SceneImageSource`, `recover_tracks` (2D, both predictions), `calibrate_matcher` |
| `multimodal_frontend.py` (new) | `Frontend` (rounds, accept / rollback, final selection, all-frame refit, discovery-format export) |

**Edits to existing modules:**

| file | edit |
|---|---|
| `geometry_fit.py` | `extra.row_q_deps` sparsity; per-observation `uv_sigma_scale` |
| `observations.py` | optional `uv_sigma_scale` |
| `visual_losses.py` | honours `uv_sigma_scale` |
| `support_consistency.py` | capsule occupancy test |
| `selection_v2.py` | support / robot-evidence overrides; new report fields |
| `physical_selection.py` | consumes contact / force evidence |
| `dense_support.py` | `extra_seed_fn` |
| `config.py` | `multimodal_frontend` section, excluded from the hash when disabled: legacy `40c8872c5dcb` and the v2 pilot hashes reproduce |
| `cli.py` | `multimodal-refine` |
| `synthetic.py` | TCP / contact fixtures, textured splat images |

**Pipeline edits:**
- `pipelines/joint_aware_reconstruction.py`: stage `multimodal_refine`; `select` / `authority` / `dense` consume the
  accepted state; flags `--multimodal-frontend --contact-segments --mm-max-iterations`.
- `pipelines/joint_aware_real_scenes.sh`: `MULTIMODAL=1`.
- `tools/export_parttwin_handoff.py`: multimodal provenance in the handoff notes.

**Configs:**
- `configs/joint_discovery_multimodal.example.yaml`
- `configs/contact_intervals/EXAMPLE.yaml`, plus one per real scene. Real-scene contact segments come from **force
  peaks + visual evidence** (`samuele/estimate_contact_segments.py`):
  - **Force:** the fork's `export_force_contact_annotations.py` (sustained 6σ force contact above the baseline tare).
  - **Visual:** the part moves while the TCP is within 5 cm of the observed object depth (D1 audit timeline).
  - **Combination:** force OR visual, with ≤ 2-frame gaps filled and runs of ≥ 3 frames kept. The slip mode is
    **sliding** (the operator stated on 2026-11-19 that every demo slides, with an almost linear TCP path). The
    operator validated the estimate, so the segments are recorded as `contact_confirmed` / `operator_label`.
  - **Resulting segments (RGB-D frames):** fridge 110–148, black box 70–128, robot dog 80–130, suitcase 62–114.
  - **Drawer:** none (no robot archive, no wrench frame declared).
  - Per-frame audits and the force exports are in `results_19_11_26/contact_annotations/<scene>/`. Sigmas are null,
    so results carry UNVERIFIED_PRIOR_SCALE.

**Equations as implemented:** see the design doc §1–2. The anchor is **analytically eliminated** (variable projection,
exact for rigid motion and isotropic TCP noise), one per segment, from training times only.

## Tests

`python -m unittest discover -s tests -p "test_joint_*.py"`: **183 OK** (about 15 min). The 143 tests that existed
before this work all pass. New suites:

| suite | tests | covers |
|---|---|---|
| `test_joint_contact_modes.py` | 12 | A01–A12 |
| `test_joint_sticking_force.py` | 9 | B01–B10 (B02 and B09 in one test) |
| `test_joint_tcp_support_refinement.py` | 6 | C01, C03, C07, C08 (unknown contact gives proposals), C10, NCC recovery |
| `test_joint_multimodal_frontend.py` | 8 | C04, C05, C06, C09, C11, C12, D01, D06 (remote short arc) |
| `test_joint_multimodal_integration.py` | 5 | D02 (+D03 virtual cameras), D04, D06, D08, D09 |

**Findings from the tests (fixed in the code):**
- **A04.** On a 15° arc from a single contact point the hinge radius is only weakly observable (sagitta ~1.3 mm vs
  0.5 mm noise). The fit reports an axis sd of about 13° that covers its 11° error. At 40° the axis error is 1.6°.
- **Force nulls.** A constant force along a slider is preserved under time shuffling, so time nulls now bind only when
  the favoured hypothesis predicts a time-varying direction. Information-free nulls (sign-reversed, gravity, random
  constant and per-frame directions) always bind.
- **Tool occupancy.** The ambiguity band runs alongside the shaft only. Object points beside the tip stay CLEAR; the
  old 5 cm rule removed them.
- **Track recovery:**
  - Recovered depth (±1.5 cm gate) corrupted anchors (0.4 mm became 11.8 mm), so recovery is now 2D-only by default.
  - The residual-based uncertainty was circular, so it is now independently calibrated by re-finding raw tracker
    observations (matcher σ ≈ 0.64 px, scale ≈ 2.1).
- **Acceptance rule.** "No hypothesis may worsen" was wrong: the rejected hypothesis is expected to fit worse with
  better data. The rule now uses the best candidate on identical held-out observations.
- **A declared mode contradicted by every candidate** (C06): its rows are dropped from the fit (the hinge stays within
  5° of the truth), the record `CONTACT_MODE_CONTRADICTED` is kept, and the result is not export-ready.

**"Beyond one-pass" (spec DoD 7):**
- **C05 (strong):** a hidden-guide drawer with a confirmed hard stick and verified force under a declared model becomes
  **IDENTIFIED_PRISMATIC**. The same data without the robot is TYPE_UNOBSERVABLE.
- **Track recovery (C04, small):** a held-out improvement of about 0.6% under 80% track loss. Ground-truth axis error
  is unchanged at the 0.1° level, so track recovery is shown to be safe, not a large gain.
- **10° short arc, two seeds:** TCP-aware support turned TYPE_UNOBSERVABLE into a correct IDENTIFIED_REVOLUTE for one
  seed and changed nothing for the other. Hinge accuracy was unchanged. Not asserted in a test.

## Real data

**Inputs available:**
- No scene has a verified wrench calibration, so force is ABSTAIN everywhere.
- The drawer has no robot archive, so it runs RGB-D iterative only.
- The other four scenes use the operator's sliding annotation. Sliding rows exist only where the observed moving
  surface supports the TCP contact for both candidates.

The frontend-only pilot (`pipelines/multimodal_frontend_pilot.sh`) was stopped before completion in favour of the
overnight full pipeline (`MULTIMODAL=1 ... full`). Its results and the real-scene comparison table are still **to be
filled from that run**.

## Not implemented / limits

- `rigid_grasp`: rejected at parse time. There is no SE(3) grasp solver and no verified grasp transform.
- **Dense masks:** produced by the controller `dense` stage from the accepted authority plus TCP pixel seeds (one
  recorded final update), not inside the frontend loop. The frontend loop updates track-level support only.
- **Tool geometry:** a capsule with an approximate radius; no measured tool mesh.
- **Contact annotations:** the real-scene sliding annotations assume contact throughout each window. They must be
  edited if the tip lost contact.
- **Real-data force evidence:** not possible until the wrench gates (sign, tare, gravity, clock) are verified.
- **Track recovery** neither replaces bad raw observations nor recovers tracks with no 3D anchor.
- **Spec E ablations V3–V5** (stick / force) cannot run on the real data: there is no sticking demonstration and no
  verified wrench.

## Export update (2026-11-19, full-budget fridge)

- **Quaternion cross-check.** After 30k iterations the checkpoint quaternions are far from unit norm (0.34–2.4), and
  ns-export writes them normalised. The producer's cross-check now compares rotations (normalised, q ≡ −q): 7.7e-8.
  The output PLYs keep the exact checkpoint values.
- **Soft membership split** (operator decision). On the fridge, 7,540 of 387,467 Gaussians have w in (0.01, 0.99).
  Instead of abstaining, the producer writes them to BOTH PLYs with opacity logit(α·p_part), p = (1−w, w): the
  two-part softmax. This is exactly the splatfacto-art renderer, which draws α(1−w) static plus α·w moving copies.
  Verified opacity error is 4.3e-7, and hard Gaussians are byte-identical in every field except the quaternion
  normalisation. Lineage: the moving copy gets uid N + row and parent_uid = row (PArtTwin `parent_uid` semantics).
  The record is in handoff `notes.membership_split`.
- **Fridge full:** export PASSED as a diagnostic asset, `BLOCKED_MISSING_PHYSICS`, visual bitwise check true.
- **Discrete part ids in the final phase** (operator decision). Densification already copied `part_weights` from the
  parent. The soft weights came from TRAINING: on the fridge, blend 7.45% → art 1.95% intermediate. Changes:
  - `splatfacto-art --pipeline.model.hard-part-ids True` snaps the blend weights to the two-part argmax and freezes
    them, so clones and splits inherit the parent id.
  - Fixed a latent bug: densify/cull rebuild parameters as `nn.Parameter` (`requires_grad=True`), which silently
    unfroze `trainable_weights=False`. Part weights are now re-frozen after every refinement.
  - The controller's art stage passes `--pipeline.model.hard-part-ids True --pipeline.model.trainable-weights False`.
  - This is a submodule change (`reconstruction/nerfstudio`, `models/splatfacto_art.py`); commit and push it to
    `nerfstudio-local` first, then update the pointer.
  - In the 2026-11-19 overnight batch, the fridge and the black box art stage ran before this change (soft weights,
    exported with the soft split). The robot dog, suitcase and drawer use hard ids.
