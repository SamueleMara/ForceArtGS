# G9 dynamics: repository and data audit (2026-11-20)

**Spec:** `FORCEARTGS_G9_GENERAL_DYNAMICS_FULL_INERTIA_PARTTWIN_CODEX.md`. This audit was written before and during the
implementation. The implementation report is `docs/G9_GENERAL_DYNAMICS_IMPLEMENTATION_REPORT.md`.

## 1. Repositories and revisions

| repository | local path | revision | state |
|---|---|---|---|
| ForceArtGS | `~/ForceArtGS` | `628e3ee` (`main` = `multimodal-tcp-contact` = `origin`) | G9 work uncommitted (no commit without the user's request) |
| ForceArtGS-Physics | `~/ForceArtGS-Physics` (cloned 2026-11-20 from `andrearichichi/ForceArtGS-Physics`) | `f3824316272d` = tag `v0.1.0` = spec pin | clean; **never edited**; not vendored |
| PArtTwin1D | `~/PArtTwin1D` | branch `shared-asset-export` at `713ffe2` | **builder uncommitted**: `src/parttwin1d/assets/sim_export/`, `cli/build_asset.py`, `tests/test_handoff_build_asset.py` are untracked |
| RigEuler3DGS | `~/Joint3DGS/RigEuler3DGS` | checkout `main` `5c1a260`; `origin/phase5-sim-asset-export` = `8464d4f` (spec pin) | read only |
| nerfstudio submodule | `reconstruction/nerfstudio` | `nerfstudio-local` (hard part ids) | unchanged by G9 |

**One canonical exporter.** The PArtTwin1D builder's `schema.py`, `physics.py` and `collision.py` are byte-identical to
RigEuler `8464d4f` `recovery/sim_asset_export/`. Its URDF, MJCF and USD writers and `validate.py` are the
producer-neutral adaptations used by G8. G9 writes **no** simulator file: it translates its result into the
builder's existing physics spec (schema_version 1) and calls `parttwin1d.cli.build_asset`.

**Blocker (reproducibility).** The shared builder exists only as untracked files in the local PArtTwin1D checkout,
and the public `main` (`b67863a`) does not have it. Until it is committed and pushed, the G9 final export is
reproducible only on this workstation. Every asset records the PArtTwin HEAD, branch and dirty count
(`final_physics_export.json` → `build.parttwin`).

## 2. Environments

| env | used for | key versions |
|---|---|---|
| `freeartgs` (conda) | ForceArtGS, G9 fitting, frozen Physics baselines | Python 3.10.20, numpy 1.26.0, scipy 1.15.3, torch 2.5.1+cu124, pyarrow 25.0.1, plyfile |
| `parttwin_sim` (conda) | PArtTwin1D builder, SAPIEN / ManiSkill checks and rollouts | Python 3.11.18, numpy 2.4.6, sapien 3.0.3, mani_skill 3.0.1 |
| `~/.venvs/rigeuler-sim-export` | MuJoCo and OpenUSD (builder runtime + G9 MuJoCo rollouts) | Python 3.11.15, mujoco 3.3.5, numpy 2.2.6, pxr 0.25.5 |
| Isaac Sim | — | **not installed**: PhysX numerical parity in Isaac is NOT_RUN |

- **Physics import.** No env was modified. `forceartgs_g9.physics_lib.load()` puts `<checkout>/src` on `sys.path` and
  refuses any HEAD other than `f382431` or a dirty tree, unless `allow_unpinned` is passed (which is recorded).
- **Physics baseline tests.** pytest is absent from `freeartgs` and pip has no index access. The baseline used a
  scratch copy of pytest 9.2.0 from `parttwin_sim` on `PYTHONPATH`. Result: **75 passed, 66 skipped**.
- **Skipped tests.** They need data that is not on this machine:
  - `FORCEARTGS_PHYSICS_BENCHMARK` (ForceSAPIEN bundle, not redistributed): the A3 → B1–B10 reproduction is
    **NOT_RUN** (only a visualizer export exists in `~/Downloads`);
  - Hoi! force data;
  - `FORCEARTGS_PHYSICS_ORIGINAL_ROOT` (provenance check).

## 3. Licensing

The ForceArtGS-Physics `LICENSE` and `NOTICE` describe a private research handoff. No license to redistribute or
publish exists, and AKF_PINN has no declared license. G9 therefore:
- imports the pinned checkout;
- copies none of its source;
- never touches `_validated/`;
- adds nothing of it to the ForceArtGS repository.

The new general models live in ForceArtGS (`forceartgs_g9/`) as the "thin callable plugin" of spec §6.1, because
there is no edit right on the Physics repository.

## 4. ForceArtGS integration points (audited)

- **Controller.** `pipelines/joint_aware_reconstruction.py` stages:
  `audit … art, refine (NOT_RUN), compare, video, final, export`.
  - G9 adds `dynamics` and `final_physics_export` after `export` (= G8 visual handoff).
  - Both are NOT_RUN unless `--dynamics-config` is given.
  - `multimodal_refine` (TCP / force-informed joint selection) is consumed, not reimplemented.
- **Kinematics:** `<job>_<kind>/joint_authority/v000/{authority.json, joint_states.npz}`, loaded with
  `JointAuthority.load` (schema + sha256). The legacy `estimated_joint.json` and the blend frame B are never used.
- **G8 handoff:** `<run>/parttwin_handoff/{accepted|diagnostic_<kind>}/articulated_3dgs_handoff.json`, with the
  Gaussians in B (`T_Wc_P = N_BW⁻¹`) and the joint in Wc. The builder works in the PLY frame. G9 mass integrals
  are in Wc at the anchor frame, which is what physics spec v1 `*_capture_*` fields mean.
- **Force:**
  - robot archive `T_base_ee`, `T_base_tcp`, `wrench`, `source_row` (50 Hz), joined exactly on `source_frame_index`;
  - `WrenchCalibration.from_scene_metadata(...).readiness()`;
  - `metadata/{robot_tool.json, contact_force_convention.json, robot_base_registration.json}`;
  - contact segments `configs/contact_intervals/<scene>.yaml`.

## 5. Real-data evidence per scene (results_19_11_26 overnight batch, hard part ids)

| scene | selection v2 | robot archive | force gates | gravity registration | mass-model mesh |
|---|---|---|---|---|---|
| red fridge | IDENTIFIED_REVOLUTE, not export-ready (UNVERIFIED_PRIOR_SCALE) | yes | 6 UNVERIFIED (rotation, sign, TCP, tare, tool gravity, clock) + registration UNVERIFIED; contact intervals PASS (operator label) | none | none |
| black box | TYPE_UNOBSERVABLE (UNRELIABLE_MOVING_SUPPORT) | yes | same | none | none |
| orange suitcase | TYPE_UNOBSERVABLE (UNRELIABLE_MOVING_SUPPORT) | yes | same | none | none |
| robot dog | MODEL_INCONSISTENT (authority FAILED, no `selection.json`) | yes | not evaluated (no kinematics) | none | none |
| white drawer | TYPE_UNOBSERVABLE (camera axis disagreement) | **no** | NO_FORCE | none | none |

**What the evidence actually is:**
- The sensor axes (`R_ee_force_sensor = I`), sign (−1, operator confirmation) and TCP offset (0.1876 m) are
  **declarations**.
- The clock is `source_frame_index / 50`.
- The robot-base registration is `OPERATOR_APPLIED_ROBOT_REGISTRATION_UNVERIFIED`.

None of these is promoted to verified hardware calibration.

## 6. Unresolved blockers (owner action needed)

1. **Measured wrench calibration** of the real recordings: per-gate evidence for sensor rotation, sign, TCP, tare,
   tool gravity and clock offset, as a calibration record (`--force-calibration`, see the report §5).
2. **Measured gravity registration** of `base_link` (G9 config `model.gravity.calibration_file`).
3. A **validated watertight / shell moving-part mesh** per object (mass-model geometry), plus collision meshes,
   physical joint stops, velocity / effort limits and the static-link inertial (`forceartgs.g9.mechanical_spec/v1`).
4. **Independent validation demonstrations** (spec target: 1 + 10 per object) with vision-only q.
5. **Commit and push** the PArtTwin1D shared builder.
6. The ForceSAPIEN benchmark bundle, to rerun A3 → B1–B10 with the frozen methods.
