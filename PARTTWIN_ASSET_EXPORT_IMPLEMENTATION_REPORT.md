# G8: articulated-3DGS simulation asset export (ForceArtGS → shared PArtTwin1D builder) — 2026-11-18

Spec: `FORCEARTGS_RIGEULER_PARTTWIN1D_ASSET_EXPORT_CODEX_SPEC.md`. **Nothing is committed or pushed in any repository.**

## Architecture as built

```
ForceArtGS joint-aware run ──tools/export_parttwin_handoff.py──> articulated3dgs.handoff/v1
                                                                     │
PArtTwin1D (branch shared-asset-export):  python -m parttwin1d.cli.build_asset --handoff …
   src/parttwin1d/assets/sim_export/  = RigEuler3DGS recovery/sim_asset_export (b1ec149), moved here unchanged except:
      export.export_sim_asset(inp, …)   producer-neutral core (was export_phase5_sim_asset; load_phase5 removed)
      handoff.py                        schema, validation, joint re-expressed in the PLY frame, bitwise check
      validate.find_runtime             also reads PARTTWIN_{SIM,USD,MUJOCO,SAPIEN}_PYTHON (RIGEULER_* kept)
```

One shared exporter, as the spec requires: the URDF/USD/MJCF writers, the PLY split, the frames, the physics
schema and the validators are RigEuler's validated code. ForceArtGS contains no simulator-format code.

**Not done:** RigEuler3DGS has not been switched to the shared package. It still uses its own copy on
`phase5-sim-asset-export`; its thin handoff adapter (spec 10.3) is NOT_RUN.

## ForceArtGS producer (`tools/export_parttwin_handoff.py`)

* **Input.** Reads only joint-aware outputs: the branch authority (joint in W), `N_BW`, the frozen splatfacto-art
  checkpoint and its part weights. It never reads vanilla `articulation.json` or calls `estimate_joint.py`.
* **Production vs diagnostic.** A production handoff requires `selection.json` IDENTIFIED and `final` PASSED.
  `--diagnostic-branch` writes a uniquely named DIAGNOSTIC handoff, flagged NOT_FOR_RESULTS downstream.
* **Hard membership.** Every Gaussian must have w < 0.01 or w > 0.99, otherwise the producer ABSTAINs (spec 4.3).
  The fridge has 187,743 static and 152,107 moving Gaussians, none intermediate. The maximum soft-vs-hard opacity
  factor change is 1.0e-4, which is recorded.
* **PLYs come from the checkpoint tensors exactly.** Values are copied as stored; quaternions are only re-indexed
  xyzw → wxyz, and `quaternion_order: xyzw` is asserted. The ns-export PLY is a cross-check only: it re-multiplies
  quaternions by the identity offset (1.2e-7 difference). The derived uint8 `red/green/blue` columns are dropped,
  and the drop is recorded.
* **Frames.** The Gaussians live in blend frame B. `T_Wc_P = N_BW^-1`; `N_BW` is projected onto SO(3), with a
  correction of about 1e-7. The builder expresses the joint in the PLY frame exactly.

## Results

| check | result |
|---|---|
| PArtTwin1D tests `tests/test_handoff_build_asset.py` | 3/3 PASS. R and P: joint in PLY frame = `T_P_Wc A_Wc T_Wc_P` to 1e-12; full export with measured synthetic physics, URDF PASS, MJCF PASS (MuJoCo 3.3.5), USD PASS (OpenUSD, rigeuler-sim-export venv); bitwise PLY round trip; fail closed on soft membership, ambiguous without opt-in, tampered PLY, prismatic pivot, edited frame map |
| PArtTwin1D full suite | 82 passed, 78 skipped (opt-in GPU), no regression |
| ForceArtGS joint-aware tests | 11/11; joint-discovery tests 92/92 |
| real red fridge (coarse, accepted revolute) | `BLOCKED_MISSING_PHYSICS`. PLY split PASS and bit-identical per field to the producer PLYs; analytic kinematics PASS over 30 frames (position 5.6e-16 m, rotation 3.3e-16); object.yaml PASS; portability PASS. No URDF/USD/MJCF, by design: no measured physics exists |
| real black box (ambiguous) | two DIAGNOSTIC assets (revolute and prismatic candidates), same checks, NOT_FOR_RESULTS |

## Not implemented / NOT_RUN

* **Collision geometry from RGB-D (spec §7).** Only a supplied measured mesh is accepted, as in RigEuler.
  `COLLISION_UNAVAILABLE` on real scenes.
* **Physics.** No dynamics identification (spec §6.2); real scenes stay `DYNAMICS_UNAVAILABLE`.
* **Runtime pose providers (spec §9).** Isaac and MuJoCo `LinkPoseProvider`s are not written; the existing
  ManiSkill renderer is unchanged. No Isaac runtime run.
* **Renderer parity (E10/E11).** Soft vs hard renders are compared only through the recorded opacity bound,
  not by rendering.
* **Producer unit test.** ForceArtGS has no unit test for intermediate-weight abstention; that path is exercised
  only through the builder-side `hard_membership: false` test.

## Commands

```bash
# after a joint-aware run: G8 for every scene (production asset if IDENTIFIED, diagnostics if ambiguous)
STAGES=export EXPORT_ARGS="--export-parttwin --export-diagnostic" bash pipelines/joint_aware_real_scenes.sh coarse
# one run, with measured physics (physics-spec format: RigEuler docs/PHASE5_FINAL_SIM_ASSET_FORMAT.md)
bash pipelines/export_parttwin_asset.sh <out_root>/<scene>/coarse --dynamics physics_spec.yaml --registration reg.yaml
```
