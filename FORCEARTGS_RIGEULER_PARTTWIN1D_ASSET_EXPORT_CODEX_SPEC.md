# ForceArtGS → PArtTwin1D: canonical articulated-3DGS asset export (URDF, USD, MJCF)

**Codex implementation contract — 18 November 2026**  
**Repositories:** `~/ForceArtGS` (`SamueleMara/ForceArtGS`) and `~/PArtTwin1D` (`SamueleMara/PArtTwin1D`). The exporter must also consume the **same canonical handoff** emitted by `~/Joint3DGS/RigEuler3DGS` (Phase 5).  
**Relationship to prior specification:** Add **G8: simulation-ready export** after G1–G7 in `FORCEARTGS_END_TO_END_JOINT_AWARE_RECONSTRUCTION_CODEX_SPEC.md`. Do **not** substitute export for missing or unsuccessful joint-aware reconstruction.  
**Deliverable:** executable code, tests, actual generated synthetic assets, clear provenance, and a real-scene export **only after upstream readiness gates pass**. Do not claim that any real asset has been produced without running and verifying it.

## 0. Non-negotiable architectural decision

Implement **one shared PArtTwin1D asset exporter** consuming a versioned, simulator-neutral articulated-3DGS handoff. Create **thin producer adapters** in ForceArtGS and RigEuler3DGS; do not build two nearly identical URDF exporters or simulator runtimes.

```text
ForceArtGS G1–G7  ──┐
                   ├──> Articulated3DGS Handoff v1 ──> PArtTwin1D canonical asset builder
RigEuler3DGS P5 ───┘                                         │
                                               ┌───────────────┼────────────────┐
                                               │               │                │
                                        ManiSkill/SAPIEN   Isaac Sim/Lab      MuJoCo
                                        URDF + gsplat     USD + adapter     MJCF + gsplat
                                               │               │                │
                                               └── physics engine link poses ──┘
                                                           │
                                          2 unchanged standard Gaussian PLYs
```

- There are exactly **two rigid material parts**, `static` and `moving`, with **one 1-DoF revolute or prismatic joint**. Do not introduce unconstrained motion, variable link count, inferred extra joints, or independent renderer animation.
- The source 3DGS is **metric, canonical, fully trained**. Both PLYs must use the original Graphdeco-style standard Gaussian fields, including positions, log-scales, opacity logits, rotations and all SH coefficients. No point-cloud downgrade, mesh-as-visual replacement, retraining, pose baking, lossy rewriting or filtering after final export.
- Collision OBJ files are **physics-only geometry**. They are not substitutes for the PLYs and must not be derived by treating Gaussian ellipsoids as exact physical surfaces.
- **All** URDF, USD and MJCF physics originate from one `asset.yaml` + `physics/dynamics.yaml`, not separate hand-edited guesses.
- **Any simulator drives articulation state;** each rendering adapter reads its corresponding live link poses and applies `T_link_ply`. Do not animate the 3DGS with recorded training `q_t` during simulation. Recorded trajectories are for validation/replay only.
- Use a **separate mesh/collision representation** in the native physics engine. Never send `static.ply`/`moving.ply` to a collision mesh importer as if they were triangle meshes.
- **Physical parameters are not inferable from visual 3DGS alone**. Unknown mass, CoM, inertia, limits, friction and damping remain unknown; do not silently invent experimentally calibrated values.
- **Fail closed** on ambiguous joint type, frame mismatches, non-metric geometry, unsupported soft labels, unsafe physics, invalid calibration, or missing dependency. A diagnostic/visualization-only output is allowed with a conspicuous manifest status; do not label it a validated physical digital twin.
- Keep the existing ForceArtGS and RigEuler3DGS renderers/training paths intact. Export is CPU-oriented serialization and collision processing; downstream PArtTwin1D uses its existing gsplat renderer. No new Gaussian renderer or retraining is needed.

## 1. Existing code and references Codex MUST inspect before modifying

### 1.1 ForceArtGS

- `FORCEARTGS_END_TO_END_JOINT_AWARE_RECONSTRUCTION_CODEX_SPEC.md`, stages G1–G7. G8 must consume their **accepted authoritative joint** and the final hardened per-Gaussian static/moving assignment, **not** the old vanilla output.
- `estimotion_2part/joint_discovery/{exports.py,rigid_kinematics.py,model_selection.py}`: `joint_candidates.json`, `joint_states.npz`, `A_R(θ)`, `A_P(d)`, uncertainty, frame map and joint identity.
- `preprocessing/prepare_multiview.py`; `preprocessing/estimate_joint.py`: beware the legacy classifier's mean-relative-IQR angle bug and the legacy ICP/virtual-camera conventions. **Never call `estimate_joint.py` in joint-aware export.**
- `pipelines/{freeartgs.sh,splatfacto_art.sh,twopart_blend.sh}`, `pipelines/train_local_datasets.py`: actual stage dispatch and opt-in output paths.
- `reconstruction/nerfstudio` **git submodule** (not parent main): `nerfstudio/{data/dataparsers/splatfacto_art_dataparser.py,models/splatfacto_art.py,models/twopart_blend.py}`. Verify exact submodule SHA and imported package file.
- In `splatfacto_art.py`, the Gaussian renderer currently duplicates each canonical Gaussian as static/moving and scales each branch's alpha by `(1-w)`/`w`. **That is not yet two disjoint immutable per-link Gaussian sets.** See §4.

### 1.2 RigEuler3DGS — mandatory parity target

Read these before implementing the adapters:

- `RigEuler3DGS_PHASE4_PHASE5_CODEX_IMPLEMENTATION_2026-11-10.md` and `RigEuler3DGS_UNIFIED_PHASE4_PHASE5_CODEX_SPEC.md`.
- `PHASE45_IMPLEMENTATION_REPORT.md` plus the actual current local Phase-5 run manifests.
- `tools/train_phase5_rgbd_gaussians.py`, `recovery/articulated_ply.py`, `tools/export_articulated_ply.py`, `recovery/{contracts.py,gaussians.py,articulated_scene.py,training.py}`.
- RigEuler Phase 5 contract: **one canonical endpoint**, explicit original-frame mapping (forward or backward acquisition order), exact 1-DoF joint, **frozen during Gaussian training**, hard material identity per Gaussian, lineage and unchanged-joint proof.
- RigEuler's Phase-5 `final/point_cloud.ply` may be an **extended articulated PLY** containing `part_id`, lineage and `joint`/`joint_state` elements. This is **not** the same as two standard visual PLY files. The adapter must extract only standard Gaussian `vertex` attributes into `visual/static.ply` and `visual/moving.ply` in original row order *within each part*, without numeric changes. Carry lineage and joint metadata separately in provenance sidecars.
- Never confuse RigEuler `processing_index` with `original_frame`; use the actual original frames and anchor from the joint contract. Do not rerun the opposite direction to fabricate an asset.

### 1.3 PArtTwin1D — shared exporter/runtime owner

Already present and reusable:

- `src/parttwin1d/assets/{gaussian_ply.py,frame_alignment.py,urdf.py}`.
- `src/parttwin1d/geometry/se3.py`: explicit `T_A_B` composition; proper 4×4 SE(3) with no embedded scale; scalar-first (w,x,y,z) Gaussian rotation.
- `src/parttwin1d/rendering/{gsplat_backend.py,link_adapter.py}`: two PLY link bindings via `T_link_ply` and **one global gsplat rasterization**; local-frame SH direction handling; live physics pose transport.
- `src/parttwin1d/sim/{env.py,articulation.py,state.py}` and `src/parttwin1d/viewer/maniskill_viewer.py`: existing ManiSkill/PhysX URDF integration and GUI (do not rebuild).
- `configs/real_object.template.yaml`, `docs/{coordinate_frames.md,urdf_and_gaussians.md}`, `python -m parttwin1d.cli.inspect`, existing unit/GPU tests.
- Existing PArtTwin1D config is **schema v1** (`object.yaml`: `urdf`, `fixed_base`, `T_world_root`, `links.<name>.{ply,T_link_ply}`, optional cameras). Keep backwards compatibility; `asset.yaml` is the **new shared physics/export source**, `object.yaml` is generated as a consumer configuration. Do not rename existing public APIs or break the hinged-box fixture.

**Git safety:** Inspect current SHA, dirty tree and submodule HEADs for all repositories; don't overwrite local experimental changes, reset/check out over dirty submodules, modify raw acquisitions, or touch existing results. Use reviewable branches/worktrees and atomic, fresh output directories. Report every submodule commit and exact runtime import path. No remote push without explicit request.

## 2. Required final asset layout

```text
assets/
  red_fridge/
    asset.yaml                    # authoritative serialized asset: source, frames, joint, physics, visual bindings
    object.yaml                   # generated PArtTwin1D viewer config, schema_version: 1
    provenance/
      export_manifest.json        # hashes, source run, software versions, statuses, warnings
      frame_map.json              # original frame IDs, timestamps, q trajectory, anchor
      gaussian_lineage.npz        # original Gaussian UID, parent UID, source UID, part, unresolved state
      joint_fingerprint.json      # fingerprint from frozen accepted joint and original frame map
      frame_validation.json
      collision_validation.json
      physics_validation.json
      simulator_equivalence.json
    visual/
      static.ply                   # STANDARD Graphdeco 3DGS PLY, exact frozen Gaussian parameters
      moving.ply                   # STANDARD Graphdeco 3DGS PLY, exact frozen Gaussian parameters
    physics/
      dynamics.yaml               # the ONLY physical parameter specification
      collision/
        static.obj                # link-frame physics geometry
        moving.obj                # link-frame physics geometry
        convex/                   # optional decomposed convex pieces (with manifest)
          static_000.obj
          moving_000.obj
    maniskill/
      model.urdf                  # generated from asset.yaml + dynamics.yaml
    isaac/
      model.usd                   # generated native USD physics, with explicit Gaussian binding metadata
      caches/                     # optional generated renderer-specific cache, never authoritative
    mujoco/
      model.xml                   # generated MJCF physics (2 bodies, 1 hinge/slide)
```

Do not change `visual/*.ply` after the export commit point. Record their SHA-256 and verify their bytes are unchanged by exporting or rendering in each simulator. Any generated cache must have explicit `source_ply_sha256` and may be deleted/rebuilt with no change to physics or appearance.

## 3. Shared versioned handoff contract — producer-independent

Define `articulated_3dgs_handoff.json`, schema `articulated3dgs.handoff/v1`, at the **ForceArtGS/G7** and **RigEuler/P5** output boundary. Use strict JSON Schema or pydantic/dataclasses with unknown-key rejection. The producer is responsible for physics-independent truths, not simulator-format specifics.

Minimum fields (the following is schema-shaped illustrative data; **no placeholder numeric values may be copied into a real export**):

```yaml
schema: articulated3dgs.handoff/v1
object_id: red_fridge
producer: ForceArtGS                 # or RigEuler3DGS
source:
  run_dir: /absolute/read-only/path/to/accepted/run
  git_sha: <recorded_commit>
  config_sha256: <recorded_hash>
  joint_fingerprint: <recorded_hash>
  gaussian_fingerprint: <recorded_hash>
  acquisition_world: Wc             # calibrated metric physical world
  coordinate_units: m_rad_s
  gaussian_parameter_space: canonical_capture_world
  canonical_anchor_original_frame: <integer>
  canonical_anchor_timestamp_s: <real>
  candidate_id: accepted_revolute
  status: accepted                 # accepted | ambiguous | invalid | diagnostic
joint:
  type: revolute                   # revolute | prismatic; no silent default
  axis_Wc: [ux, uy, uz]             # normalized signed 3-vector
  origin_Wc: [ox, oy, oz]           # required for R; NULL for P
  q_anchor: 0.0                    # enforced in exported canonical convention
  q_unit: rad                       # m for P
  q_observed_range: [min_q, max_q]
  reference_frame: Wc
  uncertainty: <provenance-bearing object or null>
  selection_status: accepted
  # q_t is in the separate frame-map file; NEVER infer it from camera frame order
frames:
  file: provenance/frame_map.json
  one_shared_q_per_original_frame: true
parts:
  static:
    ply: source/static_canonical.ply
    T_Wc_P: <4x4 proper SE3>
    immutable_gaussian_fields: [x, y, z, f_dc_*, f_rest_*, opacity, scale_*, rot_*]
    membership: hard
    lineage: source/static_lineage.npz
  moving:
    ply: source/moving_canonical.ply
    T_Wc_P: <4x4 proper SE3>
    immutable_gaussian_fields: [x, y, z, f_dc_*, f_rest_*, opacity, scale_*, rot_*]
    membership: hard
    lineage: source/moving_lineage.npz
validation:
  joint_frozen_during_final_gaussian_training: true
  joint_unchanged_proof: <path-to-proof>
  reliable_metric_scale: true
  heldout_render_checked: true
  frame_equivalence_checked: true
```

Actual on-disk JSON uses real numeric matrices and absolute or self-contained relative file paths. Support a producer's two canonical part PLYs **or** a combined final PLY plus part IDs; always materialize two STANDARD PLYs at the export boundary. Preserve exact source vertices/fields and hashes.

**Frame-map schema:** `original_frame`, `timestamp_s`, `q`, `q_valid`, `q_uncertainty`, acquisition `direction` and `processing_index` if applicable. Unique and strictly ordered by *original chronology* in serialized playback table; one q per original frame, shared by every camera. Anchor q is exactly zero. Validate units, sign/gauge, chronological interpretation and whether the final training camera frame indices equal the handoff's original IDs. A backward-processed RigEuler result must reproduce the original chronological motion exactly.

**No double source of truth:** `asset.yaml`, `object.yaml`, URDF, USD, MJCF and adapter metadata are **generated outputs** of the accepted handoff plus physical inputs; do not estimate a second joint, invert axis sign for one engine without corresponding q sign conversion, or independently infer mechanical limits.

## 4. Strict standard PLY extraction and the critical soft-assignment boundary

### 4.1 What "unchanged PLY" means

- Once `visual/static.ply` and `visual/moving.ply` are created, no simulator may edit, resample, optimize, reorient, scale, compress lossy, alter SH, or bake joint transformations into them.
- For a combined source PLY, the split operation necessarily **writes new files**. The invariance guarantee is **exact per-Gaussian numerical equality of every standard 3DGS vertex field** with the corresponding source row, and one-to-one UID lineage. It does not mean that file hashes of the two split files equal the hash of the single combined source file.
- Preserve the full standard field set and metadata needed by `parttwin1d.assets.gaussian_ply.load_gaussian_ply`, including `x/y/z`, `nx/ny/nz` if required by the writer, `f_dc_0..2`, `f_rest_*`, `opacity`, `scale_0..2`, `rot_0..3`. Copy typed scalar fields; do not decode/re-encode SH or normalize/reorder quaternions during file materialization. Beware **Nerfstudio internal XYZW vs Graphdeco PLY WXYZ**; validate against the original PLY export code.
- Standard visual PLYs **must not contain the RigEuler extra `joint`, `joint_state`, lineage elements or custom material fields**. Store those exclusively as sidecars.
- Check the combined-to-split bijection: `N_static + N_moving = N_final`, disjoint UIDs, each output UID references one source vertex and no duplicated Gaussian. Verify field-by-field bitwise equality for floating-point data and SH channel order.

### 4.2 RigEuler adapter

Use `recovery/articulated_ply.py` / `tools/export_articulated_ply.py` to read the extended Phase-5 PLY and its frozen joint, `part_id` and lineage. **Do not rerun Phase 3, Phase 4, Phase 5 or optional joint refinement**. Verify `joint_unchanged_proof.json`, `complete.json`, anchor/direction/original-frame mapping and source SHA. If `unresolved` is present, preserve it as a flag in provenance and reject `production_physics` unless an explicitly documented uncertainty acceptance criterion is satisfied. A frozen render label alone is not a claim of true material identity.

### 4.3 ForceArtGS adapter

Final G6/G7 may export one canonical 3DGS with **soft** Gaussian membership `w_i`. The current `splatfacto-art` renderer uses two effective alpha branches, `alpha_static = alpha*(1-w)` and `alpha_moving = alpha*w` for the *same* canonical Gaussian.

**It is mathematically impossible to reproduce that continuous two-branch image model with two disjoint link PLYs while leaving every original Gaussian opacity unchanged.** Duplicating rows or baking weighted opacity would change the assets. Therefore:

1. Require **persistent hard part identity** for every included Gaussian at final export. Material identity must come from the jointly refined static/moving inference and whole-sequence evidence, not a one-frame color threshold.
2. Add an opt-in **final hard-material refinement/commit stage** *before* the canonical export freeze: choose physically supported assignments, freeze them, retrain only Gaussian appearance/geometry if needed while keeping the accepted joint fixed, and validate against held-out RGB-D/masks. Freeze resulting PLYs. This is still reconstruction, not simulator conversion.
3. If a Gaussian is unresolved or a substantial class of weights remains intermediate, **ABSTAIN** from a two-PLY faithful production export. Produce an explicit diagnostic report with counts, weights and render gap; do **not** silently use `w > 0.5` or bake opacity in the "unchanged" path.
4. A separate clearly marked **approximate/diagnostic** asset mode may exist for testing serializers, but it cannot be represented as appearance-faithful or physically validated; any opacity modifications must be named and quantified, and kept out of the strict two-PLY output path.
5. Prove that rendering `static.ply` + `moving.ply` at recorded `q_t` (with original rigid joint motion) matches the final committed ForceArtGS renderer on the same cameras within stated RGB/alpha/depth tolerances, especially at occlusion boundaries.

Export the final joint and material provenance **from the authoritative G7 accepted manifest**, never by calling legacy `estimate_joint.py` or reading a vanilla `articulation.json` whose type may have been misclassified.

## 5. Exact frame algebra: capture world, two links, physics world, PLY frames

Adopt PArtTwin1D notation: `T_A_B` maps column-vector coordinates from frame B to frame A, composition left to right. All transforms are rigid, metric, right-handed, and validated as SE(3).

- `Wc` = calibrated capture/reconstruction world, meters.
- `W` = simulator world, typically z-up, meters.
- `L0` = static/root physical link frame; `L1` = moving link frame at canonical `q=0`.
- `P0`, `P1` = static and moving **stored PLY** frames.
- `J` = canonical joint frame (at `q=0`, its local +Z axis is the signed joint axis).
- `T_W_Wc` = explicit rigid placement of captured asset in simulation world. Do not silently assume identity when coordinate conventions differ.
- `T_Wc_P0`, `T_Wc_P1` = known rigid transforms of stored PLY coordinates into capture canonical world.

Choose an explicit static root frame `T_Wc_L0`; a static-support centroid with identity rotation is an acceptable documented **gauge**, not a measured physics quantity. Choose the joint frame:

- Revolute: origin `o_Wc` is on the estimated hinge line; choose an orthonormal basis `B=[e_x,e_y,u]` with `e_z=u`. The gauge component of `o` along the hinge may be chosen deterministically, and changing it must not change articulated geometry.
- Prismatic: use `e_z=u`; choose joint-frame origin deterministically near the moving-part canonical centroid for numerical conditioning. This point is **not** a physically identified hinge or slider bearing location. It must not become a fictitious revolute origin.

Let `T_Wc_J0 = [B, o_or_gauge]`. For a single URDF joint whose child frame coincides with its joint frame at q=0, set:

\[
T_{L0,J0} = T_{Wc,L0}^{-1} T_{Wc,J0},\qquad T_{Wc,L1}^{0}=T_{Wc,J0}.
\]

The Gaussian bindings are fixed:

\[
\boxed{T_{L0,P0}=T_{Wc,L0}^{-1}T_{Wc,P0}},\qquad
\boxed{T_{L1,P1}=(T_{Wc,L1}^{0})^{-1}T_{Wc,P1}}.
\]

Define `S_R(q)` as rotation about **local Z**, `S_P(q)` as translation by `q` on local Z. With the static root placed in the simulator:

\[
T_{W,L0}=T_{W,Wc}T_{Wc,L0},\quad
T_{W,L1}(q)=T_{W,L0}T_{L0,J0}S_H(q).
\]

Then for both P and R:

\[
\boxed{T_{W,P0}=T_{W,L0}T_{L0,P0}},\qquad
\boxed{T_{W,P1}(q)=T_{W,L1}(q)T_{L1,P1}}.
\]

**Required cross-check against the reconstruction model** for every observed `q_t` and additional values within validated joint limits:

\[
T_{W,P1}(q_t)
\overset{!}{=}
T_{W,Wc}\,A_H(q_t)\,T_{Wc,P1},
\]

where `A_R(q)=[R_u(q),(I-R_u(q))o]` and `A_P(q)=[I,q u]` in Wc coordinates. Test transformed Gaussian **means and orientations**, not only centroids. The static link must be unchanged for all q. Enforce no scene-scale factor inside `T_link_ply`. If Nerfstudio used auto-orientation/auto-scale or a `global_transform`, invert the *actual recorded* normalization and recover metric Wc before export—**never assume it was identity**.

**Joint zero conventions:** Canonical q=0 is the stored reference model anchor (RigEuler selected endpoint or ForceArt accepted reference). If a simulator uses a shifted zero, export the shift explicitly and transform the geometry/limits consistently; default must maintain q=0 at the canonical anchor. Do not shift q independently in each simulator.

## 6. Canonical `asset.yaml`, `dynamics.yaml`, and physical evidence levels

`asset.yaml` is the **single authoritative shared spec** created after handoff validation. It should contain:

- schema version `parttwin.asset/v1`, object ID, `producer`, source hashes, status and timestamp frame-map reference.
- canonical joint kind, signed axis and origin in Wc, `T_Wc_L0`, `T_Wc_J0`, `T_W_Wc`, observed trajectory/range and **separate** physical joint limits with evidence/source.
- `links.static` / `links.moving`, their `visual/*.ply` relative paths, exact `T_link_ply` matrices and SHA256, collision paths and collision validation statistics.
- `physics/dynamics.yaml` path and its fingerprint. Model name/part link mapping must be identical across formats.
- rendering source (= `visual/static.ply`, `visual/moving.ply`), not a converted mesh or arbitrary USD visual.
- `export_status` and validation/provenance paths.

`physics/dynamics.yaml` schema (actual numbers supplied by data or explicit user physics inputs, **not invented**):

```yaml
schema: parttwin.dynamics/v1
mode: measured                 # measured | identified | approximate | unavailable
units: {length: m, mass: kg, time: s, angle: rad, inertia: kg*m^2}
links:
  static:
    mass_kg: <measured_or_identified_or_null>
    com_link_m: <3-vector_or_null>
    inertia_com_link_kg_m2: <symmetric_3x3_or_null>
  moving:
    mass_kg: <measured_or_identified_or_null>
    com_link_m: <3-vector_or_null>
    inertia_com_link_kg_m2: <symmetric_3x3_or_null>
joint:
  name: articulation_joint
  type: revolute              # prismatic for drawer
  limits:
    lower: <physical_limit_or_null>
    upper: <physical_limit_or_null>
    source: measurement       # measurement | mechanical_spec | observed_only | unknown
  damping: <N*m*s/rad_or_N*s/m_or_null>
  coulomb_friction: <N*m_or_N_or_null>
  actuation: null            # optional explicit motor/control parameters
contact_material:
  coefficient_friction: <measured_or_selected_assumption_or_null>
  restitution: <value_or_null>
evidence:
  sources: [<artifact_paths_or_descriptions>]
  assumptions: []
```

**Important distinctions**:

1. **Observed q range ≠ mechanical hard stop.** Do not silently copy observed min/max to validated physical limits. For a **kinematic preview** one may use a clearly labelled `observed_only` temporary playback interval and a corresponding provisional URDF bounded joint, but this does not qualify as measured mechanical bounds. For production physics, require authoritative bounds or explicit accepted approximation, and mark their provenance.
2. **No automatic force-derived mass/inertia**. If the separate user dynamics-identification stage has delivered estimates, ingest them with covariance, residual and experiment provenance. If absent, `mode: unavailable` and block the dynamics-ready status; emit read-only kinematics/visualization diagnostics only or require explicit `--allow-approximate-physics` with a marked profile.
3. Validate `mass>0` for moving link, symmetric positive-definite inertia at link CoM (in kg·m²), triangle inequality of principal moments where applicable, correct translation/rotation of inertia tensors, sensible mass/length units, friction and damping nonnegative, and force/torque units differentiated by joint kind.
4. Dynamics values are in **physical link frames**, not capture Wc, local Gaussian frames, normalized Nerfstudio coordinates, or arbitrary sensor frames. The interface must expose transformations and correctly apply the parallel-axis theorem when converting inertias.
5. For fixed-base objects, `static` may be physically fixed to world; do not treat it as a dynamic floating object. Still preserve its collision geometry and explicit pose. Root inertial fields may be absent if the chosen runtime permits fixed roots; document the difference.
6. Contact material friction and **joint Coulomb friction** are separate physical effects. Map them separately in each simulator. Do not compare numerical settings that have different physical semantics as if equivalent.

## 7. Physics-only collision reconstruction and validation

### 7.1 Source of geometry

Use **observed object RGB-D surfaces and the accepted joint-consistent part segmentation**, not PLY covariance support as true collision boundaries:

- Unproject calibrated, valid, masked RGB-D pixels for static and moving parts, with per-view depth-noise and depth-edge handling.
- For moving observations, inverse-transform into the joint-canonical Wc frame using the **same accepted** `A_H(q_t)^{-1}`; static remains in Wc.
- Reject robot/wrist observations, other-part depth layers, invalid/ambiguous/unknown masks, and contradictory visibility. Do not fabricate hidden surfaces from occluded pixels as if observed.
- Fuse consistent surfaces per part (e.g. TSDF/Poisson with explicitly justified capping, or convex primitive fit) and transform resulting collision vertices to corresponding link frames using the §5 matrices. Preserve physical metric units.
- Source geometry may instead come from a supplied measured collision mesh/CAD proxy; record the source and `T_link_collision` and convert it into the physical link frame. Prefer better externally measured surfaces when RGB-D cannot recover the back/underside.
- If only thin or incomplete surface data exist (black-box lid underside, hidden hinge enclosure), return `collision_geometry: INCOMPLETE` and require an explicitly marked conservative proxy. Do not claim unseen topology is recovered.

### 7.2 Native collision requirements

- Produce `physics/collision/static.obj` and `moving.obj` as link-local mesh references. OBJ vertex positions are in **link coordinates**, so URDF collision `<origin>` can be identity.
- Validate finite vertices, index validity, normals/orientation, scale, unexpected holes, self-intersections, non-manifold edges, physical thickness, and joint clearance. When watertightness cannot be established, mark it and use a documented conservative collider only after human review or `--allow-proxy-collision`.
- For engines requiring convex collision, produce a **reproducible convex decomposition** (or a single conservative hull for diagnostic use), with each convex piece a separate collider and original mesh retained as provenance. In particular, MuJoCo generally collides with mesh convex hulls; a concave OBJ may have different contact behavior in MJCF unless decomposed. Export each hull piece as a distinct geom/reference.
- Compare collision mesh AABB and support against canonical observed surfaces and Gaussian means. A low Gaussian-containment score can indicate an error, but **100% Gaussian containment is not a valid physics objective** (Gaussians are appearance representation, not guaranteed solid points).
- Validate static/moving collider overlap at q sampled across validated range. Distinguish legitimate near-contact from deep penetration. Never automatically shift the joint to make colliders fit.
- For root-fixed articulated objects, ensure joint-adjacent parts are appropriately collision-filtered/enable self-collision only if supported and validated; avoid pathological self-contact preventing motion.
- Store source-view coverage, completion assumptions, mesh hashes and convex-decomposition settings in `provenance/collision_validation.json`.

## 8. Simulator backends — same physics, different native files

### 8.1 ManiSkill/SAPIEN: `maniskill/model.urdf`

Generate a single root link `static_link`, a single child `moving_link`, and one joint `articulation_joint`:

- `<joint type="revolute">` or `<joint type="prismatic">` with parent `static_link`, child `moving_link`.
- `<origin xyz="..." rpy="...">` is `T_L0_J0` from §5; `<axis xyz="0 0 1">` in the **joint/child frame**, never raw `u_Wc` unless the axis frame is identity.
- `<limit lower="..." upper="..." effort="..." velocity="...">` values require physical source or explicit provisional profile. Do not use arbitrary generous defaults as verified mechanical limits.
- `<inertial><origin xyz=com_link rpy="0 0 0">... full symmetric inertia about CoM ...</inertial>` using actual link-frame inertia. `<collision>` uses link-local OBJ(s). URDF `<visual>` may be absent: native PhysX renders collision proxies for debug; the PArtTwin1D Gaussian layer renders appearance.
- `<dynamics damping=... friction=...>` values are mapped from shared canonical units with source-dependent limitations logged. No independent URDF-only physics edits.
- Generate `object.yaml` using PArtTwin1D's existing schema v1: `urdf: maniskill/model.urdf`, `T_world_root`, each binding `links.<name>.ply: visual/{static,moving}.ply` plus `T_link_ply`, and optional calibrated cameras transformed via `T_W_Wc`. Use relative paths resolved against `object.yaml` and run `parttwin inspect`.
- At runtime, use existing `ArticulationDriver` and `ArticulatedGaussianRenderer`: `T_W_Pi = T_W_Li^PhysX(q) @ T_Li_Pi` from **live PhysX**. No object animation from `frame_map.json` during stepping. Kinematic preview may set PhysX q explicitly, with zero velocities, but that is not a simulation result.
- Test actual URDF load in installed ManiSkill/SAPIEN and inspect active joint type, axis, q limits, collision/inertia and link poses at q values.

### 8.2 Isaac Sim/Isaac Lab: `isaac/model.usd`

Implement a real native USD physics asset, not a renamed URDF/OBJ or a PLY path masquerading as `UsdGeomMesh`:

- Prefer USD authored from shared `asset.yaml` with `UsdPhysics.RigidBodyAPI`, `CollisionAPI`, mesh primitives/convex pieces, `UsdPhysics.RevoluteJoint` or `UsdPhysics.PrismaticJoint`, valid `body0/body1` relationships, explicit `localPos0/localRot0/localPos1/localRot1`, consistent axis, limits, link masses/inertias, damping/friction and PhysX-specific attributes where available.
- Alternatively the **Isaac Sim URDF importer** may be used as the mechanical conversion stage if verified as equivalent to the authored URDF. Do not accept the conversion blindly: it may transform axes, joint drive defaults, mass/inertia, mesh colliders or namespace. Inspect the resulting USD scene and compare to `asset.yaml`.
- Make the USD stage `metersPerUnit=1.0` (if assets are metric), with explicit up axis and compatible root transform; record any necessary world-up mapping as a rigid transform. USD angular-limit representation may differ (degrees vs radians), so explicitly convert and roundtrip-test the native schema values.
- Retain the raw PLYs as external authoritative references in custom binding metadata, e.g. per-link `source_ply`, `source_sha256`, `T_link_ply` (or a separate JSON if preferred). **The physics meshes are not the visual Gaussians.** Renderer-specific converted Gaussian data, if needed for Isaac, are `isaac/caches/` only and tied to source SHA; never replace `visual/*.ply`.
- Implement a thin Isaac Gaussian renderer adapter: resolve/link-match the USD articulation bodies and fetch **live USD/PhysX link world poses after each simulation step**. Apply `T_link_ply` for the unchanged Gaussian assets. No second joint state, no animation timeline driving splats independently.
- Do not guess USD/PhysX API names: introspect the installed Isaac Sim/OpenUSD version and use its supported APIs; register an explicit test skip/pending status if the runtime is unavailable. The builder must not write a fake `.usd` and mark it complete.
- Test by opening stage in Isaac Sim (or valid `pxr` + PhysX), retrieving joint type and link transforms, and stepping a controlled drive/contact episode. Log Isaac version and exact conversion settings.

### 8.3 MuJoCo: `mujoco/model.xml`

Generate MJCF from the *same* link/joint and dynamics model:

- `<mujoco model="...">`, worldbody with fixed base body at `T_W_L0`, nested moving body at `T_L0_L1^0`, and **exactly one `<joint type="hinge">` or `<joint type="slide">`** at the child body's origin (e.g. `pos="0 0 0" axis="0 0 1"`). Degrees/radians are explicitly controlled; don't assume default angle units.
- Collision `<asset><mesh file="...obj"></asset>` and `<geom type="mesh" mesh="...">` using **link-frame** OBJ pieces. For concave parts use separate convex hull pieces, not one implicit convexification of a complex concave surface. Use proper collision flags and avoid merging articulated links.
- Explicit `<inertial pos="com_link" mass="..." fullinertia="Ixx Iyy Izz Ixy Ixz Iyz">` in the moving link frame (or a validated equivalent principal-axis form). If the root is static, do not add a floating joint.
- Convert the common damping/friction meaning to MuJoCo's joint `damping` and `frictionloss` where appropriate, documenting differences from URDF/PhysX and how contact friction is specified. Do **not** infer actuator gains or a motor automatically. Actuators are opt-in, typed and separately documented.
- Export Gaussian link bindings in `asset.yaml`; the MJCF `geom` visuals are optional debug proxies. A new MuJoCo/gsplat adapter must read **live `mj_data` link transforms** (e.g. `xpos`, `xmat` for body IDs) and apply the same fixed `T_link_ply`, not drive the splats from an independent `q(t)` playback.
- Validate with the actual MuJoCo compiler (`mujoco.MjModel.from_xml_path`), then FK with specified `qpos`, collision contacts and a small dynamics episode. Record `mujoco.__version__`.

### 8.4 Physics equivalence means parameter parity, NOT identical trajectories

All three formats must agree on **joint type, signed axis, fixed canonical zero, the q bounds and their provenance, link geometry frames, collider shape identity, mass, CoM, inertia and intended damping/friction**. Engine solvers/contact models differ; report semantic mapping and tolerance rather than claiming numerical dynamics equivalence.

## 9. Mandatory link-pose-driven rendering adapters

PArtTwin1D's existing renderer already implements the correct per-link transport for ManiSkill:

\[
\mu_W = R_{W,Pi}\mu_{Pi}+t_{W,Pi},\quad
\Sigma_W=R_{W,Pi}\Sigma_{Pi}R_{W,Pi}^{T},\quad
q_W=q(R_{W,Pi})\otimes q_{Pi}.
\]

Preserve **local/canonical SH view-direction evaluation** when a link rotates: evaluate SH using the ray expressed in the link's stored PLY frame, not static world directions. All links must be composited by a **single depth-aware rasterization**, so a stationary base occludes a moving door correctly. Rendering order cannot change the image. No separate link RGBA images pasted back-to-front.

Implement one reusable `LinkPoseProvider` protocol for PArtTwin1D:

```python
class LinkPoseProvider(Protocol):
    def link_poses_world(self) -> dict[str, np.ndarray]: ...  # T_W_L, proper 4x4
    def joint_state(self) -> dict[str, float]: ...
    def step(self, control=None) -> None: ...                  # native engine simulation only
```

Use the existing ManiSkill provider as reference. Add Isaac and MuJoCo providers with explicit quaternion convention and coordinate-axis tests. Do not add simulation dependencies as mandatory imports for ManiSkill-only users; lazy-load backend-specific libraries. Record simulator time and pose snapshot revision, and ensure renderer uses the snapshot **after** physics advances with no frame lag.

The recorded training joint trajectory may be used for **kinematic replay** by commanding the simulator to a known `q`, but the renderer still reads the simulator's resulting link pose. Never directly apply `A_H(q)` to Gaussians in the runtime after physics has already moved the link; that would double-transform them.

## 10. Implementable repository layout and command-line interfaces

### 10.1 PArtTwin1D — shared code (authoritative generator)

Implement minimally in existing package, not a parallel project:

```text
src/parttwin1d/assets/
  handoff.py                 # read/validate producer-independent schema
  canonical_asset.py         # asset.yaml + dynamics.yaml schema and provenance
  gaussian_split.py          # strict standard PLY split, UID-preserving checks
  collision.py               # RGB-D/mesh collision processing, optional convex decomposition
  physics_inertia.py         # inertia validation / frame transforms / evidence gating
  build_asset.py             # orchestration, atomic files, hashes
  export_urdf.py             # URDF from canonical spec
  export_usd.py              # USD/Isaac conversion and verification
  export_mjcf.py             # MJCF from canonical spec
  equivalence.py             # analytical and compiled cross-format FK checks
src/parttwin1d/sim/
  pose_provider.py           # common provider interface
  isaac_provider.py          # optional runtime, only if Isaac installed
  mujoco_provider.py         # optional runtime
src/parttwin1d/cli/
  build_asset.py             # primary entry point
  validate_asset.py          # audit and compiler/renderer tests
```

PArtTwin CLI (commands are **to implement**, not claims of already existing commands):

```bash
cd ~/PArtTwin1D
source scripts/activate_sim.sh
python -m parttwin1d.cli.build_asset \
  --handoff /PATH/TO/articulated_3dgs_handoff.json \
  --dynamics /PATH/TO/dynamics.yaml \
  --collision-source /PATH/TO/observed_geometry_or_mesh_manifest.json \
  --output assets/red_fridge \
  --formats urdf,mjcf,usd \
  --mode strict

python -m parttwin1d.cli.validate_asset --asset assets/red_fridge/asset.yaml \
  --check-ply --check-frames --check-collision --check-urdf --check-mjcf --check-usd --check-render
python -m parttwin1d.cli.inspect --config assets/red_fridge/object.yaml
python -m parttwin1d.cli.maniskill_gui --config assets/red_fridge/object.yaml
```

If the Isaac runtime is unavailable, report `USD_NOT_BUILT: ISAAC_RUNTIME_UNAVAILABLE`, keep handoff/URDF/MJCF artifacts **only if individually validated**, and do not mislabel a partial export `all_formats_passed`.

### 10.2 ForceArtGS — thin upstream adapter and optional pipeline stage G8

```text
tools/export_parttwin_handoff.py
pipelines/export_parttwin_asset.sh
tests/test_parttwin_handoff_forceart.py
```

Interface (to implement):

```bash
cd ~/ForceArtGS
python tools/export_parttwin_handoff.py \
  --accepted-reconstruction /PATH/TO/G7_FINAL_ACCEPTED \
  --output /PATH/TO/G7_FINAL_ACCEPTED/parttwin_handoff \
  --strict-hard-membership

# Export to the canonical shared builder; do not reimplement URDF/USD/MJCF here.
PARTTWIN1D_PYTHON=/PATH/TO/parttwin_sim/bin/python \
  bash pipelines/export_parttwin_asset.sh \
  --handoff /PATH/TO/G7_FINAL_ACCEPTED/parttwin_handoff/articulated_3dgs_handoff.json \
  --dynamics /PATH/TO/physics/dynamics.yaml \
  --output /PATH/TO/exports/red_fridge
```

Invoke G8 **only** with `--export-parttwin` (or equivalent explicit validated config). `G7 status != ACCEPTED` must not trigger a production export. Preserve existing G1–G7 outputs. For ambiguous P/R, optionally export **candidate diagnostic handoffs with unique names**, but never label either as the established physical asset.

Never use `articulation.json` from a vanilla `splatfacto-art` run, nor legacy `estimate_joint.py`/ICP, as a fallback in `joint_aware` mode. Validate all final q values and static/moving assignments against the accepted frozen joint fingerprint.

### 10.3 RigEuler3DGS — identical handoff schema, thin adapter

```text
tools/export_parttwin_handoff.py
# reads selected Phase-5 run; no training modification
```

```bash
cd ~/Joint3DGS/RigEuler3DGS
python tools/export_parttwin_handoff.py \
  --phase5-dir /PATH/TO/COMPLETED_PHASE5_RUN \
  --output /PATH/TO/COMPLETED_PHASE5_RUN/parttwin_handoff

# The same PArtTwin1D build_asset CLI consumes the output.
```

The RigEuler producer must independently prove Phase-3 joint fingerprint unchanged through Phase-5 training and preserve `material_ids`, `gaussian_uid`, `parent_uid`, `original_frame`, processing direction and anchor. No conversion of extended PLY into nonstandard simulator visual format.

For long-term reuse, consider a versioned lightweight `handoff` Python package **only if justified**; do not introduce circular imports between PArtTwin1D and training repos. The simplest supported contract is versioned JSON + two standard PLYs + sidecars.

## 11. Tests and acceptance gates — require real parsers and simulators

Implement a deterministic test suite with at least the following cases. Test implementation correctness rather than count of passing tests; mark unavailable runtimes `SKIPPED`, not `PASSED`.

| Gate | Mandatory tests and success condition |
|---|---|
| E0: provenance | Source manifests, checkout SHAs, hashed inputs, changed-files audit; unknown/dirty source preserved and marked |
| E1: Gaussian fidelity | Combined→split 1:1 per field, SH indexing, binary equality for fields, IDs, no duplicates/omissions, immutable final PLY SHA after rendering |
| E2: ForceArt soft labels | `w=0.3` rejects strict mode; hardened/frozen reconstruction accepted; no hidden opacity baking |
| E3: RigEuler compatibility | Phase-5 extended articulated PLY ingested, extras moved to sidecars, original-frame trajectory (including backward) correct, fingerprint/proof required |
| E4: frames | Random valid SE3, axis gauges/sign, P/R q=0 and q sweeps, static invariant, `T_W_P1 = T_W_Wc A(q) T_Wc_P1` within numerical tolerance |
| E5: physical parameters | Reject negative mass, invalid inertia, nonmetric unit, unknown mechanical limits/dynamics without explicit mode; proper inertia frame transforms |
| E6: collisions | Valid OBJ, link-local scale, conservative proxy flagged, collider clearance at q samples, no fake hidden geometry |
| E7: URDF | Parser validates 2 links/1 1-DoF joint, axis/limits/inertial/colliders, ManiSkill actual URDF load and link poses at multiple q |
| E8: MJCF | Actual `mujoco.MjModel.from_xml_path`, one hinge/slide DoF, link FK/collision, no floating base, mass/CoM/inertia checks |
| E9: USD | Valid native `pxr` stage and `UsdPhysics` relationships, PhysX/Isaac runtime import, body poses at multiple q, explicit limitations; no fake USD |
| E10: renderer | Same unchanged PLYs from both producers; compare canonical and simulated output at q=0/mid/end under calibrated cameras, static invariance, SH rigid invariance, occlusion, no double transform |
| E11: synchronized dynamics | Physics stepping changes link poses and Gaussian rendering follows exactly; no q animation from logged training states; reset-to-q0 reproduces initial appearance |
| E12: cross-format | For P/R synthetic fixtures, compare link and Gaussian world transforms at q samples across engines; numerical FK errors within declared tolerance; physical trajectory differences quantified, not assumed zero |
| E13: failure modes | Ambiguous joint, unknown soft material, missing collision, absent calibrated physics, units mismatch, source SHA mismatch, invalid timestamp/frame mappings, unavailable SDK — clear failure or limited-status output |
| E14: regression | Existing PArtTwin hinged-box viewer and ForceArt/RigEuler training tests remain unchanged; no upstream renderer or training regression |

Suggested quantitative geometric tolerances for **synthetic** exact-FK tests: translation <= `1e-5 m`, rotation <= `1e-5 rad` in double precision if backend supports them; looser engine-specific float32 thresholds must be **measured and justified**. Image fidelity thresholds must be established from native renderer parity at q=0, not guessed or tuned to hide wrong source frames. Compare RGB, mask, valid depth, and SH/view-dependent behavior.

### Required synthetic fixtures

1. Hinged box: hinge not aligned with world Z, offset pivot, anisotropic Gaussians, nonconstant SH degree ≥1, nonzero `T_W_Wc`, nonidentity `T_link_ply`, known collision meshes/inertias, 20–60° sweep.
2. Prismatic drawer: signed non-axis-aligned translation, 0.12–0.25 m travel, nonidentity link frames, known collision geometry, explicit slider limits, reset and force-driven motion.
3. Degenerate remote-hinge candidate: must **not** silently change a prismatic asset into revolute.
4. Soft-assignment ForceArtGS fixture: production export fails until hardening/validation; diagnostic path clearly marked.
5. RigEuler extended-PLY fixture with backward original-frame mapping, lineage, joint_state and frozen proof; split produces standard per-part PLYs.

### Real-world gates

- **Red fridge first:** use completed **accepted ForceArtGS G7** model, not the older vanilla prismatic `articulation.json` and not the independent G1/G2 diagnostic alone. Validate hinge geometry, motion and static/moving support; if G7 has not passed, generate an explicitly limited diagnostic handoff only.
- **Black box second:** if joint is still `ambiguous/unobservable` or physically incorrect, do not export a definitive revolute URDF. Keep both candidate export diagnostics separated or await the full-render G6 result.
- **Drawer control:** only export prismatic if supported by final joint-aware evidence. White drawer has previously flagged invalid robot-tool mounting; robot force must not be treated as calibrated just to enable physics export.
- **RigEuler interoperability:** choose one already completed, verified Phase-5 R scene and one P scene; run same builder without producer-specific logic or different simulator frame rules.
- Before claiming cross-simulator asset readiness, run compiler/importer validation in ManiSkill, Isaac Sim and MuJoCo environments, and at least one actual physics-driven rendering demonstration each. If SDK unavailable, report readiness as partial.

## 12. Acceptance statuses and reporting

Provide separate, machine-readable statuses:

```text
HANDOFF_VALIDATED
GAUSSIAN_SPLIT_EXACT
JOINT_FRAME_EQUIVALENT
COLLISION_VALIDATED / COLLISION_APPROXIMATE / COLLISION_UNAVAILABLE
DYNAMICS_MEASURED / DYNAMICS_IDENTIFIED / DYNAMICS_APPROXIMATE / DYNAMICS_UNAVAILABLE
URDF_COMPILED_AND_LOADED / URDF_FAILED / URDF_NOT_RUN
MJCF_COMPILED_AND_LOADED / MJCF_FAILED / MJCF_NOT_RUN
USD_COMPILED_AND_LOADED / USD_FAILED / USD_NOT_RUN
SIM_KINEMATICS_VERIFIED
PHYSICS_RENDERING_VERIFIED
ALL_FORMATS_VALIDATED / PARTIAL / DIAGNOSTIC_ONLY
```

Never turn a missing test or library into success. Do not call an asset simulation-ready based only on file existence. Atomic commit after validating the asset, keeping failed intermediates in a non-authoritative staging directory.

Write a thorough `PARTTWIN_ASSET_EXPORT_IMPLEMENTATION_REPORT.md` with:

- Repositories/SHAs/submodules, worktree safety, actual files changed in each repo.
- Actual input artifacts and SHA-256 (joint, material labels, Gaussian checkpoint/PLY, collision and dynamics provenance).
- Source→standard-PLY row preservation proof, Gaussian counts per part, unresolved material counts, SH degree and hashes.
- `T_W_Wc`, `T_Wc_L0`, `T_Wc_J0`, `T_link_ply`, joint axis/origin/zero/limits and explicit coordinate conventions.
- Collision method and coverage, mesh validity, convex decomposition/proxy assumptions, failed geometry and penetration diagnostics.
- Physical quantities and their provenance; which are measured, identified or approximated.
- Compiler/importer results per simulator and version; cross-simulator FK and live pose-driven render comparisons.
- Exact commands and complete pass/fail/skipped test table. Show at least one screenshot or rendered frame per runnable simulator but never replace numerical validation with screenshots.
- Any unsupported parts, blocked stages, or absent real measurement without concealing them.

**Final instruction to Codex:** implement this end-to-end as **G8** while retaining RigEuler3DGS and ForceArtGS upstream independence. The **same two standard 3DGS PLYs** and the **same canonical physics parameters** must produce URDF/USD/MJCF. The physics engine owns link motion; the renderer owns appearance only. The accepted joint and hard part membership must survive every boundary unchanged. No silent fallback to vanilla estimates, arbitrary default dynamics, or fake simulator exports.

## 13. Technical source references (verify installed APIs at implementation time)

- PArtTwin1D: `https://github.com/SamueleMara/PArtTwin1D`; `docs/coordinate_frames.md`, `docs/urdf_and_gaussians.md`, existing renderer and viewer.
- ForceArtGS: `https://github.com/SamueleMara/ForceArtGS`, `estimotion_2part/joint_discovery/exports.py`, local G1–G7 contract.
- RigEuler3DGS: `https://github.com/SamueleMara/RigEuler3DGS`, Phase 4/5 contracts and `recovery/articulated_ply.py`.
- MuJoCo XML Reference: `https://mujoco.readthedocs.io/en/stable/XMLreference.html`; official compiler and mesh collision semantics.
- OpenUSD Physics: `https://openusd.org/release/api/usd_physics_page_front.html` (joint local pose, body targets, revolute/prismatic schema).
- Isaac Sim URDF importer: `https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html` and installed USD/PhysX API.
