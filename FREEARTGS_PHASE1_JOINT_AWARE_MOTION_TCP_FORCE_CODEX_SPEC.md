# Codex implementation specification — FreeArtGS Phase 1: unified joint-aware motion discovery with sliding TCP contact and force

**Status:** proposed implementation to test; NOT a claim that real-data failures have been fixed.  
**Target repository:** `https://github.com/SamueleMara/FreeArtGS` (work on the USER'S actual local worktree, including any ongoing debug work).  
**Reference method:** MoMa-SG, *Articulated 3D Scene Graphs for Open-World Mobile Manipulation* (CoRL 2026), `https://github.com/robot-learning-freiburg/MoMa-SG`, paper `https://arxiv.org/abs/2602.16356`.  
**Primary objective:** replace or optionally bypass unconstrained two-transform Phase-1 moving-support estimation with a **joint, calibrated multiview, static/revolute/prismatic estimation problem**. Incorporate the measured robot TCP as a **known point of contact that may slide on the object**, and the calibrated wrench as independent physical evidence, without treating the TCP as a fixed object material point.

> **Codex:** Implement this in small, tested and independently measurable increments. Read the LOCAL implementation first. Do not use this design to bypass, overwrite, or prematurely claim completion of the parallel six-point root-cause diagnostic. Report what is verified, what is implemented, and what remains unobservable. Existing FreeArtGS behavior must remain reproducible behind `joint_discovery.mode: legacy`.

---

## 0. Instructions, scope, evidence hierarchy and repository safety

1. **Inspect local code, not only public GitHub.** Begin with `pwd`, `git rev-parse HEAD`, `git status --short`, `git diff --stat`, `git diff --submodule=log`, `git submodule status --recursive`, active Python/CUDA/Nerfstudio packages, and the exact scripts generating the last real-data outputs. An ongoing Codex process may have changed local files. Do not reset, stash, clobber or silently discard those changes. Record all provenance under a NEW run directory.
2. **Respect pending diagnostics.** Read any available `docs/multiview_root_cause_black_box.md`, `docs/tcp_contact_tracking.md`, the six-point root-cause audit and the prior known-TCP sliding-contact formulation; if absent, record the omission, do not invent their findings. Keep the legacy Phase 1 as a control. No full 30k-step 3DGS training until cheap synthetic, recorded-track and rendering checks succeed.
3. **Physical cameras are static and calibrated.** Use `cam_001` + `cam_002` by default. `cam_000` is excluded from joint multiview fusion because of observed calibration disagreement but is allowed as explicit `--cameras cam_000 ...`/stress ablation; it may still offer better lid-motion observability. Use correct per-camera intrinsics `K_c`, fixed extrinsics `T_WC_c`, image sizes, timestamps and depth units. Never optimize true camera intrinsics/extrinsics to compensate for bad tracks. Never confuse a moving *virtual* camera in a moving part's frame with a physical camera.
4. **Object tracks are material-point measurements; TCP is NOT.** Track `i` on a moving rigid part follows one joint transform. The calibrated TCP is the instantaneous *contact location*, which may slide over different material points at different times. No rigid grasp, fixed correspondence, known tool mesh, or known tool normal is assumed.
5. **Do not treat all force ratios as joint labels.** Force parallel/perpendicular to TCP motion is evidence but is not, by itself, a universal classifier. Predictions must arise from physically supported contact surface orientations, candidate joint kinematics and a constrained friction model; do not fit free normals or arbitrary per-frame friction to force either hypothesis to win.
6. **Distinguish what is known from what is estimated.** Known conditional on validation: camera calibration; synchronized robot EE/TCP pose; valid object masks. Estimated: moving/static support, joint model/axis/origin, per-time joint state, 3D anchors, RGB-D uncertainties, contact mode, possibly contact normal. Only independently verified contact frames activate contact terms. Unknown contact surface => disabled surface-force constraints, never a fabricated plane.
7. **No training leakage.** Do not use GT joint type, axis, part labels or ground-truth surface for fitting, choosing thresholds or selecting the winning candidate. Oracle tests permitted ONLY as clearly named diagnostics. No manipulating raw source data. Run outputs outside source directories and use job-specific names to avoid overwriting earlier results.
8. **Make the implementation mathematically exact.** Physical rotations must be in `SO(3)` and homogeneous poses in `SE(3)`; never learn unconstrained 4×4 matrices. Explicitly test the order of twist components (`[omega; v]` versus other conventions), left/right multiplication, pose direction, degrees/radians, metres/mm, quaternion order, and camera CV conventions.
9. **Report uncertainty and failure.** The valid outcomes are `revolute`, `prismatic`, `static/no_motion`, **or `ambiguous/unobservable`**. A confident false decision is worse than an explicitly ambiguous result.
10. **Modular dependency policy.** MoMa-SG is a *reference implementation*, not a mandatory runtime dependency. Inspect and pin its exact commit and license if adding `third_party/MoMa-SG` as a reference submodule; implement the required joint math using the existing PyTorch/NumPy stack to avoid introducing GTSAM and its binary dependencies into FreeArtGS without justification. Credit borrowed ideas and reproduce reference behavior with independent unit tests.

### Previously reported signals to reproduce, not treat as universal truths

- Black-box cam_001/002 can be better aligned geometrically while still estimating almost no lid rotation; cam_000 can estimate rotation more successfully despite being misregistered.
- Independent two-part segmentation under occlusion can label stationary base regions as moving; view-specific part IDs can disagree.
- The old prefilter has been reported to reject considerably more lid than base tracks; remeasure on the current local code.
- `cam_002` ICP has previously explained lid motion as apparent translation/lifting even with better masks; new model must demonstrate improvement **before** ICP.
- Robot-contact prompt/match experiments were polluted by the visible robot/tool; the hidden underside may lack any directly observed RGB-D normal.
- A previous storage/prismatic reconstruction worked well in one setting; preserve it as a regression control alongside failures.

---

## 1. Code-grounded baseline: audit these exact locations first

The following paths/functions were inspected in the public fork on 2026-11-16. Local line numbers, uncommitted changes and submodule contents can differ. Before coding, resolve and document the actual local call graph, shapes and transform conventions.

| Existing code | Observed behavior | Specific replacement/interface requirement |
|---|---|---|
| `tracking/tracker_toolkit.py::get_4d_traj_maps` | AllTracker 2D tracks are lifted using bilinear depth after replacing invalid zeros by `-inf`; mask sampled separately | Retain 2D observations when depth is invalid; only interpolate depth from valid same-surface samples; audit edge/depth bias. Return per-observation validity/confidence and original coordinates |
| `tracking/tracker_toolkit.py::get_visible_trajectories` | Requires visibility **in all window frames**, then filters global displacement outliers and local displacement-density | Preserve partially visible tracks, explicitly mark missing observations, audit every rejection; do NOT require first-frame observation or filter true moving tracks as spatial outliers |
| `estimotion_2part/track_2part.py::process_single_window` | Iterates from frame 0 to selected last index, fits `T1,T2`, propagates part maps, saves `optimized_transformations.npz` | Add optional joint-aware path driven by the same track observations, shared timestamps and multiview calibration; preserve legacy mode |
| `estimotion_2part/solve_transformation.py::solve_window_transformations`, `joint_optimize_segmentation_and_transforms` | Initializes fits and jointly optimizes part weights/transforms | Replace **within new mode only** with static/revolute/prismatic hypothesis optimization and persistent support assignments |
| `estimotion_2part/models/trans_models.py::Trans` | `T1`/`T2` are `nn.Parameter` tensors shaped `(num_T,4,4)`; blended transformed positions `(1-w)T1 x + w T2 x` | No learned raw matrices and no affine interpolation of physical material-point positions; exact per-hypothesis `SE(3)` and a probabilistic observation mixture |
| `estimotion_2part/models/trans_models.py::TransformLossComputer` | Track loss plus segmentation-related penalties and optional image projection terms | Define coherent, normalized likelihoods for 2D, 3D, stationary and joint predictions; control outliers and track quality |
| `estimotion_2part/process_part_segment.py` and `preprocessing/bundle_preprocess.py` | Propagates/expands part probabilities to dense masks, converts window masks, handles ping-pong ordering | Export persistent world-consistent static/moving support; zero/unknown moving support is allowed in an occluded camera. Keep source/output-frame mappings intact |
| `icp/color_icp_gpu.py` | Independently registers per-part point clouds and emits `global_poses_*.npy` and virtual `transforms_*.json` | New joint model is fit before, and independently of, ICP. Use ICP only as optional geometric cross-check; never pass its virtual frame as fixed physical extrinsic |
| `preprocessing/prepare_multiview.py::calibration_poses` | Reads fixed `T_WC` from COLMAP format and checks per-camera stationarity | Reuse as read-only physical pose source, assert no time variation; verify calibration provenance/units |
| `preprocessing/prepare_multiview.py::prepare_multiview` | Chooses one reference camera by pose validity/count/motion and transports its part motion to other views | New mode consumes **one shared world-frame joint trajectory**, independently optimized from all selected cameras. No winner-takes-all reference camera |
| `samuele/convert_standard_to_freeart.py::load_camera_entry` | Loads intrinsics from metadata and can override with COLMAP PINHOLE parameters | Use each camera's own `K`; don't rely on a single global `cam_K.txt` |
| `reconstruction/generate_ply.py::backproject`, `process_part` | Uses per-frame transforms and masked depth to seed PLYs | Canonical moving PLY must use correct joint inverse transform, static PLY physical `T_WC` |
| `preprocessing/estimate_joint.py::decide_joint_axis_type` | On upstream path makes a post-hoc revolute/prismatic decision using rotational span, then estimates joint | New Phase 1 outputs BOTH fitted candidates and uncertainty; existing estimator remains legacy comparator, and later stages must not silently re-classify new fits |
| `pipelines/freeartgs.sh`, `pipelines/tracking_dynamic_2part.sh`, `pipelines/train_local_datasets.py` | Orchestrate the legacy tracker, ICP, part-training, blend, joint fitting | Add an opt-in new mode, clean artifacts, and explicit new-to-old adapter; ensure branch-specific commands and defaults verified from `--help` |
| `pipelines/train_parts.sh`, `pipelines/twopart_blend.sh`, `pipelines/splatfacto_art.sh` | Nerfstudio training, optional camera optimizers, exports | In fixed-calibrated mode optimizer `off` for physical cameras; do not confuse `object pose optimization` with camera calibration |
| `samuele/export_real_robot_poses.py`, `docs/real_robot_pose_export.md` | Exports measured EE and calibrated TCP `metadata/robot_poses/poses_{full,rgbd}.npz`; `timestamp_s`, `frame_index`, `source_frame_index`, `tcp_position_m`, transforms, manifest | Load, validate and synchronize these; never use control target as measured TCP or auto-assume unknown EE/TCP frame |

**Important additional baseline caveat:** The fork's `docs/official_pipeline_review.md` describes a past `xyzw`/`wxyz` Gaussian quaternion mismatch and a subsequent correction. Inspect current local flags and test a 90° rotated anisotropic Gaussian at the actual GSplat boundary before interpreting rendering residuals as bad joint motion.

### MoMa-SG reference points — read exact source and adapt concepts, not bugs

- `moma_sg/articulation/point_estimator.py`: `identify_cloud_pairs*`, `prepare_estimation_data`, `twist_transform`, `prismatic_transform`, `revolute_transform`, `twist_type_residual_pairs`, `compute_type_prior`, `estimate_dual_twist_numeric`, `estimate_articulation_model_*`.
- `moma_sg/articulation/filtering.py`: `filter_tracks_by_twist`, partial-visibility and tracking filters.
- `moma_sg/articulation/synth_experiments.py`: reference revolute/prismatic synthetic generators and axis-error plots.

**Specific reimplementation cautions:** The inspected MoMa-SG `revolute_transform()` return path at lines ~492–515 rotates points about the origin, with the translation term commented out. That is NOT a full arbitrary-hinge transform; use `Exp([u; -u×o] q)` or the exact hinge formula below. Its `twist_transform()` uses GTSAM `Pose3.Expmap`; independently test convention equivalence. Its `type_residual()` and cosine prior are useful precedents but do not assume the published hyperparameters are calibrated to our small real RGB-D objects. Separate model selection and a calibrated statistical comparison are needed. Never blindly copy its heuristics or conflate CoTracker3 data formats with AllTracker.

**Reference links:**
- Public fork: `https://github.com/SamueleMara/FreeArtGS`.
- MoMa-SG: `https://github.com/robot-learning-freiburg/MoMa-SG`.
- Paper and project: `https://arxiv.org/abs/2602.16356`, `https://momasg.cs.uni-freiburg.de/`.

---

## 2. Frames, data contracts and transformations

Adopt the convention **`T_AB` maps coordinates in frame B to coordinates in frame A, using column vectors**.

- `W`: calibrated reconstruction world frame; `B`: recorded robot base; `C_c`: OpenCV camera `c`; `M_0`: reference moving-part frame; `S`: force sensor frame.
- Each physical camera: constant `T_WC[c] ∈ SE(3)`, per-camera `K[c]`, image size. `T_CW = inverse(T_WC)`.
- Robot TCP: measured/derived `p_tcp^B(t)`; transform into reconstruction world `p_tcp^W(t)=T_WB p_tcp^B(t)`. **DO NOT implicitly set `T_WB=I` unless the dataset explicitly proves that camera calibration and robot base already share the same world frame.**
- Recorded camera/robot timestamps may differ; measure/prove their sync. `source_frame_index` is NOT automatically identical to rendered output index. Use the explicit conversion mappings; separate `frame_index`, `source_frame_index`, `timestamp_s`, and optional data-stage/ping-pong indices.
- Track observation: `(camera_id, track_id, time_index, timestamp, uv[2], uv_cov[2×2], depth_z_optional, depth_valid, mask_valid, visibility, confidence, source_frame, provenance)`. ID scope is **per camera**; no cross-camera ID equality or material correspondence may be invented. Two views can contribute independent points to the same joint-model fit.
- Valid 3D unprojection, only if depth is trustworthy: `x^W=T_WC[c] * (z K_c^-1 [u,v,1]^T)` with appropriate homogeneous coordinates. Store 3D covariance/proxies and edge-contamination flags. Missing depth does **not** erase the valid 2D track.
- Track is allowed to start/end at any time and may reappear after occlusion with independently verified identity; never infer identity purely from nearest 3D point. At least one depth-backed 3D anchor (or independently triangulated calibrated association) is needed to promote a 2D-only trajectory into an anchored material point.
- `M_base`: static material points; `M_move`: moving material points; `unknown`: ambiguous/occluded/outlier. Do not enforce a nonzero moving portion in every camera or frame.

### Required serialized input format

Implement a read-only loader returning arrays with shapes, missingness and explicit frame map; recommended sparse observation table is Parquet or NPZ+JSON, not an `(N,T,3)` array silently padded with zeros:

```text
TrackObservation[]:
  camera_id, track_id, time_idx, source_frame_idx, timestamp_s,
  u_px, v_px, uv_confidence, uv_valid,
  z_m, depth_valid, depth_edge_flag, mask_object_valid,
  x_w_m[3] if 3D valid, cov_x_w[3,3] if available,
  rgb_or_feature_reference, provenance
Global input:
  camera_id -> K, T_WC, width, height, calib_sha
  time_idx -> exact RGB-D timestamp and synchronized source-frame ids
  optional robot: t_robot, p_tcp_base[3], T_B_TCP[4,4], wrench, contact_flag,
                  T_WB, wrench_calib, sync_offset + uncertainties
```

**MUST check raw AllTracker outputs first.** If intermediate tracker data are not persisted, add non-invasive hooks to save them; do not claim the optimizer can be retrofitted from `optimized_transformations.npz` alone.

---

## 3. Exact joint kinematics (no unconstrained rigid parameters)

Work with canonical **world-aligned reference geometry** at `q(t_ref)=0`. Let `X_i` be the canonical position of moving material point `i`, and `A_H(q)` its exact `SE(3)` transform. The stationary component has `A_S=I`.

### 3.1 Prismatic model `P`

Unknowns: constant **unit** axis `u_P ∈ S²` (two DOF) and `d_t` in **meters**, with `d_ref=0`.

\[
R_P(d)=I,\quad t_P(d)=d u_P,\quad
A_P(d)=\begin{bmatrix}I&d u_P\\0&1\end{bmatrix},
\qquad \hat X_{i,t}=X_i+d_t u_P.
\]

All object material points have the **same** instantaneous velocity:

\[
\boxed{v_i(t)=\dot d_t u_P,\quad \omega(t)=0.}
\]

The velocity decomposition, per observed point, is

\[
v_{\parallel,i}=u_P^\top v_i,\quad
v_{\perp,i}=(I-u_Pu_P^\top)v_i;
\quad v_{\perp,i}=0,\quad v_{\parallel,i}=\dot d_t.
\]

A prismatic axis **has no meaningful hinge origin**. Its representation must not report a physically recovered pivot point.

### 3.2 Revolute model `R`

Unknowns: fixed unit axis `u_R ∈ S²` (two DOF), hinge line through `o_R` (two DOF after gauge `u_R^T o_R=0` in an explicitly chosen local origin), joint angles `θ_t` in **radians**, with `θ_ref=0`. The hinge is a line; location *along* `u_R` is unidentifiable. For numerical conditioning, choose the gauge in coordinates centered at the tracked object and convert the axis-line back to world coordinates for output.

\[
R_R(\theta)=\operatorname{Exp}(\theta[u_R]_\times),\quad
A_R(\theta)x=o_R+R_R(\theta)(x-o_R).
\]

Equivalently:

\[
A_R(\theta)=\begin{bmatrix}R_R(\theta)&(I-R_R(\theta))o_R\\0&1\end{bmatrix}.
\]

In twist notation, ordering **`ξ=(ω,v)`**, the revolute generator is `ω=u_R`, `v=-u_R×o_R`, with **pitch zero** `ωᵀv=0`, and `A_R(θ)=Exp_SE3(θ ξ^)`. Assert both matrix forms agree within numerical precision.

Instantaneous material velocity:

\[
\boxed{v_i(t)=\dot\theta_t\,u_R\times\big(X_i(t)-o_R\big).}
\]

Define the perpendicular hinge radius

\[
r_i=(I-u_Ru_R^\top)(X_i(t)-o_R),\quad r_i=\|r_i\|.
\]

For `r_i>0`, define `e_{r,i}=r_i/r_i`, `e_{\theta,i}=u_R×e_{r,i}`, and `e_z=u_R`. Then:

\[
\boxed{v_{r,i}=0,\quad v_{z,i}=0,\quad v_{\theta,i}=r_i\dot\theta_t.}
\]

The **tangent direction is different for each point**, and the tangential speed grows with radius. The **joint axis, hinge line and angular velocity are common** to all points/times. A point arbitrarily close to the hinge cannot reliably constrain angular velocity by division by `r_i`; report low observability.

### 3.3 General instantaneous rigid-flow identity for initialization

\[
\boxed{v_i(t)=a_t+\omega_t\times X_i(t)},\quad
v_i-v_j=\omega_t\times(X_i-X_j).
\]

For any pair `(i,j)`, the last equation eliminates the shared translation. Estimate `ω_t` robustly by a least-squares system whose rows are `-[X_i-X_j]_×`; then check whether `ω_t` has a **consistent axis direction** and whether the recovered velocity-field translation obeys the zero-pitch hinge constraint. If `ω_t≈0`, test the prismatic model; do not infer rotation solely because the free rigid-flow fit has nonzero noise-induced `ω`.

`a_t` and `ω_t` in this initialization are an **unconstrained instantaneous diagnostic**. They are **not** the final articulation: the final joint has a single fixed axis/hinge and one scalar `q_t` per time.

### 3.4 Exact finite-time motion: mandatory for final fitting

Do not equate the finite-difference chord `(X_{t+Δ}-X_t)/Δ` with the instantaneous circle tangent for large `Δθ`; doing so creates spurious radial motion. Use exact relative transformations

\[
\boxed{X_i(t_b)=A_H(q_b)A_H(q_a)^{-1}X_i(t_a)}
\]

for arbitrary observation pairs, including tracks that do not exist at frame zero. This also prevents cumulative sequential drift. Optimize global `q_t` and canonical anchors jointly; avoid composing noisy frame-by-frame 4×4 estimates into the sole motion history.

### 3.5 Joint sign and gauge equivalence

`(u,q)` and `(-u,-q)` produce the same physical motion. Revolute origins differ along `u` without changing the hinge line. Align axis signs only using explicitly documented conventions, **not GT during fitting**; compare axis direction sign-invariantly and hinge location by line-to-line distance. Fix `q_ref=0` and unit axis; do not add hidden scale freedom between the axis norm and `q`.

---

## 4. Core Phase-1 discovery: shared multiview static/revolute/prismatic inference

### 4.1 Model variables and assignments

For each hypothesis `H in {P,R}` optimize jointly:

- Global joint geometry `J_H` (constant across the full sequence and both cameras).
- Scalar joint values `q_{H,t}` at synchronized timestamps; optional smooth temporal interpolation, with `q_ref=0`.
- Canonical 3D anchors `X_i` per *material* track, initialized using its first reliable depth/triangulated observation, with anchor uncertainty.
- Persistent moving-part probabilities `π_i ∈ [0,1]` and an explicit third outlier/unknown probability. Assignments belong to tracked **material IDs**, not arbitrary per-frame pixel labels. A full-camera frame may contain zero visible moving tracks.
- Optional static-base geometry and camera-specific 2D observation covariances; all physical camera poses remain fixed.

Do not model a physical material point's trajectory as `(1-π) X_static + π A_H X_move`. **That affine path is generally not a rigid trajectory.** Instead treat static/moving/outlier as a *mixture of observation hypotheses*, update probabilities from residual likelihoods, and form render geometry only after support is established.

### 4.2 Observation likelihoods (calibrated, noise-aware)

For valid 3D observations `y^W_{i,c,t}`, predicted moving/static positions are:

\[
\mu^M_{i,t}(H)=A_H(q_t)X_i,\qquad\mu^S_i=X_i.
\]

A reasonable robust 3D residual is

\[
r^{3D,M}_{i,c,t}=\Sigma_{i,c,t}^{-1/2}(y^W_{i,c,t}-\mu^M_{i,t}),
\quad r^{3D,S}=\Sigma^{-1/2}(y^W_{i,c,t}-\mu^S_i).
\]

When depth is missing or suspect, use 2D reprojection instead:

\[
r^{2D,M}_{i,c,t}=\Sigma_{uv}^{-1/2}\left[u_{i,c,t}-\pi\left(K_cT_{CW_c}A_H(q_t)X_i\right)\right],
\]

and corresponding static prediction. Do **not** count the same pixel as statistically independent 3D and 2D evidence without an explicit joint measurement covariance or sufficiently justified loss weighting. Prefer image-space reprojection and an independent depth-axis residual (using calibrated pixel/depth covariance), or a correctly whitened joint observation model.

Estimate the pixel-to-3D covariance through a calibration/unprojection Jacobian if uncertainties exist; include depth-dependent noise, cross-camera calibration uncertainty, and inflated errors near edges. All residuals must remain in meaningful units or be divided by their standard deviations. No division by observed displacement magnitude as a means of making stationary noise look equally informative as moving tracks.

A probabilistic static/moving/outlier objective can be written

\[
\boxed{\mathcal L_{\mathrm{obs}}(H)
=-\sum_i\log\left[
\pi_{i,S}p(y_i\mid S)+\pi_{i,M}p(y_i\mid M,H)+\pi_{i,O}p(y_i\mid O)
\right]},\quad \sum_k\pi_{i,k}=1.
\]

Here `p(y_i|component)` aggregates **all valid observations for track i**, including missingness. Use a specified robust distribution (e.g. Huberized Gaussian / Student-t) for camera depth and an explicit broad outlier model; do not let an arbitrary outlier likelihood absorb all difficult moving tracks. Fix or calibrate noise/outlier models on held-out data, not per candidate to make one model win. Add conservative spatial/appearance coherence for assignments **only where supported** by observations.

### 4.3 Initialization without prematurely choosing type

1. Preserve all valid 2D AllTracker trajectories, plus per-observation RGB-D lifted 3D points and provenance. Build a motion/track-confidence diagnostic; compare fast-moving and nearly static regions. Do not discard a moving track merely because its displacement is rare.
2. From confirmed camera-fixed geometry, estimate noise-scaled evidence for **static** tracks: constant world position fits within measurement uncertainty. Use this for an initial stationary core, not as a forced majority label.
3. Generate motion-support proposals by robust local rigid-flow fits over short valid overlapping pairs of timestamps: spatially coherent position-dependent velocity indicates rotation; spatially uniform velocity indicates translation. Use multiple seeds (e.g. robust randomized subsets) and multiple time intervals. Rank by data likelihood/coverage, not raw number of static points.
4. Initialize **both** models from those proposals; `P`: robust normalized direction of moving displacements and scalar `d_t`; `R`: robust `ω_t` axis consistency, hinge-line LS from `a_t≈-ω_t×o`, and scalar `θ_t`, or direct weighted rigid finite-pair estimates. Exclude invalid angular/translation velocities and report rank/conditioning. Try multiple plausible hinge seeds for small arcs.
5. Initialize unknown or unobserved `q_t` using neighboring timestamps **only as interpolation with uncertainty**, not as an extra observed pose. Reinitialize per-track canonical anchor using its first valid observation and the current `A_H(q_t)^{-1}`. Do not require common first-frame visibility.
6. Fit each candidate over the **same** observation set, using comparable optimization budget, regularization, robust-loss and held-out splits. If there is insufficient motion, spatial span or temporal overlap, return `ambiguous/no_motion`.

### 4.4 Iterative optimization (EM-like, but retain exact support semantics)

A. **E-like step:** With joint parameters fixed, evaluate each track's full temporal static/moving/outlier likelihood, update persistent posterior responsibilities; retain ambiguous and missing states. Include object-mask / feature-based spatial continuity only as a separately logged term. Do not enforce one moving cluster in every camera.

B. **M-like geometry step:** With posteriors fixed, optimize joint axis and hinge origin (for `R`), global `q_t`, and canonical anchors under exact transforms with robust, uncertainty-scaled reprojection/depth residuals. Parameterize axis on `S²` (normalization with proper gradient / local chart), hinge origin in the perpendicular plane, `q_ref=0`. Use double precision for initialization/geometry fitting as needed; support torch autodiff and gradient-check key Jacobians.

C. **Track-recovery step:** Revisit initially excluded or partially observed points using their own valid 2D measurements and the current shared model. A track is promoted only if its predicted motion is supported across several independent observations; preservation of occluded frames is allowed, but do not fabricate depth or motion observations. Record how many genuine moving tracks are gained and how many static outliers leak in.

D. **Visual feedback step:** Evaluate reprojected track positions and, when available, image-feature/patch consistency and actual masked RGB-D rendering (Section 6). Update support and model only based on valid gradients/measurements; keep the effect of every loss separately measurable.

E. Repeat until held-out loss/parameter changes stabilize. Do not swap part identities based on per-camera motion scores. Preserve several candidate local optima and report their scores; do not use final GT axes to choose seeds.

### 4.5 Temporal regularization

Joint coordinate trajectories can have starts, stops and reversals. Regularize acceleration/jerk **only relative to actual timestamps** and measurement uncertainties, e.g.

\[
\mathcal R_q=\sum_t\frac{\|\dot q_{t+1}-\dot q_t\|^2}{\sigma_{\ddot q,t}^2},
\]

but never hard-code monotonicity. A stronger smoothness prior should not hide that the robot moves while the object stays stuck; confirmed contact onset can inform segmentation of intervals, not force `qdot != 0`. For very small motions, report uncertainty rather than claiming a stable hinge.

---

## 5. Velocity-field loss, tangent/normal agreement and finite-time consistency

The user's proposed diagnostic is mathematically sound **when applied to tracked material points**:

\[
\mathcal L_{v,P}=\sum_{i,t} w_{it}\|v_i(t)-\dot d_t u_P\|^2
\]

\[
\mathcal L_{v,R}=\sum_{i,t}w_{it}\|v_i(t)-\dot\theta_t u_R\times(X_i(t)-o_R)\|^2.
\]

For `R`, this equals the sum of squared **radial**, **axial**, and **tangential-amplitude** discrepancies in the local `(e_r,e_theta,u)` basis. For `P`, it equals squared perpendicular motion plus agreement of the parallel scalar velocity across all points. A free per-frame rigid twist is useful as initialization; for the final model the joint axis and hinge are global.

**Numerical procedure:**

- Compute local velocities only from valid, time-spaced observations with a derivative method robust to irregular timestamps; do not use zero-valued missing data.
- Prefer 3D velocity estimates with uncertainty covariance, or work directly with exact finite displacement/2D reprojection for long time gaps.
- Use `L_v` to **initialize, diagnose and optionally regularize**; the principal fit should be the exact finite-motion observation likelihood from Section 4. If both velocity and position residuals derive from the same tracks, they are statistically correlated: avoid pretending they are independent evidence or double-counting information.
- Log per-time/per-camera moving-point radial, axial and tangential residuals for revolute; perpendicular and common-speed disagreement for prismatic; visualize trajectory arrows and their predicted tangent directions.
- If a segment is very small, far from the hinge or its observed span is narrow, a curved trajectory can appear linear. Compute rank/condition and output `ambiguous` where appropriate.

### Pairwise consistency without a known hinge origin

For true moving points at the same time:

\[
\boxed{v_i-v_j=0\;(P),\qquad v_i-v_j=\dot\theta_t u_R\times(X_i-X_j)\;(R).}
\]

This is an informative *origin-invariant* discriminant and can generate robust hypotheses even when the object origin is inaccurate. It is **not** sufficient if all points occupy a tiny local patch or if velocities are dominated by different depth biases. Use pairs across spatially separated supported regions with covariance weighting; log the effective baseline distribution and rank.

---

## 6. Real visual feedback in Phase 1 (not only a final test)

The redesigned model must use visual data to judge and refine hypotheses rather than fit 3D tracks and stop.

### Level V1 — before any Gaussian model exists (mandatory)

- **Camera-fixed 2D reprojection:** residual of each tracked pixel to the predicted projection of its world/canonical material point; works even when later frames have invalid depth.
- **Valid surface-depth comparison:** compare projected predicted point depth with measured metric depth only when the point is actually visible/unoccluded and the depth is valid; reject edge/background interpolation. Use frontmost visibility consistency where feasible, not the assumption every projected point must be visible.
- **Image feature/appearance consistency:** compare existing AllTracker/DINO features or image patches at predicted pixel positions for the same track using an explicit visibility-aware robust loss; inspect the actual feature tensor conventions and avoid assigning pixel colors to occluded tracks.
- **Object silhouette support:** predicted moving/static visible points should lie inside independently supplied whole-object masks when actually visible; absence of a point from the visible mask because it is occluded is not itself a moving-part error. Do not use incorrect 2-part masks as GT.
- **Multi-view joint score:** aggregate all observations through each camera's own `K_c,T_WC_c` in the common world frame; do NOT require matching tracked IDs between cameras.

### Level V2 — coarse Gaussian rendering feedback (optional only after correct initialization)

- Inspect the current local Nerfstudio/GSplat rasterizer and Gaussian model interfaces **before** connecting them; do not invent an API. Use the already installed renderer, respecting `xyzw`/`wxyz` conversion and fixed camera matrices.
- Initialize a **coarse canonical 3DGS/point-splat proxy** from validated multiview RGB-D observations and the current joint fit. A proxy is allowed only if its provenance and coverage are explicit. Do not make expensive full GS training a prerequisite for Phase-1 motion discovery.
- For each `c,t`, render from the physical camera with canonical static points fixed and moving points transformed by exact `A_H(q_t)`; compare masked RGB, metric depth, visibility/alpha and whole-object silhouette. Restrict to actual observed pixels and confidence-bearing regions; avoid background and hidden robot-induced surfaces.
- A canonical moving-part Gaussian covariance must rotate with the rigid part. Do not transform its centroid without rotating its covariance orientation; handle SH view dependence according to current renderer. Avoid mixing virtual camera and moving Gaussian transforms in the same rendering path (double motion).
- If the existing training expects static Gaussians and dynamic part-frame virtual cameras, derive those virtual poses **exactly** from the candidate articulation (Section 11) and prove rendered equivalence on a synthetic fixture.
- Alternate a small **geometry step** and a **joint step** only after the visual-only track fit passes synthetic tests. Freeze/support limits to avoid Gaussian geometry absorbing gross pose errors. Validate on held-out views/timestamps; never report training PSNR alone as kinematic success.
- Log RGB/depth/mask loss per camera and per frame. A visually good static reconstruction with poor moving-part dynamics is a failure, not a success.

**No oracle part masks in the default inference path**; if used for diagnostic ablation, clearly mark `oracle` in output and disable before final comparison.

---

## 7. TCP contact: exact sliding-point kinematics, separately for both candidates

The robot tip is approximated as a **3D point**, without a tool mesh or tool-surface normal. The saved, calibrated and synchronized TCP is the reliable contact location **only during verified contact intervals**. This assumption does NOT imply a rigid grasp or a fixed object material point.

Let `p_t=p_tcp^W(t)` be the known point position and `v_tcp=dp/dt`. For either candidate `H`, its canonical surface contact coordinates are

\[
\boxed{x_{c,H}^0(t)=A_H(q_t)^{-1}p_t.}
\]

Different `t` generally produce **different material coordinates** `x^0`: this is physically correct sliding.

### 7.1 Revolute contact

\[
\boxed{x_{c,R}^0(t)=o_R+R_{u_R}(-\theta_t)(p_t-o_R).}
\]

Material-surface velocity at this *instantaneous* world contact location:

\[
\boxed{v_{M,R}(p_t,t)=\dot\theta_t u_R\times(p_t-o_R).}
\]

Relative TCP/object velocity, measured in the world frame:

\[
\boxed{s_R(t)=v_{tcp}(t)-v_{M,R}(p_t,t).}
\]

The canonical derivative satisfies `dot x^0_R=R_u(-θ) s_R`. Thus the TCP may follow a line while the moving material points follow circles; **it is wrong to force `s_R=0`**. If a reliable object contact-surface normal `n_R(t)` is available:

\[
\boxed{n_R(t)^T s_R(t)=0\quad\text{(maintained contact)}.}
\]

When the contact normal is not known, do not set it equal to the TCP tangent, contact force direction, or an arbitrary plane estimated solely from the inverse-transformed TCP curve.

### 7.2 Prismatic contact

\[
\boxed{x_{c,P}^0(t)=p_t-d_t u_P,\qquad
v_{M,P}(p_t,t)=\dot d_t u_P,\qquad
s_P(t)=v_{tcp}(t)-\dot d_t u_P.}
\]

For maintained contact with reliable normal:

\[
\boxed{n_P(t)^T s_P(t)=0.}
\]

Again, a prismatic motion may coexist with **nonzero sliding** along the drawer face; a linearly moving TCP is not proof that it is rigidly attached to the drawer.

### 7.3 Contact position constraints: surface must have *independent support*

If an observed/reconstructed canonical moving-part surface `S0` is reliable, contact imposes

\[
\boxed{d_{S0}(x_{c,H}^0(t))\approx0,\quad t\in\text{verified contact}.}
\]

This is a point-to-object-surface distance, **not** a distance to the nearest robot point. Find candidate surface patches from calibrated `cam_001/002` object-only RGB-D, validating moving-part membership, occlusions, depth discontinuities and local planarity. Do not infer an exact normal from a single nearest-neighbor point. A hidden underside may be unobservable and should yield `UNAVAILABLE` rather than a false contact loss.

A locally planar material surface has `n0^T(x^0-a0)=0`, with world normal `n_W(t)=R_H(q_t)n0`. For `P`, `R_H=I`. Only activate plane constraints when that actual face is observed or its geometry is independently justified. Do not fit a new free plane separately for every hypothesis and then count zero residual as evidence.

**Mathematical degeneracy:** For straight TCP `p(t)=p0+vt`, any prismatic inverse trajectory `p0+vt-d_tu` lies in `span{v,u}` plus `p0`, so a free plane always fits. Revolute inverse trajectories for a TCP line in a plane perpendicular to the axis can also lie in an uninformative plane. Contact planarity alone is NOT a joint-type classifier. Test these counterexamples explicitly.

### 7.4 Sliding is tangential to the *physical contact surface*, not necessarily to TCP path

When a reliable unit contact normal `n_H(t)` exists, define the tangential projector

\[
P_T=I-n_Hn_H^T,\qquad s_{H,T}=P_T s_H.
\]

Maintained contact requires `n_H^T s_H≈0`; `s_{H,T}` may be nonzero and may not align with `v_tcp`. This distinction must be explicit throughout the code. A point tip has **no material surface tangent** of its own; the relevant tangent plane belongs to the object. The contact may slide, stick or temporarily detach. Do not assume every contact sample has nonzero sliding speed.

### 7.5 Demonstrative 2D pin-in-slot/flat-lid analytic test

Hinge `o=(0,0)`, axis `+z`, radial lid contact line at `θ`, TCP `(x(t),h)` on fixed horizontal line `y=h`:

\[
\boxed{x(t)=h\cot\theta(t),\quad
\theta(t)=\operatorname{atan2}(h,x(t)),\quad
\dot\theta=-h\dot x/(x^2+h^2).}
\]

Canonical contact location slides to different radii `r(t)=sqrt(x(t)^2+h^2)`. Numerically verify the TCP is linear, lid material velocities are circular, canonical points lie on the contact line, and **full-vector equality `v_tcp=v_M` fails except special instants**, while the normal-velocity equality holds.

---

## 8. Wrench/force signal: parallel/perpendicular TCP force proportions plus physically consistent joint selection

### 8.1 Nonnegotiable measurement validation

- Read *measured* wrench; document whether it is object-on-tool or robot-on-object, force sensor frame `S`, sign convention, sensor origin, whether static zero was applied, and compensation for orientation-dependent tool gravity/inertia. Transform vectors and moments correctly into `W`; do not transform a force vector with a translation.
- If wrench measured about sensor origin and adjusted to TCP, `f^W=R_WS f^S`, `tau_TCP^W=R_WS tau_S+(p_S^W-p_TCP^W)×f^W`, with **sign chosen from the measurement convention**. This moment is distinct from the moment about the candidate hinge.
- Estimate/verify `T_WB`, `T_BS`, robot-camera time offset and contact flags. The existing EE/TCP export contains poses/timestamps; it does not by itself prove that wrench samples are present, synchronized, bias-corrected or in world coordinates.
- Exclude near-zero net contact force, unreliable sensor saturation, unknown sign, no-contact intervals, and near-zero TCP velocity from force-tangent direction calculations. Save reasons for exclusion.
- Treat any net sensor wrench due to other simultaneous contacts, tool inertia or uncompensated gravity as **not** a valid single-point object-contact force.

### 8.2 Measured force decomposition relative to TCP path

For speed `||v_tcp||>epsilon_v`, define the TCP tangent `t=v_tcp/||v_tcp||`. With `f_obj^W` the force **exerted by the robot on the object**:

\[
\boxed{F_\parallel=f_{obj}^T t,\quad
\boldsymbol F_\perp=f_{obj}-F_\parallel t,\quad F_\perp=\|\boldsymbol F_\perp\|.}
\]

\[
\boxed{\rho_\parallel=\frac{|F_\parallel|}{\|f_{obj}\|},\quad
\rho_\perp=\frac{F_\perp}{\|f_{obj}\|},\quad
\rho_\parallel^2+\rho_\perp^2=1.}
\]

Store **signed** `F_parallel`, `cos_phi=F_parallel/||f||`, full normalized `f/||f||`, both ratios and force magnitude. The ratios are not independent measurements; do not count each as an independent likelihood.

These ratios are potentially informative because a rotating object-fixed contact normal can rotate relative to an almost constant TCP trajectory, while a translating planar face's normal remains world-fixed. **But this is conditional on actual contact geometry and friction; it is not a universal revolute-versus-prismatic rule.** A prismatic drawer can be pushed at an angle, a revolute lid can receive an almost tangential TCP push, and friction may change both components.

### 8.3 Joint mechanics and contact power

At known `p_tcp`:

\[
J_{c,H}(p)=\begin{cases}u_R\times(p-o_R),&H=R\\u_P,&H=P.\end{cases}
\]

\[
\boxed{v_{M,H}=J_{c,H}(p)\dot q_H,
\quad Q_H=f_{obj}^T J_{c,H}(p),
\quad P_{joint,H}=Q_H\dot q_H.}
\]

`Q_R` is torque `[N·m]`, `Q_P` force `[N]`; **never compare their magnitudes without a physically meaningful normalization**. TCP input power `P_tcp=f_obj^T v_tcp`. For maintained single-point passive contact,

\[
\boxed{D_H=P_{tcp}-P_{joint,H}=f_{obj}^T (v_{tcp}-v_{M,H})=f_{obj}^T s_H\ge0}
\]

under the convention `f_obj` = robot-on-object and with no other contact-work channels. Use a **one-sided uncertainty-scaled negative-dissipation penalty**, not `D_H^2` (dissipation is typically positive during sliding). Do NOT use this necessary condition as sufficient evidence of a particular type: both can pass it, and wrench sign errors can invert the result.

### 8.4 Predict force *direction* from an actual surface/friction model, not arbitrary angles

When a validated object contact plane/normal is available and the contact is sliding, define `n_in,H` pointing **into** the object, with `s_H,T` the relative tangential slip (the sign convention is controlled by the measured force on the object). Under a simple Coulomb model:

\[
\boxed{\hat f_{pred,H}\ \propto\ n_{in,H}(t)+\mu\,\frac{s_{H,T}(t)}{\|s_{H,T}(t)\|}},\quad N_t\ge0,\quad\mu\ge0.
\]

Normalize the vector; predict the **signed** projection along TCP tangent and perpendicular magnitude:

\[
\cos\phi_H=\hat f_{pred,H}^T t,
\quad \rho_{\parallel,H}=|\cos\phi_H|,
\quad \rho_{\perp,H}=\sqrt{1-\cos^2\phi_H}.
\]

Fit at most a **shared or slowly varying, physically bounded** friction coefficient per material/contact regime, never a free `mu_t` and free normal per time independently under each candidate. If slip speed is indistinguishable from zero, Coulomb slip **direction is undefined**: switch to a stiction cone/inequality if support exists, or disable directional friction likelihood. If the contact normal/geometry is unknown, disable this particular ratio-prediction loss; still report the *measured* ratios and the weaker power test. A floating plane fitted exclusively to the TCP curve cannot justify predicting `n`.

For a known planar face `n_0`, `n_R(t)=R_u(θ_t)n_0` versus `n_P(t)=n_0`; the revolute hypothesis predicts a rotating normal. Compare measured full `f_hat` (with proper sign and uncertainties), not merely the unsigned ratios. Report agreement as cosine/angular error and held-out likelihood.

### 8.5 Factorized, confidence-aware force/contact residuals

Define contact validity `c_t`, force validity `f_t`, surface/normal support `s_t`, and observability `o_t` independently from model fitting. For example:

\[
L_{contact,H}=\sum_t c_t s_t\,\rho\!\left(\frac{d_{S_0}(A_H(q_t)^{-1}p_t)}{\sigma_{c,t}}\right)
\]

\[
L_{normal\_vel,H}=\sum_t c_t s_t\,\rho\!\left(\frac{n_H^T(v_{tcp}-v_{M,H})}{\sigma_{v,t}}\right),
\]

\[
L_{force\_direction,H}=\sum_t c_t f_t s_t\,\rho\!\left(\frac{\angle(\hat f_{obj},\hat f_{pred,H})}{\sigma_{\phi,t}}\right),
\]

\[
L_{passivity,H}=\sum_t c_t f_t\,\rho_+\!\left(\frac{\max(0,-D_H-\delta_t)}{\sigma_{P,t}}\right).
\]

The position-on-surface and normal-velocity terms are kinematically related; do not claim independent information merely by summing them. Choose one as primary and the other as a diagnostic or account for correlation. `δ_t` is a measured noise margin, not a tuning trick to hide negative dissipation.

Use physical terms **only after** visual-only candidates exist and have been independently evaluated. Fit both models under identical contact data and contact support. If valid contact normals are available in only 5/30 frames, report that fact and do not turn missing measurements into a confident preference.

### 8.6 What TCP and force can genuinely improve upstream

1. **Temporal cue:** contact onset can identify likely intervals of object actuation, but not label every track and not guarantee motion.
2. **Known geometric point:** candidate joint motion must keep the TCP on a **supported object contact surface** during maintained contact; do not nearest-neighbor match TCP to arbitrary robot points.
3. **Normal-slip cue:** candidate object surface velocity should match TCP **normal component**, allowing tangential slip.
4. **Force decomposition:** candidate rotating/translating surface + friction model should explain how `F_parallel:F_perp` and signed `f_hat` evolve over time, not merely whether tangential force is large.
5. **Generalized actuation/passivity:** exclude physically contradictory motion/force explanations where assumptions are met, without treating energy inequality as a unique classifier.
6. **Segmentation feedback:** after visual geometry is credible, contact likelihood can weakly favor the moving cluster whose surface actually contacts TCP; never seed a moving label by simply marking pixels around projected TCP as moving—those often belong to the tool.

---

## 9. Unified optimization schedule, model comparison and observability

### 9.1 Staged objective, with ablations

For each `H∈{P,R}` define:

\[
\boxed{
L_H= L_{tracks,2D/depth}+\lambda_{app} L_{appearance}
+\lambda_{sil} L_{silhouette}
+\lambda_{time}R_{time}+\lambda_{part}R_{part}
+\lambda_c L_{contact}+\lambda_f L_{force}+\lambda_D L_{passivity}.
}
\]

Every term needs units/normalization, an exact active-data mask, and a logged value. The baseline implementation should use the calibrated statistical likelihood with **no robot terms**; then enable contact and force separately. Avoid weighting arbitrary raw pixel, metre, radian and Newton errors as though they were commensurate.

**Schedule:**

- **S0 / diagnostic:** preserve raw tracks, calibrated cameras, masks, robot metadata; quantify noise, frame alignment, and valid moving support.
- **S1 / vision-only initializer:** fit static+P and static+R from the same observations, no force, no TCP geometry, no GS training.
- **S2 / global visual refinement:** optimize full-sequence `J_H,q_t,π_i,X_i` with 2D and depth evidence, then optional V2 renderer; hold out some frames/cameras to measure generalization.
- **S3 / contact:** on verified contact frames with independently supported moving surface, add TCP-on-surface/normal-slip consistency; compare against S2 without changing data split.
- **S4 / wrench:** add physically verified direction/ratio and passivity tests; fit a shared contact-friction parameter only if observable. Compare with S2/S3 and wrench-only null controls.
- **S5 / downstream 3DGS:** export candidate states to FreeArtGS when the geometry/model is supported; run a short GS integration smoke test first, then controlled full training if justified.

Do not simultaneously turn on all terms, change filtering, change cameras, change rendering and conclude “force fixed the system.” Each delta must have a separately reported effect.

### 9.2 Fair joint-type selection

A revolute hypothesis has extra geometry degrees of freedom, so it may weakly approximate a prismatic field over a short arc or with a remote hinge. Choose type by **held-out predictive evidence**, not raw minimized training residual. Report:

- Weighted 2D reprojection and depth negative log likelihood **per valid observation**, separated by camera/interval/occlusion status.
- Generalized physical contact likelihood on the SAME independently accepted contact samples; report the number of valid samples for each term.
- Effective parameter count, conditioning, hinge-radius/axis uncertainty, motion span and actual likelihood improvement; use model-complexity penalty (e.g. BIC-type diagnostics where statistical assumptions justify it), not ad-hoc favoring of either hypothesis.
- Prediction improvement on held-out timestamps and, when enough data are available, held-out camera(s); do not select using GT/annotation.
- Null/no-motion competitor. If neither joint is identifiable above the measurement noise, output `ambiguous` with both candidates preserved.
- Axis sign ambiguity handling and meaningful line-distance uncertainty for R; no pivot metric for P.

A force-ratio classifier **must abstain** when contact normal/surface is not independently supported, when force vectors are not calibrated/synchronized, or when the motion produces nearly indistinguishable predicted force directions.

### 9.3 Observability matrices

At minimum report the singular spectrum/condition of the motion-fitting Jacobian `∂r/∂(u,o,q,X_i)` after removing gauge freedoms; range of per-point distances to R axis; spatial span of points; measured rotation and translation span; per-time number of 2D/3D supported tracks; cross-camera overlap/visibility; validity of derivative estimates; and ratio of candidate predictive-score difference to its uncertainty. No numerical “confidence” without a stated construction.

Specific degeneracies to detect: `q≈0`; small arc; remote hinge approximating translation; points lying on or nearly parallel to hinge; tracks from only one tiny patch; one-camera occlusion; drift-induced apparent angular velocity; unknown robot-to-camera transform; straight TCP and freely fitted contact plane; forces near zero; missing or mis-signed wrench; contact dominated by tool or static base.

---

## 10. Concrete implementation modules and function/API contracts

**Create additive, isolated implementation** (paths may be adjusted to local package conventions, but justify changes):

```text
estimotion_2part/joint_discovery/
    __init__.py
    config.py              # validated per-mode settings, units, noise/calibration provenance
    observations.py        # 2D/3D partial-track, masks, timestamp & camera adapters
    rigid_kinematics.py    # Exp_SO3, A_R, A_P, inverse, velocity, Jacobian, gauge
    candidates.py          # rigid-flow seeds, static proposal, multiple R/P initializations
    assignments.py         # persistent static/moving/outlier responsibilities
    geometry_fit.py        # robust R/P full-sequence optimization, covariance/conditioning
    visual_losses.py       # fixed-camera 2D/depth/feature/mask evidence
    tcp_contact.py         # known sliding point, surface validity, normal-velocity terms
    wrench_physics.py      # frame/sign conversion, ratios, joint power, friction/passivity
    model_selection.py     # same-split predictive scores and ambiguity decision
    exports.py             # contracts, diagnostics, legacy adapter
    cli.py                 # standalone diagnostics/short optimizations
    README.md              # exact commands and data/provenance contracts

# Add deterministic unit tests without modifying existing tests:
tests/test_joint_discovery_se3.py
tests/test_joint_discovery_twist.py
tests/test_joint_discovery_partial_tracks.py
tests/test_joint_discovery_multiview.py
tests/test_joint_discovery_contact.py
tests/test_joint_discovery_force.py
tests/test_joint_discovery_observability.py
tests/test_joint_discovery_legacy_adapter.py
```

Suggested Python interfaces (adapt to local dataloader but preserve semantics):

```python
@dataclass(frozen=True)
class CameraCalib:
    camera_id: str
    K: np.ndarray             # [3,3] pixels, per camera
    T_WC: np.ndarray          # [4,4] OpenCV c2w, constant
    width: int
    height: int
    calibration_source: str

@dataclass
class JointCandidate:
    kind: Literal['revolute', 'prismatic']
    axis_w: np.ndarray        # [3], unit
    origin_w: np.ndarray | None    # [3], line point only for revolute
    q: np.ndarray             # [T], radians for R or meters for P
    q_valid: np.ndarray       # [T], observed/estimated support
    q_uncertainty: np.ndarray | None
    reference_time_index: int
    canonical_anchors_w: np.ndarray  # [N,3] with separate visibility/validity
    moving_probability: np.ndarray  # [N]
    diagnostics: dict

# Required exact math APIs
A_revolute(axis_w, origin_w, theta_rad) -> np.ndarray | torch.Tensor   # [...,4,4]
A_prismatic(axis_w, displacement_m) -> np.ndarray | torch.Tensor       # [...,4,4]
body_velocity(kind, joint, qdot, point_w) -> np.ndarray | torch.Tensor   # [...,3]
canonical_contact_point(joint, q, tcp_point_w) -> [...,3]
relative_slip(joint, qdot, tcp_point_w, tcp_velocity_w) -> [...,3]

# Baseline vs new mode, no CLI invention: implement and document actual argparse:
fit_joint_hypotheses(observations, camera_calibs, cfg) -> dict[str, JointCandidate]
score_candidates_on_heldout(candidates, observations, heldout, cfg) -> ModelDecision
load_robot_contact_record(scene, frame_map, T_WB, cfg) -> ContactRecord
export_for_existing_freeart(candidate, cam_calibs, observed_frame_map, output_dir) -> Manifest
```

**Config** (name/fields illustrative; implement actual validated parser, do not silently reinterpret existing `tracking` knobs):

```yaml
joint_discovery:
  mode: legacy              # legacy | calibrated_joint_aware
  cameras: [cam_001, cam_002]
  fixed_calibration: true
  hypotheses: [prismatic, revolute]
  keep_static_and_unknown: true
  allow_partial_tracks: true
  enable_visual_2d: true
  enable_depth: true
  enable_appearance: false   # opt in after tested
  enable_coarse_renderer: false
  enable_tcp_contact: false  # initial visual baseline
  enable_force_selection: false
  enable_contact_surface_fit: false
  require_calibrated_wrench: true
  no_oracle_labels: true
  comparison:
    heldout_time_fraction: 0.20
    selection: heldout_predictive
    allow_ambiguous: true
  output_subdir: joint_discovery
```

Numerical noise models, optimizer iterations and thresholds must come from a written calibration/config and be logged; do not silently import MoMa-SG default thresholds. Seed controls, deterministic summation/order and package versions are mandatory.

---

## 11. Correct integration with FreeArtGS cameras, masks, PLY and downstream stages

The current fixed multiview converter constructs virtual moving-part poses from **one chosen camera's** independently estimated motion. New mode must instead use the globally fitted shared candidate `A_H(q_t)`.

For a **world-aligned canonical reference** and fixed physical camera `T_WC_c`:

\[
\boxed{T_{M0,C_c}^{virtual}(t)=A_H(q_t)^{-1}T_{W C_c}.}
\]

For the static part use constant `T_{W C_c}`. These virtual poses describe where the camera appears in *the moving part's reference frame*, not a change to physical camera extrinsics. If the 3DGS model internally uses a distinct local `M_0` frame, explicitly include `T_{M0,W}` and test with synthetic data. Validate `T_virtual` by showing that rendering a moving object with fixed camera equals rendering its canonical object with the exact inverse-transform virtual camera, for both RGB and depth.

**Mandatory adapter checks:**

1. For `cam_001` + `cam_002` every identical world point and candidate joint transform projects consistently using each camera's own `K_c,T_WC_c`.
2. Correct masking uses a global moving-support assignment; camera-specific masks may have zero moving pixels due to occlusion and must not be force-balanced.
3. Every output row includes `camera_name`, `time_index`, `source_stem`, `fl_x,fl_y,cx,cy,w,h`, `mask_path`, `depth_path`, and explicit `pose_kind` (`physical_camera` or `virtual_part_camera`). Do not modify old schemas without an adapter and compatibility test.
4. `transforms_0.json` and `transforms_1.json` must have exact timestamp alignment and no silent dropping; invalid/occluded views are recorded rather than assigned fake observations.
5. Do not synthesize legacy `global_poses_*` named “camera trajectory” and pass them to unrelated ICP paths. If an old stage absolutely requires the files, create a named, documented adapter whose contents are verified against the mathematical conventions, with provenance `estimated_joint_world_frame`.
6. In new mode disable old independent per-camera ICP as a mandatory motion estimator. Run it as an **optional cross-check**, not as a replacement for the fit that was just obtained.
7. Nerfstudio *physical* camera optimizer must remain `off`. Trace actual local model and dataparser to ensure it truly is off in all three training stages, not just set in YAML.
8. Check Gaussian quaternion convention from existing tests before GS integration; do not start a 30k iteration job until a small 5–10 frame smoke rendering shows correct moving and static parts across both cameras.
9. Existing `preprocessing/estimate_joint.py` must not silently re-run its rotational-span classifier and override a validated new candidate. Instead either add a `--joint-candidate` import mode or preserve the original estimator as a legacy diagnostic with explicit user-selectable comparison.
10. Export `joint_type`, axis, revolute hinge line, `q_t`, confidence and joint origin convention in a format that downstream splatfacto-art can consume. Report **prismatic origin N/A**.

---

## 12. Synthetic tests: MUST pass before real-data evaluation

Use analytic generators (independent of optimizer implementations) with controlled camera geometry, known T, object points, binary masks, partial occlusions, depth bias, finite sampling, contact and force.

| ID | Test and expected outcome |
|---|---|
| T01 | `A_R(theta)` equals SE(3) twist exponential `[u;-u×o] theta`, for arbitrary non-origin hinge, and all blocks are rigid. Test `theta=0`, inverse, axis flips, derivative by finite differences. |
| T02 | `A_P(d)` exact translation, `omega=0`, unit axis, `d_ref=0`, correct meter/radian units. |
| T03 | Synthetic revolute body has heterogeneous tangential point velocities with shared `dot theta`; radial and axial residuals zero **instantaneously**, exact finite-motion residual zero even for 60° displacement. |
| T04 | Synthetic prismatic body has same velocity for all moving points, correct perpendicular-to-axis residual and scalar speed agreement. |
| T05 | Recover R hinge axis, hinge-line position and `theta_t` from noisy multi-point tracks; compare to oracle values, using recorded noise and an explicit tolerance motivated by that noise. |
| T06 | Recover P axis and `d_t` from noisy tracks; output no pivot location. |
| T07 | Both models fitted on same short-arc/remote-hinge data: report ambiguity when predictive differences are below uncertainty, not confident false P/R. |
| T08 | Mix 30% moving and 70% static with different spatial extents and outlier speeds; recover moving support without dropping all high-motion tracks or enforcing per-camera quotas. |
| T09 | Tracks visible at times `{3,4,8,9}` but absent at frame zero; canonical anchor recovery and re-observation work, no zero/identity placeholder used as measured evidence. |
| T10 | Two calibrated cameras observe disjoint moving-part points; one shared hinge still fits. One camera sees only base => that camera yields zero moving points, not a fabricated motion segment. |
| T11 | Perturb `cam_000` extrinsic; defaults keep cam_001+cam_002 unchanged. Explicit stress mode detects conflict, reports camera residual and may abstain; no auto-optimizing physical poses. |
| T12 | Same 2D tracks but invalid depth at image edge: retain 2D and uncertainty, do not bilinearly contaminate valid depth from `-inf` pixels. |
| T13 | Canonical linear-TCP/rotating-lid analytic test with sliding radius and nonzero `v_tcp-v_body`; normal-contact constraint zero; rigid-grasp constraint nonzero. |
| T14 | Prismatic sliding-on-face test: TCP path not parallel to slider, object contact can slide along face; correct `n^T s=0`. |
| T15 | Straight TCP + unconstrained canonical plane: both P/R can trivially fit; rank/independent-surface-support checks prevent false selection. |
| T16 | Contact on invisible underside; nearest visible rim/robot point rejected as surface oracle; contact loss disabled with reason. |
| T17 | Force transform: rotating sensor coordinates yield invariant world force; force-on-tool sign reversed to robot-on-object; `F_parallel`, `F_perp`, `rho` computed with proper sign and units; near-zero speed/force not classified. |
| T18 | Frictional passive contact: `D=f_obj·s>=0`; sign flip produces negative `D` and is detected; sticking motion (`|s_T|≈0`) does not produce normalized slip vector/divide by zero. |
| T19 | Joint R/P with same TCP line but different **observed** surface orientations: full force direction and parallel/perpendicular proportions support correct hypothesis under shared μ. If no surface known, ratio-prediction loss must be disabled. |
| T20 | With source RGB-D data and fixed camera, projected canonical object transformed by joint equals static canonical object rendered through derived virtual camera (RGB, metric depth and silhouettes, respecting `K`). |
| T21 | Legacy pipeline mode reproduces existing optimizer outputs within documented numerical tolerance on a fixed synthetic baseline. |
| T22 | Temporal index mapping (original frames, selected RGB-D, ping-pong, per-camera names, TCP `poses_rgbd`) never shifts by one frame. Verify with marker timestamps and metadata. |
| T23 | Gradient checks (torch autograd versus numeric differences) for `A_R`, `A_P`, contact `n·s` and reprojection, excluding truly unobservable configurations. |
| T24 | Hold-out evaluation does not leak from split into initialization, loss-weight tuning, ground-truth labels, or force model parameter choices. |

For every test save exact seed, camera matrices, noise, model initializations, a JSON result and assert conditions rather than merely render a GIF. Do not use GT in production inference.

---

## 13. Real-data study and staged acceptance matrix

**Required episodes (verify actual local directory names and provenance before running):**

- `final_black_box_episode_000000` — problematic revolute lid. Prior diagnostic window: RGB-D indices **70:128 stride 2** (30 frames). Prior cam_001/002 have weak hinge evidence while cam_000 has better view but extrinsic disagreement. Compare both camera policies explicitly and never hide that tradeoff.
- `final_red_fridge_episode_000000` — revolute control. Prior window **90:148 stride 2** (30 frames). Verify contact availability; earlier force detector yielded few confirmed samples, which is not licence to fabricate more.
- One actual prismatic drawer scene with suitable valid TCP/wrench information — identify exact scene using local dataset discovery; do not fabricate a name. `storage` or `white_drawer` are potential scene families but their tooling, calibration and timestamps may differ.

### Baselines and ablations

| Run | Tracks / geometry | Robot use | Purpose |
|---|---|---|---|
| B0 | Current FreeArtGS Phase 1 + original reconstruction | None | Immutable baseline failure and accepted successes |
| B1 | MoMa-SG-reference twist estimator on same extracted points | None | Reproduction/benchmark reference; label adapted interfaces |
| B2 | New static+P/R 3D tracks only | None | Test exact joint field vs unconstrained `T1,T2` |
| B3 | B2 + per-camera 2D reprojection and valid depth | None | Test whether image feedback prevents depth-driven translation degeneracy |
| B4 | B3 + persistent moving support / partial-track recovery | None | Test missing-part discovery and occlusion behavior |
| B5 | B4 + *actual* appearance/render feedback | None | Test whether full visuals improve held-out predictions |
| B6 | B4/B5 + known TCP contact surface, if supported | TCP only | Test geometry contact information separately |
| B7 | B6 + wrench passive/directional evidence where valid | TCP + force | Test force proportions and joint selection |
| B8 | B7 with force shuffled, sign reversed or time shifted | Null/control | Demonstrate genuine physical signal rather than label leakage |
| B9 | B4/B7 with cam_000 explicitly included | Controlled camera stress | Quantify alignment sensitivity vs motion visibility |

### Per-run measurements (mandatory)

1. **Track survival:** observed vs retained 2D/3D tracks, moving/static distribution where oracle labels available *diagnostic only*, rejection by reason and whether recovered support is coherent.
2. **Kinematics:** estimated type, `q_t`, `qdot_t`, axis direction, hinge-line location, temporal smoothness, per-point radial/axial/tangential residuals and prismatic common-speed residual.
3. **Multiview:** each camera's `K/T` validity, projection/depth residual vs time, number of moving points, calibration conflict and partial visibility.
4. **Robot/contact:** frame match rates, raw and compensated force, `Fparallel`, `Fperp`, signed angle, ratios, TCP world trajectory, canonical contact path, validated surface normals and confidence, `n·s`, generalized joint force/torque, `D_H`, omitted frames/reasons.
5. **Model selection:** both candidate NLLs on exactly the same held-out samples, any complexity correction, observability rank/condition, final label or `ambiguous`, and confidence definition.
6. **Actual rendering:** masked RGB residual, metric depth residual, object mask IoU/coverage, held-out PSNR/SSIM where justified; joint animation across all selected frames; clearly label training vs validation renders.
7. **Resource/provenance:** GPU memory, runtime, package SHAs, source scene hashes, selected time intervals, camera subset, optimizer seeds, any code differences.

**Acceptance is not “the training completed.”** The new method must recover *more actual moving support* (not just redistribute labels), produce physically valid and stable joint parameters, improve held-out visual/motion consistency versus B0/B1 and not regress on well-performing prismatic scenes. For unsuccessful/ambiguous sequences, show the first measured failure stage and the explicit observable limitation. Do not set arbitrary numeric angle/position thresholds without relating them to GT and sensor uncertainty; still report absolute errors in degrees/cm and sensitivity to thresholds.

---

## 14. Required artifacts and reports

Write under a fresh path similar to `outputs/joint_discovery_runs/<scene>/<config_hash>/<run_id>/`, outside raw source. Include at least:

```text
manifest.json                     # run config, git SHA/diff, package versions, camera sources,
                                  # exact time mapping, robot/wrench availability, parent baseline
input_audit.json                  # counts, valid cameras, uncertainty, missing robot/calibration
execution_graph.md                # exact local caller->callee->output and branch-dependent paths
tracks_observations.npz          # sparse or index-referenced raw 2D/depth/3D measurements
tracks_fate.parquet              # every track ID/observations, why retained/rejected/promoted
velocity_field_diagnostics.npz   # t, track, tangent/radial/axial/perp predictions/residuals
joint_candidates.json            # BOTH candidates, type, axis, origin if R, uncertainty,
                                  # observability, score and exact coordinate conventions
joint_states.npz                 # q_R[t], q_P[t], qdot, validity, timestamps, uncertainty
part_support/                    # camera/time masks and probabilities + confidence/unknown
camera_projection/               # 2D overlays and reprojection/depth residuals by camera
contact_force_diagnostics.parquet # measured TCP/wrench, force ratios, slip, D_H, valid mask
contact_surface/                 # observed surface patches, normals, source and uncertainty
visual_feedback/                 # fixed-camera RGB/depth/mask render comparisons
model_selection.json            # held-out score, null controls, type/ambiguous decision
integration_checks.json          # camera invariance, pose semantics, quaternion, legacy
report.md                        # evidence-backed causal findings, changes, ablations
```

Keep raw and derived quantities separately, with exact units and frame labels in schema. Require `NaN`/null+validity for missing data, not zero-filled substitute. Every plotted curve must label timestamps (not just reindexed frame numbers). Export a compact failure table with **source file:function and relevant local line numbers** for every confirmed defect.

---

## 15. Execution and delivery order for Codex (do not skip gates)

**Gate 0 — Baseline and provenance (no edits):** Inspect local repo, existing audit work, MoMa-SG reference and actual files; produce execution graph and input availability. Establish the correct camera/world/robot and timestamp conventions. Record baseline against current unit tests.

**Gate 1 — Exact mathematical kernel:** Implement SE(3), R/P motion fields, canonical anchor, Jacobians and slide-contact derivative. Pass T01–T04, T13–T15 and T23 with CPU tests. No tracking/rendering code changes.

**Gate 2 — Track adapter and calibrated projection:** Preserve raw partial 2D tracks and valid 3D lifting, verify multi-camera same-world geometry and camera intrinsics. Pass T09–T12, T22. Run a real **read-only** track fate analysis on black box.

**Gate 3 — New visual-only Phase-1 estimator:** Fit static/P/R on exactly the same observations with uncertainty, persistent support and model selection; pass T05–T10. Show both candidate trajectories and actual held-out 2D/depth residuals on black box and prismatic control. No force yet.

**Gate 4 — Visual feedback:** Add image-space appearance feedback and optional existing GS renderer with fixed calibrated cameras; pass T20. Show first-order visual changes per ablation before full training.

**Gate 5 — TCP sliding contact:** Load real TCP archive and verified contact flags, ensure common frame/time, add only independently supported surface/normal constraints; pass T13–T16, T22. Compare with and without known contact data on exactly same real visual candidates.

**Gate 6 — Wrench decomposition and physical selection:** Verify real wrench frame/sign/compensation, compute signed force + both proportions, shared-μ model if supported, passivity and null controls; pass T17–T19. Report if force samples are insufficient; don't force a benefit.

**Gate 7 — Downstream FreeArtGS adapter and regression:** Construct exact virtual camera trajectories if needed, preserve static physical cameras, derive canonical part PLY seeds, bypass old independent motion ICP safely, run short GS smoke tests and legacy regression. Only then run a controlled longer reconstruction if compute availability supports it.

**Gate 8 — Final evidence report:** Provide complete ablations B0–B9 or explicitly mark not run/unavailable, tests, source-linked failures, real-world limitations, optimization gains, resource cost and safe defaults. Do not merge incomplete/untested code to default path and do not declare final paper-quality performance on training views.

### Suggested local commands (adapt to actual scripts; never invent a success result)

```bash
cd ~/FreeArtGS
pwd
git rev-parse HEAD
git status --short
git submodule status --recursive
rg -n 'class Trans|def get_visible_trajectories|def get_4d_traj_maps|def prepare_multiview|def estimate_revolute_from_poses|camera_optimizer' estimotion_2part tracking preprocessing pipelines reconstruction
python -m pytest -q tests/test_icp_frame_alignment.py tests/test_joint_axis_evaluation.py tests/test_real_robot_pose_export.py
# After the new module and tests genuinely exist:
python -m pytest -q tests/test_joint_discovery_*.py
python -m estimotion_2part.joint_discovery.cli --help
# Use real scene selection from local --help and metadata; do not guess dataset roots.
```

**Final response from Codex must include:** `git diff --stat`, actual changed files/commit state, tests run/pass/fail/skipped, source/timestamp/camera provenance, realistic failure analysis, and a short explanation of **which measured information first recovers valid revolute/prismatic motion** and **whether adding TCP and then force genuinely improves held-out selection**. If no real-data evidence is available, say so explicitly.

---

## 16. Scientific interpretation and non-negotiable falsification checks

1. **Do not claim novelty for joint twist estimation alone.** MoMa-SG already estimates revolute/prismatic motion from point trajectories and its regularizer addresses noisy drift. The contribution here, if experiments support it, is joint/static support discovery in calibrated multiview FreeArtGS with exact image feedback and optional real TCP sliding-contact/wrench constraints.
2. **Do not claim every TCP slip vector lies along the TCP path.** Slip is tangent to the *object contact surface*, not necessarily parallel to world TCP velocity. The material point velocities obey rigid joint kinematics; the TCP is not a tracked material point.
3. **Do not claim force proportions uniquely reveal type.** Without a known/observably supported contact surface, friction/contact mode, and correct wrench frame/sign, a force proportion trace can be equally compatible with P and R. Use full vector/direction and physical constraints and **abstain** where needed.
4. **Do not infer a hidden normal from a one-dimensional TCP curve alone.** A fitted plane can be mathematically perfect but physically unrelated to the object.
5. **Do not compare raw least-squares costs across differently sized/masked datasets.** Both hypotheses must see the same eligible measurements, and missing geometry must not silently become a model advantage.
6. **Do not interpret a smaller track residual as a better articulated scene if physical cameras, moving masks or depth scaling are wrong.** Verify fixed-camera ground truth, visibility, quaternion covariance rotations and held-out rendering.
7. **Do not allow a change in `cam_000` inclusion to be hidden in a changed default.** The default is cam_001+cam_002; cam_000 is deliberately tested as observability-versus-calibration stress.
8. **Do not hard-code static/moving masks from oracle geometry in production.** Oracle masks/axes remain diagnostic-only.
9. **Do not silently include robot pixels as moving object.** A point tool does not define its 3D mesh; uncertain wrist/tool pixels remain excluded/unknown rather than assumed to be part of the contacted object.
10. **Do not optimize or fabricate an unconstrained contact geometry separately for each candidate.** The force and TCP contributions must remain independent physical evidence, not free parameters that make either model fit.

**Desired final outcome:** A reproducible, optional FreeArtGS Phase-1 estimator which discovers static and articulated moving support, fits both prismatic and revolute exact motion fields jointly across `cam_001`/`cam_002`, uses valid partial tracks and actual images as feedback, and incorporates the accurately measured sliding TCP trajectory and calibrated force-direction proportions *only when physically identifiable*. A negative or ambiguous result is acceptable if the evidence is correct and thoroughly diagnosed.
