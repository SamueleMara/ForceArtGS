# ForceArtGS / FreeArtGS — Joint-aware Phase 1: next implementation sprint

**Date:** 2026-11-17  
**Target repository:** `https://github.com/SamueleMara/ForceArtGS`  
**Audited baseline:** `main` at `468071f2dbd9c5f22f3268c0733d516736681b1d` (verify local branch and diff before implementation)  
**Environment:** existing `freeartgs` environment; preserve the existing FreeArtGS renderer and training pipeline  
**Audience:** Codex implementing, validating, and reporting improvements; this is an engineering contract, **not evidence that the described features already work**.

## 0. Executive mandate and problem definition

Implement and validate the **next stages of joint-aware Phase 1** without replacing the already implemented exact kinematics, sparse partial tracks, calibrated camera handling, static/moving/outlier mixture or the legacy fallback. A correct revolute/prismatic *label* is not a successful reconstruction if the axis, angle, moving support or RGB-D renders are wrong.

**Highest-priority outstanding failure:** `final_black_box_episode_000000` with default `cam_001+cam_002`:

- RGB-D fit (previous B4): `revolute` label but only **~2.8°**, with a **~1.21 m** distant hinge; tracker-free reference suggests **~15–21°** over the clip (reference itself uncertain; **diagnostic only**).
- RGB-only fit: **~8.0°** and more plausible hinge, but **~46%** of inferred moving tracks below the diagnostic lid-height split (base leakage).
- `cam_000` alone: **~15.0°**, hinge radius **~0.12 m**, but camera cross-view agreement is worse than `cam_001+cam_002`.
- Three cameras together: **~6.9°**, with a large axis discrepancy. Do **not** optimize physical camera extrinsics to hide this.
- Reference-seeded fit returns to the wrong RGB-D optimum: **a measurement/assignment/observability problem**, not solely an optimizer initialization problem.

**Other real observations:** red fridge from `cam_001+002` yields ~18.4–18.9° revolute motion; robot dog ~9.52 cm prismatic; orange suitcase is ambiguous with a distant-hinge degeneracy; white drawer is ~16.2 cm prismatic candidate but a remote revolute model wins numerically and is now correctly marked ambiguous/Occam-prismatic. Do not assume these estimates have verified GT. These are reported measurements, not acceptance criteria to tune toward.

**Main implementation gap:** `estimotion_2part/joint_discovery/cli.py::run_discovery` calls `fit_joint_hypotheses` and `score_candidates_on_heldout` **before** `contact_force_stage`. The output explicitly states `robot_terms_used_in_selection=False`. The `vision` and `tcp_force` results therefore cannot differ by design. Moreover, full Gaussian renderer feedback, held-out camera evaluation and downstream trained reconstruction have not been exercised. Fix those gaps **only after the evidence issue is measured**.

### Non-negotiable invariants

1. Physical cameras are **stationary and calibrated**: retain per-camera `K_c`, `T_WC_c`, image size, OpenCV optical coordinate convention and source-time mapping. No camera pose optimization. Default cameras `cam_001 cam_002`; use `cam_000` as an explicit stress/diagnostic experiment until calibrated agreement is independently established.
2. One shared **1-DoF object joint** in the world frame, with the same `q(t)` across cameras. Optimize exactly valid `SE(3)` transformations, not per-frame unconstrained affine matrices.
3. Static, moving and unknown/outlier membership must be persistent **per material track**; allow partial trajectories, changing visible surfaces and views containing no moving part. Never blend static and moving *point coordinates*.
4. The saved calibrated TCP is the **3D contact location** when physical contact is independently established. It is **not** a material track glued to the moving object: the TCP may slide across the surface. Its path may be straight for both P and R.
5. Never use an unobserved contact plane as a known surface; never infer a unique plane normal from a 1-D TCP contact curve; missing contact normals/forces must result in `UNAVAILABLE` or `ABSTAIN`, not a fabricated positive score.
6. Measured wrench is initially raw and has **unverified tare/gravity compensation and sensor-frame alignment**. Force-based joint selection must remain gated until these conditions are established, with uncertainty and provenance. Do not confuse an arbitrary force threshold with an independent contact annotation.
7. Keep original renderer and training code working; maintain opt-in flags with **legacy as default**. No GT/reference geometry in optimization, thresholds, validation-split tuning or support selection. GT/reference may be used only in a firewalled, post-hoc report.
8. Fail closed on missing metadata, stale cached observations, uncertain calibration, wrong timestamp joins, unknown wrench sign or missing renderer features. Do not silently disable a requested stage or report an unrun gate as PASS.
9. No destructive changes to historical runs, external datasets, old checkpoints or the separate `SamueleMara/FreeArtGS` remote. Do not force-push or clean unrelated branches.

## 1. Repo inventory and mandatory baseline audit

Codex MUST inspect before modifying anything (some code may have changed locally):

```text
estimotion_2part/joint_discovery/
  cli.py                 run_discovery, contact_force_stage, extract_alltracker
  config.py              typed flags, noise/fit/compare/contact configurations
  observations.py        CameraCalib, FrameMap, sparse tracks, valid depth sampling, tool exclusion
  visual_losses.py       fixed-camera reprojection, depth residual, point renderer proxy, patch NCC
  candidates.py          RANSAC rigid proposals; rigid_flow_omega
  geometry_fit.py        Problem, EM, exact-joint optimization, anchor GN, Schur covariance
  assignments.py         persistent static/moving/outlier mixture
  model_selection.py     held-out time split and paired-track score, remote-hinge flag
  rigid_kinematics.py    A_R(theta), A_P(d), contact Jacobian
  tcp_contact.py         source-frame joins, canonical TCP path, observed surface fit, slip
  wrench_physics.py      force sign/rotation, F_parallel/F_perp, Q_H, passivity, Coulomb
  exports.py             per-camera virtual camera output and FreeArtGS adapter
  README.md
pipelines/joint_discovery_scene.sh
pipelines/joint_discovery_real.sh
configs/joint_discovery.real_tcp2cm.yaml
configs/joint_discovery.example.yaml
docs/joint_discovery_phase1.md
docs/multiview_root_cause_black_box.md
docs/rigid_motion_corrections.md         # inspect if present
tracking/tracker_toolkit.py
preprocessing/prepare_multiview.py
preprocessing/estimate_joint.py
pipelines/freeartgs.sh
pipelines/splatfacto_art.sh
```

For each touched module, record exact commit, local git state, existing behavior and tests. Use `git diff --check`, run the existing 41 joint-discovery tests and the legacy smoke tests before modifications. Do not reuse a prior run when config, calibration, code, track extractor, frame subset, tool mask, selected camera set or renderer changes; make run manifests fingerprint all those dependencies. Preserve raw `observations_*.npz` and baseline `model_selection.json` in immutable A/B output directories.

**Known code properties to verify:**

- `visual_losses.whitened_residuals` uses a normalized 2-D reprojection residual plus axial depth; it does not directly optimize full RGB appearance or Gaussian renders.
- `geometry_fit.fit_geometry` uses a per-observation depth residual weighted by `depth_sigma` and point responsibilities; its temporal prior uses `q_t` including interpolated samples.
- `model_selection.make_heldout_split` holds out evenly spaced *interior times*; `score_candidates_on_heldout` computes a paired standard error over *tracks*, which are spatially/temporally correlated. Understand interpolation at holdout `q_t`, anchor leakage, and effective sample size before using z as a calibrated confidence.
- `tcp_contact.load_robot_contact_record` uses an identity `T_WB` only when cameras declare base_link, but the wrench rotation defaults to `R_WS=R_WEE`; equality of sensor axes and EE axes is an **assumption**.
- `wrench_physics.WrenchConvention.calibrated` currently checks `gravity_compensated and tare_applied`, not synchronization, sensor axis/frame or force origin accuracy. Introduce stricter readiness gates without breaking legacy behavior.
- `visual_losses.render_point_proxy` is **not a Gaussian splatting renderer**.

Reference upstream MoMa-SG, without silently vendoring or claiming equivalence:

- https://github.com/robot-learning-freiburg/MoMa-SG
- `moma_sg/articulation/point_estimator.py` and `moma_sg/articulation/filtering.py` provide twist-fitting and twist-consistent track filtering concepts.
- Record a MoMa-SG reference benchmark only if actually run. Correct any identified upstream revolute-transform convention issues in the comparison; do not transplant a known problem.

## 2. Exact mathematical model: shared joint and material tracks

**Frame convention:** `T_AB` maps coordinates from B to A, world W equals declared `base_link` only when source metadata proves that. Measured camera poses `T_WC` fixed. Metres, seconds, radians, camera-Z depth in metres.

Let `X_i` be the *canonical* material point of track `i`, observed from camera `c` at valid times `t` with tracker pixel `u_ict`, optional camera-Z measurement `z_ict`, RGB/feature patch and masks. Track identifiers are scoped to `(camera, query_window, local_id)`; cross-camera matches are not assumed.

For revolute hypothesis `H_R=(u_R,o_R,{theta_t})`, `||u_R||=1`:

```
R_t = Exp_SO3([u_R]_x theta_t)
A_R(t) = [ R_t, (I-R_t)o_R ; 0, 1 ]
X_it = o_R + R_t (X_i - o_R)
v_it = theta_dot_t u_R x (X_it - o_R)
```

For prismatic `H_P=(u_P,{d_t})`:

```
A_P(t) = [ I, d_t u_P ; 0, 1 ]
X_it = X_i + d_t u_P
v_it = d_dot_t u_P
```

Gauge: `q(t_ref)=0`; axis sign and `q` sign must flip together; revolute hinge point along `u` is gauge (choose closest-to-support-centroid representative); **prismatic has no hinge origin**. Use exact finite transforms for optimization and held-out evaluation, not an instantaneous velocity approximation of a large angle.

**Motion-field diagnostic, not a separate freely fitted deformation:**

```
P: v_i - v_j = 0
R: v_i - v_j = theta_dot u x (X_i(t)-X_j(t))
```

For R, write `r_i=(I-uu^T)(X_i(t)-o)`, `e_r=r_i/||r_i||`, `e_theta=u x e_r`; ideal material velocities have zero radial/axial components and common `v_theta_i/||r_i||=theta_dot`. For P, all velocities are identical and parallel to `u`. Calculate finite-displacement residuals from actual `A_H(t+1)A_H(t)^-1` where sampled frames have non-negligible angular change. Material tracks DO NOT SLIP with respect to the rigid part; the **robot TCP** does.

## 3. Stage A — Isolate the black-box RGB-D contradiction (mandatory before changing fitting weights)

**Goal:** identify *which observations and why* pull the default-camera estimate from the plausible revolute solution toward 2.8° remote-hinge/translation. Merely increasing depth sigma or turning off depth is not a solution.

### A1. Counterfactual likelihood audit

Add `estimotion_2part/joint_discovery/depth_conflict_audit.py` and a CLI subcommand (`audit-depth`) that **does not refit the model** for the first pass. Read the exact same saved 2D/depth tracks and candidate solutions from `vision` runs. Evaluate the following frozen hypotheses against identical original observations:

- P fit from `cam_001+002`, with and without depth.
- R fit from `cam_001+002`, with and without depth.
- R fit from `cam_000` (transformed into the common world frame), evaluated separately on cam_001 and cam_002; this is a **diagnostic**, not ground truth or a training prior.
- Tracker-free reference axis and angles **post-hoc only**; never insert these into the selector or support fitting.
- Static null and a candidate fixed at the last valid 2D-only R fit.

For each candidate and **each camera × timestamp × track × pixel-region**, export:

```
r_uv_px[2], r_z_m, sigma_z_m, depth_edge, incidence_cos, depth_valid,
mask_valid, track_id, moving_posterior, original_pixel, predicted_pixel,
measured_z, predicted_z, estimated_layer_or_patch_id, nearest_occluder,
valid_measured_depth, residual_direction, surface/depth provenance.
```

Report weighted and *unweighted* depth cost separately, medians, quantiles, signed biases, trimmed error, missing data and contribution to the **R-vs-P / plausible-R-vs-fitted-R** objective gap. Heatmaps on raw RGB and depth frames must show which regions dominate. Preserve *every* rejected/ignored observation with a reason code; exclude unknown robot points from the **object** evidence, not from provenance.

### A2. Confirm systematic versus stochastic errors

For matched static and moving patches separately, quantify: per-patch temporal depth median bias, signed residual autocorrelation, cross-camera disagreement, edge/occlusion transitions, same-surface depth interpolation, physical distance to depth discontinuity, incidence angle, and tracker 2D drift. Check whether a depth sample belongs to the same physical surface as its track across time. Compare with the available `docs/multiview_root_cause_black_box.md` diagnostics (cam_002 tracking error ~4.8 px in early window; 7–8 mm median / 24–62 mm p90 lid-depth error), but recompute on **the current new-path observations**.

Distinguish:

1. Valid surface, unbiased random depth noise.
2. Bias on a valid surface (camera/time/view/material dependent).
3. Wrong-surface depth association / occlusion.
4. Contamination by base or robot pixels.
5. Camera-relative geometry mismatch (extrinsic or systematic range bias; not automatically fixable from object motion).
6. Incorrect 2D point tracking, even with valid depth.

Audit whether `sample_depth_valid` choosing the highest-weight valid depth corner can still mix surfaces or select the wrong one at silhouette transitions. Never promote a depth value merely because it is finite. Use RGB/depth and temporal visibility checks to assign confidence; reject *unsupported* depth while retaining 2D track observations.

### A3. Controlled ablation matrix

Perform controlled fits from **identical track tables** and identical held-out folds:

- V2D: 2D only.
- V2D+rawD: existing RGB-D, unchanged baseline.
- V2D+edge-aware-D: omit demonstrably mixed/wrong-surface depth with recorded reason; do not blanket-erode masks.
- V2D+calibrated-noise-D: empirically calibrated per camera/surface noise and bias, with parameters trained only on training frames, not tuned on heldout/GT.
- V2D+independent RGB patch consistency where visible, and joint-consistent depth only as supported.
- Per-camera fits (001, 002, 000 diagnostic) and shared 001+002 fit. `cam_000` must NOT become production default from apparent single-view success.

For each, compare objective loss breakdown, recovered `q(t)`, axis/origin, moving/static support (including below-lid diagnostic contamination), uncertainty and held-out residuals. **Gate A:** submit a source-localized explanation with a quantitative effect size for the dominant corruption. If no correction addresses it, explicitly say the scene remains unresolved; do not claim recovery based on label alone.

## 4. Stage B — Improve observation likelihood and moving-support recovery

### B1. Candidate-independent calibrated observations

Build evidence-aware `depth_support`/`rgb_support` fields in `observations.py`, with provenance codes:

```
valid_surface, depth_edge_ambiguous, occluded, invalid_depth,
wrong_surface, robot_contamination, inconsistent_multiview,
track_rgb_inconsistent, unverified, etc.
```

**Validation rules:** no zero-filled missing depth; 2D survives missing depth; no moving-point promotion from a background depth sample; nearest visible **object** surface must own the pixel before depth contributes; do not use a measured background surface to update a hidden foreground point. Do not replace hard masks with probabilistic background leakage. For near-edge 2D tracks, track color/patch agreement, local plane support and temporal visibility are evidence; uncertainty must be recorded. Do not delete all minority moving trajectories via global displacement thresholds.

### B2. A likelihood that accounts for systematic depth uncertainty

Existing nominal residual:

```
r_uv = (pi(K_c T_CW_c A_H(q_t) X_i)-u_ict)/sigma_uv
r_z  = (z_hat_ict-z_ict)/sigma_depth
```

Preserve that as the baseline. Introduce a **separately ablated** depth model, for independently supported same-surface pixels, e.g.

```
z_ict = z_hat_ict + b_{c,p,t} + epsilon_ict
```

where `b` is a *small, regularized, explicitly observable* shared bias by camera / stable surface patch / temporal block, **not** a free parameter for each track or frame. Determine identifiability on the static core and synthetic truth, set a zero-centered prior or anchored reference, and penalize spatial/temporal variation. Never let `b` absorb true hinge motion or camera miscalibration. Another valid path is to use an empirically measured **correlated** residual covariance or a validated robust mixture of valid-depth and wrong-surface likelihood. Codex must implement only what tests distinguish from overfitting. Selection must use a fair predictive marginal/penalized objective; do not add nuisance parameters without a complexity or held-out cost.

### B3. Persistent static/moving/outlier support with genuine multiview coupling

Keep `assignments.py`'s per-track mixture and one shared `A_H(q_t)`. Improve assignments only through physically justified evidence:

- 2D/appearance coherence of trajectories (especially when depth is invalid).
- Shared finite-motion residual across cameras in the same world frame, with each camera's independent tracks.
- Spatial consistency of *observed object surface* patches in canonical space, including time-varying occlusions.
- Static null must remain competitive; a view can legitimately have zero moving observations.
- Do not make static/moving priors dynamically change just to absorb a badly estimated hinge; log support changes between EM iterations.

A region that is unsupported must remain `unknown`, not automatically moving. Preserve a clear split between **confidence of tracked material motion** and later **unobserved Gaussian recruitment** (out of scope here). Validate base leakage independently of fitted joint label; do not fit thresholds to the black-box height reference.

### B4. Global candidate refinement

Fit S, R and P in common 3D with exact `A_H`, fixed cameras and timestamp-aware joint states, regularized only by justified priors. Avoid affine transforms, frame-wise axis drift, per-camera independent joint states, force-derived hard axis labels or assumed stick contact. Pairwise relative-velocity diagnostics may initialize/score rigid support, but exact finite-motion likelihood remains the optimization authority.

**Gate B:** on synthetic calibrated 2-camera data with realistic edge corruption, the corrected likelihood must recover known 1-DoF motion; on the real black box it must improve supported motion **without increasing static-base contamination or degrading held-out cameras**. If evidence remains insufficient, report ambiguous/underconstrained instead of returning a confidently incorrect axis.

## 5. Stage C — Repair joint-type selection, uncertainty and remote-hinge handling

### C1. Audit the existing temporal holdout

Current `model_selection.py` holds out uniformly spaced interior frames and reports `z = sum(track delta NLL)/(std(track delta NLL)*sqrt(n_tracks))`. Tracks from the same surface/camera are *not independent*. Furthermore `q_t` at heldout frames is interpolated/smoothed by the joint trajectory, so explicitly inspect temporal-prior influence and whether predicted q is determined entirely by training data. Implement tests with simulated correlated tracks; provide effective sample size and/or block bootstrap over **spatiotemporal surface groups**, not a naive pixel-track z-score. Track-anchor and noise estimation must use training data only; no heldout surface-normal/point reconstructions should leak into candidate fitting.

### C2. Held-out camera and blocked-time evaluation

Implement independent **held-out camera** verification (fit 001, score 002; fit 002, score 001; joint-fit 001+002, score held-out times in both) while keeping fixed extrinsics. If fitting in one camera makes an axis unobservable, report that explicitly. Include contiguous held-out time blocks and forward-extrapolation blocks with their coverage and `q(t)` predictive rule. For each fold log whether q at test timestamps is interpolated, physically extrapolated, or actually constrained by another camera; do not present interpolated q as unconditional predictive motion. Define a fair candidate-independent test observation set and per-camera/part coverage. Bootstrap over tracks grouped by correlated surface and time, and disclose sensitivity to grouping.

### C3. Physical degeneracy and honest model decision

Retain the current `remote_hinge_approximates_translation` flag but **do not rely on `hinge_radius/support_extent > 5` as an oracle cutoff**. Measure:

- `theta_span` vs its noise-derived uncertainty and the observed spatial extent.
- Relative-motion Jacobian singular values / Fisher information for axis orientation and hinge line, after marginalizing canonical anchors and q.
- Difference between R and best P predicted material-point motion on observed support, in calibrated pixel/depth units.
- Stability of axis, hinge line and q under camera/time-block subsampling.
- Effect of realistic camera calibration perturbations as *diagnostics only*; keep calibration fixed in optimization.

A far-hinge small-angle model approaches translation (`theta -> 0`, `r -> infinity`, `r theta` finite); a marginally better NLL does not identify a physical revolute hinge. Selection must permit `ambiguous/unobservable`; also carry explicit `type_confidence`, `geometry_identifiable`, `q_span_uncertainty`, and whether data distinguish a finite hinge from a translation. Never report a numerically determined hinge origin if the Fisher information is singular or the axis line is outside supported physical geometry.

**Gate C:** a synthetic drawer with small joint play / 1–2° camera mismatch must not receive a confident revolute classification. A true small-angle revolute with a physically near hinge must not be rejected solely by a fixed ratio. Report sensitivity to threshold/regularizer as a *robustness test*, not to choose the answer on the real scenes.

## 6. Stage D — Reliable TCP contact intervals, frames and calibration

### D1. TCP is known; contact interval is not identical to wrench threshold

Robot TCP is `p_tcp^W(t) = T_WB T_BEE(t) p_tcp^EE`. Use the saved calibrated tip point, exactly matched to `source_frame_index`, and preserve the full-rate robot timestamps; no invented interpolation/synchronization claim. Current code uses a heuristic `||F(t)-tare|| > 1 N` (three clip-start frames); black box and red fridge yielded zero marked contacts in those windows even though the robot appears to be manipulating the object. Earlier 0.3 N threshold gave 18 black-box samples; this does **not** make them independently verified contacts.

Create `samuele/audit_joint_contact_windows.py` or equivalent. Export aligned full-episode traces of TCP xyz/velocity, raw force components, norm, signed projection along TCP motion, raw torque, RGB-D frame/source indices, camera visibility and manually reviewable contact timelines. Explicitly distinguish `contact_confirmed`, `contact_probable`, `contact_unknown`, `no_contact` and provenance (`operator label`, independent video annotation, sensor event, heuristic); only **independently confirmed** intervals activate hard surface-contact constraints. When annotations are unavailable, evaluate heuristics and show precision/false activations, never mark heuristics as GT. Black-box and fridge episode full-force peaks are outside/weak in the selected windows; verify actual manipulated clip location, time alignment and baseline before concluding the sensor data are unusable.

### D2. Wrench-frame and gravity/tare readiness

Current `R_WS = R_WEE` is justified only if measured sensor axes are proven aligned with EE. Implement a versioned `WrenchCalibration` contract containing:

```
T_EE_S (at least R_EE_S; required to rotate forces),
p_S^EE if torque transport is used,
recorded wrench quantity/sign, tare offset in sensor coordinates,
static EE/tool gravitational wrench model (if needed),
calibration method, date, fit residual/uncertainty, dataset/tool ID,
clock mapping and uncertainty.
```

Transform measured **contact-only** force to world with a validated convention:

```
F_object^W = sign * R_WEE R_EE_S [F_recorded^S - F_bias^S(q_robot,t)]
```

Here `F_bias` includes only experimentally justified zero/tare and gravity wrench; never subtract the entire force when actual contact occurs. If the original stream was already gravity-compensated/zeroed, establish this from measurement conventions; avoid double subtraction. If EE changes orientation under gravity, verify a static multi-pose sequence; quantify residual gravity coupling. Torque conversion uses correct sensor origin and moment arm and must be disabled if unknown. Preserve raw and compensated channels separately.

**Hard force-selection gate:** verified force frame rotation + sign; credible compensation and uncertainty; actual time alignment; valid contact interval; enough above-noise force and nonzero TCP velocity; physically supported contact normal if applying a direction model. If *any* gate fails, score `ABSTAIN` with reason, leaving vision selection intact.

### D3. Explicit synchronization verification

Record `t_rgbd`, `t_robot`, source frame index and any measured offset with uncertainty. The exact `source_frame_index` join alone does **not** establish hardware synchronization. If clock offset is estimated, use an independently observed event and training/calibration data—not the heldout joint score—then report uncertainty and sensitivity. Never search per hypothesis for the time offset that makes that candidate win.

**Gate D:** calibrations pass static zero-force, EE-rotation and reference-force direction tests; synthetic wrench transforms and sign / moment lever tests; contact windows are independently reviewed or remain labelled heuristic.

## 7. Stage E — Physically valid sliding TCP and force-guided comparison

### E1. One known contact location, no rigid attachment

During independently supported maintained contact, `p_c(t) = p_tcp(t)` is known. It is *not* a fixed material anchor. At each time the candidate predicts object material velocity at the instantaneous contact point:

```
J_P(p) = u_P                           [m/m]
J_R(p) = u_R x (p - o_R)             [m/rad]
v_M,H(p,t) = J_H(p) qdot_H(t)
slip_H = v_tcp - v_M,H(p_tcp,t)
```

For candidate `H`, `x0_H(t)=A_H(q_t)^(-1) p_tcp(t)` is the canonical **contact path on the object**, which changes as the tip slides. If a reliable object-fixed local surface `F0(x)=0` (or independently supported planar patch `(a0,n0)`) is available, impose:

```
F0(x0_H(t)) = 0
n_W(t)^T slip_H(t) = 0       [maintained contact, no normal separation]
```

For planar patch: `n_W=R_H(t)n0`; P has `R_H=I`. Only the **normal** contact velocity is constrained, not full 3D TCP/object velocity equality. If a contact edge or finite thickness invalidates a global plane, use a validated local surface/edge model with full provenance; do not extrapolate the visible top face to an unseen underside.

The position contact residual and the differentiated normal-velocity residual are correlated measurements from the same trajectory/surface. Do not count both as statistically independent evidence; use one primary objective and the other as a diagnostic unless a joint covariance has been validated. Near-zero projected lever-arm/surface-normal sensitivity must produce an observability warning, not `qdot=∞` or a fabricated measurement.

### E2. Surface support and hidden-contact abstention

Reuse `tcp_contact.py::surface_from_observed_patch`, but explicitly track support from **original observed moving-surface geometry**, not just optimizable canonical anchors that could drift to satisfy the contact. For each hypothesis validate the candidate normal on the same physical time/region and use **the same eligible contact frames** for P-vs-R comparison. A nearest visible point is not necessarily the contacting side of the lid. Enforce physically observed support, signed face distance, spatial footprint, surface visibility, near-tip robot contamination and uncertainty. If underside/edge is hidden, set `CONTACT_NORMAL_UNAVAILABLE` and do not fit a free plane to the 1-D canonical contact path. A weakly constrained geometric prior from robot force can be reported separately but cannot magically make a hidden plane observed.

### E3. Force decomposition: retain both proportions and signs

For compensated, frame-correct `f = force robot→object in world`, and `t=v_tcp/||v_tcp||`:

```
F_parallel = f·t
F_perp_vec = f - F_parallel t
F_perp = ||F_perp_vec||
rho_parallel = |F_parallel| / ||f||
rho_perp = F_perp / ||f||
cos_phi = F_parallel / ||f||
```

`rho_parallel²+rho_perp²=1`; **do not treat two proportions as independent evidence**. Keep `cos_phi` sign and the full unit force vector, because the absolute ratios lose whether force drives or resists the tool. The ratios alone do not determine P or R: a drawer pushed along its front-face normal tends to high parallel fraction; a point sliding against a rotating lid often changes the fractions with angle, but those are geometry- and friction-dependent examples, not universal joint signatures.

For a supported outward normal `n_H(t)` and relative slip `s_H`, compute tangent slip `s_T = (I-nn^T)s_H`. Under one passive sliding Coulomb point contact, with shared bounded friction `mu` per physical contact/material episode:

```
n_in = -n_out
f_hat_pred,H = normalize(n_in + mu * s_T / ||s_T||)
```

Only for `||s_T|| > eps`, confirmed contact and supported normal. Do not use this formula for sticking (tangent reaction then lies in friction cone, not necessarily at its boundary). Fit `mu` on training contact intervals **jointly or with a common prior**, not separately with arbitrary per-frame freedom that makes both hypotheses fit; evaluate force-vector angular likelihood on heldout contact intervals. Produce both scalar force fractions and full-vector predicted/observed angles, uncertainty and a calibrated likelihood, using the same contact frames across P and R.

### E4. Generalized effort and passive power

```
Q_P = f · u_P                          [N]
Q_R = u_R · ((p_tcp-o_R) x f)         [N m]
P_joint,H = Q_H qdot_H               [W]
P_tcp = f · v_tcp                    [W]
D_H = P_tcp - P_joint,H = f · slip_H [W]
```

For an ideal single maintained passive contact with correct force sign, `D_H >= 0` up to uncertainty. Test for violations, but **passing is necessary, not sufficient** for joint identification. Do not compare `Q_R` and `Q_P` magnitudes directly (different units), and do not equate nonzero Q with actual acceleration without inertia/friction/actuation models. Check torque-arm consistency if torque origin is reliably calibrated; otherwise force-only.

### E5. Add force/TCP to *selection* only after validated evidence

Refactor `run_discovery` so that training-stage physical terms can be included in candidate refinement and model comparison only under verified gates; otherwise no change to baseline. Every candidate uses the same raw tracks, train/test time-camera splits and *intersection* of supported contact observations. A defensible joint objective is:

```
L_H = L_visual,H
    + lambda_contact * L_contact,H
    + lambda_force * L_force_direction,H
    + lambda_passivity * L_one_sided_passivity,H
    + calibrated regularization
```

**Important:** These terms must have units/noise models and known correlation structure. Learn global weights from prior sensor noise experiments/synthetic calibration, not to force correct labels on the test scenes; prefer negative log-likelihood under declared errors. Contact force must not be used to tune camera extrinsics. All latent quantities (mu, patch normal where independently supported, q, hinge) must be trained only on train folds. The heldout score must evaluate predictive force/contact evidence **without refitting on the test split**, and must report independent vision-only and physical contributions. If contact evidence is missing for one or both hypotheses, run vision-only and report reason; do **not** let the availability of an observed patch itself reward one candidate by dropping hard cases. Add shuffled-time, sign-reversed, gravity-only and no-contact null controls.

**Critical ablation:** S2 vision-only vs S3 vision+validated TCP vs S4 vision+TCP+calibrated force, **same cameras/tracks and eligible frames**. If S3/S4 yield identical candidate parameters/decision by design, the integration has not happened; do not call this success.

**Gate E:** synthetic sliding revolute, sliding prismatic, stick/slip transitions and pure robot-axis motion pass with calibrated force, while sign inversion and time shifts fail/abstain. On real scenes without adequate contact evidence, the correct result is `ABSTAIN`, not a forced improvement.

## 8. Stage F — Actual Gaussian renderer feedback (not point proxy)

Integrate the **existing FreeArtGS Gaussian rasterizer and bindings**, not a new unrelated renderer. Initially use frozen existing canonical Gaussians/part reconstructions (if available and compatible with the current quaternion/scale convention) to score the two joint hypotheses over all calibrated views/times. If canonical models are unavailable, report missing evidence, or train identical static canonical models once under a predeclared, leakage-safe protocol; do not compare a better-trained asset for one hypothesis.

Render for each `H,t,c` using fixed physical `T_WC_c` and `A_H(q_t)` on moving Gaussians, static on base Gaussians. Existing virtual-camera representation `T_CW A_H(q_t)` or its equivalent `A_H(q_t)^(-1)T_WC` must be numerically checked against direct world-point/object transforms, including Gaussian covariances/orientations. Avoid the previously identified GSplat `XYZW` vs `WXYZ` quaternion boundary mismatch. Separate RGB, alpha and metric depth; validate metric depth semantics, occlusion and gradients against finite differences. Do not rasterize the robot as part of the articulated object; treat robot occluders as excluded/unknown measured pixels when tool geometry is not modeled.

Required per-camera losses (same heldout pixels, same static/moving support assumptions):

```
L_RGB   = robust photometric/feature residual on genuinely visible corresponding surfaces
L_depth = robust metric depth residual with foreground/object ownership and occlusion check
L_alpha = calibrated silhouette/object mask residual
```

Every pixel needs a provenance (object-visible, object-occluded by robot, background, missing depth, ambiguous); use **z-buffer/visibility**, not blind alpha differences behind occluders. Do not compare render depth against background at an occluded moving surface; enforce valid observed front surfaces only. Preserve renderer's real output and gradient; no fake proxy result can satisfy this stage. When matching learned appearance can overfit the heldout frames, freeze Gaussian colors/canonical geometry for candidate selection or define an explicitly separated training/validation protocol. Report heldout full-rasterizer scores and visual overlays for each candidate, per camera, especially black-box motion frames.

Only after frozen-asset evaluation demonstrates actual discrimination should the implementation alternate **small joint corrections** with bounded Gaussian optimization, comparing against heldout frames and refusing changes that worsen the other camera or collapse moving support. Never use full training to overwrite evidence of incorrect Phase-1 kinematics.

**Gate F:** actual renderer—not proxy—passes rotation/covariance/metric-depth/occlusion/gradient tests; identical canonical asset and heldout observations compare P/R/static fairly. Black-box and fridge renders show the correct physical surface moving in the right direction. The score is not accepted if all visible pixels can be explained by the static base or a movable Gaussian opacity trick.

## 9. Stage G — End-to-end adapter and downstream FreeArtGS training

Once Gates A–F are documented, connect selected, physically validated hypotheses to existing `preprocessing/prepare_multiview.py` / reconstruction interface through the **existing** `exports.py::export_for_existing_freeart`. Use joint `A_H(q_t)` shared across cameras; static physical camera poses fixed. For ambiguous cases, export both candidates or an explicit abstention—not a silently arbitrary joint. Avoid per-camera independent 2-part ICP and reference-camera selection for this new path; preserve old code behind the opt-in flag.

Validate the downstream virtual camera poses and `transforms_0/1.json` loading with per-camera intrinsic matrices, timestamps, mask ownership, motion direction, `q` gauge/sign, canonical 3DGS frame and quaternion layout. Stage adaptation tests on a tiny synthetic 3DGS model before real training. Real end-to-end: identical compute budget and canonical model initializations for legacy vs new mode, train on matched cameras and time frames, report heldout camera/frame RGB, depth and mask scores, joint axis/origin/q with uncertainty and support leakage. Report failures and GPU OOM explicitly. No full-batch training until small-scene gates pass.

## 10. Mandatory test suite: do not replace tests with anecdotes

Create/extend these test files (choose final filenames to match repo conventions; names below are expected targets):

```text
tests/test_joint_discovery_depth_conflict.py
tests/test_joint_discovery_depth_validity.py
tests/test_joint_discovery_correlated_noise.py
tests/test_joint_discovery_multiview_holdout.py
tests/test_joint_discovery_remote_hinge_observability.py
tests/test_joint_discovery_contact_windows.py
tests/test_joint_discovery_wrench_calibration.py
tests/test_joint_discovery_force_selection.py
tests/test_joint_discovery_gaussian_renderer.py
tests/test_joint_discovery_end_to_end.py
```

**Synthetic deterministic controls (minimal required scenarios):**

1. Static object, white depth noise: no moving support or phantom q.
2. Static object, camera-correlated depth bias: cannot be called moving by an unrealistically confident track-z.
3. Revolute with known near hinge and 15–25° motion, two cameras, one grazing surface, realistic edge occlusion and depth bias: recovered finite hinge, not a distant 2° surrogate; declare ambiguous when truly unobservable.
4. Prismatic drawer with 10–20 cm displacement, weak 1–2° nuisance play/miscalibration: no confident revolute caused by remote hinge.
5. Same moving component only visible in one camera at a time; other camera observes no moving pixels: shared q and persistent support, no forced part split.
6. Tracks enter late and leave early; 2D-only observations retained; no all-frame visibility requirement; no fake 3D anchors.
7. Tool inside whole-object mask (moving independently); known TCP exclusion must not erase adjacent true contact surface; no robot-track promotion.
8. Depth edge has one invalid neighbor, or foreground/background mixture; no `-inf`, no background-depth update of foreground point; preserve valid RGB tracks.
9. Revolute radial/axial residual vanishes for true material motion; for finite 15° steps exact SE(3) residual—not chord tangent—vanishes.
10. Transparent/occluded/behind-surface renderer visibility: no erroneous depth loss behind robot or base; image gradients match FD.
11. Signed force decomposition with rotated `T_EE_S`, known tare/gravity and contact sign, point-lever torque transport; unknown rotation must fail calibration gate.
12. Linear TCP sliding across a **rotating** object: correct surface-normal constraint, nonzero tangential slip, no full-velocity equality; force ratios vary with angle under justified planar frictionless geometry.
13. Linear TCP pushing a **prismatic** object with genuine slip; no spurious rotational preference from straight TCP path.
14. Contact absent or heuristic only, force under noise, unknown sensor rotation, hidden underside surface: `ABSTAIN`; never hallucinate valid patch normals.
15. Wrench shuffled in time, sign flipped, gravity-only: force direction/energy cannot spuriously select an articulated type.
16. Heldout track correlations and shared surface patches: naive z-score overstates significance, clustered/block procedure appropriately widens uncertainty; no train/test leakage.
17. Remote hinge translation limit and near-hinge weak-but-real revolute: distinguish numerical degeneracy without hard-wiring the expected labels.
18. All candidate likelihoods evaluated on identical observations with common support and appropriate missingness; no per-hypothesis score advantage by dropping a hard frame.
19. Virtual-camera direct vs object-motion render equivalence for RGB/alpha/depth and Gaussian covariance rotations, not only transformed point coordinates.
20. Legacy configuration unmodified, opt-in path refuses stale outputs; no changes to FreeArtGS remote or original datasets.

Test every mathematical Jacobian or custom CUDA binding against finite differences, including joint derivatives and pixel-depth projections; document tolerances by precision (float32 vs float64). Report test command, count, PASS/FAIL/SKIP reason, GPU/driver, seed and runtime. Synthetic GT may initialize test data but must be hidden from the solver.

## 11. Real experiment matrix and comparison contracts

**Required scenes and frame ranges (these ranges are historical reproductions, not guaranteed optimal contact windows):**

| Scene | RGB-D range | Cameras | Role | Expected failure to study |
|---|---|---|---|---|
| `final_black_box_episode_000000` | `70:128:2` | `001+002` default; `000` and `000+001+002` diagnostics | Primary blocker | 2.8° remote hinge vs ~15–21° independent reference; depth corrupts R |
| `final_red_fridge_episode_000000` | `90:148:2` | `001+002` | Revolute control | Contact detection/door-edge surface unavailable; no wrench in previous archive |
| `white_drawer_20261021_161154_episode_000000` | `150:330:6` | `001+002` | Prismatic control | Remote hinge falsely wins; EE-only tool mounting uncalibrated |
| `final_robot_dog_episode_000000` | `80:140:2` | `001+002` | Prismatic/force diagnostic | Reported 18 heuristic contact frames, surface OK; test actual contact calibration |
| `final_orange_suitecase_episode_000000` | `50:120:2` | `000+001+002` diagnostic | Ambiguity control | ~5 m hinge prismatic limit; insufficient observed contact face |

Full episode force traces may require a **different evaluation window**, but choose contact windows from independent annotations/measurements, not from which candidate wins. Keep the visual black-box window fixed for comparable visual A/B studies.

Predefine configurations, all with same raw tracks, heldout folds and camera calibrations:

```
B0 legacy FreeArtGS (existing archived baseline and rerun where possible)
B1 MoMa-SG reference (only if actually run and documented)
B2 new current vision-only 2D+depth (frozen baseline commit)
B3 new 2D-only
B4 new depth validity/uncertainty correction
B5 B4 + appearance/surface support
B6 B5 + blocked-time/camera holdout and revised uncertainty
B7 B6 + validated TCP contact
B8 B7 + compensated force model
B9 B8 + full Gaussian renderer feedback
B10 full downstream reconstruction vs B0 and B6
```

Do **not** conflate versions already named B4/B9 in earlier docs with these IDs; append a new `experiment_id` namespace (e.g. `NEXT_A0`, `NEXT_A1`, …) in real output manifests to avoid historical collisions. Create a `comparison.csv` and `summary.md` with:

- `decision`, ambiguity reason, label stability across folds, `z` and clustered confidence interval, number of independent camera/time groups.
- `axis` direction and hinge-line uncertainty; q range and q trajectory vs reference/GT **only after evaluation**; prismatic axis and displacement; degeneracy flags.
- Moving/static/outlier/unknown track counts per view/time, base leakage diagnostic, valid-depth fractions, edge incidence, RGB-track residuals and depth residuals in proper physical units.
- TCP contact samples by provenance (confirmed/probable/heuristic), `surface_supported` counts per hypothesis and intersection, signed contact normal/slip residual, tip-to-observed-face distance and estimated uncertainty.
- Wrench calibration status, valid-force counts, signed `F_parallel`, `F_perp`, `rho_parallel`, `rho_perp`, candidate force-direction error, Coulomb `mu`, passivity violations, all null controls, and whether **force actually changed the fitted parameters or decision**.
- Frozen-assets renderer RGB PSNR or robust loss, depth MAE/mm, foreground mask metric, visibility coverage, per-camera counts; downstream heldout metrics and runtime.
- Input artifact hash, git SHA/diff, config SHA, calibration SHA, time-map hash, GPU/environment, deterministic seed and train/validation split.

**Critical controls:** compare `cam_001+002` and `cam_000` on their own raw evidence; never call one camera 'GT'. For single-view view-conflict diagnostics provide confidence intervals and calibration-error sensitivity rather than silently trusting cam_000.

## 12. Concrete implementation order and stop gates

| Phase | Deliverable | Required before next phase |
|---|---|---|
| 0 | Audit, frozen baseline reruns, reproducible manifests | Existing tests pass or precise pre-existing failures recorded |
| A | Depth-vs-RGB counterfactuals, region/time/camera contribution attribution | Quantified first failure; per-observation evidence table and overlays |
| B | Corrected observation support/noise model and static/moving support | Synthetic correctness; default-camera black box improves **without base leakage** or honestly abstains |
| C | Block/camera holdout, physical R/P observability confidence | White drawer and remote-hinge tests avoid confident false R; positive label supported geometrically |
| D | Independently reviewed contact intervals, calibrated wrench frame/time/tare/gravity | Required calibration gates pass, or force path remains `ABSTAIN` |
| E | Joint+TCP+force candidate fitting and fair heldout comparison | Demonstrated parameter/score change on valid synthetic contact; null controls fail appropriately |
| F | Actual Gaussian RGB-D/alpha feedback | Renderer and occlusion numerical tests; heldout render comparison |
| G | FreeArtGS adapter / downstream GS training | Small-scene end-to-end passes; no camera drift, no loss of legacy capability |

If phase A identifies wrong association of depth and tracks but stage B cannot fix it without changing masks or cameras, stop and report that failure. If stage D cannot verify wrench calibration, **continue** visual and renderer work, but leave the force selection disabled rather than invent compensation. Do not train 3DGS and claim a gain from a Stage-E joint model which has never used force.

## 13. Exact expected new files and API behavior

Codex may reorganize thoughtfully but should deliver documented equivalent interfaces:

```
estimotion_2part/joint_discovery/depth_conflict_audit.py
estimotion_2part/joint_discovery/depth_evidence.py         # if Stage B is validated
estimotion_2part/joint_discovery/validation_folds.py       # grouped/camera holdouts
estimotion_2part/joint_discovery/observability.py          # matrix + remote hinge
estimotion_2part/joint_discovery/contact_annotations.py
estimotion_2part/joint_discovery/wrench_calibration.py
estimotion_2part/joint_discovery/physical_selection.py
estimotion_2part/joint_discovery/gaussian_feedback.py
samuele/audit_joint_contact_windows.py
pipelines/joint_discovery_next_steps.sh                   # staged reproducible A/B
configs/joint_discovery_next_steps.example.yaml
...new tests listed in §10...
docs/joint_discovery_next_steps_results.md
```

Preserve `--help`, existing CLI subcommands and standalone `joint_discovery_scene.sh`. New flags should use typed config and reject unsupported inputs; do not put a flag in `config.py` unless calling it causes real work or explicit `NotImplementedError`. Do not enable physics by changing `require_calibrated_wrench` to false. Exports must distinguish:

```
used_for_visual_fit: bool
used_for_contact_fit: bool
used_for_force_fit: bool
used_for_model_selection: bool
calibration_verified: bool
contact_annotation_provenance: string
holdout_camera_ids: list[str]
heldout_time_block_indices: list[int]
uncertainty_method: string
renderer_evaluation_kind: point_proxy | actual_gaussian
```

Provenance and failure-status enums are part of the output contract, not optional verbose logging. Never compare unrelated runs only because their output folders have similar names.

## 14. Mandatory final Codex report

Create **`JOINT_DISCOVERY_NEXT_STEPS_IMPLEMENTATION_REPORT.md`** in repository root, and `docs/joint_discovery_next_steps_results.md` with per-scene tables. Report:

1. Baseline commit, branch, status, changes and diff, plus full list of modified files.
2. Implemented math and code entrypoints; what is still **NOT IMPLEMENTED** or only diagnostic.
3. Per-stage test commands, exact outputs, skipped tests and why; deterministic seeds, GPU, runtime.
4. Black-box source-localized depth conflict with numeric attribution by camera/patch/time, before/after plots and *same observation* likelihood tables.
5. Candidate joint states, moving support, geometry observability and correlated fold confidence before/after; **not just joint type**.
6. Calibration evidence: `T_WB`, `T_EE_S`, force sign, tare, gravity, time sync, uncertainty; list conditions not verified.
7. TCP contact and actual force contribution to training and selection, with sample counts and zero/missing states; S2 vs S3 vs S4 with identical data.
8. Full real Gaussian renderer pixel ownership/occlusion and gradient checks; heldout images and quantitative scores. If not run, explicitly state this.
9. End-to-end FreeArtGS results or reason not run; compare exactly matched legacy and new modes.
10. Remaining blockers with measurable next actions. Any failed gate must be marked **FAIL/UNVALIDATED**—not described as fixed because tests on unrelated scenes pass.

### First action for Codex

**Before implementing any learned prior, heuristic force threshold or full GS optimization: run the immutable Stage-A counterfactual black-box depth/RGB audit.** We need to know which observations make the physically plausible hinge *less likely* under the current likelihood. Fix that evidence first; then let joint geometry, known sliding TCP and calibrated force reinforce the correct physically supported hypothesis rather than competing against corrupted depth measurements.
