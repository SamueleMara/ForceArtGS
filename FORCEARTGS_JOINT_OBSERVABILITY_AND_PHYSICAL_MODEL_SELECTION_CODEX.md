# CODEX IMPLEMENTATION DIRECTIVE — ForceArtGS: Observable Joint Motion Fields, Physical Joint Identification, and Reliable R/P Selection

**Target repository:** `~/ForceArtGS` (or the actual checkout in which the G1–G8 joint-aware pipeline was implemented).  
**Purpose:** Replace the present remote-hinge gate and statistically unreliable render-based selection with a physically grounded, uncertainty-aware joint-decision system that **does not mistake a prismatic approximation for a measured joint**.  
**Scope:** Joint discovery, static/moving support refinement, observability, held-out scoring, rendering evidence, and optional TCP/wrench constraints. Integrate with the authoritative joint-aware reconstruction handoff without changing the core 3DGS representation.  
**Key regression cases:** `final_black_box_episode_000000`, `final_red_fridge_episode_000000`, `white_drawer_20261021_161154_episode_000000`, `final_robot_dog_episode_000000`, `final_orange_suitecase_episode_000000`.  
**Existing conceptual reference:** MoMa-SG's regularized articulated twist/point-track estimator; reuse mathematically compatible ideas, **do not paste its model-selection rules as a substitute for the requirements below**.

> **Required outcome:** An output object containing (a) evidence for spatially uniform translation versus shared angular velocity, (b) a physically meaningful and uncertainty-qualified hinge or slide model when observable, (c) an explicit physical-identification status, and (d) a possibly distinct *working reconstruction motion model*. There must be no silent conversion from ambiguity or failure into a fake physical joint in URDF/USD/MJCF.

---

## 0. FIRST: Audit the actual checkout and preserve the working pipeline

1. Before editing: `git status --short`, branch, SHA, worktree, submodule SHAs, and new/dirty files. **Do not reset, checkout over, clean or overwrite** local work. Use a separate feature worktree or carefully additive changes. Check whether the subsequent G1–G8 integration already changes the files named below; adapt rather than duplicate.
2. Read the real data flow, not just the existing `selection.json`:
   - `estimotion_2part/joint_discovery/{observations,candidates,geometry_fit,assignments,rigid_kinematics,visual_losses,model_selection,config,cli,exports}.py`
   - any newer dense segmentation, rendering hypothesis comparison, contact/force, and joint-constrained Nerfstudio stages in the **local checkout**;
   - `preprocessing/prepare_multiview.py`, `preprocessing/estimate_joint.py`, `pipelines/freeartgs.sh`, and the joint-aware handoff;
   - tests named `test_joint_discovery_*.py` and any newer integration tests.
3. GitHub `SamueleMara/ForceArtGS` default branch as inspected 2026-11-18 has:
   - `candidates.py:seeds_from_proposals`: **deliberate remote-hinge revolute seeds** of radius 1 m and alternative hinge seeds;
   - `candidates.py:rigid_flow_omega`: pairwise instantaneous rotational-flow diagnostic, **not used for final articulation**;
   - `geometry_fit.py:fit_geometry`: nonlinear global R/P and canonical-point optimization with 2D and depth residuals;
   - `model_selection.py:score_candidates_on_heldout`: paired track likelihood + threshold; `occam_preference=prismatic` recorded but not consumed as a distinct working-model status;
   - `model_selection.py:observability_report`: hinge radius / moving-support extent gate;
   - `cli.py:run_discovery` and `velocity_field_diagnostics`: export integration points.
4. **The user's report describes newer local render comparison not necessarily present in the public branch.** Find its actual implementation first. Log module/function locations, current test split, training seeds, and whether it uses the *same moving-support pixels* for both hypotheses. Do not invent filenames.
5. Run and capture baseline unit tests and existing scene decisions before modifications. Save machine-readable baselines in a new output directory. Compare output hashes for legacy-mode regression.
6. Implement behind an explicit `joint_discovery.comparison.selection: observable_physical_v2` flag initially. Keep `heldout_predictive` as a reproducible legacy diagnostic, but **never silently invoke it as fallback** in v2. When v2 is accepted, make it the declared default for the joint-aware path only.

### Baseline facts to reproduce (reported, not fresh ground truth)

| Scene | Track-stage evidence | Main issue / expected diagnosis |
|---|---|---|
| Red fridge | R ~17.8–18.9°, hinge ~0.24–0.315 m, z~5.5 | Candidate R supported; ensure hinge and direction survive new tests |
| White drawer | Remote R ~1°, axis ~9.1 m, predicted curvature ~1.4 mm | R collapses to a slide; working P appropriate, physical P requires independent confidence |
| Black box | Remote R ~2.7–2.8°, hinge ~1.2 m, curvature ~1.4 mm | **KNOWN counterexample**: independently observed hinged lid ~15–21° on this clip; failure stems from support/view/depth, NOT proven prismatic |
| Robot dog | Remote R ~2.5°, hinge ~2.1 m, ~2.1 mm curvature | Type from present observations unresolved; inspect object and actual contact evidence |
| Orange suitcase | R ~15.3°, hinge ~0.73 m from object centre, only ~752 moving tracks | Significant rotational field may exist but model/support/assembly plausibility uncertain |

**Important:** The black-box reference estimate is for **evaluation only**, from `docs/multiview_root_cause_black_box.md`; never use this reference as a seed, prior, model-selection input, or a per-scene hard-coded rule. White drawer's label may be used as a **sequestered evaluation label** if user confirms it; do not train on labels. For dog and suitcase, do not fabricate ground truth.

---

## 1. Replace the conceptual model: physical type ≠ fitting approximation

The current one-label output conflates three different propositions:

1. **What rigid flow is observed?** Rigid translation, rotational flow, no motion, or inconsistent/non-rigid flow.
2. **What underlying 1-DoF physical joint is identified?** Revolute, prismatic, not identifiable, or neither single-DoF model valid.
3. **What motion model should render this particular short sequence?** It may use a prismatic approximation of a short revolute arc *without claiming a real slider*.

### 1.1 Mandatory status enum

Define:

```python
PhysicalJointStatus = Literal[
    "IDENTIFIED_REVOLUTE",
    "IDENTIFIED_PRISMATIC",
    "TYPE_UNOBSERVABLE",
    "MODEL_INCONSISTENT",
    "NO_SIGNIFICANT_MOTION",
]
WorkingMotionModel = Literal["revolute", "prismatic", "static", "none"]
```

A status of `IDENTIFIED_PRISMATIC` requires **positive evidence for one shared linear translation** *and a quantitatively bounded rotational field* plus any available assembly/contact evidence; the mere presence of an arbitrarily remote revolute fit is **not** positive proof of a real slider. `TYPE_UNOBSERVABLE` with `working_motion_model="prismatic"` is a valid, honest and usable result.

### 1.2 Explicit evidence and readiness fields

Do not remove the legacy `decision` field abruptly; version the downstream schema. Minimum output:

```json
{
  "schema_version": 2,
  "physical_joint_status": "TYPE_UNOBSERVABLE",
  "working_motion_model": "prismatic",
  "working_model_is_approximation": true,
  "physics_joint_export_ready": false,
  "evidence": {
    "moving_vs_static": {},
    "rotation_field": {},
    "translation_field": {},
    "revolute_vs_prismatic": {},
    "hinge_geometry": {},
    "assembly_geometry": {},
    "visual_counterfactual": {},
    "tcp_contact": {},
    "force": {}
  },
  "reasons": [],
  "source_manifest_sha256": "..."
}
```

**No downstream URDF/USD/MJCF producer may treat `working_motion_model` as a certified physical joint.** The G8 export must require `physics_joint_export_ready=true`, or explicit documented user-approved override with `assumed` provenance. Existing simulations created under a forced working model must be flagged non-physical/diagnostic.

---

## 2. Core insight: rotational evidence that does not depend on hinge location

Let `X_i(t)` be the position of the *same material point* in a common fixed world. Under a 1-DoF rigid joint:

```text
Prismatic: v_i(t) = d_dot(t) u_P
Revolute:  v_i(t) = theta_dot(t) u_R × (X_i(t) - o_R)
```

For a pair of points `i,j` on the same rigid moving part:

```text
Prismatic: v_i(t) - v_j(t) = 0
Revolute:  v_i(t) - v_j(t) = omega(t) × (X_i(t) - X_j(t))
```

**The hinge point `o_R` is eliminated.** The position-dependent rotational flow, rather than a fitted hinge radius, is the first test for genuine rotation.

For finite motion from frame `a` to `b`, avoid replacing finite differences with instantaneous tangent constraints:

\[
 \Delta X_i-\Delta X_j=(R_{ba}-I)(X_i(a)-X_j(a)).
\]

This is exact when the same rigid material points are observed at both frames. Translation cancels. For small angles, the right-hand side becomes `omega_ab × (X_i - X_j)`; **do not use the small-angle approximation as the final fit over a 15–40° opening**.

### 2.1 Important statistical distinction

Naively generating all `N(N−1)/2` point pairs creates strong correlations because the same point participates in many differences. **The number of pairs is not the effective sample size.** Prefer weighted centered Kabsch/Procrustes on tracked point correspondences, or explicitly use the full induced covariance of pair differences. Do not multiply a pseudo-likelihood over overlapping pairs as if independent.

### 2.2 API: `motion_field.py`

Add `estimotion_2part/joint_discovery/motion_field.py` with tested public functions:

```python
@dataclass
class RigidIntervalEvidence:
    frame_a: int
    frame_b: int
    camera_ids: tuple[str, ...]
    n_tracks: int
    n_effective_tracks: float
    spatial_baseline_m: float
    rotation_matrix: np.ndarray  # finite R_ba, SO(3)
    translation_vec_m: np.ndarray
    rotation_vector_rad: np.ndarray
    rotation_cov_rad2: np.ndarray | None
    translation_cov_m2: np.ndarray | None
    singular_values: np.ndarray
    rank: int
    condition: float
    per_camera_residuals: dict
    diagnostics: dict


def estimate_interval_rigid_evidence(observations, support_weights, frame_a, frame_b,
                                     noise, calibs, *, seed) -> RigidIntervalEvidence:
    ...


def estimate_shared_angular_axis(intervals, timestamps, *, robust_config) -> dict:
    ...


def rotation_vs_translation_evidence(intervals, camera_blocks, time_blocks,
                                     *, bootstrap_config) -> dict:
    ...
```

**Implementation requirements:**

- Use calibrated `T_WC` and valid material tracks, not TCP or robot tool points as rigid object tracks.
- Support asynchronous/partial tracks: each interval needs only tracks truly observed at both endpoints; do not require every track at frame zero or throughout the full clip.
- Use weighted proper Kabsch with determinant `+1`; center each interval to separate translation from relative rotational deformation. Provide alternative image-space reprojection refinement using the original observations where 3D depth is unreliable.
- Weights/covariances depend on actual RGB tracking uncertainty, depth confidence, camera viewing angle and local depth edges. **Never treat 3.5–6 cm mean reconstruction error as automatically equal to the uncertainty of every differential 2D motion measurement.** Conversely, do not assume 0.5 px noise when observed tracking is corrupted.
- Per-camera residuals must be reported separately. A camera with grazing depth must not be allowed to override stronger 2D rotation evidence from an informative camera.
- A frame with negligible motion is not evidence that the angular axis is absent. Skip weak intervals for axis estimation without classifying the entire scene static.
- Estimate finite rotation `log(R_ba)` with consistent sign and angle unwrapping. For sign ambiguity `u,q ↔ -u,-q`, canonicalize once per scene, not independently per interval.
- Fit a *fixed* global axis `u` while permitting per-interval signed `Δθ`; quantify out-of-plane angular residuals and conditioning. Do **not** fit one axis per frame and declare consistency after sign-independent averaging.
- Include testable output from `candidates.rigid_flow_omega` for comparison, but do not use its raw pair count as statistical power.

### 2.3 Observability of angular motion

Compute and export **all** of:

- magnitude and uncertainty of finite `log(R_ba)` and the shared angular-axis estimate;
- 2D reprojection residual improvement of R over a common translation in identical tracked pixels;
- distribution of residual differential displacement over **spatially separated** track groups;
- spatial baseline, eigenvalues of the centered point configuration, rank and condition number (a narrow strip or tiny patch cannot constrain all rotation components);
- camera-wise support and contradictory views;
- time-block bootstrap intervals, estimated from distinct independent temporal blocks/clusters rather than all overlapping pairs;
- sensitivity to removing each camera and excluding suspect depth-edge observations.

Use a calibrated null distribution for no rotation and pure translation, including tracker bias, occlusion and correlated depth. A convenient numeric `z` threshold may be reported, but **never rely on one uncalibrated z score**.

### 2.4 Shared-axis optimization

With `R_ab` from independently observed intervals, solve:

\[
 \min_{\|u\|=1,\{\Delta\theta_k\}}
 \sum_k \rho\left(\|\log(R_k)-\Delta\theta_k u\|^2_{\Sigma_{\omega,k}^{-1}}\right)
\]

where `rho` is a robust loss, with a fixed, documented sign gauge. If `R_k` contains zero/near-zero rotations, these intervals do not constrain `u`. Report identifiability, not an arbitrary normalized axis.

**Note:** `u` and `Δθ` should be initialized from distributed coherent support, *not* predominantly static-base tracks.

---

## 3. Fit moving support and articulation together, without self-fulfilling segmentation

The current `fit_hypothesis` EM architecture is useful. Retain its robust static/moving/outlier responsibilities, but augment it with a **common spatial-flow evidence term** and mandatory conflict diagnostics. Do not give a candidate carte blanche to select only the points that make it look good.

### 3.1 Common observation pool and independent core

Define three sets:

- `S_core`: confident fixed-world tracks measured consistently across time/cameras;
- `M_proposal`: object foreground tracks with demonstrable motion under at least one plausible shared flow, **before** R/P selection;
- `U`: uncertain or inconsistent tracks; never automatically mark them static.

Robot wrist/tool points must be excluded using measured tool occupancy/occlusion where available, but do not delete the true object–tool contact neighbourhood indiscriminately.

### 3.2 Joint-probability EM update

For material track `i`, estimate `p_i^S`, `p_i^M`, `p_i^O` with persistent identity across frames. Score `M` from the candidate *shared 1-DoF motion* and `S` from a fixed-world state; use robust RGB reprojection and valid depth. Unknown/tracker-outlier is distinct from physical static.

Each hypothesis may refine its responsibilities, but **for model selection also evaluate both candidates on a shared, candidate-independent observation set and report coverage**. A branch cannot win by marking difficult moving points outliers or by retaining a much smaller convenient patch.

### 3.3 Structured consistency checks

For each proposal cluster, report:

- moving/static ratio by camera and time;
- rigid residual and depth-edge contamination;
- percentage of moving support along its 3D spatial span;
- relative-velocity variation over distant point pairs;
- static/moving interface compatibility;
- evidence for two different moving rigid groups or moving static base;
- multimodal residuals suggesting inconsistent tracking, calibration or 2-DoF/non-rigid scene.

If one consistent 1-DoF field cannot explain enough real object support, set `MODEL_INCONSISTENT`; do not force R/P simply because one branch has the smaller error.

### 3.4 Avoid metric leakage

- Motion proposal stage does not read hidden reference axis, known class label or evaluation tracks.
- Fitting `p_i^M` on training timestamps only; held-out observations cannot influence assignments, noise calibration, track selection, initialization or q estimation.
- Keep the same confidence/coverage definitions for R and P. Store explicit sample counts before and after candidate-specific filtering.
- Repeat without depth / with calibrated depth / per camera as diagnostics, not per-scene oracle cherry-picking.

---

## 4. Separate joint type from hinge geometry; physically test the assembly

### 4.1 Revolute fitting: estimate angular flow first, hinge second

Only once rotation is observably nonzero, initialize a common axis `u` and estimate a finite hinge line:

\[
 X_i(t)=o+R_u(q_t)(X_i^0-o),\qquad o\sim o+\alpha u.
\]

Optimize over `u` on `S²`, **two** perpendicular coordinates for `o` (along-axis coordinate is gauge), `q_t`, canonical anchors and support probabilities. Use actual exact SE(3) transport. Include the same 2D/depth and smoothness terms as the current R fitter.

For small angles `(I−R)` is ill-conditioned. **Report huge line covariance or `HINGE_UNOBSERVABLE` instead of a precise 9 m pivot.** Do not clamp the hinge to the object box during initial estimation and silently report success.

### 4.2 Assembly plausibility prior, not an arbitrary bounding-box veto

When reliable dense static/moving geometry exists, compute a **candidate-independent attachment region** from:

- boundaries of the canonical static and moving part surfaces;
- possible static/moving interface or bearing/guide contact zones;
- uncertainty/visibility of those surfaces;
- corroboration across different calibrated views.

A hinge may legitimately lie just outside a visible object's box or in an occluded bearing. Thus:

\[
 L_{\mathrm{assembly}}(u,o)=\sum_k w_k\,\rho\Bigl(
     \frac{d(\text{hinge line},\,\mathcal A_k)}{\sigma_{\mathcal A,k}}
   \Bigr)
\]

where `A_k` are **independently grounded attachment candidates** and the term is **disabled/ABSTAIN** if the interface is not observable. Do not pick the nearest hinge edge by category semantics; do not optimize the interface to match the chosen hinge without an independent data term.

A large hinge offset relative to the whole-object extent is a *warning or a prior penalty conditional on observed assembly geometry*, never a universal impossibility proof.

### 4.3 Prismatic fitting and positive slider evidence

Prismatic is **one common velocity/displacement vector** across material points:

\[
 X_i(t)=X_i^0+d_t u_P,
\]

with per-frame deviations from that field compatible with calibrated uncertainty. Evaluate the upper confidence bound on any shared angular field and the minimum spatial rotational distortion the camera system could detect. Where linear guide/contact geometry is independently observed, add a conditional guide-alignment term.

Do not label `IDENTIFIED_PRISMATIC` merely because an R fit is implausibly remote. If no guide can be observed, prismatic may remain the preferred *working model* while physical type is `TYPE_UNOBSERVABLE`.

### 4.4 Keep remote-hinge solutions as **degeneracy controls**

Keep remote-hinge seeds in `candidates.py` for fair model-comparison and negative tests, with provenance `remote_hinge_seed=true`; they should not be silently removed to inflate P classification. The output must explain when the optimal R model approached the P limit:

- radius-to-scene/support spatial scale;
- observed angle and its confidence interval;
- best fitted P direction and displacement;
- maximum and robust-quantile **3D and 2D R/P prediction differences** over common moving support;
- distinguishability vs actual camera/track noise;
- hinge covariance/condition number;
- remaining residual angular evidence after eliminating translation.

A remote-hinge warning without those supporting quantities is insufficient to decide physical type.

---

## 5. Model distinguishability from real observation-space predictions

Do not use one point's arc-to-chord sagitta as the only discriminator. Define per-observation predictions under both hypotheses using **identical material IDs, cameras, timestamps and masks**:

\[
 \delta_k=h_R(k)-h_P(k),\quad
 I_{R/P}^2=\delta^\top\Sigma^{-1}\delta.
\]

Include `(u,v)` pixel displacement, and **only valid metric depth** as separate residual components. Correlated track/time measurements require a block covariance, sandwich estimator, or clustered resampling; a diagonal per-pixel `Σ` is permitted as a *descriptive uncalibrated diagnostic only*.

**Critical:** R/P canonical anchors and `q_t` are nuisance parameters. Refit or marginalize them on training data; do not compare an optimized R curve against a frozen, disadvantaged P line. Evaluate distinguishability on prediction sets constructed **without observing the held-out outcomes**.

Two separate quantities must be reported:

1. **Predicted separation:** could *this* measurement setup distinguish R and P with existing support/noise?
2. **Observed predictive preference:** which branch better explains independently withheld observations?

High predicted separation with poor observed fit indicates model/support failure. Low separation means the physical type is unobservable even when a nominal metric favours one.

### 5.1 Do not count common translation as rotational evidence

Report relative motion of points after eliminating the best shared translation. An R fit's full rotation effect (e.g. a quoted 133 mm total displacement) is not evidence against a P fit if both models explain that translation. The relevant signal is **the residual spatially varying component**.

### 5.2 Distinguishability acceptance logic

Implement thresholds as *calibrated rules*, not magic constants. Synthetic simulations should map false-positive/negative rates as a function of camera geometry, baseline, depth noise, motion angle, moving support and degree of motion. Accept threshold settings from config and save their provenance. When the experiment has too few independent time blocks, report `INSUFFICIENT_INDEPENDENT_EVIDENCE`; do not invent significance.

---

## 6. Proper statistical evaluation: independent blocks and matched evidence

### 6.1 Replace track-count-based z-as-certainty

The existing paired-track `z = sum(d)/(std(d)*sqrt(n_tracks))` is not enough because tracked points share images, surfaces, calibration and failure modes. Keep it as a legacy field with a warning, but use a blocked/clustered estimator for decisions.

Use temporal blocks wider than estimated residual autocorrelation; include camera/surface-cluster resampling sensitivity. Record the block partition **before fitting**. Do not make contiguous held-out frames into artificial independent replicates. Derive confidence intervals for the **difference in predictive score**, not for each branch alone.

### 6.2 Held-out joint trajectory leakage

The current global fit interpolates `q` at unobserved frames; a frame is not genuinely held out if its ground-truth tracker values influenced `q` indirectly. For selection:

- Fit axis, origin, canonical anchors and training `q` using only training frames.
- At held-out time, predict `q` from a frozen temporal model trained without those measurements (interpolation across training endpoints is allowed **only if explicitly documented as a trajectory-prediction benchmark**).
- Do not estimate or update held-out `q` using held-out RGB, depth or segmentation.
- Also report evaluation on **extrapolated held-out endpoint blocks**, which is harder and helps expose short-arc degeneracy. Never confuse a bad extrapolator with proof of wrong joint type: retain both score components.
- If users want image-conditioned inference of `q` at test time, it is a **different task**; evaluate it separately with withheld cameras/pixels and prevent using the same pixels for both inference and scoring.

### 6.3 Model-complexity correction

Use a declared likelihood-based approximation (e.g. a parametric bootstrap of the paired predictive improvement or effective degrees of freedom) to account for additional R parameters. The current naive `bic_report` with raw correlated pixel counts is diagnostic, not a principled definitive Bayes factor. Allow a physical prior from independent geometry only, never a hidden class-specific preference.

---

## 7. Joint model selector: explicit evidence hierarchy, no false certainty

Implement in `estimotion_2part/joint_discovery/physical_selection.py`:

```python
@dataclass
class PhysicalDecision:
    physical_joint_status: str
    working_motion_model: str
    working_model_is_approximation: bool
    physics_joint_export_ready: bool
    selected_candidate: str | None
    reasons: list[str]
    evidence: dict
    uncertainty: dict
    failure_codes: list[str]


def select_physical_joint(
    *, motion_evidence, model_fits, common_support, attachment_evidence,
    predictive_evidence, render_evidence=None, robot_evidence=None, cfg
) -> PhysicalDecision:
    ...
```

Run the following **in order**:

1. **Sufficient object motion?** Compare against static null, accounting for uncertainty. If no: `NO_SIGNIFICANT_MOTION`, working `static`.
2. **Can one coherent 1-DoF rigid field explain meaningful moving support?** If no, or incompatible groups/independent static-base motion dominate: `MODEL_INCONSISTENT`, working `none` or a diagnostic model (with reasons). This is **not** equivalent to `TYPE_UNOBSERVABLE`.
3. **Is rotational flow independently measurable and aligned across time?** If yes, attempt a finite-hinge model and assembly check. If R explains held-out data and its hinge is identifiable/physically plausible, choose `IDENTIFIED_REVOLUTE`. If R strongly rotates but the axis/hinge is inconsistent with the assembly or canonical support, return `MODEL_INCONSISTENT`, not P.
4. **Is shared translation supported?** If yes and the upper bound on differential rotation is sufficiently small for the observation budget, consider `IDENTIFIED_PRISMATIC` **when positive attachment/guide evidence or comparable independent support exists**. Otherwise use `TYPE_UNOBSERVABLE` plus working `prismatic` if that is the best reconstruction approximation.
5. **Are R and P observationally equivalent?** Return `TYPE_UNOBSERVABLE` even if the revolute likelihood is marginally better or the remote-hinge branch collapses to translation; prefer the lower-complexity **working** model for reconstruction only.
6. **Physically valid TCP/force evidence** can upgrade/reject a hypothesis when it genuinely distinguishes the two, with uncertainty and abstention, not a hard-coded force ratio.
7. **Render evidence** can veto a candidate if adequately repeated, predictive and demonstrably beyond training noise; it cannot confer a physical type based on six correlated frames.

Every decision must include a counterfactual reason of the form: “Why is the rejected model not also acceptable?” If both remain acceptable within uncertainty, refuse physical identification.

### 7.1 Examples that MUST NOT be hard-coded as rules

- `remote_hinge` + `short_arc` **does not entail** physical prismatic. Black-box hinged lid is a regression case.
- Hinge outside moving-track bounding box **does not entail** invalid R. Sparse tracks can cover only the centre of a door.
- Large R angle **does not entail** R valid. A moving gripper/tilting static base can fake rotation.
- Minimal RGB render error **does not entail** correct joint, especially with branch-specific densification.

### 7.2 Working reconstructions and URDF

In ambiguous cases, a reversible prismatic working branch may be used for appearance fitting, **but it must not overwrite R/P candidates or join export fields**. Carry `physical_joint_status`, uncertainty, provenance and restrictions through G1–G8. The G8 exporter must reject inferred physical joint files when physical type is unidentifiable unless an explicit user-supplied independent joint specification is approved and recorded.

---

## 8. Rendering evidence: controlled counterfactual first, stochastic end-to-end second

Locate the actual G6 render-comparison code in the local branch. Replace its current “two correlated blocks of three frames + metric bootstrap + unanimous signs” logic. Keep legacy render outputs for regression.

### 8.1 Controlled counterfactual (isolates kinematics)

**Same across both hypotheses:** selected object-only RGB-D input, fixed camera calibration, canonical Gaussian initialization, Gaussian parameters, candidate-independent moving-support comparison region, timestamps, exposure, background, resolution, visibility/occlusion logic, and held-out pixels.

Change **only** the articulated SE(3) field `A_H(q_t)`, with candidate `q_t` predicted from training. For a fair frozen-geometry comparison, ensure canonical frame alignment and model-reference gauge are verified. If geometry must be separately re-optimized, that belongs to the next experiment and cannot be called a pure kinematic test.

Metrics: masked RGB/SSIM (diagnostic), trustworthy metric depth on visible object pixels, alpha/object silhouette, and camera/part-specific residual maps. Do not use depth-edge/background measurements to update foreground geometry. Report affected pixels and per-frame/camera errors; a globally small difference should not erase valid evidence on a rare-visible moving surface.

### 8.2 Repeated full reconstruction (accounts for training noise)

For both branches, use equal training budgets, equal initialization strategy, equal data, equal topology constraints and at least two independent training seeds (prefer more when feasible). **Paired seeds** must be logged and repeatable. Compare held-out independent time blocks and report between-seed variability *of the difference* as well as median absolute reconstruction loss.

No “statistically significant” conclusion is allowed from six adjacent frames or when the observed R/P difference is below the measured between-seed variability. Do not compare different part masks without an explicit common-support ablation; report how much the branch segmentation alone changes the result.

### 8.3 Minimum independent validation

Aim for at least six genuinely separated held-out time blocks **where the sequence length and measured temporal correlation permit**. If the clip is too short, mark the render evidence `NOT_IDENTIFIABLE_FROM_AVAILABLE_BLOCKS`; do not loosen the block definition or pretend it has six independent samples.

The final exhaustive render audit may cover all frames for debugging, but it must not be represented as independent held-out evidence if those frames were used in training.

---

## 9. Optional TCP sliding contact and force: compare hypotheses without a fixed grasp

The robot TCP is a known (calibrated) *contact point trajectory* `p_tcp(t)`, **not a material point rigidly attached to the lid/drawer**. The TCP may slide over the same physical face and change contact material point each frame. Do not force `v_tcp = v_moving_part` or align the joint with TCP tangent.

For each hypothesis:

```text
v_object,P(p,t) = d_dot(t) u_P
v_object,R(p,t) = theta_dot(t) u_R × (p_tcp(t) - o_R)
slip_H(t) = v_tcp(t) - v_object,H(p_tcp(t),t)
```

For verified maintained single contact with independently measured/validated contact normal `n_H(t)`:

```text
n_H · slip_H ≈ 0
```

The robot-on-object measured force `f(t)` gives signed parallel TCP force and the orthogonal component:

```text
f_parallel = f · v_tcp_hat
f_perp_vec = f - f_parallel v_tcp_hat
rho_parallel = abs(f_parallel)/norm(f)
rho_perp = norm(f_perp_vec)/norm(f)
```

**Important:** `rho_parallel² + rho_perp² = 1`; the two ratios are *dependent*. For final likelihood prefer the full signed normalized force direction and a bounded shared Coulomb/friction model, not two independent squared error likelihood terms.

Necessary passivity for one maintained passive sliding contact, *with force exerted by robot on object*:

\[
 f\cdot\mathrm{slip}_H\ge -\delta_{\mathrm{uncertainty}}.
\]

This condition is not a sufficient joint discriminator; it can be satisfied by both hypotheses and can fail outside the simple contact assumptions. The power decomposition `P_tcp = Q_H qdot + f·slip_H` is an identity for either model, **not evidence by itself**.

### 9.1 Evidence gating

Robot evidence may enter physical decision only when all relevant conditions hold:

- exact or uncertainty-characterized timestamp/sensor registration and TCP pose;
- `T_WB`, sensor-to-EE wrench rotation, force sign and torque origin verified;
- force tare and gravity/tool compensation verified for the episode (unless an independently validated force-bias model is used);
- verified contact intervals and surface support; no robot tool tracked as object;
- common valid contact frames for competing models; independently plausible contact patch and normal, or explicitly report that only sign/passivity diagnostics are available;
- enough excitation and slip to distinguish candidates; bounded, **shared** friction parameters with no per-frame arbitrary fitting.

On failed gating, output `ABSTAIN` with reasons and **do not alter visual joint selection**. Never lower an arbitrary force threshold to manufacture contact. A verified contact event can guide moving-support inference without being a fixed material-point constraint.

### 9.2 Implementation placement

Reuse and extend `tcp_contact.py`, `wrench_physics.py`, and existing contact-force CLI; do not introduce a second unsynchronized robot data reader. Optimize contact and physics terms with the **same** joint state and canonical frame used by the visual fit; no independent robot-only joint animation.

---

## 10. Configuration, file contracts, output and pipeline integration

### 10.1 Configuration extension

Add a versioned section under `joint_discovery.comparison` (exact YAML keys may be adjusted to local typed config, but validate unknown keys):

```yaml
joint_discovery:
  comparison:
    selection: observable_physical_v2
    evidence:
      robust_finite_rotation: true
      fit_shared_axis: true
      common_support_scoring: true
      calibrate_null_distribution: true
      require_independent_holdout: true
    temporal_blocks:
      separation: auto_from_autocorrelation
      min_independent_blocks_for_render_claim: 6
    assembly:
      enabled_when_independently_observed: true
      forbid_bbox_hard_gate: true
    working_approximation:
      prefer_simpler_when_equivalent: true
      never_certify_type_from_occam_alone: true
    render:
      controlled_counterfactual: true
      repeated_training_seeds: 2
      allow_veto_only_with_replicated_difference: true
    robot:
      optional: true
      require_valid_wrench_for_force_decision: true
  output_schema_version: 2
```

Any numeric significance/practical-equivalence thresholds must be declared and calibrated on synthetic controls, with provenance recorded. **Do not encode `red_fridge`, `black_box` or any scene-specific physical truth into config defaults.** The above is a schema sketch, not a claim that these keys already exist.

### 10.2 New modules (suggested minimal structure)

```text
estimotion_2part/joint_discovery/
    motion_field.py             # robust hinge-free differential/finite rigid evidence
    rotation_observability.py   # shared-axis, rank/noise/null calibration
    support_consistency.py      # candidate-independent moving core + fair coverage audits
    assembly_geometry.py       # independent attachment/guide priors, ABSTAIN when absent
    distinguishability.py      # observation-space R/P prediction separation
    blocked_statistics.py      # time/camera/surface cluster splits and bootstrap
    physical_selection.py      # status vs working model; evidence hierarchy
    ... reuse candidates.py / geometry_fit.py / model_selection.py / cli.py
```

Do not fork all existing kinematics or create a new tracker. Reuse `rigid_kinematics` and `TrackObservations` and preserve backward-compatible serialization where practical.

### 10.3 Required output tree per run

```text
<output>/
  manifest.json
  joint_candidates.json                 # keep both R/P when fitted
  joint_states.npz                     # exact frame map, q, uncertainty/validity per branch
  joint_selection_v2.json              # authoritative physical + working model statuses
  joint_selection_v2.md                # concise interpretable report
  motion_field/
    per_interval.csv                   # R, omega, uncertainty, conditioning, supports
    per_camera_consistency.json
    spatial_flow_diagnostics.npz
    rotation_null_calibration.json
    field_evidence.json
  support/
    common_support.npz
    per_camera_frame_coverage.csv
    track_fate.csv
    conflict_report.json
  assembly/
    attachment_evidence.json
    overlays/                         # only where independently observable
  model_comparison/
    prediction_separation.json
    heldout_block_scores.csv
    block_bootstrap.json
    complexity_report.json
  render_comparison/
    controlled_counterfactual.json
    full_training_seed_report.json
    residual_overlays/
  robot/
    contact_evidence.json              # optional, ABSTAIN if missing
  regression_report.json
```

Large arrays may be NPZ; all reports must be deterministic under fixed seed and record source/file hashes. Do not write fake empty scores as zeros; write null values with explicit status/reason.

### 10.4 Pipeline handoff and protection

- New G1–G8 joint-aware mode consumes `joint_selection_v2.json` and the candidate geometry/trajectory without rerunning legacy `preprocessing/estimate_joint.py`.
- For `IDENTIFIED_REVOLUTE`/`IDENTIFIED_PRISMATIC`, make the accepted candidate authoritative for masks, joint-derived virtual cameras, blending, joint-constrained rendering and final export.
- For `TYPE_UNOBSERVABLE`, retain both candidate artifacts (and possibly both reconstruction branches) and carry the working-model approximation flag. Do not report a physically identified joint or produce a physics-ready URDF from the approximation.
- For `MODEL_INCONSISTENT`, prevent final joint export, provide reproducible frame/camera/track diagnostics; allow only an explicitly requested diagnostic rendering path.
- Protect reference frames: `q_ref=0` at the declared canonical anchor; map all `original_frame` and timestamps correctly, including reverse processed sequences; one `q_t` per physical instant across cameras.
- Freeze calibrated physical cameras; no free per-camera motion to absorb a bad joint; guard against unvalidated similarity rescaling or redundant SE(3) alignment offsets.
- Do not silently swap static/moving part IDs, invert `A(q)` or change Gaussian canonical coordinates in either branch.

---

## 11. Tests required BEFORE claiming correct joint identification

Implement tests below as parametrized pytest/unit tests with deterministic synthetic fixtures. Synthetic generators must simulate real calibrated cameras, material tracks, per-camera occlusion, correlated depth errors, and controlled tracker outliers.

### Gate A — Geometry and differential field

1. Pure P: uniform velocity field, recovered rotational component statistically consistent with zero; no invented physical hinge.
2. Pure R: shared nonzero angular velocity even when TCP trajectory is approximately straight.
3. Exact finite pair-difference identity for arbitrary hinge axis, hinge location, 15°, 40°, negative and nonuniform `Δq`.
4. Translation-invariance: shifting all points/world origin does not change angular evidence.
5. Angular sign gauge invariance `u,q ↔ -u,-q`.
6. Pair-correlation test: multiplying/duplicating track pairs does **not** shrink confidence intervals.
7. Low-rank tiny patch, planar support, line support, or same-radius support: report appropriate ill-conditioning/partial observability.
8. Partial tracks and asynchronous camera views yield correct interval measurements without inventing correspondence.
9. Static-base/robot-tool contamination does not generate confident object rotational evidence.
10. Correlated depth-edge bias can weaken evidence without falsely creating high-confidence rotation.
11. Informative RGB camera should recover rotation even when the other cameras have unreliable grazing depth (without using scene-specific GT during fitting).

### Gate B — Remote-hinge and assembly

12. Real P with R seed at 1, 5, 10 m: record R→P limit; working P allowed, physical R **not** identified.
13. Real short-arc R with remote apparent center: if genuine distributed angular evidence exceeds noise, **do not hard-switch to physical P**.
14. Real R with hinge slightly outside full-object AABB: must not fail a bounding-box hard gate.
15. Real R with highly uncertain/no observed attachment boundary: assembly result `ABSTAIN`, not fabricated hinge support.
16. Geometrically implausible hinge but substantial rotation on wrong/gripper tracks: `MODEL_INCONSISTENT` or unresolved, not `IDENTIFIED_REVOLUTE`.
17. Real P with significant tracker-induced spurious pose rotation: candidate selection remains conservative, logs residual independent angular evidence.

### Gate C — Model selection and uncertainty

18. Static/no motion is `NO_SIGNIFICANT_MOTION`.
19. Unobservable short arc vs linear slide: `TYPE_UNOBSERVABLE` + optional working P, `physics_joint_export_ready=false`.
20. Strong coherent R with finite physical hinge: `IDENTIFIED_REVOLUTE`.
21. Strong common translation and absent measurable rotation with verified guide/contact support: `IDENTIFIED_PRISMATIC`.
22. Multiple moving groups or nonrigid deformation: `MODEL_INCONSISTENT`.
23. Same data represented with 10x duplicated tracks/pixels cannot magically increase confidence.
24. Increasing independent time/camera evidence improves confidence only when it truly increases information.
25. Held-out frame observations never enter candidate optimization/noise calibration/`q` fitting for the same evaluation.
26. Uncertainty bounds widen with correlated noise and shrink only when independent evidence grows.
27. A bad but statistically ‘significant’ R with distant hinge does not become a certified physical R.

### Gate D — Render comparison

28. Two blocks of three consecutive frames cannot be reported as six independent samples.
29. If actual independent blocks are below the required number, render status is insufficient rather than `IDENTIFIED`.
30. Identical geometry and motion but different training seeds: no false R/P preference.
31. Controlled counterfactual changes only motion (camera, geometry, opacity, SH, support fixed).
32. Separate branch segmentations must report coverage and cannot be silently compared as equal-support evidence.
33. An improvement smaller than empirical seed variance is diagnostic, not a decisive veto.
34. Known synthetic hidden moving surface: correct occlusion should be neutral, false early-front occlusion should be penalized.

### Gate E — TCP/wrench and downstream

35. Known sliding contact on R with straight TCP: do not impose zero slip.
36. Force sign reversal / wrong sensor frame / unknown gravity compensation causes `ABSTAIN`, not incorrect physical identification.
37. Invalid TCP contact surface: normal-slip and force-direction tests abstain; visual decision unchanged.
38. `TYPE_UNOBSERVABLE` prevents physics-ready G8 URDF/USD/MJCF generation unless explicit approved independent physics spec is provided.
39. Legacy mode outputs remain bit-identical (or documented, reviewed unavoidable differences); correct pipeline joint-aware mode never invokes vanilla classifier silently.
40. Both R and P exact 3DGS transport and original-frame mapping tests pass in multiview, including backward ordering.

### Real-scene tests (evaluation-only references, no oracle leakage)

**Black box:** Must not automatically declare a physical prismatic joint from the current 2.7°/1.2 m fake hinge. Report rotational evidence by camera and view, especially `cam_000` vs `cam_001/002`; if the algorithm cannot recover sufficient reliable lid support, return `TYPE_UNOBSERVABLE` or `MODEL_INCONSISTENT` with evidence. The independent ~15–21° lid-opening audit is for external evaluation, not a model prior.

**Red fridge:** Reproduce a meaningful ~18° R fit and evaluate hinge stability and uncertainty across cameras/time. Selection as `IDENTIFIED_REVOLUTE` requires actual observable angular field and physical assembly support, not only a z-score.

**White drawer:** Demonstrate R remote-hinge equivalence to P and that the working P branch does not become a fake certified hinge. Check physical `IDENTIFIED_PRISMATIC` only with adequate positive slider evidence.

**Robot dog:** No asserted physical ground truth; report whether motion is consistent with one static+moving rigid articulation and why the type is unresolved or identified.

**Suitcase:** Inspect actual moving support, tool contamination, possible whole-body tilt, and candidate hinge interface. Large R angle with implausible/unsupported hinge must trigger investigation, not arbitrary P or R.

For each scene, report same cameras/window/samples as original baseline and any ablation using `cam_000`, RGB-only, depth-only, RGB-D, and tool-excluded inputs. Do not use per-scene choices based on hidden GT as an automatic algorithm.

---

## 12. Execution plan and acceptance gates

**Implement incrementally; no huge overnight training batch before unit and diagnostic gates pass.**

| Gate | Change | Pass/fail requirement |
|---|---|---|
| 0 | Local audit + baseline | Worktree, versions, exact render comparison traced, legacy results saved |
| 1 | `motion_field.py` | R/P finite-field mathematical and noise tests pass |
| 2 | `rotation_observability.py` + blocked null calibration | No pair inflation, correct rank and uncertainty, camera/time reports |
| 3 | support consistency + geometry | R/P fair support and finite hinge, no semantic ground-truth leakage |
| 4 | distinguishability + physical selection | Five statuses, working model separate, remote-hinge and black-box regressions |
| 5 | controlled rendering + proper blocks | No fake bootstrap significance, no q leakage, repeated-seed audit |
| 6 | optional TCP/wrench | Valid sliding-contact physics; calibrated-evidence gating/abstention |
| 7 | joint-aware G1–G8 handoff | No legacy overwrite; physics export readiness contract enforced |
| 8 | real-scene pilot + report | Fridge, drawer, black box, dog, suitcase; diagnose not just classify |

Required output report: `docs/JOINT_PHYSICAL_SELECTION_V2_IMPLEMENTATION_REPORT.md`, including:

- branch/commit/submodules, exact modified call graph and migration plan;
- equations and covariance assumptions as implemented (not merely copied from this spec);
- before/after decision table with both `physical_joint_status` and `working_motion_model`;
- spatial rotation signal, uncertainty/rank, per-camera support, fitted hinge location/uncertainty, P/R prediction separation in pixels/metres;
- common-support held-out likelihood, effective independent block count, render seed variance, force-valid frame counts;
- all test commands and results, CPU/GPU skip reasons, filenames for actual diagnostic artifacts;
- black-box failure analysis distinguishing “not observed” from “observably absent rotation”;
- unresolved limitations and any needed new acquisitions, with **no fabricated positive results**.

Additional user-facing report: `docs/JOINT_PHYSICAL_SELECTION_V2_DESIGN.md` with algorithm, status schema, integration, and a decision flowchart.

### Proposed CLI checks (adapt to the actual local CLI, do not invent an already existing flag)

```bash
cd ~/ForceArtGS
# Run the current existing test suite before changes.
python -m pytest -q tests/test_joint_discovery_*.py

# After implementing the new option / CLI wiring:
python -m estimotion_2part.joint_discovery.cli --help

# Invoke the actual scene runner with the new v2 config and record outputs.
# The existing scripts may differ across worktrees: check --help first.
```

**No heavy full-scene training before Gate 4 and a cheap Gate-5 counterfactual pilot**, and no declaration of a physical URDF-ready joint until the type and geometry have independent supporting evidence.

---

## 13. STOP / FAIL RULES

- **Do not replace** the user's hinged black-box lid with “prismatic” merely because one corrupted view produced a distant axis.
- **Do not turn** `occam_preference` into an automatic *physical* classification. It may choose a temporary working model.
- **Do not make hinge-within-bounding-box a universal physical requirement.** A hinge may lie outside observed surfaces; require independently grounded attachment evidence when available.
- **Do not score each pair, pixel, adjacent frame, or seed as an independent measurement.**
- **Do not fit joint coordinates from held-out measurements and then score the same measurements as predictive evidence.**
- **Do not allow different densification/segmentation/training randomness to masquerade as joint evidence.**
- **Do not force TCP to remain attached to one material point; do not treat force ratios as a universal joint classifier.**
- **Do not silently modify camera extrinsics, use GT to choose a result, or replace the authoritative v2 joint via legacy FreeArtGS preprocessing.**
- **Do not certify simulator joint physics** from an observational approximation or guessed joint stops/mass/friction.
- **Do not claim success solely from synthetic passes or a better classification count** when real geometry/observability is still incorrect.

## Definition of done

ForceArtGS can tell apart (i) a coherent rotational velocity field around a finite, supported hinge, (ii) a spatially uniform translation, (iii) an observationally indistinguishable short arc/slide, and (iv) invalid single-joint/segmentation explanations. It retains both R/P candidates and all uncertainty, improves moving-support estimation, uses robust independent held-out visual evidence and conditional sliding TCP/force measurements, and hands **one authoritative, correctly qualified joint status** to the reconstruction and G8 asset-export pipeline. The five real scenes have reproducible diagnostics, including an explicit black-box failure or successful recovery analysis rather than a false prismatic label.
