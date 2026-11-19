# ForceArtGS — Multimodal Joint + Moving-Support Refinement With Unified TCP Sliding and Zero-Slip Contact

**Codex implementation contract**  
**Date:** 2026-11-18  
**Target repositories:** `SamueleMara/ForceArtGS` `main` at `50bfdb436b7fc62078f9946a6440ecec74259862`; submodule `reconstruction/nerfstudio` → `SamueleMara/FreeArtGS` `nerfstudio-local` at `e874a4e921cea289e08e972caaad5bb63bf9de7c`.  
**Priority:** build and validate the multimodal *pre-3DGS* frontend, then hand its fixed results into existing FreeArtGS stages. **Do not replace the renderer, rewrite Nerfstudio, or start with post-3DGS alternating training.**  
**Do not claim these changes are already implemented.** This is a specification for a new implementation.

---

## 0. One-sentence requirement

Implement **one** joint-aware RGB-D + tracking + robot TCP + optional force optimizer that jointly updates object moving support, material tracks, the revolute/prismatic hypotheses, joint geometry and `q_t`; make **zero TCP slip** a selectable special case of the existing **general sliding contact** formulation, with user-declared per-interval provenance; **only after this loop**, construct joint-conditioned datasets, train the **two part 3DGS models separately**, run the remaining **joint-constrained FreeArtGS** stages, and perform the existing strict PArtTwin handoff.

### Non-negotiable decisions

1. The **same candidate joint transform** `A_H(q_t)` and `canonical_contact_point()` are used in *all* contact modes; zero slip is **not** a separate estimator.
2. Default behavior is **`sliding`**: material contact coordinates may change over the moving object surface. Do **not** force `v_TCP = v_object` in sliding mode.
3. Provide **`stick_soft`** and **`stick_hard`**, configurable **globally and independently per uninterrupted contact interval**. A known no-slip demonstration is a legitimate *operator-specified prior* with explicit provenance. It does **not** require independent force confirmation merely to activate a TCP kinematic prior.
4. `stick_hard` means **one shared unknown canonical contact point per uninterrupted sticking interval**, not `sigma_slip=0` in an unconstrained least-squares denominator. A released/re-established grasp gets a **new anchor**.
5. Optionally allow `rigid_grasp` **only when the complete 6-DoF robot-to-object attachment is independently known**; it is distinct from a sticking *point* contact, which permits relative orientation.
6. **TCP trajectory must actively guide foreground moving-support discovery and joint optimization**. Merely using it to mask gripper pixels or writing `AVAILABLE_NOT_USED` is a failure.
7. **Force direction and force–TCP-motion alignment must be computed and used for grounded R/P discrimination**, but **not** by an unjustified universal rule `high alignment ⇒ prismatic`. Both R and P can have force aligned with motion. Condition likelihood on actual contact/slip mode, surface geometry, calibrated wrench, and candidate kinematics. When underdetermined, report a diagnostic and abstain from discriminatory force scoring.
8. `sliding` uses kinetic-friction *direction* only for demonstrable nonzero tangential slip; `stick_*` uses no such kinetic direction and instead applies static-contact constraints **only when the contact model warrants them**.
9. The visual-only path must work when TCP data are absent; the **TCP-only path must work when wrench calibration fails**. Missing force must not prevent a valid, user-confirmed `stick_hard` prior from helping joint fitting.
10. **No 3DGS optimization inside the frontend loop.** This loop uses original multiview RGB, metric depth with surface-aware uncertainty, object masks, tracked features/points, lightweight surfaces and robot signals. The first `ns-train` occurs only *after* the frontend output has been validated/frozen.
11. Physical camera extrinsics are fixed. TCP points, world points, joint axes, surfaces, force vectors and time indices use one checked world coordinate system; all outputs carry provenance, uncertainties and frame-mapping hashes.
12. Never infer a real physical joint from the temporary `working_motion_model`. Preserve v2 physical statuses and diagnostic R/P branches. No legacy ICP/reference-camera/rotation-span classifier may silently overwrite the new results.
13. The new frontend is **iterative**, with explicit acceptance validation and rollbacks when joint estimates, tracks or masks degrade. Track/segmentation refinement is not a one-shot post-processing step.
14. Do not generate false evidence from operator declarations, force thresholds, unknown pixels, nearby tool points, or fitting each hypothesis on a different advantageous subset.

---

## 1. Existing implementation: reuse rather than fork

Audit these actual files before changing code and record their interfaces in the report:

| Current file | Existing capability | Required alteration |
|---|---|---|
| `estimotion_2part/joint_discovery/tcp_contact.py` | `ContactRecord`, `tcp_velocity`, `contact_kinematics`, `canonical_contact_point`, surface support, contact residuals | Extend with typed modes, segments, canonical anchor/path state and mode-specific residuals; preserve the existing sliding API as an adapter |
| `estimotion_2part/joint_discovery/rigid_kinematics.py` | `A_joint`, `apply_joint_inverse`, `body_velocity`, `relative_slip`, `contact_jacobian` | Reuse as source of truth; add only missing vectorization/validated adjoints, not parallel motion equations |
| `estimotion_2part/joint_discovery/physical_selection.py` | `PhysicalResiduals` point-to-plane rows; `physical_stage` S3/S4 fit hook, `PhysicalDecision` | Refactor physical residual hook to accept segmented contact modes; run during frontend candidate refinement, not only isolated post-hoc stage |
| `estimotion_2part/joint_discovery/wrench_physics.py` | Signed force–TCP decomposition, `Q_H`, power terms, kinetic Coulomb direction, calibration gates | Add mode-aware sticking constraints and predictive force evidence, retain signed alignment diagnostics |
| `estimotion_2part/joint_discovery/wrench_calibration.py` | Frame/sign/tare/gravity/clock readiness | Continue strict force gates; do **not** make force calibration prerequisite for TCP position constraints |
| `estimotion_2part/joint_discovery/contact_annotations.py` | `ContactTimeline`, confirmed/probable/unknown/absent with provenance | Extend versioned operator-confirmed **contact mode intervals** and release/regrasp handling without turning heuristic sensor events into verified annotations |
| `estimotion_2part/joint_discovery/support_consistency.py` | Candidate-independent `S_core/M_proposal/U`, camera support and tool proximity | Add TCP contact-seed evidence, a robust moving-object/tool separation, and support updates within loop |
| `estimotion_2part/joint_discovery/dense_support.py` | Joint-specific multiview S/M/U reprojection, masks and tool capsule exclusion | Reuse the visibility/reprojection kernels; accept per-iteration candidate joint and TCP object-seed maps, not just final joint authority |
| `estimotion_2part/joint_discovery/motion_field.py`, `rotation_observability.py`, `selection_v2.py` | Hinge-free finite rigid motion, common axis, observational physical status | Recompute from optimized tracks/support, consume genuinely used TCP/force evidence and rerun selection after accepted iterations |
| `estimotion_2part/joint_discovery/geometry_fit.py` | R/P joint optimization with `extra` physical residual hook | Extend a shared, differentiable/jacobian-correct contact hook for sliding/sticking, optimize anchor nuisance parameters appropriately |
| `estimotion_2part/joint_discovery/observations.py` | Calibrated frames/partial tracks, tool filtering | Add RGB-driven local track recovery and contact-aware tool/object split; preserve raw observations separately |
| `estimotion_2part/joint_discovery/config.py` | Validated config and legacy fingerprint behavior | Versioned multimodal frontend/contact mode settings; preserve legacy config hash/replay when new mode off |
| `pipelines/joint_aware_reconstruction.py` | `discover → select → authority → dense → build → parts → blend → art → refine → compare → final → export` | Insert **`multimodal_refine` before authority/build/parts**, make `select` consume its final evidence, avoid running the old one-pass `dense` as authoritative |
| `preprocessing/prepare_joint_aware_multiview.py` | `T_WC` static, `A_H(q_t)⁻¹ T_WC` moving datasets | Consume new final authoritative results/masks; leave transforms and time indexing unchanged |
| Nerfstudio submodule `e874a4e` | Unknown-aware part supervision, authority-driven blending, frozen articulated joint | **Do not touch without a proven integration reason**; build the frontend to feed its current data contract |
| `tools/export_parttwin_handoff.py` | Handoff, physical-v2/diagnostic gates, standard PLY split | Carry mode/force/TCP provenance; no production export for unidentified joint or guessed physics |

### Existing reported gaps this must close

- `selection_v2.robot_evidence_v2()` returns `AVAILABLE_NOT_USED` or `ABSTAIN`: **this is not multimodal selection**.
- `physical_stage()` can refit candidates with contact/force, but it is **not called in the main v2 feedback loop**.
- `st_dense()` evaluates fixed-authority masks **once**, with robot information primarily acting as tool exclusion.
- `st_refine()` marks the post-3DGS G6 stage `NOT_RUN`; do not confuse that optional future renderer refinement with the required **pre-3DGS multimodal loop** in this specification.
- `PhysicalResiduals.contact_rows()` currently uses only `((A(q)⁻¹ p_TCP - a0)·n0)/sigma_contact`; it needs a mode-aware *generalization*.
- Kinetic-friction rows currently disappear when slip is small; in sticking mode use a physically meaningful and separately gated model rather than silently assigning a perfect force fit.

---

## 2. State, coordinate conventions and equations

For each candidate `H ∈ {P,R}` and model time `t`:

- `T_WC[c]`: fixed calibrated camera-to-world transform (OpenCV, metric).
- `T_WTCP[t]`: measured tool contact-frame pose; translation `p_TCP^W[t]` is calibrated to the **actual tool point**. Do not equate a generic EE-frame origin with a contact point without a known `T_EE_TCP`.
- `A_H(q_t)`: moving-part canonical-to-world transform, gauged `A_H(q_ref)=I`.
- `X_i^0`: canonical object material point, with persistent `track_id` and S/M/U responsibility.
- `S_M`: canonical moving object surface constructed **only from independent object RGB-D evidence** (not from a freely fitted TCP curve).
- `C_k`: an uninterrupted *confirmed contact* interval, with a chosen slip mode and provenance.
- `c_t^0 = A_H(q_t)⁻¹ p_TCP^W[t]`: canonical material coordinate instantaneously under the TCP.
- `v_M,H(p,t) = J_H(p) qdot_t` and `s_H,t = v_TCP(t) − v_M,H(p_TCP(t),t)`.

**Joint transformations** (reuse `rigid_kinematics.py`):

```
P: A_P(d) X = X + d u
R: A_R(theta) X = o + R_u(theta)(X - o)
```

**Unified contact constraint** for observations during contact:

```
p_TCP^W(t) = A_H(q_t) c_t^0 + measurement_noise
c_t^0 ∈ S_M                         [for geometrically supported surface contact]
```

`c_t^0` is *not* necessarily fixed: it is a state associated with the contact condition.

### 2.1 `sliding` — most general baseline

- `c_t^0` may travel tangentially over the object surface. A nonzero `c_(t+1)^0 − c_t^0` is **allowed**.
- Primary geometry is `d(c_t^0,S_M) ≈ 0`, with visibility and independently supported surface uncertainty.
- For locally smooth surface normal `n_t`, equivalent infinitesimal condition is `n_t·s_H,t ≈ 0`.
- Optional *nonzero-slip-permitting* smoothness of the canonical contact path may be used only as a calibrated, explicitly configured prior. **Do not minimize tangential slip magnitude by default.**
- Canonical contact point can change with time and contact region. Do not assume sticking just because a robot TCP and object edge appear close.

### 2.2 `stick_soft` — expected near-zero slip

Introduce one latent canonical contact anchor `c_*k` per uninterrupted segment `C_k`:

\[
L_{\mathrm{stick-soft},k} = \sum_{t\in C_k}
\rho\left(\big(c_t^0-c_{*k}\big)^\top
\Sigma_{\mathrm{stick},t}^{-1}\big(c_t^0-c_{*k}\big)\right).
\]

- `c_*k` is **jointly optimized** or analytically eliminated on training observations; never freely estimated using held-out observations during scoring.
- `Σ_stick` represents plausible *relative material slip and calibration error*, not arbitrary fit tolerance.
- This is a **prior**, not a declaration of exact attachment. Report inferred canonical drift and standardized prior violations.
- A segment's anchor can be loosely supported by observed surface geometry; when the contact patch is occluded, the user-confirmed sticking prior can still contribute a *kinematic* constraint, provided TCP registration is valid. Distinguish `kinematic_stick_valid` from `surface_supported`.

### 2.3 `stick_hard` — user-confirmed exactly fixed contact point

Enforce a single canonical point for the segment:

\[
\boxed{c_t^0=c_{*k}\quad \forall t\in C_k}
\qquad \Leftrightarrow \qquad
\boxed{p_{\mathrm{TCP}}^W(t)=A_H(q_t)c_{*k}+\epsilon_{\mathrm{TCP},t}}.
\]

- The constraint is **structural**: one shared unknown `c_*k`, not per-frame latent contact points tied together by a near-zero sigma.
- TCP measurements retain **finite position/camera/time calibration uncertainty**. Do not put an infinite weight on measurements.
- For confirmed hard sticking, `c_*k` may be estimated from TCP kinematics even when part of the physical contact surface is hidden. It is still essential to verify the contact belongs to the moving object and not to the gripper/static fixture.
- At release or regrasp, split `C_k` and introduce a **new** `c_*k`; no cross-gap equality, velocity regularization or friction term.
- If a hard-stick hypothesis has impossible point-motion residuals, flag `CONTACT_MODE_CONTRADICTED`; do not silently relax it or mask out bad intervals to manufacture a physically identified joint. An explicit new run with `stick_soft/sliding` must remain possible.

### 2.4 `rigid_grasp` — optional later extension

Only if `T_WTCP(t)` actually represents a calibrated rigid gripper attachment with a known constant offset relative to the moving object, use:

\[
T_{WTCP}(t)\approx A_H(q_t) T_{M,TCP,k}.
\]

Optimize **one** latent SE(3) transform per segment. This adds rotation constraints; ordinary `stick_hard` **does not** constrain TCP orientation. If there is no calibrated grasp pose, report `NOT_IMPLEMENTED/UNAVAILABLE` rather than approximating orientation constraints with position.

### 2.5 Finite differences and frames

For position-only sticking, the velocity identity is `v_TCP ≈ J_H(p_TCP) qdot`. However, **prefer exact finite SE(3) positional residuals** for fit/selection: a 15–40° hinge motion should not be represented solely by instantaneous small-angle velocities. Keep velocity rows as diagnostics or appropriately correlated terms, never independent duplicated evidence from the same positions.

For point/sliding contact, use instantaneous `s_H = v_TCP − J_H qdot`; ensure consistent frame and units. Check `p_TCP`, `v_TCP`, `qdot`, surface normal and applied force directions on each timestamp. Preserve `source_frame_index` mapping and time uncertainties.

---

## 3. Typed configuration and source provenance

Add a versioned *frontend* section and a single mode contract. The example below is a **proposed configuration API**, not an assertion that the CLI already supports it. Implement typed parsing, strict unknown-key rejection and a `--help`/schema validation test.

```yaml
joint_discovery:
  mode: calibrated_joint_aware
  comparison:
    selection: observable_physical_v2
  multimodal_frontend:
    enabled: true
    max_iterations: 6
    contact:
      enabled: true
      default_mode: sliding  # sliding | stick_soft | stick_hard | rigid_grasp
      require_operator_mode_confirmation: true
      # calibration-based uncertainties; the following are placeholders to calibrate,
      # NOT production-certified sensor noise values:
      sigma_tcp_position_m: null
      sigma_contact_surface_m: null
      sigma_stick_soft_m: null
      # Contact type and interval data are independent of R/P model predictions.
      intervals_file: configs/contact_intervals/example_scene.yaml
      # No force required for TCP-only confirmed contact/zero-slip.
    tcp_support:
      enabled: true
      tool_geometry_source: metadata/robot_tool.json
      seed_contact_object_surface: true
      exclude_occluding_robot_geometry: true
    force:
      enabled: true
      require_verified_wrench: true
      use_signed_alignment_diagnostic: true
      mode_aware_friction: true
    validation:
      use_frozen_temporal_folds: true
      compare_on_common_observations: true
      forbid_test_time_anchor_refit: true
```

Example independent contact-mode annotation file:

```yaml
schema: forceartgs.contact_segments/v1
frame_id: base_link
frame_index_type: source_frame_index
scene_id: example_scene
source: operator_confirmed_demo_protocol
notes: "Operator knows robot maintained one material point during the marked segment"
segments:
  - segment_id: grasp_A
    start_source_frame: 100
    end_source_frame: 140
    contact_state: contact_confirmed
    slip_mode: stick_hard
    provenance: operator_label
    known_contact_point_frame: robot_tcp
  - segment_id: pushed_surface_B
    start_source_frame: 145
    end_source_frame: 190
    contact_state: contact_confirmed
    slip_mode: sliding
    provenance: independent_video_annotation
  - segment_id: uncertain_C
    start_source_frame: 191
    end_source_frame: 201
    contact_state: contact_unknown
    slip_mode: sliding
    provenance: none
```

Rules:

- Segment endpoints must map **exactly** through `source_frame_index` and sorted `timestamp_s`; never infer indices by approximate filename offsets.
- Unknown contact is **not hard contact**. A verified user assertion for an entire demonstration may generate confirmed intervals as a **versioned operator annotation** with its own provenance; never rewrite `contact_probable` from a force heuristic as confirmed.
- `stick_hard`/`stick_soft` only active on independently asserted/confirmed contact frames and with a calibrated TCP point. `sliding` geometry also requires supported contact evidence for an equality-like surface constraint.
- For `contact_unknown`, TCP projection may contribute **low-confidence candidate proposals** but never a hard residual or an independent physical selection statistic.
- Unknown intervals between segments are not automatically bridged; do not interpolate a fixed anchor across no-contact gaps.
- Conflicting segment annotations are errors. Wrong frame order, insufficient samples, missing `T_EE_TCP`, inconsistent pose convention, nonmonotonic timestamps, or impossible TCP-position calibration are explicit errors.
- No silent fallback from `stick_hard` to `sliding`. Fail/abstain with diagnostics. Switching mode requires a separate recorded config version.
- If `rigid_grasp` is exposed before the orientation solver is implemented, reject it explicitly at parse time; never silently interpret it as `stick_hard`.
- `stick_soft` strength must accept a real covariance/uncertainty, not only an opaque weight. A missing required uncertainty yields `UNVERIFIED_PRIOR_SCALE`, preventing unwarranted physical certification while allowing explicit diagnostic sweeps.
- Include a `contact_mode_source` and SHA256 of annotations in the run manifest, output joint authority and export provenance.

---

## 4. Common contact data model and solver API

Use narrow, typed dataclasses; preserve compatibility with current `ContactRecord` and existing tests.

```python
from dataclasses import dataclass
from typing import Literal

SlipMode = Literal["sliding", "stick_soft", "stick_hard", "rigid_grasp"]

@dataclass(frozen=True)
class ContactSegment:
    segment_id: str
    model_time_indices: tuple[int, ...]
    source_frame_indices: tuple[int, ...]
    state: str                  # contact_confirmed | probable | unknown | no_contact
    slip_mode: SlipMode
    provenance: str             # operator_label | independent_video_annotation | ...
    annotation_sha256: str
    sigma_tcp_position_m: float | None
    sigma_contact_surface_m: float | None
    sigma_stick_soft_m: float | None
    # Optionally known attachment offset / full grasp transform, independently measured.

@dataclass
class ContactModelState:
    mode: SlipMode
    segment_id: str
    canonical_contact_path: object      # [K,3] for sliding, diagnostic for stick
    shared_anchor_M: object | None      # [3] one per stick segment
    surface_support: object | None      # from independent RGB-D object evidence
    active: object                     # valid fitting times; same across R/P comparisons
    residuals: dict
    status: str
    reasons: list[str]
```

Minimal module API (exact signatures may be adjusted after tracing existing call sites):

```python
def load_contact_segments(path, frame_map, tcp_record) -> list[ContactSegment]: ...

def compute_contact_state(candidate, tcp_record, segment, surface, *, frozen_anchor=None) -> ContactModelState: ...

def build_contact_fit_rows(candidate, tcp_record, segment, surface, config, *, train_indices): ...

def evaluate_contact_heldout(candidate, tcp_record, segment, surface, trained_anchor,
                             heldout_indices, config): ...

def contact_model_diagnostics(candidate, tcp_record, segments, states, config) -> dict: ...
```

**No duplicated parameterization.** The contact hook must use the same `axis_w`, `origin_w`, `q[t]` as `geometry_fit.fit_geometry()`; the support updater uses the same candidate object motion. `c_*k` is a nuisance variable that must be solved/marginalized consistently for each candidate on training data.

### 4.1 Fit rows by mode

**Sliding:**

```text
r_surface,t = signed_distance(A_H(q_t)^−1 p_TCP(t), observed canonical moving surface)
```

Only if the independently reconstructed surface supports that location, with uncertainty. For locally planar regions the existing `((c_t-a0)·n0)/sigma_contact` is an accepted adapter. Do not fit a fresh unconstrained plane for each hypothesis from its TCP path and then score the same path against that plane; this is vacuous.

**Soft sticking:**

```text
r_soft,t = L_stick,t @ (A_H(q_t)^−1 p_TCP(t) − c_*k)
```

Optionally retain a grounded surface residual. `c_*k` fitted only on training observations. Use proper covariance whitening and robust loss.

**Hard sticking:**

```text
p_TCP_pred(t) = A_H(q_t) c_*k
r_hard,t = L_TCP,t @ (p_TCP_obs(t) − p_TCP_pred(t))
```

Here one `c_*k` is shared structurally. Do not produce one per-frame anchor and constrain pairwise differences with a huge coefficient. `L_TCP,t` incorporates finite TCP pose uncertainty; anchor has **no** independent time-varying degrees of freedom.

For hard sticking with a strongly observed contact point on the moving surface, permit an *independent* surface attachment prior on `c_*k`. When no surface is observed, report that the joint is constrained by declared robot material attachment, not by visible contact geometry.

**Optional rigid grasp:** use the group residual `Log((A_H(q_t) T_MTCP,k)^−1 T_WTCP(t))` with correctly handled covariance on SE(3); only after the requisite transform is verified.

### 4.2 Identifiability and nuisance parameters

- A single sticking point does not necessarily identify the hinge line from a short arc. A remote revolute axis can still mimic translation. Keep spatial material tracks, whole-part geometry, independent attachment priors and R/P distinguishability checks.
- Position-only sticking cannot infer a full 6-DoF rigid attachment; do not use robot wrist orientation as object rotation evidence unless `rigid_grasp` is **explicitly verified**.
- Do not declare the physical type solely from a lower hard-stick TCP trajectory residual: both candidate R and P must be optimized fairly with the same allowed nuisance parameters, covariance and held-out data.
- For sliding, a completely free canonical path may explain both models. Its apparent fit alone must **not** count as discriminatory evidence without independent object-surface support.
- Record conditioning of joint geometry and `c_*k` (rank, covariance, trajectory extent, hinge-distance-to-object diagnostic). Avoid false certainty from long chains of correlated robot samples.

---

## 5. Contact-guided RGB-D moving-support and track refinement

Create `estimotion_2part/joint_discovery/multimodal_frontend.py` as the orchestration layer; factor image/track updates into dedicated modules rather than making this file a monolith.

### 5.1 A TCP seed is an object-contact *proposal*, not a gripper-mask label

For each observed `p_TCP^W(t)`:

1. Project to each fixed camera using the existing calibrated intrinsics/extrinsics.
2. Compute expected TCP depth, observed depth and visible object foreground mask. Account for uncertainty in `T_EE_TCP`, camera calibration, depth sensor edges and tool geometry.
3. Subtract/occlusion-mask genuine tool geometry from the *tool* side, but **do not erase the contacted object neighbourhood** just because its points are within a generic 3–5 cm radius. Use a measured tool model if available; a crude TCP–EE capsule stays a diagnostic approximation with uncertainty.
4. Where contact is independently confirmed and the object surface is visible at the expected contact, generate positive **moving-object** seeds in an uncertainty-scaled support region, not an arbitrary fixed pixel disk. On a sliding contact the seed location changes from frame to frame.
5. Where object/tool separation is ambiguous, mark `UNKNOWN` and preserve both interpretations for diagnostics; never claim an observed robot point is moving-object support merely because it follows TCP.
6. When `stick_*`, the inferred canonical anchor constrains where a consistent moving surface should project at **all contact frames**: seed recovery of that object region from frames where it is visible, including after prior occlusion. This is stronger than independent TCP proximity prompts.
7. At `no_contact`, do not create new positive contact seeds. The existing learned moving surface and material tracks may continue to move predictably.

Provide per-view debug overlays with robot projection, TCP tip, depth ordering, positive object seeds, unknown/tool pixels and expected joint flow. Record how many seeds were supported/rejected per camera/time and how many material tracks were recovered.

### 5.2 Track refinement, not only deleting bad tracks

- Preserve original `TrackObservations` with immutable IDs and observation provenance.
- Refine or extend moving material tracks using RGB patch/descriptor matches constrained by the **current exact R/P predicted reprojection**, including a search radius determined by motion and calibration uncertainty, not a universal pixel radius.
- Reproject a candidate canonical material point into each camera at subsequent times; use actual image appearance, visibility/occlusion and valid foreground mask. Apply depth only for consistent foreground surfaces. Do not sample depth across object boundaries and count background depth as moving-object evidence.
- Use camera-consistency diagnostics; a grazing-depth view (black box `cam_001/002`) must not override better RGB information (`cam_000`) solely through overconfident depth noise.
- Allow point identities with missing observations; do not make visibility itself evidence for static/moving status.
- Keep static, moving and outlier/unknown separate; do not force all unknowns into static after each iteration.
- Track evidence and contact evidence cannot use held-out observations for training. A discovered track on training frames is allowed as a training update; test frames stay genuinely withheld for scoring.

### 5.3 Dense S/M/U support update

Reuse `dense_support.py` geometry and visibility routines on the latest **candidate joint**, not only on the frozen `JointAuthority`. Combine:

```text
L_support = multiview RGB correspondence
          + valid metric depth/free-space evidence
          + rigid field consistency
          + current contact-derived object seeds
          + independent calibrated object silhouette
```

The output is `S/M/U + robot/occlusion` with confidence and provenance. Use candidate-independent **common evaluation support** to compare R/P; candidate-specific support can be updated for geometry fitting but must not allow one branch to win by discarding inconvenient pixels/tracks.

Recompute support after each accepted candidate update. Preserve per-round masks and scores for ablation and rollback. Do not train a Gaussian model in any of these rounds.

---

## 6. Mode-aware force and contact-direction evidence

### 6.1 Wrench and synchronization gates

For force to enter fitting or model selection, require the existing `WrenchCalibration.readiness()` checks for sensor rotation, sign, tare/gravity compensation, tool mass/bias assumptions, and clock registration. `source_frame_index` equality is not proof of force hardware synchronization. Retain raw/compensated wrench and diagnostic reasons.

**Decouple evidence availability**:

| Available evidence | Allowed behavior |
|---|---|
| RGB-D only | Visual joint/support loop |
| RGB-D + calibrated TCP, unknown contact | Tool handling and tentative contact proposals; no hard contact constraint |
| RGB-D + calibrated TCP + independently confirmed `sliding` | Surface-contact residual and time-varying moving seeds |
| RGB-D + calibrated TCP + operator-confirmed `stick_soft` | Shared-anchor soft prior, independent of force calibration |
| RGB-D + calibrated TCP + operator-confirmed `stick_hard` | Exact persistent *material-point* model with finite TCP observation covariance, independent of force calibration |
| Any of the above + **fully verified** wrench | Add mode-aware force terms on common validated contact frames |
| Unverified force | Report `ABSTAIN`, do not change joint parameters from force rows |

Do not let a force event alone establish a verified contact label. A direct operator declaration of no-slip contact is separate from a force measurement and is permitted when its provenance is recorded.

### 6.2 Force–TCP directional agreement: required diagnostic

Compute `wrench_physics.decompose_force` and report, by segment and hypothesis:

\[
\rho_{\parallel,t}^{\mathrm{signed}} =
\frac{f_t\cdot v_{TCP,t}}{\|f_t\|\,\|v_{TCP,t}\|},\qquad
\rho_{\perp,t}=\sqrt{1-(\rho_{\parallel,t}^{\mathrm{signed}})^2}.
\]

Also record force direction relative to predicted object motion at contact:

\[
\cos\phi_{H,t} = \frac{f_t\cdot v_{M,H}(p_{TCP,t},t)}{\|f_t\|\|v_{M,H}\|},
\qquad Q_H=f_t\cdot J_H(p_{TCP,t}).
\]

For `P`, `Q_P = f·u_P` (N); for `R`, `Q_R = u_R·((p_TCP-o_R)×f)` (N·m). Their **raw magnitudes cannot be compared across types** because the units differ. Report joint power `Q_H qdot` (watts), contact slip power and sign. Preserve pushing/pulling direction and movement reversal; do not take absolute cosine before computing signed diagnostics.

**Required interpretation:** sustained positive alignment along TCP motion can be **positive support** for an actuated prismatic slider *if* the predicted prismatic object velocity, observed common translation, contact model, guide/attachment evidence and force dynamics agree. But a robot can apply a tangential force to open a revolute lid too. Thus a scalar alignment ratio is not a standalone physical likelihood or classifier. Implement an explicit calibrated/assumed mechanical model for candidate comparison; if neither model predicts a distinguishable force direction, record `FORCE_NONDISCRIMINATIVE` and do not alter the physical type.

### 6.3 `sliding`: kinetic friction only when slip is measurable

`slip = v_TCP − v_M,H`, `s_T = slip − n(n·slip)`.

On verified maintained surface contact and measurably nonzero tangential slip, the existing single-contact Coulomb direction model may be used:

\[
\hat f_H\propto -n_{out,H}+\mu_k\hat s_{T,H}.
\]

- Use a **bounded shared friction parameter** per physical contact/material interval, fitted on training observations only. Do not fit an arbitrary `mu_t` each frame or change force sign to favor one hypothesis.
- Compare signed **full force directions** for R and P over the same valid samples and supported contact patches; validate against time-shift/sign-reversed/shuffled-force nulls.
- If no normal, no verified contact or no measurable slip, the *kinetic direction* evidence is `ABSTAIN`; it is not a zero-error match.
- A point-contact force model may fail under a real multi-contact gripper; flag `CONTACT_MODEL_INADEQUATE` when the force direction is unexplained by both candidates.

### 6.4 `stick_soft` / `stick_hard`: static contact, no kinetic slip direction

For a **verified unilateral point contact** with an independently known surface normal, static tangential force is constrained by a friction cone:

\[
f_N\ge0,\qquad\|f_T\|\le\mu_s f_N
\]

with correct robot-on-object force sign and bounded friction uncertainty. This tests **feasibility**, not a uniquely predicted direction. The contact may also be bonded/grasped (including tension or multiple contact patches), in which case a single unilateral Coulomb cone is **not** applicable: use a justified grasp/wrench model or abstain from that constraint.

- `stick_hard` is enforced through **TCP position/material attachment**, not by asserting that the wrench must align with the TCP velocity.
- The measured force still contributes `Q_H qdot`, direction-of-work and mechanical consistency under a supported actuation/resistance model. It may help discriminate if R and P have different predicted moment arms, surface normals, feasible contact wrenches or learned dynamics **with shared calibrated parameters**.
- Force alignment is often high for **both** a tangent-pushed hinge and a slider with sticking TCP contact. If the difference is not identifiable, report the alignment and retain ambiguous physical type.
- At zero slip, kinetic-friction direction `s_T/||s_T||` is undefined: **never return a zero force residual as a perfect match**. Use separately gated static-contact feasibility or `ABSTAIN`.
- If gripper grasps multiple surfaces, measured wrist wrench may contain a free couple. Do not treat `p_TCP×f` as the entire applied joint torque unless contact model and torque origin justify that simplification; include measured torque only after verified torque-origin calibration.

### 6.5 Force scores must actually influence joint refinement when valid

Refactor/reuse existing `PhysicalResiduals` / Stage S3/S4 hook:

- Separate `L_contact` (mode-aware kinematic constraints), `L_force` (mode-aware force likelihood/feasibility) and `L_visual`.
- Fit candidates on identical training folds and common validated contact time indices; do not use test force to fit shared `mu` or contact anchors.
- Every accepted iteration reports parameter deltas due to TCP and due to force separately: axis/origin/q changes, moving support changes and held-out score changes.
- Force may upgrade/reject a physical hypothesis **only** if the model is independently valid, predictive separation exceeds calibrated uncertainty and negative/null controls fail to reproduce the improvement. Otherwise diagnostic only.
- Do not conflate the energy-balance identity `f·v_TCP = Q_H qdot + f·slip` with a model preference; it holds algebraically for *both* joint types.

---

## 7. Iterative multimodal frontend: required controller

**Create a new stage** between `discover` and the final `authority/build/parts` stages. The joint fitting and dense support must be coupled *here*, before Gaussian training.

```text
RAW multiview RGB-D + object masks + material tracks + TCP trajectory + contact intervals + force
          |
          v
  (0) candidate-independent tracks/S_core/M_proposal/U; TCP object-surface seeds
          |
          v
  initial R and P fits (existing geometry_fit + hinge-free observability)
          |
          v
  ┌──────────────────────────────────────────────────────────┐
  |  iteration k, both candidates, same training evidence    |
  | A. Update object RGB material correspondences / coverage |
  | B. Update static/moving/unknown + tool support           |
  | C. Fit R/P axis, pivot and q_t using visual evidence     |
  | D. Add mode-aware TCP contact rows and fit anchors       |
  | E. Add *verified*, mode-aware force rows if available    |
  | F. Re-assess motion coherence, coverage, uncertainty     |
  | G. Compare on frozen, genuinely held-out time/camera     |
  | H. Accept bounded improvement or roll back this round    |
  └──────────────────────────────────────────────────────────┘
          |
          v
  finalize authoritative R/P candidates + physical status + final S/M/U masks
          |
          v
  build per-part datasets, exact calibrated/virtual cameras
          |
          v
  train two separate standard 3DGS part models (existing Nerfstudio)
          |
          v
  joint-constrained twopart-blend → splatfacto-art → G8 handoff
```

### 7.1 Optimization state

For each hypothesis `H`, maintain:

```text
H, joint axis u_H, hinge origin o_H if R, q_H[0:T]
X_i^0 (material anchors), track observations/provenance
P_i(S), P_i(M), P_i(U/outlier)
canonical moving/static lightweight RGB-D surfaces
contact segment modes, canonical paths or anchors c_*k
calibrated force model parameters (only if identifiable)
per-source uncertainty, confidence and held-out predictive score
```

The time/camera geometry and observed TCP pose are **measured inputs**, not freely optimized into a convenient solution. A bounded calibration nuisance can be introduced only as a separate explicitly identifiable problem with independent evidence; do not silently move the cameras or robot base.

### 7.2 Coupled objective

Proposed block-coordinate objective (normalize terms by calibrated uncertainties and actual effective independent sample counts):

\[
\begin{aligned}
L_H ={}& L_{RGB\,track}+L_{RGBD\,surface}+L_{S/M/U}
        +L_{1DoF\,rigid}\\
       &+L_{TCP\,contact}(H,\text{mode})
        +L_{force}(H,\text{mode, verified})
        +L_{priors}.
\end{aligned}
\]

- `L_TCP` operates on **contact positions/paths**, not on force alignment alone.
- The `stick_soft` prior and `stick_hard` positional rows are *mutually exclusive primary contact parameterizations* for their segment. Do not count both as independent observations.
- Penalize model complexity and invalid moving-support coverage; never win by marking more observations `UNKNOWN`.
- Robust losses must not allow a strong operator-verified hard-stick segment to be silently discarded. If it is contradicted, record the contradiction and preserve the raw residuals.
- Keep the RGB tracker and geometric residuals independent of **force** measurements to enable meaningful ablations.

### 7.3 Candidate comparison / status hierarchy

Retain `PhysicalDecision` statuses:

```
IDENTIFIED_REVOLUTE
IDENTIFIED_PRISMATIC
TYPE_UNOBSERVABLE
MODEL_INCONSISTENT
NO_SIGNIFICANT_MOTION
```

Also report:

```
working_motion_model
working_model_is_approximation
joint_type_identified
joint_geometry_validated
contact_assumption_satisfied
force_evidence_used
physics_joint_export_ready
```

A fixed TCP point with an almost straight short arc does **not** prove P if a plausible remote R can imitate it. Assess motion excitation, contact-point trajectory span, R/P predicted separation in world/pixel space, moving material field and physically plausible attachment region. Refit anchors and q independently for both candidates on training folds. If R and P remain equivalent, retain `TYPE_UNOBSERVABLE` and the simpler working approximation separately.

**No physical-certification shortcut:** user-confirmed `stick_hard` confirms *contact mode*, not automatically joint type, hinge location or dynamics. Force-alignment evidence contributes only within an identifiable physical model.

### 7.4 Acceptance, stopping and rollback

Per iteration save candidate-state snapshot (hashes), track count/support coverage, S/M/U agreement, contact-mode residuals, measured force directional fit, q range, hinge covariance/conditioning, and held-out visual/TCP/force predictive metrics. Accept updates only with nonworsening hard validity constraints and a *meaningful* improvement on common held-out evidence. Stop at convergence or a declared max iteration count. Use deterministic seeds and no silent warm-start from untracked legacy models.

An accepted improvement on training loss alone is insufficient if it degrades held-out object motion, drops support, contradicts known stick or creates a remote-hinge degeneracy. Persist the **last accepted** state; do not keep a failed last optimizer step.

### 7.5 Train/evaluation leakage policy

- Declare time/camera/surface folds before fitting R/P and before support refinement.
- Canonical anchor `c_*k` and force/friction parameters learned on training intervals **never** refit on withheld data.
- For a segment that contains only held-out times, **do not fabricate a trained sticking anchor**. Score status `INSUFFICIENT_TRAIN_CONTACT`.
- For test frames, `q_t` must be predicted from training-time trajectory or other explicitly permitted independent signals; report whether the evaluation is interpolation or extrapolation. Do not use those frame tracks or TCP measurements to fit `q_t` and then claim them as held-out evidence.
- For an endpoint, validate non-extrapolation if available and report physical prediction separately.
- Dynamic moving-support proposals selected on training data only; do not select the most helpful cameras/frames from test data after checking GT.

---

## 8. Pipeline integration and exports

### 8.1 Controller changes

Target `pipelines/joint_aware_reconstruction.py` (currently 914 lines). Introduce `st_multimodal_refine()` after the raw `discover` stage; run `select-v2` on **refined**, not raw, tracks/support and candidate fits. Suggested stage ordering:

```python
STAGES = (
    "audit", "discover", "multimodal_refine", "select", "authority",
    "dense", "build", "parts", "blend", "art", "refine",
    "compare", "video", "final", "export"
)
```

`st_dense` should **consume final pre-3DGS masks** and may format/revalidate them; it must not silently overwrite them with a one-pass result based on an old authority. If the new frontend already writes complete masks, `dense` becomes a validation/packaging stage for joint-aware training. If per-candidate final S/M/U masks require extrapolation after the final joint change, do one explicitly recorded final update and consistency check.

`st_authority` must read the final candidate joint geometry and q from `multimodal_refine/accepted`, preserving the candidate-specific and physical status and referenced provenance hash. The old `authority_from_discovery()` is allowed only in explicit legacy mode.

`st_parts`, `st_blend`, `st_art` remain substantially **unchanged**. The frozen authority and joint-aware Nerfstudio training contracts stay authoritative. `refine` remains `NOT_RUN` if post-Gaussian alternation is not implemented; do not rename the new pre-3DGS loop as proof that G6 full-render alternation is complete.

`--selection v2` should select the **new multimodal physical decision** when the frontend is enabled. Retain a fully reproducible visual-only v2 ablation. No auto fallback to the old legacy classifier if robot data are missing.

### 8.2 CLI proposals

Implement actual flags (names may be adjusted to avoid existing collisions; make `--help` authoritative):

```bash
# Syntax examples AFTER the new implementation; do not claim they work before code exists.
python pipelines/joint_aware_reconstruction.py \
  --scene /path/to/real_scene \
  --frames 70:128:2 --cameras cam_000 cam_001 cam_002 \
  --output-root /media/.../forceartgs_joint_aware \
  --stage audit,discover,multimodal_refine,select,authority,dense,build \
  --discovery-config configs/joint_discovery.real_tcp2cm.yaml \
  --robot-archive /path/to/robot_archive.npz \
  --contact-segments configs/contact_intervals/scene.yaml \
  --multimodal-frontend
```

Test standalone command for faster iteration (create or extend `joint_discovery.cli`):

```bash
python -m estimotion_2part.joint_discovery.cli multimodal-refine \
  --scene /path/to/real_scene \
  --from-run /path/to/vision_discovery \
  --robot-archive /path/to/robot_archive.npz \
  --contact-segments configs/contact_intervals/scene.yaml \
  --output /path/to/new_output
```

The standalone command must **not** mutate a prior discovery run, legacy outputs, or test results in place. Honor the current SSD output and fingerprint policies.

### 8.3 Output contract

```text
<run>/multimodal_refine/
  config_resolved.yaml
  source_fingerprints.json
  contact_mode_segments.json
  contact_mode_validation.json
  robot/
    tcp_trajectory_world.npz
    contact_seed_overlays/...
    force_alignment_per_frame.csv
    force_calibration_gates.json
  iterations/
    round_000/{revolute,prismatic}/...
    round_001/{revolute,prismatic}/...
    ...
  accepted/
    tracks_refined.npz
    candidate_support.npz
    joint_candidates.json
    joint_states.npz
    joint_selection_v2.json
    contact_states.npz
    contact_residuals_per_segment.csv
    force_candidate_comparison.json
    model_comparison.json
    masks/
      mask_0/...          # static
      mask_1/...          # moving
      mask_unknown/...
      ignore_robot/...
    canonical_init/
      static_points.ply   # RGB-D point-cloud initialization, NOT trained Gaussians
      moving_points.ply
    authority_source_manifest.json
    validation_report.json
  report.md
```

For physical selection output, record **both** `contact_mode_assumption` and `contact_mode_evidence` (verified operator annotation vs measured residual consistency); force use separate. Report hard-stick anchor `c_*k`, its fit covariance/conditioning and residuals for *every* segment, plus test-fold held-out metrics. If exact contact is contradicted, keep the bad residual and failure code, never pretend `sliding` succeeded unless a separate explicit run was requested.

### 8.4 Downstream FreeArtGS

Continue existing:

1. `prepare_joint_aware_multiview.py` — exact static physical and moving virtual cameras; no ICP, no per-camera joint poses.
2. `reconstruction/generate_ply.py` — part PLY initialization from refined masks and valid metric depth.
3. `ns-train splatfacto-depth` — static and moving part reconstruction **separately**.
4. `twopart-blend --joint-aware --obj-pose-opt-enabled False` — shared joint poses, trainable part weights as currently implemented.
5. `splatfacto-art` — authoritative joint frozen during final appearance training; no legacy `estimate_joint.py`.
6. `tools/export_parttwin_handoff.py` / PArtTwin1D — two standard 3DGS PLYs with canonical joint/frames and physics/readiness provenance; physical asset only if independently ready.

Do not change 3DGS PLY semantics; no Gaussian rendering is required in the earlier multimodal frontend. Post-3DGS G6 renderer refinement can be a future bounded, separately validated extension, **not** a prerequisite for this task.

---

## 9. Mandatory tests: execute before claiming integration

Add dedicated tests under `tests/` while preserving the existing 143 joint tests and 11 joint-aware integration tests; rerun them. Assertions below define required **behavior**, not predetermined physical truth for real scenes.

### A. Analytic contact-mode kinematics

- **A01 — P, sliding:** a known sliding contact has nonconstant `c_t^0` along the canonical surface; zero normal-contact residual; nonzero tangential slip accepted.
- **A02 — R, sliding:** opening hinge with a TCP sliding on the lid; correct model has supported changing `c_t^0`; incorrectly sticking that same path is contradicted.
- **A03 — P, hard sticking:** exact straight TCP trajectory from `d_t u` and one canonical anchor; recover axis and `q` to synthetic noise tolerance; no per-frame anchors.
- **A04 — R, hard sticking:** exact finite arc about a known hinge, including 15–40° range; recover axis, hinge within observable limits and `q` without linearizing motion.
- **A05 — Soft sticking:** small nonzero canonical drift accepted with a gradual uncertainty-dependent loss; hard sticking flags a meaningful contradiction under the same data.
- **A06 — Segment restart:** release + regrasp yields two independent anchors; a single cross-gap anchor fits badly and is never used.
- **A07 — Point-contact orientation:** arbitrary TCP/tool rotation around the attached point does not violate `stick_hard`; it should violate an explicitly valid `rigid_grasp` prior.
- **A08 — Unknown contact:** no hard/soft contact residual, even if force event/proximity is high; operator-confirmed protocol activates correct segments.
- **A09 — Missing/inconsistent TCP calibration/time:** fail closed; contact fitting disabled with precise reason; vision-only still runs.
- **A10 — Hard stick numerical stability:** `sigma_slip=0` is never used; finite TCP measurement noise and one shared structural anchor produce finite gradients and stable covariance.
- **A11 — Fixed-anchor gauge:** re-gauging `q_ref` and transforming canonical anchor consistently leaves world predictions unchanged.
- **A12 — Short trajectory degeneracy:** even with perfect `stick_hard` a near-straight arc with remote hinge cannot be falsely certified R/P without sufficient distinguishability.

### B. Physically correct force behavior

- **B01 — Sliding kinetic friction:** predicted full force direction changes with signed tangential slip and plausible shared `mu`; forward and backward slide tested.
- **B02 — Zero-slip static contact:** kinetic Coulomb force **direction** is undefined/disabled, not zero residual; static-contact feasibility used only when validated as a unilateral point contact.
- **B03 — Grasp with multiple contacts/tension:** a single-point unilateral cone abstains rather than rejecting a legitimate grasp.
- **B04 — Alignment diagnostic:** both a tangentially pushed hinge and a drawer can have high `rho_parallel`; high alignment alone never certifies P.
- **B05 — Positive prismatic evidence:** with a physically specified resistance/contact model, the signed force direction and generalized power favour P in an informative synthetic slider; removing the physical model makes the evidence non-discriminative.
- **B06 — Correct torque/force handling:** `Q_P` in N and `Q_R` in N·m; compare predicted powers/likelihoods, not raw generalized-force magnitudes.
- **B07 — Calibration abstention:** wrong sensor sign/rotation, unverified gravity/tare/sync or torque origin causes the appropriate force term to abstain **without disabling valid TCP sticking constraints**.
- **B08 — Force nulls:** shuffled-time, sign-reversed, gravity-only and random force direction must not yield an artificial type improvement.
- **B09 — Zero slip but real contact force:** `L_force` does not silently become zero just because `s_T=0`; uses supported static-contact evidence or returns `ABSTAIN`.
- **B10 — Evidence accounting:** same force frames, surface provenance and calibrated parameters for R/P, with held-out friction/anchor parameters frozen.

### C. Joint/segmentation/TCP loop

- **C01 — Contact seeds:** confirmed TCP on moving object generates positive object-surface seeds; tool occupancy never becomes moving-object support.
- **C02 — Contact at surface edge:** depth-edge/background depth must not poison object material-point reconstruction; preserve valid RGB tracking.
- **C03 — Gripper near real object contact:** exclude geometrically supported tool pixels but retain object pixels nearby; proximity alone never discards the object.
- **C04 — Black-box-like synthetic corruption:** one camera accurately observes lid rotation, other views have lagged RGB tracks and grazing depth. Multimodal iteration improves moving support and recovers hinge/rotation relative to one-pass baseline, without oracle view selection.
- **C05 — Drawer-like hidden guide:** strong coherent P motion + confirmed hard stick and informative force can contribute to an identified P result **without requiring visually exposed rails**, provided R alternatives are observably excluded.
- **C06 — Realistic sliding lid:** incorrect `stick_hard` is detected rather than forcing the hinge to shift or turning the lid into an implausible slider; `sliding` remains a valid run when explicitly selected.
- **C07 — Dynamic contact point:** TCP positive seed follows changed surface contact across time; no fixed grasp assumption in default sliding mode.
- **C08 — Candidate support fairness:** both models compared on the same withheld material tracks, RGB-D observations and contact frames; an R or P branch cannot improve by deleting its difficult pixels.
- **C09 — Iteration and rollback:** an accepted round improves held-out evidence; a worse round is rolled back, with artifacts and hashes retained.
- **C10 — Unknown and occlusion:** unobserved/occluded foreground not forced static or labeled background; unknown regions produce no false negative part-supervision gradient.
- **C11 — Robot absent:** original v2 visual-only path remains usable and does not invent TCP contact.
- **C12 — RGB-D + TCP, force invalid:** TCP-driven fit changes candidate/support vs visual baseline; verified force path abstains; no hidden force use.

### D. Integration, leak prevention, provenance

- **D01 — Held-out contact:** anchor and `mu` estimated on training only; changing held-out measurements does not change fit outputs; predictive scores change normally.
- **D02 — Authority handoff:** accepted refined `axis/origin/q/frames` are the authority that datasets and blend consume; no stale discovery authority.
- **D03 — Exact virtual camera equivalence:** `T_M0,C(t)=A_H(q_t)^−1T_WC(c)` in fixed metric coordinates for both R and P.
- **D04 — Part training order:** no `ns-train` until multimodal frontend has completed and its outputs are accepted.
- **D05 — Existing FreeArtGS tripwires:** legacy `estimate_joint.py`, per-camera ICP and free object-pose adjustment never execute in joint-aware mode.
- **D06 — Status semantics:** assumption `stick_hard` ≠ physical joint identified; ambiguous branch remains diagnostic; invalid dynamics ≠ simulation-ready.
- **D07 — Manifest fingerprints:** contact annotations, robot geometry, wrench calibration, camera calibrations, source observations, part masks and submodule SHAs recorded; stale resumes refused.
- **D08 — CLI config:** default `sliding`, per-interval soft/hard overrides, precise validation errors, explicit unavailable `rigid_grasp`, backward-compatible legacy config hashes.
- **D09 — Export provenance:** downstream PArtTwin handoff references contact mode and force utilization, and refuses production physics where readiness is lacking.
- **D10 — Clean clone:** fresh submodule checkout at `e874a4e` is sufficient for separate part training, no local uncommitted modifications required.

### E. Required real-scene experiments and ablations

Run **short frontend-only pilots first**, without complete 3DGS training. Do **not** add false known-contact annotations to historical datasets. Use verified operator metadata only if available; otherwise run TCP trajectory / low-confidence contact proposals with strict gating and report the limitations.

| Scene | Diagnostic objective |
|---|---|
| White drawer | Check true linear motion support; compare sliding vs **operator-confirmed** sticking if the demonstration really maintained one grasp point; report force–TCP alignment and calibrated force discriminability separately |
| Black box | Recover rotating lid support (especially compare `cam_000` RGB against contaminated grazing depth on `cam_001/002`); compare R/P and contact assumptions without baking in the known evaluation label |
| Red fridge | Maintain reliable ~17–19° rotation while checking joint/pivot accuracy and whether TCP/force add independent evidence |
| Robot dog | Distinguish actual object material points from robot geometry using depth-aware tool occupancy and contact seeds, not a blanket 5 cm exclusion |
| Orange suitcase | Recover real lid support and detect multiple rigid motions or unreliable contact; no forced R/P classification |

Ablations, using the **same clips, folds, initial tracks, cameras and compute budget**:

```text
V0 = RGB-D only; original v2 one-pass
V1 = RGB-D only; new iterative tracks + S/M/U + joint
V2 = V1 + TCP position + sliding contact
V3 = V1 + TCP position + stick_soft (only on verified matching episode)
V4 = V1 + TCP position + stick_hard (only when independently known true)
V5 = V2/V3/V4 + calibrated force (only where its gates pass)
V6 = same as V5 with force randomized / time shifted (negative control)
```

Do **not** compare V3/V4 against V2 as a universal benchmark if they assume different actual contact physics. For synthetic experiments, where the true contact mode is known, compare correct and intentionally wrong mode priors to calibrate failure signals. For real data, separate recorded operator knowledge from inferred quantities.

Metrics: physical R/P status vs reference where known; axis angular error; hinge-line distance/uncertainty; `q_t` trajectory accuracy; RGB reprojection; valid-depth residual; moving segmentation precision/recall/coverage on *independently annotated* regions when available; material-track recovery; tool false-positive rate; moving/unknown ratio; contact anchor drift and violations; signed force alignment distribution; normalized force directional likelihood; real wall-time and memory. Use CI over independent time blocks/surfaces, not over individual correlated pixels.

---

## 10. Implementation work plan / acceptance gates

**Gate 0 — Clean audit and baseline.** Check worktrees/SHAs/submodule, read current code and report actual command signatures; run existing tests. No hidden changes to `FreeArtGS` training code, upstream submodule or historic outputs.

**Gate 1 — Unified contact-mode schema and kinematics.** Add typed config, segment annotations, shared canonical contact state and exact `sliding`/`stick_soft`/`stick_hard` residuals. Run analytic tests A01–A12. Preserve existing sliding API behavior under default mode.

**Gate 2 — Mode-aware force.** Add signed force/TCP diagnostics, proper static-vs-kinetic terms, contact/wrench gating and discriminatory model evidence with negative controls. Run B01–B10. TCP-only hard sticking must work without any force input.

**Gate 3 — TCP-guided tracks and S/M/U.** Implement object-surface seeding, tool separation, RGB track recovery, candidate-independent support and candidate-conditioned iterative masks; save intermediate overlays. Run C01–C03, C07–C11.

**Gate 4 — Iterative R/P + TCP + force frontend.** Reuse existing geometry/selection code; new controller performs multiple validated rounds, matched R/P comparisons and rollback. Run C04–C06, C09/C12 and D01–D06.

**Gate 5 — Handoff to existing FreeArtGS.** Final authoritative joint and masks feed `prepare_joint_aware_multiview`, `generate_ply`, two separate `splatfacto-depth` trainings, fixed-joint blend and art, then G8. No changes to legacy classifiers. Run D02–D10 and a small synthetic smoke build.

**Gate 6 — Real data and ablations.** Run frontend-only pilots, then a small whole-pipeline red-fridge + black-box experiment after demonstrated quantitative frontend improvement. Measure failure modes; never claim calibration or physics evidence that is unavailable.

### Definition of done

This task is complete only when:

1. All three contact modes run through **one** joint fitter; `stick_hard` recovers a persistent material point rather than approximating it with an infinite penalty.
2. Default sliding still supports changing TCP contact points, verified surface constraints and appropriate friction when observable.
3. Joint estimation **and** moving-support masks are demonstrably updated by the TCP in the pre-3DGS loop; force also changes optimization/selection **when independently calibrated and informative**, with exact contribution reported.
4. Zero-slip TCP data constrain R/P even when force is unavailable; invalid force safely abstains.
5. RGB material tracking and S/M/U segmentation genuinely iterate with joint geometry, with held-out acceptance and rollback.
6. The two separate 3DGS trainings occur **after** convergence and consume the frozen accepted joint/masks; Nerfstudio authority checks pass.
7. Existing legacy tests and all new tests pass, and at least one controlled synthetic test demonstrates recovery beyond the original one-pass baseline without ground-truth leakage.
8. Real-scene reports clearly separate observation-based improvement, user-declared contact assumptions and unavailable force evidence. No unverified URDF-ready claims.
9. The documentation gives working CLI/config examples, exact output files, per-stage provenance and remaining limitations.

---

## 11. Codex deliverables and commit protocol

Create/update as needed (do not build duplicate parallel implementations):

```text
estimotion_2part/joint_discovery/contact_modes.py          # NEW: segments, modes, anchors, residuals
estimotion_2part/joint_discovery/multimodal_frontend.py     # NEW: iterative orchestration
estimotion_2part/joint_discovery/tcp_support.py             # NEW: object contact proposals / tool-aware seeds
estimotion_2part/joint_discovery/track_refinement.py        # NEW: RGB/material track updates
estimotion_2part/joint_discovery/tcp_contact.py             # reuse/wrap existing sliding primitives
estimotion_2part/joint_discovery/physical_selection.py      # use mode-aware residuals in fit/selection
estimotion_2part/joint_discovery/wrench_physics.py          # mode-aware force, static/kinetic split
estimotion_2part/joint_discovery/contact_annotations.py    # typed contact mode provenance
estimotion_2part/joint_discovery/config.py                   # validated new config
estimotion_2part/joint_discovery/selection_v2.py             # consume actual multimodal fit
estimotion_2part/joint_discovery/support_consistency.py      # loop updates
estimotion_2part/joint_discovery/dense_support.py            # loop updates + TCP seeds
estimotion_2part/joint_discovery/cli.py                      # standalone CLI
pipelines/joint_aware_reconstruction.py                      # pre-3DGS stage ordering
configs/joint_discovery_multimodal.example.yaml
configs/contact_intervals/EXAMPLE.yaml

# New tests (split/rename thoughtfully, keeping existing suites green):
tests/test_joint_contact_modes.py
tests/test_joint_sticking_force.py
tests/test_joint_tcp_support_refinement.py
tests/test_joint_multimodal_frontend.py
tests/test_joint_multimodal_integration.py

docs/MULTIMODAL_TCP_CONTACT_SLIP_IMPLEMENTATION_REPORT.md
docs/MULTIMODAL_TCP_CONTACT_SLIP_DESIGN.md
```

The report **must** include:

- parent SHA, submodule SHA, working tree status, exact changed call graph, entry points and parser flags;
- equations **as implemented** and whether contact anchors are jointly fit or analytically eliminated;
- configuration and annotation provenance; examples for general sliding, soft sticking, exact sticking and release/regrasp;
- zero-slip residuals and canonical anchor drift before/after fitting; mode conflicts and how they are surfaced;
- separate visual/TCP/force objective terms and how each affects joint parameters, `q_t`, track recovery and moving masks;
- held-out folds, uncertainty handling, sample independence and negative controls;
- tested synthetic R/P outcomes and real-scene frontend comparison tables, including the force-abstain cases;
- confirmation that two 3DGS parts are trained **after** frontend convergence and that blend/art respect the final authority;
- complete test commands, test outcomes, runtime, failing/skipped tests with reasons, and remaining algorithmic gaps;
- producer/export provenance and explicit distinction between physical status, work model and physics-ready asset.

**Git protocol:** prefer a new feature branch for this significant change. Do not edit/delete historic results or overwrite legacy assets. Commit the new ForceArtGS frontend and tests. Only modify the Nerfstudio submodule if strictly necessary; if modified, commit/push it **first** and then commit the parent submodule pointer. Before claiming reproducibility, verify both worktrees clean and a fresh clone can run the tested stage.

---

## 12. Strict failure conditions

- Never select R/P directly from a threshold on `|f·v_TCP|/(|f||v_TCP|)`.
- Never use the kinetic Coulomb slip-direction term for sticking, or treat undefined directions as an ideal fit.
- Never assume `stick_hard` means fixed TCP **orientation** unless `rigid_grasp` is independently justified.
- Never estimate a separate free contact anchor for each hard-sticking frame.
- Never set sigma to zero/infinity as a numerical surrogate for structural hard sticking.
- Never let candidate-dependent contact-surface fitting make an arbitrary TCP curve appear perfectly explained by both models.
- Never suppress real object points because they are near the robot tool; check measured tool occupancy/occlusion.
- Never require measured force calibration before using a calibrated TCP trajectory with user-verified no-slip contact.
- Never turn an operator-confirmed contact *mode* into an automatic declaration of joint identity, contact point ground truth or dynamics.
- Never allow the joint to be selected before the final accepted RGB/TCP/force/support loop and then frozen without reconsidering its evidence.
- Never perform full 3DGS training within the multimodal estimation loop, or replace separate part training with one fused model.
- Never certify a simulator asset when the type or geometry is unidentifiable or collision/dynamics inputs are missing.

**Final guiding principle:** generalized **sliding-contact kinematics is the parent model**, and a user-confirmed **zero-slip demonstration is its exact or soft fixed-material-point specialization**. Both must **actively** improve joint estimation and static/moving support **before** the existing FreeArtGS 3DGS stages.
