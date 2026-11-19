# Multimodal pre-3DGS frontend: TCP contact modes, force and support refinement

Spec: `FORCEARTGS_MULTIMODAL_TCP_FORCE_SLIP_MODES_CODEX_IMPLEMENTATION.md`. Config: `joint_discovery.multimodal_frontend`
(example: `configs/joint_discovery_multimodal.example.yaml`; contact annotation template:
`configs/contact_intervals/EXAMPLE.yaml`).

## 1. One kinematic model, three contact assertions

For a candidate `H` with `A_H(q)` (`rigid_kinematics.apply_joint`) the canonical material point under the TCP is
`c_t = A_H(q_t)^-1 p_TCP(t)` (`canonical_contact_point`). The modes only differ in what is asserted about `c_t`
(`contact_modes.py`):

| mode | assertion | whitened row (per training time) |
|---|---|---|
| `sliding` (default, parent) | `c_t` lies on the observed moving surface S_M; slip is free | `signed_distance(c_t, S_M) / hypot(sigma_surface, sigma_TCP)` where S_M supports `c_t` for **every** candidate |
| `stick_soft` | latent `c_t ~ N(c*_k, Sigma_stick)` seen through TCP noise | `(c_t - c*_k) / hypot(sigma_stick, sigma_TCP)` (marginal), drift reported |
| `stick_hard` | one material point for the whole uninterrupted segment | `(A_H(q_t) c*_k - p_TCP(t)) / sigma_TCP`, finite TCP noise |

`S_M` is built from the observed moving-support points at the reference time, where `A_H = I` for every hypothesis.
So it is candidate independent: no plane is ever fitted to a hypothesis' own TCP curve.

**Anchor elimination (exact).** For rigid `A_t` and isotropic TCP noise,
`sum_t ||A_t c - p_t||^2 = sum_t ||c - A_t^-1 p_t||^2`. The anchor is therefore `c*_k(u, o, q) = mean_t A_t^-1 p_t`
over the segment's **training** times. This is variable projection with:
- one structural anchor per segment, never per-frame anchors and never sigma = 0;
- `row_q_deps` declaring that each row depends on every `q` of its segment (`fit_geometry`'s sparse Jacobian honours it);
- held-out scoring with the anchor frozen from training;
- a release / regrasp starting a new segment with a new anchor;
- `rigid_grasp` rejected at parse time (`NOT_IMPLEMENTED`); `rigid_grasp_residual` exists only as the documented SE(3) form.

**Contradictions.** χ²/dof above `contradiction_chi2_per_dof` gives `CONTACT_MODE_CONTRADICTED`. One hypothesis being
contradicted is discrimination. A segment contradicted by **every** hypothesis means the declared mode is wrong. Its
rows are then removed from the fit (they cannot drag the hinge), its residuals are kept, and the decision carries
`CONTACT_MODE_CONTRADICTED`, so it is never export-ready. There is no automatic switch to another mode.

## 2. Force (`wrench_physics.py`, mode aware)

**Gate.** `force_gate`: every `WrenchCalibration.readiness()` gate must pass. A failure makes force ABSTAIN only; TCP
position rows are unaffected.

**Diagnostics** (always reported, never decisive):
- signed `rho_par = f·v_TCP / (|f||v_TCP|)`;
- `cos_phi_H` against the predicted material velocity at the contact;
- `Q_H` (N for P, N·m for R, never compared raw);
- `P_joint = Q_H qdot` and `P_slip`. Their sum is the TCP power for either hypothesis: an identity, not evidence.

**Mode-specific terms:**

| term | when it is active | otherwise |
|---|---|---|
| kinetic friction direction | sliding segments with observed normal and `\|s_T\| > eps_slip`, on identical samples for R and P: `f ~ -n_out + mu s_T/\|s_T\|`, one bounded `mu` per hypothesis fitted on training only | ABSTAIN; `CONTACT_MODEL_INADEQUATE` if both hypotheses miss the direction |
| static friction cone | sticking segments with `contact_model: unilateral_point` and observed normal: `f_N >= 0, \|f_T\| <= mu_s f_N`, as feasibility, never a "perfect" zero residual | grasp / unknown contact models ABSTAIN |
| discriminative force evidence | only under an explicitly declared mechanical model (`min_effort_tangent`: vMF around the predicted material velocity, kappa fitted on training) | `FORCE_NONDISCRIMINATIVE` |

The discriminative term must also pass two checks:
- the predicted directions must differ by at least 10°;
- the held-out Δll must exceed information-free nulls: sign-reversed, gravity-only, random constant directions and
  random per-frame directions. Time-shuffled and time-shifted nulls bind only when the favoured hypothesis predicts a
  time-varying direction.

## 3. TCP-guided support and tracks

**`tcp_support.py`:**
- **Tool occupancy.** A capsule from tip + radius up to EE + extend, from `robot_tool.json` `T_ee_tcp`. The radius is
  `APPROXIMATE` (no mesh). The ambiguity band runs only alongside the shaft, so the object surface touched by the tip
  stays CLEAR. This replaces the blanket "within 5 cm of the tool" rule.
- **Seeds.** Confirmed contact gives positive moving seeds (still needs a span of at least 8 mm). Unknown contact gives
  proposals, which need the relaxed 5σ motion test and are never residuals. Sticking anchors seed the predicted contact
  region at every contact frame.
- **Pixel seeds.** The dense pass gets pixel seeds on confirmed frames only.

**`track_refinement.py`:**
- NCC re-matching of lost moving-track observations at **training** times.
- The search window covers both hypotheses' predictions.
- Recovered observations are **2D only** by default: a recovered depth association is not independently verified, and
  the pilot showed ±1.5 cm depth gating corrupted anchors.
- Their pixel sigma is calibrated independently by re-finding raw tracker observations with the same procedure
  (`calibrate_matcher`), and carried per observation in `TrackObservations.uv_sigma_scale` (absent = legacy behaviour).

## 4. Iterative controller (`multimodal_frontend.py`)

```
frozen v2 block folds -> support(+TCP) -> R,P EM fit -> + contact (+ verified force) rows      round 0
round k: A track recovery -> B support update (+ anchor seeds) -> C/D/E refit -> F diagnostics
         G held-out NLL of prev and new candidates on the IDENTICAL observation set (round-0 moving tracks, raw held-out)
         H accept iff the BEST candidate's held-out NLL improves > min_relative_improvement AND moving support does not
           shrink > 20 % AND no new contact contradiction; else roll back and stop (last accepted state persists)
final:   selection v2 on the accepted state (contact/force evidence consumed), all-frame refit for the authority,
         discovery-format export (joint_candidates.json, joint_states.npz, part_support/, tracks, contact states,
         canonical_init PLYs = RGB-D points, NOT Gaussians, authority_source_manifest.json, validation_report.json)
```

The rejected hypothesis is allowed to get worse with better data: that is what separates the hypotheses.

## 5. How robot evidence enters the physical decision (`physical_selection.select_physical_joint`)

- **Contact contradicts the revolute fit** (`contact_discrimination` rejects R): MODEL_INCONSISTENT.
- **Positive slider support.** Any one of the following replaces an unobservable guide rail, but only *together with*
  the existing exclusion of every plausible hinge on the observed interface:
  - a confirmed stick consistent with P;
  - contact that discriminates for P;
  - verified force under a declared model favouring P.

  A contact mode alone never identifies a joint (D06).
- **Force favouring P against a measured rotation:** `FORCE_DISAGREES`, so not export-ready.
- **Unverified sigma:** `UNVERIFIED_PRIOR_SCALE` blocks export readiness.

Reported per decision:
- `joint_type_identified`
- `joint_geometry_validated`
- `contact_assumption_satisfied`
- `contact_evidence_used`
- `force_evidence_used`
- `working_motion_model` / `working_model_is_approximation`
- `physics_joint_export_ready`

## 6. Pipeline

`pipelines/joint_aware_reconstruction.py --multimodal-frontend [--contact-segments FILE]`:
```
audit -> discover -> multimodal_refine -> select (copies the accepted decision) -> authority (from accepted/)
      -> dense (seeds from accepted support + TCP pixel seeds on confirmed frames) -> build -> parts (two separate
      splatfacto-depth trainings) -> blend -> art -> ... -> export (multimodal provenance in the handoff)
```
No `ns-train` runs before `multimodal_refine` has written its accepted state (D04). Standalone:
`python -m estimotion_2part.joint_discovery.cli multimodal-refine --scene S --from-run RUN [--robot-archive A]
[--contact-segments C] --output NEW_DIR`.
