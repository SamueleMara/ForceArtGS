# Multiview disagreement on `final_black_box_episode_000000`: root cause (2026-11-14)

> **Follow-up (2026-11-16):** the corrections proposed here were implemented and validated in
> [rigid_motion_corrections.md](rigid_motion_corrections.md).

Follow-up to section 9 of [tcp_contact_tracking.md](tcp_contact_tracking.md). Baseline FreeArtGS
(tracking → ICP) on clip RGB-D frames 70–128 (stride 2, **30 independent frames**, source rows
140–256) gave a final lid rotation of 14.6° (cam_000) vs 8.0° (cam_002) and screw axes 85° apart.
Nothing below changes the default pipeline. Phase A remains `diagnostic_only` and Phases B/C
remain disabled. No weights were tuned and no Gaussian training was run.

Results (lightweight; runs, configs, logs, track dumps and scripts are in tarballs):
`/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/multiview_root_cause_20261114/`.

## 1. Summary

| Hypothesis | Verdict | Evidence (sections) |
|---|---|---|
| Coordinate-frame / SE(3) handling | **Not the cause.** | §2 |
| Weak rotational observability | **Cause for cam_001/cam_002** (grazing lid top, edge-on at ~src 108). Not for cam_000. | §3, §5 |
| Wrong part assignment | **Cause.** The two-part segmentation never separates lid from base. cam_002's "moving" label is 30–56 % static base. Labels are unstable under 1–2 % mask perturbations. | §4, §6 |
| Tracker corruption | **Cause for cam_002**: window-0 2D tracking failure, 7–8 mm median / 24–62 mm p90 lid depth errors, lid tracks only 32–42 % rigidly consistent. cam_000 tracks are good. | §5 |
| Different visible geometry | **Root of the above**: cam_000 sees the end face (rotation in-plane, 0.75 m). cam_001/002 see a grazing lid top (≈1 m, 1.5–3× depth noise). | §3 |
| Multiview preparation | **Amplifies it**: the reference camera is chosen by motion-score gap and picks the camera with the most *spurious* motion. | §7 |

The 85° axis disagreement is a **cam_002 failure**. cam_000's baseline estimate (−13.8° about the
hinge, 14.6° total) matches the tracker-free reference (−15.0°). cam_002's estimate (−0.4° about the
hinge; axis 87° from the hinge) is wrong. cam_001 fails the same way (−0.7°, axis 82° off).

## 2. Frames and conventions (verified)

* **Pose convention.** `cameras.json` is `opencv_optical_camera_to_world` in `base_link`.
  COLMAP `images.txt` (used by `prepare_multiview`) is identical: 0.0000°, 0.00 mm for all three cameras.
* **Motion convention.** ICP `global_poses[t]` maps frame t into frame 0, so the object motion is `inv(G[t]) G[0]`.
  Check: moving frame-0 points with it fits observed depth to 1.8 mm, against 10.4 mm for no motion (cam_000).
  The base-frame motion is `T_bc M T_bc⁻¹`. Conjugation preserves the rotation angle, so **14.6° vs 8.0°
  cannot come from frame handling**. `camera_in_part_frame` in `prepare_multiview` uses the same convention.
* **Axis comparison.** It is sign-invariant (`|a·b|`).
* **Calibration.**
  - cam_001 ↔ cam_002 agree to 5–9 mm and ~1.5° (full-scene ICP).
  - cam_000 is offset from both by 1.7–2.5 cm at the box (scene), and by up to ~4–5 cm on the black box,
    mostly along base y, which is cam_001/002's viewing direction.
  - Extrinsic error vs a material-dependent depth bias is not resolved by this data. Extrinsics were not refitted.
  - Effect: at most a few degrees on axes and centimetres on hinge position. **No effect on angles.**

## 3. Tracker-free reference and view geometry

* **Hinge direction.** The intersection of the lid-top planes in cam_001 and cam_002 at frames 64 (closed) and
  200–228 (open) gives (0.997, −0.06, 0.02), i.e. 2–5° of yaw from base x. The hinge line passes near
  (y 0.59, z 0.17).
* **Reference angle.** A 1-DOF fit per camera: that camera's frame-70 lid points are rotated about the hinge
  direction (per-camera pivot, θ ≤ +1° because the lid cannot close through the base) to minimise the trimmed
  nearest-neighbour distance to the camera's own depth.

| lid angle rel. src 70 | cam_000 | cam_002 | TCP arc (no-slip) | cam_000 image |
|---|---|---|---|---|
| src 128 (clip end) | −15.0° (fit 1.6 mm) | −21.1° (fit 6.9 mm) | ≈17.5° | ≈16° |
| src 228 | −25.8° | −35.2° | ≈28° | ≈31° |

  - The physical clip-end opening is ≈15–21°. The spread is the reference uncertainty.
  - cam_001's fit is unreliable: its pivot moved 7 cm and its cost curves are flat.

* **What each camera sees** (`lid_view_geometry.json`):
  - **cam_000** looks almost along the hinge (view direction −x). It sees the lid's end face, whose normal
    ≈ the hinge axis, so the hinge rotation is in the image plane. Range 0.75 m.
  - **cam_002** sees the lid top at 12.5° elevation at src 70. The elevation falls to 0° at src ≈106–110,
    after which cam_002 sees the underside. Range 0.99 m.
  - **cam_001** elevation goes from 26° to 8°.
* **Raw frame-to-frame depth noise on lid pixels** (median / p90): cam_000 2.0 / 5.9 mm; cam_002 3.0 / 17.1 mm.
* **White-noise information bound** on the hinge angle from depth geometry alone: 0.05–0.21° in all cameras.
  Marginalising translation inflates it 1.9× (cam_000) and 2–4× (cam_001/002). Observability is
  therefore limited by **systematic** effects (visible surface changes, heavy-tailed depth, nearest-neighbour
  matching), not by white noise.

## 4. Part assignment (segmentation vs geometry)

Each pixel's 3D point gets a geometric class: lid = within 1.2 cm of the reference-rotated lid model;
base = z < 0.145 m; tool = within 2 cm of the kinematic tool axis. Composition of the tracker's
**moving** mask (baseline):

| clip frame | cam_000 lid / base / tool | cam_002 lid / base / tool | cam_001 lid / base / tool |
|---|---|---|---|
| 0 | 0.66 / 0.00 / 0.20 | 0.85 / 0.12 / 0.00 | 0.61 / 0.34 / 0.04 |
| 9 | 0.75 / 0.02 / 0.18 | 0.46 / 0.36 / 0.02 | 0.67 / 0.19 / 0.07 |
| 19 | 0.78 / 0.00 / 0.13 | 0.33 / 0.50 / 0.01 | 0.28 / 0.06 / 0.07 |
| 29 | 0.73 / 0.01 / 0.08 | 0.29 / 0.32 / 0.00 | 0.12 / 0.00 / 0.07 |

* cam_001's classes use cam_002's reference angle and pivot, so its lid / unmatched split is approximate.
* In every camera, 25–60 % of the true lid points sit in the **static** label.
* The label that holds the lid majority flips mid-clip: cam_000 at clip 10, cam_002 at clip 3
  (`label_consistency.json`).
* Figures: `figures/segmentation_vs_geometry_{baseline,toolclean}.jpg`.

## 5. Tracks (re-extracted with the tracker's own front end)

`samuele/dump_window_tracks.py` reproduces the pipeline's per-pair track sets exactly (same pre-filter counts
as the logs). Window 0→7 values (other forward windows in `track_analysis.json`, `track_error_split.json`):

| | cam_000 lid tracks | cam_002 lid tracks |
|---|---|---|
| 2D error vs reference motion (median, full-res px; true flow ≈6 px) | 0.8–1.8 | **4.8** (window 0), 1.3–2.4 later |
| depth error, median / p90 | ≈2 / 8–16 mm | **7–8 / 24–62 mm** |
| rigid fit of lid tracks (about hinge; reference) | −3.3, −3.6, −3.5, −3.8° (−3.5, −3.7, −3.3, −4.3) | −0.4° (axis 84° off), −14.3, −0.3, −5.1 (−5.2, −4.6, −5.0, −5.6) |
| RANSAC rigid inliers | 69–89 % | 32–42 % |
| dropped by the global 2σ displacement pre-filter | lid 0.8 %, base 0 % | **lid 12 %, base 0.7 %** |

* **Pre-filter bias.** The pre-filter (`Tracker.get_visible_trajectories`) is global, so it removes the
  minority moving part's tracks preferentially.
* **The tracker's part transforms are free 4×4 matrices, not SE(3).** Singular values deviate 1–1.3 % and
  orthogonality by up to 2.3 %. That is the same size as one frame's lid rotation (0.9–1.5 % off-diagonal).
  One T2 carries a 0.21 m translation.
* After SVD projection, the tracker's own moving-branch rotations underestimate in cam_000 and have the wrong
  sign in cam_002 window 0 (`window_motion_comparison.json`).
* These transforms are **not** used downstream: ICP uses only the masks.
* Figure: `figures/tracks_vs_reference_motion.jpg`.

## 6. Controlled ICP experiments (pipeline ICP on copies)

Final values at src 128 (reference cam_000 −15.0°, cam_002 −21.1°):

| masks given to ICP | cam_000 about hinge (axis vs hinge) | cam_002 about hinge (axis vs hinge) | axis disagreement |
|---|---|---|---|
| tracker (baseline) | −13.8° (19°) | −0.4° (87°) | 85° |
| geometric oracle: object & z > 0.165 m & ≥2 cm from tool; static z < 0.15 m | **−16.1° (10°)**, per-frame RMS vs reference 0.7° | **−1.8° (65°)** | 74° |
| robot removed upstream, then normal tracking (§8) | −8.0° (25°) | −23.3° (25°) | 44° |

* **Oracle masks.** cam_000 is solved. cam_002 still fails: its ICP **stalls at −1.2° from src 78**. The lid's
  front-edge displacement is right to within 7 mm, but the near-hinge point is lifted 49 mm instead of 0. The
  rotation has been replaced by a translation of the whole lid. cam_000's static part also drifts 3.4° / 29 mm,
  because a planar end face cannot constrain in-plane rotation.
* **Robot removed upstream.** Only 233 px/frame (cam_000) and 16 px/frame (cam_002) were removed, yet the
  segmentation changed completely:
  - cam_000's moving label became 76–95 % **base**;
  - cam_002's −23.3° is a single jump (−3.3° → −20.7°) at src 110, exactly when cam_002 crosses the lid plane.
  Neither is a correction. **The two-part segmentation is unstable under 1–2 % mask perturbations.**

## 7. Multiview preparation

`prepare_multiview` picks the reference camera by `|score_moving − score_static|`, where the score is the
translation path plus the rotation path of the ICP poses. Spurious translation inflates it:

| camera | score gap |
|---|---|
| cam_002 (chosen) | 0.29 |
| cam_000 | 0.15 |
| cam_001, if included | 0.52 (it would win) |

Every camera's training pose is then derived from the worst camera's motion.

## 8. Phase B (seeds) — evidence per seed

All positive prompts (`seed_audit.json`; figures `figures/accepted_*.jpg`, RGB + depth + 3D numbers per seed):

* **Without tool exclusion,** 135/150 (cam_000) and 144/150 (cam_002) prompts lie within 1.5 cm of the
  kinematic tool axis (tool tip). 146–148/150 are on depth edges.
* **With tool-volume exclusion,** prompts are 30 / 66 on the lid, 11 / 1 on the tool, and 109 / 83 "above the
  rim but not attributable". None are on the base or rim.
* **Accepted seeds** are those in seed-active forward windows.

  | Run | Accepted seeds (windows) | Lid | Tool | Ambiguous |
  |---|---|---|---|---|
  | cam_000, tool exclusion | 20 (4) | 6 | 2 | 12 |
  | cam_002, tool exclusion | 10 (2) | 3 | 1 | 6 |
  | cam_000, without exclusion | 5 (1) | 0 | 0 | 5 |

  "Ambiguous" means 1.5–2.1 cm from the tool axis and 0.8–2.1 cm from the lid model. The tool tip and the black
  lid lip touch there at the same depth, and the TCP uncertainty is 1.5 cm, so the data cannot separate them.
* **Upstream contamination.** Object masks contain the tool: 233 px/frame in cam_000 inside a 1.5 cm cylinder
  behind the calibrated tip. The tracker then carries 44–122 tool tracks per cam_000 window pair. The new
  opt-in `samuele/clean_robot_from_masks.py` removes these upstream of tracking. It never removes anything in
  front of the calibrated tip; tests are in `tests/test_clean_robot_from_masks.py`. It is necessary for physical
  correctness (Phase C matched tool points, §9) but **not sufficient** (§6).
* No artificial prompts were added. The contact neighbourhood here is ambiguous by construction.

## 9. Phase C observability

Reconstructed offline from the tracks, the Phase C segmentation, the transforms and the TCP artifact. The
reconstruction reproduces the recorded contact distances exactly, e.g. cam_000 window 0: 9.2, 8.6, 9.7, 8.9,
10.1, 10.7, 10.7 mm (`phasec_*.json`).

* **Canonical frame-0 moving surface.**
  - cam_000: 77–85 % lid, **12–17 % tool**.
  - cam_002 (windows 7–21): **37–75 % static base**.
  - The contacted region (the underside of the lid lip, under the tool) is **not observed**.
  - Nothing within 20 mm of the contact is attributable to the lid, given the 1.5–2 cm TCP uncertainty.
* **Accepted matches.**
  - In 26/28 (cam_000) and 27/28 (cam_002) samples, the nearest canonical point is a **tool track point**.
  - There are only 1–2 distinct matched points per window, at 4–27 mm.
  - So the term pulls the tool's own tracked point toward the TCP. This is geometrically incorrect.
* **Rank.**
  - One scalar distance per 6-DoF transform, by construction (one contact sample per window frame).
  - Stacked over a window, the normalised singular values are 1, 0.29–0.63, 0.07–0.23, ≤ 0.0025.
* **Sensitivity.**
  - The contact distance changes by 0.4–2.9 mm per degree of hinge rotation; σ = 15 mm.
  - One σ therefore equals 5–37° of hinge rotation, against 0.6–0.9° of lid motion per frame. The term is
    uninformative about the hinge.
* **Gradients on the actual parameters** (12 entries of each T_k, final state; exact pipeline losses: 200 ×
  normalised-Huber main loss + contact):
  - |∇contact| / |∇visual| = 0.15–1.16 (window medians);
  - cosine mostly −0.1 to −0.97 (opposed, near equilibrium);
  - weighted contact loss = 1–10 % of the weighted visual term (correcting the earlier "2–20×").
* **Why the motion changes while distances change by < 1 mm.**
  - Phase C vs null (weight 1e-6 = baseline) differ most in pairs where the visual term barely determines the
    transform: the null control's moving-track fit there is 11.8–25.7 mm, against 1.5–3 mm elsewhere.
  - There the rank-1 push relocates T within a flat valley. Three things make this easy:
    - the Huber term is linear (constant gradient) when d ≥ δ (windows 2–6);
    - Adam normalises step size per parameter;
    - the matrices are free affine.
  - The transforms differ by up to 31° / 385 mm as matrices, but only by 0.1–25 mm at the contact and 0.2–60 mm
    at the lid centre.
  - The matched tool point cannot reach the TCP without breaking the track fit, so the distance hardly moves.
* **Verdict.** The constraint is neither geometrically correct nor informative on this sequence. **Do not
  increase contact weights.**

## 10. Proposed minimal corrections and controlled tests

| # | correction (scope) | controlled test | status |
|---|---|---|---|
| 1 | `prepare_multiview`: choose the reference camera by self-consistency (moving-part rigid-fit residual vs no-motion on its own depth), not motion magnitude. Gate on cross-camera base-frame agreement (rotation, sign-invariant axis) and record the result in `multiview.json`. | Offline on these runs: rigid fit cam_000 1.8 mm (no-motion 10.4 mm) vs cam_002 9.9 mm (12.9 mm), so cam_000 is chosen; the gate flags the 85° disagreement. | proposed (pipeline change) |
| 2 | Tracker part transforms in SE(3) (axis-angle + translation, or projection after each step). | Same frames, compare lid-track rigid fit (§5) and moving-mask composition (§4). | proposed |
| 3 | Per-part (or local) displacement pre-filter instead of the global 2σ rule. | Lid vs base drop ratio from `dump_window_tracks.py` (cam_002 w0: 12 % vs 0.7 %). | proposed |
| 4 | Remove robot pixels from object masks upstream (opt-in). | `tests/test_clean_robot_from_masks.py` (5 tests). Real run: removes tool tracks but exposes §6 instability. | **implemented**, not enabled by default |
| 5 | Segmentation stability regression: perturb ~1–2 % of mask pixels; part labels and motion must not flip. | The baseline pipeline currently fails it (§6). | proposed test |
| 6 | Per-camera observability gate: elevation of the camera over the moving surface, depth noise at the part, and a rotation–translation substitution check (near-hinge vs far-edge displacement). Cameras that fail it don't define motion. | cam_002/cam_001 fail and cam_000 passes on this clip (§3, §6). | proposed |
| 7 | Phase B: no prompt within (tool radius + TCP uncertainty) of the contact. If nothing unambiguous remains, emit no prompt. | Needs a measured tool radius (not available, not invented). Accepted seeds here would all be rejected or kept as ambiguous. | proposed |
| 8 | Phase C prerequisites before any weight change: robot excluded (4); a canonical point attributable to the contacted surface within σ of the contact, otherwise skip; constraints aggregated over frames on one articulated motion (rank); SE(3) transforms (2). | §9 metrics as acceptance checks. | proposed |

## 11. Repeat on a scene with a clearly visible contact surface: `final_red_fridge_episode_000000`

The robot closes a red fridge door (revolute, vertical hinge) by pushing its large, planar outer face. The
same unchanged pipeline (baseline) was run on RGB-D frames 90–148 (stride 2, 30 independent frames).

* **Camera choice.** cam_001 and cam_002 see the door and the contacted face. cam_000 sees the fridge's side
  and back, not the door, so it is excluded from motion comparison.
* **Reference.** The plane of door-attached points within 10 cm of the TCP (tool excluded) is static until
  src ≈116–118, then rotates by ≈24° (cam_000 view), ≈17° (cam_001) and ≈20–24° (cam_002, to src 144).
  Each camera fits a different door face, so only the changes are compared. From src 122 on, the TCP lies
  3–8 mm from cam_002's fitted door plane, so the contacted surface is observed.

| | cam_001 | cam_002 |
|---|---|---|
| ICP rotation about vertical (src 90→148) | −22.0° | −15.9° |
| axis vs vertical | 2.6° | 6.2° |
| static part drift | 0.45° / 7 mm | 0.52° / 11 mm |
| **cross-camera axis disagreement** | **5.9°** | |

* **Agreement.** Both cameras recover a vertical axis and an angle inside the reference range. Motion starts
  at the same time as in the reference. So the pipeline and the frame conventions produce cross-camera
  agreement when the moving surface is large, well lit, non-black and viewed frontally. That supports §1:
  the black-box failure comes from view geometry and segmentation, not from frames.
* **Multiview reference.** `prepare_multiview` chose cam_001 here (score gap 0.72 vs 0.49).
* **Force contact is weak.** It is confirmed on only 5 / 30 clip frames (19 uncertain, 6 absent): the door is
  pushed below the 6σ (≈0.29 N) threshold. Phase B/C would act on very few frames. The three-state contact
  correctly does not promote those frames.
* **Not repeated on the fridge:** track-level and Phase C gradient analysis (Phase C would be gated by the
  missing force contact).

## 12. Limitations

* The reference relies on the hinge direction from two cameras' plane fits and a per-camera 1-DOF fit. The
  clip-end estimates differ by 6° between cameras (15.0° vs 21.1°). Conclusions use only differences larger than that.
* Geometric classes use the robot kinematics (TCP uncertainty 1.5 cm), base-frame heights measured from this
  scene, and the reference lid model. The "unmatched above rim" class is ambiguous by design.
* The cross-camera offset of cam_000 (2–5 cm) is unexplained. It was not refitted, per instructions.
* Single recording and clip; one segmentation-perturbation experiment.
