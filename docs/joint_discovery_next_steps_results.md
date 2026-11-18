# Joint discovery, next-steps sprint: results (2026-11-17)

Contract: [`FORCEARTGS_JOINT_DISCOVERY_NEXT_STEPS_CODEX.md`](../FORCEARTGS_JOINT_DISCOVERY_NEXT_STEPS_CODEX.md).
Code report: [`JOINT_DISCOVERY_NEXT_STEPS_IMPLEMENTATION_REPORT.md`](../JOINT_DISCOVERY_NEXT_STEPS_IMPLEMENTATION_REPORT.md).

Run directories are under `outputs/joint_discovery_next/<scene>/` (git-ignored). Experiment ids use
the `NEXT_*` namespace, so they are never confused with the earlier `B4`/`B9` runs. Every run
records the git HEAD and diff hash, the config hash, the observation-file SHA-256, the per-camera
calibration hash, the time-map hash and the tool-exclusion parameters. All runs in the tables below
come from one frozen code state (`outputs/joint_discovery_next/NEXT_rerun_frozen.sh`).

"Post hoc" columns use the tracker-free reference hinge or the 0.145 m lid-height split. These
are computed after the runs and never enter fitting, thresholds or selection.

## 1. Black box (`final_black_box_episode_000000`, RGB-D 70:128:2, 30 frames)

### 1.1 Stage A: where the 2.8° remote-hinge solution comes from

**A1, counterfactual audit** (`NEXT_A1_depth_audit/`). The same 2.4M-row observation table is
scored under frozen hypotheses:

| hypothesis | role | motion |
|---|---|---|
| `B2_R` | fitted RGB-D, default cameras | 2.76° revolute |
| `B2_P` | fitted RGB-D | 6.3 cm prismatic |
| `B3_R` | fitted 2D-only | 7.97° revolute |
| `C0_R` | cam_000-only fit (diagnostic) | 15.0° revolute |
| `REF_cam_002` | tracker-free reference (post hoc) | 21.1° revolute |

Only each track's anchor is re-estimated, with the joint frozen, using the same Gauss-Newton for
every hypothesis. The table gives the objective gap (Student-t NLL) on the union of moving
support; positive means the second hypothesis explains the data better.

| a vs b | anchors from | 2D gap | depth gap | cam_001 2D/depth | cam_002 2D/depth |
|---|---|---|---|---|---|
| B2_R vs C0_R (15°) | RGB-D | −41 578 | −5 059 | −26 611 / −3 023 | −14 967 / −2 036 |
| B2_R vs C0_R | 2D only | **+2 549** | **−13 726** | −1 105 / −13 545 | +3 654 / −181 |
| B2_R vs B3_R (8°) | 2D only | +538 | **−13 685** | +402 / −8 600 | +137 / −5 085 |
| B2_R vs REF (21°, post hoc) | 2D only | +5 475 | +7 136 | +541 / −375 | +4 934 / +7 511 |

With 2D-placed anchors, the depth terms of cam_001 favour the 2.8° fit by ≈ 13.5k NLL, while
2D favours the larger rotations. With depth-placed anchors, a 15° motion cannot reproduce the
2D tracks (2D gap −41.6k).

**A2, systematic vs stochastic** (`NEXT_A1_depth_audit/`, `NEXT_A2_posthoc_*.json`):

* **Static base: clean.** The consensus static core (static under every fitted hypothesis, 2 404
  tracks) shows:
  * median signed depth residual −0.02 mm (cam_001) and −0.04 mm (cam_002); MAD 2.4 / 3.1 mm;
  * 6–9% of samples beyond 3σ;
  * 2D residual 0.1–0.2 px, both early and late.

  There is no camera-level bias and no geometry mismatch (category 5 ruled out for the base).
* **Lid tracks: highly correlated errors.** Depth residuals have lag-1 autocorrelation 0.96–0.98
  under every hypothesis. A per-observation independent likelihood therefore counts each track's
  ~30 depth samples as ~30 independent pieces of evidence.
* **Post hoc, against the reference motion.** Consistent means a measured 3D displacement within
  ±30% of the rigid prediction.

  | camera | lid observations consistent | signed along-ray error, dt 1–5 → 20–30 frames | error ⊥ ray, dt 1–5 → 20–30 | incidence cos |
  |---|---|---|---|---|
  | cam_000 | **94%** | +0.5 → +1.6 mm | 0.5 → 2.2 mm | 0.80 |
  | cam_001 | **4%** | −0.8 → +0.4 mm | **3 → 31 mm** | 0.77 → 0.55 |
  | cam_002 | **21%** | **+1.4 → +21.5 mm** | 2.6 → 16 mm | 0.45 → 0.37 |

**Dominant corruption (Gate A).**
* cam_001: the 2D tracks lag the black lid's material motion, with drift that grows to 31 mm
  perpendicular to the ray by the end of the clip (category 6).
* cam_002: as well as a smaller drift, the measured depth of the lid becomes progressively too
  far (+21.5 mm) at grazing incidence (category 2, a material/angle-dependent ToF bias).

Both errors are **coherent across the lid patch**. Local rigidity holds, so per-observation rules
cannot detect them (§1.2). The joint fit absorbs both as a small rotation about a remote hinge.
Wrong-surface association (category 3) and robot contamination (category 4) occur but are minor:
about 1.1k observations in each class per camera are classified as wrong-surface/occlusion, and
about 130 as robot proximity.

### 1.2 Stage A3 / B: controlled ablations (identical tracks, fold from `NEXT_A0_B2`)

Every row uses cluster-robust z over 5 cm surface groups and the observability policy. z is
revolute vs prismatic, shown as naive / cluster (number of groups).

| run | cams | depth | decision | z (naive / cluster) | R arc | R hinge radius | P | moving R/P | R identifiable | axis vs ref (post hoc) | below-lid R (post hoc) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `NEXT_A0_B2` (old policy) | 001+002 | raw | revolute | 3.2 / – | 2.8° | 1.21 m | 6.3 cm | 1091/1055 | – | 12.0° | 0.18 |
| `NEXT_A3_V2D` | 001+002 | off | revolute | 7.6 / 3.0 (430) | 8.0° | 0.14 m | 2.3 cm | 3456/1343 | yes | 5.7° | **0.74** |
| `NEXT_A3_V2D_rawD` | 001+002 | raw | **ambiguous** (Occam P) | 3.2 / **1.7** (430) | 2.8° | 1.21 m | 6.3 cm | 1091/1055 | **no** | 12.0° | 0.18 |
| `NEXT_A3_V2D_evidenceD` | 001+002 | evidence | ambiguous (Occam P) | 3.7 / 1.9 (435) | 3.3° | 0.98 m | 6.3 cm | 968/930 | no | 12.4° | 0.20 |
| `NEXT_A3_V2D_ar1D` (ρ = 0.55) | 001+002 | raw + AR(1) | ambiguous (Occam P) | 3.6 / 1.9 (430) | 2.8° | 1.17 m | 6.3 cm | 1088/1060 | no | 11.1° | 0.17 |
| `NEXT_A3_cam001_rawD` | 001 | raw | ambiguous (Occam P) | 6.1 / 3.3 (194) | 5.7° | 0.55 m | 6.0 cm | 649/611 | no (hinge 0.43 m off support) | 30.2° | 0.22 |
| `NEXT_A3_cam002_rawD` | 002 | raw | ambiguous (Occam P) | 2.1 / 1.6 (236) | 1.5° | 2.58 m | 6.7 cm | 454/453 | no | 81.4° | 0.06 |
| `NEXT_A3_cam000_rawD_diag` | 000 | raw | revolute | 18.4 / 4.8 (200) | **15.0°** | 0.12 m | 4.0 cm | 827/551 | yes | 11.9° | 0.07 |

**Findings.**
* **The confidently wrong label is gone.** With the default cameras, the RGB-D fit is still a 2.8°
  remote hinge, but it is now reported `ambiguous/unobservable`. Three independent reasons:
  * cluster-robust z = 1.7 (the naive z was 3.2);
  * the hinge line is 1.16 m from the observed moving support;
  * the revolute differs from its best translation by only 1.7 σ_px.
* **The B1 evidence rules do not recover the motion** (2.8° → 3.3°). They remove isolated
  wrong-surface and robot samples, but not the coherent patch errors. This is a negative result.
* **The B2 AR(1) depth model does not recover it either** (2.8° → 2.8°).
  * ρ is calibrated on the training static core and bias-corrected (Marriott–Pope).
  * Caveat: the χ² < 4 core under-represents persistent offsets (lag-1 0.78/0.17 there, versus
    0.96–0.98 on the lid), so the inflation is conservative.
  * This is a negative result.
* **2D-only (V2D) gives a plausible 8° hinge, but it is not a recovery.**
  * The hinge radius is 0.14 m and the axis is 5.7° from the reference.
  * But 74% of its moving support lies below the lid height (post hoc), i.e. base leakage, and
    its arc is still about half the reference.
  * It fails the "without increasing static-base contamination" clause of Gate B.
* **Per-camera:** cam_001 alone (5.7°) and cam_002 alone (1.5°) are both non-identifiable. cam_000
  alone recovers 15°, as before. It stays a diagnostic, because its TCP residual (+28 mm) points
  to a registration offset.

**Gate B: FAIL / UNRESOLVED.** No correction recovers the motion from cam_001+cam_002 without
leakage. The output is now an honest abstention instead of a confidently wrong axis, which is the
behaviour the contract requires when the evidence is insufficient.

### 1.3 Stage C2: blocked-time and held-out-camera folds (`NEXT_C2_folds_rawD/`)

| fold | q at test times | R vs static z (cluster) | R vs P z (naive / cluster) | label |
|---|---|---|---|---|
| contiguous mid block (8 frames) | interpolated across the gap | 8.9 | 2.4 / 1.3 | ambiguous |
| forward block (last 8) | extrapolated (constant) | 7.8 | 1.4 / 0.6 | ambiguous |
| fit cam_002 → score cam_001 | constrained by the other camera | 6.4 | **−9.9 / −3.3** | prismatic |
| fit cam_001 → score cam_002 | constrained by the other camera | 4.7 | −3.3 / −1.5 | ambiguous |

Labels are **not stable** across folds. A revolute fitted on one default camera predicts the other
camera *worse* than the prismatic does. The two default views do not agree on any finite hinge,
which independently confirms the abstention.

### 1.4 Stage D: contact windows and wrench (`NEXT_D1_contact_windows/`)

* **Force.** A sustained force event (episode baseline 0–0.5 s, σ ≈ 0.04 N per axis) starts at
  source frame 122 (2.44 s, RGB-D frame 61). That is **before** the visual clip (frame 70), and
  the force never returns to baseline (0.5 N at the episode end). The joint-discovery heuristic
  took its tare from clip frames 0–2, which are already inside this event, so ≈ 0.7 N of contact
  force was subtracted. That is why the 1 N threshold flagged 0/30 frames.
* **Geometry.** The TCP is within 2.5 cm of observed object depth during the whole episode, so
  geometry never gives `no_contact`.
* **Timeline.** Clip timeline: 30 `contact_probable` (force event + proximity). None are
  confirmed, because there are no independent annotations.
* **EE orientation** is constant to 0.026° over the clip, so gravity is constant there. The gravity
  gate still needs a declared tool-mass bound to pass.
* **Wrench gates:**

  | gate | status | reason |
  |---|---|---|
  | rotation | UNVERIFIED | `R_EE_S` is user-declared |
  | sign | UNVERIFIED | operator-declared |
  | tare | UNVERIFIED | no second contact-free window to test drift |
  | gravity | UNVERIFIED | no tool-mass upper bound declared |
  | time sync | UNVERIFIED | exact frame join only |

  Force selection therefore **ABSTAINS**.

### 1.5 Stage F: renderer

**No canonical Gaussian asset exists for the black box**, in either `outputs/` or the external
SSD, so the status is `MISSING_CANONICAL_ASSET`. Nothing was trained.

## 2. Red fridge (`final_red_fridge_episode_000000`, 90:148:2, cam_001+cam_002)

* **Vision.** `NEXT_C3_rawD_policy`: revolute 18.9°, identifiable, cluster z 5.5 (see §6).
* **Stage D.**
  * Sustained force events at source frames 220–234 (peak **2.61 N**) and 299–313 (5.57 N).
  * The first event is inside the clip (RGB-D 110–117). It is impulsive, and the clip samples
    every 4th source frame, so the sampled peak is only 0.81 N.
  * Visual motion (masked depth change > 1 cm on 5% of pixels) starts at frame 118–119, right
    after that event.
  * The clip timeline has no `contact_probable` frames (28 unknown, 2 no_contact), because the
    TCP is never within 2.5 cm of an observed face. This matches the earlier finding that the
    door is pushed at an unobserved edge.
  * The force path abstains (same unverified gates as the black box).
* **Stage F, real frozen-asset scoring** (`NEXT_F1_frozen_asset_score.json`).
  * Asset: the existing `results_09_11_26/.../final_red_fridge_..._cams001-002/twopart_blend`
    checkpoint (236k Gaussians) rendered by the actual gsplat 0.1.11 rasterizer.
  * Its world is the **old cam_001 optical frame**. Its cam_001 and cam_002 poses imply one rigid
    map to the current `cameras.json`, consistent to 0.014° / 0.0002 mm (the asset predates the
    2026-11-13 registration correction by 3.6° / 4.9 cm). The map is applied; with any
    inconsistency the script fails closed.
  * Candidates are the training-split fits; scoring uses the held-out times; pixels are identical
    for all candidates; robot pixels come from the measured tool kinematics.

  | held-out score (lower is better) | cam_001 | cam_002 | mean |
  |---|---|---|---|
  | static | 6.13 | 5.44 | 5.79 |
  | prismatic | 6.09 | 5.42 | 5.75 |
  | **revolute** | **5.95** | **4.74** | **5.34** |

  * Render coverage is 98% of the depth-eligible pixels.
  * The asset was co-trained with the *legacy* joint, which for this fridge was **prismatic**.
    It is therefore not hypothesis-neutral, but its bias would favour P, not R.
  * Robust depth residuals are 5.6–8.3 cm. Alignment is only approximate, so these are rankings,
    not calibrated likelihoods.
  * No overlay images were produced.

## 3. White drawer (prismatic control, 150:330:6, cam_001+cam_002, stride-8 tracks)

`NEXT_C3_rawD_policy` uses the tracks and fold of the earlier B4 run and the new policy.

* Decision: **ambiguous/unobservable, Occam prismatic** (unchanged label). Cluster-robust R-vs-P
  z = 4.1 (naive 12.3).
* The revolute is *not identifiable*: its hinge is 8.9 m from the moving support (extent 0.37 m),
  and it differs from its best translation by only 1.8 σ_px.
* The prismatic candidate is 16.2 cm. No confident false revolute (Gate C on real data).

## 4. Robot dog (prismatic / force diagnostic, 80:140:2, cam_001+cam_002)

`NEXT_C3_rawD_policy`:
* The decision changes from `prismatic` (naive z −3.4) to **ambiguous, Occam prismatic**.
  Cluster-robust z is −1.2 over 983 groups: the old label rested on correlated tracks.
* The revolute is not identifiable (hinge 2.1 m away, 1.05 σ_px).
* P is 9.5 cm.
* Stage D: a sustained force event at source frames 152–260 (peak 11.6 N); 22/30 clip frames are
  `contact_probable`, none confirmed; the force path abstains (unverified gates).

## 5. Orange suitcase (ambiguity control)

* Not re-fit in this sprint. The previous run was ambiguous with Occam prismatic (5 m hinge).
* Stage D:
  * Sustained force event at source frames 122–229 (peak 4.2 N); 9/36 clip frames
    `contact_probable`.
  * The **tare gate PASSES**: there are two separated contact-free windows, with drift 0.028 N
    against a tolerance of 0.22 N.
  * The other gates are unverified, so the force path abstains.

## 6. Red fridge vision under the new policy

`NEXT_C3_rawD_policy` (same tracks and fold as `vision_cam_001_cam_002_20261117_111126`):

* **revolute, type confidence high.** Cluster-robust R-vs-P z = 5.5 over 1 453 groups (naive 24.1);
  group bootstrap p(R better) = 1.0.
* 18.9° about a hinge 0.31 m from the support median.
* Geometry identifiable:
  * R differs from its best translation by 17.7 σ_px;
  * the hinge line is 0.08 m from the moving support (extent 0.43 m).
* P candidate 10.4 cm; 7 134 / 6 366 moving tracks (R / P).

The positive control survives the stricter policy. Correlated-track inference shrinks z about
4.4×, but leaves the decision unchanged.

## 7. Synthetic gates

| gate | test | result |
|---|---|---|
| A | wrong-surface depth injected into one camera: classified non-random (> 80%), localized to that camera; true joint preferred on clean data | PASS |
| B | depth rules flag injected wrong-surface samples (> 70%) with < 10% false flags; coherent patch bias **not** detectable (documented limitation); static object with camera bias + per-track offsets: no motion | PASS (limitation stated) |
| C | drawer-like 10 cm motion on a 3 m arc (1.9° play): not a confident revolute, hinge not reported; real 8° near hinge: accepted and identifiable; correlated groups: naive z overstated, cluster z not | PASS |
| D | rotated `T_EE_S` + sign + tare transforms exact; unknown rotation or sign refuses; declared ≠ verified; tare drift needs two windows; multipose gravity recovers mass; torque lever arm | PASS |
| E | sliding revolute with calibrated force: S3/S4 change parameters, true type wins, μ recovered (0.3); sign-reversed wrench: S4 abstains (goodness of fit) while contact stays; time-shifted wrench fits worse; uncalibrated → contact only; heuristic or no annotations → ABSTAIN with vision unchanged; hidden face → CONTACT_NORMAL_UNAVAILABLE; prismatic sliding never yields a revolute | PASS |
| F | rasterizer covariance R Σ Rᵀ (xyzw/wxyz swap detected); metric depth = camera-Z (2 mm); virtual-camera equivalence for RGB/alpha/depth; occlusion and robot pixels excluded; dL/dq vs FD (5%); frozen scoring prefers true R, static scene favours static | PASS (synthetic); real: §1.5, §2 |
