# All-real-dataset registration findings — 2026-11-12

**Acquisition-group update:** the operator subsequently confirmed that all `final_*`
datasets were recorded in one go. The calibration-hash pool below mixed these with
earlier acquisitions. Its final-only replacement is in
`results_11_10_26/ee_tcp_registration_final_session_20261112`; see the acquisition
group section of `docs/robot_registration_all_datasets.md`. The original pool is
retained as a historical baseline and does not prove separate final-session camera
configurations. The final-only shared fit also remains unverified.

The audit finished with **118/118 scenes, all three cameras and all mapped RGB-D
frames**: 71,154 attempted images, 29,223 accepted shaft observations (41.1%), and
zero extraction failures. There are 354 scene/camera coverage records; 74 have
no accepted shaft observation. Low coverage is retained, not interpreted as zero
error. The code and ten synthetic/control tests are installed in Sapien, with
copies in FreeArtGS.

All results below are image-line registration diagnostics, **not measured absolute
TCP/contact-point accuracy**. No original calibration, image, depth, mask, cloud,
robot measurement or production pipeline was changed.

## Combining all recordings does not justify one universal correction

There are eleven native recordings and two distinct supplied camera calibration
hashes. The October-16 fridge has its own configuration. The other 117 scenes share
identical supplied matrices, but that alone does not establish physical stability.

The pooled fit for the newer calibration is mathematically observable and lowers
median errors on every withheld native recording. However, it fails the camera/tail
and bootstrap stability checks. Its bootstrap translation standard deviations are
approximately **15.1, 20.1 and 6.4 mm**, and the rotation-vector Z component varies
by **2.33 degrees**. It remains `UNVERIFIED_COMMON_MOUNT_HYPOTHESIS`.

The following scores withhold the *entire native recording* from fitting the
candidate used to evaluate that row:

| Withheld recording | Median before → after (px) | p95 before → after (px) |
|---|---:|---:|
| Earlier dog | 19.50 → 3.54 | 19.53 → 3.75 |
| October-21 mixed objects, usable pointed-tool observations | 16.48 → 3.84 | 23.71 → 11.02 |
| Final black box | 26.25 → 4.02 | 38.87 → 43.27 |
| Final black laptop | 4.74 → 3.60 | 8.03 → 9.45 |
| Final light blue box | 20.41 → 1.39 | 40.20 → 62.52 |
| Final matchbox | 25.48 → 1.56 | 26.55 → 6.32 |
| Final orange suitcase | 17.54 → 3.18 | 37.00 → 5.26 |
| Final red fridge | 22.71 → 2.20 | 32.57 → 15.73 |
| Final robot dog | 21.70 → 2.57 | 39.34 → 30.68 |
| Final wooden box | 28.21 → 8.10 | 36.53 → 11.03 |

A lower median alone would hide degraded tails in several objects. These results
do not distinguish physical mounting changes from detector mistakes, different
tool appearances, timing errors or a mixture of causes. They reject automatic
installation of a single pooled transform.

## Red fridge has a substantially stronger recording-specific candidate

Fit on nine red-fridge episodes, then test on five entirely withheld episodes:
`000001`, `000004`, `000007`, `000010`, `000013`.

| Camera | Withheld observations | Median before → after (px) | p95 before → after (px) |
|---|---:|---:|---:|
| cam_000 | 1,018 | 21.48 → 0.95 | 25.16 → 2.07 |
| cam_001 | 1,210 | 19.65 → 0.36 | 24.15 → 1.40 |
| cam_002 | 1,067 | 30.18 → 0.66 | 33.84 → 1.77 |
| Combined | 3,295 | **22.50 → 0.60** | **32.52 → 1.77** |

This candidate passes the defined image checks, including full scaled Jacobian
rank, condition number 193, camera coverage and 24/24 successful cluster-bootstrap
fits. The correction left-multiplies the camera-to-base transforms. Its translation
part is `[-1.181, +0.188, +34.418] mm`; its rotation vector is
`[+0.1915, +0.0171, +0.5657] degrees`. These are components of one rigid transform
about the robot base origin, not a local tool offset or standalone Z patch.

Conditional bootstrap translation standard deviations are `[1.69, 1.52, 1.28] mm`.
The 2.5–97.5 percentile translation ranges are approximately X `[-3.48, +2.57]`,
Y `[-3.01, +2.53]`, Z `[32.51, 36.94] mm`. Detector systematic errors are not
included in those intervals. The exact matrix and all intervals are in
`candidate_corrections.json` under `final_red_fridge__bdd8801c`.

This replaces the earlier background-derived transform **as the better-supported
robot-line candidate for this recording**, not as a certified production calibration.
It uses many more independent episodes, all cameras, varied poses and a direct
robot anchor rather than treating the older candidate calibration as truth. The
small perpendicular residual does not independently validate the TCP position
along the shaft, physical mounting stability, timing or the tool reference.

## Other recordings and the remaining limits

- Final matchbox, orange suitcase, robot dog and black laptop have recording-specific
  fits that improve their overall withheld medians, but at least one uncertainty,
  camera-coverage or residual check fails. They remain `UNVERIFIED`.
- Final light-blue box, the older fridge and the October-21 mixed recording have
  inadequate six-DOF observability or a fit at the allowed bounds. They remain
  `UNIDENTIFIABLE`. The older fridge already has a withheld median of 2.43 px;
  its unconstrained fitted candidate raises that to 2.63 px and must not be applied.
- Final black box, wooden box and earlier dog lack accepted observations in all
  three cameras for the recording-specific fit. They remain explicitly unverified.
- The old black monitor has no usable shaft lines. The three early white-drawer
  episodes have a different bare-sensor assembly and are explicitly unsupported by
  the pointed-shaft detector. Their different mounting is not corrected by assumption.
- Visual inspection confirms that some black-box detections can select a dark edge
  near the tool instead of its axis. More image-line samples alone will not repair
  this systematic correspondence error. Those views need geometry appropriate to
  the actual visible assembly, such as paired shaft edges or flange/tip landmarks.

The separate background control performed 81 image comparisons using three frames
per native recording/camera against an October-21 black-box anchor. Seven were
consistent within the 1.5 px identity threshold, 32 showed displacement or ambiguous
matches, and 42 lacked enough distinctive matches. Several newer recordings have
cam_000 displacements around 11.5–12.5 px and cam_001 around 2–3.5 px relative to
that anchor, despite identical supplied calibration matrices. This is evidence
against blindly assuming identical image configurations, **not proof of camera
motion**: movable background and incorrect feature identities remain alternatives.

The next accuracy improvement should therefore use the red-fridge candidate as a
recording-specific hypothesis and improve/verify tool observations for the other
assemblies, while checking physical configuration changes independently. Do not
add arbitrary EE-local X/Y offsets to absorb image discrepancies. No new manual
annotation or calibration capture was required to complete this audit.

## Files to inspect

Results root:

```
/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ee_tcp_registration_all_20261112
```

Open `index.html` for recording-fit and pooled-fit images, and `REPORT.md` for the
generated full summary. A useful fully withheld example is
`final_red_fridge_episode_000001/candidate_overlay.jpg`: cyan nominal axis, yellow
fitted axis, green observed shaft. `coverage.json`, per-scene `observations.jsonl`,
`pooled_common_mount_hypothesis.json`, `background_stability.json` and
`scene_camera_metrics.csv` preserve the numerical evidence and failures.

Commands and method details are documented in `docs/robot_registration_all_datasets.md`
in Sapien. A common rigid camera/world correction does not change robot-base
force/trajectory percentages or internal multiview rendering geometry.
