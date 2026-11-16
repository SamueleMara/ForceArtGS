# EE/TCP registration experiment: actual method and proposed extension

Date: 2026-11-12. This documents an exploratory registration hypothesis, not a certified replacement calibration. No production camera transforms, raw measurements or point clouds were changed.

The numerical artifacts and reproduction scripts are in `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ee_tcp_alignment_check_20261112`.

## What was checked first

The screenshot was identified as `final_red_fridge_episode_000000`, native Parquet frame 268 / converted RGB-D frame 134. Its measured force components and timestamp match the displayed values.

I compared it with `fridge_plastic_20261016_163314`, using frame 134 from each recording. Equal frame indices do not mean the sessions were synchronized; the cross-session estimator uses static background only.

For both recordings, the cached Cartesian EE position and rotation vector match the Parquet fields exactly. Both use the same pointed-tip transform:

```
p_base_tcp = p_base_link6 + R_base_link6 @ [0, 0, 0.1876174]
```

Across all 118 cached scenes, the maximum local X/Y component of the computed EE-to-TCP displacement is about `8.3e-14 mm`, i.e. numerical roundoff. No local X/Y correction, table-height shift or FK replacement was added. Image Y coordinates need not be equal for points sharing a local Z axis: robot orientation and perspective still apply.

The two recordings have identical native RGB intrinsics/distortion and rectified intrinsics, but different camera-to-base transforms. The supplied camera-center/rotation differences are:

| Camera | Center displacement | Rotation difference |
|---|---:|---:|
| cam_000 | 45.75 mm | 8.28 degrees |
| cam_001 | 221.89 mm | 28.16 degrees |
| cam_002 | 86.33 mm | 1.63 degrees |

These differences are expected for different physical configurations. They are not, by themselves, evidence of a bad calibration. The older source is an October-16 candidate marked `CANDIDATE_NOT_VALIDATED`; the newer source is an October-21 model marked `VALIDATED_CURRENT_CONFIGURATION`.

## The actual background experiment

For each corresponding camera, I excluded the appliance mask, dilated by 25 pixels, image borders and a generous projected robot/tool envelope. The EE pose helped define that exclusion envelope; robot observations were not targets in the pose fit.

I detected SIFT features on the remaining wall/table pixels, using a low contrast threshold for the weakly textured scene. Matches had to pass a descriptor ratio test and a mutual match check. A RANSAC homography with a 2-pixel threshold removed gross image outliers. That is a preliminary filter, not an assumption that the whole room is one plane.

For the cam_000 pair, the final 0.8 descriptor-ratio setting produced 59 mutual matches, 36 homography inliers and 33 usable RGB-D points. Each old-image feature was unprojected into the old optical-camera frame using the rectified intrinsics and the median of a 5-by-5 depth neighborhood. At least five valid depths were required, and patches spanning more than 25 mm were rejected.

PnP estimated the rigid mapping A from old-camera 3D coordinates into the new optical camera:

```
X_old = depth_old * inverse(K_old) * [u_old, v_old, 1]
pixel_new = project(K_new * A * X_old)
```

I used a random 75/25 point split. PnP RANSAC and iterative refinement retained 22 training inliers; nine points were withheld from PnP. Their median reprojection error was 0.79 pixels, compared with 17.62 pixels predicted by the supplied old/new camera-to-base models.

Important qualification: the homography inlier filter used all matches before that split. Thus the nine points are held out from the PnP optimization, but not from the complete correspondence-selection process. This is exploratory validation, not a strict independent benchmark.

Only cam_000 supplied enough reliable background points for this fit. Cam_001 and cam_002 had insufficient cross-session RGB-D matches and did not estimate the correction.

## How physical camera repositioning was handled

Let C_old and C_new map their respective optical-camera coordinates into robot-base coordinates. The measured A includes actual camera repositioning. If the robot-base frame and matched background refer to the same physical world across sessions:

```
A_reported = inverse(C_new) * C_old
C_new_from_old_reference = C_old * inverse(A_measured)
G = C_new_from_old_reference * inverse(C_new)
```

The diagnostic corrected camera pose is `G * C_new`. The old camera transforms were not directly substituted into the new configuration.

The candidate G has a rotation magnitude of about 1.96 degrees and translation part `[-15.5, +5.9, +29.6] mm`. These numbers describe a rigid transform about the chosen base origin. They are not a proposed local TCP offset or a standalone Z adjustment.

I applied the same G to all three newer camera poses in memory, solely to render diagnostic overlays. This preserves every relative camera-to-camera pose:

```
inverse(G * C_i) * (G * C_j) = inverse(C_i) * C_j
```

This explains why internal multiview reconstruction can look consistent while the entire rig is misregistered relative to robot poses.

## Robot observations used to test the candidate

I searched for the physical shaft centerline in a broad band around its approximate predicted location. The detector sampled cross-sections, accepted isolated dark runs 3–18 pixels wide, fitted a Huber line, and rejected fits with median absolute residual above 0.75 pixels. Multiframe validation additionally required at least 12 accepted cross-sections.

That detector is heuristic and uses the predicted axis to locate its search band. It is not completely calibration-independent, and it is not an exact TCP-tip detector.

Six converted frames were attempted per recording: 25, 75, 134, 180, 225 and 270. The shaft pixels were not used to fit G. In the newer red recording:

| Camera | Accepted frames out of 6 | Median line residual before | After diagnostic G |
|---|---:|---:|---:|
| cam_000 | 4 | 21.2 px | 3.1 px |
| cam_001 | 6 | 20.1 px | 2.5 px |
| cam_002 | 4 | 30.0 px | 2.9 px |

The old reference has about 2.6 pixels median residual in cam_000 and 0.8 pixels in cam_002 without correction. Applying this G to the old configuration worsens those to about 16.3 and 27.2 pixels. Old cam_001 had only one accepted frame and provides weak quantitative evidence.

The useful result is improved agreement on robot pixels, other cameras and additional frames that did not fit G. However, this still covers two recordings and a small range of robot orientations. A shaft line cannot, by itself, verify exact position along the shaft or the full tip calibration.

## Remaining uncertainties

- The old reference is not certified ground truth. Its error can propagate into G.
- Static background must actually be the same rigid room structure. Table motion or robot-base movement between sessions can invalidate the cross-session assumption.
- Features concentrate on limited wall/table structure; planar geometry and repeated texture can weaken translation/depth observability.
- Registered-depth bias can be absorbed into the estimated pose. A 5-by-5 median is a heuristic, not an uncertainty model.
- The held-out split occurs after geometric filtering. It should move earlier and use complete feature tracks/episodes.
- Match thresholds matter: changing the ratio threshold from 0.8 to 0.7 gave a 2.25-degree correction instead of 1.96 degrees and changed translation by several millimetres. No bootstrap confidence interval was calculated.
- The shaft extractor has restricted coverage and prediction-guided search. Rejected/ambiguous frames must remain visible in coverage statistics.
- Correcting a whole 3D line does not establish the precise link6 origin or TCP contact point along that line.
- Timing, pose-reference interpretation, common rig registration and per-camera errors were not jointly identified.

The proper current conclusion is that a common registration correction is promising for the tested newer recording. The experiment does not certify the original cause or justify automatically modifying all recordings.

## A more accurate all-dataset investigation

1. **Inventory actual configurations.** Group recordings using camera serials, native calibration hashes, timestamps where available, tool geometry and observed background motion. Although 117 datasets contain exactly the same October-21 extrinsics, this does not prove their physical camera configurations stayed identical.

2. **Measure robot geometry independently.** Detect shaft centerlines, visible flange centers and visible tips using RGB/depth evidence with uncertainty and ambiguity flags. Use the robot projection only as a generous proposal region, not as a rule that chooses the best-agreeing line. Include varied EE orientations and stationary holds. Keep the user-confirmed centered link6/tool geometry fixed initially; do not add arbitrary local X/Y offsets to make overlays fit.

3. **Fit absolute robot registration directly where identifiable.** Use the known link6 poses and observed tool geometry across many frames and all cameras. For a centered shaft, the robot line is `p_link6(t) + s * R_link6(t) * e_z`; compare its image projection with the measured shaft line. Fit one six-DOF G per genuinely stable configuration, keeping relative camera calibration fixed initially. This avoids treating the October-16 candidate as the absolute anchor. Varying tool poses and additional flange/tip observations are needed to resolve weak directions. Compute Jacobian rank/conditioning and report unidentifiable parameters rather than returning an arbitrary solution.

4. **Use background as a separate control.** Estimate camera motion from distinctive static-room features across all available cameras, using both depths and multiple surfaces. Separate wall and table hypotheses; do not assume movable furniture is fixed. Preserve ambiguous/repeated-texture alternatives. Background-only and robot-based registration estimates should agree within uncertainty.

5. **Separate causes by their signatures.** A common base-registration error should be coherent across cameras. An individual-camera error should remain localized. A wrong EE/tool reference tends to rotate with EE orientation. Timing residuals correlate with linear/angular motion and can be tested on stationary holds. Depth errors vary with range, surface and sensor. Compare these hypotheses on held-out data instead of allowing all parameters to compensate for one another.

6. **Validate before preprocessing the validation set.** Hold out complete poses, frames, feature tracks, recordings and, where possible, a camera or object. Split before homography/PnP selection. Freeze thresholds before evaluating held-out robot geometry. Resample tracks/episodes for uncertainty, not neighboring feature pixels. Run synthetic controls with known rig perturbations, tool-origin errors, depth bias and timing offsets to test identifiability.

7. **Report all datasets, including failures.** Save per-frame/camera/object before/after errors, median and 95th percentile, match/pose coverage, rejected counts, stationary/dynamic subsets, uncertainty and model status. Missing usable observations mean UNVERIFIED/UNIDENTIFIABLE, not zero error or silent exclusion. A pilot should cover every native recording group and varied tool poses before attempting one shared correction on all 117 recordings.

8. **Accept only a stable improvement.** The current roughly 20–30 to 3 pixel reduction is a useful target to reproduce on independent poses/scenes, not a guaranteed final threshold. Require improved median and tail errors without hiding a degraded camera or low coverage; stable transforms across reasonable matching/depth choices; and adequate observability. If only tiny changes comparable to image/depth uncertainty remain, keep the existing model and report the limit.

These tests use available recordings and do not require new manual annotations or calibration captures as prerequisites. Additional independent measurements may be recommended only when existing data cannot resolve a parameter.

## What a validated correction would change

For a genuine common rig-to-base correction, write a new versioned dataset/output:

```
C_camera_to_base_new = G * C_camera_to_base_old
P_camera_derived_new = G * P_camera_derived_old
W_colmap_world_to_camera_new = W_old * inverse(G)
```

Apply it consistently to camera-derived PLY clouds, normals, all masked/unmasked initial/final COLMAP models and corresponding metadata. Preserve endpoint image membership. Intrinsics, RGB, camera-frame depth and masks remain unchanged for a purely global rigid correction. Robot-native link6 poses and base-frame forces remain unchanged: they are the reference, not camera-derived quantities. Keep original files and the correction manifest for comparison.

A common G does not alter internal multiview geometry or standalone rendering when cameras and points are transformed consistently. Its expected benefit is robot/scene georeferencing, overlay accuracy and cross-system coordinate comparisons. The force/EE relationship F_parallel, F_perp and eta does not use camera extrinsics, so it should not change under this camera-registration correction. An improvement in those robot-base force metrics would indicate another change or a bug, not a benefit of G.

If held-out evidence instead requires per-camera calibration changes, camera-relative geometry changes too; that is a different correction and needs re-fusion/re-evaluation rather than a common world transform.
