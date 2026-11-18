# Joint-aware reconstruction (ForceArtGS frontend for FreeArtGS)

Opt-in. The legacy FreeArtGS pipeline (`pipelines/freeartgs.sh`, `train_local_datasets.py`) is unchanged and
remains the default. The joint-aware path replaces the vanilla frontend: per-camera part segmentation,
ICP motion, reference-camera propagation, free per-frame object poses and the rotation-span joint classifier.

## Call graph

| Vanilla stage | Joint-aware replacement | Where |
|---|---|---|
| `track_2part.py` / `segment.py` masks | dense S/M/U support from the authority (cross-time RGB-D reprojection) | `estimotion_2part/joint_discovery/dense_support.py` |
| `icp/color_icp_gpu.py` `global_poses_*` | none: exact joint transforms | `integration_contract.JointAuthority.transform` |
| `prepare_multiview.py` (reference camera) | `build_joint_aware_dataset`: part0 = static physical cameras, part1 = `A(q_t)^-1 T_WC` | `preprocessing/prepare_joint_aware_multiview.py` |
| `train_parts` binary mask loss | unknown-aware supervision (`--pipeline.model.joint-aware-supervision True`, `part-data --joint-aware True`) | submodule `models/joint_aware_losses.py`, `models/splatfacto.py`, `data/datasets/depth_dataset.py`, `data/dataparsers/part_dataparser.py` |
| `twopart_blend` free 6-DoF object poses | `--pipeline.model.joint-aware True --obj-pose-opt-enabled False`; the initial poses `C_ref C_mot^-1` are exactly `A_B(t)`, checked after training | submodule `models/twopart_blend.py`; controller `verify_blend` |
| `estimate_joint.py` (span < 10° ⇒ prismatic) | never runs; `estimated_joint.json` derived from the authority, hashed in `JOINT_AWARE_AUTHORITY.json` | `integration_contract.derive_estimated_joint`; submodule `splatfacto_art_dataparser.py --joint-aware True` |
| splatfacto-art axis/origin/q/offset optimisation | all frozen; export checked against the authority | controller `verify_art` |

Tripwires: with `FORCEARTGS_JOINT_AWARE_TRIPWIRE=1`, which the controller sets, `track_2part.py`, `segment.py`,
`icp/color_icp_gpu.py`, `prepare_multiview()` and `estimate_joint.py` raise. `estimate_joint.py` also refuses any
blend directory that contains `JOINT_AWARE_AUTHORITY.json`.

## Frames

World W is the `metadata/cameras.json` frame (OpenCV c2w, metres), identical to COLMAP `images.txt` on the real
scenes. All three nerfstudio stages run with `orientation none`, no centring, no auto-scale and scale 1, so no
similarity is applied. The two-part blend still re-anchors both parts with `N = inv(C[first image])`. Because the
authority has `q[0] = 0`, `A(t0) = I` and both parts receive the same `N`. The controller recovers `N_BW` from the
exported cameras, requires one rigid `N` for every image, and expresses the adapter axis, origin and q in B.
On the real scenes cam_001 is the world origin, so `N_BW = I`.

## Dense support (G1)

Each object pixel with valid depth is carried to training images at other times as static (`x`) and as moving
(`A(s)A(t)^-1 x`). Visibility classes: SUPPORTED / OCCLUDED / FREE_SPACE / UNKNOWN. A target is scored only when
it is discriminative (≥ 2 px or ≥ 3σ apart) and both predictions make a visible claim (supported or free-space),
so hiding points cannot win. Costs are Cauchy depth and colour residuals. Their sigmas combine sensor noise with
the projected pixel uncertainty times the local depth or colour gradient. Evidence is pooled over a 5×5 window,
confident discovery tracks add a bounded seed prior, and fixed thresholds decide the label; everything else is
UNKNOWN. Held-out images are never used as targets. This is a single pass: iterative re-labelling and SAM2
proposals are not implemented.

## Status labels

`PASSED`, `FAILED`, `NOT_RUN`, `ABSTAIN`, `IDENTIFIED`, `AMBIGUOUS` are recorded per stage in
`<out>/<scene>/run_manifest.json`. Branch selection is in `selection.json`. `final/` is written only for an
identified branch.

## Not implemented (explicitly NOT_RUN)

* G6 alternating membership/joint refinement with the full renderer (`refine` stage records `NOT_RUN`).
  Joints therefore stay at the discovery authority (version v000).
* G7 robot contact/force terms inside reconstruction. Real scenes have no verified contact or wrench
  calibration, so these would abstain anyway.
* A static-branch rendering baseline, held-out-camera folds, SAM2 proposals, transmittance-preserving soft
  opacity, Gaussian-group counterfactual assignment.
