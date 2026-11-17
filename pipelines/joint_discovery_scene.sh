#!/usr/bin/env bash
# Full joint-estimation pipeline for one real final_* scene (read-only on the scene):
#   robot archive -> AllTracker partial tracks per camera (robot tool removed) ->
#   vision-only fit (selection) -> same fit + TCP contact / force evidence -> summary.
#
#   bash pipelines/joint_discovery_scene.sh <scene_dir> <rgbd_frames start:end:step> <cam,cam,...> [out_root]
#
# Env: PYTHON (default: freeartgs env), CONFIG (default: configs/joint_discovery.real_tcp2cm.yaml),
#      FORCE_CONTACT_N (default 1.0; heuristic contact flags |F - tare| > N, tare = model frames 0:3).
set -euo pipefail
SCENE="$1"; FRAMES="$2"; CAMS="$3"
NAME=$(basename "$SCENE")
OUT="${4:-outputs/joint_discovery_runs/$NAME}"
PY="${PYTHON:-$HOME/miniconda3/envs/freeartgs/bin/python}"
CONFIG="${CONFIG:-configs/joint_discovery.real_tcp2cm.yaml}"
FORCE_N="${FORCE_CONTACT_N:-1.0}"
CLI="$PY -m estimotion_2part.joint_discovery.cli"
IFS=',' read -ra CAM_LIST <<< "$CAMS"
TAG=$(IFS=_; echo "${CAM_LIST[*]}")
STAMP=$(date +%Y%m%d_%H%M%S)
OBS="$OUT/observations_${FRAMES//:/-}"
mkdir -p "$OBS"

$CLI robot-archive --scene "$SCENE" --output "$OUT/robot_archive.npz"
OBS_FILES=()
for cam in "${CAM_LIST[@]}"; do
  f="$OBS/obs_$cam.npz"
  [[ -f "$f" ]] || $CLI extract --scene "$SCENE" --camera "$cam" --frames "$FRAMES" \
                       --tcp-npz "$OUT/robot_archive.npz" --output "$f"
  OBS_FILES+=("$f")
done
$CLI fit --scene "$SCENE" --config "$CONFIG" --cameras "${CAM_LIST[@]}" --observations "${OBS_FILES[@]}" \
    --output "$OUT/vision_${TAG}_$STAMP"
$CLI fit --scene "$SCENE" --config "$CONFIG" --cameras "${CAM_LIST[@]}" --observations "${OBS_FILES[@]}" \
    --tcp-npz "$OUT/robot_archive.npz" --contact-force-threshold-n "$FORCE_N" --tare-frames 0:3 \
    --output "$OUT/tcp_force_${TAG}_$STAMP"
$CLI summarize --runs "$OUT/vision_${TAG}_$STAMP" "$OUT/tcp_force_${TAG}_$STAMP"
echo "reports: $OUT/vision_${TAG}_$STAMP/report.md  $OUT/tcp_force_${TAG}_$STAMP/report.md"
