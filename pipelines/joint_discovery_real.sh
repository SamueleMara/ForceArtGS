#!/usr/bin/env bash
# Real-data runs for the opt-in joint-aware Phase 1 (estimotion_2part/joint_discovery).
# Read-only on the scene; every run writes a fresh directory under outputs/joint_discovery_runs/.
#
#   bash pipelines/joint_discovery_real.sh <scene_dir> <rgbd_frames start:end:step> <tcp_npz> <out_root> [extract]
#
# Runs: B4 (cam_001+cam_002, default), B3-2D (same, depth disabled), B9 (+cam_000 stress),
# cam_000-only (explicit single-camera diagnostic), and S3/S4 (B4 + TCP/wrench, reported separately).
# OBS_DIR overrides the observation directory (default <out_root>/observations).
set -euo pipefail
SCENE="$1"; FRAMES="$2"; TCP="$3"; OUT="$4"; DO_EXTRACT="${5:-}"
PY="${PYTHON:-python}"
CLI="$PY -m estimotion_2part.joint_discovery.cli"
OBS="${OBS_DIR:-$OUT/observations}"
mkdir -p "$OBS"
if [[ "$DO_EXTRACT" == "extract" ]]; then
  for cam in cam_000 cam_001 cam_002; do
    $CLI extract --scene "$SCENE" --camera $cam --frames "$FRAMES" --tcp-npz "$TCP" --output "$OBS/obs_$cam.npz"
  done
fi
STAMP=$(date +%Y%m%d_%H%M%S)
run() { local name="$1"; shift; $CLI fit --scene "$SCENE" --output "$OUT/${name}_$STAMP" "$@"; }
run B4_cam001_002 --observations "$OBS/obs_cam_001.npz" "$OBS/obs_cam_002.npz"
run B3_2Donly_cam001_002 --no-depth --observations "$OBS/obs_cam_001.npz" "$OBS/obs_cam_002.npz"
run B9_cam000_001_002 --cameras cam_000 cam_001 cam_002 \
    --observations "$OBS/obs_cam_000.npz" "$OBS/obs_cam_001.npz" "$OBS/obs_cam_002.npz"
run diag_cam000_only --cameras cam_000 --observations "$OBS/obs_cam_000.npz"
run S3S4_cam001_002_tcp --observations "$OBS/obs_cam_001.npz" "$OBS/obs_cam_002.npz" \
    --tcp-npz "$TCP" --contact-force-threshold-n 0.3 --tare-frames 0:3
