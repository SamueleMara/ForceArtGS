#!/usr/bin/env bash
# Joint-aware reconstruction on the real scenes (both R and P branches per scene; no vanilla stages).
#   bash pipelines/joint_aware_real_scenes.sh [budget smoke|coarse|full] [out_root]
# Env: SCENES="name ..." subset, STAGES (default all), DISC_ROOT (overnight discovery outputs),
#      EXPORT_ARGS="--export-parttwin [--export-diagnostic] [--dynamics-spec S.yaml]" for the G8 PArtTwin1D export.
# Reuses each scene's 2026-11-18 overnight discovery run (cluster-robust z + observability policy, cam_001+cam_002),
# its AllTracker observation tables (dense seeds) and robot archive (tool exclusion). Outputs on the SSD only.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BUDGET="${1:-coarse}"
OUT="${2:-/media/samuelemara/Extreme SSD/Samuele/results_$(date +%d_%m_%y)/forceartgs_joint_aware}"
PY="$HOME/miniconda3/envs/freeartgs/bin/python"
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
DISC_ROOT="${DISC_ROOT:-/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint}"
JR=outputs/joint_discovery_runs
STAGES="${STAGES:-all}"
TABLE=(
  "final_red_fridge_episode_000000|90:148:2|$JR/final_red_fridge_episode_000000/observations_90-148-2"
  "final_black_box_episode_000000|70:128:2|$JR/final_black_box_episode_000000/observations_70-128-2"
  "white_drawer_20261021_161154_episode_000000|150:330:6|$JR/white_drawer_20261021_161154_episode_000000/observations_stride8"
  "final_robot_dog_episode_000000|80:140:2|$JR/final_robot_dog_episode_000000/observations_80-140-2"
  "final_orange_suitecase_episode_000000|50:120:2|$JR/final_orange_suitecase_episode_000000/observations_50-120-2"
)
mkdir -p "$OUT/.runtime"
SEED="/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/train/.runtime"
for d in torch huggingface torch_extensions; do
  [[ -d "$SEED/$d" && ! -e "$OUT/.runtime/$d" ]] && cp -r "$SEED/$d" "$OUT/.runtime/$d"
done
LOG="$OUT/joint_aware_batch.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "start budget=$BUDGET stages=$STAGES git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) submodule_dirty=$(git -C reconstruction/nerfstudio status --porcelain | wc -l)"
for row in "${TABLE[@]}"; do
  IFS='|' read -r name frames obs <<< "$row"
  [[ -n "${SCENES:-}" && " $SCENES " != *" $name "* ]] && continue
  disc="$DISC_ROOT/$name/NEW_cluster_obs_cam001_002"
  log "$name start"
  "$PY" pipelines/joint_aware_reconstruction.py --scene "$REAL/$name" --frames "$frames" --cameras cam_001 cam_002 \
      --output-root "$OUT" --discovery-run "$disc" --observations "$obs/obs_cam_001.npz" "$obs/obs_cam_002.npz" \
      --robot-archive "$DISC_ROOT/$name/robot_archive.npz" --stage "$STAGES" --budget "$BUDGET" ${EXPORT_ARGS:-} \
      >> "$OUT/$name.$BUDGET.log" 2>&1
  rc=$?
  IFS=: read -r fa fb fk <<< "$frames"
  sel="$OUT/${name}_multiview_cams001-002_i${fa}-${fb}_fall_s${fk:-1}_joint_aware_${BUDGET}/selection.json"
  log "$name exit $rc $( [[ -f "$sel" ]] && "$PY" -c "import json,sys; s=json.load(open(sys.argv[1])); print(s['status'], s.get('selected',''))" "$sel")"
done
log "done; per-scene run_manifest.json / selection.json under $OUT"
