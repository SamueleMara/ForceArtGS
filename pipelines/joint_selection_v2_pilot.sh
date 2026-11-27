#!/usr/bin/env bash
# Joint selection v2 real-scene pilot (spec Gate 8): re-uses the 2026-11-17 discovery observation tables, writes NEW
# run directories on the SSD, never edits the source runs.
#   bash pipelines/joint_selection_v2_pilot.sh [out_root]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${FORCEARTGS_PYTHON:-$HOME/miniconda3/envs/forceartgs/bin/python}"
eval "$("$PY" "$(dirname "${BASH_SOURCE[0]}")/../utils/forceartgs_env.py")"   # clean env: forceartgs + repo only
OUT="${1:-/media/samuelemara/Extreme SSD/Samuele/results_$(date +%d_%m_%y)/joint_selection_v2/pilot}"
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
DISC="${DISC_ROOT:-/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint}"
CFG="${CFG:-configs/joint_discovery.real_tcp2cm.yaml}"
JR=outputs/joint_discovery_runs
TABLE=(
  "final_red_fridge_episode_000000|$JR/final_red_fridge_episode_000000/observations_90-148-2"
  "white_drawer_20261021_161154_episode_000000|$JR/white_drawer_20261021_161154_episode_000000/observations_stride8"
  "final_black_box_episode_000000|$JR/final_black_box_episode_000000/observations_70-128-2"
  "final_robot_dog_episode_000000|$JR/final_robot_dog_episode_000000/observations_80-140-2"
  "final_orange_suitecase_episode_000000|$JR/final_orange_suitecase_episode_000000/observations_50-120-2"
)
mkdir -p "$OUT"
run() {  # name tag args...
  local name=$1 tag=$2; shift 2
  local o="$OUT/$name/$tag"
  [[ -f "$o/joint_selection_v2.json" ]] && { echo "skip $name/$tag"; return; }
  [[ -d "$o" ]] && rmdir "$o" 2>/dev/null
  mkdir -p "$OUT/$name"
  "$PY" -m estimotion_2part.joint_discovery.cli select-v2 --scene "$REAL/$name" --config "$CFG" --output "$o" "$@" \
     > "$OUT/$name/$tag.log" 2>&1
  echo "$(date '+%F %T') $name/$tag exit $? $(grep -h '^\[v2\] [A-Z_]* (working' "$OUT/$name/$tag.log" | tail -1)" | tee -a "$OUT/pilot.log"
}
for row in "${TABLE[@]}"; do
  IFS='|' read -r name obs <<< "$row"
  [[ -n "${SCENES:-}" && " $SCENES " != *" $name "* ]] && continue
  tcp=(); [[ -f "$DISC/$name/robot_archive.npz" ]] && tcp=(--tcp-npz "$DISC/$name/robot_archive.npz")
  run "$name" cam001_002 --from-run "$DISC/$name/NEW_cluster_obs_cam001_002" "${tcp[@]}"
  [[ -d "$DISC/$name/NEW_cluster_obs_cam000_001_002_diag" ]] && \
    run "$name" cam000_001_002_diag --from-run "$DISC/$name/NEW_cluster_obs_cam000_001_002_diag" "${tcp[@]}"
  for c in 001 002; do
    [[ -f "$obs/obs_cam_$c.npz" ]] && run "$name" "cam${c}_only" --observations "$obs/obs_cam_$c.npz" "${tcp[@]}"
  done
done
echo "$(date '+%F %T') done" | tee -a "$OUT/pilot.log"
