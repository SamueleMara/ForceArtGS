#!/usr/bin/env bash
# Multimodal frontend real-scene pilot (spec section 9E): frontend only, no 3DGS training. New SSD directories only.
#   V1 = RGB-D iterative (no robot)   V2 = V1 + TCP (tool occupancy + contact proposals; unknown contact unless an
#   operator/video annotation is given in contact_intervals/<scene>.yaml). V0 = the one-pass v2 pilot (separate).
#   bash pipelines/multimodal_frontend_pilot.sh [out_root]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$HOME/miniconda3/envs/freeartgs/bin/python"
OUT="${1:-/media/samuelemara/Extreme SSD/Samuele/results_$(date +%d_%m_%y)/multimodal_frontend/pilot}"
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
DISC="${DISC_ROOT:-/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint}"
CFG="${CFG:-configs/joint_discovery.real_tcp2cm.yaml}"
ITERS="${ITERS:-3}"
SCENES_ALL="final_red_fridge_episode_000000 white_drawer_20261021_161154_episode_000000 final_black_box_episode_000000 final_robot_dog_episode_000000 final_orange_suitecase_episode_000000"
mkdir -p "$OUT"
run() {  # name tag run_dir extra...
  local name=$1 tag=$2 src=$3; shift 3
  local o="$OUT/$name/$tag"
  [[ -f "$o/accepted/joint_selection_v2.json" ]] && { echo "skip $name/$tag"; return; }
  mkdir -p "$OUT/$name"
  "$PY" -m estimotion_2part.joint_discovery.cli multimodal-refine --scene "$REAL/$name" --from-run "$src" \
     --config "$CFG" --max-iterations "$ITERS" --output "$o" "$@" > "$OUT/$name/$tag.log" 2>&1
  echo "$(date '+%F %T') $name/$tag exit $? $(grep -h '^\[v2\] [A-Z_]* (working' "$OUT/$name/$tag.log" | tail -1)" | tee -a "$OUT/pilot.log"
}
for name in ${SCENES:-$SCENES_ALL}; do
  src="$DISC/$name/NEW_cluster_obs_cam001_002"
  run "$name" V1_rgbd_iterative "$src"
  if [[ -f "$DISC/$name/robot_archive.npz" ]]; then
    seg=(); [[ -f "configs/contact_intervals/$name.yaml" ]] && seg=(--contact-segments "configs/contact_intervals/$name.yaml")
    run "$name" V2_rgbd_tcp "$src" --robot-archive "$DISC/$name/robot_archive.npz" "${seg[@]}"
  else
    echo "$(date '+%F %T') $name/V2_rgbd_tcp NOT_RUN: no robot archive" | tee -a "$OUT/pilot.log"
  fi
done
echo "$(date '+%F %T') done" | tee -a "$OUT/pilot.log"
