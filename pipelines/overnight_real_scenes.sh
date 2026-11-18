#!/usr/bin/env bash
# Overnight batch over the real scenes that have an established motion clip:
#   stream J (CPU): new joint estimator (cluster-robust z + observability policy) on cam_001+cam_002,
#                   a cam_000+001+002 diagnostic fit for black box / orange suitecase only,
#                   and the full-episode contact/wrench audit (D1); summary table at the end.
#   stream T (GPU): full RGB-D FreeArtGS multiview training (train_parts -> twopart_blend -> splatfacto_art,
#                   30k iterations each, camera optimizer off) on the same clip, cam_001+cam_002.
# The two streams run in parallel (joint fits are CPU-only because the observation tables are reused).
#
# NOTE: training does NOT consume the joint-discovery result yet (Stage G not wired); it runs the
# legacy Phase-1 tracking/ICP motion. Both outputs land side by side for comparison.
#
#   bash pipelines/overnight_real_scenes.sh [out_root]          (resumable: finished steps are skipped)
# out_root defaults to "/media/samuelemara/Extreme SSD/Samuele/results_<dd_mm_yy>/overnight_real" and must be on the SSD.
# Env: STREAMS="J T" (default both), ITERATIONS (default 30000), SCENES (subset of names below).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${1:-/media/samuelemara/Extreme SSD/Samuele/results_$(date +%d_%m_%y)/overnight_real}"  # outputs always on the SSD
case "$ROOT" in "/media/samuelemara/Extreme SSD/"*) ;; *) echo "refusing: output root must be on the Extreme SSD: $ROOT" >&2; exit 2;; esac
PY="$HOME/miniconda3/envs/freeartgs/bin/python"
CLI=("$PY" -m estimotion_2part.joint_discovery.cli)
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
CFG=configs/joint_discovery.real_tcp2cm.yaml
POL=(--uncertainty cluster_robust --remote-hinge-policy observability)
ITER="${ITERATIONS:-30000}"
STREAMS="${STREAMS:-J T}"
JR=outputs/joint_discovery_runs
# name | clip a:b (inclusive RGB-D frames) | stride | existing observation dir | diagnostic cam_000 fit
TABLE=(
  "final_black_box_episode_000000|70:128|2|$JR/final_black_box_episode_000000/observations_70-128-2|diag"
  "final_orange_suitecase_episode_000000|50:120|2|$JR/final_orange_suitecase_episode_000000/observations_50-120-2|diag"
  "final_red_fridge_episode_000000|90:148|2|$JR/final_red_fridge_episode_000000/observations_90-148-2|-"
  "final_robot_dog_episode_000000|80:140|2|$JR/final_robot_dog_episode_000000/observations_80-140-2|-"
  "white_drawer_20261021_161154_episode_000000|150:330|6|$JR/white_drawer_20261021_161154_episode_000000/observations_stride8|-"
)
mkdir -p "$ROOT/joint" "$ROOT/train/.runtime"
# Seed model weights / compiled extensions from the previous real batch (copy, never overwrite, source untouched).
SEED="/media/samuelemara/Extreme SSD/Samuele/results_10_11_26/freeart_real/.runtime"
for d in torch huggingface torch_extensions; do
  [[ -d "$SEED/$d" && ! -e "$ROOT/train/.runtime/$d" ]] && cp -r "$SEED/$d" "$ROOT/train/.runtime/$d"
done
LOG="$ROOT/overnight.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
selected() { [[ -z "${SCENES:-}" || " $SCENES " == *" $1 "* ]]; }

stream_joint() {
  local runs=()
  for row in "${TABLE[@]}"; do
    IFS='|' read -r name clip stride obs diag <<< "$row"; selected "$name" || continue
    local S="$REAL/$name" O="$ROOT/joint/$name"; mkdir -p "$O"
    [[ -f "$O/robot_archive.npz" ]] || "${CLI[@]}" robot-archive --scene "$S" --output "$O/robot_archive.npz" >> "$O/robot_archive.log" 2>&1
    local id=NEW_cluster_obs_cam001_002
    if [[ ! -f "$O/$id/summary.json" ]]; then
      rm -rf "$O/$id"; log "J $name $id start"
      "${CLI[@]}" fit --scene "$S" --config $CFG "${POL[@]}" --cameras cam_001 cam_002 \
          --observations "$obs/obs_cam_001.npz" "$obs/obs_cam_002.npz" --experiment-id $id --output "$O/$id" > "$O/$id.log" 2>&1
      log "J $name $id exit $?"
    fi
    runs+=("$O/$id")
    if [[ "$diag" == diag ]]; then
      id=NEW_cluster_obs_cam000_001_002_diag
      if [[ ! -f "$O/$id/summary.json" ]]; then
        rm -rf "$O/$id"; log "J $name $id start"
        "${CLI[@]}" fit --scene "$S" --config $CFG "${POL[@]}" --cameras cam_000 cam_001 cam_002 \
            --observations "$obs/obs_cam_000.npz" "$obs/obs_cam_001.npz" "$obs/obs_cam_002.npz" \
            --experiment-id $id --output "$O/$id" > "$O/$id.log" 2>&1
        log "J $name $id exit $?"
      fi
      runs+=("$O/$id")
    fi
    if [[ ! -d "$O/D1_contact_windows" ]]; then
      local a=${clip%%:*} b=${clip##*:}
      "$PY" samuele/audit_joint_contact_windows.py --scene "$S" --clip-frames "$a:$b:$stride" \
          --cameras cam_001 cam_002 --output "$O/D1_contact_windows" > "$O/D1.log" 2>&1
      log "J $name D1 exit $?"
    fi
  done
  local done_runs=(); for r in "${runs[@]}"; do [[ -f "$r/summary.json" ]] && done_runs+=("$r"); done
  (( ${#done_runs[@]} )) && "${CLI[@]}" summarize --runs "${done_runs[@]}" > "$ROOT/joint/summary.txt" 2>&1
  log "J stream done ($ROOT/joint/summary.txt)"
}

stream_train() {
  for row in "${TABLE[@]}"; do
    IFS='|' read -r name clip stride obs diag <<< "$row"; selected "$name" || continue
    log "T $name start (clip $clip stride $stride, $ITER it/stage)"
    bash pipelines/train_local_datasets.sh --datasets "$REAL" --objects "$name" --cameras cam_001 cam_002 \
        --frame-intervals "$clip" --frame-stride "$stride" --iterations "$ITER" \
        --outputs-root "$ROOT/train" --skip-completed --visualize-results >> "$ROOT/train/driver.log" 2>&1
    log "T $name exit $?"
  done
  log "T stream done ($ROOT/train/batch_logs/summary.json)"
}

log "overnight start: root=$ROOT streams=[$STREAMS] git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
pids=()
[[ " $STREAMS " == *" J "* ]] && { stream_joint & pids+=($!); }
[[ " $STREAMS " == *" T "* ]] && { stream_train & pids+=($!); }
wait "${pids[@]}"
log "overnight finished"
