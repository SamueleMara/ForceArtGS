#!/usr/bin/env bash
# ForceTwin-adaptation (identifiability-first effective I_eff / mu / b) on every existing real run. Append-only:
# <run>/dynamics/forcetwin_adaptation/vNNN. No calibration, no 3DGS retraining, no joint-authority change.
#   bash pipelines/g9_forcetwin_all_real.sh [RUN_ROOT]
# Env: PINN=1 also runs GeneralPINN (long); RESIDUAL=1 also runs the mechanism-residual ablations; SCENES subset.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS="${1:-/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware}"
PY="$HOME/miniconda3/envs/freeartgs/bin/python"
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
DISC_ROOT="/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint"
SCENES="${SCENES:-final_red_fridge_episode_000000 final_orange_suitecase_episode_000000 final_black_box_episode_000000 final_robot_dog_episode_000000 white_drawer_20261021_161154_episode_000000}"
LOG="$RUNS/g9_forcetwin_batch.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "start scenes=[$SCENES] PINN=${PINN:-0} RESIDUAL=${RESIDUAL:-0} git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for sc in $SCENES; do
  run=$(ls -d "$RUNS/${sc}"_multiview_*_joint_aware_full 2>/dev/null | head -1)
  [[ -z "$run" ]] && { log "$sc: no joint-aware run"; continue; }
  cmd=fit-minimal-dynamics
  args=(--run "$run" --scene "$REAL/$sc")
  [[ -f "$DISC_ROOT/$sc/robot_archive.npz" ]] && args+=(--robot-archive "$DISC_ROOT/$sc/robot_archive.npz")
  [[ -f "configs/contact_intervals/$sc.yaml" ]] && args+=(--contact-segments "configs/contact_intervals/$sc.yaml")
  [[ -d "$DISC_ROOT/$sc/NEW_cluster_obs_cam001_002" ]] && args+=(--discovery-dir "$DISC_ROOT/$sc/NEW_cluster_obs_cam001_002")
  [[ "$sc" == *fridge* ]] && args+=(--pulse-rows 229:232)
  [[ "$sc" == *black_box* ]] && args+=(--declared-joint-type revolute --declaration-note "operator 2026-11-20: the black box is revolute (declared, not identified)")
  [[ "${PINN:-0}" == 1 ]] && args+=(--pinn)
  [[ "${RESIDUAL:-0}" == 1 ]] && args+=(--residual)
  OMP_NUM_THREADS=4 "$PY" -m forceartgs_g9.cli $cmd "${args[@]}" > "$run/g9_forcetwin.log" 2>&1
  log "$sc exit $? $(grep -o '"status": "[A-Z_]*"' "$run/g9_forcetwin.log" | head -1) $(grep -o '"out_dir": "[^"]*"' "$run/g9_forcetwin.log" | sed 's#.*/##')"
done
"$PY" tools/g9_forcetwin_comparison.py "$RUNS" "/media/samuelemara/Extreme SSD/Samuele/results_21_11_26/forcetwin_comparison" >> "$LOG" 2>&1
log "done"
