#!/usr/bin/env bash
# G10 PhysX (SAPIEN CPU diagnostic) refinement on every existing real run. Append-only:
# <run>/dynamics/g9/physx_refinement/vNNN  (or $OUT_BASE/<scene>/vNNN). No calibration, no 3DGS retraining, no
# joint-authority change, no sign change.
#   bash pipelines/g9_physx_refinement_all_real.sh [RUN_ROOT]
#      SCENES subset; OUT_BASE redirect (e.g. a scratch dir for a dry run).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS="${1:-/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware}"
PY="$HOME/miniconda3/envs/freeartgs/bin/python"
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
DISC_ROOT="/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint"
SCENES="${SCENES:-final_red_fridge_episode_000000 final_orange_suitecase_episode_000000 final_black_box_episode_000000 final_robot_dog_episode_000000 white_drawer_20261021_161154_episode_000000}"
LOG="${OUT_BASE:-$RUNS}/g9_physx_refinement_batch.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "start scenes=[$SCENES] QUICK=${QUICK:-0} OUT_BASE=${OUT_BASE:-<run>/dynamics/g9/physx_refinement} git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
# Env: QUICK=1 tiny optimiser budget (smoke only); SCENES subset; OUT_BASE redirect; PARTTWIN1D_PYTHON engine env.
for sc in $SCENES; do
  run=$(ls -d "$RUNS/${sc}"_multiview_*_joint_aware_full 2>/dev/null | head -1)
  [[ -z "$run" ]] && { log "$sc: no joint-aware run"; continue; }
  args=(--run "$run" --scene "$REAL/$sc" --g10-config configs/g9_real/physx_refinement.yaml)
  [[ -n "${OUT_BASE:-}" ]] && args+=(--out-root "$OUT_BASE/$sc")
  [[ -f "$DISC_ROOT/$sc/robot_archive.npz" ]] && args+=(--robot-archive "$DISC_ROOT/$sc/robot_archive.npz")
  [[ -f "configs/contact_intervals/$sc.yaml" ]] && args+=(--contact-segments "configs/contact_intervals/$sc.yaml")
  [[ "$sc" == *fridge* ]] && args+=(--pulse-rows 229:232)
  [[ "$sc" == *black_box* ]] && args+=(--declared-joint-type revolute --declaration-note "operator 2026-11-20: the black box is revolute (declared, not identified)")
  [[ "${QUICK:-0}" == 1 ]] && args+=(--quick)
  lf="${OUT_BASE:-$run}/g9_physx_refinement_${sc}.log"
  mkdir -p "$(dirname "$lf")"
  OMP_NUM_THREADS=4 "$PY" -m forceartgs_g9.cli physx-refinement "${args[@]}" > "$lf" 2>&1
  log "$sc exit $? $(grep -o '"status": "[A-Z_]*"' "$lf" | head -1) $(grep -o '"out_dir": "[^"]*"' "$lf" | sed 's#.*/##')"
done
log "done"
