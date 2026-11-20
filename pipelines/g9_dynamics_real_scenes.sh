#!/usr/bin/env bash
# G9 dynamics on the finished joint-aware real-scene runs (no 3DGS training; writes only <run>/dynamics/g9/vNNN).
#   bash pipelines/g9_dynamics_real_scenes.sh [RUN_ROOT] [DYNAMICS_CONFIG]
# Env: SCENES="name ..." subset, PHYSICS_LIBRARY (default ~/ForceArtGS-Physics, pinned), FINAL_EXPORT=1 to also run
# final_physics_export (only acts when G9 produced a production physics spec).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS="${1:-/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware}"
CFG="${2:-configs/dynamics_general.example.yaml}"
PY="$HOME/miniconda3/envs/freeartgs/bin/python"
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
DISC_ROOT="${DISC_ROOT:-/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint}"
LOG="$RUNS/g9_dynamics_batch.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "start cfg=$CFG git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for run in "$RUNS"/*_joint_aware_full; do
  name=$(basename "$run"); scene="${name%%_multiview_*}"
  [[ -n "${SCENES:-}" && " $SCENES " != *" $scene "* ]] && continue
  args=(--run "$run" --scene "$REAL/$scene" --dynamics-config "$CFG")
  [[ -n "${PHYSICS_LIBRARY:-}" ]] && args+=(--physics-library "$PHYSICS_LIBRARY")
  [[ -f "$DISC_ROOT/$scene/robot_archive.npz" ]] && args+=(--robot-archive "$DISC_ROOT/$scene/robot_archive.npz")
  [[ -f "configs/contact_intervals/$scene.yaml" ]] && args+=(--contact-segments "configs/contact_intervals/$scene.yaml")
  log "$scene dynamics"
  "$PY" -m forceartgs_g9.cli dynamics "${args[@]}" >> "$run/g9_dynamics.log" 2>&1
  log "$scene exit $? $(tail -c 600 "$run/g9_dynamics.log" | tr '\n' ' ' | grep -o '"status": "[A-Z_]*"' | tail -1)"
  if [[ "${FINAL_EXPORT:-0}" == 1 ]]; then
    "$PY" -m forceartgs_g9.cli final-physics-export --run "$run" >> "$run/g9_dynamics.log" 2>&1
    log "$scene final_physics_export exit $?"
  fi
done
log "done"
