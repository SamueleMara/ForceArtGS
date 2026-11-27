#!/usr/bin/env bash
# G9 EXPLORATORY real-data estimation on the finished joint-aware runs, from the canonical robot record (existing
# force export + robot archive + declared conventions). Estimates are conditional on the listed calibration
# assumptions and never production-exported. Writes only <run>/dynamics/g9/vNNN (one version per model variant).
#   bash pipelines/g9_real_exploratory.sh [RUN_ROOT]
# Env: SCENES subset, VARIANTS (default: full no_gravity no_static viscous_coulomb viscous_only),
#      DIAG_KIN=1 (default) also fits the working model of TYPE_UNOBSERVABLE joints (labelled KINEMATICS_NOT_ACCEPTED)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS="${1:-/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware}"
PY="${FORCEARTGS_PYTHON:-$HOME/miniconda3/envs/forceartgs/bin/python}"
eval "$("$PY" "$(dirname "${BASH_SOURCE[0]}")/../utils/forceartgs_env.py")"   # clean env: forceartgs + repo only
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
DISC_ROOT="${DISC_ROOT:-/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint}"
VARIANTS="${VARIANTS:-full no_gravity no_static viscous_coulomb viscous_only}"
LOG="$RUNS/g9_exploratory_batch.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "start variants=[$VARIANTS] git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for run in "$RUNS"/*_joint_aware_full; do
  name=$(basename "$run"); scene="${name%%_multiview_*}"
  [[ -n "${SCENES:-}" && " $SCENES " != *" $scene "* ]] && continue
  for v in $VARIANTS; do
    args=(--run "$run" --scene "$REAL/$scene" --dynamics-config "configs/g9_real/exploratory_$v.yaml")
    [[ "${DIAG_KIN:-1}" == 1 ]] && args+=(--allow-diagnostic-kinematics)
    [[ -f "$DISC_ROOT/$scene/robot_archive.npz" ]] && args+=(--robot-archive "$DISC_ROOT/$scene/robot_archive.npz")
    [[ -f "configs/contact_intervals/$scene.yaml" ]] && args+=(--contact-segments "configs/contact_intervals/$scene.yaml")
    "$PY" -m forceartgs_g9.cli dynamics "${args[@]}" > "$run/g9_exploratory_$v.log" 2>&1
    log "$scene $v exit $? $(grep -o '"status": "[A-Z_]*"' "$run/g9_exploratory_$v.log" | head -1) $(grep -o '"out_dir": "[^"]*"' "$run/g9_exploratory_$v.log" | sed 's#.*/##')"
  done
done
log "done"
