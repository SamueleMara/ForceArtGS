#!/usr/bin/env bash
# G9 four-estimator comparison (general_integral_ls / general_rollout_fit / general_pinn / general_akf, ONE shared
# ModelSpec) on finished joint-aware runs. geometry_coupled with a TENTATIVE solid / shell hull proxy, plus the
# joint_space_gravity_coupled ablation. EXPLORATORY: conditional on the declared calibration; never production-exported.
#   bash pipelines/g9_real_general_methods.sh [RUN_ROOT]
# Env: SCENES (default: fridge + suitcase), VARIANTS (default below), FRIDGE_PULSE_ROWS (default 229:232)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS="${1:-/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware}"
PY="$HOME/miniconda3/envs/freeartgs/bin/python"
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
DISC_ROOT="${DISC_ROOT:-/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint}"
SCENES="${SCENES:-final_red_fridge_episode_000000 final_orange_suitecase_episode_000000}"
VARIANTS="${VARIANTS:-geo_solid_viscous_gravity geo_shell_viscous_gravity geo_solid_full gc_viscous_gravity_allmethods}"
LOG="$RUNS/g9_general_methods_batch.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "start scenes=[$SCENES] variants=[$VARIANTS] git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for run in "$RUNS"/*_joint_aware_full; do
  name=$(basename "$run"); scene="${name%%_multiview_*}"
  [[ " $SCENES " != *" $scene "* ]] && continue
  for v in $VARIANTS; do
    args=(--run "$run" --scene "$REAL/$scene" --dynamics-config "configs/g9_real/exploratory_$v.yaml"
          --allow-diagnostic-kinematics)
    [[ -f "$DISC_ROOT/$scene/robot_archive.npz" ]] && args+=(--robot-archive "$DISC_ROOT/$scene/robot_archive.npz")
    [[ -f "configs/contact_intervals/$scene.yaml" ]] && args+=(--contact-segments "configs/contact_intervals/$scene.yaml")
    [[ "$scene" == *fridge* ]] && args+=(--declared-pulse-rows "${FRIDGE_PULSE_ROWS:-229:232}")
    OMP_NUM_THREADS=4 "$PY" -m forceartgs_g9.cli dynamics "${args[@]}" > "$run/g9_general_$v.log" 2>&1
    log "$scene $v exit $? $(grep -o '"status": "[A-Z_]*"' "$run/g9_general_$v.log" | head -1) $(grep -o '"out_dir": "[^"]*"' "$run/g9_general_$v.log" | sed 's#.*/##')"
  done
done
log "done"
