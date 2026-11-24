#!/usr/bin/env bash
# Physics-constrained input-output identification on every existing real run (estimator side, append-only):
#   <run>/dynamics/g9/io_identification/vNNN   (or $OUT_BASE/<scene>/vNNN)
# then, with THEN_G10=1, the G10 PhysX refinement warm-started from it (--use-io-initialization; long).
# No calibration, no 3DGS retraining, no joint-authority change, no sign change, no PArtTwin1D change.
#   bash pipelines/g9_io_identification_all_real.sh [RUN_ROOT]
#   Env: SCENES subset; OUT_BASE redirect; IO_CONFIG (default the diagnostic real config); NO_PHYSX=1 skip the
#        PhysX reconciliation; THEN_G10=1 run G10 afterwards (QUICK=1 tiny G10 budget); PARTTWIN1D_PYTHON engine env.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS="${1:-/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware}"
PY="$HOME/miniconda3/envs/freeartgs/bin/python"
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
DISC_ROOT="/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint"
SCENES="${SCENES:-final_red_fridge_episode_000000 final_orange_suitecase_episode_000000 final_black_box_episode_000000 final_robot_dog_episode_000000 white_drawer_20261021_161154_episode_000000}"
IO_CONFIG="${IO_CONFIG:-configs/g9_real/io_identification_diagnostic.yaml}"
LOG="${OUT_BASE:-$RUNS}/g9_io_identification_batch.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "start scenes=[$SCENES] IO_CONFIG=$IO_CONFIG THEN_G10=${THEN_G10:-0} OUT_BASE=${OUT_BASE:-<run>/dynamics/g9} git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for sc in $SCENES; do
  run=$(ls -d "$RUNS/${sc}"_multiview_*_joint_aware_full 2>/dev/null | head -1)
  [[ -z "$run" ]] && { log "$sc: no joint-aware run"; continue; }
  common=(--run "$run" --scene "$REAL/$sc")
  [[ -f "$DISC_ROOT/$sc/robot_archive.npz" ]] && common+=(--robot-archive "$DISC_ROOT/$sc/robot_archive.npz")
  [[ -f "configs/contact_intervals/$sc.yaml" ]] && common+=(--contact-segments "configs/contact_intervals/$sc.yaml")
  [[ "$sc" == *fridge* ]] && common+=(--pulse-rows 229:232)
  [[ "$sc" == *black_box* ]] && common+=(--declared-joint-type revolute --declaration-note "operator 2026-11-20: the black box is revolute (declared, not identified)")
  io=("${common[@]}" --io-config "$IO_CONFIG")
  [[ -n "${OUT_BASE:-}" ]] && io+=(--out-root "$OUT_BASE/$sc/io_identification")
  [[ "${NO_PHYSX:-0}" == 1 ]] && io+=(--no-physx)
  lf="${OUT_BASE:-$run}/g9_io_identification_${sc}.log"
  mkdir -p "$(dirname "$lf")"
  OMP_NUM_THREADS=4 "$PY" -m forceartgs_g9.cli io-identification "${io[@]}" > "$lf" 2>&1
  ec=$?
  outd=$(grep -o '"out_dir": "[^"]*"' "$lf" | head -1 | sed 's/"out_dir": "//; s/"$//')
  log "$sc io exit $ec $(grep -o '"status": "[A-Z_]*"' "$lf" | head -1) ${outd##*/}"
  if [[ "${THEN_G10:-0}" == 1 && -f "$outd/io_response.json" ]]; then
    g=("${common[@]}" --g10-config "${G10_CONFIG:-configs/g9_real/physx_refinement_io_init.yaml}" --use-io-initialization "$outd/io_response.json")
    [[ -n "${OUT_BASE:-}" ]] && g+=(--out-root "$OUT_BASE/$sc/physx_refinement")
    [[ "${QUICK:-0}" == 1 ]] && g+=(--quick)
    gf="${OUT_BASE:-$run}/g9_physx_refinement_io_init_${sc}.log"
    OMP_NUM_THREADS=4 "$PY" -m forceartgs_g9.cli physx-refinement "${g[@]}" > "$gf" 2>&1
    log "$sc g10(io-init) exit $? $(grep -o '"status": "[A-Z_]*"' "$gf" | head -1)"
  fi
done
log "done"
