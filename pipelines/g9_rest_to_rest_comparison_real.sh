#!/usr/bin/env bash
# Rest-to-rest comparison (G9 analytics + PINN ablations + equal-budget G10 PhysX) on the existing real runs:
#   <run>/dynamics/g9/rest_to_rest_compare/vNNN   (or $OUT_BASE/<scene>/rest_to_rest_compare/vNNN)
# ONE immutable rest-to-rest episode per scene (visual plateaus -> [t0, T]; full six-axis wrench on every native row,
# no mask, no second tare; v0 = vT = 0 declared). The contact-segments file is passed ONLY for the C0 archival control.
# No 3DGS retraining, no calibration, no contact annotation, no PArtTwin1D change.
#   bash pipelines/g9_rest_to_rest_comparison_real.sh [RUN_ROOT]
#   Env: SCENES subset; OUT_BASE redirect; R2R_CONFIG; QUICK=1 tiny budgets (plumbing only); SAPIEN engine = this (forceartgs) interpreter;
#        DECLARED=revolute|prismatic (operator joint type for every scene of this call)
#        NO_DECLARATION=1: never declare (the automatic joint decision only)
#        JA_BUDGET=full|coarse|smoke: which joint-aware run budget to read (default full)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS="${1:-/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware}"
PY="${FORCEARTGS_PYTHON:-$HOME/miniconda3/envs/forceartgs/bin/python}"
eval "$("$PY" "$(dirname "${BASH_SOURCE[0]}")/../utils/forceartgs_env.py")"   # clean env: forceartgs + repo only
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
DISC_ROOT="${DISC_ROOT:-/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint}"
SCENES="${SCENES:-final_red_fridge_episode_000000 final_orange_suitecase_episode_000000 final_black_box_episode_000000 final_robot_dog_episode_000000 white_drawer_20261021_161154_episode_000000}"
R2R_CONFIG="${R2R_CONFIG:-configs/g9_real/rest_to_rest_compare.yaml}"
LOG="${OUT_BASE:-$RUNS}/g9_rest_to_rest_compare_batch.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "start scenes=[$SCENES] R2R_CONFIG=$R2R_CONFIG QUICK=${QUICK:-0} OUT_BASE=${OUT_BASE:-<run>/dynamics/g9} git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for sc in $SCENES; do
  run=$(ls -d "$RUNS/${sc}"_multiview_*_joint_aware_${JA_BUDGET:-full} 2>/dev/null | head -1)
  [[ -z "$run" ]] && { log "$sc: SKIP no joint-aware run under $RUNS"; continue; }
  a=(--run "$run" --scene "$REAL/$sc" --config "$R2R_CONFIG" --pinn-device "${PINN_DEVICE:-cpu}")
  if [[ -f "$DISC_ROOT/$sc/robot_archive.npz" ]]; then a+=(--robot-archive "$DISC_ROOT/$sc/robot_archive.npz")
  elif [[ -f "$run/discovery/robot_archive.npz" ]]; then a+=(--robot-archive "$run/discovery/robot_archive.npz")
  else log "$sc: no robot archive at $DISC_ROOT/$sc (the run's canonical record is tried)"; fi
  [[ -f "configs/contact_intervals/$sc.yaml" ]] && a+=(--contact-segments "configs/contact_intervals/$sc.yaml")
  if [[ -n "${DECLARED:-}" ]]; then a+=(--declared-joint-type "$DECLARED" --declaration-note "operator: declared $DECLARED")
  elif [[ "$sc" == *black_box* && "${NO_DECLARATION:-0}" != 1 ]]; then a+=(--declared-joint-type revolute --declaration-note "operator 2026-11-20: the black box is revolute (declared, not identified)"); fi
  ext=$(ls -d "${EXT_ROOT:-/media/samuelemara/Extreme SSD/Samuele/results_26_11_26/joint_states_extended}/$sc"/authority_v[0-9][0-9][0-9] 2>/dev/null | tail -1)
  if [[ -n "$ext" && -f "$ext/authority.json" ]]; then a+=(--authority-dir "$ext"); log "$sc: extended authority $ext"
  else log "$sc: no extended authority (clip q only; EXT_ROOT to override)"; fi
  [[ -n "${OUT_BASE:-}" ]] && a+=(--out-root "$OUT_BASE/$sc/rest_to_rest_compare")
  [[ "${QUICK:-0}" == 1 ]] && a+=(--quick)
  lf="${OUT_BASE:-$run}/g9_rest_to_rest_compare_${sc}.log"
  mkdir -p "$(dirname "$lf")"
  OMP_NUM_THREADS=4 "$PY" -m forceartgs_g9.cli rest-to-rest-compare "${a[@]}" > "$lf" 2>&1
  ec=$?
  outd=$(grep -o '"out_dir": "[^"]*"' "$lf" | head -1 | sed 's/"out_dir": "//; s/"$//')
  log "$sc exit $ec $(grep -o '"status": "[A-Z_]*"' "$lf" | head -1) ${outd}"
done
log "done"
