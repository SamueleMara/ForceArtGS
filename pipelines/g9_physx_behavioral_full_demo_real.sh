#!/usr/bin/env bash
# G10-A behavioral full-demonstration PhysX replay on every existing real run (opt-in, estimator side, append-only):
#   <run>/dynamics/g9/physx_behavioral/vNNN   (or $OUT_BASE/<scene>/vNNN)
# One continuous SAPIEN PhysX rollout per scene from (q0 = authority q(t0), v0 = 0) under the recorded canonical effort.
# Produces a behavioral simulator CANDIDATE (never physical identification / sim-to-real validation).
# No calibration, no 3DGS retraining, no joint-authority change, no sign change, no PArtTwin1D change.
#   bash pipelines/g9_physx_behavioral_full_demo_real.sh [RUN_ROOT]
#   Env: SCENES subset; OUT_BASE redirect; BEH_CONFIG (default the STRICT full-demo config;
#        configs/g9_real/physx_behavioral_rest_verified_crop.yaml is the explicit cropped-start opt-in);
#        QUICK=1 tiny budget (smoke); SAPIEN engine = this (forceartgs) interpreter.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS="${1:-/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware}"
PY="${FORCEARTGS_PYTHON:-$HOME/miniconda3/envs/forceartgs/bin/python}"
eval "$("$PY" "$(dirname "${BASH_SOURCE[0]}")/../utils/forceartgs_env.py")"   # clean env: forceartgs + repo only
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
DISC_ROOT="${DISC_ROOT:-/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint}"
SCENES="${SCENES:-final_red_fridge_episode_000000 final_orange_suitecase_episode_000000 final_black_box_episode_000000 final_robot_dog_episode_000000 white_drawer_20261021_161154_episode_000000}"
BEH_CONFIG="${BEH_CONFIG:-configs/g9_real/physx_behavioral_full_demo.yaml}"
LOG="${OUT_BASE:-$RUNS}/g9_physx_behavioral_batch.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "start scenes=[$SCENES] BEH_CONFIG=$BEH_CONFIG QUICK=${QUICK:-0} OUT_BASE=${OUT_BASE:-<run>/dynamics/g9} git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for sc in $SCENES; do
  run=$(ls -d "$RUNS/${sc}"_multiview_*_joint_aware_full 2>/dev/null | head -1)
  [[ -z "$run" ]] && { log "$sc: no joint-aware run"; continue; }
  a=(--run "$run" --scene "$REAL/$sc" --behavioral-config "$BEH_CONFIG")
  [[ -f "$DISC_ROOT/$sc/robot_archive.npz" ]] && a+=(--robot-archive "$DISC_ROOT/$sc/robot_archive.npz")
  [[ -f "configs/contact_intervals/$sc.yaml" ]] && a+=(--contact-segments "configs/contact_intervals/$sc.yaml")
  [[ "$sc" == *fridge* ]] && a+=(--pulse-rows 229:232)
  [[ "$sc" == *black_box* ]] && a+=(--declared-joint-type revolute --declaration-note "operator 2026-11-20: the black box is revolute (declared, not identified)")
  [[ -n "${OUT_BASE:-}" ]] && a+=(--out-root "$OUT_BASE/$sc/physx_behavioral")
  [[ "${QUICK:-0}" == 1 ]] && a+=(--quick)
  lf="${OUT_BASE:-$run}/g9_physx_behavioral_${sc}.log"
  mkdir -p "$(dirname "$lf")"
  OMP_NUM_THREADS=4 "$PY" -m forceartgs_g9.cli physx-behavioral-replay "${a[@]}" > "$lf" 2>&1
  ec=$?
  outd=$(grep -o '"out_dir": "[^"]*"' "$lf" | head -1 | sed 's/"out_dir": "//; s/"$//')
  log "$sc exit $ec $(grep -o '"status": "[A-Z_]*"' "$lf" | head -1) ${outd##*/}"
done
log "done"
