#!/usr/bin/env bash
# Frozen-joint q(t) on the WHOLE recording (no 3DGS, no joint re-estimation):
#   1. AllTracker + depth on every RGB-D frame (step 2: contains the reconstruction clip's frames), both cameras
#   2. estimotion extend-states: the accepted joint (type / axis / hinge / gauge) is kept exactly, q is measured on the
#      new frames, validated against the existing q on the clip overlap -> a NEW versioned authority
# Outputs (append-only, the runs are not touched):  $OUT/<scene>/{obs_cam_*.npz, authority/}
#   bash pipelines/extend_joint_states_real.sh [OUT]
#   Env: SCENES subset; RUNS (joint-aware runs); STEP (default 2); TAU_M (default 0.006);
#        DECLARED=revolute|prismatic (operator joint type for every scene of this call; black box: revolute)
#        NO_DECLARATION=1: never declare (the automatic joint decision only)
#        JA_BUDGET=full|coarse|smoke: which joint-aware run budget to read (default full)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${1:-/media/samuelemara/Extreme SSD/Samuele/results_26_11_26/joint_states_extended}"
RUNS="${RUNS:-/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware}"
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
DISC_ROOT="${DISC_ROOT:-/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint}"
PY="${FORCEARTGS_PYTHON:-$HOME/miniconda3/envs/forceartgs/bin/python}"
eval "$("$PY" "$(dirname "${BASH_SOURCE[0]}")/../utils/forceartgs_env.py")"   # clean env: forceartgs + repo only
SCENES="${SCENES:-final_red_fridge_episode_000000 final_orange_suitecase_episode_000000 final_black_box_episode_000000}"
STEP="${STEP:-2}"
mkdir -p "$OUT"
LOG="$OUT/extend_joint_states.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "start scenes=[$SCENES] step=$STEP git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for sc in $SCENES; do
  run=$(ls -d "$RUNS/${sc}"_multiview_*_joint_aware_${JA_BUDGET:-full} 2>/dev/null | head -1)
  [[ -z "$run" ]] && { log "$sc: SKIP no joint-aware run"; continue; }
  dt="${DECLARED:-}"
  [[ -z "$dt" && "$sc" == *black_box* && "${NO_DECLARATION:-0}" != 1 ]] && dt=revolute
  auth=$("$PY" - "$run" "$dt" <<'EOF' 2>/dev/null | tail -1
import sys
from forceartgs_g9.adapters.forceartgs_g9 import accepted_kinematics
dt = sys.argv[2] if len(sys.argv) > 2 else ""
kw = dict(declared_joint_type=dt, declaration_note=f"operator: declared {dt}") if dt else {}
print(accepted_kinematics(sys.argv[1], True, **kw).get("authority_dir") or "")
EOF
)
  [[ -z "$auth" ]] && { log "$sc: SKIP no accepted joint authority"; continue; }
  n=$(ls "$REAL/$sc/rgb/cam_001" | wc -l)
  # the extension grid must CONTAIN the reconstruction clip's frames (validated on their overlap): start it on the
  # clip's parity (clip 63:162:2 -> 1:...:2; an even grid would share no frame with it)
  off=0
  [[ "$(basename "$run")" =~ _i([0-9]+)-([0-9]+)_fall_s([0-9]+)_ ]] && off=$(( BASH_REMATCH[1] % STEP ))
  frames="$off:$((n - 1)):$STEP"
  d="$OUT/$sc"; mkdir -p "$d"
  arc=()
  if [[ -f "$DISC_ROOT/$sc/robot_archive.npz" ]]; then arc=(--tcp-npz "$DISC_ROOT/$sc/robot_archive.npz")
  elif [[ -f "$run/discovery/robot_archive.npz" ]]; then arc=(--tcp-npz "$run/discovery/robot_archive.npz"); fi
  obs=()
  for cam in cam_001 cam_002; do
    f="$d/obs_$cam.npz"
    if [[ ! -f "$f" || "$(cat "$d/frames_$cam.txt" 2>/dev/null)" != "$frames" ]]; then   # cache is per frame grid
      log "$sc $cam: extract frames $frames"
      "$PY" -m estimotion_2part.joint_discovery.cli extract --scene "$REAL/$sc" --camera "$cam" --frames "$frames" \
        "${arc[@]}" --output "$f" >> "$d/extract_$cam.log" 2>&1 || { log "$sc $cam: extract FAILED (see $d/extract_$cam.log)"; continue; }
      echo "$frames" > "$d/frames_$cam.txt"
    fi
    obs+=("$f")
  done
  [[ ${#obs[@]} -eq 0 ]] && continue
  v=$(ls -d "$d"/authority_v[0-9][0-9][0-9] 2>/dev/null | wc -l)
  outa="$d/authority_v$(printf %03d "$v")"
  "$PY" -m estimotion_2part.joint_discovery.cli extend-states --authority "$auth" --observations "${obs[@]}" \
    --output "$outa" --tau-m "${TAU_M:-0.006}" >> "$d/extend.log" 2>&1
  ec=$?
  log "$sc: extend-states exit $ec -> $outa ($(tail -1 "$d/extend.log" | cut -c1-200))"
done
log "done"
