#!/usr/bin/env bash
# Joint-aware ForceArtGS on EVERY real dataset (one episode per object by default, or all episodes), with an automatic
# per-episode front-end built only from existing tools -- nothing is hand-picked:
#   1. force export      samuele/export_force_contact_annotations.py   -> <OUT>/contact_annotations/<ep>/
#   2. clip              RGB-D window = [first sustained force contact - 20 frames, last contact + 6] (the margins of
#                        the hand-picked fridge clip), stride = max(2, ceil(len / 60)) (~30 model times)
#   3. visual audit      samuele/audit_joint_contact_windows.py        -> <OUT>/contact_audit/<ep>/rgbd_rate_timeline.csv
#   4. contact segments  samuele/estimate_contact_segments.py (force peaks + visual, sliding) -> configs/contact_intervals/
#                        (an existing segments file, e.g. the operator-validated ones, is NEVER overwritten)
#   5. full pipeline     pipelines/joint_aware_reconstruction.py: fresh discovery (robot archive, AllTracker, fit),
#                        multimodal frontend, selection v2, R/P branches, blend, art (hard part ids), compare, video,
#                        final, G8 export (+ diagnostic assets); MODEL_INCONSISTENT scenes still train diagnostics.
# Resumable: finished steps / stages are skipped. Outputs only on the SSD.
#
#   bash pipelines/joint_aware_all_real.sh [budget smoke|coarse|full] [OUT]
# Env: EPISODES=first (default: episode 000000 of every object) | all ;  SCENES="name ..." explicit list (overrides);
#      PREP_ONLY=1 (steps 1-4 only, no GPU training) ; G9=1 also run G9 exploratory dynamics after each scene.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BUDGET="${1:-full}"
OUT="${2:-/media/samuelemara/Extreme SSD/Samuele/results_$(date +%d_%m_%y)}"
case "$OUT" in "/media/samuelemara/Extreme SSD/"*) ;; *) echo "refusing: output must be on the Extreme SSD: $OUT" >&2; exit 2;; esac
PY="$HOME/miniconda3/envs/freeartgs/bin/python"
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
RUNS="$OUT/forceartgs_joint_aware"
mkdir -p "$RUNS/.runtime" "$OUT/contact_annotations" "$OUT/contact_audit" configs/contact_intervals
SEED="/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware/.runtime"
for d in torch huggingface torch_extensions; do
  [[ -d "$SEED/$d" && ! -e "$RUNS/.runtime/$d" ]] && cp -r "$SEED/$d" "$RUNS/.runtime/$d"
done
LOG="$OUT/joint_aware_all_real.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

if [[ -n "${SCENES:-}" ]]; then
  LIST=($SCENES)
elif [[ "${EPISODES:-first}" == all ]]; then
  mapfile -t LIST < <(ls "$REAL" | grep -E '_episode_[0-9]{6}$')
else
  mapfile -t LIST < <(ls "$REAL" | grep -E '_episode_000000$')
fi
log "start budget=$BUDGET episodes=${#LIST[@]} git=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"

clip_of() {   # prints a:b:k from the force export (sustained contact intervals, source rows -> RGB-D frames)
  "$PY" - "$1" "$2" <<'EOF'
import csv, json, math, sys
fx, scene = sys.argv[1], sys.argv[2]
iv = json.load(open(f"{fx}/contact_summary.json")).get("intervals") or []
iv = [x for x in iv if x[1] - x[0] >= 5] or iv            # sustained contacts (>= 0.1 s) when there are any
if not iv:
    sys.exit(3)
rows = list(csv.DictReader(open(f"{scene}/metadata/frame_timestamps.csv")))
src = [int(r["source_frame_index"]) for r in rows]
fr = [int(r["frame_index"]) for r in rows]
def to_rgbd(s):
    return fr[min(range(len(src)), key=lambda i: abs(src[i] - s))]
a = max(0, to_rgbd(min(x[0] for x in iv)) - 20)
b = min(fr[-1], to_rgbd(max(x[1] for x in iv)) + 6)
k = max(2, math.ceil((b - a + 1) / 60))
print(f"{a}:{b}:{k}")
EOF
}

for ep in "${LIST[@]}"; do
  S="$REAL/$ep"
  [[ -f "$S/metadata/cameras.json" && -d "$S/rgb/cam_001" ]] || { log "$ep SKIP: not a converted RGB-D scene"; continue; }
  FX="$OUT/contact_annotations/$ep"
  if [[ ! -f "$FX/contact_summary.json" ]]; then
    if ! ls "$S"/metadata/raw/episode_*.parquet >/dev/null 2>&1; then log "$ep SKIP: no robot recording (DATA_UNAVAILABLE)"; continue; fi
    "$PY" samuele/export_force_contact_annotations.py --scene "$S" --output "$FX" --no-plot > "$FX.log" 2>&1 \
      || { log "$ep SKIP: force export failed (see $FX.log)"; continue; }
  fi
  CLIP=$(clip_of "$FX" "$S") || { log "$ep SKIP: no sustained force contact (DATA_UNAVAILABLE)"; continue; }
  AU="$OUT/contact_audit/$ep"
  if [[ ! -f "$AU/rgbd_rate_timeline.csv" ]]; then
    rm -rf "$AU"
    "$PY" samuele/audit_joint_contact_windows.py --scene "$S" --clip-frames "$CLIP" --cameras cam_001 cam_002 \
        --output "$AU" > "$AU.log" 2>&1 || log "$ep WARN: visual audit failed (force-only segments)"
  fi
  SEG="configs/contact_intervals/$ep.yaml"
  if [[ ! -f "$SEG" && -f "$AU/rgbd_rate_timeline.csv" ]]; then
    "$PY" samuele/estimate_contact_segments.py --scene "$ep" --frames "$CLIP" --force-csv "$FX/contact_annotations.csv" \
        --visual-csv "$AU/rgbd_rate_timeline.csv" --output "$SEG" --audit-csv "$AU/contact_segments_audit.csv" \
        >> "$AU.log" 2>&1 || log "$ep WARN: contact segment estimation failed"
  fi
  log "$ep prepared clip=$CLIP segments=$([[ -f $SEG ]] && grep -c segment_id "$SEG" || echo none)"
  [[ "${PREP_ONLY:-0}" == 1 ]] && continue
  MM=(--multimodal-frontend --mm-max-iterations 3)
  [[ -f "$SEG" ]] && MM+=(--contact-segments "$SEG")
  "$PY" pipelines/joint_aware_reconstruction.py --scene "$S" --frames "$CLIP" --cameras cam_001 cam_002 \
      --output-root "$RUNS" --stage all --budget "$BUDGET" "${MM[@]}" \
      --export-parttwin --export-diagnostic --allow-inconsistent-diagnostics >> "$RUNS/$ep.$BUDGET.log" 2>&1
  rc=$?
  IFS=: read -r fa fb fk <<< "$CLIP"
  sel="$(ls -d "$RUNS/${ep}_multiview_cams001-002_i${fa}-${fb}_fall_s${fk}_joint_aware_${BUDGET}" 2>/dev/null)/selection.json"
  log "$ep exit $rc $( [[ -f "$sel" ]] && "$PY" -c "import json,sys; s=json.load(open(sys.argv[1])); print(s['status'], s.get('selected') or s.get('working_motion_model'))" "$sel")"
  if [[ "${G9:-0}" == 1 && $rc == 0 ]]; then
    SCENES="$ep" DIAG_KIN=1 VARIANTS="full viscous_only" bash pipelines/g9_real_exploratory.sh "$RUNS" >> "$LOG" 2>&1
  fi
done
log "done"
