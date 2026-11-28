#!/usr/bin/env bash
# Submit every episode of every object as a SLURM array (1 GPU per episode, SLURM_MAX_PARALLEL at once), then the
# summary job (afterany: it also runs when some episodes failed; each episode's status is in its row), then the
# cross-demo validation (CROSS_DEMO=1) and, independently, the synthetic benchmark (SYNTHETIC=1).
#   bash slurm/submit_all.sh            # edit slurm/config.sh first
#   DRY_RUN=1 bash slurm/submit_all.sh  # print the episode list and the sbatch commands only
# Resubmitting the same command resumes: finished episodes/steps are skipped.
set -euo pipefail
export FORCEARTGS_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FORCEARTGS_SLURM_DIR/config.sh"
for v in FORCEARTGS_REPO FORCEARTGS_PYTHON FORCEARTGS_REAL_ROOT; do
  [[ -e "${!v}" ]] || { echo "config.sh: $v=${!v} does not exist" >&2; exit 2; }
done
case "$FORCEARTGS_OUT" in "${FORCEARTGS_OUTPUT_ROOT%/}/"*) ;; *) echo "FORCEARTGS_OUT must be under FORCEARTGS_OUTPUT_ROOT" >&2; exit 2;; esac
mkdir -p "$FORCEARTGS_OUT/slurm_logs"
export EPISODE_LIST="$FORCEARTGS_OUT/slurm_logs/episodes_$(date +%Y%m%d_%H%M%S).txt"
"$FORCEARTGS_PYTHON" "$FORCEARTGS_REPO/pipelines/real_full_to_estimated_assets.py" --episodes "$EPISODES" --list-scenes \
  | grep -E "$OBJECTS_REGEX" > "$EPISODE_LIST" || true
N=$(wc -l < "$EPISODE_LIST")
[[ "$N" -gt 0 ]] || { echo "no episodes under $FORCEARTGS_REAL_ROOT matching EPISODES=$EPISODES OBJECTS_REGEX=$OBJECTS_REGEX" >&2; exit 2; }
echo "$N episodes -> $EPISODE_LIST"
common=(--export=ALL)                      # FORCEARTGS_SLURM_DIR, EPISODE_LIST and config.sh paths are exported
[[ -n "$SLURM_PARTITION" ]] && common+=(--partition="$SLURM_PARTITION")
[[ -n "$SLURM_ACCOUNT" ]] && common+=(--account="$SLURM_ACCOUNT")
[[ -n "$SLURM_QOS" ]] && common+=(--qos="$SLURM_QOS")
arr=(sbatch --parsable "${common[@]}" --job-name=forceartgs_ep --array="0-$((N - 1))%$SLURM_MAX_PARALLEL"
     --gres="$SLURM_GRES" --cpus-per-task="$SLURM_CPUS" --mem="$SLURM_MEM" --time="$SLURM_TIME"
     --output="$FORCEARTGS_OUT/slurm_logs/ep_%A_%a.out" $SLURM_EXTRA "$FORCEARTGS_SLURM_DIR/run_episode.sbatch")
sumj=(sbatch --parsable "${common[@]}" --job-name=forceartgs_summary --cpus-per-task=2 --mem=8G --time=01:00:00
      --output="$FORCEARTGS_OUT/slurm_logs/summary_%j.out")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  cat "$EPISODE_LIST"; printf '%q ' "${arr[@]}"; echo; printf '%q ' "${sumj[@]}" "--dependency=afterany:<array>" \
    "$FORCEARTGS_SLURM_DIR/summarize.sbatch"; echo
  [[ "$CROSS_DEMO" == 1 ]] && echo "+ cross_demo.sbatch after the summary (manifest $CROSS_DEMO_MANIFEST)"
  [[ "$SYNTHETIC" == 1 ]] && echo "+ synthetic.sbatch (independent)"; exit 0
fi
A=$("${arr[@]}")
S=$("${sumj[@]}" --dependency="afterany:$A" "$FORCEARTGS_SLURM_DIR/summarize.sbatch")
echo "episode array $A ($N tasks, $SLURM_MAX_PARALLEL at once); summary $S after it"
if [[ "$CROSS_DEMO" == 1 ]]; then
  [[ -f "$CROSS_DEMO_MANIFEST" ]] || { echo "CROSS_DEMO_MANIFEST $CROSS_DEMO_MANIFEST missing" >&2; exit 2; }
  C=$(sbatch --parsable "${common[@]}" --job-name=forceartgs_cross_demo --dependency="afterany:$S" --gres="$SLURM_GRES" \
        --cpus-per-task=4 --mem=32G --time="$CROSS_DEMO_TIME" --output="$FORCEARTGS_OUT/slurm_logs/cross_demo_%j.out" \
        $SLURM_EXTRA "$FORCEARTGS_SLURM_DIR/cross_demo.sbatch")
  echo "cross-demo validation $C after the summary"
fi
if [[ "$SYNTHETIC" == 1 ]]; then
  Y=$(sbatch --parsable "${common[@]}" --job-name=forceartgs_synthetic --gres="$SLURM_GRES" --cpus-per-task="$SLURM_CPUS" \
        --mem=32G --time="$CROSS_DEMO_TIME" --output="$FORCEARTGS_OUT/slurm_logs/synthetic_%j.out" \
        $SLURM_EXTRA "$FORCEARTGS_SLURM_DIR/synthetic.sbatch")
  echo "synthetic benchmark $Y (independent)"
fi
echo "monitor: squeue -u $USER ; tail -f $FORCEARTGS_OUT/real_full_to_estimated_assets.log"
