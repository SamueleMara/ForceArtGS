#!/usr/bin/env bash
# One SLURM array task = ONE episode through the whole chain (reconstruction -> q(t) extension -> M, c estimation ->
# URDF/USD + 3DGS). Resumable: a requeued / resubmitted task skips finished steps and resumes 3DGS from checkpoints.
# Submitted by slurm/submit_all.sh (it passes FORCEARTGS_SLURM_DIR and EPISODE_LIST).
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
source "$FORCEARTGS_SLURM_DIR/config.sh"
EP="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EPISODE_LIST")"
[[ -n "$EP" ]] || { echo "no episode at index $SLURM_ARRAY_TASK_ID in $EPISODE_LIST" >&2; exit 2; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$SLURM_CPUS}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$SLURM_CPUS}"
echo "$(date '+%F %T') task $SLURM_ARRAY_TASK_ID episode $EP host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?}"
cd "$FORCEARTGS_REPO"
exec "$FORCEARTGS_PYTHON" pipelines/real_full_to_estimated_assets.py --budget "$BUDGET" --scenes "$EP" \
     --out "$FORCEARTGS_OUT" --pinn-device "$PINN_DEVICE"
