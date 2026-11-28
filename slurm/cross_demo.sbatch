#!/usr/bin/env bash
# After every episode: frozen single-demo models -> every other episode's measured input in PhysX (N x N per object,
# all methods of configs/g9_real/cross_demo_validation.yaml), then the report. Resumable (per-pair content keys).
set -euo pipefail
source "$FORCEARTGS_SLURM_DIR/config.sh"
cd "$FORCEARTGS_REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
exec "$FORCEARTGS_PYTHON" pipelines/real_all_episodes_cross_demo.py --out "$FORCEARTGS_OUT" \
     --real-root "$FORCEARTGS_REAL_ROOT" --object-manifest "$CROSS_DEMO_MANIFEST" --stage inventory,transfer,loo,report
