#!/usr/bin/env bash
# Synthetic multi-push benchmark with KNOWN M, c (existing estimator, actual PhysX) + its N x N transfer report.
set -euo pipefail
source "$FORCEARTGS_SLURM_DIR/config.sh"
cd "$FORCEARTGS_REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
exec "$FORCEARTGS_PYTHON" pipelines/real_all_episodes_cross_demo.py --out "$FORCEARTGS_OUT/synthetic_cross_demo" \
     --real-root "$FORCEARTGS_REAL_ROOT" --object-manifest "$CROSS_DEMO_MANIFEST" --stage test-synthetic,report
