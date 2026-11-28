#!/usr/bin/env bash
# After the episode array: SUMMARY.md / summary.json from every episode's row.json (CPU only).
set -euo pipefail
source "$FORCEARTGS_SLURM_DIR/config.sh"
cd "$FORCEARTGS_REPO"
exec "$FORCEARTGS_PYTHON" pipelines/real_full_to_estimated_assets.py --summarize-only --out "$FORCEARTGS_OUT"
