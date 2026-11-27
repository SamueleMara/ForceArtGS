#!/usr/bin/env bash
# Full ForceArtGS test suite in the forceartgs env with the clean process env (no ROS / login-shell leakage).
#   bash scripts/run_tests.sh [pytest args]       (no path given -> tests/, even with options)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${FORCEARTGS_PYTHON:-$HOME/miniconda3/envs/forceartgs/bin/python}"
eval "$("$PY" utils/forceartgs_env.py)"
exec "$PY" -m pytest -q -p no:cacheprovider -o testpaths=tests "$@"
