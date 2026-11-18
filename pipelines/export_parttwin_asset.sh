#!/usr/bin/env bash
# G8 for one finished joint-aware run: ForceArtGS handoff -> shared PArtTwin1D builder (no URDF/USD/MJCF code here).
#   bash pipelines/export_parttwin_asset.sh <out_root>/<scene>/<budget> [--diagnostic-branch revolute|prismatic]
#        [--dynamics SPEC.yaml] [--registration REG.yaml]
# Env: PARTTWIN1D_ROOT (~/PArtTwin1D), PARTTWIN1D_PYTHON (parttwin_sim env), PARTTWIN_SIM_PYTHON (MuJoCo + USD env).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN="$1"; shift
BRANCH=""; DYN=(); REG=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --diagnostic-branch) BRANCH="$2"; shift 2 ;;
    --dynamics) DYN=(--dynamics "$2"); shift 2 ;;
    --registration) REG=(--registration "$2"); shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
PT_ROOT="${PARTTWIN1D_ROOT:-$HOME/PArtTwin1D}"
PT_PY="${PARTTWIN1D_PYTHON:-$HOME/miniconda3/envs/parttwin_sim/bin/python}"
export PARTTWIN_SIM_PYTHON="${PARTTWIN_SIM_PYTHON:-$HOME/.venvs/rigeuler-sim-export/bin/python}"
NAME="${BRANCH:+diagnostic_$BRANCH}"; NAME="${NAME:-accepted}"
H="$RUN/parttwin_handoff/$NAME"
"$HOME/miniconda3/envs/freeartgs/bin/python" tools/export_parttwin_handoff.py --run "$RUN" --output "$H" \
    ${BRANCH:+--diagnostic-branch "$BRANCH"}
PYTHONPATH="$PT_ROOT/src" "$PT_PY" -m parttwin1d.cli.build_asset --handoff "$H/articulated_3dgs_handoff.json" \
    --output "$RUN/parttwin_assets" ${BRANCH:+--allow-diagnostic} "${DYN[@]}" "${REG[@]}"
