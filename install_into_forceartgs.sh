#!/usr/bin/env bash
# Additive installer: NEVER overwrite existing local ForceArtGS code.
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "Usage: bash install_into_forceartgs.sh /path/to/ForceArtGS" >&2; exit 2; fi
repo="$(realpath "$1")"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$repo/forceartgs_g9/estimators/model_spec.py" || ! -f "$repo/forceartgs_g9/core.py" ]]; then
  echo "Not a valid ForceArtGS G9 checkout: $repo" >&2; exit 3
fi
mapfile -t files < <(cd "$root"; find forceartgs_g9 tests docs scripts -type f ! -path '*/__pycache__/*' ! -name '*.pyc' | sort)
for rel in "${files[@]}"; do
  if [[ -e "$repo/$rel" ]]; then echo "Refusing to overwrite existing file: $repo/$rel" >&2; exit 4; fi
done
for rel in "${files[@]}"; do
  install -d "$repo/$(dirname "$rel")"
  cp -- "$root/$rel" "$repo/$rel"
done
echo "Added ${#files[@]} files to $repo (no originals overwritten)."
echo "Run: cd '$repo' && python -m pytest -q tests/test_g9_physx*.py"
echo "Estimator API: forceartgs_g9.physx_refinement.api.run_forceartgs_physx_estimator"
echo "No PArtTwin1D files touched; no commits made."
