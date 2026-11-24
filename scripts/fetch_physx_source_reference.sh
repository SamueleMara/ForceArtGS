#!/usr/bin/env bash
# Optional PhysX SDK source checkout for studying exact solver behaviour.
# This is NOT the SAPIEN binary runtime and should never silently change that runtime.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/third_party/PhysX_reference"
TAG="107.3-physx-5.6.1"
if [[ -e "$OUT" ]]; then
  echo "Existing PhysX reference checkout at $OUT; not overwriting"
  exit 0
fi
git clone --depth 1 --filter=blob:none --branch "$TAG" \
  https://github.com/NVIDIA-Omniverse/PhysX.git "$OUT"
git -C "$OUT" rev-parse HEAD | tee "$OUT/.REFERENCE_SOURCE_COMMIT"
echo "Reference source only; SAPIEN / Isaac Sim each bundle their own PhysX versions"
