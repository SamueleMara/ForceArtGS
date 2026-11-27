#!/usr/bin/env bash
# The ForceArtGS conda env `forceartgs`: EVERYTHING the start-to-finish pipeline runs (3DGS training, joint discovery,
# q(t) extension, G9 estimation, SAPIEN/PhysX G10 replay, URDF/USD export, tests) in ONE env, with its own CUDA
# toolkit. Vanilla FreeArtGS keeps its own `freeartgs` env, untouched.
#
#   bash envs/create_forceartgs_env.sh            # creates `forceartgs` (refuses if it exists); then:
#   bash scripts/download_checkpoints.sh          # model weights into checkpoints/ (sha256-checked)
#
# Steps:
#   1. base: clone of the FreeArtGS CUDA stack (torch 2.5.1+cu124, tinycudann, pytorch3d, nerfacc: compiled, not
#      rebuilt). The clone only READS `freeartgs`; afterwards the two envs are independent.
#   2. CUDA 12.4.1 toolkit (nvcc, headers, libs) INSIDE the env: CUDA_HOME = the env prefix, no system CUDA.
#   3. ForceArtGS's OWN nerfstudio fork (reconstruction/nerfstudio, editable; not ~/FreeArtGS's).
#   4. gsplat 0.1.11 rebuilt WITH its compiled module gsplat.csrc (sm 8.9 by default): no JIT at run time.
#   5. SAPIEN 3.0.3 (PhysX), usd-core, mujoco, pycolmap, pytest.
#   6. ForceArtGS-Physics at the pinned commit (forceartgs_g9/physics_lib.py), from $FORCEARTGS_PHYSICS_SRC
#      (default: the local private-handoff checkout). Private, no redistribution license: installed into the env,
#      never committed to this repository.
# Pinned result: envs/forceartgs.pip-freeze.txt.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
ENV="${FORCEARTGS_ENV:-forceartgs}"
BASE="${FREEARTGS_ENV:-freeartgs}"
ARCH="${TORCH_CUDA_ARCH_LIST:-8.9}"
PHYS_COMMIT="$(sed -n 's/^PINNED_COMMIT = "\([0-9a-f]*\)".*/\1/p' forceartgs_g9/physics_lib.py)"
PHYS_SRC="${FORCEARTGS_PHYSICS_SRC:-git+file://$HOME/ForceArtGS-Physics}"
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV"; then echo "env $ENV already exists" >&2; exit 2; fi

"$CONDA" create -y -n "$ENV" --clone "$BASE"                                                   # 1
PREFIX="$("$CONDA" run -n "$ENV" python -c 'import sys; print(sys.prefix)')"
PY="$PREFIX/bin/python"
"$CONDA" install -y -n "$ENV" -c "nvidia/label/cuda-12.4.1" cuda-toolkit                        # 2
export CUDA_HOME="$PREFIX" PATH="$PREFIX/bin:$PATH" LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"
"$PY" -m pip uninstall -y nerfstudio                                                            # 3
"$PY" -m pip install --no-deps -e reconstruction/nerfstudio
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT                                                   # 4
BUILD_NO_CUDA=1 "$PY" -m pip download -q gsplat==0.1.11 --no-binary :all: --no-deps --no-build-isolation -d "$TMP"
tar xzf "$TMP/gsplat-0.1.11.tar.gz" -C "$TMP"
(cd "$TMP/gsplat-0.1.11" && TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="${MAX_JOBS:-4}" BUILD_NO_CUDA=0 \
   "$PY" -m pip install --no-deps --no-build-isolation --force-reinstall .)
"$PY" -m pip install "sapien==3.0.3" "usd-core==26.8" "mujoco==3.3.5" "pycolmap==3.12.5" "pytest==9.2.0"   # 5
"$PY" -m pip install --no-deps "$PHYS_SRC@$PHYS_COMMIT"                                         # 6

cd /tmp && PYTHONPATH="$REPO" "$PY" - "$REPO" <<'EOF'
import sys
from pathlib import Path
import torch, torch.utils.cpp_extension, sapien, mujoco, pycolmap, pytest, nerfstudio, gsplat
from gsplat import csrc                                   # compiled gsplat module: no JIT
from pxr import Usd, UsdPhysics
from forceartgs_g9 import physics_lib
repo = Path(sys.argv[1]).resolve()
assert Path(nerfstudio.__file__).resolve().is_relative_to(repo / "reconstruction/nerfstudio"), nerfstudio.__file__
prov = physics_lib.load()
assert prov["pinned"], prov
assert torch.cuda.is_available()
print("forceartgs env OK: torch", torch.__version__, "| CUDA toolkit", torch.utils.cpp_extension.CUDA_HOME,
      "| gsplat", gsplat.__version__, "compiled | sapien", sapien.__version__, "| ForceArtGS-Physics",
      prov["commit"][:12], "| nerfstudio", Path(nerfstudio.__file__).parent)
EOF
