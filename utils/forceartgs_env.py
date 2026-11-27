"""The ONE process environment every ForceArtGS process runs in: the forceartgs conda env + this repository, nothing
inherited from the login shell (ROS / other repos on PYTHONPATH, foreign LD_LIBRARY_PATH, a system CUDA_HOME, ...).

    clean_env()                      -> dict for subprocess.run(env=...)
    python utils/forceartgs_env.py   -> `export ...` lines for shell scripts:  eval "$("$PY" utils/forceartgs_env.py)"

Explicit opt-in only: FORCEARTGS_EXTRA_PYTHONPATH is prepended to PYTHONPATH (instrumentation / debugging).
"""
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREFIX = Path(sys.executable).resolve().parents[1]          # the env of the interpreter running ForceArtGS
DROP_PREFIXES = ("ROS_", "AMENT_", "COLCON_", "RMW_", "GZ_", "IGN_")
DROP = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "LD_PRELOAD", "LD_LIBRARY_PATH", "CUDA_HOME",
        "CUDA_PATH", "CUDA_ROOT", "CMAKE_PREFIX_PATH", "PKG_CONFIG_PATH", "VIRTUAL_ENV", "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV", "CONDA_SHLVL", "CONDA_PROMPT_MODIFIER"}
SYSTEM_PATH = ("/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin")


def clean_env(base=None) -> dict:
    env = {k: v for k, v in dict(os.environ if base is None else base).items()
           if k not in DROP and not k.startswith(DROP_PREFIXES)}
    pp = [str(REPO / "reconstruction" / "nerfstudio"), str(REPO)]
    extra = env.get("FORCEARTGS_EXTRA_PYTHONPATH")
    env.update(
        PYTHONPATH=os.pathsep.join(([extra] if extra else []) + pp),
        PYTHONNOUSERSITE="1",                                # no ~/.local site-packages either
        CUDA_HOME=str(PREFIX),                               # CUDA 12.4 toolkit inside the env
        LD_LIBRARY_PATH=str(PREFIX / "lib"),
        PATH=os.pathsep.join([str(PREFIX / "bin"), *SYSTEM_PATH]),
        CONDA_PREFIX=str(PREFIX),
        FORCEARTGS_PYTHON=str(PREFIX / "bin" / "python"),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",                   # deterministic cuBLAS (reproducible GPU fits)
    )
    return env


if __name__ == "__main__":
    env = clean_env()
    unset = sorted(k for k in os.environ if k not in env)
    if unset:
        print("unset " + " ".join(unset))
    for k in ("PYTHONPATH", "PYTHONNOUSERSITE", "CUDA_HOME", "LD_LIBRARY_PATH", "PATH", "CONDA_PREFIX",
              "FORCEARTGS_PYTHON", "CUBLAS_WORKSPACE_CONFIG"):
        print(f"export {k}={shlex.quote(env[k])}")
