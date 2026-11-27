"""Locate and pin ForceArtGS-Physics (never vendored into this repository: it is a private research handoff without a
redistribution license, see its LICENSE / NOTICE).

Default: the package INSTALLED in this interpreter (the forceartgs env; ``envs/create_forceartgs_env.sh`` installs it
from the pinned commit). Its commit is read from pip's ``direct_url.json`` and every installed file is checked against
pip's ``RECORD`` sha256 (a locally edited install counts as dirty).
Override: an explicit checkout path (``--physics-library DIR``) or env ``FORCEARTGS_PHYSICS_ROOT`` -> that git checkout
(``<root>/src`` on sys.path). Either way the source must be the pinned commit and clean unless ``allow_unpinned``
(recorded in the provenance, never silent).
"""
from __future__ import annotations

import base64
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "f3824316272d85cb8d788166372c81ac3105ca7c"     # andrearichichi/ForceArtGS-Physics main, tag v0.1.0
PINNED_REPO = "https://github.com/andrearichichi/ForceArtGS-Physics"
DIST = "forceartgs-physics"
LICENSE_STATUS = "private research handoff, no redistribution license (LICENSE/NOTICE)"


class PhysicsLibraryError(RuntimeError):
    pass


_STATE: dict = {}


def _git(root: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def _checkout(root: Path):
    if not (root / "src" / "forceartgs_physics" / "__init__.py").is_file():
        raise PhysicsLibraryError(f"ForceArtGS-Physics not found at {root} (clone {PINNED_REPO} and check out "
                                  f"{PINNED_COMMIT}, or drop the override to use the forceartgs env's install)")
    try:
        head = _git(root, "rev-parse", "HEAD")
        dirty = [ln for ln in _git(root, "status", "--porcelain").splitlines() if ln.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise PhysicsLibraryError(f"{root}: cannot read the git revision ({e})") from None
    return head, len(dirty), str(root / "src"), "git_checkout"


def _installed():
    try:
        dist = md.distribution(DIST)
    except md.PackageNotFoundError:
        raise PhysicsLibraryError(f"ForceArtGS-Physics is not installed in {sys.executable}: run ForceArtGS in the "
                                  "forceartgs env (envs/create_forceartgs_env.sh installs the pinned commit)") from None
    du = json.loads(dist.read_text("direct_url.json") or "{}")
    head = (du.get("vcs_info") or {}).get("commit_id")
    if not head:
        raise PhysicsLibraryError(f"installed {DIST} has no recorded git commit (direct_url.json: {du}); reinstall "
                                  f"from git at {PINNED_COMMIT}")
    dirty = 0
    for f in dist.files or ():
        if f.hash is None or f.hash.mode != "sha256":
            continue
        p = Path(dist.locate_file(f))
        if not p.is_file():
            dirty += 1
            continue
        h = base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).rstrip(b"=").decode()
        dirty += h != f.hash.value
    return head, dirty, None, "installed_package"


def load(root=None, allow_unpinned: bool = False) -> dict:
    """Make ``forceartgs_physics`` importable at the pinned commit and return the provenance record. Idempotent."""
    root = root or os.environ.get("FORCEARTGS_PHYSICS_ROOT")
    key = str(Path(root).expanduser().resolve()) if root else f"installed:{sys.executable}"
    if _STATE.get("key") == key:
        return {k: v for k, v in _STATE.items() if k != "key"}
    head, dirty, src, source = _checkout(Path(key)) if root else _installed()
    pinned = head == PINNED_COMMIT and not dirty
    if not pinned and not allow_unpinned:
        raise PhysicsLibraryError(f"ForceArtGS-Physics ({source}) is at {head[:12]} (dirty files: {dirty}); G9 is "
                                  f"pinned to {PINNED_COMMIT[:12]} clean. Pass allow_unpinned to record a deviation.")
    if src and src not in sys.path:
        sys.path.insert(0, src)
    import forceartgs_physics                                   # noqa: F401  (import check)
    _STATE.clear()
    _STATE.update(key=key, root=str(Path(forceartgs_physics.__file__).resolve().parent), source=source, commit=head,
                  dirty_files=dirty, pinned=pinned, pinned_commit=PINNED_COMMIT, repository=PINNED_REPO,
                  version=getattr(forceartgs_physics, "__version__", None), license_status=LICENSE_STATUS)
    return {k: v for k, v in _STATE.items() if k != "key"}


def provenance() -> dict:
    if not _STATE:
        raise PhysicsLibraryError("physics library not loaded (call physics_lib.load first)")
    return {k: v for k, v in _STATE.items() if k != "key"}
