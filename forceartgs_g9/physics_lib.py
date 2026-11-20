"""Locate and pin the external ForceArtGS-Physics checkout (never vendored: it is a private research handoff
without a redistribution license, see its LICENSE / NOTICE).

Resolution order: explicit path > env FORCEARTGS_PHYSICS_ROOT > ~/ForceArtGS-Physics. The checkout must be at the
pinned commit and clean unless ``allow_unpinned`` (recorded in the provenance, never silent).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "f3824316272d85cb8d788166372c81ac3105ca7c"     # andrearichichi/ForceArtGS-Physics main, tag v0.1.0
PINNED_REPO = "https://github.com/andrearichichi/ForceArtGS-Physics"
DEFAULT_ROOT = Path.home() / "ForceArtGS-Physics"


class PhysicsLibraryError(RuntimeError):
    pass


_STATE: dict = {}


def _git(root: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def load(root=None, allow_unpinned: bool = False) -> dict:
    """Put ``<root>/src`` on sys.path and return the provenance record. Idempotent for the same root."""
    root = Path(root or os.environ.get("FORCEARTGS_PHYSICS_ROOT") or DEFAULT_ROOT).expanduser().resolve()
    if _STATE.get("root") == str(root):
        return dict(_STATE)
    if not (root / "src" / "forceartgs_physics" / "__init__.py").is_file():
        raise PhysicsLibraryError(f"ForceArtGS-Physics not found at {root} (clone {PINNED_REPO} and check out "
                                  f"{PINNED_COMMIT}, or pass --physics-library)")
    try:
        head = _git(root, "rev-parse", "HEAD")
        dirty = [ln for ln in _git(root, "status", "--porcelain").splitlines() if ln.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise PhysicsLibraryError(f"{root}: cannot read the git revision ({e})") from None
    pinned = head == PINNED_COMMIT and not dirty
    if not pinned and not allow_unpinned:
        raise PhysicsLibraryError(f"{root} is at {head[:12]} (dirty files: {len(dirty)}); G9 is pinned to "
                                  f"{PINNED_COMMIT[:12]} clean. Pass allow_unpinned to record a deviation.")
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import forceartgs_physics                                   # noqa: F401  (import check)
    _STATE.clear()
    _STATE.update(root=str(root), commit=head, dirty_files=len(dirty), pinned=pinned, pinned_commit=PINNED_COMMIT,
                  repository=PINNED_REPO, version=getattr(forceartgs_physics, "__version__", None),
                  license_status="private research handoff, no redistribution license (LICENSE/NOTICE)")
    return dict(_STATE)


def provenance() -> dict:
    if not _STATE:
        raise PhysicsLibraryError("physics library not loaded (call physics_lib.load first)")
    return dict(_STATE)
