"""Single reader of a scene's ``metadata/robot_tool.json`` with the repo's operator overrides applied.

``configs/robot_tool_overrides.json`` lists, per scene, fields that REPLACE those of the scene file (an operator
decision, e.g. "run the white drawer with the bare force sensor EE settings"). The raw scene is never modified; the
returned dict carries ``tool_override`` (decision, status, caveat, replaced keys, override-file sha256) so every
output that used it can record it. Scenes without an entry are returned unchanged (``tool_override`` = None).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

OVERRIDES = Path(__file__).resolve().parents[2] / "configs" / "robot_tool_overrides.json"


def tool_override(scene_name: str, overrides: Path | None = None):
    p = Path(overrides) if overrides else OVERRIDES
    if not p.is_file():
        return None, None
    doc = json.loads(p.read_text())
    e = (doc.get("scenes") or {}).get(scene_name)
    return (e, hashlib.sha256(p.read_bytes()).hexdigest()) if e else (None, None)


def load_robot_tool(scene, *, required: bool = True, overrides: Path | None = None) -> dict | None:
    """``metadata/robot_tool.json`` of ``scene`` with the operator override (if any) applied."""
    scene = Path(scene)
    path = scene / "metadata" / "robot_tool.json"
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    tool = json.loads(path.read_text())
    e, sha = tool_override(scene.name, overrides)
    tool["tool_override"] = None
    if e:
        tool = dict(tool, **copy.deepcopy(e["set"]))
        tool["tool_override"] = dict(decision=e["decision"], status=e["status"], caveat=e.get("caveat"),
                                     replaced=sorted(e["set"]), file=str(overrides or OVERRIDES), sha256=sha)
    return tool
