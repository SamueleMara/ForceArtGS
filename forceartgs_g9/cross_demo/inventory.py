"""Real-dataset inventory and object groups (spec section 2). CPU only, read only.

Every child directory matching ^(?P<object_key>.+)_episode_(?P<episode_id>\\d{6})$ (IDs need not start at 0 or be
contiguous) is checked for the files the chain needs; duplicate pushes (identical robot recording hash or identical
frame timeline) are detected so they can never sit in train and test. Object groups come ONLY from a reviewed
manifest (configs/g9_real/object_group_manifest.yaml): the common prefix is a CANDIDATE grouping written by
`candidate_manifest`, with `verified_same_physical_object: null` until the operator confirms it; the transfer stage
refuses unverified groups (OBJECT_IDENTITY_UNVERIFIED).
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

PAT = re.compile(r"^(?P<object_key>.+)_episode_(?P<episode_id>\d{6})$")
CAMS = ("cam_001", "cam_002")
MANIFEST_SCHEMA = "forceartgs.cross_demo.object_groups/v1"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_episode(d: Path) -> dict:
    m = PAT.match(d.name)
    row = dict(episode=d.name, object_key=m["object_key"], episode_id=m["episode_id"], path=str(d))
    missing = []
    for cam in CAMS:
        n = len(list((d / "rgb" / cam).glob("*"))) if (d / "rgb" / cam).is_dir() else 0
        row[f"n_rgb_{cam}"] = n
        if n == 0:
            missing.append(f"RGB_{cam}")
    for name, rel in (("DEPTH", "depth_npy"), ("MASK", "mask")):
        if not (d / rel).is_dir() or not any((d / rel).iterdir()):
            missing.append(name)
    for name, rel in (("CAMERAS", "metadata/cameras.json"), ("TIMESTAMPS", "metadata/frame_timestamps.csv"),
                      ("FORCE_CONVENTION", "metadata/contact_force_convention.json")):
        if not (d / rel).is_file():
            missing.append(name)
    pq = sorted((d / "metadata" / "raw").glob("episode_*.parquet"))
    row["robot_recording"] = str(pq[0]) if pq else None
    row["robot_recording_sha256"] = _sha(pq[0]) if pq else None
    if not pq:
        missing.append("ROBOT_RECORDING")
    row["base_registration"] = (d / "metadata" / "robot_base_registration.json").is_file()
    ts = d / "metadata" / "frame_timestamps.csv"
    if ts.is_file():
        rows = list(csv.DictReader(open(ts)))
        row["n_frames"] = len(rows)
        tcol = next((k for k in ("timestamp_s", "time_s", "timestamp") if rows and k in rows[0]), None)
        row["duration_s"] = (float(rows[-1][tcol]) - float(rows[0][tcol])) if tcol and len(rows) > 1 else None
        row["timeline_sha256"] = _sha(ts)
    gt = sorted(d.glob("**/*joint_state*gt*")) + sorted(d.glob("**/*encoder*"))
    row["independent_q_gt"] = [str(p) for p in gt] or None
    row["missing"] = ";".join(missing)
    row["status"] = "OK" if not missing else "MISSING_" + "+".join(missing)
    return row


def scan(real_root) -> list:
    root = Path(real_root)
    eps = [check_episode(p) for p in sorted(root.iterdir()) if p.is_dir() and PAT.match(p.name)]
    by_rec, by_tl = {}, {}
    for r in eps:
        by_rec.setdefault(r.get("robot_recording_sha256"), []).append(r["episode"])
        by_tl.setdefault(r.get("timeline_sha256"), []).append(r["episode"])
    for r in eps:
        # a duplicate push = an IDENTICAL robot recording. frame_timestamps.csv is a relative uniform grid (0, 0.04,
        # ...): equal for every recording of equal length, so an equal timeline is only reported, never a duplicate
        dup = sorted(set(by_rec.get(r.get("robot_recording_sha256"), [])) - {r["episode"]}) \
            if r.get("robot_recording_sha256") else []
        r["same_timeline_as"] = len(by_tl.get(r.get("timeline_sha256"), [])) - 1
        r["duplicates"] = ";".join(dup)
        if dup:
            r["status"] = "DUPLICATE_OF_" + dup[0] if r["status"] == "OK" else r["status"]
    return eps


def candidate_manifest(eps: list, real_root) -> dict:
    """Prefix grouping as a CANDIDATE only: verified_same_physical_object stays null until the operator confirms."""
    groups = {}
    for r in eps:
        groups.setdefault(r["object_key"], []).append(r["episode"])
    return dict(schema=MANIFEST_SCHEMA, real_root=str(real_root),
                note="candidate grouping by the common prefix before _episode_XXXXXX; set "
                     "verified_same_physical_object: true only for groups that are the SAME physical object, same "
                     "fixed base / sensor / tool setup",
                objects=[dict(object_id=k, episodes=sorted(v), n_episodes=len(v), common_base_frame="base_link",
                              expected_joint_type=None, verified_same_physical_object=None)
                         for k, v in sorted(groups.items())])


def load_manifest(path) -> dict:
    import yaml
    m = yaml.safe_load(Path(path).read_text())
    if m.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path}: schema {m.get('schema')} != {MANIFEST_SCHEMA}")
    seen = {}
    for o in m["objects"]:
        for e in o["episodes"]:
            if e in seen:
                raise ValueError(f"{path}: episode {e} in both {seen[e]} and {o['object_id']}")
            seen[e] = o["object_id"]
    return m


def groups_for_transfer(manifest: dict, eps: list) -> list:
    """Object groups with their ELIGIBLE episodes (status OK, not a duplicate), or a refusal status."""
    st = {r["episode"]: r for r in eps}
    out = []
    for o in manifest["objects"]:
        good = [e for e in o["episodes"] if st.get(e, {}).get("status") == "OK"]
        g = dict(object_id=o["object_id"], episodes=o["episodes"], eligible=good,
                 excluded={e: st.get(e, {}).get("status", "NOT_ON_DISK") for e in o["episodes"] if e not in good})
        if o.get("verified_same_physical_object") is not True:
            g["status"] = "OBJECT_IDENTITY_UNVERIFIED"
        elif len(good) < 2:
            g["status"] = "INSUFFICIENT_EPISODES"
        else:
            g["status"] = "OK"
        out.append(g)
    return out
