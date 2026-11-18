#!/usr/bin/env python3
"""Move finished joint-aware runs from the first layout (<out>/<scene>[/<budget>]/branches/<branch>) to the FreeArtGS
batch layout used now:  <out>/<job>/ (run level)  and  <out>/<job>_<branch>/{part_0,part_1,twopart_blend,splatfacto-art}
with <job> = <scene>_multiview_cams001-002_i<a>-<b>_fall_s<k>_joint_aware_<budget>.

Pure renames on the same filesystem plus a rewrite of the stored absolute paths (JSON / CSV / text, and the
component-list PosixPaths of nerfstudio config.yml). parttwin_handoff/ and parttwin_assets/ are moved but NOT edited
(their files are hash-locked; their recorded source paths keep the pre-migration location -- logged). Refuses to touch
a run that is still in progress (no final/compare stage) and never overwrites an existing target.

    python tools/migrate_joint_aware_layout.py --out-root OUT --scene NAME --frames a:b:k [--dry-run]
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelines"))
from joint_aware_reconstruction import job_name  # noqa: E402

LOCKED = ("parttwin_handoff", "parttwin_assets")
TEXT = (".json", ".csv", ".txt", ".yaml", ".yml", ".log")


def rewrite_tree(root: Path, pairs, log):
    """pairs: [(old_parts list, new_parts list)] for PosixPath component lists, plus string prefixes."""
    n = 0
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix not in TEXT or any(x in p.relative_to(root).parts for x in LOCKED):
            continue
        s = p.read_text(errors="strict")
        t = s
        for old, new in pairs:
            t = t.replace("/".join(old), "/".join(new))
            if p.name == "config.yml":           # PosixPath serialised as one "- part" line per component
                pat = r"((?:^|\n)([ \t]*)- )" + r"\n\2- ".join(re.escape(x) for x in old) + r"(?=\n)"
                t = re.sub(pat, lambda m: m.group(1) + ("\n" + m.group(2) + "- ").join(new), t)
        if t != s:
            p.write_text(t)
            n += 1
    log(f"  rewrote paths in {n} files under {root.name}")


def migrate(out: Path, scene: str, frames: str, dry: bool):
    log = print
    sdir = out / scene
    if not sdir.is_dir():
        log(f"{scene}: nothing to migrate")
        return
    runs = []
    if (sdir / "run_manifest.json").is_file():                       # first smoke layout: <scene>/ is the run
        runs.append(("smoke", sdir))
    for b in ("smoke", "coarse", "full"):
        if (sdir / b / "run_manifest.json").is_file():
            runs.append((b, sdir / b))
    for tag, old_root in runs:
        man = json.loads((old_root / "run_manifest.json").read_text())
        st = man.get("stages", {})
        if st.get("compare", {}).get("status") is None and st.get("final", {}).get("status") is None:
            log(f"{scene}/{tag}: not finished -- skipped")
            continue
        cams = man.get("cameras") or ["cam_001", "cam_002"]
        job = job_name(scene, cams, man.get("frames", frames), tag)
        new_root = out / job
        moves = [(old_root / "branches" / k, out / f"{job}_{k}") for k in ("revolute", "prismatic")
                 if (old_root / "branches" / k).is_dir()]
        for _, dst in moves + [(None, new_root)]:
            if dst.exists():
                raise SystemExit(f"target exists, refusing: {dst}")
        log(f"{scene}/{tag} -> {job}")
        for src, dst in moves:
            log(f"  mv {src} -> {dst}")
        if dry:
            continue
        for src, dst in moves:
            src.rename(dst)
        (old_root / "branches").rmdir()
        if old_root == sdir:                       # smoke layout: run files live in <scene>/ itself
            new_root.mkdir()
            for p in list(sdir.iterdir()):
                if p.name not in ("smoke", "coarse", "full"):
                    p.rename(new_root / p.name)
        else:
            old_root.rename(new_root)
        base = [x for x in str(out).split("/") if x] + [scene]
        pairs = [(["/"] if False else ["", *base, *([tag] if old_root != sdir else []), "branches", k],
                  ["", *[x for x in str(out).split("/") if x], f"{job}_{k}"]) for k in ("revolute", "prismatic")]
        pairs.append((["", *base, *([tag] if old_root != sdir else [])], ["", *[x for x in str(out).split("/") if x], job]))
        # config.yml component lists start with "/" then the parts; strings start with "" -> "/"
        yml_pairs = [([*o[1:]], [*n[1:]]) for o, n in pairs]
        str_pairs = pairs
        for d in [new_root] + [dst for _, dst in moves]:
            for o, n in yml_pairs:
                rewrite_tree(d, [(o, n)], lambda *_: None)
            rewrite_tree(d, str_pairs, log)
        man_p = new_root / "run_manifest.json"
        m = json.loads(man_p.read_text())
        m.update(job=job, branch_dirs={k: str(out / f"{job}_{k}") for k in ("revolute", "prismatic")},
                 migrated_from=str(old_root), locked_dirs_with_premigration_paths=list(LOCKED))
        man_p.write_text(json.dumps(m, indent=2, sort_keys=True))
    try:
        sdir.rmdir()
        log(f"removed empty {sdir}")
    except OSError:
        log(f"{sdir} kept (not empty: {[p.name for p in sdir.iterdir()]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    migrate(a.out_root, a.scene, a.frames, a.dry_run)


if __name__ == "__main__":
    main()
