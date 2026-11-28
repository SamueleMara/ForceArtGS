#!/usr/bin/env python
"""Inventory of every real episode and the CANDIDATE object groups (CPU only, read only; spec section 2).

    python pipelines/inventory_real_object_episodes.py --real-root ROOT --out OUT/inventory [--write-manifest]

Writes episode_inventory.csv/json, object_groups_candidate.yaml and dataset_q_gt_provenance.json. --write-manifest
also writes configs/g9_real/object_group_manifest.yaml if it does not exist yet (never overwrites a reviewed one);
its groups stay `verified_same_physical_object: null` until the operator sets true.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    from forceartgs_g9.cross_demo import inventory as I
    import yaml
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--real-root", default=os.environ.get("FORCEARTGS_REAL_ROOT"), required=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="accepted for the spec's CLI; this tool never uses a GPU")
    a = ap.parse_args()
    if not a.real_root:
        ap.error("--real-root or FORCEARTGS_REAL_ROOT")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = I.scan(a.real_root)
    keys = sorted({k for r in eps for k in r})
    with open(out / "episode_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(eps)
    (out / "episode_inventory.json").write_text(json.dumps(eps, indent=1, default=str))
    man = I.candidate_manifest(eps, a.real_root)
    (out / "object_groups_candidate.yaml").write_text(yaml.safe_dump(man, sort_keys=False))
    (out / "dataset_q_gt_provenance.json").write_text(json.dumps(dict(
        n_with_independent_q_gt=sum(1 for r in eps if r.get("independent_q_gt")),
        reference_kind="RGBD_RECONSTRUCTED for every real episode unless an independent joint-state GT file is listed",
        files={r["episode"]: r["independent_q_gt"] for r in eps if r.get("independent_q_gt")}), indent=1))
    mp = REPO / "configs" / "g9_real" / "object_group_manifest.yaml"
    if a.write_manifest and not mp.exists():
        mp.write_text("# Populated from the filesystem scan; operator: set verified_same_physical_object: true per group\n"
                      + yaml.safe_dump(man, sort_keys=False))
    by = {}
    for r in eps:
        by.setdefault(r["object_key"], []).append(r)
    print(f"{len(eps)} episodes, {len(by)} candidate objects under {a.real_root}")
    for k, rs in sorted(by.items()):
        ok = [r["episode_id"] for r in rs if r["status"] == "OK"]
        print(f"  {k}: {len(rs)} episodes, {len(ok)} OK {'' if len(ok) >= 2 else '(INSUFFICIENT_EPISODES)'}")
        for r in rs:
            if r["status"] != "OK":
                print(f"     {r['episode_id']}: {r['status']}")
    print(out)


if __name__ == "__main__":
    main()
