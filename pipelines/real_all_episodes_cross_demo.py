#!/usr/bin/env python
"""All-episode cross-demonstration PhysX validation (spec FORCEARTGS_CODEX_MULTI_EPISODE_CROSS_DEMO_...).

Stages (comma list or `all`):
  inventory       scan every *_episode_XXXXXX (CPU), candidate groups, statuses, duplicates
  test-synthetic  synthetic multi-push objects with KNOWN M, c (existing estimator, actual PhysX), N x N transfer
  estimate        per-episode chain (reconstruction -> q(t) -> M, c -> assets) SEQUENTIALLY via
                  pipelines/real_full_to_estimated_assets.py (on SLURM the episode array does this: slurm/submit_all.sh)
  transfer        frozen train-episode models -> every other episode's measured input in PhysX (no fit on targets)
  loo             pooled leave-one-out: POOLED_LOO_NOT_IMPLEMENTED (reported, never substituted)
  report          tables, matrices, figures, independent re-aggregation check
`all` = inventory,transfer,loo,report (+ estimate when --with-estimate). Outputs: <OUT>/cross_demo_validation/
per_pair/ (content-keyed, resumable) and <OUT>/cross_demo_validation/vNNN/ (append-only reports).

    python pipelines/real_all_episodes_cross_demo.py --out OUT --stage all [--object-manifest M] [--resume]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def log(out: Path, msg: str):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(out / "cross_demo.log", "a") as f:
        f.write(line + "\n")


def next_version(root: Path) -> Path:
    n = len(sorted(root.glob("v[0-9][0-9][0-9]")))
    d = root / f"v{n:03d}"
    d.mkdir(parents=True)
    return d


def git_state() -> dict:
    g = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
    diff = g("diff", "HEAD")
    return dict(head=g("rev-parse", "HEAD"), dirty_entries=len(g("status", "--porcelain").splitlines()),
                diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
                submodule=g("submodule", "status", "reconstruction/nerfstudio"))


def resolve_estimates(out: Path, episodes: list) -> dict:
    """Episode -> its finished rest-to-rest run, by the driver's own row.json (exact path); else the latest COMPARED
    append-only version (deterministic, labelled). None = no estimate."""
    res = {}
    for e in episodes:
        row = out / "estimated_assets" / e / "row.json"
        d = None
        if row.is_file():
            r = json.loads(row.read_text())
            if r.get("r2r") and str(r.get("r2r_status", "")).startswith("COMPARED"):
                d = r["r2r"]
        if d is None:
            for v in sorted((out / "rest_to_rest_compare" / e / "rest_to_rest_compare").glob("v[0-9][0-9][0-9]"),
                            reverse=True):
                st = v / "STATUS.json"
                if st.is_file() and json.loads(st.read_text()).get("status") == "COMPARED":
                    d = str(v)
                    break
        res[e] = d
    return res


def main():
    import yaml
    from forceartgs_g9.cross_demo import cv, inventory as I, report as RP, replay as R
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--real-root", default=os.environ.get("FORCEARTGS_REAL_ROOT",
                    "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"))
    ap.add_argument("--object-manifest", default=str(REPO / "configs/g9_real/object_group_manifest.yaml"))
    ap.add_argument("--config", default=str(REPO / "configs/g9_real/cross_demo_validation.yaml"))
    ap.add_argument("--out", required=True, help="the per-episode sweep root (same --out as the episode runs)")
    ap.add_argument("--budget", choices=("smoke", "coarse", "full"), default="full")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--with-estimate", action="store_true", help="`all` also runs the per-episode chain sequentially")
    ap.add_argument("--input-policy", choices=("current_g10", "force_only_at_tcp_ablation"), default="current_g10")
    ap.add_argument("--fit-method", default=None, help="one method; default: primary + secondary methods of --config")
    ap.add_argument("--train-episode", default=None)
    ap.add_argument("--test-episode", default=None)
    ap.add_argument("--protocol", choices=("single_demo_transfer", "pooled_loo", "both"), default="single_demo_transfer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic-full", action="store_true", help="test-synthetic at full (not quick) G10 budget")
    ap.add_argument("--synthetic-smoke", action="store_true", help="plumbing check: 2 pushes, prismatic only")
    ap.add_argument("--dry-run", action="store_true", help="inventory + plan only, no PhysX, no GPU")
    ap.add_argument("--resume", action="store_true", default=True)
    a = ap.parse_args()
    if a.input_policy != "current_g10":
        sys.exit("force_only_at_tcp_ablation: not wired into the cross-demo replay yet (refused, not substituted)")
    out = Path(a.out)
    root = out / "cross_demo_validation"
    root.mkdir(parents=True, exist_ok=True)
    stages = ["inventory", "transfer", "loo", "report"] if a.stage == "all" else a.stage.split(",")
    if a.stage == "all" and a.with_estimate:
        stages.insert(1, "estimate")
    cfg = yaml.safe_load(Path(a.config).read_text())
    methods = [a.fit_method] if a.fit_method else [cfg["primary_method"], *cfg.get("secondary_methods", [])]
    log(root, f"start stages={stages} methods={methods} out={out}")
    cache = root / "inventory_cache.json"                    # the scan lists every RGB dir: reuse it for an unchanged set
    names = sorted(p.name for p in Path(a.real_root).iterdir() if p.is_dir() and I.PAT.match(p.name))
    c0 = json.loads(cache.read_text()) if cache.is_file() else {}
    if c0.get("real_root") == a.real_root and c0.get("names") == names:
        eps = c0["episodes"]
    else:
        eps = I.scan(a.real_root)
        cache.write_text(json.dumps(dict(real_root=a.real_root, names=names, episodes=eps), default=str))
    man = I.load_manifest(a.object_manifest) if Path(a.object_manifest).is_file() else I.candidate_manifest(eps, a.real_root)
    groups = I.groups_for_transfer(man, eps)
    for g in groups:
        log(root, f"  {g['object_id']}: {g['status']}, {len(g['eligible'])}/{len(g['episodes'])} eligible")
    if a.dry_run:
        return
    if "estimate" in stages:
        todo = [e for g in groups if g["status"] == "OK" for e in g["eligible"]]
        subprocess.run([sys.executable, str(REPO / "pipelines/real_full_to_estimated_assets.py"), "--budget", a.budget,
                        "--out", str(out), "--scenes", *todo], check=False)
    cells, models, synth_rows = [], {}, []
    if "test-synthetic" in stages:
        from forceartgs_g9.cross_demo import synthetic as SY
        from forceartgs_g9.cross_demo import metrics as MX
        objs = (("prismatic", False), ("revolute", False), ("revolute", True))
        if a.synthetic_smoke:
            SY.PUSHES = dict(list(SY.PUSHES.items())[:2])
            objs = objs[:1]
        for jt, mis in objs:
            oid, res, truth = SY.build_object(root / "synthetic_training", jt, misspecified=mis,
                                              quick=not a.synthetic_full, log=lambda *_: None)
            recs = {e: SY.record_from_result(r, e) for e, r in res.items()}
            for m in methods:
                mods = {}
                for e, r in res.items():
                    try:
                        mods[(e, m)] = SY.model_from_result(r, e, m)
                    except (ValueError, KeyError):
                        pass
                c = cv.run_pairs(oid, {e: "synthetic" for e in res}, m, cfg, root, records=recs, models=mods,
                                 log=lambda s: log(root, s))
                for x in c:
                    x["synthetic"] = True
                cells += c
                models.update({(oid, e, m): md for (e, mm), md in mods.items()})
                if m == cfg["primary_method"]:
                    Ms = [md.M for md in mods.values()]
                    cs = [md.c for md in mods.values()]
                    held = [x["metrics"]["rmse"] for x in c if x["status"] == "OK" and x["train_id"] != x["test_id"]]
                    a_ = MX.aggregate_scalars(held)

                    def ms(x, gt=None):
                        g = MX.aggregate_scalars(x)
                        if g.get("status") != "OK":
                            return g.get("status")
                        sd = "n=1" if g["sd_sample"] is None else f"± {g['sd_sample']:.2g}"
                        return f"{g['mean']:.4g} {sd}" + ("" if gt is None else f" ({100 * (g['mean'] / gt - 1):+.1f}%)")
                    synth_rows.append(dict(object=oid, joint_type=jt, misspecified=mis, gt_M=truth["M"], gt_c=truth["c"],
                                           est_M=ms(Ms, truth["M"]), est_c=ms(cs, truth["c"]), heldout_rmse=ms(held),
                                           heldout_p95_rmse=a_.get("p95"), heldout_max_rmse=a_.get("max"),
                                           n_ok=len(held), n_attempted=sum(x["train_id"] != x["test_id"] for x in c)))
            groups.append(dict(object_id=oid, episodes=sorted(res), eligible=sorted(res), excluded={}, status="OK"))
    if "transfer" in stages:
        for g in groups:
            if g["status"] != "OK" or g["object_id"].startswith("synthetic_"):
                continue
            est = resolve_estimates(out, g["eligible"])
            log(root, f"{g['object_id']}: {sum(v is not None for v in est.values())}/{len(est)} episodes have an estimate")
            for m in methods:
                mods = {}
                c = cv.run_pairs(g["object_id"], est, m, cfg, root, models=mods, train_filter=a.train_episode,
                                 test_filter=a.test_episode, resume=a.resume, log=lambda s: log(root, s))
                cells += c
                models.update({(g["object_id"], e, mm): md for (e, mm), md in mods.items()})
    if "loo" in stages or a.protocol in ("pooled_loo", "both"):
        log(root, f"pooled_loo: {cv.pooled_loo()['status']}")
    if "report" in stages and cells:
        v = next_version(root)
        pair_dir = lambda c: (root / "per_pair" / c["object"] / f"train_{cv._eid(c['train_id'])}" /  # noqa: E731
                              f"test_{cv._eid(c['test_id'])}" / c["method"] / "seed0")
        res = RP.write(v, cells, groups=groups, models=models, cfg=cfg, manifest=man, inventory=eps,
                       synthetic=synth_rows or None, pair_dir=pair_dir, log=lambda s: log(root, s))
        (v / "manifest.json").write_text(json.dumps(dict(
            git=git_state(), real_root=a.real_root, out=str(out), methods=methods, seed=a.seed, stages=stages,
            config=cfg, engine="SAPIEN PhysX CPU in-process (1-DoF frozen replay)", n_cells=len(cells),
            validation=res["validation"], protocol=a.protocol), indent=1, default=str))
        (v / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (v / "object_groups_resolved.yaml").write_text(yaml.safe_dump(dict(groups=groups), sort_keys=False))
        log(root, f"report {v}: {res['n_ok']}/{res['n_cells']} cells OK, validation {res['validation']['status']}")
        if res["validation"]["status"] != "VALIDATED":
            sys.exit(3)
    elif "report" in stages:
        log(root, "report: no cells (nothing transferred yet)")


if __name__ == "__main__":
    main()
