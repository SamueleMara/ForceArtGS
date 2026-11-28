"""Cross-demo report (spec section 9): CSV / JSONL / LaTeX tables, matrices and figures, metric definitions, and an
INDEPENDENT validation pass that recomputes every successful cell from its residuals.npz alone (no optimizer data, no
saved summary rows) and asserts agreement with the published numbers."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import metrics as MX

MATRIX_METRICS = ("rmse", "mean_abs_error", "signed_mean_error", "std_error_time")


def _j(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def _keys(o):
    """JSON-safe copy: tuple keys (e.g. exact pair weights keyed by (object, train, test)) -> 'a|b|c'."""
    if isinstance(o, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else k): _keys(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_keys(v) for v in o]
    return o


def _csv(path: Path, rows: list):
    rows = list(rows)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys or ["status"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(_keys(v), default=_j) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def _flat(cell: dict) -> dict:
    d = {k: v for k, v in cell.items() if k not in ("metrics", "alignment")}
    for k, v in (cell.get("metrics") or {}).items():
        d[k] = v
    al = cell.get("alignment") or {}
    d.update(axis_angle_deg=al.get("axis_angle_deg"), hinge_line_distance_m=al.get("hinge_line_distance_m"),
             axis_sign=al.get("sign"))
    return d


def _fmt(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}g}"


def _ms(a: dict, nd=3):
    if not a or a.get("status") != "OK":
        return "n/a"
    sd = a.get("sd_sample")
    return f"{_fmt(a['mean'], nd)} ± {_fmt(sd, nd)}" if sd is not None else f"{_fmt(a['mean'], nd)} (n=1)"


def matrices(out: Path, cells: list, object_id: str, method: str, episodes: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = [e.rsplit("_episode_", 1)[-1] for e in episodes]
    by = {(c["train_id"], c["test_id"]): c for c in cells}
    for m in MATRIX_METRICS:
        A = np.full((len(episodes), len(episodes)), np.nan)
        st = [["" for _ in episodes] for _ in episodes]
        for a, i in enumerate(episodes):
            for b, j in enumerate(episodes):
                c = by.get((i, j))
                st[a][b] = "MISSING" if c is None else c["status"]
                v = None if c is None else (c.get("metrics") or {}).get(m)
                if c is not None and c["status"] == "OK" and v is not None:
                    A[a, b] = v
        with open(out / f"cross_demo_matrix_{object_id}_{method}_{m}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["train\\test"] + ids)
            for a, i in enumerate(ids):
                w.writerow([i] + [(_fmt(A[a, b], 6) if st[a][b] == "OK" else st[a][b]) for b in range(len(ids))])
        fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(ids), 0.8 + 0.5 * len(ids)))
        fin = A[np.isfinite(A)]
        lim = (np.nanmax(np.abs(fin)) if len(fin) else 1.0) or 1.0
        im = ax.imshow(A, cmap="RdBu_r" if m == "signed_mean_error" else "viridis",
                       vmin=-lim if m == "signed_mean_error" else 0, vmax=lim)
        for a in range(len(ids)):
            for b in range(len(ids)):
                txt = ("TRAIN\n" if a == b else "") + (_fmt(A[a, b], 2) if np.isfinite(A[a, b]) else st[a][b][:10])
                ax.text(b, a, txt, ha="center", va="center", fontsize=6, color="w" if np.isfinite(A[a, b]) else "r")
        ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=7)
        ax.set_yticks(range(len(ids)), ids, fontsize=7)
        ax.set_xlabel("test episode (its own measured input)")
        ax.set_ylabel("train episode (frozen M, c)")
        unit = next((c.get("metrics", {}).get("q_units") for c in cells if c.get("metrics")), "")
        ax.set_title(f"{object_id} {method}\n{m} [{unit}]", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out / f"cross_demo_matrix_{object_id}_{method}_{m}.png", dpi=110)
        plt.close(fig)


def pair_figure(out: Path, cell: dict, pdir: Path, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(pdir / "replay.npz")
    m = cell["metrics"]
    u = m.get("q_units", "")
    fig, ax = plt.subplots(4, 1, figsize=(7, 9), sharex=True)
    ax[0].plot(z["t_s"], z["q_ref"], "k.", ms=3, label="q_ref (RGB-D, train gauge)")
    ax[0].plot(z["t_s"], z["q_sim"], "C1-", label="q_sim (PhysX, frozen train M, c)")
    ax[0].axvline(float(z["T_stop"]), color="0.6", ls="--", lw=0.8)
    ax[0].set_ylabel(f"q [{u}]")
    ax[0].legend(fontsize=7)
    e = z["q_sim"] - z["q_ref"]
    ax[1].plot(z["t_s"], e, "C3-")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel(f"signed residual [{u}]")
    ax[2].plot(z["t_s"], np.abs(e), "C4-")
    ax[2].set_ylabel(f"|residual| [{u}]")
    ax[3].step(z["input_t_s"], z["input_Q"], where="post", lw=0.8)
    ax[3].set_ylabel("Q [N m]" if u == "rad" else "Q [N]")
    ax[3].set_xlabel("time [s]")
    lab = "TRAIN (diagonal)" if cell["train_id"] == cell["test_id"] else "HELD-OUT"
    fig.suptitle(f"{cell['object']} train {cell['train_id'].rsplit('_', 1)[-1]} -> test "
                 f"{cell['test_id'].rsplit('_', 1)[-1]} ({lab}, {cell['method']})\nMAE {_fmt(m['mean_abs_error'])}  "
                 f"RMSE {_fmt(m['rmse'])}  bias {_fmt(m['signed_mean_error'])}  temporal SD {_fmt(m['std_error_time'])}"
                 f"  P95 {_fmt(m['p95_abs_error'])}  max {_fmt(m['max_abs_error'])} [{u}]", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / f"{cell['object']}_{tag}_q_overlay.png", dpi=110)
    plt.close(fig)


def validate(cells: list, pair_dir) -> dict:
    """Recompute every successful cell from residuals.npz alone; compare with the published metric values."""
    worst, n, bad = 0.0, 0, []
    for c in cells:
        if c["status"] != "OK":
            continue
        z = np.load(pair_dir(c) / "residuals.npz")
        w, e = z["w_time"], z["e_signed"]
        W = w.sum()
        rec = dict(rmse=math.sqrt(float((w * e ** 2).sum() / W)), mean_abs_error=float((w * np.abs(e)).sum() / W),
                   signed_mean_error=float((w * e).sum() / W))
        rec["std_error_time"] = math.sqrt(max(float((w * (e - rec["signed_mean_error"]) ** 2).sum() / W), 0.0))
        for k, v in rec.items():
            d = abs(v - c["metrics"][k]) / max(abs(v), 1e-12)
            worst = max(worst, d)
            if d > 1e-9:
                bad.append((c["object"], c["train_id"], c["test_id"], k, v, c["metrics"][k]))
        n += 1
    return dict(status="VALIDATED" if not bad else "VALIDATION_FAILED", n_cells_recomputed=n, worst_rel_diff=worst,
                mismatches=bad[:20])


def write(out: Path, cells: list, *, groups: list, models: dict, cfg: dict, manifest: dict, inventory=None,
          synthetic=None, pair_dir=None, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "metric_definitions.json").write_text(json.dumps(dict(
        version=MX.METRICS_VERSION, quantile_rule=MX.QUANTILE_RULE, definitions=MX.METRIC_DEFINITIONS), indent=1,
        default=_j))
    with open(out / "cross_demo_pairs.jsonl", "w") as f:
        for c in cells:
            f.write(json.dumps(_keys(c), default=_j) + "\n")
    _csv(out / "cross_demo_pairs.csv", [_flat(c) for c in cells])
    _csv(out / "geometry_alignment.csv", [dict(object=c["object"], train_id=c["train_id"], test_id=c["test_id"],
                                               **{k: v for k, v in (c.get("alignment") or {}).items()})
                                          for c in cells if c["method"] == cfg["primary_method"]])
    _csv(out / "train_models.csv", [{**{k: v for k, v in md.describe().items() if not isinstance(v, (dict, list))},
                                     "object": o, "train_id": i, "method": m}
                                    for (o, i, m), md in sorted(models.items())])
    if inventory is not None:
        _csv(out / "episode_inventory.csv", inventory)
    _csv(out / "episode_status.csv", [dict(object=g["object_id"], episode=e, group_status=g["status"],
                                           eligible=e in g["eligible"], reason=g["excluded"].get(e, ""))
                                      for g in groups for e in g["episodes"]])
    methods = sorted({c["method"] for c in cells})
    rows_train, rows_test, rows_obj, rows_sweep, rows_dev, rows_fail, rows_par, rows_cmp = [], [], [], [], [], [], [], []
    sweeps = {}
    for m in methods:
        cm = [c for c in cells if c["method"] == m]
        eps_by_obj = {g["object_id"]: g["eligible"] for g in groups if g["status"] in ("OK", "INSUFFICIENT_EPISODES")}
        try:
            sw = MX.aggregate_sweep(cm, episodes_by_object=eps_by_obj)
        except ValueError as ex:
            sw = dict(status="AGGREGATION_REFUSED", reason=str(ex))
            log(f"[cross-demo report] {m}: {ex}")
        sweeps[m] = sw
        for o, blk in (sw.get("objects") or {}).items():
            for i, r in (blk.get("rows") or {}).items():
                rows_train.append(dict(object=o, method=m, train_id=i, **r["counts"],
                                       rmse=_ms(r["metrics"].get("rmse")), mae=_ms(r["metrics"].get("mean_abs_error")),
                                       train_diagonal_rmse=(r.get("train_diagonal") or {}).get("rmse"),
                                       generalization_gap_rmse=r.get("generalization_gap_rmse"),
                                       detail=r["metrics"]))
            for j, r in (blk.get("columns") or {}).items():
                rows_test.append(dict(object=o, method=m, test_id=j, **r["counts"], rmse=_ms(r["metrics"].get("rmse")),
                                      target_duration_s=r.get("target_duration_s"), detail=r["metrics"]))
            off = blk.get("offdiag") or {}
            rows_obj.append(dict(object=o, method=m, status=blk.get("status"), joint_type=blk.get("joint_type"),
                                 q_units=blk.get("q_units"), n_episodes=blk.get("n_episodes"), **(off.get("counts") or {}),
                                 offdiag_rmse=_ms((off.get("metrics") or {}).get("rmse")),
                                 offdiag_mae=_ms((off.get("metrics") or {}).get("mean_abs_error")),
                                 diag_rmse=_ms(((blk.get("diagonal") or {}).get("metrics") or {}).get("rmse")),
                                 generalization_gap_rmse=blk.get("generalization_gap_rmse"), detail=off.get("metrics")))
        for jt, b in (sw.get("by_joint_type") or {}).items():
            for metric, agg in (b.get("metrics") or {}).items():
                for scheme in ("macro_object", "macro_train", "micro_pair"):
                    a = agg.get(scheme) or {}
                    rows_sweep.append(dict(method=m, joint_type=jt, unit=agg.get("unit"), metric=metric, scheme=scheme,
                                           n=a.get("n"), mean=a.get("mean"), sd_sample=a.get("sd_sample"),
                                           p50=a.get("p50"), p95=a.get("p95"), max=a.get("max"),
                                           ci_status=b.get("ci_status")))
            mt = b.get("micro_time") or {}
            rows_sweep.append(dict(method=m, joint_type=jt, unit=b.get("q_units"), metric="rmse/mae", scheme="micro_time",
                                   mean=mt.get("rmse_time_weighted"), p50=mt.get("mae_time_weighted"),
                                   n=b.get("counts", {}).get("n_success"), ci_status="DURATION-WEIGHTED"))
            rows_dev.append({**(b.get("pooled_residual_deviance") or {}), "method": m, "joint_type": jt})
        rows_fail += [{**f, "method": m} if isinstance(f, dict) else dict(method=m, item=f)
                      for f in (sw.get("failure_inventory") or [])]
        if m != cfg["primary_method"]:
            try:
                rows_cmp.append({**MX.paired_method_comparison(cells, m, cfg["primary_method"]), "method": m,
                                 "baseline": cfg["primary_method"]})
            except ValueError as ex:
                rows_cmp.append(dict(method=m, baseline=cfg["primary_method"], status="REFUSED", reason=str(ex)))
    for g in groups:
        ms = [md for (o, i, m), md in models.items() if o == g["object_id"] and m == cfg["primary_method"]]
        if ms:
            pv = MX.parameter_variation([dict(train_id=md.name, M=md.M, c=md.c, joint_type=md.joint_type) for md in ms])
            rows_par.append({**pv, "object": g["object_id"], "method": cfg["primary_method"]})
    _csv(out / "per_train_episode_summary.csv", rows_train)
    _csv(out / "per_test_episode_summary.csv", rows_test)
    _csv(out / "per_object_summary.csv", rows_obj)
    _csv(out / "per_object_parameter_variation.csv", rows_par)
    _csv(out / "sweep_global_summary.csv", rows_sweep)
    (out / "sweep_global_summary.json").write_text(json.dumps(_keys(sweeps), indent=1, default=_j))
    _csv(out / "sweep_pooled_residual_deviance.csv", rows_dev)
    _csv(out / "sweep_failure_inventory.csv", rows_fail)
    _csv(out / "method_paired_comparisons.csv", rows_cmp)
    _csv(out / "single_demo_cv_summary.csv", [r for r in rows_obj if r["method"] == cfg["primary_method"]])
    _csv(out / "pooled_loo_summary.csv", [dict(protocol="pooled_loo", status="POOLED_LOO_NOT_IMPLEMENTED")])
    _csv(out / "real_trajectory_transfer_table.csv",
         [_flat(c) for c in cells if c["method"] == cfg["primary_method"] and not c.get("synthetic")])
    if synthetic:
        _csv(out / "synthetic_gt_parameter_table.csv", synthetic)
    # ------------------------------------------------------------------ matrices + figures (primary method)
    for g in groups:
        cg = [c for c in cells if c["object"] == g["object_id"]]
        for m in methods:
            cm = [c for c in cg if c["method"] == m]
            if cm and g["status"] == "OK":
                try:
                    matrices(out, cm, g["object_id"], m, g["eligible"])
                except Exception as ex:
                    log(f"[cross-demo report] matrix {g['object_id']} {m}: {type(ex).__name__}: {ex}")
        ok = [c for c in cg if c["method"] == cfg["primary_method"] and c["status"] == "OK"
              and c["train_id"] != c["test_id"]]
        if ok and pair_dir:
            ok.sort(key=lambda c: c["metrics"]["rmse"])
            for tag, c in (("best_heldout", ok[0]), ("worst_heldout", ok[-1])):
                try:
                    pair_figure(out, c, pair_dir(c), f"{tag}_train{c['train_id'][-6:]}_test{c['test_id'][-6:]}")
                except Exception as ex:
                    log(f"[cross-demo report] figure: {type(ex).__name__}: {ex}")
    val = validate(cells, pair_dir) if pair_dir else dict(status="NOT_RUN")
    tables_tex(out, rows_obj, rows_sweep, cfg, synthetic)
    report_md(out, cells, groups, rows_obj, rows_sweep, rows_par, val, cfg, manifest)
    return dict(validation=val, n_cells=len(cells), n_ok=sum(c["status"] == "OK" for c in cells))


def tables_tex(out: Path, rows_obj, rows_sweep, cfg, synthetic):
    L = ["% forceartgs cross-demo validation -- generated; RGB-D references are labelled RGBD_RECONSTRUCTED", "",
         "\\begin{tabular}{llrlllll}", "\\hline",
         "Object & Joint & N & ok/attempted & off-diag RMSE & off-diag MAE & train RMSE & gap \\\\", "\\hline"]
    for r in rows_obj:
        if r["method"] != cfg["primary_method"]:
            continue
        L.append(f"{r['object'].replace('_', ' ')} & {r.get('joint_type') or '-'} [{r.get('q_units') or '-'}] & "
                 f"{r.get('n_episodes')} & {r.get('n_success', 0)}/{r.get('n_attempted', 0)} & {r['offdiag_rmse']} & "
                 f"{r['offdiag_mae']} & {r['diag_rmse']} & {_fmt(r.get('generalization_gap_rmse'), 3)} \\\\")
    L += ["\\hline", "\\end{tabular}", "", "\\begin{tabular}{lllrlll}", "\\hline",
          "Joint & Metric & Scheme & n & mean & SD & P95 \\\\", "\\hline"]
    for r in rows_sweep:
        if r["method"] == cfg["primary_method"] and r["metric"] in ("rmse", "mean_abs_error", "rmse/mae"):
            L.append(f"{r['joint_type']} [{r.get('unit')}] & {r['metric'].replace('_', ' ')} & "
                     f"{r['scheme'].replace('_', ' ')} & {r.get('n')} & {_fmt(r.get('mean'), 3)} & "
                     f"{_fmt(r.get('sd_sample'), 3)} & {_fmt(r.get('p95'), 3)} \\\\")
    L += ["\\hline", "\\end{tabular}"]
    if synthetic:
        L += ["", "% synthetic ground truth", "\\begin{tabular}{llllll}", "\\hline",
              "Object & GT M & GT c & est. M & est. c & held-out RMSE \\\\", "\\hline"]
        for s in synthetic:
            L.append(f"{s['object'].replace('_', ' ')} & {_fmt(s['gt_M'])} & {_fmt(s['gt_c'])} & {s['est_M']} & "
                     f"{s['est_c']} & {s['heldout_rmse']} \\\\")
        L += ["\\hline", "\\end{tabular}"]
    (out / "tables.tex").write_text("\n".join(L) + "\n")


def report_md(out: Path, cells, groups, rows_obj, rows_sweep, rows_par, val, cfg, manifest):
    n_ok = sum(c["status"] == "OK" for c in cells)
    fails = {}
    for c in cells:
        if c["status"] != "OK":
            fails[c["status"]] = fails.get(c["status"], 0) + 1
    L = ["# Cross-demonstration PhysX validation", "",
         f"Protocol: **single-demonstration transfer** (train on ONE episode, freeze M, c, URDF and joint; replay "
         f"every other episode's own measured input in PhysX; nothing fitted on the target). Primary method "
         f"`{cfg['primary_method']}`. Pooled leave-one-out: POOLED_LOO_NOT_IMPLEMENTED (separate protocol, never "
         f"mixed). Real references are RGBD_RECONSTRUCTED (not encoder ground truth).", "",
         f"Cells: {len(cells)} attempted, **{n_ok} OK**, failures by status: {fails or 'none'}. "
         f"Independent re-aggregation from residuals.npz: **{val.get('status')}** "
         f"({val.get('n_cells_recomputed')} cells, worst relative difference {val.get('worst_rel_diff')}).", "",
         "## Object groups", "", "| object | status | eligible episodes | excluded |", "|---|---|---|---|"]
    for g in groups:
        L.append(f"| {g['object_id']} | {g['status']} | {len(g['eligible'])} | "
                 f"{', '.join(f'{k[-6:]}:{v}' for k, v in g['excluded'].items()) or '-'} |")
    L += ["", "## Per object (primary method, off-diagonal = held-out only)", "",
          "| object | joint | N | ok / attempted | off-diag RMSE (mean ± SD) | off-diag MAE | train (diagonal) RMSE | "
          "gap |", "|---|---|---|---|---|---|---|---|"]
    for r in rows_obj:
        if r["method"] == cfg["primary_method"]:
            L.append(f"| {r['object']} | {r.get('joint_type')} [{r.get('q_units')}] | {r.get('n_episodes')} | "
                     f"{r.get('n_success', 0)}/{r.get('n_attempted', 0)} | {r['offdiag_rmse']} | {r['offdiag_mae']} | "
                     f"{r['diag_rmse']} | {_fmt(r.get('generalization_gap_rmse'), 3)} |")
    L += ["", "## Training-parameter variation across episodes", "",
          "| object | M mean ± SD (CV) | c mean ± SD (CV) |", "|---|---|---|"]
    for r in rows_par:
        M, c = r.get("M") or {}, r.get("c") or {}
        L.append(f"| {r['object']} | {_fmt(M.get('mean'))} ± {_fmt(M.get('sd_sample'))} ({_fmt(M.get('cv'), 3)}) | "
                 f"{_fmt(c.get('mean'))} ± {_fmt(c.get('sd_sample'))} ({_fmt(c.get('cv'), 3)}) |")
    L += ["", "## Limitations", "",
          "- Off-diagonal pairs share training parameters (rows) and target inputs (columns): they are correlated; "
          "descriptive statistics only, CIs need object/demo-level resampling (CI status reported per table).",
          "- The scoring interval includes the confirmed stationary tail ([t0, T_eval_end]); a 2-term inertia + "
          "damping model integrates any effort measured while the object is at rest (unmodelled stationary load).",
          "- Formulas: metric_definitions.json; every metric recomputable from per_pair/**/residuals.npz.", ""]
    (out / "report.md").write_text("\n".join(L))
