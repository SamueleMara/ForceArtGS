#!/usr/bin/env python3
"""Summarise G9 exploratory runs: per scene x model variant -> windows, effort SNR, estimates (CI95, identifiable),
scale-free combinations, fit quality, adequacy, frozen viscous baselines. Writes JSON + Markdown next to the runs.

    python tools/g9_exploratory_summary.py RUN_ROOT [--out PREFIX]
"""
import argparse
import json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).is_file() else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = []
    for run in sorted(Path(a.runs).glob("*_joint_aware_full")):
        for v in sorted((run / "dynamics" / "g9").glob("v[0-9][0-9][0-9]")):
            m = load(v / "g9_manifest.json")
            if not m or m.get("label", "").split(":")[0] not in ("EXPLORATORY_CONDITIONAL",):
                if not (m and m["status"] in ("BLOCKED",) and "exploratory" in json.dumps(load(v / "input_manifest.json") or {})):
                    continue
            fa = load(v / "force_preprocessing_report.json") or {}
            fit = load(v / "estimator_results" / "general_rollout_fit.json") or {}
            gi = (load(v / "general_identifiability.json") or {}).get("fit_identifiability") or {}
            ad = ((load(v / "model_adequacy.json") or {}).get("adequacy") or {})
            bl = load(v / "estimator_results" / "frozen_viscous_baselines.json") or {}
            terms = [t for t, s in m["terms"].items() if s == "enabled"]
            win = fa.get("interaction_windows") or []
            rows.append(dict(scene=m["scene"], version=v.name, terms=terms, status=m["status"], label=m["label"],
                             kinematics=m["status_categories"]["KINEMATICS_NOT_ACCEPTED"],
                             windows=[dict(rows=w.get("source_rows"), snr=(w.get("effort_snr") or {}).get("snr"),
                                           rms=(w.get("effort_snr") or {}).get("rms_effort_while_moving"),
                                           noise=(w.get("effort_snr") or {}).get("noise_equivalent_effort"),
                                           positive_work=(w.get("effort_snr") or {}).get("positive_work_fraction"))
                                      for w in win],
                             positive_work=fa.get("positive_work"),
                             estimates={k: dict(value=p["estimate"], ci95=p["ci95"], identifiable=p["identifiable"],
                                                reasons=p["reasons"][:2]) for k, p in (gi.get("parameters") or {}).items()},
                             derived=gi.get("derived_combinations"),
                             residual_rms_sigma=fit.get("residual_rms_sigma"), converged=fit.get("converged"),
                             adequacy=ad.get("status"), adequacy_failures=(ad.get("failures") or [])[:3],
                             baselines={k: [dict(I=r.get("inertia"), c=r.get("damping"), ident=r.get("identifiable"))
                                            for r in rs] for k, rs in (bl.get("results") or {}).items()},
                             categories={k: v2 for k, v2 in m["status_categories"].items() if k != "note"}))
    out = Path(a.out or Path(a.runs) / "g9_exploratory_summary")
    Path(str(out) + ".json").write_text(json.dumps(rows, indent=1, default=str))
    lines = ["| scene | variant terms | windows (SNR, +work) | estimates (CI95; identifiable) | derived | fit rms/sigma | adequacy |",
             "|---|---|---|---|---|---|---|"]
    f = lambda x: "n/a" if x is None else (f"{x:.3g}" if isinstance(x, (int, float)) else str(x))
    for r in rows:
        w = "; ".join(f"{x['rows']} SNR {f(x['snr'])} +W {f(x['positive_work'])}" for x in r["windows"])
        e = "; ".join(f"{k} {f(p['value'])} [{f(p['ci95'][0])}, {f(p['ci95'][1])}] {'Y' if p['identifiable'] else 'N'}"
                      for k, p in r["estimates"].items())
        d = "; ".join(f"{k} {f(p['estimate'])}±{f(p['std'])} {'Y' if p['identifiable'] else 'N'}"
                      for k, p in (r["derived"] or {}).items())
        lines.append(f"| {r['scene']} {r['version']} | {','.join(r['terms'])} | {w} | {e} | {d} | "
                     f"{f(r['residual_rms_sigma'])} | {r['adequacy']} |")
    Path(str(out) + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
