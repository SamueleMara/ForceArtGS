"""Aggregate the latest forcetwin_adaptation version of every run into one comparison (CSV / JSON / MD)."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def latest(run: Path):
    vs = sorted((run / "dynamics" / "forcetwin_adaptation").glob("v[0-9][0-9][0-9]"))
    return vs[-1] if vs else None


def compare(root, out) -> dict:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, scenes = [], {}
    for run in sorted(root.glob("*_joint_aware_full")):
        v = latest(run)
        if v is None:
            continue
        st = json.loads((v / "STATUS.json").read_text()) if (v / "STATUS.json").is_file() else {}
        scenes[run.name] = dict(version=str(v), status=st.get("status"), reason=st.get("reason"),
                                primary=st.get("primary_hypothesis"), impulse=st.get("impulse"),
                                states=st.get("states"))
        if (v / "comparison.csv").is_file():
            with open(v / "comparison.csv") as fh:
                for r in csv.DictReader(fh):
                    r["version"] = v.name
                    rows.append(r)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "comparison_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys or ["scene"])
        w.writeheader()
        w.writerows(rows)
    (out / "comparison_all.json").write_text(json.dumps(dict(scenes=scenes, n_rows=len(rows)), indent=1,
                                                        default=str))
    md = ["| scene | version | status | primary hypothesis | impulse decision |", "|---|---|---|---|---|"]
    for k, s in scenes.items():
        md.append(f"| {k.split('_multiview')[0]} | {Path(s['version']).name} | {s['status']} | {s['primary']} | "
                  f"{(s.get('impulse') or {}).get('decision')} |")
    md += ["", "| scene | hypothesis | variant | estimator | status | I_eff | I_eff_status | mu | b | replay rmse/sigma |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("variant") != "inertia_viscous_coulomb":
            continue
        md.append("| " + " | ".join(str(r.get(c) or "—")[:24] for c in (
            "scene", "hypothesis", "variant", "estimator", "status", "I_eff", "I_eff_status", "mu", "b",
            "open_loop_rmse_over_sigma")) + " |")
    (out / "comparison_all.md").write_text("\n".join(md) + "\n")
    return dict(out=str(out), scenes=list(scenes), n_rows=len(rows))
