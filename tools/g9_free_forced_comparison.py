"""Aggregate the LATEST g9_free_forced/vNNN of every scene into one table (csv / json / md).

    python tools/g9_free_forced_comparison.py <RUNS_OR_OUT_BASE> <OUT_DIR>
Finds */dynamics/g9_free_forced/v*/STATUS.json (run layout) or */v*/STATUS.json (OUT_BASE layout)."""
import csv
import json
import sys
from pathlib import Path


def latest(root: Path):
    found = {}
    for st in list(root.glob("*/dynamics/g9_free_forced/v[0-9][0-9][0-9]/STATUS.json")) + \
            list(root.glob("*/v[0-9][0-9][0-9]/STATUS.json")):
        s = json.loads(st.read_text())
        sc = s.get("scene") or st.parents[1].name
        if sc not in found or st.parent.name > found[sc].parent.name:
            found[sc] = st
    return found


def main(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, scenes = [], []
    for sc, st in sorted(latest(root).items()):
        s = json.loads(st.read_text())
        fs = s.get("factorized_status") or {}
        scenes.append(dict(scene=sc, version=st.parent.name, status=s.get("status"), reason=s.get("reason"),
                           analysis_hypothesis=s.get("analysis_hypothesis"),
                           primary_normalized=(s.get("primary_normalized_status") or {}).get("status"),
                           normalized=(fs.get("normalized_dynamics") or {}).get("status"),
                           caps=" ".join((fs.get("normalized_dynamics") or {}).get("caps") or []),
                           absolute_scale=fs.get("absolute_scale"), simulator=fs.get("simulator_readiness"),
                           free_basis=s.get("free_basis"), impulse=s.get("impulse")))
        cr = st.parent / "comparison_rows.csv"
        if cr.is_file():
            with open(cr) as fh:
                for r in csv.DictReader(fh):
                    rows.append(dict(r, version=st.parent.name))
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "free_forced_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys or ["scene"])
        w.writeheader()
        w.writerows(rows)
    (out / "free_forced_all.json").write_text(json.dumps(dict(scenes=scenes, n_rows=len(rows)), indent=1))
    f = lambda x: "—" if x in (None, "") else (f"{float(x):.3g}" if _num(x) else str(x))
    md = ["| scene | version | status | analysis hypothesis | primary | normalized | caps | absolute scale | "
          "simulator | impulse |", "|" + "---|" * 10]
    for s in scenes:
        md.append("| " + " | ".join(f(s.get(k)) for k in ("scene", "version", "status", "analysis_hypothesis",
                                                           "primary_normalized", "normalized", "caps",
                                                           "absolute_scale", "simulator", "impulse")) + " |")
    md += ["", "| scene | geometry | gravity | method | lam | lam status | gam | gam status | replay RMSE | "
               "RMSE/range | beats null |", "|" + "---|" * 11]
    for r in rows:
        md.append("| " + " | ".join(f(r.get(k)) for k in ("scene", "geometry", "gravity", "method", "lam",
                                                           "lam_status", "gam", "gam_status", "replay_rmse",
                                                           "replay_rmse_over_range", "replay_beats_null")) + " |")
    (out / "free_forced_all.md").write_text("\n".join(md) + "\n")
    print(json.dumps(dict(out=str(out), scenes=[s["scene"] for s in scenes], n_rows=len(rows)), indent=1))


def _num(x):
    try:
        float(x)
        return not isinstance(x, bool) and str(x) not in ("True", "False")
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
