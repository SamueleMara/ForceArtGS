#!/usr/bin/env python3
"""Collect NEXT_* joint-discovery runs into ``comparison.csv`` + ``summary.md`` (spec next-steps §11).

Post-hoc columns (reference axis/line distance, below-lid leakage) are computed AFTER the runs from
an optional reference JSON and a declared height split; they never feed back into fitting.

    python samuele/summarize_joint_next_steps.py --runs <run dirs> --output <dir> \
        [--reference-json hinge_reference_tracker_free.json] [--lid-height-split-m 0.145]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from estimotion_2part.joint_discovery.rigid_kinematics import axis_angle_deg, line_to_line_distance  # noqa: E402


def row(run: Path, ref=None, split=None) -> dict:
    s = json.loads((run / "summary.json").read_text())
    sel = json.loads((run / "model_selection.json").read_text())
    man = json.loads((run / "manifest.json").read_text())
    jc = json.loads((run / "joint_candidates.json").read_text())["candidates"]
    R, P = jc.get("revolute"), jc.get("prismatic")
    g = sel.get("grouped_statistics", {})
    rp = g.get("revolute_vs_prismatic", {})
    obs_r = (sel["scores"].get("revolute_observability") or {})
    out = dict(run=run.name, experiment_id=sel.get("experiment_id") or man.get("experiment_id"),
               cameras="+".join(man["cameras"]), depth=man["config"]["enable_depth"],
               depth_evidence=man["config"].get("depth_evidence", {}).get("mode", "raw"),
               depth_ar1_rho=(sel.get("noise_calibration", {}).get("calibrated") or {}).get("depth_ar1_rho"),
               uncertainty=man["config"]["comparison"].get("uncertainty_method", "track_paired"),
               policy=man["config"]["comparison"].get("remote_hinge_policy", "ratio"),
               decision=s["decision"], occam=sel["scores"].get("occam_preference"),
               reasons=" | ".join(s["reasons"]),
               z_RP_naive=rp.get("z_naive"), z_RP_cluster=rp.get("z_cluster"), n_groups=rp.get("n_groups"),
               bootstrap_p_R_better=rp.get("bootstrap_p_second_better"),
               z_R_static_cluster=g.get("revolute_vs_static", {}).get("z_cluster"),
               z_P_static_cluster=g.get("prismatic_vs_static", {}).get("z_cluster"),
               R_arc_deg=math.degrees(R["observability"]["q_span"]) if R else None,
               R_hinge_radius_m=(R["observability"].get("hinge_radius_m") or {}).get("median") if R else None,
               R_axis=json.dumps(np.round(R["joint_axis_world"], 3).tolist()) if R else None,
               R_axis_sigma_deg=R["observability"].get("axis_sigma_deg") if R else None,
               P_span_cm=100 * P["observability"]["q_span"] if P else None,
               P_axis=json.dumps(np.round(P["joint_axis_world"], 3).tolist()) if P else None,
               n_moving_R=R["n_moving_tracks"] if R else None, n_moving_P=P["n_moving_tracks"] if P else None,
               geometry_identifiable=obs_r.get("geometry_identifiable"),
               translation_distinguishability_sigma=(obs_r.get("translation_distinguishability") or {}).get("best_camera_p90_over_sigma"),
               hinge_to_support_m=(obs_r.get("hinge_support") or {}).get("hinge_to_support_min_m"),
               runtime_s=man.get("runtime_s"), git_head=man["git"].get("head", "")[:8],
               git_diff_sha=man["git"].get("diff_sha256", "")[:12], config_sha=man.get("config_hash"))
    ia = json.loads((run / "input_audit.json").read_text())
    for cid, v in ia["summary"]["per_camera"].items():
        out[f"valid_depth_frac_{cid}"] = round(v["obs_3d"] / max(v["obs_uv_valid"], 1), 3)
    proj = json.loads((run / "camera_projection" / "residuals.json").read_text())
    for k, per in proj.items():
        for cid, v in per.items():
            out[f"uv_px_moving_{k[0].upper()}_{cid}"] = v.get("median_uv_px")
    if ref is not None and R is not None:
        out["posthoc_R_axis_vs_ref_deg"] = axis_angle_deg(R["joint_axis_world"], ref["axis"])
        out["posthoc_R_line_vs_ref_cm"] = 100 * line_to_line_distance(R["joint_axis_world"], R["joint_origin_world"],
                                                                     ref["axis"], ref["point"])
    if split is not None:
        sup = np.load(run / "part_support" / "track_support.npz")
        for key in ("R", "P"):
            if f"p_moving_{key}" not in sup.files:
                continue
            mv = np.nan_to_num(sup[f"p_moving_{key}"]) > 0.5
            X = sup[f"anchor_w_{key}"][mv]
            X = X[np.isfinite(X).all(1)]
            out[f"posthoc_below_lid_frac_{key}"] = float(np.mean(X[:, 2] < split)) if len(X) else None
    pp = sel.get("physical_selection")
    out["physical_status"] = None if pp is None else pp.get("status")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--reference-json", type=Path, default=None)
    ap.add_argument("--lid-height-split-m", type=float, default=None)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    ref = None
    if a.reference_json:
        r = json.loads(a.reference_json.read_text())
        ref = dict(axis=r["hinge_direction"], point=r["hinge_point_nominal"])
    rows = [row(r, ref, a.lid_height_split_m) for r in a.runs if (r / "summary.json").is_file()]
    a.output.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999)
    with open(a.output / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    f = lambda v, d=1: "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    L = ["| run | cams | depth | decision | z R-P naive / cluster (groups) | R arc ° | R hinge r m | P cm | moving R/P | "
         "identifiable | R axis vs ref ° (post hoc) | below-lid R (post hoc) |", "|" + "---|" * 12]
    for r in rows:
        dep = "off" if not r["depth"] else (r["depth_evidence"] + ("+ar1" if r["depth_ar1_rho"] else ""))
        L.append(f"| {r['run']} | {r['cameras']} | {dep} | {r['decision']}{' (occam '+r['occam']+')' if r['occam'] else ''} | "
                 f"{f(r['z_RP_naive'])} / {f(r['z_RP_cluster'])} ({r['n_groups']}) | {f(r['R_arc_deg'])} | "
                 f"{f(r['R_hinge_radius_m'], 2)} | {f(r['P_span_cm'])} | {r['n_moving_R']}/{r['n_moving_P']} | "
                 f"{r['geometry_identifiable']} | {f(r.get('posthoc_R_axis_vs_ref_deg'))} | "
                 f"{f(r.get('posthoc_below_lid_frac_R'), 2)} |")
    (a.output / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
