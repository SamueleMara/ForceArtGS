"""Side-by-side summary of the G9 four-estimator comparison (estimator_results/method_comparison.json) for the
versions logged by pipelines/g9_real_general_methods.sh. Read-only; prints Markdown.
    python tools/g9_method_comparison_summary.py RUN_ROOT [--log g9_general_methods_batch.log]
"""
import argparse
import json
import re
from pathlib import Path


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        return f"{x:.2e}"
    return f"{x:.{nd}g}"


def version_dirs(root: Path, log: Path, all_batches=False):
    rows = []
    lines = log.read_text().splitlines()
    if not all_batches:                                   # only the most recent batch (after its last "start" line)
        starts = [i for i, l in enumerate(lines) if " start " in l]
        lines = lines[starts[-1]:] if starts else lines
        if not any(l.rstrip().endswith("done") for l in lines):
            print(f"NOTE: the latest batch in {log.name} has not finished yet (no 'done' line)\n")
    for line in lines:
        m = re.match(r"\S+ \S+ (\S+) (\S+) exit (\d+) .*?(v\d{3})\W*$", line)
        if not m:
            continue
        scene, variant, _, ver = m.groups()
        for run in root.glob(f"{scene}_multiview_*_joint_aware_full"):
            d = run / "dynamics" / "g9" / ver
            if d.is_dir():
                rows.append((scene, variant, d))
    return rows


def _j(d, n):
    p = d / f"{n}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def audit(d):
    """Gravity-projection / geometry / reference-pose audit highlights (NOT_RUN when absent)."""
    rp, ga, mc = _j(d, "reference_pose_alignment"), _j(d, "gravity_geometry_audit"), _j(d, "mass_com_inertia_audit")
    if not ga:
        print("\n(gravity / geometry audit not in this version: produced by older code)")
        return
    j = ga.get("joint") or {}
    print(f"\naudit: reference pose {rp.get('status')} (q_ref {fmt(rp.get('q_ref'))}, {rp.get('q_ref_source', '')}); "
          f"tilt {fmt(j.get('tilt_from_vertical_deg'))}°, |g×a| {fmt(j.get('g_cross_a_norm'))}; gravity "
          f"{(ga.get('gravity') or {}).get('provenance_class')}; flags {ga.get('flags')}; parity {ga.get('parity_status')}")
    for mode, x in (mc.get("modes") or {}).items():
        print(f"  {mode}: m1 {fmt(x['m1'])} {x['m1_units']}, |r⊥| {fmt(x.get('r_perp_norm_m'))} m, I_axis,1 "
              f"{fmt(x.get('I_axis1', x.get('M1')))}, G1 amp {fmt(x.get('G1_amplitude', x.get('G1')))}, "
              f"G/I {fmt(x.get('gravity_to_inertia_ratio'))}")
    for m, e in (mc.get("implied_by_method") or {}).items():
        print(f"  implied by {m}: ρ {fmt(e['density'])} {e['density_units']} → m {fmt(e['mass_kg'])} kg, "
              f"I_axis {fmt(e.get('I_axis_kg_m2'))}, max|G| {fmt(e.get('gravity_amplitude_Nm', e.get('gravity_load_N')))}")
    gs = _j(d, "gravity_joint_sensitivity")
    if gs.get("scenarios"):
        by = {x["name"]: x for x in gs["scenarios"]}
        for name in ("accepted", "exactly_vertical_hinge", "mass_mode_uniform_shell", "mass_mode_uniform_solid"):
            x = by.get(name)
            if not x:
                continue
            reps = {m[8:12]: fmt(v.get("replay_rmse_over_sigma_reprojected_effort")) for m, v in x["per_method"].items()}
            print(f"  scenario {name}: max|G| per method {[fmt(v.get('max_abs_G')) for v in x['per_method'].values()]}"
                  f", replay rmse/σ {reps}")
        for m, v in (gs.get("geometry_variability") or {}).items():
            print(f"  geometry variability {m}: max|G| {fmt(v['max_abs_G_accepted'])} in "
                  f"[{fmt(v['max_abs_G_scenario_range'][0])}, {fmt(v['max_abs_G_scenario_range'][1])}]")
    ib = _j(d, "impulse_balance_audit")
    for p in ib.get("pulses", []):
        ti = p.get("tool_inertia_hypothesis") or {}
        print(f"  impulse balance rows {p['rows']}: {p['verdict']} -- {p.get('reason')}; gauge "
              f"{(ib.get('q_axis_gauge_check') or {}).get('status')}; tool mass to cancel J "
              f"{fmt(ti.get('tool_mass_that_would_cancel_J_kg'))} kg; lag scan flips sign {p.get('lag_scan_any_sign_flip')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--log", default="g9_general_methods_batch.log")
    ap.add_argument("--all-batches", action="store_true", help="also list versions from earlier batches")
    a = ap.parse_args()
    root = Path(a.root)
    for scene, variant, d in version_dirs(root, root / a.log, a.all_batches):
        mcp = d / "estimator_results" / "method_comparison.json"
        print(f"\n### {scene} / {variant} ({d.name})\n")
        if not mcp.is_file():
            man = json.loads((d / "g9_manifest.json").read_text()) if (d / "g9_manifest.json").is_file() else {}
            print(f"no method comparison; status {man.get('status')}; blocked {man.get('blocked')}")
            continue
        mc = json.loads(mcp.read_text())
        mm = mc["model"].get("mass_model") or {}
        print(f"source `{mc['model']['parameter_source']}`, free {mc['model']['free_parameters']}, mass mode "
              f"{mm.get('mode')}, M1 {fmt(mm.get('M1'))}, m1 {fmt(mm.get('m1'))}\n")
        names = mc["model"]["free_parameters"]
        print("| method | " + " | ".join(names) + " | M | gravity amp. | replay rmse/σ | NIS | runtime s |")
        print("|---" * (len(names) + 6) + "|")
        for m, r in mc["methods"].items():
            if "parameters" not in r:
                print(f"| {m} | " + " | ".join(["—"] * len(names)) + f" | — | — | — | — | {r.get('status')} |")
                continue
            cells = []
            for n in names:
                p = r["parameters"][n]
                cells.append(f"{fmt(p['value'])} ± {fmt(p['std'])}" if p.get("std") is not None else fmt(p["value"]))
            ph = r.get("physical") or {}
            g = ph.get("gravity") or {}
            rep = r.get("open_loop_replay") or []
            rr = [x.get("rmse_over_sigma") for x in rep if x.get("rmse_over_sigma") is not None]
            rr = rr or ([r["training_residual_rms_sigma"]] if r.get("training_residual_rms_sigma") else [])
            nis = (r.get("nis") or {}).get("mean")
            print(f"| {m} | " + " | ".join(cells) + f" | {fmt(ph.get('M'))} | {fmt(g.get('amplitude'))} | "
                  f"{', '.join(fmt(x) for x in rr) or '—'} | {fmt(nis)} | {fmt(r.get('runtime_s'), 3)} |")
        audit(d)
        ic = d / "impulse_coast_analysis.json"
        if ic.is_file():
            for w in json.loads(ic.read_text()).get("windows", []):
                for dp in w.get("declared_pulses", []):
                    p, v, mm_ = dp.get("pulse") or {}, dp.get("velocity") or {}, dp.get("J_over_dqdot_CONDITIONAL_M")
                    print(f"\ndeclared pulse rows {dp['rows']}: J_Q {fmt(p.get('J_Q'))} ± {fmt(p.get('sigma_J_Q'))} "
                          f"({fmt(p.get('J_over_sigma'))}σ), SNR {fmt(p.get('SNR_pulse_peak'))}, "
                          f"q̇ {fmt(v.get('v_pre'))} → {fmt(v.get('v_post'))} (Δ {fmt(v.get('dv'))} ± "
                          f"{fmt(v.get('dv_sd'))}), oscillatory {p.get('oscillatory_transient')}" +
                          (f", J/Δq̇ {fmt(mm_['estimate'])} ± {fmt(mm_['sd'])}, sign contradiction "
                           f"{mm_['sign_contradiction']}" if mm_ else ""))
                    for k, t in (dp.get("axis_tilt_sensitivity") or {}).items():
                        print(f"  tilt {k} ({fmt(t['tilt_deg'])}°): J_Q rel change {fmt(t['rel_change'])}")
                s = w.get("summary") or {}
                print(f"\nwindow {w.get('window')}: verdict {w.get('verdict')}, SNR_coast {fmt(s.get('SNR_coast'))}, "
                      f"coast decay {fmt((s.get('coast_decay_time') or {}).get('estimate'))} s")


if __name__ == "__main__":
    main()
