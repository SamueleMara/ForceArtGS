#!/usr/bin/env python3
"""Summarise G9 exploratory runs per scene x model variant: windows, pulse vs coast evidence, estimates with their
STATISTICAL CI and the DECLARED-SCENARIO envelope (separately), model adequacy, load-model comparison, and the two
verdict dimensions (statistically resolved in this model / physically interpretable & validated). Wording is
qualified: an estimate of an inadequate model or of an unaccepted joint is never called "identified".

    python tools/g9_exploratory_summary.py RUN_ROOT [--out PREFIX]
"""
import argparse
import json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).is_file() else None


def f(x):
    if x is None:
        return "n/a"
    if isinstance(x, (int, float)):
        return f"{x:.3g}"
    return str(x)


def verdict_text(p, adequacy, kin_accepted, terms):
    if p.get("physically_interpretable_validated"):
        return "PHYSICALLY INTERPRETABLE (validated)"
    q = []
    q.append("statistically resolved in this model" if p.get("parameter_statistically_resolved_in_this_model")
             else "not statistically resolved")
    if p.get("scenario_sensitive"):
        q.append("scenario-sensitive")
    q.append(f"{'+'.join(terms)} conditional estimate")
    if adequacy != "PASS":
        q.append(f"model adequacy {adequacy}")
    if not kin_accepted:
        q.append("working joint not certified")
    return "; ".join(q)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = []
    for run in sorted(Path(a.runs).glob("*_joint_aware_full")):
        for v in sorted((run / "dynamics" / "g9").glob("v[0-9][0-9][0-9]")):
            m = load(v / "g9_manifest.json")
            if not m or not m.get("label", "").startswith("EXPLORATORY_CONDITIONAL"):
                continue
            fa = load(v / "force_preprocessing_report.json") or {}
            fit = load(v / "estimator_results" / "general_rollout_fit.json") or {}
            gi = (load(v / "general_identifiability.json") or {}).get("fit_identifiability") or {}
            ad = ((load(v / "model_adequacy.json") or {}).get("adequacy") or {})
            pp = load(v / "physics_parameters.json") or {}
            ic = load(v / "impulse_coast_analysis.json") or {}
            lm = load(v / "load_model_comparison.json")
            bl = load(v / "estimator_results" / "frozen_viscous_baselines.json") or {}
            terms = [t for t, s in m["terms"].items() if s == "enabled"]
            kin_ok = not m["status_categories"]["KINEMATICS_NOT_ACCEPTED"]
            per = {}
            for t, e in (pp.get("terms") or {}).items():
                for n, p in (e.get("parameters") or {}).items():
                    g = (gi.get("parameters") or {}).get(n, {})
                    per[n] = dict(value=p.get("estimate"), ci95_statistical=g.get("ci95_statistical"),
                                  scenario_envelope=g.get("scenario_envelope"),
                                  worst_scenario_shift=g.get("scenario_worst_single_family"),
                                  resolved=p.get("parameter_statistically_resolved_in_this_model"),
                                  physically_interpretable_validated=p.get("physically_interpretable_validated"),
                                  conditionally_interpretable=p.get("conditionally_interpretable"),
                                  blockers=p.get("interpretation_blockers"),
                                  text=verdict_text(p, ad.get("status"), kin_ok, terms),
                                  label=e.get("label"))
            ics = [dict(window=w.get("window"), verdict=w.get("verdict"), **(w.get("summary") or {}))
                   for w in ic.get("windows", [])]
            rows.append(dict(scene=m["scene"], version=v.name, terms=terms, status=m["status"],
                             kinematics_accepted=kin_ok, force_sign=(fa.get("record_sources") and
                                                                     "s=-1 (declared; explicit metadata)"),
                             windows=fa.get("interaction_windows"), impulse_coast=ics, estimates=per,
                             derived=gi.get("derived_combinations"),
                             residual_rms_sigma=fit.get("residual_rms_sigma"), adequacy=ad.get("status"),
                             adequacy_failures=(ad.get("failures") or [])[:3],
                             load_models=None if not lm else dict(
                                 variants={k: dict(chi2_dof=x.get("chi2_dof"), AIC=x.get("AIC"),
                                                   load=(x.get("effective_configuration_load") or {}).get(
                                                       "variation_over_observed_q"))
                                           for k, x in lm.get("variants", {}).items()},
                                 tests=lm.get("tests"), alternatives=lm.get("alternatives_for_A"),
                                 basis_condition_D=lm.get("basis_condition_number_D")),
                             baselines={k: [dict(I=r.get("inertia"), c=r.get("damping"), ident=r.get("identifiable"))
                                            for r in rs] for k, rs in (bl.get("results") or {}).items()},
                             baseline_note="frozen viscous baselines fit the SAME simplified model: agreement is "
                                           "not independent physical validation",
                             categories={k: v2 for k, v2 in m["status_categories"].items() if k != "note"}))
    out = Path(a.out or Path(a.runs) / "g9_exploratory_summary")
    Path(str(out) + ".json").write_text(json.dumps(rows, indent=1, default=str))
    L = ["| scene / version | model | kinematics | pulse SNR / coast SNR / impulse | estimate [stat. CI95] {scenario envelope} -> verdict | adequacy |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        ic = "; ".join(f"{f(w.get('SNR_pulse'))} / {f(w.get('SNR_coast'))} / {w.get('verdict')}" for w in r["impulse_coast"])
        est = "<br>".join(f"{n} {f(p['value'])} [{f((p['ci95_statistical'] or [None])[0])}, "
                          f"{f((p['ci95_statistical'] or [None, None])[1])}] "
                          f"{{{f((p['scenario_envelope'] or [None])[0])}, {f((p['scenario_envelope'] or [None, None])[1])}}}"
                          f" -> {p['text']}" for n, p in r["estimates"].items())
        L.append(f"| {r['scene']} {r['version']} | {'+'.join(r['terms'])} | "
                 f"{'accepted' if r['kinematics_accepted'] else 'NOT accepted (working model)'} | {ic} | {est} | "
                 f"{r['adequacy']} |")
    Path(str(out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
