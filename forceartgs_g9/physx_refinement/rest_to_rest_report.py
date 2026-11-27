"""Summary table (CSV + Markdown) and figures of the rest-to-rest comparison. Numbers only from the run's own outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

FIELDS = ("id", "method", "kind", "input_policy", "window", "model", "M", "c", "rho_scale", "mapping",
          "physx_rmse", "physx_rmse_over_sigma", "q_sim_T_minus_qT", "v_sim_T", "abs_v_sim_T_over_sigma",
          "time_to_settle_s", "objective", "n_engine_rollouts", "runtime_s", "bound_status", "own_metric",
          "episode_sha256", "full_wrench_record_sha256", "config_sha256", "engine", "seeds")


def _g(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def summary_rows(ep, analytics, pinn, px, info, hashes, cfg, legacy) -> list:
    common = dict(episode_sha256=ep.sha256(), full_wrench_record_sha256=hashes.get("full_wrench_record_sha256"),
                  config_sha256=_g(hashes, "files", "r2r_config", "sha256"),
                  engine=_g(px, "engine", "pipeline"), model="inertia + viscous (gravity 0, Coulomb off, armature 0)",
                  window=f"[{ep.t0:.4g}, {ep.T:.4g}] s", input_policy="C1_FULL_UNMASKED")
    rows = []
    own = {"B1_ENDPOINT_R2R": f"endpoint status {analytics['endpoint'].get('status')}",
           "B2_INTEGRAL_SVD_R2R": f"whitened cond {_g(analytics, 'svd', 'whitened', 'svd', 'condition_number')}",
           "B3_ANALYTIC_CONVOLUTION_1D": f"analytic rmse/sigma {analytics['conv'].get('rmse_over_sigma')}"}
    for v, r in (pinn.get("variants") or {}).items():
        own_ = _g(r, "open_loop", "rmse_over_sigma")
        own[[k for k in px["frozen"] if k.endswith(v)][0] if any(k.endswith(v) for k in px["frozen"]) else v] = \
            f"exact open-loop rmse/sigma {own_}; neural q rmse {r.get('neural_q_rmse')}"
    for lab, row in px["frozen"].items():
        mp = row["mapping"]
        m = row.get("metrics") or {}
        rows.append(dict(common, id=lab.split("_")[0], method=lab, kind="frozen theta -> PhysX replay",
                         M=mp.get("M"), c=mp.get("c"), rho_scale=_g(mp, "params", "rho_scale"),
                         mapping=mp.get("status") + (f" clip {mp.get('clip_log')}" if mp.get("clip_log") else ""),
                         physx_rmse=m.get("rmse"), physx_rmse_over_sigma=m.get("rmse_over_sigma"),
                         q_sim_T_minus_qT=m.get("q_sim_T_minus_qT"), v_sim_T=m.get("v_sim_T"),
                         abs_v_sim_T_over_sigma=m.get("abs_v_sim_T_over_sigma"),
                         time_to_settle_s=m.get("time_to_settle_s"), objective=row.get("objective_V0"),
                         n_engine_rollouts=1 if m else 0, runtime_s=None, bound_status=None,
                         own_metric=own.get(lab), seeds=None))
    for lab, f in px["fits"].items():
        m = f.get("metrics") or {}
        rows.append(dict(common, id=lab.split("_")[0], method=lab, kind="G10 PhysX fit (DE + local refine)",
                         M=f.get("M"), c=f.get("c"), rho_scale=_g(f, "params", "rho_scale"), mapping=None,
                         physx_rmse=m.get("rmse"), physx_rmse_over_sigma=m.get("rmse_over_sigma"),
                         q_sim_T_minus_qT=m.get("q_sim_T_minus_qT"), v_sim_T=m.get("v_sim_T"),
                         abs_v_sim_T_over_sigma=m.get("abs_v_sim_T_over_sigma"),
                         time_to_settle_s=m.get("time_to_settle_s"), objective=f.get("objective"),
                         n_engine_rollouts=f.get("n_engine_rollouts"), runtime_s=f.get("runtime_s"),
                         bound_status=json.dumps(f.get("at_bound")), own_metric=f.get("objective_kind"),
                         seeds=f"DE seed {f.get('de_seed')}"))
    c0 = px.get("C0") or {}
    if c0.get("status") == "FITTED":
        m = c0["metrics_on_C1_episode"]
        rows.append(dict(common, id="C0", method="C0_HISTORICAL_MASKED_INPUT_G10", kind="G10 fit on the historical "
                         "masked input / window; theta replayed on the C1 episode",
                         input_policy="C0_CONTACT_MASKED (fit) -> C1 replay",
                         window=f"fit [{c0['window']['t0']:.4g}, {c0['window']['t_end']:.4g}] s", M=c0["M"], c=c0["c"],
                         rho_scale=c0["params"]["rho_scale"], mapping=None, physx_rmse=m["rmse"],
                         physx_rmse_over_sigma=m["rmse_over_sigma"], q_sim_T_minus_qT=m["q_sim_T_minus_qT"],
                         v_sim_T=m["v_sim_T"], abs_v_sim_T_over_sigma=m["abs_v_sim_T_over_sigma"],
                         time_to_settle_s=m["time_to_settle_s"], objective=None,
                         n_engine_rollouts=c0["n_engine_rollouts"], runtime_s=c0["runtime_s"], bound_status=None,
                         own_metric=f"own-window rmse/sigma {c0['metrics_on_own_window']['rmse_over_sigma']:.4g}",
                         seeds=None))
    return rows


def assert_controlled(rows) -> list:
    """The controlled table may only hold rows on the SAME input arrays / window / model / engine (C1). Raises when
    rows with different episode hashes, models or engines are offered as one controlled comparison; C0 rows must be
    labelled as such (they are the archival input-policy control, not a controlled row)."""
    c1 = [r for r in rows if r.get("input_policy") == "C1_FULL_UNMASKED"]
    for key in ("episode_sha256", "model", "engine", "window"):
        vals = {r.get(key) for r in c1}
        if len(vals) > 1:
            raise ValueError(f"controlled comparison mixes different {key}: {sorted(map(str, vals))}")
    bad = [r["method"] for r in rows if r.get("input_policy") not in ("C1_FULL_UNMASKED",) and
           not str(r.get("input_policy", "")).startswith("C0_")]
    if bad:
        raise ValueError(f"rows without a declared input policy: {bad}")
    return c1


def _fmt(x, p=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{p}g}"
    return str(x)


def write_summary(out, rows, ep, ev, analytics, pinn, px, sigma_vT, hist, status) -> None:
    out = Path(out)
    assert_controlled(rows)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    L = [f"# Rest-to-rest comparison: {ep.scene}", "",
         f"git {status['git']}; quick {status['quick']}; episode sha256 `{ep.sha256()[:16]}`", "",
         "## Rest-to-rest window (visual plateaus only)", "",
         f"- t0 = {ep.t0:.4g} s, T = {ep.T:.4g} s, q0 = {ep.q0:.6g} +- {ep.q0_sigma:.2g}, qT = {ep.qT:.6g} +- "
         f"{ep.qT_sigma:.2g}, delta q = {ep.delta_q:.4g} (snr {_g(ev, 'motion', 'delta_q_snr'):.3g})",
         f"- {len(ep.t_q_s)} visual samples, {len(ep.force_t_s)} native wrench rows (C1: unmasked, no second tare); "
         f"effort components: {(ep.provenance or {}).get('effort_components', 'full_wrench')}",
         f"- terminal plateau: {_g(ev, 'terminal', 'confirmation_samples_after_T')} confirmation samples, "
         f"{_fmt(_g(ev, 'terminal', 'plateau_duration_s'))} s; sigma_vT = {sigma_vT['sigma_vT']:.3g} ({sigma_vT['source']})",
         f"- stationary load: initial {_g(ev, 'stationary_load', 'initial', 'status')} (mean Q "
         f"{_fmt(_g(ev, 'stationary_load', 'initial', 'mean_Q'))}), terminal "
         f"{_g(ev, 'stationary_load', 'terminal', 'status')} (mean Q {_fmt(_g(ev, 'stationary_load', 'terminal', 'mean_Q'))})",
         "", "## Main table (same episode, same model, same engine; PhysX replay or fit)", "",
         "| id | method | M | c | rho | PhysX rmse | rmse/sigma | q_sim(T)-qT | v_sim(T) | settle [s] | rollouts | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["physx_rmse_over_sigma"] is None, r["physx_rmse_over_sigma"] or 0)):
        L.append(f"| {r['id']} | {r['method']} | {_fmt(r['M'])} | {_fmt(r['c'])} | {_fmt(r['rho_scale'])} | "
                 f"{_fmt(r['physx_rmse'])} | {_fmt(r['physx_rmse_over_sigma'])} | {_fmt(r['q_sim_T_minus_qT'])} | "
                 f"{_fmt(r['v_sim_T'])} | {_fmt(r['time_to_settle_s'])} | {_fmt(r['n_engine_rollouts'])} | "
                 f"{r.get('mapping') or r.get('own_metric') or ''} |")
    f = px["fits"]

    def pick(prefix):
        c = [(k, v) for k, v in f.items() if k.startswith(prefix) and v.get("metrics")]
        return min(c, key=lambda kv: kv[1]["metrics"]["rmse_over_sigma"]) if c else (None, None)
    L += ["", "## Input-policy ablation (mandatory; same two-term model and DE budget)", "",
          "| row | input / window | rmse/sigma on the C1 episode | v_sim(T) | M | c |", "|---|---|---|---|---|---|"]
    c0 = px.get("C0") or {}
    if c0.get("status") == "FITTED":
        L.append(f"| C0 + original G10 | historical masked input, own window -> replayed on C1 | "
                 f"{_fmt(c0['metrics_on_C1_episode']['rmse_over_sigma'])} | {_fmt(c0['metrics_on_C1_episode']['v_sim_T'])} | "
                 f"{_fmt(c0['M'])} | {_fmt(c0['c'])} |")
    else:
        L.append(f"| C0 + original G10 | {c0.get('status')}: {c0.get('reason', '')} | - | - | - | - |")
    for title, pre in (("C1 + original G10 (matched window)", "B0"), ("C1 + new warm start G10", "B7"),
                       ("C1 + new warm start G10", "B8"), ("C1 + new warm start G10", "B9"),
                       ("C1 + union of new warm starts", "B789"), ("C1 + terminal-loss G10", "B10")):
        k, v = pick(pre)
        if v:
            L.append(f"| {title} | {k} | {_fmt(v['metrics']['rmse_over_sigma'])} | {_fmt(v['metrics']['v_sim_T'])} | "
                     f"{_fmt(v['M'])} | {_fmt(v['c'])} |")
    L += ["", "## Equal-budget G10 (same DE generations / population, same local refinement)", "",
          "| fit | starts | final objective | rmse/sigma | engine rollouts | runtime [s] | DE nfev |",
          "|---|---|---|---|---|---|---|"]
    for k, v in f.items():
        if v.get("status") == "FITTED":
            L.append(f"| {k} | {', '.join(v['starts'])} | {_fmt(v['objective'])} | "
                     f"{_fmt(v['metrics']['rmse_over_sigma'])} | {v['n_engine_rollouts']} | {_fmt(v['runtime_s'], 3)} | "
                     f"{_g(v, 'optimizer', 'nfev')} |")
        else:
            L.append(f"| {k} | - | {v.get('status')}: {v.get('error')} | - | - | - | - |")
    e, s, cv = analytics["endpoint"], analytics["svd"], analytics["conv"]
    L += ["", "## Analytics / diagnostics (same data; not independent validation)", "",
          f"- ENDPOINT_R2R: {e.get('status')}; J_Q {_fmt(e.get('JQ'))} +- {_fmt(e.get('JQ_sigma'))}; c_end "
          f"{_fmt(e.get('c_end'))}, M_end {_fmt(e.get('M_end'))}; {e.get('skipped_reason') or ''} {e.get('flags') or ''}",
          f"- INTEGRAL_SVD_R2R (whitened): M {_fmt(_g(s, 'whitened', 'M'))} +- {_fmt(_g(s, 'whitened', 'sd', 'M'))}, c "
          f"{_fmt(_g(s, 'whitened', 'c'))} +- {_fmt(_g(s, 'whitened', 'sd', 'c'))}, corr {_fmt(_g(s, 'whitened', 'correlation_M_c'))}, "
          f"scaled singular values {[round(x, 4) for x in (_g(s, 'whitened', 'svd', 'singular_values_column_scaled') or [])]}, "
          f"chi2/dof {_fmt(_g(s, 'whitened', 'chi2_dof'))}; unconstrained {_g(s, 'whitened', 'unconstrained')}; unique "
          f"{s.get('parameters_unique')}; output-error cross-check {_g(s, 'output_error_crosscheck', 'status')} "
          f"(LS replay {_fmt(_g(s, 'output_error_crosscheck', 'ls_replay_rmse_over_sigma'))} vs optimum "
          f"{_fmt(_g(s, 'output_error_crosscheck', 'output_error_optimum_rmse_over_sigma'))} sigma); flags {s.get('flags')}",
          f"- ANALYTIC_CONVOLUTION_1D: M {_fmt(cv.get('M'))}, c {_fmt(cv.get('c'))}, lambda {_fmt(cv.get('lam'))} 1/s, "
          f"rmse/sigma {_fmt(cv.get('rmse_over_sigma'))}, lambda valley {_g(cv, 'valley', 'lam_range')} (flat "
          f"{_g(cv, 'valley', 'flat_lambda_valley')}), analytic v(T) {_fmt(cv.get('v_T'))}",
          f"- work (model-free): {_g(analytics, 'diagnostics', 'work_model_free', 'verdict')}, W_in "
          f"{_fmt(_g(analytics, 'diagnostics', 'work_model_free', 'W_in'))}; damping energy c_energy "
          f"{_fmt(_g(analytics, 'diagnostics', 'energy', 'c_energy'))} "
          f"({_g(analytics, 'diagnostics', 'energy', 'verdict')})",
          f"- bias ablation (diagnostic only): b0 {_fmt(analytics['bias']['b0'])} (z {_fmt(analytics['bias']['b0_z'])}), "
          f"c {_fmt(analytics['bias']['c'])}", "", "## PINN ablations (open-loop replay of the frozen theta, v0 = 0)", ""]
    for v, r in (pinn.get("variants") or {}).items():
        per = r.get("per_seed") or []
        L.append(f"- {v}: {r.get('status')} M {_fmt(r.get('M'))} c {_fmt(r.get('c'))} open-loop rmse/sigma "
                 f"{_fmt(_g(r, 'open_loop', 'rmse_over_sigma'))}; per seed "
                 f"{[(p['seed'], _fmt(p['M']), _fmt(p['c']), _fmt(p['open_loop']['rmse_over_sigma'])) for p in per]}"
                 + (f"; {r.get('reason')}" if r.get("reason") else ""))
    L += ["", "## Labels", "", "- FIT_TO_DEMONSTRATION: every row is a fit / replay of ONE recorded demonstration",
          "- PARAMETERS_UNIQUE: see the SVD uniqueness flag, the convolution valley and candidate_family.json",
          "- INDEPENDENT_REAL_ACTION_VALIDATED: false (no unseen action was replayed)",
          f"- historical outputs listed in historical_baseline_inventory.json ({hist.get('n', 0)} items, "
          "HISTORICAL_DIFFERENT_INPUT_OR_BUDGET: never in the controlled table)"]
    (out / "summary.md").write_text("\n".join(L) + "\n")


# ============================================================================================ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_rest_plateaus(d, fr, ev):
    plt = _plt()
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    t, q = np.asarray(fr.q_t_s), np.asarray(fr.q)
    ax.plot(t, q, ".-", ms=3, label="q_visual (joint authority)")
    for key, col in (("initial", "tab:green"), ("terminal", "tab:red")):
        w = (ev.get(key) or {}).get("window")
        if w:
            ax.axvspan(t[w[0]], t[w[1]], color=col, alpha=0.2, label=f"{key} plateau")
    ini, ter = ev.get("initial") or {}, ev.get("terminal") or {}
    if ini.get("t0") is not None:
        ax.axvline(ini["t0"], color="g", ls="--", label="t0")
    if ter.get("T") is not None:
        ax.axvline(ter["T"], color="r", ls="--", label="T")
    ax.set_xlabel("t [s] (source frame / 50)")
    ax.set_ylabel("q")
    ax.set_title(f"rest plateaus: {ev.get('status')}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "rest_plateaus.png", dpi=120)
    plt.close(fig)


def plots(d, ep, fr, analytics, px, segs):
    plt = _plt()
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    t = ep.t_q_s
    tr = px["trajectories"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(t, ep.q_obs, yerr=ep.q_sigma, fmt="k.", ms=3, label="q_obs")
    for lab, r in tr.items():
        ax.plot(t, r["q"], lw=1, label=lab)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("q")
    ax.set_title("PhysX q(t) per method (same episode, same input)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(d / "trajectory_compare.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for lab, r in tr.items():
        ax.plot(t, (np.asarray(r["q"]) - ep.q_obs) / ep.q_sigma, lw=1, label=lab)
    ax.axhline(3, color="k", ls=":")
    ax.axhline(-3, color="k", ls=":")
    ax.set_ylabel("(q_PX - q_obs) / sigma_q")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(d / "residuals.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for lab, r in tr.items():
        if r.get("qdot") is not None:
            ax.plot(t, r["qdot"], lw=1, label=lab)
    ax.axvline(ep.T, color="r", ls="--", label="T")
    sv = px["sigma_vT"]["sigma_vT"]
    ax.axhspan(-3 * sv, 3 * sv, color="grey", alpha=0.2, label="+-3 sigma_vT")
    ax.set_xlim(ep.t0 + 0.6 * (ep.T - ep.t0), ep.T + 0.02 * (ep.T - ep.t0))
    ax.set_ylabel("qdot_PX")
    ax.set_title("simulated speed around the terminal rest")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(d / "terminal_velocity.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(fr.t_s, fr.Q_full, where="post", lw=0.8, label="Q_full (C1, unmasked, every native row)")
    if px.get("C0_masked_input"):
        ax.step(px["C0_masked_input"]["t"], px["C0_masked_input"]["Q_masked"], where="post", lw=0.8,
                label="historical C0: zero outside the contact segments")
    ax.axvline(ep.t0, color="g", ls="--")
    ax.axvline(ep.T, color="r", ls="--")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Q")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "force_input.png", dpi=120)
    plt.close(fig)
    s = analytics["svd"]
    fig, ax = plt.subplots(figsize=(5, 4))
    for var in ("unweighted", "whitened"):
        sv_ = (s.get(var) or {}).get("svd", {}).get("singular_values_column_scaled")
        if sv_:
            ax.semilogy(range(1, len(sv_) + 1), sv_, "o-", label=var)
    ax.set_xlabel("index")
    ax.set_ylabel("singular value (column-scaled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(d / "svd_spectrum.png", dpi=120)
    plt.close(fig)
    from ..estimators.rest_to_rest_convolution import replay
    cv = analytics["conv"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    prof = cv.get("profile") or []
    if prof:
        axs[0].semilogx([max(p["lam"], 1e-6) for p in prof], [p["rmse_over_sigma"] for p in prof], ".-")
        axs[0].set_xlabel("lambda = c / M [1/s]")
        axs[0].set_ylabel("analytic rmse/sigma (beta re-fitted)")
        axs[0].set_title("convolution lambda profile")
    if cv.get("status") == "OK":
        Ms = np.geomspace(cv["M"] / 4, cv["M"] * 4, 25)
        cs = np.r_[0.0, np.geomspace(max(cv["c"], 1e-6) / 4, max(cv["c"], 1e-6) * 4, 24)]
        sig = np.hypot(ep.q_sigma, 3e-4)
        Z = np.array([[np.sqrt(np.mean(((replay(ep, M, c)["q"] - ep.q_obs) / sig) ** 2)) for M in Ms] for c in cs])
        cs_ = axs[1].contourf(Ms, cs, np.log10(Z), 20)
        fig.colorbar(cs_, ax=axs[1], label="log10 rmse/sigma (exact two-term replay)")
        axs[1].set_xscale("log")
        axs[1].plot(cv["M"], cv["c"], "r*")
        axs[1].set_xlabel("M")
        axs[1].set_ylabel("c")
        axs[1].set_title("M / c loss valley")
    fig.tight_layout()
    fig.savefig(d / "param_valley.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, trc in px["traces"].items():
        if trc:
            a = np.asarray(trc)
            ax.semilogy(a[:, 0], a[:, 1], lw=1, label=lab)
    ax.set_xlabel("engine rollouts")
    ax.set_ylabel("best objective so far")
    ax.set_title("equal-budget G10 convergence")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(d / "optimizer_convergence.png", dpi=120)
    plt.close(fig)
