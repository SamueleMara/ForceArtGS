"""Effective configuration load vs constant bias vs alternatives (G9 corrections section 3).

In ``joint_space_only`` mode the revolute term gA cos q + gB sin q is an EFFECTIVE CONFIGURATION LOAD (it may be
gravity on a tilted hinge, a door closer, a cable, a residual force or a q trend): it is never physical gravity.
Only ``geometry_coupled`` with a calibrated gravity vector and a validated mass model is gravitational torque.

Matched candidates on the SAME interactions / windows (all other toggles equal):
  A  base terms (no load, no bias)          B  A + constant bias b
  C  A + effective load basis               D  A + bias + basis (only if [1, cos q, sin q] is resolvable)
Reported per variant: chi2, chi2/dof, AIC, BIC (training-only scores: there are no independent episodes),
autocorrelation-corrected likelihood-ratio tests, the load AMPLITUDE / variation over the observed q interval with
its delta-method CI, the 2-D (gA, gB) confidence ellipse, the basis condition number. Alternatives to "a load is
needed": a force/q time offset (grid of signed lags on A), a residual constant force (= B), a linear q trend
(post-hoc absorption of the A residual). No variant is selected automatically.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2 as chi2dist

from ..dynamics.general_1dof import GeneralTerms
from ..estimators.general_constrained import general_rollout_fit
from ..estimators.general_integral_ls import general_integral_ls, theta0_from_ls
from ..estimators.model_spec import ModelSpec


class _Eff:
    def __init__(self, t, lev):
        self.t_s, self.levels = np.asarray(t, float), np.asarray(lev, float)

    def at(self, t):
        idx = np.clip(np.searchsorted(self.t_s, np.asarray(t, float), side="right") - 1, 0, len(self.t_s) - 1)
        return self.levels[idx]


def _fit(its, terms, joint_type, n_starts, eff):
    spec = ModelSpec(terms, joint_type, "joint_space_only")
    ls = general_integral_ls(its, spec)
    f = general_rollout_fit(its, spec, theta0_from_ls(ls, spec, dict(effort=eff)), n_starts=n_starts,
                            scale_hint=dict(effort=eff))
    return spec, f


def _neff(res_parts):
    rhos = [float(np.corrcoef(r[:-1], r[1:])[0, 1]) for r in res_parts if len(r) > 3 and np.std(r) > 0]
    rho = float(np.clip(np.mean(rhos), 0.0, 0.95)) if rhos else 0.0
    return (1 + rho) / (1 - rho), rho


def load_model_comparison(interactions, base_terms: GeneralTerms, joint_type, *, n_starts=2, lags_s=(-0.02, -0.01,
                                                                                                     0.01, 0.02)):
    its = list(interactions)
    eff = float(np.max(np.abs(np.concatenate([np.asarray(i.effort.levels) for i in its]))))
    base = dict(base_terms.to_dict(), gravity=False, bias=False)
    base.pop("quasistatic", None)
    q_all = np.concatenate([i.q_obs for i in its])
    qg = np.linspace(q_all.min(), q_all.max(), 50)
    X = np.c_[np.ones_like(qg), np.cos(qg), np.sin(qg)]
    Xn = X / np.linalg.norm(X, axis=0)
    sv = np.linalg.svd(Xn, compute_uv=False)
    cond_D = float(sv[0] / sv[-1])
    Xc = np.c_[np.cos(qg), np.sin(qg)]
    svc = np.linalg.svd(Xc / np.linalg.norm(Xc, axis=0), compute_uv=False)
    cond_C = float(svc[0] / svc[-1])
    variants = {"A_base": GeneralTerms(**base), "B_bias": GeneralTerms(**dict(base, bias=True))}
    if joint_type == "revolute":
        variants["C_effective_load"] = GeneralTerms(**dict(base, gravity=True))
        variants["D_bias_plus_load"] = GeneralTerms(**dict(base, gravity=True, bias=True))
    out = dict(observed_q_range=[float(q_all.min()), float(q_all.max())], basis_condition_number_C=cond_C,
               basis_condition_number_D=cond_D,
               D_structurally_resolvable=bool(cond_D < 1e3), variants={},
               label_note="the load basis is an effective_configuration_load, never physical gravity")
    fits = {}
    for name, t in variants.items():
        try:
            spec, f = _fit(its, t, joint_type, n_starts, eff)
        except Exception as e:                                           # report, do not stop the comparison
            out["variants"][name] = dict(status="FAILED", error=f"{type(e).__name__}: {e}")
            continue
        P = f["problem"]
        r = P.residuals(np.asarray(f["x"]))
        parts = np.split(r, np.cumsum([len(i.t_obs) for i in its])[:-1])
        infl, rho = _neff(parts)
        k = len(f["x"])
        n = len(r)
        c2 = float(r @ r)
        rep = dict(terms=t.status(), chi2=c2, n=n, k=k, chi2_dof=c2 / max(n - k, 1), AIC=c2 + 2 * k,
                   BIC=c2 + k * math.log(n), residual_lag1_rho=rho, chi2_autocorr_inflation=infl,
                   estimates={nm: e["value"] for nm, e in f["estimates"].items()},
                   score_label="TRAINING_ONLY (no independent episodes)")
        if "gA" in f["estimates"]:
            names = P.names
            ia, ib = names.index("gA"), names.index("gB")
            C = np.asarray(f["covariance"], float)[np.ix_([ia, ib], [ia, ib])] * infl
            gA, gB = f["estimates"]["gA"]["value"], f["estimates"]["gB"]["value"]
            L = gA * np.cos(qg) + gB * np.sin(qg)
            imax, imin = int(np.argmax(L)), int(np.argmin(L))
            g = np.array([math.cos(qg[imax]) - math.cos(qg[imin]), math.sin(qg[imax]) - math.sin(qg[imin])])
            dL = float(L[imax] - L[imin])
            sdL = float(math.sqrt(max(g @ np.nan_to_num(C) @ g, 0.0)))
            mean_L = float(L.mean())
            gm = np.array([np.mean(np.cos(qg)), np.mean(np.sin(qg))])
            sd_mean = float(math.sqrt(max(gm @ np.nan_to_num(C) @ gm, 0.0)))
            w, V = np.linalg.eigh(np.nan_to_num(C))
            rep["effective_configuration_load"] = dict(
                gA=gA, gB=gB, cov_gA_gB=C.tolist(), ellipse95=dict(center=[gA, gB],
                                                                  semi_axes=(np.sqrt(np.clip(w, 0, None) * 5.991)).tolist(),
                                                                  axes=V.tolist()),
                variation_over_observed_q=dict(estimate=dL, sd=sdL, resolved=bool(abs(dL) > 3 * sdL)),
                mean_level_over_observed_q=dict(estimate=mean_L, sd=sd_mean, resolved=bool(abs(mean_L) > 3 * sd_mean)),
                interpretation="effective load; its mean level is confounded with a constant bias / force offset "
                               "when b is omitted; only its VARIATION over q distinguishes it from a bias",
                label="CONDITIONAL")
        out["variants"][name] = rep
        fits[name] = (spec, f)

    def lr(a, b, df):
        va, vb = out["variants"].get(a), out["variants"].get(b)
        if not va or not vb or "chi2" not in va or "chi2" not in vb:
            return None
        infl = max(va["chi2_autocorr_inflation"], vb["chi2_autocorr_inflation"])
        d = max(va["chi2"] - vb["chi2"], 0.0) / infl
        return dict(null=a, alternative=b, delta_chi2_corrected=d, df=df, p_value=float(chi2dist.sf(d, df)),
                    note="likelihood ratio on training data, delta chi2 divided by the residual autocorrelation "
                         "inflation; not a held-out validation")
    out["tests"] = [x for x in (lr("A_base", "B_bias", 1), lr("A_base", "C_effective_load", 2),
                                lr("B_bias", "D_bias_plus_load", 2), lr("C_effective_load", "D_bias_plus_load", 1))
                    if x]
    # alternatives to a load for the base model A
    alt = {}
    if "A_base" in fits:
        spec, f = fits["A_base"]
        P = f["problem"]
        x = np.asarray(f["x"])
        c0 = float(np.sum(P.residuals(x) ** 2))
        lag_rep = {}
        for lag in lags_s:
            effs = [_Eff(i.effort.t_s + lag, i.effort.levels) for i in its]
            try:
                from ..estimators.model_spec import G9Interaction
                its_l = [G9Interaction(i.name, i.joint_type, i.t_obs, i.q_obs, i.q_std, e2, i.initial_state, i.meta)
                         for i, e2 in zip(its, effs)]
                _, fl = _fit(its_l, variants["A_base"], joint_type, 1, eff)
                lag_rep[f"{lag * 1e3:+.0f}ms"] = float(np.sum(fl["problem"].residuals(np.asarray(fl["x"])) ** 2))
            except Exception as e:
                lag_rep[f"{lag * 1e3:+.0f}ms"] = f"FAILED: {e}"
        alt["time_offset_refit_chi2"] = dict(nominal=c0, **lag_rep)
        r = P.residuals(x)
        parts = np.split(r, np.cumsum([len(i.t_obs) for i in its])[:-1])
        c_tr = 0.0
        for it, rp in zip(its, parts):
            Xt = np.c_[np.ones(len(rp)), it.t_obs - it.t_obs[0]]
            beta, *_ = np.linalg.lstsq(Xt, rp, rcond=None)
            c_tr += float(np.sum((rp - Xt @ beta) ** 2))
        alt["linear_q_trend_post_hoc_chi2"] = dict(nominal=c0, after_trend=c_tr,
                                                    note="post-hoc absorption of a linear-in-time q residual by A "
                                                         "(upper bound on what a q drift could explain)")
        if "B_bias" in out["variants"] and "chi2" in out["variants"]["B_bias"]:
            alt["residual_constant_force_chi2"] = out["variants"]["B_bias"]["chi2"]
    out["alternatives_for_A"] = alt
    best = {k: v["AIC"] for k, v in out["variants"].items() if "AIC" in v}
    out["lowest_training_AIC"] = min(best, key=best.get) if best else None
    out["selection_policy"] = ("automatic_term_selection is false: this table is a diagnostic; a physical load claim "
                               "requires held-out episodes and, for gravity, the geometry-coupled model")
    return out
