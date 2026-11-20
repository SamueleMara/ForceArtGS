"""Identifiability of the GENERAL model (G9 spec 4.3) -- not the viscous CRLB copied to more parameters.

Per enabled parameter the verdict combines:
  1. local information: Jacobian of q(t_obs) w.r.t. every free physical parameter AND every nuisance initial
     state at the rollout estimate; scaled singular values, numerical rank, correlations, Cramer-Rao std inflated
     for residual autocorrelation (n_eff from the lag-1 coefficient: correlated visual-q errors);
  2. systematic sensitivity: Gauss-Newton parameter shifts under declared input perturbations (force lag, force
     scale, force bias / tare, joint axis, hinge position, contact point) added in quadrature to the statistical std;
  3. structural confounds read from the data (constant speed, one motion direction, prismatic gravity vs bias,
     vertical hinge, short range, no breakaway, single speed level).
A parameter can have a fitted number and still be ``identifiable: false``; such numbers stay in diagnostic JSON.
"""
from __future__ import annotations

import math

import numpy as np

from ..estimators.general_integral_ls import moving_segments, velocity_estimate

REL_STD_LIMIT = 0.10          # preregistered (same 10 % threshold as the frozen viscous diagnostic)
CORR_LIMIT = 0.95
POSITIVE = ("alpha", "M", "c", "Fc", "dFs", "Fs", "mu")


def _neff_factor(res_by_it) -> tuple:
    rhos = []
    for r in res_by_it:
        if len(r) > 3 and np.std(r) > 0:
            rhos.append(float(np.corrcoef(r[:-1], r[1:])[0, 1]))
    rho = float(np.clip(np.mean(rhos), 0.0, 0.95)) if rhos else 0.0
    return math.sqrt((1 + rho) / (1 - rho)), rho


def structural_checks(interactions, spec) -> dict:
    """Excitation / confound flags computed from q(t) and the effort only (no fit, no truth)."""
    t_ = spec.terms
    speeds, dirs, rest_with_effort, breakaways, ranges, accel_cv = [], set(), 0, 0, [], []
    linear_segments, n_segments = 0, 0
    for it in interactions:
        segs, v, eps = moving_segments(it.t_obs, it.q_obs, it.q_std)
        mov = np.zeros(len(v), bool)
        for a, z in segs:
            mov[a:z] = True
            dirs.add(int(np.sign(np.median(v[a:z]))))
            sp = np.abs(v[a:z])
            speeds.append(sp)
            n_segments += 1
            tt, qq = it.t_obs[a:z], it.q_obs[a:z]
            res = qq - np.polyval(np.polyfit(tt, qq, 1), tt)
            if np.sqrt(np.mean(res ** 2)) <= 1.5 * float(np.median(it.q_std[a:z])):
                linear_segments += 1                     # q indistinguishable from constant velocity
            if len(sp) > 4:
                accel_cv.append(float(np.std(sp) / max(np.mean(sp), 1e-12)))
        Q = it.effort.at(it.t_obs)
        rest = ~mov
        rest_with_effort += int(np.sum(rest & (np.abs(Q) > 0.05 * max(np.abs(Q).max(), 1e-12))))
        breakaways += int(np.sum(rest[:-1] & mov[1:]))
        ranges.append(float(np.ptp(it.q_obs)))
    allsp = np.concatenate(speeds) if speeds else np.zeros(0)
    out = dict(motion_directions=sorted(dirs), n_breakaway_transitions=breakaways,
               rest_samples_under_effort=rest_with_effort, q_range=max(ranges) if ranges else 0.0,
               speed_cv_within_segments=float(np.median(accel_cv)) if accel_cv else 0.0,
               constant_velocity_segments=linear_segments, moving_segments=n_segments,
               speed_levels_p10_p90=[float(np.percentile(allsp, 10)), float(np.percentile(allsp, 90))]
               if len(allsp) else None, confounds=[], unidentifiable=[])
    names = set(spec.names)
    inertia_name = "alpha" if "alpha" in names else ("M" if "M" in names else None)
    if inertia_name and n_segments and linear_segments == n_segments and breakaways == 0:
        out["unidentifiable"].append((inertia_name, "CONSTANT_SPEED: no acceleration excitation"))
    if t_.coulomb and len(dirs) < 2:
        if t_.bias:
            out["confounds"].append(("Fc", "b", "ONE_DIRECTION: Coulomb and constant bias share one regressor"))
        out["confounds"].append(("Fc", "force_offset", "ONE_DIRECTION: Coulomb is collinear with any uncorrected "
                                                       "constant force offset (tare)"))
    if t_.static and ("dFs" in names or "Fs" in names) and breakaways == 0:
        out["unidentifiable"].append(("dFs" if "dFs" in names else "Fs",
                                      "NO_BREAKAWAY_OBSERVED: no rest-to-motion transition under effort"))
    if t_.viscous and t_.coulomb and len(allsp):
        p10, p90 = out["speed_levels_p10_p90"]
        if len(dirs) < 2 and p90 < 1.5 * max(p10, 1e-12):
            out["confounds"].append(("c", "Fc", "SINGLE_SPEED_ONE_DIRECTION: viscous and Coulomb collinear"))
    if t_.gravity:
        if spec.joint_type == "prismatic" and t_.bias:
            out["confounds"].append(("gravity", "b", "PRISMATIC_GRAVITY_IS_CONSTANT: gravity load and bias collinear"))
        if spec.joint_type == "prismatic" and spec.parameter_source == "joint_space_only":   # coupled: no confound
            out["confounds"].append(("gP", "Fc" if t_.coulomb else "force_offset",
                                     "PRISMATIC_EFFECTIVE_GRAVITY_CONSTANT: an effective constant load is not "
                                     "separable from offsets without the geometry coupling"))
        if spec.gravity is not None and spec.joint_type == "revolute":
            a = np.asarray(spec.mass_model.axis if spec.mass_model is not None else [0, 0, 1], float)
            g = np.asarray(spec.gravity.g_world, float)
            tilt = math.degrees(math.asin(min(1.0, np.linalg.norm(np.cross(a, g / np.linalg.norm(g))))))
            out["axis_tilt_from_vertical_deg"] = tilt
            if tilt < 5.0:
                out["notes"] = (out.get("notes") or []) + [
                    f"VERTICAL_HINGE: axis within {tilt:.2f} deg of gravity; the gravity effort is near zero "
                    "(this does not mean gravity is disabled)"]
    return out


def systematic_shifts(fit: dict, interactions, perturb: dict) -> dict:
    """Linearised parameter shifts dtheta = -(J^T J)^-1 J^T dr for declared input perturbations.

    ``perturb`` maps a name to a callable(interaction) -> perturbed effort (same timeline), e.g. lag, scale, bias,
    axis / hinge / contact-point re-projections built by the adapter."""
    P = fit["problem"]
    x = np.asarray(fit["x"], float)
    J = fit["jacobian"]
    keep = [k for k in range(J.shape[1]) if np.any(J[:, k] != 0)]
    Jk = J[:, keep]
    H = np.linalg.pinv(Jk.T @ Jk)
    r0 = P.residuals(x)
    out = {}
    for name, fn in perturb.items():
        effs = [fn(it) for it in P.its]
        dr = P.residuals(x, effort_override=effs) - r0
        d = np.zeros(J.shape[1])
        d[keep] = -H @ (Jk.T @ dr)
        out[name] = {nm: float(d[j]) for j, nm in enumerate(P.names)}
    return out


def diagnose_general(fit: dict, interactions, spec, *, perturb: dict | None = None) -> dict:
    """Per-parameter identifiability report (see module docstring)."""
    if fit.get("status") not in ("ok", "not_converged"):
        return dict(status="NOT_RUN", reason=f"fit status {fit.get('status')}")
    P = fit["problem"]
    J = fit["jacobian"]
    x = np.asarray(fit["x"], float)
    res = P.residuals(x)
    sizes = [len(it.t_obs) for it in P.its]
    parts = np.split(res, np.cumsum(sizes)[:-1])
    infl, rho = _neff_factor(parts)
    cols = np.linalg.norm(J, axis=0)
    nz = cols > 0
    Jn = J[:, nz] / cols[nz]
    sv = np.linalg.svd(Jn, compute_uv=False) if Jn.size else np.array([1.0])
    rank = int((sv > sv[0] * 1e-8).sum())
    cov = np.asarray(fit["covariance"], float)
    names = P.names
    nphys = len(names)
    Cp = cov[:nphys, :nphys]
    d = np.sqrt(np.clip(np.diag(Cp), 1e-300, None))
    corr = Cp / np.outer(d, d)
    corr[~np.isfinite(corr)] = np.nan
    struct = structural_checks(interactions, spec)
    sysd = systematic_shifts(fit, interactions, perturb) if perturb else {}
    eff_scale = float(np.sqrt(np.mean(np.concatenate([np.asarray(it.effort.levels) ** 2 for it in P.its]))))
    fams = scenario_families(sysd)
    report = {}
    for j, nm in enumerate(names):
        e = fit["estimates"][nm]
        val = e["value"]
        stat = e["std"] * (infl if "interval95" not in e else 1.0)
        # declared scenarios are NOT standard deviations: per family the signed shifts, the largest single-family
        # excursion and a deterministic linear worst-case sum (labelled), never combined in quadrature
        fam_rep = {f: dict(shifts={k: v[nm] for k, v in d.items()}, max_abs=max(abs(v[nm]) for v in d.values()),
                           label="DECLARED_SCENARIO") for f, d in fams.items()}
        worst_single = max((v["max_abs"] for v in fam_rep.values()), default=0.0)
        linear_sum = float(sum(v["max_abs"] for v in fam_rep.values()))
        reasons = []
        if nm in POSITIVE:
            rel = stat / abs(val) if val != 0 else math.inf
            resolved = rel <= REL_STD_LIMIT
            if not resolved:
                reasons.append(f"statistical relative std {rel:.3g} > {REL_STD_LIMIT}")
            if e.get("at_lower_bound"):
                resolved = False
                reasons.append("estimate at its physical bound (0): not determined away from the bound")
            sens = worst_single / abs(val) if val != 0 else math.inf
            sensitive = sens > REL_STD_LIMIT
        else:
            rel = None
            resolved = stat <= 0.05 * eff_scale
            if not resolved:
                reasons.append(f"statistical std {stat:.3g} > 5% of the effort RMS {eff_scale:.3g}")
            sensitive = worst_single > 0.05 * eff_scale
        if "interval95" in e and not math.isfinite(e["interval95"][1]):
            resolved = False
            reasons.append("profile interval unbounded above")
        hi_corr = [other for k, other in enumerate(names)
                   if k != j and np.isfinite(corr[j, k]) and abs(corr[j, k]) > CORR_LIMIT]
        if hi_corr:
            # the marginal std already contains the correlation; it explains a failure, it is not one by itself
            msg = f"|corr| > {CORR_LIMIT} with {hi_corr}"
            reasons.append(msg + (": only a combination is determined" if not resolved else
                                  " (marginal std still small)"))
        for (p, why) in struct["unidentifiable"]:
            if p == nm:
                resolved = False
                reasons.append(why)
        for cf in struct["confounds"]:
            if nm in cf[:2] or (nm in ("gA", "gB", "gP") and "gravity" in cf[:2]):
                if cf[1] == "force_offset":
                    reasons.append(f"WARNING {cf[2]}")
                else:
                    resolved = False
                    reasons.append(cf[2])
        if sensitive:
            worst = max(fam_rep, key=lambda f: fam_rep[f]["max_abs"])
            reasons.append(f"scenario-sensitive: declared scenario {worst!r} moves the estimate by "
                           f"{worst_single:.3g} (> {REL_STD_LIMIT:.0%})")
        ok = bool(resolved and not sensitive)
        ci_stat = [val - 1.96 * stat, val + 1.96 * stat] if "interval95" not in e else e["interval95"]
        report[nm] = dict(estimate=val, std_statistical=stat, ci95_statistical=ci_stat,
                          uncertainty_labels=dict(ci95_statistical="STATISTICAL", sensitivity_scenarios="DECLARED_SCENARIO",
                                                  calibrated_uncertainty="CALIBRATED (none supplied)"),
                          sensitivity_scenarios=fam_rep, scenario_worst_single_family=worst_single,
                          scenario_linear_worst_case_sum=linear_sum,
                          scenario_envelope=[val - worst_single, val + worst_single],
                          calibrated_uncertainty=None,
                          # backward-compatible keys: std_total / ci95 are the STATISTICAL quantities (no quadrature
                          # with scenarios); std_systematic = largest single declared-scenario excursion
                          std_systematic=worst_single, std_total=stat, ci95=ci_stat,
                          rel_std=rel, statistically_resolved_in_this_model=bool(resolved),
                          scenario_sensitive=bool(sensitive), identifiable=ok, reasons=reasons,
                          failure_code=None if ok else ("SCENARIO_SENSITIVE" if resolved else "UNIDENTIFIABLE"))
    # scale-free combinations: a coasting / slowly pushed part can determine them when the absolute scale is not
    derived = {}
    iM = names.index("M") if "M" in names else (names.index("alpha") if "alpha" in names else None)
    if iM is not None:
        Mv = fit["physical"]["M"]
        dMdth = Mv / x[iM] if x[iM] != 0 else 0.0
        for nm, lbl in (("c", "decay_time_M_over_c_s"), ("Fc", "coulomb_deceleration_Fc_over_M")):
            if nm not in names:
                continue
            j = names.index(nm)
            v = x[j]
            if lbl.startswith("decay"):
                val = Mv / v if v > 0 else math.inf
                g = np.zeros(len(names))
                g[iM], g[j] = dMdth / v if v > 0 else 0.0, -Mv / v ** 2 if v > 0 else 0.0
            else:
                val = v / Mv if Mv > 0 else math.inf
                g = np.zeros(len(names))
                g[iM], g[j] = -v / Mv ** 2 * dMdth if Mv > 0 else 0.0, 1.0 / Mv if Mv > 0 else 0.0
            C = np.nan_to_num(Cp)
            sd = float(math.sqrt(max(g @ C @ g, 0.0))) * infl
            rel = sd / abs(val) if val not in (0, math.inf) else math.inf
            dsc = {f: {k: float(g @ np.array([sh[n] for n in names])) for k, sh in d.items()} for f, d in fams.items()}
            worst = max((max(abs(x2) for x2 in v.values()) for v in dsc.values()), default=0.0)
            sens = worst / abs(val) if val not in (0, math.inf) else math.inf
            derived[lbl] = dict(estimate=val, std=sd, rel_std=rel, std_label="STATISTICAL",
                                statistically_resolved_in_this_model=bool(rel <= REL_STD_LIMIT),
                                sensitivity_scenarios=dsc, scenario_worst_single_family=worst,
                                scenario_sensitive=bool(sens > REL_STD_LIMIT),
                                identifiable=bool(rel <= REL_STD_LIMIT and sens <= REL_STD_LIMIT),
                                method="delta method on the rollout covariance (statistical) + linearised declared "
                                       "scenarios (DECLARED_SCENARIO, not a std)")
    all_ok = all(v["identifiable"] for v in report.values())
    return dict(status="PASS" if all_ok else "FAIL", parameters=report, derived_combinations=derived, rank=rank,
                n_columns=int(J.shape[1]),
                scaled_singular_values=sv.tolist(),
                condition_number=float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf,
                correlation={f"{a}~{b}": (None if not np.isfinite(corr[i, k]) else float(corr[i, k]))
                             for i, a in enumerate(names) for k, b in enumerate(names) if k > i},
                residual_lag1_autocorrelation=rho, std_inflation_autocorrelation=infl,
                structural=struct, systematic_shifts=sysd, scenario_families=sorted(fams),
                thresholds=dict(rel_std=REL_STD_LIMIT, corr=CORR_LIMIT),
                note="STATISTICAL CIs from the fit covariance / profile (autocorrelation-inflated); declared scenarios "
                     "reported separately (linearised shifts), never added in quadrature; necessary, not sufficient")


# ----------------------------------------------------------------------------------------------- perturbations
class _Eff:
    def __init__(self, t, lev):
        self.t_s, self.levels = np.asarray(t, float), np.asarray(lev, float)

    def at(self, t):
        idx = np.clip(np.searchsorted(self.t_s, np.asarray(t, float), side="right") - 1, 0, len(self.t_s) - 1)
        return self.levels[idx]


def scenario_families(shifts: dict) -> dict:
    """Group 'family|+' / 'family|-' scenario names (an unsuffixed name is its own one-sided family)."""
    fams = {}
    for k, v in shifts.items():
        f, _, sgn = k.partition("|")
        fams.setdefault(f, {})[sgn or "+"] = v
    return fams


def standard_perturbations(lag_s=None, scale_rel=None, bias=None) -> dict:
    """Declared SYMMETRIC input scenarios on the generalized effort: 'family|+' and 'family|-' (adapter adds the
    geometric ones). These are sensitivity scenarios, not calibrated uncertainties."""
    out = {}
    if lag_s:
        for s_, sg in ((lag_s, "+"), (-lag_s, "-")):
            out[f"force_lag_{lag_s * 1e3:.0f}ms|{sg}"] = (lambda it, s_=s_: _Eff(it.effort.t_s + s_, it.effort.levels))
    if scale_rel:
        for r_, sg in ((scale_rel, "+"), (-scale_rel, "-")):
            out[f"force_scale_{scale_rel:.0%}_UNVALIDATED_STRESS_TEST|{sg}"] = (
                lambda it, r_=r_: _Eff(it.effort.t_s, it.effort.levels * (1 + r_)))
    if bias:
        for b_, sg in ((bias, "+"), (-bias, "-")):
            out[f"force_bias_{bias:.3g}|{sg}"] = (lambda it, b_=b_: _Eff(it.effort.t_s, it.effort.levels + b_))
    return out
