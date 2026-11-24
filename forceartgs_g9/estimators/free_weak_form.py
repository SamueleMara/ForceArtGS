"""Constrained weak-form (integral) identification of the NORMALIZED joint dynamics (spec sections 4, 5, 6.1).

Free window [ta, tb] of a monotonic segment (sign s):
    dv + lam dq + gam s dt + int a_g(q) dt = 0
Forced window (trusted measured input, beta = 1/M > 0):
    dv + lam dq + gam s dt + int a_g(q) dt = beta J_Q,        J_Q = int Q dt (exact ZOH, native rows)

Linear in (lam, gam[, beta]) for a fixed a_g -> bounded weighted LS (lsq_linear 'bvls', all >= 0). NO absolute M,
c or Fc appears in the free problem (the scale is structurally absent, never a fitted bound at 0). Velocities at the
window ends come from a GCV smoothing spline of q (no double differentiation); their Monte Carlo sd enters the row
weight. Uncertainty: covariance (chi2-scaled), profile likelihood, window bootstrap, smoothing-bandwidth sensitivity.

Status (free): FREE_RATIOS_IDENTIFIED | FREE_COMBINATION_IDENTIFIED | FREE_UNINFORMATIVE | FREE_MODEL_INADEQUATE
(the replay gate is applied by the caller); absolute scale: ABSOLUTE_SCALE_NOT_OBSERVABLE without a forced anchor.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import lsq_linear

COND_COMB = 30.0          # column-normalised whitened condition number above which only a combination is supported
CORR_COMB = 0.95          # |posterior correlation| above which only a combination is supported
REL_OK = 0.5              # 95 % half-width / estimate below which a rate counts as identified
CHI2_INADEQUATE = 25.0    # whitened chi2 / dof above which the integral equation itself fails
DCHI2_95 = 3.841


# =================================================================================================== rows
_GCV_CACHE = {}


def gcv_lambda(t, q, q_sigma, n_grid=40, rule="discrepancy"):
    """Smoothing-spline penalty, computed explicitly (hat matrix from unit-vector fits) so that a FIXED lambda can be
    scaled for the bandwidth sensitivity (scipy's BSpline does not expose it).
    rule 'discrepancy' (default): reduced chi2 = sum((r/sigma)^2)/(n - tr H) closest to 1 under the GIVEN sigma (the
    caller passes the empirically checked sigma); 'gcv': generalized cross-validation (sigma-scale free; unreliable
    for ~15 samples). Falls back to GCV when no grid value brackets chi2_red = 1."""
    from scipy.interpolate import make_smoothing_spline
    key = (np.asarray(t, float).tobytes(), np.asarray(q, float).tobytes(), np.asarray(q_sigma, float).tobytes(),
           rule)
    if key in _GCV_CACHE:
        return _GCV_CACHE[key]
    t, q, s = (np.asarray(x, float) for x in (t, q, q_sigma))
    w = 1.0 / s ** 2
    w = w / w.mean()
    n = len(t)
    T = float(t[-1] - t[0])
    best, disc = (math.inf, None), []
    for lam in np.geomspace(1e-12, 1e2, n_grid) * T ** 3:
        H = np.empty((n, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            H[:, i] = make_smoothing_spline(t, e, w=w, lam=lam)(t)
        r = q - H @ q
        tr = float(np.trace(H))
        if n - tr <= 0.5:
            continue
        g = n * float(np.sum(w * r ** 2)) / (n - tr) ** 2
        if g < best[0]:
            best = (g, float(lam))
        disc.append((float(np.sum((r / s) ** 2)) / (n - tr), float(lam)))
    out = best[1]
    if rule == "discrepancy" and disc:
        c = np.array([d[0] for d in disc])
        if c.min() <= 1.0 <= c.max():
            out = disc[int(np.argmin(np.abs(np.log(np.maximum(c, 1e-300)))))][1]
    _GCV_CACHE[key] = out
    return out


def _smooth(seg, lam_factor=1.0, method="spline", n_mc=30):
    from ..forcetwin.derivatives import local_poly, spline
    tg = np.linspace(seg.t[0], seg.t[-1], max(120, int((seg.t[-1] - seg.t[0]) * 400)))
    if method == "local_poly":
        h = max(2.5 * float(np.median(np.diff(seg.t))), 0.06)
        d = local_poly(seg.t, seg.q, seg.q_sigma, tg, h=h)
        d["lam"] = None
        return tg, d
    lam = gcv_lambda(seg.t, seg.q, seg.q_sigma)
    lam = None if lam is None else lam * lam_factor
    d = spline(seg.t, seg.q, seg.q_sigma, tg, n_mc=n_mc, lam=lam)
    d["lam"] = lam
    return tg, d


def window_rows(seg, nspec, *, L=None, edge=None, lam_factor=1.0, method="spline", prefix=False):
    """Integral rows of ONE monotonic segment. Non-overlapping windows (default) or prefixes from the start."""
    tg, d = _smooth(seg, lam_factor, method)
    dt_q = float(np.median(np.diff(seg.t)))
    edge = 0.5 * dt_q if edge is None else edge
    lo, hi = seg.t[0] + edge, seg.t[-1] - edge
    T = hi - lo
    if T <= 0:
        return []
    L = max(2.0 * dt_q, T / 6.0) if L is None else L
    n = max(1, int(T // L + 1e-9))
    if prefix:
        cuts = [(lo, lo + (k + 1) * T / n) for k in range(n)]
    else:
        Lr = T / n
        cuts = [(lo + k * Lr, lo + (k + 1) * Lr) for k in range(n)]
    rows = []
    for ta, tb in cuts:
        ia, ib = int(np.searchsorted(tg, ta)), int(min(np.searchsorted(tg, tb), len(tg) - 1))
        sub = np.linspace(tg[ia], tg[ib], 200)
        qs = np.interp(sub, tg, d["q"])
        dv = float(d["qd"][ib] - d["qd"][ia])
        sdv = float(math.hypot(d["qd_sd"][ia], d["qd_sd"][ib]))
        J, sJ = seg.J(tg[ia], tg[ib])
        rows.append(dict(seg=seg.name, label=seg.label, free=seg.is_free, ta=float(tg[ia]), tb=float(tg[ib]),
                         dv=dv, sd_dv=max(sdv, 1e-12), dq=float(d["q"][ib] - d["q"][ia]),
                         s_dt=float(seg.sign * (tg[ib] - tg[ia])), int_ag=float(np.trapz(nspec.gravity(qs), sub)),
                         int_cos=float(np.trapz(np.cos(qs), sub)), int_sin=float(np.trapz(np.sin(qs), sub)),
                         dt=float(tg[ib] - tg[ia]),
                         J=J, sd_J=sJ, v_a=float(d["qd"][ia]), v_b=float(d["qd"][ib]), smoother=method,
                         lam_spline=d.get("lam")))
    return rows


def all_rows(segments, nspec, **kw):
    out = []
    for s in segments:
        out += window_rows(s, nspec, **kw)
    return out


def _design(rows, names, beta_known=None, lam_known=None):
    """A theta = y  with y = -(dv + int a_g); columns lam: dq, gam: s dt, beta: -J (forced rows only)."""
    A, y, sd = [], [], []
    for r in rows:
        col = {"lam": r["dq"], "gam": r["s_dt"], "beta": (0.0 if r["free"] else -r["J"])}
        A.append([col[n] for n in names])
        rhs = -(r["dv"] + r["int_ag"])
        var = r["sd_dv"] ** 2
        if not r["free"]:
            b = beta_known if beta_known is not None else 0.0
            var += (b * r["sd_J"]) ** 2
        y.append(rhs)
        sd.append(math.sqrt(var))
    return np.asarray(A, float).reshape(-1, len(names)), np.asarray(y), np.asarray(sd)


def _lsq(A, y, w):
    Aw, yw = A * w[:, None], y * w
    r = lsq_linear(Aw, yw, bounds=(0.0, np.inf), method="bvls")
    res = Aw @ r.x - yw
    return r.x, float(res @ res)


def solve(rows, names, *, n_irls=2):
    """Weighted bounded LS with IRLS on the forced-row variance (beta sd_J)."""
    beta = None
    for _ in range(n_irls):
        A, y, sd = _design(rows, names, beta_known=beta)
        w = 1.0 / sd
        x, chi2 = _lsq(A, y, w)
        if "beta" in names:
            beta = float(x[names.index("beta")])
    return x, chi2, A, y, sd


def _svd(A, w, names):
    Aw = A * w[:, None]
    nrm = np.linalg.norm(Aw, axis=0)
    nrm[nrm == 0] = 1.0
    U, S, Vt = np.linalg.svd(Aw / nrm, full_matrices=False)
    cond = float(S[0] / S[-1]) if S[-1] > 0 else math.inf
    return dict(singular_values=S.tolist(), condition_number=cond,
                right_singular_vectors={f"v{i}": dict(zip(names, (Vt[i] / nrm).tolist())) for i in range(len(S))},
                rank=int(np.sum(S > 1e-6 * S[0])) if S[0] > 0 else 0,
                note="columns whitened and unit-normalised; vectors mapped back to parameter units")


def _cov(A, w, chi2, dof):
    Aw = A * w[:, None]
    try:
        C = np.linalg.inv(Aw.T @ Aw)
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(Aw.T @ Aw)
    s = max(1.0, chi2 / max(dof, 1))
    return C, C * s, s


def profile(rows, names, i, x_hat, chi2_min, *, n=81, scale=1.0):
    """Profile chi2 over parameter i (others re-minimised, >= 0)."""
    A, y, sd = _design(rows, names, beta_known=x_hat[names.index("beta")] if "beta" in names else None)
    w = 1.0 / sd
    col = float(np.linalg.norm(A[:, i] * w))
    data_scale = float(np.linalg.norm(y * w)) / col if col > 0 else 1.0      # the value that alone explains y
    hi = max(4.0 * abs(x_hat[i]), 0.1 * data_scale, 1e-9)
    # extend until the profile rises by > 4 dchi2 (or a hard cap)
    for _ in range(12):
        others = [j for j in range(len(names)) if j != i]
        Aw, yw = A * w[:, None], y * w
        r = lsq_linear(Aw[:, others], yw - Aw[:, i] * hi, bounds=(0, np.inf), method="bvls") if others else None
        c = float(np.sum((Aw[:, others] @ r.x - (yw - Aw[:, i] * hi)) ** 2)) if others else \
            float(np.sum((Aw[:, i] * hi - yw) ** 2))
        if c - chi2_min > 4 * DCHI2_95 * scale:
            break
        hi *= 2.5
    others = [j for j in range(len(names)) if j != i]
    Aw, yw = A * w[:, None], y * w

    def pchi2(g):
        if others:
            r = lsq_linear(Aw[:, others], yw - Aw[:, i] * g, bounds=(0, np.inf), method="bvls")
            return float(np.sum((Aw[:, others] @ r.x - (yw - Aw[:, i] * g)) ** 2))
        return float(np.sum((Aw[:, i] * g - yw) ** 2))
    thr = DCHI2_95 * scale
    grid = np.linspace(0.0, hi, n)
    prof = np.array([pchi2(g) for g in grid])
    edge = bool(((prof - chi2_min) <= thr)[-1])
    # zoom: refine each interval end between the last accepted and the first rejected grid point
    ok = (prof - chi2_min) <= thr
    lo_ci = hi_ci = None
    if ok.any():
        k0, k1 = int(np.flatnonzero(ok)[0]), int(np.flatnonzero(ok)[-1])

        def bisect(a, b, inside_at_a):
            for _ in range(40):
                m = 0.5 * (a + b)
                if ((pchi2(m) - chi2_min) <= thr) == inside_at_a:
                    a = m
                else:
                    b = m
            return 0.5 * (a + b)
        lo_ci = 0.0 if k0 == 0 else bisect(grid[k0], grid[k0 - 1], True)
        hi_ci = float(grid[-1]) if k1 == len(grid) - 1 else bisect(grid[k1], grid[k1 + 1], True)
    return dict(grid=grid.tolist(), dchi2=(prof - chi2_min).tolist(), ci95=[None if lo_ci is None else float(lo_ci),
                                                                             None if hi_ci is None else float(hi_ci)],
                ci_at_grid_edge=edge, threshold=thr,
                note="threshold scaled by max(1, chi2/dof); interval ends refined by bisection between grid points")


def bootstrap(rows, names, B=200, seed=0):
    if len(rows) < len(names) + 2:
        return None
    rng = np.random.default_rng(seed)
    xs = []
    for _ in range(B):
        sel = [rows[k] for k in rng.integers(0, len(rows), len(rows))]
        x, *_ = solve(sel, names)
        xs.append(x)
    xs = np.asarray(xs)
    return dict(B=B, seed=seed, sd=xs.std(0, ddof=1).tolist(), p2_5=np.percentile(xs, 2.5, 0).tolist(),
                p97_5=np.percentile(xs, 97.5, 0).tolist(), note="window resampling (windows ~independent; "
                                                                  "shared end-points correlate neighbours)")


def fit(segments, nspec, *, B=200, sensitivity=True, rows=None, **row_kw) -> dict:
    """Normalized weak-form fit on the given segments (free ones; forced ones too when nspec.forced)."""
    names = nspec.names
    segs = [s for s in segments if s.is_free or nspec.forced]
    rows = all_rows(segs, nspec, **row_kw) if rows is None else rows
    u = nspec.units
    out = dict(schema="forceartgs.g9.free_weak_fit/v1", names=names, units={n: u[n] for n in names},
               n_segments=len(segs), n_rows=len(rows), segments=[s.describe() for s in segs],
               equation=nspec.describe()["equation"], gravity=nspec.gravity.describe())
    if len(rows) < len(names) + 1:
        out.update(status="FREE_UNINFORMATIVE", reason=f"{len(rows)} integral windows for {len(names)} rates")
        return out
    x, chi2, A, y, sd = solve(rows, names)
    w = 1.0 / sd
    dof = max(len(rows) - len(names), 1)
    C, Cs, sc = _cov(A, w, chi2, dof)
    svd = _svd(A, w, names)
    # zero-dissipation (no-deceleration) test: chi2 of theta = 0
    chi2_0 = float(np.sum((y * w) ** 2))
    prof = {n: profile(rows, names, i, x, chi2, scale=sc) for i, n in enumerate(names)}
    corr = None
    if len(names) > 1:
        d = np.sqrt(np.clip(np.diag(Cs), 1e-300, None))
        corr = (Cs / np.outer(d, d)).tolist()
    out.update(estimate=dict(zip(names, x.tolist())), cov=Cs.tolist(), cov_unscaled=C.tolist(),
               sd=dict(zip(names, np.sqrt(np.clip(np.diag(Cs), 0, None)).tolist())), correlation=corr,
               chi2=chi2, dof=dof, chi2_per_dof=chi2 / dof, chi2_zero_dissipation=chi2_0,
               zero_dissipation_rejected=bool(chi2_0 - chi2 > 2 * DCHI2_95 * sc),
               svd=svd, profile=prof, rows=rows)
    if B:
        out["bootstrap"] = bootstrap(rows, names, B=B)
    if sensitivity:
        sens = {}
        for tag, kw in (("spline_lam_x0.3", dict(lam_factor=0.3)), ("spline_lam_x3", dict(lam_factor=3.0)),
                        ("local_poly", dict(method="local_poly"))):
            rr = all_rows(segs, nspec, **(row_kw | kw))
            if len(rr) >= len(names) + 1:
                xs, *_ = solve(rr, names)
                sens[tag] = dict(zip(names, xs.tolist()))
        out["smoothing_sensitivity"] = sens
    out.update(_status(out, nspec))
    return out


def _status(o, nspec):
    names = o["names"]
    x = o["estimate"]
    per = {}
    comb = None
    if o["chi2_per_dof"] > CHI2_INADEQUATE:
        st = "FREE_MODEL_INADEQUATE"
        why = [f"integral-equation chi2/dof {o['chi2_per_dof']:.1f} > {CHI2_INADEQUATE}"]
    elif not o["zero_dissipation_rejected"]:
        st = "FREE_UNINFORMATIVE"
        why = ["the data do not reject zero dissipation (no significant deceleration): upper bounds only"]
    else:
        cond = o["svd"]["condition_number"]
        rho = abs(o["correlation"][0][1]) if o.get("correlation") and len(names) > 1 else 0.0
        if len(names) > 1 and (cond > COND_COMB or rho > CORR_COMB):
            st = "FREE_COMBINATION_IDENTIFIED"
            v0 = o["svd"]["right_singular_vectors"]["v0"]
            nrm = sum(abs(v) for v in v0.values()) or 1.0
            wts = {k: v / nrm for k, v in v0.items()}
            val = sum(wts[k] * x[k] for k in names)
            C = np.asarray(o["cov"])
            wv = np.array([wts[k] for k in names])
            comb = dict(weights=wts, value=float(val), sd=float(math.sqrt(max(wv @ C @ wv, 0.0))),
                        meaning=("lam |v_mean| + gam  ~ mean deceleration" if set(names) >= {"lam", "gam"} else ""),
                        note="best-determined direction (largest singular value) of the whitened design")
            trig = ([f"condition {cond:.1f} > {COND_COMB}"] if cond > COND_COMB else []) + \
                   ([f"|corr(lam, gam)| {rho:.3f} > {CORR_COMB}"] if rho > CORR_COMB else [])
            why = [" and ".join(trig) + ": individual rates not separable (insufficient velocity variation)"]
        else:
            st = "FREE_RATIOS_IDENTIFIED"
            why = []
    for n in names:
        ci = o["profile"][n]["ci95"]
        est = x[n]
        if st in ("FREE_MODEL_INADEQUATE",):
            per[n] = dict(status="NOT_IDENTIFIED", reason="model inadequate")
        elif ci[0] is None:
            per[n] = dict(status="NOT_IDENTIFIED", reason="empty profile interval")
        elif st == "FREE_UNINFORMATIVE" or ci[0] <= 0.0:
            per[n] = dict(status="UPPER_BOUND_ONLY", upper95=ci[1], at_grid_edge=o["profile"][n]["ci_at_grid_edge"])
        elif st == "FREE_COMBINATION_IDENTIFIED":
            per[n] = dict(status="COMBINATION_ONLY", ci95=ci)
        elif est > 0 and (ci[1] - ci[0]) / 2 <= REL_OK * est:
            per[n] = dict(status="RATIO_IDENTIFIED", ci95=ci)
        else:
            per[n] = dict(status="WEAK", ci95=ci, reason=f"95 % half-width > {REL_OK:.0%} of the estimate")
    if st == "FREE_RATIOS_IDENTIFIED" and not any(p["status"] == "RATIO_IDENTIFIED" for p in per.values()):
        st, why = "FREE_COMBINATION_IDENTIFIED", why + ["no individual rate passes the interval rule"]
    return dict(status=st, reasons=why, per_parameter=per, combination=comb,
                absolute_scale="ABSOLUTE_SCALE_NOT_OBSERVABLE" if not nspec.forced else None)


# =================================================================================================== forced scale
def forced_scale(free_segments, forced_segments, nspec, *, free_fit=None, impulse_windows=(), B=200) -> dict:
    """Joint (lam, gam, beta) on free + trusted forced windows; per-window D vs J sign audit (no sign fixing);
    dimensional view (M, c, Fc) on the forced rows (quasi-static diagnostic)."""
    fs = nspec.with_forced()
    u = fs.units
    out = dict(schema="forceartgs.g9.forced_scale_fit/v1", n_forced_segments=len(forced_segments),
               n_impulse_windows=len(impulse_windows))
    rows_f = all_rows(forced_segments, fs)
    rows_free = all_rows(free_segments, fs.free())
    for r in rows_free:
        r["free"] = True
    rates = (free_fit or {}).get("estimate") or {}
    # ---------------- per-window sign audit with the FREE rates (D = dv + lam dq + gam s dt + int a_g)
    audit = []
    for r in rows_f + list(impulse_windows):
        D = r["dv"] + rates.get("lam", 0.0) * r["dq"] + rates.get("gam", 0.0) * r["s_dt"] + r["int_ag"]
        sD = r["sd_dv"]
        J, sJ = r["J"], r["sd_J"]
        sigJ, sigD = abs(J) > 3 * sJ, abs(D) > 3 * sD
        contra = bool(sigJ and sigD and np.sign(D) != np.sign(J))
        audit.append(dict(seg=r.get("seg"), ta=r.get("ta"), tb=r.get("tb"), D=D, sd_D=sD, J=J, sd_J=sJ,
                          D_over_J=D / J if J else None, J_significant=bool(sigJ), D_significant=bool(sigD),
                          contradiction=contra, source=r.get("source", "weak_window"),
                          rates_used="free fit" if rates else "none (rates = 0)"))
    out["window_audit"] = audit
    contra = [a for a in audit if a["contradiction"]]
    good = [r for r, a in zip(rows_f, audit[:len(rows_f)]) if not a["contradiction"]]
    out["n_contradicting_windows"] = len(contra)
    if not good:
        st = "SCALE_CONTRADICTED" if contra else "ABSOLUTE_SCALE_NOT_OBSERVABLE"
        out.update(status=st, reason=("every forced window contradicts D/J > 0 (no positive mass explains it)"
                                      if contra else "no trusted forced window with motion"))
        return out
    names = fs.names
    rows = rows_free + good
    x, chi2, A, y, sd = solve(rows, names)
    w = 1.0 / sd
    dof = max(len(rows) - len(names), 1)
    C, Cs, sc = _cov(A, w, chi2, dof)
    ib = names.index("beta")
    prof = profile(rows, names, ib, x, chi2, scale=sc)
    beta = float(x[ib])
    sdb = float(math.sqrt(max(Cs[ib, ib], 0.0)))
    out.update(estimate=dict(zip(names, x.tolist())), sd=dict(zip(names, np.sqrt(np.clip(np.diag(Cs), 0, None))
                                                              .tolist())),
               cov=Cs.tolist(), chi2_per_dof=chi2 / dof, profile_beta=prof, svd=_svd(A, w, names),
               units={n: u[n] for n in names})
    ci = prof["ci95"]
    if beta <= 0 or ci[0] is None or ci[0] <= 0:
        st = "ABSOLUTE_SCALE_NOT_OBSERVABLE"
        reason = "beta interval reaches 0 (the forced data do not pin 1/M)"
    elif ci[1] is None or prof["ci_at_grid_edge"] or (ci[1] - ci[0]) / 2 > REL_OK * beta:
        st = "ABSOLUTE_SCALE_WEAK"
        reason = "beta interval too wide"
    else:
        st = "FORCE_IDENTIFIED_CANDIDATE"
        reason = "beta interval excludes 0 and is narrow (still subject to the replay / kinematics gates)"
    if contra:
        reason += f"; {len(contra)} contradicting window(s) excluded and reported"
    out.update(status=st, reason=reason)
    if beta > 0:
        M = 1.0 / beta
        M_sd = sdb / beta ** 2
        out["dimensional"] = dict(M=dict(value=M, sd=M_sd, unit=u["M"], ci95=None if ci[0] is None or ci[0] <= 0
                                         else [1.0 / ci[1], 1.0 / ci[0]]),
                                  c=dict(value=x[names.index("lam")] * M, unit=u["c"]) if "lam" in names else None,
                                  Fc=dict(value=x[names.index("gam")] * M, unit=u["Fc"]) if "gam" in names else None,
                                  provenance="FORCE_IDENTIFIED" if st == "FORCE_IDENTIFIED_CANDIDATE" else
                                  "NOT_IDENTIFIED (estimate_not_for_sim)")
    out["quasi_static_view"] = _dimensional(rows_f, fs)
    return out


def _dimensional(rows_f, fs):
    """J = M (dv + int a_g) + c dq + Fc s dt on the forced rows: absolute dissipation can be force-identified in
    quasi-static pushes even when M is not (dv ~ 0)."""
    if len(rows_f) < 3:
        return dict(status="NOT_RUN", reason=f"{len(rows_f)} forced windows")
    A = np.array([[r["dv"] + r["int_ag"], r["dq"], r["s_dt"]] for r in rows_f])
    y = np.array([r["J"] for r in rows_f])
    s = np.array([max(r["sd_J"], 1e-12) for r in rows_f])
    w = 1 / s
    x, chi2 = _lsq(A, y, w)
    nm = ["M", "c", "Fc"]
    dof = max(len(y) - 3, 1)
    C, Cs, sc = _cov(A, w, chi2, dof)
    u = fs.units
    return dict(estimate=dict(zip(nm, x.tolist())), sd=dict(zip(nm, np.sqrt(np.clip(np.diag(Cs), 0, None)).tolist())),
                units={n: u[n] for n in nm}, svd=_svd(A, w, nm), chi2_per_dof=chi2 / dof,
                note="row weight from the effort integral only; M at 0 is a FAILED identification (never massless)")


def impulse_window(rec, nspec, rows, *, W=0.3, gap=0.04):
    """A short forced pulse (too few q samples for weak windows): one bracket row from one-sided velocities
    (no smoothing across the pulse) -- reuses the impulse-balance machinery."""
    from ..validation.impulse_balance import momentum_case
    m = (rec.source_force_row >= rows[0]) & (rec.source_force_row <= rows[1])
    if not m.any():
        return None
    ta, tb = float(rec.force_t_s[m][0]), float(rec.force_t_s[m][-1]) + 0.02
    mc = momentum_case(rec.force_t_s, rec.Q_object, rec.Q_sigma, rec.q_t_s, rec.q, rec.q_sigma, ta, tb, W=W, gap=gap)
    if mc.get("status") == "VELOCITY_UNAVAILABLE":
        return None
    qa, qb = np.interp([ta, tb], rec.q_t_s, rec.q)
    sub = np.linspace(ta, tb, 50)
    return dict(seg=f"pulse_{rows[0]}-{rows[1]}", label="FORCED_MEASURED", free=False, ta=ta, tb=tb, dv=mc["dv"],
                sd_dv=mc["sigma_dv"], dq=float(qb - qa), s_dt=float(mc["int_sgn_qdot"]),
                int_ag=float(np.trapz(nspec.gravity(np.interp(sub, rec.q_t_s, rec.q)), sub)), J=mc["J"],
                sd_J=mc["sigma_J"], source="impulse_bracket (one-sided velocities)", momentum_case=mc)


# =================================================================================================== diagnostics
def effective_load_fit(segments, nspec, joint_type):
    """gravity_effective_diagnostic: rates + a SIGNED normalized configuration load (revolute A cos q + B sin q,
    prismatic a constant) fitted on top of the declared a_g. Labelled EFFECTIVE (not verified gravity): its rank /
    correlation quantify how far friction and an unknown load are confounded (for one motion direction a constant
    load and gam s are exactly collinear)."""
    rows = [r for r in all_rows([s for s in segments if s.is_free], nspec.free())]
    names = nspec.free().names
    extra = ["eA", "eB"] if joint_type == "revolute" else ["e0"]
    if len(rows) < len(names) + len(extra) + 1:
        return dict(status="NOT_RUN", reason=f"{len(rows)} windows")
    A0, y, sd = _design(rows, names)
    E = np.array([[r["int_cos"], r["int_sin"]] if joint_type == "revolute" else [r["dt"]] for r in rows])
    A = np.hstack([A0, E])
    w = 1.0 / sd
    lo = np.r_[np.zeros(len(names)), np.full(len(extra), -np.inf)]
    r = lsq_linear(A * w[:, None], y * w, bounds=(lo, np.inf), method="bvls")
    nm = names + extra
    chi2 = float(np.sum((A * w[:, None] @ r.x - y * w) ** 2))
    dof = max(len(rows) - len(nm), 1)
    C, Cs, sc = _cov(A, w, chi2, dof)
    d = np.sqrt(np.clip(np.diag(Cs), 1e-300, None))
    return dict(label="EFFECTIVE configuration load (NOT verified gravity; may absorb friction / mechanism)",
                names=nm, estimate=dict(zip(nm, r.x.tolist())), sd=dict(zip(nm, d.tolist())),
                correlation=(Cs / np.outer(d, d)).tolist(), svd=_svd(A, w, nm), chi2_per_dof=chi2 / dof,
                form="a_eff(q) = eA cos q + eB sin q" if joint_type == "revolute" else "a_eff = e0",
                note="added to the declared a_g; collinearity with gam s (one direction) is reported, not resolved")


def gravity_negligible_test(segments, nspec_known, factor=0.2):
    """Is the known-shape normalized gravity below the acceleration resolution of the free data?"""
    rows = all_rows([s for s in segments if s.is_free], nspec_known.free())
    if not rows:
        return dict(status="NOT_EVALUATED", reason="no free windows")
    res = float(np.median([r["sd_dv"] / max(r["dt"], 1e-9) for r in rows]))
    qs = np.concatenate([s.q for s in segments if s.is_free])
    qq = np.linspace(qs.min(), qs.max(), 50)
    amax = float(np.max(np.abs(nspec_known.gravity(qq))))
    var = float(np.ptp(nspec_known.gravity(qq)))
    ok = amax < factor * res
    return dict(status="NEGLIGIBLE_SUPPORTED" if ok else "NOT_NEGLIGIBLE", max_abs_a_g=amax,
                a_g_variation_over_q_range=var, acceleration_resolution=res, factor=factor,
                rule=f"max |a_g| over the observed q range < {factor} x the median window acceleration sd",
                unit=nspec_known.units["a"])


def passivity_split(seg, nspec, k=3.0, min_q=5):
    """Free passive motion cannot gain normalized mechanical energy E = v^2/2 + U_g(q), U_g = int a_g dq
    (dE/dt = -lam v^2 - gam |v| <= 0). A rise of E by > k sigma proves an UNMEASURED input (or a wrong gravity
    model): the part of the segment up to the energy maximum is excluded from free identification and reported."""
    from scipy.integrate import cumulative_trapezoid
    tg, d = _smooth(seg)
    v, vsd, q = d["qd"], d["qd_sd"], d["q"]
    U = cumulative_trapezoid(nspec.gravity(q) * v, tg, initial=0.0)
    E = 0.5 * v ** 2 + U
    sdE = np.abs(v) * vsd
    i0 = 0
    imax = int(np.argmax(E))
    gain = float(E[imax] - E[i0])
    sd = float(math.hypot(sdE[i0], sdE[imax]))
    audit = dict(seg=seg.name, gravity=nspec.gravity.name, energy_gain=gain, sd=sd,
                 gain_over_sd=gain / sd if sd > 0 else None, t_energy_max=float(tg[imax]),
                 unit=f"({nspec.units['v']})^2", rule=f"E = v^2/2 + int a_g dq must not rise by > {k} sd in free motion")
    if gain > k * sd and imax > 0:
        kq = int(np.searchsorted(seg.t, tg[imax] - 1e-9))
        audit.update(verdict="ENERGY_GAIN_WITHOUT_MEASURED_INPUT",
                     excluded=[float(seg.t[0]), float(tg[imax])], n_q_excluded=int(kq))
        if len(seg.t) - kq < min_q:
            audit["kept"] = None
            return None, audit
        from ..free_forced.segments import MotionSegment
        kept = MotionSegment(seg.name + "_passive", seg.label, seg.joint_type, seg.t[kq:], seg.q[kq:],
                             seg.q_sigma[kq:], seg.sign, seg.force_t, seg.Q, seg.Q_sigma,
                             dict(seg.evidence, passivity=f"trimmed: energy gain {gain:.3g} ({gain / sd:.1f} sd) "
                                                          "before the energy maximum"))
        audit["kept"] = [float(kept.t[0]), float(kept.t[-1])]
        return kept, audit
    audit.update(verdict="PASSIVE_CONSISTENT", kept=[float(seg.t[0]), float(seg.t[-1])])
    return seg, audit
