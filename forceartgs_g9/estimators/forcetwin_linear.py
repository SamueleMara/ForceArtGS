"""ForceTwin-inspired constrained linear identification of EFFECTIVE joint dynamics (Engelbracht et al. 2026,
arXiv:2609.21751, section III-C, Eqs. (5), (7)-(9): physical linear component z^T beta solved by non-negative least
squares; adapted here -- the paper's network / optimiser are not reproduced).

Moving regime, parameter order beta = [I_eff, mu, b] (paper: [mu, b, I], equivalent):

    tau_i - tau_known_grav(q_i) = I_eff qdd_i + mu sgn(qdot_i) + b qdot_i + r_i           (direct form)
    int_a^b tau dt - int tau_grav dt = I [qdot(b) - qdot(a)] + mu int sgn(qdot) dt + b [q(b) - q(a)]   (weak form)

Extensions (ours, not the paper's): native 50 Hz force rows with q on the RGB-D clock (two independent derivative
reconstructions), an exact ZOH weak form on non-overlapping windows, block-bootstrap intervals, explicit rank /
identifiable combinations, coast-only ratios, and a known-gravity variant tied to the SAME mass geometry as inertia
(tau_grav = I_eff kappa(q), kappa = G_1 / I_axis,1; never an independently fitted gravitational mass).
I_eff = 0 is an optimisation boundary (failed identification), never a massless part.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import lsq_linear

from ..forcetwin.derivatives import local_poly, spline
from .torch_model import zoh_integral

PARAMS = ("I_eff", "mu", "b")
VARIANTS = {"inertia_only": ("I_eff",), "inertia_viscous": ("I_eff", "b"), "inertia_coulomb": ("I_eff", "mu"),
            "inertia_viscous_coulomb": ("I_eff", "mu", "b")}
REL_OK = 0.5            # 95 % half-width / estimate for ESTIMATED_CONDITIONAL
RANK_TOL = 1e-4         # singular value / largest (column-normalised, whitened) below this = not informative
COND_MAX = 1e3          # condition number above this: separate coefficients are UNOBSERVABLE
WRMS_MAX = 5.0          # whitened effort RMS above this = MODEL_INADEQUATE (derivative bias at force steps
                        # inflates whitened residuals to ~4 even with the exact generating model)


def units(jt):
    return dict(I_eff="kg m^2" if jt == "revolute" else "kg", mu="N m" if jt == "revolute" else "N",
                b="N m s/rad" if jt == "revolute" else "N s/m")


class GravityHypothesis:
    """tau_grav(q) = I_eff kappa(q). ``kappa`` None: zero gravity (e.g. the vertical-hinge HYPOTHESIS)."""

    def __init__(self, name, kappa=None, provenance=""):
        self.name, self.kappa, self.provenance = name, kappa, provenance

    def k(self, q):
        return np.zeros_like(np.asarray(q, float)) if self.kappa is None else self.kappa(np.asarray(q, float))

    @staticmethod
    def from_mass_model(mm, g_world, name):
        gu = mm.gravity_unit(g_world)
        I1 = mm.M1
        if mm.reflected() != 0.0:
            raise ValueError("known-gravity variant needs zero reflected inertia (tau_grav = I kappa)")
        return GravityHypothesis(name, lambda q: np.asarray(gu.G(q), float) / I1,
                                 f"G_1(q)/I_axis,1 of the {mm.props.mode} mass geometry (same density for inertia "
                                 f"and gravity; q_ref {mm.q_ref})")


# =================================================================================================== rows
def _windows(rec):
    for name, fs, qs, src in rec.windows:
        yield name, rec.force_t_s[fs], rec.Q_object[fs], rec.Q_sigma[fs], rec.contact_mode[fs], \
            rec.q_t_s[qs], rec.q[qs], rec.q_sigma[qs], src


def direct_rows(rec, smoother="spline", v_k=3.0, grav=None):
    """Rows z = [qdd + kappa(q), sgn(qd), qd] at native force rows inside moving, sliding / force-absent intervals."""
    Z, y, s_tau, meta = [], [], [], []
    for name, tf, Q, Qs, mode, tq, q, qs, src in _windows(rec):
        m = (tf > tq[0]) & (tf < tq[-1])
        d = (spline if smoother == "spline" else local_poly)(tq, q, qs, tf[m])
        ok = np.isfinite(d["qd"]) & np.isfinite(d["qdd"])
        ok &= np.abs(d["qd"]) > v_k * np.maximum(d["qd_sd"], 1e-6)                # clearly moving (sign known)
        ok &= ~np.isin(mode[m], ("sticking", "stick", "uncertain"))
        kap = grav.k(d["q"]) if grav is not None else np.zeros(m.sum())
        idx = np.flatnonzero(ok)
        for i in idx:
            Z.append([d["qdd"][i] + kap[i], np.sign(d["qd"][i]), d["qd"][i]])
            y.append(Q[m][i])
            s_tau.append(Qs[m][i])
            meta.append((name, float(tf[m][i]), float(d["qd_sd"][i]), float(d["qdd_sd"][i])))
    return np.asarray(Z, float).reshape(-1, 3), np.asarray(y), np.asarray(s_tau), meta


def weak_rows(rec, L=0.4, smoother="spline", grav=None, v_k=3.0, n_sub=200):
    """Non-overlapping windows of length L inside moving runs: exact ZOH force integral on the native rows."""
    from ..synthetic import ZOH
    Z, y, s_y, meta = [], [], [], []
    for name, tf, Q, Qs, mode, tq, q, qs, src in _windows(rec):
        eff = ZOH(tf, Q)
        tg = np.linspace(tq[0], tq[-1], max(200, int((tq[-1] - tq[0]) * 200)))
        d = (spline if smoother == "spline" else local_poly)(tq, q, qs, tg)
        mov = np.abs(d["qd"]) > v_k * np.maximum(d["qd_sd"], 1e-6)
        mov &= ~np.isin(mode[np.clip(np.searchsorted(tf, tg, side="right") - 1, 0, len(tf) - 1)],
                        ("sticking", "stick", "uncertain"))
        # maximal moving runs, cut into non-overlapping windows of length L
        runs, st = [], None
        for i, mv in enumerate(list(mov) + [False]):
            if mv and st is None:
                st = i
            if not mv and st is not None:
                runs.append((st, i - 1))
                st = None
        for a0, b0 in runs:
            ta = tg[a0]
            while ta + L <= tg[b0] + 1e-12:
                tb = ta + L
                ia, ib = np.searchsorted(tg, ta), min(np.searchsorted(tg, tb), len(tg) - 1)
                sub = np.linspace(ta, tb, n_sub)
                qd_sub = np.interp(sub, tg, d["qd"])
                q_sub = np.interp(sub, tg, d["q"])
                kap = grav.k(q_sub) if grav is not None else np.zeros(n_sub)
                J = zoh_integral(eff, ta, tb)
                fr = (tf >= ta) & (tf < tb)
                sJ = float(np.sqrt(np.sum((Qs[fr] * 0.02) ** 2))) if fr.any() else float(np.median(Qs) * L)
                Z.append([d["qd"][ib] - d["qd"][ia] + np.trapz(kap, sub), np.trapz(np.sign(qd_sub), sub),
                          d["q"][ib] - d["q"][ia]])
                y.append(J)
                s_y.append(sJ)
                meta.append((name, float(ta), float(tb), float(math.hypot(d["qd_sd"][ia], d["qd_sd"][ib]))))
                ta = tb
    return np.asarray(Z, float).reshape(-1, 3), np.asarray(y), np.asarray(s_y), meta


# =================================================================================================== solve
def _solve(Z, y, w, cols):
    A = Z[:, cols] * w[:, None]
    r = lsq_linear(A, y * w, bounds=(0.0, np.inf), method="bvls")
    ols = np.linalg.lstsq(A, y * w, rcond=None)[0] if len(y) >= len(cols) else np.full(len(cols), np.nan)
    return r.x, ols


def _block_bootstrap(Z, y, w, cols, block, B=200, seed=0):
    n = len(y)
    if n < 2 * block or n < len(cols) + 2:
        return None
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    out = []
    for _ in range(B):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        x, _ = _solve(Z[idx], y[idx], w[idx], cols)
        out.append(x)
    return np.asarray(out)


def fit(rec, *, form="direct", variant="inertia_viscous_coulomb", smoother="spline", L=0.4, grav=None,
        prior=None, B=200, seed=0) -> dict:
    """One constrained (bounded >= 0) weighted LS fit of one nested variant."""
    t0 = time.time()
    names = VARIANTS[variant]
    cols = [PARAMS.index(n) for n in names]
    Z, y, s, meta = (direct_rows(rec, smoother, grav=grav) if form == "direct" else
                     weak_rows(rec, L=L, smoother=smoother, grav=grav))
    base = dict(form=form, variant=variant, smoother=smoother, window_s=L if form == "weak" else None,
                gravity_hypothesis=None if grav is None else grav.name, n_rows=int(len(y)), parameters=names)
    if len(y) < len(cols) + 2:
        return dict(base, status="UNOBSERVABLE", reason=f"only {len(y)} usable rows")
    # heteroscedastic weights: effort noise + derivative noise propagated with a first-pass estimate
    w = 1.0 / np.maximum(s, 1e-12)
    x, _ = _solve(Z, y, w, cols)
    I0 = x[0]
    der = np.array([m[3] if form == "direct" else m[3] for m in meta])
    s2 = np.sqrt(s ** 2 + (I0 * der) ** 2)
    w = 1.0 / np.maximum(s2, 1e-12)
    x, ols = _solve(Z, y, w, cols)
    A = Z[:, cols] * w[:, None]
    sv = np.linalg.svd(A / np.maximum(np.linalg.norm(A, axis=0), 1e-300), compute_uv=False)
    rank = int(np.sum(sv > sv[0] * RANK_TOL)) if len(sv) else 0          # numerical rank of the whitened Z
    cond = float(sv[0] / sv[-1]) if len(sv) and sv[-1] > 0 else math.inf
    res = (y - Z[:, cols] @ x) * w
    dof = max(len(y) - len(cols), 1)
    whitened_rms = float(np.sqrt(np.mean(res ** 2)))
    lag1 = (float(np.corrcoef(res[:-1], res[1:])[0, 1])
            if len(res) > 3 and np.std(res[:-1]) > 0 and np.std(res[1:]) > 0 else None)
    block = 13 if form == "direct" else 2            # ~ the smoother correlation length (rows) / windows
    bs = _block_bootstrap(Z, y, w, cols, block, B=B, seed=seed)
    cov = None
    try:
        cov = np.linalg.inv(A.T @ A) * (whitened_rms ** 2)
    except np.linalg.LinAlgError:
        pass
    corr = None
    if cov is not None and np.all(np.diag(cov) > 0):
        d = np.sqrt(np.diag(cov))
        corr = (cov / np.outer(d, d)).tolist()
    est = {}
    for j, n in enumerate(names):
        ci = None if bs is None else [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
        occ = None if bs is None else float(np.mean(bs[:, j] <= 1e-12))
        est[n] = dict(estimate=float(x[j]), unconstrained_ols=float(ols[j]), interval95_statistical=ci,
                      bound_occupancy=occ, at_physical_bound=bool(x[j] <= 1e-12),
                      positive_confidence=None if bs is None else float(np.mean(bs[:, j] > 1e-12)))
    return dict(base, status="OK", estimates=est, rank=rank, n_parameters=len(cols),
                singular_values_whitened_normalised=sv.tolist(), condition_number=cond, correlation=corr,
                whitened_residual_rms=whitened_rms, residual_lag1_autocorr=lag1, dof=dof,
                weighted_rss=float(np.sum(res ** 2)), block_bootstrap=dict(B=B, block=block, seed=seed),
                errors_in_variables="qdot / qdd are reconstructed (not exact): the weights include their propagated "
                                    "variance, but LS remains attenuation-biased (errors-in-variables) -- limitation",
                runtime_s=time.time() - t0)


def coast_ratios(rec, smoother="spline", v_k=3.0):
    """Force-absent moving rows: qdd = -(b/I) qd - (mu/I) sgn(qd). Ratios only (scale unobservable)."""
    rows, ys = [], []
    for name, tf, Q, Qs, mode, tq, q, qs, src in _windows(rec):
        m = (tf > tq[0]) & (tf < tq[-1]) & (mode == "absent")
        if m.sum() < 8:
            continue
        d = (spline if smoother == "spline" else local_poly)(tq, q, qs, tf[m])
        ok = np.abs(d["qd"]) > v_k * np.maximum(d["qd_sd"], 1e-6)
        rows += [[np.sign(a), a] for a in d["qd"][ok]]
        ys += list(-d["qdd"][ok])
    if len(ys) < 6:
        return dict(status="NOT_RUN", reason=f"{len(ys)} force-absent moving rows (need >= 6)")
    A, yy = np.asarray(rows), np.asarray(ys)
    r = lsq_linear(A, yy, bounds=(0, np.inf))
    return dict(status="RATIO_ONLY", mu_over_I=float(r.x[0]), b_over_I=float(r.x[1]), n_rows=int(len(ys)),
                note="unforced coasting identifies only ratios: (I, b, mu) -> k (I, b, mu) is indistinguishable")


# =================================================================================================== replay
def spec_for(jt, variant, grav_mm=None, gravity=None):
    """The minimal effective ModelSpec of a variant (joint_space_only; gravity via the mass model if known)."""
    from ..dynamics.general_1dof import GeneralTerms
    from .model_spec import ModelSpec
    names = VARIANTS[variant]
    t = GeneralTerms(inertia=True, viscous="b" in names, coulomb="mu" in names, static=False,
                     gravity=grav_mm is not None, bias=False)
    if grav_mm is not None:
        return ModelSpec(t, jt, "geometry_coupled", grav_mm, gravity, allow_unverified_gravity=True)
    return ModelSpec(t, jt, "joint_space_only")


def theta_for(spec, est, grav_mm=None):
    th = []
    for n in spec.names:
        if n == "alpha":
            th.append(est["I_eff"] / grav_mm.M1)
        elif n == "M":
            th.append(est["I_eff"])
        elif n == "c":
            th.append(est.get("b", 0.0))
        elif n == "Fc":
            th.append(est.get("mu", 0.0))
    return np.asarray(th, float)


def replay(rec, spec, theta):
    from .general_ekf import _replay
    try:
        rr = _replay(spec, theta, rec.interactions())["per_interaction"]
        v = [x["rmse_over_sigma"] for x in rr if "rmse_over_sigma" in x]
        return dict(per_interaction=rr, mean_rmse_over_sigma=float(np.mean(v)) if v else None,
                    protocol="frozen theta, q0 = first observation, v0 = line fit of the first 4 observations")
    except Exception as e:                                   # e.g. I_eff = 0 is not simulable
        return dict(status="NOT_SIMULABLE", reason=f"{type(e).__name__}: {e}")


# =================================================================================================== study
REPLAY_MAX = 10.0       # frozen-theta open-loop replay rmse/sigma above this: the model does not describe the data


def gate(param, estimate, sd, *, uncertainty_label="", estimator_status="ok", I_prior=None, impulse_state=None,
         nis=None, replay_rmse=None, kin_ok=True, kin_status=None, rel_max=0.25):
    """Identifiability gate for a general estimator's parameter (identifiability -- not convergence -- decides).
    Returns (status, reasons). A seed-ensemble spread is NEVER accepted as an uncertainty, and a method whose
    frozen-parameter replay fails (> REPLAY_MAX sigma, or not simulable) is MODEL_INADEQUATE."""
    status, why = "ESTIMATED_CONDITIONAL", []
    if str(estimator_status).lower() != "ok":
        status, why = "NOT_IDENTIFIED", [f"estimator status {estimator_status}"]
    elif estimate <= 1e-9:
        status, why = "NOT_IDENTIFIED", ["at the physical bound"]
    elif param == "I_eff" and I_prior and not (I_prior[0] / 5 <= estimate <= 5 * I_prior[1]):
        status, why = "MODEL_INADEQUATE", [f"outside 1/5..5x the size prior {list(I_prior)}"]
    elif "ENSEMBLE" in str(uncertainty_label).upper():
        status, why = "NOT_IDENTIFIED", ["only a seed-ensemble spread is available: seed agreement is not an "
                                         "uncertainty (no posterior / covariance)"]
    elif sd is None or not np.isfinite(sd) or sd <= 0:
        status, why = "NOT_IDENTIFIED", [f"degenerate uncertainty ({sd}): no usable covariance for this parameter"]
    elif sd > rel_max * estimate:
        status, why = "NOT_IDENTIFIED", [f"own uncertainty {sd:.3g} not < {rel_max:.0%} of the estimate"]
    if param == "I_eff" and impulse_state in INCONSISTENT_IMPULSE:
        status = "DATA_INCONSISTENT" if status == "ESTIMATED_CONDITIONAL" else status
        why.append(f"impulse audit {impulse_state}")
    if nis is not None and nis.get("consistent") is False:
        why.append(f"EKF innovations inconsistent (mean NIS {nis.get('mean'):.3g} > {nis.get('mean_upper95'):.3g})")
        if status == "ESTIMATED_CONDITIONAL":
            status = "MODEL_INADEQUATE"
    if replay_rmse is None or not np.isfinite(replay_rmse) or replay_rmse > REPLAY_MAX:
        why.append(f"frozen-parameter replay {replay_rmse if replay_rmse is None else round(replay_rmse, 3)} "
                   f"rmse/sigma (> {REPLAY_MAX} or not simulable): the model does not describe the data")
        if status == "ESTIMATED_CONDITIONAL":
            status = "MODEL_INADEQUATE"
    if not kin_ok:
        why.append("kinematics operator-declared, not independently accepted (OPERATOR_DECLARED)"
                   if kin_status == "OPERATOR_DECLARED" else f"KINEMATICS_NOT_ACCEPTED ({kin_status})")
        if status == "ESTIMATED_CONDITIONAL":
            status = "CONDITIONAL_ON_UNACCEPTED_KINEMATICS"
    return status, why


INCONSISTENT_IMPULSE = ("IMPULSE_SIGN_OR_TIMING_INCONCLUSIVE", "NO_POSITIVE_INERTIA_EXPLANATION",
                        "CONTACT_EFFORT_UNCERTAIN", "MOTION_DERIVATIVE_UNCERTAIN",
                        "UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE")


def identifiability(fitres, jt, *, impulse_state=None, sweep=None, replay_res=None, data_window=None,
                    method="forcetwin_nnls", I_prior=None):
    """Per-parameter record (spec section 4.2). Identifiability -- not convergence -- is the gate."""
    out = {}
    u = units(jt)
    for n, e in fitres["estimates"].items():
        ci = e["interval95_statistical"]
        status, why = "ESTIMATED_CONDITIONAL", []
        if fitres["rank"] < fitres["n_parameters"] or fitres["condition_number"] > COND_MAX:
            status, why = "UNOBSERVABLE", [f"rank {fitres['rank']}/{fitres['n_parameters']}, condition "
                                           f"{fitres['condition_number']:.3g}: coefficients not separable"]
        elif e["at_physical_bound"] or (e["bound_occupancy"] or 0) > 0.05:
            status, why = "NOT_IDENTIFIED", [f"at / near the physical bound (occupancy {e['bound_occupancy']})"]
        elif ci is None:
            status, why = "NOT_IDENTIFIED", ["no interval (too few independent rows)"]
        elif ci[0] <= 0 or (ci[1] - ci[0]) / 2 > REL_OK * e["estimate"]:
            status, why = "NOT_IDENTIFIED", [f"95 % interval {ci} not resolved away from 0 (half-width > "
                                             f"{REL_OK:.0%})"]
        if n == "I_eff" and impulse_state in INCONSISTENT_IMPULSE:
            status = "DATA_INCONSISTENT" if status == "ESTIMATED_CONDITIONAL" else status
            why.append(f"signed impulse audit: {impulse_state}")
        plaus = None
        if n == "I_eff" and I_prior:
            lo_, hi_ = I_prior
            plaus = dict(prior_range=[lo_, hi_], ratio_to_prior_min=e["estimate"] / lo_ if lo_ else None,
                         plausible=bool(lo_ / 5 <= e["estimate"] <= 5 * hi_))
            if not plaus["plausible"]:
                why.append(f"estimate {e['estimate']:.3g} outside 1/5..5x the size prior {I_prior}")
                if status == "ESTIMATED_CONDITIONAL":
                    status = "MODEL_INADEQUATE"
        rr = None if replay_res is None else replay_res.get("mean_rmse_over_sigma")
        if replay_res is not None and (rr is None or rr > REPLAY_MAX):
            why.append(f"frozen-parameter replay {rr} rmse/sigma (> {REPLAY_MAX} or not simulable)")
            if status == "ESTIMATED_CONDITIONAL":
                status = "MODEL_INADEQUATE"
        if fitres["whitened_residual_rms"] > WRMS_MAX and status == "ESTIMATED_CONDITIONAL":
            status = "MODEL_INADEQUATE"
            why.append(f"whitened residual RMS {fitres['whitened_residual_rms']:.2f} > {WRMS_MAX}")
        sens = None
        if sweep:
            vals = [s["estimates"][n]["estimate"] for s in sweep if s.get("status") == "OK" and n in s["estimates"]]
            if vals:
                sens = dict(min=float(min(vals)), max=float(max(vals)), n=len(vals),
                            over="smoother x form x window length")
        out[n] = dict(estimate=e["estimate"], unit=u[n], method=method, model_variant=fitres["variant"],
                      data_window=data_window, point_or_ratio="point", at_physical_bound=e["at_physical_bound"],
                      interval95_statistical=ci, uncertainty_sensitivity=sens, identifiability_status=status,
                      reasons=why, model_adequacy=dict(whitened_residual_rms=fitres["whitened_residual_rms"],
                                                       residual_lag1_autocorr=fitres["residual_lag1_autocorr"]),
                      open_loop_replay=None if replay_res is None else replay_res.get("mean_rmse_over_sigma"),
                      provenance=dict(form=fitres["form"], smoother=fitres["smoother"],
                                      gravity_hypothesis=fitres["gravity_hypothesis"], n_rows=fitres["n_rows"]),
                      size_prior_check=plaus, production=False, production_reason="no independent held-out episode")
    return out


def study(rec, *, grav=None, grav_mm=None, gravity=None, impulse_state=None, B=200, I_prior=None) -> dict:
    """Nested variants x {direct, weak} x {spline, local_poly} x window lengths; primary = direct / spline."""
    t0 = time.time()
    variants, sweep_all = {}, []
    for v in VARIANTS:
        runs = []
        for form in ("direct", "weak"):
            for sm in ("spline", "local_poly"):
                for L in ((0.4,) if form == "direct" else (0.2, 0.4, 0.8)):
                    runs.append(fit(rec, form=form, variant=v, smoother=sm, L=L, grav=grav, B=B))
        prim = runs[0]
        weak = next((r for r in runs if r["form"] == "weak" and r["smoother"] == "spline" and r.get("status") == "OK"),
                    next((r for r in runs if r["form"] == "weak"), None))
        rp = None
        if prim.get("status") == "OK":
            sp = spec_for(rec.joint_type, v, grav_mm, gravity)
            rp = replay(rec, sp, theta_for(sp, {k: e["estimate"] for k, e in prim["estimates"].items()}, grav_mm))
        ident = identifiability(prim, rec.joint_type, impulse_state=impulse_state, sweep=runs, replay_res=rp,
                                data_window=[w[3] for w in rec.windows], I_prior=I_prior) \
            if prim.get("status") == "OK" else None
        variants[v] = dict(primary=prim, weak_primary=weak, sweep=runs, replay=rp, identifiability=ident)
        sweep_all += runs
    ok = {k: v for k, v in variants.items() if v["primary"].get("status") == "OK"}
    sel = None
    if ok:     # DIAGNOSTIC model ranking by BIC of the whitened residual (never a physical-absence claim)
        bic = {k: v["primary"]["weighted_rss"] + len(VARIANTS[k]) * math.log(max(v["primary"]["n_rows"], 2))
               for k, v in ok.items()}
        sel = dict(bic=bic, lowest=min(bic, key=bic.get),
                   label="DIAGNOSTIC (single recording, correlated rows): not a selection of physical terms")
    return dict(schema="forceartgs.g9.forcetwin_linear/v1", joint_type=rec.joint_type,
                gravity_hypothesis=None if grav is None else dict(name=grav.name, provenance=grav.provenance),
                parameter_order=list(PARAMS), paper_order="[mu, b, I] (ForceTwin Eq. 5); equivalent",
                variants=variants, model_comparison=sel, coast=coast_ratios(rec), runtime_s=time.time() - t0)
