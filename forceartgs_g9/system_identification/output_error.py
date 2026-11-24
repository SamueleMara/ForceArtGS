"""Stage A, primary estimator: continuous-time physics-parameterized OUTPUT-ERROR fit against position samples (Eq. 11)

    psi_hat = argmin sum_e sum_i rho_delta((q_model(t_ei; u_e, psi, q0_e, v0_e) - q_ei) / sigma_ei)
              + R_IC + R_phys

q_model is the NORMALIZED hybrid 1-DOF model  qdd + lam qd + gam sgn(qd) + a_g(q) [+ kappa (q - q_eq)] = g_u u(t),
rolled out by the validated hybrid integrator ``simulate_general`` (exact linear steps on every constant-effort
forward-ZOH interval -- Eq. (8) -- with the zero-velocity stop terminating the sliding segment; Heun steps only for a
configuration load). ``g_u`` (= beta = 1/M) is a parameter ONLY when an informative forced window is in the training
set; a free-only fit never contains it (the absolute scale is structurally absent, not a bound at 0).

R_IC = sum_e [(q0_e - q0_hat)^2 / s_q0^2 + (v0_e - v0_hat)^2 / s_v0^2] -- the per-episode nuisance initial state is
bounded by the observed-prefix estimate (same trajectory, NOT independent); it cannot absorb an input pulse freely.
R_phys: hard bounds lam, gam >= 0, g_u > 0 (log-parameterized), no negative mass, no sign optimisation.

Variants (nested, same windows): L0 inertial+viscous, L1 +Coulomb, L3 +configuration-dependent load (only when listed
AND coverage supports it; reported as a candidate load, never "a spring"). L2 (static breakaway) needs confirmed
start events; L4 (residual) is research-only -- both are reported NOT_RUN with the reason.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import least_squares

from .contracts import MODEL_VARIANTS, UNITS

DCHI2_95_HALF = 1.92            # 0.5 * chi2_1(0.95): profile threshold on the cost (cost = 0.5 * sum r^2)


def _huberize(z, delta):
    """r with 0.5 r^2 = rho_huber(z): smooth (C1) square-root transform so least_squares minimizes sum rho(z)."""
    if delta is None:
        return z
    a = np.abs(z)
    return np.where(a <= delta, z, np.sign(z) * np.sqrt(np.maximum(2 * delta * a - delta * delta, 0.0)))


def a_scale(episodes):
    v = [max(float(np.ptp(e.q)), 1e-9) / max(float(e.t[-1] - e.t0), 1e-6) ** 2 for e in episodes]
    return float(np.median(v)) if v else 1.0


class OEModel:
    """One model variant over a fixed list of IOEpisodes (training ones enter the cost; held-out ones are replayed
    with theta frozen from their prior initial state)."""

    def __init__(self, episodes, variant="L1", gravity=None, cfg=None, *, dimensional_M=None):
        from ..free_forced.normalized import GravityModel
        cfg = dict(cfg or {})
        self.episodes = list(episodes)
        self.train = [e for e in self.episodes if e.train]
        if not self.train:
            raise ValueError("no training episode")
        self.jt = self.train[0].joint_type
        if any(e.joint_type != self.jt for e in self.episodes):
            raise ValueError("mixed joint types")
        self.variant = variant
        self.coulomb = variant in ("L1", "L3")
        self.spring = variant == "L3"
        if variant not in ("L0", "L1", "L3"):
            raise ValueError(f"variant {variant} is not fitted by the output-error estimator")
        self.gravity = gravity if gravity is not None else GravityModel.zero(self.jt, provenance="no gravity load")
        self.forced = any(e.forced for e in self.train)
        self.dimensional_M = dimensional_M              # fixed absolute M (scale profile only)
        self.delta = float(cfg.get("huber_delta", 2.0)) if cfg.get("robust_loss", "huber") == "huber" else None
        self.floor = float(cfg.get("sigma_floor_q", 0.0))
        A = a_scale(self.train)
        Ms = cfg.get("M_search", [1e-4, 1e4])
        lam_max = float(cfg.get("lam_max", 50.0))
        gam_max = float(cfg.get("gam_max_factor", 50.0)) * A
        names, lo, hi = ["lam"], [0.0], [lam_max]
        if self.dimensional_M is not None:
            names, lo, hi = ["c"], [0.0], [lam_max * self.dimensional_M]
            if self.coulomb:
                names.append("Fc"), lo.append(0.0), hi.append(gam_max * self.dimensional_M)
        else:
            if self.coulomb:
                names.append("gam"), lo.append(0.0), hi.append(gam_max)
            if self.forced:
                names.append("log_g_u"), lo.append(-math.log(Ms[1])), hi.append(-math.log(Ms[0]))
        if self.spring:
            qa = np.concatenate([e.q for e in self.train])
            rq = max(float(np.ptp(qa)), 1e-6)
            kmax = lam_max ** 2
            names += ["kappa", "q_eq"]
            lo += [-kmax, float(qa.min() - 2 * rq)]
            hi += [kmax, float(qa.max() + 2 * rq)]
        self.names = names
        self.n_theta = len(names)
        self.lo, self.hi = np.array(lo, float), np.array(hi, float)
        # per-training-episode initial-state nuisance, bounded +-10 sd around the prefix estimate
        ic_lo, ic_hi = [], []
        for e in self.train:
            ic_lo += [e.q0 - 10 * e.sigma_q0, e.v0 - 10 * e.sigma_v0]
            ic_hi += [e.q0 + 10 * e.sigma_q0, e.v0 + 10 * e.sigma_v0]
        self.lb = np.r_[self.lo, ic_lo]
        self.ub = np.r_[self.hi, ic_hi]
        self.A = A
        self.n_eval = 0
        self.n_failed = 0
        self.sigma_input = {}            # propagated effort-noise sd per episode (errors in variables)
        self.chol = {}                   # Cholesky factor of the per-episode residual covariance (when propagated)

    # ---------------------------------------------------------------- parameters
    def theta(self, z):
        d = dict(zip(self.names, map(float, z[:self.n_theta])))
        if self.dimensional_M is not None:
            M = self.dimensional_M
            return dict(lam=d["c"] / M, gam=d.get("Fc", 0.0) / M, g_u=1.0 / M, kappa=d.get("kappa", 0.0),
                        q_eq=d.get("q_eq", 0.0), M=M)
        g_u = math.exp(d["log_g_u"]) if "log_g_u" in d else None
        return dict(lam=d["lam"], gam=d.get("gam", 0.0), g_u=g_u, kappa=d.get("kappa", 0.0), q_eq=d.get("q_eq", 0.0))

    def ic(self, z):
        x = z[self.n_theta:]
        return {e.name: (float(x[2 * i]), float(x[2 * i + 1])) for i, e in enumerate(self.train)}

    # ---------------------------------------------------------------- forward model
    def simulate(self, th, ep, q0, v0, t_query=None, record_events=False):
        from ..dynamics.general_1dof import GeneralTerms, sim_params, simulate_general
        from ..synthetic import ZOH
        ag = None if self.gravity.is_zero else self.gravity.fast()
        k, qe = th.get("kappa", 0.0), th.get("q_eq", 0.0)
        if self.spring and ag is not None:
            G = lambda q: ag(q) + k * (q - qe)                     # noqa: E731
        elif self.spring:
            G = lambda q: k * (q - qe)                             # noqa: E731
        else:
            G = ag
        terms = GeneralTerms(inertia=True, viscous=True, coulomb=self.coulomb, static=False, gravity=G is not None,
                             bias=False)
        p = sim_params(terms, self.jt, M=1.0, c=th["lam"], Fc=th["gam"] if self.coulomb else None, G=G)
        if ep.forced:
            if th.get("g_u") is None:
                raise ValueError(f"{ep.name}: forced window in a model without an input gain")
            eff = ZOH(ep.effort_t, th["g_u"] * ep.effort)
        else:
            eff = ZOH(np.array([ep.t0]), np.array([0.0]))
        tq = ep.t if t_query is None else np.asarray(t_query, float)
        return simulate_general(p, eff, tq, t0=ep.t0, q0=q0, v0=v0, max_step=2e-3 if G is not None else 10.0,
                                record_events=record_events)

    def sigma(self, ep):
        s = np.hypot(ep.q_sigma, self.floor) if self.floor else ep.q_sigma
        si = self.sigma_input.get(ep.name)
        return s if si is None else np.hypot(s, si)

    def whiten(self, ep, r):
        """Whitened residual: full covariance (q noise + time-correlated input-noise response) when propagated,
        else the diagonal sigma."""
        L = self.chol.get(ep.name)
        if L is None:
            return r / self.sigma(ep)
        from scipy.linalg import solve_triangular
        return solve_triangular(L, r, lower=True, check_finite=False)

    def residuals(self, z, *, robust=True):
        self.n_eval += 1
        th = self.theta(z)
        ics = self.ic(z)
        out = []
        for e in self.train:
            q0, v0 = ics[e.name]
            try:
                r = self.simulate(th, e, q0, v0)
                zz = self.whiten(e, r.q - e.q)
                if not np.isfinite(zz).all():
                    raise FloatingPointError("non-finite rollout")
            except (ValueError, FloatingPointError, OverflowError):
                self.n_failed += 1
                zz = np.full(len(e.t), 1e6)
            out.append(_huberize(zz, self.delta) if robust else zz)
            out.append(np.array([(q0 - e.q0) / e.sigma_q0, (v0 - e.v0) / e.sigma_v0]))
        return np.concatenate(out)

    def cost(self, z):
        r = self.residuals(z)
        return 0.5 * float(r @ r)

    def x_default(self, theta0=None):
        theta0 = dict(theta0 or {})
        z = []
        for n, lo, hi in zip(self.names, self.lo, self.hi):
            if n == "log_g_u":
                v = math.log(theta0["g_u"]) if theta0.get("g_u") else 0.5 * (lo + hi)
            elif n == "q_eq":
                v = theta0.get("q_eq", 0.5 * (lo + hi))
            elif n == "c":
                v = theta0.get("lam", 0.5) * self.dimensional_M
            elif n == "Fc":
                v = theta0.get("gam", 0.1 * self.A) * self.dimensional_M
            else:
                v = theta0.get(n, {"lam": 0.5, "gam": 0.1 * self.A, "kappa": 0.0}[n])
            z.append(min(max(v, lo + 1e-12 * (hi - lo)), hi - 1e-12 * (hi - lo)))
        for e in self.train:
            z += [e.q0, e.v0]
        return np.array(z, float)


def _solve(model, z0, fixed=None):
    """Bounded robust least squares from z0; ``fixed`` = {index: value} holds parameters (profiles)."""
    free = np.ones(len(z0), bool)
    zf = np.array(z0, float)
    for i, v in (fixed or {}).items():
        free[i] = False
        zf[i] = v
    if not free.any():
        return zf, model.cost(zf), dict(nfev=1, status=1, message="all fixed")

    def res(x):
        z = zf.copy()
        z[free] = x
        return model.residuals(z)
    lb, ub = model.lb[free], model.ub[free]
    x0 = np.clip(zf[free], lb + 1e-12 * (ub - lb), ub - 1e-12 * (ub - lb))
    r = least_squares(res, x0, bounds=(lb, ub), method="trf", x_scale="jac", max_nfev=400, xtol=1e-10, ftol=1e-10,
                      gtol=1e-10)
    z = zf.copy()
    x = r.x.copy()
    tol = 1e-10 * (ub - lb)
    x = np.where(x - lb <= tol, lb, np.where(ub - x <= tol, ub, x))     # trf returns e.g. 5e-324 for a zero bound
    z[free] = x
    return z, float(r.cost), dict(nfev=int(r.nfev), status=int(r.status), message=str(r.message), jac=r.jac,
                                  free=free)


def _starts(model, init, n):
    base = dict(init or {})
    out = [model.x_default(base)]
    lams = [0.05, 1.0, 5.0, 0.3]
    gams = [0.01, 0.3, 0.05, 1.0]
    gus = [None, 1e-2, 1.0, 1e2]
    for k in range(max(0, n - 1)):
        th = dict(lam=lams[k % 4], gam=gams[k % 4] * model.A)
        if model.forced and model.dimensional_M is None:
            g = gus[k % 4]
            th["g_u"] = base.get("g_u") if g is None else g
        out.append(model.x_default(th))
    return out


def nested_start(model, smaller) -> np.ndarray:
    """Start vector for ``model`` at the optimum of a NESTED smaller fit (extra terms at zero / their neutral value,
    the same initial states): the larger variant can then never end above the smaller one's cost."""
    raw = dict(smaller.get("theta_raw") or {})
    z = []
    for n, lo, hi in zip(model.names, model.lo, model.hi):
        v = raw.get(n, {"gam": 0.0, "kappa": 0.0, "Fc": 0.0}.get(n, 0.5 * (lo + hi)))
        z.append(min(max(v, lo + 1e-12 * (hi - lo)), hi - 1e-12 * (hi - lo)))
    zs = np.asarray(smaller["_z"], float)
    return np.r_[np.array(z, float), zs[smaller["_model"].n_theta:]]


def fit(episodes, variant="L1", gravity=None, cfg=None, *, init=None, profiles=True, extra_starts=None,
        noise=None) -> dict:
    """Fit one variant. Returns estimates (estimate_not_for_sim), the local covariance, per-episode replay metrics
    (train AND held-out with theta frozen), multi-start agreement, bound hits and profiles.

    ``extra_starts``: additional start vectors (e.g. the nested smaller optimum). ``noise``: (sigma_input, chol)
    inherited from another fit -- a COMMON residual weighting, required for a likelihood-ratio comparison of nested
    variants (the propagated covariance depends on theta; without it the costs are on different scales)."""
    t_start = time.time()
    cfg = dict(cfg or {})
    model = OEModel(episodes, variant, gravity, cfg)
    if noise is not None:
        model.sigma_input, model.chol = dict(noise[0]), dict(noise[1])
    sols = []
    starts = list(_starts(model, init, int(cfg.get("n_starts", 4)))) + [np.asarray(s, float) for s in
                                                                         (extra_starts or [])]
    for z0 in starts:
        z, c, info = _solve(model, z0)
        sols.append((c, z, info))
    sols.sort(key=lambda s: s[0])
    cbest, zbest, ibest = sols[0]
    noise_info = dict(status="NOT_APPLICABLE (no forced window with an effort sigma)")
    if noise is not None:
        noise_info = dict(status="INHERITED", method="input-noise covariance of the nested minimal fit, held fixed: "
                          "common weighting for the likelihood-ratio comparison",
                          max_sigma_input={k: float(np.max(v)) for k, v in model.sigma_input.items()})
    elif cfg.get("input_noise_propagation", True) and any(e.forced and e.effort_sigma is not None
                                                          for e in model.episodes):
        model.sigma_input, model.chol = input_noise_sigma(model, zbest, n_mc=int(cfg.get("input_noise_mc", 200)))
        z2, c2, i2 = _solve(model, zbest)
        sols = [(model.cost(s[1]), s[1], s[2]) for s in sols]          # re-score every start under the new sigma
        sols.append((c2, z2, i2))
        sols.sort(key=lambda s: s[0])
        cbest, zbest, ibest = sols[0]
        noise_info = dict(status="PROPAGATED", method="Monte Carlo of the recorded effort sigma through the model "
                          "at the first optimum; residuals whitened by chol(diag(sigma_q^2) + C_input) (time "
                          "correlation kept); refit once",
                          max_sigma_input={k: float(np.max(v)) for k, v in model.sigma_input.items()})
    out = _summarize(model, zbest, cbest, ibest)
    out["input_noise"] = noise_info
    # multi-start agreement: every start within the profile threshold must agree on theta
    near = [s for s in sols if s[0] - cbest <= DCHI2_95_HALF * max(1.0, out["chi2_red"])]
    spread = {}
    for i, n in enumerate(model.names):
        vals = np.array([s[1][i] for s in near])
        spread[n] = float(np.ptp(vals))
    out["multistart"] = dict(n_starts=len(sols), costs=[s[0] for s in sols], n_equivalent=len(near),
                             equivalent_spread=spread,
                             unstable=bool(any(spread[n] > max(0.25 * abs(out["theta_raw"][n]), 1e-6 * (h - lo_))
                                               for n, lo_, h in zip(model.names, model.lo, model.hi))
                                           and len(near) > 1))
    if profiles:
        npts = int(cfg.get("profile_points", 15))
        out["profiles"] = {n: profile(model, zbest, cbest, i, out["chi2_red"], n=npts,
                                      sd=(out.get("sd") or {}).get(n)) for i, n in enumerate(model.names)}
    out["runtime_s"] = time.time() - t_start
    out["_model"] = model
    out["_z"] = zbest
    return out


def _summarize(model, z, cost, info):
    th = model.theta(z)
    ics = model.ic(z)
    r_plain = model.residuals(z, robust=False)
    n_obs = int(sum(len(e.t) for e in model.train))
    data_idx = []
    k = 0
    for e in model.train:
        data_idx += list(range(k, k + len(e.t)))
        k += len(e.t) + 2
    chi2 = float(np.sum(r_plain[data_idx] ** 2))
    dof = max(n_obs - model.n_theta, 1)
    chi2_red = chi2 / dof
    # full Gaussian negative log-likelihood INCLUDING 0.5 log det(Sigma): the propagated input-noise covariance
    # depends on theta (larger gain -> larger predicted noise), so whitened costs of different hypotheses are not
    # comparable; the NLL is (the initial-state prior terms are included in both)
    logdet = 0.0
    for e in model.train:
        L = model.chol.get(e.name)
        logdet += float(np.sum(np.log(np.abs(np.diag(L))))) if L is not None else float(np.sum(np.log(model.sigma(e))))
    nll = 0.5 * float(r_plain @ r_plain) + logdet + 0.5 * n_obs * math.log(2 * math.pi)
    # local Gauss-Newton covariance of theta (robust residual Jacobian), chi2-scaled
    cov = None
    J = info.get("jac")
    if J is not None:
        free = info["free"]
        try:
            C = np.linalg.pinv(J.T @ J) * max(1.0, chi2_red)
            Cf = np.full((len(z), len(z)), np.nan)
            Cf[np.ix_(free, free)] = C
            cov = Cf[:model.n_theta, :model.n_theta]
        except np.linalg.LinAlgError:
            cov = None
    raw = dict(zip(model.names, map(float, z[:model.n_theta])))
    at_bound = {n: bool(min(raw[n] - lo, hi - raw[n]) <= 1e-6 * (hi - lo))
                for n, lo, hi in zip(model.names, model.lo, model.hi)}
    eps = {}
    for e in model.episodes:
        q0, v0 = ics.get(e.name, (e.q0, e.v0))
        try:
            r = model.simulate(th, e, q0, v0, record_events=True)
            eps[e.name] = replay_metrics(e, r.q, model.sigma(e), events=r.events)
        except (ValueError, FloatingPointError) as ex:
            eps[e.name] = dict(status="NOT_SIMULABLE", reason=str(ex))
        eps[e.name].update(train=e.train, mode=e.mode, q0_fit=q0, v0_fit=v0, v0_prior=e.v0, sigma_v0=e.sigma_v0,
                           ic_fitted=e.train)
    sd = {}
    if cov is not None:
        for i, n in enumerate(model.names):
            sd[n] = float(math.sqrt(cov[i, i])) if np.isfinite(cov[i, i]) and cov[i, i] >= 0 else None
    est = dict(lam=th["lam"], gam=th["gam"] if model.coulomb else None,
               g_u=th.get("g_u") if (model.forced or model.dimensional_M) else None)
    if model.spring:
        est.update(kappa=th["kappa"], q_eq=th["q_eq"])
    g_u_sd = None
    if est["g_u"] is not None and "log_g_u" in sd and sd["log_g_u"] is not None:
        g_u_sd = est["g_u"] * sd["log_g_u"]
    return dict(variant=model.variant, variant_name=MODEL_VARIANTS[model.variant], names=model.names,
                theta_raw=raw, estimate=est, g_u_sd=g_u_sd, sd=sd, cov=None if cov is None else cov.tolist(),
                at_bound=at_bound, bounds=dict(zip(model.names, zip(model.lo.tolist(), model.hi.tolist()))),
                cost=cost, chi2=chi2, nll=nll, half_logdet_sigma=logdet, n_obs=n_obs, n_params=model.n_theta,
                n_nuisance=2 * len(model.train), dof=dof, chi2_red=chi2_red,
                aic=2 * cost + 2 * model.n_theta, bic=2 * cost + model.n_theta * math.log(max(n_obs, 2)),
                episodes=eps, gravity=model.gravity.describe(), forced_in_training=model.forced,
                units=UNITS[model.jt], n_eval=model.n_eval, n_failed=model.n_failed,
                optimizer=dict(nfev=info.get("nfev"), status=info.get("status"), message=info.get("message")),
                uncertainty_label="local Gauss-Newton covariance of the robust output-error cost (chi2-scaled); "
                                  "profile intervals are the reported confidence regions")


ADEQ_RANGE, ADEQ_SIGMA = 0.05, 3.0


def replay_metrics(ep, q_model, sigma, events=None):
    err = np.asarray(q_model) - ep.q
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rng = float(np.ptp(ep.q)) or 1e-12
    z = err / sigma
    zm = err / np.asarray(ep.q_sigma, float)
    rs, rm = float(np.sqrt(np.mean(z * z))), float(np.sqrt(np.mean(zm * zm)))
    stop_model = next((ev["t_s"] for ev in (events or []) if ev.get("kind") == "stop"), None)
    stop_obs = observed_stop_time(ep)
    # a misfit larger than the WHOLE observed motion and the measurement noise is never adequate, even when the
    # propagated input noise of a large fitted gain would cover it
    swallowed = bool(rmse / rng > 1.0 and rm > ADEQ_SIGMA)
    return dict(status="OK", rmse=rmse, max_abs_error=float(np.max(np.abs(err))), rmse_over_range=rmse / rng,
                rmse_over_sigma=rs, rmse_over_sigma_q_measurement=rm, final_error=float(err[-1]),
                adequate=bool((rmse / rng <= ADEQ_RANGE or rs <= ADEQ_SIGMA) and not swallowed),
                misfit_only_within_propagated_input_noise=swallowed,
                stop_time_model=stop_model, stop_time_observed=stop_obs,
                stop_time_error=(stop_model - stop_obs) if (stop_model is not None and stop_obs is not None) else None,
                rule=f"adequate: (rmse/range <= {ADEQ_RANGE} or rmse/sigma <= {ADEQ_SIGMA}) and not (rmse/range > 1 "
                     f"and rmse/sigma_q_measurement > {ADEQ_SIGMA})")


def observed_stop_time(ep, k=3.0):
    """First sample after which q stays within k sigma of its final value AND the window ends at rest-like motion
    (None when the window ends moving)."""
    q, s = ep.q, ep.q_sigma
    if len(q) < 5:
        return None
    tail = np.abs(q - q[-1]) <= k * np.maximum(s, s[-1])
    if not tail[-3:].all():
        return None
    i = len(q) - 1
    while i > 0 and tail[i - 1]:
        i -= 1
    if i >= len(q) - 3:
        return None                                   # the window ends while still moving
    return float(ep.t[i])


def input_noise_sigma(model, z, n_mc=40, seed=0):
    """sd of q(t) induced by the recorded effort noise (independent per native row), at theta(z) and the fitted
    initial state: the output-error likelihood otherwise treats the measured input as exact."""
    rng = np.random.default_rng(seed)
    th, ics = model.theta(z), model.ic(z)
    out, chol = {}, {}
    from dataclasses import replace
    for e in model.episodes:
        if not e.forced or e.effort_sigma is None or th.get("g_u") is None:
            continue
        q0, v0 = ics.get(e.name, (e.q0, e.v0))
        sims = []
        for _ in range(int(n_mc)):
            ep = replace(e, effort=e.effort + rng.standard_normal(len(e.effort)) * e.effort_sigma)
            sims.append(model.simulate(th, ep, q0, v0).q)
        X = np.asarray(sims)
        out[e.name] = np.std(X, axis=0, ddof=1)
        Cin = np.cov(X, rowvar=False)
        s_q = np.hypot(e.q_sigma, model.floor) if model.floor else e.q_sigma
        S = np.diag(s_q ** 2) + Cin
        try:
            chol[e.name] = np.linalg.cholesky(S + 1e-15 * np.trace(S) * np.eye(len(S)))
        except np.linalg.LinAlgError:
            pass                                               # falls back to the diagonal sigma
    return out, chol


# ======================================================================================================== profiles
def profile_scan(evaluate, grid, thr, *, log=False, n_bisect=5, lo=None, hi=None):
    """Evaluate delta-cost on ``grid``; on each side of the minimum bisect the threshold crossing (``n_bisect`` extra
    evaluations) -- linear interpolation of a convex profile between sparse points underestimates the interval."""
    pts = {}
    for g in np.unique(np.asarray(grid, float)):
        try:
            pts[float(g)] = float(evaluate(float(g)))
        except (ValueError, FloatingPointError):
            pts[float(g)] = np.nan
    X = lambda x: math.log(x) if log else x             # noqa: E731
    Xi = lambda y: math.exp(y) if log else y            # noqa: E731

    def table():
        xs = np.array(sorted(pts))
        d = np.array([pts[x] for x in xs])
        return xs, d - np.nanmin(d) if np.isfinite(d).any() else d
    xs, d = table()
    i0 = int(np.nanargmin(d))
    res = {}
    for side in (-1, 1):
        j = i0
        while 0 <= j + side < len(xs) and not (np.isfinite(d[j + side]) and d[j + side] > thr):
            j += side
        if not (0 <= j + side < len(xs)):
            res[side] = (float(xs[0] if side < 0 else xs[-1]), True)
            continue
        a, b = xs[j], xs[j + side]
        base = float(np.nanmin([pts[x] for x in pts]))
        for _ in range(int(n_bisect)):
            m = Xi(0.5 * (X(a) + X(b)))
            try:
                pts[m] = float(evaluate(m))
            except (ValueError, FloatingPointError):
                break
            if pts[m] - base > thr:
                b = m
            else:
                a = m
        da, db = pts[a] - base, pts[b] - base
        f = 0.0 if db == da else (thr - da) / (db - da)
        res[side] = (float(Xi(X(a) + min(max(f, 0.0), 1.0) * (X(b) - X(a)))), False)
    xs, d = table()
    return dict(grid=xs.tolist(), delta_cost=d.tolist(), ci95=[res[-1][0], res[1][0]], open_low=res[-1][1],
                open_high=res[1][1], threshold=thr, x_best=float(xs[int(np.nanargmin(d))]))


def profile(model, zbest, cbest, i, chi2_red, n=None, sd=None):
    """Profile of parameter ``i``: fix it on a grid, re-optimise everything else (warm-started at the optimum)."""
    name = model.names[i]
    lo, hi = model.lo[i], model.hi[i]
    v = zbest[i]
    n = int(n or 17)
    if name in ("lam", "gam", "c", "Fc"):
        top = min(hi, max(4.0 * v, 1e-3 * (hi - lo) + v, 10 * (v if v > 0 else 0.0)))
        grid = np.unique(np.r_[0.0, np.geomspace(max(v, top) * 1e-3, top, n - 1), v])
        log = False
    elif name == "log_g_u":
        grid = np.unique(np.r_[np.linspace(max(lo, v - 4.0), min(hi, v + 4.0), n), v])
        log = False
    else:
        w = 0.5 * (hi - lo)
        grid = np.unique(np.r_[np.linspace(max(lo, v - w), min(hi, v + w), n), v])
        log = False
    if sd is not None and math.isfinite(sd) and sd > 0:
        grid = np.r_[grid, v + sd * np.array([-6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6])]
    grid = np.unique(np.clip(grid, lo, hi))

    def ev(g):
        _, c, _ = _solve(model, zbest, fixed={i: float(g)})
        return c - cbest
    thr = DCHI2_95_HALF * max(1.0, chi2_red)
    sc = profile_scan(ev, grid, thr, log=log)
    a, b = sc["ci95"]
    ol, oh = sc["open_low"], sc["open_high"]
    out = dict(parameter=name, grid=sc["grid"], delta_cost=sc["delta_cost"], threshold=thr, ci95=[a, b],
               open_low=ol, open_high=oh, estimate_not_for_sim=float(v),
               note="cost = sum rho_huber + 0.5 IC prior; 95 % set: delta cost <= 1.92 max(1, chi2_red); "
                    "crossings refined by bisection")
    if name == "log_g_u":
        out["ci95_g_u"] = [math.exp(a), math.exp(b)]
        out["ci95_M"] = [math.exp(-b), math.exp(-a)]
    if name in ("lam", "gam", "c", "Fc"):
        out["reaches_zero"] = bool(a is not None and a <= 1e-12)
    return out


def scale_profile(episodes, variant, gravity, cfg, M_grid, sigma_input=None) -> dict:
    """Absolute-scale profile in the DIMENSIONAL parameterization: at each fixed M, fit (c, Fc) and the initial states
    with the input u unscaled and the geometry gravity scaled with the same density (G = M a_g). A free-only data set
    gives a flat profile (scale symmetry) -- reported as RATIO_ONLY, never a minimum at the grid edge or at M = 0."""
    cfg = dict(cfg or {})
    rec = {}
    sig_in, chol = ({}, {}) if sigma_input is None else sigma_input

    def fit_at(M):
        m = OEModel(episodes, variant, gravity, cfg, dimensional_M=float(M))
        m.sigma_input, m.chol = sig_in, chol
        best = None
        for z0 in _starts(m, None, 2):
            z, c, _ = _solve(m, z0)
            if best is None or c < best[0]:
                best = (c, z)
        th = m.theta(best[1])
        rec[float(M)] = dict(M=float(M), cost=best[0], c=th["lam"] * M, Fc=th["gam"] * M, lam=th["lam"],
                             gam=th["gam"])
        return best[0]
    for M in M_grid:
        fit_at(M)
    cmin = min(p["cost"] for p in rec.values())
    n_obs = sum(len(e.t) for e in episodes if e.train)
    npar = 1 + (variant in ("L1", "L3"))
    chi2_red = max(1.0, 2 * cmin / max(n_obs - npar - 1, 1))
    thr = DCHI2_95_HALF * chi2_red
    flat = bool(max(p["cost"] for p in rec.values()) - cmin <= thr)
    if not flat:
        sc = profile_scan(fit_at, list(rec), thr, log=True)
        lo, hi = sc["ci95"]
        ol, oh = sc["open_low"], sc["open_high"]
    else:
        lo = hi = None
        ol = oh = True
    pts = [rec[k] for k in sorted(rec)]
    costs = np.array([p["cost"] for p in pts])
    dL = costs - costs.min()
    Ms = np.array([p["M"] for p in pts])
    if flat:
        st = "FLAT_SCALE_SYMMETRY"
    elif ol or oh:
        st = "OPEN"
    else:
        st = "CLOSED"
    i0 = int(np.argmin(costs))
    return dict(schema="forceartgs.g9.io_scale_profile/v1", parameterization="dimensional (M fixed; c, Fc, q0, v0 "
                "re-optimised; G = M a_g)", points=pts, delta_cost=dL.tolist(), threshold=thr, status=st,
                M_best_not_for_sim=float(Ms[i0]), M_best_at_grid_edge=bool(i0 in (0, len(Ms) - 1)),
                ci95_M=None if flat else [lo, hi], open_low=ol, open_high=oh, flat_range=float(np.ptp(dL)),
                note=("flat: (M, c, Fc) -> (kM, kc, kFc) leaves the trajectories unchanged -- ratios only" if flat else
                      "the measured input breaks the scale symmetry over this grid" if st == "CLOSED" else
                      "the profile reaches the grid edge: bounded on one side only"))


# ======================================================================================================== bootstrap
def block_bootstrap(fitres, B=8, seed=0, block=None) -> dict:
    """Moving-block RESIDUAL bootstrap on the training windows (keeps local time correlation): q* = q_fit + blocks
    of the fitted residuals, refit from the optimum. A sensitivity summary -- NOT a calibrated confidence interval
    (the profile intervals are reported for that)."""
    from dataclasses import replace
    model, z = fitres["_model"], fitres["_z"]
    if B <= 0:
        return dict(status="NOT_RUN")
    rng = np.random.default_rng(seed)
    th, ics = model.theta(z), model.ic(z)
    fits, res = {}, {}
    for e in model.train:
        q0, v0 = ics[e.name]
        fits[e.name] = model.simulate(th, e, q0, v0).q
        res[e.name] = e.q - fits[e.name]
    out = []
    for b in range(int(B)):
        eps = []
        for e in model.episodes:
            if not e.train:
                eps.append(e)
                continue
            r = res[e.name]
            n = len(r)
            L = int(block or max(2, n // 5))
            idx = []
            while len(idx) < n:
                s = int(rng.integers(0, max(n - L + 1, 1)))
                idx += list(range(s, min(s + L, n)))
            q_star = fits[e.name] + r[np.array(idx[:n])]
            eps.append(replace(e, q=q_star))
        m2 = OEModel(eps, model.variant, model.gravity, dict(huber_delta=model.delta or 2.0,
                                                             robust_loss="huber" if model.delta else "l2",
                                                             sigma_floor_q=model.floor))
        m2.lo, m2.hi, m2.lb, m2.ub = model.lo, model.hi, model.lb, model.ub
        m2.sigma_input, m2.chol = model.sigma_input, model.chol
        zb, _, _ = _solve(m2, z)
        out.append(dict(zip(model.names, map(float, zb[:model.n_theta]))))
    summ = {n: dict(median=float(np.median([o[n] for o in out])),
                    p05=float(np.quantile([o[n] for o in out], 0.05)),
                    p95=float(np.quantile([o[n] for o in out], 0.95))) for n in model.names}
    return dict(status="OK", B=int(B), seed=seed, samples=out, summary=summ,
                label="MOVING_BLOCK_RESIDUAL_BOOTSTRAP (sensitivity; NOT a calibrated CI)")


# ======================================================================================================== variants
def fit_variants(episodes, gravity, cfg, *, init=None, variants=None, breakaway_events=None) -> dict:
    """Fit the nested variants on the SAME windows; L2 / L4 are recorded NOT_RUN with the reason."""
    cfg = dict(cfg or {})
    listed = variants or ["inertial_viscous", "inertial_viscous_coulomb"]
    from .contracts import VARIANT_BY_NAME
    keys = [VARIANT_BY_NAME[v] for v in listed]
    out = {}
    for k in ("L0", "L1", "L2", "L3", "L4"):
        if k not in keys:
            out[k] = dict(variant=k, variant_name=MODEL_VARIANTS[k], status="NOT_RUN", reason="not listed in config")
            continue
        if k == "L2":
            out[k] = dict(variant=k, variant_name=MODEL_VARIANTS[k], status="NOT_RUN",
                          reason="no independently confirmed start / stop (breakaway) events: Fs is never inferred "
                                 "from non-motion at a single unknown effort" if not breakaway_events else
                                 "static breakaway fitting is not implemented by the output-error stage")
            continue
        if k == "L4":
            out[k] = dict(variant=k, variant_name=MODEL_VARIANTS[k], status="NOT_RUN",
                          reason="research-only: requires L0-L3 to fail adequacy AND independent validation data")
            continue
        if k == "L3":
            tr = [e for e in episodes if e.train]
            qa = np.concatenate([e.q for e in tr])
            cov_q = float(np.ptp(qa))
            if len(tr) < 2 or cov_q < float(cfg.get("spring_min_q_coverage", 0.3)):
                out[k] = dict(variant=k, variant_name=MODEL_VARIANTS[k], status="NOT_RUN",
                              reason=f"configuration coverage {cov_q:.3g} over {len(tr)} training window(s) is too "
                                     "small for a repeatable configuration-dependent load")
                continue
        base = out.get("L0") if k != "L0" else None
        if base is not None and base.get("status") == "FITTED":
            m0 = base["_model"]
            m = OEModel(episodes, k, gravity, cfg)
            r = fit(episodes, k, gravity, cfg, init=init, extra_starts=[nested_start(m, base)],
                    noise=(m0.sigma_input, m0.chol) if m0.chol or m0.sigma_input else None)
        else:
            r = fit(episodes, k, gravity, cfg, init=init)
        r["status"] = "FITTED"
        out[k] = r
    return out
