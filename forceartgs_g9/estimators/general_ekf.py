"""GeneralAugmentedEKF: augmented-state EKF for the FULL general model (G9 PINN/EKF spec section 3).

State  x = [q, v, eta_1 .. eta_p]   eta = constraint-preserving coordinates of EXACTLY the ModelSpec parameters
       (torch_model: log for alpha / M / mu / c / Fc / dFs / Fs, identity for b / gA / gB / gP); in geometry_coupled
       mode one log(rho) carries BOTH inertia and gravity (dM/drho and dG/drho enter the same Jacobian column).
Process q' = v,  v' = (Q - c v - Fc sgn v - G(q) - b) / M  (hybrid Karnopp stick / slip),  eta' = 0.
       The transition over each interval between consecutive events (native 50 Hz force knots and camera times) is
       the SAME event-driven simulator as the rollout fit (simulate_general, constant ZOH Q on the interval).
       F = d f / d x by central differences (verified in the tests); a switch flag is raised when the perturbed
       transitions disagree in regime (stick vs slip) -- the linearisation is then not trusted.
Noise  process: declared generalized-effort noise s_Q (adapter: force sigma x lever) as an acceleration noise
       (s_Q / M0)^2 dt on v, a tiny q noise; parameters exactly constant (zero process noise).
       measurement: z = q_obs + nu, R = q_std^2; H = [1, 0, ..., 0]; no finite-difference velocity is fed.
Update Joseph form P+ = (I - K H) P- (I - K H)^T + K R K^T; innovation and NIS logged.
Episodes  forward filter per training interaction; the PARAMETER posterior (mean + covariance) carries over,
       q / v are re-initialised (q0 = first observation, v0 from a local fit with its variance). The episode-order
       effect is reported (forward vs reversed order).
Small posterior covariance is NOT identifiability: structural / excitation checks are reported alongside.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from ..dynamics.general_1dof import ModelConfigError, simulate_general
from .torch_model import LOG_PARAMS


@dataclass
class EKFConfig:
    eta_prior_sd_log: float = 1.0          # 1-sigma prior on log parameters (factor e)
    eta_prior_sd_signed_rel: float = 1.0   # 1-sigma prior on signed params, relative to max(|x0|, effort scale)
    v0_sd_factor: float = 3.0
    q_process_sd: float = 1e-6
    accel_noise_floor: float = 0.02        # effort-noise floor: 2 % of the effort RMS (declared)
    fd_rel: float = 1e-6
    max_step: float = 2e-3
    reverse_order_check: bool = True

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        bad = sorted(set(d) - set(cls.__dataclass_fields__))
        if bad:
            raise ModelConfigError(f"fitting.ekf: unknown keys {bad}")
        return cls(**d)


class _ConstQ:
    def __init__(self, Q):
        self.t_s = np.array([0.0])
        self.levels = np.array([float(Q)])

    def at(self, t):
        return np.full(np.shape(t), self.levels[0])


def _theta(names, eta):
    return np.array([math.exp(e) if n in LOG_PARAMS else e for n, e in zip(names, eta)])


NONSMOOTH = ("dFs", "Fs")        # zero local sensitivity: profiled on the filter likelihood, never a linearised state


class GeneralAugmentedEKF:
    def __init__(self, spec, cfg: EKFConfig | None = None):
        self.spec, self.cfg = spec, cfg or EKFConfig()
        self.all_names = list(spec.names)
        self.names = [n for n in self.all_names if n not in NONSMOOTH]
        self.fixed = {}                                  # non-smooth parameter values (set by the profile scan)
        self.p = len(self.names)

    def full_theta(self, eta):
        th = dict(zip(self.names, _theta(self.names, eta)))
        th.update(self.fixed)
        return np.array([th[n] for n in self.all_names])

    # ------------------------------------------------------------------ transition
    def f(self, x, Q, dt):
        """One interval of the hybrid dynamics with constant Q; returns (x_next, stuck_flag)."""
        sp = self.spec.sim(self.full_theta(x[2:]))
        r = simulate_general(sp, _ConstQ(Q), [dt], t0=0.0, q0=float(x[0]), v0=float(x[1]),
                             max_step=self.cfg.max_step)
        xn = x.copy()
        xn[0], xn[1] = float(r.q[0]), float(r.qdot[0])
        return xn, bool(r.stuck[0])

    def jacobian(self, x, Q, dt):
        n = len(x)
        F = np.eye(n)
        regimes = set()                                                 # parameter rows stay identity (eta' = 0)
        for j in range(n):
            h = self.cfg.fd_rel * max(1.0, abs(x[j])) if j >= 2 else max(1e-7, self.cfg.fd_rel * max(abs(x[j]), 1e-3))
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            fp, sp_ = self.f(xp, Q, dt)
            fm, sm_ = self.f(xm, Q, dt)
            regimes |= {sp_, sm_}
            F[:2, j] = (fp[:2] - fm[:2]) / (2 * h)
        return F, len(regimes) > 1

    # ------------------------------------------------------------------ one episode
    def run_episode(self, it, eta_mean, eta_cov, s_Q, M0):
        cfg = self.cfg
        tq, qo, sq = it.t_obs, it.q_obs, it.q_std
        tf = np.asarray(it.effort.t_s, float)
        lev = np.asarray(it.effort.levels, float)
        times = np.unique(np.r_[tf[(tf > tq[0]) & (tf < tq[-1])], tq])
        k0 = min(4, len(tq))
        A = np.c_[np.ones(k0), tq[:k0] - tq[0]]
        coef, *_ = np.linalg.lstsq(A, qo[:k0], rcond=None)
        v0 = float(coef[1])
        v0_sd = cfg.v0_sd_factor * float(np.median(sq)) / max(tq[k0 - 1] - tq[0], 1e-3)
        n = 2 + self.p
        x = np.r_[qo[0], v0, eta_mean]
        P = np.zeros((n, n))
        P[0, 0], P[1, 1] = sq[0] ** 2, v0_sd ** 2
        P[2:, 2:] = eta_cov
        obs_idx = {float(t): i for i, t in enumerate(tq)}
        acc_sd = s_Q / max(M0, 1e-12)
        log = dict(t=[], q=[], q_sd=[], v=[], v_sd=[], stuck=[], switch=[], innov=[], nis=[], t_obs=[], nll=0.0)
        t = float(tq[0])
        # first observation is the initial condition (no double counting)
        for tn in times[1:]:
            dt = float(tn - t)
            if dt <= 0:
                continue
            Q = float(lev[int(np.clip(np.searchsorted(tf, t + 1e-12, side="right") - 1, 0, len(lev) - 1))])
            F, sw = self.jacobian(x, Q, dt)
            x, stuck = self.f(x, Q, dt)
            Qp = np.zeros((n, n))
            Qp[0, 0] = (cfg.q_process_sd ** 2) * dt
            Qp[1, 1] = 0.0 if stuck else (acc_sd ** 2) * dt
            P = F @ P @ F.T + Qp
            t = float(tn)
            if t in obs_idx:
                i = obs_idx[t]
                H = np.zeros((1, n))
                H[0, 0] = 1.0
                R = float(sq[i] ** 2)
                S = float((H @ P @ H.T)[0, 0]) + R
                K = (P @ H.T) / S
                nu = float(qo[i] - x[0])
                x = x + (K * nu).ravel()
                IKH = np.eye(n) - K @ H
                P = IKH @ P @ IKH.T + (K * R) @ K.T                     # Joseph form
                P = 0.5 * (P + P.T)
                log["innov"].append(nu)
                log["nis"].append(nu * nu / S)
                log["nll"] += math.log(2 * math.pi * S) + nu * nu / S
                log["t_obs"].append(t)
            log["t"].append(t)
            log["q"].append(float(x[0]))
            log["v"].append(float(x[1]))
            log["q_sd"].append(float(math.sqrt(max(P[0, 0], 0))))
            log["v_sd"].append(float(math.sqrt(max(P[1, 1], 0))))
            log["stuck"].append(stuck)
            log["switch"].append(sw)
        return x[2:], P[2:, 2:], log, float(np.min(np.linalg.eigvalsh(P)))

    def fit(self, interactions, theta0, s_Q=None):
        t0 = time.time()
        cfg = self.cfg
        eff = float(np.sqrt(np.mean(np.concatenate([np.asarray(i.effort.levels) ** 2 for i in interactions]))))
        ph0 = self.spec.physical(np.asarray(theta0, float))
        M0 = float(ph0["M"])
        sQ = max(float(s_Q or 0.0), cfg.accel_noise_floor * eff)
        th0 = dict(zip(self.all_names, np.asarray(theta0, float)))
        eta0 = np.array([math.log(max(th0[n], 1e-12)) if n in LOG_PARAMS else float(th0[n]) for n in self.names])
        sd = np.array([cfg.eta_prior_sd_log if n in LOG_PARAMS else cfg.eta_prior_sd_signed_rel * max(abs(th0[n]), eff)
                       for n in self.names])

        def run(order):
            m, C = eta0.copy(), np.diag(sd ** 2)
            logs, minev = {}, []
            for k in order:
                m, C, lg, me = self.run_episode(interactions[k], m, C, sQ, M0)
                logs[interactions[k].name] = lg
                minev.append(me)
            return m, C, logs, min(minev)
        order = list(range(len(interactions)))
        profile = None
        ns = [n for n in self.all_names if n in NONSMOOTH]
        if ns:
            # profile of the EKF prediction-error likelihood over the breakaway parameter (piecewise constant in it)
            grid = np.unique(np.r_[np.linspace(0.0, float(np.max(np.abs(np.concatenate(
                [i.effort.levels for i in interactions])))), 13), th0[ns[0]]])
            vals = []
            for gv in grid:
                self.fixed = {ns[0]: float(gv)}
                try:
                    _, _, lg, _ = run(order)
                    vals.append(sum(l["nll"] for l in lg.values()))
                except (ModelConfigError, np.linalg.LinAlgError, FloatingPointError):
                    vals.append(np.inf)
            vals = np.asarray(vals)
            k = int(np.argmin(vals))
            lo_, hi_ = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
            fine = np.linspace(lo_, hi_, 9)
            fv = []
            for gv in fine:
                self.fixed = {ns[0]: float(gv)}
                _, _, lg, _ = run(order)
                fv.append(sum(l["nll"] for l in lg.values()))
            allg, allv = np.r_[grid, fine], np.r_[vals, fv]
            best = float(allg[int(np.argmin(allv))])
            inside = allg[allv <= np.min(allv) + 3.84]
            self.fixed = {ns[0]: best}
            profile = dict(parameter=ns[0], best=best, interval95=[float(inside.min()), float(inside.max())],
                           unbounded_above=bool(inside.max() >= grid.max()), grid=allg.tolist(), nll=allv.tolist(),
                           method="profile of the EKF prediction-error likelihood (sum log S + nu^2 / S)")
        m, C, logs, minev = run(order)
        rev = None
        if cfg.reverse_order_check and len(order) > 1:
            m2, C2, _, _ = run(order[::-1])
            a2, a1 = self.full_theta(m2), self.full_theta(m)
            rev = dict(theta_reversed=a2.tolist(),
                       max_rel_difference=float(np.max(np.abs(a2 - a1) / np.maximum(np.abs(a1), 1e-12))))
        th_s = _theta(self.names, m)
        J = np.array([th_s[j] if n in LOG_PARAMS else 1.0 for j, n in enumerate(self.names)])   # d theta / d eta
        Cs = (J[:, None] * C) * J[None, :]
        th = self.full_theta(m)
        Ct = np.zeros((len(self.all_names),) * 2)
        idx = [self.all_names.index(n) for n in self.names]
        Ct[np.ix_(idx, idx)] = Cs
        sdv = np.sqrt(np.clip(np.diag(Ct), 0, None))
        nis = np.concatenate([np.asarray(l["nis"]) for l in logs.values()]) if logs else np.zeros(0)
        nsw = int(sum(np.sum(l["switch"]) for l in logs.values()))
        est = {}
        for j, n in enumerate(self.all_names):
            if n in NONSMOOTH:
                lo95, hi95 = profile["interval95"]
                est[n] = dict(value=float(th[j]), std=(hi95 - lo95) / 3.92 if not profile["unbounded_above"] else math.inf,
                              interval95=[lo95, hi95], uncertainty_label="EKF_LIKELIHOOD_PROFILE", at_lower_bound=bool(
                                  th[j] < 1e-9))
            else:
                k = self.names.index(n)
                est[n] = dict(value=float(th[j]), std=float(sdv[j]), eta_mean=float(m[k]),
                              eta_sd=float(math.sqrt(max(C[k, k], 0.0))),
                              uncertainty_label="EKF_POSTERIOR (linearised; not identifiability by itself)",
                              at_lower_bound=bool(th[j] < 1e-9))
        replay = _replay(self.spec, th, interactions)
        ceig = float(np.min(np.linalg.eigvalsh(0.5 * (C + C.T)))) if C.size else 0.0
        bad = (not np.all(np.isfinite(C))) or ceig < -1e-9 * max(1.0, float(np.max(np.abs(np.diag(C))))) \
            or minev < -1e-9
        status = "covariance_not_psd" if bad else "ok"          # e.g. force sign error / model inconsistent with data
        return dict(status=status, method="general_akf", estimates=est, physical=self.spec.physical(th), x=th.tolist(),
                    covariance=Ct.tolist(), eta_covariance=C.tolist(),
                    nis=dict(mean=float(nis.mean()) if len(nis) else None, n=int(len(nis)),
                             frac_above_3_84=float(np.mean(nis > 3.84)) if len(nis) else None,
                             expected_mean=1.0,
                             mean_upper95=float(1 + 1.96 * math.sqrt(2.0 / len(nis))) if len(nis) else None,
                             consistent=bool(len(nis) and nis.mean() <= 1 + 1.96 * math.sqrt(2.0 / len(nis))),
                             note="mean NIS of n chi2(1) innovations; >> 1 means model / force / sign inconsistent"),
                    switch_linearisation_events=nsw, min_state_cov_eigenvalue=minev, min_eta_cov_eigenvalue=ceig, order_effect=rev,
                    settings=asdict(cfg), effort_noise_sQ=sQ, M0_for_process_noise=M0, nonsmooth_profile=profile,
                    augmented_state=["q", "v"] + [f"eta_{n}" for n in self.names],
                    timelines={k: {kk: (vv if kk != "switch" else [bool(x) for x in vv]) for kk, vv in v.items()}
                               for k, v in logs.items()},
                    replay=replay, runtime_s=time.time() - t0,
                    note="parameter posterior from the forward filter over the training episodes in their given "
                         "order; q / v re-initialised per episode; parameters constant (no random walk)")


def _replay(spec, theta, interactions):
    out = []
    for it in interactions:
        v0 = float(np.polyfit(it.t_obs[:4] - it.t_obs[0], it.q_obs[:4], 1)[0])
        try:
            r = simulate_general(spec.sim(theta), it.effort, it.t_obs, t0=float(it.t_obs[0]), q0=float(it.q_obs[0]),
                                 v0=v0, max_step=1e-3)
            out.append(dict(interaction=it.name, rmse_over_sigma=float(np.sqrt(np.mean(((r.q - it.q_obs) /
                                                                                         it.q_std) ** 2)))))
        except ModelConfigError as e:
            out.append(dict(interaction=it.name, error=str(e)))
    return dict(per_interaction=out, note="open-loop replay with the frozen final EKF parameters (not the filter)")


def general_akf_fit(interactions, spec, theta0, cfg: EKFConfig | None = None, s_Q=None) -> dict:
    return GeneralAugmentedEKF(spec, cfg).fit(interactions, theta0, s_Q)
