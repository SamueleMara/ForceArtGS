"""Normalized augmented EKF + RTS smoother (spec 6.3).

State x = [q, v, eta_1 .. eta_p], rates = softplus(eta) (lam, gam [, beta in the FORCED stage only]).
Moving-regime transition per substep h (semi-implicit Euler, the same normalized equation as every estimator):
    v' = v + h (-lam v - gam s - a_g(q) + beta Q),   q' = q + h v',   eta' = eta      (s = sgn v at substep start)
Zero crossing with gam > 0: v' := 0 and the joint sticks while |beta Q - a_g(q)| <= gam (the kinetic level is the
stick threshold; no Jacobian of sgn at 0 -- the stuck substep has dv'/d. = 0). Measurement z = q_obs + N(0, q_sigma^2),
Joseph covariance update, NIS per update. Process noise: white acceleration sigma_a on v, CALIBRATED by the mean
NIS over a declared grid (recorded). RTS smoothing per segment (future-conditioned -- never a held-out prediction).
Parameter identifiability: posterior / prior sd of eta (shrinkage); a bounded / prior-dominated covariance never
counts as identified.
"""
from __future__ import annotations

import math
import time

import numpy as np


def softplus(x):
    return float(np.logaddexp(0.0, x))


def sigmoid(x):
    return float(1.0 / (1.0 + math.exp(-x))) if x > -700 else 0.0


def inv_softplus(y):
    y = max(float(y), 1e-12)
    return float(y + math.log(-math.expm1(-y))) if y < 30 else float(y)


class NormalizedAEKF:
    def __init__(self, nspec, *, sigma_a=1.0, sigma_eta_rw=0.0, h_max=2e-3):
        self.ns, self.sigma_a, self.sigma_eta_rw, self.h_max = nspec, float(sigma_a), float(sigma_eta_rw), h_max
        self.names = nspec.names
        self.p = len(self.names)
        self.n = 2 + self.p

    def rates(self, eta):
        r = dict(lam=0.0, gam=0.0, beta=0.0)
        for n, e in zip(self.names, eta):
            r[n] = softplus(e)
        return r

    # ------------------------------------------------------------------ one substep + analytic Jacobian
    def substep(self, x, Q, h, stuck):
        q, v = x[0], x[1]
        eta = x[2:]
        r = self.rates(eta)
        ag = self.ns.gravity(q)
        F = np.eye(self.n)
        drive = r["beta"] * Q - ag
        if stuck or v == 0.0:
            if self.ns.coulomb and abs(drive) <= r["gam"]:
                F[1, :] = 0.0
                y = x.copy()
                y[1] = 0.0
                return y, F, True
            s = 1.0 if drive > 0 else -1.0
        else:
            s = 1.0 if v > 0 else -1.0
        a = -r["lam"] * v - r["gam"] * s + drive
        v1 = v + h * a
        if self.ns.coulomb and r["gam"] > 0 and v != 0.0 and v1 * v < 0:      # zero crossing inside the substep
            y = x.copy()
            y[0] = q + 0.5 * h * v * (-v / (v1 - v))                          # stop inside the substep (linear v)
            y[1] = 0.0
            F[1, :] = 0.0                                                     # v' = 0: no Jacobian of sgn at 0
            return y, F, abs(drive) <= r["gam"]
        dv = np.zeros(self.n)
        dv[0] = -h * self.ns.gravity.deriv(q)
        dv[1] = 1.0 - h * r["lam"]
        for i, nm in enumerate(self.names):
            e = eta[i]
            if nm == "lam":
                dv[2 + i] = -h * v * sigmoid(e)
            elif nm == "gam":
                dv[2 + i] = -h * s * sigmoid(e)
            elif nm == "beta":
                dv[2 + i] = h * Q * sigmoid(e)
        y = x.copy()
        y[1] = v1
        y[0] = q + h * v1
        F[1, :] = dv
        F[0, :] = np.eye(self.n)[0] + h * dv
        return y, F, False

    def propagate(self, x, dt, Qfun, t, stuck=False):
        nsub = max(1, int(math.ceil(dt / self.h_max)))
        h = dt / nsub
        Ft = np.eye(self.n)
        for k in range(nsub):
            x, F, stuck = self.substep(x, Qfun(t + k * h), h, stuck)
            Ft = F @ Ft
        return x, Ft, stuck

    def jacobian_fd(self, x, Q, h, eps=1e-7):
        """Finite-difference Jacobian of one substep (tests; away from discontinuities)."""
        J = np.empty((self.n, self.n))
        for j in range(self.n):
            d = np.zeros(self.n)
            d[j] = eps
            J[:, j] = (self.substep(x + d, Q, h, False)[0] - self.substep(x - d, Q, h, False)[0]) / (2 * eps)
        return J

    def process_noise(self, dt):
        Qd = np.zeros((self.n, self.n))
        sa2 = self.sigma_a ** 2
        Qd[0, 0], Qd[0, 1], Qd[1, 0], Qd[1, 1] = sa2 * dt ** 3 / 3, sa2 * dt ** 2 / 2, sa2 * dt ** 2 / 2, sa2 * dt
        Qd[2:, 2:] = np.eye(self.p) * self.sigma_eta_rw ** 2 * dt
        return Qd

    # ------------------------------------------------------------------ filter + smoother over segments
    def run(self, segments, eta0, P_eta0, smooth=True):
        eta, Pe = np.asarray(eta0, float).copy(), np.asarray(P_eta0, float).copy()
        out_segs, nis_all = [], []
        for seg in segments:
            k = min(4, len(seg.t))
            A = np.vander(seg.t[:k] - seg.t[0], 2, increasing=True)
            b = np.linalg.lstsq(A, seg.q[:k], rcond=None)[0]
            x = np.r_[seg.q[0], b[1], eta]
            P = np.zeros((self.n, self.n))
            P[0, 0] = seg.q_sigma[0] ** 2
            vdt = max(abs(b[1]), float(np.median(seg.q_sigma)) / max(seg.t[1] - seg.t[0], 1e-3))
            P[1, 1] = (2.0 * vdt) ** 2
            P[2:, 2:] = Pe
            eff = seg.effort if (self.ns.forced and not seg.is_free) else None
            Qfun = (lambda t, e=eff: float(e.at(t))) if eff is not None else (lambda t: 0.0)
            xs_f, Ps_f, xs_p, Ps_p, Fs, nis = [], [], [], [], [], []
            stuck = False
            for i in range(len(seg.t)):
                if i > 0:
                    dt = float(seg.t[i] - seg.t[i - 1])
                    x, F, stuck = self.propagate(x, dt, Qfun, float(seg.t[i - 1]), stuck)
                    P = F @ P @ F.T + self.process_noise(dt)
                    Fs.append(F)
                xs_p.append(x.copy())
                Ps_p.append(P.copy())
                H = np.zeros(self.n)
                H[0] = 1.0
                R = float(seg.q_sigma[i]) ** 2
                S = float(H @ P @ H + R)
                K = (P @ H) / S
                nu = float(seg.q[i] - x[0])
                x = x + K * nu
                IKH = np.eye(self.n) - np.outer(K, H)
                P = IKH @ P @ IKH.T + R * np.outer(K, K)                        # Joseph form
                P = 0.5 * (P + P.T)
                nis.append(nu * nu / S)
                xs_f.append(x.copy())
                Ps_f.append(P.copy())
            eta, Pe = x[2:].copy(), P[2:, 2:].copy()
            seg_out = dict(seg=seg.name, nis=nis, x_filtered=[v.tolist() for v in xs_f],
                           q_sd_filtered=[float(math.sqrt(max(p[0, 0], 0))) for p in Ps_f],
                           psd_ok=bool(all(np.linalg.eigvalsh(p).min() > -1e-12 * max(1.0, abs(p).max())
                                           for p in Ps_f)))
            if smooth and len(xs_f) > 1:
                xs, Ps = [xs_f[-1]], [Ps_f[-1]]
                for i in range(len(xs_f) - 2, -1, -1):
                    G = Ps_f[i] @ Fs[i].T @ np.linalg.pinv(Ps_p[i + 1])
                    xs.insert(0, xs_f[i] + G @ (xs[0] - xs_p[i + 1]))
                    Ps.insert(0, Ps_f[i] + G @ (Ps[0] - Ps_p[i + 1]) @ G.T)
                seg_out["x_smoothed"] = [v.tolist() for v in xs]
                seg_out["q_sd_smoothed"] = [float(math.sqrt(max(p[0, 0], 0))) for p in Ps]
            out_segs.append(seg_out)
            nis_all += nis
        return eta, Pe, out_segs, nis_all


def fit(segments, nspec, *, init=None, eta_sd=1.5, sigma_a_grid=(0.01, 0.03, 0.1, 0.3, 1.0), smooth=True) -> dict:
    """Single forward pass (no data reuse) per sigma_a; sigma_a chosen by |log mean NIS| (recorded)."""
    from ..free_forced.normalized import a_scale
    t0 = time.time()
    segs = [s for s in segments if s.is_free or nspec.forced]
    if not segs:
        return dict(status="NOT_RUN", reason="no segments")
    A = a_scale(segs)
    init = dict(init or {})
    m0 = dict(lam=init.get("lam", 0.5), gam=init.get("gam", 0.3 * A), beta=init.get("beta", 1.0))
    eta0 = np.array([inv_softplus(m0[n]) for n in nspec.names])
    P0 = np.eye(len(eta0)) * eta_sd ** 2
    trials = []
    for g in sigma_a_grid:
        f = NormalizedAEKF(nspec, sigma_a=g * A)
        try:
            eta, Pe, so, nis = f.run(segs, eta0, P0, smooth=smooth)
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as e:
            trials.append(dict(sigma_a=g * A, status=f"failed: {e}"))
            continue
        trials.append(dict(sigma_a=g * A, eta=eta, Pe=Pe, segs=so, nis=float(np.mean(nis)),
                           score=abs(math.log(max(float(np.mean(nis)), 1e-12)))))
    good = [t for t in trials if "eta" in t]
    if not good:
        return dict(status="FAILED", trials=[dict(sigma_a=t["sigma_a"], status=t["status"]) for t in trials])
    best = min(good, key=lambda t: t["score"])
    eta, Pe = best["eta"], best["Pe"]
    est, sd, shrink = {}, {}, {}
    for i, n in enumerate(nspec.names):
        est[n] = softplus(eta[i])
        sd[n] = sigmoid(eta[i]) * math.sqrt(max(Pe[i, i], 0.0))
        shrink[n] = math.sqrt(max(Pe[i, i], 0.0)) / eta_sd
    J = np.diag([sigmoid(e) for e in eta])
    cov = (J @ Pe @ J.T).tolist()
    nis_m = best["nis"]
    psd = all(s["psd_ok"] for s in best["segs"])
    return dict(schema="forceartgs.g9.free_ekf_fit/v1", status="ok" if psd else "covariance_not_psd",
                estimate=est, sd=sd, cov=cov, posterior_over_prior_sd=shrink,
                prior=dict(mean=m0, eta_sd=eta_sd, note="broad softplus prior; NOT the weak-form estimate"),
                uncertainty_label="EKF posterior (delta method through softplus)",
                nis=dict(mean=nis_m, consistent=bool(0.5 <= nis_m <= 2.0), n=sum(len(s["nis"]) for s in best["segs"])),
                sigma_a=best["sigma_a"], sigma_a_selection=dict(grid=[t["sigma_a"] for t in trials],
                                                                mean_nis=[t.get("nis") for t in trials],
                                                                rule="closest mean NIS to 1 (log distance)"),
                per_segment=[dict(seg=s["seg"], mean_nis=float(np.mean(s["nis"])), psd_ok=s["psd_ok"],
                                  q_smoothed=[x[0] for x in s.get("x_smoothed", [])],
                                  q_sd_smoothed=s.get("q_sd_smoothed")) for s in best["segs"]],
                smoother="RTS per segment (future-conditioned; not an independent prediction)",
                runtime_s=time.time() - t0)
