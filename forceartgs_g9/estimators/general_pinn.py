"""GeneralPINN: physics-informed estimation of the FULL general model (G9 PINN/EKF spec section 2).

Per episode k a smooth tanh MLP q_k(t) (float64) is trained on the measured q AND on the residual of the actual
dynamic equation, differentiated with PyTorch autograd in PHYSICAL seconds:

    qd  = d q / d t        (torch.autograd.grad(..., create_graph=True), no detach / finite differences)
    qdd = d qd / d t
    r   = M(theta) qdd + c qd + Fc sgn(qd) + G(q; theta) + b - Q(t)      (torch_model.TorchDynamics.residual)

Loss  L = l_data L_data + l_phys L_phys + l_stick L_stick + l_brk L_breakaway + l_imp L_impulse + l_ic L_ic
  L_data      Huber((q_k(t_j) - q_j) / sigma_q)
  L_phys      Huber(r / s_r) on MOVING collocation points (oversampled where the measured force changes)
  L_stick     at independently detected rest: (qd / s_v)^2 + (qdd / s_a)^2 + relu(|Q - G - b| - Fs)^2 / s_r^2
  L_breakaway one-sided at the first moving sample after rest: relu(Fs - |Q - G - b|)^2 / s_r^2
  L_impulse   weak form over force pulses: (M [qd(tb) - qd(ta)] + int (c qd + Fc sgn + G + b) - int Q) / s_J
              with int Q EXACT on the native 50 Hz ZOH record
  L_weak      the same equation integrated over sliding moving windows [a, b] of weak_window_obs samples:
              (M [qd(b) - qd(a)] + c [q(b) - q(a)] + int (Fc sgn + G + b) - int Q) / (s_r (b - a)); it brings the
              well-measured DISPLACEMENT into the damping term (pointwise strong form alone cannot separate viscous
              from Coulomb friction in near-steady motion with network-derived velocities)
  L_ic        (q_k(t_0) - q_0)^2 / sigma_q^2 (anchors the start)
Physical theta is SHARED by all episodes; trajectory nets are per episode. Scales s_r (effort noise / RMS floor),
s_v, s_a, s_J are declared from the data before training and recorded. Deterministic seeds; a seed ensemble gives
the PINN's own spread (label ENSEMBLE). Reported fit quality is an OPEN-LOOP replay with the frozen theta through
the hybrid simulator -- never the trained q-net.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..dynamics.general_1dof import ModelConfigError, simulate_general
from .general_integral_ls import moving_segments
from .torch_model import DT, TorchDynamics, zoh_integral, zoh_torch


@dataclass
class PINNConfig:
    hidden: int = 32
    layers: int = 3
    n_seeds: int = 3
    steps_data: int = 3000
    steps_joint: int = 4000
    lr_net: float = 3e-3
    lr_theta: float = 1e-2
    n_colloc: int = 300
    phys_ramp_fraction: float = 0.3            # l_phys ramps 0.05 -> l_phys over the first 30 % of joint steps
    event_exclusion_samples: int = 1           # no strong-form collocation within 1 obs sample of a stop / breakaway
    force_change_fraction: float = 0.4
    resample_every: int = 200
    l_data: float = 1.0
    l_phys: float = 1.0
    l_stick: float = 1.0
    l_breakaway: float = 1.0
    l_impulse: float = 1.0
    l_weak: float = 1.0                        # weak (interval-integrated) form on moving windows
    weak_window_obs: int = 3                   # window length in observation intervals
    l_ic: float = 0.1
    phys_scale_rms_floor: float = 0.02         # s_r >= 2 % of the effort RMS (declared numerical floor)
    huber_delta: float = 1.0

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        bad = sorted(set(d) - set(cls.__dataclass_fields__))
        if bad:
            raise ModelConfigError(f"fitting.pinn: unknown keys {bad}")
        return cls(**d)


class _Net(torch.nn.Module):
    def __init__(self, t0, t1, q0, q_scale, hidden, layers, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        dims = [1] + [hidden] * layers + [1]
        self.lin = torch.nn.ModuleList()
        for a, b in zip(dims[:-1], dims[1:]):
            L = torch.nn.Linear(a, b, dtype=DT)
            with torch.no_grad():
                L.weight.copy_(torch.randn(b, a, generator=g, dtype=DT) * math.sqrt(1.0 / a))
                L.bias.zero_()
            self.lin.append(L)
        self.t0, self.T = float(t0), float(max(t1 - t0, 1e-6))
        self.q0, self.qs = float(q0), float(q_scale)

    def forward(self, t):
        x = (2.0 * (t - self.t0) / self.T - 1.0).unsqueeze(-1)       # normalised input; chain rule via autograd
        for L in self.lin[:-1]:
            x = torch.tanh(L(x))
        return self.q0 + self.qs * self.lin[-1](x).squeeze(-1)


def _derivs(net, t):
    t = t.clone().requires_grad_(True)
    q = net(t)
    qd = torch.autograd.grad(q.sum(), t, create_graph=True)[0]
    qdd = torch.autograd.grad(qd.sum(), t, create_graph=True)[0]
    return q, qd, qdd


def _huber(x, d):
    a = torch.abs(x)
    return torch.where(a <= d, 0.5 * x * x, d * (a - 0.5 * d)).mean()


def _savgol_gain(window=7, poly=2):
    from scipy.signal import savgol_coeffs
    g1 = float(np.sqrt(np.sum(savgol_coeffs(window, poly, deriv=1) ** 2)))
    g2 = float(np.sqrt(np.sum(savgol_coeffs(window, poly, deriv=2) ** 2)))
    return g1, g2


class _Episode:
    """Fixed, data-derived quantities of one interaction (no fitted parameter enters their definition; the scale
    s_r uses only the INITIALIZER's M0, c0 and the measured noise)."""

    def __init__(self, it, cfg: PINNConfig, M0=None, c0=None):
        self.it = it
        self.t = torch.as_tensor(it.t_obs, dtype=DT)
        self.q = torch.as_tensor(it.q_obs, dtype=DT)
        self.s = torch.as_tensor(it.q_std, dtype=DT)
        tf = np.asarray(it.effort.t_s, float)
        lev = np.asarray(it.effort.levels, float)
        self.t0, self.t1 = float(it.t_obs[0]), float(it.t_obs[-1])
        segs, v, eps = moving_segments(it.t_obs, it.q_obs, it.q_std)
        mov = np.zeros(len(it.t_obs), bool)
        for a, z in segs:
            mov[a:z] = True
        self.mov_obs = mov
        self.v_eps = eps
        inwin = (tf >= self.t0) & (tf <= self.t1)
        Qw = lev[inwin] if inwin.any() else lev
        rms = float(np.sqrt(np.mean(Qw ** 2))) if len(Qw) else 1.0
        sig = float(it.meta.get("effort_sigma") or 0.0)
        dt_obs = float(np.median(np.diff(it.t_obs)))
        g1, g2 = _savgol_gain()
        sq = float(np.median(it.q_std))
        s_qd, s_qdd = g1 * sq / dt_obs, g2 * sq / dt_obs ** 2
        self.s_r_components = dict(force_noise=sig, inertia_x_sigma_qdd=(M0 or 0.0) * s_qdd,
                                   damping_x_sigma_qd=(c0 or 0.0) * s_qd, rms_floor=cfg.phys_scale_rms_floor * rms,
                                   sigma_qd=s_qd, sigma_qdd=s_qdd,
                                   rule="sqrt(force^2 + (M0 sigma_qdd)^2 + (c0 sigma_qd)^2), floor 2 % effort RMS; "
                                        "sigma_qd / sigma_qdd = noise gains of a 7-sample local quadratic")
        self.s_r = max(math.sqrt(sig ** 2 + ((M0 or 0.0) * s_qdd) ** 2 + ((c0 or 0.0) * s_qd) ** 2),
                       cfg.phys_scale_rms_floor * rms, 1e-9)
        qrange = float(np.ptp(it.q_obs)) or float(np.median(it.q_std)) * 10
        T = self.t1 - self.t0
        self.s_v = max(3 * self.v_eps, 1e-9)
        self.s_a = max(self.s_v / max(np.median(np.diff(it.t_obs)), 1e-3), 1e-9)
        self.q_scale = max(qrange, 5 * float(np.median(it.q_std)))
        # rest runs (>= 3 samples) and breakaway samples, from q only
        self.rest_runs, start = [], None
        for i, m in enumerate(list(~mov) + [False]):
            if m and start is None:
                start = i
            if not m and start is not None:
                if i - start >= 3:
                    self.rest_runs.append((float(it.t_obs[start]), float(it.t_obs[i - 1])))
                start = None
        self.breakaway_t = [float(it.t_obs[i + 1]) for i in range(len(mov) - 1) if (not mov[i]) and mov[i + 1]]
        # force pulses on the native record: contiguous rows |Q| > 3 s_r (description only, not a fitted window)
        act = np.abs(lev) > 3 * self.s_r
        self.pulses = []
        st = None
        for i, a in enumerate(list(act) + [False]):
            if a and st is None:
                st = i
            if not a and st is not None:
                ta = float(tf[st])
                tb = float(tf[i]) if i < len(tf) else float(tf[-1] + np.median(np.diff(tf)))
                if self.t0 < ta and tb < self.t1:
                    self.pulses.append((ta, tb, zoh_integral(it.effort, ta, tb)))
                st = None
        w = max(int(cfg.weak_window_obs), 1)
        self.weak = []
        for a, z in segs:
            for i in range(a, z - w):
                ta, tb = float(it.t_obs[i]), float(it.t_obs[i + w])
                self.weak.append((ta, tb, zoh_integral(it.effort, ta, tb)))
        self.knots = tf[(tf > self.t0) & (tf < self.t1)]
        self.effort = it.effort
        self.cfg_excl = cfg.event_exclusion_samples

    def colloc(self, n, frac, rng):
        n_ch = int(frac * n)
        base = rng.uniform(self.t0, self.t1, n - n_ch)
        ch = np.zeros(0)
        if len(self.knots) and n_ch:
            dQ = np.abs(np.diff(np.r_[self.effort.at(self.knots - 1e-6), 0.0]))[: len(self.knots)]
            w = dQ + 1e-12
            k = rng.choice(len(self.knots), n_ch, p=w / w.sum())
            dt = float(np.median(np.diff(self.effort.t_s))) if len(self.effort.t_s) > 1 else 0.02
            ch = self.knots[k] + rng.uniform(0.05, 0.95, n_ch) * dt * rng.choice([-1, 1], n_ch)
            ch = np.clip(ch, self.t0, self.t1)
        t = np.sort(np.r_[base, ch])
        # never exactly on a ZOH discontinuity
        if len(self.knots):
            d = np.min(np.abs(t[:, None] - self.knots[None, :]), axis=1)
            t = t[d > 1e-6]
        mov = np.interp(t, self.it.t_obs, self.mov_obs.astype(float)) > 0.5
        ev = np.flatnonzero(np.diff(self.mov_obs.astype(int)) != 0)        # stop / breakaway sample indices
        if len(ev):
            dt_obs = float(np.median(np.diff(self.it.t_obs)))
            te = self.it.t_obs[ev] + 0.5 * dt_obs
            near = np.min(np.abs(t[:, None] - te[None, :]), axis=1) <= self.cfg_excl * dt_obs
            mov = mov & ~near
        rest = np.zeros(len(t), bool)
        for a, b in self.rest_runs:
            rest |= (t >= a) & (t <= b)
        return torch.as_tensor(t, dtype=DT), torch.as_tensor(mov & ~rest), torch.as_tensor(rest)


def pinn_losses(dyn: TorchDynamics, eta, ep: _Episode, net, cfg: PINNConfig, tc, mov, rest, w_phys=None) -> dict:
    v = dyn.values(eta)
    M, c, Fc, Fs, b, G = dyn.terms(v)
    out = {}
    qo = net(ep.t)
    out["data"] = _huber((qo - ep.q) / ep.s, cfg.huber_delta)
    out["ic"] = ((qo[0] - ep.q[0]) / ep.s[0]) ** 2
    q, qd, qdd = _derivs(net, tc)
    Q = zoh_torch(ep.effort, tc)
    r = dyn.residual(v, q, qd, qdd, Q)
    out["phys"] = _huber(r[mov] / ep.s_r, cfg.huber_delta) if mov.any() else torch.zeros((), dtype=DT)
    if rest.any():
        net_eff = torch.abs(Q[rest] - G(q[rest]) - b)
        thr = Fs if Fs is not None else (Fc if dyn.has["coulomb"] else None)
        st = ((qd[rest] / ep.s_v) ** 2 + (qdd[rest] / ep.s_a) ** 2).mean()
        if thr is not None:
            st = st + (torch.relu(net_eff - thr) ** 2).mean() / ep.s_r ** 2
        out["stick"] = st
    else:
        out["stick"] = torch.zeros((), dtype=DT)
    if Fs is not None and ep.breakaway_t:
        tb = torch.as_tensor(ep.breakaway_t, dtype=DT)
        qb = net(tb)
        Qb = zoh_torch(ep.effort, tb)
        out["breakaway"] = (torch.relu(Fs - torch.abs(Qb - G(qb) - b)) ** 2).mean() / ep.s_r ** 2
    else:
        out["breakaway"] = torch.zeros((), dtype=DT)
    imp = []
    for ta, tb, JQ in ep.pulses:
        tt = torch.linspace(ta, tb, 33, dtype=DT)
        qq, qqd, _ = _derivs(net, tt)
        integ = torch.trapz(c * qqd + Fc * torch.sign(qqd) + G(qq) + b, tt)
        s_J = ep.s_r * (tb - ta) + 1e-12
        imp.append(((M * (qqd[-1] - qqd[0]) + integ - JQ) / s_J) ** 2)
    out["impulse"] = torch.stack(imp).mean() if imp else torch.zeros((), dtype=DT)
    if ep.weak:
        A = torch.as_tensor([w[0] for w in ep.weak], dtype=DT)
        B = torch.as_tensor([w[1] for w in ep.weak], dtype=DT)
        JQ = torch.as_tensor([w[2] for w in ep.weak], dtype=DT)
        k = 9
        s01 = torch.linspace(0, 1, k, dtype=DT)
        tt = (A[:, None] + (B - A)[:, None] * s01[None, :]).reshape(-1)
        qq, qqd, _ = _derivs(net, tt)
        qq, qqd = qq.reshape(-1, k), qqd.reshape(-1, k)
        dt_ = (B - A)[:, None] * s01[None, :]
        integ = torch.trapz(Fc * torch.sign(qqd) + G(qq) + b, dt_, dim=1)
        rw = M * (qqd[:, -1] - qqd[:, 0]) + c * (qq[:, -1] - qq[:, 0]) + integ - JQ
        out["weak"] = _huber(rw / (ep.s_r * (B - A)), cfg.huber_delta)
    else:
        out["weak"] = torch.zeros((), dtype=DT)
    lp = cfg.l_phys if w_phys is None else w_phys
    out["total"] = (cfg.l_data * out["data"] + lp * out["phys"] + cfg.l_stick * out["stick"]
                    + cfg.l_breakaway * out["breakaway"] + cfg.l_impulse * out["impulse"] + cfg.l_weak * out["weak"]
                    + cfg.l_ic * out["ic"])
    return out


def _fit_one(interactions, spec, theta0, cfg: PINNConfig, seed: int, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dyn = TorchDynamics(spec)
    ph0 = spec.physical(np.asarray(theta0, float))
    eps = [_Episode(it, cfg, ph0.get("M"), ph0.get("c")) for it in interactions]
    nets = [_Net(e.t0, e.t1, float(e.it.q_obs[0]), e.q_scale, cfg.hidden, cfg.layers, seed + 17 * k)
            for k, e in enumerate(eps)]
    eta = dyn.eta0(theta0)
    opt_n = torch.optim.Adam([p for n in nets for p in n.parameters()], lr=cfg.lr_net)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt_n, max(cfg.steps_data, 1), eta_min=cfg.lr_net * 0.05)
    for _ in range(cfg.steps_data):                                   # data warm-up (theta frozen)
        opt_n.zero_grad()
        L = sum(_huber((n(e.t) - e.q) / e.s, cfg.huber_delta) for n, e in zip(nets, eps))
        L.backward()
        opt_n.step()
        sch.step()
    opt = torch.optim.Adam([{"params": [p for n in nets for p in n.parameters()], "lr": cfg.lr_net},
                            {"params": list(eta.values()), "lr": cfg.lr_theta}])
    cols = [e.colloc(cfg.n_colloc, cfg.force_change_fraction, rng) for e in eps]
    hist = []
    for step in range(cfg.steps_joint):
        if step and step % cfg.resample_every == 0:
            cols = [e.colloc(cfg.n_colloc, cfg.force_change_fraction, rng) for e in eps]
        opt.zero_grad()
        ramp = min(1.0, step / max(cfg.phys_ramp_fraction * cfg.steps_joint, 1.0))
        wp = cfg.l_phys * (0.05 + 0.95 * ramp)
        parts = [pinn_losses(dyn, eta, e, n, cfg, *c, w_phys=wp) for e, n, c in zip(eps, nets, cols)]
        L = sum(p["total"] for p in parts)
        L.backward()
        opt.step()
        if step % 500 == 0 or step == cfg.steps_joint - 1:
            hist.append(dict(step=step, total=float(L), **{k: float(sum(p[k] for p in parts)) for k in
                                                           ("data", "phys", "stick", "breakaway", "impulse",
                                                            "weak", "ic")}))
    theta = dyn.theta_from_eta({k: v.detach() for k, v in eta.items()})
    return theta, hist, eps, nets


def open_loop_replay(spec, theta, interactions, max_step=1e-3) -> dict:
    """Frozen-theta hybrid-simulator replay from (q0 = first observation, v0 from a local fit) -- the PINN's
    reported fit quality, independent of its trajectory nets."""
    out = []
    for it in interactions:
        v0 = float(np.polyfit(it.t_obs[:4] - it.t_obs[0], it.q_obs[:4], 1)[0])
        try:
            r = simulate_general(spec.sim(theta), it.effort, it.t_obs, t0=float(it.t_obs[0]), q0=float(it.q_obs[0]),
                                 v0=v0, max_step=max_step)
            e = (r.q - it.q_obs) / it.q_std
            out.append(dict(interaction=it.name, rmse_over_sigma=float(np.sqrt(np.mean(e ** 2)))))
        except ModelConfigError as ex:
            out.append(dict(interaction=it.name, error=str(ex)))
    return dict(per_interaction=out, note="open-loop hybrid-simulator replay with the frozen PINN theta (v0 from a "
                                          "4-sample local fit); not the trained trajectory nets")


def general_pinn_fit(interactions, spec, theta0, cfg: PINNConfig | None = None, log=None) -> dict:
    cfg = cfg or PINNConfig()
    t_start = time.time()
    thetas, hists = [], []
    eps = None
    for s in range(cfg.n_seeds):
        th, h, eps, nets = _fit_one(interactions, spec, theta0, cfg, seed=s)
        thetas.append(th)
        hists.append(h)
    T = np.array(thetas)
    mean, std = T.mean(0), T.std(0, ddof=1) if len(T) > 1 else np.full(T.shape[1], np.nan)
    best = int(np.argmin([h[-1]["total"] for h in hists]))
    th = T[best]
    est = {n: dict(value=float(th[j]), ensemble_mean=float(mean[j]), ensemble_std=float(std[j]),
                   std=float(std[j]), uncertainty_label="ENSEMBLE (seeds; not a calibrated posterior)",
                   at_lower_bound=bool(th[j] < 1e-9))
           for j, n in enumerate(spec.names)}
    return dict(status="ok", method="general_pinn", estimates=est, physical=spec.physical(th), x=th.tolist(),
                seeds=list(range(cfg.n_seeds)), per_seed_theta=T.tolist(), best_seed=best,
                loss_history=hists, settings=asdict(cfg),
                scales={e.it.name: dict(s_r=e.s_r, s_r_components=e.s_r_components, s_v=e.s_v, s_a=e.s_a,
                                        pulses=[(a, b, J) for a, b, J in e.pulses],
                                        rest_runs=e.rest_runs, breakaway_t=e.breakaway_t) for e in eps},
                replay=open_loop_replay(spec, th, interactions), runtime_s=time.time() - t_start,
                differentiation="torch.autograd (create_graph=True) in physical seconds; no finite differences",
                note="PINN parameters from the seed with the lowest final total loss; ensemble spread reported")
