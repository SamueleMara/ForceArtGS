"""Normalized physics-informed network (spec 6.2) -- the normalized joint equation IS the network's differentiable
physics residual (not labels from another estimator).

One continuous trajectory MLP q_hat_k(t) per segment (the existing GeneralPINN ``_Net``), SHARED physical rates.
Time derivatives by torch.autograd (create_graph) in physical units.

Phase A (free; NO trainable M / rho / beta):
    r_free = qdd_hat + lam qd_hat + gam s + a_g(q_hat)                        (s = the segment's fixed sign)
    loss   = data (q_obs, sigma_eff) + strong-form r_free on interior collocation (edges excluded)
             + integrated weak form  dqd_hat + lam dq_hat + gam s dt + int a_g(q_hat) dt  on sub-windows
Phase B (only with trusted forced segments): beta = softplus(eta_beta) > 0 enters  r_forced = r_free - beta Q(t)
    (exact ZOH impulse in the weak form); the free rates are refined under an anchor to their phase-A values.

Never torch.sign() at zero speed: segments are monotonic by construction (sign fixed). Deterministic seeds; the seed
spread is reported but is NOT a posterior.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .general_pinn import DT, _derivs, _Net


@dataclass
class NPINNConfig:
    hidden: int = 32
    layers: int = 3
    epochs_warm: int = 600
    epochs: int = 2500
    ramp: int = 300
    lr: float = 3e-3
    n_colloc: int = 160
    edge_frac: float = 0.06
    n_weak: int = 6
    w_data: float = 1.0
    w_strong: float = 1.0
    w_weak: float = 1.0
    seeds: tuple = (0, 1, 2)
    sigma_floor: float = 0.0          # the pipeline passes its declared floor; 0 = the measured sigma
    phase_b_epochs: int = 1500
    rate_anchor: float = 1.0
    lbfgs_iters: int = 300            # full-batch L-BFGS polish after Adam (same objective)
    baseline_degree: int = 2          # q_hat = WLS polynomial baseline + q_scale * MLP (the net learns the remainder)

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        if "seeds" in d:
            d["seeds"] = tuple(d["seeds"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _inv_softplus(y):
    y = max(float(y), 1e-9)
    return y + math.log(-math.expm1(-y)) if y < 30 else y


class _Seg:
    def __init__(self, seg, cfg, seed, forced):
        self.s = seg
        self.t = torch.as_tensor(seg.t, dtype=DT)
        self.q = torch.as_tensor(seg.q, dtype=DT)
        self.se = torch.as_tensor(np.maximum(seg.q_sigma, cfg.sigma_floor), dtype=DT)
        T = float(seg.t[-1] - seg.t[0])
        e = cfg.edge_frac * T
        self.tc = torch.linspace(float(seg.t[0]) + e, float(seg.t[-1]) - e, cfg.n_colloc, dtype=DT)
        qs = max(float(np.ptp(seg.q)), 10 * float(np.median(seg.q_sigma)), 1e-6)
        self.net = _TrajNet(seg, cfg, qs, seed)
        self.sign = float(seg.sign)
        self.forced = forced and not seg.is_free
        n = cfg.n_colloc
        cuts = np.linspace(0, n - 1, cfg.n_weak + 1).astype(int)
        self.win = [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
        tcn = self.tc.numpy()
        if self.forced:
            from .torch_model import zoh_torch
            eff = seg.effort
            self.Qc = zoh_torch(eff, self.tc)
            self.Jw = torch.as_tensor([seg.J(float(tcn[a]), float(tcn[b]))[0] for a, b in self.win], dtype=DT)
        else:
            self.Qc, self.Jw = None, None


class _TrajNet(torch.nn.Module):
    """q_hat(t) = p(t) + MLP(t): p = weighted LS polynomial of the observations (fixed), MLP = the existing GeneralPINN
    trajectory net (output scale ~ the residual size). Derivatives of both parts come from autograd."""

    def __init__(self, seg, cfg, q_scale, seed):
        super().__init__()
        t0, t1 = float(seg.t[0]), float(seg.t[-1])
        self.tm, self.T = 0.5 * (t0 + t1), max(0.5 * (t1 - t0), 1e-6)
        x = (np.asarray(seg.t) - self.tm) / self.T
        d = int(min(cfg.baseline_degree, len(x) - 1))
        A = np.vander(x, d + 1, increasing=True)
        w = 1.0 / np.asarray(seg.q_sigma) ** 2
        c = np.linalg.solve((A.T * w) @ A, (A.T * w) @ np.asarray(seg.q))
        self.register_buffer("c", torch.as_tensor(c, dtype=DT))
        res = float(np.max(np.abs(np.asarray(seg.q) - A @ c)))
        self.mlp = _Net(t0, t1, 0.0, max(res, 3 * float(np.median(seg.q_sigma)), 1e-9), cfg.hidden, cfg.layers, seed)

    def forward(self, t):
        x = (t - self.tm) / self.T
        p = sum(self.c[k] * x ** k for k in range(len(self.c)))
        return p + self.mlp(t)


def _losses(S, rates, ag, A):
    q_hat = S.net(S.t)
    data = torch.mean(((q_hat - S.q) / S.se) ** 2)
    q, qd, qdd = _derivs(S.net, S.tc)
    r = qdd + rates["lam"] * qd + rates["gam"] * S.sign + ag(q)
    if S.forced:
        r = r - rates["beta"] * S.Qc
    strong = torch.mean((r / A) ** 2)
    ws = []
    tc = S.tc
    for k, (a, b) in enumerate(S.win):
        tt, qq = tc[a:b + 1], q[a:b + 1]
        integ = torch.trapz(ag(qq), tt)
        w = (qd[b] - qd[a]) + rates["lam"] * (q[b] - q[a]) + rates["gam"] * S.sign * (tt[-1] - tt[0]) + integ
        if S.forced:
            w = w - rates["beta"] * S.Jw[k]
        ws.append(w / (A * (tt[-1] - tt[0])))
    weak = torch.mean(torch.stack(ws) ** 2) if ws else torch.zeros((), dtype=DT)
    return data, strong, weak


def _train(segments, nspec, cfg, seed, init, *, epochs, warm, anchor=None):
    from ..free_forced.normalized import a_scale
    torch.manual_seed(seed)
    names = nspec.names
    A = max(a_scale(segments), 1e-6)
    segs = [_Seg(s, cfg, seed + 17 * k, nspec.forced) for k, s in enumerate(segments)]
    eta = torch.nn.Parameter(torch.as_tensor([_inv_softplus(init[n]) for n in names], dtype=DT))
    params = [eta] + [p for S in segs for p in S.net.parameters()]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    ag = nspec.gravity.torch
    hist, grad_seen = [], {n: 0.0 for n in names}
    nd = sum(len(S.s.t) for S in segs)

    def objective(ramp):
        sp = torch.nn.functional.softplus(eta)
        rates = {n: sp[i] for i, n in enumerate(names)}
        rates.setdefault("lam", torch.zeros((), dtype=DT))
        rates.setdefault("gam", torch.zeros((), dtype=DT))
        rates.setdefault("beta", torch.zeros((), dtype=DT))
        L_d = L_s = L_w = torch.zeros((), dtype=DT)
        for S in segs:
            d, s_, w_ = _losses(S, rates, ag, A)
            L_d, L_s, L_w = L_d + d * len(S.s.t) / nd, L_s + s_ / len(segs), L_w + w_ / len(segs)
        loss = cfg.w_data * L_d + ramp * (cfg.w_strong * L_s + cfg.w_weak * L_w)
        if anchor is not None:
            for n, (m, sd) in anchor.items():
                if n in names:
                    loss = loss + cfg.rate_anchor * ((rates[n] - m) / max(sd, 1e-9)) ** 2
        return loss, L_d, L_s, L_w

    for ep in range(epochs):
        opt.zero_grad()
        ramp = 0.0 if ep < warm else min(1.0, (ep - warm + 1) / max(cfg.ramp, 1))
        loss, L_d, L_s, L_w = objective(ramp)
        loss.backward()
        if eta.grad is not None:
            for i, n in enumerate(names):
                grad_seen[n] = max(grad_seen[n], float(abs(eta.grad[i])))
        opt.step()
        if ep % 100 == 0 or ep == epochs - 1:
            hist.append(dict(ep=ep, stage="adam", loss=float(loss), data=float(L_d), strong=float(L_s),
                             weak=float(L_w)))
    if cfg.lbfgs_iters > 0 and epochs > warm:
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg.lbfgs_iters, history_size=50,
                               line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)

        def closure():
            lb.zero_grad()
            l_, *_ = objective(1.0)
            l_.backward()
            return l_
        try:
            lb.step(closure)
        except RuntimeError as e:                       # a failed line search keeps the Adam state
            hist.append(dict(stage="lbfgs", error=str(e)))
        loss, L_d, L_s, L_w = objective(1.0)
        hist.append(dict(ep=epochs, stage="lbfgs", loss=float(loss), data=float(L_d), strong=float(L_s),
                         weak=float(L_w), iters=cfg.lbfgs_iters))
    sp = torch.nn.functional.softplus(eta).detach().numpy()
    est = {n: float(sp[i]) for i, n in enumerate(names)}
    traj = {S.s.name: dict(t=S.s.t.tolist(), q_hat=S.net(S.t).detach().numpy().tolist()) for S in segs}
    return dict(seed=seed, estimate=est, final=hist[-1], history=hist, max_abs_eta_grad=grad_seen,
                trajectories=traj, accel_scale=A)


def fit_free(segments, nspec, cfg: NPINNConfig | None = None, init=None, log=None) -> dict:
    """Phase A. Returns the selected seed (lowest final training objective) and the seed spread."""
    cfg = cfg or NPINNConfig()
    t0 = time.time()
    fs = nspec.free()
    segs = [s for s in segments if s.is_free]
    if not segs:
        return dict(status="NOT_RUN", reason="no free segment")
    from ..free_forced.normalized import a_scale
    A = a_scale(segs)
    init = dict(lam=0.5, gam=0.3 * A) | {k: max(float(v), 1e-6) for k, v in (init or {}).items()
                                         if k in ("lam", "gam")}
    runs = []
    for sd in cfg.seeds:
        if log:
            log(f"[npinn] phase A seed {sd}")
        runs.append(_train(segs, fs, cfg, sd, init, epochs=cfg.epochs, warm=cfg.epochs_warm))
    best = min(runs, key=lambda r: r["final"]["loss"])
    spread = {n: float(np.std([r["estimate"][n] for r in runs], ddof=1)) if len(runs) > 1 else None
              for n in fs.names}
    return dict(schema="forceartgs.g9.free_pinn_fit/v1", status="ok", phase="A_free", parameter_names=fs.names,
                estimate=best["estimate"], selected_seed=best["seed"],
                seed_selection="lowest final training objective (data + physics); NOT a held-out criterion",
                seed_spread=spread, uncertainty_label="SEED_ENSEMBLE_SPREAD (not a posterior)",
                seeds=[dict(seed=r["seed"], estimate=r["estimate"], final=r["final"],
                            max_abs_eta_grad=r["max_abs_eta_grad"]) for r in runs],
                history=best["history"], trajectories=best["trajectories"], init=init,
                config=asdict(cfg), runtime_s=time.time() - t0,
                architecture=dict(net="WLS polynomial baseline + GeneralPINN _Net (tanh MLP, normalised time)",
                                  hidden=cfg.hidden, layers=cfg.layers, baseline_degree=cfg.baseline_degree,
                                  per_segment=True, optimiser=f"Adam {cfg.epochs} + L-BFGS {cfg.lbfgs_iters}"),
                beta_trainable=False)


def fit_forced(free_segments, forced_segments, nspec, free_result, cfg: NPINNConfig | None = None,
               rate_sd=None, beta0=1.0, log=None) -> dict:
    """Phase B: beta only with trusted forced segments; free rates refined under an anchor."""
    cfg = cfg or NPINNConfig()
    if not forced_segments:
        return dict(status="NOT_RUN", reason="no trusted forced segment (beta not instantiated)")
    t0 = time.time()
    fs = nspec.with_forced()
    e = (free_result or {}).get("estimate") or {}
    init = dict(lam=e.get("lam", 0.5), gam=e.get("gam", 0.1), beta=beta0)
    anchor = None
    if e:
        sd = rate_sd or {}
        anchor = {n: (e[n], sd.get(n) or max(0.5 * e[n], 1e-3)) for n in ("lam", "gam") if n in e}
    runs = []
    for sd_ in cfg.seeds:
        if log:
            log(f"[npinn] phase B seed {sd_}")
        runs.append(_train(list(free_segments) + list(forced_segments), fs, cfg, sd_, init,
                           epochs=cfg.phase_b_epochs, warm=min(cfg.epochs_warm, cfg.phase_b_epochs // 3),
                           anchor=anchor))
    best = min(runs, key=lambda r: r["final"]["loss"])
    return dict(schema="forceartgs.g9.forced_pinn_fit/v1", status="ok", phase="B_forced", parameter_names=fs.names,
                estimate=best["estimate"], selected_seed=best["seed"],
                seed_spread={n: float(np.std([r["estimate"][n] for r in runs], ddof=1)) if len(runs) > 1 else None
                             for n in fs.names},
                uncertainty_label="SEED_ENSEMBLE_SPREAD (not a posterior)", anchor=anchor,
                seeds=[dict(seed=r["seed"], estimate=r["estimate"], final=r["final"],
                            max_abs_eta_grad=r["max_abs_eta_grad"]) for r in runs],
                history=best["history"], config=asdict(cfg), runtime_s=time.time() - t0)
