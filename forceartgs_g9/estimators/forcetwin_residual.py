"""Structured mechanism residual (ForceTwin section III-C, adapted): fitted ONLY after the physical baseline, with the
baseline coefficients FROZEN.

    r_mech(q, qdot) = g_theta(q) + c_phi(q, |qdot|, sgn qdot) qdot,     c_phi = softplus(.) >= 0  (dissipative)

g_theta: sign-unconstrained configuration-dependent load (closer / spring / latch / unmodelled gravity -- NOT
identified as any of them without independent evidence). c_phi: separate positive- / negative-velocity heads
(direction-dependent dissipation, never energy injection: -c qdot^2 <= 0). No acceleration or measured-torque input.

Protocol (paper: full-batch AdamW lr 1e-3, weight decay 1e-3, <= 1200 epochs, patience 120, 75/25 RANDOM split).
Deliberate change: the split is BLOCKED CHRONOLOGICAL (first 75 % of each window / last 25 %) because consecutive
rows are strongly correlated -- this is a protocol improvement, not a reproduction of the paper.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .forcetwin_linear import direct_rows
from .torch_model import on_torch_device

DT = torch.float64
ABLATIONS = ("none", "g", "damping", "both")


class _MLP(torch.nn.Module):
    def __init__(self, n_in, h, seed, out_bias=None, legacy_init=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.l1 = torch.nn.Linear(n_in, h, dtype=DT)
        self.l2 = torch.nn.Linear(h, 1, dtype=DT)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g, dtype=DT, device="cpu") * 0.3)
        if not legacy_init:                       # start EXACTLY at the physical baseline (zero output layer)
            with torch.no_grad():
                self.l2.weight.zero_()
                self.l2.bias.fill_(0.0 if out_bias is None else out_bias)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x))).squeeze(-1)


class MechanismResidual(torch.nn.Module):
    DAMPING_BIAS0 = -8.0      # softplus(-8) ~ 3e-4: damping heads start near zero (Adam steps are scale-free)

    def __init__(self, mode, q_range, v_scale, hidden=8, seed=0, legacy_init=False):
        super().__init__()
        self.mode = mode
        self.q_lo, self.q_hi = map(float, q_range)
        self.v_scale = float(v_scale)
        b0 = None if legacy_init else self.DAMPING_BIAS0
        self.g = _MLP(1, hidden, seed, None, legacy_init) if mode in ("g", "both") else None
        self.cpos = _MLP(2, hidden, seed + 1, b0, legacy_init) if mode in ("damping", "both") else None
        self.cneg = _MLP(2, hidden, seed + 2, b0, legacy_init) if mode in ("damping", "both") else None

    def _qn(self, q):
        return (2 * (q - self.q_lo) / max(self.q_hi - self.q_lo, 1e-9) - 1).unsqueeze(-1)

    def c(self, q, v):
        x = torch.cat([self._qn(q), (v.abs() / self.v_scale).unsqueeze(-1)], -1)
        return torch.where(v >= 0, torch.nn.functional.softplus(self.cpos(x)),
                           torch.nn.functional.softplus(self.cneg(x)))

    def forward(self, q, v):
        out = torch.zeros_like(q)
        if self.g is not None:
            out = out + self.g(self._qn(q))
        if self.cpos is not None:
            out = out + self.c(q, v) * v
        return out

    def n_parameters(self):
        return int(sum(p.numel() for p in self.parameters()))


def _rows_with_state(rec, beta, grav=None, smoother="spline"):
    """Direct rows + the (q, qdot) state: target r = tau - (I qdd + mu sgn qd + b qd + I kappa(q))."""
    from ..forcetwin.derivatives import spline, local_poly
    Z, y, s, meta = direct_rows(rec, smoother, grav=grav)
    I, mu, b = beta
    base = Z @ np.array([I, mu, b])
    # state at the same rows
    qs, vs = [], []
    by_win = {}
    for name, tf, *_ in meta:
        by_win.setdefault(name, []).append(tf)
    for name, fs, qsl, src in rec.windows:
        if name not in by_win:
            continue
        tt = np.asarray(by_win[name])
        d = (spline if smoother == "spline" else local_poly)(rec.q_t_s[qsl], rec.q[qsl], rec.q_sigma[qsl], tt)
        qs += list(d["q"])
        vs += list(d["qd"])
    win = np.array([m[0] for m in meta])
    tt = np.array([m[1] for m in meta])
    return dict(q=np.asarray(qs), v=np.asarray(vs), r=y - base, tau=y, base=base, w=1 / np.maximum(s, 1e-12),
                win=win, t=tt, Z=Z)


def _split(win, t, frac=0.75):
    tr = np.zeros(len(t), bool)
    for w in np.unique(win):
        m = np.flatnonzero(win == w)
        tt = t[m]
        cut = tt.min() + frac * (tt.max() - tt.min())
        tr[m[tt <= cut]] = True
    return tr, ~tr


@on_torch_device
def fit_residual(rec, beta, mode, *, grav=None, hidden=8, seed=0, lr=1e-3, wd=1e-3, epochs=1200, patience=120,
                 lam_smooth=1e-3, orthogonalize=True, legacy_init=False, min_improvement=0.05) -> dict:
    """Train one residual head on the frozen baseline. Model selection keeps the NO-RESIDUAL baseline as a candidate
    (selected unless a trained state is better on the blocked validation split), the output layer starts at zero,
    and -- with ``orthogonalize`` -- the residual prediction is projected out of the span of the physical regressors
    [qdd + kappa, sgn qd, qd] (whitened), so it cannot absorb inertia / Coulomb / viscous effects (the constant part
    of c_phi would otherwise duplicate b). ``accepted`` needs a validation improvement >= ``min_improvement``."""
    D = _rows_with_state(rec, beta, grav)
    if len(D["r"]) < 20:
        return dict(mode=mode, status="NOT_RUN", reason=f"{len(D['r'])} rows")
    tr, va = _split(D["win"], D["t"])
    q, v, r, w = (torch.as_tensor(D[k], dtype=DT) for k in ("q", "v", "r", "w"))
    trm, vam = torch.as_tensor(tr), torch.as_tensor(va)
    if orthogonalize:
        Zw = D["Z"] * D["w"][:, None]
        Pm = torch.as_tensor(np.eye(len(Zw)) - Zw @ np.linalg.pinv(Zw), dtype=DT)
        Winv = torch.as_tensor(1.0 / D["w"], dtype=DT)
        Wd = torch.as_tensor(D["w"], dtype=DT)
        proj = lambda p_: Winv * (Pm @ (Wd * p_))            # projection in the whitened metric
    else:
        proj = lambda p_: p_

    def wrms(pred, m):
        e = ((r - pred) * w)[m]
        return float(torch.sqrt(torch.mean(e ** 2))) if m.any() else None
    zero = torch.zeros_like(r)
    base_tr, base_va = wrms(zero, trm), wrms(zero, vam)
    if mode == "none":
        return dict(mode=mode, status="OK", train_whitened_rms=base_tr, val_whitened_rms=base_va, n_parameters=0,
                    selected="baseline", accepted=False)
    net = MechanismResidual(mode, (float(D["q"].min()), float(D["q"].max())),
                            max(float(np.abs(D["v"]).max()), 1e-6), hidden, seed, legacy_init=legacy_init)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    # the no-residual baseline is the incumbent (legacy: the first trained state)
    best, best_state, wait, hist = (base_va if not legacy_init else math.inf), None, 0, []
    qg = torch.linspace(float(D["q"].min()), float(D["q"].max()), 64, dtype=DT, requires_grad=True)
    for ep in range(epochs):
        opt.zero_grad()
        pred = proj(net(q, v))
        loss = torch.mean(((r - pred) * w)[trm] ** 2)
        if net.g is not None and lam_smooth > 0:                # smoothness of the configuration load
            gq = net.g(net._qn(qg))
            dg = torch.autograd.grad(gq.sum(), qg, create_graph=True)[0]
            loss = loss + lam_smooth * torch.mean(dg ** 2) / max(float(torch.mean((r * w) ** 2)), 1e-12)
        loss.backward()
        opt.step()
        with torch.no_grad():
            vl = wrms(proj(net(q, v)), vam)
        hist.append((ep, float(loss), vl))
        if vl is not None and vl < best - 1e-9:
            best, wait = vl, 0
            best_state = {k: x.clone() for k, x in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    selected = "trained" if best_state is not None else "baseline"
    if best_state is not None:
        net.load_state_dict(best_state)
    with torch.no_grad():
        pred = proj(net(q, v)) if selected == "trained" else torch.zeros_like(r)
    P = pred.cpu().numpy()
    Zw_, Pw_ = D["Z"] * D["w"][:, None], P * D["w"]
    nz = np.linalg.norm(Zw_, axis=0) * max(float(np.linalg.norm(Pw_)), 1e-300)
    ortho = float(np.max(np.abs(Zw_.T @ Pw_) / np.where(nz > 0, nz, 1.0))) if np.any(P) else 0.0
    # overlap with the physical regressor space (non-unique decomposition diagnostic)
    Zc = D["Z"] - D["Z"].mean(0)
    corr = {n: float(np.corrcoef(Zc[:, j], P)[0, 1]) if np.std(Zc[:, j]) > 0 and np.std(P) > 0 else None
            for j, n in enumerate(("qdd", "sgn_qdot", "qdot"))}
    diss = None
    if net.cpos is not None:
        with torch.no_grad():
            cval = net.c(q, v).cpu().numpy()
        diss = dict(min_c=float(cval.min()), max_c=float(cval.max()),
                    dissipated_power_max=float(np.max(-cval * D["v"] ** 2)), note="-c qdot^2 <= 0 by construction "
                    "(before the orthogonal projection, which removes the part collinear with b qdot)")
    dt = float(np.median(np.diff(np.sort(D["t"])))) if len(D["t"]) > 1 else 0.02
    work = float(np.sum(P * D["v"]) * dt)
    va_rms = wrms(pred, vam)
    impr = (1 - va_rms / base_va) if (base_va and va_rms is not None) else None
    return dict(mode=mode, status="OK", train_whitened_rms=wrms(pred, trm), val_whitened_rms=va_rms,
                baseline_train_whitened_rms=base_tr, baseline_val_whitened_rms=base_va, n_parameters=net.n_parameters(),
                epochs_run=len(hist), best_val=best, selected=selected,
                accepted=bool(selected == "trained" and impr is not None and impr >= min_improvement),
                acceptance_rule=f"trained state selected AND validation improvement >= {min_improvement:.0%} "
                                "(within-record split; export still needs an independent episode)",
                orthogonalized=bool(orthogonalize), max_abs_cosine_with_physical_regressors=ortho,
                overlap_with_physical_regressors=corr, dissipation=diss,
                residual_work_proxy=work, domain=dict(q=[float(D["q"].min()), float(D["q"].max())],
                                                      abs_qdot_max=float(np.abs(D["v"]).max()),
                                                      policy="clamp q to the domain; abstain outside |qdot| range"),
                architecture=dict(hidden=hidden, activation="tanh", heads="g(q); softplus c+(q,|v|), c-(q,|v|)",
                                  inputs="q (normalised), |qdot|/v_scale, sgn(qdot) as a discrete branch",
                                  init="legacy random" if legacy_init else
                                  f"zero output layer (g); damping bias {MechanismResidual.DAMPING_BIAS0}"),
                optimiser=dict(name="AdamW full batch", lr=lr, weight_decay=wd, max_epochs=epochs, patience=patience,
                               smoothness_penalty=lam_smooth, seed=seed),
                split="blocked chronological 75/25 within each window (correlated rows: NOT the paper's random split)",
                _net=net)


def ablations(rec, beta, *, grav=None, independent_episodes=0, **kw) -> dict:
    res = {m: fit_residual(rec, beta, m, grav=grav, **kw) for m in ABLATIONS}
    for v in res.values():
        v.pop("_net", None)
    base = res["none"].get("val_whitened_rms")
    for m, v in res.items():
        if v.get("status") == "OK" and base and v.get("val_whitened_rms") is not None:
            v["val_improvement_vs_none"] = float(1 - v["val_whitened_rms"] / base)
    label = "EXPLORATORY_TRAIN_ONLY" if independent_episodes == 0 else "VALIDATED_ON_INDEPENDENT_EPISODES"
    acc = [m for m, v in res.items() if v.get("accepted")]
    sel = min(acc, key=lambda m: res[m]["val_whitened_rms"]) if acc else "none"
    return dict(schema="forceartgs.g9.forcetwin_residual/v1", frozen_baseline=dict(I_eff=beta[0], mu=beta[1], b=beta[2]),
                ablations=res, selected_mode=sel, validation_label=label, independent_episodes=independent_episodes,
                selection_rule="the no-residual baseline unless an accepted head improves the blocked validation split",
                export=("NOT_EXPORTED (no independent episode)" if independent_episodes == 0 else
                        "RUNTIME_PLUGIN_REQUIRED (not representable in a bare URDF)"),
                decomposition="non-unique in general (c_phi's constant part overlaps b, g_theta overlaps gravity / "
                              "bias): the residual is projected orthogonal to the physical regressors by default; the "
                              "physical coefficients stay frozen")
