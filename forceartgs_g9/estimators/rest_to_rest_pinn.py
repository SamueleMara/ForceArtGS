"""Rest-to-rest PINN ablations of the two-term model  M q'' + c q' = Q  (opt-in; the historical GeneralPINN and its
``_Net`` are untouched and run as the LEGACY baseline).

Hard rest-to-rest trajectory (Eq. 9), s = (t - t0) / (T - t0), h(s) = 3 s^2 - 2 s^3:

    q_psi(t) = q0 + (qT - q0) h(s) + S_q s^2 (1 - s)^2 f_psi(s)

so q(t0) = q0, q(T) = qT, qdot(t0) = qdot(T) = 0 hold EXACTLY for any weights (S_q: fixed scale from the measured q range
and uncertainty, never optimized; q0 / qT: measured plateau anchors, never trainable). Zero net displacement still
allows an out-and-back motion through the correction term.

Derivative-free physics (Eq. 10), every visual time t_i in (t0, T]:

    r2(t_i) = M [q_psi(t_i) - q0] + c int_{t0}^{t_i} [q_psi - q0] ds - int_{t0}^{t_i} (t_i - s) Q(s) ds

the q integral by a DIFFERENTIABLE torch cumulative trapezoid on fixed nodes (uniform grid + every visual time) over the
actual interval; the Q term is fixed measured input precomputed exactly on the native ZOH rows. No q'' / autograd
acceleration in this residual, nothing detached. Scales s2_i = sqrt(diag Cov) of the DECLARED noise propagated through
the same integrals (rest_to_rest_integral.residual_covariance), fixed before training, not trainable.

Variants (identical width / steps / seeds / data / input):
  LEGACY_PINN_STRONG            historical GeneralPINN (its net, strong + stick + impulse + weak terms, own segmentation)
  PINN_HARD_BC_STRONG           hard BC net + strong residual  M q'' + c q' - Q  (autograd), no contact / stick terms
  PINN_HARD_BC_DOUBLE_INTEGRAL  hard BC net + ONLY the double-integral residual
  PINN_HARD_BC_HYBRID           hard BC net + strong + double integral (declared fixed weights; correlated terms)
  PINN_SOFT_BC_DOUBLE_INTEGRAL  legacy-style free net + soft endpoint penalties + double integral (control)
Physics quality is judged by OPEN-LOOP replay of the frozen (M, c) from (q0, v0 = 0) on [t0, T] -- never by the net.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict

import numpy as np
import torch

from . import general_pinn as GP
from .model_spec import G9Interaction, inertia_viscous_spec
from .rest_to_rest import RestToRestEpisode, terminal_speed_sigma
from .rest_to_rest_convolution import replay as exact_replay
from .rest_to_rest_integral import design, residual_covariance
from .torch_model import DT, TorchDynamics, on_torch_device, torch_device, zoh_torch

VARIANTS = ("LEGACY_PINN_STRONG", "PINN_HARD_BC_STRONG", "PINN_HARD_BC_DOUBLE_INTEGRAL", "PINN_HARD_BC_HYBRID",
            "PINN_SOFT_BC_DOUBLE_INTEGRAL")
_MODES = {"PINN_HARD_BC_STRONG": ("hard_rest_to_rest", "strong_only"),
          "PINN_HARD_BC_DOUBLE_INTEGRAL": ("hard_rest_to_rest", "double_integral_rest2rest"),
          "PINN_HARD_BC_HYBRID": ("hard_rest_to_rest", "hybrid"),
          "PINN_SOFT_BC_DOUBLE_INTEGRAL": ("soft_rest_to_rest", "double_integral_rest2rest")}


class RestToRestNet(torch.nn.Module):
    """Eq. (9); the MLP is initialised exactly like the historical ``_Net`` (same widths / init law / seed use)."""

    def __init__(self, t0, T, q0, qT, S_q, hidden, layers, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        dims = [1] + [hidden] * layers + [1]
        self.lin = torch.nn.ModuleList()
        for a, b in zip(dims[:-1], dims[1:]):
            L = torch.nn.Linear(a, b, dtype=DT)
            with torch.no_grad():
                L.weight.copy_(torch.randn(b, a, generator=g, dtype=DT, device="cpu") * math.sqrt(1.0 / a))
                L.bias.zero_()
            self.lin.append(L)
        self.t0, self.D = float(t0), float(T - t0)
        self.q0, self.dq, self.S = float(q0), float(qT - q0), float(S_q)

    def mlp(self, s):
        x = (2.0 * s - 1.0).unsqueeze(-1)
        for L in self.lin[:-1]:
            x = torch.tanh(L(x))
        return self.lin[-1](x).squeeze(-1)

    def forward(self, t):
        s = (t - self.t0) / self.D
        h = s * s * (3.0 - 2.0 * s)
        return self.q0 + self.dq * h + self.S * (s * s * (1.0 - s) ** 2) * self.mlp(s)


def _scale(ep):
    return max(float(np.ptp(ep.q_obs)), 5.0 * float(np.median(ep.q_sigma)), abs(ep.delta_q), 1e-9)


class _R2RData:
    """Fixed data-derived quantities (declared before training; nothing here is fitted)."""

    def __init__(self, ep: RestToRestEpisode, cfg, theta0):
        self.ep = ep
        self.t = torch.as_tensor(ep.t_q_s, dtype=DT)
        self.q = torch.as_tensor(ep.q_obs, dtype=DT)
        self.s = torch.as_tensor(ep.q_sigma, dtype=DT)
        n_quad = max(int(cfg.n_quad), 16)
        nodes = np.unique(np.r_[np.linspace(ep.t0, ep.T, n_quad), ep.t_q_s])
        self.nodes = torch.as_tensor(nodes, dtype=DT)
        self.idx_obs = torch.as_tensor(np.searchsorted(nodes, ep.t_q_s), dtype=torch.long)
        _, y = design(ep)
        self.y = torch.as_tensor(y, dtype=DT)                      # rows 1..n-1 (exact ZOH, fixed input)
        C, comp, _ = residual_covariance(ep, theta0)
        self.s2 = torch.as_tensor(np.sqrt(np.clip(np.diag(C), 1e-300, None)), dtype=DT)
        self.s2_components = comp
        lev = ep.force_Q
        rms = float(np.sqrt(np.mean(lev ** 2))) if len(lev) else 1.0
        sig = float(np.sqrt(np.mean(np.asarray(ep.force_sigma) ** 2))) if ep.force_sigma is not None else 0.0
        dt_obs = float(np.median(np.diff(ep.t_q_s)))
        g1, g2 = GP._savgol_gain()
        sq = float(np.median(ep.q_sigma))
        M0, c0 = float(theta0[0]), float(theta0[1])
        self.s_r = max(math.sqrt(sig ** 2 + (M0 * g2 * sq / dt_obs ** 2) ** 2 + (c0 * g1 * sq / dt_obs) ** 2),
                       cfg.phys_scale_rms_floor * rms, 1e-9)
        self.s_v = max(terminal_speed_sigma(ep)["sigma_vT"], 1e-9)
        self.effort = ep.effort
        self.knots = ep.force_t_s[(ep.force_t_s > ep.t0) & (ep.force_t_s < ep.T)]

    def colloc(self, n, rng):
        t = np.sort(rng.uniform(self.ep.t0, self.ep.T, n))
        if len(self.knots):                                       # never exactly on a ZOH discontinuity
            d = np.min(np.abs(t[:, None] - self.knots[None, :]), axis=1)
            t = t[d > 1e-6]
        return torch.as_tensor(t, dtype=DT)


def double_integral_residual(net, M, c, d: _R2RData):
    qn = net(d.nodes)
    dqn = qn - d.ep.q0
    I = torch.cat([torch.zeros(1, dtype=DT), torch.cumulative_trapezoid(dqn, d.nodes)])
    qi = qn[d.idx_obs][1:]
    Ii = I[d.idx_obs][1:]
    return (M * (qi - d.ep.q0) + c * Ii - d.y) / d.s2


def losses(variant, dyn, eta, net, d: _R2RData, cfg, tc, w_phys) -> dict:
    v = dyn.values(eta)
    M, c = v["M"], v["c"]
    out = {}
    qo = net(d.t)
    out["data"] = GP._huber((qo - d.q) / d.s, cfg.huber_delta)
    z = torch.zeros((), dtype=DT)
    out["strong"] = out["int2"] = out["soft_bc"] = z
    form = _MODES[variant][1]
    if form in ("strong_only", "hybrid"):
        q, qd, qdd = GP._derivs(net, tc)
        Q = zoh_torch(d.effort, tc)
        out["strong"] = GP._huber((M * qdd + c * qd - Q) / d.s_r, cfg.huber_delta)
    if form in ("double_integral_rest2rest", "hybrid"):
        out["int2"] = GP._huber(double_integral_residual(net, M, c, d), cfg.huber_delta)
    if _MODES[variant][0] == "soft_rest_to_rest":
        te = torch.as_tensor([d.ep.t0, d.ep.T], dtype=DT)
        qe, qde, _ = GP._derivs(net, te)
        ep = d.ep
        out["soft_bc"] = (((qe[0] - ep.q0) / max(ep.q0_sigma, 1e-12)) ** 2 + ((qe[1] - ep.qT) / max(ep.qT_sigma, 1e-12))
                          ** 2 + (qde[0] / d.s_v) ** 2 + (qde[1] / d.s_v) ** 2) / 4.0
    l_strong = cfg.l_phys if form in ("strong_only", "hybrid") else 0.0
    l_int = cfg.l_double_integral if form in ("double_integral_rest2rest", "hybrid") else 0.0
    out["total"] = (cfg.l_data * out["data"] + w_phys * (l_strong * out["strong"] + l_int * out["int2"]) +
                    cfg.l_soft_bc * out["soft_bc"])
    return out


@on_torch_device
def _fit_seed(variant, ep, cfg, theta0, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    spec = inertia_viscous_spec(ep.joint_type)
    dyn = TorchDynamics(spec)
    d = _R2RData(ep, cfg, theta0)
    if _MODES[variant][0] == "hard_rest_to_rest":
        net = RestToRestNet(ep.t0, ep.T, ep.q0, ep.qT, _scale(ep), cfg.hidden, cfg.layers, seed)
    else:
        net = GP._Net(ep.t0, ep.T, float(ep.q0), _scale(ep), cfg.hidden, cfg.layers, seed)
    eta = dyn.eta0(theta0)
    opt_n = torch.optim.Adam(net.parameters(), lr=cfg.lr_net)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt_n, max(cfg.steps_data, 1), eta_min=cfg.lr_net * 0.05)
    for _ in range(cfg.steps_data):
        opt_n.zero_grad()
        L = GP._huber((net(d.t) - d.q) / d.s, cfg.huber_delta)
        if _MODES[variant][0] == "soft_rest_to_rest":
            L = L + cfg.l_soft_bc * losses(variant, dyn, eta, net, d, cfg, d.t, 0.0)["soft_bc"]
        L.backward()
        opt_n.step()
        sch.step()
    opt = torch.optim.Adam([{"params": list(net.parameters()), "lr": cfg.lr_net},
                            {"params": list(eta.values()), "lr": cfg.lr_theta}])
    tc = d.colloc(cfg.n_colloc, rng)
    hist = []
    for step in range(cfg.steps_joint):
        if step and step % cfg.resample_every == 0:
            tc = d.colloc(cfg.n_colloc, rng)
        opt.zero_grad()
        ramp = min(1.0, step / max(cfg.phys_ramp_fraction * cfg.steps_joint, 1.0))
        parts = losses(variant, dyn, eta, net, d, cfg, tc, 0.05 + 0.95 * ramp)
        parts["total"].backward()
        opt.step()
        if step % 500 == 0 or step == cfg.steps_joint - 1:
            hist.append(dict(step=step, **{k: float(v) for k, v in parts.items()}))
    th = dyn.theta_from_eta({k: v.detach() for k, v in eta.items()})
    with torch.no_grad():
        qn = net(d.t).cpu().numpy()
    return th, hist, qn, d, net


def open_loop(ep, M, c, numeric_floor_q=3e-4) -> dict:
    r = exact_replay(ep, M, c)
    e = r["q"] - ep.q_obs
    sig = np.hypot(ep.q_sigma, numeric_floor_q)
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), rmse_over_sigma=float(np.sqrt(np.mean((e / sig) ** 2))),
                v_T=float(r["v"][-1]), q_T_error=float(r["q"][-1] - ep.qT),
                replay="exact two-term solution from (q0, v0 = 0) under the native ZOH effort (frozen theta)")


def fit_variant(variant, ep: RestToRestEpisode, cfg: GP.PINNConfig, theta0, seeds=(0, 1, 2), numeric_floor_q=3e-4,
                log=None) -> dict:
    """One ablation, every seed; seed chosen by frozen-theta open-loop replay (v0 = 0) on the SAME window / input."""
    t_start = time.time()
    if variant == "LEGACY_PINN_STRONG":
        return fit_legacy(ep, cfg, theta0, seeds, numeric_floor_q)
    mode, form = _MODES[variant]
    c2 = GP.PINNConfig(**{**asdict(cfg), "trajectory_boundary_mode": mode, "physics_form": form, "replay_v0": "zero"})
    rows = []
    for s in seeds:
        th, hist, qn, d, _ = _fit_seed(variant, ep, c2, theta0, int(s))
        ol = open_loop(ep, th[0], th[1], numeric_floor_q)
        rows.append(dict(seed=int(s), M=float(th[0]), c=float(th[1]), open_loop=ol,
                         neural_q_rmse=float(np.sqrt(np.mean((qn - ep.q_obs) ** 2))), final_loss=hist[-1],
                         history=hist))
        if log:
            log(f"[r2r-pinn] {variant} seed {s}: M {th[0]:.4g} c {th[1]:.4g} open-loop rmse/sigma "
                f"{ol['rmse_over_sigma']:.3g}")
    best = min(rows, key=lambda r: r["open_loop"]["rmse_over_sigma"])
    Ms, cs = np.array([r["M"] for r in rows]), np.array([r["c"] for r in rows])
    return dict(variant=variant, status="OK", M=best["M"], c=best["c"], best_seed=best["seed"],
                open_loop=best["open_loop"], neural_q_rmse=best["neural_q_rmse"], per_seed=rows,
                ensemble=dict(M_mean=float(Ms.mean()), M_sd=float(Ms.std(ddof=1)) if len(Ms) > 1 else None,
                              c_mean=float(cs.mean()), c_sd=float(cs.std(ddof=1)) if len(cs) > 1 else None,
                              label="ENSEMBLE (seeds; not a calibrated posterior)"),
                seed_selection="frozen-theta open-loop replay from (q0, v0 = 0) on the same [t0, T] and input "
                               "(neural q error reported separately, never used as the physics score)",
                settings=dict(asdict(c2), seeds=list(map(int, seeds)), theta0=list(map(float, theta0)),
                              device=str(torch_device())),
                scales=dict(s_r=d.s_r, s_v=d.s_v, s2_rms=float(d.s2.pow(2).mean().sqrt()),
                            s2_components=d.s2_components,
                            s2_rule="sqrt(diag Cov) of the declared q / anchor / wrench / tare noise propagated "
                                    "through the double integral at theta0 (fixed, not trainable)"),
                runtime_s=time.time() - t_start)


def fit_legacy(ep, cfg, theta0, seeds, numeric_floor_q) -> dict:
    """(M) matched-data adaptation of the HISTORICAL GeneralPINN: same [t0, T] samples and unmasked native effort,
    the two-term ModelSpec, its own net / loss terms / moving-segment logic UNCHANGED. Its own replay uses its
    historical 4-sample v0 (reported) -- the comparison uses the v0 = 0 open-loop replay of its frozen theta."""
    t_start = time.time()
    spec = inertia_viscous_spec(ep.joint_type)
    it = G9Interaction(name=f"{ep.scene}_r2r", joint_type=ep.joint_type, t_obs=np.array(ep.t_q_s), q_obs=np.array(ep.q_obs),
                       q_std=np.array(ep.q_sigma), effort=ep.effort,
                       meta=dict(effort_sigma=float(np.sqrt(np.mean(np.asarray(ep.force_sigma) ** 2)))
                                 if ep.force_sigma is not None else 0.0))
    c2 = GP.PINNConfig(**{**asdict(cfg), "trajectory_boundary_mode": "legacy", "physics_form": "legacy",
                          "replay_v0": "prefix_fit", "n_seeds": len(seeds)})
    try:
        r = GP.general_pinn_fit([it], spec, np.asarray(theta0, float), c2)
    except Exception as e:                                              # recorded, not hidden
        return dict(variant="LEGACY_PINN_STRONG", status="FAILED", reason=f"{type(e).__name__}: {e}",
                    runtime_s=time.time() - t_start)
    th = np.asarray(r["x"], float)
    per = []
    for k, x in enumerate(r["per_seed_theta"]):
        per.append(dict(seed=k, M=float(x[0]), c=float(x[1]), open_loop=open_loop(ep, x[0], x[1], numeric_floor_q),
                        historical_prefix_v0_replay=r["seed_selection"]["per_seed_replay_rmse_over_sigma"][k]))
    return dict(variant="LEGACY_PINN_STRONG", status="OK", M=float(th[0]), c=float(th[1]), best_seed=r["best_seed"],
                open_loop=open_loop(ep, th[0], th[1], numeric_floor_q), per_seed=per,
                historical_replay_prefix_v0=r["replay"], seed_selection=r["seed_selection"],
                adaptation="(M) matched data: same samples / unmasked effort / two-term spec; historical net, loss "
                           "terms, segmentation and 4-sample-v0 seed selection unchanged",
                settings=dict(r["settings"], seeds=list(map(int, seeds))), runtime_s=time.time() - t_start)
