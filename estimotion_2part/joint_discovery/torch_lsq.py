"""GPU solver of the geometry fit (``geometry_fit.fit_geometry``): the SAME objective scipy's
``least_squares(method="trf", loss=..., f_scale=..., x_scale="jac")`` minimises, by scipy's own unbounded TRF iteration,
on torch (GPU).

    cost(p) = 0.5 f^2 sum_i rho((r_i(p) / f)^2)          (rho = linear | soft_l1 | huber | cauchy | arctan, scipy's)

Rows: the whitened 2D-reprojection + depth rows of every fitted observation, the temporal q-acceleration prior and
the optional Stage-E ``extra`` rows, exactly as the scipy residual function builds them. Parameters: joint chart
(2 axis + 2 hinge for revolute), free q_t, one 3-D anchor per fitted track.

Structure (bundle adjustment): an observation row depends on the joint chart, ONE q_t and ONE anchor, so
  * its exact Jacobian is 8 forward-mode directions over all rows at once (joint 4, q 1, anchor 3; no finite
    differences), through the same ``rigid_kinematics`` code the NumPy path uses;
  * the Gauss-Newton system is reduced by the Schur complement of the block-diagonal 3x3 anchor blocks to a dense
    (joint + q) system of ~100 unknowns, solved directly -- this exact solve replaces scipy's iterative lsmr for the
    Gauss-Newton direction of TRF's 2-D subspace (gradient, Gauss-Newton) trust-region step.
TRF control flow, trust-radius update, lsmr regularisation term and termination tests are scipy's
(``trf_no_bounds`` with tr_solver="lsmr", regularize=True; ``solve_trust_region_2d``, ``update_tr_radius`` and
``check_termination`` are called directly), so the same ``max_nfev`` budget means the same thing.
Robust loss: scipy's own residual / Jacobian scaling (``scale_for_robust_loss_function``), so the model Hessian and
the returned Jacobian blocks are the ones scipy's ``res.jac`` would give. Extra rows (NumPy callables of the joint
parameters only) get a forward-difference Jacobian with scipy's 2-point step. Device: CUDA when available,
``FORCEARTGS_DISCOVERY_DEVICE`` overrides (``cpu`` reference runs); float64 throughout.
"""
from __future__ import annotations

import contextlib
import functools
import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize._lsq.common import (check_termination, minimize_quadratic_1d,
                                       solve_trust_region_2d, update_tr_radius)

from .rigid_kinematics import apply_joint

DT = torch.float64
EPS = float(np.finfo(float).eps)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")       # deterministic cuBLAS (before CUDA init)


@contextlib.contextmanager
def _deterministic():
    """Bit-reproducible GPU sums (sort-based scatter instead of atomics): the held-out no-leakage guarantee (T24)
    needs a training fit that is bit-identical whenever its inputs are."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


def deterministic(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        with _deterministic():
            return fn(*args, **kwargs)
    return run


def device() -> torch.device:
    want = os.environ.get("FORCEARTGS_DISCOVERY_DEVICE", "").strip()
    return torch.device(want if want else ("cuda" if torch.cuda.is_available() else "cpu"))


def robust_rho(loss: str, z: torch.Tensor):
    """scipy ``least_squares`` loss functions of z = (r / f)^2 -> (rho, rho', rho'')."""
    if loss == "linear":
        return z, torch.ones_like(z), torch.zeros_like(z)
    if loss == "soft_l1":
        t = 1 + z
        return 2 * (t.sqrt() - 1), t.rsqrt(), -0.5 * t ** -1.5
    if loss == "huber":
        m = z <= 1
        zs = torch.where(m, torch.ones_like(z), z)
        return (torch.where(m, z, 2 * zs.sqrt() - 1), torch.where(m, torch.ones_like(z), zs.rsqrt()),
                torch.where(m, torch.zeros_like(z), -0.5 * zs ** -1.5))
    if loss == "cauchy":
        t = 1 + z
        return torch.log1p(z), 1 / t, -1 / t ** 2
    if loss == "arctan":
        t = 1 + z ** 2
        return torch.atan(z), 1 / t, -2 * z / t ** 2
    raise ValueError(f"unsupported robust loss {loss!r}")


class RobustLoss:
    def __init__(self, loss, f_scale):
        self.loss, self.f = loss, float(f_scale)

    def cost(self, r):
        if self.loss == "linear":
            return 0.5 * float((r * r).sum())
        return 0.5 * self.f ** 2 * float(robust_rho(self.loss, (r / self.f) ** 2)[0].sum())

    def scale(self, r):
        """scipy scale_for_robust_loss_function: (scaled residual, Jacobian row factor)."""
        if self.loss == "linear":
            return r, torch.ones_like(r)
        _, r1, r2 = robust_rho(self.loss, (r / self.f) ** 2)
        r2 = r2 / self.f ** 2
        js = (r1 + 2 * r2 * r * r).clamp_min(EPS).sqrt()
        return r * r1 / js, js


@dataclass
class VisionRows:
    """Fixed data of the fitted observations (one row triple u, v, z per observation; z rows masked by ``dz``)."""
    kind: str
    axis0: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    origin0: np.ndarray | None
    q0: np.ndarray            # [T] (q_ref = 0, non-free entries constant)
    free_q: np.ndarray        # [n_q] time indices of the free q
    k: np.ndarray             # [n] local anchor index of each observation
    t: np.ndarray             # [n] time index
    R_cw: np.ndarray          # [n,3,3]
    t_cw: np.ndarray          # [n,3]
    Kc: np.ndarray            # [n,3,3]
    uv_obs: np.ndarray        # [n,2]
    z_obs: np.ndarray         # [n]
    dz: np.ndarray            # [n] bool: depth row present
    sz: np.ndarray            # [n] depth sigma
    sw: np.ndarray            # [n] sqrt responsibility weight
    uv_scale: np.ndarray | None
    sigma_px: float
    timestamps: np.ndarray    # [T]
    prior_sigma: float | None  # temporal q-acceleration prior sigma (None = off)


class _Problem:
    def __init__(self, v: VisionRows, n_tracks: int, dev):
        def f(a):
            return torch.as_tensor(np.asarray(a, float), dtype=DT, device=dev)

        def i(a):
            return torch.as_tensor(np.asarray(a, np.int64), device=dev)
        self.kind, self.dev, self.K = v.kind, dev, n_tracks
        self.nj = 4 if v.kind == "revolute" else 2
        self.axis0, self.e1, self.e2 = f(v.axis0), f(v.e1), f(v.e2)
        self.origin0 = f(v.origin0) if v.origin0 is not None else None
        self.q0, self.free = f(v.q0), i(v.free_q)
        self.nq = len(v.free_q)
        self.k, self.t = i(v.k), i(v.t)
        qidx = -np.ones(len(v.q0), np.int64)
        qidx[v.free_q] = np.arange(self.nq)
        self.qi = i(qidx[v.t])                                   # [n] free-q column of each row (-1: fixed)
        self.R, self.tc, Kc = f(v.R_cw), f(v.t_cw), f(v.Kc)
        self.fx, self.fy, self.sk, self.cx, self.cy = Kc[:, 0, 0], Kc[:, 1, 1], Kc[:, 0, 1], Kc[:, 0, 2], Kc[:, 1, 2]
        self.uv, self.z = f(v.uv_obs), f(v.z_obs)
        self.dz = torch.as_tensor(np.asarray(v.dz, bool), device=dev)
        self.sz = f(np.where(v.dz, v.sz, 1.0))
        self.sw = f(v.sw)
        self.uvs = f(v.uv_scale) if v.uv_scale is not None else torch.ones_like(self.sw)
        self.sigma_px = float(v.sigma_px)
        self.ts = f(v.timestamps)
        self.prior_sigma = v.prior_sigma

    def joint(self, pj):
        a = self.axis0 + pj[0] * self.e1 + pj[1] * self.e2
        a = a / torch.linalg.norm(a)
        o = self.origin0 + pj[2] * self.e1 + pj[3] * self.e2 if self.kind == "revolute" else None
        return a, o

    def q_full(self, qf):
        return self.q0.index_put((self.free,), qf)

    def obs_rows(self, pj, qf, X):
        """[n,3] whitened (u, v, z) residuals; z = 0 where there is no depth row."""
        a, o = self.joint(pj)
        q = self.q_full(qf)
        xw = apply_joint(self.kind, a, o, q[self.t], X[self.k])
        xc = (self.R @ xw.unsqueeze(-1)).squeeze(-1) + self.tc
        z = xc[:, 2]
        zs = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
        u = self.fx * xc[:, 0] / zs + self.sk * xc[:, 1] / zs + self.cx
        v = self.fy * xc[:, 1] / zs + self.cy
        w = self.sw / self.sigma_px / self.uvs
        ru = (u - self.uv[:, 0]) * w
        rv = (v - self.uv[:, 1]) * w
        rz = torch.where(self.dz, (z - self.z) / self.sz * self.sw, torch.zeros_like(z))
        return torch.stack([ru, rv, rz], 1)

    def prior_rows(self, qf):
        if self.prior_sigma is None:
            return qf[:0]
        q = self.q_full(qf)
        v = torch.diff(q) / torch.diff(self.ts)
        acc = torch.diff(v) / ((self.ts[2:] - self.ts[:-2]) / 2)
        return acc / self.prior_sigma


class TorchLSQResult:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@deterministic
def solve(v: VisionRows, X0: np.ndarray, loss: str, f_scale: float, max_nfev: int, extra=None,
          ftol=1e-8, xtol=1e-8, gtol=1e-8) -> TorchLSQResult:
    """scipy's unbounded TRF iteration on the Schur-reduced normal equations; returns x in the scipy packing
    [joint chart, free q, anchors], cost / nfev / status / message like ``least_squares`` and the robust-scaled
    Gauss-Newton blocks at the solution (A [g,g], B [g,K,3], D [K,3,3]) for the marginal covariance."""
    dev = device()
    P = _Problem(v, len(X0), dev)
    L = RobustLoss(loss, f_scale)
    nj, nq, K = P.nj, P.nq, P.K
    ng = nj + nq
    pj = torch.zeros(nj, dtype=DT, device=dev)
    qf = P.q0[P.free].clone()
    X = torch.as_tensor(np.asarray(X0, float), dtype=DT, device=dev)

    def extra_rows(pj_, qf_):
        if extra is None:
            return None
        a, o = P.joint(pj_)
        q = P.q_full(qf_)
        r = extra.residuals(v.kind, a.cpu().numpy(), None if o is None else o.cpu().numpy(), q.cpu().numpy())
        return torch.as_tensor(np.nan_to_num(np.asarray(r, float)), dtype=DT, device=dev)

    def all_rows(pj_, qf_, X_):
        parts = [P.obs_rows(pj_, qf_, X_).reshape(-1), P.prior_rows(qf_)]
        e = extra_rows(pj_, qf_)
        if e is not None:
            parts.append(e)
        return torch.cat(parts)

    f_obs = P.obs_rows
    has_q = P.qi >= 0
    qi = P.qi.clamp_min(0)

    def jt(Jj, Jq, Jx, r):
        """J' r for observation rows [n,3] (scaled blocks)."""
        g = torch.zeros(ng, dtype=DT, device=dev)
        g[:nj] = torch.einsum("nci,nc->i", Jj, r)
        g[nj:] = torch.zeros(nq, dtype=DT, device=dev).index_add_(0, qi, (Jq * r).sum(1) * has_q)
        gx = torch.zeros(K, 3, dtype=DT, device=dev).index_add_(0, P.k, torch.einsum("ncd,nc->nd", Jx, r))
        return g, gx

    def linearize(pj_, qf_, X_):
        """Robust-scaled residual and Gauss-Newton blocks at (pj, qf, X)."""
        zj, zq, zx = torch.zeros_like(pj_), torch.zeros_like(qf_), torch.zeros_like(X_)
        cols = []
        for j in range(nj):
            tj = zj.clone()
            tj[j] = 1.0
            r_obs, d = torch.func.jvp(f_obs, (pj_, qf_, X_), (tj, zq, zx))
            cols.append(d)
        Jj = torch.stack(cols, -1)                                    # [n,3,nj]
        Jq = torch.func.jvp(f_obs, (pj_, qf_, X_), (zj, torch.ones_like(qf_), zx))[1]   # [n,3] (own q_t only)
        Jx = []
        for dd in range(3):
            tx = zx.clone()
            tx[:, dd] = 1.0
            Jx.append(torch.func.jvp(f_obs, (pj_, qf_, X_), (zj, zq, tx))[1])
        Jx = torch.stack(Jx, -1)                                      # [n,3,3]
        rs, js = L.scale(r_obs)
        Jj, Jq, Jx = Jj * js[..., None], Jq * js, Jx * js[..., None]
        A = torch.zeros(ng, ng, dtype=DT, device=dev)
        A[:nj, :nj] = torch.einsum("nci,ncj->ij", Jj, Jj)
        jq = torch.einsum("nci,nc->ni", Jj, Jq) * has_q[:, None]
        A[nj:, :nj] = torch.zeros(nq, nj, dtype=DT, device=dev).index_add_(0, qi, jq)
        A[:nj, nj:] = A[nj:, :nj].T
        A[nj:, nj:] += torch.diag(torch.zeros(nq, dtype=DT, device=dev).index_add_(0, qi, (Jq * Jq).sum(1) * has_q))
        B = torch.zeros(ng, K, 3, dtype=DT, device=dev)
        B[:nj] = torch.zeros(K, nj, 3, dtype=DT, device=dev).index_add_(
            0, P.k, torch.einsum("nci,ncd->nid", Jj, Jx)).permute(1, 0, 2)
        bq = torch.einsum("nc,ncd->nd", Jq, Jx) * has_q[:, None]
        B[nj:] = torch.zeros(nq * K, 3, dtype=DT, device=dev).index_add_(0, qi * K + P.k, bq).reshape(nq, K, 3)
        D = torch.zeros(K, 3, 3, dtype=DT, device=dev).index_add_(0, P.k, torch.einsum("ncd,nce->nde", Jx, Jx))
        g, gx = jt(Jj, Jq, Jx, rs)
        # rows of the joint parameters only: temporal prior (exact Jacobian) and extra rows (scipy 2-point)
        G_rows, G_r = [], []
        if P.prior_sigma is not None and nq:
            rp = P.prior_rows(qf_)
            Jp = torch.zeros(len(rp), ng, dtype=DT, device=dev)
            Jp[:, nj:] = torch.func.jacfwd(P.prior_rows)(qf_)
            G_rows.append(Jp)
            G_r.append(rp)
        if extra is not None:
            x0 = torch.cat([pj_, qf_])
            e0 = extra_rows(pj_, qf_)
            Je = torch.zeros(len(e0), ng, dtype=DT, device=dev)
            for c in range(ng):
                h = EPS ** 0.5 * max(1.0, abs(float(x0[c])))
                h = h if float(x0[c]) >= 0 else -h
                x1 = x0.clone()
                x1[c] += h
                Je[:, c] = (extra_rows(x1[:nj], x1[nj:]) - e0) / ((x0[c] + h) - x0[c])
            G_rows.append(Je)
            G_r.append(e0)
        if G_rows:
            Jg_ = torch.cat(G_rows)
            rg, jsg = L.scale(torch.cat(G_r))
            Jg_ = Jg_ * jsg[:, None]
            A += Jg_.T @ Jg_
            g += Jg_.T @ rg
        return dict(A=A, B=B, D=D, g=g, gx=gx, Jj=Jj, Jq=Jq, Jx=Jx, js=js)

    def reg_solve(lin, reg_g, reg_x, rg, rx):
        """(H + diag(reg)) y = [rg; rx] through the anchor Schur complement (exact; replaces scipy's lsmr)."""
        A, B, D = lin["A"], lin["B"], lin["D"]
        Di = torch.linalg.inv(D + torch.diag_embed(reg_x) + torch.eye(3, dtype=DT, device=dev) * 1e-12)
        BD = torch.einsum("gkd,kde->gke", B, Di)
        S = A + torch.diag(reg_g) - torch.einsum("gke,hke->gh", BD, B)
        yg = torch.linalg.solve(S, rg - torch.einsum("gke,ke->g", BD, rx))
        yx = torch.einsum("kde,ke->kd", Di, rx - torch.einsum("gkd,g->kd", B, yg))
        return yg, yx

    def hmul(lin, vg, vx):
        A, B, D = lin["A"], lin["B"], lin["D"]
        return (A @ vg + torch.einsum("gkd,kd->g", B, vx),
                torch.einsum("gkd,g->kd", B, vg) + torch.einsum("kde,ke->kd", D, vx))

    def jac_scale(lin, old=None):
        """scipy compute_jac_scale: column norms of the (robust-scaled) Jacobian."""
        si = torch.cat([torch.diagonal(lin["A"]), torch.diagonal(lin["D"], dim1=1, dim2=2).reshape(-1)]).sqrt()
        if old is None:
            si = torch.where(si == 0, torch.ones_like(si), si)
        else:
            si = torch.maximum(si, old)
        return 1 / si, si

    def split(v):
        return v[:ng], v[ng:].reshape(K, 3)

    def flat(vg, vx):
        return torch.cat([vg, vx.reshape(-1)])

    def quad(lin, d, s):
        """J_h s squared norm (J_h = J diag(d)) for a flat vector s."""
        vg, vx = split(d * s)
        hg, hx = hmul(lin, vg, vx)
        return float(vg @ hg + (vx * hx).sum())

    # ---- scipy trf_no_bounds (tr_solver="lsmr", regularize=True, x_scale="jac"), same control flow
    x = flat(torch.cat([pj, qf]), X)
    def unpack(x_):
        return x_[:nj], x_[nj:ng], x_[ng:].reshape(K, 3)

    cost = L.cost(all_rows(*unpack(x)))
    nfev = 1
    lin = linearize(*unpack(x))
    g = flat(lin["g"], lin["gx"])
    scale, scale_inv = jac_scale(lin)
    Delta = float(torch.linalg.norm(x * scale_inv)) or 1.0
    history = []
    status = None
    while True:
        if float(g.abs().max()) < gtol:
            status = 1
        if status is not None or nfev >= max_nfev:
            break
        d = scale
        g_h = d * g
        gg = float(g_h @ g_h)
        a_ = quad(lin, d, -g_h)                                   # build_quadratic_1d(J_h, g_h, -g_h)
        b_ = -gg
        to_tr = Delta / np.sqrt(gg) if gg > 0 else 0.0
        ag_value = minimize_quadratic_1d(a_, b_, 0, to_tr)[1]
        reg = -ag_value / Delta ** 2
        sg, sx = split(reg * scale_inv ** 2)                     # (J_h'J_h + reg I) p = g_h, p = y / d
        yg, yx = reg_solve(lin, sg, sx, lin["g"], lin["gx"])
        gn_h = flat(yg, yx) * scale_inv
        S, _ = torch.linalg.qr(torch.stack([g_h, gn_h], 1))
        B_S = np.array([[quad(lin, d, S[:, i]) if i == j else 0.0 for j in range(2)] for i in range(2)])
        vg0, vx0 = split(d * S[:, 0])
        hg1, hx1 = hmul(lin, *split(d * S[:, 1]))
        B_S[0, 1] = B_S[1, 0] = float(vg0 @ hg1 + (vx0 * hx1).sum())
        g_S = (S.T @ g_h).cpu().numpy()
        actual_reduction = -1.0
        while actual_reduction <= 0 and nfev < max_nfev:
            p_S, _ = solve_trust_region_2d(B_S, g_S, Delta)
            step_h = S @ torch.as_tensor(p_S, dtype=DT, device=dev)
            predicted_reduction = -(0.5 * quad(lin, d, step_h) + float(g_h @ step_h))
            step = d * step_h
            x_new = x + step
            cost_new = L.cost(all_rows(*unpack(x_new)))
            nfev += 1
            step_h_norm = float(torch.linalg.norm(step_h))
            if not np.isfinite(cost_new):
                Delta = 0.25 * step_h_norm
                continue
            actual_reduction = cost - cost_new
            Delta_new, ratio = update_tr_radius(Delta, actual_reduction, predicted_reduction, step_h_norm,
                                                step_h_norm > 0.95 * Delta)
            history.append((cost_new, Delta, ratio, actual_reduction > 0))
            status = check_termination(actual_reduction, cost, float(torch.linalg.norm(step)),
                                       float(torch.linalg.norm(x)), ratio, ftol, xtol)
            if status is not None:
                break
            Delta = Delta_new
        if actual_reduction > 0:
            x, cost = x_new, cost_new
            lin = linearize(*unpack(x))
            g = flat(lin["g"], lin["gx"])
            scale, scale_inv = jac_scale(lin, scale_inv)
    status = 0 if status is None else status
    message = {0: "The maximum number of function evaluations is exceeded.",
               1: "`gtol` termination condition is satisfied.",
               2: "`ftol` termination condition is satisfied.",
               3: "`xtol` termination condition is satisfied.",
               4: "Both `ftol` and `xtol` termination conditions are satisfied."}[status]
    F = cost
    pj, qf, X = unpack(x)
    A, B, D = lin["A"], lin["B"], lin["D"]
    x = np.concatenate([pj.cpu().numpy(), qf.cpu().numpy(), X.cpu().numpy().ravel()])
    return TorchLSQResult(x=x, cost=F, nfev=nfev, status=status, message=message, device=str(dev),
                          A=A.cpu().numpy(), B=B.cpu().numpy(), D=D.cpu().numpy(), trust_radius_final=Delta,
                          history=history,
                          solver="scipy trf_no_bounds (2-D subspace trust region, x_scale=jac) on torch; exact "
                                 "forward-mode Jacobian; Gauss-Newton step by the anchor Schur complement (not lsmr)")


@deterministic
def refine_anchors(kind, axis, origin, q, X0, obs, K, T_CW, noise, track_mask, iters=6, use_depth=True):
    """GPU port of ``geometry_fit.refine_anchors`` (same robust Student-t IRLS Gauss-Newton per track, same
    regularisation, clipping and stopping rule); NumPy in, NumPy out."""
    from .observations import depth_sigma
    X = np.array(X0, float)
    sel = track_mask[obs.track] & obs.uv_valid & np.isfinite(X[obs.track]).all(1)
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        return X
    dev = device()
    def f(a):
        return torch.as_tensor(np.asarray(a, float), dtype=DT, device=dev)
    tr_np = obs.track[idx]
    active_np = np.unique(tr_np)
    loc = np.searchsorted(active_np, tr_np)
    tr = torch.as_tensor(loc, device=dev)
    t, cam = obs.time[idx], obs.cam[idx]
    Rcw = f(T_CW[cam, :3, :3])
    tcw = f(T_CW[cam, :3, 3])
    Kc = f(K[cam])
    fx, fy = Kc[:, 0, 0], Kc[:, 1, 1]
    qt = f(np.asarray(q, float)[t])
    if kind == "revolute":
        from .rigid_kinematics import exp_so3
        a = f(axis)
        a = a / torch.linalg.norm(a)
        RA = exp_so3(a[None] * qt[:, None])
        M = Rcw @ RA
    else:
        M = Rcw
    dz_np = (obs.depth_valid[idx] & obs.mask_valid[idx] & np.isfinite(obs.z[idx])) if use_depth \
        else np.zeros(len(idx), bool)
    dz = torch.as_tensor(dz_np, device=dev)
    sz = f(depth_sigma(np.nan_to_num(obs.z[idx], nan=1.0), noise, obs.depth_edge[idx], obs.depth_cos[idx]))
    zo = f(np.nan_to_num(obs.z[idx]))
    uvo = f(obs.uv[idx])
    scl = getattr(obs, "uv_sigma_scale", None)
    su = f(noise.sigma_px * (scl[idx] if scl is not None else np.ones(len(idx))))
    dof = float(noise.student_t_dof)
    ax_t = f(axis) if kind != "static" else None
    o_t = f(origin) if kind == "revolute" else None
    Xa = f(X[active_np])
    n_act = len(active_np)
    eye = torch.eye(3, dtype=DT, device=dev)
    for _ in range(iters):
        Xi = Xa[tr]
        xw = Xi if kind == "static" else apply_joint(kind, ax_t, o_t, qt, Xi)
        xc = (Rcw @ xw.unsqueeze(-1)).squeeze(-1) + tcw
        z = xc[:, 2]
        zs = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
        u = Kc[:, 0, 0] * xc[:, 0] / zs + Kc[:, 0, 1] * xc[:, 1] / zs + Kc[:, 0, 2]
        v = Kc[:, 1, 1] * xc[:, 1] / zs + Kc[:, 1, 2]
        zz = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
        Dp = torch.zeros(len(idx), 2, 3, dtype=DT, device=dev)
        Dp[:, 0, 0], Dp[:, 0, 2] = fx / zz, -fx * xc[:, 0] / zz ** 2
        Dp[:, 1, 1], Dp[:, 1, 2] = fy / zz, -fy * xc[:, 1] / zz ** 2
        Juv = Dp @ M / su[:, None, None]
        ruv = (torch.stack([u, v], 1) - uvo) / su[:, None]
        wuv = (dof + 1) / (dof + (ruv ** 2).sum(1) / 2)
        H = torch.einsum("mki,mkj->mij", Juv, Juv) * wuv[:, None, None]
        g = torch.einsum("mki,mk->mi", Juv, ruv) * wuv[:, None]
        if dz_np.any():
            Jz = M[:, 2, :] / sz[:, None]
            rz = torch.where(dz, (z - zo) / sz, torch.zeros_like(z))
            wz = torch.where(dz, (dof + 1) / (dof + rz ** 2), torch.zeros_like(z))
            H = H + torch.einsum("mi,mj->mij", Jz, Jz) * wz[:, None, None]
            g = g + Jz * (rz * wz)[:, None]
        Ht = torch.zeros(n_act, 3, 3, dtype=DT, device=dev).index_add_(0, tr, H)
        gt = torch.zeros(n_act, 3, dtype=DT, device=dev).index_add_(0, tr, g)
        tr_h = torch.diagonal(Ht, dim1=1, dim2=2).sum(1)
        Ha = Ht + eye[None] * (1e-6 + 1e-9 * tr_h)[:, None, None]
        step = torch.linalg.solve(Ha, gt.unsqueeze(-1)).squeeze(-1).clamp(-0.05, 0.05)
        Xa = Xa - step
        if float(step.abs().max()) < 1e-7:
            break
    X[active_np] = Xa.cpu().numpy()
    return X
