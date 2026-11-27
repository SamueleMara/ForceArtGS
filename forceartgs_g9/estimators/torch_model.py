"""One differentiable formulation of the G9 general model, shared by GeneralPINN and GeneralAugmentedEKF.

It is DERIVED from ``ModelSpec`` (same parameter names, same switches, same gravity law, same units) so the
three estimators (PINN, augmented EKF, rollout) and the integral-LS initializer cannot drift apart:

    M(theta) q'' + c q' + Fc sgn(q') + G(q; theta) + b = Q          (moving)
    q' = 0, |Q - G - b| <= Fs                                       (sticking)

gravity by parameter source
  geometry_coupled            M = rho I_axis,1 + I_refl,  G = rho (A1 cos q + B1 sin q)  (R)  /  rho G_P,1  (P):
                              ONE density for inertia AND gravity (A1, B1 from the mass model's COM, r_perp)
  joint_space_gravity_coupled conditional diagnostic: P  G = -M g.a ;  R  G = mu (A_u cos q + B_u sin q), M and mu
                              independent (assumed COM direction)
  joint_space_only            effective load gA cos q + gB sin q (R) / gP (P): not gravity

Parameter coordinates eta (unconstrained): log for the strictly / weakly positive quantities
(alpha, M, mu, c, Fc, dFs, Fs), identity for signed ones (b, gA, gB, gP).
"""
from __future__ import annotations

import functools
import math
import os

import numpy as np
import torch

LOG_PARAMS = ("alpha", "M", "mu", "c", "Fc", "dFs", "Fs")
DT = torch.float64


PINN_DEVICES = ("cpu", "cuda")


def torch_device() -> torch.device:
    """Device of the PINN / torch estimators: ``FORCEARTGS_PINN_DEVICE`` (``--pinn-device`` of the driver and of
    ``forceartgs_g9.cli rest-to-rest-compare``), default ``cpu``. Measured on the real fridge episode: these small
    nets (32 x 3, a few hundred points, 7000 sequential steps) are launch-bound on the GPU, 101 s vs 59 s per variant,
    same estimates. Weight initialisation always draws from a seeded CPU generator, so the initial nets are
    bit-identical on every device."""
    want = os.environ.get("FORCEARTGS_PINN_DEVICE", "cpu").strip() or "cpu"
    if want not in PINN_DEVICES:
        raise ValueError(f"FORCEARTGS_PINN_DEVICE must be one of {PINN_DEVICES}, got {want!r}")
    if want == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FORCEARTGS_PINN_DEVICE=cuda but CUDA is not available")
    return torch.device(want)


def on_torch_device(fn):
    """Run ``fn`` with every tensor it creates on ``torch_device()``."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        with torch_device():
            return fn(*args, **kwargs)
    return run


def to_eta(name, value):
    if name in LOG_PARAMS:
        return math.log(max(float(value), 1e-12))
    return float(value)


def from_eta(name, eta):
    return torch.exp(eta) if name in LOG_PARAMS else eta


class TorchDynamics:
    def __init__(self, spec):
        self.spec = spec
        self.names = list(spec.names)
        t = spec.terms
        self.has = dict(inertia=t.inertia, viscous=t.viscous, coulomb=t.coulomb, static=t.static,
                        gravity=t.gravity, bias=t.bias)
        self.src = spec.parameter_source
        self.jt = spec.joint_type
        self.M1 = self.refl = None
        self.gA1 = self.gB1 = 0.0
        if self.src == "geometry_coupled":
            self.M1 = float(spec.mass_model.M1)
            self.refl = float(spec.mass_model.reflected())
            if t.gravity:
                self.gA1, self.gB1 = spec._g_unit_coeffs()
        elif self.src == "joint_space_gravity_coupled" and t.gravity:
            if self.jt == "revolute":
                self.gA1, self.gB1 = spec._g_js_coeffs({"mu": 1.0})
            else:
                g = np.asarray(spec.gravity.g_world, float)
                a = np.asarray(spec.joint_axis, float)
                self.gA1 = float(-(g @ (a / np.linalg.norm(a))))        # G_P = M * gA1

    def values(self, eta: dict) -> dict:
        v = {n: from_eta(n, eta[n]) for n in self.names}
        for k, p in self.spec.fixed_priors.items():
            v[k] = torch.as_tensor(float(p["value"]), dtype=DT)
        return v

    def terms(self, v: dict):
        """(M, c, Fc, Fs, b, G(q) callable) as torch expressions of the parameters."""
        z = torch.zeros((), dtype=DT)
        if self.src == "geometry_coupled":
            M = v["alpha"] * self.M1 + self.refl
        else:
            M = v["M"]
        c = v["c"] if self.has["viscous"] else z
        Fc = v["Fc"] if self.has["coulomb"] else z
        if self.has["static"]:
            Fs = (v["Fc"] + v["dFs"]) if self.has["coulomb"] else v["Fs"]
        else:
            Fs = None
        b = v["b"] if self.has["bias"] else z
        if not self.has["gravity"]:
            G = lambda q: torch.zeros_like(q)
        elif self.src == "geometry_coupled":
            s = v["alpha"]
            G = (lambda q: s * (self.gA1 * torch.cos(q) + self.gB1 * torch.sin(q))) if self.jt == "revolute" else \
                (lambda q: s * self.gA1 + 0.0 * q)
        elif self.src == "joint_space_gravity_coupled":
            if self.jt == "revolute":
                mu = v["mu"]
                G = lambda q: mu * (self.gA1 * torch.cos(q) + self.gB1 * torch.sin(q))
            else:
                G = lambda q: M * self.gA1 + 0.0 * q
        else:
            if self.jt == "revolute":
                gA, gB = v["gA"], v["gB"]
                G = lambda q: gA * torch.cos(q) + gB * torch.sin(q)
            else:
                gP = v["gP"]
                G = lambda q: gP + 0.0 * q
        return M, c, Fc, Fs, b, G

    def residual(self, v, q, qd, qdd, Q):
        """Moving-regime residual r = M q'' + c q' + Fc sgn(q') + G(q) + b - Q (the full equation)."""
        M, c, Fc, Fs, b, G = self.terms(v)
        return M * qdd + c * qd + Fc * torch.sign(qd) + G(q) + b - Q

    def theta_from_eta(self, eta: dict) -> np.ndarray:
        """Physical theta vector in ModelSpec order (for rollout / ModelSpec.sim)."""
        return np.array([float(from_eta(n, eta[n])) for n in self.names])

    def eta0(self, theta0) -> dict:
        return {n: torch.tensor(to_eta(n, x), dtype=DT, requires_grad=True) for n, x in zip(self.names, theta0)}


def zoh_torch(effort, t: torch.Tensor) -> torch.Tensor:
    """Q(t) of a forward-ZOH effort at torch times (no gradient w.r.t. t: Q is a measured input)."""
    tf = torch.as_tensor(np.asarray(effort.t_s, float), dtype=t.dtype, device=t.device)
    lev = torch.as_tensor(np.asarray(effort.levels, float), dtype=DT, device=t.device)
    idx = (torch.searchsorted(tf, t.detach().contiguous(), right=True) - 1).clamp(0, len(lev) - 1)
    return lev[idx]                                                   # on t's device: no host round trip per step


def zoh_integral(effort, ta, tb) -> float:
    """Exact int_{ta}^{tb} Q dt of a forward-ZOH effort (native samples)."""
    tf = np.asarray(effort.t_s, float)
    lev = np.asarray(effort.levels, float)
    knots = np.r_[ta, tf[(tf > ta) & (tf < tb)], tb]
    J = 0.0
    for a_, b_ in zip(knots[:-1], knots[1:]):
        i = int(np.clip(np.searchsorted(tf, a_ + 1e-12, side="right") - 1, 0, len(lev) - 1))
        J += float(lev[i]) * (b_ - a_)
    return J
