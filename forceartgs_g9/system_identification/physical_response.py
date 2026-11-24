"""Expected joint physics (spec section 2): hinge inertia, geometric gravity load / stiffness, the local sliding-regime
linearization, the effort -> displacement transfer function, its poles, and the exact constant-effort transient.

    M qdd + c qd + Fc sgn(qd) + G(q) [+ tau_mech] = u(t)                                   (1)
    revolute: c(q) = o + R_a(q - q_ref)(c0 - o),  G = -[a x (c(q) - o)]^T m g = dU/dq,
              I_h = a^T I_COM a + m |(I - a a^T)(c0 - o)|^2                                (2)
    prismatic: G = -m a^T g (constant), M = m
    local (moving, fixed sign): M dqdd + B_eff dqd + K_eff dq = du,
              B_eff = c + d tau_mech/d qd,  K_eff = G'(q_b) + d tau_mech/d q                (3,4)
    H_q(s) = 1/(M s^2 + B_eff s + K_eff) = g_u / (s^2 + lam s + kappa)                      (5,6)
    p_1,2 = (-lam +- sqrt(lam^2 - 4 kappa)) / 2                                             (15)

Ideal Coulomb friction is a FIXED SIGNED OFFSET  Fc sgn(v_b) in the moving regime; its velocity derivative is zero
(no invented viscous coefficient). K_eff may be negative (unstable gravitational equilibrium): never clamped. There is
no transfer function at rest, across a reversal / stick / impact / limit (``linearize`` refuses v_b = 0).

The gravity law reuses ``forceartgs_g9.dynamics.general_1dof.RigidBodyGravity`` (G = dU/dq, the generalized LOAD).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..dynamics.general_1dof import RigidBodyGravity, rotation


class PhysicalResponseError(ValueError):
    pass


# ================================================================================================ inertia / gravity
def hinge_inertia(axis, origin, com, mass, I_com) -> float:
    """Eq. (2): I_h = a^T I_COM a + m |r_perp|^2 (parallel-axis theorem about the hinge line)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    J = np.asarray(I_com, float).reshape(3, 3)
    if not np.allclose(J, J.T, atol=1e-12) or np.min(np.linalg.eigvalsh(J)) <= 0:
        raise PhysicalResponseError("I_COM must be symmetric positive definite")
    if not mass > 0:
        raise PhysicalResponseError("mass must be > 0")
    r = np.asarray(com, float) - np.asarray(origin, float)
    rp = r - (r @ a) * a
    return float(a @ J @ a + mass * rp @ rp)


def rigid_gravity(joint_type, axis, origin, com_ref, mass, g_world, q_ref=0.0) -> RigidBodyGravity:
    return RigidBodyGravity(joint_type, tuple(np.asarray(axis, float)), tuple(np.zeros(3) if origin is None else
                                                                           np.asarray(origin, float)),
                            tuple(np.asarray(com_ref, float)), float(mass), tuple(np.asarray(g_world, float)),
                            float(q_ref))


def gravity_load(rbg: RigidBodyGravity, q):
    """(G(q), G'(q)): the generalized gravity load and its analytic configuration derivative (stiffness)."""
    A, B = rbg.basis_coefficients()
    q = np.asarray(q, float)
    if rbg.joint_type == "prismatic":
        return np.full(q.shape, A) if q.ndim else A, np.zeros(q.shape) if q.ndim else 0.0
    G = A * np.cos(q) + B * np.sin(q)
    dG = -A * np.sin(q) + B * np.cos(q)
    return (G, dG) if q.ndim else (float(G), float(dG))


def _perturb_axis(a, sd, rng):
    a = np.asarray(a, float) / np.linalg.norm(a)
    w = rng.normal(0, sd, 3)
    w -= (w @ a) * a                                       # small rotation perpendicular to the axis
    th = np.linalg.norm(w)
    return a if th == 0 else rotation(w / th, th) @ a


def gravity_stiffness_uncertainty(rbg: RigidBodyGravity, q_bar, *, axis_sd_rad=0.0, origin_sd_m=0.0, com_sd_m=0.0,
                                  inertia=None, n=400, seed=0) -> dict:
    """Monte Carlo propagation of the accepted-axis / hinge-line / COM uncertainty into G(q_b), G'(q_b) (and the
    normalized a_g, kappa_g = G'/M when ``inertia`` M is given). The density / scale is NOT varied here -- axis and
    COM errors must be propagated, never absorbed into density."""
    rng = np.random.default_rng(seed)
    G0, dG0 = gravity_load(rbg, q_bar)
    Gs, dGs = [], []
    for _ in range(int(n)):
        a = _perturb_axis(rbg.axis, axis_sd_rad, rng) if axis_sd_rad > 0 else np.asarray(rbg.axis, float)
        o = np.asarray(rbg.origin, float) + (rng.normal(0, origin_sd_m, 3) if origin_sd_m > 0 else 0.0)
        c = np.asarray(rbg.com0, float) + (rng.normal(0, com_sd_m, 3) if com_sd_m > 0 else 0.0)
        r = RigidBodyGravity(rbg.joint_type, tuple(a), tuple(o), tuple(c), rbg.mass, rbg.g_world, rbg.q_ref)
        g, d = gravity_load(r, q_bar)
        Gs.append(g)
        dGs.append(d)
    Gs, dGs = np.asarray(Gs), np.asarray(dGs)
    out = dict(G=G0, G_sd=float(np.std(Gs)), dG=dG0, dG_sd=float(np.std(dGs)),
               dG_interval95=[float(np.quantile(dGs, 0.025)), float(np.quantile(dGs, 0.975))],
               n_mc=int(n), axis_sd_rad=axis_sd_rad, origin_sd_m=origin_sd_m, com_sd_m=com_sd_m,
               note="geometry uncertainty only (density / scale not varied)")
    if inertia:
        out.update(a_g=G0 / inertia, a_g_sd=float(np.std(Gs)) / inertia, kappa_g=dG0 / inertia,
                   kappa_g_sd=float(np.std(dGs)) / inertia)
    return out


# ================================================================================================ local linear model
def poles(lam, kappa):
    """Eq. (15); complex for an underdamped pair, a positive real pole for kappa < 0 (unstable, NOT clamped)."""
    disc = complex(lam * lam - 4.0 * kappa)
    r = disc ** 0.5
    p1, p2 = 0.5 * (-lam + r), 0.5 * (-lam - r)
    real = abs(disc.imag) < 1e-300 and disc.real >= 0
    return (p1.real, p2.real) if real else (p1, p2)


def classify_poles(p1, p2, tol=1e-12):
    ps = [p1, p2]
    if any(isinstance(p, complex) and abs(p.imag) > tol for p in ps):
        kind = "UNDERDAMPED_COMPLEX_PAIR"
    else:
        kind = "REAL"
    re = [p.real if isinstance(p, complex) else p for p in ps]
    if max(re) > tol:
        stab = "UNSTABLE_EQUILIBRIUM (K_eff < 0: e.g. a gravitational operating point above the hinge)"
    elif max(abs(r) for r in re) <= tol or min(abs(r) for r in re) <= tol:
        stab = "MARGINAL (integrator pole at 0: no stiffness)"
    else:
        stab = "STABLE"
    return dict(kind=kind, stability=stab)


@dataclass(frozen=True)
class LocalLinearModel:
    """M dqdd + B dqd + K dq = du about a MOVING operating point (fixed velocity sign)."""

    M: float
    B: float
    K: float
    joint_type: str = "revolute"
    offset: float = 0.0                 # nominal generalized load at the operating point (Coulomb + G(q_b) + ...)

    def __post_init__(self):
        if not (self.M > 0 and math.isfinite(self.M)):
            raise PhysicalResponseError("M must be > 0 (no zero / negative inertia)")
        if self.B < 0:
            raise PhysicalResponseError("B_eff < 0: an active (non-passive) local damping is not a passive joint")

    @property
    def g_u(self):
        return 1.0 / self.M

    @property
    def lam(self):
        return self.B / self.M

    @property
    def kappa(self):
        return self.K / self.M

    def tf_q(self, normalized=True):
        """(num, den) in descending powers of s: H_q = num/den."""
        if normalized:
            return [self.g_u], [1.0, self.lam, self.kappa]
        return [1.0], [self.M, self.B, self.K]

    def tf_v(self, normalized=True):
        num, den = self.tf_q(normalized)
        return list(num) + [0.0], den

    def poles(self):
        return poles(self.lam, self.kappa)

    def H(self, w, output="q"):
        s = 1j * np.asarray(w, float)
        Hq = self.g_u / (s * s + self.lam * s + self.kappa)
        return Hq if output == "q" else s * Hq

    def describe(self):
        p1, p2 = self.poles()
        u = UNITS_LOCAL[self.joint_type]
        cls = classify_poles(p1, p2)

        def c(p):
            return dict(re=p.real, im=p.imag) if isinstance(p, complex) else dict(re=float(p), im=0.0)
        return dict(M=self.M, B_eff=self.B, K_eff=self.K, g_u=self.g_u, lam=self.lam, kappa=self.kappa,
                    H_q=dict(num=self.tf_q()[0], den=self.tf_q()[1], form="g_u / (s^2 + lam s + kappa)"),
                    H_v=dict(num=self.tf_v()[0], den=self.tf_v()[1], form="g_u s / (s^2 + lam s + kappa)"),
                    poles=[c(p1), c(p2)], **cls, units=u,
                    validity="local, continuous-time, sliding (fixed velocity sign) only; NOT the static-friction or "
                             "solver model")


UNITS_LOCAL = {"revolute": dict(M="kg m^2", B="N m s/rad", K="N m/rad", u="N m"),
               "prismatic": dict(M="kg", B="N s/m", K="N/m", u="N")}


def linearize(q_bar, v_bar, *, M, c, Fc=0.0, G=None, dG=None, joint_type="revolute", mech=None, dmech_dq=None,
              dmech_dv=None) -> LocalLinearModel:
    """Eq. (3)-(4) about a moving operating point. ``G``/``dG`` callables (gravity load and stiffness), ``mech`` an
    independently supported mechanism load with its derivatives. Coulomb enters only the offset."""
    if not (math.isfinite(v_bar) and v_bar != 0.0):
        raise PhysicalResponseError("no local transfer function at rest / stick / reversal (v_b = 0): use the hybrid "
                                    "model and event tests")
    s = 1.0 if v_bar > 0 else -1.0
    B = float(c) + (float(dmech_dv(q_bar, v_bar)) if dmech_dv else 0.0)
    K = (float(dG(q_bar)) if dG else 0.0) + (float(dmech_dq(q_bar, v_bar)) if dmech_dq else 0.0)
    off = float(Fc) * s + (float(G(q_bar)) if G else 0.0) + (float(mech(q_bar, v_bar)) if mech else 0.0)
    return LocalLinearModel(float(M), B, K, joint_type, off)


# ================================================================================================ exact transient
def sliding_transient(t, *, M, c, Fc, u0, G0=0.0, s_v, q0, v0):
    """Eq. (7)-(8): constant effort u0, constant load G0, velocity sign s_v, from (q0, v0) at t = 0.

        M vdot + c v = f,  f = u0 - G0 - s_v Fc,   v_inf = f / c,
        v = v_inf + (v0 - v_inf) e^{-lam t},  q = q0 + v_inf t + (v0 - v_inf)(1 - e^{-lam t}) / lam

    c -> 0 uses the finite limit (v = v0 + f t / M) through expm1-based phi functions (no division by a tiny c).
    The exponential is NOT continued past a zero-velocity crossing: the segment terminates at t_stop, and after it
    the state is held at rest when |u0 - G0| <= Fc (kinetic level as the stick threshold); otherwise ``t_stop`` is
    reported with ``continues_after_stop=False`` and the caller must invoke the breakaway / reversed-sign model."""
    if not M > 0:
        raise PhysicalResponseError("M must be > 0")
    if s_v not in (-1.0, 1.0, -1, 1):
        raise PhysicalResponseError("s_v must be +-1 (fixed velocity sign)")
    if v0 != 0 and np.sign(v0) != s_v:
        raise PhysicalResponseError("v0 sign disagrees with the declared sliding sign")
    t = np.asarray(t, float)
    lam = c / M
    f = u0 - G0 - s_v * Fc
    x = lam * t
    small = np.abs(x) < 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        p1 = np.where(small, 1 - x / 2 + x * x / 6, -np.expm1(-x) / np.where(small, 1, x))
        p2 = np.where(small, 0.5 - x / 6 + x * x / 24, (x + np.expm1(-x)) / np.where(small, 1, x * x))
    a = f / M
    q = q0 + v0 * t * p1 + a * t * t * p2
    v = v0 * np.exp(-x) + a * t * p1
    # zero crossing (only when the net effort opposes the motion)
    t_stop = None
    if v0 != 0 and f * v0 < 0:
        if c <= 0:
            t_stop = -v0 * M / f
        else:
            vinf = f / c
            t_stop = -(1 / lam) * math.log(vinf / (vinf - v0))
    stuck_after = abs(u0 - G0) <= Fc
    stopped = np.zeros(t.shape, bool)
    if t_stop is not None:
        xs = lam * t_stop
        if abs(xs) < 1e-6:
            q_s = q0 + v0 * t_stop * (1 - xs / 2) + a * t_stop ** 2 * (0.5 - xs / 6)
        else:
            q_s = q0 + v0 * t_stop * (-math.expm1(-xs) / xs) + a * t_stop ** 2 * ((xs + math.expm1(-xs)) / xs ** 2)
        stopped = t >= t_stop
        q = np.where(stopped, q_s if stuck_after else np.nan, q)
        v = np.where(stopped, 0.0 if stuck_after else np.nan, v)
    return dict(q=q, v=v, t_stop=t_stop, stopped=stopped, stuck_after_stop=bool(stuck_after) if t_stop else None,
                continues_after_stop=bool(stuck_after) if t_stop is not None else True,
                v_inf=(f / c) if c > 0 else None, lam=lam, net_effort=f,
                note=("terminated at the zero-velocity crossing; held at rest (|u0 - G0| <= Fc)" if t_stop is not None
                      and stuck_after else "terminated at the zero-velocity crossing; breakaway / reversed-sign "
                      "model required (NaN after t_stop)" if t_stop is not None else "no zero crossing"))


def steady_velocity(*, u0, G0, Fc, c, s_v):
    """v_inf = (u0 - G0 - s_v Fc)/c; None for c = 0 (no terminal velocity: constant acceleration)."""
    return None if c <= 0 else (u0 - G0 - s_v * Fc) / c
