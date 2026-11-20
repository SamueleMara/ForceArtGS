"""General one-degree-of-freedom rigid joint dynamics with explicit on/off terms (G9 spec section 2).

    M q'' + c q' + Fc sign(q') + G(q) + b = Q_robot(t)          while |q'| > 0
    q' = 0 and |Q_robot - G(q) - b| <= Fs  ->  q'' = 0             (sticking, set-valued static reaction)

M  effective inertia about the axis [kg m^2] (revolute) or effective mass [kg] (prismatic), constant (ideal 1-DoF
   rigid mechanism: no Coriolis / centrifugal term exists for one joint coordinate).
c  viscous damping [N m s/rad | N s/m]; Fc kinetic Coulomb [N m | N]; Fs static / breakaway level [N m | N], Fs >= Fc;
b  constant NUISANCE generalized load (never a sensor tare, never exported as friction);
G  dU/dq, the derivative of the potential energy U(q) = -m g_W . c_W(q)  (NOT the applied gravity torque):
   G_R(q) = -m g_W . (a x (c(q) - o)),   G_P = -m g_W . a.

Disabled terms contribute identically zero and their parameters are ``None`` (``disabled_by_user``), never 0.0
estimates. Inertia may be disabled only in an explicit quasistatic mode, which has no dynamic rollout.

Integrator (``simulate_general``): an event-driven exponential integrator. Between events the net non-viscous effort
f = Q - G(q_k) - b - Fc s is held constant on a sub-step (Q is a forward zero-order hold; G is frozen at the sub-step
start, the only discretisation error, controlled by ``max_step``) and the linear part M q'' + c q' = f is solved in
closed form (the same cancellation-free phi functions as the validated viscous solver). Velocity zero crossings are
located analytically; with dry friction the joint stops exactly there and the Karnopp set-valued stick test decides
whether it stays at rest (|net| <= Fs) or breaks away in the sign of the net effort. No sign(0), no smoothing, and
dry friction can never accelerate the joint (no force-free energy creation).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

TERMS = ("inertia", "viscous", "coulomb", "static", "gravity", "bias")
PARAM_OF_TERM = {"inertia": "M", "viscous": "c", "coulomb": "Fc", "static": "Fs", "gravity": "G", "bias": "b"}
UNITS = {
    "revolute": {"q": "rad", "qdot": "rad/s", "effort": "N*m", "M": "kg*m^2", "c": "N*m*s/rad", "Fc": "N*m",
                 "Fs": "N*m", "b": "N*m", "alpha": "density scale (see mass model)"},
    "prismatic": {"q": "m", "qdot": "m/s", "effort": "N", "M": "kg", "c": "N*s/m", "Fc": "N", "Fs": "N", "b": "N",
                  "alpha": "density scale (see mass model)"},
}
STANDARD_GRAVITY = 9.80665


class ModelConfigError(ValueError):
    """Invalid term combination or parameter set (fails closed)."""


@dataclass(frozen=True)
class GeneralTerms:
    inertia: bool = True
    viscous: bool = True
    coulomb: bool = True
    static: bool = True
    gravity: bool = True
    bias: bool = False
    quasistatic: bool = False

    def __post_init__(self):
        for k in TERMS + ("quasistatic",):
            if not isinstance(getattr(self, k), bool):
                raise ModelConfigError(f"terms.{k} must be true/false")
        if not self.inertia and not self.quasistatic:
            raise ModelConfigError("inertia OFF is only allowed in the explicit quasistatic mode (M = 0 is not a "
                                   "valid rigid link for a dynamic rollout)")
        if self.quasistatic and self.inertia:
            raise ModelConfigError("quasistatic mode neglects inertia: set terms.inertia false")

    @classmethod
    def from_dict(cls, d: dict | None) -> "GeneralTerms":
        d = dict(d or {})
        bad = sorted(set(d) - set(TERMS) - {"quasistatic"})
        if bad:
            raise ModelConfigError(f"unknown dynamics term(s) {bad}; allowed {list(TERMS)} + quasistatic")
        return cls(**d)

    @property
    def enabled(self) -> tuple:
        return tuple(t for t in TERMS if getattr(self, t))

    def to_dict(self) -> dict:
        return asdict(self)

    def status(self) -> dict:
        return {t: ("enabled" if getattr(self, t) else "disabled_by_user") for t in TERMS}

    @property
    def model_id(self) -> str:
        return "general_rigid_1dof[" + ",".join(self.enabled) + ("|quasistatic" if self.quasistatic else "") + "]"


FULL = GeneralTerms()
VISCOUS_ONLY = GeneralTerms(coulomb=False, static=False, gravity=False, bias=False)   # the ForceSAPIEN model


# ----------------------------------------------------------------------------------------------- gravity
def rotation(axis, q) -> np.ndarray:
    """Right-hand rotation by q [rad] about the unit ``axis`` (Rodrigues)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)


@dataclass(frozen=True)
class GravityField:
    """Physical gravitational acceleration in the calibrated world W [m/s^2] with its provenance. ``verified`` must
    come from a measured base-to-gravity registration; arbitrary capture axes are never assumed gravity aligned."""

    g_world: tuple
    verified: bool
    provenance: str
    frame: str = "W"

    def __post_init__(self):
        g = np.asarray(self.g_world, float)
        if g.shape != (3,) or not np.isfinite(g).all():
            raise ModelConfigError("gravity vector must be 3 finite numbers [m/s^2]")


@dataclass(frozen=True)
class RigidBodyGravity:
    """Exact rigid-body gravity of one moving body: U(q) = -m g.c(q); G(q) = dU/dq. ``mass`` may be a unit-density
    mass m_1 (geometry-coupled G_1, scaled by alpha later)."""

    joint_type: str
    axis: tuple
    origin: tuple            # a point on the hinge (revolute); unused for prismatic
    com0: tuple              # COM at the canonical q = 0
    mass: float
    g_world: tuple

    def com(self, q) -> np.ndarray:
        a, o, c0 = (np.asarray(x, float) for x in (self.axis, self.origin, self.com0))
        if self.joint_type == "prismatic":
            return c0 + float(q) * a / np.linalg.norm(a)
        return o + rotation(a, float(q)) @ (c0 - o)

    def potential(self, q) -> float:
        return float(-self.mass * np.asarray(self.g_world, float) @ self.com(q))

    def G(self, q):
        """dU/dq, vectorised over q."""
        g = np.asarray(self.g_world, float)
        a = np.asarray(self.axis, float)
        a = a / np.linalg.norm(a)
        if self.joint_type == "prismatic":
            v = float(-self.mass * g @ a)
            return np.full(np.shape(q), v) if np.ndim(q) else v
        o, c0 = np.asarray(self.origin, float), np.asarray(self.com0, float)
        r = c0 - o
        r_perp = r - (r @ a) * a
        A = -self.mass * g @ np.cross(a, r_perp)           # G = A cos q + B sin q (exact, see docstring)
        B = self.mass * g @ r_perp
        qq = np.asarray(q, float)
        out = A * np.cos(qq) + B * np.sin(qq)
        return out if np.ndim(q) else float(out)

    def basis_coefficients(self) -> tuple:
        """(A, B) with G(q) = A cos q + B sin q (revolute) or (G_P, 0) (prismatic)."""
        if self.joint_type == "prismatic":
            return float(self.G(0.0)), 0.0
        return float(self.G(0.0)), float(self.G(math.pi / 2))

    def amplitude_bound(self) -> float:
        """max_q |G(q)| = m |g_perp| |r_perp| (revolute), |G_P| (prismatic)."""
        A, B = self.basis_coefficients()
        return float(math.hypot(A, B)) if self.joint_type == "revolute" else abs(A)


# ----------------------------------------------------------------------------------------------- parameters
@dataclass
class SimParams:
    """Numerical parameters of ONE simulation. Built from a terms selection by ``sim_params``: disabled terms are
    exactly zero here (they cannot leak into a rollout)."""

    joint_type: str
    M: float
    c: float = 0.0
    Fc: float = 0.0
    Fs: Optional[float] = None
    b: float = 0.0
    G: Optional[Callable] = None
    terms: Optional[GeneralTerms] = None
    meta: dict = field(default_factory=dict)


def sim_params(terms: GeneralTerms, joint_type: str, *, M=None, c=None, Fc=None, Fs=None, b=None, G=None) -> SimParams:
    """Validate a parameter set against the terms. Enabled terms need a finite value of the right sign; a value
    passed for a DISABLED term is an error (no silent reuse of a gravity-ON fit in a gravity-OFF rollout)."""
    if joint_type not in UNITS:
        raise ModelConfigError(f"joint_type {joint_type!r}")
    if terms.quasistatic:
        raise ModelConfigError("quasistatic model: no dynamic rollout (use quasistatic_equilibrium)")
    vals = dict(M=M, c=c, Fc=Fc, Fs=Fs, b=b, G=G)
    for t in TERMS:
        p = PARAM_OF_TERM[t]
        on = getattr(terms, t)
        if not on and vals[p] is not None:
            raise ModelConfigError(f"{p} given for disabled term {t!r} (disabled terms are null, not reused)")
        if on and vals[p] is None:
            raise ModelConfigError(f"term {t!r} enabled but {p} is missing")
    for p in ("M", "c", "Fc", "Fs", "b"):
        v = vals[p]
        if v is not None and (isinstance(v, bool) or not np.isfinite(float(v))):
            raise ModelConfigError(f"{p} = {v!r} is not finite")
    if not float(M) > 0:
        raise ModelConfigError(f"M = {M} must be > 0 for a dynamic rigid body")
    for p in ("c", "Fc", "Fs"):
        if vals[p] is not None and float(vals[p]) < 0:
            raise ModelConfigError(f"{p} = {vals[p]} must be >= 0")
    if terms.static and terms.coulomb and float(Fs) < float(Fc):
        raise ModelConfigError(f"static level Fs = {Fs} < kinetic Fc = {Fc}: not a physical breakaway model")
    if G is not None and not callable(G):
        raise ModelConfigError("G must be a callable G(q)")
    return SimParams(joint_type, float(M), float(c or 0.0), float(Fc or 0.0), None if Fs is None else float(Fs),
                     float(b or 0.0), G, terms)


# ----------------------------------------------------------------------------------------------- integrator
def _phi(x):
    if x < 1e-2:
        return (1 - x / 2 + x * x / 6 - x ** 3 / 24 + x ** 4 / 120,
                .5 - x / 6 + x * x / 24 - x ** 3 / 120 + x ** 4 / 720)
    em1 = math.expm1(-x)
    return -em1 / x, (x + em1) / (x * x)


def _lin_step(q, v, f, M, c, h):
    """Exact solution of M q'' + c q' = f (constant) over h."""
    x = (c / M) * h
    p1, p2 = _phi(x)
    a = f / M
    return q + v * h * p1 + a * h * h * p2, v * math.exp(-x) + a * h * p1


def _stop_time(v, f, M, c, h):
    """Time in (0, h] at which the velocity of M q'' + c q' = f reaches 0 (None if it does not)."""
    if v == 0 or f * v >= 0:
        return None                                         # the net effort does not oppose the motion
    if c <= 0:
        tau = -v * M / f
    else:
        vinf = f / c                                        # opposite sign to v
        tau = -(M / c) * math.log(vinf / (vinf - v))
    return tau if 0 < tau <= h else None


@dataclass
class Rollout:
    t_s: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    stuck: np.ndarray
    events: list
    n_steps: int


def _effort_fn(effort):
    tf, lev = np.asarray(effort.t_s, float), np.asarray(effort.levels, float)

    def at(t):
        i = int(np.searchsorted(tf, t, side="right")) - 1
        return float(lev[min(max(i, 0), len(lev) - 1)])
    return tf, at


def simulate_general(p: SimParams, effort, t_query, t0: float = 0.0, q0: float = 0.0, v0: float = 0.0,
                     max_step: float = 2e-3, record_events: bool = False) -> Rollout:
    """Roll the hybrid model out from the DECLARED state (t0, q0, v0). ``effort`` is a forward-ZOH generalized
    effort (``levels[k]`` on [t_s[k], t_s[k+1]), first level also before t_s[0]), e.g.
    forceartgs_physics.forces.projection.GeneralizedEffort."""
    if not (max_step > 0 and np.isfinite(max_step)):
        raise ModelConfigError("max_step must be > 0")
    tq = np.asarray(t_query, float)
    if np.any(tq < t0 - 1e-12):
        raise ModelConfigError("query times must not precede t0")
    order = np.argsort(tq, kind="stable")
    tf = np.asarray(effort.t_s, float).tolist()
    lev = np.asarray(effort.levels, float).tolist()
    nk = len(tf)
    M, c, Fc, b = p.M, p.c, p.Fc, p.b
    Fs = p.Fs
    dry = Fc > 0 or (Fs is not None and Fs > 0)
    thr = Fs if Fs is not None else Fc                      # stick threshold at rest
    G = p.G
    q, v, t = float(q0), float(v0), float(t0)
    out_q, out_v, out_s = np.empty(len(tq)), np.empty(len(tq)), np.zeros(len(tq), bool)
    events, n_steps = [], 0
    ki = 0                                                  # index of the active ZOH level: tf[ki] <= t < tf[ki+1]
    while ki + 1 < nk and tf[ki + 1] <= t + 1e-15:
        ki += 1
    inf = float("inf")
    for idx in order:
        tn = float(tq[idx])
        while t < tn - 1e-15:
            while ki + 1 < nk and tf[ki + 1] <= t + 1e-15:
                ki += 1
            nxt = tf[ki + 1] if ki + 1 < nk else inf
            t_end = min(tn, t + max_step, nxt)
            h = t_end - t
            Q = lev[ki]
            g = G(q) if G is not None else 0.0
            net = Q - g - b
            n_steps += 1
            if v == 0.0:
                if dry and abs(net) <= thr:
                    t = t_end                               # sticking: set-valued reaction holds the joint
                    continue
                if net == 0.0:
                    t = t_end
                    continue
                s = 1.0 if net > 0 else -1.0                # breakaway in the sign of the net applied effort
                if record_events:
                    events.append(dict(t_s=t, kind="breakaway", net=net))
            else:
                s = 1.0 if v > 0 else -1.0
            f = net - Fc * s
            if G is not None:
                # Heun predictor-corrector on the configuration load (2nd order in h): trial step with G(q_k),
                # then the trapezoidal mean of G over the step
                q1, _ = _lin_step(q, v, f, M, c, h)
                g1 = G(q1)
                net = Q - 0.5 * (g + g1) - b
                f = net - Fc * s
            if dry:
                tau = _stop_time(v, f, M, c, h) if v != 0.0 else None
                if v == 0.0:
                    # starting from rest: kinetic effort f has the sign of s (|net| > thr >= Fc), no stop in (0, h]
                    tau = None
                if tau is not None:
                    q, _ = _lin_step(q, v, f, M, c, tau)
                    v = 0.0
                    t = t + tau
                    if record_events:
                        events.append(dict(t_s=t, kind="stop"))
                    continue
            q, v = _lin_step(q, v, f, M, c, h)
            t = t_end
        out_q[idx], out_v[idx], out_s[idx] = q, v, (v == 0.0 and dry)
    return Rollout(tq, out_q, out_v, out_s, events, n_steps)


def quasistatic_equilibrium(terms: GeneralTerms, Q, G=None, b=0.0, Fc=0.0, Fs=None) -> dict:
    """Quasistatic mode (inertia OFF): equilibrium residual Q - G - b against the friction set; no rollout and no
    dynamic certification."""
    if not terms.quasistatic:
        raise ModelConfigError("quasistatic_equilibrium requires terms.quasistatic")
    Q = np.asarray(Q, float)
    r = Q - (0.0 if G is None else np.asarray(G, float)) - b
    thr = Fs if Fs is not None else Fc
    return dict(residual=r, within_friction_set=np.abs(r) <= thr, dynamic_rollout="NOT_APPLICABLE_QUASISTATIC")


def mechanical_energy(p: SimParams, q, v, potential: Callable | None = None) -> np.ndarray:
    """Kinetic + potential energy (for no-energy-creation checks)."""
    q, v = np.asarray(q, float), np.asarray(v, float)
    U = 0.0 if potential is None else np.array([potential(x) for x in q])
    return 0.5 * p.M * v * v + U
