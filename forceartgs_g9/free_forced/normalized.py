"""ONE normalized 1-DoF joint model for every estimator (weak form, PINN, EKF, rollout) and both joint types.

Dimensional moving-mode effort (constant effective M > 0):

    M qdd + c qd + Fc sgn(qd) + G(q) [+ tau_mech] = Q(t)

Divided by M (the only admissible normalization; M > 0 is a structural fact, not a fitted bound):

    qdd + lam qd + gam sgn(qd) + a_g(q) [+ a_mech] = beta Q(t),
    lam = c/M [1/s],  gam = Fc/M [q/s^2],  beta = 1/M,  a_g = G/M.

FREE intervals carry beta Q ~ 0: beta (hence M) is ABSENT from the free model -- a free trajectory determines at
most (lam, gam, a_g); (M, c, Fc) -> (kM, kc, kFc) is an exact symmetry. beta enters only with a trusted forced input.

The rollout reuses the validated hybrid integrator ``simulate_general`` with the exact conversion
M = 1, c = lam, Fc = gam, G = a_g, effort = beta Q (no second integrator). Static breakaway (Fs) is NOT part of the
free model: at rest the joint sticks while |beta Q - a_g| <= gam (kinetic level as the stick threshold).

Gravity (dataset base_link z = operator-declared true vertical, g_B = (0, 0, -9.81)):
    prismatic, single rigid mass (M = m):   a_g = -g.a                      (mass cancels)
    revolute, uniform-density shape:        a_g(q) = G_1(q) / I_axis,1       (density cancels)
    revolute with reflected inertia I_r:    a_g = rho G_1 / (rho I_1 + I_r)  (depends on the scale -> flagged)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

G_STD = 9.80665

UNITS = {
    "revolute": dict(q="rad", v="rad/s", a="rad/s^2", lam="1/s", gam="rad/s^2", beta="1/(kg m^2)", M="kg m^2",
                     Q="N m", J="N m s", c="N m s/rad", Fc="N m"),
    "prismatic": dict(q="m", v="m/s", a="m/s^2", lam="1/s", gam="m/s^2", beta="1/kg", M="kg", Q="N", J="N s",
                      c="N s/m", Fc="N"),
}


class NormalizedModelError(ValueError):
    pass


class GravityModel:
    """Normalized configuration load a_g(q) = A cos q + B sin q (revolute, stored coordinate q, q_ref folded in) or a
    constant (prismatic). ``kind``: known_shape | negligible | vertical_exact | effective | prismatic_ideal | zero."""

    def __init__(self, name, kind, joint_type, A=0.0, B=0.0, provenance="", depends_on_scale=False, extra=None):
        if joint_type not in UNITS:
            raise NormalizedModelError(f"joint_type {joint_type!r}")
        self.name, self.kind, self.joint_type = name, kind, joint_type
        self.A, self.B = float(A), float(B)
        self.provenance, self.depends_on_scale = provenance, bool(depends_on_scale)
        self.extra = dict(extra or {})

    @property
    def is_zero(self):
        return self.A == 0.0 and self.B == 0.0

    def __call__(self, q):
        q = np.asarray(q, float)
        if self.joint_type == "prismatic":
            out = np.full(q.shape, self.A)
        else:
            out = self.A * np.cos(q) + self.B * np.sin(q)
        return out if out.ndim else float(out)

    def deriv(self, q):
        q = np.asarray(q, float)
        if self.joint_type == "prismatic":
            out = np.zeros(q.shape)
        else:
            out = -self.A * np.sin(q) + self.B * np.cos(q)
        return out if out.ndim else float(out)

    def torch(self, q):
        import torch
        if self.joint_type == "prismatic":
            return torch.full_like(q, self.A)
        return self.A * torch.cos(q) + self.B * torch.sin(q)

    def amplitude(self):
        return math.hypot(self.A, self.B) if self.joint_type == "revolute" else abs(self.A)

    def fast(self):
        """Scalar closure for the integrator inner loop."""
        A, B = self.A, self.B
        if self.joint_type == "prismatic":
            return lambda q: A
        cos, sin = math.cos, math.sin
        return lambda q: A * cos(q) + B * sin(q)

    def reversed(self):
        """The same load in the reversed coordinate q' = -q (axis a' = -a): a_g'(q') = -a_g(-q')."""
        if self.joint_type == "prismatic":
            return GravityModel(self.name + "__reversed", self.kind, self.joint_type, -self.A, 0.0, self.provenance,
                                self.depends_on_scale, self.extra)
        return GravityModel(self.name + "__reversed", self.kind, self.joint_type, -self.A, self.B, self.provenance,
                            self.depends_on_scale, self.extra)

    def describe(self):
        return dict(name=self.name, kind=self.kind, joint_type=self.joint_type, A=self.A, B=self.B,
                    form="a_g(q) = A cos q + B sin q" if self.joint_type == "revolute" else "a_g = A (constant)",
                    amplitude=self.amplitude(), unit=UNITS[self.joint_type]["a"], provenance=self.provenance,
                    depends_on_absolute_scale=self.depends_on_scale, **self.extra)

    # ------------------------------------------------------------------ constructors
    @staticmethod
    def zero(joint_type, name="zero", kind="zero", provenance=""):
        return GravityModel(name, kind, joint_type, 0.0, 0.0, provenance)

    @staticmethod
    def prismatic_ideal(axis, g_world, name="prismatic_ideal"):
        a = np.asarray(axis, float)
        a = a / np.linalg.norm(a)
        g = np.asarray(g_world, float)
        return GravityModel(name, "prismatic_ideal", "prismatic", float(-g @ a), 0.0,
                            "a_g = -g.a (single rigid translating mass, M = m, no reflected / carried mass)")

    @staticmethod
    def from_mass_model(mm, g_world, name="known_shape", rho=None):
        """a_g from the SAME unit-density geometry as the inertia. With zero reflected inertia the density cancels;
        otherwise ``rho`` is required and the model is flagged depends_on_scale."""
        gu = mm.gravity_unit(g_world)
        A1, B1 = gu.basis_coefficients()
        Ir = mm.reflected()
        if Ir == 0.0:
            return GravityModel(name, "known_shape", mm.joint_type, A1 / mm.M1, B1 / mm.M1,
                                f"G_1(q)/M_1 of the {mm.props.mode} geometry (density cancels; q_ref {mm.q_ref:.6g}; "
                                "TENTATIVE shape, DECLARED vertical)", extra=dict(M1=mm.M1, lever_m=mm.lever_m))
        if rho is None:
            raise NormalizedModelError("reflected inertia != 0: normalized gravity depends on the density "
                                       "(rho G_1 / (rho I_1 + I_refl)); supply rho (scale) explicitly")
        den = rho * mm.M1 + Ir
        return GravityModel(name, "known_shape", mm.joint_type, rho * A1 / den, rho * B1 / den,
                            f"rho G_1/(rho M_1 + I_refl), rho = {rho:g}, I_refl = {Ir:g}", depends_on_scale=True,
                            extra=dict(M1=mm.M1, rho=rho, I_refl=Ir))


@dataclass
class NormalizedDynamicsSpec:
    """Shared normalized model. ``forced`` adds beta = 1/M (only with a trusted forced input)."""

    joint_type: str
    gravity: GravityModel
    viscous: bool = True
    coulomb: bool = True
    forced: bool = False
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.joint_type not in UNITS:
            raise NormalizedModelError(f"joint_type {self.joint_type!r}")
        if self.gravity.joint_type != self.joint_type:
            raise NormalizedModelError("gravity model joint type differs")
        if not (self.viscous or self.coulomb):
            raise NormalizedModelError("at least one dissipative term (the free model is otherwise parameter-free)")

    @property
    def names(self):
        n = (["lam"] if self.viscous else []) + (["gam"] if self.coulomb else [])
        return n + (["beta"] if self.forced else [])

    @property
    def units(self):
        return UNITS[self.joint_type]

    def free(self):
        return NormalizedDynamicsSpec(self.joint_type, self.gravity, self.viscous, self.coulomb, False,
                                      dict(self.provenance))

    def with_forced(self):
        return NormalizedDynamicsSpec(self.joint_type, self.gravity, self.viscous, self.coulomb, True,
                                      dict(self.provenance))

    def with_gravity(self, g):
        return NormalizedDynamicsSpec(self.joint_type, g, self.viscous, self.coulomb, self.forced,
                                      dict(self.provenance))

    def unpack(self, theta) -> dict:
        th = list(np.asarray(theta, float))
        out = dict(lam=0.0, gam=0.0, beta=0.0)
        for n, x in zip(self.names, th):
            out[n] = float(x)
        return out

    def pack(self, lam=0.0, gam=0.0, beta=None):
        d = dict(lam=lam, gam=gam, beta=beta)
        return np.array([d[n] for n in self.names], float)

    # ------------------------------------------------------------------ model equation
    def accel(self, q, v, s, theta, Q=0.0):
        p = self.unpack(theta)
        return -p["lam"] * v - p["gam"] * s - self.gravity(q) + p["beta"] * Q

    def residual(self, q, qd, qdd, s, theta, Q=0.0):
        """qdd + lam qd + gam s + a_g(q) - beta Q (zero for the model)."""
        return np.asarray(qdd) - self.accel(q, qd, s, theta, Q)

    # ------------------------------------------------------------------ rollout (validated hybrid integrator)
    def sim_params(self, theta):
        from ..dynamics.general_1dof import GeneralTerms, sim_params
        p = self.unpack(theta)
        if p["lam"] < 0 or p["gam"] < 0 or p["beta"] < 0:
            raise NormalizedModelError(f"negative normalized rate {p}")
        grav = not self.gravity.is_zero
        t = GeneralTerms(inertia=True, viscous=self.viscous, coulomb=self.coulomb, static=False, gravity=grav,
                         bias=False)
        return sim_params(t, self.joint_type, M=1.0, c=p["lam"] if self.viscous else None,
                          Fc=p["gam"] if self.coulomb else None, G=self.gravity.fast() if grav else None)

    def simulate(self, theta, t_query, t0, q0, v0, effort=None, max_step=2e-3):
        """Normalized hybrid rollout from the DECLARED (t0, q0, v0). ``effort``: ZOH of the MEASURED Q (scaled by
        beta here); None = zero input (free)."""
        from ..dynamics.general_1dof import simulate_general
        from ..synthetic import ZOH
        p = self.unpack(theta)
        if effort is None or not self.forced:
            eff = ZOH(np.array([float(t0)]), np.array([0.0]))
        else:
            eff = ZOH(np.asarray(effort.t_s, float), p["beta"] * np.asarray(effort.levels, float))
        return simulate_general(self.sim_params(theta), eff, np.asarray(t_query, float), t0=float(t0), q0=float(q0),
                                v0=float(v0), max_step=max_step)

    # ------------------------------------------------------------------ physical reconstruction (scale needed)
    def physical(self, theta, M, M_sd=None, cov=None, provenance="EXPLICIT_PRIOR_NOT_MEASURED"):
        """c = lam M, Fc = gam M -- ONLY with an absolute scale M of declared provenance."""
        if not (M is not None and M > 0 and math.isfinite(M)):
            raise NormalizedModelError("physical reconstruction needs a positive, finite absolute scale M")
        p = self.unpack(theta)
        u = self.units
        out = dict(M=dict(value=M, sd=M_sd, unit=u["M"]), provenance=provenance)
        sd = {} if cov is None else {n: math.sqrt(max(cov[i][i], 0.0)) for i, n in enumerate(self.names)}
        for n, phys in (("lam", "c"), ("gam", "Fc")):
            if n in self.names:
                v = p[n] * M
                var = (M * sd.get(n, 0.0)) ** 2 + (p[n] * (M_sd or 0.0)) ** 2
                out[phys] = dict(value=v, sd=math.sqrt(var) if (sd or M_sd) else None, unit=u[phys])
        return out

    def describe(self):
        u = self.units
        return dict(schema="forceartgs.g9.normalized_model_spec/v1", joint_type=self.joint_type,
                    equation="qdd + lam qd + gam sgn(qd) + a_g(q) = beta Q(t)" if self.forced else
                    "qdd + lam qd + gam sgn(qd) + a_g(q) = 0   (FREE: beta absent)",
                    parameters={n: dict(meaning={"lam": "c/M", "gam": "Fc/M", "beta": "1/M"}[n], unit=u[n],
                                        bound=">= 0" if n != "beta" else "> 0 (forced stage only)")
                                for n in self.names},
                    gravity=self.gravity.describe(), viscous=self.viscous, coulomb=self.coulomb, forced=self.forced,
                    static_breakaway="NOT MODELLED (free coasting cannot identify Fs; stick threshold = gam at rest)",
                    integrator="simulate_general (validated hybrid): M = 1, c = lam, Fc = gam, G = a_g, "
                               "effort = beta Q",
                    scale_symmetry="(M, c, Fc) -> (kM, kc, kFc) leaves every free trajectory unchanged",
                    units=u, provenance=self.provenance)

    # ------------------------------------------------------------------ bridge from the legacy ModelSpec
    @classmethod
    def from_model_spec(cls, spec, forced=False, gravity=None):
        """Adapter over the existing ModelSpec: same terms (viscous / coulomb), same mass geometry and gravity
        registration -> the normalized a_g. Joint-space effective gravity bases have no normalized meaning without
        the scale and are refused (pass an explicit GravityModel instead)."""
        t = spec.terms
        if gravity is None:
            if not t.gravity:
                gravity = GravityModel.zero(spec.joint_type, provenance="ModelSpec: gravity term disabled")
            elif spec.parameter_source == "geometry_coupled":
                gravity = GravityModel.from_mass_model(spec.mass_model, spec.gravity.g_world)
            elif spec.joint_type == "prismatic" and spec.parameter_source == "joint_space_gravity_coupled":
                gravity = GravityModel.prismatic_ideal(spec.joint_axis, spec.gravity.g_world)
            else:
                raise NormalizedModelError(f"ModelSpec gravity source {spec.parameter_source!r} has no normalized "
                                           "form without an absolute scale; pass an explicit GravityModel")
        return cls(spec.joint_type, gravity, bool(t.viscous), bool(t.coulomb), forced,
                   dict(derived_from="ModelSpec", model_id=t.model_id, parameter_source=spec.parameter_source))


def a_scale(seg_list):
    """A data-derived acceleration scale [q/s^2] for normalizing residuals: range / duration^2."""
    vals = []
    for s in seg_list:
        T = float(s.t[-1] - s.t[0])
        if T > 0:
            vals.append(max(float(np.ptp(s.q)), 1e-9) / T ** 2)
    return float(np.median(vals)) if vals else 1.0
