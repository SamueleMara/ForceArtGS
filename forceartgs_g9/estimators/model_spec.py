"""Model specification shared by the general estimators: terms, parameter source and the theta -> rollout map.

parameter_source
  geometry_coupled : theta has ``alpha`` (density scale). M = alpha M1 + M_reflected, G(q) = alpha G_1(q) with the
                     validated mass model and a VERIFIED gravity vector. One scale for inertia AND gravity.
  joint_space_only : theta has ``M`` (effective joint inertia / mass). Gravity, if enabled, is an EFFECTIVE basis
                     load (revolute: gA cos q + gB sin q; prismatic: constant gP) that is never converted into a
                     mass / COM and never exported as a link inertial.
Remaining parameters: c (viscous), Fc (Coulomb), Fs (static; stored as dFs = Fs - Fc >= 0 when Coulomb is on), b
(nuisance). Disabled terms have no parameter at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..dynamics.general_1dof import GeneralTerms, GravityField, ModelConfigError, SimParams, sim_params
from ..dynamics.geometry_inertia import GeometryMassModel

SOURCES = ("geometry_coupled", "joint_space_only", "joint_space_gravity_coupled", "user_link_inertial")
FIT_SOURCES = SOURCES[:3]


def fast_gravity(joint_type, A, B):
    """Scalar closure G(q) = A cos q + B sin q (revolute) or A (prismatic): the exact rigid-body form (see
    RigidBodyGravity.G), evaluated with math for the integrator inner loop."""
    if joint_type == "prismatic":
        return lambda q: A
    cos, sin = math.cos, math.sin
    return lambda q: A * cos(q) + B * sin(q)


class GravityRegistrationError(ModelConfigError):
    """Gravity-dependent physical identification without a verified gravity registration."""


@dataclass
class G9Interaction:
    """One demonstration: observed q(t) + generalized effort, same clock, same joint."""

    name: str
    joint_type: str
    t_obs: np.ndarray
    q_obs: np.ndarray
    q_std: np.ndarray
    effort: object                   # forward-ZOH effort with .t_s, .levels, .at
    initial_state: Optional[dict] = None     # DECLARED {t_s, q, qdot, source}; None -> nuisance q0/v0 fitted
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t_obs = np.asarray(self.t_obs, float)
        self.q_obs = np.asarray(self.q_obs, float)
        s = np.asarray(self.q_std, float)
        self.q_std = np.full(len(self.t_obs), float(s)) if s.ndim == 0 else s
        if not (len(self.t_obs) == len(self.q_obs) == len(self.q_std)) or len(self.t_obs) < 4:
            raise ValueError(f"{self.name}: t_obs / q_obs / q_std mismatch or < 4 samples")
        if np.any(np.diff(self.t_obs) <= 0) or not np.isfinite(self.q_obs).all() or np.any(self.q_std <= 0):
            raise ValueError(f"{self.name}: t_obs must increase, q finite, q_std > 0")


@dataclass
class ModelSpec:
    terms: GeneralTerms
    joint_type: str
    parameter_source: str = "geometry_coupled"
    mass_model: Optional[GeometryMassModel] = None
    gravity: Optional[GravityField] = None
    allow_unverified_gravity: bool = False          # diagnostic only; recorded
    fixed_priors: dict = field(default_factory=dict)   # name -> {value, std, label ASSUMED|MEASURED, source}
    # joint_space_gravity_coupled (no mesh): the joint geometry + the unit direction from the hinge towards the moving
    # part's COM (assumed = moving-support centroid direction, perpendicular to the axis)
    joint_axis: Optional[np.ndarray] = None
    joint_origin: Optional[np.ndarray] = None
    com_direction: Optional[np.ndarray] = None
    com_lever_m: Optional[float] = None

    def __post_init__(self):
        if self.parameter_source not in FIT_SOURCES:
            raise ModelConfigError(f"parameter_source {self.parameter_source!r}: estimators support "
                                   f"{FIT_SOURCES} (user_link_inertial is a consistency check, not a fit)")
        if self.parameter_source == "joint_space_gravity_coupled" and self.terms.gravity:
            if self.gravity is None:
                raise GravityRegistrationError("joint_space_gravity_coupled needs a gravity vector")
            if not self.gravity.verified and not self.allow_unverified_gravity:
                raise GravityRegistrationError("GRAVITY_REGISTRATION_UNVERIFIED (declared gravity is accepted only "
                                               "under the exploratory policy)")
            if self.joint_axis is None:
                raise ModelConfigError("joint_space_gravity_coupled needs the joint axis")
            if self.joint_type == "revolute" and (self.joint_origin is None or self.com_direction is None):
                raise ModelConfigError("revolute joint_space_gravity_coupled needs the hinge origin and the COM "
                                       "direction (moving-support centroid)")
        if self.parameter_source == "geometry_coupled":
            if self.mass_model is None:
                raise ModelConfigError("geometry_coupled needs a validated mass model (else joint_space_only)")
            if self.mass_model.joint_type != self.joint_type:
                raise ModelConfigError("mass model joint type differs from the model")
            if self.terms.gravity:
                if self.gravity is None:
                    raise GravityRegistrationError("gravity term enabled but no gravity vector supplied")
                if not self.gravity.verified and not self.allow_unverified_gravity:
                    raise GravityRegistrationError(
                        "GRAVITY_REGISTRATION_UNVERIFIED: gravity-dependent physical identification refused (supply a "
                        "measured base-to-gravity registration, or disable the gravity term)")
        for k, v in self.fixed_priors.items():
            if v.get("label") not in ("ASSUMED", "MEASURED"):
                raise ModelConfigError(f"prior {k}: label must be ASSUMED or MEASURED")

    # ------------------------------------------------------------------ parameter layout
    @property
    def names(self) -> list:
        t, out = self.terms, []
        if t.inertia:
            out.append("alpha" if self.parameter_source == "geometry_coupled" else "M")
        if t.viscous:
            out.append("c")
        if t.coulomb:
            out.append("Fc")
        if t.static:
            out.append("dFs" if t.coulomb else "Fs")
        if t.gravity and self.parameter_source == "joint_space_only":
            out += ["gA", "gB"] if self.joint_type == "revolute" else ["gP"]
        if t.gravity and self.parameter_source == "joint_space_gravity_coupled" and self.joint_type == "revolute":
            out.append("mu")                      # first mass moment m r about the hinge [kg m]; prismatic: G = -M g.a
        if t.bias:
            out.append("b")
        return [n for n in out if n not in self.fixed_priors]

    @property
    def all_names(self) -> list:
        keep = self.fixed_priors
        self.fixed_priors = {}
        try:
            return self.names
        finally:
            self.fixed_priors = keep

    def lower_bounds(self) -> np.ndarray:
        lo = {"alpha": 1e-12, "M": 1e-12, "c": 0.0, "Fc": 0.0, "dFs": 0.0, "Fs": 0.0, "mu": 0.0,
              "gA": -np.inf, "gB": -np.inf, "gP": -np.inf, "b": -np.inf}
        return np.array([lo[n] for n in self.names])

    def values(self, theta) -> dict:
        v = {n: float(x) for n, x in zip(self.names, theta)}
        for k, p in self.fixed_priors.items():
            v[k] = float(p["value"])
        return v

    def physical(self, theta) -> dict:
        """theta -> named physical parameters (M, c, Fc, Fs, b; alpha; gravity description)."""
        v = self.values(theta)
        t = self.terms
        out = dict(alpha=None, M=None, c=None, Fc=None, Fs=None, b=None, gravity=None)
        if t.inertia:
            if self.parameter_source == "geometry_coupled":
                out["alpha"] = v["alpha"]
                out["M"] = v["alpha"] * self.mass_model.M1 + self.mass_model.reflected()
            else:
                out["M"] = v["M"]
        if t.viscous:
            out["c"] = v["c"]
        if t.coulomb:
            out["Fc"] = v["Fc"]
        if t.static:
            out["Fs"] = v["Fc"] + v["dFs"] if t.coulomb else v["Fs"]
        if t.bias:
            out["b"] = v["b"]
        if t.gravity and self.parameter_source == "joint_space_gravity_coupled":
            A, B = self._g_js_coeffs(v)
            out["gravity"] = dict(kind="declared_gravity_coupled", name="gravity_torque_declared_vertical", A=A, B=B,
                                  amplitude=math.hypot(A, B), physical_gravity=True,
                                  gravity_status="DECLARED (operator: dataset z is the true vertical)",
                                  mu_kg_m=v.get("mu"),
                                  coupling=("G_P = -M g.a (single rigid body: effective mass = moving mass)"
                                            if self.joint_type == "prismatic" else
                                            "G_R = mu g . (a x R(q) u), u = hinge -> moving-support centroid"))
            if self.joint_type == "revolute" and self.com_lever_m:
                out["gravity"]["consistency_M_ge_mu_r"] = dict(M=out["M"], mu_r=v.get("mu", 0) * self.com_lever_m,
                                                               ok=bool(out["M"] is None or out["M"] >=
                                                                       v.get("mu", 0) * self.com_lever_m),
                                                               note="point-mass lower bound I >= m r^2 = mu r, with "
                                                                    "the COM assumed at the support-centroid lever")
        elif t.gravity:
            if self.parameter_source == "geometry_coupled":
                gu = self.mass_model.gravity_unit(self.gravity.g_world)
                A, B = gu.basis_coefficients()
                out["gravity"] = dict(kind="geometry_coupled", A=v["alpha"] * A, B=v["alpha"] * B,
                                      amplitude=v["alpha"] * gu.amplitude_bound())
            elif self.joint_type == "revolute":
                out["gravity"] = dict(kind="effective_basis", name="effective_configuration_load", A=v["gA"],
                                      B=v["gB"], amplitude=math.hypot(v["gA"], v["gB"]), physical_gravity=False)
            else:
                out["gravity"] = dict(kind="effective_basis", name="effective_configuration_load", A=v["gP"], B=0.0,
                                      amplitude=abs(v["gP"]), physical_gravity=False)
        return out

    def _g_js_coeffs(self, v):
        from ..dynamics.general_1dof import RigidBodyGravity
        g = np.asarray(self.gravity.g_world, float)
        a = np.asarray(self.joint_axis, float)
        a = a / np.linalg.norm(a)
        if self.joint_type == "prismatic":
            M = v.get("M", 0.0) if "M" in v else 0.0
            return float(-M * g @ a), 0.0
        o = np.asarray(self.joint_origin, float)
        u = np.asarray(self.com_direction, float)
        rb = RigidBodyGravity("revolute", tuple(a), tuple(o), tuple(o + u), 1.0, tuple(g))
        A1, B1 = rb.basis_coefficients()                    # unit mass at unit lever along u
        mu = v.get("mu", 0.0)
        return mu * A1, mu * B1

    def G_fn(self, theta):
        if not self.terms.gravity:
            return None
        v = self.values(theta)
        if self.parameter_source == "joint_space_gravity_coupled":
            A, B = self._g_js_coeffs(v)
            return fast_gravity(self.joint_type, A, B)
        if self.parameter_source == "geometry_coupled":
            A1, B1 = self._g_unit_coeffs()
            A, B = v["alpha"] * A1, v["alpha"] * B1
        elif self.joint_type == "revolute":
            A, B = v["gA"], v["gB"]
        else:
            A, B = v["gP"], 0.0
        return fast_gravity(self.joint_type, A, B)

    def _g_unit_coeffs(self):
        key = (id(self.mass_model), tuple(self.gravity.g_world))
        if getattr(self, "_gcache", (None,))[0] != key:
            gu = self.mass_model.gravity_unit(self.gravity.g_world)
            self._gcache = (key, gu.basis_coefficients())
        return self._gcache[1]

    def sim(self, theta) -> SimParams:
        p = self.physical(theta)
        t = self.terms
        return sim_params(t, self.joint_type, M=p["M"], c=p["c"] if t.viscous else None,
                          Fc=p["Fc"] if t.coulomb else None, Fs=p["Fs"] if t.static else None,
                          b=p["b"] if t.bias else None, G=self.G_fn(theta))

    def describe(self) -> dict:
        return dict(model_id=self.terms.model_id, terms=self.terms.status(), parameter_source=self.parameter_source,
                    free_parameters=self.names, fixed_priors=self.fixed_priors,
                    gravity=None if self.gravity is None else dict(g_world=list(self.gravity.g_world),
                                                                   verified=self.gravity.verified,
                                                                   provenance=self.gravity.provenance),
                    allow_unverified_gravity=self.allow_unverified_gravity,
                    mass_model=None if self.mass_model is None else dict(
                        mode=self.mass_model.props.mode, M1=self.mass_model.M1, m1=self.mass_model.props.m1,
                        reflected_actuator_inertia=self.mass_model.reflected_actuator_inertia))
