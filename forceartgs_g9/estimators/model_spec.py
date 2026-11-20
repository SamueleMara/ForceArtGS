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

SOURCES = ("geometry_coupled", "joint_space_only", "user_link_inertial")


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

    def __post_init__(self):
        if self.parameter_source not in SOURCES[:2]:
            raise ModelConfigError(f"parameter_source {self.parameter_source!r}: estimators support "
                                   f"{SOURCES[:2]} (user_link_inertial is a consistency check, not a fit)")
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
        lo = {"alpha": 1e-12, "M": 1e-12, "c": 0.0, "Fc": 0.0, "dFs": 0.0, "Fs": 0.0,
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
        if t.gravity:
            if self.parameter_source == "geometry_coupled":
                gu = self.mass_model.gravity_unit(self.gravity.g_world)
                A, B = gu.basis_coefficients()
                out["gravity"] = dict(kind="geometry_coupled", A=v["alpha"] * A, B=v["alpha"] * B,
                                      amplitude=v["alpha"] * gu.amplitude_bound())
            elif self.joint_type == "revolute":
                out["gravity"] = dict(kind="effective_basis", A=v["gA"], B=v["gB"],
                                      amplitude=math.hypot(v["gA"], v["gB"]))
            else:
                out["gravity"] = dict(kind="effective_basis", A=v["gP"], B=0.0, amplitude=abs(v["gP"]))
        return out

    def G_fn(self, theta):
        if not self.terms.gravity:
            return None
        v = self.values(theta)
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
