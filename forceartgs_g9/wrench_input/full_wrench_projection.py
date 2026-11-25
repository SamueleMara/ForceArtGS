"""Full six-axis wrench -> generalized joint effort (spec FORCEARTGS_G10_WRENCH_DRIVEN_BEHAVIORAL_PHYSX, Eqs. 1-3).

Experimental facts (E1-E3, given, not re-estimated): the sensor-to-global transform is KNOWN, the wrench is already
TARED at the start, the robot is in contact throughout. So this module has no calibration, tare, contact-detection or
contact-point option. A spatial wrench is (F, tau) with tau the moment ABOUT a reference point ``ref``; its
representation is tagged (frame: sensor | base, side: robot_reaction | object) so the known transform and the
object-side sign are each applied EXACTLY once (a second application raises):

    F_B = s R_BS F_S,  tau_B = s R_BS tau_S          (moment about p_BS; s = existing object-side multiplier)   (3)
    Q_R = a . [ tau_B + (p_BS - o) x F_B ]           (revolute, about the hinge line)                            (1)
    Q_P = a . F_B                                    (prismatic; a moment does no work on a pure translation)   (2)

No object contact point appears anywhere: the measured 6D wrench is the net transmitted object wrench.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WRENCH_ORDER = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")      # archive 'wrench' columns (force N, moment N m)
FRAMES = ("sensor", "base")
SIDES = ("robot_reaction", "object")


class WrenchContractError(ValueError):
    pass


def _rot(R):
    R = np.asarray(R, float)
    if R.ndim == 2:
        R = R[None]
    if R.shape[-2:] != (3, 3) or not np.isfinite(R).all():
        raise WrenchContractError(f"rotation(s) must be finite [N,3,3], got {R.shape}")
    E = np.einsum("nji,njk->nik", R, R) - np.eye(3)
    if np.abs(E).max() > 1e-5 or np.min(np.linalg.det(R)) < 1 - 1e-4:
        raise WrenchContractError("not a proper rotation")
    return R


@dataclass(frozen=True)
class SpatialWrench:
    F: np.ndarray                  # [N, 3] N
    tau: np.ndarray                # [N, 3] N m, ABOUT ref
    ref: np.ndarray                # [N, 3] m, reference point of tau, expressed in ``frame`` (base) or None (sensor)
    frame: str
    side: str
    history: tuple = field(default_factory=tuple)

    def __post_init__(self):
        for k in ("F", "tau"):
            a = np.asarray(getattr(self, k), float)
            if a.ndim != 2 or a.shape[1] != 3:
                raise WrenchContractError(f"{k} must be [N, 3]")
            object.__setattr__(self, k, a)
        if self.F.shape != self.tau.shape:
            raise WrenchContractError("force / moment length mismatch")
        if self.frame not in FRAMES or self.side not in SIDES:
            raise WrenchContractError(f"frame {self.frame!r} / side {self.side!r}")
        if self.ref is not None:
            r = np.asarray(self.ref, float)
            r = np.broadcast_to(r, self.F.shape).copy() if r.ndim == 1 else r
            if r.shape != self.F.shape:
                raise WrenchContractError("reference point length mismatch")
            object.__setattr__(self, "ref", r)
        if self.frame == "base" and self.ref is None:
            raise WrenchContractError("a base-frame spatial wrench needs its reference point")

    @classmethod
    def from_archive(cls, W, *, frame="sensor", side="robot_reaction", ref=None):
        W = np.asarray(W, float)
        if W.ndim != 2 or W.shape[1] != 6:
            raise WrenchContractError(f"wrench must be [N, 6] in the order {WRENCH_ORDER}")
        return cls(W[:, :3], W[:, 3:], ref, frame, side, ("as_recorded",))


def sensor_to_base(w: SpatialWrench, R_BS, p_BS, sign: float) -> SpatialWrench:
    """The KNOWN transform + the EXISTING object-side sign, each applied once (Eq. 3). Refuses a base-frame or an
    already object-side input (no double transform / double sign)."""
    if w.frame != "sensor":
        raise WrenchContractError("wrench is already in the base frame: refusing a second transform")
    if w.side != "robot_reaction":
        raise WrenchContractError("wrench is already object-side: refusing a second sign application")
    if float(sign) not in (-1.0, 1.0):
        raise WrenchContractError(f"object-side multiplier must be +-1, got {sign!r}")
    R = _rot(R_BS)
    if R.shape[0] not in (1, len(w.F)):
        raise WrenchContractError("rotation count mismatch")
    p = np.asarray(p_BS, float)
    p = np.broadcast_to(p, w.F.shape).copy() if p.ndim == 1 else p
    F = float(sign) * np.einsum("nij,nj->ni", R if R.shape[0] > 1 else np.repeat(R, len(w.F), 0), w.F)
    T = float(sign) * np.einsum("nij,nj->ni", R if R.shape[0] > 1 else np.repeat(R, len(w.F), 0), w.tau)
    return SpatialWrench(F, T, p, "base", "object", w.history + (f"R_BS + sign {float(sign):+.0f} (once)",))


def transport(w: SpatialWrench, new_ref) -> SpatialWrench:
    """Same physical wrench about another point: tau_new = tau + (ref - new_ref) x F."""
    if w.frame != "base":
        raise WrenchContractError("transport in the base frame")
    n = np.asarray(new_ref, float)
    n = np.broadcast_to(n, w.F.shape).copy() if n.ndim == 1 else n
    return SpatialWrench(w.F, w.tau + np.cross(w.ref - n, w.F), n, "base", w.side, w.history + ("transported",))


def joint_effort(w: SpatialWrench, joint_type: str, axis, origin=None) -> np.ndarray:
    """Eqs. (1) / (2) on an object-side base-frame spatial wrench."""
    if w.frame != "base" or w.side != "object":
        raise WrenchContractError("joint effort needs the object-side base-frame wrench")
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    if joint_type == "prismatic":
        return w.F @ a
    if joint_type != "revolute":
        raise WrenchContractError(f"joint_type {joint_type!r}")
    if origin is None:
        raise WrenchContractError("revolute effort needs the hinge origin")
    o = np.asarray(origin, float)
    return (w.tau + np.cross(w.ref - o, w.F)) @ a


def effort_sigma(R_BS, ref, sigma_F_S, sigma_T_S, joint_type, axis, origin=None) -> np.ndarray:
    """sd(Q) per row through the LINEAR projection with a DIAGONAL sensor-axis noise model (per-axis sigmas only;
    no verified 6x6 covariance exists -- reported as an approximation). Revolute: Q = a.tau_B + a.(r x F_B) =
    a.tau_B + (a x r).F_B with r = ref - o, so dQ/dtau_B = a and dQ/dF_B = a x r; prismatic: dQ/dF_B = a. The
    object-side sign does not change the variance."""
    R = _rot(R_BS)
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    sF = np.diag(np.asarray(sigma_F_S, float) ** 2)
    sT = np.diag(np.asarray(sigma_T_S, float) ** 2)
    if joint_type == "prismatic":
        g = np.broadcast_to(a, (len(R), 3))
        return np.sqrt(np.einsum("ni,nij,jk,nlk,nl->n", g, R, sF, R, g))
    r = np.asarray(ref, float) - np.asarray(origin, float)
    gF = np.cross(a, r)                                  # a.(r x F) = (a x r).F
    gT = np.broadcast_to(a, gF.shape)
    vF = np.einsum("ni,nij,jk,nlk,nl->n", gF, R, sF, R, gF)
    vT = np.einsum("ni,nij,jk,nlk,nl->n", gT, R, sT, R, gT)
    return np.sqrt(vF + vT)
