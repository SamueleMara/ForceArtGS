"""Wrench frame/sign handling and physically constrained force evidence (spec section 8).

Sign convention used throughout: ``f_obj`` is the force exerted BY THE ROBOT ON THE OBJECT, in
the world frame. The recorded real datasets store the *reaction on the robot*
(``metadata/contact_force_convention.json``: ``object_side_force_multiplier = -1``), so the
conversion negates once. Force ratios are reported but are never a joint classifier on their
own; directional predictions are only made with an independently supported contact normal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rigid_kinematics import contact_jacobian, normalize


@dataclass(frozen=True)
class WrenchConvention:
    recorded_quantity: str = "object_on_robot"   # 'object_on_robot' | 'robot_on_object'
    gravity_compensated: bool = False
    tare_applied: bool = False
    torque_origin_known: bool = False
    source: str = "unspecified"

    @property
    def object_side_multiplier(self) -> float:
        if self.recorded_quantity == "object_on_robot":
            return -1.0
        if self.recorded_quantity == "robot_on_object":
            return 1.0
        raise ValueError(f"unknown recorded wrench quantity {self.recorded_quantity!r}; refusing to guess a sign")

    @property
    def calibrated(self) -> bool:
        return self.gravity_compensated and self.tare_applied

    @classmethod
    def from_scene_metadata(cls, convention_json: dict, robot_tool_json: dict | None = None):
        q = convention_json.get("recorded_quantity", "")
        rq = "object_on_robot" if ("on_robot" in q or convention_json.get("force_acting_on") == "robot") else (
            "robot_on_object" if convention_json.get("force_acting_on") == "object" else "unknown")
        wf = (robot_tool_json or {}).get("wrench_frame", {})
        return cls(rq, gravity_compensated=False, tare_applied=False,
                   torque_origin_known=not str(wf.get("torque_origin", "unspecified")).startswith("recorded_origin_unspecified"),
                   source="metadata/contact_force_convention.json (no tare/gravity compensation declared)")


def force_on_object_world(f_sensor, R_WS, convention: WrenchConvention):
    """``f_obj^W = s * R_WS f^S`` (rotation only; a force vector is never translated)."""
    f = np.einsum("...ij,...j->...i", np.asarray(R_WS, float), np.asarray(f_sensor, float))
    return convention.object_side_multiplier * f


def torque_about_tcp_world(f_sensor, tau_sensor, R_WS, p_sensor_w, p_tcp_w, convention: WrenchConvention):
    """``tau_TCP = s (R tau_S) + (p_S - p_TCP) x f_obj``; requires a known torque origin."""
    if not convention.torque_origin_known:
        raise ValueError("torque origin not known; refusing to transport the moment to the TCP")
    f_obj = force_on_object_world(f_sensor, R_WS, convention)
    tau = convention.object_side_multiplier * np.einsum("...ij,...j->...i", np.asarray(R_WS, float), tau_sensor)
    return tau + np.cross(np.asarray(p_sensor_w) - np.asarray(p_tcp_w), f_obj)


def decompose_force(f_obj, v_tcp, eps_v=0.002, eps_f=1.0) -> dict:
    """Signed F_par along the TCP tangent, perpendicular vector/magnitude, both ratios.

    Samples with |v| <= eps_v or |f| <= eps_f (or NaN) are excluded with a reason."""
    f = np.asarray(f_obj, float)
    v = np.asarray(v_tcp, float)
    fn = np.linalg.norm(f, axis=-1)
    vn = np.linalg.norm(v, axis=-1)
    reason = np.full(len(f), "", dtype=object)
    reason[~np.isfinite(fn)] = "force_missing"
    reason[(reason == "") & ~np.isfinite(vn)] = "tcp_velocity_missing"
    reason[(reason == "") & (fn <= eps_f)] = "force_below_threshold"
    reason[(reason == "") & (vn <= eps_v)] = "tcp_speed_below_threshold"
    valid = reason == ""
    t = np.where(valid[:, None], v / np.where(vn > 0, vn, 1)[:, None], np.nan)
    Fpar = np.einsum("ti,ti->t", f, t)
    Fperp_vec = f - Fpar[:, None] * t
    Fperp = np.linalg.norm(Fperp_vec, axis=1)
    fs = np.where(fn > 0, fn, 1)
    return dict(valid=valid, reason=reason, F_parallel=Fpar, F_perp_vec=Fperp_vec, F_perp=Fperp,
                cos_phi=Fpar / fs, rho_parallel=np.abs(Fpar) / fs, rho_perp=Fperp / fs,
                f_hat=np.where(valid[:, None], f / fs[:, None], np.nan), force_norm=fn, tangent=t)


def generalized_force(kind, axis, origin, p_tcp, f_obj):
    """``Q_H = f^T J_c,H(p)``: N*m (revolute) or N (prismatic). Not comparable across kinds."""
    J = contact_jacobian(kind, axis, origin, p_tcp)
    return np.einsum("ti,ti->t", np.asarray(f_obj, float), J)


def power_terms(kind, axis, origin, p_tcp, v_tcp, f_obj, qdot) -> dict:
    Q = generalized_force(kind, axis, origin, p_tcp, f_obj)
    P_joint = Q * np.asarray(qdot, float)
    P_tcp = np.einsum("ti,ti->t", np.asarray(f_obj, float), np.asarray(v_tcp, float))
    return dict(Q=Q, P_joint=P_joint, P_tcp=P_tcp, D=P_tcp - P_joint,
                Q_units="N*m" if kind == "revolute" else "N")


def passivity_penalty(D, sigma, delta=0.0, active=None):
    """One-sided ``sum max(0, -D - delta)^2 / sigma^2`` (dissipation must be >= 0)."""
    D = np.asarray(D, float)
    m = np.isfinite(D) if active is None else (np.asarray(active, bool) & np.isfinite(D))
    viol = np.maximum(0.0, -D[m] - delta)
    return dict(penalty=float(np.sum((viol / sigma) ** 2)), n_active=int(m.sum()), n_violations=int((viol > 0).sum()),
                min_D=float(np.min(D[m])) if m.any() else None)


def coulomb_direction(n_in, slip_tangential, mu, eps_slip=1e-3):
    """``f_hat ~ n_in + mu * s_T/|s_T|``; NaN (undefined) when sticking (|s_T| <= eps_slip)."""
    n_in = np.asarray(n_in, float)
    s = np.asarray(slip_tangential, float)
    sn = np.linalg.norm(s, axis=-1)
    stick = sn <= eps_slip
    dirn = n_in + mu * s / np.where(stick, 1.0, sn)[..., None]
    out = dirn / np.linalg.norm(dirn, axis=-1, keepdims=True)
    out[stick] = np.nan
    return out, stick


def predicted_force_directions(kind, axis, q, n0_out, slip, mu, eps_slip=1e-3):
    """Predicted robot-on-object force direction at each time for one hypothesis.

    ``n0_out`` is the canonical OUTWARD face normal; the robot pushes INTO the object, so
    ``n_in = -R_H(q) n0_out``. The object's tangential friction on the tool opposes the tool's
    slip, so the force ON the object has a ``+mu * s_T`` component."""
    from .tcp_contact import surface_normal_world
    n_out = surface_normal_world(kind, axis, q, n0_out)
    n_in = -n_out
    s = np.asarray(slip, float)
    sT = s - np.einsum("ti,ti->t", s, n_out)[:, None] * n_out
    return coulomb_direction(n_in, sT, mu, eps_slip)


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = np.einsum("ti,ti->t", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def fit_shared_mu(f_hat_meas, kind, axis, q, n0_out, slip, active, mu_max=1.5, n_grid=61, eps_slip=1e-3):
    """One shared, bounded mu per hypothesis by grid search on the mean angular error."""
    best = (np.inf, 0.0, None)
    for mu in np.linspace(0.0, mu_max, n_grid):
        pred, stick = predicted_force_directions(kind, axis, q, n0_out, slip, mu, eps_slip)
        m = np.asarray(active, bool) & ~stick & np.isfinite(f_hat_meas).all(1)
        if m.sum() == 0:
            return dict(mu=None, mean_angle_deg=None, n=0, reason="no sliding samples with valid force")
        err = float(np.mean(angle_deg(f_hat_meas[m], pred[m])))
        if err < best[0]:
            best = (err, mu, m)
    return dict(mu=float(best[1]), mean_angle_deg=float(best[0]), n=int(best[2].sum()))


def force_direction_evidence(f_obj, contact_kin_by_kind: dict, candidates: dict, surface_by_kind: dict, active,
                             cfg, sigma_deg=15.0) -> dict:
    """Compare measured full force directions against each hypothesis' Coulomb prediction.

    Abstains (``status = ABSTAIN``) unless every hypothesis has an independently supported surface
    and enough sliding, valid-force samples. Same samples for all hypotheses."""
    fn = np.linalg.norm(f_obj, axis=1)
    f_hat = f_obj / np.where(fn > 0, fn, 1)[:, None]
    out = dict(per_kind={})
    for kind, c in candidates.items():
        surf = surface_by_kind.get(kind)
        if surf is None or surf.status != "OK":
            return dict(status="ABSTAIN", reason=f"no independently supported contact surface for {kind}")
    act = np.asarray(active, bool) & (fn > cfg.min_force_n)
    for kind, c in candidates.items():
        fit = fit_shared_mu(f_hat, kind, c.axis_w, c.q, surface_by_kind[kind].n0, contact_kin_by_kind[kind]["slip"],
                            act, cfg.mu_max)
        out["per_kind"][kind] = fit
    ns = [v["n"] for v in out["per_kind"].values()]
    if min(ns) < 3:
        return dict(out, status="ABSTAIN", reason=f"too few sliding samples with valid force ({ns})")
    out["status"] = "OK"
    out["samples"] = int(min(ns))
    return out


def null_controls(f_obj, rng, shift=5):
    """Wrench null controls: shuffled in time, sign reversed, and time shifted."""
    f = np.asarray(f_obj, float)
    return dict(shuffled=f[rng.permutation(len(f))], sign_reversed=-f, time_shifted=np.roll(f, shift, axis=0))
