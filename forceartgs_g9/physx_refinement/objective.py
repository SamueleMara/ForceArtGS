"""G10: objective for *real* PhysX rollouts, not a custom ODE substituted for PhysX.

The backend is a callable `(PhysicalParameters, Episode) -> SimulationResult`.
Free-motion intervals have zero applied effort, not zero initial velocity.
Do not infer absolute inertia from free motion; provide explicit scale provenance.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import numpy as np


def _a(x):
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True)
class Episode:
    name: str
    t_s: np.ndarray
    q: np.ndarray
    sigma_q: np.ndarray
    q0: float
    v0: float
    input_t_s: np.ndarray
    input_effort: np.ndarray
    mode: str             # free | forced; unknown-contact is NEVER fit
    calibration_status: str = "UNVERIFIED"
    used_for_training: bool = True
    exploratory_assumptions_accepted: bool = False
    sigma_v0: float | None = None
    # what 'mode' means for THIS episode (provenance only; legacy default = a classifier-segmented regime window).
    # The behavioral full-demo adapter sets 'behavioral_full_demo': mode 'forced' then means "the recorded
    # generalized effort is applied throughout", NOT that contact was proven at every instant.
    regime_semantics: str = "segmented_regime_window"

    def __post_init__(self):
        for k in ("t_s", "q", "sigma_q", "input_t_s", "input_effort"):
            object.__setattr__(self, k, _a(getattr(self, k)))
        if self.mode not in ("free", "forced"):
            raise ValueError("mode must be free or forced; split mixed windows explicitly")
        if len(self.t_s) < 3 or len(self.q) != len(self.t_s) or len(self.sigma_q) != len(self.t_s):
            raise ValueError("joint trajectory and uncertainties must have equal length >= 3")
        if np.any(np.diff(self.t_s) <= 0) or np.any(self.sigma_q <= 0):
            raise ValueError("strictly increasing times and positive uncertainties required")
        if len(self.input_t_s) == 0 or len(self.input_t_s) != len(self.input_effort) or np.any(np.diff(self.input_t_s) <= 0):
            raise ValueError("input force timeline must be nonempty and monotonic")
        if self.input_t_s[0] > self.t_s[0]:
            raise ValueError("input must start no later than the joint trajectory")
        if self.mode == "forced" and self.input_t_s[-1] > self.t_s[-1] + 1e-10:
            # Later force samples are allowed (they can be outside this window).
            pass
        if not all(np.isfinite(x).all() for x in (self.t_s,self.q,self.sigma_q,self.input_t_s,self.input_effort)):
            raise ValueError("nonfinite episode data")
        if self.sigma_v0 is not None and (not np.isfinite(self.sigma_v0) or self.sigma_v0 <= 0):
            raise ValueError("sigma_v0 must be positive")
        if self.mode == "free" and np.max(np.abs(self.input_effort)) > 1e-12:
            raise ValueError("free interval must explicitly have zero applied generalized effort")
        if self.mode == "forced" and self.calibration_status != "VERIFIED" and not self.exploratory_assumptions_accepted:
            raise ValueError("forced exploratory input requires explicit assumption opt-in")

    def effort_at(self, t):
        t = _a(t)
        i = np.searchsorted(self.input_t_s, t, side="right") - 1
        if np.any(i < 0) or np.any(t > self.t_s[-1] + 1e-10):
            raise ValueError("attempted input extrapolation")
        return self.input_effort[np.clip(i, 0, len(self.input_effort)-1)]


@dataclass(frozen=True)
class PhysicalParameters:
    """Mass scaling preserves COM and inertia consistency; armature is extra joint inertia.

    `rho_scale` multiplies the asset's baseline link mass and entire COM inertia tensor.
    `armature` is kg m^2 for revolute and kg for prismatic joints.
    `static_effort`, `dynamic_effort` are N*m for revolute, N for prismatic.
    `viscous` is N*m*s/rad or N*s/m (NOT USD's degree-based representation).
    """
    rho_scale: float
    armature: float
    static_effort: float
    dynamic_effort: float
    viscous: float
    inertia_scale_source: str = "PRIOR_ONLY"

    def __post_init__(self):
        nums = (self.rho_scale,self.armature,self.static_effort,self.dynamic_effort,self.viscous)
        if not all(math.isfinite(float(v)) for v in nums):
            raise ValueError("all parameters must be finite")
        if self.rho_scale <= 0 or min(self.armature,self.static_effort,self.dynamic_effort,self.viscous) < 0:
            raise ValueError("positive link mass and nonnegative armature/friction required")
        if self.static_effort < self.dynamic_effort:
            raise ValueError("PhysX requires static effort >= dynamic effort")


@dataclass(frozen=True)
class SimulationResult:
    q: np.ndarray
    qdot: np.ndarray | None = None
    backend: str = "unknown"
    engine_version: str | None = None
    runtime_semantics: str = "UNVERIFIED"
    diagnostics: dict | None = None


@dataclass(frozen=True)
class LossSettings:
    huber_delta: float = 2.0
    q_weight: float = 1.0
    increment_weight: float = 0.0   # correlated residuals: off by default
    scale_prior_weight: float = 0.0
    prior_log_rho: float | None = None
    numeric_floor_q: float = 0.0
    equal_episode_weight: bool = True
    fit_initial_velocity: bool = False


def _huber(z, delta):
    r = np.abs(z)
    return np.where(r <= delta, 0.5*z*z, delta*(r-0.5*delta))


class RefinementLoss:
    """Black-box objective over actual PhysX trajectories; no surrogate dynamics in this class.

    A solver may optimize this with Powell / differential_evolution. Only `train` episodes
    influence the objective. `evaluate` exposes per-episode and train/heldout scores.
    Untrusted forced data are rejected at ingestion. Parameters are never marked
    identified here; that requires profile/observability and heldout evidence.
    """
    def __init__(self, rollout: Callable[[PhysicalParameters, Episode], SimulationResult],
                 episodes: list[Episode], settings: LossSettings | None = None):
        self.rollout = rollout
        self.episodes = tuple(episodes)
        self.settings = settings or LossSettings()
        if not self.episodes:
            raise ValueError("at least one episode is required")
        if len({e.name for e in self.episodes}) != len(self.episodes):
            raise ValueError("unique episode names required")

    def evaluate(self, theta: PhysicalParameters, v0_offsets: dict[str, float] | None = None):
        from dataclasses import replace
        c = self.settings
        v0_offsets = v0_offsets or {}
        per = {}
        velocity_prior = 0.0
        for ep in self.episodes:
            dv = float(v0_offsets.get(ep.name, 0.0))
            if dv != 0.0 and (not c.fit_initial_velocity or ep.sigma_v0 is None):
                raise ValueError("initial velocity fitting requires explicit sigma_v0 and opt-in")
            if dv and ep.used_for_training:
                velocity_prior += 0.5 * (dv / ep.sigma_v0)**2
            if dv and not ep.used_for_training:
                raise ValueError("held-out initial velocities must not be fitted")
            ep_run = replace(ep, v0=ep.v0+dv) if dv else ep
            sim = self.rollout(theta, ep_run)
            q = _a(sim.q)
            if q.shape != ep.q.shape or not np.isfinite(q).all():
                raise ValueError(f"invalid PhysX rollout for {ep.name}")
            sigma = np.hypot(ep.sigma_q, c.numeric_floor_q)
            z = (q - ep.q) / sigma
            l_pos = float(np.mean(_huber(z, c.huber_delta)))
            l_inc = 0.0
            if c.increment_weight:
                inc = np.diff(q) - np.diff(ep.q)
                # conservative diagonal approximation; neighboring increments are correlated
                inc_sig = np.sqrt(sigma[1:]**2 + sigma[:-1]**2)
                l_inc = float(np.mean(_huber(inc/inc_sig, c.huber_delta)))
            per[ep.name] = {
                "mode":ep.mode, "train":ep.used_for_training,
                "loss":c.q_weight*l_pos + c.increment_weight*l_inc,
                "n_samples":int(len(q)),
                "position_loss":l_pos, "increment_loss":l_inc,
                "rmse_rad_or_m":float(np.sqrt(np.mean((q-ep.q)**2))),
                "rmse_over_sigma":float(np.sqrt(np.mean(z*z))),
                "engine":sim.backend, "version":sim.engine_version,
                "runtime_semantics":sim.runtime_semantics,
            }
        train = [p["loss"] for p in per.values() if p["train"]]
        held = [p["loss"] for p in per.values() if not p["train"]]
        if not train:
            raise ValueError("no training episodes")
        prior = 0.0
        if c.scale_prior_weight:
            if c.prior_log_rho is None:
                raise ValueError("prior weight requires declared prior_log_rho")
            prior = c.scale_prior_weight*(math.log(theta.rho_scale)-c.prior_log_rho)**2
        prior += velocity_prior / max(len(train),1)
        return {"objective":float(np.mean(train)+prior), "train_position":float(np.mean(train)),
                "heldout_position":float(np.mean(held)) if held else None,
                "prior_penalty":prior, "episodes":per,
                "absolute_scale_identified":False,
                "reason":"optimizer objective alone cannot establish identifiability"}

    def __call__(self, theta: PhysicalParameters):
        return self.evaluate(theta)["objective"]

    def objective_many(self, thetas):
        """Objectives of many candidates (None for a failed rollout). When the rollout can ``prefetch`` (a batched
        engine, e.g. PhysX GPU), every episode is simulated for all candidates in one batched call first; the loss
        itself is the SAME ``evaluate`` as for a single candidate."""
        thetas = list(thetas)
        if hasattr(self.rollout, "prefetch"):
            for ep in self.episodes:
                self.rollout.prefetch(thetas, ep)
        out = []
        for th in thetas:
            try:
                out.append(self.evaluate(th)["objective"])
            except (ValueError, FloatingPointError, RuntimeError):
                out.append(None)
        return out


def effective_joint_inertia(*, joint_type: str, axis, hinge_origin, com,
                            unit_mass: float, unit_inertia_com, rho_scale: float, armature: float) -> float:
    """Revolute parallel-axis theorem; prismatic moving mass. All in one frame."""
    a = _a(axis).reshape(3); a = a / np.linalg.norm(a)
    if joint_type == "prismatic":
        return float(unit_mass*rho_scale + armature)
    if joint_type != "revolute":
        raise ValueError("joint_type must be revolute/prismatic")
    r = _a(com).reshape(3)-_a(hinge_origin).reshape(3)
    rp = r - a*np.dot(a,r)
    return float(rho_scale*(a@_a(unit_inertia_com).reshape(3,3)@a + unit_mass*np.dot(rp,rp))+armature)


def rotational_viscous_to_usd_degree_units(c_nm_s_per_rad: float) -> float:
    """USD angular-friction velocity is in degrees/s; coefficient N*m*s/degree."""
    return float(c_nm_s_per_rad)*math.pi/180.0


def normalized_free_dynamics(I: float, c: float, Td: float) -> tuple[float,float]:
    if I <= 0:
        raise ValueError("positive effective inertia required")
    return c/I, Td/I
