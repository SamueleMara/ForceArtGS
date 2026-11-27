"""Opt-in rest-to-rest terminal-speed objective for G10 (V1, Eq. 13). The default PhysX position objective (V0) is
untouched: with weight 0 this is EXACTLY RefinementLoss.

    L1(theta) = L_q(theta) + lambda_T * huber_delta( qdot_PX(T; theta) / sigma_vT )

L_q is the existing RefinementLoss (same samples, numeric floor, Huber delta, clock). qdot_PX(T) is the engine's own
joint velocity at the LAST episode sample (= T), read from SimulationResult.qdot -- never imposed: the state is never
overwritten, the simulated stop has to EMERGE from (rho, c) under the recorded wrench. sigma_vT is derived once from
the terminal visual plateau (estimators.rest_to_rest.terminal_speed_sigma). The term is a physically motivated
ENGINEERING constraint correlated with the q-plateau samples, not an independent Gaussian measurement.
"""
from __future__ import annotations

import numpy as np

from .behavioral_fit import CachingRollout
from .objective import RefinementLoss, _huber


class TerminalVelocityLoss(RefinementLoss):
    def __init__(self, rollout, episodes, settings=None, *, weight=0.0, sigma_vT=1.0):
        if not isinstance(rollout, CachingRollout):          # the terminal read must reuse the SAME engine result
            rollout = CachingRollout(rollout)
        super().__init__(rollout, episodes, settings)
        if len(self.episodes) != 1:
            raise ValueError("the rest-to-rest terminal objective is defined on ONE full-demonstration episode")
        if not (weight >= 0 and np.isfinite(weight)) or not (sigma_vT > 0 and np.isfinite(sigma_vT)):
            raise ValueError("terminal weight >= 0 and sigma_vT > 0 required")
        self.weight, self.sigma_vT = float(weight), float(sigma_vT)

    def terminal(self, theta) -> dict:
        res = self.rollout(theta, self.episodes[0])
        if res.qdot is None:
            raise ValueError("engine returned no qdot: the terminal speed cannot be read")
        vT = float(np.asarray(res.qdot, float)[-1])
        z = vT / self.sigma_vT
        return dict(v_sim_T=vT, z=z, term=self.weight * float(_huber(np.asarray(z), self.settings.huber_delta)))

    def evaluate(self, theta, v0_offsets=None):
        out = super().evaluate(theta, v0_offsets)
        if self.weight == 0.0:
            return out                                       # V0: bitwise the existing objective
        tv = self.terminal(theta)
        out = dict(out, objective=out["objective"] + tv["term"], terminal_velocity=tv,
                   terminal_weight=self.weight, sigma_vT=self.sigma_vT)
        return out


def factory(weight, sigma_vT):
    """objective_factory for behavioral_fit.fit_full_demo: (rollout, [pe], settings) -> V1 objective."""
    def make(rollout, episodes, settings):
        return TerminalVelocityLoss(rollout, episodes, settings, weight=weight, sigma_vT=sigma_vT)
    return make
