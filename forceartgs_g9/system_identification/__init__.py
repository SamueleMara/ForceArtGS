"""Physics-constrained input-output identification of one-DOF articulated parts (estimator side, opt-in).

Layer between the G9 analytical / weak-form / PINN / EKF estimators and the G10 PhysX refinement. It fits the
minimal passive joint equation  M qdd + c qd + Fc sgn(qd) + G(q) = u(t)  in its NORMALIZED form
(lam = c/M, gam = Fc/M, g_u = 1/M, kappa = K_eff/M) to the measured q(t) driven by the measured u(t) (output error
on the native clocks, forward-ZOH input), reports the local sliding-regime transfer function and its poles, audits
work / impulse consistency, and hands warm starts + a conditional candidate family to the G10 PhysX refinement.

Nothing here re-tares, re-projects or sign-changes force, reclassifies contact, or writes a simulator ``value``.
Schema: forceartgs.g9.io_response/v1.
"""
from .contracts import SCHEMA, ResponseFeature, IOEpisode, load_config  # noqa: F401
