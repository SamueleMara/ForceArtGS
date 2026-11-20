"""Held-out (independent-episode) prediction with FROZEN parameters (G9 spec 4.4, C13/C14).

Only the declared initial state (q0, q'0 at t0) and the measured generalized effort of the new episode are used;
its future q is evaluation-only. A held-out q that was itself refined with that episode's force in the multimodal
frontend is NOT an independent force-based target (leakage): reported and excluded from certification.
"""
from __future__ import annotations

import math

import numpy as np

from ..estimators.general_constrained import simulate_interaction

TINY_RANGE_REL = 0.05       # range-normalised RMSE is flagged when the q range < 5 x the q noise / this factor


def leakage(it) -> dict:
    m = it.meta or {}
    force_used = bool(m.get("frontend_force_used"))
    contact_used = bool(m.get("frontend_contact_used"))
    return dict(force_used_to_refine_q=force_used, tcp_contact_used_to_refine_q=contact_used,
                independent_force_target=not force_used,
                status="CONTAMINATED" if force_used else ("SHARED_TCP_GEOMETRY" if contact_used else "INDEPENDENT"))


def _events(t, q, q_std):
    v = np.gradient(q, t)
    thr = 4 * float(np.median(q_std)) / max(float(np.median(np.diff(t))) * 3, 1e-12)
    mov = np.abs(v) > thr
    onsets = t[1:][(~mov[:-1]) & mov[1:]]
    stops = t[1:][mov[:-1] & (~mov[1:])]
    return onsets, stops


def predict_heldout(spec, theta_phys, it, *, max_step=1e-3) -> dict:
    if it.initial_state is None:
        return dict(status="NOT_RUN", reason=f"{it.name}: no DECLARED initial state; a held-out forecast may not "
                                             "estimate its start from the test q")
    r = simulate_interaction(spec, theta_phys, it, max_step=max_step)
    e = r.q - it.q_obs
    rng = float(np.ptp(it.q_obs))
    sig = float(np.median(it.q_std))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    o_on, o_st = _events(it.t_obs, it.q_obs, it.q_std)
    p_on, p_st = _events(it.t_obs, r.q, it.q_std)

    def timing(a, b):
        if len(a) == 0 or len(b) == 0:
            return None
        return float(np.median([np.min(np.abs(b - x)) for x in a]))
    rest_obs, rest_pred = float(np.median(it.q_obs[-5:])), float(np.median(r.q[-5:]))
    return dict(status="OK", interaction=it.name, n=len(e), rmse=rmse, rmse_over_sigma=rmse / sig,
                range=rng, nrmse_range=rmse / rng if rng > 0 else math.inf,
                tiny_range_flag=bool(rng < 10 * sig), max_abs_error=float(np.abs(e).max()),
                onset_timing_error_s=timing(o_on, p_on), stop_timing_error_s=timing(o_st, p_st),
                final_rest_error=abs(rest_obs - rest_pred), initial_state=it.initial_state,
                q_pred=r.q.tolist(), t_s=it.t_obs.tolist(), leakage=leakage(it))


def heldout_report(spec, theta_phys, heldout, *, rmse_sigma_max=3.0) -> dict:
    """Preregistered acceptance: every INDEPENDENT held-out episode predicted with RMSE <= rmse_sigma_max x the q
    noise; contaminated episodes are reported but never count."""
    per = [predict_heldout(spec, theta_phys, it) for it in heldout]
    ok = [p for p in per if p["status"] == "OK"]
    indep = [p for p in ok if p["leakage"]["independent_force_target"]]
    if not indep:
        st = "NOT_RUN"
        reason = "no independent held-out episode with a declared initial state"
    else:
        bad = [p["interaction"] for p in indep if p["rmse_over_sigma"] > rmse_sigma_max]
        st = "PASS" if not bad else "FAIL"
        reason = None if not bad else f"held-out RMSE above {rmse_sigma_max} sigma: {bad}"
    vals = [p["rmse_over_sigma"] for p in indep]
    return dict(status=st, reason=reason, per_episode=per, n_independent=len(indep),
                n_contaminated=len(ok) - len(indep),
                summary=dict(mean_rmse_over_sigma=float(np.mean(vals)) if vals else None,
                             p95_rmse_over_sigma=float(np.percentile(vals, 95)) if vals else None,
                             max_rmse_over_sigma=float(np.max(vals)) if vals else None),
                threshold=dict(rmse_sigma_max=rmse_sigma_max))
