"""Bridge to the EXISTING G10 PhysX refinement (estimator side, the parttwin_sim worker) -- no estimation in PArtTwin1D.

* ``engine_capabilities``: what the installed runtime exposes (SAPIEN 3.0.3: legacy dimensionless joint friction,
  armature, drives -- NO native static / dynamic / viscous axis friction -> friction coverage is the labelled
  explicit-effort surrogate, never native identification).
* ``initialize_physx_refinement``: response features -> a G10 warm start x0 (M = 1/g_u when the gain is supported,
  else the prior reference scale with c = M lam, Td = M gam). x0 starts the search; it is NOT a loss term and does
  not narrow the G10 bounds (lambda_R = 0: the features come from the same q, u samples).
* ``candidate_rollouts`` / ``physx_sensitivity``: actual PhysX rollouts of the conditional candidate family and
  finite-difference sensitivities d q_PX / d log(theta).
* ``validate_response_consistency``: analytical-vs-PhysX trajectory parity and feature errors, engine readbacks.

Effective joint inertia is mapped to the link density (rho_scale); armature is held at 0: link inertia and armature
enter the ideal one-DOF equation only through their sum and are never claimed separately identified.
"""
from __future__ import annotations

import math

import numpy as np

G10_NAMES = ("rho_scale", "armature", "dynamic_effort", "static_excess", "viscous")


def engine_capabilities(sim) -> dict:
    try:
        cap = sim.capabilities()
    except Exception as e:                                       # reported, not assumed
        return dict(status="UNAVAILABLE", reason=f"{type(e).__name__}: {e}")
    cap["status"] = "OK"
    cap["coverage_label"] = ("NATIVE_PHYSX_AXIS_FRICTION" if cap.get("native_axis_friction_api") else
                             "DIAGNOSTIC_EXPLICIT_EFFORT_SURROGATE (not native PhysX friction identification)")
    return cap


def _gain(io_result):
    feats = io_result.get("features") or []
    g = next((f for f in feats if f.get("name") == "gain"), None)
    return g


def g10_parameters(M, lam, gam, axis_inertia_unit, armature=0.0):
    """Physical G10 parameters of the normalized candidate at absolute scale M (rho_scale relative to the URDF)."""
    from ..physx_refinement.objective import PhysicalParameters
    rho = (M - armature) / axis_inertia_unit
    if not rho > 0:
        raise ValueError(f"M = {M} below the armature {armature}: no positive link density")
    Td = M * (gam or 0.0)
    return PhysicalParameters(rho, armature, Td, Td, M * (lam or 0.0), inertia_scale_source="IO_RESPONSE_MAPPING")


def initialize_physx_refinement(io_result: dict, existing_g10_config: dict, existing_g10_episodes,
                                require_same_windows: bool = True) -> dict:
    """``existing_g10_config``: {fitting: {bounds: ...}, axis_inertia: I_h(rho_s = 1), prior_rho_range?};
    ``existing_g10_episodes``: the G10 Episodes. The io windows must be the SAME episodes on the same clock, unless
    ``require_same_windows`` is False (the behavioral full-demo replay: the io windows are segmented sub-windows of the
    same demonstration and x0 is ONLY a search start, scored like any other candidate)."""
    from .contracts import SCHEMA
    if io_result.get("schema") != SCHEMA:
        raise ValueError(f"io_result schema {io_result.get('schema')!r} != {SCHEMA}")
    io_w = {k: v["t"] for k, v in (io_result.get("episodes") or {}).items()}
    mism = []
    for e in (existing_g10_episodes if require_same_windows else []):
        w = io_w.get(e.name)
        if w is None:
            mism.append(f"{e.name}: not in the io identification")
        elif abs(w[0] - float(e.t_s[0])) > 1e-6 or abs(w[1] - float(e.t_s[-1])) > 1e-6:
            mism.append(f"{e.name}: window {w} != G10 [{e.t_s[0]}, {e.t_s[-1]}]")
    if mism:
        return dict(status="REFUSED_WINDOW_MISMATCH", reasons=mism, x0=None)
    norm = ((io_result.get("handoff") or {}).get("estimate_not_for_sim") or {}).get("normalized") or {}
    lam, gam = norm.get("lam"), norm.get("gam")
    if lam is None:
        return dict(status="REFUSED_NO_FIT", reasons=["io identification has no fitted rates"], x0=None)
    b = existing_g10_config["fitting"]["bounds"]
    I1 = float(existing_g10_config["axis_inertia"])
    arm = float(b["armature"][0])
    g = _gain(io_result) or {}
    if g.get("status") in ("IDENTIFIED", "CONDITIONAL") and g.get("estimate"):
        M = 1.0 / float(g["estimate"])
        prov = f"M = 1/g_u ({g['status']}) from the input-output output-error fit"
    else:
        rr = existing_g10_config.get("prior_rho_range")
        rho_ref = math.sqrt(rr[0] * rr[1]) if rr else 1.0
        M = rho_ref * I1 + arm
        prov = (f"gain {g.get('status', 'absent')}: prior reference scale (rho_s = {rho_ref:.4g}, PRIOR_ONLY) with "
                "c = M lam, Td = M gam -- one member of the ratio family")
    p = g10_parameters(M, lam, gam, I1, arm)
    x0 = dict(rho_scale=p.rho_scale, armature=arm, dynamic_effort=p.dynamic_effort, static_excess=0.0,
              viscous=p.viscous)
    clipped = {}
    for k in G10_NAMES:
        lo, hi = map(float, b[k])
        v = min(max(x0[k], lo), hi)
        if v != x0[k]:
            clipped[k] = dict(requested=x0[k], used=v, bounds=[lo, hi])
        x0[k] = v
    return dict(status="OK", x0=x0, x0_provenance=prov, clipped_to_g10_bounds=clipped,
                M_not_for_sim=M, mapping="rho_s = (M - armature) / I_h(1), Td = M gam, c = M lam",
                armature_note="armature held at the G10 value; I_link vs armature are not separable",
                response_regularizer_weight=0.0,
                rule="warm start only: no extra loss term, G10 bounds unchanged")


def candidate_rollouts(sim, g10_episodes, candidates: dict) -> dict:
    """{label: PhysicalParameters} -> {label: {episode: SimulationResult}} (actual PhysX)."""
    out = {}
    for lab, p in candidates.items():
        out[lab] = {}
        for e in g10_episodes:
            try:
                out[lab][e.name] = sim(p, e)
            except Exception as ex:                              # every failure recorded
                out[lab][e.name] = ex
    return out


def physx_sensitivity(sim, g10_episodes, p, *, h=0.02) -> dict:
    """Forward differences of q_PX in eta = log(theta) for rho_scale, viscous, dynamic_effort (positives only),
    whitened by sigma_q; singular values / correlations. Static_excess and armature are fixed in G10."""
    from dataclasses import replace
    names = [n for n in ("rho_scale", "viscous", "dynamic_effort") if getattr(p, n) > 0]
    base = {e.name: np.asarray(sim(p, e).q) for e in g10_episodes}
    cols = []
    for n in names:
        kw = {n: getattr(p, n) * math.exp(h)}
        if n == "dynamic_effort":
            kw["static_effort"] = kw[n]
        pp = replace(p, **kw)
        col = []
        for e in g10_episodes:
            col.append((np.asarray(sim(pp, e).q) - base[e.name]) / h / e.sigma_q)
        cols.append(np.concatenate(col))
    if not cols:
        return dict(status="NOT_RUN", reason="no positive parameter")
    J = np.column_stack(cols)
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    C = np.linalg.pinv(J.T @ J)
    d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    return dict(status="OK", parameters=names, coordinates="log", singular_values=S.tolist(),
                condition_number=float(S[0] / S[-1]) if S[-1] > 0 else math.inf,
                weakest_direction=dict(zip(names, Vt[-1].tolist())), correlation=(C / np.outer(d, d)).tolist(),
                note="free-only data: the (rho, c, Td) log-direction (1, 1, 1) is a null direction (scale gauge)")


def validate_response_consistency(io_result: dict, physx_rollouts: dict, config: dict) -> dict:
    """``physx_rollouts``: {candidate: {episode: dict(t, q_physx, q_analytic, q_obs, sigma, params, I_eff,
    diagnostics)}}. Analytical-vs-PhysX trajectory parity (same physical parameters, same initial state, same ZOH
    input) and the PhysX-equivalent normalized features against the io features."""
    feats = {f["name"]: f for f in (io_result.get("features") or [])}
    tol = float((config or {}).get("parity_tol_sigma", 1.0))
    out = dict(schema="forceartgs.g9.io_physx_reconciliation/v1", candidates={}, parity_rule=(
        f"engine parity gap rmse(q_PX - q_analytic)/sigma_q <= {tol} (PhysX semi-implicit Euler + float32 + "
        "Coulomb-surrogate regularisation vs the exact hybrid ODE)"))
    for lab, eps in physx_rollouts.items():
        rows = {}
        for name, r in eps.items():
            if "error" in r:
                rows[name] = dict(status="ROLLOUT_FAILED", error=r["error"])
                continue
            qp, qa, qo, s = (np.asarray(r[k], float) for k in ("q_physx", "q_analytic", "q_obs", "sigma"))
            gap = qp - qa
            rows[name] = dict(status="OK", parity_rmse=float(np.sqrt(np.mean(gap ** 2))),
                              parity_max=float(np.max(np.abs(gap))),
                              parity_over_sigma=float(np.sqrt(np.mean((gap / s) ** 2))),
                              parity_ok=bool(np.sqrt(np.mean((gap / s) ** 2)) <= tol),
                              physx_vs_observed_rmse=float(np.sqrt(np.mean((qp - qo) ** 2))),
                              analytic_vs_observed_rmse=float(np.sqrt(np.mean((qa - qo) ** 2))),
                              physx_vs_observed_over_sigma=float(np.sqrt(np.mean(((qp - qo) / s) ** 2))),
                              initial_state=r.get("initial_state"), q0=r.get("q0"), v0=r.get("v0"),
                              readback=r.get("diagnostics"))
        p = next((r["params"] for r in eps.values() if "params" in r), None)
        I = next((r["I_eff"] for r in eps.values() if "I_eff" in r), None)
        fe = {}
        if p and I:
            px = dict(lambda_=p["viscous"] / I, gamma=p["dynamic_effort"] / I, gain=1.0 / I)
            for k, v in px.items():
                f = feats.get(k.rstrip("_"))
                if f is None:
                    continue
                ci = (f.get("uncertainty") or {}).get("ci95")
                fe[k.rstrip("_")] = dict(physx_equivalent=v, io_estimate=f.get("estimate"), io_status=f.get("status"),
                                         io_ci95=ci, within_io_ci95=(None if not ci or ci[0] is None else
                                                                     bool(ci[0] <= v <= ci[1])))
        out["candidates"][lab] = dict(params=p, I_eff=I, episodes=rows, feature_errors=fe)
    oks = [r["parity_ok"] for c in out["candidates"].values() for r in c["episodes"].values() if r["status"] == "OK"]
    out["parity"] = "PASS" if oks and all(oks) else ("FAIL" if oks else "NOT_RUN")
    return out
