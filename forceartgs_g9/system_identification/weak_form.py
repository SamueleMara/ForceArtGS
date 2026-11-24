"""Stage A, complementary estimator: the integral / weak-form normalized regression (Eq. 12), REUSING the existing
G9 implementation ``forceartgs_g9.estimators.free_weak_form`` on the SAME source segments (no second smoother, no
second row builder):

    free  : dv + lam dq + gam s dt + int a_g(q) dt = 0
    forced: dv + lam dq + gam s dt + int a_g(q) dt = g_u int u dt      (exact ZOH impulse on native rows)

Velocities at the window ends come from the uncertainty-aware smoothing spline (no raw second differences); windows
never cross a stick / reversal / force-transition boundary (each source segment is monotonic, one regime).
The weak form covers the WHOLE source segment (including the prefix the output-error stage uses for its initial
state); the window difference is recorded, never hidden.
"""
from __future__ import annotations

import math

from .contracts import rename_value_fields


def nspec_for(joint_type, gravity, variant):
    from ..free_forced.normalized import NormalizedDynamicsSpec
    return NormalizedDynamicsSpec(joint_type, gravity, viscous=True, coulomb=variant in ("L1", "L3"))


def fit(episodes, gravity, variant="L1", *, B=0) -> dict:
    from ..estimators import free_weak_form as FW
    tr = [e for e in episodes if e.train]
    if not tr:
        return dict(status="NOT_RUN", reason="no training episode")
    if any(e.segment is None for e in tr):
        return dict(status="NOT_RUN", reason="episodes carry no source segment (weak form needs the full window)")
    jt = tr[0].joint_type
    ns = nspec_for(jt, gravity, variant)
    free = [e.segment for e in tr if not e.forced]
    forced = [e.segment for e in tr if e.forced]
    out = dict(method="weak_form", variant=variant, implementation="forceartgs_g9.estimators.free_weak_form",
               windows={e.name: [float(e.segment.t[0]), float(e.segment.t[-1])] for e in tr},
               window_note="full source segment (the output-error window starts at the prefix end)")
    ff = None
    if free:
        ff = FW.fit(free, ns, B=B, sensitivity=False)
        out["free_fit"] = _slim(ff)
    if forced:
        fs = FW.forced_scale(free, forced, ns, free_fit=ff, B=B)
        out["forced_fit"] = _slim(fs)
        est = fs.get("estimate") or {}
        sd = fs.get("sd") or {}
        out.update(status=fs.get("status"), reason=fs.get("reason"),
                   estimate=dict(lam=est.get("lam"), gam=est.get("gam"), g_u=est.get("beta")),
                   sd=dict(lam=sd.get("lam"), gam=sd.get("gam"), g_u=sd.get("beta")),
                   g_u_ci95=(fs.get("profile_beta") or {}).get("ci95"),
                   n_contradicting_windows=fs.get("n_contradicting_windows"))
    elif ff is not None:
        est = ff.get("estimate") or {}
        sd = ff.get("sd") or {}
        out.update(status=ff.get("status"), reason="; ".join(ff.get("reasons") or []),
                   estimate=dict(lam=est.get("lam"), gam=est.get("gam"), g_u=None),
                   sd=dict(lam=sd.get("lam"), gam=sd.get("gam"), g_u=None),
                   per_parameter=ff.get("per_parameter"), combination=ff.get("combination"))
    return out


def _slim(d):
    """Drop the per-row dumps; keep estimates, sd, profiles' intervals and status (legacy 'value' keys renamed)."""
    keep = {}
    for k, v in d.items():
        if k in ("rows", "segments", "bootstrap"):
            continue
        if k == "profile" and isinstance(v, dict):
            v = {n: dict(ci95=p.get("ci95"), ci_at_grid_edge=p.get("ci_at_grid_edge")) for n, p in v.items()}
        if k == "profile_beta" and isinstance(v, dict):
            v = dict(ci95=v.get("ci95"), ci_at_grid_edge=v.get("ci_at_grid_edge"))
        if k == "window_audit":
            v = [{kk: vv for kk, vv in a.items()} for a in v]
        keep[k] = v
    return rename_value_fields(keep)


def init_from(wf) -> dict:
    """Warm start for the output-error fit (estimates only; their intervals are not propagated as priors)."""
    est = (wf or {}).get("estimate") or {}
    out = {}
    for k in ("lam", "gam", "g_u"):
        v = est.get(k)
        if v is not None and math.isfinite(v) and v > 0:
            out[k] = float(v)
    return out
