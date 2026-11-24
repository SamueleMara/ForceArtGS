"""Stage C bridge to the EXISTING G9 normalized estimators (augmented EKF, PINN): optional initialization from the
output-error response features, run on the SAME source segments with the same normalized model and gravity, and a
method comparison that reports incompatible estimates instead of averaging them.

The EKF keeps its own Joseph covariance / NIS calibration; the PINN keeps its data + weak-form + start-state losses and
its seed spread is labelled SEED_ENSEMBLE_SPREAD (not a posterior). No zero-scale (M, c, Fc) residual is trained: the
free stage is normalized (g_u absent), g_u enters only with trusted forced segments.
"""
from __future__ import annotations

import math


def init_from_io(oe) -> dict:
    est = (oe or {}).get("estimate") or {}
    out = {}
    for k_io, k_g9 in (("lam", "lam"), ("gam", "gam"), ("g_u", "beta")):
        v = est.get(k_io)
        if v is not None and math.isfinite(v) and v > 0:
            out[k_g9] = float(v)
    return out


def run_methods(episodes, gravity, variant, oe, weak, methods, log=None) -> dict:
    from .weak_form import nspec_for
    tr = [e for e in episodes if e.train]
    segs = [e.segment for e in tr if e.segment is not None]
    jt = tr[0].joint_type if tr else None
    out = {}
    if oe is not None:
        out["output_error"] = dict(estimate=dict(oe["estimate"]), sd=dict(lam=(oe.get("sd") or {}).get("lam"),
                                                                            gam=(oe.get("sd") or {}).get("gam"),
                                                                            g_u=oe.get("g_u_sd")),
                                   uncertainty_label=oe.get("uncertainty_label"), windows="G10 episode windows")
    if weak is not None and weak.get("estimate"):
        out["weak_form"] = dict(estimate=weak["estimate"], sd=weak.get("sd"), status=weak.get("status"),
                                uncertainty_label="weak-form covariance (chi2-scaled)", windows="full source segments")
    if not segs:
        return out
    ns = nspec_for(jt, gravity, variant)
    init = init_from_io(oe)
    forced_segs = [s for s in segs if not s.is_free]
    free_segs = [s for s in segs if s.is_free]
    if "normalized_ekf" in methods:
        from ..estimators import normalized_ekf as NE
        try:
            spec = ns.with_forced() if forced_segs else ns
            r = NE.fit(segs, spec, init=init)
            est = r.get("estimate") or {}
            sd = r.get("sd") or {}
            out["normalized_ekf"] = dict(estimate=dict(lam=est.get("lam"), gam=est.get("gam"), g_u=est.get("beta")),
                                         sd=dict(lam=sd.get("lam"), gam=sd.get("gam"), g_u=sd.get("beta")),
                                         status=r.get("status"), nis=r.get("nis"),
                                         posterior_over_prior_sd=r.get("posterior_over_prior_sd"),
                                         uncertainty_label=r.get("uncertainty_label"), init=init,
                                         init_source="output-error response features" if init else "EKF default")
        except Exception as e:                                       # reported, never silently dropped
            out["normalized_ekf"] = dict(status="FAILED", reason=f"{type(e).__name__}: {e}")
    if "normalized_pinn" in methods:
        from ..estimators import normalized_pinn as NP
        try:
            ra = NP.fit_free(free_segs, ns, init=init, log=log) if free_segs else None
            if forced_segs:
                rb = NP.fit_forced(free_segs, forced_segs, ns, ra or dict(estimate=init), beta0=init.get("beta", 1.0),
                                   log=log)
                est = rb.get("estimate") or {}
                lab = rb.get("uncertainty_label")
            else:
                est = (ra or {}).get("estimate") or {}
                lab = (ra or {}).get("uncertainty_label")
            out["normalized_pinn"] = dict(estimate=dict(lam=est.get("lam"), gam=est.get("gam"), g_u=est.get("beta")),
                                          sd=None, uncertainty_label=lab or "SEED_ENSEMBLE_SPREAD (not a posterior)",
                                          init=init)
        except Exception as e:
            out["normalized_pinn"] = dict(status="FAILED", reason=f"{type(e).__name__}: {e}")
    return out


def compare(methods: dict, z_crit=3.0) -> dict:
    """Pairwise |a - b| / sqrt(sa^2 + sb^2) per parameter; > z_crit -> INCOMPATIBLE (reported, never averaged).
    Methods without a calibrated sd are compared descriptively only."""
    names = [m for m, v in methods.items() if v.get("estimate")]
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for p in ("lam", "gam", "g_u"):
                xa, xb = methods[a]["estimate"].get(p), methods[b]["estimate"].get(p)
                if xa is None or xb is None:
                    continue
                sa = (methods[a].get("sd") or {}).get(p)
                sb = (methods[b].get("sd") or {}).get(p)
                row = dict(a=a, b=b, parameter=p, a_est=xa, b_est=xb, rel_diff=abs(xa - xb) / max(abs(xa), abs(xb),
                                                                                                   1e-300))
                if sa is not None and sb is not None and (sa > 0 or sb > 0):
                    z = abs(xa - xb) / math.hypot(sa, sb)
                    row.update(z=z, verdict="INCOMPATIBLE" if z > z_crit else "COMPATIBLE")
                else:
                    row.update(z=None, verdict="NO_CALIBRATED_UNCERTAINTY")
                pairs.append(row)
    return dict(pairs=pairs, incompatible=[p for p in pairs if p["verdict"] == "INCOMPATIBLE"],
                rule=f"|a - b| / sqrt(sa^2 + sb^2) > {z_crit}: INCOMPATIBLE (reported, not averaged)",
                window_note="output error and EKF / PINN / weak form share the source segments; the output-error "
                            "window starts at the prefix end")
