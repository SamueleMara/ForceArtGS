"""G10 identifiability / adequacy / downstream contract on top of the PhysX point fit.

A point fit never establishes identifiability (spec G10). This module adds what the spec requires before any claim:

  * ``profile_scale``: the objective profiled over rho_s (friction re-optimised at each fixed rho_s, same engine),
    converted to a summed robust loss  L = sum_e sum_i rho_delta(z)  (~ chi^2 / 2 in the Gaussian core) and
    inflated by max(1, chi^2_red) because rollout residuals are serially correlated. The 95% set is
    {rho : L(rho) - L_min <= 1.92 s^2}. An interval that reaches a grid end is OPEN -> scale not identified.
  * ``adequacy``: rollout rmse per episode against the shared rule (rmse/range <= 0.05 or rmse/sigma <= 3; the
    same thresholds as forceartgs_g9.free_forced.replay), train and held-out separately.
  * ``status_table``: per parameter, with ``estimate_not_for_sim`` (never a simulator-ready ``value``).
  * ``correlated_family``: if absolute inertia is unknown, a prior-conditioned family of (I_eff, c, Td) with c/I_eff
    and Td/I_eff held at the fitted normalized values (free-only: exact gauge), or the profile family (forced: the
    friction re-optimised at each rho_s).
  * ``downstream_point_allowed``: the exporter-side refusal rule (NOT_IDENTIFIED / MODEL_INADEQUATE point -> refuse
    unless the user explicitly asks for a prior-only preview).
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .objective import PhysicalParameters, RefinementLoss
from .optimizer import NAMES, FitConfiguration, fit_physx

ADEQ_RANGE, ADEQ_SIGMA = 0.05, 3.0
Z95 = 1.92


def summed_loss(report) -> tuple[float, int]:
    eps = [p for p in report["episodes"].values() if p["train"]]
    return float(sum(p["position_loss"] * p["n_samples"] for p in eps)), int(sum(p["n_samples"] for p in eps))


def _crossing(r, d, k, step, thr):
    """Walk from the minimum k in direction step to the first evaluated point above the threshold; the crossing inside
    that bracket is interpolated in x = log rho -- quadratically through (k-1, k, k+1) when the bracket touches the
    minimum (the profile is ~quadratic there; a linear chord would put the crossing too close to the minimum),
    linearly otherwise. Returns (rho, open, bracket)."""
    x = np.log(r)
    j = k
    while 0 <= j + step < len(r):
        if d[j + step] > thr:
            a, b = j, j + step
            xc = None
            if a == k and 0 < k < len(r) - 1:
                A = np.vander(x[k - 1:k + 2], 3)
                try:
                    c2, c1, c0 = np.linalg.solve(A, d[k - 1:k + 2])
                except np.linalg.LinAlgError:
                    c2 = 0.0
                if c2 > 0:
                    xm = -c1 / (2 * c2)
                    disc = c1 * c1 - 4 * c2 * (c0 - thr)
                    if disc > 0:
                        for root in ((-c1 - math.sqrt(disc)) / (2 * c2), (-c1 + math.sqrt(disc)) / (2 * c2)):
                            if min(x[a], x[b]) <= root <= max(x[a], x[b]) and (root - xm) * step >= 0:
                                xc = root
            if xc is None:
                f = (thr - d[a]) / (d[b] - d[a])
                xc = x[a] + f * (x[b] - x[a])
            return float(math.exp(xc)), False, (float(r[a]), float(r[b]))
        j += step
    return float(r[j]), True, None


def profile_scale(objective: RefinementLoss, config: FitConfiguration, grid, *, inner_maxiter=15, inner_popsize=6,
                  inner_polish=True, n_free_params=None, best=None, refine=2, log=None) -> dict:
    """Profile over rho_s (armature / static_excess keep their configured bounds). ``best``: the global fit report
    (its rho_s joins the profile so the minimum is not limited by the grid). Inner fits are warm-started from the
    nearest finished point and polished. ``refine`` rounds evaluate the profile AT the interpolated crossings and
    recompute (the bracket shrinks to the evaluated points)."""
    pts = []
    if best is not None:
        L, N = summed_loss(best["loss"])
        pts.append(dict(rho_scale=float(best["params"]["rho_scale"]), objective=best["loss"]["objective"],
                        summed_loss=L, n=N, params=best["params"], heldout=best["loss"]["heldout_position"],
                        n_failed=best["optimizer"]["n_failed_evaluations"], source="global_fit"))

    def point(rho, source):
        b = dict(config.bounds, rho_scale=(rho, rho))
        cfg = replace(config, bounds=b, maxiter=int(inner_maxiter), popsize=int(inner_popsize),
                      polish=bool(inner_polish), enforce_free_gauge=False)
        done = [p for p in pts if "params" in p]
        x0 = None
        if done:
            nb = min(done, key=lambda p: abs(math.log(p["rho_scale"] / rho)))["params"]
            x0 = dict(rho_scale=rho, armature=nb["armature"], dynamic_effort=nb["dynamic_effort"],
                      static_excess=nb["static_effort"] - nb["dynamic_effort"], viscous=nb["viscous"])
        try:
            fit = fit_physx(objective, cfg, x0=x0)
            L, N = summed_loss(fit["loss"])
            pts.append(dict(rho_scale=rho, objective=fit["loss"]["objective"], summed_loss=L, n=N,
                            params=fit["params"], heldout=fit["loss"]["heldout_position"],
                            n_failed=fit["optimizer"]["n_failed_evaluations"], source=source))
        except (RuntimeError, ValueError) as e:
            pts.append(dict(rho_scale=rho, error=f"{type(e).__name__}: {e}", source=source))
        if log:
            log(f"[g10] profile ({source}) rho_s={rho:.4g}: {pts[-1].get('objective', pts[-1].get('error'))}")

    def analyse():
        pts.sort(key=lambda p: p["rho_scale"])
        ok = [p for p in pts if "summed_loss" in p]
        if len(ok) < 3:
            return None
        L = np.array([p["summed_loss"] for p in ok])
        k = int(np.argmin(L))
        chi2_red = 2.0 * L[k] / max(ok[0]["n"] - (n_free_params if n_free_params is not None else 3), 1)
        thr = Z95 * max(1.0, chi2_red)
        rhos = np.array([p["rho_scale"] for p in ok])
        d = L - L[k]
        return dict(k=k, chi2_red=chi2_red, thr=thr, rhos=rhos, d=d, lo=_crossing(rhos, d, k, -1, thr),
                    hi=_crossing(rhos, d, k, +1, thr))
    centre = pts[0]["rho_scale"] if pts else None
    for rho in sorted(map(float, grid), key=lambda r: abs(math.log(r / centre)) if centre else r):
        point(rho, "profile_grid")
    A = analyse()
    for _ in range(int(refine) if A else 0):
        new = [c[0] for c in (A["lo"], A["hi"]) if not c[1]
               and min(abs(math.log(c[0] / r)) for r in A["rhos"]) > 1e-3]
        if not new:
            break
        for rho in new:
            point(rho, "crossing_refinement")
        A = analyse()
    if A is None:
        return dict(status="PROFILE_FAILED", points=pts, reason="fewer than 3 profile points evaluated")
    (lo, lo_open, lob), (hi, hi_open, hib) = A["lo"], A["hi"]
    status = ("SCALE_PROFILE_OPEN" if (lo_open or hi_open) else "SCALE_PROFILE_BOUNDED_CANDIDATE")
    return dict(status=status, points=pts, rho_best=float(A["rhos"][A["k"]]), chi2_red_at_best=float(A["chi2_red"]),
                inflation_s2=float(max(1.0, A["chi2_red"])), threshold=float(A["thr"]), ci95=[lo, hi],
                open_low=lo_open, open_high=hi_open, brackets=[lob, hib],
                rho_evaluated=[float(x) for x in A["rhos"]], delta_L=[float(x) for x in A["d"]],
                rule=f"95% set = {{rho : L(rho) - L_min <= {Z95} max(1, chi2_red)}}; crossings interpolated in log rho "
                     "(quadratic next to the minimum) and refined by evaluating the profile at them; the global fit is "
                     "a profile point; no crossing before a grid end = open (not identified). Serially correlated "
                     "rollout residuals make even the inflated interval optimistic: a bounded profile is a CANDIDATE, "
                     "never ESTIMATED without an independent episode.")


def adequacy(report, episodes) -> dict:
    by = {e.name: e for e in episodes}
    out = {}
    for n, p in report["episodes"].items():
        e = by[n]
        rng = float(np.ptp(e.q))
        rr = p["rmse_rad_or_m"] / rng if rng > 0 else math.inf
        ok = rr <= ADEQ_RANGE or p["rmse_over_sigma"] <= ADEQ_SIGMA
        out[n] = dict(train=p["train"], mode=p["mode"], rmse=p["rmse_rad_or_m"], range=rng, rmse_over_range=rr,
                      rmse_over_sigma=p["rmse_over_sigma"], adequate=bool(ok))
    tr = [v["adequate"] for v in out.values() if v["train"]]
    ho = [v["adequate"] for v in out.values() if not v["train"]]
    return dict(rule=f"rmse/range <= {ADEQ_RANGE} or rmse/sigma <= {ADEQ_SIGMA}", episodes=out,
                train_adequate=bool(tr) and all(tr), heldout_adequate=(all(ho) if ho else None),
                n_heldout=len(ho))


def normalized(params: PhysicalParameters, I_eff: float) -> dict:
    return dict(lam=params.viscous / I_eff, gam=params.dynamic_effort / I_eff, I_eff=I_eff)


def correlated_family(params: PhysicalParameters, I_of_rho, prior_rho_range, *, profile=None, n=7) -> dict:
    """Prior-conditioned family. ``I_of_rho(rho)`` = engine effective inertia at (rho, fitted armature)."""
    I_fit = I_of_rho(params.rho_scale)
    nz = normalized(params, I_fit)
    lo, hi = prior_rho_range
    members = []
    for rho in np.geomspace(lo, hi, n):
        I = I_of_rho(float(rho))
        members.append(dict(rho_scale=float(rho), I_eff=I, viscous=nz["lam"] * I, dynamic_effort=nz["gam"] * I,
                            static_effort=nz["gam"] * I))
    fam = dict(kind="NORMALIZED_GAUGE_FAMILY", normalized_held=dict(lam=nz["lam"], gam=nz["gam"]),
               prior_rho_range=[float(lo), float(hi)], members=members,
               note="c/I_eff and Td/I_eff held at the fitted values; every member reproduces the same FREE motion "
                    "(exact for zero armature and gravity-free / gravity-proportional loads); NOT a posterior")
    if profile and profile.get("points"):
        fam["profile_family"] = [dict(rho_scale=p["rho_scale"], I_eff=I_of_rho(p["rho_scale"]),
                                      viscous=p["params"]["viscous"], dynamic_effort=p["params"]["dynamic_effort"],
                                      delta_L=dl, source=p.get("source"))
                                 for p, dl in zip([q for q in profile["points"] if "summed_loss" in q],
                                                  profile.get("delta_L", []))]
        fam["profile_family_note"] = "forced data: friction re-optimised at each rho_s (the conditional family)"
    return fam


def status_table(params: PhysicalParameters, I_eff: float, *, scale_status, profile, adeq, runtime_semantics,
                 fixed=()) -> dict:
    inadequate = not adeq["train_adequate"] or adeq["heldout_adequate"] is False
    surrogate = runtime_semantics != "NATIVE_PHYSX_VERIFIED"
    pst = (profile or {}).get("status")
    if inadequate:
        scale = "MODEL_INADEQUATE"
    elif scale_status == "ABSOLUTE_SCALE_UNIDENTIFIED_FREE_ONLY":
        scale = "NOT_IDENTIFIED"
    elif pst == "SCALE_PROFILE_BOUNDED_CANDIDATE":
        scale = "PROFILE_BOUNDED_CANDIDATE_NOT_CERTIFIED"
    else:
        scale = "NOT_IDENTIFIED"
    fric = "MODEL_INADEQUATE" if inadequate else ("CONDITIONAL_ON_SCALE" if scale != "PROFILE_BOUNDED_CANDIDATE_NOT_CERTIFIED"
                                                  else "CANDIDATE_CONDITIONAL_ON_SURROGATE")
    nz = normalized(params, I_eff)
    rows = {
        "rho_scale": dict(estimate_not_for_sim=params.rho_scale,
                          status="FIXED_PRIOR" if "rho_scale" in fixed else scale),
        "I_eff": dict(estimate_not_for_sim=I_eff, status="FIXED_PRIOR" if "rho_scale" in fixed else scale),
        "armature": dict(estimate_not_for_sim=params.armature,
                         status="FIXED_PRIOR" if "armature" in fixed else scale),
        "viscous": dict(estimate_not_for_sim=params.viscous, status=fric),
        "dynamic_effort": dict(estimate_not_for_sim=params.dynamic_effort, status=fric),
        "static_effort": dict(estimate_not_for_sim=params.static_effort,
                              status="NOT_MODELLED_BY_SURROGATE" if surrogate else fric),
        "lam=c/I_eff": dict(estimate_not_for_sim=nz["lam"],
                            status="MODEL_INADEQUATE" if inadequate else "FIT_CONDITIONAL_NOT_IDENTIFIED"),
        "gam=Td/I_eff": dict(estimate_not_for_sim=nz["gam"],
                             status="MODEL_INADEQUATE" if inadequate else "FIT_CONDITIONAL_NOT_IDENTIFIED"),
    }
    overall = ("MODEL_INADEQUATE" if inadequate else
               "DIAGNOSTIC_ONLY_NOT_IDENTIFIED" if surrogate or scale != "PROFILE_BOUNDED_CANDIDATE_NOT_CERTIFIED"
               else "CANDIDATE_NOT_CERTIFIED")
    return dict(schema="forceartgs.g10.physx_status_table/v1", overall=overall, absolute_scale=scale,
                parameters=rows, runtime_semantics=runtime_semantics,
                never=["ESTIMATED from the optimizer alone", "a simulator-ready `value` field"],
                requires_for_any_promotion=["native PhysX joint friction with runtime readback (NATIVE_PHYSX_VERIFIED)",
                                            "bounded scale profile", "frozen-parameter independent real episode",
                                            "joint impulse / momentum audit consistent"])


def downstream_point_allowed(report, *, explicit_prior_preview=False) -> tuple[bool, str]:
    """Exporter-side rule: refuse a point estimate whose status is NOT_IDENTIFIED / MODEL_INADEQUATE (or any
    surrogate-only fit) unless the user explicitly requests a prior-only simulation preview."""
    st = report.get("status_table") or {}
    bad = st.get("overall") in (None, "MODEL_INADEQUATE", "DIAGNOSTIC_ONLY_NOT_IDENTIFIED") or \
        st.get("absolute_scale") in (None, "NOT_IDENTIFIED", "MODEL_INADEQUATE") or \
        not report.get("simulation_export_allowed", False)
    if not bad:
        return True, "status table permits a point estimate"
    if explicit_prior_preview:
        return True, ("PRIOR_ONLY_SIMULATION_PREVIEW explicitly requested: use `correlated_family` members, labelled "
                      "PRIOR_SCALED_PREVIEW, never as identified physics")
    return False, f"refused: overall {st.get('overall')}, absolute scale {st.get('absolute_scale')}"
