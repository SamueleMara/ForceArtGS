"""Bounded black-box optimization of the G10 PhysX rollout objective.

Never promotes fitted point estimates to identified physical parameters.
The calling backend and final manifest must disclose native/surrogate semantics.
Fixed parameters (lo == hi) are held out of the differential-evolution population (SciPy sizes the population by the
number of dimensions, fixed or not). Failed rollouts are COUNTED and reported, never silently absorbed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import differential_evolution
from .objective import PhysicalParameters, RefinementLoss

NAMES = ("rho_scale", "armature", "dynamic_effort", "static_excess", "viscous")
PENALTY = 1e15


def from_vector(x, *, source="FITTED_EXPLORATORY") -> PhysicalParameters:
    rho,arm,fd,extra,c = map(float,x[:5])
    return PhysicalParameters(rho,arm,fd+extra,fd,c,inertia_scale_source=source)


@dataclass(frozen=True)
class FitConfiguration:
    bounds: dict[str,tuple[float,float]]
    seed: int = 0
    maxiter: int = 50
    popsize: int = 8
    polish: bool = True
    enforce_free_gauge: bool = True  # require fixed rho_scale and armature for free-only fit
    tol: float = 1e-3
    max_failed_fraction: float = 0.5
    v0_nuisance_bound_sd: float = 0.0   # > 0: per-training-episode v0 offset in +-k sigma_v0 (needs loss opt-in)

    def vector_bounds(self):
        if set(self.bounds)!=set(NAMES):
            raise ValueError(f"required bounds: {list(NAMES)}")
        b=[tuple(map(float,self.bounds[k])) for k in NAMES]
        if any(lo<0 or hi<lo for lo,hi in b) or b[0][0]<=0:
            raise ValueError("positive inertia/mass, nonnegative friction and ordered bounds required")
        return b


def scale_status(episodes):
    fit_eps=[e for e in episodes if e.used_for_training]
    if all(e.mode == "free" for e in fit_eps):
        return "ABSOLUTE_SCALE_UNIDENTIFIED_FREE_ONLY"
    if any(e.calibration_status!="VERIFIED" for e in fit_eps if e.mode=="forced"):
        return "FORCED_EXPLORATORY_NOT_CERTIFIED"
    return "FORCED_INPUT_PRESENT_OBSERVABILITY_NOT_YET_PROVEN"


def fit_physx(objective: RefinementLoss, config: FitConfiguration, *, logger=None, x0=None):
    """``x0``: optional warm start {name: value} (clipped into the bounds; DE still explores the whole box)."""
    bounds = config.vector_bounds()
    fit_eps=[e for e in objective.episodes if e.used_for_training]
    # A pure free-motion run does not give any absolute scale. We keep an
    # optimizer point only as one member of an unidentifiable family.
    free_only = all(e.mode == "free" for e in fit_eps)
    if free_only and config.enforce_free_gauge:
        for key in ("rho_scale", "armature"):
            lo, hi = config.bounds[key]
            if lo != hi:
                raise ValueError(f"free-only fit requires fixed {key} (absolute-scale gauge); "
                                 "fit normalized friction at a declared nonzero mass scale")
    free_idx=[i for i,(lo,hi) in enumerate(bounds) if hi>lo]
    x_fixed=np.array([lo for lo,_ in bounds])
    stats=dict(n_eval=0,n_failed=0,last_error=None)
    # optional initial-velocity nuisance (spec Eq. 18/20): bounded +-k sigma_v0, Gaussian prior in the objective
    v0_eps=[]
    if config.v0_nuisance_bound_sd>0:
        if not objective.settings.fit_initial_velocity:
            raise ValueError("v0 nuisance needs LossSettings.fit_initial_velocity (explicit opt-in)")
        v0_eps=[e for e in fit_eps if e.sigma_v0 is not None]
    v0_bounds=[(-config.v0_nuisance_bound_sd*e.sigma_v0, config.v0_nuisance_bound_sd*e.sigma_v0) for e in v0_eps]
    nf=len(free_idx)

    def full(z):
        x=x_fixed.copy(); x[free_idx]=z[:nf]
        return x

    def offsets(z):
        return {e.name: float(d) for e, d in zip(v0_eps, z[nf:])}

    def f(z):
        stats["n_eval"]+=1
        try:
            if v0_eps:
                return objective.evaluate(from_vector(full(z)), offsets(z))["objective"]
            return objective(from_vector(full(z)))
        except (ValueError,FloatingPointError,RuntimeError) as e:
            stats["n_failed"]+=1; stats["last_error"]=f"{type(e).__name__}: {e}"
            return PENALTY
    if free_idx or v0_eps:
        kw={}
        if x0 is not None:
            kw["x0"]=np.r_[[min(max(float(x0[NAMES[i]]),bounds[i][0]),bounds[i][1]) for i in free_idx],
                           np.zeros(len(v0_eps))]
        result=differential_evolution(f,[bounds[i] for i in free_idx]+v0_bounds,seed=config.seed,
                                      maxiter=config.maxiter,popsize=config.popsize,polish=config.polish,
                                      tol=config.tol,workers=1,**kw)
        x=full(result.x)
        v0_off=offsets(result.x)
        opt=dict(success=bool(result.success),message=str(result.message),nfev=int(result.nfev),nit=int(result.nit))
    else:
        x=x_fixed
        v0_off={}
        opt=dict(success=True,message="all parameters fixed: evaluation only",nfev=0,nit=0)
    if stats["n_eval"] and stats["n_failed"]>config.max_failed_fraction*stats["n_eval"]:
        raise RuntimeError(f"{stats['n_failed']}/{stats['n_eval']} PhysX evaluations failed "
                           f"(last: {stats['last_error']}); no fit reported")
    p=from_vector(x, source=("ASSUMED_FIXED_FREE_SCALE" if free_only else
                             "FITTED_EXPLORATORY_NOT_IDENTIFIED"))
    report=objective.evaluate(p, v0_off or None)
    opt.update(n_failed_evaluations=stats["n_failed"], n_evaluations=stats["n_eval"], last_error=stats["last_error"],
               free_parameters=[NAMES[i] for i in free_idx], v0_offsets=v0_off,
               v0_nuisance=("FITTED +-%g sigma_v0 with Gaussian prior" % config.v0_nuisance_bound_sd if v0_eps
                            else "FIXED (prefix estimate)"))
    return dict(status="FIT_ONLY_NOT_IDENTIFIED", scale_status=scale_status(objective.episodes),
                params=p.__dict__, vector=dict(zip(NAMES, map(float, x))), loss=report, optimizer=opt,
                disclaimer="Engine-conditional point fit. No identifiability or heldout predictive claim.")
