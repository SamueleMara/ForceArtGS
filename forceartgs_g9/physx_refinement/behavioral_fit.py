"""G10-A behavioral calibration: fit simulator parameters to the WHOLE demonstration with actual PhysX rollouts.

Objective (Eq. 3 of the spec), one continuous rollout from (q0 = q_authority(t0), v0 = 0), recorded effort applied:

    L_q(theta) = mean_i huber_delta( (q_PX(t_i; theta) - q_obs(t_i)) / sqrt(sigma_q,i^2 + sigma_num^2) )

theta = (rho_scale, viscous c, dynamic_effort Td); static_excess = 0 (the SAPIEN surrogate has no stiction), armature
fixed. Only simulator parameters are searched -- never q0, v0, a per-window reset or a time-varying force. The scorer
is the existing ``RefinementLoss`` over the existing ``IsolatedSapienCpu`` (real PhysX); nothing here integrates an
ODE in place of PhysX. A near-optimal FAMILY (profile with the other parameters re-optimized, an ENGINEERING tolerance
on rmse/sigma -- not a confidence interval) is kept because one demonstration rarely pins every parameter.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import replace

import numpy as np

from .behavioral_contracts import BehavioralContractError, FullDemoBehavioralEpisode
from .objective import Episode, LossSettings, PhysicalParameters, RefinementLoss
from .optimizer import NAMES, FitConfiguration, fit_physx, from_vector

PROFILE_NAMES = ("rho_scale", "viscous", "dynamic_effort")


class CachingRollout:
    """Memoizes PhysX rollouts by (parameters, episode name, input bytes, q0, v0): the scorer and the reporting use
    the SAME engine result instead of re-simulating. Counts every engine call."""

    def __init__(self, sim, maxsize=8192):
        self.sim, self.maxsize, self._c = sim, maxsize, OrderedDict()
        self.n_engine_calls = 0
        self.n_batches = 0

    def _key(self, p, ep):
        h = hashlib.sha1(np.ascontiguousarray(ep.input_effort).tobytes()).hexdigest()
        return (p.rho_scale, p.armature, p.static_effort, p.dynamic_effort, p.viscous, ep.name, h, ep.q0, ep.v0,
                float(ep.t_s[0]), len(ep.t_s))

    @property
    def batched(self):
        return hasattr(self.sim, "batch")

    def prefetch(self, thetas, ep):
        """Simulate every not-yet-cached candidate of ``ep`` in ONE engine batch (GPU); failures are cached as their
        exception and re-raised on lookup. No-op for an engine without ``batch``."""
        if not self.batched:
            return
        miss, seen = [], set()
        for p in thetas:
            k = self._key(p, ep)
            if k not in self._c and k not in seen:
                seen.add(k)
                miss.append(p)
        if not miss:
            return
        res = self.sim.batch(miss, ep)
        self.n_batches += 1
        self.n_engine_calls += len(miss)
        for p, r in zip(miss, res):
            self._c[self._key(p, ep)] = r
        while len(self._c) > self.maxsize:
            self._c.popitem(last=False)

    def __call__(self, p, ep):
        k = self._key(p, ep)
        if k in self._c:
            self._c.move_to_end(k)
            r = self._c[k]
            if isinstance(r, Exception):
                raise r
            return r
        r = self.sim(p, ep)
        self.n_engine_calls += 1
        self._c[k] = r
        if len(self._c) > self.maxsize:
            self._c.popitem(last=False)
        return r

    def __getattr__(self, n):                                    # effective_inertia, axis, dt, ... of the engine
        return getattr(self.sim, n)


def loss_settings(cfg) -> LossSettings:
    lo = cfg["loss"]
    if lo["optimize_initial_velocity"] or lo["optimize_initial_position"]:
        raise BehavioralContractError("full-demo mode never fits q0 / v0")
    return LossSettings(huber_delta=float(lo["huber_delta"]), numeric_floor_q=float(lo["numeric_floor_q"]),
                        increment_weight=float(lo["increment_weight"]), fit_initial_velocity=False)


def fit_configuration(bounds, f, *, maxiter=None, popsize=None, polish=None, vectorized=False) -> FitConfiguration:
    return FitConfiguration(bounds={k: tuple(map(float, v)) for k, v in bounds.items()}, seed=int(f["seed"]),
                            maxiter=int(f["maxiter"] if maxiter is None else maxiter),
                            popsize=int(f["popsize"] if popsize is None else popsize),
                            polish=bool(f["polish"] if polish is None else polish), tol=float(f["tol"]),
                            v0_nuisance_bound_sd=0.0, vectorized=bool(vectorized))


def _batched(rollout):
    return bool(getattr(rollout, "batched", False))


def assert_full_demo_fit(objective: RefinementLoss, config: FitConfiguration):
    """The contract, checked on the objects actually passed to the optimizer."""
    if objective.settings.fit_initial_velocity or config.v0_nuisance_bound_sd != 0.0:
        raise BehavioralContractError("full-demo mode: v0 nuisance / initial-velocity fitting must be off")
    for e in objective.episodes:
        if not (e.v0 == 0.0 and math.copysign(1.0, e.v0) > 0) or e.sigma_v0 is not None:
            raise BehavioralContractError(f"{e.name}: full-demo episode must start at v0 = +0.0 with no sigma_v0")
        if e.regime_semantics.split(":")[0] != "behavioral_full_demo":
            raise BehavioralContractError(f"{e.name}: not a behavioral full-demo episode")
    if config.bounds["static_excess"][1] > 0:
        raise BehavioralContractError("SAPIEN surrogate: static_excess fixed to 0 (no stiction identification)")
    if config.bounds["armature"][0] != config.bounds["armature"][1]:
        raise BehavioralContractError("armature is held at its declared value (never fitted to fake a link mass)")


def bounds_for(model, I_ref, Q_max) -> dict:
    """Existing G10 physical bounds (same rule as real_data._bounds for forced data): rho in model.rho_bounds,
    Td <= max(gam_max I_hi, k |Q|max), c <= lam_max I_hi, static_excess = 0, armature fixed."""
    rlo, rhi = map(float, model["rho_bounds"])
    I_hi = I_ref * rhi
    Td = max(float(model["gam_max"]) * I_hi, float(model["effort_bound_factor"]) * Q_max)
    c = float(model["lam_max"]) * I_hi
    if model.get("friction_mode") == "frictionless_physx":        # no passive term at all: c = Td = 0, not searched
        Td = c = 0.0
    if model.get("coulomb_friction", "fitted") == "off":          # Coulomb friction switched off: Td = 0, not searched
        Td = 0.0
    return dict(rho_scale=(rlo, rhi), armature=(float(model["armature"]),) * 2, dynamic_effort=(0.0, float(Td)),
                static_excess=(0.0, 0.0), viscous=(0.0, c))


def params_from(d) -> PhysicalParameters:
    if isinstance(d, PhysicalParameters):
        return d
    x = [float(d[k]) for k in NAMES]
    return from_vector(x)


def vec(p: PhysicalParameters) -> dict:
    return dict(rho_scale=p.rho_scale, armature=p.armature, dynamic_effort=p.dynamic_effort,
                static_excess=p.static_effort - p.dynamic_effort, viscous=p.viscous)


def clip_to(p: PhysicalParameters, bounds) -> PhysicalParameters:
    v = vec(p)
    return params_from({k: min(max(v[k], bounds[k][0]), bounds[k][1]) for k in NAMES})


# ------------------------------------------------------------------------------------------------- metrics
def replay_metrics(ep: Episode, res, numeric_floor) -> dict:
    q = np.asarray(res.q, float)
    err = q - ep.q
    s_tot = np.hypot(ep.sigma_q, numeric_floor)
    rng = float(np.ptp(ep.q)) or 1e-12
    i_obs, i_sim = int(np.argmax(np.abs(ep.q - ep.q[0]))), int(np.argmax(np.abs(q - q[0])))
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), max_abs_error=float(np.max(np.abs(err))),
                rmse_over_sigma=float(np.sqrt(np.mean((err / s_tot) ** 2))),
                rmse_over_sigma_q_measurement=float(np.sqrt(np.mean((err / ep.sigma_q) ** 2))),
                rmse_over_range=float(np.sqrt(np.mean(err ** 2))) / rng, q_range=rng,
                final_error=float(err[-1]), peak_displacement_time_error_s=float(ep.t_s[i_sim] - ep.t_s[i_obs]),
                peak_displacement_error=float((q[i_sim] - q[0]) - (ep.q[i_obs] - ep.q[0])),
                n_samples=int(len(q)),
                note="the same q(t) defines every metric: they are diagnostics of ONE fit, not independent evidence")


def score(rollout, objective: RefinementLoss, p: PhysicalParameters, numeric_floor) -> dict:
    ev = objective.evaluate(p)
    ep = objective.episodes[0]
    res = rollout(p, ep)
    return dict(params=p.__dict__, objective=ev["objective"], position_loss=ev["train_position"],
                metrics=replay_metrics(ep, res, numeric_floor), runtime_semantics=res.runtime_semantics,
                engine_version=res.engine_version)


def verify_engine_start(res, ep: Episode) -> dict:
    """Engine readback of the initial state: q(t0) == float32(q0), qdot(t0) == 0.0 bit for bit."""
    q0e, v0e = float(np.asarray(res.q)[0]), float(np.asarray(res.qdot)[0]) if res.qdot is not None else None
    ok_q = q0e == float(np.float32(ep.q0))
    ok_v = v0e is not None and v0e == 0.0
    if not (ok_q and ok_v):
        raise BehavioralContractError(f"engine initial state ({q0e!r}, {v0e!r}) != (float32 {ep.q0!r}, 0.0)")
    return dict(q0_requested=ep.q0, q0_engine=q0e, v0_requested=ep.v0, v0_engine=v0e, exact=True,
                note="read back from the PhysX articulation before the first step (float32 joint state)")


FREE = ("rho_scale", "viscous", "dynamic_effort")


LM_DAMPING = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _huber_sum(r, delta):
    a = np.abs(r)
    return float(np.sum(np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))))


def lm_lockstep(rollout, pe: Episode, bounds, starts, cfg, *, fixed=None, max_nfev=None) -> list:
    """Batched robust Levenberg-Marquardt run on MANY starts in LOCKSTEP (PhysX GPU): per iteration ONE batch holds
    every active start's forward-difference Jacobian columns, and ONE batch every start's damping ladder of trial
    steps (the best one that lowers that start's cost is accepted). The GPU's cost is per call, not per candidate
    (measured break-even ~40 candidates / call), so starts, profile points and trials share each call.
    Same Huber loss (IRLS weights), residuals and sigma floor as the CPU TRF; bounds by projection; q0 / v0 never
    touched. ``fixed``: parameter names held at each start's own value (profiles). Returns one dict per start."""
    f = cfg["fitting"]
    delta = float(cfg["loss"]["huber_delta"])
    sig = np.hypot(pe.sigma_q, float(cfg["loss"]["numeric_floor_q"]))
    free = [n for n in FREE if n not in (fixed or ()) and bounds[n][1] > bounds[n][0]]
    bases = [vec(clip_to(s0, bounds)) for s0 in starts]
    if not free:
        return [dict(status="NOTHING_FREE", params=params_from(b), nfev=0, n_failed=0) for b in bases]
    lo = np.array([bounds[n][0] for n in free])
    hi = np.array([bounds[n][1] for n in free])
    ds = float(f["local_diff_step"])
    budget = int(max_nfev or f["local_max_nfev"])
    S = len(starts)
    nf, nfail = np.zeros(S, int), np.zeros(S, int)

    def P(k, x):
        d = dict(bases[k])
        d.update({n: float(v) for n, v in zip(free, x)})
        return params_from(d)

    def resid_many(items):                         # items: [(start k, x)]
        ps = [P(k, x) for k, x in items]
        rollout.prefetch(ps, pe)
        out = []
        for (k, _), pp in zip(items, ps):
            nf[k] += 1
            try:
                q = np.asarray(rollout(pp, pe).q, float)
                if q.shape != pe.q.shape or not np.isfinite(q).all():
                    raise ValueError("invalid rollout")
                out.append((q - pe.q) / sig)
            except (ValueError, FloatingPointError, RuntimeError):
                nfail[k] += 1
                out.append(None)
        return out
    X = [np.clip(np.array([b[n] for n in free]), lo, hi) for b in bases]
    R0 = resid_many(list(enumerate(X)))
    C0 = [None if r is None else _huber_sum(r, delta) for r in R0]
    active = [r is not None for r in R0]
    msg = ["start rollout failed" if r is None else "budget used" for r in R0]
    iters = [[] for _ in range(S)]
    while any(active):
        ks = [k for k in range(S) if active[k] and nf[k] + len(free) + len(LM_DAMPING) <= budget]
        for k in range(S):
            if active[k] and k not in ks:
                active[k] = False
        if not ks:
            break
        items, meta = [], []
        for k in ks:
            h = ds * np.maximum(1.0, np.abs(X[k]))
            for i in range(len(free)):
                xi = X[k].copy()
                xi[i] = X[k][i] + h[i] if X[k][i] + h[i] <= hi[i] else X[k][i] - h[i]
                items.append((k, xi))
                meta.append((k, i))
        cols = resid_many(items)
        trial_items, tmeta = [], []
        for k in ks:
            cj = [(cols[m], items[m][1]) for m, (kk, _) in enumerate(meta) if kk == k]
            if any(cc is None for cc, _ in cj):
                active[k], msg[k] = False, "Jacobian rollout failed"
                continue
            J = np.column_stack([(cc - R0[k]) / (xi[i] - X[k][i]) for i, (cc, xi) in enumerate(cj)])
            w = np.sqrt(np.minimum(1.0, delta / np.maximum(np.abs(R0[k]), 1e-300)))
            Jw, rw = J * w[:, None], R0[k] * w
            A, g = Jw.T @ Jw, Jw.T @ rw
            dA = np.maximum(np.diag(A), 1e-300)
            for lam in LM_DAMPING:
                try:
                    dx = -np.linalg.solve(A + lam * np.diag(dA), g)
                except np.linalg.LinAlgError:
                    continue
                trial_items.append((k, np.clip(X[k] + dx, lo, hi)))
                tmeta.append(k)
        rs = resid_many(trial_items)
        for k in ks:
            if not active[k]:
                continue
            cs = [(_huber_sum(rr, delta), it[1], rr) for rr, it, kk in zip(rs, trial_items, tmeta)
                  if kk == k and rr is not None]
            if not cs:
                active[k], msg[k] = False, "every trial rollout failed"
                continue
            cb, xb, rb = min(cs, key=lambda z: z[0])
            iters[k].append(dict(cost=cb, accepted=bool(cb < C0[k])))
            if cb >= C0[k] * (1 - 1e-9):
                active[k], msg[k] = False, "no damping level lowers the cost (converged)"
                continue
            rel = (C0[k] - cb) / max(C0[k], 1e-300)
            X[k], R0[k], C0[k] = xb, rb, cb
            if rel < 1e-7:
                active[k], msg[k] = False, "relative improvement < 1e-7"
    out = []
    for k in range(S):
        if R0[k] is None or (nf[k] and nfail[k] > 0.5 * nf[k]):
            out.append(dict(status="FAILED", reason=f"{nfail[k]}/{nf[k]} rollouts failed ({msg[k]})",
                            params=starts[k], nfev=int(nf[k]), n_failed=int(nfail[k])))
            continue
        out.append(dict(status="OK", params=P(k, X[k]), nfev=int(nf[k]), n_failed=int(nfail[k]), message=msg[k],
                        restarts=iters[k], free=free, diff_step=ds,
                        method="batched robust Levenberg-Marquardt in lockstep (Huber IRLS; every start's Jacobian and "
                               f"damping ladder of {len(LM_DAMPING)} trial steps in ONE engine batch each); "
                               f"finite-difference step {ds} relative"))
    return out


def _local_refine_batched(rollout, pe, bounds, start, cfg, *, fixed=None, max_nfev=None) -> dict:
    return lm_lockstep(rollout, pe, bounds, [start], cfg, fixed=fixed, max_nfev=max_nfev)[0]


def local_refine(rollout, pe: Episode, bounds, start: PhysicalParameters, cfg, *, fixed=None, max_nfev=None) -> dict:
    """Bounded robust TRF (SciPy least_squares, loss='huber', f_scale = the LossSettings Huber delta) on the SAME
    PhysX residuals (q_PX - q_obs) / sqrt(sigma_q^2 + sigma_num^2): cost = sum huber, i.e. n x the position term of
    RefinementLoss. Only rho_scale / viscous / dynamic_effort move (``fixed``: names held at the start value); armature
    and static_excess stay fixed; q0 / v0 are never touched. Failed rollouts are counted (> 50 % -> no result).
    A batched engine (PhysX GPU) uses the batched Levenberg-Marquardt instead (same loss, one batch per step)."""
    if _batched(rollout):
        return _local_refine_batched(rollout, pe, bounds, start, cfg, fixed=fixed, max_nfev=max_nfev)
    from scipy.optimize import least_squares
    f = cfg["fitting"]
    delta = float(cfg["loss"]["huber_delta"])
    sig = np.hypot(pe.sigma_q, float(cfg["loss"]["numeric_floor_q"]))
    free = [n for n in FREE if n not in (fixed or ()) and bounds[n][1] > bounds[n][0]]
    v0 = vec(clip_to(start, bounds))
    if not free:
        return dict(status="NOTHING_FREE", params=params_from(v0), nfev=0, n_failed=0)
    lo = np.array([bounds[n][0] for n in free])
    hi = np.array([bounds[n][1] for n in free])
    stats = dict(n=0, failed=0, last=None)

    def P(x):
        d = dict(v0)
        d.update({n: float(v) for n, v in zip(free, x)})
        return params_from(d)

    def R(x):
        stats["n"] += 1
        try:
            q = np.asarray(rollout(P(x), pe).q, float)
            if q.shape != pe.q.shape or not np.isfinite(q).all():
                raise ValueError("invalid rollout")
            return (q - pe.q) / sig
        except (ValueError, FloatingPointError, RuntimeError) as e:
            stats["failed"] += 1
            stats["last"] = f"{type(e).__name__}: {e}"
            return np.full(len(pe.q), 1e6)
    # start strictly inside the box: TRF's bound scaling (Coleman-Li) freezes a variable that starts ON a bound
    x = np.array([v0[n] for n in free])
    budget = int(max_nfev or f["local_max_nfev"])
    ds = float(f["local_diff_step"])
    jac = "2-point"
    restarts, cost_prev, msg, used = [], None, "", 0       # budget in SciPy nfev units (excludes FD Jacobian calls)
    # TRF on this nonsmooth surface (friction reversals, stops) can terminate on 'xtol' after its trust region
    # collapsed on a slope; restarting from its own end point resets the region (measured: 4.05 sigma -> 0.58 sigma on
    # a reversing slide). Restart until the cost stops improving or the budget / restart cap is used.
    for k in range(1 + int(f["local_restarts"])):
        left = budget - used
        if left < 4:
            break
        x0 = np.clip(x, lo + 1e-5 * (hi - lo), hi - 1e-5 * (hi - lo))    # strictly inside (Coleman-Li scaling)
        r = least_squares(R, x0, bounds=(lo, hi), loss="huber", f_scale=delta, method="trf", x_scale="jac",
                          jac=jac, diff_step=ds, max_nfev=left)
        used += int(r.nfev)
        restarts.append(dict(cost=float(r.cost), nfev=int(r.nfev), message=str(r.message)))
        msg = str(r.message)
        improved = cost_prev is None or r.cost < cost_prev * (1 - 1e-6)
        if cost_prev is None or r.cost < cost_prev:
            x, cost_prev = r.x, float(r.cost)
        if not improved:
            break
    if stats["n"] and stats["failed"] > 0.5 * stats["n"]:
        return dict(status="FAILED", reason=f"{stats['failed']}/{stats['n']} rollouts failed ({stats['last']})",
                    params=start, nfev=stats["n"], n_failed=stats["failed"])
    return dict(status="OK", params=P(x), nfev=stats["n"], n_failed=stats["failed"], message=msg, restarts=restarts,
                free=free, diff_step=float(f["local_diff_step"]),
                method="bounded robust TRF (huber) on the PhysX position residuals; finite-difference step "
                       f"{f['local_diff_step']} relative (above the float32 engine-state resolution)")


# ------------------------------------------------------------------------------------------------- fit
def fit_full_demo(rollout, ep: FullDemoBehavioralEpisode, cfg, bounds, starts: dict, *, log=None) -> dict:
    """``starts``: {label: PhysicalParameters} deterministic baselines / warm starts (prior reference, io warm start,
    admissible corners). Each is SCORED (baseline table); the best starts the bounded differential evolution."""
    f, floor = cfg["fitting"], float(cfg["loss"]["numeric_floor_q"])
    pe = ep.to_physx_episode()
    obj = RefinementLoss(rollout, [pe], loss_settings(cfg))
    fc = fit_configuration(bounds, f, vectorized=_batched(rollout))
    assert_full_demo_fit(obj, fc)
    if _batched(rollout):
        rollout.prefetch([clip_to(params_from(p), bounds) for p in starts.values()], pe)
    base = {}
    for lab, p in starts.items():
        try:
            base[lab] = score(rollout, obj, clip_to(params_from(p), bounds), floor)
            base[lab]["label"] = lab
        except Exception as e:                                   # recorded, never silently dropped
            base[lab] = dict(label=lab, status="ROLLOUT_FAILED", error=f"{type(e).__name__}: {e}")
    ok = {k: v for k, v in base.items() if "objective" in v}
    if not ok:
        raise RuntimeError(f"every baseline rollout failed: {base}")
    best_start = min(ok, key=lambda k: ok[k]["objective"])
    if log:
        log(f"[g10-behavioral] baselines: " + ", ".join(f"{k} L={v['objective']:.4g}" for k, v in ok.items()) +
            f"; DE from '{best_start}'")
    rep = fit_physx(obj, fc, x0=vec(PhysicalParameters(**ok[best_start]["params"])))
    p = PhysicalParameters(**rep["params"])
    best = score(rollout, obj, p, floor)
    local = []
    if f["local_refine"]:
        seeds = [("global_de", p)] + [(k, PhysicalParameters(**ok[k]["params"])) for k in
                                      sorted(ok, key=lambda k: ok[k]["objective"])[:max(int(f["local_starts"]) - 1, 0)]]
        lrs = (lm_lockstep(rollout, pe, bounds, [s0 for _, s0 in seeds], cfg) if _batched(rollout) else
               [None] * len(seeds))
        for (lab, s0), lr_b in zip(seeds, lrs):
            lr = lr_b if lr_b is not None else local_refine(rollout, pe, bounds, s0, cfg)
            row = dict(start=lab, status=lr["status"], nfev=lr["nfev"], n_failed=lr["n_failed"],
                       message=lr.get("message") or lr.get("reason"))
            if lr["status"] == "OK":
                sc = score(rollout, obj, lr["params"], floor)
                row.update(objective=sc["objective"], params=sc["params"],
                           rmse_over_sigma=sc["metrics"]["rmse_over_sigma"])
                if sc["objective"] < best["objective"]:
                    p, best = lr["params"], sc
            local.append(row)
        if log:
            log("[g10-behavioral] local TRF: " + ", ".join(f"{r_['start']} L={r_.get('objective', float('nan')):.4g}"
                                                         for r_ in local) + f" -> best L={best['objective']:.4g}")
    res = rollout(p, pe)
    start = verify_engine_start(res, pe)
    ref = ok.get("prior_reference") or ok[best_start]
    bv = vec(p)
    def _tol(n, b):                                      # the local search starts 1e-5 of the range inside the box
        return max(1e-4 * (bounds[n][1] - bounds[n][0]), 2e-3 * abs(b))
    at_bound = {n: ("lower" if bv[n] <= bounds[n][0] + _tol(n, bounds[n][0]) else
                    "upper" if bv[n] >= bounds[n][1] - _tol(n, bounds[n][1]) else None)
                for n in FREE if bounds[n][1] > bounds[n][0]}
    return dict(at_bound=at_bound, objective_definition="mean Huber(delta) of (q_PX - q_obs) / sqrt(sigma_q^2 + sigma_num^2) over ALL "
                                     "valid authority samples, one continuous rollout from (q0, 0)",
                loss_settings=obj.settings.__dict__, bounds={k: list(v) for k, v in bounds.items()},
                baselines=base, warm_start_used=best_start, optimizer=rep["optimizer"], local_refinement=local,
                fit_status=rep["status"], best=best, engine_initial_state=start,
                improvement=dict(reference=ref["label"], objective_ratio=best["objective"] / max(ref["objective"], 1e-300),
                                 rmse_ratio=best["metrics"]["rmse"] / max(ref["metrics"]["rmse"], 1e-300),
                                 reference_rmse=ref["metrics"]["rmse"], best_rmse=best["metrics"]["rmse"]),
                fitted_parameters=[n for n in FREE if bounds[n][1] > bounds[n][0]],
                fixed=dict(armature=bounds["armature"][0], static_excess=0.0, q0=pe.q0, v0=0.0),
                _params=p, _objective=obj, _result=res)


def _grid(name, best, lo, hi, n):
    if name == "rho_scale":
        return np.unique(np.geomspace(max(lo, best / 4), min(hi, best * 4), n))
    if best > 0:
        g = np.geomspace(max(best / 4, hi * 1e-4), min(best * 4, hi), max(n - 1, 1))
        return np.unique(np.r_[0.0, g])
    return np.unique(np.linspace(0.0, hi * 0.25, n))


def _grid_starts(rollout, obj, bounds, bv, name, xs, n=7):
    """``_grid_start`` for every profile value at once: ONE engine batch for all values x all grid points."""
    per = [_grid_cands(bounds, bv, name, float(x), n) for x in xs]
    flat = [c for cs in per for c in cs]
    vals = obj.objective_many(flat)
    out, k = [], 0
    for x, cs in zip(xs, per):
        v = vals[k:k + len(cs)]
        k += len(cs)
        ok = [(vv, c) for vv, c in zip(v, cs) if vv is not None]
        out.append(min(ok, key=lambda t: t[0])[1] if ok else params_from(dict(bv, **{name: float(x)})))
    return out


def _grid_cands(bounds, bv, name, x, n=7):
    others = [k for k in FREE if k != name and bounds[k][1] > bounds[k][0]]
    axes = []
    for k in others:
        lo, hi = bounds[k]
        b = bv[k]
        g = np.geomspace(max(lo, b / 4), min(hi, b * 4), n) if (k == "rho_scale" or b > 0) else \
            np.linspace(lo, lo + 0.05 * (hi - lo), n)
        axes.append(np.unique(np.r_[g, b]))
    cands = []
    for combo in (np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T if axes else [[]]):
        d = dict(bv, **{name: x})
        d.update({k: float(v) for k, v in zip(others, combo)})
        cands.append(params_from(d))
    return cands


def _grid_start(rollout, obj, bounds, bv, name, x, n=7):
    """Batched engine only: best start for a profile point from an n x n log grid of the OTHER free parameters around
    the best point (one GPU batch), so the local search starts in the right valley."""
    others = [k for k in FREE if k != name and bounds[k][1] > bounds[k][0]]
    axes = []
    for k in others:
        lo, hi = bounds[k]
        b = bv[k]
        g = np.geomspace(max(lo, b / 4), min(hi, b * 4), n) if (k == "rho_scale" or b > 0) else \
            np.linspace(lo, lo + 0.05 * (hi - lo), n)
        axes.append(np.unique(np.r_[g, b]))
    cands = []
    for combo in np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T if axes else [[]]:
        d = dict(bv, **{name: x})
        d.update({k: float(v) for k, v in zip(others, combo)})
        cands.append(params_from(d))
    vals = obj.objective_many(cands)
    ok = [(v, c) for v, c in zip(vals, cands) if v is not None]
    return min(ok, key=lambda t: t[0])[1] if ok else params_from(dict(bv, **{name: x}))


def profile_family(rollout, fit: dict, cfg, bounds, *, log=None) -> dict:
    """Profiles of rho_scale / viscous / dynamic_effort with the OTHER free parameters re-optimized (local TRF from the
    best point) on the SAME full-demo loss; family = every evaluated point with rmse/sigma <= best + tolerance."""
    f, floor = cfg["fitting"], float(cfg["loss"]["numeric_floor_q"])
    obj, best_p = fit["_objective"], fit["_params"]
    best_ros = fit["best"]["metrics"]["rmse_over_sigma"]
    tol = float(f["family_tolerance_sigma"])
    profiles, members = {}, [dict(fit["best"], label="best")]
    bv = vec(best_p)
    for name in f["profile_parameters"]:
        lo, hi = bounds[name]
        if hi <= lo:
            profiles[name] = dict(status="FIXED", value=lo)
            continue
        pts = []
        grid = _grid(name, bv[name], lo, hi, int(f["profile_points"]))
        if _batched(rollout):
            starts = _grid_starts(rollout, obj, bounds, bv, name, grid)
            lrs = lm_lockstep(rollout, obj.episodes[0], bounds, starts, cfg, fixed=(name,),
                              max_nfev=int(f["profile_inner_max_nfev"]))
            for x, r in zip(grid, lrs):
                if r["status"] == "FAILED":
                    pts.append(dict(label=f"profile_{name}={x:.4g}", profiled=name, value=float(x), status="FAILED",
                                    error=r["reason"]))
                    continue
                s_ = score(rollout, obj, r["params"], floor)
                s_.update(label=f"profile_{name}={x:.4g}", profiled=name, value=float(x))
                pts.append(s_)
            grid = []
        for x in grid:
            try:
                start = params_from(dict(bv, **{name: float(x)}))
                if _batched(rollout):
                    start = _grid_start(rollout, obj, bounds, bv, name, float(x))
                r = local_refine(rollout, obj.episodes[0], bounds, start, cfg, fixed=(name,),
                                 max_nfev=int(f["profile_inner_max_nfev"]))
                if r["status"] == "FAILED":
                    raise RuntimeError(r["reason"])
                p = r["params"]
                s = score(rollout, obj, p, floor)
                s.update(label=f"profile_{name}={x:.4g}", profiled=name, value=float(x))
                pts.append(s)
            except Exception as e:                               # recorded
                pts.append(dict(label=f"profile_{name}={x:.4g}", profiled=name, value=float(x),
                                status="FAILED", error=f"{type(e).__name__}: {e}"))
        good = [p for p in pts if "metrics" in p]
        inside = [p for p in good if p["metrics"]["rmse_over_sigma"] <= best_ros + tol]
        vals = [p["value"] for p in inside] + [bv[name]]
        lo_v, hi_v = min(vals), max(vals)
        edge = bool(inside) and (min(p["value"] for p in good) in vals or max(p["value"] for p in good) in vals)
        if name == "rho_scale":
            wide = hi_v / max(lo_v, 1e-300) > 1.5
        else:
            wide = (hi_v - lo_v) > 0.5 * max(bv[name], 1e-12) or (lo_v == 0.0 and bv[name] > 0)
        profiles[name] = dict(points=pts, family_range=[lo_v, hi_v], best=bv[name], wide=bool(wide),
                              reaches_grid_edge=edge, n_inside=len(inside))
        members += inside
        if log:
            log(f"[g10-behavioral] profile {name}: family range [{lo_v:.4g}, {hi_v:.4g}] (best {bv[name]:.4g})")
    bound_limited = sorted(n for n, b in (fit.get("at_bound") or {}).items()
                           if b and not (n == "dynamic_effort" and b == "lower") and not (n == "viscous" and b == "lower"))
    not_unique = bound_limited or any(v.get("wide") or v.get("reaches_grid_edge") for v in profiles.values()
                                      if "points" in v)
    return dict(bound_limited=bound_limited, rule=f"family = evaluated candidates with rmse/sigma <= best ({best_ros:.4g}) + {tol} -- an "
                     "ENGINEERING tolerance on the same full-demo loss, NOT a frequentist confidence interval",
                tolerance_sigma=tol, profiles=profiles, members=members, not_unique=bool(not_unique),
                n_members=len(members))


def input_sensitivity(rollout, ep: FullDemoBehavioralEpisode, hyps, p, cfg) -> dict:
    """The SAME fitted theta replayed under each grounded input hypothesis (not independent demonstrations)."""
    floor, tol = float(cfg["loss"]["numeric_floor_q"]), float(cfg["fitting"]["family_tolerance_sigma"])
    rows = {}
    for h in hyps:
        pe = ep.to_physx_episode(name=f"hyp_{h.name}", effort=h.effort_object)
        try:
            rows[h.name] = dict(kind=h.kind, basis=h.basis, **replay_metrics(pe, rollout(p, pe), floor))
        except Exception as e:
            rows[h.name] = dict(kind=h.kind, basis=h.basis, status="ROLLOUT_FAILED", error=f"{type(e).__name__}: {e}")
    ref = rows.get("recorded", {}).get("rmse_over_sigma")
    d = [abs(v["rmse_over_sigma"] - ref) for k, v in rows.items() if k != "recorded" and "rmse_over_sigma" in v]
    spread = max(d) if d and ref is not None else None
    return dict(hypotheses=rows, max_delta_rmse_over_sigma=spread,
                status=("NOT_EVALUATED" if spread is None else
                        "INPUT_CONDITIONED_FIT" if spread > tol else "INPUT_ROBUST_AT_THETA"),
                rule=f"INPUT_CONDITIONED_FIT when a grounded hypothesis moves rmse/sigma by > {tol} at the fitted theta",
                note="hypotheses are alternative readings of ONE recording (clock / noise / contact), never a sign flip")


# ------------------------------------------------------------------------------------------------- audits
def work_audit(ep: FullDemoBehavioralEpisode, *, potential=None, k=3.0) -> dict:
    """Model-free passivity check from rest: the energy supplied to the object, W_in - Delta U, cannot be negative
    (KE(t0) = 0, KE_end >= 0, dissipation >= 0). W_in = sum_k Qbar_k Delta q_k on the q clock (Qbar = ZOH mean of the
    recorded rows over the interval). ``potential``: U(q) of the candidate (None: gravity not evaluated). A deliberate
    sign reversal of the input makes W_in < 0 and fails here -- it is never 'fixed' by a negative inertia."""
    t, q, s = ep.t_q_s, ep.q_observed, ep.q_sigma
    ft, Q = ep.effort_t_s, ep.effort_object
    Qs = ep.effort_sigma if ep.effort_sigma is not None else np.zeros_like(Q)
    Qbar, Qbar_sd = np.empty(len(t) - 1), np.empty(len(t) - 1)
    for i in range(len(t) - 1):
        a, b = t[i], t[i + 1]
        edges = np.r_[a, ft[(ft > a) & (ft < b)], b]
        idx = np.clip(np.searchsorted(ft, edges[:-1] + 1e-12, side="right") - 1, 0, len(Q) - 1)
        w = np.diff(edges) / (b - a)
        Qbar[i] = float(w @ Q[idx])
        Qbar_sd[i] = float(np.sqrt((w ** 2) @ (Qs[idx] ** 2)))
    dq = np.diff(q)
    sd_dq = np.hypot(s[1:], s[:-1])
    W = float(Qbar @ dq)
    W_sd = float(np.sqrt(np.sum((dq * Qbar_sd) ** 2) + np.sum((Qbar * sd_dq) ** 2)))
    out = dict(W_in=W, W_in_sd=W_sd, units="J", sign_convention="canonical object-side effort, unchanged")
    if potential is not None:
        dU = float(potential(q[-1]) - potential(q[0]))
        h = 1e-6
        dUdq = (potential(q[-1] + h) - potential(q[-1] - h)) / (2 * h)
        dU_sd = float(abs(dUdq) * math.hypot(s[-1], s[0]))
        out.update(delta_U=dU, delta_U_sd=dU_sd)
    else:
        dU, dU_sd = 0.0, 0.0
        out.update(delta_U=None, gravity="NOT_EVALUATED (no candidate potential)")
    B, B_sd = W - dU, math.hypot(W_sd, dU_sd)
    out.update(energy_supplied=B, energy_supplied_sd=B_sd,
               verdict=("INPUT_OUTPUT_SIGN_INCONSISTENT" if B < -k * B_sd else "PASSIVE_ENERGY_CONSISTENT"),
               rule=f"from rest: W_in - Delta U >= -{k} sd (passive object, KE >= 0, dissipation >= 0)")
    return out


def gravity_potential(info, rho, g):
    """U(q) = -m(rho) g . c(q) of the URDF candidate (frame B; c(0) = com_q0)."""
    from ..dynamics.general_1dof import rotation
    m = float(info["mass"]) * float(rho)
    c0, a, o = (np.asarray(info[k], float) for k in ("com_q0", "axis", "origin"))
    g = np.asarray(g, float)
    if info["joint_type"] == "revolute":
        return lambda q: float(-m * g @ (o + rotation(a, float(q)) @ (c0 - o)))
    return lambda q: float(-m * g @ (c0 + float(q) * a))


def probe_episode(ep: FullDemoBehavioralEpisode, cfg) -> Episode:
    """A NEW synthetic force profile (not a real demonstration): push at amplitude_factor x median |Q| of the
    confirmed-contact rows, in their mean direction, for push_fraction of the probe, then zero; from (q0, 0)."""
    pp = cfg["policy_probe"]
    lab = np.asarray(ep.contact_evidence.get("labels_per_row") or [], str)
    m = lab == "CONTACT_CONFIRMED"
    if m.any():
        base = ep.effort_object[m]
    else:                    # no labels (contact always on): the rows carrying the top decile of |Q|
        e = ep.effort_object
        base = e[np.abs(e) >= np.quantile(np.abs(e), 0.9)]
    A = float(pp["amplitude_factor"]) * float(np.median(np.abs(base)))
    sgn = float(np.sign(np.mean(base))) or 1.0
    T = float(pp["duration_s"])
    dtq = float(np.median(np.diff(ep.t_q_s)))
    t = ep.t0_s + np.arange(0.0, T + 1e-9, dtq)
    rdt = float(np.median(np.diff(ep.effort_t_s))) if len(ep.effort_t_s) > 1 else 0.02
    ft = ep.t0_s + np.arange(0.0, T + 1e-9, rdt)
    Q = np.where(ft - ep.t0_s < float(pp["push_fraction"]) * T, sgn * A, 0.0)
    return Episode(name="policy_probe", t_s=t, q=np.full_like(t, ep.q0_observed), sigma_q=np.ones_like(t),
                   q0=float(ep.q0_observed), v0=0.0, input_t_s=ft, input_effort=Q, mode="forced",
                   calibration_status="SYNTHETIC_PROBE", exploratory_assumptions_accepted=True,
                   regime_semantics="behavioral_full_demo: SYNTHETIC probe input (NOT a real demonstration)")


def policy_spread(rollout, ep: FullDemoBehavioralEpisode, members, cfg) -> dict:
    """T15: near-equal demo fits replayed under a NEW force profile -> prediction spread (motivates correlated domain
    randomization; no real demonstration is invented)."""
    if not cfg["policy_probe"]["enabled"]:
        return dict(status="DISABLED")
    pe = probe_episode(ep, cfg)
    if _batched(rollout):
        rollout.prefetch([PhysicalParameters(**m["params"]) for m in members], pe)
    seen, rows = set(), []
    for m in members:
        p = PhysicalParameters(**m["params"])
        key = (p.rho_scale, p.viscous, p.dynamic_effort)
        if key in seen:
            continue
        seen.add(key)
        try:
            rows.append(dict(label=m.get("label"), params=m["params"], demo_rmse_over_sigma=m["metrics"]["rmse_over_sigma"],
                             q=np.asarray(rollout(p, pe).q, float)))
        except Exception as e:
            rows.append(dict(label=m.get("label"), params=m["params"], error=f"{type(e).__name__}: {e}"))
    ok = [r for r in rows if "q" in r]
    if len(ok) < 2:
        return dict(status="NOT_EVALUATED", reason="fewer than 2 distinct family members", members=len(ok))
    Qm = np.array([r["q"] for r in ok])
    spread = Qm.max(0) - Qm.min(0)
    return dict(status="EVALUATED", probe=dict(t=[float(pe.t_s[0]), float(pe.t_s[-1])],
                                               effort_level=float(np.max(np.abs(pe.input_effort))),
                                               push_s=float(cfg["policy_probe"]["push_fraction"]) *
                                               float(cfg["policy_probe"]["duration_s"]),
                                               semantics="SYNTHETIC_PROBE_NOT_A_REAL_DEMONSTRATION"),
                n_members=len(ok), final_q_spread=float(spread[-1]), max_q_spread=float(spread.max()),
                final_q=[float(r["q"][-1]) for r in ok], labels=[r["label"] for r in ok],
                t=pe.t_s.tolist(), q_members=Qm.tolist(),
                note="candidates that are near-equal on the demonstration can diverge under a new input: randomize "
                     "them JOINTLY (correlated family), not independently")


def zero_gravity_alternative(rollout0, ep: FullDemoBehavioralEpisode, cfg, bounds, x0, *, log=None) -> dict:
    """Separately LABELLED hypothesis 'no gravity load' (vertical hinge / horizontal slide): its own fit on the same
    full-demo loss with a zero-gravity PhysX scene. Never the selected candidate."""
    a, floor = cfg["alternatives"], float(cfg["loss"]["numeric_floor_q"])
    pe = ep.to_physx_episode(name="full_demo_zero_gravity")
    obj = RefinementLoss(rollout0, [pe], loss_settings(cfg))
    fc = fit_configuration(bounds, cfg["fitting"], maxiter=a["maxiter"], popsize=a["popsize"], polish=False,
                           vectorized=_batched(rollout0))
    assert_full_demo_fit(obj, fc)
    try:
        rep = fit_physx(obj, fc, x0=vec(x0))
    except RuntimeError as e:
        return dict(status="FIT_FAILED", reason=str(e))
    p = PhysicalParameters(**rep["params"])
    s = score(rollout0, obj, p, floor)
    lr = None
    if cfg["fitting"]["local_refine"]:
        lr = local_refine(rollout0, pe, bounds, p, cfg, max_nfev=int(a["local_max_nfev"]))
        if lr["status"] == "OK":
            s2 = score(rollout0, obj, lr["params"], floor)
            if s2["objective"] < s["objective"]:
                p, s = lr["params"], s2
    if log:
        log(f"[g10-behavioral] zero-gravity alternative: rmse/sigma {s['metrics']['rmse_over_sigma']:.4g}")
    return dict(status="FITTED", hypothesis="zero_gravity_load (vertical hinge / horizontal slide)",
                label="SEPARATELY_LABELLED_ALTERNATIVE_NOT_SELECTED", best=s, optimizer=rep["optimizer"],
                local_refinement=None if lr is None else {k: v for k, v in lr.items() if k != "params"})
