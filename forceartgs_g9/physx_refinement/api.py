"""ForceArtGS-only estimator API; run *after* upstream G9 kinematic/force preparation.

An opt-in stage entrypoint which never mutates the G9 fitting pipeline or 3DGS
reconstruction and never exports an unvalidated physical asset.
"""
from __future__ import annotations
from pathlib import Path
import json
import hashlib
import numpy as np
from .g9_adapter import episodes_from_g9_interactions, write_g9_physx_episodes
from .objective import LossSettings, PhysicalParameters, RefinementLoss
from .optimizer import FitConfiguration, fit_physx
from .sapien_diagnostic import IsolatedSapienCpu
from .mapping import physx_axis_properties
from . import identifiability as ID


class PhysXIsaacRollout:
    """Native Isaac Sim / PhysX articulation rollout -- NOT IMPLEMENTED (no Isaac runtime on this machine).

    Required before it may exist: an isolated USD articulation, native per-axis staticFrictionEffort /
    dynamicFrictionEffort / viscousFrictionCoefficient / armature set AND read back, active drives and legacy friction
    zeroed, the preserved effort ZOH applied, and solver-level acceptance tests; only then
    runtime_semantics=NATIVE_PHYSX_VERIFIED."""

    def __init__(self, *a, **k):
        raise NotImplementedError("native Isaac PhysX adapter is not implemented; cannot declare native parity")


def _validate(physics, fitting):
    if physics.get('engine') != 'sapien_cpu_diagnostic':
        raise ValueError('native Isaac PhysX adapter is not implemented; cannot declare native parity')
    if physics['friction_mode'] == 'frictionless_physx' and any(
        fitting['bounds'][k][1] > 0 for k in ('dynamic_effort', 'static_excess', 'viscous')
    ):
        raise ValueError('frictionless runner cannot fit nonzero friction')
    if physics['friction_mode'] == 'external_effort_surrogate' and fitting['bounds']['static_excess'][1] > 0:
        raise ValueError('surrogate does not model stiction; fix static_excess to zero')


def estimate(sim, episodes, *, physics, loss, fitting, profile=None, prior_rho_range=None, log=None, x0=None,
             x0_provenance=None) -> dict:
    """Fit + held-out (frozen theta) + adequacy + scale profile + status table + correlated family.
    ``x0``: optional warm start {name: value} (e.g. from the analytical input-output identification); it starts the
    search, it is NOT a prior term and does not narrow the bounds."""
    _validate(physics, fitting)
    obj = RefinementLoss(sim, episodes, LossSettings(**loss))
    cfg = FitConfiguration(**dict(fitting, bounds={k: tuple(map(float, v)) for k, v in fitting["bounds"].items()}))
    if log:
        log(f"[g10] fitting {sum(e.used_for_training for e in episodes)} training episode(s) ...")
    report = fit_physx(obj, cfg, x0=x0)
    report['initialization'] = dict(x0=x0, provenance=x0_provenance or ("none (differential evolution only)"
                                                                         if x0 is None else "caller-supplied"))
    p = PhysicalParameters(**report['params'])
    fixed = [k for k, (lo, hi) in cfg.bounds.items() if lo == hi]
    adeq = ID.adequacy(report['loss'], episodes)
    prof = None
    if report["scale_status"] != "ABSOLUTE_SCALE_UNIDENTIFIED_FREE_ONLY" and profile and profile.get("n", 0) > 0:
        lo, hi = cfg.bounds["rho_scale"]
        if hi > lo:
            grid = np.geomspace(lo, hi, int(profile["n"]))
            prof = ID.profile_scale(obj, cfg, grid, inner_maxiter=profile.get("inner_maxiter", 15),
                                    inner_popsize=profile.get("inner_popsize", 6),
                                    inner_polish=profile.get("inner_polish", True), best=report,
                                    n_free_params=len(report["optimizer"]["free_parameters"]), log=log)
    I_eff = sim.effective_inertia(p) if hasattr(sim, "effective_inertia") else None
    sem = {v.get("runtime_semantics") for v in report["loss"]["episodes"].values()}
    sem = sem.pop() if len(sem) == 1 else "MIXED"
    report['model_adequacy'] = adeq
    report['scale_profile'] = prof if prof is not None else dict(
        status=("NOT_RUN_FREE_ONLY_GAUGE" if report["scale_status"] == "ABSOLUTE_SCALE_UNIDENTIFIED_FREE_ONLY"
                else "NOT_RUN"))
    if I_eff is not None:
        report['normalized'] = ID.normalized(p, I_eff)
        report['status_table'] = ID.status_table(p, I_eff, scale_status=report["scale_status"], profile=prof,
                                                 adeq=adeq, runtime_semantics=sem, fixed=fixed)
        if float(loss.get("scale_prior_weight", 0.0) or 0.0) > 0 and \
                report['status_table']['absolute_scale'] not in ("MODEL_INADEQUATE", "NOT_IDENTIFIED"):
            # an optimizer conditioned on lambda_rho (log rho - log rho_prior)^2 stays PRIOR_ONLY (spec G10)
            report['status_table']['absolute_scale'] = "PRIOR_ONLY"
            for k in ("rho_scale", "I_eff"):
                report['status_table']['parameters'][k]['status'] = "PRIOR_ONLY"
        rr = prior_rho_range or ((cfg.bounds["rho_scale"]) if cfg.bounds["rho_scale"][1] > cfg.bounds["rho_scale"][0]
                                 else (p.rho_scale, p.rho_scale))
        report['correlated_family'] = ID.correlated_family(
            p, lambda r: sim.effective_inertia(PhysicalParameters(r, p.armature, p.static_effort, p.dynamic_effort,
                                                                  p.viscous)),
            rr, profile=prof)
    report['native_properties_suggestions_only'] = physx_axis_properties(physics['joint_type'], p)
    report['simulation_export_allowed'] = False
    report['status'] = (report.get('status_table') or {}).get('overall', 'DIAGNOSTIC_ONLY_NOT_IDENTIFIED')
    report['downstream'] = dict(zip(("point_estimate_allowed", "reason"), ID.downstream_point_allowed(report)))
    report['heldout'] = dict(status=("EVALUATED_WITH_FROZEN_THETA" if any(not e.used_for_training for e in episodes)
                                     else "NO_HELDOUT_EPISODE"),
                             heldout_position=report["loss"]["heldout_position"])
    report['v0_nuisance'] = (report["optimizer"].get("v0_nuisance") or
                             "NOT_OPTIMIZED (v0 fixed per episode; the evaluation API accepts explicit offsets only)")
    return report


def run_forceartgs_physx_estimator(interactions, regime_by_name: dict, v0_by_name: dict,
                                    *, calibration_by_name: dict, urdf,
                                    physics: dict, loss: dict, fitting: dict, out_dir,
                                    python=None, profile=None, prior_rho_range=None, log=None):
    """Run a new, estimator-side diagnostic using existing prepared G9Interaction objects.

    `regime_by_name`: {name: {mode:free|forced, evidence:{source:..., ...}, train:bool}}
    `v0_by_name`: independently estimated/declared initial velocities (same q convention)
    `calibration_by_name`: {name: {status:VERIFIED|..., exploratory_opt_in:bool}}
    `python`: SAPIEN interpreter (None: this one, the forceartgs env)
    `profile`: {n, inner_maxiter, inner_popsize} -- rho_s profile (forced data only)

    This module does not derive force from sensor data: the existing canonical
    G9Interaction.effort is the only source, with its existing force sign.
    """
    episodes = episodes_from_g9_interactions(interactions, regime_by_name, v0_by_name,
                                            calibration_by_name=calibration_by_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_g9_physx_episodes(episodes,
                            {k: v['evidence'] for k, v in regime_by_name.items()},
                            out_dir / 'physx_episodes.json')
    _validate(physics, fitting)
    sim=IsolatedSapienCpu(urdf, moving_link=physics['moving_link'], joint_name=physics['joint_name'],
                          dt=physics['timestep'],gravity=physics.get('gravity',[0.,0.,-9.81]),
                          friction_mode=physics['friction_mode'], python=python)
    try:
        report = estimate(sim, episodes, physics=physics, loss=loss, fitting=fitting, profile=profile,
                          prior_rho_range=prior_rho_range, log=log)
    finally:
        sim.close()
    up=Path(urdf)
    report['provenance']={'owner':'ForceArtGS', 'source':'G9Interaction',
        'urdf_sha256':hashlib.sha256(up.read_bytes()).hexdigest(),
        'run_mode':physics['engine'],'friction_semantics':physics['friction_mode'],
        'engine_process': sim.backend_process, 'n_rollouts': sim.n_rollouts,
        'force_source':'existing G9Interaction.effort; no retare or sign change',
        'g9_local_checkout_note':'run against current uncommitted ForceArtGS source'}
    (out_dir/'physx_refinement.json').write_text(json.dumps(report,indent=2,default=str))
    return report
