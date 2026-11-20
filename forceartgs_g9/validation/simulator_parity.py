"""Per-engine term mapping and numerical dynamics parity (G9 spec 7, Gate D).

Three different claims, never merged:
  portable kinematics       the asset loads and its link poses follow q (checked by the shared builder);
  nominal dynamics export   each identified term has an engine field (term matrix below);
  verified numerical parity a withheld external generalized effort produces the same q(t) in the engine and in
                            the standalone G9 model restricted to the terms that engine represents (this module).
Static breakaway (Fs > Fc), the nuisance bias and PhysX Coulomb are NOT_REPRESENTED: an engine with an enabled
unrepresented term can never be ENGINE_PARITY_PASS for the full model.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..dynamics.general_1dof import GeneralTerms, sim_params, simulate_general

SCRIPTS = Path(__file__).resolve().parents[1] / "runtime_scripts"
ENGINE_GRAVITY_W = (0.0, 0.0, -9.81)          # PArtTwin1D shared exporter convention (model.GRAVITY_W)
REPRESENTED, APPROXIMATE, NOT_REPRESENTED, EMPIRICAL = "REPRESENTED", "APPROXIMATE", "NOT_REPRESENTED", "EMPIRICAL"


def term_matrix(terms: GeneralTerms, Fc=None, Fs=None, gravity_bound=None, effort_scale=None,
                negligible_rel=0.01) -> dict:
    """Engine x term mapping of the PArtTwin1D physics spec v1 as written by the shared exporter."""
    static_eq = Fs is not None and Fc is not None and abs(Fs - Fc) <= 1e-9 * max(Fs, 1.0)
    m = {}
    for eng in ("mujoco", "sapien_urdf", "isaac_usd"):
        row = {}
        row["inertia"] = (REPRESENTED, "moving_link <inertial>: mass, COM, full tensor (no armature)")
        row["viscous"] = ((REPRESENTED, "joint damping") if eng != "isaac_usd" else
                          (APPROXIMATE, "DriveAPI damping, stiffness 0, target velocity 0 (unit-converted)")) \
            if terms.viscous else ("DISABLED", "damping written as 0")
        if terms.coulomb:
            row["coulomb"] = {"mujoco": (APPROXIMATE, "frictionloss: soft-constraint dry friction (one level)"),
                              "sapien_urdf": (EMPIRICAL, "URDF <dynamics friction> -> PhysX articulation joint "
                                                         "friction; semantics measured by the rollout, not assumed"),
                              "isaac_usd": (NOT_REPRESENTED, "PhysX jointFriction is dimensionless, not N*m")}[eng]
        else:
            row["coulomb"] = ("DISABLED", "friction written as 0")
        if terms.static:
            row["static"] = ((APPROXIMATE, "Fs == Fc: frictionloss single level") if (eng == "mujoco" and static_eq)
                             else (NOT_REPRESENTED, "no engine-independent breakaway (Fs > Fc) field"))
        else:
            row["static"] = ("DISABLED", "")
        if terms.gravity:
            row["gravity"] = (REPRESENTED, "engine gravity on the link mass / COM; NO fitted G(q) torque injected")
        else:
            small = (gravity_bound is not None and effort_scale and gravity_bound <= negligible_rel * effort_scale)
            row["gravity"] = (("NEGLIGIBLE", f"max|G| = {gravity_bound:.3g} <= {negligible_rel:.0%} of effort")
                              if small else ("GRAVITY_CONFIGURATION_MISMATCH",
                                             "identified gravity-OFF but the engine applies gravity to the link mass; "
                                             "parity only in an explicitly gravity-free scene"))
        row["bias"] = (NOT_REPRESENTED, "nuisance load: never exported") if terms.bias else ("DISABLED", "")
        enabled = [t for t in ("inertia", "viscous", "coulomb", "static", "gravity", "bias") if getattr(terms, t)]
        blocking = [t for t in enabled if row[t][0] in (NOT_REPRESENTED, "GRAVITY_CONFIGURATION_MISMATCH")]
        row["full_model_representable"] = not blocking
        row["blocking_terms"] = blocking
        m[eng] = {k: (dict(status=v[0], mapping=v[1]) if isinstance(v, tuple) else v) for k, v in row.items()}
    return m


def representable_terms(terms: GeneralTerms, engine: str, Fc=None, Fs=None) -> GeneralTerms:
    """Largest sub-model the engine can simulate (for the numerical comparison)."""
    tm = term_matrix(terms, Fc, Fs)[engine]
    keep = {t: getattr(terms, t) and tm[t]["status"] in (REPRESENTED, APPROXIMATE, EMPIRICAL) for t in
            ("viscous", "coulomb", "static", "gravity", "bias")}
    return GeneralTerms(inertia=True, **keep)


def _run(py, script, args, timeout=900):
    p = subprocess.run([py, str(SCRIPTS / script), *map(str, args)], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{script} failed: {p.stderr[-2000:]}")


def engine_rollout(engine, model_path, python, effort, t_query, q0=0.0, v0=0.0, gravity=ENGINE_GRAVITY_W,
                   T_world_root=None, timestep=None, friction_override=None) -> dict:
    with tempfile.TemporaryDirectory(prefix="g9-parity-") as td:
        req = dict(gravity=list(gravity), effort_t_s=np.asarray(effort.t_s).tolist(),
                   effort_levels=np.asarray(effort.levels).tolist(), t_query=np.asarray(t_query).tolist(),
                   q0=q0, v0=v0, T_world_root=(np.eye(4) if T_world_root is None else np.asarray(T_world_root)).tolist())
        if timestep:
            req["timestep"] = timestep
        if friction_override is not None:
            req["diagnostic_joint_friction_override"] = friction_override
        rp, op = Path(td) / "req.json", Path(td) / "out.json"
        rp.write_text(json.dumps(req))
        _run(python, "mujoco_rollout.py" if engine == "mujoco" else "sapien_rollout.py", [model_path, rp, op])
        return json.loads(op.read_text())


def numerical_parity(engine, model_path, python, *, joint_type, M, c, Fc, Fs, G, terms: GeneralTerms, effort,
                     t_query, q0=0.0, v0=0.0, gravity=ENGINE_GRAVITY_W, T_world_root=None, tol_abs=None,
                     tol_rel_range=0.03) -> dict:
    """Engine q(t) vs standalone G9 rollout of the engine-representable sub-model (same effort, same state)."""
    if python is None:
        return dict(status="NOT_RUN", reason=f"no {engine} runtime")
    sub = representable_terms(terms, engine, Fc, Fs)
    p = sim_params(sub, joint_type, M=M, c=c if sub.viscous else None, Fc=Fc if sub.coulomb else None,
                   Fs=(Fs if sub.static else None), b=None, G=G if sub.gravity else None)
    ref = simulate_general(p, effort, t_query, t0=0.0, q0=q0, v0=v0, max_step=2.5e-4).q
    try:
        eng = engine_rollout(engine, model_path, python, effort, t_query, q0, v0,
                             gravity if sub.gravity else (0.0, 0.0, 0.0), T_world_root)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        return dict(status="FAIL", reason=str(e)[:1500])
    qe = np.asarray(eng["q"], float)
    err = qe - ref
    rng = float(np.ptp(ref)) or 1.0
    rmse = float(np.sqrt(np.mean(err ** 2)))
    tol = tol_abs if tol_abs is not None else tol_rel_range * rng
    st = "PASS" if rmse <= tol else "FAIL"
    # EMPIRICAL terms: also test the hypothesis "the engine does not apply this term at all"
    hyp = {}
    tmx = term_matrix(terms, Fc, Fs)[engine]
    for t in ("coulomb",):
        if getattr(sub, t) and tmx[t]["status"] == EMPIRICAL:
            alt = GeneralTerms(**{**sub.to_dict(), t: False, "static": False if t == "coulomb" else sub.static})
            pa = sim_params(alt, joint_type, M=M, c=c if alt.viscous else None, Fc=None, Fs=None, b=None,
                            G=G if alt.gravity else None)
            ra = simulate_general(pa, effort, t_query, t0=0.0, q0=q0, v0=v0, max_step=2.5e-4).q
            r_alt = float(np.sqrt(np.mean((qe - ra) ** 2)))
            hyp[t] = dict(rmse_if_engine_ignores_term=r_alt, rmse_if_engine_applies_term=rmse,
                          conclusion=("ENGINE_IGNORES_TERM -> NOT_REPRESENTED" if r_alt <= tol < rmse else
                                      "ENGINE_APPLIES_TERM" if rmse <= tol else "UNEXPLAINED_SEMANTICS"))
    if hyp and all(v["conclusion"].startswith("ENGINE_IGNORES") for v in hyp.values()):
        st = "FAIL_TERM_NOT_REPRESENTED"
    diag = None
    if engine == "sapien_urdf" and st != "PASS":
        # diagnostic only: is the mismatch the loader's PhysX joint-friction coefficient? (asset unchanged)
        try:
            e0 = engine_rollout(engine, model_path, python, effort, t_query, q0, v0,
                                gravity if sub.gravity else (0.0, 0.0, 0.0), T_world_root, friction_override=0.0)
            sub0 = GeneralTerms(**{**sub.to_dict(), "coulomb": False, "static": False})
            p0 = sim_params(sub0, joint_type, M=M, c=c if sub0.viscous else None, Fc=None, Fs=None, b=None,
                            G=G if sub0.gravity else None)
            r0 = simulate_general(p0, effort, t_query, t0=0.0, q0=q0, v0=v0, max_step=2.5e-4).q
            rm0 = float(np.sqrt(np.mean((np.asarray(e0["q"]) - r0) ** 2)))
            diag = dict(loaded_physx_joint_friction=eng.get("joint", {}).get("friction"),
                        rmse_with_physx_friction_zeroed_vs_model_without_dry_friction=rm0,
                        conclusion=("the loader applies a PhysX joint-friction coefficient (dimensionless, normal-"
                                    "force proportional) that the URDF <dynamics friction> does not control as a "
                                    "Coulomb torque; inertia / damping / gravity match once it is zeroed"
                                    if rm0 <= tol else "mismatch persists without PhysX joint friction"))
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            diag = dict(error=str(e)[:500])
    return dict(status=st, engine=engine, compared_terms=sub.status(), rmse=rmse, max_abs=float(np.abs(err).max()),
                empirical_term_tests=hyp, sapien_friction_diagnostic=diag,
                q_range=rng, tolerance=tol, engine_readback=eng.get("joint"), engine_gravity=eng.get("gravity"),
                engine_version=eng.get("mujoco_version") or eng.get("sapien_version"),
                gravity_free_scene=not sub.gravity,
                q_engine=qe.tolist(), q_reference=ref.tolist(), t_s=np.asarray(t_query).tolist())


def gravity_projection_check(joint_type, axis, origin, com0, mass, q_values, g=ENGINE_GRAVITY_W) -> dict:
    """Analytic G(q) = dU/dq vs the generalized projection of the engine gravity force m g at the COM:
    Q_g = (a x (c - o)) . m g (R) or a . m g (P); G must equal -Q_g (spec 7.2)."""
    from ..dynamics.general_1dof import RigidBodyGravity
    rb = RigidBodyGravity(joint_type, tuple(axis), tuple(origin), tuple(com0), mass, tuple(g))
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    errs = []
    for q in q_values:
        c = rb.com(q)
        f = mass * np.asarray(g, float)
        Qg = (np.cross(a, c - np.asarray(origin, float)) @ f) if joint_type == "revolute" else a @ f
        errs.append(abs(rb.G(q) + Qg))
    return dict(status="PASS" if max(errs) <= 1e-9 * max(1.0, mass * 10) else "FAIL", max_abs_err=float(max(errs)))
