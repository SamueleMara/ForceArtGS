"""G9 core: one fail-closed sequence from verified inputs to physics documents (G9 spec 9, stop rules).

    kinematics gate -> force-calibration gate -> mass model -> gravity gate -> fit (integral LS -> rollout)
    -> identifiability -> adequacy -> held-out prediction -> [variants] -> [frozen viscous baselines]
    -> geometry-constrained inertia -> parameter table -> dynamics handoff -> physics spec v1 (only if complete)

``run_g9`` never trains Gaussians and never touches PLYs. Every stop rule ends in an explicit readiness / BLOCKED_*
state with the reason; a blocked run still writes its diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import G9_VERSION
from .config import terms_of
from .dynamics.general_1dof import GeneralTerms, GravityField, ModelConfigError
from .dynamics.geometry_inertia import (GeometryMassModel, InertiaReconstructionError, alignment_check,
                                       check_user_inertial)
from .dynamics.mass_properties import MeshError, TriMesh
from .estimators.general_constrained import general_rollout_fit, step_convergence
from .estimators.general_integral_ls import general_integral_ls, theta0_from_ls
from .estimators.model_spec import GravityRegistrationError, ModelSpec
from .export import dynamics_v2 as X
from .validation.general_identifiability import diagnose_general, standard_perturbations, structural_checks
from .validation.heldout_prediction import heldout_report
from .validation.model_adequacy import adequacy, compare_variants
from .validation.simulator_parity import term_matrix

READINESS = ("KINEMATICS_ACCEPTED", "DYNAMICS_DIAGNOSTIC", "PARAMETERS_IDENTIFIED", "MASS_MODEL_ASSUMED",
             "INERTIA_RECONSTRUCTED", "MODEL_VALIDATED", "VISUAL_ASSET_READY", "COLLISION_READY", "URDF_COMPILES",
             "USD_LOADS", "MJCF_COMPILES", "ENGINE_PARITY_PASS", "FULL_PHYSICS_READY")


@dataclass
class G9Inputs:
    scene: str
    joint_type: str
    axis: np.ndarray
    origin: Optional[np.ndarray]
    kinematic_status: str                         # IDENTIFIED | AMBIGUOUS | UNOBSERVABLE | REJECTED | STATIC | ...
    kinematics_export_ready: bool                 # upstream physics_joint_export_ready (G7/G8)
    kinematics_provenance: dict
    train: list
    heldout: list
    force_audit: dict                             # {status: VERIFIED | FORCE_CALIBRATION_UNVERIFIED | NO_FORCE, ...}
    gravity: Optional[GravityField] = None
    mass_mesh: Optional[TriMesh] = None
    mass_mesh_provenance: Optional[dict] = None
    axis_std_rad: float = 0.0
    origin_std_m: float = 0.0
    anchor_frame: Optional[int] = None
    visual_handoff: Optional[str] = None
    mechanical_spec: Optional[dict] = None
    excitation_only: list = field(default_factory=list)
    input_hashes: dict = field(default_factory=dict)
    synthetic: bool = False
    moving_gaussian_xyz: Optional[np.ndarray] = None     # canonical moving Gaussian means (mass-mesh alignment)
    extra_perturbations: dict = field(default_factory=dict)   # evidence-based input sweeps (adapter)
    com_direction: Optional[np.ndarray] = None                 # hinge -> moving-support centroid (perp. to axis)
    com_lever_m: Optional[float] = None
    record_context: Optional[dict] = None                      # canonical record + q for the impulse / coast analysis


def _sig(obj) -> str:
    return hashlib.sha256(json.dumps(X._clean(obj), sort_keys=True, default=str).encode()).hexdigest()


def _write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(X._clean(obj), indent=1, default=str))


def _strip(fit):
    if fit is None:
        return None
    return {k: v for k, v in fit.items() if k not in ("problem", "jacobian")}


def _mass_model(inp: G9Inputs, cfg, log) -> tuple:
    mm = cfg["mass_model"]
    if inp.mass_mesh is None:
        return None, "NO_MASS_MODEL_GEOMETRY: no validated moving-part surface mesh (mass_model.mesh)"
    if mm["mode"] not in ("uniform_solid", "uniform_shell"):
        return None, f"mass_model.mode {mm['mode']}: not a geometry-integration mode"
    try:
        g = GeometryMassModel.from_validated_mesh(
            inp.mass_mesh, mm["mode"], inp.joint_type, inp.axis, inp.origin, canonical_frame=mm["geometry_frame"],
            source_provenance=inp.mass_mesh_provenance or mm.get("source_provenance") or {},
            reflected_actuator_inertia=mm["reflected_actuator_inertia"],
            is_collision_proxy=bool(mm.get("mesh_is_collision_proxy")),
            allow_collision_proxy=bool(mm["allow_collision_proxy_as_mass_geometry"]))
        g.reflected()                                   # undeclared reflected inertia blocks the coupling here
        if inp.moving_gaussian_xyz is not None:
            al = alignment_check(inp.mass_mesh, inp.moving_gaussian_xyz)
            g.source_provenance["alignment_to_moving_gaussians"] = al
            if al["status"] != "PASS":
                return None, f"MASS_MODEL_MISALIGNED: {'; '.join(al['reasons'])}"
        return g, None
    except (MeshError, InertiaReconstructionError) as e:
        return None, f"MASS_MODEL_REJECTED: {e}"


def run_g9(inp: G9Inputs, cfg: dict, out_dir, log=print) -> dict:
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    terms = terms_of(cfg)
    R = {k: False for k in READINESS}
    blocked, notes = [], []
    X.write_yaml(out / "general_model_config.yaml", {k: v for k, v in cfg.items() if not k.startswith("_")},
                 "resolved G9 dynamics configuration (all toggles)")
    # ------------------------------------------------------------------ kinematics
    kin_ok = inp.kinematic_status == "IDENTIFIED"
    kin_declared = inp.kinematic_status == "OPERATOR_DECLARED"     # conditional use only, never certified
    R["KINEMATICS_ACCEPTED"] = kin_ok
    diagnostic_kin = not kin_ok
    if kin_declared:
        blocked.append("NOT_CERTIFIED_OPERATOR_DECLARED_KINEMATICS: joint type declared by the operator "
                       f"({inp.kinematics_provenance.get('declaration_note')}); upstream status "
                       f"{inp.kinematics_provenance.get('selection_status')}")
    elif not kin_ok:
        blocked.append(f"BLOCKED_KINEMATICS: upstream status {inp.kinematic_status} (diagnostic only)")
        if not cfg["pipeline"]["allow_diagnostic_kinematics"]:
            notes.append("dynamics not fitted: allow_diagnostic_kinematics is false")
    if kin_ok and not inp.kinematics_export_ready:
        blocked.append("BLOCKED_KINEMATIC_EXPORT_READINESS: upstream physics_joint_export_ready is false "
                       f"({inp.kinematics_provenance.get('failure_codes')})")
    # ------------------------------------------------------------------ force
    fa = dict(inp.force_audit)
    rec = fa.pop("canonical_record", None)
    if rec is not None:
        from estimotion_2part.joint_discovery.robot_record import save_robot_record
        save_robot_record(rec, out / "robot_record")
        fa["canonical_record_file"] = "robot_record/canonical_robot_record.{npz,json}"
    _write(out / "force_preprocessing_report.json", fa)
    force_ok = fa.get("status") == "VERIFIED"
    exploratory = fa.get("status") == "EXPLORATORY_ASSUMPTIONS_ACCEPTED"
    diag_force = False
    if not force_ok:
        blocked.append(f"{'NOT_CERTIFIED_' if exploratory_pre(fa) else 'BLOCKED_'}"
                       f"{fa.get('status', 'FORCE_CALIBRATION_UNVERIFIED')}: "
                       f"{'; '.join(fa.get('failing', [])[:9]) or fa.get('reason', '')}")
        diag_force = cfg["pipeline"]["on_missing_calibration"] == "diagnostic_only" and bool(inp.train)
    can_fit = bool(inp.train) and (force_ok or exploratory or diag_force) and (
        kin_ok or kin_declared or cfg["pipeline"]["allow_diagnostic_kinematics"])
    physical_claim = force_ok and kin_ok and not inp.force_audit.get("diagnostic")
    if exploratory:
        label = ("EXPLORATORY_CONDITIONAL: estimates conditional on the listed calibration assumptions"
                 + ("" if kin_ok else "; OPERATOR_DECLARED_KINEMATICS" if kin_declared else
                    f"; KINEMATICS_NOT_ACCEPTED ({inp.kinematic_status}, working model)"))
    else:
        label = ("PHYSICAL" if physical_claim else
                 "NOT_PHYSICAL: " + ", ".join(x.split(":")[0] for x in blocked[:3]))
    # ------------------------------------------------------------------ mass model / parameter source
    src = cfg["model"]["parameter_source"]
    mm, mm_reason = _mass_model(inp, cfg, log) if src == "geometry_coupled" else (None, "not requested")
    if src == "geometry_coupled" and mm is None:
        blocked.append(f"BLOCKED_MASS_MODEL: {mm_reason}")
        if cfg["pipeline"]["on_missing_geometry"] == "abstain":
            can_fit = False
        src = "joint_space_only"
        notes.append("fallback JOINT_SPACE_ONLY diagnostic: no link inertia tensor will be produced")
    if mm is not None:
        R["MASS_MODEL_ASSUMED"] = True
        _write(out / "unit_density_inertia.json", dict(**mm.props.to_dict(), joint_type=inp.joint_type,
                                                       M1=mm.M1, axis=inp.axis.tolist(),
                                                       origin=None if inp.origin is None else inp.origin.tolist()))
        _write(out / "mass_model_geometry_manifest.json", dict(
            mode=mm.props.mode, mesh_sha256=mm.props.mesh_sha256, provenance=mm.source_provenance,
            frame=mm.props.frame, assumptions=mm.assumptions, collision_proxy=mm.collision_proxy,
            reflected_actuator_inertia=mm.reflected_actuator_inertia, topology=mm.props.topology))
    # ------------------------------------------------------------------ model spec / gravity gate
    spec, fit, ls, ident, adeq, held, variants, baselines, conv = (None,) * 9
    try:
        spec = ModelSpec(terms, inp.joint_type, src, mm, inp.gravity, **_js_kw(inp, exploratory))
    except GravityRegistrationError as e:
        blocked.append(f"BLOCKED_GRAVITY_REGISTRATION: {e}")
        can_fit = False
    except ModelConfigError as e:
        blocked.append(f"BLOCKED_MODEL_CONFIG: {e}")
        can_fit = False
    # ------------------------------------------------------------------ fitting
    struct = None
    if spec is not None:
        struct = structural_checks(inp.train or inp.excitation_only, spec) if (inp.train or inp.excitation_only) \
            else None
    est_dir = out / "estimator_results"
    if can_fit and spec is not None:
        log(f"G9 fit: {terms.model_id} [{src}] on {len(inp.train)} interaction(s), {label}")
        ls = general_integral_ls(inp.train, spec)
        _write(est_dir / "general_integral_ls.json", dict(**ls, label=label))
        eff = float(np.max(np.abs(np.concatenate([np.asarray(i.effort.levels) for i in inp.train]))))
        th0 = theta0_from_ls(ls, spec, dict(effort=eff))
        fit = general_rollout_fit(inp.train, spec, th0, n_starts=int(cfg["fitting"]["n_starts"]),
                                  seed=int(cfg["fitting"]["seed"]), robust=cfg["fitting"]["robust_loss"],
                                  max_step=float(cfg["model"]["integrator_max_step_s"]), scale_hint=dict(effort=eff))
        _write(est_dir / "general_rollout_fit.json", dict(**_strip(fit), label=label))
        if fit.get("status") in ("ok", "not_converged"):
            th = np.asarray(fit["x"][:len(spec.names)])
            conv = step_convergence(spec, th, inp.train[0])
            idc = cfg["identifiability"]
            pert = standard_perturbations(lag_s=idc["force_lag_sweep_s"], scale_rel=idc["force_scale_sweep_rel"],
                                          bias=idc["force_bias_sweep"]) if idc["propagate_calibration_uncertainty"] \
                else {}
            pert.update(inp.extra_perturbations or {})
            ident = diagnose_general(fit, inp.train, spec, perturb=pert)
            adeq = adequacy(fit)
            if cfg["fitting"]["holdout_mode"] == "independent_episodes":
                held = heldout_report(spec, th, inp.heldout)
            var = cfg["fitting"].get("model_variants")
            if var and inp.heldout:
                vres = {}
                for name, tdict in var.items():
                    try:
                        vt = GeneralTerms(**tdict)
                        vs = ModelSpec(vt, inp.joint_type, src, mm, inp.gravity, **_js_kw(inp, exploratory))
                        vls = general_integral_ls(inp.train, vs)
                        vf = general_rollout_fit(inp.train, vs, theta0_from_ls(vls, vs, dict(effort=eff)),
                                                 n_starts=2, seed=0)
                        vres[name] = heldout_report(vs, np.asarray(vf["x"][:len(vs.names)]), inp.heldout)
                        vres[name]["terms"] = vt.status()
                    except (ModelConfigError, ValueError) as e:
                        vres[name] = dict(status="NOT_RUN", reason=str(e))
                variants = dict(variants={k: {kk: vv for kk, vv in v.items() if kk != "per_episode"}
                                          for k, v in vres.items()}, selection=compare_variants(vres),
                                note="opt-in comparison; the requested model is NOT replaced automatically")
            bl = cfg["fitting"]["compare_frozen_viscous_baselines"]
            if bl:
                from .estimators.frozen_baselines import run_baselines
                baselines = run_baselines(inp.train, inp.axis, inp.origin, bl)
                _write(est_dir / "frozen_viscous_baselines.json", dict(**baselines, label=label))
    elif not can_fit:
        _write(est_dir / "general_rollout_fit.json", dict(status="NOT_RUN", blocked=blocked, label=label))
    if inp.record_context is not None:
        from .validation.impulse_coast import analyze as _ic, plot as _icplot
        rc = inp.record_context
        ics = []
        for it in (inp.train or inp.excitation_only):
            w = it.meta.get("source_rows")
            if not w:
                continue
            try:
                an = _ic(rc["rec"], rc["joint_type"], rc["axis"], rc["origin"], rc["q_src"], rc["q"], rc["q_std"], w,
                         observability=rc.get("observability"))
                _icplot(an, out / f"impulse_coast_{it.name}.png")
                an.pop("_plot", None)
                an["interaction"] = it.name
                ics.append(an)
            except (ValueError, np.linalg.LinAlgError) as e:
                ics.append(dict(interaction=it.name, verdict="ANALYSIS_FAILED", error=str(e)))
        _write(out / "impulse_coast_analysis.json", dict(
            schema="forceartgs.g9.impulse_coast/v1", windows=ics,
            note="pulse and coast evaluated separately on native 50 Hz force rows; verdicts "
                 "IMPULSE_INFORMATIVE / IMPULSE_UNRESOLVED / IMPULSE_NOT_INFORMATIVE"))
    _write(out / "general_identifiability.json", dict(fit_identifiability=ident, structural_from_data=struct,
                                                      step_convergence=conv, label=label) if ident or struct else
           dict(status="NOT_RUN", blocked=blocked))
    _write(out / "model_adequacy.json", dict(adequacy=adeq, variants=variants, label=label) if adeq else
           dict(status="NOT_RUN", blocked=blocked))
    hd = out / "heldout_prediction"
    hd.mkdir(exist_ok=True)
    _write(hd / "heldout_report.json", held or dict(status="NOT_RUN", reason="no fit or no held-out episodes"))
    R["DYNAMICS_DIAGNOSTIC"] = fit is not None and fit.get("status") in ("ok", "not_converged")
    # ------------------------------------------------------------------ parameter table
    table = X.parameter_table(terms, inp.joint_type, fit if R["DYNAMICS_DIAGNOSTIC"] else
                              dict(blocked_reason="; ".join(blocked)), ident, spec, physical_claim=physical_claim,
                              label=label, exploratory=exploratory)
    if (R["DYNAMICS_DIAGNOSTIC"] and cfg["fitting"].get("load_model_comparison") and
            spec is not None and spec.parameter_source == "joint_space_only"):
        from .validation.load_models import load_model_comparison
        try:
            lm = load_model_comparison(inp.train, terms, inp.joint_type)
        except Exception as e:
            lm = dict(status="FAILED", error=f"{type(e).__name__}: {e}")
        _write(out / "load_model_comparison.json", dict(lm, label=label))
    params_ok = bool(R["DYNAMICS_DIAGNOSTIC"] and fit.get("converged") and ident and ident["status"] == "PASS"
                     and physical_claim and all(X.term_ok(table, t) for t in table))
    R["PARAMETERS_IDENTIFIED"] = params_ok
    if R["DYNAMICS_DIAGNOSTIC"] and not params_ok:
        bad = [t for t in table if not X.term_ok(table, t)]
        blocked.append(f"BLOCKED_UNIDENTIFIABLE: enabled term(s) {bad} not identified (simpler model only by explicit "
                       f"user choice)" if bad else "BLOCKED_PARAMETERS: identifiability gate FAIL")
    held_ok = bool(held and held["status"] == "PASS")
    verdicts(table, ident, adeq, kin_ok or kin_declared, force_ok and kin_ok, held_ok, struct)
    R["MODEL_VALIDATED"] = bool(params_ok and adeq and adeq["status"] == "PASS" and
                                (held_ok or not cfg["export"]["production_requires_heldout_real_prediction"]))
    if params_ok and not R["MODEL_VALIDATED"]:
        blocked.append(f"BLOCKED_MODEL_VALIDATION: adequacy {adeq and adeq['status']}, held-out "
                       f"{held and held['status']} ({held and held.get('reason')})")
    # ------------------------------------------------------------------ inertia reconstruction
    recon = None
    if mm is not None and R["DYNAMICS_DIAGNOSTIC"] and "alpha" in fit["estimates"]:
        a = fit["estimates"]["alpha"]["value"]
        a_sd = (ident or {}).get("parameters", {}).get("alpha", {}).get("std_total") or fit["estimates"]["alpha"]["std"]
        try:
            mc = mm.monte_carlo(a, a_sd, axis_std_rad=inp.axis_std_rad, origin_std_m=inp.origin_std_m,
                                mesh_std_m=float((inp.mass_mesh_provenance or {}).get("mesh_std_m") or 0.0))
            recon = mm.reconstruct(a, a_sd, fitted_M_eff=fit["physical"]["M"], mc=mc)
            recon["alpha_identifiable"] = table["inertia"].get("status") == "identified"
            if cfg["mass_model"].get("user_link_inertial"):
                recon["user_inertial_check"] = check_user_inertial(cfg["mass_model"]["user_link_inertial"], mm,
                                                                   inp.joint_type, inp.axis, inp.origin,
                                                                   fit["physical"]["M"], a_sd * mm.M1)
            R["INERTIA_RECONSTRUCTED"] = bool(params_ok and recon["projection_check"] == "PASS" and
                                              recon["tensor_checks"]["ok"] and
                                              recon.get("user_inertial_check", {}).get("status", "CONSISTENT") ==
                                              "CONSISTENT")
        except InertiaReconstructionError as e:
            recon = dict(status="BLOCKED", reason=str(e))
    _write(out / "inertia_reconstruction_report.json", recon or dict(
        status="NOT_RUN", reason=mm_reason if mm is None else "no fitted density scale",
        full_tensor=None, note="joint-space-only results never produce a link inertia tensor"))
    if recon and recon.get("user_inertial_check", {}).get("status") == "INERTIA_GEOMETRY_CONFLICT":
        blocked.append("BLOCKED_INERTIA_GEOMETRY_CONFLICT: user inertial disagrees with the identified joint inertia")
    # ------------------------------------------------------------------ documents
    eff_scale = None
    if inp.train:
        eff_scale = float(np.sqrt(np.mean(np.concatenate([np.asarray(i.effort.levels) ** 2 for i in inp.train]))))
    gbound = None
    if mm is not None and inp.gravity is not None and fit is not None and "alpha" in fit.get("estimates", {}):
        gbound = fit["estimates"]["alpha"]["value"] * mm.gravity_unit(inp.gravity.g_world).amplitude_bound()
    phys = (fit or {}).get("physical") or {}
    tm = term_matrix(terms, phys.get("Fc"), phys.get("Fs"), gbound, eff_scale)
    physics_parameters = dict(schema="forceartgs.g9.physics_parameters/v1", label=label, model=spec.describe() if spec
                              else dict(terms=terms.status()), joint_type=inp.joint_type, terms=table,
                              physical=phys if R["DYNAMICS_DIAGNOSTIC"] else None,
                              production_fields_valid=params_ok,
                              note="numbers of unidentifiable or NOT_PHYSICAL terms are diagnostic only")
    _write(out / "physics_parameters.json", physics_parameters)
    recon_capture = recon if R["INERTIA_RECONSTRUCTED"] else None
    spec_v1, missing = X.physics_spec_v1(table=table, recon_capture=recon_capture, mech=inp.mechanical_spec,
                                         anchor_frame=inp.anchor_frame if inp.anchor_frame is not None else 0,
                                         g9_gravity=inp.gravity, joint_type=inp.joint_type, phys=phys)
    production = bool(R["MODEL_VALIDATED"] and R["INERTIA_RECONSTRUCTED"] and inp.kinematics_export_ready)
    if not inp.kinematics_export_ready:
        missing = list(missing) + ["upstream kinematics not export-ready"]
    if spec_v1 is not None and production:
        X.write_yaml(out / "physics_spec_v1.yaml", spec_v1, "PArtTwin1D physics spec v1 GENERATED by G9; pass to "
                     "parttwin1d.cli.build_asset --dynamics (final_physics_export)")
    elif spec_v1 is not None:
        missing = list(missing) + [f"production gates: {[b.split(':')[0] for b in blocked]}"]
    handoff = {"schema": X.HANDOFF_SCHEMA, "g9_version": G9_VERSION, "scene": inp.scene, "label": label,
               "kinematics": dict(status=inp.kinematic_status, export_ready=inp.kinematics_export_ready,
                                  joint_type=inp.joint_type, axis_W=inp.axis.tolist(),
                                  origin_W=None if inp.origin is None else inp.origin.tolist(),
                                  provenance=inp.kinematics_provenance),
               X.PARTTWIN_V2: {"joint_space_identification": table,
                               "link_inertial_moving": None if recon_capture is None else dict(
                                   frame="capture W at the canonical anchor (q = 0)", anchor_frame=inp.anchor_frame,
                                   mass_kg=recon["mass_kg"], com_m=recon["com_m"],
                                   inertia_com_kg_m2=recon["inertia_com_kg_m2"], provenance=recon["provenance"],
                                   uncertainty=recon.get("uncertainty"),
                                   systematic=recon["systematic_not_bootstrappable"]),
                               "engine_term_matrix": tm,
                               "gravity": None if inp.gravity is None else dict(
                                   g_world=list(inp.gravity.g_world), verified=inp.gravity.verified,
                                   provenance=inp.gravity.provenance,
                                   simulator_application="engines apply gravity from mass/COM; no G(q) torque "
                                                         "is injected")},
               "physics_spec_v1": "physics_spec_v1.yaml" if (spec_v1 is not None and production) else None,
               "missing_for_physics_spec_v1": missing, "blocked": blocked,
               "not_a_full_physics_spec": "this file is never passed to the builder's --dynamics"}
    X.write_yaml(out / "dynamics_handoff.yaml", handoff, "G9 dynamics handoff (diagnostic + production fields)")
    # ------------------------------------------------------------------ manifest
    sig_obj = dict(g9=G9_VERSION, cfg={k: v for k, v in cfg.items() if not k.startswith("_")},
                   inputs=inp.input_hashes, train=[(i.name, _sig([i.t_obs, i.q_obs, i.effort.levels])) for i in inp.train],
                   heldout=[(i.name, _sig([i.t_obs, i.q_obs, i.effort.levels])) for i in inp.heldout])
    st = ("PRODUCTION_PHYSICS_SPEC_READY" if (spec_v1 is not None and production) else
          ("DYNAMICS_DIAGNOSTIC" if R["DYNAMICS_DIAGNOSTIC"] else "BLOCKED"))
    if st != "PRODUCTION_PHYSICS_SPEC_READY" and exploratory and R["DYNAMICS_DIAGNOSTIC"]:
        st = "EXPLORATORY_ESTIMATE"
    categories = status_categories(inp, fa, kin_ok, ident, table, missing, blocked, mm_reason if mm is None else None)
    man = dict(schema="forceartgs.g9.manifest/v1", g9_version=G9_VERSION, scene=inp.scene, status=st,
               status_categories=categories,
               readiness=R, blocked=blocked, notes=notes, label=label, synthetic=inp.synthetic,
               input_signature=_sig(sig_obj), input_hashes=inp.input_hashes,
               physics_library=_physics_prov(), terms=terms.status(), parameter_source=src,
               fit_summary=None if fit is None else dict(status=fit.get("status"), converged=fit.get("converged"),
                                                         residual_rms_sigma=fit.get("residual_rms_sigma"),
                                                         runtime_s=fit.get("runtime_s")),
               identifiability=None if ident is None else ident["status"], adequacy=None if adeq is None else
               adeq["status"], heldout=None if held is None else held["status"],
               physics_spec_v1="physics_spec_v1.yaml" if (spec_v1 is not None and production) else None,
               missing_for_physics_spec_v1=missing, runtime_s=time.time() - t_start,
               readiness_note="physics_joint_export_ready (upstream kinematics) is NOT a dynamics claim; "
                              "FULL_PHYSICS_READY is set only by final_physics_export after engine checks")
    _write(out / "g9_manifest.json", man)
    _write(out / "input_manifest.json", dict(sig_obj, physics_library=_physics_prov(), signature=man["input_signature"]))
    log(f"G9 {inp.scene}: {st}; blocked {[b.split(':')[0] for b in blocked]}")
    return dict(manifest=man, fit=fit, ident=ident, recon=recon, table=table, spec=spec, held=held, adeq=adeq,
                term_matrix=tm, physics_spec_v1=spec_v1 if production else None, out_dir=str(out),
                baselines=baselines, variants=variants, ls=ls)


def verdicts(table, ident, adeq, kin_ok, force_ok, held_ok, struct):
    """physically_interpretable_validated needs: statistically resolved in this model, not scenario-sensitive,
    adequate fit, sufficient excitation (no structural flag on it), an ACCEPTED joint, verified force and a passed
    held-out validation. conditionally_interpretable drops the last two (exploratory meaning only)."""
    unexcited = {p for p, _ in (struct or {}).get("unidentifiable", [])}
    adeq_ok = bool(adeq and adeq.get("status") == "PASS")
    for t, e in table.items():
        for n, v in (e.get("parameters") or {}).items():
            if v.get("fitted_or_fixed") != "fitted":
                continue
            base = bool(v.get("parameter_statistically_resolved_in_this_model") and not v.get("scenario_sensitive")
                        and adeq_ok and kin_ok and n not in unexcited)
            v["conditionally_interpretable"] = base
            v["physically_interpretable_validated"] = bool(base and force_ok and held_ok)
            why = []
            if not v.get("parameter_statistically_resolved_in_this_model"):
                why.append("not statistically resolved in this model")
            if v.get("scenario_sensitive"):
                why.append("sensitive to declared scenarios")
            if not adeq_ok:
                why.append(f"model adequacy {None if adeq is None else adeq.get('status')}")
            if not kin_ok:
                why.append("joint type not accepted (working model)")
            if n in unexcited:
                why.append("insufficient excitation")
            if not force_ok:
                why.append("force calibration not certified")
            if not held_ok:
                why.append("no passed independent held-out validation")
            v["interpretation_blockers"] = why


def _js_kw(inp, exploratory):
    """Joint geometry for joint_space_gravity_coupled; DECLARED gravity only under the exploratory policy."""
    g = inp.gravity
    declared = bool(g is not None and not g.verified and str(g.provenance).startswith("DECLARED"))
    return dict(joint_axis=inp.axis, joint_origin=inp.origin, com_direction=inp.com_direction,
                com_lever_m=inp.com_lever_m, allow_unverified_gravity=bool(exploratory and declared))


def exploratory_pre(fa):
    return fa.get("status") == "EXPLORATORY_ASSUMPTIONS_ACCEPTED"


def status_categories(inp, fa, kin_ok, ident, table, missing, blocked, mesh_reason) -> dict:
    """Separate WHY something is not a certified physical result (never one generic BLOCKED)."""
    data = []
    if not inp.train and not inp.excitation_only:
        data.append(fa.get("reason") or f"no usable force / q interaction (force audit {fa.get('status')})")
    if fa.get("status") in ("NO_FORCE", "DATA_UNAVAILABLE", "NOT_EVALUATED"):
        data.append(f"force: {fa.get('status')} {fa.get('reason', '')}".strip())
    gates = fa.get("gates") or {}
    calib = [f"{g}: {v['status']}" for g, v in gates.items()
             if v.get("status") not in ("PASS", "NOT_NEEDED") and not v["status"].startswith("ESTIMATED")]
    estimated = [f"{g}: {v['status']}" for g, v in gates.items() if v.get("status", "").startswith("ESTIMATED")]
    dyn = {}
    if ident:
        dyn = {k: v["reasons"] for k, v in ident["parameters"].items() if not v["identifiable"]}
    for t, e in table.items():
        if e.get("status") == "geometry_coupled" and e.get("identifiable") is False:
            dyn[t] = ["geometry-coupled gravity follows alpha, which is not identified"]
    asset = list(missing)
    if mesh_reason and mesh_reason != "not requested":
        asset.append(f"mass-model geometry: {mesh_reason} (does not block joint-space estimation)")
    return dict(DATA_UNAVAILABLE=data,
                KINEMATICS_NOT_ACCEPTED=[] if kin_ok else [f"upstream joint status {inp.kinematic_status}"],
                CALIBRATION_ASSUMPTION_UNVERIFIED=calib, CALIBRATION_ESTIMATED_FROM_DATA=estimated,
                DYNAMICS_NOT_IDENTIFIABLE=dyn,
                SIMULATION_ASSET_INCOMPLETE=asset,
                note="SIMULATION_ASSET_INCOMPLETE never blocks joint-space estimation; "
                     "CALIBRATION_ASSUMPTION_UNVERIFIED blocks certification, not exploratory estimation")


def _physics_prov():
    from . import physics_lib
    try:
        return physics_lib.provenance()
    except physics_lib.PhysicsLibraryError:
        return dict(status="NOT_LOADED")
