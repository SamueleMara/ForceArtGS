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
    reference_pose: Optional[dict] = None                      # q_ref of the mass geometry (verified / declared)


def code_fingerprint() -> str:
    """sha256 over the G9 package sources and the shared force-convention / robot-record code: a code change writes
    a new version instead of silently reusing results produced by older code."""
    root = Path(__file__).resolve().parent
    files = sorted(root.rglob("*.py")) + [root.parent / "estimotion_2part" / "joint_discovery" / n
                                          for n in ("force_convention.py", "robot_record.py")]
    h = hashlib.sha256()
    for f in files:
        if f.is_file():
            h.update(str(f.relative_to(root.parent)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


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
    rp = inp.reference_pose
    if rp is not None and rp.get("status") in ("PASS", "DECLARED"):
        q_ref, ref_ev = float(rp["q_ref"]), rp
    elif inp.synthetic:
        q_ref, ref_ev = 0.0, dict(status="SYNTHETIC", q_ref=0.0, q_ref_source="synthetic construction at q = 0")
    elif mm.get("reference_q") is not None:
        q_ref, ref_ev = float(mm["reference_q"]), dict(status="DECLARED", q_ref=float(mm["reference_q"]),
                                                       q_ref_source="mass_model.reference_q")
    else:
        return None, ("MASS_MODEL_REFERENCE_POSE_UNVERIFIED: the joint coordinate of the mass geometry's pose is "
                      "not established (declare mass_model.reference_q or supply a verified handoff anchor)"
                      + (f"; {rp.get('reasons')}" if rp else ""))
    try:
        g = GeometryMassModel.from_validated_mesh(
            inp.mass_mesh, mm["mode"], inp.joint_type, inp.axis, inp.origin, canonical_frame=mm["geometry_frame"],
            source_provenance=inp.mass_mesh_provenance or mm.get("source_provenance") or {},
            reflected_actuator_inertia=mm["reflected_actuator_inertia"],
            is_collision_proxy=bool(mm.get("mesh_is_collision_proxy")),
            allow_collision_proxy=bool(mm["allow_collision_proxy_as_mass_geometry"]), q_ref=q_ref,
            reference_evidence=ref_ev)
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
    others = {}
    th0, ics = None, None
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
        methods = cfg["fitting"]["methods"]
        sQ = [float(i.meta.get("effort_sigma") or 0.0) for i in inp.train]
        if "general_pinn" in methods:
            from .estimators.general_pinn import PINNConfig, general_pinn_fit
            log("G9 GeneralPINN (full-equation residual, autograd)")
            others["general_pinn"] = general_pinn_fit(inp.train, spec, th0, PINNConfig.from_dict(cfg["fitting"]["pinn"]))
            _write(est_dir / "general_pinn.json", dict(others["general_pinn"], label=label))
        if "general_akf" in methods:
            from .estimators.general_ekf import EKFConfig, general_akf_fit
            log("G9 GeneralAugmentedEKF (parameters in the augmented state)")
            others["general_akf"] = general_akf_fit(inp.train, spec, th0, EKFConfig.from_dict(cfg["fitting"]["ekf"]),
                                                    s_Q=float(np.median(sQ)) if sQ else None)
            _write(est_dir / "general_akf.json", dict({k: v for k, v in others["general_akf"].items()
                                                       if k != "timelines"}, label=label))
            _write(est_dir / "general_akf_timelines.json", others["general_akf"]["timelines"])
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
                         observability=rc.get("observability"),
                         declared_pulse_rows=cfg["identifiability"].get("declared_pulse_rows"))
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
    if spec is not None and (fit is not None or others):
        _write(out / "estimator_results" / "method_comparison.json",
               method_comparison(spec, fit, others, ident, struct, inp.heldout, label, train=inp.train))
    gravity_geometry_reports(out, inp, cfg, spec, mm, th0, fit, others, ics, label, log)
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
    sig_obj = dict(g9=G9_VERSION, code=code_fingerprint(), cfg={k: v for k, v in cfg.items() if not k.startswith("_")},
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


def method_comparison(spec, rollout, others, ident, struct, heldout, label, train=None) -> dict:
    """Side-by-side estimators on IDENTICAL inputs / ModelSpec / initializer. No winner by training loss."""
    from .estimators.general_ekf import _replay
    from .validation.heldout_prediction import heldout_report
    rows = {}
    entries = dict(general_rollout_fit=rollout, **others)
    unexcited = {p for p, _ in (struct or {}).get("unidentifiable", [])}
    for m, r in entries.items():
        if not r or r.get("status") not in ("ok", "not_converged"):
            rows[m] = dict(status=(r or {}).get("status", "NOT_RUN"))
            continue
        th = np.asarray(r["x"][:len(spec.names)], float)
        par = {}
        for n in spec.names:
            e = r["estimates"][n]
            sd = e.get("std")
            if m == "general_rollout_fit" and ident:
                sd = ident["parameters"][n]["std_statistical"]
            rel = (sd / abs(e["value"])) if (sd is not None and np.isfinite(sd) and e["value"] not in (0,)) else None
            at_floor = bool(e.get("at_lower_bound") or (n in ("c", "Fc", "dFs", "Fs", "alpha", "M", "mu") and
                                                        abs(float(e["value"])) < 1e-9))
            par[n] = dict(value=e["value"], std=None if at_floor else sd, rel_std=None if at_floor else rel,
                          at_bound=at_floor,
                          uncertainty_label=("AT_NUMERICAL_FLOOR: exp-parameterised value only approaches 0; the "
                                             "spread / posterior there is not an uncertainty" if at_floor else
                                             e.get("uncertainty_label", "STATISTICAL (rollout covariance)")),
                          interval95=e.get("interval95"),
                          resolved_by_own_uncertainty=bool(rel is not None and rel <= 0.10 and not at_floor
                                                           and n not in unexcited),
                          structural_flag=n in unexcited)
        rep = r.get("replay") or {}
        common = _replay(spec, th, train)["per_interaction"] if train else None
        if train:
            base = np.mean([x["rmse_over_sigma"] for x in common if "rmse_over_sigma" in x] or [np.nan])
            for n in spec.names:
                if not par[n]["at_bound"]:
                    continue
                th0_ = th.copy()
                th0_[spec.names.index(n)] = 0.0                   # truly zero (the physical model allows it)
                try:
                    z = _replay(spec, th0_, train)["per_interaction"]
                    zz = np.mean([x["rmse_over_sigma"] for x in z if "rmse_over_sigma" in x] or [np.nan])
                    if not np.isfinite(zz):
                        raise ValueError("the model at exactly zero is not simulable (e.g. zero inertia)")
                    par[n]["profile_at_zero"] = dict(replay_rmse_over_sigma_at_estimate=float(base),
                                                     replay_rmse_over_sigma_at_zero=float(zz),
                                                     consistent_with_zero=bool(abs(zz - base) <= 0.01 * max(base, 1)))
                except Exception as ex:                       # e.g. M = 0 is not a valid simulation
                    par[n]["profile_at_zero"] = dict(status="NOT_EVALUABLE", reason=str(ex))
        rows[m] = dict(status=r.get("status"), parameters=par, physical=r.get("physical"),
                       open_loop_replay=common,
                       method_own_replay=rep.get("per_interaction"),
                       training_residual_rms_sigma=r.get("residual_rms_sigma"),
                       heldout=heldout_report(spec, th, heldout) if heldout else dict(status="NOT_RUN",
                                                                                      reason="no held-out episode"),
                       runtime_s=r.get("runtime_s"), nis=r.get("nis"), order_effect=r.get("order_effect"))
    return dict(schema="forceartgs.g9.method_comparison/v1", label=label, model=spec.describe(), methods=rows,
                same_inputs="identical interactions, effort, q, q_std, ModelSpec (terms / gravity / mass model) and "
                            "integral-LS initializer for every method",
                open_loop_replay_protocol="every method's frozen theta simulated with the SAME protocol: q0 = first "
                                          "observation, v0 = line fit of the first 4 observations, native ZOH effort, "
                                          "max step 1 ms (training episodes; held-out reported separately)",
                policy="no method is selected by training loss; a parameter is reported resolved by a method only "
                       "under that method's own uncertainty AND the independent structural / excitation checks")


def gravity_geometry_reports(out, inp, cfg, spec, mm, th0, fit, others, ics, label, log=print):
    """The six gravity-projection / geometry / reference-pose audits (always written; NOT_RUN with a reason)."""
    from .validation import gravity_geometry_audit as GA
    names = ("reference_pose_alignment", "mass_com_inertia_audit", "gravity_geometry_audit",
             "gravity_joint_sensitivity", "method_gravity_parity", "impulse_balance_audit")
    rep = {n: dict(schema=f"forceartgs.g9.{n}/v1", status="NOT_RUN") for n in names}
    rep["reference_pose_alignment"] = inp.reference_pose or dict(
        schema="forceartgs.g9.reference_pose_alignment/v1", status="NOT_RUN",
        reason="no mass geometry with a pose (no mesh / no tentative proxy)" if inp.mass_mesh is None else
        "synthetic or declared mesh pose" if inp.synthetic else "no handoff anchor evaluated")
    thetas = {}
    if th0 is not None:            # the rollout initializer (LS estimate, or a documented fallback where LS violates
        thetas["integral_ls_initializer"] = np.asarray(th0, float)   # the physical sign -- see general_integral_ls.json)
    if fit is not None:
        thetas["general_rollout_fit"] = np.asarray(fit["x"], float)
    for k, r in (others or {}).items():
        if r and r.get("status") in ("ok", "not_converged"):
            thetas[k] = np.asarray(r["x"], float)
    tent = bool(inp.mass_mesh_provenance and str(inp.mass_mesh_provenance.get("status", "")).startswith("TENTATIVE"))
    g = None if inp.gravity is None else np.asarray(inp.gravity.g_world, float)
    q_all = np.concatenate([it.q_obs for it in inp.train]) if inp.train else None
    try:
        if inp.mass_mesh is not None and g is not None and mm is not None:
            dens = {k: float(spec.values(v[:len(spec.names)])["alpha"]) for k, v in thetas.items()} \
                if spec is not None and "alpha" in spec.names else None
            rep["mass_com_inertia_audit"] = GA.mass_com_inertia_audit(
                inp.mass_mesh, inp.joint_type, inp.axis, inp.origin, g, mm.q_ref, q_obs=q_all, densities=dens,
                fitted_mode=mm.props.mode, reflected=mm.reflected(), provenance=inp.mass_mesh_provenance,
                tentative=tent)
        elif inp.mass_mesh is None:
            rep["mass_com_inertia_audit"]["reason"] = "no mass geometry (no mesh, no verified tentative proxy)"
        else:
            rep["mass_com_inertia_audit"]["reason"] = "mass model rejected or no gravity field"
        if spec is not None and thetas:
            th = thetas.get("general_rollout_fit", next(iter(thetas.values())))
            rep["method_gravity_parity"] = GA.method_gravity_parity(spec, th)
        if spec is not None and mm is not None and spec.terms.gravity and thetas:
            rc = inp.record_context or {}
            rep["gravity_joint_sensitivity"] = GA.gravity_joint_sensitivity(
                spec, inp.mass_mesh, mm.props.mode, thetas, inp.train, rc.get("rec"),
                observability=rc.get("observability"), q_ref=mm.q_ref)
        elif spec is not None and spec.parameter_source == "joint_space_gravity_coupled":
            rep["gravity_joint_sensitivity"]["reason"] = ("joint_space_gravity_coupled: ablation only "
                                                          "(GRAVITY_MASS_NOT_COUPLED); no shared mass geometry")
        if spec is not None and thetas:
            rc = inp.record_context or {}
            rep["impulse_balance_audit"] = GA.impulse_balance_audit(ics, spec, thetas, rc.get("rec"), inp.train,
                                                                    inp.axis, inp.origin)
    except Exception as e:                                        # an audit failure is reported, never hidden
        log(f"[g9] gravity / geometry audit error: {type(e).__name__}: {e}")
        rep["gravity_geometry_audit"]["error"] = f"{type(e).__name__}: {e}"
    a = np.asarray(inp.axis, float) / np.linalg.norm(inp.axis)
    gsum = dict(schema="forceartgs.g9.gravity_geometry_audit/v1", label=label,
                parameter_source=None if spec is None else spec.parameter_source,
                law="G(q) = dU/dq, U = -m g.c(q); applied generalized gravity effort = -G",
                gravity=None if inp.gravity is None else dict(
                    g_B=list(inp.gravity.g_world), frame=inp.gravity.frame, verified=inp.gravity.verified,
                    provenance=inp.gravity.provenance,
                    provenance_class="DECLARED_OPERATOR_VERTICAL" if "DECLARED" in str(inp.gravity.provenance) else
                    "MEASURED" if inp.gravity.verified else "UNVERIFIED"),
                joint=dict(type=inp.joint_type, axis_B=a.tolist(),
                           origin_B=None if inp.origin is None else np.asarray(inp.origin, float).tolist(),
                           tilt_from_vertical_deg=float(np.degrees(np.arccos(min(1.0, abs(a[2]))))),
                           g_cross_a_norm=None if g is None else float(np.linalg.norm(np.cross(g, a)))),
                q_ref=None if mm is None else mm.q_ref,
                mass_mode=None if mm is None else mm.props.mode,
                reflected_inertia=None if mm is None else mm.reflected_actuator_inertia,
                force_sign=(inp.record_context or {}).get("rec", {}).get("object_side_multiplier")
                if inp.record_context else None,
                flags=(GA.TENTATIVE_FLAGS if tent else []) + (
                    ["GRAVITY_MASS_NOT_COUPLED"] if spec is not None and spec.parameter_source ==
                    "joint_space_gravity_coupled" and inp.joint_type == "revolute" else []),
                parity_status=rep["method_gravity_parity"].get("status"),
                sub_reports=list(names), source_hashes=inp.input_hashes,
                initializer_note="'integral_ls_initializer' = theta0_from_ls: the integral-LS estimate where it is "
                                 "physically consistent, else the documented fallback (e.g. alpha = 1 when LS alpha "
                                 "<= 0); it is NOT an independent estimate",
                error=rep["gravity_geometry_audit"].get("error"))
    rep["gravity_geometry_audit"] = gsum
    for n in names:
        _write(out / f"{n}.json", rep[n])
    return rep


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
