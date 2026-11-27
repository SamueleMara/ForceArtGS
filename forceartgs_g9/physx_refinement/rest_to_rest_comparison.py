"""Controlled rest-to-rest comparison (append-only <run>/dynamics/g9/rest_to_rest_compare/vNNN).

ONE immutable rest-to-rest episode per scene (visual rest plateaus -> [t0, T]; full six-axis wrench effort on every native
row, unmasked; q0 / qT plateau anchors; v0 = vT = 0 declared) is consumed by EVERY comparable method:

  B1 ENDPOINT_R2R, B2 INTEGRAL_SVD_R2R, B3 ANALYTIC_CONVOLUTION_1D          (analytics, frozen theta -> PhysX replay)
  B4..B6 rest-to-rest PINN ablations (+ the historical GeneralPINN, matched adaptation)   (frozen theta -> PhysX replay)
  B0 G10 (existing baseline starts) and B7 / B8 / B9 / union (proposed warm starts) at IDENTICAL DE budget
  B10 G10 + terminal-speed loss (V1) with the same starts / budget
  C0 historical contact-masked input + original G10 (archival behaviour)

The model is the existing simplified G10 model (configs/g9_real/physx_behavioral_inertia_damping.yaml: gravity zero,
Coulomb off, explicit viscous surrogate, armature 0) in every row; the final evaluation is always the actual PhysX
trajectory. Nothing here re-calibrates the sensor, re-tares, classifies contact, fits v0 or zeroes measured effort.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

R2R_SCHEMA = "forceartgs.g9_g10.rest_to_rest_compare/v1"
DEFAULT_R2R_CONFIG = {
    "schema": R2R_SCHEMA,
    "enabled": True,
    "experimental_facts": {"known_global_sensor_transform": True, "wrench_already_tared": True,
                           "continuous_contact": True, "initial_velocity": 0.0, "final_velocity": 0.0},
    "episode": {"selection": "full_visual_rest_to_rest", "source": "existing_joint_authority",
                "q0_source": "initial_visual_plateau", "qT_source": "final_visual_plateau",
                "rest_plateau_confidence_sigma": 3.0, "rest_plateau_min_samples": 4,
                "rest_plateau_min_duration_s": 0.12, "rest_plateau_sigma_floor": 3e-4,
                "reject_cropped_moving_start": True, "no_contact_segmentation": True, "max_row_gap_factor": 1.5},
    "input": {"source": "full_six_axis_tared_wrench", "coordinate_transform": "existing_known_global_transform",
              "native_sampling": True, "zoh": True, "additional_tare": False, "contact_mask": "none",
              "compare_historical_masked_input": True,
              # full_wrench: Q = a.tau_B + a.(r x F_B) (moment about the EE origin, the default G10 projection);
              # force_only_at_tcp: LABELLED variant Q = a.((p_TCP - o) x F_B): the measured moment is dropped and the
              # force is applied at the TCP (prismatic: identical, Q = a.F_B)
              "effort_components": "full_wrench"},
    "model": {"terms": ["inertia", "viscous"], "gravity": "zero", "coulomb": False, "bias": False, "armature": 0.0,
              "use_same_urdf_mass_geometry_as_g10": True},
    "analytical": {"endpoint_formulas": True, "double_integral_constrained_ls": True, "convolution_1d": True,
                   "svd_scale_columns": True, "return_signed_unconstrained_estimate": True,
                   "lambda_max_per_s": 50.0, "lambda_grid_points": 61, "bootstrap_draws": 200,
                   "tare_reference_rows": 25, "family_k_sd": 2.0},
    "pinn": {"compare_legacy": True, "hard_rest_to_rest": True, "strong_ablation": True, "double_integral_ablation": True,
             "hybrid_ablation": False, "soft_bc_control": False, "seeds": [0, 1, 2], "use_full_input": True,
             "initial_velocity_nuisance": False, "hidden": 32, "layers": 3, "steps_data": 3000, "steps_joint": 4000,
             "lr_net": 3e-3, "lr_theta": 1e-2, "n_colloc": 300, "n_quad": 256, "l_double_integral": 1.0,
             "l_strong": 1.0},
    "physx": {"engine": "sapien_gpu_physx", "timestep_s": 0.001,
              "use_existing_simplified_config": "configs/g9_real/physx_behavioral_inertia_damping.yaml",
              "primary_loss": "existing_position_output_error", "terminal_velocity_loss_weights": [0.0, 0.1, 1.0],
              "terminal_velocity_scale_from_rest_plateau": True, "compare_each_initializer_equal_budget": True,
              "export_candidate_preview_only": True, "profile_baseline_family": True},
    "benchmark": {"include_original_historical_results": True, "include_matched_input_baseline": True,
                  "include_input_policy_ablation": True, "same_data_same_model_same_budget": True,
                  "include_legacy_g9_matched_adaptations": True, "write_plots": True},
    "output": {"append_only": True, "preserve_original_results": True, "preserve_existing_g9_exports": True,
               "write_report": True},
}
R2R_QUICK = {"analytical": {"bootstrap_draws": 20, "lambda_grid_points": 31},
             "pinn": {"seeds": [0], "steps_data": 200, "steps_joint": 300, "n_quad": 96}}
BANNED_R2R_KEYS = ("recalibrate_sensor", "estimate_tare", "tare", "detect_contact", "contact_segments", "free_hypotheses",
                   "prefix_velocity", "v0_prefix_samples", "reset_qdot", "recenter_q", "force_scale",
                   "free_forced_classification")


class R2RConfigError(ValueError):
    pass


def _merge(base, over, path=""):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise R2RConfigError(f"unknown key {path + k}")
        if isinstance(out[k], dict):
            if not isinstance(v, dict):
                raise R2RConfigError(f"{path + k} must be a mapping")
            out[k] = _merge(out[k], v, path + k + ".")
        else:
            out[k] = v
    return out


def load_r2r_config(src=None, quick=False) -> dict:
    import yaml
    doc = {} if src is None else (copy.deepcopy(src) if isinstance(src, dict) else
                                  (yaml.safe_load(Path(src).read_text()) or {}))

    def banned(d, path=""):
        for k, v in (d or {}).items():
            if k in BANNED_R2R_KEYS:
                raise R2RConfigError(f"{path + k}: banned (no qdot reset, q recentering, hidden tare, contact "
                                     "classification or fitted force scale in the rest-to-rest comparison)")
            if isinstance(v, dict):
                banned(v, path + k + ".")
    banned(doc)
    cfg = _merge(DEFAULT_R2R_CONFIG, doc)
    if quick:
        cfg = _merge(cfg, R2R_QUICK)
    validate_r2r_config(cfg)
    return cfg


def validate_r2r_config(cfg):
    if cfg["schema"] != R2R_SCHEMA:
        raise R2RConfigError(f"schema {cfg['schema']!r} != {R2R_SCHEMA}")
    ef = cfg["experimental_facts"]
    if not (ef["known_global_sensor_transform"] is True and ef["wrench_already_tared"] is True and
            ef["continuous_contact"] is True and float(ef["initial_velocity"]) == 0.0 and
            float(ef["final_velocity"]) == 0.0):
        raise R2RConfigError("experimental_facts are given (known transform, tared, continuous contact, v0 = vT = 0)")
    fixed = {("episode", "selection"): "full_visual_rest_to_rest", ("episode", "source"): "existing_joint_authority",
             ("episode", "q0_source"): "initial_visual_plateau", ("episode", "qT_source"): "final_visual_plateau",
             ("episode", "reject_cropped_moving_start"): True, ("episode", "no_contact_segmentation"): True,
             ("input", "source"): "full_six_axis_tared_wrench",
             ("input", "coordinate_transform"): "existing_known_global_transform", ("input", "native_sampling"): True,
             ("input", "zoh"): True, ("input", "additional_tare"): False, ("input", "contact_mask"): "none",
             ("model", "gravity"): "zero", ("model", "coulomb"): False, ("model", "bias"): False,
             ("model", "use_same_urdf_mass_geometry_as_g10"): True,
             ("analytical", "svd_scale_columns"): True, ("analytical", "return_signed_unconstrained_estimate"): True,
             ("pinn", "use_full_input"): True, ("pinn", "initial_velocity_nuisance"): False,
             ("physx", "primary_loss"): "existing_position_output_error",
             ("physx", "terminal_velocity_scale_from_rest_plateau"): True,
             ("physx", "export_candidate_preview_only"): True,
             ("benchmark", "same_data_same_model_same_budget"): True,
             ("output", "append_only"): True, ("output", "preserve_original_results"): True,
             ("output", "preserve_existing_g9_exports"): True}
    for (sec, k), v in fixed.items():
        if cfg[sec][k] != v:
            raise R2RConfigError(f"{sec}.{k} must be {v!r} in the rest-to-rest comparison, got {cfg[sec][k]!r}")
    if list(cfg["model"]["terms"]) != ["inertia", "viscous"]:
        raise R2RConfigError("model.terms must be [inertia, viscous]: gravity / Coulomb would be a SEPARATE, labelled "
                             "model for which Eq. (2) does not hold without the known load terms")
    if float(cfg["model"]["armature"]) != 0.0:
        raise R2RConfigError("model.armature must be 0 (same as the simplified G10 model)")
    e = cfg["episode"]
    if not (float(e["rest_plateau_confidence_sigma"]) > 0 and int(e["rest_plateau_min_samples"]) >= 3 and
            float(e["rest_plateau_min_duration_s"]) >= 0 and float(e["rest_plateau_sigma_floor"]) >= 0 and
            float(e["max_row_gap_factor"]) >= 1.0):
        raise R2RConfigError("episode plateau settings: confidence > 0, min_samples >= 3, durations / floor >= 0")
    if cfg["input"]["effort_components"] not in ("full_wrench", "force_only_at_tcp"):
        raise R2RConfigError("input.effort_components full_wrench | force_only_at_tcp")
    w = [float(x) for x in cfg["physx"]["terminal_velocity_loss_weights"]]
    if not w or 0.0 not in w or any(x < 0 for x in w):
        raise R2RConfigError("physx.terminal_velocity_loss_weights must contain 0 (the V0 baseline) and be >= 0")
    if cfg["physx"]["engine"] not in ("sapien_gpu_physx", "sapien_cpu_diagnostic"):
        raise R2RConfigError("physx.engine sapien_gpu_physx | sapien_cpu_diagnostic")
    p = cfg["pinn"]
    if not p["seeds"] or len(set(map(int, p["seeds"]))) != len(p["seeds"]):
        raise R2RConfigError("pinn.seeds must be distinct integers")
    a = cfg["analytical"]
    if not (float(a["lambda_max_per_s"]) > 0 and int(a["lambda_grid_points"]) >= 5 and int(a["bootstrap_draws"]) >= 0):
        raise R2RConfigError("analytical: lambda_max_per_s > 0, lambda_grid_points >= 5, bootstrap_draws >= 0")
    return cfg


# =============================================================================================== engine helpers
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if p and Path(p).is_file() else None


def _huber(z, d):
    a = np.abs(z)
    return np.where(a <= d, 0.5 * z * z, d * (a - 0.5 * d))


def make_tracing(sim, pe, delta, floor, terminal=None):
    """CachingRollout that records (engine calls, best objective so far) for the convergence-vs-budget plot."""
    from .behavioral_fit import CachingRollout

    class Tracing(CachingRollout):
        def __init__(self):
            super().__init__(sim)
            self.trace, self._best, self._seen = [], math.inf, set()

        def _rec(self, p, ep, r):
            if ep.name != pe.name or isinstance(r, Exception):
                return
            q = np.asarray(r.q, float)
            if q.shape != pe.q.shape or not np.isfinite(q).all():
                return
            L = float(np.mean(_huber((q - pe.q) / np.hypot(pe.sigma_q, floor), delta)))
            if terminal and float(terminal["weight"]) > 0 and r.qdot is not None:
                L += float(terminal["weight"]) * float(_huber(np.asarray(float(r.qdot[-1]) / terminal["sigma"]), delta))
            self._best = min(self._best, L)
            self.trace.append([int(self.n_engine_calls), self._best])

        def prefetch(self, thetas, ep):
            super().prefetch(thetas, ep)
            for p in thetas:
                k = self._key(p, ep)
                if k not in self._seen and k in self._c:
                    self._seen.add(k)
                    self._rec(p, ep, self._c[k])

        def __call__(self, p, ep):
            k = self._key(p, ep)
            r = super().__call__(p, ep)
            if k not in self._seen:
                self._seen.add(k)
                self._rec(p, ep, r)
            return r
    return Tracing()


def physx_metrics(res, pe, ep, floor, sigma_vT, k=3.0) -> dict:
    from .behavioral_fit import replay_metrics
    m = replay_metrics(pe, res, floor)
    qd = None if res.qdot is None else np.asarray(res.qdot, float)
    q = np.asarray(res.q, float)
    vT = None if qd is None else float(qd[-1])
    settle = None
    if qd is not None:
        above = np.flatnonzero(np.abs(qd) > k * sigma_vT)
        settle = (float(pe.t_s[0]) if len(above) == 0 else
                  float(pe.t_s[above[-1] + 1]) if above[-1] + 1 < len(qd) else None)
    m.update(v_sim_T=vT, abs_v_sim_T_over_sigma=None if vT is None else abs(vT) / sigma_vT,
             q_sim_T_minus_qT=float(q[-1] - ep.qT), time_to_settle_s=None if settle is None else settle - ep.t0,
             settled_before_T=settle is not None,
             settle_rule=f"first sample after which |qdot_PX| <= {k} sigma_vT on the episode clock (None: not settled "
                         "by T)", observed_plateau_s=float((ep.endpoint_evidence.get("terminal") or {})
                                                           .get("plateau_duration_s") or 0.0))
    return m


# =============================================================================================== main entry
def run(run_root, scene, *, config=None, robot_archive=None, contact_segments=None, out_root=None,
        declared_joint_type=None, declaration_note=None, quick=False, python=None, authority_dir=None,
        log=print) -> dict:
    from ..estimators import rest_to_rest as RR
    from ..forcetwin.pipeline import _clean, _git, _mass_model, _write, next_version
    from .behavioral_contracts import load_wrench_config, wrench_to_fit_config
    from .real_data import reference_urdf
    t_start = time.time()
    cfg = load_r2r_config(config, quick=quick)
    if not cfg["enabled"]:
        raise SystemExit("rest-to-rest config has enabled: false")
    wpath = cfg["physx"]["use_existing_simplified_config"]
    wcfg = load_wrench_config(wpath, quick=quick)
    if wcfg["physics"]["gravity"] != "zero" or wcfg["physics"]["coulomb_friction"] != "off":
        raise R2RConfigError(f"{wpath}: not the inertia + damping model (gravity zero, Coulomb off)")
    if (wcfg["physics"]["engine"] != cfg["physx"]["engine"] or
            float(wcfg["physics"]["timestep_s"]) != float(cfg["physx"]["timestep_s"])):
        raise R2RConfigError(f"physx.engine / timestep_s must equal {wpath} (identical physics for every row)")
    if python:
        wcfg["physics"]["python"] = python
    fcfg = wrench_to_fit_config(wcfg)
    out = next_version(Path(out_root) if out_root else Path(run_root) / "dynamics" / "g9" / "rest_to_rest_compare")
    git = _git()
    status = dict(schema="forceartgs.g9_g10.rest_to_rest_compare_status/v1", scene=Path(scene).name, out_dir=str(out),
                  git=git, quick=bool(quick), claim="FIT_TO_DEMONSTRATION comparison (no sim-to-real transfer claim)")
    import yaml
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(_clean(dict(
        rest_to_rest_compare=cfg, simplified_g10_config=str(wpath), simplified_g10_resolved={
            k: v for k, v in wcfg.items() if not k.startswith("_")}, run=str(run_root), scene=str(scene),
        robot_archive=robot_archive, contact_segments=str(contact_segments) if contact_segments else None,
        declared_joint_type=declared_joint_type, declaration_note=declaration_note, quick=bool(quick), git=git)),
        sort_keys=False))
    hashes = dict(files=dict(r2r_config=dict(path=str(config) if isinstance(config, (str, Path)) else None,
                                             sha256=_sha(config) if isinstance(config, (str, Path)) else None),
                             simplified_g10_config=dict(path=str(wpath), sha256=_sha(wpath))))

    def finish(st, reason, **extra):
        status.update(status=st, reason=reason, runtime_s=time.time() - t_start, **extra)
        _write(out / "source_hashes.json", hashes)
        _write(out / "STATUS.json", status)
        _write(out / "manifest.json", dict(status=st, git=git, out=str(out), quick=bool(quick), hashes=hashes,
                                           engine=cfg["physx"]["engine"], model="inertia + viscous damping (gravity "
                                           "zero, Coulomb off, armature 0)", budgets=dict(
                                               de_maxiter=wcfg["fitting"]["maxiter"], de_popsize=wcfg["fitting"]["popsize"],
                                               local_max_nfev=wcfg["fitting"]["local_max_nfev"],
                                               pinn=dict(steps_data=cfg["pinn"]["steps_data"],
                                                         steps_joint=cfg["pinn"]["steps_joint"],
                                                         seeds=cfg["pinn"]["seeds"]))))
        log(f"[r2r] {Path(scene).name}: {st} ({reason})")
        return status

    # ------------------------------------------------------------------ inputs: the SAME sources as G10
    prep = prepare_inputs(run_root, scene, wcfg, robot_archive=robot_archive, declared_joint_type=declared_joint_type,
                          declaration_note=declaration_note, authority_dir=authority_dir,
                          effort_components=cfg["input"]["effort_components"], log=log)
    if prep["status"] != "OK":
        return finish(prep["status"], prep["reason"])
    fr, contract, inp, kin, arc, dcfg_p = (prep[k] for k in ("fr", "contract", "inp", "kin", "arc", "dcfg_p"))
    jt = fr.joint_type
    fr.save_npz(out / "full_wrench_joint_effort.npz")
    hashes.update(input_hashes=inp.input_hashes, full_wrench_record_sha256=fr.sha256())
    hashes["files"].update(robot_archive=dict(path=str(arc), sha256=contract["robot_archive_sha256"]),
                           robot_tool_json=dict(path=contract["robot_tool_json"], sha256=contract["robot_tool_sha256"],
                                                tool_override=contract.get("tool_override")),
                           joint_authority=dict(path=str(kin.get("authority_dir")), id=kin.get("authority_id")),
                           dynamics_config=dict(path=str(dcfg_p), sha256=_sha(dcfg_p)))
    built = episode_from_record(fr, cfg)
    ev = built["evidence"].to_dict()
    ev["visual_authority"] = dict(first_source_frame=int(fr.source_q_frame[0]), last_source_frame=int(fr.source_q_frame[-1]),
                                  n_valid=int(len(fr.q)), recording_rows=[int(fr.source_row[0]), int(fr.source_row[-1])],
                                  note="the joint authority covers only the reconstruction clip of the recording")
    _write(out / "rest_endpoint_evidence.json", ev)
    plots = out / "figures"
    if cfg["benchmark"]["write_plots"]:
        try:
            from . import rest_to_rest_report as RP
            RP.plot_rest_plateaus(plots, fr, ev)
        except Exception as ex:
            log(f"[r2r] rest plateau plot failed: {type(ex).__name__}: {ex}")
    if built["status"] != "OK":
        return finish(built["status"], built["reason"], endpoint_evidence=ev)
    ep = built["episode"]
    ep.save_npz(out / "rest_to_rest_episode.npz")
    hashes["episode_sha256"] = ep.sha256()
    log(f"[r2r] {ep.scene}: {built['reason']}")
    sigma_vT = RR.terminal_speed_sigma(ep)
    res = dict(episode=ep.describe(), sigma_vT=sigma_vT, flags=built.get("flags", []))

    # ------------------------------------------------------------------ analytics (B1-B3) + diagnostics
    analytics = run_analytics(ep, cfg, log=log)
    _write(out / "endpoint_analytic.json", analytics["endpoint"])
    _write(out / "integral_svd.json", dict(analytics["svd"], bias_ablation=analytics["bias"],
                                           diagnostics=analytics["diagnostics"]))
    _write(out / "analytical_convolution_1d.json", analytics["conv"])
    theta0 = analytics["theta0"]

    # ------------------------------------------------------------------ PINN ablations (B4-B6 + legacy)
    pinn = run_pinns(ep, cfg, theta0, log=log)
    _write(out / "pinn_ablation.json", pinn)

    # ------------------------------------------------------------------ historical inventory + legacy G9 (M)
    hist = historical_inventory(run_root) if cfg["benchmark"]["include_original_historical_results"] else \
        dict(status="DISABLED")
    legacy = run_legacy_g9(ep, theta0, log=log) if cfg["benchmark"]["include_legacy_g9_matched_adaptations"] else {}
    hist["matched_adaptations_M"] = legacy
    _write(out / "historical_baseline_inventory.json", hist)

    # ------------------------------------------------------------------ PhysX
    mm = _mass_model(inp, wcfg["physics"]["mass_mode"])
    urdf = out / "reference.urdf"
    info = reference_urdf(inp, SimpleNamespace(joint_type=jt, axis_B=fr.axis_B), mm, urdf)
    if info is None:
        return finish("MASS_GEOMETRY_UNAVAILABLE", "no validated mass geometry / reference pose for the PhysX model",
                      analytics_only=True)
    _write(out / "urdf_info.json", info)
    hashes["files"]["urdf"] = dict(path=str(urdf), sha256=_sha(urdf))
    segs = None
    if cfg["input"]["compare_historical_masked_input"] and cfg["benchmark"]["include_input_policy_ablation"]:
        if contact_segments and Path(contact_segments).is_file():
            from .behavioral_adapter import load_interaction_segments
            segs = load_interaction_segments(contact_segments)
            hashes["files"]["historical_contact_segments"] = dict(path=str(contact_segments),
                                                                   sha256=_sha(contact_segments), segments=segs,
                                                                   use="C0 archival input policy ONLY")
    try:
        px = run_physx(ep, fr, cfg, wcfg, fcfg, info, urdf, analytics, pinn, legacy, sigma_vT, segs, log=log)
    except RuntimeError as ex:
        return finish("ENGINE_FAILED", f"{type(ex).__name__}: {ex}")
    _write(out / "physx_comparison.json", {k: v for k, v in px.items() if k not in ("trajectories", "traces")})
    _write(out / "equal_budget_convergence.json", px["traces"])
    _write(out / "candidate_family.json", px.get("family") or dict(status="NOT_EVALUATED"))
    from . import rest_to_rest_report as RP
    rows = RP.summary_rows(ep, analytics, pinn, px, info, hashes, cfg, legacy)
    if cfg["output"]["write_report"]:
        RP.write_summary(out, rows, ep, ev, analytics, pinn, px, sigma_vT, hist, status=dict(git=git, quick=quick))
    if cfg["benchmark"]["write_plots"]:
        try:
            RP.plots(plots, ep, fr, analytics, px, segs)
        except Exception as ex:                                   # plotting never hides the numbers
            log(f"[r2r] plotting failed: {type(ex).__name__}: {ex}")
    best = min((r for r in rows if r.get("physx_rmse_over_sigma") is not None),
               key=lambda r: r["physx_rmse_over_sigma"], default=None)
    return finish("COMPARED", f"{len(rows)} rows; best PhysX rmse/sigma {best['physx_rmse_over_sigma']:.3g} ({best['id']})"
                  if best else "no PhysX row", episode=res["episode"], flags=res["flags"],
                  best_row=None if best is None else {k: best.get(k) for k in ("id", "method", "M", "c",
                                                                              "physx_rmse_over_sigma", "v_sim_T")},
                  labels=dict(FIT_TO_DEMONSTRATION=True, PARAMETERS_UNIQUE=None,
                              INDEPENDENT_REAL_ACTION_VALIDATED=False))



def load_extended_authority(authority_dir, accepted):
    """An extended authority (estimotion extend-states) is used ONLY if its joint is the accepted one exactly (type,
    axis, hinge point) and it descends from it: the joint stays frozen, only q gets more frames."""
    from estimotion_2part.joint_discovery.integration_contract import JointAuthority
    ext = JointAuthority.load(authority_dir)
    same = (ext.joint_type == accepted.joint_type and np.array_equal(ext.axis_world, accepted.axis_world) and
            ((ext.origin_world is None and accepted.origin_world is None) or
             (ext.origin_world is not None and accepted.origin_world is not None and
              np.array_equal(ext.origin_world, accepted.origin_world))))
    if not same:
        raise R2RConfigError(f"{authority_dir}: its joint differs from the accepted authority (frozen joint required)")
    if ext.parent_authority_id != accepted.authority_id():
        raise R2RConfigError(f"{authority_dir}: parent {ext.parent_authority_id} is not the accepted authority "
                             f"{accepted.authority_id()}")
    return ext


def apply_force_only(fr, robot_archive):
    """LABELLED input variant: Q_F = a.((p_TCP - o) x F_B) (revolute) / a.F_B (prismatic) on the SAME rows, the same
    known transform, object-side sign and tare as the full wrench; only the measured moment is dropped. sd(Q_F) from
    the force noise alone through the same linear projection."""
    from ..wrench_input.full_wrench_projection import effort_sigma
    z = np.load(robot_archive, allow_pickle=False)
    rows = np.asarray(z["source_row"], int)
    if not np.array_equal(rows, np.asarray(fr.source_row, int)):
        raise R2RConfigError("robot archive rows differ from the full-wrench record rows")
    a = np.asarray(fr.axis_B, float)
    a = a / np.linalg.norm(a)
    F = np.asarray(fr.F_B, float)
    if fr.joint_type == "prismatic":
        Q = F @ a
        sig = effort_sigma(fr.R_BS, fr.p_ref_B, fr.noise["force_sigma_N"], [0.0] * 3, "prismatic", a)
    else:
        p_tcp = np.asarray(z["T_base_tcp"], float)[:, :3, 3]
        o = np.asarray(fr.origin_B, float)
        Q = np.einsum("j,nj->n", a, np.cross(p_tcp - o, F))
        sig = effort_sigma(fr.R_BS, p_tcp, fr.noise["force_sigma_N"], [0.0] * 3, "revolute", a, o)
    full = np.asarray(fr.Q_full, float)
    fr.noise["effort_components"] = dict(
        mode="force_only_at_tcp", note=f"Q_F vs Q_full: impulse {float(np.sum(Q[:-1] * np.diff(fr.t_s))):.4g} vs "
        f"{float(np.sum(full[:-1] * np.diff(fr.t_s))):.4g}, max |Q| {float(np.max(np.abs(Q))):.3g} vs "
        f"{float(np.max(np.abs(full))):.3g}", rule="Q = a.((p_TCP - o) x F_B); the measured moment is NOT used")
    fr.Q_full_original = full
    fr.Q_full = Q
    fr.Q_sigma = np.maximum(sig, 1e-9)
    return fr


def prepare_inputs(run_root, scene, wcfg, *, robot_archive=None, declared_joint_type=None, declaration_note=None,
                   authority_dir=None, effort_components="full_wrench", log=print) -> dict:
    """The SAME sources as the G10 wrench path: joint authority, robot archive, known contract, FR.build (every native
    row, no mask, no extra tare). No legacy / alternative-origin side products are needed here."""
    from ..adapters.forceartgs_g9 import _q_noise, accepted_kinematics, build_inputs
    from ..config import load_dynamics_config
    from ..wrench_input import full_record as FR
    from ..wrench_input.full_wrench_projection import WrenchContractError
    dcfg_p = wcfg["inputs"]["dynamics_config"]
    dcfg = load_dynamics_config(dcfg_p)
    dcfg["pipeline"]["allow_diagnostic_kinematics"] = True
    kin = accepted_kinematics(run_root, True, declared_joint_type, declaration_note)
    if kin.get("kind") is None:
        return dict(status="DATA_UNAVAILABLE", reason=f"no joint authority under {run_root} ({kin.get('reason')})")
    auth = kin["authority"]
    if authority_dir:
        auth = load_extended_authority(authority_dir, auth)
        log(f"[r2r] extended authority {authority_dir}: {len(auth.q)} frames (parent {auth.parent_authority_id[:12]})")
    inp = build_inputs(run_root, scene, dcfg, robot_archive=robot_archive, contact_segments=None,
                       declared_joint_type=declared_joint_type, declaration_note=declaration_note, log=log)
    rc = inp.record_context or {}
    arc = robot_archive or ((rc.get("rec") or {}).get("sources") or {}).get("robot_archive")
    if arc is None:
        return dict(status="DATA_UNAVAILABLE", reason="no robot archive (TCP pose + six-axis wrench): pass "
                                                      "--robot-archive")
    fx = Path(run_root).parent.parent / "contact_annotations" / Path(scene).name / "contact_annotations.json"
    try:
        contract = FR.load_contract(scene, arc, fx if fx.is_file() else None)
    except (WrenchContractError, FileNotFoundError) as e:
        return dict(status="DATA_UNAVAILABLE", reason=f"{e}")
    m = np.asarray(auth.q_valid, bool)
    qs = auth.q_sigma[m] if auth.q_sigma is not None else _q_noise(auth, m)
    qs = np.asarray(qs, float).copy()
    pos = qs[np.isfinite(qs) & (qs > 0)]
    qs[~np.isfinite(qs) | (qs <= 0)] = float(np.median(pos)) if len(pos) else 1e-3
    fr = FR.build(contract, scene=Path(scene).name, joint_type=auth.joint_type, axis=auth.axis_world,
                  origin=auth.origin_world, q_src=np.asarray(auth.source_frame_index)[m], q=np.asarray(auth.q, float)[m],
                  q_sigma=qs, legacy=None, baseline_s=float(wcfg["input"]["noise_baseline_s"]),
                  compare_alternative_origin=False)
    if effort_components == "force_only_at_tcp":
        apply_force_only(fr, arc)
        log(f"[r2r] effort: FORCE ONLY at the TCP (measured moment dropped), {fr.noise['effort_components']['note']}")
    return dict(status="OK", fr=fr, contract=contract, inp=inp, kin=dict(kin, authority=auth,
                authority_dir=str(authority_dir) if authority_dir else kin.get("authority_dir"),
                authority_id=auth.authority_id()), arc=arc, dcfg_p=dcfg_p)


def episode_from_record(fr, cfg) -> dict:
    from ..estimators import rest_to_rest as RR
    e = cfg["episode"]
    return RR.from_full_wrench_record(fr, k_sigma=float(e["rest_plateau_confidence_sigma"]),
                                      min_samples=int(e["rest_plateau_min_samples"]),
                                      min_duration_s=float(e["rest_plateau_min_duration_s"]),
                                      sigma_floor=float(e["rest_plateau_sigma_floor"]),
                                      max_gap_factor=float(e["max_row_gap_factor"]))


def episode_only(run_root, scene, *, config=None, robot_archive=None, declared_joint_type=None, declaration_note=None,
                 authority_dir=None, log=print) -> dict:
    """Inputs + rest-to-rest endpoint selection only (no PhysX): the per-scene coverage check."""
    from .behavioral_contracts import load_wrench_config
    cfg = load_r2r_config(config)
    wcfg = load_wrench_config(cfg["physx"]["use_existing_simplified_config"])
    prep = prepare_inputs(run_root, scene, wcfg, robot_archive=robot_archive, declared_joint_type=declared_joint_type,
                          declaration_note=declaration_note, authority_dir=authority_dir,
                          effort_components=cfg["input"]["effort_components"], log=log)
    if prep["status"] != "OK":
        return dict(status=prep["status"], reason=prep["reason"])
    b = episode_from_record(prep["fr"], cfg)
    fr = prep["fr"]
    return dict(status=b["status"], reason=b["reason"], episode=b["episode"], evidence=b["evidence"].to_dict(),
                visual_frames=[int(fr.source_q_frame[0]), int(fr.source_q_frame[-1])],
                recording_rows=[int(fr.source_row[0]), int(fr.source_row[-1])], fr=fr)


# =============================================================================================== stages
def run_analytics(ep, cfg, log=print) -> dict:
    from ..estimators import rest_to_rest_convolution as RC
    from ..estimators import rest_to_rest_diagnostics as RD
    from ..estimators import rest_to_rest_integral as RI
    a = cfg["analytical"]
    k = float(cfg["episode"]["rest_plateau_confidence_sigma"])
    tr = int(a["tare_reference_rows"])
    draws = RI.noise_draws(ep, int(a["bootstrap_draws"]), seed=0, tare_rows=tr) if int(a["bootstrap_draws"]) else None
    endpoint = RI.endpoint_r2r(ep, k_sigma=k, tare_rows=tr, draws=draws) if a["endpoint_formulas"] else \
        dict(status="DISABLED")
    svd = RI.integral_svd_r2r(ep, tare_rows=tr, draws=draws, family_k=float(a["family_k_sd"])) \
        if a["double_integral_constrained_ls"] else dict(status="DISABLED")
    conv = RC.convolution_1d(ep, lam_max=float(a["lambda_max_per_s"]), n_grid=int(a["lambda_grid_points"])) \
        if a["convolution_1d"] else dict(status="DISABLED")
    diag = dict(work_model_free=RD.work_model_free(ep, k), energy=RD.energy_audit(ep, k=k, n_boot=min(50, len(draws or [])
                                                                                                    or 50)),
                stationary_load=ep.endpoint_evidence.get("stationary_load"))
    if conv.get("status") == "OK":
        diag["orthogonality_at_convolution_fit"] = RD.strong_form_orthogonality_check(ep, conv["M"], conv["c"])
    bias = RI.bias_ablation(ep, tare_rows=tr)
    if svd.get("method") and conv.get("status") == "OK":
        svd["output_error_crosscheck"] = RI.output_error_crosscheck(ep, svd["estimate"]["M"], svd["estimate"]["c"], conv)
        if svd["output_error_crosscheck"]["status"] != "CONSISTENT":
            svd["parameters_unique"] = False                      # a precise but wrong solution is not "unique"
            svd["flags"] = sorted(set(svd.get("flags", [])) | {svd["output_error_crosscheck"]["status"]})
    th = None
    est = (svd.get("estimate") or {}) if svd.get("method") else {}
    if est.get("M", 0) > 0 and np.isfinite(est.get("M", np.nan)):
        th = [est["M"], max(est["c"], 1e-6)]
    elif conv.get("status") == "OK":
        th = [conv["M"], max(conv["c"], 1e-6)]
    else:
        th = [1.0, 1.0]
    log(f"[r2r] analytics: endpoint {endpoint.get('status')} c_end {endpoint.get('c_end')} M_end {endpoint.get('M_end')}; "
        f"SVD M {est.get('M')} c {est.get('c')}; convolution M {conv.get('M')} c {conv.get('c')}")
    return dict(endpoint=endpoint, svd=svd, conv=conv, diagnostics=diag, bias=bias, theta0=th,
                theta0_source="INTEGRAL_SVD_R2R whitened (else the convolution, else (1, 1))")


def pinn_config(cfg):
    from ..estimators.general_pinn import PINNConfig
    p = cfg["pinn"]
    return PINNConfig(hidden=int(p["hidden"]), layers=int(p["layers"]), n_seeds=len(p["seeds"]),
                      steps_data=int(p["steps_data"]), steps_joint=int(p["steps_joint"]), lr_net=float(p["lr_net"]),
                      lr_theta=float(p["lr_theta"]), n_colloc=int(p["n_colloc"]), n_quad=int(p["n_quad"]),
                      l_double_integral=float(p["l_double_integral"]), l_phys=float(p["l_strong"]))


def run_pinns(ep, cfg, theta0, log=print) -> dict:
    from ..estimators import rest_to_rest_pinn as RPN
    p = cfg["pinn"]
    want = []
    if p["compare_legacy"]:
        want.append("LEGACY_PINN_STRONG")
    if p["hard_rest_to_rest"]:
        if p["strong_ablation"]:
            want.append("PINN_HARD_BC_STRONG")
        if p["double_integral_ablation"]:
            want.append("PINN_HARD_BC_DOUBLE_INTEGRAL")
        if p["hybrid_ablation"]:
            want.append("PINN_HARD_BC_HYBRID")
    if p["soft_bc_control"]:
        want.append("PINN_SOFT_BC_DOUBLE_INTEGRAL")
    pc = pinn_config(cfg)
    out = dict(theta0=list(map(float, theta0)), common=dict(seeds=list(map(int, p["seeds"])), steps_data=pc.steps_data,
                                                             steps_joint=pc.steps_joint, hidden=pc.hidden,
                                                             layers=pc.layers),
               variants={})
    for v in want:
        try:
            out["variants"][v] = RPN.fit_variant(v, ep, pc, theta0, seeds=tuple(map(int, p["seeds"])), log=log)
        except Exception as ex:                                     # recorded, never hidden
            out["variants"][v] = dict(variant=v, status="FAILED", reason=f"{type(ex).__name__}: {ex}")
        log(f"[r2r] PINN {v}: {out['variants'][v].get('status')} M {out['variants'][v].get('M')} "
            f"c {out['variants'][v].get('c')}")
    return out


def run_legacy_g9(ep, theta0, log=print) -> dict:
    """(M) matched-data two-term adaptations of the historical G9 LS / rollout / AKF: same samples and unmasked native
    effort, two-term spec. Where an estimator's core definition differs (moving segments + per-segment nuisance for the
    LS, a fitted 4-sample v0 for the AKF) it is run UNCHANGED and the difference is stated, not hidden."""
    from ..estimators.general_ekf import general_akf_fit
    from ..estimators.general_integral_ls import general_integral_ls
    from ..estimators.model_spec import G9Interaction, inertia_viscous_spec
    from ..estimators.rest_to_rest_pinn import open_loop
    spec = inertia_viscous_spec(ep.joint_type)
    it = G9Interaction(name=f"{ep.scene}_r2r", joint_type=ep.joint_type, t_obs=np.array(ep.t_q_s), q_obs=np.array(ep.q_obs),
                       q_std=np.array(ep.q_sigma), effort=ep.effort,
                       meta=dict(effort_sigma=float(np.sqrt(np.mean(np.asarray(ep.force_sigma) ** 2)))
                                 if ep.force_sigma is not None else 0.0))
    out = {}
    try:
        ls = general_integral_ls([it], spec)
        if ls.get("status") == "OK":
            M, c = ls["estimates"]["M"]["value"], ls["estimates"]["c"]["value"]
            out["general_integral_ls"] = dict(status="OK", M=M, c=c, open_loop=open_loop(ep, M, c) if M > 0 else None,
                                              segments=ls["segments"], physically_consistent=ls["physically_consistent"],
                                              definition="moving-segment double integral with per-segment M v_a and "
                                                         "offset nuisances (historical definition, unchanged)")
        else:
            out["general_integral_ls"] = dict(status=ls.get("status"), reason=ls.get("reason"))
    except Exception as ex:
        out["general_integral_ls"] = dict(status="INCOMPATIBLE", reason=f"{type(ex).__name__}: {ex}")
    try:
        from ..estimators.general_constrained import general_rollout_fit
        it0 = G9Interaction(name=it.name, joint_type=it.joint_type, t_obs=it.t_obs, q_obs=it.q_obs, q_std=it.q_std,
                            effort=it.effort, meta=it.meta,
                            initial_state=dict(t_s=ep.t0, q=ep.q0, qdot=0.0, source="REST_TO_REST_DECLARED"))
        rf = general_rollout_fit([it0], spec, np.asarray(theta0, float), n_starts=2, max_step=1e-3)
        x = rf.get("x") or [rf["estimates"][n]["value"] for n in spec.names]
        out["general_rollout_fit"] = dict(status=rf.get("status", "OK"), M=float(x[0]), c=float(x[1]),
                                          open_loop=open_loop(ep, x[0], x[1]),
                                          definition="hybrid-simulator rollout fit from the DECLARED (q0, v0 = 0)")
    except Exception as ex:
        out["general_rollout_fit"] = dict(status="INCOMPATIBLE", reason=f"{type(ex).__name__}: {ex}")
    try:
        ak = general_akf_fit([it], spec, np.asarray(theta0, float))
        x = ak.get("x") or [ak["estimates"][n]["value"] for n in spec.names]
        out["general_akf"] = dict(status=ak.get("status", "OK"), M=float(x[0]), c=float(x[1]),
                                  open_loop=open_loop(ep, x[0], x[1]),
                                  definition="augmented EKF with its historical 4-sample v0 initialisation (unchanged); "
                                             "frozen theta replayed from (q0, v0 = 0)")
    except Exception as ex:
        out["general_akf"] = dict(status="INCOMPATIBLE", reason=f"{type(ex).__name__}: {ex}")
    for k, v in out.items():
        log(f"[r2r] legacy {k}: {v.get('status')} M {v.get('M')} c {v.get('c')}")
    return out


def historical_inventory(run_root) -> dict:
    """Existing outputs under <run>/dynamics/g9 (read only). Different input / window / model / budget than the
    matched comparison: HISTORICAL_DIFFERENT_INPUT_OR_BUDGET, never mixed into the controlled table."""
    root = Path(run_root) / "dynamics" / "g9"
    items = []
    for sub in ("io_identification", "physx_behavioral_wrench", "physx_behavioral", "physx_refinement",
                "rest_to_rest_compare"):
        for v in sorted((root / sub).glob("v[0-9][0-9][0-9]")):
            st = v / "STATUS.json"
            d = json.loads(st.read_text()) if st.is_file() else {}
            items.append(dict(kind=sub, path=str(v), status=d.get("status"), reason=str(d.get("reason"))[:200],
                              quick=d.get("quick"), parameters=d.get("parameters"),
                              training_rmse_over_sigma=d.get("training_rmse_over_sigma"),
                              label="HISTORICAL_DIFFERENT_INPUT_OR_BUDGET"))
    for v in sorted(root.glob("v[0-9][0-9][0-9]")):
        mf = v / "manifest.json"
        d = json.loads(mf.read_text()) if mf.is_file() else {}
        items.append(dict(kind="g9_dynamics_stage", path=str(v), status=d.get("status"),
                          label="HISTORICAL_DIFFERENT_INPUT_OR_BUDGET"))
    return dict(root=str(root), items=items, n=len(items),
                archival_reference=dict(source="docs/G10_FULL_WRENCH_BEHAVIORAL_IMPLEMENTATION_REPORT.md",
                                        note="quick FULL-model G10 numbers (fridge / suitcase / black box): tiny budget, "
                                             "different model and masked input -- archival only"))


def run_physx(ep, fr, cfg, wcfg, fcfg, info, urdf, analytics, pinn, legacy, sigma_vT, segs, log=print) -> dict:
    from ..estimators.rest_to_rest_integral import map_to_physx
    from . import behavioral_fit as BF
    from .behavioral_real import starts_for
    from .sapien_diagnostic import IsolatedSapienCpu, IsolatedSapienGpu
    floor = float(fcfg["loss"]["numeric_floor_q"])
    delta = float(fcfg["loss"]["huber_delta"])
    s_vT = float(sigma_vT["sigma_vT"])
    pe = ep.to_physx_episode()
    bounds = BF.bounds_for(fcfg["model"], info["axis_inertia"], float(np.max(np.abs(ep.force_Q))) or 1.0)
    gpu = wcfg["physics"]["engine"] == "sapien_gpu_physx"
    sim_kw = dict(moving_link=info["moving_link"], joint_name=info["joint_name"], dt=float(wcfg["physics"]["timestep_s"]),
                  friction_mode=wcfg["physics"]["friction_mode"], python=wcfg["physics"]["python"])
    if gpu:
        sim_kw["device"] = wcfg["physics"]["device"]
    sim = (IsolatedSapienGpu if gpu else IsolatedSapienCpu)(urdf, gravity=[0.0, 0.0, 0.0], **sim_kw)
    out = dict(bounds={k: list(v) for k, v in bounds.items()}, sigma_vT=sigma_vT, frozen={}, fits={}, warm_starts={},
               trajectories={}, traces={})
    try:
        caps = sim.capabilities()
        # -------------------------------------------- frozen theta of every non-PhysX method -> the same PhysX replay
        cands = {"B1_ENDPOINT_R2R": (analytics["endpoint"].get("M_end"), analytics["endpoint"].get("c_end")),
                 "B2_INTEGRAL_SVD_R2R": ((analytics["svd"].get("estimate") or {}).get("M"),
                                         (analytics["svd"].get("estimate") or {}).get("c")),
                 "B3_ANALYTIC_CONVOLUTION_1D": (analytics["conv"].get("M"), analytics["conv"].get("c"))}
        ids = {"PINN_HARD_BC_STRONG": "B4", "PINN_HARD_BC_DOUBLE_INTEGRAL": "B5", "PINN_HARD_BC_HYBRID": "B6",
               "PINN_SOFT_BC_DOUBLE_INTEGRAL": "B5s", "LEGACY_PINN_STRONG": "H_PINN"}
        for v, r in (pinn.get("variants") or {}).items():
            cands[f"{ids.get(v, v)}_{v}"] = (r.get("M"), r.get("c"))
        for k, r in (legacy or {}).items():
            cands[f"M_{k}"] = (r.get("M"), r.get("c"))
        maps = {lab: map_to_physx(M, c, info, bounds, label=lab) for lab, (M, c) in cands.items()}
        roll = BF.CachingRollout(sim)
        ok = {lab: m for lab, m in maps.items() if m["status"].startswith("OK")}
        roll.prefetch([m["physical"] for m in ok.values()], pe)
        for lab, mp in maps.items():
            row = dict(mapping={k: v for k, v in mp.items() if k != "physical"})
            if mp["status"].startswith("OK"):
                try:
                    r = roll(mp["physical"], pe)
                    row.update(metrics=physx_metrics(r, pe, ep, floor, s_vT),
                               objective_V0=float(np.mean(_huber((np.asarray(r.q) - pe.q) /
                                                                np.hypot(pe.sigma_q, floor), delta))))
                    out["trajectories"][lab] = dict(q=np.asarray(r.q).tolist(),
                                                    qdot=None if r.qdot is None else np.asarray(r.qdot).tolist())
                except Exception as ex:
                    row.update(status="ROLLOUT_FAILED", error=f"{type(ex).__name__}: {ex}")
            out["frozen"][lab] = row
        out["warm_starts"] = {lab: {k: v for k, v in mp.items() if k != "physical"} for lab, mp in maps.items()}
        # -------------------------------------------- G10 fits at IDENTICAL budget (C1 episode)
        base_starts = starts_for(bounds, info["axis_inertia"], None)
        prop = {lab: maps[lab]["physical"] for lab in ("B2_INTEGRAL_SVD_R2R", "B3_ANALYTIC_CONVOLUTION_1D",
                                                       "B5_PINN_HARD_BC_DOUBLE_INTEGRAL")
                if lab in maps and maps[lab]["status"].startswith("OK")}
        fit_specs = []
        if cfg["benchmark"]["include_matched_input_baseline"]:
            fit_specs.append(("B0_G10_BASELINE_STARTS", base_starts, 0.0))
        if cfg["physx"]["compare_each_initializer_equal_budget"]:
            for lab, short in (("B2_INTEGRAL_SVD_R2R", "B7_G10_SVD_START"),
                               ("B3_ANALYTIC_CONVOLUTION_1D", "B8_G10_CONVOLUTION_START"),
                               ("B5_PINN_HARD_BC_DOUBLE_INTEGRAL", "B9_G10_INTEGRAL_PINN_START")):
                if lab in prop:
                    fit_specs.append((short, {lab: prop[lab]}, 0.0))
            if len(prop) > 1:
                fit_specs.append(("B789_G10_UNION_PROPOSED_STARTS", dict(prop), 0.0))
        for w in cfg["physx"]["terminal_velocity_loss_weights"]:
            if float(w) > 0:
                fit_specs.append((f"B10_G10_TERMINAL_LOSS_w{float(w):g}", base_starts, float(w)))
        for lab, starts, w in fit_specs:
            out["fits"][lab] = _one_fit(sim, ep, pe, fcfg, bounds, starts, w, s_vT, floor, delta, out, lab,
                                        profile=(lab == "B0_G10_BASELINE_STARTS" and
                                                 cfg["physx"]["profile_baseline_family"]), log=log)
        if "B0_G10_BASELINE_STARTS" in out["fits"] and "_family" in out["fits"]["B0_G10_BASELINE_STARTS"]:
            out["family"] = out["fits"]["B0_G10_BASELINE_STARTS"].pop("_family")
        # -------------------------------------------- C0: historical masked input (archival), same budget
        if segs is not None:
            out["C0"] = _c0(sim, ep, fr, wcfg, fcfg, bounds, base_starts, segs, s_vT, floor, delta, out, log=log)
        else:
            out["C0"] = dict(status="SKIPPED", reason="no contact-segments file passed (C0 = historical masked input "
                                                      "needs the existing segments)")
        out["engine"] = dict(backend=sim.backend_process, pipeline=("PhysX GPU (batched)" if gpu else "PhysX CPU"),
                             timestep=sim.dt, gravity=[0.0, 0.0, 0.0], friction_mode=wcfg["physics"]["friction_mode"],
                             friction_realization="EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE (joint effort -c v; Td = 0)",
                             coulomb="OFF (Td = 0, not searched)", armature=bounds["armature"][0],
                             static_excess=0.0, capabilities=caps,
                             model="inertia + viscous damping (gravity zero, Coulomb off)")
    finally:
        sim.close()
    return out


def _one_fit(sim, ep, pe, fcfg, bounds, starts, w, s_vT, floor, delta, out, lab, *, profile=False, log=print) -> dict:
    from . import behavioral_fit as BF
    from .rest_to_rest_objective import factory
    term = dict(weight=float(w), sigma=s_vT) if w > 0 else None
    roll = make_tracing(sim, pe, delta, floor, term)
    t0 = time.time()
    try:
        fit = BF.fit_full_demo(roll, ep, fcfg, bounds, starts, log=log,
                               objective_factory=factory(float(w), s_vT) if w > 0 else None, terminal=term)
    except Exception as ex:
        return dict(status="FIT_FAILED", error=f"{type(ex).__name__}: {ex}", starts=list(starts))
    p = fit["_params"]
    r = roll(p, pe)
    row = dict(status="FITTED", starts=list(starts), warm_start_used=fit["warm_start_used"], params=p.__dict__,
               M=float(p.rho_scale * fit_axis(sim)), c=float(p.viscous), objective=fit["best"]["objective"],
               objective_kind=("V1 = position + %g x huber(qdot(T) / sigma_vT)" % w) if w > 0 else "V0 position only",
               metrics=physx_metrics(r, pe, ep, floor, s_vT), at_bound=fit["at_bound"],
               n_engine_rollouts=int(roll.n_engine_calls), n_gpu_batches=int(roll.n_batches),
               runtime_s=time.time() - t0, optimizer=fit["optimizer"], local_refinement=fit["local_refinement"],
               baselines={k: dict(objective=v.get("objective"), rmse_over_sigma=(v.get("metrics") or {}).get(
                   "rmse_over_sigma")) for k, v in fit["baselines"].items()},
               engine_initial_state=fit["engine_initial_state"], terminal_weight=float(w),
               de_seed=int(fcfg["fitting"]["seed"]), de_maxiter=int(fcfg["fitting"]["maxiter"]),
               de_popsize=int(fcfg["fitting"]["popsize"]))
    out["trajectories"][lab] = dict(q=np.asarray(r.q).tolist(), qdot=None if r.qdot is None else np.asarray(r.qdot).tolist())
    out["traces"][lab] = roll.trace
    if profile:
        try:
            row["_family"] = {k: v for k, v in BF.profile_family(roll, fit, fcfg, bounds, log=log).items()}
        except Exception as ex:
            row["_family"] = dict(status="FAILED", error=f"{type(ex).__name__}: {ex}")
    log(f"[r2r] {lab}: rmse/sigma {row['metrics']['rmse_over_sigma']:.4g}, v_T {row['metrics']['v_sim_T']}, "
        f"{row['n_engine_rollouts']} rollouts")
    return row


def fit_axis(sim):
    return float(sim.axis["axis_inertia"])


def _c0(sim, ep, fr, wcfg, fcfg, bounds, starts, segs, s_vT, floor, delta, out, log=print) -> dict:
    """C0: the EXISTING build_wrench_episode (segments + zero outside, its own window from the segment start) and the
    original G10 at the same budget -- archival behaviour; its theta is also replayed on the C1 episode."""
    from . import behavioral_adapter as BA
    from . import behavioral_fit as BF
    built = BA.build_wrench_episode(fr, wcfg, segs)
    if built["status"] != "OK":
        return dict(status=built["status"], reason=built["reason"])
    ep0 = built["episode"]
    pe0 = ep0.to_physx_episode()
    roll = make_tracing(sim, pe0, delta, floor)
    t0 = time.time()
    try:
        fit = BF.fit_full_demo(roll, ep0, fcfg, bounds, starts, log=log)
    except Exception as ex:
        return dict(status="FIT_FAILED", error=f"{type(ex).__name__}: {ex}")
    p = fit["_params"]
    pe1 = ep.to_physx_episode()
    r1 = roll(p, pe1)
    Qm = np.where(BA.interaction_mask(fr.source_row, segs), fr.Q_full, 0.0)
    out["traces"]["C0_HISTORICAL_MASKED_INPUT"] = roll.trace
    out["trajectories"]["C0_theta_on_C1"] = dict(q=np.asarray(r1.q).tolist(),
                                                 qdot=None if r1.qdot is None else np.asarray(r1.qdot).tolist())
    out["C0_masked_input"] = dict(t=np.asarray(fr.t_s).tolist(), Q_masked=Qm.tolist())
    return dict(status="FITTED", input_policy="C0_HISTORICAL_CONTACT_MASKED (zero outside the existing segments)",
                window=dict(t0=ep0.t0_s, t_end=float(ep0.t_q_s[-1]), n_q=int(len(ep0.t_q_s)), segments=segs),
                params=p.__dict__, M=float(p.rho_scale * fit_axis(sim)), c=float(p.viscous),
                metrics_on_own_window=fit["best"]["metrics"], objective_on_own_window=fit["best"]["objective"],
                metrics_on_C1_episode=physx_metrics(r1, pe1, ep, floor, s_vT),
                n_engine_rollouts=int(roll.n_engine_calls), runtime_s=time.time() - t0,
                note="different window AND input from the C1 rows: compare its C1 replay, not its own-window score")


# =============================================================================================== synthetic benchmark
def _deep_update(d, u):
    for k, v in (u or {}).items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def run_synthetic(out_root, *, joint_type="prismatic", rho_true=1.5, c_true=None, config=None, quick=False,
                  engine=None, python=None, wrench_overrides=None, r2r_overrides=None, noise=2e-4, seed=0,
                  log=print) -> dict:
    """The SAME comparison on an exact synthetic rest-to-rest demonstration (truth known), with ACTUAL PhysX on a
    single-joint reference URDF (gravity zero). Truth trajectory: the exact two-term solution with M = rho_true x the
    URDF axis inertia (PhysX itself integrates semi-implicit Euler at dt: its O(dt) offset is part of the result)."""
    import yaml
    from ..estimators.rest_to_rest_synthetic import episode as syn_episode
    from ..estimators.rest_to_rest_synthetic import exact_rest_to_rest
    from ..forcetwin.pipeline import _git, _write, next_version
    from . import rest_to_rest_report as RP
    from .behavioral_contracts import load_wrench_config, wrench_to_fit_config
    from .urdf_builder import write_single_joint_urdf
    t_start = time.time()
    raw = yaml.safe_load(Path(load_r2r_config(config)["physx"]["use_existing_simplified_config"]).read_text())
    if engine:
        raw["physics"]["engine"] = engine
    _deep_update(raw, wrench_overrides)
    wcfg = load_wrench_config(raw, quick=quick)
    if python:
        wcfg["physics"]["python"] = python
    base = (copy.deepcopy(config) if isinstance(config, dict) else
            (yaml.safe_load(Path(config).read_text()) if config else {}))
    base = _deep_update(base or {}, {"physx": {"engine": wcfg["physics"]["engine"]}})
    _deep_update(base, r2r_overrides)
    cfg = load_r2r_config(base, quick=quick)
    fcfg = wrench_to_fit_config(wcfg)
    out = next_version(Path(out_root))
    git = _git()
    urdf = out / "reference.urdf"
    if joint_type == "prismatic":
        info = write_single_joint_urdf(urdf, joint_type="prismatic", axis=[1.0, 0.0, 0.0], origin=[0, 0, 0], mass=2.0,
                                       com_ref=[0.3, 0.1, 0.0], inertia_com_ref=np.diag([0.02, 0.03, 0.04]))
    else:
        info = write_single_joint_urdf(urdf, joint_type="revolute", axis=[0.0, 0.0, 1.0], origin=[0, 0, 0], mass=2.0,
                                       com_ref=[0.4, 0.0, 0.0], inertia_com_ref=np.diag([0.02, 0.03, 0.04]))
    M = rho_true * info["axis_inertia"]
    c = c_true if c_true is not None else 1.8 * M
    sy = exact_rest_to_rest(M, c, q0=0.12 if joint_type == "prismatic" else -0.3, pushes=((0.3, 3.0 * M),),
                            noise=noise, q_sigma=max(noise, 1e-9), seed=seed, jitter=0.0)
    b = syn_episode(sy, joint_type=joint_type)
    _write(out / "rest_endpoint_evidence.json", b["evidence"].to_dict())
    if b["status"] != "OK":
        raise RuntimeError(f"synthetic episode: {b['status']} {b['reason']}")
    ep = b["episode"]
    ep.save_npz(out / "rest_to_rest_episode.npz")
    from ..estimators import rest_to_rest as RR
    sigma_vT = RR.terminal_speed_sigma(ep)
    analytics = run_analytics(ep, cfg, log=log)
    pinn = run_pinns(ep, cfg, analytics["theta0"], log=log)
    legacy = run_legacy_g9(ep, analytics["theta0"], log=log) if cfg["benchmark"]["include_legacy_g9_matched_adaptations"] \
        else {}
    fr = SimpleNamespace(t_s=ep.force_t_s, Q_full=ep.force_Q, source_row=ep.source_rows_force)
    px = run_physx(ep, fr, cfg, wcfg, fcfg, info, urdf, analytics, pinn, legacy, sigma_vT, None, log=log)
    hashes = dict(files=dict(r2r_config=dict(path=None, sha256=None)), full_wrench_record_sha256=None,
                  episode_sha256=ep.sha256())
    rows = RP.summary_rows(ep, analytics, pinn, px, info, hashes, cfg, legacy)
    hist = dict(n=0, items=[], note="synthetic: no historical outputs")
    RP.write_summary(out, rows, ep, b["evidence"].to_dict(), analytics, pinn, px, sigma_vT, hist,
                     status=dict(git=git, quick=quick))
    _write(out / "physx_comparison.json", {k: v for k, v in px.items() if k not in ("trajectories", "traces")})
    _write(out / "equal_budget_convergence.json", px["traces"])
    _write(out / "pinn_ablation.json", pinn)
    _write(out / "integral_svd.json", analytics["svd"])
    _write(out / "analytical_convolution_1d.json", analytics["conv"])
    _write(out / "endpoint_analytic.json", analytics["endpoint"])
    truth = dict(M=M, c=c, rho_scale=rho_true, joint_type=joint_type, axis_inertia=info["axis_inertia"])
    if cfg["benchmark"]["write_plots"]:
        try:
            RP.plots(out / "figures", ep, fr, analytics, px, None)
        except Exception as ex:
            log(f"[r2r-synthetic] plotting failed: {type(ex).__name__}: {ex}")
    st = dict(status="COMPARED_SYNTHETIC", out_dir=str(out), truth=truth, git=git, quick=bool(quick),
              runtime_s=time.time() - t_start, rows=[{k: r.get(k) for k in ("id", "method", "M", "c",
                                                                         "physx_rmse_over_sigma", "v_sim_T")}
                                                    for r in rows])
    _write(out / "STATUS.json", st)
    return dict(st, _rows=rows, _px=px, _analytics=analytics, _pinn=pinn, _ep=ep, _info=info)
