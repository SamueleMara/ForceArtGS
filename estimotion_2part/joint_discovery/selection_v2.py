"""Joint selection v2 (``comparison.selection: observable_physical_v2``): orchestration and output tree.

Order (spec 3.4 / 6 -- nothing downstream of the split sees held-out measurements):
  1. candidate-independent static core on ALL times -> residual autocorrelation -> block length
     (measurement noise only; no joint is fitted)            -> block partition written BEFORE fitting
  2. R and P fitted on training blocks only (existing EM / geometry fit, unchanged)
  3. on training observations only: candidate-independent support, hinge-free rigid intervals, shared axis,
     calibrated static-patch null, hinge-from-rotation, consistency, assembly interface, R/P prediction separation
  4. held-out blocks: per-block predictive preference (common support), block bootstrap
  5. ``physical_selection.select_physical_joint`` -> joint_selection_v2.json (+ every evidence file)
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .assembly_geometry import attachment_evidence, guide_evidence, hinge_plausibility, implied_min_rotation_deg
from .blocked_statistics import block_length_from_autocorrelation, make_block_partition
from .candidates import rigid_flow_omega
from .distinguishability import complexity_report, heldout_block_preference, prediction_separation
from .exports import write_json
from .model_selection import calibrated_noise, choose_reference_time, fit_joint_hypotheses
from .motion_field import (PointTable, chained_intervals, estimate_shared_angular_axis, hinge_from_intervals,
                           rotation_vs_translation_evidence)
from .physical_selection import select_physical_joint
from .rotation_observability import calibrate_rotation_null, rotation_observability_report
from .support_consistency import candidate_independent_support, consistency_report, per_camera_frame_coverage

SCHEMA_VERSION = 2


def _log(msg, log):
    if log:
        log(msg)


def declare_partition(obs, cfg, ref=None):
    """Step 1: block length from static-core residual autocorrelation (all times, no joint fit), then blocks."""
    v2 = cfg.selection_v2
    ref = choose_reference_time(obs) if ref is None else ref
    noise, _ = calibrated_noise(obs, cfg)
    sup = candidate_independent_support(obs, noise)
    T = obs.frame_map.T
    series = []
    core = np.flatnonzero(sup["S_core"])
    h = obs.has_3d & sup["S_core"][obs.track]
    tr, tt, z = obs.track[h], obs.time[h], obs.z[h]
    order = np.argsort(tr, kind="stable")
    tr, tt, z = tr[order], tt[order], z[order]
    bounds = np.flatnonzero(np.diff(tr)) + 1
    for seg_t, seg_z in zip(np.split(tt, bounds), np.split(z, bounds)):
        if len(seg_t) >= 6:
            s = np.full(T, np.nan)
            s[seg_t] = seg_z - np.median(seg_z)
            series.append(s)
        if len(series) >= 2000:
            break
    acf = block_length_from_autocorrelation(series, v2.block_acf_threshold, v2.min_block_length,
                                            max_len=max(v2.min_block_length, T // 8))
    part = make_block_partition(T, acf["block_length"], ref, v2.heldout_fraction,
                                source=f"static-core depth residual ACF ({len(core)} core tracks)")
    return part, acf


def robot_evidence_v2(contact=None, calibration=None) -> dict:
    """Gate report only (spec 9.1). Robot terms never alter the visual decision unless every gate passes."""
    if contact is None:
        return dict(contact=dict(status="ABSTAIN", reason="no TCP/wrench archive given"),
                    force=dict(status="ABSTAIN", reason="no TCP/wrench archive given"))
    gates = {}
    cv = getattr(contact, "contact_valid", None)
    gates["contact_annotated"] = bool(cv is not None and getattr(contact, "contact_source", "") and
                                      str(contact.contact_source).startswith("annotations"))
    ready = None
    if calibration is not None:
        try:
            ready = calibration.readiness()
        except Exception as e:  # noqa: BLE001 -- recorded, never fatal
            ready = dict(ok=False, error=str(e))
    gates["wrench_calibration_ready"] = bool(ready and ready.get("ok"))
    reasons = list(getattr(contact, "reasons", []) or [])
    contact_ev = dict(status="ABSTAIN" if not gates["contact_annotated"] else "AVAILABLE_NOT_USED",
                      gates=gates, reasons=reasons,
                      note="TCP is a sliding contact point, never a material point; not used for type selection here")
    force_ev = dict(status="ABSTAIN" if not gates["wrench_calibration_ready"] else "AVAILABLE_NOT_USED",
                    readiness=ready, note="no hard-coded force ratio is ever used as a joint classifier")
    return dict(contact=contact_ev, force=force_ev)


def run_selection_v2(obs, calibs, cfg, out_dir, *, scene=None, fit_tr=None, partition=None, acf=None,
                     test_obs=None, robot=None, contact=None, wrench_calibration=None, legacy=None,
                     source_files=None, log=print) -> dict:
    v2 = cfg.selection_v2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref = choose_reference_time(obs) if partition is None else partition.reference_time_index
    if partition is None:
        partition, acf = declare_partition(obs, cfg, ref)
    write_json(out / "model_comparison" / "block_partition.json", dict(partition.to_json(), acf=acf,
                                                                       declared_before_fitting=True))
    held = np.asarray(partition.heldout_times, int)
    train = obs.select_obs(~np.isin(obs.time, held))
    test = test_obs if test_obs is not None else obs.select_obs(np.isin(obs.time, held))
    if np.isin(train.time, held).any():
        raise AssertionError("held-out times leaked into the training observations")
    _log(f"[v2] blocks of {partition.block_length}: {len(partition.blocks)} blocks, held out "
         f"{sum(partition.heldout)} ({partition.heldout_times})", log)
    if fit_tr is None:
        fit_tr = fit_joint_hypotheses(train, calibs, cfg, ref=ref, log=log)
    noise = fit_tr.problem.noise
    # ------------------------------------------------------------- support + motion field (training only)
    support = candidate_independent_support(train, noise)
    _log(f"[v2] candidate-independent support {support['counts']}", log)
    tab = PointTable(train, noise, calibs)
    tr_t = partition.train_times
    bnd = sorted({b[0] for b, h in zip(partition.blocks, partition.heldout) if not h} | {tr_t[-1]})
    M = support["M_proposal"].astype(float)
    ivs = chained_intervals(tab, M, bnd, noise, calibs, seed=cfg.fit.seed)
    cam_ivs = {c: chained_intervals(tab, M, bnd, noise, calibs, cameras=[c], seed=cfg.fit.seed)
               for c in train.camera_ids} if len(train.camera_ids) > 1 else None
    shared = estimate_shared_angular_axis(ivs, min_sigma=v2.axis_min_sigma)
    null = calibrate_rotation_null(tab, support["S_core"], ivs, axis=shared.get("axis"), n_patches=v2.null_patches,
                                   seed=cfg.fit.seed, calibs=calibs)
    rot = rotation_vs_translation_evidence(ivs, shared, null=null, camera_intervals=cam_ivs,
                                           min_blocks=3)
    hinge = hinge_from_intervals(ivs, shared.get("axis"))
    st_scale = [r.get("static_scale_mm") for r in null.get("per_interval", []) if r.get("status") == "OK"]
    mv_scale = [1000 * iv.diagnostics["robust_scale_m"] for iv in ivs if iv.status == "OK"]
    quality = dict(moving_robust_residual_mm=float(np.median(mv_scale)) if mv_scale else None,
                   static_patch_robust_residual_mm=float(np.median(st_scale)) if st_scale else None)
    quality["moving_over_static_residual"] = (quality["moving_robust_residual_mm"] /
                                              max(quality["static_patch_robust_residual_mm"], 1e-6)
                                              if mv_scale and st_scale else None)
    cons = consistency_report(train, tab, support, ivs, robot=robot)
    # ------------------------------------------------------------- assembly (candidate independent interface)
    att = attachment_evidence(tab, support, ref, gap_m=v2.interface_gap_m, min_cameras=v2.interface_min_cameras,
                              sigma_calib_m=v2.sigma_calib_m)
    fitR, fitP = fit_tr.candidates.get("revolute"), fit_tr.candidates.get("prismatic")
    assembly = dict(interface={k: v for k, v in att.items() if k != "points"},
                    hinge_field=hinge_plausibility(att, hinge.get("axis"), hinge.get("origin")),
                    hinge_fit=hinge_plausibility(att, None if fitR is None else fitR.axis_w,
                                                 None if fitR is None else fitR.origin_w),
                    guide=guide_evidence(att, None if fitP is None else fitP.axis_w),
                    implied=implied_min_rotation_deg(att, rot.get("net_centroid_translation_norm_m", np.nan)))
    # ------------------------------------------------------------- R/P distinguishability + held-out preference
    sep = prediction_separation(fit_tr.candidates, train, calibs, support["M_proposal"], partition,
                                k_sigma=v2.separation_k_sigma)
    ho = heldout_block_preference(fit_tr, test, partition, support["M_proposal"], min_blocks=v2.min_independent_blocks)
    robot_ev = robot_evidence_v2(contact, wrench_calibration)
    fits = {k: dict(axis=np.asarray(c.axis_w).tolist(), origin=None if c.origin_w is None else np.asarray(c.origin_w).tolist(),
                    q_span=float(np.nanmax(c.q) - np.nanmin(c.q))) for k, c in fit_tr.candidates.items()}
    dec = select_physical_joint(
        motion_evidence=dict(rotation=rot, shared_axis=shared, hinge=hinge, null=null, support_quality=quality),
        model_fits=fits, common_support=dict(counts=support["counts"], consistency=cons),
        attachment_evidence=assembly, predictive_evidence=dict(separation=sep, heldout=ho),
        render_evidence=None, robot_evidence=robot_ev,
        cfg=dict(max_support_noise_ratio=v2.max_support_noise_ratio, prismatic_margin=v2.prismatic_margin,
                 max_axis_disagreement_deg=v2.max_axis_disagreement_deg,
                 max_camera_axis_disagreement_deg=v2.max_camera_axis_disagreement_deg))
    # ------------------------------------------------------------- outputs (spec 10.3)
    mf = out / "motion_field"
    _write_csv(mf / "per_interval.csv", [iv.to_row() for iv in ivs])
    write_json(mf / "per_camera_consistency.json", dict(
        per_camera_net_rotation=rot.get("per_camera"),
        per_interval={f"{iv.frame_a}-{iv.frame_b}": iv.per_camera_residuals for iv in ivs},
        camera_only_intervals={c: [iv.to_row() for iv in v] for c, v in (cam_ivs or {}).items()}))
    flow = {}
    for iv in ivs:
        if iv.status == "OK" and iv.inlier_tracks is not None and len(iv.inlier_tracks) >= 3:
            Xa, Xb = tab.X[iv.frame_a, iv.inlier_tracks], tab.X[iv.frame_b, iv.inlier_tracks]
            rf = rigid_flow_omega(Xa, Xb - Xa)
            flow[f"pairwise_omega_{iv.frame_a}_{iv.frame_b}"] = np.asarray(rf["omega"])
            flow[f"kabsch_omega_{iv.frame_a}_{iv.frame_b}"] = iv.rotation_vector_rad
    np.savez(mf / "spatial_flow_diagnostics.npz", **flow,
             note=np.array("pairwise omega = candidates.rigid_flow_omega (small-angle, diagnostic only; its pair "
                           "count is not statistical power)"))
    write_json(mf / "rotation_null_calibration.json", null)
    obs_rep = rotation_observability_report(ivs, shared, null, None)
    write_json(mf / "field_evidence.json", dict(shared_axis=shared, rotation_vs_translation=rot, hinge=hinge,
                                                support_quality=quality, observability=obs_rep))
    sd = out / "support"
    sd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sd / "common_support.npz", S_core=support["S_core"], M_proposal=support["M_proposal"], U=support["U"],
                        chi2_per_dof=support["chi2_per_dof"], max_disp_m=support["max_disp_m"],
                        track_camera=train.track_camera, track_local_id=train.track_local_id)
    _write_csv(sd / "per_camera_frame_coverage.csv", per_camera_frame_coverage(train, support))
    inl = set()
    for iv in ivs:
        if iv.inlier_tracks is not None:
            inl |= set(iv.inlier_tracks.tolist())
    _write_csv(sd / "track_fate.csv", [dict(track=i, camera=train.camera_ids[train.track_camera[i]],
                                            label="S_core" if support["S_core"][i] else
                                            "M_proposal" if support["M_proposal"][i] else "U",
                                            interval_inlier=int(i in inl),
                                            chi2_per_dof=round(float(support["chi2_per_dof"][i]), 3),
                                            max_disp_mm=round(1000 * float(support["max_disp_m"][i]), 2),
                                            robust_disp_mm=round(1000 * float(np.nan_to_num(support["robust_disp_m"][i])), 2))
                                       for i in range(train.n_tracks)])
    write_json(sd / "conflict_report.json", dict(counts=support["counts"], rule=support["rule"], consistency=cons))
    write_json(out / "assembly" / "attachment_evidence.json", dict(assembly, interface_points=att.get("points")))
    mc = out / "model_comparison"
    write_json(mc / "prediction_separation.json", sep)
    _write_csv(mc / "heldout_block_scores.csv", [dict(block=b, delta_ll_R_minus_P=v) for b, v in
                                                 zip([k for k, h in enumerate(partition.heldout) if h],
                                                     ho.get("per_block_delta_ll", []))])
    write_json(mc / "block_bootstrap.json", ho)
    write_json(mc / "complexity_report.json", complexity_report(fit_tr))
    rc = out / "render_comparison"
    write_json(rc / "controlled_counterfactual.json",
               dict(status="NOT_RUN", reason="filled by the joint-aware reconstruction stage (G-render)"))
    write_json(rc / "full_training_seed_report.json",
               dict(status="NOT_RUN", reason="filled by the joint-aware reconstruction stage; needs >= "
                                             f"{v2.render_repeated_seeds} paired seeds"))
    write_json(out / "robot" / "contact_evidence.json", robot_ev)
    cj = {k: dict(fits[k], q=np.asarray(c.q).tolist(), n_moving_tracks=int(np.nansum(c.moving_probability > 0.5)),
                  fitted_on="training blocks only") for k, c in fit_tr.candidates.items()}
    write_json(out / "joint_candidates_v2.json", dict(candidates=cj, frame_map=train.frame_map.to_json()))
    np.savez(out / "joint_states_v2.npz", **{f"q_{k}": np.asarray(c.q) for k, c in fit_tr.candidates.items()},
             q_valid_note=np.array("q at held-out times = training-only interpolation"),
             rgbd_frame_index=train.frame_map.rgbd_frame_index, timestamp_s=train.frame_map.timestamp_s)
    src = dict(source_files=source_files or {}, config_hash=cfg.config_hash())
    src_sha = hashlib.sha256(json.dumps(src, sort_keys=True, default=str).encode()).hexdigest()
    sel = dict(schema_version=SCHEMA_VERSION, **{k: v for k, v in dec.to_json().items()
                                                 if k not in ("evidence",)},
               evidence=dec.evidence, source_manifest_sha256=src_sha, scene=None if scene is None else str(scene),
               cameras=list(obs.camera_ids), block_partition=partition.to_json(),
               thresholds=dict(cfg.selection_v2.__dict__), legacy_decision=legacy,
               export_contract="URDF/USD/MJCF producers must require physics_joint_export_ready=true; "
                               "working_motion_model is never a certified physical joint")
    write_json(out / "joint_selection_v2.json", sel)
    write_json(out / "regression_report.json", dict(
        legacy_decision=legacy, v2_status=dec.physical_joint_status, v2_working_model=dec.working_motion_model,
        changed=None if legacy is None else not (
            (legacy == "revolute" and dec.physical_joint_status == "IDENTIFIED_REVOLUTE") or
            (legacy == "prismatic" and dec.physical_joint_status == "IDENTIFIED_PRISMATIC") or
            (legacy in ("ambiguous/unobservable", "static/no_motion") and not dec.physical_joint_status.startswith("IDENTIFIED")))))
    write_json(out / "manifest.json", dict(created_unix=time.time(), runtime_s=round(time.time() - t0, 1),
                                           schema_version=SCHEMA_VERSION, config=cfg.to_dict(),
                                           config_hash=cfg.config_hash(), source=src, source_sha256=src_sha))
    (out / "joint_selection_v2.md").write_text(_report_md(sel, rot, shared, hinge, null, quality, cons, assembly, sep, ho))
    _log(f"[v2] {dec.physical_joint_status} (working {dec.working_motion_model}, export ready "
         f"{dec.physics_joint_export_ready}) in {time.time() - t0:.0f} s", log)
    return sel


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def _f(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_f(v, nd) for v in x) + "]"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _report_md(sel, rot, shared, hinge, null, quality, cons, assembly, sep, ho) -> str:
    L = [f"# Joint selection v2 -- {sel.get('scene') or ''}", "",
         f"* **physical_joint_status: {sel['physical_joint_status']}**",
         f"* working_motion_model: {sel['working_motion_model']} (approximation: {sel['working_model_is_approximation']})",
         f"* physics_joint_export_ready: {sel['physics_joint_export_ready']}",
         f"* failure codes: {', '.join(sel['failure_codes']) or 'none'}", "", "## Reasons", ""]
    L += [f"* {r}" for r in sel["reasons"]]
    L += ["", "## Why not the other model", ""] + [f"* {k}: {v}" for k, v in (sel.get("counterfactual") or {}).items()]
    L += ["", "## Evidence", "",
          f"* hinge-free net rotation: {_f(rot.get('net_rotation_deg'))} deg, 95% block CI "
          f"{_f(rot.get('net_rotation_ci95_deg'))}; calibrated null q95 {_f(null.get('net_abs_rotation_q95_deg'))} deg; "
          f"detected {rot.get('rotation_detected')}",
          f"* shared axis {_f(shared.get('axis'), 3)}, cone95 {_f(shared.get('axis_cone95_deg'))} deg, informative "
          f"intervals {shared.get('n_informative')}",
          f"* per camera: " + "; ".join(f"{c}: {_f(v.get('net_angle_deg'))} deg (axis vs pooled "
                                        f"{_f(v.get('axis_vs_pooled_deg'))} deg)"
                                        for c, v in (rot.get('per_camera') or {}).items()),
          f"* net centroid translation {_f(rot.get('net_centroid_translation_norm_m'), 4)} m (null q95 "
          f"{_f(null.get('net_translation_q95_m'), 4)} m)",
          f"* hinge from rotation: {hinge.get('status')}, origin {_f(hinge.get('origin'), 3)}, line sd "
          f"{_f(hinge.get('line_sd_m'), 3)} m, pitch {_f(hinge.get('screw_pitch_m'), 4)} m",
          f"* moving/static robust residual {_f(quality.get('moving_robust_residual_mm'))} / "
          f"{_f(quality.get('static_patch_robust_residual_mm'))} mm (ratio {_f(quality.get('moving_over_static_residual'))})",
          f"* consistency flags: {cons.get('inconsistency_flags') or 'none'}",
          f"* assembly interface: {assembly['interface'].get('status')} ({assembly['interface'].get('reason', '')}); "
          f"field hinge {assembly['hinge_field'].get('verdict', assembly['hinge_field'].get('status'))}, fitted hinge "
          f"{assembly['hinge_fit'].get('verdict', assembly['hinge_fit'].get('status'))}, guide "
          f"{assembly['guide'].get('verdict', assembly['guide'].get('status'))}, implied min hinge rotation "
          f"{_f(assembly['implied'].get('implied_min_rotation_deg'))} deg",
          f"* R/P prediction separation: {sep.get('status')}, median block {_f(sep.get('median_block_separation_over_sigma'))}"
          f" sigma (empirical sigma {_f(sep.get('empirical_sigma_px'))} px), spatially varying 3D p90 "
          f"{_f((sep.get('metric_3d_spatially_varying_mm') or {}).get('p90'))} mm",
          f"* held-out blocks: {ho.get('status')}, preference {ho.get('preference')}, "
          f"{ho.get('n_heldout_blocks')} blocks, CI {_f((ho.get('bootstrap') or {}).get('ci'))}", ""]
    return "\n".join(L)
