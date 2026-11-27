"""Force-guided automatic joint selection -- 3D FORCE ONLY (spec FORCEARTGS_CODEX_FORCE_GUIDED_JOINT_SELECTION).

Scope: joint-type SELECTION only. Inputs: the already-tared measured force vector F = wrench[:, 0:3] (the torque
columns are never read), converted ONCE to the object-side robot-global frame by the existing
``force_convention.sensor_corr_to_object_base`` (known R_base_ee x R_ee_sensor, existing object-side sign); the measured
TCP position / linear velocity; the cached RGB-D candidate fits, support, folds and evidence. No torque, no moment, no
generalized effort, no dynamics / simulation / fitting, no tare, no contact classifier (contact is continuous: a weak
force or zero speed makes a DIRECTION undefined, never 'no contact').

Evidence (Section 3):
  rho_EE(t) = F.v_EE / (|F||v_EE|)                         -- shared observation, NOT a classifier
  rho_H(t)  = F.v_H / (|F||v_H|),  v_P = a_P qdot_P, v_R = qdot_R a_R x (p - o_R)   (pure kinematics)
on the SAME source-frame rows (intersection of valid rows). A force verdict needs (i) predicted P / R directions that
differ by more than the measured direction uncertainty and (ii) a held-out time-block score difference with a block
bootstrap CI excluding 0 and exceeding the circular-time-shift null; otherwise FORCE_NONDISCRIMINATIVE.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "forceartgs.joint_force_guided_selection/v1"
FORCE_UNAVAILABLE = "FORCE_UNAVAILABLE"
NONDISC = "FORCE_NONDISCRIMINATIVE"
DEFAULT_CONFIG = dict(enabled=True, force_input="known_transform_already_tared_3d_force_only", use_torque=False,
                      continuous_contact=True, infer_contact_state=False, apply_additional_tare=False,
                      eps_v_mps=0.002, eps_f_N=0.05, eps_f_sigma_k=3.0, direction_separation_k=2.0,
                      bootstrap=1000, null_shifts=500, seed=0)
FIXED = dict(force_input="known_transform_already_tared_3d_force_only", use_torque=False, continuous_contact=True,
             infer_contact_state=False, apply_additional_tare=False)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if p and Path(p).is_file() else None


def load_config(src=None) -> dict:
    import yaml
    doc = {} if src is None else (dict(src) if isinstance(src, dict) else (yaml.safe_load(Path(src).read_text()) or {}))
    doc = doc.get("force_guided_selection", doc)
    bad = sorted(set(doc) - set(DEFAULT_CONFIG))
    if bad:
        raise ValueError(f"force_guided_selection: unknown keys {bad}")
    cfg = dict(DEFAULT_CONFIG, **doc)
    for k, v in FIXED.items():
        if cfg[k] != v:
            raise ValueError(f"force_guided_selection.{k} must be {v!r} (3D force only, tared once, continuous contact)")
    if not (cfg["eps_v_mps"] > 0 and cfg["eps_f_N"] > 0 and cfg["direction_separation_k"] > 0):
        raise ValueError("eps_v_mps, eps_f_N and direction_separation_k must be > 0")
    return cfg


# =============================================================================================== cached visual inputs
def load_cached(accepted_dir, v2_dir=None) -> dict:
    from .observations import TrackObservations
    acc = Path(accepted_dir)
    v2 = Path(v2_dir) if v2_dir else acc / "selection"
    obs = TrackObservations.load(acc / "tracks_observations.npz")
    cj = json.loads((v2 / "joint_candidates_v2.json").read_text())["candidates"]
    js = np.load(v2 / "joint_states_v2.npz")
    cands = {k: dict(kind=k, axis=np.asarray(c["axis"], float) / np.linalg.norm(c["axis"]),
                     origin=None if c.get("origin") is None else np.asarray(c["origin"], float),
                     q=np.asarray(js[f"q_{k}"], float)) for k, c in cj.items()}
    return dict(obs=obs, cands=cands, partition=json.loads((v2 / "model_comparison" / "block_partition.json").read_text()),
                v2_dir=str(v2), accepted_dir=str(acc), selection=json.loads((v2 / "joint_selection_v2.json").read_text()))


def world_is_base_link(scene) -> bool:
    cams = json.loads((Path(scene) / "metadata" / "cameras.json").read_text())
    frames = {str(cams.get("coordinate_frame"))} | {str(c.get("coordinate_frame")) for c in cams.get("cameras", [])}
    frames.discard("None")
    return frames == {"base_link"}


# =============================================================================================== 3D force source
def force_source(scene, robot_archive, frame_map) -> dict:
    """F_obj^B on the visual frames (exact source-index join, no interpolation) from wrench[:, 0:3] ONLY."""
    from .force_convention import SensorCorrForce, load_scene_convention, sensor_corr_to_object_base
    from .robot_tool import load_robot_tool
    from .tcp_contact import tcp_velocity
    scene = Path(scene)
    if not world_is_base_link(scene):
        raise ValueError("metadata/cameras.json is not declared in base_link: T_world_base unknown (refusing)")
    tool = load_robot_tool(scene)
    wf = tool.get("wrench_frame") or {}
    if wf.get("axes") != "ee" or wf.get("R_ee_force_sensor") is None:
        raise ValueError(f"robot_tool.json wrench_frame not usable: {wf}")
    sign, sign_prov = load_scene_convention(scene / "metadata")
    z = np.load(robot_archive, allow_pickle=False)
    rows = np.asarray(z["source_row"], int)
    pos = {int(r): i for i, r in enumerate(rows)}
    want = np.asarray(frame_map.source_frame_index, int)
    missing = [int(s) for s in want if int(s) not in pos]
    if missing:
        raise ValueError(f"robot archive lacks source rows {missing[:5]} (no interpolation)")
    sel = np.array([pos[int(s)] for s in want])
    F_s = np.asarray(z["wrench"], float)[:, 0:3]                              # FORCE ONLY: columns Fx, Fy, Fz
    R_be = np.asarray(z["T_base_ee"], float)[:, :3, :3]
    fo = sensor_corr_to_object_base(SensorCorrForce(F_s, source=str(robot_archive),
                                                    tare="already tared once at the demonstration start (given)"),
                                    R_be, np.asarray(wf["R_ee_force_sensor"], float), sign,
                                    rotation_source="robot_tool.json wrench_frame (known)", sign_prov=sign_prov)
    F_all = np.asarray(fo.values, float)
    T_tcp = np.asarray(z["T_base_tcp"] if "T_base_tcp" in z.files else z["T_base_ee"], float)
    p_tcp = T_tcp[sel, :3, 3]
    t = np.asarray(frame_map.timestamp_s, float)
    nb = max(int(round(0.5 * 50)), 5)
    sigma_F = float(np.linalg.norm(1.4826 * np.median(np.abs(F_all[:nb] - np.median(F_all[:nb], 0)), 0)))
    return dict(F=F_all[sel], p_tcp=p_tcp, v_tcp=tcp_velocity(p_tcp, t), t=t, source_row=rows[sel],
                sigma_F_N=max(sigma_F, 1e-3),
                contract=dict(columns_read="wrench[:, 0:3] (Fx, Fy, Fz) ONLY; torque columns never read",
                              conversion="force_convention.sensor_corr_to_object_base: s R_base_ee R_ee_sensor F (once)",
                              sign=float(sign), sign_provenance=sign_prov, R_ee_force_sensor=wf["R_ee_force_sensor"],
                              tool_override=tool.get("tool_override"),
                              tare="already tared at the demonstration start; no additional tare / bias",
                              contact="continuous for the whole interaction; no contact classification",
                              frame="robot base_link = camera world (metadata/cameras.json)",
                              time_join="exact archive source_row == visual source_frame_index",
                              robot_archive_sha256=_sha(robot_archive),
                              noise_level=f"|1.4826 MAD| of the first {nb} recorded rows (a noise level, not a tare)"))


# =============================================================================================== evidence
def _cos(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return np.einsum("...i,...i->...", a, b) / np.maximum(na * nb, 1e-300)


def force_guided_evidence(fs, cands, partition, cfg) -> dict:
    """Section 3 on the SAME visual rows for every candidate."""
    from .wrench_physics import decompose_force, material_velocity_at_tcp
    rng = np.random.default_rng(int(cfg["seed"]))
    F, p, v_ee, t = fs["F"], fs["p_tcp"], fs["v_tcp"], fs["t"]
    eps_f = max(float(cfg["eps_f_N"]), float(cfg["eps_f_sigma_k"]) * float(fs["sigma_F_N"]))   # direction definedness
    dec = decompose_force(F, v_ee, eps_v=float(cfg["eps_v_mps"]), eps_f=eps_f)
    ee_rows = []
    for i in range(len(t)):
        ok = bool(dec["valid"][i])
        c = float(dec["cos_phi"][i]) if ok else None
        ee_rows.append(dict(time_index=i, t_s=float(t[i]), source_row=int(fs["source_row"][i]), valid=ok,
                            undefined_reason=str(dec["reason"][i]), force_N=float(dec["force_norm"][i]),
                            cos_F_vEE=c, angle_deg=None if c is None else math.degrees(math.acos(max(-1.0, min(1.0, c)))),
                            F_parallel=float(dec["F_parallel"][i]) if ok else None,
                            F_perp=float(dec["F_perp"][i]) if ok else None))
    vH, qd = {}, {}
    for k, c in cands.items():
        vH[k], qd[k] = material_velocity_at_tcp(c, p, t)
    fn = np.linalg.norm(F, axis=1)
    common = np.isfinite(F).all(1) & (fn > eps_f)
    for k in vH:
        common &= np.linalg.norm(vH[k], axis=1) > float(cfg["eps_v_mps"])
    valid_ee = np.array([r["valid"] for r in ee_rows])
    ce = np.array([r["cos_F_vEE"] for r in ee_rows if r["valid"]], float)
    out = dict(
        force_vs_ee=dict(n_rows=len(t), n_valid=int(valid_ee.sum()), fraction_valid=float(valid_ee.mean()) if len(t) else 0,
                         median_cos=float(np.median(ce)) if len(ce) else None,
                         median_angle_deg=float(np.degrees(np.arccos(np.clip(np.median(ce), -1, 1)))) if len(ce) else None,
                         fraction_negative=float(np.mean(ce < 0)) if len(ce) else None,
                         note="signed (retraction / dragging kept); NOT a joint classifier: a tangential push on a hinge "
                              "aligns with the TCP motion too"),
        ee_rows=ee_rows, n_common=int(common.sum()), eps_f_N=eps_f, sigma_F_N=float(fs["sigma_F_N"]),
        common_rule="rows with |F| > max(eps_f_N, k sigma_F) and BOTH predicted linear speeds > eps_v (same source rows "
                    "for P and R); undefined directions are a numerical state, not 'no contact'")
    cand_rows = []
    for i in np.flatnonzero(common):
        r = dict(time_index=int(i), t_s=float(t[i]), source_row=int(fs["source_row"][i]))
        for k in vH:
            c = float(_cos(F[i], vH[k][i]))
            r.update({f"v_{k}": vH[k][i].tolist(), f"cos_F_v{k[0].upper()}": c,
                      f"angle_F_v{k[0].upper()}_deg": math.degrees(math.acos(max(-1.0, min(1.0, c))))})
        cand_rows.append(r)
    out["candidate_rows"] = cand_rows
    if "prismatic" not in vH or "revolute" not in vH:
        out.update(status=FORCE_UNAVAILABLE, reason="both P and R candidates are needed")
        return out
    if common.sum() < 6:
        out.update(status=NONDISC, reason=f"{int(common.sum())} common rows with a defined direction")
        return out
    cP = _cos(F[common], vH["prismatic"][common])
    cR = _cos(F[common], vH["revolute"][common])
    sep = np.degrees(np.arccos(np.clip(np.abs(_cos(vH["prismatic"][common], vH["revolute"][common])), -1, 1)))
    noise = np.degrees(np.median(fs["sigma_F_N"] / np.maximum(fn[common], 1e-12)))
    out.update(median_angle_F_vP_deg=float(np.degrees(np.arccos(np.clip(np.median(cP), -1, 1)))),
               median_angle_F_vR_deg=float(np.degrees(np.arccos(np.clip(np.median(cR), -1, 1)))),
               predicted_direction_separation_deg=dict(median=float(np.median(sep)), p90=float(np.quantile(sep, 0.9))),
               force_direction_uncertainty_deg=float(noise))
    sep_ok = float(np.median(sep)) > float(cfg["direction_separation_k"]) * float(noise)
    if not sep_ok:                                   # the fundamental reason first: no held-out test can separate them
        out.update(status=NONDISC, direction_separable=False,
                   reason=(f"P and R predict the same linear direction (median separation {np.median(sep):.2f} deg <= "
                           f"{cfg['direction_separation_k']} x direction uncertainty {noise:.2f} deg): remote-hinge / "
                           "small-angle degeneracy"))
        return out
    # held-out blocks (the frozen visual folds)
    held = [(min(b), max(b)) for b, h in zip(partition["blocks"], partition["heldout"]) if h]
    ti = np.flatnonzero(common)
    blk = np.full(len(ti), -1)
    for j, (a, b) in enumerate(held):
        blk[(ti >= a) & (ti <= b)] = j
    used = blk >= 0
    d = (cP - cR)
    ub = np.unique(blk[used])
    if len(ub) < 3:
        out.update(status=NONDISC, reason=f"{len(ub)} held-out blocks with common direction rows (< 3)",
                   direction_separable=bool(sep_ok))
        return out
    per = np.array([d[used][blk[used] == u].mean() for u in ub])
    est = float(per.mean())
    boots = np.array([rng.choice(per, len(per)).mean() for _ in range(int(cfg["bootstrap"]))])
    ci = np.percentile(boots, [2.5, 97.5]).tolist()
    Fc = F[common]
    null = []
    for _ in range(int(cfg["null_shifts"])):
        s = int(rng.integers(1, max(2, len(Fc))))
        Fs_ = np.roll(Fc, s, axis=0)
        dd = _cos(Fs_, vH["prismatic"][common]) - _cos(Fs_, vH["revolute"][common])
        null.append(np.mean([dd[used][blk[used] == u].mean() for u in ub]))
    q95 = float(np.quantile(np.abs(null), 0.95))
    fdir = Fc / np.maximum(np.linalg.norm(Fc, axis=1), 1e-300)[:, None]
    resultant = float(np.linalg.norm(fdir.mean(0)))
    out.update(delta_cos_P_minus_R=est, block_bootstrap_ci=ci, n_heldout_blocks=int(len(ub)),
               null_controls=dict(circular_time_shift_abs_q95=q95, n_shifts=int(cfg["null_shifts"]),
                                  sign_reversed_estimate=-est, force_direction_resultant_length=resultant,
                                  constant_direction=bool(resultant > 0.98),
                                  note="sign reversal flips the estimate exactly; a near-constant force direction makes "
                                       "the time-shift null as large as any effect"),
               direction_separable=bool(sep_ok))
    if ci[0] > 0 and abs(est) > q95:
        out.update(status="FAVOURS_PRISMATIC", reason="held-out force direction aligns better with the slide, beyond "
                                                      "the bootstrap CI and the time-shift null")
    elif ci[1] < 0 and abs(est) > q95:
        out.update(status="FAVOURS_REVOLUTE", reason="held-out force direction aligns better with the hinge, beyond "
                                                     "the bootstrap CI and the time-shift null")
    else:
        out.update(status=NONDISC, reason="held-out difference inside the bootstrap CI or the time-shift null")
    return out


# =============================================================================================== selection
def reselect(v2_dir, *, force_guided=None, legacy_r_first=False) -> dict:
    """``select_physical_joint`` on the CACHED evidence of a v2 selection directory (same folds, support, scores)."""
    from .physical_selection import select_physical_joint
    v2 = Path(v2_dir)
    fe = json.loads((v2 / "motion_field" / "field_evidence.json").read_text())
    null = json.loads((v2 / "motion_field" / "rotation_null_calibration.json").read_text())
    conf = json.loads((v2 / "support" / "conflict_report.json").read_text())
    att = json.loads((v2 / "assembly" / "attachment_evidence.json").read_text())
    sep = json.loads((v2 / "model_comparison" / "prediction_separation.json").read_text())
    ho = json.loads((v2 / "model_comparison" / "block_bootstrap.json").read_text())
    cj = json.loads((v2 / "joint_candidates_v2.json").read_text())["candidates"]
    rob = dict(json.loads((v2 / "robot" / "contact_evidence.json").read_text()))
    th = json.loads((v2 / "joint_selection_v2.json").read_text()).get("thresholds") or {}
    if force_guided is not None:
        rob["force_guided"] = force_guided
    dec = select_physical_joint(
        motion_evidence=dict(rotation=fe["rotation_vs_translation"], shared_axis=fe["shared_axis"], hinge=fe["hinge"],
                             null=null, support_quality=fe["support_quality"]),
        model_fits={k: dict(axis=v["axis"], origin=v.get("origin")) for k, v in cj.items()},
        common_support=dict(counts=conf["counts"], consistency=conf["consistency"]),
        attachment_evidence={k: att[k] for k in ("interface", "hinge_field", "hinge_fit", "guide", "implied") if k in att},
        predictive_evidence=dict(separation=sep, heldout=ho), render_evidence=None, robot_evidence=rob,
        cfg={k: th[k] for k in ("max_support_noise_ratio", "prismatic_margin", "max_axis_disagreement_deg",
                                "max_camera_axis_disagreement_deg") if k in th},
        legacy_r_first=legacy_r_first)
    return dec.to_json()


KEEP = ("physical_joint_status", "working_motion_model", "working_model_is_approximation", "physics_joint_export_ready",
        "selected_candidate", "failure_codes", "reasons", "counterfactual")


def run(scene, accepted_dir, out_dir, *, robot_archive=None, v2_dir=None, config=None, log=print) -> dict:
    """B0-B3 on identical cached inputs; append-only ``out_dir`` (must not exist)."""
    import yaml
    cfg = load_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(dict(force_guided_selection=cfg)))
    cached = load_cached(accepted_dir, v2_dir)
    v2 = Path(cached["v2_dir"])
    hashes = dict(tracks_observations=_sha(Path(accepted_dir) / "tracks_observations.npz"),
                  joint_candidates_v2=_sha(v2 / "joint_candidates_v2.json"),
                  joint_states_v2=_sha(v2 / "joint_states_v2.npz"),
                  joint_selection_v2=_sha(v2 / "joint_selection_v2.json"),
                  block_partition=_sha(v2 / "model_comparison" / "block_partition.json"), robot_archive=_sha(robot_archive))
    (out / "source_hashes.json").write_text(json.dumps(hashes, indent=1))
    fs = None
    if robot_archive is None or not Path(robot_archive).is_file():
        f_reason = f"robot archive unavailable: {robot_archive}"
    else:
        try:
            fs = force_source(scene, robot_archive, cached["obs"].frame_map)
            f_reason = None
        except Exception as e:                                                # recorded, never fatal
            f_reason = f"{type(e).__name__}: {e}"
    if fs is not None:
        (out / "force_source_contract.json").write_text(json.dumps(fs["contract"], indent=1, default=str))
        ev = force_guided_evidence(fs, cached["cands"], cached["partition"], cfg)
    else:
        (out / "force_source_contract.json").write_text(json.dumps(dict(status=FORCE_UNAVAILABLE, reason=f_reason), indent=1))
        ev = dict(status=FORCE_UNAVAILABLE, reason=f_reason, ee_rows=[], candidate_rows=[])
    _csv(out / "force_vs_ee.csv", ev.get("ee_rows") or [])
    _csv(out / "force_vs_candidate_motion.csv", ev.get("candidate_rows") or [])
    fg = {k: v for k, v in ev.items() if k not in ("ee_rows", "candidate_rows")}
    B0 = reselect(v2, legacy_r_first=True)
    B1 = reselect(v2)
    B2 = reselect(v2, force_guided=dict(status="DIAGNOSTIC_ONLY", force_vs_ee=fg.get("force_vs_ee")))
    B3 = reselect(v2, force_guided=fg)
    base = cached["selection"]
    for name, d in (("selection_baseline.json", B0), ("selection_corrected_visual.json", B1),
                    ("selection_force_guided.json", B3)):
        full = dict(base, **{k: d[k] for k in KEEP}, force_evidence_status=fg.get("status"),
                    force_guided=fg if name == "selection_force_guided.json" else None,
                    evidence=dict(base.get("evidence") or {}, remote_hinge=d["evidence"].get("remote_hinge"),
                                  force_guided=fg if name == "selection_force_guided.json" else None),
                    selection_variant=name.split(".")[0], automatic_only="no operator declaration is an input")
        (out / name).write_text(json.dumps(full, indent=1, default=str))
    diff = {lab: {k: d[k] for k in ("physical_joint_status", "working_motion_model", "physics_joint_export_ready",
                                    "selected_candidate", "failure_codes")}
            for lab, d in (("B0_baseline", B0), ("B1_corrected_visual", B1), ("B2_plus_force_diagnostics", B2),
                           ("B3_force_guided", B3))}
    diff["force_changed_decision_beyond_B1"] = (B3["physical_joint_status"], B3["working_motion_model"]) != \
        (B1["physical_joint_status"], B1["working_motion_model"])
    diff["B2_equals_B1"] = (B2["physical_joint_status"], B2["working_motion_model"]) == \
        (B1["physical_joint_status"], B1["working_motion_model"])
    (out / "selection_diff.json").write_text(json.dumps(diff, indent=1, default=str))
    se = base.get("evidence") or {}
    rot = se.get("rotation_field") or {}
    hf = (se.get("assembly_geometry") or {}).get("hinge_fit") or {}
    rp = se.get("revolute_vs_prismatic") or {}
    mc = dict(visual=dict(heldout=rp.get("heldout_block_preference"), separation=rp.get("prediction_separation")),
              hinge=dict(d10_to_interface_m=hf.get("d10_to_interface_m"), object_half_diag_m=hf.get("object_half_diag_m"),
                         verdict=hf.get("verdict"), remote=B1["evidence"].get("remote_hinge")),
              rotation=dict(net_rotation_deg=rot.get("net_rotation_deg"), ci95_deg=rot.get("net_rotation_ci95_deg"),
                            detected=rot.get("rotation_detected"), null_q95_deg=(se.get("uncertainty") or {}).get(
                                "null_q95_deg") or (base.get("uncertainty") or {}).get("null_q95_deg"),
                            net_translation_m=rot.get("net_centroid_translation_norm_m")),
              force=fg)
    (out / "model_comparison.json").write_text(json.dumps(mc, indent=1, default=str))
    st = dict(schema=SCHEMA, scene=Path(scene).name, out_dir=str(out), source_hashes=hashes,
              B0=f"{B0['physical_joint_status']}/{B0['working_motion_model']}",
              B1=f"{B1['physical_joint_status']}/{B1['working_motion_model']}",
              B3=f"{B3['physical_joint_status']}/{B3['working_motion_model']}",
              export_ready=B3["physics_joint_export_ready"], force_evidence_status=fg.get("status"),
              force_reason=fg.get("reason"), force_changed_decision_beyond_B1=diff["force_changed_decision_beyond_B1"])
    (out / "STATUS.json").write_text(json.dumps(st, indent=1, default=str))
    (out / "summary.md").write_text(summary_md([dict(st, model_comparison=mc)]))
    try:
        _plots(out / "plots", ev, mc)
    except Exception as e:                                                    # plots never hide numbers
        log(f"[force-guided] plotting failed: {type(e).__name__}: {e}")
    log(f"[force-guided] {st['scene']}: B0 {st['B0']} | B1 {st['B1']} | B3 {st['B3']} | force {st['force_evidence_status']}")
    return st


def _f(x, nd=3):
    if x is None:
        return "DATA_UNAVAILABLE"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_f(v, nd) for v in x) + "]"
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


def summary_md(rows) -> str:
    L = ["# Force-guided joint selection (3D force only)", "",
         "| Scene | R hinge distance / part extent | Measured rotation vs null | Held-out visual P vs R | Median F-EE angle | "
         "Median F-P angle | Median F-R angle | Force verdict | Baseline working joint | Corrected working joint | "
         "Physical status |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mc = r["model_comparison"]
        h, rot, f = mc["hinge"], mc["rotation"], mc["force"]
        ho = (mc["visual"].get("heldout") or {})
        L.append(f"| {r['scene']} | {_f(h['d10_to_interface_m'])} m / half-diag {_f(h['object_half_diag_m'])} m "
                 f"({h.get('verdict')}) | {_f(rot['net_rotation_deg'])} deg CI {_f(rot['ci95_deg'])} vs null q95 "
                 f"{_f(rot['null_q95_deg'])} | {ho.get('preference')} (CI {_f((ho.get('bootstrap') or {}).get('ci'))}) | "
                 f"{_f((f.get('force_vs_ee') or {}).get('median_angle_deg'))} deg | {_f(f.get('median_angle_F_vP_deg'))} deg | "
                 f"{_f(f.get('median_angle_F_vR_deg'))} deg | {f.get('status')} | {r['B0']} | {r['B3']} | "
                 f"export ready {r['export_ready']} |")
    L += ["", "Force reasons:", ""] + [f"- {r['scene']}: {r.get('force_reason')}" for r in rows]
    return "\n".join(L) + "\n"


def _csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _plots(d, ev, mc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    ee = [r for r in ev.get("ee_rows") or [] if r["valid"]]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if ee:
        ax.plot([r["t_s"] for r in ee], [r["angle_deg"] for r in ee], ".", label="angle(F, v_EE)")
    ax.set_ylabel("deg")
    ax.set_xlabel("t [s]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(d / "force_ee_angle.png", dpi=110)
    plt.close(fig)
    cr = ev.get("candidate_rows") or []
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if cr and "angle_F_vP_deg" in cr[0]:
        ax.plot([r["t_s"] for r in cr], [r["angle_F_vP_deg"] for r in cr], "o", ms=3, label="angle(F, v_P)")
        ax.plot([r["t_s"] for r in cr], [r["angle_F_vR_deg"] for r in cr], "x", ms=3, label="angle(F, v_R)")
    ax.set_ylabel("deg")
    ax.legend()
    fig.tight_layout()
    fig.savefig(d / "force_vs_candidate_angle.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    h = mc["hinge"]
    vals = [h.get("d10_to_interface_m") or 0, h.get("object_half_diag_m") or 0]
    ax.bar(["fitted hinge -> interface", "object half-diagonal"], vals)
    ax.set_ylabel("m")
    ax.set_title(f"rotation {_f(mc['rotation']['net_rotation_deg'])} deg, translation "
                 f"{_f(mc['rotation']['net_translation_m'])} m")
    fig.tight_layout()
    fig.savefig(d / "hinge_and_motion.png", dpi=110)
    plt.close(fig)
