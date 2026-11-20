"""Canonical calibrated robot record: ONE preprocessing of the recorded robot data, shared by the joint frontend
and G9 dynamics (schema ``forceartgs.robot_record/v1``).

Built only from outputs that already exist -- nothing is re-tared, re-calibrated or re-synchronised:
  * ``samuele/export_force_contact_annotations.py`` export (``contact_annotations.csv`` + ``.json``): tare-corrected
    force in sensor axes at the native rows, recorded timestamps, baseline tare / noise, per-row contact state,
    EE-orientation (tool-gravity) bound, declared wrench-frame convention;
  * the robot archive (``T_base_ee``, ``T_base_tcp``, ``source_row``) used by joint discovery;
  * ``metadata/contact_force_convention.json`` (object-side multiplier), ``metadata/robot_tool.json`` (TCP),
    ``metadata/robot_base_registration.json``, the operator contact segments.

Force on the object in base_link: F_obj = multiplier * R_base_ee @ R_ee_sensor @ F_sensor_corr, applied at the TCP.

Every quantity carries an evidence status, never upgraded:
  MEASURED                  recorded by the hardware (samples, poses, timestamps)
  ESTIMATED                 computed from the data with a stated method (tare, noise, bounds)
  DATA_CONSISTENT           a data check supports a declaration (e.g. positive work); still not a calibration
  DECLARED                  operator / dataset declaration (axes, sign, TCP offset, contact labels)
  VERIFIED_BY_CONSTRUCTION  exact by construction (row association by source_frame_index)
  NOT_ESTABLISHED           no evidence exists (e.g. hardware camera-robot exposure offset)
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "forceartgs.robot_record/v1"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _ev(status, value=None, source=None, method=None, note=None):
    d = dict(status=status)
    for k, v in (("value", value), ("source", source), ("method", method), ("note", note)):
        if v is not None:
            d[k] = v
    return d


def build_robot_record(scene, force_export_dir, robot_archive, contact_segments=None) -> dict:
    """Arrays at the native recorded rows + an evidence table. Raises FileNotFoundError when an input is missing
    (the caller reports DATA_UNAVAILABLE)."""
    scene, fx, arc = Path(scene), Path(force_export_dir), Path(robot_archive)
    meta = scene / "metadata"
    csv_p, js_p = fx / "contact_annotations.csv", fx / "contact_annotations.json"
    for p in (csv_p, js_p, arc):
        if not p.is_file():
            raise FileNotFoundError(str(p))
    ann = json.loads(js_p.read_text())
    rows = list(csv.DictReader(open(csv_p)))
    src = np.array([int(r["source_frame_index"]) for r in rows])
    ts = np.array([float(r["timestamp_s"]) for r in rows])
    Fs = np.array([[float(r[k]) for k in ("Fx_corr_N", "Fy_corr_N", "Fz_corr_N")] for r in rows])
    state = np.array([r["contact_state"] for r in rows])
    z = np.load(arc, allow_pickle=False)
    arow = np.asarray(z["source_row"], int)
    index = {int(r): i for i, r in enumerate(arow)}
    missing = [int(s) for s in src if int(s) not in index]
    if missing:
        raise ValueError(f"robot archive lacks rows {missing[:5]} of the force export (no interpolation)")
    sel = np.array([index[int(s)] for s in src])
    T_ee = np.asarray(z["T_base_ee"], float)[sel]
    T_tcp = np.asarray(z["T_base_tcp"], float)[sel]
    wf = ann.get("wrench_frame") or {}
    R_es = np.asarray(wf.get("R_ee_force_sensor", np.eye(3)), float)
    conv = json.loads((meta / "contact_force_convention.json").read_text())
    mult = float(conv.get("object_side_force_multiplier", -1))
    F_obj = mult * np.einsum("nij,jk,nk->ni", T_ee[:, :3, :3], R_es, Fs)
    P = T_tcp[:, :3, 3]
    thr = ann.get("thresholds") or {}
    sigma = np.asarray(thr.get("sigma_N", [np.nan] * 3), float)
    tare = np.asarray(thr.get("tare_N", [np.nan] * 3), float)
    # tare stability in a SECOND, separated contact-free window: force-absent rows after the last contact row
    contact_rows = np.flatnonzero(state == "contact")
    late = np.zeros(len(src), bool)
    if len(contact_rows):
        late[contact_rows.max() + 1:] = state[contact_rows.max() + 1:] == "absent"
    late_mean = Fs[late].mean(0) if late.sum() >= 10 else None
    late_se = Fs[late].std(0) / np.sqrt(late.sum()) if late.sum() >= 10 else None
    # contact segments (operator labels): rows inside are contact-confirmed
    in_seg = np.zeros(len(src), bool)
    segs = []
    if contact_segments and Path(contact_segments).is_file():
        import yaml
        doc = yaml.safe_load(Path(contact_segments).read_text())
        segs = doc.get("segments") or []
        for s in segs:
            in_seg |= (src >= int(s["start_source_frame"])) & (src <= int(s["end_source_frame"]))
    cams = json.loads((meta / "cameras.json").read_text())
    frames = {c.get("coordinate_frame") for c in cams["cameras"]}
    tool = json.loads((meta / "robot_tool.json").read_text()) if (meta / "robot_tool.json").is_file() else {}
    reg = json.loads((meta / "robot_base_registration.json").read_text()) if (
        meta / "robot_base_registration.json").is_file() else {}
    gc = ann.get("gravity_compensation") or {}
    rot_deg = gc.get("max_rotation_from_baseline_deg")
    g_per_kg = 9.80665 * np.sin(np.deg2rad(rot_deg)) if rot_deg is not None else None
    dt = np.diff(ts)
    evidence = {
        "force_samples": _ev("MEASURED", source=str(csv_p), note=f"{len(src)} native rows; tare-corrected sensor-axis "
                             "force exported by export_force_contact_annotations.py"),
        "tare_initial": _ev("ESTIMATED", tare.tolist(), method=thr.get("noise_model") or "baseline mean",
                            note=f"first {thr.get('baseline_duration_s')} s contact-free baseline"),
        "force_noise_sigma_N": _ev("ESTIMATED", sigma.tolist(), method="1.4826 x MAD per axis over the baseline"),
        "tare_second_window": (_ev("ESTIMATED", dict(mean_N=late_mean.tolist(), se_N=late_se.tolist(),
                                                     n=int(late.sum())),
                                   method="mean of force-absent rows after the last contact (separated window)")
                               if late_mean is not None else _ev("NOT_ESTABLISHED",
                                                                 note="fewer than 10 force-absent rows after contact")),
        "tool_gravity_change": _ev("ESTIMATED", dict(max_rotation_deg=rot_deg, force_bound_N_per_kg_tool=g_per_kg),
                                   method="EE orientation change from the baseline; tool mass unknown, so the bound "
                                          "is per kg of tool + sensor mass",
                                   note=gc.get("reason")),
        "sensor_axes": _ev("DECLARED", R_es.tolist(), source=wf.get("alignment_source")),
        "force_sign": _ev("DECLARED", mult, source=conv.get("confirmation_source"),
                          note=f"recorded quantity {conv.get('recorded_quantity')}"),
        "torque_origin": _ev("NOT_ESTABLISHED", note=wf.get("torque_origin") or "unspecified; torques unused"),
        "tcp_offset": _ev("DECLARED", tool.get("T_ee_tcp"), source=tool.get("calibration_source"),
                          note=tool.get("reference_assumption")),
        "row_association": _ev("VERIFIED_BY_CONSTRUCTION", note="force, TCP and q joined on the exact "
                               "source_frame_index (no interpolation)"),
        "recorded_timestamps": _ev("MEASURED", dict(dt_min=float(dt.min()), dt_max=float(dt.max()),
                                                    max_dev_from_50Hz_s=float(np.abs(ts - src / 50.0).max())),
                                   note="LeRobot recorded timestamps"),
        "hardware_clock_offset": _ev("NOT_ESTABLISHED", note="camera exposure vs wrench sample offset never "
                                     "measured; the exact row join bounds it only by the row period"),
        "camera_robot_registration": _ev("DECLARED" if "UNVERIFIED" in str(reg.get("status")) else "MEASURED",
                                         reg.get("status"), note="cameras.json frames " + ",".join(sorted(map(str, frames)))),
        "contact_intervals": _ev("DECLARED" if segs else "NOT_ESTABLISHED",
                                 [[int(s["start_source_frame"]), int(s["end_source_frame"])] for s in segs],
                                 source=str(contact_segments) if segs else None,
                                 note="operator-validated force + visual estimate; outside: per-row force state"),
    }
    return dict(schema=SCHEMA, scene=scene.name, source_frame_index=src, timestamp_s=ts, F_sensor_corr_N=Fs,
                F_object_base_N=F_obj, p_tcp_base_m=P, R_base_ee=T_ee[:, :3, :3], contact_state=state,
                in_contact_segment=in_seg, force_sigma_N=sigma, object_side_multiplier=mult,
                T_world_base=np.eye(4) if frames == {"base_link"} else None, evidence=evidence,
                sources=dict(force_export_csv=str(csv_p), force_export_csv_sha256=_sha(csv_p),
                             force_export_json_sha256=_sha(js_p), robot_archive=str(arc),
                             robot_archive_sha256=_sha(arc), contact_segments=str(contact_segments) if segs else None,
                             parquet_sha256=(ann.get("sources") or {}).get("parquet_sha256")))


def save_robot_record(rec, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arr = {k: v for k, v in rec.items() if isinstance(v, np.ndarray)}
    np.savez(out / "canonical_robot_record.npz", **arr)
    meta = {k: v for k, v in rec.items() if not isinstance(v, np.ndarray)}
    (out / "canonical_robot_record.json").write_text(json.dumps(meta, indent=1, default=str))
    return out


def positive_work_check(rec, authority, min_force_sigma=3.0) -> dict:
    """Data consistency of the declared sign / sensor axes: during contact the robot pushes the part ALONG its
    motion, so the generalized effort and q' have the same sign (Q q' > 0). Uses rows with force above
    min_force_sigma x noise and |q'| above its noise. Supports a declaration; never verifies it."""
    src = np.asarray(authority.source_frame_index, int)
    q = np.asarray(authority.q, float)
    t = np.asarray(authority.timestamp_s, float)
    qd = np.gradient(q, t)
    a = np.asarray(authority.axis_world, float)
    o = np.zeros(3) if authority.origin_world is None else np.asarray(authority.origin_world, float)
    idx = {int(s): i for i, s in enumerate(rec["source_frame_index"])}
    rows = [(k, idx[int(s)]) for k, s in enumerate(src) if int(s) in idx]
    F, P = rec["F_object_base_N"], rec["p_tcp_base_m"]
    sig = float(np.nanmax(rec["force_sigma_N"]))
    pw = []
    for k, i in rows:
        f = F[i]
        if np.linalg.norm(f) < min_force_sigma * sig:
            continue
        Q = a @ f if authority.joint_type == "prismatic" else a @ np.cross(P[i] - o, f)
        pw.append(Q * qd[k])
    pw = np.asarray(pw)
    big = pw[np.abs(pw) > np.percentile(np.abs(pw), 25)] if len(pw) >= 8 else pw
    frac = float((big > 0).mean()) if len(big) else None
    return dict(status=("DATA_CONSISTENT" if frac is not None and frac >= 0.8 else
                        "DATA_CONTRADICTS" if frac is not None and frac <= 0.2 else "INCONCLUSIVE"),
                fraction_positive_work=frac, n_samples=int(len(big)),
                method="sign of Q q' on force-active rows (largest 75 % of |Q q'|); pushing does positive work")
