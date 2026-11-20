"""ForceArtGS run -> G9 inputs (G9 spec 5): accepted final joint authority, calibrated force audit, contact
intervals, mass-model geometry, mechanical spec. Versioned contract ``forceartgs.g9.inputs/v1``.

Kinematics: ``<job>_<kind>/joint_authority/v000`` of the ACCEPTED branch (selection v2), loaded with
``JointAuthority.load`` (schema + joint_states sha256 checked); never the legacy ``estimated_joint.json`` or the
splatfacto blend frame B. Diagnostic branches are read only with ``allow_diagnostic_kinematics`` and never yield a
physical claim.

Force: the robot archive (``T_base_ee``, ``T_base_tcp``, ``wrench``, ``source_row``) joined EXACTLY on source frame
indices. Gates (each PASS / UNVERIFIED / FAIL / NOT_NEEDED with a reason): sensor_rotation, force_sign, contact_point,
tare, tool_gravity_compensation, time_sync, torque_origin, camera_robot_registration, contact_intervals -- taken from
``WrenchCalibration.readiness()`` and the scene metadata; declarations are provenance, not proof. Efforts are built
only inside contact-confirmed segments (unknown contact is excluded, never set to zero).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import resolve_path
from ..core import G9Inputs
from ..dynamics.general_1dof import GravityField
from ..dynamics.mass_properties import TriMesh
from ..estimators.model_spec import G9Interaction

STATUS_MAP = {"IDENTIFIED_REVOLUTE": "IDENTIFIED", "IDENTIFIED_PRISMATIC": "IDENTIFIED",
              "TYPE_UNOBSERVABLE": "UNOBSERVABLE", "MODEL_INCONSISTENT": "REJECTED", "NO_SIGNIFICANT_MOTION": "STATIC"}
GATES = ("sensor_rotation", "force_sign", "contact_point", "tare", "tool_gravity_compensation", "time_sync",
         "torque_origin", "camera_robot_registration", "contact_intervals")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class _ZOH:
    def __init__(self, t, lev, model):
        self.t_s, self.levels, self.model = np.asarray(t, float), np.asarray(lev, float), model

    def at(self, t):
        idx = np.clip(np.searchsorted(self.t_s, np.asarray(t, float), side="right") - 1, 0, len(self.t_s) - 1)
        return self.levels[idx]


def accepted_kinematics(run_root, allow_diagnostic=False, declared_joint_type=None, declaration_note=None) -> dict:
    """Select the accepted final branch (or the diagnostic working model) and load its authority."""
    from estimotion_2part.joint_discovery.integration_contract import JointAuthority
    run = Path(run_root)
    man = json.loads((run / "run_manifest.json").read_text())
    if not (run / "selection.json").is_file():
        st = man.get("stages", {}).get("select", {}).get("status") or "NOT_RUN"
        return dict(status=STATUS_MAP.get(st, "UNKNOWN"), selection_status=st, kind=None, selection=None,
                    reason=f"no selection.json (select stage {st}; authority "
                           f"{man.get('stages', {}).get('authority', {}).get('status')}): no accepted joint authority")
    sel = json.loads((run / "selection.json").read_text())
    if sel.get("schema_version") != 2:
        raise ValueError("G9 requires selection v2 (physical joint selection); legacy selections are diagnostic only")
    st = sel["status"]
    kin = STATUS_MAP.get(st, "UNKNOWN")
    kind = sel.get("selected") if kin == "IDENTIFIED" else sel.get("working_motion_model")
    if declared_joint_type is not None and kin != "IDENTIFIED":
        if not declaration_note:
            raise ValueError("--declared-joint-type needs --declaration-note (who declared it, when)")
        kind, kin = declared_joint_type, "OPERATOR_DECLARED"
    elif declared_joint_type is not None and declared_joint_type != kind:
        raise ValueError(f"declared joint type {declared_joint_type} contradicts the IDENTIFIED {kind}: refusing")
    if kind not in ("revolute", "prismatic"):
        return dict(status=kin, selection_status=st, kind=None, reason=f"no branch for status {st}",
                    selection=sel)
    bdir = Path(man["branch_dirs"][kind])
    adir = bdir / "joint_authority" / "v000"
    auth = JointAuthority.load(adir)                              # schema + joint_states sha256 (fail closed)
    meta = json.loads((adir / "authority.json").read_text())
    v2 = json.loads(Path(sel["selection_v2"]).read_text()) if sel.get("selection_v2") else {}
    obs = {}
    unc = v2.get("uncertainty") or {}
    if unc.get("axis_cone95_deg") is not None:          # 2-D 95 % cone radius -> 1-sigma per tilt direction
        obs["axis_sigma_deg"] = float(unc["axis_cone95_deg"]) / 2.4477
    if unc.get("hinge_line_sd_m") is not None:
        obs["hinge_line_sigma_m"] = float(unc["hinge_line_sd_m"])
    for k in ("axis_sigma_deg", "hinge_line_sigma_m"):
        for blk in (v2.get("evidence", {}), v2.get("observability", {}), v2):
            if k not in obs and isinstance(blk, dict) and blk.get(k) is not None:
                obs[k] = blk[k]
                break
    obs["source"] = "joint_selection_v2.uncertainty (upstream statistical, cluster/bootstrap)" if unc else None
    hand_name = "accepted" if kin == "IDENTIFIED" and sel.get("physics_joint_export_ready") else f"diagnostic_{kind}"
    hand = run / "parttwin_handoff" / hand_name / "articulated_3dgs_handoff.json"
    return dict(status=kin, selection_status=st, kind=kind, authority=auth, authority_dir=str(adir),
                authority_id=meta.get("authority_id"), joint_states_sha256=meta.get("joint_states_sha256"),
                world_frame=auth.world_frame, export_ready=bool(sel.get("physics_joint_export_ready")),
                failure_codes=sel.get("failure_codes"), selection=sel, observability=obs,
                multimodal=dict(force_evidence_used=v2.get("force_evidence_used"),
                                contact_evidence_used=v2.get("contact_evidence_used")),
                visual_handoff=str(hand) if hand.is_file() else None,
                diagnostic=kin != "IDENTIFIED", allow_diagnostic=allow_diagnostic, scene=man.get("scene"),
                declaration_note=declaration_note if kin == "OPERATOR_DECLARED" else None,
                support_centroid=meta.get("support_centroid_world"))


def force_audit(scene, authority, robot_archive, contact_segments, use_torque=False) -> dict:
    """Gate-by-gate calibration audit + per-frame rejection counts. Never upgrades a declaration to VERIFIED."""
    from estimotion_2part.joint_discovery.wrench_calibration import WrenchCalibration
    scene = Path(scene)
    meta = scene / "metadata"
    g, info = {}, {}
    if robot_archive is None or not Path(robot_archive).is_file():
        return dict(status="NO_FORCE", reason="no robot archive (TCP / wrench) for this scene", gates={},
                    failing=["robot_archive"])
    try:
        cal = WrenchCalibration.from_scene_metadata(scene)
        rd = cal.readiness(use_torque=use_torque)["gates"]
    except (OSError, KeyError, ValueError) as e:
        rd = {}
        info["wrench_calibration_error"] = str(e)
    m = {"sensor_rotation": "frame_rotation", "force_sign": "sign", "tare": "tare",
         "tool_gravity_compensation": "gravity", "time_sync": "time_sync", "torque_origin": "torque_transport"}
    for gk, wk in m.items():
        r = rd.get(wk, dict(status="UNVERIFIED", reason="not evaluated"))
        g[gk] = dict(status=r["status"], reason=r["reason"])
    tool = json.loads((meta / "robot_tool.json").read_text()) if (meta / "robot_tool.json").is_file() else {}
    g["contact_point"] = dict(status="UNVERIFIED", reason=tool.get(
        "reference_assumption", "TCP offset declared (robot_tool.json), not independently verified"),
        T_ee_tcp=tool.get("T_ee_tcp"))
    reg = json.loads((meta / "robot_base_registration.json").read_text()) if (
        meta / "robot_base_registration.json").is_file() else {}
    rs = str(reg.get("status", "MISSING"))
    g["camera_robot_registration"] = dict(status="PASS" if ("VERIFIED" in rs and "UNVERIFIED" not in rs) else
                                          "UNVERIFIED", reason=f"robot_base_registration.json status {rs}")
    if contact_segments and Path(contact_segments).is_file():
        import yaml
        doc = yaml.safe_load(Path(contact_segments).read_text())
        segs = doc.get("segments") or []
        prov = sorted({s.get("provenance") for s in segs})
        g["contact_intervals"] = dict(status="PASS" if segs else "FAIL",
                                      reason=f"{len(segs)} contact_confirmed segment(s), provenance {prov} "
                                             f"({doc.get('source')}); outside them contact is UNKNOWN (excluded)",
                                      sha256=_sha(contact_segments))
    else:
        segs = []
        g["contact_intervals"] = dict(status="FAIL", reason="no contact segments: contact unknown everywhere")
    z = np.load(robot_archive, allow_pickle=False)
    rows = np.asarray(z["source_row"], int)
    src = np.asarray(authority.source_frame_index, int)
    covered = np.isin(src, rows)
    info.update(archive=str(robot_archive), archive_sha256=_sha(robot_archive), archive_rows=int(len(rows)),
                authority_frames=int(len(src)), authority_frames_covered=int(covered.sum()),
                clock="t = source_frame_index / 50 Hz (LeRobot frame index); hardware sync not measured")
    W = np.asarray(z["wrench"], float)
    rej = dict(missing_force=int((~np.isfinite(W).all(1)).sum()),
               invalid_tcp=int((~np.isfinite(np.asarray(z["T_base_tcp"], float).reshape(len(rows), -1)).all(1)).sum()),
               authority_frames_without_robot_row=int((~covered).sum()))
    if segs:
        inside = np.zeros(len(rows), bool)
        for s in segs:
            inside |= (rows >= int(s["start_source_frame"])) & (rows <= int(s["end_source_frame"]))
        rej["unknown_contact_excluded"] = int((~inside).sum())
    failing = [k for k in GATES if g.get(k, {}).get("status") not in ("PASS", "NOT_NEEDED")]
    status = "VERIFIED" if not failing else "FORCE_CALIBRATION_UNVERIFIED"
    return dict(schema="forceartgs.g9.force_audit/v1", status=status, gates=g, failing=failing,
                rejections=rej, segments=segs, info=info,
                policy="declarations (sensor axes, sign, TCP, operator contact labels) are provenance, not proof")


def contact_efforts(authority, robot_archive, segments, multiplier=-1.0) -> list:
    """Per contact segment: generalized effort Q(t) at 50 Hz and q(t) at the authority times (DIAGNOSTIC unless
    every force gate passed). Q = a.F (P) or a.((p - o) x F) (R), F = multiplier * R_WS F_sensor on the object."""
    from forceartgs_physics.forces.projection import generalized_effort
    from forceartgs_physics.schemas import ForceRecord, JointKinematics, KinematicStatus
    z = np.load(robot_archive, allow_pickle=False)
    rows = np.asarray(z["source_row"], int)
    T_ee, T_tcp, W = (np.asarray(z[k], float) for k in ("T_base_ee", "T_base_tcp", "wrench"))
    F = multiplier * np.einsum("nij,nj->ni", T_ee[:, :3, :3], W[:, :3])
    P = T_tcp[:, :3, 3]
    a = authority.axis_world
    o = authority.origin_world if authority.origin_world is not None else np.zeros(3)
    joint = JointKinematics(authority.joint_type, a, o, "base_link", KinematicStatus.IDENTIFIED)
    src = np.asarray(authority.source_frame_index, int)
    out = []
    for s in segments:
        lo, hi = int(s["start_source_frame"]), int(s["end_source_frame"])
        m = (rows >= lo) & (rows <= hi)
        k = (src >= lo) & (src <= hi) & np.asarray(authority.q_valid, bool)
        if m.sum() < 5 or k.sum() < 5:
            continue
        fr = ForceRecord(t_s=rows[m] / 50.0, force_N=F[m], point_m=P[m], frame="base_link")
        eff = generalized_effort(joint, fr)
        qs = authority.q_sigma[k] if authority.q_sigma is not None else None
        out.append(G9Interaction(f"{s.get('segment_id', 'contact')}_{lo}-{hi}", authority.joint_type,
                                 np.asarray(authority.timestamp_s)[k], np.asarray(authority.q)[k],
                                 qs if qs is not None and np.all(np.isfinite(qs)) and np.all(qs > 0) else
                                 _q_noise(authority, k), _ZOH(eff.t_s, eff.levels, eff.model), None,
                                 dict(source_rows=[lo, hi], slip_mode=s.get("slip_mode"))))
    return out


def _q_noise(authority, k):
    q = np.asarray(authority.q, float)[k]
    d2 = np.diff(q, 2)
    s = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6) if len(d2) > 3 else 1e-3
    return np.full(int(k.sum()), max(float(s), 1e-5))

# ----------------------------------------------------------------------------------------------- canonical record
GATE_TRANSLATION = {
    # gate: (record evidence keys, additional verification for CERTIFIED export, joint-space handling)
    "sensor_rotation": (("sensor_axes", "positive_work"), "reference push of known direction or multi-pose gravity "
                        "test of R_ee_sensor", "assumption; supported by the positive-work check"),
    "force_sign": (("force_sign", "positive_work"), "reference push of known direction", "assumption; a wrong sign "
                   "is caught by positive work and by physically inconsistent fits"),
    "tare": (("tare_initial", "tare_second_window", "tare_independent_window", "tare_not_duplicated",
              "force_noise_sigma_N"), "documented tare procedure / an operator-designated second contact-free window "
             "within noise", "threshold-conditioned post-contact residual used as a declared bias SCENARIO"),
    "tool_gravity_compensation": (("tool_gravity_change",), "tool + sensor mass and COM, or a measured bound",
                                  "bounded by the EE-orientation change per kg of tool mass"),
    "time_sync": (("row_association", "recorded_timestamps", "hardware_clock_offset"), "measured camera-robot "
                  "exposure offset", "exact row join; residual offset swept as +/- half a row period"),
    "contact_point": (("tcp_offset",), "independent TCP-to-surface verification", "declared TCP; lever swept "
                      "by +/- contact_point_sweep_m"),
    "camera_robot_registration": (("camera_robot_registration",), "measured camera-to-base registration",
                                  "q, axis and TCP share base_link; registration error enters via the axis / "
                                  "lever uncertainty"),
    "contact_intervals": (("contact_intervals",), "independent contact sensing", "operator segments + per-row "
                          "force state; uncertain rows outside segments are excluded"),
    "torque_origin": (("torque_origin",), "documented torque origin", "torques unused (point-force model)"),
}


def record_force_audit(rec, authority, cfg) -> dict:
    """Translate the canonical record evidence into the G9 gates, under the configured force policy."""
    from estimotion_2part.joint_discovery.robot_record import positive_work_check
    ev = dict(rec["evidence"])
    ev["positive_work"] = positive_work_check(rec, authority)
    sig = np.asarray(rec["force_sigma_N"], float)
    tw = ev["tare_second_window"]
    ti = ev.get("tare_independent_window", {})
    tare_ok = None                       # only an INDEPENDENT (operator-designated) window can verify drift
    if ti.get("status") in ("INDEPENDENT_WINDOW_CONSISTENT", "INDEPENDENT_WINDOW_DRIFT"):
        tare_ok = ti["status"] == "INDEPENDENT_WINDOW_CONSISTENT"
    tare_diag = None
    if tw["status"] == "THRESHOLD_CONDITIONED_DIAGNOSTIC":
        tare_diag = bool(np.all(np.abs(np.asarray(tw["value"]["mean_N"])) <= np.maximum(3 * np.asarray(
            tw["value"]["se_N"]), sig)))
    tg = ev["tool_gravity_change"]["value"] or {}
    bound = tg.get("force_bound_N_per_kg_tool")
    gates = {}
    for g, (keys, more, handling) in GATE_TRANSLATION.items():
        sts = {k: ev[k]["status"] for k in keys if k in ev}
        if g == "torque_origin":
            st = "NOT_NEEDED"
        elif g in ("sensor_rotation", "force_sign"):
            st = ("CALIBRATION_ASSUMPTION_CONTRADICTED" if ev["positive_work"]["status"] == "DATA_CONTRADICTS" else
                  "CALIBRATION_ASSUMPTION_UNVERIFIED")
        elif g == "tare":
            st = ("ESTIMATED_INDEPENDENT_WINDOW_STABLE" if tare_ok else "ESTIMATED_INDEPENDENT_WINDOW_DRIFT"
                  if tare_ok is False else ("ESTIMATED_INITIAL_TARE_THRESHOLD_CONDITIONED_CONSISTENT" if tare_diag
                                            else "CALIBRATION_ASSUMPTION_UNVERIFIED"))
        elif g == "tool_gravity_compensation":
            st = "ESTIMATED_BOUNDED" if bound is not None and bound <= float(np.min(sig)) else \
                "CALIBRATION_ASSUMPTION_UNVERIFIED"
        elif g == "time_sync":
            st = "ROW_ALIGNED_OFFSET_NOT_ESTABLISHED"
        else:
            st = "CALIBRATION_ASSUMPTION_UNVERIFIED" if g != "contact_intervals" or rec["in_contact_segment"].any() \
                else "DATA_UNAVAILABLE"
        gates[g] = dict(status=st, evidence=sts, existing=[ev[k] for k in keys if k in ev],
                        additional_verification_for_certified_export=more, joint_space_estimation=handling,
                        needed_for_joint_space_estimation=False if g != "contact_intervals" else True,
                        needed_for_certified_export=g != "torque_origin")
    policy = cfg["pipeline"]["force_policy"]
    contradicted = [g for g, v in gates.items() if v["status"].endswith("CONTRADICTED")]
    unavailable = [g for g, v in gates.items() if v["status"] == "DATA_UNAVAILABLE"]
    failing = [g for g, v in gates.items() if v["status"] not in ("NOT_NEEDED",)]
    if contradicted or unavailable:
        status = "CALIBRATION_ASSUMPTION_CONTRADICTED" if contradicted else "DATA_UNAVAILABLE"
    elif policy == "exploratory":
        status = "EXPLORATORY_ASSUMPTIONS_ACCEPTED"
    else:
        status = "FORCE_CALIBRATION_UNVERIFIED"
    assumptions = [f"{g}: {v['status']} ({v['joint_space_estimation']})" for g, v in gates.items()
                   if v["status"] not in ("NOT_NEEDED",)]
    return dict(schema="forceartgs.g9.force_audit/v2", status=status, policy=policy, gates=gates, failing=failing,
                assumptions_used=assumptions, positive_work=ev["positive_work"],
                tare_independent_window_stable=tare_ok, tare_threshold_conditioned_consistent=tare_diag,
                tare_second_window_stable=tare_ok, tool_gravity_bound_N_per_kg=bound,
                record_sources=rec["sources"], segments=[], info=dict(record=rec["schema"]),
                policy_note="exploratory: declared / estimated quantities are USED as stated, listed here, never "
                            "marked VERIFIED; results are conditional and never production-exported")


def record_interactions(rec, authority, cfg, meta) -> list:
    """Interaction windows = maximal runs of rows that are contact-confirmed (operator segment) or force-absent
    (measured force at the noise floor: known ~zero effort). Uncertain rows outside the segments break a window.
    Effort: Q = a.F (P) or a.((p - o) x F) (R) from the record's object-side base-frame force at the TCP, on the
    SAME clock as q (t = source_frame_index / 50, = the authority clock; recorded timestamps differ <= 2 ms)."""
    src = rec["source_frame_index"]
    F, P = rec["F_object_base_N"], rec["p_tcp_base_m"]
    ok = rec["in_contact_segment"] | (rec["contact_state"] == "absent")
    a = np.asarray(authority.axis_world, float)
    o = np.zeros(3) if authority.origin_world is None else np.asarray(authority.origin_world, float)
    if authority.joint_type == "prismatic":
        Q = F @ a
    else:
        Q = np.einsum("j,nj->n", a, np.cross(P - o, F))
    asrc = np.asarray(authority.source_frame_index, int)
    valid = np.asarray(authority.q_valid, bool)
    out, start = [], None
    for i, good in enumerate(list(ok) + [False]):
        if good and start is None:
            start = i
        if not good and start is not None:
            z = i
            if rec["in_contact_segment"][start:z].any():
                lo, hi = int(src[start]), int(src[z - 1])
                k = (asrc >= lo) & (asrc <= hi) & valid
                if k.sum() >= 6:
                    qs = authority.q_sigma[k] if authority.q_sigma is not None else None
                    it = G9Interaction(f"window_{lo}-{hi}", authority.joint_type, asrc[k] / 50.0,
                                       np.asarray(authority.q, float)[k],
                                       qs if qs is not None and np.all(np.isfinite(qs)) and np.all(qs > 0)
                                       else _q_noise(authority, k),
                                       _ZOH(src[start:z] / 50.0, Q[start:z], "canonical robot record"), None,
                                       dict(meta, source_rows=[lo, hi], n_force_rows=int(z - start),
                                            n_contact_rows=int(rec["in_contact_segment"][start:z].sum()),
                                            n_absent_rows=int((rec["contact_state"][start:z] == "absent").sum())))
                    it.meta["_record_slice"] = (start, z)
                    # effort signal-to-noise: noise-equivalent effort = force sigma x lever (R) or force sigma (P)
                    sig = float(np.sqrt(np.mean(np.asarray(rec["force_sigma_N"], float) ** 2)))
                    if authority.joint_type == "revolute":
                        lev = np.linalg.norm(np.cross(a, P[start:z] - o), axis=1)
                        noise_eq = sig * float(np.median(lev))
                    else:
                        noise_eq = sig
                    qd = np.abs(np.gradient(it.q_obs, it.t_obs))
                    tq = it.t_obs[qd > 0.5 * qd.max()] if qd.max() > 0 else it.t_obs
                    mv = (src[start:z] / 50.0 >= tq.min()) & (src[start:z] / 50.0 <= tq.max())
                    Qm = Q[start:z][mv] if mv.any() else Q[start:z]
                    it.meta["effort_snr"] = dict(
                        rms_effort_while_moving=float(np.sqrt(np.mean(Qm ** 2))), noise_equivalent_effort=noise_eq,
                        snr=float(np.sqrt(np.mean(Qm ** 2)) / noise_eq) if noise_eq > 0 else None,
                        peak_effort=float(np.abs(Q[start:z]).max()),
                        positive_work_fraction=float(np.mean((Qm * np.interp(src[start:z] / 50.0, it.t_obs,
                                                                             np.gradient(it.q_obs, it.t_obs))[mv])
                                                             > 0)) if mv.any() else None,
                        note="RMS of the generalized effort during the fastest motion vs the force noise projected "
                             "on the joint; SNR ~ 1-3 means the effort is at the sensor noise floor")
                    out.append(it)
            start = None
    return out


def record_perturbations(rec, authority, cfg, interactions, observability=None) -> dict:
    """Declared SYMMETRIC scenarios 'family|+/-' built from the record (linearised in the identifiability report;
    never treated as standard deviations):
      contact point (revolute only; Q_P = a.F does not depend on it): +/- d along the lever, the path tangent and
          the hinge axis -- the TCP / tool uncertainty;
      hinge axis tilt (two perpendicular directions) and hinge-line offset (revolute): the upstream JOINT
          uncertainty (joint_selection_v2), separate from the TCP so nothing is double counted;
      threshold-conditioned post-contact tare residual: +/- the residual mean as a force bias scenario."""
    idc = cfg["identifiability"]
    a = np.asarray(authority.axis_world, float)
    a = a / np.linalg.norm(a)
    o = np.zeros(3) if authority.origin_world is None else np.asarray(authority.origin_world, float)
    F, P = rec["F_object_base_N"], rec["p_tcp_base_m"]
    rev = authority.joint_type == "revolute"
    out = {}

    def eff_from(it, Fx=None, Px=None, ax=None, ox=None):
        s0, s1 = it.meta["_record_slice"]
        f = (F if Fx is None else Fx)[s0:s1]
        p = (P if Px is None else Px)[s0:s1]
        aa = a if ax is None else ax / np.linalg.norm(ax)
        oo = o if ox is None else ox
        Q = f @ aa if not rev else np.einsum("j,nj->n", aa, np.cross(p - oo, f))
        return _ZOH(it.effort.t_s, Q, "scenario")
    u1 = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(a, u1)
    d = idc.get("contact_point_sweep_m")
    if d and rev:
        r = P - o
        r = r - np.outer(r @ a, a)
        lev = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
        tan = np.cross(a, lev)
        for name, dirv in (("lever", lev), ("tangent", tan), ("axial", np.tile(a, (len(P), 1)))):
            for sg, k in (("+", 1.0), ("-", -1.0)):
                out[f"tcp_contact_{name}_{d * 100:.1f}cm|{sg}"] = (lambda it, dirv=dirv, k=k:
                                                                    eff_from(it, Px=P + k * d * dirv))
    obs = observability or {}
    sa = obs.get("axis_sigma_deg")
    if sa:
        th = np.deg2rad(sa)
        for name, u in (("u1", u1), ("u2", u2)):
            for sg, k in (("+", 1.0), ("-", -1.0)):
                out[f"hinge_axis_tilt_{name}_{sa:.2f}deg|{sg}"] = (lambda it, u=u, k=k:
                                                                    eff_from(it, ax=a + k * np.tan(th) * u))
    so = obs.get("hinge_line_sigma_m")
    if so and rev:
        for name, u in (("u1", u1), ("u2", u2)):
            for sg, k in (("+", 1.0), ("-", -1.0)):
                out[f"hinge_line_offset_{name}_{so * 100:.1f}cm|{sg}"] = (lambda it, u=u, k=k:
                                                                          eff_from(it, ox=o + k * so * u))
    if idc.get("use_second_window_tare_sweep"):
        tw = rec["evidence"]["tare_second_window"]
        if tw["status"] == "THRESHOLD_CONDITIONED_DIAGNOSTIC":
            b_s = np.asarray(tw["value"]["mean_N"], float)
            b_obj = rec["object_side_multiplier"] * np.einsum("nij,j->ni", rec["R_base_ee"], b_s)
            for sg, k in (("+", 1.0), ("-", -1.0)):
                out[f"tare_threshold_conditioned_residual|{sg}"] = (lambda it, k=k: eff_from(it, Fx=F - k * b_obj))
    return out


def _com_dir(auth, centroid):
    """Unit direction hinge -> moving-support centroid, perpendicular to the axis (the assumed COM direction)."""
    if auth.joint_type != "revolute" or centroid is None or auth.origin_world is None:
        return None, None
    a = np.asarray(auth.axis_world, float)
    a = a / np.linalg.norm(a)
    r = np.asarray(centroid, float) - np.asarray(auth.origin_world, float)
    r = r - (r @ a) * a
    n = float(np.linalg.norm(r))
    return (r / n, n) if n > 1e-6 else (None, None)


def build_inputs(run_root, scene, cfg, *, robot_archive=None, contact_segments=None, force_export=None,
                 declared_joint_type=None, declaration_note=None, log=print) -> G9Inputs:
    from .. import physics_lib
    physics_lib.provenance()
    kin = accepted_kinematics(run_root, cfg["pipeline"]["allow_diagnostic_kinematics"], declared_joint_type,
                              declaration_note)
    if kin.get("kind") is None:
        fa = dict(status="NOT_EVALUATED", reason=kin["reason"], failing=["kinematics"])
        return G9Inputs(Path(scene).name, "revolute", np.array([0, 0, 1.0]), None, kin["status"], False,
                        dict(selection_status=kin["selection_status"]), [], [], fa)
    auth = kin["authority"]
    train, excit, extra_pert, rec = [], [], {}, None
    meta = dict(frontend_force_used=bool(kin["multimodal"]["force_evidence_used"]),
                frontend_contact_used=bool(kin["multimodal"]["contact_evidence_used"]))
    fx = Path(force_export) if force_export else Path(run_root).parent.parent / "contact_annotations" / Path(scene).name
    if robot_archive and Path(robot_archive).is_file() and (fx / "contact_annotations.csv").is_file():
        from estimotion_2part.joint_discovery.robot_record import build_robot_record, save_robot_record
        rec = build_robot_record(scene, fx, robot_archive, contact_segments)
        fa = record_force_audit(rec, auth, cfg)
        its = record_interactions(rec, auth, cfg, meta)
        extra_pert = record_perturbations(rec, auth, cfg, its, kin["observability"])
        fa["interaction_windows"] = [dict(name=i.name, **{k: v for k, v in i.meta.items() if not k.startswith("_")})
                                     for i in its]
        if fa["status"] in ("EXPLORATORY_ASSUMPTIONS_ACCEPTED", "VERIFIED"):
            train = its
        else:
            excit = its
            if cfg["pipeline"]["on_missing_calibration"] == "diagnostic_only":
                train = its
                fa["diagnostic"] = True
        if not its:
            fa["status"] = "DATA_UNAVAILABLE"
            fa["reason"] = "no contact window with >= 6 valid q samples"
        fa["canonical_record"] = rec
    else:
        fa = force_audit(scene, auth, robot_archive, contact_segments,
                         use_torque=bool(cfg["contact"]["use_pure_sensor_torque"]))
        if robot_archive and not (fx / "contact_annotations.csv").is_file():
            fa["note"] = f"no force export at {fx}: raw archive audit only (no canonical record)"
    if rec is None and fa.get("segments") and robot_archive:
        its = contact_efforts(auth, robot_archive, fa["segments"])
        for it in its:
            it.meta.update(frontend_force_used=bool(kin["multimodal"]["force_evidence_used"]),
                           frontend_contact_used=bool(kin["multimodal"]["contact_evidence_used"]))
        if fa["status"] == "VERIFIED":
            train = its
        else:
            excit = its
            if cfg["pipeline"]["on_missing_calibration"] == "diagnostic_only":
                train = its
                fa["diagnostic"] = True
    gcfg = cfg["model"]["gravity"]
    grav = None
    if gcfg["vector_world_mps2"] is not None:
        cal = resolve_path(cfg, gcfg["calibration_file"])
        verified = False
        prov = "vector supplied without a calibration file: UNVERIFIED"
        if cal is not None and cal.is_file():
            c = json.loads(cal.read_text())
            verified = c.get("status") == "MEASURED"
            prov = (f"DECLARED by the operator: {c.get('method')} ({cal.name})" if c.get("status") == "DECLARED_OPERATOR"
                    else f"{cal.name}: status {c.get('status')} ({c.get('method')})")
        grav = GravityField(tuple(gcfg["vector_world_mps2"]), verified, prov, "base_link")
    mesh_p = resolve_path(cfg, cfg["mass_model"]["mesh"])
    mesh = TriMesh.read_obj(mesh_p, cfg["mass_model"]["mesh_units"]) if mesh_p is not None else None
    mech = None
    mp = resolve_path(cfg, cfg["export"]["mechanical_spec"])
    if mp is not None:
        from ..export.dynamics_v2 import load_mechanical_spec
        mech = load_mechanical_spec(mp)
    obs = kin["observability"]
    hashes = dict(authority_id=kin["authority_id"], joint_states_sha256=kin["joint_states_sha256"],
                  selection_sha256=_sha(Path(run_root) / "selection.json"),
                  robot_archive_sha256=fa.get("info", {}).get("archive_sha256") or
                  (fa.get("record_sources") or {}).get("robot_archive_sha256"),
                  force_export_sha256=(fa.get("record_sources") or {}).get("force_export_csv_sha256"),
                  contact_segments_sha256=(fa.get("gates", {}).get("contact_intervals") or {}).get("sha256"),
                  visual_handoff_sha256=_sha(kin["visual_handoff"]) if kin["visual_handoff"] else None,
                  mass_mesh_sha256=None if mesh is None else mesh.sha256())
    return G9Inputs(scene=Path(scene).name, joint_type=auth.joint_type, axis=np.asarray(auth.axis_world, float),
                    origin=None if auth.origin_world is None else np.asarray(auth.origin_world, float),
                    kinematic_status=kin["status"], kinematics_export_ready=kin["export_ready"],
                    kinematics_provenance=dict(selection_status=kin["selection_status"],
                                               authority_dir=kin["authority_dir"], authority_id=kin["authority_id"],
                                               world_frame=kin["world_frame"], q_units=auth.q_units,
                                               failure_codes=kin["failure_codes"], multimodal=kin["multimodal"],
                                               branch=kin["kind"], diagnostic=kin["diagnostic"],
                                               declaration_note=kin.get("declaration_note")),
                    train=train, heldout=[], force_audit=fa, gravity=grav, mass_mesh=mesh,
                    mass_mesh_provenance=dict(source=str(mesh_p)) if mesh_p else None,
                    axis_std_rad=float(np.deg2rad(obs.get("axis_sigma_deg") or 0.0)),
                    origin_std_m=float(obs.get("hinge_line_sigma_m") or 0.0),
                    anchor_frame=int(auth.rgbd_frame_index[auth.reference_time_index]),
                    visual_handoff=kin["visual_handoff"], mechanical_spec=mech, excitation_only=excit,
                    input_hashes=hashes, extra_perturbations=extra_pert,
                    com_direction=_com_dir(auth, kin.get("support_centroid"))[0],
                    com_lever_m=_com_dir(auth, kin.get("support_centroid"))[1],
                    record_context=None if rec is None else dict(
                        rec=rec, joint_type=auth.joint_type, axis=np.asarray(auth.axis_world, float),
                        origin=None if auth.origin_world is None else np.asarray(auth.origin_world, float),
                        q_src=np.asarray(auth.source_frame_index)[np.asarray(auth.q_valid, bool)],
                        q=np.asarray(auth.q, float)[np.asarray(auth.q_valid, bool)],
                        q_std=(auth.q_sigma[np.asarray(auth.q_valid, bool)] if auth.q_sigma is not None else
                               _q_noise(auth, np.asarray(auth.q_valid, bool))),
                        observability=kin["observability"]))
