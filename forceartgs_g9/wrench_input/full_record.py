"""Full native six-axis wrench record for the G10 wrench-driven behavioral calibration.

Built ONLY from existing, known inputs -- the robot archive (``wrench`` [N,6], ``T_base_ee``, ``T_base_tcp``,
``source_row``), the scene metadata (``robot_tool.json/wrench_frame``: sensor axes = EE axes, R_ee_force_sensor;
``contact_force_convention.json``: object-side multiplier) and the joint authority. Experimental facts (given):
E1 the sensor-to-global transform is known; E2 the wrench is already tared at the start (it is used AS RECORDED --
the force export's software baseline is NOT subtracted, that would be a second tare); E3 contact on during the interaction (from the EXISTING segments); E6 rows are joined on source_frame_index (t = row / 50).

Moment reference: the contract records the wrench in EE axes with the pose frame ``ee``; the moment is therefore taken
about the EE frame origin p_BE = T_base_ee[:3, 3]. (The export notes the torque origin as 'recorded_origin_unspecified';
a TCP-origin reading is computed side by side as a DIAGNOSTIC only, never used in fitting.)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .full_wrench_projection import (WRENCH_ORDER, SpatialWrench, WrenchContractError, effort_sigma, joint_effort,
                                     sensor_to_base)

CONTRACT_SCHEMA = "forceartgs.g10.wrench_input_contract/v1"
ROW_HZ = 50.0


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if p and Path(p).is_file() else None


def load_contract(scene_dir, robot_archive, force_export_json=None) -> dict:
    """Parse the KNOWN contract (no calibration step). Raises WrenchContractError naming the missing file / array."""
    scene_dir, arc = Path(scene_dir), Path(robot_archive) if robot_archive else None
    meta = scene_dir / "metadata"
    if arc is None or not arc.is_file():
        raise WrenchContractError(f"robot archive missing: {arc}")
    z = np.load(arc, allow_pickle=False)
    need = ("wrench", "T_base_ee", "source_row")
    miss = [k for k in need if k not in z.files]
    if miss:
        raise WrenchContractError(f"{arc}: missing array(s) {miss}")
    if z["wrench"].ndim != 2 or z["wrench"].shape[1] != 6:
        raise WrenchContractError(f"{arc}: wrench shape {z['wrench'].shape}, expected [N, 6] {WRENCH_ORDER}")
    tool_p = meta / "robot_tool.json"
    if not tool_p.is_file():
        raise WrenchContractError(f"known sensor-frame contract missing: {tool_p}")
    tool = json.loads(tool_p.read_text())
    wf = tool.get("wrench_frame") or {}
    if "R_ee_force_sensor" not in wf or wf.get("axes") != "ee":
        raise WrenchContractError(f"{tool_p}: wrench_frame.axes 'ee' + R_ee_force_sensor expected, got {wf}")
    from estimotion_2part.joint_discovery.force_convention import load_scene_convention, validate_rotation
    R_es = validate_rotation(wf["R_ee_force_sensor"], "R_ee_force_sensor")
    sign, sign_prov = load_scene_convention(meta)           # existing explicit multiplier; raises if absent
    exp = None
    if force_export_json and Path(force_export_json).is_file():
        ann = json.loads(Path(force_export_json).read_text())
        thr = ann.get("thresholds") or {}
        exp = dict(path=str(force_export_json), sha256=_sha(force_export_json),
                   software_baseline_tare_N=thr.get("tare_N"), force_sigma_N=thr.get("sigma_N"),
                   baseline_duration_s=thr.get("baseline_duration_s"), tare_note=ann.get("tare"),
                   gravity_compensation=ann.get("gravity_compensation"),
                   use="force noise sigma only; its software baseline tare is NOT applied (E2: already tared)")
    return dict(schema=CONTRACT_SCHEMA, robot_archive=str(arc), robot_archive_sha256=_sha(arc),
                arrays={k: list(z[k].shape) for k in z.files}, wrench_order=list(WRENCH_ORDER),
                wrench_units=dict(force="N", moment="N m"), wrench_frame="sensor axes = EE axes",
                R_ee_force_sensor=R_es.tolist(), known_transform="R_BS(t) = R_base_ee(t) R_ee_force_sensor (E1)",
                transform_source=f"{tool_p} wrench_frame ({wf.get('alignment_source')}) + archive T_base_ee",
                robot_tool_json=str(tool_p), robot_tool_sha256=_sha(tool_p),
                recorded_pose_frame=tool.get("recorded_pose_frame"),
                moment_reference="ee_frame_origin (p_BE = T_base_ee[:3, 3]); wrench recorded in EE axes with the EE "
                                 "pose frame", recorded_torque_origin_note=wf.get("torque_origin"),
                tcp_torque_transport_applied=bool(wf.get("tcp_torque_transport_applied", False)),
                T_ee_tcp=tool.get("T_ee_tcp"),
                object_side_multiplier=float(sign), sign_provenance=sign_prov,
                sign_semantics="the recorded wrench is the reaction on the robot; the object-side wrench is "
                               "multiplier x recorded (force AND moment about the same point), applied once",
                tare=("already tared at the demonstration start (E2): used as recorded; no additional tare, no "
                      "baseline subtraction"),
                contact=("ON during the interaction (existing force-peak + visual segments), OFF outside; no new "
                         "contact classification"),
                clock="t = source_row / 50 Hz on the shared source_frame_index (E6)", force_export=exp)


@dataclass
class FullWrenchRecord:
    scene: str
    joint_type: str
    axis_B: np.ndarray
    origin_B: object
    t_s: np.ndarray                  # native wrench rows (source_row / 50)
    source_row: np.ndarray
    W_recorded: np.ndarray           # [N, 6] as recorded (sensor axes, tared, robot reaction)
    R_BS: np.ndarray                 # [N, 3, 3] known transform
    p_ref_B: np.ndarray              # [N, 3] moment reference (EE origin)
    F_B: np.ndarray                  # object-side base-frame force
    tau_B: np.ndarray                # object-side base-frame moment about p_ref_B
    Q_full: np.ndarray
    Q_sigma: np.ndarray
    q_t_s: np.ndarray
    source_q_frame: np.ndarray
    q: np.ndarray
    q_sigma: np.ndarray
    contract: dict
    Q_legacy_tcp_force_only: np.ndarray = None   # same rows; NaN where the legacy record has no row
    Q_alt_tcp_moment_origin: np.ndarray = None   # diagnostic: moment read as about the TCP
    noise: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def sha256(self):
        h = hashlib.sha256()
        for a in (self.source_row, self.W_recorded, self.Q_full, self.source_q_frame, self.q):
            h.update(np.ascontiguousarray(a, float).tobytes())
        return h.hexdigest()

    def save_npz(self, path):
        np.savez(path, t_s=self.t_s, source_row=self.source_row, W_recorded=self.W_recorded,
                 wrench_order=np.array(WRENCH_ORDER), R_BS=self.R_BS, p_ref_B=self.p_ref_B, F_object_B=self.F_B,
                 tau_object_B_about_ref=self.tau_B, Q_full=self.Q_full, Q_sigma=self.Q_sigma,
                 Q_legacy_tcp_force_only=(self.Q_legacy_tcp_force_only if self.Q_legacy_tcp_force_only is not None
                                          else np.full_like(self.Q_full, np.nan)),
                 Q_alt_tcp_moment_origin=(self.Q_alt_tcp_moment_origin if self.Q_alt_tcp_moment_origin is not None
                                          else np.full_like(self.Q_full, np.nan)),
                 q_t_s=self.q_t_s, source_q_frame=self.source_q_frame, q=self.q, q_sigma=self.q_sigma,
                 axis_B=np.asarray(self.axis_B, float),
                 origin_B=np.asarray(self.origin_B if self.origin_B is not None else [np.nan] * 3, float))


def _mad(x):
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x, 0)), 0)


def build(contract: dict, *, scene, joint_type, axis, origin, q_src, q, q_sigma, legacy=None,
          baseline_s=0.5, compare_alternative_origin=True) -> FullWrenchRecord:
    """``legacy``: optional canonical robot record (dict with source_frame_index, F_object_base_N, p_tcp_base_m) for
    the TCP-force-only comparison. Every archive row is kept; nothing is thresholded, filtered or contact-masked."""
    z = np.load(contract["robot_archive"], allow_pickle=False)
    rows = np.asarray(z["source_row"], int)
    if np.any(np.diff(rows) <= 0):
        raise WrenchContractError("archive source_row not strictly increasing (no re-timing is done)")
    W = np.asarray(z["wrench"], float)
    T_ee = np.asarray(z["T_base_ee"], float)
    R_es = np.asarray(contract["R_ee_force_sensor"], float)
    R_BS = np.einsum("nij,jk->nik", T_ee[:, :3, :3], R_es)
    p_E = T_ee[:, :3, 3]
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    o = None if joint_type == "prismatic" else np.asarray(origin, float)
    w = sensor_to_base(SpatialWrench.from_archive(W), R_BS, p_E, contract["object_side_multiplier"])
    Q = joint_effort(w, joint_type, a, o)
    # noise: force sigma from the existing export (sensor axes) when present; moment sigma ESTIMATED by the SAME
    # baseline-MAD method over the first baseline_s of the recording (a noise level, not a tare)
    nb = max(int(round(baseline_s * ROW_HZ)), 5)
    exp = contract.get("force_export") or {}
    sF = np.asarray(exp.get("force_sigma_N") or _mad(W[:nb, :3]), float)
    sT = np.asarray(_mad(W[:nb, 3:]), float)
    sF, sT = np.maximum(sF, 1e-6), np.maximum(sT, 1e-7)
    Qs = effort_sigma(R_BS, p_E, sF, sT, joint_type, a, o)
    noise = dict(force_sigma_N=sF.tolist(), moment_sigma_Nm=sT.tolist(),
                 force_sigma_source=("force export thresholds.sigma_N" if exp.get("force_sigma_N") else
                                     f"ESTIMATED 1.4826 MAD over the first {baseline_s} s"),
                 moment_sigma_source=f"ESTIMATED 1.4826 MAD over the first {baseline_s} s of the archive (no moment "
                                     "noise in the existing record)",
                 covariance="diagonal per-axis sensor-frame approximation (no verified 6x6 covariance), propagated "
                            "through the linear projection")
    Q_leg = None
    if legacy is not None:
        Q_leg = np.full(len(rows), np.nan)
        lr = np.asarray(legacy["source_frame_index"], int)
        Fo = np.asarray(legacy["F_object_base_N"], float)
        P = np.asarray(legacy["p_tcp_base_m"], float)
        ql = Fo @ a if joint_type == "prismatic" else np.einsum("j,nj->n", a, np.cross(P - o, Fo))
        idx = {int(r): i for i, r in enumerate(rows)}
        for i, r in enumerate(lr):
            if int(r) in idx:
                Q_leg[idx[int(r)]] = ql[i]
    Q_alt = None
    diag = {}
    if compare_alternative_origin and "T_base_tcp" in z.files:
        p_T = np.asarray(z["T_base_tcp"], float)[:, :3, 3]
        w_alt = sensor_to_base(SpatialWrench.from_archive(W), R_BS, p_T, contract["object_side_multiplier"])
        Q_alt = joint_effort(w_alt, joint_type, a, o)
        # moment-reference consistency (diagnostic only, never changes the input): recorded moment vs the moment about
        # the EE origin of a point force at the TCP, in sensor axes, on rows with |F| > 0.5 N
        r_tcp = np.einsum("nji,nj->ni", R_BS, p_T - p_E)
        pred = np.cross(r_tcp, W[:, :3])
        m = np.linalg.norm(W[:, :3], axis=1) > 0.5
        if m.sum() >= 10:
            out = {}
            for i, ax in enumerate(("x", "y")):
                A = np.c_[pred[m, i], np.ones(m.sum())]
                k = float(np.linalg.lstsq(A, W[m, 3 + i], rcond=None)[0][0])
                c = float(np.corrcoef(pred[m, i], W[m, 3 + i])[0, 1])
                out[f"tau_{ax}"] = dict(slope_vs_rtcp_x_F=k, corr=c)
            diag["moment_reference_consistency"] = dict(
                rows=int(m.sum()), **out, reading="slope ~ 1 and high correlation: the recorded moment is about a "
                "point ~|r_tcp| behind the TCP (the EE / sensor origin); slope ~ 0: about the TCP. Diagnostic only.")
    q_src, q, q_sigma = (np.asarray(x) for x in (q_src, q, q_sigma))
    return FullWrenchRecord(scene=str(scene), joint_type=joint_type, axis_B=a, origin_B=o, t_s=rows / ROW_HZ,
                            source_row=rows, W_recorded=W, R_BS=R_BS, p_ref_B=p_E, F_B=w.F, tau_B=w.tau, Q_full=Q,
                            Q_sigma=Qs, q_t_s=q_src.astype(int) / ROW_HZ, source_q_frame=q_src.astype(int),
                            q=q.astype(float), q_sigma=q_sigma.astype(float), contract=contract,
                            Q_legacy_tcp_force_only=Q_leg, Q_alt_tcp_moment_origin=Q_alt, noise=noise,
                            diagnostics=diag)
