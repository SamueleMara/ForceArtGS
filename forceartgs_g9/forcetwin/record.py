"""Canonical JointInteractionRecord (ForceTwin-adaptation spec section 2.3): ONE shared observation set for every
identification method (NNLS, weak form, integral LS, rollout, PINN, EKF, residual).

Built from the existing canonical robot record (signed object-side force, sign applied exactly once from metadata)
plus the accepted / hypothesis joint authority. Nothing is re-tared, re-synchronised or interpolated: force stays on
its native 50 Hz rows (forward ZOH), q on the RGB-D clock, joined on ``source_frame_index``.

    tau_R = a.((p_contact - o) x F_obj)   (point-force approximation: no sensor moment -- torque origin unknown)
    tau_P = a.F_obj
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

SCHEMA = "forceartgs.g9.joint_interaction_record/v1"


class RecordError(ValueError):
    pass


@dataclass
class JointInteractionRecord:
    scene: str
    joint_hypothesis_id: str
    joint_type: str
    axis_B: np.ndarray
    origin_B: Optional[np.ndarray]
    q_reference: dict
    force_t_s: np.ndarray            # native force-row clock (source_frame_index / 50 = the q clock)
    source_force_row: np.ndarray
    Q_object: np.ndarray             # signed generalized effort on the object
    Q_sigma: np.ndarray
    force_object_B: np.ndarray       # [Nf, 3] robot-on-object force, base_link
    contact_point_B: np.ndarray      # [Nf, 3] point-force application point (TCP by default)
    contact_mode: np.ndarray         # per force row: sliding / sticking / absent / uncertain
    in_contact: np.ndarray
    q_t_s: np.ndarray
    source_q_frame: np.ndarray
    q: np.ndarray
    q_sigma: np.ndarray
    windows: list                    # [(name, force_row_slice, q_index_slice, source_rows)]
    contact_intervals: list = field(default_factory=list)
    force_calibration_assumptions: dict = field(default_factory=dict)
    geometry_hypothesis: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ checks
    def validate(self):
        if self.joint_type not in ("revolute", "prismatic"):
            raise RecordError(f"joint_type {self.joint_type!r}")
        a = np.asarray(self.axis_B, float)
        if a.shape != (3,) or not np.isfinite(a).all() or abs(np.linalg.norm(a) - 1) > 1e-9:
            raise RecordError("axis_B must be a finite unit 3-vector")
        if self.joint_type == "revolute" and (self.origin_B is None or not np.isfinite(self.origin_B).all()):
            raise RecordError("revolute record needs a finite hinge origin")
        for name in ("force_t_s", "q_t_s"):
            t = getattr(self, name)
            if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
                raise RecordError(f"{name} must be finite, sorted and unique")
        for name in ("Q_object", "Q_sigma", "q", "q_sigma"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise RecordError(f"{name} has non-finite values")
        if np.any(self.q_sigma <= 0) or np.any(self.Q_sigma <= 0):
            raise RecordError("uncertainties must be > 0")
        if self.provenance.get("object_side_multiplier") not in (-1.0, 1.0):
            raise RecordError("signed force provenance missing (object_side_multiplier)")
        return self

    # ------------------------------------------------------------------ derived
    def effort_for(self, axis, origin=None, contact_point=None) -> np.ndarray:
        """Re-project the SAME measured 3-D force onto another joint hypothesis (virtual work)."""
        a = np.asarray(axis, float)
        a = a / np.linalg.norm(a)
        if self.joint_type == "prismatic":
            return self.force_object_B @ a
        P = self.contact_point_B if contact_point is None else contact_point
        return np.einsum("j,nj->n", a, np.cross(P - np.asarray(origin, float), self.force_object_B))

    def reversed_coordinate(self) -> "JointInteractionRecord":
        """The equivalent convention a' = -a, q' = -q (tau' = -tau): physics and work tau qdot are invariant."""
        r = JointInteractionRecord(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        r.axis_B = -self.axis_B
        r.q = -self.q
        r.Q_object = -self.Q_object
        r.joint_hypothesis_id = self.joint_hypothesis_id + "__reversed"
        r.q_reference = dict(self.q_reference, q_ref=-float(self.q_reference.get("q_ref", 0.0)))
        return r

    def interactions(self):
        """G9Interaction windows (the existing estimators' input) from THIS record."""
        from ..estimators.model_spec import G9Interaction
        from ..synthetic import ZOH
        out = []
        for name, fs, qs, src in self.windows:
            it = G9Interaction(name, self.joint_type, self.q_t_s[qs], self.q[qs], self.q_sigma[qs],
                               ZOH(self.force_t_s[fs], self.Q_object[fs]),
                               meta=dict(source_rows=list(src), _record_slice=(fs.start, fs.stop),
                                         effort_sigma=float(np.median(self.Q_sigma[fs]))))
            out.append(it)
        return out

    def sha256(self) -> str:
        h = hashlib.sha256()
        for k in ("axis_B", "force_t_s", "Q_object", "q_t_s", "q"):
            h.update(np.ascontiguousarray(getattr(self, k), float).tobytes())
        return h.hexdigest()

    def save(self, out_dir) -> dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        arr = {k: np.asarray(getattr(self, k)) for k in ("axis_B", "force_t_s", "source_force_row", "Q_object",
                                                         "Q_sigma", "force_object_B", "contact_point_B", "in_contact",
                                                         "q_t_s", "source_q_frame", "q", "q_sigma")}
        arr["contact_mode"] = np.asarray(self.contact_mode, dtype="U16")
        if self.origin_B is not None:
            arr["origin_B"] = np.asarray(self.origin_B)
        np.savez(out / "joint_interaction_record.npz", **arr)
        meta = dict(schema=SCHEMA, scene=self.scene, joint_hypothesis_id=self.joint_hypothesis_id,
                    joint_type=self.joint_type, axis_B=self.axis_B.tolist(),
                    origin_B=None if self.origin_B is None else self.origin_B.tolist(), q_reference=self.q_reference,
                    windows=[dict(name=n, force_rows=[fs.start, fs.stop], q_index=[qs.start, qs.stop],
                                  source_rows=list(src)) for n, fs, qs, src in self.windows],
                    contact_intervals=self.contact_intervals,
                    force_calibration_assumptions=self.force_calibration_assumptions,
                    geometry_hypothesis=self.geometry_hypothesis, provenance=self.provenance,
                    units=dict(t="s", q="rad" if self.joint_type == "revolute" else "m",
                               Q="N m" if self.joint_type == "revolute" else "N", force="N", position="m"),
                    frames=dict(force="base_link", axis="base_link", contact_point="base_link"),
                    record_sha256=self.sha256(),
                    contact_point_assumption="TCP (pointed-tip calibration, DECLARED); no sensor moment used "
                                             "(torque origin not established) -- point-force approximation")
        (out / "joint_interaction_record.json").write_text(json.dumps(meta, indent=1, default=str))
        return meta


def _slip_modes(segments_yaml, src):
    mode = np.full(len(src), "", dtype="U16")
    if not segments_yaml or not Path(segments_yaml).is_file():
        return mode, []
    import yaml
    doc = yaml.safe_load(Path(segments_yaml).read_text()) or {}
    segs = doc.get("segments") or []
    for s in segs:
        m = (src >= int(s["start_source_frame"])) & (src <= int(s["end_source_frame"]))
        mode[m] = str(s.get("slip_mode") or s.get("contact_state") or "contact")
    return mode, [dict(start=int(s["start_source_frame"]), end=int(s["end_source_frame"]),
                       slip_mode=s.get("slip_mode"), provenance=s.get("provenance")) for s in segs]


def from_g9_inputs(inp, *, hypothesis_id="accepted", axis=None, origin=None, q_override=None,
                   contact_segments=None) -> JointInteractionRecord:
    """Canonical record from the existing G9 adapter output (no new preprocessing). ``axis`` / ``origin`` /
    ``q_override`` build a JOINT HYPOTHESIS (the same measured force re-projected; its own q trajectory)."""
    rc = inp.record_context
    if rc is None or not inp.train:
        raise RecordError(f"{inp.scene}: no canonical robot record / interaction window (force data unavailable)")
    rec = rc["rec"]
    mult = rec.get("object_side_multiplier")
    if mult is None or float(mult) not in (-1.0, 1.0):
        raise RecordError("object_side_multiplier absent or invalid in the canonical record")
    jt = inp.joint_type
    a = np.asarray(inp.axis if axis is None else axis, float)
    a = a / np.linalg.norm(a)
    o = None if jt == "prismatic" else np.asarray(inp.origin if origin is None else origin, float)
    src = np.asarray(rec["source_frame_index"], int)
    F = np.asarray(rec["F_object_base_N"], float)
    P = np.asarray(rec["p_tcp_base_m"], float)
    Q = F @ a if jt == "prismatic" else np.einsum("j,nj->n", a, np.cross(P - o, F))
    sf = np.asarray(rec["force_sigma_N"], float)
    sigF = float(np.sqrt(np.mean(sf ** 2))) if sf.ndim == 1 and len(sf) == 3 else sf   # per-axis sigma -> RMS
    lever = np.ones(len(src)) if jt == "prismatic" else np.linalg.norm(np.cross(a, P - o), axis=1)
    Qs = np.maximum(sigF * lever, 1e-9)
    modes, segs = _slip_modes(contact_segments, src)
    state = np.asarray(rec["contact_state"]).astype("U16")
    # the segment slip mode labels FORCE-CONFIRMED contact rows only; force-absent / uncertain rows keep their state
    modes = np.where((state == "contact") & (modes != ""), modes, state)
    if q_override is not None:
        q_src, q, q_std = (np.asarray(x) for x in q_override)
    else:
        q_src, q, q_std = (np.asarray(rc[k]) for k in ("q_src", "q", "q_std"))
    q_src = q_src.astype(int)
    q_std = np.asarray(q_std, float).copy()
    pos = q_std[np.isfinite(q_std) & (q_std > 0)]
    floor = float(np.median(pos)) if len(pos) else 1e-3
    q_std[~np.isfinite(q_std) | (q_std <= 0)] = floor            # e.g. the gauge frame (q = 0 by construction)
    windows = []
    for it in inp.train:
        s0, s1 = it.meta["_record_slice"]
        lo, hi = it.meta["source_rows"]
        k = np.flatnonzero((q_src >= lo) & (q_src <= hi))
        if len(k) >= 6:
            windows.append((it.name, slice(int(s0), int(s1)), slice(int(k[0]), int(k[-1]) + 1), (int(lo), int(hi))))
    r = JointInteractionRecord(
        scene=inp.scene, joint_hypothesis_id=hypothesis_id, joint_type=jt, axis_B=a, origin_B=o,
        q_reference=dict(inp.reference_pose or {"status": "NOT_EVALUATED"}),
        force_t_s=src / 50.0, source_force_row=src, Q_object=Q, Q_sigma=Qs, force_object_B=F, contact_point_B=P,
        contact_mode=modes, in_contact=np.asarray(rec["in_contact_segment"], bool), q_t_s=q_src / 50.0,
        source_q_frame=q_src, q=np.asarray(q, float), q_sigma=np.asarray(q_std, float), windows=windows,
        contact_intervals=segs,
        force_calibration_assumptions={g: v.get("status") for g, v in (inp.force_audit.get("gates") or {}).items()},
        geometry_hypothesis=dict(id=hypothesis_id, axis_B=a.tolist(), origin_B=None if o is None else o.tolist(),
                                 kinematic_status=inp.kinematic_status),
        provenance=dict(object_side_multiplier=float(mult), force_conversion=rec.get("force_conversion"),
                        record_sources=rec.get("sources"), input_hashes=inp.input_hashes,
                        clock="force rows and q joined on source_frame_index (t = row / 50); hardware clock offset "
                              "NOT_ESTABLISHED (scenario variable)"))
    return r.validate()


def from_synthetic(obj, interactions, truths=None, hypothesis_id="synthetic_truth", gap_s=5.0):
    """Record from synthetic episodes (projection-consistent point force at the material contact point). Episodes
    are concatenated on one clock with a gap; windows never cross episodes."""
    from ..synthetic import force_record_for
    ft, rows, Q, Fs, Ps, mode, qt, qq, qs, wins = [], [], [], [], [], [], [], [], [], []
    off, nf, nq = 0.0, 0, 0
    for k, it in enumerate(interactions):
        tf = np.asarray(it.effort.t_s, float)
        lev = np.asarray(it.effort.levels, float)
        qtraj = truths[k]["q"] if truths is not None else it.q_obs
        P, F = force_record_for(obj, tf, lev, it.t_obs, qtraj)
        ft.append(tf + off)
        Q.append(lev)
        Fs.append(F)
        Ps.append(P)
        mode.append(np.where(np.abs(lev) > 0, "sliding", "absent"))
        qt.append(it.t_obs + off)
        qq.append(it.q_obs)
        qs.append(np.broadcast_to(it.q_std, it.q_obs.shape).astype(float))
        wins.append((it.name, slice(nf, nf + len(tf)), slice(nq, nq + len(it.t_obs)), (nf, nf + len(tf) - 1)))
        nf += len(tf)
        nq += len(it.t_obs)
        off = float(max(tf[-1], it.t_obs[-1]) + off + gap_s)
    ftc = np.concatenate(ft)
    r = JointInteractionRecord(
        scene="synthetic", joint_hypothesis_id=hypothesis_id, joint_type=obj.joint_type, axis_B=np.asarray(obj.axis),
        origin_B=None if obj.joint_type == "prismatic" else np.asarray(obj.origin), q_reference=dict(q_ref=0.0),
        force_t_s=ftc, source_force_row=np.arange(len(ftc)), Q_object=np.concatenate(Q),
        Q_sigma=np.full(len(ftc), 0.02), force_object_B=np.concatenate(Fs), contact_point_B=np.concatenate(Ps),
        contact_mode=np.concatenate(mode), in_contact=np.concatenate(mode) == "sliding",
        q_t_s=np.concatenate(qt), source_q_frame=np.arange(nq), q=np.concatenate(qq), q_sigma=np.concatenate(qs),
        windows=wins, provenance=dict(object_side_multiplier=-1.0, synthetic=True))
    return r.validate()
