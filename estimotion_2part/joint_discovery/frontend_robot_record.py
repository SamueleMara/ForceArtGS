"""The canonical robot record (``robot_record.py``) inside the PRE-3DGS joint frontend (G9 corrections section 5).

``align_record_to_frame_map`` takes the 50 Hz native record to the frontend's model times by an EXACT
source_frame_index join (no interpolation; a missing row raises), checks that the record's TCP is the frontend's
TCP row for row (same robot archive, same base frame), and returns the object-side base-frame force ALREADY
converted once by ``force_convention`` -- the frontend never re-reads the raw wrench, never re-tares and never
re-applies the sign when a record is supplied.

Policies (frontend argument, NOT a config field: certified runs keep their config hash and behaviour):
  certified   (default) unchanged: raw wrench + WrenchCalibration gate; the record is only aligned and fingerprinted
  exploratory the record's force may enter candidate fits as CONDITIONAL evidence; everything it influences is
              DIAGNOSTIC; it never sets physics_joint_export_ready and never overwrites a certified decision
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

POLICIES = ("certified", "exploratory")


class RecordAlignmentError(ValueError):
    pass


def record_fingerprint(rec) -> str:
    h = hashlib.sha256()
    for k in ("source_frame_index", "F_object_base_N", "p_tcp_base_m"):
        h.update(np.ascontiguousarray(np.asarray(rec[k]), dtype=np.float64).tobytes())
    h.update(json.dumps(rec.get("force_conversion", {}), sort_keys=True, default=str).encode())
    return h.hexdigest()


def align_record_to_frame_map(rec, frame_map, p_tcp_w=None, T_WB=None, tcp_tol_m=1e-6) -> dict:
    src = np.asarray(rec["source_frame_index"], int)
    idx = {int(s): i for i, s in enumerate(src)}
    want = np.asarray(frame_map.source_frame_index, int)
    missing = [int(s) for s in want if int(s) not in idx]
    if missing:
        raise RecordAlignmentError(f"canonical record lacks source rows {missing[:5]} (no interpolation)")
    sel = np.array([idx[int(s)] for s in want])
    Tw = rec.get("T_world_base")
    if Tw is None:
        raise RecordAlignmentError("record has no T_world_base (cameras not declared in base_link): refusing")
    Tw = np.asarray(Tw, float)
    if T_WB is not None and np.abs(np.asarray(T_WB, float) - Tw).max() > 1e-9:
        raise RecordAlignmentError("record T_world_base differs from the frontend's T_WB")
    R, t = Tw[:3, :3], Tw[:3, 3]
    F_w = np.asarray(rec["F_object_base_N"], float)[sel] @ R.T
    P_w = np.asarray(rec["p_tcp_base_m"], float)[sel] @ R.T + t
    tcp_check = None
    if p_tcp_w is not None:
        d = np.abs(np.asarray(p_tcp_w, float) - P_w)
        ok = np.isfinite(d).all(1)
        mx = float(d[ok].max()) if ok.any() else 0.0
        tcp_check = dict(max_abs_m=mx, rows=int(ok.sum()))
        if mx > tcp_tol_m:
            raise RecordAlignmentError(f"record TCP differs from the frontend TCP by {mx:.3g} m: not the same source row "
                                       "/ archive")
    ts_rec = np.asarray(rec["timestamp_s"], float)[sel]
    return dict(f_object_w=F_w, p_tcp_w=P_w, contact_state=np.asarray(rec["contact_state"])[sel],
                in_contact_segment=np.asarray(rec["in_contact_segment"], bool)[sel],
                recorded_timestamp_s=ts_rec,
                clock=dict(model_time="frame_map.timestamp_s", force_rows="recorded timestamps",
                           max_abs_difference_s=float(np.max(np.abs(ts_rec - np.asarray(frame_map.timestamp_s)))),
                           join="exact source_frame_index"),
                force_sigma_N=np.asarray(rec["force_sigma_N"], float).tolist(), tcp_check=tcp_check,
                force_conversion=rec.get("force_conversion"), fingerprint=record_fingerprint(rec),
                note="force = robot on object, converted ONCE by force_convention (sign from metadata)")
