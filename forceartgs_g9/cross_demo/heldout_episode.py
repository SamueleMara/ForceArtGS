"""Held-out TARGET episode j evaluated under TRAINING episode i (spec section 5), from measured target data only.

Record (`load_real_record` from a finished rest-to-rest run, or `synthetic_record`): the target's whole-recording visual
q (q extension authority), its rest-plateau evidence (initial / terminal windows indexing that q), and its native
wrench rows in the robot base frame B (object side, sign applied once upstream, no second tare).

`build(train, record, alignment)`:
  * generalized effort Q_{j|i}(t) = projection of the TARGET's native wrench onto the TRAINING joint
      revolute   Q = (tau_ref + (p_ref - o_i) x F) . a_i       prismatic   Q = F . a_i
    (never the target's own pre-projected Q when its geometry differs); every native row, no mask, ZOH, no fill;
  * evaluation interval [t0_j, T_eval_end_j]: t0 = start of the target's initial rest plateau (v0 = 0 only because
    that plateau is confirmed), T_stop = first terminal-plateau sample (historical train-only T), T_eval_end = LAST
    sample of the confirmed terminal plateau (the stationary tail is scored). A tail without force rows is
    TAIL_INPUT_INCOMPLETE and only the [t0, T_stop] scope is built;
  * q_ref = target q in the training gauge (kinematics_alignment); it enters the METRICS only. The PhysX Episode carries
    it because its API requires a time grid, but the rollout uses only t_s, q0, v0 and the input (leakage test).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .kinematics_alignment import reference_in_train_gauge

REGIME = "cross_demo_unmasked_measured_input: target's own native effort projected on the TRAINING joint, every row"


def _sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, float)).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ records
def load_real_record(r2r_dir, *, scene=None) -> dict:
    """Everything the target side needs, from ONE finished rest-to-rest run directory (exact path, no wildcard)."""
    d = Path(r2r_dir)
    fw = np.load(d / "full_wrench_joint_effort.npz", allow_pickle=False)
    ep = np.load(d / "rest_to_rest_episode.npz", allow_pickle=False)
    ev = json.loads((d / "rest_endpoint_evidence.json").read_text())
    jt = str(ep["joint_type"])
    rec = dict(scene=scene or str(ep["scene"]), joint_type=jt, source=str(d), ref_kind="RGBD_RECONSTRUCTED",
               axis_B=fw["axis_B"].astype(float), origin_B=fw["origin_B"].astype(float) if jt == "revolute" else None,
               t_q=fw["q_t_s"].astype(float), q=fw["q"].astype(float), q_sigma=fw["q_sigma"].astype(float),
               source_frame=fw["source_q_frame"].astype(int), q_true=None,
               force_t=fw["t_s"].astype(float), source_row=fw["source_row"].astype(int),
               F_B=fw["F_object_B"].astype(float), tau_B=fw["tau_object_B_about_ref"].astype(float),
               p_ref_B=fw["p_ref_B"].astype(float), Q_own=fw["Q_full"].astype(float), Q_rows=None,
               evidence=ev, episode_sha256=str(ep["sha256"]))
    rec["record_sha256"] = _sha(rec["t_q"], rec["q"], rec["force_t"], rec["F_B"], rec["tau_B"], rec["p_ref_B"])
    return rec


def synthetic_record(name, joint_type, axis_B, origin_B, t_q, q, q_sigma, force_t, Q, evidence, *, q_true=None) -> dict:
    """A synthetic target: the effort is already the generalized effort on the (shared, exact) synthetic joint."""
    rec = dict(scene=name, joint_type=joint_type, source="synthetic", ref_kind="TRUE_GT" if q_true is not None
               else "SYNTHETIC_OBSERVED", axis_B=np.asarray(axis_B, float),
               origin_B=None if origin_B is None else np.asarray(origin_B, float),
               t_q=np.asarray(t_q, float), q=np.asarray(q, float), q_sigma=np.asarray(q_sigma, float),
               source_frame=np.arange(len(t_q)), q_true=None if q_true is None else np.asarray(q_true, float),
               force_t=np.asarray(force_t, float), source_row=np.arange(len(force_t)), F_B=None, tau_B=None,
               p_ref_B=None, Q_own=np.asarray(Q, float), Q_rows=np.asarray(Q, float), evidence=evidence,
               episode_sha256=None)
    rec["record_sha256"] = _sha(rec["t_q"], rec["q"], rec["force_t"], rec["Q_own"])
    return rec


def project_effort(rec: dict, joint_type: str, axis_B, origin_B) -> np.ndarray:
    """The target's native wrench rows projected on a given joint (the TRAINING joint for transfer)."""
    if rec["Q_rows"] is not None:                                 # synthetic: already generalized on the shared joint
        return np.asarray(rec["Q_rows"], float).copy()
    a = np.asarray(axis_B, float) / np.linalg.norm(axis_B)
    if joint_type == "prismatic":
        return rec["F_B"] @ a
    o = np.asarray(origin_B, float)
    return (rec["tau_B"] + np.cross(rec["p_ref_B"] - o, rec["F_B"])) @ a


# ------------------------------------------------------------------------------------------------ the held-out episode
@dataclass
class HeldoutEpisode:
    status: str
    reason: str
    scope: str                                    # "eval_end" (primary, includes the stationary tail) | "stop"
    episode: object = None                        # objective.Episode, used_for_training=False
    t_s: np.ndarray = None
    q_ref: np.ndarray = None                      # training gauge, displacement from the target's initial rest
    q_sigma: np.ndarray = None
    q_true: np.ndarray = None                     # synthetic GT in the same gauge (else None)
    source_frame: np.ndarray = None
    input_t_s: np.ndarray = None
    input_Q: np.ndarray = None
    t0: float = None
    T_stop: float = None
    T_eval_end: float = None
    segments: dict = field(default_factory=dict)  # INITIAL_REST / MOVING_INTERVAL / TERMINAL_REST boundaries [s]
    provenance: dict = field(default_factory=dict)


def _fail(status, reason, scope="eval_end", **prov):
    return HeldoutEpisode(status=status, reason=reason, scope=scope, provenance=prov)


def build(train: dict, rec: dict, alignment: dict, *, scope: str = "eval_end", max_gap_factor: float = 1.5):
    """train: dict(joint_type, axis_B, origin_B, name). Returns a HeldoutEpisode (status OK or a precise refusal)."""
    from ..physx_refinement.objective import Episode
    prov = dict(train=train.get("name"), test=rec["scene"], test_record_sha256=rec["record_sha256"],
                test_source=rec["source"], ref_kind=rec["ref_kind"], alignment_status=alignment.get("status"),
                used_for_training=False, input_policy="C1_FULL_MEASURED_UNMASKED projected on the TRAINING joint")
    if alignment.get("status") != "OK":
        return _fail(alignment.get("status", "ALIGNMENT_UNAVAILABLE"), alignment.get("reason", ""), scope, **prov)
    ev = rec["evidence"]
    ini, ter = ev.get("initial") or {}, ev.get("terminal") or {}
    if not ini.get("found"):
        return _fail("INITIAL_REST_NOT_COVERED", "no confirmed initial rest plateau: v0 = 0 would be an assumption",
                     scope, **prov)
    if not ter.get("found"):
        return _fail("TERMINAL_REST_NOT_FOUND", "no confirmed terminal plateau: the evaluation end is not established",
                     scope, **prov)
    tq, q, qs = rec["t_q"], rec["q"], rec["q_sigma"]
    i0 = int(ini["window"][0])
    i_stop, i_end = int(ter["window"][0]), int(ter["window"][1])
    if not (0 <= i0 < i_stop <= i_end < len(tq)):
        return _fail("SCORING_FAILED", f"plateau windows {ini['window']} / {ter['window']} outside the q record", scope,
                     **prov)
    t0, T_stop, T_end = float(tq[i0]), float(tq[i_stop]), float(tq[i_end])
    q0 = float(ini.get("q0", ini.get("test", {}).get("mean")))
    # ---------------------------------------------------------------- input: native rows covering [t0, T]
    ft = rec["force_t"]
    Q = project_effort(rec, train["joint_type"], train["axis_B"], train.get("origin_B"))
    ok = np.isfinite(ft) & np.isfinite(Q)
    dt_row = float(np.median(np.diff(ft[ok]))) if ok.sum() > 2 else np.nan
    T = T_end if scope == "eval_end" else T_stop
    first = np.flatnonzero(ok & (ft <= t0 + 1e-9))
    if not len(first):
        return _fail("INPUT_COVERAGE_INCOMPLETE", f"no native force row at or before t0 = {t0:.3f} s", scope, **prov)
    rows = np.flatnonzero(ok & (ft >= ft[first[-1]] - 1e-12) & (ft <= T + 1e-9))
    tail_ok = len(rows) and ft[rows[-1]] >= T - max_gap_factor * dt_row
    gaps = np.diff(ft[rows]) if len(rows) > 1 else np.array([np.inf])
    if not tail_ok and scope == "eval_end":
        return _fail("TAIL_INPUT_INCOMPLETE", f"force rows end at {ft[rows[-1]] if len(rows) else None} s < "
                     f"T_eval_end {T_end:.3f} s: only the [t0, T_stop] scope is scorable", scope, **prov)
    if not tail_ok or np.any(gaps > max_gap_factor * dt_row):
        return _fail("INPUT_COVERAGE_INCOMPLETE", f"native rows have gaps > {max_gap_factor} x {dt_row:.4f} s in "
                     f"[{t0:.3f}, {T:.3f}] s (no hold, no zero fill)", scope, **prov)
    # ---------------------------------------------------------------- reference in the training gauge (metrics only)
    sel = slice(i0, (i_end if scope == "eval_end" else i_stop) + 1)
    sign = float(alignment["sign"])
    q_ref = reference_in_train_gauge(q[sel], q0, sign)
    q_true = None if rec.get("q_true") is None else reference_in_train_gauge(rec["q_true"][sel], q0, sign)
    sig = np.where(np.isfinite(qs[sel]) & (qs[sel] > 0), qs[sel], np.nanmedian(qs[qs > 0]))
    ep = Episode(name=f"{rec['scene']}|under|{train.get('name')}", t_s=tq[sel], q=q_ref, sigma_q=sig, q0=0.0, v0=0.0,
                 input_t_s=ft[rows], input_effort=Q[rows], mode="forced", calibration_status="GIVEN_EXPERIMENTAL_FACTS",
                 used_for_training=False, exploratory_assumptions_accepted=True, sigma_v0=None, regime_semantics=REGIME)
    segs = dict(INITIAL_REST=(t0, float(tq[int(ini["window"][1])])), MOVING_INTERVAL=(float(tq[int(ini["window"][1])]),
                T_stop), TERMINAL_REST=(T_stop, T_end) if scope == "eval_end" and i_end > i_stop else None)
    prov.update(t0=t0, T_stop=T_stop, T_eval_end=T_end, q0_test_raw=q0, sign=sign, n_q=int(len(q_ref)),
                n_rows=int(len(rows)), row_dt_s=dt_row, input_sha256=_sha(ft[rows], Q[rows]),
                reference_sha256=_sha(tq[sel], q_ref), source_rows=[int(rec["source_row"][rows[0]]),
                                                                      int(rec["source_row"][rows[-1]])])
    return HeldoutEpisode(status="OK", reason="", scope=scope, episode=ep, t_s=tq[sel], q_ref=q_ref, q_sigma=sig,
                          q_true=q_true, source_frame=rec["source_frame"][sel], input_t_s=ft[rows], input_Q=Q[rows],
                          t0=t0, T_stop=T_stop, T_eval_end=T_end, segments=segs, provenance=prov)
