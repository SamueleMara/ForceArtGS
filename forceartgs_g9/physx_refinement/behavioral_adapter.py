"""Canonical robot record + joint authority -> ONE full-demonstration behavioral episode (G10-A).

This path never calls ``g9_adapter.interactions_from_segments`` (that legacy path crops to classifier windows, estimates
v0 from a 4-sample prefix and starts the replay at the last prefix sample). Here:

* t0 = the first authority observation; q0 = q_authority(t0) in the authority coordinate (never re-zeroed);
  v0 = 0.0 exactly (the user's rest contract), justified by one of two declared bases:
    - START_FULL_RECORDING: the authority covers the recording start itself (no q-sampling interval missing);
    - START_REST_VERIFIED_CROP (explicit opt-in): the authority starts later, the force record covers t0, and the first
      authority samples are CONSISTENT with rest (a hypothesis test of v = 0, never an estimate used as v0).
  Otherwise FULL_START_UNAVAILABLE with the exact uncovered source frames -- no rest state is invented.
* the recorded canonical Q_object rows are applied from the last row <= t0 to the end, sign unchanged; a row gap or a
  non-finite row over the replay is INPUT_COVERAGE_INCOMPLETE (no hold / zero fill).
* every force row carries a contact-evidence label; labels drive claims and input hypotheses, never a zero-input
  substitution of the recorded effort.
"""
from __future__ import annotations

import math

import numpy as np

from .behavioral_contracts import (START_FULL_RECORDING, START_REST_VERIFIED_CROP, FullDemoBehavioralEpisode,
                                   InputHypothesis)

CONFIRMED_MODES = ("contact", "sliding", "sticking", "stick", "slip", "slide")


def _nominal_dt(t):
    d = np.diff(np.asarray(t, float))
    return float(np.median(d)) if len(d) else math.nan


def input_onset(ft, Q, Qs, t0, *, k_sigma, n_rows):
    """First row time >= t0 starting n_rows consecutive rows with |Q| > k sigma_Q (None: no onset)."""
    ft, Q, Qs = (np.asarray(x, float) for x in (ft, Q, Qs))
    m = (ft >= t0 - 1e-12)
    big = (np.abs(Q) > k_sigma * Qs) & m
    run = 0
    for i in np.flatnonzero(m):
        run = run + 1 if big[i] else 0
        if run >= n_rows:
            return float(ft[i - n_rows + 1])
    return None


def rest_consistency(t, q, s, *, window_s, min_samples, k_sigma, t_onset=None) -> dict:
    """Hypothesis test of v(t0) = 0 on the authority samples BEFORE the measured input onset (and within window_s):
    weighted LS q = c + v (t - t0) + a (t - t0)^2 / 2 (a constant acceleration absorbs a gravity load), |v_hat| <=
    k sd(v_hat). From q alone, 'already moving' cannot be told from 'accelerating from rest under an active input',
    so samples after the onset are never used. The estimate is NEVER used as v0."""
    t, q, s = (np.asarray(x, float) for x in (t, q, s))
    end = window_s if t_onset is None else min(window_s, t_onset - t[0] - 1e-9)
    m = (t - t[0]) <= end + 1e-12
    n = int(m.sum())
    out = dict(test="weighted LS q = c + v dt + a dt^2/2 on the authority samples before the measured input onset; "
                    "H0: v = 0", window_s=window_s, t_input_onset=t_onset, window_used_s=max(end, 0.0), n_samples=n,
               k_sigma=k_sigma, used_as_v0=False)
    if n < max(int(min_samples), 4):
        return dict(out, status="UNCHECKABLE", reason=f"{n} authority sample(s) before the input onset / within "
                    f"{window_s} s (need >= {max(int(min_samples), 4)}): rest cannot be tested from q while the "
                    "recorded input is already active")
    dt = t[m] - t[0]
    A = np.column_stack([np.ones(n), dt, 0.5 * dt * dt]) / s[m, None]
    b = q[m] / s[m]
    C = np.linalg.pinv(A.T @ A)
    x = C @ A.T @ b
    r = b - A @ x
    dof = max(n - 3, 1)
    chi2r = float(r @ r) / dof
    sd_v = math.sqrt(C[1, 1] * max(1.0, chi2r))
    z = float(x[1] / sd_v) if sd_v > 0 else math.inf
    ok = abs(z) <= k_sigma
    return dict(out, status="CONSISTENT_WITH_REST" if ok else "MOVING_AT_START", v_hat=float(x[1]), sd_v=sd_v, z_v=z,
                chi2_red=chi2r, max_abs_dq=float(np.max(np.abs(q[m] - q[0]))))


def input_coverage(force_t, Q, t0, t_end, *, max_gap_factor) -> dict:
    """Rows [last row <= t0, last row <= t_end]; refuses a gap (> factor x nominal row period), a non-finite row, a
    missing row at t0 or an effort record ending before t_end. Nothing is held / filled / extrapolated."""
    force_t, Q = np.asarray(force_t, float), np.asarray(Q, float)
    dt = _nominal_dt(force_t)
    i0 = int(np.searchsorted(force_t, t0 + 1e-12, side="right") - 1)
    i1 = int(np.searchsorted(force_t, t_end + 1e-12, side="right") - 1)
    missing = []
    if i0 < 0:
        missing.append(dict(kind="NO_EFFORT_AT_T0", t=[float(t0), float(force_t[0]) if len(force_t) else None]))
        i0 = 0
    if len(force_t) == 0 or force_t[i1] < t_end - max_gap_factor * dt:
        missing.append(dict(kind="EFFORT_ENDS_BEFORE_LAST_OBSERVATION",
                            t=[float(force_t[i1]) if len(force_t) else None, float(t_end)]))
    seg_t = force_t[i0:i1 + 1]
    gaps = np.flatnonzero(np.diff(seg_t) > max_gap_factor * dt)
    missing += [dict(kind="ROW_GAP", t=[float(seg_t[g]), float(seg_t[g + 1])]) for g in gaps]
    bad = np.flatnonzero(~np.isfinite(Q[i0:i1 + 1]))
    missing += [dict(kind="NONFINITE_EFFORT_ROW", t=[float(seg_t[b]), float(seg_t[b])]) for b in bad]
    return dict(status="COMPLETE" if not missing else "INPUT_COVERAGE_INCOMPLETE", row_range=[i0, i1],
                nominal_row_dt_s=dt, t=[float(t0), float(t_end)], missing=missing,
                rule=f"forward ZOH on native rows; gaps > {max_gap_factor} x row period, non-finite rows or no row "
                     "at t0 are refused (no hold, no zero fill, no extrapolation)")


def contact_labels(modes, Q, Qs, *, near_zero_k) -> np.ndarray:
    modes = np.asarray(modes).astype(str)
    lab = np.empty(len(modes), dtype="U32")
    seen = False
    for i, m in enumerate(modes):
        ml = m.lower()
        if any(c in ml for c in CONFIRMED_MODES) and ml not in ("absent", "uncertain"):
            lab[i] = "MEASURED_NEAR_ZERO_CONTACT" if abs(Q[i]) < near_zero_k * Qs[i] else "CONTACT_CONFIRMED"
            seen = True
        elif ml == "absent":
            lab[i] = "RELEASE_CONFIRMED" if seen else "NO_CONTACT_MEASURED"
        else:
            lab[i] = "CONTACT_UNKNOWN"
    return lab


def _runs(lab, t, rows, Q):
    out, i = [], 0
    while i < len(lab):
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        out.append(dict(label=str(lab[i]), rows=[int(rows[i]), int(rows[j])], t=[float(t[i]), float(t[j])],
                        n_rows=int(j - i + 1), max_abs_effort=float(np.max(np.abs(Q[i:j + 1])))))
        i = j + 1
    return out


def build_full_demo_episode(rec, cfg, *, kinematics_evidence=None, calibration_evidence=None, provenance=None) -> dict:
    """``rec``: a forcetwin JointInteractionRecord (or the same attributes). Returns dict(status, reason, episode,
    evidence); status OK | FULL_START_UNAVAILABLE | INPUT_COVERAGE_INCOMPLETE | DATA_UNAVAILABLE."""
    ini, inc = cfg["initial_state"], cfg["input"]
    tq = np.asarray(rec.q_t_s, float)
    q = np.asarray(rec.q, float)
    qs = np.asarray(rec.q_sigma, float)
    src_q = np.asarray(rec.source_q_frame, int)
    ok = np.isfinite(tq) & np.isfinite(q) & np.isfinite(qs) & (qs > 0)
    tq, q, qs, src_q = tq[ok], q[ok], qs[ok], src_q[ok]
    rows = np.asarray(rec.source_force_row, int)
    ft = np.asarray(rec.force_t_s, float)
    Q = np.asarray(rec.Q_object, float)
    Qs = np.asarray(rec.Q_sigma, float)
    ev = dict(invalid_q_samples_excluded=int((~ok).sum()))
    if len(tq) < 3 or len(ft) == 0:
        return dict(status="DATA_UNAVAILABLE", reason="fewer than 3 valid authority samples or no effort rows",
                    episode=None, evidence=ev)
    rec_start = int(rows.min())
    auth_start = int(src_q[0])
    q_step = int(np.median(np.diff(src_q))) if len(src_q) > 1 else 1
    full = (auth_start - rec_start) < q_step
    ev.update(recording_start_frame=rec_start, joint_authority_start_frame=auth_start, q_sampling_rows=q_step,
              t0_s=float(tq[0]), start_is_full_demo=bool(full),
              uncovered_source_frames=None if full else [rec_start, auth_start - 1],
              start_frame_evidence="min(source_force_row) of the canonical robot record vs the first valid joint "
                                   "authority source_q_frame (both on source_frame_index; t = frame / 50)")
    t_on = input_onset(ft, Q, Qs, float(tq[0]), k_sigma=float(ini["rest_check_input_onset_k_sigma"]),
                       n_rows=int(ini["rest_check_input_onset_rows"]))
    rest = rest_consistency(tq, q, qs, window_s=float(ini["rest_check_window_s"]),
                            min_samples=int(ini["rest_check_min_samples"]), k_sigma=float(ini["rest_check_k_sigma"]),
                            t_onset=t_on)
    ev["rest_check"] = rest
    if ini["mode"] == "full_demonstration_rest" and not full:
        return dict(status="FULL_START_UNAVAILABLE", episode=None, evidence=ev, reason=(
            f"the joint authority starts at source frame {auth_start}, the recording at {rec_start}: frames "
            f"[{rec_start}, {auth_start - 1}] are not covered (cropped clip). The full-demonstration rest start "
            "cannot be asserted; rest_verified_cropped_start is the explicit opt-in"))
    if rest["status"] == "MOVING_AT_START":
        return dict(status="FULL_START_UNAVAILABLE", episode=None, evidence=ev, reason=(
            f"the object is already moving at the first authority sample (v test z = {rest['z_v']:.3g}, "
            f"k = {rest['k_sigma']}): no rest initial state is invented"))
    if rest["status"] == "UNCHECKABLE" and not full:
        return dict(status="FULL_START_UNAVAILABLE", episode=None, evidence=ev, reason=(
            f"cropped authority start and the rest test is not possible: {rest['reason']}"))
    basis = START_FULL_RECORDING if full else START_REST_VERIFIED_CROP
    cov = input_coverage(ft, Q, float(tq[0]), float(tq[-1]), max_gap_factor=float(inc["max_row_gap_factor"]))
    ev["input_coverage"] = cov
    if cov["status"] != "COMPLETE":
        return dict(status="INPUT_COVERAGE_INCOMPLETE", episode=None, evidence=ev,
                    reason=f"effort coverage incomplete over the replay: {cov['missing']}")
    i0, i1 = cov["row_range"]
    sl = slice(i0, i1 + 1)
    lab = contact_labels(np.asarray(rec.contact_mode)[sl], Q[sl], Qs[sl], near_zero_k=float(inc["near_zero_k_sigma"]))
    runs = _runs(lab, ft[sl], rows[sl], Q[sl])
    counts = {k: int((lab == k).sum()) for k in np.unique(lab)}
    contact = dict(labels_per_row=lab.tolist(), intervals=runs, counts=counts,
                   conditional_rows=int(sum(v for k, v in counts.items() if k != "CONTACT_CONFIRMED")),
                   note="labels are evidence for claims and input hypotheses; the recorded Q_object is applied on "
                        "EVERY row (never replaced by a free zero effort)")
    prov = dict(provenance or {})
    prov.update(record_sha256=rec.sha256() if hasattr(rec, "sha256") else None,
                clock="force rows and q joined on source_frame_index (t = frame / 50)",
                force_sign="canonical object_side_multiplier applied once upstream; unchanged here")
    ep = FullDemoBehavioralEpisode(
        scene=str(rec.scene), joint_type=str(rec.joint_type), t0_s=float(tq[0]), source_start_frame=auth_start,
        recording_start_frame=rec_start, t_q_s=tq, source_q_frame=src_q, q_observed=q, q_sigma=qs,
        q0_observed=float(q[0]), v0_fixed=0.0, effort_t_s=ft[sl], effort_source_row=rows[sl], effort_object=Q[sl],
        effort_sigma=Qs[sl], contact_evidence=contact, kinematics_evidence=dict(kinematics_evidence or {}),
        calibration_evidence=dict(calibration_evidence or {}), start_basis=basis, full_demo_start_verified=True,
        start_is_full_demo=bool(full), provenance=prov)
    return dict(status="OK", reason=f"full-demo episode from t0 = {tq[0]:.4g} s (source frame {auth_start}), "
                f"q0 = {q[0]:.6g}, v0 = 0 ({basis})", episode=ep, evidence=ev)


def input_hypotheses(rec, ep: FullDemoBehavioralEpisode, cfg, *, seed=0) -> list:
    """Grounded alternatives of the recorded input on the SAME rows (never a sign flip, never fabricated rows):
    clock offsets by whole rows (from the full record outside the replay; skipped if out of range), effort-noise
    realizations from the recorded Q_sigma, and the contact interpretation 'force-absent rows are sensor noise'."""
    inc = cfg["input"]
    out = [InputHypothesis("recorded", "recorded", ep.effort_object, "canonical recorded Q_object")]
    skipped = []
    if not inc["uncertainty_sensitivity"]:
        return out, skipped
    rows_all = np.asarray(rec.source_force_row, int)
    Q_all = np.asarray(rec.Q_object, float)
    pos = {int(r): i for i, r in enumerate(rows_all)}
    idx = np.array([pos[int(r)] for r in ep.effort_source_row])
    for s in inc["time_offset_rows"]:
        j = idx - int(s)
        if j.min() < 0 or j.max() >= len(Q_all):
            skipped.append(dict(name=f"time_offset_{int(s):+d}_rows", reason="shift leaves the recorded rows"))
            continue
        out.append(InputHypothesis(f"time_offset_{int(s):+d}_rows", "time_offset", Q_all[j],
                                   f"input delayed by {int(s)} row(s) ({int(s) * 20} ms): robot/camera clock offset "
                                   "NOT_ESTABLISHED"))
    rng = np.random.default_rng(seed)
    if ep.effort_sigma is not None:
        for k in range(int(inc["noise_realizations"])):
            out.append(InputHypothesis(f"noise_realization_{k}", "noise_realization",
                                       ep.effort_object + rng.standard_normal(len(ep.effort_object)) * ep.effort_sigma,
                                       "recorded Q + N(0, Q_sigma) (measured wrench noise)"))
    if inc["contact_interpretation_hypotheses"]:
        lab = np.asarray(ep.contact_evidence.get("labels_per_row") or [], str)
        m = np.isin(lab, ("NO_CONTACT_MEASURED", "RELEASE_CONFIRMED"))
        if m.any():
            Qh = ep.effort_object.copy()
            Qh[m] = 0.0
            out.append(InputHypothesis("absent_rows_are_sensor_noise", "contact_interpretation", Qh,
                                       f"{int(m.sum())} force-absent row(s) read as zero object-side effort (a "
                                       "labelled interpretation, NOT the recorded input)"))
        else:
            skipped.append(dict(name="absent_rows_are_sensor_noise", reason="no force-absent rows in the replay"))
    return out, skipped


# =============================================================================== wrench-driven path (schema v2)
def load_interaction_segments(path) -> list:
    """The EXISTING force-peak + visual contact segments (configs/contact_intervals/<scene>.yaml) as
    [(start_source_frame, end_source_frame)] -- read only, never re-estimated."""
    import yaml
    from pathlib import Path
    doc = yaml.safe_load(Path(path).read_text()) or {}
    segs = [(int(s["start_source_frame"]), int(s["end_source_frame"])) for s in (doc.get("segments") or [])]
    if not segs:
        raise ValueError(f"{path}: no segments")
    return sorted(segs)


def interaction_mask(rows, segments) -> np.ndarray:
    rows = np.asarray(rows, int)
    m = np.zeros(len(rows), bool)
    for a, b in segments:
        m |= (rows >= a) & (rows <= b)
    return m


def build_wrench_episode(fr, cfg, segments) -> dict:
    """ONE interaction episode (facts: contact ON during the interaction given by the EXISTING force-peak + visual
    segments, OFF outside; qdot = 0 at the interaction start; q0 from vision). No new classification, no rest test,
    no prefix velocity:
      t0  = the last valid visual sample at or before the first segment start (the object is at rest before contact);
      Q   = Q_full (six-axis wrench projection) on rows inside a segment, exactly 0 outside (contact off);
      end = the last visual sample (post-release coast, zero effort) or the last one inside the interaction.
    Status OK | FULL_START_NOT_AVAILABLE | INPUT_COVERAGE_ERROR | DATA_UNAVAILABLE."""
    from .behavioral_contracts import START_EXPERIMENT_KNOWN_ZERO
    tq, q, qs, src_q = (np.asarray(x) for x in (fr.q_t_s, fr.q, fr.q_sigma, fr.source_q_frame))
    ok = np.isfinite(tq) & np.isfinite(q) & np.isfinite(qs) & (qs > 0)
    tq, q, qs, src_q = tq[ok].astype(float), q[ok].astype(float), qs[ok].astype(float), src_q[ok].astype(int)
    segs = sorted((int(a), int(b)) for a, b in segments)
    s0, s1 = segs[0][0], segs[-1][1]
    ev = dict(invalid_visual_samples_excluded=int((~ok).sum()), interaction_segments=[list(x) for x in segs],
              interaction_source="existing force-peak + visual contact segments (read only)",
              contact="ON inside the interaction segments, OFF (zero object-side effort) outside",
              initial_velocity_source=START_EXPERIMENT_KNOWN_ZERO,
              recording_start_row=int(fr.source_row.min()) if len(fr.source_row) else None)
    if len(tq) < 3:
        return dict(status="DATA_UNAVAILABLE", reason="fewer than 3 valid visual joint samples", episode=None,
                    evidence=ev)
    before = np.flatnonzero(src_q <= s0)
    if len(before) == 0:
        return dict(status="FULL_START_NOT_AVAILABLE", episode=None,
                    evidence=dict(ev, uncovered_source_frames=[s0, int(src_q[0]) - 1]),
                    reason=f"the interaction starts at source frame {s0} but the visual joint trajectory starts at "
                           f"{int(src_q[0])}: no rest replay from a frame without a visual q is fabricated")
    i_start = int(before[-1])
    end_rows = src_q if cfg["input"]["include_post_interaction_coast"] else src_q[src_q <= s1]
    i_end = int(np.flatnonzero(src_q == end_rows[-1])[0])
    if i_end - i_start + 1 < 3:
        return dict(status="DATA_UNAVAILABLE", reason="fewer than 3 visual samples in the interaction window",
                    episode=None, evidence=ev)
    sel = slice(i_start, i_end + 1)
    tq, q, qs, src_q_w = tq[sel], q[sel], qs[sel], src_q[sel]
    ev.update(t0_s=float(tq[0]), visual_start_frame=int(src_q_w[0]), visual_end_frame=int(src_q_w[-1]),
              n_visual_samples=int(len(tq)), pre_interaction_visual_samples_not_replayed=int(i_start),
              post_interaction_coast_samples=int(np.sum(src_q_w > s1)))
    rows = np.asarray(fr.source_row, int)
    m_in = interaction_mask(rows, segs)
    Q = np.where(m_in, np.asarray(fr.Q_full, float), 0.0)            # contact off -> exactly zero
    cov = input_coverage(fr.t_s, np.where(m_in, Q, 0.0), float(tq[0]), float(tq[-1]),
                         max_gap_factor=float(cfg["input"]["max_row_gap_factor"]))
    ev["input_coverage"] = cov
    if cov["status"] != "COMPLETE":
        if any(m["kind"] == "NO_EFFORT_AT_T0" for m in cov["missing"]):
            return dict(status="FULL_START_NOT_AVAILABLE", episode=None, evidence=ev,
                        reason=f"no recorded wrench row at or before t0 = {tq[0]:.4g} s")
        return dict(status="INPUT_COVERAGE_ERROR", episode=None, evidence=ev,
                    reason=f"recorded wrench missing / non-finite inside the interaction: {cov['missing']}")
    i0, i1 = cov["row_range"]
    sl = slice(i0, i1 + 1)
    m_w = m_in[sl]
    ev.update(rows_in_contact=int(m_w.sum()), rows_contact_off_zero_effort=int((~m_w).sum()))
    Qs = np.where(m_w, np.asarray(fr.Q_sigma, float)[sl], 1e-9)
    ep = FullDemoBehavioralEpisode(
        scene=str(fr.scene), joint_type=str(fr.joint_type), t0_s=float(tq[0]), source_start_frame=int(src_q_w[0]),
        recording_start_frame=int(fr.source_row.min()), t_q_s=tq, source_q_frame=src_q_w, q_observed=q, q_sigma=qs,
        q0_observed=float(q[0]), v0_fixed=0.0, effort_t_s=np.asarray(fr.t_s, float)[sl],
        effort_source_row=rows[sl], effort_object=Q[sl], effort_sigma=Qs,
        contact_evidence=dict(mode="ON_DURING_INTERACTION", segments=[list(x) for x in segs], labels_per_row=None,
                              in_contact_per_row=m_w.tolist(),
                              note="existing force-peak + visual segments; recorded wrench inside (every row, low "
                                   "values included), zero effort outside (contact off)"),
        kinematics_evidence={}, calibration_evidence=dict(force_audit_status="GIVEN_EXPERIMENTAL_FACTS",
                                                          transform="known (E1)", tare="already tared (E2)"),
        start_basis=START_EXPERIMENT_KNOWN_ZERO, full_demo_start_verified=True, start_is_full_demo=True,
        provenance=dict(full_wrench_record_sha256=fr.sha256(), input="full six-axis wrench joint-screw projection",
                        interaction_segments=[list(x) for x in segs]))
    return dict(status="OK", reason=f"interaction {segs} -> t0 = {tq[0]:.4g} s (visual frame {int(src_q_w[0])}), q0 = "
                f"{q[0]:.6g}, v0 = 0 (EXPERIMENT_KNOWN_ZERO); {len(tq)} visual samples "
                f"({ev['post_interaction_coast_samples']} post-release), {int(m_w.sum())} wrench rows in contact, "
                f"{int((~m_w).sum())} contact-off rows at zero effort", episode=ep, evidence=ev)


def wrench_input_hypotheses(fr, ep, cfg, *, seed=0):
    """Grounded alternatives of the SAME recorded wrench input on the SAME interaction (never a sign flip, never a
    contact re-interpretation): whole-row clock offsets of the wrench from the full record, and noise realizations of
    Q_sigma -- both only on in-contact rows (contact off stays exactly zero)."""
    inc = cfg["input"]
    out = [InputHypothesis("recorded", "recorded", ep.effort_object, "full six-axis wrench projection as recorded")]
    skipped = []
    if not inc["uncertainty_sensitivity"]:
        return out, skipped
    m = np.asarray(ep.contact_evidence.get("in_contact_per_row") or np.ones(len(ep.effort_object), bool), bool)
    rows = np.asarray(fr.source_row, int)
    pos = {int(r): i for i, r in enumerate(rows)}
    idx = np.array([pos[int(r)] for r in ep.effort_source_row])
    for s in inc["time_offset_rows"]:
        j = idx - int(s)
        if j.min() < 0 or j.max() >= len(rows) or not np.isfinite(np.asarray(fr.Q_full)[j][m]).all():
            skipped.append(dict(name=f"time_offset_{int(s):+d}_rows", reason="shift leaves the recorded rows"))
            continue
        out.append(InputHypothesis(f"time_offset_{int(s):+d}_rows", "time_offset",
                                   np.where(m, np.asarray(fr.Q_full)[j], 0.0),
                                   f"wrench delayed by {int(s)} row(s) ({int(s) * 20} ms) inside the interaction: "
                                   "sensitivity to the shared row clock (no re-synchronization)"))
    rng = np.random.default_rng(seed)
    for k in range(int(inc["noise_realizations"])):
        out.append(InputHypothesis(f"noise_realization_{k}", "noise_realization",
                                   np.where(m, ep.effort_object + rng.standard_normal(len(m)) * ep.effort_sigma, 0.0),
                                   "recorded Q + N(0, Q_sigma) on in-contact rows (wrench noise propagated)"))
    return out, skipped
