"""External-effort interval evidence (spec section 2.3): FREE_CONFIRMED / FORCED_MEASURED / UNKNOWN_OR_UNMEASURED,
never a single contact boolean and never "low force => free".

Per native force row (50 Hz, the canonical robot record -- no re-tare, no second sign conversion):
  * effort significance  : running RMS of |Q|/sigma_Q over ``win_rows`` rows > ``z_sig``
  * contact evidence     : the front-end contact segment (force peaks + visual; ``in_contact``) OR a force-detected
                           contact / sliding / sticking state
  * release evidence     : outside every annotated contact segment AND the force detector does not report contact
                           (``absent`` / ``uncertain``) -- OR geometric TCP separation from the moving part (part
                           points supplied) exceeding the separation measured DURING force-confirmed contact by more
                           than ``sep_margin_m`` (the hull / TCP offset is calibrated out, never assumed zero)
  * TCP-follows-part     : distance between the TCP and the material point it touched at the interval start, carried
                           with q -- a diagnostic that the tool moves WITH the part (contact plausible)

Row label:
  released & effort noise-consistent              -> FREE_CONFIRMED
  released & effort significant                   -> UNKNOWN (EFFORT_WITHOUT_ANNOTATED_CONTACT)
  contact  & effort significant & not sticking    -> FORCED_MEASURED
  contact  & sticking                             -> UNKNOWN (STICK_INTERVAL: set-valued, not moving-mode data)
  contact  & effort at the noise floor            -> UNKNOWN (LOW_MEASURED_EFFORT_IN_ANNOTATED_CONTACT)
  non-finite wrench                               -> UNKNOWN (ABSENT_WRENCH)

An UNKNOWN interval is never used as free data -- except under an EXPLICITLY NAMED free hypothesis
(``free_hypotheses``), whose every downstream result is capped at CONDITIONAL_ON_FREE_HYPOTHESIS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

LABELS = ("FREE_CONFIRMED", "FORCED_MEASURED", "UNKNOWN_OR_UNMEASURED")
HYPOTHESES = {"measured_near_zero_effort_in_contact": "LOW_MEASURED_EFFORT_IN_ANNOTATED_CONTACT"}
CONTACT_MODES = ("contact", "sliding", "sticking", "stick")


@dataclass
class MotionSegment:
    """One contiguous, MONOTONIC moving interval (fixed velocity sign) with its q samples and measured effort."""

    name: str
    label: str                      # FREE_CONFIRMED | FORCED_MEASURED | FREE_HYPOTHESIS:<name>
    joint_type: str
    t: np.ndarray
    q: np.ndarray
    q_sigma: np.ndarray
    sign: float
    force_t: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    Q_sigma: Optional[np.ndarray] = None
    evidence: dict = field(default_factory=dict)
    q_sigma_reported: Optional[np.ndarray] = None   # the upstream sigma, kept when q_sigma was inflated

    @property
    def is_free(self):
        return self.label == "FREE_CONFIRMED" or self.label.startswith("FREE_HYPOTHESIS")

    @property
    def effort(self):
        from ..synthetic import ZOH
        if self.Q is None:
            return None
        return ZOH(self.force_t, self.Q)

    def J(self, ta, tb):
        """Exact ZOH impulse and its sigma on [ta, tb] (native rows)."""
        if self.Q is None:
            return 0.0, 0.0
        tf = np.asarray(self.force_t, float)
        dt = np.diff(np.r_[tf, tf[-1] + (np.median(np.diff(tf)) if len(tf) > 1 else 0.02)])
        ov = np.clip(np.minimum(tf + dt, tb) - np.maximum(tf, ta), 0, None)
        return float(np.sum(self.Q * ov)), float(math.sqrt(np.sum((self.Q_sigma * ov) ** 2)))

    def split(self, frac):
        """Chronological split at ``frac`` of the duration -> (head, tail) (tail may be None)."""
        tc = self.t[0] + frac * (self.t[-1] - self.t[0])
        k = int(np.searchsorted(self.t, tc, side="right"))
        if k < 5 or len(self.t) - k < 2:
            return self, None
        rep = self.q_sigma_reported
        mk = lambda sl, nm: MotionSegment(nm, self.label, self.joint_type, self.t[sl], self.q[sl], self.q_sigma[sl],
                                          self.sign, self.force_t, self.Q, self.Q_sigma, dict(self.evidence),
                                          None if rep is None else rep[sl])
        return mk(slice(0, k), self.name + "_head"), mk(slice(k - 1, None), self.name + "_tail")

    def describe(self):
        u = "rad" if self.joint_type == "revolute" else "m"
        return dict(name=self.name, label=self.label, n_q=len(self.t), t=[float(self.t[0]), float(self.t[-1])],
                    duration_s=float(self.t[-1] - self.t[0]), dq=float(self.q[-1] - self.q[0]), unit=u,
                    sign=self.sign, median_q_sigma=float(np.median(self.q_sigma)),
                    median_q_sigma_reported=None if self.q_sigma_reported is None else
                    float(np.median(self.q_sigma_reported)),
                    has_measured_effort=self.Q is not None, evidence=self.evidence)


def empirical_q_sigma(t, q, k=5):
    """Pooled noise sd from local quadratic fits on k consecutive samples (k - 3 dof each). Curvature below the
    sample scale is absorbed by the quadratic, so this measures the q measurement scatter, not the motion."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    if len(t) < k:
        return None
    rss = []
    for i in range(len(t) - k + 1):
        x = t[i:i + k] - t[i + k // 2]
        A = np.vander(x, 3, increasing=True)
        b = np.linalg.lstsq(A, q[i:i + k], rcond=None)[0]
        rss.append(float(np.sum((q[i:i + k] - A @ b) ** 2)) / (k - 3))
    return float(np.sqrt(np.mean(rss)))


def inflate_noise(seg):
    """q_sigma := max(reported, empirical) with both recorded (spec 2.4 / 7.1: preserve the measured variance,
    transparent floor -- never shrink it)."""
    emp = empirical_q_sigma(seg.t, seg.q)
    rep = np.asarray(seg.q_sigma, float)
    seg.evidence["q_noise"] = dict(reported_median=float(np.median(rep)), empirical=emp,
                                   method="pooled residual sd of local quadratic fits on 5 consecutive samples",
                                   inflated=bool(emp is not None and emp > np.median(rep)))
    if emp is not None and emp > 0:
        seg.q_sigma_reported = rep.copy()
        seg.q_sigma = np.maximum(rep, emp)
    return seg


def _running_rms(z, w):
    if w <= 1:
        return np.abs(z)
    k = np.ones(w) / w
    return np.sqrt(np.convolve(z ** 2, k, mode="same"))


def _velocity(rec):
    from ..forcetwin.derivatives import spline
    if len(rec.q_t_s) < 5:
        return np.zeros(len(rec.q_t_s)), np.full(len(rec.q_t_s), np.inf)
    d = spline(rec.q_t_s, rec.q, rec.q_sigma, n_mc=30)
    return d["qd"], np.maximum(d["qd_sd"], 1e-9)


def _tcp_tracking(rec, i0, i1):
    """TCP distance to the material point it touched at row i0, carried with q (diagnostic)."""
    from ..dynamics.general_1dof import rotation
    P = rec.contact_point_B
    if not np.isfinite(P[i0:i1 + 1]).all():
        return None
    qi = np.interp(rec.force_t_s[i0:i1 + 1], rec.q_t_s, rec.q)
    p0 = P[i0]
    if rec.joint_type == "revolute":
        o, a = np.asarray(rec.origin_B, float), np.asarray(rec.axis_B, float)
        pm = np.array([o + rotation(a, dq) @ (p0 - o) for dq in qi - qi[0]])
    else:
        pm = p0 + np.outer(qi - qi[0], rec.axis_B)
    d = np.linalg.norm(P[i0:i1 + 1] - pm, axis=1)
    moved = np.linalg.norm(P[i0:i1 + 1] - p0, axis=1)
    return dict(max_tcp_to_carried_point_m=float(d.max()), median_tcp_to_carried_point_m=float(np.median(d)),
                tcp_travel_m=float(moved.max()), part_point_travel_m=float(np.linalg.norm(pm[-1] - p0)))


def _separation(rec, part_points_ref, q_ref):
    """Min TCP distance to the part's sample points carried to q(t) (geometric release evidence)."""
    from ..dynamics.general_1dof import rotation
    if part_points_ref is None:
        return None
    X = np.asarray(part_points_ref, float)
    qi = np.interp(rec.force_t_s, rec.q_t_s, rec.q)
    out = np.empty(len(qi))
    for k, (qq, p) in enumerate(zip(qi, rec.contact_point_B)):
        if rec.joint_type == "revolute":
            o, a = np.asarray(rec.origin_B, float), np.asarray(rec.axis_B, float)
            Y = o + (X - o) @ rotation(a, qq - q_ref).T
        else:
            Y = X + (qq - q_ref) * np.asarray(rec.axis_B, float)
        out[k] = float(np.min(np.linalg.norm(Y - p, axis=1)))
    return out


def _merge_contact_runs(sig, z, Q, Qs, mask, *, min_run=10, pulse_z=10.0, z_J=5.0, dt=0.02):
    """Inside annotated (non-sticking) contact, row-level significance fragments a sustained weak push. Runs shorter
    than ``min_run`` rows are merged into their neighbours (a sharp pulse -- peak |Q|/sigma >= ``pulse_z`` -- stays
    its own run); each merged run is FORCED if |J|/sigma_J >= ``z_J`` or it is a pulse, else low-effort."""
    out = sig.copy()
    idx = np.flatnonzero(mask)
    if not len(idx):
        return out, []
    blocks, st = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            blocks.append((st, a))
            st = b
    blocks.append((st, idx[-1]))
    decisions = []
    for a, b in blocks:
        runs = []
        k = a
        while k <= b:
            e = k
            while e + 1 <= b and sig[e + 1] == sig[k]:
                e += 1
            runs.append([k, e])
            k = e + 1
        is_pulse = lambda r: bool(np.max(z[r[0]:r[1] + 1]) >= pulse_z)
        while len(runs) > 1:
            short = [i for i, r in enumerate(runs) if len(r) == 2 and r[1] - r[0] + 1 < min_run and not is_pulse(r)]
            if not short:
                break
            i = min(short, key=lambda k_: runs[k_][1] - runs[k_][0])
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(runs) and not is_pulse(runs[j])]
            if not nb:                      # isolated between pulses / edges: stays as its own run
                runs[i].append("keep")
                continue
            j = max(nb, key=lambda k_: runs[k_][1] - runs[k_][0])
            lo, hi = sorted((i, j))
            runs[lo:hi + 1] = [[runs[lo][0], runs[hi][1]]]
        for r in runs:
            r0, r1 = r[0], r[1]
            sl = slice(r0, r1 + 1)
            J = float(np.sum(Q[sl]) * dt)
            sJ = float(math.sqrt(np.sum((Qs[sl] * dt) ** 2)))
            pulse = is_pulse([r0, r1])
            forced = pulse or (sJ > 0 and abs(J) / sJ >= z_J)
            out[sl] = forced
            decisions.append(dict(rows=[int(r0), int(r1)], J_over_sigma=J / sJ if sJ > 0 else None, pulse=pulse,
                                  forced=bool(forced)))
    return out, decisions


def classify(rec, *, z_sig=3.0, win_rows=5, v_k=3.0, sep_margin_m=0.03, part_points_ref=None, q_ref=0.0,
             min_rows=3, min_run=10, pulse_z=10.0, z_J=5.0) -> dict:
    """Row-level labels + contiguous intervals with evidence. Pure function of the canonical record."""
    n = len(rec.force_t_s)
    Q, Qs = np.asarray(rec.Q_object, float), np.asarray(rec.Q_sigma, float)
    finite = np.isfinite(Q) & np.isfinite(Qs) & (Qs > 0)
    z = np.where(finite, np.abs(Q) / np.where(Qs > 0, Qs, 1.0), 0.0)
    zr = _running_rms(z, win_rows)
    sig = zr > z_sig
    mode = np.asarray(rec.contact_mode).astype(str)
    annotated = np.asarray(rec.in_contact, bool)
    contact = annotated | np.isin(mode, CONTACT_MODES)
    sticking = np.isin(mode, ("sticking", "stick"))
    sep = _separation(rec, part_points_ref, q_ref)
    # geometric release is CALIBRATED against the separation measured while contact is force-confirmed (the tentative
    # hull and the TCP calibration carry a cm-level offset): release = sep - sep_ref > margin. Without >= 3
    # force-confirmed contact rows the test is reported only (never overrides the annotation).
    geo = dict(status="NOT_AVAILABLE (no part points)")
    geo_released = None
    if sep is not None:
        ref_rows = contact & sig & ~sticking & finite
        if ref_rows.sum() >= 3:
            sep_ref = float(np.median(sep[ref_rows]))
            geo_released = (sep - sep_ref) > sep_margin_m
            geo = dict(status="CALIBRATED", sep_ref_m=sep_ref, n_ref_rows=int(ref_rows.sum()),
                       rule=f"released if separation - sep_ref > {sep_margin_m} m",
                       sep_ref_spread_m=[float(sep[ref_rows].min()), float(sep[ref_rows].max())])
        else:
            geo = dict(status="UNCALIBRATED (fewer than 3 force-confirmed contact rows): reported, not used",
                       sep_range_m=[float(np.min(sep)), float(np.max(sep))])
    released = (~annotated) & ~np.isin(mode, CONTACT_MODES)
    if geo_released is not None:
        released = released | geo_released
        contact = contact & ~geo_released
    sig, run_decisions = _merge_contact_runs(sig, z, np.where(finite, Q, 0.0), np.where(finite, Qs, 0.0),
                                             contact & ~sticking & finite & ~(released & ~contact),
                                             min_run=min_run, pulse_z=pulse_z, z_J=z_J)
    # motion at the force rows (from the q clock)
    v, vsd = _velocity(rec)
    vi = np.interp(rec.force_t_s, rec.q_t_s, v)
    vsdi = np.interp(rec.force_t_s, rec.q_t_s, vsd)
    in_q = (rec.force_t_s >= rec.q_t_s[0]) & (rec.force_t_s <= rec.q_t_s[-1])
    moving = in_q & (np.abs(vi) > v_k * vsdi)
    label = np.empty(n, dtype=object)
    why = np.empty(n, dtype=object)
    for i in range(n):
        if not finite[i]:
            label[i], why[i] = LABELS[2], "ABSENT_WRENCH"
        elif released[i] and not contact[i]:
            label[i], why[i] = (LABELS[2], "EFFORT_WITHOUT_ANNOTATED_CONTACT") if sig[i] else (LABELS[0], "RELEASED")
        elif contact[i] and sticking[i]:
            label[i], why[i] = LABELS[2], "STICK_INTERVAL"
        elif contact[i] and sig[i]:
            label[i], why[i] = LABELS[1], "MEASURED_CONTACT_EFFORT"
        elif contact[i]:
            label[i], why[i] = LABELS[2], "LOW_MEASURED_EFFORT_IN_ANNOTATED_CONTACT"
        else:
            label[i], why[i] = LABELS[2], "AMBIGUOUS_CONTACT"
    # contiguous intervals of identical (label, reason)
    ivs, st = [], 0
    for i in range(1, n + 1):
        if i == n or label[i] != label[st] or why[i] != why[st] or \
                rec.source_force_row[i] != rec.source_force_row[i - 1] + 1:
            ivs.append((st, i - 1))
            st = i
    intervals = []
    for a, b in ivs:
        sl = slice(a, b + 1)
        t0, t1 = float(rec.force_t_s[a]), float(rec.force_t_s[b]) + 0.02
        kq = np.flatnonzero((rec.q_t_s >= t0) & (rec.q_t_s <= t1))
        dt = 0.02
        J = float(np.sum(Q[sl][finite[sl]]) * dt)
        sJ = float(math.sqrt(np.sum((Qs[sl][finite[sl]] * dt) ** 2)))
        zz = z[sl]
        iv = dict(label=str(label[a]), reason=str(why[a]), rows=[int(rec.source_force_row[a]),
                                                                  int(rec.source_force_row[b])],
                  t=[t0, t1], n_rows=b - a + 1, q_frames=[int(rec.source_q_frame[k]) for k in kq],
                  n_q=int(len(kq)), moving_fraction=float(np.mean(moving[sl])),
                  dq=float(np.interp(t1, rec.q_t_s, rec.q) - np.interp(t0, rec.q_t_s, rec.q)) if in_q[sl].any()
                  else None,
                  effort=dict(J=J, sigma_J=sJ, J_over_sigma=J / sJ if sJ > 0 else None,
                              chi2_per_row=float(np.mean(zz ** 2)) if len(zz) else None,
                              peak_z=float(zz.max()) if len(zz) else None),
                  contact_evidence=dict(annotated_rows=int(annotated[sl].sum()),
                                        force_modes={m: int((mode[sl] == m).sum()) for m in np.unique(mode[sl])},
                                        geometric_separation_m=None if sep is None else
                                        [float(sep[sl].min()), float(sep[sl].max())]),
                  short=bool(b - a + 1 < min_rows))
        if iv["label"] != LABELS[0] and annotated[sl].any() and b > a:
            iv["tcp_tracking"] = _tcp_tracking(rec, a, b)
        intervals.append(iv)
    return dict(schema="forceartgs.g9.segment_evidence/v1", labels=list(LABELS),
                contact_run_decisions=run_decisions,
                policy=dict(z_sig=z_sig, win_rows=win_rows, v_k=v_k, sep_margin_m=sep_margin_m, min_run=min_run,
                            pulse_z=pulse_z, z_J=z_J,
                            contact_runs="inside annotated contact: runs < min_run rows merged (pulses kept); a merged "
                                         "run is FORCED if |J|/sigma_J >= z_J or a pulse (peak >= pulse_z)",
                            geometric_release=geo,
                            rule="FREE_CONFIRMED needs release evidence AND noise-consistent effort; low force inside "
                                 "an annotated contact is UNKNOWN (LOW_MEASURED_EFFORT_IN_ANNOTATED_CONTACT)"),
                row_label=[str(x) for x in label], row_reason=[str(x) for x in why],
                row_moving=moving.tolist(), intervals=intervals,
                summary={lab: dict(n_rows=int(np.sum(label == lab)),
                                   n_moving_rows=int(np.sum((label == lab) & moving))) for lab in LABELS} |
                {"reasons": {r: int(np.sum(why == r)) for r in np.unique(why.astype(str))}})


def build_segments(rec, ev, *, free_hypotheses=(), min_q=5, v_k=3.0, noise="empirical") -> list:
    """Monotonic moving MotionSegments from the classified intervals. FREE_CONFIRMED and FORCED_MEASURED are used;
    UNKNOWN only under an explicitly named free hypothesis (labelled FREE_HYPOTHESIS:<name>)."""
    allowed = {}
    for h in free_hypotheses:
        if h not in HYPOTHESES:
            raise ValueError(f"unknown free hypothesis {h!r}; known {sorted(HYPOTHESES)}")
        allowed[HYPOTHESES[h]] = h
    v, vsd = _velocity(rec)
    out = []
    for k, iv in enumerate(ev["intervals"]):
        lab = iv["label"]
        if lab == LABELS[2]:
            if iv["reason"] not in allowed:
                continue
            seg_label = f"FREE_HYPOTHESIS:{allowed[iv['reason']]}"
        else:
            seg_label = lab
        t0, t1 = iv["t"]
        m = (rec.q_t_s >= t0 - 1e-9) & (rec.q_t_s <= t1 + 1e-9)
        idx = np.flatnonzero(m)
        if len(idx) < min_q:
            continue
        mv = np.abs(v[idx]) > v_k * vsd[idx]
        sg = np.sign(v[idx])
        # maximal runs of moving samples with one sign
        runs, st = [], None
        for j in range(len(idx) + 1):
            ok = j < len(idx) and mv[j] and (st is None or sg[j] == sg[st])
            if ok and st is None:
                st = j
            elif not ok and st is not None:
                runs.append((st, j - 1))
                st = j if (j < len(idx) and mv[j]) else None
        for r, (a, b) in enumerate(runs):
            if b - a + 1 < min_q:
                continue
            ii = idx[a:b + 1]
            fm = (rec.force_t_s >= rec.q_t_s[ii[0]] - 0.1) & (rec.force_t_s <= rec.q_t_s[ii[-1]] + 0.02)
            out.append(MotionSegment(
                f"{seg_label.split(':')[0].lower()}_{iv['rows'][0]}-{iv['rows'][1]}_{r}", seg_label, rec.joint_type,
                rec.q_t_s[ii], rec.q[ii], rec.q_sigma[ii], float(sg[a]), rec.force_t_s[fm], rec.Q_object[fm],
                rec.Q_sigma[fm], dict(interval_index=k, interval_rows=iv["rows"], reason=iv["reason"],
                                      q_frames=[int(x) for x in rec.source_q_frame[ii]],
                                      tcp_tracking=iv.get("tcp_tracking"))))
    if noise == "empirical":
        for sg in out:
            inflate_noise(sg)
    return out
