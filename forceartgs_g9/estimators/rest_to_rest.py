"""Rest-to-rest demonstration episode: ONE immutable object consumed by every method of the rest-to-rest comparison.

Experimental facts (given by the experiment owner, never re-estimated here): known sensor transform, wrench tared at
the start, robot-object contact ON for the whole interaction, qdot(t0) = qdot(T) = 0. Consequences implemented here:

* the joint effort Q(t) is the full six-axis wrench projection on EVERY native row in [t0, T] -- no contact mask, no
  force threshold, no second tare (C1 input policy);
* t0 / T are located on the VISUAL trajectory only, by a sliding-window constant-position test against a linear trend
  with the declared q uncertainty (q-stability, not contact detection), independent of any dynamics model or fit;
* q0 / qT are robust plateau estimates (with their own uncertainty and source frames); q_obs itself is untouched and
  never re-zeroed; v0 = vT = 0 are declared facts, never fitted and never estimated from a noisy derivative.

If the supplied visual crop does not cover a stationary plateau at its start / end the episode is not built
(REST_START_NOT_COVERED / REST_END_NOT_COVERED with the exact frames) -- the experimental statement is not challenged,
the crop just cannot support imposing it.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

EPISODE_SCHEMA = "forceartgs.g9.rest_to_rest_episode/v1"
STATUS_OK = "OK"
REST_START_NOT_COVERED = "REST_START_NOT_COVERED"
REST_END_NOT_COVERED = "REST_END_NOT_COVERED"
NO_MOTION_OBSERVED = "NO_MOTION_OBSERVED"
INPUT_COVERAGE_ERROR = "INPUT_COVERAGE_ERROR"
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
UNMODELED_STATIONARY_LOAD = "UNMODELED_STATIONARY_LOAD"


class ZOHEffort:
    """Forward-ZOH generalized effort: levels[k] on [t_s[k], t_s[k+1]) (same contract as GeneralizedEffort)."""

    def __init__(self, t_s, levels, model="rest_to_rest_full_wrench"):
        self.t_s, self.levels, self.model = np.asarray(t_s, float), np.asarray(levels, float), model

    def at(self, t):
        idx = np.clip(np.searchsorted(self.t_s, np.asarray(t, float), side="right") - 1, 0, len(self.t_s) - 1)
        return self.levels[idx]


def _ro(x):
    a = np.array(x, dtype=float, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class RestToRestEpisode:
    scene: str
    joint_type: str
    t_q_s: np.ndarray                  # valid visual timestamps in [t0, T], physical seconds (source frame / 50)
    q_obs: np.ndarray                  # original joint coordinate (rad / m), never re-zeroed
    q_sigma: np.ndarray
    q0: float                          # initial stationary plateau estimate
    qT: float                          # terminal stationary plateau estimate
    t0: float                          # first physical rest sample of the demonstration (visual)
    T: float                           # first sustained stationary sample after the last motion
    force_t_s: np.ndarray              # native wrench rows: last row <= t0 ... last row <= T
    force_Q: np.ndarray                # full six-axis wrench joint effort, UNMASKED
    force_sigma: np.ndarray | None = None
    source_frames_q: np.ndarray | None = None
    source_rows_force: np.ndarray | None = None
    q0_sigma: float = 0.0
    qT_sigma: float = 0.0
    v0: float = 0.0                    # declared fact, MUST stay +0.0
    vT_observed: float = 0.0           # declared fact (no noisy derivative is used as a label)
    provenance: dict = field(default_factory=dict)
    endpoint_evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        for k in ("t_q_s", "q_obs", "q_sigma", "force_t_s", "force_Q"):
            object.__setattr__(self, k, _ro(getattr(self, k)))
        for k in ("force_sigma", "source_frames_q", "source_rows_force"):
            if getattr(self, k) is not None:
                object.__setattr__(self, k, _ro(getattr(self, k)))
        if self.joint_type not in ("revolute", "prismatic"):
            raise ValueError(f"joint_type {self.joint_type!r}")
        for nm in ("v0", "vT_observed"):
            v = getattr(self, nm)
            if not (v == 0.0 and math.copysign(1.0, v) > 0):
                raise ValueError(f"{nm} must be exactly +0.0 (declared rest-to-rest fact), got {v!r}")
        n = len(self.t_q_s)
        if n < 3 or len(self.q_obs) != n or len(self.q_sigma) != n:
            raise ValueError("t_q_s / q_obs / q_sigma must have equal length >= 3")
        if np.any(np.diff(self.t_q_s) <= 0) or np.any(self.q_sigma <= 0):
            raise ValueError("strictly increasing visual times and positive q sigmas required")
        if abs(self.t_q_s[0] - self.t0) > 1e-12 or abs(self.t_q_s[-1] - self.T) > 1e-12:
            raise ValueError("the visual samples must span exactly [t0, T]")
        m = len(self.force_t_s)
        if m == 0 or len(self.force_Q) != m or np.any(np.diff(self.force_t_s) <= 0):
            raise ValueError("native effort rows must be nonempty and strictly increasing")
        if self.force_t_s[0] > self.t0 + 1e-12:
            raise ValueError("the native effort must be defined at t0 (no extrapolation)")
        if self.force_sigma is not None and len(self.force_sigma) != m:
            raise ValueError("force_sigma length mismatch")
        if not all(np.isfinite(a).all() for a in (self.t_q_s, self.q_obs, self.q_sigma, self.force_t_s,
                                                    self.force_Q)) or not np.isfinite([self.q0, self.qT]).all():
            raise ValueError("non-finite episode data")

    # ------------------------------------------------------------------ helpers
    @property
    def delta_q(self) -> float:
        return float(self.qT - self.q0)

    @property
    def effort(self) -> ZOHEffort:
        return ZOHEffort(self.force_t_s, self.force_Q)

    def sha256(self) -> str:
        h = hashlib.sha256()
        for a in (self.t_q_s, self.q_obs, self.q_sigma, self.force_t_s, self.force_Q,
                  np.array([self.q0, self.qT, self.t0, self.T, self.v0])):
            h.update(np.ascontiguousarray(a, float).tobytes())
        h.update(self.joint_type.encode())
        return h.hexdigest()

    def to_physx_episode(self, name="rest_to_rest", effort=None):
        """ONE G10 Episode over [t0, T]: q0 = initial plateau estimate, v0 = +0.0, the UNMASKED full-wrench effort on
        every native row. No reset, no gating, no state overwrite at T."""
        from ..physx_refinement.objective import Episode
        lev = self.force_Q if effort is None else np.asarray(effort, float)
        if lev.shape != self.force_Q.shape:
            raise ValueError("an input variant must live on the recorded effort timeline")
        return Episode(name=name, t_s=self.t_q_s, q=self.q_obs, sigma_q=self.q_sigma, q0=float(self.q0), v0=0.0,
                       input_t_s=self.force_t_s, input_effort=lev, mode="forced",
                       calibration_status="GIVEN_EXPERIMENTAL_FACTS", used_for_training=True,
                       exploratory_assumptions_accepted=True, sigma_v0=None,
                       regime_semantics="behavioral_full_demo: rest-to-rest full record [t0, T]; the full six-axis "
                                        "wrench effort is applied on every native row (contact ON throughout, no mask)")

    def describe(self) -> dict:
        return dict(schema=EPISODE_SCHEMA, scene=self.scene, joint_type=self.joint_type, t0=self.t0, T=self.T,
                    q0=self.q0, qT=self.qT, q0_sigma=self.q0_sigma, qT_sigma=self.qT_sigma, delta_q=self.delta_q,
                    v0=self.v0, vT=self.vT_observed, n_q=int(len(self.t_q_s)), n_force_rows=int(len(self.force_t_s)),
                    source_frames=[int(self.source_frames_q[0]), int(self.source_frames_q[-1])]
                    if self.source_frames_q is not None else None,
                    force_rows=[int(self.source_rows_force[0]), int(self.source_rows_force[-1])]
                    if self.source_rows_force is not None else None, sha256=self.sha256(),
                    input_policy="C1_FULL_MEASURED_UNMASKED", provenance=self.provenance)

    def save_npz(self, path):
        np.savez(path, schema=np.array(EPISODE_SCHEMA), scene=np.array(self.scene), joint_type=np.array(self.joint_type),
                 t_q_s=self.t_q_s, q_obs=self.q_obs, q_sigma=self.q_sigma, q0=self.q0, qT=self.qT, t0=self.t0,
                 T=self.T, q0_sigma=self.q0_sigma, qT_sigma=self.qT_sigma, force_t_s=self.force_t_s,
                 force_Q=self.force_Q,
                 force_sigma=self.force_sigma if self.force_sigma is not None else np.full(len(self.force_Q), np.nan),
                 source_frames_q=(self.source_frames_q if self.source_frames_q is not None
                                  else np.full(len(self.t_q_s), -1)),
                 source_rows_force=(self.source_rows_force if self.source_rows_force is not None
                                    else np.full(len(self.force_Q), -1)),
                 sha256=np.array(self.sha256()))

    @classmethod
    def load_npz(cls, path, **kw):
        z = np.load(path, allow_pickle=False)
        fs = z["force_sigma"]
        ep = cls(scene=str(z["scene"]), joint_type=str(z["joint_type"]), t_q_s=z["t_q_s"], q_obs=z["q_obs"],
                 q_sigma=z["q_sigma"], q0=float(z["q0"]), qT=float(z["qT"]), t0=float(z["t0"]), T=float(z["T"]),
                 force_t_s=z["force_t_s"], force_Q=z["force_Q"], force_sigma=None if np.isnan(fs).all() else fs,
                 source_frames_q=z["source_frames_q"], source_rows_force=z["source_rows_force"],
                 q0_sigma=float(z["q0_sigma"]), qT_sigma=float(z["qT_sigma"]), **kw)
        if ep.sha256() != str(z["sha256"]):
            raise ValueError(f"{path}: episode hash mismatch (file modified)")
        return ep


# ============================================================================== endpoint selection (visual only)
@dataclass(frozen=True)
class RestEndpointEvidence:
    status: str
    reason: str
    k_sigma: float
    min_samples: int
    min_duration_s: float
    sigma_floor: float
    initial: dict = field(default_factory=dict)
    terminal: dict = field(default_factory=dict)
    motion: dict = field(default_factory=dict)
    stationary_load: dict = field(default_factory=dict)
    input_coverage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__, rule=(
            "stationary window = constant-position fit chi2 <= chi2_{n-1} quantile at the two-sided k-sigma confidence AND linear-trend slope "
            "|b| / se(b) <= k (sigma = hypot(q_sigma, sigma_floor)); a plateau needs >= min_samples and >= "
            "min_duration_s. t0 = first sample of the initial plateau; T = first sample of the terminal plateau "
            "(the rest of the plateau = settling confirmation, kept as evidence, not fitted). Visual q only: no force, "
            "no contact label, no dynamics residual is consulted."))


def stationary_test(t, q, s, k) -> dict:
    """Constant-position fit vs a linear trend with the declared uncertainty (weighted). Stationary when BOTH
    (a) the constant fit is consistent: chi2 <= chi2_{n-1} quantile at the two-sided k-sigma confidence (k = 3 ->
        0.9973; a per-sample max |z| test would reject long true plateaus far more often), and
    (b) the fitted slope is not significant: |b| / se(b) <= k."""
    from scipy.special import erf
    from scipy.stats import chi2 as _chi2
    t, q, s = (np.asarray(x, float) for x in (t, q, s))
    w = 1.0 / s ** 2
    m = float(np.sum(w * q) / np.sum(w))
    r = (q - m) / s
    tc = t - float(np.sum(w * t) / np.sum(w))
    stt = float(np.sum(w * tc * tc))
    b = float(np.sum(w * tc * (q - m)) / stt) if stt > 0 else 0.0
    se_b = 1.0 / math.sqrt(stt) if stt > 0 else math.inf
    z_b = abs(b) / se_b if se_b > 0 else math.inf
    dof = max(len(q) - 1, 1)
    conf = float(erf(k / math.sqrt(2.0)))
    chi2 = float(np.sum(r * r))
    chi2_max = float(_chi2.ppf(conf, dof))
    ok = bool(chi2 <= chi2_max and z_b <= k)
    return dict(stationary=ok, mean=m, sigma_mean=float(1.0 / math.sqrt(np.sum(w))), max_abs_z=float(np.max(np.abs(r))),
                chi2=chi2, chi2_max=chi2_max, chi2_dof=chi2 / dof, confidence=conf, slope=b, slope_se=se_b,
                slope_z=float(z_b), n=int(len(q)), duration_s=float(t[-1] - t[0]))


def _min_window(t, start, step, min_samples, min_dur):
    """Index of the first window end (step +1 forward / -1 backward) satisfying both minima, or None."""
    n = len(t)
    j = start
    while 0 <= j < n:
        cnt = abs(j - start) + 1
        if cnt >= min_samples and abs(t[j] - t[start]) >= min_dur - 1e-12:
            return j
        j += step
    return None


def find_plateau(t, q, s, *, at_start, k, min_samples, min_dur) -> dict:
    n = len(t)
    a = 0 if at_start else n - 1
    step = 1 if at_start else -1
    j = _min_window(t, a, step, min_samples, min_dur)
    if j is None:
        return dict(found=False, reason="the crop is shorter than one minimal plateau window")
    lo, hi = (a, j) if at_start else (j, a)
    st = stationary_test(t[lo:hi + 1], q[lo:hi + 1], s[lo:hi + 1], k)
    if not st["stationary"]:
        return dict(found=False, window=[lo, hi], test=st,
                    reason=f"the minimal {'initial' if at_start else 'terminal'} window [{lo}, {hi}] is not stationary "
                           f"(chi2 {st['chi2']:.3g} > {st['chi2_max']:.3g} or trend z {st['slope_z']:.3g} > {k}; max |z| {st['max_abs_z']:.3g})")
    while True:                                   # grow while the plateau stays stationary
        nj = (hi + 1) if at_start else (lo - 1)
        if not (0 <= nj < n):
            break
        # sequential rule: the NEW sample itself must agree with the current plateau mean (a window chi2 alone dilutes
        # one moving sample in a long plateau and would put T inside the motion); conservative: T can only move later
        if abs(q[nj] - st["mean"]) > k * math.hypot(s[nj], st["sigma_mean"]):
            break
        lo2, hi2 = (lo, nj) if at_start else (nj, hi)
        st2 = stationary_test(t[lo2:hi2 + 1], q[lo2:hi2 + 1], s[lo2:hi2 + 1], k)
        if not st2["stationary"]:
            break
        lo, hi, st = lo2, hi2, st2
    return dict(found=True, window=[int(lo), int(hi)], test=st)


def _coverage(force_t, Q, t0, t_end, max_gap_factor):
    force_t, Q = np.asarray(force_t, float), np.asarray(Q, float)
    dt = float(np.median(np.diff(force_t))) if len(force_t) > 1 else math.nan
    i0 = int(np.searchsorted(force_t, t0 + 1e-12, side="right") - 1)
    i1 = int(np.searchsorted(force_t, t_end + 1e-12, side="right") - 1)
    missing = []
    if i0 < 0:
        missing.append(dict(kind="NO_EFFORT_AT_T0", t=[float(t0), float(force_t[0]) if len(force_t) else None]))
        i0 = 0
    if len(force_t) == 0 or i1 < 0 or force_t[i1] < t_end - max_gap_factor * dt:
        missing.append(dict(kind="EFFORT_ENDS_BEFORE_T", t=[float(force_t[i1]) if len(force_t) else None,
                                                             float(t_end)]))
    seg = force_t[i0:i1 + 1]
    missing += [dict(kind="ROW_GAP", t=[float(seg[g]), float(seg[g + 1])])
                for g in np.flatnonzero(np.diff(seg) > max_gap_factor * dt)]
    missing += [dict(kind="NONFINITE_EFFORT_ROW", t=[float(seg[b])] * 2)
                for b in np.flatnonzero(~np.isfinite(Q[i0:i1 + 1]))]
    return dict(status="COMPLETE" if not missing else "INPUT_COVERAGE_INCOMPLETE", row_range=[i0, i1],
                nominal_row_dt_s=dt, t=[float(t0), float(t_end)], missing=missing,
                rule=f"forward ZOH on native rows; gaps > {max_gap_factor} x row period, non-finite rows or no row at "
                     "t0 are refused (no hold, no zero fill, no extrapolation)")


def stationary_load(force_t, Q, Qs, ta, tb, k, *, tare_rows=25) -> dict:
    """Mean recorded effort over a stationary plateau [ta, tb] and its uncertainty: white row noise + the shared
    tare-reference uncertainty (sigma / sqrt(tare_rows)). A purely viscous model needs Q = 0 at rest: a significant
    mean is UNMODELED_STATIONARY_LOAD (reported, never re-tared, never relabelled contact-off)."""
    force_t, Q = np.asarray(force_t, float), np.asarray(Q, float)
    m = (force_t >= ta - 1e-12) & (force_t <= tb + 1e-12)
    if m.sum() == 0:
        return dict(status="NOT_EVALUATED", reason="no native effort row inside the plateau")
    q = Q[m]
    s = np.asarray(Qs, float)[m] if Qs is not None else np.full(m.sum(), np.std(q) if m.sum() > 1 else 0.0)
    sd_row = float(np.sqrt(np.mean(s ** 2)))
    se = math.hypot(sd_row / math.sqrt(m.sum()), sd_row / math.sqrt(tare_rows))
    mean = float(np.mean(q))
    z = abs(mean) / se if se > 0 else math.inf
    return dict(status=(UNMODELED_STATIONARY_LOAD if z > k else "NO_SIGNIFICANT_STATIONARY_LOAD"), mean_Q=mean,
                se=se, z=float(z), n_rows=int(m.sum()), t=[float(ta), float(tb)], max_abs_Q=float(np.max(np.abs(q))),
                rule=f"|mean Q| > {k} x hypot(row sd / sqrt(n), row sd / sqrt({tare_rows}) tare reference)",
                note="model-consistency diagnostic of the two-term model, NOT a contact state")


def build_rest_to_rest_episode(*, scene, joint_type, t_q, q, q_sigma, q_frames, force_t, force_Q, force_sigma,
                               force_rows, k_sigma=3.0, min_samples=4, min_duration_s=0.12, sigma_floor=3e-4,
                               max_gap_factor=1.5, provenance=None) -> dict:
    """Locate the rest-to-rest window on the VISUAL trajectory and freeze the episode. Returns
    dict(status, reason, episode | None, evidence: RestEndpointEvidence)."""
    t_q, q, q_sigma, q_frames = (np.asarray(x) for x in (t_q, q, q_sigma, q_frames))
    ok = np.isfinite(t_q) & np.isfinite(q) & np.isfinite(q_sigma) & (q_sigma > 0)
    t, qq, qs, fr = t_q[ok].astype(float), q[ok].astype(float), q_sigma[ok].astype(float), q_frames[ok].astype(int)
    se = np.hypot(qs, float(sigma_floor))
    base = dict(k_sigma=float(k_sigma), min_samples=int(min_samples), min_duration_s=float(min_duration_s),
                sigma_floor=float(sigma_floor))
    if len(t) < 2 * int(min_samples) + 1:
        ev = RestEndpointEvidence(DATA_UNAVAILABLE, f"{len(t)} valid visual samples", **base)
        return dict(status=DATA_UNAVAILABLE, reason=ev.reason, episode=None, evidence=ev)
    ini = find_plateau(t, qq, se, at_start=True, k=k_sigma, min_samples=min_samples, min_dur=min_duration_s)
    ter = find_plateau(t, qq, se, at_start=False, k=k_sigma, min_samples=min_samples, min_dur=min_duration_s)

    def frames(w):
        return [int(fr[w[0]]), int(fr[w[1]])] if w else None
    for p in (ini, ter):
        if "window" in p:
            p["source_frames"] = frames(p["window"])
    if not ini["found"]:
        w = ini.get("window") or [0, len(t) - 1]
        ev = RestEndpointEvidence(REST_START_NOT_COVERED, ini["reason"] + f"; first visual source frame {int(fr[0])}: "
                                  "the crop starts while the object moves (no v0 = 0 is imposed on a moving crop)",
                                  initial=dict(ini, missing_frames=f"before source frame {int(fr[0])} (not in the "
                                               "joint authority)", window_frames=frames(w)), terminal=ter, **base)
        return dict(status=REST_START_NOT_COVERED, reason=ev.reason, episode=None, evidence=ev)
    if not ter["found"]:
        w = ter.get("window") or [0, len(t) - 1]
        ev = RestEndpointEvidence(REST_END_NOT_COVERED, ter["reason"] + f"; last visual source frame {int(fr[-1])}: "
                                  "the crop ends before a sustained final rest", initial=ini,
                                  terminal=dict(ter, missing_frames=f"after source frame {int(fr[-1])} (not in the "
                                                "joint authority)", window_frames=frames(w)), **base)
        return dict(status=REST_END_NOT_COVERED, reason=ev.reason, episode=None, evidence=ev)
    i_a, i_T = ini["window"][1], ter["window"][0]
    if i_T <= i_a:
        ev = RestEndpointEvidence(NO_MOTION_OBSERVED, "the initial and terminal plateaus overlap: the whole visual "
                                  "trajectory is stationary at the declared resolution", initial=ini, terminal=ter, **base)
        return dict(status=NO_MOTION_OBSERVED, reason=ev.reason, episode=None, evidence=ev)
    q0, s0 = ini["test"]["mean"], ini["test"]["sigma_mean"]
    qT, sT = ter["test"]["mean"], ter["test"]["sigma_mean"]
    mid = slice(i_a + 1, i_T)
    dev = np.abs(qq[mid] - q0) / se[mid] if i_T > i_a + 1 else np.zeros(0)
    motion = dict(max_abs_deviation_from_q0_sigma=float(dev.max()) if len(dev) else 0.0,
                  motion_samples=int(np.sum(dev > k_sigma)), delta_q=float(qT - q0),
                  delta_q_sigma=float(math.hypot(s0, sT)),
                  delta_q_snr=float(abs(qT - q0) / max(math.hypot(s0, sT), 1e-300)),
                  return_to_start=bool(abs(qT - q0) <= k_sigma * math.hypot(s0, sT)))
    i_end = len(t) - 1
    sel = slice(0, i_T + 1)
    t0, T = float(t[0]), float(t[i_T])
    cov = _coverage(force_t, force_Q, t0, T, max_gap_factor)
    if cov["status"] != "COMPLETE":
        ev = RestEndpointEvidence(INPUT_COVERAGE_ERROR, f"native wrench missing in [t0, T]: {cov['missing']}",
                                  initial=ini, terminal=ter, motion=motion, input_coverage=cov, **base)
        return dict(status=INPUT_COVERAGE_ERROR, reason=ev.reason, episode=None, evidence=ev)
    r0, r1 = cov["row_range"]
    rs = slice(r0, r1 + 1)
    fs = None if force_sigma is None else np.asarray(force_sigma, float)
    # the half of each plateau FARTHEST from the motion: at measurement resolution a plateau's motion-side edge can
    # contain the onset of the push / the end of the braking (declared rule; evidence only)
    load = dict(initial=stationary_load(force_t, force_Q, fs, t0, t0 + 0.5 * (float(t[i_a]) - t0), k_sigma),
                terminal=stationary_load(force_t, force_Q, fs, T + 0.5 * (float(t[i_end]) - T), float(t[i_end]),
                                         k_sigma))
    for v in load.values():
        v["window_rule"] = "half of the plateau farthest from the motion"
    flags = sorted({v["status"] for v in load.values() if v.get("status") == UNMODELED_STATIONARY_LOAD})
    ev = RestEndpointEvidence(STATUS_OK, f"t0 = {t0:.4g} s (frame {int(fr[0])}), T = {T:.4g} s (frame {int(fr[i_T])}); "
                              f"q0 = {q0:.6g}, qT = {qT:.6g}, delta_q = {qT - q0:.4g} (snr {motion['delta_q_snr']:.3g}); "
                              f"{i_end - i_T} confirmation sample(s) after T" + (f"; {', '.join(flags)}" if flags else ""),
                              initial=dict(ini, t0=t0, q0=q0, q0_sigma=s0, n_samples=ini["test"]["n"]),
                              terminal=dict(ter, T=T, qT=qT, qT_sigma=sT, n_samples=ter["test"]["n"],
                                            confirmation_samples_after_T=int(i_end - i_T),
                                            plateau_duration_s=float(t[i_end] - T)),
                              motion=motion, stationary_load=load, input_coverage=cov, **base)
    ep = RestToRestEpisode(scene=str(scene), joint_type=str(joint_type), t_q_s=t[sel], q_obs=qq[sel], q_sigma=qs[sel],
                           q0=float(q0), qT=float(qT), t0=t0, T=T, force_t_s=np.asarray(force_t, float)[rs],
                           force_Q=np.asarray(force_Q, float)[rs], force_sigma=None if fs is None else fs[rs],
                           source_frames_q=fr[sel], source_rows_force=np.asarray(force_rows, int)[rs],
                           q0_sigma=float(s0), qT_sigma=float(sT), provenance=dict(provenance or {}),
                           endpoint_evidence=ev.to_dict())
    return dict(status=STATUS_OK, reason=ev.reason, episode=ep, evidence=ev, flags=flags)


def from_full_wrench_record(fr, **kw) -> dict:
    """The rest-to-rest episode from an existing FullWrenchRecord (forceartgs_g9.wrench_input.full_record.build):
    the SAME known transform, tared source, object-side sign and native rows -- every row, no mask."""
    ec = (getattr(fr, "noise", None) or {}).get("effort_components") or {}
    prov = dict(full_wrench_record_sha256=fr.sha256(),
                effort_components=ec.get("mode", "full_wrench"), effort_components_note=ec.get("note"),
                input=("full six-axis wrench joint-screw projection (C1, unmasked)" if not ec else
                       "FORCE ONLY at the TCP (measured moment dropped; labelled variant), C1 unmasked"),
                contract_schema=(fr.contract or {}).get("schema"),
                robot_archive_sha256=(fr.contract or {}).get("robot_archive_sha256"),
                tool_override=(fr.contract or {}).get("tool_override"))
    return build_rest_to_rest_episode(scene=fr.scene, joint_type=fr.joint_type, t_q=fr.q_t_s, q=fr.q,
                                      q_sigma=fr.q_sigma, q_frames=fr.source_q_frame, force_t=fr.t_s,
                                      force_Q=fr.Q_full, force_sigma=fr.Q_sigma, force_rows=fr.source_row,
                                      provenance=prov, **kw)


def terminal_speed_sigma(ep: RestToRestEpisode) -> dict:
    """sigma_vT, derived ONCE from measurement: the standard error of the slope of the terminal plateau's
    linear-trend test (declared q sigma incl. floor, plateau timestamps / length). Not fitted, not a velocity label."""
    ter = (ep.endpoint_evidence or {}).get("terminal") or {}
    se = (ter.get("test") or {}).get("slope_se")
    if se is not None and np.isfinite(se) and se > 0:
        return dict(sigma_vT=float(se), source="slope s.e. of the terminal-plateau trend test "
                                               f"(n = {ter['test']['n']}, {ter['test']['duration_s']:.3g} s)")
    t = ep.t_q_s[-4:]
    stt = float(np.sum((t - t.mean()) ** 2))
    return dict(sigma_vT=float(np.median(ep.q_sigma)) / math.sqrt(stt) if stt > 0 else 1e-3,
                source="fallback: slope s.e. over the last 4 samples (no terminal-plateau evidence attached)")
