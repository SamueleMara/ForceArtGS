"""Unified TCP contact: general ``sliding`` with ``stick_soft`` / ``stick_hard`` as specialisations.

One joint transform ``A_H(q_t)`` and one canonical contact point ``c_t = A_H(q_t)^-1 p_TCP(t)`` serve every mode
(``rigid_kinematics.canonical_contact_point``). Modes differ only in what is asserted about ``c_t``:

``sliding``     c_t may travel over the moving surface: r_t = signed_distance(c_t, S_M) / sigma, where S_M is the
                *observed* moving surface at the reference time (A = I there, so S_M is candidate independent) and a
                row exists only where S_M supports c_t for every candidate (common samples). Slip is never penalised.
``stick_soft``  latent c_t ~ N(c*_k, Sigma_stick) observed through TCP noise; marginalised:
                r_t = (c_t - c*_k) / sqrt(sigma_stick^2 + sigma_tcp^2), robust loss upstream; drift is reported.
``stick_hard``  ONE structural anchor per uninterrupted segment: p_TCP(t) = A_H(q_t) c*_k + e_t,
                r_t = (A_H(q_t) c*_k - p_TCP(t)) / sigma_tcp with finite TCP noise.

The anchor c*_k is eliminated by variable projection: for rigid A_t and isotropic TCP noise
sum_t ||A_t c - p_t||^2 = sum_t ||c - A_t^-1 p_t||^2, so c*_k(axis, origin, q) = mean_t A_t^-1 p_t over the
segment's TRAINING times -- exact, one anchor per segment, never a per-frame anchor tied by a huge weight. Each row
therefore depends on every q of its segment (declared in ``row_q_deps`` for the sparse Jacobian). Held-out scoring
uses the anchor frozen from training. A release / re-grasp is a new segment and a new anchor.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .contact_annotations import INDEPENDENT_PROVENANCE, STATES
from .rigid_kinematics import apply_joint, canonical_contact_point, exp_so3, log_so3, normalize

SEGMENT_SCHEMA = "forceartgs.contact_segments/v1"
SLIP_MODES = ("sliding", "stick_soft", "stick_hard", "rigid_grasp")
CONTACT_MODELS = ("unilateral_point", "grasp", "unknown")


class ContactSegmentError(ValueError):
    pass


@dataclass(frozen=True)
class ContactSegment:
    segment_id: str
    model_time_indices: tuple
    source_frame_indices: tuple
    state: str
    slip_mode: str
    provenance: str
    annotation_sha256: str
    sigma_tcp_position_m: float
    sigma_contact_surface_m: float
    sigma_stick_soft_m: float
    sigma_provenance: str = "configured"
    contact_model: str = "unknown"

    @property
    def confirmed(self) -> bool:
        return self.state == "contact_confirmed" and self.provenance in INDEPENDENT_PROVENANCE

    @property
    def active_for_residuals(self) -> bool:
        """Only independently confirmed contact may create contact rows; unknown/probable never."""
        return self.confirmed and self.slip_mode in ("sliding", "stick_soft", "stick_hard")

    @property
    def scale_verified(self) -> bool:
        return self.sigma_provenance == "configured"

    def to_json(self) -> dict:
        d = asdict(self)
        d.update(confirmed=self.confirmed, active_for_residuals=self.active_for_residuals,
                 scale_status="VERIFIED_SCALE" if self.scale_verified else "UNVERIFIED_PRIOR_SCALE")
        return d


def _read(source):
    if isinstance(source, dict):
        text = json.dumps(source, sort_keys=True)
        return source, hashlib.sha256(text.encode()).hexdigest(), "<dict>"
    path = Path(source)
    raw = path.read_bytes()
    if path.suffix in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(raw.decode())
    else:
        data = json.loads(raw.decode())
    return data, hashlib.sha256(raw).hexdigest(), str(path)


def load_contact_segments(source, frame_map, p_tcp_w=None, cfg=None) -> tuple:
    """Parse and validate a ``forceartgs.contact_segments/v1`` annotation (spec section 3 rules).

    Endpoints map EXACTLY through ``source_frame_index`` of the model frame map (inclusive). Returns
    (segments, report). Every rule violation raises ``ContactSegmentError`` -- never a silent fallback."""
    from .config import ContactModeConfig
    cfg = cfg or ContactModeConfig()
    data, sha, origin = _read(source)
    if data.get("schema") != SEGMENT_SCHEMA:
        raise ContactSegmentError(f"{origin}: schema must be {SEGMENT_SCHEMA!r}, got {data.get('schema')!r}")
    if data.get("frame_index_type") != "source_frame_index":
        raise ContactSegmentError("frame_index_type must be source_frame_index (no filename offsets are inferred)")
    sfi = np.asarray(frame_map.source_frame_index, int)
    ts = np.asarray(frame_map.timestamp_s, float)
    win = data.get("window")
    if win is not None:                      # segments are estimated per window: never reuse them on another one
        lo, hi, k = (int(x) for x in str(win).split(":"))
        grid = set(range(lo, hi + 1, k))
        model = [int(f) for f in getattr(frame_map, "rgbd_frame_index", [])]
        if model and not set(model) <= grid:
            raise ContactSegmentError(f"{origin}: segments were estimated on RGB-D window {win}, but the model frames are "
                                      f"{model[0]}..{model[-1]} (step {model[1] - model[0] if len(model) > 1 else '-'}); "
                                      "re-estimate them on this window (samuele/estimate_contact_segments.py)")
    if np.any(np.diff(sfi) <= 0) or np.any(np.diff(ts) <= 0):
        raise ContactSegmentError("frame map is not strictly increasing in source_frame_index / timestamp")
    index = {int(s): i for i, s in enumerate(sfi)}
    used = {}
    segs, notes = [], []
    p = None if p_tcp_w is None else np.asarray(p_tcp_w, float)
    for raw in data.get("segments", []):
        sid = str(raw.get("segment_id", ""))
        if not sid:
            raise ContactSegmentError("segment without segment_id")
        mode = raw.get("slip_mode", cfg.default_mode)
        if mode == "rigid_grasp":
            raise ContactSegmentError(f"{sid}: rigid_grasp is NOT_IMPLEMENTED (needs a verified 6-DoF grasp transform "
                                      "and SE(3) solver); it is never reinterpreted as stick_hard")
        if mode not in SLIP_MODES:
            raise ContactSegmentError(f"{sid}: unknown slip_mode {mode!r}")
        state = raw.get("contact_state")
        if state not in STATES:
            raise ContactSegmentError(f"{sid}: unknown contact_state {state!r}")
        prov = str(raw.get("provenance", "none"))
        if state == "contact_confirmed" and prov not in INDEPENDENT_PROVENANCE:
            raise ContactSegmentError(f"{sid}: contact_confirmed needs independent provenance {INDEPENDENT_PROVENANCE}, "
                                      f"got {prov!r} (a force threshold never confirms contact)")
        a, b = int(raw["start_source_frame"]), int(raw["end_source_frame"])
        if b < a:
            raise ContactSegmentError(f"{sid}: end_source_frame < start_source_frame")
        frames = [f for f in sfi.tolist() if a <= f <= b]
        if a not in index or b not in index:
            raise ContactSegmentError(f"{sid}: endpoints {a}..{b} are not exact source frames of the model frame map")
        times = tuple(index[f] for f in frames)
        for t in times:
            if t in used:
                raise ContactSegmentError(f"{sid}: overlaps segment {used[t]} at model time {t} (conflicting annotations)")
            used[t] = sid
        cm = raw.get("contact_model", "unknown")
        if cm not in CONTACT_MODELS:
            raise ContactSegmentError(f"{sid}: contact_model must be one of {CONTACT_MODELS}")
        sig, sprov = {}, "configured"
        needed = dict(sliding=("sigma_tcp_position_m", "sigma_contact_surface_m"),
                      stick_soft=("sigma_tcp_position_m", "sigma_stick_soft_m"),
                      stick_hard=("sigma_tcp_position_m",)).get(mode, ())
        for name, fb in (("sigma_tcp_position_m", cfg.fallback_sigma_tcp_position_m),
                         ("sigma_contact_surface_m", cfg.fallback_sigma_contact_surface_m),
                         ("sigma_stick_soft_m", cfg.fallback_sigma_stick_soft_m)):
            v = raw.get(name, getattr(cfg, name))
            if v is None:
                v = fb
                if name in needed:            # only the uncertainties this mode uses decide its scale status
                    sprov = "fallback_unverified"
            if not (float(v) > 0):
                raise ContactSegmentError(f"{sid}: {name} must be > 0 (no sigma = 0 surrogate for hard sticking)")
            sig[name] = float(v)
        seg = ContactSegment(sid, times, tuple(frames), state, mode, prov, sha, sig["sigma_tcp_position_m"],
                             sig["sigma_contact_surface_m"], sig["sigma_stick_soft_m"], sprov, cm)
        if seg.active_for_residuals:
            if len(times) < cfg.min_segment_samples:
                raise ContactSegmentError(f"{sid}: {len(times)} samples < {cfg.min_segment_samples}")
            if p is None:
                raise ContactSegmentError(f"{sid}: confirmed contact needs a calibrated TCP trajectory")
            if not np.isfinite(p[list(times)]).all():
                raise ContactSegmentError(f"{sid}: TCP position missing on confirmed contact frames")
        elif mode.startswith("stick") and state != "contact_confirmed":
            notes.append(f"{sid}: {mode} on {state} -> no contact residual (proposals only)")
        segs.append(seg)
    report = dict(schema=SEGMENT_SCHEMA, source=origin, annotation_sha256=sha, scene_id=data.get("scene_id"),
                  frame_id=data.get("frame_id"), annotation_source=data.get("source"), notes=notes,
                  segments=[s.to_json() for s in segs],
                  n_active=sum(s.active_for_residuals for s in segs),
                  unverified_scale=any(not s.scale_verified for s in segs if s.active_for_residuals))
    return segs, report


# --------------------------------------------------------------------------- observed moving surface
@dataclass
class MovingSurface:
    """Canonical moving surface = observed moving-support points at the reference time (A_H(q_ref) = I, hence
    candidate independent). Normals from local PCA, oriented towards the cameras."""
    status: str
    reason: str = ""
    points: np.ndarray | None = None
    normals: np.ndarray | None = None
    patch_radius_m: float = 0.03
    min_points: int = 8
    _tree: object = field(default=None, repr=False)

    @classmethod
    def from_points(cls, points, camera_centers=None, patch_radius_m=0.03, min_points=8, k=12):
        P = np.asarray(points, float)
        P = P[np.isfinite(P).all(1)]
        if len(P) < max(min_points, k):
            return cls("UNAVAILABLE", f"only {len(P)} observed moving points at the reference time")
        from scipy.spatial import cKDTree
        tree = cKDTree(P)
        _, nb = tree.query(P, k=k)
        N = np.zeros_like(P)
        for i, idx in enumerate(nb):
            Q = P[idx] - P[idx].mean(0)
            N[i] = np.linalg.svd(Q, full_matrices=False)[2][2]
        if camera_centers is not None and len(camera_centers):
            C = np.mean(np.asarray(camera_centers, float), 0)
            flip = np.einsum("ij,ij->i", N, C - P) < 0
            N[flip] = -N[flip]
        return cls("OK", "observed moving support at the reference time", P, N, patch_radius_m, min_points, tree)

    def signed_distance(self, c):
        """(signed distance to the local tangent plane, supported flag) for canonical points ``c`` [K,3]."""
        c = np.atleast_2d(np.asarray(c, float))
        if self.status != "OK":
            return np.full(len(c), np.nan), np.zeros(len(c), bool)
        d, i = self._tree.query(c)
        cnt = np.array([len(x) for x in self._tree.query_ball_point(c, self.patch_radius_m)])
        sd = np.einsum("ij,ij->i", c - self.points[i], self.normals[i])
        tangential = np.sqrt(np.maximum(d ** 2 - sd ** 2, 0))
        supported = (cnt >= self.min_points) & (tangential < self.patch_radius_m)
        return sd, supported


# --------------------------------------------------------------------------- kinematics shared by every mode
def canonical_path(kind, axis, origin, q, p_tcp, times):
    times = np.asarray(times, int)
    return canonical_contact_point(kind, axis, origin, np.asarray(q, float)[times], np.asarray(p_tcp, float)[times])


def stick_anchor(kind, axis, origin, q, p_tcp, train_times):
    """Variable-projection anchor c*_k = mean_t A_t^-1 p_t over training times (exact for rigid A, isotropic noise)."""
    c = canonical_path(kind, axis, origin, q, p_tcp, train_times)
    return c.mean(0)


def predict_tcp(kind, axis, origin, q, anchor, times):
    times = np.asarray(times, int)
    X = np.repeat(np.asarray(anchor, float)[None], len(times), 0)
    return apply_joint(kind, axis, origin, np.asarray(q, float)[times], X)


class ContactRows:
    """``fit_geometry`` extra hook (L_TCP): whitened contact rows of every active segment on TRAINING times.

    ``common_supported``: [T] bool, sliding rows only where the observed surface supports the contact for every
    candidate (computed once, before fitting, from the initial fits) so R and P use identical samples."""

    def __init__(self, segments, p_tcp_w, train_mask, surface: MovingSurface | None = None,
                 common_supported=None, kinds=("revolute", "prismatic")):
        self.segments = [s for s in segments if s.active_for_residuals]
        self.p = np.asarray(p_tcp_w, float)
        T = len(self.p)
        self.train = np.asarray(train_mask, bool)
        self.surface = surface
        cs = np.ones(T, bool) if common_supported is None else np.asarray(common_supported, bool)
        self.blocks = []        # (segment, times, rows_per_time)
        rt, deps = [], []
        for s in self.segments:
            times = np.array([t for t in s.model_time_indices if self.train[t]], int)
            if s.slip_mode == "sliding":
                if surface is None or surface.status != "OK":
                    continue
                times = times[cs[times]]
                k = 1
            else:
                k = 3
            if len(times) < 2 and s.slip_mode != "sliding":
                continue
            if not len(times):
                continue
            self.blocks.append((s, times, k))
            for t in times:
                for _ in range(k):
                    rt.append(int(t))
                    deps.append(times if s.slip_mode != "sliding" else np.array([t]))
        self.row_times = np.asarray(rt, int)
        self.row_q_deps = deps

    @property
    def n_rows(self):
        return len(self.row_times)

    def residuals(self, kind, axis, origin, q):
        out = []
        for s, times, _ in self.blocks:
            out.append(segment_residuals(kind, axis, origin, q, self.p, s, times, self.surface))
        return np.concatenate(out) if out else np.zeros(0)

    def anchors(self, kind, axis, origin, q) -> dict:
        return {s.segment_id: stick_anchor(kind, axis, origin, q, self.p, times).tolist()
                for s, times, _ in self.blocks if s.slip_mode != "sliding"}


def segment_residuals(kind, axis, origin, q, p_tcp, seg: ContactSegment, times, surface=None, anchor=None):
    """Whitened rows of one segment. ``anchor`` given -> frozen (held-out scoring), else variable projection."""
    times = np.asarray(times, int)
    if seg.slip_mode == "sliding":
        c = canonical_path(kind, axis, origin, q, p_tcp, times)
        d, sup = surface.signed_distance(c)
        sig = np.hypot(seg.sigma_contact_surface_m, seg.sigma_tcp_position_m)
        return np.where(sup, d, 0.0) / sig
    a = stick_anchor(kind, axis, origin, q, p_tcp, times) if anchor is None else np.asarray(anchor, float)
    if seg.slip_mode == "stick_hard":
        r = (predict_tcp(kind, axis, origin, q, a, times) - p_tcp[times]) / seg.sigma_tcp_position_m
    else:   # stick_soft: marginal of the latent canonical drift + TCP noise
        c = canonical_path(kind, axis, origin, q, p_tcp, times)
        r = (c - a) / np.hypot(seg.sigma_stick_soft_m, seg.sigma_tcp_position_m)
    return r.ravel()


# --------------------------------------------------------------------------- diagnostics and held-out scoring
def segment_diagnostics(kind, axis, origin, q, p_tcp, seg: ContactSegment, train_mask, surface=None,
                        contradiction_chi2=9.0, heldout_mask=None, frozen_anchor=None) -> dict:
    p_tcp = np.asarray(p_tcp, float)
    all_t = np.asarray(seg.model_time_indices, int)
    tr = all_t[np.asarray(train_mask, bool)[all_t]]
    ho = all_t[np.asarray(heldout_mask, bool)[all_t]] if heldout_mask is not None else np.zeros(0, int)
    out = dict(segment_id=seg.segment_id, mode=seg.slip_mode, state=seg.state, provenance=seg.provenance,
               active=seg.active_for_residuals, n_train=int(len(tr)), n_heldout=int(len(ho)),
               scale_status="VERIFIED_SCALE" if seg.scale_verified else "UNVERIFIED_PRIOR_SCALE",
               contact_mode_assumption=seg.slip_mode,
               contact_mode_evidence="independent annotation (" + seg.provenance + ")" if seg.confirmed else "none")
    if not seg.active_for_residuals:
        out["status"] = "INACTIVE (contact not independently confirmed)"
        return out
    c_tr = canonical_path(kind, axis, origin, q, p_tcp, tr) if len(tr) else np.zeros((0, 3))
    W = p_tcp[all_t]
    s = np.linalg.svd(W - W.mean(0), compute_uv=False) if len(W) >= 3 else np.zeros(3)
    out["tcp_trajectory_singular_values_m"] = (s / np.sqrt(max(len(W), 1))).tolist()
    out["tcp_path_length_m"] = float(np.linalg.norm(np.diff(W, axis=0), axis=1).sum())
    if seg.slip_mode == "sliding":
        if surface is None or surface.status != "OK" or not len(tr):
            out["status"] = "UNAVAILABLE (no observed surface support)"
            return out
        d, sup = surface.signed_distance(c_tr)
        out.update(status="OK", n_supported=int(sup.sum()),
                   normal_residual_rms_m=float(np.sqrt(np.mean(d[sup] ** 2))) if sup.any() else None,
                   canonical_path_length_m=float(np.linalg.norm(np.diff(c_tr, axis=0), axis=1).sum()),
                   note="slip magnitude is not penalised in sliding mode")
        r = segment_residuals(kind, axis, origin, q, p_tcp, seg, tr[sup], surface) if sup.any() else np.zeros(0)
    else:
        if len(tr) == 0:
            out["status"] = "INSUFFICIENT_TRAIN_CONTACT"
            return out
        a = stick_anchor(kind, axis, origin, q, p_tcp, tr) if frozen_anchor is None else np.asarray(frozen_anchor)
        drift = np.linalg.norm(c_tr - a, axis=1)
        sig = seg.sigma_tcp_position_m if seg.slip_mode == "stick_hard" else np.hypot(seg.sigma_stick_soft_m,
                                                                                         seg.sigma_tcp_position_m)
        r = segment_residuals(kind, axis, origin, q, p_tcp, seg, tr, anchor=a)
        out.update(status="OK", anchor_canonical=a.tolist(), anchor_sd_m=float(sig / np.sqrt(len(tr))),
                   anchor_frozen=frozen_anchor is not None,
                   canonical_drift_max_m=float(drift.max()), canonical_drift_rms_m=float(np.sqrt(np.mean(drift ** 2))),
                   structural="one shared anchor per segment (variable projection)" if seg.slip_mode == "stick_hard"
                   else "latent per-frame contact marginalised against one anchor")
    chi = float(np.mean(r ** 2)) if len(r) else None
    out["chi2_per_dof"] = chi
    out["contradicted"] = bool(chi is not None and chi > contradiction_chi2)
    if out["contradicted"]:
        out["failure_code"] = "CONTACT_MODE_CONTRADICTED"
    if len(ho):
        if seg.slip_mode != "sliding" and len(tr) == 0:
            out["heldout"] = dict(status="INSUFFICIENT_TRAIN_CONTACT")
        else:
            a = None if seg.slip_mode == "sliding" else (np.asarray(frozen_anchor) if frozen_anchor is not None else
                                                          stick_anchor(kind, axis, origin, q, p_tcp, tr))
            if seg.slip_mode == "sliding":
                d, sup = surface.signed_distance(canonical_path(kind, axis, origin, q, p_tcp, ho))
                rh = d[sup] / np.hypot(seg.sigma_contact_surface_m, seg.sigma_tcp_position_m)
            else:
                rh = segment_residuals(kind, axis, origin, q, p_tcp, seg, ho, anchor=a)
            out["heldout"] = dict(status="OK", n=int(len(ho)), chi2_per_dof=float(np.mean(rh ** 2)) if len(rh) else None,
                                  anchor="frozen from training" if a is not None else "n/a (sliding)")
    return out


def contact_discrimination(diag_by_kind: dict, min_ratio=4.0) -> dict:
    """R vs P on IDENTICAL confirmed contact segments: a hypothesis is contact-contradicted when its held-out (else
    training) chi2/dof exceeds the contradiction threshold while the other's does not. Equal fits -> NONDISCRIMINATIVE
    (a remote hinge can follow a straight contact path exactly as well as a slider)."""
    kinds = list(diag_by_kind)
    if len(kinds) < 2:
        return dict(status="NOT_APPLICABLE")
    per = {}
    for k in kinds:
        segs = [d for d in diag_by_kind[k] if d.get("active") and d.get("status") == "OK"]
        chis = [(d.get("heldout") or {}).get("chi2_per_dof") or d.get("chi2_per_dof") for d in segs]
        chis = [c for c in chis if c is not None]
        per[k] = dict(n_segments=len(segs), mean_chi2_per_dof=float(np.mean(chis)) if chis else None,
                      contradicted=[d["segment_id"] for d in segs if d.get("contradicted")])
    if any(v["mean_chi2_per_dof"] is None for v in per.values()):
        return dict(status="ABSTAIN", per_kind=per, reason="no scored confirmed contact segment")
    a, b = kinds
    ca, cb = per[a]["mean_chi2_per_dof"], per[b]["mean_chi2_per_dof"]
    hi, lo = (a, b) if ca > cb else (b, a)
    if per[hi]["contradicted"] and not per[lo]["contradicted"] and max(ca, cb) > min_ratio * max(min(ca, cb), 1e-9):
        return dict(status="OK", favours=lo, rejected=hi, per_kind=per)
    return dict(status="NONDISCRIMINATIVE", per_kind=per,
                reason="both hypotheses explain the confirmed contact equally (or both are contradicted)")


# --------------------------------------------------------------------------- rigid grasp (not wired; see config)
def rigid_grasp_residual(T_WTCP, A_t, T_M_TCP):
    """Log((A_t T_M_TCP)^-1 T_WTCP) as [rotvec, translation]: the SE(3) residual a verified rigid grasp would use.
    Not used by the fitter (rigid_grasp is rejected at parse time until a verified grasp transform exists)."""
    E = np.linalg.inv(A_t @ T_M_TCP) @ T_WTCP
    return np.r_[log_so3(E[:3, :3]), E[:3, 3]]


def tool_rotation_about_point(T_WTCP, angle_rad, axis=(0, 0, 1)):
    """Rotate the TCP frame about its own origin (a point contact allows this; a rigid grasp does not)."""
    T = np.array(T_WTCP, float)
    T[:3, :3] = T[:3, :3] @ exp_so3(normalize(np.asarray(axis, float)) * angle_rad)
    return T


# --------------------------------------------------------------------------- contact-only fit (analytic tests / diagnostic)
def fit_contact_only(kind, rows: ContactRows, T, ref, axis0, origin0=None, q0=None, n_starts=4, seed=0) -> dict:
    """Least squares over (axis, hinge line, q_t != ref) using ONLY contact rows (exact finite SE(3), no
    linearisation). A diagnostic of what the contact alone constrains; the production fit adds these rows to the
    visual ``fit_geometry`` problem."""
    from scipy.optimize import least_squares

    from .rigid_kinematics import tangent_basis
    times = np.unique(np.concatenate([np.atleast_1d(d) for d in rows.row_q_deps])) if rows.n_rows else np.zeros(0, int)
    free = np.array([t for t in times if t != ref], int)
    rng = np.random.default_rng(seed)

    def unpack(p, a0):
        B = np.stack(tangent_basis(a0), 1)
        a = normalize(a0 + B @ p[:2])
        o = None
        k = 2
        if kind == "revolute":
            Bo = np.stack(tangent_basis(a), 1)
            o = np.asarray(origin0, float) + Bo @ p[2:4]
            k = 4
        q = np.zeros(T)
        q[free] = p[k:]
        return a, o, q

    best = None
    for s in range(n_starts):
        a0 = normalize(np.asarray(axis0, float) + (0 if s == 0 else rng.normal(0, 0.2, 3)))
        qi = np.zeros(len(free)) if q0 is None else np.asarray(q0, float)[free]
        p0 = np.r_[np.zeros(2), np.zeros(2) if kind == "revolute" else [], qi]
        sol = least_squares(lambda p: rows.residuals(kind, *unpack(p, a0)), p0, method="trf")
        if best is None or sol.cost < best[0].cost:
            best = (sol, a0)
    sol, a0 = best
    a, o, q = unpack(sol.x, a0)
    J = sol.jac
    try:
        cov = np.linalg.pinv(J.T @ J)
        sv = np.linalg.svd(J, compute_uv=False)
    except np.linalg.LinAlgError:
        cov, sv = None, np.zeros(0)
    return dict(axis=a, origin=o, q=q, cost=float(sol.cost), n_rows=int(len(sol.fun)),
                chi2_per_dof=float(2 * sol.cost / max(len(sol.fun) - len(sol.x), 1)),
                jac_singular_values=sv.tolist() if len(sv) else [],
                condition=float(sv[0] / max(sv[-1], 1e-12)) if len(sv) else None, cov=cov)
