"""Wrench frame/sign handling and physically constrained force evidence (spec section 8).

Sign convention used throughout: ``f_obj`` is the force exerted BY THE ROBOT ON THE OBJECT, in
the world frame. The recorded real datasets store the *reaction on the robot*
(``metadata/contact_force_convention.json``: ``object_side_force_multiplier = -1``), so the
conversion negates once. Force ratios are reported but are never a joint classifier on their
own; directional predictions are only made with an independently supported contact normal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rigid_kinematics import contact_jacobian, normalize


@dataclass(frozen=True)
class WrenchConvention:
    recorded_quantity: str = "object_on_robot"   # 'object_on_robot' | 'robot_on_object'
    gravity_compensated: bool = False
    tare_applied: bool = False
    torque_origin_known: bool = False
    source: str = "unspecified"

    @property
    def object_side_multiplier(self) -> float:
        if self.recorded_quantity == "object_on_robot":
            return -1.0
        if self.recorded_quantity == "robot_on_object":
            return 1.0
        raise ValueError(f"unknown recorded wrench quantity {self.recorded_quantity!r}; refusing to guess a sign")

    @property
    def calibrated(self) -> bool:
        return self.gravity_compensated and self.tare_applied

    @classmethod
    def from_scene_metadata(cls, convention_json: dict, robot_tool_json: dict | None = None):
        q = convention_json.get("recorded_quantity", "")
        rq = "object_on_robot" if ("on_robot" in q or convention_json.get("force_acting_on") == "robot") else (
            "robot_on_object" if convention_json.get("force_acting_on") == "object" else "unknown")
        wf = (robot_tool_json or {}).get("wrench_frame", {})
        return cls(rq, gravity_compensated=False, tare_applied=False,
                   torque_origin_known=not str(wf.get("torque_origin", "unspecified")).startswith("recorded_origin_unspecified"),
                   source="metadata/contact_force_convention.json (no tare/gravity compensation declared)")


def force_on_object_world(f_sensor, R_WS, convention: WrenchConvention):
    """``f_obj^W = s * R_WS f^S`` (rotation only; a force vector is never translated)."""
    f = np.einsum("...ij,...j->...i", np.asarray(R_WS, float), np.asarray(f_sensor, float))
    return convention.object_side_multiplier * f


def torque_about_tcp_world(f_sensor, tau_sensor, R_WS, p_sensor_w, p_tcp_w, convention: WrenchConvention):
    """``tau_TCP = s (R tau_S) + (p_S - p_TCP) x f_obj``; requires a known torque origin."""
    if not convention.torque_origin_known:
        raise ValueError("torque origin not known; refusing to transport the moment to the TCP")
    f_obj = force_on_object_world(f_sensor, R_WS, convention)
    tau = convention.object_side_multiplier * np.einsum("...ij,...j->...i", np.asarray(R_WS, float), tau_sensor)
    return tau + np.cross(np.asarray(p_sensor_w) - np.asarray(p_tcp_w), f_obj)


def decompose_force(f_obj, v_tcp, eps_v=0.002, eps_f=1.0) -> dict:
    """Signed F_par along the TCP tangent, perpendicular vector/magnitude, both ratios.

    Samples with |v| <= eps_v or |f| <= eps_f (or NaN) are excluded with a reason."""
    f = np.asarray(f_obj, float)
    v = np.asarray(v_tcp, float)
    fn = np.linalg.norm(f, axis=-1)
    vn = np.linalg.norm(v, axis=-1)
    reason = np.full(len(f), "", dtype=object)
    reason[~np.isfinite(fn)] = "force_missing"
    reason[(reason == "") & ~np.isfinite(vn)] = "tcp_velocity_missing"
    reason[(reason == "") & (fn <= eps_f)] = "force_below_threshold"
    reason[(reason == "") & (vn <= eps_v)] = "tcp_speed_below_threshold"
    valid = reason == ""
    t = np.where(valid[:, None], v / np.where(vn > 0, vn, 1)[:, None], np.nan)
    Fpar = np.einsum("ti,ti->t", f, t)
    Fperp_vec = f - Fpar[:, None] * t
    Fperp = np.linalg.norm(Fperp_vec, axis=1)
    fs = np.where(fn > 0, fn, 1)
    return dict(valid=valid, reason=reason, F_parallel=Fpar, F_perp_vec=Fperp_vec, F_perp=Fperp,
                cos_phi=Fpar / fs, rho_parallel=np.abs(Fpar) / fs, rho_perp=Fperp / fs,
                f_hat=np.where(valid[:, None], f / fs[:, None], np.nan), force_norm=fn, tangent=t)


def generalized_force(kind, axis, origin, p_tcp, f_obj):
    """``Q_H = f^T J_c,H(p)``: N*m (revolute) or N (prismatic). Not comparable across kinds."""
    J = contact_jacobian(kind, axis, origin, p_tcp)
    return np.einsum("ti,ti->t", np.asarray(f_obj, float), J)


def power_terms(kind, axis, origin, p_tcp, v_tcp, f_obj, qdot) -> dict:
    Q = generalized_force(kind, axis, origin, p_tcp, f_obj)
    P_joint = Q * np.asarray(qdot, float)
    P_tcp = np.einsum("ti,ti->t", np.asarray(f_obj, float), np.asarray(v_tcp, float))
    return dict(Q=Q, P_joint=P_joint, P_tcp=P_tcp, D=P_tcp - P_joint,
                Q_units="N*m" if kind == "revolute" else "N")


def passivity_penalty(D, sigma, delta=0.0, active=None):
    """One-sided ``sum max(0, -D - delta)^2 / sigma^2`` (dissipation must be >= 0)."""
    D = np.asarray(D, float)
    m = np.isfinite(D) if active is None else (np.asarray(active, bool) & np.isfinite(D))
    viol = np.maximum(0.0, -D[m] - delta)
    return dict(penalty=float(np.sum((viol / sigma) ** 2)), n_active=int(m.sum()), n_violations=int((viol > 0).sum()),
                min_D=float(np.min(D[m])) if m.any() else None)


def coulomb_direction(n_in, slip_tangential, mu, eps_slip=1e-3):
    """``f_hat ~ n_in + mu * s_T/|s_T|``; NaN (undefined) when sticking (|s_T| <= eps_slip)."""
    n_in = np.asarray(n_in, float)
    s = np.asarray(slip_tangential, float)
    sn = np.linalg.norm(s, axis=-1)
    stick = sn <= eps_slip
    dirn = n_in + mu * s / np.where(stick, 1.0, sn)[..., None]
    out = dirn / np.linalg.norm(dirn, axis=-1, keepdims=True)
    out[stick] = np.nan
    return out, stick


def predicted_force_directions(kind, axis, q, n0_out, slip, mu, eps_slip=1e-3):
    """Predicted robot-on-object force direction at each time for one hypothesis.

    ``n0_out`` is the canonical OUTWARD face normal; the robot pushes INTO the object, so
    ``n_in = -R_H(q) n0_out``. The object's tangential friction on the tool opposes the tool's
    slip, so the force ON the object has a ``+mu * s_T`` component."""
    from .tcp_contact import surface_normal_world
    n_out = surface_normal_world(kind, axis, q, n0_out)
    n_in = -n_out
    s = np.asarray(slip, float)
    sT = s - np.einsum("ti,ti->t", s, n_out)[:, None] * n_out
    return coulomb_direction(n_in, sT, mu, eps_slip)


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = np.einsum("ti,ti->t", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def fit_shared_mu(f_hat_meas, kind, axis, q, n0_out, slip, active, mu_max=1.5, n_grid=61, eps_slip=1e-3):
    """One shared, bounded mu per hypothesis by grid search on the mean angular error."""
    best = (np.inf, 0.0, None)
    for mu in np.linspace(0.0, mu_max, n_grid):
        pred, stick = predicted_force_directions(kind, axis, q, n0_out, slip, mu, eps_slip)
        m = np.asarray(active, bool) & ~stick & np.isfinite(f_hat_meas).all(1)
        if m.sum() == 0:
            return dict(mu=None, mean_angle_deg=None, n=0, reason="no sliding samples with valid force")
        err = float(np.mean(angle_deg(f_hat_meas[m], pred[m])))
        if err < best[0]:
            best = (err, mu, m)
    return dict(mu=float(best[1]), mean_angle_deg=float(best[0]), n=int(best[2].sum()))


def force_direction_evidence(f_obj, contact_kin_by_kind: dict, candidates: dict, surface_by_kind: dict, active,
                             cfg, sigma_deg=15.0) -> dict:
    """Compare measured full force directions against each hypothesis' Coulomb prediction.

    Abstains (``status = ABSTAIN``) unless every hypothesis has an independently supported surface
    and enough sliding, valid-force samples. Same samples for all hypotheses."""
    fn = np.linalg.norm(f_obj, axis=1)
    f_hat = f_obj / np.where(fn > 0, fn, 1)[:, None]
    out = dict(per_kind={})
    for kind, c in candidates.items():
        surf = surface_by_kind.get(kind)
        if surf is None or surf.status != "OK":
            return dict(status="ABSTAIN", reason=f"no independently supported contact surface for {kind}")
    act = np.asarray(active, bool) & (fn > cfg.min_force_n)
    for kind, c in candidates.items():
        fit = fit_shared_mu(f_hat, kind, c.axis_w, c.q, surface_by_kind[kind].n0, contact_kin_by_kind[kind]["slip"],
                            act, cfg.mu_max)
        out["per_kind"][kind] = fit
    ns = [v["n"] for v in out["per_kind"].values()]
    if min(ns) < 3:
        return dict(out, status="ABSTAIN", reason=f"too few sliding samples with valid force ({ns})")
    out["status"] = "OK"
    out["samples"] = int(min(ns))
    return out


def null_controls(f_obj, rng, shift=5):
    """Wrench null controls: shuffled in time, sign reversed, and time shifted."""
    f = np.asarray(f_obj, float)
    return dict(shuffled=f[rng.permutation(len(f))], sign_reversed=-f, time_shifted=np.roll(f, shift, axis=0))


# =========================================================================== mode-aware force (multimodal spec 6)
def force_gate(calibration=None, f_obj=None) -> dict:
    """Force may enter fitting/selection only when every WrenchCalibration.readiness() gate passes. Failure never
    disables TCP position constraints (those do not depend on the wrench)."""
    if f_obj is None:
        return dict(status="ABSTAIN", reason="no wrench recorded")
    if calibration is None:
        return dict(status="ABSTAIN", reason="no wrench calibration")
    r = calibration.readiness()
    if not r["force_selection_ready"]:
        return dict(status="ABSTAIN", reason=f"wrench gates not passed: {r['failing']}", readiness=r)
    return dict(status="PASS", readiness=r)


def _cand(c):
    return (c.kind if hasattr(c, "kind") else c["kind"], np.asarray(c.axis_w if hasattr(c, "axis_w") else c["axis"], float),
            None if (c.origin_w if hasattr(c, "origin_w") else c.get("origin")) is None
            else np.asarray(c.origin_w if hasattr(c, "origin_w") else c["origin"], float),
            np.asarray(c.q if hasattr(c, "q") else c["q"], float))


def material_velocity_at_tcp(cand, p_tcp, timestamps):
    from .rigid_kinematics import body_velocity
    from .tcp_contact import joint_rate
    kind, a, o, q = _cand(cand)
    qd = joint_rate(q, timestamps)
    return body_velocity(kind, a, o, qd, np.asarray(p_tcp, float)), qd


def force_alignment_diagnostics(f_obj, p_tcp, v_tcp, timestamps, cands: dict, active=None, eps_v=0.002,
                                eps_f=1.0) -> dict:
    """Signed force/TCP-motion alignment and per-hypothesis force vs predicted material motion (spec 6.2).

    rho_par_signed = f.v_TCP / (|f||v_TCP|); cos_phi_H = f.v_M,H / (|f||v_M,H|); Q_H = f.J_H (N for P, N*m for R --
    never compared across types); P_joint = Q_H qdot and P_slip = f.(v_TCP - v_M,H) [W] (their sum is the TCP power
    for EITHER hypothesis: an identity, not evidence)."""
    f = np.asarray(f_obj, float)
    v = np.asarray(v_tcp, float)
    dec = decompose_force(f, v, eps_v, eps_f)
    T = len(f)
    act = np.ones(T, bool) if active is None else np.asarray(active, bool)
    rows = [dict(t=t, valid=bool(dec["valid"][t] and act[t]), reason=str(dec["reason"][t]) or ("inactive" if not act[t] else ""),
                 force_N=float(dec["force_norm"][t]) if np.isfinite(dec["force_norm"][t]) else None,
                 rho_parallel_signed=float(dec["cos_phi"][t]) if dec["valid"][t] else None,
                 rho_perp=float(dec["rho_perp"][t]) if dec["valid"][t] else None) for t in range(T)]
    per = {}
    p = np.asarray(p_tcp, float)
    for k, c in cands.items():
        kind, a, o, q = _cand(c)
        vm, qd = material_velocity_at_tcp(c, p, timestamps)
        Q = generalized_force(kind, a, o, p, f)
        vmn = np.linalg.norm(vm, axis=1)
        fn = np.linalg.norm(f, axis=1)
        cosH = np.einsum("ti,ti->t", f, vm) / np.where((vmn > 0) & (fn > 0), vmn * fn, np.nan)
        Pj = Q * qd
        Ps = np.einsum("ti,ti->t", f, v - vm)
        for t in range(T):
            rows[t][f"cos_phi_{k}"] = float(cosH[t]) if np.isfinite(cosH[t]) else None
            rows[t][f"Q_{k}"] = float(Q[t]) if np.isfinite(Q[t]) else None
            rows[t][f"P_joint_{k}_W"] = float(Pj[t]) if np.isfinite(Pj[t]) else None
            rows[t][f"P_slip_{k}_W"] = float(Ps[t]) if np.isfinite(Ps[t]) else None
        m = np.array([r["valid"] for r in rows]) & np.isfinite(cosH)
        per[k] = dict(Q_units="N*m" if kind == "revolute" else "N", n=int(m.sum()),
                      median_cos_phi=float(np.median(cosH[m])) if m.any() else None,
                      mean_joint_power_W=float(np.mean(Pj[m])) if m.any() else None,
                      positive_joint_power_fraction=float(np.mean(Pj[m] > 0)) if m.any() else None)
    vm_ = np.array([r["valid"] for r in rows])
    rp = np.array([r["rho_parallel_signed"] if r["rho_parallel_signed"] is not None else np.nan for r in rows])
    return dict(per_frame=rows, per_hypothesis=per, n_valid=int(vm_.sum()),
                median_rho_parallel_signed=float(np.nanmedian(rp[vm_])) if vm_.any() else None,
                interpretation="diagnostic only: high force/TCP alignment is expected for a tangentially pushed hinge "
                               "AND for a slider; it is never a joint-type classifier on its own")


def _normals_world(kind, a, o, q, surface, c_canon):
    from .tcp_contact import surface_normal_world
    _, i = surface._tree.query(c_canon)
    n0 = surface.normals[i]
    if kind == "prismatic":
        return n0
    from .rigid_kinematics import exp_so3_axis_angle
    R = exp_so3_axis_angle(normalize(a), q)
    return np.einsum("tij,tj->ti", R, n0)


def kinetic_direction_evidence(f_obj, p_tcp, v_tcp, timestamps, cands: dict, surface, active, train_mask, cfg,
                               rng=None) -> dict:
    """Sliding only: predicted full force direction ~ -n_out + mu s_T/|s_T| (robot on object) with ONE bounded shared
    mu per hypothesis fitted on TRAINING samples; identical samples for every hypothesis (spec 6.3)."""
    from .contact_modes import canonical_path
    f = np.asarray(f_obj, float)
    fn = np.linalg.norm(f, axis=1)
    if surface is None or surface.status != "OK":
        return dict(status="ABSTAIN", reason="no independently observed contact surface (normal unknown)")
    T = len(f)
    act = np.asarray(active, bool) & (fn > cfg.min_force_n)
    sTs, nouts = {}, {}
    common = act.copy()
    for k, c in cands.items():
        kind, a, o, q = _cand(c)
        t_all = np.arange(T)
        cc = canonical_path(kind, a, o, q, p_tcp, t_all)
        d, sup = surface.signed_distance(cc)
        nW = _normals_world(kind, a, o, q, surface, cc)
        vm, _ = material_velocity_at_tcp(c, p_tcp, timestamps)
        s = np.asarray(v_tcp, float) - vm
        sT = s - np.einsum("ti,ti->t", s, nW)[:, None] * nW
        sTs[k], nouts[k] = sT, nW
        common &= sup & (np.linalg.norm(sT, axis=1) > cfg.eps_slip_mps) & np.isfinite(sT).all(1)
    tr = common & np.asarray(train_mask, bool)
    te = common & ~np.asarray(train_mask, bool)
    if tr.sum() < 3:
        return dict(status="ABSTAIN", reason=f"{int(tr.sum())} training samples with verified contact, observed normal "
                                              "and measurable tangential slip (kinetic direction undefined otherwise)",
                    n_common=int(common.sum()))
    fh = f / np.where(fn > 0, fn, 1)[:, None]
    per = {}
    for k in cands:
        best = None
        for mu in np.linspace(0, cfg.mu_max, 61):
            d = -nouts[k] + mu * sTs[k] / np.maximum(np.linalg.norm(sTs[k], axis=1, keepdims=True), 1e-12)
            pred = d / np.linalg.norm(d, axis=1, keepdims=True)
            err = angle_deg(fh[tr], pred[tr]).mean()
            if best is None or err < best[0]:
                best = (err, mu, pred)
        err_te = float(angle_deg(fh[te], best[2][te]).mean()) if te.any() else None
        per[k] = dict(mu=float(best[1]), train_mean_angle_deg=float(best[0]), heldout_mean_angle_deg=err_te,
                      n_train=int(tr.sum()), n_heldout=int(te.sum()))
    worst = max(v["train_mean_angle_deg"] for v in per.values())
    out = dict(status="OK", per_kind=per, n_common=int(common.sum()), mu_fitted_on="training samples only")
    if min(v["train_mean_angle_deg"] for v in per.values()) > 3 * cfg.sigma_direction_deg:
        out.update(status="CONTACT_MODEL_INADEQUATE", reason=f"force direction unexplained by every hypothesis "
                                                             f"(best mean error {worst:.1f} deg)")
    return out


def static_contact_feasibility(f_obj, p_tcp, cands: dict, surface, segments, cfg) -> dict:
    """Sticking contacts: NO kinetic direction (undefined at zero slip). For a declared unilateral point contact with an
    observed normal, test the friction cone f_N >= 0, |f_T| <= mu_s f_N -- feasibility, never a 'perfect' zero
    residual. Grasps / unknown contact models abstain (tension or a free couple are legitimate there)."""
    from .contact_modes import canonical_path
    f = np.asarray(f_obj, float)
    out = dict(per_segment=[])
    for s in segments:
        if not s.active_for_residuals or s.slip_mode == "sliding":
            continue
        if s.contact_model != "unilateral_point":
            out["per_segment"].append(dict(segment_id=s.segment_id, status="ABSTAIN",
                                           reason=f"contact_model {s.contact_model}: single-point Coulomb cone not "
                                                  "applicable (grasp / multi-contact / unknown)"))
            continue
        if surface is None or surface.status != "OK":
            out["per_segment"].append(dict(segment_id=s.segment_id, status="ABSTAIN", reason="no observed normal"))
            continue
        t = np.asarray(s.model_time_indices, int)
        seg = dict(segment_id=s.segment_id, status="OK", kinetic_direction="ABSTAIN (zero slip: undefined)",
                   per_kind={})
        for k, c in cands.items():
            kind, a, o, q = _cand(c)
            cc = canonical_path(kind, a, o, q, p_tcp, t)
            nW = _normals_world(kind, a, o, q[t], surface, cc)
            fN = -np.einsum("ti,ti->t", f[t], nW)                # robot pushes INTO the object (-n_out)
            fT = np.linalg.norm(f[t] + fN[:, None] * nW, axis=1)
            feas = (fN >= 0) & (fT <= cfg.mu_static_max * np.maximum(fN, 0))
            seg["per_kind"][k] = dict(feasible_fraction=float(feas.mean()), min_normal_force_N=float(fN.min()),
                                      max_tangent_ratio=float(np.max(fT / np.maximum(fN, 1e-9))))
        fr = [v["feasible_fraction"] for v in seg["per_kind"].values()]
        seg["discriminative"] = bool(max(fr) - min(fr) > 0.5)
        out["per_segment"].append(seg)
    out["status"] = "OK" if any(s.get("status") == "OK" for s in out["per_segment"]) else "ABSTAIN"
    return out


def _vmf_kappa(rbar):
    rbar = float(np.clip(rbar, 1e-6, 0.999))
    return float(np.clip((3 * rbar - rbar ** 3) / (1 - rbar ** 2), 1e-3, 500.0))


def _vmf_ll(cos, kappa):
    # 3D von Mises-Fisher: log C3 + kappa cos, C3 = kappa / (4 pi sinh kappa)
    return np.log(kappa) - np.log(4 * np.pi) - (kappa + np.log1p(-np.exp(-2 * kappa)) - np.log(2)) + kappa * cos


def mechanical_model_evidence(f_obj, p_tcp, timestamps, cands: dict, active, train_mask, cfg, seed=0) -> dict:
    """Discriminative force evidence under an EXPLICITLY DECLARED actuation model (spec 6.2):

    ``min_effort_tangent``: the robot pushes along the predicted material velocity at the contact,
    f_hat ~ vMF(v_M,H / |v_M,H|, kappa_H), kappa_H fitted on training samples only. Evidence counts only if
    (1) the hypotheses predict directions that differ by >= min_direction_separation_deg on the common samples, and
    (2) the held-out log-likelihood difference exceeds every null control (shuffled, sign-reversed, time-shifted,
    random directions). Otherwise FORCE_NONDISCRIMINATIVE. ``none`` (default): never discriminative."""
    if cfg.mechanical_model == "none":
        return dict(status="FORCE_NONDISCRIMINATIVE", reason="no mechanical actuation model declared; a force/motion "
                                                             "alignment ratio is not a likelihood")
    f = np.asarray(f_obj, float)
    fn = np.linalg.norm(f, axis=1)
    fh = f / np.where(fn > 0, fn, 1)[:, None]
    vh = {}
    common = np.asarray(active, bool) & (fn > cfg.min_force_n)
    for k, c in cands.items():
        vm, _ = material_velocity_at_tcp(c, p_tcp, timestamps)
        n = np.linalg.norm(vm, axis=1)
        vh[k] = vm / np.where(n > 0, n, 1)[:, None]
        common &= n > 1e-4
    ks = list(cands)
    sep = float(np.mean(angle_deg(vh[ks[0]][common], vh[ks[1]][common]))) if common.any() and len(ks) == 2 else 0.0
    tr, te = common & np.asarray(train_mask, bool), common & ~np.asarray(train_mask, bool)
    out = dict(model=cfg.mechanical_model, n_train=int(tr.sum()), n_heldout=int(te.sum()),
               predicted_direction_separation_deg=sep)
    if tr.sum() < 3 or te.sum() < 2:
        return dict(out, status="ABSTAIN", reason="too few common training/held-out force samples")
    if sep < cfg.min_direction_separation_deg:
        return dict(out, status="FORCE_NONDISCRIMINATIVE",
                    reason=f"hypotheses predict force directions only {sep:.1f} deg apart")

    def delta(F):
        ll = {}
        for k in ks:
            kap = _vmf_kappa(np.mean(np.einsum("ti,ti->t", F[tr], vh[k][tr])))
            ll[k] = float(np.sum(_vmf_ll(np.einsum("ti,ti->t", F[te], vh[k][te]), kap)))
        return ll[ks[1]] - ll[ks[0]], ll
    d, ll = delta(fh)
    rng = np.random.default_rng(seed)
    unit = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
    # information-free nulls: no joint information at all -> must never reach the real improvement
    free = dict(sign_reversed=delta(-fh)[0], gravity_only=delta(np.tile([0.0, 0.0, -1.0], (len(fh), 1)))[0])
    const = [delta(np.tile(unit(rng.normal(size=3)), (len(fh), 1)))[0] for _ in range(cfg.n_null)]
    perframe = [delta(unit(rng.normal(size=fh.shape)))[0] for _ in range(cfg.n_null)]
    free["random_constant_direction_q99"] = float(np.percentile(np.abs(const), 99))
    free["random_per_frame_direction_q99"] = float(np.percentile(np.abs(perframe), 99))
    # time-structure nulls: only binding when a hypothesis predicts a time-varying force direction
    timed = dict(time_shuffled=delta(fh[rng.permutation(len(fh))])[0], time_shifted=delta(np.roll(fh, 5, axis=0))[0])
    fav = ks[1] if d > 0 else ks[0]
    temporal = {k: float(np.degrees(np.arccos(np.clip(np.einsum("ti,i->t", vh[k][common],
                                                                unit(vh[k][common].mean(0))), -1, 1))).max())
                for k in ks}
    # time-structure nulls bind only when the favoured hypothesis claims a time-varying direction: a constant push
    # along a slider is legitimately preserved by shuffling time (its information is the mean direction)
    time_binding = temporal[fav] > 10.0
    binding = [abs(v) for v in free.values()] + ([abs(v) for v in timed.values()] if time_binding else [])
    out.update(heldout_ll=ll, delta_ll=dict(value=d, sense=f"{ks[1]} - {ks[0]}"), null_controls=dict(free, **timed),
               predicted_direction_time_variation_deg=temporal,
               time_nulls_binding=bool(time_binding), null_threshold=float(max(binding)))
    if abs(d) <= max(binding):
        return dict(out, status="FORCE_NONDISCRIMINATIVE", reason="improvement not above the null controls")
    return dict(out, status="OK", favours=fav)


class ForceDirectionRows:
    """L_force for ``fit_geometry`` (declared mechanical model, verified wrench only): (f_hat - v_hat_M,H)/sigma per
    contact sample. The material velocity uses the finite-difference qdot (rows depend on q at t-1, t, t+1)."""

    def __init__(self, f_obj, p_tcp, timestamps, times, sigma_deg):
        f = np.asarray(f_obj, float)
        self.fh = f / np.linalg.norm(f, axis=1, keepdims=True)
        self.p = np.asarray(p_tcp, float)
        self.ts = np.asarray(timestamps, float)
        self.times = np.asarray(times, int)
        self.sig = np.radians(sigma_deg)
        self.row_times = np.repeat(self.times, 3)
        self.row_q_deps = [np.array([max(t - 1, 0), t, min(t + 1, len(self.ts) - 1)]) for t in self.row_times]

    @property
    def n_rows(self):
        return len(self.row_times)

    def residuals(self, kind, axis, origin, q):
        vm, _ = material_velocity_at_tcp(dict(kind=kind, axis=axis, origin=origin, q=q), self.p, self.ts)
        v = vm[self.times]
        vh = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        return ((self.fh[self.times] - vh) / self.sig).ravel()


class CombinedRows:
    """Concatenate extra hooks (TCP contact rows, force rows) for ``fit_geometry``; keeps them separable."""

    def __init__(self, *hooks):
        self.hooks = [h for h in hooks if h is not None and h.n_rows > 0]
        self.row_times = np.concatenate([h.row_times for h in self.hooks]) if self.hooks else np.zeros(0, int)
        self.row_q_deps = [d for h in self.hooks for d in getattr(h, "row_q_deps", [[t] for t in h.row_times])]

    @property
    def n_rows(self):
        return len(self.row_times)

    def residuals(self, kind, axis, origin, q):
        return np.concatenate([h.residuals(kind, axis, origin, q) for h in self.hooks]) if self.hooks else np.zeros(0)
