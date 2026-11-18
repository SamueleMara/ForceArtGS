"""Stage E: TCP contact and calibrated force IN candidate refinement and model comparison (gated).

S2 = vision only (the existing selection). S3 = vision + independently confirmed sliding TCP contact.
S4 = S3 + calibrated force direction (Coulomb, one shared bounded mu) + one-sided passivity.

Gates (any failure -> ``ABSTAIN`` for that level, vision decision untouched):
* contact: >= ``min_samples`` model times with ``contact_confirmed`` (independent annotation) AND an
  observed, planar moving-surface patch along the canonical contact path for EVERY hypothesis;
  all hypotheses use the same (intersection) contact samples. A force threshold never qualifies.
* force: ``WrenchCalibration.readiness()`` passes every gate (rotation, sign, tare, gravity, sync).

Contact rows (whitened, primary): ``n0 . (A_H(q_t)^-1 p_tcp(t) - a0) / sigma_contact`` with the
plane (a0, n0) fixed from ORIGINAL observed geometry before refinement (anchors cannot drift to
satisfy the contact). The normal-velocity residual is correlated with it and stays a diagnostic.
Force rows: ``(f_hat_meas - f_hat_pred,H) / sigma_dir`` on sliding samples (|s_T| > eps), with
``f_hat_pred = normalize(n_in + mu s_T/|s_T|)``; mu is fit once on training samples per hypothesis
(bounded grid) and then held fixed. Passivity rows: ``max(0, -f . slip) / sigma_P``.

Held-out evaluation never refits on test times. The combined score adds the held-out contact/force
NLL differences (per sample) to the cluster-robust vision difference; variances add.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np

from .geometry_fit import _select_fit_tracks, fit_geometry
from .rigid_kinematics import apply_joint_inverse, body_velocity, exp_so3_axis_angle, normalize
from .tcp_contact import UNAVAILABLE, joint_rate, surface_from_observed_patch, tcp_velocity
from .wrench_physics import fit_shared_mu

ABSTAIN = "ABSTAIN"


@dataclasses.dataclass
class PhysicalConfig:
    min_samples: int = 3
    sigma_contact_m: float = 0.005
    sigma_dir: float = 0.25             # unit-vector difference ~ 14 deg
    sigma_power_w: float = 0.01
    min_force_n: float = 1.0
    eps_slip_mps: float = 1e-3
    mu_max: float = 1.5
    max_nfev: int = 30
    max_force_angle_deg: float = 30.0   # goodness of fit: best hypothesis must explain force directions


def _rot(kind, axis, q):
    if kind == "prismatic":
        return np.broadcast_to(np.eye(3), (len(np.atleast_1d(q)), 3, 3))
    return exp_so3_axis_angle(normalize(axis), np.atleast_1d(np.asarray(q, float)))


class PhysicalResiduals:
    """Residual hook for ``geometry_fit.fit_geometry`` (see its ``extra`` argument)."""

    def __init__(self, p_tcp, timestamps, contact_t, plane, pc: PhysicalConfig, force=None, force_t=None, mu=None):
        self.p, self.ts, self.pc = np.asarray(p_tcp, float), np.asarray(timestamps, float), pc
        self.ct = np.asarray(contact_t, int)
        self.a0, self.n0 = np.asarray(plane[0], float), normalize(np.asarray(plane[1], float))
        self.f = None if force is None else np.asarray(force, float)
        self.ft = np.zeros(0, int) if force_t is None else np.asarray(force_t, int)
        self.mu = mu
        self.v_tcp = tcp_velocity(self.p, self.ts)
        self.row_times = np.concatenate([self.ct, np.repeat(self.ft, 4)])

    def contact_rows(self, kind, axis, origin, q):
        x0 = np.stack([apply_joint_inverse(kind, axis, origin, q[t], self.p[t]) for t in self.ct]) if len(self.ct) else np.zeros((0, 3))
        return ((x0 - self.a0) @ self.n0) / self.pc.sigma_contact_m

    def force_rows(self, kind, axis, origin, q):
        if self.f is None or len(self.ft) == 0:
            return np.zeros(0)
        qd = joint_rate(q, self.ts)
        out = []
        for t in self.ft:
            n_out = _rot(kind, axis, q[t])[0] @ self.n0
            vm = body_velocity(kind, axis, origin, qd[t], self.p[t])
            s = self.v_tcp[t] - vm
            sT = s - (s @ n_out) * n_out
            fm = self.f[t] / max(np.linalg.norm(self.f[t]), 1e-12)
            if np.linalg.norm(sT) > self.pc.eps_slip_mps:
                pred = -n_out + self.mu * sT / np.linalg.norm(sT)
                pred = pred / np.linalg.norm(pred)
                r = (fm - pred) / self.pc.sigma_dir
            else:
                r = np.zeros(3)
            D = self.f[t] @ s
            out.append(np.r_[r, max(0.0, -D) / self.pc.sigma_power_w])
        return np.concatenate(out)

    def residuals(self, kind, axis, origin, q):
        return np.concatenate([self.contact_rows(kind, axis, origin, q), self.force_rows(kind, axis, origin, q)])


def _surface(c, p_tcp, times, calibs, cams, cfg_contact):
    x0 = np.stack([apply_joint_inverse(c.kind, c.axis_w, c.origin_w, c.q[t], p_tcp[t]) for t in times])
    mv = np.nan_to_num(c.moving_probability) > 0.5
    centers = [apply_joint_inverse(c.kind, c.axis_w, c.origin_w, 0.0, calibs[k].center_w) for k in cams]
    return surface_from_observed_patch(c.canonical_anchors_w[mv], x0, cfg_contact, centers), x0


def physical_stage(fit_tr, test_obs, decision, cfg, physical: dict, out_dir=None) -> dict:
    """Run S3/S4 under gates. ``physical``: dict(contact=ContactRecord, calibration=WrenchCalibration,
    calibs=..., timeline=ContactTimeline|None, physical_config=PhysicalConfig|None)."""
    from .model_selection import score_candidates_on_heldout
    pc = physical.get("physical_config") or PhysicalConfig()
    rec, cal, timeline = physical["contact"], physical["calibration"], physical.get("timeline")
    calibs = physical["calibs"]
    obs = fit_tr.problem.obs
    T = obs.frame_map.T
    heldout = np.zeros(T, bool)
    heldout[np.unique(test_obs.time)] = True
    ready = cal.readiness()
    res = dict(status=ABSTAIN, selection_terms=[], used_for_contact_fit=False, used_for_force_fit=False,
               calibration_verified=bool(ready["force_selection_ready"]), wrench_readiness=ready,
               S2=dict(decision=decision.label, reasons=decision.reasons))
    # ---------------- contact gate
    if timeline is None:
        res["reason"] = "no independently confirmed contact intervals (force thresholds are heuristics, not annotations)"
        res["contact_gate"] = "FAIL"
        return res
    tl = timeline.select(obs.frame_map.source_frame_index)
    confirmed = tl.hard_constraint_mask & np.isfinite(rec.p_tcp_w).all(1)
    res["n_confirmed_contact_times"] = int(confirmed.sum())
    if confirmed.sum() < pc.min_samples:
        res.update(reason=f"only {int(confirmed.sum())} confirmed contact times", contact_gate="FAIL")
        return res
    cands = fit_tr.candidates
    surf, support = {}, np.ones(T, bool)
    times = np.flatnonzero(confirmed)
    for k, c in cands.items():
        s, x0 = _surface(c, rec.p_tcp_w, times, calibs, obs.camera_ids, cfg.contact)
        surf[k] = s
        if s.status != "OK":
            res.update(reason=f"CONTACT_NORMAL_UNAVAILABLE for {k}: {s.reason}", contact_gate="FAIL")
            return res
        sup = np.zeros(T, bool)
        sup[times] = s.support_count(x0, cfg.contact.surface_patch_radius_m) >= 3
        support &= sup
    common = confirmed & support
    tr_t, te_t = np.flatnonzero(common & ~heldout), np.flatnonzero(common & heldout)
    res.update(contact_gate="PASS", n_common_contact_train=int(len(tr_t)), n_common_contact_test=int(len(te_t)),
               surfaces={k: dict(n0=s.n0.tolist(), a0=s.a0.tolist(), rms_m=s.rms_m, n_points=s.n_points) for k, s in surf.items()})
    if len(tr_t) < pc.min_samples:
        res.update(status=ABSTAIN, reason=f"only {len(tr_t)} common supported training contact samples")
        return res
    # ---------------- force gate
    force_ok = bool(ready["force_selection_ready"]) and rec.wrench_sensor is not None and rec.R_WS is not None
    f_w = None
    if force_ok:
        f_w = cal.force_on_object_world(rec.wrench_sensor[:, :3], rec.R_WS)["f_object_w"]
    res["force_gate"] = "PASS" if force_ok else f"FAIL: {ready['failing'] or 'no wrench'}"

    def refine(level):
        new = {}
        mus = {}
        for k, c in cands.items():
            ft = None
            mu = None
            if level == "S4":
                fn = np.linalg.norm(f_w, axis=1)
                ft = tr_t[fn[tr_t] > pc.min_force_n]
                fh = f_w / np.where(fn > 0, fn, 1)[:, None]
                v_tcp = tcp_velocity(rec.p_tcp_w, rec.timestamp_s)
                from .rigid_kinematics import relative_slip
                slip = relative_slip(k, c.axis_w, c.origin_w, joint_rate(c.q, rec.timestamp_s), rec.p_tcp_w, v_tcp)
                act = np.zeros(T, bool)
                act[ft] = True
                m = fit_shared_mu(fh, k, c.axis_w, c.q, surf[k].n0, slip, act, pc.mu_max)
                mu = m["mu"] if m["mu"] is not None else 0.0
                mus[k] = m
            hook = PhysicalResiduals(rec.p_tcp_w, rec.timestamp_s, tr_t, (surf[k].a0, surf[k].n0), pc,
                                     force=f_w if level == "S4" else None, force_t=ft, mu=mu)
            prob = fit_tr.problem
            post = np.nan_to_num(c.moving_probability)
            tracks = _select_fit_tracks(prob, post, np.random.default_rng(cfg.fit.seed))
            a, o, q, X, q_unc, obs_t, info = fit_geometry(k, c.axis_w, c.origin_w, c.q, np.nan_to_num(c.canonical_anchors_w),
                                                         tracks, post, prob, max_nfev=pc.max_nfev, extra=hook)
            X = np.where(np.isfinite(c.canonical_anchors_w), X, np.nan)
            nc = dataclasses.replace(c, axis_w=a, origin_w=o, q=q, q_uncertainty=q_unc, canonical_anchors_w=X,
                                     diagnostics=dict(c.diagnostics, physical_refinement=dict(level=level, info=info)))
            nc._ll_moving = getattr(c, "_ll_moving", None)
            new[k] = nc
        return new, mus

    def heldout_physical_nll(cs, level, mus):
        out = {}
        for k, c in cs.items():
            hook = PhysicalResiduals(rec.p_tcp_w, rec.timestamp_s, te_t, (surf[k].a0, surf[k].n0), pc)
            r = hook.contact_rows(k, c.axis_w, c.origin_w, c.q)
            nll = 0.5 * r ** 2
            if level == "S4":
                fn = np.linalg.norm(f_w, axis=1)
                ft = te_t[fn[te_t] > pc.min_force_n]
                hf = PhysicalResiduals(rec.p_tcp_w, rec.timestamp_s, np.zeros(0, int), (surf[k].a0, surf[k].n0), pc,
                                       force=f_w, force_t=ft, mu=(mus.get(k) or {}).get("mu") or 0.0)
                rf = hf.force_rows(k, c.axis_w, c.origin_w, c.q).reshape(-1, 4)
                per = 0.5 * np.sum(rf ** 2, 1)
                full = np.zeros(len(te_t))
                full[np.isin(te_t, ft)] = per
                nll = nll + full
            out[k] = nll
        return out

    levels = ["S3"] + (["S4"] if force_ok else [])
    for level in list(levels):
        cs, mus = refine(level)
        if level == "S4":
            best = min((m.get("mean_angle_deg") if m.get("mean_angle_deg") is not None else np.inf) for m in mus.values())
            if not best <= pc.max_force_angle_deg:
                res["S4"] = dict(status=ABSTAIN, mu=mus, reason=f"force directions unexplained by every hypothesis "
                                 f"(best mean angle {best:.1f} deg > {pc.max_force_angle_deg}): sign/frame/time error suspected")
                levels.remove("S4")
                force_ok = False
                res["force_gate"] = "FAIL: goodness of fit"
                continue
        fit_mod = copy.copy(fit_tr)
        fit_mod.candidates = cs
        dec = score_candidates_on_heldout(fit_mod, test_obs, cfg)
        phys_nll = heldout_physical_nll(cs, level, mus)
        entry = dict(vision_decision_after_refinement=dec.label, vision_reasons=dec.reasons, mu=mus,
                     candidates={k: dict(axis=c.axis_w.tolist(), origin=None if c.origin_w is None else c.origin_w.tolist(),
                                         q_span=float(np.nanmax(c.q) - np.nanmin(c.q))) for k, c in cs.items()},
                     parameter_change_vs_S2={k: _change(cands[k], cs[k]) for k in cs},
                     heldout_physical_nll={k: float(v.sum()) for k, v in phys_nll.items()})
        if "revolute" in cs and "prismatic" in cs:
            v = dec.scores.get("revolute_vs_prismatic", {})
            dphys = phys_nll["prismatic"] - phys_nll["revolute"]
            var_v = (v.get("delta_nll", 0.0) / v["z"]) ** 2 if v.get("z") else np.inf
            var_p = float(len(dphys) * np.var(dphys, ddof=1)) if len(dphys) > 1 else 0.0
            tot = v.get("delta_nll", 0.0) + float(dphys.sum())
            z = tot / np.sqrt(var_v + var_p) if np.isfinite(var_v + var_p) and var_v + var_p > 0 else 0.0
            entry["combined_revolute_vs_prismatic"] = dict(delta_vision=v.get("delta_nll"), delta_physical=float(dphys.sum()),
                                                           z=float(z), n_physical_samples=int(len(dphys)))
            zt = cfg.comparison.z_threshold
            entry["decision"] = dec.label if dec.label in ("static/no_motion",) else (
                "ambiguous/unobservable" if abs(z) < zt else ("revolute" if z > 0 else "prismatic"))
        res[level] = entry
        if level == "S4":
            entry["null_controls"] = _force_nulls(cs, rec, f_w, te_t, tr_t, surf, pc, cfg)
    res.update(status="OK", selection_terms=["contact"] + (["force_direction", "passivity"] if force_ok else []),
               used_for_contact_fit=True, used_for_force_fit=force_ok,
               reason="S3 run" + (" and S4 run" if force_ok else "; S4 abstained (wrench calibration gates)"))
    res["integration_changed_parameters"] = any(
        res[lv]["parameter_change_vs_S2"][k]["axis_change_deg"] > 1e-6 or abs(res[lv]["parameter_change_vs_S2"][k]["q_span_change"]) > 1e-9
        for lv in levels for k in res[lv]["parameter_change_vs_S2"])
    res["decision_changed_vs_S2"] = {lv: res[lv].get("decision") != decision.label for lv in levels}
    return res


def _change(a, b) -> dict:
    from .rigid_kinematics import axis_angle_deg, line_to_line_distance
    out = dict(axis_change_deg=float(axis_angle_deg(a.axis_w, b.axis_w)),
               q_span_change=float((np.nanmax(b.q) - np.nanmin(b.q)) - (np.nanmax(a.q) - np.nanmin(a.q))))
    if a.kind == "revolute":
        out["hinge_line_change_m"] = float(line_to_line_distance(a.axis_w, a.origin_w, b.axis_w, b.origin_w))
    return out


def _force_nulls(cs, rec, f_w, te_t, tr_t, surf, pc, cfg) -> dict:
    """Shuffled-time, sign-reversed and gravity-only wrench: held-out force-direction NLL difference
    (prismatic - revolute) per null; a working pipeline must not let these decide the type."""
    rng = np.random.default_rng(cfg.fit.seed)
    nulls = dict(shuffled_time=f_w[rng.permutation(len(f_w))], sign_reversed=-f_w,
                 gravity_only=np.tile([0.0, 0.0, -9.81 * 0.5], (len(f_w), 1)))
    out = {}
    for name, f in nulls.items():
        tot = {}
        for k, c in cs.items():
            fn = np.linalg.norm(f, axis=1)
            ft = te_t[fn[te_t] > pc.min_force_n]
            fh = f / np.where(fn > 0, fn, 1)[:, None]
            v_tcp = tcp_velocity(rec.p_tcp_w, rec.timestamp_s)
            from .rigid_kinematics import relative_slip
            slip = relative_slip(k, c.axis_w, c.origin_w, joint_rate(c.q, rec.timestamp_s), rec.p_tcp_w, v_tcp)
            act = np.zeros(len(f), bool)
            act[tr_t[fn[tr_t] > pc.min_force_n]] = True
            mu = fit_shared_mu(fh, k, c.axis_w, c.q, surf[k].n0, slip, act, pc.mu_max)["mu"] or 0.0
            h = PhysicalResiduals(rec.p_tcp_w, rec.timestamp_s, np.zeros(0, int), (surf[k].a0, surf[k].n0), pc,
                                  force=f, force_t=ft, mu=mu)
            r = h.force_rows(k, c.axis_w, c.origin_w, c.q).reshape(-1, 4)
            tot[k] = 0.5 * np.sum(r ** 2, 1)
        if "revolute" in tot and "prismatic" in tot and len(tot["revolute"]) > 1:
            d = tot["prismatic"] - tot["revolute"]
            se = np.std(d, ddof=1) * np.sqrt(len(d))
            gof = {k: float(np.degrees(2 * np.arcsin(np.clip(np.sqrt(2 * tot[k].mean()) * pc.sigma_dir / 2, 0, 1))))
                   for k in tot}  # rough mean angle from the direction-row NLL
            out[name] = dict(delta=float(d.sum()), z=float(d.sum() / se) if se > 0 else 0.0, n=int(len(d)),
                             would_abstain_goodness_of_fit=bool(min(gof.values()) > pc.max_force_angle_deg),
                             approx_mean_angle_deg=gof,
                             mean_nll_per_sample=dict(revolute=float(tot["revolute"].mean()),
                                                      prismatic=float(tot["prismatic"].mean())))
        else:
            out[name] = dict(status=ABSTAIN, reason="too few valid force samples")
    return out


# =========================================================================== joint selection v2 (spec section 7)
PHYSICAL_STATUSES = ("IDENTIFIED_REVOLUTE", "IDENTIFIED_PRISMATIC", "TYPE_UNOBSERVABLE", "MODEL_INCONSISTENT",
                     "NO_SIGNIFICANT_MOTION")
WORKING_MODELS = ("revolute", "prismatic", "static", "none")


@dataclasses.dataclass
class PhysicalDecision:
    physical_joint_status: str
    working_motion_model: str
    working_model_is_approximation: bool
    physics_joint_export_ready: bool
    selected_candidate: str | None
    reasons: list
    evidence: dict
    uncertainty: dict
    failure_codes: list
    counterfactual: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.physical_joint_status not in PHYSICAL_STATUSES:
            raise ValueError(f"unknown physical_joint_status {self.physical_joint_status}")
        if self.working_motion_model not in WORKING_MODELS:
            raise ValueError(f"unknown working_motion_model {self.working_motion_model}")
        if self.physics_joint_export_ready and not self.physical_joint_status.startswith("IDENTIFIED_"):
            raise ValueError("physics_joint_export_ready requires an IDENTIFIED_* physical status")

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def _axis_angle(a, b):
    if a is None or b is None:
        return None
    a, b = normalize(np.asarray(a, float)), normalize(np.asarray(b, float))
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def select_physical_joint(*, motion_evidence, model_fits, common_support, attachment_evidence, predictive_evidence,
                          render_evidence=None, robot_evidence=None, cfg=None) -> PhysicalDecision:
    """Evidence hierarchy of spec section 7, in order. Never converts ambiguity into a physical joint.

    ``motion_evidence``: dict(rotation=rotation_vs_translation_evidence, shared_axis, hinge, null, support_quality)
    ``model_fits``: {kind: dict(axis, origin)} training-split fits
    ``common_support``: dict(counts, consistency)  (support_consistency)
    ``attachment_evidence``: dict(interface, hinge_field, hinge_fit, guide, implied)
    ``predictive_evidence``: dict(separation, heldout)
    """
    c = dict(min_moving_tracks=20, max_axis_disagreement_deg=15.0, max_camera_axis_disagreement_deg=25.0,
             max_support_noise_ratio=2.0, prismatic_margin=3.0, render_min_blocks=6, render_min_seeds=2)
    c.update(cfg or {})
    rot = motion_evidence.get("rotation", {})
    sa = motion_evidence.get("shared_axis", {})
    hinge = motion_evidence.get("hinge", {})
    null = motion_evidence.get("null", {}) or {}
    quality = motion_evidence.get("support_quality", {})
    cons = common_support.get("consistency", {})
    counts = common_support.get("counts", {})
    sep = predictive_evidence.get("separation", {})
    ho = predictive_evidence.get("heldout", {})
    reasons, codes = [], []
    cf = {}
    ev = dict(moving_vs_static={}, rotation_field=rot, translation_field={}, revolute_vs_prismatic=dict(
        prediction_separation=sep, heldout_block_preference=ho), hinge_geometry=hinge,
        assembly_geometry=attachment_evidence, visual_counterfactual=render_evidence or dict(status="NOT_RUN"),
        tcp_contact=(robot_evidence or {}).get("contact", dict(status="ABSTAIN", reason="no robot evidence given")),
        force=(robot_evidence or {}).get("force", dict(status="ABSTAIN", reason="no robot evidence given")))
    unc = dict(net_rotation_ci95_deg=rot.get("net_rotation_ci95_deg"), rotation_upper_bound_deg=rot.get(
        "rotation_upper_bound_deg"), null_q95_deg=null.get("net_abs_rotation_q95_deg"),
        axis_cone95_deg=sa.get("axis_cone95_deg"), hinge_line_sd_m=hinge.get("line_sd_m"))

    def decide(status, working, sel, ready=False):
        return PhysicalDecision(status, working, not status.startswith("IDENTIFIED_"), ready, sel, reasons, ev, unc,
                                codes, cf)

    # ---------------------------------------------------------------- 1. significant motion?
    trans = rot.get("net_centroid_translation_norm_m")
    tnull = null.get("net_translation_q95_m")
    n_mov = counts.get("M_proposal", 0)
    moving = n_mov >= c["min_moving_tracks"] and (
        bool(rot.get("rotation_detected")) or (trans is not None and tnull is not None and trans > tnull))
    ev["moving_vs_static"] = dict(n_moving_proposal=n_mov, net_translation_m=trans, translation_null_q95_m=tnull,
                                  rotation_detected=rot.get("rotation_detected"), significant=bool(moving))
    ev["translation_field"] = dict(net_translation_m=trans, null_q95_m=tnull,
                                   path_length_m=rot.get("path_length_m"))
    if not moving:
        reasons.append(f"no significant motion: {n_mov} moving-proposal tracks, net translation {trans} m vs null "
                       f"q95 {tnull} m, rotation_detected={rot.get('rotation_detected')}")
        codes.append("NO_MOTION")
        return decide("NO_SIGNIFICANT_MOTION", "static", None)
    # ---------------------------------------------------------------- 2. one coherent 1-DoF rigid field?
    if cons.get("inconsistent"):
        reasons += cons.get("inconsistency_flags", [])
        codes.append("SUPPORT_INCONSISTENT")
        cf = dict(revolute="one rigid field does not explain the moving support", prismatic="idem")
        return decide("MODEL_INCONSISTENT", "none", None)
    noise_ratio = quality.get("moving_over_static_residual")
    reliable = noise_ratio is not None and noise_ratio <= c["max_support_noise_ratio"]
    cam = rot.get("per_camera") or {}
    cam_axes_ok = all(v.get("axis_vs_pooled_deg") is None or v["axis_vs_pooled_deg"] <= c["max_camera_axis_disagreement_deg"]
                      for v in cam.values())
    # ---------------------------------------------------------------- 3. rotational flow measurable?
    if rot.get("rotation_detected") and sa.get("axis_identifiable"):
        fitR = model_fits.get("revolute", {})
        d_axis = _axis_angle(sa.get("axis"), fitR.get("axis"))
        ev["rotation_field"] = dict(rot, fitted_vs_field_axis_deg=d_axis)
        hp_field = attachment_evidence.get("hinge_field", {})
        hp_fit = attachment_evidence.get("hinge_fit", {})
        if hinge.get("status") != "OK":
            reasons.append(f"rotation of {rot.get('net_rotation_deg'):.1f} deg is measured about a stable axis but the "
                           f"hinge line is not identifiable ({hinge.get('status')})")
            codes.append("HINGE_UNOBSERVABLE")
            cf = dict(prismatic="a shared rotation is measured, so a pure slide is rejected",
                      revolute="the hinge line itself is not identified")
            return decide("TYPE_UNOBSERVABLE", "revolute", None)
        if hp_field.get("verdict") == "CONTRADICTED" or hp_fit.get("verdict") == "CONTRADICTED":
            reasons.append("substantial rotation, but the hinge line is far from the observed static/moving interface "
                           f"(field d10={hp_field.get('d10_to_interface_m')}, fit d10={hp_fit.get('d10_to_interface_m')})")
            codes.append("HINGE_ASSEMBLY_CONTRADICTED")
            cf = dict(revolute="hinge inconsistent with the observed assembly", prismatic="rotation is measured")
            return decide("MODEL_INCONSISTENT", "none", None)
        if d_axis is not None and d_axis > c["max_axis_disagreement_deg"]:
            reasons.append(f"fitted revolute axis differs by {d_axis:.1f} deg from the hinge-free rotation axis")
            codes.append("FIT_FIELD_AXIS_DISAGREE")
            cf = dict(revolute="the reconstruction candidate is not the measured rotation", prismatic="rotation measured")
            return decide("MODEL_INCONSISTENT", "none", None)
        if ho.get("preference") == "prismatic":
            reasons.append("rotation is measured but the revolute fit predicts held-out blocks worse than the slide")
            codes.append("PREDICTIVE_CONTRADICTION")
            cf = dict(revolute="worse held-out prediction", prismatic="cannot explain the measured rotation")
            return decide("MODEL_INCONSISTENT", "none", None)
        if not cam_axes_ok:
            codes.append("CAMERA_AXIS_DISAGREEMENT")
            reasons.append("per-camera rotation axes disagree with the pooled axis")
        reasons.append(f"shared rotation {rot.get('net_rotation_deg'):.1f} deg (95% block CI "
                       f"{np.round(rot.get('net_rotation_ci95_deg'), 1).tolist()}) exceeds the calibrated no-rotation "
                       f"null q95 {null.get('net_abs_rotation_q95_deg'):.2f} deg; axis cone95 "
                       f"{sa.get('axis_cone95_deg'):.1f} deg; hinge line sd {hinge.get('line_sd_m'):.3f} m")
        if hp_field.get("status") == "OK":
            reasons.append(f"hinge vs observed interface: {hp_field.get('verdict')} (d10 "
                           f"{hp_field.get('d10_to_interface_m'):.3f} m)")
        else:
            codes.append("ASSEMBLY_ABSTAIN")
            reasons.append("assembly interface not independently observable: hinge plausibility ABSTAIN")
        if ho.get("preference") == "revolute":
            reasons.append("held-out time blocks prefer the revolute fit")
        elif ho.get("status") != "OK":
            reasons.append(f"held-out block preference: {ho.get('status')}")
        cf = dict(prismatic=f"rejected: a slide has zero rotation, the measured net rotation CI "
                            f"{np.round(rot.get('net_rotation_ci95_deg'), 1).tolist()} deg excludes 0 and the null")
        ready = (not [x for x in codes if x in ("CAMERA_AXIS_DISAGREEMENT",)]) and hinge.get("status") == "OK"
        return decide("IDENTIFIED_REVOLUTE", "revolute", "revolute", ready=ready)
    # ---------------------------------------------------------------- 4/5. translation without measurable rotation
    imp = attachment_evidence.get("implied", {})
    guide = attachment_evidence.get("guide", {})
    ub = rot.get("rotation_upper_bound_deg")
    mdr = null.get("net_abs_rotation_q95_deg")
    implied = imp.get("implied_min_rotation_deg")
    excluded = (implied is not None and ub is not None and mdr is not None
                and implied > c["prismatic_margin"] * max(ub, mdr))
    ev["translation_field"].update(rotation_upper_bound_deg=ub, minimum_detectable_rotation_deg=mdr,
                                   implied_min_hinge_rotation_deg=implied, plausible_hinges_excluded=bool(excluded),
                                   support_noise_ratio=noise_ratio, per_camera_axes_consistent=cam_axes_ok)
    positive_guide = guide.get("verdict") == "GUIDE_ALIGNED"
    if rot.get("rotation_detected") and not sa.get("axis_identifiable"):
        codes.append("ROTATION_WITHOUT_STABLE_AXIS")
        reasons.append("net rotation exceeds the null but its axis is not stable across intervals")
    if excluded and positive_guide and reliable and cam_axes_ok and not codes:
        reasons.append(f"rotation upper bound {ub:.2f} deg and detectability {mdr:.2f} deg are far below the "
                       f"{implied:.1f} deg any hinge on the observed interface would need; observed guide aligned")
        cf = dict(revolute="rejected: every hinge on the observed interface implies a rotation that would have been "
                           "detected")
        return decide("IDENTIFIED_PRISMATIC", "prismatic", "prismatic", ready=True)
    why = []
    if not excluded:
        why.append(f"a hinge on the object would rotate only {implied} deg vs upper bound {ub} / detectability "
                   f"{mdr} deg -- a short arc and a slide are not separable" if implied is not None else
                   "no observable interface to bound plausible hinges")
    if not positive_guide:
        why.append(f"no positive slider/guide evidence ({guide.get('verdict', guide.get('status'))})")
    if not reliable:
        why.append(f"moving-support tracking noise is {noise_ratio if noise_ratio is None else round(noise_ratio, 2)}x "
                   f"the static core (> {c['max_support_noise_ratio']}): rotation may be present but not observed")
        codes.append("UNRELIABLE_MOVING_SUPPORT")
    if not cam_axes_ok:
        why.append("cameras disagree on the residual rotation axis")
        codes.append("CAMERA_AXIS_DISAGREEMENT")
    reasons.append("type not identified: " + "; ".join(why))
    reasons.append(f"working model prismatic (lower complexity) is an APPROXIMATION of the observed "
                   f"{trans:.3f} m motion, not a measured slider" if trans is not None else "working model prismatic")
    sep_status = sep.get("status")
    if sep_status:
        reasons.append(f"fitted R/P prediction separation: {sep_status} "
                       f"(median {sep.get('median_block_separation_over_sigma', float('nan')):.2f} sigma)")
    cf = dict(revolute="not identified: no measurable shared rotation about a stable axis",
              prismatic="not identified: " + "; ".join(why))
    codes.append("TYPE_UNOBSERVABLE")
    # render / robot may not confer a type here (spec 7.6-7.7); they are recorded only
    return decide("TYPE_UNOBSERVABLE", "prismatic", None)


def render_veto_allowed(render: dict, min_blocks=6, min_seeds=2) -> dict:
    """Render evidence can veto only when replicated across seeds, with >= min_blocks independent held-out blocks
    and a difference larger than the between-seed variability (spec 8.2/8.3)."""
    if not render or render.get("status") != "OK":
        return dict(allowed=False, reason="no render evidence")
    nb, ns = render.get("n_independent_blocks", 0), render.get("n_seeds", 1)
    diff, seed_sd = render.get("difference"), render.get("between_seed_sd")
    if nb < min_blocks:
        return dict(allowed=False, reason=f"NOT_IDENTIFIABLE_FROM_AVAILABLE_BLOCKS ({nb} < {min_blocks})")
    if ns < min_seeds or seed_sd is None:
        return dict(allowed=False, reason=f"{ns} training seed(s): between-seed variability unknown")
    if diff is None or abs(diff) <= 2 * seed_sd:
        return dict(allowed=False, reason="difference within training-seed variability")
    return dict(allowed=True, favours="revolute" if diff < 0 else "prismatic")
