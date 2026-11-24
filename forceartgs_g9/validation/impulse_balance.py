"""Signed impulse-momentum audit of a declared force pulse (ForceTwin-adaptation spec section 3), BEFORE any fit:

    J_tau = int_pulse tau dt = I_eff dqdot + mu int sgn(qdot) dt + b int qdot dt + int tau_other dt

on the native force rows (exact ZOH) and the native RGB-D q samples (one-sided velocity reconstructions that never
smooth across the pulse). Every admissible perturbation is evaluated; the decision distinguishes plain
non-identifiability from a contradiction that NO positive inertia can satisfy. Nothing here changes a sign, an axis,
the q gauge or the metadata -- sign / axis reversals are labelled counterfactuals.
"""
from __future__ import annotations

import math

import numpy as np

STATES = ("IMPULSE_CONSISTENT", "IMPULSE_SIGN_OR_TIMING_INCONCLUSIVE", "CONTACT_EFFORT_UNCERTAIN",
          "MOTION_DERIVATIVE_UNCERTAIN", "UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE", "NO_POSITIVE_INERTIA_EXPLANATION")


def _zoh_J(t, Q, Qs, ta, tb):
    dt = np.diff(np.r_[t, t[-1] + 0.02])
    seg_lo, seg_hi = t, t + dt
    ov = np.clip(np.minimum(seg_hi, tb) - np.maximum(seg_lo, ta), 0, None)
    return float(np.sum(Q * ov)), float(math.sqrt(np.sum((Qs * ov) ** 2)))


def _side_velocity(tq, q, qs, t0, side, W, gap, method):
    """Velocity at the pulse boundary t0 from samples on ONE side only."""
    need = 4 if method == "poly" else 5
    for f in (1.0, 1.5, 2.0):
        if side == "pre":
            m = (tq <= t0 - gap) & (tq >= t0 - gap - f * W)
        else:
            m = (tq >= t0 + gap) & (tq <= t0 + gap + f * W)
        if m.sum() >= need:
            break
    if m.sum() < need:
        return None
    x, y, s = tq[m] - t0, q[m], qs[m]
    if method == "poly":
        deg = 1 if side == "pre" else 2
        if m.sum() < deg + 2:
            deg = 1
        A = np.vander(x, deg + 1, increasing=True)
        w = 1 / s ** 2
        C = np.linalg.inv((A.T * w) @ A)
        beta = C @ (A.T * w) @ y
        chi2 = float(np.sum(w * (y - A @ beta) ** 2) / max(m.sum() - deg - 1, 1))
        return dict(v=float(beta[1]), sd=float(math.sqrt(C[1, 1]) * max(1.0, math.sqrt(chi2))), n=int(m.sum()),
                    chi2_dof=chi2, method=f"one-sided weighted polynomial deg {deg}")
    from scipy.interpolate import make_smoothing_spline
    try:
        sp = make_smoothing_spline(x, y, w=1 / s ** 2 / np.mean(1 / s ** 2))
    except Exception:
        return None
    xe = float(x.max() if side == "pre" else x.min())          # the data edge nearest the pulse (no extrapolation)
    v = float(sp.derivative(1)(xe))
    rng = np.random.default_rng(0)
    vs = []
    for _ in range(40):
        spk = make_smoothing_spline(x, y + rng.standard_normal(len(y)) * s, w=1 / s ** 2 / np.mean(1 / s ** 2),
                                    lam=getattr(sp, "lam", None))
        vs.append(float(spk.derivative(1)(xe)))
    return dict(v=v, sd=float(np.std(vs, ddof=1)), n=int(m.sum()), method="one-sided smoothing spline (GCV), "
                f"evaluated at the window edge {xe:+.3f} s from the pulse boundary")


def momentum_case(t, Q, Qs, tq, q, qs, ta, tb, *, W=0.3, gap=0.04, method="poly", bounds=None):
    J, sJ = _zoh_J(t, Q, Qs, ta, tb)
    pre = _side_velocity(tq, q, qs, ta, "pre", W, gap, method)
    post = _side_velocity(tq, q, qs, tb, "post", W, gap, method)
    out = dict(J=J, sigma_J=sJ, J_over_sigma=J / sJ if sJ > 0 else math.inf, pre=pre, post=post)
    if pre is None or post is None:
        out["status"] = "VELOCITY_UNAVAILABLE"
        return out
    dv, sdv = post["v"] - pre["v"], math.hypot(pre["sd"], post["sd"])
    T = tb - ta
    S = T * (np.sign(pre["v"]) + np.sign(post["v"])) / 2                # int sgn(qdot) dt (linear qdot across)
    D = T * (pre["v"] + post["v"]) / 2                                  # int qdot dt
    out.update(dv=dv, sigma_dv=sdv, dv_over_sigma=abs(dv) / sdv if sdv > 0 else math.inf, T=T,
               int_sgn_qdot=S, int_qdot=D)
    b_ = bounds or {}
    Imax, mumax, bmax = b_.get("I_max", 5.0), b_.get("mu_max", 1.0), b_.get("b_max", 5.0)
    # achievable RHS over the admissible box  I in (0, Imax], mu in [0, mumax], b in [0, bmax], dv within +/- 2 sigma
    best = None
    for dvx in (dv - 2 * sdv, dv, dv + 2 * sdv):
        for I in np.r_[1e-6, np.geomspace(1e-4, Imax, 60)]:
            for mu in (0.0, mumax):
                for bb in (0.0, bmax):
                    rhs = I * dvx + mu * S + bb * D
                    z = (J - rhs) / sJ           # dqdot uncertainty enters through the +/- 2 sigma scan only
                    if best is None or abs(z) < abs(best["z"]):
                        best = dict(I=float(I), mu=mu, b=bb, dv_used=float(dvx), rhs=float(rhs), z=float(z))
    out["best_positive_inertia"] = best
    out["positive_inertia_feasible"] = bool(abs(best["z"]) <= 2.0)
    out["terms_sign"] = dict(J=float(np.sign(J)), I_dv=float(np.sign(dv)), mu_sgn=float(np.sign(S)),
                             b_qdot=float(np.sign(D)),
                             note="with I, mu, b >= 0 the RHS terms carry the signs of dqdot, sgn(qdot) and qdot")
    if abs(dv) > 0:
        out["J_over_dv"] = J / dv
    out["tau_other_needed_for_I"] = {f"I={I:g}": float((J - I * dv) / T) for I in (0.05, 0.2, 0.5, 1.0, 2.0)}
    return out


def onset_bracket(rec, ta, tb, J, sJ, *, n_sigma=3.0, plateau_s=0.4, I_prior=None) -> dict:
    """Did the pulse move the part?  (a) the BRACKET: the last q sample before and the first after the pulse (no fit
    across it); (b) the motion ONSET: first sample leaving the pre-pulse level by > n_sigma; (c) the effort measured
    between the pulse and the end of the acceleration vs the impulse a plausible inertia would need."""
    tq, q, qs = rec.q_t_s, rec.q, rec.q_sigma
    pre = np.flatnonzero(tq < ta)
    post = np.flatnonzero(tq >= tb)
    if len(pre) < 3 or len(post) < 4:
        return dict(status="NOT_RUN", reason="too few q samples around the pulse")
    i0, i1 = pre[-1], post[0]
    lvl = float(np.mean(q[pre[-3:]]))
    lvl_sd = float(max(np.std(q[pre[-3:]], ddof=1), np.median(qs[pre[-3:]])))
    dq_br = float(q[i1] - q[i0])
    sd_br = float(math.hypot(qs[i0], qs[i1]))
    dt_br = float(tq[i1] - tq[i0])
    on = next((int(k) for k in post if abs(q[k] - lvl) > n_sigma * math.hypot(lvl_sd, qs[k])), None)
    out = dict(bracket=dict(t_last_pre=float(tq[i0]), t_first_post=float(tq[i1]), dq=dq_br, sd=sd_br, dt=dt_br,
                            mean_velocity=dq_br / dt_br, mean_velocity_sd=sd_br / dt_br,
                            displacement_resolved=bool(abs(dq_br) > n_sigma * sd_br)),
               pre_level=dict(q=lvl, sd=lvl_sd))
    if on is None:
        out["onset"] = None
        out["verdict"] = "NO_MOTION_AFTER_PULSE"
        return out
    t_on = float(tq[on])
    k = np.flatnonzero((tq >= t_on) & (tq <= t_on + plateau_s))
    if len(k) >= 3:
        A = np.c_[np.ones(len(k)), tq[k] - t_on]
        coef = np.linalg.lstsq(A, q[k], rcond=None)[0]
        v_pl = float(coef[1])
    else:
        v_pl = float((q[k[-1]] - q[on - 1]) / (tq[k[-1]] - tq[on - 1])) if len(k) else math.nan
    t_end = t_on + 0.5 * plateau_s
    Jpost, sJpost = _zoh_J(rec.force_t_s, rec.Q_object, rec.Q_sigma, tb, t_end)
    out["onset"] = dict(t_onset=t_on, delay_after_pulse_s=t_on - tb, plateau_velocity=v_pl,
                        J_measured_pulse_end_to_acceleration_end=Jpost, sigma=sJpost, window=[tb, t_end])
    Ilo, Ihi = I_prior or (0.05, 2.0)
    need = (Ilo * v_pl, Ihi * v_pl)
    out["required_impulse_for_prior_inertia"] = dict(I_range=[Ilo, Ihi], J_needed=list(need),
                                                     J_measured=Jpost + J,
                                                     fraction_explained_at_I_min=float((Jpost + J) / need[0])
                                                     if need[0] else None)
    # conditional LOWER bound on I if the part was free during the pulse and its bracket velocity is unresolved
    if not out["bracket"]["displacement_resolved"] and J != 0:
        vmax = n_sigma * sd_br / dt_br
        out["conditional_inertia_lower_bound"] = dict(
            I_min=float(abs(J) / (2 * vmax)), label="CONDITIONAL: only if the part was FREE (unlatched, "
                                                    "unconstrained) during the pulse; a latch absorbs any impulse",
            basis="|J| over the pulse produced no velocity change resolved by the bracket samples")
    explained = (Jpost + J) / need[0] if need[0] else 0.0
    if on == i1 and out["bracket"]["displacement_resolved"]:
        out["verdict"] = "ONSET_WITHIN_BRACKET"          # moved between the last pre and first post sample
        out["verdict_note"] = (f"motion started somewhere in [{tq[i0]:.3f}, {tq[i1]:.3f}] s: the q sampling "
                               f"({1 / np.median(np.diff(tq)):.1f} Hz) cannot separate the pulse from the onset")
    elif t_on > tb + 1e-9 and not out["bracket"]["displacement_resolved"]:
        out["verdict"] = "PULSE_PRECEDES_MOTION_ONSET"
    else:
        out["verdict"] = "MOTION_DURING_PULSE"
    out["measured_effort_explains_motion"] = bool(np.sign(Jpost + J) == np.sign(v_pl) and explained >= 0.5)
    return out


def audit(rec, pulse_rows, *, context_rows=None, kin_hyp=None, observability=None, tool=None, robot_record=None,
          bounds=None, I_prior=None) -> dict:
    """Full sensitivity study of one declared pulse on a JointInteractionRecord (accepted hypothesis)."""
    a0, o0 = rec.axis_B, rec.origin_B
    jt = rec.joint_type
    t, src = rec.force_t_s, rec.source_force_row
    lo, hi = pulse_rows
    ta, tb = lo / 50.0, (hi + 1) / 50.0
    tq, q, qs = rec.q_t_s, rec.q, rec.q_sigma
    sig_cases = {}

    def run(name, Q=None, tq_=None, q_=None, label="ADMISSIBLE_SCENARIO", **kw):
        tq2 = tq if tq_ is None else tq_
        qs2 = qs if len(tq2) == len(qs) else np.interp(tq2, tq, qs)
        r = momentum_case(t, rec.Q_object if Q is None else Q, rec.Q_sigma, tq2,
                          q if q_ is None else q_, qs2, ta, tb, bounds=bounds, **kw)
        r["label"] = label
        sig_cases[name] = r
        return r

    base = run("accepted", label="ACCEPTED (declared sign, accepted joint, TCP contact point)")
    run("accepted_spline_velocity", method="spline")
    for W in (0.2, 0.45):
        for gap in (0.0, 0.04, 0.08):
            if (W, gap) != (0.3, 0.04):
                run(f"velocity_window_W{W}_gap{gap}", W=W, gap=gap)
    for lag in (0.01, 0.02, 0.04):
        for sg in (1, -1):
            run(f"clock_offset_{sg * lag * 1e3:+.0f}ms", tq_=tq + sg * lag)
    # contact point / lever (revolute): +/- 1.5 cm along tangent, radial and axial
    if jt == "revolute":
        P = rec.contact_point_B
        rad = P - o0
        rad = rad - np.outer(rad @ a0, a0)
        rad /= np.linalg.norm(rad, axis=1, keepdims=True)
        tan = np.cross(a0, rad)
        for nm, d in (("tangent", tan), ("radial", rad), ("axial", np.tile(a0, (len(P), 1)))):
            for sg in (1, -1):
                run(f"contact_point_{nm}_{sg * 1.5:+.1f}cm", Q=rec.effort_for(a0, o0, P + sg * 0.015 * d))
    obs = observability or {}
    if obs.get("axis_sigma_deg"):
        u = np.cross(a0, [1.0, 0, 0] if abs(a0[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(a0, u)
        th = math.tan(math.radians(obs["axis_sigma_deg"]))
        for nm, d in (("u1", u), ("u2", v)):
            for sg in (1, -1):
                a2 = a0 + sg * th * d
                run(f"axis_{nm}_{sg:+d}sigma", Q=rec.effort_for(a2 / np.linalg.norm(a2), o0))
    if obs.get("hinge_line_sigma_m") and jt == "revolute":
        u = np.cross(a0, [1.0, 0, 0] if abs(a0[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        for sg in (1, -1):
            run(f"hinge_origin_{sg:+d}sigma", Q=rec.effort_for(a0, o0 + sg * obs["hinge_line_sigma_m"] * u))
    # refit kinematic hypotheses: their OWN q(t) and their own projected effort
    if kin_hyp and kin_hyp.get("hypotheses"):
        fm_src = np.asarray(kin_hyp["tracks"]["fm_source"], int)
        for name, h in kin_hyp["hypotheses"].items():
            qv = np.asarray([np.nan if x is None else x for x in h["q_rgbd_frames"]], float)
            ok = np.isfinite(qv)
            # the refit q has the reference frame at 0 like the authority; map to the q clock
            run(f"kinematic_hypothesis_{name}", Q=rec.effort_for(h["axis_B"], h["origin_B"]),
                tq_=fm_src[ok] / 50.0, q_=qv[ok], label=f"KINEMATIC_HYPOTHESIS {name} (re-estimated from RGB-D "
                                                         "tracks; own q and own projected effort)")
    # counterfactuals (diagnostic only)
    run("COUNTERFACTUAL_force_sign_flipped", Q=-rec.Q_object,
        label="COUNTERFACTUAL ONLY (the declared metadata sign is never changed)")
    rev = momentum_case(t, -rec.Q_object, rec.Q_sigma, tq, -q, qs, ta, tb, bounds=bounds)
    rev["label"] = "EQUIVALENT CONVENTION (a, q, tau) -> (-a, -q, -tau): must give the same conclusion"
    sig_cases["convention_reversed_a_q_tau"] = rev
    # tool dynamic wrench (per kg of tool; tool mass not declared)
    tool_rep = dict(status="NOT_ESTABLISHED", tool_mass_kg=None)
    if robot_record is not None:
        rr = robot_record
        rsrc = np.asarray(rr["source_frame_index"], int)
        idx = np.flatnonzero((rsrc >= lo - 2) & (rsrc <= hi + 2))
        if len(idx) >= 5:
            tt = rsrc[idx] / 50.0
            Pp = np.asarray(rr["p_tcp_base_m"], float)[idx]
            acc = np.gradient(np.gradient(Pp, tt, axis=0), tt, axis=0)
            Qa = acc @ a0 if jt == "prismatic" else np.einsum("j,nj->n", a0, np.cross(Pp - o0, acc))
            inside = (rsrc[idx] >= lo) & (rsrc[idx] <= hi)
            Ja = float(np.sum(Qa[inside] * 0.02))
            tool_rep = dict(status="BOUND_PER_KG", J_per_kg_tool=Ja, peak_tcp_acc_mps2=float(
                np.max(np.linalg.norm(acc[inside], axis=1))), tool_mass_to_cancel_J_kg=(base["J"] / Ja) if Ja else None,
                tool_mass_kg=(tool or {}).get("mass_kg"),
                note="computed object force contains +m_tool a_TCP (sensor between flange and tool); a plausible "
                     "positive mass would make tool inertia a candidate explanation")
    # context: work of the effort over the whole contact window
    work = None
    if rec.windows:
        name, fs, qsl, srcw = rec.windows[0]
        for w in rec.windows:
            if w[3][0] <= lo <= w[3][1]:
                name, fs, qsl, srcw = w
        tw, Qw = t[fs], rec.Q_object[fs]
        qd = np.gradient(np.interp(tw, tq[qsl], q[qsl]), tw)
        inc = rec.in_contact[fs]
        P_ = Qw * qd
        work = dict(window=list(srcw), work_total=float(np.sum(P_ * 0.02)),
                    work_contact_rows=float(np.sum(P_[inc] * 0.02)),
                    positive_power_fraction_contact=float(np.mean(P_[inc] > 0)) if inc.any() else None,
                    work_in_pulse=float(np.sum(P_[(src[fs] >= lo) & (src[fs] <= hi)] * 0.02)))
    ob = onset_bracket(rec, ta, tb, base["J"], base["sigma_J"], I_prior=I_prior)
    decision = decide(sig_cases, base, tool_rep, work, ob)
    return dict(schema="forceartgs.g9.impulse_balance/v1", pulse_rows=[lo, hi], t_start=ta, t_end=tb,
                onset_bracket=ob,
                pulse_row_definition="force rows with source_frame_index in [lo, hi]; ZOH sample-and-hold to row + 1",
                joint_hypothesis=rec.joint_hypothesis_id, cases=sig_cases, tool_dynamic_wrench=tool_rep,
                contact_window_work=work, bounds=bounds, decision=decision)


def decide(cases, base, tool_rep, work, ob=None) -> dict:
    states, ev = [], {}
    if ob and ob.get("onset"):
        ev["onset_bracket"] = dict(verdict=ob["verdict"], note=ob.get("verdict_note"), bracket=ob["bracket"],
                                   onset=ob.get("onset"), required=ob.get("required_impulse_for_prior_inertia"),
                                   lower_bound=ob.get("conditional_inertia_lower_bound"))
        if ob["verdict"] == "PULSE_PRECEDES_MOTION_ONSET" or (ob["verdict"] == "ONSET_WITHIN_BRACKET" and
                                                              not ob.get("measured_effort_explains_motion")):
            states.append("MOTION_DERIVATIVE_UNCERTAIN")
            ev["one_sided_fit_artifact"] = ("one-sided post-pulse fits extrapolate the subsequent motion back to the "
                                            "pulse boundary; the pulse-only velocity change is not resolved by the q "
                                            "sampling")
        if not ob.get("measured_effort_explains_motion"):
            states.append("UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE")
            ev["motion_without_measured_effort"] = (
                "the part reaches its plateau velocity while the measured net impulse is a small fraction (or of the "
                "opposite sign) of what any plausible inertia needs: an unmeasured torque (latch / seal release, "
                "self-opening tilt, contact outside the sensed tool) drives the motion")
    adm = {k: v for k, v in cases.items() if v.get("label", "").startswith(("ADMISSIBLE", "ACCEPTED", "KINEMATIC"))
           or k == "accepted_spline_velocity"}
    feas = {k: v.get("positive_inertia_feasible") for k, v in adm.items() if "positive_inertia_feasible" in v}
    ev["positive_inertia_feasible_by_case"] = feas
    if base.get("status") == "VELOCITY_UNAVAILABLE":
        states.append("MOTION_DERIVATIVE_UNCERTAIN")
        return dict(primary="MOTION_DERIVATIVE_UNCERTAIN", states=states, evidence=ev)
    if base.get("positive_inertia_feasible"):
        states.append("IMPULSE_CONSISTENT")
    dv_res = base["dv_over_sigma"]
    sgn_dv = {k: np.sign(v["dv"]) for k, v in adm.items() if "dv" in v}
    if dv_res < 2 or len(set(sgn_dv.values())) > 1:
        states.append("MOTION_DERIVATIVE_UNCERTAIN")
        ev["dv_over_sigma"] = dv_res
        ev["dv_sign_changes_across_cases"] = len(set(sgn_dv.values())) > 1
    Js = {k: v["J"] for k, v in adm.items()}
    if any(np.sign(x) != np.sign(base["J"]) for x in Js.values()) or \
            max(abs(x - base["J"]) for x in Js.values()) > 0.5 * abs(base["J"]):
        states.append("CONTACT_EFFORT_UNCERTAIN")
        ev["J_range_admissible"] = [min(Js.values()), max(Js.values())]
    timing = [k for k in cases if k.startswith("clock_offset") and cases[k].get("positive_inertia_feasible")]
    if timing and not base.get("positive_inertia_feasible"):
        states.append("IMPULSE_SIGN_OR_TIMING_INCONCLUSIVE")
        ev["timing_cases_feasible"] = timing
    if feas and not any(feas.values()):
        states.append("NO_POSITIVE_INERTIA_EXPLANATION")
        ev["note"] = ("no admissible case (velocity windows / smoother / clock offsets / contact point / axis / hinge "
                      "/ refit kinematics) admits a positive inertia within 2 sigma with mu, b >= 0")
        cf = cases.get("COUNTERFACTUAL_force_sign_flipped", {})
        ev["counterfactual_flipped_sign_feasible"] = cf.get("positive_inertia_feasible")
        tm = tool_rep.get("tool_mass_to_cancel_J_kg")
        ev["tool_inertia_plausible"] = bool(tm is not None and 0.05 <= tm <= 3.0)
        ev["tau_other_needed"] = base.get("tau_other_needed_for_I")
        states.append("UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE")
    rv = cases.get("convention_reversed_a_q_tau", {})
    ev["convention_invariance"] = dict(
        same_feasibility=rv.get("positive_inertia_feasible") == base.get("positive_inertia_feasible"),
        J_over_dv_equal=bool(abs(rv.get("J_over_dv", 0) - base.get("J_over_dv", 0)) < 1e-9 * max(1, abs(
            base.get("J_over_dv", 1)))))
    if work:
        ev["contact_window_work"] = work
    order = ["UNMODELED_MECHANISM_OR_EXTERNAL_TORQUE", "NO_POSITIVE_INERTIA_EXPLANATION",
             "IMPULSE_SIGN_OR_TIMING_INCONCLUSIVE", "CONTACT_EFFORT_UNCERTAIN", "MOTION_DERIVATIVE_UNCERTAIN",
             "IMPULSE_CONSISTENT"]
    states = list(dict.fromkeys(states))
    primary = next((s for s in order if s in states), "IMPULSE_CONSISTENT")
    return dict(primary=primary, states=states, evidence=ev,
                absolute_inertia=("NOT_IDENTIFIED_ABSOLUTE_INERTIA" if primary != "IMPULSE_CONSISTENT" else
                                  "PULSE_CONSISTENT_WITH_POSITIVE_INERTIA (not by itself an identification)"))


def plot(rec, aud, robot_record, path, context_rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..forcetwin.derivatives import local_poly, spline
    lo, hi = aud["pulse_rows"]
    c0, c1 = context_rows or (lo - 30, hi + 40)
    src = rec.source_force_row
    m = (src >= c0) & (src <= c1)
    t = rec.force_t_s[m]
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    F = rec.force_object_B[m]
    for j, c in enumerate("xyz"):
        ax[0].step(t, F[:, j], where="post", lw=1, label=f"F_obj,B {c} (robot on object)")
    if robot_record is not None:
        rs = np.asarray(robot_record["source_frame_index"])
        mm = (rs >= c0) & (rs <= c1)
        Fs = np.asarray(robot_record["F_sensor_corr_N"])[mm]
        Fr = np.asarray(robot_record["F_sensor_raw_N"])[mm]
        ax[0].plot(rs[mm] / 50.0, np.linalg.norm(Fs, axis=1), "k--", lw=0.8, label="|F_sensor_corr| (sensor axes)")
        if np.isfinite(Fr).all():
            ax[0].plot(rs[mm] / 50.0, np.linalg.norm(Fr - Fs, axis=1), "c:", lw=0.8, label="|tare baseline|")
        sf = np.asarray(robot_record["force_sigma_N"], float)
        sig = float(np.sqrt(np.mean(sf ** 2))) if sf.ndim == 1 and len(sf) == 3 else sf[mm]
        ax[0].fill_between(rs[mm] / 50.0, -2 * sig, 2 * sig, color="gray", alpha=0.2, label="+/- 2 sigma_F (RMS)")
    ax[0].set_ylabel("force [N]")
    ax[0].legend(fontsize=7, ncol=2)
    a, o = rec.axis_B, rec.origin_B
    P = rec.contact_point_B[m]
    if rec.joint_type == "revolute":
        tan = np.cross(a, P - o)
        lever = np.linalg.norm(tan, axis=1)
        Ft = np.einsum("ij,ij->i", F, tan / lever[:, None])
    else:
        lever, Ft = np.ones(m.sum()), F @ a
    tau = rec.Q_object[m]
    ax[1].step(t, tau, where="post", label="tau (signed generalized effort)")
    ax[1].plot(t, np.linalg.norm(F, axis=1), lw=0.8, label="||F||")
    ax[1].plot(t, Ft, lw=0.8, label="F tangential")
    qd = np.gradient(np.interp(t, rec.q_t_s, rec.q), t)
    ax[1].plot(t, tau * qd, lw=0.8, label="power tau qdot")
    ax1b = ax[1].twinx()
    ax1b.plot(t, lever, "k:", lw=0.8, label="lever |a x (p - o)| [m]")
    ax1b.set_ylabel("lever [m]")
    ax[1].legend(fontsize=7, loc="upper left")
    dt = np.diff(np.r_[t, t[-1] + 0.02])
    J = np.cumsum(tau * dt)
    sJ = np.sqrt(np.cumsum((rec.Q_sigma[m] * dt) ** 2))
    ax[2].plot(t + dt, J, label="cumulative impulse J_tau(t)")
    ax[2].fill_between(t + dt, J - 2 * sJ, J + 2 * sJ, alpha=0.3)
    ax[2].set_ylabel("J [N m s]")
    ax[2].legend(fontsize=7)
    qm = (rec.q_t_s >= c0 / 50.0) & (rec.q_t_s <= c1 / 50.0)
    ax[3].errorbar(rec.q_t_s[qm], rec.q[qm], yerr=rec.q_sigma[qm], fmt=".", label="q (RGB-D)")
    ax3b = ax[3].twinx()
    tg = np.linspace(rec.q_t_s[qm].min(), rec.q_t_s[qm].max(), 300)
    for fn, sty in ((local_poly, "g-"), (spline, "m-")):
        d = fn(rec.q_t_s[qm], rec.q[qm], rec.q_sigma[qm], tg)
        ax3b.plot(tg, d["qd"], sty, lw=1, label=f"qdot {d['method']['name']}")
        ax3b.fill_between(tg, d["qd"] - 2 * d["qd_sd"], d["qd"] + 2 * d["qd_sd"], color=sty[0], alpha=0.15)
    ax3b.set_ylabel("qdot")
    ax3b.legend(fontsize=7, loc="lower left")
    for x in ax:
        x.axvspan(lo / 50.0, (hi + 1) / 50.0, color="r", alpha=0.15)
    ax[3].set_xlabel("t [s] = source_frame_index / 50")
    ax[0].set_title(f"pulse rows {lo}-{hi}: {aud['decision']['primary']}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
