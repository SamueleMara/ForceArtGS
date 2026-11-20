"""Short push impulses and following low-force coasts, analysed SEPARATELY (G9 corrections section 2).

Native force rows (50 Hz, recorded timestamps) are never downsampled to the q clock before integration.

pulse  : contiguous rows inside a contact-confirmed segment whose measured |F| >= k sigma_F (k = 6 primary, the
         exporter's contact threshold; k = 3 / 10 reported as threshold sensitivity). Generalized impulse
         J_Q = sum_i Q_i (t_{i+1} - t_i) (forward ZOH, the recording's sample-and-hold model; trapezoid reported),
         sigma_J = sqrt(sum (sigma_Q dt)^2), sigma_Q = sigma_F x lever (R) or sigma_F (P).
coast  : rows after the pulse with |F| < 3 sigma_F up to the next pulse / window end.
velocity: q from the joint authority only (independent of the force); local polynomial fits on windows that
         end / start one camera frame away from the pulse, uncertainty from q_std and the fit covariance.
balance: M (q'_b - q'_a) = J_Q - int (c q' + Fc s + G + b) dt.
         (a) impulse-only conditional M = J_Q / (Dv + (c/M v_mean + Fc/M) dt_pulse) with the coast ratios when
             they are resolved (else Dv alone, flagged); (b) the joint pulse + coast fit is the standard G9 rollout
             fit of the same window (referenced, not duplicated); (c) sensitivities: k threshold, +/- lag, sign
             (diagnostic only, never applied), baseline bias +/- sigma_F, hinge geometry, coast load model.
verdict: IMPULSE_INFORMATIVE (J and Dv both resolved > 3 sigma, pre / post velocities valid),
         IMPULSE_UNRESOLVED (no valid pre / post velocity window, or Dv not resolved although J is),
         IMPULSE_NOT_INFORMATIVE (J within the force noise).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

REL = 0.10


def _runs(mask):
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def effort_rows(rec, joint_type, axis, origin, rows=slice(None)):
    from estimotion_2part.joint_discovery.force_convention import ObjectBaseForce, generalized_effort
    F = ObjectBaseForce(rec["F_object_base_N"][rows], rec.get("force_conversion", {}))
    return generalized_effort(joint_type, axis, origin, F, rec["p_tcp_base_m"][rows])


def _poly_velocity(t, q, s, t_eval, deg):
    """Weighted polynomial fit of q(t); returns (v(t_eval), sd) from the fit covariance (inflated by chi2/dof)."""
    t0 = t_eval
    X = np.vander(t - t0, deg + 1, increasing=True)
    W = 1.0 / np.asarray(s, float)
    A = X * W[:, None]
    b = q * W
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ coef
    dof = max(len(t) - deg - 1, 1)
    chi = max(float(r @ r) / dof, 1.0)
    cov = chi * np.linalg.pinv(A.T @ A)
    return float(coef[1]), float(math.sqrt(max(cov[1, 1], 0.0))), dict(chi2_dof=float(r @ r) / dof, n=len(t),
                                                                       degree=deg)


def coast_model(t, q0, v0, k, a):
    """M q'' + c q' + Fc s = 0, k = c/M, a = Fc/M, s = sign(v0); stops when v reaches 0 (dry friction)."""
    s = np.sign(v0) if v0 != 0 else 0.0
    t = np.asarray(t, float)
    if k <= 1e-9:
        ts = abs(v0) / a if a > 0 else np.inf
        tt = np.minimum(t, ts)
        return q0 + v0 * tt - 0.5 * s * a * tt * tt
    vinf = -s * a / k
    ts = (1 / k) * math.log((v0 - vinf) / (-vinf)) if (a > 0 and s != 0) else np.inf
    tt = np.minimum(t, ts)
    return q0 + vinf * tt + (v0 - vinf) * (1 - np.exp(-k * tt)) / k


def fit_coast(t, q, s):
    if len(t) < 6:
        return dict(status="INSUFFICIENT_SAMPLES", n=len(t))
    tt = t - t[0]
    v_guess = float(np.polyfit(tt[:4], q[:4], 1)[0])

    def res(p):
        return (coast_model(tt, p[0], p[1], p[2], p[3]) - q) / s
    best = None
    for k0 in (0.5, 3.0, 15.0):
        r = least_squares(res, [q[0], v_guess, k0, 0.0], bounds=([-np.inf, -np.inf, 0.0, 0.0],
                                                                  [np.inf, np.inf, 200.0, 1e3]), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    J = best.jac
    n, p = len(tt), 4
    chi = float(2 * best.cost) / max(n - p, 1)
    try:
        cov = max(chi, 1.0) * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = max(chi, 1.0) * np.linalg.pinv(J.T @ J)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    q0, v0, k, a = best.x
    out = dict(status="OK", n=n, chi2_dof=chi, v0=float(v0), v0_sd=float(sd[1]),
               k_c_over_M=dict(estimate=float(k), sd=float(sd[2]), label="STATISTICAL"),
               a_Fc_over_M=dict(estimate=float(a), sd=float(sd[3]), label="STATISTICAL"))
    tau = 1 / k if k > 0 else math.inf
    out["decay_time_M_over_c_s"] = dict(estimate=tau, sd=float(sd[2] / k ** 2) if k > 0 else math.inf,
                                        label="STATISTICAL")
    for key in ("k_c_over_M", "a_Fc_over_M", "decay_time_M_over_c_s"):
        e = out[key]
        rel = e["sd"] / abs(e["estimate"]) if e["estimate"] not in (0, math.inf) else math.inf
        e["rel_sd"] = rel
        e["statistically_resolved"] = bool(rel <= REL)
    out["at_bounds"] = dict(k_zero=bool(k <= 1e-6), a_zero=bool(a <= 1e-9))
    return out


def analyze(rec, joint_type, axis, origin, q_src, q, q_std, window, *, observability=None, k_primary=6.0,
            lags=(0.01, 0.02), cam_gap_frames=1, pre_window_s=0.3, post_window_s=0.3) -> dict:
    """One interaction window [lo, hi] (source rows)."""
    src = np.asarray(rec["source_frame_index"], int)
    lo, hi = window
    m = (src >= lo) & (src <= hi)
    rows = np.flatnonzero(m)
    t = np.asarray(rec["timestamp_s"], float)[rows]                    # recorded timestamps (force sampling)
    tc = src[rows] / 50.0                                                # the q clock (row-exact)
    Fn = np.linalg.norm(rec["F_object_base_N"][rows], axis=1)
    Q = effort_rows(rec, joint_type, axis, origin, rows)
    sigF = float(np.sqrt(np.mean(np.asarray(rec["force_sigma_N"], float) ** 2)))
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    if joint_type == "revolute":
        lever = np.linalg.norm(np.cross(a, rec["p_tcp_base_m"][rows] - np.asarray(origin, float)), axis=1)
        sigQ = sigF * lever
    else:
        sigQ = np.full(len(rows), sigF)
    seg = np.asarray(rec["in_contact_segment"], bool)[rows]
    state = np.asarray(rec["contact_state"])[rows]
    # q is measured by the cameras independently of the force: the velocity windows may use every valid q sample,
    # also outside the force window (whose limits come from the contact / force-state evidence)
    qs, qq, qsd = np.asarray(q_src, int), np.asarray(q, float), np.asarray(q_std, float)
    tq, qv, sv = qs / 50.0, qq, qsd.copy()
    pos = sv[np.isfinite(sv) & (sv > 0)]
    floor = float(np.median(pos)) if len(pos) else 1e-3
    sv = np.where(np.isfinite(sv) & (sv > 0), sv, floor)    # q_sigma = 0 at the gauge reference frame (q = 0 by def.)
    cam_dt = float(np.median(np.diff(tq))) if len(tq) > 1 else math.inf
    dt_rows = np.diff(np.r_[tc, tc[-1] + (tc[-1] - tc[-2] if len(tc) > 1 else 0.02)])

    def pulses_for(k):
        act = seg & (state == "contact") & (Fn >= k * sigF)
        rr = _runs(act)
        merged = []
        for r in rr:                                                     # bridge single-row gaps
            if merged and r[0] - merged[-1][1] <= 2:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        return merged

    from scipy.signal import savgol_filter
    Pw = rec["p_tcp_base_m"][rows]
    v_tcp = np.stack([savgol_filter(Pw[:, k], min(7, len(Pw) - (1 - len(Pw) % 2)), 2, deriv=1,
                                    delta=float(np.median(np.diff(t)))) for k in range(3)], 1) if len(Pw) >= 7 \
        else np.gradient(Pw, t, axis=0)
    Fw = rec["F_object_base_N"][rows]

    def pulse_stats(i0, i1, Qx=Q):
        sl = slice(i0, i1 + 1)
        J = float(np.sum(Qx[sl] * dt_rows[sl]))
        Jabs = float(np.sum(np.abs(Qx[sl]) * dt_rows[sl]))
        sgn = np.sign(Qx[sl][np.abs(Qx[sl]) > 2 * np.median(sigQ[sl])])
        pw = float(np.sum(np.einsum("ij,ij->i", Fw[sl], v_tcp[sl]) * dt_rows[sl]))
        Jt = float(np.trapz(Qx[max(i0 - 1, 0):i1 + 2], tc[max(i0 - 1, 0):i1 + 2]))
        sJ = float(math.sqrt(np.sum((sigQ[sl] * dt_rows[sl]) ** 2)))
        return dict(rows=[int(src[rows[i0]]), int(src[rows[i1]])], t_start=float(tc[i0]),
                    t_end=float(tc[i1] + dt_rows[i1]), duration_s=float(tc[i1] + dt_rows[i1] - tc[i0]),
                    n_force_samples=int(i1 - i0 + 1), peak_abs_F_N=float(Fn[sl].max()),
                    peak_abs_Q=float(np.abs(Qx[sl]).max()), J_Q=J, J_Q_trapezoid=Jt, sigma_J_Q=sJ,
                    SNR_pulse_peak=float(np.abs(Qx[sl]).max() / np.median(sigQ[sl])),
                    J_over_sigma=float(abs(J) / sJ) if sJ > 0 else math.inf,
                    camera_frames_inside=int(np.sum((tq >= tc[i0]) & (tq <= tc[i1] + dt_rows[i1]))),
                    net_to_abs_impulse=float(abs(J) / Jabs) if Jabs > 0 else None,
                    significant_sign_changes=int(np.sum(sgn[1:] != sgn[:-1])) if len(sgn) > 1 else 0,
                    work_along_tcp_motion_J=pw,
                    oscillatory_transient=bool(Jabs > 0 and abs(J) / Jabs < 0.6 and
                                               (np.sum(sgn[1:] != sgn[:-1]) if len(sgn) > 1 else 0) >= 1),
                    integration="forward ZOH on native rows (recorded sample-and-hold); trapezoid for comparison")

    def velocities(t_a, t_b, shift=0.0, t_prev=-np.inf, t_next=np.inf):
        """Pre / post windows one camera frame away from the pulse, never overlapping a neighbouring pulse; the
        window grows from the declared length up to 2x until it holds 4 independent q samples."""
        gap = cam_gap_frames * cam_dt
        tqs = tq + shift
        for f in (1.0, 1.5, 2.0):
            pre = (tqs <= t_a - gap) & (tqs >= max(t_a - gap - f * pre_window_s, t_prev + gap))
            if pre.sum() >= 4:
                break
        for f2 in (1.0, 1.5, 2.0):
            post = (tqs >= t_b + gap) & (tqs <= min(t_b + gap + f2 * post_window_s, t_next - gap))
            if post.sum() >= 4:
                break
        near = 3 * cam_dt                     # never extrapolate a velocity across a q gap around the pulse
        if pre.sum() and (t_a - gap) - tqs[pre].max() > near:
            return None, dict(n_pre=int(pre.sum()), n_post=int(post.sum()),
                              reason=f"q gap before the pulse ({(t_a - gap) - tqs[pre].max():.3f} s > 3 frames)")
        if post.sum() and tqs[post].min() - (t_b + gap) > near:
            return None, dict(n_pre=int(pre.sum()), n_post=int(post.sum()),
                              reason=f"q gap after the pulse ({tqs[post].min() - (t_b + gap):.3f} s > 3 frames)")
        if pre.sum() < 4 or post.sum() < 4:
            return None, dict(n_pre=int(pre.sum()), n_post=int(post.sum()),
                              reason="fewer than 4 independent q samples before / after the pulse (windows limited "
                                     "by the neighbouring pulses and 2x the declared length)")
        va, sa, ia = _poly_velocity(tqs[pre], qv[pre], sv[pre], t_a, 1)
        vb, sb, ib = _poly_velocity(tqs[post], qv[post], sv[post], t_b, 2)
        return dict(v_pre=va, v_pre_sd=sa, v_post=vb, v_post_sd=sb, dv=vb - va, dv_sd=math.hypot(sa, sb),
                    pre_fit=ia, post_fit=ib, pre_window_s=[float(tqs[pre].min()), float(tqs[pre].max())],
                    post_window_s=[float(tqs[post].min()), float(tqs[post].max())]), None

    pulses = pulses_for(k_primary)
    out = dict(window=[int(lo), int(hi)], sigma_F_N=sigF, camera_dt_s=cam_dt, force_dt_s=float(np.median(np.diff(t))),
               clock=dict(force="recorded timestamps", q="source_frame_index / 50 (row-exact with the force rows)",
                          max_recorded_vs_row_clock_s=float(np.max(np.abs(t - tc))),
                          hardware_exposure_offset="NOT_ESTABLISHED (lag sensitivity below)"),
               aliasing=f"force {1 / max(float(np.median(np.diff(t))), 1e-9):.0f} Hz vs q "
                        f"{1 / cam_dt if math.isfinite(cam_dt) and cam_dt > 0 else 0:.0f} Hz: a pulse shorter than "
                        "one camera frame is resolved only through the pre / post velocities",
               threshold_sensitivity={}, pulses=[], coast=None)
    for k in (3.0, 6.0, 10.0):
        out["threshold_sensitivity"][f"k={k:g}"] = [pulse_stats(i0, i1) for i0, i1 in pulses_for(k)]
    coast_rep = None
    for pi, (i0, i1) in enumerate(pulses):
        ps = pulse_stats(i0, i1)
        nxt = pulses[pi + 1][0] if pi + 1 < len(pulses) else len(rows)
        low = Fn < 3 * sigF
        j = i1 + 1
        c_rows = [r for r in range(j, nxt) if low[r]]
        coast = None
        if c_rows:
            c0, c1 = c_rows[0], c_rows[-1]
            qc = (tq >= tc[c0] + cam_gap_frames * cam_dt) & (tq <= tc[c1])
            fit = fit_coast(tq[qc], qv[qc], sv[qc])
            Qc = Q[c0:c1 + 1]
            still_in_contact = bool(np.any(seg[c0:c1 + 1]))
            coast = dict(rows=[int(src[rows[c0]]), int(src[rows[c1]])], duration_s=float(tc[c1] - tc[c0]),
                         inside_contact_segment=still_in_contact,
                         interpretation=("low-force interval INSIDE the operator contact segment: the TCP may still "
                                         "drive the part kinematically (sliding contact at low force) -- not "
                                         "necessarily a free coast") if still_in_contact else "free coast",
                         rms_Q=float(np.sqrt(np.mean(Qc ** 2))),
                         SNR_coast=float(np.sqrt(np.mean(Qc ** 2)) / np.median(sigQ[c0:c1 + 1])),
                         model="M q'' + c q' + Fc sign(q') = 0 (no configuration load; see load-model sensitivity)",
                         fit=fit)
        sig_p = [pulse_stats(a0, b0) for a0, b0 in pulses]
        prev_end = max([x["t_end"] for qi, x in enumerate(sig_p) if qi < pi and x["J_over_sigma"] >= 2], default=-np.inf)
        next_start = min([x["t_start"] for qi, x in enumerate(sig_p) if qi > pi and x["J_over_sigma"] >= 2],
                         default=np.inf)
        vel, why = velocities(ps["t_start"], ps["t_end"], t_prev=prev_end, t_next=next_start)
        ps["pulse_type"] = ("impulsive" if ps["duration_s"] <= 3 * cam_dt else "sustained_push")
        rec_p = dict(pulse=ps, coast=coast)
        if ps["pulse_type"] == "sustained_push" and vel is None:
            why = dict(why or {}, note="sustained push (longer than 3 camera frames): the momentum balance needs q "
                                       "before AND after the whole push; the standard rollout fit is the estimator "
                                       "for sustained pushes")
        if vel is None:
            rec_p.update(verdict="IMPULSE_UNRESOLVED", reason=why)
        else:
            rec_p["velocity"] = vel
            dv_res = abs(vel["dv"]) / vel["dv_sd"] if vel["dv_sd"] > 0 else math.inf
            if ps["J_over_sigma"] < 3:
                rec_p["verdict"] = "IMPULSE_NOT_INFORMATIVE"
                rec_p["reason"] = f"|J_Q| = {ps['J_over_sigma']:.2f} sigma_J: within the force noise"
            elif dv_res < 3:
                rec_p["verdict"] = "IMPULSE_UNRESOLVED"
                rec_p["reason"] = f"velocity change {dv_res:.2f} sigma: not resolved by q"
            elif ps["oscillatory_transient"]:
                rec_p["verdict"] = "IMPULSE_UNRESOLVED"
                rec_p["reason"] = (f"oscillatory impact transient: net/abs impulse {ps['net_to_abs_impulse']:.2f}, "
                                   f"{ps['significant_sign_changes']} significant sign change(s) within "
                                   f"{ps['n_force_samples']} samples at 50 Hz -- the net J_Q is aliased, not a "
                                   "resolved momentum input")
            else:
                rec_p["verdict"] = "IMPULSE_INFORMATIVE"
            rec_p["sign_consistency"] = dict(
                J_Q_sign=float(np.sign(ps["J_Q"])), dqdot_sign=float(np.sign(vel["dv"])),
                contradiction=bool(np.sign(ps["J_Q"]) != np.sign(vel["dv"]) and ps["J_over_sigma"] >= 3 and dv_res >= 3),
                work_along_tcp_motion_J=ps["work_along_tcp_motion_J"],
                note="a contradiction is REPORTED (declared sign / axes / transient aliasing); the declared force "
                     "sign is never changed or inferred from it")
            if rec_p["sign_consistency"]["contradiction"] and rec_p["verdict"] == "IMPULSE_INFORMATIVE":
                rec_p["verdict"] = "IMPULSE_UNRESOLVED"
                rec_p["reason"] = "J_Q and the velocity change have opposite signs (negative inertia): contradiction"
            # (a) conditional impulse-only scale
            ratios_ok = bool(coast and coast["fit"].get("status") == "OK" and
                             coast["fit"]["k_c_over_M"]["statistically_resolved"])
            vbar = 0.5 * (vel["v_pre"] + vel["v_post"])
            diss = 0.0
            if ratios_ok:
                kk = coast["fit"]["k_c_over_M"]["estimate"]
                aa = coast["fit"]["a_Fc_over_M"]["estimate"]
                diss = (kk * vbar + aa * np.sign(vbar)) * ps["duration_s"]
            denom = vel["dv"] + diss
            M = ps["J_Q"] / denom if denom != 0 else math.inf
            sM = abs(M) * math.hypot(ps["sigma_J_Q"] / abs(ps["J_Q"]) if ps["J_Q"] else math.inf,
                                     vel["dv_sd"] / abs(denom) if denom else math.inf)
            rec_p["impulse_only_conditional_M"] = dict(
                estimate=M, sd=sM, label="CONDITIONAL",
                assumptions=["no configuration load during the pulse", "rigid 1-DoF joint, accepted axis / hinge",
                             "dissipation over the pulse " + ("corrected with the coast ratios" if ratios_ok else
                                                              "NOT corrected (coast ratios unresolved)")],
                dissipation_fraction_of_dv=float(abs(diss / vel["dv"])) if vel["dv"] else None,
                statistically_resolved=bool(math.isfinite(sM) and abs(M) > 0 and sM / abs(M) <= REL and M > 0),
                physically_interpretable_validated=False)
            # (c) sensitivities
            sens = {}
            for lag in lags:
                for sg in (1, -1):
                    v2, _ = velocities(ps["t_start"], ps["t_end"], shift=sg * lag, t_prev=prev_end,
                                       t_next=next_start)
                    if v2:
                        sens[f"lag_{sg * lag * 1e3:+.0f}ms"] = dict(dv=v2["dv"], M=ps["J_Q"] / (v2["dv"] + diss)
                                                                  if (v2["dv"] + diss) else math.inf,
                                                                  label="DECLARED_SCENARIO")
            for sg in (1, -1):
                dJ = sg * sigF * float(np.sum(dt_rows[i0:i1 + 1] * (np.median(sigQ[i0:i1 + 1]) / sigF)))
                sens[f"baseline_bias_{'+' if sg > 0 else '-'}sigmaF"] = dict(
                    J_Q=ps["J_Q"] + dJ, M=(ps["J_Q"] + dJ) / denom if denom else math.inf, label="DECLARED_SCENARIO")
            sens["force_sign_flipped_DIAGNOSTIC_ONLY"] = dict(
                M=-M, note="a flipped sign gives a negative inertia: the declared sign is never changed by this")
            obs = observability or {}
            if obs.get("axis_sigma_deg"):
                u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
                u /= np.linalg.norm(u)
                th = math.tan(math.radians(obs["axis_sigma_deg"]))
                for sg in (1, -1):
                    Qx = effort_rows(rec, joint_type, a + sg * th * u, origin, rows)
                    psx = pulse_stats(i0, i1, Qx)
                    sens[f"hinge_axis_tilt_{'+' if sg > 0 else '-'}1sigma"] = dict(
                        J_Q=psx["J_Q"], M=psx["J_Q"] / denom if denom else math.inf, label="DECLARED_SCENARIO")
            if obs.get("hinge_line_sigma_m") and joint_type == "revolute":
                u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
                u /= np.linalg.norm(u)
                for sg in (1, -1):
                    Qx = effort_rows(rec, joint_type, a, np.asarray(origin) + sg * obs["hinge_line_sigma_m"] * u, rows)
                    psx = pulse_stats(i0, i1, Qx)
                    sens[f"hinge_line_{'+' if sg > 0 else '-'}1sigma"] = dict(
                        J_Q=psx["J_Q"], M=psx["J_Q"] / denom if denom else math.inf, label="DECLARED_SCENARIO")
            if coast and coast["fit"].get("status") == "OK":
                sens["coast_load_model"] = dict(note="the coast is fitted without a configuration load; a hinge "
                                                     "load would bias c/M (reported, not corrected)")
            rec_p["sensitivity"] = sens
            Ms = [v["M"] for k2, v in sens.items() if "M" in v and "flipped" not in k2 and math.isfinite(v["M"])]
            rec_p["impulse_only_conditional_M"]["scenario_envelope"] = [min(Ms + [M]), max(Ms + [M])] if Ms else None
        out["pulses"].append(rec_p)
        if coast is not None and coast_rep is None:
            coast_rep = coast
    out["coast"] = coast_rep
    out["joint_pulse_coast_fit"] = ("the standard G9 general_rollout_fit of this window (estimator_results/"
                                    "general_rollout_fit.json) uses the same native ZOH effort for pulse and coast "
                                    "with shared parameters")
    vs = [p["verdict"] for p in out["pulses"]]
    out["verdict"] = ("IMPULSE_INFORMATIVE" if "IMPULSE_INFORMATIVE" in vs else
                      "IMPULSE_UNRESOLVED" if "IMPULSE_UNRESOLVED" in vs else
                      "IMPULSE_NOT_INFORMATIVE" if vs else "NO_PULSE_FOUND")
    out["summary"] = dict(
        n_pulses=len(out["pulses"]), SNR_pulse=max((p["pulse"]["SNR_pulse_peak"] for p in out["pulses"]), default=None),
        J_Q=[p["pulse"]["J_Q"] for p in out["pulses"]], J_over_sigma=[p["pulse"]["J_over_sigma"] for p in out["pulses"]],
        dqdot=[p.get("velocity", {}).get("dv") for p in out["pulses"]],
        dqdot_sd=[p.get("velocity", {}).get("dv_sd") for p in out["pulses"]],
        SNR_coast=None if coast_rep is None else coast_rep["SNR_coast"],
        coast_decay_time=None if coast_rep is None or coast_rep["fit"].get("status") != "OK" else
        coast_rep["fit"]["decay_time_M_over_c_s"])
    out["_plot"] = dict(t=tc, Q=Q, sigQ=sigQ, state=state, tq=tq, q=qv, sq=sv)
    return out


def plot(an, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = an["_plot"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].step(d["t"], d["Q"], where="post", lw=1, label="Q (native, ZOH)")
    ax[0].fill_between(d["t"], -2 * d["sigQ"], 2 * d["sigQ"], step="post", alpha=0.2, label="+/- 2 sigma_Q")
    for p in an["pulses"]:
        ax[0].axvspan(p["pulse"]["t_start"], p["pulse"]["t_end"], color="r", alpha=0.2)
        ax[1].axvspan(p["pulse"]["t_start"], p["pulse"]["t_end"], color="r", alpha=0.2)
        if p.get("coast"):
            ax[1].axvspan(*(np.array(p["coast"]["rows"]) / 50.0), color="g", alpha=0.08)
        v = p.get("velocity")
        if v:
            for tt, vv, ss in ((p["pulse"]["t_start"], v["v_pre"], v["v_pre_sd"]),
                               (p["pulse"]["t_end"], v["v_post"], v["v_post_sd"])):
                ax[1].annotate(f"q'={vv:.3f}+/-{ss:.3f}", (tt, float(np.interp(tt, d["tq"], d["q"]))), fontsize=7)
    con = d["state"] == "contact"
    ax[0].plot(d["t"][con], np.zeros(con.sum()), "k|", ms=6, label="contact rows")
    ax[0].set_ylabel("Q")
    ax[0].legend(fontsize=7)
    ax[1].errorbar(d["tq"], d["q"], yerr=d["sq"], fmt=".", ms=3)
    ax[1].set_ylabel("q")
    ax[1].set_xlabel("t [s] (source_frame_index / 50)")
    ax[0].set_title(f"impulse / coast: {an['verdict']}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
