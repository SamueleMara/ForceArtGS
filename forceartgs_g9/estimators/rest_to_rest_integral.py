"""Rest-to-rest integral identification of the two-term model  M q'' + c q' = Q  (initializers + diagnostics).

ENDPOINT_R2R (Eqs. 2, 5) -- integrate once / twice over [t0, T] with v(t0) = v(T) = 0:

    c_end = J_Q / dq,                J_Q = int_{t0}^{T} Q dt,  dq = qT - q0
    M_end = (y(T) - c_end int_{t0}^{T} (q - q0) ds) / dq,     y(t) = int_{t0}^{t} (t - s) Q(s) ds

INTEGRAL_SVD_R2R (Eqs. 3, 4) -- every visual time t_i in (t0, T]:

    M [q(t_i) - q0] + c int_{t0}^{t_i} [q - q0] ds = y(t_i)      (v0 = 0; NO per-segment M v_a / offset nuisance)

y and J_Q are EXACT for the forward-ZOH native effort (``general_integral_ls.zoh_double_integral``); the q integral is a
trapezoid on the observed nonuniform visual clock. No velocity / acceleration is differentiated from q.

Noise model (declared, never inflated by fitted residuals): q samples (q_sigma), the plateau anchors q0 / qT (their own
sigma), native wrench rows (Q_sigma, white) and the shared tare reference (row sd / sqrt(tare_rows), fully correlated
over the episode). Cov(A theta - y) is propagated LINEARLY through the cumulative integrals at an initial theta; the
same model drives a parametric bootstrap that recomputes every integral per draw (time correlation preserved). These
are DIAGNOSTICS ON ONE DATA SET, not independent validation; the primary ranking is the PhysX output error.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import lsq_linear

from .general_integral_ls import zoh_double_integral
from .rest_to_rest import RestToRestEpisode, ZOHEffort

ENDPOINT_ILL = "ENDPOINT_ESTIMATE_ILL_CONDITIONED"
NAMES = ("M", "c")


# ------------------------------------------------------------------------------------------------ building blocks
def interval_kernel(force_t, t0, t_eval):
    """G[i, k] = int over row k's ZOH interval within [t0, t_i] of (t_i - s) ds, so y = G @ Q exactly, and
    H[i, k] = length of that interval (J_Q(t_i) = H @ Q)."""
    ft = np.asarray(force_t, float)
    a = np.maximum(ft, t0)
    b = np.r_[ft[1:], np.inf]
    b = np.maximum(b, t0)
    t = np.asarray(t_eval, float)[:, None]
    e = np.minimum(b[None, :], t)
    act = t > a[None, :]
    G = np.where(act, 0.5 * (t - a[None, :]) ** 2 - 0.5 * (t - e) ** 2, 0.0)
    H = np.where(act, e - a[None, :], 0.0)
    return G, H


def trapz_weights(t):
    """W[i, j]: int_{t_0}^{t_i} f ds ~= sum_j W[i, j] f(t_j) (cumulative trapezoid, nonuniform clock)."""
    t = np.asarray(t, float)
    n = len(t)
    W = np.zeros((n, n))
    for i in range(1, n):
        W[i] = W[i - 1]
        h = t[i] - t[i - 1]
        W[i, i - 1] += 0.5 * h
        W[i, i] += 0.5 * h
    return W


def design(ep: RestToRestEpisode, q=None, q0=None, Q=None):
    """Rows i = 1 .. n-1 (row 0 is t0: all zero). Returns A [n-1, 2], y, and the kernels for the covariance."""
    t = ep.t_q_s
    q = ep.q_obs if q is None else q
    q0 = ep.q0 if q0 is None else q0
    d = q - q0
    A2 = cumulative_trapezoid(d, t, initial=0.0)
    if Q is None:
        _, y = zoh_double_integral(ep.effort, ep.t0, t)
    else:
        G, _ = interval_kernel(ep.force_t_s, ep.t0, t)
        y = G @ np.asarray(Q, float)
    A = np.c_[d, A2]
    return A[1:], y[1:]


def tare_sigma(ep, tare_rows=25):
    if ep.force_sigma is None:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(ep.force_sigma) ** 2)) / math.sqrt(tare_rows))


def residual_covariance(ep: RestToRestEpisode, theta, *, tare_rows=25) -> tuple:
    """Declared Cov(A theta - y) on rows 1..n-1, linearized at theta = (M, c), and its decomposition."""
    M, c = (float(x) for x in theta)
    t = ep.t_q_s
    W = trapz_weights(t)
    n = len(t)
    Jq = M * np.eye(n) + c * W                                   # d r_i / d q_j
    u = -(M + c * (t - ep.t0))                                    # d r_i / d q0
    G, _ = interval_kernel(ep.force_t_s, ep.t0, t)
    sQ = np.zeros(len(ep.force_Q)) if ep.force_sigma is None else np.asarray(ep.force_sigma, float)
    g = 0.5 * (t - ep.t0) ** 2
    sb = tare_sigma(ep, tare_rows)
    parts = dict(q_samples=(Jq * ep.q_sigma ** 2) @ Jq.T, q0_anchor=ep.q0_sigma ** 2 * np.outer(u, u),
                 force_rows=(G * sQ ** 2) @ G.T, tare_reference=sb ** 2 * np.outer(g, g))
    C = sum(parts.values())[1:, 1:]
    return C, {k: float(np.sqrt(np.mean(np.diag(v[1:, 1:])))) for k, v in parts.items()}, sb


def _bounded(A, y, lo=(1e-12, 0.0)):
    sc = np.linalg.norm(A, axis=0)
    sc[sc == 0] = 1.0
    An = A / sc
    xu, *_ = np.linalg.lstsq(An, y, rcond=None)
    r = lsq_linear(An, y, bounds=(np.asarray(lo) * sc, np.full(A.shape[1], np.inf)), method="bvls")
    return r.x / sc, xu / sc, An


def _svd_report(An):
    sv = np.linalg.svd(An, compute_uv=False)
    rank = int((sv > sv[0] * 1e-10).sum()) if len(sv) and sv[0] > 0 else 0
    return dict(singular_values_column_scaled=sv.tolist(), rank=rank, n_columns=int(An.shape[1]),
                condition_number=float(sv[0] / sv[-1]) if len(sv) and sv[-1] > 0 else math.inf,
                note="singular values of the column-scaled (dimensionless) design: a numerical property of THIS "
                     "excitation; a large condition number is not by itself a proof of physical unobservability")


def _gls(A, y, C):
    Lc = np.linalg.cholesky(C + np.eye(len(C)) * 1e-12 * max(np.trace(C) / len(C), 1e-300))
    Aw = np.linalg.solve(Lc, A)
    yw = np.linalg.solve(Lc, y)
    return Aw, yw


def _huber_irls(Aw, yw, x0, delta=2.0, iters=30):
    x = np.array(x0, float)
    for _ in range(iters):
        r = Aw @ x - yw
        w = np.sqrt(np.minimum(1.0, delta / np.maximum(np.abs(r), 1e-300)))
        xn, _, _ = _bounded(Aw * w[:, None], yw * w)
        if np.allclose(xn, x, rtol=1e-10, atol=0):
            x = xn
            break
        x = xn
    return x


# ------------------------------------------------------------------------------------------------ estimators
def endpoint_r2r(ep: RestToRestEpisode, *, k_sigma=3.0, tare_rows=25, draws=None) -> dict:
    """Eqs. (2), (5) with explicit uncertainty and denominator safety. Signed values are kept; nothing is projected
    into the admissible set. ``draws``: optional parametric-bootstrap draws (see ``noise_draws``)."""
    t = ep.t_q_s
    I1, I2 = zoh_double_integral(ep.effort, ep.t0, np.array([ep.T]))
    JQ, yT = float(I1[0]), float(I2[0])
    dq = ep.delta_q
    s_dq = math.hypot(ep.q0_sigma, ep.qT_sigma)
    _, H = interval_kernel(ep.force_t_s, ep.t0, np.array([ep.T]))
    sQ = np.zeros(len(ep.force_Q)) if ep.force_sigma is None else np.asarray(ep.force_sigma, float)
    sb = tare_sigma(ep, tare_rows)
    s_JQ = math.hypot(float(np.sqrt(np.sum((H[0] * sQ) ** 2))), sb * (ep.T - ep.t0))
    Iq = float(np.trapezoid(ep.q_obs - ep.q0, t)) if hasattr(np, "trapezoid") else float(np.trapz(ep.q_obs - ep.q0, t))
    snr_dq = abs(dq) / max(s_dq, 1e-300)
    snr_J = abs(JQ) / max(s_JQ, 1e-300)
    out = dict(method="ENDPOINT_R2R", JQ=JQ, JQ_sigma=s_JQ, yT=yT, q0=ep.q0, qT=ep.qT, delta_q=dq, delta_q_sigma=s_dq,
               delta_q_snr=snr_dq, impulse_snr=snr_J, int_q_minus_q0=Iq, t0=ep.t0, T=ep.T,
               formulas=dict(c_end="J_Q / (qT - q0)", M_end="(y(T) - c_end int (q - q0)) / (qT - q0)"),
               provenance=dict(t0=ep.t0, T=ep.T, frames=ep.describe().get("source_frames"),
                               episode_sha256=ep.sha256()))
    reasons = []
    if snr_dq < k_sigma:
        reasons.append(f"|qT - q0| = {abs(dq):.3g} is not distinguishable from its uncertainty {s_dq:.3g} "
                       f"(snr {snr_dq:.3g} < {k_sigma})")
    if snr_J < k_sigma:
        reasons.append(f"|J_Q| = {abs(JQ):.3g} is within {k_sigma} x its uncertainty {s_JQ:.3g}")
    if reasons:
        out.update(admissible=False, status=ENDPOINT_ILL, skipped_reason="; ".join(reasons), c_end=None, M_end=None,
                   fallback="multi-time double-integral regression and PhysX output-error fitting")
        return out
    c = JQ / dq
    M = (yT - c * Iq) / dq
    out.update(admissible=True, c_end=c, M_end=M,
               c_end_sigma_first_order=abs(c) * math.hypot(s_JQ / abs(JQ), s_dq / abs(dq)))
    flags = []
    if c < 0:
        flags.append("NEGATIVE_ENDPOINT_DAMPING")
    if M <= 0:
        flags.append("NONPOSITIVE_ENDPOINT_INERTIA")
    out.update(flags=flags, status=("MODEL_INPUT_INCONSISTENT" if flags else "OK"),
               interpretation=("a signed contradiction of the two-term model with the recorded input (force bias, "
                               "gravity, Coulomb friction or a sustained holding wrench can do this); NOT clipped"
                               if flags else "candidate estimate on the same data, not the best-fit simulator"))
    if draws:
        cs, Ms = [], []
        for d in draws:
            I1d, I2d = zoh_double_integral(ZOHEffort(ep.force_t_s, d["Q"]), ep.t0, np.array([ep.T]))
            dqd = d["qT"] - d["q0"]
            if abs(dqd) < 1e-300:
                continue
            cd = float(I1d[0]) / dqd
            Iqd = float(np.trapezoid(d["q"] - d["q0"], t)) if hasattr(np, "trapezoid") else \
                float(np.trapz(d["q"] - d["q0"], t))
            cs.append(cd)
            Ms.append((float(I2d[0]) - cd * Iqd) / dqd)
        if cs:
            out["bootstrap"] = dict(n=len(cs), c_p2_5_p97_5=np.percentile(cs, [2.5, 97.5]).tolist(),
                                    M_p2_5_p97_5=np.percentile(Ms, [2.5, 97.5]).tolist(), c_sd=float(np.std(cs)),
                                    M_sd=float(np.std(Ms)))
    return out


def noise_draws(ep: RestToRestEpisode, n, seed=0, tare_rows=25) -> list:
    """Parametric bootstrap of the DECLARED noise: q samples, plateau anchors, native rows (white) and one shared
    tare-reference offset per draw; every integral is recomputed from the perturbed series (time correlation kept)."""
    rng = np.random.default_rng(seed)
    sQ = np.zeros(len(ep.force_Q)) if ep.force_sigma is None else np.asarray(ep.force_sigma, float)
    sb = tare_sigma(ep, tare_rows)
    out = []
    for _ in range(int(n)):
        out.append(dict(q=ep.q_obs + rng.standard_normal(len(ep.q_obs)) * ep.q_sigma,
                        q0=ep.q0 + rng.standard_normal() * ep.q0_sigma, qT=ep.qT + rng.standard_normal() * ep.qT_sigma,
                        Q=ep.force_Q + rng.standard_normal(len(sQ)) * sQ + rng.standard_normal() * sb))
    return out


def integral_svd_r2r(ep: RestToRestEpisode, *, tare_rows=25, draws=None, huber_delta=2.0,
                     family_k=2.0) -> dict:
    """Constrained (M > 0, c >= 0) double-integral LS: unweighted, declared-covariance whitened (GLS) and a robust
    (Huber IRLS on the whitened residual) variant; signed unconstrained solutions; column-scaled SVD."""
    A, y = design(ep)
    xb_u, xu_u, An_u = _bounded(A, y)
    theta0 = np.array([max(xb_u[0], 1e-9), max(xb_u[1], 0.0)])
    if not np.isfinite(theta0).all() or theta0[0] <= 1e-9:
        theta0 = np.array([max(abs(xu_u[0]), 1e-6), max(xu_u[1], 0.0)])
    C, comp, sb = residual_covariance(ep, theta0, tare_rows=tare_rows)
    Aw, yw = _gls(A, y, C)
    xb_w, xu_w, An_w = _bounded(Aw, yw)
    # one re-linearization of the covariance at the GLS estimate (declared noise only, no residual inflation)
    th1 = np.array([max(xb_w[0], 1e-9), max(xb_w[1], 0.0)])
    C, comp, sb = residual_covariance(ep, th1, tare_rows=tare_rows)
    Aw, yw = _gls(A, y, C)
    xb_w, xu_w, An_w = _bounded(Aw, yw)
    x_rob = _huber_irls(Aw, yw, xb_w, delta=huber_delta)
    cov = np.linalg.pinv(Aw.T @ Aw)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = float(cov[0, 1] / max(sd[0] * sd[1], 1e-300))
    rw = Aw @ xb_w - yw
    dof = max(len(yw) - 2, 1)
    ev, V = np.linalg.eigh(cov)
    fam = [dict(label=f"axis{j}_{s:+d}sd", M=float(xb_w[0] + s * family_k * math.sqrt(max(ev[j], 0)) * V[0, j]),
                c=float(xb_w[1] + s * family_k * math.sqrt(max(ev[j], 0)) * V[1, j]))
           for j in range(2) for s in (-1, 1)]
    rel = sd / np.maximum(np.abs(xb_w), 1e-300)
    svd_w = _svd_report(An_w)
    unique = bool(svd_w["rank"] == 2 and np.all(rel < 0.5) and svd_w["condition_number"] < 1e8)
    lo_active = dict(M=bool(xb_w[0] <= 1e-9 * max(abs(xu_w[0]), 1.0)), c=bool(xb_w[1] <= 0.0))
    viol = [n for n, v in zip(NAMES, xu_w) if v < 0]
    out = dict(method="INTEGRAL_SVD_R2R", n_rows=int(len(y)), episode_sha256=ep.sha256(),
               rows="every visual time t_i in (t0, T]; no per-segment v_a or offset nuisance (v0 = 0 declared)",
               unweighted=dict(M=float(xb_u[0]), c=float(xb_u[1]), unconstrained=dict(M=float(xu_u[0]),
                                                                                         c=float(xu_u[1])),
                               svd=_svd_report(An_u)),
               whitened=dict(M=float(xb_w[0]), c=float(xb_w[1]), unconstrained=dict(M=float(xu_w[0]), c=float(xu_w[1])),
                             sd=dict(M=float(sd[0]), c=float(sd[1])), correlation_M_c=corr,
                             chi2_dof=float(rw @ rw / dof), svd=svd_w, covariance_components_rms=comp,
                             tare_reference_sigma=sb, linearized_at=th1.tolist(),
                             covariance="declared: q samples + q0 anchor + native rows (white) + shared tare reference, "
                                        "propagated through the cumulative integrals at linearized_at"),
               robust=dict(M=float(x_rob[0]), c=float(x_rob[1]), loss=f"Huber(delta={huber_delta}) IRLS on the "
                           "whitened residuals: down-weights rows with |r_w| > delta (changes the loss semantics: no "
                           "longer the Gaussian likelihood of the declared noise)"),
               estimate=dict(M=float(xb_w[0]), c=float(xb_w[1]), variant="whitened (GLS)"),
               bound_activity=lo_active, sign_violations_unconstrained=viol,
               physically_consistent=not viol, parameters_unique=unique, flags=[],
               near_optimal_family=dict(rule=f"+-{family_k} sd along the principal axes of the GLS covariance "
                                             "(declared noise)", members=fam,
                                        relative_sd=dict(M=float(rel[0]), c=float(rel[1]))))
    if draws:
        bs = []
        for d in draws:
            Ad, yd = design(ep, q=d["q"], q0=d["q0"], Q=d["Q"])
            Adw, ydw = _gls(Ad, yd, C)
            xd, _, _ = _bounded(Adw, ydw)
            bs.append(xd)
        bs = np.array(bs)
        bias = bs.mean(0) - xb_w
        bsd = np.maximum(bs.std(0), 1e-300)
        eiv = bool(np.any(np.abs(bias) > bsd))
        out["whitened"]["bootstrap"] = dict(n=len(bs), M_p2_5_p97_5=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                                            c_p2_5_p97_5=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                                            M_sd=float(bs[:, 0].std()), c_sd=float(bs[:, 1].std()),
                                            c_at_zero_fraction=float(np.mean(bs[:, 1] <= 0.0)),
                                            bootstrap_bias=dict(M=float(bias[0]), c=float(bias[1])),
                                            note="parametric bootstrap of the declared noise (integrals recomputed)")
        if eiv:
            out["flags"] = sorted(set(out.get("flags", [])) | {"ERRORS_IN_VARIABLES_BIAS_SUSPECTED"})
            out["parameters_unique"] = False
            out["errors_in_variables"] = (
                "the noisy q enters the design columns (q - q0, int (q - q0)): the LS is attenuated toward M -> 0 when "
                "the q noise is not small against the motion; detected as |bootstrap mean - estimate| > bootstrap sd. "
                "Whitening does not remove this bias; use the output-error methods (convolution / PhysX) for ranking")
    return out


def bias_ablation(ep: RestToRestEpisode, *, tare_rows=25) -> dict:
    """LABELLED consistency model M q'' + c q' + b0 = Q (b0: constant unmodelled load, NOT static friction). With
    rest-to-rest, c dq + b0 (T - t0) = J_Q: the endpoint relation alone cannot separate c from b0."""
    A, y = design(ep)
    g = 0.5 * (ep.t_q_s[1:] - ep.t0) ** 2
    A3 = np.c_[A, g]
    sc = np.linalg.norm(A3, axis=0)
    sc[sc == 0] = 1.0
    th0, *_ = np.linalg.lstsq(A, y, rcond=None)
    C, _, _ = residual_covariance(ep, [max(th0[0], 1e-9), max(th0[1], 0.0)], tare_rows=tare_rows)
    Aw, yw = _gls(A3, y, C)
    scw = np.linalg.norm(Aw, axis=0)
    scw[scw == 0] = 1.0
    x, *_ = np.linalg.lstsq(Aw / scw, yw, rcond=None)
    x = x / scw
    cov = np.linalg.pinv(Aw.T @ Aw)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    return dict(model="ABLATION: M q'' + c q' + b0 = Q (constant load; diagnostic only, never in the G10 fit)",
                M=float(x[0]), c=float(x[1]), b0=float(x[2]), sd=dict(M=float(sd[0]), c=float(sd[1]), b0=float(sd[2])),
                b0_z=float(abs(x[2]) / max(sd[2], 1e-300)), svd=_svd_report(Aw / scw),
                note="b0 is NOT static friction; with b0 free the rest-to-rest endpoint relation c dq + b0 (T - t0) = "
                     "J_Q has one equation for two unknowns")


def map_to_physx(M, c, info, bounds, *, label) -> dict:
    """(M, c) -> PhysX (rho_scale = M / axis_inertia of the SAME reference URDF, viscous = c, Td = 0, static excess 0,
    armature fixed). Rejected (for the warm start only) when not physical; clipped into the declared bounds with a log."""
    from ..physx_refinement.behavioral_fit import params_from
    I = float(info["axis_inertia"])
    rec = dict(label=label, M=M, c=c, axis_inertia=I, joint_type=info.get("joint_type"),
               rule="rho_scale = M / axis_inertia(reference.urdf at rho = 1) (revolute: axis inertia incl. parallel axis; "
                    "prismatic: moving mass); viscous = c; dynamic_effort = static_excess = 0; armature fixed")
    if M is None or c is None or not (np.isfinite(M) and np.isfinite(c)) or M <= 0 or c < 0 or I <= 0:
        rec.update(status="REJECTED", reason="non-physical or missing (M, c): not usable as a PhysX warm start")
        return rec
    rho = float(M) / I
    raw = dict(rho_scale=rho, armature=bounds["armature"][0], dynamic_effort=0.0, static_excess=0.0, viscous=float(c))
    clipped = {k: min(max(v, bounds[k][0]), bounds[k][1]) for k, v in raw.items()}
    clip_log = {k: [raw[k], clipped[k]] for k in raw if clipped[k] != raw[k]}
    rec.update(status="OK" if not clip_log else "OK_CLIPPED", rho_scale_raw=rho, params=clipped, clip_log=clip_log,
               physical=params_from(clipped))
    return rec


def output_error_crosscheck(ep: RestToRestEpisode, M, c, conv: dict, *, tol_sigma=0.5, numeric_floor_q=3e-4) -> dict:
    """Replay the integral-LS (M, c) EXACTLY (two-term model from (q0, 0), native ZOH effort) and compare its raw-q error
    with the output-error optimum of the SAME model (ANALYTIC_CONVOLUTION_1D). A gap > tol_sigma means the LS solution
    is not near the trajectory optimum (equation-error bias: noisy q in the design, a T placed inside motion that is
    below the q resolution, ...): rank by the output error, use the LS only as a start."""
    from .rest_to_rest_convolution import replay
    if M is None or not (M > 0) or conv.get("status") != "OK":
        return dict(status="NOT_EVALUATED")
    r = replay(ep, M, max(c, 0.0))
    sig = np.hypot(ep.q_sigma, numeric_floor_q)
    ros = float(np.sqrt(np.mean(((r["q"] - ep.q_obs) / sig) ** 2)))
    gap = ros - float(conv["rmse_over_sigma"])
    return dict(status=("INTEGRAL_LS_OUTPUT_ERROR_INCONSISTENT" if gap > tol_sigma else "CONSISTENT"),
                ls_replay_rmse_over_sigma=ros, output_error_optimum_rmse_over_sigma=float(conv["rmse_over_sigma"]),
                gap_sigma=gap, tol_sigma=tol_sigma,
                rule=f"exact replay of the LS theta vs the convolution optimum; gap > {tol_sigma} sigma -> inconsistent")
