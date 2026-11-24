"""Observability of the analytical output-error model: finite-difference sensitivities in DIMENSIONLESS transformed
coordinates (eta = log theta for positive parameters), noise-whitened; singular values, rank, correlations; and the
structural / practical classification (separate status dimensions -- seed spread is never a confidence interval)."""
from __future__ import annotations

import math

import numpy as np

RANK_TOL = 1e-6


def sensitivity(oe, *, h=1e-4) -> dict:
    """J_ij = d q(t_i) / d eta_j / sigma_i on the training windows, eta = log(theta) for positives (log_g_u is
    already a log), linear for signed parameters; initial-state nuisance columns included (rank) and excluded
    (theta-only correlations)."""
    model, z = oe["_model"], np.array(oe["_z"], float)
    names = list(model.names)
    cols, labels, kinds = [], [], []

    def q_all(zz):
        th, ics = model.theta(zz), model.ic(zz)
        out = []
        for e in model.train:
            q0, v0 = ics[e.name]
            out.append(model.whiten(e, model.simulate(th, e, q0, v0).q))
        return np.concatenate(out)
    base = q_all(z)
    nonfinite = []
    for i in range(len(z)):
        name = names[i] if i < len(names) else f"ic[{i - len(names)}]"
        x = z[i]
        lo, hi = model.lb[i], model.ub[i]
        # a rate AT its zero bound (the solver may return a denormal) has no log coordinate: one-sided linear step
        logc = name in ("lam", "gam", "c", "Fc") and x > 1e-6 * max(hi - lo, 1e-12)
        if logc:
            dz = x * (math.exp(h * 10) - 1)
        else:
            dz = h * max(abs(x), 1e-3) * 10
        zp = z.copy()
        if zp[i] + dz > hi:
            dz = -dz
        zp[i] += dz
        try:
            d = (q_all(zp) - base) / dz if dz != 0 else np.full_like(base, np.nan)
        except (ValueError, FloatingPointError, OverflowError):
            d = np.full_like(base, np.nan)
        if not np.isfinite(d).all():
            nonfinite.append(name)
            d = np.zeros_like(base)                # reported; the zero column makes the rank test fail closed
        cols.append(d * (x if logc else 1.0))       # d q / d log x = x d q / d x
        labels.append(name)
        kinds.append("log" if logc or name == "log_g_u" else ("linear_at_zero_bound" if name in (
            "lam", "gam", "c", "Fc") else "linear"))
    Jf = np.column_stack(cols)
    nt = len(names)

    def svd(J, labs):
        if J.shape[1] == 0:
            return dict(singular_values=[], rank=0, condition_number=None)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        rank = int(np.sum(S > RANK_TOL * S[0])) if S[0] > 0 else 0
        out = dict(singular_values=S.tolist(), rank=rank, n=len(labs),
                   condition_number=float(S[0] / S[-1]) if S[-1] > 0 else math.inf,
                   weakest_direction={l: float(v) for l, v in zip(labs, Vt[-1])})
        C = np.linalg.pinv(J.T @ J)
        d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
        out["correlation"] = (C / np.outer(d, d)).tolist()
        out["labels"] = labs
        return out
    full = svd(Jf, labels)
    theta = svd(Jf[:, :nt], labels[:nt])
    # theta correlations after the nuisance (initial states) is profiled out (Schur complement)
    C = np.linalg.pinv(Jf.T @ Jf)[:nt, :nt]
    d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    return dict(coordinates={l: k for l, k in zip(labels, kinds)}, full=full, theta_only=theta,
                nonfinite_columns=nonfinite,
                theta_correlation_with_nuisance=(C / np.outer(d, d)).tolist(), theta_labels=labels[:nt],
                whitening="chol of diag(sigma_q^2) + propagated input-noise covariance (forced windows), "
                          "diagonal sigma_q otherwise",
                note="d q / d eta with eta = log theta for positive rates; columns of the nuisance initial states "
                     "are part of the rank test")


def structural(oe, sens, n_episodes) -> tuple:
    if n_episodes == 0 or oe is None:
        return "NOT_TESTABLE", "no admissible episode"
    nt = len(oe["names"])
    if sens and sens.get("nonfinite_columns"):
        return "RANK_DEFICIENT", f"non-finite sensitivity columns {sens['nonfinite_columns']} (zeroed, fail closed)"
    if sens and sens["full"]["rank"] < sens["full"]["n"]:
        return "RANK_DEFICIENT", f"sensitivity rank {sens['full']['rank']} < {sens['full']['n']} (including the " \
                                 "initial-state nuisance)"
    if not oe.get("forced_in_training"):
        return "RATIO_ONLY", "free-only training data: (M, c, Fc) -> (kM, kc, kFc) is an exact symmetry"
    return "FULL", f"forced input breaks the scale symmetry; sensitivity rank {sens['full']['rank'] if sens else nt}"


def practical(oe, feats, scale_profile) -> tuple:
    if oe is None:
        return "NOT_TESTABLE", "no fit"
    if (oe.get("multistart") or {}).get("unstable"):
        return "UNSTABLE", "multi-start optima within the profile threshold disagree"
    if oe.get("n_eval") and oe.get("n_failed", 0) > 0.5 * oe["n_eval"]:
        return "UNSTABLE", f"{oe['n_failed']}/{oe['n_eval']} rollouts failed"
    hits = [k for k, v in (oe.get("at_bound") or {}).items() if v and k not in ("lam", "gam", "c", "Fc")]
    if hits:
        return "BOUNDARY_ONLY", f"estimate at the search bound: {hits}"
    if oe.get("forced_in_training") and scale_profile and scale_profile.get("status") == "OPEN":
        return "BOUNDARY_ONLY", "absolute-scale profile open at the grid edge"
    key = [f for f in feats if f.name in ("lambda", "gamma", "gain") and f.status != "NOT_APPLICABLE"]
    if not oe.get("forced_in_training"):
        key = [f for f in key if f.name != "gain"]           # the gain is structurally absent, not "weak"
    weak = [f.name for f in key if f.status == "NOT_IDENTIFIED"]
    if weak:
        return "WEAK", f"not identified: {weak}"
    return "INFORMATIVE", "closed, narrow profile intervals for " + ", ".join(f.name for f in key)
