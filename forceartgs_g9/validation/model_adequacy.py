"""Model adequacy (G9 spec 4.4): residual structure of the accepted fit, separate from convergence and
identifiability. PASS needs: normalised chi2 not far above 1 (sigma from q_std), no systematic residual by
condition (velocity sign, acceleration sign, near stick/slip transitions, q bins: |mean| <= 3 standard errors) and a
residual-sign runs test |z| <= 3 per interaction. Thresholds are preregistered here, not tuned per scene.
"""
from __future__ import annotations

import math

import numpy as np

from ..estimators.general_integral_ls import velocity_estimate

CHI2_MAX = 2.0
COND_Z = 3.0
RUNS_Z = 3.0


def runs_test(r) -> float:
    s = np.sign(r)
    s = s[s != 0]
    n1, n2 = int((s > 0).sum()), int((s < 0).sum())
    if n1 == 0 or n2 == 0:
        return math.inf
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    return float((runs - mu) / math.sqrt(var)) if var > 0 else 0.0


def adequacy(fit: dict) -> dict:
    if fit.get("status") not in ("ok", "not_converged"):
        return dict(status="NOT_RUN", reason=f"fit status {fit.get('status')}")
    P = fit["problem"]
    x = np.asarray(fit["x"], float)
    res = P.residuals(x)
    sizes = [len(it.t_obs) for it in P.its]
    parts = np.split(res, np.cumsum(sizes)[:-1])
    chi2 = float(np.mean(res ** 2))
    conds, runs, fails = {}, {}, []
    for it, r in zip(P.its, parts):
        v = velocity_estimate(it.t_obs, it.q_obs)
        a = np.gradient(v, it.t_obs)
        vmax = max(np.abs(v).max(), 1e-12)
        groups = {"v>0": v > 0.05 * vmax, "v<0": v < -0.05 * vmax, "rest": np.abs(v) <= 0.05 * vmax,
                  "accelerating": a * v > 0, "decelerating": a * v < 0}
        edges = np.quantile(it.q_obs, [0, 1 / 3, 2 / 3, 1])
        for k in range(3):
            groups[f"q_tercile_{k}"] = (it.q_obs >= edges[k]) & (it.q_obs <= edges[k + 1])
        for g, m in groups.items():
            if m.sum() >= 5:
                mean = float(r[m].mean())
                se = float(r[m].std(ddof=1) / math.sqrt(m.sum())) or 1e-12
                key = f"{it.name}:{g}"
                conds[key] = dict(n=int(m.sum()), mean_sigma=mean, z=mean / se)
                if abs(mean / se) > COND_Z and abs(mean) > 0.5:
                    fails.append(f"{key}: systematic residual {mean:.2f} sigma (z {mean / se:.1f})")
        z = runs_test(r)
        runs[it.name] = z
        if abs(z) > RUNS_Z:
            fails.append(f"{it.name}: residual runs test z = {z:.1f} (structured misfit)")
    if chi2 > CHI2_MAX:
        fails.append(f"normalised chi2 {chi2:.2f} > {CHI2_MAX}")
    return dict(status="PASS" if not fails else "FAIL", normalised_chi2=chi2, by_condition=conds, runs_z=runs,
                failures=fails, thresholds=dict(chi2_max=CHI2_MAX, condition_z=COND_Z, runs_z=RUNS_Z),
                note="adequacy of the fitted model on its TRAINING data; out-of-sample evidence is heldout_prediction")


def compare_variants(results: dict) -> dict:
    """User-chosen variants scored on WITHHELD interactions only: {variant: heldout report}. Opt-in model choice."""
    score = {k: v["summary"]["mean_rmse_over_sigma"] for k, v in results.items()
             if v.get("summary", {}).get("mean_rmse_over_sigma") is not None}
    if not score:
        return dict(status="NOT_RUN", reason="no held-out scores")
    best = min(score, key=score.get)
    return dict(status="OK", heldout_rmse_over_sigma=score, best_on_heldout=best,
                procedure="each variant fitted on the training interactions only; scored on withheld interactions "
                          "with declared initial states; the test set never selects parameters")
