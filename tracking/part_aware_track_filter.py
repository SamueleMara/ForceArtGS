"""Part-aware trajectory pre-filter (``track_filter_mode: part_aware``).

The legacy pre-filter in ``Tracker.get_visible_trajectories`` rejects every trajectory whose
displacement magnitude is more than 2 standard deviations from the *global* mean.  With two
rigid parts the displacement distribution is a mixture, so the minority (usually the moving
part) is rejected preferentially, before segmentation can use it.

This filter instead explains every candidate with one of two rigid motions and rejects a
trajectory only when it is inconsistent with *both*:

1. Hypothesis A: least-median-of-squares (LMedS) rigid fit on all candidates (no threshold,
   no part labels, 50% breakdown).
2. Hypothesis B: LMedS on the candidates that A does not explain.
3. Alternate (bounded number of rounds): assign each candidate to the hypothesis with the
   smaller residual, re-estimate each part's robust residual scale from its own assignments,
   and refit each motion by weighted Kabsch on the candidates it explains.  Candidates that
   both motions explain (e.g. near the joint, or when a motion is small) are ambiguous; they
   enter both refits with weight 1/2 and are kept.
4. Keep a candidate iff its residual is within the inlier bound of A or of B.

No part assignment ever removes a candidate before the final test, so an initially wrong
assignment cannot permanently discard the tracks needed to correct it.  The inlier bound
mirrors the legacy rule's tail probability (2 sigma two-sided, 4.55%) applied to the correct
per-part distribution: for isotropic Gaussian 3D noise the residual norm r satisfies
r^2/sigma^2 ~ chi^2_3, so P(r > 2.834 sigma) = 4.55% and median(r) = 1.538 sigma, i.e.
bound = (2.834 / 1.538) * median(r_part).  No new tuning constant is introduced.
"""
from __future__ import annotations

import math

import torch

# chi^2_3 quantile at 1 - 0.0455 (the legacy two-sided 2-sigma tail) and chi^2_3 median.
_CHI3_Q = math.sqrt(8.0249)
_CHI3_MEDIAN = math.sqrt(2.3660)
INLIER_BOUND_PER_MEDIAN = _CHI3_Q / _CHI3_MEDIAN  # = 1.842


def weighted_kabsch(P, Q, w=None):
    if w is None:
        w = torch.ones(P.shape[0], dtype=P.dtype, device=P.device)
    w = w / torch.clamp(w.sum(), min=1e-12)
    cp = (P * w[:, None]).sum(0)
    cq = (Q * w[:, None]).sum(0)
    H = ((P - cp) * w[:, None]).T @ (Q - cq)
    U, _, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vh.T @ U.T))
    D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    R = Vh.T @ D @ U.T
    return R, cq - R @ cp


def _batched_kabsch(P, Q):
    cp, cq = P.mean(1, keepdim=True), Q.mean(1, keepdim=True)
    H = (P - cp).transpose(1, 2) @ (Q - cq)
    U, _, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vh.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vh.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = cq.squeeze(1) - (R @ cp.transpose(1, 2)).squeeze(-1)
    return R, t


def residuals(R, t, P, Q):
    return torch.linalg.norm(P @ R.T + t - Q, dim=1)


def lmeds_rigid(P, Q, trials=500, seed=0, chunk=100):
    """Least-median-of-squares rigid fit (threshold free; needs > 50% inliers)."""
    n = P.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.stack([torch.randperm(n, generator=gen)[:3] for _ in range(trials)]).to(P.device)
    best = (None, None, float("inf"))
    for s in range(0, trials, chunk):
        sel = idx[s:s + chunk]
        R, t = _batched_kabsch(P[sel], Q[sel])
        r = torch.linalg.norm(torch.einsum("bij,nj->bni", R, P) + t[:, None] - Q[None], dim=2)
        med = r.median(dim=1).values
        j = int(torch.argmin(med))
        if float(med[j]) < best[2]:
            best = (R[j], t[j], float(med[j]))
    return best


def part_aware_filter(trajectories, trials=500, rounds=5, seed=0, min_part=3):
    """Return ``(keep, weight, info)`` for ``trajectories`` (N, 2, 3) (first / last window frame).

    ``weight`` is 1 for a trajectory explained by exactly one motion, 1/2 when ambiguous and 0
    when rejected.
    """
    P = trajectories[:, 0, :3].to(torch.float64)
    Q = trajectories[:, -1, :3].to(torch.float64)
    n = P.shape[0]
    info = dict(candidates=int(n), hypotheses=0)
    if n < 2 * min_part:
        ones = torch.ones(n, dtype=torch.bool, device=P.device)
        return ones, ones.to(trajectories.dtype), dict(info, reason="too_few_candidates")

    def bound(r):
        return INLIER_BOUND_PER_MEDIAN * torch.clamp(r.median(), min=1e-9)

    Ra, ta, _ = lmeds_rigid(P, Q, trials=trials, seed=seed)
    ra = residuals(Ra, ta, P, Q)
    rest = ra > bound(ra)          # candidates A does not explain (provisional, never discarded)
    hyps = [(Ra, ta)]
    if int(rest.sum()) >= 2 * min_part:
        Rb, tb, _ = lmeds_rigid(P[rest], Q[rest], trials=trials, seed=seed + 1)
        hyps.append((Rb, tb))
    for _ in range(rounds):
        r = torch.stack([residuals(R, t, P, Q) for R, t in hyps])        # (H, N)
        own = torch.argmin(r, dim=0)
        bounds = []
        for h in range(len(hyps)):
            mine = own == h
            bounds.append(bound(r[h][mine]) if int(mine.sum()) >= min_part else torch.zeros((), dtype=P.dtype, device=P.device))
        bounds = torch.stack(bounds)
        explained = r <= bounds[:, None]                                  # (H, N)
        ambiguous = explained.sum(0) > 1
        new = []
        for h, (R, t) in enumerate(hyps):
            sel = explained[h]
            if int(sel.sum()) < min_part:
                new.append((R, t))
                continue
            w = torch.where(ambiguous[sel], 0.5, 1.0).to(P.dtype)
            new.append(weighted_kabsch(P[sel], Q[sel], w))
        change = max(float(torch.linalg.norm(Rn - R) + torch.linalg.norm(tn - t)) for (R, t), (Rn, tn) in zip(hyps, new))
        hyps = new
        if change < 1e-9:
            break
    r = torch.stack([residuals(R, t, P, Q) for R, t in hyps])
    own = torch.argmin(r, dim=0)
    bounds = torch.stack([bound(r[h][own == h]) if int((own == h).sum()) >= min_part
                          else torch.zeros((), dtype=P.dtype, device=P.device) for h in range(len(hyps))])
    explained = r <= bounds[:, None]
    keep = explained.any(0)
    ambiguous = explained.sum(0) > 1
    weight = torch.where(ambiguous, 0.5, 1.0) * keep
    info.update(hypotheses=len(hyps), kept=int(keep.sum()), ambiguous=int(ambiguous.sum()),
                per_hypothesis=[dict(explained=int(explained[h].sum()), assigned=int((own == h).sum()),
                                     bound_m=float(bounds[h]),
                                     rotation_deg=float(torch.rad2deg(torch.arccos(torch.clamp((torch.trace(R) - 1) / 2, -1, 1)))),
                                     translation_m=float(torch.linalg.norm(t)))
                                for h, (R, t) in enumerate(hyps)])
    return keep, weight.to(trajectories.dtype), info
