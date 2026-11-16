"""Phase C: sliding-contact constraint between a robot contact point and the moving surface.

    L_contact = mean_{t in C_valid} rho( d(p_contact(t), T_m(t) S_m^0)^2 / sigma(t)^2 )
    d(p, T S)^2 = min_{x in S} || p - T x ||^2           (min recomputed at every t)

There is no fixed material contact point: the minimising canonical sample is chosen
independently at each time, so a straight TCP path may slide over a rotating surface.
Inputs are camera-frame metres (the frame of the tracked 3D points), ``T_m(t)`` maps
reference-frame (window frame 0) points to frame ``t`` and is the single selected moving
transform, never the soft blend of T1/T2.
"""
from __future__ import annotations

import numpy as np
import torch

BRANCH_T1 = 1
BRANCH_T2 = 2


def _huber(e, delta):
    e = e.abs()
    return torch.where(e <= delta, 0.5 * e * e, delta * (e - 0.5 * delta))


def nearest_surface_distance(canonical_xyz, T, points, chunk=8192):
    """Distances from ``points`` (K,3) to ``T[k] @ canonical`` (K transforms) and argmin ids.

    Differentiable w.r.t. ``T`` and ``points`` through the selected nearest sample.
    """
    K = points.shape[0]
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    best_sq = torch.full((K,), float("inf"), dtype=points.dtype, device=points.device)
    best_idx = torch.zeros((K,), dtype=torch.long, device=points.device)
    for start in range(0, canonical_xyz.shape[0], chunk):
        block = canonical_xyz[start:start + chunk]
        moved = torch.einsum("kij,mj->kmi", R, block) + t[:, None, :]  # (K, M, 3)
        sq = ((moved - points[:, None, :]) ** 2).sum(dim=-1)
        block_sq, block_idx = sq.min(dim=1)
        better = block_sq < best_sq
        best_sq = torch.where(better, block_sq, best_sq)
        best_idx = torch.where(better, block_idx + start, best_idx)
    # Recompute the selected squared distance so gradients flow only through the argmin sample.
    selected = canonical_xyz[best_idx]
    moved_sel = torch.einsum("kij,kj->ki", R, selected) + t
    dist = torch.linalg.norm(moved_sel - points, dim=-1)
    return dist, best_idx


def moving_surface_contact_loss(*, canonical_moving_xyz, moving_T_ref_to_t, tool_contact_camera_xyz,
                                valid_contact_t, sigma_m, robust_delta_m):
    """Robust, uncertainty-scaled point-to-moving-surface distance with per-time correspondence.

    Args:
        canonical_moving_xyz: (M,3) reference-frame moving-surface samples (camera metres).
        moving_T_ref_to_t: (K,4,4) selected moving transform per contact time.
        tool_contact_camera_xyz: (K,3) calibrated contact point per time (camera metres).
        valid_contact_t: (K,) bool, contact confirmed and supported.
        sigma_m: positive float or (K,) uncertainty in metres.
        robust_delta_m: positive Huber transition in metres.

    Returns (loss, info).  With no valid sample, loss is an exact zero connected to the
    transforms (zero gradient) and ``info['status'] == 'skipped'``.
    """
    T = moving_T_ref_to_t
    device, dtype = T.device, T.dtype
    canonical = torch.as_tensor(canonical_moving_xyz, dtype=dtype, device=device).reshape(-1, 3)
    points = torch.as_tensor(tool_contact_camera_xyz, dtype=dtype, device=device).reshape(-1, 3)
    valid = torch.as_tensor(valid_contact_t, dtype=torch.bool, device=device).reshape(-1)
    if robust_delta_m is None or not (np.isfinite(robust_delta_m) and robust_delta_m > 0):
        raise ValueError("contact_robust_delta_m must be explicitly configured and positive")
    if sigma_m is None:
        raise ValueError("contact_sigma_m must be explicitly configured, finite and positive")
    sigma = torch.as_tensor(sigma_m, dtype=dtype, device=device)
    if sigma.ndim == 0:
        sigma = sigma.expand(points.shape[0])
    if not torch.all(torch.isfinite(sigma) & (sigma > 0)):
        raise ValueError("contact_sigma_m must be explicitly configured, finite and positive")
    finite_canonical = torch.isfinite(canonical).all(dim=1)
    rejected_canonical = int((~finite_canonical).sum())
    canonical = canonical[finite_canonical]
    finite_points = torch.isfinite(points).all(dim=1) & torch.isfinite(T.reshape(T.shape[0], -1)).all(dim=1)
    use = valid & finite_points
    info = dict(N_contact=int(use.sum()), rejected_nonfinite_contact=int((valid & ~finite_points).sum()),
                rejected_nonfinite_canonical=rejected_canonical, canonical_samples=int(canonical.shape[0]),
                sigma_contact_m=float(sigma[use].mean()) if bool(use.any()) else None)
    zero = (T * 0.0).sum()
    if canonical.shape[0] == 0 or not bool(use.any()):
        info.update(status="skipped", L_contact=0.0, distances_m=[], argmin_canonical=[],
                    reason="no canonical moving support" if canonical.shape[0] == 0 else "no valid contact sample")
        return zero, info
    dist, argmin = nearest_surface_distance(canonical, T[use], points[use])
    e = dist / sigma[use]
    loss = _huber(e, float(robust_delta_m) / sigma[use]).mean()
    d = dist.detach().cpu().numpy()
    info.update(status="available", L_contact=float(loss.detach()), distances_m=d.tolist(),
                median_m=float(np.median(d)), p95_m=float(np.percentile(d, 95)),
                argmin_canonical=argmin.detach().cpu().tolist(),
                distinct_contact_samples=int(len(set(argmin.detach().cpu().tolist()))))
    return loss, info


def select_moving_transform(T1, T2, moving_branch):
    """The single rigid moving transform of a resolved branch (never the soft blend)."""
    if moving_branch == BRANCH_T1:
        return T1
    if moving_branch == BRANCH_T2:
        return T2
    raise ValueError(f"Unresolved moving branch {moving_branch!r}")


def branch_assignment(weights, moving_branch):
    """Hard per-point assignment to the moving branch using the existing 0.5 split."""
    weights = torch.as_tensor(weights).reshape(-1)
    return weights > 0.5 if moving_branch == BRANCH_T2 else weights < 0.5


def canonical_moving_surface(source_points, weights, moving_branch, min_depth=None, max_depth=None):
    """Frozen reference-frame moving-surface samples from tracked RGB-D source points.

    Only finite points with positive depth (inside the optional depth range) that are hard
    assigned to the moving branch are kept; robot pixels never enter tracked object points.
    """
    points = torch.as_tensor(source_points).reshape(-1, 3)
    keep = branch_assignment(weights, moving_branch).to(points.device)
    keep &= torch.isfinite(points).all(dim=1) & (points[:, 2] > 0)
    if min_depth is not None:
        keep &= points[:, 2] >= float(min_depth)
    if max_depth is not None:
        keep &= points[:, 2] <= float(max_depth)
    return points[keep].detach().clone()


def transform_motion_magnitude(T, points):
    """Median displacement of ``points`` under each transform in T (K,4,4) -> float."""
    if points.shape[0] == 0:
        return 0.0
    moved = torch.einsum("kij,mj->kmi", T[:, :3, :3], points) + T[:, None, :3, 3]
    return float(torch.linalg.norm(moved - points[None], dim=-1).median())


def resolve_motion_branch(T1, T2, source_points, weights, contact_points=None, contact_valid=None):
    """Moving branch from motion, cross-checked against contact geometry; None if ambiguous.

    The branch whose own points move more is the candidate.  When contact samples exist,
    its moving surface must also be at least as close to the contact points as the other
    branch's surface; otherwise the evidence disagrees and the window is skipped.
    """
    with torch.no_grad():
        motions = {}
        for branch, T in ((BRANCH_T1, T1), (BRANCH_T2, T2)):
            pts = canonical_moving_surface(source_points, weights, branch)
            motions[branch] = transform_motion_magnitude(T, pts)
        if motions[BRANCH_T1] == motions[BRANCH_T2]:
            return None, dict(reason="equal_branch_motion", motion_m=motions)
        candidate = BRANCH_T1 if motions[BRANCH_T1] > motions[BRANCH_T2] else BRANCH_T2
        info = dict(motion_m=motions, candidate=candidate)
        if contact_points is not None and contact_valid is not None and bool(torch.as_tensor(contact_valid).any()):
            valid = torch.as_tensor(contact_valid, dtype=torch.bool, device=T1.device)
            cp = torch.as_tensor(contact_points, dtype=T1.dtype, device=T1.device)
            medians = {}
            for branch, T in ((BRANCH_T1, T1), (BRANCH_T2, T2)):
                pts = canonical_moving_surface(source_points, weights, branch)
                if pts.shape[0] == 0:
                    medians[branch] = float("inf")
                    continue
                d, _ = nearest_surface_distance(pts.to(T.dtype), T[valid], cp[valid])
                medians[branch] = float(d.median())
            info["contact_median_m"] = medians
            other = BRANCH_T2 if candidate == BRANCH_T1 else BRANCH_T1
            if medians[candidate] > medians[other]:
                info["reason"] = "motion_and_contact_evidence_disagree"
                return None, info
        info["reason"] = "resolved_by_motion"
        return candidate, info
