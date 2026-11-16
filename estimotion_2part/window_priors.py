"""Optional per-window prior terms for the two-part transform optimiser.

* :class:`SeedTerm`    (Phase B) - BCE on prompt-supported moving-object seed points.
* :class:`ContactTerm` (Phase C) - sliding-contact distance to the moving surface.

Both resolve which *branch* (T1 or T2; tracking labels are arbitrary) is moving from
observed motion before contributing any gradient, follow label swaps, and report
``skipped`` with a reason instead of guessing.  When a term is ``None`` the optimiser is
byte-for-byte the baseline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from estimotion_2part.contact_surface_loss import (
    BRANCH_T1,
    BRANCH_T2,
    canonical_moving_surface,
    moving_surface_contact_loss,
    nearest_surface_distance,
    resolve_motion_branch,
    select_moving_transform,
)


def flip_branch(branch):
    return {BRANCH_T1: BRANCH_T2, BRANCH_T2: BRANCH_T1}.get(branch, branch)


def resolve_seed_branch(source_points, target_points, T1_per_sample, T2_per_sample, seed_mask, min_seeds):
    """Which branch explains the motion of prompt-supported seed tracks.

    The seeds are moving-object pixels; the moving branch must both fit their observed
    displacement better and move them more than the other branch.  If the two criteria
    disagree (or tie) the identity is ambiguous and ``None`` is returned: a point prompt
    never inverts static/moving labels by fiat.
    """
    seed_mask = seed_mask.to(device=source_points.device, dtype=torch.bool)
    n = int(seed_mask.sum())
    if n < max(1, int(min_seeds)):
        return None, dict(reason="insufficient_seed_tracks", seed_tracks=n)
    with torch.no_grad():
        x = source_points[seed_mask]
        y = target_points[seed_mask]
        ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        xh = torch.cat([x, ones], dim=1).unsqueeze(-1)
        p1 = torch.bmm(T1_per_sample[seed_mask], xh).squeeze(-1)[:, :3]
        p2 = torch.bmm(T2_per_sample[seed_mask], xh).squeeze(-1)[:, :3]
        r1 = float(torch.linalg.norm(p1 - y, dim=1).median())
        r2 = float(torch.linalg.norm(p2 - y, dim=1).median())
        m1 = float(torch.linalg.norm(p1 - x, dim=1).median())
        m2 = float(torch.linalg.norm(p2 - x, dim=1).median())
    info = dict(seed_tracks=n, residual_m={BRANCH_T1: r1, BRANCH_T2: r2}, motion_m={BRANCH_T1: m1, BRANCH_T2: m2})
    if r1 == r2 or m1 == m2:
        info["reason"] = "tie"
        return None, info
    by_fit = BRANCH_T1 if r1 < r2 else BRANCH_T2
    by_motion = BRANCH_T1 if m1 > m2 else BRANCH_T2
    if by_fit != by_motion:
        info["reason"] = "fit_and_motion_disagree"
        return None, info
    info["reason"] = "resolved"
    return by_fit, info


def moving_seed_bce(weight_logits, seed_point_idx, moving_branch):
    """L_seed = -mean log p_m at positive seeds; p_m = sigmoid(w) for T2, 1 - sigmoid(w) for T1."""
    logits = weight_logits.reshape(-1)[seed_point_idx]
    signed = logits if moving_branch == BRANCH_T2 else -logits
    return F.binary_cross_entropy_with_logits(signed, torch.ones_like(signed))


class SeedTerm:
    """Phase B moving-part seed loss on per-point positive seeds (robot pixels never included)."""

    def __init__(self, seed_mask, weight, min_seeds=1, inherited_branch=None):
        if not weight or weight <= 0:
            raise ValueError("moving_prompt_weight must be explicitly configured > 0 when Phase B is enabled")
        self.seed_mask = seed_mask.to(dtype=torch.bool)
        self.weight = float(weight)
        self.min_seeds = int(min_seeds)
        self.branch = inherited_branch
        self.info = dict(status="pending", seed_points=int(self.seed_mask.sum()), seed_weight=self.weight)

    def resolve(self, source_points, target_points, T1_per_sample, T2_per_sample, sample_point_idx=None):
        """Resolve on per-sample tracks; ``sample_point_idx`` maps samples to weight rows."""
        mask = self.seed_mask.to(source_points.device)
        sample_mask = mask if sample_point_idx is None else mask[sample_point_idx]
        branch, info = resolve_seed_branch(source_points, target_points, T1_per_sample, T2_per_sample,
                                           sample_mask, self.min_seeds)
        self.branch = branch
        self.info.update(info)
        self.info["status"] = "active" if branch is not None else "skipped"
        self.info["moving_branch"] = branch
        return branch

    def loss(self, weight_logits):
        if self.branch is None:
            return None
        idx = torch.nonzero(self.seed_mask.to(weight_logits.device), as_tuple=False).flatten()
        return self.weight * moving_seed_bce(weight_logits, idx, self.branch)

    def swap_labels(self):
        self.branch = flip_branch(self.branch)
        self.info["moving_branch"] = self.branch
        self.info["label_swap_applied"] = True


class ContactTerm:
    """Phase C sliding-contact term.

    ``contact_xyz[k]`` is the calibrated contact point (camera metres) at the frame mapped by
    transform row ``k``; ``valid[k]`` says contact is confirmed there.
    """

    def __init__(self, contact_xyz, valid, *, weight, sigma_m, robust_delta_m, support_max_distance_m,
                 inherited_branch=None, min_depth=None, max_depth=None, tool_axis=None, tool_radius_m=None):
        for name, value in (("contact_surface_weight", weight), ("contact_sigma_m", sigma_m),
                            ("contact_robust_delta_m", robust_delta_m),
                            ("contact_support_max_distance_m", support_max_distance_m)):
            if value is None or not value > 0:
                raise ValueError(f"{name} must be explicitly configured > 0 when Phase C is enabled")
        self.contact_xyz = contact_xyz
        self.valid = valid.to(dtype=torch.bool)
        self.weight = float(weight)
        self.sigma_m = float(sigma_m)
        self.robust_delta_m = float(robust_delta_m)
        self.support_max = float(support_max_distance_m)
        self.inherited_branch = inherited_branch
        self.branch = None
        self.canonical = None
        self.supported = None
        self.min_depth, self.max_depth = min_depth, max_depth
        if tool_radius_m is not None and not tool_radius_m > 0:
            raise ValueError("contact_exclude_tool_radius_m must be > 0 when set")
        if tool_radius_m is not None and tool_axis is None:
            raise ValueError("contact_exclude_tool_radius_m needs the tool axis (TCP artifact with T_base_ee)")
        self.tool_axis, self.tool_radius_m = tool_axis, tool_radius_m
        self.info = dict(status="pending", contact_samples_confirmed=int(self.valid.sum()))

    def prepare(self, T1, T2, source_points, weights):
        """Resolve the moving branch, freeze canonical support and gate unsupported samples."""
        device = T1.device
        self.contact_xyz = self.contact_xyz.to(device=device, dtype=T1.dtype)
        self.valid = self.valid.to(device)
        if not bool(self.valid.any()):
            self.info.update(status="skipped", reason="no_confirmed_contact_in_window")
            return None
        if self.inherited_branch is not None:
            branch, why = self.inherited_branch, dict(reason="inherited_from_phase_b")
        else:
            branch, why = resolve_motion_branch(T1, T2, source_points, weights, self.contact_xyz, self.valid)
        self.info["branch_resolution"] = why
        if branch is None:
            self.info.update(status="skipped", reason=why.get("reason", "ambiguous_branch"))
            return None
        self.branch = branch
        self.rebuild(source_points, weights)
        with torch.no_grad():
            T = select_moving_transform(T1, T2, branch)
            if self.canonical.shape[0] == 0:
                self.info.update(status="skipped", reason="no_canonical_moving_support")
                self.branch = None
                return None
            dist = torch.full((self.valid.shape[0],), float("inf"), device=device, dtype=T.dtype)
            idx = torch.nonzero(self.valid, as_tuple=False).flatten()
            d, _ = nearest_surface_distance(self.canonical, T[idx], self.contact_xyz[idx])
            dist[idx] = d
            self.supported = self.valid & torch.isfinite(dist) & (dist <= self.support_max)
        self.info.update(status="active" if bool(self.supported.any()) else "skipped",
                         moving_branch=branch, N_contact=int(self.supported.sum()),
                         rejected_unsupported=int((self.valid & ~self.supported).sum()),
                         distance_before_m=[float(v) for v in dist[idx].tolist()])
        if not bool(self.supported.any()):
            self.info["reason"] = "contact_surface_not_supported_by_observed_moving_geometry"
            self.branch = None
        return self.branch

    def rebuild(self, source_points, weights):
        """Refresh frozen canonical support (called only at outer iterations)."""
        if self.branch is None:
            return
        self.canonical = canonical_moving_surface(source_points, weights, self.branch,
                                                  self.min_depth, self.max_depth).to(self.contact_xyz.dtype)
        if self.tool_radius_m is not None and self.canonical.shape[0]:
            # Robot geometry is not object surface: reject support points inside the tool cylinder
            # (radius around the calibrated tip -> EE axis, from the tip backwards; nothing in front of it).
            from samuele.contact_moving_part_prompts import tool_volume_mask
            inside = tool_volume_mask(self.canonical.detach().cpu().numpy().astype("float64"),
                                      self.tool_axis[0], self.tool_axis[1], self.tool_radius_m, 0.0)
            keep = torch.as_tensor(~inside, device=self.canonical.device)
            self.info["canonical_tool_points_rejected"] = int((~keep).sum())
            self.canonical = self.canonical[keep]

    def loss(self, T1, T2):
        if self.branch is None or self.supported is None:
            return None, dict(status="skipped")
        T = select_moving_transform(T1, T2, self.branch)
        loss, info = moving_surface_contact_loss(
            canonical_moving_xyz=self.canonical, moving_T_ref_to_t=T, tool_contact_camera_xyz=self.contact_xyz,
            valid_contact_t=self.supported, sigma_m=self.sigma_m, robust_delta_m=self.robust_delta_m)
        return self.weight * loss, info

    def finalize(self, T1, T2):
        if self.branch is None:
            return self.info
        with torch.no_grad():
            _, info = self.loss(T1, T2)
        unweighted = info.get("L_contact")
        self.info.update(distance_after_m=info.get("distances_m", []), L_contact_final=unweighted,
                         L_contact_weighted_final=None if unweighted is None else self.weight * unweighted,
                         contact_weight=self.weight, N_contact_final=info.get("N_contact"),
                         distinct_contact_samples=info.get("distinct_contact_samples"),
                         sigma_contact_m=self.sigma_m, robust_delta_m=self.robust_delta_m)
        return self.info

    def swap_labels(self):
        self.branch = flip_branch(self.branch)
        self.info["moving_branch"] = self.branch
        self.info["label_swap_applied"] = True
