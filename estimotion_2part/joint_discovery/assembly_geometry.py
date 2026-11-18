"""Candidate-independent attachment evidence (spec 4.2 / 4.3).

The attachment region is the observed static/moving *interface*: static-core tracks that lie within ``gap_m`` of a
moving-proposal track (and vice versa) at the reference time, i.e. where the part meets the base. It is built from
the candidate-independent support only, never from a fitted hinge.  It must be corroborated by >= ``min_cameras``
calibrated views and >= ``min_clusters`` surface clusters, otherwise every assembly term is ABSTAIN.

* Revolute: a soft prior cost d(hinge line, interface) / sigma_A. Never a bounding-box veto: a hinge may sit in an
  occluded bearing just outside the visible surfaces.  Distances are reported, the verdict is graded.
* Prismatic: a *guide* is positive slider evidence only if the interface is elongated along the slide direction.
  A gap around a drawer front is perpendicular to the motion and is not a guide -> ABSTAIN (honest, conservative).
"""
from __future__ import annotations

import numpy as np

from .blocked_statistics import spatial_clusters
from .motion_field import PointTable, support_principal
from .rigid_kinematics import normalize


def reference_positions(tab: PointTable, mask, ref, window=2):
    """Per-track position at the reference time (or the nearest frame within ``window``)."""
    X = np.full((tab.N, 3), np.nan)
    for dt in [0] + [s * k for k in range(1, window + 1) for s in (1, -1)]:
        t = ref + dt
        if 0 <= t < tab.T:
            fill = np.isnan(X).any(1) & np.isfinite(tab.X[t]).all(1) & mask
            X[fill] = tab.X[t, fill]
    return X


def attachment_evidence(tab: PointTable, support: dict, ref: int, *, gap_m=0.03, voxel_m=0.03, min_points=20,
                        min_clusters=3, min_cameras=2, sigma_calib_m=0.01) -> dict:
    S = reference_positions(tab, support["S_core"], ref)
    M = reference_positions(tab, support["M_proposal"], ref)
    si, mi = np.flatnonzero(np.isfinite(S).all(1)), np.flatnonzero(np.isfinite(M).all(1))
    if len(si) < 3 or len(mi) < 3:
        return dict(status="ABSTAIN", reason="static or moving support not observed at the reference time")
    from scipy.spatial import cKDTree
    ts, tm = cKDTree(S[si]), cKDTree(M[mi])
    ds, _ = tm.query(S[si])
    dm, _ = ts.query(M[mi])
    s_int, m_int = si[ds < gap_m], mi[dm < gap_m]
    P = np.concatenate([S[s_int], M[m_int]])
    tracks = np.concatenate([s_int, m_int])
    cams = np.unique(tab.track_camera[tracks]) if len(tracks) else np.zeros(0)
    ncl = len(np.unique(spatial_clusters(P, voxel_m))) if len(P) else 0
    base = dict(n_points=int(len(P)), n_static_side=int(len(s_int)), n_moving_side=int(len(m_int)),
                n_clusters=int(ncl), n_cameras=int(len(cams)), gap_m=gap_m,
                rule="static-core <-> moving-proposal tracks closer than gap_m at the reference time")
    if len(P) < min_points or ncl < min_clusters or len(cams) < min_cameras:
        return dict(base, status="ABSTAIN",
                    reason=f"interface not independently observable (points {len(P)}/{min_points}, clusters "
                           f"{ncl}/{min_clusters}, cameras {len(cams)}/{min_cameras})")
    sd, V, m = support_principal(P)
    sigma_A = float(np.sqrt(sigma_calib_m ** 2 + (voxel_m / 2) ** 2))
    mov_all = M[mi]
    r_far = float(np.max(np.linalg.norm(mov_all - m, axis=1)))
    obj = np.concatenate([S[si], M[mi]])
    lo, hi = np.percentile(obj, 2, 0), np.percentile(obj, 98, 0)
    return dict(base, status="OK", points=P.round(5).tolist(), centroid=m.tolist(), principal_std_m=sd.tolist(),
                principal_axes=V.T.tolist(), elongation=float(sd[0] / max(sd[1], 1e-6)), sigma_A_m=sigma_A,
                moving_centroid_ref=np.mean(mov_all, 0).tolist(),
                max_interface_to_moving_m=float(np.max(np.linalg.norm(mov_all[:, None] - P[None, ::max(1, len(P) // 200)],
                                                                      axis=2).min(1))),
                interface_to_far_moving_m=r_far, object_aabb=[lo.tolist(), hi.tolist()],
                object_diag_m=float(np.linalg.norm(hi - lo)))


def _line_dist(P, u, o):
    r = P - o
    return np.linalg.norm(r - np.outer(r @ u, u), axis=1)


def hinge_plausibility(evidence: dict, axis, origin, *, k_sigma=3.0) -> dict:
    """Soft assembly term for a hinge line: robust distance to the observed interface in units of sigma_A."""
    if evidence.get("status") != "OK" or axis is None or origin is None:
        return dict(status="ABSTAIN", reason=evidence.get("reason", "no hinge line"))
    P = np.asarray(evidence["points"], float)
    u, o = normalize(np.asarray(axis, float)), np.asarray(origin, float)
    d = _line_dist(P, u, o)
    sA = evidence["sigma_A_m"]
    d10 = float(np.percentile(d, 10))
    frac = float(np.mean(d < k_sigma * sA))
    lo, hi = (np.asarray(x) for x in evidence["object_aabb"])
    cen = (lo + hi) / 2
    cost = float(np.log1p((d10 / sA) ** 2))      # Cauchy-type prior cost, reported, never a hard gate
    verdict = "SUPPORTED" if d10 < k_sigma * sA else ("WEAK" if d10 < 3 * k_sigma * sA else "CONTRADICTED")
    return dict(status="OK", verdict=verdict, d10_to_interface_m=d10, fraction_interface_within_k_sigma=frac,
                sigma_A_m=sA, k_sigma=k_sigma, prior_cost=cost,
                distance_to_object_centre_m=float(_line_dist(cen[None], u, o)[0]),
                object_half_diag_m=float(np.linalg.norm(hi - lo) / 2),
                note="graded prior conditional on the observed interface; not a bounding-box veto")


def guide_evidence(evidence: dict, slide_axis, *, max_angle_deg=15.0, min_elongation=3.0) -> dict:
    if evidence.get("status") != "OK" or slide_axis is None:
        return dict(status="ABSTAIN", reason=evidence.get("reason", "no slide axis"))
    u = normalize(np.asarray(slide_axis, float))
    a0 = np.asarray(evidence["principal_axes"][0])
    ang = float(np.degrees(np.arccos(min(1, abs(a0 @ u)))))
    if evidence["elongation"] >= min_elongation and ang <= max_angle_deg:
        return dict(status="OK", verdict="GUIDE_ALIGNED", angle_deg=ang, elongation=evidence["elongation"])
    return dict(status="ABSTAIN", verdict="NO_OBSERVABLE_GUIDE", angle_deg=ang, elongation=evidence["elongation"],
                reason="observed interface is not an elongated guide along the slide direction (e.g. a gap around a "
                       "drawer front); positive slider evidence is unavailable, not contradicted")


def implied_min_rotation_deg(evidence: dict, travel_m: float) -> dict:
    """Smallest rotation a hinge on the observed interface would need to move the support by ``travel_m``.
    A hinge farther than the farthest moving point would not be on the object; uses the far moving distance."""
    if evidence.get("status") != "OK" or not np.isfinite(travel_m):
        return dict(status="ABSTAIN")
    r = max(evidence["interface_to_far_moving_m"], 1e-3)
    return dict(status="OK", radius_m=r, travel_m=float(travel_m),
                implied_min_rotation_deg=float(np.degrees(2 * np.arcsin(min(1.0, travel_m / (2 * r))))))
