"""Tentative mass geometry of the moving part from the FROZEN reconstruction (G9 PINN/EKF spec section 1A.A).

No validated watertight mesh exists for the real objects, so a separately labelled TENTATIVE_GEOMETRIC_PREVIEW is
built from the moving-part Gaussian MEANS of the G8 visual handoff (the frozen final PLY, never modified):
  1. points in the capture world W: x_W = T_Wc_P x_P (the handoff's frame), the canonical q = 0 pose;
  2. robust surface-point filtering: opacity sigmoid >= 0.5 (if stored) and a k-NN density test (drop points whose
     8-NN distance > 3x the median) -- isolated floaters are not mass;
  3. a CONVEX HULL as the closed surface (watertight by construction), integrated as
       solid_hull : uniform volumetric density (kg/m^3)        -- a filled body bounded by the hull
       shell_hull : uniform areal density (kg/m^2) on the hull -- a hollow shell
     (two explicit, alternative mass-distribution hypotheses; their spread is a geometry SCENARIO).
Gaussian covariance / opacity is never used as density, and the COM is the integrated hull centroid, never the
naive mean of the Gaussian centres. Limitations (recorded): visible surfaces only, a hull fills concavities, and
the real part may be hollow / heterogeneous -- this is geometry-ASSUMED, not measured.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .mass_properties import TriMesh

MODES = {"solid_hull": "uniform_solid", "shell_hull": "uniform_shell"}


def load_moving_points(handoff_json) -> tuple:
    from plyfile import PlyData
    h = json.loads(Path(handoff_json).read_text())
    part = h["parts"]["moving"]
    ply = Path(handoff_json).parent / part["ply"]
    v = PlyData.read(str(ply))["vertex"].data
    P = np.stack([np.asarray(v[k], float) for k in ("x", "y", "z")], 1)
    T = np.asarray(part["T_Wc_P"], float)
    X = P @ T[:3, :3].T + T[:3, 3]
    op = None
    if "opacity" in v.dtype.names:
        op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], float)))
    return X, op, dict(ply=str(ply), ply_sha256=part.get("ply_sha256"), n_gaussians=int(len(X)),
                       frame="capture world W (T_Wc_P applied)", anchor_frame=h["source"].get(
                           "canonical_anchor_original_frame"))


class ProxyGeometryError(ValueError):
    """The moving part has no usable geometry for a tentative proxy (recorded, never a crash)."""


def filter_points(X, opacity=None, k=8, factor=3.0):
    from scipy.spatial import cKDTree
    keep = np.ones(len(X), bool)
    if opacity is not None:
        keep &= opacity >= 0.5
    Y = X[keep]
    if len(Y) < 4:                                      # a hull needs >= 4 points
        mx = f"{float(np.max(opacity)):.3f}" if opacity is not None and len(opacity) else "n/a"
        raise ProxyGeometryError(f"{int(keep.sum())} of {len(X)} moving Gaussians pass opacity >= 0.5 (max opacity "
                                 f"{mx}): no geometry for a tentative proxy (e.g. a very short 3DGS training)")
    d, _ = cKDTree(Y).query(Y, k=min(k + 1, len(Y)))
    dk = d[:, -1]
    ok = dk <= factor * np.median(dk)
    idx = np.flatnonzero(keep)[ok]
    return X[idx], dict(n_in=int(len(X)), n_opacity=int(keep.sum()), n_kept=int(len(idx)), knn_k=k,
                        knn_factor=factor)


def hull_mesh(X) -> TriMesh:
    from scipy.spatial import ConvexHull
    hull = ConvexHull(X)
    V = X[hull.vertices]
    remap = {int(v): i for i, v in enumerate(hull.vertices)}
    F = np.array([[remap[int(a)] for a in s] for s in hull.simplices], np.int64)
    c = V.mean(0)
    for i, f in enumerate(F):                                    # orient every face outward
        n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        if n @ (V[f[0]] - c) < 0:
            F[i] = f[[0, 2, 1]]
    return TriMesh.from_arrays(V, F, source="convex hull of filtered moving-part Gaussian means")


def build_proxy(handoff_json, kind: str, out_dir=None) -> dict:
    if kind not in MODES:
        raise ValueError(f"tentative proxy {kind!r} not in {list(MODES)}")
    X, op, prov = load_moving_points(handoff_json)
    Xf, frep = filter_points(X, op)
    mesh = hull_mesh(Xf)
    mode = MODES[kind]
    p = dict(prov, filtering=frep, proxy=kind, mass_mode=mode, status="TENTATIVE_GEOMETRIC_PREVIEW",
             source=f"tentative {kind} from the moving-part Gaussians of {Path(handoff_json).parent.name}",
             limitations=["visible surfaces only", "a convex hull fills concavities / openings",
                          "true part may be hollow or heterogeneous: density is an assumption",
                          "Gaussian covariance / opacity is NOT used as density; COM = integrated hull centroid"],
             mesh_sha256=mesh.sha256())
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        mesh.write_obj(out / f"tentative_{kind}.obj", [p["source"], p["status"]])
        p["obj"] = str(out / f"tentative_{kind}.obj")
    return dict(mesh=mesh, mode=mode, provenance=p, points=Xf)
