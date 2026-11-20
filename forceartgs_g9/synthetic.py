"""Synthetic, fully instrumented 1-DoF objects with KNOWN physics (tests only; truth never enters a real fit).

``make_object`` builds a moving part (watertight box = uniform_solid, or open rectangular panel = uniform_shell)
on a revolute hinge or a prismatic slider, a known density, damping, Coulomb / static friction and gravity, and
``episode`` drives it with a force profile (pushes, pauses, reversals) applied at a material point, simulated with
a fine step, observed at camera rate with seeded noise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .dynamics.general_1dof import (STANDARD_GRAVITY, GeneralTerms, GravityField, rotation, sim_params,
                                    simulate_general)
from .dynamics.geometry_inertia import PASSIVE, GeometryMassModel
from .dynamics.mass_properties import TriMesh
from .estimators.model_spec import G9Interaction

BOX_FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def box_mesh(lo, hi) -> TriMesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return TriMesh.from_arrays(V, BOX_FACES.copy(), source="synthetic box")


def subdivided_box(lo, hi, n=4) -> TriMesh:
    """Watertight box with n x n quads per face (for mesh-noise tests)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    verts, faces, index = [], [], {}

    def vid(p):
        k = tuple(np.round(p, 12))
        if k not in index:
            index[k] = len(verts)
            verts.append(p)
        return index[k]
    for ax in range(3):
        for side in (0, 1):
            u, w = [a for a in range(3) if a != ax]
            for i in range(n):
                for j in range(n):
                    def P(ii, jj):
                        p = np.empty(3)
                        p[ax] = hi[ax] if side else lo[ax]
                        p[u] = lo[u] + (hi[u] - lo[u]) * ii / n
                        p[w] = lo[w] + (hi[w] - lo[w]) * jj / n
                        return vid(p)
                    a, b, c, d = P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)
                    tri = [[a, b, c], [a, c, d]]
                    # outward orientation: normal along +ax for side 1, -ax for side 0
                    nrm = np.cross(np.array(verts[b]) - verts[a], np.array(verts[c]) - verts[a])
                    sgn = 1 if side else -1
                    if np.sign(nrm[ax]) != sgn:
                        tri = [[a, c, b], [a, d, c]]
                    faces += tri
    return TriMesh.from_arrays(np.array(verts), np.array(faces), source=f"synthetic subdivided box n={n}")


def panel_mesh(lo, hi) -> TriMesh:
    """Open rectangular plate z = lo[2] (two triangles): a valid shell, NOT a solid."""
    x0, y0, z = lo
    x1, y1 = hi[0], hi[1]
    V = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)
    return TriMesh.from_arrays(V, np.array([[0, 1, 2], [0, 2, 3]]), source="synthetic open panel")


@dataclass
class SyntheticObject:
    joint_type: str
    axis: np.ndarray
    origin: np.ndarray
    mesh: TriMesh
    mode: str
    density: float
    c: float
    Fc: float
    Fs: float
    b: float
    gravity: GravityField
    terms: GeneralTerms
    mass_model: GeometryMassModel
    contact_point0: np.ndarray            # material application point at q = 0
    meta: dict = field(default_factory=dict)

    @property
    def M_true(self) -> float:
        return self.density * self.mass_model.M1

    def G_true(self):
        if not self.terms.gravity:
            return None
        from .estimators.model_spec import fast_gravity
        A, B = self.mass_model.gravity_unit(self.gravity.g_world).basis_coefficients()
        return fast_gravity(self.joint_type, self.density * A, self.density * B)

    def sim(self):
        t = self.terms
        return sim_params(t, self.joint_type, M=self.M_true, c=self.c if t.viscous else None,
                          Fc=self.Fc if t.coulomb else None, Fs=self.Fs if t.static else None,
                          b=self.b if t.bias else None, G=self.G_true())

    def contact_point(self, q) -> np.ndarray:
        if self.joint_type == "prismatic":
            return self.contact_point0 + q * self.axis
        return self.origin + rotation(self.axis, q) @ (self.contact_point0 - self.origin)


def make_object(kind="revolute", *, axis=None, mode="uniform_solid", density=None, c=None, Fc=None, Fs=None, b=0.0,
                terms: GeneralTerms | None = None, gravity_verified=True, g=(0, 0, -STANDARD_GRAVITY),
                hinge="horizontal") -> SyntheticObject:
    terms = terms or GeneralTerms()
    if kind == "revolute":
        if axis is None:
            axis = [1.0, 0.0, 0.0] if hinge == "horizontal" else [0.0, 0.0, 1.0]
        origin = np.array([0.0, 0.0, 0.5])
        if hinge == "horizontal":                       # flap hanging from a horizontal hinge (stable at q = 0)
            lo, hi = [-0.25, -0.01, 0.1], [0.25, 0.01, 0.5]
            cp = np.array([0.0, 0.01, 0.15])
        else:                                           # vertical door, panel along +y
            lo, hi = [-0.01, 0.0, 0.2], [0.01, 0.5, 0.9]
            cp = np.array([0.01, 0.45, 0.55])
        density = density if density is not None else (700.0 if mode == "uniform_solid" else 8.0)
        c = 0.15 if c is None else c
        Fc = 0.25 if Fc is None else Fc
        Fs = 0.45 if Fs is None else Fs
    else:
        axis = [0.0, 1.0, 0.0] if axis is None else axis
        origin = np.zeros(3)
        lo, hi = [-0.2, -0.3, 0.3], [0.2, 0.0, 0.45]
        cp = np.array([0.0, 0.0, 0.40])
        density = density if density is not None else (300.0 if mode == "uniform_solid" else 6.0)
        c = 4.0 if c is None else c
        Fc = 1.5 if Fc is None else Fc
        Fs = 2.5 if Fs is None else Fs
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    mesh = box_mesh(lo, hi) if mode == "uniform_solid" else panel_mesh(lo, hi)
    mm = GeometryMassModel.from_validated_mesh(mesh, mode, kind, a, origin if kind == "revolute" else None,
                                               source_provenance=dict(source="synthetic ground truth mesh"),
                                               reflected_actuator_inertia=PASSIVE)
    grav = GravityField(tuple(g), gravity_verified, "synthetic: exact world gravity" if gravity_verified else
                        "synthetic: UNVERIFIED")
    return SyntheticObject(kind, a, origin, mesh, mode, density, c, Fc, Fs, b, grav, terms, mm, cp)


def profile(segments, rate_hz=100.0):
    """[(duration_s, level) | (duration_s, start, end), ...] -> ZOH effort samples (t, Q) at ``rate_hz``
    (a 3-tuple is a linear ramp, sampled like a force sensor)."""
    t, Q, now = [], [], 0.0
    for seg in segments:
        dur, a = seg[0], seg[1]
        z = seg[2] if len(seg) > 2 else a
        n = max(int(round(dur * rate_hz)), 1)
        t += list(now + np.arange(n) / rate_hz)
        Q += list(a + (z - a) * np.arange(n) / n)
        now += n / rate_hz
    return np.asarray(t), np.asarray(Q, float)


class ZOH:
    """Minimal forward-ZOH effort (same contract as forceartgs_physics GeneralizedEffort)."""

    def __init__(self, t_s, levels, model="synthetic"):
        self.t_s, self.levels, self.model = np.asarray(t_s, float), np.asarray(levels, float), model

    def at(self, t):
        idx = np.clip(np.searchsorted(self.t_s, np.asarray(t, float), side="right") - 1, 0, len(self.t_s) - 1)
        return self.levels[idx]


def default_profile(obj: SyntheticObject, variant=0):
    """Ramped pushes from rest (breakaway time pins Fs), holds, releases (free swing / coast to a stop),
    reversals; amplitudes relative to the breakaway level, different per variant (independent excitation)."""
    s = obj.Fs if (obj.terms.static and obj.Fs > 0) else obj.Fc
    s = max(s, 0.25 * obj.c * (1.0 if obj.joint_type == "prismatic" else 3.0))     # frictionless objects too
    k = [1.0, 1.35, 0.8, 1.15, 0.9, 1.25][variant % 6]
    if obj.joint_type == "revolute":
        A = k * 3.0 * s
        return [(0.3, 0.0), (0.8, 0.0, A), (0.5, A), (0.9, 0.0), (0.7, 0.0, -0.8 * A), (0.4, -0.8 * A),
                (1.0, 0.0), (0.5, 0.0, 1.4 * A), (0.8, 0.0)]
    A = k * 2.2 * s
    return [(0.3, 0.0), (0.6, 0.0, A), (0.5, A), (0.5, 0.0), (0.6, 0.0, -A), (0.4, -1.2 * A), (0.6, 0.0),
            (0.4, 0.0, 1.5 * A), (0.6, 0.0)]


def episode(obj: SyntheticObject, segments=None, *, variant=0, cam_hz=30.0, q_noise=None, force_noise=0.0, seed=0,
            q0=0.0, fine_step=2.5e-4, name=None, declared_initial=False) -> tuple:
    """Simulate one demonstration. Returns (G9Interaction, truth dict)."""
    rng = np.random.default_rng(seed)
    segs = segments or default_profile(obj, variant)
    tf, Q = profile(segs)
    eff_true = ZOH(tf, Q, "synthetic exact generalized effort")
    T = tf[-1] + 0.01
    t_obs = np.arange(0.0, T, 1.0 / cam_hz)
    roll = simulate_general(obj.sim(), eff_true, t_obs, t0=0.0, q0=q0, v0=0.0, max_step=fine_step,
                            record_events=True)
    qn = q_noise if q_noise is not None else (2e-3 if obj.joint_type == "revolute" else 5e-4)
    q_obs = roll.q + qn * rng.standard_normal(len(t_obs))
    Qm = Q + force_noise * rng.standard_normal(len(Q))
    eff_meas = ZOH(tf, Qm, "synthetic measured generalized effort")
    init = dict(t_s=0.0, q=q0, qdot=0.0, source="declared: at rest before the push") if declared_initial else None
    it = G9Interaction(name or f"{obj.joint_type}_ep{variant}_s{seed}", obj.joint_type, t_obs, q_obs, qn, eff_meas,
                       init, dict(synthetic=True, frontend_force_used=False))
    truth = dict(q=roll.q, qdot=roll.qdot, stuck=roll.stuck, events=roll.events, M=obj.M_true, c=obj.c, Fc=obj.Fc,
                 Fs=obj.Fs, b=obj.b, alpha=obj.density)
    return it, truth


def force_record_for(obj: SyntheticObject, t, Q, q_traj_t, q_traj, frame="W"):
    """A physical point force on the object whose generalized effort equals Q (for projection tests): applied at the
    moving material contact point, along the motion direction J = a x (p - o) (R) or a (P)."""
    q = np.interp(t, q_traj_t, q_traj)
    P = np.array([obj.contact_point(x) for x in q])
    if obj.joint_type == "prismatic":
        d = np.tile(obj.axis, (len(t), 1))
        F = Q[:, None] * d
    else:
        Jv = np.cross(obj.axis, P - obj.origin)
        n2 = np.sum(Jv * Jv, 1)
        F = (Q / n2)[:, None] * Jv
    return P, F


# ----------------------------------------------------------------------------------------------- visual handoff
def _surface_samples(mesh: TriMesh, n, rng):
    v = mesh.vertices[mesh.faces]
    area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    f = rng.choice(len(area), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return (1 - s)[:, None] * v[f, 0] + (s * (1 - r2))[:, None] * v[f, 1] + (s * r2)[:, None] * v[f, 2]


def _write_gaussians(path, xyz, rgb, rng):
    from plyfile import PlyData, PlyElement
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
              "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    a = np.zeros(len(xyz), dtype=[(n, "f4") for n in names])
    a["x"], a["y"], a["z"] = xyz.T
    for k in range(3):
        a[f"f_dc_{k}"] = (rgb[k] - 0.5) / 0.28209479177387814
        a[f"scale_{k}"] = np.log(0.004)
    a["opacity"] = 3.0
    q = rng.standard_normal((len(xyz), 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for k in range(4):
        a[f"rot_{k}"] = q[:, k]
    PlyData([PlyElement.describe(a, "vertex")]).write(str(path))


def write_visual_handoff(obj: SyntheticObject, out_dir, *, static_mesh: TriMesh, q_traj, frames=None, t_s=None,
                         n_gauss=(1500, 1500), seed=0, object_id=None):
    """Two standard Graphdeco PLYs (static base + moving part at q = 0) in the capture world (T_Wc_P = I), the frame
    map and lineage sidecar, articulated3dgs.handoff/v1 exactly as tools/export_parttwin_handoff.py writes it."""
    import hashlib
    import json
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    xs = _surface_samples(static_mesh, n_gauss[0], rng)
    xm = _surface_samples(obj.mesh, n_gauss[1], rng)
    _write_gaussians(out / "static_canonical.ply", xs, (0.6, 0.6, 0.6), rng)
    _write_gaussians(out / "moving_canonical.ply", xm, (0.8, 0.3, 0.2), rng)
    ns, nm = len(xs), len(xm)
    np.savez(out / "gaussian_lineage.npz", static_gaussian_uid=np.arange(ns), moving_gaussian_uid=ns + np.arange(nm),
             static_parent_uid=np.full(ns, -1), moving_parent_uid=np.full(nm, -1),
             static_unresolved=np.zeros(ns, np.uint8), moving_unresolved=np.zeros(nm, np.uint8))
    q = np.asarray(q_traj, float)
    frames = np.arange(len(q)) if frames is None else np.asarray(frames)
    t_s = np.arange(len(q)) / 30.0 if t_s is None else np.asarray(t_s)
    k0 = int(np.argmin(np.abs(q)))
    q = q - q[k0]
    q[k0] = 0.0
    rows = [dict(original_frame=int(f), source_frame_index=int(f), timestamp_s=float(t), q=float(x), q_valid=True,
                 q_uncertainty=None, processing_index=i, direction="forward") for i, (f, t, x) in
            enumerate(zip(frames, t_s, q))]
    (out / "frame_map.json").write_text(json.dumps(dict(rows=rows, anchor_original_frame=int(frames[k0]))))
    joint = dict(type=obj.joint_type, axis_Wc=(obj.axis / np.linalg.norm(obj.axis)).tolist(),
                 origin_Wc=obj.origin.tolist() if obj.joint_type == "revolute" else None, q_anchor=0.0,
                 q_unit="rad" if obj.joint_type == "revolute" else "m",
                 q_observed_range=[float(q.min()), float(q.max())], reference_frame="Wc",
                 uncertainty=dict(per_time_q_sigma=False, fit_covariance="UNAVAILABLE", authority_id="synthetic"),
                 selection_status="accepted")
    fp = hashlib.sha256((out / "frame_map.json").read_bytes() + json.dumps(joint, sort_keys=True).encode()).hexdigest()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    part = lambda nm_, n: dict(ply=f"{nm_}_canonical.ply", ply_sha256=sha(out / f"{nm_}_canonical.ply"),
                               T_Wc_P=np.eye(4).tolist(), immutable_gaussian_fields=["x", "y", "z", "f_dc_0", "f_dc_1",
                                                                                    "f_dc_2", "opacity", "scale_0",
                                                                                    "scale_1", "scale_2", "rot_0",
                                                                                    "rot_1", "rot_2", "rot_3"],
                               membership="hard", lineage="gaussian_lineage.npz", n_gaussians=n)
    h = dict(schema="articulated3dgs.handoff/v1", object_id=object_id or f"synthetic_{obj.joint_type}",
             producer="ForceArtGS-G9-synthetic",
             source=dict(run_dir=str(out), git_sha="synthetic", config_sha256="synthetic", joint_fingerprint=fp,
                         gaussian_fingerprint=sha(out / "moving_canonical.ply"),
                         acquisition_world="synthetic metric world W (z up, gravity -z)", coordinate_units="m_rad_s",
                         gaussian_parameter_space="W", canonical_anchor_original_frame=int(frames[k0]),
                         canonical_anchor_timestamp_s=float(t_s[k0]), candidate_id="synthetic", status="accepted",
                         notes=dict(synthetic=True)),
             joint=joint, frames=dict(file="frame_map.json", one_shared_q_per_original_frame=True),
             parts=dict(static=part("static", ns), moving=part("moving", nm)),
             validation=dict(joint_frozen_during_final_gaussian_training=True, joint_unchanged_proof="synthetic",
                             reliable_metric_scale=True, heldout_render_checked=False, frame_equivalence_checked=True,
                             hard_membership=True))
    (out / "articulated_3dgs_handoff.json").write_text(json.dumps(h, indent=1))
    return out / "articulated_3dgs_handoff.json", int(frames[k0])
