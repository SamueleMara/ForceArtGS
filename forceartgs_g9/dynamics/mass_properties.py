"""Unit-density mass properties of a validated moving-part surface (G9 spec section 3.1-3.2).

uniform_solid : watertight, consistently outward-oriented triangle mesh; exact polyhedral volume moments
                (divergence theorem) at unit VOLUMETRIC density 1 kg/m^3 -> m_1 [kg per kg/m^3] = volume [m^3].
uniform_shell : any valid triangulation (open allowed); exact triangle AREA moments at unit AREAL density
                1 kg/m^2 -> m_1 [kg per kg/m^2] = area [m^2].

All quantities are in the frame the mesh vertices are expressed in (normally the canonical capture world W at the
reference q = 0) and in metres after the declared unit scale. J_com is the full symmetric 3x3 tensor about the COM.
3DGS Gaussians are NOT a density: nothing here integrates Gaussian covariance or opacity.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001}
MODES = ("uniform_solid", "uniform_shell")
DENSITY_UNITS = {"uniform_solid": "kg/m^3", "uniform_shell": "kg/m^2"}
M1_UNITS = {"uniform_solid": "m^3 (kg per kg/m^3)", "uniform_shell": "m^2 (kg per kg/m^2)"}


class MeshError(ValueError):
    """Invalid / unsuitable mass-model geometry (readable reason)."""


@dataclass
class TriMesh:
    vertices: np.ndarray            # [V,3] metres (after unit scaling)
    faces: np.ndarray               # [F,3] int
    units: str = "m"
    source: str = ""

    @classmethod
    def from_arrays(cls, vertices, faces, units="m", source=""):
        if units not in UNIT_SCALE:
            raise MeshError(f"mesh units {units!r} not in {list(UNIT_SCALE)}")
        V = np.asarray(vertices, float) * UNIT_SCALE[units]
        F = np.asarray(faces)
        if V.ndim != 2 or V.shape[1] != 3 or len(V) < 3:
            raise MeshError("vertices must be [V,3] with V >= 3")
        if not np.isfinite(V).all():
            raise MeshError("non-finite vertex coordinates")
        if F.ndim != 2 or F.shape[1] != 3 or len(F) < 1 or not np.issubdtype(F.dtype, np.integer):
            raise MeshError("faces must be an integer [F,3] array")
        if F.min() < 0 or F.max() >= len(V):
            raise MeshError("face indices out of range")
        return cls(V, F.astype(np.int64), units, source)

    @classmethod
    def read_obj(cls, path, units="m"):
        vs, fs = [], []
        for line in open(path):
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                vs.append([float(x) for x in p[1:4]])
            elif p[0] == "f":
                idx = [int(x.split("/")[0]) for x in p[1:]]
                idx = [i - 1 if i > 0 else len(vs) + i for i in idx]
                for k in range(1, len(idx) - 1):        # fan triangulation of polygons
                    fs.append([idx[0], idx[k], idx[k + 1]])
        return cls.from_arrays(vs, np.asarray(fs, dtype=np.int64), units, str(path))

    def transformed(self, T) -> "TriMesh":
        T = np.asarray(T, float)
        return TriMesh(self.vertices @ T[:3, :3].T + T[:3, 3], self.faces.copy(), "m", self.source)

    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.vertices, np.float64).tobytes())
        h.update(np.ascontiguousarray(self.faces, np.int64).tobytes())
        return h.hexdigest()

    def write_obj(self, path, header=()):
        with open(path, "w") as fh:
            for h in header:
                fh.write(f"# {h}\n")
            for v in self.vertices:
                fh.write(f"v {v[0]!r} {v[1]!r} {v[2]!r}\n")
            for f in self.faces:
                fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


def topology(m: TriMesh) -> dict:
    """Edge-manifold / orientation / closure report."""
    F = m.faces
    tri = m.vertices[F]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    scale = float(np.ptp(m.vertices, axis=0).max()) or 1.0
    degenerate = int((area2 <= 1e-12 * scale * scale).sum())
    directed = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    und = np.sort(directed, axis=1)
    _, inv, counts = np.unique(und, axis=0, return_inverse=True, return_counts=True)
    n_edges = len(counts)
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    dkeys, dcounts = np.unique(directed, axis=0, return_counts=True)
    dup_directed = int((dcounts > 1).sum())                 # same directed edge twice: inconsistent orientation
    used = np.unique(F)
    chi = int(len(used) - n_edges + len(F))
    return dict(n_vertices=int(len(m.vertices)), n_faces=int(len(F)), n_edges=n_edges, boundary_edges=boundary,
                nonmanifold_edges=nonmanifold, inconsistently_oriented_edges=dup_directed,
                degenerate_faces=degenerate, euler_characteristic=chi,
                watertight=bool(boundary == 0 and nonmanifold == 0),
                consistently_oriented=bool(dup_directed == 0),
                self_intersection="NOT_CHECKED (no exact triangle-triangle test; closure, orientation and positive "
                                  "signed volume are checked)")


def _solid_integrals(V, F):
    v = V[F]
    x0, x1, x2 = v[:, 0], v[:, 1], v[:, 2]
    d = np.cross(x1 - x0, x2 - x0)
    f1 = x0 + x1 + x2
    f2 = x0 * x0 + x1 * (x0 + x1) + x2 * f1
    f3 = x0 * x0 * x0 + x1 * (x0 * x0 + x0 * x1 + x1 * x1) + x2 * f2
    g0, g1, g2 = f2 + x0 * (f1 + x0), f2 + x1 * (f1 + x1), f2 + x2 * (f1 + x2)
    s = np.zeros(10)
    s[0] = (d[:, 0] * f1[:, 0]).sum()
    s[1:4] = (d * f2).sum(0)
    s[4:7] = (d * f3).sum(0)
    s[7] = (d[:, 0] * (x0[:, 1] * g0[:, 0] + x1[:, 1] * g1[:, 0] + x2[:, 1] * g2[:, 0])).sum()
    s[8] = (d[:, 1] * (x0[:, 2] * g0[:, 1] + x1[:, 2] * g1[:, 1] + x2[:, 2] * g2[:, 1])).sum()
    s[9] = (d[:, 2] * (x0[:, 0] * g0[:, 2] + x1[:, 0] * g1[:, 2] + x2[:, 0] * g2[:, 2])).sum()
    s *= np.array([1 / 6, 1 / 24, 1 / 24, 1 / 24, 1 / 60, 1 / 60, 1 / 60, 1 / 120, 1 / 120, 1 / 120])
    vol = s[0]
    if not vol > 0:
        raise MeshError(f"signed volume {vol:.6g} <= 0: open, inside-out or degenerate surface")
    com = s[1:4] / vol
    # second moments about the origin: S_ij = int x_i x_j dV
    Sxx, Syy, Szz = s[4], s[5], s[6]
    Sxy, Syz, Szx = s[7], s[8], s[9]
    S = np.array([[Sxx, Sxy, Szx], [Sxy, Syy, Syz], [Szx, Syz, Szz]])
    return vol, com, S


def _shell_integrals(V, F):
    v = V[F]
    a, b, c = v[:, 0], v[:, 1], v[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    A = float(area.sum())
    if not A > 0:
        raise MeshError("total surface area is zero")
    ssum = a + b + c
    com = (area[:, None] * ssum).sum(0) / (3 * A)
    # int_T x x^T dA = A_T/12 (a a^T + b b^T + c c^T + s s^T), s = a + b + c
    outer = (np.einsum("fi,fj->fij", a, a) + np.einsum("fi,fj->fij", b, b) + np.einsum("fi,fj->fij", c, c)
             + np.einsum("fi,fj->fij", ssum, ssum))
    S = (area[:, None, None] * outer).sum(0) / 12.0
    return A, com, S


def _inertia_about_com(m, com, S):
    """J_com = tr(C) I - C with C = S - m c c^T (second moment about the COM)."""
    C = S - m * np.outer(com, com)
    J = np.trace(C) * np.eye(3) - C
    return 0.5 * (J + J.T)


def check_inertia_tensor(J, where="inertia") -> dict:
    J = np.asarray(J, float)
    s = float(np.abs(J).max())
    sym = float(np.abs(J - J.T).max())
    ev = np.linalg.eigvalsh(0.5 * (J + J.T)) if s > 0 else np.zeros(3)
    pd = bool(s > 0 and ev.min() > 0)
    tri = bool(pd and ev[2] <= (ev[0] + ev[1]) * (1 + 1e-9))
    ok = bool(s > 0 and sym <= 1e-9 * s and pd and tri)
    return dict(ok=ok, symmetric_abs_err=sym, principal_moments=ev.tolist(), positive_definite=pd,
                triangle_inequality=tri, where=where)


@dataclass
class UnitDensityProperties:
    mode: str
    m1: float                      # volume [m^3] (solid) or area [m^2] (shell): mass per unit density
    c1: np.ndarray                 # COM [m] in `frame`
    J1: np.ndarray                 # full 3x3 inertia about the COM per unit density, in `frame` axes
    frame: str
    mesh_sha256: str
    topology: dict
    checks: dict = field(default_factory=dict)

    @property
    def density_units(self) -> str:
        return DENSITY_UNITS[self.mode]

    def axis_inertia(self, axis, origin) -> float:
        """I_a,1 = a^T [J_com,1 + m_1 ((r.r) I - r r^T)] a, r = c_1 - o (independent of o along a)."""
        a = np.asarray(axis, float)
        a = a / np.linalg.norm(a)
        r = self.c1 - np.asarray(origin, float)
        Jo = self.J1 + self.m1 * ((r @ r) * np.eye(3) - np.outer(r, r))
        return float(a @ Jo @ a)

    def principal(self):
        ev, V = np.linalg.eigh(self.J1)
        if np.linalg.det(V) < 0:
            V[:, 0] *= -1
        return ev, V

    def to_dict(self) -> dict:
        ev, V = self.principal()
        return dict(mode=self.mode, frame=self.frame, m1=self.m1, m1_units=M1_UNITS[self.mode],
                    density_units=self.density_units, c1_m=self.c1.tolist(), J_com_1=self.J1.tolist(),
                    J_units=f"kg*m^2 per {self.density_units}", principal_moments_1=ev.tolist(),
                    principal_axes=V.tolist(), mesh_sha256=self.mesh_sha256, topology=self.topology,
                    checks=self.checks)


def unit_density_properties(mesh: TriMesh, mode: str, frame: str = "W") -> UnitDensityProperties:
    """Validate the geometry for ``mode`` and integrate it at unit density. Fails with a readable reason."""
    if mode not in MODES:
        raise MeshError(f"mass mode {mode!r} not in {MODES} (segmented_material / user_link_inertial are handled "
                        f"by geometry_inertia)")
    topo = topology(mesh)
    if topo["degenerate_faces"]:
        raise MeshError(f"{topo['degenerate_faces']} degenerate (zero-area) faces")
    if mode == "uniform_solid":
        if not topo["watertight"]:
            raise MeshError(f"uniform_solid needs a watertight surface: {topo['boundary_edges']} boundary and "
                            f"{topo['nonmanifold_edges']} non-manifold edges (an open shell must use uniform_shell)")
        if not topo["consistently_oriented"]:
            raise MeshError(f"uniform_solid needs consistent face orientation ({topo['inconsistently_oriented_edges']}"
                            f" edges traversed twice in the same direction)")
        m1, c1, S = _solid_integrals(mesh.vertices, mesh.faces)
    else:
        if topo["nonmanifold_edges"]:
            raise MeshError(f"{topo['nonmanifold_edges']} non-manifold edges in the shell triangulation")
        m1, c1, S = _shell_integrals(mesh.vertices, mesh.faces)
    J1 = _inertia_about_com(m1, c1, S)
    chk = check_inertia_tensor(J1, f"{mode} J_com_1")
    if not chk["ok"]:
        raise MeshError(f"unit-density tensor is not physical: {chk}")
    return UnitDensityProperties(mode, float(m1), c1, J1, frame, mesh.sha256(), topo, dict(tensor=chk))


def transform_inertial(T_dst_src, c_src, J_src):
    """COM and COM-tensor in another frame: c' = T c, J' = R J R^T (no parallel-axis term: COM-origin tensor)."""
    T = np.asarray(T_dst_src, float)
    R = T[:3, :3]
    return R @ np.asarray(c_src, float) + T[:3, 3], R @ np.asarray(J_src, float) @ R.T
