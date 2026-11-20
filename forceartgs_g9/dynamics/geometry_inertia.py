"""Geometry-constrained full rigid-body inertia (G9 spec section 3).

A single joint-space scalar (I_eff about the hinge, or M_eff along a slider) cannot determine mass, COM and a full
3x3 tensor. A VALIDATED mass-distribution model gives their SHAPE at unit density; one density scale alpha remains:

    revolute : M(alpha) = alpha I_a,1        G(q; alpha) = alpha G_1(q),  G_1 = -m_1 g . (a x (c_1(q) - o))
    prismatic: M(alpha) = alpha m_1          G(alpha)    = -alpha m_1 g . a
    m = alpha m_1,  c = c_1,  J_com = alpha J_com,1

``alpha`` is a uniform density (kg/m^3 for uniform_solid, kg/m^2 for uniform_shell): provenance
``estimated_uniform_solid_density`` / ``estimated_uniform_shell_areal_density`` -- never ``measured_full_inertia``.
Measurement uncertainty (fit, axis, hinge, registration, mesh) is propagated by Monte Carlo; the mass-model
assumptions (uniform material, solid vs shell, no reflected actuator inertia) are systematic and listed separately.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .general_1dof import RigidBodyGravity
from .mass_properties import (DENSITY_UNITS, MeshError, TriMesh, UnitDensityProperties, check_inertia_tensor,
                              transform_inertial, unit_density_properties)

MASS_MODES = ("uniform_solid", "uniform_shell", "segmented_material", "user_link_inertial")
PROVENANCE = {"uniform_solid": "estimated_uniform_solid_density",
              "uniform_shell": "estimated_uniform_shell_areal_density"}
# reflected actuator inertia must be DECLARED: a number (with provenance) or this passive-mechanism statement
PASSIVE = "none_passive_mechanism"


class InertiaReconstructionError(ValueError):
    pass


@dataclass
class GeometryMassModel:
    """Validated unit-density shape of the canonical moving part (q = 0), tied to the accepted joint."""

    props: UnitDensityProperties
    joint_type: str
    axis: np.ndarray
    origin: np.ndarray               # hinge point (revolute); unused for prismatic
    source_provenance: dict
    reflected_actuator_inertia: object = None    # float (declared) | PASSIVE | None (undeclared -> block)
    collision_proxy: bool = False
    assumptions: list = field(default_factory=list)

    @classmethod
    def from_validated_mesh(cls, mesh: TriMesh, mode: str, joint_type: str, axis, origin, *, canonical_frame="W",
                            source_provenance: dict | None = None, reflected_actuator_inertia=None,
                            is_collision_proxy=False, allow_collision_proxy=False) -> "GeometryMassModel":
        prov = dict(source_provenance or {})
        if not prov.get("source"):
            raise InertiaReconstructionError("mass-model geometry needs source_provenance.source (where the mesh "
                                             "comes from, scale, canonical pose, tool exclusion)")
        if is_collision_proxy and not allow_collision_proxy:
            raise InertiaReconstructionError("MASS_GEOMETRY_IS_COLLISION_PROXY: convex collision proxies can grossly "
                                             "overestimate the mass distribution; supply mass_model.mesh or opt in "
                                             "allow_collision_proxy_as_mass_geometry (labelled preview)")
        a = np.asarray(axis, float)
        if a.shape != (3,) or not np.isfinite(a).all() or np.linalg.norm(a) < 1e-9:
            raise InertiaReconstructionError("degenerate joint axis")
        a = a / np.linalg.norm(a)
        o = np.zeros(3) if origin is None else np.asarray(origin, float)
        if joint_type == "revolute" and (origin is None or not np.isfinite(o).all()):
            raise InertiaReconstructionError("revolute mass model needs the hinge origin")
        props = unit_density_properties(mesh, mode, canonical_frame)
        asm = [f"{mode}: uniform material over the validated surface ({DENSITY_UNITS[mode]})",
               "single rigid moving body; geometry posed at the canonical reference q = 0"]
        if mode == "uniform_shell":
            asm.append("thin shell: thickness << size; the areal density absorbs the (unknown) thickness")
        if is_collision_proxy:
            asm.append("PREVIEW: collision proxy used as mass geometry (opt-in)")
        m = cls(props, joint_type, a, o, prov, reflected_actuator_inertia, bool(is_collision_proxy), asm)
        if m.M1 <= 0 or not math.isfinite(m.M1):
            raise InertiaReconstructionError(f"unit-density joint inertia {m.M1} is not positive")
        return m

    # ---------------------------------------------------------------- unit-density joint-space quantities
    @property
    def M1(self) -> float:
        """I_a,1 (revolute) or m_1 (prismatic): M(alpha) = alpha M1 (+ declared reflected inertia)."""
        return self.props.axis_inertia(self.axis, self.origin) if self.joint_type == "revolute" else self.props.m1

    def gravity_unit(self, g_world) -> RigidBodyGravity:
        return RigidBodyGravity(self.joint_type, tuple(self.axis), tuple(self.origin), tuple(self.props.c1),
                                self.props.m1, tuple(np.asarray(g_world, float)))

    def reflected(self) -> float:
        r = self.reflected_actuator_inertia
        if r is None:
            raise InertiaReconstructionError("REFLECTED_INERTIA_UNDECLARED: declare mass_model."
                                             f"reflected_actuator_inertia (a measured value, or '{PASSIVE}' for a "
                                             "passive mechanism) before a full link inertia is exported")
        if r == PASSIVE:
            return 0.0
        if isinstance(r, bool) or not isinstance(r, (int, float)) or r < 0 or not math.isfinite(r):
            raise InertiaReconstructionError(f"reflected_actuator_inertia {r!r} must be >= 0 or '{PASSIVE}'")
        return float(r)

    def alpha_from_effective(self, M_eff: float) -> float:
        """Post-fit projection alpha = (M_eff - M_reflected) / M1."""
        return (float(M_eff) - self.reflected()) / self.M1

    # ---------------------------------------------------------------- reconstruction
    def reconstruct(self, alpha: float, alpha_std: float | None = None, *, fitted_M_eff: float | None = None,
                    tol_rel: float = 1e-6, mc: dict | None = None) -> dict:
        """Full moving-link mass properties from alpha. ``mc``: Monte-Carlo samples (see ``monte_carlo``)."""
        if not (math.isfinite(alpha) and alpha > 0):
            raise InertiaReconstructionError(f"alpha = {alpha} must be finite and > 0 (density scale)")
        refl = self.reflected()
        m = alpha * self.props.m1
        J = alpha * self.props.J1
        I_axis = alpha * self.M1
        chk = check_inertia_tensor(J, "J_com")
        rep = dict(mode=self.props.mode, provenance=PROVENANCE[self.props.mode] + (
            "_collision_proxy_preview" if self.collision_proxy else ""),
                   density_estimate=alpha, density_units=self.props.density_units, density_std=alpha_std,
                   mass_kg=m, com_m=self.props.c1.tolist(), frame=self.props.frame,
                   inertia_com_kg_m2=J.tolist(), principal_moments_kg_m2=np.linalg.eigvalsh(J).tolist(),
                   joint_space=dict(M1=self.M1, reflected_actuator_inertia=refl,
                                    predicted_M_eff=I_axis + refl,
                                    **({"I_axis_kg_m2": I_axis} if self.joint_type == "revolute" else
                                       {"m_axis_kg": I_axis})),
                   tensor_checks=chk, assumptions=list(self.assumptions),
                   systematic_not_bootstrappable=[
                       "uniform density over the whole moving part (real parts are heterogeneous)",
                       f"{self.props.mode} geometry model of the true mass distribution",
                       "reflected actuator inertia " + ("declared passive (zero)" if self.reflected_actuator_inertia ==
                                                        PASSIVE else f"declared {refl}")],
                   not_a_claim="estimated density x validated geometry; NOT measured_full_inertia")
        if fitted_M_eff is not None:
            err = abs(I_axis + refl - fitted_M_eff) / max(abs(fitted_M_eff), 1e-300)
            rep["projection_error_rel"] = float(err)
            rep["projection_check"] = "PASS" if err <= tol_rel else "FAIL"
        if mc is not None:
            rep["uncertainty"] = mc
        if not chk["ok"]:
            raise InertiaReconstructionError(f"reconstructed tensor not physical: {chk}")
        return rep

    def monte_carlo(self, alpha: float, alpha_std: float, *, axis_std_rad=0.0, origin_std_m=0.0, mesh_std_m=0.0,
                    n=400, seed=0, hold="M_eff", mesh: TriMesh | None = None) -> dict:
        """Propagate measurement uncertainty. Per sample: perturb axis (cone), hinge (perpendicular offset), mesh
        (rigid jitter of the whole surface, plus per-vertex noise if ``mesh``) and the fitted scale. With
        hold='M_eff' the data-determined joint-space value is held fixed: alpha_s = alpha M1 / M1_s * xi_s."""
        rng = np.random.default_rng(seed)
        out = dict(mass=[], com=[], J=[], I_axis=[], alpha=[])
        base_M1 = self.M1
        for _ in range(n):
            a = self.axis + axis_std_rad * rng.standard_normal(3)
            a = a / np.linalg.norm(a)
            o = self.origin + origin_std_m * rng.standard_normal(3)
            props = self.props
            if mesh is not None and mesh_std_m > 0:
                jit = TriMesh(mesh.vertices + mesh_std_m * rng.standard_normal(mesh.vertices.shape), mesh.faces)
                try:
                    props = unit_density_properties(jit, self.props.mode, self.props.frame)
                except MeshError:
                    continue
            shift = mesh_std_m * rng.standard_normal(3) if mesh is None else np.zeros(3)
            c1 = props.c1 + shift
            if self.joint_type == "revolute":
                r = c1 - o
                M1s = float(a @ (props.J1 + props.m1 * ((r @ r) * np.eye(3) - np.outer(r, r))) @ a)
            else:
                M1s = props.m1
            xi = 1.0 + (alpha_std / alpha) * rng.standard_normal() if alpha_std else 1.0
            al = alpha * (base_M1 / M1s if hold == "M_eff" else 1.0) * xi
            if al <= 0:
                continue
            out["alpha"].append(al)
            out["mass"].append(al * props.m1)
            out["com"].append(c1)
            out["J"].append(al * props.J1)
            out["I_axis"].append(al * M1s)
        res = {}
        for k, v in out.items():
            arr = np.asarray(v)
            res[k] = dict(mean=arr.mean(0).tolist(), std=arr.std(0).tolist(),
                          ci95=[np.percentile(arr, 2.5, 0).tolist(), np.percentile(arr, 97.5, 0).tolist()])
        res["n_samples"] = len(out["mass"])
        res["method"] = ("Monte Carlo over fitted scale, axis direction, hinge position and mesh perturbation; "
                         "measurement uncertainty only (mass-model assumptions are systematic, not sampled)")
        res["inputs"] = dict(alpha_std=alpha_std, axis_std_rad=axis_std_rad, origin_std_m=origin_std_m,
                             mesh_std_m=mesh_std_m, hold=hold, seed=seed)
        return res

    def to_link_frame(self, rep: dict, T_link_W) -> dict:
        """COM / COM-tensor in the moving-link frame (J_L = R J R^T; no second parallel-axis shift)."""
        c, J = transform_inertial(T_link_W, rep["com_m"], np.asarray(rep["inertia_com_kg_m2"]))
        return dict(com_link_m=c.tolist(), inertia_com_link_kg_m2=(0.5 * (J + J.T)).tolist())


def check_user_inertial(user: dict, model: GeometryMassModel | None, joint_type, axis, origin, M_eff, M_eff_std,
                        n_sigma=3.0) -> dict:
    """user_link_inertial: supplied mass/COM/full tensor (measured or estimated, with covariance). It is checked
    against the identified joint-space value; a disagreement is INERTIA_GEOMETRY_CONFLICT, never an overwrite."""
    for k in ("mass_kg", "com_m", "inertia_com_kg_m2", "provenance"):
        if user.get(k) is None:
            raise InertiaReconstructionError(f"user_link_inertial needs {k}")
    m = float(user["mass_kg"])
    c = np.asarray(user["com_m"], float)
    J = np.asarray(user["inertia_com_kg_m2"], float)
    chk = check_inertia_tensor(J, "user J_com")
    if not (m > 0 and chk["ok"]):
        raise InertiaReconstructionError(f"user inertial is not physical (mass {m}, {chk})")
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    if joint_type == "revolute":
        r = c - np.asarray(origin, float)
        pred = float(a @ (J + m * ((r @ r) * np.eye(3) - np.outer(r, r))) @ a)
    else:
        pred = m
    refl = 0.0 if model is None else model.reflected()
    pred += refl
    u_std = float(user.get("joint_space_std") or 0.0)
    tot = math.hypot(M_eff_std or 0.0, u_std)
    z = abs(pred - M_eff) / tot if tot > 0 else (0.0 if pred == M_eff else math.inf)
    status = "CONSISTENT" if z <= n_sigma else "INERTIA_GEOMETRY_CONFLICT"
    return dict(status=status, predicted_M_eff_from_user=pred, identified_M_eff=M_eff, z=float(z), n_sigma=n_sigma,
                action="user inertial kept as supplied" if status == "CONSISTENT" else
                "conflict reported; neither value overwritten (explicit user decision required)",
                user_provenance=user["provenance"])


def segmented_rank(n_regions: int, n_independent_joint_measurements: int, fixed_ratios: bool) -> dict:
    """segmented_material: k free densities from m independent joint-space measurements (one per joint / mode)."""
    free = 1 if fixed_ratios else n_regions
    ok = n_independent_joint_measurements >= free
    return dict(free_densities=free, independent_measurements=n_independent_joint_measurements,
                identifiable=bool(ok), reason=None if ok else
                f"{free} free densities but only {n_independent_joint_measurements} independent joint-space "
                f"measurement(s): rank deficient; fix the density ratios from external material data")


def alignment_check(mesh: TriMesh, ref_xyz, ratio_range=(0.5, 2.0), max_offset_frac=0.25, inside_tol_frac=0.05,
                    min_near=0.5) -> dict:
    """Mass-model mesh vs the moving part's own Gaussian means (same canonical frame, q = 0): extent ratio
    (catches mm declared as m), AABB-centre offset (wrong anchor frame / swapped link) and the fraction of means
    near the surface. Declared gates, not measurements."""
    x = np.asarray(ref_xyz, float)
    lo, hi = np.percentile(x, 2, 0), np.percentile(x, 98, 0)
    gd = float(np.linalg.norm(hi - lo))
    mlo, mhi = mesh.vertices.min(0), mesh.vertices.max(0)
    md = float(np.linalg.norm(mhi - mlo))
    ratio = md / gd if gd > 0 else float("inf")
    off = float(np.linalg.norm(0.5 * (mlo + mhi) - 0.5 * (lo + hi)))
    v = mesh.vertices[mesh.faces]
    cen = v.mean(1)
    from scipy.spatial import cKDTree
    sample = x[:: max(1, len(x) // 5000)]
    d, _ = cKDTree(np.vstack([cen, mesh.vertices])).query(sample)
    edge = float(np.max(np.linalg.norm(v[:, 1] - v[:, 0], axis=1)))
    near = float(np.mean(d <= inside_tol_frac * gd + edge))
    reasons = []
    if not ratio_range[0] <= ratio <= ratio_range[1]:
        reasons.append(f"mesh extent {ratio:.3g} x the moving Gaussians' extent (wrong units / wrong mesh)")
    if off > max_offset_frac * gd:
        reasons.append(f"mesh centre {off:.3g} m from the moving Gaussians (wrong anchor frame / swapped link)")
    if near < min_near:
        reasons.append(f"only {near:.0%} of the moving Gaussian means lie near the mesh surface")
    return dict(status="PASS" if not reasons else "FAIL", extent_ratio=ratio, centre_offset_m=off,
                gaussian_extent_m=gd, fraction_near_surface=near, reasons=reasons)
