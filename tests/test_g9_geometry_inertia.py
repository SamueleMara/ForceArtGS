"""G9 Gate B: geometry-constrained full inertia (synthetic truth only)."""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9.dynamics.general_1dof import GeneralTerms, sim_params, simulate_general  # noqa: E402
from forceartgs_g9.dynamics.geometry_inertia import (PASSIVE, GeometryMassModel,  # noqa: E402
                                                     InertiaReconstructionError, alignment_check,
                                                     check_user_inertial)
from forceartgs_g9.dynamics.mass_properties import (MeshError, TriMesh, check_inertia_tensor,  # noqa: E402
                                                    transform_inertial, unit_density_properties)
from forceartgs_g9.estimators.model_spec import ModelSpec  # noqa: E402
from forceartgs_g9.synthetic import ZOH, box_mesh, make_object, panel_mesh, profile, subdivided_box  # noqa: E402

PROV = dict(source="unit test")


def uv_sphere(r=0.1, n=48):
    V, F = [[0, 0, r], [0, 0, -r]], []
    rings = []
    for i in range(1, n):
        th = math.pi * i / n
        ring = []
        for j in range(2 * n):
            ph = 2 * math.pi * j / (2 * n)
            ring.append(len(V))
            V.append([r * math.sin(th) * math.cos(ph), r * math.sin(th) * math.sin(ph), r * math.cos(th)])
        rings.append(ring)
    m = 2 * n
    for j in range(m):
        F.append([0, rings[0][j], rings[0][(j + 1) % m]])
        F.append([1, rings[-1][(j + 1) % m], rings[-1][j]])
    for a, b in zip(rings[:-1], rings[1:]):
        for j in range(m):
            F += [[a[j], b[j], b[(j + 1) % m]], [a[j], b[(j + 1) % m], a[(j + 1) % m]]]
    return TriMesh.from_arrays(np.array(V), np.array(F))


def cylinder(r=0.05, h=0.3, n=256):
    V = [[0, 0, 0], [0, 0, h]]
    for z in (0, h):
        for j in range(n):
            V.append([r * math.cos(2 * math.pi * j / n), r * math.sin(2 * math.pi * j / n), z])
    F = []
    for j in range(n):
        a, b = 2 + j, 2 + (j + 1) % n
        c, d = 2 + n + j, 2 + n + (j + 1) % n
        F += [[0, b, a], [1, c, d], [a, b, d], [a, d, c]]
    return TriMesh.from_arrays(np.array(V), np.array(F))


def rot(axis, ang):
    from forceartgs_g9.dynamics.general_1dof import rotation
    return rotation(axis, ang)


class GateB(unittest.TestCase):
    def test_b01_solids_and_parallel_axis(self):
        a, b, c = 0.4, 0.2, 0.1
        p = unit_density_properties(box_mesh([0, 0, 0], [a, b, c]), "uniform_solid")
        m = a * b * c
        np.testing.assert_allclose(np.diag(p.J1), m / 12 * np.array([b * b + c * c, a * a + c * c, a * a + b * b]),
                                   rtol=1e-12)
        self.assertAlmostEqual(p.axis_inertia([0, 0, 1], [0, 0, 0]), m * (a * a + b * b) / 3, places=12)  # edge axis
        s = unit_density_properties(uv_sphere(0.1, 64), "uniform_solid")
        V = 4 / 3 * math.pi * 1e-3
        self.assertAlmostEqual(s.m1 / V, 1.0, delta=3e-3)
        np.testing.assert_allclose(np.diag(s.J1), 0.4 * s.m1 * 0.01, rtol=5e-3)
        cy = unit_density_properties(cylinder(), "uniform_solid")
        self.assertAlmostEqual(cy.J1[2, 2] / (0.5 * cy.m1 * 0.05 ** 2), 1.0, delta=1e-3)
        self.assertAlmostEqual(cy.J1[0, 0] / (cy.m1 * (3 * 0.05 ** 2 + 0.09) / 12), 1.0, delta=1e-3)
        # rigid transform: J' = R J R^T, c' = T c
        T = np.eye(4)
        T[:3, :3] = rot([0.3, 0.5, 0.8], 0.7)
        T[:3, 3] = [0.5, -0.2, 1.0]
        pt = unit_density_properties(box_mesh([0, 0, 0], [a, b, c]).transformed(T), "uniform_solid")
        c2, J2 = transform_inertial(T, p.c1, p.J1)
        np.testing.assert_allclose(pt.c1, c2, atol=1e-12)
        np.testing.assert_allclose(pt.J1, J2, atol=1e-14)

    def test_b02_thin_shell_and_solid_on_open_shell_fails(self):
        a, b = 0.6, 0.4
        sh = unit_density_properties(panel_mesh([0, 0, 0], [a, b, 0]), "uniform_shell")
        A = a * b
        np.testing.assert_allclose(np.diag(sh.J1), A / 12 * np.array([b * b, a * a, a * a + b * b]), rtol=1e-12)
        with self.assertRaises(MeshError):
            unit_density_properties(panel_mesh([0, 0, 0], [a, b, 0]), "uniform_solid")

    def test_b03_alpha_recovers_mass_com_tensor_and_axis_inertia(self):
        obj = make_object("revolute")
        mm = obj.mass_model
        rep = mm.reconstruct(obj.density, 1.0, fitted_M_eff=obj.M_true)
        p = unit_density_properties(obj.mesh, "uniform_solid")
        self.assertAlmostEqual(rep["mass_kg"], obj.density * p.m1, places=10)
        np.testing.assert_allclose(rep["com_m"], p.c1)
        np.testing.assert_allclose(rep["inertia_com_kg_m2"], obj.density * p.J1)
        self.assertAlmostEqual(rep["joint_space"]["I_axis_kg_m2"], obj.M_true, places=12)
        self.assertEqual(rep["projection_check"], "PASS")
        self.assertEqual(rep["provenance"], "estimated_uniform_solid_density")
        self.assertIn("NOT measured_full_inertia", rep["not_a_claim"])

    def test_b04_hinge_translation_along_axis_is_a_gauge(self):
        obj = make_object("revolute")
        p = obj.mass_model.props
        o2 = obj.origin + 0.37 * obj.axis
        self.assertAlmostEqual(p.axis_inertia(obj.axis, obj.origin), p.axis_inertia(obj.axis, o2), places=13)
        g1 = obj.mass_model.gravity_unit(obj.gravity.g_world)
        m2 = GeometryMassModel.from_validated_mesh(obj.mesh, "uniform_solid", "revolute", obj.axis, o2,
                                                   source_provenance=PROV, reflected_actuator_inertia=PASSIVE)
        g2 = m2.gravity_unit(obj.gravity.g_world)
        for q in (-0.7, 0.0, 0.4, 1.3):
            self.assertAlmostEqual(g1.G(q), g2.G(q), places=11)

    def test_b05_prismatic_mass_from_effective_mass(self):
        obj = make_object("prismatic")
        mm = obj.mass_model
        a = mm.alpha_from_effective(obj.M_true)
        self.assertAlmostEqual(a, obj.density, places=9)
        rep = mm.reconstruct(a, None, fitted_M_eff=obj.M_true)
        self.assertAlmostEqual(rep["mass_kg"], obj.M_true, places=9)
        self.assertIn("single rigid moving body", " ".join(rep["assumptions"]))

    def test_b06_scalar_inertia_cannot_reveal_heterogeneity_or_reflected_inertia(self):
        # two-material lid: dense strip at the free edge; the TRUE axis inertia is matched by a uniform density
        a = np.array([1.0, 0, 0])
        o = np.array([0.0, 0.0, 0.5])
        light = unit_density_properties(box_mesh([-0.25, -0.01, 0.15], [0.25, 0.01, 0.5]), "uniform_solid")
        heavy = unit_density_properties(box_mesh([-0.25, -0.01, 0.10], [0.25, 0.01, 0.15]), "uniform_solid")
        I_true = 300 * light.axis_inertia(a, o) + 3000 * heavy.axis_inertia(a, o)
        m_true = 300 * light.m1 + 3000 * heavy.m1
        whole = box_mesh([-0.25, -0.01, 0.10], [0.25, 0.01, 0.5])
        mm = GeometryMassModel.from_validated_mesh(whole, "uniform_solid", "revolute", a, o, source_provenance=PROV,
                                                   reflected_actuator_inertia=PASSIVE)
        rep = mm.reconstruct(mm.alpha_from_effective(I_true), None, fitted_M_eff=I_true)
        self.assertAlmostEqual(rep["joint_space"]["I_axis_kg_m2"], I_true, places=10)   # same joint inertia ...
        self.assertGreater(abs(rep["mass_kg"] - m_true) / m_true, 0.1)                  # ... different link mass
        self.assertTrue(rep["provenance"].startswith("estimated_"))
        und = GeometryMassModel.from_validated_mesh(whole, "uniform_solid", "revolute", a, o, source_provenance=PROV)
        with self.assertRaises(InertiaReconstructionError):
            und.reconstruct(500.0)                                                    # reflected inertia undeclared
        dec = GeometryMassModel.from_validated_mesh(whole, "uniform_solid", "revolute", a, o, source_provenance=PROV,
                                                    reflected_actuator_inertia=0.01)
        self.assertLess(dec.alpha_from_effective(I_true), mm.alpha_from_effective(I_true))

    def test_b07_invalid_geometry_wrong_scale_and_anchor(self):
        good = box_mesh([0, 0, 0], [0.2, 0.2, 0.2])
        open_box = TriMesh.from_arrays(good.vertices, good.faces[:-2])
        with self.assertRaises(MeshError):
            unit_density_properties(open_box, "uniform_solid")
        inverted = TriMesh.from_arrays(good.vertices, good.faces[:, ::-1])
        with self.assertRaises(MeshError):
            unit_density_properties(inverted, "uniform_solid")
        with self.assertRaises(MeshError):
            TriMesh.from_arrays(np.r_[good.vertices[:-1], [[np.nan, 0, 0]]], good.faces)
        with self.assertRaises(MeshError):
            unit_density_properties(TriMesh.from_arrays([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]]),
                                    "uniform_shell")
        with self.assertRaises(InertiaReconstructionError):
            GeometryMassModel.from_validated_mesh(good, "uniform_solid", "revolute", [0, 0, 0], [0, 0, 0],
                                                  source_provenance=PROV, reflected_actuator_inertia=PASSIVE)
        rng = np.random.default_rng(0)
        xyz = rng.uniform([0, 0, 0], [0.2, 0.2, 0.2], (3000, 3))
        self.assertEqual(alignment_check(good, xyz)["status"], "PASS")
        mm_as_m = TriMesh.from_arrays(good.vertices * 1000.0, good.faces)               # mm numbers declared as m
        self.assertEqual(alignment_check(mm_as_m, xyz)["status"], "FAIL")
        shifted = good.transformed(np.r_[np.c_[np.eye(3), [0.5, 0, 0]], [[0, 0, 0, 1]]])  # wrong anchor frame
        self.assertEqual(alignment_check(shifted, xyz)["status"], "FAIL")

    def test_b08_tensor_validity_and_frame_invariance(self):
        p = unit_density_properties(box_mesh([0, 0, 0], [0.4, 0.2, 0.1]), "uniform_solid")
        chk = check_inertia_tensor(p.J1)
        self.assertTrue(chk["ok"] and chk["positive_definite"] and chk["triangle_inequality"])
        T = np.eye(4)
        T[:3, :3] = rot([1, 2, 3], 1.1)
        _, J2 = transform_inertial(T, p.c1, p.J1)
        np.testing.assert_allclose(np.linalg.eigvalsh(J2), np.linalg.eigvalsh(p.J1), rtol=1e-12)
        bad = np.diag([1.0, 1.0, 3.0])
        self.assertFalse(check_inertia_tensor(bad)["ok"])                             # violates I3 <= I1 + I2

    def test_b09_geometry_coupled_equals_direct_rigid_body(self):
        obj = make_object("revolute")
        spec = ModelSpec(obj.terms, "revolute", "geometry_coupled", obj.mass_model, obj.gravity)
        th = np.array([obj.density, obj.c, obj.Fc, obj.Fs - obj.Fc])
        t, Q = profile([(0.3, 0.0), (0.8, 0.0, 1.5), (1.0, 0.0)])
        e = ZOH(t, Q)
        tq = np.linspace(0, 2.0, 120)
        a = simulate_general(spec.sim(th), e, tq).q
        rb = obj.mass_model.gravity_unit(obj.gravity.g_world)
        m = obj.density * obj.mass_model.props.m1
        from forceartgs_g9.dynamics.general_1dof import RigidBodyGravity
        direct = RigidBodyGravity("revolute", tuple(obj.axis), tuple(obj.origin), tuple(rb.com0), m,
                                  obj.gravity.g_world)
        p = sim_params(obj.terms, "revolute", M=obj.M_true, c=obj.c, Fc=obj.Fc, Fs=obj.Fs, G=direct.G)
        np.testing.assert_allclose(a, simulate_general(p, e, tq).q, atol=1e-9)
        js = ModelSpec(obj.terms, "revolute", "joint_space_only")
        A, B = rb.basis_coefficients()
        th2 = np.array([obj.M_true, obj.c, obj.Fc, obj.Fs - obj.Fc, 2 * obj.density * A, 2 * obj.density * B])
        self.assertGreater(np.abs(simulate_general(js.sim(th2), e, tq).q - a).max(), 0.05)  # free amplitude differs
        self.assertEqual(js.physical(th2)["gravity"]["kind"], "effective_basis")

    def test_b10_user_inertial_conflict_is_reported_not_overwritten(self):
        obj = make_object("revolute")
        p = obj.mass_model.props
        user = dict(mass_kg=obj.density * p.m1, com_m=p.c1.tolist(), inertia_com_kg_m2=(obj.density * p.J1).tolist(),
                    provenance="measured", joint_space_std=1e-3)
        ok = check_user_inertial(user, obj.mass_model, "revolute", obj.axis, obj.origin, obj.M_true, 1e-3)
        self.assertEqual(ok["status"], "CONSISTENT")
        bad = check_user_inertial(user, obj.mass_model, "revolute", obj.axis, obj.origin, 2 * obj.M_true, 1e-3)
        self.assertEqual(bad["status"], "INERTIA_GEOMETRY_CONFLICT")
        self.assertIn("neither value overwritten", bad["action"])

    def test_b11_uncertainty_widens_and_systematics_stay_separate(self):
        mesh = subdivided_box([-0.25, -0.01, 0.1], [0.25, 0.01, 0.5], n=3)
        mm = GeometryMassModel.from_validated_mesh(mesh, "uniform_solid", "revolute", [1, 0, 0], [0, 0, 0.5],
                                                   source_provenance=PROV, reflected_actuator_inertia=PASSIVE)
        lo = mm.monte_carlo(700.0, 1.0, axis_std_rad=0.001, origin_std_m=0.0005, n=300)
        hi = mm.monte_carlo(700.0, 1.0, axis_std_rad=0.03, origin_std_m=0.02, n=300)
        self.assertGreater(hi["mass"]["std"], lo["mass"]["std"])
        self.assertGreater(np.max(hi["J"]["std"]), np.max(lo["J"]["std"]))
        msh = mm.monte_carlo(700.0, 1.0, mesh_std_m=0.002, n=100, mesh=mesh)
        self.assertGreater(msh["n_samples"], 50)
        rep = mm.reconstruct(700.0, 1.0, mc=hi)
        self.assertIn("measurement uncertainty only", rep["uncertainty"]["method"])
        self.assertTrue(any("uniform density" in s for s in rep["systematic_not_bootstrappable"]))

    def test_b12_collision_proxy_refused_by_default(self):
        mesh = box_mesh([0, 0, 0], [0.2, 0.2, 0.2])
        with self.assertRaises(InertiaReconstructionError):
            GeometryMassModel.from_validated_mesh(mesh, "uniform_solid", "prismatic", [1, 0, 0], None,
                                                  source_provenance=PROV, is_collision_proxy=True)
        mm = GeometryMassModel.from_validated_mesh(mesh, "uniform_solid", "prismatic", [1, 0, 0], None,
                                                   source_provenance=PROV, is_collision_proxy=True,
                                                   allow_collision_proxy=True, reflected_actuator_inertia=PASSIVE)
        self.assertTrue(mm.reconstruct(100.0)["provenance"].endswith("_collision_proxy_preview"))
        self.assertTrue(any("PREVIEW" in a for a in mm.assumptions))

    def test_b13_static_base_mass_is_never_fabricated(self):
        from forceartgs_g9.export.dynamics_v2 import physics_spec_v1
        from forceartgs_g9.testing import mechanical_spec, static_base_for
        obj = make_object("prismatic")
        table = {t: dict(status="identified") for t in ("inertia", "viscous", "coulomb")}
        table.update(static=dict(status="disabled_by_user"), bias=dict(status="disabled_by_user"),
                     gravity=dict(status="geometry_coupled", identifiable=True))
        recon = obj.mass_model.reconstruct(obj.density, None, fitted_M_eff=obj.M_true)
        with tempfile.TemporaryDirectory() as td:
            ms = mechanical_spec(obj, Path(td), static_base_for(obj), 0, with_static=False)
            spec, missing = physics_spec_v1(table=table, recon_capture=recon, mech=ms, anchor_frame=0,
                                            g9_gravity=obj.gravity, joint_type="prismatic", phys=dict(c=4.0, Fc=1.5))
            self.assertIsNone(spec)
            self.assertTrue(any("SCHEMA_GAP" in m for m in missing))
            ms2 = mechanical_spec(obj, Path(td), static_base_for(obj), 0)
            spec2, missing2 = physics_spec_v1(table=table, recon_capture=recon, mech=ms2, anchor_frame=0,
                                              g9_gravity=obj.gravity, joint_type="prismatic",
                                              phys=dict(c=4.0, Fc=1.5))
            self.assertEqual(missing2, [])
            self.assertEqual(spec2["links"]["static_link"]["provenance"], "measured")
            self.assertEqual(spec2["links"]["moving_link"]["provenance"], "estimated")
            self.assertIsNone(spec2["joint"]["armature"])


if __name__ == "__main__":
    unittest.main()
