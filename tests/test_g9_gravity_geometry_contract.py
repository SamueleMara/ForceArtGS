"""Gravity-projection / mass-geometry / reference-pose contract (G9 gravity corrections spec, sections 3, 8A, 8B).
Synthetic analytic truth only: a PASS means the law and its implementation agree, not that a real fit is physical."""
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9 import physics_lib  # noqa: E402
from forceartgs_g9.dynamics.general_1dof import (GeneralTerms, GravityField, RigidBodyGravity, rotation,  # noqa: E402
                                                 simulate_general)
from forceartgs_g9.dynamics.geometry_inertia import PASSIVE, GeometryMassModel  # noqa: E402
from forceartgs_g9.dynamics.mass_properties import TriMesh, transform_inertial, unit_density_properties  # noqa: E402
from forceartgs_g9.estimators.model_spec import ModelSpec  # noqa: E402
from forceartgs_g9.synthetic import ZOH, box_mesh  # noqa: E402
from forceartgs_g9.validation import gravity_geometry_audit as GA  # noqa: E402

physics_lib.load()
PROV = dict(source="test")
G = np.array([0.0, 0.0, -9.81])
rng = np.random.default_rng(7)


def l_part():
    a = box_mesh([0.0, 0.0, 0.0], [0.4, 0.05, 0.3])
    b = box_mesh([0.3, 0.05, 0.0], [0.4, 0.25, 0.1])
    return TriMesh.from_arrays(np.r_[a.vertices, b.vertices], np.r_[a.faces, b.faces + len(a.vertices)])


def mm(mesh, jt="revolute", axis=(0, 0, 1), origin=(-0.05, 0.02, 0.0), mode="uniform_solid", q_ref=0.0, refl=PASSIVE):
    return GeometryMassModel.from_validated_mesh(mesh, mode, jt, axis, None if jt == "prismatic" else origin,
                                                 source_provenance=PROV, reflected_actuator_inertia=refl, q_ref=q_ref)


def rand_unit():
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


# ===================================================================================== A. geometry and reference
class GeometryReference(unittest.TestCase):
    def test_A01_axis_normalisation(self):
        m1, m2 = mm(l_part(), axis=(0, 0, 1)), mm(l_part(), axis=(0, 0, 7.5))
        self.assertAlmostEqual(m1.M1, m2.M1, places=12)
        np.testing.assert_allclose(m1.gravity_unit(G).G([0.1, 0.7]), m2.gravity_unit(G).G([0.1, 0.7]), atol=1e-15)

    def test_A02_origin_shift_along_axis_invariance(self):
        a = rand_unit()
        o = np.array([0.1, -0.2, 0.05])
        b0, b1 = mm(l_part(), axis=a, origin=o), mm(l_part(), axis=a, origin=o + 0.9 * a)
        self.assertAlmostEqual(b0.M1, b1.M1, places=12)
        np.testing.assert_allclose(b0.r_perp, b1.r_perp, atol=1e-14)
        q = np.linspace(-2, 2, 9)
        np.testing.assert_allclose(b0.gravity_unit(G).G(q), b1.gravity_unit(G).G(q), atol=1e-14)

    def test_A03_com_on_axis_zero_lever(self):
        mesh = box_mesh([-0.1, -0.1, 0.0], [0.1, 0.1, 0.5])                 # COM on the z line through 0
        m = mm(mesh, axis=(0.3, 0.2, 1.0), origin=(0.0, 0.0, 0.25))
        a = m.axis
        c = m.props.c1 - m.origin
        self.assertLess(np.linalg.norm(c - (c @ a) * a), 1e-12)
        self.assertLess(m.lever_m, 1e-12)
        self.assertLess(np.max(np.abs(m.gravity_unit(G).G(np.linspace(-3, 3, 13)))), 1e-12)
        self.assertAlmostEqual(m.M1, float(a @ m.props.J1 @ a), places=12)

    def test_A04_perpendicular_offset_vector(self):
        mesh = box_mesh([0.2, -0.05, 0.0], [0.6, 0.05, 0.8])                 # COM at (0.4, 0, 0.4)
        m = mm(mesh, axis=(0, 0, 1), origin=(0.0, 0.0, 0.0))
        np.testing.assert_allclose(m.r_perp, [0.4, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(m.lever_m, 0.4, places=12)

    def test_A05_full_tensor_and_parallel_axis(self):
        p = unit_density_properties(l_part(), "uniform_solid")
        J = np.asarray(p.J1)
        self.assertLess(np.max(np.abs(J - J.T)), 1e-14)
        self.assertTrue(np.all(np.linalg.eigvalsh(J) > 0))
        self.assertGreater(abs(J[0, 1]) + abs(J[0, 2]) + abs(J[1, 2]), 1e-6)    # genuinely off-diagonal
        for _ in range(5):
            a, o = rand_unit(), rng.normal(0, 0.3, 3)
            r = p.c1 - o
            rp = r - (r @ a) * a
            self.assertAlmostEqual(p.axis_inertia(a, o), float(a @ J @ a + p.m1 * rp @ rp), places=12)

    def test_A06_watertight_winding(self):
        mesh = box_mesh([0, 0, 0], [0.3, 0.2, 0.1])
        h = GA.hull_diagnostics(mesh)
        self.assertTrue(h["watertight"])
        self.assertEqual(h["connected_components"], 1)
        self.assertAlmostEqual(h["signed_volume_m3"], 0.006, places=12)
        self.assertIn("outward", h["winding"])
        inv = TriMesh.from_arrays(mesh.vertices, mesh.faces[:, [0, 2, 1]])
        self.assertIn("INWARD", GA.hull_diagnostics(inv)["winding"])

    def test_A07_solid_and_shell_units(self):
        a, b, c = 0.3, 0.2, 0.1
        mesh = box_mesh([0, 0, 0], [a, b, c])
        s, h = unit_density_properties(mesh, "uniform_solid"), unit_density_properties(mesh, "uniform_shell")
        self.assertAlmostEqual(s.m1, a * b * c, places=12)
        self.assertAlmostEqual(h.m1, 2 * (a * b + b * c + c * a), places=12)
        np.testing.assert_allclose(s.c1, h.c1, atol=1e-12)
        rep = GA.mass_com_inertia_audit(mesh, "revolute", (0, 0, 1), (0, 0, 0), G, 0.0)
        self.assertEqual(rep["modes"]["uniform_solid"]["m1_units"], "m^3")
        self.assertEqual(rep["modes"]["uniform_shell"]["m1_units"], "m^2")
        self.assertEqual(rep["modes"]["uniform_solid"]["density_units"], "kg/m^3")
        self.assertEqual(rep["modes"]["uniform_shell"]["density_units"], "kg/m^2")

    def test_A08_T_Wc_P_applied_and_proxy_tentative(self):
        from plyfile import PlyData, PlyElement
        from forceartgs_g9.dynamics.geometry_proxy import build_proxy, load_moving_points
        P = rng.uniform(-0.1, 0.1, (400, 3)) * [3, 1, 4]
        R = rotation(rand_unit(), 0.7)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, [0.5, -0.2, 0.3]
        with tempfile.TemporaryDirectory() as td:
            v = np.array([tuple(x) + (5.0,) for x in P], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                                                ("opacity", "f4")])
            PlyData([PlyElement.describe(v, "vertex")]).write(f"{td}/m.ply")
            hj = Path(td) / "h.json"
            hj.write_text(json.dumps(dict(parts=dict(moving=dict(ply="m.ply", T_Wc_P=T.tolist())),
                                          source=dict(canonical_anchor_original_frame=3))))
            X, op, _ = load_moving_points(hj)
            np.testing.assert_allclose(X, P.astype("f4").astype(float) @ R.T + T[:3, 3], atol=1e-6)
            px = build_proxy(hj, "solid_hull", td)
            self.assertEqual(px["provenance"]["status"], "TENTATIVE_GEOMETRIC_PREVIEW")
            rep = GA.mass_com_inertia_audit(px["mesh"], "revolute", (0, 0, 1), (0, 0, 0), G, 0.0)
            self.assertEqual(rep["flags"], ["TENTATIVE_GEOMETRIC_PREVIEW", "PHYSICAL_MASS_UNVERIFIED"])

    def test_A09_nonzero_q_ref(self):
        mesh = l_part()
        a, o, qr = np.array([0.1, 0.2, 1.0]) / np.linalg.norm([0.1, 0.2, 1.0]), np.array([-0.05, 0.02, 0.0]), 0.37
        m_ref = mm(mesh, axis=a, origin=o, q_ref=qr)                       # geometry measured at q = q_ref
        c0 = o + rotation(a, -qr) @ (m_ref.props.c1 - o)                    # the same body re-posed at q = 0
        rb0 = RigidBodyGravity("revolute", tuple(a), tuple(o), tuple(c0), m_ref.props.m1, tuple(G), 0.0)
        q = np.linspace(-1.5, 1.5, 11)
        np.testing.assert_allclose(m_ref.gravity_unit(G).G(q), rb0.G(q), atol=1e-13)
        A, B = m_ref.gravity_unit(G).basis_coefficients()
        np.testing.assert_allclose(A * np.cos(q) + B * np.sin(q), rb0.G(q), atol=1e-13)
        self.assertGreater(np.max(np.abs(mm(mesh, axis=a, origin=o).gravity_unit(G).G(q) - rb0.G(q))), 1e-4)
        np.testing.assert_allclose(m_ref.gravity_unit(G).com(qr), m_ref.props.c1, atol=1e-14)

    def test_A10_paired_gauge_reversal(self):
        a, o = rand_unit(), np.array([0.1, 0.0, 0.0])
        c = np.array([0.4, 0.3, -0.1])
        r1 = RigidBodyGravity("revolute", tuple(a), tuple(o), tuple(c), 2.0, tuple(G), 0.2)
        r2 = RigidBodyGravity("revolute", tuple(-a), tuple(o), tuple(c), 2.0, tuple(G), -0.2)
        for q in rng.uniform(-2, 2, 6):
            self.assertAlmostEqual(r1.potential(q), r2.potential(-q), places=12)        # same physical pose
            self.assertAlmostEqual(r1.G(q), -r2.G(-q), places=12)                       # signed load reverses
        p1 = RigidBodyGravity("prismatic", tuple(a), (0, 0, 0), tuple(c), 2.0, tuple(G))
        p2 = RigidBodyGravity("prismatic", tuple(-a), (0, 0, 0), tuple(c), 2.0, tuple(G))
        self.assertAlmostEqual(p1.potential(0.3), p2.potential(-0.3), places=12)
        self.assertAlmostEqual(p1.G(0.3), -p2.G(-0.3), places=12)

    def test_A11_rigid_rotation_of_everything_is_invariant(self):
        mesh, a, o = l_part(), rand_unit(), np.array([0.05, -0.1, 0.02])
        R = rotation(rand_unit(), 1.1)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, [0.3, 0.1, -0.2]
        m0 = mm(mesh, axis=a, origin=o)
        m1 = mm(mesh.transformed(T), axis=R @ a, origin=R @ o + T[:3, 3])
        self.assertAlmostEqual(m0.M1, m1.M1, places=12)
        q = np.linspace(-2, 2, 7)
        np.testing.assert_allclose(m0.gravity_unit(G).G(q), m1.gravity_unit(R @ G).G(q), atol=1e-12)

    def test_A12_malformed_transform_fails_closed(self):
        bad = np.eye(4)
        bad[0, 0] = 1.3
        with self.assertRaises(GA.ReferencePoseError):
            GA.validate_transform(bad)
        with self.assertRaises(GA.ReferencePoseError):
            GA.validate_transform(np.full((4, 4), np.nan))
        refl = np.diag([1.0, 1.0, -1.0, 1.0])
        with self.assertRaises(GA.ReferencePoseError):
            GA.validate_transform(refl)

    def _handoff(self, td, q_anchor=0.0, anchor=90, frame_map_q=None, axis=(0, 0, 1)):
        h = dict(source=dict(canonical_anchor_original_frame=anchor, canonical_anchor_timestamp_s=3.6),
                 joint=dict(q_anchor=q_anchor, axis_Wc=list(axis), origin_Wc=[0.1, 0.2, 0.0], reference_frame="Wc"),
                 parts=dict(moving=dict(T_Wc_P=np.eye(4).tolist(), ply_sha256="x")), frames=dict(file="frame_map.json"))
        if q_anchor is None:
            h["joint"].pop("q_anchor")
        p = Path(td) / "articulated_3dgs_handoff.json"
        p.write_text(json.dumps(h))
        (Path(td) / "frame_map.json").write_text(json.dumps(dict(
            anchor_original_frame=anchor, original_frame_definition="RGB-D frame id",
            rows=[dict(original_frame=anchor, q=q_anchor if frame_map_q is None else frame_map_q)])))
        md = Path(td) / "metadata"
        md.mkdir(exist_ok=True)
        (md / "cameras.json").write_text(json.dumps(dict(coordinate_frame="base_link", calibration_status="X")))
        return p, md

    def _auth(self, q_at_anchor=0.0):
        return types.SimpleNamespace(rgbd_frame_index=np.array([90, 92, 94]), source_frame_index=np.array([180, 184, 188]),
                                     q=np.array([q_at_anchor, -0.1, -0.2]), q_valid=np.ones(3, bool),
                                     q_sigma=np.full(3, 2e-4), axis_world=np.array([0, 0, 1.0]),
                                     origin_world=np.array([0.1, 0.2, 0.0]), world_frame="Wc")

    def test_A13_reference_pose_proven_from_handoff_and_authority(self):
        with tempfile.TemporaryDirectory() as td:
            hp, md = self._handoff(td)
            r = GA.reference_pose_alignment(hp, self._auth(), md)
            self.assertEqual(r["status"], "PASS", r["reasons"])
            self.assertEqual(r["q_ref"], 0.0)
            self.assertEqual(r["authority_source_row_at_anchor"], 180)             # RGB-D frame 90 <-> row 180
            self.assertTrue(r["frames"]["T_B_Wc"].startswith("IDENTITY"))
            bad = GA.reference_pose_alignment(hp, self._auth(q_at_anchor=0.05), md)
            self.assertEqual(bad["status"], "FAIL")
            hp2, _ = self._handoff(td, frame_map_q=0.3)
            self.assertEqual(GA.reference_pose_alignment(hp2, self._auth(), md)["status"], "FAIL")
            hp3, _ = self._handoff(td, axis=(0, 0.1, 1))
            self.assertEqual(GA.reference_pose_alignment(hp3, self._auth(), md)["status"], "FAIL")

    def test_A14_missing_reference_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            hp, md = self._handoff(td, q_anchor=None)
            with self.assertRaises(GA.ReferencePoseError):
                GA.reference_pose_alignment(hp, self._auth(), md)
            hp, md = self._handoff(td, anchor=500)
            with self.assertRaises(GA.ReferencePoseError):
                GA.reference_pose_alignment(hp, self._auth(), md)

    def test_A15_core_refuses_unverified_reference_pose(self):
        from forceartgs_g9.config import load_dynamics_config
        from forceartgs_g9.core import G9Inputs, _mass_model
        cfg = load_dynamics_config(None)
        cfg["mass_model"].update(mode="uniform_solid", reflected_actuator_inertia=PASSIVE)
        inp = G9Inputs("real", "revolute", np.array([0, 0, 1.0]), np.zeros(3), "IDENTIFIED", True, {}, [], [], {},
                       mass_mesh=l_part(), mass_mesh_provenance=PROV)
        m, why = _mass_model(inp, cfg, print)
        self.assertIsNone(m)
        self.assertIn("MASS_MODEL_REFERENCE_POSE_UNVERIFIED", why)
        inp.reference_pose = dict(status="PASS", q_ref=0.25)
        m, why = _mass_model(inp, cfg, print)
        self.assertIsNone(why)
        self.assertEqual(m.q_ref, 0.25)


# ===================================================================================== B. gravity physics
class GravityPhysics(unittest.TestCase):
    def test_B01_G_is_dU_dq_random(self):
        for _ in range(20):
            a, o, c = rand_unit(), rng.normal(0, 0.3, 3), rng.normal(0, 0.5, 3)
            rb = RigidBodyGravity("revolute", tuple(a), tuple(o), tuple(c), 1.7, tuple(G), float(rng.normal()))
            q, h = float(rng.uniform(-3, 3)), 1e-6
            self.assertAlmostEqual(rb.G(q), (rb.potential(q + h) - rb.potential(q - h)) / (2 * h), places=6)

    def test_B02_vertical_hinge_no_gravity(self):
        for s in (1, -1):
            rb = RigidBodyGravity("revolute", (0, 0, s), (0.1, 0.2, 0.0), (0.7, -0.4, 0.3), 5.0, tuple(G), 0.4)
            self.assertLess(np.max(np.abs(rb.G(np.linspace(-3, 3, 25)))), 1e-12)

    def test_B03_tilted_amplitude_m_g_sinbeta_ell(self):
        for beta in (2.0, 6.03, 30.0, 90.0):
            b = math.radians(beta)
            a = np.array([math.sin(b), 0.0, math.cos(b)])
            m = mm(box_mesh([0.2, 0.0, 0.0], [0.6, 0.05, 0.5]), axis=a, origin=(0, 0, 0))
            amp = m.gravity_unit(G).amplitude_bound()
            self.assertAlmostEqual(amp, m.props.m1 * 9.81 * math.sin(b) * m.lever_m, places=10)
            grid = m.gravity_unit(G).G(np.linspace(-math.pi, math.pi, 20001))
            self.assertAlmostEqual(float(np.max(np.abs(grid))), amp, places=6)

    def test_B04_revolute_sign(self):
        rb = RigidBodyGravity("revolute", (1, 0, 0), (0, 0, 0), (0, 0.5, 0), 2.0, tuple(G))
        self.assertAlmostEqual(rb.G(0.0), 2.0 * 9.81 * 0.5, places=12)      # +q lifts the COM: load resists
        self.assertAlmostEqual(rb.G(math.pi / 2), 0.0, places=12)          # COM straight above the hinge
        self.assertAlmostEqual(rb.G(-0.3), -rb.G(math.pi + 0.3), places=12)

    def test_B05_prismatic_projection(self):
        mesh = box_mesh([0, 0, 0], [0.3, 0.2, 0.1])
        for axis, expect in (((1, 0, 0), 0.0), ((0, 0, 1), 9.81), ((0, 0, -1), -9.81), ((0.6, 0, 0.8), 9.81 * 0.8)):
            m = mm(mesh, jt="prismatic", axis=axis, q_ref=0.4)
            self.assertAlmostEqual(m.gravity_unit(G).G(0.0) / m.props.m1, expect, places=10)
            self.assertAlmostEqual(m.gravity_unit(G).G(1.3), m.gravity_unit(G).G(-2.0), places=12)

    def test_B06_inertia_and_gravity_linear_in_one_density(self):
        m = mm(l_part(), axis=(0.1, 0.0, 1.0))
        spec = ModelSpec(GeneralTerms(static=False, coulomb=False), "revolute", "geometry_coupled", m,
                         GravityField(tuple(G), True, "test"))
        p1, p2 = spec.physical(np.array([100.0, 0.5])), spec.physical(np.array([250.0, 0.5]))
        self.assertAlmostEqual(p2["M"] / p1["M"], 2.5, places=12)
        self.assertAlmostEqual(p2["gravity"]["A"] / p1["gravity"]["A"], 2.5, places=12)
        self.assertAlmostEqual(p2["gravity"]["amplitude"] / p1["gravity"]["amplitude"], 2.5, places=12)
        self.assertEqual(spec.names, ["alpha", "c"])

    def test_B07_tensor_frame_transform(self):
        p = unit_density_properties(l_part(), "uniform_solid")
        R = rotation(rand_unit(), 0.9)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, [0.1, 0.2, 0.3]
        c2, J2 = transform_inertial(T, p.c1, p.J1)
        p2 = unit_density_properties(l_part().transformed(T), "uniform_solid")
        np.testing.assert_allclose(c2, p2.c1, atol=1e-12)
        np.testing.assert_allclose(J2, p2.J1, atol=1e-12)

    def test_B08_reflected_inertia_kinetic_only(self):
        m0, m1 = mm(l_part(), axis=(0.1, 0, 1), refl=PASSIVE), mm(l_part(), axis=(0.1, 0, 1), refl=0.5)
        gf = GravityField(tuple(G), True, "test")
        t = GeneralTerms(static=False, coulomb=False)
        s0 = ModelSpec(t, "revolute", "geometry_coupled", m0, gf)
        s1 = ModelSpec(t, "revolute", "geometry_coupled", m1, gf)
        th = np.array([300.0, 0.2])
        self.assertAlmostEqual(s1.physical(th)["M"] - s0.physical(th)["M"], 0.5, places=12)
        self.assertEqual(s1.physical(th)["gravity"]["A"], s0.physical(th)["gravity"]["A"])
        self.assertAlmostEqual(m1.alpha_from_effective(300.0 * m1.M1 + 0.5), 300.0, places=10)

    def test_B09_gravity_switch_exact(self):
        import torch
        from forceartgs_g9.estimators.torch_model import TorchDynamics
        m = mm(l_part(), axis=(0.3, 0, 1))
        spec = ModelSpec(GeneralTerms(static=False, coulomb=False, gravity=False), "revolute", "geometry_coupled", m,
                         GravityField(tuple(G), True, "test"))
        self.assertIsNone(spec.G_fn(np.array([300.0, 0.2])))
        self.assertIsNone(spec.sim(np.array([300.0, 0.2])).G)
        self.assertIsNone(spec.physical(np.array([300.0, 0.2]))["gravity"])
        dyn = TorchDynamics(spec)
        G_t = dyn.terms(dyn.values(dyn.eta0([300.0, 0.2])))[5]
        self.assertEqual(float(G_t(torch.linspace(-2, 2, 9, dtype=torch.float64)).abs().max()), 0.0)

    def test_B10_com_along_axis_irrelevant(self):
        a = np.array([0.2, -0.1, 1.0]) / np.linalg.norm([0.2, -0.1, 1.0])
        T = np.eye(4)
        T[:3, 3] = 0.35 * a
        m0, m1 = mm(l_part(), axis=a), mm(l_part().transformed(T), axis=a)
        q = np.linspace(-2, 2, 9)
        np.testing.assert_allclose(m0.gravity_unit(G).G(q), m1.gravity_unit(G).G(q), atol=1e-13)
        self.assertAlmostEqual(m0.M1, m1.M1, places=12)

    def test_B11_density_and_q_derivatives(self):
        m = mm(l_part(), axis=(0.1, 0.05, 1.0), q_ref=0.2)
        spec = ModelSpec(GeneralTerms(static=False), "revolute", "geometry_coupled", m, GravityField(tuple(G), True, "t"))
        rep = GA.method_gravity_parity(spec, np.array([400.0, 0.3, 0.05]))
        self.assertEqual(rep["status"], "PASS", rep["checks"])
        gu = m.gravity_unit(G)
        for q in (-1.0, 0.2, 1.7):
            h = 1e-6
            dG_fd = (gu.G(q + h) - gu.G(q - h)) / (2 * h)
            A, B = gu.basis_coefficients()
            self.assertAlmostEqual(dG_fd, -A * math.sin(q) + B * math.cos(q), places=6)

    def test_B12_energy_conservation_and_work_balance(self):
        m = mm(l_part(), axis=(0.25, 0.1, 1.0))
        gf = GravityField(tuple(G), True, "test")
        rho = 300.0
        gu = m.gravity_unit(G)
        free = ModelSpec(GeneralTerms(viscous=False, coulomb=False, static=False), "revolute", "geometry_coupled", m, gf)
        t = np.linspace(0, 2.0, 2001)
        r = simulate_general(free.sim(np.array([rho])), ZOH([0.0], [0.0]), t, q0=0.3, v0=0.0, max_step=1e-4)
        Mv = rho * m.M1
        E = 0.5 * Mv * r.qdot ** 2 + rho * np.array([gu.potential(x) for x in r.q])
        self.assertLess(np.ptp(E) / (np.max(np.abs(E - E[0])) + abs(rho * gu.amplitude_bound())), 1e-4)
        c = 0.4
        damp = ModelSpec(GeneralTerms(coulomb=False, static=False), "revolute", "geometry_coupled", m, gf)
        eff = ZOH([0.0, 0.5, 1.0, 1.5], [2.0, -1.0, 0.5, 0.0])
        r2 = simulate_general(damp.sim(np.array([rho, c])), eff, t, q0=0.0, v0=0.0, max_step=1e-4)
        K = 0.5 * Mv * r2.qdot ** 2
        U = rho * np.array([gu.potential(x) for x in r2.q])
        ts, lev = np.asarray(eff.t_s, float), np.asarray(eff.levels, float)
        qk = np.interp(np.r_[ts, t[-1]], t, r2.q)                     # knots are grid points: exact q there
        work = float(np.sum(lev * np.diff(qk))) - np.trapz(c * r2.qdot ** 2, t)   # int Q q' dt exact for ZOH
        self.assertAlmostEqual((K[-1] + U[-1]) - (K[0] + U[0]), work, delta=2e-3 * max(1.0, abs(work)))


if __name__ == "__main__":
    unittest.main()
