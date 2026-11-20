"""G9 Gate D: export through the shared PArtTwin1D builder + engine parity (synthetic, ~4 min; needs the
parttwin_sim env (SAPIEN) and the rigeuler-sim-export venv (MuJoCo, USD) -- skipped otherwise)."""
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9 import physics_lib  # noqa: E402
from forceartgs_g9.core import run_g9  # noqa: E402
from forceartgs_g9.dynamics.general_1dof import GeneralTerms  # noqa: E402
from forceartgs_g9.export import final_build as FB  # noqa: E402
from forceartgs_g9.testing import synthetic_cfg, synthetic_inputs  # noqa: E402
from forceartgs_g9.validation.simulator_parity import term_matrix  # noqa: E402

physics_lib.load()
RUNTIMES = Path(FB.PARTTWIN_PY).is_file() and Path(FB.SIM_PY).is_file()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


@unittest.skipUnless(RUNTIMES, "PArtTwin1D / simulator runtimes not installed")
class GateD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp(prefix="g9-gate-d-")
        cls.case = {}
        for name, kind, terms in (("P", "prismatic", GeneralTerms(static=False)),
                                  ("R", "revolute", GeneralTerms())):
            out = Path(cls.td) / name
            obj, inp, hand = synthetic_inputs(kind, out, terms=terms)
            before = {p.name: sha(p) for p in hand.parent.iterdir()}
            r = run_g9(inp, synthetic_cfg(terms), out / "g9", log=lambda m: None)
            fe = FB.final_physics_export(out / "g9", hand, out / "assets", parity_effort=inp.heldout[0].effort,
                                         log=lambda m: None)
            cls.case[name] = dict(obj=obj, inp=inp, hand=hand, r=r, fe=fe, out=out, before=before,
                                  asset=Path(fe["build"]["asset_dir"]) if fe.get("build") else None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def test_d01_standard_plys_unchanged(self):
        for c in self.case.values():
            self.assertEqual(c["r"]["manifest"]["status"], "PRODUCTION_PHYSICS_SPEC_READY", c["r"]["manifest"]["blocked"])
            self.assertEqual({p.name: sha(p) for p in c["hand"].parent.iterdir()}, c["before"])   # handoff read-only
            self.assertEqual(c["fe"]["asset"]["visual_ply_identical_to_handoff"], dict(static=True, moving=True))
            for part in ("static", "moving"):
                self.assertEqual(FB.payload_sha256(c["asset"] / "visual" / f"{part}.ply"),
                                 FB.payload_sha256(c["hand"].parent / f"{part}_canonical.ply"))

    def test_d02_dynamics_documents_complete(self):
        for c in self.case.values():
            hd = yaml.safe_load((c["out"] / "g9" / "dynamics_handoff.yaml").read_text())
            self.assertEqual(hd["schema"], "forceartgs.g9.dynamics_handoff/v1")
            v2 = hd["parttwin.dynamics/v2"]
            self.assertIn("joint_space_identification", v2)
            li = v2["link_inertial_moving"]
            for k in ("mass_kg", "com_m", "inertia_com_kg_m2", "provenance", "uncertainty", "systematic"):
                self.assertIsNotNone(li[k], k)
            self.assertTrue(li["provenance"].startswith("estimated_uniform"))
            dyn = yaml.safe_load((c["asset"] / "physics" / "dynamics.yaml").read_text())
            mv = dyn["links"]["moving_link"]
            self.assertIn("com_link_m", mv)
            self.assertIn("inertia_com_link_kg_m2", mv)
            # v1 carries one joint provenance (identified damping -> estimated); the measured stops are recorded
            self.assertEqual(dyn["joint"]["limits_provenance"], "physical_stops_estimated")
            spec = yaml.safe_load((c["out"] / "g9" / "physics_spec_v1.yaml").read_text())
            self.assertIn("SCHEMA_GAP", spec["notes"])
            self.assertIn("'measured'", spec["notes"])
            pp = json.loads((c["out"] / "g9" / "physics_parameters.json").read_text())
            for t, e in pp["terms"].items():
                self.assertIn(e["status"], ("identified", "disabled_by_user", "geometry_coupled"), (t, e))

    def test_d03_urdf_inertial_is_com_tensor_and_projects_to_the_hinge_inertia(self):
        for c in self.case.values():
            root = ET.parse(c["asset"] / "maniskill" / "model.urdf").getroot()
            link = [l for l in root.findall("link") if l.get("name") == "moving_link"][0]
            ine = link.find("inertial")
            com = np.array(list(map(float, ine.find("origin").get("xyz").split())))
            m = float(ine.find("mass").get("value"))
            I = ine.find("inertia").attrib
            J = np.array([[float(I["ixx"]), float(I["ixy"]), float(I["ixz"])],
                          [float(I["ixy"]), float(I["iyy"]), float(I["iyz"])],
                          [float(I["ixz"]), float(I["iyz"]), float(I["izz"])]])
            dyn = yaml.safe_load((c["asset"] / "physics" / "dynamics.yaml").read_text())["links"]["moving_link"]
            np.testing.assert_allclose(com, dyn["com_link_m"], atol=1e-12)
            obj = c["obj"]
            self.assertAlmostEqual(m / (obj.density * obj.mass_model.props.m1), 1.0, delta=0.02)
            if obj.joint_type == "revolute":              # joint frame J0: +Z = axis, origin on the hinge
                ez = np.array([0, 0, 1.0])
                I_ax = ez @ (J + m * ((com @ com) * np.eye(3) - np.outer(com, com))) @ ez
                self.assertAlmostEqual(I_ax / obj.M_true, 1.0, delta=0.02)

    def test_d04_no_duplicated_gravity_or_inertia(self):
        for c in self.case.values():
            urdf = (c["asset"] / "maniskill" / "model.urdf").read_text()
            mjcf = (c["asset"] / "mujoco" / "model.xml").read_text()
            self.assertNotIn("armature", urdf)
            self.assertNotIn("armature", mjcf)
            self.assertNotIn("gravcomp", mjcf)
            self.assertNotIn("qfrc", mjcf)
            spec = yaml.safe_load((c["out"] / "g9" / "physics_spec_v1.yaml").read_text())
            self.assertIsNone(spec["joint"]["armature"])
        tm = term_matrix(GeneralTerms(gravity=False), 0.2, 0.4, gravity_bound=3.0, effort_scale=1.0)
        self.assertEqual(tm["mujoco"]["gravity"]["status"], "GRAVITY_CONFIGURATION_MISMATCH")

    def test_d05_mujoco_damping_and_rollout(self):
        c = self.case["P"]
        par = c["fe"]["numerical_parity"]["mujoco"]
        self.assertEqual(par["status"], "PASS", par.get("reason"))
        phys = c["r"]["fit"]["physical"]
        self.assertAlmostEqual(par["engine_readback"]["damping"], phys["c"], places=9)
        self.assertAlmostEqual(par["engine_readback"]["frictionloss"], phys["Fc"], places=9)
        self.assertIn("frictionloss", term_matrix(GeneralTerms(), 1.0, 1.0)["mujoco"]["coulomb"]["mapping"])

    def test_d06_sapien_loads_fixed_root_and_kinematics(self):
        for c in self.case.values():
            v = json.loads((c["asset"] / "export_manifest.json").read_text())["validation"]
            self.assertEqual(v["sapien_parttwin1d"]["status"], "PASS")
            self.assertEqual(v["analytic_kinematics"]["status"], "PASS")
            self.assertLess(v["sapien_parttwin1d"]["pose_max_error"], 1e-5)

    def test_d07_usd_loads_with_unit_conversion(self):
        c = self.case["R"]
        v = json.loads((c["asset"] / "export_manifest.json").read_text())["validation"]["usd"]
        if v["status"] == "NOT_RUN":
            self.skipTest("no USD runtime")
        self.assertEqual(v["status"], "PASS")
        self.assertIn("deg", v["angle_units"])
        tm = term_matrix(GeneralTerms(), 0.25, 0.45)
        self.assertEqual(tm["isaac_usd"]["coulomb"]["status"], "NOT_REPRESENTED")

    def test_d08_static_breakaway_never_all_pass(self):
        c = self.case["R"]
        tm = c["r"]["term_matrix"]
        for e in ("mujoco", "sapien_urdf", "isaac_usd"):
            self.assertEqual(tm[e]["static"]["status"], "NOT_REPRESENTED")
            self.assertFalse(tm[e]["full_model_representable"])
        self.assertFalse(c["fe"]["readiness"]["ENGINE_PARITY_PASS"])
        self.assertFalse(c["fe"]["readiness"]["FULL_PHYSICS_READY"])
        self.assertTrue(c["fe"]["claim"].startswith("NOMINAL_DYNAMICS_EXPORT"))

    def test_d09_withheld_force_parity_reported(self):
        for c in self.case.values():
            for e in ("mujoco", "sapien_urdf"):
                p = c["fe"]["numerical_parity"][e]
                self.assertIn(p["status"], ("PASS", "FAIL", "FAIL_TERM_NOT_REPRESENTED"))
                self.assertTrue(np.isfinite(p["rmse"]))
                self.assertEqual(len(p["q_engine"]), len(p["q_reference"]))
        sap = self.case["P"]["fe"]["numerical_parity"]["sapien_urdf"]
        if sap["status"] != "PASS":                        # the measured SAPIEN loader semantics are recorded
            self.assertIsNotNone(sap["sapien_friction_diagnostic"])

    def test_d10_missing_inputs_are_differentiated_blockers(self):
        td = Path(self.td) / "d10"
        t = GeneralTerms(static=False)
        obj, inp, hand = synthetic_inputs("prismatic", td / "a", terms=t, n_heldout=1, mech=False)
        r = run_g9(inp, synthetic_cfg(t), td / "a" / "g9", log=lambda m: None)
        self.assertTrue(any("mechanical spec" in m for m in r["manifest"]["missing_for_physics_spec_v1"]))
        self.assertIsNone(r["physics_spec_v1"])
        fe = FB.final_physics_export(td / "a" / "g9", hand, td / "a" / "assets", log=lambda m: None)
        self.assertEqual(fe["status"], "BLOCKED_MISSING_PHYSICS")
        obj, inp, hand = synthetic_inputs("prismatic", td / "b", terms=t, n_heldout=1)
        inp.mass_mesh = None
        r = run_g9(inp, synthetic_cfg(t), td / "b" / "g9", log=lambda m: None)
        self.assertTrue(any(b.startswith("BLOCKED_MASS_MODEL") for b in r["manifest"]["blocked"]))
        self.assertEqual(json.loads((td / "b" / "g9" / "inertia_reconstruction_report.json").read_text())
                         ["full_tensor"], None)
        obj, inp, hand = synthetic_inputs("prismatic", td / "c", terms=t, n_heldout=1, force_status="NO_FORCE")
        r = run_g9(inp, synthetic_cfg(t), td / "c" / "g9", log=lambda m: None)
        self.assertTrue(any(b.startswith("BLOCKED_NO_FORCE") for b in r["manifest"]["blocked"]))
        self.assertFalse(any((Path(td) / x / "assets").exists() for x in ("a", "b", "c")))

    def test_d11_replay_uses_live_link_pose_times_T_link_ply(self):
        for c in self.case.values():
            v = json.loads((c["asset"] / "export_manifest.json").read_text())["validation"]["sapien_parttwin1d"]
            self.assertEqual(v["parttwin1d_consumer"]["status"], "PASS")
            self.assertLess(v["parttwin1d_consumer"]["means_max_error_m"], 5e-5)
            ay = yaml.safe_load((c["asset"] / "asset.yaml").read_text())
            self.assertIn("never re-animate with q", ay["visual"]["render_rule"])

    def test_d12_provenance_survives_and_tampering_is_detected(self):
        from forceartgs_g9.stage import run_final_physics_export_stage
        c = self.case["P"]
        man = json.loads((c["out"] / "g9" / "g9_manifest.json").read_text())
        self.assertEqual(man["physics_library"]["commit"], physics_lib.PINNED_COMMIT)
        self.assertEqual(man["input_hashes"]["visual_handoff_sha256"], sha(c["hand"]))
        self.assertIn(man["input_signature"][:12], c["fe"]["build"]["asset_dir"])
        b = c["fe"]["build"]
        self.assertIsNotNone(b["parttwin"]["head"])
        run = Path(self.td) / "d12_run"
        (run / "parttwin_handoff").mkdir(parents=True)
        shutil.copytree(c["hand"].parent, run / "parttwin_handoff" / "accepted")
        shutil.copytree(c["out"] / "g9", run / "dynamics" / "g9" / "v000")
        (run / "selection.json").write_text(json.dumps(dict(selected="prismatic")))
        h = run / "parttwin_handoff" / "accepted" / "articulated_3dgs_handoff.json"
        h.write_text(h.read_text().replace('"synthetic"', '"edited"', 1))
        res = run_final_physics_export_stage(run)
        self.assertEqual(res["status"], "BLOCKED_STALE_DYNAMICS")


if __name__ == "__main__":
    unittest.main()
