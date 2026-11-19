"""Gate D of the multimodal TCP spec: authority handoff, stage order, status semantics, config, provenance."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipelines"))
from estimotion_2part.joint_discovery.config import load_joint_discovery_config  # noqa: E402
from estimotion_2part.joint_discovery.contact_modes import ContactSegmentError, load_contact_segments  # noqa: E402
from estimotion_2part.joint_discovery.integration_contract import authority_from_discovery  # noqa: E402
from estimotion_2part.joint_discovery.multimodal_frontend import run_multimodal_frontend  # noqa: E402
from estimotion_2part.joint_discovery.physical_selection import PhysicalDecision  # noqa: E402
from estimotion_2part.joint_discovery.synthetic import make_contact_record, make_scene, make_tcp_contact  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_joint_multimodal_frontend import cfg  # noqa: E402


def fake_scene(td, sc):
    meta = Path(td) / "scene" / "metadata"
    meta.mkdir(parents=True)
    cams = [dict(name=c, fx=k.K[0, 0], fy=k.K[1, 1], cx=k.K[0, 2], cy=k.K[1, 2], width=k.width, height=k.height,
                 pose_convention="opencv_optical_camera_to_world", coordinate_frame="base_link",
                 pose_matrix_4x4=k.T_WC.tolist()) for c, k in sc.calibs.items()]
    (meta / "cameras.json").write_text(json.dumps(dict(cameras=cams)))
    fm = sc.obs.frame_map
    rows = ["frame_index,source_frame_index,timestamp_s"] + [f"{a},{b},{c}" for a, b, c in zip(
        fm.rgbd_frame_index, fm.source_frame_index, fm.timestamp_s)]
    (meta / "frame_timestamps.csv").write_text("\n".join(rows) + "\n")
    return meta.parent


class GateD(unittest.TestCase):
    def test_d02_authority_is_built_from_the_accepted_frontend_state(self):        # E2E (~1 min)
        sc = make_scene("revolute", seed=1, T=30, n_moving=150, max_angle_deg=30)
        rec, cal, doc = make_contact_record(sc, "stick_hard", verified=False)
        with tempfile.TemporaryDirectory() as td:
            scene = fake_scene(td, sc)
            run_multimodal_frontend(sc.obs, sc.calibs, cfg(iters=1), Path(td) / "mm", scene=scene, rec=rec,
                                    segments_source=doc, wrench_calibration=cal, log=lambda m: None)
            acc = Path(td) / "mm" / "accepted"
            a = authority_from_discovery(acc, "revolute", scene, list(sc.calibs), decision_status="identified")
            cj = json.loads((acc / "joint_candidates.json").read_text())["candidates"]["revolute"]
            ax = np.asarray(cj["joint_axis_world"])
            self.assertAlmostEqual(abs(float(a.axis_world @ ax)), 1.0, places=9)        # same joint, no stale source
            self.assertIn("joint_candidates.json", a.source_hashes)
            self.assertEqual(Path(a.derived_from["discovery_run"]), acc)
            # exact virtual-camera relation T_M0,C(t) = A(q_t)^-1 T_WC for every camera/time (D03)
            cam = list(a.cameras)[0]
            for t in (0, 10, 29):
                np.testing.assert_allclose(a.virtual_camera(cam, t), np.linalg.inv(a.transform(t)) @ a.cameras[cam].T_WC,
                                           atol=1e-12)

    def test_d04_frontend_runs_before_any_gaussian_training(self):
        import joint_aware_reconstruction as jar
        S = list(jar.STAGES)
        self.assertLess(S.index("multimodal_refine"), S.index("select"))
        for later in ("authority", "dense", "build", "parts", "blend", "art"):
            self.assertLess(S.index("multimodal_refine"), S.index(later))
        self.assertLess(S.index("select"), S.index("parts"))

    def test_d06_contact_mode_is_not_a_joint_identity(self):
        from test_joint_physical_selection_v2 import evidence
        from estimotion_2part.joint_discovery.physical_selection import select_physical_joint
        with self.assertRaises(ValueError):
            PhysicalDecision("TYPE_UNOBSERVABLE", "prismatic", True, True, None, [], {}, {}, [])
        stick = dict(contact=dict(prismatic_consistent_stick=True, assumption_failed_segments=[]),
                     force=dict(status="ABSTAIN"))
        # a confirmed stick consistent with a slide, but a short arc is NOT excluded -> still unobservable
        d = select_physical_joint(**evidence(implied=2.0, ub=1.5, mdr=1.2), robot_evidence=stick)
        self.assertEqual(d.physical_joint_status, "TYPE_UNOBSERVABLE")
        self.assertFalse(d.physics_joint_export_ready)
        # only together with the excluded plausible hinges may it replace an unobservable guide
        d2 = select_physical_joint(**evidence(implied=20.0, ub=1.0, mdr=1.0), robot_evidence=stick)
        self.assertEqual(d2.physical_joint_status, "IDENTIFIED_PRISMATIC")
        bad = dict(contact=dict(prismatic_consistent_stick=True, assumption_failed_segments=["seg0"]))
        d3 = select_physical_joint(**evidence(implied=20.0, ub=1.0, mdr=1.0), robot_evidence=bad)
        self.assertNotEqual(d3.physical_joint_status, "IDENTIFIED_PRISMATIC")
        self.assertIn("CONTACT_MODE_CONTRADICTED", d3.failure_codes)

    def test_d08_config_defaults_and_validation(self):
        c = load_joint_discovery_config({"mode": "calibrated_joint_aware", "comparison": {"selection":
                                         "observable_physical_v2"}, "multimodal_frontend": {"enabled": True}})
        self.assertEqual(c.multimodal_frontend.contact.default_mode, "sliding")
        legacy = load_joint_discovery_config(None)
        self.assertNotIn("multimodal_frontend", legacy.to_dict())
        with self.assertRaises(ValueError):
            load_joint_discovery_config({"multimodal_frontend": {"enabled": True}})       # needs selection v2
        with self.assertRaises(ValueError):
            load_joint_discovery_config({"mode": "calibrated_joint_aware",
                                         "comparison": {"selection": "observable_physical_v2"},
                                         "multimodal_frontend": {"force": {"require_verified_wrench": False}}})
        sc = make_tcp_contact("revolute", "stick_hard")
        doc = dict(sc["segments_doc"])
        doc["segments"] = [dict(doc["segments"][0], slip_mode="stick_soft")]          # per-interval override
        segs, _ = load_contact_segments(doc, sc["frame_map"], sc["p_tcp"])
        self.assertEqual(segs[0].slip_mode, "stick_soft")
        doc["segments"] = [dict(doc["segments"][0], slip_mode="rigid_grasp")]
        with self.assertRaises(ContactSegmentError):
            load_contact_segments(doc, sc["frame_map"], sc["p_tcp"])

    def test_d09_export_requires_physics_readiness_and_records_multimodal_provenance(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import export_parttwin_handoff as ex
        from export_parttwin_handoff import HandoffAbstain, physical_provenance
        with self.assertRaises(HandoffAbstain):
            physical_provenance(dict(schema_version=2, status="TYPE_UNOBSERVABLE", physics_joint_export_ready=False),
                                "PASSED")
        self.assertIn("multimodal_provenance", Path(ex.__file__).read_text())


if __name__ == "__main__":
    unittest.main()
