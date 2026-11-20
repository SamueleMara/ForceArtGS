"""Gate C/D of the multimodal TCP spec: the iterative frontend end to end (synthetic, ~1 min per E2E test)."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.config import JointDiscoveryConfig  # noqa: E402
from estimotion_2part.joint_discovery.contact_modes import ContactRows, load_contact_segments  # noqa: E402
from estimotion_2part.joint_discovery.multimodal_frontend import Frontend, run_multimodal_frontend  # noqa: E402
from estimotion_2part.joint_discovery.synthetic import (SplatImageSource, make_contact_record,  # noqa: E402
                                                        make_scene, make_tcp_contact)


def cfg(mech="none", iters=2):
    c = JointDiscoveryConfig(mode="calibrated_joint_aware")
    c.fit.em_iterations, c.fit.n_seeds = 3, 3
    c.comparison.selection = "observable_physical_v2"
    c.comparison.uncertainty_method = "cluster_robust"
    c.comparison.remote_hinge_policy = "observability"
    c.selection_v2.null_patches = 20
    c.multimodal_frontend.enabled = True
    c.multimodal_frontend.max_iterations = iters
    c.multimodal_frontend.force.mechanical_model = mech
    return c.validate()


def run(sc, rec=None, doc=None, cal=None, c=None, images=None):
    with tempfile.TemporaryDirectory() as td:
        sel = run_multimodal_frontend(sc.obs, sc.calibs, c or cfg(), Path(td) / "mm", rec=rec, segments_source=doc,
                                      wrench_calibration=cal, images=images, log=lambda m: None)
        acc = Path(td) / "mm" / "accepted"
        val = json.loads((acc / "validation_report.json").read_text())
        force = json.loads((acc / "force_candidate_comparison.json").read_text())
        files = sorted(p.name for p in acc.iterdir())
        cands = json.loads((acc / "joint_candidates.json").read_text())
    return sel, val, force, files, cands


class Frontend_E2E(unittest.TestCase):
    def test_c11_robot_absent_visual_path(self):
        sc = make_scene("revolute", seed=1, T=30, n_moving=150, max_angle_deg=30)
        sel, val, force, files, _ = run(sc)
        self.assertEqual(sel["physical_joint_status"], "IDENTIFIED_REVOLUTE")
        self.assertFalse(sel["contact_evidence_used"])
        self.assertEqual(force["status"], "ABSTAIN")
        for f in ("joint_candidates.json", "joint_states.npz", "part_support", "joint_selection_v2.json",
                  "authority_source_manifest.json", "canonical_init"):
            self.assertIn(f, files)

    def test_c12_tcp_used_force_invalid_abstains(self):
        sc = make_scene("revolute", seed=1, T=30, n_moving=150, max_angle_deg=30)
        rec, cal, doc = make_contact_record(sc, "stick_hard", verified=False)
        sel, val, force, _, _ = run(sc, rec, doc, cal)
        self.assertTrue(sel["contact_evidence_used"])
        self.assertFalse(sel["force_evidence_used"])
        self.assertEqual(force["status"], "ABSTAIN")
        r0 = val["rounds"][0]
        self.assertGreater(r0["fit"]["hook"]["n_contact_rows"], 0)
        self.assertTrue(r0["fit"]["tcp_force_parameter_deltas"])         # TCP changed the joint parameters
        self.assertEqual(sel["physical_joint_status"], "IDENTIFIED_REVOLUTE")

    def test_c05_hidden_guide_drawer_with_confirmed_stick(self):
        sc = make_scene("prismatic", seed=2, T=30, n_moving=150)
        rec, cal, doc = make_contact_record(sc, "stick_hard", force="min_effort", verified=True)
        sel, _, force, _, _ = run(sc, rec, doc, cal, c=cfg("min_effort_tangent"))
        self.assertEqual(sel["physical_joint_status"], "IDENTIFIED_PRISMATIC", sel["reasons"])
        self.assertTrue(sel["contact_evidence_used"])
        no_robot, _, _, _, _ = run(sc)                                   # same data without the robot: no rails seen
        self.assertEqual(no_robot["physical_joint_status"], "TYPE_UNOBSERVABLE")

    def test_c06_wrong_hard_stick_on_sliding_lid_is_detected(self):
        sc = make_scene("revolute", seed=1, T=30, n_moving=150, max_angle_deg=30)
        rec, cal, doc = make_contact_record(sc, "sliding", verified=False)
        doc["segments"][0]["slip_mode"] = "stick_hard"                   # operator declared the wrong mode
        sel, val, _, _, cands = run(sc, rec, doc, cal)
        self.assertIn("CONTACT_MODE_CONTRADICTED", sel["failure_codes"])
        self.assertFalse(sel["physics_joint_export_ready"])
        self.assertNotEqual(sel["physical_joint_status"], "IDENTIFIED_PRISMATIC")
        self.assertTrue(val["rounds"][0]["fit"]["hook"].get("excluded_contradicted_segments"))
        ax = np.asarray(cands["candidates"]["revolute"]["joint_axis_world"])
        self.assertLess(np.degrees(np.arccos(abs(ax @ sc.gt_axis))), 5.0)  # the hinge was not dragged

    def test_d06_stick_on_remote_short_arc_is_not_identified(self):
        sc = make_scene("revolute", seed=8, T=30, n_moving=150, axis=[0, 0, 1], hinge_offset=[3.0, 0.0, 0.05],
                        max_angle_deg=1.9)
        rec, cal, doc = make_contact_record(sc, "stick_hard", anchor=[0.0, -0.05, 0.13], verified=False)
        sel, _, _, _, _ = run(sc, rec, doc, cal)
        self.assertFalse(sel["physical_joint_status"].startswith("IDENTIFIED_"), sel["physical_joint_status"])
        self.assertFalse(sel["physics_joint_export_ready"])

    def test_c04_track_recovery_iteration_beats_one_pass(self):
        sc = make_scene("revolute", seed=5, T=24, n_moving=150, max_angle_deg=12, sigma_px=0.3, dropout=0.8)
        imgs = SplatImageSource(sc, n_texture=8000)
        sel, val, _, _, _ = run(sc, images=imgs, c=cfg(iters=2))
        r = val["rounds"]
        self.assertGreaterEqual(len(r), 2)
        self.assertGreater(r[1]["recovery"]["accepted"], 0)
        self.assertIn("accepted", r[1]["verdict"], r[1]["verdict"])
        self.assertGreaterEqual(val["accepted_round"], 1)


class Frontend_Units(unittest.TestCase):
    def test_d01_heldout_tcp_never_enters_contact_rows(self):
        sc = make_tcp_contact("revolute", "stick_hard")
        segs, _ = load_contact_segments(sc["segments_doc"], sc["frame_map"], sc["p_tcp"])
        train = np.r_[np.ones(20, bool), np.zeros(10, bool)]
        r1 = ContactRows(segs, sc["p_tcp"], train).residuals("revolute", sc["axis"], sc["origin"], sc["q"])
        p2 = sc["p_tcp"].copy()
        p2[~train] += 0.05
        r2 = ContactRows(segs, p2, train).residuals("revolute", sc["axis"], sc["origin"], sc["q"])
        np.testing.assert_array_equal(r1, r2)

    def test_c09_acceptance_and_rollback(self):
        f = Frontend.__new__(Frontend)
        f.mf = cfg().multimodal_frontend

        class Fit:
            candidates = dict(revolute=None, prismatic=None)
        prev = dict(support=dict(M_proposal=np.ones(100, bool)), contact={}, fit=Fit())
        better = dict(support=dict(M_proposal=np.ones(100, bool)), contact={}, fit=Fit())
        common = dict(prev_revolute=1000.0, prev_prismatic=2000.0, new_revolute=900.0, new_prismatic=2000.0, n_obs=50)
        self.assertTrue(f._accept(prev, better, common)[0])
        worse_best = dict(common, new_revolute=1100.0)
        self.assertFalse(f._accept(prev, better, worse_best)[0])
        other_worse = dict(common, new_prismatic=2100.0)                 # the other hypothesis may get worse
        self.assertTrue(f._accept(prev, better, other_worse)[0])
        same = dict(common, new_revolute=1000.0)
        self.assertFalse(f._accept(prev, better, same)[0])
        shrunk = dict(support=dict(M_proposal=np.r_[np.ones(50, bool), np.zeros(50, bool)]), contact={}, fit=Fit())
        self.assertFalse(f._accept(prev, shrunk, common)[0])
        contra = dict(better, contact=dict(revolute=[dict(segment_id="s", contradicted=True)]))
        self.assertFalse(f._accept(prev, contra, common)[0])


if __name__ == "__main__":
    unittest.main()


def canonical_record_from(sc, rec, scale=1.0, rotate=None):
    """A canonical robot record (robot_record.py format) built from the synthetic recording: force converted ONCE
    by the shared force_convention contract (reaction on robot, s = -1)."""
    from estimotion_2part.joint_discovery.force_convention import SensorCorrForce, sensor_corr_to_object_base
    fm = sc.obs.frame_map
    F = np.asarray(rec.wrench_sensor[:, :3], float) * scale
    if rotate is not None:
        F = F @ np.asarray(rotate, float).T
    f = sensor_corr_to_object_base(SensorCorrForce(F), rec.R_WS, np.eye(3), -1.0, rotation_source="synthetic")
    n = len(fm.source_frame_index)
    return dict(source_frame_index=np.asarray(fm.source_frame_index), timestamp_s=np.asarray(fm.timestamp_s),
                F_object_base_N=f.values, p_tcp_base_m=np.asarray(rec.p_tcp_w), contact_state=np.full(n, "contact"),
                in_contact_segment=np.ones(n, bool), force_sigma_N=np.array([0.05, 0.05, 0.05]),
                force_conversion=f.provenance, T_world_base=np.eye(4))


class SharedRobotRecord(unittest.TestCase):
    def test_alignment_same_signed_force_and_row_mismatch(self):
        from estimotion_2part.joint_discovery.frontend_robot_record import (RecordAlignmentError,
                                                                            align_record_to_frame_map)
        sc = make_scene("revolute", seed=1, T=30, n_moving=150, max_angle_deg=30)
        rec, cal, doc = make_contact_record(sc, "stick_hard", verified=False)
        R = canonical_record_from(sc, rec)
        al = align_record_to_frame_map(R, sc.obs.frame_map, p_tcp_w=rec.p_tcp_w, T_WB=rec.T_WB)
        np.testing.assert_array_equal(al["f_object_w"], R["F_object_base_N"])
        np.testing.assert_allclose(al["f_object_w"], -np.asarray(rec.wrench_sensor[:, :3]))   # once, s = -1
        bad = dict(R, source_frame_index=R["source_frame_index"] + 1000)
        with self.assertRaises(RecordAlignmentError):
            align_record_to_frame_map(bad, sc.obs.frame_map)
        shifted = dict(R, p_tcp_base_m=R["p_tcp_base_m"] + 0.01)
        with self.assertRaises(RecordAlignmentError):                   # TCP of another row / archive
            align_record_to_frame_map(shifted, sc.obs.frame_map, p_tcp_w=rec.p_tcp_w)

    def test_certified_unchanged_and_exploratory_diagnostic(self):
        sc = make_scene("revolute", seed=1, T=30, n_moving=150, max_angle_deg=30)
        rec, cal, doc = make_contact_record(sc, "stick_hard", verified=False)
        R = canonical_record_from(sc, rec)
        outs = {}
        with tempfile.TemporaryDirectory() as td:
            for name, kw, c in (("plain", {}, cfg()),
                                ("certified_record", dict(robot_record=R, force_policy="certified"), cfg()),
                                ("exploratory", dict(robot_record=R, force_policy="exploratory"),
                                 cfg("min_effort_tangent")),
                                ("exploratory_wrong_axis", dict(robot_record=canonical_record_from(
                                    sc, rec, rotate=[[0, -1, 0], [1, 0, 0], [0, 0, 1]]), force_policy="exploratory"),
                                 cfg("min_effort_tangent")),
                                ("exploratory_low_force", dict(robot_record=canonical_record_from(sc, rec, scale=0.01),
                                                               force_policy="exploratory"), cfg("min_effort_tangent"))):
                sel = run_multimodal_frontend(sc.obs, sc.calibs, c, Path(td) / name, rec=rec, segments_source=doc,
                                              wrench_calibration=cal, log=lambda m: None, **kw)
                acc = Path(td) / name / "accepted"
                outs[name] = dict(sel=sel, files=sorted(p.name for p in acc.iterdir()),
                                  usage=json.loads((acc / "force_usage.json").read_text()) if (acc / "force_usage.json").is_file()
                                  else None,
                                  align=(Path(td) / name / "robot" / "canonical_record_alignment.json").is_file())
        p, c_, e = outs["plain"]["sel"], outs["certified_record"]["sel"], outs["exploratory"]["sel"]
        self.assertEqual(p["physical_joint_status"], c_["physical_joint_status"])        # certified unchanged
        self.assertEqual(p["physics_joint_export_ready"], c_["physics_joint_export_ready"])
        self.assertTrue(outs["certified_record"]["align"])
        self.assertFalse(outs["certified_record"]["usage"]["conditional"])
        for k in ("exploratory", "exploratory_wrong_axis", "exploratory_low_force"):
            s = outs[k]["sel"]
            self.assertFalse(s["physics_joint_export_ready"])                         # never promoted
            self.assertIn("EXPLORATORY_FORCE_DIAGNOSTIC", s["failure_codes"])
            self.assertTrue(outs[k]["usage"]["conditional"])
            self.assertIn("exploratory_force_diagnostics.json", outs[k]["files"])
            self.assertFalse(s.get("force_evidence_used") and s["physical_joint_status"].startswith("IDENTIFIED_P"))
