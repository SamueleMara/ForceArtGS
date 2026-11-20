"""G9 <- existing robot preprocessing: canonical robot record, gate translation, exploratory policy, status
categories (synthetic scene metadata + force export written in the exact formats of the real pipeline)."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9 import physics_lib  # noqa: E402
from forceartgs_g9.adapters.forceartgs_g9 import (record_force_audit, record_interactions,  # noqa: E402
                                                  record_perturbations)
from forceartgs_g9.config import load_dynamics_config  # noqa: E402
from estimotion_2part.joint_discovery.robot_record import build_robot_record, positive_work_check  # noqa: E402

physics_lib.load()


def fake_dataset(td, sign=-1.0, n=300, seg=(100, 220)):
    """Prismatic slider along +y pushed by the robot: recorded force = reaction on the robot (sign -1)."""
    td = Path(td)
    meta = td / "scene" / "metadata"
    meta.mkdir(parents=True)
    (meta / "cameras.json").write_text(json.dumps(dict(cameras=[dict(name="cam_001", coordinate_frame="base_link")])))
    (meta / "contact_force_convention.json").write_text(json.dumps(dict(
        object_side_force_multiplier=-1, recorded_quantity="contact_reaction_force_on_robot",
        confirmation_source="test operator")))
    (meta / "robot_tool.json").write_text(json.dumps(dict(T_ee_tcp=np.eye(4).tolist(), calibration_source="test",
                                                          reference_assumption="declared")))
    (meta / "robot_base_registration.json").write_text(json.dumps(dict(status="OPERATOR_APPLIED_ROBOT_REGISTRATION_UNVERIFIED")))
    rows = np.arange(n)
    t = rows / 50.0
    rng = np.random.default_rng(0)
    push = (rows >= seg[0] + 5) & (rows <= seg[1] - 20)
    F_obj = np.zeros((n, 3))
    F_obj[push, 1] = 2.0                                   # robot pushes the part along +y
    F_rec = sign * F_obj + 0.04 * rng.standard_normal((n, 3))
    state = np.where(push, "contact", "absent")
    fx = td / "force"
    fx.mkdir()
    with open(fx / "contact_annotations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_frame_index", "timestamp_s", "contact_state", "contact_active", "force_z", "Fx_corr_N",
                    "Fy_corr_N", "Fz_corr_N", "ee_speed_m_s", "reason"])
        for i in rows:
            w.writerow([i, f"{t[i] + 0.0003:.6f}", state[i], int(push[i]), 0, *F_rec[i], 0, ""])
    (fx / "contact_annotations.json").write_text(json.dumps(dict(
        thresholds=dict(tare_N=[0.001, 0.002, 0.0], sigma_N=[0.04, 0.04, 0.04], baseline_duration_s=0.5,
                        noise_model="MAD"),
        gravity_compensation=dict(max_rotation_from_baseline_deg=0.05, applied=False, reason="test"),
        wrench_frame=dict(R_ee_force_sensor=np.eye(3).tolist(), alignment_source="test declaration",
                          torque_origin="unspecified"),
        sources=dict(parquet_sha256="x"))))
    T = np.tile(np.eye(4), (n, 1, 1))
    q = np.cumsum(np.where(push, 0.002, 0.0))              # moves only while pushed (+y)
    T[:, :3, 3] = np.c_[np.zeros(n), q, np.full(n, 0.3)]
    tare = np.array([0.001, 0.002, 0.0])
    np.savez(td / "robot_archive.npz", T_base_ee=T, T_base_tcp=T, source_row=rows,
             wrench=np.c_[F_rec + tare, np.zeros((n, 3))])        # raw = corrected + export tare (no double tare)
    (td / "seg.yaml").write_text("schema: forceartgs.contact_segments/v1\nsource: test\nsegments:\n"
                                 f"  - {{segment_id: c0, start_source_frame: {seg[0]}, end_source_frame: {seg[1]}, "
                                 "contact_state: contact_confirmed, slip_mode: sliding, provenance: operator_label}\n")
    k = rows[::2]
    auth = SimpleNamespace(joint_type="prismatic", axis_world=np.array([0, 1.0, 0]), origin_world=None,
                           source_frame_index=k, q_valid=np.ones(len(k), bool), timestamp_s=k / 50.0, q=q[k],
                           q_sigma=None, q_units="m", rgbd_frame_index=k, reference_time_index=0)
    return td, auth


class RobotRecord(unittest.TestCase):
    def test_record_reuses_export_and_marks_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            td, auth = fake_dataset(td)
            rec = build_robot_record(td / "scene", td / "force", td / "robot_archive.npz", td / "seg.yaml")
            ev = rec["evidence"]
            self.assertEqual(ev["force_samples"]["status"], "MEASURED")
            self.assertEqual(ev["tare_initial"]["status"], "ESTIMATED")           # reused, not re-estimated
            self.assertEqual(ev["tare_initial"]["value"], [0.001, 0.002, 0.0])
            self.assertEqual(ev["sensor_axes"]["status"], "DECLARED")
            self.assertEqual(ev["row_association"]["status"], "VERIFIED_BY_CONSTRUCTION")
            self.assertEqual(ev["hardware_clock_offset"]["status"], "NOT_ESTABLISHED")
            self.assertEqual(ev["tare_second_window"]["status"], "THRESHOLD_CONDITIONED_DIAGNOSTIC")
            self.assertEqual(ev["tare_independent_window"]["status"], "NOT_ESTABLISHED")
            self.assertEqual(ev["tare_not_duplicated"]["status"], "VERIFIED_BY_CONSTRUCTION")
            self.assertEqual(rec["force_conversion"]["sign"], -1.0)
            self.assertIn("metadata_sha256", rec["force_conversion"])
            push = rec["contact_state"] == "contact"
            self.assertAlmostEqual(float(np.median(rec["F_object_base_N"][push, 1])), 2.0, delta=0.05)  # object side

    def test_policies_and_positive_work(self):
        with tempfile.TemporaryDirectory() as td:
            td, auth = fake_dataset(td)
            rec = build_robot_record(td / "scene", td / "force", td / "robot_archive.npz", td / "seg.yaml")
            cert = record_force_audit(rec, auth, load_dynamics_config(None))
            self.assertEqual(cert["status"], "FORCE_CALIBRATION_UNVERIFIED")
            exp = record_force_audit(rec, auth, load_dynamics_config({"pipeline": {"force_policy": "exploratory"}}))
            self.assertEqual(exp["status"], "EXPLORATORY_ASSUMPTIONS_ACCEPTED")
            self.assertNotIn("VERIFIED", [g["status"] for g in exp["gates"].values()])   # never upgraded
            self.assertEqual(exp["positive_work"]["status"], "DATA_CONSISTENT")
            # threshold-selected rows are a consistency diagnostic, never independent drift verification
            self.assertEqual(exp["gates"]["tare"]["status"], "ESTIMATED_INITIAL_TARE_THRESHOLD_CONDITIONED_CONSISTENT")
            self.assertIsNone(exp["tare_independent_window_stable"])
            for g in exp["gates"].values():
                self.assertIn("additional_verification_for_certified_export", g)
                self.assertIn("needed_for_joint_space_estimation", g)
        with tempfile.TemporaryDirectory() as td:
            td, auth = fake_dataset(td, sign=+1.0)                 # a sign error in the recording convention
            rec = build_robot_record(td / "scene", td / "force", td / "robot_archive.npz", td / "seg.yaml")
            self.assertEqual(positive_work_check(rec, auth)["status"], "DATA_CONTRADICTS")
            exp = record_force_audit(rec, auth, load_dynamics_config({"pipeline": {"force_policy": "exploratory"}}))
            self.assertEqual(exp["status"], "CALIBRATION_ASSUMPTION_CONTRADICTED")

    def test_windows_include_known_zero_effort_and_exclude_uncertain(self):
        with tempfile.TemporaryDirectory() as td:
            td, auth = fake_dataset(td)
            rec = build_robot_record(td / "scene", td / "force", td / "robot_archive.npz", td / "seg.yaml")
            cfg = load_dynamics_config({"pipeline": {"force_policy": "exploratory"}})
            its = record_interactions(rec, auth, cfg, {})
            self.assertEqual(len(its), 1)                           # rest before + push + rest after: one window
            self.assertGreater(its[0].meta["n_absent_rows"], 0)
            self.assertGreater(its[0].meta["effort_snr"]["snr"], 10)
            rec["contact_state"][50] = "uncertain"                  # an uncertain row outside the segment splits it
            its2 = record_interactions(rec, auth, cfg, {})
            self.assertTrue(all(i.meta["source_rows"][0] > 50 or i.meta["source_rows"][1] < 50 for i in its2))
            pert = record_perturbations(rec, auth, load_dynamics_config(
                {"identifiability": {"contact_point_sweep_m": 0.01}}), its)
            self.assertIn("tare_threshold_conditioned_residual|+", pert)
            self.assertIn("tare_threshold_conditioned_residual|-", pert)

    def test_exploratory_fit_categories_and_no_mesh_requirement(self):
        from forceartgs_g9.core import G9Inputs, run_g9
        with tempfile.TemporaryDirectory() as td:
            td, auth = fake_dataset(td)
            rec = build_robot_record(td / "scene", td / "force", td / "robot_archive.npz", td / "seg.yaml")
            cfg = load_dynamics_config({"pipeline": {"force_policy": "exploratory"},
                                        "model": {"terms": {"static": False, "gravity": False},
                                                  "parameter_source": "joint_space_only"},
                                        "fitting": {"compare_frozen_viscous_baselines": [], "n_starts": 2}})
            fa = record_force_audit(rec, auth, cfg)
            its = record_interactions(rec, auth, cfg, {})
            fa["canonical_record"] = rec
            inp = G9Inputs("fake", "prismatic", auth.axis_world, None, "IDENTIFIED", True, {}, its, [], fa)
            r = run_g9(inp, cfg, Path(td) / "g9", log=lambda m: None)
            m = r["manifest"]
            self.assertEqual(m["status"], "EXPLORATORY_ESTIMATE")
            cats = m["status_categories"]
            for k in ("DATA_UNAVAILABLE", "KINEMATICS_NOT_ACCEPTED", "CALIBRATION_ASSUMPTION_UNVERIFIED",
                      "DYNAMICS_NOT_IDENTIFIABLE", "SIMULATION_ASSET_INCOMPLETE"):
                self.assertIn(k, cats)
            self.assertEqual(cats["DATA_UNAVAILABLE"], [])
            self.assertTrue(cats["CALIBRATION_ASSUMPTION_UNVERIFIED"])
            self.assertTrue(cats["SIMULATION_ASSET_INCOMPLETE"])         # reported, but the fit still ran
            self.assertIsNotNone(r["fit"])
            self.assertIsNone(r["physics_spec_v1"])                       # never production from exploratory
            self.assertTrue((Path(td) / "g9" / "robot_record" / "canonical_robot_record.npz").is_file())
            self.assertIn(r["table"]["viscous"]["status"], ("estimated_conditional", "not_identifiable"))


    def test_fail_closed_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            td, auth = fake_dataset(td)
            (td / "scene" / "metadata" / "contact_force_convention.json").write_text(json.dumps({}))
            with self.assertRaises(ValueError):                    # no multiplier -> hard error, no default
                build_robot_record(td / "scene", td / "force", td / "robot_archive.npz", td / "seg.yaml")
        with tempfile.TemporaryDirectory() as td:
            td, auth = fake_dataset(td)
            z = dict(np.load(td / "robot_archive.npz"))
            z["wrench"] = z["wrench"] - np.r_[0.001, 0.002, 0.0, 0, 0, 0]          # export would tare twice
            np.savez(td / "robot_archive.npz", **z)
            with self.assertRaises(ValueError):
                build_robot_record(td / "scene", td / "force", td / "robot_archive.npz", td / "seg.yaml")
        with tempfile.TemporaryDirectory() as td:
            td, auth = fake_dataset(td)
            z = dict(np.load(td / "robot_archive.npz"))
            np.savez(td / "robot_archive.npz", **{k: v[:-10] for k, v in z.items()})   # rows missing
            with self.assertRaises(ValueError):
                build_robot_record(td / "scene", td / "force", td / "robot_archive.npz", td / "seg.yaml")

    def test_independent_designated_window_and_record_is_stable(self):
        with tempfile.TemporaryDirectory() as td:
            td, auth = fake_dataset(td)
            rec = build_robot_record(td / "scene", td / "force", td / "robot_archive.npz", td / "seg.yaml",
                                     independent_tare_windows=[(240, 290)])
            self.assertEqual(rec["evidence"]["tare_independent_window"]["status"], "INDEPENDENT_WINDOW_CONSISTENT")
            cfg = load_dynamics_config({"pipeline": {"force_policy": "exploratory"}})
            fa = record_force_audit(rec, auth, cfg)
            self.assertEqual(fa["gates"]["tare"]["status"], "ESTIMATED_INDEPENDENT_WINDOW_STABLE")
            rec2 = build_robot_record(td / "scene", td / "force", td / "robot_archive.npz", td / "seg.yaml")
            np.testing.assert_array_equal(rec["F_object_base_N"], rec2["F_object_base_N"])   # re-export: same sign
            csv_bytes = (td / "force" / "contact_annotations.csv").read_bytes()
            build_robot_record(td / "scene", td / "force", td / "robot_archive.npz", td / "seg.yaml")
            self.assertEqual(csv_bytes, (td / "force" / "contact_annotations.csv").read_bytes())   # export untouched


if __name__ == "__main__":
    unittest.main()
