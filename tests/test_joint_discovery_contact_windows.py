"""Stage D1: contact timelines with state + provenance; heuristics are never confirmed (§10 item 14)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.contact_annotations import (ContactTimeline, evaluate_heuristic,  # noqa: E402
                                                                  fuse_sensor_cues, load_annotation_csv,
                                                                  tcp_proximity_to_observed_object)
from estimotion_2part.joint_discovery.synthetic import look_at_camera  # noqa: E402


class TestContactWindows(unittest.TestCase):
    def test_confirmed_requires_independent_provenance(self):
        with self.assertRaises(ValueError):
            ContactTimeline([0, 2], ["contact_confirmed", "no_contact"], ["sensor_event", "sensor_event"])
        tl = ContactTimeline([0, 2, 4], ["contact_confirmed", "contact_probable", "no_contact"],
                             ["operator_label", "sensor_event+geometric_proximity", "operator_label"])
        np.testing.assert_array_equal(tl.hard_constraint_mask, [True, False, False])
        self.assertEqual(tl.select([4, 0]).state.tolist(), ["no_contact", "contact_confirmed"])
        with self.assertRaises(KeyError):
            tl.select([6])

    def test_legacy_force_annotations_never_become_confirmed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.csv"
            p.write_text("source_frame_index,timestamp_s,contact_state,contact_active,force_z\n"
                         "0,0,absent,0,1\n2,0.04,contact,1,9\n4,0.08,uncertain,0,4\n")
            tl = load_annotation_csv(p)
            self.assertEqual(tl.state.tolist(), ["contact_unknown"] * 3)
            self.assertFalse(tl.hard_constraint_mask.any())
            q = Path(td) / "b.csv"
            q.write_text("source_frame_index,state,provenance\n0,absent,operator_label\n2,contact,operator_label\n")
            self.assertEqual(load_annotation_csv(q).state.tolist(), ["no_contact", "contact_confirmed"])
            r = Path(td) / "c.csv"
            r.write_text("source_frame_index,state,provenance\n0,contact,my_guess\n")
            with self.assertRaises(ValueError):
                load_annotation_csv(r)

    def test_fusion_and_heuristic_false_activations(self):
        fs = np.array([1, 1, 0, 0, -1, 1])
        near = np.array([1, 0, 0, 0, 1, 0], bool)
        far = np.array([0, 0, 1, 0, 0, 1], bool)
        tl = fuse_sensor_cues(np.arange(6) * 2, fs, near, far)
        self.assertEqual(tl.state.tolist(), ["contact_probable", "contact_unknown", "no_contact", "contact_unknown",
                                             "contact_unknown", "contact_unknown"])
        self.assertFalse(tl.hard_constraint_mask.any())       # sensor cues alone never activate hard terms
        ev = evaluate_heuristic(np.array([1, 0, 1, 0, 0, 0], bool), tl)
        self.assertEqual(ev["false_activations_on_no_contact"], 1)
        self.assertEqual(ev["flagged_and_positive"], 1)
        self.assertIn("NOT ground truth", ev["reference"])

    def test_tcp_proximity_to_measured_object_depth(self):
        cal = look_at_camera("cam_001", [0, -1, 0], [0, 0, 0], up=(0, 0, 1), width=80, height=60, f=80)
        depth = np.full((60, 80), 1.0)        # fronto-parallel object plane at y = 0 (1 m from the camera)
        mask = np.ones((60, 80), bool)
        d, st = tcp_proximity_to_observed_object(np.array([0.0, -0.02, 0.0]), cal, depth, mask, window_px=10)
        self.assertEqual(st, "ok")
        self.assertAlmostEqual(d, 0.02, places=3)
        d2, st2 = tcp_proximity_to_observed_object(np.array([0.0, -2.0, 0.0]), cal, depth, mask)
        self.assertEqual(st2, "tcp_not_in_view")
        d3, st3 = tcp_proximity_to_observed_object(np.array([0.0, -0.02, 0.0]), cal, depth, np.zeros((60, 80), bool))
        self.assertEqual(st3, "no_masked_depth_near_tcp")


if __name__ == "__main__":
    unittest.main()
