import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from samuele.tcp_contact_evidence import (  # noqa: E402
    load_aligned_tcp, load_contact_state, load_tcp_contact_artifact, main, map_freeart_frames, pinhole,
    project_tcp, save_tcp_contact,
)
from tcp_contact_fixtures import K, SIZE, look_at_camera, make_freeart_camera, make_scene  # noqa: E402


class FrameMappingTests(unittest.TestCase):
    def archive(self):
        return dict(frame_index=np.arange(10), source_frame_index=np.arange(10) * 2 + 7)

    def test_stride_interval_and_nonzero_raw_indices(self):
        frame_map = [dict(output_stem=f"{i:05d}", source_stem=f"frame_{o:06d}") for i, o in enumerate([3, 5, 7])]
        conversion = dict(source_frames=(np.arange(10) * 2 + 7).tolist(), output_frames=list(range(10)))
        mapped = map_freeart_frames(frame_map, self.archive(), conversion)
        self.assertEqual([m["source_frame_index"] for m in mapped], [13, 17, 21])
        self.assertEqual([m["local_index"] for m in mapped], [0, 1, 2])
        self.assertNotEqual(mapped[1]["source_frame_index"], mapped[1]["output_frame_index"])

    def test_inconsistent_real_conversion_rejected(self):
        frame_map = [dict(output_stem="00000", source_stem="frame_000003")]
        conversion = dict(source_frames=list(range(10)), output_frames=list(range(10)))
        with self.assertRaisesRegex(ValueError, "real_conversion maps"):
            map_freeart_frames(frame_map, self.archive(), conversion)

    def test_non_chronological_and_missing_frames_rejected(self):
        bad_order = [dict(output_stem="00000", source_stem="frame_000005"),
                     dict(output_stem="00001", source_stem="frame_000002")]
        with self.assertRaisesRegex(ValueError, "chronology"):
            map_freeart_frames(bad_order, self.archive())
        with self.assertRaisesRegex(ValueError, "absent"):
            map_freeart_frames([dict(output_stem="00000", source_stem="frame_000099")], self.archive())
        with self.assertRaisesRegex(ValueError, "not a real-scene"):
            map_freeart_frames([dict(output_stem="00000", source_stem="img7")], self.archive())


class ProjectionTests(unittest.TestCase):
    def test_handedness_and_validity(self):
        T_cam = look_at_camera([0, -1, 0], [0, 0, 0])  # camera on -Y looking +Y, base Z up
        tcp = np.repeat(np.eye(4)[None], 4, axis=0)
        tcp[:, :3, 3] = [[0, 0, 0], [0.1, 0, 0], [0, 0, 0.1], [0, -2, 0]]  # last is behind the camera
        uv = project_tcp(tcp, T_cam, K, SIZE)
        np.testing.assert_allclose(uv[0], [K[0, 2], K[1, 2]], atol=1e-9)
        self.assertGreater(uv[1, 0], uv[0, 0])  # base +X appears to the right
        self.assertLess(uv[2, 1], uv[0, 1])     # base +Z (up) appears higher (smaller v)
        self.assertTrue(np.isnan(uv[3]).all())

    def test_tcp_local_point_not_double_applied(self):
        T_cam = look_at_camera([0, -1, 0], [0, 0, 0])
        tcp = np.eye(4)[None].copy()
        uv_origin = project_tcp(tcp, T_cam, K)
        uv_local = project_tcp(tcp, T_cam, K, point_local_tool=np.zeros(3))
        np.testing.assert_allclose(uv_origin, uv_local)

    def test_resized_intrinsics_scale_pixels(self):
        p = np.array([[0.05, -0.02, 1.0]])
        uv, _ = pinhole(K, p)
        Ks = K.copy()
        Ks[:2] *= 0.5
        uv_s, _ = pinhole(Ks, p)
        np.testing.assert_allclose(uv_s, uv * 0.5)


class ContactStateTests(unittest.TestCase):
    def test_missing_source_never_becomes_contact(self):
        active, conf, prov, _ = load_contact_state(Path("."), np.array([1, 2, 3]), None)
        self.assertFalse(active.any())
        self.assertEqual(prov["status"], "unavailable")

    def test_intervals_and_per_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            intervals = Path(tmp) / "c.json"
            intervals.write_text(json.dumps(dict(index="source_frame_index", intervals=[[12, 14]],
                                                 provenance="operator")))
            active, _, prov, _ = load_contact_state(Path(tmp), np.array([10, 12, 14, 16]), intervals)
            self.assertEqual(active.tolist(), [False, True, True, False])
            self.assertEqual(prov["mode"], "confirmed_intervals")
            per_frame = Path(tmp) / "p.json"
            per_frame.write_text(json.dumps(dict(source_frame_index=[10, 12], contact_active=[True, False],
                                                 contact_confidence=[0.9, 0.1], provenance="labelled")))
            active, conf, _, _ = load_contact_state(Path(tmp), np.array([12, 10]), per_frame)
            self.assertEqual(active.tolist(), [False, True])
            np.testing.assert_allclose(conf, [0.1, 0.9])
            with self.assertRaisesRegex(ValueError, "does not cover"):
                load_contact_state(Path(tmp), np.array([10, 99]), per_frame)
            unproven = Path(tmp) / "u.json"
            unproven.write_text(json.dumps(dict(intervals=[[0, 1]])))
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_contact_state(Path(tmp), np.array([0]), unproven)


class AdapterEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.scene = base / "scene"
        T_ee_tcp = np.eye(4)
        T_ee_tcp[2, 3] = 0.2
        self.meta = make_scene(self.scene, T_ee_tcp=T_ee_tcp)
        self.cam0 = make_freeart_camera(base / "work/cam0", self.scene, "cam_000", [2, 4, 6, 8])
        self.cam1 = make_freeart_camera(base / "work/cam1", self.scene, "cam_001", [2, 4, 6, 8])
        self.contact = base / "contact.json"
        self.contact.write_text(json.dumps(dict(intervals=[[18, 22]], provenance="test operator")))

    def tearDown(self):
        self.tmp.cleanup()

    def test_multicamera_projection_and_alignment(self):
        seq0 = load_aligned_tcp(self.scene, self.cam0, contact_source=self.contact)
        seq1 = load_aligned_tcp(self.scene, self.cam1, contact_source=self.contact)
        self.assertEqual(seq0.source_frame_index.tolist(), [14, 18, 22, 26])
        self.assertEqual(seq0.contact_active.tolist(), [False, True, True, False])
        # Same base-frame TCP in both cameras; archive tcp_transform used without re-applying T_ee_tcp.
        np.testing.assert_allclose(seq0.p_tcp_base_m, seq1.p_tcp_base_m)
        np.testing.assert_allclose(seq0.p_tcp_base_m, self.meta["tcp"][[2, 4, 6, 8], :3, 3])
        for seq, name in ((seq0, "cam_000"), (seq1, "cam_001")):
            T_cam_base = np.linalg.inv(self.meta["poses"][name])
            p_cam = seq.p_tcp_base_m @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
            np.testing.assert_allclose(seq.p_tcp_cam_m, p_cam, atol=1e-12)
            self.assertTrue(seq.valid_projection.all())
        # TCP moves along +X: rightwards in cam_000 (facing +Y), towards camera in cam_001 (facing -X).
        self.assertTrue(np.all(np.diff(seq0.uv_tcp_fullres_xy[:, 0]) > 0))
        self.assertTrue(np.all(np.diff(seq1.p_tcp_cam_m[:, 2]) < 0))
        self.assertFalse(seq0.tool_geometry["contact_point_calibrated"])

    def test_intrinsics_mismatch_rejected(self):
        np.savetxt(self.cam0 / "cam_K.txt", K * np.array([[1.1], [1], [1]]))
        with self.assertRaisesRegex(ValueError, "intrinsics"):
            load_aligned_tcp(self.scene, self.cam0)

    def test_save_load_no_pickle_and_refuse_scene_output(self):
        seq = load_aligned_tcp(self.scene, self.cam0, contact_source=self.contact)
        out = Path(self.tmp.name) / "work/cam0/metadata/tcp_contact"
        manifest = save_tcp_contact(seq, out, forbidden_roots=[self.scene])
        arrays, loaded = load_tcp_contact_artifact(out)
        self.assertEqual(loaded["num_frames"], 4)
        self.assertEqual(manifest["contact"]["mode"], "confirmed_intervals")
        self.assertTrue(all(np.asarray(v).dtype != object for v in arrays.values()))
        with self.assertRaisesRegex(ValueError, "read-only"):
            save_tcp_contact(seq, self.scene / "metadata/x", forbidden_roots=[self.scene])

    def test_cli_dry_run_writes_nothing(self):
        out = Path(self.tmp.name) / "dry"
        code = main(["--scene", str(self.scene), "--freeart-camera-data", str(self.cam0),
                     "--contact-file", str(self.contact), "--output", str(out), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
