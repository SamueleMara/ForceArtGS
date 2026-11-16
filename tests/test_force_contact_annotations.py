import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "samuele"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_force_contact_annotations as fc  # noqa: E402
from samuele.tcp_contact_evidence import load_contact_state  # noqa: E402

RATE = 50.0


def synthetic_force(seed=0):
    """1 s quiet, 0.04 s spike, 1 s quiet, 2 s sustained push (0.6 N) with a mid dip, 1 s release."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 5.0, 1 / RATE)
    F = rng.normal(0, 0.04, (len(t), 3)) + np.array([0.03, -0.03, 0.01])  # sensor bias
    F[(t >= 1.0) & (t < 1.04), 0] += 1.0  # transient spike: not a sustained interaction
    push = (t >= 2.0) & (t < 4.0)
    F[push] += np.array([0.5, 0.2, 0.3])
    dip = (t >= 3.0) & (t < 3.04)
    F[dip] -= np.array([0.5, 0.2, 0.3]) * 0.6  # brief drop between thresholds
    return t, F


class DetectorTests(unittest.TestCase):
    def detect(self, t, F, **kw):
        base = fc.baseline_mask_from(t, np.arange(len(t)), duration_s=0.5)
        opts = dict(contact_z=6.0, absent_z=3.0, min_contact_duration_s=0.2)
        opts.update(kw)
        return fc.detect_contact_states(t, F, base, **opts)

    def test_three_states_and_sustained_interval(self):
        t, F = synthetic_force()
        r = self.detect(t, F)
        state = r["state"]
        np.testing.assert_allclose(r["tare_N"], [0.03, -0.03, 0.01], atol=0.03)
        push = (t >= 2.05) & (t < 3.95) & ~((t >= 2.98) & (t < 3.06))
        self.assertGreater((state[push] == 1).mean(), 0.97)
        self.assertTrue(np.all(state[(t > 4.2)] != 1))
        self.assertGreater((state[t < 0.9] == 0).mean(), 0.9)
        # The 40 ms spike is a transient: uncertain, never contact.
        spike = (t >= 1.0) & (t < 1.04)
        self.assertTrue(np.all(state[spike] != 1))
        self.assertIn("transient_force_shorter_than_min_duration", [r["reason"][i] for i in np.nonzero(spike)[0]])
        sustained = [e for e in r["episodes"] if e["sustained"]]
        self.assertEqual(len(sustained), 1)
        self.assertAlmostEqual(t[sustained[0]["first_index"]], 2.0, delta=0.05)

    def test_uncertain_never_becomes_absence(self):
        t, F = synthetic_force()
        r = self.detect(t, F)
        between = (r["z"] > 3.0) & (r["z"] < 6.0)
        self.assertTrue(between.any())
        self.assertTrue(np.all(r["state"][between] == -1))
        F[10, :] = np.nan
        r = self.detect(t, F)
        self.assertEqual(r["state"][10], -1)
        self.assertEqual(r["reason"][10], "nonfinite_measurement")

    def test_gravity_residual_removes_orientation_change(self):
        t = np.arange(0, 2.0, 1 / RATE)
        angle = np.where(t < 1.0, 0.0, np.deg2rad(30.0))
        R = fc.rotvec_to_matrix(np.stack([angle, 0 * angle, 0 * angle], 1))
        weight = 0.4 * np.einsum("nji,j->ni", R, fc.G_BASE)  # tool weight in sensor axes
        F = weight + np.random.default_rng(1).normal(0, 0.03, weight.shape)
        base = t < 0.5
        uncompensated = fc.detect_contact_states(t, F, base, contact_z=6, absent_z=3, min_contact_duration_s=0.2)
        self.assertTrue((uncompensated["state"][t > 1.1] == 1).all())  # gravity change looks like contact
        compensated = fc.detect_contact_states(t, F, base, contact_z=6, absent_z=3, min_contact_duration_s=0.2,
                                               gravity_correction=fc.gravity_residual(R, base, 0.4))
        self.assertFalse((compensated["state"] == 1).any())

    def test_parameters_validated(self):
        t, F = synthetic_force()
        with self.assertRaises(ValueError):
            self.detect(t, F, absent_z=7.0)
        with self.assertRaises(ValueError):
            fc.baseline_mask_from(t, np.arange(len(t)), duration_s=0.01)

    def test_manual_corrections_need_provenance_and_reason(self):
        frames = np.arange(10)
        state = np.zeros(10, np.int8)
        new, reason, applied = fc.apply_corrections(frames, state, ["x"] * 10, dict(
            provenance="operator, video review", corrections=[dict(source_frames=[2, 4], state="contact", reason="visible push")]))
        self.assertEqual(new[2:5].tolist(), [1, 1, 1])
        self.assertEqual(applied[0]["samples"], 3)
        with self.assertRaises(ValueError):
            fc.apply_corrections(frames, state, ["x"] * 10, dict(corrections=[]))
        with self.assertRaises(ValueError):
            fc.apply_corrections(frames, state, ["x"] * 10, dict(provenance="p", corrections=[
                dict(source_frames=[0, 1], state="maybe", reason="r")]))
        self.assertEqual(fc.intervals_of(frames, new), [[2, 4]])


def fake_scene(root, n=250):
    """Converted-scene layout with a native episode table read through a mocked Parquet reader."""
    scene = Path(root) / "scene"
    (scene / "metadata/raw").mkdir(parents=True)
    (scene / "metadata/raw/episode_000000.parquet").write_bytes(b"not read; reader is mocked")
    (scene / "metadata/raw/info.json").write_text(json.dumps(dict(
        wrench_names=["Fx_N", "Fy_N", "Fz_N", "Tx_Nm", "Ty_Nm", "Tz_Nm"],
        state_names=["pos_x", "pos_y", "pos_z", "rotvec_x", "rotvec_y", "rotvec_z"])))
    (scene / "metadata/robot_tool.json").write_text(json.dumps(dict(
        tool="pointed_tip", wrench_frame=dict(axes="ee", R_ee_force_sensor=np.eye(3).tolist()))))
    source = list(range(0, n, 2))
    (scene / "metadata/real_conversion.json").write_text(json.dumps(dict(
        source="unused", episode=0, source_frames=source, output_frames=list(range(len(source))))))
    t, F = synthetic_force()
    t, F = t[:n], F[:n]
    episode = dict(frame_index=np.arange(n), timestamp=t,
                   wrench=np.c_[F, np.zeros((n, 3))], state=np.c_[np.zeros((n, 3)), np.zeros((n, 3))])
    return scene, episode


class ExportTests(unittest.TestCase):
    def test_export_contract_alignment_and_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, episode = fake_scene(tmp)
            out = Path(tmp) / "out"
            with mock.patch.object(fc, "read_episode", return_value=episode):
                summary = fc.export(scene, out, plot=False)
                with self.assertRaises(ValueError):
                    fc.export(scene, scene / "metadata/contact", plot=False)  # never inside the scene
            self.assertEqual(summary["status"], "available")
            data = json.loads((out / "contact_annotations.json").read_text())
            self.assertEqual(data["index"], "source_frame_index")
            self.assertEqual(len(data["rgbd_alignment"]), 125)
            row = data["rgbd_alignment"][60]
            self.assertEqual((row["rgbd_frame_index"], row["source_frame_index"]), (60, 120))
            self.assertEqual(row["state"], data["contact_state"][120])
            self.assertTrue(data["thresholds"]["tare_N"])
            sources = np.array([0, 50, 120, 150, 152])
            active, _, prov, state = load_contact_state(scene, sources, out / "contact_annotations.json")
            self.assertEqual(active.tolist(), (state == 1).tolist())
            self.assertEqual(state[0], 0)  # quiet start: confirmed absence
            self.assertEqual(state[2], 1)  # t = 2.4 s: sustained push
            self.assertEqual(prov["state_counts"]["contact"], int(active.sum()))
            csv_active, _, _, csv_state = load_contact_state(scene, sources, out / "contact_annotations.csv")
            self.assertEqual(csv_state.tolist(), state.tolist())

    def test_rotation_beyond_limit_requires_mass(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, episode = fake_scene(tmp)
            episode["state"][100:, 3] = np.deg2rad(5.0)
            with mock.patch.object(fc, "read_episode", return_value=episode):
                with self.assertRaisesRegex(ValueError, "tool gravity"):
                    fc.export(scene, Path(tmp) / "out", plot=False)
                summary = fc.export(scene, Path(tmp) / "out2", plot=False, tool_mass_kg=0.2)
            self.assertTrue(summary["gravity_compensation"]["applied"])


class ContactStateContractTests(unittest.TestCase):
    def test_missing_source_is_uncertain_not_absent(self):
        active, _, _, state = load_contact_state(Path("."), np.array([1, 2]), None)
        self.assertEqual(state.tolist(), [-1, -1])
        self.assertFalse(active.any())

    def test_interval_outside_defaults_to_uncertain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(dict(intervals=[[2, 3]], provenance="operator")))
            _, _, _, state = load_contact_state(Path(tmp), np.array([1, 2, 4]), path)
            self.assertEqual(state.tolist(), [-1, 1, -1])
            path.write_text(json.dumps(dict(intervals=[[2, 3]], provenance="operator", outside_intervals="absent")))
            _, _, _, state = load_contact_state(Path(tmp), np.array([1, 2, 4]), path)
            self.assertEqual(state.tolist(), [0, 1, 0])
            path.write_text(json.dumps(dict(source_frame_index=[1, 2], contact_state=["contact", "sometimes"],
                                            provenance="x")))
            with self.assertRaisesRegex(ValueError, "Unknown contact_state"):
                load_contact_state(Path(tmp), np.array([1]), path)


class PhaseFailureTests(unittest.TestCase):
    """B/C must fail explicitly without confirmed contact instead of a successful no-op."""

    def setUp(self):
        from test_tcp_tracking_pipeline import loader, write_inputs, N_FRAMES
        self.tmp = tempfile.TemporaryDirectory()
        self.input = write_inputs(Path(self.tmp.name) / "in")
        self.loader, self.n = loader().resize(32)[0], N_FRAMES

    def tearDown(self):
        self.tmp.cleanup()

    def context(self, **tracking):
        from estimotion_2part.prior_context import TrackingPriorContext
        from utils.pipeline_config import resolve_tracking
        return TrackingPriorContext(SimpleNamespace(**resolve_tracking(dict(tracking=tracking))), self.loader,
                                    self.input, self.n)

    def test_phase_c_without_confirmed_contact_fails(self):
        art = self.input / "metadata/tcp_contact/tcp_contact.npz"
        with np.load(art) as data:
            arrays = dict(data)
        arrays["contact_active"][:] = False
        np.savez_compressed(art, **arrays)
        with self.assertRaisesRegex(ValueError, "no confirmed contact"):
            self.context(contact_surface_enabled=True)
        self.context(tcp_supervision_enabled=True)  # Phase A does not need contact

    def test_phase_b_without_seeds_fails(self):
        seeds = self.input / "metadata/moving_prompts/moving_seed_maps.npz"
        with np.load(seeds) as data:
            arrays = dict(data)
        arrays["valid"][:] = False
        np.savez_compressed(seeds, **arrays)
        with self.assertRaisesRegex(ValueError, "no usable moving-part seed"):
            self.context(moving_prompt_enabled=True)

    def test_secondary_track_point(self):
        with self.assertRaisesRegex(ValueError, "secondary_feature"):
            self.context(tcp_supervision_enabled=True, tcp_track_point="secondary_feature")
        art = self.input / "metadata/tcp_contact/tcp_contact.npz"
        with np.load(art) as data:
            arrays = dict(data)
        arrays["p_feature2_cam_m"] = arrays["p_tcp_cam_m"] + np.array([0.01, 0.0, 0.0])
        np.savez_compressed(art, **arrays)
        ctx = self.context(tcp_supervision_enabled=True, tcp_track_point="secondary_feature")
        tcp = self.context(tcp_supervision_enabled=True)
        self.assertEqual(ctx.report["phase_a_track_point"], "secondary_feature")
        self.assertGreater(float(np.abs(ctx.tcp["uv"][0] - tcp.tcp["uv"][0]).max()), 0.1)
        with self.assertRaisesRegex(ValueError, "Unknown tcp_track_point"):
            self.context(tcp_supervision_enabled=True, tcp_track_point="flange")


if __name__ == "__main__":
    unittest.main()
