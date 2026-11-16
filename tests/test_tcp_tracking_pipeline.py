import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.pipeline_config import TRACKING_DEFAULTS, resolve_tracking  # noqa: E402
from estimotion_2part.prior_context import TrackingPriorContext  # noqa: E402
from tcp_contact_fixtures import K  # noqa: E402

H, W, N_FRAMES = 48, 64, 5
U0, V0 = 48.0, 24.0  # contact pixel on the moving (right) half at frame 0


def contact_cam(t):
    u = U0 + t  # moving half translates +1 px/frame at z = 1 m
    return np.array([(u - K[0, 2]) / K[0, 0], (V0 - K[1, 2]) / K[1, 1], 1.0])


def write_inputs(root, n=N_FRAMES):
    root = Path(root)
    (root / "images_ori").mkdir(parents=True)
    for i in range(n):
        (root / "images_ori" / f"{i:05d}.png").write_bytes(b"")
    art = root / "metadata/tcp_contact"
    art.mkdir(parents=True)
    p = np.stack([contact_cam(t) for t in range(n)])
    np.savez_compressed(art / "tcp_contact.npz", local_frame_index=np.arange(n), source_frame_index=np.arange(n) * 2 + 4,
                        p_tcp_cam_m=p, p_contact_cam_m=p, contact_active=np.array([True] * (n - 1) + [False]),
                        K_fullres=K, image_size_wh=np.array([W, H]))
    (art / "manifest.json").write_text(json.dumps(dict(num_frames=n, tool_geometry=dict(contact_point_calibrated=False))))
    seeds = root / "metadata/moving_prompts"
    seeds.mkdir(parents=True)
    pts = [np.array([[int(U0) + t - 2, int(V0)], [int(U0) + t - 2, int(V0) + 1]]) for t in range(n)]
    np.savez_compressed(seeds / "moving_seed_maps.npz", points_xy=np.concatenate(pts),
                        frame_offsets=np.arange(n + 1) * 2, valid=np.ones(n, bool), K_fullres=K,
                        source_frame_index=np.arange(n) * 2 + 4, image_size_wh=np.array([W, H]))
    return root


def loader(n=N_FRAMES, pingpong=True):
    from utils.utils import ImageLoader
    rng = np.random.default_rng(0)
    rgbs = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(n)]
    depths = [np.ones((H, W), np.float32) for _ in range(n)]
    masks = [np.full((H, W, 3), 255, np.uint8) for _ in range(n)]
    feats = [torch.from_numpy(np.stack([np.tile(np.linspace(0, 1, 16), (12, 1))] * 2)).float() for _ in range(n)]
    if pingpong:
        rgbs, depths, masks, feats = [x + list(reversed(x[:-1])) for x in (rgbs, depths, masks, feats)]
    data = dict(rgbs=rgbs, rgbs_tensor=torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in rgbs])[None].float(),
                depths=depths, masks=masks, cam_K=K.astype(np.float32), features=feats)
    return ImageLoader(images_dict=data)


class TwoPartTracker:
    """Real Tracker 3D extraction; fake AllTracker maps: right half moves +1 px per frame."""

    def __init__(self):
        from tracking.tracker_toolkit import Tracker
        self.base = Tracker(SimpleNamespace(query_frame=0))
        self.cfg = self.base.cfg

    def get_track_maps(self, data):
        T = len(data["rgbs"])
        h, w = data["rgbs"][0].shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        maps = np.zeros((1, T, 2, h, w), np.float32)
        order = [int(np.argmax([np.array_equal(data["rgbs"][t], r) for r in data["rgbs"]])) for t in range(T)]
        for t in range(T):
            maps[0, t, 0] = xs + (xs >= w // 2) * (order[t] - order[0])
            maps[0, t, 1] = ys
        device = "cuda" if torch.cuda.is_available() else "cpu"  # AllTracker maps live on the GPU
        return torch.from_numpy(maps).to(device), torch.ones(1, T, 2, h, w, device=device)

    def get_4d_traj_maps(self, *a):
        return self.base.get_4d_traj_maps(*a)

    def get_visible_trajectories(self, *a):
        return self.base.get_visible_trajectories(*a)


def tracking_args(input_dir, output_dir, **flags):
    args = SimpleNamespace(**resolve_tracking({}))
    args.input_dir, args.output_dir = str(input_dir), str(output_dir)
    args.update = None
    for key, value in dict(window_size=4, downscale_factor=1, solve_window_epochs=6, joint_optimize_epochs=6,
                           knn_k=8, min_depth=0.1, max_depth=5.0).items():
        setattr(args, key, value)
    for key, value in flags.items():
        setattr(args, key, value)
    return args


class ConfigTests(unittest.TestCase):
    def test_all_phases_off_by_default_without_invented_tolerances(self):
        for flag in ("tcp_supervision_enabled", "moving_prompt_enabled", "contact_surface_enabled",
                     "contact_multiview_refine_enabled"):
            self.assertFalse(TRACKING_DEFAULTS[flag])
        for key in ("tcp_requery_error_px", "tcp_visibility_depth_tol_m", "contact_sigma_m",
                    "contact_robust_delta_m", "contact_support_max_distance_m"):
            self.assertIsNone(TRACKING_DEFAULTS[key])
        self.assertEqual(TRACKING_DEFAULTS["contact_surface_weight"], 0.0)
        self.assertEqual(TRACKING_DEFAULTS["tcp_track_correction_mode"], "diagnostic_only")
        cfg = resolve_tracking(dict(tracking=dict(contact_surface_enabled=True, contact_sigma_m=0.01)))
        self.assertTrue(cfg["contact_surface_enabled"])
        self.assertEqual(cfg["contact_sigma_m"], 0.01)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.input = write_inputs(Path(self.tmp.name) / "in")

    def tearDown(self):
        self.tmp.cleanup()

    def resized(self, pingpong=True):
        return loader(pingpong=pingpong).resize(32)[0]  # half resolution: K scaled by 0.5

    def test_flags_off_reads_nothing(self):
        args = SimpleNamespace(**resolve_tracking({}))
        ctx = TrackingPriorContext(args, self.resized(), Path(self.tmp.name) / "missing", N_FRAMES)
        self.assertFalse(ctx.enabled)
        self.assertIsNone(ctx.tcp)

    def test_missing_artifact_fails_clearly(self):
        args = SimpleNamespace(**resolve_tracking(dict(tracking=dict(moving_prompt_enabled=True))))
        with self.assertRaises(FileNotFoundError):
            TrackingPriorContext(args, self.resized(), Path(self.tmp.name) / "missing", N_FRAMES)
        args = SimpleNamespace(**resolve_tracking(dict(tracking=dict(contact_multiview_refine_enabled=True))))
        with self.assertRaises(NotImplementedError):
            TrackingPriorContext(args, self.resized(), self.input, N_FRAMES)

    def test_pingpong_resolution_and_duplicates(self):
        args = SimpleNamespace(**resolve_tracking(dict(tracking=dict(tcp_supervision_enabled=True,
                                                                     moving_prompt_enabled=True))))
        ctx = TrackingPriorContext(args, self.resized(), self.input, N_FRAMES)
        self.assertEqual(ctx.report["reversed_copies"], N_FRAMES - 1)
        self.assertAlmostEqual(ctx.tcp["scale"], 0.5)
        np.testing.assert_allclose(ctx.tcp["uv"][0], [U0 * 0.5, V0 * 0.5], atol=1e-9)
        # Sequence index 6 replays original frame 2 (0 1 2 3 4 3 2 1 0).
        np.testing.assert_allclose(ctx.tcp["uv"][6], ctx.tcp["uv"][2])
        self.assertEqual(int(ctx.tcp["source_frame_index"][6]), 8)
        w = ctx.window_tcp(7, 2, duplicated_last=True)
        self.assertEqual(w["sequence_index"].tolist(), [7, 8, 8])
        self.assertEqual(w["usable"].tolist(), [True, True, False])
        seed = ctx.seed_map(0)
        self.assertEqual(seed.shape, (24, 32))
        self.assertTrue(seed[12, 23])
        p, valid, _ = ctx.window_contact(3, 3, False)  # frames 3, 4, 3(rev): frame 4 has no contact
        self.assertEqual(valid.tolist(), [True, False, True])

    def test_conversion_hash_mismatch_rejected(self):
        (self.input / "metadata/freeart_conversion.json").write_text("{}")
        manifest = self.input / "metadata/tcp_contact/manifest.json"
        manifest.write_text(json.dumps(dict(freeart_conversion_sha256="0" * 64)))
        args = SimpleNamespace(**resolve_tracking(dict(tracking=dict(tcp_supervision_enabled=True))))
        with self.assertRaisesRegex(ValueError, "different FreeArt conversion"):
            TrackingPriorContext(args, self.resized(), self.input, N_FRAMES)


class SlideWindowIntegrationTests(unittest.TestCase):
    """Runs the real two-part tracking loop on a tiny synthetic sequence (GPU if available)."""

    def run_tracking(self, **flags):
        from estimotion_2part.track_2part import slide_window_tracking
        tmp = tempfile.mkdtemp()
        inputs = write_inputs(Path(tmp) / "in")
        args = tracking_args(inputs, Path(tmp) / "tracking", **flags)
        torch.manual_seed(0)
        slide_window_tracking(args, {}, TwoPartTracker(), loader())
        return Path(tmp) / "tracking"

    def test_baseline_and_all_phases_produce_same_output_contract(self):
        base = self.run_tracking()
        self.assertFalse((base / "tracking_priors.json").exists())
        self.assertFalse(list(base.glob("window_*/priors")))
        flags = dict(tcp_supervision_enabled=True, tcp_visibility_depth_tol_m=0.01,
                     moving_prompt_enabled=True, moving_prompt_weight=1.0,
                     contact_surface_enabled=True, contact_surface_weight=1.0, contact_sigma_m=0.01,
                     contact_robust_delta_m=0.02, contact_support_max_distance_m=0.05)
        priors = self.run_tracking(**flags)
        base_windows = sorted(p.name for p in base.glob("window_*"))
        self.assertEqual(base_windows, sorted(p.name for p in priors.glob("window_*")))
        for window in base_windows:
            base_maps = sorted(p.name for p in (base / window).glob("seg_map_*.npy"))
            self.assertEqual(base_maps, sorted(p.name for p in (priors / window).glob("seg_map_*.npy")))
            shape = np.load(base / window / base_maps[0]).shape
            self.assertEqual(shape, np.load(priors / window / base_maps[0]).shape)
        report = json.loads((priors / "window_00000/priors/priors.json").read_text())
        self.assertTrue(report["seed_map_available"])
        self.assertTrue(report["pairs"])
        contact = report["joint"]["contact"]
        self.assertIn(contact["status"], ("active", "skipped"))
        if contact["status"] == "active":
            self.assertGreater(contact["N_contact"], 0)
            self.assertIn(contact["moving_branch"], (1, 2))
        self.assertTrue((priors / "window_00000/tcp_track/tcp_track.json").is_file())
        summary = json.loads((priors / "tcp_track_summary.json").read_text())
        self.assertIn(summary["status"], ("available", "skipped"))
        flags_report = json.loads((priors / "tracking_priors.json").read_text())
        self.assertTrue(all(flags_report["flags"].values()))


if __name__ == "__main__":
    unittest.main()
