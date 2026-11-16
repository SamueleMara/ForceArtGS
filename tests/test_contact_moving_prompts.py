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

from samuele.contact_moving_part_prompts import (  # noqa: E402
    STATUS_NO_POSITIVE, STATUS_VALID, generate, select_contact_prompts, unproject, write_outputs,
)
from samuele.contact_moving_part_sam2 import constrain_proposal, frame_prompt, read_contact_guidance  # noqa: E402
from samuele.tcp_contact_evidence import load_aligned_tcp, pinhole, save_tcp_contact  # noqa: E402
from estimotion_2part.process_part_segment import apply_part_seed_map  # noqa: E402
from estimotion_2part.window_priors import SeedTerm, moving_seed_bce, resolve_seed_branch  # noqa: E402
from estimotion_2part.contact_surface_loss import BRANCH_T1, BRANCH_T2  # noqa: E402
from tcp_contact_fixtures import K, make_freeart_camera, make_scene  # noqa: E402

H, W = 48, 64


def synthetic_frame(occluder=False):
    """Plane object at z=1 m; a tool rod touches it at pixel (40, 25) coming from the camera side."""
    depth = np.zeros((H, W))
    mask = np.zeros((H, W), bool)
    depth[8:42, 8:58] = 1.0
    mask[8:42, 8:58] = True
    contact = unproject(K, np.array(40.0), np.array(25.0), np.array(1.0))
    ee = contact + np.array([-0.15, 0.0, -0.3])
    s = np.linspace(0, 1, 40)[:, None]
    rod = contact[None] * (1 - s) + ee[None] * s
    uv, ok = pinhole(K, rod, (W, H))
    tool_pixels = set()
    for (u, v), valid, p in zip(uv, ok, rod):
        if valid:
            x, y = int(round(u)), int(round(v))
            depth[y, x] = p[2]
            mask[y, x] = False  # robot excluded from object foreground
            tool_pixels.add((x, y))
    if occluder:
        depth[18:33, 33:48] = 0.6  # nearer object part hides the contact region
        mask[18:33, 33:48] = True
    return depth, mask, contact, rod[::5], tool_pixels


class PromptSelectionTests(unittest.TestCase):
    def test_positive_on_object_negative_on_tool(self):
        depth, mask, contact, tool, tool_pixels = synthetic_frame()
        result = select_contact_prompts(K, depth, mask, contact, tool, search_radius_m=0.03, depth_margin_m=0.01,
                                        max_positive=5)
        self.assertEqual(result["status"], STATUS_VALID)
        self.assertEqual(len(result["positive_xy"]), 5)
        for x, y in result["positive_xy"]:
            self.assertTrue(mask[y, x])
            self.assertNotIn((x, y), tool_pixels)
        self.assertNotIn([40, 25], result["positive_xy"])  # the TCP pixel is on the tool
        self.assertTrue(result["negative_robot_xy"])
        for x, y in result["negative_robot_xy"]:
            self.assertIn((x, y), tool_pixels)
            self.assertFalse(mask[y, x])
        self.assertLess(np.linalg.norm(np.asarray(result["contact_candidate_3d"]) - contact), 0.03)

    def test_hidden_contact_gives_no_positive(self):
        depth, mask, contact, tool, _ = synthetic_frame(occluder=True)
        result = select_contact_prompts(K, depth, mask, contact, tool, search_radius_m=0.03, depth_margin_m=0.01,
                                        max_positive=5)
        self.assertEqual(result["status"], STATUS_NO_POSITIVE)
        self.assertEqual(result["positive_xy"], [])
        self.assertEqual(result["reason"], "contact_surface_occluded_by_nearer_surface")

    def test_tool_inside_object_mask_is_rejected_by_tool_volume(self):
        """A dark tool segmented as object must not yield positives once the tool volume is excluded."""
        depth, mask, contact, tool, tool_pixels = synthetic_frame()
        # Thick tool (radius ~1.5 cm) wrongly included in the object mask, covering the contact pixel area.
        rod = np.asarray(tool, dtype=np.float64)
        ys, xs = np.mgrid[0:H, 0:W]
        for p in np.linspace(rod[0], rod[-1], 60):
            u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
            near = (xs - u) ** 2 + (ys - v) ** 2 <= (K[0, 0] * 0.012 / p[2]) ** 2
            depth[near] = np.minimum(np.where(depth[near] > 0, depth[near], np.inf), p[2] - 0.004)
            mask[near] = True
        plain = select_contact_prompts(K, depth, mask, contact, tool, search_radius_m=0.04, depth_margin_m=0.03,
                                       max_positive=50)
        excluded = select_contact_prompts(K, depth, mask, contact, tool, search_radius_m=0.04, depth_margin_m=0.03,
                                          max_positive=50, tool_exclusion_radius_m=0.015, tool_exclusion_start_m=-0.015)
        self.assertEqual(plain["tool_volume"], dict(status="not_configured"))
        self.assertGreater(excluded["tool_volume"]["object_mask_pixels_inside_tool_volume"], 0)
        axis = (rod[-1] - contact) / np.linalg.norm(rod[-1] - contact)

        def radial(xy):
            X = unproject(K, float(xy[0]), float(xy[1]), float(depth[xy[1], xy[0]]))
            rel = X - contact
            return np.linalg.norm(rel - (rel @ axis) * axis)
        self.assertTrue(any(radial(xy) <= 0.015 for xy in plain["positive_xy"]))  # tool pixels picked
        self.assertTrue(excluded["positive_xy"])
        self.assertTrue(all(radial(xy) > 0.015 for xy in excluded["positive_xy"]))
        with self.assertRaises(ValueError):
            select_contact_prompts(K, depth, mask, contact, tool, search_radius_m=0.03, depth_margin_m=0.01,
                                   max_positive=5, tool_exclusion_radius_m=0.01)

    def test_thresholds_must_be_explicit(self):
        depth, mask, contact, tool, _ = synthetic_frame()
        with self.assertRaises(ValueError):
            select_contact_prompts(K, depth, mask, contact, tool, search_radius_m=None, depth_margin_m=0.01,
                                   max_positive=5)


class Sam2AdapterTests(unittest.TestCase):
    def test_proposal_cannot_leave_object_mask_or_include_robot(self):
        depth, mask, contact, tool, tool_pixels = synthetic_frame()
        everything = np.ones((H, W), np.uint8)
        clipped = constrain_proposal(everything, mask)
        self.assertFalse((clipped & ~mask).any())
        with self.assertRaisesRegex(ValueError, "robot"):
            constrain_proposal(everything, np.ones((H, W), bool), [next(iter(tool_pixels))])

    def test_invalid_prompt_creates_no_guidance(self):
        with self.assertRaisesRegex(ValueError, "no_valid_positive_prompt"):
            read_contact_guidance(dict(status=STATUS_NO_POSITIVE, positive_xy=[], reason="hidden"), np.ones((4, 4)))
        depth, mask, contact, tool, _ = synthetic_frame()
        frame = select_contact_prompts(K, depth, mask, contact, tool, search_radius_m=0.03, depth_margin_m=0.01,
                                       max_positive=3)
        self.assertEqual(frame["status"], STATUS_VALID)
        box, points, labels = read_contact_guidance(frame, mask)
        self.assertEqual(labels[:3].tolist(), [1, 1, 1])
        self.assertTrue((labels[3:] == 0).all())
        self.assertTrue(box[0] <= points[0, 0] < box[2])


class PromptArtifactTests(unittest.TestCase):
    def test_end_to_end_prompts_and_seed_artifact(self):
        depth, mask, contact, _, _ = synthetic_frame()
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene"
            # cam_000 looks along base +Y from y=-1: base point (x, y, z) -> camera (x, -z, y + 1).
            p_base = np.array([contact[0], contact[2] - 1.0, -contact[1]])
            make_scene(scene, num_output=4, tcp_positions=np.repeat(p_base[None], 4, 0))
            cam = make_freeart_camera(Path(tmp) / "cam0", scene, "cam_000", [0, 1, 2, 3], depth=depth, mask=mask)
            contact_file = Path(tmp) / "contact.json"
            contact_file.write_text(json.dumps(dict(intervals=[[12, 14]], provenance="test")))
            seq = load_aligned_tcp(scene, cam, contact_source=contact_file)
            np.testing.assert_allclose(seq.uv_contact_fullres_xy[0], [40, 25], atol=1e-6)
            save_tcp_contact(seq, cam / "metadata/tcp_contact")
            prompts, report = generate(cam, cam / "metadata/tcp_contact", search_radius_m=0.03, depth_margin_m=0.01,
                                       max_positive=4)
            self.assertEqual(report["status_counts"], {"no_contact": 2, "valid": 2})
            frame = frame_prompt(prompts, "cam_000", 12)
            self.assertEqual(frame["status"], STATUS_VALID)
            manifest = write_outputs(prompts, report, cam / "metadata", cam)
            self.assertEqual(manifest["frames_with_seeds"], 2)
            with np.load(cam / "metadata/moving_prompts/moving_seed_maps.npz") as data:
                self.assertEqual(data["valid"].tolist(), [False, True, True, False])
                pts = data["points_xy"]
                self.assertTrue(all(mask[y, x] for x, y in pts))
            self.assertTrue((cam / "metadata/contact_guidance/proposed_moving_positive/cam_000/frame_000001.png").is_file())
            self.assertTrue((cam / "masks/00001.png").is_file())  # object masks untouched


class SeedSegmentationTests(unittest.TestCase):
    def test_seed_map_consolidates_without_inverting(self):
        weights = np.full((4, 4), 0.2, np.float32)
        weights[:, 2:] = 0.8
        seeds = np.zeros((4, 4), bool)
        seeds[0, 1:4] = True  # 2 of 3 seeds in the 0.8 cluster
        out, info = apply_part_seed_map(weights, np.ones((4, 4), bool), seeds)
        self.assertEqual(info["seed_cluster_value"], 0.8)
        self.assertAlmostEqual(float(out[0, 1]), 0.8, places=6)
        self.assertAlmostEqual(float(out[1, 1]), 0.2, places=6)  # non-seed pixels untouched
        _, info = apply_part_seed_map(weights, np.zeros((4, 4), bool), seeds)
        self.assertEqual(info["status"], "skipped")

    def make_tracks(self, n=400, seeded=60, stuck=20, seed=0):
        """Moving half translates; ``stuck`` moving tracks near the contact are uninformative
        (symmetric noise as large as the motion, e.g. tool occlusion): GT moving, visually ambiguous."""
        rng = np.random.default_rng(seed)
        src = rng.uniform(-0.2, 0.2, (n, 3)) + np.array([0, 0, 1.0])
        moving = np.arange(n) >= n // 2
        disp = np.array([0.05, 0.0, 0.0])
        tgt = src.copy()
        tgt[moving] += disp
        amb = np.arange(n // 2, n // 2 + stuck)
        tgt[amb] = src[amb] + 0.5 * disp + rng.normal(0, 0.05, (stuck, 3))
        tgt += rng.normal(0, 0.0005, tgt.shape)
        seeds = np.zeros(n, bool)
        seeds[n // 2:n // 2 + seeded] = True  # contact-adjacent prompts: stuck tracks + clean moving ones
        return src, tgt, moving, seeds, amb

    def test_branch_resolution_follows_motion_and_swaps(self):
        src, tgt, _, seeds, _ = self.make_tracks()
        s, t = torch.tensor(src, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)
        T_static = torch.eye(4).repeat(len(src), 1, 1)
        T_moving = T_static.clone()
        T_moving[:, 0, 3] = 0.05
        mask = torch.tensor(seeds)
        self.assertEqual(resolve_seed_branch(s, t, T_static, T_moving, mask, 1)[0], BRANCH_T2)
        self.assertEqual(resolve_seed_branch(s, t, T_moving, T_static, mask, 1)[0], BRANCH_T1)
        # A seed on a static-looking track cannot claim the moving label by fiat.
        static_seeds = torch.zeros(len(src), dtype=torch.bool)
        static_seeds[:10] = True
        branch, info = resolve_seed_branch(s, t, T_static, T_moving, static_seeds, 1)
        self.assertIsNone(branch)
        self.assertEqual(info["reason"], "fit_and_motion_disagree")
        self.assertIsNone(resolve_seed_branch(s, t, T_static, T_moving, torch.zeros(len(src), dtype=torch.bool), 1)[0])

    def test_seed_bce_label_swap_invariant(self):
        logits = torch.randn(10, 1)
        idx = torch.tensor([1, 3, 5])
        self.assertAlmostEqual(float(moving_seed_bce(logits, idx, BRANCH_T2)),
                               float(moving_seed_bce(-logits, idx, BRANCH_T1)), places=6)
        with self.assertRaises(ValueError):
            SeedTerm(torch.ones(3, dtype=torch.bool), weight=0.0)

    def solve(self, init_moving_value, use_seeds):
        from estimotion_2part.solve_transformation import solve_window_transformations
        torch.manual_seed(0)
        src, tgt, moving, seeds, amb = self.make_tracks()
        n = len(src)
        side = int(np.ceil(np.sqrt(n)))
        ys, xs = np.divmod(np.arange(n), side)
        init = np.where(moving, init_moving_value, 1 - init_moving_value).astype(np.float32)
        init[amb] = 1 - init_moving_value  # mistaken initial labels near the contact
        args = SimpleNamespace(lr_w=0.05, lr_T=1e-4, w_p=0.0, projection_normalizer=64.0, w_m=200.0, scale_obj=0.4,
                               w_e=-0.01, w_s=0.0, w_init=10.0, loss_type="normalized_huber",
                               solve_window_epochs=150, neighbor_selection_method="knn", knn_k=4, knn_radius=0.3,
                               debug=False, use_scheduler=False)
        first_frame = dict(cam_K=K.astype(np.float32), features=torch.randn(4, side, side),
                           rgbs=np.zeros((side, side, 3), np.uint8))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # solver's own device rule
        term = SeedTerm(torch.tensor(seeds, device=device), weight=5.0) if use_seeds else None
        _, _, weights, _ = solve_window_transformations(
            args, torch.tensor(np.stack([src, tgt], 1), dtype=torch.float32, device=device),
            torch.tensor(init, device=device)[:, None],
            (torch.tensor(ys, device=device), torch.tensor(xs, device=device)), None, first_frame,
            torch.ones((side, side), device=device), seed_term=term)
        weights = weights.detach().cpu().numpy()
        moving_label = (weights > 0.5) if init_moving_value > 0.5 else (weights < 0.5)
        return float(moving_label[amb].mean()), float(moving_label[moving].mean()), term

    def test_seeds_improve_moving_part_accuracy_for_either_label_convention(self):
        for init_value, expected_branch in ((0.8, BRANCH_T2), (0.2, BRANCH_T1)):
            base_amb, base_all, _ = self.solve(init_value, use_seeds=False)
            seed_amb, seed_all, term = self.solve(init_value, use_seeds=True)
            self.assertEqual(term.branch, expected_branch)
            self.assertLess(base_amb, 0.5)      # baseline keeps most mistaken initial labels
            self.assertGreater(seed_amb, 0.8)   # prompt-supported seeds recover the moving label
            self.assertGreater(seed_all, base_all)


if __name__ == "__main__":
    unittest.main()
