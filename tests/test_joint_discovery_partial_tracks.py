"""T09, T12, T22: partial tracks, valid-only depth sampling, explicit time mapping."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.config import JointDiscoveryConfig  # noqa: E402
from estimotion_2part.joint_discovery.geometry_fit import Problem  # noqa: E402
from estimotion_2part.joint_discovery.model_selection import fit_joint_hypotheses  # noqa: E402
from estimotion_2part.joint_discovery.observations import (FrameMap, TrackObservations,  # noqa: E402
                                                           build_from_dense_tracks, concatenate_observations,
                                                           from_window_track_dumps, load_scene_frame_map,
                                                           parse_frame_range, project, sample_depth_valid, unproject)
from estimotion_2part.joint_discovery.rigid_kinematics import axis_angle_deg  # noqa: E402
from estimotion_2part.joint_discovery.synthetic import look_at_camera, make_scene  # noqa: E402


def cfg():
    c = JointDiscoveryConfig(mode="calibrated_joint_aware")
    c.fit.em_iterations = 2
    c.fit.n_seeds = 2
    return c.validate()


class TestPartialTracks(unittest.TestCase):
    def test_T09_tracks_absent_at_frame_zero(self):
        sc = make_scene("revolute", seed=4, T=10, late_tracks_fraction=0.6, dropout=0.15)
        obs = sc.obs
        late = obs.track_query_time > 0
        self.assertGreater(late.sum(), 50)
        # no row exists before a track's query time: nothing is zero/identity filled
        self.assertTrue(np.all(obs.time >= obs.track_query_time[obs.track]))
        fit = fit_joint_hypotheses(obs, sc.calibs, cfg())
        c = fit.candidates["revolute"]
        self.assertLess(axis_angle_deg(c.axis_w, sc.gt_axis), 2.0)
        # late tracks receive anchors and support like any other track
        mv = np.nan_to_num(c.moving_probability) > 0.5
        late_moving_gt = late & sc.gt_moving
        self.assertGreater((mv & late_moving_gt).sum() / max(late_moving_gt.sum(), 1), 0.8)
        self.assertTrue(np.isfinite(c.canonical_anchors_w[late & fit.problem.eligible]).all())

    def test_T12_invalid_depth_keeps_2d_and_never_mixes_minus_inf(self):
        depth = np.full((20, 20), 1.0)
        depth[:, 10:] = 0.0          # missing depth on the right half
        depth[:5, :] = 2.5           # different surface on top
        uv = np.array([[9.5, 10.0],   # straddles valid/missing -> valid neighbours only, edge flag
                       [14.0, 10.0],  # fully missing
                       [5.0, 4.5],    # straddles two surfaces (1.0 vs 2.5)
                       [3.0, 12.0]])  # clean interior
        z, valid, edge = sample_depth_valid(depth, uv)
        self.assertTrue(valid[0] and edge[0])
        self.assertAlmostEqual(z[0], 1.0)          # not contaminated by -inf / 0
        self.assertFalse(valid[1])
        self.assertTrue(np.isnan(z[1]))
        self.assertTrue(edge[2])
        self.assertIn(round(float(z[2]), 6), (1.0, 2.5))  # one surface, never the 1.75 blend
        self.assertTrue(valid[3] and not edge[3])
        # the 2D observation survives invalid depth in the table
        cal = look_at_camera("cam_001", [0, -1, 0.5], [0, 0, 0], width=20, height=20, f=20)
        fm = FrameMap([0, 1], [0, 2], [0.0, 0.1])
        uvt = np.repeat(uv[None], 2, 0)
        masks = [np.ones((20, 20), bool)] * 2
        obs = build_from_dense_tracks(cal, fm, uvt, np.ones((2, 4)), [depth, depth], masks, 0)
        self.assertEqual(obs.n_tracks, 4)
        self.assertEqual(int(obs.uv_valid.sum()), 8)
        self.assertEqual(int(obs.depth_valid.sum()), 2)  # edge samples rejected by default; only the clean one
        self.assertTrue(np.isnan(obs.x_w[~obs.depth_valid]).all())

    def test_incidence_angle_depth_noise(self):
        from estimotion_2part.joint_discovery.observations import depth_sigma, incidence_cos_map
        K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
        v, u = np.mgrid[0:48, 0:64].astype(float)
        fronto = np.full((48, 64), 1.0)
        c = incidence_cos_map(fronto, K)
        self.assertAlmostEqual(float(c[24, 32]), 1.0, places=6)
        # plane tilted by 70 deg about the camera x axis: z = 1 + tan(70) * y  ->  ray at centre has cos = cos(70)
        t = np.tan(np.radians(70))
        y_over_z = (v - 24) / 100.0
        tilted = 1.0 / (1 - t * y_over_z)
        tilted[(1 - t * y_over_z) <= 0.2] = 0.0
        self.assertAlmostEqual(float(incidence_cos_map(tilted, K)[24, 32]), np.cos(np.radians(70)), places=3)
        n = JointDiscoveryConfig().noise
        s0 = depth_sigma(np.array([1.0]), n)
        np.testing.assert_allclose(depth_sigma(np.array([1.0]), n, None, np.array([0.5])), s0 / 0.5)
        np.testing.assert_allclose(depth_sigma(np.array([1.0]), n, None, np.array([0.01])), s0 / n.incidence_cos_floor)
        np.testing.assert_allclose(depth_sigma(np.array([1.0]), n, None, np.array([np.nan])), s0)  # unknown: no inflation

    def test_project_unproject_roundtrip(self):
        cal = look_at_camera("c", [0.3, -0.8, 0.6], [0, 0, 0.1])
        X = np.random.default_rng(0).normal(size=(10, 3)) * 0.1
        uv, z = project(X, cal.K, cal.T_CW)
        np.testing.assert_allclose(unproject(uv, z, cal.K, cal.T_WC), X, atol=1e-12)

    def test_T22_frame_map_never_shifts(self):
        with tempfile.TemporaryDirectory() as td:
            meta = Path(td) / "metadata"
            meta.mkdir()
            rows = ["frame_index,source_frame_index,timestamp_s"] + [f"{i},{2 * i},{0.04 * i:.2f}" for i in range(200)]
            (meta / "frame_timestamps.csv").write_text("\n".join(rows))
            idx = parse_frame_range("70:128:2")
            self.assertEqual(len(idx), 30)
            self.assertEqual((idx[0], idx[-1]), (70, 128))
            fm = load_scene_frame_map(td, idx)
            np.testing.assert_array_equal(fm.source_frame_index, 2 * np.array(idx))
            np.testing.assert_allclose(fm.timestamp_s, 0.04 * np.array(idx))
            self.assertEqual(fm.time_of_rgbd(72), 1)
            with self.assertRaises(KeyError):
                load_scene_frame_map(td, [70, 500])
        with self.assertRaises(ValueError):
            FrameMap([0, 2, 1], [0, 4, 2], [0, 0.1, 0.2]).validate()

    def test_T22_dump_adapter_time_and_pixel_mapping(self):
        cal = look_at_camera("cam_002", [0, -1, 0.4], [0, 0, 0.05], width=64, height=48, f=60)
        K_half = cal.K.copy()
        K_half[:2] *= 0.5
        K_half[:2, 2] = cal.K[:2, 2] * 0.5 - 0.25  # pixel-centre resize convention
        X = np.array([[0.0, 0.0, 0.05], [0.02, 0.01, 0.06]])
        xc = (X - cal.T_WC[:3, 3]) @ cal.T_WC[:3, :3]
        with tempfile.TemporaryDirectory() as td:
            pre = np.stack([xc, xc + [0.001, 0, 0]], 1)
            np.savez(Path(td) / "tracks_w007.npz", K=K_half, scale=0.5, frames=np.arange(7, 9),
                     pre_xyz_1=pre, pre_yx_1=np.array([[3, 4], [5, 6]]), keep_1=np.ones(2, bool))
            fm = FrameMap(np.arange(10), np.arange(10) * 2, np.arange(10) * 0.1)
            obs = from_window_track_dumps(td, cal, fm)
        self.assertEqual(sorted(set(obs.time.tolist())), [7, 8])     # clip frames 7, 8, not 0, 1
        self.assertTrue(np.all(obs.track_query_time == 7))
        uv_true, _ = project(X, cal.K, cal.T_CW)
        np.testing.assert_allclose(obs.uv[obs.time == 7], uv_true, atol=1e-9)
        np.testing.assert_allclose(obs.x_w[obs.time == 7], X, atol=1e-12)
        self.assertIn("-inf", obs.provenance["depth_sampler"])

    def test_concatenate_refuses_unsynchronized_cameras(self):
        a = make_scene("revolute", seed=0, T=6, cameras=("cam_001",)).obs
        b = make_scene("revolute", seed=0, T=7, cameras=("cam_002",)).obs
        with self.assertRaises(ValueError):
            concatenate_observations([a, b])

    def test_save_load_roundtrip_and_no_zero_padding(self):
        sc = make_scene("prismatic", seed=2, T=6, dropout=0.3)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "obs.npz"
            sc.obs.save(p)
            o2 = TrackObservations.load(p)
            json.loads(p.with_suffix(".json").read_text())
        np.testing.assert_array_equal(o2.track, sc.obs.track)
        np.testing.assert_allclose(o2.uv, sc.obs.uv)
        self.assertLess(o2.n_obs, o2.n_tracks * 6)  # missing observations are absent rows
        Problem(o2, sc.calibs, cfg(), 0)


if __name__ == "__main__":
    unittest.main()
