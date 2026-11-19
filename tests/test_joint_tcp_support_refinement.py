"""Gate C (support/tracks part) of the multimodal TCP spec: tool occupancy, contact seeds, track recovery."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.config import JointDiscoveryConfig, TcpSupportConfig  # noqa: E402
from estimotion_2part.joint_discovery.model_selection import calibrated_noise  # noqa: E402
from estimotion_2part.joint_discovery.motion_field import PointTable  # noqa: E402
from estimotion_2part.joint_discovery.rigid_kinematics import apply_joint  # noqa: E402
from estimotion_2part.joint_discovery.support_consistency import (candidate_independent_support,  # noqa: E402
                                                                  tool_proximity)
from estimotion_2part.joint_discovery.synthetic import (SplatImageSource, make_contact_record,  # noqa: E402
                                                        make_scene)
from estimotion_2part.joint_discovery.tcp_support import (CLEAR, TOOL, ToolModel, classify_tracks_tool,  # noqa: E402
                                                          contact_seeds, sigma_track, update_support)
from estimotion_2part.joint_discovery.track_refinement import recover_tracks  # noqa: E402

CFG = TcpSupportConfig()
TOOLM = ToolModel(CFG.tool_radius_m, CFG.tool_ambiguity_margin_m, CFG.tool_extend_m, CFG.tool_radius_m, "APPROXIMATE",
                  "test")


def drawer_with_tool(seed=2):
    """Drawer pushed by a TCP stuck on its front face; a 'gripper shaft' rigid group moves with the TCP."""
    sc0 = make_scene("prismatic", seed=seed, T=24, n_moving=150)
    rec0, _, _ = make_contact_record(sc0, "stick_hard", sigma_tcp=0.0)
    tip = rec0.p_tcp_w[0]
    shaft = dict(n=80, center=(tip + [0, -0.08, 0]).tolist(), size=[0.016, 0.1, 0.016], axis=[0, 0, 1],
                 max_angle_deg=0.0, translation=(sc0.gt_axis * sc0.gt_q[-1]).tolist())
    sc = make_scene("prismatic", seed=seed, T=24, n_moving=150, extra_rigid_group=shaft)
    rec, cal, doc = make_contact_record(sc, "stick_hard", sigma_tcp=0.0)
    return sc, rec


def table(sc):
    noise = calibrated_noise(sc.obs, JointDiscoveryConfig())[0]
    return PointTable(sc.obs, noise, sc.calibs), noise


class GateC(unittest.TestCase):
    def test_c01_contact_seeds_and_tool_never_moving_support(self):
        sc, rec = drawer_with_tool()
        tab, noise = table(sc)
        tool = classify_tracks_tool(tab, rec.p_tcp_w, rec.p_ee_w, TOOLM)
        n_extra = int((sc.gt_moving.sum() - 2 * 150))          # shaft tracks (both cameras)
        self.assertGreater(tool["counts"][TOOL], 0.8 * n_extra)
        state = np.array(["contact_confirmed"] * 24, dtype=object)
        seeds = contact_seeds(tab, rec.p_tcp_w, state, tool["label"], CFG)
        self.assertGreater(seeds["n_positive"], 0)
        self.assertFalse((seeds["positive"] & (tool["label"] == TOOL)).any())
        base = candidate_independent_support(sc.obs, noise)
        sup = update_support(base, tool, seeds, sigma_track(tab), CFG)
        self.assertFalse((sup["M_proposal"] & sup["TOOL"]).any())
        self.assertGreater(sup["tcp_provenance"]["removed_as_tool"], 0)

    def test_c03_object_points_beside_the_tip_are_kept(self):
        sc, rec = drawer_with_tool()
        tab, _ = table(sc)
        tool = classify_tracks_tool(tab, rec.p_tcp_w, rec.p_ee_w, TOOLM)
        d0 = np.linalg.norm(tab.X[0] - rec.p_tcp_w[0], axis=1)
        drawer = sc.gt_moving & ~(np.arange(len(sc.gt_moving)) >= 0)  # placeholder overwritten below
        # drawer tracks (not the shaft): the shaft is the extra group, i.e. tracks whose GT anchor is above the face
        shaft = sc.gt_anchor_ref[:, 1] < rec.p_tcp_w[0, 1] - 0.025     # in front of the face, along the tool
        near = np.isfinite(d0) & (d0 < 0.03) & ~shaft & sc.gt_moving
        self.assertGreater(near.sum(), 0)
        self.assertTrue((tool["label"][near] == CLEAR).all())    # occupancy keeps the contacted object
        old = tool_proximity(tab, near, dict(p_tcp_w=rec.p_tcp_w, p_ee_w=rec.p_ee_w), radius_m=0.05)
        self.assertGreater(old["fraction_near_tool"], 0.5)       # the blanket 5 cm rule would have discarded them
        del drawer

    def test_c07_sliding_seed_follows_the_contact(self):
        sc = make_scene("revolute", seed=1, T=24, n_moving=200, max_angle_deg=20)
        rec, _, _ = make_contact_record(sc, "sliding", sigma_tcp=0.0)
        tab, _ = table(sc)
        lab = np.array([CLEAR] * tab.N, dtype=object)
        early = contact_seeds(tab, rec.p_tcp_w, np.array(["contact_confirmed"] * 3 + ["no_contact"] * 21, dtype=object),
                              lab, CFG)
        late = contact_seeds(tab, rec.p_tcp_w, np.array(["no_contact"] * 21 + ["contact_confirmed"] * 3, dtype=object),
                             lab, CFG)
        self.assertGreater(early["n_positive"], 0)
        self.assertGreater(late["n_positive"], 0)
        self.assertLess((early["positive"] & late["positive"]).sum(), 0.5 * min(early["n_positive"], late["n_positive"]))

    def test_c10_unobserved_tracks_stay_unknown(self):
        sc = make_scene("revolute", seed=3, T=12, n_moving=80, depth_invalid_fraction=0.9)
        noise = calibrated_noise(sc.obs, JointDiscoveryConfig())[0]
        sup = candidate_independent_support(sc.obs, noise)
        no3d = sup["n3d"] < 3
        self.assertTrue(no3d.any())
        self.assertFalse((sup["S_core"] & no3d).any())          # never forced static
        self.assertTrue(sup["U"][no3d].all())

    def test_c08_unknown_contact_gives_proposals_not_positives(self):
        sc, rec = drawer_with_tool()
        tab, _ = table(sc)
        tool = classify_tracks_tool(tab, rec.p_tcp_w, rec.p_ee_w, TOOLM)
        seeds = contact_seeds(tab, rec.p_tcp_w, np.array(["contact_unknown"] * 24, dtype=object), tool["label"], CFG)
        self.assertEqual(seeds["n_positive"], 0)
        self.assertGreater(seeds["n_proposal"], 0)

    def test_track_recovery_restores_dropped_observations(self):
        sc = make_scene("revolute", seed=4, T=16, n_moving=120, max_angle_deg=25, sigma_px=0.3)
        imgs = SplatImageSource(sc, n_texture=8000)
        obs = sc.obs
        rng = np.random.default_rng(0)
        mv = np.flatnonzero(sc.gt_moving)
        drop = np.isin(obs.track, mv) & np.isin(obs.time, [5, 6, 7, 9]) & (rng.random(obs.n_obs) < 0.6)
        held = [12, 13]
        kept = obs.select_obs(~drop & ~np.isin(obs.time, held))

        class C:
            pass
        cands = {}
        for kind in ("revolute",):
            c = C()
            c.kind, c.axis_w, c.origin_w, c.q, c.canonical_anchors_w = kind, sc.gt_axis, sc.gt_origin, sc.gt_q, sc.gt_anchor_ref
            cands[kind] = c
        train = [t for t in range(16) if t not in held]
        new, rec_mask, rep = recover_tracks(kept, imgs, sc.calibs, cands, sc.gt_moving, train, ncc_min=0.8)
        self.assertGreater(rep["accepted"], 20, rep)
        nn = np.flatnonzero(rec_mask)
        self.assertFalse(np.isin(new.time[nn], held).any())      # never a held-out time
        errs = []
        for k in nn:
            i, t, ci = new.track[k], new.time[k], new.cam[k]
            cal = sc.calibs[new.camera_ids[ci]]
            xw = apply_joint("revolute", sc.gt_axis, sc.gt_origin, np.array([sc.gt_q[t]]), sc.gt_anchor_ref[i][None])[0]
            xc = (xw - cal.T_WC[:3, 3]) @ cal.T_WC[:3, :3]
            uv = np.array([cal.K[0, 0] * xc[0] / xc[2] + cal.K[0, 2], cal.K[1, 1] * xc[1] / xc[2] + cal.K[1, 2]])
            errs.append(np.linalg.norm(new.uv[k] - uv))
        self.assertLess(np.median(errs), 1.5)
        self.assertFalse((new.depth_valid[nn] & new.depth_edge[nn]).any())   # no depth across object boundaries


if __name__ == "__main__":
    unittest.main()
