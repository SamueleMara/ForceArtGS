"""Gate A of the multimodal TCP spec: analytic contact-mode kinematics (sliding / stick_soft / stick_hard)."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.config import ContactModeConfig, load_joint_discovery_config  # noqa: E402
from estimotion_2part.joint_discovery.contact_modes import (ContactRows, ContactSegmentError,  # noqa: E402
                                                            MovingSurface, contact_discrimination, fit_contact_only,
                                                            load_contact_segments, predict_tcp, rigid_grasp_residual,
                                                            segment_diagnostics, segment_residuals, stick_anchor,
                                                            tool_rotation_about_point)
from estimotion_2part.joint_discovery.rigid_kinematics import apply_joint, line_to_line_distance, normalize  # noqa: E402
from estimotion_2part.joint_discovery.synthetic import make_tcp_contact  # noqa: E402


def setup(sc, mode=None, doc=None):
    doc = copy.deepcopy(doc or sc["segments_doc"])
    if mode:
        for s in doc["segments"]:
            s["slip_mode"] = mode
    segs, rep = load_contact_segments(doc, sc["frame_map"], sc["p_tcp"])
    surf = MovingSurface.from_points(sc["surface_points"], sc["camera_centers"])
    T = sc["frame_map"].T
    return segs, rep, surf, np.ones(T, bool)


def diag(sc, segs, surf, kind=None, axis=None, origin=None, q=None, train=None):
    T = sc["frame_map"].T
    tr = np.ones(T, bool) if train is None else train
    return [segment_diagnostics(kind or sc["kind"], sc["axis"] if axis is None else axis,
                                sc["origin"] if origin is None else origin, sc["q"] if q is None else q,
                                sc["p_tcp"], s, tr, surf) for s in segs]


class GateA(unittest.TestCase):
    def test_a01_prismatic_sliding(self):
        sc = make_tcp_contact("prismatic", "sliding", sigma_tcp=0.001)
        segs, _, surf, tr = setup(sc)
        d = diag(sc, segs, surf)[0]
        self.assertEqual(d["status"], "OK")
        self.assertGreater(d["canonical_path_length_m"], 0.1)          # contact point travels over the face
        self.assertLess(d["normal_residual_rms_m"], 0.003)
        self.assertFalse(d["contradicted"])

    def test_a02_revolute_sliding_and_wrong_stick(self):
        sc = make_tcp_contact("revolute", "sliding", sigma_tcp=0.001)
        segs, _, surf, _ = setup(sc)
        self.assertFalse(diag(sc, segs, surf)[0]["contradicted"])
        segs_h, _, _, _ = setup(sc, mode="stick_hard")
        dh = diag(sc, segs_h, surf)[0]
        self.assertTrue(dh["contradicted"])
        self.assertEqual(dh["failure_code"], "CONTACT_MODE_CONTRADICTED")

    def test_a03_prismatic_hard_stick_recovers_axis_and_q(self):
        sc = make_tcp_contact("prismatic", "stick_hard", sigma_tcp=0.001)
        segs, _, surf, tr = setup(sc)
        rows = ContactRows(segs, sc["p_tcp"], tr)
        fit = fit_contact_only("prismatic", rows, sc["frame_map"].T, 0, sc["axis"] + [0.2, 0.1, -0.1])
        self.assertLess(np.degrees(np.arccos(abs(fit["axis"] @ sc["axis"]))), 1.0)
        qs = fit["q"] * np.sign(fit["axis"] @ sc["axis"])
        self.assertLess(np.max(np.abs(qs - sc["q"])), 0.004)
        self.assertEqual(len(rows.anchors("prismatic", fit["axis"], None, fit["q"])), 1)   # one anchor, not per-frame
        self.assertEqual(rows.n_rows, 3 * sc["frame_map"].T)

    def test_a04_revolute_hard_stick_finite_arcs(self):
        # exact finite arcs (no small-angle linearisation). On a 15 deg arc the hinge radius is weakly observable from
        # one point (sagitta ~1.3 mm vs 0.5 mm noise): the fit must then REPORT a large axis uncertainty that covers
        # its error; the well-excited 40 deg arc must be recovered tightly.
        for deg, tol_axis, tol_q in ((15.0, None, 1.5), (40.0, 3.0, 1.0)):
            sc = make_tcp_contact("revolute", "stick_hard", max_angle_deg=deg, sigma_tcp=0.0005)
            segs, _, _, tr = setup(sc)
            rows = ContactRows(segs, sc["p_tcp"], tr)
            o0 = sc["origin"] + np.array([0.0, 0.02, -0.02])
            fit = fit_contact_only("revolute", rows, sc["frame_map"].T, 0, normalize(sc["axis"] + [0.05, 0.1, 0.0]), o0)
            err = np.degrees(np.arccos(min(1, abs(fit["axis"] @ sc["axis"]))))
            sd = np.degrees(np.sqrt(np.diag(fit["cov"])[:2]).max())
            self.assertLess(abs(fit["chi2_per_dof"] - 1.0), 0.3, deg)          # global optimum at the noise floor
            self.assertLess(err, 3 * sd + 0.5, deg)                              # uncertainty is honest
            if tol_axis is not None:
                self.assertLess(err, tol_axis, deg)
            qs = fit["q"] * np.sign(fit["axis"] @ sc["axis"])
            self.assertLess(np.degrees(np.max(np.abs(qs - sc["q"]))), tol_q, deg)

    def test_a05_soft_stick_accepts_drift_hard_flags_it(self):
        sc = make_tcp_contact("revolute", "stick_soft", sigma_tcp=0.001, drift_m=0.03)
        segs, _, surf, _ = setup(sc)
        self.assertFalse(diag(sc, segs, surf)[0]["contradicted"])
        segs_h, _, _, _ = setup(sc, mode="stick_hard")
        dh = diag(sc, segs_h, surf)[0]
        self.assertTrue(dh["contradicted"])
        self.assertGreater(dh["canonical_drift_max_m"], 0.006)

    def test_a06_release_regrasp_two_anchors(self):
        sc = make_tcp_contact("revolute", "stick_hard", sigma_tcp=0.001, release_at=15)
        segs, _, surf, tr = setup(sc)
        self.assertEqual(len(segs), 2)
        rows = ContactRows(segs, sc["p_tcp"], tr)
        an = rows.anchors("revolute", sc["axis"], sc["origin"], sc["q"])
        self.assertEqual(len(an), 2)
        self.assertGreater(np.linalg.norm(np.subtract(*an.values())), 0.05)
        self.assertFalse(any(d["contradicted"] for d in diag(sc, segs, surf)))
        # one anchor bridging the release gap is wrong -- and the gap is never interpolated
        doc = copy.deepcopy(sc["segments_doc"])
        doc["segments"] = [dict(doc["segments"][0], end_source_frame=doc["segments"][1]["end_source_frame"])]
        with self.assertRaises(ContactSegmentError):          # TCP missing during the released frames
            load_contact_segments(doc, sc["frame_map"], sc["p_tcp"])
        p = sc["p_tcp"].copy()
        p[15:17] = p[14]                                      # even with a filled gap the single anchor is contradicted
        segs1, _ = load_contact_segments(doc, sc["frame_map"], p)
        d1 = segment_diagnostics("revolute", sc["axis"], sc["origin"], sc["q"], p, segs1[0], tr, surf)
        self.assertTrue(d1["contradicted"])

    def test_a07_point_contact_allows_tool_rotation_rigid_grasp_does_not(self):
        sc = make_tcp_contact("revolute", "stick_hard", sigma_tcp=0.0)
        segs, _, _, tr = setup(sc)
        r0 = segment_residuals("revolute", sc["axis"], sc["origin"], sc["q"], sc["p_tcp"], segs[0], np.arange(30))
        self.assertLess(np.abs(r0).max(), 1e-6)            # positions only: wrist orientation is irrelevant
        T = np.eye(4)
        T[:3, 3] = sc["p_tcp"][10]
        A = np.eye(4)
        Tm = np.eye(4)
        Tm[:3, 3] = T[:3, 3]
        self.assertLess(np.abs(rigid_grasp_residual(T, A, Tm)).max(), 1e-12)
        rot = rigid_grasp_residual(tool_rotation_about_point(T, 0.3), A, Tm)
        self.assertAlmostEqual(np.linalg.norm(rot[:3]), 0.3, places=6)  # a rigid grasp WOULD be violated
        doc = copy.deepcopy(sc["segments_doc"])
        doc["segments"][0]["slip_mode"] = "rigid_grasp"
        with self.assertRaises(ContactSegmentError):
            load_contact_segments(doc, sc["frame_map"], sc["p_tcp"])

    def test_a08_unknown_contact_creates_no_rows(self):
        sc = make_tcp_contact("revolute", "stick_hard", state="contact_unknown", provenance="sensor_event")
        segs, rep, _, tr = setup(sc)
        self.assertEqual(ContactRows(segs, sc["p_tcp"], tr).n_rows, 0)
        self.assertTrue(rep["notes"])
        sc2 = make_tcp_contact("revolute", "stick_hard", state="contact_confirmed", provenance="sensor_event")
        with self.assertRaises(ContactSegmentError):
            load_contact_segments(sc2["segments_doc"], sc2["frame_map"], sc2["p_tcp"])
        sc3 = make_tcp_contact("revolute", "stick_hard")
        segs3, _, _, _ = setup(sc3)
        self.assertGreater(ContactRows(segs3, sc3["p_tcp"], tr).n_rows, 0)

    def test_a09_fail_closed_on_bad_tcp_or_frames(self):
        sc = make_tcp_contact("revolute", "stick_hard")
        p = sc["p_tcp"].copy()
        p[5] = np.nan
        with self.assertRaises(ContactSegmentError):
            load_contact_segments(sc["segments_doc"], sc["frame_map"], p)
        with self.assertRaises(ContactSegmentError):
            load_contact_segments(sc["segments_doc"], sc["frame_map"], None)
        bad = copy.deepcopy(sc["segments_doc"])
        bad["segments"][0]["start_source_frame"] = 1          # not an exact source frame (frames are 0,2,4,...)
        with self.assertRaises(ContactSegmentError):
            load_contact_segments(bad, sc["frame_map"], sc["p_tcp"])
        two = copy.deepcopy(sc["segments_doc"])
        two["segments"].append(dict(two["segments"][0], segment_id="dup"))
        with self.assertRaises(ContactSegmentError):
            load_contact_segments(two, sc["frame_map"], sc["p_tcp"])
        wrong = dict(sc["segments_doc"], frame_index_type="filename")
        with self.assertRaises(ContactSegmentError):
            load_contact_segments(wrong, sc["frame_map"], sc["p_tcp"])

    def test_a09b_segments_estimated_on_another_window_are_refused(self):
        sc = make_tcp_contact("revolute", "stick_hard")
        f = np.asarray(sc["frame_map"].rgbd_frame_index, int)
        k = int(f[1] - f[0])
        same = dict(sc["segments_doc"], window=f"{f[0]}:{f[-1]}:{k}")
        load_contact_segments(same, sc["frame_map"], sc["p_tcp"])               # estimated on these frames: accepted
        shifted = dict(sc["segments_doc"], window=f"{f[0] + 1}:{f[-1] + 1}:{k}")
        with self.assertRaisesRegex(ContactSegmentError, "re-estimate"):
            load_contact_segments(shifted, sc["frame_map"], sc["p_tcp"])
        legacy = {key: v for key, v in sc["segments_doc"].items() if key != "window"}
        load_contact_segments(legacy, sc["frame_map"], sc["p_tcp"])             # no window recorded: exact-frame rule only

    def test_a10_hard_stick_is_structural_and_numerically_stable(self):
        sc = make_tcp_contact("revolute", "stick_hard", sigma_tcp=0.001)
        segs, _, _, tr = setup(sc)
        rows = ContactRows(segs, sc["p_tcp"], tr)
        self.assertTrue(all(len(d) == sc["frame_map"].T for d in rows.row_q_deps))   # anchor couples the segment
        fit = fit_contact_only("revolute", rows, sc["frame_map"].T, 0, sc["axis"], sc["origin"])
        self.assertTrue(np.isfinite(fit["cov"]).all())
        self.assertTrue(np.isfinite(fit["condition"]))
        bad = copy.deepcopy(sc["segments_doc"])
        bad["segments"][0]["sigma_tcp_position_m"] = 0.0
        with self.assertRaises(ContactSegmentError):
            load_contact_segments(bad, sc["frame_map"], sc["p_tcp"])
        with self.assertRaises(ValueError):
            load_joint_discovery_config({"mode": "calibrated_joint_aware",
                                         "comparison": {"selection": "observable_physical_v2"},
                                         "multimodal_frontend": {"contact": {"sigma_tcp_position_m": 0}}})

    def test_a11_anchor_gauge(self):
        sc = make_tcp_contact("revolute", "stick_hard", sigma_tcp=0.0)
        t = np.arange(30)
        a = stick_anchor("revolute", sc["axis"], sc["origin"], sc["q"], sc["p_tcp"], t)
        r = 12                                                 # re-gauge q_ref -> 12, move the anchor with it
        q2 = sc["q"] - sc["q"][r]
        a2 = apply_joint("revolute", sc["axis"], sc["origin"], sc["q"][r], a)
        np.testing.assert_allclose(predict_tcp("revolute", sc["axis"], sc["origin"], sc["q"], a, t),
                                   predict_tcp("revolute", sc["axis"], sc["origin"], q2, a2, t), atol=1e-12)
        np.testing.assert_allclose(stick_anchor("revolute", sc["axis"], sc["origin"], q2, sc["p_tcp"], t), a2, atol=1e-12)

    def test_a12_remote_hinge_short_arc_is_not_discriminated_by_contact(self):
        sc = make_tcp_contact("revolute", "stick_hard", sigma_tcp=0.002, axis=[0, 0, 1], hinge=[3.0, 0.0, 0.05],
                              max_angle_deg=2.0)
        segs, _, surf, tr = setup(sc)
        rows = ContactRows(segs, sc["p_tcp"], tr)
        T = sc["frame_map"].T
        chord = sc["p_tcp"][-1] - sc["p_tcp"][0]
        fr = fit_contact_only("revolute", rows, T, 0, sc["axis"], sc["origin"])
        fp = fit_contact_only("prismatic", rows, T, 0, normalize(chord))
        dr = [segment_diagnostics("revolute", fr["axis"], fr["origin"], fr["q"], sc["p_tcp"], s, tr, surf) for s in segs]
        dp = [segment_diagnostics("prismatic", fp["axis"], None, fp["q"], sc["p_tcp"], s, tr, surf) for s in segs]
        self.assertNotEqual(contact_discrimination(dict(revolute=dr, prismatic=dp))["status"], "OK")


if __name__ == "__main__":
    unittest.main()
