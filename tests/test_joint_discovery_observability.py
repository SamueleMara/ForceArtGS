"""T07, T24 and the static/no-motion outcome: abstention, held-out selection, no leakage."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.config import JointDiscoveryConfig  # noqa: E402
from estimotion_2part.joint_discovery.model_selection import (fit_joint_hypotheses, make_heldout_split,  # noqa: E402
                                                              observability_report, score_candidates_on_heldout)
from estimotion_2part.joint_discovery.synthetic import make_scene  # noqa: E402


def cfg():
    c = JointDiscoveryConfig(mode="calibrated_joint_aware")
    c.fit.em_iterations = 3
    c.fit.n_seeds = 5
    return c.validate()


def select(sc, c=None):
    c = c or cfg()
    T = sc.obs.frame_map.T
    ho = make_heldout_split(T, 0, c.comparison.heldout_time_fraction)
    tr = sc.obs.select_obs(~np.isin(sc.obs.time, ho))
    te = sc.obs.select_obs(np.isin(sc.obs.time, ho))
    fit = fit_joint_hypotheses(tr, sc.calibs, c, ref=0)
    return fit, score_candidates_on_heldout(fit, te, c), ho


class TestObservability(unittest.TestCase):
    def test_clear_revolute_is_selected(self):
        fit, dec, _ = select(make_scene("revolute", seed=21, max_angle_deg=35))
        self.assertEqual(dec.label, "revolute", dec.reasons)

    def test_T07_short_arc_remote_hinge_is_ambiguous(self):
        # 1.2 deg arc about a hinge 1.5 m away: curvature far below the noise -> must not be a confident R or P
        sc = make_scene("revolute", seed=22, max_angle_deg=1.2, hinge_offset=[0.0, 1.5, 0.13], sigma_px=0.8,
                        sigma_z=0.004)
        fit, dec, _ = select(sc)
        self.assertEqual(dec.label, "ambiguous/unobservable", dec.reasons)
        # both candidates are preserved, with degeneracy diagnostics
        self.assertEqual(set(fit.candidates), {"prismatic", "revolute"})
        rep = observability_report(fit.candidates["revolute"], fit.problem)
        self.assertTrue(rep["degeneracy_flags"], rep)

    def test_remote_hinge_win_is_not_a_confident_revolute(self):
        # 3 deg about a hinge 4 m away: the rotation is detectable, the 1-DOF type is not identifiable
        sc = make_scene("revolute", seed=25, max_angle_deg=3.0, hinge_offset=[0.0, 4.0, 0.13])
        fit, dec, _ = select(sc)
        self.assertEqual(dec.label, "ambiguous/unobservable", dec.reasons)
        self.assertEqual(dec.scores.get("occam_preference"), "prismatic", dec.reasons)

    def test_no_motion_is_static(self):
        sc = make_scene("revolute", seed=23, max_angle_deg=0.0)
        fit, dec, _ = select(sc)
        self.assertEqual(dec.label, "static/no_motion", dec.reasons)

    def test_T24_heldout_split_has_no_leakage(self):
        sc = make_scene("revolute", seed=24, T=10)
        c = cfg()
        ho = make_heldout_split(10, 0, 0.2)
        self.assertTrue(len(ho) >= 1 and 0 not in ho and 9 not in ho)
        tr = sc.obs.select_obs(~np.isin(sc.obs.time, ho))
        self.assertFalse(np.isin(tr.time, ho).any())
        fit_a = fit_joint_hypotheses(tr, sc.calibs, c, ref=0)
        # corrupt every held-out observation: the training fit must be bit-identical
        obs_bad = sc.obs.select_obs(np.ones(sc.obs.n_obs, bool))
        m = np.isin(obs_bad.time, ho)
        obs_bad.uv[m] += 50.0
        obs_bad.x_w[m] += 0.3
        obs_bad.z[m] += 0.3
        fit_b = fit_joint_hypotheses(obs_bad.select_obs(~np.isin(obs_bad.time, ho)), sc.calibs, c, ref=0)
        for k in fit_a.candidates:
            np.testing.assert_array_equal(fit_a.candidates[k].axis_w, fit_b.candidates[k].axis_w)
            np.testing.assert_array_equal(fit_a.candidates[k].q, fit_b.candidates[k].q)
        self.assertEqual(fit_a.noise_report, fit_b.noise_report)
        # ground truth is never an input: fitting APIs have no GT parameters
        import inspect
        for fn in (fit_joint_hypotheses, score_candidates_on_heldout):
            self.assertFalse(any("gt" in p or "oracle" in p for p in inspect.signature(fn).parameters))


if __name__ == "__main__":
    unittest.main()
