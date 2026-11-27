"""Joint selection v2 (observable physical model selection): spec section 11, Gates A-E.

Unit tests are deterministic and fast; the few end-to-end synthetic runs (marked E2E) take ~1 min each.
Ground truth of the synthetic scenes is read ONLY in assertions, never by the estimator.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery.assembly_geometry import (attachment_evidence, guide_evidence,  # noqa: E402
                                                                hinge_plausibility)
from estimotion_2part.joint_discovery.blocked_statistics import (block_bootstrap, cluster_jackknife,  # noqa: E402
                                                                 count_independent_blocks, make_block_partition,
                                                                 spatial_clusters)
from estimotion_2part.joint_discovery.cli import run_discovery  # noqa: E402
from estimotion_2part.joint_discovery.config import JointDiscoveryConfig, load_joint_discovery_config  # noqa: E402
from estimotion_2part.joint_discovery.motion_field import (PointTable, chained_intervals,  # noqa: E402
                                                           estimate_interval_rigid_evidence,
                                                           estimate_shared_angular_axis, rotation_vs_translation_evidence)
from estimotion_2part.joint_discovery.observations import TrackObservations  # noqa: E402
from estimotion_2part.joint_discovery.physical_selection import (PhysicalDecision, render_veto_allowed,  # noqa: E402
                                                                 select_physical_joint)
from estimotion_2part.joint_discovery.rigid_kinematics import exp_so3_axis_angle, normalize  # noqa: E402
from estimotion_2part.joint_discovery.rotation_observability import (calibrate_rotation_null,  # noqa: E402
                                                                     support_configuration)
from estimotion_2part.joint_discovery.selection_v2 import robot_evidence_v2  # noqa: E402
from estimotion_2part.joint_discovery.support_consistency import (candidate_independent_support,  # noqa: E402
                                                                  consistency_report)
from estimotion_2part.joint_discovery.synthetic import make_scene  # noqa: E402

NOISE = JointDiscoveryConfig().noise


def calibrated(obs):
    from estimotion_2part.joint_discovery.model_selection import calibrated_noise
    return calibrated_noise(obs, JointDiscoveryConfig())[0]


def field(sc, step=3, cams=None):
    noise = calibrated(sc.obs)
    tab = PointTable(sc.obs, noise, sc.calibs)
    sup = candidate_independent_support(sc.obs, noise)
    T = sc.obs.frame_map.T
    bnd = sorted(set(range(0, T, step)) | {T - 1})
    ivs = chained_intervals(tab, sup["M_proposal"].astype(float), bnd, noise, sc.calibs, cameras=cams)
    sa = estimate_shared_angular_axis(ivs, robust_config=dict(n_boot=200))
    null = calibrate_rotation_null(tab, sup["S_core"], ivs, axis=sa.get("axis"), n_patches=20, calibs=sc.calibs)
    ev = rotation_vs_translation_evidence(ivs, sa, null=null)
    return tab, sup, ivs, sa, null, ev


def duplicate_tracks(obs: TrackObservations, times=2) -> TrackObservations:
    N = obs.n_tracks
    kw = {}
    from estimotion_2part.joint_discovery.observations import OBS_FIELDS
    for name in OBS_FIELDS:
        arr = getattr(obs, name)
        kw[name] = np.concatenate([arr] * times)
    kw["track"] = np.concatenate([obs.track + k * N for k in range(times)])
    return TrackObservations(obs.camera_ids, obs.frame_map, track_camera=np.tile(obs.track_camera, times),
                             track_local_id=np.tile(obs.track_local_id, times),
                             track_query_time=np.tile(obs.track_query_time, times), **kw)


def v2_cfg():
    c = JointDiscoveryConfig(mode="calibrated_joint_aware")
    c.fit.em_iterations, c.fit.n_seeds = 3, 3
    c.comparison.selection = "observable_physical_v2"
    c.comparison.uncertainty_method = "cluster_robust"
    c.comparison.remote_hinge_policy = "observability"
    c.selection_v2.null_patches = 20
    return c.validate()


def run_v2(sc):
    with tempfile.TemporaryDirectory() as td:
        run_discovery(sc.obs, sc.calibs, v2_cfg(), Path(td) / "r", scene="synthetic")
        sel = json.loads((Path(td) / "r" / "joint_selection_v2.json").read_text())
        part = json.loads((Path(td) / "r" / "model_comparison" / "block_partition.json").read_text())
        adapter = sorted(p.name for p in (Path(td) / "r" / "freeart_adapter").iterdir())
    return sel, part, adapter


# =========================================================================== Gate A
class GateA(unittest.TestCase):
    def test_a1_pure_prismatic_has_no_confident_rotation(self):
        sc = make_scene("prismatic", seed=2, T=30, n_moving=150)
        *_, sa, null, ev = field(sc)
        self.assertFalse(ev["rotation_detected"])
        if ev.get("net_rotation_deg") is not None:
            self.assertLess(abs(ev["net_rotation_deg"]), 1.0)

    def test_a2_pure_revolute_shared_rotation(self):
        sc = make_scene("revolute", seed=1, T=30, n_moving=150, max_angle_deg=30)
        *_, sa, null, ev = field(sc)
        self.assertTrue(ev["rotation_detected"])
        self.assertAlmostEqual(abs(ev["net_rotation_deg"]), 30.0, delta=2.0)
        self.assertLess(np.degrees(np.arccos(abs(np.dot(sa["axis"], sc.gt_axis)))), 2.0)

    def test_a3_exact_finite_pair_identity(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 3)) * 0.1
        for deg in (15.0, 40.0, -25.0):
            u = normalize(rng.normal(size=3))
            o = rng.normal(size=3)
            R = exp_so3_axis_angle(u, np.radians(deg))
            Y = (X - o) @ R.T + o
            dX = Y - X
            i, j = 3, 17
            np.testing.assert_allclose(dX[i] - dX[j], (R - np.eye(3)) @ (X[i] - X[j]), atol=1e-12)
            # the robust interval estimator recovers the exact finite rotation from noiseless correspondences
            sc = make_scene("revolute", seed=4, T=6, n_moving=80, max_angle_deg=abs(deg), sigma_px=1e-6, sigma_z=1e-9,
                            axis=u, hinge_offset=o * 0.1)
            tab = PointTable(sc.obs, NOISE, sc.calibs)
            ev = estimate_interval_rigid_evidence(tab, sc.gt_moving.astype(float), 0, 5, NOISE, sc.calibs,
                                                  refine_2d=False)
            Rgt = exp_so3_axis_angle(sc.gt_axis, sc.gt_q[5] - sc.gt_q[0])
            self.assertLess(np.linalg.norm(ev.rotation_matrix - Rgt), 1e-5)

    def test_a4_translation_invariance(self):
        sc = make_scene("revolute", seed=5, T=8, n_moving=100, max_angle_deg=20)
        tab = PointTable(sc.obs, NOISE, sc.calibs)
        e1 = estimate_interval_rigid_evidence(tab, sc.gt_moving.astype(float), 0, 7, NOISE, None, refine_2d=False)
        tab.X = tab.X + np.array([3.0, -2.0, 1.0])
        e2 = estimate_interval_rigid_evidence(tab, sc.gt_moving.astype(float), 0, 7, NOISE, None, refine_2d=False)
        np.testing.assert_allclose(e1.rotation_vector_rad, e2.rotation_vector_rad, atol=1e-9)

    def test_a5_sign_gauge(self):
        a = make_scene("revolute", seed=6, T=24, n_moving=120, max_angle_deg=25)
        b = make_scene("revolute", seed=6, T=24, n_moving=120, max_angle_deg=-25)
        sa_a, sa_b = field(a)[3], field(b)[3]
        self.assertGreaterEqual(sa_a["net_angle_rad"], 0)
        self.assertGreaterEqual(sa_b["net_angle_rad"], 0)
        self.assertLess(np.dot(sa_a["axis"], sa_b["axis"]), -0.99)   # reversed motion -> flipped axis, net >= 0

    def test_a6_duplicated_tracks_do_not_shrink_uncertainty(self):
        sc = make_scene("revolute", seed=7, T=10, n_moving=120, max_angle_deg=20, sigma_px=1.0, sigma_z=0.004)
        e1 = estimate_interval_rigid_evidence(PointTable(sc.obs, NOISE, sc.calibs), sc.gt_moving.astype(float), 0, 9,
                                              NOISE, None, refine_2d=False)
        dup = duplicate_tracks(sc.obs, 3)
        e3 = estimate_interval_rigid_evidence(PointTable(dup, NOISE, sc.calibs), np.tile(sc.gt_moving, 3).astype(float),
                                              0, 9, NOISE, None, refine_2d=False)
        s1, s3 = np.sqrt(np.trace(e1.rotation_cov_rad2)), np.sqrt(np.trace(e3.rotation_cov_rad2))
        self.assertGreater(s3, 0.7 * s1)
        self.assertAlmostEqual(e1.n_effective_tracks, e3.n_effective_tracks, delta=0.5)

    def test_a7_low_rank_support_reports_ill_conditioning(self):
        rng = np.random.default_rng(0)
        line = np.c_[np.linspace(0, 0.3, 50), np.zeros(50), np.zeros(50)] + rng.normal(0, 1e-4, (50, 3))
        conf = support_configuration(line)
        self.assertEqual(conf["rank"], 1)
        self.assertGreater(conf["condition"], 50)
        self.assertLess(conf["baseline_m"], 0.01)

    def test_a8_partial_and_asynchronous_tracks(self):
        sc = make_scene("revolute", seed=9, T=30, n_moving=180, max_angle_deg=30, late_tracks_fraction=0.5, dropout=0.3)
        *_, ev = field(sc)
        self.assertTrue(ev["rotation_detected"])
        self.assertAlmostEqual(abs(ev["net_rotation_deg"]), 30.0, delta=3.0)

    def test_a9_static_base_contamination_does_not_create_rotation(self):
        sc = make_scene("prismatic", seed=10, T=24, n_moving=150)
        tab = PointTable(sc.obs, NOISE, sc.calibs)
        w = sc.gt_moving.astype(float)
        w[np.flatnonzero(~sc.gt_moving)[:60]] = 1.0       # 60 static base tracks mislabelled moving
        ivs = chained_intervals(tab, w, [0, 6, 12, 18, 23], NOISE, None, refine_2d=False)
        sa = estimate_shared_angular_axis(ivs, robust_config=dict(n_boot=100))
        tot = sum(np.degrees(iv.angle_rad) for iv in ivs)
        self.assertLess(tot, 1.5)
        self.assertTrue(sa.get("axis") is None or abs(np.degrees(sa["net_angle_rad"])) < 1.5)


# =========================================================================== Gate B
class GateB(unittest.TestCase):
    def test_b12_remote_hinge_slide_is_not_a_physical_revolute(self):          # E2E
        sc = make_scene("revolute", seed=8, T=30, n_moving=150, axis=[0, 0, 1], hinge_offset=[3.0, 0.0, 0.05],
                        max_angle_deg=1.9)
        sel, part, adapter = run_v2(sc)
        self.assertNotEqual(sel["physical_joint_status"], "IDENTIFIED_REVOLUTE")
        self.assertFalse(sel["physics_joint_export_ready"])
        self.assertIn("ABSTAINED.json", adapter)

    def test_b13_short_arc_real_hinge_is_not_switched_to_prismatic(self):    # E2E
        sc = make_scene("revolute", seed=9, T=30, n_moving=150, max_angle_deg=8.0)
        sel, _, _ = run_v2(sc)
        self.assertNotEqual(sel["physical_joint_status"], "IDENTIFIED_PRISMATIC")
        self.assertEqual(sel["physical_joint_status"], "IDENTIFIED_REVOLUTE")

    def _interface(self):
        sc = make_scene("revolute", seed=3, T=12, n_moving=150, max_angle_deg=30)
        noise = calibrated(sc.obs)
        tab = PointTable(sc.obs, noise, sc.calibs)
        sup = candidate_independent_support(sc.obs, noise)
        return sc, attachment_evidence(tab, sup, 0, gap_m=0.04)

    def test_b14_hinge_just_outside_box_is_not_vetoed(self):
        sc, att = self._interface()
        self.assertEqual(att["status"], "OK")
        hi = np.asarray(att["object_aabb"][1])
        o = sc.gt_origin.copy()
        o[1] = hi[1] + 0.02                                   # 2 cm outside the observed box
        hp = hinge_plausibility(att, sc.gt_axis, o)
        self.assertIn(hp["verdict"], ("SUPPORTED", "WEAK"))
        self.assertNotIn("veto", json.dumps(hp).lower().replace("not a bounding-box veto", ""))

    def test_b15_unobservable_interface_abstains(self):
        sc = make_scene("revolute", seed=3, T=12, n_moving=150, cameras=("cam_001",))
        tab = PointTable(sc.obs, NOISE, sc.calibs)
        sup = candidate_independent_support(sc.obs, NOISE)
        att = attachment_evidence(tab, sup, 0, min_cameras=2)
        self.assertEqual(att["status"], "ABSTAIN")
        self.assertEqual(hinge_plausibility(att, [1, 0, 0], [0, 0, 0])["status"], "ABSTAIN")

    def test_b16_rotation_on_a_second_rigid_group_is_inconsistent(self):
        sc = make_scene("prismatic", seed=2, T=24, n_moving=150, extra_rigid_group=dict(
            n=120, center=[0.25, -0.25, 0.15], size=[0.08, 0.08, 0.08], axis=[0, 0, 1], max_angle_deg=35,
            translation=[0.0, -0.02, 0.06]))
        noise = calibrated(sc.obs)
        tab = PointTable(sc.obs, noise, sc.calibs)
        sup = candidate_independent_support(sc.obs, noise)
        ivs = chained_intervals(tab, sup["M_proposal"].astype(float), [0, 6, 12, 18, 23], noise, None, refine_2d=False)
        rep = consistency_report(sc.obs, tab, sup, ivs)
        self.assertTrue(rep["inconsistent"], rep["inconsistency_flags"])

    def test_guide_requires_elongated_aligned_interface(self):
        ev = dict(status="OK", principal_axes=[[0, 1, 0], [1, 0, 0], [0, 0, 1]], elongation=5.0)
        self.assertEqual(guide_evidence(ev, [0, 1, 0])["verdict"], "GUIDE_ALIGNED")
        self.assertEqual(guide_evidence(ev, [1, 0, 0])["status"], "ABSTAIN")
        self.assertEqual(guide_evidence(dict(ev, elongation=1.2), [0, 1, 0])["status"], "ABSTAIN")


# =========================================================================== Gate C (selector logic)
def evidence(rotation_detected=False, axis_ok=False, net=0.0, trans=0.1, tnull=0.005, inconsistent=False, hinge_ok=True,
             hinge_verdict="SUPPORTED", guide=False, implied=20.0, ub=1.0, mdr=1.0, ratio=1.0, n_mov=200,
             heldout_pref="none", fit_axis=(0, 0, 1)):
    rot = dict(rotation_detected=rotation_detected, net_rotation_deg=net, net_rotation_ci95_deg=[net - 2, net + 2],
               rotation_upper_bound_deg=ub, net_centroid_translation_norm_m=trans, per_camera={})
    return dict(
        motion_evidence=dict(rotation=rot, shared_axis=dict(axis=[0, 0, 1] if axis_ok else None, axis_identifiable=axis_ok,
                                                            axis_cone95_deg=2.0),
                             hinge=dict(status="OK" if hinge_ok else "HINGE_UNOBSERVABLE", line_sd_m=0.01),
                             null=dict(net_abs_rotation_q95_deg=mdr, net_translation_q95_m=tnull),
                             support_quality=dict(moving_over_static_residual=ratio)),
        model_fits=dict(revolute=dict(axis=list(fit_axis), origin=[0, 0, 0]), prismatic=dict(axis=[1, 0, 0])),
        common_support=dict(counts=dict(M_proposal=n_mov), consistency=dict(inconsistent=inconsistent,
                                                                             inconsistency_flags=["x"] if inconsistent else [])),
        attachment_evidence=dict(hinge_field=dict(status="OK", verdict=hinge_verdict, d10_to_interface_m=0.01),
                                 hinge_fit=dict(status="OK", verdict=hinge_verdict),
                                 guide=dict(verdict="GUIDE_ALIGNED" if guide else "NO_OBSERVABLE_GUIDE"),
                                 implied=dict(implied_min_rotation_deg=implied)),
        predictive_evidence=dict(separation=dict(status="INDISTINGUISHABLE", median_block_separation_over_sigma=0.3),
                                 heldout=dict(status="OK", preference=heldout_pref)))


class GateC(unittest.TestCase):
    def test_c18_no_motion(self):
        d = select_physical_joint(**evidence(trans=0.001, tnull=0.004))
        self.assertEqual((d.physical_joint_status, d.working_motion_model), ("NO_SIGNIFICANT_MOTION", "static"))

    def test_c19_unobservable_short_arc_vs_slide(self):
        d = select_physical_joint(**evidence(implied=2.0, ub=1.5, mdr=1.2))
        self.assertEqual(d.physical_joint_status, "TYPE_UNOBSERVABLE")
        self.assertEqual(d.working_motion_model, "prismatic")
        self.assertTrue(d.working_model_is_approximation)
        self.assertFalse(d.physics_joint_export_ready)
        self.assertIn("revolute", d.counterfactual)

    def test_c20_strong_revolute(self):
        d = select_physical_joint(**evidence(rotation_detected=True, axis_ok=True, net=25.0, heldout_pref="revolute"))
        self.assertEqual(d.physical_joint_status, "IDENTIFIED_REVOLUTE")
        self.assertTrue(d.physics_joint_export_ready)
        self.assertIn("prismatic", d.counterfactual)

    def test_c21_prismatic_needs_positive_guide_and_reliable_support(self):
        ok = select_physical_joint(**evidence(guide=True, implied=20.0, ub=1.0, mdr=1.0))
        self.assertEqual(ok.physical_joint_status, "IDENTIFIED_PRISMATIC")
        self.assertTrue(ok.physics_joint_export_ready)
        no_guide = select_physical_joint(**evidence(guide=False))
        self.assertEqual(no_guide.physical_joint_status, "TYPE_UNOBSERVABLE")
        noisy = select_physical_joint(**evidence(guide=True, ratio=3.5))   # corrupted moving tracks (black-box-like)
        self.assertEqual(noisy.physical_joint_status, "TYPE_UNOBSERVABLE")
        self.assertIn("UNRELIABLE_MOVING_SUPPORT", noisy.failure_codes)

    def test_c22_inconsistent_support(self):
        d = select_physical_joint(**evidence(inconsistent=True, rotation_detected=True, axis_ok=True, net=30))
        self.assertEqual((d.physical_joint_status, d.working_motion_model), ("MODEL_INCONSISTENT", "none"))

    def test_c_rotation_with_contradicted_hinge_fails_R_then_evaluates_P_never_certifies_it(self):
        # force-guided selection spec: a contradicted hinge fails THAT revolute geometry; the slide is evaluated on
        # the same evidence and kept as an uncertified WORKING model (a rotation was measured)
        ev = evidence(rotation_detected=True, axis_ok=True, net=15, hinge_verdict="CONTRADICTED")
        d = select_physical_joint(**ev)
        self.assertEqual((d.physical_joint_status, d.working_motion_model), ("TYPE_UNOBSERVABLE", "prismatic"))
        self.assertIn("HINGE_ASSEMBLY_CONTRADICTED", d.failure_codes)
        self.assertFalse(d.physics_joint_export_ready)
        self.assertEqual(select_physical_joint(**ev, legacy_r_first=True).physical_joint_status, "MODEL_INCONSISTENT")

    def test_c_rotation_but_heldout_prefers_slide_keeps_a_prismatic_working_model(self):
        ev = evidence(rotation_detected=True, axis_ok=True, net=15, heldout_pref="prismatic")
        d = select_physical_joint(**ev)
        self.assertEqual((d.physical_joint_status, d.working_motion_model), ("TYPE_UNOBSERVABLE", "prismatic"))
        self.assertIn("PREDICTIVE_CONTRADICTION", d.failure_codes)
        self.assertEqual(select_physical_joint(**ev, legacy_r_first=True).physical_joint_status, "MODEL_INCONSISTENT")

    def test_c_unobservable_hinge_is_not_identified(self):
        d = select_physical_joint(**evidence(rotation_detected=True, axis_ok=True, net=2, hinge_ok=False))
        self.assertEqual(d.physical_joint_status, "TYPE_UNOBSERVABLE")
        self.assertFalse(d.physics_joint_export_ready)

    def test_c23_duplication_cannot_increase_block_confidence(self):
        vals = [np.array([1.0, -0.5]), np.array([0.2]), np.array([-0.4, 0.1]), np.array([0.6])]
        a = block_bootstrap(vals, n_boot=3000, seed=1)
        b = block_bootstrap([np.tile(v, 10) / 10 for v in vals], n_boot=3000, seed=1)   # 10x duplicated pixels
        np.testing.assert_allclose(a["ci"], b["ci"], rtol=1e-9)
        cov1, _ = cluster_jackknife(lambda k: np.array([k.sum()]), spatial_clusters(np.random.default_rng(0).random((30, 3))))
        self.assertIsNotNone(cov1)

    def test_c25_heldout_blocks_never_enter_training(self):                   # E2E
        sc = make_scene("revolute", seed=1, T=30, n_moving=150, max_angle_deg=30)
        sel, part, _ = run_v2(sc)
        self.assertTrue(part["declared_before_fitting"])
        ho = set(part["heldout_times"])
        self.assertTrue(ho)
        self.assertTrue(set(part["train_times"]).isdisjoint(ho))
        self.assertEqual(sel["physical_joint_status"], "IDENTIFIED_REVOLUTE")
        self.assertTrue(sel["physics_joint_export_ready"])

    def test_decision_invariant_export_requires_identified(self):
        with self.assertRaises(ValueError):
            PhysicalDecision("TYPE_UNOBSERVABLE", "prismatic", True, True, None, [], {}, {}, [])


# =========================================================================== Gate D
class GateD(unittest.TestCase):
    def test_d28_adjacent_frames_are_not_independent_blocks(self):
        self.assertEqual(count_independent_blocks([9, 10, 11, 19, 20, 21], 3), 2)
        self.assertEqual(count_independent_blocks([9, 10, 11, 19, 20, 21], 1), 6)

    def test_d29_too_few_blocks_cannot_veto(self):
        r = render_veto_allowed(dict(status="OK", n_independent_blocks=2, n_seeds=3, difference=-0.01,
                                     between_seed_sd=0.001))
        self.assertFalse(r["allowed"])
        self.assertIn("NOT_IDENTIFIABLE_FROM_AVAILABLE_BLOCKS", r["reason"])

    def test_d30_d33_difference_within_seed_noise_is_not_a_veto(self):
        self.assertFalse(render_veto_allowed(dict(status="OK", n_independent_blocks=8, n_seeds=1, difference=-0.01,
                                                  between_seed_sd=None))["allowed"])
        self.assertFalse(render_veto_allowed(dict(status="OK", n_independent_blocks=8, n_seeds=3, difference=-0.01,
                                                  between_seed_sd=0.008))["allowed"])
        self.assertTrue(render_veto_allowed(dict(status="OK", n_independent_blocks=8, n_seeds=3, difference=-0.05,
                                                 between_seed_sd=0.005))["allowed"])

    def test_block_partition_separates_heldout_blocks(self):
        p = make_block_partition(30, 3, 0, 0.35)
        held = [k for k, h in enumerate(p.heldout) if h]
        self.assertTrue(held)
        self.assertTrue(all(b - a >= 2 for a, b in zip(held, held[1:])))
        self.assertNotIn(0, held)
        self.assertNotIn(len(p.blocks) - 1, held)


# =========================================================================== Gate E
class GateE(unittest.TestCase):
    def test_e36_robot_evidence_abstains_without_verified_gates(self):
        r = robot_evidence_v2(None, None)
        self.assertEqual(r["contact"]["status"], "ABSTAIN")
        self.assertEqual(r["force"]["status"], "ABSTAIN")

        class FakeContact:
            contact_valid = np.ones(5, bool)
            contact_source = "heuristic: |F| > 1 N"
            reasons = ["no independently verified contact"]

        class FakeCal:
            def readiness(self):
                return dict(ok=False, failed=["force sign not verified"])

        r = robot_evidence_v2(FakeContact(), FakeCal())
        self.assertEqual(r["contact"]["status"], "ABSTAIN")
        self.assertEqual(r["force"]["status"], "ABSTAIN")

    def test_e39_legacy_default_unchanged(self):
        c = load_joint_discovery_config(None)
        self.assertEqual(c.comparison.selection, "heldout_predictive")
        self.assertNotIn("selection_v2", c.to_dict())
        sc = make_scene("revolute", seed=3, T=12, n_moving=80)
        cfg = JointDiscoveryConfig(mode="calibrated_joint_aware")
        cfg.fit.em_iterations, cfg.fit.n_seeds = 2, 2
        with tempfile.TemporaryDirectory() as td:
            run_discovery(sc.obs, sc.calibs, cfg.validate(), Path(td) / "r", scene="synthetic")
            self.assertFalse((Path(td) / "r" / "joint_selection_v2.json").exists())

    def test_e38_g8_refuses_unidentified_or_unready_production_export(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from export_parttwin_handoff import HandoffAbstain, physical_provenance
        for sel in (dict(schema_version=2, status="TYPE_UNOBSERVABLE", physics_joint_export_ready=False),
                    dict(schema_version=2, status="MODEL_INCONSISTENT", physics_joint_export_ready=False),
                    dict(schema_version=2, status="IDENTIFIED_REVOLUTE", physics_joint_export_ready=False),
                    dict(status="IDENTIFIED", selected="revolute")):                  # legacy, no approval note
            with self.assertRaises(HandoffAbstain):
                physical_provenance(sel, "PASSED")
        ok = physical_provenance(dict(schema_version=2, status="IDENTIFIED_REVOLUTE", physics_joint_export_ready=True),
                                 "PASSED")
        self.assertEqual(ok["provenance"], "identified")
        with self.assertRaises(HandoffAbstain):
            physical_provenance(dict(schema_version=2, status="IDENTIFIED_REVOLUTE", physics_joint_export_ready=True),
                                "ABSTAIN")
        assumed = physical_provenance(dict(status="IDENTIFIED", selected="revolute"), "PASSED", "approved by user X")
        self.assertEqual(assumed["provenance"], "assumed")
        self.assertFalse(assumed["physics_joint_export_ready"])

    def test_v2_refuses_fold_override(self):
        sc = make_scene("revolute", seed=3, T=12, n_moving=60)
        with tempfile.TemporaryDirectory() as td, self.assertRaises(ValueError):
            run_discovery(sc.obs, sc.calibs, v2_cfg(), Path(td) / "r", fold=dict(reference_time_index=0,
                                                                                  heldout_times=[3, 6]))


if __name__ == "__main__":
    unittest.main()
