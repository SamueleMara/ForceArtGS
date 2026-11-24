"""G10 real-data bridge, URDF builder, optimizer bookkeeping, profile and status logic -- no engine needed."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from forceartgs_g9.dynamics.general_1dof import RigidBodyGravity, rotation
from forceartgs_g9.free_forced.segments import MotionSegment
from forceartgs_g9.physx_refinement import identifiability as ID
from forceartgs_g9.physx_refinement.api import PhysXIsaacRollout, estimate
from forceartgs_g9.physx_refinement.g9_adapter import episodes_from_g9_interactions, interactions_from_segments
from forceartgs_g9.physx_refinement.objective import Episode, PhysicalParameters, RefinementLoss, SimulationResult
from forceartgs_g9.physx_refinement.optimizer import FitConfiguration, fit_physx
from forceartgs_g9.physx_refinement.sapien_diagnostic import urdf_axis_inertia
from forceartgs_g9.physx_refinement.urdf_builder import pose_at_zero, write_single_joint_urdf


# ------------------------------------------------------------------------------------------------- URDF builder
def test_urdf_pose_at_zero_reproduces_g9_gravity_and_inertia(tmp_path):
    a = np.array([0.3, 0.1, 0.95]); a /= np.linalg.norm(a)
    o, c, m, q_ref = np.array([0.2, -0.1, 0.4]), np.array([0.6, 0.2, 0.5]), 3.0, 0.9
    J = np.array([[0.2, 0.02, 0.01], [0.02, 0.15, 0.0], [0.01, 0.0, 0.1]])
    info = write_single_joint_urdf(tmp_path / "m.urdf", joint_type="revolute", axis=a, origin=o, mass=m, com_ref=c,
                                   inertia_com_ref=J, q_ref=q_ref)
    # the COM at q = 0, carried to q_ref by the joint rotation, is the G9 reference COM
    c0 = np.array(info["com_q0"])
    np.testing.assert_allclose(o + rotation(a, q_ref) @ (c0 - o), c, atol=1e-12)
    # hinge inertia is invariant under the pose change and equals the URDF parse
    r = c - o
    Jo = J + m * ((r @ r) * np.eye(3) - np.outer(r, r))
    assert info["axis_inertia"] == pytest.approx(a @ Jo @ a, rel=1e-12)
    assert urdf_axis_inertia(info["path"], "moving_link", "joint")["axis_inertia"] == pytest.approx(
        info["axis_inertia"], rel=1e-9)
    # gravity of the q = 0 pose (q_ref = 0) equals the G9 law at every q
    g = (0.0, 0.0, -9.81)
    g9 = RigidBodyGravity("revolute", tuple(a), tuple(o), tuple(c), m, g, q_ref)
    urdf_pose = RigidBodyGravity("revolute", tuple(a), tuple(o), tuple(c0), m, g, 0.0)
    for q in np.linspace(-2, 2, 9):
        assert urdf_pose.G(q) == pytest.approx(g9.G(q), abs=1e-12)
    root = ET.parse(info["path"]).getroot()
    assert root.find(".//joint").get("type") == "continuous"
    assert root.find(".//collision") is None and root.find(".//visual") is None


def test_urdf_prismatic_and_refusals(tmp_path):
    a = np.array([0.0, 0.6, 0.8])
    _, _, c0, _ = pose_at_zero("prismatic", a, None, [1.0, 2.0, 3.0], np.eye(3), 0.5)
    np.testing.assert_allclose(c0, np.array([1.0, 2.0, 3.0]) - 0.5 * a)
    info = write_single_joint_urdf(tmp_path / "p.urdf", joint_type="prismatic", axis=a, origin=None, mass=4.0,
                                   com_ref=[1, 2, 3], inertia_com_ref=np.eye(3) * 0.1, q_ref=0.5)
    assert info["axis_inertia"] == 4.0
    assert ET.parse(info["path"]).getroot().find(".//limit") is not None
    with pytest.raises(ValueError, match="not physical"):
        write_single_joint_urdf(tmp_path / "x.urdf", joint_type="revolute", axis=[0, 0, 1], origin=[0, 0, 0],
                                mass=1.0, com_ref=[1, 0, 0], inertia_com_ref=np.diag([1.0, 0.1, 0.1]))
    with pytest.raises(ValueError, match="positive"):
        write_single_joint_urdf(tmp_path / "y.urdf", joint_type="revolute", axis=[0, 0, 1], origin=[0, 0, 0],
                                mass=0.0, com_ref=[1, 0, 0], inertia_com_ref=np.eye(3))


# ------------------------------------------------------------------------------------------------- segments
def seg(name, label, n=12, Qlev=1.5):
    t = np.round(np.arange(n) / 15.0 + 2.0, 6)
    ft = np.round(np.arange(t[0] - 0.1, t[-1] + 0.02, 0.02), 6)
    return MotionSegment(name, label, "revolute", t, 0.1 + 0.2 * (t - t[0]) + 0.05 * (t - t[0]) ** 2,
                         np.full(n, 1e-3), 1.0, ft, np.full(len(ft), Qlev), np.full(len(ft), 0.1),
                         dict(reason="MEASURED_CONTACT_EFFORT", interval_rows=[1, 2]))


def test_segments_bridge_strict_regimes_and_untouched_force():
    segs = [seg("forced_a", "FORCED_MEASURED"), seg("free_b", "FREE_CONFIRMED"),
            seg("hyp_c", "FREE_HYPOTHESIS:measured_near_zero_effort_in_contact"), seg("short", "FORCED_MEASURED", n=6)]
    Q_before = segs[0].Q.copy()
    inter, regime, v0s, cal, excl = interactions_from_segments(segs, force_audit_status="EXPLORATORY_ASSUMPTIONS_ACCEPTED")
    assert [i.name for i in inter] == ["forced_a", "free_b"]
    assert {x["name"] for x in excl} == {"hyp_c", "short"}
    assert "never converted to force-free" in next(x for x in excl if x["name"] == "hyp_c")["reason"]
    assert regime["forced_a"]["evidence"]["contact_confirmed"] is True
    assert regime["free_b"]["evidence"]["contact_independently_absent"] is True
    assert cal["forced_a"] == dict(status="EXPLORATORY_ASSUMPTIONS_ACCEPTED", exploratory_opt_in=True)
    np.testing.assert_array_equal(segs[0].Q, Q_before)                       # source effort untouched
    np.testing.assert_array_equal(inter[0].effort.levels, Q_before)          # same values, same sign
    np.testing.assert_array_equal(inter[0].effort.t_s, segs[0].force_t)      # native rows, no resampling
    # initial state from the 4-sample prefix: the episode starts at the 4th sample, q0/v0 from the local fit
    it = inter[0]
    assert it.t_obs[0] == segs[0].t[3]
    assert it.initial_state["qdot"] == pytest.approx(0.2 + 0.1 * (segs[0].t[3] - segs[0].t[0]), rel=1e-6)
    assert "NOT independent" in it.initial_state["source"]
    for e in episodes_from_g9_interactions(inter, regime, v0s, calibration_by_name=cal):
        assert e.q0 == pytest.approx(next(i for i in inter if i.name == e.name).initial_state["q"])
        assert e.mode == ("forced" if e.name == "forced_a" else "free")
    # a contradicted / unavailable force audit makes forced input inadmissible
    inter2, _, _, _, excl2 = interactions_from_segments(segs[:1], force_audit_status="CALIBRATION_ASSUMPTION_CONTRADICTED")
    assert inter2 == [] and "not admissible" in excl2[0]["reason"]


# ------------------------------------------------------------------------------------------------- optimizer / profile
class Mock:
    """Closed-form forced response on a vertical hinge: q = q0 + v0 t + 0.5 (Q - Td - c v0)/I t^2 (short horizon)."""
    I0 = 0.5

    def __init__(self):
        self.n = 0

    def effective_inertia(self, p):
        return p.rho_scale * self.I0 + p.armature

    def __call__(self, p, e):
        self.n += 1
        I = self.effective_inertia(p)
        t = e.t_s - e.t_s[0]
        Q = e.effort_at(e.t_s) if e.mode == "forced" else 0.0 * t
        return SimulationResult(q=e.q0 + e.v0 * t + 0.5 * (Q - p.dynamic_effort - p.viscous * e.v0) / I * t ** 2,
                                runtime_semantics="EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE")


B = dict(rho_scale=(0.2, 5.0), armature=(0.0, 0.0), dynamic_effort=(0.0, 2.0), static_excess=(0.0, 0.0),
         viscous=(0.0, 0.0))
PHYS = dict(engine="sapien_cpu_diagnostic", joint_type="revolute", friction_mode="external_effort_surrogate")


def forced(name, Qlev, sim, truth, train=True, noise=1e-3, seed=0):
    t = np.linspace(0, 1, 26)
    proto = Episode(name, t, np.zeros_like(t), np.full_like(t, noise), 0.0, 0.1, np.array([0.0]), np.array([Qlev]),
                    "forced", "VERIFIED", used_for_training=train)
    q = sim(truth, proto).q + np.random.default_rng(seed).normal(0, noise, len(t))
    return Episode(name, t, q, np.full_like(t, noise), 0.0, 0.1, np.array([0.0]), np.array([Qlev]), "forced",
                   "VERIFIED", used_for_training=train)


def test_two_force_levels_bound_the_scale_one_level_does_not():
    sim = Mock()
    truth = PhysicalParameters(1.5, 0.0, 0.4, 0.4, 0.0)
    eps2 = [forced("a", 1.0, sim, truth), forced("b", 1.8, sim, truth, seed=1), forced("h", 1.4, sim, truth,
                                                                                         train=False, seed=2)]
    rep = estimate(sim, eps2, physics=PHYS, loss={}, fitting=dict(bounds=B, maxiter=30, popsize=8),
                   profile=dict(n=7, inner_maxiter=10, inner_popsize=6))
    assert rep["params"]["rho_scale"] == pytest.approx(1.5, rel=0.05)
    pr = rep["scale_profile"]
    assert pr["status"] == "SCALE_PROFILE_BOUNDED_CANDIDATE" and pr["ci95"][0] <= 1.5 <= pr["ci95"][1]
    assert any(p.get("source") == "global_fit" for p in pr["points"])
    assert rep["heldout"]["status"] == "EVALUATED_WITH_FROZEN_THETA" and rep["model_adequacy"]["heldout_adequate"]
    assert rep["status_table"]["absolute_scale"] == "PROFILE_BOUNDED_CANDIDATE_NOT_CERTIFIED"
    assert rep["status_table"]["overall"] == "DIAGNOSTIC_ONLY_NOT_IDENTIFIED"
    # ONE force level: (Q - Td)/I is all the data see -> rho and Td trade off, the profile stays open
    rep1 = estimate(sim, [forced("a", 1.0, sim, truth)], physics=PHYS, loss={},
                    fitting=dict(bounds=B, maxiter=20, popsize=6), profile=dict(n=7, inner_maxiter=10, inner_popsize=6))
    assert rep1["scale_profile"]["status"] == "SCALE_PROFILE_OPEN"
    assert rep1["status_table"]["absolute_scale"] == "NOT_IDENTIFIED"
    assert rep1["downstream"]["point_estimate_allowed"] is False


def test_optimizer_fixed_dims_failures_and_warm_start():
    sim = Mock()
    truth = PhysicalParameters(1.0, 0.0, 0.2, 0.2, 0.0)
    obj = RefinementLoss(sim, [forced("a", 1.0, sim, truth)])
    fit = fit_physx(obj, FitConfiguration(bounds=dict(B, rho_scale=(1.0, 1.0)), maxiter=5, popsize=5),
                    x0=dict(rho_scale=1.0, armature=0.0, dynamic_effort=0.2, static_excess=0.0, viscous=0.0))
    assert fit["optimizer"]["free_parameters"] == ["dynamic_effort"]
    assert fit["params"]["dynamic_effort"] == pytest.approx(0.2, abs=0.05)

    def broken(p, e):
        raise RuntimeError("engine down")
    with pytest.raises(RuntimeError, match="evaluations failed"):
        fit_physx(RefinementLoss(broken, obj.episodes), FitConfiguration(bounds=B, maxiter=2, popsize=4, polish=False))


def test_inadequate_model_is_labelled_and_refused():
    sim = Mock()
    t = np.linspace(0, 1, 26)
    e = Episode("osc", t, 0.2 * np.sin(12 * t), np.full_like(t, 1e-3), 0.0, 2.4, np.array([0.0]), np.array([0.5]),
                "forced", "VERIFIED")
    rep = estimate(sim, [e], physics=PHYS, loss={}, fitting=dict(bounds=B, maxiter=5, popsize=5),
                   profile=dict(n=0))
    assert rep["model_adequacy"]["train_adequate"] is False
    assert rep["status_table"]["overall"] == "MODEL_INADEQUATE"
    assert all(v["status"] in ("MODEL_INADEQUATE", "FIXED_PRIOR", "NOT_MODELLED_BY_SURROGATE")
               for v in rep["status_table"]["parameters"].values())
    assert ID.downstream_point_allowed(rep)[0] is False


def test_native_isaac_adapter_is_not_faked():
    with pytest.raises(NotImplementedError, match="not implemented"):
        PhysXIsaacRollout()
    with pytest.raises(ValueError, match="native Isaac"):
        estimate(Mock(), [], physics=dict(PHYS, engine="isaac_native"), loss={},
                 fitting=dict(bounds=B))


def test_scale_prior_keeps_scale_prior_only():
    sim = Mock()
    truth = PhysicalParameters(1.5, 0.0, 0.4, 0.4, 0.0)
    eps = [forced("a", 1.0, sim, truth), forced("b", 1.8, sim, truth, seed=1)]
    rep = estimate(sim, eps, physics=PHYS, loss=dict(scale_prior_weight=0.5, prior_log_rho=0.0),
                   fitting=dict(bounds=B, maxiter=20, popsize=6), profile=dict(n=5, inner_maxiter=6, inner_popsize=5))
    assert rep["status_table"]["absolute_scale"] == "PRIOR_ONLY"
    assert rep["status_table"]["parameters"]["rho_scale"]["status"] == "PRIOR_ONLY"
