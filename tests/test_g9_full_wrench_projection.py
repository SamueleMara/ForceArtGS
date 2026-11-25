"""Full six-axis wrench input (no engine): T01 known transform loaded from the contract, never estimated; T02 the
tared wrench is used as recorded (no extra baseline); T03 no double transform (sensor vs pre-transformed base frame
give identical Q); T04 revolute transport with a sensor moment and an offset sensor origin; T05 prismatic ignores the
moment; T06 independent of any contact-point annotation; T07 reference-point invariance and axis / q reversal
preserve virtual power; T08 pure moment -> nonzero revolute Q (legacy force-only gives 0); T09 object-side sign applied
once, never a parameter; T18 a non-finite wrench row is a coverage error, never a zero."""
import inspect
import json

import numpy as np
import pytest

from forceartgs_g9.wrench_input import full_record as FR
from forceartgs_g9.wrench_input.full_wrench_projection import (SpatialWrench, WrenchContractError, effort_sigma,
                                                               joint_effort, sensor_to_base, transport)

RNG = np.random.default_rng(0)


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rotx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def scene(tmp_path, n=60, *, W=None, sign=-1, with_export=True):
    """A minimal real-format scene: metadata/robot_tool.json (wrench_frame), contact_force_convention.json, archive."""
    meta = tmp_path / "scene" / "metadata"
    meta.mkdir(parents=True)
    (meta / "robot_tool.json").write_text(json.dumps(dict(
        recorded_pose_frame="ee", T_ee_tcp=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.19], [0, 0, 0, 1]],
        wrench_frame=dict(axes="ee", R_ee_force_sensor=np.eye(3).tolist(), alignment_source="test",
                          torque_origin="recorded_origin_unspecified", tcp_torque_transport_applied=False))))
    conv = dict(object_side_force_multiplier=sign)
    if sign == -1:                                    # the existing validator checks multiplier vs recorded quantity
        conv["recorded_quantity"] = "contact_reaction_force_on_robot"
    (meta / "contact_force_convention.json").write_text(json.dumps(conv))
    T_ee = np.repeat(np.eye(4)[None], n, 0)
    for i in range(n):
        T_ee[i, :3, :3] = rotz(0.3 + 0.01 * i) @ rotx(0.2)
        T_ee[i, :3, 3] = [0.4 + 0.002 * i, 0.1, 0.5]
    T_tcp = T_ee.copy()
    T_tcp[:, :3, 3] = T_ee[:, :3, 3] + np.einsum("nij,j->ni", T_ee[:, :3, :3], [0, 0, 0.19])
    W = RNG.normal(size=(n, 6)) if W is None else W
    arc = tmp_path / "robot_archive.npz"
    np.savez(arc, wrench=W, T_base_ee=T_ee, T_base_tcp=T_tcp, source_row=np.arange(n))
    exp = None
    if with_export:
        exp = tmp_path / "contact_annotations.json"
        exp.write_text(json.dumps(dict(thresholds=dict(tare_N=[5.0, 5.0, 5.0], sigma_N=[0.05, 0.05, 0.05],
                                                       baseline_duration_s=0.5), tare="baseline mean removed")))
    return tmp_path / "scene", arc, exp, W, T_ee, T_tcp


AX, O = np.array([0.0, 0.6, 0.8]), np.array([0.2, -0.1, 0.3])


def build(tmp_path, **kw):
    sc, arc, exp, W, T_ee, T_tcp = scene(tmp_path, **kw)
    c = FR.load_contract(sc, arc, exp)
    q_src = np.arange(0, len(W), 4)
    fr = FR.build(c, scene="t", joint_type="revolute", axis=AX, origin=O, q_src=q_src, q=np.zeros(len(q_src)),
                  q_sigma=np.full(len(q_src), 1e-3))
    return c, fr, W, T_ee, T_tcp


# -------------------------------------------------------------------------------------------------- T01 / T02
def test_known_transform_is_loaded_never_estimated(tmp_path):
    c, fr, W, T_ee, _ = build(tmp_path)
    assert c["R_ee_force_sensor"] == np.eye(3).tolist() and "E1" in c["known_transform"]
    np.testing.assert_allclose(fr.R_BS, T_ee[:, :3, :3])                       # R_BS = R_base_ee R_ee_sensor
    for fn in (FR.load_contract, FR.build, sensor_to_base, joint_effort):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"recalibrate_sensor", "estimate_tare", "detect_contact", "infer_contact_point",
                             "contact_point", "tare"}


def test_tared_wrench_used_as_recorded_without_extra_baseline(tmp_path):
    c, fr, W, T_ee, _ = build(tmp_path)                                         # export says tare_N = 5 N per axis
    np.testing.assert_array_equal(fr.W_recorded, W)
    F_expected = -np.einsum("nij,nj->ni", T_ee[:, :3, :3], W[:, :3])            # NOT W - 5 N
    np.testing.assert_allclose(fr.F_B, F_expected, atol=1e-12)
    assert "no additional tare" in c["tare"]
    assert "NOT applied" in c["force_export"]["use"]


# -------------------------------------------------------------------------------------------------- T03
def test_no_double_transform_and_frame_equivalence(tmp_path):
    c, fr, W, T_ee, _ = build(tmp_path)
    R, p = T_ee[:, :3, :3], T_ee[:, :3, 3]
    ws = SpatialWrench.from_archive(W)
    wb = sensor_to_base(ws, R, p, -1.0)
    with pytest.raises(WrenchContractError):
        sensor_to_base(wb, R, p, -1.0)                                            # already base frame
    pre = SpatialWrench(wb.F, wb.tau, p, "base", "object")                       # an equivalent pre-transformed record
    np.testing.assert_allclose(joint_effort(pre, "revolute", AX, O), fr.Q_full, atol=1e-12)
    np.testing.assert_allclose(joint_effort(pre, "prismatic", AX), wb.F @ (AX / np.linalg.norm(AX)), atol=1e-12)


# -------------------------------------------------------------------------------------------------- T04 / T06
def test_revolute_transport_with_sensor_moment_and_offset_origin():
    n = 20
    R = np.repeat(rotz(0.4)[None], n, 0)
    p_S = np.array([0.5, 0.2, 0.6])
    F_S = RNG.normal(size=(n, 3))
    tau_S = RNG.normal(size=(n, 3)) * 0.3
    # brute force: base-frame object-side wrench, moment about the hinge origin computed directly
    F_B, tau_B = -(F_S @ R[0].T), -(tau_S @ R[0].T)
    tau_o = tau_B + np.cross(p_S - O, F_B)
    Q_ref = tau_o @ (AX / np.linalg.norm(AX))
    w = sensor_to_base(SpatialWrench.from_archive(np.c_[F_S, tau_S]), R, p_S, -1.0)
    np.testing.assert_allclose(joint_effort(w, "revolute", AX, O), Q_ref, atol=1e-12)
    # independent of any contact point: the TCP pose is never an input to Q_full
    assert "contact" not in str(inspect.signature(joint_effort)) and "tcp" not in str(inspect.signature(joint_effort))


def test_q_full_does_not_depend_on_the_tcp_annotation(tmp_path):
    c1, fr1, W, *_ = build(tmp_path / "a")
    sc, arc, exp, *_ = scene(tmp_path / "b", W=W)
    z = dict(np.load(arc))
    z["T_base_tcp"][:, :3, 3] += 0.3                                               # move the declared TCP
    np.savez(arc, **z)
    c2 = FR.load_contract(sc, arc, exp)
    fr2 = FR.build(c2, scene="t", joint_type="revolute", axis=AX, origin=O, q_src=np.arange(0, 60, 4),
                   q=np.zeros(15), q_sigma=np.full(15, 1e-3))
    np.testing.assert_allclose(fr1.Q_full, fr2.Q_full, atol=1e-12)


# -------------------------------------------------------------------------------------------------- T05
def test_prismatic_ignores_the_moment():
    n = 10
    R = np.repeat(np.eye(3)[None], n, 0)
    F = RNG.normal(size=(n, 3))
    a = sensor_to_base(SpatialWrench.from_archive(np.c_[F, np.zeros((n, 3))]), R, np.zeros(3), -1.0)
    b = sensor_to_base(SpatialWrench.from_archive(np.c_[F, 5 * RNG.normal(size=(n, 3))]), R, np.zeros(3), -1.0)
    np.testing.assert_allclose(joint_effort(a, "prismatic", AX), joint_effort(b, "prismatic", AX), atol=1e-12)


# -------------------------------------------------------------------------------------------------- T07
def test_reference_point_invariance_and_reversal_preserves_virtual_power():
    n = 15
    w = sensor_to_base(SpatialWrench.from_archive(RNG.normal(size=(n, 6))), np.repeat(rotz(1.1)[None], n, 0),
                       np.array([0.3, 0.1, 0.2]), -1.0)
    Q = joint_effort(w, "revolute", AX, O)
    for new in (np.array([1.0, 2.0, -3.0]), O):
        np.testing.assert_allclose(joint_effort(transport(w, new), "revolute", AX, O), Q, atol=1e-12)
    qdot = RNG.normal(size=n)
    Q_rev = joint_effort(w, "revolute", -AX, O)
    np.testing.assert_allclose(Q_rev, -Q, atol=1e-12)
    np.testing.assert_allclose(Q_rev * (-qdot), Q * qdot, atol=1e-12)               # Q qdot invariant


# -------------------------------------------------------------------------------------------------- T08
def test_pure_moment_gives_nonzero_revolute_effort_and_legacy_misses_it():
    n = 5
    tau = np.tile(AX / np.linalg.norm(AX) * 0.7, (n, 1))                        # pure moment, zero net force
    w = sensor_to_base(SpatialWrench.from_archive(np.c_[np.zeros((n, 3)), tau]), np.repeat(np.eye(3)[None], n, 0),
                       np.array([0.4, 0.0, 0.1]), -1.0)
    Q = joint_effort(w, "revolute", AX, O)
    np.testing.assert_allclose(Q, -0.7, atol=1e-12)
    legacy = np.einsum("j,nj->n", AX / np.linalg.norm(AX), np.cross(np.array([0.6, 0, 0.1]) - O, np.zeros((n, 3))))
    assert np.all(legacy == 0) and np.all(np.abs(Q) > 0.5)


# -------------------------------------------------------------------------------------------------- T09
def test_object_side_sign_applied_once_and_read_from_metadata(tmp_path):
    c, fr, W, T_ee, _ = build(tmp_path)
    assert c["object_side_multiplier"] == -1.0
    w = sensor_to_base(SpatialWrench.from_archive(W), T_ee[:, :3, :3], T_ee[:, :3, 3], -1.0)
    with pytest.raises(WrenchContractError):
        sensor_to_base(SpatialWrench(w.F, w.tau, None, "sensor", "object"), T_ee[:, :3, :3], T_ee[:, :3, 3], -1.0)
    with pytest.raises(WrenchContractError):
        sensor_to_base(SpatialWrench.from_archive(W), T_ee[:, :3, :3], T_ee[:, :3, 3], 0.5)
    c2, fr2, *_ = build(tmp_path / "pos", sign=1, W=W)
    np.testing.assert_allclose(fr2.Q_full, -fr.Q_full, atol=1e-12)               # the metadata sign, nothing else
    from forceartgs_g9.physx_refinement.behavioral_contracts import BehavioralContractError, load_wrench_config
    with pytest.raises(BehavioralContractError):
        load_wrench_config({"input": {"force_sign": -1}})                        # never a tunable option


def test_effort_sigma_matches_monte_carlo():
    n = 4
    R = np.repeat(rotz(0.7)[None], n, 0)
    ref = np.tile([0.5, 0.3, 0.2], (n, 1))
    sF, sT = np.array([0.05, 0.08, 0.03]), np.array([0.002, 0.004, 0.001])
    sd = effort_sigma(R, ref, sF, sT, "revolute", AX, O)
    W0 = np.zeros((n, 6))
    draws = []
    for _ in range(20000):
        W = W0 + np.c_[RNG.normal(size=(n, 3)) * sF, RNG.normal(size=(n, 3)) * sT]
        draws.append(joint_effort(sensor_to_base(SpatialWrench.from_archive(W), R, ref, -1.0), "revolute", AX, O))
    np.testing.assert_allclose(np.std(draws, axis=0), sd, rtol=0.03)


# -------------------------------------------------------------------------------------------------- T18
def test_nonfinite_wrench_row_is_a_coverage_error_not_zero(tmp_path):
    from forceartgs_g9.physx_refinement import behavioral_adapter as BA
    from forceartgs_g9.physx_refinement.behavioral_contracts import load_wrench_config
    W = RNG.normal(size=(60, 6)) * 0.1
    W[30, 4] = np.nan
    c, fr, *_ = build(tmp_path, W=W)
    out = BA.build_wrench_episode(fr, load_wrench_config(), [(0, 59)])
    assert out["status"] == "INPUT_COVERAGE_ERROR" and out["episode"] is None
    assert any(m["kind"] == "NONFINITE_EFFORT_ROW" for m in out["evidence"]["input_coverage"]["missing"])
    # ordinary zero rows are valid measurements
    W2 = np.zeros((60, 6))
    c2, fr2, *_ = build(tmp_path / "z", W=W2)
    assert BA.build_wrench_episode(fr2, load_wrench_config(), [(0, 59)])["status"] == "OK"


def test_contract_names_the_missing_file(tmp_path):
    sc, arc, exp, *_ = scene(tmp_path)
    (sc / "metadata" / "robot_tool.json").unlink()
    with pytest.raises(WrenchContractError, match="robot_tool.json"):
        FR.load_contract(sc, arc, exp)
    with pytest.raises(WrenchContractError, match="robot archive missing"):
        FR.load_contract(sc, tmp_path / "nope.npz", exp)
