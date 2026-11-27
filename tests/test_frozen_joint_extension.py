"""Frozen-joint q(t) extension: the accepted joint is kept exactly; q is measured on frames outside the clip."""
import numpy as np
import pytest

from estimotion_2part.joint_discovery.frozen_joint_extension import extend_authority, write_extension
from estimotion_2part.joint_discovery.integration_contract import CameraEntry, JointAuthority
from estimotion_2part.joint_discovery.observations import FrameMap, TrackObservations
from estimotion_2part.joint_discovery.rigid_kinematics import A_joint


def _q_true(T, jt):
    t = np.arange(T)
    lo, hi = (0.1, 0.9) if jt == "revolute" else (0.02, 0.32)
    s = np.clip((t - 20) / 40.0, 0, 1)
    return lo + (hi - lo) * (3 * s ** 2 - 2 * s ** 3)                    # rest 0-20, motion 20-60, rest 60-79


def _scene(jt="revolute", axis=(0.0, 0.0, 1.0), noise=0.002, T=80, clip=(25, 50), seed=0):
    rng = np.random.default_rng(seed)
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    o = np.array([0.3, 0.0, 0.0]) if jt == "revolute" else np.zeros(3)
    q = _q_true(T, jt)
    ref = clip[0]
    qa = q - q[ref]                                                        # authority gauge: q(ref) = 0
    Xc = o + np.c_[rng.uniform(0.25, 0.6, 60), rng.uniform(-0.1, 0.1, 60), rng.uniform(-0.3, 0.3, 60)]
    Xs = rng.uniform(-1, 1, (40, 3)) + np.array([0, 1.5, 0])
    A = np.asarray(A_joint(jt, u, o, qa), float)
    trk, tim, xw = [], [], []
    for i, X in enumerate(Xc):
        x = np.einsum("tij,j->ti", A[:, :3, :3], X) + A[:, :3, 3]
        trk += [i] * T
        tim += list(range(T))
        xw.append(x + rng.standard_normal((T, 3)) * noise)
    for j, X in enumerate(Xs):
        trk += [60 + j] * T
        tim += list(range(T))
        xw.append(np.tile(X, (T, 1)) + rng.standard_normal((T, 3)) * noise)
    xw = np.concatenate(xw)
    M = len(trk)
    rgbd = np.arange(T)
    fm = FrameMap(rgbd, 2 * rgbd, 2 * rgbd / 50.0, provenance="synthetic")
    obs = TrackObservations(["cam_001"], fm, cam=np.zeros(M, int), track=np.array(trk), time=np.array(tim),
                            uv=np.zeros((M, 2)), uv_conf=np.ones(M), uv_valid=np.ones(M, bool), z=np.ones(M),
                            depth_valid=np.ones(M, bool), depth_edge=np.zeros(M, bool), mask_valid=np.ones(M, bool),
                            x_w=xw, track_camera=np.zeros(100, int), track_local_id=np.arange(100),
                            track_query_time=np.zeros(100, int))
    a, b = clip
    cams = {"cam_001": CameraEntry("cam_001", np.eye(3), np.eye(4), 640, 480, "synthetic")}
    auth = JointAuthority(joint_type=jt, axis_world=u, origin_world=o if jt == "revolute" else None,
                          q=qa[a:b + 1].copy(), q_valid=np.ones(b - a + 1, bool), reference_time_index=0,
                          rgbd_frame_index=rgbd[a:b + 1], source_frame_index=2 * rgbd[a:b + 1],
                          timestamp_s=2 * rgbd[a:b + 1] / 50.0, cameras=cams,
                          q_sigma=np.full(b - a + 1, 0.002 if jt == "revolute" else 0.001))
    auth.validate()
    return auth, obs, qa


@pytest.mark.parametrize("jt", ["revolute", "prismatic"])
def test_extension_recovers_q_outside_the_clip_with_the_joint_frozen(jt, tmp_path):
    auth, obs, qa = _scene(jt)
    aid = auth.authority_id()
    r = extend_authority(auth, obs)
    assert r["status"] == "EXTENDED", r["report"].get("reason")
    new = r["authority"]
    assert new.parent_authority_id == aid and auth.authority_id() == aid          # parent untouched
    np.testing.assert_array_equal(new.axis_world, auth.axis_world)
    assert (new.origin_world is None) == (auth.origin_world is None)
    if new.origin_world is not None:
        np.testing.assert_array_equal(new.origin_world, auth.origin_world)
    assert len(new.q) == 80 and new.q[new.reference_time_index] == 0.0
    assert int(new.source_frame_index[new.reference_time_index]) == int(auth.source_frame_index[0])
    np.testing.assert_array_equal(new.q[25:51], auth.q)                          # parent values unchanged
    out = np.r_[0:25, 51:80]
    tol = 0.005 if jt == "revolute" else 0.002
    assert np.max(np.abs(new.q[out] - qa[out])) < tol                            # rest plateaus now covered
    v = r["report"]["validation"]
    assert v["rms_difference"] <= v["tolerance"] and abs(v["gauge_offset"]) < tol
    rep = write_extension(r, tmp_path / "ext")
    assert JointAuthority.load(tmp_path / "ext").authority_id() == rep["authority_id"]


def test_a_different_joint_is_refused_not_absorbed():
    auth, obs, _ = _scene("revolute")
    wrong = JointAuthority(**{**auth.__dict__, "axis_world": np.array([0.0, np.sin(0.4), np.cos(0.4)])})
    r = extend_authority(wrong, obs)
    assert r["status"] in ("INCONSISTENT_WITH_AUTHORITY", "TOO_FEW_MOVING_TRACKS") and r["authority"] is None
