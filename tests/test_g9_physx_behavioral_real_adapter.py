"""Wrench-driven path on the REAL records (no engine; skipped when the results SSD is not mounted): T01 the known
transform / sign / wrench frame are parsed from the existing files, T02 the archive wrench is used exactly as recorded
(no extra tare), T15 native wrench rows + every valid visual sample with q0 = visual q(t0) and v0 = 0, T19 the fridge
pulse rows 229-232 Q_full vs legacy TCP-force-only comparison with real values (no sign correction)."""
import glob
from pathlib import Path

import numpy as np
import pytest

RUNS = Path("/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware")
REAL = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real")
DISC = Path("/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint")
SC = "final_red_fridge_episode_000000"
pytestmark = pytest.mark.skipif(not (DISC / SC / "robot_archive.npz").is_file(), reason="real records not mounted")


@pytest.fixture(scope="module")
def real():
    from forceartgs_g9.adapters.forceartgs_g9 import _q_noise, accepted_kinematics
    from forceartgs_g9.physx_refinement import behavioral_adapter as BA
    from forceartgs_g9.physx_refinement import behavioral_report as BR
    from forceartgs_g9.physx_refinement.behavioral_contracts import load_wrench_config
    from forceartgs_g9.wrench_input import full_record as FR
    run = Path(sorted(glob.glob(str(RUNS / f"{SC}_multiview_*_joint_aware_full")))[0])
    kin = accepted_kinematics(run, True)
    auth = kin["authority"]
    m = np.asarray(auth.q_valid, bool)
    qs = auth.q_sigma[m] if auth.q_sigma is not None else _q_noise(auth, m)
    qs = np.where(np.isfinite(qs) & (qs > 0), qs, np.nanmedian(np.where(qs > 0, qs, np.nan)))
    exp = RUNS.parent / "contact_annotations" / SC / "contact_annotations.json"
    c = FR.load_contract(REAL / SC, DISC / SC / "robot_archive.npz", exp)
    fr = FR.build(c, scene=SC, joint_type=auth.joint_type, axis=auth.axis_world, origin=auth.origin_world,
                  q_src=np.asarray(auth.source_frame_index)[m], q=np.asarray(auth.q, float)[m], q_sigma=qs)
    cfg = load_wrench_config()
    segs = BA.load_interaction_segments(Path(__file__).resolve().parents[1] / "configs" / "contact_intervals" / f"{SC}.yaml")
    return dict(c=c, fr=fr, auth=auth, segs=segs, out=BA.build_wrench_episode(fr, cfg, segs),
                pulse=BR.pulse_comparison(fr, (229, 232)))


def test_contract_parsed_from_existing_files(real):
    c = real["c"]
    assert c["wrench_frame"] == "sensor axes = EE axes" and c["R_ee_force_sensor"] == np.eye(3).tolist()
    assert c["object_side_multiplier"] == -1.0 and c["wrench_order"] == ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    assert c["moment_reference"].startswith("ee_frame_origin")


def test_archive_wrench_used_exactly_as_recorded(real):
    z = np.load(DISC / SC / "robot_archive.npz")
    fr = real["fr"]
    np.testing.assert_array_equal(fr.W_recorded, z["wrench"])
    R = z["T_base_ee"][:, :3, :3]
    F = -np.einsum("nij,nj->ni", R, z["wrench"][:, :3])
    T = -np.einsum("nij,nj->ni", R, z["wrench"][:, 3:])
    a = np.asarray(real["auth"].axis_world, float)
    a = a / np.linalg.norm(a)
    Q = (T + np.cross(z["T_base_ee"][:, :3, 3] - np.asarray(real["auth"].origin_world, float), F)) @ a
    np.testing.assert_allclose(fr.Q_full, Q, atol=1e-12)


def test_interaction_episode_from_the_existing_segments(real):
    out, fr = real["out"], real["fr"]
    assert out["status"] == "OK", out["reason"]
    assert real["segs"] == [(220, 296)]                                             # force-peak + visual segment
    ep = out["episode"]
    assert ep.source_start_frame == 220 and ep.v0_fixed == 0.0
    assert ep.q0_observed == fr.q[np.flatnonzero(fr.source_q_frame == 220)[0]]
    assert out["evidence"]["input_coverage"]["status"] == "COMPLETE"
    assert np.all(np.diff(ep.effort_source_row) == 1) and ep.contact_evidence["mode"] == "ON_DURING_INTERACTION"
    inside = (ep.effort_source_row >= 220) & (ep.effort_source_row <= 296)
    assert np.all(ep.effort_object[~inside] == 0.0)


def test_fridge_pulse_full_wrench_vs_force_only(real):
    dp = real["pulse"]["declared_pulse"]
    assert dp["rows"] == [229, 232] and len(dp["Q_full"]) == 4
    assert np.isfinite(dp["impulse_full"]) and np.isfinite(dp["impulse_moment_term"])
    assert "no sign correction" in real["pulse"]["note"]
