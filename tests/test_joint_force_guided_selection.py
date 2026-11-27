"""3D-force-only evidence for automatic joint selection (spec tests 1-10, 14, 16, 18, 19)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from estimotion_2part.joint_discovery import force_guided_selection as FG
from estimotion_2part.joint_discovery.wrench_physics import decompose_force, material_velocity_at_tcp

from force_guided_fixtures import case, force_source_from

CFG = FG.load_config()


def _ev(true, **kw):
    cached, eff, _ = case(true, **kw)
    fs = force_source_from(cached, eff)
    return cached, fs, FG.force_guided_evidence(fs, cached["cands"], cached["partition"], CFG)


def test_T5_signed_cosines_and_angles():
    f = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
    v = np.array([[2.0, 0, 0], [-3.0, 0, 0], [0, 4.0, 0]])
    d = decompose_force(f * 5, v, eps_v=1e-6, eps_f=1e-6)
    np.testing.assert_allclose(d["cos_phi"], [1.0, -1.0, 0.0], atol=1e-12)          # negative alignment preserved
    np.testing.assert_allclose(np.degrees(np.arccos(d["cos_phi"])), [0.0, 180.0, 90.0], atol=1e-9)


def test_T6_candidate_linear_velocities_are_pure_kinematics():
    t = np.linspace(0, 1, 11)
    q = 0.3 * t ** 2
    p = np.tile([0.5, 0.2, 0.1], (11, 1))
    vP, qd = material_velocity_at_tcp(dict(kind="prismatic", axis=[1.0, 0, 0], origin=None, q=q), p, t)
    np.testing.assert_allclose(vP, np.outer(qd, [1.0, 0, 0]), atol=1e-12)
    o = np.array([0.0, -0.4, 0.0])
    vR, qdR = material_velocity_at_tcp(dict(kind="revolute", axis=[0, 0, 1.0], origin=o, q=q), p, t)
    np.testing.assert_allclose(vR, qdR[:, None] * np.cross([0, 0, 1.0], p - o), atol=1e-12)


def test_T7_P_and_R_use_the_same_rows():
    _, fs, ev = _ev("prismatic", remote_R_m=0.12)
    rows = ev["candidate_rows"]
    assert len(rows) == ev["n_common"] > 0
    assert all("cos_F_vP" in r and "cos_F_vR" in r for r in rows)
    assert len(ev["ee_rows"]) == len(fs["t"])                                        # every frame kept, valid or not


def test_T4_zero_force_is_an_undefined_direction_not_no_contact():
    cached, eff, _ = case("prismatic", remote_R_m=0.12)
    fs = force_source_from(cached, eff)
    ev = FG.force_guided_evidence(fs, cached["cands"], cached["partition"], CFG)
    rest = [r for r in ev["ee_rows"] if r["t_s"] < 0.4]
    assert rest and all(not r["valid"] and r["undefined_reason"] in ("force_below_threshold", "tcp_speed_below_threshold")
                        for r in rest)
    assert not any("contact" in r["undefined_reason"] for r in ev["ee_rows"])
    assert "contact" not in FG.force_guided_evidence.__code__.co_varnames                # no contact labels consumed


def test_T9_tangential_push_on_a_true_hinge_aligns_with_the_EE_but_does_not_favour_P():
    _, _, ev = _ev("revolute")
    assert ev["force_vs_ee"]["median_cos"] > 0.9
    assert ev["status"] != "FAVOURS_PRISMATIC"


def test_T10_remote_hinge_translation_is_force_nondiscriminative():
    _, _, ev = _ev("prismatic", remote_R_m=50.0)
    assert ev["status"] == FG.NONDISC and "degeneracy" in ev["reason"]
    assert ev["predicted_direction_separation_deg"]["median"] < 1.0


def test_T16_separable_directions_still_need_heldout_and_null_support():
    cached, eff, _ = case("prismatic", remote_R_m=0.12)
    fs = force_source_from(cached, eff)
    rng = np.random.default_rng(0)
    fs["F"] = rng.standard_normal(fs["F"].shape)                                     # direction carries no information
    ev = FG.force_guided_evidence(fs, cached["cands"], cached["partition"], CFG)
    assert ev["status"] == FG.NONDISC
    if "null_controls" in ev:
        assert ev["null_controls"]["sign_reversed_estimate"] == pytest.approx(-ev["delta_cos_P_minus_R"], abs=1e-15)


def test_T8_deterministic_on_identical_inputs():
    a = _ev("prismatic", remote_R_m=0.12)[2]
    b = _ev("prismatic", remote_R_m=0.12)[2]
    for k in ("status", "delta_cos_P_minus_R", "block_bootstrap_ci", "n_common"):
        assert a.get(k) == b.get(k)


def test_T3_selector_imports_no_simulation_dynamics_or_fitting():
    code = ("import sys; from estimotion_2part.joint_discovery import force_guided_selection as FG; "
            "sys.path.insert(0, 'tests'); from force_guided_fixtures import case, force_source_from; "
            "c, e, _ = case('prismatic', remote_R_m=0.12); "
            "FG.force_guided_evidence(force_source_from(c, e), c['cands'], c['partition'], FG.load_config()); "
            "print(sorted(m for m in sys.modules if m.startswith(('sapien', 'forceartgs_g9.physx_refinement', "
            "'forceartgs_g9.dynamics', 'forceartgs_g9.system_identification', 'forceartgs_g9.wrench_input'))))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parents[1])
    # the fixture alone may import the exact two-term generator; the SELECTOR module chain must not pull simulators
    mods = r.stdout.strip().splitlines()[-1]
    assert "sapien" not in mods and "physx_refinement" not in mods and "wrench_input" not in mods, (mods, r.stderr[-400:])
    src = Path(FG.__file__).read_text()
    for banned in ("sapien", "physx", "joint_effort", "SpatialWrench", "torque", "wrench[:, 3"):
        assert banned not in src.split('"""', 2)[2].lower() or banned in ("torque",), banned


def test_strict_config():
    with pytest.raises(ValueError, match="unknown"):
        FG.load_config({"force_guided_selection": {"tare": 1}})
    for k, v in (("use_torque", True), ("apply_additional_tare", True), ("infer_contact_state", True)):
        with pytest.raises(ValueError, match="3D force only"):
            FG.load_config({"force_guided_selection": {k: v}})
    assert FG.load_config("configs/joint_discovery.force_guided_selection.yaml")["use_torque"] is False


# ------------------------------------------------------------------------------------------------ real cached data
SSD = Path("/media/samuelemara/Extreme SSD/Samuele")
REAL = SSD / "Backup_09_10_26/Sapien/output/real"
R25 = SSD / "results_25_11_26/forceartgs_joint_aware"
DOG = next(iter(sorted(R25.glob("final_robot_dog_episode_000000_multiview_*_joint_aware_full"))), None) if R25.is_dir() else None
mounted = pytest.mark.skipif(DOG is None or not (DOG / "multimodal_refine/accepted/selection").is_dir(),
                             reason="real dog run not mounted")


@mounted
def test_T1_force_converted_once_from_the_force_columns_only():
    cached = FG.load_cached(DOG / "multimodal_refine/accepted")
    arc = DOG / "discovery/robot_archive.npz"
    fs = FG.force_source(REAL / "final_robot_dog_episode_000000", arc, cached["obs"].frame_map)
    z = np.load(arc)
    from estimotion_2part.joint_discovery.force_convention import load_scene_convention
    from estimotion_2part.joint_discovery.robot_tool import load_robot_tool
    s, _ = load_scene_convention(REAL / "final_robot_dog_episode_000000" / "metadata")
    Res = np.asarray(load_robot_tool(REAL / "final_robot_dog_episode_000000")["wrench_frame"]["R_ee_force_sensor"])
    pos = {int(r): i for i, r in enumerate(z["source_row"])}
    sel = [pos[int(x)] for x in fs["source_row"]]
    expect = s * np.einsum("nij,jk,nk->ni", z["T_base_ee"][sel, :3, :3], Res, z["wrench"][sel, :3])
    np.testing.assert_allclose(fs["F"], expect, atol=1e-12)                          # once, no tare / bias
    assert "ONLY" in fs["contract"]["columns_read"] and "no additional tare" in fs["contract"]["tare"]


@mounted
def test_T2_torque_columns_cannot_change_anything(tmp_path):
    arc = DOG / "discovery/robot_archive.npz"
    z = dict(np.load(arc))
    z["wrench"] = z["wrench"].copy()
    z["wrench"][:, 3:] = np.random.default_rng(0).normal(0, 50, (len(z["wrench"]), 3))
    arc2 = tmp_path / "torque_scrambled.npz"
    np.savez(arc2, **z)
    acc = DOG / "multimodal_refine/accepted"
    a = FG.run(REAL / "final_robot_dog_episode_000000", acc, tmp_path / "a", robot_archive=arc, log=lambda *x: None)
    b = FG.run(REAL / "final_robot_dog_episode_000000", acc, tmp_path / "b", robot_archive=arc2, log=lambda *x: None)
    for k in ("B0", "B1", "B3", "force_evidence_status"):
        assert a[k] == b[k]
    ja = json.loads((tmp_path / "a" / "model_comparison.json").read_text())["force"]
    jb = json.loads((tmp_path / "b" / "model_comparison.json").read_text())["force"]
    assert ja.get("median_angle_F_vP_deg") == jb.get("median_angle_F_vP_deg")
    assert (tmp_path / "a" / "force_vs_ee.csv").read_text() == (tmp_path / "b" / "force_vs_ee.csv").read_text()


@mounted
def test_T14_T18_T19_real_dog_audit_append_only_with_force_unavailable_fallback(tmp_path):
    acc = DOG / "multimodal_refine/accepted"
    src = acc / "selection" / "joint_selection_v2.json"
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    st = FG.run(REAL / "final_robot_dog_episode_000000", acc, tmp_path / "x", robot_archive=DOG / "discovery/robot_archive.npz",
                log=lambda *x: None)
    assert st["B0"] == "MODEL_INCONSISTENT/none" and st["B3"] == "TYPE_UNOBSERVABLE/prismatic"
    assert st["force_evidence_status"] == FG.NONDISC and st["export_ready"] is False
    for f in ("force_vs_ee.csv", "force_vs_candidate_motion.csv", "selection_baseline.json",
              "selection_corrected_visual.json", "selection_force_guided.json", "selection_diff.json", "summary.md"):
        assert (tmp_path / "x" / f).is_file(), f
    nof = FG.run(REAL / "final_robot_dog_episode_000000", acc, tmp_path / "y", robot_archive=None, log=lambda *x: None)
    assert nof["force_evidence_status"] == FG.FORCE_UNAVAILABLE and nof["B3"] == st["B1"]  # visual still works
    with pytest.raises(FileExistsError):
        FG.run(REAL / "final_robot_dog_episode_000000", acc, tmp_path / "x", log=lambda *x: None)
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before                    # no source rewritten
