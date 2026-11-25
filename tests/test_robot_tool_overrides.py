"""configs/robot_tool_overrides.json: operator decisions applied at read time; the raw scene file is never modified."""
import json
from pathlib import Path

import pytest

from estimotion_2part.joint_discovery.robot_tool import OVERRIDES, load_robot_tool

DRAWER = "white_drawer_20261021_161154_episode_000000"
REAL = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real")
INVALID = {"tool": "bare_force_sensor", "T_ee_tcp": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.03487365], [0, 0, 0, 1]],
           "wrench_frame": {"axes": "unspecified", "R_ee_force_sensor": None}, "tcp_calibration_valid": False,
           "calibration_status": "UNVERIFIED_DIFFERENT_MOUNTING", "recorded_pose_frame": "ee"}


def _scene(tmp_path, name, tool):
    (tmp_path / name / "metadata").mkdir(parents=True)
    p = tmp_path / name / "metadata" / "robot_tool.json"
    p.write_text(json.dumps(tool))
    return tmp_path / name, p


def test_drawer_gets_the_bare_force_sensor_settings_and_the_raw_file_is_untouched(tmp_path):
    scene, p = _scene(tmp_path, DRAWER, INVALID)
    before = p.read_bytes()
    t = load_robot_tool(scene)
    assert t["tcp_calibration_valid"] is True and t["tool"] == "bare_force_sensor"
    assert t["wrench_frame"]["axes"] == "ee" and t["wrench_frame"]["R_ee_force_sensor"] == [[1, 0, 0], [0, 1, 0],
                                                                                              [0, 0, 1]]
    assert t["T_ee_tcp"][2][3] == 0.03487365 and t["recorded_pose_frame"] == "ee"      # untouched keys kept
    ov = t["tool_override"]
    assert ov["status"] == "OPERATOR_NOMINAL_BARE_FORCE_SENSOR" and "NOT resolved" in ov["caveat"]
    assert set(ov["replaced"]) == {"tool", "T_ee_tcp", "wrench_frame", "tcp_calibration_valid", "calibration_status"}
    assert len(ov["sha256"]) == 64
    assert p.read_bytes() == before


def test_other_scenes_are_returned_unchanged(tmp_path):
    scene, _ = _scene(tmp_path, "final_red_fridge_episode_000000", INVALID)
    t = load_robot_tool(scene)
    assert t.pop("tool_override") is None and t == INVALID
    assert load_robot_tool(tmp_path / "nope", required=False) is None
    with pytest.raises(FileNotFoundError):
        load_robot_tool(tmp_path / "nope")


def test_override_file_only_lists_the_drawer_episode_000000():
    doc = json.loads(OVERRIDES.read_text())
    assert list(doc["scenes"]) == [DRAWER]


@pytest.mark.skipif(not (REAL / DRAWER).is_dir(), reason="real datasets not mounted")
def test_real_drawer_now_passes_the_g10_wrench_contract(tmp_path):
    import numpy as np
    from forceartgs_g9.wrench_input import full_record as FR
    arc = tmp_path / "a.npz"
    T = np.repeat(np.eye(4)[None], 3, 0)
    np.savez(arc, wrench=np.zeros((3, 6)), T_base_ee=T, T_base_tcp=T, source_row=np.arange(3))
    c = FR.load_contract(REAL / DRAWER, arc)
    assert c["tool_override"]["status"] == "OPERATOR_NOMINAL_BARE_FORCE_SENSOR"
