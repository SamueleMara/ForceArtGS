"""Controlled comparison plumbing (T21, T23, T24): matched rows, strict config, append-only output, real-data coverage."""
import glob
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from forceartgs_g9.physx_refinement import rest_to_rest_comparison as RC
from forceartgs_g9.physx_refinement import rest_to_rest_report as RP
from forceartgs_g9.physx_refinement.sapien_diagnostic import default_engine_python

CFG = "configs/g9_real/rest_to_rest_compare.yaml"
REAL = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real")
RUNS = Path("/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware")
DISC = Path("/media/samuelemara/Extreme SSD/Samuele/results_17_11_26/overnight_real/joint")
HAVE = importlib.util.find_spec("sapien") is not None or default_engine_python() is not None


# ------------------------------------------------------------------------------------------------ T24 config / CLI
def test_T24_config_is_strict_and_rejects_hidden_resets_tares_and_contact_logic():
    cfg = RC.load_r2r_config(CFG)
    assert cfg["model"]["terms"] == ["inertia", "viscous"] and 0.0 in cfg["physx"]["terminal_velocity_loss_weights"]
    for bad, msg in (({"episode": {"foo": 1}}, "unknown key"), ({"reset_qdot": True}, "banned"),
                     ({"input": {"tare": 0.1}}, "banned"), ({"input": {"contact_mask": "segments"}}, "contact_mask"),
                     ({"experimental_facts": {"continuous_contact": False}}, "experimental_facts"),
                     ({"model": {"terms": ["inertia", "viscous", "coulomb"]}}, "SEPARATE"),
                     ({"physx": {"terminal_velocity_loss_weights": [0.1]}}, "contain 0"),
                     ({"pinn": {"initial_velocity_nuisance": True}}, "initial_velocity_nuisance"),
                     ({"episode": {"reject_cropped_moving_start": False}}, "reject_cropped_moving_start")):
        with pytest.raises(RC.R2RConfigError, match=msg):
            RC.load_r2r_config(bad)


def test_T24_old_cli_and_simplified_config_are_unchanged():
    py = sys.executable
    for cmd in ("physx-behavioral-wrench", "rest-to-rest-compare", "io-identification"):
        r = subprocess.run([str(py), "-m", "forceartgs_g9.cli", cmd, "--help"], capture_output=True, text=True)
        assert r.returncode == 0, (cmd, r.stderr[-400:])
    head = subprocess.run(["git", "show", "HEAD:configs/g9_real/physx_behavioral_inertia_damping.yaml"],
                          capture_output=True, text=True)
    if head.returncode == 0:
        assert Path("configs/g9_real/physx_behavioral_inertia_damping.yaml").read_text() == head.stdout


def test_T24_outputs_are_append_only_versions(tmp_path):
    from forceartgs_g9.forcetwin.pipeline import next_version
    a = next_version(tmp_path / "rest_to_rest_compare")
    (a / "STATUS.json").write_text("{}")
    b = next_version(tmp_path / "rest_to_rest_compare")
    assert a.name == "v000" and b.name == "v001" and (a / "STATUS.json").read_text() == "{}"


# ------------------------------------------------------------------------------------------------ T21 controlled rows
def test_T21_controlled_table_refuses_mixed_inputs_and_carries_hashes():
    base = dict(input_policy="C1_FULL_UNMASKED", episode_sha256="a", model="m", engine="e", window="w", method="x")
    RP.assert_controlled([base, dict(base, method="y")])
    with pytest.raises(ValueError, match="episode_sha256"):
        RP.assert_controlled([base, dict(base, episode_sha256="b", method="y")])
    with pytest.raises(ValueError, match="model"):
        RP.assert_controlled([base, dict(base, model="full", method="y")])
    with pytest.raises(ValueError, match="input policy"):
        RP.assert_controlled([base, dict(base, input_policy=None, method="z")])
    RP.assert_controlled([base, dict(base, input_policy="C0_CONTACT_MASKED (fit) -> C1 replay", episode_sha256="old",
                                     method="C0")])               # C0 is the labelled archival control, not a C1 row


@pytest.mark.skipif(not HAVE, reason="no SAPIEN interpreter")
def test_T21_T22_synthetic_end_to_end_benchmark_on_actual_physx(tmp_path):
    wo = {"fitting": {"maxiter": 2, "popsize": 5, "local_max_nfev": 20, "local_starts": 1, "profile_points": 3,
                      "profile_inner_max_nfev": 8},
          "diagnostics": {"policy_probe": False}}
    ro = {"pinn": {"seeds": [0], "steps_data": 150, "steps_joint": 250, "n_quad": 64, "hidden": 8, "layers": 2},
          "analytical": {"bootstrap_draws": 10, "lambda_grid_points": 21},
          "physx": {"terminal_velocity_loss_weights": [0.0, 1.0]}, "benchmark": {"write_plots": True}}
    st = RC.run_synthetic(tmp_path / "syn", engine="sapien_cpu_diagnostic", wrench_overrides=wo, r2r_overrides=ro,
                          log=lambda *a: None)
    out = Path(st["out_dir"])
    rows = st["_rows"]
    ids = {r["method"] for r in rows}
    for need in ("B0_G10_BASELINE_STARTS", "B7_G10_SVD_START", "B8_G10_CONVOLUTION_START", "B10_G10_TERMINAL_LOSS_w1",
                 "B1_ENDPOINT_R2R", "B2_INTEGRAL_SVD_R2R", "B3_ANALYTIC_CONVOLUTION_1D"):
        assert need in ids, need
    c1 = RP.assert_controlled(rows)
    assert len({r["episode_sha256"] for r in c1}) == 1 and all(r["engine"] for r in c1)
    fits = st["_px"]["fits"]
    budgets = {(f["de_maxiter"], f["de_popsize"], f["de_seed"]) for f in fits.values() if f.get("status") == "FITTED"}
    assert len(budgets) == 1                                                          # identical DE budget everywhere
    for k, f in fits.items():
        if f.get("status") == "FITTED":
            assert f["n_engine_rollouts"] > 0 and f["metrics"]["v_sim_T"] is not None
    truth = st["truth"]
    b8 = fits["B8_G10_CONVOLUTION_START"]
    assert b8["M"] == pytest.approx(truth["M"], rel=0.1)                             # T22 plausibility
    for f in ("summary.csv", "summary.md", "physx_comparison.json", "equal_budget_convergence.json",
              "figures/trajectory_compare.png", "figures/optimizer_convergence.png", "figures/terminal_velocity.png"):
        assert (out / f).is_file(), f
    md = (out / "summary.md").read_text()
    assert "Input-policy ablation" in md and "Equal-budget G10" in md


# ------------------------------------------------------------------------------------------------ T23 real data
SCENES = ["final_red_fridge_episode_000000", "final_orange_suitecase_episode_000000", "final_black_box_episode_000000"]


@pytest.mark.skipif(not RUNS.is_dir(), reason="real runs not mounted")
@pytest.mark.parametrize("sc", SCENES)
def test_T23_real_scenes_give_an_episode_or_an_exact_coverage_status(sc):
    run = (glob.glob(str(RUNS / f"{sc}_multiview_*_joint_aware_full")) or [None])[0]
    if run is None:
        pytest.skip("no run")
    kw = dict(declared_joint_type="revolute", declaration_note="operator: declared revolute") if "black_box" in sc else {}
    r = RC.episode_only(run, str(REAL / sc), config=CFG, robot_archive=str(DISC / sc / "robot_archive.npz"),
                        log=lambda *a: None, **kw)
    assert r["status"] in ("OK", "REST_START_NOT_COVERED", "REST_END_NOT_COVERED", "NO_MOTION_OBSERVED")
    if r["status"] != "OK":
        assert "source frame" in r["reason"] and r["episode"] is None                 # precise frames, no fake crop
        key = "initial" if r["status"] == "REST_START_NOT_COVERED" else "terminal"
        assert r["evidence"][key]["missing_frames"]
    else:
        ep = r["episode"]
        assert ep.v0 == 0.0 and len(ep.force_Q) > 0
