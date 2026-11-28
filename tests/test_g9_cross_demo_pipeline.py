"""Cross-demo contracts with actual PhysX on tiny synthetic fixtures (spec section 11: inventory T01-T04, geometry
T08-T13, held-out T14/T47, leakage T16/T17, frozen engine T18-T20/T22, matrix T23, no-fit T26, GT T29, resume T30)."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9.cross_demo import cv, heldout_episode as H, inventory as I, kinematics_alignment as K  # noqa: E402
from forceartgs_g9.cross_demo import replay as R  # noqa: E402

CFG = dict(primary_method="B0", alignment=dict(axis_tol_deg=10.0, hinge_tol_m=0.05),
           scoring=dict(sigma_physx_floor=0.0, travel_noise_k=5.0))


# ------------------------------------------------------------------------------------------------ fixtures
def _model(tmp_path, jt, M, c, name="train", axis=None, origin=None):
    from forceartgs_g9.physx_refinement.urdf_builder import write_single_joint_urdf
    axis = [1.0, 0, 0] if axis is None else axis
    origin = [0, 0, 0] if origin is None else origin
    u = tmp_path / f"{name}.urdf"
    info = write_single_joint_urdf(u, joint_type=jt, axis=axis, origin=origin, mass=2.0,
                                   com_ref=[0.3, 0.1, 0.0] if jt == "prismatic" else [0.4, 0.0, 0.0],
                                   inertia_com_ref=np.diag([0.02, 0.03, 0.04]))
    rho = M / info["axis_inertia"]
    return R.TrainModel(name=name, r2r_dir="", method="B0", chosen_method="B0",
                        params=dict(rho_scale=rho, viscous=c, armature=0.0, static_effort=0.0, dynamic_effort=0.0),
                        M=M, c=c, rho_scale=rho, axis_inertia=info["axis_inertia"], joint_type=jt,
                        axis_B=np.asarray(info["axis"], float),
                        origin_B=np.asarray(info["origin"], float) if jt == "revolute" else None, urdf=str(u),
                        urdf_sha256="t", moving_link=info["moving_link"], joint_name=info["joint_name"],
                        timestep=1e-3, friction_mode="external_effort_surrogate", training_objective=0.0,
                        at_bound={}, episode_sha256="t")


def _record(jt, M, c, name, pushes=((0.3, 3.0),), q0=0.1, axis=(1.0, 0, 0), origin=(0, 0, 0), noise=0.0):
    from forceartgs_g9.estimators.rest_to_rest_synthetic import episode as syn_episode, exact_rest_to_rest
    sy = exact_rest_to_rest(M, c, q0=q0, pushes=tuple((d, k * M) for d, k in pushes), noise=noise,
                            q_sigma=max(noise, 2e-4), seed=0)
    b = syn_episode(sy, joint_type=jt)
    assert b["status"] == "OK", b["reason"]
    return H.synthetic_record(name, jt, axis, origin if jt == "revolute" else None, sy["t_q"], sy["q"],
                              np.full(len(sy["t_q"]), sy["q_sigma"]), sy["force_t"], sy["Q"], b["evidence"].to_dict(),
                              q_true=sy["q_true"])


def _geo(rec):
    return dict(joint_type=rec["joint_type"], axis_B=rec["axis_B"], origin_B=rec["origin_B"])


def _held(model, rec, **kw):
    al = K.align(model.geometry(), _geo(rec), **CFG["alignment"])
    return H.build(model.geometry(), rec, al, **kw)


# ------------------------------------------------------------------------------------------------ inventory
def _fake_episode(root, name, *, rec_bytes=b"x", missing=()):
    d = root / name
    for cam in ("cam_001", "cam_002"):
        if f"RGB_{cam}" not in missing:
            (d / "rgb" / cam).mkdir(parents=True)
            (d / "rgb" / cam / "0.png").write_bytes(b"p")
    for rel in ("depth_npy", "mask"):
        (d / rel).mkdir(parents=True)
        (d / rel / "0").write_bytes(b"d")
    (d / "metadata" / "raw").mkdir(parents=True)
    (d / "metadata" / "cameras.json").write_text("{}")
    (d / "metadata" / "contact_force_convention.json").write_text("{}")
    (d / "metadata" / "frame_timestamps.csv").write_text("frame_index,source_frame_index,timestamp_s\n0,0,0.0\n1,2,0.04\n")
    if "ROBOT_RECORDING" not in missing:
        (d / "metadata" / "raw" / f"episode_{name[-6:]}.parquet").write_bytes(rec_bytes)


def test_T01_T03_T04_inventory_ids_duplicates_and_missing(tmp_path):
    for e, b in (("000000", b"a"), ("000001", b"b"), ("000005", b"c")):
        _fake_episode(tmp_path, f"obj_episode_{e}", rec_bytes=b)
    _fake_episode(tmp_path, "obj_episode_000007", rec_bytes=b"a")                    # same push as 000000
    _fake_episode(tmp_path, "obj_episode_000009", missing=("ROBOT_RECORDING",))
    _fake_episode(tmp_path, "other_episode_000002", missing=("RGB_cam_002",))
    eps = {r["episode"]: r for r in I.scan(tmp_path)}
    man = I.candidate_manifest(list(eps.values()), tmp_path)
    obj = next(o for o in man["objects"] if o["object_id"] == "obj")
    assert obj["episodes"] == [f"obj_episode_{e}" for e in ("000000", "000001", "000005", "000007", "000009")]
    assert obj["verified_same_physical_object"] is None                                # never self-certified
    assert eps["obj_episode_000007"]["status"].startswith("DUPLICATE_OF_")
    assert eps["obj_episode_000009"]["status"] == "MISSING_ROBOT_RECORDING"
    assert eps["other_episode_000002"]["status"] == "MISSING_RGB_cam_002"
    g = {x["object_id"]: x for x in I.groups_for_transfer(man, list(eps.values()))}
    assert g["obj"]["status"] == "OBJECT_IDENTITY_UNVERIFIED"
    obj["verified_same_physical_object"] = True
    g = {x["object_id"]: x for x in I.groups_for_transfer(man, list(eps.values()))}
    assert g["obj"]["status"] == "OK" and "obj_episode_000007" not in g["obj"]["eligible"]


def test_T02_manifest_refuses_an_episode_in_two_objects(tmp_path):
    import yaml
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(dict(schema=I.MANIFEST_SCHEMA, objects=[dict(object_id="a", episodes=["x_episode_000000"]),
                                                                        dict(object_id="b", episodes=["x_episode_000000"])])))
    with pytest.raises(ValueError, match="both"):
        I.load_manifest(p)


# ------------------------------------------------------------------------------------------------ geometry / gauge
def test_T08_test_effort_projected_on_the_TRAINING_joint():
    rng = np.random.default_rng(0)
    n = 20
    rec = dict(Q_rows=None, F_B=rng.normal(size=(n, 3)), tau_B=rng.normal(size=(n, 3)), p_ref_B=rng.normal(size=(n, 3)))
    a, o = np.array([0, 0, 1.0]), np.array([0.1, 0.2, 0.0])
    Q_train = H.project_effort(rec, "revolute", a, o)
    Q_test_geo = H.project_effort(rec, "revolute", a, o + [0.03, 0, 0])
    exp = (rec["tau_B"] + np.cross(rec["p_ref_B"] - o, rec["F_B"])) @ a
    np.testing.assert_allclose(Q_train, exp, atol=1e-14)
    assert np.abs(Q_train - Q_test_geo).max() > 1e-3                                 # target geometry would leak
    np.testing.assert_allclose(H.project_effort(rec, "prismatic", a, None), rec["F_B"] @ a, atol=1e-14)


def test_T10_T11_mismatched_geometry_is_refused_not_optimized():
    tr = dict(joint_type="revolute", axis_B=[0, 0, 1.0], origin_B=[0, 0, 0])
    assert K.align(tr, dict(tr, joint_type="prismatic", origin_B=None), **CFG["alignment"])["status"] == \
        "KINEMATICS_TYPE_MISMATCH"
    assert K.align(tr, dict(tr, axis_B=[0, np.sin(0.5), np.cos(0.5)]), **CFG["alignment"])["status"] == "AXIS_MISMATCH"
    assert K.align(tr, dict(tr, origin_B=[0.2, 0, 0]), **CFG["alignment"])["status"] == "HINGE_MISMATCH"
    ok = K.align(tr, dict(tr, origin_B=[0.01, 0, 0.5]), **CFG["alignment"])                  # same line, other point
    assert ok["status"] == "OK" and ok["hinge_line_distance_m"] == pytest.approx(0.01)


def test_T09_prismatic_axis_sign_reversal_gives_the_same_physical_prediction(tmp_path):
    m = _model(tmp_path, "prismatic", 0.8, 1.5)
    rec = _record("prismatic", 0.8, 1.5, "t")
    ev = json.loads(json.dumps(rec["evidence"]))                       # the flipped joint's own plateau evidence
    ev["initial"]["q0"] = -ev["initial"]["q0"]
    flip = dict(rec, axis_B=-rec["axis_B"], q=-rec["q"], q_true=-rec["q_true"], Q_rows=rec["Q_rows"], evidence=ev)
    h1, h2 = _held(m, rec), _held(m, flip)
    assert h2.provenance["sign"] == -1.0
    np.testing.assert_allclose(h1.q_ref, h2.q_ref, atol=1e-12)                        # same displacement, train gauge


def test_T12_T13_gauge_uses_only_the_initial_rest_and_respects_q0(tmp_path):
    m = _model(tmp_path, "prismatic", 0.8, 1.5)
    a, b = _record("prismatic", 0.8, 1.5, "a", q0=0.10), _record("prismatic", 0.8, 1.5, "b", q0=0.37)
    ha, hb = _held(m, a), _held(m, b)
    np.testing.assert_allclose(ha.q_ref, hb.q_ref, atol=1e-9)                          # q0 removed by the gauge only
    assert hb.provenance["q0_test_raw"] == pytest.approx(0.37, abs=1e-6)
    assert ha.episode.q0 == 0.0 and ha.episode.v0 == 0.0 and ha.episode.used_for_training is False


def test_T14_no_initial_plateau_means_no_v0_assumption(tmp_path):
    m = _model(tmp_path, "prismatic", 0.8, 1.5)
    rec = _record("prismatic", 0.8, 1.5, "t")
    ev = json.loads(json.dumps(rec["evidence"]))
    ev["initial"]["found"] = False
    assert _held(m, dict(rec, evidence=ev)).status == "INITIAL_REST_NOT_COVERED"


def test_T47_tail_without_force_rows_is_flagged_and_stop_scope_survives(tmp_path):
    m = _model(tmp_path, "prismatic", 0.8, 1.5)
    rec = _record("prismatic", 0.8, 1.5, "t")
    h_full = _held(m, rec)
    keep = rec["force_t"] <= h_full.T_stop + 1e-9
    cut = dict(rec, force_t=rec["force_t"][keep], Q_rows=rec["Q_rows"][keep], Q_own=rec["Q_own"][keep],
               source_row=rec["source_row"][keep])
    assert _held(m, cut).status == "TAIL_INPUT_INCOMPLETE"
    h = _held(m, cut, scope="stop")
    assert h.status == "OK" and h.t_s[-1] == pytest.approx(h_full.T_stop)
    assert h_full.T_eval_end > h_full.T_stop and h_full.t_s[-1] == pytest.approx(h_full.T_eval_end)


# ------------------------------------------------------------------------------------------------ engine
@pytest.fixture(scope="module")
def pris(tmp_path_factory):
    pytest.importorskip("sapien")
    t = tmp_path_factory.mktemp("pris")
    m = _model(t, "prismatic", 0.8, 1.5)
    eng = R.FrozenEngine(m)
    yield m, eng
    eng.close()


def test_T16_T17_reference_never_reaches_the_simulation(pris):
    m, eng = pris
    h = _held(m, _record("prismatic", 0.8, 1.5, "t", noise=1e-4))
    r1 = eng.rollout(h.episode)
    r2 = eng.rollout(dataclasses.replace(h.episode, q=np.random.default_rng(1).normal(size=len(h.q_ref))))
    assert np.array_equal(r1["q_sim"], r2["q_sim"])
    with pytest.raises(ValueError, match="used_for_training"):
        eng.rollout(dataclasses.replace(h.episode, used_for_training=True))


def test_T18_T19_T22_frozen_train_asset_readback_and_input_dependence(pris):
    m, eng = pris
    h1 = _held(m, _record("prismatic", 0.8, 1.5, "a", pushes=((0.3, 3.0),)))
    h2 = _held(m, _record("prismatic", 0.8, 1.5, "b", pushes=((0.4, -2.0),)))
    r1, r2 = eng.rollout(h1.episode), eng.rollout(h2.episode)
    assert r1["axis_inertia_engine"] == pytest.approx(m.M, rel=1e-4)
    assert r1["initial_q_error"] == 0.0
    assert np.sign(r1["q_sim"][-1]) != np.sign(r2["q_sim"][-1])                          # the target's own input


def test_T29_frozen_true_parameters_predict_unseen_pushes_against_exact_GT(pris):
    m, eng = pris
    for pushes in (((0.3, 3.0),), ((0.2, 4.0), (0.2, 1.0)), ((0.5, -1.5),)):
        h = _held(m, _record("prismatic", 0.8, 1.5, "u", pushes=pushes))
        q = eng.rollout(h.episode)["q_sim"]
        span = np.ptp(h.q_true)
        assert np.max(np.abs(q - h.q_true)) < 0.02 * span                                 # PhysX vs exact two-term


def test_T20_failed_or_nonfinite_training_fit_is_refused(tmp_path):
    d = tmp_path / "r2r"
    d.mkdir()
    (d / "STATUS.json").write_text(json.dumps(dict(status="COMPARED")))
    (d / "physx_comparison.json").write_text(json.dumps(dict(fits=dict(
        B0=dict(status="FIT_FAILED", error="x"), B7=dict(status="FITTED", params=dict(
            rho_scale=float("nan"), viscous=1.0, armature=0, static_effort=0, dynamic_effort=0))))))
    for m in ("B0", "B7", "B9"):
        with pytest.raises(R.TrainModelError):
            R.load_train_model(d, m)
    (d / "STATUS.json").write_text(json.dumps(dict(status="DATA_UNAVAILABLE")))
    with pytest.raises(R.TrainModelError, match="training status"):
        R.load_train_model(d, "B0")


# ------------------------------------------------------------------------------------------------ matrix
def test_T23_T26_T30_four_episode_matrix_no_fit_and_resume(tmp_path, monkeypatch):
    pytest.importorskip("sapien")
    from forceartgs_g9.physx_refinement import behavioral_fit as BF
    for f in ("fit_full_demo", "profile_family"):
        monkeypatch.setattr(BF, f, lambda *a, **k: (_ for _ in ()).throw(AssertionError("fit called on a target")))
    pushes = (((0.3, 3.0),), ((0.4, -2.0),), ((0.2, 4.0), (0.2, 1.0)), ((0.5, 1.5),))
    names = [f"syn_episode_{k:06d}" for k in range(4)]
    recs = {n: _record("prismatic", 0.8, 1.5, n, pushes=p, q0=0.05 * k) for k, (n, p) in enumerate(zip(names, pushes))}
    models = {(n, "B0"): _model(tmp_path, "prismatic", 0.8 * (1 + 0.05 * k), 1.5, name=n) for k, n in enumerate(names)}
    cells = cv.run_pairs("syn", {n: "synthetic" for n in names}, "B0", CFG, tmp_path, records=recs, models=models,
                         log=lambda *_: None)
    assert len(cells) == 16 and sum(c["is_train_diagonal"] for c in cells) == 4
    assert all(c["status"] == "OK" for c in cells)
    by = {(c["train_id"], c["test_id"]): c for c in cells}
    assert by[(names[0], names[1])]["metrics"]["rmse"] != by[(names[1], names[0])]["metrics"]["rmse"]
    p = tmp_path / "per_pair" / "syn" / "train_000000" / "test_000001" / "B0" / "seed0"
    assert {f.name for f in p.iterdir()} >= {"STATUS.json", "replay.npz", "residuals.npz", "metrics.json",
                                             "provenance.json"}
    mt = p.joinpath("STATUS.json").stat().st_mtime_ns
    again = cv.run_pairs("syn", {n: "synthetic" for n in names}, "B0", CFG, tmp_path, records=recs, models=models,
                         log=lambda *_: None)
    assert p.joinpath("STATUS.json").stat().st_mtime_ns == mt                                     # reused by key
    assert [c["metrics"]["rmse"] for c in again] == [c["metrics"]["rmse"] for c in cells]
    cfg2 = dict(CFG, scoring=dict(CFG["scoring"], travel_noise_k=6.0))
    cv.run_pairs("syn", {n: "synthetic" for n in names}, "B0", cfg2, tmp_path, records=recs, models=models,
                 log=lambda *_: None)
    assert p.joinpath("STATUS.json").stat().st_mtime_ns != mt                                     # new key: redone


def test_T24_single_episode_object_has_no_transfer(tmp_path):
    pytest.importorskip("sapien")
    n = "solo_episode_000000"
    cells = cv.run_pairs("solo", {n: "synthetic"}, "B0", CFG, tmp_path,
                         records={n: _record("prismatic", 0.8, 1.5, n)},
                         models={(n, "B0"): _model(tmp_path, "prismatic", 0.8, 1.5, name=n)}, log=lambda *_: None)
    from forceartgs_g9.cross_demo import metrics as MX
    assert len(cells) == 1 and MX.aggregate_matrix(cells, episodes=[n])["status"] == "INSUFFICIENT_EPISODES"


def test_T28_pooled_loo_is_reported_not_substituted():
    assert cv.pooled_loo()["status"] == "POOLED_LOO_NOT_IMPLEMENTED"


def test_T07_continuous_input_keeps_every_native_row_including_near_zero(tmp_path):
    m = _model(tmp_path, "prismatic", 0.8, 1.5)
    rec = _record("prismatic", 0.8, 1.5, "t")
    h = _held(m, rec)
    inside = (rec["force_t"] >= h.input_t_s[0] - 1e-12) & (rec["force_t"] <= h.T_eval_end + 1e-9)
    assert len(h.input_t_s) == inside.sum()                                             # no mask, no gating
    assert np.sum(np.abs(h.input_Q) < 1e-12) > 0                                         # zero-effort rows kept
    np.testing.assert_array_equal(h.input_Q, rec["Q_rows"][inside])                      # values untouched (no tare)


REAL_R2R = Path("/media/samuelemara/Extreme SSD/Samuele/results_28_11_26/rest_to_rest_compare/"
                "final_red_fridge_episode_000000/rest_to_rest_compare/v000")


@pytest.mark.skipif(not (REAL_R2R / "physx_comparison.json").is_file(), reason="REAL_TEST_NOT_RUN: SSD results absent")
def test_T21_T32_real_diagonal_parity_and_one_real_replay_with_provenance(tmp_path):
    pytest.importorskip("sapien")
    m = R.load_train_model(REAL_R2R, "B0_G10_BASELINE_STARTS")
    eng = R.FrozenEngine(m)
    try:
        par = R.diagonal_parity(eng)
        assert par["status"] == "PARITY_OK", par
        cfg = dict(CFG, primary_method="B0_G10_BASELINE_STARTS")
        cells = cv.run_pairs("fridge", {"final_red_fridge_episode_000000": str(REAL_R2R)}, "B0_G10_BASELINE_STARTS",
                             cfg, tmp_path, log=lambda *_: None)
    finally:
        eng.close()
    assert cells[0]["status"] == "OK" and cells[0]["scope"] == "T_EVAL_END"
    prov = json.loads(next(tmp_path.glob("per_pair/**/provenance.json")).read_text())
    assert prov["no_fit_on_target"] and prov["train"]["urdf_sha256"] == m.urdf_sha256
    assert prov["test"]["ref_kind"] == "RGBD_RECONSTRUCTED" and prov["parity"]["status"] == "PARITY_OK"
