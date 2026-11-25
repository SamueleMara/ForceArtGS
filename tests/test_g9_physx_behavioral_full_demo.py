"""G10 wrench-driven interaction replay on the TEST DOUBLE of the worker's step law (the real engine is in
test_g9_physx_behavioral_engine.py): T10 contact ON inside the EXISTING interaction segments (every row incl. zero /
low effort), OFF (exactly zero) outside, no label read,
T11 no segment / regime gate is called, T12 q0 != 0 kept and v0 = 0 exactly in every candidate, T13 configuration
refusals, T14 one uninterrupted rollout per candidate, T15 ZOH native input + every valid visual sample, T17 the
near-optimal family is scored by actual forced replays, T22 an inadequate fit still saves a numeric candidate."""
import numpy as np
import pytest

from forceartgs_g9.physx_refinement import behavioral_adapter as BA
from forceartgs_g9.physx_refinement import behavioral_fit as BF
from forceartgs_g9.physx_refinement import behavioral_report as BR
from forceartgs_g9.physx_refinement.behavioral_contracts import (START_EXPERIMENT_KNOWN_ZERO, BehavioralContractError,
                                                                 load_wrench_config, wrench_to_fit_config)
from forceartgs_g9.physx_refinement.behavioral_real import starts_for

from g10_behavioral_fixtures import TestDoubleEngine, wrench_record

TRUTH = BF.params_from(dict(rho_scale=1.6, armature=0.0, dynamic_effort=0.4, static_excess=0.0, viscous=3.0))
CFG = load_wrench_config({"fitting": {"maxiter": 4, "popsize": 4, "local_max_nfev": 150, "local_starts": 2,
                                      "profile_points": 3, "profile_inner_max_nfev": 25,
                                      "profile_parameters": ["rho_scale"], "family_tolerance_sigma": 1.0},
                          "diagnostics": {"zero_gravity_alternative": False}})
FCFG = wrench_to_fit_config(CFG)


def _forbid(monkeypatch):
    """Any classifier / segment / contact-label / rest-test call fails the test."""
    from forceartgs_g9.free_forced import segments as SG
    from forceartgs_g9.physx_refinement import g9_adapter as GA

    def boom(*a, **k):
        raise AssertionError("a contact / regime / rest gate was called in the wrench-driven path")
    for mod, name in ((GA, "interactions_from_segments"), (SG, "classify"), (SG, "build_segments"),
                      (BA, "contact_labels"), (BA, "rest_consistency"), (BA, "input_onset"),
                      (BA, "build_full_demo_episode")):
        monkeypatch.setattr(mod, name, boom)


@pytest.fixture(scope="module")
def fitted():
    fr, rec, q_true, segs = wrench_record(TestDoubleEngine(), TRUTH, q0=0.15, seed=3, nan_q=7)   # truth engine
    out = BA.build_wrench_episode(fr, CFG, segs)
    assert out["status"] == "OK", out["reason"]
    ep = out["episode"]
    eng = TestDoubleEngine()                                                   # records ONLY the fit's rollouts
    roll = BF.CachingRollout(eng)
    b = BF.bounds_for(FCFG["model"], 1.0, 2.5)
    fit = BF.fit_full_demo(roll, ep, FCFG, b, starts_for(b, 1.0))
    fam = BF.profile_family(roll, fit, FCFG, b)
    return dict(eng=eng, fr=fr, ep=ep, roll=roll, fit=fit, fam=fam, bounds=b)


# ------------------------------------------------------------------------------------------- T10 / T11
def test_contact_on_inside_segments_zero_outside_and_no_gate_is_called(monkeypatch):
    _forbid(monkeypatch)
    eng = TestDoubleEngine()
    fr, rec, _, segs = wrench_record(eng, TRUTH)
    out = BA.build_wrench_episode(fr, CFG, segs)
    ep = out["episode"]
    assert out["status"] == "OK" and ep.contact_evidence["mode"] == "ON_DURING_INTERACTION"
    assert ep.contact_evidence["segments"] == [list(segs[0])]
    inside = (ep.effort_source_row >= segs[0][0]) & (ep.effort_source_row <= segs[0][1])
    rows = ep.effort_source_row.astype(int)
    np.testing.assert_array_equal(ep.effort_object[inside], fr.Q_full[rows[inside]])
    assert np.all(ep.effort_object[~inside] == 0.0) and np.any(fr.Q_full[rows[~inside]] != 0)
    assert (ep.effort_object[inside] == 0).sum() > 0 and (np.abs(ep.effort_object[inside]) == 0.02).sum() > 0
    assert out["evidence"]["post_interaction_coast_samples"] > 0                    # free coast after release kept
    roll = BF.CachingRollout(eng)
    b = BF.bounds_for(FCFG["model"], 1.0, 2.5)
    BF.fit_full_demo(roll, ep, FCFG, b, starts_for(b, 1.0))                  # the fit core needs no gate either


def test_interaction_start_after_visual_start_and_before():
    fr, rec, _, segs = wrench_record(TestDoubleEngine(), TRUTH)
    late = BA.build_wrench_episode(fr, CFG, [(13, segs[0][1])])
    assert late["status"] == "OK" and late["episode"].source_start_frame == 12      # last visual sample <= start
    fr.source_q_frame = fr.source_q_frame + 40
    fr.q_t_s = fr.source_q_frame / 50.0
    early = BA.build_wrench_episode(fr, CFG, segs)                                  # interaction before any visual q
    assert early["status"] == "FULL_START_NOT_AVAILABLE" and early["episode"] is None


def test_contact_labels_have_zero_influence():
    eng = TestDoubleEngine()
    fr1, rec, _, segs = wrench_record(eng, TRUTH)
    fr2, _, _, _ = wrench_record(eng, TRUTH)
    rec.contact_mode[:] = "uncertain"                                           # irrelevant: never an input
    e1, e2 = (BA.build_wrench_episode(f, CFG, segs)["episode"] for f in (fr1, fr2))
    for k in ("effort_object", "effort_t_s", "q_observed", "t_q_s"):
        np.testing.assert_array_equal(getattr(e1, k), getattr(e2, k))


# ------------------------------------------------------------------------------------------- T12 / T14 / T15
def test_q0_kept_and_v0_exactly_zero_in_every_candidate(fitted):
    ep, eng = fitted["ep"], fitted["eng"]
    assert ep.q0_observed == 0.15 and ep.start_basis == START_EXPERIMENT_KNOWN_ZERO
    reqs = eng.requests
    assert reqs and all(r["v0"] == 0.0 and r["q0"] == 0.15 and r["t0"] == ep.t0_s for r in reqs)
    assert fitted["fit"]["engine_initial_state"]["v0_engine"] == 0.0


def test_one_uninterrupted_rollout_per_candidate(fitted):
    n = len(fitted["ep"].t_q_s)
    assert all(r["n"] == n for r in fitted["eng"].requests)                    # whole demo, never a window / reset


def test_native_input_and_every_valid_visual_sample(fitted):
    fr, ep = fitted["fr"], fitted["ep"]
    valid = np.isfinite(fr.q)
    assert len(ep.t_q_s) == int(valid.sum()) == len(fr.q) - 1                 # only the NaN sample is dropped
    rows = ep.effort_source_row
    assert np.all(np.diff(rows) == 1) and ep.effort_t_s[0] <= ep.t0_s and ep.effort_t_s[-1] >= ep.t_q_s[-1] - 0.02
    np.testing.assert_allclose(ep.effort_t_s, rows / 50.0)


# ------------------------------------------------------------------------------------------- T13
@pytest.mark.parametrize("over", [
    {"initial_state": {"v0_nuisance_bound_sd": 3.0}}, {"initial_state": {"optimize_v0": True}},
    {"initial_state": {"optimize_q0": True}}, {"loss": {"fit_initial_velocity": True}},
    {"input": {"additional_tare": True}}, {"input": {"contact_classifier": "enabled"}},
    {"input": {"interaction_segments_required": False}}, {"input": {"estimate_tare": True}},
    {"input": {"outside_interaction_effort": "recorded"}}, {"physics": {"engine": "isaac"}},
    {"detect_contact": True}, {"input": {"prefix_velocity": True}},
    {"experimental_facts": {"contact_on_during_interaction": False}}, {"experimental_facts": {"initial_joint_velocity": 0.1}},
    {"physics": {"static_excess": 0.5}}, {"input": {"apply_transform_once": False}}, {"unknown_key": 1}])
def test_config_refuses_initial_state_tare_and_contact_options(over):
    with pytest.raises(BehavioralContractError):
        load_wrench_config(over)


# ------------------------------------------------------------------------------------------- T16-lite / T17
def test_fit_improves_over_a_wrong_baseline(fitted):
    fit = fitted["fit"]
    assert fit["improvement"]["rmse_ratio"] < 0.2 and fit["best"]["metrics"]["rmse_over_sigma"] < 3.0


def test_family_members_are_actual_forced_replays(fitted):
    fam, roll = fitted["fam"], fitted["roll"]
    pts = [p for p in fam["profiles"]["rho_scale"]["points"] if "metrics" in p]
    assert len(pts) >= 2
    ep = fitted["ep"].to_physx_episode()
    for p in pts:                                                              # re-simulate: the stored metric is real
        m = BF.replay_metrics(ep, roll(BF.PhysicalParameters(**p["params"]), ep), FCFG["loss"]["numeric_floor_q"])
        assert m["rmse_over_sigma"] == pytest.approx(p["metrics"]["rmse_over_sigma"], rel=1e-12)
    # with a measured input the members are NOT M lam* rescalings: each has its own re-optimized viscous / mass ratio
    ratios = {round(p["params"]["viscous"] / p["params"]["rho_scale"], 6) for p in pts}
    assert len(ratios) > 1


# ------------------------------------------------------------------------------------------- T22
def test_inadequate_fit_still_saves_a_numeric_candidate():
    eng = TestDoubleEngine(G1=2.0)
    fr, rec, _, segs = wrench_record(eng, TRUTH, Q_scale=0.4, phases=[(10, 0.0, "absent"), (60, 6.0, "sliding"),
                                                                      (220, 0.0, "absent")], seed=1)
    ep = BA.build_wrench_episode(fr, CFG, segs)["episode"]
    roll = BF.CachingRollout(eng)
    b = BF.bounds_for(FCFG["model"], 1.0, 6.0)
    fit = BF.fit_full_demo(roll, ep, FCFG, b, starts_for(b, 1.0))
    fam = BF.profile_family(roll, fit, FCFG, b)
    cls = BR.classify_wrench(fit, FCFG)
    assert cls["status"] == "BEHAVIORAL_FIT_INADEQUATE"
    info = dict(mass=1.0, axis_inertia=1.0, inertia_com_q0=np.eye(3).tolist(), com_q0=[0, 0, 0], sha256="t")
    cand = BR.wrench_candidate(scene="s", run="r", ep=ep, fr=fr, fit=fit, family=fam, cls=cls, info=info,
                               readback=dict(engine=dict(runtime_semantics="TEST_DOUBLE_NOT_PHYSX")), hashes={},
                               input_sens=dict(status="NOT_EVALUATED"), work=BF.work_audit(ep),
                               policy=dict(status="DISABLED"), alternative=None, pulse=BR.pulse_comparison(fr),
                               provenance={})
    pr = cand["candidate"]["parameters"]
    assert all(isinstance(v, float) for v in pr.values())
    assert cand["labels"]["status"] == "BEHAVIORAL_FIT_INADEQUATE" and not cand["labels"]["FIT_TO_DEMO"]
    assert cand["status"]["behavioral_preview_allowed"] is False
    assert cand["labels"]["TRANSFER_VALIDATED"] is False and cand["status"]["simulation_export_allowed"] is False
    assert cand["initial_state"]["v0_fixed"] == 0.0
