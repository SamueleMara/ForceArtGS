"""G10-A behavioral replay -- contracts (no engine): T1 initial-velocity contract, T2 no false zero position (data side),
T4 unknown contact stays a conditional measured input / missing input refused, start-state availability, T7 sign
reversal fails the passivity audit, T10 refusals, T11 output-claim contract."""
import copy

import numpy as np
import pytest

from forceartgs_g9.physx_refinement import behavioral_adapter as BA
from forceartgs_g9.physx_refinement import behavioral_fit as BF
from forceartgs_g9.physx_refinement import behavioral_report as BR
from forceartgs_g9.physx_refinement.behavioral_contracts import (DEFAULT_CONFIG, START_FULL_RECORDING,
                                                                 START_REST_VERIFIED_CROP, BehavioralContractError,
                                                                 FullDemoBehavioralEpisode, InputHypothesis,
                                                                 load_config)
from forceartgs_g9.physx_refinement.objective import Episode, LossSettings, RefinementLoss
from forceartgs_g9.physx_refinement.optimizer import FitConfiguration

from g10_behavioral_fixtures import TestDoubleEngine, push_profile, record, synthetic_demo

CFG = load_config()
CROP = load_config({"initial_state": {"mode": "rest_verified_cropped_start", "require_complete_demo_start": False}})
P = BF.params_from(dict(rho_scale=1.0, armature=0.0, dynamic_effort=0.3, static_excess=0.0, viscous=2.0))


def _ep(**kw):
    rec, _ = synthetic_demo(TestDoubleEngine(), P, **kw)
    out = BA.build_full_demo_episode(rec, CFG)
    assert out["status"] == "OK", out["reason"]
    return rec, out["episode"]


# ------------------------------------------------------------------------------------------- T1 / T2
def test_v0_is_exactly_zero_and_q0_is_the_authority_value():
    rec, ep = _ep(q0=0.15)
    assert ep.v0_fixed == 0.0 and ep.q0_observed == 0.15 == rec.q[0]
    pe = ep.to_physx_episode()
    assert pe.v0 == 0.0 and pe.sigma_v0 is None and pe.q0 == 0.15
    assert pe.regime_semantics.startswith("behavioral_full_demo")
    d = {f: getattr(ep, f) for f in ep.__dataclass_fields__}
    for bad in (1e-12, -0.0, np.float64(1e-9), True):
        with pytest.raises(BehavioralContractError):
            FullDemoBehavioralEpisode(**dict(d, v0_fixed=bad))
    with pytest.raises(BehavioralContractError):                       # q0 must be q(t0), never imposed
        FullDemoBehavioralEpisode(**dict(d, q0_observed=0.0))


def test_config_rejects_any_initial_state_fitting_and_unknown_keys():
    for over in ({"initial_state": {"optimize_v0": True}}, {"initial_state": {"optimize_q0": True}},
                 {"initial_state": {"initial_velocity_rad_or_m_per_s": 0.01}},
                 {"loss": {"optimize_initial_velocity": True}}, {"input": {"preserve_original_force_sign": False}},
                 {"input": {"missing_effort_policy": "hold_last"}}, {"simulation": {"engine": "isaac"}},
                 {"input": {"include_unknown_contact_conditionally": False}}, {"bogus": 1},
                 {"fitting": {"v0_nuisance_bound_sd": 3.0}},
                 {"initial_state": {"mode": "rest_verified_cropped_start"}}):          # opt-in must be explicit
        with pytest.raises(BehavioralContractError):
            load_config(over)


def test_optimizer_contract_rejects_v0_nuisance_and_static_friction():
    _, ep = _ep()
    eng = TestDoubleEngine()
    obj = RefinementLoss(eng, [ep.to_physx_episode()], BF.loss_settings(CFG))
    b = BF.bounds_for(CFG["model"], 1.0, 3.0)
    BF.assert_full_demo_fit(obj, BF.fit_configuration(b, CFG["fitting"]))
    with pytest.raises(BehavioralContractError):
        BF.assert_full_demo_fit(obj, FitConfiguration(bounds=b, v0_nuisance_bound_sd=3.0))
    with pytest.raises(BehavioralContractError):
        BF.assert_full_demo_fit(obj, BF.fit_configuration(dict(b, static_excess=(0.0, 1.0)), CFG["fitting"]))
    with pytest.raises(BehavioralContractError):
        BF.assert_full_demo_fit(RefinementLoss(eng, [ep.to_physx_episode()], LossSettings(fit_initial_velocity=True)),
                                BF.fit_configuration(b, CFG["fitting"]))
    legacy = Episode("x", ep.t_q_s, ep.q_observed, ep.q_sigma, ep.q0_observed, 0.05, ep.effort_t_s, ep.effort_object,
                     "forced", "VERIFIED", sigma_v0=0.01)
    assert legacy.regime_semantics == "segmented_regime_window"      # legacy default unchanged
    with pytest.raises(BehavioralContractError):
        BF.assert_full_demo_fit(RefinementLoss(eng, [legacy]), BF.fit_configuration(b, CFG["fitting"]))


# ------------------------------------------------------------------------------------------- start state
def test_cropped_authority_is_full_start_unavailable_in_strict_mode():
    rec, _ = synthetic_demo(TestDoubleEngine(), P)
    cut = record(q_frames=rec.source_q_frame[5:], q=rec.q[5:], force_rows=rec.source_force_row, Q=rec.Q_object,
                 modes=rec.contact_mode)
    out = BA.build_full_demo_episode(cut, CFG)
    assert out["status"] == "FULL_START_UNAVAILABLE" and out["episode"] is None
    assert out["evidence"]["uncovered_source_frames"] == [0, int(rec.source_q_frame[5]) - 1]


def test_cropped_start_opt_in_requires_rest_and_refuses_motion():
    eng = TestDoubleEngine()
    # rest at the crop (crop before the push onset) -> accepted, labelled
    rec, _ = synthetic_demo(eng, P, phases=[(40, 0.0, "absent"), (60, 2.5, "sliding"), (200, 0.0, "absent")])
    cut = record(q_frames=rec.source_q_frame[2:], q=rec.q[2:], force_rows=rec.source_force_row, Q=rec.Q_object,
                 modes=rec.contact_mode)
    ok = BA.build_full_demo_episode(cut, CROP)
    assert ok["status"] == "OK" and ok["episode"].start_basis == START_REST_VERIFIED_CROP
    assert not ok["episode"].start_is_full_demo and ok["evidence"]["rest_check"]["used_as_v0"] is False
    # crop in the middle of the motion -> refused (no rest state invented)
    mid = record(q_frames=rec.source_q_frame[20:], q=rec.q[20:], force_rows=rec.source_force_row, Q=rec.Q_object,
                 modes=rec.contact_mode)
    bad = BA.build_full_demo_episode(mid, CROP)
    assert bad["status"] == "FULL_START_UNAVAILABLE" and bad["episode"] is None       # input active: not testable
    # a crop where the object coasts with NO input (after release) is detected as moving
    coast = record(q_frames=rec.source_q_frame[27:], q=rec.q[27:], force_rows=rec.source_force_row, Q=rec.Q_object,
                   modes=rec.contact_mode)
    mv = BA.build_full_demo_episode(coast, CROP)
    assert mv["status"] == "FULL_START_UNAVAILABLE" and "moving" in mv["reason"]


def test_full_recording_start_basis():
    _, ep = _ep()
    assert ep.start_basis == START_FULL_RECORDING and ep.start_is_full_demo and ep.source_start_frame == 0


# ------------------------------------------------------------------------------------------- T4
def test_unknown_and_low_effort_contact_keep_the_recorded_input():
    rec, _ = synthetic_demo(TestDoubleEngine(), P)
    modes = rec.contact_mode.copy()
    modes[100:130] = "uncertain"
    rec2 = record(q_frames=rec.source_q_frame, q=rec.q, force_rows=rec.source_force_row, Q=rec.Q_object, modes=modes)
    ep = BA.build_full_demo_episode(rec2, CFG)["episode"]
    lab = np.asarray(ep.contact_evidence["labels_per_row"])
    assert (lab[100:130] == "CONTACT_UNKNOWN").all()
    assert np.array_equal(ep.effort_object, rec2.Q_object[:len(ep.effort_object)])      # never zeroed
    assert "MEASURED_NEAR_ZERO_CONTACT" in lab and "RELEASE_CONFIRMED" in lab
    assert np.array_equal(ep.to_physx_episode().input_effort, ep.effort_object)


def test_missing_effort_is_refused_not_filled():
    rec, _ = synthetic_demo(TestDoubleEngine(), P)
    keep = np.r_[np.arange(0, 80), np.arange(95, len(rec.source_force_row))]           # 15-row gap during the push
    gap = record(q_frames=rec.source_q_frame, q=rec.q, force_rows=rec.source_force_row[keep], Q=rec.Q_object[keep],
                 modes=rec.contact_mode[keep])
    out = BA.build_full_demo_episode(gap, CFG)
    assert out["status"] == "INPUT_COVERAGE_INCOMPLETE"
    assert any(m["kind"] == "ROW_GAP" for m in out["evidence"]["input_coverage"]["missing"])
    late = record(q_frames=rec.source_q_frame, q=rec.q, force_rows=rec.source_force_row[3:], Q=rec.Q_object[3:],
                  modes=rec.contact_mode[3:])
    out2 = BA.build_full_demo_episode(late, CFG)
    assert out2["status"] in ("INPUT_COVERAGE_INCOMPLETE", "FULL_START_UNAVAILABLE")


def test_input_hypotheses_are_grounded_and_never_sign_flips():
    rec, ep = _ep()
    hyps, skipped = BA.input_hypotheses(rec, ep, CFG)
    kinds = {h.kind for h in hyps}
    assert kinds == {"recorded", "time_offset", "noise_realization", "contact_interpretation"} or \
        kinds >= {"recorded", "noise_realization", "contact_interpretation"}
    for h in hyps:
        assert h.effort_object.shape == ep.effort_object.shape
        assert float(h.effort_object @ ep.effort_object) > 0                         # same direction
    with pytest.raises(BehavioralContractError):
        InputHypothesis("sign_flip", "recorded", -ep.effort_object, "x")


# ------------------------------------------------------------------------------------------- T7 (audit)
def test_sign_reversal_fails_the_passivity_audit():
    rec, ep = _ep()
    assert BF.work_audit(ep)["verdict"] == "PASSIVE_ENERGY_CONSISTENT"
    d = {f: getattr(ep, f) for f in ep.__dataclass_fields__}
    rev = FullDemoBehavioralEpisode(**dict(d, effort_object=-ep.effort_object))
    w = BF.work_audit(rev)
    assert w["verdict"] == "INPUT_OUTPUT_SIGN_INCONSISTENT" and w["W_in"] < 0


# ------------------------------------------------------------------------------------------- T10 / T11
def test_native_backends_refuse_and_g10b_is_not_implemented():
    from forceartgs_g9.physx_refinement.api import PhysXIsaacRollout
    from forceartgs_g9.physx_refinement.robot_replay import detect_backends
    with pytest.raises(NotImplementedError):
        PhysXIsaacRollout()
    b = detect_backends(dict(rec=dict(p_tcp_base_m=1, R_base_ee=1)))
    assert b["g10b_status"] == "NOT_IMPLEMENTED_BACKEND"
    assert b["backends"]["robot_action_contact_replay"]["status"] == "NOT_IMPLEMENTED_BACKEND"
    assert b["backends"]["measured_effort_replay"]["status"] == "AVAILABLE"


def test_candidate_claims_are_honest_and_certified_contracts_unchanged():
    ok = dict(status=dict(behavioral_status="BEHAVIORAL_CANDIDATE_UNVALIDATED", behavioral_preview_allowed=True,
                          physical_parameters_uniquely_identified=False, independent_real_prediction_validated=False,
                          native_target_physx_friction_verified=False, sim_to_real_validated=False,
                          simulation_export_allowed=False, native_engine_target_ready=False,
                          validated_on_new_real_action=False, replay_label="GOOD_DEMO_REPLAY_UNDER_CONDITIONAL_INPUT"))
    BR.assert_claims_honest(ok)
    for k in ("physical_parameters_uniquely_identified", "sim_to_real_validated", "simulation_export_allowed"):
        bad = copy.deepcopy(ok)
        bad["status"][k] = True
        with pytest.raises(AssertionError):
            BR.assert_claims_honest(bad)
    bad = copy.deepcopy(ok)
    bad["status"]["replay_label"] = "PRODUCTION_READY"
    with pytest.raises(AssertionError):
        BR.assert_claims_honest(bad)
    bad = copy.deepcopy(ok)
    bad["status"]["behavioral_status"] = "BEHAVIORAL_FIT_INADEQUATE"
    with pytest.raises(AssertionError):
        BR.assert_claims_honest(bad)
    from forceartgs_g9.system_identification.contracts import IOContractError, assert_no_value_field
    with pytest.raises(IOContractError):
        assert_no_value_field({"a": {"value": 1.0}})                                # certified G9 contract intact
    assert DEFAULT_CONFIG["output"]["do_not_modify_certified_g9_handoff"] is True
