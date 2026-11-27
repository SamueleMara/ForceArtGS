"""Repaired candidate-wise selector (spec tests 9-13, 15, 17): an R failure fails THAT revolute geometry, the slide is
evaluated on the same evidence, and certification / export gates are unchanged."""
import inspect

from estimotion_2part.joint_discovery import force_guided_selection as FG
from estimotion_2part.joint_discovery.physical_selection import select_physical_joint

from force_guided_fixtures import selector_inputs


def sel(force=None, legacy=False, **kw):
    ev = selector_inputs(**kw)
    rob = dict(contact=dict(status="ABSTAIN"), force=dict(status="ABSTAIN"))
    if force is not None:
        rob["force_guided"] = force
    return select_physical_joint(**ev, robot_evidence=rob, legacy_r_first=legacy)


def test_T10_remote_hinge_selects_a_prismatic_working_model_instead_of_model_inconsistent():
    old, new = sel(legacy=True), sel()
    assert (old.physical_joint_status, old.working_motion_model) == ("MODEL_INCONSISTENT", "none")
    assert (new.physical_joint_status, new.working_motion_model) == ("TYPE_UNOBSERVABLE", "prismatic")
    assert "REMOTE_HINGE_DEGENERACY" in new.failure_codes and new.selected_candidate is None
    assert any("slide evaluated on the same evidence" in r for r in new.reasons)


def test_T11_each_R_failure_fails_only_the_revolute_candidate():
    hinge = sel()
    axis = sel(hinge_fit="PLAUSIBLE", hinge_field="PLAUSIBLE", d10_fit=0.05, axis_dis=True)
    pred = sel(hinge_fit="PLAUSIBLE", hinge_field="PLAUSIBLE", d10_fit=0.05, ho_pref="prismatic")
    for d, code in ((hinge, "HINGE_ASSEMBLY_CONTRADICTED"), (axis, "FIT_FIELD_AXIS_DISAGREE"),
                    (pred, "PREDICTIVE_CONTRADICTION")):
        assert code in d.failure_codes and d.working_motion_model == "prismatic"
        assert d.physical_joint_status == "TYPE_UNOBSERVABLE"
    for kw in (dict(), dict(hinge_fit="PLAUSIBLE", hinge_field="PLAUSIBLE", d10_fit=0.05, axis_dis=True),
               dict(hinge_fit="PLAUSIBLE", hinge_field="PLAUSIBLE", d10_fit=0.05, ho_pref="prismatic")):
        assert sel(legacy=True, **kw).physical_joint_status == "MODEL_INCONSISTENT"        # the old early returns


def test_T12_genuine_rotation_with_a_plausible_hinge_stays_revolute_and_force_cannot_flip_it():
    d = sel(hinge_fit="PLAUSIBLE", hinge_field="PLAUSIBLE", d10_fit=0.05, ho_pref="revolute", sep="SEPARABLE")
    assert d.physical_joint_status == "IDENTIFIED_REVOLUTE"
    f = sel(hinge_fit="PLAUSIBLE", hinge_field="PLAUSIBLE", d10_fit=0.05, ho_pref="revolute", sep="SEPARABLE",
            force=dict(status="FAVOURS_PRISMATIC"))
    assert f.physical_joint_status == "IDENTIFIED_REVOLUTE" and "FORCE_DISAGREES" in f.failure_codes
    assert f.physics_joint_export_ready is False


def test_T13_neither_model_explains_the_motion_keeps_model_inconsistent_with_reasons():
    a = sel(sep="SEPARABLE", ho_pref="revolute")                       # real rotation, R better, but impossible hinge
    b = sel(inconsistent=True)                                         # two rigid groups
    for d in (a, b):
        assert d.physical_joint_status == "MODEL_INCONSISTENT" and d.working_motion_model == "none" and d.reasons
    assert "ROTATION_INCOMPATIBLE_WITH_SLIDE" in a.failure_codes
    c = sel(force=dict(status="FAVOURS_REVOLUTE"))                     # force favours a geometrically failed R
    assert c.physical_joint_status == "MODEL_INCONSISTENT"


def test_T15_no_declaration_input_anywhere_in_the_automatic_path():
    for fn in (select_physical_joint, FG.run, FG.reselect, FG.force_guided_evidence):
        assert not any("declar" in p for p in inspect.signature(fn).parameters)


def test_T17_working_branch_never_bypasses_the_export_gate():
    d = sel()
    assert d.physical_joint_status == "TYPE_UNOBSERVABLE" and d.physics_joint_export_ready is False
    assert d.working_model_is_approximation is True and d.selected_candidate is None


def test_remote_rule_is_object_relative():
    assert "REMOTE_HINGE_DEGENERACY" in sel(d10_fit=2.3, half=0.34).failure_codes
    assert "REMOTE_HINGE_DEGENERACY" not in sel(d10_fit=2.3, half=5.0).failure_codes


def test_T14_force_unavailable_leaves_the_visual_decision_intact():
    a, b = sel(), sel(force=dict(status="FORCE_UNAVAILABLE"))
    assert (a.physical_joint_status, a.working_motion_model) == (b.physical_joint_status, b.working_motion_model)
