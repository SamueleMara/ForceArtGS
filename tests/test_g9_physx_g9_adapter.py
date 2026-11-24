"""Strict G9Interaction adapter tests without requiring a full ForceArtGS checkout."""
from types import SimpleNamespace
import numpy as np
import pytest
from forceartgs_g9.physx_refinement.g9_adapter import (episode_from_g9_interaction,
                                                        write_g9_physx_episodes)
from forceartgs_g9.physx_refinement.optimizer import FitConfiguration, fit_physx
from forceartgs_g9.physx_refinement.objective import RefinementLoss, SimulationResult, PhysicalParameters


def interaction(name='fridge'):
    t=np.linspace(0.0,0.4,11)
    return SimpleNamespace(name=name, t_obs=t, q_obs=0.3*t, q_std=np.full(len(t),.01),
                           effort=SimpleNamespace(t_s=np.linspace(0,.4,21),levels=np.ones(21)*2.),
                           initial_state={'qdot':0.3})


def test_free_motion_requires_independent_contact_absence():
    it=interaction()
    with pytest.raises(ValueError,match='independent absence'):
        episode_from_g9_interaction(it,mode='free',v0=.3,regime_evidence={'source':'robot','contact_independently_absent':False})
    e=episode_from_g9_interaction(it,mode='free',v0=.3,regime_evidence={'source':'operator','contact_independently_absent':True})
    assert e.mode=='free' and (e.input_effort==0.).all()
    assert (it.effort.levels==2.).all() # original canonical force not modified


def test_forced_requires_contact_and_explicit_calibration_optin():
    it=interaction()
    with pytest.raises(ValueError,match='confirmed'):
        episode_from_g9_interaction(it,mode='forced',v0=.3,regime_evidence={'source':'operator'})
    with pytest.raises(ValueError,match='opt-in'):
        episode_from_g9_interaction(it,mode='forced',v0=.3,regime_evidence={'source':'operator','contact_confirmed':True})
    e=episode_from_g9_interaction(it,mode='forced',v0=.3,
                                  regime_evidence={'source':'operator','contact_confirmed':True},
                                  calibration_status='EXPLORATORY_ASSUMPTIONS_ACCEPTED',
                                  exploratory_assumptions_accepted=True)
    np.testing.assert_allclose(e.input_effort,2.)
    assert e.t_s.shape[0]==11 and e.input_t_s.shape[0]==21


def test_g9_initial_state_velocity_cannot_be_redefined():
    with pytest.raises(ValueError,match='disagrees'):
        episode_from_g9_interaction(interaction(),mode='free',v0=1.,
                    regime_evidence={'source':'independent','contact_independently_absent':True})


def test_episodes_serialization_and_provenance(tmp_path):
    e=episode_from_g9_interaction(interaction(),mode='free',v0=.3,
                 regime_evidence={'source':'operator','contact_independently_absent':True})
    p=write_g9_physx_episodes([e],{'fridge':{'source':'operator','contact_independently_absent':True}},tmp_path/'episodes.json')
    import json
    d=json.loads(p.read_text())
    assert d['schema']=='forceartgs.g9.physx_episodes/v1'
    assert d['episodes'][0]['q']==pytest.approx(e.q.tolist())


def test_free_only_fitting_requires_fixed_mass_scale_and_armature():
    e=episode_from_g9_interaction(interaction(),mode='free',v0=.3,
                 regime_evidence={'source':'operator','contact_independently_absent':True})
    obj=RefinementLoss(lambda p,e: SimulationResult(q=e.q),[e])
    b={'rho_scale':(.5,2.),'armature':(0.,0.),'dynamic_effort':(0.,1.),
       'static_excess':(0.,0.),'viscous':(0.,1.)}
    with pytest.raises(ValueError,match='fixed rho_scale'):
        fit_physx(obj,FitConfiguration(bounds=b,maxiter=1,popsize=3,polish=False))
