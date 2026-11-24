"""CPU-only objectives tests; no PhysX/SAPIEN installation needed."""
import math
import numpy as np
import pytest
from forceartgs_g9.physx_refinement.objective import (
    Episode, PhysicalParameters, RefinementLoss, LossSettings,
    SimulationResult, effective_joint_inertia, rotational_viscous_to_usd_degree_units,
    normalized_free_dynamics,
)


def ep(mode="free", train=True, s=0.02):
    t = np.arange(0, 1.01, .1)
    # Positive initial velocity and passive damping; zero applied force does not mean no motion.
    q = (1-np.exp(-t))*0.4
    return Episode("train" if train else "heldout", t, q, np.full_like(t, s), q[0], .4,
                   np.array([0., .5]), np.array([0., 0.]), mode, "VERIFIED", train)


def par(scale=1.):
    return PhysicalParameters(scale,0.,0.04*scale,0.02*scale,0.1*scale)


def runner(params, episode):
    return SimulationResult(q=episode.q.copy(), backend="test", runtime_semantics="TEST")


def test_perfect_physx_trajectory_zero_loss():
    L = RefinementLoss(runner, [ep()]); r=L.evaluate(par())
    assert r["objective"] == 0
    assert r["absolute_scale_identified"] is False


def test_free_motion_cannot_identify_absolute_inertia_by_scale():
    # A mock solver that uses only passive ratios demonstrates the structural gauge.
    def free_ratios(p, e):
        I = .5 * p.rho_scale + p.armature
        lam,gam = normalized_free_dynamics(I,p.viscous,p.dynamic_effort)
        return SimulationResult(q=.4*(1-np.exp(-lam*e.t_s))/lam,backend="mock")
    e=ep()
    a=PhysicalParameters(1, 0, .08, .04, .2)
    b=PhysicalParameters(2, 0, .16, .08, .4)
    # Mock free trajectories under both parameter sets are identical.
    np.testing.assert_allclose(free_ratios(a,e).q,free_ratios(b,e).q)
    assert abs(RefinementLoss(free_ratios,[e])(a)-RefinementLoss(free_ratios,[e])(b))<1e-12


def test_forced_unverified_fails_closed_without_explicit_optin():
    e=ep()
    with pytest.raises(ValueError, match="opt-in"):
        Episode("x",e.t_s,e.q,e.sigma_q,e.q0,e.v0,e.input_t_s,np.array([1.,0.]),"forced","DECLARED")
    ok=Episode("x",e.t_s,e.q,e.sigma_q,e.q0,e.v0,e.input_t_s,np.array([1.,0.]),
               "forced","DECLARED",exploratory_assumptions_accepted=True)
    assert ok.calibration_status == "DECLARED"


def test_no_zero_inertia_and_friction_order():
    with pytest.raises(ValueError):
        PhysicalParameters(0,0,0,0,0)
    with pytest.raises(ValueError):
        PhysicalParameters(1,0,0.1,0.2,0)


def test_heldout_not_optimized():
    e0,e1=ep(),ep(train=False)
    def sim(p,e):
        return SimulationResult(e.q+(0 if e.used_for_training else p.rho_scale))
    r=RefinementLoss(sim,[e0,e1]).evaluate(par())
    assert r["objective"] == 0
    assert r["heldout_position"]>0


def test_revolute_axis_parallel_theorem_and_gravity_independent_of_armature():
    I=effective_joint_inertia(joint_type="revolute",axis=[0,0,1],hinge_origin=[0,0,0],com=[1,0,0],
                              unit_mass=2,unit_inertia_com=np.eye(3)*.3,rho_scale=2,armature=.1)
    assert I == pytest.approx(4.7)
    P=effective_joint_inertia(joint_type="prismatic",axis=[0,0,1],hinge_origin=[0,0,0],com=[1,0,0],
                              unit_mass=2,unit_inertia_com=np.eye(3),rho_scale=2,armature=.1)
    assert P == pytest.approx(4.1)


def test_usd_rad_to_degree_viscous():
    assert rotational_viscous_to_usd_degree_units(180/math.pi) == pytest.approx(1)
