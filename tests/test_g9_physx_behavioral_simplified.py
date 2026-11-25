"""G10 wrench path, simplified "easy first" model: gravity OFF and Coulomb friction OFF, inertia + viscous damping
fitted (full model = inertia + viscous damping + Coulomb friction + gravity). Contract checks need no engine; the
fits run on actual PhysX (CPU, and GPU when CUDA is present)."""
import copy
import importlib.util
import math
import subprocess

import numpy as np
import pytest

from forceartgs_g9.physx_refinement import behavioral_adapter as BA
from forceartgs_g9.physx_refinement import behavioral_fit as BF
from forceartgs_g9.physx_refinement.behavioral_contracts import (BehavioralContractError, load_config,
                                                                 load_wrench_config, wrench_to_fit_config)
from forceartgs_g9.physx_refinement.behavioral_real import starts_for
from forceartgs_g9.physx_refinement.behavioral_wrench_real import model_description
from forceartgs_g9.physx_refinement.sapien_diagnostic import IsolatedSapienCpu, default_engine_python
from forceartgs_g9.physx_refinement.urdf_builder import write_single_joint_urdf

from g10_behavioral_fixtures import synthetic_demo

SIMPLE = "configs/g9_real/physx_behavioral_inertia_damping.yaml"
FULL = "configs/g9_real/physx_behavioral_full_wrench.yaml"


def test_shipped_simplified_config_is_inertia_plus_damping_without_gravity_or_coulomb():
    cfg = load_wrench_config(SIMPLE)
    assert cfg["physics"]["gravity"] == "zero" and cfg["physics"]["coulomb_friction"] == "off"
    assert cfg["physics"]["engine"] == "sapien_gpu_physx"
    assert model_description(cfg) == ("inertia + viscous damping (OFF: Coulomb friction, gravity); "
                                      "fitted: rho_scale, viscous")
    b = BF.bounds_for(wrench_to_fit_config(cfg)["model"], 0.5, 2.0)
    assert b["dynamic_effort"] == (0.0, 0.0) and b["viscous"][1] > 0 and b["rho_scale"][1] > b["rho_scale"][0]
    assert load_wrench_config(SIMPLE, quick=True)["physics"]["gravity"] == "zero"


def test_default_and_full_config_stay_the_full_model():
    for src in (None, FULL):
        cfg = load_wrench_config(src)
        assert cfg["physics"]["gravity"] == "geometry" and cfg["physics"]["coulomb_friction"] == "fitted"
        assert model_description(cfg).endswith("(full model); fitted: rho_scale, viscous, dynamic_effort")
        assert BF.bounds_for(wrench_to_fit_config(cfg)["model"], 0.5, 2.0)["dynamic_effort"][1] > 0


@pytest.mark.parametrize("override,msg", [
    ({"physics": {"gravity": "zero"}}, "zero-gravity alternative IS the main fit"),
    ({"physics": {"coulomb_friction": "off"}}, "do not profile it"),
    ({"physics": {"coulomb_friction": False}, "fitting": {"profile_parameters": ["rho_scale"]}}, "quote it"),
    ({"physics": {"gravity": "tilted"}}, "physics.gravity geometry | zero"),
    ({"physics": {"friction_mode": "frictionless_physx"}}, "profile only rho_scale"),
])
def test_inconsistent_simplification_is_rejected(override, msg):
    with pytest.raises(BehavioralContractError, match=msg):
        load_wrench_config(override)


def test_frictionless_mode_removes_damping_too():
    cfg = load_wrench_config({"physics": {"friction_mode": "frictionless_physx", "gravity": "zero"},
                              "fitting": {"profile_parameters": ["rho_scale"]},
                              "diagnostics": {"zero_gravity_alternative": False}})
    b = BF.bounds_for(wrench_to_fit_config(cfg)["model"], 0.5, 2.0)
    assert b["viscous"] == (0.0, 0.0) and b["dynamic_effort"] == (0.0, 0.0)
    assert model_description(cfg).startswith("inertia (OFF: viscous damping, Coulomb friction, gravity)")


# ---------------------------------------------------------------- actual PhysX
HAVE_ENGINE = importlib.util.find_spec("sapien") is not None or default_engine_python() is not None
engine = pytest.mark.skipif(not HAVE_ENGINE, reason="no SAPIEN interpreter")
J_COM = np.array([[0.05, 0.01, 0.0], [0.01, 0.04, 0.002], [0.0, 0.002, 0.03]])
O, COM = np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.3, 0.4])
Z = [0.0, 0.0, 0.0]
CFG = load_config({"fitting": {"maxiter": 6, "popsize": 6, "polish": False, "local_max_nfev": 150, "local_starts": 2,
                               "profile_points": 3, "profile_inner_max_nfev": 25, "profile_parameters": ["rho_scale"],
                               "family_tolerance_sigma": 1.0}})
FC = copy.deepcopy(CFG)
FC["model"]["coulomb_friction"] = "off"


def P(rho, c=0.0, Td=0.0):
    return BF.params_from(dict(rho_scale=rho, armature=0.0, dynamic_effort=Td, static_excess=0.0, viscous=c))


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("g10_simplified")


def _urdf(tmp, name):
    t = math.radians(25.0)                                         # tilted hinge: gravity WOULD load it if switched on
    return write_single_joint_urdf(tmp / f"{name}.urdf", joint_type="revolute", axis=[math.sin(t), 0.0, math.cos(t)],
                                   origin=O, mass=2.0, com_ref=COM, inertia_com_ref=J_COM, q_ref=0.0)


def _fit_and_check(sim_fit, sim_truth, info):
    truth = P(1.5, 2.0, 0.0)
    rec, q_true = synthetic_demo(sim_truth, truth, q0=0.1, rows=150, noise=2e-4, seed=3,
                                 phases=[(10, 0.0, "absent"), (40, 3.0, "sliding"), (15, 0.02, "contact"),
                                         (85, 0.0, "absent")])
    rec.joint_type = "revolute"
    out = BA.build_full_demo_episode(rec, FC)
    assert out["status"] == "OK", out["reason"]
    ep = out["episode"]
    roll = BF.CachingRollout(sim_fit)
    b = BF.bounds_for(FC["model"], info["axis_inertia"], 3.0)
    assert b["dynamic_effort"] == (0.0, 0.0)
    fit = BF.fit_full_demo(roll, ep, FC, b, starts_for(b, info["axis_inertia"]))
    p = fit["_params"]
    assert fit["fitted_parameters"] == ["rho_scale", "viscous"] and p.dynamic_effort == 0.0 == p.static_effort
    assert fit["improvement"]["rmse_ratio"] < 0.2 and fit["best"]["metrics"]["rmse_over_sigma"] < 3.0
    assert abs(p.rho_scale / 1.5 - 1) < 0.1 and abs(p.viscous / 2.0 - 1) < 0.15
    return roll


@engine
def test_inertia_damping_fit_without_gravity_recovers_rho_and_c_on_cpu(tmp):
    info = _urdf(tmp, "cpu")
    sim = IsolatedSapienCpu(info["path"], moving_link="moving_link", joint_name="joint", dt=1e-3, gravity=Z,
                            friction_mode="external_effort_surrogate")
    try:
        _fit_and_check(sim, sim, info)
    finally:
        sim.close()


def _have_gpu():
    py = default_engine_python()
    if py is None:
        return False
    r = subprocess.run([py, "-c", "import torch; print(torch.cuda.is_available())"], capture_output=True, text=True)
    return r.stdout.strip() == "True"


@pytest.mark.skipif(not (HAVE_ENGINE and _have_gpu()), reason="no CUDA in the engine interpreter")
def test_inertia_damping_fit_without_gravity_recovers_rho_and_c_on_gpu(tmp):
    from forceartgs_g9.physx_refinement.sapien_diagnostic import IsolatedSapienGpu
    info = _urdf(tmp, "gpu")
    kw = dict(moving_link="moving_link", joint_name="joint", dt=1e-3, gravity=Z,
              friction_mode="external_effort_surrogate")
    g, c = IsolatedSapienGpu(info["path"], **kw), IsolatedSapienCpu(info["path"], **kw)
    try:
        roll = _fit_and_check(g, c, info)
        assert roll.n_batches > 0
    finally:
        g.close()
        c.close()
