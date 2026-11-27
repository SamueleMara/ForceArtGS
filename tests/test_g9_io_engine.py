"""Input-output identification WITH the actual engine (SAPIEN 3 / PhysX CPU, forceartgs env): T8 clean room
(revolute + prismatic, every readback), T9 scale family / armature degeneracy / capability coverage, and the
analytical-to-PhysX reconciliation end to end. Skipped without a SAPIEN interpreter. CPU only; native axis friction,
static breakaway and Isaac are NOT covered (not exposed / not installed)."""
import importlib.util
import math

import numpy as np
import pytest

from forceartgs_g9.physx_refinement.objective import Episode, PhysicalParameters
from forceartgs_g9.physx_refinement.sapien_diagnostic import IsolatedSapienCpu, default_engine_python
from forceartgs_g9.physx_refinement.urdf_builder import write_single_joint_urdf
from forceartgs_g9.system_identification import physical_response as PR
from forceartgs_g9.system_identification.physx_bridge import g10_parameters

HAVE_ENGINE = importlib.util.find_spec("sapien") is not None or default_engine_python() is not None
pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="no SAPIEN interpreter")

G = [0.0, 0.0, -9.81]
J_COM = np.array([[0.05, 0.01, 0.0], [0.01, 0.04, 0.002], [0.0, 0.002, 0.03]])
O = np.array([0.1, 0.2, 0.3])
COM = np.array([0.5, 0.3, 0.4])
DT = 1e-3


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("io_engine")


def make(tmp, name, *, axis, jt, mode, mass=2.0, q_ref=0.0):
    info = write_single_joint_urdf(tmp / f"{name}.urdf", joint_type=jt, axis=axis, origin=O, mass=mass, com_ref=COM,
                                   inertia_com_ref=J_COM, q_ref=q_ref)
    return info, IsolatedSapienCpu(info["path"], moving_link="moving_link", joint_name="joint", dt=DT, gravity=G,
                                   friction_mode=mode)


def free_ep(t, q0=0.0, v0=0.7, name="free"):
    return Episode(name, t, np.zeros_like(t), np.full_like(t, 1e-3), q0, v0, np.array([t[0]]), np.array([0.0]),
                   "free")


def forced_ep(t, ft, fq, q0=0.0, v0=0.0, name="forced"):
    return Episode(name, t, np.zeros_like(t), np.full_like(t, 1e-3), q0, v0, ft, fq, "forced", "VERIFIED")


P0 = PhysicalParameters(1.0, 0.0, 0.0, 0.0, 0.0)


# ------------------------------------------------------------------------------------------------- capabilities
def test_engine_capabilities_no_native_axis_friction(tmp):
    _, sim = make(tmp, "cap", axis=(0, 0, 1), jt="revolute", mode="frictionless_physx")
    try:
        cap = sim.capabilities()
    finally:
        sim.close()
    assert cap["joint_api"]["set_friction"] and cap["joint_api"]["set_armature"]
    assert cap["native_axis_friction_api"] is False
    assert "NOT_EXPOSED" in cap["friction_coverage"]
    assert "NOT a Coulomb effort" in cap["legacy_joint_friction_semantics"]


# ------------------------------------------------------------------------------------------------- T8
@pytest.mark.parametrize("jt,axis", [("revolute", (0, 0, 1)), ("prismatic", (1, 0, 0))])
def test_clean_room_constant_velocity_and_readbacks(tmp, jt, axis):
    _, sim = make(tmp, f"clean_{jt}", axis=axis, jt=jt, mode="frictionless_physx")
    try:
        t = np.arange(0, 3.0, 0.05)
        r = sim(P0, free_ep(t))
    finally:
        sim.close()
    d = r.diagnostics
    rb = d["readback"]
    np.testing.assert_allclose(r.qdot, 0.7, rtol=2e-6)                        # no hidden dissipation / drive
    np.testing.assert_allclose(r.q, 0.7 * t, atol=3e-4 * max(1.0, 0.7 * t[-1]))
    for k in ("joint_friction", "joint_damping", "joint_stiffness", "sleep_threshold"):
        assert rb[k] == 0.0, k
    assert all(v == 0.0 for v in rb["link_linear_damping"].values())
    assert all(v == 0.0 for v in rb["link_angular_damping"].values())
    assert rb["drive_target"] == [0.0] and rb["drive_velocity_target"] == [0.0]
    assert rb["timestep"] == pytest.approx(DT) and rb["gravity"] == pytest.approx(G)
    assert rb["moving_inertia_spd"] and d["n_contacts"] == 0 and d["sleeping_at_end"] is False
    # the loader's defaults were really there (and were zeroed, not assumed absent)
    assert d["engine_defaults_as_loaded"]["joint_friction"] > 0


def test_prismatic_known_impulse_direction_and_magnitude(tmp):
    _, sim = make(tmp, "pulse_p", axis=(1, 0, 0), jt="prismatic", mode="frictionless_physx", mass=2.0)
    try:
        t = np.arange(0, 1.0, 0.02)
        ft = np.array([0.0, 0.2, 0.3])
        for s in (1.0, -1.0):
            r = sim(P0, forced_ep(t, ft, np.array([0.0, 4.0 * s, 0.0])))
            dv = r.qdot[-1] - r.qdot[0]
            assert dv == pytest.approx(s * 4.0 * 0.1 / 2.0, rel=1e-4)              # J / m, predicted sign
    finally:
        sim.close()


def test_tilted_hinge_gravity_stiffness_matches_engine(tmp):
    """Engine gravity torque at q +- h (one step from rest) -> G'(q) agrees with the analytic stiffness."""
    a = np.array([math.sin(math.radians(30)), 0.0, math.cos(math.radians(30))])
    info, sim = make(tmp, "tilt", axis=a, jt="revolute", mode="frictionless_physx")
    rbg = PR.rigid_gravity("revolute", a, O, COM, 2.0, G, 0.0)
    I = info["axis_inertia"]
    t = np.array([0.0, DT, 2 * DT])                                            # an episode needs >= 3 samples

    def G_engine(q):
        r = sim(P0, free_ep(t, q0=q, v0=0.0))
        return -I * r.qdot[1] / DT                                             # load G = -I a (first step)
    try:
        for q in (-0.6, 0.4, 1.3):
            h = 0.02
            Ge, (Ga, dGa) = G_engine(q), PR.gravity_load(rbg, q)
            assert Ge == pytest.approx(Ga, rel=2e-3, abs=1e-4)
            dGe = (G_engine(q + h) - G_engine(q - h)) / (2 * h)
            assert dGe == pytest.approx(dGa, rel=5e-3, abs=2e-3)
    finally:
        sim.close()


# ------------------------------------------------------------------------------------------------- T9
def test_free_family_identical_and_known_force_breaks_it(tmp):
    info, sim = make(tmp, "fam", axis=(1, 0, 0), jt="prismatic", mode="external_effort_surrogate", mass=1.0)
    I1 = info["axis_inertia"]
    lam, gam = 0.6, 0.4
    t = np.arange(0, 1.2, 0.03)
    try:
        free, forced = [], []
        for M in (0.5, 2.0, 6.0):
            p = g10_parameters(M, lam, gam, I1)
            assert sim.effective_inertia(p) == pytest.approx(M)
            free.append(sim(p, free_ep(t, v0=1.0)).q)
            forced.append(sim(p, forced_ep(t, np.array([0.0]), np.array([3.0]), v0=0.2)).q)
            assert sim(p, free_ep(t[:3])).diagnostics["readback"]["moving_inertia_spd"]
    finally:
        sim.close()
    for q in free[1:]:
        np.testing.assert_allclose(q, free[0], atol=2e-5)                      # scale gauge in the engine
    assert np.max(np.abs(forced[0] - forced[1])) > 0.05                        # measured input breaks it


def test_link_inertia_and_armature_enter_only_through_their_sum(tmp):
    t = np.arange(0, 1.0, 0.02)
    e = forced_ep(t, np.array([0.0, 0.3]), np.array([1.5, -0.5]), v0=0.1)
    out = {}
    for name, axis in (("vertical", (0, 0, 1)), ("tilted", (math.sin(0.5), 0, math.cos(0.5)))):
        info, sim = make(tmp, f"arm_{name}", axis=axis, jt="revolute", mode="frictionless_physx")
        I1 = info["axis_inertia"]
        try:
            a = sim(PhysicalParameters(1.0, 0.0, 0.0, 0.0, 0.0), e).q
            b = sim(PhysicalParameters(0.5, 0.5 * I1, 0.0, 0.0, 0.0), e).q
        finally:
            sim.close()
        out[name] = float(np.max(np.abs(a - b)))
    assert out["vertical"] < 2e-5            # no gravity: link inertia vs armature indistinguishable
    assert out["tilted"] > 1e-3              # gravity scales with the link only: separable in principle


# ------------------------------------------------------------------------------------------------- reconciliation
def test_io_identification_reconciles_with_physx(tmp):
    from forceartgs_g9.free_forced.normalized import GravityModel
    from forceartgs_g9.system_identification import synthetic_io as S
    from forceartgs_g9.system_identification.api import identify_io_response, reconcile_physx
    from forceartgs_g9.system_identification.contracts import load_config
    M, c, Fc = 2.0, 1.2, 0.8
    segs = [S.segment(f"f{k}", label="FORCED_MEASURED", joint_type="prismatic", M=M, c=c, Fc=Fc, v0=0.3 + 0.2 * k,
                      T=1.5, effort_levels=S.pulse_train([4.0, 1.5, 3.0][k:] + [2.0], 0.3), seed=20 + k,
                      t_start=3 * k, q_noise=2e-3, effort_sigma=0.03) for k in range(2)]
    eps = S.episodes(segs)
    cfg = load_config({"fit": {"compare_methods": ["output_error"], "profile_points": 9, "n_starts": 2,
                               "feature_bootstrap_seeds": 2, "energy_mc": 40}, "physx": {"reconcile_family_members": 2}})
    flat = GravityModel.prismatic_ideal((1, 0, 0), G)
    res = identify_io_response(eps, dict(joint_type="prismatic", kinematic_status="IDENTIFIED", export_ready=True),
                               dict(force_audit_status="VERIFIED"), dict(primary=flat), cfg, None,
                               log=lambda *a: None)
    gain = next(f for f in res["features"] if f["name"] == "gain")
    assert gain["status"] == "IDENTIFIED" and gain["uncertainty"]["ci95_M"][0] <= M <= gain["uncertainty"]["ci95_M"][1]
    info, sim = make(tmp, "recon", axis=(1, 0, 0), jt="prismatic", mode="external_effort_surrogate", mass=1.0)
    try:
        rec, physx_q = reconcile_physx(res, sim, eps, axis_inertia_unit=info["axis_inertia"], cfg=cfg,
                                       log=lambda *a: None)
    finally:
        sim.close()
    assert rec["status"] == "DIAGNOSTIC_ONLY_NOT_IDENTIFIED" and rec["parity"] == "PASS"
    assert rec["engine_capabilities"]["native_axis_friction_api"] is False
    best = rec["candidates"]["M_from_gain"]
    for name, row in best["episodes"].items():
        assert row["status"] == "OK" and row["parity_ok"]
        assert row["readback"]["n_contacts"] == 0
    for k in ("lambda", "gain"):
        assert best["feature_errors"][k]["within_io_ci95"] is True
    assert rec["physx_sensitivity"]["status"] == "OK"
    assert set(physx_q) == {e.name for e in eps}
