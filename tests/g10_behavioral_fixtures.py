"""Test fixtures for the G10-A behavioral replay.

``TestDoubleEngine`` is a TEST DOUBLE of the SAPIEN worker's step law (semi-implicit Euler: v += a dt, q += v dt; the
same explicit friction surrogate -c v - Td sat(v / v_eps), v_eps = Td dt / I; forward-ZOH input on integer steps). It
labels itself ``TEST_DOUBLE_NOT_PHYSX`` and lives only under tests/: production scoring always uses the real engine
(``IsolatedSapienCpu``). The engine tests in test_g9_physx_behavioral_engine.py use actual PhysX.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from forceartgs_g9.physx_refinement.objective import SimulationResult


class TestDoubleEngine:
    """Prismatic / revolute 1-DOF with I(rho) = rho I1 + armature and a constant gravity load G1 rho (per unit
    density). Returns q, qdot at the query times + cumulative impulses, like the worker."""
    __test__ = False

    def __init__(self, I1=1.0, G1=0.0, dt=1e-3):
        self.axis = dict(axis_inertia=I1, joint_type="prismatic")
        self.I1, self.G1, self.dt = float(I1), float(G1), float(dt)
        self.n_rollouts = 0
        self.requests = []

    def effective_inertia(self, p):
        return p.rho_scale * self.I1 + p.armature

    def __call__(self, p, ep):
        self.n_rollouts += 1
        self.requests.append(dict(q0=ep.q0, v0=ep.v0, t0=float(ep.t_s[0]), n=len(ep.t_s)))
        I = self.effective_inertia(p)
        steps = np.rint((ep.t_s - ep.t_s[0]) / self.dt).astype(int)
        q, v = np.float32(ep.q0), np.float32(ep.v0)
        tf, lev = ep.input_t_s, ep.input_effort
        v_eps = p.dynamic_effort * self.dt / I if p.dynamic_effort > 0 else 0.0
        out_q, out_v, Je, Jf = (np.empty(len(steps)) for _ in range(4))
        je = jf = 0.0
        k = 0
        for n in range(int(steps[-1]) + 1):
            while k < len(steps) and steps[k] == n:
                out_q[k], out_v[k], Je[k], Jf[k] = float(q), float(v), je, jf
                k += 1
            if n == steps[-1]:
                break
            t = ep.t_s[0] + n * self.dt
            Qe = float(lev[max(int(np.searchsorted(tf, t + 1e-12, side="right") - 1), 0)])
            Qf = -p.viscous * float(v)
            if p.dynamic_effort > 0:
                Qf -= p.dynamic_effort * max(-1.0, min(1.0, float(v) / v_eps))
            a = (Qe + Qf - self.G1 * p.rho_scale) / I
            v = np.float32(float(v) + a * self.dt)
            q = np.float32(float(q) + float(v) * self.dt)
            je += Qe * self.dt
            jf += Qf * self.dt
        return SimulationResult(out_q, out_v, backend="test_double", engine_version="test",
                                runtime_semantics="TEST_DOUBLE_NOT_PHYSX",
                                diagnostics=dict(external_impulse=Je.tolist(), friction_impulse=Jf.tolist(),
                                                 readback=dict(), engine_defaults_as_loaded=dict()))

    def capabilities(self):
        return dict(native_axis_friction_api=False, backend="TEST_DOUBLE_NOT_PHYSX")


def record(*, q_frames, q, q_sigma=2e-4, force_rows=None, Q=None, Q_sigma=0.02, modes=None, scene="synthetic",
           joint_type="prismatic"):
    """A JointInteractionRecord-shaped namespace on the canonical clock (t = frame / 50)."""
    q_frames = np.asarray(q_frames, int)
    force_rows = np.arange(int(q_frames[-1]) + 1) if force_rows is None else np.asarray(force_rows, int)
    Q = np.zeros(len(force_rows)) if Q is None else np.asarray(Q, float)
    modes = np.full(len(force_rows), "absent", dtype="U16") if modes is None else np.asarray(modes, dtype="U16")
    qs = np.broadcast_to(np.asarray(q_sigma, float), np.shape(q)).astype(float)

    def sha():
        h = hashlib.sha256()
        for a in (force_rows, Q, q_frames, q):
            h.update(np.ascontiguousarray(a, float).tobytes())
        return h.hexdigest()
    return SimpleNamespace(scene=scene, joint_type=joint_type, force_t_s=force_rows / 50.0, source_force_row=force_rows,
                           Q_object=Q, Q_sigma=np.full(len(force_rows), float(Q_sigma)), contact_mode=modes,
                           q_t_s=q_frames / 50.0, source_q_frame=q_frames, q=np.asarray(q, float), q_sigma=qs,
                           sha256=sha, force_calibration_assumptions={}, provenance=dict(object_side_multiplier=-1.0))


def push_profile(n_rows, phases):
    """phases: [(n_rows, level, mode)] -> (Q, modes) on n_rows 50 Hz rows (remaining rows: 0, 'absent')."""
    Q, modes = np.zeros(n_rows), np.full(n_rows, "absent", dtype="U16")
    i = 0
    for n, lev, m in phases:
        Q[i:i + n], modes[i:i + n] = lev, m
        i += n
    return Q, modes


def synthetic_demo(engine, p, *, q0=0.15, rows=300, q_every=4, phases=None, noise=2e-4, seed=0, Q_scale=1.0):
    """Rest -> push -> low-effort contact -> release -> coast -> stop, simulated ONCE from (q0, 0) by ``engine``;
    q at 12.5 Hz (every 4 rows), effort rows at 50 Hz. Returns (record, truth q)."""
    from forceartgs_g9.physx_refinement.objective import Episode
    phases = phases or [(10, 0.0, "absent"), (60, 2.5, "sliding"), (30, 0.02, "contact"), (200, 0.0, "absent")]
    Q, modes = push_profile(rows, phases)
    frames = np.arange(0, rows - 1, q_every)
    t = frames / 50.0
    ft = np.arange(rows) / 50.0
    e = Episode("truth", t, np.zeros_like(t), np.full_like(t, 1.0), q0, 0.0, ft, Q, "forced", "VERIFIED")
    q_true = np.asarray(engine(p, e).q, float)
    rng = np.random.default_rng(seed)
    q_obs = q_true + rng.standard_normal(len(t)) * noise
    q_obs[0] = q0                                            # the authority defines q(t0) (gauge sample)
    return record(q_frames=frames, q=q_obs, q_sigma=noise, force_rows=np.arange(rows), Q=Q * Q_scale,
                  modes=modes), q_true


def wrench_record(engine, p, *, q0=0.15, rows=300, q_every=4, phases=None, noise=2e-4, seed=0, Q_scale=1.0,
                  nan_q=None):
    """A FullWrenchRecord (prismatic, the TestDoubleEngine's joint) whose Q_full is the designed effort profile;
    q is simulated ONCE from (q0, 0). ``nan_q``: index of a visual sample made invalid. Returns (fr, rec, q_true,
    segments) with the interaction = rows 0 .. last non-absent phase row (contact off after it: zero effort)."""
    from forceartgs_g9.wrench_input.full_record import FullWrenchRecord
    rec, q_true = synthetic_demo(engine, p, q0=q0, rows=rows, q_every=q_every, phases=phases, noise=noise, seed=seed,
                                 Q_scale=Q_scale)
    n = len(rec.Q_object)
    a = np.array([1.0, 0.0, 0.0])
    F = np.outer(rec.Q_object, a)
    q = rec.q.copy()
    if nan_q is not None:
        q[nan_q] = np.nan
    contract = dict(wrench_order=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], known_transform="test identity",
                    transform_source="test", moment_reference="ee_frame_origin", object_side_multiplier=-1.0,
                    robot_archive_sha256="test", wrench_frame="sensor axes = EE axes")
    fr = FullWrenchRecord(scene="synthetic", joint_type="prismatic", axis_B=a, origin_B=None, t_s=rec.force_t_s,
                          source_row=rec.source_force_row, W_recorded=np.c_[-F, np.zeros((n, 3))],
                          R_BS=np.repeat(np.eye(3)[None], n, 0), p_ref_B=np.zeros((n, 3)), F_B=F,
                          tau_B=np.zeros((n, 3)), Q_full=rec.Q_object.copy(), Q_sigma=np.full(n, 0.02),
                          q_t_s=rec.q_t_s, source_q_frame=rec.source_q_frame, q=q, q_sigma=rec.q_sigma,
                          contract=contract, Q_legacy_tcp_force_only=rec.Q_object.copy(),
                          noise=dict(force_sigma_N=[0.02] * 3, moment_sigma_Nm=[1e-3] * 3, force_sigma_source="test",
                                     moment_sigma_source="test"))
    on = np.flatnonzero(np.asarray(rec.contact_mode) != "absent")
    segs = [(0, int(on[-1]))]
    fr.Q_full = np.where(np.arange(n) <= on[-1], fr.Q_full, fr.Q_full + 0.05)   # recorded noise after release (ignored)
    return fr, rec, q_true, segs
