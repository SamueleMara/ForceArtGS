"""Synthetic R/P cases for the force-guided selection tests (3D force + kinematics only; truth motion from the
exact two-term solution, used only to generate realistic q(t))."""
from __future__ import annotations

import numpy as np

from estimotion_2part.joint_discovery.observations import FrameMap, TrackObservations
from forceartgs_g9.estimators.rest_to_rest_convolution import simulate_two_term

ROW_HZ = 50.0


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _rot(axis, th):
    a = _unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def case(true="prismatic", *, remote_R_m=None, M=2.0, c=8.0, T=60, noise=1e-3, crop=0, seed=0, push=1.5):
    """Visual frames 0..T-1 at 25 Hz (source frame = 2 x rgbd), robot rows at 50 Hz; rest 0.48 s, push 0.6 s, then
    a viscous coast (no brake needed for these channel tests). Returns (cached, eff, truth)."""
    rng = np.random.default_rng(seed)
    n_rows = 2 * T + 2
    rows = np.arange(n_rows)
    ft = rows / ROW_HZ
    tv = (2 * np.arange(T)) / ROW_HZ
    level = push * M
    Q = np.where((ft >= 0.48) & (ft < 1.08), level, 0.0)
    q = simulate_two_term(ft, Q, 0.0, 0.0, M, c, tv)["q"]
    a_slide = np.array([1.0, 0.0, 0.0])
    o_hinge = np.array([0.0, -0.4, 0.0])
    a_hinge = np.array([0.0, 0.0, 1.0])
    X0 = np.array([0.0, 0.0, 0.0]) + rng.uniform(-0.05, 0.05, (40, 3))
    tcp0 = np.array([0.0, 0.02, 0.0])
    if true == "prismatic":
        xs = X0[None] + q[:, None, None] * a_slide
        p_tcp = tcp0 + np.outer(np.interp(ft, tv, q), a_slide)
        F = np.outer(Q, a_slide)
    else:
        xs = np.stack([(X0 - o_hinge) @ _rot(a_hinge, qq).T + o_hinge for qq in q])
        qr = np.interp(ft, tv, q)
        p_tcp = np.stack([_rot(a_hinge, qq) @ (tcp0 - o_hinge) + o_hinge for qq in qr])
        r = p_tcp - o_hinge
        tang = np.cross(a_hinge, r)
        tang /= np.linalg.norm(tang, axis=1)[:, None]
        F = (Q / np.linalg.norm(r - (r @ a_hinge)[:, None] * a_hinge, axis=1))[:, None] * tang
    static = rng.uniform(-0.3, 0.3, (20, 3)) + np.array([0.0, 0.5, 0.0])
    # candidates: the fitted geometry of each hypothesis
    if true == "prismatic":
        cP = dict(kind="prismatic", axis=a_slide, origin=None, q=q.copy())
        rho = remote_R_m or 0.4
        oR = tcp0 + np.array([0.0, rho, 0.0])
        cR = dict(kind="revolute", axis=np.array([0.0, 0.0, 1.0]), origin=oR, q=q / rho)   # z x (-rho y) = rho x
    else:
        rad = np.linalg.norm((tcp0 - o_hinge)[:2])
        chord = (xs[-1] - xs[0]).mean(0)
        cP = dict(kind="prismatic", axis=_unit(chord), origin=None, q=q * rad)
        cR = dict(kind="revolute", axis=a_hinge, origin=o_hinge, q=q.copy())
    sl = slice(crop, T)
    obs = _obs(xs[sl], static, tv[sl], noise, rng)
    T2 = len(tv[sl])
    cands = {k: dict(v, q=np.asarray(v["q"])[sl]) for k, v in (("prismatic", cP), ("revolute", cR))}
    blocks = [list(range(s, min(s + 3, T2))) for s in range(0, T2, 3)]
    held = [False] * len(blocks)
    for k in range(2, len(blocks) - 1, 3):
        held[k] = True
    part = dict(blocks=blocks, heldout=held, heldout_times=[t for b, h in zip(blocks, held) if h for t in b],
                train_times=[t for b, h in zip(blocks, held) if not h for t in b], reference_time_index=0)
    M_prop = np.r_[np.ones(40, bool), np.zeros(20, bool)]
    sel = dict(evidence=dict(assembly_geometry=dict(hinge_fit=dict(object_half_diag_m=0.15))))
    cached = dict(obs=obs, cands=cands, partition=part, M_proposal=M_prop, S_core=~M_prop, selection=sel)
    fr0 = 2 * crop
    eff = dict(t_s=ft[fr0:], source_row=rows[fr0:], F_B=F[fr0:], p_tcp=p_tcp[fr0:])
    return cached, eff, dict(M=M, c=c, true=true, q=q)


def _obs(xs, static, tv, noise, rng):
    T, Nm, _ = xs.shape
    Ns = len(static)
    trk, tim, xw = [], [], []
    for i in range(Nm):
        trk += [i] * T
        tim += list(range(T))
        xw.append(xs[:, i] + rng.standard_normal((T, 3)) * noise)
    for j in range(Ns):
        trk += [Nm + j] * T
        tim += list(range(T))
        xw.append(np.tile(static[j], (T, 1)) + rng.standard_normal((T, 3)) * noise)
    xw = np.concatenate(xw)
    M = len(trk)
    src = np.round(tv * ROW_HZ).astype(int)
    fm = FrameMap(np.arange(T), src, src / ROW_HZ, provenance="synthetic")
    return TrackObservations(["cam_001"], fm, cam=np.zeros(M, int), track=np.array(trk), time=np.array(tim),
                             uv=np.zeros((M, 2)), uv_conf=np.ones(M), uv_valid=np.ones(M, bool), z=np.ones(M),
                             depth_valid=np.ones(M, bool), depth_edge=np.zeros(M, bool), mask_valid=np.ones(M, bool),
                             x_w=xw, track_camera=np.zeros(Nm + Ns, int), track_local_id=np.arange(Nm + Ns),
                             track_query_time=np.zeros(Nm + Ns, int))


def selector_inputs(*, rotation=True, hinge_fit="CONTRADICTED", hinge_field="WEAK", sep="INDISTINGUISHABLE",
                    ho_pref="none", inconsistent=False, d10_fit=2.3, half=0.34, axis_dis=False):
    """Minimal evidence dicts for ``select_physical_joint`` (the dog-like remote-hinge case by default)."""
    rot = dict(rotation_detected=rotation, net_rotation_deg=46.9 if rotation else 0.4,
               net_rotation_ci95_deg=[15.9, 82.5] if rotation else [-0.5, 1.1],
               net_centroid_translation_norm_m=0.062, rotation_upper_bound_deg=82.5 if rotation else 1.1,
               path_length_m=0.066, per_camera={})
    shared = dict(axis=[0, 0, 1], axis_identifiable=rotation, axis_cone95_deg=16.8)
    hinge = dict(status="OK", line_sd_m=0.037, axis=[0, 0, 1], origin=[0, 0, 0])
    null = dict(net_abs_rotation_q95_deg=13.5, net_translation_q95_m=0.004)
    fits = dict(revolute=dict(axis=[0, 0, 1] if not axis_dis else [1, 0, 0], origin=[2.3, 0, 0]),
                prismatic=dict(axis=[1, 0, 0], origin=None))
    att = dict(interface=dict(status="OK"),
               hinge_field=dict(status="OK", verdict=hinge_field, d10_to_interface_m=0.06, object_half_diag_m=half),
               hinge_fit=dict(status="OK", verdict=hinge_fit, d10_to_interface_m=d10_fit, object_half_diag_m=half),
               guide=dict(status="ABSTAIN", verdict="NO_OBSERVABLE_GUIDE"),
               implied=dict(status="OK", implied_min_rotation_deg=13.7))
    return dict(motion_evidence=dict(rotation=rot, shared_axis=shared, hinge=hinge, null=null,
                                     support_quality=dict(moving_over_static_residual=1.2)),
                model_fits=fits,
                common_support=dict(counts=dict(M_proposal=1000),
                                    consistency=dict(inconsistent=inconsistent,
                                                     inconsistency_flags=["two rigid groups"] if inconsistent else [])),
                attachment_evidence=att,
                predictive_evidence=dict(separation=dict(status=sep, median_block_separation_over_sigma=1.5),
                                         heldout=dict(status="OK", preference=ho_pref)))


def force_source_from(cached, eff, sigma_F=0.01):
    """The force_source() dict on the visual frames: exact source-row join of the synthetic 50 Hz rows."""
    from estimotion_2part.joint_discovery.tcp_contact import tcp_velocity
    src = np.asarray(cached["obs"].frame_map.source_frame_index, int)
    pos = {int(r): i for i, r in enumerate(eff["source_row"])}
    sel = np.array([pos[int(s_)] for s_ in src])
    t = np.asarray(cached["obs"].frame_map.timestamp_s, float)
    p = eff["p_tcp"][sel]
    return dict(F=eff["F_B"][sel], p_tcp=p, v_tcp=tcp_velocity(p, t), t=t, source_row=src, sigma_F_N=sigma_F)
