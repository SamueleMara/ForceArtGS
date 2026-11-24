"""Gravity-projection / mass-geometry / reference-pose audit (G9 gravity-projection corrections spec, sections 3-7).

ONE gravity law for every estimator (ModelSpec, TorchDynamics, integral LS, rollout, EKF):

    U(q) = -m g.c(q),  G(q) = dU/dq  (the generalized LOAD; the applied gravity effort is -G)
    revolute : c(q) = o + R_a(q - q_ref)(c_ref - o),  G = -rho m1 g.(a x R_a(q - q_ref) r_perp),
               max_q |G| = rho m1 ||g x a|| ||r_perp||
    prismatic: c(q) = c_ref + (q - q_ref) a,          G = -rho m1 g.a
    M = rho I_axis,1 + I_reflected (reflected inertia is kinetic only, never gravitational mass)

Everything here is read-only on the inputs: it never changes the accepted joint, the force sign, the q gauge or the
frozen reconstruction. Alternative kinematics (e.g. an exactly vertical hinge) are LABELLED counterfactuals, never
substitutions. Outputs (dynamics/g9/vNNN/):
    reference_pose_alignment.json, mass_com_inertia_audit.json, gravity_geometry_audit.json,
    gravity_joint_sensitivity.json, method_gravity_parity.json, impulse_balance_audit.json
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..dynamics.general_1dof import RigidBodyGravity, rotation
from ..dynamics.mass_properties import DENSITY_UNITS, TriMesh, topology, unit_density_properties

TENTATIVE_FLAGS = ["TENTATIVE_GEOMETRIC_PREVIEW", "PHYSICAL_MASS_UNVERIFIED"]
Z = np.array([0.0, 0.0, 1.0])


class ReferencePoseError(ValueError):
    pass


def _sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def _unit(a):
    a = np.asarray(a, float)
    n = np.linalg.norm(a)
    if a.shape != (3,) or not np.isfinite(n) or n < 1e-12:
        raise ReferencePoseError(f"degenerate axis {a}")
    return a / n


def _perp_basis(a):
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def validate_transform(T, name="T") -> np.ndarray:
    """4x4 proper rigid transform (finite, orthonormal rotation, det +1, last row 0 0 0 1); raises otherwise."""
    T = np.asarray(T, float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ReferencePoseError(f"{name}: not a finite 4x4 matrix")
    R = T[:3, :3]
    if np.max(np.abs(R.T @ R - np.eye(3))) > 1e-5 or abs(np.linalg.det(R) - 1) > 1e-5:
        raise ReferencePoseError(f"{name}: rotation block is not a proper rotation")
    if np.max(np.abs(T[3] - [0, 0, 0, 1])) > 1e-9:
        raise ReferencePoseError(f"{name}: last row is not [0 0 0 1]")
    return T


# ============================================================================== 3.1 reference pose
def reference_pose_alignment(handoff_json, authority, scene_meta_dir=None, n_sigma=3.0) -> dict:
    """q_ref of the frozen mass geometry, PROVEN from the handoff and the joint authority (never assumed 0).

    The moving-part Gaussians of the handoff are posed at ``source.canonical_anchor_original_frame`` with joint
    coordinate ``joint.q_anchor``; the stored authority q at that same frame must agree within n_sigma. The frame
    chain Wc -> base_link comes from metadata/cameras.json (``coordinate_frame``); it is recorded, never assumed."""
    hp = Path(handoff_json)
    h = json.loads(hp.read_text())
    src, joint = h.get("source") or {}, h.get("joint") or {}
    anchor, q_anchor = src.get("canonical_anchor_original_frame"), joint.get("q_anchor")
    if anchor is None or q_anchor is None or not math.isfinite(float(q_anchor)):
        raise ReferencePoseError(f"{hp}: canonical_anchor_original_frame / joint.q_anchor missing: q_ref unknown")
    T = validate_transform(h["parts"]["moving"]["T_Wc_P"], "parts.moving.T_Wc_P")
    # "original frame" = RGB-D frame id (frame_map.json: original_frame_definition); NOT the robot source row
    sfi = np.asarray(authority.rgbd_frame_index, int)
    valid = np.asarray(authority.q_valid, bool)
    q = np.asarray(authority.q, float)
    exact = np.flatnonzero((sfi == int(anchor)) & valid)
    how = "exact anchor frame"
    if not len(exact):
        cand = np.flatnonzero(valid & (np.abs(sfi - int(anchor)) <= 2))
        if not len(cand):
            raise ReferencePoseError(f"authority has no valid q within 2 frames of the anchor frame {anchor}")
        exact = cand[[int(np.argmin(np.abs(sfi[cand] - int(anchor))))]]
        how = f"nearest valid frame {int(sfi[exact[0]])} (anchor {anchor} not valid)"
    i = int(exact[0])
    qs = authority.q_sigma
    sig = float(qs[i]) if qs is not None and np.isfinite(qs[i]) and qs[i] > 0 else float(
        np.nanmedian(np.asarray(qs, float))) if qs is not None else 0.0
    q_auth = float(q[i])
    dq = q_auth - float(q_anchor)
    fmap = None
    fm_p = hp.parent / ((h.get("frames") or {}).get("file") or "frame_map.json")
    if fm_p.is_file():
        fm = json.loads(fm_p.read_text())
        row = next((r for r in fm.get("rows", []) if int(r.get("original_frame", -1)) == int(anchor)), None)
        fmap = dict(file=str(fm_p), sha256=_sha(fm_p), anchor_original_frame=fm.get("anchor_original_frame"),
                    definition=fm.get("original_frame_definition"), row_at_anchor=row)
    tol = max(n_sigma * sig, 1e-6)
    a_h, a_auth = _unit(joint["axis_Wc"]), _unit(authority.axis_world)
    axis_deg = float(math.degrees(math.acos(np.clip(a_h @ a_auth, -1, 1))))
    org = None
    if joint.get("origin_Wc") is not None and authority.origin_world is not None:
        d = np.asarray(joint["origin_Wc"], float) - np.asarray(authority.origin_world, float)
        org = float(np.linalg.norm(d - (d @ a_auth) * a_auth))
    frame = dict(handoff_reference_frame=joint.get("reference_frame"), authority_world_frame=authority.world_frame,
                 acquisition_world=src.get("acquisition_world"))
    if scene_meta_dir is not None and (Path(scene_meta_dir) / "cameras.json").is_file():
        cj = json.loads((Path(scene_meta_dir) / "cameras.json").read_text())
        frame.update(cameras_json_coordinate_frame=cj.get("coordinate_frame"),
                     cameras_json_calibration_status=cj.get("calibration_status"),
                     cameras_json_sha256=_sha(Path(scene_meta_dir) / "cameras.json"))
        frame["T_B_Wc"] = ("IDENTITY: cameras.json declares coordinate_frame base_link (registration applied; status "
                           f"{cj.get('calibration_status')})" if cj.get("coordinate_frame") == "base_link" else
                           f"UNKNOWN: cameras.json coordinate_frame {cj.get('coordinate_frame')!r}")
    else:
        frame["T_B_Wc"] = "NOT_EVALUATED (no metadata/cameras.json)"
    reasons = []
    if abs(dq) > tol:
        reasons.append(f"authority q at the anchor ({q_auth:.6g}) differs from the handoff q_anchor ({q_anchor}) "
                       f"by {dq:.3g} > {tol:.3g}")
    if axis_deg > 1e-3 and axis_deg < 179.999:
        reasons.append(f"handoff axis differs from the authority axis by {axis_deg:.4g} deg")
    if org is not None and org > 1e-5:
        reasons.append(f"handoff hinge line is {org:.3g} m from the authority hinge line")
    if fmap is not None:
        if fmap["anchor_original_frame"] is not None and int(fmap["anchor_original_frame"]) != int(anchor):
            reasons.append(f"frame_map anchor {fmap['anchor_original_frame']} != handoff anchor {anchor}")
        r_ = fmap["row_at_anchor"]
        if r_ is not None and r_.get("q") is not None and abs(float(r_["q"]) - q_auth) > tol:
            reasons.append(f"frame_map q at the anchor {r_['q']} != authority q {q_auth}")
    if not str(frame.get("T_B_Wc", "")).startswith("IDENTITY"):
        reasons.append(f"frame chain Wc -> base_link not established: {frame['T_B_Wc']}")
    return dict(schema="forceartgs.g9.reference_pose_alignment/v1", status="PASS" if not reasons else "FAIL",
                reasons=reasons, q_ref=q_auth, q_ref_source=f"joint authority q at the handoff anchor ({how})",
                q_anchor_handoff=float(q_anchor), q_authority_at_anchor=q_auth, q_sigma_at_anchor=sig,
                authority_index_at_anchor=i,
                authority_source_row_at_anchor=int(np.asarray(authority.source_frame_index)[i]), frame_map=fmap,
                difference=dq, tolerance=tol, anchor_original_frame=int(anchor),
                anchor_timestamp_s=src.get("canonical_anchor_timestamp_s"),
                axis_handoff_vs_authority_deg=axis_deg, hinge_line_distance_m=org,
                T_Wc_P=T.tolist(), T_Wc_P_check="proper rigid transform (orthonormal, det +1)",
                frames=frame, handoff=str(hp), handoff_sha256=_sha(hp),
                ply_sha256=h["parts"]["moving"].get("ply_sha256"),
                q_observed_range_handoff=joint.get("q_observed_range"),
                note="the mass geometry is posed at q_ref; c(q) = o + R_a(q - q_ref)(c_ref - o) in every estimator")


# ============================================================================== 4. hull / unit-density audit
def _components(F):
    parent = list(range(int(F.max()) + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for f in F:
        a, b, c = (find(int(v)) for v in f)
        parent[b] = a
        parent[find(c)] = a
    return len({find(int(v)) for v in np.unique(F)})


def hull_diagnostics(mesh: TriMesh) -> dict:
    V, F = mesh.vertices, mesh.faces
    top = topology(mesh)
    c = V.mean(0)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    outward = np.einsum("ij,ij->i", n, V[F[:, 0]] - c) > 0
    sv = float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)
    return dict(bounds_min=V.min(0).tolist(), bounds_max=V.max(0).tolist(), extent_m=np.ptp(V, 0).tolist(),
                n_vertices=int(len(V)), n_faces=int(len(F)), connected_components=_components(F),
                watertight=top["watertight"], consistently_oriented=top["consistently_oriented"],
                fraction_faces_outward=float(outward.mean()), signed_volume_m3=sv,
                winding="outward (positive signed volume)" if sv > 0 else "INWARD / INVALID", topology=top)


def unit_gravity_report(props, joint_type, axis, origin, g, q_ref, q_obs=None) -> dict:
    """Unit-density quantities of ONE geometric source and joint (section 3.2): m1, c_ref, J_C1, r_perp, I_axis,1,
    G_1 coefficients (stored q and local q - q_ref), phase, amplitude, gravity/inertia ratio."""
    a = _unit(axis)
    g = np.asarray(g, float)
    o = np.zeros(3) if origin is None else np.asarray(origin, float)
    r = props.c1 - o
    rp = r - (r @ a) * a if joint_type == "revolute" else np.zeros(3)
    ell = float(np.linalg.norm(rp))
    I1 = float(a @ props.J1 @ a + props.m1 * ell * ell) if joint_type == "revolute" else float(props.m1)
    gu = RigidBodyGravity(joint_type, tuple(a), tuple(o), tuple(props.c1), props.m1, tuple(g), float(q_ref))
    A, B = gu.basis_coefficients()
    A0, B0 = gu._AB_local() if joint_type == "revolute" else (A, 0.0)
    gxa = float(np.linalg.norm(np.cross(g, a)))
    beta = float(math.degrees(math.acos(min(1.0, abs(a @ Z)))))
    amp = gu.amplitude_bound()
    out = dict(mode=props.mode, m1=float(props.m1), m1_units="m^3" if props.mode == "uniform_solid" else "m^2",
               density_units=DENSITY_UNITS[props.mode], c_ref_B=props.c1.tolist(), J_C1=np.asarray(props.J1).tolist(),
               J_C1_principal=np.linalg.eigvalsh(props.J1).tolist(), axis_B=a.tolist(), origin_B=o.tolist(),
               q_ref=float(q_ref), hinge_tilt_from_vertical_deg=beta, g_cross_a_norm=gxa)
    if joint_type == "revolute":
        out.update(r_perp_B=rp.tolist(), r_perp_norm_m=ell, COM_axial_offset_m=float(r @ a),
                   I_axis1=I1, I_axis1_check_props=float(props.axis_inertia(a, o)),
                   G1_stored_q=dict(A=A, B=B, form="G1(q) = A cos q + B sin q"),
                   G1_local=dict(A0=A0, B0=B0, form="G1(q) = A0 cos(q - q_ref) + B0 sin(q - q_ref)"),
                   G1_phase_rad=float(math.atan2(B, A)), G1_amplitude=amp,
                   G1_amplitude_analytic=float(props.m1 * gxa * ell),
                   gravity_to_inertia_ratio=float(amp / I1) if I1 > 0 else None)
    else:
        out.update(M1=I1, G1=A, G1_analytic=float(-props.m1 * g @ a), gravity_to_inertia_ratio=float(abs(A) / I1))
    if q_obs is not None and len(q_obs):
        qq = np.asarray(q_obs, float)
        Gq = gu.G(qq)
        out["G1_on_observed_q"] = dict(q_min=float(qq.min()), q_max=float(qq.max()), G1_min=float(np.min(Gq)),
                                       G1_max=float(np.max(Gq)))
    return out


def mass_com_inertia_audit(mesh, joint_type, axis, origin, g, q_ref, *, q_obs=None, densities=None,
                           fitted_mode=None, reflected=0.0, provenance=None, tentative=True) -> dict:
    """Both mass-distribution hypotheses of the SAME closed surface (solid and shell), plus the mass implied by every
    method's density. Densities only make sense with the mode they were fitted with (units differ)."""
    modes = {}
    for mode in ("uniform_solid", "uniform_shell"):
        props = unit_density_properties(mesh, mode, "B")
        modes[mode] = unit_gravity_report(props, joint_type, axis, origin, g, q_ref, q_obs)
    s, h = modes["uniform_solid"], modes["uniform_shell"]
    diff = dict(com_shift_shell_minus_solid_m=float(np.linalg.norm(np.subtract(h["c_ref_B"], s["c_ref_B"]))))
    if joint_type == "revolute":
        diff.update(r_perp_norm_solid_m=s["r_perp_norm_m"], r_perp_norm_shell_m=h["r_perp_norm_m"],
                    gravity_to_inertia_ratio_solid=s["gravity_to_inertia_ratio"],
                    gravity_to_inertia_ratio_shell=h["gravity_to_inertia_ratio"],
                    note="for the SAME fitted M the gravity amplitude scales with this ratio: the shell/solid "
                         "spread is a geometry SCENARIO, not statistical uncertainty")
    implied = {}
    if densities and fitted_mode in modes:
        rep = modes[fitted_mode]
        for meth, rho in densities.items():
            if rho is None or not math.isfinite(rho):
                continue
            e = dict(density=float(rho), density_units=rep["density_units"], mass_kg=float(rho * rep["m1"]),
                     inertia_com_kg_m2=(float(rho) * np.asarray(rep["J_C1"])).tolist())
            if joint_type == "revolute":
                e.update(I_axis_kg_m2=float(rho * rep["I_axis1"]), M_eff_kg_m2=float(rho * rep["I_axis1"] + reflected),
                         gravity_amplitude_Nm=float(rho * rep["G1_amplitude"]))
                if rep.get("G1_on_observed_q"):
                    go = rep["G1_on_observed_q"]
                    e["gravity_load_on_observed_q_Nm"] = [float(rho * go["G1_min"]), float(rho * go["G1_max"])]
            else:
                e.update(M_eff_kg=float(rho * rep["M1"] + reflected), gravity_load_N=float(rho * rep["G1"]))
            implied[meth] = e
    return dict(schema="forceartgs.g9.mass_com_inertia_audit/v1", flags=TENTATIVE_FLAGS if tentative else [],
                hull=hull_diagnostics(mesh), mesh_sha256=mesh.sha256(), provenance=provenance, modes=modes,
                solid_vs_shell=diff, fitted_mode=fitted_mode, reflected_inertia=reflected,
                implied_by_method=implied,
                note="COM = integrated hull centroid (never the mean of Gaussian centres); Gaussian covariance / "
                     "opacity never used as density; implied masses are NOT physical masses (tentative geometry, "
                     "no independent mass measurement, no held-out episode)")


# ============================================================================== 5. axis / COM sensitivity
def _effort_for(rec, jt, axis, origin, sl):
    F = rec["F_object_base_N"][sl[0]:sl[1]]
    if jt == "prismatic":
        return F @ axis
    P = rec["p_tcp_base_m"][sl[0]:sl[1]]
    return np.einsum("j,nj->n", axis, np.cross(P - origin, F))


def _scenario_interactions(train, rec, jt, axis, origin):
    from ..estimators.model_spec import G9Interaction
    from ..synthetic import ZOH
    out = []
    for it in train:
        sl = it.meta.get("_record_slice")
        if sl is None or rec is None:
            return None
        Q = _effort_for(rec, jt, axis, origin, sl)
        e = ZOH(np.asarray(it.effort.t_s, float), Q)
        out.append(G9Interaction(it.name, it.joint_type, it.t_obs, it.q_obs, it.q_std, e, it.initial_state,
                                 dict(it.meta)))
    return out


def gravity_joint_sensitivity(spec, mesh, mode, thetas: dict, train, rec, *, observability=None, q_ref=0.0,
                              n_random=12, seed=0, reg_shift_m=0.01) -> dict:
    """Counterfactual geometry scenarios with the observed force and q held fixed. Every method's FROZEN theta is
    replayed open loop. Axis / hinge changes REPROJECT the same measured 3-D force onto the new joint (physically
    consistent); ``GRAVITY_ONLY_COUNTERFACTUAL`` rows keep the original effort and change only the gravity law."""
    from ..dynamics.geometry_inertia import GeometryMassModel
    from ..estimators.general_ekf import _replay
    from ..estimators.model_spec import ModelSpec
    jt = spec.joint_type
    a0 = _unit(spec.mass_model.axis)
    o0 = np.asarray(spec.mass_model.origin, float)
    g = np.asarray(spec.gravity.g_world, float)
    obs = observability or {}
    s_ax = math.radians(float(obs.get("axis_sigma_deg") or 0.0))
    s_o = float(obs.get("hinge_line_sigma_m") or 0.0)
    u1, u2 = _perp_basis(a0)
    refl = spec.mass_model.reflected()
    q_all = np.concatenate([it.q_obs for it in train]) if train else np.zeros(0)

    sc = [dict(name="accepted", axis=a0, origin=o0, label="ACCEPTED_KINEMATICS", mesh=mesh, mode=mode)]
    if s_ax > 0:
        for k in (1, 2):
            for sg in (1, -1):
                for nm, u in (("u1", u1), ("u2", u2)):
                    sc.append(dict(name=f"axis_tilt_{nm}_{sg * k:+d}sigma", axis=_unit(a0 + sg * k * math.tan(s_ax) * u),
                                   origin=o0, label="AXIS_UNCERTAINTY_SCENARIO (stored covariance)", mesh=mesh,
                                   mode=mode))
        rng = np.random.default_rng(seed)
        for i in range(n_random):
            d = rng.standard_normal(2) * math.tan(s_ax)
            sc.append(dict(name=f"axis_random3d_{i:02d}", axis=_unit(a0 + d[0] * u1 + d[1] * u2), origin=o0,
                           label="AXIS_UNCERTAINTY_SCENARIO (random 3-D draw from the stored cone)", mesh=mesh,
                           mode=mode))
    if s_o > 0 and jt == "revolute":
        for sg in (1, -1):
            for nm, u in (("u1", u1), ("u2", u2)):
                sc.append(dict(name=f"hinge_line_{nm}_{sg:+d}sigma", axis=a0, origin=o0 + sg * s_o * u,
                               label="HINGE_LINE_UNCERTAINTY_SCENARIO", mesh=mesh, mode=mode))
    if jt == "revolute":
        v = Z * (1.0 if a0 @ Z >= 0 else -1.0)
        sc.append(dict(name="exactly_vertical_hinge", axis=v, origin=o0,
                       label="ALTERNATIVE_KINEMATIC_HYPOTHESIS (not the accepted joint; G9 replay can never select it)",
                       mesh=mesh, mode=mode))
        for sg in (1, -1):
            for nm, u in (("u1", u1), ("u2", u2)):
                T = np.eye(4)
                T[:3, 3] = sg * reg_shift_m * u
                sc.append(dict(name=f"com_registration_{nm}_{sg * reg_shift_m * 100:+.0f}cm", axis=a0, origin=o0,
                               label=f"COM_GEOMETRY_SCENARIO (mass geometry shifted {reg_shift_m} m, declared)",
                               mesh=mesh.transformed(T), mode=mode))
    other = "uniform_shell" if mode == "uniform_solid" else "uniform_solid"
    sc.append(dict(name=f"mass_mode_{other}", axis=a0, origin=o0, mesh=mesh, mode=other,
                   label="COM_GEOMETRY_SCENARIO (other mass-distribution hypothesis; density re-matched to the same "
                         "effective inertia M)"))

    base_its = train
    names = list(spec.names)
    rows = []
    base_M1 = spec.mass_model.M1
    for s in sc:
        try:
            mm = GeometryMassModel.from_validated_mesh(
                s["mesh"], s["mode"], jt, s["axis"], s["origin"] if jt == "revolute" else None,
                source_provenance=dict(spec.mass_model.source_provenance, scenario=s["name"]),
                reflected_actuator_inertia=spec.mass_model.reflected_actuator_inertia, q_ref=q_ref)
        except ValueError as e:
            rows.append(dict(name=s["name"], label=s["label"], status="FAILED", error=str(e)))
            continue
        sp = ModelSpec(spec.terms, jt, "geometry_coupled", mm, spec.gravity,
                       allow_unverified_gravity=spec.allow_unverified_gravity)
        rep = unit_gravity_report(mm.props, jt, s["axis"], s["origin"], g, q_ref, q_all)
        its_p = _scenario_interactions(base_its, rec, jt, s["axis"], s["origin"]) if s["name"] != "accepted" else \
            base_its
        dQ = None
        if its_p is not None and s["name"] != "accepted":
            dQ = float(np.sqrt(np.mean(np.concatenate([np.asarray(b.effort.levels) - np.asarray(a.effort.levels)
                                                       for a, b in zip(base_its, its_p)]) ** 2)))
        row = dict(name=s["name"], label=s["label"], mode=s["mode"], axis_B=np.asarray(s["axis"]).tolist(),
                   origin_B=np.asarray(s["origin"]).tolist(),
                   axis_change_deg=float(math.degrees(math.acos(np.clip(_unit(s["axis"]) @ a0, -1, 1)))),
                   tilt_from_vertical_deg=rep["hinge_tilt_from_vertical_deg"], g_cross_a_norm=rep["g_cross_a_norm"],
                   r_perp_norm_m=rep.get("r_perp_norm_m"), M1=mm.M1,
                   G1_amplitude=rep.get("G1_amplitude", abs(rep.get("G1", 0.0))),
                   effort_rms_change=dQ, per_method={})
        for meth, th in thetas.items():
            if th is None:
                continue
            th = np.asarray(th, float)[:len(names)].copy()
            if s["mode"] != mode and "alpha" in names:
                M = th[names.index("alpha")] * base_M1 + refl
                th[names.index("alpha")] = (M - refl) / mm.M1          # same effective inertia, other geometry
            rho = float(th[names.index("alpha")]) if "alpha" in names else None
            pm = dict(density=rho, M=None if rho is None else rho * mm.M1 + refl,
                      max_abs_G=None if rho is None else rho * row["G1_amplitude"])
            if rho is not None and rep.get("G1_on_observed_q"):
                go = rep["G1_on_observed_q"]
                pm["G_on_observed_q"] = [rho * go["G1_min"], rho * go["G1_max"]]
            for label, its_use in (("reprojected_effort", its_p),
                                   ("GRAVITY_ONLY_COUNTERFACTUAL", base_its if s["name"] != "accepted" else None)):
                if its_use is None:
                    continue
                rr = _replay(sp, th, its_use)["per_interaction"]
                vals = [x["rmse_over_sigma"] for x in rr if "rmse_over_sigma" in x]
                pm[f"replay_rmse_over_sigma_{label}"] = float(np.mean(vals)) if vals else None
            row["per_method"][meth] = pm
        rows.append(row)
    acc = rows[0]
    geo = [r for r in rows if r.get("label", "").startswith(("AXIS", "HINGE", "COM"))]
    summ = {}
    for meth in thetas:
        vals = [r["per_method"].get(meth, {}).get("max_abs_G") for r in geo if r.get("per_method")]
        vals = [v for v in vals if v is not None]
        if vals:
            summ[meth] = dict(max_abs_G_accepted=acc["per_method"].get(meth, {}).get("max_abs_G"),
                              max_abs_G_scenario_range=[min(vals), max(vals)],
                              note="GEOMETRY-induced variability (frozen theta); statistical parameter uncertainty is "
                                   "reported separately per method")
    vert = next((r for r in rows if r["name"] == "exactly_vertical_hinge"), None)
    return dict(schema="forceartgs.g9.gravity_joint_sensitivity/v1", joint_type=jt, q_ref=q_ref,
                accepted_axis_B=a0.tolist(), accepted_tilt_from_vertical_deg=acc["tilt_from_vertical_deg"],
                observability=obs, scenarios=rows, geometry_variability=summ,
                vertical_alternative=None if vert is None else dict(
                    G_identically_zero=bool(vert["G1_amplitude"] < 1e-12),
                    upstream_kinematic_residual="NOT_RUN: the RGB-D reprojection / track / contact residual of a "
                                                "vertical hinge must be evaluated in the joint-discovery model; a "
                                                "better G9 replay is never evidence for it",
                    label=vert["label"]),
                rules=["axis / joint type stay upstream decisions", "reprojected rows use the same measured 3-D "
                       "force on the new axis", "GRAVITY_ONLY_COUNTERFACTUAL rows isolate the gravity law"])


# ============================================================================== 6. method parity
def method_gravity_parity(spec, theta, n=25, seed=0) -> dict:
    """ModelSpec (rollout / EKF transition), TorchDynamics (PINN), the integral-LS basis and the reference
    RigidBodyGravity evaluate the SAME M, G and dr/drho at random states."""
    import torch
    from ..estimators.torch_model import DT, TorchDynamics
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta, float)[:len(spec.names)]
    v = spec.values(theta)
    q = rng.uniform(-math.pi, math.pi, n)
    qdd = rng.normal(0, 2.0, n)
    out = dict(schema="forceartgs.g9.method_gravity_parity/v1", parameter_source=spec.parameter_source,
               n_states=n, checks={})
    G_spec = spec.G_fn(theta)
    Gs = np.array([G_spec(x) for x in q]) if G_spec is not None else np.zeros(n)
    sim = spec.sim(theta)
    Gsim = np.array([sim.G(x) for x in q]) if sim.G is not None else np.zeros(n)
    dyn = TorchDynamics(spec)
    eta = dyn.eta0(theta)
    vt = dyn.values(eta)
    M_t, c_t, Fc_t, Fs_t, b_t, G_t = dyn.terms(vt)
    qt = torch.as_tensor(q, dtype=DT)
    Gt = G_t(qt).detach().numpy() if callable(G_t) else np.full(n, float(G_t))
    ph = spec.physical(theta)
    scale = max(1.0, float(np.max(np.abs(Gs))))
    out["checks"]["G_ModelSpec_vs_simulator"] = float(np.max(np.abs(Gs - Gsim)) / scale)
    out["checks"]["G_ModelSpec_vs_Torch"] = float(np.max(np.abs(Gs - Gt)) / scale)
    out["checks"]["M_ModelSpec_vs_Torch"] = float(abs(float(M_t) - float(ph["M"])) / max(abs(float(ph["M"])), 1e-300))
    out["checks"]["M_ModelSpec_vs_simulator"] = float(abs(sim.M - float(ph["M"])) / max(abs(float(ph["M"])), 1e-300))
    if spec.parameter_source == "geometry_coupled":
        mm = spec.mass_model
        rho = float(v["alpha"])
        gu = mm.gravity_unit(spec.gravity.g_world)
        ref = rho * gu.G(q) if spec.terms.gravity else np.zeros(n)
        out["checks"]["G_ModelSpec_vs_reference_RigidBodyGravity"] = float(np.max(np.abs(Gs - ref)) / scale)
        out["checks"]["M_vs_rho_M1_plus_reflected"] = float(abs(float(ph["M"]) - (rho * mm.M1 + mm.reflected())) /
                                                            max(abs(float(ph["M"])), 1e-300))
        # dr/drho: autograd (through log rho) vs analytic M1 qdd + G1(q) vs central finite difference
        e2 = {k: x.detach().clone().requires_grad_(k == "alpha") for k, x in eta.items()}
        r = dyn.residual(dyn.values(e2), qt, torch.zeros(n, dtype=DT), torch.as_tensor(qdd, dtype=DT),
                         torch.zeros(n, dtype=DT))
        gr = np.array([float(torch.autograd.grad(r[i], e2["alpha"], retain_graph=True)[0]) for i in range(n)]) / rho
        ana = mm.M1 * qdd + (gu.G(q) if spec.terms.gravity else 0.0)
        h = 1e-6 * rho
        thp, thm = theta.copy(), theta.copy()
        k = spec.names.index("alpha")
        thp[k] += h
        thm[k] -= h

        def res(th):
            p = spec.physical(th)
            Gf = spec.G_fn(th)
            return p["M"] * qdd + (np.array([Gf(x) for x in q]) if Gf is not None else 0.0)
        fd = (res(thp) - res(thm)) / (2 * h)
        sc2 = max(1.0, float(np.max(np.abs(ana))))
        out["checks"]["dr_drho_autograd_vs_analytic"] = float(np.max(np.abs(gr - ana)) / sc2)
        out["checks"]["dr_drho_fd_vs_analytic"] = float(np.max(np.abs(fd - ana)) / sc2)
        out["dr_drho_form"] = ("I_axis,1 qdd + G1(q)" if spec.joint_type == "revolute" else "m1 qdd - m1 g.a")
        out["one_density"] = dict(inertia="M = rho M1 + I_refl", gravity="G = rho G1(q)", rho=rho, M1=mm.M1,
                                  q_ref=mm.q_ref, reflected=mm.reflected())
    elif spec.parameter_source == "joint_space_gravity_coupled":
        out["flags"] = ["GRAVITY_MASS_NOT_COUPLED" if spec.joint_type == "revolute" else
                        "PRISMATIC_SHARED_EFFECTIVE_MASS (no reflected mass declared)"]
    tol = 1e-8
    out["status"] = "PASS" if all(v_ <= (tol if not k_.startswith("dr_drho_fd") else 1e-5)
                                  for k_, v_ in out["checks"].items()) else "FAIL"
    return out


# ============================================================================== 7. impulse balance
def impulse_balance_audit(ic_windows, spec, thetas: dict, rec, train, axis, origin, *, chi2_unreliable=5.0) -> dict:
    """M dq' + int(c q' + Fc sgn q' + G + b) dt = J_Q over each declared pulse, per method, with the gauge / sign /
    timing evidence. Never flips a sign; a contradiction is reported with its evidence."""
    out = dict(schema="forceartgs.g9.impulse_balance_audit/v1", pulses=[])
    if not ic_windows:
        out["status"] = "NOT_RUN"
        out["reason"] = "no impulse / coast analysis (no record context or no declared pulse rows)"
        return out
    a = _unit(axis)
    o = np.asarray(origin, float) if spec.joint_type == "revolute" and origin is not None else None
    out["force_sign"] = dict(object_side_multiplier=None if rec is None else float(rec.get("object_side_multiplier")),
                             conversion=None if rec is None else rec.get("force_conversion"),
                             rule="explicit metadata multiplier applied ONCE (record); never changed here")
    gauge = _gauge_check(rec, spec.joint_type, a, o, train)
    out["q_axis_gauge_check"] = gauge
    for w in ic_windows:
        for dp in w.get("declared_pulses", []):
            p, vel = dp.get("pulse") or {}, dp.get("velocity")
            row = dict(rows=dp["rows"], J_Q=p.get("J_Q"), sigma_J_Q=p.get("sigma_J_Q"),
                       duration_s=p.get("duration_s"), oscillatory_transient=p.get("oscillatory_transient"),
                       n_force_samples=p.get("n_force_samples"), camera_frames_inside=p.get("camera_frames_inside"),
                       velocity=vel, per_method={})
            if vel is None:
                row["verdict"] = "SIGN_OR_TIMING_INCONCLUSIVE"
                row["reason"] = f"velocities unavailable: {dp.get('velocity_unavailable')}"
                out["pulses"].append(row)
                continue
            t0, t1 = p["t_start"], p["t_end"]
            tt = np.linspace(t0, t1, 41)
            qd = vel["v_pre"] + (vel["v_post"] - vel["v_pre"]) * (tt - t0) / max(t1 - t0, 1e-9)
            qq = _q_at(train, tt)
            post_chi2 = (vel.get("post_fit") or {}).get("chi2_dof")
            pre_chi2 = (vel.get("pre_fit") or {}).get("chi2_dof")
            for meth, th in thetas.items():
                if th is None:
                    continue
                th = np.asarray(th, float)[:len(spec.names)]
                ph = spec.physical(th)
                Gf = spec.G_fn(th)
                c = ph.get("c") or 0.0
                Fc = ph.get("Fc") or 0.0
                b = ph.get("b") or 0.0
                integ = dict(viscous=float(np.trapz(c * qd, tt)), coulomb=float(np.trapz(Fc * np.sign(qd), tt)),
                             gravity=float(np.trapz([Gf(x) for x in qq], tt)) if Gf is not None and qq is not None
                             else 0.0, bias=float(b * (t1 - t0)))
                Mdv = float(ph["M"] * vel["dv"])
                resid = Mdv + sum(integ.values()) - p["J_Q"]
                sd = math.hypot(p["sigma_J_Q"], abs(ph["M"]) * vel["dv_sd"])
                row["per_method"][meth] = dict(M=float(ph["M"]), M_dqdot=Mdv, integrals=integ, residual=resid,
                                               residual_sd=sd, z=float(resid / sd) if sd > 0 else math.inf)
            row["tool_inertia_hypothesis"] = _tool_inertia(rec, spec.joint_type, a, o, p, train)
            lagd = (dp.get("camera_force_lag_scan") or {}).get("dv_by_q_shift") or {}
            row["lag_scan_dv"] = lagd
            row["lag_scan_any_sign_flip"] = bool(any(v and np.sign(v["dv"]) == np.sign(p["J_Q"])
                                                     for v in lagd.values()))
            zs = [abs(v["z"]) for v in row["per_method"].values()]
            timing_bad = bool((post_chi2 or 0) > chi2_unreliable or (pre_chi2 or 0) > chi2_unreliable)
            sign_contra = bool(np.sign(p["J_Q"]) != np.sign(vel["dv"]))
            row.update(sign_contradiction=sign_contra, velocity_fit_chi2_dof=dict(pre=pre_chi2, post=post_chi2),
                       velocity_fits_unreliable=timing_bad,
                       other_torques_can_explain=bool(zs and min(zs) <= 2.0))
            if gauge.get("status") == "INCONSISTENT":
                row["verdict"] = "SIGN_OR_TIMING_INCONCLUSIVE"
                row["reason"] = "q / axis gauge inconsistent with the TCP motion (see q_axis_gauge_check)"
            elif sign_contra and (timing_bad or p.get("oscillatory_transient") or
                                  (p.get("camera_frames_inside") or 0) <= 1):
                row["verdict"] = "SIGN_OR_TIMING_INCONCLUSIVE"
                row["reason"] = ("J_Q and dq' have opposite signs, but the pulse is "
                                 + ", ".join(x for x, f in (("an oscillatory 50 Hz transient", p.get("oscillatory_transient")),
                                                            ("bracketed by unreliable velocity fits", timing_bad),
                                                            ("resolved by <= 1 camera frame",
                                                             (p.get("camera_frames_inside") or 0) <= 1)) if f))
            elif zs and min(zs) > 3:
                row["verdict"] = "IMPULSE_DYNAMIC_MODEL_CONTRADICTED"
                row["reason"] = f"impulse balance off by >= {min(zs):.1f} sigma for every method"
            else:
                row["verdict"] = "IMPULSE_BALANCE_CONSISTENT (not an identification)"
            out["pulses"].append(row)
    out["status"] = "OK" if out["pulses"] else "NOT_RUN"
    if not out["pulses"]:
        out["reason"] = "no declared pulse rows (identifiability.declared_pulse_rows / --declared-pulse-rows)"
    return out


def _tool_inertia(rec, jt, a, o, p, train, pad=2) -> dict:
    """The sensor sits between flange and tool: during an impact it also reads the tool's own inertial force
    m_tool a_TCP. No tool mass is declared, so this is a HYPOTHESIS: the generalized impulse of a 1 kg tool moving
    with the measured TCP acceleration, and the tool mass that would cancel the measured J_Q (never applied)."""
    if rec is None:
        return dict(status="NOT_RUN", reason="no canonical record")
    src = np.asarray(rec["source_frame_index"], int)
    lo, hi = p["rows"]
    idx = np.flatnonzero((src >= lo - pad) & (src <= hi + pad))
    if len(idx) < 5:
        return dict(status="NOT_RUN", reason="too few rows around the pulse")
    t = src[idx] / 50.0
    P = np.asarray(rec["p_tcp_base_m"], float)[idx]
    acc = np.gradient(np.gradient(P, t, axis=0), t, axis=0)
    inside = (src[idx] >= lo) & (src[idx] <= hi)
    # sensor reads F_rec = F_object_on_tool - m a  =>  computed object force -F_rec = F_robot_on_object + m a:
    # the unmodelled error is +m a; cancelling J_Q needs m = J_Q / J_a, J_a = generalized impulse of +a per kg
    F1 = -acc
    Q1 = F1 @ a if jt == "prismatic" else np.einsum("j,nj->n", a, np.cross(P - o, F1))
    dt = np.diff(np.r_[t, t[-1] + 0.02])
    J1 = float(np.sum((Q1 * dt)[inside]))
    m_cancel = float(p["J_Q"] / -J1) if J1 != 0 else None
    return dict(status="HYPOTHESIS", J_Q_per_kg_tool=J1, peak_tcp_acceleration_mps2=float(np.max(np.linalg.norm(
        acc[inside], axis=1))), tool_mass_that_would_cancel_J_kg=m_cancel,
                plausible=bool(m_cancel is not None and 0.05 <= m_cancel <= 3.0),
                tool_mass_declared=None,
                note="TCP acceleration from second differences of the 50 Hz pose (noisy); a plausible positive mass "
                     "makes unmodelled tool inertia a candidate explanation -- NOT a correction")


def _q_at(train, tt):
    for it in train or []:
        if it.t_obs[0] <= tt[0] and tt[-1] <= it.t_obs[-1]:
            return np.interp(tt, it.t_obs, it.q_obs)
    return None


def _gauge_check(rec, jt, a, o, train) -> dict:
    """Does q increase in the physical sense of +a?  Compare the TCP velocity projected on the joint's tangent
    (a x (p - o) for R, a for P) with q' over contact rows (the TCP rides on the part while in contact)."""
    if rec is None or not train:
        return dict(status="NOT_RUN", reason="no canonical record")
    src = np.asarray(rec["source_frame_index"], int)
    P = np.asarray(rec["p_tcp_base_m"], float)
    con = np.asarray(rec["in_contact_segment"], bool)
    vals_t, vals_q = [], []
    for it in train:
        sl = it.meta.get("_record_slice")
        if sl is None:
            continue
        idx = np.arange(sl[0], sl[1])
        idx = idx[con[idx]]
        if len(idx) < 5:
            continue
        t = src[idx] / 50.0
        vt = np.gradient(P[idx], t, axis=0)
        tan = a[None, :] if jt == "prismatic" else np.cross(a, P[idx] - o)
        proj = np.einsum("ij,ij->i", vt, tan)
        qd = np.gradient(np.interp(t, it.t_obs, it.q_obs), t)
        vals_t.append(proj)
        vals_q.append(qd)
    if not vals_t:
        return dict(status="NOT_RUN", reason="fewer than 5 contact rows")
    x, y = np.concatenate(vals_t), np.concatenate(vals_q)
    m = (np.abs(y) > 1e-3) & (np.abs(x) > 1e-5)
    if m.sum() < 5:
        return dict(status="NOT_RUN", reason="the part barely moves during contact")
    agree = float(np.mean(np.sign(x[m]) == np.sign(y[m])))
    corr = float(np.corrcoef(x[m], y[m])[0, 1])
    st = "CONSISTENT" if agree >= 0.7 else "INCONSISTENT" if agree <= 0.3 else "INCONCLUSIVE"
    return dict(status=st, sign_agreement_fraction=agree, correlation=corr, n_rows=int(m.sum()),
                method="sign(v_TCP . tangent) vs sign(q') on contact rows; tangent = a x (p - o) (R) or a (P)",
                note="the paired gauge (a, q) -> (-a, -q) leaves this test invariant; only a mismatched pair fails")
