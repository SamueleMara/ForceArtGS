"""URDF + USD carrying ONLY the estimated joint dynamics of the rest-to-rest comparison.

Estimated (best PhysX-replayed row of `rest_to_rest_compare/vNNN`, model M q'' + c q' = Q, gravity off, Coulomb off):
  joint damping        c   (N s/m prismatic, N m s/rad revolute)            -> URDF <dynamics damping>, USD DriveAPI
  moving-part inertia  M   along the joint: revolute = inertia about the joint axis (kg m^2), prismatic = mass (kg)
Kinematics (joint type, axis, axis point) are the run's own 3DGS hand-off (the same accepted / working joint).

Nothing else is written: no collision, no joint stops (revolute -> URDF `continuous`, prismatic -> infinite limit),
no static-link inertial, no Coulomb friction, no visual mesh. URDF / USD cannot store "axis inertia only", so the
moving inertial is put where nothing else can enter the 1-DoF dynamics:
  revolute   COM ON the joint axis, tensor M/2 (I + a a^T): exactly M about the axis; the off-axis moments M/2 are the
             smallest valid (triangle-inequality) completion and, with a fixed base and the COM on the axis, change
             neither the joint dynamics nor any gravity torque. Mass NOT identified: a negligible placeholder.
  prismatic  mass M at the joint origin; rotational inertia NOT identified (cannot rotate): negligible placeholder.
Every placeholder is listed in estimated_physics.json. Verification: the URDF is loaded in SAPIEN (PhysX) and a
constant effort is replayed against the closed-form response of M q'' + c q' = Q; the USD is read back with pxr.
FIT_TO_DEMONSTRATION, not a validated physical twin. Append-only <out_root>/vNNN/.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import subprocess
import time
from pathlib import Path

import numpy as np

SCHEMA = "forceartgs.g9.estimated_joint_dynamics_asset/v1"
SIM_PY = sys.executable  # the forceartgs env (pxr + sapien); never another env
PLACEHOLDER = 1e-6        # not identified, dynamically inert (see module docstring)


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _next_version(root: Path) -> Path:
    n = len(sorted(root.glob("v[0-9][0-9][0-9]"))) if root.is_dir() else 0
    return root / f"v{n:03d}"


def best_fit(r2r_dir) -> dict:
    d = Path(r2r_dir)
    st = json.loads((d / "STATUS.json").read_text())
    row = st.get("best_row") or {}
    M, c = row.get("M"), row.get("c")
    if st.get("status") != "COMPARED" or M is None or c is None:
        raise ValueError(f"{d}: no PhysX-replayed estimate (status {st.get('status')}: {st.get('reason')})")
    M, c = float(M), float(c)
    if not (math.isfinite(M) and M > 0 and math.isfinite(c) and c >= 0):
        raise ValueError(f"{d}: non-physical estimate M={M}, c={c}")
    jt = (st.get("episode") or {}).get("joint_type")
    return dict(r2r_dir=str(d), row=row, M=M, c=c, joint_type=jt,
                episode_sha256=(st.get("episode") or {}).get("sha256"))


def choose_handoff(run, joint_type) -> Path | None:
    """The run's own G8 hand-off for this joint type: `accepted` first, else `diagnostic_<type>`."""
    found = {}
    for h in sorted(Path(run).glob("parttwin_handoff/*/articulated_3dgs_handoff.json")):
        if json.loads(h.read_text()).get("joint", {}).get("type") == joint_type:
            found[h.parent.name] = h
    return found.get("accepted") or found.get(f"diagnostic_{joint_type}")


def joint_dynamics_model(joint: dict, M: float, c: float) -> dict:
    """Link-frame quantities (moving link frame = joint frame at q = 0, axes parallel to the capture world Wc)."""
    a = np.asarray(joint["axis_Wc"], float)
    a = a / np.linalg.norm(a)
    o = np.zeros(3) if joint.get("origin_Wc") is None else np.asarray(joint["origin_Wc"], float)
    if joint["type"] == "revolute":
        I = 0.5 * M * (np.eye(3) + np.outer(a, a))
        mass, placeholders = PLACEHOLDER, {"moving_link.mass_kg": "not identified; COM on the joint axis -> no effect "
                                           "on the joint dynamics or gravity torque",
                                           "moving_link.off_axis_moments": "M/2 = minimal valid completion; no effect "
                                           "on a fixed-base 1-DoF revolute"}
    else:
        I = PLACEHOLDER * np.eye(3)
        mass, placeholders = M, {"moving_link.inertia_tensor": "not identified; a slide cannot rotate -> no effect"}
        if joint.get("origin_Wc") is None:
            placeholders["joint.origin"] = "a slide has no axis point: Wc origin (does not affect the slide)"
    return dict(type=joint["type"], axis=a, origin=o, mass=float(mass), com=np.zeros(3), inertia=I, damping=float(c),
                M=float(M), placeholders=placeholders,
                axis_inertia=float(a @ I @ a) if joint["type"] == "revolute" else None)


def write_urdf(path: Path, name: str, m: dict) -> None:
    I = m["inertia"]
    f = lambda v: " ".join(repr(float(x)) for x in v)
    jt = "continuous" if m["type"] == "revolute" else "prismatic"
    lim = "" if jt == "continuous" else '\n    <limit lower="-inf" upper="inf" effort="inf" velocity="inf"/>'
    path.write_text(f"""<?xml version="1.0"?>
<!-- ESTIMATED joint dynamics only (forceartgs_g9.export.estimated_asset): damping c and the moving-part inertia M
     along the joint; frame = capture world Wc (metres). No collision, stops, static inertial or friction. -->
<robot name="{name}">
  <link name="static_link"/>
  <link name="moving_link">
    <inertial>
      <origin xyz="{f(m['com'])}" rpy="0 0 0"/>
      <mass value="{m['mass']!r}"/>
      <inertia ixx="{I[0, 0]!r}" ixy="{I[0, 1]!r}" ixz="{I[0, 2]!r}" iyy="{I[1, 1]!r}" iyz="{I[1, 2]!r}" izz="{I[2, 2]!r}"/>
    </inertial>
  </link>
  <joint name="articulation_joint" type="{jt}">
    <parent link="static_link"/>
    <child link="moving_link"/>
    <origin xyz="{f(m['origin'])}" rpy="0 0 0"/>
    <axis xyz="{f(m['axis'])}"/>{lim}
    <dynamics damping="{m['damping']!r}" friction="0"/>
  </joint>
</robot>
""")


_SIM_SCRIPT = r'''
import json, math, sys
import numpy as np
req = json.loads(sys.argv[1])
out = {}
# ---------------------------------------------------------------- USD (UsdPhysics; angular units are DEGREES)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
m = req["model"]
a = np.asarray(m["axis"], float); o = np.asarray(m["origin"], float)
x = a / np.linalg.norm(a)
t = np.cross(x, [0, 0, 1.0]) if abs(x[2]) < 0.9 else np.cross(x, [1.0, 0, 0])
y = t / np.linalg.norm(t); z = np.cross(x, y)
R = np.stack([x, y, z], 1)                                  # joint frame (X = axis) in Wc
q = Gf.Quatf(Gf.Matrix3d(*R.T.flatten().tolist()).ExtractRotation().GetQuat())   # Gf row-vector convention
stage = Usd.Stage.CreateNew(req["usd"])
UsdGeom.SetStageMetersPerUnit(stage, 1.0); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/" + req["name"]); stage.SetDefaultPrim(root.GetPrim())
UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
st = UsdGeom.Xform.Define(stage, f"/{req['name']}/static_link")
UsdPhysics.RigidBodyAPI.Apply(st.GetPrim())
fj = UsdPhysics.FixedJoint.Define(stage, f"/{req['name']}/root_joint")
fj.CreateBody1Rel().SetTargets([st.GetPath()])
mv = UsdGeom.Xform.Define(stage, f"/{req['name']}/moving_link")
mv.AddTranslateOp().Set(Gf.Vec3d(*o.tolist())); mv.AddOrientOp().Set(q)
UsdPhysics.RigidBodyAPI.Apply(mv.GetPrim())
ma = UsdPhysics.MassAPI.Apply(mv.GetPrim())
Ij = R.T @ np.asarray(m["inertia"]) @ R                     # tensor in the moving-link (joint) frame
ev, V = np.linalg.eigh(0.5 * (Ij + Ij.T))
if np.linalg.det(V) < 0: V[:, 0] *= -1
pq = Gf.Matrix3d(*V.T.flatten().tolist()).ExtractRotation().GetQuat()
ma.CreateMassAttr(m["mass"]); ma.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*ev.tolist())); ma.CreatePrincipalAxesAttr(Gf.Quatf(pq))
J = (UsdPhysics.RevoluteJoint if m["type"] == "revolute" else UsdPhysics.PrismaticJoint).Define(
    stage, f"/{req['name']}/articulation_joint")
J.CreateAxisAttr("X")
J.CreateBody0Rel().SetTargets([st.GetPath()]); J.CreateBody1Rel().SetTargets([mv.GetPath()])
J.CreateLocalPos0Attr(Gf.Vec3f(*o.tolist())); J.CreateLocalRot0Attr(q)
J.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0)); J.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
kind = "angular" if m["type"] == "revolute" else "linear"
d_usd = m["damping"] * (math.pi / 180.0 if kind == "angular" else 1.0)   # N m s/rad -> N m s/deg
dr = UsdPhysics.DriveAPI.Apply(J.GetPrim(), kind)
dr.CreateTypeAttr("force"); dr.CreateStiffnessAttr(0.0); dr.CreateDampingAttr(d_usd); dr.CreateTargetVelocityAttr(0.0)
# no dissipation beyond the identified c: PhysX defaults (link damping 0.05, joint friction) explicitly zeroed
for pr in (st.GetPrim(), mv.GetPrim()):
    pr.AddAppliedSchema("PhysxRigidBodyAPI")
    pr.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(0.0)
    pr.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(0.0)
J.GetPrim().AddAppliedSchema("PhysxJointAPI")
J.GetPrim().CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(0.0)
root.GetPrim().SetCustomDataByKey("forceartgs:estimated", json.dumps(req["provenance"]))
stage.GetRootLayer().Save()
s2 = Usd.Stage.Open(req["usd"])
mp = s2.GetPrimAtPath(f"/{req['name']}/moving_link"); jp = s2.GetPrimAtPath(f"/{req['name']}/articulation_joint")
diag = np.asarray(UsdPhysics.MassAPI(mp).GetDiagonalInertiaAttr().Get(), float)
pa = UsdPhysics.MassAPI(mp).GetPrincipalAxesAttr().Get()
Pm = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(pa.GetReal(), *pa.GetImaginary()))), float).T
I_back = R @ (Pm @ np.diag(diag) @ Pm.T) @ R.T
out["usd"] = dict(path=req["usd"], mass=UsdPhysics.MassAPI(mp).GetMassAttr().Get(),
                  axis_inertia=float(x @ I_back @ x), drive=kind,
                  damping_usd=UsdPhysics.DriveAPI(jp, kind).GetDampingAttr().Get(),
                  damping_usd_units="N m s/deg" if kind == "angular" else "N s/m",
                  stiffness=UsdPhysics.DriveAPI(jp, kind).GetStiffnessAttr().Get())
# ---------------------------------------------------------------- SAPIEN (PhysX) constant-effort replay of the URDF
import sapien
scene = sapien.Scene(); dt = 1e-3; scene.set_timestep(dt)
ld = scene.create_urdf_loader(); ld.fix_root_link = True
art = ld.load(req["urdf"])
for l in art.get_links():                                  # engine defaults the identification zeroed (G10):
    l.disable_gravity = True; l.linear_damping = 0.0; l.angular_damping = 0.0   # gravity off, no link damping
j0 = art.get_active_joints()[0]; j0.set_friction(0.0)
links_rb = [dict(name=l.name, linear_damping=l.linear_damping, angular_damping=l.angular_damping) for l in art.get_links()]
j = [jj for jj in art.get_active_joints()][0]
Q = req["Q"]; Mm = m["M"]; c = m["damping"]; T = req["T"]
qs, vs = [], []
for k in range(int(round(T / dt))):
    art.set_qf(np.array([Q], np.float32)); scene.step()
    qs.append(float(art.get_qpos()[0])); vs.append(float(art.get_qvel()[0]))
tt = dt * np.arange(1, len(vs) + 1)
v_ref = Q / c * (1 - np.exp(-c * tt / Mm)) if c > 0 else Q * tt / Mm
out["sapien"] = dict(sapien_version=sapien.__version__, joint_type=j.type, joint_damping=float(j.damping), joint_friction=float(j.friction),
                     engine_overrides="link linear/angular damping 0 and joint friction 0 (SAPIEN defaults 0.05), "
                                      "gravity off -- as in the G10 identification", links=links_rb,
                     Q=Q, T=T, dt=dt, v_end_sim=vs[-1], v_end_closed_form=float(v_ref[-1]),
                     max_rel_err_v=float(np.max(np.abs(np.asarray(vs) - v_ref)) / max(abs(float(v_ref[-1])), 1e-12)))
print("RESULT " + json.dumps(out))
'''


def build(run, r2r_dir, out_root, *, python=None, log=print) -> dict:
    t0 = time.time()
    fit = best_fit(r2r_dir)
    out = _next_version(Path(out_root))
    out.mkdir(parents=True)

    def finish(status, reason, **kw):
        st = dict(schema=SCHEMA, status=status, reason=reason, run=str(run), r2r_dir=str(r2r_dir), out_dir=str(out),
                  estimate=dict(M=fit["M"], c=fit["c"], joint_type=fit["joint_type"],
                                M_meaning=("inertia about the joint axis [kg m^2]" if fit["joint_type"] == "revolute"
                                           else "moving mass [kg]"),
                                c_units="N m s/rad" if fit["joint_type"] == "revolute" else "N s/m",
                                method=fit["row"].get("method"),
                                physx_rmse_over_sigma=fit["row"].get("physx_rmse_over_sigma"),
                                episode_sha256=fit["episode_sha256"]),
                  labels=dict(FIT_TO_DEMONSTRATION=True, PHYSICAL_TWIN=False), runtime_s=time.time() - t0, **kw)
        (out / "STATUS.json").write_text(json.dumps(st, indent=1, default=str))
        log(f"[estimated-asset] {status}: {reason}")
        return st

    h = choose_handoff(run, fit["joint_type"])
    if h is None:
        return finish("BLOCKED_NO_HANDOFF", f"no {fit['joint_type']} hand-off under {run}/parttwin_handoff "
                                            "(pipeline flags --export-parttwin --export-diagnostic)")
    hj = json.loads(h.read_text())
    m = joint_dynamics_model(hj["joint"], fit["M"], fit["c"])
    name = hj["object_id"]
    urdf, usd = out / "model.urdf", out / "model.usda"
    write_urdf(urdf, name, m)
    jm = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
    prov = dict(r2r_dir=str(r2r_dir), method=fit["row"].get("method"), M=fit["M"], c=fit["c"], handoff=str(h),
                note="estimated joint damping + moving inertia along the joint only; see estimated_physics.json")
    (out / "estimated_physics.json").write_text(json.dumps(dict(
        schema=SCHEMA, handoff=str(h), handoff_sha256=_sha(h), handoff_status=hj["source"]["status"],
        joint_from_handoff=hj["joint"], frame="capture world Wc (metres); moving link frame = joint frame at q = 0",
        written=dict(joint_damping=m["damping"], moving_axis_inertia=m["axis_inertia"],
                     moving_mass=m["mass"] if m["type"] == "prismatic" else None),
        placeholders=m["placeholders"],
        not_written=["collision", "joint stops / velocity / effort limits", "static-link inertial",
                     "Coulomb / static friction", "armature", "visual meshes (3DGS PLYs stay in the run)"],
        model=jm), indent=1, default=str))
    Q = float(m["damping"] * 0.1 if m["damping"] > 0 else m["M"])        # probe effort (only scales the response)
    T = float(min(2.0, max(0.2, 3 * m["M"] / m["damping"]))) if m["damping"] > 0 else 0.5
    req = dict(model=jm, name=name, urdf=str(urdf), usd=str(usd), provenance=prov, Q=Q, T=T)
    p = subprocess.run([python or SIM_PY, "-c", _SIM_SCRIPT, json.dumps(req)], capture_output=True, text=True)
    (out / "verify.log").write_text(f"--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}")
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if p.returncode != 0 or not lines:
        return finish("VERIFY_FAILED", f"USD / SAPIEN step exit {p.returncode}: {p.stderr.strip()[-400:]}",
                      urdf=str(urdf), handoff=str(h))
    ver = json.loads(lines[-1][7:])
    (out / "verification.json").write_text(json.dumps(ver, indent=1))
    usd_ok = (math.isclose(ver["usd"]["axis_inertia"], m["M"], rel_tol=1e-4) if m["type"] == "revolute" else
              math.isclose(ver["usd"]["mass"], m["M"], rel_tol=1e-6))
    sim_ok = ver["sapien"]["max_rel_err_v"] < 0.02
    status = "EXPORTED" if usd_ok and sim_ok else "EXPORTED_CHECK_FAILED"
    return finish(status, f"{m['type']} M={m['M']:.4g} c={m['damping']:.4g}; SAPIEN replay max rel err "
                          f"{ver['sapien']['max_rel_err_v']:.2%}; USD read-back {'OK' if usd_ok else 'MISMATCH'}",
                  handoff=str(h), urdf=str(urdf), urdf_sha256=_sha(urdf), usd=str(usd), usd_sha256=_sha(usd),
                  verification=ver)
