"""URDF + USD carrying ONLY the estimated joint dynamics of the rest-to-rest comparison.

Estimated (best PhysX-replayed row of `rest_to_rest_compare/vNNN`, model M q'' + c q' = Q, gravity off, Coulomb off):
  joint damping        c   (N s/m prismatic, N m s/rad revolute)            -> URDF <dynamics damping>, USD DriveAPI
  moving-part inertia  M   along the joint: revolute = inertia about the joint axis (kg m^2), prismatic = mass (kg)
Kinematics (joint type, axis, axis point) are the run's own 3DGS hand-off (the same accepted / working joint).

Visuals = the run's own 3DGS parts (hand-off static_canonical.ply / moving_canonical.ply), copied BYTE-IDENTICAL into
<vNNN>/gaussians/ so the folder is self-contained. The Gaussians stay in the run's canonical frame P (moving part at
q = 0); each link's <visual><origin> is T_link_P (static_link = Wc: T_Wc_P; moving_link = joint frame at q = 0, axes
parallel to Wc at the joint origin o: Trans(-o) T_Wc_P), so the joint articulates the moving Gaussians exactly as in
the run: x_Wc(q) = T_Wc_moving(q) T_moving_P x_P. The USD carries the same placement (a `gaussians_3dgs` Xform per link
with the PLY as an asset path). A 3DGS PLY is not a triangle mesh: a viewer without a Gaussian renderer draws nothing.
Nothing else is written: no collision, no joint stops (revolute -> URDF `continuous`, prismatic -> infinite limit),
no static-link inertial, no Coulomb friction. URDF / USD cannot store "axis inertia only", so the
moving inertial is put where nothing else can enter the 1-DoF dynamics:
  revolute   COM ON the joint axis, tensor M/2 (I + a a^T): exactly M about the axis; the off-axis moments M/2 are the
             smallest valid (triangle-inequality) completion and, with a fixed base and the COM on the axis, change
             neither the joint dynamics nor any gravity torque. Mass NOT identified: a negligible placeholder.
  prismatic  mass M at the joint origin; rotational inertia NOT identified (cannot rotate): negligible placeholder.
Every placeholder is listed in estimated_physics.json. Verification: the URDF is loaded in SAPIEN (PhysX) and a
constant effort is replayed against the closed-form response of M q'' + c q' = Q, and the engine pose of every 3DGS
visual (link pose x visual pose, at q = 0 and at a moved q) is checked against the articulation above; the USD is
read back with pxr (inertia, damping, Gaussian placement and PLY paths).
FIT_TO_DEMONSTRATION, not a validated physical twin. Append-only <out_root>/vNNN/.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import subprocess
import time
from pathlib import Path

import numpy as np

SCHEMA = "forceartgs.g9.estimated_joint_dynamics_asset/v2"    # v2: + the 3DGS parts
GAUSSIAN_DIR = "gaussians"
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


def rpy_from_matrix(R) -> np.ndarray:
    """URDF rpy (fixed axes, R = Rz(yaw) Ry(pitch) Rx(roll))."""
    R = np.asarray(R, float)
    cp = math.hypot(R[0, 0], R[1, 0])
    pitch = math.atan2(-R[2, 0], cp)
    if cp < 1e-9:                                                # gimbal lock: yaw folded into roll
        return np.array([math.atan2(-R[1, 2], R[1, 1]), pitch, 0.0])
    return np.array([math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])])


def matrix_from_rpy(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def gaussian_parts(handoff, hj: dict, m: dict) -> list:
    """The hand-off's 3DGS parts as link visuals (see module docstring). Fails closed on a missing PLY, a sha256
    different from the hand-off's, or a non-rigid T_Wc_P."""
    out = []
    for part, link in (("static", "static_link"), ("moving", "moving_link")):
        p = (hj.get("parts") or {}).get(part) or {}
        if not p.get("ply"):
            raise ValueError(f"hand-off has no {part} PLY")
        src = Path(handoff).parent / p["ply"]
        if not src.is_file():
            raise ValueError(f"{src}: missing")
        sha = _sha(src)
        if p.get("ply_sha256") and sha != p["ply_sha256"]:
            raise ValueError(f"{src}: sha256 {sha[:12]} differs from the hand-off's {p['ply_sha256'][:12]}")
        T_Wc_P = np.asarray(p["T_Wc_P"], float)
        R = T_Wc_P[:3, :3] if T_Wc_P.shape == (4, 4) else None
        if (R is None or np.abs(R.T @ R - np.eye(3)).max() > 1e-6 or np.linalg.det(R) < 0.999999
                or not np.allclose(T_Wc_P[3], [0, 0, 0, 1])):
            raise ValueError(f"{part}: T_Wc_P is not a rigid transform")
        T = T_Wc_P.copy()
        if link == "moving_link":
            T[:3, 3] -= m["origin"]                              # moving link frame: Wc axes at the joint origin
        rpy = rpy_from_matrix(T[:3, :3])
        if np.abs(matrix_from_rpy(rpy) - T[:3, :3]).max() > 1e-9:
            raise ValueError(f"{part}: rpy round trip failed")
        out.append(dict(part=part, link=link, src=str(src), file=f"{GAUSSIAN_DIR}/{src.name}", sha256=sha,
                        n_gaussians=p.get("n_gaussians"), fields=p.get("immutable_gaussian_fields"),
                        T_Wc_P=T_Wc_P.tolist(), T_link_P=T.tolist(), xyz=T[:3, 3].tolist(), rpy=rpy.tolist()))
    return out


def write_urdf(path: Path, name: str, m: dict, gaussians=()) -> None:
    I = m["inertia"]
    f = lambda v: " ".join(repr(float(x)) for x in v)
    jt = "continuous" if m["type"] == "revolute" else "prismatic"
    lim = "" if jt == "continuous" else '\n    <limit lower="-inf" upper="inf" effort="inf" velocity="inf"/>'
    vis = {g["link"]: f"""
    <visual name="{g['part']}_3dgs">
      <origin xyz="{f(g['xyz'])}" rpy="{f(g['rpy'])}"/>
      <geometry><mesh filename="{g['file']}"/></geometry>
    </visual>""" for g in gaussians}
    static = f'<link name="static_link">{vis["static_link"]}\n  </link>' if "static_link" in vis else '<link name="static_link"/>'
    path.write_text(f"""<?xml version="1.0"?>
<!-- ESTIMATED joint dynamics only (forceartgs_g9.export.estimated_asset): damping c and the moving-part inertia M
     along the joint; frame = capture world Wc (metres). No collision, stops, static inertial or friction.
     Visuals = the run's 3DGS parts (Gaussian PLYs in {GAUSSIAN_DIR}/, not triangle meshes), placed so the joint
     articulates the moving Gaussians as in the run. -->
<robot name="{name}">
  {static}
  <link name="moving_link">{vis.get("moving_link", "")}
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
T_mv = np.eye(4); T_mv[:3, :3] = R; T_mv[:3, 3] = o                # USD moving_link frame in Wc
for g in req["gaussians"]:
    parent = st if g["link"] == "static_link" else mv
    T = np.asarray(g["T_Wc_P"], float) if g["link"] == "static_link" else np.linalg.inv(T_mv) @ np.asarray(g["T_Wc_P"], float)
    gx = UsdGeom.Xform.Define(stage, parent.GetPath().AppendChild("gaussians_3dgs"))
    gx.AddTransformOp().Set(Gf.Matrix4d(*T.T.flatten().tolist()))       # Gf row-vector convention
    gp = gx.GetPrim()
    gp.CreateAttribute("forceartgs:gaussians_ply", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./" + g["file"]))
    gp.CreateAttribute("forceartgs:ply_sha256", Sdf.ValueTypeNames.String).Set(g["sha256"])
    gp.SetCustomDataByKey("forceartgs:frame", "3DGS PLY in the run's canonical frame P (moving part at q = 0); "
                                              "this Xform = T_link_P; not a triangle mesh")
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
gu = []
for g in req["gaussians"]:
    gp = s2.GetPrimAtPath(f"/{req['name']}/{g['link']}/gaussians_3dgs")
    W = np.array(UsdGeom.Xformable(gp).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float).T
    ap = gp.GetAttribute("forceartgs:gaussians_ply").Get()
    gu.append(dict(link=g["link"], ply=ap.path, resolved=bool(ap.resolvedPath),
                   max_err_T_Wc_P=float(np.abs(W - np.asarray(g["T_Wc_P"], float)).max())))
out["usd"]["gaussians"] = gu
# ---------------------------------------------------------------- SAPIEN (PhysX) constant-effort replay of the URDF
import sapien
scene = sapien.Scene(); dt = 1e-3; scene.set_timestep(dt)
ld = scene.create_urdf_loader(); ld.fix_root_link = True
art = ld.load(req["urdf"])
for l in art.get_links():                                  # engine defaults the identification zeroed (G10):
    l.disable_gravity = True; l.linear_damping = 0.0; l.angular_damping = 0.0   # gravity off, no link damping
j0 = art.get_active_joints()[0]; j0.set_friction(0.0)
# 3DGS visuals: engine pose (link pose x visual pose) vs the run's articulation, at q = 0 and at a moved q
def _rot(u, th):
    u = np.asarray(u, float) / np.linalg.norm(u); K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
shapes = {}
for l in art.get_links():
    for comp in l.entity.components:
        if isinstance(comp, sapien.render.RenderBodyComponent):
            for sh in comp.render_shapes:
                shapes[sh.name] = (l, sh)
gs = []
for qt in (0.0, 0.5 if m["type"] == "revolute" else 0.1):
    art.set_qpos(np.array([qt], np.float32))
    J_ = np.eye(4)
    if m["type"] == "revolute":
        J_[:3, :3] = _rot(a, qt); J_[:3, 3] = o - J_[:3, :3] @ o
    else:
        J_[:3, 3] = qt * a / np.linalg.norm(a)
    for g in req["gaussians"]:
        if g["part"] + "_3dgs" not in shapes:
            gs.append(dict(part=g["part"], q=qt, error="visual not loaded")); continue
        l, sh = shapes[g["part"] + "_3dgs"]
        W = l.entity_pose.to_transformation_matrix() @ sh.local_pose.to_transformation_matrix()
        E = (J_ if g["link"] == "moving_link" else np.eye(4)) @ np.asarray(g["T_Wc_P"], float)
        gs.append(dict(part=g["part"], q=qt, link=l.name, max_err=float(np.abs(W - E).max())))
art.set_qpos(np.zeros(1, np.float32)); art.set_qvel(np.zeros(1, np.float32))
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
                     gaussians=gs, Q=Q, T=T, dt=dt, v_end_sim=vs[-1], v_end_closed_form=float(v_ref[-1]),
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
    try:
        gauss = gaussian_parts(h, hj, m)
    except (ValueError, KeyError) as ex:
        return finish("BLOCKED_GAUSSIANS", f"3DGS parts of {h}: {ex}", handoff=str(h))
    (out / GAUSSIAN_DIR).mkdir()
    for g in gauss:                                              # byte copy: the folder exports on its own
        shutil.copyfile(g["src"], out / g["file"])
        if _sha(out / g["file"]) != g["sha256"]:
            return finish("BLOCKED_GAUSSIANS", f"copy of {g['src']} differs from the source", handoff=str(h))
    name = hj["object_id"]
    urdf, usd = out / "model.urdf", out / "model.usda"
    write_urdf(urdf, name, m, gauss)
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
                     "Coulomb / static friction", "armature", "triangle visual meshes (the visuals are the 3DGS PLYs)"],
        gaussians=dict(
            format="3DGS PLY, byte-identical to the run's hand-off (fields as listed; colour = SH degree 0)",
            frame="Gaussians in the run's canonical frame P; the moving part at joint value q = 0",
            placement="URDF <visual><origin> = T_link_P (xyz + rpy, R = Rz Ry Rx); x_Wc(q) = T_Wc_link(q) T_link_P x_P "
                      "(static_link = Wc; moving_link = joint frame at q = 0, axes parallel to Wc at the joint origin); "
                      "USD: Xform <link>/gaussians_3dgs = T_link_P in its own link frame, PLY in forceartgs:gaussians_ply",
            caveat="not a triangle mesh: URDF/USD viewers without a 3DGS renderer draw nothing for it",
            parts=[{k: v for k, v in g.items() if k != "src"} | dict(source=g["src"]) for g in gauss]),
        model=jm), indent=1, default=str))
    Q = float(m["damping"] * 0.1 if m["damping"] > 0 else m["M"])        # probe effort (only scales the response)
    T = float(min(2.0, max(0.2, 3 * m["M"] / m["damping"]))) if m["damping"] > 0 else 0.5
    req = dict(model=jm, name=name, urdf=str(urdf), usd=str(usd), provenance=prov, Q=Q, T=T,
               gaussians=[{k: g[k] for k in ("part", "link", "file", "sha256", "T_Wc_P")} for g in gauss])
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
    g_err = [e.get("max_err", math.inf) for e in ver["sapien"]["gaussians"]] + \
            [e["max_err_T_Wc_P"] if e["resolved"] else math.inf for e in ver["usd"]["gaussians"]]
    g_ok = len(g_err) == 3 * len(gauss) and max(g_err) < 1e-5               # float32 engine poses
    status = "EXPORTED" if usd_ok and sim_ok and g_ok else "EXPORTED_CHECK_FAILED"
    return finish(status, f"{m['type']} M={m['M']:.4g} c={m['damping']:.4g}; SAPIEN replay max rel err "
                          f"{ver['sapien']['max_rel_err_v']:.2%}; USD read-back {'OK' if usd_ok else 'MISMATCH'}; "
                          f"3DGS static+moving placed {'OK' if g_ok else 'MISMATCH'} (max err {max(g_err):.1e})",
                  gaussians=[out / g["file"] for g in gauss],
                  handoff=str(h), urdf=str(urdf), urdf_sha256=_sha(urdf), usd=str(usd), usd_sha256=_sha(usd),
                  verification=ver)
