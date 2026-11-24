"""Minimal single-joint URDF for the G10 estimator rollouts (a DERIVED estimator input, never an export asset).

Frames: the base link is the G9 record frame B (the frame of the accepted joint axis / origin, the mass geometry and
``g_B``), so the scene gravity is ``g_B`` itself. The joint frame sits at the hinge origin with identity rotation; the
child link frame therefore equals B translated to o at q = 0. The G9 mass geometry is posed at the stored coordinate
``q_ref``; the URDF needs the pose at q = 0:

    revolute : c(0) = o + R_a(-q_ref)(c_ref - o),   J(0) = R_a(-q_ref) J_ref R_a(-q_ref)^T
    prismatic: c(0) = c_ref - q_ref a,              J(0) = J_ref

so PhysX's own gravity torque equals the G9 RigidBodyGravity load G(q) at every q (tested). The moving link has
inertia only (no collision / visual shapes): nothing can touch it. One density ``rho`` scales m and J together.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..dynamics.general_1dof import rotation


def _f(x):
    return f"{float(x):.12g}"


def pose_at_zero(joint_type, axis, origin, com_ref, J_ref, q_ref):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    c = np.asarray(com_ref, float)
    J = np.asarray(J_ref, float)
    if joint_type == "revolute":
        o = np.asarray(origin, float)
        R = rotation(a, -float(q_ref))
        return a, o, o + R @ (c - o), R @ J @ R.T
    o = np.zeros(3)
    return a, o, c - float(q_ref) * a, J


def write_single_joint_urdf(path, *, joint_type, axis, origin, mass, com_ref, inertia_com_ref, q_ref=0.0,
                            name="g10_object", moving_link="moving_link", joint_name="joint",
                            prismatic_limit_m=100.0) -> dict:
    if joint_type not in ("revolute", "prismatic"):
        raise ValueError(f"joint_type {joint_type!r}")
    if not (mass > 0 and np.isfinite(mass)):
        raise ValueError("positive finite mass required")
    a, o, c0, J0 = pose_at_zero(joint_type, axis, origin, com_ref, inertia_com_ref, q_ref)
    J0 = 0.5 * (J0 + J0.T)
    ev = np.linalg.eigvalsh(J0)
    if ev.min() < -1e-12 * max(1.0, ev.max()) or ev[0] + ev[1] < ev[2] * (1 - 1e-9):
        raise ValueError(f"inertia tensor not physical (eigenvalues {ev})")
    r = c0 - o
    jt = "continuous" if joint_type == "revolute" else "prismatic"
    lim = "" if joint_type == "revolute" else (f'<limit lower="{-prismatic_limit_m}" upper="{prismatic_limit_m}" '
                                               'effort="1e9" velocity="1e9"/>')
    xml = f"""<?xml version="1.0"?>
<!-- G10 estimator input (derived; NOT an export asset). Frame = G9 record frame B. q_ref = {_f(q_ref)} -->
<robot name="{name}">
  <link name="base"/>
  <link name="{moving_link}">
    <inertial>
      <origin xyz="{_f(r[0])} {_f(r[1])} {_f(r[2])}" rpy="0 0 0"/>
      <mass value="{_f(mass)}"/>
      <inertia ixx="{_f(J0[0, 0])}" iyy="{_f(J0[1, 1])}" izz="{_f(J0[2, 2])}" ixy="{_f(J0[0, 1])}" ixz="{_f(J0[0, 2])}" iyz="{_f(J0[1, 2])}"/>
    </inertial>
  </link>
  <joint name="{joint_name}" type="{jt}">
    <parent link="base"/>
    <child link="{moving_link}"/>
    <origin xyz="{_f(o[0])} {_f(o[1])} {_f(o[2])}" rpy="0 0 0"/>
    <axis xyz="{_f(a[0])} {_f(a[1])} {_f(a[2])}"/>
    {lim}
    <dynamics damping="0" friction="0"/>
  </joint>
</robot>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)
    rp = r - (r @ a) * a
    Ih = float(a @ J0 @ a + mass * rp @ rp) if joint_type == "revolute" else float(mass)
    return dict(path=str(path), sha256=hashlib.sha256(xml.encode()).hexdigest(), joint_type=joint_type,
                moving_link=moving_link, joint_name=joint_name, mass=float(mass), com_q0=c0.tolist(),
                inertia_com_q0=J0.tolist(), axis=a.tolist(), origin=o.tolist(), q_ref=float(q_ref),
                axis_inertia=Ih, frame="G9 record frame B (base link at identity)")


def from_mass_model(mm, rho, path, **kw) -> dict:
    """URDF at density ``rho`` from a G9 GeometryMassModel (unit-density m1, c1, J1 posed at mm.q_ref)."""
    if mm.reflected() != 0.0:
        raise ValueError("declared reflected actuator inertia: pass it as G10 armature, not baked into the link")
    pr = mm.props
    info = write_single_joint_urdf(path, joint_type=mm.joint_type, axis=mm.axis,
                                   origin=(mm.origin if mm.joint_type == "revolute" else np.zeros(3)),
                                   mass=float(rho) * pr.m1, com_ref=pr.c1, inertia_com_ref=float(rho) * pr.J1,
                                   q_ref=mm.q_ref, **kw)
    info.update(density=float(rho), density_units=pr.density_units, mass_mode=pr.mode, mesh_sha256=pr.mesh_sha256,
                M1=float(mm.M1), axis_inertia_check=float(rho) * float(mm.M1))
    return info
