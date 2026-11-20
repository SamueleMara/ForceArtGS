"""Standalone MuJoCo rollout (run with an interpreter that has `mujoco` + numpy; no ForceArtGS import).

    python mujoco_rollout.py model.xml request.json out.json

request: {gravity: [3] | null (null keeps the model option), timestep, integrator, effort_t_s, effort_levels,
          t_query, q0, v0}. The generalized effort is applied as qfrc_applied[0] (forward ZOH). Returns q, qdot at
t_query plus the compiled joint / inertial readback.
"""
import json
import sys

import mujoco
import numpy as np


def main(xml, req_path, out_json):
    r = json.load(open(req_path))
    m = mujoco.MjModel.from_xml_path(xml)
    if r.get("gravity") is not None:
        m.opt.gravity[:] = r["gravity"]
    m.opt.timestep = float(r.get("timestep", 1e-4))
    m.opt.integrator = getattr(mujoco.mjtIntegrator, r.get("integrator", "mjINT_IMPLICITFAST"))
    d = mujoco.MjData(m)
    d.qpos[0], d.qvel[0] = float(r["q0"]), float(r["v0"])
    mujoco.mj_forward(m, d)
    tf, lev = np.asarray(r["effort_t_s"], float), np.asarray(r["effort_levels"], float)
    tq = np.asarray(r["t_query"], float)
    t0 = float(r.get("t0", 0.0))
    q, v = np.empty(len(tq)), np.empty(len(tq))
    t = t0
    for i, tt in enumerate(tq):
        while t < tt - 1e-12:
            k = int(np.clip(np.searchsorted(tf, t + 1e-12, side="right") - 1, 0, len(lev) - 1))
            d.qfrc_applied[0] = lev[k]
            mujoco.mj_step(m, d)
            t += m.opt.timestep
        q[i], v[i] = d.qpos[0], d.qvel[0]
    j = m.joint(0)
    out = dict(mujoco_version=mujoco.__version__, q=q.tolist(), qdot=v.tolist(), timestep=m.opt.timestep,
               gravity=m.opt.gravity.tolist(), integrator=r.get("integrator", "mjINT_IMPLICITFAST"),
               joint=dict(name=j.name, damping=float(m.dof_damping[0]), frictionloss=float(m.dof_frictionloss[0]),
                          armature=float(m.dof_armature[0]), range=m.jnt_range[0].tolist()),
               moving_mass=float(m.body_mass[m.body("moving_link").id]))
    json.dump(out, open(out_json, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
