"""Standalone SAPIEN / ManiSkill rollout (interpreter with `sapien` + numpy; no ForceArtGS import).

    python sapien_rollout.py model.urdf request.json out.json

request: {T_world_root, gravity [3], timestep, effort_t_s, effort_levels, t_query, q0, v0}. URDF loaded with
fix_root_link; the generalized effort is applied with set_qf every step (forward ZOH). Reads back the joint's
damping / friction as SAPIEN interpreted the URDF <dynamics> tag (empirical semantics check).
"""
import json
import sys

import numpy as np


def main(urdf, req_path, out_json):
    import sapien
    import sapien.physx as px
    r = json.load(open(req_path))
    cfg = px.PhysxSceneConfig()
    cfg.gravity = np.asarray(r["gravity"], np.float32)
    px.set_scene_config(cfg)
    scene = sapien.Scene([px.PhysxCpuSystem()])
    dt = float(r.get("timestep", 1e-3))
    scene.set_timestep(dt)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    art = loader.load(urdf)
    art.set_root_pose(sapien.Pose(np.asarray(r["T_world_root"], float)))
    j = art.active_joints[0]
    loaded_friction = float(j.friction)
    if r.get("diagnostic_joint_friction_override") is not None:      # diagnostic only; the asset is not changed
        j.set_friction(float(r["diagnostic_joint_friction_override"]))
    art.set_qpos(np.array([r["q0"]], np.float32))
    art.set_qvel(np.array([r["v0"]], np.float32))
    tf, lev = np.asarray(r["effort_t_s"], float), np.asarray(r["effort_levels"], float)
    tq = np.asarray(r["t_query"], float)
    t = float(r.get("t0", 0.0))
    q, v = np.empty(len(tq)), np.empty(len(tq))
    for i, tt in enumerate(tq):
        while t < tt - 1e-12:
            k = int(np.clip(np.searchsorted(tf, t + 1e-12, side="right") - 1, 0, len(lev) - 1))
            art.set_qf(np.array([lev[k]], np.float32))
            scene.step()
            t += dt
        q[i], v[i] = float(art.get_qpos()[0]), float(art.get_qvel()[0])
    out = dict(sapien_version=sapien.__version__, q=q.tolist(), qdot=v.tolist(), timestep=dt,
               gravity=list(map(float, cfg.gravity)),
               joint=dict(name=j.name, damping=float(j.damping), friction=loaded_friction,
                          friction_used=float(j.friction),
                          drive_mode=str(j.drive_mode), armature=np.asarray(j.armature).tolist()),
               moving_mass=float([l for l in art.links if l.name == "moving_link"][0].mass))
    json.dump(out, open(out_json, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
