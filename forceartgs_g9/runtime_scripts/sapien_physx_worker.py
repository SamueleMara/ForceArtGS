"""G10 SAPIEN / PhysX CPU rollout worker (interpreter with `sapien` + numpy; NO ForceArtGS import).

One request = one clean PhysX scene: load the (already mass-scaled) single-joint URDF with a fixed root, zero every
passive term the engine adds by default, apply the generalized effort with ``set_qf`` every step (forward ZOH), and
return q / qdot at the query times plus a readback of what the engine actually uses.

    python sapien_physx_worker.py --serve              # JSON lines on stdin -> "@@G10@@ {json}" lines on stdout
    python sapien_physx_worker.py request.json out.json  # one shot

Engine defaults zeroed here (each one was measured on SAPIEN 3.0.3, see docs/G10_PHYSX_REFINEMENT_IMPLEMENTATION_REPORT):
  * joint ``friction`` 0.05 (legacy, dimensionless, load-proportional) applied by the URDF loader whatever the tag;
  * articulation-link linear AND angular damping 0.05 (PhysX PxArticulationLink default): a frictionless vertical
    hinge otherwise loses exp(-0.05 t) of its velocity;
  * articulation sleeping (a slow coast would freeze below the sleep threshold, which looks like stiction);
  * drive stiffness / damping.

Friction modes (``friction_mode``):
  * ``frictionless``             : no passive term at all (native PhysX integration of inertia + gravity + input);
  * ``external_effort_surrogate``: an EXPLICIT input  -c v - Td sat(v / v_eps), v_eps = Td dt / I_eff  (the velocity
    change one step of Coulomb friction produces; the saturation stops step-to-step chatter through v = 0).  It is
    NOT PhysX joint friction, has no stiction branch and never certifies native parity.
"""
import json
import sys

import numpy as np

MARK = "@@G10@@ "


def _zoh_index(tf, t):
    return int(np.searchsorted(tf, t + 1e-12, side="right") - 1)


NATIVE_FRICTION_API = ("set_static_friction_effort", "set_dynamic_friction_effort", "set_viscous_friction_coefficient",
                       "static_friction_effort", "dynamic_friction_effort", "viscous_friction_coefficient")


def capabilities():
    """What THIS installed engine exposes (checked, never assumed from documentation of newer PhysX / Omni APIs)."""
    import sapien
    import sapien.physx as px
    J = px.PhysxArticulationJoint
    A = px.PhysxArticulation
    L = px.PhysxArticulationLinkComponent
    has = {m: hasattr(J, m) for m in ("set_friction", "set_armature", "set_drive_properties", "set_drive_target",
                                      "set_drive_velocity_target") + NATIVE_FRICTION_API}
    native = all(hasattr(J, m) for m in NATIVE_FRICTION_API[:3])
    return dict(sapien_version=sapien.__version__, joint_api=has, native_axis_friction_api=native,
                link_api={m: hasattr(L, m) for m in ("set_linear_damping", "set_angular_damping", "sleeping",
                                                     "put_to_sleep", "set_inertia", "set_cmass_local_pose")},
                articulation_api={m: hasattr(A, m) for m in ("set_sleep_threshold", "compute_passive_force",
                                                             "get_solver_position_iterations")},
                scene_config_fields=[m for m in dir(px.PhysxSceneConfig()) if not m.startswith("_")],
                legacy_joint_friction_semantics="set_friction(float): legacy dimensionless, load-proportional "
                                                "coefficient -- NOT a Coulomb effort in N m / N",
                friction_coverage=("NATIVE_AXIS_FRICTION_AVAILABLE" if native else
                                   "NATIVE_AXIS_FRICTION_NOT_EXPOSED (diagnostic explicit-effort surrogate only)"),
                backend="PhysX CPU (sapien.physx.PhysxCpuSystem)")


def _full_inertia(principal, q_wxyz):
    w, x, y, z = q_wxyz
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R @ np.diag(principal) @ R.T


def run_request(r):
    if r.get("kind") == "capabilities":
        return capabilities()
    if r.get("kind") == "batch_gpu":
        return run_batch_gpu(r)
    import sapien
    import sapien.physx as px
    dt = float(r["timestep"])
    if not dt > 0:
        raise ValueError("timestep must be > 0")
    mode = r["friction_mode"]
    if mode not in ("frictionless", "external_effort_surrogate"):
        raise ValueError(f"friction_mode {mode!r}")
    c, Td = float(r.get("viscous", 0.0)), float(r.get("dynamic_effort", 0.0))
    if mode == "frictionless" and (c != 0.0 or Td != 0.0):
        raise ValueError("frictionless mode with nonzero friction")
    I_eff = float(r.get("effective_inertia") or 0.0)
    if Td > 0 and not I_eff > 0:
        raise ValueError("Coulomb surrogate needs the effective joint inertia (velocity regularisation)")
    cfg = px.PhysxSceneConfig()
    cfg.gravity = np.asarray(r["gravity"], np.float32)
    if r.get("enhanced_determinism", True):
        cfg.enable_enhanced_determinism = True
    px.set_scene_config(cfg)
    scene = sapien.Scene([px.PhysxCpuSystem()])
    scene.set_timestep(dt)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    art = loader.load(r["urdf"])
    if art is None:
        raise RuntimeError("SAPIEN failed to load the URDF")
    joints = [j for j in art.active_joints if j.name == r["joint_name"]]
    if len(joints) != 1 or len(art.active_joints) != 1:
        raise ValueError("G10 runner requires exactly one active joint named " + repr(r["joint_name"]))
    j = joints[0]
    links = {ln.name: ln for ln in art.links}
    if r["moving_link"] not in links:
        raise ValueError(f"moving link {r['moving_link']!r} not in the articulation")
    ml = links[r["moving_link"]]
    loaded = dict(joint_friction=float(j.friction), joint_damping=float(j.damping), joint_stiffness=float(j.stiffness),
                  link_linear_damping={k: float(v.linear_damping) for k, v in links.items()},
                  link_angular_damping={k: float(v.angular_damping) for k, v in links.items()},
                  sleep_threshold=float(art.sleep_threshold))
    j.set_drive_properties(0.0, 0.0)
    j.set_friction(0.0)
    j.set_armature(np.array([float(r.get("armature", 0.0))], np.float32))
    for ln in links.values():
        ln.set_linear_damping(0.0)
        ln.set_angular_damping(0.0)
    art.set_sleep_threshold(0.0)
    p = ml.cmass_local_pose
    rb = dict(joint_friction=float(j.friction), joint_damping=float(j.damping), joint_stiffness=float(j.stiffness),
              armature=np.asarray(j.armature, float).tolist(),
              link_linear_damping={k: float(v.linear_damping) for k, v in links.items()},
              link_angular_damping={k: float(v.angular_damping) for k, v in links.items()},
              sleep_threshold=float(art.sleep_threshold), moving_mass=float(ml.mass),
              moving_principal_inertia=np.asarray(ml.inertia, float).tolist(),
              moving_cmass_local_p=np.asarray(p.p, float).tolist(), moving_cmass_local_q_wxyz=np.asarray(p.q, float).tolist(),
              joint_type=str(j.type), qlimits=np.asarray(art.qlimit, float).tolist(),
              drive_target=np.asarray(j.drive_target, float).tolist(),
              drive_velocity_target=np.asarray(j.drive_velocity_target, float).tolist(),
              timestep=float(scene.get_timestep()), gravity=np.asarray(scene.physx_system.config.gravity,
                                                                       float).tolist(),
              solver_iterations=[int(art.solver_position_iterations), int(art.solver_velocity_iterations)])
    Jm = _full_inertia(rb["moving_principal_inertia"], rb["moving_cmass_local_q_wxyz"])
    rb["moving_inertia_tensor_cmass_link_frame"] = Jm.tolist()
    rb["moving_inertia_spd"] = bool(np.min(np.linalg.eigvalsh(0.5 * (Jm + Jm.T))) > 0)
    bad = [k for k in ("joint_friction", "joint_damping", "joint_stiffness", "sleep_threshold") if abs(rb[k]) > 1e-9]
    bad += [f"{k}:{n}" for k in ("link_linear_damping", "link_angular_damping") for n, v in rb[k].items() if abs(v) > 1e-9]
    if abs(rb["timestep"] - dt) > 1e-6 * dt:
        bad.append(f"timestep {rb['timestep']} != {dt}")
    if np.max(np.abs(np.asarray(rb["gravity"]) - np.asarray(r["gravity"], float))) > 1e-5:
        bad.append(f"gravity {rb['gravity']} != {r['gravity']}")
    if not rb["moving_inertia_spd"] or not rb["moving_mass"] > 0:
        bad.append("moving link mass / inertia not physical")
    if bad:
        raise RuntimeError(f"passive engine terms still nonzero after zeroing: {bad}")
    t0 = float(r["t0"])
    tq = np.asarray(r["t_query"], float)
    steps = np.rint((tq - t0) / dt).astype(np.int64)
    if steps[0] < 0 or np.any(np.diff(steps) < 0):
        raise ValueError("query times must be >= t0 and nondecreasing")
    tf = np.asarray(r.get("effort_t_s", [t0]), float)
    lev = np.asarray(r.get("effort_levels", [0.0]), float)
    if len(tf) != len(lev) or len(tf) == 0:
        raise ValueError("effort timeline / levels mismatch")
    if _zoh_index(tf, t0) < 0:
        raise ValueError("effort starts after t0 (no extrapolation)")
    art.set_qpos(np.array([float(r["q0"])], np.float32))
    art.set_qvel(np.array([float(r["v0"])], np.float32))
    v_eps = Td * dt / I_eff if Td > 0 else 0.0
    nq = len(tq)
    q, v = np.empty(nq), np.empty(nq)
    J_ext, J_fric = np.empty(nq), np.empty(nq)
    je = jf = 0.0
    n_contacts = 0
    k = 0
    for n in range(int(steps[-1]) + 1):
        while k < nq and steps[k] == n:
            q[k], v[k] = float(art.get_qpos()[0]), float(art.get_qvel()[0])
            J_ext[k], J_fric[k] = je, jf
            k += 1
        if n == steps[-1]:
            break
        t = t0 + n * dt
        Qe = float(lev[_zoh_index(tf, t)])
        Qf = 0.0
        if mode == "external_effort_surrogate":
            vel = float(art.get_qvel()[0])
            Qf = -c * vel
            if Td > 0:
                Qf -= Td * max(-1.0, min(1.0, vel / v_eps))
        art.set_qf(np.array([Qe + Qf], np.float32))
        scene.step()
        n_contacts += len(scene.physx_system.get_contacts())
        je += Qe * dt
        jf += Qf * dt
    return dict(sapien_version=sapien.__version__, q=q.tolist(), qdot=v.tolist(), timestep=dt, n_steps=int(steps[-1]),
                gravity=list(map(float, cfg.gravity)), friction_mode=mode, v_eps=v_eps,
                external_impulse=J_ext.tolist(), friction_impulse=J_fric.tolist(),
                engine_defaults_as_loaded=loaded, readback=rb, n_contacts=int(n_contacts),
                moving_link_sleeping_at_end=bool(ml.sleeping) if hasattr(ml, "sleeping") else None,
                runtime_semantics=("EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE" if mode == "external_effort_surrogate"
                                   else "FRICTIONLESS_NATIVE_PHYSX"))


def run_batch_gpu(r):
    """BATCHED rollout on PhysX GPU (sapien.physx.PhysxGpuSystem): one sub-scene per candidate (its own mass-scaled
    URDF, viscous / Coulomb surrogate, armature), ONE shared episode (q0, v0, t0, query times, ZOH effort). Same step
    law as ``run_request`` (read qvel -> explicit surrogate -c v - Td sat(v/v_eps) + recorded effort -> set qf -> step)
    and the same passive-term zeroing + readback, per candidate. Links carry no collision shapes (checked)."""
    import sapien
    import sapien.physx as px
    import torch
    dt = float(r["timestep"])
    mode = r["friction_mode"]
    if mode not in ("frictionless", "external_effort_surrogate"):
        raise ValueError(f"friction_mode {mode!r}")
    cands = r["candidates"]
    n_c = len(cands)
    if n_c == 0:
        raise ValueError("empty batch")
    c = np.array([float(k.get("viscous", 0.0)) for k in cands])
    Td = np.array([float(k.get("dynamic_effort", 0.0)) for k in cands])
    I_eff = np.array([float(k.get("effective_inertia") or 0.0) for k in cands])
    if mode == "frictionless" and (np.any(c != 0) or np.any(Td != 0)):
        raise ValueError("frictionless mode with nonzero friction")
    if np.any((Td > 0) & ~(I_eff > 0)):
        raise ValueError("Coulomb surrogate needs the effective joint inertia")
    px.enable_gpu()
    cfg = px.PhysxSceneConfig()
    cfg.gravity = np.asarray(r["gravity"], np.float32)
    px.set_scene_config(cfg)
    sysg = px.PhysxGpuSystem(device=r.get("device", "cuda"))
    sysg.set_timestep(dt)
    scenes, rbs, loaded = [], [], []
    for i, k in enumerate(cands):
        sc = sapien.Scene([sysg])
        sysg.set_scene_offset(sc, [float(i) * 20.0, 0.0, 0.0])
        ld = sc.create_urdf_loader()
        ld.fix_root_link = True
        art = ld.load(k["urdf"])
        if art is None:
            raise RuntimeError(f"SAPIEN failed to load {k['urdf']}")
        joints = [j for j in art.active_joints if j.name == r["joint_name"]]
        if len(joints) != 1 or len(art.active_joints) != 1:
            raise ValueError("G10 runner requires exactly one active joint named " + repr(r["joint_name"]))
        j = joints[0]
        links = {ln.name: ln for ln in art.links}
        ml = links[r["moving_link"]]
        loaded.append(dict(joint_friction=float(j.friction), joint_damping=float(j.damping),
                           link_linear_damping={n: float(v.linear_damping) for n, v in links.items()},
                           link_angular_damping={n: float(v.angular_damping) for n, v in links.items()}))
        j.set_drive_properties(0.0, 0.0)
        j.set_friction(0.0)
        j.set_armature(np.array([float(k.get("armature", 0.0))], np.float32))
        for ln in links.values():
            ln.set_linear_damping(0.0)
            ln.set_angular_damping(0.0)
        p = ml.cmass_local_pose
        n_shapes = sum(len(ln.collision_shapes) for ln in links.values())
        rb = dict(joint_friction=float(j.friction), joint_damping=float(j.damping), joint_stiffness=float(j.stiffness),
                  armature=np.asarray(j.armature, float).tolist(),
                  link_linear_damping={n: float(v.linear_damping) for n, v in links.items()},
                  link_angular_damping={n: float(v.angular_damping) for n, v in links.items()},
                  sleep_threshold=0.0, moving_mass=float(ml.mass),
                  moving_principal_inertia=np.asarray(ml.inertia, float).tolist(),
                  moving_cmass_local_p=np.asarray(p.p, float).tolist(),
                  moving_cmass_local_q_wxyz=np.asarray(p.q, float).tolist(), joint_type=str(j.type),
                  qlimits=np.asarray(art.qlimit, float).tolist(),
                  drive_target=np.asarray(j.drive_target, float).tolist(),
                  drive_velocity_target=np.asarray(j.drive_velocity_target, float).tolist(),
                  timestep=float(sysg.get_timestep()), gravity=np.asarray(px.get_scene_config().gravity, float).tolist(),
                  solver_iterations=[int(art.solver_position_iterations), int(art.solver_velocity_iterations)],
                  n_collision_shapes=int(n_shapes), sleeping="NOT_APPLICABLE_GPU (no sleeping on the GPU pipeline)")
        Jm = _full_inertia(rb["moving_principal_inertia"], rb["moving_cmass_local_q_wxyz"])
        rb["moving_inertia_tensor_cmass_link_frame"] = Jm.tolist()
        rb["moving_inertia_spd"] = bool(np.min(np.linalg.eigvalsh(0.5 * (Jm + Jm.T))) > 0)
        bad = [kk for kk in ("joint_friction", "joint_damping", "joint_stiffness") if abs(rb[kk]) > 1e-9]
        bad += [f"{kk}:{n}" for kk in ("link_linear_damping", "link_angular_damping") for n, v in rb[kk].items()
                if abs(v) > 1e-9]
        if abs(rb["timestep"] - dt) > 1e-6 * dt:
            bad.append(f"timestep {rb['timestep']} != {dt}")
        if np.max(np.abs(np.asarray(rb["gravity"]) - np.asarray(r["gravity"], float))) > 1e-5:
            bad.append(f"gravity {rb['gravity']} != {r['gravity']}")
        if not rb["moving_inertia_spd"] or not rb["moving_mass"] > 0:
            bad.append("moving link mass / inertia not physical")
        if n_shapes:
            bad.append(f"{n_shapes} collision shape(s): the isolated joint replay must be contact-free by construction")
        if bad:
            raise RuntimeError(f"candidate {i}: passive engine terms / setup not clean after zeroing: {bad}")
        scenes.append(sc)
        rbs.append(rb)
    sysg.gpu_init()
    dev = torch.device(r.get("device", "cuda"))
    qpos, qvel, qf = (sysg.cuda_articulation_qpos.torch(), sysg.cuda_articulation_qvel.torch(),
                      sysg.cuda_articulation_qf.torch())
    if qpos.shape[0] != n_c:
        raise RuntimeError(f"GPU articulation count {qpos.shape[0]} != {n_c}")
    t0 = float(r["t0"])
    tq = np.asarray(r["t_query"], float)
    steps = np.rint((tq - t0) / dt).astype(np.int64)
    if steps[0] < 0 or np.any(np.diff(steps) < 0):
        raise ValueError("query times must be >= t0 and nondecreasing")
    tf = np.asarray(r.get("effort_t_s", [t0]), float)
    lev = np.asarray(r.get("effort_levels", [0.0]), float)
    if len(tf) != len(lev) or len(tf) == 0 or _zoh_index(tf, t0) < 0:
        raise ValueError("effort timeline invalid or starts after t0 (no extrapolation)")
    qpos[:, 0] = float(r["q0"])
    qvel[:, 0] = float(r["v0"])
    sysg.gpu_apply_articulation_qpos()
    sysg.gpu_apply_articulation_qvel()
    sysg.gpu_update_articulation_kinematics()
    c_t = torch.tensor(c, dtype=torch.float32, device=dev)
    Td_t = torch.tensor(Td, dtype=torch.float32, device=dev)
    veps = torch.tensor(np.where(Td > 0, Td * dt / np.maximum(I_eff, 1e-30), 1.0), dtype=torch.float32, device=dev)
    fric = mode == "external_effort_surrogate"
    nq = len(tq)
    Q, V = np.empty((n_c, nq)), np.empty((n_c, nq))
    Je, Jf = np.zeros((n_c, nq)), np.zeros((n_c, nq))
    je = torch.zeros(n_c, dtype=torch.float64, device=dev)
    jf = torch.zeros(n_c, dtype=torch.float64, device=dev)
    k = 0
    for n in range(int(steps[-1]) + 1):
        need_state = fric or (k < nq and steps[k] == n)
        if need_state:
            sysg.gpu_fetch_articulation_qvel()
        if k < nq and steps[k] == n:
            sysg.gpu_fetch_articulation_qpos()
            qp, qv = qpos[:, 0].double().cpu().numpy(), qvel[:, 0].double().cpu().numpy()
            jeh, jfh = je.cpu().numpy(), jf.cpu().numpy()
            while k < nq and steps[k] == n:
                Q[:, k], V[:, k], Je[:, k], Jf[:, k] = qp, qv, jeh, jfh
                k += 1
        if n == steps[-1]:
            break
        t = t0 + n * dt
        Qe = float(lev[_zoh_index(tf, t)])
        if fric:
            v = qvel[:, 0]
            Qf = -c_t * v - Td_t * torch.clamp(v / veps, -1.0, 1.0)
        else:
            Qf = torch.zeros(n_c, dtype=torch.float32, device=dev)
        qf[:, 0] = Qe + Qf
        sysg.gpu_apply_articulation_qf()
        sysg.step()
        je += Qe * dt
        jf += Qf.double() * dt
    return dict(sapien_version=sapien.__version__, backend="PhysX GPU (sapien.physx.PhysxGpuSystem, CUDA)",
                device=str(dev), n_candidates=n_c, q=Q.tolist(), qdot=V.tolist(), timestep=dt,
                n_steps=int(steps[-1]), gravity=list(map(float, cfg.gravity)), friction_mode=mode,
                v_eps=[float(x) if t_ > 0 else 0.0 for x, t_ in zip(veps.cpu().numpy(), Td)],
                external_impulse=Je.tolist(), friction_impulse=Jf.tolist(), engine_defaults_as_loaded=loaded,
                readback=rbs, n_contacts=0, contact_note="no collision shapes on any link (checked per candidate)",
                runtime_semantics=("EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE" if fric else "FRICTIONLESS_NATIVE_PHYSX"))


def serve():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            out = dict(ok=True, result=run_request(json.loads(line)))
        except Exception as e:                                         # reported to the caller, never swallowed
            out = dict(ok=False, error=f"{type(e).__name__}: {e}")
        sys.stdout.write(MARK + json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--serve"]:
        serve()
    else:
        json.dump(run_request(json.load(open(sys.argv[1]))), open(sys.argv[2], "w"))
