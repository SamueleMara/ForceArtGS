"""final_physics_export: the LAST, atomic physics build (G9 spec 0: visual_handoff -> dynamics -> final_physics_export).

Calls the shared PArtTwin1D builder (``parttwin1d.cli.build_asset``) with the G8 visual handoff and the G9
``physics_spec_v1.yaml`` -- the same canonical exporter G8 uses; no simulator file is written here. Then:
  * reads the builder's own validation (URDF / MJCF compile, USD readback, SAPIEN load, visual PLY bitwise);
  * proves the two standard PLYs are byte-identical to the G8 visual handoff (D01);
  * runs numerical dynamics parity in MuJoCo and SAPIEN on a WITHHELD effort (D05, D09);
  * sets the remaining readiness states; FULL_PHYSICS_READY needs every enabled term represented and parity PASS.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

PARTTWIN_ROOT = Path(os.environ.get("PARTTWIN1D_ROOT", str(Path.home() / "PArtTwin1D")))
PARTTWIN_PY = os.environ.get("PARTTWIN1D_PYTHON", str(Path.home() / "miniconda3/envs/parttwin_sim/bin/python"))
SIM_PY = os.environ.get("PARTTWIN_SIM_PYTHON", str(Path.home() / ".venvs/rigeuler-sim-export/bin/python"))


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def payload_sha256(ply) -> str:
    """SHA-256 of the vertex payload (dtype descriptor + every stored field of every row); the builder only adds
    provenance header comments, which are not Gaussian data."""
    from plyfile import PlyData
    v = PlyData.read(str(ply))["vertex"].data
    h = hashlib.sha256(str(v.dtype.descr).encode())
    h.update(v.tobytes())
    return h.hexdigest()


def _git(root, *a):
    try:
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def build_asset(handoff, physics_spec, out_root, revision, *, allow_diagnostic=False, log=print) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PARTTWIN_ROOT / "src")
    env["PARTTWIN_SIM_PYTHON"] = SIM_PY
    env.pop("FORCEARTGS_JOINT_AWARE_TRIPWIRE", None)
    cmd = [PARTTWIN_PY, "-m", "parttwin1d.cli.build_asset", "--handoff", str(handoff), "--output", str(out_root),
           "--dynamics", str(physics_spec), "--revision", revision, "--sapien-python", PARTTWIN_PY]
    if allow_diagnostic:
        cmd.append("--allow-diagnostic")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    rec = dict(cmd=cmd, returncode=p.returncode, stdout=p.stdout[-6000:], stderr=p.stderr[-6000:],
               parttwin=dict(root=str(PARTTWIN_ROOT), head=_git(PARTTWIN_ROOT, "rev-parse", "HEAD"),
                             branch=_git(PARTTWIN_ROOT, "rev-parse", "--abbrev-ref", "HEAD"),
                             dirty=len((_git(PARTTWIN_ROOT, "status", "--porcelain") or "").splitlines())))
    if p.returncode != 0:
        rec["status"] = "FAILED"
        return rec
    info = json.loads(p.stdout[p.stdout.index("{"):])
    rec.update(status="BUILT", asset_dir=info["asset_dir"], asset_status=info["status"], missing=info["missing"])
    return rec


def readiness_from_asset(asset_dir, handoff) -> dict:
    a = Path(asset_dir)
    man = json.loads((a / "export_manifest.json").read_text()) if (a / "export_manifest.json").is_file() else None
    if man is None:
        cand = sorted(a.glob("*manifest*.json"))
        man = json.loads(cand[0].read_text()) if cand else {}
    val, rd = man.get("validation", {}), man.get("readiness", {})
    h = json.loads(Path(handoff).read_text())
    base = Path(handoff).parent
    bitwise = {}
    for part, link in (("static", "static"), ("moving", "moving")):
        src = base / h["parts"][part]["ply"]
        dst = a / "visual" / f"{link}.ply"
        bitwise[part] = dst.is_file() and payload_sha256(src) == payload_sha256(dst)
    return dict(VISUAL_ASSET_READY=bool(rd.get("visual_complete")) and all(bitwise.values()),
                visual_ply_identical_to_handoff=bitwise,
                COLLISION_READY=bool(val.get("urdf", {}).get("status") == "PASS"),
                URDF_COMPILES=val.get("urdf", {}).get("status") == "PASS",
                MJCF_COMPILES=val.get("mjcf", {}).get("status") == "PASS",
                USD_LOADS=val.get("usd", {}).get("status") == "PASS",
                SAPIEN_LOADS=val.get("sapien_parttwin1d", {}).get("status") == "PASS",
                builder_status=man.get("status"), builder_readiness=rd,
                builder_validation_status={k: v.get("status") for k, v in val.items() if isinstance(v, dict)})


def final_physics_export(g9_dir, handoff, out_root, *, parity_effort=None, parity_q0=0.0, log=print) -> dict:
    """``parity_effort``: a withheld generalized effort (forward ZOH) not used in the fit."""
    from ..validation.simulator_parity import numerical_parity, ENGINE_GRAVITY_W
    from ..dynamics.general_1dof import GeneralTerms
    from .dynamics_v2 import _clean
    g9 = Path(g9_dir)
    man = json.loads((g9 / "g9_manifest.json").read_text())
    res = dict(schema="forceartgs.g9.final_physics_export/v1", g9_signature=man["input_signature"])
    spec = g9 / "physics_spec_v1.yaml"
    if man.get("physics_spec_v1") is None or not spec.is_file():
        res.update(status="BLOCKED_MISSING_PHYSICS", reason="no production physics spec from G9",
                   missing=man.get("missing_for_physics_spec_v1"), blocked=man.get("blocked"))
        (g9 / "final_physics_export.json").write_text(json.dumps(_clean(res), indent=1))
        return res
    rev = "g9-" + man["input_signature"][:12]
    b = build_asset(handoff, spec, out_root, rev, log=log)
    res["build"] = b
    if b["status"] != "BUILT":
        res["status"] = "BUILD_FAILED"
        (g9 / "final_physics_export.json").write_text(json.dumps(_clean(res), indent=1))
        return res
    rd = readiness_from_asset(b["asset_dir"], handoff)
    res["asset"] = rd
    params = json.loads((g9 / "physics_parameters.json").read_text())
    phys = params["physical"]
    terms = GeneralTerms(**{t: man["terms"][t] == "enabled" for t in man["terms"]})
    import yaml
    dyn = yaml.safe_load((Path(b["asset_dir"]) / "physics" / "dynamics.yaml").read_text())
    mov = dyn["links"]["moving_link"]
    par = {}
    if parity_effort is not None:
        from ..dynamics.general_1dof import RigidBodyGravity
        from ..estimators.model_spec import fast_gravity
        asset = yaml.safe_load((Path(b["asset_dir"]) / "asset.yaml").read_text())
        T_W_B = np.asarray(asset["frames"]["T_world_root"], float)
        jt = params["joint_type"]
        hand = json.loads(Path(handoff).read_text())
        u = np.asarray(hand["joint"]["axis_Wc"], float)
        o = np.asarray(hand["joint"]["origin_Wc"] or [0, 0, 0], float)
        recon = json.loads((g9 / "inertia_reconstruction_report.json").read_text())
        g_capture = T_W_B[:3, :3].T @ np.asarray(ENGINE_GRAVITY_W, float)     # engine gravity seen in Wc
        rb = RigidBodyGravity(jt, tuple(u), tuple(o), tuple(recon["com_m"]), recon["mass_kg"], tuple(g_capture))
        A, B = rb.basis_coefficients()
        G = fast_gravity(jt, A, B)
        tq = np.arange(0.0, float(parity_effort.t_s[-1]) + 0.01, 1.0 / 60.0)
        for eng, model, py in (("mujoco", Path(b["asset_dir"]) / "mujoco" / "model.xml", SIM_PY),
                               ("sapien_urdf", Path(b["asset_dir"]) / "maniskill" / "model.urdf", PARTTWIN_PY)):
            par[eng] = numerical_parity(eng, model, py, joint_type=jt, M=phys["M"], c=phys["c"] or 0.0,
                                        Fc=phys["Fc"] or 0.0, Fs=phys["Fs"], G=G, terms=terms, effort=parity_effort,
                                        t_query=tq, q0=parity_q0, gravity=ENGINE_GRAVITY_W, T_world_root=T_W_B)
    else:
        par = dict(status="NOT_RUN", reason="no withheld effort supplied for numerical parity")
    res["numerical_parity"] = par
    hd = yaml.safe_load((g9 / "dynamics_handoff.yaml").read_text())
    tm = hd["parttwin.dynamics/v2"]["engine_term_matrix"]
    full_rep = {e: tm[e]["full_model_representable"] for e in tm}
    parity_pass = {e: (isinstance(par, dict) and par.get(e, {}).get("status") == "PASS") for e in
                   ("mujoco", "sapien_urdf")}
    readiness = dict(man["readiness"])
    readiness.update({k: rd[k] for k in ("VISUAL_ASSET_READY", "COLLISION_READY", "URDF_COMPILES", "MJCF_COMPILES",
                                         "USD_LOADS")})
    readiness["ENGINE_PARITY_PASS"] = bool(all(parity_pass.values()) and all(full_rep[e] for e in
                                                                           ("mujoco", "sapien_urdf")))
    readiness["FULL_PHYSICS_READY"] = bool(readiness["ENGINE_PARITY_PASS"] and readiness["MODEL_VALIDATED"] and
                                           readiness["INERTIA_RECONSTRUCTED"] and rd["VISUAL_ASSET_READY"] and
                                           rd["USD_LOADS"] and full_rep["isaac_usd"] and
                                           rd["builder_status"] == "FULL_PHYSICS_READY")
    res.update(status="EXPORTED", readiness=readiness, full_model_representable=full_rep,
               engine_parity=parity_pass, moving_link_from_builder=mov,
               claim=("FULL_PHYSICS_READY" if readiness["FULL_PHYSICS_READY"] else
                      "NOMINAL_DYNAMICS_EXPORT (not full physical parity: see full_model_representable / parity)"))
    (g9 / "final_physics_export.json").write_text(json.dumps(_clean(res), indent=1))
    return res
