#!/usr/bin/env python3
"""Joint-aware FreeArtGS reconstruction controller (end-to-end spec section 10).

    python pipelines/joint_aware_reconstruction.py --scene SCENE --frames 90:148:2 --cameras cam_001 cam_002 \
        --output-root OUT --stage all --budget coarse [--discovery-run RUN --observations obs_*.npz] [--dry-run]

Stages (each branch-scoped, resumable only when the recorded hashes match):
  audit      environment, git/submodule state, which nerfstudio is imported (must be the ForceArtGS submodule)
  discover   reuse a finished joint_discovery run (hash-recorded) or run `fit` with the cluster/observability policy
  authority  one validated JointAuthority per dynamic hypothesis (revolute, prismatic), q re-gauged to t_ref = 0
  dense      candidate-specific S/M/U masks (dense_support), held-out images never used as evidence targets
  build      dataset/v000 with exact physical + virtual cameras (no ICP, no reference camera)
  parts      generate_ply + splatfacto-depth per part with unknown-aware supervision, no normalisation, cams fixed
  blend      twopart-blend with exact joint poses (free SE(3) adjustment forbidden), verified A_B(t) after training
  art        splatfacto-art from the authority-derived adapter, joint/axis/origin/offset frozen, verified export
  refine     G6 alternating refinement: NOT IMPLEMENTED in this version (recorded as NOT_RUN, never PASSED)
  compare    held-out-time RGB-D rendering of every finished branch, paired block bootstrap -> selection.json
  video      FreeArtGS comparison folder per branch (evaluate/visualize_local_comparison.py, held-out views rendered first)
  final      world-frame articulation + assets only for an identified branch; otherwise diagnostics only
  export     G8 (opt-in --export-parttwin): handoff + shared PArtTwin1D builder -> PLYs, asset.yaml, URDF/USD/MJCF

Legacy entry points (track_2part, segment, ICP, prepare_multiview, estimate_joint) raise inside this controller
(FORCEARTGS_JOINT_AWARE_TRIPWIRE=1). Nothing falls back to vanilla.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from estimotion_2part.joint_discovery.integration_contract import (  # noqa: E402
    AuthorityError, JointAuthority, atomic_write_json, authority_from_discovery, derive_estimated_joint,
    frame_alignment, sha256_file)

SUBMODULE = REPO / "reconstruction" / "nerfstudio"
STAGES = ("audit", "discover", "select", "authority", "dense", "build", "parts", "blend", "art", "refine", "compare", "video",
          "final", "export")
PARTTWIN_ROOT = Path(os.environ.get("PARTTWIN1D_ROOT", str(Path.home() / "PArtTwin1D")))
PARTTWIN_PY = os.environ.get("PARTTWIN1D_PYTHON", str(Path.home() / "miniconda3/envs/parttwin_sim/bin/python"))
SIM_PY = os.environ.get("PARTTWIN_SIM_PYTHON", str(Path.home() / ".venvs/rigeuler-sim-export/bin/python"))
BUDGETS = {"smoke": 300, "coarse": 2000, "full": 30000}
BRANCHES = ("revolute", "prismatic")
SSD = "/media/samuelemara/Extreme SSD/"
# matched to the vanilla real-scene configs (sh 0, xyzw, camera optimiser off, same loss weights)
HP = dict(sh_degree=0, quaternion_order="xyzw", fg_mask_lambda=1.0, depth_lambda_parts=10.0, depth_lambda_blend=10.0,
          depth_lambda_art=1.0)


class StageError(RuntimeError):
    pass


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- environment
def stage_env(output_root: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SUBMODULE), str(REPO), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["FORCEARTGS_JOINT_AWARE_TRIPWIRE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_HOME"] = "/usr/local/cuda-12.8"          # same toolkit as pipelines/train_local_datasets.sh
    env["PATH"] = f"{env['CUDA_HOME']}/bin:{Path(sys.executable).parent}:{env.get('PATH', '')}"
    prefix = Path(sys.executable).parents[1]
    env["LD_LIBRARY_PATH"] = f"{prefix}/lib:{env['CUDA_HOME']}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    from utils.local_storage import runtime_environment
    rt = runtime_environment(output_root)
    for key, value in rt.items():
        if key not in ("PYTHONDONTWRITEBYTECODE", "MPLBACKEND"):
            Path(value).mkdir(parents=True, exist_ok=True)
    env.update(rt)
    env["MAX_JOBS"] = env.get("MAX_JOBS", "2")
    return env


def run_cmd(cmd, log_path: Path, env, dry=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = " ".join(str(c) for c in cmd)
    if dry:
        print("   DRY:", pretty)
        return 0
    with open(log_path, "a") as fh:
        fh.write(f"\n$ {pretty}\n")
        fh.flush()
        rc = subprocess.run([str(c) for c in cmd], cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise StageError(f"command failed (rc={rc}); see {log_path}: {pretty[:200]}")
    return rc


def ns(tool):
    return Path(sys.executable).parent / tool


def job_name(scene: str, cameras, frames: str, tag: str) -> str:
    """FreeArtGS batch naming (train_local_datasets.py): <scene>_multiview_cams001-002_i<a>-<b>_fall_s<k>, plus
    _joint_aware_<budget>; each branch directory appends _<revolute|prismatic>."""
    a, b, k = (frames.split(":") + ["1"])[:3]
    ids = "-".join(c.removeprefix("cam_") for c in sorted(cameras))
    return f"{scene}_multiview_cams{ids}_i{int(a)}-{int(b)}_fall_s{int(k)}_joint_aware_{tag}"


# ----------------------------------------------------------------------------- run state
class Run:
    def __init__(self, args):
        self.args = args
        self.scene = Path(args.scene)
        self.iters = args.iterations or BUDGETS[args.budget]
        # one directory per training budget: a resumed stage can never pick up checkpoints of another budget
        tag = args.budget if not args.iterations else f"it{args.iterations}"
        self.job = job_name(self.scene.name, args.cameras, args.frames, tag)
        self.root = Path(args.output_root) / self.job
        self.manifest_path = self.root / "run_manifest.json"
        self.m = json.loads(self.manifest_path.read_text()) if self.manifest_path.is_file() else dict(stages={})
        if self.m.get("iterations", self.iters) != self.iters or self.m.get("frames", args.frames) != args.frames \
                or self.m.get("cameras", args.cameras) != args.cameras:
            raise SystemExit(f"{self.root} was run with other iterations/frames/cameras; refusing to resume")
        self.m.update(iterations=self.iters, frames=args.frames, cameras=args.cameras, scene=str(self.scene),
                      job=self.job, branch_dirs={k: str(self.branch(k)) for k in BRANCHES})
        self.env = stage_env(Path(args.output_root))

    def branch(self, kind):
        # same layout as the FreeArtGS batch: <job>/{part_0,part_1,twopart_blend,splatfacto-art/comparison}
        return self.root.parent / f"{self.job}_{kind}"

    def set(self, stage, status, **info):
        self.m["stages"][stage] = dict(status=status, time=time.strftime("%F %T"), **info)
        if not self.args.dry_run:
            atomic_write_json(self.manifest_path, self.m)

    def branches(self):
        return [k for k in BRANCHES if (self.branch(k) / "joint_authority" / "v000" / "authority.json").is_file()]


# ----------------------------------------------------------------------------- stages
def st_audit(r: Run):
    git = lambda *a, cwd=REPO: subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
    probe = subprocess.run([sys.executable, "-c", "import nerfstudio, gsplat, torch; print(nerfstudio.__file__); "
                            "print(gsplat.__version__); print(torch.__version__, torch.cuda.is_available())"],
                           env=r.env, capture_output=True, text=True, cwd=REPO)
    lines = probe.stdout.strip().splitlines()
    if probe.returncode or not lines or not lines[0].startswith(str(SUBMODULE)):
        raise StageError(f"stage processes would not import the ForceArtGS submodule nerfstudio: {probe.stdout} {probe.stderr[-400:]}")
    sub_diff = git("diff", cwd=SUBMODULE)
    info = dict(parent_head=git("rev-parse", "HEAD"), parent_dirty=len(git("status", "--porcelain").splitlines()),
                submodule_head=git("rev-parse", "HEAD", cwd=SUBMODULE),
                submodule_branch=git("rev-parse", "--abbrev-ref", "HEAD", cwd=SUBMODULE),
                submodule_diff_sha256=__import__("hashlib").sha256(sub_diff.encode()).hexdigest(),
                nerfstudio_import=lines[0], gsplat=lines[1] if len(lines) > 1 else None,
                torch=lines[2] if len(lines) > 2 else None, python=sys.executable,
                gpu=subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                   capture_output=True, text=True).stdout.strip())
    r.root.mkdir(parents=True, exist_ok=True)
    (r.root / "audit").mkdir(exist_ok=True)
    (r.root / "audit" / "submodule.diff").write_text(sub_diff)
    (r.root / "audit" / "parent.diff").write_text(git("diff"))
    r.m["audit"] = info
    r.set("audit", "PASSED", **info)


def st_discover(r: Run):
    a = r.args
    d = r.root / "discovery"
    if a.discovery_run:
        run = Path(a.discovery_run)
        for f in ("joint_candidates.json", "joint_states.npz", "model_selection.json", "part_support/track_support.npz"):
            if not (run / f).is_file():
                raise StageError(f"discovery run lacks {f}: {run}")
        r.m["discovery"] = dict(run=str(run), reused=True,
                                hashes={f: sha256_file(run / f) for f in ("joint_candidates.json", "joint_states.npz")},
                                observations=[str(p) for p in a.observations or []])
    else:
        obs = []
        tcp = d / "robot_archive.npz"
        cli = [sys.executable, "-m", "estimotion_2part.joint_discovery.cli"]
        if not tcp.is_file():
            run_cmd(cli + ["robot-archive", "--scene", r.scene, "--output", tcp], d / "discover.log", r.env, a.dry_run)
        for cam in a.cameras:
            f = d / "observations" / f"obs_{cam}.npz"
            if not f.is_file():
                run_cmd(cli + ["extract", "--scene", r.scene, "--camera", cam, "--frames", a.frames, "--tcp-npz", tcp,
                               "--output", f], d / "discover.log", r.env, a.dry_run)
            obs.append(f)
        run = d / "fit"
        if not (run / "joint_candidates.json").is_file():
            run_cmd(cli + ["fit", "--scene", r.scene, "--config", a.discovery_config, "--cameras", *a.cameras,
                           "--observations", *obs, "--uncertainty", "cluster_robust", "--remote-hinge-policy",
                           "observability", "--experiment-id", "JOINT_AWARE_DISCOVERY", "--output", run],
                    d / "discover.log", r.env, a.dry_run)
        r.m["discovery"] = dict(run=str(run), reused=False, observations=[str(p) for p in obs],
                                robot_archive=str(tcp))
    r.set("discover", "PASSED", run=r.m["discovery"]["run"])


def _decision(run):
    return json.loads((Path(run) / "joint_candidates.json").read_text()).get("decision")


V2_STATUS = ("IDENTIFIED_REVOLUTE", "IDENTIFIED_PRISMATIC", "TYPE_UNOBSERVABLE", "MODEL_INCONSISTENT",
             "NO_SIGNIFICANT_MOTION")


def st_select(r: Run):
    """Joint selection v2 (default for the joint-aware path): physical status + working model from hinge-free rigid
    motion evidence (estimotion_2part/joint_discovery/selection_v2.py). Legacy selection is a diagnostic only."""
    if r.args.selection == "legacy":
        r.set("select", "NOT_RUN", reason="--selection legacy: discovery decision used (diagnostic mode)")
        return
    run = Path(r.m["discovery"]["run"])
    out = r.root / "joint_selection_v2"
    if not (out / "joint_selection_v2.json").is_file():
        if out.exists() and any(out.iterdir()):
            raise StageError(f"incomplete v2 selection directory {out}; inspect and remove it")
        if not (run / "tracks_observations.npz").is_file():
            raise StageError(f"discovery run lacks tracks_observations.npz: {run}")
        cmd = [sys.executable, "-m", "estimotion_2part.joint_discovery.cli", "select-v2", "--scene", r.scene,
               "--from-run", run, "--config", r.args.discovery_config, "--output", out]
        tcp = r.m["discovery"].get("robot_archive") or r.args.robot_archive
        if tcp and Path(tcp).is_file():
            cmd += ["--tcp-npz", tcp]
        env = dict(r.env)
        env.pop("FORCEARTGS_JOINT_AWARE_TRIPWIRE", None)
        run_cmd(cmd, r.root / "logs" / "select_v2.log", env, r.args.dry_run)
    if r.args.dry_run:
        r.set("select", "DRY_RUN")
        return
    sel = json.loads((out / "joint_selection_v2.json").read_text())
    keep = ("physical_joint_status", "working_motion_model", "working_model_is_approximation",
            "physics_joint_export_ready", "selected_candidate", "failure_codes", "reasons", "counterfactual")
    r.m["selection_v2"] = dict({k: sel[k] for k in keep}, path=str(out / "joint_selection_v2.json"),
                               sha256=sha256_file(out / "joint_selection_v2.json"))
    r.set("select", sel["physical_joint_status"], **{k: sel[k] for k in keep[:6]})


def st_authority(r: Run):
    run = r.m["discovery"]["run"]
    decision = _decision(run)
    v2 = r.m.get("selection_v2") if r.args.selection == "v2" else None
    if r.args.selection == "v2" and v2 is None:
        raise StageError("run the select stage first (joint selection v2)")
    if v2 is None:
        plan = {k: ("provisional" if decision == k else "candidate") for k in BRANCHES}
    else:
        st = v2["physical_joint_status"]
        if st == "NO_SIGNIFICANT_MOTION":
            raise StageError("joint selection v2: NO_SIGNIFICANT_MOTION -- nothing articulated to reconstruct")
        if st == "MODEL_INCONSISTENT" and not r.args.allow_inconsistent_diagnostics:
            raise StageError("joint selection v2: MODEL_INCONSISTENT -- no single 1-DoF joint explains the motion "
                             f"({'; '.join(v2['reasons'][:2])}). Diagnostic branches only with "
                             "--allow-inconsistent-diagnostics")
        if st.startswith("IDENTIFIED_"):
            plan = {v2["selected_candidate"]: "identified"}          # the accepted candidate alone is authoritative
        elif st == "TYPE_UNOBSERVABLE":
            plan = {k: "ambiguous" for k in BRANCHES}                 # both kept; working model flagged
        else:
            plan = {k: "candidate" for k in BRANCHES}                 # MODEL_INCONSISTENT diagnostics
    made = {}
    for kind, status in plan.items():
        try:
            auth = authority_from_discovery(run, kind, r.scene, r.args.cameras, decision_status=status,
                                            reference_time_index=0)
        except AuthorityError as e:
            made[kind] = f"NOT_EVALUATED: {e}"
            continue
        aid = auth.write(r.branch(kind) / "joint_authority" / "v000")
        made[kind] = aid
        log(f"authority {kind}: {aid[:12]} span {np.ptp(auth.q):.4f} {auth.q_units} status {status}")
    r.m["discovery_decision"] = decision
    r.set("authority", "PASSED" if r.branches() else "FAILED", authorities=made, discovery_decision=decision,
          decision_source="joint_selection_v2" if v2 else "legacy discovery decision (diagnostic mode)",
          physical_joint_status=None if v2 is None else v2["physical_joint_status"],
          working_motion_model=None if v2 is None else v2["working_motion_model"])
    if not r.branches():
        raise StageError("no dynamic branch could be built (static scene or invalid candidates)")


def heldout_times(T, block=3):
    """Two predeclared interior time blocks (~1/3 and ~2/3 of the clip); never t=0 (reference) or the last time."""
    if T < 12:
        return []
    out = []
    for frac in (1 / 3, 2 / 3):
        s = int(round(frac * T)) - block // 2
        out += list(range(max(s, 1), min(s + block, T - 1)))
    return sorted(set(out))


def _robot(r: Run, auth: JointAuthority):
    tcp = r.m["discovery"].get("robot_archive") or r.args.robot_archive
    if not tcp or not Path(tcp).is_file():
        return None, "none"
    from estimotion_2part.joint_discovery.observations import FrameMap
    from estimotion_2part.joint_discovery.tcp_contact import load_robot_contact_record
    try:
        rec = load_robot_contact_record(r.scene, FrameMap(auth.rgbd_frame_index, auth.source_frame_index,
                                                          auth.timestamp_s), tcp_npz=tcp)
    except (ValueError, RuntimeError, KeyError) as e:      # e.g. tool calibration marked invalid: vision-only
        return None, f"UNAVAILABLE ({e}); no robot pixels excluded"
    return dict(p_tcp_w=rec.p_tcp_w, p_ee_w=rec.p_ee_w), str(tcp)


def st_dense(r: Run):
    from estimotion_2part.joint_discovery.dense_support import DenseConfig, run_dense_support, seeds_from_discovery
    cfg = DenseConfig.from_dict(json.loads(Path(r.args.dense_config).read_text())) if r.args.dense_config else DenseConfig()
    res = {}
    for kind in r.branches():
        out = r.branch(kind) / "dense_support" / "v000"
        auth_dir = r.branch(kind) / "joint_authority" / "v000"
        auth = JointAuthority.load(auth_dir)
        aid = json.loads((auth_dir / "authority.json").read_text())["authority_id"]
        if (out / "label_manifest.json").is_file():
            if json.loads((out / "label_manifest.json").read_text())["authority_id"] != aid:
                raise StageError(f"stale dense support in {out}")
            res[kind] = "reused"
            continue
        held = heldout_times(auth.T)
        train = np.array([t not in held for t in range(auth.T) for _ in r.args.cameras])
        robot, robot_src = _robot(r, auth)
        obs = r.m["discovery"].get("observations") or []
        seeds, rep = seeds_from_discovery(r.m["discovery"]["run"], kind, obs, cfg) if obs else (None, {"seeds": "none"})
        if r.args.dry_run:
            print(f"   DRY: dense {kind} -> {out} (held-out times {held})")
            continue
        m = run_dense_support(r.scene, auth, aid, r.args.cameras, out, train, cfg, seeds=seeds, seed_report=rep,
                              robot=robot)
        res[kind] = dict(totals=m["totals"], walltime_s=m["walltime_s"], peak_vram_mib=m["peak_vram_mib"],
                         heldout_times=held, robot_ignore=robot_src)
        log(f"dense {kind}: {m['totals']}")
    r.set("dense", "PASSED", branches=res)


def st_build(r: Run):
    from preprocessing.prepare_joint_aware_multiview import build_joint_aware_dataset
    res = {}
    for kind in r.branches():
        b = r.branch(kind)
        auth = JointAuthority.load(b / "joint_authority" / "v000")
        if r.args.dry_run:
            print(f"   DRY: build {kind} -> {b / 'dataset' / 'v000'}")
            continue
        m = build_joint_aware_dataset(r.scene, b / "joint_authority" / "v000", b / "dense_support" / "v000",
                                      b / "dataset" / "v000", r.args.cameras, heldout_times(auth.T))
        res[kind] = dict(n_train=m["n_train_images"], n_eval=m["n_eval_images"],
                         virtual_camera_max_error=m["virtual_camera_max_error"])
    r.set("build", "PASSED", branches=res)


def _part_complete(d: Path):
    return (d / "object_3dgs.ply").is_file() and (d / "config.yml").is_file()


def st_parts(r: Run):
    for kind in r.branches():
        b = r.branch(kind)
        ds = b / "dataset" / "v000"
        lg = b / "logs" / "parts.log"
        if r.args.dry_run or not (ds / "transforms_0.json").is_file() or "ply_file_path" not in json.loads((ds / "transforms_0.json").read_text()):
            run_cmd([sys.executable, "reconstruction/generate_ply.py", "--dataset_dir", ds, "--part_ids", "0", "1",
                     "--pixel_stride", "4", "--depth_scale", "1.0"], lg, r.env, r.args.dry_run)
        for part in (0, 1):
            out = b / f"part_{part}"
            if _part_complete(out):
                continue
            run_cmd([ns("ns-train"), "splatfacto-depth", "--data", ds, "--output-dir", out, "--experiment-name",
                     f"part_{part}", "--specialize-output-dir", "True", "--vis", "tensorboard",
                     "--enable-frame-filtering", "False", "--max-num-iterations", r.iters,
                     "--pipeline.model.joint-aware-supervision", "True",
                     "--pipeline.model.fg-mask-lambda", HP["fg_mask_lambda"],
                     "--pipeline.model.sensor-depth-lambda", HP["depth_lambda_parts"],
                     "--pipeline.model.camera-optimizer.mode", "off",
                     "--pipeline.model.quaternion-order", HP["quaternion_order"],
                     "--pipeline.model.sh-degree", HP["sh_degree"], "--pipeline.datamanager.cache-images", "cpu",
                     "part-data", "--part_id", part, "--joint-aware", "True", "--orientation-method", "none",
                     "--center-poses", "False", "--auto-scale-poses", "False", "--scale-factor", "1.0"],
                    lg, r.env, r.args.dry_run)
            run_cmd([ns("ns-export"), "gaussian-splat", "--load-config", out / "config.yml", "--output-dir", out,
                     "--world-frame", "True"], lg, r.env, r.args.dry_run)
            run_cmd([ns("ns-export"), "cameras", "--load-config", out / "config.yml", "--output-dir", out,
                     "--optimized", "False", "--world-frame", "True"], lg, r.env, r.args.dry_run)
    r.set("parts", "PASSED", iterations=r.iters)


def _transforms(path):
    data = json.loads(Path(path).read_text())
    frames = data if isinstance(data, list) else data["frames"]
    out = {}
    for f in frames:
        T = np.eye(4)
        T[:3] = np.asarray(f.get("transform", f.get("transform_matrix")), float)[:3]
        out[Path(f["file_path"]).name] = T
    return out


def verify_blend(b: Path, auth: JointAuthority, tol=2e-4) -> dict:
    """N_BW from blend cameras vs dataset physical cameras, then exported object poses == N A_W(t) N^-1."""
    ds = b / "dataset" / "v000"
    blend = b / "twopart_blend"
    phys = {Path(f["file_path"]).name: np.asarray(f["transform_matrix"], float)
            for f in json.loads((ds / "transforms_0.json").read_text())["frames"]}
    bt = _transforms(blend / "transforms_train.json")
    names = [n for n in bt if n in phys]
    N = frame_alignment(np.stack([phys[n] for n in names]), np.stack([bt[n] for n in names]), 1e-4, 1e-4)
    objp = json.loads((blend / "object_poses_train.json").read_text())
    err = 0.0
    Ninv = np.linalg.inv(N)
    for rec in objp:          # multiview blend keeps one object pose per acquisition time: frame_index == time index
        T = np.eye(4)
        T[:3] = np.asarray(rec["T_obj"], float)
        A_B = N @ auth.transform(int(rec["frame_index"])) @ Ninv
        err = max(err, float(np.abs(T - A_B).max()))
    if err > tol:
        raise StageError(f"blend object poses deviate from the authority (max {err:.2e}): free motion leaked in")
    return dict(N_BW=N.tolist(), object_pose_max_error=err, n_checked=len(objp),
                N_is_identity=bool(np.abs(N - np.eye(4)).max() < 1e-6))


def st_blend(r: Run):
    res = {}
    for kind in r.branches():
        b = r.branch(kind)
        out = b / "twopart_blend"
        lg = b / "logs" / "blend.log"
        if not (out / "object_3dgs.ply").is_file():
            run_cmd([ns("ns-train"), "twopart-blend", "--data", b / "dataset" / "v000", "--output-dir", out,
                     "--experiment-name", "twopart_blend", "--specialize-output-dir", "True", "--vis", "tensorboard",
                     "--enable-frame-filtering", "False", "--max-num-iterations", r.iters,
                     "--pipeline.model.joint-aware", "True", "--pipeline.model.obj-pose-opt-enabled", "False",
                     "--pipeline.model.trainable-weights", "True", "--pipeline.model.camera-optimizer.mode", "off",
                     "--pipeline.model.quaternion-order", HP["quaternion_order"],
                     "--pipeline.model.enable-gs-refinement", "False",
                     "--pipeline.model.fg-mask-lambda", HP["fg_mask_lambda"],
                     "--pipeline.model.sensor-depth-lambda", HP["depth_lambda_blend"],
                     "--pipeline.model.sh-degree", HP["sh_degree"], "--pipeline.datamanager.cache-images", "cpu",
                     "twopart-blend-data", "--data", b / "dataset" / "v000", "--part0_dir", b / "part_0",
                     "--part1_dir", b / "part_1", "--output-dir", out, "--motion_part", "1",
                     "--orientation-method", "none", "--center-poses", "False", "--auto-scale-poses", "False",
                     "--scale-factor", "1.0", "--joint-aware", "True"], lg, r.env, r.args.dry_run)
            run_cmd([ns("ns-export"), "gaussian-splat", "--load-config", out / "config.yml", "--output-dir", out,
                     "--world-frame", "True"], lg, r.env, r.args.dry_run)
            run_cmd([ns("ns-export"), "cameras", "--load-config", out / "config.yml", "--output-dir", out,
                     "--optimized", "True", "--world-frame", "True"], lg, r.env, r.args.dry_run)
        if r.args.dry_run:
            continue
        auth_dir = b / "joint_authority" / "v000"
        auth = JointAuthority.load(auth_dir)
        aid = json.loads((auth_dir / "authority.json").read_text())["authority_id"]
        v = verify_blend(b, auth)
        est = derive_estimated_joint(auth, np.asarray(v["N_BW"]), aid)
        atomic_write_json(out / "estimated_joint.json", est)
        atomic_write_json(out / "JOINT_AWARE_AUTHORITY.json", dict(
            authority_id=aid, estimated_joint_sha256=sha256_file(out / "estimated_joint.json"),
            num_timestamps=auth.T, N_BW=v["N_BW"], note="estimated_joint.json is a derived adapter, not a fit"))
        res[kind] = v
    r.set("blend", "PASSED", iterations=r.iters, branches=res)


def verify_art(b: Path, auth: JointAuthority, N_BW, tol_axis_deg=1e-3, tol_q=1e-5) -> dict:
    art = json.loads((b / "splatfacto-art" / "articulation.json").read_text())
    N = np.asarray(N_BW, float)
    if art["joint_type"] != auth.joint_type:
        raise StageError(f"art export changed the joint type {auth.joint_type} -> {art['joint_type']}")
    axis_W = np.linalg.inv(N)[:3, :3] @ np.asarray(art["joint_axis_world"], float)
    axis_W = axis_W / np.linalg.norm(axis_W)
    # atan2(|a x b|, |a.b|): arccos is ill-conditioned near 1 (a float32 axis gave false 8e-3 deg alarms)
    ang = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(axis_W, auth.axis_world)), abs(axis_W @ auth.axis_world))))
    q_opt = np.asarray(art["optimization_details"]["params_optimized"], float)
    dq = float(np.abs(q_opt - auth.q[:len(q_opt)]).max()) if len(q_opt) else float("nan")
    out = dict(axis_change_deg=ang, q_max_abs_change=dq, n_q=len(q_opt))
    if auth.joint_type == "revolute":
        o_W = (np.linalg.inv(N) @ np.r_[np.asarray(art["joint_origin_world"], float), 1])[:3]
        d = o_W - auth.origin_world
        out["hinge_line_shift_m"] = float(np.linalg.norm(d - (d @ auth.axis_world) * auth.axis_world))
        if out["hinge_line_shift_m"] > 1e-4:
            raise StageError(f"art export moved the hinge line by {out['hinge_line_shift_m']:.2e} m")
    if ang > tol_axis_deg or not dq <= tol_q:
        raise StageError(f"art export drifted from the frozen authority: axis {ang:.2e} deg, q {dq:.2e}")
    return out


def st_art(r: Run):
    res = {}
    for kind in r.branches():
        b = r.branch(kind)
        out = b / "splatfacto-art"
        lg = b / "logs" / "art.log"
        if not (out / "articulation.json").is_file():
            if not r.args.dry_run and not (b / "twopart_blend" / "JOINT_AWARE_AUTHORITY.json").is_file():
                raise StageError(f"{kind}: blend stage did not produce the authority adapter")
            run_cmd([ns("ns-train"), "splatfacto-art", "--data", b / "dataset" / "v000", "--output-dir", out,
                     "--experiment-name", "splatfacto-art", "--specialize-output-dir", "True", "--vis", "tensorboard",
                     "--max-num-iterations", r.iters,
                     "--pipeline.model.optimize-joint-axis", "False", "--pipeline.model.optimize-joint-origin", "False",
                     "--pipeline.model.optimize-joint-params", "False",
                     "--pipeline.model.optimize-rotation-offset", "False",
                     "--pipeline.model.enable-gs-refinement", "True",
                     "--pipeline.model.fg-mask-lambda", HP["fg_mask_lambda"],
                     "--pipeline.model.sensor-depth-lambda", HP["depth_lambda_art"],
                     "--pipeline.model.camera-optimizer.mode", "off",
                     "--pipeline.model.quaternion-order", HP["quaternion_order"],
                     "--pipeline.model.sh-degree", HP["sh_degree"], "--pipeline.datamanager.cache-images", "cpu",
                     "splatfacto-art-data", "--twopart-blend-dir", b / "twopart_blend", "--motion_part", "1",
                     "--data-dir", b / "dataset" / "v000", "--output-dir", out, "--joint-aware", "True",
                     "--orientation-method", "none", "--center-poses", "False", "--auto-scale-poses", "False",
                     "--scale-factor", "1.0"], lg, r.env, r.args.dry_run)
            for what in ("gaussian-splat", "cameras", "articulation"):
                extra = ["--optimized", "True"] if what == "cameras" else []
                run_cmd([ns("ns-export"), what, "--load-config", out / "config.yml", "--output-dir", out,
                         "--world-frame", "True", *extra], lg, r.env, r.args.dry_run)
        if r.args.dry_run:
            continue
        auth = JointAuthority.load(b / "joint_authority" / "v000")
        N = json.loads((b / "twopart_blend" / "JOINT_AWARE_AUTHORITY.json").read_text())["N_BW"]
        res[kind] = verify_art(b, auth, N)
    r.set("art", "PASSED", iterations=r.iters, branches=res)


def st_refine(r: Run):
    r.set("refine", "NOT_RUN", reason="G6 alternating membership/joint refinement with the full renderer is not "
                                      "implemented in this version; joints stay frozen at the discovery authority")


def _final_ckpt(d: Path):
    ck = sorted((d / "nerfstudio_models").glob("step-*.ckpt"))
    if not ck:
        raise StageError(f"no checkpoint in {d}")
    return ck[-1]


def _authority_for(r: Run, kind: str) -> JointAuthority:
    """The branch authority if the branch exists, else the same discovery candidate re-derived (not written)."""
    d = r.branch(kind) / "joint_authority" / "v000"
    if (d / "authority.json").is_file():
        return JointAuthority.load(d)
    return authority_from_discovery(r.m["discovery"]["run"], kind, r.scene, r.args.cameras,
                                    decision_status="candidate", reference_time_index=0)


def evaluate_branch(r: Run, kind: str, motion_kind: str | None = None) -> list:
    """Held-out-time RGB-D renders of the frozen final branch model, per (camera, time) image.

    ``motion_kind`` (controlled counterfactual, spec 8.1): keep this branch's Gaussians, cameras, support and pixels
    fixed and change ONLY the articulated SE(3) field to the other candidate's A(q_t). Both authorities are gauged
    q(t_ref = 0) = 0, so the canonical Gaussians are valid for either motion."""
    import cv2
    import torch
    from estimotion_2part.joint_discovery.gaussian_feedback import SimpleCalib, load_canonical_asset, render
    b = r.branch(kind)
    auth = JointAuthority.load(b / "joint_authority" / "v000")
    motion = auth if motion_kind in (None, kind) else _authority_for(r, motion_kind)
    if motion.reference_time_index != auth.reference_time_index or len(motion.q) != len(auth.q):
        raise StageError("counterfactual motion has a different gauge / frame map")
    N = np.asarray(json.loads((b / "twopart_blend" / "JOINT_AWARE_AUTHORITY.json").read_text())["N_BW"], float)
    asset = load_canonical_asset(checkpoint=str(_final_ckpt(b / "splatfacto-art")), motion_part=1)
    if asset["status"] != "OK":
        raise StageError(f"{kind}: cannot load final asset: {asset['reason']}")
    ds = b / "dataset" / "v000"
    held = set(json.loads((ds / "joint_aware_manifest.json").read_text())["heldout_times"])
    rows = []
    for f in json.loads((ds / "multiview.json").read_text())["frames"]:
        t = f["time_index"]
        if t not in held:
            continue
        cam = auth.cameras[f["camera_name"]]
        cal = SimpleCalib(K=np.asarray(cam.K), T_WC=N @ np.asarray(cam.T_WC), width=cam.width, height=cam.height)
        A_B = N @ motion.transform(t) @ np.linalg.inv(N)
        with torch.no_grad():
            out = render(asset["static"], asset["moving"], A_B, cal)
        name = Path(f["file_path"]).name
        rgb = cv2.imread(str(ds / "images_ori" / name))[:, :, ::-1].astype(np.float32) / 255
        fg = cv2.imread(str(ds / "masks" / name), cv2.IMREAD_GRAYSCALE) > 127
        ign = cv2.imread(str(ds / "ignore_mask" / name), cv2.IMREAD_GRAYSCALE) > 127
        dep = np.load(ds / "depth" / name.replace(".png", ".npz"))["depth"]
        pr = out.rgb.detach().cpu().numpy()
        pd = out.depth.detach().cpu().numpy()
        pa = out.alpha.detach().cpu().numpy()
        valid = ~ign
        dmask = fg & (dep > 0) & (pa > 0.5)
        sil = pa > 0.5
        rows.append(dict(camera=f["camera_name"], time_index=t, rgb_l1_fg=float(np.abs(pr - rgb)[fg].mean()),
                         depth_mae_fg_m=float(np.abs(pd - dep)[dmask].mean()) if dmask.any() else float("nan"),
                         depth_coverage=float(dmask.sum() / max(fg.sum(), 1)),
                         silhouette_iou=float((sil & fg & valid).sum() / max(((sil | fg) & valid).sum(), 1))))
    return rows


def st_compare(r: Run):
    if r.args.selection == "legacy":
        return _legacy_compare(r)
    _compare_v2(r)


def _compare_v2(r: Run):
    """Render evidence under v2: diagnostic, may only VETO (spec 7.7 / 8): needs >= 6 independent held-out blocks,
    >= 2 paired training seeds and a difference beyond the between-seed variability. It never confers a type."""
    from estimotion_2part.joint_discovery.blocked_statistics import count_independent_blocks
    from estimotion_2part.joint_discovery.physical_selection import render_veto_allowed
    v2 = json.loads(Path(r.m["selection_v2"]["path"]).read_text())
    trained = [k for k in r.branches() if (r.branch(k) / "splatfacto-art" / "articulation.json").is_file()]
    per, cf = {}, {}
    for k in trained:
        per[k] = evaluate_branch(r, k)
        atomic_write_json(r.branch(k) / "branch_eval.json", dict(heldout_rows=per[k]))
        rows = {}
        for m in BRANCHES:
            try:
                rows[m] = evaluate_branch(r, k, motion_kind=m)
            except (AuthorityError, StageError) as e:
                rows[m] = f"NOT_EVALUATED: {e}"
        if all(isinstance(v, list) for v in rows.values()):
            key = lambda x: (x["camera"], x["time_index"])
            P = {key(x): x for x in rows["prismatic"]}
            d = [x["depth_mae_fg_m"] - P[key(x)]["depth_mae_fg_m"] for x in rows["revolute"] if key(x) in P]
            iou = [x["silhouette_iou"] - P[key(x)]["silhouette_iou"] for x in rows["revolute"] if key(x) in P]
            cf[k] = dict(geometry=f"{k} branch Gaussians", motions_compared=list(BRANCHES),
                         depth_mae_revolute_minus_prismatic_m=float(np.nanmean(d)),
                         iou_revolute_minus_prismatic=float(np.nanmean(iou)), n_images=len(d))
        else:
            cf[k] = rows
    held = sorted({x["time_index"] for v in per.values() for x in v})
    nb = count_independent_blocks(held, 3)
    diff = None
    if {"revolute", "prismatic"} <= set(per):
        dm = {k: float(np.nanmean([x["depth_mae_fg_m"] for x in v])) for k, v in per.items()}
        diff = dm["revolute"] - dm["prismatic"]
    render = dict(status="OK" if per else "NOT_RUN", n_independent_blocks=nb, heldout_times=held, n_seeds=1,
                  between_seed_sd=None, difference=diff, metric="depth MAE on held-out frames (m), R - P",
                  per_branch={k: {m: float(np.nanmean([x[m] for x in v])) for m in
                                  ("rgb_l1_fg", "depth_mae_fg_m", "silhouette_iou", "depth_coverage")}
                              for k, v in per.items()},
                  controlled_counterfactual=cf,
                  seed_replication="NOT_RUN: one training seed per branch (between-seed variability unknown)")
    veto = render_veto_allowed(render, r.args.render_min_blocks, 2)
    if nb < r.args.render_min_blocks:
        render["claim_status"] = f"NOT_IDENTIFIABLE_FROM_AVAILABLE_BLOCKS ({nb} < {r.args.render_min_blocks})"
    sel = dict(schema_version=2, status=v2["physical_joint_status"], selected=v2["selected_candidate"],
               working_motion_model=v2["working_motion_model"],
               working_model_is_approximation=v2["working_model_is_approximation"],
               physics_joint_export_ready=v2["physics_joint_export_ready"], reasons=v2["reasons"],
               failure_codes=list(v2["failure_codes"]), render_evidence=render, render_veto=veto,
               selection_v2=r.m["selection_v2"]["path"], selection_v2_sha256=r.m["selection_v2"]["sha256"],
               discovery_decision_legacy=r.m.get("discovery_decision"))
    if veto.get("allowed") and sel["selected"] and veto["favours"] != sel["selected"]:
        sel.update(status="MODEL_INCONSISTENT", physics_joint_export_ready=False)
        sel["failure_codes"].append("RENDER_VETO")
        sel["reasons"] = sel["reasons"] + [f"replicated render evidence vetoes {sel['selected']}"]
    out = r.root / "joint_selection_v2" / "render_comparison"
    atomic_write_json(out / "controlled_counterfactual.json", dict(status="OK" if cf else "NOT_RUN", per_geometry=cf,
                      note="same Gaussians/cameras/pixels; only A(q_t) changes; q at held-out times from discovery "
                           "tracks (transductive, documented)"))
    atomic_write_json(out / "full_training_seed_report.json", dict(status="NOT_RUN", n_seeds=1,
                      reason="paired repeated training seeds not run; render evidence cannot veto"))
    atomic_write_json(r.root / "selection.json", sel)
    r.set("compare", sel["status"], render_veto=veto, n_independent_blocks=nb)


def _legacy_compare(r: Run):
    per = {}
    for kind in r.branches():
        if not (r.branch(kind) / "splatfacto-art" / "articulation.json").is_file():
            per[kind] = "NOT_EVALUATED (no finished art model)"
            continue
        if r.args.dry_run:
            continue
        per[kind] = evaluate_branch(r, kind)
        atomic_write_json(r.branch(kind) / "branch_eval.json", dict(heldout_rows=per[kind]))
    sel = dict(discovery_decision=r.m.get("discovery_decision"), branches={}, protocol=(
        "held-out interior time blocks, all cameras; q at held-out times from the discovery tracks (transductive "
        "tracking, not optimised on held-out RGB-D); paired bootstrap over time blocks"), status="ABSTAIN")
    done = {k: v for k, v in per.items() if isinstance(v, list) and v}
    for k, v in done.items():
        sel["branches"][k] = {m: float(np.nanmean([row[m] for row in v])) for m in
                              ("rgb_l1_fg", "depth_mae_fg_m", "silhouette_iou", "depth_coverage")}
    if len(done) == 2:
        rr, pp = done["revolute"], done["prismatic"]
        key = lambda row: (row["camera"], row["time_index"])
        pm = {key(x): x for x in pp}
        pairs = [(x, pm[key(x)]) for x in rr if key(x) in pm]
        rng = np.random.default_rng(0)
        times = sorted({x["time_index"] for x, _ in pairs})
        cis = {}
        for metric, better_low in (("rgb_l1_fg", True), ("depth_mae_fg_m", True), ("silhouette_iou", False)):
            d = {t: [((x[metric] - y[metric]) * (1 if better_low else -1)) for x, y in pairs if x["time_index"] == t]
                 for t in times}
            boots = []
            for _ in range(2000):
                pick = rng.choice(times, len(times))
                vals = np.concatenate([d[t] for t in pick])
                boots.append(np.nanmean(vals))
            cis[metric] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        sel["paired_ci_revolute_minus_prismatic_loss"] = cis
        favours = {m: ("revolute" if c[1] < 0 else "prismatic" if c[0] > 0 else None) for m, c in cis.items()}
        winners = {w for w in favours.values() if w}
        if len(winners) == 1 and len(times) >= 4:
            w = winners.pop()
            sel.update(status="IDENTIFIED", selected=w, agrees_with_discovery=(w == r.m.get("discovery_decision")))
        else:
            sel.update(status="AMBIGUOUS", reason=f"held-out render metrics do not single out one branch: {favours}"
                       + ("" if len(times) >= 4 else f"; only {len(times)} held-out time steps (low power)"))
    elif len(done) == 1:
        k = next(iter(done))
        sel.update(status="IDENTIFIED" if r.m.get("discovery_decision") == k else "AMBIGUOUS", selected=k,
                   reason="only one branch reconstructed; no render competition was possible")
    if not r.args.dry_run:
        atomic_write_json(r.root / "selection.json", sel)
    r.set("compare", sel["status"] if done else "NOT_RUN", selection=sel)



def st_video(r: Run):
    """The FreeArtGS fork comparison folder (<branch>/splatfacto-art/comparison: motion_axis_debug.mp4/.gif,
    comparison_*.png, comparison_info.json) made by evaluate/visualize_local_comparison.py, unchanged. Views the
    model never trained on (held-out times) are first rendered by the same model (render_joint_aware_views.py)."""
    res = {}
    for kind in r.branches():
        b = r.branch(kind)
        art = b / "splatfacto-art"
        if not (art / "articulation.json").is_file():
            res[kind] = "NOT_RUN (no finished art model)"
            continue
        lg = b / "logs" / "video.log"
        run_cmd([sys.executable, "evaluate/render_joint_aware_views.py", "--config", art / "config.yml",
                 "--dataset", b / "dataset" / "v000", "--blend", b / "twopart_blend"], lg, r.env, r.args.dry_run)
        run_cmd([sys.executable, "evaluate/visualize_local_comparison.py", "--scene", r.scene,
                 "--dataset", b / "dataset" / "v000", "--results", art], lg, r.env, r.args.dry_run)
        res[kind] = str(art / "comparison")
        log(f"comparison {kind}: {res[kind]}")
    r.set("video", "PASSED" if any(v.endswith("comparison") for v in res.values()) else "NOT_RUN", branches=res)


def st_final(r: Run):
    sel_path = r.root / "selection.json"
    if not sel_path.is_file():
        r.set("final", "NOT_RUN", reason="no selection")
        return
    sel = json.loads(sel_path.read_text())
    v2 = sel.get("schema_version") == 2
    accepted = sel.get("status", "").startswith("IDENTIFIED_") if v2 else sel.get("status") == "IDENTIFIED"
    if not accepted:
        st = sel.get("status")
        if v2 and st == "TYPE_UNOBSERVABLE":
            wk = sel.get("working_motion_model")
            r.set("final", "ABSTAIN", reason="TYPE_UNOBSERVABLE: no physical joint; the working model is an "
                                             "approximation for appearance only",
                  working_motion_model=wk, working_branch=str(r.branch(wk)) if wk in BRANCHES else None,
                  diagnostics={k: str(r.branch(k)) for k in r.branches()})
        else:
            r.set("final", "BLOCKED" if st == "MODEL_INCONSISTENT" else "ABSTAIN",
                  reason=f"selection is {st}: diagnostic branch outputs only",
                  diagnostics={k: str(r.branch(k)) for k in r.branches()})
        return
    k = sel["selected"]
    b = r.branch(k)
    auth_dir = b / "joint_authority" / "v000"
    auth = JointAuthority.load(auth_dir)
    fin = r.root / "final"
    fin.mkdir(exist_ok=True)
    marker = json.loads((b / "twopart_blend" / "JOINT_AWARE_AUTHORITY.json").read_text())
    for f in ("object_3dgs.ply", "3dgs_part_weight.txt"):
        shutil.copy2(b / "splatfacto-art" / f, fin / f)
    shutil.copy2(auth_dir / "authority.json", fin / "joint_authority.json")
    atomic_write_json(fin / "articulation.json", dict(
        joint_type=auth.joint_type, joint_axis_world=auth.axis_world.tolist(),
        joint_origin_world=None if auth.origin_world is None else auth.origin_world.tolist(),
        q=auth.q.tolist(), q_valid=auth.q_valid.tolist(), q_units=auth.q_units,
        reference_time_index=auth.reference_time_index, rgbd_frame_index=auth.rgbd_frame_index.tolist(),
        source_frame_index=auth.source_frame_index.tolist(), timestamp_s=auth.timestamp_s.tolist(),
        authority_id=marker["authority_id"], world_frame=auth.world_frame,
        gaussian_frame=("W" if np.allclose(marker["N_BW"], np.eye(4), atol=1e-6) else
                        "blend frame B = N_BW W (see N_BW); Gaussians were not re-expressed"),
        N_BW=marker["N_BW"], selection=sel,
        physical_joint_status=sel.get("status") if v2 else "LEGACY_SELECTION_NOT_PHYSICALLY_CERTIFIED",
        physics_joint_export_ready=bool(sel.get("physics_joint_export_ready")) if v2 else False))
    if v2:
        shutil.copy2(sel["selection_v2"], fin / "joint_selection_v2.json")
    r.set("final", "PASSED", selected=k, final_dir=str(fin),
          physics_joint_export_ready=bool(sel.get("physics_joint_export_ready")) if v2 else False)


def st_export(r: Run):
    """G8 (opt-in --export-parttwin): handoff from the accepted branch -> shared PArtTwin1D builder (two standard
    3DGS PLYs + asset.yaml + URDF / USD / MJCF when a measured physics spec is given). Ambiguous scenes: nothing,
    unless --export-diagnostic (one uniquely named DIAGNOSTIC asset per branch, never a physical twin)."""
    if not r.args.export_parttwin:
        r.set("export", "NOT_RUN", reason="opt-in: pass --export-parttwin")
        return
    sel = json.loads((r.root / "selection.json").read_text()) if (r.root / "selection.json").is_file() else {}
    jobs = []
    final_ok = r.m["stages"].get("final", {}).get("status") == "PASSED"
    ready = sel.get("schema_version") == 2 and sel.get("physics_joint_export_ready") is True
    if final_ok and (ready or (sel.get("status") == "IDENTIFIED" and r.args.assume_physical_joint)):
        jobs.append(("accepted", None))
    elif r.args.export_diagnostic:
        jobs += [(f"diagnostic_{k}", k) for k in r.branches()
                 if (r.branch(k) / "splatfacto-art" / "articulation.json").is_file()]
    if not jobs:
        r.set("export", "ABSTAIN", reason=f"selection {sel.get('status')}: no accepted joint-aware asset "
                                          "(--export-diagnostic for candidate diagnostics)")
        return
    env = dict(r.env)
    env["PYTHONPATH"] = str(PARTTWIN_ROOT / "src")
    env["PARTTWIN_SIM_PYTHON"] = SIM_PY
    env.pop("FORCEARTGS_JOINT_AWARE_TRIPWIRE", None)
    lg = r.root / "logs" / "export.log"
    res = {}
    for name, branch in jobs:
        hdir = r.root / "parttwin_handoff" / name
        if not (hdir / "articulated_3dgs_handoff.json").is_file():
            cmd = [sys.executable, "tools/export_parttwin_handoff.py", "--run", r.root, "--output", hdir]
            if branch:
                cmd += ["--diagnostic-branch", branch]
            elif not ready:
                cmd += ["--assumed-physical-joint", r.args.assume_physical_joint]
            run_cmd(cmd, lg, r.env, r.args.dry_run)
        cmd = [PARTTWIN_PY, "-m", "parttwin1d.cli.build_asset", "--handoff", hdir / "articulated_3dgs_handoff.json",
               "--output", r.root / "parttwin_assets"]
        if branch:
            cmd.append("--allow-diagnostic")
        if r.args.dynamics_spec:
            cmd += ["--dynamics", r.args.dynamics_spec]
        if r.args.registration_spec:
            cmd += ["--registration", r.args.registration_spec]
        run_cmd(cmd, lg, env, r.args.dry_run)
        if r.args.dry_run:
            continue
        side = sorted((r.root / "parttwin_assets").glob("*.handoff_verification.json"), key=lambda p: p.stat().st_mtime)
        info = json.loads(side[-1].read_text()) if side else {}
        res[name] = dict(asset_status=info.get("asset_status"), visual_bitwise=all(
            v.get("bitwise_equal") for v in info.get("visual_bitwise", {}).values()) if info else None)
        log(f"parttwin asset {name}: {res[name]}")
    status = "PASSED" if res and all(v["visual_bitwise"] for v in res.values()) else "FAILED"
    r.set("export", status, assets=res, assets_root=str(r.root / "parttwin_assets"),
          physics="measured spec given" if r.args.dynamics_spec else "none: visual + kinematics only")


FUNCS = dict(audit=st_audit, discover=st_discover, select=st_select, authority=st_authority, dense=st_dense, build=st_build,
             parts=st_parts, blend=st_blend, art=st_art, refine=st_refine, compare=st_compare, video=st_video,
             final=st_final, export=st_export)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frames", required=True, help="RGB-D clip start:end:step (inclusive), as in the discovery")
    ap.add_argument("--cameras", nargs="+", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--stage", default="all", help=f"one of {STAGES + ('all',)} or a comma list")
    ap.add_argument("--budget", choices=tuple(BUDGETS), default="smoke")
    ap.add_argument("--iterations", type=int, help="override the per-stage iteration budget")
    ap.add_argument("--discovery-run", help="finished joint_discovery fit run to reuse (read-only)")
    ap.add_argument("--observations", nargs="*", help="observation tables of the reused discovery (dense seeds)")
    ap.add_argument("--robot-archive", help="robot_archive.npz for tool exclusion when reusing a discovery run")
    ap.add_argument("--discovery-config", default="configs/joint_discovery.real_tcp2cm.yaml")
    ap.add_argument("--dense-config", help="JSON with DenseConfig overrides (unknown keys are rejected)")
    ap.add_argument("--allow-non-ssd", action="store_true", help="tests only: allow an output root off the SSD")
    ap.add_argument("--export-parttwin", action="store_true", help="G8: export the accepted asset via PArtTwin1D")
    ap.add_argument("--export-diagnostic", action="store_true", help="G8: also diagnostic assets for ambiguous scenes")
    ap.add_argument("--dynamics-spec", help="measured physics spec YAML (collision, inertia, joint stops, ...)")
    ap.add_argument("--registration-spec", help="registration YAML (T_world_capture, gravity alignment)")
    ap.add_argument("--selection", choices=["v2", "legacy"], default="v2",
                    help="v2 (default): observable physical joint selection decides; legacy: discovery decision + "
                         "render comparison (diagnostic, never certifies a physical joint)")
    ap.add_argument("--render-min-blocks", type=int, default=6,
                    help="independent held-out time blocks required before render evidence may veto (spec 8.3)")
    ap.add_argument("--assume-physical-joint", default=None, metavar="APPROVAL_NOTE",
                    help="user-approved override: export a production asset from a LEGACY 'IDENTIFIED' selection; "
                         "recorded as 'assumed' provenance (never needed under v2)")
    ap.add_argument("--allow-inconsistent-diagnostics", action="store_true",
                    help="MODEL_INCONSISTENT: still train both candidate branches as diagnostics (never exported)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.allow_non_ssd and not str(Path(args.output_root).resolve()).startswith(SSD):
        ap.error(f"--output-root must be on the Extreme SSD ({SSD})")
    stages = STAGES if args.stage == "all" else tuple(s.strip() for s in args.stage.split(","))
    bad = [s for s in stages if s not in STAGES]
    if bad:
        ap.error(f"unknown stage(s) {bad}")
    r = Run(args)
    if args.dry_run:
        print(f"joint-aware call graph for {r.scene.name} (budget {args.budget}: {r.iters} it/stage):")
        print("   bypassed legacy: track_2part, segment.py, bundle_preprocess, icp/color_icp_gpu, prepare_multiview, "
              "estimate_joint, twopart obj-pose adjustment, art axis/origin/offset/q optimisation")
    r.root.mkdir(parents=True, exist_ok=True)
    for s in stages:
        t0 = time.time()
        log(f"== {s}")
        try:
            if s != "audit" and s != "discover" and "discovery" not in r.m:
                raise StageError("run the discover stage first")
            FUNCS[s](r)
        except (StageError, AuthorityError, FileNotFoundError, ValueError) as e:
            r.set(s, "FAILED", error=str(e), elapsed_s=time.time() - t0)
            log(f"FAILED {s}: {e}")
            return 1
        r.m["stages"][s]["elapsed_s"] = time.time() - t0
        if not args.dry_run:
            atomic_write_json(r.manifest_path, r.m)
        log(f"   {s}: {r.m['stages'][s]['status']} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
