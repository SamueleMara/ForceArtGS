#!/usr/bin/env python3
"""Root-cause audit runner: traced baseline runs with an explicit camera policy (no training).

Camera policy (``camera_policy.json``): the default fused set is ``cam_001 cam_002``; if either is
missing the run fails (no silent substitution).  ``--control-cameras cam_000`` adds cameras that
are run and audited *individually* (labelled ``isolated_control``) and are never fused unless
``--stress-test-fusion`` is given, which writes a separately labelled multiview run.

Everything is written under ``--audit-root/<scene>/<run_id>/``; the raw scene is read only.
Each camera is converted, preprocessed, tracked and registered with the unmodified pipeline
(``samuele/run_tcp_contact_ablation.py`` with ``--config baseline=none``) while
``FREEARTGS_TRACE_ROOT`` points to ``trace/<camera>/`` (samuele/debug_audit/trace.py), so the
only difference from a production run is the trace output.  ``--no-trace`` gives the
trace-off control used to prove that.

    python samuele/debug_audit/run_audit.py --scene "<scene>" --audit-root <dir> --run-id B0 \\
        --frame-intervals 70:129 --frame-stride 2 --artifacts <artifacts.yaml> --control-cameras cam_000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_CAMERAS = ("cam_001", "cam_002")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def available_cameras(scene):
    images = scene / "colmap/sparse/0/images.txt"
    cams = sorted(p.name for p in (scene / "rgb").iterdir() if p.is_dir()) if (scene / "rgb").is_dir() else []
    return cams, images.is_file()


def camera_policy(scene, requested, controls, stress):
    cams, has_calibration = available_cameras(scene)
    fused = list(requested) if requested else list(DEFAULT_CAMERAS)
    missing = [c for c in fused + list(controls) if c not in cams]
    if missing:
        raise SystemExit(f"camera policy: {missing} not present in {scene / 'rgb'} (available {cams}); "
                         "no other camera is substituted -- pass --cameras explicitly")
    overlap = set(fused) & set(controls)
    if overlap:
        raise SystemExit(f"camera policy: {sorted(overlap)} listed both as fused and as control")
    return dict(default_fused=list(DEFAULT_CAMERAS), fused=fused, fused_is_default=not requested,
                isolated_controls=list(controls), stress_test_fusion=bool(stress),
                available=cams, calibration_file_present=has_calibration,
                note=("cam_000 is excluded from calibrated fusion by default (suspect cross-camera alignment) and "
                      "kept as an individually audited observability/calibration control; extrinsic confidence and "
                      "motion observability are scored separately"))


def tool_versions():
    code = ("import json,sys,torch,open3d,numpy;import nerfstudio,gsplat;"
            "print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,"
            "cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
            "open3d=open3d.__version__,numpy=numpy.__version__,nerfstudio_file=nerfstudio.__file__,"
            "gsplat=gsplat.__version__,gsplat_file=gsplat.__file__)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    return json.loads(out.stdout) if out.returncode == 0 else {"error": out.stderr[-2000:]}


def git_state():
    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=REPO, capture_output=True).stdout
    return dict(head=git("rev-parse", "HEAD").strip(), branch=git("branch", "--show-current").strip(),
                diff_sha256=hashlib.sha256(diff).hexdigest(), status_short=git("status", "--short").splitlines(),
                gitlinks=[l for l in git("ls-files", "-s").splitlines() if l.startswith("160000")])


def write_source_map(dataset, out_csv, pingpong=True, exclude_boundary=True):
    """raw source frame <-> converted output stem <-> tracking sequence index (ping-pong)."""
    from estimotion_2part.prior_context import build_sequence_index_map
    conv = json.loads((dataset / "metadata/freeart_conversion.json").read_text())
    rows = conv["frame_map"]
    seq = build_sequence_index_map(len(rows), pingpong, exclude_boundary)
    with open(out_csv, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["sequence_index", "sequence_kind", "pingpong_iteration", "output_index", "output_stem",
                    "source_stem", "source_rgbd_index", "unique_physical_frame"])
        for i, (orig, kind) in enumerate(zip(seq["original"], seq["kind"])):
            row = rows[int(orig)]
            w.writerow([i, kind, 0 if kind == "forward" else 1, int(orig), row["output_stem"], row["source_stem"],
                        row["source_index"], kind == "forward"])
    return len(rows), len(seq["original"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--cameras", nargs="+", default=None, help="explicit override of the fused set")
    p.add_argument("--control-cameras", nargs="*", default=[])
    p.add_argument("--stress-test-fusion", action="store_true")
    p.add_argument("--frame-intervals", nargs="+", required=True)
    p.add_argument("--frame-stride", type=int, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--stages", nargs="+", default=["convert", "preprocess", "track", "icp", "multiview"])
    p.add_argument("--no-trace", action="store_true", help="trace-off control run")
    args = p.parse_args(argv)

    scene = args.scene.resolve()
    run = (args.audit_root / scene.name / args.run_id).resolve()
    if scene in run.parents or run == scene:
        raise SystemExit("audit root must be outside the scene")
    policy = camera_policy(scene, args.cameras, args.control_cameras, args.stress_test_fusion)
    run.mkdir(parents=True, exist_ok=True)
    (run / "camera_policy.json").write_text(json.dumps(policy, indent=1))
    work = run / "work"
    manifest = dict(run_id=args.run_id, scene=str(scene), argv=sys.argv, started=time.time(), git=git_state(),
                    host=platform.node(), conda_env=os.environ.get("CONDA_DEFAULT_ENV"), tools=tool_versions(),
                    calibration_sha256=sha256(scene / "colmap/sparse/0/images.txt"),
                    cameras_json_sha256=sha256(scene / "metadata/cameras.json") if (scene / "metadata/cameras.json").is_file() else "MISSING",
                    artifacts_yaml=str(args.artifacts), artifacts_sha256=sha256(args.artifacts),
                    trace=not args.no_trace, policy=policy, runs=[])
    driver = [sys.executable, str(REPO / "samuele/run_tcp_contact_ablation.py"), "--scene", str(scene),
              "--frame-intervals", *args.frame_intervals, "--frame-stride", str(args.frame_stride),
              "--work-root", str(work), "--artifacts", str(args.artifacts), "--config", "baseline=none"]
    per_camera_stages = [s for s in args.stages if s != "multiview"]
    for cam in policy["fused"] + policy["isolated_controls"]:
        env = dict(os.environ)
        if not args.no_trace:
            env.update(FREEARTGS_TRACE_ROOT=str(run / "trace" / cam), FREEARTGS_TRACE_RUN_ID=args.run_id,
                       FREEARTGS_TRACE_SCENE=scene.name, FREEARTGS_TRACE_CAMERA=cam,
                       FREEARTGS_TRACE_CONFIG=str(work / "configs" / f"baseline_{cam}.yaml"))
            if (run / "trace" / cam).exists() and "track" in per_camera_stages:
                shutil.rmtree(run / "trace" / cam)   # one trace per run; never mix two executions
        started = time.time()
        if per_camera_stages:
            subprocess.run(driver + ["--cameras", cam, "--stages", *per_camera_stages], cwd=REPO, env=env, check=True)
        role = "isolated_control" if cam in policy["isolated_controls"] else "fused"
        manifest["runs"].append(dict(camera=cam, role=role, stages=per_camera_stages, seconds=time.time() - started))
        data = work / "datasets" / f"baseline_{cam}"
        if (data / "metadata/freeart_conversion.json").is_file():
            n_unique, n_seq = write_source_map(data, run / f"source_map_{cam}.csv")
            manifest["runs"][-1].update(unique_frames=n_unique, tracking_sequence_length=n_seq)
            cfg = work / "configs" / f"baseline_{cam}.yaml"
            if cfg.is_file():
                shutil.copy2(cfg, run / f"config_effective_{cam}.yaml")
    if "multiview" in args.stages:
        sets = [("baseline", policy["fused"], "fused_default" if policy["fused_is_default"] else "fused_override")]
        if args.stress_test_fusion and policy["isolated_controls"]:
            sets.append(("stress_all", policy["fused"] + policy["isolated_controls"], "stress_test_not_accepted_calibration"))
        from preprocessing.prepare_multiview import prepare_multiview
        for name, cams, label in sets:
            env_backup = dict(os.environ)
            if not args.no_trace:
                os.environ.update(FREEARTGS_TRACE_ROOT=str(run / "trace" / f"multiview_{name}"), FREEARTGS_TRACE_RUN_ID=args.run_id,
                                  FREEARTGS_TRACE_SCENE=scene.name, FREEARTGS_TRACE_CAMERA="+".join(cams))
            out = work / "datasets" / f"{name}_mv"
            try:
                prepare_multiview(scene, {c: work / "datasets" / f"baseline_{c}" for c in cams}, out)
                manifest["runs"].append(dict(multiview=name, cameras=cams, label=label, output=str(out)))
            finally:
                os.environ.clear()
                os.environ.update(env_backup)
    manifest["finished"] = time.time()
    (run / "audit_run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    with open(args.audit_root / "run_matrix.csv", "a", newline="") as handle:
        w = csv.writer(handle)
        if handle.tell() == 0:
            w.writerow(["run_id", "scene", "fused_cameras", "controls", "frame_intervals", "frame_stride", "trace", "stages", "intervention"])
        w.writerow([args.run_id, scene.name, " ".join(policy["fused"]), " ".join(policy["isolated_controls"]),
                    " ".join(args.frame_intervals), args.frame_stride, not args.no_trace, " ".join(args.stages),
                    "none (instrumentation only)" if not args.no_trace else "none (trace-off control)"])
    print(f"audit run written to {run}")


if __name__ == "__main__":
    main()
