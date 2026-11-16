#!/usr/bin/env python3
"""Controlled A/B/C ablation up to ICP and multiview preparation (no Gaussian training).

Every configuration shares the *same* converted frames, contact artifacts and preprocessing
(features, masks): each camera is converted and preprocessed once, then the prepared
dataset is copied per configuration and only tracking (with that configuration's
``tracking:`` overrides), ICP and ``prepare_multiview`` are re-run.  Original datasets are
read only; everything is written under ``--work-root``.

    python samuele/run_tcp_contact_ablation.py --scene <real scene> --cameras cam_000 cam_002 \\
        --frame-intervals 70:129 --frame-stride 2 --work-root <dir> \\
        --artifacts <overrides building all artifacts>.yaml \\
        --config baseline=none --config phaseA=<a.yaml> --config phaseB=<b.yaml> ...

``--config NAME=none`` is the unmodified baseline.  Stages can be limited with ``--stages``.
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

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

STAGES = ("convert", "preprocess", "track", "icp", "multiview")


def run(cmd, log, env=None):
    started = time.time()
    with open(log, "a") as handle:
        handle.write(f"\n$ {' '.join(map(str, cmd))}\n")
        handle.flush()
        subprocess.run(list(map(str, cmd)), cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True, env=env)
    return time.time() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", required=True)
    parser.add_argument("--frame-intervals", nargs="+", default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True,
                        help="Overrides YAML whose tcp_contact_evidence/contact_moving_prompts build the shared artifacts")
    parser.add_argument("--config", action="append", required=True, metavar="NAME=YAML|none")
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--multiview-reference-selection", choices=("motion", "quality"), default="motion")
    args = parser.parse_args(argv)

    from pipelines.train_local_datasets import apply_tracking_overrides, load_tracking_overrides, prepare_contact_artifacts
    work = args.work_root.resolve()
    scene = args.scene.resolve()
    if scene == work or scene in work.parents or work in scene.parents:
        raise ValueError("work root must be outside the scene")
    for sub in ("input", "datasets", "configs", "outputs", "logs"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
    artifacts = load_tracking_overrides(args.artifacts)
    configs = {}
    for item in args.config:
        name, path = item.split("=", 1)
        # Per-configuration files only switch tracking flags; contact artifacts come from --artifacts.
        configs[name] = None if path == "none" else (yaml.safe_load(Path(path).read_text()) or {})
        if configs[name] is not None and set(configs[name]) - {"tracking", "contact_moving_prompts"}:
            raise ValueError(f"{path}: per-configuration files may only contain tracking: and "
                             "contact_moving_prompts: sections (other artifacts are shared, from --artifacts)")
    timings = {}
    for cam in args.cameras:
        base = f"base_{cam}"
        input_dir, config_path = work / "input" / base, work / "configs" / f"{base}.yaml"
        log = work / "logs" / f"{base}.log"
        if "convert" in args.stages:
            cmd = [sys.executable, "samuele/convert_standard_to_freeart.py", "--input", scene, "--camera", cam,
                   "--output", input_dir, "--overwrite", "--write-config", config_path,
                   "--dataset-root", work / "datasets", "--outputs-root", work / "outputs",
                   "--frame-stride", args.frame_stride]
            if args.frame_intervals:
                cmd += ["--frame-intervals", *args.frame_intervals]
            timings[f"{base}/convert"] = run(cmd, log, env)
            prepare_contact_artifacts(scene, input_dir, artifacts)
        if "preprocess" in args.stages:
            timings[f"{base}/preprocess"] = run(["bash", "pipelines/freeartgs.sh", "--config", config_path,
                                                 "--preprocess_only"], log, env)
        for name, overrides in configs.items():
            job = f"{name}_{cam}"
            data = work / "datasets" / job
            job_config = work / "configs" / f"{job}.yaml"
            job_log = work / "logs" / f"{job}.log"
            if "track" in args.stages:
                if data.exists():
                    shutil.rmtree(data)
                shutil.copytree(work / "datasets" / base, data, ignore=shutil.ignore_patterns("tracking"))
                prompts_cfg = (overrides or {}).get("contact_moving_prompts")
                if prompts_cfg:  # per-configuration Phase B prompts on the identical prepared frames
                    from samuele.contact_moving_part_prompts import generate, write_outputs
                    prompts, report = generate(data, data / "metadata/tcp_contact",
                                               search_radius_m=float(prompts_cfg["search_radius_m"]),
                                               depth_margin_m=float(prompts_cfg["depth_margin_m"]),
                                               max_positive=int(prompts_cfg.get("max_positive", 5)),
                                               tool_exclusion_radius_m=prompts_cfg.get("tool_exclusion_radius_m"),
                                               tool_exclusion_start_m=prompts_cfg.get("tool_exclusion_start_m"))
                    write_outputs(prompts, report, data / "metadata", data)
                config = yaml.safe_load(config_path.read_text())
                config["paths"]["object_name"] = job
                job_config.write_text(yaml.safe_dump(config, sort_keys=False))
                if overrides is not None:
                    apply_tracking_overrides(job_config, overrides)
                timings[f"{job}/track"] = run(["bash", "pipelines/freeartgs.sh", "--config", job_config,
                                               "--tracking_only"], job_log, env)
            if "icp" in args.stages:
                timings[f"{job}/icp"] = run(["bash", "pipelines/icp.sh", "--config", job_config], job_log, env)
    if "multiview" in args.stages and len(args.cameras) >= 2:
        from preprocessing.prepare_multiview import prepare_multiview
        for name in configs:
            started = time.time()
            from preprocessing.prepare_multiview import UnresolvedReferenceError
            try:
                prepare_multiview(scene, {cam: work / "datasets" / f"{name}_{cam}" for cam in args.cameras},
                                  work / "datasets" / f"{name}_mv", reference_selection=args.multiview_reference_selection)
            except UnresolvedReferenceError as error:   # recorded, not hidden: see <name>_mv/multiview_quality.json
                timings[f"{name}_mv/unresolved"] = str(error)
            timings[f"{name}_mv/multiview"] = time.time() - started
    manifest = work / "ablation_manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.is_file() else {}
    previous.setdefault("runs", []).append(dict(argv=sys.argv, scene=str(scene), cameras=args.cameras,
                                                frame_intervals=args.frame_intervals, frame_stride=args.frame_stride,
                                                configs={k: (None if v is None else v) for k, v in configs.items()},
                                                artifacts=str(args.artifacts), stages=args.stages, seconds=timings))
    manifest.write_text(json.dumps(previous, indent=1, default=str) + "\n")
    print(json.dumps(timings, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
