#!/usr/bin/env python3
"""Train one shared FreeArtGS model per object using selected calibrated cameras (default: all)."""
import argparse
import errno
import json
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import yaml
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_ROOT = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render")
sys.path.insert(0, str(REPO_ROOT))

from samuele.convert_standard_to_freeart import (
    build_triplets, convert, load_camera_entry, maybe_write_config, parse_args as conversion_args,
    normalize_frame_intervals, parse_frame_interval, select_triplets,
)
from preprocessing.prepare_multiview import calibration_poses, prepare_multiview
from evaluate.evaluate_local_joint_axis import evaluate_joint_axis, load_gt_axis
from utils.local_storage import DEFAULT_OUTPUTS_ROOT, configure_runtime, ensure_free_space


def storage_failure(error, log_path):
    if isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT):
        return True
    if isinstance(error, subprocess.CalledProcessError) and log_path.is_file():
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 65536))
            tail = log.read().lower()
        return any(message in tail for message in (b"no space left on device", b"disk quota exceeded",
                                                   b"no usable temporary directory"))
    return False


def write_summary(path, rows):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, indent=2))
    temporary.replace(path)


def result_is_complete(record, output, iterations, evaluate=False, visualize=False):
    if not record or record.get("status") != "passed" or not record.get("training_completed"):
        return False
    if visualize and record.get("visualization_error"):
        return False
    required = ["config.yml", f"nerfstudio_models/step-{iterations - 1:09d}.ckpt",
                "object_3dgs.ply", "3dgs_part_weight.txt", "articulation.json", "transforms_train.json"]
    json_files = ["articulation.json", "transforms_train.json"]
    if evaluate:
        required.append("joint_axis_error.json")
        json_files.append("joint_axis_error.json")
    if visualize:
        basename = "motion_vs_gt" if evaluate else "motion_axis_debug"
        required.extend([f"comparison/{basename}.mp4", f"comparison/{basename}.gif",
                         "comparison/comparison_info.json"])
        json_files.append("comparison/comparison_info.json")
    try:
        if any(not (output / name).is_file() or (output / name).stat().st_size == 0 for name in required):
            return False
        contents = {name: json.loads((output / name).read_text()) for name in json_files}
        if not contents["articulation.json"].get("frames") or not contents["transforms_train.json"]:
            return False
        if evaluate and not np.isfinite(contents["joint_axis_error.json"]["joint_axis_error_deg"]):
            return False
        if visualize and not contents["comparison/comparison_info.json"].get("timestamps"):
            return False
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False
    return True


def discover_jobs(dataset_root):
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Raw datasets directory is unavailable: {dataset_root}. "
                                "Pass --datasets with the existing source directory.")
    scenes = sorted(p for p in dataset_root.iterdir() if p.is_dir()
                    and p.name != "episode_000000"
                    and (p / "metadata/cameras.json").is_file()
                    and any(d.is_dir() for d in (p / "rgb").glob("*")))
    if not scenes:
        raise ValueError(f"No calibrated multi-camera RGB-D datasets found in {dataset_root}")
    return scenes


def require_source(scene):
    if not scene.is_dir():
        raise FileNotFoundError(f"Raw source disappeared after validation: {scene}. "
                                "Do not move/delete source datasets during a run; retry with --datasets at their current location.")


def run_stage(config_path, log, stage):
    subprocess.run(["bash", "pipelines/freeartgs.sh", "--config", str(config_path), stage],
                   cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


CONTACT_PHASE_FLAGS = ("tcp_supervision_enabled", "moving_prompt_enabled", "contact_surface_enabled")


def load_tracking_overrides(path):
    """Opt-in TCP/contact phase overrides (docs/tcp_contact_tracking.md); None keeps the baseline."""
    if path is None:
        return None
    overrides = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(overrides) - {"tracking", "tcp_contact_evidence", "contact_moving_prompts", "force_contact"}
    if unknown:
        raise ValueError(f"Unknown tracking-override sections: {sorted(unknown)}")
    tracking = overrides.get("tracking", {}) or {}
    evidence = overrides.get("tcp_contact_evidence", {}) or {}
    for key in ("contact_file", "tool_geometry", "pose_archive", "parquet_python", "calibration_map"):
        if key == "contact_file" and str(evidence.get(key, "")).lower() == "force":
            continue  # generated from the recorded wrench at preparation time
        if evidence.get(key) and not Path(evidence[key]).expanduser().exists():
            raise FileNotFoundError(f"tcp_contact_evidence.{key} not found: {evidence[key]}")
    if (tracking.get("moving_prompt_enabled") or tracking.get("contact_surface_enabled")) and not evidence.get("contact_file"):
        raise ValueError("Phases B/C need tcp_contact_evidence.contact_file: a JSON/CSV path, or 'force' to derive "
                         "contact automatically from the recorded wrench")
    if tracking.get("tcp_supervision_enabled") and not evidence.get("contact_file"):
        print("NOTE: no contact source; Phase A does not need one (contact state is uncertain everywhere)", flush=True)
    if tracking.get("moving_prompt_enabled"):
        prompts = overrides.get("contact_moving_prompts", {}) or {}
        missing = [k for k in ("search_radius_m", "depth_margin_m") if not prompts.get(k)]
        if missing:
            raise ValueError(f"moving_prompt_enabled needs contact_moving_prompts.{missing} set explicitly")
    return overrides


def export_force_contact(scene, output_dir, evidence, options):
    """Automatic contact annotations from the recorded wrench (needs a Python with polars/pyarrow)."""
    python = Path(evidence["parquet_python"]).expanduser() if evidence.get("parquet_python") else Path(sys.executable)
    command = [str(python), "-B", str(REPO_ROOT / "samuele/export_force_contact_annotations.py"),
               "--scene", str(scene), "--output", str(output_dir)]
    flags = dict(baseline_duration_s="--baseline-duration-s", contact_z="--contact-z", absent_z="--absent-z",
                 min_contact_duration_s="--min-contact-duration-s", tool_mass_kg="--tool-mass-kg",
                 max_uncompensated_rotation_deg="--max-uncompensated-rotation-deg", corrections="--corrections")
    unknown = set(options) - set(flags) - {"baseline_frames"}
    if unknown:
        raise ValueError(f"Unknown force_contact options: {sorted(unknown)}")
    for key, flag in flags.items():
        if options.get(key) is not None:
            command += [flag, str(Path(options[key]).expanduser() if key == "corrections" else options[key])]
    if options.get("baseline_frames") is not None:
        command += ["--baseline-frames", *(str(int(v)) for v in options["baseline_frames"])]
    subprocess.run(command, check=True)
    annotations = Path(output_dir) / "contact_annotations.json"
    summary = json.loads((Path(output_dir) / "contact_summary.json").read_text())
    print(f"Force contact annotations: {summary['counts_rgbd']} (RGB-D frames), intervals {summary['intervals']}")
    return annotations


def prepare_contact_artifacts(scene, camera_input_dir, overrides):
    """Build tcp_contact (and Phase B seeds) inside the converted input; preprocess copies metadata."""
    from samuele.tcp_contact_evidence import load_aligned_tcp, save_tcp_contact
    tracking = overrides.get("tracking", {}) or {}
    if not any(tracking.get(flag) for flag in CONTACT_PHASE_FLAGS):
        return
    evidence = overrides.get("tcp_contact_evidence", {}) or {}
    path = lambda key: Path(evidence[key]).expanduser() if evidence.get(key) else None
    geometry = tracking.get("tool_contact_geometry") or evidence.get("tool_geometry")
    contact_file = path("contact_file")
    if str(evidence.get("contact_file", "")).lower() == "force":
        contact_file = export_force_contact(scene, camera_input_dir / "metadata/force_contact", evidence,
                                            overrides.get("force_contact", {}) or {})
    sequence = load_aligned_tcp(scene, camera_input_dir, contact_source=contact_file,
                                tool_geometry=Path(geometry).expanduser() if geometry else None,
                                pose_archive=path("pose_archive"), parquet_python=path("parquet_python"),
                                calibration_map=path("calibration_map"))
    save_tcp_contact(sequence, camera_input_dir / "metadata/tcp_contact", forbidden_roots=[scene])
    if tracking.get("moving_prompt_enabled"):
        from samuele.contact_moving_part_prompts import generate, write_outputs
        prompts_cfg = overrides["contact_moving_prompts"]
        prompts, report = generate(camera_input_dir, camera_input_dir / "metadata/tcp_contact",
                                   search_radius_m=float(prompts_cfg["search_radius_m"]),
                                   depth_margin_m=float(prompts_cfg["depth_margin_m"]),
                                   max_positive=int(prompts_cfg.get("max_positive", 5)),
                                   tool_exclusion_radius_m=prompts_cfg.get("tool_exclusion_radius_m"),
                                   tool_exclusion_start_m=prompts_cfg.get("tool_exclusion_start_m"))
        write_outputs(prompts, report, camera_input_dir / "metadata", camera_input_dir)


def apply_tracking_overrides(config_path, overrides):
    if not overrides or not overrides.get("tracking"):
        return
    config = yaml.safe_load(Path(config_path).read_text())
    config.setdefault("tracking", {}).update(overrides["tracking"])
    Path(config_path).write_text(yaml.safe_dump(config, sort_keys=False))


def camera_argument(value):
    match = re.fullmatch(r"(?:cam_?)?(\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError("Use a camera ID such as cam_001, cam001 or 001")
    return f"cam_{int(match[1]):03d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATASETS_ROOT, help="Read-only raw source datasets.")
    camera_selection = parser.add_mutually_exclusive_group()
    camera_selection.add_argument("--all-cameras", action="store_true", help="All cameras are used together by default.")
    camera_selection.add_argument("--cameras", nargs="+", type=camera_argument,
                                  help="Use only these cameras, e.g. cam_001 cam_002 (also cam001 or 001). "
                                       "Select at least two; default: all cameras.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--object", help="Optional raw object folder name to train.")
    selection.add_argument("--objects", nargs="+", help="Train exactly these raw object folder names, in order.")
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT,
                        help="Target for results, converted inputs, prepared datasets, configs, caches and temporary files.")
    parser.add_argument("--iterations", type=int, default=30000, help="Iterations per training stage.")
    parser.add_argument("--max-frames", type=int, help="Limit timestamps per camera for a smoke test.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-intervals", nargs="+", type=parse_frame_interval,
                        help="Inclusive zero-based positions in sorted source frames, e.g. 20:80 120:180. "
                             "Shared by all cameras/objects; applied before frame-stride and max-frames.")
    parser.add_argument("--dry-run", action="store_true", help="Validate calibration and data without writing or training.")
    parser.add_argument("--evaluate-joint-axis", action="store_true", help="Compare the trained joint axis to GT and report error in degrees.")
    parser.add_argument("--visualize-results", action="store_true",
                        help="Make videos for the selected cameras with estimated axes; also evaluate joint GT when available.")
    parser.add_argument("--reuse-tracking", action="store_true", help="Reuse matching prepared camera sequences; recompute ICP if poses have missing timestamp slots.")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip successful jobs with final checkpoint, exports and any requested GT reports/videos present.")
    parser.add_argument("--tracking-overrides", type=Path,
                        help="Opt-in YAML with tracking:/tcp_contact_evidence:/contact_moving_prompts: sections for the "
                             "robot-TCP contact phases (see configs/tcp_contact_tracking.example.yaml). "
                             "Job names get a _<file stem> suffix so ablations do not collide.")
    parser.add_argument("--multiview-reference-selection", choices=("motion", "quality"), default="motion",
                        help="motion: legacy largest-motion reference camera; quality: RGB-D motion-quality evaluation "
                             "(preprocessing/multiview_quality.py), which stops when no camera is independently supported.")
    args = parser.parse_args()
    overrides = load_tracking_overrides(args.tracking_overrides)
    if args.cameras is not None:
        if len(args.cameras) != len(set(args.cameras)):
            parser.error("Camera selection contains duplicate IDs")
        if len(args.cameras) < 2:
            parser.error("Multi-view training requires at least two selected cameras")
        args.cameras = sorted(args.cameras)
    args.frame_intervals = normalize_frame_intervals(args.frame_intervals)
    if args.iterations < 1 or args.frame_stride < 1 or (args.max_frames is not None and args.max_frames < 2):
        parser.error("Iterations/stride must be positive and max-frames must be at least two")
    dataset_root = args.datasets.expanduser().resolve()
    outputs_root = args.outputs_root.expanduser().resolve()
    scenes = discover_jobs(dataset_root)
    selected_names = args.objects or ([args.object] if args.object else None)
    if selected_names:
        available = {scene.name: scene for scene in scenes}
        missing = [name for name in selected_names if name not in available]
        if missing:
            parser.error(f"Raw objects not found: {', '.join(missing)}")
        if len(set(selected_names)) != len(selected_names):
            parser.error("Object selection contains duplicate names")
        scenes = [available[name] for name in selected_names]
    print(f"Found {len(scenes)} multi-view object jobs; episode_000000 is excluded", flush=True)
    print("Experimental multiview adapter: independent camera segmentations and one-view motion "
          "initialization. Known part-label errors under occlusion; training completion does not "
          "validate reconstruction quality. See docs/official_pipeline_review.md.", flush=True)
    work = outputs_root / "work"
    input_root = work / "input"
    prepared_root = work / "datasets"
    configs = work / "configs"
    log_root = outputs_root / "batch_logs"
    for directory in (input_root, prepared_root, configs, log_root):
        if not directory.resolve().is_relative_to(outputs_root):
            raise ValueError(f"Working directory escapes the output target: {directory}")
    if any(outputs_root == scene or outputs_root.is_relative_to(scene) for scene in scenes):
        raise ValueError("The output target must be outside the raw source object folders")
    print(f"All generated data and runtime caches: {outputs_root}", flush=True)
    existing_results = []
    if (selected_names or args.skip_completed or args.frame_intervals or args.cameras) and (log_root / "summary.json").is_file():
        existing_results = json.loads((log_root / "summary.json").read_text())
    records = {row["job"]: row for row in existing_results}
    jobs = []
    skipped = 0
    for scene in scenes:
        cameras = sorted(p.name for p in (scene / "rgb").iterdir() if p.is_dir())
        if args.cameras:
            missing = sorted(set(args.cameras) - set(cameras))
            if missing:
                raise ValueError(f"Selected cameras not found in {scene}: {', '.join(missing)}; "
                                 f"available: {', '.join(cameras)}")
            cameras = args.cameras
        evaluate_axis = args.evaluate_joint_axis or (
            args.visualize_results and (scene / "pointcloud/joint_metadata.json").is_file())
        name = f"{scene.name}_multiview"
        if args.cameras:
            name += "_cams" + "-".join(camera.removeprefix("cam_") for camera in cameras)
        if args.frame_intervals:
            name += "_i" + "_".join(f"{start}-{end}" for start, end in args.frame_intervals)
        if args.max_frames is not None or args.frame_stride != 1:
            name += f"_f{args.max_frames or 'all'}_s{args.frame_stride}"
        if args.tracking_overrides is not None:
            name += f"_{args.tracking_overrides.stem}"
        if args.skip_completed and result_is_complete(records.get(name), outputs_root / name / "splatfacto-art",
                                                      args.iterations, evaluate_axis, args.visualize_results):
            print(f"SKIP {name}: completed results verified", flush=True)
            skipped += 1
            continue
        if evaluate_axis:
            load_gt_axis(scene)
        if len(cameras) < 2:
            raise ValueError(f"Multi-view training requires at least two cameras: {scene}")
        poses = calibration_poses(scene, cameras)
        if any(camera not in poses for camera in cameras):
            raise ValueError(f"Missing COLMAP extrinsics in {scene}")
        options = []
        source_stems = None
        for camera in cameras:
            camera_name = f"{name}_{camera.replace('_', '')}"
            out = input_root / camera_name
            command = ["--input", str(scene), "--camera", camera, "--output", str(out),
                       "--write-config", str(configs / f"{camera_name}.yaml"),
                       "--dataset-root", str(prepared_root), "--outputs-root", str(outputs_root),
                       "--frame-stride", str(args.frame_stride), "--dry-run"]
            if args.max_frames is not None:
                command.extend(["--max-frames", str(args.max_frames)])
            if args.frame_intervals:
                command.extend(["--frame-intervals", *[f"{start}:{end}" for start, end in args.frame_intervals]])
            opt = conversion_args(command)
            convert(opt)
            triplets, _ = build_triplets(scene, camera, opt.depth_source, opt.allow_missing_mask)
            stems = [frame.stem for frame in select_triplets(triplets, opt)]
            if len(stems) < 2:
                raise ValueError(f"Select at least two timestamps for reconstruction: {scene}, {camera}")
            if source_stems is not None and stems != source_stems:
                raise ValueError(f"Unsynchronized camera timestamps in {scene}: {camera}")
            source_stems = stems
            if out.exists():
                provenance = out / "metadata/freeart_conversion.json"
                if not provenance.exists():
                    raise ValueError(f"Refusing to replace an unrelated input: {out}")
                meta = json.loads(provenance.read_text())
                old_scene = Path(meta["scene_root"]).resolve()
                rebuild_moved_source = (not args.reuse_tracking and not old_scene.exists()
                                        and old_scene.name == scene.name)
                if (old_scene != scene and not rebuild_moved_source) or meta["selected_camera"] != camera:
                    raise ValueError(f"Prepared input belongs to another source: {out}")
                if old_scene != scene:
                    print(f"Source path changed: {old_scene} -> {scene}; rebuilding prepared data.", flush=True)
                opt.overwrite = True
            options.append((camera, camera_name, opt))
        print(f"  {name}: {len(cameras)} cameras will train one shared model", flush=True)
        print(f"    Selected cameras: {', '.join(cameras)}", flush=True)
        jobs.append((scene, name, options))
    print(f"Batch selection: {len(jobs)} jobs to run, {skipped} completed jobs skipped", flush=True)
    if args.dry_run:
        print("Dry run complete: no files written and no training started.")
        return
    if not jobs:
        print("All selected jobs are already complete; no training started.")
        return

    configure_runtime(outputs_root)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; run in the freeartgs environment outside the sandbox")
    configs.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    results = []
    previous_results = []
    if selected_names or args.skip_completed or args.frame_intervals or args.cameras:
        selected_jobs = {name for _, name, _ in jobs}
        previous_results = [row for row in existing_results
                            if row["job"] not in selected_jobs]
    for scene, name, options in jobs:
        log_path = log_root / f"{name}.log"
        print(f"TRAINING {name} — log: {log_path}", flush=True)
        stop_batch = False
        try:
            require_source(scene)
            ensure_free_space(outputs_root)
            with log_path.open("a" if args.reuse_tracking else "w") as log:
                camera_data = {}
                for camera, camera_name, opt in options:
                    require_source(scene)
                    ensure_free_space(outputs_root)
                    data = prepared_root / camera_name
                    provenance = data / "metadata/freeart_conversion.json"
                    reuse = args.reuse_tracking and provenance.is_file()
                    if reuse:
                        meta = json.loads(provenance.read_text())
                        triplets, _ = build_triplets(scene, camera, opt.depth_source, opt.allow_missing_mask)
                        expected = [frame.stem for frame in select_triplets(triplets, opt)]
                        if (Path(meta["scene_root"]).resolve() != scene or meta["selected_camera"] != camera
                                or [row["source_stem"] for row in meta["frame_map"]] != expected):
                            raise ValueError(f"Prepared tracking does not match selected raw frames: {data}")
                        print(f"REUSING tracking: {data}", flush=True)
                        maybe_write_config(opt, camera_name, load_camera_entry(scene, camera))
                        apply_tracking_overrides(opt.write_config, overrides)
                        rerun_icp = False
                        for part in (0, 1):
                            path = data / f"global_poses_{part}.npy"
                            poses = np.load(path) if path.is_file() else np.empty(0)
                            rerun_icp |= poses.shape != (len(expected), 4, 4) or not np.isfinite(poses).all()
                    else:
                        opt.dry_run = False
                        convert(opt)
                        if overrides is not None:
                            prepare_contact_artifacts(scene, opt.output.expanduser().resolve(), overrides)
                            apply_tracking_overrides(opt.write_config, overrides)
                        run_stage(opt.write_config, log, "--preprocess_only")
                        run_stage(opt.write_config, log, "--tracking_only")
                        rerun_icp = True
                    if rerun_icp:
                        subprocess.run(["bash", "pipelines/icp.sh", "--config", str(opt.write_config)],
                                       cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    camera_data[camera] = data
                ensure_free_space(outputs_root)
                require_source(scene)
                dataset = prepared_root / name
                prepare_multiview(scene, camera_data, dataset, reference_selection=args.multiview_reference_selection)
                config = yaml.safe_load(options[0][2].write_config.read_text())
                config["paths"]["object_name"] = name
                config["multiview"] = {"enabled": True}
                for stage in ("train_parts", "twopart_blend", "splatfacto_art"):
                    config[stage].update(max_num_iterations=args.iterations, camera_optimizer_mode="off")
                config_path = configs / f"{name}.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                run_stage(config_path, log, "--reconstruction_only")
            result = dict(job=name, status="passed", training_completed=True,
                          reconstruction_quality_validated=False, cameras=len(options),
                          camera_names=[camera for camera, _, _ in options], log=str(log_path))
            final_output = outputs_root / name / "splatfacto-art"
            evaluate_axis = args.evaluate_joint_axis or (
                args.visualize_results and (scene / "pointcloud/joint_metadata.json").is_file())
            if evaluate_axis:
                report = evaluate_joint_axis(scene, dataset, final_output)
                result.update(joint_axis_error_deg=report["joint_axis_error_deg"],
                              joint_axis_evaluation=str(final_output / "joint_axis_error.json"))
                for key in ("joint_type_correct", "joint_state_mae", "joint_state_unit", "evaluated_timestamps"):
                    if key in report:
                        result[key] = report[key]
                print(f"{name}: joint axis error = {report['joint_axis_error_deg']:.6f} degrees", flush=True)
                if "joint_state_mae" in report:
                    print(f"{name}: relative joint-state MAE = {report['joint_state_mae']:.6f} "
                          f"{report['joint_state_unit']}; joint type correct: {report['joint_type_correct']}", flush=True)
            if args.visualize_results:
                try:
                    with log_path.open("a") as log:
                        subprocess.run([sys.executable, str(REPO_ROOT / "evaluate/visualize_local_comparison.py"),
                                        "--scene", str(scene), "--dataset", str(dataset),
                                        "--results", str(final_output)], cwd=REPO_ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    basename = "motion_vs_gt" if evaluate_axis else "motion_axis_debug"
                    result["motion_comparison"] = str(final_output / "comparison" / f"{basename}.mp4")
                except Exception as error:
                    result["visualization_error"] = str(error)
                    stop_batch = storage_failure(error, log_path)
                    print(f"Comparison failed for {name}: {error}; see {log_path}", file=sys.stderr, flush=True)
            results.append(result)
        except Exception as error:
            stop_batch = not scene.is_dir() or storage_failure(error, log_path)
            print(f"FAILED {name}: {error}; see {log_path}", file=sys.stderr, flush=True)
            results.append(dict(job=name, status="failed", error=str(error), log=str(log_path)))
        write_summary(log_root / "summary.json", previous_results + results)
        if stop_batch:
            print("Stopping the batch after a source or storage failure; later objects were not started.", file=sys.stderr, flush=True)
            sys.exit(1)
    failed = sum(row["status"] == "failed" for row in results)
    print(f"Finished: {len(results) - failed} training jobs completed, {failed} failed. Logs: {log_root}")
    errors = [row["joint_axis_error_deg"] for row in results if "joint_axis_error_deg" in row]
    if errors:
        print(f"Mean joint axis error over {len(errors)} evaluated objects: {sum(errors) / len(errors):.6f} degrees")
    if failed or any("visualization_error" in row for row in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
