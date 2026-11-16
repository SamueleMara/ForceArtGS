#!/usr/bin/env python3
"""Offline robot-contact annotations from the recorded wrench (Phases B/C contact source).

Reads the native episode Parquet of a converted real scene, removes the sensor bias
measured in a contact-free baseline (and, when the tool mass is given, the change of tool
gravity with EE orientation), and classifies every native sample into three states:

    contact    noise-normalised force  z >= contact_z  inside a sustained episode
    absent     z <= absent_z  (indistinguishable from baseline noise)
    uncertain  anything else: between the thresholds, transient excursions shorter than
               the sustained duration, non-finite samples, or manual "uncertain" labels

``z = || (F - tare) / sigma ||`` with per-axis baseline noise ``sigma`` (robust MAD).  Uncertain
samples are never converted into confirmed absence or contact.  The export follows the
per-frame contract of :func:`samuele.tcp_contact_evidence.load_contact_state` (indexed by
raw ``source_frame_index``), so Phases B/C consume it unchanged, and stores thresholds,
provenance, RGB-D frame alignment and a diagnostic plot next to it.

Force identifies *when* the tool interacts.  Which object part moves is not decided here
(the operator confirmed the robot contacts the moving part); joint type and dynamics are
out of scope.  Torques are not used.

Run with a Python that has polars (or pyarrow) and matplotlib, e.g. the ``yolo26`` env::

    python samuele/export_force_contact_annotations.py --scene <converted scene> --output <dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

STATE_CODES = {"contact": 1, "absent": 0, "uncertain": -1}
STATE_NAMES = {v: k for k, v in STATE_CODES.items()}
G_BASE = np.array([0.0, 0.0, -9.80665])  # base_link z up (robot base convention)

DEFAULTS = dict(
    baseline_duration_s=0.5,      # same quiet start window as the existing force-event detector
    contact_z=6.0,                # same six-sigma rule as the existing force-event detector
    absent_z=3.0,                 # chi^2(3) cdf(9) = 0.971: most baseline samples fall below
    min_contact_duration_s=0.2,   # sustained interaction, not a single-sample spike
    max_uncompensated_rotation_deg=1.0,  # without tool mass, refuse larger orientation changes
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rotvec_to_matrix(rotvec):
    """(N,3) rotation vectors -> (N,3,3) matrices (Rodrigues)."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(rotvec, axis=1)
    out = np.repeat(np.eye(3)[None], len(rotvec), axis=0)
    nz = theta > 1e-12
    k = rotvec[nz] / theta[nz, None]
    K = np.zeros((k.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(theta[nz])[:, None, None], np.cos(theta[nz])[:, None, None]
    out[nz] = np.eye(3) + s * K + (1 - c) * (K @ K)
    return out


def rotation_angle_deg(R_a, R_b):
    rel = np.einsum("nji,njk->nik", R_a, R_b)
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def baseline_mask_from(t, frame_index, *, duration_s=None, frames=None):
    """Contact-free baseline samples: the first ``duration_s`` seconds or an explicit frame range."""
    if frames is not None:
        lo, hi = int(frames[0]), int(frames[1])
        mask = (frame_index >= lo) & (frame_index <= hi)
    else:
        mask = (t - t[0]) <= float(duration_s)
    if mask.sum() < 5:
        raise ValueError(f"Baseline has {int(mask.sum())} samples; need at least 5 to estimate noise")
    return mask


def gravity_residual(R_base_sensor, baseline_mask, tool_mass_kg):
    """Change of tool weight in sensor axes relative to the baseline mean orientation.

    The recorded force acts on the robot (object-on-robot reaction), so the tool weight
    appears as ``m * R^T g``; the constant baseline part is already removed by the tare.
    """
    weight = tool_mass_kg * np.einsum("nji,j->ni", R_base_sensor, G_BASE)
    return weight - weight[baseline_mask].mean(axis=0)


def detect_contact_states(t, force, baseline_mask, *, contact_z, absent_z, min_contact_duration_s,
                          gravity_correction=None):
    """Per-sample three-state labels from a force time series.

    Args:
        t: (N,) seconds, increasing.  force: (N,3) recorded force (N).
        baseline_mask: contact-free samples for tare and noise.
        gravity_correction: optional (N,3) to subtract (see :func:`gravity_residual`).
    Returns dict with ``state`` (int8: 1/0/-1), ``reason`` (list[str]), ``z``,
    ``force_corrected`` (N,3), ``tare_N``, ``sigma_N``, ``episodes`` (list of dicts).
    """
    if not (absent_z > 0 and contact_z > absent_z):
        raise ValueError("Require 0 < absent_z < contact_z")
    if min_contact_duration_s is None or min_contact_duration_s < 0:
        raise ValueError("min_contact_duration_s must be >= 0")
    t = np.asarray(t, dtype=np.float64)
    F = np.asarray(force, dtype=np.float64).reshape(-1, 3)
    if gravity_correction is not None:
        F = F - gravity_correction
    finite = np.isfinite(F).all(axis=1) & np.isfinite(t)
    base = baseline_mask & finite
    tare = F[base].mean(axis=0)
    mad = np.median(np.abs(F[base] - np.median(F[base], axis=0)), axis=0)
    sigma = 1.4826 * mad
    fallback = F[base].std(axis=0)
    sigma = np.where(sigma > 0, sigma, fallback)
    if not np.all(sigma > 0):
        raise ValueError("Baseline noise is zero on some axis; cannot normalise force")
    Fc = F - tare
    z = np.linalg.norm(Fc / sigma, axis=1)
    z[~finite] = np.nan

    state = np.full(len(t), STATE_CODES["uncertain"], dtype=np.int8)
    reason = ["between_thresholds"] * len(t)
    for i in range(len(t)):
        if not finite[i]:
            reason[i] = "nonfinite_measurement"
        elif z[i] <= absent_z:
            state[i], reason[i] = STATE_CODES["absent"], "force_within_baseline_noise"
    # Episodes: maximal runs without confirmed absence that contain z >= contact_z samples.
    candidate = finite & (z >= contact_z)
    episodes = []
    i = 0
    while i < len(t):
        if state[i] == STATE_CODES["absent"]:
            i += 1
            continue
        j = i
        while j + 1 < len(t) and state[j + 1] != STATE_CODES["absent"]:
            j += 1
        run = np.arange(i, j + 1)
        hits = run[candidate[run]]
        if len(hits):
            duration = float(t[hits[-1]] - t[hits[0]])
            sustained = duration >= min_contact_duration_s
            for k in hits:
                if sustained:
                    state[k], reason[k] = STATE_CODES["contact"], "sustained_force_above_contact_threshold"
                else:
                    reason[k] = "transient_force_shorter_than_min_duration"
            for k in run[~candidate[run]]:
                if finite[k] and sustained and hits[0] < k < hits[-1]:
                    reason[k] = "between_thresholds_within_contact_episode"
            episodes.append(dict(first_index=int(hits[0]), last_index=int(hits[-1]),
                                 duration_s=duration, sustained=bool(sustained),
                                 confirmed_samples=int(len(hits)) if sustained else 0,
                                 peak_force_N=float(np.linalg.norm(Fc[hits], axis=1).max())))
        i = j + 1
    return dict(state=state, reason=reason, z=z, force_corrected=Fc, tare_N=tare, sigma_N=sigma, episodes=episodes)


def apply_corrections(frame_index, state, reason, corrections):
    """Manual overrides: ``{"provenance": str, "corrections": [{"source_frames": [a, b], "state": ..., "reason": ...}]}``."""
    if not str(corrections.get("provenance", "")).strip():
        raise ValueError("Manual contact corrections must state their provenance")
    applied = []
    state, reason = state.copy(), list(reason)
    for item in corrections.get("corrections", []):
        lo, hi = (int(v) for v in item["source_frames"])
        if lo > hi:
            raise ValueError(f"Correction range {lo}>{hi}")
        name = item["state"]
        if name not in STATE_CODES:
            raise ValueError(f"Unknown correction state {name!r}; use contact/absent/uncertain")
        if not str(item.get("reason", "")).strip():
            raise ValueError("Every correction needs a reason")
        sel = np.nonzero((frame_index >= lo) & (frame_index <= hi))[0]
        for k in sel:
            state[k] = STATE_CODES[name]
            reason[k] = f"manual_correction: {item['reason']}"
        applied.append(dict(source_frames=[lo, hi], state=name, reason=item["reason"], samples=int(len(sel))))
    return state, reason, applied


def intervals_of(frame_index, state, code=STATE_CODES["contact"]):
    out, start, prev = [], None, None
    for f, s in zip(frame_index, state):
        if s == code and start is None:
            start = int(f)
        if s != code and start is not None:
            out.append([start, int(prev)])
            start = None
        prev = f
    if start is not None:
        out.append([start, int(prev)])
    return out


# ----------------------------------------------------------------------------- I/O


def read_episode(parquet):
    """Return dict(frame_index, timestamp, wrench (N,6), state (N,>=6)) from a LeRobot-style Parquet."""
    try:
        import polars as pl
        df = pl.read_parquet(parquet)
        get = lambda name: df[name].to_list()  # noqa: E731
        columns = df.columns
    except ImportError:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet)
        get = lambda name: table.column(name).to_pylist()  # noqa: E731
        columns = table.column_names
    for name in ("frame_index", "timestamp", "observation.wrench", "observation.state"):
        if name not in columns:
            raise KeyError(f"{parquet} has no column {name!r}; columns: {columns}")
    return dict(frame_index=np.asarray(get("frame_index"), dtype=np.int64),
                timestamp=np.asarray(get("timestamp"), dtype=np.float64),
                wrench=np.asarray(get("observation.wrench"), dtype=np.float64),
                state=np.asarray(get("observation.state"), dtype=np.float64))


def locate_inputs(scene):
    scene = Path(scene)
    conversion = json.loads((scene / "metadata/real_conversion.json").read_text())
    episode = int(conversion["episode"])
    raw = scene / "metadata/raw"
    parquet = raw / f"episode_{episode:06d}.parquet"
    info = raw / "info.json"
    if not parquet.is_file():
        parquet = Path(conversion["source"]) / "data/chunk-000" / f"episode_{episode:06d}.parquet"
        info = Path(conversion["source"]) / "meta/info.json"
    if not parquet.is_file():
        raise FileNotFoundError(f"No episode Parquet for {scene}")
    return conversion, parquet, info


def check_conventions(info, tool):
    """The wrench layout and EE-aligned sensor axes must be declared, not assumed."""
    names = info.get("wrench_names")
    if names[:3] != ["Fx_N", "Fy_N", "Fz_N"]:
        raise ValueError(f"Unexpected wrench_names {names}; refusing to guess force columns")
    state_names = info.get("state_names", [])
    if state_names[:6] != ["pos_x", "pos_y", "pos_z", "rotvec_x", "rotvec_y", "rotvec_z"]:
        raise ValueError(f"Unexpected state_names {state_names[:6]}")
    frame = (tool or {}).get("wrench_frame") or {}
    R = np.asarray(frame.get("R_ee_force_sensor", np.eye(3)), dtype=np.float64)
    if frame.get("axes") != "ee":
        raise ValueError("metadata/robot_tool.json must declare wrench_frame.axes == 'ee'")
    return R, frame


def plot_diagnostics(path, t, frame_index, result, state, thresholds, rgbd_frames, speed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    Fc = result["force_corrected"]
    ax[0].plot(frame_index, Fc, lw=0.8)
    ax[0].plot(frame_index, np.linalg.norm(Fc, axis=1), "k", lw=1.0)
    ax[0].legend(["Fx", "Fy", "Fz", "|F - tare|"], fontsize=8, ncol=4)
    ax[0].set_ylabel("force (N), sensor=EE axes")
    ax[1].plot(frame_index, result["z"], "k", lw=0.9)
    ax[1].axhline(thresholds["contact_z"], color="tab:red", ls="--", label=f"contact z={thresholds['contact_z']}")
    ax[1].axhline(thresholds["absent_z"], color="tab:green", ls="--", label=f"absent z={thresholds['absent_z']}")
    ax[1].set_yscale("symlog", linthresh=10)
    ax[1].set_ylabel("noise-normalised |F|")
    ax[1].legend(fontsize=8)
    colors = {1: "tab:red", 0: "tab:green", -1: "tab:orange"}
    for a in ax[:2]:
        for code, color in colors.items():
            sel = state == code
            a.fill_between(frame_index, 0, 1, where=sel, transform=a.get_xaxis_transform(), color=color, alpha=0.12, step="mid")
    ax[2].plot(frame_index, speed, lw=0.8)
    ax[2].set_ylabel("EE speed (m/s)")
    ax[2].vlines(rgbd_frames, 0, 0.08, transform=ax[2].get_xaxis_transform(), color="gray", lw=0.5)
    ax[2].set_xlabel("native source frame_index (gray ticks: selected RGB-D frames)")
    fig.suptitle("contact (red) / absent (green) / uncertain (orange)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def export(scene, output, *, baseline_duration_s=DEFAULTS["baseline_duration_s"], baseline_frames=None,
           contact_z=DEFAULTS["contact_z"], absent_z=DEFAULTS["absent_z"],
           min_contact_duration_s=DEFAULTS["min_contact_duration_s"], tool_mass_kg=None,
           max_uncompensated_rotation_deg=DEFAULTS["max_uncompensated_rotation_deg"], corrections=None,
           plot=True):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    if output == scene or scene in output.parents:
        raise ValueError("Refusing to write contact annotations inside the source scene")
    conversion, parquet, info_path = locate_inputs(scene)
    info = json.loads(Path(info_path).read_text())
    tool_path = scene / "metadata/robot_tool.json"
    tool = json.loads(tool_path.read_text()) if tool_path.is_file() else None
    R_ee_sensor, wrench_frame = check_conventions(info, tool)
    ep = read_episode(parquet)
    order = np.argsort(ep["frame_index"])
    frame_index, t = ep["frame_index"][order], ep["timestamp"][order]
    if len(np.unique(frame_index)) != len(frame_index):
        raise ValueError("Duplicate Parquet frame_index values")
    force = ep["wrench"][order, :3]
    R_base_ee = rotvec_to_matrix(ep["state"][order, 3:6])
    R_base_sensor = R_base_ee @ R_ee_sensor
    base = baseline_mask_from(t, frame_index, duration_s=baseline_duration_s, frames=baseline_frames)
    rot_change = rotation_angle_deg(R_base_sensor, np.repeat(R_base_sensor[base][:1], len(t), axis=0))
    pos = ep["state"][order, :3]
    speed = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.maximum(np.diff(t), 1e-9)]
    gravity = None
    gravity_info = dict(max_rotation_from_baseline_deg=float(rot_change.max()))
    if tool_mass_kg is not None:
        gravity = gravity_residual(R_base_sensor, base, float(tool_mass_kg))
        gravity_info.update(applied=True, tool_mass_kg=float(tool_mass_kg),
                            max_correction_N=float(np.linalg.norm(gravity, axis=1).max()),
                            assumption="base_link z up; recorded force acts on the robot")
    else:
        if rot_change.max() > max_uncompensated_rotation_deg:
            raise ValueError(
                f"EE orientation changes by {rot_change.max():.2f} deg (> {max_uncompensated_rotation_deg}); "
                "tool gravity would bias the force. Supply --tool-mass-kg or raise the limit explicitly.")
        gravity_info.update(applied=False, reason="tool mass unknown; orientation change within limit",
                            max_uncompensated_rotation_deg=max_uncompensated_rotation_deg)
    result = detect_contact_states(t, force, base, contact_z=contact_z, absent_z=absent_z,
                                   min_contact_duration_s=min_contact_duration_s, gravity_correction=gravity)
    state, reason = result["state"], result["reason"]
    applied = []
    corrections_info = None
    if corrections is not None:
        corrections = Path(corrections)
        data = json.loads(corrections.read_text())
        state, reason, applied = apply_corrections(frame_index, state, reason, data)
        corrections_info = dict(path=str(corrections), sha256=sha256_file(corrections),
                                provenance=data["provenance"], applied=applied)
    if (state[base] == STATE_CODES["contact"]).any():
        raise ValueError("Baseline samples were labelled contact; the baseline is not contact-free")

    # RGB-D alignment: converted output frame -> native Parquet frame_index (exact join).
    lookup = {int(f): i for i, f in enumerate(frame_index)}
    rgbd = []
    for out_frame, src in zip(conversion["output_frames"], conversion["source_frames"]):
        k = lookup.get(int(src))
        rgbd.append(dict(rgbd_frame_index=int(out_frame), source_frame_index=int(src),
                         state=STATE_NAMES[int(state[k])] if k is not None else "uncertain",
                         timestamp_s=float(t[k]) if k is not None else None,
                         force_z=float(result["z"][k]) if k is not None else None,
                         reason=reason[k] if k is not None else "no_parquet_row"))
    names = [STATE_NAMES[int(s)] for s in state]
    thresholds = dict(contact_z=contact_z, absent_z=absent_z, min_contact_duration_s=min_contact_duration_s,
                      baseline_duration_s=None if baseline_frames else baseline_duration_s,
                      baseline_frames=list(baseline_frames) if baseline_frames else None,
                      noise_model="z = ||(F - tare)/sigma||, sigma = 1.4826*MAD per axis over baseline",
                      tare_N=result["tare_N"].tolist(), sigma_N=result["sigma_N"].tolist(),
                      contact_force_equivalent_N=float(contact_z * np.sqrt(np.mean(result["sigma_N"] ** 2))),
                      absent_force_equivalent_N=float(absent_z * np.sqrt(np.mean(result["sigma_N"] ** 2))))
    counts = {name: int((state == code).sum()) for name, code in STATE_CODES.items()}
    rgbd_counts = {name: sum(r["state"] == name for r in rgbd) for name in STATE_CODES}
    annotation = dict(
        index="source_frame_index", mode="per_frame",
        provenance=(f"automatic force-based contact detection ({Path(__file__).name}) on {parquet.name}; "
                    "operator-confirmed: robot contacts the moving part during interaction")
                   + ("; with manual corrections" if applied else ""),
        source_frame_index=frame_index.tolist(), contact_state=names,
        contact_active=[n == "contact" for n in names], force_z=[None if not np.isfinite(v) else float(v) for v in result["z"]],
        reason=reason, intervals=intervals_of(frame_index, state), episodes=[
            dict(e, first_source_frame=int(frame_index[e["first_index"]]), last_source_frame=int(frame_index[e["last_index"]]))
            for e in result["episodes"]],
        thresholds=thresholds, gravity_compensation=gravity_info, manual_corrections=corrections_info,
        tare="baseline mean removed (recorded data appear pre-tared; residual bias re-estimated)",
        wrench_frame=wrench_frame, counts_native=counts, counts_rgbd=rgbd_counts, rgbd_alignment=rgbd,
        sources=dict(scene=str(scene), parquet=str(parquet), parquet_sha256=sha256_file(parquet),
                     info=str(info_path), robot_tool=str(tool_path) if tool else None,
                     real_conversion_sha256=sha256_file(scene / "metadata/real_conversion.json")),
        notes=["Force decides when the tool interacts, not which part moves.",
               "Uncertain samples are neither contact nor absence; Phases B/C only use 'contact'.",
               "A sustained static load after the robot stops is reported as contact if it exceeds the "
               "threshold; check images before trusting it."])
    output.mkdir(parents=True, exist_ok=True)
    (output / "contact_annotations.json").write_text(json.dumps(annotation, indent=1) + "\n")
    with (output / "contact_annotations.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_frame_index", "timestamp_s", "contact_state", "contact_active", "force_z",
                         "Fx_corr_N", "Fy_corr_N", "Fz_corr_N", "ee_speed_m_s", "reason"])
        for k in range(len(t)):
            Fc = result["force_corrected"][k]
            writer.writerow([int(frame_index[k]), f"{t[k]:.6f}", names[k], int(state[k] == 1),
                             f"{result['z'][k]:.4f}", *(f"{v:.5f}" for v in Fc), f"{speed[k]:.5f}", reason[k]])
    if plot:
        plot_diagnostics(output / "force_contact.png", t, frame_index, result, state, thresholds,
                         [r["source_frame_index"] for r in rgbd], speed)
    summary = dict(status="available" if counts["contact"] else "no_confirmed_contact", output=str(output),
                   counts_native=counts, counts_rgbd=rgbd_counts, intervals=annotation["intervals"],
                   thresholds=thresholds, gravity_compensation=gravity_info)
    (output / "contact_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True, help="Converted real scene (read-only).")
    parser.add_argument("--output", type=Path, required=True, help="Output directory (outside the scene).")
    parser.add_argument("--baseline-duration-s", type=float, default=DEFAULTS["baseline_duration_s"])
    parser.add_argument("--baseline-frames", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="Explicit contact-free native frame range instead of the first seconds.")
    parser.add_argument("--contact-z", type=float, default=DEFAULTS["contact_z"])
    parser.add_argument("--absent-z", type=float, default=DEFAULTS["absent_z"])
    parser.add_argument("--min-contact-duration-s", type=float, default=DEFAULTS["min_contact_duration_s"])
    parser.add_argument("--tool-mass-kg", type=float, help="Enables gravity compensation for orientation changes.")
    parser.add_argument("--max-uncompensated-rotation-deg", type=float,
                        default=DEFAULTS["max_uncompensated_rotation_deg"])
    parser.add_argument("--corrections", type=Path, help="Optional manual corrections JSON (see docs).")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    summary = export(args.scene, args.output, baseline_duration_s=args.baseline_duration_s,
                     baseline_frames=args.baseline_frames, contact_z=args.contact_z, absent_z=args.absent_z,
                     min_contact_duration_s=args.min_contact_duration_s, tool_mass_kg=args.tool_mass_kg,
                     max_uncompensated_rotation_deg=args.max_uncompensated_rotation_deg,
                     corrections=args.corrections, plot=not args.no_plot)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
