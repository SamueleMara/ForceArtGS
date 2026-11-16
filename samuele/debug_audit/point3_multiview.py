#!/usr/bin/env python3
"""Point 3 -- common-world motion per camera, cross-view residuals, reference counterfactuals.

Convention (tests/test_fixed_camera_conventions.py): G_t maps frame-t camera points to frame 0,
so the camera-frame part motion 0 -> t is M_t = inv(G_t) G_0 and the world motion is
A_t = T_WC M_t T_WC^-1.  Axis comparisons are sign-invariant; the line distance uses the shared
world frame.  References are diagnostic only (black box: cam_000/cam_002 hinge fits; fridge:
rotation about the base vertical, door-plane range 17-24 deg from the earlier report).

Counterfactuals (no retraining): A legacy selection on the default pair; B forced reference per
camera; C quality-based selection (preprocessing/multiview_quality.py) over camera subsets;
D a diagnostic shared rigid fit per window pair on the union of cam_001+cam_002 frozen 3D tracks
in the world frame -- with (i) the pipeline's own moving-branch labels (no oracle) and (ii) oracle
labels (black box only); E leave-one-camera-out = the held-out depth tests of the quality report.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from samuele.debug_audit.trace import read_events  # noqa: E402
from samuele.debug_audit.point2_tracks import Oracle, seq_to_clip  # noqa: E402
from preprocessing.prepare_multiview import calibration_poses, motion_score  # noqa: E402


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def axis_of(R):
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(w)
    return w / n if n > 1e-9 else None


def about(R, a):
    from scipy.spatial.transform import Rotation as Ro
    q = Ro.from_matrix(R).as_quat()
    a = a / np.linalg.norm(a)
    return float(np.degrees(2 * np.arctan2(q[:3] @ a, q[3])))


def axis_point(A):
    """A point on the rotation axis of world motion A (least squares; None for tiny rotations)."""
    R, t = A[:3, :3], A[:3, 3]
    if rot_deg(R) < 1.0:
        return None
    return np.linalg.lstsq(np.eye(3) - R, t, rcond=None)[0]


def line_distance(p1, d1, p2, d2):
    n = np.cross(d1, d2)
    if np.linalg.norm(n) < 1e-6:
        return float(np.linalg.norm(np.cross(p2 - p1, d1)))
    return float(abs((p2 - p1) @ n) / np.linalg.norm(n))


def kabsch(P, Q):
    cp, cq = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cq - R @ cp
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--scene-kind", choices=("blackbox", "fridge"), required=True)
    ap.add_argument("--cameras", nargs="+", required=True)
    ap.add_argument("--default-pair", nargs=2, default=["cam_001", "cam_002"])
    ap.add_argument("--oracles", type=Path, required=True)
    args = ap.parse_args()
    out = args.run / "point3_shared_multiview"
    out.mkdir(parents=True, exist_ok=True)
    data = {c: args.run / "work/datasets" / f"baseline_{c}" for c in args.cameras}
    cal = calibration_poses(args.scene, args.cameras)
    conv = json.loads((data[args.cameras[0]] / "metadata/freeart_conversion.json").read_text())
    frames = [r["source_index"] for r in conv["frame_map"]]
    N = len(frames)
    sys.path.insert(0, str(args.oracles))
    if args.scene_kind == "blackbox":
        import common_bb as C
        AX = C.HINGE_D
        refs = {rc: np.array([C.HREF["cameras"][rc]["frames"][str(f)]["theta_deg"] for f in frames]) for rc in ("cam_000", "cam_002")}
        centroid = C.lid_points("cam_002", frames[0], zmin=0.15)[0].mean(0)
        ref_text = "hinge fits cam_000 / cam_002 (negative = opening)"
    else:
        import fcommon as C
        AX = np.array([0, 0, 1.0])
        refs = {}
        centroid = None
        ref_text = "door rotation about the vertical, plane-based range 17-24 deg at the clip end (no per-frame series)"
    per_cam, motions = {}, {}
    for c in args.cameras:
        G = [np.load(data[c] / f"global_poses_{p}.npy") for p in (0, 1)]
        valid = [np.load(data[c] / f"pose_valid_{p}.npy") for p in (0, 1)]
        scores = [float(motion_score(g)) for g in G]
        mp = int(np.argmax(scores))
        M = [np.linalg.inv(G[mp][t]) @ G[mp][0] for t in range(N)]
        A = [cal[c] @ m @ np.linalg.inv(cal[c]) for m in M]
        motions[c] = A
        ang = [about(a[:3, :3], AX) for a in A]
        steps = np.diff(ang)
        ref_cent = centroid if centroid is not None else np.mean([np.linalg.inv(cal[c])[:3, 3]], 0) * 0
        row = dict(moving_part=mp, motion_scores=scores, valid_fraction=float(valid[mp].mean()),
                   angle_about_reference_axis_deg=ang,
                   final_total_rotation_deg=rot_deg(A[-1][:3, :3]),
                   final_axis_angle_to_reference_axis_deg=None if axis_of(A[-1][:3, :3]) is None else
                   float(np.degrees(np.arccos(min(1, abs(axis_of(A[-1][:3, :3]) @ AX / np.linalg.norm(AX)))))),
                   largest_step_deg=float(np.max(np.abs(steps))), largest_step_frame=int(np.argmax(np.abs(steps)) + 1),
                   backward_steps=[(int(i + 1), float(s)) for i, s in enumerate(steps) if s > 0.5])
        if centroid is not None:
            row["centroid_translation_mm"] = [float(np.linalg.norm(a[:3, :3] @ centroid + a[:3, 3] - centroid) * 1e3) for a in A]
            for rc, r in refs.items():
                row[f"rms_vs_ref_{rc}_deg"] = float(np.sqrt(np.mean((np.array(ang) - r) ** 2)))
                row[f"final_err_vs_ref_{rc}_deg"] = float(ang[-1] - r[-1])
        per_cam[c] = row
    (out / "per_camera_motion_world.json").write_text(json.dumps(dict(reference=ref_text, frames=frames, cameras=per_cam), indent=1))
    rows = []
    for a, b in itertools.combinations(args.cameras, 2):
        for t in range(1, N):
            Ra, Rb = motions[a][t][:3, :3], motions[b][t][:3, :3]
            xa, xb = axis_of(Ra), axis_of(Rb)
            r = dict(pair=f"{a}|{b}", pair_label="default" if {a, b} == set(args.default_pair) else "involves_control", t=t,
                     source_frame=frames[t], relative_rotation_deg=rot_deg(Ra.T @ Rb),
                     axis_angle_deg=None if xa is None or xb is None or rot_deg(Ra) < 1 or rot_deg(Rb) < 1 else float(np.degrees(np.arccos(min(1, abs(xa @ xb))))))
            pa, pb = axis_point(motions[a][t]), axis_point(motions[b][t])
            r["axis_line_distance_mm"] = None if pa is None or pb is None else line_distance(pa, xa, pb, xb) * 1e3
            if centroid is not None:
                ca = motions[a][t][:3, :3] @ centroid + motions[a][t][:3, 3]
                cb = motions[b][t][:3, :3] @ centroid + motions[b][t][:3, 3]
                r["centroid_disagreement_mm"] = float(np.linalg.norm(ca - cb) * 1e3)
            rows.append(r)
    with open(out / "cross_view_motion_residuals.csv", "w", newline="") as h:
        keys = list(rows[0])
        w = csv.DictWriter(h, keys + [k for k in ("centroid_disagreement_mm",) if k not in keys])
        w.writeheader()
        w.writerows(rows)
    # ---- reference selection counterfactuals ----
    from preprocessing.multiview_quality import camera_quality, select_reference
    stems = [r["output_stem"] for r in conv["frame_map"]]
    cf = {}
    subsets = [tuple(args.default_pair)] + [s for k in (2, 3) for s in itertools.combinations(args.cameras, k)
                                            if set(s) != set(args.default_pair)]
    for sub in subsets:
        if len(sub) < 2:
            continue
        mp = {c: per_cam[c]["moving_part"] for c in sub}
        valid = {c: np.load(data[c] / f"pose_valid_{mp[c]}.npy") for c in sub}
        legacy = max(sub, key=lambda c: (bool(valid[c][0]), int(valid[c].sum()),
                                         abs(per_cam[c]["motion_scores"][1] - per_cam[c]["motion_scores"][0])))
        rep = camera_quality({c: data[c] for c in sub}, stems, cal, mp, valid)
        ref, status, reasons = select_reference(rep)
        cf["+".join(sub)] = dict(label="default" if set(sub) == set(args.default_pair) else "stress_or_subset",
                                 legacy_reference=legacy, legacy_reference_final_angle_deg=per_cam[legacy]["angle_about_reference_axis_deg"][-1],
                                 quality_reference=ref, quality_status=status, reasons=reasons,
                                 held_out={c: {d: dict(verdict=h["verdict"], frames=h["frames"], better=h["motion_better"],
                                                       floor_mm=None if h["frame0_floor_m"] is None else h["frame0_floor_m"] * 1e3)
                                               for d, h in rep[c]["held_out"].items()} for c in sub})
    cf["forced_reference"] = {c: dict(final_angle_about_reference_axis_deg=per_cam[c]["angle_about_reference_axis_deg"][-1],
                                      **{k: v for k, v in per_cam[c].items() if k.startswith("final_err_vs_ref") or k.startswith("rms_vs_ref")},
                                      note="the moving-part virtual cameras of every camera inherit exactly this world motion")
                              for c in args.cameras}
    # ---- D: shared rigid fit on frozen tracks of the default pair ----
    shared = []
    trk = {}
    for c in args.default_pair:
        tr = args.run / "trace" / c / "events.jsonl"
        if not tr.is_file():
            continue
        orc = Oracle(args.scene_kind, c, args.oracles)
        for e in read_events(tr):
            if e["event"] != "pair_solution" or e["seq_frame_b"] >= N:
                continue
            a = np.load(args.run / "trace" / c / e["artifact_refs"][0])
            xyz = a["xyz"].astype(float)
            Pw, Qw = (xyz[:, 0] @ cal[c][:3, :3].T + cal[c][:3, 3]), (xyz[:, 1] @ cal[c][:3, :3].T + cal[c][:3, 3])
            fa = frames[seq_to_clip(e["seq_frame_a"], N)]
            lab = orc.labels(Pw, fa)
            wf = a["final_weights"]
            mov_branch = "T2" if args.scene_kind else None
            moving_pipeline = wf > 0.5 if (e["value"]["labels_swapped"] is False) else wf > 0.5
            trk[(c, e["seq_frame_a"], e["seq_frame_b"])] = dict(P=Pw, Q=Qw, oracle=lab == ("lid" if args.scene_kind == "blackbox" else "door"),
                                                               pipeline=moving_pipeline, fa=fa,
                                                               fb=frames[seq_to_clip(e["seq_frame_b"], N)])
    keys = sorted({(sa, sb) for (_, sa, sb) in trk})
    for sa, sb in keys:
        parts = [trk.get((c, sa, sb)) for c in args.default_pair]
        if any(p is None for p in parts):
            continue
        fa, fb = parts[0]["fa"], parts[0]["fb"]
        r = dict(seq_a=sa, seq_b=sb, source_a=fa, source_b=fb)
        if refs:
            r["reference_deg"] = {rc: float(v[frames.index(fb)] - v[frames.index(fa)]) for rc, v in refs.items()}
        for sel in ("pipeline", "oracle"):
            if sel == "oracle" and args.scene_kind != "blackbox":
                continue
            for name, use in (("cam_001_only", [0]), ("cam_002_only", [1]), ("shared_001_002", [0, 1])):
                P = np.vstack([parts[i]["P"][parts[i][sel]] for i in use])
                Q = np.vstack([parts[i]["Q"][parts[i][sel]] for i in use])
                if len(P) < 6:
                    continue
                T = kabsch(P, Q)
                r[f"{sel}_{name}_deg"] = about(T[:3, :3], AX)
                r[f"{sel}_{name}_n"] = int(len(P))
        shared.append(r)
    (out / "reference_selection_counterfactuals.json").write_text(json.dumps(dict(selection=cf, shared_track_fit=shared), indent=1, default=float))
    print(json.dumps({c: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k in
                          ("moving_part", "final_total_rotation_deg", "final_axis_angle_to_reference_axis_deg", "largest_step_deg",
                           "rms_vs_ref_cam_000_deg", "rms_vs_ref_cam_002_deg", "valid_fraction")} | {"final_angle": round(r["angle_about_reference_axis_deg"][-1], 2)}
                      for c, r in per_cam.items()}, indent=0))
    print({k: (v.get("legacy_reference"), v.get("quality_reference"), v.get("quality_status")) for k, v in cf.items() if k != "forced_reference"})
    big = [r for r in shared if refs and abs(r["reference_deg"]["cam_002"]) >= 2] if refs else shared
    for key in sorted({k for r in big for k in r if k.endswith("_deg") and k != "reference_deg"}):
        if refs:
            ratio = [r[key] / r["reference_deg"]["cam_002"] for r in big if key in r]
            print(key, "median recovered fraction vs ref002", round(float(np.median(ratio)), 2), "pairs", len(ratio))
        else:
            vals = [r[key] for r in big if key in r]
            print(key, "median", round(float(np.median(vals)), 2))


if __name__ == "__main__":
    main()
