#!/usr/bin/env python3
"""Point 2 -- non-rigidity of the raw 4x4 fits, existence of a rigid solution, and label lineage.

From one camera's audit trace (solver_trace / pair_solution / window_merged events):

* transform_nonrigidity.json: ‖RᵀR−I‖, det R, singular values / max stretch, homogeneous-row
  error of T1/T2 at init, before/after every re-initialisation, during the joint stage and final;
  weighted and unweighted loss terms at the same snapshots; displacement on the *actual lid*
  (oracle-labelled frame-a points) between the raw matrix and its nearest rotation (polar);
* rigid-existence check (diagnostic weighted Kabsch on the frozen tracks of each soft part, and
  on oracle-lid tracks): residuals of the raw nonrigid fit vs the best rigid fit on the same
  points and weights, plus the rigid fit's rotation about the reference hinge;
* segmentation_window_lineage.json: per pair, the fraction of oracle-lid / oracle-base tracks
  assigned to soft part 2 (w > 0.5) before and after ``check_segmentation_consistency``, whether
  labels were swapped, and which branch carries the physical moving part.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from samuele.debug_audit.trace import matrix_rigidity, read_events  # noqa: E402
from samuele.debug_audit.point2_tracks import Oracle, seq_to_clip  # noqa: E402


def polar(T):
    U, S, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = T[:3, 3]
    return out


def wkabsch(P, Q, w):
    w = w / w.sum()
    cp, cq = w @ P, w @ Q
    H = ((P - cp) * w[:, None]).T @ (Q - cq)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cq - R @ cp
    return T


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def about_axis_deg(R, axis):
    from scipy.spatial.transform import Rotation as Ro
    q = Ro.from_matrix(R).as_quat()
    a = axis / np.linalg.norm(axis)
    return float(np.degrees(2 * np.arctan2(q[:3] @ a, q[3])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--scene-kind", choices=("blackbox", "fridge"), required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--oracles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ev = read_events(args.trace / "events.jsonl")
    conv = json.loads((args.dataset / "metadata/freeart_conversion.json").read_text())
    n_clip = len(conv["frame_map"])
    src = [r["source_index"] for r in conv["frame_map"]]
    oracle = Oracle(args.scene_kind, args.camera, args.oracles)
    B = oracle.T
    solver = {(e["window_id"], e["pair_k"]): e for e in ev if e["event"] == "solver_trace"}
    nonrig, lineage = [], []
    for e in (x for x in ev if x["event"] == "pair_solution"):
        w, k = e["window_id"], e["pair_k"]
        ca, cb = seq_to_clip(e["seq_frame_a"], n_clip), seq_to_clip(e["seq_frame_b"], n_clip)
        fa, fb = src[ca], src[cb]
        it = 0 if e["seq_frame_b"] < n_clip else 1
        a = np.load(args.trace / e["artifact_refs"][0])
        xyz = a["xyz"].astype(float)
        P, Q = xyz[:, 0], xyz[:, 1]
        Pw = P @ B[:3, :3].T + B[:3, 3]
        lab = oracle.labels(Pw, fa)
        mov = "lid" if args.scene_kind == "blackbox" else "door"
        stat = "base" if args.scene_kind == "blackbox" else "other"
        v = e["value"]
        row = dict(window=w, pair_k=k, pingpong_iteration=it, source_a=fa, source_b=fb, n=int(len(P)),
                   n_moving_oracle=int((lab == mov).sum()), n_static_oracle=int((lab == stat).sum()),
                   labels_swapped=v["labels_swapped"])
        for tag, wts in (("solved", a["solved_weights"]), ("final", a["final_weights"])):
            for L in (mov, stat):
                s = lab == L
                row[f"{tag}_{L}_frac_part2"] = float((wts[s] > 0.5).mean()) if s.any() else None
        # physical moving branch: the branch holding the majority of oracle-moving tracks
        wf = a["final_weights"]
        s_m = lab == mov
        row["moving_branch"] = None if not s_m.any() else ("T2" if (wf[s_m] > 0.5).mean() >= 0.5 else "T1")
        lineage.append(row)
        # non-rigidity and rigid existence for the physical moving branch
        nr = dict(window=w, pair_k=k, pingpong_iteration=it, source_a=fa, source_b=fb)
        for name in ("T1", "T2"):
            T = np.array(v[name])
            nr[f"{name}_rigidity"] = matrix_rigidity(T)
            if s_m.any():
                L = P[s_m]
                nr[f"{name}_lid_displacement_raw_vs_polar_mm"] = dict(
                    median=float(np.median(np.linalg.norm(apply(T, L) - apply(polar(T), L), axis=1)) * 1e3),
                    max=float(np.max(np.linalg.norm(apply(T, L) - apply(polar(T), L), axis=1)) * 1e3))
        if s_m.sum() >= 6 and row["moving_branch"]:
            Tm = np.array(v[row["moving_branch"]])
            wpart = wf if row["moving_branch"] == "T2" else 1 - wf
            Tk = wkabsch(P, Q, wpart + 1e-9)
            Tlid = wkabsch(P[s_m], Q[s_m], np.ones(s_m.sum()))
            res = lambda T, S: np.linalg.norm(apply(T, P[S]) - Q[S], axis=1)
            nr["moving_branch"] = row["moving_branch"]
            nr["residual_on_oracle_moving_mm"] = dict(raw_fit=float(np.median(res(Tm, s_m)) * 1e3),
                                                      raw_fit_polar=float(np.median(res(polar(Tm), s_m)) * 1e3),
                                                      rigid_weighted_kabsch=float(np.median(res(Tk, s_m)) * 1e3),
                                                      rigid_kabsch_on_oracle_lid=float(np.median(res(Tlid, s_m)) * 1e3))
            if args.scene_kind == "blackbox":
                ax = oracle.C.HINGE_D
                to_b = lambda T: B @ T @ np.linalg.inv(B)
                ref = oracle.refs[oracle.primary]["frames"]
                nr["hinge_angle_deg"] = dict(reference=ref[str(fb)]["theta_deg"] - ref[str(fa)]["theta_deg"],
                                             raw_fit_polar=about_axis_deg(to_b(polar(Tm))[:3, :3], ax),
                                             rigid_weighted_kabsch=about_axis_deg(to_b(Tk)[:3, :3], ax),
                                             rigid_kabsch_on_oracle_lid=about_axis_deg(to_b(Tlid)[:3, :3], ax))
        st = solver.get((w, k))
        if st is not None:
            snaps = st["value"]["snapshots"]
            nr["solver_snapshots"] = [dict(tag=s["tag"], epoch=s["epoch"],
                                           T1_max_stretch=s["T1_rigidity"]["max_stretch"], T2_max_stretch=s["T2_rigidity"]["max_stretch"],
                                           T1_orth_err=s["T1_rigidity"]["orth_err_fro"], T2_orth_err=s["T2_rigidity"]["orth_err_fro"],
                                           T1_det=s["T1_rigidity"]["det"], T2_det=s["T2_rigidity"]["det"],
                                           loss_weighted=s.get("loss_weighted"), loss_unweighted=s.get("loss_unweighted"),
                                           weight_frac_part2=s.get("weight_frac_part2")) for s in snaps]
            reinit = [(s["epoch"], s["T2_rigidity"]["translation"]) for s in snaps if s["tag"] in ("before_reinit", "after_reinit")]
            nr["reinit_changed_T2"] = bool(any(np.abs(np.subtract(reinit[i][1], reinit[i + 1][1])).max() > 0 for i in range(0, len(reinit) - 1, 2)))
        nonrig.append(nr)
    merged = [dict(window=e["value"]["window_id"], frames=e["value"]["frame_indices"],
                   T1_max_stretch=[r["max_stretch"] for r in e["value"]["T1_rigidity"]],
                   T2_max_stretch=[r["max_stretch"] for r in e["value"]["T2_rigidity"]])
              for e in ev if e["event"] == "window_merged"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "transform_nonrigidity.json").write_text(json.dumps(dict(pairs=nonrig, merged_windows=merged), indent=1))
    (args.out / "segmentation_window_lineage.json").write_text(json.dumps(lineage, indent=1))
    fw = [r for r in nonrig if r["pingpong_iteration"] == 0]
    def m(key, sub):
        x = [r[key][sub] for r in fw if key in r]
        return (float(np.median(x)), float(np.max(x))) if x else None
    print("pairs", len(nonrig), "forward", len(fw))
    print("max_stretch T1/T2 (median,max):", m("T1_rigidity", "max_stretch"), m("T2_rigidity", "max_stretch"))
    for key in ("raw_fit", "raw_fit_polar", "rigid_weighted_kabsch", "rigid_kabsch_on_oracle_lid"):
        x = [r["residual_on_oracle_moving_mm"][key] for r in fw if "residual_on_oracle_moving_mm" in r]
        print(" residual", key, round(float(np.median(x)), 2) if x else None)
    if args.scene_kind == "blackbox":
        big = [r for r in fw if "hinge_angle_deg" in r]
        big.sort(key=lambda r: r["hinge_angle_deg"]["reference"])
        for r in big[:6]:
            print(" hinge", r["window"], r["pair_k"], {k: round(v, 2) for k, v in r["hinge_angle_deg"].items()})
    print("reinit changed T2 in any pair:", any(r.get("reinit_changed_T2") for r in nonrig))
    lw = [r for r in lineage if r["pingpong_iteration"] == 0]
    print("moving branch counts", {b: sum(r["moving_branch"] == b for r in lw) for b in ("T1", "T2", None)},
          "swapped", sum(r["labels_swapped"] for r in lw))


if __name__ == "__main__":
    main()
