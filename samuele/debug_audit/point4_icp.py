#!/usr/bin/env python3
"""Point 4 -- ICP counterfactuals on fixed inputs (black box, geometry-oracle part masks).

Inputs: a dataset whose mask_1 are the DIAGNOSTIC geometry-oracle lid masks
(make_geometry_masks.py) and whose ICP was run with tracing (icp_frame events).  Evaluation only:
the tracker-free hinge fits of cam_000 and cam_002 (their spread = reference uncertainty).

Conventions verified in tests/test_fixed_camera_conventions.py: global_poses[t] maps frame-t camera
points to frame-0 camera points, so the camera-frame motion frame 0 -> t is M_t = inv(G_t) G_0.

Outputs (``--out``): icp_init_counterfactual.json (per frame: pipeline vs reference angle about the
hinge, lid-centroid displacement/lift error, axis error, validity, fallback, first failing edge,
direct/sequential/role-swap/init variants, objective variants), icp_cost_landscape.npz (hinge angle
x vertical lift grid: own truncated hybrid objective and held-out depth residual in other cameras),
icp_correspondence_audit.npz (overlap / correspondences at reference vs pipeline), pose_graph_explainer.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from samuele.debug_audit.trace import read_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--heldout", nargs="+", default=["cam_001", "cam_000"])
    ap.add_argument("--oracles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--landscape-frames", type=int, nargs="+", default=[10, 15, 18, 20, 25, 29])
    args = ap.parse_args()
    sys.path.insert(0, str(args.oracles))
    import common_bb as C
    import icp.color_icp_gpu as icpm
    from utils.pipeline_config import load_pipeline_config, resolve_icp
    cfg = resolve_icp(load_pipeline_config(str(args.config)))
    icpm.ICP_T_DEVICE, icpm.TORCH_DEVICE = icpm.resolve_gpu_devices("cuda")
    icpm.HYBRID_PLANE_WEIGHT, icpm.HYBRID_POINT_WEIGHT = float(cfg["hybrid_plane_weight"]), float(cfg["hybrid_point_weight"])
    icpm.HYBRID_DAMPING = max(0.0, float(cfg["hybrid_damping"]))
    wp = icpm.HYBRID_POINT_WEIGHT
    cam = args.camera
    conv = json.loads((args.dataset / "metadata/freeart_conversion.json").read_text())
    stems = [r["output_stem"] for r in conv["frame_map"]]
    frames = [int(r["source_index"]) for r in conv["frame_map"]]
    N = len(stems)
    clouds = [o3d.io.read_point_cloud(str(args.dataset / "point_clouds_1" / f"{s}.ply")) for s in stems]
    vox = float(icpm.compute_adaptive_voxel_size(clouds))
    B, Binv, K = C.POSE[cam], np.linalg.inv(C.POSE[cam]), C.KK[cam]
    refs = {rc: C.HREF["cameras"][rc] for rc in ("cam_000", "cam_002")}
    primary = cam if cam in refs else "cam_002"

    def hinge_cam(theta, rc):
        piv = np.array(refs[rc]["pivot"])
        R = C.rot_about(C.HINGE_D, np.radians(theta))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = piv - R @ piv
        return Binv @ T @ B

    ref_deg = {rc: np.array([refs[rc]["frames"][str(f)]["theta_deg"] for f in frames]) for rc in refs}
    M_ref = [hinge_cam(th, primary) for th in ref_deg[primary]]
    G = np.load(args.dataset / "global_poses_1.npy")
    valid = np.load(args.dataset / "pose_valid_1.npy")
    M_pipe = [np.linalg.inv(G[t]) @ G[0] for t in range(N)]
    P0 = np.asarray(clouds[0].points)
    c0 = P0.mean(0)
    up_cam = Binv[:3, :3] @ np.array([0, 0, 1.0])

    def hinge_of(M):
        return C.about_hinge_deg((B @ M @ Binv)[:3, :3])

    def axis_err(M):
        R = (B @ M @ Binv)[:3, :3]
        ax = C.axis_of(R)
        return None if ax is None or C.rot_deg(R) < 0.5 else float(np.degrees(np.arccos(min(1, abs(ax @ C.HINGE_D)))))

    def centroid_err(M, Mr):
        a, b = M[:3, :3] @ c0 + M[:3, 3], Mr[:3, :3] @ c0 + Mr[:3, 3]
        return float(np.linalg.norm(a - b) * 1e3), float((a - b) @ up_cam * 1e3)

    def disp(M, Mr):
        d = np.linalg.norm(P0 @ M[:3, :3].T + M[:3, 3] - (P0 @ Mr[:3, :3].T + Mr[:3, 3]), axis=1)
        return float(np.median(d) * 1e3), float(np.percentile(d, 90) * 1e3)

    normals = {}

    def target(t):
        if t not in normals:
            q = o3d.geometry.PointCloud(clouds[t])
            q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * vox, max_nn=100))
            normals[t] = (np.asarray(q.points), np.asarray(q.normals), cKDTree(np.asarray(q.points)))
        return normals[t]

    def objective(M, t, max_corr):
        Q, Nq, tree = target(t)
        X = P0 @ M[:3, :3].T + M[:3, 3]
        d, j = tree.query(X)
        inl = d <= max_corr
        diff = X - Q[j]
        r2 = np.where(inl, np.sum(diff * Nq[j], 1) ** 2 + wp * np.sum(diff ** 2, 1), max_corr ** 2 * (1 + wp))
        return float(np.mean(r2)), float(inl.mean())

    def run(src, tgt, init):
        r = icpm.refine_registration_tensor(clouds[src], clouds[tgt], vox, init=init)
        return np.asarray(r.transformation), r.fitness

    def variant(src, tgt, init, kind):
        s, t_ = icpm._ensure_tensor_pcd(clouds[src]), icpm._ensure_tensor_pcd(clouds[tgt])
        s.estimate_normals(max_nn=100, radius=2 * vox)
        t_.estimate_normals(max_nn=100, radius=2 * vox)
        est = {"point_to_point": o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
               "point_to_plane": o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
               "colored": o3d.t.pipelines.registration.TransformationEstimationForColoredICP()}[kind]
        crit = o3d.t.pipelines.registration.ICPConvergenceCriteria(1e-6, 1e-6, 100)
        res = o3d.t.pipelines.registration.icp(s, t_, 3 * vox, o3d.core.Tensor(init, o3d.core.Dtype.Float64, icpm.ICP_T_DEVICE), est, crit)
        return np.asarray(res.transformation.cpu().numpy())

    ev = [e for e in read_events(args.trace / "events.jsonl") if e["event"] == "icp_frame" and e.get("part_id") == 1]
    ev = {e["value"]["frame_index"]: e["value"] for e in ev}
    rows = []
    seq = np.eye(4)
    first_bad = None
    for t in range(N):
        spread = abs(ref_deg["cam_000"][t] - ref_deg["cam_002"][t])
        row = dict(t=t, source_frame=frames[t], valid=bool(valid[t]),
                   reference_deg={rc: float(ref_deg[rc][t]) for rc in refs}, reference_spread_deg=float(spread),
                   pipeline_deg=hinge_of(M_pipe[t]), pipeline_axis_err_deg=axis_err(M_pipe[t]),
                   pipeline_lid_disp_mm=disp(M_pipe[t], M_ref[t]), lid_points=len(clouds[t].points))
        row["pipeline_centroid_err_mm"], row["pipeline_centroid_lift_err_mm"] = centroid_err(M_pipe[t], M_ref[t])
        if t in ev:
            e = ev[t]
            row.update(icp_fallback=e["fallback"], icp_attempts=[dict(retry=a["retry"], init=a["init_kind"], fitness=round(a["fitness"], 4),
                                                                      init_fitness=round(a["init_fitness"], 4), plausible=a["plausible"],
                                                                      not_worse=a["fit_not_worse_than_init"]) for a in (e["attempts"] or [])],
                       icp_ref_idx=e["ref_idx"], icp_success=e["success"])
        err = row["pipeline_deg"] - ref_deg[primary][t]
        if first_bad is None and t > 0 and abs(err) > spread + 1e-9 and abs(err) > 1.0:
            first_bad = t
        if t > 0:
            for name, M in (("reference", M_ref[t]), ("pipeline", M_pipe[t]), ("identity", np.eye(4))):
                row[f"J_1vox_{name}"], row[f"overlap_1vox_{name}"] = objective(M, t, vox)
            inits = {"identity": np.eye(4), "reference": M_ref[t],
                     "pipeline_constant_velocity": (np.linalg.inv(np.array(ev[t]["attempts"][0]["init_pose"])) if t in ev and ev[t]["attempts"] else None)}
            for name, init in inits.items():
                if init is None:
                    continue
                M, fit = run(0, t, init)
                row[f"direct_from_{name}"] = dict(deg=hinge_of(M), disp_mm=disp(M, M_ref[t]), fitness=fit, J_1vox=objective(M, t, vox)[0],
                                                  lift_err_mm=centroid_err(M, M_ref[t])[1])
            # role swap: register t -> 0 then invert
            Ms, _ = run(t, 0, np.eye(4))
            row["role_swap_inverted_deg"] = hinge_of(np.linalg.inv(Ms))
            # sequential chain from identity
            Mq, _ = run(t - 1, t, np.eye(4))
            seq = Mq @ seq
            row["sequential_deg"] = hinge_of(seq)
            for kind in ("point_to_point", "point_to_plane", "colored"):
                M = variant(0, t, np.eye(4), kind)
                row[f"objective_{kind}_deg"] = hinge_of(M)
                row[f"objective_{kind}_lift_err_mm"] = centroid_err(M, M_ref[t])[1]
        rows.append(row)
    # ---- cost landscape: own objective vs held-out depth (other cameras), angle x vertical lift ----
    thetas = np.arange(-35, 6, 1.0)
    lifts = np.arange(-0.06, 0.0601, 0.005)
    land = {}
    held = {h: (C.POSE[h], C.KK[h]) for h in args.heldout if h != cam}
    for t in args.landscape_frames:
        if t >= N:
            continue
        f = frames[t]
        J = np.zeros((len(thetas), len(lifts)))
        H = {h: np.full((len(thetas), len(lifts)), np.nan) for h in held}
        Hcov = {h: np.zeros((len(thetas), len(lifts))) for h in held}
        dmaps = {h: (C.depth(h, f), C.mask(h, f)) for h in held}
        P0w = P0 @ B[:3, :3].T + B[:3, 3]
        piv = np.array(refs[primary]["pivot"])
        for i, th in enumerate(thetas):
            R = C.rot_about(C.HINGE_D, np.radians(th))
            Xw0 = (P0w - piv) @ R.T + piv
            for j, lf in enumerate(lifts):
                Xw = Xw0 + np.array([0, 0, lf])
                Xc = Xw @ Binv[:3, :3].T + Binv[:3, 3]
                Q, Nq, tree = target(t)
                d, jj = tree.query(Xc)
                inl = d <= vox
                diff = Xc - Q[jj]
                J[i, j] = np.mean(np.where(inl, np.sum(diff * Nq[jj], 1) ** 2 + wp * np.sum(diff ** 2, 1), vox ** 2 * (1 + wp)))
                for h, (Th, Kh) in held.items():
                    uv, z = C.project(h, Xw)
                    D, Mk = dmaps[h]
                    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
                    ok = (u >= 0) & (u < D.shape[1]) & (v >= 0) & (v < D.shape[0]) & (z > 0)
                    ok[ok] &= Mk[v[ok], u[ok]] & (D[v[ok], u[ok]] > 0)
                    Hcov[h][i, j] = ok.mean()
                    if ok.sum() >= 30:
                        H[h][i, j] = float(np.median(np.abs(D[v[ok], u[ok]] - z[ok])))
        land[t] = dict(J=J, held=H, coverage=Hcov)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "icp_cost_landscape.npz", thetas=thetas, lifts=lifts, frames=np.array(sorted(land)),
                        **{f"J_t{t}": v["J"] for t, v in land.items()},
                        **{f"heldout_{h}_t{t}": v["held"][h] for t, v in land.items() for h in v["held"]},
                        **{f"heldout_cov_{h}_t{t}": v["coverage"][h] for t, v in land.items() for h in v["coverage"]})
    land_summary = {}
    for t, v in land.items():
        i, j = np.unravel_index(np.argmin(v["J"]), v["J"].shape)
        s = dict(reference_deg=float(ref_deg[primary][t]), own_objective_argmin=dict(theta=float(thetas[i]), lift_mm=float(lifts[j] * 1e3)))
        for h, Hm in v["held"].items():
            if np.isfinite(Hm).any():
                a, b = np.unravel_index(np.nanargmin(Hm), Hm.shape)
                s[f"heldout_{h}_argmin"] = dict(theta=float(thetas[a]), lift_mm=float(lifts[b] * 1e3), residual_mm=float(Hm[a, b] * 1e3),
                                                coverage=float(v["coverage"][h][a, b]))
                jr = int(np.argmin(np.abs(thetas - ref_deg[primary][t])))
                s[f"heldout_{h}_at_reference_mm"] = float(Hm[jr, len(lifts) // 2] * 1e3) if np.isfinite(Hm[jr, len(lifts) // 2]) else None
                ip = int(np.argmin(np.abs(thetas - rows[t]["pipeline_deg"])))
                s[f"heldout_{h}_at_pipeline_angle_zero_lift_mm"] = float(Hm[ip, len(lifts) // 2] * 1e3) if np.isfinite(Hm[ip, len(lifts) // 2]) else None
        land_summary[str(t)] = s
    # ---- correspondence audit at reference vs pipeline ----
    corr = {}
    for t in (5, 10, 15, 20, 25, N - 1):
        Q, Nq, tree = target(t)
        for name, M in (("reference", M_ref[t]), ("pipeline", M_pipe[t])):
            X = P0 @ M[:3, :3].T + M[:3, 3]
            d, j = tree.query(X)
            corr[f"t{t}_{name}_nn_dist"] = d.astype(np.float32)
            corr[f"t{t}_{name}_normal_dot"] = np.abs(np.sum((X - Q[j]) / np.maximum(np.linalg.norm(X - Q[j], axis=1, keepdims=True), 1e-9) * Nq[j], 1)).astype(np.float32)
    np.savez_compressed(args.out / "icp_correspondence_audit.npz", voxel=vox, **corr)
    out = dict(camera=cam, voxel=vox, primary_reference=primary, hybrid_point_weight=wp, first_failing_frame=first_bad,
               first_failing_definition="first t>0 with |pipeline - reference| > max(1 deg, |ref_cam000 - ref_cam002|)",
               frames=rows, landscape=land_summary)
    (args.out / "icp_init_counterfactual.json").write_text(json.dumps(out, indent=1, default=float))
    for r in rows[::3] + [rows[-1]]:
        print(r["t"], r["source_frame"], "ref", round(r["reference_deg"][primary], 1), "pipe", round(r["pipeline_deg"], 1),
              "lift_err", round(r["pipeline_centroid_lift_err_mm"], 1), "fallback", r.get("icp_fallback"),
              {k: round(r[k]["deg"], 1) for k in r if k.startswith("direct_from")},
              {k.replace("objective_", "")[:14]: round(r[k], 1) for k in r if k.startswith("objective_") and k.endswith("_deg")})
    print("first failing frame", first_bad)
    print(json.dumps(land_summary, indent=0)[:3000])


if __name__ == "__main__":
    main()
