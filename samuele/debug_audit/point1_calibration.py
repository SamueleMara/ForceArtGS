#!/usr/bin/env python3
"""Point 1 -- fixed-camera calibration and cross-camera static consistency (no refit, read only).

Outputs (``--out``): calibration_audit.json, static_cross_camera_residuals.csv,
feature_reprojection.csv, frame_map.csv, figures/reproj_<a>_<b>_<frame>.jpg.

* calibration table per camera: K (cameras.json, COLMAP cameras.txt), image geometry, distortion
  status, T_WC from cameras.json, COLMAP images.txt (sparse and *_final models) and from the
  source hand-eye calibration (G_left · T_base_color_optical); SE(3) residuals between sources and
  whether any physical pose/K changes across the listed images;
* synchronized-instant cross-camera consistency, *same instant* so object motion cannot enter:
  dense depth (cam a back-projected, transformed with the fixed calibration, compared with cam b's
  depth at the projected pixel; signed residual, z_b observed - z predicted) stratified into
  object/background x interior/edge x dark/non-dark; and RGB features (SIFT, Lowe ratio 0.75,
  mutual best) with symmetric epipolar distance from the calibration and depth-based reprojection
  error -- common visible features only.

Strata are evaluation definitions, declared in the output: "edge" = 3x3-dilated pixels whose
largest 8-neighbour depth jump exceeds 3x the image's median absolute neighbour jump; "dark" =
max RGB channel < 64.  Nothing here changes calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else "MISSING"


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap(model):
    cams, imgs = {}, {}
    for line in (model / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            p = line.split()
            cams[int(p[0])] = dict(model=p[1], w=int(p[2]), h=int(p[3]), params=[float(x) for x in p[4:]])
    lines = [l for l in (model / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    for line in lines[0::2]:
        p = line.split()
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rot([float(x) for x in p[1:5]])
        w2c[:3, 3] = [float(x) for x in p[5:8]]
        name = p[9]
        imgs.setdefault(Path(name).parts[0], []).append(dict(name=name, camera_id=int(p[8]), c2w=np.linalg.inv(w2c)))
    return cams, imgs


def se3_diff(A, B):
    D = np.linalg.inv(A) @ B
    ang = np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
    return dict(rotation_deg=float(ang), translation_mm=float(np.linalg.norm(D[:3, 3]) * 1e3),
                camera_centre_offset_mm=float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1e3))


def calibration_table(scene):
    meta = json.loads((scene / "metadata/cameras.json").read_text())
    src = json.loads((scene / "metadata/camera_calibration_robot_base.json").read_text())
    G = np.array(meta["robot_base_registration"]["G_left_multiply_camera_to_base"])
    models = {name: read_colmap(scene / name / "sparse/0") for name in ("colmap", "colmap_final")
              if (scene / name / "sparse/0/images.txt").is_file()}
    table, _source_T = {}, {}
    for entry in meta["cameras"]:
        cam = entry["name"]
        K = np.array([[entry["fx"], 0, entry["cx"]], [0, entry["fy"], entry["cy"]], [0, 0, 1.0]])
        T_meta = np.array(entry["pose_matrix_4x4"])
        row = dict(width=entry["width"], height=entry["height"], K_cameras_json=K.tolist(),
                   pose_convention=entry.get("pose_convention"), coordinate_frame=entry.get("coordinate_frame"),
                   T_WC_cameras_json=T_meta.tolist(), sources={})
        native = src["cameras"].get(entry["source_camera_name"], {})
        if native:
            Kc = np.array(native["color"]["K"]).reshape(3, 3)
            row["native_color"] = dict(width=native["color"]["width"], height=native["color"]["height"],
                                       distortion_model=native["color"]["distortion_model"], D=native["color"]["D"],
                                       K=Kc.tolist(), K_scaled_to_dataset=(Kc * np.array([[entry["width"] / native["color"]["width"]] * 3,
                                                                                            [entry["height"] / native["color"]["height"]] * 3,
                                                                                            [1, 1, 1]])).tolist())
            T_src = G @ np.array(native["T_base_color_optical"])
            row["sources"]["G_times_T_base_color_optical"] = se3_diff(T_meta, T_src)
            _source_T[cam] = T_src
        for name, (cams, imgs) in models.items():
            ims = imgs.get(cam, [])
            if not ims:
                row["sources"][name] = "MISSING"
                continue
            cid = ims[0]["camera_id"]
            p = cams[cid]["params"]
            Kcol = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]]) if cams[cid]["model"] == "PINHOLE" else None
            row["sources"][name] = dict(images=[i["name"] for i in ims], camera_model=cams[cid]["model"],
                                        K_equal_cameras_json=None if Kcol is None else bool(np.allclose(Kcol, K, atol=1e-9)),
                                        T_WC_vs_cameras_json=se3_diff(T_meta, ims[0]["c2w"]),
                                        max_pose_change_across_images=max(se3_diff(ims[0]["c2w"], i["c2w"])["translation_mm"] for i in ims))
        table[cam] = row
    cams = sorted(table)
    pair_geometry = {}
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            Ta, Tb = np.array(table[a]["T_WC_cameras_json"]), np.array(table[b]["T_WC_cameras_json"])
            d = se3_diff(Ta, Tb)
            pair_geometry[f"{a}|{b}"] = dict(baseline_mm=d["camera_centre_offset_mm"], relative_rotation_deg=d["rotation_deg"])
    shared_K = len({json.dumps(table[c]["K_cameras_json"]) for c in cams}) == 1
    return dict(cameras=table, pairs=pair_geometry, shared_K_all_cameras=shared_K, _source_T=_source_T,
                calibration_status=meta.get("calibration_status"), source_calibration_status=meta.get("source_calibration_status"),
                source_warning=src.get("warning"), extrinsic_convention=src.get("extrinsic_convention"),
                rectification=("dataset README: RGB, registered depth and COLMAP share one centered, undistorted PINHOLE K; "
                               "the native per-camera K and plumb_bob distortion differ (see native_color). The remap itself is "
                               "not re-executed here: UNVERIFIED beyond the cross-camera residuals below."),
                file_sha256={name: sha(scene / rel) for name, rel in [("cameras.json", "metadata/cameras.json"),
                                                                      ("camera_calibration_robot_base.json", "metadata/camera_calibration_robot_base.json"),
                                                                      ("colmap_images.txt", "colmap/sparse/0/images.txt"),
                                                                      ("colmap_cameras.txt", "colmap/sparse/0/cameras.txt"),
                                                                      ("colmap_final_images.txt", "colmap_final/sparse/0/images.txt")]})


def relative_pose_consistency(cal):
    """Relative camera poses from cameras.json vs from the source hand-eye calibration (common-mode check)."""
    import itertools
    out = {}
    src = cal.get("_source_T", {})
    for a, b in itertools.combinations(sorted(cal["cameras"]), 2):
        if a not in src or b not in src:
            out[f"{a}|{b}"] = "MISSING"
            continue
        Ta, Tb = np.array(cal["cameras"][a]["T_WC_cameras_json"]), np.array(cal["cameras"][b]["T_WC_cameras_json"])
        Sa, Sb = src[a], src[b]
        out[f"{a}|{b}"] = se3_diff(np.linalg.inv(Ta) @ Tb, np.linalg.inv(Sa) @ Sb)
    return out


def dominant_plane(points, rng, trials=400):
    """RANSAC plane with an inlier band equal to 2x the median NN spacing-free depth noise proxy (declared)."""
    best, best_n = None, -1
    n = len(points)
    band = 0.01   # 1 cm declared evaluation band (not a pipeline parameter)
    for _ in range(trials):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(nrm) < 1e-9:
            continue
        nrm /= np.linalg.norm(nrm)
        d = np.abs((points - p0) @ nrm)
        c = int((d < band).sum())
        if c > best_n:
            best_n, best = c, (nrm, p0)
    nrm, p0 = best
    inl = np.abs((points - p0) @ nrm) < band
    P = points[inl]
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    nrm = Vt[-1]
    if nrm[2] < 0:
        nrm = -nrm
    return nrm, float(nrm @ c), int(inl.sum()), P


def plane_consistency(scene, cal, frames):
    """Fit the dominant background plane per camera in the world frame; compare across cameras."""
    rng = np.random.default_rng(0)
    cams = sorted(cal["cameras"])
    K = np.array(cal["cameras"][cams[0]]["K_cameras_json"])
    result = {}
    for frame in frames[:1]:
        planes = {}
        for c in cams:
            rgb, depth, mask = load_frame(scene, c, frame)
            v, u = np.mgrid[0:depth.shape[0]:3, 0:depth.shape[1]:3]
            z = depth[v, u]
            ok = (z > 0) & (~mask[v, u] if mask is not None else True)
            X = np.stack([(u[ok] - K[0, 2]) * z[ok] / K[0, 0], (v[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
            T = np.array(cal["cameras"][c]["T_WC_cameras_json"])
            W = X @ T[:3, :3].T + T[:3, 3]
            nrm, off, cnt, P = dominant_plane(W, rng)
            planes[c] = dict(normal_world=nrm.tolist(), offset_m=off, inliers=cnt, points=int(len(W)),
                             centroid_world=P.mean(0).tolist())
        # Same physical surface in every camera: points within a 5 cm band (declared) of the reference
        # camera's dominant plane, refitted per camera; if a camera does not see it, say so.
        ref = "cam_001" if "cam_001" in planes else cams[0]
        rn, ro = np.array(planes[ref]["normal_world"]), planes[ref]["offset_m"]
        for c in cams:
            rgb, depth, mask = load_frame(scene, c, frame)
            v, u = np.mgrid[0:depth.shape[0]:3, 0:depth.shape[1]:3]
            z = depth[v, u]
            ok = (z > 0) & (~mask[v, u] if mask is not None else True)
            X = np.stack([(u[ok] - K[0, 2]) * z[ok] / K[0, 0], (v[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
            T = np.array(cal["cameras"][c]["T_WC_cameras_json"])
            W = X @ T[:3, :3].T + T[:3, 3]
            band = W[np.abs(W @ rn - ro) < 0.05]
            if len(band) < 200:
                planes[c]["shared_surface"] = dict(status="UNOBSERVABLE", points_in_band=int(len(band)))
                continue
            nrm, off, cnt, P = dominant_plane(band, rng)
            if nrm @ rn < 0:
                nrm, off = -nrm, -off
            mid = P.mean(0)
            planes[c]["shared_surface"] = dict(status="fitted", reference_camera=ref, points_in_band=int(len(band)), inliers=cnt,
                                               normal_angle_to_reference_deg=float(np.degrees(np.arccos(np.clip(nrm @ rn, -1, 1)))),
                                               offset_vs_reference_at_own_centroid_mm=float(((nrm @ mid - off) - (rn @ mid - ro)) * 1e3),
                                               reference_plane_residual_of_own_inliers_mm=dict(
                                                   median=float(np.median(P @ rn - ro) * 1e3),
                                                   p95_abs=float(np.percentile(np.abs(P @ rn - ro), 95) * 1e3)))
        pairs = {}
        for i, a in enumerate(cams):
            for b in cams[i + 1:]:
                na, nb = np.array(planes[a]["normal_world"]), np.array(planes[b]["normal_world"])
                ang = float(np.degrees(np.arccos(np.clip(abs(na @ nb), -1, 1))))
                # offset difference evaluated at the midpoint of the two inlier centroids
                mid = 0.5 * (np.array(planes[a]["centroid_world"]) + np.array(planes[b]["centroid_world"]))
                da = na @ mid - planes[a]["offset_m"]
                db = nb @ mid - planes[b]["offset_m"]
                pairs[f"{a}|{b}"] = dict(normal_angle_deg=ang, offset_difference_mm=float((da - db) * 1e3),
                                         centroid_distance_mm=float(np.linalg.norm(np.array(planes[a]["centroid_world"]) - np.array(planes[b]["centroid_world"])) * 1e3))
        result[str(frame)] = dict(planes=planes, pairs=pairs,
                                  definition="dominant RANSAC plane of non-object points per camera in the world frame, 1 cm inlier band (declared)")
    return result


def load_frame(scene, cam, frame):
    stem = f"frame_{frame:06d}"
    rgb = cv2.imread(str(scene / "rgb" / cam / f"{stem}.png"))[:, :, ::-1]
    depth = np.load(scene / "depth_npy" / cam / f"{stem}.npy").astype(np.float64)
    m = scene / "mask" / cam / f"{stem}.png"
    mask = (cv2.imread(str(m), 0) > 0) if m.is_file() else None
    return rgb, depth, mask


def edge_map(depth):
    d = depth.copy()
    d[d <= 0] = np.nan
    jumps = []
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        jumps.append(np.abs(d - np.roll(np.roll(d, dy, 0), dx, 1)))
    J = np.nanmax(np.stack(jumps), axis=0)
    thr = 3.0 * np.nanmedian(J)
    E = np.nan_to_num(J, nan=0.0) > thr
    return cv2.dilate(E.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, float(thr)


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0)
    a = np.abs(x)
    return dict(n=int(x.size), median_signed_mm=float(np.median(x) * 1e3), mad_mm=float(np.median(np.abs(x - np.median(x))) * 1e3),
                median_abs_mm=float(np.median(a) * 1e3), p95_abs_mm=float(np.percentile(a, 95) * 1e3),
                p5_signed_mm=float(np.percentile(x, 5) * 1e3), p95_signed_mm=float(np.percentile(x, 95) * 1e3))


def dense_residuals(K, Ta, Tb, fa, fb, step=2):
    rgb_a, da, ma = fa
    rgb_b, db, mb = fb
    H, W = da.shape
    v, u = np.mgrid[0:H:step, 0:W:step]
    z = da[v, u]
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    X = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    Tab = np.linalg.inv(Tb) @ Ta
    Xb = X @ Tab[:3, :3].T + Tab[:3, 3]
    zb = Xb[:, 2]
    ub = K[0, 0] * Xb[:, 0] / zb + K[0, 2]
    vb = K[1, 1] * Xb[:, 1] / zb + K[1, 2]
    inside = (zb > 0) & (ub >= 0) & (ub <= W - 1) & (vb >= 0) & (vb <= H - 1)
    ui, vi = np.round(ub[inside]).astype(int), np.round(vb[inside]).astype(int)
    zobs = db[vi, ui]
    has = zobs > 0
    res = np.full(inside.sum(), np.nan)
    res[has] = zobs[has] - zb[inside][has]
    edge_a, thr_a = edge_map(da)
    dark_a = rgb_a.max(axis=2) < 64
    obj = ma[v, u][inside] if ma is not None else np.zeros(inside.sum(), bool)
    edge = edge_a[v, u][inside]
    dark = dark_a[v, u][inside]
    strata = {}
    for name, sel in [("all", np.ones_like(obj)), ("object", obj), ("background", ~obj),
                      ("object_interior", obj & ~edge), ("object_edge", obj & edge),
                      ("background_interior", ~obj & ~edge), ("background_edge", ~obj & edge),
                      ("object_dark", obj & dark), ("object_not_dark", obj & ~dark)]:
        strata[name] = stats(res[sel])
    return strata, dict(projected_inside=int(inside.sum()), with_observed_depth=int(has.sum()), edge_threshold_m=thr_a,
                        object_mask_available=ma is not None), (u[inside], v[inside], ub[inside], vb[inside], res)


def feature_reprojection(K, Ta, Tb, fa, fb):
    sift = cv2.SIFT_create()
    ga, gb = cv2.cvtColor(np.ascontiguousarray(fa[0]), cv2.COLOR_RGB2GRAY), cv2.cvtColor(np.ascontiguousarray(fb[0]), cv2.COLOR_RGB2GRAY)
    ka, dA = sift.detectAndCompute(ga, None)
    kb, dB = sift.detectAndCompute(gb, None)
    if dA is None or dB is None or len(ka) < 8 or len(kb) < 8:
        return dict(matches=0)
    bf = cv2.BFMatcher()
    ab = bf.knnMatch(dA, dB, k=2)
    ba = {m.queryIdx: m.trainIdx for m in bf.match(dB, dA)}
    good = [m for m, n in ab if m.distance < 0.75 * n.distance and ba.get(m.trainIdx) == m.queryIdx]
    if len(good) < 8:
        return dict(matches=len(good))
    pa = np.array([ka[m.queryIdx].pt for m in good])
    pb = np.array([kb[m.trainIdx].pt for m in good])
    Tba = np.linalg.inv(Tb) @ Ta                      # a-camera coords -> b-camera coords
    R, t = Tba[:3, :3], Tba[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ tx @ R @ Kinv
    ha, hb = np.c_[pa, np.ones(len(pa))], np.c_[pb, np.ones(len(pb))]
    lb, la = ha @ F.T, hb @ F
    db_ = np.abs((hb * lb).sum(1)) / np.linalg.norm(lb[:, :2], axis=1)
    da_ = np.abs((ha * la).sum(1)) / np.linalg.norm(la[:, :2], axis=1)
    epi = 0.5 * (da_ + db_)
    za = fa[1][np.round(pa[:, 1]).astype(int), np.round(pa[:, 0]).astype(int)]
    zb_obs = fb[1][np.round(pb[:, 1]).astype(int), np.round(pb[:, 0]).astype(int)]
    ok = (za > 0) & (zb_obs > 0)
    X = (Kinv @ ha[ok].T).T * za[ok, None]
    Xb = X @ R.T + t
    proj = (K @ Xb.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    reproj = np.linalg.norm(proj - pb[ok], axis=1)
    dz = zb_obs[ok] - Xb[:, 2]
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else None
    return dict(matches=len(good), epipolar_px=dict(median=q(epi, 50), p95=q(epi, 95)),
                with_depth=int(ok.sum()), reprojection_px=dict(median=q(reproj, 50), p95=q(reproj, 95)),
                depth_residual_mm=dict(median=None if not ok.any() else float(np.median(dz) * 1e3),
                                       p95_abs=None if not ok.any() else float(np.percentile(np.abs(dz), 95) * 1e3)))


def overlay(path, fb, proj):
    u, v, ub, vb, res = proj
    img = np.ascontiguousarray(fb[0][:, :, ::-1]).copy()
    ok = np.isfinite(res)
    sel = np.where(ok)[0][::7]
    for i in sel:
        r = float(np.clip(res[i] / 0.03, -1, 1))
        color = (int(255 * max(0, -r)), int(255 * (1 - abs(r))), int(255 * max(0, r)))
        cv2.circle(img, (int(round(ub[i])), int(round(vb[i]))), 1, color, -1)
    cv2.putText(img, "a->b depth residual: green 0, red +3cm (b farther), blue -3cm", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True, help="synchronized source RGB-D indices")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--default-pair", nargs=2, default=["cam_001", "cam_002"])
    args = ap.parse_args()
    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cal = calibration_table(args.scene)
    cams = sorted(cal["cameras"])
    K = np.array(cal["cameras"][cams[0]]["K_cameras_json"])
    T = {c: np.array(cal["cameras"][c]["T_WC_cameras_json"]) for c in cams}
    ts = args.scene / "metadata/frame_timestamps.csv"
    cal["synchronization"] = dict(per_camera_timestamps="MISSING (one shared frame_timestamps.csv; capture-level sync assumed, UNVERIFIED)",
                                  frame_timestamps_sha256=sha(ts))
    rows, feats = [], []
    pairs = [(a, b) for i, a in enumerate(cams) for b in cams[i + 1:]]
    for frame in args.frames:
        F = {c: load_frame(args.scene, c, frame) for c in cams}
        for a, b in pairs:
            label = "default_fused_pair" if {a, b} == set(args.default_pair) else "stress_control_pair"
            for src, dst in ((a, b), (b, a)):
                strata, info, proj = dense_residuals(K, T[src], T[dst], F[src], F[dst])
                for name, s in strata.items():
                    rows.append(dict(frame=frame, source=src, target=dst, pair_label=label, stratum=name, **s,
                                     projected_inside=info["projected_inside"], with_observed_depth=info["with_observed_depth"]))
                if frame == args.frames[0]:
                    overlay(out / "figures" / f"reproj_{src}_to_{dst}_{frame}.jpg", F[dst], proj)
            fr = feature_reprojection(K, T[a], T[b], F[a], F[b])
            feats.append(dict(frame=frame, a=a, b=b, pair_label=label, **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in fr.items()}))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("frame", "source", "target", "pair_label", "stratum"), k))
    with open(out / "static_cross_camera_residuals.csv", "w", newline="") as h:
        w = csv.DictWriter(h, keys)
        w.writeheader()
        w.writerows(rows)
    fk = sorted({k for r in feats for k in r})
    with open(out / "feature_reprojection.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fk)
        w.writeheader()
        w.writerows(feats)
    cal["relative_pose_cameras_json_vs_source_handeye"] = relative_pose_consistency(cal)
    cal["dominant_plane_consistency"] = plane_consistency(args.scene, cal, args.frames)
    cal.pop("_source_T", None)
    cal["cross_camera_definitions"] = dict(signed_residual="z_observed_in_target - z_predicted_from_source (m); positive = target sees farther",
                                           edge="3x3-dilated pixels with max 8-neighbour depth jump > 3x image median jump",
                                           dark="max RGB channel < 64", frames=args.frames,
                                           note="same synchronized instant: object motion cannot produce these residuals")
    (out / "calibration_audit.json").write_text(json.dumps(cal, indent=1))
    print(json.dumps({k: v for k, v in cal.items() if k in ("shared_K_all_cameras", "calibration_status", "pairs")}, indent=1))


if __name__ == "__main__":
    main()
