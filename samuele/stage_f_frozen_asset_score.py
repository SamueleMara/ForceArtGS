#!/usr/bin/env python3
"""Stage F on real data: score static / P / R with ONE frozen canonical Gaussian asset (actual gsplat
rasterizer, not the point proxy) on held-out times and identical pixels.

* Candidates are the TRAINING-SPLIT fits of a ``fit`` run (``model_selection.json:
  training_split_candidates``), so held-out frames were never used to estimate the joint.
* The asset's world frame is checked against the scene's current ``metadata/cameras.json``. If the
  asset was trained with superseded camera poses, the old->current transform is estimated from every
  matched camera; it is applied ONLY if all cameras agree on one rigid transform (else fail closed).
* Robot pixels: the measured tool segment (TCP -> EE + 10 cm, radius 3 cm) is projected and marked
  ``occluded_by_robot`` (no loss there). Nothing is trained; the asset fingerprint is checked.

    python samuele/stage_f_frozen_asset_score.py --scene <scene> --run <fit run> --asset-dir <nerfstudio exp dir>
        --tcp-npz <robot_archive.npz> --output <out.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from estimotion_2part.joint_discovery.gaussian_feedback import (load_canonical_asset, score_candidates_frozen,  # noqa: E402
                                                                transform_gaussians)
from estimotion_2part.joint_discovery.observations import load_scene_cameras, load_scene_frame_map, project  # noqa: E402


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def asset_to_current_world(asset_dir: Path, calibs, asset="twopart_blend", tol_deg=0.1, tol_m=0.002):
    """Asset world -> current world from the ASSET MODEL'S OWN camera poses (OpenCV c2w in its
    transforms_train.json). The distinct asset camera poses are assigned to calibrated cameras by the
    assignment whose implied rigid maps agree best; fail closed unless they agree within tolerance."""
    import itertools
    frames = json.loads((asset_dir / asset / "transforms_train.json").read_text())
    distinct = []
    for f in frames:
        T = np.eye(4)
        T[:3] = np.asarray(f.get("transform", f.get("transform_matrix")), float)[:3]
        if not any(np.allclose(T, D, atol=1e-6) for D in distinct):
            distinct.append(T)
    names = list(calibs)
    best = None
    for perm in itertools.permutations(names, min(len(distinct), len(names))):
        Ms = [calibs[c].T_WC @ np.linalg.inv(T) for c, T in zip(perm, distinct)]
        worst = (0.0, 0.0)
        for M in Ms[1:]:
            d = np.linalg.inv(Ms[0]) @ M
            worst = (max(worst[0], rot_deg(d[:3, :3])), max(worst[1], float(np.linalg.norm(d[:3, 3]))))
        if best is None or worst[0] + 100 * worst[1] < best[2][0] + 100 * best[2][1]:
            best = (perm, Ms[0], worst)
    perm, M, worst = best
    ok = len(distinct) >= 2 and len(distinct) <= len(names) and worst[0] <= tol_deg and worst[1] <= tol_m
    return dict(ok=ok, M=M, cameras_matched=list(perm), n_distinct_asset_poses=len(distinct),
                consistency_rot_deg=worst[0], consistency_trans_m=worst[1],
                offset_rot_deg=rot_deg(M[:3, :3]), offset_trans_m=float(np.linalg.norm(M[:3, 3])))


def robot_mask(cal, p_tcp, p_ee, radius=0.03, extend=0.10, n=40):
    d = p_ee - p_tcp
    dn = d / max(np.linalg.norm(d), 1e-9)
    pts = p_tcp[None] + np.linspace(0, 1, n)[:, None] * ((p_ee + extend * dn) - p_tcp)[None]
    uv, z = project(pts, cal.K, cal.T_CW)
    H, W = cal.height, cal.width
    m = np.zeros((H, W), bool)
    vv, uu = np.mgrid[0:H, 0:W]
    for (u, v), zz in zip(uv, z):
        if zz <= 0.05:
            continue
        r = cal.K[0, 0] * radius / zz
        if -r <= u < W + r and -r <= v < H + r:
            m |= (uu - u) ** 2 + (vv - v) ** 2 <= r * r
    return m


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--asset-dir", required=True, type=Path)
    ap.add_argument("--asset", default="twopart_blend")
    ap.add_argument("--tcp-npz", type=Path, default=None)
    ap.add_argument("--all-times", action="store_true", help="also score every time (diagnostic, includes training)")
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args(argv)
    ia = json.loads((a.run / "input_audit.json").read_text())
    sel = json.loads((a.run / "model_selection.json").read_text())
    cams = list(ia["cameras"])
    calibs, _ = load_scene_cameras(a.scene, cams)
    fm = load_scene_frame_map(a.scene, ia["frame_map"]["rgbd_frame_index"])
    ck = sorted((a.asset_dir / a.asset / "nerfstudio_models").glob("step-*.ckpt"))[-1]
    asset = load_canonical_asset(checkpoint=str(ck))
    out = dict(scene=str(a.scene), run=str(a.run), asset=str(ck), asset_status=asset["status"],
               asset_reason=asset["reason"], asset_warnings=asset["warnings"], asset_frame=asset["frame"])
    if asset["status"] != "OK":
        a.output.write_text(json.dumps(out, indent=1, default=str))
        print(json.dumps(out, indent=1, default=str))
        return 1
    wm = asset_to_current_world(a.asset_dir, calibs, a.asset)
    out["world_alignment"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in wm.items()}
    if not wm["ok"]:
        out["status"] = "INCOMPATIBLE_CANONICAL_ASSET: cameras disagree on the old->current world map"
        a.output.write_text(json.dumps(out, indent=1, default=str))
        return 1
    gs = transform_gaussians(asset["static"], wm["M"]) if asset["static"] is not None else None
    gm = transform_gaussians(asset["moving"], wm["M"])
    rec = None
    if a.tcp_npz is not None:
        from estimotion_2part.joint_discovery.tcp_contact import load_robot_contact_record
        rec = load_robot_contact_record(a.scene, fm, tcp_npz=a.tcp_npz)
    tc = sel["training_split_candidates"]
    T = fm.T
    cands = [dict(name="static", kind="static", q=np.zeros(T))]
    for k, v in tc.items():
        cands.append(dict(name=k, kind=k, axis_w=np.asarray(v["axis"], float),
                          origin_w=None if v["origin"] is None else np.asarray(v["origin"], float), q=np.asarray(v["q"], float)))
    heldout = list(ia["heldout_times"])
    sets = {"heldout": heldout} | ({"all": list(range(T))} if a.all_times else {})
    measured = {c: {} for c in cams}
    for c in cams:
        for t in sorted(set(sum(sets.values(), []))):
            f = int(fm.rgbd_frame_index[t])
            rgb = cv2.imread(str(a.scene / "rgb" / c / f"frame_{f:06d}.png"))[:, :, ::-1].astype(np.float32) / 255
            depth = np.load(a.scene / "depth_npy" / c / f"frame_{f:06d}.npy").astype(np.float32)
            mask = cv2.imread(str(a.scene / "mask" / c / f"frame_{f:06d}.png"), cv2.IMREAD_GRAYSCALE) > 127
            rm = robot_mask(calibs[c], rec.p_tcp_w[t], rec.p_ee_w[t]) if rec is not None else None
            measured[c][t] = dict(rgb=rgb, depth=depth, object_mask=mask, robot_mask=rm)
    out["results"] = {}
    for name, frames in sets.items():
        r = score_candidates_frozen(gs, gm, cands, calibs, frames, measured)
        r.pop("pixel_sets", None)
        out["results"][name] = r
        print(name, "ranking", r["ranking"], {c["name"]: round(c["total"], 4) for c in r["candidates"]})
    out["vision_decision_of_run"] = sel["decision"]
    out["notes"] = ["asset co-trained with the legacy joint (not hypothesis-neutral); identical pixels for all candidates",
                    "renderer_evaluation_kind = actual_gaussian (gsplat 0.1.11 FreeArtGS path)"]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
