#!/usr/bin/env python3
"""G8 producer adapter: write an articulated3dgs.handoff/v1 package from a finished joint-aware run.

    python tools/export_parttwin_handoff.py --run <out_root>/<scene>/<budget> [--output DIR]
        [--diagnostic-branch revolute|prismatic]

Reads ONLY the joint-aware outputs: the branch authority (accepted joint, W frame), the blend-frame map N_BW, the
frozen splatfacto-art checkpoint / its exported Graphdeco PLY and the per-Gaussian part weights. Never reads a vanilla
articulation.json, never calls estimate_joint.py. Production (status accepted) requires a joint selection v2 result with
physical_joint_status IDENTIFIED_* AND physics_joint_export_ready=true, and final/ PASSED. A legacy (render-based)
'IDENTIFIED' selection is NOT a certified physical joint: it is exported as production only with an explicit
user-approved --assumed-physical-joint note (recorded as 'assumed' provenance). --diagnostic-branch writes a uniquely
named DIAGNOSTIC handoff (TYPE_UNOBSERVABLE / ambiguous scenes), never a physical twin.

Hard membership: every Gaussian must have w < 0.01 or w > 0.99 (splatfacto-art renders alpha(1-w) / alpha w). Rows in
between make an exact two-PLY split impossible (spec 4.3) and the adapter ABSTAINS. The render gap of the hard split
(max opacity factor change min(w, 1-w)) is recorded. Each output PLY keeps the source vertex dtype for the standard
3DGS fields only (the derived uint8 display colours red/green/blue are dropped, explicitly recorded); no value changes.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from estimotion_2part.joint_discovery.integration_contract import JointAuthority, atomic_write_json, sha256_file  # noqa

STANDARD = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
            "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
DERIVED_DROPPED = ("red", "green", "blue")       # uint8 display colours in the ns-export PLY, not 3DGS parameters
HARD_EPS = 0.01


class HandoffAbstain(RuntimeError):
    pass


def _checkpoint_vertices(art: Path, v_export):
    """Standard Graphdeco vertex table built from the FROZEN checkpoint tensors exactly as stored (means, log-scales,
    opacity logits, SH DC/rest; quaternions only re-indexed xyzw -> wxyz). The ns-export PLY is a cross-check only:
    it re-multiplies quaternions by the identity rotation offset (float32 rounding)."""
    import torch
    import yaml
    cfg = yaml.load((art / "config.yml").read_text(), Loader=yaml.Loader)
    order = getattr(cfg.pipeline.model, "quaternion_order", None)
    if order != "xyzw":
        raise HandoffAbstain(f"quaternion_order is {order!r}; only the stored-xyzw layout is supported")
    dp = json.loads((art / "dataparser_transforms.json").read_text())
    if np.abs(np.asarray(dp["transform"], float) - np.eye(4)[:3]).max() > 0 or float(dp["scale"]) != 1.0:
        raise HandoffAbstain("splatfacto-art used a dataparser normalisation; Gaussians are not in the blend frame B")
    ck = sorted((art / "nerfstudio_models").glob("step-*.ckpt"))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)["pipeline"]
    g = lambda name: sd[[k for k in sd if k.endswith(f"gauss_params.{name}")][0]].numpy()
    means, quats, scales, opac = g("means"), g("quats"), g("scales"), g("opacities").reshape(-1)
    dc = g("features_dc").reshape(len(means), -1)
    rest = g("features_rest")
    rest = rest.transpose(0, 2, 1).reshape(len(means), -1) if rest.size else np.zeros((len(means), 0), np.float32)
    cols = [("x", means[:, 0]), ("y", means[:, 1]), ("z", means[:, 2])] + \
           [(n, np.zeros(len(means), np.float32)) for n in ("nx", "ny", "nz")] + \
           [(f"f_dc_{i}", dc[:, i]) for i in range(3)] + [(f"f_rest_{i}", rest[:, i]) for i in range(rest.shape[1])] + \
           [("opacity", opac)] + [(f"scale_{i}", scales[:, i]) for i in range(3)] + \
           [(f"rot_{i}", quats[:, k]) for i, k in enumerate((3, 0, 1, 2))]
    vert = np.empty(len(means), dtype=[(n, "<f4") for n, _ in cols])
    for n, c in cols:
        vert[n] = np.asarray(c, np.float32)
    cross = {n: float(np.abs(np.asarray(v_export[n], np.float64) - vert[n]).max())
             for n, _ in cols if n in v_export.dtype.names and not n.startswith("n")}
    if max(cross.values()) > 1e-6:
        raise HandoffAbstain(f"ns-export PLY and checkpoint disagree beyond float32 rounding: {cross}")
    w = 1 / (1 + np.exp(-sd[[k for k in sd if k.endswith("part_weights")][0]].numpy().astype(np.float64).ravel()))
    return ck, cross, w, vert


def physical_provenance(sel: dict, final_status, assumed_note=None) -> dict:
    """Production gate (spec 1.2 / 7.2): a certified v2 physical joint, or an explicit user-approved assumption."""
    if sel.get("schema_version") == 2:
        ok = (str(sel.get("status", "")).startswith("IDENTIFIED_") and sel.get("physics_joint_export_ready") is True
              and final_status == "PASSED")
        if not ok:
            raise HandoffAbstain(f"no certified physical joint (v2 status {sel.get('status')}, physics_joint_export_ready "
                                 f"{sel.get('physics_joint_export_ready')}, final {final_status}); use --diagnostic-branch")
        return dict(provenance="identified", physical_joint_status=sel["status"], physics_joint_export_ready=True)
    if sel.get("status") == "IDENTIFIED" and final_status == "PASSED" and assumed_note:
        return dict(provenance="assumed", physical_joint_status="LEGACY_SELECTION_NOT_PHYSICALLY_CERTIFIED",
                    physics_joint_export_ready=False, user_approval_note=str(assumed_note))
    raise HandoffAbstain(f"run is not accepted (selection {sel.get('status')}, final {final_status}): legacy selections "
                         "need --assumed-physical-joint '<approval note>'; otherwise use --diagnostic-branch")


def build_handoff(run: Path, out: Path, diagnostic_branch=None, assumed_note=None) -> dict:
    from plyfile import PlyData, PlyElement
    sel = json.loads((run / "selection.json").read_text()) if (run / "selection.json").is_file() else {}
    man = json.loads((run / "run_manifest.json").read_text())
    if diagnostic_branch:
        branch, status, cand = diagnostic_branch, "diagnostic", f"diagnostic_{diagnostic_branch}"
        phys = dict(provenance="diagnostic", physical_joint_status=sel.get("status"), physics_joint_export_ready=False,
                    note="candidate diagnostic: NOT a physical joint, NOT_FOR_RESULTS")
    else:
        phys = physical_provenance(sel, man["stages"].get("final", {}).get("status"), assumed_note)
        branch, status, cand = sel["selected"], "accepted", f"accepted_{sel['selected']}"
    b = Path(man.get("branch_dirs", {}).get(branch) or run / "branches" / branch)
    art = b / "splatfacto-art"
    auth = JointAuthority.load(b / "joint_authority" / "v000")
    aid = json.loads((b / "joint_authority" / "v000" / "authority.json").read_text())["authority_id"]
    marker = json.loads((b / "twopart_blend" / "JOINT_AWARE_AUTHORITY.json").read_text())
    if marker["authority_id"] != aid:
        raise HandoffAbstain("blend/art adapter was built from another authority")
    for st in ("blend", "art"):
        if man["stages"].get(st, {}).get("status") != "PASSED":
            raise HandoffAbstain(f"stage {st} did not pass")
    N = np.asarray(marker["N_BW"], float)
    ply = PlyData.read(str(art / "object_3dgs.ply"))
    v = ply["vertex"].data
    unknown = [n for n in v.dtype.names if n not in STANDARD + DERIVED_DROPPED and not n.startswith("f_rest_")]
    if unknown:
        raise HandoffAbstain(f"unexpected PLY fields {unknown}")
    ck, ck_err, w_ck, vert = _checkpoint_vertices(art, v)
    w_txt = np.loadtxt(art / "3dgs_part_weight.txt")
    if len(w_txt) != len(v) or np.abs(w_txt - w_ck).max() > 1e-6:
        raise HandoffAbstain("3dgs_part_weight.txt does not match the checkpoint part weights row for row")
    inter = (w_ck > HARD_EPS) & (w_ck < 1 - HARD_EPS)
    if inter.any():
        raise HandoffAbstain(f"{int(inter.sum())} of {len(w_ck)} Gaussians have intermediate membership "
                             f"(w in ({HARD_EPS}, {1 - HARD_EPS})): strict two-PLY export impossible (spec 4.3)")
    moving = w_ck >= 0.5
    gap = float(np.minimum(w_ck, 1 - w_ck).max())
    keep = list(vert.dtype.names)
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for nm, m in (("static", ~moving), ("moving", moving)):
        idx = np.flatnonzero(m)
        sub = vert[idx]
        PlyData([PlyElement.describe(sub, "vertex")], text=False).write(str(out / f"{nm}_canonical.ply"))
        rows[nm] = idx
    np.savez(out / "gaussian_lineage.npz",
             **{f"{nm}_gaussian_uid": rows[nm].astype(np.int64) for nm in rows},
             **{f"{nm}_parent_uid": np.full(len(rows[nm]), -1, np.int64) for nm in rows},
             **{f"{nm}_unresolved": np.zeros(len(rows[nm]), np.uint8) for nm in rows},
             **{f"{nm}_part_weight": w_ck[rows[nm]] for nm in rows},
             note=np.array("uid = row of splatfacto-art/object_3dgs.ply (= checkpoint row); weights are the frozen "
                           "splatfacto-art memberships (all < 0.01 or > 0.99)"))
    t_ref = auth.reference_time_index
    fm_rows = [dict(original_frame=int(auth.rgbd_frame_index[t]), source_frame_index=int(auth.source_frame_index[t]),
                    timestamp_s=float(auth.timestamp_s[t]), q=float(auth.q[t]), q_valid=bool(auth.q_valid[t]),
                    q_uncertainty=None if auth.q_sigma is None else float(auth.q_sigma[t]),
                    processing_index=t, direction="forward") for t in range(auth.T)]
    atomic_write_json(out / "frame_map.json", dict(
        rows=fm_rows, original_frame_definition="RGB-D frame id (rgb/<cam>/frame_XXXXXX.png); source_frame_index "
        "joins the robot recording", anchor_original_frame=int(auth.rgbd_frame_index[t_ref])))
    joint = dict(type=auth.joint_type, axis_Wc=auth.axis_world.tolist(),
                 origin_Wc=None if auth.origin_world is None else auth.origin_world.tolist(), q_anchor=0.0,
                 q_unit=auth.q_units, q_observed_range=[float(auth.q.min()), float(auth.q.max())], reference_frame="Wc",
                 uncertainty=dict(per_time_q_sigma=auth.q_sigma is not None, fit_covariance=auth.fit_covariance_status,
                                  authority_id=aid),
                 selection_status="accepted" if status == "accepted" else "diagnostic")
    fp = hashlib.sha256((out / "frame_map.json").read_bytes() + json.dumps(joint, sort_keys=True).encode()).hexdigest()
    # N_BW was recovered from float32 camera exports: project it onto SO(3) exactly and record the correction
    U, _, Vt = np.linalg.svd(N[:3, :3])
    Rn = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    n_fix = float(np.abs(Rn - N[:3, :3]).max())
    if n_fix > 1e-5:
        raise HandoffAbstain(f"N_BW is not rigid (rotation deviates by {n_fix:.2e})")
    N = N.copy()
    N[:3, :3] = Rn
    T_Wc_P = np.eye(4)
    T_Wc_P[:3, :3] = Rn.T
    T_Wc_P[:3, 3] = -Rn.T @ N[:3, 3]
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    part = lambda nm: dict(ply=f"{nm}_canonical.ply", ply_sha256=sha256_file(out / f"{nm}_canonical.ply"),
                           T_Wc_P=T_Wc_P.tolist(), immutable_gaussian_fields=keep, membership="hard",
                           lineage="gaussian_lineage.npz", n_gaussians=int(len(rows[nm])))
    art_eval = man["stages"].get("art", {}).get("branches", {}).get(branch, {})
    handoff = dict(
        schema="articulated3dgs.handoff/v1", object_id=Path(man.get("scene", run.parent.name)).name if status == "accepted"
        else f"{Path(man.get('scene', run.parent.name)).name}__{cand}", producer="ForceArtGS",
        source=dict(run_dir=str(run.resolve()), git_sha=f"{git('rev-parse', 'HEAD')} dirty={len(git('status', '--porcelain').splitlines())}",
                    config_sha256=sha256_file(art / "config.yml"), joint_fingerprint=fp,
                    gaussian_fingerprint=sha256_file(art / "object_3dgs.ply"),
                    acquisition_world="Wc = metadata/cameras.json world (OpenCV, metres)", coordinate_units="m_rad_s",
                    gaussian_parameter_space="canonical frame B = N_BW Wc of the joint-aware blend/art stages; T_Wc_P = N_BW^-1",
                    canonical_anchor_original_frame=int(auth.rgbd_frame_index[t_ref]),
                    canonical_anchor_timestamp_s=float(auth.timestamp_s[t_ref]), candidate_id=cand, status=status,
                    notes=dict(selection=sel, physical_joint=phys,
                               multimodal_provenance=sel.get("multimodal") or dict(status="NOT_RUN",
                                                                                    note="visual-only selection"),
                               dropped_ply_fields=list(DERIVED_DROPPED), checkpoint=str(ck),
                               gaussian_source="frozen splatfacto-art checkpoint tensors (exact); ns-export PLY used as cross-check",
                               ns_export_vs_checkpoint_max_abs=ck_err, budget_iterations=man.get("iterations"), N_BW_orthonormalisation_max_abs=n_fix)),
        joint=joint, frames=dict(file="frame_map.json", one_shared_q_per_original_frame=True),
        parts=dict(static=part("static"), moving=part("moving")),
        validation=dict(joint_frozen_during_final_gaussian_training=True,
                        joint_unchanged_proof=dict(run_manifest=str((run / "run_manifest.json").resolve()), art=art_eval),
                        reliable_metric_scale=True, heldout_render_checked=bool(sel.get("branches", {}).get(branch) or
                                                    (sel.get("render_evidence") or {}).get("per_branch", {}).get(branch)),
                        frame_equivalence_checked=man["stages"]["blend"]["status"] == "PASSED", hard_membership=True,
                        soft_render_gap=dict(max_opacity_factor_change=gap,
                                             note="hard split drops alpha*min(w,1-w) per Gaussian vs the soft renderer")))
    atomic_write_json(out / "articulated_3dgs_handoff.json", handoff)
    return dict(handoff=str(out / "articulated_3dgs_handoff.json"), status=status, branch=branch,
                n_static=int(len(rows["static"])), n_moving=int(len(rows["moving"])), soft_render_gap=gap)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, type=Path, help="<output-root>/<scene>/<budget> of joint_aware_reconstruction")
    ap.add_argument("--output", type=Path, help="default <run>/parttwin_handoff[/diagnostic_<branch>]")
    ap.add_argument("--diagnostic-branch", choices=("revolute", "prismatic"))
    ap.add_argument("--assumed-physical-joint", default=None, metavar="APPROVAL_NOTE",
                    help="legacy selections only: user-approved assumption, recorded as 'assumed' provenance")
    a = ap.parse_args(argv)
    out = a.output or (a.run / "parttwin_handoff" / (f"diagnostic_{a.diagnostic_branch}" if a.diagnostic_branch else "accepted"))
    try:
        print(json.dumps(build_handoff(a.run, out, a.diagnostic_branch, a.assumed_physical_joint), indent=2))
    except HandoffAbstain as e:
        print(f"ABSTAIN: {e}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
