"""Standalone entry point: a small, tested synthetic run (ground truth known) of the input-output identification.

    python -m forceartgs_g9.system_identification.cli synthetic-demo --out DIR [--free-only] [--physx]

Real runs go through ``python -m forceartgs_g9.cli io-identification --run RUN --scene SCENE``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def synthetic_demo(out, free_only=False, physx=False, log=print):
    from . import synthetic_io as S
    from .api import identify_io_response, reconcile_physx
    from .contracts import load_config
    M, c, Fc = 0.8, 0.6, 0.3
    segs = [S.segment("coast", label="FREE_CONFIRMED", M=M, c=c, Fc=Fc, v0=1.4, T=1.1, seed=3, t_start=6.0)]
    if not free_only:
        segs = [S.segment("push_a", label="FORCED_MEASURED", M=M, c=c, Fc=Fc, v0=0.4, T=1.6,
                          effort_levels=S.pulse_train([2.0, 0.6, 1.4, 0.2], 0.3), seed=1),
                S.segment("push_b", label="FORCED_MEASURED", M=M, c=c, Fc=Fc, v0=0.8, T=1.6,
                          effort_levels=S.pulse_train([1.0, 2.5], 0.4), seed=2, t_start=3.0)] + segs
    eps = S.episodes(segs, heldout="last_if_multiple")
    cfg = load_config({"fit": {"compare_methods": ["output_error", "weak_form", "normalized_ekf"]}})
    out = Path(out)
    res = identify_io_response(eps, dict(joint_type="revolute", kinematic_status="IDENTIFIED", export_ready=True,
                                         note="SYNTHETIC ground truth"), dict(force_audit_status="VERIFIED",
                                                                              source="synthetic_io"),
                               dict(prior_M_range=[0.2, 5.0], prior_provenance="ASSUMED (demo)"), cfg, out, log=log)
    truth = dict(M=M, c=c, Fc=Fc, lam=c / M, gam=Fc / M, g_u=1 / M)
    if physx and res["status"] == "OK":
        import numpy as np
        from ..physx_refinement.sapien_diagnostic import IsolatedSapienCpu
        from ..physx_refinement.urdf_builder import write_single_joint_urdf
        info = write_single_joint_urdf(out / "demo.urdf", joint_type="revolute", axis=(0, 0, 1),
                                       origin=np.zeros(3), mass=1.0, com_ref=np.array([0.4, 0.0, 0.0]),
                                       inertia_com_ref=np.diag([0.02, 0.02, 0.02]))
        sim = IsolatedSapienCpu(info["path"], moving_link="moving_link", joint_name="joint", dt=1e-3,
                                gravity=[0, 0, -9.81], friction_mode="external_effort_surrogate")
        try:
            rec, physx_q = reconcile_physx(res, sim, eps, axis_inertia_unit=info["axis_inertia"], cfg=cfg, log=log)
        finally:
            sim.close()
        from .api import _write, write_replays
        _write(out / "physx_reconciliation.json", rec)
        write_replays(out, eps, res["_oe"], physx_q)
    summary = dict(truth=truth, status=res["status"],
                   features={f["name"]: dict(estimate=f.get("estimate"), status=f["status"],
                                             ci95=(f.get("uncertainty") or {}).get("ci95"))
                             for f in res["features"] if f["name"] in ("lambda", "gamma", "gain")},
                   scale_profile=res["scale_profile"]["status"], ci95_M=res["scale_profile"].get("ci95_M"),
                   status_dimensions={k: v for k, v in res["status_dimensions"].items() if not k.startswith("_")})
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="forceartgs_g9.system_identification.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("synthetic-demo")
    d.add_argument("--out", required=True)
    d.add_argument("--free-only", action="store_true")
    d.add_argument("--physx", action="store_true", help="also reconcile in actual PhysX (SAPIEN worker)")
    a = ap.parse_args(argv)
    if a.cmd == "synthetic-demo":
        print(json.dumps(synthetic_demo(a.out, a.free_only, a.physx), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
