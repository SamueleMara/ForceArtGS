"""G9 command line (also wired into pipelines/joint_aware_reconstruction.py as stages dynamics / final_physics_export).

    python -m forceartgs_g9.cli dynamics --run RUN --scene SCENE --dynamics-config CFG.yaml
        [--physics-library DIR] [--robot-archive A.npz] [--contact-segments C.yaml] [--force-calibration F.json]
        [--mass-model-spec M.yaml] [--dynamics-episodes E.yaml]
    python -m forceartgs_g9.cli final-physics-export --run RUN [--g9-dir DIR]
    python -m forceartgs_g9.cli synthetic-demo --out DIR [--kind revolute|prismatic] [--model full|no_static|viscous]
        (fully instrumented synthetic object with KNOWN physics: fit -> inertia -> PArtTwin asset -> engine parity)
"""
import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dynamics")
    d.add_argument("--run", required=True)
    d.add_argument("--scene", required=True)
    d.add_argument("--dynamics-config", required=True)
    d.add_argument("--physics-library", default=None)
    d.add_argument("--robot-archive", default=None)
    d.add_argument("--contact-segments", default=None)
    d.add_argument("--force-calibration", default=None)
    d.add_argument("--mass-model-spec", default=None)
    d.add_argument("--dynamics-episodes", default=None)
    d.add_argument("--allow-diagnostic-kinematics", action="store_true",
                   help="fit the working model of a TYPE_UNOBSERVABLE joint (labelled KINEMATICS_NOT_ACCEPTED)")
    d.add_argument("--declared-joint-type", default=None, choices=("revolute", "prismatic"),
                   help="operator-declared joint type for a TYPE_UNOBSERVABLE scene (OPERATOR_DECLARED_KINEMATICS: "
                        "conditional, never certified)")
    d.add_argument("--declaration-note", default=None, help="who declared it and when (required with the above)")
    d.add_argument("--force-export", default=None,
                   help="export_force_contact_annotations.py output dir (default <results>/contact_annotations/<scene>)")
    f = sub.add_parser("final-physics-export")
    f.add_argument("--run", required=True)
    f.add_argument("--g9-dir", default=None)
    f.add_argument("--physics-library", default=None)
    s = sub.add_parser("synthetic-demo")
    s.add_argument("--out", required=True)
    s.add_argument("--kind", default="revolute", choices=("revolute", "prismatic"))
    s.add_argument("--model", default="full", choices=("full", "no_static", "viscous"))
    s.add_argument("--physics-library", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "synthetic-demo":
        from pathlib import Path
        from . import physics_lib as pl
        pl.load(a.physics_library)
        from .core import run_g9
        from .dynamics.general_1dof import GeneralTerms
        from .export.final_build import final_physics_export
        from .testing import synthetic_cfg, synthetic_inputs
        t = {"full": GeneralTerms(), "no_static": GeneralTerms(static=False),
             "viscous": GeneralTerms(coulomb=False, static=False)}[a.model]
        out = Path(a.out) / f"{a.kind}_{a.model}"
        obj, inp, hand = synthetic_inputs(a.kind, out, terms=t)
        r = run_g9(inp, synthetic_cfg(t), out / "g9")
        ph = r["fit"]["physical"]
        truth = dict(alpha=obj.density, c=obj.c, Fc=obj.Fc if t.coulomb else None, Fs=obj.Fs if t.static else None)
        fe = final_physics_export(out / "g9", hand, out / "assets", parity_effort=inp.heldout[0].effort)
        print(json.dumps(dict(out=str(out), g9_status=r["manifest"]["status"],
                              estimate={k: ph.get(k) for k in truth}, truth=truth,
                              mass_kg=(r["recon"] or {}).get("mass_kg"),
                              true_mass_kg=obj.density * obj.mass_model.props.m1,
                              heldout=r["held"]["summary"], asset=(fe.get("build") or {}).get("asset_dir"),
                              claim=fe.get("claim"),
                              parity={e: v.get("status") for e, v in (fe.get("numerical_parity") or {}).items()}),
                         indent=1, default=str))
        return 0
    from . import physics_lib
    if a.cmd == "dynamics":
        from .stage import run_dynamics_stage
        r = run_dynamics_stage(a.run, a.scene, a.dynamics_config, physics_library=a.physics_library,
                               robot_archive=a.robot_archive, contact_segments=a.contact_segments,
                               force_calibration=a.force_calibration, mass_model_spec=a.mass_model_spec,
                               episodes=a.dynamics_episodes, force_export=a.force_export,
                               allow_diagnostic_kinematics=a.allow_diagnostic_kinematics,
                               declared_joint_type=a.declared_joint_type, declaration_note=a.declaration_note)
        print(json.dumps(dict(out_dir=r["out_dir"], status=r["manifest"]["status"],
                              status_categories=r["manifest"].get("status_categories"), blocked=r["manifest"]["blocked"],
                              reused=r.get("reused")), indent=1))
        return 0
    physics_lib.load(a.physics_library)
    from .stage import run_final_physics_export_stage
    r = run_final_physics_export_stage(a.run, g9_dir=a.g9_dir)
    print(json.dumps({k: r.get(k) for k in ("status", "reason", "claim", "readiness")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
