"""G9 command line (also wired into pipelines/joint_aware_reconstruction.py as stages dynamics / final_physics_export).

    python -m forceartgs_g9.cli dynamics --run RUN --scene SCENE --dynamics-config CFG.yaml
        [--physics-library DIR] [--robot-archive A.npz] [--contact-segments C.yaml] [--force-calibration F.json]
        [--mass-model-spec M.yaml] [--dynamics-episodes E.yaml]
    python -m forceartgs_g9.cli final-physics-export --run RUN [--g9-dir DIR]
    python -m forceartgs_g9.cli synthetic-demo --out DIR [--kind revolute|prismatic] [--model full|no_static|viscous]
        (fully instrumented synthetic object with KNOWN physics: fit -> inertia -> PArtTwin asset -> engine parity)
    python -m forceartgs_g9.cli fridge-audit | fit-minimal-dynamics [--pinn] | fit-mechanism-residual
    python -m forceartgs_g9.cli free-forced-dynamics [--pinn]   (normalized free motion + forced scale)
        --run RUN --scene SCENE [--robot-archive A.npz] [--contact-segments C.yaml] [--discovery-dir D]
        [--pulse-rows A:B] [--declared-joint-type T --declaration-note N]
        (ForceTwin-adaptation: identifiability-first effective I_eff / mu / b; append-only forcetwin_adaptation/vNNN)
    python -m forceartgs_g9.cli physx-refinement --run RUN --scene SCENE [--quick] [--pulse-rows A:B] ...
        (G10: SAPIEN PhysX CPU diagnostic refinement; append-only <run>/dynamics/g9/physx_refinement/vNNN)
        [--use-io-initialization [IO_RESPONSE]]  (warm start from the input-output identification)
    python -m forceartgs_g9.cli io-identification --run RUN --scene SCENE [--io-config YAML] [--pulse-rows A:B]
        (physics-constrained input-output identification; append-only <run>/dynamics/g9/io_identification/vNNN)
    python -m forceartgs_g9.cli compare-dynamics --root RUNS --out DIR
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
    d.add_argument("--declared-pulse-rows", default=None, action="append",
                   help="A:B source-row range reported as a declared pulse beside threshold detection (repeatable)")
    f = sub.add_parser("final-physics-export")
    f.add_argument("--run", required=True)
    f.add_argument("--g9-dir", default=None)
    f.add_argument("--physics-library", default=None)
    s = sub.add_parser("synthetic-demo")
    s.add_argument("--out", required=True)
    s.add_argument("--kind", default="revolute", choices=("revolute", "prismatic"))
    s.add_argument("--model", default="full", choices=("full", "no_static", "viscous"))
    s.add_argument("--physics-library", default=None)
    ft_steps = {"fridge-audit": ("audit",), "fit-minimal-dynamics": ("audit", "identify", "general"),
                "fit-mechanism-residual": ("audit", "identify", "residual")}
    for name, steps in ft_steps.items():
        x = sub.add_parser(name, help=f"ForceTwin-adaptation steps {steps} -> <run>/dynamics/forcetwin_adaptation/vNNN")
        x.add_argument("--run", required=True)
        x.add_argument("--scene", required=True)
        x.add_argument("--config", default="configs/g9_real/forcetwin_effective.yaml")
        x.add_argument("--robot-archive", default=None)
        x.add_argument("--contact-segments", default=None)
        x.add_argument("--discovery-dir", default=None, help="joint-discovery dir with tracks_observations.npz")
        x.add_argument("--pulse-rows", default=None, help="A:B declared pulse source rows (signed impulse audit)")
        x.add_argument("--declared-joint-type", default=None, choices=("revolute", "prismatic"))
        x.add_argument("--declaration-note", default=None)
        x.add_argument("--bootstrap", type=int, default=200)
        x.add_argument("--out-root", default=None)
        if name == "fit-minimal-dynamics":
            x.add_argument("--pinn", action="store_true", help="also run GeneralPINN (long)")
            x.add_argument("--residual", action="store_true", help="also run the mechanism-residual ablations")
    ff = sub.add_parser("free-forced-dynamics", help="normalized free-motion + forced-scale dynamics -> "
                                                    "<run>/dynamics/g9_free_forced/vNNN")
    ff.add_argument("--run", required=True)
    ff.add_argument("--scene", required=True)
    ff.add_argument("--config", default=None, help="base dynamics config (default: the free_forced config's "
                                                    "dynamics_config)")
    ff.add_argument("--ff-config", default="configs/g9_real/free_forced.yaml")
    ff.add_argument("--robot-archive", default=None)
    ff.add_argument("--contact-segments", default=None)
    ff.add_argument("--discovery-dir", default=None)
    ff.add_argument("--pulse-rows", default=None, help="A:B declared pulse source rows (signed audit)")
    ff.add_argument("--declared-joint-type", default=None, choices=("revolute", "prismatic"))
    ff.add_argument("--declaration-note", default=None)
    ff.add_argument("--pinn", action="store_true", help="also run the normalized PINN (phase A, and B if forced)")
    ff.add_argument("--out-root", default=None)
    px = sub.add_parser("physx-refinement", help="G10 PhysX (SAPIEN CPU diagnostic) refinement -> "
                                                 "<run>/dynamics/g9/physx_refinement/vNNN")
    px.add_argument("--run", required=True)
    px.add_argument("--scene", required=True)
    px.add_argument("--g10-config", default="configs/g9_real/physx_refinement.yaml")
    px.add_argument("--config", default=None, help="base dynamics config (default: the free_forced config's)")
    px.add_argument("--robot-archive", default=None)
    px.add_argument("--contact-segments", default=None)
    px.add_argument("--pulse-rows", default=None, help="A:B declared pulse source rows (impulse audit -> blocker)")
    px.add_argument("--declared-joint-type", default=None, choices=("revolute", "prismatic"))
    px.add_argument("--declaration-note", default=None)
    px.add_argument("--engine-python", default=None, help="SAPIEN interpreter (default $PARTTWIN1D_PYTHON / parttwin_sim)")
    px.add_argument("--quick", action="store_true", help="tiny optimiser budget (smoke test only)")
    px.add_argument("--out-root", default=None)
    px.add_argument("--use-io-initialization", nargs="?", const="latest", default=None, metavar="IO_RESPONSE",
                    help="warm-start G10 from an input-output identification (io_response.json path; default: the "
                         "latest <run>/dynamics/g9/io_identification/vNNN). A start point only, not a loss term")
    io = sub.add_parser("io-identification", help="physics-constrained input-output identification (opt-in) -> "
                                                  "<run>/dynamics/g9/io_identification/vNNN")
    io.add_argument("--run", required=True)
    io.add_argument("--scene", required=True)
    io.add_argument("--io-config", default="configs/g9_real/io_identification.example.yaml")
    io.add_argument("--robot-archive", default=None)
    io.add_argument("--contact-segments", default=None)
    io.add_argument("--pulse-rows", default=None, help="A:B declared pulse source rows (impulse audit -> blocker)")
    io.add_argument("--declared-joint-type", default=None, choices=("revolute", "prismatic"))
    io.add_argument("--declaration-note", default=None)
    io.add_argument("--engine-python", default=None, help="SAPIEN interpreter for the PhysX reconciliation")
    io.add_argument("--no-physx", action="store_true", help="skip the PhysX reconciliation rollouts")
    io.add_argument("--out-root", default=None)
    c = sub.add_parser("compare-dynamics", help="aggregate forcetwin_adaptation comparison tables")
    c.add_argument("--root", required=True, help="directory with *_joint_aware_full runs")
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "io-identification":
        from .system_identification.pipeline import run as io_run
        pr = tuple(int(x) for x in a.pulse_rows.split(":")) if a.pulse_rows else None
        st = io_run(a.run, a.scene, io_config=a.io_config, robot_archive=a.robot_archive,
                    contact_segments=a.contact_segments, pulse_rows=pr, out_root=a.out_root,
                    declared_joint_type=a.declared_joint_type, declaration_note=a.declaration_note,
                    python=a.engine_python, no_physx=a.no_physx)
        print(json.dumps(st, indent=1, default=str))
        return 0
    if a.cmd == "physx-refinement":
        from .physx_refinement.real_data import run as px_run
        pr = tuple(int(x) for x in a.pulse_rows.split(":")) if a.pulse_rows else None
        io_init = a.use_io_initialization
        if io_init == "latest":
            from pathlib import Path as _P
            vs = sorted((_P(a.run) / "dynamics" / "g9" / "io_identification").glob("v[0-9][0-9][0-9]/io_response.json"))
            if not vs:
                ap.error("--use-io-initialization: no io_identification/vNNN/io_response.json under the run")
            io_init = str(vs[-1])
        st = px_run(a.run, a.scene, g10_config=a.g10_config, cfg_path=a.config, robot_archive=a.robot_archive,
                    contact_segments=a.contact_segments, pulse_rows=pr, out_root=a.out_root,
                    declared_joint_type=a.declared_joint_type, declaration_note=a.declaration_note, quick=a.quick,
                    python=a.engine_python, io_init=io_init)
        print(json.dumps(st, indent=1, default=str))
        return 0
    if a.cmd in ft_steps:
        from .forcetwin.pipeline import run as ft_run
        pr = tuple(int(x) for x in a.pulse_rows.split(":")) if a.pulse_rows else None
        st = ft_run(a.run, a.scene, cfg_path=a.config, robot_archive=a.robot_archive,
                    contact_segments=a.contact_segments, discovery_dir=a.discovery_dir, pulse_rows=pr,
                    steps=ft_steps[a.cmd] + (("residual",) if getattr(a, "residual", False) else ()),
                    pinn=bool(getattr(a, "pinn", False)), out_root=a.out_root,
                    declared_joint_type=a.declared_joint_type, declaration_note=a.declaration_note,
                    bootstrap=a.bootstrap)
        print(json.dumps({k: st.get(k) for k in ("status", "out_dir", "primary_hypothesis", "impulse", "states",
                                                 "reason")}, indent=1, default=str))
        return 0
    if a.cmd == "free-forced-dynamics":
        from .free_forced.pipeline import run as ff_run
        pr = tuple(int(x) for x in a.pulse_rows.split(":")) if a.pulse_rows else None
        st = ff_run(a.run, a.scene, cfg_path=a.config, ff_config=a.ff_config, robot_archive=a.robot_archive,
                    contact_segments=a.contact_segments, discovery_dir=a.discovery_dir, pulse_rows=pr, pinn=a.pinn,
                    out_root=a.out_root, declared_joint_type=a.declared_joint_type,
                    declaration_note=a.declaration_note)
        print(json.dumps({k: st.get(k) for k in ("status", "out_dir", "factorized_status", "free_basis", "impulse",
                                                 "reason")}, indent=1, default=str))
        return 0
    if a.cmd == "compare-dynamics":
        from .forcetwin.compare import compare
        print(json.dumps(compare(a.root, a.out), indent=1, default=str))
        return 0
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
                               declared_joint_type=a.declared_joint_type, declaration_note=a.declaration_note,
                               declared_pulse_rows=[w.split(":") for w in a.declared_pulse_rows or []])
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
