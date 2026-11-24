"""Input-output identification on a REAL ForceArtGS run: append-only <run>/dynamics/g9/io_identification/vNNN.

Data owner: the existing G9 inputs (joint authority, canonical robot record, force audit, mass geometry, gravity),
through the SAME classifier + options as the free/forced stage and the SAME G10 bridge (identical windows / clock).
Nothing is recalibrated, re-projected, re-tared or sign-flipped; no PArtTwin1D import.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from .contracts import load_config


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if p and Path(p).is_file() else None


def gravity_hypotheses(inp, mm, jt):
    """Primary = accepted axis + G9 geometry gravity (density cancels); alternatives labelled separately."""
    from ..dynamics.general_1dof import RigidBodyGravity
    from ..free_forced.normalized import GravityModel
    if inp.gravity is None:
        return dict(primary=GravityModel.zero(jt, "gravity_unknown_zero_assumed", provenance="NO registered gravity: "
                                              "zero load ASSUMED (conditional)"), alternatives={}, rigid_unit=None,
                    M1=None, verified=False)
    g = inp.gravity.g_world
    if jt == "prismatic":
        a = np.asarray(inp.axis, float)
        prim = GravityModel.prismatic_ideal(a, g, "prismatic_ideal_accepted_axis")
        rbg = RigidBodyGravity("prismatic", tuple(a / np.linalg.norm(a)), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0,
                               tuple(g), 0.0)
        alt = dict(horizontal_slide_zero_gravity=GravityModel.zero(
            jt, "horizontal_slide_zero_gravity", provenance="HYPOTHESIS: exactly horizontal slide (G = 0)"))
        return dict(primary=prim, alternatives=alt, rigid_unit=rbg, M1=1.0, verified=bool(inp.gravity.verified))
    if mm is None:
        return dict(primary=GravityModel.zero(jt, "gravity_geometry_unavailable_zero_assumed",
                                              provenance="NO validated mass geometry: zero load ASSUMED"),
                    alternatives={}, rigid_unit=None, M1=None, verified=False)
    prim = GravityModel.from_mass_model(mm, g, "known_shape_accepted_axis")
    alt = dict(vertical_axis_zero_gravity=GravityModel.zero(
        jt, "vertical_axis_zero_gravity", "vertical_exact",
        provenance="HYPOTHESIS: hinge exactly vertical (G = G' = 0) -- a separately labelled geometry test, not "
                   "ground truth"))
    return dict(primary=prim, alternatives=alt, rigid_unit=mm.gravity_unit(g), M1=mm.M1,
                verified=bool(inp.gravity.verified))


def run(run_root, scene, *, io_config=None, robot_archive=None, contact_segments=None, pulse_rows=None,
        out_root=None, declared_joint_type=None, declaration_note=None, python=None, no_physx=False,
        log=print) -> dict:
    from ..forcetwin.pipeline import _clean, _git, _mass_model, _prior_for_sim, _write, next_version
    from ..physx_refinement.real_data import reference_urdf
    from . import api
    from . import segmentation_adapter as SA
    from .selection import kinematics_validity
    t0 = time.time()
    cfg = load_config(io_config)
    if python:
        cfg["physx"]["engine_python"] = python
    out = next_version(Path(out_root) if out_root else Path(run_root) / "dynamics" / "g9" / "io_identification")
    git = _git()
    status = dict(schema="forceartgs.g9.io_identification_status/v1", scene=Path(scene).name, out_dir=str(out),
                  git=git, owner="ForceArtGS (estimator side; no PArtTwin1D import)")
    import yaml
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(_clean(dict(
        io=cfg, io_config=str(io_config) if io_config else None, run=str(run_root), scene=str(scene),
        robot_archive=robot_archive, contact_segments=contact_segments,
        pulse_rows=list(pulse_rows) if pulse_rows else None, declared_joint_type=declared_joint_type,
        declaration_note=declaration_note, git=git)), sort_keys=False))
    ctx = SA.prepare_run(run_root, scene, ff_config=cfg["free_forced_config"], robot_archive=robot_archive,
                         contact_segments=contact_segments, declared_joint_type=declared_joint_type,
                         declaration_note=declaration_note, log=log)
    inp = ctx["inp"]
    kp = inp.kinematics_provenance or {}
    hashes = dict(input_hashes=inp.input_hashes, files={k: dict(path=str(v), sha256=_sha(v)) for k, v in dict(
        io_config=io_config, dynamics_config=ctx["cfg_path"], free_forced_config=cfg["free_forced_config"],
        robot_archive=robot_archive, contact_segments=contact_segments,
        joint_authority=kp.get("authority_path") or kp.get("path")).items() if v})

    def finish(st, reason, **extra):
        status.update(status=st, reason=reason, runtime_s=time.time() - t0, **extra)
        _write(out / "source_hashes.json", hashes)
        _write(out / "STATUS.json", status)
        log(f"[io] {Path(scene).name}: {st} ({reason})")
        return status
    kin = kinematics_validity(inp.kinematic_status, inp.kinematics_export_ready, cfg["joint_source"])
    if ctx["status"] != "OK":
        dims = dict(INPUT_VALIDITY="UNKNOWN", KINEMATICS_VALIDITY=kin, STRUCTURAL_IDENTIFIABILITY="NOT_TESTABLE",
                    PRACTICAL_IDENTIFIABILITY="NOT_TESTABLE", MODEL_ADEQUACY="NOT_RUN",
                    INDEPENDENT_VALIDATION="NOT_AVAILABLE", SIMULATOR_READINESS="BLOCKED")
        (out / "summary.md").write_text(f"# IO identification -- {Path(scene).name}\n\n**{ctx['status']}**: "
                                        f"{ctx['reason']}\n\nNot tested (no episode invented).\n")
        return finish(ctx["status"], ctx["reason"], status_dimensions=dims)
    if kin == "UNACCEPTED" and cfg["joint_source"] != "diagnostic_working_authority":
        return finish("KINEMATICS_NOT_ACCEPTED", f"joint status {inp.kinematic_status}; set joint_source: "
                      "diagnostic_working_authority to run explicitly conditional diagnostics")
    rec = ctx["rec"]
    hashes["record_sha256"] = rec.sha256()
    jt = rec.joint_type
    fc = cfg["fit"]
    eps, g10, excluded, regime = SA.from_segments(ctx["segments"], force_audit_status=ctx["force_audit_status"],
                                                   v0_prefix_samples=fc["v0_prefix_samples"],
                                                   min_samples=fc["min_samples"], heldout=cfg["heldout"],
                                                   ev=ctx["ev"])
    _write(out / "regime_evidence.json", dict(interval_summary=ctx["ev"].get("summary"),
                                              force_audit=ctx["force_audit_status"],
                                              segments=[s.describe() for s in ctx["segments"]],
                                              admissible=[e.name for e in eps], excluded=excluded))
    mm = _mass_model(inp, cfg["mass_mode"])
    gm = gravity_hypotheses(inp, mm, jt)
    prior = _prior_for_sim(inp, mm) if mm is not None else (
        dict(status="PRIOR_FOR_SIM", I_eff_range=[2.0, 20.0], basis="moving mass 2-20 kg (assumed)")
        if jt == "prismatic" else dict(status="NOT_AVAILABLE"))
    gm["prior_M_range"] = prior.get("I_eff_range")
    gm["prior_provenance"] = f"ASSUMED ({prior.get('basis', 'no basis')})"
    pulse = None
    if pulse_rows:
        from ..validation.impulse_balance import audit
        try:
            pulse = audit(rec, pulse_rows, observability=(inp.record_context or {}).get("observability"),
                          robot_record=(inp.record_context or {})["rec"])
        except Exception as e:                                   # reported, never silently skipped
            pulse = dict(decision=dict(primary="AUDIT_FAILED", reasons=[f"{type(e).__name__}: {e}"]))
        pulse["rows"] = list(pulse_rows)
    accepted = dict(joint_type=jt, kinematic_status=inp.kinematic_status,
                    export_ready=bool(inp.kinematics_export_ready), axis=np.asarray(inp.axis).tolist(),
                    origin=None if inp.origin is None else np.asarray(inp.origin).tolist(),
                    axis_std_rad=inp.axis_std_rad, origin_std_m=inp.origin_std_m, joint_source=cfg["joint_source"],
                    declaration_note=declaration_note, provenance=kp)
    cc = dict(excluded=excluded, force_audit_status=ctx["force_audit_status"], pulse_audit=pulse,
              source="forceartgs_g9.free_forced.segments.classify + physx_refinement.g9_adapter")
    res = api.identify_io_response(eps, accepted, cc, gm, cfg, out, log=log)
    rec_px, physx_q = None, None
    if res["status"] == "OK" and cfg["physx"]["reconcile"] and not no_physx and inp.gravity is not None:
        urdf = out / "model_prior_reference.urdf"
        info = reference_urdf(inp, rec, mm, urdf)
        if info is not None:
            _write(out / "urdf_info.json", info)
            hashes["files"]["urdf"] = dict(path=str(urdf), sha256=_sha(urdf))
            from ..physx_refinement.sapien_diagnostic import IsolatedSapienCpu
            try:
                sim = IsolatedSapienCpu(urdf, moving_link=info["moving_link"], joint_name=info["joint_name"],
                                        dt=0.001, gravity=list(map(float, inp.gravity.g_world)),
                                        friction_mode="external_effort_surrogate",
                                        python=cfg["physx"]["engine_python"])
                try:
                    rec_px, physx_q = api.reconcile_physx(res, sim, eps, axis_inertia_unit=sim.axis["axis_inertia"],
                                                          cfg=cfg, log=log)
                    rec_px["n_rollouts"] = sim.n_rollouts
                finally:
                    sim.close()
            except Exception as e:                               # reported in the reconciliation file
                rec_px = dict(status="FAILED", reason=f"{type(e).__name__}: {e}")
        else:
            rec_px = dict(status="NOT_RUN", reason="revolute joint without validated mass geometry (no URDF)")
        _write(out / "physx_reconciliation.json", rec_px)
        if physx_q:
            api.write_replays(out, eps, res["_oe"], physx_q)
    blockers = []
    if pulse:
        pr = (pulse.get("decision") or {}).get("primary")
        if pr != "IMPULSE_CONSISTENT":
            blockers.append(f"declared pulse rows {pulse_rows[0]}-{pulse_rows[1]}: {pr} -- BLOCKER for any "
                            "absolute-inertia claim (no torque / displacement sign flip)")
    if res["status"] == "OK" and (res.get("gravity_check") or {}).get("status") == "CONTRADICTED":
        blockers.append(f"GRAVITY_GEOMETRY_CONTRADICTED: {res['gravity_check']['reason']}")
    po = (res.get("physical_mapping") or {}).get("prior_overlap") or {}
    if po.get("status") == "PRIOR_DATA_CONFLICT":
        blockers.append(f"PRIOR_DATA_CONFLICT: assumed M range {po['prior_M_range']} outside the data 95 % set "
                        f"{po['data_ci95_M']}")
    if res["status"] == "OK" and not any(not e.train for e in eps):
        blockers.append("NO_INDEPENDENT_EPISODE: INDEPENDENT_VALIDATION = NOT_AVAILABLE")
    if kin != "ACCEPTED":
        blockers.append(f"kinematics {kin} ({inp.kinematic_status}): every result is conditional")
    res["blockers"] = blockers
    if res["status"] == "OK":
        import json
        io = json.loads((out / "io_response.json").read_text())
        io["blockers"] = blockers
        (out / "io_response.json").write_text(json.dumps(io, indent=1))
    try:
        plots(out / "plots", eps, res, rec_px, physx_q)
    except Exception as e:                                       # plotting never hides the numbers
        log(f"[io] plotting failed: {type(e).__name__}: {e}")
    (out / "summary.md").write_text(summary_md(Path(scene).name, jt, res, rec_px, excluded, blockers))
    dims = {k: v for k, v in res["status_dimensions"].items() if not k.startswith("_")}
    return finish(res["status"], f"readiness {dims['SIMULATOR_READINESS']}", status_dimensions=dims,
                  n_train=sum(e.train for e in eps), n_heldout=sum(not e.train for e in eps),
                  n_excluded=len(excluded), blockers=blockers,
                  selected=(res.get("selection") or {}).get("selected"),
                  physx_reconciliation=(rec_px or {}).get("status"), simulation_export_allowed=False)


# ===================================================================================================== reporting
def plots(d, eps, res, rec_px, physx_q):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d.mkdir(parents=True, exist_ok=True)
    if res.get("status") != "OK":
        return
    oe = res["_oe"]
    from .api import replay_arrays
    tr, ho = replay_arrays(eps, oe, physx_q)
    arr = {**tr, **ho}
    fig, ax = plt.subplots(len(eps), 1, figsize=(8, 3 * len(eps)), squeeze=False)
    for a, e in zip(ax[:, 0], eps):
        a.errorbar(e.t, e.q, yerr=e.q_sigma, fmt=".", ms=3, color="k", label="observed q")
        a.plot(e.t, arr[f"{e.name}/q_analytic"], "C0-", label=f"analytical {oe['variant']} (output error)")
        if f"{e.name}/q_physx" in arr:
            a.plot(e.t, arr[f"{e.name}/q_physx"], "C3--", label="PhysX (first candidate)")
        a.set_title(f"{e.name} ({e.mode}, {'train' if e.train else 'held-out'})", fontsize=9)
        a.legend(fontsize=7)
    ax[-1, 0].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(d / "replay.png", dpi=120)
    plt.close(fig)
    sp = res.get("scale_profile") or {}
    if sp.get("points"):
        fig, a = plt.subplots(figsize=(6, 4))
        a.semilogx([p["M"] for p in sp["points"]], sp["delta_cost"], "o-")
        a.axhline(sp["threshold"], color="r", ls="--", label="95 % threshold")
        a.set_xlabel(f"M [{res['units']['M']}]")
        a.set_ylabel("profile delta cost")
        a.set_title(f"absolute-scale profile: {sp['status']}", fontsize=9)
        a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(d / "scale_profile.png", dpi=120)
        plt.close(fig)


def _g(x, n=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{n}g}"
    return str(x)


def summary_md(scene, jt, res, rec_px, excluded, blockers):
    L = [f"# Input-output identification -- {scene}", "", f"status **{res['status']}**", ""]
    dims = res.get("status_dimensions") or {}
    L += ["| dimension | status | note |", "|---|---|---|"]
    notes = dims.get("_notes") or {}
    for k, v in dims.items():
        if not k.startswith("_"):
            L.append(f"| {k} | **{v}** | {notes.get(k) or ''} |")
    L += ["", "## Windows", "", "| name | mode | train | n | t | q range |", "|---|---|---|---|---|---|"]
    for n, e in (res.get("episodes") or {}).items():
        L.append(f"| {n} | {e['mode']} | {e['train']} | {e['n']} | {_g(e['t'][0])}-{_g(e['t'][1])} | "
                 f"{_g(e['q_range'])} {res['units']['q']} |")
    L += ["", f"Excluded intervals: {len(excluded)}"] + [f"- `{x['name']}` **{x.get('code')}** ({x.get('label')}): "
                                                       f"{x.get('reason')}" for x in excluded]
    if res["status"] != "OK":
        L += ["", res.get("reason", "")]
    else:
        sel = res["selection"]
        L += ["", "## Model selection (same windows)", "", f"selected **{sel['selected']}** -- {sel['reason']}", "",
              "| variant | status | cost | NLL | chi2_red | k | adequate |", "|---|---|---|---|---|---|---|"]
        vs = res.get("variants") or {}
        for r in sel["table"]:
            L.append(f"| {r['variant']} | {r['status']} | {_g(r.get('cost'))} | "
                     f"{_g((vs.get(r['variant']) or {}).get('nll'))} | {_g(r.get('chi2_red'))} | "
                     f"{_g(r.get('n_params'))} | {_g(r.get('adequate_train'))} |")
        L += ["", "Replay of the selected fit (training windows):", ""] + [
            f"- {n}: rmse {_g(e.get('rmse'))} {res['units']['q']} = {_g(e.get('rmse_over_range'))} x the observed "
            f"range; rmse/sigma {_g(e.get('rmse_over_sigma'))} (measurement-only "
            f"{_g(e.get('rmse_over_sigma_q_measurement'))})"
            for n, e in ((res.get("selected_fit") or {}).get("episodes") or {}).items() if e.get("train")]
        L += ["", "## Response features (estimate_not_for_sim)", "", "| feature | estimate | units | status | "
              "interval / note |", "|---|---|---|---|---|"]
        for f in res["features"]:
            u = f.get("uncertainty") or {}
            note = u.get("ci95") or u.get("reason") or ""
            if f.get("estimate_real") is not None:
                est = f"{_g(f['estimate_real'])}{f['estimate_imag']:+.3g}j"
            elif f["name"].startswith("poles"):
                est = ", ".join(f"{_g(p['re'])}{p['im']:+.3g}j" for p in (u.get("p1"), u.get("p2")) if p) or "n/a"
                note = u.get("stability") or u.get("reason")
            else:
                est = _g(f.get("estimate"))
            L.append(f"| {f['name']} | {est} | {f['units']} | {f['status']} | {note} |")
        sp = res["scale_profile"]
        L += ["", f"## Absolute scale: profile {sp['status']}", "", sp["note"],
              f"- best M (not for sim) {_g(sp['M_best_not_for_sim'])} {res['units']['M']} "
              f"(at grid edge {sp['M_best_at_grid_edge']}); 95 % set {sp.get('ci95_M')}"]
        mp = res["physical_mapping"]
        L += ["", f"## Physical mapping: {mp['status']}", "", mp.get("reason") or mp.get("provenance") or ""]
        po = mp.get("prior_overlap")
        if po:
            L.append(f"- prior vs data: **{po['status']}** (prior {[_g(x) for x in po['prior_M_range']]}, data 95 % "
                     f"set {[_g(x) for x in po['data_ci95_M']]}, open low {po['open_low']} / high {po['open_high']})")
        ea = res["work_energy_audit_summary"]
        L += ["", "## Work / impulse audit (audit only)", ""] + [f"- {k}: **{v}**" for k, v in ea["verdicts"].items()]
        mc = res["method_comparison"]["comparison"]
        L += ["", "## Method comparison (not averaged)", ""] + [
            f"- {p['a']} vs {p['b']} [{p['parameter']}]: {_g(p['a_est'])} vs {_g(p['b_est'])} -> {p['verdict']}"
            for p in mc["pairs"]]
        alt = res.get("gravity_alternatives") or {}
        if alt:
            L += ["", "## Gravity / geometry alternatives (separately labelled)", ""] + [
                f"- {k}: cost {_g(v.get('cost'))}, NLL {_g(v.get('nll'))} (primary {_g(res['selected_fit'].get('nll'))})"
                f", adequate {v.get('adequate_train')}, rates "
                f"{ {kk: _g(vv) for kk, vv in (v.get('estimate') or {}).items()} }" for k, v in alt.items()]
            gc = res.get("gravity_check") or {}
            L.append(f"- gravity check: **{gc.get('status')}** {gc.get('reason') or ''}")
    if rec_px:
        L += ["", f"## PhysX reconciliation: {rec_px.get('status')}", ""]
        cap = rec_px.get("engine_capabilities") or {}
        if cap:
            L.append(f"- engine {cap.get('sapien_version')}: {cap.get('coverage_label')}")
        for lab, c in (rec_px.get("candidates") or {}).items():
            for n, r in c["episodes"].items():
                if r["status"] == "OK":
                    L.append(f"- {lab} / {n}: parity rmse/sigma {_g(r['parity_over_sigma'])} "
                             f"(ok {r['parity_ok']}); PhysX vs observed rmse/sigma "
                             f"{_g(r['physx_vs_observed_over_sigma'])}")
                else:
                    L.append(f"- {lab} / {n}: {r['status']} {r.get('error', '')}")
        if rec_px.get("reason"):
            L.append(f"- {rec_px['reason']}")
    L += ["", "## Blockers", ""] + [f"- {b}" for b in blockers] + [
        "", "No simulator `value` field is emitted; point estimates are `estimate_not_for_sim`."]
    return "\n".join(L) + "\n"
