"""G10-A behavioral replay outputs: the opt-in ``behavioral_simulator_candidate.json`` (a SIMULATOR EXPERIMENT asset,
not a certified physics identification), the near-optimal family, plots and the summary.

Claims are separate flags and are never implied by each other: a good replay of the one demonstration is
BEHAVIORAL_FIT_TO_DEMONSTRATION only -- never PHYSICALLY_IDENTIFIED, PRODUCTION_READY or SIM_TO_REAL_VALIDATED.
The certified G9 / io contracts (``simulation_export_allowed = False``, no ``value`` field) are untouched.
"""
from __future__ import annotations

import numpy as np

from .behavioral_contracts import CANDIDATE_SCHEMA, START_FULL_RECORDING

FORBIDDEN_LABELS = ("PRODUCTION_READY", "PHYSICALLY_IDENTIFIED", "SIM_TO_REAL_VALIDATED", "VERIFIED_REAL_PHYSICS")


def units(jt):
    return dict(q="rad" if jt == "revolute" else "m", I="kg m^2" if jt == "revolute" else "kg",
                c="N m s/rad" if jt == "revolute" else "N s/m", T="N m" if jt == "revolute" else "N")


def classify(fit, cfg, *, input_verified, conditional) -> dict:
    m = fit["best"]["metrics"]
    adequate = m["rmse_over_sigma"] <= float(cfg["loss"]["adequacy_sigma"])
    status = "BEHAVIORAL_CANDIDATE_UNVALIDATED" if adequate else "BEHAVIORAL_FIT_INADEQUATE"
    replay = ("DEMO_REPLAY_INADEQUATE" if not adequate else
              "GOOD_DEMO_REPLAY" if (input_verified and not conditional) else
              "GOOD_DEMO_REPLAY_UNDER_CONDITIONAL_INPUT")
    return dict(status=status, replay_label=replay, adequate=bool(adequate),
                rule=f"adequate: rmse / sqrt(sigma_q^2 + sigma_num^2) <= {cfg['loss']['adequacy_sigma']} (a small "
                     "range-relative error alone is not adequacy)")


def candidate(*, scene, run, ep, fit, family, cls, info, sim_settings, hashes, conditional, input_sens, work,
              policy, alternative, capabilities, backends, candidate_urdf=None, cfg) -> dict:
    p = fit["best"]["params"]
    u = units(ep.joint_type)
    rho = p["rho_scale"]
    m = fit["best"]["metrics"]
    base = fit["baselines"].get("prior_reference") or {}
    preview = bool(cls["adequate"] and cfg["output"]["allow_preview_without_unique_physical_id"])
    out = dict(
        schema=CANDIDATE_SCHEMA, owner="ForceArtGS", scene=scene,
        claim="BEHAVIORAL_FIT_TO_DEMONSTRATION",
        semantics="a simulator experiment asset (best PhysX replay of ONE demonstration under its recorded input); "
                  "NOT a certified physics identification",
        source=dict(run=str(run), authority_sha256=(hashes.get("files", {}).get("joint_authority") or {}).get("sha256"),
                    robot_record_sha256=hashes.get("record_sha256"), input_hashes=hashes.get("input_hashes")),
        initial_state=dict(source_start_frame=int(ep.source_start_frame),
                           recording_start_frame=int(ep.recording_start_frame),
                           start_is_full_demo=bool(ep.start_is_full_demo), q0_from_visual=float(ep.q0_observed),
                           v0_fixed=ep.v0_fixed, v0_source=ep.start_basis,
                           engine_readback=fit["engine_initial_state"]),
        replay=dict(mode="recorded_generalized_effort", full_demo=bool(ep.start_basis == START_FULL_RECORDING),
                    continuous_rollout=True, resets=0,
                    force_semantics="CANONICAL_OBJECT_SIDE_CONDITIONAL", t=[float(ep.t0_s), float(ep.t_q_s[-1])],
                    n_observations=int(len(ep.t_q_s)), effort_rows=[int(ep.effort_source_row[0]),
                                                                   int(ep.effort_source_row[-1])]),
        candidate=dict(
            parameters=dict(rho_scale=rho, armature=p["armature"], dynamic_effort=p["dynamic_effort"],
                            static_effort=p["static_effort"], viscous=p["viscous"]),
            units=dict(rho_scale="relative to the reference URDF density", armature=u["I"],
                       dynamic_effort=u["T"], static_effort=u["T"], viscous=u["c"]),
            physical_equivalents=dict(moving_link_mass_kg=float(info["mass"]) * rho,
                                      axis_inertia=float(info["axis_inertia"]) * rho + p["armature"],
                                      axis_inertia_units=u["I"],
                                      inertia_com_q0=(np.asarray(info["inertia_com_q0"]) * rho).tolist(),
                                      com_q0=info["com_q0"], label="effective behavioral values, NOT measured"),
            bounds=fit["bounds"], at_bound=fit.get("at_bound"), bound_limited=family.get("bound_limited"),
            not_unique=bool(family.get("not_unique", True)),
            reference_urdf_sha256=info.get("sha256"), candidate_urdf=candidate_urdf,
            friction_realization="EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE (joint effort -c v - Td sat(v/v_eps) "
                                 "applied with set_qf; the URDF carries mass / inertia only)"),
        engine=sim_settings,
        quality=dict(training_rmse=m["rmse"], training_rmse_over_sigma=m["rmse_over_sigma"],
                     training_rmse_over_sigma_q_measurement=m["rmse_over_sigma_q_measurement"],
                     training_rmse_over_range=m["rmse_over_range"], final_error=m["final_error"],
                     q_range=m["q_range"], units=u["q"], baseline_rmse=(base.get("metrics") or {}).get("rmse"),
                     improvement=fit["improvement"], out_of_sample_rmse=None,
                     out_of_sample="NOT_EVALUATED (no independent real action / held-out demonstration)"),
        family=dict(n_members=family.get("n_members"), rule=family.get("rule"),
                    ranges={k: v.get("family_range") for k, v in (family.get("profiles") or {}).items()
                            if "family_range" in v},
                    members=[dict(label=x.get("label"), parameters=x["params"],
                                  rmse_over_sigma=x["metrics"]["rmse_over_sigma"]) for x in family.get("members", [])]),
        input_sensitivity=dict(status=input_sens.get("status"),
                               max_delta_rmse_over_sigma=input_sens.get("max_delta_rmse_over_sigma")),
        work_audit=dict(verdict=work.get("verdict"), energy_supplied=work.get("energy_supplied"),
                        energy_supplied_sd=work.get("energy_supplied_sd")),
        policy_probe=dict(status=policy.get("status"), final_q_spread=policy.get("final_q_spread"),
                          max_q_spread=policy.get("max_q_spread")),
        alternative_hypotheses={"zero_gravity_load": dict(status=alternative.get("status"), label=alternative.get(
            "label"), rmse_over_sigma=((alternative.get("best") or {}).get("metrics") or {}).get("rmse_over_sigma"))}
        if alternative else {},
        status=dict(behavioral_status=cls["status"], replay_label=cls["replay_label"],
                    behavioral_preview_allowed=preview,
                    physical_parameters_uniquely_identified=False,
                    independent_real_prediction_validated=False, validated_on_new_real_action=False,
                    native_target_physx_friction_verified=False, native_engine_target_ready=False,
                    sim_to_real_validated=False, simulation_export_allowed=False,
                    conditional_on_input=bool(conditional["input"]), not_unique=bool(family.get("not_unique", True)),
                    certified_export_note="the certified G9 export gate is unchanged (simulation_export_allowed = "
                                          "False); this file is a separate opt-in preview"),
        conditional_assumptions=conditional["list"],
        engine_capabilities=capabilities, g10b=backends.get("g10b_status"))
    assert_claims_honest(out)
    return out


def assert_claims_honest(c):
    """No production / identification / transfer claim from a single replay; preview only when adequate."""
    s = c["status"]
    for k in ("physical_parameters_uniquely_identified", "independent_real_prediction_validated",
              "native_target_physx_friction_verified", "sim_to_real_validated", "simulation_export_allowed",
              "native_engine_target_ready", "validated_on_new_real_action"):
        if s[k]:
            raise AssertionError(f"single-demonstration replay cannot set {k}")
    blob = str({k: v for k, v in s.items() if k != "certified_export_note"})
    for lab in FORBIDDEN_LABELS:
        if lab in blob:
            raise AssertionError(f"forbidden label {lab} in the behavioral candidate")
    if s["behavioral_preview_allowed"] and s["behavioral_status"] != "BEHAVIORAL_CANDIDATE_UNVALIDATED":
        raise AssertionError("preview only for an adequate behavioral fit")


# --------------------------------------------------------------------------------------------------- plots
def plots(d, ep, fit, family, policy, rollout_best, alternative_q=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .objective import PhysicalParameters
    d.mkdir(parents=True, exist_ok=True)
    u = units(ep.joint_type)
    t = ep.t_q_s
    res = fit["_result"]
    q_best = np.asarray(res.q)
    pe = ep.to_physx_episode()
    # 1 q replay
    fig, a = plt.subplots(figsize=(9, 4))
    a.errorbar(t, ep.q_observed, yerr=ep.q_sigma, fmt=".", ms=3, color="k", label="observed q (joint authority)")
    a.plot(t, q_best, "C0-", lw=1.6, label="PhysX best (q0 observed, v0 = 0)")
    base = fit["baselines"].get("prior_reference")
    if base and "params" in base:
        a.plot(t, rollout_best(PhysicalParameters(**base["params"]), pe).q, "C3:", label="prior reference baseline")
    for k, mbr in enumerate([x for x in family.get("members", []) if x.get("label") != "best"][:6]):
        a.plot(t, rollout_best(PhysicalParameters(**mbr["params"]), pe).q, "--", lw=0.7, color="C2",
               label="near-optimal family" if k == 0 else None)
    if alternative_q is not None:
        a.plot(t, alternative_q, "C4-.", lw=1, label="zero-gravity alternative (labelled, not selected)")
    a.axvline(ep.t0_s, color="0.6", lw=0.8)
    a.set_xlabel("t [s]")
    a.set_ylabel(f"q [{u['q']}]")
    a.set_title(f"full-demo replay from rest (one continuous rollout), {ep.start_basis}", fontsize=8)
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "q_replay.png", dpi=120)
    plt.close(fig)
    # 2 effort + contact labels + cumulative impulses
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    s = ep.effort_sigma if ep.effort_sigma is not None else np.zeros_like(ep.effort_object)
    a.step(ep.effort_t_s, ep.effort_object, where="post", color="C1", lw=1, label="recorded Q_object (50 Hz rows)")
    a.fill_between(ep.effort_t_s, ep.effort_object - s, ep.effort_object + s, step="post", color="C1", alpha=0.2)
    col = dict(CONTACT_CONFIRMED="C2", MEASURED_NEAR_ZERO_CONTACT="C8", CONTACT_UNKNOWN="C7",
               RELEASE_CONFIRMED="C0", NO_CONTACT_MEASURED="C9")
    for iv in ep.contact_evidence.get("intervals", []):
        a.axvspan(iv["t"][0], iv["t"][1] + 0.02, color=col.get(iv["label"], "C5"), alpha=0.12, lw=0)
    a.set_ylabel(f"Q [{u['T']}]")
    a.set_title("recorded input (all rows applied; shading = contact evidence, never a zero substitution)",
                fontsize=8)
    a.legend(fontsize=7)
    dg = res.diagnostics or {}
    if dg.get("external_impulse") is not None:
        b.plot(t, dg["external_impulse"], label="cumulative applied impulse (engine)")
        b.plot(t, dg["friction_impulse"], label="cumulative friction-surrogate impulse")
    b.set_xlabel("t [s]")
    b.set_ylabel(f"impulse [{u['T']} s]")
    b.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "effort.png", dpi=120)
    plt.close(fig)
    # 3 velocity
    fig, a = plt.subplots(figsize=(9, 3.5))
    if res.qdot is not None:
        a.plot(t, res.qdot, "C0-", label="PhysX qdot (best)")
    tm = 0.5 * (t[1:] + t[:-1])
    vd = np.diff(ep.q_observed) / np.diff(t)
    vs = np.hypot(ep.q_sigma[1:], ep.q_sigma[:-1]) / np.diff(t)
    a.errorbar(tm, vd, yerr=vs, fmt=".", ms=3, color="k", label="observed finite difference (+-1 sd; diagnostic, "
                                                                 "not ground truth)")
    a.set_xlabel("t [s]")
    a.set_ylabel(f"qdot [{u['q']}/s]")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "velocity.png", dpi=120)
    plt.close(fig)
    # 4 residual
    fig, a = plt.subplots(figsize=(9, 3.5))
    sig = np.hypot(ep.q_sigma, fit["loss_settings"]["numeric_floor_q"])
    a.fill_between(t, -sig, sig, color="0.85", label="+-1 sigma (q + numerical floor)")
    a.fill_between(t, -3 * sig, 3 * sig, color="0.93", zorder=0)
    a.plot(t, q_best - ep.q_observed, "C0.-", ms=3, label="best - observed")
    if base and "params" in base:
        a.plot(t, rollout_best(PhysicalParameters(**base["params"]), pe).q - ep.q_observed, "C3:",
               label="prior reference - observed")
    a.set_xlabel("t [s]")
    a.set_ylabel(f"residual [{u['q']}]")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "displacement_residual.png", dpi=120)
    plt.close(fig)
    # 5 profiles
    prof = {k: v for k, v in (family.get("profiles") or {}).items() if "points" in v}
    if prof:
        fig, ax = plt.subplots(1, len(prof), figsize=(4 * len(prof), 3.4), squeeze=False)
        for a, (k, v) in zip(ax[0], prof.items()):
            pts = [p for p in v["points"] if "metrics" in p]
            if pts:
                a.plot([p["value"] for p in pts], [p["metrics"]["rmse_over_sigma"] for p in pts], "o-")
            a.axhline(fit["best"]["metrics"]["rmse_over_sigma"] + family["tolerance_sigma"], color="r", ls="--",
                      label="family tolerance (engineering)")
            a.axvline(v["best"], color="C0", lw=0.8)
            if k == "rho_scale":
                a.set_xscale("log")
            a.set_xlabel(k)
            a.set_ylabel("rmse/sigma (others re-optimized)")
            a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(d / "parameter_profiles.png", dpi=120)
        plt.close(fig)
    # 6 policy probe
    if policy.get("status") == "EVALUATED":
        fig, a = plt.subplots(figsize=(8, 3.5))
        for lab, qq in zip(policy["labels"], policy["q_members"]):
            a.plot(policy["t"], qq, lw=0.9, label=str(lab))
        a.set_title("near-equal demo fits under a NEW synthetic force profile (not a real demonstration)", fontsize=8)
        a.set_xlabel("t [s]")
        a.set_ylabel(f"q [{u['q']}]")
        a.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(d / "policy_probe.png", dpi=120)
        plt.close(fig)


def _g(x, n=4):
    return "n/a" if x is None else (f"{x:.{n}g}" if isinstance(x, (int, float)) else str(x))


def summary_md(scene, st, ep=None, fit=None, family=None, cand=None, input_sens=None, work=None, policy=None,
               alternative=None, evidence=None):
    L = [f"# G10-A behavioral PhysX replay -- {scene}", "", f"status **{st['status']}** -- {st.get('reason', '')}", ""]
    if evidence:
        L += ["## Start state and input coverage", "",
              f"- recording start frame {evidence.get('recording_start_frame')}, joint-authority start frame "
              f"{evidence.get('joint_authority_start_frame')} (t0 = {_g(evidence.get('t0_s'))} s); full demo start: "
              f"**{evidence.get('start_is_full_demo')}**"]
        rc = evidence.get("rest_check") or {}
        L.append(f"- rest test (H0: v = 0; never used as v0): {rc.get('status')} z = {_g(rc.get('z_v'))} "
                 f"({rc.get('n_samples')} samples in {rc.get('window_s')} s)")
        cov = evidence.get("input_coverage")
        if cov:
            L.append(f"- input coverage: **{cov['status']}** rows {cov['row_range']} {cov['missing'] or ''}")
    if ep is None or fit is None:
        return "\n".join(L) + "\n"
    u = units(ep.joint_type)
    m, s = fit["best"]["metrics"], cand["status"]
    p = fit["best"]["params"]
    L += [f"- q0 = {_g(ep.q0_observed, 6)} {u['q']} (authority), v0 = 0 exactly ({ep.start_basis}); engine "
          f"readback q0 {_g(fit['engine_initial_state']['q0_engine'], 6)}, v0 "
          f"{fit['engine_initial_state']['v0_engine']}",
          f"- contact evidence rows: {ep.contact_evidence['counts']} (recorded input applied on every row)", "",
          "## Behavioral candidate (simulator experiment asset)", "",
          f"- **{s['behavioral_status']}** / {s['replay_label']}; preview allowed **{s['behavioral_preview_allowed']}**;"
          f" not unique {s['not_unique']}; conditional on input {s['conditional_on_input']}",
          f"- rho_scale {_g(p['rho_scale'])} (moving mass {_g(cand['candidate']['physical_equivalents']['moving_link_mass_kg'])}"
          f" kg, axis inertia {_g(cand['candidate']['physical_equivalents']['axis_inertia'])} {u['I']}), viscous "
          f"{_g(p['viscous'])} {u['c']}, dynamic effort {_g(p['dynamic_effort'])} {u['T']}, armature {_g(p['armature'])}"
          ", static excess 0 (surrogate)",
          f"- training rmse {_g(m['rmse'])} {u['q']} = {_g(m['rmse_over_sigma'])} sigma (measurement-only "
          f"{_g(m['rmse_over_sigma_q_measurement'])}), {_g(m['rmse_over_range'])} x the range "
          f"{_g(m['q_range'])}; final error {_g(m['final_error'])}",
          f"- vs `{fit['improvement']['reference']}`: rmse {_g(fit['improvement']['reference_rmse'])} -> "
          f"{_g(fit['improvement']['best_rmse'])} (ratio {_g(fit['improvement']['rmse_ratio'])})",
          f"- out-of-sample: NOT_EVALUATED; native PhysX friction verified: False; sim-to-real validated: False", "",
          "## Baselines (same full-demo loss)", "", "| start | L | rmse | rmse/sigma |", "|---|---|---|---|"]
    for k, v in fit["baselines"].items():
        if "metrics" in v:
            L.append(f"| {k} | {_g(v['objective'])} | {_g(v['metrics']['rmse'])} | {_g(v['metrics']['rmse_over_sigma'])} |")
        else:
            L.append(f"| {k} | {v.get('status')} | {v.get('error', '')[:60]} | |")
    L += ["", f"## Near-optimal family ({family['n_members']} members)", "", family["rule"], ""]
    if family.get("bound_limited"):
        L.append(f"- **bound-limited** (best on a search bound, the optimum may lie outside): {family['bound_limited']} "
                 f"{ {k: v for k, v in fit.get('at_bound', {}).items() if v} }")
    for k, v in (family.get("profiles") or {}).items():
        if "family_range" in v:
            L.append(f"- {k}: best {_g(v['best'])}, family [{_g(v['family_range'][0])}, {_g(v['family_range'][1])}]"
                     f"{' (reaches the grid edge)' if v['reaches_grid_edge'] else ''}")
    L += ["", f"## Input hypotheses at the fitted theta: {input_sens['status']}", ""]
    for k, v in input_sens["hypotheses"].items():
        L.append(f"- {k} ({v['kind']}): rmse/sigma {_g(v.get('rmse_over_sigma'))}")
    L += ["", f"## Work audit: **{work['verdict']}**", "",
          f"- W_in {_g(work['W_in'])} +- {_g(work['W_in_sd'])} J, Delta U {_g(work.get('delta_U'))} J, energy supplied "
          f"{_g(work['energy_supplied'])} +- {_g(work['energy_supplied_sd'])} J ({work['rule']})"]
    if policy.get("status") == "EVALUATED":
        L += ["", "## New-input probe (synthetic, not a real demonstration)", "",
              f"- {policy['n_members']} near-equal demo fits: final q spread {_g(policy['final_q_spread'])} {u['q']}, "
              f"max spread {_g(policy['max_q_spread'])} -> randomize the family JOINTLY"]
    if alternative:
        b = (alternative.get("best") or {}).get("metrics") or {}
        L += ["", "## Zero-gravity alternative (separately labelled, not selected)", "",
              f"- {alternative.get('status')}: rmse/sigma {_g(b.get('rmse_over_sigma'))}, params "
              f"{ {k: _g(v) for k, v in ((alternative.get('best') or {}).get('params') or {}).items() if k != 'inertia_scale_source'} }"]
    L += ["", "## Conditional assumptions", ""] + [f"- {x}" for x in cand["conditional_assumptions"]]
    L += ["", "Behavioral fit to ONE demonstration under its recorded input (SAPIEN 3 PhysX CPU, explicit friction "
              "surrogate). Not a physical identification, not validated on a new real action, not native-PhysX parity."]
    return "\n".join(L) + "\n"


# =============================================================================== wrench-driven path (schema v2)
def classify_wrench(fit, cfg) -> dict:
    m = fit["best"]["metrics"]
    ok = m["rmse_over_sigma"] <= float(cfg["loss"]["adequacy_sigma"])
    return dict(status="BEHAVIORAL_FIT_OK" if ok else "BEHAVIORAL_FIT_INADEQUATE", adequate=bool(ok),
                rule=f"BEHAVIORAL_FIT_OK: rmse / sqrt(sigma_q^2 + sigma_num^2) <= {cfg['loss']['adequacy_sigma']} on "
                     "every valid visual sample of the one demonstration (predeclared)")


def pulse_comparison(fr, pulse_rows=None) -> dict:
    """Q_full (6D wrench) vs legacy TCP-force-only Q on declared rows (fridge 229-232) and over the whole record:
    per-row values, impulses, and the split of Q_full into its moment and force terms. Source numbers only."""
    a = np.asarray(fr.axis_B, float)
    if fr.joint_type == "revolute":
        q_mom = fr.tau_B @ a
        q_force = np.cross(fr.p_ref_B - np.asarray(fr.origin_B, float), fr.F_B) @ a
    else:
        q_mom, q_force = np.zeros(len(fr.Q_full)), fr.F_B @ a
    leg = fr.Q_legacy_tcp_force_only
    dt = 1.0 / 50.0

    def block(sel, label):
        d = dict(label=label, rows=[int(fr.source_row[sel][0]), int(fr.source_row[sel][-1])], n=int(sel.sum()),
                 Q_full=fr.Q_full[sel].tolist(), Q_full_moment_term=q_mom[sel].tolist(),
                 Q_full_force_term=q_force[sel].tolist(), impulse_full=float(np.sum(fr.Q_full[sel]) * dt),
                 impulse_moment_term=float(np.sum(q_mom[sel]) * dt))
        if leg is not None:
            ok = np.isfinite(leg[sel])
            d.update(Q_legacy_tcp_force_only=[None if not np.isfinite(x) else float(x) for x in leg[sel]],
                     impulse_legacy=float(np.sum(leg[sel][ok]) * dt) if ok.any() else None,
                     sign_agreement=(None if not ok.any() else
                                     float(np.mean(np.sign(leg[sel][ok]) == np.sign(fr.Q_full[sel][ok])))))
        if fr.Q_alt_tcp_moment_origin is not None:
            d["impulse_alt_tcp_moment_origin"] = float(np.sum(fr.Q_alt_tcp_moment_origin[sel]) * dt)
        return d
    out = dict(whole_record=block(np.ones(len(fr.Q_full), bool), "whole record"),
               note="no sign correction is applied or claimed; the moment term is the part a TCP-force-only model "
                    "cannot see")
    if pulse_rows:
        sel = (fr.source_row >= int(pulse_rows[0])) & (fr.source_row <= int(pulse_rows[1]))
        out["declared_pulse"] = block(sel, f"declared pulse rows {pulse_rows[0]}-{pulse_rows[1]}") if sel.any() else \
            dict(status="ROWS_NOT_IN_RECORD")
    return out


def wrench_candidate(*, scene, run, ep, fr, fit, family, cls, info, readback, hashes, input_sens, work, policy,
                     alternative, pulse, provenance, candidate_urdf=None) -> dict:
    from .behavioral_contracts import WRENCH_CANDIDATE_SCHEMA
    p = fit["best"]["params"]
    u = units(ep.joint_type)
    rho = p["rho_scale"]
    m = fit["best"]["metrics"]
    base = fit["baselines"].get("prior_reference") or {}
    ok = cls["status"] == "BEHAVIORAL_FIT_OK"
    unique = not bool(family.get("not_unique", True))
    c = fr.contract
    out = dict(
        schema=WRENCH_CANDIDATE_SCHEMA, owner="ForceArtGS", scene=scene, claim="DEMO_REPLAY_FIT",
        semantics="effective PhysX simulator parameters that best replay ONE demonstration under its recorded "
                  "six-axis wrench; a behavioral-simulator preview, NOT certified physics, NOT tested on unseen actions",
        input_provenance=dict(
            experimental_facts=dict(known_sensor_to_global_transform=True, wrench_already_tared_at_start=True,
                                    contact_on_during_interaction=True, initial_joint_velocity=0.0),
            interaction=dict(segments=ep.contact_evidence.get("segments"),
                             source="existing force-peak + visual contact segments (read only)",
                             outside_interaction="zero object-side effort (contact off)"),
            wrench="full six-axis, as recorded (tared at the start; no additional tare)", order=c["wrench_order"],
            transform=c["known_transform"], transform_source=c["transform_source"],
            moment_reference=c["moment_reference"], object_side_multiplier=c["object_side_multiplier"],
            projection="Q_R = a.[tau_B + (p_ref - o) x F_B] | Q_P = a.F_B (no contact point)",
            noise=fr.noise, robot_archive_sha256=c["robot_archive_sha256"],
            full_wrench_record_sha256=fr.sha256()),
        source=dict(run=str(run), joint_authority=(hashes.get("files") or {}).get("joint_authority")),
        initial_state=dict(t0_s=float(ep.t0_s), visual_start_frame=int(ep.source_start_frame),
                           q0_from_visual=float(ep.q0_observed), v0_fixed=ep.v0_fixed,
                           initial_velocity_source="EXPERIMENT_KNOWN_ZERO", engine_readback=fit["engine_initial_state"]),
        replay=dict(mode="recorded_full_wrench_generalized_effort", continuous_rollout=True, resets=0,
                    contact="ON during the interaction segments, OFF (zero effort) outside",
                    t=[float(ep.t0_s), float(ep.t_q_s[-1])], n_visual_samples=int(len(ep.t_q_s)),
                    wrench_rows=[int(ep.effort_source_row[0]), int(ep.effort_source_row[-1])],
                    n_wrench_rows=int(len(ep.effort_t_s))),
        candidate=dict(
            parameters=dict(rho_scale=rho, armature=p["armature"], dynamic_effort=p["dynamic_effort"],
                            static_effort=p["static_effort"], viscous=p["viscous"]),
            units=dict(rho_scale="relative to the reference URDF density", armature=u["I"], dynamic_effort=u["T"],
                       static_effort=u["T"], viscous=u["c"]),
            physical_equivalents=dict(moving_link_mass_kg=float(info["mass"]) * rho,
                                      axis_inertia=float(info["axis_inertia"]) * rho + p["armature"],
                                      axis_inertia_units=u["I"],
                                      inertia_com_q0=(np.asarray(info["inertia_com_q0"]) * rho).tolist(),
                                      com_q0=info["com_q0"], label="effective behavioral values, NOT measured"),
            bounds=fit["bounds"], at_bound=fit.get("at_bound"), bound_limited=family.get("bound_limited"),
            static_friction="F_s = F_d (the SAPIEN surrogate has no stiction; static friction NOT identified)",
            reference_urdf_sha256=info.get("sha256"), candidate_urdf=candidate_urdf,
            model=(provenance or {}).get("model"),
            friction_realization=("NONE (frictionless_physx: no passive term; c = Td = 0 fixed, not fitted)"
                                  if (readback.get("engine") or {}).get("friction_mode") == "frictionless_physx" else
                                  "EXPLICIT_FRICTION_SURROGATE_NOT_NATIVE (joint effort -c v - Td sat(v/v_eps) "
                                  "applied with set_qf; the URDF carries mass / inertia only)" +
                                  ("; Coulomb friction OFF (Td = 0 fixed, not fitted)"
                                   if fit["bounds"]["dynamic_effort"][1] <= 0 else ""))),
        engine=dict(readback.get("engine") or {}, passive_terms_invariant=readback.get(
            "passive_terms_invariant_across_runs")),
        quality=dict(rmse=m["rmse"], units=u["q"], rmse_over_range=m["rmse_over_range"], q_range=m["q_range"],
                     rmse_over_sigma=m["rmse_over_sigma"],
                     rmse_over_sigma_q_measurement=m["rmse_over_sigma_q_measurement"], end_error=m["final_error"],
                     baseline=dict(label="prior_reference (rho_s = 1, frictionless)",
                                   rmse=(base.get("metrics") or {}).get("rmse"),
                                   rmse_over_sigma=(base.get("metrics") or {}).get("rmse_over_sigma")),
                     improvement=fit["improvement"], out_of_sample="NOT_EVALUATED (no independent real action)"),
        family=dict(n_members=family.get("n_members"), rule=family.get("rule"),
                    ranges={k: v.get("family_range") for k, v in (family.get("profiles") or {}).items()
                            if "family_range" in v},
                    members=[dict(label=x.get("label"), parameters=x["params"],
                                  rmse_over_sigma=x["metrics"]["rmse_over_sigma"]) for x in family.get("members", [])],
                    scored_by="actual PhysX replay of the SAME recorded wrench input (no free-motion scale gauge)"),
        diagnostics=dict(input_sensitivity=dict(status=input_sens.get("status"),
                                                max_delta_rmse_over_sigma=input_sens.get("max_delta_rmse_over_sigma")),
                         work_audit=dict(verdict=work.get("verdict"), energy_supplied=work.get("energy_supplied"),
                                         energy_supplied_sd=work.get("energy_supplied_sd")),
                         policy_probe=dict(status=policy.get("status"), final_q_spread=policy.get("final_q_spread")),
                         zero_gravity_alternative=(None if not alternative else dict(
                             status=alternative.get("status"), label=alternative.get("label"),
                             rmse_over_sigma=((alternative.get("best") or {}).get("metrics") or {}).get(
                                 "rmse_over_sigma"))),
                         full_vs_legacy_effort=dict(
                             whole_record_impulse_full=pulse["whole_record"]["impulse_full"],
                             whole_record_impulse_legacy=pulse["whole_record"].get("impulse_legacy"),
                             declared_pulse=pulse.get("declared_pulse"))),
        provenance=provenance,
        labels=dict(status=cls["status"], FIT_TO_DEMO=bool(ok),
                    PARAMETERS_UNIQUE=bool(unique and ok),
                    PARAMETERS_UNIQUE_scope="within the profiled engineering tolerance on THIS demonstration only",
                    NATIVE_TARGET_ENGINE_PARITY=False, TRANSFER_VALIDATED=False),
        status=dict(behavioral_status=cls["status"], behavioral_preview_allowed=bool(ok),
                    physical_parameters_uniquely_identified=False, independent_real_prediction_validated=False,
                    validated_on_new_real_action=False, native_target_physx_friction_verified=False,
                    native_engine_target_ready=False, sim_to_real_validated=False, simulation_export_allowed=False,
                    replay_label="DEMO_REPLAY_FIT" if ok else "DEMO_REPLAY_INADEQUATE",
                    certified_export_note="certified G9 export rules unchanged (simulation_export_allowed = False); "
                                          "this is a separate behavioral-simulator preview"))
    s = out["status"]
    for k in ("physical_parameters_uniquely_identified", "independent_real_prediction_validated", "sim_to_real_validated",
              "simulation_export_allowed", "native_target_physx_friction_verified", "native_engine_target_ready",
              "validated_on_new_real_action"):
        if s[k]:
            raise AssertionError(f"single-demonstration replay cannot set {k}")
    if out["labels"]["TRANSFER_VALIDATED"] or out["labels"]["NATIVE_TARGET_ENGINE_PARITY"]:
        raise AssertionError("no transfer / native-parity claim from one demonstration")
    return out


def wrench_plots(d, ep, fr, fit, family, rollout, res_base, pulse_rows=None, alternative_q=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .objective import PhysicalParameters
    d.mkdir(parents=True, exist_ok=True)
    u = units(ep.joint_type)
    t = ep.t_q_s
    q_best = np.asarray(fit["_result"].q)
    sl = (fr.t_s >= ep.effort_t_s[0] - 1e-9) & (fr.t_s <= ep.effort_t_s[-1] + 1e-9)
    # 1 wrench + joint effort
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for i, n in enumerate(("Fx", "Fy", "Fz")):
        ax[0].plot(fr.t_s[sl], fr.W_recorded[sl, i], lw=0.9, label=f"{n} (sensor axes, as recorded)")
    ax[0].set_ylabel("force [N]")
    ax[0].legend(fontsize=7, ncol=3)
    for i, n in enumerate(("Tx", "Ty", "Tz")):
        ax[1].plot(fr.t_s[sl], fr.W_recorded[sl, 3 + i], lw=0.9, label=f"{n}")
    ax[1].set_ylabel("moment [N m]")
    ax[1].legend(fontsize=7, ncol=3)
    ax[2].step(fr.t_s[sl], fr.Q_full[sl], where="post", color="C0", lw=0.6, alpha=0.5, label="Q_full (6D wrench)")
    ax[2].step(ep.effort_t_s, ep.effort_object, where="post", color="C0", label="replayed Q (contact off -> 0)")
    ax[2].fill_between(fr.t_s[sl], fr.Q_full[sl] - fr.Q_sigma[sl], fr.Q_full[sl] + fr.Q_sigma[sl], step="post",
                       color="C0", alpha=0.2)
    if fr.Q_legacy_tcp_force_only is not None:
        ax[2].step(fr.t_s[sl], fr.Q_legacy_tcp_force_only[sl], where="post", color="C3", lw=0.8,
                   label="legacy Q (TCP force only)")
    ax[2].set_ylabel(f"Q [{u['T']}]")
    ax[2].set_xlabel("t [s] (source row / 50)")
    ax[2].legend(fontsize=7)
    ax[0].set_title("recorded tared wrench; the replay applies it inside the interaction segments only", fontsize=8)
    for a_, b_ in ep.contact_evidence.get("segments") or []:
        for axx in ax:
            axx.axvspan(a_ / 50.0, b_ / 50.0, color="C2", alpha=0.08, lw=0)
    fig.tight_layout()
    fig.savefig(d / "wrench_and_joint_effort.png", dpi=120)
    plt.close(fig)
    # 2 q real vs sim
    fig, a = plt.subplots(figsize=(9, 4))
    a.errorbar(t, ep.q_observed, yerr=ep.q_sigma, fmt=".", ms=3, color="k", label="visual q (3DGS joint authority)")
    a.plot(t, q_best, "C0-", lw=1.6, label="PhysX best (q0 visual, v0 = 0)")
    a.plot(t, res_base.q, "C3:", label="prior reference baseline")
    for k, mbr in enumerate([x for x in family.get("members", []) if x.get("label") != "best"][:6]):
        a.plot(t, rollout(PhysicalParameters(**mbr["params"]), ep.to_physx_episode()).q, "--", lw=0.7, color="C2",
               label="near-optimal family" if k == 0 else None)
    if alternative_q is not None:
        a.plot(t, alternative_q, "C4-.", lw=1, label="zero-gravity alternative (labelled, not selected)")
    a.set_xlabel("t [s]")
    a.set_ylabel(f"q [{u['q']}]")
    a.set_title("one uninterrupted PhysX rollout under the recorded six-axis wrench", fontsize=8)
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "q_real_vs_sim.png", dpi=120)
    plt.close(fig)
    # 3 residual
    fig, a = plt.subplots(figsize=(9, 3.5))
    sig = np.hypot(ep.q_sigma, fit["loss_settings"]["numeric_floor_q"])
    a.fill_between(t, -3 * sig, 3 * sig, color="0.92", label="+-3 sigma")
    a.fill_between(t, -sig, sig, color="0.82", label="+-1 sigma")
    a.plot(t, q_best - ep.q_observed, "C0.-", ms=3, label="best - visual")
    a.plot(t, np.asarray(res_base.q) - ep.q_observed, "C3:", label="baseline - visual")
    a.set_xlabel("t [s]")
    a.set_ylabel(f"residual [{u['q']}]")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "residual_over_time.png", dpi=120)
    plt.close(fig)
    # 4 force-only vs full wrench (declared pulse rows, else the replay)
    fig, a = plt.subplots(figsize=(8, 3.5))
    if pulse_rows:
        ps = (fr.source_row >= int(pulse_rows[0]) - 10) & (fr.source_row <= int(pulse_rows[1]) + 10)
        a.axvspan(int(pulse_rows[0]), int(pulse_rows[1]), color="C1", alpha=0.15, label="declared pulse rows")
    else:
        ps = sl
    a.step(fr.source_row[ps], fr.Q_full[ps], where="post", color="C0", label="Q_full (6D)")
    if fr.Q_legacy_tcp_force_only is not None:
        a.step(fr.source_row[ps], fr.Q_legacy_tcp_force_only[ps], where="post", color="C3", label="legacy TCP force only")
    if fr.Q_alt_tcp_moment_origin is not None:
        a.step(fr.source_row[ps], fr.Q_alt_tcp_moment_origin[ps], where="post", color="C7", lw=0.7,
               label="diagnostic: moment read about the TCP")
    a.axhline(0, color="0.5", lw=0.6)
    a.set_xlabel("source row")
    a.set_ylabel(f"Q [{u['T']}]")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "force_only_vs_full_wrench_pulse.png", dpi=120)
    plt.close(fig)
    # 5 profiles
    prof = {k: v for k, v in (family.get("profiles") or {}).items() if "points" in v}
    if prof:
        fig, ax = plt.subplots(1, len(prof), figsize=(4 * len(prof), 3.4), squeeze=False)
        for a, (k, v) in zip(ax[0], prof.items()):
            pts = [p for p in v["points"] if "metrics" in p]
            if pts:
                a.plot([p["value"] for p in pts], [p["metrics"]["rmse_over_sigma"] for p in pts], "o-")
            a.axhline(fit["best"]["metrics"]["rmse_over_sigma"] + family["tolerance_sigma"], color="r", ls="--",
                      label="family tolerance (engineering)")
            a.axvline(v["best"], color="C0", lw=0.8)
            if k == "rho_scale":
                a.set_xscale("log")
            a.set_xlabel(k)
            a.set_ylabel("rmse/sigma (others re-optimized)")
            a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(d / "parameter_profiles.png", dpi=120)
        plt.close(fig)


def wrench_summary_md(scene, st, *, evidence=None, fr=None, ep=None, fit=None, family=None, cand=None,
                      input_sens=None, work=None, policy=None, alternative=None, pulse=None):
    L = [f"# G10 wrench-driven behavioral PhysX calibration -- {scene}", "",
         f"status **{st['status']}** -- {st.get('reason', '')}", "",
         "Experimental facts (given): known sensor-to-global transform, wrench tared at the start, contact ON during the "
         "interaction (existing force-peak + visual segments) and OFF outside, qdot = 0 at the interaction start; "
         "q(t0) = visual q(t0).", ""]
    if fr is not None:
        c = fr.contract
        L += ["## Input", "",
              f"- wrench {c['wrench_order']} in {c['wrench_frame']}; transform {c['known_transform']}; moment about "
              f"{c['moment_reference']}; object-side multiplier {c['object_side_multiplier']:+.0f} (applied once)",
              f"- noise: force {fr.noise['force_sigma_source']}, moment {fr.noise['moment_sigma_source']}"]
        mc = (fr.diagnostics or {}).get("moment_reference_consistency")
        if mc:
            L.append(f"- moment-reference consistency (diagnostic): tau_x slope {_g(mc['tau_x']['slope_vs_rtcp_x_F'])} "
                     f"corr {_g(mc['tau_x']['corr'])}, tau_y slope {_g(mc['tau_y']['slope_vs_rtcp_x_F'])} corr "
                     f"{_g(mc['tau_y']['corr'])} vs (r_tcp x F) on {mc['rows']} rows")
    if evidence:
        cov = evidence.get("input_coverage") or {}
        L += [f"- interaction segments (existing force-peak + visual): {evidence.get('interaction_segments')}; "
              f"{evidence.get('rows_in_contact')} wrench rows in contact, {evidence.get('rows_contact_off_zero_effort')} "
              "contact-off rows at zero effort",
              f"- t0 = {_g(evidence.get('t0_s'))} s (visual frame {evidence.get('visual_start_frame')}); "
              f"{evidence.get('n_visual_samples')} visual samples ({evidence.get('post_interaction_coast_samples')} "
              f"post-release coast, {evidence.get('pre_interaction_visual_samples_not_replayed')} pre-contact samples "
              f"not replayed); wrench coverage **{cov.get('status')}** rows {cov.get('row_range')} "
              f"{cov.get('missing') or ''}"]
    if pulse:
        w = pulse["whole_record"]
        L += ["", "## Full wrench vs legacy TCP force only", "",
              f"- whole record impulse: full {_g(w['impulse_full'])}, legacy {_g(w.get('impulse_legacy'))}, moment term "
              f"{_g(w['impulse_moment_term'])}; sign agreement {_g(w.get('sign_agreement'))}"]
        dp = pulse.get("declared_pulse")
        if dp and "Q_full" in dp:
            L.append(f"- {dp['label']}: Q_full {[_g(x, 3) for x in dp['Q_full']]}, legacy "
                     f"{[_g(x, 3) for x in (dp.get('Q_legacy_tcp_force_only') or [])]}, moment term "
                     f"{[_g(x, 3) for x in dp['Q_full_moment_term']]}; impulse full {_g(dp['impulse_full'])} vs legacy "
                     f"{_g(dp.get('impulse_legacy'))} (no sign correction applied or claimed)")
    if fit is None:
        return "\n".join(L) + "\n"
    u = units(ep.joint_type)
    m, p, lab = fit["best"]["metrics"], fit["best"]["params"], cand["labels"]
    pe = cand["candidate"]["physical_equivalents"]
    L += ["", "## Best behavioral simulator candidate", "",
          f"- model: {cand['candidate'].get('model') or (cand.get('engine') or {}).get('model')}",
          f"- **{lab['status']}**; FIT_TO_DEMO {lab['FIT_TO_DEMO']}, PARAMETERS_UNIQUE {lab['PARAMETERS_UNIQUE']}, "
          f"NATIVE_TARGET_ENGINE_PARITY False, TRANSFER_VALIDATED False; preview "
          f"{cand['status']['behavioral_preview_allowed']}",
          f"- rho_scale {_g(p['rho_scale'])} (moving mass {_g(pe['moving_link_mass_kg'])} kg, axis inertia "
          f"{_g(pe['axis_inertia'])} {u['I']}), viscous {_g(p['viscous'])} {u['c']}, dynamic effort "
          f"{_g(p['dynamic_effort'])} {u['T']} (= static, surrogate), armature {_g(p['armature'])}",
          f"- rmse {_g(m['rmse'])} {u['q']} = {_g(m['rmse_over_range'])} x range {_g(m['q_range'])}; "
          f"{_g(m['rmse_over_sigma'])} sigma (measurement-only {_g(m['rmse_over_sigma_q_measurement'])}); end error "
          f"{_g(m['final_error'])}",
          f"- vs prior reference: rmse {_g(fit['improvement']['reference_rmse'])} -> {_g(fit['improvement']['best_rmse'])}",
          f"- q0 {_g(ep.q0_observed, 6)} (visual), v0 0 (EXPERIMENT_KNOWN_ZERO); engine readback q0 "
          f"{_g(fit['engine_initial_state']['q0_engine'], 6)}, v0 {fit['engine_initial_state']['v0_engine']}", "",
          "## Baselines", "", "| start | rmse | rmse/sigma |", "|---|---|---|"]
    for k, v in fit["baselines"].items():
        L.append(f"| {k} | {_g((v.get('metrics') or {}).get('rmse'))} | "
                 f"{_g((v.get('metrics') or {}).get('rmse_over_sigma'))} |")
    L += ["", f"## Near-optimal family ({family['n_members']} members, actual PhysX replays)", "", family["rule"]]
    if family.get("bound_limited"):
        L.append(f"- bound-limited: {family['bound_limited']} {fit.get('at_bound')}")
    for k, v in (family.get("profiles") or {}).items():
        if "family_range" in v:
            L.append(f"- {k}: best {_g(v['best'])}, family [{_g(v['family_range'][0])}, {_g(v['family_range'][1])}]")
    L += ["", f"## Diagnostics of the same fit", "",
          f"- input sensitivity: {input_sens['status']} (max delta rmse/sigma {_g(input_sens.get('max_delta_rmse_over_sigma'))})",
          f"- work audit: {work['verdict']} (W_in {_g(work['W_in'])} J, Delta U {_g(work.get('delta_U'))} J)"]
    if policy.get("status") == "EVALUATED":
        L.append(f"- new-input probe (synthetic): final q spread {_g(policy['final_q_spread'])} {u['q']} over "
                 f"{policy['n_members']} near-equal fits")
    if alternative:
        b = (alternative.get("best") or {}).get("metrics") or {}
        L.append(f"- zero-gravity alternative (labelled, not selected): rmse/sigma {_g(b.get('rmse_over_sigma'))}")
    eng = cand.get("engine") or {}
    L += ["", f"Engine: SAPIEN {eng.get('sapien_version')} {eng.get('pipeline', 'PhysX')}, "
              f"{eng.get('runtime_semantics')}. A behavioral fit to one demonstration; not unique laboratory "
              "parameters, not validated on unseen actions."]
    return "\n".join(L) + "\n"
