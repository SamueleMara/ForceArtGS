"""Iterative multimodal pre-3DGS frontend (multimodal spec section 7): RGB-D tracks + TCP contact modes + verified force.

    round 0  block folds (frozen) -> candidate-independent support + TCP tool occupancy / contact seeds
             -> R and P fitted on training blocks (existing EM) -> refined with contact (+ verified force) rows
    round k  A track recovery (NCC, both predictions) B support update (+ sticking-anchor seeds) C/D/E refit + contact/force
             F/G identical held-out set for both candidates H accept (bounded improvement, validity) or roll back
    final    v2 physical decision on the accepted state (contact/force evidence consumed), all-frame refit for the
             authority, discovery-format export so ``authority_from_discovery`` consumes it unchanged.

No Gaussian is trained here. Held-out times never enter fitting, track recovery, anchors, mu or kappa.
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import assignments as asg
from .contact_modes import (ContactRows, MovingSurface, contact_discrimination, load_contact_segments,
                            segment_diagnostics)
from .exports import candidate_to_json, write_joint_states, write_json
from .geometry_fit import _select_fit_tracks, fit_geometry
from .model_selection import calibrated_noise, fit_joint_hypotheses
from .motion_field import PointTable
from .observations import depth_sigma
from .selection_v2 import declare_partition, run_selection_v2
from .support_consistency import candidate_independent_support
from .tcp_support import (CLEAR, ToolModel, anchor_seeds, classify_tracks_tool, contact_seeds, sigma_track,
                          update_support, write_overlays)
from .visual_losses import camera_stacks, whitened_residuals
from .wrench_physics import (CombinedRows, ForceDirectionRows, force_alignment_diagnostics, force_gate,
                             kinetic_direction_evidence, mechanical_model_evidence, static_contact_feasibility)


def recovered_sigma_scale(obs, recovered, cands, calibs, noise, lo=1.0, hi=10.0):
    """Pixel-sigma multiplier of recovered observations = robust residual scale of recovered / raw observations under
    the current fits, minimum over hypotheses (symmetric: neither candidate can down-weight inconvenient matches)."""
    if not recovered.any():
        return 1.0, dict(status="NONE")
    K, T_CW = camera_stacks(obs, calibs)
    ratios = {}
    for k, c in cands.items():
        ok = np.isfinite(c.canonical_anchors_w[obs.track]).all(1) & obs.uv_valid
        res = whitened_residuals(c.kind, c.axis_w, c.origin_w, c.q, np.nan_to_num(c.canonical_anchors_w), obs, K, T_CW,
                                 noise, obs_mask=ok)
        r = np.linalg.norm(res["r_uv"], axis=1)
        a, b = r[ok & recovered], r[ok & ~recovered]
        if len(a) >= 10 and len(b) >= 10:
            ratios[k] = float(np.median(a) / max(np.median(b), 1e-9))
    if not ratios:
        return 3.0, dict(status="DEFAULT", scale=3.0, reason="too few recovered/raw residuals; conservative default")
    sc = float(np.clip(min(ratios.values()), lo, hi))
    return sc, dict(status="MEASURED", scale=sc, per_hypothesis_ratio=ratios)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _clean(obs, support):
    """Training observations without TOOL tracks (indices preserved; tool tracks simply have no observations)."""
    tool = support.get("TOOL")
    return obs if tool is None else obs.select_obs(~tool[obs.track])


def refine_with_hook(fit, hook, cfg, max_nfev=None):
    """Refit every candidate's axis/hinge/q/anchors with extra contact/force rows (same parameterisation)."""
    if hook is None or hook.n_rows == 0:
        return fit, {}
    new, info = {}, {}
    prob = fit.problem
    for k, c in fit.candidates.items():
        post = np.nan_to_num(c.moving_probability)
        tracks = _select_fit_tracks(prob, post, np.random.default_rng(cfg.fit.seed))
        a, o, q, X, q_unc, obs_t, inf = fit_geometry(k, c.axis_w, c.origin_w, c.q, np.nan_to_num(c.canonical_anchors_w),
                                                    tracks, post, prob, max_nfev=max_nfev, extra=hook)
        X = np.where(np.isfinite(c.canonical_anchors_w), X, np.nan)
        nc = dataclasses.replace(c, axis_w=a, origin_w=o, q=q, q_uncertainty=q_unc, canonical_anchors_w=X,
                                 diagnostics=dict(c.diagnostics, multimodal_refinement=inf))
        nc._ll_moving = getattr(c, "_ll_moving", None)
        new[k] = nc
        info[k] = dict(axis_change_deg=float(np.degrees(np.arccos(min(1, abs(np.dot(a, c.axis_w)))))),
                       q_span_change=float(np.ptp(q) - np.ptp(c.q)), n_extra_rows=int(hook.n_rows),
                       cost=inf.get("cost"))
    out = copy.copy(fit)
    out.candidates = new
    return out, info


def heldout_nll(cands, noise, test_obs, calibs, track_set, use_depth=True) -> dict:
    """Per-hypothesis held-out NLL on an IDENTICAL observation set (moving prediction, fixed noise)."""
    K, T_CW = camera_stacks(test_obs, calibs)
    ok = track_set[test_obs.track].copy()
    for c in cands:
        ok &= np.isfinite(c[1].canonical_anchors_w[test_obs.track]).all(1)
    sz = depth_sigma(np.nan_to_num(test_obs.z, nan=1.0), noise, test_obs.depth_edge, test_obs.depth_cos)
    out = {}
    for name, c in cands:
        res = whitened_residuals(c.kind, c.axis_w, c.origin_w, c.q, np.nan_to_num(c.canonical_anchors_w), test_obs, K,
                                 T_CW, noise, obs_mask=ok, use_depth=use_depth)
        out[name] = float(-asg.observation_loglik(res, noise.sigma_px, sz, noise)[ok].sum())
    out["n_obs"] = int(ok.sum())
    return out


class Frontend:
    def __init__(self, obs, calibs, cfg, out_dir, *, scene=None, images=None, rec=None, segments_source=None,
                 wrench_calibration=None, legacy=None, source_files=None, log=print, robot_record=None,
                 force_policy="certified"):
        from .frontend_robot_record import POLICIES
        if force_policy not in POLICIES:
            raise ValueError(f"force_policy {force_policy!r} not in {POLICIES}")
        self.robot_record, self.force_policy, self.force_conditional = robot_record, force_policy, False
        self.obs, self.calibs, self.cfg, self.scene = obs, calibs, cfg, scene
        self.mf = cfg.multimodal_frontend
        self.out = Path(out_dir)
        self.images, self.rec, self.cal = images, rec, wrench_calibration
        self.segments_source = segments_source or self.mf.contact.intervals_file
        self.legacy, self.source_files, self.log = legacy, source_files or {}, log
        self.T = obs.frame_map.T

    # ------------------------------------------------------------------ setup
    def setup(self):
        out = self.out
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty {out}")
        out.mkdir(parents=True)
        import yaml
        (out / "config_resolved.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(self.cfg.to_dict(), default=str)),
                                                                 sort_keys=True))
        self.part, self.acf = declare_partition(self.obs, self.cfg)
        held = np.asarray(self.part.heldout_times, int)
        self.train_mask = ~np.isin(np.arange(self.T), held)
        self.train_obs = self.obs.select_obs(~np.isin(self.obs.time, held))
        self.test_obs = self.obs.select_obs(np.isin(self.obs.time, held))
        self.ref = self.part.reference_time_index
        self.noise0, _ = calibrated_noise(self.train_obs, self.cfg)
        # robot / contact / force
        r = self.rec
        self.robot_ok = r is not None and self.mf.contact.enabled
        self.segments, seg_rep = [], dict(status="NO_ROBOT_DATA")
        self.contact_state = np.full(self.T, "no_contact", dtype=object)
        if self.robot_ok:
            ok = np.isfinite(r.p_tcp_w).all(1)
            self.contact_state[ok] = "contact_unknown"           # TCP present: proposals only, never residuals
            if self.segments_source:
                self.segments, seg_rep = load_contact_segments(self.segments_source, self.obs.frame_map, r.p_tcp_w,
                                                               self.mf.contact)
                for s in self.segments:
                    for t in s.model_time_indices:
                        self.contact_state[t] = s.state if s.confirmed else ("contact_unknown" if s.state != "no_contact"
                                                                             else "no_contact")
            else:
                seg_rep = dict(status="NO_ANNOTATION", reason="no contact-segment annotation: every TCP frame is "
                                                             "contact_unknown (low-confidence proposals only)")
        write_json(out / "contact_mode_segments.json", seg_rep)
        write_json(out / "contact_mode_validation.json", dict(
            n_segments=len(self.segments), n_active=sum(s.active_for_residuals for s in self.segments),
            state_counts={k: int((self.contact_state == k).sum()) for k in set(self.contact_state.tolist())},
            rules="confirmed + independent provenance -> residuals; unknown -> proposals; no silent mode fallback"))
        self.tool = ToolModel.from_scene(self.scene, self.mf.tcp_support) if self.robot_ok else None
        self.f_obj, self.fgate = None, dict(status="ABSTAIN", reason="no robot data")
        if self.robot_ok and self.mf.force.enabled and r.wrench_sensor is not None:
            self.fgate = force_gate(self.cal, r.wrench_sensor)
            if self.fgate["status"] == "PASS":
                self.f_obj = self.cal.force_on_object_world(r.wrench_sensor[:, :3], r.R_WS)["f_object_w"]
        (out / "robot").mkdir(exist_ok=True)
        self.record_alignment = None
        if self.robot_record is not None:
            # the canonical robot record (shared with G9): exact row join, same TCP rows, force converted once
            from .frontend_robot_record import align_record_to_frame_map
            al = align_record_to_frame_map(self.robot_record, self.obs.frame_map,
                                           p_tcp_w=r.p_tcp_w if self.robot_ok else None,
                                           T_WB=r.T_WB if self.robot_ok else None)
            self.record_alignment = al
            write_json(out / "robot" / "canonical_record_alignment.json",
                       {k: v for k, v in al.items() if k not in ("f_object_w", "p_tcp_w", "contact_state",
                                                                 "in_contact_segment", "recorded_timestamp_s")}
                       | dict(force_policy=self.force_policy))
            if self.force_policy == "exploratory" and self.robot_ok and self.mf.force.enabled and self.f_obj is None:
                self.f_obj = al["f_object_w"]
                self.force_conditional = True
                self.fgate = dict(self.fgate, status="EXPLORATORY_CONDITIONAL", certified_gate=self.fgate.get("status"),
                                  reason="canonical-record force used as CONDITIONAL evidence (declared sign / axes / "
                                         "TCP); outputs are DIAGNOSTIC, never physics-export ready")
        write_json(out / "robot" / "force_calibration_gates.json", self.fgate)
        if self.robot_ok:
            np.savez(out / "robot" / "tcp_trajectory_world.npz", p_tcp_w=r.p_tcp_w, v_tcp_w=r.v_tcp_w,
                     p_ee_w=r.p_ee_w if r.p_ee_w is not None else np.full_like(r.p_tcp_w, np.nan),
                     timestamp_s=r.timestamp_s, T_WB=r.T_WB, T_WB_source=np.array(r.T_WB_source))
        fp = dict(source_files=self.source_files, config_hash=self.cfg.config_hash(),
                  frame_map=_sha(self.obs.frame_map.to_json()),
                  contact_annotation_sha256=seg_rep.get("annotation_sha256"),
                  calibration={c: self.calibs[c].calib_sha for c in self.obs.camera_ids},
                  tool=None if self.tool is None else self.tool.to_json(),
                  wrench_calibration=None if self.cal is None else _sha(self.cal.to_json()))
        if self.robot_record is not None:
            fp["canonical_robot_record"] = self.record_alignment["fingerprint"]
            fp["force_policy"] = self.force_policy
        write_json(out / "source_fingerprints.json", fp)
        self.fingerprints = fp

    # ------------------------------------------------------------------ one round
    def support_for(self, obs_tr, cands=None):
        noise = self.noise0
        tab = PointTable(obs_tr, noise, self.calibs)
        base = candidate_independent_support(obs_tr, noise)
        if not self.robot_ok or not self.mf.tcp_support.enabled:
            base["TOOL"] = np.zeros(obs_tr.n_tracks, bool)
            return base, tab, dict(status="NO_ROBOT_DATA")
        r = self.rec
        tool = classify_tracks_tool(tab, r.p_tcp_w, r.p_ee_w, self.tool, np.flatnonzero(self.train_mask))
        seeds = contact_seeds(tab, r.p_tcp_w, np.where(self.train_mask, self.contact_state, "no_contact"), tool["label"],
                              self.mf.tcp_support)
        hits = None
        if cands:
            hits = np.zeros(obs_tr.n_tracks, bool)
            for s in self.segments:
                if s.active_for_residuals and s.slip_mode != "sliding":
                    times = [t for t in s.model_time_indices if self.train_mask[t]]
                    for c in cands.values():
                        from .contact_modes import stick_anchor
                        a = stick_anchor(c.kind, c.axis_w, c.origin_w, c.q, r.p_tcp_w, times)
                        hits |= anchor_seeds(tab, c.kind, c.axis_w, c.origin_w, c.q, a, times,
                                             self.mf.tcp_support.seed_radius_m, tool["label"])
        sup = update_support(base, tool, seeds, sigma_track(tab), self.mf.tcp_support, anchor_hits=hits)
        return sup, tab, dict(tool=dict(counts=tool["counts"], tool_model=tool["tool"]),
                              seeds={k: v for k, v in seeds.items() if k not in ("positive", "proposal", "per_time")},
                              anchor_seed_tracks=None if hits is None else int(hits.sum()),
                              provenance=sup["tcp_provenance"])

    def contact_hook(self, fit, support, train_tab):
        if not self.robot_ok or not any(s.active_for_residuals for s in self.segments):
            return None, None, dict(status="NO_CONFIRMED_CONTACT")
        r = self.rec
        M = support["M_proposal"]
        pts = np.array([train_tab.X[t, M] for t in range(max(self.ref - 1, 0), min(self.ref + 2, self.T))]).reshape(-1, 3)
        cam_c = [self.calibs[c].T_WC[:3, 3] for c in self.obs.camera_ids]
        surf = MovingSurface.from_points(pts, cam_c, self.mf.contact.surface_patch_radius_m,
                                         self.mf.contact.surface_min_points)
        common = np.ones(self.T, bool)
        from .contact_modes import canonical_path
        for c in fit.candidates.values():
            if surf.status == "OK":
                _, sup = surf.signed_distance(canonical_path(c.kind, c.axis_w, c.origin_w, c.q, np.nan_to_num(r.p_tcp_w),
                                                             np.arange(self.T)))
                common &= sup
        rows = ContactRows(self.segments, r.p_tcp_w, self.train_mask, surf, common)
        frows = None
        if self.f_obj is not None and self.mf.force.mechanical_model != "none":
            ft = np.array([t for s in self.segments if s.active_for_residuals for t in s.model_time_indices
                           if self.train_mask[t] and np.linalg.norm(self.f_obj[t]) > self.mf.force.min_force_n], int)
            if len(ft):
                frows = ForceDirectionRows(self.f_obj, r.p_tcp_w, r.timestamp_s, ft, self.mf.force.sigma_direction_deg)
        return CombinedRows(rows, frows), surf, dict(status="OK", n_contact_rows=rows.n_rows,
                                                      n_force_rows=0 if frows is None else frows.n_rows,
                                                      surface=surf.status, common_supported=int(common.sum()))

    def fit_round(self, obs_tr, support, tab):
        fit = fit_joint_hypotheses(_clean(obs_tr, support), self.calibs, self.cfg, ref=self.ref, log=None)
        hook, surf, hinfo = self.contact_hook(fit, support, tab)
        fit_c, deltas = refine_with_hook(fit, hook, self.cfg)
        # a declared mode contradicted by EVERY candidate must not drag the joint: drop those segments' rows, keep the
        # raw residuals and the CONTACT_MODE_CONTRADICTED record (no silent switch to another contact mode)
        failed = self._failed_segments(fit_c, surf)
        if failed:
            kept = self.segments
            self.segments = [s for s in kept if s.segment_id not in failed]
            hook2, _, hinfo2 = self.contact_hook(fit, support, tab)
            self.segments = kept
            fit_c, deltas = refine_with_hook(fit, hook2, self.cfg)
            hinfo = dict(hinfo2, excluded_contradicted_segments=sorted(failed),
                         note="rows of segments contradicted by every candidate were removed from the fit; their "
                              "residuals are kept in the contact diagnostics")
        return fit, fit_c, surf, dict(hook=hinfo, tcp_force_parameter_deltas=deltas)

    def _failed_segments(self, fit, surf):
        diag = self.contact_diag(fit, surf)
        if not diag:
            return set()
        bad = [{d["segment_id"] for d in ds if d.get("contradicted")} for ds in diag.values()]
        return set.intersection(*bad) if bad else set()

    def contact_diag(self, fit, surf):
        if not self.robot_ok:
            return {}
        out = {}
        for k, c in fit.candidates.items():
            out[k] = [segment_diagnostics(c.kind, c.axis_w, c.origin_w, c.q, self.rec.p_tcp_w, s, self.train_mask, surf,
                                          self.mf.contact.contradiction_chi2_per_dof, heldout_mask=~self.train_mask)
                      for s in self.segments]
        return out

    # ------------------------------------------------------------------ main loop
    def run(self):
        t0 = time.time()
        self.setup()
        log = self.log
        obs_tr = self.train_obs
        support, tab, tinfo = self.support_for(obs_tr)
        fit_v, fit, surf, finfo = self.fit_round(obs_tr, support, tab)
        hold = support["M_proposal"].copy()
        state = dict(round=0, obs_tr=obs_tr, recovered=np.zeros(obs_tr.n_obs, bool), support=support, fit=fit,
                     fit_visual=fit_v, surf=surf, tcp=tinfo, fitinfo=finfo, contact=self.contact_diag(fit, surf))
        metrics = []
        state["heldout"] = heldout_nll([(k, c) for k, c in fit.candidates.items()], self.noise0, self.test_obs,
                                       self.calibs, hold)
        metrics.append(self._save_round(state, "accepted (initial)"))
        log(f"[mm] round 0: support {support['counts']} held-out NLL {state['heldout']}")
        for k in range(1, self.mf.max_iterations + 1):
            new = self._propose(state, k)
            if new is None:
                log(f"[mm] round {k}: converged (no new evidence)")
                break
            common = heldout_nll([("prev_" + kk, c) for kk, c in state["fit"].candidates.items()] +
                                 [("new_" + kk, c) for kk, c in new["fit"].candidates.items()],
                                 self.noise0, self.test_obs, self.calibs, hold)
            ok, why = self._accept(state, new, common)
            new["heldout"] = {kk: common["new_" + kk] for kk in new["fit"].candidates} | dict(n_obs=common["n_obs"])
            metrics.append(self._save_round(new, ("accepted: " if ok else "rolled back: ") + why, common))
            log(f"[mm] round {k}: {'ACCEPT' if ok else 'ROLLBACK'} {why}")
            if not ok:
                break
            state = new
        self.state, self.metrics = state, metrics
        sel = self._finalize(state)
        sel["runtime_s"] = round(time.time() - t0, 1)
        write_json(self.out / "accepted" / "validation_report.json", dict(
            rounds=metrics, accepted_round=state["round"], selection=dict((k, sel.get(k)) for k in (
                "physical_joint_status", "working_motion_model", "physics_joint_export_ready", "failure_codes")),
            no_gaussian_training="the frontend never trains or renders Gaussians", runtime_s=sel["runtime_s"]))
        self._report(sel, metrics)
        return sel

    def _propose(self, state, k):
        recovered_obs, rec_mask, rrep = state["obs_tr"], state["recovered"], dict(accepted=0, status="SKIPPED")
        if self.images is not None and self.mf.track_refinement:
            from .track_refinement import recover_tracks
            raw = state["obs_tr"].select_obs(~state["recovered"])
            new_obs, newrec, rrep = recover_tracks(state["obs_tr"], self.images, self.calibs, state["fit"].candidates,
                                                   state["support"]["M_proposal"], np.flatnonzero(self.train_mask),
                                                   ncc_min=self.mf.ncc_min, max_recover=self.mf.max_recover_per_round)
            del raw
            rec_mask = np.r_[state["recovered"], newrec[state["obs_tr"].n_obs:]]
            mc = rrep.get("matcher_calibration", {})
            if mc.get("status") == "OK":
                scale = float(np.clip(mc["sigma_per_axis_px"] / self.noise0.sigma_px, 1.0, 10.0))
                srep = dict(status="MATCHER_CALIBRATED", scale=scale, matcher=mc,
                            note="independent: re-finding raw tracker observations with the recovery procedure")
            else:
                scale, srep = 3.0, dict(status="DEFAULT", scale=3.0, reason=f"matcher calibration {mc.get('status')}")
            new_obs.uv_sigma_scale = np.where(rec_mask, scale, 1.0)
            rrep["sigma_scale"] = srep
            recovered_obs = new_obs
        support, tab, tinfo = self.support_for(recovered_obs, state["fit"].candidates)
        changed = (rrep.get("accepted", 0) > 0 or
                   not np.array_equal(support["M_proposal"], state["support"]["M_proposal"]))
        if not changed:
            return None
        fit_v, fit, surf, finfo = self.fit_round(recovered_obs, support, tab)
        return dict(round=k, obs_tr=recovered_obs, recovered=rec_mask, support=support, fit=fit, fit_visual=fit_v,
                    surf=surf, tcp=tinfo, fitinfo=finfo, recovery=rrep, contact=self.contact_diag(fit, surf))

    def _accept(self, prev, new, common):
        v = self.mf.validation
        kinds = list(new["fit"].candidates)
        deltas = {kk: common["new_" + kk] - common["prev_" + kk] for kk in kinds}
        rel = {kk: deltas[kk] / max(abs(common["prev_" + kk]), 1e-9) for kk in kinds}
        # candidate-agnostic criterion: the BEST explanation of the identical held-out observations must improve.
        # A worse fit of the other hypothesis with better data is expected (it is what separates them), not a defect.
        best_prev = min(common["prev_" + kk] for kk in kinds)
        best_new = min(common["new_" + kk] for kk in kinds)
        rel_best = (best_new - best_prev) / max(abs(best_prev), 1e-9)
        if rel_best > -v.min_relative_improvement:
            return False, f"best held-out NLL did not improve meaningfully ({rel_best:+.4f}); per hypothesis {rel}"
        nm_prev, nm_new = prev["support"]["M_proposal"].sum(), new["support"]["M_proposal"].sum()
        if nm_new < (1 - v.max_support_loss_fraction) * nm_prev:
            return False, f"moving support dropped {nm_prev} -> {nm_new}"
        newly = [d["segment_id"] for ds in new["contact"].values() for d in ds if d.get("contradicted")]
        old = [d["segment_id"] for ds in prev["contact"].values() for d in ds if d.get("contradicted")]
        if set(newly) - set(old):
            return False, f"new contact contradictions {sorted(set(newly) - set(old))}"
        return True, (f"best held-out NLL {rel_best:+.4f} on {common['n_obs']} identical observations; per hypothesis "
                      f"{rel}")

    def _save_round(self, st, verdict, common=None):
        d = self.out / "iterations" / f"round_{st['round']:03d}"
        d.mkdir(parents=True, exist_ok=True)
        for k, c in st["fit"].candidates.items():
            write_json(d / k / "candidate.json", dict(axis=c.axis_w, origin=c.origin_w, q=c.q,
                                                      n_moving=int(np.nansum(c.moving_probability > 0.5))))
        np.savez_compressed(d / "support.npz", **{k: st["support"][k] for k in ("S_core", "M_proposal", "U", "TOOL")},
                            recovered=st["recovered"])
        m = dict(round=st["round"], verdict=verdict, support=st["support"]["counts"], heldout_nll=st.get("heldout"),
                 identical_set_comparison=common, tcp=st["tcp"], fit=st["fitinfo"], recovery=st.get("recovery"),
                 contact={k: [dict((kk, d.get(kk)) for kk in ("segment_id", "mode", "chi2_per_dof", "contradicted",
                                                              "canonical_drift_max_m", "heldout")) for d in ds]
                          for k, ds in st["contact"].items()},
                 candidate_sha={k: _sha(dict(a=c.axis_w, o=c.origin_w, q=c.q)) for k, c in st["fit"].candidates.items()})
        write_json(d / "metrics.json", m)
        return m

    # ------------------------------------------------------------------ final
    def _robot_evidence(self, st):
        r = self.rec
        if not self.robot_ok:
            return dict(contact=dict(status="ABSTAIN", reason="no robot data", used_in_fit=False),
                        force=dict(status="ABSTAIN", reason="no robot data", used_in_fit=False))
        diag = st["contact"]
        disc = contact_discrimination(diag) if any(s.active_for_residuals for s in self.segments) else \
            dict(status="ABSTAIN", reason="no independently confirmed contact segment")
        contradicted = sorted({d["segment_id"] for ds in diag.values() for d in ds if d.get("contradicted")})
        # the contact assumption fails only if NO candidate explains a confirmed segment (one hypothesis being
        # contradicted is discrimination, not a failed assumption)
        per_kind_bad = {k: {d["segment_id"] for d in ds if d.get("contradicted")} for k, ds in diag.items()}
        failed = sorted(set.intersection(*per_kind_bad.values())) if per_kind_bad else []
        stick_active = [s for s in self.segments if s.active_for_residuals and s.slip_mode != "sliding"]
        p_diag = {d["segment_id"]: d for d in diag.get("prismatic", [])}
        prismatic_consistent_stick = bool(stick_active) and all(
            p_diag.get(s.segment_id, {}).get("status") == "OK" and not p_diag[s.segment_id].get("contradicted")
            and p_diag[s.segment_id].get("n_train", 0) >= 3 for s in stick_active)
        contact = dict(status="USED" if (st["fitinfo"]["hook"].get("n_contact_rows") or 0) > 0 else "PROPOSALS_ONLY",
                       prismatic_consistent_stick=prismatic_consistent_stick,
                       used_in_fit=bool(st["fitinfo"]["hook"].get("n_contact_rows")), discrimination=disc,
                       contradicted_segments={k: sorted(v) for k, v in per_kind_bad.items()},
                       assumption_satisfied=None if not any(s.active_for_residuals for s in self.segments)
                       else not failed, assumption_failed_segments=failed,
                       unverified_scale=any(not s.scale_verified for s in self.segments if s.active_for_residuals),
                       tcp_support=st["tcp"], parameter_deltas=st["fitinfo"]["tcp_force_parameter_deltas"],
                       contact_mode_assumption=sorted({s.slip_mode for s in self.segments if s.active_for_residuals}),
                       contact_mode_evidence="independent operator/video annotation" if self.segments else "none")
        cands = st["fit"].candidates
        force = dict(gate=self.fgate, used_in_fit=bool(st["fitinfo"]["hook"].get("n_force_rows")))
        if self.f_obj is None:
            force["status"] = "ABSTAIN"
            force["reason"] = self.fgate.get("reason")
            if r.wrench_sensor is not None:      # signed diagnostics on the raw channel, never decisive
                raw = np.asarray(r.wrench_sensor[:, :3], float)
                force["raw_alignment_diagnostic_uncalibrated"] = {
                    k: v for k, v in force_alignment_diagnostics(raw, r.p_tcp_w, r.v_tcp_w, r.timestamp_s,
                                                                 cands).items() if k != "per_frame"}
        else:
            act = np.isin(self.contact_state, ["contact_confirmed"])
            force.update(status="OK",
                         alignment={k: v for k, v in force_alignment_diagnostics(
                             self.f_obj, r.p_tcp_w, r.v_tcp_w, r.timestamp_s, cands, act).items() if k != "per_frame"},
                         kinetic=kinetic_direction_evidence(self.f_obj, r.p_tcp_w, r.v_tcp_w, r.timestamp_s, cands,
                                                            st["surf"], act & np.isin(np.arange(self.T), [
                                                                t for s in self.segments if s.slip_mode == "sliding"
                                                                for t in s.model_time_indices]),
                                                            self.train_mask, self.mf.force),
                         static=static_contact_feasibility(self.f_obj, r.p_tcp_w, cands, st["surf"], self.segments,
                                                           self.mf.force),
                         mechanical=mechanical_model_evidence(self.f_obj, r.p_tcp_w, r.timestamp_s, cands, act,
                                                              self.train_mask, self.mf.force))
        return dict(contact=contact, force=force)

    def _finalize(self, st):
        acc = self.out / "accepted"
        acc.mkdir(parents=True, exist_ok=True)
        rob = self._robot_evidence(st)
        write_json(acc / "force_candidate_comparison.json", rob["force"])
        if self.force_conditional:
            # conditional force never decides the physical selection: the selector sees ABSTAIN, the conditional
            # candidate comparison is kept as a diagnostic next to it
            write_json(acc / "exploratory_force_diagnostics.json", dict(rob["force"], policy="exploratory",
                                                                         label="DIAGNOSTIC_CONDITIONAL"))
            rob = dict(rob, force=dict(status="ABSTAIN", reason="exploratory force is diagnostic only",
                                       used_in_fit=rob["force"].get("used_in_fit")))
        write_json(acc / "force_usage.json", dict(
            force_policy=self.force_policy, conditional=self.force_conditional,
            canonical_robot_record=None if self.record_alignment is None else self.record_alignment["fingerprint"],
            force_rows_in_fit=st["fitinfo"]["hook"].get("n_force_rows") if st.get("fitinfo") else None,
            contact_rows_in_fit=st["fitinfo"]["hook"].get("n_contact_rows") if st.get("fitinfo") else None,
            dense_masks="updated at the final controller stage only (not inside the frontend loop)"))
        robot_tool = None
        if self.robot_ok:
            robot_tool = dict(p_tcp_w=self.rec.p_tcp_w, p_ee_w=self.rec.p_ee_w, tool=self.tool)
        obs_v2 = self._concat(st["obs_tr"], self.test_obs)
        sel = run_selection_v2(obs_v2, self.calibs, self.cfg, acc / "selection", scene=self.scene,
                               fit_tr=st["fit"], partition=self.part, acf=self.acf, test_obs=self.test_obs,
                               robot=robot_tool, legacy=self.legacy, source_files=self.fingerprints, log=self.log,
                               support_override=st["support"], robot_evidence_override=rob)
        import shutil
        if self.force_conditional:
            p_sel = acc / "selection" / "joint_selection_v2.json"
            js = json.loads(p_sel.read_text())
            js["physics_joint_export_ready"] = False
            js["failure_codes"] = sorted(set(js.get("failure_codes", [])) | {"EXPLORATORY_FORCE_DIAGNOSTIC"})
            js["force_policy"] = "exploratory"
            js["diagnostic"] = True
            write_json(p_sel, js)
            sel = dict(sel, physics_joint_export_ready=False, failure_codes=js["failure_codes"],
                       force_policy="exploratory", diagnostic=True)
        shutil.copy2(acc / "selection" / "joint_selection_v2.json", acc / "joint_selection_v2.json")
        # all-frame refit for the authority (selection already made on held-out blocks)
        full = self._concat(st["obs_tr"], self.test_obs)
        allm = np.ones(self.T, bool)
        sup_all, tab_all, _ = self.support_for(full, st["fit"].candidates)
        fit_all = fit_joint_hypotheses(_clean(full, sup_all), self.calibs, self.cfg, ref=self.ref, log=None)
        hook = None
        if self.robot_ok and any(s.active_for_residuals for s in self.segments):
            saved = self.train_mask
            self.train_mask = allm
            hook, surf_all, _ = self.contact_hook(fit_all, sup_all, tab_all)
            self.train_mask = saved
        fit_all, deltas_all = refine_with_hook(fit_all, hook, self.cfg)
        self._export_discovery_format(acc, full, fit_all, sup_all, sel, st)
        return sel

    @staticmethod
    def _concat(a, b):
        from .observations import OBS_FIELDS, TrackObservations
        kw = {f: np.concatenate([getattr(a, f), getattr(b, f)]) for f in OBS_FIELDS}
        if a.uv_sigma_scale is not None or b.uv_sigma_scale is not None:
            kw["uv_sigma_scale"] = np.concatenate([a.uv_sigma_scale if a.uv_sigma_scale is not None else np.ones(a.n_obs),
                                                   b.uv_sigma_scale if b.uv_sigma_scale is not None else np.ones(b.n_obs)])
        return TrackObservations(a.camera_ids, a.frame_map, track_camera=a.track_camera, track_local_id=a.track_local_id,
                                 track_query_time=a.track_query_time, provenance=dict(a.provenance), **kw)

    def _export_discovery_format(self, acc, obs, fit, support, sel, st):
        cands = fit.candidates
        prob = fit.problem
        cj = {k: candidate_to_json(c, obs.frame_map) for k, c in cands.items()}
        decision = sel["selected_candidate"] if sel["physical_joint_status"].startswith("IDENTIFIED_") else \
            "ambiguous/unobservable"
        write_json(acc / "joint_candidates.json", dict(candidates=cj, decision=decision,
                                                       physical_joint_status=sel["physical_joint_status"],
                                                       working_motion_model=sel["working_motion_model"],
                                                       fitted_on="all frames after held-out selection (authority)"))
        write_joint_states(acc / "joint_states.npz", cands, obs.frame_map)
        (acc / "part_support").mkdir(exist_ok=True)
        arrays = dict(track_camera=obs.track_camera, track_local_id=obs.track_local_id, eligible=prob.eligible,
                      static_anchor_w=prob.static_anchors)
        for k, c in cands.items():
            key = "R" if k == "revolute" else "P"
            arrays.update({f"p_static_{key}": c.static_probability, f"p_moving_{key}": c.moving_probability,
                           f"p_outlier_{key}": c.outlier_probability, f"anchor_w_{key}": c.canonical_anchors_w})
        np.savez(acc / "part_support" / "track_support.npz", **arrays)
        np.savez_compressed(acc / "candidate_support.npz", **{k: support[k] for k in ("S_core", "M_proposal", "U", "TOOL")})
        obs.save(acc / "tracks_observations.npz")
        np.savez_compressed(acc / "tracks_refined.npz", recovered_train=st["recovered"],
                            note=np.array("recovered_train flags NCC-recovered observations within the training part"))
        write_json(acc / "model_selection.json", dict(decision=decision, source="multimodal frontend + selection v2",
                                                      physical_joint_status=sel["physical_joint_status"]))
        write_json(acc / "manifest.json", dict(created_unix=time.time(), config=self.cfg.to_dict(),
                                               config_hash=self.cfg.config_hash(), fingerprints=self.fingerprints,
                                               frame_map=obs.frame_map.to_json(), scene=str(self.scene)))
        # contact states / residuals
        rows = []
        np_states = {}
        for k, ds in st["contact"].items():
            for d in ds:
                rows.append(dict(kind=k, segment_id=d["segment_id"], mode=d["mode"], status=d.get("status"),
                                 n_train=d.get("n_train"), chi2_per_dof=d.get("chi2_per_dof"),
                                 contradicted=d.get("contradicted"), drift_max_m=d.get("canonical_drift_max_m"),
                                 heldout_chi2=(d.get("heldout") or {}).get("chi2_per_dof"),
                                 scale=d.get("scale_status")))
                if d.get("anchor_canonical") is not None:
                    np_states[f"anchor_{k}_{d['segment_id']}"] = np.asarray(d["anchor_canonical"])
        _csv(acc / "contact_residuals_per_segment.csv", rows)
        np.savez(acc / "contact_states.npz", contact_state=self.contact_state.astype(str), **np_states)
        write_json(acc / "model_comparison.json", dict(rounds=self.metrics if hasattr(self, "metrics") else None))
        # RGB-D point-cloud initialisation (NOT Gaussians)
        tab = PointTable(obs, self.noise0, self.calibs)
        ci = acc / "canonical_init"
        ci.mkdir(exist_ok=True)
        for name, m in (("static_points.ply", support["S_core"]), ("moving_points.ply", support["M_proposal"])):
            P = tab.X[self.ref, m]
            _write_ply(ci / name, P[np.isfinite(P).all(1)])
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in acc.iterdir() if p.is_file()}
        write_json(acc / "authority_source_manifest.json", dict(files=hashes, fingerprints=self.fingerprints,
                                                                 consumer="integration_contract.authority_from_discovery"))

    def _report(self, sel, metrics):
        L = ["# Multimodal frontend", "", f"* physical_joint_status: **{sel['physical_joint_status']}**",
             f"* working model: {sel['working_motion_model']} (approximation {sel['working_model_is_approximation']})",
             f"* physics_joint_export_ready: {sel['physics_joint_export_ready']}",
             f"* contact evidence used in fit: {sel.get('contact_evidence_used')}; force evidence used: "
             f"{sel.get('force_evidence_used')}; contact assumption satisfied: {sel.get('contact_assumption_satisfied')}",
             f"* force gate: {self.fgate.get('status')} ({self.fgate.get('reason', '')})", "", "## Rounds", ""]
        for m in metrics:
            L.append(f"* round {m['round']}: {m['verdict']}; support {m['support']}; held-out NLL {m['heldout_nll']}")
        L += ["", "## Reasons", ""] + [f"* {r}" for r in sel["reasons"]]
        (self.out / "report.md").write_text("\n".join(L) + "\n")


def _csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def _write_ply(path, P):
    with open(path, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "end_header\n")
        for p in P:
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def run_multimodal_frontend(obs, calibs, cfg, out_dir, **kw) -> dict:
    return Frontend(obs, calibs, cfg, out_dir, **kw).run()
