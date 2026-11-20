"""R/P evidence ablation of the pre-3DGS frontend (G9 corrections section 5.4-5.5).

Three runs on the SAME observation table, the SAME candidate hypotheses and the SAME frozen held-out blocks
(``declare_partition`` is deterministic in the observations / config):
  rgbd            RGB-D tracks only                                     (certified policy)
  rgbd_tcp        + TCP contact modes (the production configuration)    (certified policy)
  rgbd_tcp_force  + canonical-record force as CONDITIONAL evidence in candidate fits under an explicitly declared
                  actuation model (min_effort_tangent)                  (exploratory policy -> DIAGNOSTIC)
Each run writes its own directory; the production ``multimodal_refine`` authority is never touched. The
comparison records joint status / type, axis, hinge, q range, the held-out objective of the accepted round, the
force / contact consistency and whether the decision changed. Accept / rollback: the certified decision stands
unless a candidate improves the predeclared held-out objective without breaking a physical constraint -- and an
exploratory (conditional-force) candidate is NEVER promoted; force-work alignment is never a prismatic selector.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else None


def _summary(run_dir):
    acc = Path(run_dir) / "accepted"
    sel = _load(acc / "joint_selection_v2.json") or {}
    val = _load(acc / "validation_report.json") or {}
    cand = (_load(acc / "joint_candidates.json") or {}).get("candidates", {})
    kind = sel.get("selected_candidate") or sel.get("working_motion_model")
    c = cand.get(kind) or {}
    rounds = val.get("rounds") or []
    k = val.get("accepted_round")
    acc_round = rounds[k] if isinstance(k, int) and 0 <= k < len(rounds) else (rounds[-1] if rounds else {})
    q = np.asarray(c.get("q") or [0.0], float)
    return dict(physical_joint_status=sel.get("physical_joint_status"), selected=sel.get("selected_candidate"),
                working_model=sel.get("working_motion_model"),
                physics_joint_export_ready=sel.get("physics_joint_export_ready"),
                failure_codes=sel.get("failure_codes"), axis_world=c.get("joint_axis_world"),
                origin_world=c.get("joint_origin_world"), q_range=float(np.ptp(q)),
                heldout_objective=acc_round.get("heldout_nll"), accepted_round=k,
                contact_evidence_used=sel.get("contact_evidence_used"), force_evidence_used=sel.get("force_evidence_used"),
                force_usage=_load(acc / "force_usage.json"),
                exploratory_force=_load(acc / "exploratory_force_diagnostics.json"))


def _axis_angle(a, b):
    if a is None or b is None:
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def run_ablation(obs, calibs, cfg, out_dir, *, scene, rec, record, segments_source, wrench_calibration, images=None,
                 legacy=None, source_files=None, log=print) -> dict:
    from .multimodal_frontend import run_multimodal_frontend
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    plan = [("rgbd", dict(rec=None, segments_source=None, wrench_calibration=None), cfg),
            ("rgbd_tcp", dict(rec=rec, segments_source=segments_source, wrench_calibration=wrench_calibration,
                              robot_record=record, force_policy="certified"), cfg)]
    cfg_f = copy.deepcopy(cfg)
    cfg_f.multimodal_frontend.force.mechanical_model = "min_effort_tangent"     # explicitly declared assumption
    plan.append(("rgbd_tcp_force", dict(rec=rec, segments_source=segments_source,
                                        wrench_calibration=wrench_calibration, robot_record=record,
                                        force_policy="exploratory"), cfg_f))
    for name, kw, c in plan:
        d = out / name
        if (d / "accepted" / "joint_selection_v2.json").is_file():
            log(f"[ablation] {name}: reuse {d}")
        else:
            log(f"[ablation] {name}: run")
            run_multimodal_frontend(obs, calibs, c, d, scene=scene, images=images, legacy=legacy,
                                    source_files=source_files, log=log, **kw)
        runs[name] = _summary(d)
    base = runs["rgbd_tcp"]
    comp = {}
    for name, r in runs.items():
        comp[name] = dict(r, decision_changed_vs_certified=(r["physical_joint_status"], r["selected"] or
                                                            r["working_model"]) != (base["physical_joint_status"],
                                                                                    base["selected"] or
                                                                                    base["working_model"]),
                          axis_change_vs_certified_deg=_axis_angle(r["axis_world"], base["axis_world"]),
                          q_range_change_vs_certified=r["q_range"] - base["q_range"])
    hb, hf = base.get("heldout_objective"), runs["rgbd_tcp_force"].get("heldout_objective")
    improves = None
    if isinstance(hb, dict) and isinstance(hf, dict):
        kb, kf = base.get("selected") or base.get("working_model"), runs["rgbd_tcp_force"].get("selected") or \
            runs["rgbd_tcp_force"].get("working_model")
        vb, vf = hb.get(kb), hf.get(kf)
        if isinstance(vb, (int, float)) and isinstance(vf, (int, float)):
            improves = bool(vf < vb)
    res = dict(schema="forceartgs.frontend_ablation/v1", variants=comp, certified_reference="rgbd_tcp",
               partition="identical frozen temporal blocks (declare_partition on the same observation table)",
               exploratory_heldout_improves=improves,
               decision=dict(kept="rgbd_tcp (certified)",
                             rule="an exploratory conditional-force candidate is never promoted over the certified "
                                  "authority; it is reported with its held-out comparison; force-work alignment is "
                                  "not a joint-type selector"),
               force_changed_selected_hypothesis=comp["rgbd_tcp_force"]["decision_changed_vs_certified"])
    (out / "ablation_report.json").write_text(json.dumps(res, indent=1, default=str))
    return res
