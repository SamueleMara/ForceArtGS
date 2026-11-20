"""G9 controller stages: ``dynamics`` and ``final_physics_export`` (opt-in; reruns from an existing final/ and G8
visual handoff without any 3DGS training).

Run directory: ``<forceartgs_run>/dynamics/g9/vNNN`` (same scene and branch). A version is reused only when its
input signature (authority id, joint_states hash, robot archive, contact segments, mesh, config, physics library
revision) is identical; any change writes the next version -- results are never silently overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import physics_lib
from .config import load_dynamics_config, resolve_path
from .core import run_g9


def _versions(root: Path):
    return sorted(p for p in root.glob("v[0-9][0-9][0-9]") if p.is_dir())


def apply_force_calibration(fa: dict, calibration_file) -> dict:
    """A MEASURED calibration record may upgrade individual gates: {gates: {name: {status: PASS, method,
    evidence_sha256}}}. A gate without method + evidence hash stays as audited."""
    if not calibration_file:
        return fa
    p = Path(calibration_file)
    rec = json.loads(p.read_text())
    applied = []
    for k, v in (rec.get("gates") or {}).items():
        if k in fa.get("gates", {}) and v.get("status") == "PASS" and v.get("method") and v.get("evidence_sha256"):
            fa["gates"][k] = dict(status="PASS", reason=f"measured calibration record {p.name}: {v['method']}",
                                  evidence_sha256=v["evidence_sha256"])
            applied.append(k)
    fa["failing"] = [k for k, g in fa["gates"].items() if g["status"] not in ("PASS", "NOT_NEEDED")]
    if fa.get("status") != "NO_FORCE":
        fa["status"] = "VERIFIED" if not fa["failing"] else "FORCE_CALIBRATION_UNVERIFIED"
    fa["calibration_record"] = dict(path=str(p), applied=applied)
    return fa


def run_dynamics_stage(run_root, scene, dynamics_config, *, physics_library=None, robot_archive=None,
                       contact_segments=None, force_calibration=None, mass_model_spec=None, episodes=None,
                       force_export=None, allow_diagnostic_kinematics=False, declared_joint_type=None,
                       declaration_note=None, log=print) -> dict:
    from .adapters.forceartgs_g9 import build_inputs
    prov = physics_lib.load(physics_library)
    cfg = load_dynamics_config(dynamics_config)
    if allow_diagnostic_kinematics:
        cfg["pipeline"]["allow_diagnostic_kinematics"] = True
    if mass_model_spec:
        import yaml
        mm = yaml.safe_load(Path(mass_model_spec).read_text()) or {}
        bad = sorted(set(mm) - set(cfg["mass_model"]))
        if bad:
            raise ValueError(f"mass-model spec: unknown keys {bad}")
        base = Path(mass_model_spec).resolve().parent
        if mm.get("mesh") and not Path(mm["mesh"]).is_absolute():
            mm["mesh"] = str(base / mm["mesh"])
        cfg["mass_model"].update(mm)
    inp = build_inputs(run_root, scene, cfg, robot_archive=robot_archive, contact_segments=contact_segments,
                       force_export=force_export, declared_joint_type=declared_joint_type,
                       declaration_note=declaration_note, log=log)
    inp.force_audit = apply_force_calibration(inp.force_audit, force_calibration)
    if inp.force_audit.get("status") == "VERIFIED" and not inp.train and inp.excitation_only:
        inp.train, inp.excitation_only = inp.excitation_only, []
    if episodes:
        import yaml
        ep = yaml.safe_load(Path(episodes).read_text()) or {}
        for h in ep.get("heldout") or []:
            hi = build_inputs(h["run_root"], h["scene"], cfg, robot_archive=h.get("robot_archive"),
                              contact_segments=h.get("contact_segments"), log=log)
            hi.force_audit = apply_force_calibration(hi.force_audit, h.get("force_calibration") or force_calibration)
            its = hi.train if hi.force_audit.get("status") == "VERIFIED" else []
            for it in its:
                if h.get("initial_state") == "declared_rest":
                    it.initial_state = dict(t_s=float(it.t_obs[0]), q=float(it.q_obs[0]), qdot=0.0,
                                            source="declared: at rest at the start of the contact segment")
                it.meta["episode_run_root"] = h["run_root"]
            inp.heldout += its
    inp.input_hashes["physics_library"] = prov["commit"]
    inp.input_hashes["dynamics_config"] = json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                                                     sort_keys=True, default=str)
    root = Path(run_root) / "dynamics" / "g9"
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / ".staging"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    res = run_g9(inp, cfg, tmp, log=log)
    sig = res["manifest"]["input_signature"]
    for v in _versions(root):
        m = v / "g9_manifest.json"
        if m.is_file() and json.loads(m.read_text()).get("input_signature") == sig:
            import shutil
            shutil.rmtree(tmp)
            log(f"G9: identical inputs already in {v}; reused")
            res["out_dir"] = str(v)
            res["reused"] = True
            return res
    nv = root / f"v{len(_versions(root)):03d}"
    tmp.rename(nv)
    res["out_dir"] = str(nv)
    res["reused"] = False
    res["visual_handoff"] = inp.visual_handoff
    return res


def latest_version(run_root):
    v = _versions(Path(run_root) / "dynamics" / "g9")
    return v[-1] if v else None


def run_final_physics_export_stage(run_root, *, g9_dir=None, log=print) -> dict:
    from .export.final_build import final_physics_export
    g9 = Path(g9_dir) if g9_dir else latest_version(run_root)
    if g9 is None:
        return dict(status="NOT_RUN", reason="no dynamics/g9 result: run the dynamics stage first")
    man = json.loads((g9 / "g9_manifest.json").read_text())
    hand = man.get("input_hashes", {}).get("visual_handoff_sha256")
    sel = json.loads((Path(run_root) / "selection.json").read_text())
    kind = sel.get("selected") or sel.get("working_motion_model")
    cands = [Path(run_root) / "parttwin_handoff" / n / "articulated_3dgs_handoff.json"
             for n in ("accepted", f"diagnostic_{kind}")]
    h = next((c for c in cands if c.is_file()), None)
    if h is None:
        return dict(status="BLOCKED_NO_VISUAL_HANDOFF", reason="run the G8 export stage (visual handoff) first")
    import hashlib
    if hand and hashlib.sha256(h.read_bytes()).hexdigest() != hand:
        return dict(status="BLOCKED_STALE_DYNAMICS", reason="the visual handoff changed after the dynamics fit")
    return final_physics_export(g9, h, Path(run_root) / "parttwin_assets", log=log)
