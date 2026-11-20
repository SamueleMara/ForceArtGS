"""G9 physics documents and the ONE translation into the shared PArtTwin1D builder input (G9 spec 6.4, 7).

physics_parameters.json   per term: estimate, units, ci95, fitted_or_fixed, evidence, identifiable, failure_code;
                          disabled terms: {estimate: null, status: disabled_by_user}; priors keep ASSUMED/MEASURED
dynamics_handoff.yaml     schema forceartgs.g9.dynamics_handoff/v1 with a versioned ``parttwin.dynamics/v2`` block
                          (joint-space identification + geometry-constrained inertial + engine term matrix). It is NOT
                          the builder's physics spec and is never passed as one.
physics_spec_v1.yaml      the builder's existing full physics spec (schema_version 1), written ONLY when every field
                          it requires is supported: identified damping / Coulomb, reconstructed moving-link inertial,
                          and the user's MEASURED mechanical spec (collision meshes, physical stops, velocity / effort
                          limits, static-link inertial, registration). Otherwise the exact missing fields are listed.
No URDF / USD / MJCF writer exists here: the shared builder writes all three.
"""
from __future__ import annotations

import math

import numpy as np

from ..dynamics.general_1dof import TERMS, UNITS

HANDOFF_SCHEMA = "forceartgs.g9.dynamics_handoff/v1"
PARTTWIN_V2 = "parttwin.dynamics/v2"
MECH_SCHEMA = "forceartgs.g9.mechanical_spec/v1"
TERM_PARAM = {"inertia": ("alpha", "M"), "viscous": ("c",), "coulomb": ("Fc",), "static": ("dFs", "Fs"),
              "gravity": ("gA", "gB", "gP"), "bias": ("b",)}


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


def parameter_table(terms, joint_type, fit, ident, spec, *, physical_claim: bool, label: str,
                    exploratory: bool = False) -> dict:
    """One entry per TERM (and the parameters it owns)."""
    u = UNITS[joint_type]
    out = {}
    est = (fit or {}).get("estimates", {})
    par = (ident or {}).get("parameters", {})
    phys = (fit or {}).get("physical", {})
    for t in TERMS:
        if not getattr(terms, t):
            out[t] = dict(estimate=None, status="disabled_by_user")
            continue
        names = [n for n in TERM_PARAM[t] if n in est or n in (spec.fixed_priors if spec else {})]
        if not names:
            if t == "gravity" and spec is not None and spec.parameter_source == "geometry_coupled":
                g = phys.get("gravity")
                out[t] = dict(status="geometry_coupled" if g else "not_fitted",
                              note="G(q) = alpha G_1(q): no free gravity parameter; determined by alpha and the mass "
                                   "model", coefficients=g,
                              identifiable=par.get("alpha", {}).get("identifiable") if g else None)
            else:
                out[t] = dict(estimate=None, status="not_fitted", reason=(fit or {}).get("blocked_reason"))
            continue
        e = {}
        for n in names:
            if n in (spec.fixed_priors if spec else {}):
                p = spec.fixed_priors[n]
                e[n] = dict(estimate=p["value"], std=p.get("std"), fitted_or_fixed="fixed_prior", label=p["label"],
                            source=p.get("source"), identifiable=None)
                continue
            pi = par.get(n, {})
            e[n] = dict(estimate=est[n]["value"], ci95=pi.get("ci95"), std_total=pi.get("std_total"),
                        fitted_or_fixed="fitted", identifiable=pi.get("identifiable"),
                        failure_code=pi.get("failure_code") or (None if pi else "NOT_ASSESSED"),
                        evidence=pi.get("reasons"))
        unit = {"inertia": u["M"], "viscous": u["c"], "coulomb": u["Fc"], "static": u["Fs"], "gravity": u["effort"],
                "bias": u["b"]}[t]
        ok = all(v.get("identifiable") is True for v in e.values() if v.get("fitted_or_fixed") == "fitted")
        if exploratory:
            stt = "estimated_conditional" if ok else "not_identifiable"
        else:
            stt = "identified" if (ok and physical_claim) else ("diagnostic_only" if not physical_claim
                                                               else "unidentifiable")
        entry = dict(status=stt,
                     units=unit, parameters=e, label=label)
        if t == "static" and "dFs" in e:
            entry["Fs_total"] = phys.get("Fs")
            entry["note"] = "stored as dFs = Fs - Fc >= 0 (Fs >= Fc enforced)"
        if t == "inertia":
            entry["M_eff"] = phys.get("M")
        if t == "bias":
            entry["note"] = "nuisance generalized load: never a sensor tare, never exported as friction"
        if not physical_claim:
            entry["not_physical"] = label
        out[t] = entry
    return out


def term_ok(table, t) -> bool:
    e = table.get(t, {})
    return e.get("status") in ("disabled_by_user", "identified") or (
        t == "gravity" and e.get("status") == "geometry_coupled" and e.get("identifiable") is True)


# ----------------------------------------------------------------------------------------------- mechanical spec
def load_mechanical_spec(path) -> dict:
    """User-supplied MEASURED mechanical facts (never estimated by G9). Strict keys."""
    import yaml
    from pathlib import Path
    p = Path(path)
    d = yaml.safe_load(p.read_text()) or {}
    allowed = {"schema", "registration", "collision", "static_link_inertial", "joint_limits", "contact", "notes",
               "validation"}
    bad = sorted(set(d) - allowed)
    if bad:
        raise ValueError(f"{p}: unknown mechanical-spec keys {bad}; allowed {sorted(allowed)}")
    if d.get("schema") != MECH_SCHEMA:
        raise ValueError(f"{p}: schema must be {MECH_SCHEMA}")
    d["_base_dir"] = str(p.resolve().parent)
    return d


def physics_spec_v1(*, table, recon_capture, mech: dict | None, anchor_frame: int, g9_gravity, joint_type,
                    phys: dict) -> tuple:
    """Translate (G9 result + mechanical spec) -> PArtTwin1D physics spec v1 dict, or (None, missing)."""
    missing = []
    for t in ("inertia", "viscous", "coulomb"):
        if not term_ok(table, t):
            missing.append(f"joint dynamics: term {t} not identified ({table.get(t, {}).get('status')})")
    if table.get("static", {}).get("status") not in ("disabled_by_user",):
        pass                                     # static friction has no v1 field: recorded in the term matrix
    if not term_ok(table, "gravity"):
        missing.append(f"gravity term not identified / not geometry-coupled ({table.get('gravity', {}).get('status')})")
    if table.get("bias", {}).get("status") not in ("disabled_by_user", None):
        missing.append("bias enabled: a nuisance load cannot be part of a production physics spec (disable it or "
                       "explain it physically)")
    if recon_capture is None:
        missing.append("moving_link inertial: no geometry-constrained reconstruction (mass model / identification)")
    if mech is None:
        missing.append("mechanical spec (forceartgs.g9.mechanical_spec/v1): collision meshes, physical joint stops, "
                       "velocity / effort limits, static-link inertial, registration")
        return None, missing
    for k in ("registration", "collision", "static_link_inertial", "joint_limits"):
        if mech.get(k) is None:
            msg = {"static_link_inertial": "static_link inertial: the PArtTwin v1 schema requires one even for a "
                                           "fixed root (SCHEMA_GAP); G9 never fabricates base mass -- supply a "
                                           "measured value",
                   "collision": "collision.static / collision.moving meshes (validated closed geometry)",
                   "joint_limits": "physical joint stops lower/upper + velocity_limit + effort_limit (NOT the "
                                   "observed demonstration range)",
                   "registration": "registration: T_world_capture + gravity_alignment (measured / declared)"}[k]
            missing.append(msg)
    reg = mech.get("registration") or {}
    if reg and reg.get("gravity_alignment") not in ("measured", "declared"):
        missing.append("registration.gravity_alignment must be measured or declared for gravity-consistent physics")
    if reg and g9_gravity is not None:
        T = np.eye(4) if reg.get("T_world_capture") is None else np.asarray(reg["T_world_capture"], float)
        g_engine_capture = T[:3, :3].T @ np.array([0.0, 0.0, -9.81])
        gv = np.asarray(g9_gravity.g_world, float)
        ang = math.degrees(math.acos(np.clip(gv @ g_engine_capture / (np.linalg.norm(gv) * 9.81), -1, 1)))
        if ang > 0.5:
            missing.append(f"GRAVITY_REGISTRATION_INCONSISTENT: the G9 gravity differs by {ang:.2f} deg from the "
                           f"engine gravity mapped through registration.T_world_capture")
    if missing:
        return None, missing
    lim = mech["joint_limits"]
    for k in ("lower", "upper", "velocity_limit", "effort_limit", "provenance"):
        if lim.get(k) is None:
            missing.append(f"joint_limits.{k}")
    if lim.get("provenance") not in (None, "measured", "estimated"):
        missing.append("joint_limits.provenance must be measured or estimated")
    if missing:
        return None, missing
    base = mech["_base_dir"]
    col = {}
    for side in ("static", "moving"):
        c = dict(mech["collision"][side])
        src = c.get("source")
        if src and not src.startswith("/"):
            c["source"] = f"{base}/{src}"
        if side == "moving":
            c.setdefault("at_original_frame", int(anchor_frame))
        col[side] = c
    st = mech["static_link_inertial"]
    spec = {"schema_version": 1, "registration": {k: v for k, v in reg.items() if k in (
        "T_world_capture", "gauge", "gravity_alignment", "provenance")},
            "collision": col,
            "links": {"static_link": {k: st[k] for k in ("mass_kg", "com_capture_m", "inertia_com_capture_kg_m2",
                                                         "provenance")},
                      "moving_link": {"mass_kg": recon_capture["mass_kg"], "com_capture_m": recon_capture["com_m"],
                                      "inertia_com_capture_kg_m2": recon_capture["inertia_com_kg_m2"],
                                      "provenance": "estimated", "at_original_frame": int(anchor_frame)}},
            "joint": {"lower": float(lim["lower"]), "upper": float(lim["upper"]),
                      "velocity_limit": float(lim["velocity_limit"]), "effort_limit": float(lim["effort_limit"]),
                      "damping": float(phys["c"] if phys.get("c") is not None else 0.0),
                      "coulomb_friction": float(phys["Fc"] if phys.get("Fc") is not None else 0.0),
                      "armature": None, "actuation": None, "provenance": "estimated"},
            "notes": "GENERATED by forceartgs_g9.export.dynamics_v2 from the G9 identification (damping, Coulomb, "
                     "moving-link inertial = estimated density x validated geometry) and the user's measured "
                     "mechanical spec; armature null (the inertia is in the link tensor, never duplicated). "
                     f"SCHEMA_GAP: v1 has ONE joint provenance field; it is 'estimated' because damping / Coulomb "
                     f"are identified, while the stops / velocity / effort limits are "
                     f"'{lim['provenance']}' (mechanical spec)"}
    if mech.get("contact") is not None:
        spec["contact"] = mech["contact"]
    if mech.get("validation") is not None:
        spec["validation"] = mech["validation"]
    return spec, []


def write_yaml(path, doc, header=""):
    import yaml
    from pathlib import Path
    Path(path).write_text((f"# {header}\n" if header else "") + yaml.safe_dump(_clean(doc), sort_keys=False,
                                                                                 width=1000))
