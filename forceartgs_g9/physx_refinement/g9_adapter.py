"""Strict G9Interaction -> isolated PhysX refinement episode bridge.

Source objects: forceartgs_g9.estimators.model_spec.G9Interaction.
The adapter NEVER re-tares sensors, guesses contact states, recomputes
generalized effort, guesses v0, or silently makes unknown contact free.
"""
from __future__ import annotations
import numpy as np
from .objective import Episode

def episode_from_g9_interaction(interaction, *, mode: str, v0: float,
                                regime_evidence: dict,
                                calibration_status: str = "UNVERIFIED",
                                exploratory_assumptions_accepted: bool = False,
                                train: bool = True,
                                sigma_v0: float | None = None) -> Episode:
    if mode not in ("free", "forced"):
        raise ValueError("G9 interval must have independent FREE/FORCED evidence; UNKNOWN excluded")
    if not isinstance(regime_evidence, dict) or not regime_evidence.get("source"):
        raise ValueError("regime evidence needs non-empty source/provenance")
    if mode == "free":
        if not regime_evidence.get("contact_independently_absent", False):
            raise ValueError("FREE cannot be inferred from near-zero wrench; independent absence required")
        # A free interval has zero external generalized force by the stated physical regime.
        # This does not modify the source G9 interaction or its canonical force record.
        effort_t = np.array([float(interaction.t_obs[0])])
        effort = np.array([0.])
    else:
        if not regime_evidence.get("contact_confirmed", False):
            raise ValueError("FORCED needs confirmed interaction contact, not a force threshold alone")
        effort_t = np.asarray(interaction.effort.t_s, dtype=float)
        effort = np.asarray(interaction.effort.levels, dtype=float)
    initial = getattr(interaction, "initial_state", None)
    q0 = float(interaction.q_obs[0])
    if initial and initial.get("qdot") is not None:
        if abs(float(initial["qdot"]) - float(v0)) > 1e-9:
            raise ValueError("v0 disagrees with G9 declared initial_state.qdot")
    if initial and initial.get("q") is not None:
        # a declared state (e.g. a local fit at t_obs[0]) replaces the raw first sample; its time must match
        if initial.get("t_s") is not None and abs(float(initial["t_s"]) - float(interaction.t_obs[0])) > 1e-9:
            raise ValueError("declared initial_state.t_s is not the first trajectory time")
        q0 = float(initial["q"])
    e = Episode(name=str(interaction.name),
                t_s=np.asarray(interaction.t_obs), q=np.asarray(interaction.q_obs),
                sigma_q=np.asarray(interaction.q_std),
                q0=q0, v0=float(v0),
                input_t_s=effort_t, input_effort=effort,
                mode=mode, calibration_status=calibration_status,
                used_for_training=train,
                exploratory_assumptions_accepted=exploratory_assumptions_accepted,
                sigma_v0=sigma_v0)
    return e

def episodes_from_g9_interactions(interactions, regime_by_name: dict, v0_by_name: dict,
                                  *, calibration_by_name: dict | None = None) -> list[Episode]:
    calibration_by_name = calibration_by_name or {}
    out = []
    for it in interactions:
        if it.name not in regime_by_name or it.name not in v0_by_name:
            raise ValueError(f"{it.name}: missing explicit regime evidence or independent v0")
        cfg = regime_by_name[it.name]
        cal = calibration_by_name.get(it.name, {})
        out.append(episode_from_g9_interaction(it, mode=cfg["mode"], v0=v0_by_name[it.name],
                   regime_evidence=cfg["evidence"], train=cfg.get("train",True),
                   sigma_v0=cfg.get("sigma_v0"),
                   calibration_status=cal.get("status","UNVERIFIED"),
                   exploratory_assumptions_accepted=cal.get("exploratory_opt_in",False)))
    return out


def write_g9_physx_episodes(episodes: list[Episode], regime_provenance: dict, path):
    """Persist already-validated ForceArtGS episodes for estimator-side CLI.

    Does NOT reproject sensor forces or infer contact states. The full upstream
    canonical record remains the source of truth; this JSON is a derived input.
    """
    from dataclasses import asdict
    from pathlib import Path
    import json
    if not regime_provenance or set(regime_provenance) != {e.name for e in episodes}:
        raise ValueError("regime provenance must cover precisely the supplied episodes")
    for e in episodes:
        if not regime_provenance[e.name].get("source"):
            raise ValueError(f"missing source provenance for {e.name}")
    def js(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): js(v) for k, v in x.items()}
        if isinstance(x, list):
            return [js(v) for v in x]
        return x
    doc={"schema":"forceartgs.g9.physx_episodes/v1",
         "episodes":[js(asdict(e)) for e in episodes],
         "regime_provenance":regime_provenance}
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    return path


# ----------------------------------------------------------------------------------------------------------------
# G9 free/forced segments -> G9Interaction + strict regime evidence (no new classification, no new force)

SEGMENT_FREE = "FREE_CONFIRMED"
SEGMENT_FORCED = "FORCED_MEASURED"


def interactions_from_segments(segments, *, force_audit_status, v0_prefix_samples=4, min_samples=4):
    """Admissible G10 inputs from forceartgs_g9.free_forced MotionSegments (the SAME classifier as G9 free/forced).

    * FREE_CONFIRMED  -> free, ``contact_independently_absent`` (release evidence: outside every front-end contact
      segment and no force-detected contact, or calibrated geometric TCP separation; effort noise-consistent).
    * FORCED_MEASURED -> forced, ``contact_confirmed`` (front-end contact segment = force peaks + visual, or a
      force-detected contact / sliding state; significant effort; not sticking).  The effort is the segment's
      canonical object-side ``Q`` on its native 50 Hz rows (ZOH), unchanged.
    * anything else (UNKNOWN, FREE_HYPOTHESIS:*) -> EXCLUDED, listed with the reason.
    * initial state: weighted quadratic on the first ``v0_prefix_samples`` q samples (free_forced.replay.prefix_state);
      the episode starts at the last prefix sample. This v0 is ESTIMATED FROM THE SAME TRAJECTORY (not independent),
      and is recorded as such with its standard deviation.
    Forced input needs force-audit status VERIFIED, or EXPLORATORY_ASSUMPTIONS_ACCEPTED (explicit exploratory opt-in
    recorded in the G9 force audit); CONTRADICTED / DATA_UNAVAILABLE forced segments are excluded."""
    from ..estimators.model_spec import G9Interaction
    from ..free_forced.replay import prefix_state
    from ..synthetic import ZOH
    inter, regime, v0s, cal, excluded = [], {}, {}, {}, []
    for s in segments:
        if s.label not in (SEGMENT_FREE, SEGMENT_FORCED):
            excluded.append(dict(name=s.name, label=s.label, reason="UNKNOWN / hypothesis-only contact regime: "
                                 "excluded, never converted to force-free"))
            continue
        k = int(v0_prefix_samples)
        if len(s.t) - (k - 1) < min_samples:
            excluded.append(dict(name=s.name, label=s.label, reason=f"{len(s.t)} q samples: fewer than "
                                 f"{k - 1 + min_samples} (prefix {k} + {min_samples} rollout samples)"))
            continue
        tc, qc, vc, C, info = prefix_state(s, k)
        sl = slice(k - 1, None)
        if s.label == SEGMENT_FORCED:
            if force_audit_status not in ("VERIFIED", "EXPLORATORY_ASSUMPTIONS_ACCEPTED"):
                excluded.append(dict(name=s.name, label=s.label,
                                     reason=f"force audit {force_audit_status}: forced input not admissible"))
                continue
            if len(s.force_t) == 0 or s.force_t[0] > tc + 1e-9:
                excluded.append(dict(name=s.name, label=s.label, reason="force rows start after the initial state"))
                continue
            regime[s.name] = dict(mode="forced", evidence=dict(
                source="forceartgs_g9.free_forced.segments.classify: FORCED_MEASURED "
                       f"({s.evidence.get('reason')}; front-end contact segment = force peaks + visual, or a "
                       "force-detected contact mode; significant effort; not sticking)",
                contact_confirmed=True, interval_rows=s.evidence.get("interval_rows"),
                tcp_tracking=s.evidence.get("tcp_tracking")))
            cal[s.name] = dict(status=("VERIFIED" if force_audit_status == "VERIFIED" else force_audit_status),
                               exploratory_opt_in=force_audit_status == "EXPLORATORY_ASSUMPTIONS_ACCEPTED")
            effort = ZOH(s.force_t, s.Q, model="canonical object-side Q (native 50 Hz rows, forward ZOH)")
        else:
            regime[s.name] = dict(mode="free", evidence=dict(
                source="forceartgs_g9.free_forced.segments.classify: FREE_CONFIRMED (release evidence AND "
                       "noise-consistent effort)", contact_independently_absent=True,
                interval_rows=s.evidence.get("interval_rows")))
            cal[s.name] = {}
            effort = ZOH(np.array([tc]), np.array([0.0]), model="free: zero applied effort by regime")
        regime[s.name]["sigma_v0"] = float(np.sqrt(max(C[1, 1], 0.0))) or None
        v0s[s.name] = vc
        inter.append(G9Interaction(s.name, s.joint_type, s.t[sl], s.q[sl], s.q_sigma[sl], effort,
                                   initial_state=dict(t_s=tc, q=qc, qdot=vc, source=(
                                       f"ESTIMATED_FROM_OBSERVED_PREFIX: weighted quadratic on {k} samples "
                                       "(same trajectory, NOT independent)"), q_sd=float(np.sqrt(max(C[0, 0], 0))),
                                       qdot_sd=float(np.sqrt(max(C[1, 1], 0))), fit=info),
                                   meta=dict(segment_label=s.label, sign=s.sign, evidence=s.evidence)))
    return inter, regime, v0s, cal, excluded
