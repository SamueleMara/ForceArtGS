"""Contact timelines with explicit state and provenance (Stage D1).

States, per RGB-D model time (joined on ``source_frame_index``):

``contact_confirmed``  only from an independent annotation (operator label, independent video
                       annotation). The ONLY state that may activate hard surface-contact terms.
``contact_probable``   two independent sensor cues agree: a sustained force event AND the TCP is
                       within the geometric tolerance of observed object geometry.
``contact_unknown``    a single cue, conflicting cues, or no usable cue.
``no_contact``         the TCP is geometrically far from every observed object surface and the
                       force is within baseline noise (or an annotation says absent).

A force threshold is a *heuristic*; it never yields ``contact_confirmed``. Heuristics can be
evaluated against the timeline (false activations where ``no_contact``).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STATES = ("contact_confirmed", "contact_probable", "contact_unknown", "no_contact")
INDEPENDENT_PROVENANCE = ("operator_label", "independent_video_annotation")
PROVENANCE = INDEPENDENT_PROVENANCE + ("sensor_event", "geometric_proximity", "sensor_event+geometric_proximity",
                                      "heuristic_force_threshold", "none")


@dataclass
class ContactTimeline:
    source_frame_index: np.ndarray
    state: np.ndarray            # [T] str in STATES
    provenance: np.ndarray       # [T] str in PROVENANCE
    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        self.source_frame_index = np.asarray(self.source_frame_index, int)
        self.state = np.asarray(self.state, dtype=object)
        self.provenance = np.asarray(self.provenance, dtype=object)
        bad = set(self.state.tolist()) - set(STATES)
        if bad:
            raise ValueError(f"unknown contact states {bad}")
        conf = self.state == "contact_confirmed"
        if np.any(conf & ~np.isin(self.provenance, INDEPENDENT_PROVENANCE)):
            raise ValueError("contact_confirmed requires independent provenance (operator/video annotation)")

    @property
    def hard_constraint_mask(self) -> np.ndarray:
        """Only independently confirmed contact activates hard surface-contact terms."""
        return self.state == "contact_confirmed"

    def counts(self) -> dict:
        return {s: int((self.state == s).sum()) for s in STATES}

    def provenance_counts(self) -> dict:
        return {p: int((self.provenance == p).sum()) for p in set(self.provenance.tolist())}

    def select(self, source_frames) -> "ContactTimeline":
        idx = {int(s): i for i, s in enumerate(self.source_frame_index)}
        missing = [int(s) for s in source_frames if int(s) not in idx]
        if missing:
            raise KeyError(f"contact timeline lacks source frames {missing[:5]} (no interpolation)")
        sel = np.array([idx[int(s)] for s in source_frames], int)
        ev = {k: (np.asarray(v)[sel] if isinstance(v, np.ndarray) and len(v) == len(self.state) else v)
              for k, v in self.evidence.items()}
        return ContactTimeline(self.source_frame_index[sel], self.state[sel], self.provenance[sel], ev)


def load_annotation_csv(path) -> ContactTimeline:
    """Independent annotations: columns ``source_frame_index, state, provenance``.

    ``state`` in {contact, absent, uncertain}; ``provenance`` must be an independent source to
    produce confirmed contact. Legacy force-z-score exports (columns ``contact_state`` and
    ``reason``) are read as ``sensor_event`` provenance, i.e. at most ``contact_unknown`` here."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: empty annotation file")
    sf, st, pv = [], [], []
    for r in rows:
        sf.append(int(r["source_frame_index"]))
        raw = (r.get("state") or r.get("contact_state") or "").strip().lower()
        prov = (r.get("provenance") or "sensor_event").strip()
        if prov not in PROVENANCE:
            raise ValueError(f"{path}: unknown provenance {prov!r}")
        indep = prov in INDEPENDENT_PROVENANCE
        if raw == "contact":
            st.append("contact_confirmed" if indep else "contact_unknown")
        elif raw in ("absent", "no_contact"):
            st.append("no_contact" if indep else "contact_unknown")
        elif raw in ("uncertain", "unknown", ""):
            st.append("contact_unknown")
        else:
            raise ValueError(f"{path}: unknown state {raw!r}")
        pv.append(prov)
    return ContactTimeline(np.array(sf), np.array(st, dtype=object), np.array(pv, dtype=object),
                           dict(source=str(path)))


def fuse_sensor_cues(source_frame_index, force_state, near_tcp, far_tcp) -> ContactTimeline:
    """Fuse a force-event state (1 contact / 0 absent / -1 uncertain) with geometric proximity
    (``near_tcp``: TCP within tolerance of observed object geometry; ``far_tcp``: beyond the far
    threshold in every camera that sees the TCP region; both False = unknown)."""
    fs = np.asarray(force_state, int)
    near = np.asarray(near_tcp, bool)
    far = np.asarray(far_tcp, bool)
    state = np.full(len(fs), "contact_unknown", dtype=object)
    prov = np.full(len(fs), "none", dtype=object)
    prob = (fs == 1) & near
    state[prob] = "contact_probable"
    prov[prob] = "sensor_event+geometric_proximity"
    nc = (fs == 0) & far
    state[nc] = "no_contact"
    prov[nc] = "sensor_event+geometric_proximity"
    one = ~prob & ~nc & ((fs == 1) | near | far | (fs == 0))
    prov[one] = np.where((fs[one] == 1) | (fs[one] == 0), "sensor_event", "geometric_proximity")
    return ContactTimeline(source_frame_index, state, prov,
                           dict(force_state=fs, near_tcp=near, far_tcp=far,
                                rule="probable = sustained force event AND TCP near observed geometry; "
                                     "no_contact = force within noise AND TCP far; else unknown"))


def evaluate_heuristic(flags, timeline: ContactTimeline) -> dict:
    """Precision-style summary of a heuristic contact flag against the fused/annotated timeline.

    Without confirmed labels this is not ground truth; it reports agreement with independent
    cues and false activations where the timeline says ``no_contact``."""
    f = np.asarray(flags, bool)
    s = timeline.state
    pos = np.isin(s, ("contact_confirmed", "contact_probable"))
    out = dict(n_flagged=int(f.sum()), n_timeline_positive=int(pos.sum()),
               flagged_and_positive=int((f & pos).sum()), false_activations_on_no_contact=int((f & (s == "no_contact")).sum()),
               flagged_on_unknown=int((f & (s == "contact_unknown")).sum()),
               missed_positive=int((~f & pos).sum()),
               reference="confirmed annotations" if (s == "contact_confirmed").any() else
               "fused sensor cues (NOT ground truth)")
    out["precision_vs_timeline"] = (out["flagged_and_positive"] / out["n_flagged"]) if out["n_flagged"] else None
    out["recall_vs_timeline"] = (out["flagged_and_positive"] / out["n_timeline_positive"]) if out["n_timeline_positive"] else None
    return out


def tcp_proximity_to_observed_object(p_tcp_w, calib, depth, mask, window_px=60, tool_mask=None):
    """Min 3D distance [m] from the TCP to measured, masked object depth points around its image
    projection; NaN when the TCP is outside the image/behind the camera or no masked depth exists."""
    from .observations import project, unproject
    uv, z = project(np.asarray(p_tcp_w, float)[None], calib.K, calib.T_CW)
    u, v = uv[0]
    H, W = depth.shape
    if not (z[0] > 0 and 0 <= u < W and 0 <= v < H):
        return np.nan, "tcp_not_in_view"
    x0, x1 = int(max(u - window_px, 0)), int(min(u + window_px + 1, W))
    y0, y1 = int(max(v - window_px, 0)), int(min(v + window_px + 1, H))
    d = depth[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1] & np.isfinite(d) & (d > 0)
    if tool_mask is not None:
        m &= ~tool_mask[y0:y1, x0:x1]
    if not m.any():
        return np.nan, "no_masked_depth_near_tcp"
    vv, uu = np.nonzero(m)
    P = unproject(np.c_[uu + x0, vv + y0].astype(float), d[m], calib.K, calib.T_WC)
    return float(np.min(np.linalg.norm(P - p_tcp_w, axis=1))), "ok"
