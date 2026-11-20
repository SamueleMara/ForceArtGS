"""Versioned wrench-calibration contract and force-selection readiness gates (Stage D2/D3).

``F_object^W = sign * R_WEE R_EE_S [F_recorded^S - F_bias^S(q_robot, t)]``

Every element is carried with its provenance and a *verification status*. A value that is only
declared (e.g. "user confirmation: sensor axes equal EE axes") is ``DECLARED`` and is not the same
as an experimental verification (``VERIFIED``). Force-based joint selection requires every gate
to be ``PASS``; otherwise the force path returns ``ABSTAIN`` with the failing gates, and vision
selection is left untouched. Raw and compensated channels are kept separately; nothing here
subtracts the entire force during contact.

Gates
-----
frame_rotation   R_EE_S known AND experimentally verified (reference-force direction test or
                 multi-pose gravity fit).
sign             recorded quantity known AND verified by a reference push of known direction.
tare             zero offset estimated on a contact-free window with stated noise; drift between
                 two separated contact-free windows below tolerance.
gravity          either a verified tool-gravity model, or the EE orientation is constant over the
                 evaluated interval within a bound that, with a declared tool-mass upper bound,
                 keeps the gravity change below the force noise.
time_sync        clock offset/uncertainty established from an independent event (an exact
                 ``source_frame_index`` join alone is NOT hardware synchronization).
torque_transport sensor origin p_S^EE known (only needed when torques are used).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
PASS, FAIL, UNVERIFIED, NOT_NEEDED = "PASS", "FAIL", "UNVERIFIED", "NOT_NEEDED"
G_BASE = np.array([0.0, 0.0, -9.80665])


@dataclass
class Provenanced:
    value: object = None
    status: str = "UNKNOWN"         # UNKNOWN | DECLARED | VERIFIED | ESTIMATED
    method: str = ""
    uncertainty: object = None
    date: str = ""


@dataclass
class WrenchCalibration:
    dataset_id: str = ""
    tool_id: str = ""
    schema_version: int = SCHEMA_VERSION
    R_EE_S: Provenanced = field(default_factory=Provenanced)          # 3x3
    p_S_EE: Provenanced = field(default_factory=Provenanced)          # 3, sensor origin in EE (torque transport)
    recorded_quantity: Provenanced = field(default_factory=Provenanced)  # object_on_robot | robot_on_object
    already_compensated: Provenanced = field(default_factory=Provenanced)  # bool: stream zeroed/gravity-free?
    tare_S: Provenanced = field(default_factory=Provenanced)          # 3 N, + noise sigma in uncertainty
    gravity_model: Provenanced = field(default_factory=Provenanced)   # dict(mass_kg, com_S) or None
    tool_mass_upper_bound_kg: Provenanced = field(default_factory=Provenanced)
    clock: Provenanced = field(default_factory=Provenanced)           # dict(offset_s, uncertainty_s)
    tests: dict = field(default_factory=dict)                         # results of the data checks below

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_scene_metadata(cls, scene) -> "WrenchCalibration":
        """Declared conventions from ``metadata/robot_tool.json`` + ``contact_force_convention.json``.

        Declarations become ``DECLARED`` (never ``VERIFIED``)."""
        meta = Path(scene) / "metadata"
        tool = json.loads((meta / "robot_tool.json").read_text())
        conv_p = meta / "contact_force_convention.json"
        conv = json.loads(conv_p.read_text()) if conv_p.is_file() else {}
        wf = tool.get("wrench_frame", {})
        c = cls(dataset_id=tool.get("dataset", Path(scene).name), tool_id=str(tool.get("tool", "")))
        if "R_ee_force_sensor" in wf and wf.get("axes") == "ee":
            c.R_EE_S = Provenanced(np.asarray(wf["R_ee_force_sensor"], float).tolist(), "DECLARED",
                                   f"robot_tool.json wrench_frame: {wf.get('alignment_source', 'unspecified')}")
        torque_origin = str(wf.get("torque_origin", "unspecified"))
        if not torque_origin.startswith("recorded_origin_unspecified") and "p_S_EE" in wf:
            c.p_S_EE = Provenanced(wf["p_S_EE"], "DECLARED", "robot_tool.json")
        else:
            c.p_S_EE = Provenanced(None, "UNKNOWN", f"robot_tool.json: {torque_origin}")
        on = conv.get("force_acting_on")
        rq = "object_on_robot" if on == "robot" else ("robot_on_object" if on == "object" else None)
        if rq:
            c.recorded_quantity = Provenanced(rq, "DECLARED", conv.get("confirmation_source", "contact_force_convention.json"))
        c.already_compensated = Provenanced(None, "UNKNOWN", "no tare/gravity processing declared in metadata")
        return c

    def to_json(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=lambda o: np.asarray(o).tolist()))

    # ------------------------------------------------------------------ data checks
    def estimate_tare(self, F_S, free_mask, label="contact-free window", drift_tol_n=None, second_mask=None,
                      second_label="second contact-free window"):
        """Tare and robust noise on a contact-free window. Drift (zero stability) is tested only
        between two *separated* contact-free windows; with one window the gate is UNVERIFIED."""
        F_all = np.asarray(F_S, float)
        F = F_all[np.asarray(free_mask, bool)]
        F = F[np.isfinite(F).all(1)]
        if len(F) < 10:
            self.tests["tare"] = dict(status=FAIL, reason=f"only {len(F)} contact-free samples in {label}")
            return
        med = np.median(F, 0)
        sigma = 1.4826 * np.median(np.abs(F - med), 0)
        tol = float(drift_tol_n if drift_tol_n is not None else 3 * np.linalg.norm(sigma))
        self.tare_S = Provenanced(med.tolist(), "ESTIMATED", f"median over {len(F)} samples of {label}",
                                  dict(sigma_N=sigma.tolist()))
        entry = dict(n=len(F), sigma_N=sigma.tolist(), window=label, drift_tolerance_N=tol)
        F2 = None if second_mask is None else F_all[np.asarray(second_mask, bool)]
        F2 = None if F2 is None else F2[np.isfinite(F2).all(1)]
        if F2 is None or len(F2) < 10:
            entry.update(status=UNVERIFIED, reason=f"tare estimated on {label}, but no second separated contact-free "
                                                   "window exists to test zero drift")
        else:
            drift = float(np.linalg.norm(np.median(F2, 0) - med))
            entry.update(status=PASS if drift <= tol else FAIL, drift_N=drift, second_window=second_label,
                         n_second=len(F2), reason=f"drift {drift:.3f} N vs tol {tol:.3f} N")
        self.tests["tare"] = entry

    def check_orientation_constancy(self, R_B_EE, interval_mask, max_deg=1.0):
        """Gravity coupling bound when no gravity model exists: orientation change over the interval."""
        R = np.asarray(R_B_EE, float)[np.asarray(interval_mask, bool)]
        rel = np.einsum("ji,njk->nik", R[0], R)
        ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        dmax = float(ang.max()) if len(ang) else np.nan
        mub = self.tool_mass_upper_bound_kg.value
        bound = None if mub is None else float(mub * 9.80665 * 2 * np.sin(np.radians(dmax) / 2))
        self.tests["orientation"] = dict(max_rotation_deg=dmax, threshold_deg=max_deg,
                                         gravity_change_bound_N=bound,
                                         note="bound needs a declared tool-mass upper bound" if bound is None else "")

    def check_static_multipose_gravity(self, F_S, R_B_EE, free_mask, min_rotation_deg=10.0):
        """Fit F = tare + m R_SB g on contact-free samples with orientation diversity (verification of
        R_EE_S and gravity). Unidentifiable (UNVERIFIED) without enough rotation."""
        free = np.asarray(free_mask, bool)
        R = np.asarray(R_B_EE, float)[free]
        F = np.asarray(F_S, float)[free]
        if len(R) < 10:
            self.tests["multipose_gravity"] = dict(status=UNVERIFIED, reason="too few contact-free samples")
            return
        rel = np.einsum("ji,njk->nik", R[0], R)
        ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        if ang.max() < min_rotation_deg:
            self.tests["multipose_gravity"] = dict(
                status=UNVERIFIED, max_rotation_deg=float(ang.max()),
                reason=f"contact-free EE orientations span {ang.max():.2f} deg < {min_rotation_deg} deg: "
                       "tool gravity and sensor rotation are not identifiable from this recording")
            return
        Rs = np.asarray(self.R_EE_S.value if self.R_EE_S.value is not None else np.eye(3), float)
        g_S = np.einsum("ij,njk,k->ni", Rs.T, np.transpose(R, (0, 2, 1)), G_BASE)  # gravity in sensor axes
        A = np.concatenate([np.tile(np.eye(3), (len(R), 1)), g_S.reshape(-1, 1)], 1)
        sol, *_ = np.linalg.lstsq(A, F.reshape(-1), rcond=None)
        res = F.reshape(-1) - A @ sol
        self.tests["multipose_gravity"] = dict(status=PASS if np.sqrt(np.mean(res ** 2)) < 0.2 else FAIL,
                                               max_rotation_deg=float(ang.max()), mass_kg=float(sol[3]),
                                               rms_residual_N=float(np.sqrt(np.mean(res ** 2))))

    # ------------------------------------------------------------------ gates
    def readiness(self, use_torque=False) -> dict:
        g = {}
        mp = self.tests.get("multipose_gravity", {})
        ref = self.tests.get("reference_force_direction", {})
        if self.R_EE_S.value is None:
            g["frame_rotation"] = (FAIL, "R_EE_S unknown: forces cannot be rotated")
        elif self.R_EE_S.status == "VERIFIED" or ref.get("status") == PASS or mp.get("status") == PASS:
            g["frame_rotation"] = (PASS, "verified")
        else:
            g["frame_rotation"] = (UNVERIFIED, f"R_EE_S {self.R_EE_S.status}: {self.R_EE_S.method}; "
                                               "no reference-force or multi-pose test passed")
        if self.recorded_quantity.value is None:
            g["sign"] = (FAIL, "recorded wrench quantity/sign unknown")
        elif self.recorded_quantity.status == "VERIFIED" or ref.get("status") == PASS:
            g["sign"] = (PASS, "verified")
        else:
            g["sign"] = (UNVERIFIED, f"sign {self.recorded_quantity.status} ({self.recorded_quantity.method}); "
                                     "no reference push of known direction")
        t = self.tests.get("tare") or {}
        g["tare"] = (t.get("status", UNVERIFIED), t.get("reason", "no contact-free tare window evaluated"))
        o = self.tests.get("orientation", {})
        sig = np.linalg.norm(self.tests.get("tare", {}).get("sigma_N", [np.inf] * 3))
        if self.gravity_model.status == "VERIFIED" or mp.get("status") == PASS:
            g["gravity"] = (PASS, "verified gravity model")
        elif o.get("gravity_change_bound_N") is not None and o["gravity_change_bound_N"] <= sig:
            g["gravity"] = (PASS, f"orientation constant to {o['max_rotation_deg']:.2f} deg: gravity change "
                                  f"<= {o['gravity_change_bound_N']:.3f} N <= noise {sig:.3f} N")
        else:
            g["gravity"] = (UNVERIFIED, f"no gravity model; orientation change {o.get('max_rotation_deg')} deg, "
                                        f"{o.get('note') or 'bound above noise'}")
        if self.clock.status == "VERIFIED":
            g["time_sync"] = (PASS, self.clock.method)
        else:
            g["time_sync"] = (UNVERIFIED, "exact source_frame_index join only; hardware clock offset not measured")
        if use_torque:
            g["torque_transport"] = (PASS if self.p_S_EE.status == "VERIFIED" else UNVERIFIED, self.p_S_EE.method)
        else:
            g["torque_transport"] = (NOT_NEEDED, "torques not used")
        ready = all(v[0] in (PASS, NOT_NEEDED) for v in g.values())
        return dict(gates={k: dict(status=v[0], reason=v[1]) for k, v in g.items()}, force_selection_ready=ready,
                    failing=[k for k, v in g.items() if v[0] not in (PASS, NOT_NEEDED)])

    # ------------------------------------------------------------------ conversion
    def bias_S(self, R_B_EE=None):
        """``F_bias^S``: tare (+ gravity change w.r.t. the tare pose if a verified model exists)."""
        if self.tare_S.value is None:
            raise ValueError("no tare estimated; refusing to compensate")
        b = np.asarray(self.tare_S.value, float)
        gm = self.gravity_model.value
        if gm is not None and self.gravity_model.status == "VERIFIED":
            if R_B_EE is None:
                raise ValueError("gravity model needs the EE orientation")
            Rs = np.asarray(self.R_EE_S.value, float)
            R = np.asarray(R_B_EE, float)
            w = gm["mass_kg"] * np.einsum("ij,njk,k->ni", Rs.T, np.transpose(R, (0, 2, 1)), G_BASE)
            ref = gm["mass_kg"] * Rs.T @ np.asarray(gm["tare_pose_R_B_EE"], float).T @ G_BASE
            return b[None] + (w - ref[None])
        return np.broadcast_to(b, (1, 3))

    def force_on_object_world(self, F_rec_S, R_W_EE, R_B_EE=None) -> dict:
        """Contact force by the robot on the object in world axes, plus the raw channel."""
        if self.R_EE_S.value is None:
            raise ValueError("R_EE_S unknown: refusing to rotate forces")
        from .force_convention import sign_from_recorded_quantity        # the one shared sign contract
        sign = sign_from_recorded_quantity(self.recorded_quantity.value)
        F = np.asarray(F_rec_S, float)
        Fc = F - self.bias_S(R_B_EE)
        Rs = np.asarray(self.R_EE_S.value, float)
        R = np.asarray(R_W_EE, float)
        f_w = sign * np.einsum("nij,jk,nk->ni", R, Rs, Fc)
        raw_w = sign * np.einsum("nij,jk,nk->ni", R, Rs, F)
        return dict(f_object_w=f_w, f_object_w_raw_uncompensated=raw_w, f_sensor_compensated=Fc, sign=sign)

    def torque_on_object_about_tcp_world(self, F_rec_S, tau_rec_S, R_W_EE, p_W_EE, p_W_tcp):
        """``tau_TCP = s R (tau_S) + (p_S - p_TCP) x f_obj``; needs a known sensor origin."""
        if self.p_S_EE.value is None:
            raise ValueError("sensor/torque origin unknown: refusing to transport torques")
        conv = self.force_on_object_world(F_rec_S, R_W_EE)
        Rs = np.asarray(self.R_EE_S.value, float)
        R = np.asarray(R_W_EE, float)
        tau = conv["sign"] * np.einsum("nij,jk,nk->ni", R, Rs, np.asarray(tau_rec_S, float))
        p_S_W = np.asarray(p_W_EE, float) + np.einsum("nij,j->ni", R, np.asarray(self.p_S_EE.value, float))
        return tau + np.cross(p_S_W - np.asarray(p_W_tcp, float), conv["f_object_w"])
