"""Frozen-model PhysX replay (spec section 6): training episode i's (M_i, c_i), URDF and engine settings, the target's
measured input. NO optimizer anywhere in this module: no fit_full_demo, local refinement, differential evolution,
PINN, profiling or parameter selection on a target (tests monkeypatch those to raise).

  TrainModel   frozen from ONE finished rest-to-rest run of episode i (exact directory): the fitted G10 parameters of
               a PREDECLARED method (or `train_best_predeclared`: the lowest TRAINING objective among a fixed list,
               decided on episode i's own fit only), its reference.urdf (sha-checked), joint geometry in base frame B,
               engine timestep / friction mode / gravity off.
  replay()     one SAPIEN/PhysX rollout per (train, target) pair, CPU engine in-process (1-DoF; the GPU batch of one is
               slower), the engine reads back the scaled axis inertia every rollout (IsolatedSapienCpu checks).
  diagonal_parity()  re-runs episode i's own training episode with the frozen theta and compares with the metrics the
               training run saved (same engine model, effort policy and parameter mapping; CPU vs the training's GPU
               engine: agreement within a stated tolerance).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

G10_METHODS = ("B0_G10_BASELINE_STARTS", "B7_G10_SVD_START", "B8_G10_CONVOLUTION_START", "B9_G10_INTEGRAL_PINN_START",
               "B789_G10_UNION_PROPOSED_STARTS")
PREDECLARED_BEST_OF = G10_METHODS            # train_best_predeclared chooses among these by the TRAINING objective only


class TrainModelError(ValueError):
    pass


def _sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@dataclass
class TrainModel:
    name: str
    r2r_dir: str
    method: str
    chosen_method: str
    params: dict
    M: float
    c: float
    rho_scale: float
    axis_inertia: float
    joint_type: str
    axis_B: np.ndarray
    origin_B: np.ndarray | None
    urdf: str
    urdf_sha256: str
    moving_link: str
    joint_name: str
    timestep: float
    friction_mode: str
    training_objective: float
    at_bound: dict
    episode_sha256: str

    def geometry(self) -> dict:
        return dict(name=self.name, joint_type=self.joint_type, axis_B=self.axis_B, origin_B=self.origin_B)

    def describe(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("axis_B", "origin_B")}
        d.update(axis_B=self.axis_B.tolist(), origin_B=None if self.origin_B is None else self.origin_B.tolist(),
                 M_units="kg m^2 (axis inertia)" if self.joint_type == "revolute" else "kg",
                 c_units="N m s/rad" if self.joint_type == "revolute" else "N s/m",
                 gravity="zero", coulomb="off", armature=self.params.get("armature", 0.0))
        return d


def load_train_model(r2r_dir, method="B0_G10_BASELINE_STARTS", *, name=None) -> TrainModel:
    """Fails closed (TrainModelError) on FIT_FAILED / ROLLOUT_FAILED / non-finite / non-physical parameters, a URDF
    whose hash differs from the training record, or a missing training artifact."""
    import yaml
    d = Path(r2r_dir)
    st = json.loads((d / "STATUS.json").read_text())
    if st.get("status") != "COMPARED":
        raise TrainModelError(f"{d}: training status {st.get('status')}: {st.get('reason')}")
    px = json.loads((d / "physx_comparison.json").read_text())
    fits = px.get("fits") or {}
    chosen = method
    if method == "train_best_predeclared":
        ok = {m: f for m, f in fits.items() if m in PREDECLARED_BEST_OF and f.get("status") == "FITTED"
              and math.isfinite(float(f.get("objective", math.nan)))}
        if not ok:
            raise TrainModelError(f"{d}: no FITTED method among {PREDECLARED_BEST_OF}")
        chosen = min(ok, key=lambda m: (float(ok[m]["objective"]), m))     # TRAINING objective of episode i only
    f = fits.get(chosen)
    if f is None:
        raise TrainModelError(f"{d}: method {chosen} not in the training run")
    if f.get("status") != "FITTED":
        raise TrainModelError(f"{d}: {chosen} {f.get('status')}: {f.get('error') or f.get('reason')}")
    p = dict(f["params"])
    vals = [p.get(k) for k in ("rho_scale", "viscous", "armature", "static_effort", "dynamic_effort")]
    if not all(v is not None and math.isfinite(float(v)) for v in vals):
        raise TrainModelError(f"{d}: {chosen} non-finite parameters {p}")
    if float(p["rho_scale"]) <= 0 or float(p["viscous"]) < 0:
        raise TrainModelError(f"{d}: {chosen} non-physical parameters rho {p['rho_scale']}, c {p['viscous']}")
    info = json.loads((d / "urdf_info.json").read_text())
    urdf = d / "reference.urdf"
    sha = _sha_file(urdf)
    if sha != info["sha256"]:
        raise TrainModelError(f"{urdf}: sha256 differs from the training record")
    fw = np.load(d / "full_wrench_joint_effort.npz", allow_pickle=False)
    phys = (yaml.safe_load((d / "resolved_config.yaml").read_text()).get("simplified_g10_resolved") or {})["physics"]
    M = float(p["rho_scale"]) * float(info["axis_inertia"])
    if not (M > 0 and math.isclose(M, float(f["M"]), rel_tol=1e-9)):
        raise TrainModelError(f"{d}: M = rho x axis_inertia = {M} disagrees with the recorded {f['M']}")
    ep = np.load(d / "rest_to_rest_episode.npz", allow_pickle=False)
    return TrainModel(name=name or str(ep["scene"]), r2r_dir=str(d), method=method, chosen_method=chosen, params=p,
                      M=M, c=float(p["viscous"]), rho_scale=float(p["rho_scale"]),
                      axis_inertia=float(info["axis_inertia"]), joint_type=info["joint_type"],
                      axis_B=fw["axis_B"].astype(float),
                      origin_B=fw["origin_B"].astype(float) if info["joint_type"] == "revolute" else None,
                      urdf=str(urdf), urdf_sha256=sha, moving_link=info["moving_link"], joint_name=info["joint_name"],
                      timestep=float(phys["timestep_s"]), friction_mode=phys["friction_mode"],
                      training_objective=float(f["objective"]), at_bound=f.get("at_bound") or {},
                      episode_sha256=str(ep["sha256"]))


class FrozenEngine:
    """One engine per training model (its URDF), reused for every target: rollouts only."""

    def __init__(self, model: TrainModel, python=None):
        from ..physx_refinement.sapien_diagnostic import IsolatedSapienCpu
        self.model = model
        self.sim = IsolatedSapienCpu(model.urdf, moving_link=model.moving_link, joint_name=model.joint_name,
                                     dt=model.timestep, gravity=[0.0, 0.0, 0.0], friction_mode=model.friction_mode,
                                     python=python)
        self.theta = self._theta()

    def _theta(self):
        from ..physx_refinement.objective import PhysicalParameters
        p = self.model.params
        return PhysicalParameters(rho_scale=float(p["rho_scale"]), armature=float(p.get("armature", 0.0)),
                                  static_effort=float(p.get("static_effort", 0.0)),
                                  dynamic_effort=float(p.get("dynamic_effort", 0.0)), viscous=float(p["viscous"]),
                                  inertia_scale_source="FROZEN_FROM_TRAINING")

    def rollout(self, episode) -> dict:
        if getattr(episode, "used_for_training", True):
            raise ValueError("a transfer target must be an Episode with used_for_training=False")
        r = self.sim(self.theta, episode)
        q = np.asarray(r.q, float)
        qd = None if r.qdot is None else np.asarray(r.qdot, float)
        if not np.all(np.isfinite(q)) or (qd is not None and not np.all(np.isfinite(qd))):
            raise RuntimeError("non-finite PhysX rollout")
        rb = r.diagnostics or {}
        if not math.isclose(float(rb["axis_inertia_engine"]), self.model.M, rel_tol=1e-4, abs_tol=1e-9):
            raise RuntimeError(f"engine axis inertia {rb['axis_inertia_engine']} != frozen M {self.model.M}")
        return dict(q_sim=q, qdot_sim=qd, engine=r.backend, engine_version=r.engine_version,
                    axis_inertia_engine=float(rb["axis_inertia_engine"]), initial_q_error=float(q[0] - episode.q0),
                    readback=rb.get("readback"))

    def close(self):
        self.sim.close()


def diagonal_parity(engine: FrozenEngine) -> dict:
    """Episode i's own training episode, frozen theta_i, this engine -> the metrics its training run saved."""
    from ..estimators.rest_to_rest import RestToRestEpisode
    from ..physx_refinement.rest_to_rest_comparison import physx_metrics
    d = Path(engine.model.r2r_dir)
    import dataclasses
    ev = json.loads((d / "rest_endpoint_evidence.json").read_text())
    ep = RestToRestEpisode.load_npz(d / "rest_to_rest_episode.npz", endpoint_evidence=ev)
    px = json.loads((d / "physx_comparison.json").read_text())
    saved = px["fits"][engine.model.chosen_method]["metrics"]
    pe = dataclasses.replace(ep.to_physx_episode(), used_for_training=False)
    r = engine.sim(engine.theta, pe)
    import yaml
    floor = float((yaml.safe_load((d / "resolved_config.yaml").read_text())["simplified_g10_resolved"]
                   .get("loss") or {}).get("numeric_floor_q", 0.0))
    m = physx_metrics(r, pe, ep, floor, float(px["sigma_vT"]["sigma_vT"]))
    keys = ("rmse", "rmse_over_sigma", "q_sim_T_minus_qT", "v_sim_T")
    diff = {k: abs(float(m[k]) - float(saved[k])) for k in keys if m.get(k) is not None and saved.get(k) is not None}
    rel = {k: diff[k] / max(abs(float(saved[k])), 1e-12) for k in diff}
    return dict(status="PARITY_OK" if max(rel.values(), default=1.0) < 2e-2 else "PARITY_MISMATCH",
                saved={k: saved.get(k) for k in keys}, replayed={k: m.get(k) for k in keys}, abs_diff=diff,
                rel_diff=rel, tolerance_rel=2e-2,
                note="training ran the batched PhysX GPU engine, this replay the in-process CPU engine (same model)")
