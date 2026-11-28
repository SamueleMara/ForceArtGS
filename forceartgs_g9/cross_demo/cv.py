"""Cross-demo protocols (spec section 7).

single_demo_transfer (primary): for every training episode i (frozen theta_i, URDF_i, joint_i, method m) and every
target j of the SAME physical object, replay j's own measured input once; i == j is TRAIN_DIAGONAL, i != j
HELDOUT_OTHER_DEMO (N x N ordered cells, N(N-1) held-out). theta_i is loaded ONCE per (i, m); nothing is fitted on j.
pooled_loo (secondary): POOLED_LOO_NOT_IMPLEMENTED (a pooled multi-episode G10 fit is not part of the current
backend; per-episode parameter averaging is NOT substituted for it).

Per pair: per_pair/<object>/train_<i>/test_<j>/<method>/seed0/{STATUS,provenance,metrics}.json + replay.npz +
residuals.npz. Resume: a pair is reused only if its key (train model hash, target record hash, scope, config hash,
metrics version) is identical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import heldout_episode as H
from . import kinematics_alignment as K
from . import metrics as MX
from . import replay as R

PROTOCOL = "single_demo_transfer"


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _write_json(p: Path, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(d, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    tmp.replace(p)


def _eid(name: str) -> str:
    return name.rsplit("_episode_", 1)[-1] if "_episode_" in name else name


def travel_threshold(rec, k):
    s = np.asarray(rec["q_sigma"], float)
    s = s[np.isfinite(s) & (s > 0)]
    return float(k * np.median(s)) if len(s) else None


def score_pair(h: H.HeldoutEpisode, roll: dict, joint_type: str, cfg: dict, rec: dict, tail_ok: bool) -> dict:
    seg = dict(t_motion_start=h.segments.get("MOVING_INTERVAL", (None,))[0])
    kw = dict(q_sigma=h.q_sigma, qdot_sim=roll["qdot_sim"], joint_type=joint_type,
              sigma_floor=float(cfg["scoring"]["sigma_physx_floor"]),
              travel_threshold=travel_threshold(rec, float(cfg["scoring"]["travel_noise_k"])), segments=seg)
    sc = MX.score_scopes(h.t_s, h.q_ref, roll["q_sim"], t_stop=h.T_stop, tail_input_complete=tail_ok, **kw)
    if h.q_true is not None:                                     # synthetic: the same prediction against exact GT
        sc["TRUE_GT"] = MX.score_trajectory(h.t_s, h.q_true, roll["q_sim"], **kw)
    return sc


def run_pairs(object_id: str, episodes: dict, method: str, cfg: dict, out: Path, *, records=None, models=None,
              train_filter=None, test_filter=None, resume=True, log=print) -> list:
    """episodes: {episode_name: r2r_dir | None (not reconstructed / no estimate) | dict(synthetic=..)}.
    records / models: optional caches or injected synthetic objects. Returns one cell per ordered (i, j)."""
    records = {} if records is None else records
    models = {} if models is None else models
    names = sorted(episodes)
    cfg_hash = _key(cfg)
    cells = []

    def rec_of(e):
        if e not in records:
            records[e] = H.load_real_record(episodes[e], scene=e)
        return records[e]

    for i in names:
        if train_filter and i != train_filter:
            continue
        base = dict(object=object_id, train_id=i, method=method, seed=0, protocol=PROTOCOL)
        model, err = models.get((i, method)), None
        if model is None:
            if episodes[i] is None:
                err = ("TRAIN_UNAVAILABLE", "episode has no finished rest-to-rest estimate")
            else:
                try:
                    model = models[(i, method)] = R.load_train_model(episodes[i], method, name=i)
                except (R.TrainModelError, FileNotFoundError, KeyError) as ex:
                    err = ("TRAIN_FIT_INVALID", f"{type(ex).__name__}: {ex}")
        if err:
            cells += [dict(base, test_id=j, is_train_diagonal=i == j, status=err[0], reason=err[1]) for j in names
                      if not test_filter or j == test_filter]
            continue
        eng = R.FrozenEngine(model)
        try:
            parity = R.diagonal_parity(eng) if model.r2r_dir and Path(model.r2r_dir, "STATUS.json").is_file() \
                and (Path(model.r2r_dir) / "rest_to_rest_episode.npz").is_file() else dict(status="PARITY_NOT_EVALUATED")
        except Exception as ex:
            parity = dict(status="PARITY_FAILED", error=f"{type(ex).__name__}: {ex}")
        for j in names:
            if test_filter and j != test_filter:
                continue
            cell = dict(base, test_id=j, is_train_diagonal=i == j, joint_type=model.joint_type,
                        label="TRAIN_DIAGONAL" if i == j else "HELDOUT_OTHER_DEMO")
            pdir = out / "per_pair" / object_id / f"train_{_eid(i)}" / f"test_{_eid(j)}" / method / "seed0"
            if episodes[j] is None:
                cells.append(dict(cell, status="TEST_UNAVAILABLE", reason="target has no reconstructed record"))
                continue
            try:
                rec = rec_of(j)
            except (FileNotFoundError, KeyError, ValueError) as ex:
                cells.append(dict(cell, status="TEST_RECORD_INVALID", reason=f"{type(ex).__name__}: {ex}"))
                continue
            key = _key(model.describe(), rec["record_sha256"], cfg_hash, MX.METRICS_VERSION)
            if resume and (pdir / "STATUS.json").is_file():
                st = json.loads((pdir / "STATUS.json").read_text())
                if st.get("key") == key:
                    cells.append(st["cell"])
                    continue
            al = K.align(model.geometry(), dict(joint_type=rec["joint_type"], axis_B=rec["axis_B"],
                                                origin_B=rec["origin_B"]), **cfg["alignment"])
            h = H.build(model.geometry(), rec, al, scope="eval_end")
            tail_ok = h.status == "OK"
            if h.status == "TAIL_INPUT_INCOMPLETE":
                h = H.build(model.geometry(), rec, al, scope="stop")
            cell.update(alignment={k: v for k, v in al.items() if k != "gauge"})
            if h.status != "OK":
                cell.update(status=h.status if h.status != "KINEMATICS_TYPE_MISMATCH" else "KINEMATICS_TYPE_MISMATCH",
                            reason=h.reason)
                _write_json(pdir / "STATUS.json", dict(key=key, cell=cell))
                cells.append(cell)
                continue
            try:
                roll = eng.rollout(h.episode)
            except Exception as ex:
                cell.update(status="REPLAY_FAILED", reason=f"{type(ex).__name__}: {ex}")
                _write_json(pdir / "STATUS.json", dict(key=key, cell=cell))
                cells.append(cell)
                continue
            sc = score_pair(h, roll, model.joint_type, cfg, rec, tail_ok)
            prim = sc[sc["primary"]]
            if prim.get("status") != "OK":
                cell.update(status="SCORING_FAILED", reason=prim.get("reason"))
            else:
                cell.update(status="OK", scope=sc["primary"], coverage=sc["coverage"],
                            metrics={k: v for k, v in prim.items() if not isinstance(v, (dict, list))},
                            M=model.M, c=model.c, chosen_method=model.chosen_method,
                            duration_s=prim.get("duration_s"), ref_kind=rec["ref_kind"],
                            parity=parity.get("status") if i == j else None)
            res = MX.residual_arrays(h.t_s, h.q_ref, roll["q_sim"], q_sigma=h.q_sigma,
                                     sigma_floor=float(cfg["scoring"]["sigma_physx_floor"]), qdot_sim=roll["qdot_sim"],
                                     segments=dict(t_motion_start=h.segments.get("MOVING_INTERVAL", (None,))[0],
                                                   t_stop=h.T_stop), source_frame=h.source_frame)
            pdir.mkdir(parents=True, exist_ok=True)
            np.savez(pdir / "replay.npz", t_s=h.t_s, q_sim=roll["q_sim"],
                     qdot_sim=roll["qdot_sim"] if roll["qdot_sim"] is not None else np.full(len(h.t_s), np.nan),
                     q_ref=h.q_ref, q_sigma=h.q_sigma, q_true=h.q_true if h.q_true is not None else np.full(len(h.t_s), np.nan),
                     input_t_s=h.input_t_s, input_Q=h.input_Q, t0=h.t0, T_stop=h.T_stop, T_eval_end=h.T_eval_end)
            np.savez(pdir / "residuals.npz", **{k: np.asarray(v) for k, v in res.items()})
            _write_json(pdir / "metrics.json", sc)
            _write_json(pdir / "provenance.json", dict(train=model.describe(), test=h.provenance, alignment=al,
                                                       parity=parity if i == j else None,
                                                       engine=dict(backend=roll["engine"], version=roll["engine_version"],
                                                                   axis_inertia_engine=roll["axis_inertia_engine"],
                                                                   initial_q_error=roll["initial_q_error"]),
                                                       no_fit_on_target=True, used_for_training=False))
            _write_json(pdir / "STATUS.json", dict(key=key, cell=cell))
            cells.append(cell)
        eng.close()
        log(f"[cross-demo] {object_id} {method} train {_eid(i)}: "
            f"{sum(c['status'] == 'OK' for c in cells if c['train_id'] == i)}/{len(names)} cells OK")
    return cells


def pooled_loo(*_a, **_k) -> dict:
    return dict(status="POOLED_LOO_NOT_IMPLEMENTED",
                reason="the current backend fits one episode at a time; a pooled N-1 episode G10 fit would be a new "
                       "fitting feature. Per-episode parameter averaging is NOT reported as LOO.")
