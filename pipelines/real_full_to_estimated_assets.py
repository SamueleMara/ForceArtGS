#!/usr/bin/env python
"""Every real episode_000000, from scratch, through the current pipeline to URDF + USD with the ESTIMATED joint dynamics.

Per scene (sequential; resumable -- finished steps are skipped, interrupted 3DGS training resumes from its checkpoint):
  1. prep + reconstruction   pipelines/joint_aware_all_real.sh full <OUT>  (force export, clip, contact segments,
                             discovery, multimodal frontend, force-guided automatic joint selection, ONLY the working
                             branch, blend / art / final, G8 hand-off + diagnostic visual asset)
  2. q(t) extension          pipelines/extend_joint_states_real.sh   (frozen joint, whole recording)
  3. parameter estimation    pipelines/g9_rest_to_rest_comparison_real.sh  (M, c: analytics, PINNs, PhysX replay)
  4. URDF + USD              forceartgs_g9.cli estimated-asset  (joint damping c + moving inertia M along the joint
                             ONLY; replay-verified in SAPIEN, USD read back)
Outputs (SSD only): <OUT>/{forceartgs_joint_aware, joint_states_extended, rest_to_rest_compare, estimated_assets}
and <OUT>/estimated_assets/SUMMARY.md. Only the AUTOMATIC joint decision is trained (no operator declaration anywhere;
a MODEL_INCONSISTENT scene has no best branch and is not trained).

    python pipelines/suspend_resilient.py -- python pipelines/real_full_to_estimated_assets.py [--out OUT] [--scenes ...]
    --budget smoke (300 it/stage, QUICK rest-to-rest): end-to-end plumbing check of the identical chain.
A reconstruction stage that dies on a Python programming error (NameError, ImportError, ...) stops the WHOLE batch:
every other scene would die the same way.

Parallel (SLURM array, slurm/): every job runs ONE scene with the same --out; each scene writes
estimated_assets/<scene>/row.json and SUMMARY.md is rebuilt from all of them (atomic), or at the end by
--summarize-only. --episodes all = every *_episode_XXXXXX, not only 000000; --list-scenes prints them.
Paths: FORCEARTGS_REAL_ROOT (dataset), FORCEARTGS_OUTPUT_ROOT (allowed output prefix), FORCEARTGS_CONTACT_DIR
(contact segments; default the repo's configs/contact_intervals), FORCEARTGS_PYTHON (the forceartgs interpreter).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REAL = Path(os.environ.get("FORCEARTGS_REAL_ROOT",                 # server: export the dataset root
                           "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"))
OUTPUT_ROOT = os.environ.get("FORCEARTGS_OUTPUT_ROOT", "/media/samuelemara/Extreme SSD/")   # allowed output prefix
PY = sys.executable                                               # the forceartgs env (run the driver with it)
FIRST = ["final_red_fridge_episode_000000", "final_orange_suitecase_episode_000000", "final_black_box_episode_000000",
         "final_robot_dog_episode_000000", "white_drawer_20261021_161154_episode_000000"]


def log(out: Path, msg: str) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(out / "real_full_to_estimated_assets.log", "a") as f:
        f.write(line + "\n")


def sh(cmd, env_extra, logfile: Path) -> int:
    env = dict(os.environ, **env_extra)
    with open(logfile, "a") as f:
        f.write(f"\n### {time.strftime('%F %T')} {' '.join(map(str, cmd))} env={env_extra}\n")
        f.flush()
        return subprocess.run([str(c) for c in cmd], cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def npz_ok(p: Path) -> bool:
    import zipfile
    try:
        with zipfile.ZipFile(p) as z:
            return z.testzip() is None
    except Exception:
        return False


CODE_ERRORS = ("NameError", "ImportError", "ModuleNotFoundError", "SyntaxError", "AttributeError", "TypeError",
               "UnboundLocalError", "IndentationError")


ASSET_SCHEMA = "forceartgs.g9.estimated_joint_dynamics_asset/v2"   # + the 3DGS parts next to the URDF


class CodeError(RuntimeError):
    """A pipeline stage died on a programming error: stop the batch (every scene would hit it)."""


def g10_fits(d):
    """(failed {fit: error}, n fits) of a rest-to-rest result: the G10 PhysX fits catch their exceptions as FIT_FAILED,
    so a crash there would otherwise pass as COMPARED (coarse fridge 27/11: every fit lost a temp URDF)."""
    f = None if d is None else Path(d) / "physx_comparison.json"
    if f is None or not f.is_file():
        return {}, 0
    px = json.loads(f.read_text())
    fits = dict(px.get("fits") or {})
    if isinstance(px.get("C0"), dict) and px["C0"].get("status") != "SKIPPED":
        fits["C0"] = px["C0"]
    return ({k: str(v.get("error") or v.get("reason") or "") for k, v in fits.items()
             if str(v.get("status", "")).endswith("FAILED")}, len(fits))


def code_bugs(failed: dict) -> dict:
    """Failures that are programming errors (or a vanished engine file), not a property of the data."""
    pat = re.compile(r"\b(" + "|".join(CODE_ERRORS + ("FileNotFoundError",)) + r")\b")
    return {k: e for k, e in failed.items() if pat.search(e)}


def run_dir(out: Path, sc: str, budget: str):
    c = sorted((out / "forceartgs_joint_aware").glob(f"{sc}_multiview_*_joint_aware_{budget}"))
    return c[0] if c else None


def reconstruction_failure(out: Path, sc: str, budget: str):
    """(failed stage line, exception line) of the LATEST pass in <RUNS>/<sc>.<budget>.log (a resumed run appends to it;
    an earlier pass's failure must not condemn a pass that got past it); the batch script itself always exits 0."""
    lg = out / "forceartgs_joint_aware" / f"{sc}.{budget}.log"
    if not lg.is_file():
        return "no reconstruction log", None
    lines = lg.read_text(errors="replace").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.rstrip().endswith("== audit")]   # each (resumed) pass starts here
    last_pass = lines[starts[-1]:] if starts else lines                                # a resumed pass appends: judge it alone
    fails = [ln for ln in last_pass if " FAILED " in ln]
    if not fails:
        return None, None
    exc = None
    m = re.search(r"see (/.+?\.log)", fails[-1])
    if m and Path(m.group(1)).is_file():
        tb = [ln for ln in Path(m.group(1)).read_text(errors="replace").splitlines()
              if re.match(r"^[A-Za-z_.]*(Error|Exception)\b", ln)]
        exc = tb[-1] if tb else None
    return fails[-1].strip()[:300], exc


def latest_status(root: Path):
    for d in sorted(root.glob("v[0-9][0-9][0-9]"), reverse=True):
        if (d / "STATUS.json").is_file():
            return d, json.loads((d / "STATUS.json").read_text())
    return None, None


def fresh(result: Path, *inputs) -> bool:
    """mtime make rule: only right after the step that produced `result` (see run_inputs for why not on resume)."""
    if result is None or not Path(result).exists():
        return False
    t = Path(result).stat().st_mtime
    return all(t >= Path(i).stat().st_mtime for i in inputs if i is not None and Path(i).exists())


DECISION_KEYS = ("status", "selected", "working_motion_model", "physics_joint_export_ready", "failure_codes")
JOINT_FILES = tuple(f"{d}/{f}" for d in ("discovery/fit", "multimodal_refine/accepted")
                    for f in ("joint_candidates.json", "joint_states.npz", "model_selection.json",
                              "part_support/track_support.npz"))


def run_inputs(run: Path) -> dict:
    """What steps 2-3 consume from the reconstruction, by CONTENT: the selection decision and the joint files. Not the
    mtime of selection.json: every resumed step-1 pass recomputes the compare stage and rewrites it with the same
    decision, which made every finished step 2-3 look stale (coarse fridge resume 27/11)."""
    import hashlib
    f = Path(run) / "selection.json"
    sel = json.loads(f.read_text()) if f.is_file() else {}
    return dict(decision={k: sel.get(k) for k in DECISION_KEYS},
                joint_files={j: hashlib.sha256((Path(run) / j).read_bytes()).hexdigest()
                             for j in JOINT_FILES if (Path(run) / j).is_file()})


def _inputs_path(out: Path, sc: str) -> Path:
    return out / "logs" / f"{sc}.inputs.json"


def recorded(out: Path, sc: str) -> dict:
    f = _inputs_path(out, sc)
    return json.loads(f.read_text()) if f.is_file() else {}


def record(out: Path, sc: str, step: str, **kw) -> None:
    d = recorded(out, sc)
    d[step] = kw
    f = _inputs_path(out, sc)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(f)


def authorities(out: Path, sc: str) -> list:
    return sorted((out / "joint_states_extended" / sc).glob("authority_v[0-9][0-9][0-9]/authority.json"))


def adopt(out: Path, sc: str, run) -> None:
    """Results made before inputs were recorded (by an older driver): BEFORE step 1 may rewrite selection.json, the
    mtime rule is still valid for them -- record their inputs now so the content rule can take over."""
    if run is None or not (Path(run) / "selection.json").is_file():
        return
    rec, sel = recorded(out, sc), Path(run) / "selection.json"
    done = [a for a in authorities(out, sc) if fresh(a, sel)]
    if done and "step2" not in rec:
        record(out, sc, "step2", inputs=run_inputs(run), result=str(done[-1]), adopted=True)
    d, st = latest_status(out / "rest_to_rest_compare" / sc / "rest_to_rest_compare")
    if d is not None and "step3" not in rec and fresh(d / "STATUS.json", sel, done[-1] if done else None):
        record(out, sc, "step3", inputs=run_inputs(run), authority=str(done[-1]) if done else None, result=str(d),
               adopted=True)


def scene(out: Path, sc: str, args) -> dict:
    row = dict(scene=sc)
    # ---------------------------------------------------------------- 1. prep + full reconstruction (resumable)
    extra = "--force-guided-selection --working-branch-only"          # the automatic decision's branch only
    adopt(out, sc, run_dir(out, sc, args.budget))
    log(out, f"{sc} [1/4] reconstruction budget={args.budget} ({extra})")
    rc = sh(["bash", "pipelines/joint_aware_all_real.sh", args.budget, out], dict(SCENES=sc, EXTRA_ARGS=extra),
            out / "logs" / f"{sc}.1_reconstruction.log")
    fail, exc = reconstruction_failure(out, sc, args.budget)
    if fail:
        rc = rc or 1
        row["step"] = f"reconstruction FAILED: {exc or fail}"
        log(out, f"{sc} [1/4] {fail}" + (f" || {exc}" if exc else ""))
        if exc and exc.split(":")[0].split(".")[-1] in CODE_ERRORS:
            raise CodeError(f"{sc}: {exc}")
    run = run_dir(out, sc, args.budget)
    row["run"] = str(run) if run else None
    if run and (run / "selection.json").is_file():
        s = json.loads((run / "selection.json").read_text())
        row["selection"] = f"{s.get('status')}/{s.get('selected') or s.get('working_motion_model')}"
    if run is None or rc != 0:
        row["step"] = row.get("step") or (f"reconstruction exit {rc}" + ("" if run else " (no run dir)"))
        log(out, f"{sc} [1/4] {row['step']}: later steps attempted only if a run exists")
        if run is None:
            return row
    # ---------------------------------------------------------------- 2. q(t) on the whole recording
    ext = out / "joint_states_extended"
    sel = run / "selection.json"
    inp = run_inputs(run)
    r2 = recorded(out, sc).get("step2") or {}
    done = [Path(r2["result"])] if r2.get("inputs") == inp and Path(r2.get("result", "")).is_file() else []
    if not done:
        for f in (ext / sc).glob("obs_cam_*.npz"):
            if not npz_ok(f):                                     # killed mid-write (suspend): redo that camera
                f.rename(f.with_suffix(f".npz.incomplete_{int(time.time())}"))
        log(out, f"{sc} [2/4] q(t) extension")
        sh(["bash", "pipelines/extend_joint_states_real.sh", ext],
           dict(SCENES=sc, RUNS=str(out / "forceartgs_joint_aware"), DISC_ROOT=str(out / "no_external_archive"),
                NO_DECLARATION="1", JA_BUDGET=args.budget),
           out / "logs" / f"{sc}.2_extension.log")
        done = [a for a in authorities(out, sc) if fresh(a, sel)]    # made by this pass
        if done:
            record(out, sc, "step2", inputs=inp, result=str(done[-1]))
    row["extended_authority"] = str(done[-1].parent) if done else None
    if not done:
        log(out, f"{sc} [2/4] no extended authority (see logs/{sc}.2_extension.log): step 3 uses the clip q only")
    # ---------------------------------------------------------------- 3. rest-to-rest parameter estimation
    r2r_root = out / "rest_to_rest_compare" / sc / "rest_to_rest_compare"
    d, st = latest_status(r2r_root)
    stale_bug = code_bugs(g10_fits(d)[0])
    if stale_bug:
        log(out, f"{sc} [3/4] earlier result has G10 fits that died on a code error: recomputed "
                 f"({next(iter(stale_bug.values()))[:120]})")
    r3 = recorded(out, sc).get("step3") or {}
    same = (r3.get("inputs") == inp and r3.get("result") == str(d)
            and r3.get("authority") == (str(done[-1]) if done else None))
    if st is None or st.get("status") != "COMPARED" or stale_bug or not same:
        log(out, f"{sc} [3/4] rest-to-rest parameter estimation")
        sh(["bash", "pipelines/g9_rest_to_rest_comparison_real.sh", out / "forceartgs_joint_aware"],
           dict(SCENES=sc, OUT_BASE=str(out / "rest_to_rest_compare"), EXT_ROOT=str(ext), PINN_DEVICE=args.pinn_device,
                DISC_ROOT=str(out / "no_external_archive"), NO_DECLARATION="1", JA_BUDGET=args.budget,
                **(dict(QUICK="1") if args.budget == "smoke" else {})),
           out / "logs" / f"{sc}.3_rest_to_rest.log")
        d, st = latest_status(r2r_root)
        if d is not None:
            record(out, sc, "step3", inputs=inp, authority=str(done[-1]) if done else None, result=str(d))
    row["r2r"] = str(d) if d else None
    row["r2r_status"] = None if st is None else f"{st.get('status')}: {st.get('reason')}"
    failed, n_fits = g10_fits(d)
    row["g10_fits"] = f"{n_fits - len(failed)}/{n_fits} fitted" if n_fits else "-"
    if failed:
        log(out, f"{sc} [3/4] G10 fits failed: " + "; ".join(f"{k}: {e[:100]}" for k, e in failed.items()))
    bug = code_bugs(failed)
    if bug:
        k, e = next(iter(bug.items()))
        raise CodeError(f"{sc}: G10 fit {k}: {e}")
    if st is None or st.get("status") != "COMPARED" or not (st.get("best_row") or {}).get("M"):
        log(out, f"{sc} [3/4] no PhysX-replayed estimate ({row['r2r_status']}): no URDF / USD")
        return row
    b = st["best_row"]
    row.update(joint_type=st["episode"]["joint_type"], M=b["M"], c=b["c"], method=b.get("method"),
               rmse_over_sigma=b.get("physx_rmse_over_sigma"))
    # ---------------------------------------------------------------- 4. URDF + USD with the estimate
    a_root = out / "estimated_assets" / sc
    ad, ast = latest_status(a_root)
    if ast is None or ast.get("r2r_dir") != str(d) or ast.get("schema") != ASSET_SCHEMA:   # older asset: re-export
        log(out, f"{sc} [4/4] URDF + USD (M={b['M']:.4g}, c={b['c']:.4g})")
        sh([PY, "-m", "forceartgs_g9.cli", "estimated-asset", "--run", run, "--r2r", d, "--out-root", a_root], {},
           out / "logs" / f"{sc}.4_asset.log")
        ad, ast = latest_status(a_root)
    row["asset_status"] = None if ast is None else f"{ast['status']}: {ast['reason']}"
    row["urdf"], row["usd"] = (ast or {}).get("urdf"), (ast or {}).get("usd")
    log(out, f"{sc} done: {row.get('asset_status')}")
    return row


def save_row(out: Path, row: dict) -> None:
    d = out / "estimated_assets" / row["scene"]
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".row.{os.getpid()}.json"
    tmp.write_text(json.dumps(row, indent=1, default=str))
    tmp.replace(d / "row.json")


def all_rows(out: Path) -> list:
    """Every scene's latest row (this job's and the other parallel jobs'), FIRST scenes first."""
    rows = [json.loads(f.read_text()) for f in (out / "estimated_assets").glob("*/row.json")]
    return sorted(rows, key=lambda r: (r["scene"] not in FIRST, FIRST.index(r["scene"]) if r["scene"] in FIRST else 0,
                                       r["scene"]))


def summary(out: Path, rows=None) -> None:
    rows = all_rows(out) if rows is None else rows
    L = ["# Real episode_000000: estimated joint dynamics -> URDF / USD", "",
         f"{time.strftime('%F %T')}. M = inertia about the joint axis [kg m^2] (revolute) or moving mass [kg] "
         "(prismatic); c = viscous joint damping. FIT_TO_DEMONSTRATION, not a validated physical twin.", "",
         "| scene | selection | joint | M | c | best method | PhysX rmse/sigma | G10 fits | asset | URDF |",
         "|" + "---|" * 10]
    for r in rows:
        f = lambda k, fmt="{:.4g}": fmt.format(r[k]) if isinstance(r.get(k), (int, float)) else (r.get(k) or "-")
        L.append(f"| {r['scene']} | {r.get('selection', '-')} | {r.get('joint_type', '-')} | {f('M')} | {f('c')} | "
                 f"{r.get('method', '-')} | {f('rmse_over_sigma', '{:.3g}')} | {r.get('g10_fits', '-')} | "
                 f"{(r.get('asset_status') or r.get('r2r_status') or r.get('step') or '-')[:90]} | {r.get('urdf') or '-'} |")
    (out / "estimated_assets").mkdir(parents=True, exist_ok=True)
    for name, text in (("SUMMARY.md", "\n".join(L) + "\n"), ("summary.json", json.dumps(rows, indent=1, default=str))):
        tmp = out / "estimated_assets" / f".{name}.{os.getpid()}"      # parallel jobs: whole files only
        tmp.write_text(text)
        tmp.replace(out / "estimated_assets" / name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=f"{OUTPUT_ROOT.rstrip('/')}/Samuele/results_{time.strftime('%d_%m_%y')}")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--episodes", choices=("first", "all"), default="first",
                    help="without --scenes: first = every object's episode 000000 (default), all = every episode")
    ap.add_argument("--list-scenes", action="store_true", help="print the scenes that would run (one per line), exit")
    ap.add_argument("--summarize-only", action="store_true", help="rebuild SUMMARY.md from every scene's row.json")
    ap.add_argument("--budget", choices=("smoke", "coarse", "full"), default="full")
    ap.add_argument("--pinn-device", choices=("cpu", "cuda"), default="cpu",
                    help="device of the step-3 PINNs (default cpu: faster for these small nets, measured)")
    a = ap.parse_args()
    pat = re.compile(r"_episode_\d{6}$" if a.episodes == "all" else r"_episode_000000$")
    allsc = sorted(p.name for p in REAL.iterdir() if p.is_dir() and pat.search(p.name))
    scenes = a.scenes or ([s for s in FIRST if s in allsc] + [s for s in allsc if s not in FIRST])
    if a.list_scenes:
        print("\n".join(scenes))
        return
    sys.path.insert(0, str(REPO))
    from utils.forceartgs_env import clean_env
    env = clean_env()                         # every child: forceartgs env + this repo, nothing from the login shell
    os.environ.clear()
    os.environ.update(env)
    out = Path(a.out)
    if not str(out).startswith(OUTPUT_ROOT.rstrip("/") + "/"):
        sys.exit(f"refusing: output must be under {OUTPUT_ROOT} (FORCEARTGS_OUTPUT_ROOT): {out}")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    if a.summarize_only:
        summary(out)
        print(out / "estimated_assets" / "SUMMARY.md")
        return
    log(out, f"start: {len(scenes)} scenes {scenes}")
    rows = []
    for sc in scenes:
        try:
            rows.append(scene(out, sc, a))
        except CodeError as e:
            rows.append(dict(scene=sc, step=f"CODE ERROR, batch stopped: {e}"))
            save_row(out, rows[-1])
            summary(out)
            log(out, f"STOPPED: programming error in the pipeline, not a data problem -- {e}")
            sys.exit(4)
        except Exception as e:                                    # one scene never stops the others
            rows.append(dict(scene=sc, step=f"driver error {type(e).__name__}: {e}"))
            log(out, f"{sc} driver error {type(e).__name__}: {e}")
        save_row(out, rows[-1])
        summary(out)
    log(out, f"all done: {out / 'estimated_assets' / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
