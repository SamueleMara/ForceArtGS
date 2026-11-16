"""Motion-quality evaluation for multiview reference-camera selection (``reference_selection: quality``).

``prepare_multiview`` used to pick the reference camera whose tracked moving part moves the
most.  Spurious motion (a wrong part assignment, or rotation replaced by translation) also
"moves a lot", so this module scores every camera's moving-part motion against RGB-D
observations instead, in the shared calibrated world frame and at the same physical timestamps:

* coverage      - fraction of timestamps with a valid ICP pose, and of the frame-0 moving
                  surface that still projects onto observed object depth at each timestamp;
* in-sample     - own-camera depth residual of the predicted moving surface, compared with the
                  static (no-motion) hypothesis on the same points;
* held-out      - the same comparison in every *other* calibrated camera's depth, which was not
                  used to estimate this camera's motion;
* agreement     - rotation angle and axis difference between cameras' world-frame motions;
* stability     - largest single-step share of the accumulated rotation (tracking jumps);
* head-to-head  - each pair of cameras' motions applied to the same world points and scored in
                  every camera's depth; views of third cameras decide, a camera's own view only
                  counts when it prefers the other camera's motion.

Residuals are |z_observed - z_predicted| at the predicted pixel, restricted to the observing
camera's object mask.  A frame only counts when the two hypotheses' predicted surfaces are
further apart (median 3D separation) than the observing view's resolution floor, so a view that
cannot resolve the difference does not vote:

* points another camera measured: their residual in the observing view at frame 0, before
  anything moved -- the cross-camera calibration offset plus depth bias;
* points the observing camera measured itself (in-sample test; that camera's own view in a
  head-to-head): their residual at frame 0 is zero by construction, so the floor is the view's
  temporal depth repeatability instead -- the smallest mean absolute static-hypothesis residual
  of its own frame-0 points over the clip (an upper bound on it when the surface moves from
  frame 1 on).  The mean, not the median: with ~1 mm depth quantisation the median residual of
  an unchanged surface is exactly 0 in many frames.
  Without it, frames where both hypotheses coincide (before motion onset) would enter the sign
  test as coin flips.

A comparison "supports" a motion when the motion residual is below the
static residual in significantly more frames than chance (one-sided sign test, p < 0.05 -- the
conventional level, not a tuned threshold), "contradicts" it when the reverse is significant,
and is "uninformative" otherwise.  A camera is eligible as reference when its own depth and at
least one other camera support its motion and no other camera contradicts it; among eligible
cameras, a reference must also win or tie its head-to-head contest against every camera whose
motion is not discredited; when the two motions are distinguishable (a camera's own depth
prefers its own motion) and no independent view decides, the reconstruction is unresolved.  The
lowest held-out residual ratio (motion / static) only breaks true ties, where no view
distinguishes the motions.  With no eligible camera the
reconstruction is reported as unresolved instead of silently using the largest motion.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

SIGNIFICANCE = 0.05  # conventional one-sided level for the frame-wise sign test


def sign_test_p(k, n):
    """P(X >= k) for X ~ Binomial(n, 1/2)."""
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def _rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _axis(R):
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(w)
    return w / n if n > 1e-12 else None


def _rigid(T):
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = T[:3, 3]
    return out


class CameraSequence:
    def __init__(self, data, stems, erode_kernel=3, erode_iterations=1):
        self.data = Path(data)
        self.stems = stems
        self.K = np.loadtxt(self.data / "cam_K.txt")
        self.kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
        self.iterations = erode_iterations
        self._depth = {}
        self._mask = {}

    def depth(self, i):
        if i not in self._depth:
            with np.load(self.data / "depth" / f"{self.stems[i]}.npz") as d:
                self._depth[i] = d["depth"].astype(np.float64)
        return self._depth[i]

    def object_mask(self, i):
        if i not in self._mask:
            self._mask[i] = cv2.imread(str(self.data / "masks" / f"{self.stems[i]}.png"), cv2.IMREAD_GRAYSCALE) > 127
        return self._mask[i]

    def part_points(self, part, i, max_points, seed):
        m = cv2.imread(str(self.data / f"mask_{part}" / f"{self.stems[i]}.png"), cv2.IMREAD_GRAYSCALE) > 127
        m = cv2.erode(m.astype(np.uint8), self.kernel, iterations=self.iterations) > 0
        D = self.depth(i)
        ys, xs = np.nonzero(m & np.isfinite(D) & (D > 0))
        if len(ys) > max_points:
            sel = np.random.default_rng(seed).choice(len(ys), max_points, replace=False)
            ys, xs = ys[sel], xs[sel]
        z = D[ys, xs]
        return np.stack([(xs - self.K[0, 2]) / self.K[0, 0] * z, (ys - self.K[1, 2]) / self.K[1, 1] * z, z], 1)

    def residuals(self, points_cam, i):
        """|z_obs - z_pred| for points projecting into this camera's object mask with valid depth."""
        D, M = self.depth(i), self.object_mask(i)
        z = points_cam[:, 2]
        front = z > 1e-6
        u = np.full(len(z), -1)
        v = np.full(len(z), -1)
        u[front] = np.round(self.K[0, 0] * points_cam[front, 0] / z[front] + self.K[0, 2]).astype(int)
        v[front] = np.round(self.K[1, 1] * points_cam[front, 1] / z[front] + self.K[1, 2]).astype(int)
        h, w = D.shape
        inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ok = np.zeros(len(z), bool)
        ok[inside] = M[v[inside], u[inside]] & np.isfinite(D[v[inside], u[inside]]) & (D[v[inside], u[inside]] > 0)
        r = np.full(len(z), np.nan)
        r[ok] = np.abs(D[v[ok], u[ok]] - z[ok])
        return r


def _median(r):
    return float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan


def _compare(r_motion, r_static, informative=None, floor=None):
    """Frame-wise sign test of motion vs static residuals (medians over points, per frame).

    ``informative[t]`` is False when the two hypotheses predict surfaces closer together than the
    observing view's own frame-0 residual (``floor``: calibration offset + depth bias + noise,
    measured before anything moved); such frames cannot discriminate and are not counted.
    """
    if informative is None:
        informative = [True] * len(r_motion)
    pairs = [(m, s) for m, s, ok in zip(r_motion, r_static, informative)
             if ok and np.isfinite(m) and np.isfinite(s) and m != s]
    better = sum(m < s for m, s in pairs)
    worse = len(pairs) - better
    p_support, p_contradict = sign_test_p(better, len(pairs)), sign_test_p(worse, len(pairs))
    verdict = "support" if p_support < SIGNIFICANCE else "contradict" if p_contradict < SIGNIFICANCE else "uninformative"
    finite_m = [m for m in r_motion if np.isfinite(m)]
    finite_s = [s for s in r_static if np.isfinite(s)]
    ratio = (float(np.median(finite_m)) / max(float(np.median(finite_s)), 1e-12)) if finite_m and finite_s else None
    return dict(verdict=verdict, frames=len(pairs), motion_better=int(better), p_support=p_support,
                frame0_floor_m=None if floor is None or not np.isfinite(floor) else float(floor),
                informative_frames=int(sum(bool(x) for x in informative)),
                p_contradict=p_contradict, median_motion_residual_m=float(np.median(finite_m)) if finite_m else None,
                median_static_residual_m=float(np.median(finite_s)) if finite_s else None, residual_ratio=ratio)


def camera_quality(camera_datasets, stems, calibrations, moving_parts, pose_validity=None, max_points=2000, seed=0):
    """Quality components of every camera's moving-part motion (see module docstring)."""
    cameras = sorted(camera_datasets)
    seqs = {c: CameraSequence(camera_datasets[c], stems) for c in cameras}
    n = len(stems)
    motions, world = {}, {}
    for c in cameras:
        G = np.load(Path(camera_datasets[c]) / f"global_poses_{moving_parts[c]}.npy")
        motions[c] = [np.linalg.inv(G[t]) @ G[0] for t in range(n)]          # frame-0 camera coords -> t
        C = calibrations[c]
        world[c] = [C @ _rigid(M) @ np.linalg.inv(C) for M in motions[c]]
    report, repeatability = {}, {}
    for c in cameras:
        valid = np.ones(n, bool) if pose_validity is None else np.asarray(pose_validity[c], bool)
        S0 = seqs[c].part_points(moving_parts[c], 0, max_points, seed)
        own_m, own_s, own_s_mean, coverage, separation = [], [], [], [], []
        held = {d: ([], []) for d in cameras if d != c}
        floors = {d: _median(seqs[d].residuals(S0 @ (np.linalg.inv(calibrations[d]) @ calibrations[c])[:3, :3].T
                                                + (np.linalg.inv(calibrations[d]) @ calibrations[c])[:3, 3], 0))
                  for d in cameras}
        for t in range(1, n):
            if not valid[t] or len(S0) == 0:
                for lst in (own_m, own_s, own_s_mean, coverage, separation):
                    lst.append(np.nan)
                for d in held:
                    held[d][0].append(np.nan)
                    held[d][1].append(np.nan)
                continue
            M = motions[c][t]
            Xt = S0 @ M[:3, :3].T + M[:3, 3]
            separation.append(float(np.median(np.linalg.norm(Xt - S0, axis=1))))
            rm, rs = seqs[c].residuals(Xt, t), seqs[c].residuals(S0, t)
            coverage.append(float(np.isfinite(rm).mean()))
            own_m.append(float(np.nanmedian(rm)) if np.isfinite(rm).any() else np.nan)
            own_s.append(float(np.nanmedian(rs)) if np.isfinite(rs).any() else np.nan)
            own_s_mean.append(float(np.nanmean(rs)) if np.isfinite(rs).any() else np.nan)
            Cc = calibrations[c]
            for d in held:
                to_d = np.linalg.inv(calibrations[d]) @ Cc
                Xd_m = Xt @ to_d[:3, :3].T + to_d[:3, 3]
                Xd_s = S0 @ to_d[:3, :3].T + to_d[:3, 3]
                hm, hs = seqs[d].residuals(Xd_m, t), seqs[d].residuals(Xd_s, t)
                held[d][0].append(float(np.nanmedian(hm)) if np.isfinite(hm).any() else np.nan)
                held[d][1].append(float(np.nanmedian(hs)) if np.isfinite(hs).any() else np.nan)
        repeatability[c] = float(np.nanmin(own_s_mean)) if np.isfinite(own_s_mean).any() else np.nan
        R_final = world[c][-1][:3, :3]
        steps =[_rot_deg(world[c][t][:3, :3] @ world[c][t - 1][:3, :3].T) for t in range(1, n)]
        path = float(np.sum(steps))
        report[c] = dict(
            moving_part=int(moving_parts[c]), frame0_points=int(len(S0)),
            valid_pose_fraction=float(valid.mean()),
            surface_coverage_median=float(np.nanmedian(coverage)) if np.isfinite(coverage).any() else None,
            surface_coverage_min=float(np.nanmin(coverage)) if np.isfinite(coverage).any() else None,
            depth_repeatability_m=None if not np.isfinite(repeatability[c]) else repeatability[c],
            in_sample=_compare(own_m, own_s, [x > repeatability[c] for x in separation], repeatability[c]),
            held_out={d: _compare(*held[d], [x > floors[d] for x in separation], floors[d]) for d in held},
            final_rotation_deg=_rot_deg(R_final),
            final_axis_world=None if _axis(R_final) is None else _axis(R_final).tolist(),
            rotation_path_deg=path,
            largest_step_deg=float(max(steps)) if steps else 0.0,
            largest_step_share=float(max(steps) / path) if steps and path > 0 else None,
            largest_step_frame=int(np.argmax(steps) + 1) if steps else None,
            per_frame=dict(own_motion=own_m, own_static=own_s, coverage=coverage, separation_m=separation,
                           held_out={d: dict(motion=held[d][0], static=held[d][1]) for d in held}))
    for c in cameras:
        agreement = {}
        for d in cameras:
            if d == c:
                continue
            Ra, Rb = world[c][-1][:3, :3], world[d][-1][:3, :3]
            a, b = _axis(Ra), _axis(Rb)
            agreement[d] = dict(relative_rotation_deg=_rot_deg(Ra.T @ Rb),
                                axis_angle_deg=None if a is None or b is None else
                                float(np.degrees(np.arccos(min(1.0, abs(float(a @ b)))))))
        report[c]["agreement"] = agreement
    # Head-to-head: every pair of motions applied to the same world points (union of both
    # cameras' frame-0 moving surfaces), scored in every camera's depth.
    S0w = {}
    for c in cameras:
        S0 = seqs[c].part_points(moving_parts[c], 0, max_points, seed)
        C = calibrations[c]
        S0w[c] = S0 @ C[:3, :3].T + C[:3, 3]
    for c in cameras:
        report[c]["contests"] = {}
    for i, c in enumerate(cameras):
        for d in cameras[i + 1:]:
            pts = np.vstack([S0w[c], S0w[d]])
            views = {}
            sep = []
            for t in range(1, n):
                A, B = world[c][t], world[d][t]
                sep.append(float(np.median(np.linalg.norm(pts @ A[:3, :3].T + A[:3, 3] - (pts @ B[:3, :3].T + B[:3, 3]), axis=1))))
            for v in cameras:
                to_v = np.linalg.inv(calibrations[v])
                # Resolution floor (module docstring): a third view's frame-0 residual of the points;
                # a contestant's own view uses its depth repeatability, because its own frame-0
                # points have zero residual there by construction.
                if v in (c, d):
                    floor = repeatability[v]
                else:
                    floor = _median(seqs[v].residuals(pts @ to_v[:3, :3].T + to_v[:3, 3], 0))
                rc, rd = [], []
                for t in range(1, n):
                    out = []
                    for cam in (c, d):
                        Mw = world[cam][t]
                        X = pts @ Mw[:3, :3].T + Mw[:3, 3]
                        r = seqs[v].residuals(X @ to_v[:3, :3].T + to_v[:3, 3], t)
                        out.append(float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan)
                    rc.append(out[0])
                    rd.append(out[1])
                cmp = _compare(rc, rd, [x > floor for x in sep], floor)   # "support" = c better than d in v
                views[v] = dict(verdict={"support": c, "contradict": d}.get(cmp["verdict"], "tie"),
                                frames=cmp["frames"], c_better=cmp["motion_better"],
                                frame0_floor_m=cmp["frame0_floor_m"], median_separation_m=float(np.median(sep)),
                                median_residual_m={c: cmp["median_motion_residual_m"], d: cmp["median_static_residual_m"]})
            report[c]["contests"][d] = views
            report[d]["contests"][c] = views
    return report


def contest_winner(report, c, d):
    """Outcome of the c-vs-d contest: a camera name, ``"tie"``, ``"disagreement"`` or ``"conflict"``.

    Independent views (neither c nor d) decide; a camera's own view counts only when it prefers
    the *other* camera.  ``"disagreement"``: a camera's own depth prefers its own motion (one or
    both cameras) and no independent view resolves it -- the motions are distinguishable but
    nothing independent says which is right.  ``"tie"``: no view distinguishes the two motions.
    """
    views = report[c]["contests"][d]
    votes = set()
    self_preference = 0
    for v, res in views.items():
        w = res["verdict"]
        if w == "tie":
            continue
        if v in (c, d) and w == v:
            self_preference += 1
            continue
        votes.add(w)
    if votes:
        return votes.pop() if len(votes) == 1 else "conflict"
    return "disagreement" if self_preference else "tie"


def select_reference(report):
    """Return ``(reference or None, status, reasons)`` from :func:`camera_quality` output.

    Eligible: own depth supports the motion, some other camera supports it, none contradicts it.
    Discredited: own depth does not support it, or another camera contradicts it.
    A reference must be eligible and, against every camera that is not discredited, must win or
    tie its head-to-head contest; a ``disagreement`` (both self-preferring, nothing independent)
    or ``conflict`` leaves the reconstruction unresolved.
    """
    reasons, eligible, discredited = {}, [], set()
    for c, q in report.items():
        why = []
        if q["in_sample"]["verdict"] != "support":
            why.append(f"own depth does not support the motion ({q['in_sample']['verdict']})")
            discredited.add(c)
        verdicts = {d: h["verdict"] for d, h in q["held_out"].items()}
        if not any(v == "support" for v in verdicts.values()):
            why.append("no other camera supports the motion")
        contra = [d for d, v in verdicts.items() if v == "contradict"]
        if contra:
            why.append(f"contradicted by {', '.join(contra)}")
            discredited.add(c)
        reasons[c] = why or ["eligible"]
        if not why:
            eligible.append(c)
    if not eligible:
        return None, "unresolved", reasons
    verified = []
    for c in eligible:
        ok = True
        for d in report:
            if d == c or d in discredited or "contests" not in report[c]:
                continue
            w = contest_winner(report, c, d)
            if w in ("tie", c):
                continue
            ok = False
            reasons[c] = [r for r in reasons[c] if r != "eligible"] + [
                {"disagreement": f"disagrees with {d}; no independent view decides",
                 "conflict": f"conflicting independent evidence against {d}"}.get(w, f"motion worse than {d}'s in independent views")]
        if ok:
            verified.append(c)
    if not verified:
        return None, "unresolved", reasons

    def held_ratio(c):
        ratios = [h["residual_ratio"] for h in report[c]["held_out"].values()
                  if h["verdict"] == "support" and h["residual_ratio"] is not None]
        return float(np.exp(np.mean(np.log(ratios)))) if ratios else float("inf")
    best = min(verified, key=held_ratio)
    return best, "resolved", reasons


def write_report(path, report, reference, status, reasons, legacy_reference=None):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    Path(path).write_text(json.dumps(clean(dict(selected_reference=reference, status=status, reasons=reasons,
                                                legacy_motion_reference=legacy_reference, cameras=report)), indent=1) + "\n")
