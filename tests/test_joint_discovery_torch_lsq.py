"""GPU geometry solver (joint_discovery/torch_lsq.py): same objective as scipy least_squares, exact Jacobian, the
Schur-reduced LM reaches scipy's optimum (or better), ground truth recovered; anchors refined on the GPU identical to
the NumPy reference."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimotion_2part.joint_discovery import geometry_fit as GF  # noqa: E402
from estimotion_2part.joint_discovery import torch_lsq  # noqa: E402
from estimotion_2part.joint_discovery.config import JointDiscoveryConfig  # noqa: E402
from estimotion_2part.joint_discovery.rigid_kinematics import axis_angle_deg, line_to_line_distance  # noqa: E402
from estimotion_2part.joint_discovery.synthetic import make_scene  # noqa: E402


def _problem(kind, seed=0, **kw):
    sc = make_scene(kind, seed=seed, T=10, **kw)
    cfg = JointDiscoveryConfig(mode="calibrated_joint_aware").validate()
    P = GF.Problem(sc.obs, sc.calibs, cfg, ref=0, use_depth=True)
    return sc, cfg, P


def _perturbed_start(sc, rng):
    a = sc.gt_axis + 0.05 * rng.normal(size=3)
    o = None if sc.gt_origin is None else sc.gt_origin + 0.01 * rng.normal(size=3)
    q = sc.gt_q * (1 + 0.05 * rng.normal(size=len(sc.gt_q)))
    q[0] = 0.0
    return a / np.linalg.norm(a), o, q


def _capture(kind, sc, P, start, nfev=60):
    """Run fit_geometry and capture the solver inputs it built (to replay them through scipy)."""
    a, o, q = start
    X = GF.initial_moving_anchors(P, kind, a, o, q)
    fit_tracks = np.flatnonzero(sc.gt_moving & P.eligible)
    w = sc.gt_moving.astype(float)
    seen = {}
    real = torch_lsq.solve

    def spy(rows, X0, loss, f_scale, max_nfev, extra=None):
        seen.update(rows=rows, X0=X0, loss=loss, f_scale=f_scale)
        return real(rows, X0, loss, f_scale, max_nfev, extra=extra)
    torch_lsq.solve = spy
    try:
        out = GF.fit_geometry(kind, a, o, q, X, fit_tracks, w, P, max_nfev=nfev)
    finally:
        torch_lsq.solve = real
    return out, seen


def _numpy_fun(rows):
    p = torch_lsq._Problem(rows, 0, torch.device("cpu"))
    nj, nq = p.nj, p.nq

    def f(x):
        x = torch.as_tensor(x, dtype=torch.float64)
        X = x[nj + nq:].reshape(-1, 3)
        return torch.cat([p.obs_rows(x[:nj], x[nj:nj + nq], X).reshape(-1), p.prior_rows(x[nj:nj + nq])]).numpy()
    return f, p


def _refine_anchors_numpy(kind, axis, origin, q, X0, obs, K, T_CW, noise, track_mask, iters=6, use_depth=True):
    """The original NumPy implementation (reference for the GPU port)."""
    from estimotion_2part.joint_discovery.observations import depth_sigma
    from estimotion_2part.joint_discovery.rigid_kinematics import apply_joint, exp_so3, normalize
    from estimotion_2part.joint_discovery.visual_losses import project_stacked
    X = np.array(X0, float)
    sel = track_mask[obs.track] & obs.uv_valid & np.isfinite(X[obs.track]).all(1)
    idx = np.flatnonzero(sel)
    tr, t, cam = obs.track[idx], obs.time[idx], obs.cam[idx]
    RA = (np.broadcast_to(np.eye(3), (len(idx), 3, 3)) if kind in ("static", "prismatic")
          else exp_so3(normalize(axis)[None] * q[t][:, None]))
    M = T_CW[cam, :3, :3] @ RA
    dz_ok = (obs.depth_valid[idx] & obs.mask_valid[idx] & np.isfinite(obs.z[idx])) if use_depth else np.zeros(len(idx), bool)
    sz = depth_sigma(np.nan_to_num(obs.z[idx], nan=1.0), noise, obs.depth_edge[idx], obs.depth_cos[idx])
    dof = noise.student_t_dof
    for _ in range(iters):
        xw = X[tr] if kind == "static" else apply_joint(kind, axis, origin, q[t], X[tr])
        uv, z, xc = project_stacked(xw, cam, K, T_CW)
        zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
        fx, fy = K[cam, 0, 0], K[cam, 1, 1]
        D = np.zeros((len(idx), 2, 3))
        D[:, 0, 0], D[:, 0, 2] = fx / zz, -fx * xc[:, 0] / zz ** 2
        D[:, 1, 1], D[:, 1, 2] = fy / zz, -fy * xc[:, 1] / zz ** 2
        Juv = D @ M / noise.sigma_px
        ruv = (uv - obs.uv[idx]) / noise.sigma_px
        wuv = (dof + 1) / (dof + np.sum(ruv ** 2, 1) / 2)
        H = np.einsum("mki,mkj->mij", Juv, Juv) * wuv[:, None, None]
        g = np.einsum("mki,mk->mi", Juv, ruv) * wuv[:, None]
        Jz = M[:, 2, :] / sz[:, None]
        rz = np.where(dz_ok, (z - np.nan_to_num(obs.z[idx])) / sz, 0.0)
        wz = np.where(dz_ok, (dof + 1) / (dof + rz ** 2), 0.0)
        H += np.einsum("mi,mj->mij", Jz, Jz) * wz[:, None, None]
        g += Jz * (rz * wz)[:, None]
        Ht, gt = np.zeros((len(X), 3, 3)), np.zeros((len(X), 3))
        np.add.at(Ht, tr, H)
        np.add.at(gt, tr, g)
        active = np.unique(tr)
        Ha = Ht[active] + np.eye(3)[None] * (1e-6 + 1e-9 * np.trace(Ht[active], axis1=1, axis2=2))[:, None, None]
        step = np.clip(np.linalg.solve(Ha, gt[active][..., None])[..., 0], -0.05, 0.05)
        X[active] -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return X


class TestTorchLSQ(unittest.TestCase):
    def test_robust_losses_match_scipy(self):
        z = np.linspace(0, 30, 301)
        for name in ("soft_l1", "huber", "cauchy", "arctan"):
            ref = np.empty((3, len(z)))
            IMPLEMENTED_LOSSES[name](z, ref, cost_only=False)
            got = [t.numpy() for t in torch_lsq.robust_rho(name, torch.as_tensor(z))]
            for k in range(3):
                np.testing.assert_allclose(got[k], ref[k], rtol=1e-12, atol=1e-14, err_msg=f"{name} rho[{k}]")

    def test_residual_rows_equal_numpy_whitened_residuals(self):
        for kind in ("revolute", "prismatic"):
            sc, cfg, P = _problem(kind, seed=3)
            _, seen = _capture(kind, sc, P, (sc.gt_axis, sc.gt_origin, sc.gt_q.copy()), nfev=2)
            f, p = _numpy_fun(seen["rows"])
            x0 = np.concatenate([np.zeros(p.nj), seen["rows"].q0[seen["rows"].free_q], seen["X0"].ravel()])
            r = f(x0)[: 3 * len(seen["rows"].k)].reshape(-1, 3)
            sel = np.flatnonzero(np.isin(sc.obs.track, np.flatnonzero(sc.gt_moving & P.eligible)) & sc.obs.uv_valid)
            anchors = np.full((sc.obs.n_tracks, 3), np.nan)
            anchors[np.flatnonzero(sc.gt_moving & P.eligible)] = seen["X0"]
            ref = GF.whitened_residuals(kind, seen["rows"].axis0, seen["rows"].origin0, seen["rows"].q0,
                                        np.nan_to_num(anchors), sc.obs, P.K, P.T_CW, P.noise, use_depth=True)
            np.testing.assert_allclose(r[:, :2], ref["r_uv"][sel], rtol=1e-9, atol=1e-9)
            dz = seen["rows"].dz
            np.testing.assert_allclose(r[dz, 2], ref["r_z"][sel][dz], rtol=1e-9, atol=1e-9)
            self.assertTrue(np.all(r[~dz, 2] == 0))

    def test_forward_mode_jacobian_matches_finite_differences(self):
        sc, cfg, P = _problem("revolute", seed=4)
        _, seen = _capture("revolute", sc, P, _perturbed_start(sc, np.random.default_rng(0)), nfev=2)
        p = torch_lsq._Problem(seen["rows"], len(seen["X0"]), torch.device("cpu"))
        pj = torch.tensor([0.01, -0.02, 0.003, 0.001], dtype=torch.float64)
        qf = p.q0[p.free].clone() + 0.01
        X = torch.as_tensor(seen["X0"], dtype=torch.float64)
        f = p.obs_rows
        for j in range(4):
            t = torch.zeros(4, dtype=torch.float64)
            t[j] = 1
            d = torch.func.jvp(f, (pj, qf, X), (t, torch.zeros_like(qf), torch.zeros_like(X)))[1]
            h = 1e-6
            fd = (f(pj + h * t, qf, X) - f(pj - h * t, qf, X)) / (2 * h)
            np.testing.assert_allclose(d.numpy(), fd.numpy(), rtol=1e-5, atol=1e-6)

    def test_reaches_scipy_optimum_and_recovers_ground_truth(self):
        for kind, seed in (("revolute", 5), ("prismatic", 6)):
            sc, cfg, P = _problem(kind, seed=seed, outlier_fraction=0.05)
            (a, o, q, X, q_unc, obs_t, info), seen = _capture(kind, sc, P, _perturbed_start(sc, np.random.default_rng(seed)))
            f, p = _numpy_fun(seen["rows"])
            x0 = np.concatenate([np.zeros(p.nj), seen["rows"].q0[seen["rows"].free_q], seen["X0"].ravel()])
            ref = least_squares(f, x0, method="trf", loss=seen["loss"], f_scale=seen["f_scale"], max_nfev=200,
                                x_scale="jac")
            self.assertLessEqual(info["cost"], ref.cost * (1 + 1e-6) + 1e-9, f"{kind}: torch {info['cost']} scipy {ref.cost}")
            self.assertLess(axis_angle_deg(a, sc.gt_axis), 1.0)
            if kind == "revolute":
                self.assertLess(line_to_line_distance(a, o, sc.gt_axis, sc.gt_origin), 0.01)
            self.assertLess(np.max(np.abs(np.abs(q) - np.abs(sc.gt_q))), 0.02)
            self.assertTrue(np.all(np.isfinite(q_unc[obs_t])))
            self.assertIn("Schur", info["solver"])

    def test_gpu_anchor_refinement_equals_numpy_reference(self):
        for kind in ("revolute", "prismatic", "static"):
            sc, cfg, P = _problem("prismatic" if kind == "prismatic" else "revolute", seed=7, outlier_fraction=0.05)
            a, o, q = _perturbed_start(sc, np.random.default_rng(1))
            if kind == "static":
                a, o, q = None, None, np.zeros_like(q)
            X0 = P.first_3d_points() + 0.01
            ref = _refine_anchors_numpy(kind, a, o, q, X0, sc.obs, P.K, P.T_CW, P.noise, P.eligible)
            got = GF.refine_anchors(kind, a, o, q, X0, sc.obs, P.K, P.T_CW, P.noise, P.eligible)
            self.assertTrue(np.array_equal(np.isnan(ref), np.isnan(got)))
            m = np.isfinite(ref)
            np.testing.assert_allclose(got[m], ref[m], rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
