"""Rigid SE(3) transform parameterization and part-aware track filtering (Stages 1-2)."""
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from estimotion_2part.models.trans_models import (  # noqa: E402
    Optimizer, RigidTrans, Trans, TransformLossComputer, make_trans, project_to_se3, se3_exp)
from estimotion_2part.solve_transformation import joint_reinit_part_trans, reinit_part_trans  # noqa: E402
from tracking.part_aware_track_filter import part_aware_filter  # noqa: E402

DEV = "cpu"
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    W = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * W + (1 - math.cos(t)) * W @ W


def rigid(axis, deg, pivot, shift=(0, 0, 0)):
    T = np.eye(4)
    R = rot(axis, deg)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pivot) - R @ np.asarray(pivot) + np.asarray(shift)
    return T


def two_part_scene(n=1200, moving_fraction=0.3, noise=0.002, outliers=0.1, seed=0, deg=12.0):
    """Static base + lid rotating about a hinge, camera frame (z ~ 1 m)."""
    rng = np.random.default_rng(seed)
    nm = int(n * moving_fraction)
    base = rng.uniform([-0.15, 0.0, 0.95], [0.15, 0.1, 1.05], (n - nm, 3))
    lid = rng.uniform([-0.15, -0.15, 0.92], [0.15, 0.0, 0.95], (nm, 3))
    T_lid = rigid([1, 0, 0], deg, [0, 0, 0.95])
    P = np.vstack([base, lid])
    Q = np.vstack([base, lid @ T_lid[:3, :3].T + T_lid[:3, 3]])
    Q = Q + rng.normal(0, noise, Q.shape)
    label = np.r_[np.zeros(n - nm), np.ones(nm)].astype(int)
    out = rng.random(n) < outliers
    Q[out] += rng.normal(0, 0.05, (int(out.sum()), 3))
    return P, Q, label, out, T_lid


def rot_err_deg(A, B):
    R = A[:3, :3].T @ B[:3, :3]
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def args_ns(**kw):
    d = dict(loss_type="normalized_huber", w_p=0.0, w_m=200.0, w_e=-0.01, w_s=10.0, w_init=20.0,
             projection_normalizer=640.0, scale_obj=0.3, use_scheduler=True, scheduler_type="cosine")
    d.update(kw)
    return types.SimpleNamespace(**d)


def optimise(model, P, Q, w_init, epochs=300, lr_T=1e-3, lr_w=1e-2, check=None):
    args = args_ns()
    loss = TransformLossComputer(loss_type="normalized_huber", cam_k_tensor=torch.tensor(K, dtype=torch.float32),
                                 w_p=0.0, w_m=args.w_m / args.scale_obj, w_e=args.w_e, w_s=args.w_s, w_init=args.w_init,
                                 weight_layout="flat", allow_normalized_l1=False)
    opt = Optimizer(model, weight_lr=lr_w, transform_lr=lr_T, args=args)
    src = torch.tensor(P, dtype=torch.float32)
    tgt = torch.tensor(Q, dtype=torch.float32)
    idx = torch.arange(len(P))
    T_id = torch.zeros(len(P), dtype=torch.long)
    init = torch.tensor(w_init, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        p1, p2, w, b = model.forward_components(src, idx, T_id)
        total, _ = loss.compute_total_loss(pred_t1=p1, pred_t2=p2, w_sel=w, blended=b, target_points=tgt,
                                           source_points=src, weight_logits=model.weight, init_weights_tensor=init)
        total.backward()
        opt.step()
        opt.scheduler_step(None)
        if check is not None:
            check(model)
    return model


class SE3ParameterizationTests(unittest.TestCase):
    def test_exp_is_rigid_and_matches_rodrigues(self):
        xi = torch.tensor([[0.1, -0.2, 0.3, 0.2, -0.1, 0.4]], dtype=torch.float64)
        T = se3_exp(xi)[0].numpy()
        R = T[:3, :3]
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)
        phi = xi[0, 3:].numpy()
        np.testing.assert_allclose(R, rot(phi, math.degrees(np.linalg.norm(phi))), atol=1e-12)
        np.testing.assert_allclose(se3_exp(torch.zeros(1, 6))[0].numpy(), np.eye(4), atol=0)

    def test_rotation_stays_orthonormal_with_positive_determinant_during_optimisation(self):
        P, Q, label, _, _ = two_part_scene()
        bad = []

        def check(model):
            for T in (model.T1.detach(), model.T2.detach()):
                R = T[0, :3, :3].double()
                bad.append(float(torch.linalg.norm(R.T @ R - torch.eye(3, dtype=torch.float64))))
                self.assertGreater(float(torch.linalg.det(R)), 0.0)
                np.testing.assert_allclose(T[0, 3].numpy(), [0, 0, 0, 1], atol=0)
        model = RigidTrans(len(P), 1, init_weights=np.clip(label, 0.2, 0.8), cam_K=K, device=DEV,
                           pivot=torch.tensor(P.mean(0), dtype=torch.float32))
        optimise(model, P, Q, np.clip(label, 0.2, 0.8).astype(np.float32), epochs=200, check=check)
        self.assertLess(max(bad), 1e-5)

    def test_legacy_matrix_is_not_rigid_under_the_same_optimisation(self):
        """Control: the free 4x4 parameterization leaves SO(3) (the defect being fixed)."""
        P, Q, label, _, _ = two_part_scene()
        model = Trans(len(P), 1, init_weights=np.clip(label, 0.2, 0.8), cam_K=K, device=DEV)
        optimise(model, P, Q, np.clip(label, 0.2, 0.8).astype(np.float32), epochs=200)
        R = model.T2.detach()[0, :3, :3].double()
        self.assertGreater(float(torch.linalg.norm(R.T @ R - torch.eye(3, dtype=torch.float64))), 1e-4)

    def test_gradients_match_finite_differences_for_translation_and_rotation(self):
        torch.manual_seed(0)
        model = RigidTrans(5, 1, cam_K=K, device=DEV, init_T1=torch.tensor(rigid([0, 1, 0], 10, [0, 0, 1]), dtype=torch.float32),
                           pivot=torch.tensor([0.0, 0.0, 1.0])).double()
        x = torch.randn(5, 3, dtype=torch.float64) * 0.1 + torch.tensor([0, 0, 1.0], dtype=torch.float64)
        y = torch.randn(5, 3, dtype=torch.float64) * 0.1 + torch.tensor([0, 0, 1.0], dtype=torch.float64)

        def f(xi):
            T = model._compose(xi, model.T1_init)[0]
            return ((x @ T[:3, :3].T + T[:3, 3] - y) ** 2).sum()

        xi0 = torch.tensor([[0.01, -0.02, 0.03, 0.05, -0.04, 0.02]], dtype=torch.float64, requires_grad=True)
        g = torch.autograd.grad(f(xi0), xi0)[0][0]
        eps = 1e-6
        for i in range(6):  # 0-2 translation, 3-5 rotation
            d = torch.zeros_like(xi0)
            d[0, i] = eps
            fd = (f(xi0 + d) - f(xi0 - d)) / (2 * eps)
            self.assertAlmostEqual(float(g[i]), float(fd), delta=1e-6 * max(1.0, abs(float(fd))))
        self.assertTrue(torch.autograd.gradcheck(lambda v: model._compose(v, model.T1_init), (xi0,)))

    def test_synthetic_rigid_motion_recovery_under_noise_and_outliers(self):
        """Pipeline learning rate (lr_T = 1e-4) and the unchanged legacy loss terms.

        Confident part weights isolate the parameterization: with soft weights (0.2/0.8) the
        legacy mixture objective itself biases T2 towards the static points by ~1-2 degrees in
        *both* parameterizations (measured; reported, not changed here).
        """
        P, Q, label, _, T_lid = two_part_scene(noise=0.002, outliers=0.1)
        w0 = np.clip(label, 0.01, 0.99).astype(np.float32)
        lid = P[label == 1]
        truth = lid @ T_lid[:3, :3].T + T_lid[:3, 3]
        for name, T_init, max_err in (("from truth", T_lid, 0.5),
                                      ("3 deg / 5 mm off", T_lid @ rigid([0.3, 1, 0.2], 3.0, [0, -0.07, 0.94],
                                                                           shift=(0.005, 0, 0)), 1.5)):
            model = RigidTrans(len(P), 1, init_weights=w0, init_T2=torch.tensor(T_init, dtype=torch.float32),
                               cam_K=K, device=DEV, pivot=torch.tensor(P.mean(0), dtype=torch.float32))
            optimise(model, P, Q, w0, epochs=600, lr_T=1e-4)
            T2 = model.T2.detach()[0].double().numpy()
            self.assertLess(rot_err_deg(T2, T_lid), max_err, name)
            disp = np.linalg.norm(lid @ T2[:3, :3].T + T2[:3, 3] - truth, axis=1)
            self.assertLess(float(np.median(disp)), 0.001, name)       # below the 2 mm/axis noise
            self.assertLess(rot_err_deg(model.T1.detach()[0].double().numpy(), np.eye(4)), 0.5, name)


class LegacyAndReinitTests(unittest.TestCase):
    def test_matrix_mode_is_the_unmodified_legacy_module(self):
        model = make_trans("matrix", num_points=4, num_T=2, cam_K=K, device=DEV)
        self.assertIs(type(model), Trans)
        self.assertEqual([p.shape for p in model.transform_parameters()], [torch.Size([2, 4, 4])] * 2)
        self.assertIs(model.transform_parameters()[0], model.T1)
        with self.assertRaises(ValueError):
            make_trans("quaternion", num_points=4, num_T=1, cam_K=K, device=DEV)

    def test_legacy_reinit_does_not_change_matrix_transforms(self):
        """Documents the legacy defect: ``T1[0].data = ...`` writes into a temporary view."""
        P, Q, label, _, _ = two_part_scene(outliers=0.0)
        trajs = torch.tensor(np.stack([P, Q], 1), dtype=torch.float32)
        model = Trans(len(P), 1, init_weights=np.clip(label, 0.2, 0.8), cam_K=K, device=DEV)
        before = model.T2.detach().clone()
        reinit_part_trans(model, trajs)
        self.assertTrue(torch.equal(model.T2.detach(), before))
        reinit_part_trans(model, trajs, apply=True)          # opt-in fix actually writes
        self.assertFalse(torch.equal(model.T2.detach(), before))

    def test_rigid_reinit_resets_the_tangent_parameters(self):
        P, Q, label, _, T_lid = two_part_scene(outliers=0.0)
        trajs = torch.tensor(np.stack([P, Q], 1), dtype=torch.float32)
        model = RigidTrans(len(P), 1, init_weights=np.clip(label, 0.05, 0.95), cam_K=K, device=DEV)
        model.xi2.data[:] = 0.1
        before = model.T2.detach().clone()
        reinit_part_trans(model, trajs)                       # default: legacy-equivalent no-op
        self.assertTrue(torch.equal(model.T2.detach(), before))
        reinit_part_trans(model, trajs, apply=True)
        self.assertTrue(torch.all(model.xi2.detach() == 0))
        self.assertLess(rot_err_deg(model.T2.detach()[0].double().numpy(), T_lid), 0.5)
        pairs = {0: [dict(point_idx=i, traj=trajs[i]) for i in range(len(P))]}
        joint_reinit_part_trans(model, pairs)                 # legacy: nothing computed
        joint_reinit_part_trans(model, pairs, apply=True)

    def test_projection_of_rigid_initialisation_is_exact(self):
        T = torch.tensor(rigid([1, 2, 3], 25, [0.1, 0.2, 1.0]), dtype=torch.float64)
        self.assertTrue(torch.allclose(project_to_se3(T), T, atol=1e-12))


class PartAwareFilterTests(unittest.TestCase):
    def legacy_keep(self, P, Q):
        d = np.linalg.norm(Q - P, axis=1)
        return np.abs(d - d.mean()) <= 2.0 * d.std()

    def test_moving_minority_is_kept_while_outliers_are_rejected(self):
        P, Q, label, out, _ = two_part_scene(n=3000, moving_fraction=0.2, outliers=0.05, deg=15.0)
        keep, weight, info = part_aware_filter(torch.tensor(np.stack([P, Q], 1), dtype=torch.float32))
        keep = keep.numpy()
        moving_inliers = (label == 1) & ~out
        static_inliers = (label == 0) & ~out
        self.assertEqual(info["hypotheses"], 2)
        self.assertGreater(keep[moving_inliers].mean(), 0.9)
        self.assertGreater(keep[static_inliers].mean(), 0.9)
        self.assertGreater(1 - keep[out].mean(), 0.8)                  # gross outliers rejected

    def test_legacy_global_filter_is_biased_against_the_moving_part(self):
        """Without gross outliers the global 2-sigma rule cuts the far end of the moving part."""
        P, Q, label, _, _ = two_part_scene(n=3000, moving_fraction=0.2, outliers=0.0, deg=15.0)
        legacy = self.legacy_keep(P, Q)
        keep = part_aware_filter(torch.tensor(np.stack([P, Q], 1), dtype=torch.float32))[0].numpy()
        self.assertGreater(1 - legacy[label == 1].mean(), 0.1)
        self.assertLess(1 - legacy[label == 0].mean(), 0.01)
        self.assertGreater(keep[label == 1].mean(), 0.9)

    def test_wrong_initial_partition_cannot_remove_the_correct_part(self):
        """Hypothesis A is fitted on all candidates; B is refit after reassignment, so a moving
        majority (A = moving) and a static majority (A = static) give the same kept sets."""
        for frac in (0.3, 0.7):
            P, Q, label, out, _ = two_part_scene(n=2000, moving_fraction=frac, outliers=0.05)
            keep, _, info = part_aware_filter(torch.tensor(np.stack([P, Q], 1), dtype=torch.float32))
            keep = keep.numpy()
            for part in (0, 1):
                self.assertGreater(keep[(label == part) & ~out].mean(), 0.9, (frac, part, info))

    def test_single_motion_keeps_everything_consistent(self):
        rng = np.random.default_rng(1)
        P = rng.uniform(-0.1, 0.1, (500, 3)) + [0, 0, 1]
        Q = P + rng.normal(0, 0.001, P.shape)
        keep, weight, info = part_aware_filter(torch.tensor(np.stack([P, Q], 1), dtype=torch.float32))
        self.assertGreater(keep.float().mean().item(), 0.9)
        self.assertTrue(set(np.unique(weight.numpy())).issubset({0.0, 0.5, 1.0}))

    def test_global_mode_is_the_default(self):
        from utils.pipeline_config import TRACKING_DEFAULTS
        self.assertEqual(TRACKING_DEFAULTS["track_filter_mode"], "global")
        self.assertEqual(TRACKING_DEFAULTS["transform_parameterization"], "matrix")
        self.assertFalse(TRACKING_DEFAULTS["transform_reinit_apply"])


if __name__ == "__main__":
    unittest.main()
