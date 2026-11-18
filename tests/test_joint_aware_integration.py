"""Joint-aware reconstruction integration tests (end-to-end spec section 11; IDs in test names)."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reconstruction" / "nerfstudio"))

from estimotion_2part.joint_discovery.integration_contract import (  # noqa: E402
    AuthorityError, CameraEntry, JointAuthority, derive_estimated_joint, frame_alignment, regauge)

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    torch, HAS_CUDA = None, False


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = eye
    return T


def rodrigues(u, q):
    u = np.asarray(u, float) / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def make_authority(kind="revolute", T=12, span=0.5, ref=0, cams=None, origin=(0.0, 0.0, 0.1), axis=(1.0, 0.0, 0.0)):
    q = np.linspace(0, span, T) if kind != "static" else np.zeros(T)
    q = q - q[ref]
    cams = cams or {"cam_a": CameraEntry("cam_a", np.array([[120.0, 0, 79.5], [0, 120.0, 59.5], [0, 0, 1]]),
                                         look_at((0.0, -0.6, 0.35), (0, 0, 0.05)), 160, 120)}
    return JointAuthority(kind, None if kind == "static" else np.asarray(axis, float) / np.linalg.norm(axis),
                          np.asarray(origin, float) if kind == "revolute" else None, q, np.ones(T, bool), ref,
                          np.arange(T) * 2 + 10, np.arange(T) * 4 + 20, np.arange(T) * 0.1 + 1.0, cams).validate()


class AuthorityContract(unittest.TestCase):
    def test_I01_write_load_and_tamper(self):
        a = make_authority()
        with tempfile.TemporaryDirectory() as d:
            aid = a.write(d)
            b = JointAuthority.load(d)
            self.assertEqual(b.authority_id(), aid)
            meta = json.loads((Path(d) / "authority.json").read_text())
            meta["joint_type"] = "prismatic"
            meta["origin_world"] = None
            (Path(d) / "authority.json").write_text(json.dumps(meta))
            with self.assertRaises(AuthorityError):
                JointAuthority.load(d)
        with tempfile.TemporaryDirectory() as d:
            a.write(d)
            z = dict(np.load(Path(d) / "joint_states.npz"))
            z["q"] = z["q"] * 1.01
            np.savez(Path(d) / "joint_states.npz", **z)
            with self.assertRaises(AuthorityError):
                JointAuthority.load(d)
        with tempfile.TemporaryDirectory() as d:
            a.write(d)
            b = make_authority(span=0.4)
            with self.assertRaises(AuthorityError):
                b.write(d)          # refuses to overwrite a different authority

    def test_I02_I07_regauge_reference_and_sign(self):
        for kind in ("revolute", "prismatic"):
            a = make_authority(kind, ref=5)
            b = regauge(a, 0)
            self.assertEqual(b.q[0], 0.0)
            for s in range(a.T):
                for t in range(a.T):
                    np.testing.assert_allclose(a.relative(s, t), b.relative(s, t), atol=1e-12)
            flipped = JointAuthority(**{**b.__dict__, "axis_world": -b.axis_world, "q": -b.q}).validate()
            np.testing.assert_allclose(flipped.transforms(), b.transforms(), atol=1e-12)
        with self.assertRaises(AuthorityError):
            JointAuthority(**{**make_authority().__dict__, "q": np.linspace(0.1, 0.5, 12)}).validate()
        with self.assertRaises(AuthorityError):    # prismatic must not carry a pivot
            JointAuthority(**{**make_authority("prismatic").__dict__, "origin_world": np.zeros(3)}).validate()

    def test_I04_virtual_cameras(self):
        a = make_authority()
        for t in range(a.T):
            C_S = a.cameras["cam_a"].T_WC
            C_M = a.virtual_camera("cam_a", t)
            np.testing.assert_allclose(C_S @ np.linalg.inv(C_M), a.transform(t), atol=1e-12)

    def test_I05_I06_frame_firewall_and_adapter(self):
        rng = np.random.default_rng(0)
        N = np.eye(4)
        N[:3, :3] = rodrigues(rng.normal(size=3), 0.7)
        N[:3, 3] = [0.3, -0.2, 0.5]
        C = np.stack([look_at(rng.normal(size=3) + [0, -1, 0.5], [0, 0, 0]) for _ in range(5)])
        np.testing.assert_allclose(frame_alignment(C, N[None] @ C), N, atol=1e-12)
        bad = N[None] @ C
        bad[2, :3, 3] += 0.01
        with self.assertRaises(AuthorityError):
            frame_alignment(C, bad)
        for kind in ("revolute", "prismatic"):
            a = make_authority(kind)
            est = derive_estimated_joint(a, N, "id")
            u, o = np.asarray(est["axis"]), np.asarray(est["origin"])
            X = rng.normal(size=(20, 3))
            for t in (0, 4, 11):
                q = est["joint_params"][t]
                # splatfacto_art formula: p' = R(u q)(p - o) + o (revolute), p + u q (prismatic)
                Y = (X - o) @ rodrigues(u, q).T + o if kind == "revolute" else X + u * q
                A_B = N @ a.transform(t) @ np.linalg.inv(N)
                np.testing.assert_allclose(Y, X @ A_B[:3, :3].T + A_B[:3, 3], atol=1e-9)
        with self.assertRaises(AuthorityError):
            derive_estimated_joint(make_authority(), np.diag([2.0, 2.0, 2.0, 1.0]), "id")


# ----------------------------------------------------------------------------- synthetic RGB-D hinge scene
def _texture(P):
    """Deterministic high-contrast colour from canonical coordinates (2 cm checker x hash)."""
    k = np.floor(P / 0.02).astype(int)
    h = (k[:, 0] * 73856093 ^ k[:, 1] * 19349663 ^ k[:, 2] * 83492791) & 0xFFFFFF
    return np.stack([(h & 255), (h >> 8) & 255, (h >> 16) & 255], 1).astype(np.uint8)


def _surface_points(step=0.0025):
    g = np.arange(-0.1, 0.1 + 1e-9, step)
    gz = np.arange(0.0, 0.1 + 1e-9, step)
    # static base: front face (y=-0.1), top-less box sides; moving lid: plate on top hinged at y=+0.1, z=0.1
    base = [np.stack(np.meshgrid(g, [-0.1], gz, indexing="ij"), -1).reshape(-1, 3),
            np.stack(np.meshgrid([-0.1], g, gz, indexing="ij"), -1).reshape(-1, 3),
            np.stack(np.meshgrid([0.1], g, gz, indexing="ij"), -1).reshape(-1, 3)]
    lid = np.stack(np.meshgrid(g, g, [0.1], indexing="ij"), -1).reshape(-1, 3)
    return np.concatenate(base), lid


def write_synthetic_scene(root: Path, a: JointAuthority, moving=True):
    import cv2
    base, lid = _surface_points()
    cb, cl = _texture(base), _texture(lid + 0.5)
    (root / "metadata").mkdir(parents=True)
    cams = []
    for cid, cam in a.cameras.items():
        cams.append(dict(name=cid, fx=cam.K[0, 0], fy=cam.K[1, 1], cx=cam.K[0, 2], cy=cam.K[1, 2], width=cam.width,
                         height=cam.height, pose_matrix_4x4=cam.T_WC.tolist(),
                         pose_convention="opencv_optical_camera_to_world"))
    (root / "metadata" / "cameras.json").write_text(json.dumps(dict(cameras=cams)))
    with open(root / "metadata" / "frame_timestamps.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_index", "source_frame_index", "timestamp_s"])
        for t in range(a.T):
            w.writerow([int(a.rgbd_frame_index[t]), int(a.source_frame_index[t]), float(a.timestamp_s[t])])
    gt = {}
    for t in range(a.T):
        A = a.transform(t) if moving else np.eye(4)
        lid_t = lid @ A[:3, :3].T + A[:3, 3]
        P = np.concatenate([base, lid_t])
        C = np.concatenate([cb, cl])
        lab = np.r_[np.zeros(len(base), int), np.ones(len(lid), int)]
        for cid, cam in a.cameras.items():
            T_CW = np.linalg.inv(cam.T_WC)
            y = P @ T_CW[:3, :3].T + T_CW[:3, 3]
            u = np.round(cam.K[0, 0] * y[:, 0] / y[:, 2] + cam.K[0, 2]).astype(int)
            v = np.round(cam.K[1, 1] * y[:, 1] / y[:, 2] + cam.K[1, 2]).astype(int)
            ok = (y[:, 2] > 0.05) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
            depth = np.full((cam.height, cam.width), np.inf)
            rgb = np.full((cam.height, cam.width, 3), 128, np.uint8)
            glab = np.full((cam.height, cam.width), -1)
            order = np.argsort(-y[ok, 2])                       # far to near: nearer overwrites
            uu, vv, zz = u[ok][order], v[ok][order], y[ok, 2][order]
            depth[vv, uu] = zz
            rgb[vv, uu] = C[ok][order]
            glab[vv, uu] = lab[ok][order]
            depth[~np.isfinite(depth)] = 0.0
            # background wall far behind (valid depth, not object)
            bg = depth == 0
            depth[bg] = 2.0
            stem = f"frame_{int(a.rgbd_frame_index[t]):06d}"
            for d in ("rgb", "depth_npy", "mask"):
                (root / d / cid).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / "rgb" / cid / f"{stem}.png"), rgb[:, :, ::-1])
            np.save(root / "depth_npy" / cid / f"{stem}.npy", depth.astype(np.float32))
            cv2.imwrite(str(root / "mask" / cid / f"{stem}.png"), ((~bg).astype(np.uint8) * 255))
            gt[(cid, t)] = glab
    return gt


def two_cameras():
    K = np.array([[120.0, 0, 79.5], [0, 120.0, 59.5], [0, 0, 1]])
    return {"cam_a": CameraEntry("cam_a", K, look_at((0.15, -0.55, 0.40), (0, 0, 0.05)), 160, 120),
            "cam_b": CameraEntry("cam_b", K, look_at((-0.45, -0.35, 0.45), (0, 0, 0.05)), 160, 120)}


@unittest.skipUnless(HAS_CUDA, "dense support runs on CUDA in the freeartgs environment")
class DenseSupport(unittest.TestCase):
    def _run(self, a_true, a_hyp, moving=True):
        from estimotion_2part.joint_discovery.dense_support import DenseConfig, run_dense_support
        d = Path(tempfile.mkdtemp())
        gt = write_synthetic_scene(d / "scene", a_true, moving=moving)
        cfg = DenseConfig(work_downscale=1, sigma_depth_m0=0.001, sigma_depth_m2=0.0)
        n = a_hyp.T * len(a_hyp.cameras)
        m = run_dense_support(d / "scene", a_hyp, "test", list(a_hyp.cameras), d / "dense", np.ones(n, bool), cfg)
        return d, gt, m

    def _confusion(self, d, gt, a):
        import cv2
        c = np.zeros((2, 3), int)   # gt (static, moving) x label (S, M, U)
        for (cid, t), g in gt.items():
            stem = f"frame_{int(a.rgbd_frame_index[t]):06d}"
            L = [cv2.imread(str(d / "dense" / cid / k / f"{stem}.png"), 0) > 0 for k in ("mask_0", "mask_1", "mask_unknown")]
            for gi in (0, 1):
                for li in range(3):
                    c[gi, li] += int(((g == gi) & L[li]).sum())
        return c

    def test_I09_hinge_labels(self):
        a = make_authority("revolute", T=10, span=0.6, cams=two_cameras(), origin=(0, 0.1, 0.1))
        d, gt, m = self._run(a, a)
        c = self._confusion(d, gt, a)
        precision_m = c[1, 1] / max(c[:, 1].sum(), 1)
        precision_s = c[0, 0] / max(c[:, 0].sum(), 1)
        recall_m = c[1, 1] / max(c[1].sum(), 1)
        self.assertGreater(precision_m, 0.95, c)
        self.assertGreater(precision_s, 0.95, c)
        self.assertGreater(recall_m, 0.4, c)                 # the rest is UNKNOWN (small displacement near the hinge), never wrong
        self.assertGreater(m["totals"]["unknown"], 0)        # unknown is a real label (near hinge / unobserved)

    def test_I08_static_scene_has_no_moving_support(self):
        a = make_authority("revolute", T=10, span=0.6, cams=two_cameras(), origin=(0, 0.1, 0.1))
        d, gt, m = self._run(a, a, moving=False)               # scene does not move, hypothesis says it does
        c = self._confusion(d, gt, a)
        self.assertLess(c[:, 1].sum(), 0.02 * c.sum(), c)

    def test_I09b_wrong_axis_does_not_create_confident_moving_lid(self):
        a = make_authority("revolute", T=10, span=0.6, cams=two_cameras(), origin=(0, 0.1, 0.1))
        wrong = make_authority("prismatic", T=10, span=0.08, cams=two_cameras(), axis=(0, 0, 1))
        d, gt, m = self._run(a, wrong)
        c = self._confusion(d, gt, wrong)
        self.assertLess(c[0, 1], 0.02 * c[0].sum(), c)          # static base never becomes moving

    def test_G2_dataset_builder_partition_and_cameras(self):
        from preprocessing.prepare_joint_aware_multiview import build_joint_aware_dataset
        a = make_authority("revolute", T=12, span=0.6, cams=two_cameras(), origin=(0, 0.1, 0.1))
        d, gt, m = self._run(a, a)
        aid = a.write(d / "auth")
        lm = json.loads((d / "dense" / "label_manifest.json").read_text())
        lm["authority_id"] = aid
        (d / "dense" / "label_manifest.json").write_text(json.dumps(lm))
        man = build_joint_aware_dataset(d / "scene", d / "auth", d / "dense", d / "ds", list(a.cameras), [4, 5])
        self.assertLess(man["virtual_camera_max_error"], 1e-12)
        t0 = json.loads((d / "ds" / "transforms_0.json").read_text())
        t1 = json.loads((d / "ds" / "transforms_1.json").read_text())
        self.assertEqual(len(t0["eval_filenames"]), 4)
        self.assertFalse(set(t0["eval_filenames"]) & set(t0["train_filenames"]))
        for f0, f1 in zip(t0["frames"], t1["frames"]):
            A = np.asarray(f0["transform_matrix"]) @ np.linalg.inv(np.asarray(f1["transform_matrix"]))
            np.testing.assert_allclose(A, a.transform(f0["time_index"]), atol=1e-12)
        bad = json.loads((d / "dense" / "label_manifest.json").read_text())
        bad["authority_id"] = "other"
        (d / "dense" / "label_manifest.json").write_text(json.dumps(bad))
        with self.assertRaises(AuthorityError):
            build_joint_aware_dataset(d / "scene", d / "auth", d / "dense", d / "ds2", list(a.cameras), [4])


@unittest.skipIf(torch is None, "torch unavailable")
class LossSemantics(unittest.TestCase):
    def test_I13_unknown_and_other_part_pixels_get_zero_gradient(self):
        from nerfstudio.models.joint_aware_losses import joint_aware_part_losses
        H, W = 8, 8
        pred = torch.rand(H, W, 3, requires_grad=True)
        alpha = torch.rand(H, W, 1, requires_grad=True)
        depth = torch.rand(H, W, 1, requires_grad=True)
        gt = torch.rand(H, W, 3)
        gtd = torch.rand(H, W, 1) + 0.5
        pos = torch.zeros(H, W, 1, dtype=torch.bool)
        pos[:4] = True                       # this part
        sil = pos.clone()
        sil[6:] = True                       # verified background
        # rows 4-5: unknown / other part -> must receive no gradient of any kind
        out = joint_aware_part_losses(pred, gt, alpha, pos, sil, depth, gtd)
        sum(v for k, v in out.items() if not k.startswith("_")).backward()
        self.assertEqual(float(pred.grad[4:6].abs().sum()), 0.0)
        self.assertEqual(float(alpha.grad[4:6].abs().sum()), 0.0)
        self.assertEqual(float(depth.grad[4:].abs().sum()), 0.0)
        self.assertGreater(float(alpha.grad[6:].abs().sum()), 0.0)     # verified absence pushes alpha down
        self.assertGreater(float(pred.grad[:4].abs().sum()), 0.0)
        empty = joint_aware_part_losses(pred, gt, alpha, torch.zeros_like(pos), torch.zeros_like(pos))
        self.assertTrue(all(torch.isfinite(v) for v in empty.values()))


class Tripwires(unittest.TestCase):
    def test_I26_legacy_stages_refuse_inside_joint_aware_runs(self):
        env = dict(os.environ, FORCEARTGS_JOINT_AWARE_TRIPWIRE="1", PYTHONPATH=str(REPO))
        for script in ("preprocessing/estimate_joint.py", "icp/color_icp_gpu.py", "estimotion_2part/track_2part.py",
                       "preprocessing/segment.py"):
            p = subprocess.run([sys.executable, script, "--help"], cwd=REPO, env=env, capture_output=True, text=True)
            if script == "preprocessing/estimate_joint.py":
                code = ("import sys; sys.path.insert(0,'.'); from pathlib import Path; import tempfile;"
                        "from preprocessing.estimate_joint import process_twopart_blend_dir;"
                        "process_twopart_blend_dir(Path(tempfile.mkdtemp()))")
                p = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0, script)
            self.assertIn("joint-aware run", p.stderr, script)
        code = ("import sys; sys.path.insert(0,'.'); from preprocessing.prepare_multiview import prepare_multiview;"
                "prepare_multiview(None, {}, None)")
        p = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
        self.assertIn("joint-aware run", p.stderr)

    def test_I26_estimate_joint_refuses_joint_aware_blend_dir(self):
        from preprocessing.estimate_joint import process_twopart_blend_dir
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "JOINT_AWARE_AUTHORITY.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                process_twopart_blend_dir(Path(d))


if __name__ == "__main__":
    unittest.main()
