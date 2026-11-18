"""Stage F tests (§10 items 10 and 19): actual gsplat rasterizer feedback for joint hypotheses.

Run: cd /home/samuelemara/ForceArtGS && python -m unittest tests.test_joint_discovery_gaussian_renderer -v

Every renderer test goes through ``gaussian_feedback.render`` -> gsplat 0.1.11 ``project_gaussians`` +
``rasterize_gaussians`` (the FreeArtGS splatfacto path, CUDA float32). Tests skip with an explicit reason when
CUDA or the gsplat CUDA backend is unavailable. Images are 64x48 .. 96x72. Seeds are fixed.
Float32 tolerances are stated per test.
"""
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    from estimotion_2part.joint_discovery import gaussian_feedback as gf  # noqa: E402
    from estimotion_2part.joint_discovery import rigid_kinematics as rk  # noqa: E402
    _BACKEND = gf.ensure_gsplat_backend()
else:  # pragma: no cover
    gf = rk = None
    _BACKEND = dict(available=False, reason="torch not installed")

SKIP_REASON = f"gsplat CUDA rasterizer unavailable: {_BACKEND['reason']}"
DEV = "cuda"


# --------------------------------------------------------------------------- synthetic helpers
def look_at_T_WC(center, target, up=(0.0, -1.0, 0.0)):
    """OpenCV camera-to-world pose (z forward, y down) at ``center`` looking at ``target``."""
    c, t = np.asarray(center, float), np.asarray(target, float)
    z = (t - c) / np.linalg.norm(t - c)
    x = np.cross(-np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = c
    return T


def make_calib(T_WC, W=80, H=60, f=70.0, cid="cam0"):
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
    return gf.SimpleCalib(K, np.asarray(T_WC, float), W, H, cid)


def make_plane(center, u_dir, v_dir, size_u, size_v, spacing, opacity=0.95, color_fn=None, thickness=0.002):
    """Grid of flattened Gaussians lying in the plane span(u, v) (WXYZ quats from the frame [u v n])."""
    u = np.asarray(u_dir, float) / np.linalg.norm(u_dir)
    v = np.asarray(v_dir, float) / np.linalg.norm(v_dir)
    n = np.cross(u, v)
    nu, nv = int(round(size_u / spacing)) + 1, int(round(size_v / spacing)) + 1
    a, b = np.meshgrid(np.linspace(-size_u / 2, size_u / 2, nu), np.linspace(-size_v / 2, size_v / 2, nv))
    means = np.asarray(center, float) + a.reshape(-1, 1) * u + b.reshape(-1, 1) * v
    R = torch.tensor(np.stack([u, v, n], 1), dtype=torch.float32)
    q = gf.rotmat_to_quat_wxyz(R)[None].repeat(len(means), 1)
    s = torch.log(torch.tensor([spacing * 0.75, spacing * 0.75, thickness], dtype=torch.float32)).repeat(len(means), 1)
    if color_fn is None:
        col = np.full((len(means), 3), 0.6)
    else:
        col = np.stack([color_fn(ai, bi) for ai, bi in zip(a.ravel(), b.ravel())])
    f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=DEV)  # noqa: E731
    return gf.GaussianSet(f(means), q.to(DEV), s.to(DEV), torch.full((len(means), 1), opacity, device=DEV),
                          f(col), "wxyz")


def checker(a, b, k=0.08):
    i = int(np.floor(a / k)) + int(np.floor(b / k))
    return np.array([0.9, 0.3, 0.2]) if i % 2 else np.array([0.2, 0.5, 0.9])


def smooth_tex(a, b):
    return np.array([0.5 + 0.4 * np.sin(6 * a), 0.5 + 0.4 * np.cos(5 * b), 0.5 + 0.3 * np.sin(4 * (a + b))])


HINGE_O = np.array([-0.25, 0.0, 2.2])
HINGE_U = np.array([0.0, 1.0, 0.0])


def door_scene(spacing=0.03, tex=checker):
    wall = make_plane([0, 0, 3.0], [1, 0, 0], [0, 1, 0], 1.8, 1.4, 0.06, opacity=0.95,
                      color_fn=lambda a, b: np.array([0.35, 0.35, 0.3]) + 0.1 * np.sin(3 * a))
    door = make_plane([0.0, 0.0, 2.2], [1, 0, 0], [0, 1, 0], 0.5, 0.6, spacing, opacity=0.95, color_fn=tex)
    return wall, door


def measure(r, robot_mask=None):
    """'Measured' arrays from a ground-truth render (synthetic GT is used only to create data)."""
    depth = r.depth.detach().cpu().numpy().copy()
    depth[~r.depth_valid.cpu().numpy()] = 0.0
    out = dict(rgb=r.rgb.detach().cpu().numpy(), depth=depth, object_mask=r.alpha.detach().cpu().numpy() > 0.5)
    if robot_mask is not None:
        out["robot_mask"] = robot_mask
    return out


def cov_full(c6):
    c = c6
    return torch.stack([torch.stack([c[..., 0], c[..., 1], c[..., 2]], -1),
                        torch.stack([c[..., 1], c[..., 3], c[..., 4]], -1),
                        torch.stack([c[..., 2], c[..., 4], c[..., 5]], -1)], -2)


def image_principal_angle(alpha):
    a = alpha.detach().cpu().numpy().astype(float)
    ys, xs = np.mgrid[:a.shape[0], :a.shape[1]]
    w = a / a.sum()
    mx, my = (w * xs).sum(), (w * ys).sum()
    cxx, cyy, cxy = (w * (xs - mx) ** 2).sum(), (w * (ys - my) ** 2).sum(), (w * (xs - mx) * (ys - my)).sum()
    return 0.5 * np.degrees(np.arctan2(2 * cxy, cxx - cyy)), (cxx, cyy, cxy)


@unittest.skipUnless(torch is not None and _BACKEND["available"], SKIP_REASON)
class TestGaussianRendererStageF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        np.random.seed(0)
        cls.t0 = time.time()
        print(f"\n[gsplat backend] {_BACKEND['source']}; GPU {torch.cuda.get_device_name(0)}; torch {torch.__version__}")

    # ------------------------------------------------------------------ quaternion / covariance
    def test_rotated_covariance_through_rasterizer_equals_R_Sigma_RT(self):
        """CUDA ``project_gaussians`` cov3d of A·G equals R Σ Rᵀ (float32 rtol 1e-4); xyzw/wxyz swap detected."""
        cal = make_calib(np.eye(4))
        q = torch.tensor([[0.8, 0.2, 0.3, 0.4]], device=DEV)
        q = q / q.norm()
        g = gf.GaussianSet(torch.tensor([[0.05, -0.02, 2.0]], device=DEV), q,
                           torch.log(torch.tensor([[0.01, 0.05, 0.2]], device=DEV)),
                           torch.full((1, 1), 0.8, device=DEV), torch.full((1, 3), 0.5, device=DEV), "wxyz")
        A = rk.A_revolute(np.array([0.3, -0.5, 0.8]), np.array([0.1, 0.2, 1.9]), 0.7)
        R = torch.tensor(A[:3, :3], dtype=torch.float32, device=DEV)
        sig0 = cov_full(gf.render(None, g, None, cal).cov3d)[0]
        sig_ref = gf.gaussian_covariances(g)[0]  # gsplat torch quat->rotmat reference (WXYZ)
        torch.testing.assert_close(sig0, sig_ref, rtol=1e-4, atol=1e-8)
        sig1 = cov_full(gf.render(None, g, A, cal).cov3d)[0]
        torch.testing.assert_close(sig1, R @ sig0 @ R.T, rtol=1e-4, atol=1e-7)
        # transformed mean too
        gA = gf.transform_gaussians(g, A)
        torch.testing.assert_close(gA.means[0], torch.tensor(rk.transform_points(A, np.array([0.05, -0.02, 2.0])),
                                                             dtype=torch.float32, device=DEV), rtol=0, atol=1e-6)
        # same physical Gaussian declared in XYZW must give the identical covariance through the rasterizer
        g_xyzw = gf.GaussianSet(g.means, q[:, [1, 2, 3, 0]], g.log_scales, g.opacities, g.colors, "xyzw")
        torch.testing.assert_close(cov_full(gf.render(None, g_xyzw, A, cal).cov3d)[0], sig1, rtol=1e-4, atol=1e-7)
        # wrong declaration (WXYZ numbers labelled XYZW) is detected: covariance is far from R Σ Rᵀ
        g_wrong = gf.GaussianSet(g.means, q, g.log_scales, g.opacities, g.colors, "xyzw")
        sig_wrong = cov_full(gf.render(None, g_wrong, A, cal).cov3d)[0]
        self.assertGreater(float((sig_wrong - sig1).abs().max() / sig1.abs().max()), 0.1)

    def test_layout_matches_repo_apply_left_se3_to_gaussians(self):
        """Our XYZW path agrees with the repo helper ``apply_left_se3_to_gaussians`` (if importable)."""
        try:
            from nerfstudio.data.utils.dataparsers_utils import apply_left_se3_to_gaussians
            import pytorch3d  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"nerfstudio/pytorch3d not importable: {exc!r}")
        rng = np.random.default_rng(3)
        q = torch.tensor(rng.normal(size=(5, 4)), dtype=torch.float32)
        q = q / q.norm(dim=-1, keepdim=True)
        A = torch.tensor(rk.A_revolute(np.array([1.0, 2.0, -0.5]), np.zeros(3), 1.1), dtype=torch.float32)
        ref = apply_left_se3_to_gaussians(dict(means=torch.zeros(5, 3), quats=q, scales=torch.zeros(5, 3),
                                               features_dc=torch.zeros(5, 3), features_rest=torch.zeros(5, 0, 3),
                                               opacities=torch.zeros(5, 1)), A, torch.device("cpu"))["quats"]
        g = gf.GaussianSet(torch.zeros(5, 3), q, torch.zeros(5, 3), torch.ones(5, 1) * .5, torch.zeros(5, 3), "xyzw")
        ours = gf.transform_gaussians(g, A).quats
        sign = torch.sign((ours * ref).sum(-1, keepdim=True))
        torch.testing.assert_close(ours * sign, ref, rtol=1e-5, atol=1e-5)

    def test_image_space_orientation_follows_rotation(self):
        """Rendered alpha image (not only cov3d) of an elongated Gaussian rotates by the applied angle (±1.5°)."""
        cal = make_calib(np.eye(4), W=96, H=72, f=80.0)
        c = np.array([0.0, 0.0, 2.0])
        g = gf.GaussianSet(torch.tensor(np.array([c]), dtype=torch.float32, device=DEV), torch.tensor([[1.0, 0, 0, 0]], device=DEV),
                           torch.log(torch.tensor([[0.25, 0.05, 0.01]], device=DEV)), torch.full((1, 1), 0.5, device=DEV),
                           torch.ones(1, 3, device=DEV), "wxyz")
        ang0, _ = image_principal_angle(gf.render(None, g, None, cal).alpha)
        for deg in (30.0, 60.0, -45.0):
            A = rk.A_revolute(np.array([0, 0, 1.0]), c, np.radians(deg))
            ang, _ = image_principal_angle(gf.render(None, g, A, cal).alpha)
            d = (ang - ang0 - deg + 90) % 180 - 90
            self.assertLess(abs(d), 1.5, f"rotation {deg}: image angle change {ang - ang0}")

    # ------------------------------------------------------------------ depth semantics
    def test_metric_depth_is_camera_z(self):
        """Opaque fronto-parallel cluster at known camera-Z (off-axis) renders depth == Z within 2 mm (float32);
        a second layer 1 m behind does not leak through; a tiny Gaussian lands on its OpenCV pixel."""
        T_WC = look_at_T_WC([0.2, -0.1, -0.3], [0.2, -0.1, 1.0])
        cal = make_calib(T_WC, W=80, H=60, f=70.0)
        Z = 1.5
        R_wc = T_WC[:3, :3]
        center_c = np.array([0.25, 0.1, Z])  # off-axis: ray length != Z
        center_w = R_wc @ center_c + T_WC[:3, 3]
        front = make_plane(center_w, R_wc[:, 0], R_wc[:, 1], 0.5, 0.4, 0.01, opacity=0.99, thickness=0.001)
        back = make_plane(R_wc @ np.array([0.25, 0.1, Z + 1.0]) + T_WC[:3, 3], R_wc[:, 0], R_wc[:, 1], 0.5, 0.4, 0.01,
                          opacity=0.99, thickness=0.001)
        r = gf.render(back, front, None, cal)
        K = cal.K
        u0, v0 = K[0, 0] * center_c[0] / Z + K[0, 2], K[1, 1] * center_c[1] / Z + K[1, 2]
        hu, hv = 0.18 * K[0, 0] / Z, 0.13 * K[1, 1] / Z  # interior, away from the edges
        rows = slice(int(np.ceil(v0 - hv)), int(np.floor(v0 + hv)))
        cols = slice(int(np.ceil(u0 - hu)), int(np.floor(u0 + hu)))
        d = r.depth[rows, cols]
        self.assertTrue(bool(r.depth_valid[rows, cols].all()))
        self.assertLess(float((d - Z).abs().max()), 2e-3, f"max |depth-Z| = {float((d - Z).abs().max())}")
        self.assertLess(float(r.depth_std[rows, cols].max()), 0.03)
        self.assertGreater(float(r.alpha[rows, cols].min()), 0.99)
        # pixel-centre convention: tiny Gaussian projected to integer OpenCV pixel (u=30, v=20) peaks there
        cal0 = make_calib(np.eye(4), W=64, H=48, f=60.0)
        u, v, z = 30, 20, 2.0
        X = np.array([(u - cal0.K[0, 2]) * z / 60.0, (v - cal0.K[1, 2]) * z / 60.0, z])
        tiny = gf.GaussianSet(torch.tensor(np.array([X]), dtype=torch.float32, device=DEV), torch.tensor([[1.0, 0, 0, 0]], device=DEV),
                              torch.log(torch.full((1, 3), 0.008, device=DEV)), torch.full((1, 1), 0.9, device=DEV),
                              torch.ones(1, 3, device=DEV))
        a = gf.render(None, tiny, None, cal0).alpha
        self.assertEqual(tuple(int(i) for i in np.unravel_index(int(a.argmax()), a.shape)), (v, u))
        self.assertAlmostEqual(float(a[v, u - 1]), float(a[v, u + 1]), places=5)
        self.assertAlmostEqual(float(a[v - 1, u]), float(a[v + 1, u]), places=5)

    # ------------------------------------------------------------------ virtual camera equivalence
    def test_virtual_camera_equivalence_rgb_alpha_depth(self):
        """Physical camera + A_H(q)·G == canonical G from virtual camera A_H(q)^-1 T_WC (RGB/alpha 2e-3, depth 1 mm)."""
        _, door = door_scene(tex=checker)
        for kind, axis, origin, q in (("revolute", HINGE_U, HINGE_O, 0.45), ("prismatic", np.array([0.3, 0, -1.0]), None, 0.15)):
            for T_WC in (np.eye(4), look_at_T_WC([0.7, -0.3, 0.1], [0, 0, 2.3])):
                cal = make_calib(T_WC)
                A = rk.A_joint(kind, axis, origin if origin is not None else np.zeros(3), q)
                r_phys = gf.render(None, door, A, cal)
                r_virt = gf.render(None, door, None, gf.virtual_camera_calib(cal, A))
                self.assertGreater(float(r_phys.alpha.sum()), 50.0)
                self.assertLess(float((r_phys.rgb - r_virt.rgb).abs().max()), 2e-3)
                self.assertLess(float((r_phys.alpha - r_virt.alpha).abs().max()), 2e-3)
                both = r_phys.depth_valid & r_virt.depth_valid
                self.assertTrue(bool((r_phys.depth_valid == r_virt.depth_valid).float().mean() > 0.999))
                self.assertLess(float((r_phys.depth - r_virt.depth)[both].abs().max()), 1e-3)
                # and the virtual pose is exactly exports.virtual_camera_poses' formula
                np.testing.assert_allclose(gf.virtual_camera_calib(cal, A).T_WC, rk.se3_inverse(A) @ T_WC, atol=1e-12)

    # ------------------------------------------------------------------ occlusion / provenance
    def test_occlusion_and_robot_pixels_get_no_moving_depth_loss(self):
        wall, door = door_scene(tex=checker)
        occluder = make_plane([0.05, 0.0, 1.6], [1, 0, 0], [0, 1, 0], 0.16, 0.16, 0.01, opacity=0.99)
        static = gf.concat_gaussians([wall, occluder])
        cal = make_calib(np.eye(4))
        A = rk.A_revolute(HINGE_U, HINGE_O, 0.3)
        A_t = torch.tensor(A, dtype=torch.float32, device=DEV)
        gt = gf.render(static, door, A_t, cal)
        robot = np.zeros((cal.height, cal.width), bool)
        robot[8:18, 22:30] = True  # a tool over part of the door
        m = measure(gt, robot)
        prov = gf.pixel_provenance(m["depth"], m["object_mask"], robot)
        self.assertTrue(bool((prov[robot] == gf.OCCLUDED_BY_ROBOT).all()))
        sets = gf.loss_pixel_sets(prov)
        for k in sets:
            self.assertFalse(bool(sets[k][robot].any()), f"robot pixel in {k} set")
        w = gf.depth_loss_weights(prov, gt, m["depth"], part="moving").cpu().numpy()
        occ_pix = (gt.moving_fraction.cpu().numpy() < 0.05) & (np.abs(m["depth"] - 1.6) < 0.02)
        door_pix = (gt.moving_fraction.cpu().numpy() > 0.95) & (prov == gf.OBJECT_VISIBLE)
        self.assertGreater(int(occ_pix.sum()), 20)
        self.assertGreater(int(door_pix.sum()), 100)
        self.assertFalse(bool(w[occ_pix].any()), "static occluder pixels contribute to moving-part depth loss")
        self.assertFalse(bool(w[robot].any()))
        self.assertGreater(w[door_pix].mean(), 0.9)
        # corrupting measured depth behind the occluder / under the robot changes neither loss nor dL/dq
        def moving_depth_loss(meas):
            q = torch.tensor(0.27, device=DEV, requires_grad=True)
            r = gf.render(static, door, gf.joint_transform("revolute", HINGE_U, HINGE_O, q), cal)
            wts = gf.depth_loss_weights(gf.pixel_provenance(meas["depth"], meas["object_mask"], meas.get("robot_mask")),
                                        r, meas["depth"], part="moving")
            L = gf.render_losses(r, meas, gf.loss_pixel_sets(prov), depth_weights=wts)["L_depth_m"]
            L.backward()
            return float(L), float(q.grad), wts
        L0, g0, w0 = moving_depth_loss(m)
        bad = dict(m)
        bad["depth"] = m["depth"].copy()
        bad["depth"][occ_pix] = 2.9
        bad["depth"][robot] = 0.4
        L1, g1, w1 = moving_depth_loss(bad)
        self.assertTrue(bool(torch.equal(w0, w1)))
        self.assertAlmostEqual(L0, L1, places=6)
        self.assertAlmostEqual(g0, g1, places=5)

    def test_provenance_missing_depth_and_edges(self):
        D = np.full((20, 30), 2.0)
        obj = np.zeros((20, 30), bool)
        obj[4:16, 5:25] = True
        D[8, 10] = 0.0
        D[9, 12] = np.nan
        D[10, 14] = -np.inf
        D[5:15, 18:25] = 1.0  # depth step inside the mask
        rob = np.zeros_like(obj)
        rob[0:2, 0:3] = True
        prov = gf.pixel_provenance(D, obj, rob)
        for rc in ((8, 10), (9, 12), (10, 14)):
            self.assertEqual(prov[rc], gf.MISSING_DEPTH)
        self.assertEqual(prov[10, 18], gf.AMBIGUOUS)  # depth discontinuity
        self.assertEqual(prov[4, 10], gf.AMBIGUOUS)  # mask edge
        self.assertEqual(prov[10, 8], gf.OBJECT_VISIBLE)
        self.assertEqual(prov[19, 29], gf.BACKGROUND)
        self.assertEqual(prov[0, 0], gf.OCCLUDED_BY_ROBOT)
        self.assertFalse(gf.loss_pixel_sets(prov)["depth"][8, 10])

    # ------------------------------------------------------------------ gradients vs finite differences
    def _fd_check(self, kind, axis, origin, q_true, q0, h, also_axis=False, axis_eval=None):
        """GT data rendered with ``axis``; loss evaluated (and differentiated) at ``axis_eval``/``q0``.

        Alpha compositing is only piecewise smooth: when two overlapping Gaussians have *equal* camera depth
        (e.g. one grid column of a plane rotated about the exact y axis) an infinitesimal axis tilt flips their
        sort order and the loss jumps. FD across such a tie is meaningless, so ``axis_eval`` is a generic
        (tie-free) direction."""
        wall, door = door_scene(spacing=0.03, tex=smooth_tex)
        cals = [make_calib(np.eye(4), W=64, H=48, f=56.0), make_calib(look_at_T_WC([0.6, -0.2, 0.2], [0, 0, 2.4]),
                                                                       W=64, H=48, f=56.0)]
        data = []
        for cal in cals:
            gt = gf.render(wall, door, gf.joint_transform(kind, axis, origin, q_true), cal)
            m = measure(gt)
            data.append((cal, m, gf.loss_pixel_sets(gf.pixel_provenance(m["depth"], m["object_mask"]))))

        def loss(q, ax):
            tot = 0
            for cal, m, sets in data:
                r = gf.render(wall, door, gf.joint_transform(kind, ax, origin, q), cal)
                tot = tot + gf.render_losses(r, m, sets)["total"]
            return tot
        q = torch.tensor(q0, device=DEV, requires_grad=True)
        axis_eval = axis if axis_eval is None else axis_eval
        ax = torch.tensor(np.asarray(axis_eval, float), dtype=torch.float32, device=DEV, requires_grad=True)
        L = loss(q, ax)
        L.backward()
        with torch.no_grad():
            fd = (float(loss(torch.tensor(q0 + h, device=DEV), ax)) - float(loss(torch.tensor(q0 - h, device=DEV), ax))) / (2 * h)
        g = float(q.grad)
        print(f"  [{kind}] dL/dq autograd={g:.6g} central-FD(h={h})={fd:.6g} rel.err={abs(g - fd) / abs(fd):.3%}")
        self.assertGreater(abs(fd), 1e-2)
        self.assertLess(abs(g - fd) / abs(fd), 0.05)
        self.assertTrue(bool(torch.isfinite(ax.grad).all()) and float(ax.grad.abs().sum()) > 0)
        if also_axis:  # one axis component (axis is normalised inside A_joint)
            e = torch.zeros(3, device=DEV)
            e[0] = 1.0
            with torch.no_grad():
                hx = 2e-3
                fdx = (float(loss(torch.tensor(q0, device=DEV), ax + hx * e)) - float(loss(torch.tensor(q0, device=DEV), ax - hx * e))) / (2 * hx)
            print(f"  [{kind}] dL/daxis_x autograd={float(ax.grad[0]):.6g} central-FD(h={hx})={fdx:.6g}")
            self.assertGreater(abs(fdx), 1e-2)
            self.assertLess(abs(float(ax.grad[0]) - fdx) / abs(fdx), 0.10)

    def test_gradient_revolute_q_matches_finite_difference(self):
        """Float32 gsplat: dL/dq within 5% of central FD (h = 0.01 rad); dL/daxis_x within 10% (h = 2e-3)."""
        self._fd_check("revolute", HINGE_U, HINGE_O, q_true=0.35, q0=0.25, h=1e-2, also_axis=True,
                       axis_eval=np.array([0.06, 1.0, 0.04]))

    def test_gradient_prismatic_q_matches_finite_difference(self):
        """Float32 gsplat: |autograd - central FD| / |FD| < 5% (h = 4 mm)."""
        self._fd_check("prismatic", np.array([0.2, 0.0, -1.0]), None, q_true=0.12, q0=0.08, h=4e-3)

    # ------------------------------------------------------------------ frozen-asset fairness
    def _two_cam_measurements(self, wall, door, kind, axis, origin, qs, robot=False):
        cams = {"cam0": make_calib(np.eye(4), cid="cam0"),
                "cam1": make_calib(look_at_T_WC([0.8, -0.25, 0.3], [0, 0, 2.5]), cid="cam1")}
        meas = {}
        for cid, cal in cams.items():
            meas[cid] = {}
            for t, q in enumerate(qs):
                A = None if kind == "static" else gf.joint_transform(kind, axis, origin, q)
                rmask = None
                if robot and cid == "cam0":
                    rmask = np.zeros((cal.height, cal.width), bool)
                    rmask[20:30, 44:52] = True
                m = measure(gf.render(wall, door, A, cal), rmask)
                ev = np.zeros((cal.height, cal.width), bool)
                ev[::2, :] = True  # held-out pixel subset: even rows
                m["eval_mask"] = ev
                meas[cid][t] = m
        return cams, meas

    def test_frozen_scoring_prefers_true_revolute_on_identical_pixels(self):
        wall, door = door_scene(tex=checker)
        qs = [0.0, 0.25, 0.45]
        cams, meas = self._two_cam_measurements(wall, door, "revolute", HINGE_U, HINGE_O, qs, robot=True)
        door_center = np.array([0.0, 0.0, 2.2])
        cands = [dict(name="static", kind="static"),
                 dict(name="revolute_true", kind="revolute", axis_w=HINGE_U, origin_w=HINGE_O, q=qs),
                 dict(name="prismatic_wrong", kind="prismatic", axis_w=np.array([0.3, 0, -1.0]),
                      q=[0.0, 0.08, 0.15]),
                 dict(name="revolute_wrong_hinge", kind="revolute", axis_w=HINGE_U,
                      origin_w=door_center + np.array([0.25, 0, 0]), q=[0.0, -0.25, -0.45])]
        fp = (wall.fingerprint(), door.fingerprint())
        res = gf.score_candidates_frozen(wall, door, cands, cams, range(len(qs)), meas)
        self.assertEqual(res["renderer_evaluation_kind"], "actual_gaussian")
        self.assertEqual(res["best"], "revolute_true", res["ranking"])
        self.assertEqual((wall.fingerprint(), door.fingerprint()), fp)
        self.assertLess(res["best_minus_static"], 0)
        for c in res["candidates"]:
            self.assertEqual(set(c["per_camera"]), {"cam0", "cam1"})
        tot = {c["name"]: c["total"] for c in res["candidates"]}
        print("  [frozen scoring, revolute GT] totals:", {k: round(v, 4) for k, v in tot.items()})
        # per camera too
        for cid in cams:
            per = {c["name"]: c["per_camera"][cid]["total"] for c in res["candidates"]}
            self.assertEqual(min(per, key=per.get), "revolute_true", (cid, per))
        # identical pixels: sets derive from measurements only; robot pixels excluded
        ps = res["pixel_sets"]["cam0"][1]
        self.assertGreater(ps["provenance_counts"]["occluded_by_robot"], 0)
        self.assertGreater(ps["n_depth"], 100)

    def test_static_scene_does_not_favour_moving_hypothesis(self):
        wall, door = door_scene(tex=checker)
        qs = [0.0, 0.0, 0.0]
        cams, meas = self._two_cam_measurements(wall, door, "static", None, None, qs)
        cands = [dict(name="revolute", kind="revolute", axis_w=HINGE_U, origin_w=HINGE_O, q=[0.0, 0.2, 0.4]),
                 dict(name="static", kind="static"),
                 dict(name="prismatic", kind="prismatic", axis_w=np.array([0, 0, -1.0]), q=[0.0, 0.05, 0.1])]
        res = gf.score_candidates_frozen(wall, door, cands, cams, range(3), meas)
        self.assertEqual(res["best"], "static", res["ranking"])
        self.assertAlmostEqual(res["best_minus_static"], 0.0)
        tot = {c["name"]: c["total"] for c in res["candidates"]}
        for cid in cams:  # GT-consistent static: RGB and depth residuals vanish; alpha keeps the binary-mask floor
            st = next(c for c in res["candidates"] if c["name"] == "static")["per_camera"][cid]
            self.assertLess(st["L_rgb"], 1e-4)
            self.assertLess(st["L_depth_m"], 1e-4)
        self.assertLess(tot["static"], min(v for k, v in tot.items() if k != "static"))
        print("  [frozen scoring, static GT] totals:", {k: round(v, 4) for k, v in tot.items()})

    # ------------------------------------------------------------------ canonical asset loading (fail closed)
    def test_loader_fails_closed_and_reads_art_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = gf.load_canonical_asset("final_black_box_episode_000000", search_roots=[tmp])
            self.assertEqual(res["status"], gf.STATUS_MISSING)
            self.assertIsNone(res["moving"])
            run = Path(tmp) / "final_black_box_episode_000000" / "splatfacto_art" / "run"
            (run / "nerfstudio_models").mkdir(parents=True)
            n = 7
            q = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
            sd = {"_model.gauss_params.means": torch.randn(n, 3), "_model.gauss_params.quats": q,
                  "_model.gauss_params.scales": torch.full((n, 3), -3.0),
                  "_model.gauss_params.opacities": torch.zeros(n, 1),
                  "_model.gauss_params.features_dc": torch.zeros(n, 3),
                  "_model.gauss_params.features_rest": torch.zeros(n, 15, 3),
                  "_model.gauss_params.part_weights": torch.full((n, 1), 2.0)}
            torch.save({"step": 100, "pipeline": sd}, run / "nerfstudio_models" / "step-000000100.ckpt")
            res = gf.load_canonical_asset("final_black_box_episode_000000", search_roots=[tmp], motion_part=1)
            self.assertEqual(res["status"], gf.STATUS_INCOMPATIBLE)  # no config.yml -> unknown quaternion layout
            (run / "config.yml").write_text("model: !!python/object:SplatfactoArtModelConfig\n  sh_degree: 3\n"
                                            "  quaternion_order: xyzw\n")
            res = gf.load_canonical_asset("final_black_box_episode_000000", search_roots=[tmp], motion_part=1)
            self.assertEqual(res["status"], gf.STATUS_INCOMPATIBLE)  # no dataparser_transforms.json -> frame unknown
            Rn = rk.exp_so3(np.array([0.0, 0.0, np.pi / 2]))
            tn, sc = np.array([0.1, -0.2, 0.3]), 0.25
            (run / "dataparser_transforms.json").write_text(
                '{"transform": %s, "scale": %r}' % (np.concatenate([Rn, tn[:, None]], 1).tolist(), sc))
            res = gf.load_canonical_asset("final_black_box_episode_000000", search_roots=[tmp])
            self.assertEqual(res["status"], gf.STATUS_INCOMPATIBLE)  # motion_part unknown
            res = gf.load_canonical_asset("final_black_box_episode_000000", search_roots=[tmp], motion_part=1)
            self.assertEqual(res["status"], gf.STATUS_OK, res["reason"])
            self.assertEqual(res["quat_layout"], "xyzw")
            w = torch.sigmoid(torch.tensor(2.0))
            torch.testing.assert_close(res["moving"].opacities, torch.full((n, 1), 0.5 * float(w)))
            torch.testing.assert_close(res["static"].opacities + res["moving"].opacities, torch.full((n, 1), 0.5))
            self.assertTrue(any("hypothesis" in s for s in res["warnings"]))
            # world frame: x_ns = s (R_n x_w + t_n) inverted for means, scales and covariances
            xw = res["static"].means.numpy().astype(float)
            np.testing.assert_allclose(sc * (xw @ Rn.T + tn), sd["_model.gauss_params.means"].numpy(), atol=1e-5)
            torch.testing.assert_close(res["static"].log_scales, torch.full((n, 3), -3.0 - np.log(sc)))
            raw = gf.GaussianSet(sd["_model.gauss_params.means"], q, sd["_model.gauss_params.scales"],
                                 torch.full((n, 1), .5), torch.zeros(n, 3), "xyzw")
            Rt = torch.tensor(Rn, dtype=torch.float32)
            torch.testing.assert_close(gf.gaussian_covariances(res["static"]) * sc ** 2,
                                       Rt.T @ gf.gaussian_covariances(raw) @ Rt, rtol=1e-4, atol=1e-7)

    def test_loader_reads_existing_repo_splatfacto_art_checkpoint(self):
        """Read-only smoke test on an existing real splatfacto-art checkpoint (not a target scene)."""
        ck = Path(__file__).resolve().parents[1] / "outputs/USB_100109_multiview/splatfacto-art/nerfstudio_models/step-000008000.ckpt"
        if not ck.exists():
            self.skipTest(f"{ck} not on disk")
        res = gf.load_canonical_asset(checkpoint=str(ck))
        print(f"  [USB_100109_multiview splatfacto-art] status={res['status']} layout={res['quat_layout']} "
              f"({res['quat_layout_source']}) frame={res['frame']} motion_part={res.get('motion_part')} "
              f"N={res['static'].n if res['static'] is not None else 0}")
        self.assertEqual(res["status"], gf.STATUS_OK, res["reason"])
        self.assertEqual(res["quat_layout"], "wxyz")  # legacy quaternion_order => gsplat consumed raw WXYZ
        for g in (res["static"], res["moving"]):
            self.assertTrue(bool(torch.isfinite(g.means).all() and torch.isfinite(g.quats).all()))

    def test_repo_outputs_have_no_canonical_asset_for_target_scenes(self):
        """Reports (fail-closed) what the repo outputs contain; does not train anything."""
        root = Path(__file__).resolve().parents[1] / "outputs"
        for scene in ("final_black_box_episode_000000", "final_red_fridge_episode_000000"):
            res = gf.load_canonical_asset(scene, search_roots=[str(root)])
            print(f"  [{scene}] canonical asset status={res['status']} reason={res['reason']}")
            self.assertIn(res["status"], (gf.STATUS_MISSING, gf.STATUS_OK, gf.STATUS_INCOMPATIBLE))
            if res["status"] != gf.STATUS_OK:
                self.assertIsNone(res["moving"])


if __name__ == "__main__":
    unittest.main()
