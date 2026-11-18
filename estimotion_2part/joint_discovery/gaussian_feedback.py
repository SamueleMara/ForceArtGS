"""Stage F: actual Gaussian-renderer feedback for joint hypotheses (not a point proxy).

This module renders frozen canonical 3D Gaussians through the **existing FreeArtGS rasterizer
path** -- ``gsplat`` 0.1.11 ``project_gaussians`` + ``rasterize_gaussians``, exactly the two calls
made by ``reconstruction/nerfstudio/nerfstudio/models/splatfacto.py::get_outputs`` and
``splatfacto_art.py::get_outputs`` (classic rasterize mode, ``BLOCK_WIDTH = 16``, two passes: colour
pass with ``return_alpha=True`` and a second pass whose "colours" are per-Gaussian camera depths).

Conventions (shared with ``rigid_kinematics`` / ``observations``)
---------------------------------------------------------------
* ``T_AB`` maps frame-B coordinates to frame A, column vectors, metres.
* ``calib.T_WC`` is the OpenCV camera-to-world pose (x right, y down, z forward). The gsplat view
  matrix is ``T_CW = T_WC^-1`` directly (no OpenGL y/z flip: that flip in splatfacto only converts
  nerfstudio's OpenGL camera_to_worlds to OpenCV).
* The moving part is rendered from the **fixed physical camera** with its Gaussians transformed by
  ``A_H(q)`` (``rigid_kinematics.A_joint``); this is numerically equivalent to rendering canonical
  Gaussians from the virtual camera ``A_H(q)^-1 T_WC`` (``exports.virtual_camera_poses``), which the
  tests check for RGB, alpha and depth.
* Pixel centres: gsplat 0.1.11 evaluates output pixel ``(row i, col j)`` at continuous image
  coordinate ``(j + 0.5, i + 0.5)`` (verified empirically: a Gaussian projected to (20, 30) splits
  equally between columns 19/20 and rows 29/30). OpenCV intrinsics put the centre of pixel ``j`` at
  ``u = j``. ``render`` therefore passes ``cx + 0.5, cy + 0.5`` (``pixel_center_offset=0.5``) so that
  output pixel ``(i, j)`` is the OpenCV pixel ``(u=j, v=i)`` that measured depth/mask arrays index.

Quaternion layout (the XYZW/WXYZ boundary)
------------------------------------------
gsplat's CUDA kernels and ``gsplat._torch_impl.quat_to_rotmat`` read quaternions as **WXYZ**
(scalar first). ``GaussianSet.quats`` carries an explicit ``quat_layout`` field, ``"wxyz"``
(default, gsplat-native) or ``"xyzw"`` (the internal layout of ``apply_left_se3_to_gaussians`` and of
``SplatfactoModel`` with ``quaternion_order="xyzw"``). Every hand-off to gsplat goes through
``quats_for_gsplat`` which mirrors ``SplatfactoModel.quats_for_gsplat`` (``xyzw -> [3,0,1,2]``, then
normalise). ``transform_gaussians`` left-multiplies the rotation quaternion of ``A`` in Hamilton
WXYZ algebra and returns the input layout, so the rendered covariance is ``R Sigma R^T`` (tested
through the CUDA projection's returned ``cov3d``).

Depth formation (metric camera-Z)
---------------------------------
``project_gaussians`` returns, per Gaussian, ``z_i`` = camera-frame Z of its mean (metres; not
ray length). A second rasterization pass composites ``[z_i, m_i, z_i^2]`` with the same opacities and
a zero background, giving ``S_z = sum_i w_i z_i`` with ``w_i = alpha_i T_i`` (front-to-back
transmittance weights). With ``alpha = sum_i w_i`` from the colour pass:

    depth    = S_z / alpha               (alpha-normalised expected camera-Z, metres)
    depth_std= sqrt(max(E[z^2] - depth^2, 0))   (spread of composited depths: mixed surfaces)
    moving_fraction = sum_i w_i m_i / alpha     (z-buffer ownership: m_i = 1 for moving Gaussians)

``depth_valid = alpha >= depth_alpha_min`` (default 0.5); depth is 0 where invalid. This is the same
normalisation as splatfacto's ``depth_im / alpha`` but never fills invalid pixels with a max depth.
gsplat kernels are float32 only; tolerances in the tests are float32 tolerances.

Nothing in this module trains or updates Gaussians: candidate scoring is ``torch.no_grad`` on a single
frozen canonical asset whose fingerprint is checked before and after scoring.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.machinery
import importlib.util
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .rigid_kinematics import A_joint

RENDERER_EVALUATION_KIND = "actual_gaussian"
RENDERER_BACKEND = "gsplat==0.1.11 project_gaussians + rasterize_gaussians (FreeArtGS splatfacto path)"
DEPTH_DEFINITION = ("alpha-normalised expected camera-Z: sum_i w_i z_i / sum_i w_i, z_i = camera-frame Z of "
                    "Gaussian mean (metres), w_i = alpha_i * T_i; valid where alpha >= depth_alpha_min")
BLOCK_WIDTH = 16  # same tile size as splatfacto / splatfacto_art
QUAT_LAYOUTS = ("wxyz", "xyzw")

# canonical-asset statuses (output contract; fail closed)
STATUS_OK = "OK"
STATUS_MISSING = "MISSING_CANONICAL_ASSET"
STATUS_INCOMPATIBLE = "INCOMPATIBLE_CANONICAL_ASSET"

# pixel provenance codes
OBJECT_VISIBLE = 0
OCCLUDED_BY_ROBOT = 1
BACKGROUND = 2
MISSING_DEPTH = 3
AMBIGUOUS = 4
PROVENANCE_NAMES = {OBJECT_VISIBLE: "object_visible", OCCLUDED_BY_ROBOT: "occluded_by_robot",
                    BACKGROUND: "background", MISSING_DEPTH: "missing_depth", AMBIGUOUS: "ambiguous"}


# =========================================================================== gsplat backend
_BACKEND_STATUS: Optional[Dict[str, Any]] = None


def ensure_gsplat_backend() -> Dict[str, Any]:
    """Make the existing gsplat 0.1.11 CUDA extension importable; never compiles anything.

    gsplat JIT-loads ``gsplat_cuda`` only when ``nvcc`` is on PATH (the repo pipelines export
    ``CUDA_HOME=/usr/local/cuda-12.8``). When it is not, we load the *already built* extension from the
    torch extension cache (``~/.cache/torch_extensions/py*_cu*/gsplat_cuda/gsplat_cuda.so``) and bind it
    to ``gsplat.cuda._backend._C``. Same kernels, no rebuild, no write to the cache.
    Returns ``dict(available, source, reason)``."""
    global _BACKEND_STATUS
    if _BACKEND_STATUS is not None and _BACKEND_STATUS["available"]:
        return _BACKEND_STATUS
    status: Dict[str, Any] = dict(available=False, source=None, reason="")
    if not torch.cuda.is_available():
        status["reason"] = "CUDA device unavailable"
        _BACKEND_STATUS = status
        return status
    try:
        import gsplat  # noqa: F401
        import gsplat.cuda._backend as backend
    except Exception as exc:  # pragma: no cover - environment dependent
        status["reason"] = f"gsplat import failed: {exc!r}"
        _BACKEND_STATUS = status
        return status
    version = getattr(gsplat, "__version__", "?")
    if getattr(backend, "_C", None) is not None:
        status.update(available=True, source=f"gsplat {version} (compiled/JIT module)")
    else:
        from torch.utils.cpp_extension import _get_build_directory
        so = Path(_get_build_directory("gsplat_cuda", verbose=False)) / "gsplat_cuda.so"
        if not so.exists():
            status["reason"] = f"gsplat {version} has no CUDA backend: no nvcc and no prebuilt {so}"
        else:
            try:
                loader = importlib.machinery.ExtensionFileLoader("gsplat_cuda", str(so))
                spec = importlib.util.spec_from_file_location("gsplat_cuda", str(so), loader=loader)
                mod = importlib.util.module_from_spec(spec)
                loader.exec_module(mod)
                for name in ("project_gaussians_forward", "rasterize_forward", "project_gaussians_backward"):
                    getattr(mod, name)
                backend._C = mod
                status.update(available=True, source=f"gsplat {version} prebuilt torch-extension cache {so}")
            except Exception as exc:  # pragma: no cover
                status["reason"] = f"failed to load prebuilt gsplat extension {so}: {exc!r}"
    if status["available"] and not version.startswith("0.1."):
        status.update(available=False, reason=f"gsplat {version} != 0.1.x API used by splatfacto")
    _BACKEND_STATUS = status
    return status


def _gsplat():
    st = ensure_gsplat_backend()
    if not st["available"]:
        raise RuntimeError(f"gsplat rasterizer unavailable: {st['reason']}")
    from gsplat.project_gaussians import project_gaussians
    from gsplat.rasterize import rasterize_gaussians
    return project_gaussians, rasterize_gaussians


# =========================================================================== Gaussians
@dataclass
class GaussianSet:
    """Frozen-or-optimisable 3D Gaussians.

    means [N,3] world metres; quats [N,4] in the declared ``quat_layout`` (``"wxyz"`` gsplat-native, or
    ``"xyzw"``); log_scales [N,3] (natural log of std-dev, metres); opacities [N,1] **probabilities** in
    (0,1) (already sigmoided); colors [N,3] RGB in [0,1] (view-independent; SH degree 0)."""
    means: torch.Tensor
    quats: torch.Tensor
    log_scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    quat_layout: str = "wxyz"

    def __post_init__(self):
        if self.quat_layout not in QUAT_LAYOUTS:
            raise ValueError(f"quat_layout must be one of {QUAT_LAYOUTS}, got {self.quat_layout!r}")
        n = self.means.shape[0]
        if self.opacities.ndim == 1:
            self.opacities = self.opacities[:, None]
        for name, t, d in (("means", self.means, 3), ("quats", self.quats, 4), ("log_scales", self.log_scales, 3),
                           ("opacities", self.opacities, 1), ("colors", self.colors, 3)):
            if t.ndim != 2 or t.shape != (n, d):
                raise ValueError(f"GaussianSet.{name} must be [{n},{d}], got {tuple(t.shape)}")

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    def quats_wxyz(self) -> torch.Tensor:
        return self.quats if self.quat_layout == "wxyz" else self.quats[..., [3, 0, 1, 2]]

    def to(self, device=None, dtype=torch.float32) -> "GaussianSet":
        f = lambda t: t.to(device=device, dtype=dtype)  # noqa: E731
        return GaussianSet(f(self.means), f(self.quats), f(self.log_scales), f(self.opacities), f(self.colors),
                           self.quat_layout)

    def detach(self) -> "GaussianSet":
        return GaussianSet(self.means.detach(), self.quats.detach(), self.log_scales.detach(),
                           self.opacities.detach(), self.colors.detach(), self.quat_layout)

    def fingerprint(self) -> str:
        h = hashlib.sha256(self.quat_layout.encode())
        for t in (self.means, self.quats, self.log_scales, self.opacities, self.colors):
            h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()[:16]


def quats_for_gsplat(g: GaussianSet) -> torch.Tensor:
    """WXYZ unit quaternions for gsplat; mirrors ``SplatfactoModel.quats_for_gsplat``."""
    q = g.quats_wxyz()
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_multiply_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a (x) b`` for WXYZ quaternions (broadcasting); ``R(a (x) b) = R(a) R(b)``."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Differentiable rotation matrix -> unit WXYZ quaternion (Shepperd's branch selection)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    s = lambda x: 0.5 * torch.sqrt(torch.clamp(x, min=1e-12))  # noqa: E731
    w0 = s(1 + tr)
    x1 = s(1 + m00 - m11 - m22)
    y2 = s(1 - m00 + m11 - m22)
    z3 = s(1 - m00 - m11 + m22)
    c0 = torch.stack([w0, (m21 - m12) / (4 * w0), (m02 - m20) / (4 * w0), (m10 - m01) / (4 * w0)], -1)
    c1 = torch.stack([(m21 - m12) / (4 * x1), x1, (m01 + m10) / (4 * x1), (m02 + m20) / (4 * x1)], -1)
    c2 = torch.stack([(m02 - m20) / (4 * y2), (m01 + m10) / (4 * y2), y2, (m12 + m21) / (4 * y2)], -1)
    c3 = torch.stack([(m10 - m01) / (4 * z3), (m02 + m20) / (4 * z3), (m12 + m21) / (4 * z3), z3], -1)
    cands = torch.stack([c0, c1, c2, c3], -2)  # [...,4,4]
    choice = torch.argmax(torch.stack([tr, m00, m11, m22], -1).detach(), -1)
    onehot = torch.nn.functional.one_hot(choice, 4).to(R.dtype)
    q = (onehot[..., :, None] * cands).sum(-2)
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)  # w >= 0 (q and -q are the same rotation)
    return q / q.norm(dim=-1, keepdim=True)


def gaussian_covariances(g: GaussianSet) -> torch.Tensor:
    """World covariances ``R S^2 R^T`` [N,3,3] (uses gsplat's own quaternion->matrix map)."""
    from gsplat._torch_impl import quat_to_rotmat
    R = quat_to_rotmat(quats_for_gsplat(g))
    return R @ torch.diag_embed(torch.exp(g.log_scales) ** 2) @ R.transpose(-1, -2)


def transform_gaussians(g: GaussianSet, A) -> GaussianSet:
    """Apply a rigid world transform ``A`` [4,4] (torch or numpy) to means **and** orientations.

    means' = R means + t; quats' = q(R) (x) quats (WXYZ algebra, returned in ``g.quat_layout``), so
    covariance' = R Sigma R^T. Scales/opacities/colours unchanged. Differentiable w.r.t. ``A``."""
    A = torch.as_tensor(A, dtype=g.means.dtype, device=g.means.device) if not torch.is_tensor(A) else \
        A.to(dtype=g.means.dtype, device=g.means.device)
    R, t = A[:3, :3], A[:3, 3]
    means = g.means @ R.transpose(0, 1) + t
    q = quat_multiply_wxyz(rotmat_to_quat_wxyz(R)[None], g.quats_wxyz())
    if g.quat_layout == "xyzw":
        q = q[..., [1, 2, 3, 0]]
    return GaussianSet(means, q, g.log_scales, g.opacities, g.colors, g.quat_layout)


def concat_gaussians(parts: Sequence[GaussianSet]) -> GaussianSet:
    parts = [p if p.quat_layout == "wxyz" else GaussianSet(p.means, p.quats_wxyz(), p.log_scales, p.opacities,
                                                           p.colors, "wxyz") for p in parts]
    cat = lambda name: torch.cat([getattr(p, name) for p in parts], 0)  # noqa: E731
    return GaussianSet(cat("means"), cat("quats"), cat("log_scales"), cat("opacities"), cat("colors"), "wxyz")


def joint_transform(kind: str, axis=None, origin=None, q=0.0, like: Optional[torch.Tensor] = None):
    """``A_H(q)`` as a torch [4,4] (identity for ``kind == "static"``); keeps autograd to q/axis/origin."""
    dev = like.device if like is not None else None
    if kind == "static":
        return torch.eye(4, dtype=torch.float32, device=dev)
    q_t = q if torch.is_tensor(q) else torch.tensor(float(q), dtype=torch.float32, device=dev)
    ax = axis if torch.is_tensor(axis) else torch.as_tensor(np.asarray(axis, float), dtype=q_t.dtype, device=q_t.device)
    if kind == "revolute":
        org = origin if torch.is_tensor(origin) else torch.as_tensor(np.asarray(origin, float), dtype=q_t.dtype,
                                                                     device=q_t.device)
    else:
        org = ax * 0
    return A_joint(kind, ax, org, q_t)


# =========================================================================== rendering
@dataclass
class RenderOutput:
    rgb: torch.Tensor              # [H,W,3] composited over ``background``
    alpha: torch.Tensor            # [H,W]
    depth: torch.Tensor            # [H,W] metric camera-Z (0 where not depth_valid)
    depth_valid: torch.Tensor      # [H,W] bool, alpha >= depth_alpha_min
    depth_std: torch.Tensor        # [H,W] metres
    moving_fraction: torch.Tensor  # [H,W] in [0,1]: share of composited weight owned by moving Gaussians
    cov3d: Optional[torch.Tensor] = None  # [N,6] world covariances returned by the CUDA projection
    radii: Optional[torch.Tensor] = None
    depth_definition: str = DEPTH_DEFINITION


def _calib_arrays(calib):
    K = np.asarray(calib.K, float)
    T_WC = np.asarray(calib.T_WC, float)
    T_CW = np.linalg.inv(T_WC)
    return K, T_CW, int(calib.width), int(calib.height)


def render(g_static: Optional[GaussianSet], g_moving: Optional[GaussianSet], A_moving, calib, *,
           background=(0.0, 0.0, 0.0), depth_alpha_min: float = 0.5, pixel_center_offset: float = 0.5,
           device: Optional[torch.device] = None) -> RenderOutput:
    """Render static Gaussians as-is and moving Gaussians transformed by ``A_moving`` (``A_H(q)``, torch
    [4,4] with autograd, numpy, or None = identity) from the fixed physical camera ``calib``.

    ``calib`` needs ``K``, ``T_WC`` (OpenCV c2w), ``width``, ``height`` (``observations.CameraCalib``).
    The robot is never rasterized here; robot pixels are handled by ``pixel_provenance``."""
    project_gaussians, rasterize_gaussians = _gsplat()
    ref = g_moving if g_moving is not None else g_static
    if ref is None:
        raise ValueError("render needs at least one GaussianSet")
    device = device or (ref.means.device if ref.means.is_cuda else torch.device("cuda"))
    parts, flags = [], []
    if g_static is not None:
        parts.append(g_static.to(device))
        flags.append(torch.zeros(g_static.n, device=device))
    if g_moving is not None:
        gm = g_moving.to(device)
        if A_moving is not None:
            gm = transform_gaussians(gm, A_moving)
        parts.append(gm)
        flags.append(torch.ones(g_moving.n, device=device))
    g = concat_gaussians(parts)
    m_flag = torch.cat(flags, 0)

    K, T_CW, W, H = _calib_arrays(calib)
    viewmat = torch.as_tensor(T_CW, dtype=torch.float32, device=device)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]) + pixel_center_offset, float(K[1, 2]) + pixel_center_offset
    bg = torch.as_tensor(background, dtype=torch.float32, device=device)

    xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
        g.means.contiguous(), torch.exp(g.log_scales).contiguous(), 1, quats_for_gsplat(g).contiguous(),
        viewmat[:3, :], fx, fy, cx, cy, H, W, BLOCK_WIDTH)
    if int(radii.sum()) == 0:
        z = torch.zeros(H, W, device=device)
        return RenderOutput(bg.expand(H, W, 3).clone(), z, z.clone(), z.bool(), z.clone(), z.clone(), cov3d, radii)
    opac = g.opacities.contiguous()  # classic rasterize mode (splatfacto default)
    rgb, alpha = rasterize_gaussians(xys, depths, radii, conics, num_tiles_hit, g.colors.contiguous(), opac,
                                     H, W, BLOCK_WIDTH, background=bg, return_alpha=True)
    rgb = torch.clamp(rgb, max=1.0)
    aux_colors = torch.stack([depths, m_flag, depths * depths], -1).contiguous()
    aux = rasterize_gaussians(xys, depths, radii, conics, num_tiles_hit, aux_colors, opac, H, W, BLOCK_WIDTH,
                              background=torch.zeros(3, device=device))
    valid = alpha >= depth_alpha_min
    a_safe = torch.where(alpha > 1e-6, alpha, torch.ones_like(alpha))
    depth = torch.where(valid, aux[..., 0] / a_safe, torch.zeros_like(alpha))
    ez2 = torch.where(valid, aux[..., 2] / a_safe, torch.zeros_like(alpha))
    std = torch.sqrt(torch.clamp(ez2 - depth.detach() ** 2, min=0.0) + 1e-12)
    mov = torch.where(alpha > 1e-6, aux[..., 1] / a_safe, torch.zeros_like(alpha)).clamp(0, 1)
    return RenderOutput(rgb, alpha, depth, valid, std, mov, cov3d, radii)


def virtual_camera_calib(calib, A):
    """Calibration of the virtual part camera ``A^-1 T_WC`` (same K/size)."""
    A = np.asarray(A.detach().cpu() if torch.is_tensor(A) else A, float)
    T_v = np.linalg.inv(A) @ np.asarray(calib.T_WC, float)
    if dataclasses.is_dataclass(calib):
        return dataclasses.replace(calib, T_WC=T_v)
    return SimpleCalib(np.asarray(calib.K, float), T_v, int(calib.width), int(calib.height))


@dataclass
class SimpleCalib:
    K: np.ndarray
    T_WC: np.ndarray
    width: int
    height: int
    camera_id: str = "cam"


# =========================================================================== provenance
def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    H, W = mask.shape
    p = np.pad(mask, r, constant_values=False)
    out = np.zeros_like(mask)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= p[r + dy:r + dy + H, r + dx:r + dx + W]
    return out


def _local_range(depth: np.ndarray, valid: np.ndarray, r: int = 1) -> np.ndarray:
    H, W = depth.shape
    big = np.where(valid, depth, np.nan)
    p = np.pad(big, r, constant_values=np.nan)
    stack = np.stack([p[r + dy:r + dy + H, r + dx:r + dx + W] for dy in range(-r, r + 1) for dx in range(-r, r + 1)])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows (no valid depth) -> range 0
        rng = np.nanmax(stack, 0) - np.nanmin(stack, 0)
    return np.nan_to_num(rng, nan=0.0)


@dataclass
class ProvenanceConfig:
    min_depth_m: float = 0.05
    max_depth_m: float = 10.0
    robot_dilate_px: int = 2       # tool/robot edge band -> ambiguous
    mask_edge_px: int = 1          # object-mask boundary band -> ambiguous
    depth_edge_m: float = 0.05     # local 3x3 measured depth range above this -> ambiguous (mixed pixels)


def pixel_provenance(measured_depth, object_mask, robot_mask=None, cfg: ProvenanceConfig = ProvenanceConfig()):
    """Candidate-independent per-pixel provenance from measured data only (int8 [H,W]).

    Priority: robot -> OCCLUDED_BY_ROBOT; robot edge band / object-mask edge band / depth discontinuity
    -> AMBIGUOUS; object & invalid depth -> MISSING_DEPTH; object -> OBJECT_VISIBLE; else BACKGROUND.
    Measured depth <= 0, NaN/inf, or outside [min,max] is invalid (never -inf or background-filled)."""
    D = np.asarray(measured_depth, float)
    obj = np.asarray(object_mask, bool)
    rob = np.zeros_like(obj) if robot_mask is None else np.asarray(robot_mask, bool)
    valid_d = np.isfinite(D) & (D >= cfg.min_depth_m) & (D <= cfg.max_depth_m)
    prov = np.full(D.shape, BACKGROUND, np.int8)
    prov[obj & valid_d] = OBJECT_VISIBLE
    prov[obj & ~valid_d] = MISSING_DEPTH
    edge = _dilate(obj, cfg.mask_edge_px) & _dilate(~obj, cfg.mask_edge_px)
    disc = obj & valid_d & (_local_range(D, valid_d & obj, 1) > cfg.depth_edge_m)
    prov[edge | disc] = AMBIGUOUS
    prov[_dilate(rob, cfg.robot_dilate_px) & ~rob] = AMBIGUOUS
    prov[rob] = OCCLUDED_BY_ROBOT
    return prov


def loss_pixel_sets(prov: np.ndarray, eval_mask=None) -> Dict[str, np.ndarray]:
    """Candidate-independent pixel sets (bool [H,W]) used identically for every hypothesis.

    rgb: object_visible | missing_depth; depth: object_visible; alpha: object_visible | missing_depth |
    background. Robot and ambiguous pixels are never used. ``eval_mask`` restricts to held-out pixels."""
    ev = np.ones(prov.shape, bool) if eval_mask is None else np.asarray(eval_mask, bool)
    return dict(rgb=ev & np.isin(prov, [OBJECT_VISIBLE, MISSING_DEPTH]),
                depth=ev & (prov == OBJECT_VISIBLE),
                alpha=ev & np.isin(prov, [OBJECT_VISIBLE, MISSING_DEPTH, BACKGROUND]))


def depth_loss_weights(prov: np.ndarray, r: RenderOutput, measured_depth, *, part: str = "all",
                       ownership_min: float = 0.5, front_margin_m: float = 0.05, std_max_m: float = 0.05,
                       eval_mask=None) -> torch.Tensor:
    """Render-dependent z-buffer mask for **part-specific** depth losses (e.g. refining q of the moving part).

    A pixel contributes iff: provenance is OBJECT_VISIBLE; the render has a valid single front surface
    (alpha >= depth_alpha_min, depth_std <= std_max); for ``part`` in {moving, static} the composited front
    surface belongs to that part (ownership >= ownership_min, so a static occluder in front of the moving
    part removes the pixel from the moving part's loss); and the measured surface is not an unmodelled
    occluder in front of the render (``D >= depth - front_margin``). Robot pixels are excluded by provenance.
    Not used for candidate *selection* (which needs identical pixels for every hypothesis)."""
    dev = r.alpha.device
    D = torch.as_tensor(np.nan_to_num(np.asarray(measured_depth, float)), dtype=torch.float32, device=dev)
    base = torch.as_tensor(loss_pixel_sets(prov, eval_mask)["depth"], device=dev)
    w = base & r.depth_valid & (r.depth_std <= std_max_m)
    if part == "moving":
        w &= r.moving_fraction >= ownership_min
    elif part == "static":
        w &= (1 - r.moving_fraction) >= ownership_min
    elif part != "all":
        raise ValueError(part)
    w &= D >= (r.depth - front_margin_m)
    return w.detach()


# =========================================================================== losses
@dataclass
class LossConfig:
    rgb_delta: float = 0.05        # pseudo-Huber scale, intensity units
    depth_delta_m: float = 0.02    # pseudo-Huber scale, metres
    depth_cap_m: float = 0.25      # residuals beyond this are truncated (robust; constant penalty)
    sigma_rgb: float = 0.05        # normalisers for the combined score
    sigma_depth_m: float = 0.02
    sigma_alpha: float = 0.1
    w_rgb: float = 1.0
    w_depth: float = 1.0
    w_alpha: float = 1.0


def pseudo_huber(r: torch.Tensor, delta: float) -> torch.Tensor:
    """``delta (sqrt(1 + (r/delta)^2) - 1)``: ~r^2/(2 delta) near 0, ~|r| for large r (same units as r)."""
    return delta * (torch.sqrt(1 + (r / delta) ** 2) - 1)


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.to(x.dtype)
    n = m.sum()
    return (x * m).sum() / n.clamp_min(1.0)


def render_losses(r: RenderOutput, measured: Mapping[str, Any], sets: Mapping[str, Any],
                  cfg: LossConfig = LossConfig(), depth_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """L_RGB, L_depth (metres), L_alpha on fixed pixel sets (bool [H,W] numpy/torch).

    L_RGB  = mean pseudo-Huber of per-pixel RGB residual norm on ``sets['rgb']``.
    L_depth= mean pseudo-Huber(min(|depth - D|, cap)) on ``sets['depth']`` (or ``depth_weights``); a pixel whose
             render has no valid depth gets the capped penalty -- it is never dropped per hypothesis.
    L_alpha= mean |alpha - object_mask| on ``sets['alpha']``.
    ``total`` = sum_k w_k L_k / sigma_k."""
    dev = r.alpha.device
    T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev) if not torch.is_tensor(a) \
        else a.to(device=dev, dtype=dt)  # noqa: E731
    rgb_m = T(measured["rgb"])
    D = T(np.nan_to_num(np.asarray(measured["depth"], float)) if not torch.is_tensor(measured["depth"])
          else torch.nan_to_num(measured["depth"]))
    obj = T(measured["object_mask"])
    s_rgb, s_alpha = T(sets["rgb"], torch.bool), T(sets["alpha"], torch.bool)
    s_depth = T(sets["depth"], torch.bool) if depth_weights is None else T(depth_weights, torch.bool)
    res_rgb = torch.sqrt(((r.rgb - rgb_m) ** 2).sum(-1) + 1e-12)
    L_rgb = _masked_mean(pseudo_huber(res_rgb, cfg.rgb_delta), s_rgb)
    res_d = torch.where(r.depth_valid, (r.depth - D).abs(), torch.full_like(D, cfg.depth_cap_m))
    res_d = torch.clamp(res_d, max=cfg.depth_cap_m)
    L_depth = _masked_mean(pseudo_huber(res_d, cfg.depth_delta_m), s_depth)
    L_alpha = _masked_mean((r.alpha - obj).abs(), s_alpha)
    total = (cfg.w_rgb * L_rgb / cfg.sigma_rgb + cfg.w_depth * L_depth / cfg.sigma_depth_m
             + cfg.w_alpha * L_alpha / cfg.sigma_alpha)
    return dict(L_rgb=L_rgb, L_depth_m=L_depth, L_alpha=L_alpha, total=total,
                n_rgb=s_rgb.sum(), n_depth=s_depth.sum(), n_alpha=s_alpha.sum())


# =========================================================================== frozen candidate scoring
def _cand_get(c, *names, default=None):
    for n in names:
        if isinstance(c, Mapping) and n in c:
            return c[n]
        if not isinstance(c, Mapping) and hasattr(c, n):
            return getattr(c, n)
    return default


def _mask_hash(sets: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(sets):
        h.update(k.encode())
        h.update(np.packbits(np.asarray(sets[k], bool)).tobytes())
    return h.hexdigest()[:16]


def score_candidates_frozen(canonical_static: Optional[GaussianSet], canonical_moving: GaussianSet,
                            candidates: Sequence[Any], cameras: Mapping[str, Any], frames: Iterable[int],
                            measured: Mapping[str, Mapping[int, Mapping[str, Any]]], *,
                            loss_cfg: LossConfig = LossConfig(), prov_cfg: ProvenanceConfig = ProvenanceConfig(),
                            background=(0.0, 0.0, 0.0)) -> Dict[str, Any]:
    """Score static / prismatic / revolute hypotheses with ONE frozen canonical asset on IDENTICAL pixels.

    candidates: dicts/objects with ``name``, ``kind`` in {static, revolute, prismatic}, ``axis_w``/``axis``,
    ``origin_w``/``origin`` (revolute), ``q`` (sequence indexed by frame index t; radians or metres).
    cameras: ``{camera_id: CameraCalib}``. frames: time indices. measured: ``measured[cid][t]`` = dict(rgb
    [H,W,3] in [0,1], depth [H,W] metres (<=0/NaN invalid), object_mask [H,W] bool, optional robot_mask,
    optional eval_mask = held-out pixel subset). Pixel sets come from measured data only and are shared by
    every candidate (their hashes are reported). No Gaussian parameter is updated (``no_grad``; the asset
    fingerprint is verified before/after). Returns per-candidate, per-camera L_RGB/L_depth/L_alpha/total."""
    frames = [int(t) for t in frames]
    fp_before = (canonical_static.fingerprint() if canonical_static is not None else None,
                 canonical_moving.fingerprint())
    dev = torch.device("cuda")
    gs = canonical_static.detach().to(dev) if canonical_static is not None else None
    gm = canonical_moving.detach().to(dev)
    pixel_sets, sets_cache = {}, {}
    for cid in cameras:
        pixel_sets[cid] = {}
        for t in frames:
            m = measured[cid][t]
            prov = pixel_provenance(m["depth"], m["object_mask"], m.get("robot_mask"), prov_cfg)
            sets = loss_pixel_sets(prov, m.get("eval_mask"))
            sets_cache[(cid, t)] = sets
            counts = {PROVENANCE_NAMES[k]: int((prov == k).sum()) for k in PROVENANCE_NAMES}
            pixel_sets[cid][t] = dict(hash=_mask_hash(sets), n_rgb=int(sets["rgb"].sum()),
                                      n_depth=int(sets["depth"].sum()), n_alpha=int(sets["alpha"].sum()),
                                      provenance_counts=counts)
    results = []
    with torch.no_grad():
        for c in candidates:
            kind = _cand_get(c, "kind")
            name = _cand_get(c, "name", default=kind)
            if kind not in ("static", "revolute", "prismatic"):
                raise ValueError(f"unknown candidate kind {kind!r}")
            q_all = _cand_get(c, "q", default=None)
            per_cam = {}
            for cid, cal in cameras.items():
                acc = dict(L_rgb=0.0, L_depth_m=0.0, L_alpha=0.0, total=0.0, render_coverage=0.0)
                for t in frames:
                    if kind == "static":
                        A = None
                    else:
                        A = joint_transform(kind, _cand_get(c, "axis_w", "axis"), _cand_get(c, "origin_w", "origin"),
                                            float(np.asarray(q_all, float)[t]), like=gm.means)
                    r = render(gs, gm, A, cal, background=background, device=dev)
                    L = render_losses(r, measured[cid][t], sets_cache[(cid, t)], loss_cfg)
                    for k in ("L_rgb", "L_depth_m", "L_alpha", "total"):
                        acc[k] += float(L[k]) / len(frames)
                    dset = torch.as_tensor(sets_cache[(cid, t)]["depth"], device=dev)
                    acc["render_coverage"] += float((r.depth_valid & dset).sum() / dset.sum().clamp_min(1)) / len(frames)
                per_cam[cid] = acc
            total = float(np.mean([v["total"] for v in per_cam.values()]))
            results.append(dict(name=name, kind=kind, per_camera=per_cam, total=total))
    fp_after = (gs.fingerprint() if gs is not None else None, gm.fingerprint())
    if fp_after != fp_before:
        raise RuntimeError("canonical asset changed during frozen scoring")
    ranking = [r["name"] for r in sorted(results, key=lambda r: r["total"])]
    statics = [r["total"] for r in results if r["kind"] == "static"]
    best = next(r for r in results if r["name"] == ranking[0])
    return dict(renderer_evaluation_kind=RENDERER_EVALUATION_KIND, renderer_backend=RENDERER_BACKEND,
                backend_status=ensure_gsplat_backend(), depth_definition=DEPTH_DEFINITION,
                quat_layout=canonical_moving.quat_layout,
                canonical_asset_fingerprint=dict(static=fp_before[0], moving=fp_before[1]),
                identical_pixels_across_candidates=True, pixel_sets=pixel_sets, frames=frames,
                candidates=results, ranking=ranking, best=ranking[0],
                best_minus_static=(best["total"] - min(statics)) if statics else None,
                loss_config=dataclasses.asdict(loss_cfg))


# =========================================================================== canonical asset loading
DEFAULT_SEARCH_ROOTS = (
    "/home/samuelemara/ForceArtGS/outputs",
    "/home/samuelemara/ForceArtGS/checkpoints",
    "/media/samuelemara/Extreme SSD/Samuele",
)


def find_canonical_checkpoints(scene_name: str, search_roots: Sequence[str] = DEFAULT_SEARCH_ROOTS) -> List[Path]:
    """All nerfstudio ``step-*.ckpt`` files whose path contains ``scene_name``."""
    found = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            if scene_name not in dirpath:
                continue
            found += [Path(dirpath) / f for f in files if re.fullmatch(r"step-\d+\.ckpt", f)]
    return sorted(found)


def _read_config_fields(ckpt: Path) -> Dict[str, Any]:
    run = ckpt.parent.parent
    cfg = run / "config.yml"
    out: Dict[str, Any] = dict(config_path=None, dataparser_transforms=None)
    dp = run / "dataparser_transforms.json"
    if dp.exists():
        import json
        d = json.loads(dp.read_text())
        out["dataparser_transforms"] = dict(path=str(dp), transform=np.asarray(d["transform"], float),
                                            scale=float(d["scale"]))
    if not cfg.exists():
        return out
    text = cfg.read_text(errors="replace")
    out["config_path"] = str(cfg)
    m = re.search(r"^\s*quaternion_order:\s*(\S+)", text, re.M)
    out["quaternion_order"] = m.group(1).strip("'\"") if m else "legacy"  # field absent => predates it => legacy
    m = re.search(r"^\s*sh_degree:\s*(\d+)", text, re.M)
    out["sh_degree"] = int(m.group(1)) if m else None
    m = re.search(r"^\s*motion_part:\s*(\d+)", text, re.M)
    out["motion_part"] = int(m.group(1)) if m else None
    m = re.search(r"^\s*rasterize_mode:\s*(\S+)", text, re.M)
    out["rasterize_mode"] = m.group(1) if m else None
    out["is_art"] = "SplatfactoArt" in text
    return out


def _torch_load_weights_only(path: Path):
    """``torch.load(weights_only=True)`` allowing only numpy scalar/dtype globals (nerfstudio stores numpy
    scalars in optimizer/scheduler state). Never falls back to arbitrary unpickling."""
    safe = [np.dtype]
    try:
        from numpy.core.multiarray import scalar
        safe.append(scalar)
    except Exception:  # pragma: no cover
        pass
    safe += [getattr(np.dtypes, n) for n in dir(getattr(np, "dtypes", object)) if n.endswith("DType")]
    ctx = torch.serialization.safe_globals(safe) if hasattr(torch.serialization, "safe_globals") else None
    if ctx is None:  # pragma: no cover
        return torch.load(path, map_location="cpu", weights_only=True)
    with ctx:
        return torch.load(path, map_location="cpu", weights_only=True)


def _nerfstudio_exp_map_SO3xR3(tangent: torch.Tensor) -> torch.Tensor:
    """Exact copy of ``nerfstudio.cameras.lie_groups.exp_map_SO3xR3`` for one 6-vector -> [4,4]
    (translation first, then so(3); same 1e-4 clamp as the trainer used)."""
    log_rot = tangent[3:]
    nrms = (log_rot * log_rot).sum()
    ang = torch.clamp(nrms, 1e-4).sqrt()
    fac1, fac2 = ang.sin() / ang, (1.0 - ang.cos()) / (ang * ang)
    K = torch.zeros(3, 3, dtype=tangent.dtype)
    K[0, 1], K[0, 2], K[1, 0] = -log_rot[2], log_rot[1], log_rot[2]
    K[1, 2], K[2, 0], K[2, 1] = -log_rot[0], -log_rot[1], log_rot[0]
    T = torch.eye(4, dtype=tangent.dtype)
    T[:3, :3] = fac1 * K + fac2 * (K @ K) + torch.eye(3, dtype=tangent.dtype)
    T[:3, 3] = tangent[:3]
    return T


def _ns_to_world(g: GaussianSet, transform_3x4: np.ndarray, scale: float) -> GaussianSet:
    """Invert the dataparser similarity ``x_ns = s (R_n x_w + t_n)`` (means, log-scales and orientations)."""
    Rn = torch.as_tensor(transform_3x4[:3, :3], dtype=torch.float32)
    tn = torch.as_tensor(transform_3x4[:3, 3], dtype=torch.float32)
    unscaled = GaussianSet(g.means / scale, g.quats, g.log_scales - math.log(scale), g.opacities, g.colors,
                           g.quat_layout)
    A = torch.eye(4)
    A[:3, :3] = Rn.T
    A[:3, 3] = -Rn.T @ tn
    return transform_gaussians(unscaled, A)


def load_canonical_asset(scene_name: Optional[str] = None, *, checkpoint: Optional[str] = None,
                         search_roots: Sequence[str] = DEFAULT_SEARCH_ROOTS, motion_part: Optional[int] = None,
                         quaternion_layout: Optional[str] = None, sh_degree: Optional[int] = None,
                         allow_normalized_frame: bool = False) -> Dict[str, Any]:
    """Load frozen canonical Gaussians from an existing nerfstudio checkpoint; FAIL CLOSED otherwise.

    Returns dict(status in {OK, MISSING_CANONICAL_ASSET, INCOMPATIBLE_CANONICAL_ASSET}, reason, checkpoint,
    static, moving, quat_layout, quat_layout_source, frame, warnings). ``static``/``moving`` are ``GaussianSet``.

    * Keys: ``*gauss_params.{means,quats,scales,opacities,features_dc}`` (``features_rest`` ignored: colours
      are the frozen view-independent DC term, ``SH2RGB`` if sh_degree>0 else sigmoid, as ``SplatfactoModel.colors``).
      Loaded with ``torch.load(weights_only=True)`` (numpy scalar/dtype globals allow-listed only).
    * Quaternion layout: what gsplat actually consumed. ``quaternion_order: xyzw`` in config.yml => stored XYZW;
      ``legacy`` (or field absent) => passed raw to gsplat => WXYZ. Unknown config => INCOMPATIBLE unless
      ``quaternion_layout`` is given explicitly.
    * Frame: nerfstudio trains in the dataparser-normalised frame ``x_ns = s (R_n x_w + t_n)``; the inverse is
      applied from ``dataparser_transforms.json`` so Gaussians are returned in the dataset world frame (the
      frame of the transforms.json camera poses; metres only if that dataset was metric). Missing file =>
      INCOMPATIBLE unless ``allow_normalized_frame``.
    * splatfacto_art (``part_weights`` present): static = opacity*(1-w), moving = opacity*w over all Gaussians
      (as ``SplatfactoArtModel.get_outputs`` composes them); the learned ``rotation_offset`` (base @ exp(delta))
      is applied to the moving copy so that ``A_H(0)`` is the identity on it. ``motion_part`` from argument or
      config.yml. Such an asset was trained jointly with ONE joint hypothesis -> warning (not hypothesis-neutral).
    * Single splatfacto model: returned as ``static`` only with ``moving=None`` and status INCOMPATIBLE (a
      part split is required and is not invented here)."""
    out: Dict[str, Any] = dict(status=STATUS_MISSING, reason="", checkpoint=None, static=None, moving=None,
                               quat_layout=None, quat_layout_source=None, frame=None, warnings=[],
                               searched=list(search_roots), renderer_evaluation_kind=RENDERER_EVALUATION_KIND)
    if checkpoint is None:
        if scene_name is None:
            out["reason"] = "no scene_name or checkpoint given"
            return out
        cks = find_canonical_checkpoints(scene_name, search_roots)
        if not cks:
            out["reason"] = f"no nerfstudio step-*.ckpt under {list(search_roots)} for scene {scene_name!r}"
            return out
        if len(cks) > 1:
            out["warnings"].append(f"{len(cks)} checkpoints found; using latest step: {[str(c) for c in cks]}")
        checkpoint = str(max(cks, key=lambda p: int(re.findall(r"\d+", p.name)[0])))
    ck = Path(checkpoint)
    out["checkpoint"] = str(ck)
    if not ck.exists():
        out["reason"] = f"checkpoint {ck} does not exist"
        return out
    out["status"] = STATUS_INCOMPATIBLE
    meta = _read_config_fields(ck)
    out["config"] = {k: v for k, v in meta.items() if k != "dataparser_transforms"}
    try:
        state = _torch_load_weights_only(ck)
    except Exception as exc:
        out["reason"] = f"torch.load(weights_only=True) failed: {exc!r}"
        return out
    sd = state.get("pipeline", state)

    def find(key):
        hits = [k for k in sd if k.endswith("gauss_params." + key) or k == key or k.endswith("_model." + key)]
        return sd[hits[0]] if hits else None
    params = {k: find(k) for k in ("means", "quats", "scales", "opacities", "features_dc", "part_weights",
                                   "rotation_offset_base", "rotation_offset_delta")}
    missing = [k for k in ("means", "quats", "scales", "opacities", "features_dc") if params[k] is None]
    if missing:
        out["reason"] = f"checkpoint lacks Gaussian params {missing}"
        return out
    if quaternion_layout is not None:
        layout, src = quaternion_layout, "explicit_argument"
    elif meta.get("config_path"):
        layout = "xyzw" if meta["quaternion_order"] == "xyzw" else "wxyz"
        src = f"config.yml quaternion_order={meta['quaternion_order']}"
    else:
        out["reason"] = "config.yml missing: quaternion layout consumed by gsplat is unknown"
        return out
    sh = sh_degree if sh_degree is not None else meta.get("sh_degree", 3 if meta.get("config_path") else None)
    if sh is None:
        out["reason"] = "sh_degree unknown"
        return out
    if meta.get("rasterize_mode") not in (None, "classic"):
        out["warnings"].append(f"asset trained with rasterize_mode={meta['rasterize_mode']}; rendered here as classic")
    fdc = params["features_dc"].float()
    colors = (fdc * 0.28209479177387814 + 0.5).clamp(0, 1) if sh > 0 else torch.sigmoid(fdc)
    base = dict(means=params["means"].float(), quats=params["quats"].float(), log_scales=params["scales"].float(),
                colors=colors)
    opac = torch.sigmoid(params["opacities"].float()).reshape(-1, 1)
    out.update(quat_layout=layout, quat_layout_source=src)
    dpt = meta.get("dataparser_transforms")
    if dpt is None and not allow_normalized_frame:
        out["reason"] = "dataparser_transforms.json missing: cannot map nerfstudio-normalised Gaussians to world"
        return out
    to_world = (lambda g: _ns_to_world(g, dpt["transform"], dpt["scale"])) if dpt is not None else (lambda g: g)
    out["frame"] = ("dataset_world (dataparser similarity inverted)" if dpt is not None
                    else "nerfstudio_normalized (NOT world)")
    if params["part_weights"] is None:
        out["static"] = to_world(GaussianSet(opacities=opac, quat_layout=layout, **base))
        out["reason"] = "single-part splatfacto checkpoint: no static/moving split available"
        return out
    mp = motion_part if motion_part is not None else meta.get("motion_part")
    if mp not in (0, 1):
        out["reason"] = "splatfacto_art checkpoint needs motion_part (0/1); not given and not in config.yml"
        return out
    out["motion_part"] = mp
    w = torch.sigmoid(params["part_weights"].float()).reshape(-1, 1)
    if mp == 0:
        w = 1 - w
    g_static = GaussianSet(opacities=opac * (1 - w), quat_layout=layout, **base)
    g_moving = GaussianSet(opacities=opac * w, quat_layout=layout, **base)
    if params["rotation_offset_base"] is not None:
        T_off = params["rotation_offset_base"].float()
        if params["rotation_offset_delta"] is not None:
            T_off = T_off @ _nerfstudio_exp_map_SO3xR3(params["rotation_offset_delta"].float())
        g_moving = transform_gaussians(g_moving, T_off)
        out["rotation_offset_applied"] = T_off.numpy().tolist()
    out["static"], out["moving"] = to_world(g_static), to_world(g_moving)
    out["warnings"].append("splatfacto_art asset was optimised jointly with one joint hypothesis; it is not "
                           "hypothesis-neutral for P/R comparison")
    if layout == "wxyz" and meta.get("is_art"):
        out["warnings"].append("legacy splatfacto_art composed rotations assuming XYZW while gsplat read WXYZ; the "
                               "trained moving-part orientations were rendered inconsistently during training")
    out["status"] = STATUS_OK
    out["reason"] = "loaded"
    return out
