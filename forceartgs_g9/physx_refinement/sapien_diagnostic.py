"""Isolated SAPIEN/PhysX CPU reference runner for G10, never a fake native friction adapter.

The public SAPIEN 3.0.3 Python joint API exposes a *legacy* dimensionless
joint friction coefficient and drive damping, not the newer PhysX per-axis
(staticEffort, dynamicEffort, viscous) trio. Therefore:
  * 'frictionless_physx': real PhysX but requires zero desired friction;
  * 'external_effort_surrogate': real PhysX integrating an explicit joint-
    effort approximation of viscous + dynamic friction, labelled NOT NATIVE.
There is no static-friction surrogate: Fs > Td explicitly fails closed.
Native friction optimization needs a version-compatible Isaac/PhysX adapter.

The simulation itself lives in ``runtime_scripts/sapien_physx_worker.py`` (sapien + numpy only). It runs in this
interpreter when ``sapien`` is importable, otherwise in a persistent worker process of the engine interpreter
(``PARTTWIN1D_PYTHON`` or ``~/miniconda3/envs/parttwin_sim/bin/python``; the freeartgs env has no SAPIEN).
Engine defaults that would silently add dissipation (legacy joint friction 0.05, link linear/angular damping 0.05,
articulation sleeping, drives) are zeroed and READ BACK every rollout; a nonzero readback fails closed.
"""
from __future__ import annotations

import atexit
import importlib.util
import json
import math
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .objective import Episode, PhysicalParameters, SimulationResult

WORKER = Path(__file__).resolve().parents[1] / "runtime_scripts" / "sapien_physx_worker.py"
MODES = {"frictionless_physx": "frictionless", "external_effort_surrogate": "external_effort_surrogate"}


def default_engine_python():
    p = os.environ.get("PARTTWIN1D_PYTHON", str(Path.home() / "miniconda3/envs/parttwin_sim/bin/python"))
    return p if Path(p).is_file() else None


def scale_urdf_mass_inertia(source, target, moving_link, scale):
    """Scale m and I_COM together; keep COM fixed, never scale collision geometry."""
    source, target = Path(source), Path(target)
    root = ET.parse(source)
    el = root.find(f".//link[@name='{moving_link}']/inertial")
    if el is None:
        raise ValueError(f"{moving_link}: missing URDF <inertial>")
    mass = el.find("mass")
    inertia = el.find("inertia")
    if mass is None or inertia is None:
        raise ValueError("moving link must have both mass and 6-component inertia")
    mass.set("value", f"{float(mass.get('value'))*scale:.12g}")
    for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
        if k not in inertia.attrib:
            raise ValueError(f"missing URDF inertia coefficient {k}")
        inertia.set(k, f"{float(inertia.get(k))*scale:.12g}")
    for mesh in root.findall(".//mesh"):
        path = mesh.get("filename")
        if path and not Path(path).is_absolute():
            if path.startswith("package://"):
                raise ValueError("package:// URDF mesh URI must be resolved by the caller")
            resolved = (source.parent/path).resolve()
            if not resolved.exists():
                raise FileNotFoundError(resolved)
            mesh.set("filename", str(resolved))
    # SAPIEN's URDF dynamics tags do not encode the new PhysX native friction law.
    # Disable the imported passive terms. We configure the solver explicitly below.
    for j in root.findall(".//joint"):
        d = j.find("dynamics")
        if d is not None:
            d.set("damping", "0")
            d.set("friction", "0")
    root.write(target, encoding="utf-8", xml_declaration=True)


def _rpy(s):
    r, p, y = (float(x) for x in (s or "0 0 0").split())
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _xyz(s):
    return np.array([float(x) for x in (s or "0 0 0").split()])


def urdf_axis_inertia(urdf, moving_link, joint_name) -> dict:
    """Spec G10 hinge inertia at rho_s = 1 from the URDF itself, in the child-link (= joint at q = 0) frame:
    I_h = a^T I_COM a + m ||(I - a a^T)(c - o)||^2  (revolute; o = joint origin = link frame origin),  M = m (prismatic).
    Returns dict(joint_type, mass, com_link, inertia_com_link, axis_link, axis_inertia)."""
    root = ET.parse(urdf).getroot()
    js = [j for j in root.findall("joint") if j.get("name") == joint_name]
    if len(js) != 1:
        raise ValueError(f"joint {joint_name!r} not found exactly once")
    j = js[0]
    if j.find("child").get("link") != moving_link:
        raise ValueError(f"joint {joint_name!r} child is not {moving_link!r}")
    jt = j.get("type")
    jt = "revolute" if jt in ("revolute", "continuous") else jt
    if jt not in ("revolute", "prismatic"):
        raise ValueError(f"unsupported joint type {jt!r}")
    ax = j.find("axis")
    a = _xyz(ax.get("xyz") if ax is not None else "1 0 0")
    a = a / np.linalg.norm(a)
    el = root.find(f".//link[@name='{moving_link}']/inertial")
    if el is None:
        raise ValueError(f"{moving_link}: missing URDF <inertial>")
    o = el.find("origin")
    com = _xyz(o.get("xyz") if o is not None else None)
    R = _rpy(o.get("rpy") if o is not None else None)
    m = float(el.find("mass").get("value"))
    i = {k: float(el.find("inertia").get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    J = np.array([[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]])
    J = R @ J @ R.T
    rp = com - (com @ a) * a
    Ih = float(a @ J @ a + m * rp @ rp) if jt == "revolute" else m
    return dict(joint_type=jt, mass=m, com_link=com.tolist(), inertia_com_link=J.tolist(), axis_link=a.tolist(),
                axis_inertia=Ih)


def readback_axis_inertia(rb, axis_link, joint_type) -> float:
    """The same I_h from what the ENGINE uses (mass, principal inertia, COM-frame pose read back)."""
    m = float(rb["moving_mass"])
    if joint_type == "prismatic":
        return m
    w, x, y, z = rb["moving_cmass_local_q_wxyz"]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    J = R @ np.diag(rb["moving_principal_inertia"]) @ R.T
    a = np.asarray(axis_link, float)
    c = np.asarray(rb["moving_cmass_local_p"], float)
    rp = c - (c @ a) * a
    return float(a @ J @ a + m * rp @ rp)


class _Worker:
    """Persistent JSON-lines worker in the engine interpreter (one clean scene per request)."""

    def __init__(self, python, timeout=600):
        self.python, self.timeout = str(python), timeout
        self._err = tempfile.TemporaryFile()
        self.p = subprocess.Popen([self.python, "-u", str(WORKER), "--serve"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self._err, text=True, bufsize=1)
        atexit.register(self.close)

    def __call__(self, req):
        if self.p.poll() is not None:
            raise RuntimeError(f"SAPIEN worker exited ({self.p.returncode}): {self._stderr()}")
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"SAPIEN worker closed its output: {self._stderr()}")
            if line.startswith("@@G10@@ "):
                out = json.loads(line[8:])
                if not out["ok"]:
                    raise RuntimeError(f"SAPIEN worker: {out['error']}")
                return out["result"]

    def _stderr(self):
        self._err.seek(0)
        return self._err.read().decode(errors="replace")[-2000:]

    def close(self):
        if getattr(self, "p", None) is not None and self.p.poll() is None:
            try:
                self.p.stdin.close()
                self.p.wait(timeout=10)
            except Exception:
                self.p.kill()


def _in_process():
    spec = importlib.util.spec_from_file_location("_g10_sapien_physx_worker", WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_request


class IsolatedSapienCpu:
    """Callable compatible with RefinementLoss. Instantiates clean PhysX per candidate.

    `friction_mode` must be explicit; no implicit fallback to the legacy
    dimensionless coefficient. Use the exact version used in policy training.
    This runner is a diagnostic if `external_effort_surrogate` is selected.
    `python`: engine interpreter for the worker (None: this interpreter if it has sapien, else the default engine env).
    """

    def __init__(self, urdf, *, moving_link, joint_name, dt=1/1000,
                 gravity=(0., 0., -9.81), friction_mode="frictionless_physx", python=None,
                 readback_rtol=1e-4):
        self.urdf = Path(urdf).resolve()
        self.moving_link = moving_link
        self.joint_name = joint_name
        self.dt = float(dt)
        self.gravity = np.asarray(gravity, dtype=float)
        self.friction_mode = friction_mode
        self.readback_rtol = float(readback_rtol)
        if friction_mode not in MODES:
            raise ValueError("SAPIEN 3 does not expose the native PhysX effort-friction API; no silent fallback")
        if not self.urdf.is_file() or self.dt <= 0:
            raise ValueError("missing URDF or invalid timestep")
        self.axis = urdf_axis_inertia(self.urdf, moving_link, joint_name)
        if python is None and importlib.util.find_spec("sapien") is not None:
            self._run, self.backend_process = _in_process(), "in_process"
        else:
            py = python or default_engine_python()
            if py is None:
                raise RuntimeError("no SAPIEN interpreter: install sapien here or set PARTTWIN1D_PYTHON")
            self._run, self.backend_process = _Worker(py), f"worker:{py}"
        self._tmp = tempfile.TemporaryDirectory(prefix="forceartgs_physx_estimator_")
        self._urdfs = {}
        self.n_rollouts = 0

    def effective_inertia(self, theta: PhysicalParameters) -> float:
        """I_h(rho_s, A) = rho_s I_h(1) + A (revolute) / rho_s m_0 + A (prismatic)."""
        return float(theta.rho_scale * self.axis["axis_inertia"] + theta.armature)

    def _scaled(self, rho):
        key = f"{float(rho):.12g}"
        if key not in self._urdfs:
            p = Path(self._tmp.name) / f"candidate_{len(self._urdfs):06d}.urdf"
            scale_urdf_mass_inertia(self.urdf, p, self.moving_link, float(rho))
            self._urdfs[key] = p
            if len(self._urdfs) > 4096:                          # bounded temp usage during long optimisations
                old = next(iter(self._urdfs))
                Path(self._urdfs.pop(old)).unlink(missing_ok=True)
        return self._urdfs[key]

    def request(self, theta: PhysicalParameters, ep: Episode) -> dict:
        if theta.static_effort > theta.dynamic_effort + 1e-12:
            raise ValueError("SAPIEN surrogate cannot reproduce static breakaway: use native PhysX axis friction")
        if self.friction_mode == "frictionless_physx" and (theta.dynamic_effort or theta.viscous):
            raise ValueError("frictionless_physx cannot identify nonzero friction")
        forced = ep.mode == "forced"
        return dict(urdf=str(self._scaled(theta.rho_scale)), moving_link=self.moving_link, joint_name=self.joint_name,
                    timestep=self.dt, gravity=self.gravity.tolist(), friction_mode=MODES[self.friction_mode],
                    viscous=float(theta.viscous), dynamic_effort=float(theta.dynamic_effort),
                    armature=float(theta.armature), effective_inertia=self.effective_inertia(theta),
                    q0=float(ep.q0), v0=float(ep.v0), t0=float(ep.t_s[0]), t_query=ep.t_s.tolist(),
                    effort_t_s=(ep.input_t_s.tolist() if forced else [float(ep.t_s[0])]),
                    effort_levels=(ep.input_effort.tolist() if forced else [0.0]))

    def __call__(self, theta: PhysicalParameters, ep: Episode) -> SimulationResult:
        out = self._run(self.request(theta, ep))
        self.n_rollouts += 1
        rb = out["readback"]
        I_engine = readback_axis_inertia(rb, self.axis["axis_link"], self.axis["joint_type"])
        I_expect = theta.rho_scale * self.axis["axis_inertia"]
        if abs(I_engine - I_expect) > self.readback_rtol * max(abs(I_expect), 1e-12) + 1e-9:
            raise RuntimeError(f"engine axis inertia {I_engine:.9g} != scaled URDF {I_expect:.9g}")
        if abs(float(rb["armature"][0]) - theta.armature) > 1e-6 * max(1.0, theta.armature):
            raise RuntimeError("armature readback mismatch")
        if int(out.get("n_contacts", 0)) > 0:
            raise RuntimeError(f"{out['n_contacts']} unintended contact(s) in a passive single-joint rollout")
        lim = np.asarray(rb.get("qlimits") or [[-np.inf, np.inf]], float).reshape(-1, 2)[0]
        qq = np.asarray(out["q"], float)
        if np.isfinite(lim).all() and (qq.min() <= lim[0] + 1e-6 or qq.max() >= lim[1] - 1e-6):
            raise RuntimeError("rollout reached a joint limit (limit impulse in an assumed sliding window)")
        return SimulationResult(np.asarray(out["q"]), np.asarray(out["qdot"]), backend="sapien_physx_cpu",
                                engine_version=out.get("sapien_version"), runtime_semantics=out["runtime_semantics"],
                                diagnostics=dict(axis_inertia_engine=I_engine, axis_inertia_expected=I_expect,
                                                 external_impulse=out["external_impulse"],
                                                 friction_impulse=out["friction_impulse"], v_eps=out["v_eps"],
                                                 engine_defaults_as_loaded=out["engine_defaults_as_loaded"],
                                                 readback=rb, process=self.backend_process,
                                                 n_contacts=out.get("n_contacts"),
                                                 sleeping_at_end=out.get("moving_link_sleeping_at_end")))

    def capabilities(self) -> dict:
        """Engine API coverage of the installed runtime (native axis friction exposed or not)."""
        return self._run(dict(kind="capabilities"))

    def close(self):
        if isinstance(self._run, _Worker):
            self._run.close()
        self._tmp.cleanup()


class IsolatedSapienGpu(IsolatedSapienCpu):
    """The same isolated one-joint replay on PhysX **GPU** (sapien.physx.PhysxGpuSystem, CUDA), BATCHED: ``batch``
    rolls out many candidates of one episode in parallel sub-scenes (one GPU call per DE generation / Jacobian /
    profile slice). Same step law, friction surrogate, passive-term zeroing and per-candidate readback checks as the
    CPU runner; parity against it is tested. ``__call__`` = a batch of one (slow: the GPU pays per step, not per env).
    Always runs in the engine interpreter's worker process (CUDA context isolated from the estimator)."""

    def __init__(self, urdf, *, moving_link, joint_name, dt=1 / 1000, gravity=(0., 0., -9.81),
                 friction_mode="external_effort_surrogate", python=None, readback_rtol=1e-4, device="cuda"):
        py = python or default_engine_python()
        if py is None:
            raise RuntimeError("no SAPIEN interpreter: set PARTTWIN1D_PYTHON")
        super().__init__(urdf, moving_link=moving_link, joint_name=joint_name, dt=dt, gravity=gravity,
                         friction_mode=friction_mode, python=py, readback_rtol=readback_rtol)
        self.device = device
        self.n_batches = 0

    def _check(self, theta, rb, q):
        I_engine = readback_axis_inertia(rb, self.axis["axis_link"], self.axis["joint_type"])
        I_expect = theta.rho_scale * self.axis["axis_inertia"]
        if abs(I_engine - I_expect) > self.readback_rtol * max(abs(I_expect), 1e-12) + 1e-9:
            raise RuntimeError(f"engine axis inertia {I_engine:.9g} != scaled URDF {I_expect:.9g}")
        if abs(float(rb["armature"][0]) - theta.armature) > 1e-6 * max(1.0, theta.armature):
            raise RuntimeError("armature readback mismatch")
        lim = np.asarray(rb.get("qlimits") or [[-np.inf, np.inf]], float).reshape(-1, 2)[0]
        if np.isfinite(lim).all() and (q.min() <= lim[0] + 1e-6 or q.max() >= lim[1] - 1e-6):
            raise RuntimeError("rollout reached a joint limit (limit impulse in an assumed sliding window)")
        return I_engine, I_expect

    def batch(self, thetas, ep: Episode) -> list:
        """Rollouts of every theta on ONE episode; a candidate that fails its checks returns its exception."""
        thetas = list(thetas)
        for th in thetas:
            if th.static_effort > th.dynamic_effort + 1e-12:
                raise ValueError("SAPIEN surrogate cannot reproduce static breakaway: use native PhysX axis friction")
            if self.friction_mode == "frictionless_physx" and (th.dynamic_effort or th.viscous):
                raise ValueError("frictionless_physx cannot identify nonzero friction")
        forced = ep.mode == "forced"
        req = dict(kind="batch_gpu", device=self.device, moving_link=self.moving_link, joint_name=self.joint_name,
                   timestep=self.dt, gravity=self.gravity.tolist(), friction_mode=MODES[self.friction_mode],
                   candidates=[dict(urdf=str(self._scaled(th.rho_scale)), viscous=float(th.viscous),
                                    dynamic_effort=float(th.dynamic_effort), armature=float(th.armature),
                                    effective_inertia=self.effective_inertia(th)) for th in thetas],
                   q0=float(ep.q0), v0=float(ep.v0), t0=float(ep.t_s[0]), t_query=ep.t_s.tolist(),
                   effort_t_s=(ep.input_t_s.tolist() if forced else [float(ep.t_s[0])]),
                   effort_levels=(ep.input_effort.tolist() if forced else [0.0]))
        out = self._run(req)
        self.n_batches += 1
        self.n_rollouts += len(thetas)
        res = []
        for i, th in enumerate(thetas):
            rb = out["readback"][i]
            q = np.asarray(out["q"][i], float)
            try:
                if not np.isfinite(q).all():
                    raise RuntimeError("non-finite GPU rollout")
                I_engine, I_expect = self._check(th, rb, q)
            except RuntimeError as e:
                res.append(e)
                continue
            res.append(SimulationResult(q, np.asarray(out["qdot"][i], float), backend="sapien_physx_gpu",
                                        engine_version=out.get("sapien_version"),
                                        runtime_semantics=out["runtime_semantics"],
                                        diagnostics=dict(axis_inertia_engine=I_engine, axis_inertia_expected=I_expect,
                                                         external_impulse=out["external_impulse"][i],
                                                         friction_impulse=out["friction_impulse"][i],
                                                         v_eps=out["v_eps"][i],
                                                         engine_defaults_as_loaded=out["engine_defaults_as_loaded"][i],
                                                         readback=rb, process=self.backend_process,
                                                         device=out.get("device"), gpu_backend=out.get("backend"),
                                                         n_contacts=0, contact_note=out.get("contact_note"),
                                                         sleeping_at_end=None)))
        return res

    def __call__(self, theta: PhysicalParameters, ep: Episode) -> SimulationResult:
        r = self.batch([theta], ep)[0]
        if isinstance(r, Exception):
            raise r
        return r
