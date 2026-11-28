"""URDF / USD bridge writes ONLY the estimated joint damping and the moving inertia along the joint."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from forceartgs_g9.export import estimated_asset as EA


def test_revolute_inertia_is_exactly_M_about_the_axis_with_com_on_the_axis(tmp_path):
    a = np.array([0.1, -0.2, 0.97])
    m = EA.joint_dynamics_model(dict(type="revolute", axis_Wc=a.tolist(), origin_Wc=[0.3, 0.1, 0.2]), 0.39, 0.036)
    u = a / np.linalg.norm(a)
    assert np.isclose(u @ m["inertia"] @ u, 0.39) and np.allclose(m["com"], 0)
    ev = np.linalg.eigvalsh(m["inertia"])
    assert ev.min() > 0 and ev[2] <= ev[0] + ev[1] + 1e-12                    # physical tensor
    assert m["damping"] == 0.036 and m["mass"] == EA.PLACEHOLDER
    EA.write_urdf(tmp_path / "r.urdf", "x", m)
    j = ET.parse(tmp_path / "r.urdf").getroot().find("joint")
    assert j.get("type") == "continuous" and j.find("limit") is None           # no invented stops
    assert float(j.find("dynamics").get("damping")) == 0.036
    assert float(j.find("dynamics").get("friction")) == 0.0


def test_prismatic_mass_is_M_and_nothing_else_is_identified(tmp_path):
    m = EA.joint_dynamics_model(dict(type="prismatic", axis_Wc=[0, 0.32, 0.95], origin_Wc=None), 8.43, 21.56)
    assert m["mass"] == 8.43 and m["damping"] == 21.56
    assert "moving_link.inertia_tensor" in m["placeholders"] and "joint.origin" in m["placeholders"]
    EA.write_urdf(tmp_path / "p.urdf", "x", m)
    r = ET.parse(tmp_path / "p.urdf").getroot()
    assert float(r.find("link[@name='moving_link']/inertial/mass").get("value")) == 8.43
    lim = r.find("joint/limit")
    assert lim.get("lower") == "-inf" and lim.get("upper") == "inf"
    assert r.find("link[@name='static_link']/inertial") is None and r.find(".//collision") is None


# ------------------------------------------------------------------------- the run's 3DGS parts next to the URDF
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402

FIELDS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
          "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def _rigid(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    T = np.eye(4)
    T[:3, :3] = Q * np.sign(np.linalg.det(Q))
    T[:3, 3] = rng.normal(size=3)
    return T


def _ply(path, n, rng):
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n%send_header\n" % (
        n, "".join(f"property float {f}\n" for f in FIELDS))
    path.write_bytes(hdr.encode() + rng.normal(size=(n, len(FIELDS))).astype("<f4").tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _handoff(root, kind, rng, T=None):
    d = root / "parttwin_handoff" / f"diagnostic_{kind}"
    d.mkdir(parents=True)
    T = _rigid(rng) if T is None else T
    parts = {p: dict(ply=f"{p}_canonical.ply", ply_sha256=_ply(d / f"{p}_canonical.ply", 50, rng), T_Wc_P=T.tolist(),
                     n_gaussians=50, immutable_gaussian_fields=FIELDS) for p in ("static", "moving")}
    a = rng.normal(size=3)
    joint = dict(type=kind, axis_Wc=(a / np.linalg.norm(a)).tolist(),
                 origin_Wc=rng.normal(size=3).tolist() if kind == "revolute" else None)
    hj = dict(object_id=f"obj_{kind}", joint=joint, parts=parts, source=dict(status="diagnostic"))
    (d / "articulated_3dgs_handoff.json").write_text(json.dumps(hj))
    return d / "articulated_3dgs_handoff.json", hj


def test_rpy_round_trip_including_gimbal_lock():
    rng = np.random.default_rng(0)
    Rs = [_rigid(rng)[:3, :3] for _ in range(200)] + [EA.matrix_from_rpy([0.3, s * np.pi / 2, 0.7]) for s in (1, -1)]
    for R in Rs:
        np.testing.assert_allclose(EA.matrix_from_rpy(EA.rpy_from_matrix(R)), R, atol=1e-12)


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_urdf_visuals_place_the_gaussians_where_the_run_has_them(tmp_path, kind):
    rng = np.random.default_rng(1)
    h, hj = _handoff(tmp_path, kind, rng)
    m = EA.joint_dynamics_model(hj["joint"], 0.4, 0.05)
    g = EA.gaussian_parts(h, hj, m)
    EA.write_urdf(tmp_path / "m.urdf", "x", m, g)
    r = ET.parse(tmp_path / "m.urdf").getroot()
    T_Wc_P = np.asarray(hj["parts"]["static"]["T_Wc_P"])
    T_Wc_link = dict(static_link=np.eye(4), moving_link=np.eye(4))
    T_Wc_link["moving_link"][:3, 3] = m["origin"]                     # joint origin, rpy 0, at q = 0
    for link, part in (("static_link", "static"), ("moving_link", "moving")):
        v = r.find(f"link[@name='{link}']/visual")
        assert v.get("name") == f"{part}_3dgs"
        assert v.find("geometry/mesh").get("filename") == f"gaussians/{part}_canonical.ply"
        o = v.find("origin")
        T = np.eye(4)
        T[:3, :3] = EA.matrix_from_rpy([float(x) for x in o.get("rpy").split()])
        T[:3, 3] = [float(x) for x in o.get("xyz").split()]
        np.testing.assert_allclose(T_Wc_link[link] @ T, T_Wc_P, atol=1e-12)
    assert r.find(".//collision") is None


def test_gaussians_fail_closed_on_wrong_sha_or_non_rigid_frame(tmp_path):
    rng = np.random.default_rng(2)
    h, hj = _handoff(tmp_path, "revolute", rng)
    m = EA.joint_dynamics_model(hj["joint"], 0.4, 0.05)
    bad = json.loads(json.dumps(hj))
    bad["parts"]["moving"]["ply_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        EA.gaussian_parts(h, bad, m)
    bad = json.loads(json.dumps(hj))
    bad["parts"]["static"]["T_Wc_P"] = (2 * np.eye(4)).tolist()
    with pytest.raises(ValueError, match="rigid"):
        EA.gaussian_parts(h, bad, m)


def test_exported_folder_is_self_contained_and_articulates_the_gaussians(tmp_path):
    pytest.importorskip("sapien")
    pytest.importorskip("pxr")
    rng = np.random.default_rng(3)
    run = tmp_path / "run"
    h, hj = _handoff(run, "revolute", rng)
    r2r = tmp_path / "r2r"
    r2r.mkdir()
    (r2r / "STATUS.json").write_text(json.dumps(dict(status="COMPARED", best_row=dict(M=0.38, c=0.039, method="t"),
                                                     episode=dict(joint_type="revolute", sha256="x"))))
    st = EA.build(run, r2r, tmp_path / "assets", log=lambda *_: None)
    assert st["status"] == "EXPORTED", st["reason"]
    out = Path(st["out_dir"])
    for p in ("static", "moving"):                                    # byte-identical copies
        assert (out / "gaussians" / f"{p}_canonical.ply").read_bytes() == (h.parent / f"{p}_canonical.ply").read_bytes()
    errs = [e["max_err"] for e in st["verification"]["sapien"]["gaussians"]]
    assert len(errs) == 4 and max(errs) < 1e-5                          # 2 parts x (q = 0, q = 0.5 rad)
    assert all(e["resolved"] and e["max_err_T_Wc_P"] < 1e-5 for e in st["verification"]["usd"]["gaussians"])
    moved = tmp_path / "elsewhere"                                     # the folder alone, run deleted
    shutil.copytree(out, moved)
    shutil.rmtree(run)
    code = ("import sapien,sys; s=sapien.Scene(); ld=s.create_urdf_loader(); ld.fix_root_link=True; "
            "a=ld.load(sys.argv[1]); print(sorted(sh.name for l in a.get_links() for c in l.entity.components "
            "if isinstance(c, sapien.render.RenderBodyComponent) for sh in c.render_shapes))")
    p = subprocess.run([sys.executable, "-c", code, str(moved / "model.urdf")], capture_output=True, text=True,
                       cwd=tmp_path)
    assert p.returncode == 0 and "['moving_3dgs', 'static_3dgs']" in p.stdout, p.stderr[-500:]
