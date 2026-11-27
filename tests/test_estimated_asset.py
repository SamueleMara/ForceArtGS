"""URDF / USD bridge writes ONLY the estimated joint damping and the moving inertia along the joint."""
import xml.etree.ElementTree as ET

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
