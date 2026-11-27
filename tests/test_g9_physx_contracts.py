"""Native-friction property mapping and safe URDF mass-scaling tests, no PhysX dependency."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

from forceartgs_g9.physx_refinement.objective import PhysicalParameters
from forceartgs_g9.physx_refinement.mapping import physx_axis_properties
from forceartgs_g9.physx_refinement.sapien_diagnostic import scale_urdf_mass_inertia


def test_native_usd_angular_units_and_efforts():
    p=PhysicalParameters(1, .2, 1.5, 1., 180/math.pi)
    d=physx_axis_properties("revolute",p)["usd_attribute_suggestions_not_verified"]
    assert d["physxJointAxis:angular:staticFrictionEffort"]==1.5
    assert d["physxJointAxis:angular:dynamicFrictionEffort"]==1.
    assert d["physxJointAxis:angular:viscousFrictionCoefficient"]==pytest.approx(1.)
    assert d["physxJointAxis:angular:armature"]==.2


def test_native_usd_prismatic_preserves_si_units():
    p=PhysicalParameters(1.,0.,4.,2.,3.)
    d=physx_axis_properties("prismatic",p)["usd_attribute_suggestions_not_verified"]
    assert d["physxJointAxis:linear:viscousFrictionCoefficient"]==3.


def test_scale_mass_and_all_inertia_components_but_not_com(tmp_path):
    source=tmp_path/"source.urdf"; out=tmp_path/"changed.urdf"
    source.write_text('''<robot name="test">
      <link name="moving_link"><inertial><origin xyz="0.1 0.2 0.3"/>
      <mass value="2"/><inertia ixx="1" iyy="2" izz="3" ixy="0.1" ixz="0.2" iyz="0.3"/>
      </inertial></link><link name="base"/>
      <joint name="joint"><parent link="base"/><child link="moving_link"/>
      <dynamics damping="5" friction="4"/></joint></robot>''')
    scale_urdf_mass_inertia(source,out,"moving_link",2.)
    root=ET.parse(out).getroot()
    assert root.find(".//mass").get("value")=="4"
    assert float(root.find(".//inertia").get("ixx"))==2.
    assert float(root.find(".//inertia").get("iyz"))==pytest.approx(.6)
    assert root.find(".//origin").get("xyz")=="0.1 0.2 0.3"
    assert root.find(".//dynamics").get("friction")=="0"
    assert root.find(".//dynamics").get("damping")=="0"


def _engine_with_scaled_cache(tmp_path, cap):
    """The engine's scaled-URDF cache without SAPIEN (only the attributes _scaled uses)."""
    import tempfile
    from collections import OrderedDict
    from forceartgs_g9.physx_refinement.sapien_diagnostic import IsolatedSapienCpu
    src = tmp_path / "source.urdf"
    src.write_text('''<robot name="test"><link name="base"/>
      <link name="moving_link"><inertial><mass value="1"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <joint name="joint" type="revolute"><parent link="base"/><child link="moving_link"/></joint></robot>''')
    e = object.__new__(IsolatedSapienCpu)
    e.urdf, e.moving_link, e.URDF_CACHE = src, "moving_link", cap
    e._tmp = tempfile.TemporaryDirectory(dir=tmp_path)
    e._urdfs, e._n_urdfs_written = OrderedDict(), 0
    return e


def _mass(path):
    return float(ET.parse(path).getroot().find(".//mass").get("value"))


def test_scaled_urdf_cache_never_hands_out_a_deleted_or_overwritten_file(tmp_path):
    # regression (coarse fridge 27/11: every G10 fit FIT_FAILED on a missing candidate_*.urdf): one engine serves all
    # fits; past the cap every new candidate reused one file name, and a batch could evict a file it had just taken
    e = _engine_with_scaled_cache(tmp_path, cap=4)
    for rho in (1., 2., 3., 4.):
        e._scaled(rho)
    batch = [1., 5., 6., 7.]                       # a hit on the oldest entry, then misses that force evictions
    paths = [e._scaled(r) for r in batch]
    assert len(set(paths)) == len(batch)
    for r, p in zip(batch, paths):
        assert p.is_file() and _mass(p) == pytest.approx(r)
    for rho in range(10, 40):                      # long optimisation far past the cap
        p = e._scaled(float(rho))
        assert p.is_file() and _mass(p) == pytest.approx(rho)
    assert len(e._urdfs) == 4 and len(list(Path(e._tmp.name).iterdir())) == 4
    assert len({str(p) for p in e._urdfs.values()}) == 4
