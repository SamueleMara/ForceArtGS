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
