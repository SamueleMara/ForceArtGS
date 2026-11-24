"""Estimator-side descriptions of simulation-ready friction settings.

DOES NOT write USD, URDF, MJCF or import PArtTwin1D. The consuming
simulator/exporter must independently validate runtime semantics.
"""
from __future__ import annotations
from .objective import PhysicalParameters, rotational_viscous_to_usd_degree_units

def physx_axis_properties(joint_type: str, params: PhysicalParameters) -> dict:
    if joint_type not in ("revolute", "prismatic"):
        raise ValueError(f"unsupported joint type: {joint_type}")
    axis = "angular" if joint_type == "revolute" else "linear"
    c = (rotational_viscous_to_usd_degree_units(params.viscous)
         if axis == "angular" else params.viscous)
    prefix = f"physxJointAxis:{axis}:"
    return {
        "schema": "forceartgs.g9.physx_axis_properties/v1",
        "joint_type": joint_type,
        "requested_SI": {
            "dynamic_effort": params.dynamic_effort,
            "static_effort": params.static_effort,
            "viscous": params.viscous,
            "armature": params.armature,
            "rho_scale": params.rho_scale,
            "rho_source": params.inertia_scale_source,
        },
        "usd_attribute_suggestions_not_verified": {
            prefix+"staticFrictionEffort": params.static_effort,
            prefix+"dynamicFrictionEffort": params.dynamic_effort,
            prefix+"viscousFrictionCoefficient": c,
            prefix+"armature": params.armature,
        },
        "runtime_semantics": "UNVERIFIED_UNTIL_NATIVE_PHYSX_ROLLOUT_AND_READBACK",
        "note": "No asset is written; USD angular viscous coefficient may use degrees/s. "
                "Verify mapping in the policy-engine version before production export.",
    }
