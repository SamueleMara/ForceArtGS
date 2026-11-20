"""ForceArtGS G9: general 1-DoF articulated dynamics, geometry-constrained full inertia, PArtTwin1D physics export.

A thin plugin layer over the pinned ForceArtGS-Physics library (``physics_lib``): the frozen viscous estimators
(api.identify, estimators/*, _validated/) are imported unchanged and used only as viscous baselines; every general
model, estimator and gate here is new, versioned and separately named.

Phases (pipelines/joint_aware_reconstruction.py): ``export`` (G8 visual handoff) -> ``dynamics`` (G9) ->
``final_physics_export`` (shared PArtTwin1D builder with the G9 physics spec).
"""
G9_VERSION = "forceartgs.g9/0.1.0"
