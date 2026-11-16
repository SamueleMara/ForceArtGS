# Contact force sign convention

Operator-confirmed on 2026-11-13: the recorded forces are **contact reaction forces
acting on the robot**, exerted by the object at the robot–object contact point.
The existing force arrays, CSV columns, arrows and plots retain that robot-side
reaction sign.

The force exerted by the robot on the object has the opposite sign. Expressing
both vectors in the **same coordinate frame A**, at the same contact and time:

```
F_robot_on_object_A = -F_object_on_robot_A
```

Componentwise, `(Fx, Fy, Fz)_object = (-Fx, -Fy, -Fz)_robot_reaction`.
For EE-aligned recordings, for example:

```
F_robot_reaction_base = R_base_ee @ F_recorded_ee
F_robot_on_object_base = -R_base_ee @ F_recorded_ee
```

Use each dataset's existing sensor-to-EE axis mapping before rotating into the
base or camera frame. The sign convention does not establish missing mounting
calibration for the early drawers. A coordinate rotation and the sign reversal
commute; negate once when deriving an object-side force, and retain the original
recorded arrays as robot-side reactions.

For a fixed trajectory tangent `t`, the signed projection changes sign:

```
F_parallel_object = -F_parallel_robot_reaction
```

The perpendicular force **vector** also reverses. Total force magnitude,
perpendicular magnitude and the absolute tangential fraction
`eta = abs(F dot t) / (norm(F) + epsilon)` remain unchanged.

This declaration concerns force vectors. It does not change the existing recorded
torque origin or apply a torque-reference shift to the TCP/contact point. No new
tare, gravity compensation, numerical sign reversal or calibration change is
performed by this metadata update.

Machine-readable definitions are stored in:

- `configs/real_robot_tools.json`, under `contact_force_convention`.
- Each converted real scene's `metadata/contact_force_convention.json`.
- `contact_force_convention.json` alongside the shared robot measurements and the
  registered-camera force/EE visualization results.

The JSON field `object_side_force_multiplier: -1` specifies the conversion.
`automatic_sign_reversal_applied: false` explicitly states that existing data and
visualizations still show the measured robot-side reactions.
