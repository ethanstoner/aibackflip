# 3D observation and reward layout

204 observation values, 24 actions, 18 reward terms. The layout lives in
`cpp/humanoid/Observation3D.h`; this file explains the choices behind it.

The design rules are the 2D ones, and every one of them matters more in 3D
rather than less.

## Observation: 204 values

| slot | count | contents |
|---|---|---|
| 0 | 1 | pelvis height, divided by the rest height |
| 1 | 6 | pelvis orientation, first two columns of its rotation matrix |
| 7 | 3 | pelvis linear velocity |
| 10 | 3 | pelvis angular velocity |
| 13 | 36 | offset of each non-root link from the pelvis, 3 per link |
| 49 | 36 | each non-root link's own +Y axis in world terms |
| 85 | 36 | each non-root link's angular velocity |
| 121 | 36 | the six ball joints as 6D relative rotations |
| 157 | 12 | the six hinges as (sin, cos) of their angle |
| 169 | 24 | joint rates: 3 per ball joint, 1 per hinge |
| 193 | 2 | foot contact flags |
| 195 | 3 | centre of mass, relative to the pelvis |
| 198 | 3 | centre of mass velocity |
| 201 | 1 | head height |
| 202 | 2 | phase as (sin, cos) |

`ObservationLayout3D::fieldNames()` returns a name per slot, and those names are
sent to Python in the SPEC handshake, so nothing on the far side hardcodes any of
this.

## Why rotations are never Euler angles or raw quaternions

Every orientation in the vector is the **first two columns of its rotation
matrix**, six numbers. The third column is their cross product, so nothing is
lost.

Euler angles gimbal lock, and they do it exactly where a shoulder spends most of
its time. A raw quaternion has the double-cover problem: `q` and `-q` are the
same rotation, so the same pose can arrive at the network as two completely
different inputs, and the network has to learn that they mean the same thing
before it can learn anything else.

The 6D form is continuous everywhere. It is the 3D counterpart of feeding 2D
angles in as `(sin, cos)`, and for the same reason.

## Why there is no absolute horizontal position

The task is identical wherever the figure stands, so `x` and `z` never appear,
only offsets and velocities. A policy that could see them would be free to learn
something about where the episode happened to spawn.

Heights *are* absolute, divided by the rest height, because gravity makes height
meaningful. For the same reason there is no root-relative rotation frame:
gravity fixes "up", and expressing everything relative to a tumbling root would
throw that away.

## Why it is deliberately redundant

Link offsets, link axes and joint rotations all describe the same pose. The
network is free to ignore whichever it does not need, and carrying them costs
far less than discovering later that one was needed.

## Reward terms: 18 values

The same `RewardTermId` enum as the 2D figure, in the same order, so a reward
config written for 2D standing loads unchanged and the two are comparable. Every
term is a non-negative magnitude and the sign lives in the weight.

Two of them mean something new rather than something wider:

- **`com_over_support`** now measures distance from the support centre in *both*
  horizontal axes. A sagittal figure could not fall sideways at all; this one
  can, and it has hinge ankles with no roll, so it has to answer a lateral push
  with its hips the way a person does.
- **`horizontal_drift_cost`** is likewise the length of the horizontal velocity
  rather than the magnitude of one component.

The five imitation terms stay at zero until 3D reference motions exist. A term
that reports something plausible before its input exists is exactly the failure
this project has already hit three times, so they report nothing instead.

## Action vector: 24 values

Three per ball joint and one per hinge, not one per joint:

| joint | kind | actions |
|---|---|---|
| waist, neck, shoulder_l, shoulder_r, hip_l, hip_r | ball | 3 each |
| elbow_l, elbow_r, knee_l, knee_r, ankle_l, ankle_r | hinge | 1 each |

A ball joint's three values become a target rotation through the exponential
map. Swing and twist are scaled and clamped **separately** and then composed,
because they are separately limited: treating the three values as one rotation
vector and clamping its magnitude would let a large twist eat into the swing
budget, and a shoulder would lose reach as it rotated.

The SPEC handshake describes the action vector rather than the joint list, so a
ball joint contributes three names and three bounds. Contributing one would have
sized Python's policy head two thirds too small.
