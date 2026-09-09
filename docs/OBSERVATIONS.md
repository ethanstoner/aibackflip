# Observation and reward layout

Defined in `cpp/humanoid/Observation.cpp` and `cpp/humanoid/RewardTerms.cpp`. The names for
every element travel with the SPEC handshake, so the trainer never hardcodes an index and
`ObservationLayout.fieldNames()` is checked against the dimension by a test.

## Observation: 111 values

The 2D humanoid has 13 links and 12 joints; "non-root" below means the 12 links other than the
pelvis, in this order: chest, head, upper/lower arm L, upper/lower arm R, upper/lower leg L,
foot L, upper/lower leg R, foot R.

| Index | Count | Contents |
|---|---|---|
| 0 | 1 | pelvis height ÷ rest pelvis height |
| 1 | 2 | sin, cos of pelvis angle |
| 3 | 2 | pelvis linear velocity (m/s) |
| 5 | 1 | pelvis angular velocity (rad/s) |
| 6 | 24 | per non-root link: position − pelvis position, ÷ body height |
| 30 | 24 | per non-root link: sin, cos of (link angle − pelvis angle) |
| 54 | 12 | per non-root link: angular velocity (rad/s) |
| 66 | 24 | per joint: sin, cos of the joint angle |
| 90 | 12 | per joint: joint angular velocity (rad/s) |
| 102 | 2 | foot contact flags, left then right (0 or 1) |
| 104 | 2 | centre of mass − pelvis position, ÷ body height |
| 106 | 2 | centre of mass velocity (m/s) |
| 108 | 1 | head height ÷ rest head height |
| 109 | 2 | sin, cos of the motion phase × 2π |

### Why it looks like this

**No absolute horizontal position.** The task is identical wherever the figure stands, so world
X never appears; every position is relative to the pelvis. `Observation.absoluteHorizontalPositionIsInvisible`
translates the whole figure and asserts nothing changes beyond float rounding, with a tolerance
derived from float epsilon at that distance rather than picked by hand.

**No root-relative rotation frame.** Gravity fixes "up", so "the chest is tilted 40 degrees" is
meaningful in world terms. Rotating observations into a root frame would discard exactly the
signal a balance policy needs. This is the opposite choice from a locomotion setup on uneven
terrain, and it is deliberate.

**Angles as sin/cos, never raw.** A limb rotating through π during a flip would otherwise jump
from +3.14 to −3.14 between two adjacent frames, and the network would see a discontinuity where
the physics has none. The same reasoning applies to the phase variable: at phase 0.99 and 0.01
the motion has just looped, and the encoding keeps those adjacent.

**Deliberately redundant.** Per-link relative orientations are derivable from the joint angles
by walking the chain, and the link offsets are derivable from both. Feeding a network the same
fact in several forms costs a few dozen floats and reliably beats making it learn forward
kinematics. This follows DeepMimic, which does the same.

**Normalisation is split.** The simulator divides lengths by the figure's own dimensions, so a
value means the same thing if the humanoid is rescaled. Velocities are left in SI units, because
their spread depends on the task rather than the body, and the running mean/std normaliser in
Python handles them. Arbitrary constants like "÷ 10" are deliberately absent: they would be
guesses with no physical meaning, and the running normaliser does the job properly.

**Phase is reserved.** It stays 0 until imitation learning lands in M7. It is in the layout from
the start so the observation size does not change under the policy later.

## Reward terms: 13 values

The simulator computes raw, unweighted terms; Python applies the weights and sums them
(`python/communication/vec_env.py`). Weights become a config change rather than a rebuild, and
every component gets logged separately, the only practical way to catch reward hacking, since
an exploit shows up as one term saturating while the others flatline.

**Every term is a non-negative magnitude. The sign lives in the weight.** Terms named `*_cost`
are expected to carry a negative weight; nothing in the simulator knows whether a term is good
or bad.

| Index | Name | Range | Meaning |
|---|---|---|---|
| 0 | `alive` | 0 or 1 | 1 while running; 0 only on a real termination, not on truncation |
| 1 | `pelvis_height` | [0, 1] | pelvis height ÷ rest height, clamped |
| 2 | `head_height` | [0, 1] | head height ÷ rest height, clamped |
| 3 | `chest_upright` | [0, 1] | max(0, cos of chest tilt from its rest orientation) |
| 4 | `head_upright` | [0, 1] | max(0, cos of head tilt) |
| 5 | `com_over_support` | [0, 1] | exp(−2·d²), d = horizontal COM offset from the support centre ÷ foot length; 0 when nothing is touching the ground |
| 6 | `foot_contact` | 0, 0.5, 1 | fraction of feet in contact |
| 7 | `horizontal_drift_cost` | ≥ 0 | \|pelvis vx\| (m/s) |
| 8 | `vertical_drift_cost` | ≥ 0 | \|pelvis vy\| (m/s) |
| 9 | `angular_drift_cost` | ≥ 0 | \|pelvis angular velocity\| (rad/s) |
| 10 | `action_cost` | [0, 1] | mean squared normalized action |
| 11 | `torque_cost` | [0, 1] | mean \|torque\| ÷ maxTorque across joints |
| 12 | `joint_limit_cost` | ≥ 0 | worst limit violation (radians) |

`uprightness` compares each link against **its own rest orientation**, so a foot, which rests
horizontal, is not permanently scored as fallen over.

Starting weights for the standing task live in `RewardWeights.standing()`. They are a starting
point, not a tuned result; the per-component logs exist precisely so they can be argued with
once training has run.

## Termination

Handled in C++, next to the state it inspects. Thresholds are fractions of the rest pose rather
than absolute distances, so they scale with the figure.

| Reason | Condition |
|---|---|
| `unstable` | the solver reported a non-finite state |
| `non_finite` | a body position, velocity or angle is NaN or infinite |
| `pelvis_low` | pelvis height < 0.55 × rest height |
| `head_low` | head height < 0.55 × rest head height |
| `chest_fallen` | cos of the chest's tilt < 0.2 (about 78° over) |
| `time_limit` | episode step reached `max_episode_steps`, reported as **truncated**, not terminated |

Truncation is kept strictly distinct from termination. Hitting the time limit is not a failure,
and the value function has to bootstrap through it; collapsing the two flags into one is the
classic time-limit bug. The `alive` term reflects this: it stays 1 on truncation and drops to 0
only on a genuine termination.
