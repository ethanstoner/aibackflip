// What the 3D policy sees. The full layout is in docs/OBSERVATIONS3D.md.
//
// The design rules are the 2D ones, and they matter more here rather than less:
//
//   * No absolute horizontal position. The task is identical wherever the
//     figure stands, so x and z never appear, only offsets and velocities.
//   * Heights are divided by the rest-pose height, so the numbers are
//     dimensionless and a taller figure produces the same observation standing.
//   * **Rotations are never Euler angles and never raw quaternions.** Every
//     orientation appears as the first two columns of its rotation matrix, six
//     numbers. Euler angles gimbal lock; a quaternion has the double-cover
//     problem, where q and -q are the same rotation and the network sees two
//     unrelated inputs. The 6D form is continuous everywhere, which is the 3D
//     version of why 2D angles were fed in as (sin, cos).
//   * Deliberately redundant. Link offsets, link axes and joint rotations all
//     describe the same pose; the network is free to ignore whichever it does
//     not need, and having them costs far less than discovering one was needed.
//
// There is no root-relative rotation frame: gravity fixes "up", so the world
// frame is meaningful and expressing everything relative to a tumbling root
// would throw that away.
#pragma once

#include <string>
#include <vector>

#include "humanoid/Humanoid3D.h"

namespace aibf {

struct ObservationLayout3D {
    static constexpr int kNonRootLinks = kLinkCount - 1;
    static constexpr int kBallJoints = 6;
    static constexpr int kHingeJoints = 6;

    static constexpr int kPelvisHeight = 0;                            // 1
    static constexpr int kPelvisRotation = kPelvisHeight + 1;          // 6
    static constexpr int kPelvisLinearVelocity = kPelvisRotation + 6;  // 3
    static constexpr int kPelvisAngularVelocity = kPelvisLinearVelocity + 3;  // 3
    // Offsets from the pelvis, in the world frame, divided by rest height.
    static constexpr int kLinkOffsets = kPelvisAngularVelocity + 3;    // 3 per link
    // Each link's own +Y axis in world terms, which is where the limb points.
    static constexpr int kLinkAxes = kLinkOffsets + 3 * kNonRootLinks;  // 3 per link
    static constexpr int kLinkAngularVelocities = kLinkAxes + 3 * kNonRootLinks;  // 3 per link
    // Ball joints as 6D rotations, hinges as (sin, cos).
    static constexpr int kBallRotations = kLinkAngularVelocities + 3 * kNonRootLinks;  // 6 each
    static constexpr int kHingeAngles = kBallRotations + 6 * kBallJoints;              // 2 each
    // Three rates per ball joint, one per hinge.
    static constexpr int kJointVelocities = kHingeAngles + 2 * kHingeJoints;
    static constexpr int kFootContacts = kJointVelocities + 3 * kBallJoints + kHingeJoints;  // 2
    static constexpr int kComOffset = kFootContacts + 2;      // 3
    static constexpr int kComVelocity = kComOffset + 3;       // 3
    static constexpr int kHeadHeight = kComVelocity + 3;      // 1
    static constexpr int kPhase = kHeadHeight + 1;            // 2 (sin, cos)
    static constexpr int kDimension = kPhase + 2;

    static std::vector<std::string> fieldNames();
};

// Divisors that bring each group into roughly the same range. Velocities in
// particular span orders of magnitude between standing and a backflip, and a
// network fed raw metres per second spends its first thousand updates learning
// the scale instead of the task.
struct ObservationScales3D {
    Real height = Real(1.64);        // rest height, filled in from the figure
    Real linearVelocity = Real(5);
    Real angularVelocity = Real(10);
    Real jointVelocity = Real(10);
};

// Writes exactly ObservationLayout3D::kDimension values.
void writeObservation3D(const World3D& world, const Humanoid3D& figure,
                        const ObservationScales3D& scales, Real phase, Real* out);

}  // namespace aibf
