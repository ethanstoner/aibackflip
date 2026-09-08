// The observation vector handed to the policy.
//
// Two rules shape it. Nothing absolute in X: the task is identical whether the
// figure stands at the origin or ten metres away, so world X never appears and
// every position is expressed relative to the pelvis. And nothing is rotated
// into a root frame: gravity fixes "up", so "the chest is tilted 40 degrees" is
// meaningful in world terms and turning it into a root-relative quantity would
// throw away exactly the signal a balance policy needs.
//
// The layout is deliberately redundant. Per-link relative orientations are
// derivable from the joint angles by walking the chain, and link offsets are
// derivable from both. Handing a network the same fact in several forms costs a
// few dozen floats and reliably beats making it learn forward kinematics.
// docs/OBSERVATIONS.md documents the exact index map.
#pragma once

#include <string>
#include <vector>

#include "humanoid/Humanoid2D.h"

namespace aibf {

struct ObservationLayout {
    static constexpr int kNonRootLinks = kLinkCount - 1;

    // Block offsets, in order.
    static constexpr int kPelvisHeight = 0;                       // 1
    static constexpr int kPelvisOrientation = kPelvisHeight + 1;  // 2  (sin, cos)
    static constexpr int kPelvisLinearVelocity = kPelvisOrientation + 2;   // 2
    static constexpr int kPelvisAngularVelocity = kPelvisLinearVelocity + 2;  // 1
    static constexpr int kLinkOffsets = kPelvisAngularVelocity + 1;          // 2 per link
    static constexpr int kLinkOrientations = kLinkOffsets + 2 * kNonRootLinks;   // 2 per link
    static constexpr int kLinkAngularVelocities =
        kLinkOrientations + 2 * kNonRootLinks;                                   // 1 per link
    static constexpr int kJointAngles = kLinkAngularVelocities + kNonRootLinks;  // 2 per joint
    static constexpr int kJointVelocities = kJointAngles + 2 * kJointCount;      // 1 per joint
    static constexpr int kFootContacts = kJointVelocities + kJointCount;         // 2
    static constexpr int kComOffset = kFootContacts + 2;                         // 2
    static constexpr int kComVelocity = kComOffset + 2;                          // 2
    static constexpr int kHeadHeight = kComVelocity + 2;                         // 1
    static constexpr int kPhase = kHeadHeight + 1;                               // 2 (sin, cos)
    static constexpr int kDimension = kPhase + 2;

    // One name per element. Used by the SPEC handshake and by the tests that
    // check the layout has not silently shifted.
    static std::vector<std::string> fieldNames();
};

// Scales that turn raw physical quantities into roughly unit-magnitude inputs.
// These are semantic, not arbitrary: lengths are divided by the figure's own
// dimensions, so the same numbers mean the same thing if the humanoid is
// rescaled. Velocities are left in SI units for the running normaliser in
// Python to handle, since their spread depends on the task, not the body.
struct ObservationScales {
    Real pelvisRestHeight = 1;
    Real headRestHeight = 1;
    Real bodyHeight = 1;

    static ObservationScales fromConfig(const Humanoid2DConfig& config);
};

// Writes exactly ObservationLayout::kDimension values. `phase` is the imitation
// motion phase in [0, 1); it stays 0 until M7 and is encoded as (sin, cos) so
// the wrap from 1 back to 0 is continuous rather than a cliff.
void writeObservation(const World2D& world, const Humanoid2D& figure,
                      const ObservationScales& scales, Real phase, Real* out);

}  // namespace aibf
