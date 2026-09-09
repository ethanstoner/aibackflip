// Raw reward components.
//
// The simulator computes the *terms*; Python applies the weights and sums them.
// That split exists for two reasons. Reward weights become a config change
// rather than a C++ rebuild, and every component gets logged separately - which
// is the only practical way to catch reward hacking, since an exploit shows up
// as one term saturating while the rest flatline.
//
// Every term is a non-negative magnitude. The *sign* lives entirely in the
// weight: terms named `*_cost` are things to penalise and are expected to carry
// a negative weight in the config. Nothing here knows whether it is good or bad.
#pragma once

#include <string>
#include <vector>

#include "humanoid/Humanoid2D.h"
#include "humanoid/Observation.h"

namespace aibf {

enum RewardTermId : int {
    kTermAlive = 0,          // 1 while the episode is running
    kTermPelvisHeight,       // pelvis height / rest height, clamped to [0, 1]
    kTermHeadHeight,         // head height / rest head height, clamped to [0, 1]
    kTermChestUpright,       // max(0, cos of chest tilt from its rest orientation)
    kTermHeadUpright,        // max(0, cos of head tilt)
    kTermComOverSupport,     // exp(-k * horizontal COM offset from the support centre)
    kTermFootContact,        // fraction of feet touching the ground: 0, 0.5 or 1
    kTermHorizontalDriftCost,  // |pelvis vx|, m/s
    kTermVerticalDriftCost,    // |pelvis vy|, m/s
    kTermAngularDriftCost,     // |pelvis angular velocity|, rad/s
    kTermActionCost,           // mean squared normalized action, in [0, 1]
    kTermTorqueCost,           // mean |torque| / maxTorque across joints, in [0, 1]
    kTermJointLimitCost,       // worst limit violation, radians

    // --- imitation (DeepMimic-style tracking), all zero without a reference ---
    //
    // Every one is an exponential of a squared error, exp(-k * e^2), so it sits
    // in (0, 1] and falls off smoothly. A linear penalty would be dominated by
    // whichever joint happens to be worst; the exponential keeps the gradient
    // meaningful once tracking is already close, which is where the difference
    // between "roughly a backflip" and "a backflip" lives.
    kTermPoseMatch,            // joint angles against the reference
    kTermJointVelocityMatch,   // joint velocities against the reference
    kTermEndEffectorMatch,     // hands and feet, relative to the root
    kTermRootMatch,            // root height and orientation
    kTermComMatch,             // centre of mass, relative to the root

    kTermCount
};

const std::vector<std::string>& rewardTermNames();

// What the reference motion asks for at the current phase, already expanded
// through forward kinematics so the reward does not have to.
struct ImitationTargets {
    bool valid = false;
    std::vector<Real> jointAngles;      // kJointCount
    std::vector<Real> jointVelocities;  // kJointCount
    Real rootHeight = 0;
    Real rootAngle = 0;
    // Hand and foot positions *relative to the root*, so the end-effector term
    // measures limb configuration rather than where the figure happens to be.
    std::vector<Vec2> endEffectors;
    Vec2 comOffset;  // centre of mass relative to the root
};

// Exponential falloff rates. Defaults follow DeepMimic's, rescaled where the 2D
// figure has fewer joints contributing to a sum.
struct ImitationScales {
    Real pose = Real(2.0);
    Real jointVelocity = Real(0.1);
    Real endEffector = Real(40.0);
    Real root = Real(20.0);
    Real com = Real(10.0);
    // Root orientation error is in radians and position in metres; this weights
    // the angle against the height inside the single root term.
    Real rootAngleWeight = Real(0.5);
};

// Which links count as end effectors for the tracking term.
constexpr int kEndEffectors[4] = {kLowerArmL, kLowerArmR, kFootL, kFootR};

// Fills the five imitation entries of `out` (which must be the full term
// vector). Leaves them at zero when the targets are not valid.
void writeImitationTerms(const World2D& world, const Humanoid2D& figure,
                         const ImitationTargets& targets, const ImitationScales& scales,
                         const ObservationScales& observationScales, Real* out);

// Root-mean-square joint angle error against the reference, in radians. Used
// for early termination: an episode that has drifted this far from the
// reference is not going to recover, and continuing it only fills the rollout
// with samples from a part of the state space the motion never visits.
Real poseTrackingError(const World2D& world, const Humanoid2D& figure,
                       const ImitationTargets& targets);

// `lastActions` is the normalized action vector most recently applied (kJointCount
// entries) and may be null before the first step. `controlDt` converts the
// solver's accumulated motor impulses back into torques.
void writeRewardTerms(const World2D& world, const Humanoid2D& figure,
                      const ObservationScales& scales, const Real* lastActions, Real physicsDt,
                      Real* out);

}  // namespace aibf
