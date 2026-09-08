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
    kTermCount
};

const std::vector<std::string>& rewardTermNames();

// `lastActions` is the normalized action vector most recently applied (kJointCount
// entries) and may be null before the first step. `controlDt` converts the
// solver's accumulated motor impulses back into torques.
void writeRewardTerms(const World2D& world, const Humanoid2D& figure,
                      const ObservationScales& scales, const Real* lastActions, Real physicsDt,
                      Real* out);

}  // namespace aibf
