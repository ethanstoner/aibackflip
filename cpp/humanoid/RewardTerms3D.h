// Raw reward components for the 3D figure.
//
// Deliberately the same `RewardTermId` enum, in the same order, as the 2D
// figure. The weights live in Python, so reusing the enum means a reward config
// written for the 2D standing task loads unchanged against the 3D one and the
// two are directly comparable. Anything genuinely new in 3D has to earn a new
// term rather than quietly redefining an old one.
//
// Every term is still a non-negative magnitude, with the sign living entirely in
// the weight.
#pragma once

#include "humanoid/Humanoid3D.h"
#include "humanoid/Observation3D.h"
#include "humanoid/RewardTerms.h"  // RewardTermId and rewardTermNames()

namespace aibf {

// Fills all kTermCount entries. The five imitation terms are left at zero: 3D
// reference motions arrive with M10, and a term that silently reports something
// plausible before its input exists is exactly the failure this project has hit
// three times already.
void writeRewardTerms3D(const World3D& world, const Humanoid3D& figure,
                        const ObservationScales3D& scales, const Real* actions, int actionCount,
                        bool alive, Real* out);

}  // namespace aibf
