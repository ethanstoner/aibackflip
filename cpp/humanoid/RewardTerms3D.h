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

// What the reference motion asks for at the current phase, already expanded
// through the figure so the reward does not have to.
struct ImitationTargets3D {
    bool valid = false;
    std::vector<Quat> jointRotations;   // kJointCount
    std::vector<Vec3> jointRates;       // kJointCount, in the parent frame
    Real rootHeight = 0;
    Quat rootOrientation = Quat::identity();
    // Hand and foot positions *relative to the root*, so the end-effector term
    // measures limb configuration rather than where the figure happens to be.
    std::vector<Vec3> endEffectors;
    Vec3 comOffset;                     // centre of mass relative to the root
};

// Exponential falloff rates. Carried over from the 2D figure, which measured
// them against what a working policy actually achieves, and retuned there once
// already because two terms read exactly 0.000 for a policy that was visibly
// performing the motion.
struct ImitationScales3D {
    Real pose = Real(2.0);
    Real jointVelocity = Real(0.01);
    Real endEffector = Real(15.0);
    Real root = Real(20.0);
    Real com = Real(10.0);
    // Root orientation error is in radians and height in metres; this weights
    // the two against each other inside the single root term.
    Real rootAngleWeight = Real(0.5);
};

// A joint's rotation and rate relative to its rest pose, uniform across ball
// joints and hinges so the reward has one representation to compare.
Quat jointRotation(const World3D& world, const Humanoid3D& figure, int jointId);
Vec3 jointRate(const World3D& world, const Humanoid3D& figure, int jointId);

// Fills the five imitation entries of `out`, leaving them at zero when the
// targets are not valid.
void writeImitationTerms3D(const World3D& world, const Humanoid3D& figure,
                           const ImitationTargets3D& targets, const ImitationScales3D& scales,
                           Real* out);

// RMS rotation error against the reference, radians. Zero without a clip.
Real poseError3D(const World3D& world, const Humanoid3D& figure,
                 const ImitationTargets3D& targets);
// Root deviation: height and orientation combined, in metres.
Real rootError3D(const World3D& world, const Humanoid3D& figure,
                 const ImitationTargets3D& targets, Real rootAngleWeight);

// Fills all kTermCount entries. The five imitation terms are left at zero here;
// writeImitationTerms3D fills them afterwards when a reference clip exists. A
// term that reports something plausible before its input exists is exactly the
// failure this project has hit three times already.
void writeRewardTerms3D(const World3D& world, const Humanoid3D& figure,
                        const ObservationScales3D& scales, const Real* actions, int actionCount,
                        bool alive, Real* out);

}  // namespace aibf
