#include <cmath>
#include <filesystem>
#include <system_error>
#include <vector>

#include "core/Test.h"
#include "motion/Motion3D.h"

using namespace aibf;

namespace {

MotionKeyframe3D makeKey(Real time, const Quat& rootOrientation = Quat::identity(),
                         Real turns = 0) {
    MotionKeyframe3D key;
    key.time = time;
    key.rootPosition = Vec3(0, Real(1), 0);
    key.rootOrientation = rootOrientation;
    key.rootTurns = turns;
    key.jointRotations.assign(kJointCount, Quat::identity());
    return key;
}

// A clip that turns the root through a full backward rotation, which is the case
// the 2D format kept a scalar unwrapped angle for.
Motion3D backflipLike() {
    Motion3D motion;
    motion.name = "flip";
    const Vec3 sagittal(0, 0, 1);
    for (int i = 0; i <= 4; ++i) {
        const Real u = Real(i) / Real(4);
        MotionKeyframe3D key = makeKey(u, Quat::fromAxisAngle(sagittal, u * Real(2) * kPi), u);
        key.rootPosition = Vec3(0, Real(1) + std::sin(u * kPi) * Real(0.6), 0);
        motion.insertKeyframe(key);
    }
    return motion;
}

}  // namespace

TEST(Motion3D, keyframesStayInTimeOrder) {
    Motion3D motion;
    motion.insertKeyframe(makeKey(Real(2)));
    motion.insertKeyframe(makeKey(Real(0)));
    motion.insertKeyframe(makeKey(Real(1)));
    CHECK(motion.keyframes.size() == 3);
    CHECK_NEAR(motion.keyframes[0].time, 0.0, 1e-6);
    CHECK_NEAR(motion.keyframes[1].time, 1.0, 1e-6);
    CHECK_NEAR(motion.keyframes[2].time, 2.0, 1e-6);
    CHECK(motion.validate().empty());
}

TEST(Motion3D, insertingAtAnExistingTimeReplacesRatherThanDuplicates) {
    Motion3D motion;
    motion.insertKeyframe(makeKey(Real(0)));
    motion.insertKeyframe(makeKey(Real(1), Quat::fromAxisAngle(Vec3(1, 0, 0), Real(0.5))));
    motion.insertKeyframe(makeKey(Real(1), Quat::fromAxisAngle(Vec3(1, 0, 0), Real(0.9))));
    CHECK(motion.keyframes.size() == 2);
    CHECK_NEAR(angleOf(motion.keyframes[1].rootOrientation), 0.9, 1e-4);
}

TEST(Motion3D, samplingLandsExactlyOnTheKeyframes) {
    const Motion3D motion = backflipLike();
    for (const MotionKeyframe3D& key : motion.keyframes) {
        const MotionPose3D pose = motion.sample(key.time);
        CHECK(angleOf(pose.rootOrientation * conjugate(key.rootOrientation)) < Real(1e-3));
        CHECK_NEAR(pose.rootTurns, key.rootTurns, 1e-4);
    }
}

TEST(Motion3D, rotationsAreSlerpedAndNotLerped) {
    // Halfway through a 180 degree arc, slerp gives exactly 90 degrees. A
    // component lerp gives the same *direction* but a different angle, and the
    // difference is largest exactly in the middle of the arc, which is where a
    // tracking term is looking hardest.
    Motion3D motion;
    motion.insertKeyframe(makeKey(Real(0), Quat::identity()));
    motion.insertKeyframe(makeKey(Real(1), Quat::fromAxisAngle(Vec3(0, 0, 1), kPi * Real(0.9))));

    const MotionPose3D middle = motion.sample(Real(0.5));
    CHECK_NEAR(angleOf(middle.rootOrientation), 0.9 * kPi * 0.5, 1e-3);
    CHECK_NEAR(length(middle.rootOrientation), 1.0, 1e-5);
}

TEST(Motion3D, aFullTurnIsDistinguishableFromNoTurn) {
    // The reason MotionPose3D carries a turn count at all.
    //
    // In 2D the root angle was a scalar stored unwrapped, so a backflip was
    // -6.28 radians and a stand was 0, and nothing could confuse them. A
    // quaternion cannot express that: it double-covers, so one full turn and no
    // turn are the same value. Without the count, a backflip clip and a standing
    // clip agree at their endpoints.
    const Motion3D motion = backflipLike();
    const MotionPose3D start = motion.sample(Real(0));
    const MotionPose3D end = motion.sample(Real(1));

    // The orientations really are indistinguishable, which is the trap.
    CHECK(angleOf(end.rootOrientation * conjugate(start.rootOrientation)) < Real(1e-2));
    // The turn count is not.
    CHECK_NEAR(end.rootTurns - start.rootTurns, 1.0, 1e-3);
}

TEST(Motion3D, velocityIsNonZeroWhereTheClipMoves) {
    const Motion3D motion = backflipLike();
    Vec3 rootVelocity, rootSpin;
    std::vector<Vec3> jointRates;
    motion.sampleVelocity(Real(0.5), rootVelocity, rootSpin, jointRates);

    CHECK(aibf::isFinite(rootVelocity));
    CHECK(aibf::isFinite(rootSpin));
    // A full turn over one second is 2*pi rad/s about the sagittal axis.
    CHECK_NEAR(std::abs(rootSpin.z), 2.0 * kPi, 0.6);
    CHECK(jointRates.size() == static_cast<size_t>(kJointCount));
}

TEST(Motion3D, aStillClipHasZeroVelocity) {
    // The control. Without it, a velocity routine that returned noise would pass
    // the test above just as well.
    Motion3D motion;
    motion.insertKeyframe(makeKey(Real(0)));
    motion.insertKeyframe(makeKey(Real(1)));

    Vec3 rootVelocity, rootSpin;
    std::vector<Vec3> jointRates;
    motion.sampleVelocity(Real(0.5), rootVelocity, rootSpin, jointRates);
    CHECK(length(rootVelocity) < Real(1e-4));
    CHECK(length(rootSpin) < Real(1e-4));
}

TEST(Motion3D, clampingPullsAnImpossiblePoseBackInsideTheLimits) {
    const Humanoid3DConfig cfg = Humanoid3DConfig::defaults();
    Motion3D motion;
    MotionKeyframe3D key = makeKey(Real(0));
    // A knee bent the wrong way past its stop, and a shoulder swung far outside
    // its cone.
    key.jointRotations[kKneeL] = Quat::fromAxisAngle(Vec3(0, 0, 1), Real(2.0));
    key.jointRotations[kShoulderL] = Quat::fromAxisAngle(Vec3(1, 0, 0), Real(3.0));
    motion.insertKeyframe(key);

    const int changed = motion.clampToLimits(cfg);
    CHECK(changed >= 2);

    const Quat knee = motion.keyframes[0].jointRotations[kKneeL];
    const Real kneeAngle = twistAngle(knee, normalize(cfg.joints[kKneeL].hingeAxis));
    CHECK(kneeAngle <= cfg.joints[kKneeL].upperLimit + Real(1e-3));
    CHECK(kneeAngle >= cfg.joints[kKneeL].lowerLimit - Real(1e-3));

    Quat swing, twist;
    swingTwistDecomposition(motion.keyframes[0].jointRotations[kShoulderL], Vec3(0, 1, 0), swing,
                            twist);
    CHECK(angleOf(swing) <= cfg.joints[kShoulderL].coneAngle + Real(1e-3));
}

TEST(Motion3D, clampingLeavesALegalPoseAlone) {
    const Humanoid3DConfig cfg = Humanoid3DConfig::defaults();
    Motion3D motion;
    MotionKeyframe3D key = makeKey(Real(0));
    key.jointRotations[kKneeL] = Quat::fromAxisAngle(Vec3(0, 0, 1), Real(-1.0));
    motion.insertKeyframe(key);
    CHECK(motion.clampToLimits(cfg) == 0);
}

TEST(Motion3D, aHingeIsReducedToItsOwnAxis) {
    // A clip that asks a knee to rotate off its hinge axis is asking for a pose
    // the joint does not have. Keeping that component would author a motion the
    // figure can never perform, and the tracking reward would then punish it
    // forever for a shape that was never reachable.
    const Humanoid3DConfig cfg = Humanoid3DConfig::defaults();
    Motion3D motion;
    MotionKeyframe3D key = makeKey(Real(0));
    key.jointRotations[kKneeL] = Quat::fromAxisAngle(normalize(Vec3(1, 1, 1)), Real(0.8));
    motion.insertKeyframe(key);
    motion.clampToLimits(cfg);

    const Quat knee = motion.keyframes[0].jointRotations[kKneeL];
    const Vec3 axis = normalize(cfg.joints[kKneeL].hingeAxis);
    Quat swing, twist;
    swingTwistDecomposition(knee, axis, swing, twist);
    CHECK(angleOf(swing) < Real(1e-3));
}

TEST(Motion3D, roundTripsThroughJson) {
    const Motion3D original = backflipLike();
    std::string error;
    const Json json = Json::parse(original.toJson().dump(), &error);
    CHECK(error.empty());
    const Motion3D restored = Motion3D::fromJson(json);

    CHECK(restored.validate().empty());
    CHECK(restored.keyframes.size() == original.keyframes.size());
    CHECK(restored.name == original.name);
    for (size_t i = 0; i < restored.keyframes.size(); ++i) {
        CHECK_NEAR(restored.keyframes[i].time, original.keyframes[i].time, 1e-5);
        CHECK_NEAR(restored.keyframes[i].rootTurns, original.keyframes[i].rootTurns, 1e-5);
        CHECK(angleOf(restored.keyframes[i].rootOrientation *
                      conjugate(original.keyframes[i].rootOrientation)) < Real(1e-3));
    }
}

TEST(Motion3D, validationRejectsAClipThatCannotBeSampled) {
    {
        Motion3D empty;
        CHECK(!empty.validate().empty());
    }
    {
        // Not starting at zero, so phase 0 would not mean the start.
        Motion3D late;
        late.insertKeyframe(makeKey(Real(0.5)));
        CHECK(!late.validate().empty());
    }
    {
        Motion3D wrongWidth;
        MotionKeyframe3D key = makeKey(Real(0));
        key.jointRotations.pop_back();
        wrongWidth.insertKeyframe(key);
        CHECK(!wrongWidth.validate().empty());
    }
}

TEST(Motion3D, aLoopingClipWrapsAndANonLoopingOneClamps) {
    Motion3D looping = backflipLike();
    looping.loop = true;
    const MotionPose3D wrapped = looping.sample(Real(1.25));
    const MotionPose3D early = looping.sample(Real(0.25));
    CHECK_NEAR(wrapped.rootTurns, early.rootTurns, 1e-3);

    Motion3D once = backflipLike();
    once.loop = false;
    const MotionPose3D clamped = once.sample(Real(5));
    CHECK_NEAR(clamped.rootTurns, once.keyframes.back().rootTurns, 1e-4);
}
