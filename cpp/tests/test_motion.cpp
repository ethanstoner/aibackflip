#include <cmath>
#include <limits>

#include "core/Test.h"
#include "motion/Motion2D.h"

using namespace aibf;

namespace {

MotionKeyframe key(Real time, Real rootAngle, Real firstJoint, Real height = Real(1)) {
    MotionKeyframe frame;
    frame.time = time;
    frame.rootPosition = Vec2(0, height);
    frame.rootAngle = rootAngle;
    frame.jointAngles.assign(kJointCount, Real(0));
    frame.jointAngles[0] = firstJoint;
    return frame;
}

Motion2D simpleClip(MotionInterpolation mode = MotionInterpolation::Hermite) {
    Motion2D motion;
    motion.name = "test";
    motion.interpolation = mode;
    motion.insertKeyframe(key(Real(0.0), Real(0.0), Real(0.0)));
    motion.insertKeyframe(key(Real(0.5), Real(0.3), Real(0.4)));
    motion.insertKeyframe(key(Real(1.0), Real(0.1), Real(-0.2)));
    motion.insertKeyframe(key(Real(1.5), Real(0.0), Real(0.0)));
    return motion;
}

}  // namespace

// ---------------------------------------------------------------- editing

TEST(Motion, keyframesStaySortedAndReplaceOnTheSameTime) {
    Motion2D motion;
    motion.insertKeyframe(key(Real(1.0), 0, Real(0.5)));
    motion.insertKeyframe(key(Real(0.0), 0, Real(0.1)));
    motion.insertKeyframe(key(Real(0.5), 0, Real(0.3)));

    CHECK(motion.frameCount() == 3);
    CHECK_NEAR(motion.keyframes[0].time, 0.0, 1e-6);
    CHECK_NEAR(motion.keyframes[1].time, 0.5, 1e-6);
    CHECK_NEAR(motion.keyframes[2].time, 1.0, 1e-6);

    // Authoring over an existing key edits it rather than stacking a duplicate.
    motion.insertKeyframe(key(Real(0.5), 0, Real(-0.9)));
    CHECK(motion.frameCount() == 3);
    CHECK_NEAR(motion.keyframes[1].jointAngles[0], -0.9, 1e-6);
}

TEST(Motion, insertPadsShortJointVectors) {
    Motion2D motion;
    MotionKeyframe frame;
    frame.time = 0;
    frame.jointAngles = {Real(0.5)};  // only one angle supplied
    motion.insertKeyframe(frame);
    CHECK(motion.keyframes[0].jointAngles.size() == static_cast<size_t>(kJointCount));
    CHECK_NEAR(motion.keyframes[0].jointAngles[0], 0.5, 1e-6);
    CHECK_NEAR(motion.keyframes[0].jointAngles[kJointCount - 1], 0.0, 1e-6);
}

TEST(Motion, removeAndRetime) {
    Motion2D motion = simpleClip();
    CHECK_NEAR(motion.duration(), 1.5, 1e-6);
    motion.removeKeyframe(1);
    CHECK(motion.frameCount() == 3);

    motion.retime(Real(3.0));
    CHECK_NEAR(motion.duration(), 3.0, 1e-5);
    CHECK_NEAR(motion.keyframes.front().time, 0.0, 1e-6);
    motion.removeKeyframe(99);  // out of range is a no-op, not a crash
    CHECK(motion.frameCount() == 3);
}

// ---------------------------------------------------------------- sampling

TEST(Motion, samplingAtAKeyframeReturnsThatKeyframeExactly) {
    // True for both schemes: the Hermite basis is designed to interpolate its
    // control points, so if this drifts the basis is wrong.
    for (const MotionInterpolation mode :
         {MotionInterpolation::Linear, MotionInterpolation::Hermite}) {
        const Motion2D motion = simpleClip(mode);
        for (const MotionKeyframe& frame : motion.keyframes) {
            const MotionPose pose = motion.sample(frame.time);
            CHECK_NEAR(pose.rootAngle, frame.rootAngle, 1e-5);
            CHECK_NEAR(pose.jointAngles[0], frame.jointAngles[0], 1e-5);
            CHECK_NEAR(pose.rootPosition.y, frame.rootPosition.y, 1e-5);
        }
    }
}

TEST(Motion, linearInterpolationIsTheStraightLineBetweenKeys) {
    Motion2D motion;
    motion.interpolation = MotionInterpolation::Linear;
    motion.insertKeyframe(key(Real(0), Real(0), Real(0)));
    motion.insertKeyframe(key(Real(2), Real(1), Real(-1)));

    const MotionPose pose = motion.sample(Real(0.5));
    CHECK_NEAR(pose.rootAngle, 0.25, 1e-5);
    CHECK_NEAR(pose.jointAngles[0], -0.25, 1e-5);
}

TEST(Motion, samplingOutsideTheClipClampsWhenNotLooping) {
    const Motion2D motion = simpleClip();
    CHECK_NEAR(motion.sample(Real(-5)).jointAngles[0], motion.keyframes.front().jointAngles[0],
               1e-5);
    CHECK_NEAR(motion.sample(Real(99)).jointAngles[0], motion.keyframes.back().jointAngles[0],
               1e-5);
}

TEST(Motion, aLoopingClipWrapsInTime) {
    Motion2D motion = simpleClip();
    motion.loop = true;
    const MotionPose atQuarter = motion.sample(Real(0.375));
    const MotionPose wrapped = motion.sample(Real(0.375) + motion.duration() * Real(3));
    CHECK_NEAR(wrapped.jointAngles[0], atQuarter.jointAngles[0], 1e-4);
    CHECK_NEAR(wrapped.rootAngle, atQuarter.rootAngle, 1e-4);
}

TEST(Motion, phaseSpansTheWholeClip) {
    const Motion2D motion = simpleClip();
    CHECK_NEAR(motion.samplePhase(Real(0)).jointAngles[0], motion.keyframes.front().jointAngles[0],
               1e-5);
    CHECK_NEAR(motion.samplePhase(Real(1)).jointAngles[0], motion.keyframes.back().jointAngles[0],
               1e-5);
    CHECK_NEAR(motion.phaseAt(Real(0.75)), 0.5, 1e-5);
}

TEST(Motion, rootAngleIsNeverWrapped) {
    // A backflip rotates the root through a full turn. If sampling wrapped the
    // angle into (-pi, pi], the reference would jump discontinuously partway
    // through the flip and the tracking reward would fight it.
    Motion2D motion;
    motion.insertKeyframe(key(Real(0), Real(0), 0));
    motion.insertKeyframe(key(Real(0.5), Real(-kPi), 0));
    motion.insertKeyframe(key(Real(1.0), Real(-kTwoPi), 0));

    CHECK_NEAR(motion.sample(Real(1.0)).rootAngle, -kTwoPi, 1e-4);
    // And it decreases monotonically the whole way through, with no jump.
    Real previous = Real(1);
    for (int i = 0; i <= 100; ++i) {
        const Real angle = motion.sample(Real(i) * Real(0.01)).rootAngle;
        CHECK(angle <= previous + Real(1e-4));
        previous = angle;
    }
}

TEST(Motion, hermiteSamplingStaysContinuousAcrossSegments) {
    const Motion2D motion = simpleClip(MotionInterpolation::Hermite);
    Real previous = motion.sample(Real(0)).jointAngles[0];
    for (int i = 1; i <= 300; ++i) {
        const Real angle = motion.sample(Real(i) * Real(0.005)).jointAngles[0];
        // No step larger than a smooth curve at this sample rate could produce.
        CHECK(std::abs(angle - previous) < Real(0.05));
        previous = angle;
    }
}

// ---------------------------------------------------------------- velocity

TEST(Motion, velocityMatchesTheDerivativeOfTheSampledPose) {
    // The strongest check available: the analytic derivative has to agree with
    // a numerical one taken from the sampling path itself. The imitation
    // velocity reward is only meaningful if these match.
    for (const MotionInterpolation mode :
         {MotionInterpolation::Linear, MotionInterpolation::Hermite}) {
        const Motion2D motion = simpleClip(mode);
        const Real h = Real(1e-4);
        for (int i = 1; i < 60; ++i) {
            const Real t = Real(i) * Real(0.025);
            // Skip a neighbourhood of the keyframes, where linear interpolation
            // has a genuine kink and a central difference straddles it.
            bool nearKey = false;
            for (const MotionKeyframe& frame : motion.keyframes) {
                if (std::abs(frame.time - t) < Real(0.01)) nearKey = true;
            }
            if (nearKey) continue;

            const MotionPose analytic = motion.sampleVelocity(t);
            const Real numeric =
                (motion.sample(t + h).jointAngles[0] - motion.sample(t - h).jointAngles[0]) /
                (Real(2) * h);
            CHECK_NEAR(analytic.jointAngles[0], numeric, 2e-2);

            const Real numericRoot =
                (motion.sample(t + h).rootAngle - motion.sample(t - h).rootAngle) / (Real(2) * h);
            CHECK_NEAR(analytic.rootAngle, numericRoot, 2e-2);
        }
    }
}

TEST(Motion, linearVelocityIsTheSegmentSlope) {
    Motion2D motion;
    motion.interpolation = MotionInterpolation::Linear;
    motion.insertKeyframe(key(Real(0), Real(0), Real(0)));
    motion.insertKeyframe(key(Real(2), Real(1), Real(-3)));
    const MotionPose velocity = motion.sampleVelocity(Real(1));
    CHECK_NEAR(velocity.rootAngle, 0.5, 1e-5);
    CHECK_NEAR(velocity.jointAngles[0], -1.5, 1e-5);
}

TEST(Motion, hermiteVelocityIsContinuousWhereLinearIsNot) {
    // The reason Hermite is the default: sparse keyframes under linear
    // interpolation give a piecewise-constant reference velocity, and the
    // velocity term of the imitation reward would be chasing a step function.
    const Motion2D linear = simpleClip(MotionInterpolation::Linear);
    const Motion2D hermite = simpleClip(MotionInterpolation::Hermite);
    const Real before = Real(0.49), after = Real(0.51);

    const Real linearJump = std::abs(linear.sampleVelocity(after).jointAngles[0] -
                                     linear.sampleVelocity(before).jointAngles[0]);
    const Real hermiteJump = std::abs(hermite.sampleVelocity(after).jointAngles[0] -
                                      hermite.sampleVelocity(before).jointAngles[0]);
    CHECK(linearJump > Real(1.0));
    CHECK(hermiteJump < Real(0.2));
}

TEST(Motion, aSingleKeyframeClipHasNoVelocity) {
    Motion2D motion;
    motion.insertKeyframe(key(Real(0), Real(0.4), Real(0.2)));
    const MotionPose pose = motion.sample(Real(5));
    CHECK_NEAR(pose.rootAngle, 0.4, 1e-6);
    const MotionPose velocity = motion.sampleVelocity(Real(5));
    CHECK_NEAR(velocity.rootAngle, 0.0, 1e-6);
    CHECK_NEAR(velocity.jointAngles[0], 0.0, 1e-6);
}

TEST(Motion, anEmptyClipSamplesToZeroRatherThanCrashing) {
    const Motion2D motion;
    const MotionPose pose = motion.sample(Real(1));
    CHECK(pose.jointAngles.size() == static_cast<size_t>(kJointCount));
    CHECK_NEAR(pose.rootAngle, 0.0, 1e-9);
    CHECK_NEAR(motion.duration(), 0.0, 1e-9);
    CHECK_NEAR(motion.phaseAt(Real(5)), 0.0, 1e-9);
}

// ---------------------------------------------------------------- json

TEST(Motion, roundTripsThroughJson) {
    Motion2D original = simpleClip();
    original.name = "backflip";
    original.loop = false;
    original.keyframes[1].rootPosition = Vec2(Real(0.35), Real(1.42));
    original.keyframes[2].jointAngles[kKneeL] = Real(-2.1);

    std::string error;
    const Motion2D restored = Motion2D::fromJson(Json::parse(original.toJson().dump()), &error);

    CHECK(error.empty());
    CHECK(restored.name == "backflip");
    CHECK(restored.frameCount() == original.frameCount());
    CHECK(restored.interpolation == original.interpolation);
    for (int i = 0; i < original.frameCount(); ++i) {
        CHECK_NEAR(restored.keyframes[i].time, original.keyframes[i].time, 1e-6);
        CHECK_NEAR(restored.keyframes[i].rootAngle, original.keyframes[i].rootAngle, 1e-6);
        CHECK_NEAR(restored.keyframes[i].rootPosition.x, original.keyframes[i].rootPosition.x, 1e-6);
        for (int j = 0; j < kJointCount; ++j) {
            CHECK_NEAR(restored.keyframes[i].jointAngles[static_cast<size_t>(j)],
                       original.keyframes[i].jointAngles[static_cast<size_t>(j)], 1e-6);
        }
    }
}

TEST(Motion, jsonKeyframesAreSortedOnLoad) {
    const Json json = Json::parse(R"({
        "name": "unsorted",
        "joint_count": 12,
        "frames": [
            {"time": 1.0, "root_angle": 0.2, "joints": [0,0,0,0,0,0,0,0,0,0,0,0]},
            {"time": 0.0, "root_angle": 0.0, "joints": [0,0,0,0,0,0,0,0,0,0,0,0]},
            {"time": 0.5, "root_angle": 0.1, "joints": [0,0,0,0,0,0,0,0,0,0,0,0]}
        ]
    })");
    std::string error;
    const Motion2D motion = Motion2D::fromJson(json, &error);
    CHECK(error.empty());
    CHECK_NEAR(motion.keyframes[0].time, 0.0, 1e-6);
    CHECK_NEAR(motion.keyframes[2].time, 1.0, 1e-6);
}

TEST(Motion, validationRejectsBrokenClips) {
    CHECK(!Motion2D{}.validate().empty());  // no keyframes

    {
        Motion2D motion;
        motion.insertKeyframe(key(Real(0), 0, 0));
        CHECK(!motion.validate().empty());  // zero duration
    }
    {
        Motion2D motion = simpleClip();
        motion.keyframes[2].time = motion.keyframes[1].time;  // does not advance
        CHECK(!motion.validate().empty());
    }
    {
        Motion2D motion = simpleClip();
        motion.keyframes[1].jointAngles.pop_back();
        CHECK(!motion.validate().empty());
    }
    {
        Motion2D motion = simpleClip();
        motion.keyframes[1].rootAngle = std::numeric_limits<Real>::quiet_NaN();
        CHECK(!motion.validate().empty());
    }
    CHECK(simpleClip().validate().empty());
}

TEST(Motion, clampToLimitsReportsWhatItChanged) {
    // An authored pose the humanoid physically cannot hold is a reference it can
    // never track. Better to find that before training than to spend an hour
    // wondering why the tracking reward plateaus.
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    Motion2D motion = simpleClip();
    motion.keyframes[1].jointAngles[kKneeL] = Real(5.0);   // far past the upper limit
    motion.keyframes[2].jointAngles[kHipR] = Real(-9.0);   // far past the lower limit

    const int clamped = motion.clampToLimits(config);
    CHECK(clamped == 2);
    CHECK_NEAR(motion.keyframes[1].jointAngles[kKneeL], config.joints[kKneeL].upperLimit, 1e-6);
    CHECK_NEAR(motion.keyframes[2].jointAngles[kHipR], config.joints[kHipR].lowerLimit, 1e-6);
    CHECK(motion.clampToLimits(config) == 0);  // idempotent
}

TEST(Motion, clampingEveryKeyframeDoesNotMakeTheSampledClipLegal) {
    // The trap this exists to name: clampToLimits reports zero and the clip is
    // still asking for an impossible pose. Cubic Hermite overshoots between
    // keyframes by construction, so a joint driven hard towards its limit and
    // then away again sweeps past it in the middle of a segment where no
    // keyframe is.
    //
    // Two of the five shipped clips do exactly this - forward_roll by 0.027 rad
    // and backflip by 0.017 - and nothing said so until this was measured.
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    const JointConfig& knee = config.joints[kKneeL];

    Motion2D motion;
    motion.interpolation = MotionInterpolation::Hermite;
    for (int i = 0; i < 4; ++i) {
        MotionKeyframe frame;
        frame.time = Real(i) * Real(0.25);
        frame.jointAngles.assign(kJointCount, Real(0));
        // Rest, hard to the limit, hold, back. The overshoot lands between the
        // second and third keys, where the tangent is still driving inwards.
        frame.jointAngles[kKneeL] = (i == 1 || i == 2) ? knee.lowerLimit : Real(0);
        motion.insertKeyframe(frame);
    }
    motion.keyframes[2].jointAngles[kKneeL] = knee.lowerLimit * Real(0.6);

    CHECK(motion.clampToLimits(config) == 0);  // every authored pose is legal
    const Real excess = motion.worstSampledLimitExcess(config, 512);
    CHECK(excess > Real(1e-3));  // and the clip still leaves the limits

    // Linear interpolation cannot overshoot, so the same keyframes are clean.
    Motion2D straight = motion;
    straight.interpolation = MotionInterpolation::Linear;
    CHECK_NEAR(straight.worstSampledLimitExcess(config, 512), Real(0), Real(1e-9));
}

TEST(Motion, aClipInsideItsLimitsReportsNoSampledExcess) {
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    Motion2D motion = simpleClip();
    motion.clampToLimits(config);
    for (MotionKeyframe& frame : motion.keyframes) {
        for (size_t j = 0; j < frame.jointAngles.size(); ++j) {
            // Well inside, so no amount of Hermite overshoot can escape.
            frame.jointAngles[j] *= Real(0.25);
        }
    }
    CHECK_NEAR(motion.worstSampledLimitExcess(config, 256), Real(0), Real(1e-9));
}
