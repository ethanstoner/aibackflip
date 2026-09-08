#include <cmath>

#include "core/Test.h"
#include "humanoid/Humanoid2D.h"

using namespace aibf;

namespace {

constexpr Real kDt = Real(1) / Real(240);

// The figure hung from a fixed pelvis. Isolates what a joint motor can do from
// whether the figure can stay upright, which is a separate question and a much
// later milestone.
struct PinnedFigure {
    World2D world;
    Humanoid2D figure;
    Humanoid2DConfig config = Humanoid2DConfig::defaults();

    PinnedFigure() {
        // Floor well below, so nothing lands on it during a hold test.
        world.addHalfSpace(HalfSpace{Vec2(0, 1), Real(-5), Real(1.0), Real(0)});
        figure.build(world, config);
        figure.reset(world, config.links[kPelvis].restPosition);
        figure.setMotorsEnabled(world, true);
        figure.link(world, kPelvis).makeStatic();
    }

    void run(Real seconds) {
        const int steps = static_cast<int>(std::lround(seconds / kDt));
        for (int i = 0; i < steps; ++i) world.step(kDt);
    }
    Real angle(int jointId) const { return figure.jointAngle(world, jointId); }
};

// The angle 70% of the way from rest towards whichever limit is further away,
// so a joint has to do real work to reach it.
Real workingTarget(const JointConfig& jc) {
    const Real reach =
        (std::abs(jc.upperLimit) > std::abs(jc.lowerLimit)) ? jc.upperLimit : jc.lowerLimit;
    return reach * Real(0.7);
}

}  // namespace

TEST(Motor, everyJointReachesItsCommandedTarget) {
    // Checks all twelve, not a representative sample: gains are per joint, and
    // one joint that cannot lift its own limb becomes a limb the policy simply
    // never learns to use.
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    for (int j = 0; j < kJointCount; ++j) {
        PinnedFigure scene;
        const Real target = workingTarget(config.joints[static_cast<size_t>(j)]);
        scene.figure.setJointTarget(scene.world, j, target);
        scene.run(Real(2.5));

        const Real error = std::abs(scene.angle(j) - target);
        CHECK(error < Real(0.05));
        CHECK(!scene.world.stats().unstable);
    }
}

TEST(Motor, holdingTorqueLeavesHeadroomForDynamicMotion) {
    // A joint that needs most of its ceiling merely to hold a static pose has
    // nothing left for a jump or a flip. Measured at 11% at worst; asserted
    // loosely so the test tracks the property, not the exact number.
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    for (int j = 0; j < kJointCount; ++j) {
        PinnedFigure scene;
        const JointConfig& jc = config.joints[static_cast<size_t>(j)];
        scene.figure.setJointTarget(scene.world, j, workingTarget(jc));
        scene.run(Real(2.0));

        Real holdTorque = 0;
        for (int i = 0; i < 120; ++i) {
            scene.world.step(kDt);
            holdTorque = std::max(holdTorque,
                                  std::abs(scene.figure.joint(scene.world, j).motorImpulse) / kDt);
        }
        CHECK(holdTorque < jc.maxTorque * Real(0.5));
    }
}

TEST(Motor, jointsAreDrivenByTorqueNotTeleported) {
    // The failure this catches is a motor implemented as a position constraint:
    // the pose would look right in a screenshot while the physics did nothing.
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    for (int j = 0; j < kJointCount; ++j) {
        PinnedFigure scene;
        const Real target = workingTarget(config.joints[static_cast<size_t>(j)]);
        scene.figure.setJointTarget(scene.world, j, target);

        scene.world.step(kDt);
        const Real afterOneStep = scene.angle(j);
        // One 1/240 s substep can only move the joint a small fraction of the
        // way, and it must have picked up angular velocity doing so.
        CHECK(std::abs(afterOneStep) < std::abs(target) * Real(0.1));
        CHECK(std::abs(scene.figure.jointVelocity(scene.world, j)) > Real(1e-3));
    }
}

TEST(Motor, aTorqueCeilingActuallyLimitsWhatAJointCanLift) {
    PinnedFigure scene;
    // The hip has to lift the whole leg against gravity; 1 N m cannot.
    scene.figure.joint(scene.world, kHipL).maxTorque = Real(1);
    scene.figure.setJointTarget(scene.world, kHipL, Real(1.5));
    scene.run(Real(3));
    CHECK(std::abs(scene.angle(kHipL)) < Real(0.3));

    // The same command with the configured ceiling gets there.
    PinnedFigure strong;
    strong.figure.setJointTarget(strong.world, kHipL, Real(1.5));
    strong.run(Real(3));
    CHECK_NEAR(strong.angle(kHipL), 1.5, 0.05);
}

TEST(Motor, disabledMotorsLeaveTheFigureLimp) {
    PinnedFigure scene;
    scene.figure.setMotorsEnabled(scene.world, false);
    scene.figure.setJointTarget(scene.world, kShoulderL, Real(2.5));
    scene.run(Real(2));
    // The arm hangs rather than following the target.
    CHECK(std::abs(scene.angle(kShoulderL)) < Real(0.2));
}

TEST(Motor, posingIsReversible) {
    // Drive into a deep pose and back. If anything integrates or accumulates
    // where it should not, the return trip lands somewhere else.
    PinnedFigure scene;
    const Real pose[kJointCount] = {Real(-0.5), Real(0.3), Real(1.8),  Real(1.2),
                                    Real(1.8),  Real(1.2), Real(1.4),  Real(-1.9),
                                    Real(-0.4), Real(1.4), Real(-1.9), Real(-0.4)};
    for (int j = 0; j < kJointCount; ++j) scene.figure.setJointTarget(scene.world, j, pose[j]);
    scene.run(Real(2.5));
    for (int j = 0; j < kJointCount; ++j) {
        CHECK(std::abs(scene.angle(j) - pose[j]) < Real(0.06));
    }

    scene.figure.holdRestPose(scene.world);
    scene.run(Real(2.5));
    for (int j = 0; j < kJointCount; ++j) CHECK(std::abs(scene.angle(j)) < Real(0.05));
}

TEST(Motor, normalizedActionsSpanExactlyTheJointRange) {
    PinnedFigure scene;
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    for (int j = 0; j < kJointCount; ++j) {
        const JointConfig& jc = config.joints[static_cast<size_t>(j)];

        scene.figure.setJointTargetNormalized(scene.world, j, Real(-1));
        CHECK_NEAR(scene.figure.joint(scene.world, j).targetAngle, jc.lowerLimit, 1e-5);

        scene.figure.setJointTargetNormalized(scene.world, j, Real(1));
        CHECK_NEAR(scene.figure.joint(scene.world, j).targetAngle, jc.upperLimit, 1e-5);

        scene.figure.setJointTargetNormalized(scene.world, j, Real(0));
        CHECK_NEAR(scene.figure.joint(scene.world, j).targetAngle,
                   (jc.lowerLimit + jc.upperLimit) * Real(0.5), 1e-5);
    }
}

TEST(Motor, outOfRangeActionsClampRatherThanExtrapolate) {
    // A Gaussian policy samples outside [-1, 1] constantly. Extrapolating would
    // hand the solver targets outside the joint's own limits, which the limit
    // constraints would then fight every step.
    PinnedFigure scene;
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    for (int j = 0; j < kJointCount; ++j) {
        const JointConfig& jc = config.joints[static_cast<size_t>(j)];
        scene.figure.setJointTargetNormalized(scene.world, j, Real(9.5));
        CHECK_NEAR(scene.figure.joint(scene.world, j).targetAngle, jc.upperLimit, 1e-5);
        scene.figure.setJointTargetNormalized(scene.world, j, Real(-9.5));
        CHECK_NEAR(scene.figure.joint(scene.world, j).targetAngle, jc.lowerLimit, 1e-5);
    }
}

TEST(Motor, applyNormalizedActionsHandlesShortAndLongVectors) {
    PinnedFigure scene;
    const std::vector<Real> few(3, Real(1));
    scene.figure.applyNormalizedActions(scene.world, few.data(), static_cast<int>(few.size()));
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    CHECK_NEAR(scene.figure.joint(scene.world, 0).targetAngle, config.joints[0].upperLimit, 1e-5);
    // Joints past the end of the vector are untouched, not zeroed by accident.
    CHECK_NEAR(scene.figure.joint(scene.world, 5).targetAngle, 0.0, 1e-6);

    // A longer vector must not read past the joint array.
    const std::vector<Real> many(kJointCount + 8, Real(-1));
    scene.figure.applyNormalizedActions(scene.world, many.data(), static_cast<int>(many.size()));
    CHECK_NEAR(scene.figure.joint(scene.world, kJointCount - 1).targetAngle,
               config.joints[kJointCount - 1].lowerLimit, 1e-5);
}

TEST(Motor, aCommandedTargetOutsideTheLimitsIsClampedAtTheSource) {
    PinnedFigure scene;
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    scene.figure.setJointTarget(scene.world, kKneeL, Real(5));
    CHECK_NEAR(scene.figure.joint(scene.world, kKneeL).targetAngle,
               config.joints[kKneeL].upperLimit, 1e-5);
    scene.run(Real(1.5));
    CHECK_NEAR(scene.figure.worstLimitViolation(scene.world), 0.0, 0.02);
}

TEST(Motor, aDrivenFigureStaysStableUnderRapidTargetChanges) {
    // A freshly initialised policy emits near-random targets at 60 Hz. The
    // solver has to survive that without the figure exploding.
    PinnedFigure scene;
    Rng rng(31);
    for (int step = 0; step < 240 * 6; ++step) {
        if (step % 4 == 0) {
            for (int j = 0; j < kJointCount; ++j) {
                scene.figure.setJointTargetNormalized(scene.world, j, rng.uniform(Real(-1), Real(1)));
            }
        }
        scene.world.step(kDt);
    }
    CHECK(!scene.world.stats().unstable);
    CHECK(scene.world.stats().velocityClampEvents == 0);
    CHECK(scene.world.stats().maxJointAnchorError < Real(5e-3));
    for (int i = 0; i < kLinkCount; ++i) CHECK(scene.figure.link(scene.world, i).isFinite());
}

TEST(Motor, anUnpinnedFigureDrivenRandomlyStaysFinite) {
    // The same shake-out with the figure free to fall, so contacts and motors
    // are fighting at once - the actual condition of an early training rollout.
    World2D world;
    world.addHalfSpace(HalfSpace{Vec2(0, 1), 0, Real(1.0), Real(0)});
    Humanoid2D figure;
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    figure.build(world, config);
    figure.reset(world, config.links[kPelvis].restPosition);
    figure.setMotorsEnabled(world, true);

    Rng rng(77);
    for (int step = 0; step < 240 * 10; ++step) {
        if (step % 4 == 0) {
            for (int j = 0; j < kJointCount; ++j) {
                figure.setJointTargetNormalized(world, j, rng.uniform(Real(-1), Real(1)));
            }
        }
        world.step(kDt);
    }
    CHECK(!world.stats().unstable);
    CHECK(world.stats().maxJointAnchorError < Real(1e-2));
    CHECK(world.stats().maxPenetration < Real(0.02));
    for (int i = 0; i < kLinkCount; ++i) CHECK(figure.link(world, i).isFinite());
}
