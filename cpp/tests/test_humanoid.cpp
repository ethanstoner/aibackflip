#include <cmath>

#include "core/Test.h"
#include "humanoid/Humanoid2D.h"

using namespace aibf;

namespace {

constexpr Real kDt = Real(1) / Real(240);

struct Scene {
    World2D world;
    Humanoid2D figure;

    explicit Scene(const Humanoid2DConfig& cfg = Humanoid2DConfig::defaults()) {
        world.addHalfSpace(HalfSpace{Vec2(0, 1), 0, Real(1.0), Real(0)});
        figure.build(world, cfg);
        figure.reset(world, cfg.links[kPelvis].restPosition);
    }

    void run(Real seconds) {
        const int steps = static_cast<int>(std::lround(seconds / kDt));
        for (int i = 0; i < steps; ++i) world.step(kDt);
    }
    Real pelvisHeight() const { return figure.link(world, kPelvis).position.y; }
    bool allFinite() const {
        for (const RigidBody2D& b : world.bodies()) {
            if (!b.isFinite()) return false;
        }
        return true;
    }
};

}  // namespace

// ---------------------------------------------------------------- config

TEST(HumanoidConfig, theDefaultConfigIsValid) {
    const Humanoid2DConfig cfg = Humanoid2DConfig::defaults();
    const std::string problem = cfg.validate();
    CHECK(problem.empty());
    CHECK(cfg.links.size() == static_cast<size_t>(kLinkCount));
    CHECK(cfg.joints.size() == static_cast<size_t>(kJointCount));
}

TEST(HumanoidConfig, theFigureHasHumanProportions) {
    const Humanoid2DConfig cfg = Humanoid2DConfig::defaults();
    CHECK_NEAR(cfg.totalMass(), 69.0, 0.5);
    CHECK_NEAR(cfg.restPelvisHeight(), 1.00, 0.01);
    CHECK_NEAR(cfg.restHeadHeight(), 1.64, 0.02);
}

TEST(HumanoidConfig, theFeetRestExactlyOnTheGround) {
    // If the rest pose floats or intersects, every episode begins with either a
    // drop or a penetration impulse, and the standing reward starts from a lie.
    const Humanoid2DConfig cfg = Humanoid2DConfig::defaults();
    const LinkConfig& foot = cfg.links[kFootL];
    CHECK_NEAR(foot.restPosition.y - foot.radius, 0.0, 1e-5);
}

TEST(HumanoidConfig, everyLimbSegmentMeetsTheNextAtItsJoint) {
    // Walks the skeleton and checks the pivot really is on both links' surfaces
    // in the rest pose. This is what makes "all joints read zero at rest" true.
    const Humanoid2DConfig cfg = Humanoid2DConfig::defaults();
    for (const JointConfig& jc : cfg.joints) {
        for (const int side : {jc.parent, jc.child}) {
            const LinkConfig& link = cfg.links[static_cast<size_t>(side)];
            const Vec2 axis = rotate(Vec2(0, link.halfLength), link.restAngle);
            const Real toEndA = length(jc.restAnchor - (link.restPosition + axis));
            const Real toEndB = length(jc.restAnchor - (link.restPosition - axis));
            CHECK(std::min(toEndA, toEndB) < link.radius + Real(0.06));
        }
    }
}

TEST(HumanoidConfig, roundTripsThroughJson) {
    const Humanoid2DConfig original = Humanoid2DConfig::defaults();
    std::string error;
    const Json json = Json::parse(original.toJson().dump(), &error);
    CHECK(error.empty());

    const Humanoid2DConfig restored = Humanoid2DConfig::fromJson(json);
    CHECK(restored.validate().empty());
    CHECK_NEAR(restored.totalMass(), original.totalMass(), 1e-3);
    for (size_t i = 0; i < original.joints.size(); ++i) {
        CHECK(restored.joints[i].name == original.joints[i].name);
        CHECK_NEAR(restored.joints[i].lowerLimit, original.joints[i].lowerLimit, 1e-5);
        CHECK_NEAR(restored.joints[i].stiffness, original.joints[i].stiffness, 1e-3);
    }
    for (size_t i = 0; i < original.links.size(); ++i) {
        CHECK_NEAR(restored.links[i].mass, original.links[i].mass, 1e-4);
        CHECK_NEAR(restored.links[i].restAngle, original.links[i].restAngle, 1e-5);
    }
}

TEST(HumanoidConfig, aPartialJsonOverrideKeepsEveryOtherDefault) {
    const Json patch = Json::parse(R"({
        "joints": [ {"name": "knee_l", "kp": 12345, "max_torque": 999} ]
    })");
    const Humanoid2DConfig cfg = Humanoid2DConfig::fromJson(patch);
    CHECK(cfg.validate().empty());
    CHECK_NEAR(cfg.joints[kKneeL].stiffness, 12345.0, 1e-3);
    CHECK_NEAR(cfg.joints[kKneeL].maxTorque, 999.0, 1e-3);
    // Untouched fields and every other joint keep their defaults.
    CHECK_NEAR(cfg.joints[kKneeL].damping, Humanoid2DConfig::defaults().joints[kKneeL].damping, 1e-3);
    CHECK_NEAR(cfg.joints[kKneeR].stiffness, Humanoid2DConfig::defaults().joints[kKneeR].stiffness, 1e-3);
}

TEST(HumanoidConfig, validationRejectsBrokenConfigs) {
    {
        Humanoid2DConfig cfg = Humanoid2DConfig::defaults();
        cfg.links[kChest].mass = 0;
        CHECK(!cfg.validate().empty());
    }
    {
        Humanoid2DConfig cfg = Humanoid2DConfig::defaults();
        cfg.joints[kKneeL].lowerLimit = Real(1);
        cfg.joints[kKneeL].upperLimit = Real(-1);
        CHECK(!cfg.validate().empty());
    }
    {
        Humanoid2DConfig cfg = Humanoid2DConfig::defaults();
        cfg.joints[kHipR].child = cfg.joints[kHipR].parent;
        CHECK(!cfg.validate().empty());
    }
    {
        // A mistyped anchor is the authoring error most likely to happen and
        // hardest to read once the figure has snapped into a knot.
        Humanoid2DConfig cfg = Humanoid2DConfig::defaults();
        cfg.joints[kAnkleR].restAnchor = Vec2(0, 5);
        CHECK(!cfg.validate().empty());
    }
}

// ---------------------------------------------------------------- rest pose

TEST(Humanoid, everyJointReadsZeroInTheRestPose) {
    Scene scene;
    for (int i = 0; i < kJointCount; ++i) {
        CHECK_NEAR(scene.figure.jointAngle(scene.world, i), 0.0, 1e-5);
    }
}

TEST(Humanoid, theRestPoseSatisfiesEveryAnchorExactly) {
    // setPose is forward kinematics, so the solver should have nothing to fix.
    Scene scene;
    for (int i = 0; i < kJointCount; ++i) {
        const RevoluteJoint2D& rj = scene.figure.joint(scene.world, i);
        const Real error =
            rj.anchorError(scene.world.body(rj.bodyA), scene.world.body(rj.bodyB));
        CHECK(error < Real(1e-5));
    }
}

TEST(Humanoid, theRestPoseIsWithinEveryJointLimit) {
    Scene scene;
    CHECK_NEAR(scene.figure.worstLimitViolation(scene.world), 0.0, 1e-6);
}

TEST(Humanoid, forwardKinematicsHoldsAnchorsForArbitraryPoses) {
    Scene scene;
    Rng rng(11);
    for (int trial = 0; trial < 20; ++trial) {
        Real angles[kJointCount];
        for (int i = 0; i < kJointCount; ++i) {
            const JointConfig& jc = scene.figure.config().joints[static_cast<size_t>(i)];
            angles[i] = rng.uniform(jc.lowerLimit, jc.upperLimit);
        }
        scene.figure.setPose(scene.world, Vec2(rng.uniform(-2, 2), rng.uniform(1, 3)),
                             rng.uniform(-kPi, kPi), angles);

        for (int i = 0; i < kJointCount; ++i) {
            const RevoluteJoint2D& rj = scene.figure.joint(scene.world, i);
            CHECK(rj.anchorError(scene.world.body(rj.bodyA), scene.world.body(rj.bodyB)) < 1e-4f);
            CHECK_NEAR(scene.figure.jointAngle(scene.world, i), angles[i], 1e-4);
        }
    }
}

TEST(Humanoid, resetClearsMotionAndPose) {
    Scene scene;
    scene.world.body(scene.figure.bodyIndex(kChest)).velocity = Vec2(5, 5);
    scene.run(Real(0.5));
    scene.figure.reset(scene.world, Vec2(0, Real(1.0)));

    for (int i = 0; i < kLinkCount; ++i) {
        const RigidBody2D& body = scene.figure.link(scene.world, i);
        CHECK_NEAR(length(body.velocity), 0.0, 1e-9);
        CHECK_NEAR(body.angularVelocity, 0.0, 1e-9);
    }
    CHECK_NEAR(scene.pelvisHeight(), 1.0, 1e-5);
    CHECK_NEAR(scene.figure.worstLimitViolation(scene.world), 0.0, 1e-6);
}

TEST(Humanoid, randomisedResetsStayLegal) {
    Scene scene;
    Rng rng(4242);
    ResetNoise noise;
    noise.rootAngle = Real(0.15);
    noise.rootHeight = Real(0.05);
    noise.jointAngle = Real(0.2);
    noise.linearVelocity = Real(0.3);
    noise.angularVelocity = Real(0.3);

    for (int trial = 0; trial < 50; ++trial) {
        scene.figure.reset(scene.world, Vec2(0, Real(1.0)), noise, rng);
        CHECK(scene.allFinite());
        CHECK_NEAR(scene.figure.worstLimitViolation(scene.world), 0.0, 1e-5);
        for (int i = 0; i < kJointCount; ++i) {
            const RevoluteJoint2D& rj = scene.figure.joint(scene.world, i);
            CHECK(rj.anchorError(scene.world.body(rj.bodyA), scene.world.body(rj.bodyB)) < 1e-4f);
        }
    }
}

// ---------------------------------------------------------------- dynamics

TEST(Humanoid, collapsesUnderGravityWhenUnactuated) {
    // M1's headline claim. With no motors the figure is a ragdoll and must fall,
    // stay finite, and keep every joint attached on the way down.
    Scene scene;
    scene.figure.setMotorsEnabled(scene.world, false);
    const Real startHeight = scene.pelvisHeight();

    scene.run(Real(3));

    CHECK(scene.pelvisHeight() < startHeight - Real(0.3));
    CHECK(scene.allFinite());
    CHECK(!scene.world.stats().unstable);
    CHECK(scene.world.stats().maxJointAnchorError < Real(2e-3));
    CHECK(scene.world.stats().velocityClampEvents == 0);
}

TEST(Humanoid, aCollapsedRagdollComesToRestOnTheFloor) {
    Scene scene;
    scene.figure.setMotorsEnabled(scene.world, false);
    scene.run(Real(8));

    Real fastest = 0;
    Real lowest = Real(1e9);
    for (int i = 0; i < kLinkCount; ++i) {
        const RigidBody2D& body = scene.figure.link(scene.world, i);
        fastest = std::max(fastest, length(body.velocity));
        lowest = std::min(lowest, body.position.y - body.radius);
    }
    CHECK(fastest < Real(0.35));                  // settled, not still tumbling
    CHECK(lowest > -Real(3) * kLinearSlop);       // and not sunk through the floor
    CHECK(scene.world.stats().maxPenetration < Real(3) * kLinearSlop);
}

TEST(Humanoid, jointLimitsHoldAllTheWayThroughACollapse) {
    Scene scene;
    scene.figure.setMotorsEnabled(scene.world, false);

    Real worst = 0;
    for (int i = 0; i < 240 * 6; ++i) {
        scene.world.step(kDt);
        worst = std::max(worst, scene.figure.worstLimitViolation(scene.world));
    }
    // Impulse limits are not hard constraints, but a knee should never bend
    // backwards by anything a viewer would notice.
    CHECK(worst < Real(0.12));
}

TEST(Humanoid, survivesBeingThrownAtTheFloor) {
    Scene scene;
    scene.figure.setMotorsEnabled(scene.world, false);
    scene.figure.setPose(scene.world, Vec2(0, Real(4)), Real(2.0), nullptr);
    for (int i = 0; i < kLinkCount; ++i) {
        RigidBody2D& body = scene.figure.link(scene.world, i);
        body.velocity = Vec2(Real(6), Real(-25));
        body.angularVelocity = Real(12);
    }

    scene.run(Real(4));

    CHECK(scene.allFinite());
    CHECK(!scene.world.stats().unstable);
    CHECK(scene.world.stats().maxJointAnchorError < Real(5e-3));
}

TEST(Humanoid, mouseDraggingPullsTheWholeChain) {
    // Grabbing one hand must move the body through the joints rather than
    // detaching the limb - the check that the constraint chain actually
    // transmits force.
    Scene scene;
    scene.figure.setMotorsEnabled(scene.world, false);
    const int32_t hand = scene.figure.bodyIndex(kLowerArmL);
    const Vec2 grabPoint = scene.world.body(hand).position;

    scene.world.grab(hand, grabPoint);
    scene.world.mouse().target = grabPoint + Vec2(Real(1.5), Real(1.2));
    scene.run(Real(1.5));

    const Real handMoved = length(scene.world.body(hand).position - grabPoint);
    const Real chestMoved =
        length(scene.figure.link(scene.world, kChest).position -
               scene.figure.config().links[kChest].restPosition);

    CHECK(handMoved > Real(0.3));
    CHECK(chestMoved > Real(0.1));  // the pull propagated through the shoulder
    CHECK(scene.world.stats().maxJointAnchorError < Real(5e-3));
    CHECK(scene.allFinite());
}

TEST(Humanoid, physicsIsDeterministicForAGivenSeed) {
    // Reproducibility is a hard requirement for diagnosing a training run.
    auto simulate = []() {
        Scene scene;
        Rng rng(99);
        ResetNoise noise;
        noise.rootAngle = Real(0.1);
        noise.jointAngle = Real(0.15);
        scene.figure.reset(scene.world, Vec2(0, Real(1.0)), noise, rng);
        scene.figure.setMotorsEnabled(scene.world, false);
        scene.run(Real(2));
        return scene.figure.link(scene.world, kHead).position;
    };
    const Vec2 a = simulate();
    const Vec2 b = simulate();
    CHECK(a == b);
}
