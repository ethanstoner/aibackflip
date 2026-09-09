#include <cmath>
#include <cstdio>
#include <vector>

#include "core/Test.h"
#include "physics/World3D.h"

using namespace aibf;

namespace {

constexpr Real kDt = Real(1) / Real(240);

RigidBody3D makeCapsule(const Vec3& position, Real radius, Real halfLength, Real mass) {
    RigidBody3D body;
    body.setCapsuleWithMass(radius, halfLength, mass);
    body.position = position;
    return body;
}

// A capsule pinned to the immovable world by a joint at its upper end, hanging
// down under gravity. The standard test rig: if the joint holds, the anchors
// stay together no matter how hard the limb swings.
struct PendulumRig {
    World3D world;
    int32_t anchor = -1;
    int32_t limb = -1;
};

void run(World3D& world, Real seconds) {
    const int steps = int(seconds / kDt);
    for (int i = 0; i < steps; ++i) world.step(kDt);
}

}  // namespace

// ---------------------------------------------------------------- point constraint

TEST(BallJoint3D, holdsALimbAgainstGravity) {
    World3D world;

    RigidBody3D anchor = makeCapsule(Vec3(0, 2, 0), Real(0.05), Real(0.05), Real(1));
    anchor.makeStatic();
    anchor.collisionGroup = 1;
    const int32_t anchorIndex = world.addBody(anchor);

    RigidBody3D limb = makeCapsule(Vec3(0, Real(1.7), 0), Real(0.05), Real(0.3), Real(3));
    limb.collisionGroup = 1;
    const int32_t limbIndex = world.addBody(limb);

    BallJoint3D joint;
    joint.bodyA = anchorIndex;
    joint.bodyB = limbIndex;
    joint.localAnchorA = Vec3(0, 0, 0);
    joint.localAnchorB = Vec3(0, Real(0.3), 0);
    world.addBallJoint(joint);

    run(world, Real(4));

    CHECK(!world.stats().unstable);
    // Millimetres, not centimetres. This is the same standard the 2D engine is
    // held to, and it is what makes a limb stay in its socket under load.
    CHECK(world.stats().maxJointAnchorError < Real(0.005));
    // It hangs below the anchor rather than floating or drifting sideways.
    CHECK(world.body(limbIndex).position.y < Real(1.75));
    CHECK(std::abs(world.body(limbIndex).position.x) < Real(0.05));
}

TEST(BallJoint3D, anchorsSurviveBeingSwungHard) {
    // A joint that holds a body at rest but fails under load is the failure mode
    // that matters, so this one starts with a large sideways velocity.
    World3D world;

    RigidBody3D anchor = makeCapsule(Vec3(0, 3, 0), Real(0.05), Real(0.05), Real(1));
    anchor.makeStatic();
    anchor.collisionGroup = 1;
    const int32_t anchorIndex = world.addBody(anchor);

    RigidBody3D limb = makeCapsule(Vec3(0, Real(2.6), 0), Real(0.06), Real(0.35), Real(6));
    limb.collisionGroup = 1;
    limb.velocity = Vec3(Real(8), 0, Real(5));
    const int32_t limbIndex = world.addBody(limb);

    BallJoint3D joint;
    joint.bodyA = anchorIndex;
    joint.bodyB = limbIndex;
    joint.localAnchorA = Vec3(0, 0, 0);
    joint.localAnchorB = Vec3(0, Real(0.35), 0);
    world.addBallJoint(joint);

    Real worst = 0;
    for (int i = 0; i < 1200; ++i) {
        world.step(kDt);
        worst = std::max(worst, world.stats().maxJointAnchorError);
    }

    CHECK(!world.stats().unstable);
    CHECK(worst < Real(0.01));
}

// ---------------------------------------------------------------- cone limit

TEST(BallJoint3D, theConeLimitStopsTheSwing) {
    World3D world;

    RigidBody3D anchor = makeCapsule(Vec3(0, 2, 0), Real(0.05), Real(0.05), Real(1));
    anchor.makeStatic();
    anchor.collisionGroup = 1;
    const int32_t anchorIndex = world.addBody(anchor);

    // Starts hanging straight down, which is a swing of pi from the anchor's +Y
    // axis, so the reference rotation puts the rest pose at zero swing.
    RigidBody3D limb = makeCapsule(Vec3(0, Real(1.7), 0), Real(0.05), Real(0.3), Real(3));
    limb.collisionGroup = 1;
    limb.velocity = Vec3(Real(6), 0, 0);
    const int32_t limbIndex = world.addBody(limb);

    BallJoint3D joint;
    joint.bodyA = anchorIndex;
    joint.bodyB = limbIndex;
    joint.localAnchorA = Vec3(0, 0, 0);
    joint.localAnchorB = Vec3(0, Real(0.3), 0);
    joint.enableConeLimit = true;
    joint.coneAngle = Real(0.4);  // about 23 degrees
    const int32_t jointIndex = world.addBallJoint(joint);

    Real worstSwing = 0;
    for (int i = 0; i < 1200; ++i) {
        world.step(kDt);
        worstSwing = std::max(worstSwing, world.ballJoint(jointIndex).swingAngle(
                                              world.body(anchorIndex), world.body(limbIndex)));
    }

    CHECK(!world.stats().unstable);
    // Some overshoot is expected from a velocity-level constraint, but a limb
    // shoved at 6 m/s must not sail through a 0.4 rad stop.
    CHECK(worstSwing < Real(0.4) + Real(0.08));
    // And it genuinely reached the limit rather than never moving.
    CHECK(worstSwing > Real(0.3));
}

TEST(BallJoint3D, aDisabledConeLimitLetsTheLimbSwingFreely) {
    // The control for the test above. Without it, a cone limit that silently
    // never releases would look identical to one that works.
    World3D world;

    RigidBody3D anchor = makeCapsule(Vec3(0, 2, 0), Real(0.05), Real(0.05), Real(1));
    anchor.makeStatic();
    anchor.collisionGroup = 1;
    const int32_t anchorIndex = world.addBody(anchor);

    RigidBody3D limb = makeCapsule(Vec3(0, Real(1.7), 0), Real(0.05), Real(0.3), Real(3));
    limb.collisionGroup = 1;
    limb.velocity = Vec3(Real(6), 0, 0);
    const int32_t limbIndex = world.addBody(limb);

    BallJoint3D joint;
    joint.bodyA = anchorIndex;
    joint.bodyB = limbIndex;
    joint.localAnchorA = Vec3(0, 0, 0);
    joint.localAnchorB = Vec3(0, Real(0.3), 0);
    joint.enableConeLimit = false;
    const int32_t jointIndex = world.addBallJoint(joint);

    Real worstSwing = 0;
    for (int i = 0; i < 1200; ++i) {
        world.step(kDt);
        worstSwing = std::max(worstSwing, world.ballJoint(jointIndex).swingAngle(
                                              world.body(anchorIndex), world.body(limbIndex)));
    }
    CHECK(worstSwing > Real(0.6));
}

// ---------------------------------------------------------------- hinge

TEST(HingeJoint3D, refusesToBendOutOfItsOwnPlane) {
    // The constraint with no 2D counterpart. A knee has to actively forbid the
    // two axes it does not turn about; without that lock it is a ball joint with
    // a limit on one axis, and the shin swings sideways under load.
    World3D world;

    RigidBody3D thigh = makeCapsule(Vec3(0, 2, 0), Real(0.06), Real(0.2), Real(8));
    thigh.makeStatic();
    thigh.collisionGroup = 1;
    const int32_t thighIndex = world.addBody(thigh);

    RigidBody3D shin = makeCapsule(Vec3(0, Real(1.6), 0), Real(0.05), Real(0.2), Real(4));
    shin.collisionGroup = 1;
    // Shoved sideways, perpendicular to the hinge axis, which is exactly the
    // direction the joint must refuse.
    shin.velocity = Vec3(0, 0, Real(7));
    const int32_t shinIndex = world.addBody(shin);

    HingeJoint3D joint;
    joint.bodyA = thighIndex;
    joint.bodyB = shinIndex;
    joint.localAnchorA = Vec3(0, Real(-0.2), 0);
    joint.localAnchorB = Vec3(0, Real(0.2), 0);
    joint.localHingeAxisA = Vec3(0, 0, 1);
    joint.localHingeAxisB = Vec3(0, 0, 1);
    const int32_t jointIndex = world.addHingeJoint(joint);

    Real worstMisalignment = 0;
    for (int i = 0; i < 1200; ++i) {
        world.step(kDt);
        worstMisalignment =
            std::max(worstMisalignment, world.hingeJoint(jointIndex).axisMisalignment(
                                            world.body(thighIndex), world.body(shinIndex)));
    }

    CHECK(!world.stats().unstable);
    // Under a degree of out-of-plane bend over five seconds of being shoved.
    CHECK(worstMisalignment < Real(0.02));
    CHECK(world.stats().maxJointAnchorError < Real(0.005));
}

TEST(HingeJoint3D, stillTurnsAboutItsOwnAxis) {
    // The control: a hinge that locked all three axes would pass the test above
    // perfectly and be useless.
    World3D world;

    RigidBody3D thigh = makeCapsule(Vec3(0, 2, 0), Real(0.06), Real(0.2), Real(8));
    thigh.makeStatic();
    thigh.collisionGroup = 1;
    const int32_t thighIndex = world.addBody(thigh);

    RigidBody3D shin = makeCapsule(Vec3(0, Real(1.6), 0), Real(0.05), Real(0.2), Real(4));
    shin.collisionGroup = 1;
    shin.velocity = Vec3(Real(5), 0, 0);  // in the hinge's plane this time
    const int32_t shinIndex = world.addBody(shin);

    HingeJoint3D joint;
    joint.bodyA = thighIndex;
    joint.bodyB = shinIndex;
    joint.localAnchorA = Vec3(0, Real(-0.2), 0);
    joint.localAnchorB = Vec3(0, Real(0.2), 0);
    joint.localHingeAxisA = Vec3(0, 0, 1);
    joint.localHingeAxisB = Vec3(0, 0, 1);
    const int32_t jointIndex = world.addHingeJoint(joint);

    Real widest = 0;
    for (int i = 0; i < 600; ++i) {
        world.step(kDt);
        widest = std::max(widest, std::abs(world.hingeJoint(jointIndex).jointAngle(
                                      world.body(thighIndex), world.body(shinIndex))));
    }
    CHECK(widest > Real(0.5));
}

TEST(HingeJoint3D, respectsItsAngularLimits) {
    World3D world;

    RigidBody3D thigh = makeCapsule(Vec3(0, 2, 0), Real(0.06), Real(0.2), Real(8));
    thigh.makeStatic();
    thigh.collisionGroup = 1;
    const int32_t thighIndex = world.addBody(thigh);

    RigidBody3D shin = makeCapsule(Vec3(0, Real(1.6), 0), Real(0.05), Real(0.2), Real(4));
    shin.collisionGroup = 1;
    shin.velocity = Vec3(Real(6), 0, 0);
    const int32_t shinIndex = world.addBody(shin);

    HingeJoint3D joint;
    joint.bodyA = thighIndex;
    joint.bodyB = shinIndex;
    joint.localAnchorA = Vec3(0, Real(-0.2), 0);
    joint.localAnchorB = Vec3(0, Real(0.2), 0);
    joint.localHingeAxisA = Vec3(0, 0, 1);
    joint.localHingeAxisB = Vec3(0, 0, 1);
    joint.enableLimit = true;
    joint.lowerAngle = Real(-0.3);
    joint.upperAngle = Real(0.3);
    const int32_t jointIndex = world.addHingeJoint(joint);

    Real worst = 0;
    for (int i = 0; i < 1200; ++i) {
        world.step(kDt);
        const Real angle = world.hingeJoint(jointIndex).jointAngle(world.body(thighIndex),
                                                                  world.body(shinIndex));
        worst = std::max(worst, std::abs(angle));
    }

    CHECK(!world.stats().unstable);
    CHECK(worst < Real(0.3) + Real(0.08));
    CHECK(worst > Real(0.2));
}

// ---------------------------------------------------------------- motors

TEST(HingeJoint3D, aMotorHoldsItsTargetAgainstGravity) {
    World3D world;

    RigidBody3D root = makeCapsule(Vec3(0, 2, 0), Real(0.06), Real(0.2), Real(8));
    root.makeStatic();
    root.collisionGroup = 1;
    const int32_t rootIndex = world.addBody(root);

    RigidBody3D limb = makeCapsule(Vec3(0, Real(1.6), 0), Real(0.05), Real(0.2), Real(4));
    limb.collisionGroup = 1;
    const int32_t limbIndex = world.addBody(limb);

    HingeJoint3D joint;
    joint.bodyA = rootIndex;
    joint.bodyB = limbIndex;
    joint.localAnchorA = Vec3(0, Real(-0.2), 0);
    joint.localAnchorB = Vec3(0, Real(0.2), 0);
    joint.localHingeAxisA = Vec3(0, 0, 1);
    joint.localHingeAxisB = Vec3(0, 0, 1);
    joint.enableMotor = true;
    joint.targetAngle = Real(0.6);
    joint.stiffness = 400;
    joint.damping = 40;
    joint.maxTorque = 200;
    const int32_t jointIndex = world.addHingeJoint(joint);

    run(world, Real(4));

    const Real reached = world.hingeJoint(jointIndex).jointAngle(world.body(rootIndex),
                                                                world.body(limbIndex));
    CHECK(!world.stats().unstable);
    CHECK_NEAR(reached, 0.6, 0.08);
}

TEST(BallJoint3D, aMotorDrivesTheRelativeOrientation) {
    // The 3D motor's error is a rotation vector rather than a scalar, so this
    // asks it for a target that is not about any single axis.
    World3D world;

    RigidBody3D root = makeCapsule(Vec3(0, 2, 0), Real(0.06), Real(0.1), Real(20));
    root.makeStatic();
    root.collisionGroup = 1;
    const int32_t rootIndex = world.addBody(root);

    RigidBody3D limb = makeCapsule(Vec3(0, Real(1.7), 0), Real(0.05), Real(0.2), Real(3));
    limb.collisionGroup = 1;
    const int32_t limbIndex = world.addBody(limb);

    BallJoint3D joint;
    joint.bodyA = rootIndex;
    joint.bodyB = limbIndex;
    joint.localAnchorA = Vec3(0, Real(-0.1), 0);
    joint.localAnchorB = Vec3(0, Real(0.2), 0);
    joint.enableMotor = true;
    joint.targetRotation = normalize(Quat::fromAxisAngle(normalize(Vec3(1, Real(0.4), Real(0.7))),
                                                         Real(0.5)));
    joint.stiffness = 600;
    joint.damping = 60;
    joint.maxTorque = 400;
    const int32_t jointIndex = world.addBallJoint(joint);

    run(world, Real(5));

    const Quat current = world.ballJoint(jointIndex).relativeRotation(world.body(rootIndex),
                                                                     world.body(limbIndex));
    const Quat error = conjugate(joint.targetRotation) * current;
    CHECK(!world.stats().unstable);
    CHECK(length(rotationVector(error)) < Real(0.12));
}

TEST(BallJoint3D, aMotorRespectsItsTorqueCeiling) {
    // With a ceiling far below what the load needs, the motor must fall short
    // rather than quietly exceeding the limit the config asked for.
    World3D world;

    RigidBody3D root = makeCapsule(Vec3(0, 2, 0), Real(0.06), Real(0.1), Real(50));
    root.makeStatic();
    root.collisionGroup = 1;
    const int32_t rootIndex = world.addBody(root);

    RigidBody3D limb = makeCapsule(Vec3(0, Real(1.5), 0), Real(0.05), Real(0.4), Real(25));
    limb.collisionGroup = 1;
    const int32_t limbIndex = world.addBody(limb);

    BallJoint3D joint;
    joint.bodyA = rootIndex;
    joint.bodyB = limbIndex;
    joint.localAnchorA = Vec3(0, Real(-0.1), 0);
    joint.localAnchorB = Vec3(0, Real(0.4), 0);
    joint.enableMotor = true;
    // Straight out sideways, which gravity fights the whole way.
    joint.targetRotation = Quat::fromAxisAngle(Vec3(0, 0, 1), Real(1.4));
    joint.stiffness = 4000;
    joint.damping = 400;
    joint.maxTorque = Real(0.5);
    const int32_t jointIndex = world.addBallJoint(joint);

    run(world, Real(3));

    const Quat current = world.ballJoint(jointIndex).relativeRotation(world.body(rootIndex),
                                                                     world.body(limbIndex));
    const Quat error = conjugate(joint.targetRotation) * current;
    CHECK(!world.stats().unstable);
    CHECK(length(rotationVector(error)) > Real(0.5));
}

// ---------------------------------------------------------------- chains

TEST(World3D, aJointedChainCollapsesWithoutComingApart) {
    // The 3D analogue of the ragdoll check: five links, no motors, dropped.
    // Anchor drift is the number that says whether the solver held together.
    World3D world;
    world.addHalfSpace(HalfSpace3D{Vec3(0, 1, 0), Real(0), Real(0.9), Real(0)});

    std::vector<int32_t> links;
    for (int i = 0; i < 5; ++i) {
        RigidBody3D link = makeCapsule(Vec3(0, Real(2.5) - Real(0.3) * i, 0),
                                       Real(0.05), Real(0.13), Real(3));
        link.collisionGroup = 1;
        links.push_back(world.addBody(link));
    }

    for (int i = 0; i + 1 < 5; ++i) {
        BallJoint3D joint;
        joint.bodyA = links[i];
        joint.bodyB = links[i + 1];
        joint.localAnchorA = Vec3(0, Real(-0.15), 0);
        joint.localAnchorB = Vec3(0, Real(0.15), 0);
        world.addBallJoint(joint);
    }

    Real worst = 0;
    for (int i = 0; i < 1440; ++i) {
        world.step(kDt);
        worst = std::max(worst, world.stats().maxJointAnchorError);
    }

    CHECK(!world.stats().unstable);
    CHECK(worst < Real(0.01));
    // It fell and settled rather than sinking through the floor or exploding.
    for (int32_t index : links) {
        CHECK(world.body(index).position.y > Real(-0.1));
        CHECK(world.body(index).position.y < Real(2.6));
    }
}

TEST(World3D, aCapsuleRestsOnTheGroundWithoutSinkingOrBuzzing) {
    World3D world;
    world.addHalfSpace(HalfSpace3D{Vec3(0, 1, 0), Real(0), Real(0.9), Real(0)});

    RigidBody3D body = makeCapsule(Vec3(0, Real(0.5), 0), Real(0.1), Real(0.25), Real(10));
    body.orientation = Quat::fromAxisAngle(Vec3(0, 0, 1), kPi * Real(0.5));
    const int32_t index = world.addBody(body);

    run(world, Real(4));

    CHECK(!world.stats().unstable);
    // Resting on its side, so the centre sits one radius above the ground.
    CHECK_NEAR(world.body(index).position.y, 0.1, 0.01);
    CHECK(length(world.body(index).velocity) < Real(0.05));
    CHECK(world.stats().maxPenetration < kLinearSlop3D * Real(3));
    CHECK(world.hasContact(index));
}

namespace {

// Two jointed capsules tumbling with no gravity and no contacts, run for three
// simulated seconds. Returns the percentage change in the system's total
// angular momentum about its own centre of mass.
//
// The anchors are placed coincident on purpose. An earlier version of this rig
// put them 1 cm apart, so the joint yanked them together in the first few steps
// and the resulting 2.5% jump looked exactly like a solver leak.
Real momentumDriftPercent(Real hz) {
    World3D world;
    world.gravity = Vec3(0, 0, 0);

    RigidBody3D a = makeCapsule(Vec3(0, 0, 0), Real(0.06), Real(0.2), Real(4));
    a.collisionGroup = 1;
    a.angularVelocity = Vec3(Real(2), Real(5), Real(1));
    const int32_t indexA = world.addBody(a);

    RigidBody3D b = makeCapsule(Vec3(0, Real(-0.44), 0), Real(0.05), Real(0.2), Real(3));
    b.collisionGroup = 1;
    b.velocity = Vec3(Real(1.5), 0, Real(-0.7));
    const int32_t indexB = world.addBody(b);

    BallJoint3D joint;
    joint.bodyA = indexA;
    joint.bodyB = indexB;
    joint.localAnchorA = Vec3(0, Real(-0.22), 0);
    joint.localAnchorB = Vec3(0, Real(0.22), 0);
    world.addBallJoint(joint);

    const Vec3 initial = world.angularMomentum();
    for (int i = 0; i < int(hz * Real(3)); ++i) world.step(Real(1) / hz);
    const Vec3 final = world.angularMomentum();

    return length(final - initial) / length(initial) * Real(100);
}

}  // namespace

TEST(World3D, angularMomentumDriftAcrossAJointIsTruncationErrorAndNotALeak) {
    // A sequential-impulse solver does not conserve angular momentum exactly for
    // a constrained system, and this one does not either: a jointed tumbling
    // pair loses 3.6% over three seconds at the rate the engine runs.
    //
    // The number on its own says nothing about whether that is a bug. What
    // settles it is how the number responds to the things that would change it:
    //
    //   velocity iterations   5 -> 80      3.567% -> 3.568%   (no effect)
    //   position iterations   4 -> 0       3.567% -> 3.695%   (barely any)
    //   step rate             120 -> 1920  6.77% -> 0.49%     (halves with dt)
    //
    // Flat in both iteration counts and cleanly first order in dt. That is
    // truncation error in the integrator, converging to zero as the step
    // shrinks, rather than a constraint quietly destroying momentum. A leak
    // would have shown up as a drift the solver could not iterate away and that
    // a smaller step did not touch.
    const Real coarse = momentumDriftPercent(Real(240));
    const Real fine = momentumDriftPercent(Real(480));
    const Real finer = momentumDriftPercent(Real(960));

    // Bounded at the rate the engine actually runs. Over a one-second backflip
    // that is a bit over a percent, which is well under the noise a policy sees.
    CHECK(coarse < Real(5));

    // Each halving of the step halves the error, within a generous band. The
    // ratio is the assertion; the absolute number above is context.
    CHECK(fine < coarse * Real(0.65));
    CHECK(finer < fine * Real(0.65));
    CHECK(fine > coarse * Real(0.35));
    CHECK(finer > fine * Real(0.35));
}
