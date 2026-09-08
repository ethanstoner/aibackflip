#include <cmath>

#include "core/Test.h"
#include "physics/World2D.h"

using namespace aibf;

namespace {

constexpr Real kDt = Real(1) / Real(240);

// Jointed bodies necessarily overlap at the pivot, so everything in an
// articulated assembly shares a collision group and never collides with itself.
// The humanoid does exactly the same thing.
constexpr int32_t kFigureGroup = 1;

RigidBody2D limb(Vec2 position, Real angle, Real radius, Real halfLength, Real density) {
    RigidBody2D b;
    b.position = position;
    b.angle = angle;
    b.setCapsule(radius, halfLength, density);
    b.collisionGroup = kFigureGroup;
    return b;
}

int32_t addAnchor(World2D& world, Vec2 at) {
    RigidBody2D a;
    a.position = at;
    a.radius = Real(0.02);
    a.halfLength = 0;
    a.makeStatic();
    a.collisionGroup = kFigureGroup;
    return world.addBody(a);
}

// A rod pinned by its upper end to a static anchor, started horizontal so it
// swings. `density` is deliberately light so the motor tests need believable
// torques rather than thousands of newton-metres.
struct Pendulum {
    World2D world;
    int32_t anchor = -1;
    int32_t rod = -1;
    int32_t joint = -1;
    Real halfLength = Real(0.5);

    explicit Pendulum(Real startAngle = kHalfPi, Real density = Real(100)) {
        world.gravity = Vec2(0, Real(-9.81));
        const Vec2 pivot(0, 2);
        anchor = addAnchor(world, pivot);

        // Place the rod so its local (0, +halfLength) lands exactly on the pivot.
        const Vec2 offset = rotate(Vec2(0, halfLength), startAngle);
        rod = world.addBody(limb(pivot - offset, startAngle, Real(0.05), halfLength, density));

        RevoluteJoint2D j;
        j.bodyA = anchor;
        j.bodyB = rod;
        j.localAnchorA = Vec2(0, 0);
        j.localAnchorB = Vec2(0, halfLength);
        j.referenceAngle = 0;
        joint = world.addJoint(j);
    }

    void run(Real seconds) {
        const int steps = static_cast<int>(std::lround(seconds / kDt));
        for (int i = 0; i < steps; ++i) world.step(kDt);
    }
    Real angle() const { return world.body(rod).angle; }
    Real anchorError() const {
        return world.joint(joint).anchorError(world.body(anchor), world.body(rod));
    }
    Real totalEnergy() const { return world.kineticEnergy() + world.potentialEnergy(); }
};

}  // namespace

// ---------------------------------------------------------------- anchors

TEST(Joint, aPendulumKeepsItsAnchorTogether) {
    Pendulum p;
    p.run(Real(5));
    CHECK(p.anchorError() < Real(1e-4));
    CHECK(p.world.stats().maxJointAnchorError < Real(1e-4));
    CHECK(!p.world.stats().unstable);
}

TEST(Joint, aPendulumActuallySwings) {
    // Guards against a joint so over-constrained that it simply freezes, which
    // would pass the anchor-error test while being useless.
    Pendulum p;
    const Real start = p.angle();
    p.run(Real(0.5));
    CHECK(std::abs(p.angle() - start) > Real(0.3));
}

TEST(Joint, aPendulumRoughlyConservesEnergy) {
    // An impulse solver is not symplectic, so some drift is expected. What must
    // not happen is systematic gain: a joint that adds energy turns into a
    // motor and the humanoid launches itself.
    Pendulum p;
    const Real start = p.totalEnergy();
    p.run(Real(4));
    const Real finish = p.totalEnergy();
    CHECK_NEAR(finish, start, std::abs(start) * Real(0.08));
}

TEST(Joint, aHangingChainStaysConnectedUnderLoad) {
    // Five links is more than the humanoid's longest chain (hip-knee-ankle-foot),
    // so if this holds the body will.
    World2D world;
    const Vec2 pivot(0, 3);
    const int32_t anchor = addAnchor(world, pivot);

    const Real half = Real(0.25);
    int32_t previous = anchor;
    Vec2 attachAt = pivot;
    Vec2 previousAnchor(0, 0);

    for (int i = 0; i < 5; ++i) {
        const int32_t link =
            world.addBody(limb(attachAt - Vec2(0, half), 0, Real(0.05), half, Real(400)));
        RevoluteJoint2D j;
        j.bodyA = previous;
        j.bodyB = link;
        j.localAnchorA = previousAnchor;
        j.localAnchorB = Vec2(0, half);
        world.addJoint(j);

        previous = link;
        previousAnchor = Vec2(0, -half);
        attachAt = attachAt - Vec2(0, 2 * half);
    }

    // Yank the bottom link sideways to load every joint at once.
    world.body(previous).velocity = Vec2(12, 0);
    for (int i = 0; i < 240 * 4; ++i) world.step(kDt);

    CHECK(world.stats().maxJointAnchorError < Real(2e-3));
    CHECK(!world.stats().unstable);
}

TEST(Joint, theBaumgarteVelocityBiasAlsoHoldsTheAnchor) {
    // Both stabilisation strategies are implemented; this pins that the
    // non-default one is not quietly broken. The position solver is expected to
    // be the tighter of the two - see docs/PROGRESS.md for the measured gap.
    Pendulum p;
    p.world.solver.useJointPositionSolver = false;
    p.world.solver.positionIterations = 0;
    p.run(Real(5));
    CHECK(!p.world.stats().unstable);
    CHECK(p.anchorError() < Real(5e-3));
}

TEST(Joint, aJointBetweenTwoDynamicBodiesConservesMomentum) {
    World2D world;
    world.gravity = Vec2(0, 0);
    const int32_t a = world.addBody(limb(Vec2(0, 0), 0, Real(0.05), Real(0.25), Real(400)));
    const int32_t b = world.addBody(limb(Vec2(0, Real(-0.5)), 0, Real(0.05), Real(0.25), Real(400)));

    RevoluteJoint2D j;
    j.bodyA = a;
    j.bodyB = b;
    j.localAnchorA = Vec2(0, Real(-0.25));
    j.localAnchorB = Vec2(0, Real(0.25));
    world.addJoint(j);

    world.body(a).velocity = Vec2(2, 0);
    world.body(b).velocity = Vec2(-2, 0);
    const Vec2 momentum0 =
        world.body(a).velocity * world.body(a).mass + world.body(b).velocity * world.body(b).mass;

    for (int i = 0; i < 240 * 2; ++i) world.step(kDt);

    const Vec2 momentum1 =
        world.body(a).velocity * world.body(a).mass + world.body(b).velocity * world.body(b).mass;
    CHECK_NEAR(length(momentum1 - momentum0), 0.0, 1e-2);
}

// ---------------------------------------------------------------- limits

TEST(JointLimit, aSwingingRodStopsAtItsLimit) {
    Pendulum p(kHalfPi);
    RevoluteJoint2D& j = p.world.joint(p.joint);
    j.enableLimit = true;
    j.lowerAngle = Real(0.3);
    j.upperAngle = Real(2.0);

    p.run(Real(6));

    const Real relative = j.relativeAngle(p.world.body(p.anchor), p.world.body(p.rod));
    CHECK(relative > Real(0.3) - Real(0.02));
    CHECK(relative < Real(2.0) + Real(0.02));
}

TEST(JointLimit, aLimitedJointNeverEscapesUnderHardImpact) {
    // Starts hanging at relative angle 0, i.e. inside the limits. Starting
    // outside them would just measure how fast the solver drags the joint back
    // into range, which is a different question.
    Pendulum p(0, Real(400));
    RevoluteJoint2D& j = p.world.joint(p.joint);
    j.enableLimit = true;
    j.lowerAngle = Real(-0.2);
    j.upperAngle = Real(0.2);

    Real worstViolation = 0;
    for (int i = 0; i < 240 * 5; ++i) {
        if (i % 200 == 0) p.world.body(p.rod).angularVelocity += Real(60);
        p.world.step(kDt);
        const Real relative = j.relativeAngle(p.world.body(p.anchor), p.world.body(p.rod));
        worstViolation = std::max(worstViolation,
                                  std::max(relative - Real(0.2), Real(-0.2) - relative));
    }
    CHECK(!p.world.stats().unstable);
    CHECK_NEAR(worstViolation, 0.0, 0.05);
}

TEST(JointLimit, anUnlimitedJointRotatesFreelyPastPi) {
    // Relative angles must accumulate rather than wrap, or a limb spinning
    // through a backflip would appear to jump between +pi and -pi.
    Pendulum p(0);
    p.world.gravity = Vec2(0, 0);
    p.world.body(p.rod).angularVelocity = Real(20);
    p.run(Real(1));
    const Real relative =
        p.world.joint(p.joint).relativeAngle(p.world.body(p.anchor), p.world.body(p.rod));
    CHECK(std::abs(relative) > kPi);
}

// ---------------------------------------------------------------- motor

TEST(JointMotor, holdsAPoseAgainstGravity) {
    Pendulum p(0);  // starts hanging straight down
    RevoluteJoint2D& j = p.world.joint(p.joint);
    j.enableMotor = true;
    j.targetAngle = Real(1.0);
    j.stiffness = Real(2000);
    j.damping = Real(200);
    j.maxTorque = Real(400);

    p.run(Real(4));

    const Real relative = j.relativeAngle(p.world.body(p.anchor), p.world.body(p.rod));
    // A proportional controller holding a load settles slightly short of target;
    // the residual is (gravity torque)/kp, a couple of hundredths of a radian.
    CHECK_NEAR(relative, 1.0, 0.06);
    CHECK(!p.world.stats().unstable);
}

TEST(JointMotor, appliesTorqueRatherThanTeleporting) {
    // One substep must not snap the limb to the target. If it does, the motor is
    // acting as a position constraint and the physics is decorative.
    Pendulum p(0);
    RevoluteJoint2D& j = p.world.joint(p.joint);
    j.enableMotor = true;
    j.targetAngle = Real(1.2);
    j.stiffness = Real(2000);
    j.damping = Real(200);
    j.maxTorque = Real(400);

    p.world.step(kDt);
    const Real afterOneStep = j.relativeAngle(p.world.body(p.anchor), p.world.body(p.rod));
    CHECK(std::abs(afterOneStep) < Real(0.05));
    CHECK(std::abs(p.world.body(p.rod).angularVelocity) > Real(0));
}

TEST(JointMotor, respectsItsTorqueCeiling) {
    Pendulum p(0, Real(400));
    RevoluteJoint2D& j = p.world.joint(p.joint);
    j.enableMotor = true;
    j.targetAngle = Real(1.5);
    j.stiffness = Real(5000);
    j.damping = Real(500);
    j.maxTorque = Real(2);  // nowhere near enough to lift the rod

    p.run(Real(4));

    const Real relative = j.relativeAngle(p.world.body(p.anchor), p.world.body(p.rod));
    CHECK(std::abs(relative) < Real(0.2));
}

TEST(JointMotor, staysStableAtStiffnessThatWouldBlowUpAnExplicitPD) {
    // With kp*dt^2 well above the link's inertia about the pivot, an explicit
    // `torque = kp*error - kd*rate` diverges within a few steps. The implicit
    // soft-constraint form has to survive it.
    Pendulum p(0, Real(400));
    RevoluteJoint2D& j = p.world.joint(p.joint);
    j.enableMotor = true;
    j.targetAngle = Real(0.8);
    j.stiffness = Real(500000);
    j.damping = Real(5000);
    j.maxTorque = Real(20000);

    p.run(Real(3));

    CHECK(!p.world.stats().unstable);
    CHECK(p.world.body(p.rod).isFinite());
    CHECK_NEAR(j.relativeAngle(p.world.body(p.anchor), p.world.body(p.rod)), 0.8, 0.05);
    CHECK(p.world.stats().velocityClampEvents == 0);
}

TEST(JointMotor, trackingAMovingTargetDoesNotAccumulateError) {
    Pendulum p(0, Real(100));
    RevoluteJoint2D& j = p.world.joint(p.joint);
    j.enableMotor = true;
    j.stiffness = Real(4000);
    j.damping = Real(300);
    j.maxTorque = Real(800);

    Real worst = 0;
    for (int i = 0; i < 240 * 4; ++i) {
        const Real t = static_cast<Real>(i) * kDt;
        j.targetAngle = Real(0.6) * std::sin(t * Real(2));
        p.world.step(kDt);
        if (t > Real(1)) {  // ignore the initial transient
            const Real relative = j.relativeAngle(p.world.body(p.anchor), p.world.body(p.rod));
            worst = std::max(worst, std::abs(relative - j.targetAngle));
        }
    }
    CHECK(worst < Real(0.25));
    CHECK(!p.world.stats().unstable);
}

TEST(JointMotor, aDisabledMotorAppliesNothing) {
    Pendulum p(0);
    p.world.joint(p.joint).enableMotor = false;
    p.world.joint(p.joint).targetAngle = Real(1.5);
    p.run(Real(2));
    CHECK(std::abs(p.angle()) < Real(0.05));
}
