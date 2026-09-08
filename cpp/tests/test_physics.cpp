#include <cmath>

#include "core/Test.h"
#include "physics/World2D.h"

using namespace aibf;

namespace {

constexpr Real kDt = Real(1) / Real(240);

RigidBody2D capsule(Vec2 position, Real radius, Real halfLength, Real density = Real(1000)) {
    RigidBody2D b;
    b.position = position;
    b.setCapsule(radius, halfLength, density);
    return b;
}

void addGround(World2D& world, Real friction = Real(0.9), Real restitution = 0) {
    world.addHalfSpace(HalfSpace{Vec2(0, 1), 0, friction, restitution});
}

// Rounds rather than truncates: kDt is not exactly representable, so
// int(1.0 / kDt) lands on 239 and every duration would be one step short.
void run(World2D& world, Real seconds) {
    const int steps = static_cast<int>(std::lround(seconds / kDt));
    for (int i = 0; i < steps; ++i) world.step(kDt);
}

}  // namespace

// ---------------------------------------------------------------- mass properties

TEST(MassProperties, aZeroLengthCapsuleIsADisc) {
    const Real r = Real(0.3), density = Real(1000);
    MassProperties mp = capsuleMassProperties(r, 0, density);
    const Real expectedMass = density * kPi * r * r;
    CHECK_NEAR(mp.mass, expectedMass, expectedMass * 1e-5);
    CHECK_NEAR(mp.inertia, expectedMass * r * r * Real(0.5), expectedMass * r * r * 1e-5);
}

TEST(MassProperties, aVanishinglyThinCapsuleIsARod) {
    // I = m*L^2/12 for a rod of length L = 2h.
    const Real r = Real(1e-4), h = Real(0.5), density = Real(1000);
    MassProperties mp = capsuleMassProperties(r, h, density);
    const Real length = Real(2) * h;
    CHECK_NEAR(mp.inertia, mp.mass * length * length / Real(12), mp.mass * length * length * 1e-3);
}

TEST(MassProperties, massScalesWithDensityAndInertiaIsPositive) {
    MassProperties a = capsuleMassProperties(Real(0.08), Real(0.2), Real(1000));
    MassProperties b = capsuleMassProperties(Real(0.08), Real(0.2), Real(2000));
    CHECK_NEAR(b.mass, a.mass * 2, a.mass * 1e-4);
    CHECK_NEAR(b.inertia, a.inertia * 2, a.inertia * 1e-4);
    CHECK(a.inertia > 0);
}

TEST(MassProperties, longerLimbsHaveMoreInertia) {
    MassProperties shortLimb = capsuleMassProperties(Real(0.05), Real(0.1), Real(1000));
    MassProperties longLimb = capsuleMassProperties(Real(0.05), Real(0.4), Real(1000));
    CHECK(longLimb.inertia > shortLimb.inertia);
}

// ---------------------------------------------------------------- integration

TEST(Integration, freeFallMatchesSemiImplicitEulerExactly) {
    // Semi-implicit Euler updates velocity first, so after N steps
    //   v = N*g*dt   and   y = y0 + g*dt^2 * N(N+1)/2,
    // which is the analytic g*T^2/2 plus a known O(dt) offset. Pinning the exact
    // discrete result rather than the continuous one means this test fails if
    // the integration order is ever silently changed.
    World2D world;
    const int32_t id = world.addBody(capsule(Vec2(0, 10), Real(0.1), 0));

    const int steps = 240;
    for (int i = 0; i < steps; ++i) world.step(kDt);

    const Real g = world.gravity.y;
    const Real expectedV = static_cast<Real>(steps) * g * kDt;
    const Real expectedY = Real(10) + g * kDt * kDt * Real(steps) * Real(steps + 1) / Real(2);

    CHECK_NEAR(world.body(id).velocity.y, expectedV, 1e-4);
    CHECK_NEAR(world.body(id).position.y, expectedY, 1e-3);
}

TEST(Integration, horizontalMotionIsUnaffectedByGravity) {
    World2D world;
    const int32_t id = world.addBody(capsule(Vec2(0, 10), Real(0.1), 0));
    world.body(id).velocity = Vec2(3, 0);
    run(world, Real(1));
    CHECK_NEAR(world.body(id).velocity.x, 3.0, 1e-5);
    CHECK_NEAR(world.body(id).position.x, 3.0, 1e-3);
}

TEST(Integration, aStaticBodyNeverMoves) {
    World2D world;
    RigidBody2D b = capsule(Vec2(1, 2), Real(0.2), Real(0.3));
    b.makeStatic();
    const int32_t id = world.addBody(b);
    run(world, Real(2));
    CHECK_VEC2_NEAR(world.body(id).position, 1.0, 2.0, 1e-9);
    CHECK_NEAR(world.body(id).velocity.y, 0.0, 1e-9);
}

TEST(Integration, aFreeSpinningBodyConservesAngularVelocityAndEnergy) {
    World2D world;
    world.gravity = Vec2(0, 0);
    const int32_t id = world.addBody(capsule(Vec2(0, 0), Real(0.1), Real(0.5)));
    world.body(id).angularVelocity = Real(7);
    world.body(id).velocity = Vec2(Real(2), Real(-1));
    const Real energy0 = world.kineticEnergy();

    run(world, Real(5));

    CHECK_NEAR(world.body(id).angularVelocity, 7.0, 1e-6);
    CHECK_NEAR(world.kineticEnergy(), energy0, energy0 * 1e-5);
}

TEST(Integration, dampingBleedsOffVelocityAndStaysStableAtAnyTimestep) {
    World2D world;
    world.gravity = Vec2(0, 0);
    const int32_t id = world.addBody(capsule(Vec2(0, 0), Real(0.1), 0));
    world.body(id).linearDamping = Real(2);
    world.body(id).velocity = Vec2(10, 0);

    run(world, Real(3));
    CHECK(world.body(id).velocity.x < Real(1));
    CHECK(world.body(id).velocity.x > Real(0));  // implicit damping never overshoots into reverse

    // A timestep far larger than 1/damping would make explicit damping diverge.
    World2D coarse;
    coarse.gravity = Vec2(0, 0);
    const int32_t cid = coarse.addBody(capsule(Vec2(0, 0), Real(0.1), 0));
    coarse.body(cid).linearDamping = Real(50);
    coarse.body(cid).velocity = Vec2(10, 0);
    for (int i = 0; i < 20; ++i) coarse.step(Real(0.1));
    CHECK(std::abs(coarse.body(cid).velocity.x) < Real(10));
    CHECK(coarse.body(cid).velocity.x > 0);
}

// ---------------------------------------------------------------- impulses

TEST(Impulse, aCentralImpulseChangesOnlyLinearVelocity) {
    RigidBody2D b = capsule(Vec2(0, 0), Real(0.1), Real(0.5));
    b.applyImpulse(Vec2(5, 0), Vec2(0, 0));
    CHECK_NEAR(b.velocity.x, 5.0 / b.mass, 1e-5);
    CHECK_NEAR(b.angularVelocity, 0.0, 1e-9);
}

TEST(Impulse, anOffsetImpulseSpinsTheBodyByCrossOverInertia) {
    RigidBody2D b = capsule(Vec2(0, 0), Real(0.1), Real(0.5));
    const Vec2 offset(0, Real(0.4));
    const Vec2 impulse(Real(3), 0);
    b.applyImpulse(impulse, offset);
    CHECK_NEAR(b.velocity.x, 3.0 / b.mass, 1e-5);
    CHECK_NEAR(b.angularVelocity, cross(offset, impulse) / b.inertia, 1e-5);
    CHECK(b.angularVelocity < 0);  // pushing the top to the right rotates clockwise
}

TEST(Impulse, equalAndOppositeImpulsesConserveMomentum) {
    RigidBody2D a = capsule(Vec2(0, 0), Real(0.1), Real(0.3));
    RigidBody2D b = capsule(Vec2(1, 0), Real(0.2), Real(0.1));
    const Vec2 impulse(Real(2), Real(-1));
    const Vec2 point(Real(0.5), Real(0.1));
    a.applyImpulseAtPoint(-impulse, point);
    b.applyImpulseAtPoint(impulse, point);
    const Vec2 momentum = a.velocity * a.mass + b.velocity * b.mass;
    CHECK_VEC2_NEAR(momentum, 0.0, 0.0, 1e-4);
}

TEST(Impulse, velocityAtOffsetCombinesLinearAndAngularTerms) {
    RigidBody2D b = capsule(Vec2(0, 0), Real(0.1), Real(0.5));
    b.velocity = Vec2(1, 2);
    b.angularVelocity = Real(3);
    const Vec2 r(Real(0.5), 0);
    // omega x r for r along +x points along +y with magnitude omega*|r|
    CHECK_VEC2_NEAR(b.velocityAtOffset(r), 1.0, 2.0 + 1.5, 1e-5);
}

// ---------------------------------------------------------------- contacts

TEST(Contact, aDroppedDiscComesToRestOnTheSurface) {
    World2D world;
    addGround(world);
    const Real radius = Real(0.2);
    const int32_t id = world.addBody(capsule(Vec2(0, 3), radius, 0));

    run(world, Real(4));

    CHECK_NEAR(world.body(id).position.y, radius, 2 * kLinearSlop);
    CHECK_NEAR(world.body(id).velocity.y, 0.0, 1e-2);
    CHECK(world.stats().maxPenetration < 2 * kLinearSlop);
    CHECK(!world.stats().unstable);
}

TEST(Contact, restingBodiesDoNotSinkOverTime) {
    // The failure this guards against is gradual: a body that loses a fraction
    // of a millimetre per second looks fine for a second and is buried after a
    // minute of training.
    World2D world;
    addGround(world);
    const Real radius = Real(0.15);
    const int32_t id = world.addBody(capsule(Vec2(0, radius), radius, 0));

    run(world, Real(1));
    const Real earlyY = world.body(id).position.y;
    run(world, Real(10));
    const Real lateY = world.body(id).position.y;

    CHECK_NEAR(lateY, earlyY, 1e-4);
    CHECK(world.stats().maxPenetration < 2 * kLinearSlop);
}

TEST(Contact, aFlatCapsuleRestsLevelInsteadOfRocking) {
    World2D world;
    addGround(world);
    RigidBody2D c = capsule(Vec2(0, Real(0.4)), Real(0.08), Real(0.5));
    c.angle = kHalfPi;
    const int32_t id = world.addBody(c);

    run(world, Real(3));

    CHECK_NEAR(world.body(id).position.y, 0.08, 2 * kLinearSlop);
    CHECK_NEAR(std::abs(wrapAngle(world.body(id).angle - kHalfPi)), 0.0, 0.02);
    CHECK_NEAR(world.body(id).angularVelocity, 0.0, 0.05);
}

TEST(Contact, restitutionZeroDoesNotBounce) {
    World2D world;
    addGround(world, Real(0.9), Real(0));
    const int32_t id = world.addBody(capsule(Vec2(0, 2), Real(0.2), 0));

    Real highestAfterLanding = 0;
    bool landed = false;
    for (int i = 0; i < 240 * 4; ++i) {
        world.step(kDt);
        if (world.hasContact(id)) landed = true;
        if (landed) highestAfterLanding = std::max(highestAfterLanding, world.body(id).position.y);
    }
    CHECK(landed);
    CHECK(highestAfterLanding < Real(0.2) + Real(0.02));
}

TEST(Contact, restitutionProducesAProportionalBounce) {
    // A perfectly elastic bounce returns to e^2 of the drop height. Discrete
    // solvers lose some, so this checks the bounce is clearly present and not
    // above the physical ceiling.
    World2D world;
    addGround(world, Real(0.9), Real(0.8));
    const Real radius = Real(0.1);
    const Real dropHeight = Real(2);
    const int32_t id = world.addBody(capsule(Vec2(0, dropHeight), radius, 0));

    bool landed = false;
    Real apex = 0;
    for (int i = 0; i < 240 * 3; ++i) {
        world.step(kDt);
        if (!landed && world.hasContact(id)) landed = true;
        if (landed && !world.hasContact(id)) apex = std::max(apex, world.body(id).position.y);
    }
    const Real fallDistance = dropHeight - radius;
    const Real idealRebound = Real(0.8) * Real(0.8) * fallDistance;
    CHECK(landed);
    CHECK(apex - radius > idealRebound * Real(0.6));   // clearly bouncing
    CHECK(apex - radius < idealRebound * Real(1.15));  // and not gaining energy
}

TEST(Contact, frictionBringsASlidingBodyToRest) {
    World2D world;
    addGround(world, Real(0.8));
    RigidBody2D c = capsule(Vec2(0, Real(0.08)), Real(0.08), Real(0.4));
    c.angle = kHalfPi;  // lying flat, so it slides rather than rolls
    c.friction = Real(0.8);
    const int32_t id = world.addBody(c);
    world.body(id).velocity = Vec2(4, 0);

    run(world, Real(3));

    CHECK_NEAR(world.body(id).velocity.x, 0.0, 0.05);
    CHECK(world.body(id).position.x > 0);   // it did travel before stopping
    CHECK(world.body(id).position.x < 4);   // and it did not slide forever
}

TEST(Contact, aFrictionlessBodyKeepsSliding) {
    World2D world;
    addGround(world, Real(0));
    RigidBody2D c = capsule(Vec2(0, Real(0.08)), Real(0.08), Real(0.4));
    c.angle = kHalfPi;
    c.friction = Real(0);
    const int32_t id = world.addBody(c);
    world.body(id).velocity = Vec2(4, 0);

    run(world, Real(2));

    CHECK_NEAR(world.body(id).velocity.x, 4.0, 0.05);
}

TEST(Contact, frictionDecelerationIsRoughlyMuTimesG) {
    // mu = 0.5 on a flat surface should decelerate at about 0.5*g. Checked to
    // 20 percent, which is loose enough for solver discretisation but tight
    // enough to catch a friction cone that is off by a factor of two.
    World2D world;
    addGround(world, Real(0.5));
    RigidBody2D c = capsule(Vec2(0, Real(0.08)), Real(0.08), Real(0.4));
    c.angle = kHalfPi;
    c.friction = Real(0.5);
    const int32_t id = world.addBody(c);
    run(world, Real(0.5));  // settle first
    world.body(id).velocity = Vec2(5, 0);

    const Real before = world.body(id).velocity.x;
    run(world, Real(0.5));
    const Real after = world.body(id).velocity.x;

    const Real measured = (before - after) / Real(0.5);
    const Real expected = Real(0.5) * Real(9.81);
    CHECK_NEAR(measured, expected, expected * Real(0.2));
}

TEST(Contact, wallsStopHorizontalMotion) {
    World2D world;
    addGround(world);
    world.addHalfSpace(HalfSpace{Vec2(-1, 0), Real(-3), Real(0.9), Real(0)});  // solid where x > 3
    const int32_t id = world.addBody(capsule(Vec2(0, Real(0.2)), Real(0.2), 0));
    world.body(id).velocity = Vec2(20, 0);

    run(world, Real(2));

    CHECK(world.body(id).position.x < Real(3.05));
    CHECK(!world.stats().unstable);
}

TEST(Contact, contactImpulseSupportsTheBodysWeight) {
    World2D world;
    addGround(world);
    const Real radius = Real(0.2);
    const int32_t id = world.addBody(capsule(Vec2(0, radius), radius, 0));
    run(world, Real(2));

    // Impulse over one substep divided by dt is the support force, which at rest
    // must equal m*g.
    const Vec2 impulse = world.contactImpulseOn(id);
    const Real force = impulse.y / kDt;
    const Real weight = world.body(id).mass * Real(9.81);
    CHECK_NEAR(force, weight, weight * Real(0.1));
}

TEST(Contact, highSpeedImpactsStayFinite) {
    World2D world;
    addGround(world);
    const int32_t id = world.addBody(capsule(Vec2(0, 5), Real(0.1), Real(0.3)));
    world.body(id).velocity = Vec2(0, -150);
    world.body(id).angularVelocity = 80;

    run(world, Real(2));

    CHECK(world.body(id).isFinite());
    CHECK(!world.stats().unstable);
    CHECK(world.body(id).position.y > Real(-0.1));
}

TEST(Contact, collisionGroupsSuppressSelfCollision) {
    World2D world;
    world.gravity = Vec2(0, 0);
    RigidBody2D a = capsule(Vec2(0, 0), Real(0.3), 0);
    RigidBody2D b = capsule(Vec2(Real(0.1), 0), Real(0.3), 0);
    a.collisionGroup = 7;
    b.collisionGroup = 7;
    const int32_t ia = world.addBody(a);
    const int32_t ib = world.addBody(b);

    run(world, Real(1));

    // Deeply overlapping, same group: they must simply ignore each other.
    CHECK_NEAR(world.body(ia).position.x, 0.0, 1e-6);
    CHECK_NEAR(world.body(ib).position.x, 0.1, 1e-6);
    CHECK(world.stats().contactCount == 0);
}

TEST(Contact, differentGroupsStillCollide) {
    World2D world;
    world.gravity = Vec2(0, 0);
    RigidBody2D a = capsule(Vec2(0, 0), Real(0.3), 0);
    RigidBody2D b = capsule(Vec2(Real(0.1), 0), Real(0.3), 0);
    a.collisionGroup = 1;
    b.collisionGroup = 2;
    const int32_t ia = world.addBody(a);
    const int32_t ib = world.addBody(b);

    run(world, Real(1));

    const Real gap = world.body(ib).position.x - world.body(ia).position.x;
    CHECK(gap > Real(0.55));  // pushed apart towards the sum of radii
}
