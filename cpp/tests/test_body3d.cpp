#include <cmath>
#include <cstdio>
#include <vector>

#include "core/Test.h"
#include "physics/RigidBody3D.h"

using namespace aibf;

// ---------------------------------------------------------------- mass properties

TEST(Capsule3D, aZeroLengthCapsuleIsASolidSphere) {
    const Real r = Real(0.4), density = Real(900);
    const MassProperties3D mp = capsuleMassProperties3D(r, Real(0), density);

    const Real sphereMass = density * Real(4) / Real(3) * kPi * r * r * r;
    const Real sphereInertia = Real(0.4) * sphereMass * r * r;

    CHECK_NEAR(mp.mass, sphereMass, sphereMass * Real(1e-5));
    // A sphere has no preferred axis, so all three principal moments agree.
    CHECK_NEAR(mp.inertia.x, sphereInertia, sphereInertia * Real(1e-5));
    CHECK_NEAR(mp.inertia.y, sphereInertia, sphereInertia * Real(1e-5));
    CHECK_NEAR(mp.inertia.z, sphereInertia, sphereInertia * Real(1e-5));
}

TEST(Capsule3D, aVanishinglyThinCapsuleIsAThinRod) {
    // The caps contribute at order r^3 and the cylinder at order r^2, so the
    // rod limit is approached from a shrinking radius rather than reached.
    const Real h = Real(0.9), density = Real(1000);
    const Real r = Real(1e-4);
    const MassProperties3D mp = capsuleMassProperties3D(r, h, density);

    const Real rodInertia = mp.mass * (Real(2) * h) * (Real(2) * h) / Real(12);
    CHECK_NEAR(mp.inertia.x, rodInertia, rodInertia * Real(1e-3));
    CHECK_NEAR(mp.inertia.z, rodInertia, rodInertia * Real(1e-3));
    // Nothing is off the axis, so the axial moment vanishes with the radius.
    CHECK(mp.inertia.y < rodInertia * Real(1e-6));
}

TEST(Capsule3D, massMatchesTheAnalyticVolume) {
    const Real r = Real(0.11), h = Real(0.23), density = Real(1050);
    const MassProperties3D mp = capsuleMassProperties3D(r, h, density);
    const Real expected = capsuleVolume(r, h) * density;
    CHECK_NEAR(mp.mass, expected, expected * Real(1e-5));
}

TEST(Capsule3D, aLimbIsHarderToTumbleEndOverEndThanToSpin) {
    // A limb is long and thin, so the transverse moments must exceed the axial
    // one. If this ever inverts, the principal axes have been mixed up and the
    // figure will rotate about the wrong ones while looking plausible at rest.
    const MassProperties3D mp = capsuleMassProperties3D(Real(0.06), Real(0.2), Real(1000));
    CHECK(mp.inertia.x > mp.inertia.y * Real(3));
    CHECK_NEAR(mp.inertia.x, mp.inertia.z, mp.inertia.x * Real(1e-6));
}

TEST(Capsule3D, backSolvingDensityPreservesTheInertiaRatios) {
    RigidBody3D body;
    body.setCapsuleWithMass(Real(0.07), Real(0.18), Real(4.2));
    CHECK_NEAR(body.mass, 4.2, 1e-5);

    const MassProperties3D unit = capsuleMassProperties3D(Real(0.07), Real(0.18), Real(1));
    // Scaling density scales mass and inertia together, so the shape's ratios
    // survive naming the mass directly.
    CHECK_NEAR(body.inertia.x / body.inertia.y, unit.inertia.x / unit.inertia.y, 1e-4);
}

// ---------------------------------------------------------------- inertia tensor

TEST(RigidBody3D, theWorldInverseInertiaRotatesWithTheBody) {
    RigidBody3D body;
    body.setCapsuleWithMass(Real(0.05), Real(0.5), Real(3));

    // Unrotated, the world tensor is just the body-frame diagonal.
    const Mat3& at_rest = body.invInertiaWorld;
    CHECK_NEAR(at_rest.c0.x, body.invInertiaLocal.x, body.invInertiaLocal.x * 1e-4);
    CHECK_NEAR(at_rest.c1.y, body.invInertiaLocal.y, body.invInertiaLocal.y * 1e-4);
    CHECK_NEAR(std::abs(at_rest.c0.y), 0.0, 1e-6);

    // Rotating the capsule's long axis from +Y onto +X must swap which world
    // axis is the easy one to spin about.
    body.orientation = Quat::fromAxisAngle(Vec3(0, 0, 1), -kPi * Real(0.5));
    body.refreshInertiaWorld();

    const Mat3& turned = body.invInertiaWorld;
    CHECK_NEAR(turned.c0.x, body.invInertiaLocal.y, body.invInertiaLocal.y * 1e-3);
    CHECK_NEAR(turned.c1.y, body.invInertiaLocal.x, body.invInertiaLocal.x * 1e-3);
}

TEST(RigidBody3D, theInverseInertiaIsSymmetric) {
    // R * D * R^T is symmetric for any diagonal D. Losing that means a transpose
    // is missing somewhere, and the solver will quietly inject energy.
    RigidBody3D body;
    body.setCapsuleWithMass(Real(0.09), Real(0.31), Real(5));
    body.orientation = normalize(Quat(Real(0.31), Real(-0.42), Real(0.15), Real(0.84)));
    body.refreshInertiaWorld();

    const Mat3& m = body.invInertiaWorld;
    const Real scale = std::abs(m.c0.x) + std::abs(m.c1.y) + std::abs(m.c2.z);
    CHECK_NEAR(m.c0.y, m.c1.x, scale * 1e-5);
    CHECK_NEAR(m.c0.z, m.c2.x, scale * 1e-5);
    CHECK_NEAR(m.c1.z, m.c2.y, scale * 1e-5);
}

TEST(RigidBody3D, aStaticBodyHasNoInverseInertiaAtAll) {
    RigidBody3D body;
    body.setCapsuleWithMass(Real(0.1), Real(0.4), Real(8));
    body.makeStatic();
    body.orientation = Quat::fromAxisAngle(Vec3(1, 1, 0), Real(0.7));
    body.refreshInertiaWorld();

    body.applyImpulse(Vec3(500, 0, 0), Vec3(0, Real(0.3), 0));
    CHECK_NEAR(length(body.velocity), 0.0, 1e-9);
    CHECK_NEAR(length(body.angularVelocity), 0.0, 1e-9);
}

// ---------------------------------------------------------------- dynamics

TEST(RigidBody3D, anImpulseThroughTheCentreOfMassAddsNoSpin) {
    RigidBody3D body;
    body.setCapsuleWithMass(Real(0.08), Real(0.25), Real(6));
    body.orientation = normalize(Quat(Real(0.2), Real(0.1), Real(-0.3), Real(0.9)));
    body.refreshInertiaWorld();

    body.applyImpulse(Vec3(12, -4, 7), Vec3(0, 0, 0));
    CHECK_NEAR(body.velocity.x, 12.0 / 6.0, 1e-5);
    CHECK_NEAR(body.velocity.y, -4.0 / 6.0, 1e-5);
    CHECK_NEAR(length(body.angularVelocity), 0.0, 1e-9);
}

TEST(RigidBody3D, anOffsetImpulseProducesTheAngularImpulseItShould) {
    RigidBody3D body;
    body.setCapsuleWithMass(Real(0.06), Real(0.4), Real(4));

    const Vec3 offset(0, Real(0.3), 0);
    const Vec3 impulse(9, 0, 0);
    body.applyImpulse(impulse, offset);

    // L = r x p, and with the body unrotated the world tensor is the body one.
    const Vec3 expectedL = cross(offset, impulse);
    const Vec3 actualL = body.angularMomentum();
    CHECK_NEAR(actualL.x, expectedL.x, 1e-4);
    CHECK_NEAR(actualL.y, expectedL.y, 1e-4);
    CHECK_NEAR(actualL.z, expectedL.z, std::abs(expectedL.z) * 1e-4 + 1e-4);
}

TEST(RigidBody3D, velocityAtAnOffsetMatchesAFiniteDifference) {
    // The same check the 2D engine uses, and for the same reason: it compares
    // omega x r against the numerical derivative of an actual rotation rather
    // than restating the formula twice.
    RigidBody3D body;
    body.setCapsuleWithMass(Real(0.05), Real(0.3), Real(2));
    body.angularVelocity = Vec3(Real(0.7), Real(-1.3), Real(2.1));
    body.velocity = Vec3(Real(0.4), Real(0.2), Real(-0.1));

    const Vec3 local(Real(0.02), Real(0.25), Real(-0.03));
    const Real dt = Real(1e-5);

    const Vec3 before = body.localToWorld(local);
    RigidBody3D moved = body;
    moved.position += moved.velocity * dt;
    moved.orientation = integrate(moved.orientation, moved.angularVelocity, dt);
    const Vec3 after = moved.localToWorld(local);

    const Vec3 numerical = (after - before) / dt;
    const Vec3 analytic = body.velocityAtOffset(body.localToWorldDir(local));

    CHECK_NEAR(numerical.x, analytic.x, 1e-3);
    CHECK_NEAR(numerical.y, analytic.y, 1e-3);
    CHECK_NEAR(numerical.z, analytic.z, 1e-3);
}

TEST(RigidBody3D, freeFlightConservesAngularMomentumWhileOmegaWanders) {
    // The reason orientation is a quaternion and inertia is a tensor. A body
    // spun about a non-principal axis keeps constant angular momentum, but its
    // angular velocity moves around in the world frame: it precesses. A scalar
    // moment of inertia, or a stale world tensor, produces a constant omega
    // instead, which looks stable and is wrong.
    RigidBody3D body;
    body.setCapsuleWithMass(Real(0.07), Real(0.45), Real(5));
    body.angularVelocity = Vec3(Real(3.0), Real(6.0), Real(0.4));

    const Vec3 initialL = body.angularMomentum();
    const Vec3 initialOmega = body.angularVelocity;
    const Real initialEnergy = body.kineticEnergy();

    // Five seconds at the rate the engine actually runs, rather than at some
    // rate chosen to make the numbers look better.
    const Real dt = Real(1) / Real(240);
    for (int step = 0; step < 1200; ++step) {
        body.integrateOrientation(dt);
    }

    const Vec3 finalL = body.angularMomentum();
    CHECK_NEAR(finalL.x, initialL.x, length(initialL) * 1e-3);
    CHECK_NEAR(finalL.y, initialL.y, length(initialL) * 1e-3);
    CHECK_NEAR(finalL.z, initialL.z, length(initialL) * 1e-3);

    // Torque-free rotation is also energy conserving.
    CHECK_NEAR(body.kineticEnergy(), initialEnergy, initialEnergy * 1e-3);

    // And omega genuinely moved, so the conservation above is not the trivial
    // consequence of nothing having happened.
    CHECK(length(body.angularVelocity - initialOmega) > length(initialOmega) * Real(0.05));
}

namespace {

RigidBody3D spinner() {
    // Deliberately spun about no principal axis, so the tensor turns underneath
    // the angular velocity and the body precesses. A body spun about a
    // principal axis would integrate perfectly under any of these schemes and
    // would prove nothing.
    RigidBody3D body;
    body.setCapsuleWithMass(Real(0.07), Real(0.45), Real(5));
    body.angularVelocity = Vec3(Real(3.0), Real(6.0), Real(0.4));
    return body;
}

// Percentage change in kinetic energy over `seconds` of torque-free rotation,
// integrating with omega sampled once at the start of each step.
Real explicitEnergyDrift(Real hz, Real seconds, bool exactRotation) {
    RigidBody3D body = spinner();
    const Vec3 momentum = body.angularMomentum();
    const Real start = body.kineticEnergy();
    const Real dt = Real(1) / hz;
    for (int i = 0; i < int(hz * seconds); ++i) {
        body.orientation = exactRotation
                               ? integrateExact(body.orientation, body.angularVelocity, dt)
                               : integrate(body.orientation, body.angularVelocity, dt);
        body.refreshInertiaWorld();
        body.angularVelocity = body.invInertiaWorld * momentum;
    }
    return (body.kineticEnergy() - start) / start * Real(100);
}

Real midpointEnergyDrift(Real hz, Real seconds) {
    RigidBody3D body = spinner();
    const Real start = body.kineticEnergy();
    const Real dt = Real(1) / hz;
    for (int i = 0; i < int(hz * seconds); ++i) body.integrateOrientation(dt);
    return (body.kineticEnergy() - start) / start * Real(100);
}

}  // namespace

TEST(RigidBody3D, samplingOmegaOnceAStepMakesATumblingBodyGainEnergy) {
    // The measurement that chose the orientation integrator, kept as a test
    // because the tempting simplification is to drop `integrateOrientation` and
    // write `orientation = integrate(orientation, angularVelocity, dt)` inline.
    //
    // Energy gain, percent of initial, on a capsule spun about a non-principal
    // axis:
    //
    //       hz   sec   first-order    exact-rotation      midpoint
    //      240     1       +25.73%           +25.74%      +0.0136%
    //      240     5     +2017.44%         +2040.96%      +0.0650%
    //     2400     1        +1.36%            +1.36%      +0.1127%
    //
    // At the rate this engine actually runs, sampling omega once a step
    // multiplies a tumbling body's kinetic energy by twenty over five seconds.
    // A backflip is a tumbling body in free flight, so this is not an edge case.
    CHECK(explicitEnergyDrift(Real(240), Real(1), false) > Real(10));
    CHECK(explicitEnergyDrift(Real(240), Real(5), false) > Real(500));

    // Applying the rotation exactly instead of to first order changes nothing,
    // which is the whole point: the error is in *when* omega is sampled, not in
    // how the rotation is applied. Both columns agree to three decimal places.
    CHECK_NEAR(explicitEnergyDrift(Real(240), Real(1), true),
               explicitEnergyDrift(Real(240), Real(1), false), Real(0.1));

    // Re-deriving omega from conserved angular momentum at the midpoint holds
    // the same body to a small fraction of a percent.
    CHECK(std::abs(midpointEnergyDrift(Real(240), Real(1))) < Real(0.1));
    CHECK(std::abs(midpointEnergyDrift(Real(240), Real(5))) < Real(0.5));
}

TEST(RigidBody3D, theMidpointIntegratorIsLimitedByFloatPrecisionNotTruncation) {
    // Backwards for a convergent scheme, and worth pinning: shortening the step
    // makes the midpoint integrator *worse*, 0.0136% at 240 Hz against 0.1127%
    // at 2400 Hz over the same simulated second.
    //
    // Truncation error is not what is being measured any more. `Real` is float,
    // and each step renormalises a quaternion and rebuilds a 3x3 tensor, so ten
    // times the steps accumulates roughly ten times the rounding. The practical
    // consequence is that substepping is not a way to buy accuracy here, and the
    // engine's 240 Hz is already on the right side of the trade.
    const Real slow = std::abs(midpointEnergyDrift(Real(240), Real(1)));
    const Real fast = std::abs(midpointEnergyDrift(Real(2400), Real(1)));
    CHECK(fast > slow * Real(2));

    // The explicit scheme, by contrast, behaves like the first-order method it
    // is: ten times the rate cuts its error by roughly ten.
    const Real explicitSlow = explicitEnergyDrift(Real(240), Real(1), false);
    const Real explicitFast = explicitEnergyDrift(Real(2400), Real(1), false);
    CHECK(explicitFast < explicitSlow * Real(0.2));
}

TEST(RigidBody3D, quaternionIntegrationStaysNormalised) {
    RigidBody3D body;
    body.setCapsuleWithMass(Real(0.05), Real(0.2), Real(1));
    body.angularVelocity = Vec3(Real(12), Real(-9), Real(17));

    for (int step = 0; step < 100000; ++step) {
        body.orientation = integrate(body.orientation, body.angularVelocity, Real(1) / Real(240));
    }
    CHECK_NEAR(length(body.orientation), 1.0, 1e-5);
    CHECK(body.isFinite());
}

TEST(RigidBody3D, localAndWorldRoundTrip) {
    RigidBody3D body;
    body.position = Vec3(Real(1.5), Real(-2.25), Real(0.75));
    body.orientation = normalize(Quat(Real(0.4), Real(0.2), Real(-0.5), Real(0.7)));

    const Vec3 point(Real(0.13), Real(-0.4), Real(0.06));
    const Vec3 back = body.worldToLocal(body.localToWorld(point));
    CHECK_NEAR(back.x, point.x, 1e-5);
    CHECK_NEAR(back.y, point.y, 1e-5);
    CHECK_NEAR(back.z, point.z, 1e-5);
}

TEST(RigidBody3D, capsuleEndpointsFollowTheLocalYAxis) {
    // The same convention as RigidBody2D, so rest-pose authoring carries over.
    RigidBody3D body;
    body.setCapsuleWithMass(Real(0.05), Real(0.3), Real(2));
    body.position = Vec3(1, 2, 3);

    CHECK_NEAR(body.endpointB().y - body.endpointA().y, 0.6, 1e-5);

    body.orientation = Quat::fromAxisAngle(Vec3(0, 0, 1), -kPi * Real(0.5));
    const Vec3 a = body.endpointA(), b = body.endpointB();
    CHECK_NEAR(b.x - a.x, 0.6, 1e-4);
    CHECK_NEAR(b.y - a.y, 0.0, 1e-4);
}
