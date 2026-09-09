#include <cmath>

#include "core/Test.h"
#include "physics/Collision3D.h"

using namespace aibf;

namespace {

RigidBody3D capsule(const Vec3& position, Real radius, Real halfLength,
                    const Quat& orientation = Quat::identity()) {
    RigidBody3D body;
    body.setCapsuleWithMass(radius, halfLength, Real(1));
    body.position = position;
    body.orientation = orientation;
    body.refreshInertiaWorld();
    return body;
}

const HalfSpace3D kGround{Vec3(0, 1, 0), Real(0), Real(0.9), Real(0)};

}  // namespace

// ---------------------------------------------------------------- segments

TEST(Segments3D, skewSegmentsFindTheirMutualPerpendicular) {
    // The classic case with no 2D analogue: two lines that never meet and are
    // not parallel. In 2D they would have to intersect.
    Real s = 0, t = 0;
    Vec3 c1, c2;
    const Real distSq = closestPointsBetweenSegments3D(
        Vec3(-1, 0, 0), Vec3(1, 0, 0),      // along X at y = 0
        Vec3(0, 2, -1), Vec3(0, 2, 1),      // along Z at y = 2
        s, t, c1, c2);

    CHECK_NEAR(std::sqrt(distSq), 2.0, 1e-5);
    CHECK_NEAR(s, 0.5, 1e-5);
    CHECK_NEAR(t, 0.5, 1e-5);
    CHECK_NEAR(c1.x, 0.0, 1e-5);
    CHECK_NEAR(c2.y, 2.0, 1e-5);

    // The connecting segment is perpendicular to both, which is what "closest"
    // means and is not implied by the distance alone.
    const Vec3 link = c2 - c1;
    CHECK_NEAR(dot(link, Vec3(2, 0, 0)), 0.0, 1e-4);
    CHECK_NEAR(dot(link, Vec3(0, 0, 2)), 0.0, 1e-4);
}

TEST(Segments3D, parallelSegmentsResolveWithoutDividingByZero) {
    Real s = 0, t = 0;
    Vec3 c1, c2;
    const Real distSq = closestPointsBetweenSegments3D(
        Vec3(0, 0, 0), Vec3(2, 0, 0),
        Vec3(0, Real(0.5), 0), Vec3(2, Real(0.5), 0),
        s, t, c1, c2);

    CHECK_NEAR(std::sqrt(distSq), 0.5, 1e-5);
    CHECK(std::isfinite(s) && std::isfinite(t));
    CHECK_NEAR(c2.y - c1.y, 0.5, 1e-5);
}

TEST(Segments3D, disjointParallelSegmentsMeetAtTheirEndpoints) {
    Real s = 0, t = 0;
    Vec3 c1, c2;
    const Real distSq = closestPointsBetweenSegments3D(
        Vec3(0, 0, 0), Vec3(1, 0, 0),
        Vec3(3, 0, 0), Vec3(4, 0, 0),
        s, t, c1, c2);

    CHECK_NEAR(std::sqrt(distSq), 2.0, 1e-5);
    CHECK_NEAR(s, 1.0, 1e-5);
    CHECK_NEAR(t, 0.0, 1e-5);
}

TEST(Segments3D, degeneratePointsDoNotDivideByZero) {
    Real s = 0, t = 0;
    Vec3 c1, c2;

    // Both degenerate: two spheres.
    Real distSq = closestPointsBetweenSegments3D(Vec3(1, 2, 3), Vec3(1, 2, 3),
                                                 Vec3(1, 2, 5), Vec3(1, 2, 5),
                                                 s, t, c1, c2);
    CHECK_NEAR(std::sqrt(distSq), 2.0, 1e-5);

    // One degenerate: a sphere against a capsule.
    distSq = closestPointsBetweenSegments3D(Vec3(0, 3, 0), Vec3(0, 3, 0),
                                            Vec3(-2, 0, 0), Vec3(2, 0, 0),
                                            s, t, c1, c2);
    CHECK_NEAR(std::sqrt(distSq), 3.0, 1e-5);
    CHECK_NEAR(t, 0.5, 1e-5);
    CHECK(std::isfinite(s));
}

TEST(Segments3D, distanceIsSymmetricInTheArguments) {
    const Vec3 p1(-1, Real(0.2), Real(0.3)), q1(Real(1.5), 0, -1);
    const Vec3 p2(0, 2, Real(0.4)), q2(1, Real(1.2), 2);

    Real s1 = 0, t1 = 0, s2 = 0, t2 = 0;
    Vec3 a1, b1, a2, b2;
    const Real forward = closestPointsBetweenSegments3D(p1, q1, p2, q2, s1, t1, a1, b1);
    const Real reverse = closestPointsBetweenSegments3D(p2, q2, p1, q1, s2, t2, a2, b2);

    CHECK_NEAR(forward, reverse, 1e-5);
    CHECK_NEAR(s1, t2, 1e-4);
    CHECK_NEAR(t1, s2, 1e-4);
}

// ---------------------------------------------------------------- tangent basis

TEST(TangentBasis, isOrthonormalForEveryNormalIncludingTheAxes) {
    const Vec3 normals[] = {
        Vec3(0, 1, 0), Vec3(1, 0, 0), Vec3(0, 0, 1),
        Vec3(0, -1, 0), Vec3(-1, 0, 0),
        normalize(Vec3(1, 1, 1)), normalize(Vec3(Real(0.3), Real(-0.9), Real(0.2))),
    };
    for (const Vec3& n : normals) {
        Vec3 t1, t2;
        buildTangentBasis(n, t1, t2);
        CHECK_NEAR(length(t1), 1.0, 1e-4);
        CHECK_NEAR(length(t2), 1.0, 1e-4);
        CHECK_NEAR(dot(t1, n), 0.0, 1e-4);
        CHECK_NEAR(dot(t2, n), 0.0, 1e-4);
        CHECK_NEAR(dot(t1, t2), 0.0, 1e-4);
    }
}

// ---------------------------------------------------------------- half-space

TEST(CapsuleHalfSpace3D, anUprightCapsuleTouchesTheGroundAtOnePoint) {
    // Standing on end: only the lower cap is near the plane, so a second contact
    // would be inventing support the shape does not have.
    RigidBody3D body = capsule(Vec3(0, Real(0.3), 0), Real(0.1), Real(0.2));
    Manifold3D manifold;
    const int count = collideCapsuleHalfSpace3D(body, kGround, manifold);

    CHECK(count == 1);
    CHECK_NEAR(manifold.points[0].separation, 0.0, 1e-5);
    CHECK_NEAR(manifold.points[0].position.y, 0.0, 1e-5);
}

TEST(CapsuleHalfSpace3D, theNormalPointsFromTheCapsuleIntoTheGround) {
    // The solver pushes body A along -normal, so for a capsule above the floor
    // the manifold normal has to point down. The first version of this file
    // asserted +1 here, which matched the implementation and was wrong in both
    // places: with the sign flipped the ground never resolves anything and a
    // dropped capsule falls to y = -599 over four seconds without a single
    // contact test failing.
    RigidBody3D body = capsule(Vec3(0, Real(0.29), 0), Real(0.1), Real(0.2));
    Manifold3D manifold;
    CHECK(collideCapsuleHalfSpace3D(body, kGround, manifold) == 1);
    CHECK_NEAR(manifold.normal.y, -1.0, 1e-6);
}

TEST(CapsuleHalfSpace3D, aCapsuleLyingFlatTouchesAtBothEnds) {
    // Two points is what lets a foot lie flat instead of pivoting on one end.
    const Quat onItsSide = Quat::fromAxisAngle(Vec3(0, 0, 1), kPi * Real(0.5));
    RigidBody3D body = capsule(Vec3(0, Real(0.1), 0), Real(0.1), Real(0.25), onItsSide);

    Manifold3D manifold;
    const int count = collideCapsuleHalfSpace3D(body, kGround, manifold);

    CHECK(count == 2);
    CHECK_NEAR(manifold.points[0].separation, 0.0, 1e-4);
    CHECK_NEAR(manifold.points[1].separation, 0.0, 1e-4);
    // Separated along the axis it is lying on, not stacked at one place.
    CHECK_NEAR(std::abs(manifold.points[0].position.x - manifold.points[1].position.x), 0.5, 1e-4);
    CHECK(manifold.points[0].id != manifold.points[1].id);
}

TEST(CapsuleHalfSpace3D, aSphereNeverProducesTwoContactsAtTheSamePlace) {
    // A sphere's two segment endpoints coincide, so the naive loop would emit a
    // duplicate contact with a different id, and the solver would apply the
    // support impulse twice.
    RigidBody3D body = capsule(Vec3(0, Real(0.1), 0), Real(0.1), Real(0));
    Manifold3D manifold;
    const int count = collideCapsuleHalfSpace3D(body, kGround, manifold);
    CHECK(count == 1);
}

TEST(CapsuleHalfSpace3D, aCapsuleWellClearOfTheGroundHasNoContacts) {
    RigidBody3D body = capsule(Vec3(0, Real(2.0), 0), Real(0.1), Real(0.2));
    Manifold3D manifold;
    CHECK(collideCapsuleHalfSpace3D(body, kGround, manifold) == 0);
}

TEST(CapsuleHalfSpace3D, contactsAppearBeforeTouchingSoTheSolverCanAnticipate) {
    // Speculative contacts: generated within the margin, with a positive
    // separation. A solver that only saw contacts after interpenetration would
    // have to push bodies back out instead of stopping them arriving.
    const Real gap = kSpeculativeMargin3D * Real(0.5);
    RigidBody3D body = capsule(Vec3(0, Real(0.3) + gap, 0), Real(0.1), Real(0.2));

    Manifold3D manifold;
    CHECK(collideCapsuleHalfSpace3D(body, kGround, manifold) == 1);
    CHECK(manifold.points[0].separation > Real(0));
    CHECK_NEAR(manifold.points[0].separation, gap, 1e-5);
}

TEST(CapsuleHalfSpace3D, penetrationIsReportedNegative) {
    RigidBody3D body = capsule(Vec3(0, Real(0.25), 0), Real(0.1), Real(0.2));
    Manifold3D manifold;
    CHECK(collideCapsuleHalfSpace3D(body, kGround, manifold) == 1);
    CHECK_NEAR(manifold.points[0].separation, -0.05, 1e-5);
}

TEST(CapsuleHalfSpace3D, theManifoldCarriesAUsableFrictionBasis) {
    RigidBody3D body = capsule(Vec3(0, Real(0.29), 0), Real(0.1), Real(0.2));
    Manifold3D manifold;
    CHECK(collideCapsuleHalfSpace3D(body, kGround, manifold) == 1);

    CHECK_NEAR(length(manifold.tangent1), 1.0, 1e-4);
    CHECK_NEAR(length(manifold.tangent2), 1.0, 1e-4);
    CHECK_NEAR(dot(manifold.tangent1, manifold.normal), 0.0, 1e-5);
    CHECK_NEAR(dot(manifold.tangent2, manifold.normal), 0.0, 1e-5);
    CHECK_NEAR(dot(manifold.tangent1, manifold.tangent2), 0.0, 1e-5);
}

// ---------------------------------------------------------------- capsule pairs

TEST(CapsuleCapsule3D, crossedCapsulesTouchOnTheirMutualPerpendicular) {
    // The genuinely 3D arrangement: one along X, one along Z, passing over it.
    const Quat alongX = Quat::fromAxisAngle(Vec3(0, 0, 1), kPi * Real(0.5));
    const Quat alongZ = Quat::fromAxisAngle(Vec3(1, 0, 0), kPi * Real(0.5));
    RigidBody3D a = capsule(Vec3(0, 0, 0), Real(0.1), Real(0.5), alongX);
    RigidBody3D b = capsule(Vec3(0, Real(0.15), 0), Real(0.1), Real(0.5), alongZ);
    a.index = 0;
    b.index = 1;

    Manifold3D manifold;
    CHECK(collideCapsuleCapsule3D(a, b, manifold) == 1);
    // Centres are 0.15 apart, radii sum to 0.2, so they overlap by 0.05.
    CHECK_NEAR(manifold.points[0].separation, -0.05, 1e-4);
    // The normal points from A towards B, which here is straight up.
    CHECK_NEAR(manifold.normal.y, 1.0, 1e-3);
}

TEST(CapsuleCapsule3D, separatedCapsulesProduceNothing) {
    RigidBody3D a = capsule(Vec3(0, 0, 0), Real(0.1), Real(0.2));
    RigidBody3D b = capsule(Vec3(0, 0, Real(3.0)), Real(0.1), Real(0.2));
    a.index = 0;
    b.index = 1;
    Manifold3D manifold;
    CHECK(collideCapsuleCapsule3D(a, b, manifold) == 0);
}

TEST(CapsuleCapsule3D, coincidentCapsulesStillProduceAFiniteNormal) {
    // Two bodies exactly on top of each other. The distance is zero, so the
    // normal cannot come from the separation direction, and returning a zero or
    // NaN normal here would poison the solver for the rest of the run.
    RigidBody3D a = capsule(Vec3(1, 1, 1), Real(0.1), Real(0.2));
    RigidBody3D b = capsule(Vec3(1, 1, 1), Real(0.1), Real(0.2));
    a.index = 0;
    b.index = 1;

    Manifold3D manifold;
    CHECK(collideCapsuleCapsule3D(a, b, manifold) == 1);
    CHECK(aibf::isFinite(manifold.normal));
    CHECK_NEAR(length(manifold.normal), 1.0, 1e-3);
    CHECK(aibf::isFinite(manifold.points[0].position));
}

TEST(CapsuleCapsule3D, twoSpheresTouchAlongTheLineOfCentres) {
    RigidBody3D a = capsule(Vec3(0, 0, 0), Real(0.5), Real(0));
    RigidBody3D b = capsule(Vec3(Real(0.8), 0, 0), Real(0.5), Real(0));
    a.index = 0;
    b.index = 1;

    Manifold3D manifold;
    CHECK(collideCapsuleCapsule3D(a, b, manifold) == 1);
    CHECK_NEAR(manifold.normal.x, 1.0, 1e-5);
    CHECK_NEAR(manifold.points[0].separation, -0.2, 1e-5);
    // The contact sits on A's surface.
    CHECK_NEAR(manifold.points[0].position.x, 0.5, 1e-5);
}
