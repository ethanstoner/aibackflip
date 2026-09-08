#include "core/Test.h"
#include "physics/Collision2D.h"

using namespace aibf;

namespace {

RigidBody2D makeCapsule(Vec2 position, Real angle, Real radius, Real halfLength) {
    RigidBody2D b;
    b.position = position;
    b.angle = angle;
    b.setCapsule(radius, halfLength, Real(1000));
    return b;
}

HalfSpace ground() { return HalfSpace{Vec2(0, 1), 0, Real(0.9), Real(0)}; }

}  // namespace

// ------------------------------------------------------- segment closest point

TEST(Segments, crossingSegmentsTouchAtTheIntersection) {
    Real s = 0, t = 0;
    Vec2 c1, c2;
    Real d2 = closestPointsBetweenSegments(Vec2(-1, 0), Vec2(1, 0), Vec2(0, -1), Vec2(0, 1),
                                           s, t, c1, c2);
    CHECK_NEAR(d2, 0.0, 1e-6);
    CHECK_VEC2_NEAR(c1, 0.0, 0.0, 1e-5);
    CHECK_VEC2_NEAR(c2, 0.0, 0.0, 1e-5);
}

TEST(Segments, parallelSegmentsResolveToAConsistentPair) {
    Real s = 0, t = 0;
    Vec2 c1, c2;
    Real d2 = closestPointsBetweenSegments(Vec2(0, 0), Vec2(2, 0), Vec2(0, 1), Vec2(2, 1),
                                           s, t, c1, c2);
    CHECK_NEAR(std::sqrt(d2), 1.0, 1e-5);
    CHECK_NEAR(c1.y, 0.0, 1e-5);
    CHECK_NEAR(c2.y, 1.0, 1e-5);
    CHECK_NEAR(c1.x, c2.x, 1e-5);  // the pair must be perpendicular to both
}

TEST(Segments, endpointsWinWhenSegmentsDoNotOverlap) {
    Real s = 0, t = 0;
    Vec2 c1, c2;
    closestPointsBetweenSegments(Vec2(0, 0), Vec2(1, 0), Vec2(3, 0), Vec2(4, 0), s, t, c1, c2);
    CHECK_VEC2_NEAR(c1, 1.0, 0.0, 1e-5);
    CHECK_VEC2_NEAR(c2, 3.0, 0.0, 1e-5);
}

TEST(Segments, degeneratePointsDoNotDivideByZero) {
    Real s = 0, t = 0;
    Vec2 c1, c2;
    Real d2 = closestPointsBetweenSegments(Vec2(1, 1), Vec2(1, 1), Vec2(4, 5), Vec2(4, 5),
                                           s, t, c1, c2);
    CHECK_NEAR(std::sqrt(d2), 5.0, 1e-5);
    CHECK(isFinite(c1));
    CHECK(isFinite(c2));
}

// ------------------------------------------------------- capsule vs half-space

TEST(CapsulePlane, restingCapsuleReportsZeroSeparation) {
    // Upright capsule whose lower cap exactly touches the ground.
    RigidBody2D c = makeCapsule(Vec2(0, Real(0.5) + Real(0.1)), 0, Real(0.1), Real(0.5));
    Manifold m;
    CHECK(collideCapsuleHalfSpace(c, ground(), m) == 1);
    CHECK_NEAR(m.points[0].separation, 0.0, 1e-5);
    CHECK_VEC2_NEAR(m.points[0].position, 0.0, 0.0, 1e-5);
}

TEST(CapsulePlane, normalPointsFromTheCapsuleIntoTheGround) {
    // The solver pushes body A along -normal, so for a capsule above the floor
    // the manifold normal has to point down. Getting this backwards makes the
    // ground suck bodies in instead of holding them up.
    RigidBody2D c = makeCapsule(Vec2(0, Real(0.4)), 0, Real(0.1), Real(0.5));
    Manifold m;
    collideCapsuleHalfSpace(c, ground(), m);
    CHECK_VEC2_NEAR(m.normal, 0.0, -1.0, 1e-6);
}

TEST(CapsulePlane, penetrationIsReportedAsNegativeSeparation) {
    RigidBody2D c = makeCapsule(Vec2(0, Real(0.55)), 0, Real(0.1), Real(0.5));
    Manifold m;
    collideCapsuleHalfSpace(c, ground(), m);
    CHECK_NEAR(m.points[0].separation, -0.05, 1e-5);
}

TEST(CapsulePlane, aFlatCapsuleProducesTwoContactPoints) {
    // A foot lying flat needs both ends supported, otherwise it pivots on one
    // corner and the humanoid rocks.
    RigidBody2D c = makeCapsule(Vec2(0, Real(0.1)), kHalfPi, Real(0.1), Real(0.5));
    Manifold m;
    CHECK(collideCapsuleHalfSpace(c, ground(), m) == 2);
    CHECK_NEAR(m.points[0].separation, 0.0, 1e-5);
    CHECK_NEAR(m.points[1].separation, 0.0, 1e-5);
    CHECK(m.points[0].id != m.points[1].id);
    // The two contacts are at opposite ends of the segment.
    CHECK_NEAR(std::abs(m.points[0].position.x - m.points[1].position.x), 1.0, 1e-4);
}

TEST(CapsulePlane, aDiscProducesOnlyOneContact) {
    // halfLength 0 means both endpoints coincide; emitting two contacts there
    // would double the support force under every ball in the world.
    RigidBody2D disc = makeCapsule(Vec2(0, Real(0.1)), 0, Real(0.1), 0);
    Manifold m;
    CHECK(collideCapsuleHalfSpace(disc, ground(), m) == 1);
}

TEST(CapsulePlane, aTiltedCapsuleOnlyReportsTheEndThatIsDown) {
    RigidBody2D c = makeCapsule(Vec2(0, Real(0.45)), Real(0.6), Real(0.1), Real(0.5));
    Manifold m;
    CHECK(collideCapsuleHalfSpace(c, ground(), m) == 1);
    CHECK(m.points[0].position.y < Real(0.1));
}

TEST(CapsulePlane, farAwayCapsulesProduceNoContact) {
    RigidBody2D c = makeCapsule(Vec2(0, 5), 0, Real(0.1), Real(0.5));
    Manifold m;
    CHECK(collideCapsuleHalfSpace(c, ground(), m) == 0);
}

TEST(CapsulePlane, contactsAppearInsideTheSpeculativeMargin) {
    // Just inside the margin: a contact must exist with positive separation, so
    // the solver can bleed off approach speed before anything overlaps.
    const Real gap = kSpeculativeMargin * Real(0.5);
    RigidBody2D c = makeCapsule(Vec2(0, Real(0.6) + gap), 0, Real(0.1), Real(0.5));
    Manifold m;
    CHECK(collideCapsuleHalfSpace(c, ground(), m) == 1);
    CHECK(m.points[0].separation > 0);
    CHECK_NEAR(m.points[0].separation, gap, 1e-5);
}

TEST(CapsulePlane, wallsWorkTheSameAsTheFloor) {
    HalfSpace wall{Vec2(1, 0), Real(-2), Real(0.5), Real(0)};  // solid where x < -2
    RigidBody2D disc = makeCapsule(Vec2(Real(-1.95), 1), 0, Real(0.1), 0);
    Manifold m;
    CHECK(collideCapsuleHalfSpace(disc, wall, m) == 1);
    CHECK_NEAR(m.points[0].separation, -0.05, 1e-5);
    CHECK_VEC2_NEAR(m.normal, -1.0, 0.0, 1e-6);
}

// ------------------------------------------------------- capsule vs capsule

TEST(CapsuleCapsule, overlappingDiscsSeparateAlongTheLineOfCentres) {
    RigidBody2D a = makeCapsule(Vec2(0, 0), 0, Real(0.5), 0);
    RigidBody2D b = makeCapsule(Vec2(Real(0.8), 0), 0, Real(0.5), 0);
    Manifold m;
    CHECK(collideCapsuleCapsule(a, b, m) == 1);
    CHECK_VEC2_NEAR(m.normal, 1.0, 0.0, 1e-5);
    CHECK_NEAR(m.points[0].separation, -0.2, 1e-5);
    // Contact sits on A's surface, along the normal.
    CHECK_VEC2_NEAR(m.points[0].position, 0.5, 0.0, 1e-5);
}

TEST(CapsuleCapsule, separatedShapesProduceNothing) {
    RigidBody2D a = makeCapsule(Vec2(0, 0), 0, Real(0.2), Real(0.5));
    RigidBody2D b = makeCapsule(Vec2(3, 0), 0, Real(0.2), Real(0.5));
    Manifold m;
    CHECK(collideCapsuleCapsule(a, b, m) == 0);
}

TEST(CapsuleCapsule, crossedCapsulesCollideAtTheCrossing) {
    RigidBody2D a = makeCapsule(Vec2(0, 0), 0, Real(0.1), Real(1));           // vertical
    RigidBody2D b = makeCapsule(Vec2(0, 0), kHalfPi, Real(0.1), Real(1));     // horizontal
    Manifold m;
    CHECK(collideCapsuleCapsule(a, b, m) == 1);
    CHECK_NEAR(m.points[0].separation, -0.2, 1e-4);
    CHECK(isFinite(m.normal));
    CHECK_NEAR(length(m.normal), 1.0, 1e-5);
}

TEST(CapsuleCapsule, coincidentBodiesStillGetAFiniteNormal) {
    // Exactly co-located capsules have no line of centres to work from. The
    // fallback has to produce a unit normal rather than a NaN, or one bad frame
    // poisons the whole episode.
    RigidBody2D a = makeCapsule(Vec2(1, 1), 0, Real(0.3), Real(0.4));
    RigidBody2D b = makeCapsule(Vec2(1, 1), 0, Real(0.3), Real(0.4));
    Manifold m;
    CHECK(collideCapsuleCapsule(a, b, m) == 1);
    CHECK(isFinite(m.normal));
    CHECK_NEAR(length(m.normal), 1.0, 1e-5);
    CHECK(isFinite(m.points[0].position));
}

TEST(CapsuleCapsule, materialsCombine) {
    RigidBody2D a = makeCapsule(Vec2(0, 0), 0, Real(0.5), 0);
    RigidBody2D b = makeCapsule(Vec2(Real(0.8), 0), 0, Real(0.5), 0);
    a.friction = Real(0.25);
    b.friction = Real(1.0);
    a.restitution = Real(0.1);
    b.restitution = Real(0.7);
    Manifold m;
    collideCapsuleCapsule(a, b, m);
    CHECK_NEAR(m.friction, 0.5, 1e-5);        // geometric mean
    CHECK_NEAR(m.restitution, 0.7, 1e-5);     // max
}
