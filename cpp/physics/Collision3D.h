// Narrow-phase collision for 3D capsules.
//
// As in 2D, every shape in the world is a capsule (a sphere is a capsule with
// halfLength 0) and the ground is a half-space, so there are only two routines
// and only two places a sign error can hide.
//
// The one structural difference from 2D: friction needs two tangent directions
// instead of one, because a contact plane in 3D is a plane rather than a line.
// Those are built per contact and stored with it, so the solver never has to
// re-derive a basis and risk picking a different one between iterations.
#pragma once

#include <cstdint>

#include "core/Math.h"
#include "physics/RigidBody3D.h"

namespace aibf {

// Solid region is { p : dot(normal, p) < offset }. The ground is
// normal = (0,1,0), offset = 0, so everything below y = 0 is solid.
struct HalfSpace3D {
    Vec3 normal{0, 1, 0};
    Real offset = 0;
    Real friction = Real(0.9);
    Real restitution = 0;

    Real signedDistance(const Vec3& p) const { return dot(normal, p) - offset; }
};

// Any two unit vectors perpendicular to `n` and to each other. The branch keeps
// the construction away from the degenerate case where n is parallel to the axis
// being crossed with, which would otherwise produce a zero-length tangent
// exactly when the ground contact is most ordinary.
inline void buildTangentBasis(const Vec3& n, Vec3& t1, Vec3& t2) {
    if (std::abs(n.x) >= Real(0.57735)) {
        t1 = normalize(Vec3(n.y, -n.x, 0));
    } else {
        t1 = normalize(Vec3(0, n.z, -n.y));
    }
    t2 = cross(n, t1);
}

inline Real combineFriction3D(Real a, Real b) { return std::sqrt(a * b); }
inline Real combineRestitution3D(Real a, Real b) { return a > b ? a : b; }

// Contacts are generated slightly before shapes touch, so the solver can bleed
// off approach velocity in advance instead of resolving an interpenetration
// after the fact.
constexpr Real kSpeculativeMargin3D = Real(0.02);

// Penetration this shallow is left alone, or the position solver jitters bodies
// that are resting perfectly still.
constexpr Real kLinearSlop3D = Real(0.0025);

struct ContactPoint3D {
    Vec3 position;         // world-space point on A's surface
    Real separation = 0;   // negative means interpenetrating
    uint32_t id = 0;       // stable feature id, matches impulses across steps

    // --- solver scratch, refilled each substep ---
    Vec3 rA, rB;                  // offsets from each centre of mass
    Real normalMass = 0;
    Real tangentMass1 = 0;
    Real tangentMass2 = 0;
    Real velocityBias = 0;
    Real normalImpulse = 0;       // accumulated, survives substeps (warm start)
    Real tangentImpulse1 = 0;
    Real tangentImpulse2 = 0;
};

struct Manifold3D {
    int32_t bodyA = -1;       // index into World3D bodies; -1 is the static world
    int32_t bodyB = -1;
    int32_t planeIndex = -1;  // which half-space, when bodyB is the static world
    Vec3 normal{0, 1, 0};     // unit, points from A towards B
    Vec3 tangent1, tangent2;  // an orthonormal basis of the contact plane
    int pointCount = 0;
    ContactPoint3D points[2];
    Real friction = Real(0.8);
    Real restitution = 0;
};

// Closest points between segments p1->q1 and p2->q2 in 3D. Handles both
// degenerate segments and the parallel case, which turns up constantly the
// moment two limbs line up. Returns the squared distance; s and t are the
// parameters along each segment.
Real closestPointsBetweenSegments3D(const Vec3& p1, const Vec3& q1,
                                    const Vec3& p2, const Vec3& q2,
                                    Real& s, Real& t, Vec3& c1, Vec3& c2);

// The half-space is body B (infinite mass), so the normal points from the
// capsule out along the plane normal. Produces up to two points, which is what
// lets a foot lie flat instead of pivoting on one end.
int collideCapsuleHalfSpace3D(const RigidBody3D& capsule, const HalfSpace3D& plane,
                              Manifold3D& out);

// Single-point manifold. Two capsules touch at a point unless they are exactly
// parallel and overlapping, and treating that measure-zero case as a line
// contact costs more than it buys.
int collideCapsuleCapsule3D(const RigidBody3D& a, const RigidBody3D& b, Manifold3D& out);

}  // namespace aibf
