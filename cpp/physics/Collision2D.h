// Narrow-phase collision for 2D capsules.
//
// Only two routines are needed because every shape in the world is a capsule
// (a disc is a capsule with halfLength 0) and the ground and walls are
// half-spaces. That keeps the number of places a sign error can hide small.
#pragma once

#include <cstdint>

#include "core/Math.h"
#include "physics/RigidBody2D.h"

namespace aibf {

// Solid region is { p : dot(normal, p) < offset }. The ground is
// normal = (0,1), offset = 0, so everything below y = 0 is solid.
struct HalfSpace {
    Vec2 normal{0, 1};
    Real offset = 0;
    Real friction = Real(0.9);
    Real restitution = 0;

    Real signedDistance(const Vec2& p) const { return dot(normal, p) - offset; }
};

// Combination rules, matching the usual conventions: friction multiplies
// geometrically so a slippery surface dominates, restitution takes the max so
// one bouncy body is enough to make a pair bounce.
inline Real combineFriction(Real a, Real b) { return std::sqrt(a * b); }
inline Real combineRestitution(Real a, Real b) { return a > b ? a : b; }

// Contacts are generated slightly before shapes touch, so the solver can bleed
// off approach velocity in advance instead of resolving an interpenetration
// after the fact. Anything closer than this margin produces a contact.
constexpr Real kSpeculativeMargin = Real(0.02);

// Penetration this shallow is left alone. Without it the position solver jitters
// bodies that are resting perfectly still.
constexpr Real kLinearSlop = Real(0.0025);

struct ContactPoint {
    Vec2 position;         // world-space point on A's surface
    Real separation = 0;   // negative means interpenetrating
    uint32_t id = 0;       // stable feature id, used to match impulses across steps

    // --- solver scratch, filled in by the solver each substep ---
    Vec2 rA, rB;                  // offsets from each centre of mass
    Real normalMass = 0;
    Real tangentMass = 0;
    Real velocityBias = 0;
    Real normalImpulse = 0;       // accumulated, survives across substeps (warm start)
    Real tangentImpulse = 0;
};

struct Manifold {
    int32_t bodyA = -1;      // index into World2D bodies; -1 is the static world
    int32_t bodyB = -1;
    int32_t planeIndex = -1; // which half-space, when bodyB is the static world
    Vec2 normal{0, 1};       // unit, points from A towards B
    int pointCount = 0;
    ContactPoint points[2];
    Real friction = Real(0.8);
    Real restitution = 0;
};

// Closest points between segments p1->q1 and p2->q2. Handles both-degenerate
// and parallel cases, which show up constantly once limbs line up.
// Returns the squared distance; s and t are the barycentric parameters.
Real closestPointsBetweenSegments(const Vec2& p1, const Vec2& q1, const Vec2& p2, const Vec2& q2,
                                  Real& s, Real& t, Vec2& c1, Vec2& c2);

// The half-space is treated as body B (infinite mass), so the normal points
// from the capsule out along the plane normal. Produces up to two points, which
// is what lets a foot lie flat instead of pivoting on one corner.
int collideCapsuleHalfSpace(const RigidBody2D& capsule, const HalfSpace& plane, Manifold& out);

// Single-point manifold; adequate here because the humanoid disables its own
// self-collision and the only free bodies are discs.
int collideCapsuleCapsule(const RigidBody2D& a, const RigidBody2D& b, Manifold& out);

}  // namespace aibf
