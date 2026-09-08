#include "physics/Collision2D.h"

namespace aibf {

namespace {
constexpr Real kDegenerate = Real(1e-12);
}

Real closestPointsBetweenSegments(const Vec2& p1, const Vec2& q1, const Vec2& p2, const Vec2& q2,
                                  Real& s, Real& t, Vec2& c1, Vec2& c2) {
    const Vec2 d1 = q1 - p1;
    const Vec2 d2 = q2 - p2;
    const Vec2 r = p1 - p2;
    const Real a = dot(d1, d1);
    const Real e = dot(d2, d2);
    const Real f = dot(d2, r);

    if (a <= kDegenerate && e <= kDegenerate) {
        // Both segments are points (two discs).
        s = t = 0;
        c1 = p1;
        c2 = p2;
        return lengthSq(c1 - c2);
    }

    if (a <= kDegenerate) {
        s = 0;
        t = clamp(f / e, Real(0), Real(1));
    } else {
        const Real c = dot(d1, r);
        if (e <= kDegenerate) {
            t = 0;
            s = clamp(-c / a, Real(0), Real(1));
        } else {
            const Real b = dot(d1, d2);
            const Real denom = a * e - b * b;
            // denom == 0 means the segments are parallel; any point on the first
            // is equally close, so pin s and let the clamps below place t.
            s = (denom > kDegenerate) ? clamp((b * f - c * e) / denom, Real(0), Real(1)) : Real(0);
            t = (b * s + f) / e;
            if (t < Real(0)) {
                t = 0;
                s = clamp(-c / a, Real(0), Real(1));
            } else if (t > Real(1)) {
                t = 1;
                s = clamp((b - c) / a, Real(0), Real(1));
            }
        }
    }

    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
    return lengthSq(c1 - c2);
}

int collideCapsuleHalfSpace(const RigidBody2D& capsule, const HalfSpace& plane, Manifold& out) {
    out.pointCount = 0;
    // A is the capsule, B is the plane, and the manifold normal runs from A to
    // B - which is *into* the ground. The solver pushes A along -normal, so this
    // is the sign that lifts the capsule out.
    out.normal = -plane.normal;
    out.friction = combineFriction(capsule.friction, plane.friction);
    out.restitution = combineRestitution(capsule.restitution, plane.restitution);

    const Vec2 ends[2] = {capsule.endpointA(), capsule.endpointB()};
    // A disc has coincident endpoints; emitting both would double the contact.
    const int candidates = (capsule.halfLength > kDegenerate) ? 2 : 1;

    for (int i = 0; i < candidates; ++i) {
        const Real separation = plane.signedDistance(ends[i]) - capsule.radius;
        if (separation >= kSpeculativeMargin) continue;

        ContactPoint& cp = out.points[out.pointCount];
        cp = ContactPoint{};
        cp.position = ends[i] - plane.normal * capsule.radius;
        cp.separation = separation;
        cp.id = static_cast<uint32_t>(i);
        ++out.pointCount;
    }
    return out.pointCount;
}

int collideCapsuleCapsule(const RigidBody2D& a, const RigidBody2D& b, Manifold& out) {
    out.pointCount = 0;

    Real s = 0, t = 0;
    Vec2 ca, cb;
    const Real distSq =
        closestPointsBetweenSegments(a.endpointA(), a.endpointB(), b.endpointA(), b.endpointB(),
                                     s, t, ca, cb);

    const Real radiusSum = a.radius + b.radius;
    const Real dist = std::sqrt(distSq);
    const Real separation = dist - radiusSum;
    if (separation >= kSpeculativeMargin) return 0;

    Vec2 normal;
    if (dist > Real(1e-9)) {
        normal = (cb - ca) / dist;
    } else {
        // Axes exactly coincide. Fall back to the line between centres, and to
        // straight up if even that is degenerate, so the pair still separates
        // instead of producing a NaN normal.
        normal = normalize(b.position - a.position);
        if (lengthSq(normal) < Real(0.5)) normal = Vec2(0, 1);
    }

    out.normal = normal;
    out.friction = combineFriction(a.friction, b.friction);
    out.restitution = combineRestitution(a.restitution, b.restitution);
    out.pointCount = 1;

    ContactPoint& cp = out.points[0];
    cp = ContactPoint{};
    cp.position = ca + normal * a.radius;
    cp.separation = separation;
    cp.id = 0;
    return 1;
}

}  // namespace aibf
