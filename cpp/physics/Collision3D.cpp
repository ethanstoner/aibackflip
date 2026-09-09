#include "physics/Collision3D.h"

#include <algorithm>

namespace aibf {

Real closestPointsBetweenSegments3D(const Vec3& p1, const Vec3& q1,
                                    const Vec3& p2, const Vec3& q2,
                                    Real& s, Real& t, Vec3& c1, Vec3& c2) {
    const Vec3 d1 = q1 - p1;   // direction of segment 1
    const Vec3 d2 = q2 - p2;   // direction of segment 2
    const Vec3 r = p1 - p2;
    const Real a = dot(d1, d1);
    const Real e = dot(d2, d2);
    const Real f = dot(d2, r);

    // Either or both segments can be a single point, which happens for real
    // here rather than theoretically: a sphere is a capsule with halfLength 0.
    if (a <= kEpsilon && e <= kEpsilon) {
        s = t = 0;
        c1 = p1;
        c2 = p2;
        return lengthSq(c1 - c2);
    }
    if (a <= kEpsilon) {
        s = 0;
        t = clamp(f / e, Real(0), Real(1));
    } else {
        const Real c = dot(d1, r);
        if (e <= kEpsilon) {
            t = 0;
            s = clamp(-c / a, Real(0), Real(1));
        } else {
            const Real b = dot(d1, d2);
            const Real denom = a * e - b * b;
            // denom is zero exactly when the segments are parallel. Any point on
            // segment 1 is then equally close, so s is pinned to 0 and t is
            // solved for it; the clamping below moves both onto the overlap.
            s = denom > kEpsilon ? clamp((b * f - c * e) / denom, Real(0), Real(1)) : Real(0);

            const Real tNumerator = b * s + f;
            if (tNumerator < Real(0)) {
                t = 0;
                s = clamp(-c / a, Real(0), Real(1));
            } else if (tNumerator > e) {
                t = 1;
                s = clamp((b - c) / a, Real(0), Real(1));
            } else {
                t = tNumerator / e;
            }
        }
    }

    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
    return lengthSq(c1 - c2);
}

int collideCapsuleHalfSpace3D(const RigidBody3D& capsule, const HalfSpace3D& plane,
                              Manifold3D& out) {
    // A is the capsule, B is the plane, and the manifold normal runs from A to
    // B, which is *into* the ground. The solver pushes A along -normal, so this
    // is the sign that lifts the capsule out. Getting it backwards does not
    // produce a visibly wrong contact, it produces a ground that never resolves
    // anything and a body in permanent free fall.
    out.normal = -plane.normal;
    buildTangentBasis(out.normal, out.tangent1, out.tangent2);
    out.friction = combineFriction3D(capsule.friction, plane.friction);
    out.restitution = combineRestitution3D(capsule.restitution, plane.restitution);
    out.pointCount = 0;

    const Vec3 ends[2] = {capsule.endpointA(), capsule.endpointB()};

    for (int i = 0; i < 2; ++i) {
        const Real distance = plane.signedDistance(ends[i]) - capsule.radius;
        if (distance > kSpeculativeMargin3D) continue;

        ContactPoint3D& point = out.points[out.pointCount];
        // The surface point, not the segment endpoint, so the contact sits where
        // the capsule actually touches.
        point.position = ends[i] - plane.normal * capsule.radius;
        point.separation = distance;
        point.id = static_cast<uint32_t>(i);
        point.normalImpulse = 0;
        point.tangentImpulse1 = 0;
        point.tangentImpulse2 = 0;
        ++out.pointCount;

        // A sphere has both endpoints in the same place, so the second one would
        // be a duplicate contact at the same position with the same id.
        if (capsule.halfLength <= kEpsilon) break;
    }

    return out.pointCount;
}

int collideCapsuleCapsule3D(const RigidBody3D& a, const RigidBody3D& b, Manifold3D& out) {
    out.pointCount = 0;

    Real s = 0, t = 0;
    Vec3 ca, cb;
    const Real distanceSq = closestPointsBetweenSegments3D(a.endpointA(), a.endpointB(),
                                                           b.endpointA(), b.endpointB(),
                                                           s, t, ca, cb);
    const Real radii = a.radius + b.radius;
    const Real limit = radii + kSpeculativeMargin3D;
    if (distanceSq > limit * limit) return 0;

    const Real distance = std::sqrt(distanceSq);

    Vec3 normal;
    if (distance > kEpsilon) {
        normal = (cb - ca) / distance;
    } else {
        // Exactly coincident axes. Any direction perpendicular to A's own axis
        // separates them; picking one deterministically keeps warm starting
        // meaningful instead of flipping the normal between steps.
        Vec3 t1, t2;
        buildTangentBasis(normalize(a.endpointB() - a.endpointA() + Vec3(0, kEpsilon, 0)), t1, t2);
        normal = t1;
    }

    out.bodyA = a.index;
    out.bodyB = b.index;
    out.planeIndex = -1;
    out.normal = normal;
    buildTangentBasis(out.normal, out.tangent1, out.tangent2);
    out.friction = combineFriction3D(a.friction, b.friction);
    out.restitution = combineRestitution3D(a.restitution, b.restitution);

    ContactPoint3D& point = out.points[0];
    point.position = ca + normal * a.radius;
    point.separation = distance - radii;
    point.id = 0;
    point.normalImpulse = 0;
    point.tangentImpulse1 = 0;
    point.tangentImpulse2 = 0;
    out.pointCount = 1;

    return 1;
}

}  // namespace aibf
