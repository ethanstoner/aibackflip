// 2D rigid body. Every collidable shape is a capsule: a segment of length
// 2*halfLength along the body's local +/-Y axis, swept by `radius`. Setting
// halfLength to 0 gives a disc, which is how balls and joints-as-markers are
// represented, so there is only ever one collision routine to get right.
//
// Local +Y is "up" for an unrotated body, so a limb at angle 0 stands upright
// and joint angles read the way an animator expects.
#pragma once

#include <cstdint>

#include "core/Math.h"

namespace aibf {

struct MassProperties {
    Real mass = 0;
    Real inertia = 0;  // about the centre of mass
};

// Area of a 2D capsule: a 2r x 2h rectangle plus a full disc of radius r
// (the two hemispherical caps combine into one).
inline Real capsuleArea(Real radius, Real halfLength) {
    return Real(4) * radius * halfLength + kPi * radius * radius;
}

// Exact second moment for the rectangle-plus-two-half-discs decomposition.
//
// The caps term needs care: a half-disc's moment about the centre of its flat
// edge is m*r^2/2, but the parallel-axis theorem only applies from the centroid,
// which sits 4r/(3*pi) further out. Working that through, the two caps together
// contribute m_caps * (r^2/2 + h^2 + 2*h*4r/(3*pi)).
//
// Degenerate cases are covered by tests: halfLength 0 must reduce to a disc's
// m*r^2/2, and radius -> 0 must reduce to a rod's m*(2h)^2/12.
inline MassProperties capsuleMassProperties(Real radius, Real halfLength, Real density) {
    const Real r = radius, h = halfLength;
    const Real rectMass = density * Real(4) * r * h;
    const Real capsMass = density * kPi * r * r;

    const Real rectInertia = rectMass * (r * r + h * h) / Real(3);
    const Real centroidOffset = Real(4) * r / (Real(3) * kPi);
    const Real capsInertia = capsMass * (r * r * Real(0.5) + h * h + Real(2) * h * centroidOffset);

    return {rectMass + capsMass, rectInertia + capsInertia};
}

struct RigidBody2D {
    // ---- state ----
    Vec2 position;            // centre of mass, world space
    Vec2 velocity;
    Real angle = 0;           // radians, counter-clockwise
    Real angularVelocity = 0;

    // ---- mass ----
    Real mass = 0, invMass = 0;
    Real inertia = 0, invInertia = 0;

    // ---- shape ----
    Real radius = Real(0.05);
    Real halfLength = 0;

    // ---- material ----
    Real friction = Real(0.9);
    Real restitution = 0;
    Real linearDamping = 0;
    Real angularDamping = 0;

    // ---- per-step accumulators ----
    Vec2 force;
    Real torque = 0;

    bool isStatic = false;

    // Bodies sharing a positive group never collide with each other. The 2D
    // humanoid puts all of its parts in one group because left and right limbs
    // occupy the same plane by construction and would otherwise fight.
    int32_t collisionGroup = 0;

    // Index into the owning World's body array; set on creation.
    int32_t index = -1;

    void setMass(Real m, Real i) {
        mass = m;
        inertia = i;
        invMass = m > Real(0) ? Real(1) / m : Real(0);
        invInertia = i > Real(0) ? Real(1) / i : Real(0);
    }

    void setCapsule(Real r, Real h, Real density) {
        radius = r;
        halfLength = h;
        MassProperties mp = capsuleMassProperties(r, h, density);
        setMass(mp.mass, mp.inertia);
    }

    // Body segment masses are known from anthropometry, but the density that
    // would produce them is not, so the config names the mass and the density
    // is back-solved to keep the inertia consistent with the shape.
    void setCapsuleWithMass(Real r, Real h, Real m) {
        radius = r;
        halfLength = h;
        MassProperties unit = capsuleMassProperties(r, h, Real(1));
        const Real density = unit.mass > Real(0) ? m / unit.mass : Real(0);
        setMass(m, unit.inertia * density);
    }

    void makeStatic() {
        isStatic = true;
        mass = inertia = invMass = invInertia = 0;
        velocity = Vec2(0, 0);
        angularVelocity = 0;
    }

    Vec2 localToWorld(const Vec2& local) const { return position + rotate(local, angle); }
    Vec2 localToWorldDir(const Vec2& local) const { return rotate(local, angle); }
    Vec2 worldToLocal(const Vec2& world) const { return rotate(world - position, -angle); }

    // Velocity of the material point currently at world offset `r` from the
    // centre of mass. This is the quantity every constraint is written against.
    Vec2 velocityAtOffset(const Vec2& r) const { return velocity + cross(angularVelocity, r); }
    Vec2 velocityAtPoint(const Vec2& world) const { return velocityAtOffset(world - position); }

    void applyImpulse(const Vec2& impulse, const Vec2& offset) {
        velocity += impulse * invMass;
        angularVelocity += invInertia * cross(offset, impulse);
    }
    void applyImpulseAtPoint(const Vec2& impulse, const Vec2& world) {
        applyImpulse(impulse, world - position);
    }
    void applyAngularImpulse(Real impulse) { angularVelocity += invInertia * impulse; }

    void applyForce(const Vec2& f) { force += f; }
    void applyForceAtPoint(const Vec2& f, const Vec2& world) {
        force += f;
        torque += cross(world - position, f);
    }
    void applyTorque(Real t) { torque += t; }
    void clearForces() {
        force = Vec2(0, 0);
        torque = 0;
    }

    // Capsule segment endpoints in world space.
    Vec2 endpointA() const { return localToWorld(Vec2(0, -halfLength)); }
    Vec2 endpointB() const { return localToWorld(Vec2(0, halfLength)); }

    Real kineticEnergy() const {
        return Real(0.5) * mass * lengthSq(velocity) +
               Real(0.5) * inertia * angularVelocity * angularVelocity;
    }

    bool isFinite() const {
        return aibf::isFinite(position) && aibf::isFinite(velocity) &&
               std::isfinite(angle) && std::isfinite(angularVelocity);
    }
};

}  // namespace aibf
