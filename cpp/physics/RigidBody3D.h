// 3D rigid body. As in 2D, every collidable shape is a capsule: a segment of
// length 2*halfLength along the body's local +/-Y axis, swept by `radius`.
// Setting halfLength to 0 gives a sphere, so there is still only one collision
// routine to get right.
//
// Local +Y stays "up" for an unrotated body, matching RigidBody2D, so the same
// rest-pose authoring convention carries over from the 2D humanoid.
//
// Two things are genuinely different from 2D and both are here rather than in
// the solver:
//
//   * Inertia is a tensor, not a scalar, and it rotates with the body. The
//     body-frame tensor is constant and diagonal (a capsule's principal axes are
//     its own axes), so it is stored as three numbers and the world-space
//     inverse is rebuilt from the orientation as R * Ibody^-1 * R^T.
//   * Angular velocity and angular momentum are no longer parallel, which is
//     what makes a tumbling body precess. Nothing has to be added for that; it
//     falls out of the tensor as long as the world inverse is kept in step with
//     the orientation.
#pragma once

#include <cstdint>

#include "core/Math.h"

namespace aibf {

struct MassProperties3D {
    Real mass = 0;
    // Principal moments about the centre of mass, in the body frame. For a
    // capsule along local Y these are (transverse, axial, transverse).
    Vec3 inertia;
};

// Volume of a 3D capsule: a cylinder of length 2h plus one whole sphere (the
// two hemispherical caps combine).
inline Real capsuleVolume(Real radius, Real halfLength) {
    return Real(2) * kPi * radius * radius * halfLength +
           Real(4) / Real(3) * kPi * radius * radius * radius;
}

// Exact principal moments for the cylinder-plus-two-hemispheres decomposition.
//
// The caps term needs the same care as the 2D version. A solid hemisphere's
// moment about a transverse axis through the centre of its flat face is
// (2/5)*m*r^2, because that axis passes through the centre of the sphere the two
// halves came from. Shifting to the capsule centre has to go via the
// hemisphere's own centroid, which sits 3r/8 from the flat face, so the parallel
// axis theorem is applied twice and the 3r/8 term does not cancel:
//
//   I_transverse(caps) = m_caps * ( (2/5)r^2 + h^2 + (3/4)hr )
//
// Both degenerate cases are covered by tests: halfLength 0 must reduce to a
// sphere's (2/5)m r^2 on every axis, and radius -> 0 must reduce to a thin rod's
// m*(2h)^2/12 transverse with zero axial moment.
inline MassProperties3D capsuleMassProperties3D(Real radius, Real halfLength, Real density) {
    const Real r = radius, h = halfLength;
    const Real cylinderMass = density * Real(2) * kPi * r * r * h;
    const Real capsMass = density * Real(4) / Real(3) * kPi * r * r * r;

    const Real axial = cylinderMass * r * r * Real(0.5) +
                       capsMass * Real(0.4) * r * r;

    const Real cylinderTransverse =
        cylinderMass * (Real(3) * r * r + Real(4) * h * h) / Real(12);
    const Real capsTransverse =
        capsMass * (Real(0.4) * r * r + h * h + Real(0.75) * h * r);
    const Real transverse = cylinderTransverse + capsTransverse;

    return {cylinderMass + capsMass, Vec3(transverse, axial, transverse)};
}

// R * Ibody^-1 * R^T. Free rather than a member because the orientation
// integrator needs it at a hypothetical midpoint orientation, not just the
// body's current one.
inline Mat3 worldInverseInertia(const Quat& orientation, const Vec3& invInertiaLocal) {
    const Mat3 r = toMat3(orientation);
    const Mat3 scaled(r.c0 * invInertiaLocal.x,
                      r.c1 * invInertiaLocal.y,
                      r.c2 * invInertiaLocal.z);
    return scaled * transpose(r);
}

struct RigidBody3D {
    // ---- state ----
    Vec3 position;                    // centre of mass, world space
    Vec3 velocity;
    Quat orientation = Quat::identity();
    Vec3 angularVelocity;             // world frame, radians per second

    // ---- mass ----
    Real mass = 0, invMass = 0;
    Vec3 inertia;                     // principal moments, body frame
    Vec3 invInertiaLocal;             // their reciprocals, body frame
    // R * Ibody^-1 * R^T, refreshed whenever the orientation changes. Cached
    // rather than recomputed per constraint because a solver iteration touches
    // it several times per body and it is fixed for the duration of the step.
    Mat3 invInertiaWorld = Mat3::zero();

    // ---- shape ----
    Real radius = Real(0.05);
    Real halfLength = 0;

    // ---- material ----
    Real friction = Real(0.9);
    Real restitution = 0;
    Real linearDamping = 0;
    Real angularDamping = 0;

    // ---- per-step accumulators ----
    Vec3 force;
    Vec3 torque;

    bool isStatic = false;

    // Bodies sharing a positive group never collide with each other. Unlike 2D,
    // a 3D figure's left and right limbs do not share a plane, so self-collision
    // is meaningful here and the group is used more sparingly: only for pairs
    // that are jointed together and would otherwise fight their own constraint.
    int32_t collisionGroup = 0;

    // Index into the owning World's body array; set on creation.
    int32_t index = -1;

    void setMass(Real m, const Vec3& i) {
        mass = m;
        inertia = i;
        invMass = m > Real(0) ? Real(1) / m : Real(0);
        invInertiaLocal = Vec3(i.x > Real(0) ? Real(1) / i.x : Real(0),
                               i.y > Real(0) ? Real(1) / i.y : Real(0),
                               i.z > Real(0) ? Real(1) / i.z : Real(0));
        refreshInertiaWorld();
    }

    // Must be called after any direct write to `orientation`. The solver does
    // this once per step; anything that teleports a body has to do it too, and
    // forgetting leaves the body rotating about the axes it used to have.
    void refreshInertiaWorld() {
        if (isStatic || (invInertiaLocal.x == 0 && invInertiaLocal.y == 0 &&
                         invInertiaLocal.z == 0)) {
            invInertiaWorld = Mat3::zero();
            return;
        }
        invInertiaWorld = worldInverseInertia(orientation, invInertiaLocal);
    }

    // Advances the orientation over `dt` of torque-free rotation.
    //
    // Not a plain `orientation = integrate(orientation, angularVelocity, dt)`,
    // because for a tumbling body that **gains energy**, and at this engine's
    // rate it gains a lot of it.
    //
    // Angular velocity is not constant across a step even with no torque on the
    // body: the inertia tensor turns underneath it, so omega precesses.
    // Evaluating omega once at the start of the step is therefore first order in
    // dt no matter how exactly the rotation is then applied. That is worth
    // stating because the obvious suspect is the quaternion update, and it is
    // not: swapping the first-order update for an exact exponential map moved
    // the measured drift by less than 0.02 percentage points.
    //
    // So this integrates the quantity that is actually conserved. Angular
    // momentum is held fixed, omega is re-derived from it at the midpoint
    // orientation, and the full step is taken with that.
    //
    // Measured on a capsule spun about a non-principal axis, energy gain as a
    // percentage of the initial kinetic energy:
    //
    //       hz   sec   first-order   exact-rotation      midpoint
    //      240     1       +25.73%          +25.74%      +0.0136%
    //      240     5     +2017.44%        +2040.96%      +0.0650%
    //     2400     1        +1.36%           +1.36%      +0.1127%
    //
    // Angular momentum is conserved to machine precision in every column, so the
    // usual momentum check passes the broken version without complaint.
    //
    // Note the midpoint column gets *worse* at 2400 Hz. Its truncation error is
    // already below the noise floor of single-precision arithmetic, so more
    // steps just accumulate more rounding. Substepping does not buy accuracy
    // here; see the test of the same name.
    void integrateOrientation(Real dt) {
        if (isStatic) return;
        const Vec3 momentum = angularMomentum();
        const Quat midpoint = integrateExact(orientation, angularVelocity, dt * Real(0.5));
        const Vec3 omegaMid = worldInverseInertia(midpoint, invInertiaLocal) * momentum;
        orientation = integrateExact(orientation, omegaMid, dt);
        refreshInertiaWorld();
        angularVelocity = invInertiaWorld * momentum;
    }

    void setCapsule(Real r, Real h, Real density) {
        radius = r;
        halfLength = h;
        const MassProperties3D mp = capsuleMassProperties3D(r, h, density);
        setMass(mp.mass, mp.inertia);
    }

    // Body segment masses are known from anthropometry, but the density that
    // would produce them is not, so the config names the mass and the density is
    // back-solved to keep the inertia consistent with the shape.
    void setCapsuleWithMass(Real r, Real h, Real m) {
        radius = r;
        halfLength = h;
        const MassProperties3D unit = capsuleMassProperties3D(r, h, Real(1));
        const Real density = unit.mass > Real(0) ? m / unit.mass : Real(0);
        setMass(m, unit.inertia * density);
    }

    void makeStatic() {
        isStatic = true;
        mass = 0;
        invMass = 0;
        inertia = Vec3(0, 0, 0);
        invInertiaLocal = Vec3(0, 0, 0);
        invInertiaWorld = Mat3::zero();
        velocity = Vec3(0, 0, 0);
        angularVelocity = Vec3(0, 0, 0);
    }

    Vec3 localToWorld(const Vec3& local) const {
        return position + rotate(orientation, local);
    }
    Vec3 localToWorldDir(const Vec3& local) const { return rotate(orientation, local); }
    Vec3 worldToLocal(const Vec3& world) const {
        return rotateInverse(orientation, world - position);
    }
    Vec3 worldToLocalDir(const Vec3& world) const {
        return rotateInverse(orientation, world);
    }

    // Velocity of the material point currently at world offset `r` from the
    // centre of mass. This is the quantity every constraint is written against.
    Vec3 velocityAtOffset(const Vec3& r) const {
        return velocity + cross(angularVelocity, r);
    }
    Vec3 velocityAtPoint(const Vec3& world) const {
        return velocityAtOffset(world - position);
    }

    void applyImpulse(const Vec3& impulse, const Vec3& offset) {
        velocity += impulse * invMass;
        angularVelocity += invInertiaWorld * cross(offset, impulse);
    }
    void applyImpulseAtPoint(const Vec3& impulse, const Vec3& world) {
        applyImpulse(impulse, world - position);
    }
    void applyAngularImpulse(const Vec3& impulse) {
        angularVelocity += invInertiaWorld * impulse;
    }

    void applyForce(const Vec3& f) { force += f; }
    void applyForceAtPoint(const Vec3& f, const Vec3& world) {
        force += f;
        torque += cross(world - position, f);
    }
    void applyTorque(const Vec3& t) { torque += t; }
    void clearForces() {
        force = Vec3(0, 0, 0);
        torque = Vec3(0, 0, 0);
    }

    // Capsule segment endpoints in world space.
    Vec3 endpointA() const { return localToWorld(Vec3(0, -halfLength, 0)); }
    Vec3 endpointB() const { return localToWorld(Vec3(0, halfLength, 0)); }

    // Angular momentum about the centre of mass, world frame. Not parallel to
    // the angular velocity unless the body is spinning about a principal axis,
    // and it is the conserved quantity in free flight, so it is what the tests
    // check rather than omega.
    Vec3 angularMomentum() const {
        const Vec3 localOmega = rotateInverse(orientation, angularVelocity);
        const Vec3 localL(localOmega.x * inertia.x,
                          localOmega.y * inertia.y,
                          localOmega.z * inertia.z);
        return rotate(orientation, localL);
    }

    Real kineticEnergy() const {
        return Real(0.5) * mass * lengthSq(velocity) +
               Real(0.5) * dot(angularVelocity, angularMomentum());
    }

    bool isFinite() const {
        return aibf::isFinite(position) && aibf::isFinite(velocity) &&
               aibf::isFinite(orientation) && aibf::isFinite(angularVelocity);
    }
};

}  // namespace aibf
