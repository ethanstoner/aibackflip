#include "physics/World3D.h"

#include <algorithm>
#include <cmath>

namespace aibf {

namespace {

// Key for the warm-start cache. Contacts are identified by the pair of bodies,
// the plane, and the feature id, so an impulse follows the same physical contact
// across substeps even as the manifold list is rebuilt.
uint64_t contactKey(int32_t bodyA, int32_t bodyB, int32_t plane, uint32_t pointId) {
    const uint64_t a = static_cast<uint64_t>(static_cast<uint32_t>(bodyA + 1));
    const uint64_t b = static_cast<uint64_t>(static_cast<uint32_t>(bodyB + 1));
    const uint64_t p = static_cast<uint64_t>(static_cast<uint32_t>(plane + 1));
    return (a << 40) ^ (b << 20) ^ (p << 8) ^ pointId;
}

}  // namespace

World3D::World3D() { worldBody_.makeStatic(); }

int32_t World3D::addBody(const RigidBody3D& body) {
    RigidBody3D copy = body;
    copy.index = static_cast<int32_t>(bodies_.size());
    copy.refreshInertiaWorld();
    bodies_.push_back(copy);
    return copy.index;
}

int32_t World3D::addBallJoint(const BallJoint3D& joint) {
    balls_.push_back(joint);
    balls_.back().resetAccumulators();
    return static_cast<int32_t>(balls_.size()) - 1;
}

int32_t World3D::addHingeJoint(const HingeJoint3D& joint) {
    hinges_.push_back(joint);
    hinges_.back().resetAccumulators();
    return static_cast<int32_t>(hinges_.size()) - 1;
}

int32_t World3D::addHalfSpace(const HalfSpace3D& plane) {
    planes_.push_back(plane);
    return static_cast<int32_t>(planes_.size()) - 1;
}

void World3D::clear() {
    bodies_.clear();
    balls_.clear();
    hinges_.clear();
    planes_.clear();
    manifolds_.clear();
    impulseCache_.clear();
    nextImpulseCache_.clear();
    stats_ = WorldStats3D{};
}

void World3D::clearContactCache() {
    impulseCache_.clear();
    nextImpulseCache_.clear();
    manifolds_.clear();
    for (BallJoint3D& j : balls_) j.resetAccumulators();
    for (HingeJoint3D& j : hinges_) j.resetAccumulators();
}

void World3D::step(Real dt) {
    stats_.velocityClampEvents = 0;

    integrateVelocities(dt);
    generateManifolds(manifolds_);
    restoreCachedImpulses();

    for (BallJoint3D& j : balls_) j.prepare(bodyRef(j.bodyA), bodyRef(j.bodyB), solver, dt);
    for (HingeJoint3D& j : hinges_) j.prepare(bodyRef(j.bodyA), bodyRef(j.bodyB), solver, dt);
    prepareContacts();
    warmStart();

    for (int i = 0; i < solver.velocityIterations; ++i) solveVelocities(dt);

    integratePositions(dt);

    for (int i = 0; i < solver.positionIterations; ++i) solvePositions();

    storeImpulses();

    for (RigidBody3D& b : bodies_) b.clearForces();

    stats_.maxJointAnchorError = 0;
    for (const BallJoint3D& j : balls_) {
        stats_.maxJointAnchorError =
            std::max(stats_.maxJointAnchorError, j.anchorError(bodyRef(j.bodyA), bodyRef(j.bodyB)));
    }
    for (const HingeJoint3D& j : hinges_) {
        stats_.maxJointAnchorError =
            std::max(stats_.maxJointAnchorError, j.anchorError(bodyRef(j.bodyA), bodyRef(j.bodyB)));
    }

    checkForInstability();
}

void World3D::refreshContacts() {
    generateManifolds(manifolds_);
    stats_.contactCount = 0;
    stats_.manifoldCount = static_cast<int>(manifolds_.size());
    Real deepest = 0;
    for (const Manifold3D& m : manifolds_) {
        stats_.contactCount += m.pointCount;
        for (int i = 0; i < m.pointCount; ++i) deepest = std::max(deepest, -m.points[i].separation);
    }
    stats_.maxPenetration = deepest;
}

void World3D::integrateVelocities(Real dt) {
    for (RigidBody3D& b : bodies_) {
        if (b.isStatic || b.invMass == Real(0)) continue;
        b.velocity += (gravity + b.force * b.invMass) * dt;
        b.angularVelocity += b.invInertiaWorld * b.torque * dt;

        // Implicit damping: stable for any dt, unlike v *= (1 - c*dt).
        b.velocity *= Real(1) / (Real(1) + dt * b.linearDamping);
        b.angularVelocity *= Real(1) / (Real(1) + dt * b.angularDamping);
    }
}

void World3D::integratePositions(Real dt) {
    Real fastest = 0;
    for (RigidBody3D& b : bodies_) {
        if (b.isStatic) continue;

        const Real speed = length(b.velocity);
        if (speed > maxLinearVelocity) {
            b.velocity = b.velocity * (maxLinearVelocity / speed);
            ++stats_.velocityClampEvents;
        }
        const Real spin = length(b.angularVelocity);
        if (spin > maxAngularVelocity) {
            b.angularVelocity = b.angularVelocity * (maxAngularVelocity / spin);
            ++stats_.velocityClampEvents;
        }
        fastest = std::max(fastest, speed);

        b.position += b.velocity * dt;
        // Not a plain quaternion step: see RigidBody3D::integrateOrientation for
        // why sampling omega once a step makes a tumbling body gain energy.
        b.integrateOrientation(dt);
    }
    stats_.maxSpeed = fastest;
}

// ---------------------------------------------------------------- collision

void World3D::generateManifolds(std::vector<Manifold3D>& out) const {
    out.clear();
    Manifold3D manifold;

    for (size_t i = 0; i < bodies_.size(); ++i) {
        const RigidBody3D& body = bodies_[i];
        if (body.isStatic) continue;

        for (size_t p = 0; p < planes_.size(); ++p) {
            if (collideCapsuleHalfSpace3D(body, planes_[p], manifold) > 0) {
                manifold.bodyA = static_cast<int32_t>(i);
                manifold.bodyB = -1;  // the immovable world
                manifold.planeIndex = static_cast<int32_t>(p);
                out.push_back(manifold);
            }
        }
    }

    for (size_t i = 0; i < bodies_.size(); ++i) {
        for (size_t j = i + 1; j < bodies_.size(); ++j) {
            const RigidBody3D& a = bodies_[i];
            const RigidBody3D& b = bodies_[j];
            if (a.isStatic && b.isStatic) continue;
            // Bodies sharing a positive group never collide. Unlike 2D this is
            // used sparingly, because in 3D a figure's left and right limbs do
            // not share a plane and self-collision is meaningful.
            if (a.collisionGroup > 0 && a.collisionGroup == b.collisionGroup) continue;
            if (collideCapsuleCapsule3D(a, b, manifold) > 0) out.push_back(manifold);
        }
    }
}

void World3D::restoreCachedImpulses() {
    for (Manifold3D& m : manifolds_) {
        for (int i = 0; i < m.pointCount; ++i) {
            ContactPoint3D& cp = m.points[i];
            const auto it = impulseCache_.find(contactKey(m.bodyA, m.bodyB, m.planeIndex, cp.id));
            if (it == impulseCache_.end()) continue;
            cp.normalImpulse = it->second.normal;
            cp.tangentImpulse1 = it->second.tangent1;
            cp.tangentImpulse2 = it->second.tangent2;
        }
    }
}

void World3D::storeImpulses() {
    nextImpulseCache_.clear();
    for (const Manifold3D& m : manifolds_) {
        for (int i = 0; i < m.pointCount; ++i) {
            const ContactPoint3D& cp = m.points[i];
            nextImpulseCache_[contactKey(m.bodyA, m.bodyB, m.planeIndex, cp.id)] =
                CachedImpulse3D{cp.normalImpulse, cp.tangentImpulse1, cp.tangentImpulse2};
        }
    }
    impulseCache_.swap(nextImpulseCache_);
}

void World3D::prepareContacts() {
    for (Manifold3D& m : manifolds_) {
        RigidBody3D& a = bodyRef(m.bodyA);
        RigidBody3D& b = bodyRef(m.bodyB);

        for (int i = 0; i < m.pointCount; ++i) {
            ContactPoint3D& cp = m.points[i];
            cp.rA = cp.position - a.position;
            cp.rB = cp.position - b.position;

            auto effectiveMass = [&](const Vec3& direction) {
                const Vec3 crossA = cross(cp.rA, direction);
                const Vec3 crossB = cross(cp.rB, direction);
                const Real k = a.invMass + b.invMass +
                               dot(crossA, a.invInertiaWorld * crossA) +
                               dot(crossB, b.invInertiaWorld * crossB);
                return k > Real(0) ? Real(1) / k : Real(0);
            };

            cp.normalMass = effectiveMass(m.normal);
            cp.tangentMass1 = effectiveMass(m.tangent1);
            cp.tangentMass2 = effectiveMass(m.tangent2);

            // Restitution captured from the approach speed before the solver
            // touches it, and suppressed below the threshold so resting bodies
            // do not buzz.
            const Real vn = dot(b.velocityAtOffset(cp.rB) - a.velocityAtOffset(cp.rA), m.normal);
            cp.velocityBias = (vn < -solver.restitutionThreshold) ? -m.restitution * vn : Real(0);
        }
    }
}

void World3D::warmStart() {
    for (Manifold3D& m : manifolds_) {
        RigidBody3D& a = bodyRef(m.bodyA);
        RigidBody3D& b = bodyRef(m.bodyB);
        for (int i = 0; i < m.pointCount; ++i) {
            ContactPoint3D& cp = m.points[i];
            const Vec3 p = m.normal * cp.normalImpulse + m.tangent1 * cp.tangentImpulse1 +
                           m.tangent2 * cp.tangentImpulse2;
            a.applyImpulse(-p, cp.rA);
            b.applyImpulse(p, cp.rB);
        }
    }
    for (BallJoint3D& j : balls_) j.warmStart(bodyRef(j.bodyA), bodyRef(j.bodyB));
    for (HingeJoint3D& j : hinges_) j.warmStart(bodyRef(j.bodyA), bodyRef(j.bodyB));
}

void World3D::solveVelocities(Real dt) {
    const Real invDt = Real(1) / dt;

    for (BallJoint3D& j : balls_) j.solveVelocity(bodyRef(j.bodyA), bodyRef(j.bodyB), solver, dt);
    for (HingeJoint3D& j : hinges_) j.solveVelocity(bodyRef(j.bodyA), bodyRef(j.bodyB), solver, dt);

    for (Manifold3D& m : manifolds_) {
        RigidBody3D& a = bodyRef(m.bodyA);
        RigidBody3D& b = bodyRef(m.bodyB);

        // Friction first, bounded by the normal impulse accumulated so far, then
        // the normal constraint. The two tangents are solved as a pair and the
        // friction cone is applied to their combined magnitude, not per axis: a
        // per-axis clamp would make a square friction limit, and a body sliding
        // diagonally would grip harder than one sliding along an axis.
        for (int i = 0; i < m.pointCount; ++i) {
            ContactPoint3D& cp = m.points[i];
            const Vec3 relative = b.velocityAtOffset(cp.rB) - a.velocityAtOffset(cp.rA);

            Real lambda1 = cp.tangentMass1 * -dot(relative, m.tangent1);
            Real lambda2 = cp.tangentMass2 * -dot(relative, m.tangent2);

            const Real maxFriction = m.friction * cp.normalImpulse;
            Real newImpulse1 = cp.tangentImpulse1 + lambda1;
            Real newImpulse2 = cp.tangentImpulse2 + lambda2;
            const Real magnitude =
                std::sqrt(newImpulse1 * newImpulse1 + newImpulse2 * newImpulse2);
            if (magnitude > maxFriction && magnitude > Real(0)) {
                const Real scale = maxFriction / magnitude;
                newImpulse1 *= scale;
                newImpulse2 *= scale;
            }
            lambda1 = newImpulse1 - cp.tangentImpulse1;
            lambda2 = newImpulse2 - cp.tangentImpulse2;
            cp.tangentImpulse1 = newImpulse1;
            cp.tangentImpulse2 = newImpulse2;

            const Vec3 p = m.tangent1 * lambda1 + m.tangent2 * lambda2;
            a.applyImpulse(-p, cp.rA);
            b.applyImpulse(p, cp.rB);
        }

        for (int i = 0; i < m.pointCount; ++i) {
            ContactPoint3D& cp = m.points[i];
            const Real vn = dot(b.velocityAtOffset(cp.rB) - a.velocityAtOffset(cp.rA), m.normal);

            // A still-separated contact closes at exactly the rate that lands it
            // on the surface this substep and no faster. That is what
            // speculative contacts buy: penetration is prevented rather than
            // repaired, so no energy has to be injected to push bodies apart.
            const Real target =
                (cp.separation > Real(0)) ? (-cp.separation * invDt) : cp.velocityBias;

            Real lambda = -cp.normalMass * (vn - target);
            const Real previous = cp.normalImpulse;
            cp.normalImpulse = std::max(previous + lambda, Real(0));
            lambda = cp.normalImpulse - previous;
            const Vec3 p = m.normal * lambda;
            a.applyImpulse(-p, cp.rA);
            b.applyImpulse(p, cp.rB);
        }
    }
}

void World3D::solvePositions() {
    for (BallJoint3D& j : balls_) j.solvePosition(bodyRef(j.bodyA), bodyRef(j.bodyB), solver);
    for (HingeJoint3D& j : hinges_) j.solvePosition(bodyRef(j.bodyA), bodyRef(j.bodyB), solver);

    // Regenerated from the current transforms. Reusing the velocity pass's
    // anchors would correct against geometry the position pass has already
    // moved, which shows up as bodies creeping into the floor under a stack.
    generateManifolds(positionScratch_);

    Real deepest = 0;
    for (Manifold3D& m : positionScratch_) {
        RigidBody3D& a = bodyRef(m.bodyA);
        RigidBody3D& b = bodyRef(m.bodyB);
        for (int i = 0; i < m.pointCount; ++i) {
            const ContactPoint3D& cp = m.points[i];
            deepest = std::max(deepest, -cp.separation);
            const Real correction = cp.separation + kLinearSlop3D;
            if (correction >= Real(0)) continue;

            const Vec3 rA = cp.position - a.position;
            const Vec3 rB = cp.position - b.position;
            const Vec3 crossA = cross(rA, m.normal);
            const Vec3 crossB = cross(rB, m.normal);
            const Real k = a.invMass + b.invMass + dot(crossA, a.invInertiaWorld * crossA) +
                           dot(crossB, b.invInertiaWorld * crossB);
            if (k <= Real(0)) continue;

            const Real magnitude =
                std::min(-solver.contactBaumgarte * correction / k,
                         solver.maxLinearCorrection / k);
            const Vec3 p = m.normal * magnitude;

            a.position -= p * a.invMass;
            b.position += p * b.invMass;
        }
    }
    stats_.maxPenetration = deepest;
}

void World3D::checkForInstability() {
    stats_.unstable = false;
    for (const RigidBody3D& b : bodies_) {
        if (!b.isFinite()) {
            stats_.unstable = true;
            return;
        }
    }
    stats_.contactCount = 0;
    stats_.manifoldCount = static_cast<int>(manifolds_.size());
    for (const Manifold3D& m : manifolds_) stats_.contactCount += m.pointCount;
}

// ---------------------------------------------------------------- queries

bool World3D::hasContact(int32_t bodyIndex) const {
    for (const Manifold3D& m : manifolds_) {
        if (m.bodyA != bodyIndex && m.bodyB != bodyIndex) continue;
        for (int i = 0; i < m.pointCount; ++i) {
            if (m.points[i].separation <= kSpeculativeMargin3D * Real(0.5)) return true;
        }
    }
    return false;
}

Real World3D::totalMass() const {
    Real total = 0;
    for (const RigidBody3D& b : bodies_) {
        if (!b.isStatic) total += b.mass;
    }
    return total;
}

Vec3 World3D::centerOfMass() const {
    Vec3 weighted(0, 0, 0);
    Real total = 0;
    for (const RigidBody3D& b : bodies_) {
        if (b.isStatic) continue;
        weighted += b.position * b.mass;
        total += b.mass;
    }
    return total > Real(0) ? weighted / total : Vec3(0, 0, 0);
}

Vec3 World3D::centerOfMassVelocity() const {
    Vec3 weighted(0, 0, 0);
    Real total = 0;
    for (const RigidBody3D& b : bodies_) {
        if (b.isStatic) continue;
        weighted += b.velocity * b.mass;
        total += b.mass;
    }
    return total > Real(0) ? weighted / total : Vec3(0, 0, 0);
}

Real World3D::kineticEnergy() const {
    Real total = 0;
    for (const RigidBody3D& b : bodies_) {
        if (!b.isStatic) total += b.kineticEnergy();
    }
    return total;
}

Real World3D::potentialEnergy(Real datum) const {
    Real total = 0;
    for (const RigidBody3D& b : bodies_) {
        if (!b.isStatic) total += b.mass * -gravity.y * (b.position.y - datum);
    }
    return total;
}

Vec3 World3D::angularMomentum() const {
    const Vec3 com = centerOfMass();
    const Vec3 comVelocity = centerOfMassVelocity();
    Vec3 total(0, 0, 0);
    for (const RigidBody3D& b : bodies_) {
        if (b.isStatic) continue;
        // Spin about the body's own centre of mass, plus the orbital term from
        // the body moving around the system's centre of mass.
        total += b.angularMomentum();
        total += cross(b.position - com, (b.velocity - comVelocity) * b.mass);
    }
    return total;
}

}  // namespace aibf
