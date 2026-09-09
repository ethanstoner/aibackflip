#include "physics/World2D.h"

#include <algorithm>
#include <cmath>

namespace aibf {

namespace {

// Exact bit packing rather than a hash, so two different contacts can never
// share a warm-start slot and inherit each other's impulse.
//   bodyA: 24 bits | bodyB: 24 bits | plane: 12 bits | point id: 4 bits
uint64_t contactKey(int32_t bodyA, int32_t bodyB, int32_t planeIndex, uint32_t pointId) {
    const uint64_t a = static_cast<uint64_t>(static_cast<uint32_t>(bodyA + 1)) & 0xFFFFFFull;
    const uint64_t b = static_cast<uint64_t>(static_cast<uint32_t>(bodyB + 1)) & 0xFFFFFFull;
    const uint64_t p = static_cast<uint64_t>(static_cast<uint32_t>(planeIndex + 1)) & 0xFFFull;
    return (a << 40) | (b << 16) | (p << 4) | (pointId & 0xFull);
}

// Perpendicular to the contact normal; the friction direction.
Vec2 contactTangent(const Vec2& normal) { return Vec2(normal.y, -normal.x); }

}  // namespace

World2D::World2D() { worldBody_.makeStatic(); }

int32_t World2D::addBody(const RigidBody2D& body) {
    RigidBody2D copy = body;
    copy.index = static_cast<int32_t>(bodies_.size());
    bodies_.push_back(copy);
    return copy.index;
}

int32_t World2D::addJoint(const RevoluteJoint2D& joint) {
    joints_.push_back(joint);
    return static_cast<int32_t>(joints_.size()) - 1;
}

int32_t World2D::addHalfSpace(const HalfSpace& plane) {
    planes_.push_back(plane);
    return static_cast<int32_t>(planes_.size()) - 1;
}

void World2D::clear() {
    bodies_.clear();
    joints_.clear();
    planes_.clear();
    manifolds_.clear();
    impulseCache_.clear();
    nextImpulseCache_.clear();
    mouse_ = MouseSpring{};
    stats_ = WorldStats{};
}

void World2D::clearContactCache() {
    impulseCache_.clear();
    nextImpulseCache_.clear();
    manifolds_.clear();
    for (RevoluteJoint2D& j : joints_) j.resetAccumulators();
    mouse_.accumulated = Vec2(0, 0);
}

// ---------------------------------------------------------------- step

void World2D::step(Real dt) {
    stats_.velocityClampEvents = 0;

    integrateVelocities(dt);
    generateManifolds(manifolds_);
    restoreCachedImpulses();

    for (RevoluteJoint2D& j : joints_) {
        j.prepare(bodyRef(j.bodyA), bodyRef(j.bodyB), solver, dt);
    }
    prepareContacts(dt);
    warmStart();

    for (int i = 0; i < solver.velocityIterations; ++i) solveVelocities(dt);

    integratePositions(dt);

    for (int i = 0; i < solver.positionIterations; ++i) solvePositions();

    storeImpulses();

    for (RigidBody2D& b : bodies_) b.clearForces();

    // Anchor drift is reported after the position pass, which is the number
    // that actually matters: a joint that drifts during the velocity solve but
    // is pulled back before the frame ends is not a visible failure.
    stats_.maxJointAnchorError = 0;
    for (const RevoluteJoint2D& j : joints_) {
        stats_.maxJointAnchorError =
            std::max(stats_.maxJointAnchorError, j.anchorError(bodyRef(j.bodyA), bodyRef(j.bodyB)));
    }

    checkForInstability();
}

void World2D::refreshContacts() {
    generateManifolds(manifolds_);
    stats_.contactCount = 0;
    stats_.manifoldCount = static_cast<int>(manifolds_.size());
    Real deepest = 0;
    for (const Manifold& m : manifolds_) {
        stats_.contactCount += m.pointCount;
        for (int i = 0; i < m.pointCount; ++i) {
            deepest = std::max(deepest, -m.points[i].separation);
        }
    }
    stats_.maxPenetration = deepest;
}

void World2D::integrateVelocities(Real dt) {
    for (RigidBody2D& b : bodies_) {
        if (b.isStatic || b.invMass == Real(0)) continue;
        b.velocity += (gravity + b.force * b.invMass) * dt;
        b.angularVelocity += b.torque * b.invInertia * dt;

        // Implicit damping: stable for any dt, unlike v *= (1 - c*dt).
        b.velocity *= Real(1) / (Real(1) + dt * b.linearDamping);
        b.angularVelocity *= Real(1) / (Real(1) + dt * b.angularDamping);
    }
}

void World2D::integratePositions(Real dt) {
    Real fastest = 0;
    for (RigidBody2D& b : bodies_) {
        if (b.isStatic) continue;

        const Real speed = length(b.velocity);
        if (speed > maxLinearVelocity) {
            b.velocity = b.velocity * (maxLinearVelocity / speed);
            ++stats_.velocityClampEvents;
        }
        if (std::abs(b.angularVelocity) > maxAngularVelocity) {
            b.angularVelocity = sign(b.angularVelocity) * maxAngularVelocity;
            ++stats_.velocityClampEvents;
        }
        fastest = std::max(fastest, speed);

        b.position += b.velocity * dt;
        b.angle += b.angularVelocity * dt;
    }
    stats_.maxSpeed = fastest;
}

// ---------------------------------------------------------------- collision

void World2D::generateManifolds(std::vector<Manifold>& out) const {
    out.clear();
    const int32_t n = static_cast<int32_t>(bodies_.size());
    const int32_t planeCount = static_cast<int32_t>(planes_.size());

    for (int32_t i = 0; i < n; ++i) {
        const RigidBody2D& a = bodies_[static_cast<size_t>(i)];
        if (a.isStatic) continue;

        for (int32_t p = 0; p < planeCount; ++p) {
            Manifold m;
            m.bodyA = i;
            m.bodyB = -1;
            m.planeIndex = p;
            if (collideCapsuleHalfSpace(a, planes_[static_cast<size_t>(p)], m) > 0) {
                out.push_back(m);
            }
        }
    }

    for (int32_t i = 0; i < n; ++i) {
        const RigidBody2D& a = bodies_[static_cast<size_t>(i)];
        for (int32_t j = i + 1; j < n; ++j) {
            const RigidBody2D& b = bodies_[static_cast<size_t>(j)];
            if (a.isStatic && b.isStatic) continue;
            // Parts of the same articulated figure never collide with each
            // other. In 2D the left and right limbs occupy the same plane by
            // construction, so self-collision would be a permanent fight.
            if (a.collisionGroup > 0 && a.collisionGroup == b.collisionGroup) continue;

            Manifold m;
            m.bodyA = i;
            m.bodyB = j;
            m.planeIndex = -1;
            if (collideCapsuleCapsule(a, b, m) > 0) out.push_back(m);
        }
    }
}

void World2D::restoreCachedImpulses() {
    stats_.contactCount = 0;
    stats_.manifoldCount = static_cast<int>(manifolds_.size());
    Real deepest = 0;

    for (Manifold& m : manifolds_) {
        stats_.contactCount += m.pointCount;
        for (int i = 0; i < m.pointCount; ++i) {
            ContactPoint& cp = m.points[i];
            deepest = std::max(deepest, -cp.separation);
            const auto it =
                impulseCache_.find(contactKey(m.bodyA, m.bodyB, m.planeIndex, cp.id));
            if (it != impulseCache_.end()) {
                cp.normalImpulse = it->second.normal;
                cp.tangentImpulse = it->second.tangent;
            }
        }
    }
    stats_.maxPenetration = deepest;
}

void World2D::storeImpulses() {
    nextImpulseCache_.clear();
    for (const Manifold& m : manifolds_) {
        for (int i = 0; i < m.pointCount; ++i) {
            const ContactPoint& cp = m.points[i];
            nextImpulseCache_[contactKey(m.bodyA, m.bodyB, m.planeIndex, cp.id)] =
                CachedImpulse{cp.normalImpulse, cp.tangentImpulse};
        }
    }
    impulseCache_.swap(nextImpulseCache_);
}

// ---------------------------------------------------------------- contacts

void World2D::prepareContacts(Real dt) {
    const Real invDt = Real(1) / dt;
    (void)invDt;

    for (Manifold& m : manifolds_) {
        RigidBody2D& a = bodyRef(m.bodyA);
        RigidBody2D& b = bodyRef(m.bodyB);
        const Vec2 tangent = contactTangent(m.normal);

        for (int i = 0; i < m.pointCount; ++i) {
            ContactPoint& cp = m.points[i];
            cp.rA = cp.position - a.position;
            cp.rB = cp.position - b.position;

            const Real rnA = cross(cp.rA, m.normal);
            const Real rnB = cross(cp.rB, m.normal);
            const Real kNormal =
                a.invMass + b.invMass + a.invInertia * rnA * rnA + b.invInertia * rnB * rnB;
            cp.normalMass = kNormal > Real(0) ? Real(1) / kNormal : Real(0);

            const Real rtA = cross(cp.rA, tangent);
            const Real rtB = cross(cp.rB, tangent);
            const Real kTangent =
                a.invMass + b.invMass + a.invInertia * rtA * rtA + b.invInertia * rtB * rtB;
            cp.tangentMass = kTangent > Real(0) ? Real(1) / kTangent : Real(0);

            // Restitution is captured from the approach speed *before* the
            // solver touches it, and suppressed below the threshold so resting
            // bodies do not buzz.
            const Real vn = dot(b.velocityAtOffset(cp.rB) - a.velocityAtOffset(cp.rA), m.normal);
            cp.velocityBias = (vn < -solver.restitutionThreshold) ? -m.restitution * vn : Real(0);
        }
    }
}

void World2D::warmStart() {
    for (Manifold& m : manifolds_) {
        RigidBody2D& a = bodyRef(m.bodyA);
        RigidBody2D& b = bodyRef(m.bodyB);
        const Vec2 tangent = contactTangent(m.normal);
        for (int i = 0; i < m.pointCount; ++i) {
            ContactPoint& cp = m.points[i];
            const Vec2 p = m.normal * cp.normalImpulse + tangent * cp.tangentImpulse;
            a.applyImpulse(-p, cp.rA);
            b.applyImpulse(p, cp.rB);
        }
    }
    for (RevoluteJoint2D& j : joints_) j.warmStart(bodyRef(j.bodyA), bodyRef(j.bodyB));
}

void World2D::solveVelocities(Real dt) {
    const Real invDt = Real(1) / dt;

    for (RevoluteJoint2D& j : joints_) {
        j.solveVelocity(bodyRef(j.bodyA), bodyRef(j.bodyB), solver, dt);
    }

    solveMouseSpring(dt);

    for (Manifold& m : manifolds_) {
        RigidBody2D& a = bodyRef(m.bodyA);
        RigidBody2D& b = bodyRef(m.bodyB);
        const Vec2 tangent = contactTangent(m.normal);

        // Friction first, bounded by the normal impulse accumulated so far, then
        // the normal constraint. Solving friction against a stale-by-one-
        // iteration normal impulse is the standard trade: it converges, and the
        // alternative couples the two constraints into a much larger solve.
        for (int i = 0; i < m.pointCount; ++i) {
            ContactPoint& cp = m.points[i];
            const Real vt = dot(b.velocityAtOffset(cp.rB) - a.velocityAtOffset(cp.rA), tangent);
            Real lambda = cp.tangentMass * (-vt);
            const Real maxFriction = m.friction * cp.normalImpulse;
            const Real previous = cp.tangentImpulse;
            cp.tangentImpulse = clamp(previous + lambda, -maxFriction, maxFriction);
            lambda = cp.tangentImpulse - previous;
            const Vec2 p = tangent * lambda;
            a.applyImpulse(-p, cp.rA);
            b.applyImpulse(p, cp.rB);
        }

        for (int i = 0; i < m.pointCount; ++i) {
            ContactPoint& cp = m.points[i];
            const Real vn = dot(b.velocityAtOffset(cp.rB) - a.velocityAtOffset(cp.rA), m.normal);

            // A still-separated contact is allowed to close at exactly the rate
            // that lands it on the surface this substep and no faster. That is
            // what "speculative" buys: penetration is prevented rather than
            // repaired, so no energy has to be injected to push bodies apart.
            const Real target =
                (cp.separation > Real(0)) ? (-cp.separation * invDt) : cp.velocityBias;

            Real lambda = -cp.normalMass * (vn - target);
            const Real previous = cp.normalImpulse;
            cp.normalImpulse = std::max(previous + lambda, Real(0));
            lambda = cp.normalImpulse - previous;
            const Vec2 p = m.normal * lambda;
            a.applyImpulse(-p, cp.rA);
            b.applyImpulse(p, cp.rB);
        }
    }
}

void World2D::solvePositions() {
    for (RevoluteJoint2D& j : joints_) {
        j.solvePosition(bodyRef(j.bodyA), bodyRef(j.bodyB), solver);
    }

    // Regenerated from the current transforms. Reusing the velocity pass's
    // anchors would correct against geometry that the position pass has already
    // moved, which shows up as bodies creeping into the floor under a stack.
    generateManifolds(positionScratch_);

    for (Manifold& m : positionScratch_) {
        RigidBody2D& a = bodyRef(m.bodyA);
        RigidBody2D& b = bodyRef(m.bodyB);
        for (int i = 0; i < m.pointCount; ++i) {
            const ContactPoint& cp = m.points[i];
            const Vec2 rA = cp.position - a.position;
            const Vec2 rB = cp.position - b.position;

            const Real rnA = cross(rA, m.normal);
            const Real rnB = cross(rB, m.normal);
            const Real kNormal =
                a.invMass + b.invMass + a.invInertia * rnA * rnA + b.invInertia * rnB * rnB;
            if (kNormal <= Real(0)) continue;

            const Real c = clamp(solver.contactBaumgarte * (cp.separation + kLinearSlop),
                                 -solver.maxLinearCorrection, Real(0));
            const Real impulse = -c / kNormal;
            const Vec2 p = m.normal * impulse;

            a.position -= p * a.invMass;
            a.angle -= a.invInertia * cross(rA, p);
            b.position += p * b.invMass;
            b.angle += b.invInertia * cross(rB, p);
        }
    }
}

void World2D::solveMouseSpring(Real dt) {
    if (!mouse_.active || mouse_.body < 0) return;
    RigidBody2D& b = bodies_[static_cast<size_t>(mouse_.body)];
    if (b.invMass == Real(0) && b.invInertia == Real(0)) return;

    const Vec2 r = b.localToWorldDir(mouse_.localAnchor);
    const Vec2 c = (b.position + r) - mouse_.target;

    const Real denom = mouse_.damping + dt * mouse_.stiffness;
    if (denom <= Real(0)) return;
    const Real gamma = Real(1) / (dt * denom);
    const Vec2 bias = c * (mouse_.stiffness / denom);

    Mat2 k;
    k.c0.x = b.invMass + b.invInertia * r.y * r.y + gamma;
    k.c0.y = -b.invInertia * r.x * r.y;
    k.c1.x = k.c0.y;
    k.c1.y = b.invMass + b.invInertia * r.x * r.x + gamma;

    const Vec2 cdot = b.velocityAtOffset(r);
    Vec2 impulse = -(inverse(k) * (cdot + bias + mouse_.accumulated * gamma));

    const Vec2 previous = mouse_.accumulated;
    mouse_.accumulated += impulse;
    const Real maxImpulse = dt * mouse_.maxForce;
    const Real magnitude = length(mouse_.accumulated);
    if (magnitude > maxImpulse) {
        mouse_.accumulated = mouse_.accumulated * (maxImpulse / magnitude);
    }
    impulse = mouse_.accumulated - previous;
    b.applyImpulse(impulse, r);
}

// ---------------------------------------------------------------- queries

void World2D::checkForInstability() {
    for (const RigidBody2D& b : bodies_) {
        if (!b.isFinite()) {
            stats_.unstable = true;
            return;
        }
    }
}

bool World2D::hasContact(int32_t bodyIndex) const {
    for (const Manifold& m : manifolds_) {
        if (m.pointCount == 0) continue;
        if (m.bodyA == bodyIndex || m.bodyB == bodyIndex) return true;
    }
    return false;
}

Vec2 World2D::contactImpulseOn(int32_t bodyIndex) const {
    Vec2 total(0, 0);
    for (const Manifold& m : manifolds_) {
        const Vec2 tangent = contactTangent(m.normal);
        for (int i = 0; i < m.pointCount; ++i) {
            const ContactPoint& cp = m.points[i];
            const Vec2 p = m.normal * cp.normalImpulse + tangent * cp.tangentImpulse;
            if (m.bodyA == bodyIndex) total -= p;
            if (m.bodyB == bodyIndex) total += p;
        }
    }
    return total;
}

Real World2D::totalMass() const {
    Real m = 0;
    for (const RigidBody2D& b : bodies_) {
        if (!b.isStatic) m += b.mass;
    }
    return m;
}

Vec2 World2D::centerOfMass() const {
    Vec2 sum(0, 0);
    Real m = 0;
    for (const RigidBody2D& b : bodies_) {
        if (b.isStatic) continue;
        sum += b.position * b.mass;
        m += b.mass;
    }
    return m > Real(0) ? sum / m : Vec2(0, 0);
}

Vec2 World2D::centerOfMassVelocity() const {
    Vec2 sum(0, 0);
    Real m = 0;
    for (const RigidBody2D& b : bodies_) {
        if (b.isStatic) continue;
        sum += b.velocity * b.mass;
        m += b.mass;
    }
    return m > Real(0) ? sum / m : Vec2(0, 0);
}

Real World2D::kineticEnergy() const {
    Real e = 0;
    for (const RigidBody2D& b : bodies_) {
        if (!b.isStatic) e += b.kineticEnergy();
    }
    return e;
}

Real World2D::potentialEnergy(Real datum) const {
    Real e = 0;
    for (const RigidBody2D& b : bodies_) {
        if (!b.isStatic) e += b.mass * (-gravity.y) * (b.position.y - datum);
    }
    return e;
}

int32_t World2D::pickBody(const Vec2& worldPoint, Real radius) const {
    int32_t best = -1;
    Real bestDistance = radius;
    for (size_t i = 0; i < bodies_.size(); ++i) {
        const RigidBody2D& b = bodies_[i];
        if (b.isStatic) continue;

        // Distance from the point to the capsule's segment, minus its radius.
        const Vec2 p0 = b.endpointA();
        const Vec2 p1 = b.endpointB();
        const Vec2 seg = p1 - p0;
        const Real segLenSq = lengthSq(seg);
        const Real t = segLenSq > Real(1e-12)
                           ? clamp(dot(worldPoint - p0, seg) / segLenSq, Real(0), Real(1))
                           : Real(0);
        const Real distance = length(worldPoint - (p0 + seg * t)) - b.radius;
        if (distance < bestDistance) {
            bestDistance = distance;
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}

void World2D::grab(int32_t bodyIndex, const Vec2& worldPoint) {
    if (bodyIndex < 0 || bodyIndex >= bodyCount()) return;
    mouse_.active = true;
    mouse_.body = bodyIndex;
    mouse_.localAnchor = bodies_[static_cast<size_t>(bodyIndex)].worldToLocal(worldPoint);
    mouse_.target = worldPoint;
    mouse_.accumulated = Vec2(0, 0);
}

}  // namespace aibf
