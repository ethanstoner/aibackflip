#include "humanoid/Humanoid2D.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aibf {

void Humanoid2D::build(World2D& world, const Humanoid2DConfig& config) {
    config_ = config;
    bodyIndices_.assign(config_.links.size(), -1);
    jointIndices_.assign(config_.joints.size(), -1);

    for (size_t i = 0; i < config_.links.size(); ++i) {
        const LinkConfig& link = config_.links[i];
        RigidBody2D body;
        body.position = link.restPosition;
        body.angle = link.restAngle;
        body.setCapsuleWithMass(link.radius, link.halfLength, link.mass);
        body.friction = link.friction;
        body.restitution = link.restitution;
        body.collisionGroup = config_.collisionGroup;
        bodyIndices_[i] = world.addBody(body);
    }

    jointFrames_.assign(config_.joints.size(), JointFrame{});
    for (size_t i = 0; i < config_.joints.size(); ++i) {
        const JointConfig& jc = config_.joints[i];
        const LinkConfig& parent = config_.links[static_cast<size_t>(jc.parent)];
        const LinkConfig& child = config_.links[static_cast<size_t>(jc.child)];

        RevoluteJoint2D joint;
        joint.bodyA = bodyIndices_[static_cast<size_t>(jc.parent)];
        joint.bodyB = bodyIndices_[static_cast<size_t>(jc.child)];
        // Local anchors are the rest-pose pivot expressed in each link's frame.
        joint.localAnchorA = rotate(jc.restAnchor - parent.restPosition, -parent.restAngle);
        joint.localAnchorB = rotate(jc.restAnchor - child.restPosition, -child.restAngle);
        // Chosen so the rest pose reads as a relative angle of exactly zero.
        joint.referenceAngle = child.restAngle - parent.restAngle;
        joint.enableLimit = true;
        joint.lowerAngle = jc.lowerLimit;
        joint.upperAngle = jc.upperLimit;
        joint.enableMotor = false;
        joint.targetAngle = 0;
        joint.stiffness = jc.stiffness;
        joint.damping = jc.damping;
        joint.maxTorque = jc.maxTorque;
        jointIndices_[i] = world.addJoint(joint);

        jointFrames_[i] = JointFrame{jc.parent, jc.child, joint.localAnchorA, joint.localAnchorB,
                                     joint.referenceAngle};
    }

    // Topological order: a joint may be applied once its parent link has been
    // placed. The pelvis is the root and needs no joint.
    kinematicOrder_.clear();
    std::vector<bool> placed(config_.links.size(), false);
    placed[kPelvis] = true;
    std::vector<bool> used(config_.joints.size(), false);
    for (size_t pass = 0; pass < config_.joints.size(); ++pass) {
        bool progressed = false;
        for (size_t i = 0; i < config_.joints.size(); ++i) {
            if (used[i]) continue;
            const JointConfig& jc = config_.joints[i];
            if (!placed[static_cast<size_t>(jc.parent)]) continue;
            kinematicOrder_.push_back(static_cast<int>(i));
            placed[static_cast<size_t>(jc.child)] = true;
            used[i] = true;
            progressed = true;
        }
        if (!progressed) break;
    }
}

// ---------------------------------------------------------------- pose

void Humanoid2D::forwardKinematics(Vec2 rootPosition, Real rootAngle, const Real* jointAngles,
                                   Vec2* linkPositions, Real* linkAngles) const {
    std::vector<Vec2> positions(config_.links.size());
    std::vector<Real> angles(config_.links.size(), Real(0));

    positions[kPelvis] = rootPosition;
    angles[kPelvis] = rootAngle;

    for (const int jointSlot : kinematicOrder_) {
        const JointFrame& frame = jointFrames_[static_cast<size_t>(jointSlot)];
        const size_t parent = static_cast<size_t>(frame.parent);
        const size_t child = static_cast<size_t>(frame.child);

        const Real relative = jointAngles ? jointAngles[jointSlot] : Real(0);
        angles[child] = angles[parent] + frame.referenceAngle + relative;
        const Vec2 anchorWorld = positions[parent] + rotate(frame.localAnchorParent, angles[parent]);
        positions[child] = anchorWorld - rotate(frame.localAnchorChild, angles[child]);
    }

    if (linkPositions) {
        for (size_t i = 0; i < positions.size(); ++i) linkPositions[i] = positions[i];
    }
    if (linkAngles) {
        for (size_t i = 0; i < angles.size(); ++i) linkAngles[i] = angles[i];
    }
}

void Humanoid2D::forwardKinematicsVelocity(const Vec2* linkPositions, const Real* linkAngles,
                                           Vec2 rootVelocity, Real rootAngularVelocity,
                                           const Real* jointVelocities, Vec2* linkVelocities,
                                           Real* linkAngularVelocities) const {
    std::vector<Vec2> velocities(config_.links.size(), Vec2(0, 0));
    std::vector<Real> angularVelocities(config_.links.size(), Real(0));

    velocities[kPelvis] = rootVelocity;
    angularVelocities[kPelvis] = rootAngularVelocity;

    for (const int jointSlot : kinematicOrder_) {
        const JointFrame& frame = jointFrames_[static_cast<size_t>(jointSlot)];
        const size_t parent = static_cast<size_t>(frame.parent);
        const size_t child = static_cast<size_t>(frame.child);

        const Real relative = jointVelocities ? jointVelocities[jointSlot] : Real(0);
        angularVelocities[child] = angularVelocities[parent] + relative;

        // The shared anchor has one velocity; both bodies must agree on it.
        const Vec2 anchorWorld =
            linkPositions[parent] + rotate(frame.localAnchorParent, linkAngles[parent]);
        const Vec2 rParent = anchorWorld - linkPositions[parent];
        const Vec2 rChild = anchorWorld - linkPositions[child];
        const Vec2 anchorVelocity = velocities[parent] + cross(angularVelocities[parent], rParent);
        velocities[child] = anchorVelocity - cross(angularVelocities[child], rChild);
    }

    if (linkVelocities) {
        for (size_t i = 0; i < velocities.size(); ++i) linkVelocities[i] = velocities[i];
    }
    if (linkAngularVelocities) {
        for (size_t i = 0; i < angularVelocities.size(); ++i) {
            linkAngularVelocities[i] = angularVelocities[i];
        }
    }
}

void Humanoid2D::setPoseAndVelocity(World2D& world, Vec2 rootPosition, Real rootAngle,
                                    const Real* jointAngles, Vec2 rootVelocity,
                                    Real rootAngularVelocity,
                                    const Real* jointVelocities) const {
    const size_t links = config_.links.size();
    std::vector<Vec2> positions(links);
    std::vector<Real> angles(links);
    forwardKinematics(rootPosition, rootAngle, jointAngles, positions.data(), angles.data());

    std::vector<Vec2> velocities(links);
    std::vector<Real> angularVelocities(links);
    forwardKinematicsVelocity(positions.data(), angles.data(), rootVelocity, rootAngularVelocity,
                              jointVelocities, velocities.data(), angularVelocities.data());

    for (size_t i = 0; i < links; ++i) {
        RigidBody2D& body = link(world, static_cast<int>(i));
        body.position = positions[i];
        body.angle = angles[i];
        body.velocity = velocities[i];
        body.angularVelocity = angularVelocities[i];
        body.clearForces();
    }
}

Real Humanoid2D::groundedRootHeight(Real rootAngle, const Real* jointAngles) const {
    const size_t links = config_.links.size();
    std::vector<Vec2> positions(links);
    std::vector<Real> angles(links);
    forwardKinematics(Vec2(0, 0), rootAngle, jointAngles, positions.data(), angles.data());

    Real lowest = std::numeric_limits<Real>::max();
    for (size_t i = 0; i < links; ++i) {
        const LinkConfig& link = config_.links[i];
        // A capsule's lowest point is the lower of its two end caps.
        const Vec2 axis = rotate(Vec2(0, link.halfLength), angles[i]);
        const Real bottom =
            std::min(positions[i].y + axis.y, positions[i].y - axis.y) - link.radius;
        lowest = std::min(lowest, bottom);
    }
    return -lowest;
}

void Humanoid2D::setPose(World2D& world, Vec2 rootPosition, Real rootAngle,
                         const Real* jointAngles) const {
    RigidBody2D& pelvis = link(world, kPelvis);
    pelvis.position = rootPosition;
    pelvis.angle = rootAngle;
    pelvis.velocity = Vec2(0, 0);
    pelvis.angularVelocity = 0;
    pelvis.clearForces();

    for (const int jointSlot : kinematicOrder_) {
        const JointConfig& jc = config_.joints[static_cast<size_t>(jointSlot)];
        const RevoluteJoint2D& rj = joint(world, jointSlot);
        const RigidBody2D& parent = link(world, jc.parent);
        RigidBody2D& child = link(world, jc.child);

        const Real relative = jointAngles ? jointAngles[jointSlot] : Real(0);
        child.angle = parent.angle + rj.referenceAngle + relative;
        // Put the child where its own anchor lands on the parent's anchor.
        const Vec2 anchorWorld = parent.position + rotate(rj.localAnchorA, parent.angle);
        child.position = anchorWorld - rotate(rj.localAnchorB, child.angle);
        child.velocity = Vec2(0, 0);
        child.angularVelocity = 0;
        child.clearForces();
    }
}

void Humanoid2D::reset(World2D& world, Vec2 rootPosition) const {
    setPose(world, rootPosition, 0, nullptr);
    for (size_t i = 0; i < config_.joints.size(); ++i) {
        RevoluteJoint2D& rj = joint(world, static_cast<int>(i));
        rj.targetAngle = 0;
        rj.resetAccumulators();
    }
    // Stale warm-start impulses belong to the previous episode's contacts and
    // would be applied to a body that has just been teleported.
    world.clearContactCache();
    world.resetStats();
}

void Humanoid2D::reset(World2D& world, Vec2 rootPosition, const ResetNoise& noise,
                       Rng& rng) const {
    std::vector<Real> angles(config_.joints.size(), Real(0));
    for (size_t i = 0; i < angles.size(); ++i) {
        const JointConfig& jc = config_.joints[i];
        angles[i] = clamp(rng.uniform(-noise.jointAngle, noise.jointAngle), jc.lowerLimit,
                          jc.upperLimit);
    }

    const Vec2 position = rootPosition + Vec2(0, rng.uniform(-noise.rootHeight, noise.rootHeight));
    setPose(world, position, rng.uniform(-noise.rootAngle, noise.rootAngle), angles.data());

    if (noise.linearVelocity > Real(0) || noise.angularVelocity > Real(0)) {
        const Vec2 v(rng.uniform(-noise.linearVelocity, noise.linearVelocity),
                     rng.uniform(-noise.linearVelocity, noise.linearVelocity));
        const Real w = rng.uniform(-noise.angularVelocity, noise.angularVelocity);
        // Applied to the whole figure so it starts as a rigid drift rather than
        // an internal explosion the joints immediately have to absorb.
        for (size_t i = 0; i < config_.links.size(); ++i) {
            RigidBody2D& body = link(world, static_cast<int>(i));
            body.velocity = v + cross(w, body.position - position);
            body.angularVelocity = w;
        }
    }

    for (size_t i = 0; i < config_.joints.size(); ++i) {
        RevoluteJoint2D& rj = joint(world, static_cast<int>(i));
        rj.targetAngle = angles[i];
        rj.resetAccumulators();
    }
    world.clearContactCache();
    world.resetStats();
}

Real Humanoid2D::jointAngle(const World2D& world, int jointId) const {
    const RevoluteJoint2D& rj = joint(world, jointId);
    return rj.relativeAngle(world.body(rj.bodyA), world.body(rj.bodyB));
}

Real Humanoid2D::jointVelocity(const World2D& world, int jointId) const {
    const RevoluteJoint2D& rj = joint(world, jointId);
    return rj.relativeAngularVelocity(world.body(rj.bodyA), world.body(rj.bodyB));
}

// ---------------------------------------------------------------- actuation

void Humanoid2D::setJointTarget(World2D& world, int jointId, Real angle) const {
    const JointConfig& jc = config_.joints[static_cast<size_t>(jointId)];
    joint(world, jointId).targetAngle = clamp(angle, jc.lowerLimit, jc.upperLimit);
}

void Humanoid2D::setJointTargetNormalized(World2D& world, int jointId, Real action) const {
    const JointConfig& jc = config_.joints[static_cast<size_t>(jointId)];
    const Real a = clamp(action, Real(-1), Real(1));

    // Piecewise-linear about the rest pose rather than linear across the range:
    // action 0 commands the rest angle, +1 the upper limit, -1 the lower.
    //
    // The obvious mapping - lerp(lower, upper) - puts action 0 at the middle of
    // each range, which for the knee's [-2.6, 0.05] is 1.3 rad of flexion. A
    // freshly initialised policy outputs values near zero, so every early
    // rollout would begin by folding the figure into a deep squat and learning
    // would have to climb out of that before it could start. Centring on the
    // rest pose costs nothing and makes "do nothing" mean "stand".
    //
    // Joint limits are validated to straddle zero, so both branches are usable.
    const Real target = (a >= Real(0)) ? a * jc.upperLimit : -a * jc.lowerLimit;
    joint(world, jointId).targetAngle = clamp(target, jc.lowerLimit, jc.upperLimit);
}

void Humanoid2D::applyNormalizedActions(World2D& world, const Real* actions, int count) const {
    const int n = std::min(count, static_cast<int>(config_.joints.size()));
    for (int i = 0; i < n; ++i) setJointTargetNormalized(world, i, actions[i]);
}

void Humanoid2D::setMotorsEnabled(World2D& world, bool enabled) const {
    for (size_t i = 0; i < config_.joints.size(); ++i) {
        joint(world, static_cast<int>(i)).enableMotor = enabled;
    }
}

void Humanoid2D::holdRestPose(World2D& world) const {
    for (size_t i = 0; i < config_.joints.size(); ++i) {
        joint(world, static_cast<int>(i)).targetAngle = 0;
    }
}

// ---------------------------------------------------------------- queries

Vec2 Humanoid2D::centerOfMass(const World2D& world) const {
    Vec2 sum(0, 0);
    Real mass = 0;
    for (size_t i = 0; i < config_.links.size(); ++i) {
        const RigidBody2D& body = link(world, static_cast<int>(i));
        sum += body.position * body.mass;
        mass += body.mass;
    }
    return mass > Real(0) ? sum / mass : Vec2(0, 0);
}

Vec2 Humanoid2D::centerOfMassVelocity(const World2D& world) const {
    Vec2 sum(0, 0);
    Real mass = 0;
    for (size_t i = 0; i < config_.links.size(); ++i) {
        const RigidBody2D& body = link(world, static_cast<int>(i));
        sum += body.velocity * body.mass;
        mass += body.mass;
    }
    return mass > Real(0) ? sum / mass : Vec2(0, 0);
}

Real Humanoid2D::headHeight(const World2D& world) const {
    const RigidBody2D& head = link(world, kHead);
    return head.position.y + head.halfLength + head.radius;
}

Real Humanoid2D::uprightness(const World2D& world, int linkId) const {
    const RigidBody2D& body = link(world, linkId);
    const Real rest = config_.links[static_cast<size_t>(linkId)].restAngle;
    // Compare against the link's own rest orientation, so a foot (which rests
    // horizontal) is not permanently scored as fallen over.
    return std::cos(body.angle - rest);
}

bool Humanoid2D::footContact(const World2D& world, bool left) const {
    return world.hasContact(bodyIndex(left ? kFootL : kFootR));
}

Real Humanoid2D::worstLimitViolation(const World2D& world) const {
    Real worst = 0;
    for (size_t i = 0; i < config_.joints.size(); ++i) {
        const JointConfig& jc = config_.joints[i];
        const Real angle = jointAngle(world, static_cast<int>(i));
        worst = std::max(worst, std::max(angle - jc.upperLimit, jc.lowerLimit - angle));
    }
    return std::max(worst, Real(0));
}

}  // namespace aibf
