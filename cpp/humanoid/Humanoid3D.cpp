#include "humanoid/Humanoid3D.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace aibf {

void Humanoid3D::build(World3D& world, const Humanoid3DConfig& config,
                       const Vec3& rootPosition) {
    config_ = config;
    restHeight_ = config.restHeight();
    bodyIndex_.assign(kLinkCount, -1);
    ballIndex_.assign(kJointCount, -1);
    hingeIndex_.assign(kJointCount, -1);

    // The rest pose is authored with the pelvis at its own rest position, so the
    // whole figure is translated by the difference rather than each link being
    // re-authored.
    const Vec3 offset = rootPosition - config.links[kPelvis].restPosition;

    for (int i = 0; i < kLinkCount; ++i) {
        const LinkConfig3D& link = config.links[static_cast<size_t>(i)];
        RigidBody3D body;
        body.setCapsuleWithMass(link.radius, link.halfLength, link.mass);
        body.position = link.restPosition + offset;
        body.orientation = link.restOrientation;
        body.friction = link.friction;
        body.restitution = link.restitution;
        // Every part of the figure shares one collision group, so jointed
        // neighbours do not fight their own constraints. In 3D this is a
        // simplification rather than a necessity, and it is the first thing to
        // revisit if limbs start passing through each other visibly.
        body.collisionGroup = 1;
        bodyIndex_[static_cast<size_t>(i)] = world.addBody(body);
    }

    for (int i = 0; i < kJointCount; ++i) {
        const JointConfig3D& jc = config.joints[static_cast<size_t>(i)];
        const RigidBody3D& parent = link(world, jc.parent);
        const RigidBody3D& child = link(world, jc.child);
        const Vec3 anchor = jc.restAnchor + offset;

        if (jc.kind == JointKind::Ball) {
            BallJoint3D joint;
            joint.bodyA = bodyIndex(jc.parent);
            joint.bodyB = bodyIndex(jc.child);
            joint.localAnchorA = parent.worldToLocal(anchor);
            joint.localAnchorB = child.worldToLocal(anchor);
            // Derived, not authored: this is what makes "every joint reads zero
            // at rest" true by construction.
            joint.referenceRotation = conjugate(parent.orientation) * child.orientation;
            joint.localAxisA = Vec3(0, 1, 0);
            joint.localAxisB = Vec3(0, 1, 0);
            joint.enableConeLimit = true;
            joint.coneAngle = jc.coneAngle;
            joint.enableTwistLimit = true;
            joint.lowerTwist = jc.lowerTwist;
            joint.upperTwist = jc.upperTwist;
            joint.enableMotor = jc.stiffness > Real(0);
            joint.targetRotation = Quat::identity();
            joint.stiffness = jc.stiffness;
            joint.damping = jc.damping;
            joint.maxTorque = jc.maxTorque;
            ballIndex_[static_cast<size_t>(i)] = world.addBallJoint(joint);
        } else {
            HingeJoint3D joint;
            joint.bodyA = bodyIndex(jc.parent);
            joint.bodyB = bodyIndex(jc.child);
            joint.localAnchorA = parent.worldToLocal(anchor);
            joint.localAnchorB = child.worldToLocal(anchor);
            joint.referenceRotation = conjugate(parent.orientation) * child.orientation;
            // The hinge axis is given in world terms in the rest pose, so both
            // bodies express it in their own frames and it lines up by
            // construction however the limb happens to be oriented at rest. A
            // foot is rotated 90 degrees, so authoring its axis in its own frame
            // by hand would be a standing invitation to get it wrong.
            joint.localHingeAxisA = parent.worldToLocalDir(jc.hingeAxis);
            joint.localHingeAxisB = child.worldToLocalDir(jc.hingeAxis);
            joint.enableLimit = true;
            joint.lowerAngle = jc.lowerLimit;
            joint.upperAngle = jc.upperLimit;
            joint.enableMotor = jc.stiffness > Real(0);
            joint.targetAngle = 0;
            joint.stiffness = jc.stiffness;
            joint.damping = jc.damping;
            joint.maxTorque = jc.maxTorque;
            hingeIndex_[static_cast<size_t>(i)] = world.addHingeJoint(joint);
        }
    }
}

void Humanoid3D::setJointTargetsNormalized(World3D& world, const Real* actions,
                                           int count) const {
    if (actions == nullptr) return;

    for (int i = 0; i < kJointCount; ++i) {
        const JointConfig3D& jc = config_.joints[static_cast<size_t>(i)];
        const int offset = config_.actionOffset(i);
        if (offset + jc.dof() > count) break;

        if (jc.kind == JointKind::Ball) {
            // Swing and twist are scaled and clamped separately, then composed,
            // because they are separately limited. Treating the three action
            // values as one rotation vector and clamping its magnitude to the
            // cone would let a large twist eat into the swing budget, and a
            // shoulder would lose reach as it rotated.
            //
            // The clamp matters: with per-axis ranges sized so a single-axis
            // action reaches the cone, a diagonal action asks for sqrt(2) times
            // as much. Left unclamped the policy spends part of its action
            // range pushing into a limit that will simply refuse it, which is
            // action resolution thrown away rather than a stability problem.
            Vec3 swing(clamp(actions[offset + 0], Real(-1), Real(1)) * jc.targetRange.x,
                       Real(0),
                       clamp(actions[offset + 2], Real(-1), Real(1)) * jc.targetRange.z);
            const Real swingMagnitude = length(swing);
            if (swingMagnitude > jc.coneAngle && swingMagnitude > kEpsilon) {
                swing = swing * (jc.coneAngle / swingMagnitude);
            }

            const Real twist = clamp(clamp(actions[offset + 1], Real(-1), Real(1)) *
                                         jc.targetRange.y,
                                     jc.lowerTwist, jc.upperTwist);

            world.ballJoint(ballIndex_[static_cast<size_t>(i)]).targetRotation =
                quatFromRotationVector(swing) *
                Quat::fromAxisAngle(Vec3(0, 1, 0), twist);
        } else {
            // Piecewise about the rest pose, matching Humanoid2D. A knee's range
            // is roughly -2.6 to +0.05, so mapping [-1, 1] linearly onto that
            // span would leave the policy with almost no resolution where the
            // joint actually works.
            const Real a = clamp(actions[offset], Real(-1), Real(1));
            const Real target = (a >= Real(0)) ? a * jc.upperLimit : -a * jc.lowerLimit;
            world.hingeJoint(hingeIndex_[static_cast<size_t>(i)]).targetAngle =
                clamp(target, jc.lowerLimit, jc.upperLimit);
        }
    }
}

void Humanoid3D::relaxToRestPose(World3D& world) const {
    for (int i = 0; i < kJointCount; ++i) {
        if (ballIndex_[static_cast<size_t>(i)] >= 0) {
            world.ballJoint(ballIndex_[static_cast<size_t>(i)]).targetRotation = Quat::identity();
        }
        if (hingeIndex_[static_cast<size_t>(i)] >= 0) {
            world.hingeJoint(hingeIndex_[static_cast<size_t>(i)]).targetAngle = 0;
        }
    }
}

void Humanoid3D::setPose(World3D& world, const Vec3& rootPosition,
                         const Quat& rootOrientation) const {
    const Vec3 restRoot = config_.links[kPelvis].restPosition;
    for (int i = 0; i < kLinkCount; ++i) {
        const LinkConfig3D& lc = config_.links[static_cast<size_t>(i)];
        RigidBody3D& body = link(world, i);
        // Rigidly transform the whole rest pose, so relative geometry survives
        // and the joints stay satisfied.
        body.position = rootPosition + rotate(rootOrientation, lc.restPosition - restRoot);
        body.orientation = rootOrientation * lc.restOrientation;
        body.refreshInertiaWorld();
    }
}


namespace {

// Anchor and reference constants for one joint, whichever kind it is. These were
// derived from the rest pose at build time and never change, which is what lets
// forward kinematics run without touching the live figure's state.
struct JointFrame {
    Vec3 localAnchorParent;
    Vec3 localAnchorChild;
    Quat referenceRotation;
};

JointFrame frameOf(const World3D& world, const Humanoid3D& figure, int jointId) {
    JointFrame frame;
    const int32_t ball = figure.ballIndex(jointId);
    if (ball >= 0) {
        const BallJoint3D& joint = world.ballJoint(ball);
        frame.localAnchorParent = joint.localAnchorA;
        frame.localAnchorChild = joint.localAnchorB;
        frame.referenceRotation = joint.referenceRotation;
        return frame;
    }
    const HingeJoint3D& joint = world.hingeJoint(figure.hingeIndex(jointId));
    frame.localAnchorParent = joint.localAnchorA;
    frame.localAnchorChild = joint.localAnchorB;
    frame.referenceRotation = joint.referenceRotation;
    return frame;
}

}  // namespace

void Humanoid3D::forwardKinematics(const World3D& world, const Vec3& rootPosition,
                                   const Quat& rootOrientation, const Quat* jointRotations,
                                   Vec3* outPositions, Quat* outOrientations) const {
    outPositions[kPelvis] = rootPosition;
    outOrientations[kPelvis] = rootOrientation;

    // The joint order is parent-before-child by construction of the JointId
    // enum, so one pass in order is a full tree walk.
    for (int j = 0; j < kJointCount; ++j) {
        const JointConfig3D& jc = config_.joints[static_cast<size_t>(j)];
        const JointFrame frame = frameOf(world, *this, j);
        const Quat relative = jointRotations ? jointRotations[j] : Quat::identity();

        // relativeRotation() is conj(parent) * child * conj(reference), so the
        // child's orientation is the inverse of that relation. Deriving it here
        // rather than writing it down means the two cannot disagree.
        const Quat parentOrientation = outOrientations[jc.parent];
        const Quat childOrientation = parentOrientation * relative * frame.referenceRotation;

        const Vec3 anchorWorld =
            outPositions[jc.parent] + rotate(parentOrientation, frame.localAnchorParent);
        outOrientations[jc.child] = childOrientation;
        outPositions[jc.child] = anchorWorld - rotate(childOrientation, frame.localAnchorChild);
    }
}

void Humanoid3D::forwardKinematicsVelocity(const World3D& world, const Vec3& rootVelocity,
                                           const Vec3& rootAngularVelocity, const Vec3* jointRates,
                                           const Vec3* positions, const Quat* orientations,
                                           Vec3* outVelocities,
                                           Vec3* outAngularVelocities) const {
    outVelocities[kPelvis] = rootVelocity;
    outAngularVelocities[kPelvis] = rootAngularVelocity;

    for (int j = 0; j < kJointCount; ++j) {
        const JointConfig3D& jc = config_.joints[static_cast<size_t>(j)];
        const JointFrame frame = frameOf(world, *this, j);

        // The joint rate arrives in the parent's frame, matching how the
        // observation and the reward report it.
        const Vec3 relativeSpin =
            jointRates ? rotate(orientations[jc.parent], jointRates[j]) : Vec3(0, 0, 0);
        outAngularVelocities[jc.child] = outAngularVelocities[jc.parent] + relativeSpin;

        // Both bodies share the anchor point, so its velocity computed from the
        // parent is also its velocity computed from the child.
        const Vec3 anchorWorld =
            positions[jc.parent] + rotate(orientations[jc.parent], frame.localAnchorParent);
        const Vec3 anchorVelocity =
            outVelocities[jc.parent] +
            cross(outAngularVelocities[jc.parent], anchorWorld - positions[jc.parent]);
        outVelocities[jc.child] =
            anchorVelocity - cross(outAngularVelocities[jc.child], anchorWorld - positions[jc.child]);
    }
}

void Humanoid3D::applyPose(World3D& world, const Vec3* positions, const Quat* orientations,
                           const Vec3* velocities, const Vec3* angularVelocities) const {
    for (int i = 0; i < kLinkCount; ++i) {
        RigidBody3D& body = link(world, i);
        body.position = positions[i];
        body.orientation = normalize(orientations[i]);
        body.velocity = velocities ? velocities[i] : Vec3(0, 0, 0);
        body.angularVelocity = angularVelocities ? angularVelocities[i] : Vec3(0, 0, 0);
        body.clearForces();
        body.refreshInertiaWorld();
    }
}

Real Humanoid3D::groundedRootHeight(const World3D& world, const Quat& rootOrientation,
                                    const Quat* jointRotations) const {
    std::vector<Vec3> positions(kLinkCount);
    std::vector<Quat> orientations(kLinkCount);
    forwardKinematics(world, Vec3(0, 0, 0), rootOrientation, jointRotations, positions.data(),
                      orientations.data());

    Real lowest = std::numeric_limits<Real>::max();
    for (int i = 0; i < kLinkCount; ++i) {
        const LinkConfig3D& lc = config_.links[static_cast<size_t>(i)];
        const Vec3 axis = rotate(orientations[i], Vec3(0, lc.halfLength, 0));
        lowest = std::min(lowest, positions[i].y - std::abs(axis.y) - lc.radius);
    }
    return -lowest;
}

Vec3 Humanoid3D::centerOfMass(const World3D& world) const {
    Vec3 weighted(0, 0, 0);
    Real total = 0;
    for (int i = 0; i < kLinkCount; ++i) {
        const RigidBody3D& body = link(world, i);
        weighted += body.position * body.mass;
        total += body.mass;
    }
    return total > Real(0) ? weighted / total : Vec3(0, 0, 0);
}

Vec3 Humanoid3D::centerOfMassVelocity(const World3D& world) const {
    Vec3 weighted(0, 0, 0);
    Real total = 0;
    for (int i = 0; i < kLinkCount; ++i) {
        const RigidBody3D& body = link(world, i);
        weighted += body.velocity * body.mass;
        total += body.mass;
    }
    return total > Real(0) ? weighted / total : Vec3(0, 0, 0);
}

Real Humanoid3D::lowestPoint(const World3D& world) const {
    Real lowest = std::numeric_limits<Real>::max();
    for (int i = 0; i < kLinkCount; ++i) {
        const RigidBody3D& body = link(world, i);
        const Vec3 a = body.endpointA();
        const Vec3 b = body.endpointB();
        lowest = std::min(lowest, std::min(a.y, b.y) - body.radius);
    }
    return lowest;
}

}  // namespace aibf
