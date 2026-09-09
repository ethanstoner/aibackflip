#include "humanoid/Observation3D.h"

#include <cmath>

namespace aibf {

namespace {

// The continuous 6D rotation representation: the first two columns of the
// rotation matrix. The third is their cross product, so nothing is lost.
void writeRotation6D(const Quat& q, Real* out) {
    const Mat3 m = toMat3(q);
    out[0] = m.c0.x;
    out[1] = m.c0.y;
    out[2] = m.c0.z;
    out[3] = m.c1.x;
    out[4] = m.c1.y;
    out[5] = m.c1.z;
}

void writeVec3(const Vec3& v, Real* out) {
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
}

}  // namespace

ObservationScales3D ObservationScales3D::fromConfig(const Humanoid3DConfig& config) {
    ObservationScales3D scales;
    scales.pelvisRestHeight = config.links[kPelvis].restPosition.y;
    scales.headRestHeight = config.links[kHead].restPosition.y;
    scales.bodyHeight = config.restHeight();
    return scales;
}

std::vector<std::string> ObservationLayout3D::fieldNames() {
    std::vector<std::string> names;
    names.reserve(static_cast<size_t>(kDimension));

    names.push_back("pelvis_height");
    for (int i = 0; i < 6; ++i) names.push_back("pelvis_rot_" + std::to_string(i));
    for (const char* axis : {"x", "y", "z"}) names.push_back(std::string("pelvis_v") + axis);
    for (const char* axis : {"x", "y", "z"}) names.push_back(std::string("pelvis_w") + axis);

    const Humanoid3DConfig cfg = Humanoid3DConfig::defaults();
    auto forEachNonRootLink = [&](const char* suffix) {
        for (int i = 0; i < kLinkCount; ++i) {
            if (i == kPelvis) continue;
            for (const char* axis : {"x", "y", "z"}) {
                names.push_back(cfg.links[static_cast<size_t>(i)].name + "_" + suffix + "_" + axis);
            }
        }
    };
    forEachNonRootLink("offset");
    forEachNonRootLink("axis");
    forEachNonRootLink("w");

    for (int i = 0; i < kJointCount; ++i) {
        const JointConfig3D& joint = cfg.joints[static_cast<size_t>(i)];
        if (joint.kind != JointKind::Ball) continue;
        for (int k = 0; k < 6; ++k) names.push_back(joint.name + "_rot_" + std::to_string(k));
    }
    for (int i = 0; i < kJointCount; ++i) {
        const JointConfig3D& joint = cfg.joints[static_cast<size_t>(i)];
        if (joint.kind != JointKind::Hinge) continue;
        names.push_back(joint.name + "_sin");
        names.push_back(joint.name + "_cos");
    }
    for (int i = 0; i < kJointCount; ++i) {
        const JointConfig3D& joint = cfg.joints[static_cast<size_t>(i)];
        if (joint.kind != JointKind::Ball) continue;
        for (const char* axis : {"x", "y", "z"}) names.push_back(joint.name + "_w" + axis);
    }
    for (int i = 0; i < kJointCount; ++i) {
        const JointConfig3D& joint = cfg.joints[static_cast<size_t>(i)];
        if (joint.kind != JointKind::Hinge) continue;
        names.push_back(joint.name + "_rate");
    }

    names.push_back("foot_contact_l");
    names.push_back("foot_contact_r");
    for (const char* axis : {"x", "y", "z"}) names.push_back(std::string("com_offset_") + axis);
    for (const char* axis : {"x", "y", "z"}) names.push_back(std::string("com_v") + axis);
    names.push_back("head_height");
    names.push_back("phase_sin");
    names.push_back("phase_cos");

    return names;
}

void writeObservation3D(const World3D& world, const Humanoid3D& figure,
                        const ObservationScales3D& scales, Real phase, Real* out) {
    using L = ObservationLayout3D;

    const RigidBody3D& pelvis = figure.link(world, kPelvis);
    // Offsets are divided by the whole body's height; the pelvis and head are
    // each divided by their own rest height, so both read 1.0 when standing.
    const Real invHeight = scales.bodyHeight > Real(0) ? Real(1) / scales.bodyHeight : Real(1);

    out[L::kPelvisHeight] = pelvis.position.y / scales.pelvisRestHeight;
    writeRotation6D(pelvis.orientation, out + L::kPelvisRotation);
    writeVec3(pelvis.velocity / scales.linearVelocity, out + L::kPelvisLinearVelocity);
    writeVec3(pelvis.angularVelocity / scales.angularVelocity, out + L::kPelvisAngularVelocity);

    int offsetSlot = L::kLinkOffsets;
    int axisSlot = L::kLinkAxes;
    int spinSlot = L::kLinkAngularVelocities;
    for (int i = 0; i < kLinkCount; ++i) {
        if (i == kPelvis) continue;
        const RigidBody3D& body = figure.link(world, i);
        writeVec3((body.position - pelvis.position) * invHeight, out + offsetSlot);
        writeVec3(body.localToWorldDir(Vec3(0, 1, 0)), out + axisSlot);
        writeVec3(body.angularVelocity / scales.angularVelocity, out + spinSlot);
        offsetSlot += 3;
        axisSlot += 3;
        spinSlot += 3;
    }

    const Humanoid3DConfig& cfg = figure.config();
    int ballSlot = L::kBallRotations;
    int hingeSlot = L::kHingeAngles;
    int ballRateSlot = L::kJointVelocities;
    int hingeRateSlot = L::kJointVelocities + 3 * L::kBallJoints;

    for (int i = 0; i < kJointCount; ++i) {
        const JointConfig3D& jc = cfg.joints[static_cast<size_t>(i)];
        if (jc.kind != JointKind::Ball) continue;
        const BallJoint3D& joint = world.ballJoint(figure.ballIndex(i));
        const RigidBody3D& parent = world.body(joint.bodyA);
        const RigidBody3D& child = world.body(joint.bodyB);
        writeRotation6D(joint.relativeRotation(parent, child), out + ballSlot);
        // Relative angular velocity expressed in the parent's frame, so the
        // number means the same thing however the figure is oriented.
        const Vec3 relative = rotateInverse(parent.orientation,
                                            child.angularVelocity - parent.angularVelocity);
        writeVec3(relative / scales.jointVelocity, out + ballRateSlot);
        ballSlot += 6;
        ballRateSlot += 3;
    }

    for (int i = 0; i < kJointCount; ++i) {
        const JointConfig3D& jc = cfg.joints[static_cast<size_t>(i)];
        if (jc.kind != JointKind::Hinge) continue;
        const HingeJoint3D& joint = world.hingeJoint(figure.hingeIndex(i));
        const RigidBody3D& parent = world.body(joint.bodyA);
        const RigidBody3D& child = world.body(joint.bodyB);
        const Real angle = joint.jointAngle(parent, child);
        out[hingeSlot + 0] = std::sin(angle);
        out[hingeSlot + 1] = std::cos(angle);
        out[hingeRateSlot] = joint.jointRate(parent, child) / scales.jointVelocity;
        hingeSlot += 2;
        hingeRateSlot += 1;
    }

    out[L::kFootContacts + 0] = figure.footInContact(world, kFootL) ? Real(1) : Real(0);
    out[L::kFootContacts + 1] = figure.footInContact(world, kFootR) ? Real(1) : Real(0);

    const Vec3 com = figure.centerOfMass(world);
    writeVec3((com - pelvis.position) * invHeight, out + L::kComOffset);
    writeVec3(figure.centerOfMassVelocity(world) / scales.linearVelocity, out + L::kComVelocity);

    out[L::kHeadHeight] = figure.link(world, kHead).position.y / scales.headRestHeight;

    // Phase on the circle, so the wrap from 1 back to 0 is not a discontinuity
    // in the input.
    out[L::kPhase + 0] = std::sin(phase * Real(2) * kPi);
    out[L::kPhase + 1] = std::cos(phase * Real(2) * kPi);
}

}  // namespace aibf
