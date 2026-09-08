#include "humanoid/Observation.h"

#include <algorithm>
#include <cmath>

namespace aibf {

namespace {

// Every link except the pelvis, in a fixed order. The pelvis is the reference
// frame, so including it would just add a block of constant zeros.
constexpr int kNonRootOrder[ObservationLayout::kNonRootLinks] = {
    kChest,     kHead,      kUpperArmL, kLowerArmL, kUpperArmR, kLowerArmR,
    kUpperLegL, kLowerLegL, kFootL,     kUpperLegR, kLowerLegR, kFootR,
};

const char* linkShortName(int linkId) {
    switch (linkId) {
        case kPelvis: return "pelvis";
        case kChest: return "chest";
        case kHead: return "head";
        case kUpperArmL: return "upper_arm_l";
        case kLowerArmL: return "lower_arm_l";
        case kUpperArmR: return "upper_arm_r";
        case kLowerArmR: return "lower_arm_r";
        case kUpperLegL: return "upper_leg_l";
        case kLowerLegL: return "lower_leg_l";
        case kFootL: return "foot_l";
        case kUpperLegR: return "upper_leg_r";
        case kLowerLegR: return "lower_leg_r";
        case kFootR: return "foot_r";
        default: return "link";
    }
}

const char* jointShortName(int jointId) {
    static const char* kNames[kJointCount] = {
        "waist", "neck",  "shoulder_l", "elbow_l", "shoulder_r", "elbow_r",
        "hip_l", "knee_l", "ankle_l",   "hip_r",   "knee_r",     "ankle_r",
    };
    return (jointId >= 0 && jointId < kJointCount) ? kNames[jointId] : "joint";
}

}  // namespace

ObservationScales ObservationScales::fromConfig(const Humanoid2DConfig& config) {
    ObservationScales scales;
    scales.pelvisRestHeight = std::max(config.restPelvisHeight(), Real(1e-3));
    scales.headRestHeight = std::max(config.restHeadHeight(), Real(1e-3));
    scales.bodyHeight = scales.headRestHeight;
    return scales;
}

std::vector<std::string> ObservationLayout::fieldNames() {
    std::vector<std::string> names;
    names.reserve(kDimension);

    names.push_back("pelvis_height");
    names.push_back("pelvis_sin");
    names.push_back("pelvis_cos");
    names.push_back("pelvis_vx");
    names.push_back("pelvis_vy");
    names.push_back("pelvis_w");

    for (const int link : kNonRootOrder) {
        names.push_back(std::string(linkShortName(link)) + "_dx");
        names.push_back(std::string(linkShortName(link)) + "_dy");
    }
    for (const int link : kNonRootOrder) {
        names.push_back(std::string(linkShortName(link)) + "_sin");
        names.push_back(std::string(linkShortName(link)) + "_cos");
    }
    for (const int link : kNonRootOrder) {
        names.push_back(std::string(linkShortName(link)) + "_w");
    }
    for (int j = 0; j < kJointCount; ++j) {
        names.push_back(std::string(jointShortName(j)) + "_sin");
        names.push_back(std::string(jointShortName(j)) + "_cos");
    }
    for (int j = 0; j < kJointCount; ++j) {
        names.push_back(std::string(jointShortName(j)) + "_vel");
    }
    names.push_back("contact_foot_l");
    names.push_back("contact_foot_r");
    names.push_back("com_dx");
    names.push_back("com_dy");
    names.push_back("com_vx");
    names.push_back("com_vy");
    names.push_back("head_height");
    names.push_back("phase_sin");
    names.push_back("phase_cos");

    return names;
}

void writeObservation(const World2D& world, const Humanoid2D& figure,
                      const ObservationScales& scales, Real phase, Real* out) {
    const RigidBody2D& pelvis = figure.link(world, kPelvis);
    const Real invBody = Real(1) / scales.bodyHeight;

    int i = ObservationLayout::kPelvisHeight;
    out[i++] = pelvis.position.y / scales.pelvisRestHeight;
    out[i++] = std::sin(pelvis.angle);
    out[i++] = std::cos(pelvis.angle);
    out[i++] = pelvis.velocity.x;
    out[i++] = pelvis.velocity.y;
    out[i++] = pelvis.angularVelocity;

    // Link positions relative to the pelvis. This is what removes absolute X
    // from the observation entirely.
    for (const int link : kNonRootOrder) {
        const Vec2 offset = figure.link(world, link).position - pelvis.position;
        out[i++] = offset.x * invBody;
        out[i++] = offset.y * invBody;
    }

    // Orientation relative to the pelvis, as sin/cos so there is no
    // discontinuity when a limb rotates through pi during a flip.
    for (const int link : kNonRootOrder) {
        const Real relative = figure.link(world, link).angle - pelvis.angle;
        out[i++] = std::sin(relative);
        out[i++] = std::cos(relative);
    }

    for (const int link : kNonRootOrder) {
        out[i++] = figure.link(world, link).angularVelocity;
    }

    for (int j = 0; j < kJointCount; ++j) {
        const Real angle = figure.jointAngle(world, j);
        out[i++] = std::sin(angle);
        out[i++] = std::cos(angle);
    }
    for (int j = 0; j < kJointCount; ++j) {
        out[i++] = figure.jointVelocity(world, j);
    }

    out[i++] = figure.footContact(world, true) ? Real(1) : Real(0);
    out[i++] = figure.footContact(world, false) ? Real(1) : Real(0);

    const Vec2 com = figure.centerOfMass(world);
    const Vec2 comOffset = com - pelvis.position;
    out[i++] = comOffset.x * invBody;
    out[i++] = comOffset.y * invBody;

    const Vec2 comVelocity = figure.centerOfMassVelocity(world);
    out[i++] = comVelocity.x;
    out[i++] = comVelocity.y;

    out[i++] = figure.headHeight(world) / scales.headRestHeight;

    // Encoded on the circle so that phase 0.99 and phase 0.01 are adjacent
    // inputs, which they are: the motion has just looped.
    out[i++] = std::sin(phase * kTwoPi);
    out[i++] = std::cos(phase * kTwoPi);
}

}  // namespace aibf
