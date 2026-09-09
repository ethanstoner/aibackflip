#include "humanoid/HumanoidConfig3D.h"

#include <algorithm>
#include <cmath>

namespace aibf {

namespace {

LinkConfig3D makeLink(const char* name, Real radius, Real halfLength, Real mass,
                      const Vec3& position, const Quat& orientation = Quat::identity()) {
    LinkConfig3D link;
    link.name = name;
    link.radius = radius;
    link.halfLength = halfLength;
    link.mass = mass;
    link.restPosition = position;
    link.restOrientation = orientation;
    return link;
}

JointConfig3D makeBall(const char* name, int parent, int child, const Vec3& anchor, Real cone,
                       Real twist, const Vec3& range, Real kp, Real kd, Real maxTorque) {
    JointConfig3D joint;
    joint.name = name;
    joint.kind = JointKind::Ball;
    joint.parent = parent;
    joint.child = child;
    joint.restAnchor = anchor;
    joint.coneAngle = cone;
    joint.lowerTwist = -twist;
    joint.upperTwist = twist;
    // Defaults to the joint's own limits, so a full-scale action on one axis
    // asks for exactly the cone and a full-scale twist asks for exactly the
    // twist limit. Passing a smaller range narrows what the policy can command
    // without changing what the joint physically permits.
    joint.targetRange = Vec3(std::min(range.x, cone), std::min(range.y, twist),
                             std::min(range.z, cone));
    joint.stiffness = kp;
    joint.damping = kd;
    joint.maxTorque = maxTorque;
    return joint;
}

JointConfig3D makeHinge(const char* name, int parent, int child, const Vec3& anchor,
                        const Vec3& axis, Real lower, Real upper, Real kp, Real kd,
                        Real maxTorque) {
    JointConfig3D joint;
    joint.name = name;
    joint.kind = JointKind::Hinge;
    joint.parent = parent;
    joint.child = child;
    joint.restAnchor = anchor;
    joint.hingeAxis = axis;
    joint.lowerLimit = lower;
    joint.upperLimit = upper;
    joint.stiffness = kp;
    joint.damping = kd;
    joint.maxTorque = maxTorque;
    return joint;
}

}  // namespace

Humanoid3DConfig Humanoid3DConfig::defaults() {
    Humanoid3DConfig cfg;
    cfg.links.resize(kLinkCount);
    cfg.joints.resize(kJointCount);

    // Masses and lengths are the 2D figure's, so the two are the same person and
    // a 3D result can be compared against a 2D one. What is new is z: the left
    // and right limbs now occupy a frontal plane instead of sharing one.
    const Real shoulderZ = Real(0.18);
    const Real hipZ = Real(0.09);

    // Feet lie flat: rotating local +Y by -90 degrees about z points the capsule
    // axis along +X, so endpointB is the toe.
    const Quat footOrientation = Quat::fromAxisAngle(Vec3(0, 0, 1), -kHalfPi);

    cfg.links[kPelvis]    = makeLink("pelvis",      Real(0.09),  Real(0.08), Real(12.0), Vec3(0, Real(1.00), 0));
    cfg.links[kChest]     = makeLink("chest",       Real(0.11),  Real(0.19), Real(22.0), Vec3(0, Real(1.27), 0));
    cfg.links[kHead]      = makeLink("head",        Real(0.10),  Real(0.04), Real(5.0),  Vec3(0, Real(1.50), 0));
    cfg.links[kUpperArmL] = makeLink("upper_arm_l", Real(0.045), Real(0.14), Real(2.0),  Vec3(0, Real(1.26),  shoulderZ));
    cfg.links[kLowerArmL] = makeLink("lower_arm_l", Real(0.040), Real(0.14), Real(1.5),  Vec3(0, Real(0.98),  shoulderZ));
    cfg.links[kUpperArmR] = makeLink("upper_arm_r", Real(0.045), Real(0.14), Real(2.0),  Vec3(0, Real(1.26), -shoulderZ));
    cfg.links[kLowerArmR] = makeLink("lower_arm_r", Real(0.040), Real(0.14), Real(1.5),  Vec3(0, Real(0.98), -shoulderZ));
    cfg.links[kUpperLegL] = makeLink("upper_leg_l", Real(0.06),  Real(0.21), Real(7.0),  Vec3(0, Real(0.71),  hipZ));
    cfg.links[kLowerLegL] = makeLink("lower_leg_l", Real(0.05),  Real(0.21), Real(3.5),  Vec3(0, Real(0.29),  hipZ));
    cfg.links[kFootL]     = makeLink("foot_l",      Real(0.04),  Real(0.09), Real(1.0),  Vec3(Real(0.05), Real(0.04),  hipZ), footOrientation);
    cfg.links[kUpperLegR] = makeLink("upper_leg_r", Real(0.06),  Real(0.21), Real(7.0),  Vec3(0, Real(0.71), -hipZ));
    cfg.links[kLowerLegR] = makeLink("lower_leg_r", Real(0.05),  Real(0.21), Real(3.5),  Vec3(0, Real(0.29), -hipZ));
    cfg.links[kFootR]     = makeLink("foot_r",      Real(0.04),  Real(0.09), Real(1.0),  Vec3(Real(0.05), Real(0.04), -hipZ), footOrientation);

    // Feet carry the whole figure through a small contact patch, so they get the
    // most friction; everything else rarely touches the ground except in a fall,
    // where sliding is realistic.
    cfg.links[kFootL].friction = Real(1.0);
    cfg.links[kFootR].friction = Real(1.0);

    const Vec3 waist(0, Real(1.08), 0);
    const Vec3 neck(0, Real(1.46), 0);
    const Vec3 shoulderL(0, Real(1.40),  shoulderZ);
    const Vec3 elbowL(0, Real(1.12),  shoulderZ);
    const Vec3 shoulderR(0, Real(1.40), -shoulderZ);
    const Vec3 elbowR(0, Real(1.12), -shoulderZ);
    const Vec3 hipL(0, Real(0.92),  hipZ);
    const Vec3 kneeL(0, Real(0.50),  hipZ);
    const Vec3 ankleL(0, Real(0.08),  hipZ);
    const Vec3 hipR(0, Real(0.92), -hipZ);
    const Vec3 kneeR(0, Real(0.50), -hipZ);
    const Vec3 ankleR(0, Real(0.08), -hipZ);

    // The hinge axis is z for every hinge, which is the sagittal axis: knees,
    // elbows and ankles all bend in the plane the 2D figure lived in. That is
    // not an accident of convenience, it is anatomy, and it is why the 2D figure
    // was a reasonable model of a backflip in the first place.
    const Vec3 sagittal(0, 0, 1);

    // Gains carried over from the 2D figure, which M2 measured and M5 through M8
    // trained against. They are a starting point in 3D, not a tuned result: a
    // 3D figure has to resist toppling sideways as well as forwards, and the
    // hips in particular may need more.
    cfg.joints[kWaist] = makeBall("waist", kPelvis, kChest, waist,
                                  Real(0.7), Real(0.5), Vec3(Real(0.6), Real(0.5), Real(0.6)),
                                  Real(4000), Real(400), Real(400));
    cfg.joints[kNeck] = makeBall("neck", kChest, kHead, neck,
                                 Real(0.6), Real(0.6), Vec3(Real(0.5), Real(0.6), Real(0.5)),
                                 Real(400), Real(40), Real(60));
    cfg.joints[kShoulderL] = makeBall("shoulder_l", kChest, kUpperArmL, shoulderL,
                                      Real(1.9), Real(1.0), Vec3(Real(1.6), Real(1.0), Real(1.6)),
                                      Real(800), Real(80), Real(120));
    cfg.joints[kElbowL] = makeHinge("elbow_l", kUpperArmL, kLowerArmL, elbowL, sagittal,
                                    Real(-0.1), Real(2.7), Real(500), Real(50), Real(80));
    cfg.joints[kShoulderR] = makeBall("shoulder_r", kChest, kUpperArmR, shoulderR,
                                      Real(1.9), Real(1.0), Vec3(Real(1.6), Real(1.0), Real(1.6)),
                                      Real(800), Real(80), Real(120));
    cfg.joints[kElbowR] = makeHinge("elbow_r", kUpperArmR, kLowerArmR, elbowR, sagittal,
                                    Real(-0.1), Real(2.7), Real(500), Real(50), Real(80));
    cfg.joints[kHipL] = makeBall("hip_l", kPelvis, kUpperLegL, hipL,
                                 Real(1.3), Real(0.5), Vec3(Real(1.1), Real(0.5), Real(0.7)),
                                 Real(4000), Real(400), Real(400));
    cfg.joints[kKneeL] = makeHinge("knee_l", kUpperLegL, kLowerLegL, kneeL, sagittal,
                                   Real(-2.6), Real(0.05), Real(3000), Real(300), Real(300));
    cfg.joints[kAnkleL] = makeHinge("ankle_l", kLowerLegL, kFootL, ankleL, sagittal,
                                    Real(-0.9), Real(0.5), Real(1500), Real(150), Real(200));
    cfg.joints[kHipR] = makeBall("hip_r", kPelvis, kUpperLegR, hipR,
                                 Real(1.3), Real(0.5), Vec3(Real(1.1), Real(0.5), Real(0.7)),
                                 Real(4000), Real(400), Real(400));
    cfg.joints[kKneeR] = makeHinge("knee_r", kUpperLegR, kLowerLegR, kneeR, sagittal,
                                   Real(-2.6), Real(0.05), Real(3000), Real(300), Real(300));
    cfg.joints[kAnkleR] = makeHinge("ankle_r", kLowerLegR, kFootR, ankleR, sagittal,
                                    Real(-0.9), Real(0.5), Real(1500), Real(150), Real(200));

    return cfg;
}

int Humanoid3DConfig::actionDim() const {
    int total = 0;
    for (const JointConfig3D& j : joints) total += j.dof();
    return total;
}

int Humanoid3DConfig::actionOffset(int jointId) const {
    int offset = 0;
    for (int i = 0; i < jointId && i < static_cast<int>(joints.size()); ++i) {
        offset += joints[static_cast<size_t>(i)].dof();
    }
    return offset;
}

Real Humanoid3DConfig::totalMass() const {
    Real total = 0;
    for (const LinkConfig3D& link : links) total += link.mass;
    return total;
}

Real Humanoid3DConfig::restHeight() const {
    Real top = 0;
    for (const LinkConfig3D& link : links) {
        const Vec3 axis = rotate(link.restOrientation, Vec3(0, link.halfLength, 0));
        top = std::max(top, link.restPosition.y + std::abs(axis.y) + link.radius);
    }
    return top;
}

std::string Humanoid3DConfig::validate() const {
    if (links.size() != static_cast<size_t>(kLinkCount)) return "wrong number of links";
    if (joints.size() != static_cast<size_t>(kJointCount)) return "wrong number of joints";

    for (const LinkConfig3D& link : links) {
        if (!(link.mass > Real(0))) return "link " + link.name + " has non-positive mass";
        if (!(link.radius > Real(0))) return "link " + link.name + " has non-positive radius";
        if (link.halfLength < Real(0)) return "link " + link.name + " has negative half length";
    }

    for (const JointConfig3D& joint : joints) {
        if (joint.parent < 0 || joint.parent >= kLinkCount) return "joint " + joint.name + " has a bad parent";
        if (joint.child < 0 || joint.child >= kLinkCount) return "joint " + joint.name + " has a bad child";
        if (joint.parent == joint.child) return "joint " + joint.name + " connects a link to itself";

        if (joint.kind == JointKind::Hinge) {
            if (joint.upperLimit < joint.lowerLimit) return "joint " + joint.name + " has inverted limits";
            if (length(joint.hingeAxis) < kEpsilon) return "joint " + joint.name + " has a zero hinge axis";
        } else {
            if (joint.upperTwist < joint.lowerTwist) return "joint " + joint.name + " has inverted twist limits";
            if (!(joint.coneAngle > Real(0))) return "joint " + joint.name + " has a non-positive cone";
        }

        // The anchor has to lie on both capsules, or the rest pose is already
        // pulling itself apart before anything moves. Generous tolerance,
        // because this is catching typos, not measuring precision.
        for (int which = 0; which < 2; ++which) {
            const LinkConfig3D& link =
                links[static_cast<size_t>(which == 0 ? joint.parent : joint.child)];
            const Vec3 local = rotateInverse(link.restOrientation, joint.restAnchor - link.restPosition);
            const Real along = clamp(local.y, -link.halfLength, link.halfLength);
            const Vec3 nearest(0, along, 0);
            if (length(local - nearest) > link.radius * Real(2.5) + Real(0.02)) {
                return "joint " + joint.name + " anchor is not on link " + link.name;
            }
        }
    }

    return std::string();
}

// ---------------------------------------------------------------- json

Humanoid3DConfig Humanoid3DConfig::fromJson(const Json& json) {
    Humanoid3DConfig cfg = defaults();
    if (!json.isObject()) return cfg;

    // Joints are matched by name rather than index, so a config can override one
    // gain without restating the whole figure.
    const Json& joints = json["joints"];
    if (joints.isArray()) {
        for (int i = 0; i < joints.size(); ++i) {
            const Json& entry = joints[i];
            const std::string name = entry["name"].string("");
            for (JointConfig3D& joint : cfg.joints) {
                if (joint.name != name) continue;
                joint.stiffness = entry["kp"].real(joint.stiffness);
                joint.damping = entry["kd"].real(joint.damping);
                joint.maxTorque = entry["max_torque"].real(joint.maxTorque);
                joint.coneAngle = entry["cone"].real(joint.coneAngle);
                joint.lowerTwist = entry["lower_twist"].real(joint.lowerTwist);
                joint.upperTwist = entry["upper_twist"].real(joint.upperTwist);
                joint.lowerLimit = entry["lower"].real(joint.lowerLimit);
                joint.upperLimit = entry["upper"].real(joint.upperLimit);
            }
        }
    }

    const Json& links = json["links"];
    if (links.isArray()) {
        for (int i = 0; i < links.size(); ++i) {
            const Json& entry = links[i];
            const std::string name = entry["name"].string("");
            for (LinkConfig3D& link : cfg.links) {
                if (link.name != name) continue;
                link.mass = entry["mass"].real(link.mass);
                link.radius = entry["radius"].real(link.radius);
                link.friction = entry["friction"].real(link.friction);
            }
        }
    }

    return cfg;
}

Json Humanoid3DConfig::toJson() const {
    Json root = Json::object();

    Json linkArray = Json::array();
    for (const LinkConfig3D& link : links) {
        Json entry = Json::object();
        entry.set("name", Json(link.name));
        entry.set("mass", Json(double(link.mass)));
        entry.set("radius", Json(double(link.radius)));
        entry.set("friction", Json(double(link.friction)));
        linkArray.push(std::move(entry));
    }
    root.set("links", std::move(linkArray));

    Json jointArray = Json::array();
    for (const JointConfig3D& joint : joints) {
        Json entry = Json::object();
        entry.set("name", Json(joint.name));
        entry.set("kind", Json(std::string(joint.kind == JointKind::Ball ? "ball" : "hinge")));
        entry.set("kp", Json(double(joint.stiffness)));
        entry.set("kd", Json(double(joint.damping)));
        entry.set("max_torque", Json(double(joint.maxTorque)));
        if (joint.kind == JointKind::Ball) {
            entry.set("cone", Json(double(joint.coneAngle)));
            entry.set("lower_twist", Json(double(joint.lowerTwist)));
            entry.set("upper_twist", Json(double(joint.upperTwist)));
        } else {
            entry.set("lower", Json(double(joint.lowerLimit)));
            entry.set("upper", Json(double(joint.upperLimit)));
        }
        jointArray.push(std::move(entry));
    }
    root.set("joints", std::move(jointArray));

    return root;
}

}  // namespace aibf
