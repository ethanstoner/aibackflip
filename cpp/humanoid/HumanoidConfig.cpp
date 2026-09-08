#include "humanoid/HumanoidConfig.h"

#include <cmath>
#include <unordered_map>

namespace aibf {

namespace {

// The rest pose is stacked from the ground up so the numbers stay checkable by
// hand: each segment's lower endpoint sits exactly on the joint below it.
//
//   ground      0.00
//   foot centre 0.04   (lying flat, toe towards +X)
//   ankle       0.08
//   knee        0.50
//   hip         0.92
//   pelvis c.   1.00
//   waist       1.08
//   shoulder    1.40
//   neck        1.46
//   head top    1.64
//
// The figure faces +X. A positive relative joint angle rotates the child
// counter-clockwise, which for a downward-hanging limb swings its free end
// forwards; every limit range below is written in that convention.

LinkConfig makeLink(const char* name, Real radius, Real halfLength, Real mass, Vec2 position,
                    Real angle = 0) {
    LinkConfig link;
    link.name = name;
    link.radius = radius;
    link.halfLength = halfLength;
    link.mass = mass;
    link.restPosition = position;
    link.restAngle = angle;
    return link;
}

JointConfig makeJoint(const char* name, int parent, int child, Vec2 anchor, Real lower, Real upper,
                      Real kp, Real kd, Real maxTorque) {
    JointConfig joint;
    joint.name = name;
    joint.parent = parent;
    joint.child = child;
    joint.restAnchor = anchor;
    joint.lowerLimit = lower;
    joint.upperLimit = upper;
    joint.stiffness = kp;
    joint.damping = kd;
    joint.maxTorque = maxTorque;
    return joint;
}

}  // namespace

Real Humanoid2DConfig::totalMass() const {
    Real sum = 0;
    for (const LinkConfig& link : links) sum += link.mass;
    return sum;
}

Humanoid2DConfig Humanoid2DConfig::defaults() {
    Humanoid2DConfig cfg;
    cfg.links.resize(kLinkCount);
    cfg.joints.resize(kJointCount);

    // Feet lie flat: rotating local +Y by -90 degrees points the capsule axis
    // along +X, so endpointB is the toe.
    const Real footAngle = -kHalfPi;

    cfg.links[kPelvis]    = makeLink("pelvis",     Real(0.09),  Real(0.08), Real(12.0), Vec2(0, Real(1.00)));
    cfg.links[kChest]     = makeLink("chest",      Real(0.11),  Real(0.19), Real(22.0), Vec2(0, Real(1.27)));
    cfg.links[kHead]      = makeLink("head",       Real(0.10),  Real(0.04), Real(5.0),  Vec2(0, Real(1.50)));
    cfg.links[kUpperArmL] = makeLink("upper_arm_l", Real(0.045), Real(0.14), Real(2.0),  Vec2(0, Real(1.26)));
    cfg.links[kLowerArmL] = makeLink("lower_arm_l", Real(0.040), Real(0.14), Real(1.5),  Vec2(0, Real(0.98)));
    cfg.links[kUpperArmR] = makeLink("upper_arm_r", Real(0.045), Real(0.14), Real(2.0),  Vec2(0, Real(1.26)));
    cfg.links[kLowerArmR] = makeLink("lower_arm_r", Real(0.040), Real(0.14), Real(1.5),  Vec2(0, Real(0.98)));
    cfg.links[kUpperLegL] = makeLink("upper_leg_l", Real(0.06),  Real(0.21), Real(7.0),  Vec2(0, Real(0.71)));
    cfg.links[kLowerLegL] = makeLink("lower_leg_l", Real(0.05),  Real(0.21), Real(3.5),  Vec2(0, Real(0.29)));
    cfg.links[kFootL]     = makeLink("foot_l",      Real(0.04),  Real(0.09), Real(1.0),  Vec2(Real(0.05), Real(0.04)), footAngle);
    cfg.links[kUpperLegR] = makeLink("upper_leg_r", Real(0.06),  Real(0.21), Real(7.0),  Vec2(0, Real(0.71)));
    cfg.links[kLowerLegR] = makeLink("lower_leg_r", Real(0.05),  Real(0.21), Real(3.5),  Vec2(0, Real(0.29)));
    cfg.links[kFootR]     = makeLink("foot_r",      Real(0.04),  Real(0.09), Real(1.0),  Vec2(Real(0.05), Real(0.04)), footAngle);

    // Feet carry the whole figure through a single small contact patch, so they
    // get the most friction; everything else rarely touches the ground except
    // during a fall, where sliding is realistic.
    cfg.links[kFootL].friction = Real(1.0);
    cfg.links[kFootR].friction = Real(1.0);

    const Vec2 waist(0, Real(1.08));
    const Vec2 neck(0, Real(1.46));
    const Vec2 shoulder(0, Real(1.40));
    const Vec2 elbow(0, Real(1.12));
    const Vec2 hip(0, Real(0.92));
    const Vec2 knee(0, Real(0.50));
    const Vec2 ankle(0, Real(0.08));

    // Gains are starting points sized to each joint's load, not tuned results.
    // M2 measures them and M5 will move them again if standing needs it.
    cfg.joints[kWaist]     = makeJoint("waist",      kPelvis,    kChest,     waist,    Real(-0.8), Real(0.6),  Real(4000), Real(400), Real(400));
    cfg.joints[kNeck]      = makeJoint("neck",       kChest,     kHead,      neck,     Real(-0.6), Real(0.6),  Real(400),  Real(40),  Real(60));
    cfg.joints[kShoulderL] = makeJoint("shoulder_l", kChest,     kUpperArmL, shoulder, Real(-1.2), Real(3.0),  Real(800),  Real(80),  Real(120));
    cfg.joints[kElbowL]    = makeJoint("elbow_l",    kUpperArmL, kLowerArmL, elbow,    Real(-0.1), Real(2.7),  Real(500),  Real(50),  Real(80));
    cfg.joints[kShoulderR] = makeJoint("shoulder_r", kChest,     kUpperArmR, shoulder, Real(-1.2), Real(3.0),  Real(800),  Real(80),  Real(120));
    cfg.joints[kElbowR]    = makeJoint("elbow_r",    kUpperArmR, kLowerArmR, elbow,    Real(-0.1), Real(2.7),  Real(500),  Real(50),  Real(80));
    cfg.joints[kHipL]      = makeJoint("hip_l",      kPelvis,    kUpperLegL, hip,      Real(-0.5), Real(2.1),  Real(4000), Real(400), Real(400));
    cfg.joints[kKneeL]     = makeJoint("knee_l",     kUpperLegL, kLowerLegL, knee,     Real(-2.6), Real(0.0),  Real(3000), Real(300), Real(300));
    cfg.joints[kAnkleL]    = makeJoint("ankle_l",    kLowerLegL, kFootL,     ankle,    Real(-0.9), Real(0.5),  Real(1500), Real(150), Real(200));
    cfg.joints[kHipR]      = makeJoint("hip_r",      kPelvis,    kUpperLegR, hip,      Real(-0.5), Real(2.1),  Real(4000), Real(400), Real(400));
    cfg.joints[kKneeR]     = makeJoint("knee_r",     kUpperLegR, kLowerLegR, knee,     Real(-2.6), Real(0.0),  Real(3000), Real(300), Real(300));
    cfg.joints[kAnkleR]    = makeJoint("ankle_r",    kLowerLegR, kFootR,     ankle,    Real(-0.9), Real(0.5),  Real(1500), Real(150), Real(200));

    return cfg;
}

// ---------------------------------------------------------------- json

Humanoid2DConfig Humanoid2DConfig::fromJson(const Json& json) {
    Humanoid2DConfig cfg = defaults();
    if (!json.isObject()) return cfg;

    cfg.collisionGroup = json["collision_group"].integer(cfg.collisionGroup);

    // Entries are matched by name, so a config file can override two gains
    // without restating the entire skeleton.
    std::unordered_map<std::string, size_t> linkByName;
    for (size_t i = 0; i < cfg.links.size(); ++i) linkByName[cfg.links[i].name] = i;

    const Json& links = json["links"];
    for (int i = 0; i < static_cast<int>(links.size()); ++i) {
        const Json& entry = links[i];
        const auto it = linkByName.find(entry["name"].string());
        if (it == linkByName.end()) continue;
        LinkConfig& link = cfg.links[it->second];
        link.radius = entry["radius"].real(link.radius);
        link.halfLength = entry["half_length"].real(link.halfLength);
        link.mass = entry["mass"].real(link.mass);
        link.restPosition = entry["rest_position"].vec2(link.restPosition);
        link.restAngle = entry["rest_angle"].real(link.restAngle);
        link.friction = entry["friction"].real(link.friction);
        link.restitution = entry["restitution"].real(link.restitution);
    }

    std::unordered_map<std::string, size_t> jointByName;
    for (size_t i = 0; i < cfg.joints.size(); ++i) jointByName[cfg.joints[i].name] = i;

    const Json& joints = json["joints"];
    for (int i = 0; i < static_cast<int>(joints.size()); ++i) {
        const Json& entry = joints[i];
        const auto it = jointByName.find(entry["name"].string());
        if (it == jointByName.end()) continue;
        JointConfig& joint = cfg.joints[it->second];
        joint.restAnchor = entry["rest_anchor"].vec2(joint.restAnchor);
        joint.lowerLimit = entry["lower_limit"].real(joint.lowerLimit);
        joint.upperLimit = entry["upper_limit"].real(joint.upperLimit);
        joint.stiffness = entry["kp"].real(joint.stiffness);
        joint.damping = entry["kd"].real(joint.damping);
        joint.maxTorque = entry["max_torque"].real(joint.maxTorque);
    }

    return cfg;
}

Humanoid2DConfig Humanoid2DConfig::loadFile(const std::string& path, std::string* error) {
    std::string parseError;
    Json json = Json::parseFile(path, &parseError);
    if (!parseError.empty()) {
        if (error) *error = parseError;
        return defaults();
    }
    Humanoid2DConfig cfg = fromJson(json);
    const std::string problem = cfg.validate();
    if (error) *error = problem;
    return cfg;
}

Json Humanoid2DConfig::toJson() const {
    Json root = Json::object();
    root.set("collision_group", Json(collisionGroup));

    Json linkArray = Json::array();
    for (const LinkConfig& link : links) {
        Json entry = Json::object();
        entry.set("name", Json(link.name));
        entry.set("radius", Json(double(link.radius)));
        entry.set("half_length", Json(double(link.halfLength)));
        entry.set("mass", Json(double(link.mass)));
        Json position = Json::array();
        position.push(Json(double(link.restPosition.x)));
        position.push(Json(double(link.restPosition.y)));
        entry.set("rest_position", std::move(position));
        entry.set("rest_angle", Json(double(link.restAngle)));
        entry.set("friction", Json(double(link.friction)));
        entry.set("restitution", Json(double(link.restitution)));
        linkArray.push(std::move(entry));
    }
    root.set("links", std::move(linkArray));

    Json jointArray = Json::array();
    for (const JointConfig& joint : joints) {
        Json entry = Json::object();
        entry.set("name", Json(joint.name));
        entry.set("parent", Json(links[static_cast<size_t>(joint.parent)].name));
        entry.set("child", Json(links[static_cast<size_t>(joint.child)].name));
        Json anchor = Json::array();
        anchor.push(Json(double(joint.restAnchor.x)));
        anchor.push(Json(double(joint.restAnchor.y)));
        entry.set("rest_anchor", std::move(anchor));
        entry.set("lower_limit", Json(double(joint.lowerLimit)));
        entry.set("upper_limit", Json(double(joint.upperLimit)));
        entry.set("kp", Json(double(joint.stiffness)));
        entry.set("kd", Json(double(joint.damping)));
        entry.set("max_torque", Json(double(joint.maxTorque)));
        jointArray.push(std::move(entry));
    }
    root.set("joints", std::move(jointArray));
    return root;
}

// ---------------------------------------------------------------- validation

std::string Humanoid2DConfig::validate() const {
    if (links.size() != static_cast<size_t>(kLinkCount)) {
        return "expected " + std::to_string(kLinkCount) + " links, got " +
               std::to_string(links.size());
    }
    if (joints.size() != static_cast<size_t>(kJointCount)) {
        return "expected " + std::to_string(kJointCount) + " joints, got " +
               std::to_string(joints.size());
    }

    for (const LinkConfig& link : links) {
        if (!(link.mass > Real(0))) return "link '" + link.name + "' has non-positive mass";
        if (!(link.radius > Real(0))) return "link '" + link.name + "' has non-positive radius";
        if (link.halfLength < Real(0)) return "link '" + link.name + "' has negative half length";
        if (!isFinite(link.restPosition) || !std::isfinite(link.restAngle)) {
            return "link '" + link.name + "' has a non-finite rest pose";
        }
        if (link.friction < Real(0)) return "link '" + link.name + "' has negative friction";
    }

    for (const JointConfig& joint : joints) {
        if (joint.parent < 0 || joint.parent >= kLinkCount || joint.child < 0 ||
            joint.child >= kLinkCount) {
            return "joint '" + joint.name + "' references a link that does not exist";
        }
        if (joint.parent == joint.child) {
            return "joint '" + joint.name + "' connects a link to itself";
        }
        if (!(joint.lowerLimit < joint.upperLimit)) {
            return "joint '" + joint.name + "' has an empty limit range";
        }
        if (joint.stiffness < Real(0) || joint.damping < Real(0) || joint.maxTorque < Real(0)) {
            return "joint '" + joint.name + "' has a negative gain";
        }

        // A pivot far from both of the links it connects means a typo in the
        // rest pose. Left alone it produces an enormous constraint violation on
        // step one and the figure snaps into a knot before anything can be
        // observed - a failure that is much harder to read than this message.
        for (const int side : {joint.parent, joint.child}) {
            const LinkConfig& link = links[static_cast<size_t>(side)];
            const Vec2 axis = rotate(Vec2(0, link.halfLength), link.restAngle);
            Real best = length(joint.restAnchor - (link.restPosition + axis));
            best = std::min(best, length(joint.restAnchor - (link.restPosition - axis)));
            best = std::min(best, length(joint.restAnchor - link.restPosition));
            if (best > link.radius + link.halfLength + Real(0.05)) {
                return "joint '" + joint.name + "' anchors far from link '" + link.name +
                       "' in the rest pose";
            }
        }
    }

    return std::string();
}

}  // namespace aibf
