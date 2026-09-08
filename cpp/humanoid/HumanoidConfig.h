// Description of the 2D humanoid: 13 capsule links joined by 12 revolute joints.
//
// The config is authored as a *rest pose* - every link's world position and
// angle when standing straight, plus the world point each joint pivots about.
// Local anchors and joint reference angles are derived from that, which
// guarantees the invariant the rest of the system leans on: in the rest pose
// every joint reads a relative angle of exactly zero. Authoring local anchors
// by hand instead would make that invariant something to hope for.
#pragma once

#include <string>
#include <vector>

#include "core/Json.h"
#include "core/Math.h"

namespace aibf {

// Index into Humanoid2DConfig::links. Order is fixed because the observation
// vector and the reference-motion format both index by it.
enum LinkId : int {
    kPelvis = 0,
    kChest,
    kHead,
    kUpperArmL,
    kLowerArmL,
    kUpperArmR,
    kLowerArmR,
    kUpperLegL,
    kLowerLegL,
    kFootL,
    kUpperLegR,
    kLowerLegR,
    kFootR,
    kLinkCount
};

// Index into Humanoid2DConfig::joints, and therefore into the action vector.
enum JointId : int {
    kWaist = 0,
    kNeck,
    kShoulderL,
    kElbowL,
    kShoulderR,
    kElbowR,
    kHipL,
    kKneeL,
    kAnkleL,
    kHipR,
    kKneeR,
    kAnkleR,
    kJointCount
};

struct LinkConfig {
    std::string name;
    Real radius = Real(0.05);
    Real halfLength = Real(0.1);
    Real mass = 1;
    Vec2 restPosition;      // centre of mass, world space, standing
    Real restAngle = 0;     // capsule axis is local +Y, so 0 means upright
    Real friction = Real(0.9);
    Real restitution = 0;
};

struct JointConfig {
    std::string name;
    int parent = -1;          // LinkId
    int child = -1;           // LinkId
    Vec2 restAnchor;          // world-space pivot in the rest pose
    Real lowerLimit = -kPi;
    Real upperLimit = kPi;
    Real stiffness = 0;       // kp
    Real damping = 0;         // kd
    Real maxTorque = 0;
};

struct Humanoid2DConfig {
    std::vector<LinkConfig> links;
    std::vector<JointConfig> joints;

    // Everything in one figure shares this group, so parts that necessarily
    // overlap at their pivots do not generate contacts against each other.
    int32_t collisionGroup = 1;

    // Height of the pelvis in the rest pose, cached because the standing reward
    // and the termination height are both expressed relative to it.
    Real restPelvisHeight() const { return links[kPelvis].restPosition.y; }
    Real restHeadHeight() const {
        const LinkConfig& head = links[kHead];
        return head.restPosition.y + head.halfLength + head.radius;
    }
    Real totalMass() const;

    // A 1.64 m, 69 kg figure with anthropometric segment masses.
    static Humanoid2DConfig defaults();

    // Reads over defaults(): any field the file omits keeps its default, so a
    // config can override just the two gains being tuned.
    static Humanoid2DConfig fromJson(const Json& json);
    static Humanoid2DConfig loadFile(const std::string& path, std::string* error);
    Json toJson() const;

    // Returns an empty string when the config is usable, otherwise a
    // human-readable reason. Catches the failures that would otherwise show up
    // as a humanoid quietly exploding on the first step.
    std::string validate() const;
};

}  // namespace aibf
