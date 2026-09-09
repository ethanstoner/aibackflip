// Description of the 3D humanoid: 13 capsule links, 6 ball joints and 6 hinges.
//
// Authored the same way as the 2D figure, as a *rest pose*: every link's world
// position and orientation when standing straight, plus the world point each
// joint pivots about. Local anchors and joint reference rotations are derived
// from that, which guarantees the invariant everything else leans on, that in
// the rest pose every joint reads zero. Authoring local anchors by hand would
// make that something to hope for rather than something that holds.
//
// The link and joint orderings are deliberately the same as the 2D figure's, so
// a reference motion, an observation layout or a reward term written against one
// indexes the other the same way.
#pragma once

#include <string>
#include <vector>

#include "core/Json.h"
#include "core/Math.h"
#include "humanoid/HumanoidConfig.h"  // LinkId and JointId are shared

namespace aibf {

// Which joint type each JointId is. Shoulders and hips need three degrees of
// freedom; knees and elbows have one and must actively refuse the other two.
enum class JointKind : int { Ball, Hinge };

// Actions per joint kind. A ball joint takes a rotation vector, a hinge takes an
// angle, so the action vector is not simply one value per joint the way 2D was.
constexpr int kBallJointDof = 3;
constexpr int kHingeJointDof = 1;

struct LinkConfig3D {
    std::string name;
    Real radius = Real(0.05);
    Real halfLength = Real(0.1);
    Real mass = 1;
    Vec3 restPosition;                       // centre of mass, world, standing
    Quat restOrientation = Quat::identity(); // capsule axis is local +Y
    Real friction = Real(0.9);
    Real restitution = 0;
};

struct JointConfig3D {
    std::string name;
    JointKind kind = JointKind::Ball;
    int parent = -1;   // LinkId
    int child = -1;    // LinkId
    Vec3 restAnchor;   // world-space pivot in the rest pose

    // --- ball ---
    // Half-angle of the swing cone, and the twist range about the limb's own
    // axis. A shoulder swings far and twists little; the two are separate limits
    // because they are separate freedoms.
    Real coneAngle = Real(1.2);
    Real lowerTwist = Real(-0.6);
    Real upperTwist = Real(0.6);
    // Per-axis scale mapping a normalised action in [-1, 1] onto a target
    // rotation vector. Sized so a full-scale action asks for roughly the cone
    // limit rather than something the limits will simply refuse.
    Vec3 targetRange{Real(1.0), Real(0.6), Real(1.0)};

    // --- hinge ---
    Vec3 hingeAxis{0, 0, 1};  // in the parent's rest frame
    Real lowerLimit = -kPi;
    Real upperLimit = kPi;

    // --- motor, both kinds ---
    Real stiffness = 0;
    Real damping = 0;
    Real maxTorque = 0;

    int dof() const { return kind == JointKind::Ball ? kBallJointDof : kHingeJointDof; }
};

struct Humanoid3DConfig {
    std::vector<LinkConfig3D> links;
    std::vector<JointConfig3D> joints;

    static Humanoid3DConfig defaults();
    static Humanoid3DConfig fromJson(const Json& json);
    Json toJson() const;
    // Empty when the config describes a figure that can actually be built.
    std::string validate() const;

    Real totalMass() const;
    Real restHeight() const;   // head top above the ground in the rest pose
    // Total length of the action vector: three per ball joint, one per hinge.
    int actionDim() const;
    // Index into the action vector where a joint's values begin.
    int actionOffset(int jointId) const;
};

}  // namespace aibf
