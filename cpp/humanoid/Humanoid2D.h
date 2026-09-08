// Builds the 2D humanoid into a World2D and provides pose-level access to it.
//
// Everything here is a thin, stateless-ish view over bodies and joints that live
// in the world; the humanoid owns indices, not state. That keeps many humanoids
// in many worlds cheap, which is what parallel environments need.
#pragma once

#include <vector>

#include "core/Rng.h"
#include "humanoid/HumanoidConfig.h"
#include "physics/World2D.h"

namespace aibf {

// Randomisation applied on reset. Small perturbations stop a policy from
// memorising one exact initial state; the ranges get widened in M6 for push
// robustness.
struct ResetNoise {
    Real rootAngle = 0;        // radians, uniform +/-
    Real rootHeight = 0;       // metres, uniform +/-
    Real jointAngle = 0;       // radians, uniform +/- per joint
    Real linearVelocity = 0;   // m/s, uniform +/- on the root
    Real angularVelocity = 0;  // rad/s, uniform +/- on the root
};

class Humanoid2D {
public:
    // Creates the bodies and joints inside `world`. Safe to call once per world.
    void build(World2D& world, const Humanoid2DConfig& config);

    const Humanoid2DConfig& config() const { return config_; }

    int32_t bodyIndex(int linkId) const { return bodyIndices_[static_cast<size_t>(linkId)]; }
    int32_t jointIndex(int jointId) const { return jointIndices_[static_cast<size_t>(jointId)]; }

    RigidBody2D& link(World2D& world, int linkId) const { return world.body(bodyIndex(linkId)); }
    const RigidBody2D& link(const World2D& world, int linkId) const {
        return world.body(bodyIndex(linkId));
    }
    RevoluteJoint2D& joint(World2D& world, int jointId) const {
        return world.joint(jointIndex(jointId));
    }
    const RevoluteJoint2D& joint(const World2D& world, int jointId) const {
        return world.joint(jointIndex(jointId));
    }

    // ---- pose ----

    // Places every link by forward kinematics from the root, so the joint
    // anchors are satisfied exactly and the solver has nothing to repair on the
    // first step. `jointAngles` is kJointCount values in joint-relative
    // convention; nullptr means the rest pose.
    void setPose(World2D& world, Vec2 rootPosition, Real rootAngle,
                 const Real* jointAngles = nullptr) const;

    // Rest pose with the feet on y = 0, optionally perturbed.
    void reset(World2D& world, Vec2 rootPosition) const;
    void reset(World2D& world, Vec2 rootPosition, const ResetNoise& noise, Rng& rng) const;

    Real jointAngle(const World2D& world, int jointId) const;
    Real jointVelocity(const World2D& world, int jointId) const;

    // ---- actuation ----

    void setJointTarget(World2D& world, int jointId, Real angle) const;
    // Maps an action in [-1, 1] onto the joint's limit range. Values outside
    // that interval are clamped rather than extrapolated.
    void setJointTargetNormalized(World2D& world, int jointId, Real action) const;
    void applyNormalizedActions(World2D& world, const Real* actions, int count) const;
    void setMotorsEnabled(World2D& world, bool enabled) const;
    void holdRestPose(World2D& world) const;

    // ---- queries ----

    Vec2 rootPosition(const World2D& world) const { return link(world, kPelvis).position; }
    Real rootAngle(const World2D& world) const { return link(world, kPelvis).angle; }
    Vec2 centerOfMass(const World2D& world) const;
    Vec2 centerOfMassVelocity(const World2D& world) const;
    Real headHeight(const World2D& world) const;
    // cos of the tilt away from vertical: 1 upright, 0 horizontal, -1 inverted.
    Real uprightness(const World2D& world, int linkId) const;
    bool footContact(const World2D& world, bool left) const;
    Real totalMass() const { return config_.totalMass(); }

    // Largest |relative angle| by which any joint exceeds its own limits.
    // Zero for any pose the config considers legal.
    Real worstLimitViolation(const World2D& world) const;

private:
    Humanoid2DConfig config_;
    std::vector<int32_t> bodyIndices_;
    std::vector<int32_t> jointIndices_;
    // Joint order with parents before children, so forward kinematics is a
    // single pass. Derived at build time rather than assumed from the enum.
    std::vector<int> kinematicOrder_;
};

}  // namespace aibf
