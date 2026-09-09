// The 3D humanoid: builds the figure described by a Humanoid3DConfig into a
// World3D and drives it from a normalised action vector.
//
// Mirrors Humanoid2D, including the part that matters most: the figure is
// created from a rest pose, and the joint reference rotations are derived from
// that pose rather than authored, so every joint reads zero when standing.
#pragma once

#include <vector>

#include "humanoid/HumanoidConfig3D.h"
#include "physics/World3D.h"

namespace aibf {

class Humanoid3D {
public:
    // Builds the figure into `world`, with the pelvis at `rootPosition`.
    void build(World3D& world, const Humanoid3DConfig& config, const Vec3& rootPosition);

    const Humanoid3DConfig& config() const { return config_; }
    int actionDim() const { return config_.actionDim(); }

    RigidBody3D& link(World3D& world, int linkId) const {
        return world.body(bodyIndex_[static_cast<size_t>(linkId)]);
    }
    const RigidBody3D& link(const World3D& world, int linkId) const {
        return world.body(bodyIndex_[static_cast<size_t>(linkId)]);
    }
    int32_t bodyIndex(int linkId) const { return bodyIndex_[static_cast<size_t>(linkId)]; }

    // Joints are stored in two arrays because they are different types, so a
    // JointId maps to one or the other. -1 means "not this kind".
    int32_t ballIndex(int jointId) const { return ballIndex_[static_cast<size_t>(jointId)]; }
    int32_t hingeIndex(int jointId) const { return hingeIndex_[static_cast<size_t>(jointId)]; }

    // Drives every joint from a normalised action vector in [-1, 1].
    //
    // A ball joint consumes three values as a rotation vector scaled by its
    // per-axis range; a hinge consumes one, mapped piecewise about the rest pose
    // so that the whole action range reaches the joint's real limits rather than
    // half of it being dead.
    void setJointTargetsNormalized(World3D& world, const Real* actions, int count) const;

    // Sets every motor target back to the rest pose.
    void relaxToRestPose(World3D& world) const;

    // Places the figure at a pose without touching velocities.
    void setPose(World3D& world, const Vec3& rootPosition, const Quat& rootOrientation) const;

    Vec3 centerOfMass(const World3D& world) const;
    Vec3 centerOfMassVelocity(const World3D& world) const;
    Real totalMass() const { return config_.totalMass(); }
    Real restHeight() const { return restHeight_; }

    // Lowest point of any capsule, used to drop the figure onto the ground.
    Real lowestPoint(const World3D& world) const;

    bool footInContact(const World3D& world, int footLinkId) const {
        return world.hasContact(bodyIndex(footLinkId));
    }

private:
    Humanoid3DConfig config_;
    std::vector<int32_t> bodyIndex_;
    std::vector<int32_t> ballIndex_;
    std::vector<int32_t> hingeIndex_;
    Real restHeight_ = 0;
};

}  // namespace aibf
