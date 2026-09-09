// Hand-authored reference motions for the 3D figure.
//
// A clip is a list of keyframes: a time, a root transform, and one rotation per
// joint. It is *not* physics. It says what the motion should look like; the
// policy has to discover the torques that produce something close to it, and
// where the reference is physically impossible the physics wins.
//
// Three things differ from Motion2D, and two of them are the whole reason this
// file exists rather than a `z` being added to the old one.
//
// **Joint values are quaternions, not angles.** A ball joint has three degrees
// of freedom, and its rotations do not commute, so there is no ordering of three
// numbers that behaves like an angle does.
//
// **Root orientation cannot be stored unwrapped.** In 2D the root angle was a
// scalar and a backflip was simply -6.28 radians, one full turn, never wrapped.
// A quaternion has no such freedom: it double-covers, so a full turn and no turn
// are the same value. The rotation *count* is therefore stored alongside the
// orientation, and the sampler reconstructs the total. Without this a backflip
// clip and a standing clip are indistinguishable at their endpoints, which is
// exactly the property the 2D version went out of its way to preserve.
//
// **Interpolation of rotations is slerp**, not a lerp of components. Lerping
// quaternions moves at the wrong angular rate through the middle of an arc,
// which is precisely where a tracking term is looking hardest.
#pragma once

#include <string>
#include <vector>

#include "core/Json.h"
#include "core/Math.h"
#include "humanoid/HumanoidConfig3D.h"

namespace aibf {

struct MotionPose3D {
    Vec3 rootPosition;
    Quat rootOrientation = Quat::identity();
    // Turns completed about the sagittal axis, so a backflip's reference is a
    // continuous quantity the reward can compare against.
    Real rootTurns = 0;
    // One per joint. A hinge stores its single angle as a rotation about the
    // hinge axis, so the clip has one uniform representation and the reward does
    // not need a special case per joint kind.
    std::vector<Quat> jointRotations;

    void resize(size_t joints) { jointRotations.assign(joints, Quat::identity()); }
};

struct MotionKeyframe3D {
    Real time = 0;
    Vec3 rootPosition{0, 1, 0};
    Quat rootOrientation = Quat::identity();
    Real rootTurns = 0;
    std::vector<Quat> jointRotations;
};

class Motion3D {
public:
    std::string name = "motion";
    bool loop = false;
    int jointCount = kJointCount;

    std::vector<MotionKeyframe3D> keyframes;

    Real duration() const {
        return keyframes.empty() ? Real(0) : keyframes.back().time;
    }

    // Inserts in time order, replacing any keyframe already at that time.
    void insertKeyframe(const MotionKeyframe3D& key);

    // Samples at `time` seconds. Out-of-range times clamp for a non-looping
    // clip and wrap for a looping one.
    MotionPose3D sample(Real time) const;
    // Central-difference derivative of the pose, as a root velocity and one
    // angular velocity per joint, expressed in each joint's parent frame.
    void sampleVelocity(Real time, Vec3& rootVelocity, Vec3& rootAngularVelocity,
                        std::vector<Vec3>& jointRates) const;

    // Clamps every joint rotation into the limits the config allows, so an
    // authored clip cannot ask for a pose the figure physically refuses. Returns
    // the number of joints that had to be changed.
    int clampToLimits(const Humanoid3DConfig& config);

    bool writeFile(const std::string& path) const;
    static Motion3D readFile(const std::string& path, std::string* error);
    Json toJson() const;
    static Motion3D fromJson(const Json& json);

    // Empty when the clip is usable.
    std::string validate() const;
};

}  // namespace aibf
