// Hand-authored reference motions for imitation learning.
//
// A clip is a list of keyframes: a time, a root transform, and one angle per
// joint. It is *not* physics. It says what the motion should look like; the
// policy has to discover the torques that produce something close to it, and
// where the reference is physically impossible the physics wins.
//
// Two design points matter for what comes later.
//
// Root angle is stored *unwrapped*. A backflip's root rotates through a full
// turn, so the clip must be able to say "-6.28 radians" and mean one complete
// revolution rather than "the same as zero". Nothing here ever wraps it.
//
// Interpolation is cubic Hermite by default, not linear. Linear interpolation
// between sparse keyframes produces piecewise-constant reference velocities,
// and the DeepMimic velocity term would then be chasing a step function that no
// physical motion can track.
#pragma once

#include <string>
#include <vector>

#include "core/Json.h"
#include "core/Math.h"
#include "humanoid/HumanoidConfig.h"

namespace aibf {

struct MotionPose {
    Vec2 rootPosition;
    Real rootAngle = 0;
    std::vector<Real> jointAngles;

    void resize(size_t joints) { jointAngles.assign(joints, Real(0)); }
};

struct MotionKeyframe {
    Real time = 0;
    Vec2 rootPosition{0, 1};
    Real rootAngle = 0;
    std::vector<Real> jointAngles;
};

enum class MotionInterpolation {
    Linear,
    // Cubic Hermite with finite-difference tangents, which reduces to
    // Catmull-Rom for evenly spaced keyframes and handles uneven spacing.
    Hermite,
};

class Motion2D {
public:
    std::string name = "motion";
    // A looping clip wraps from the last keyframe back to the first, so phase
    // is periodic. A backflip does not loop; an arm raise might.
    bool loop = false;
    MotionInterpolation interpolation = MotionInterpolation::Hermite;
    int jointCount = kJointCount;

    std::vector<MotionKeyframe> keyframes;

    // ---- queries ----
    int frameCount() const { return static_cast<int>(keyframes.size()); }
    bool empty() const { return keyframes.empty(); }
    Real duration() const;

    // Pose at an absolute time in seconds. Times outside the clip clamp, or
    // wrap when `loop` is set.
    MotionPose sample(Real time) const;
    // Pose at a normalized phase in [0, 1].
    MotionPose samplePhase(Real phase) const;

    // Reference velocities: the analytic derivative of whichever interpolation
    // is in use, in units per second. The imitation reward needs these, and
    // differencing two sampled poses would add discretisation noise on top of
    // an already-approximate reference.
    MotionPose sampleVelocity(Real time) const;
    MotionPose samplePhaseVelocity(Real phase) const;

    Real phaseAt(Real time) const;
    Real timeAt(Real phase) const { return duration() > Real(0) ? phase * duration() : Real(0); }

    // ---- editing ----
    // Inserts keeping keyframes ordered by time; replaces one at the same time.
    int insertKeyframe(const MotionKeyframe& frame);
    void removeKeyframe(int index);
    void sortByTime();
    // Shifts every keyframe so the clip starts at zero and lasts `seconds`.
    void retime(Real seconds);

    // ---- serialization ----
    static Motion2D fromJson(const Json& json, std::string* error = nullptr);
    static Motion2D loadFile(const std::string& path, std::string* error = nullptr);
    Json toJson() const;
    bool writeFile(const std::string& path) const;

    // Empty when the clip is usable, otherwise a human-readable reason.
    std::string validate() const;
    // Clamps every *keyframe* joint angle into the config's limits and reports
    // how many were out of range. An authored pose the humanoid physically
    // cannot hold is a reference it can never track, so this is worth knowing
    // before training rather than after.
    int clampToLimits(const Humanoid2DConfig& config);

    // The worst amount, in radians, by which the *sampled* clip leaves the
    // joint limits, and zero when it never does.
    //
    // This is not the same question as clampToLimits, and clamping does not
    // answer it. Cubic Hermite overshoots between keyframes by construction, so
    // a clip whose every authored pose is legal can still sweep past a limit on
    // the way between two of them. The solver refuses to hold that pose, which
    // puts a ceiling on the tracking reward that no policy can reach and that
    // nothing in the clip, the config or the logs would mention.
    Real worstSampledLimitExcess(const Humanoid2DConfig& config, int samples = 512) const;
};

}  // namespace aibf
