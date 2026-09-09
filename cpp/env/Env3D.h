// One episodic reinforcement-learning environment wrapping a World3D.
//
// Mirrors Env2D: the environment owns physics, the observation, the raw reward
// terms and the termination decision, and it does not own the reward weights or
// any learning state. Those live in Python.
#pragma once

#include <string>
#include <vector>

#include "core/Json.h"
#include "core/Rng.h"
#include "humanoid/Observation3D.h"
#include "humanoid/RewardTerms3D.h"
#include "motion/Motion3D.h"

namespace aibf {

struct ResetNoise3D {
    Real rootTilt = Real(0.06);        // radians, about a random horizontal axis
    Real rootHeight = Real(0.02);
    Real jointAngle = Real(0.08);
    Real linearVelocity = Real(0.15);
    Real angularVelocity = Real(0.15);
};

struct ImitationSettings3D {
    std::string motionPath;
    bool enabled = false;

    // Reference State Initialization: start each episode at a random phase of
    // the clip, with the velocities belonging to that phase.
    //
    // Not a refinement. Without it a policy has to master the takeoff before it
    // ever observes the landing, and for a backflip the takeoff is only worth
    // anything if the landing already works. It is the single reason the 2D
    // backflip trained at all, and nothing about 3D makes it less necessary.
    bool referenceStateInit = true;

    // Ends an episode once RMS joint error exceeds this, in radians; 0 disables.
    Real earlyTerminationPoseError = Real(1.0);

    // Ends an episode once the *root* has drifted this far, as
    // sqrt(dHeight^2 + rootAngleWeight * dAngle^2) in metres; 0 disables.
    //
    // This is what makes imitation mean anything rather than a refinement of
    // the pose test. Joint rotations are root-relative, so a figure lying on its
    // back can hold the reference pose exactly, and the first 2D squat policy
    // did precisely that, reaching pose_match 0.855 while flat on the ground
    // with its feet in the air.
    Real earlyTerminationRootError = Real(0.6);

    // Finishing a non-looping clip is a success, so it is reported as
    // truncation rather than termination.
    bool endEpisodeAtMotionEnd = true;

    ImitationScales3D scales;
};

struct EnvConfig3D {
    Humanoid3DConfig humanoid = Humanoid3DConfig::defaults();

    Real physicsHz = 240;
    int substepsPerControl = 4;   // control runs at 60 Hz
    int maxEpisodeSteps = 1000;

    ResetNoise3D resetNoise;
    Vec3 spawnPosition{0, 1, 0};

    // --- termination, as fractions of the rest pose ---
    Real terminatePelvisHeightRatio = Real(0.55);
    Real terminateHeadHeightRatio = Real(0.55);
    Real terminateChestUprightBelow = Real(0.2);

    // --- disturbances ---
    Real pushProbabilityPerStep = 0;
    int pushIntervalSteps = 0;
    int pushAtStep = 0;
    Real pushImpulseMin = 0;
    Real pushImpulseMax = 0;
    Real pushOffsetMax = 0;

    ImitationSettings3D imitation;

    Real controlHz() const {
        return substepsPerControl > 0 ? physicsHz / Real(substepsPerControl) : physicsHz;
    }
    Real physicsDt() const { return Real(1) / physicsHz; }

    static EnvConfig3D defaults() { return EnvConfig3D{}; }
    static EnvConfig3D fromJson(const Json& json);
    static EnvConfig3D loadFile(const std::string& path, std::string* error);
    Json toJson() const;
    std::string validate() const;
};

class Env3D {
public:
    using Config = EnvConfig3D;

    void initialize(const EnvConfig3D& config, uint64_t seed);

    void reset();
    void step(const Real* actions, int count);

    // Structural, not config-driven: which joints are balls and which are
    // hinges is fixed by the JointId enum, and a JSON config can retune gains
    // and limits but cannot change a knee into a shoulder. EnvConfig3D::validate
    // checks that the config it was handed agrees.
    static constexpr int kActionDim = 6 * kBallJointDof + 6 * kHingeJointDof;

    static int observationDim() { return ObservationLayout3D::kDimension; }
    static int actionDim() { return kActionDim; }
    static int rewardTermCount() { return kTermCount; }

    void writeObservation(Real* out) const;
    void writeRewardTerms(Real* out) const;

    bool terminated() const { return terminated_; }
    bool truncated() const { return truncated_; }
    bool done() const { return terminated_ || truncated_; }
    int episodeStep() const { return episodeStep_; }
    const char* terminationReason() const { return terminationReason_; }

    World3D& world() { return world_; }
    const World3D& world() const { return world_; }
    Humanoid3D& figure() { return figure_; }
    const Humanoid3D& figure() const { return figure_; }
    const EnvConfig3D& config() const { return config_; }
    Rng& rng() { return rng_; }

    Real phase() const { return phase_; }
    void setPhase(Real phase) { phase_ = phase; }

    bool hasMotion() const { return motionLoaded_; }
    const Motion3D& motion() const { return motion_; }
    const ImitationTargets3D& imitationTargets() const { return targets_; }
    // RMS joint rotation error against the reference, radians. Zero without a
    // clip.
    Real poseError() const;
    // Root deviation from the reference: height and orientation combined.
    Real rootError() const;

    // Applies an impulse to the pelvis. `offset` is along the torso in the
    // pelvis's own frame; a non-zero value carries torque as well as momentum,
    // which is the only kind of disturbance that can perturb free flight.
    void push(const Vec3& impulse, Real offset = 0);

private:
    void evaluateTermination();
    void maybeDisturb();
    void updateImitationTargets();
    void applyReferenceStateInit(Real phase);

    EnvConfig3D config_;
    World3D world_;
    Humanoid3D figure_;
    ObservationScales3D scales_;
    Rng rng_;

    std::vector<Real> lastActions_;
    int episodeStep_ = 0;
    bool terminated_ = false;
    bool truncated_ = false;
    const char* terminationReason_ = "";
    Real phase_ = 0;
    int pushCount_ = 0;

    Motion3D motion_;
    bool motionLoaded_ = false;
    Real motionTime_ = 0;
    ImitationTargets3D targets_;

    // Scratch for expanding a reference pose, kept as members so a control step
    // does not allocate.
    mutable std::vector<Vec3> fkPositions_;
    mutable std::vector<Quat> fkOrientations_;
    mutable std::vector<Vec3> fkVelocities_;
    mutable std::vector<Vec3> fkAngularVelocities_;
    mutable std::vector<Vec3> fkJointRates_;
};

}  // namespace aibf
