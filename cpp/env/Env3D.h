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

namespace aibf {

struct ResetNoise3D {
    Real rootTilt = Real(0.06);        // radians, about a random horizontal axis
    Real rootHeight = Real(0.02);
    Real jointAngle = Real(0.08);
    Real linearVelocity = Real(0.15);
    Real angularVelocity = Real(0.15);
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

    // Applies an impulse to the pelvis. `offset` is along the torso in the
    // pelvis's own frame; a non-zero value carries torque as well as momentum,
    // which is the only kind of disturbance that can perturb free flight.
    void push(const Vec3& impulse, Real offset = 0);

private:
    void evaluateTermination();
    void maybeDisturb();

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
};

}  // namespace aibf
