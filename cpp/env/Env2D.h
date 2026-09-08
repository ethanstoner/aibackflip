// One episodic reinforcement-learning environment wrapping a World2D.
//
// The environment owns physics, the observation, the raw reward terms and the
// termination decision - everything that needs access to contacts, the centre of
// mass, or the solver's internals. It does not own the reward *weights* or any
// learning state; those live in Python.
#pragma once

#include <string>
#include <vector>

#include "core/Json.h"
#include "core/Rng.h"
#include "humanoid/Observation.h"
#include "humanoid/RewardTerms.h"
#include "physics/World2D.h"

namespace aibf {

struct EnvConfig {
    // Not a default-constructed Humanoid2DConfig: that one has empty link and
    // joint vectors, so anything indexing kPelvis walks off the end. A struct
    // whose default state is unusable is a trap, so the default is the real
    // figure.
    Humanoid2DConfig humanoid = Humanoid2DConfig::defaults();

    Real physicsHz = 240;
    // Control runs at physicsHz / substepsPerControl, i.e. 60 Hz by default.
    // The policy sees one observation and issues one action per control step.
    int substepsPerControl = 4;
    int maxEpisodeSteps = 1000;  // ~16.7 s at 60 Hz

    // Every episode starts slightly differently. Without this a policy can
    // memorise one exact initial state and never learn to correct anything -
    // and, because the rest pose is a perfectly symmetric equilibrium, it would
    // score well while having learned nothing at all.
    //
    // Deliberately small: enough that standing requires active correction, not
    // so much that the figure is already falling before it can act. M6 widens
    // these for push robustness.
    ResetNoise resetNoise{
        /*rootAngle=*/Real(0.06),        // ~3.4 degrees
        /*rootHeight=*/Real(0.02),
        /*jointAngle=*/Real(0.08),
        /*linearVelocity=*/Real(0.15),
        /*angularVelocity=*/Real(0.15),
    };
    Vec2 spawnPosition{0, 1};

    // --- termination ---
    // Fractions of the rest pose, so they scale with the figure rather than
    // being magic distances.
    Real terminatePelvisHeightRatio = Real(0.55);
    Real terminateHeadHeightRatio = Real(0.55);
    // cos of the chest's tilt from upright; 0.2 is roughly 78 degrees over.
    Real terminateChestUprightBelow = Real(0.2);

    // --- disturbances (M6) ---
    Real pushProbabilityPerStep = 0;
    Real pushImpulseMin = 0;
    Real pushImpulseMax = 0;

    Real controlHz() const {
        return substepsPerControl > 0 ? physicsHz / Real(substepsPerControl) : physicsHz;
    }
    Real physicsDt() const { return Real(1) / physicsHz; }

    static EnvConfig defaults() { return EnvConfig{}; }
    static EnvConfig fromJson(const Json& json);
    static EnvConfig loadFile(const std::string& path, std::string* error);
    Json toJson() const;
    std::string validate() const;
};

class Env2D {
public:
    void initialize(const EnvConfig& config, uint64_t seed);

    void reset();
    // Applies `actions` (normalized to [-1, 1], one per joint) and advances one
    // control step. Does not auto-reset; the caller decides when, so a policy
    // always sees the terminal observation.
    void step(const Real* actions, int count);

    static int observationDim() { return ObservationLayout::kDimension; }
    static int actionDim() { return kJointCount; }
    static int rewardTermCount() { return kTermCount; }

    void writeObservation(Real* out) const;
    void writeRewardTerms(Real* out) const;

    bool terminated() const { return terminated_; }
    bool truncated() const { return truncated_; }
    bool done() const { return terminated_ || truncated_; }
    int episodeStep() const { return episodeStep_; }

    // Reason the last episode ended, for logging. Empty while running.
    const char* terminationReason() const { return terminationReason_; }

    World2D& world() { return world_; }
    const World2D& world() const { return world_; }
    Humanoid2D& figure() { return figure_; }
    const Humanoid2D& figure() const { return figure_; }
    const EnvConfig& config() const { return config_; }
    Rng& rng() { return rng_; }

    // Phase of the reference motion, 0 until imitation lands in M7.
    Real phase() const { return phase_; }
    void setPhase(Real phase) { phase_ = phase; }

    // Applies an impulse to the pelvis. Used by the disturbance schedule and by
    // the interactive push tool.
    void push(const Vec2& impulse);

private:
    void evaluateTermination();
    void maybeDisturb();

    EnvConfig config_;
    World2D world_;
    Humanoid2D figure_;
    ObservationScales scales_;
    Rng rng_;

    std::vector<Real> lastActions_;
    int episodeStep_ = 0;
    bool terminated_ = false;
    bool truncated_ = false;
    const char* terminationReason_ = "";
    Real phase_ = 0;
};

}  // namespace aibf
