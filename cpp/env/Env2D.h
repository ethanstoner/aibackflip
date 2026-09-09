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
#include "motion/Motion2D.h"
#include "physics/World2D.h"

namespace aibf {

struct ImitationSettings {
    std::string motionPath;
    bool enabled = false;

    // Reference State Initialization: start each episode at a random phase of
    // the motion rather than always at the beginning.
    //
    // This is not a refinement, it is the reason acrobatic imitation works at
    // all. Without it a policy has to master the takeoff before it ever
    // observes the landing, so the later half of the motion receives no
    // gradient until the earlier half is solved - and for a backflip the
    // earlier half is only worth anything if the later half already works.
    bool referenceStateInit = true;

    // Ends an episode once RMS joint error exceeds this (radians); 0 disables.
    // An episode that has drifted this far is not coming back, and letting it
    // run fills the rollout with samples from states the motion never visits.
    Real earlyTerminationPoseError = Real(0.9);

    // Ends an episode once the *root* has drifted this far from the reference,
    // as sqrt(dHeight^2 + rootAngleWeight * dAngle^2) in metres; 0 disables.
    //
    // This is not a refinement of the pose test, it is what makes imitation
    // mean anything. Joint angles are root-relative, so a figure lying flat on
    // its back can hold exactly the reference pose - and the first squat policy
    // trained here did precisely that, reaching pose_match 0.855 while on the
    // ground with its feet in the air. Nothing in the joint-angle reward or the
    // pose-error termination distinguishes "squatting" from "lying down making
    // squat-shaped leg motions".
    Real earlyTerminationRootError = Real(0.35);

    // For a non-looping clip, finishing it is a success, not a failure, so it
    // is reported as truncation.
    bool endEpisodeAtMotionEnd = true;

    ImitationScales scales;
};

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
    //
    // Two schedules. During training, a random one: shoves arrive at
    // `pushProbabilityPerStep` so the policy cannot learn to brace on a timer.
    // For measurement, a deterministic one: `pushIntervalSteps` fires exactly
    // every N control steps with alternating direction, which is what turns
    // "seems robust" into a survival curve against impulse magnitude.
    Real pushProbabilityPerStep = 0;
    int pushIntervalSteps = 0;  // 0 uses the probability instead
    Real pushImpulseMin = 0;
    Real pushImpulseMax = 0;

    // --- imitation (M7) ---
    ImitationSettings imitation;

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

    // Phase of the reference motion; 0 when no motion is loaded.
    Real phase() const { return phase_; }
    void setPhase(Real phase) { phase_ = phase; }

    bool hasMotion() const { return motionLoaded_; }
    const Motion2D& motion() const { return motion_; }
    // What the reference asks for at the current phase. Invalid without a clip.
    const ImitationTargets& imitationTargets() const { return targets_; }
    // RMS joint error against the reference, radians. Zero without a clip.
    Real poseError() const;
    // Root deviation from the reference: height and orientation combined.
    Real rootError() const;

    // Applies an impulse to the pelvis. Used by the disturbance schedule and by
    // the interactive push tool.
    void push(const Vec2& impulse);

private:
    void evaluateTermination();
    void maybeDisturb();
    void updateImitationTargets();
    void applyReferenceStateInit(Real phase);

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
    int pushCount_ = 0;

    Motion2D motion_;
    bool motionLoaded_ = false;
    Real motionTime_ = 0;
    ImitationTargets targets_;
};

}  // namespace aibf
