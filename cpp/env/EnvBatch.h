// A batch of independent environments stepped together.
//
// Batching exists for the wire protocol as much as for throughput: one datagram
// carries every environment's observation, so a control step costs one round
// trip instead of N.
//
// Environments auto-reset, and the observation an episode ended on is reported
// alongside the fresh one. That combination is what PPO needs:
//
//   * auto-reset keeps every step a real transition, so a rollout is a fixed
//     (steps x environments) block with no holes to mask out;
//   * the final observation is still required, because a *truncated* episode
//     has to bootstrap its value estimate from the state it was cut off in.
//     Dropping it silently teaches the critic that hitting the time limit is
//     worth zero, which is the classic time-limit bootstrapping bug.
//
// Reward terms, done flags and the step counter reported after a step are the
// terminal ones; only the observation is post-reset.
#pragma once

#include <cstdint>
#include <vector>

#include "env/Env2D.h"

namespace aibf {

class EnvBatch {
public:
    void initialize(const EnvConfig& config, int count, uint64_t seed);

    int size() const { return static_cast<int>(envs_.size()); }
    Env2D& env(int i) { return envs_[static_cast<size_t>(i)]; }
    const Env2D& env(int i) const { return envs_[static_cast<size_t>(i)]; }

    static int observationDim() { return Env2D::observationDim(); }
    static int actionDim() { return Env2D::actionDim(); }
    static int rewardTermCount() { return Env2D::rewardTermCount(); }

    void resetAll();

    // `actions` is size() * actionDim() values in [-1, 1], row-major by
    // environment. `resetMask` may be null; where it is non-zero the
    // environment is force-reset instead of stepped and its action ignored,
    // which is how a caller starts a fresh rollout mid-stream.
    void step(const Real* actions, const uint8_t* resetMask);

    // Struct-of-arrays views of the most recent step, matching the wire layout
    // so Python can wrap each block in a single numpy view.
    const std::vector<Real>& observations() const { return observations_; }
    const std::vector<Real>& rewardTerms() const { return rewardTerms_; }
    const std::vector<uint8_t>& terminated() const { return terminated_; }
    const std::vector<uint8_t>& truncated() const { return truncated_; }
    const std::vector<uint32_t>& episodeStep() const { return episodeStep_; }

    // Which environments ended on the most recent step, and the observation
    // each of them ended on. `finalObservations` is dense: one row per set flag,
    // in environment order.
    const std::vector<uint8_t>& finalMask() const { return finalMask_; }
    const std::vector<Real>& finalObservations() const { return finalObservations_; }

    int doneCount() const;

private:
    void captureAll();
    void captureEnv(int index);

    std::vector<Env2D> envs_;

    std::vector<Real> observations_;
    std::vector<Real> rewardTerms_;
    std::vector<uint8_t> terminated_;
    std::vector<uint8_t> truncated_;
    std::vector<uint32_t> episodeStep_;
    std::vector<uint8_t> finalMask_;
    std::vector<Real> finalObservations_;
    std::vector<Real> finalScratch_;
};

}  // namespace aibf
