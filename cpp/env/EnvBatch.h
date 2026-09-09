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
//
// Templated on the environment so the 2D and 3D figures share one batch, one
// wire format and one server. The two environments differ in dimension and in
// almost nothing else, and a copy of this file with `2` changed to `3` would be
// a place for the two to quietly drift apart.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "env/Env2D.h"
#include "env/Env3D.h"

namespace aibf {

template <typename Env>
class EnvBatchT {
public:
    using Config = typename Env::Config;

    void initialize(const Config& config, int count, uint64_t seed) {
        envs_.clear();
        envs_.resize(static_cast<size_t>(count < 1 ? 1 : count));
        for (size_t i = 0; i < envs_.size(); ++i) {
            // Distinct streams rather than nearby seeds: PCG32 streams are
            // guaranteed independent, whereas seeds one apart are not.
            envs_[i].initialize(config, seed + static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull);
        }

        const size_t n = envs_.size();
        observations_.assign(n * static_cast<size_t>(observationDim()), Real(0));
        rewardTerms_.assign(n * static_cast<size_t>(rewardTermCount()), Real(0));
        terminated_.assign(n, 0);
        truncated_.assign(n, 0);
        episodeStep_.assign(n, 0);
        finalMask_.assign(n, 0);
        finalObservations_.clear();
        finalScratch_.assign(static_cast<size_t>(observationDim()), Real(0));

        captureAll();
    }

    int size() const { return static_cast<int>(envs_.size()); }
    Env& env(int i) { return envs_[static_cast<size_t>(i)]; }
    const Env& env(int i) const { return envs_[static_cast<size_t>(i)]; }

    static int observationDim() { return Env::observationDim(); }
    static int actionDim() { return Env::actionDim(); }
    static int rewardTermCount() { return Env::rewardTermCount(); }

    void resetAll() {
        for (Env& e : envs_) e.reset();
        std::fill(finalMask_.begin(), finalMask_.end(), uint8_t(0));
        finalObservations_.clear();
        captureAll();
    }

    // `actions` is size() * actionDim() values in [-1, 1], row-major by
    // environment. `resetMask` may be null; where it is non-zero the
    // environment is force-reset instead of stepped and its action ignored,
    // which is how a caller starts a fresh rollout mid-stream.
    void step(const Real* actions, const uint8_t* resetMask) {
        const size_t obsDim = static_cast<size_t>(observationDim());
        const int stride = actionDim();

        std::fill(finalMask_.begin(), finalMask_.end(), uint8_t(0));
        finalObservations_.clear();

        for (size_t i = 0; i < envs_.size(); ++i) {
            if (resetMask && resetMask[i]) {
                envs_[i].reset();
                captureEnv(static_cast<int>(i));
                continue;
            }

            const Real* row = actions ? actions + i * static_cast<size_t>(stride) : nullptr;
            if (row) envs_[i].step(row, stride);

            // Everything except the observation is captured *before* the reset,
            // so the reported reward, flags and length describe the episode that
            // just ended rather than the one that just began.
            captureEnv(static_cast<int>(i));

            if (envs_[i].done()) {
                envs_[i].writeObservation(finalScratch_.data());
                finalObservations_.insert(finalObservations_.end(), finalScratch_.begin(),
                                          finalScratch_.end());
                finalMask_[i] = 1;

                envs_[i].reset();
                // Only the observation is overwritten with the fresh episode's.
                envs_[i].writeObservation(observations_.data() + i * obsDim);
            }
        }
    }

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

    int doneCount() const {
        int count = 0;
        for (const uint8_t flag : finalMask_) count += flag ? 1 : 0;
        return count;
    }

private:
    void captureAll() {
        for (int i = 0; i < size(); ++i) captureEnv(i);
    }

    void captureEnv(int index) {
        const size_t i = static_cast<size_t>(index);
        envs_[i].writeObservation(observations_.data() +
                                  i * static_cast<size_t>(observationDim()));
        envs_[i].writeRewardTerms(rewardTerms_.data() +
                                  i * static_cast<size_t>(rewardTermCount()));
        terminated_[i] = envs_[i].terminated() ? 1u : 0u;
        truncated_[i] = envs_[i].truncated() ? 1u : 0u;
        episodeStep_[i] = static_cast<uint32_t>(envs_[i].episodeStep());
    }

    std::vector<Env> envs_;

    std::vector<Real> observations_;
    std::vector<Real> rewardTerms_;
    std::vector<uint8_t> terminated_;
    std::vector<uint8_t> truncated_;
    std::vector<uint32_t> episodeStep_;
    std::vector<uint8_t> finalMask_;
    std::vector<Real> finalObservations_;
    std::vector<Real> finalScratch_;
};

using EnvBatch = EnvBatchT<Env2D>;
using EnvBatch3D = EnvBatchT<Env3D>;

}  // namespace aibf
