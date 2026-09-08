#include "env/EnvBatch.h"

#include <algorithm>

namespace aibf {

void EnvBatch::initialize(const EnvConfig& config, int count, uint64_t seed) {
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

void EnvBatch::resetAll() {
    for (Env2D& env : envs_) env.reset();
    std::fill(finalMask_.begin(), finalMask_.end(), uint8_t(0));
    finalObservations_.clear();
    captureAll();
}

void EnvBatch::captureEnv(int index) {
    const size_t i = static_cast<size_t>(index);
    envs_[i].writeObservation(observations_.data() + i * static_cast<size_t>(observationDim()));
    envs_[i].writeRewardTerms(rewardTerms_.data() + i * static_cast<size_t>(rewardTermCount()));
    terminated_[i] = envs_[i].terminated() ? 1u : 0u;
    truncated_[i] = envs_[i].truncated() ? 1u : 0u;
    episodeStep_[i] = static_cast<uint32_t>(envs_[i].episodeStep());
}

void EnvBatch::captureAll() {
    for (int i = 0; i < size(); ++i) captureEnv(i);
}

void EnvBatch::step(const Real* actions, const uint8_t* resetMask) {
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

        // Everything except the observation is captured *before* the reset, so
        // the reported reward, flags and length describe the episode that just
        // ended rather than the one that just began.
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

int EnvBatch::doneCount() const {
    int count = 0;
    for (const uint8_t flag : finalMask_) count += flag ? 1 : 0;
    return count;
}

}  // namespace aibf
