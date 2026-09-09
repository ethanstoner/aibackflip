// Serves an EnvBatch over UDP.
//
// The exchange is a strict request/response over control steps:
//
//   HELLO  -> SPEC     dimensions, joint limits, names; establishes a session
//   RESET  -> STATE    every environment reset, step counter back to 0
//   ACTION -> STATE    apply, advance one control step, report
//
// Reliability comes from making the exchange idempotent rather than from
// retransmission timers. Every ACTION names the step it is answering. An ACTION
// for the current step is applied; an ACTION for the step just completed means
// the client never received that reply, so the cached STATE is sent again;
// anything else is stale and dropped. A datagram lost in either direction
// therefore costs one retry and nothing else.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "env/EnvBatch.h"
#include "humanoid/Observation.h"
#include "humanoid/Observation3D.h"
#include "humanoid/RewardTerms.h"
#include "net/Protocol.h"
#include "net/UDPSocket.h"

namespace aibf::net {

// What differs between the 2D and 3D figures, which is only two things: how the
// action vector is described in the SPEC handshake, and what the observation
// slots are called. Everything else about serving a batch over UDP is identical,
// so the server itself is written once.
// Hoisted out of EnvServerT on purpose. As a nested type these would be two
// unrelated types, one per instantiation, and a caller could not hand the same
// parsed command line to the 2D and 3D servers.
struct ServerOptions {
    uint16_t port = 51234;
    std::string bindAddress = "127.0.0.1";
    int numEnvs = 25;
    uint64_t seed = 1;
    bool verbose = false;
    // A disconnecting client normally means "this run is over", not "shut
    // the simulator down": a trainer that crashes and restarts should be
    // able to reconnect without the environment host having to be relaunched
    // too. Scripted one-shot runs can opt into exiting instead.
    bool exitOnBye = false;
};

struct ServerStats {
    uint64_t packetsReceived = 0;
    uint64_t packetsSent = 0;
    uint64_t duplicateRequestsAnswered = 0;
    uint64_t stalePacketsDropped = 0;
    uint64_t malformedPacketsDropped = 0;
    uint64_t controlStepsServed = 0;
    uint64_t episodesFinished = 0;
    uint32_t step = 0;
    bool clientKnown = false;
};



// Largest STATE datagram this batch can produce, in bytes.
//
// The worst case is every environment finishing on the same step, because then
// the final-observation block is as large as the observation block. That is not
// a hypothetical: with a short episode limit and synchronised resets it happens
// on the first step and then periodically forever.
//
// Worth computing rather than assuming. One datagram per control step is the
// whole reason the bridge is fast, and the 2D figure at 32 environments used
// about 26 KB of the budget, which made the limit feel far away. The 3D figure
// has a 204-value observation instead of 111, and 64 environments of it needs
// roughly 110 KB. The design has a real ceiling and 3D is the first thing to
// reach it.
template <typename Env>
constexpr size_t worstCaseStateBytes(int numEnvs) {
    const size_t n = static_cast<size_t>(numEnvs);
    const size_t obs = n * static_cast<size_t>(Env::observationDim()) * sizeof(float);
    const size_t rewards = n * static_cast<size_t>(Env::rewardTermCount()) * sizeof(float);
    const size_t flags = n * (1 + 1 + 4 + 1);  // terminated, truncated, step, final mask
    return kStateHeaderBytes + obs + rewards + flags + obs;  // the trailing obs is the finals
}

// How many environments fit in one datagram. Reported in the startup error so
// the answer is a number rather than an invitation to bisect.
template <typename Env>
int maxEnvsPerDatagram() {
    int best = 1;
    while (best < 4096 && worstCaseStateBytes<Env>(best + 1) <= kMaxPacketBytes) ++best;
    return best;
}

template <typename Env>
struct EnvTraits;

template <>
struct EnvTraits<Env2D> {
    static void describeActions(const EnvConfig& config, SpecMessage& spec) {
        for (const JointConfig& joint : config.humanoid.joints) {
            spec.actionLower.push_back(static_cast<float>(joint.lowerLimit));
            spec.actionUpper.push_back(static_cast<float>(joint.upperLimit));
            spec.actionNames.push_back(joint.name);
        }
    }
    static std::vector<std::string> observationNames() {
        return ObservationLayout::fieldNames();
    }
};

template <>
struct EnvTraits<Env3D> {
    static void describeActions(const EnvConfig3D& config, SpecMessage& spec) {
        // Three entries per ball joint and one per hinge, so the SPEC describes
        // the action *vector* rather than the joint list. Python sizes its
        // policy head from this, so a ball joint contributing one name would
        // silently produce a policy two thirds too small.
        for (const JointConfig3D& joint : config.humanoid.joints) {
            if (joint.kind == JointKind::Ball) {
                const char* axes[3] = {"_swing_x", "_twist", "_swing_z"};
                const Real bounds[3] = {joint.coneAngle, joint.upperTwist, joint.coneAngle};
                for (int i = 0; i < 3; ++i) {
                    spec.actionLower.push_back(static_cast<float>(-bounds[i]));
                    spec.actionUpper.push_back(static_cast<float>(bounds[i]));
                    spec.actionNames.push_back(joint.name + axes[i]);
                }
            } else {
                spec.actionLower.push_back(static_cast<float>(joint.lowerLimit));
                spec.actionUpper.push_back(static_cast<float>(joint.upperLimit));
                spec.actionNames.push_back(joint.name);
            }
        }
    }
    static std::vector<std::string> observationNames() {
        return ObservationLayout3D::fieldNames();
    }
};
template <typename Env>
class EnvServerT {
public:
    using Batch = EnvBatchT<Env>;
    using Config = typename Env::Config;

    using Options = ServerOptions;
    using Stats = ServerStats;

    bool start(const Config& config, const Options& options);
    void stop();
    bool isRunning() const { return socket_.isOpen(); }

    // Handles up to `maxMessages` datagrams, waiting at most timeoutMs for the
    // first one (0 polls, negative blocks). Returns how many were handled.
    int poll(int timeoutMs, int maxMessages = 64);

    // Blocks serving requests until stop() is called or the client says BYE.
    void run();

    Batch& batch() { return batch_; }
    const Batch& batch() const { return batch_; }
    const Stats& stats() const { return stats_; }
    uint16_t port() const { return socket_.boundPort(); }
    uint32_t session() const { return session_; }
    const std::string& lastError() const { return lastError_; }
    // True once a client has said goodbye and not yet reconnected.
    bool sawBye() const { return sawBye_; }
    // Whether the serving loop should stop, honouring the exitOnBye option.
    bool shouldStop() const { return options_.exitOnBye && sawBye_; }

private:
    void handlePacket(const uint8_t* data, size_t size, const Endpoint& from);
    void handleHello(const uint8_t* data, size_t size, const Endpoint& from);
    void handleReset(const uint8_t* data, size_t size, const Endpoint& from);
    void handleAction(const Header& header, const uint8_t* data, size_t size,
                      const Endpoint& from);
    void buildAndSendState(const Endpoint& to);
    void sendCachedState(const Endpoint& to);
    void sendError(const Endpoint& to, const std::string& reason);
    void rebuild(int numEnvs, uint64_t seed);
    // False, with lastError_ set, when a batch this large cannot fit in one
    // datagram. Checked at start and on every reconnect that resizes the batch,
    // so the failure is a clear message before any training rather than a
    // refused packet after the first step.
    bool checkBatchFits(int numEnvs);

    UDPSocket socket_;
    Batch batch_;
    Config config_;
    Options options_;

    uint32_t session_ = 0;
    uint32_t currentStep_ = 0;
    Endpoint client_;
    bool sawBye_ = false;

    std::vector<uint8_t> receiveBuffer_;
    std::vector<uint8_t> sendBuffer_;
    std::vector<uint8_t> cachedState_;
    uint32_t cachedStateStep_ = 0xFFFFFFFFu;

    // Scratch reused every step. The batch owns the observation and reward
    // buffers itself, so only the inbound action side needs staging here.
    std::vector<Real> actionScratch_;
    std::vector<uint8_t> resetScratch_;

    Stats stats_;
    std::string lastError_;
};

using EnvServer = EnvServerT<Env2D>;
using EnvServer3D = EnvServerT<Env3D>;

}  // namespace aibf::net

#include "net/EnvServer.inl"
