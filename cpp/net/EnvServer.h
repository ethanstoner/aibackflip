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
#include "net/Protocol.h"
#include "net/UDPSocket.h"

namespace aibf::net {

class EnvServer {
public:
    struct Options {
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

    struct Stats {
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

    bool start(const EnvConfig& config, const Options& options);
    void stop();
    bool isRunning() const { return socket_.isOpen(); }

    // Handles up to `maxMessages` datagrams, waiting at most timeoutMs for the
    // first one (0 polls, negative blocks). Returns how many were handled.
    int poll(int timeoutMs, int maxMessages = 64);

    // Blocks serving requests until stop() is called or the client says BYE.
    void run();

    EnvBatch& batch() { return batch_; }
    const EnvBatch& batch() const { return batch_; }
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

    UDPSocket socket_;
    EnvBatch batch_;
    EnvConfig config_;
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

}  // namespace aibf::net
