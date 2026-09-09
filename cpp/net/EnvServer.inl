// Implementation of the EnvServerT template. Included by EnvServer.h rather
// than compiled on its own: a template definition has to be visible where it is
// instantiated, and keeping it out of the header keeps the interface readable.


namespace aibf::net {

namespace {

uint32_t makeSessionId() {
    // Only needs to differ between runs of the simulator, so that a client
    // talking to a restarted server notices and re-handshakes. Not security.
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    uint64_t x = static_cast<uint64_t>(micros);
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDull;
    x ^= x >> 33;
    return static_cast<uint32_t>(x) | 1u;  // never zero, which means "no session"
}

}  // namespace

template <typename Env>
bool EnvServerT<Env>::start(const Config& config, const Options& options) {
    const std::string problem = config.validate();
    if (!problem.empty()) {
        lastError_ = "invalid environment config: " + problem;
        return false;
    }

    config_ = config;
    options_ = options;
    session_ = makeSessionId();
    sawBye_ = false;
    stats_ = Stats{};

    if (!checkBatchFits(options.numEnvs)) return false;

    if (!socket_.bind(options.port, options.bindAddress)) {
        lastError_ = socket_.lastError();
        return false;
    }
    receiveBuffer_.resize(kMaxPacketBytes);
    rebuild(options.numEnvs, options.seed);
    return true;
}

template <typename Env>
void EnvServerT<Env>::stop() {
    socket_.close();
}

template <typename Env>
bool EnvServerT<Env>::checkBatchFits(int numEnvs) {
    if (worstCaseStateBytes<Env>(numEnvs) <= kMaxPacketBytes) return true;
    const int limit = maxEnvsPerDatagram<Env>();
    lastError_ = "a batch of " + std::to_string(numEnvs) + " environments needs " +
                 std::to_string(worstCaseStateBytes<Env>(numEnvs)) +
                 " bytes for a STATE packet, over the " + std::to_string(kMaxPacketBytes) +
                 " byte datagram budget; the most that fits is " + std::to_string(limit);
    return false;
}

template <typename Env>
void EnvServerT<Env>::rebuild(int numEnvs, uint64_t seed) {
    batch_.initialize(config_, numEnvs, seed);

    const size_t n = static_cast<size_t>(batch_.size());
    actionScratch_.assign(n * static_cast<size_t>(Batch::actionDim()), Real(0));
    resetScratch_.assign(n, 0);

    currentStep_ = 0;
    cachedState_.clear();
    cachedStateStep_ = 0xFFFFFFFFu;
    stats_.step = 0;
}

template <typename Env>
int EnvServerT<Env>::poll(int timeoutMs, int maxMessages) {
    if (!socket_.isOpen()) return 0;
    int handled = 0;
    for (int i = 0; i < maxMessages; ++i) {
        Endpoint from;
        // Only the first receive waits; the rest drain whatever else has
        // already arrived without blocking.
        const int received = socket_.receive(receiveBuffer_.data(), receiveBuffer_.size(), from,
                                             i == 0 ? timeoutMs : 0);
        if (received <= 0) break;
        ++stats_.packetsReceived;
        handlePacket(receiveBuffer_.data(), static_cast<size_t>(received), from);
        ++handled;
    }
    return handled;
}

template <typename Env>
void EnvServerT<Env>::run() {
    while (socket_.isOpen() && !shouldStop()) {
        poll(100);
    }
}

template <typename Env>
void EnvServerT<Env>::handlePacket(const uint8_t* data, size_t size, const Endpoint& from) {
    Header header;
    if (!decodeHeader(data, size, header)) {
        ++stats_.malformedPacketsDropped;
        return;
    }

    switch (header.type) {
        case MessageType::Hello:
            handleHello(data, size, from);
            return;
        case MessageType::Reset:
            handleReset(data, size, from);
            return;
        case MessageType::Action:
            handleAction(header, data, size, from);
            return;
        case MessageType::Bye:
            sawBye_ = true;
            // Forget the client but keep listening. A trainer that restarts
            // reconnects with a HELLO and gets a fresh batch.
            client_ = Endpoint{};
            stats_.clientKnown = false;
            if (options_.verbose) std::printf("client %s disconnected\n", from.toString().c_str());
            return;
        default:
            ++stats_.malformedPacketsDropped;
            sendError(from, "unexpected message type");
            return;
    }
}

template <typename Env>
void EnvServerT<Env>::handleHello(const uint8_t* data, size_t size, const Endpoint& from) {
    HelloMessage hello;
    if (!decodeHello(data, size, hello)) {
        ++stats_.malformedPacketsDropped;
        return;
    }

    client_ = from;
    stats_.clientKnown = true;
    sawBye_ = false;

    // A client asking for a different environment count rebuilds the batch.
    // Reconnecting with a different setting is a normal thing to want and the
    // alternative is a silent mismatch between the two sides.
    const int requested = hello.numEnvs > 0 ? hello.numEnvs : options_.numEnvs;
    if (!checkBatchFits(requested)) {
        sendError(from, lastError_);
        return;
    }
    if (requested != batch_.size()) {
        rebuild(requested, hello.seed ? hello.seed : options_.seed);
    } else if (hello.seed != 0) {
        rebuild(requested, hello.seed);
    }

    SpecMessage spec;
    spec.numEnvs = static_cast<uint16_t>(batch_.size());
    spec.obsDim = static_cast<uint16_t>(Batch::observationDim());
    spec.actionDim = static_cast<uint16_t>(Batch::actionDim());
    spec.rewardDim = static_cast<uint16_t>(Batch::rewardTermCount());
    spec.controlHz = static_cast<float>(config_.controlHz());
    spec.physicsHz = static_cast<float>(config_.physicsHz);
    spec.maxEpisodeSteps = static_cast<uint16_t>(config_.maxEpisodeSteps);

    EnvTraits<Env>::describeActions(config_, spec);
    spec.rewardNames = rewardTermNames();
    spec.observationNames = EnvTraits<Env>::observationNames();

    Header reply;
    reply.type = MessageType::Spec;
    reply.session = session_;
    reply.step = currentStep_;
    encodeSpec(reply, spec, sendBuffer_);

    if (sendBuffer_.size() > kMaxPacketBytes) {
        // Names make SPEC the largest message by far; if the observation vector
        // ever grows past what one datagram holds, this is where it shows up as
        // a clear failure rather than a truncated read.
        sendError(from, "spec packet exceeds the maximum datagram size");
        return;
    }
    if (socket_.sendTo(from, sendBuffer_.data(), sendBuffer_.size())) ++stats_.packetsSent;

    if (options_.verbose) {
        std::printf("client %s connected: %d envs, obs %d, act %d, reward terms %d\n",
                    from.toString().c_str(), batch_.size(), spec.obsDim, spec.actionDim,
                    spec.rewardDim);
    }
}

template <typename Env>
void EnvServerT<Env>::handleReset(const uint8_t* data, size_t size, const Endpoint& from) {
    ResetMessage message;
    if (!decodeReset(data, size, message)) {
        ++stats_.malformedPacketsDropped;
        return;
    }
    client_ = from;
    stats_.clientKnown = true;

    if (message.reseed) {
        rebuild(batch_.size(), message.seed);
    } else {
        batch_.resetAll();
    }
    currentStep_ = 0;
    stats_.step = 0;
    buildAndSendState(from);
}

template <typename Env>
void EnvServerT<Env>::handleAction(const Header& header, const uint8_t* data, size_t size,
                             const Endpoint& from) {
    // An ACTION for the step just completed means our reply was lost. Resending
    // the cached STATE is exactly right and must not advance the simulation:
    // stepping twice for one action would silently corrupt the rollout.
    if (cachedStateStep_ == currentStep_ && header.step + 1 == currentStep_) {
        ++stats_.duplicateRequestsAnswered;
        sendCachedState(from);
        return;
    }
    if (header.step != currentStep_) {
        ++stats_.stalePacketsDropped;
        return;
    }
    if (header.session != 0 && header.session != session_) {
        // The client is addressing a previous run of the simulator.
        sendError(from, "session mismatch; send HELLO to reconnect");
        return;
    }

    ActionMessage action;
    if (!decodeAction(data, size, action)) {
        ++stats_.malformedPacketsDropped;
        return;
    }
    if (action.numEnvs != batch_.size() || action.actionDim != Batch::actionDim()) {
        sendError(from, "action shape does not match the spec");
        return;
    }

    client_ = from;
    stats_.clientKnown = true;

    const size_t count = actionScratch_.size();
    for (size_t i = 0; i < count && i < action.actions.size(); ++i) {
        actionScratch_[i] = static_cast<Real>(action.actions[i]);
    }
    for (size_t i = 0; i < resetScratch_.size() && i < action.resetMask.size(); ++i) {
        resetScratch_[i] = action.resetMask[i];
    }

    batch_.step(actionScratch_.data(), resetScratch_.data());
    ++currentStep_;
    ++stats_.controlStepsServed;
    stats_.step = currentStep_;
    stats_.episodesFinished += static_cast<uint64_t>(batch_.doneCount());

    buildAndSendState(from);
}

template <typename Env>
void EnvServerT<Env>::buildAndSendState(const Endpoint& to) {
    StateMessage state;
    state.numEnvs = static_cast<uint16_t>(batch_.size());
    state.obsDim = static_cast<uint16_t>(Batch::observationDim());
    state.rewardDim = static_cast<uint16_t>(Batch::rewardTermCount());
    state.observations.assign(batch_.observations().begin(), batch_.observations().end());
    state.rewardTerms.assign(batch_.rewardTerms().begin(), batch_.rewardTerms().end());
    state.terminated = batch_.terminated();
    state.truncated = batch_.truncated();
    state.episodeStep = batch_.episodeStep();
    state.finalMask = batch_.finalMask();
    state.finalObservations.assign(batch_.finalObservations().begin(),
                                   batch_.finalObservations().end());

    Header header;
    header.type = MessageType::State;
    header.session = session_;
    header.step = currentStep_;
    // Encoded straight into the cache, so answering a duplicate request costs a
    // send and nothing else.
    encodeState(header, state, cachedState_);
    cachedStateStep_ = currentStep_;

    if (cachedState_.size() > kMaxPacketBytes) {
        sendError(to, "state packet exceeds the maximum datagram size");
        return;
    }
    if (socket_.sendTo(to, cachedState_.data(), cachedState_.size())) ++stats_.packetsSent;
}

template <typename Env>
void EnvServerT<Env>::sendCachedState(const Endpoint& to) {
    if (cachedState_.empty()) return;
    if (socket_.sendTo(to, cachedState_.data(), cachedState_.size())) ++stats_.packetsSent;
}

template <typename Env>
void EnvServerT<Env>::sendError(const Endpoint& to, const std::string& reason) {
    ErrorMessage message;
    message.reason = reason;
    Header header;
    header.type = MessageType::Error;
    header.session = session_;
    header.step = currentStep_;
    encodeError(header, message, sendBuffer_);
    if (socket_.sendTo(to, sendBuffer_.data(), sendBuffer_.size())) ++stats_.packetsSent;
}


}  // namespace aibf::net
