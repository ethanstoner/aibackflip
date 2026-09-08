#include <cstring>

#include "core/Rng.h"
#include "core/Test.h"
#include "net/Protocol.h"
#include "net/UDPSocket.h"

using namespace aibf;
using namespace aibf::net;

namespace {

Header makeHeader(MessageType type, uint32_t session = 12345, uint32_t step = 7) {
    Header header;
    header.type = type;
    header.session = session;
    header.step = step;
    return header;
}

}  // namespace

// ---------------------------------------------------------------- header

TEST(Protocol, headerRoundTripsAndCarriesItsFields) {
    std::vector<uint8_t> bytes;
    HelloMessage hello;
    hello.numEnvs = 25;
    hello.seed = 0xDEADBEEFCAFEF00Dull;
    encodeHello(makeHeader(MessageType::Hello, 99, 4), hello, bytes);

    Header header;
    CHECK(decodeHeader(bytes.data(), bytes.size(), header));
    CHECK(header.magic == kMagic);
    CHECK(header.version == kVersion);
    CHECK(header.type == MessageType::Hello);
    CHECK(header.session == 99u);
    CHECK(header.step == 4u);
}

TEST(Protocol, theHeaderIsLittleEndianOnTheWire) {
    // Pinned explicitly rather than assumed: the whole point of hand-rolled
    // serialization is that neither side depends on host byte order, and a
    // round-trip test alone would pass even if both sides were big-endian.
    std::vector<uint8_t> bytes;
    encodeHeader(makeHeader(MessageType::State, 0x11223344u, 0xAABBCCDDu), bytes);
    CHECK(bytes.size() == kHeaderBytes);
    CHECK(bytes[0] == 'A');
    CHECK(bytes[1] == 'I');
    CHECK(bytes[2] == 'B');
    CHECK(bytes[3] == 'F');
    CHECK(bytes[4] == 1 && bytes[5] == 0);                       // version, LE
    CHECK(bytes[6] == 4 && bytes[7] == 0);                       // type State, LE
    CHECK(bytes[8] == 0x44 && bytes[11] == 0x11);                // session, LE
    CHECK(bytes[12] == 0xDD && bytes[15] == 0xAA);               // step, LE
}

TEST(Protocol, aWrongMagicOrVersionIsRejected) {
    std::vector<uint8_t> bytes;
    encodeHeader(makeHeader(MessageType::State), bytes);

    std::vector<uint8_t> badMagic = bytes;
    badMagic[0] = 'X';
    Header header;
    CHECK(!decodeHeader(badMagic.data(), badMagic.size(), header));

    std::vector<uint8_t> badVersion = bytes;
    badVersion[4] = 99;
    CHECK(!decodeHeader(badVersion.data(), badVersion.size(), header));
}

// ---------------------------------------------------------------- messages

TEST(Protocol, helloRoundTrips) {
    HelloMessage sent;
    sent.numEnvs = 32;
    sent.seed = 0x0123456789ABCDEFull;
    std::vector<uint8_t> bytes;
    encodeHello(makeHeader(MessageType::Hello), sent, bytes);

    HelloMessage received;
    CHECK(decodeHello(bytes.data(), bytes.size(), received));
    CHECK(received.numEnvs == sent.numEnvs);
    CHECK(received.seed == sent.seed);
}

TEST(Protocol, specRoundTripsIncludingNames) {
    SpecMessage sent;
    sent.numEnvs = 25;
    sent.obsDim = 110;
    sent.actionDim = 3;
    sent.rewardDim = 2;
    sent.controlHz = 60.0f;
    sent.physicsHz = 240.0f;
    sent.maxEpisodeSteps = 1000;
    sent.actionLower = {-0.5f, -2.6f, 0.0f};
    sent.actionUpper = {2.1f, 0.0f, 1.5f};
    sent.actionNames = {"hip_l", "knee_l", "ankle_l"};
    sent.rewardNames = {"alive", "pelvis_height"};
    sent.observationNames = {"pelvis_height", "pelvis_sin"};

    std::vector<uint8_t> bytes;
    encodeSpec(makeHeader(MessageType::Spec), sent, bytes);

    SpecMessage received;
    CHECK(decodeSpec(bytes.data(), bytes.size(), received));
    CHECK(received.numEnvs == 25);
    CHECK(received.obsDim == 110);
    CHECK(received.actionDim == 3);
    CHECK(received.rewardDim == 2);
    CHECK_NEAR(received.controlHz, 60.0, 1e-6);
    CHECK_NEAR(received.physicsHz, 240.0, 1e-6);
    CHECK(received.maxEpisodeSteps == 1000);
    CHECK(received.actionLower.size() == 3);
    CHECK_NEAR(received.actionLower[1], -2.6, 1e-6);
    CHECK_NEAR(received.actionUpper[0], 2.1, 1e-6);
    CHECK(received.actionNames.size() == 3);
    CHECK(received.actionNames[2] == "ankle_l");
    CHECK(received.rewardNames[1] == "pelvis_height");
    CHECK(received.observationNames[0] == "pelvis_height");
}

TEST(Protocol, actionRoundTripsIncludingResetFlags) {
    ActionMessage sent;
    sent.numEnvs = 4;
    sent.actionDim = 3;
    sent.resetMask = {0, 1, 0, 1};
    sent.actions = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f,
                    0.7f, -0.8f, 0.9f, -1.0f, 1.0f, 0.0f};

    std::vector<uint8_t> bytes;
    encodeAction(makeHeader(MessageType::Action), sent, bytes);

    ActionMessage received;
    CHECK(decodeAction(bytes.data(), bytes.size(), received));
    CHECK(received.numEnvs == 4);
    CHECK(received.actionDim == 3);
    CHECK(received.resetMask == sent.resetMask);
    CHECK(received.actions.size() == sent.actions.size());
    for (size_t i = 0; i < sent.actions.size(); ++i) {
        CHECK(received.actions[i] == sent.actions[i]);  // float32 must be exact
    }
}

TEST(Protocol, stateRoundTripsEveryBlock) {
    StateMessage sent;
    sent.numEnvs = 3;
    sent.obsDim = 4;
    sent.rewardDim = 2;
    sent.observations = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    sent.rewardTerms = {0.5f, 0.25f, 1.5f, 1.25f, 2.5f, 2.25f};
    sent.terminated = {0, 1, 0};
    sent.truncated = {0, 0, 1};
    sent.episodeStep = {10, 999, 100000};
    sent.finalMask = {0, 1, 1};
    sent.finalObservations = {21, 22, 23, 24, 31, 32, 33, 34};  // two rows of four

    std::vector<uint8_t> bytes;
    encodeState(makeHeader(MessageType::State), sent, bytes);

    StateMessage received;
    CHECK(decodeState(bytes.data(), bytes.size(), received));
    CHECK(received.numEnvs == 3);
    CHECK(received.observations == sent.observations);
    CHECK(received.rewardTerms == sent.rewardTerms);
    CHECK(received.terminated == sent.terminated);
    CHECK(received.truncated == sent.truncated);
    CHECK(received.episodeStep == sent.episodeStep);
    CHECK(received.finalMask == sent.finalMask);
    CHECK(received.finalObservations == sent.finalObservations);
}

TEST(Protocol, aStateWithNoFinishedEpisodesCarriesAnEmptyFinalBlock) {
    StateMessage sent;
    sent.numEnvs = 2;
    sent.obsDim = 3;
    sent.rewardDim = 1;
    sent.observations = {1, 2, 3, 4, 5, 6};
    sent.rewardTerms = {0.5f, 0.75f};
    sent.terminated = {0, 0};
    sent.truncated = {0, 0};
    sent.episodeStep = {5, 5};
    sent.finalMask = {0, 0};

    std::vector<uint8_t> bytes;
    encodeState(makeHeader(MessageType::State), sent, bytes);
    StateMessage received;
    CHECK(decodeState(bytes.data(), bytes.size(), received));
    CHECK(received.finalObservations.empty());
    CHECK(received.finalMask == sent.finalMask);
}

TEST(Protocol, aFinalCountLargerThanTheBatchIsRejected) {
    // popcount(finalMask) can never exceed the environment count, so a packet
    // claiming otherwise is corrupt and must not be used to size an allocation.
    StateMessage sent;
    sent.numEnvs = 2;
    sent.obsDim = 2;
    sent.rewardDim = 1;
    sent.observations = {1, 2, 3, 4};
    sent.rewardTerms = {1, 1};
    sent.terminated = {0, 0};
    sent.truncated = {0, 0};
    sent.episodeStep = {0, 0};
    sent.finalMask = {0, 0};

    std::vector<uint8_t> bytes;
    encodeState(makeHeader(MessageType::State), sent, bytes);
    // The count sits immediately after the mask, at the end of the packet.
    bytes[bytes.size() - 2] = 0xFF;
    bytes[bytes.size() - 1] = 0xFF;

    StateMessage received;
    CHECK(!decodeState(bytes.data(), bytes.size(), received));
}

TEST(Protocol, resetAndErrorRoundTrip) {
    ResetMessage reset;
    reset.seed = 4242;
    reset.reseed = 1;
    std::vector<uint8_t> bytes;
    encodeReset(makeHeader(MessageType::Reset), reset, bytes);
    ResetMessage receivedReset;
    CHECK(decodeReset(bytes.data(), bytes.size(), receivedReset));
    CHECK(receivedReset.seed == 4242u);
    CHECK(receivedReset.reseed == 1);

    ErrorMessage error;
    error.reason = "action shape does not match the spec";
    encodeError(makeHeader(MessageType::Error), error, bytes);
    ErrorMessage receivedError;
    CHECK(decodeError(bytes.data(), bytes.size(), receivedError));
    CHECK(receivedError.reason == error.reason);
}

TEST(Protocol, floatsSurviveExactlyIncludingAwkwardValues) {
    // Any rounding here would corrupt observations in a way that looks like a
    // learning problem rather than a transport problem.
    StateMessage sent;
    sent.numEnvs = 1;
    sent.obsDim = 8;
    sent.rewardDim = 0;
    sent.observations = {0.0f,        -0.0f,       1.0f / 3.0f, -1e-30f,
                         3.4028235e38f, 1.1754944e-38f, 12345.6789f, -0.1f};
    sent.terminated = {0};
    sent.truncated = {0};
    sent.episodeStep = {0};

    std::vector<uint8_t> bytes;
    encodeState(makeHeader(MessageType::State), sent, bytes);
    StateMessage received;
    CHECK(decodeState(bytes.data(), bytes.size(), received));
    for (size_t i = 0; i < sent.observations.size(); ++i) {
        CHECK(std::memcmp(&received.observations[i], &sent.observations[i], sizeof(float)) == 0);
    }
}

TEST(Protocol, theEncoderWritesExactlyTheCountsItDeclares) {
    // A caller that leaves one block unsized must not be able to emit a packet
    // no decoder can read. Here every array is deliberately short.
    StateMessage sparse;
    sparse.numEnvs = 3;
    sparse.obsDim = 4;
    sparse.rewardDim = 2;
    sparse.observations = {1.0f, 2.0f};  // should be 12 values

    std::vector<uint8_t> bytes;
    encodeState(makeHeader(MessageType::State), sparse, bytes);

    StateMessage received;
    CHECK(decodeState(bytes.data(), bytes.size(), received));
    CHECK(received.observations.size() == 12);
    CHECK_NEAR(received.observations[0], 1.0, 1e-9);
    CHECK_NEAR(received.observations[1], 2.0, 1e-9);
    CHECK_NEAR(received.observations[11], 0.0, 1e-9);  // zero-filled
    CHECK(received.rewardTerms.size() == 6);
    CHECK(received.terminated.size() == 3);
    CHECK(received.truncated.size() == 3);
    CHECK(received.episodeStep.size() == 3);
    CHECK(received.finalMask.size() == 3);
    CHECK(received.finalObservations.empty());

    ActionMessage sparseAction;
    sparseAction.numEnvs = 2;
    sparseAction.actionDim = 5;
    sparseAction.actions = {0.5f};
    encodeAction(makeHeader(MessageType::Action), sparseAction, bytes);
    ActionMessage receivedAction;
    CHECK(decodeAction(bytes.data(), bytes.size(), receivedAction));
    CHECK(receivedAction.actions.size() == 10);
    CHECK(receivedAction.resetMask.size() == 2);
}

TEST(Protocol, randomFloatPayloadsRoundTripBitExactly) {
    Rng rng(1234);
    for (int trial = 0; trial < 200; ++trial) {
        ActionMessage sent;
        sent.numEnvs = static_cast<uint16_t>(1 + rng.below(8));
        sent.actionDim = static_cast<uint16_t>(1 + rng.below(16));
        sent.resetMask.assign(sent.numEnvs, 0);
        for (uint16_t i = 0; i < sent.numEnvs; ++i) {
            sent.resetMask[i] = rng.chance(Real(0.3)) ? 1 : 0;
        }
        const size_t count = size_t(sent.numEnvs) * size_t(sent.actionDim);
        sent.actions.resize(count);
        for (size_t i = 0; i < count; ++i) sent.actions[i] = float(rng.uniform(-3, 3));

        std::vector<uint8_t> bytes;
        encodeAction(makeHeader(MessageType::Action), sent, bytes);
        ActionMessage received;
        CHECK(decodeAction(bytes.data(), bytes.size(), received));
        CHECK(received.actions == sent.actions);
        CHECK(received.resetMask == sent.resetMask);
    }
}

// ---------------------------------------------------------------- robustness

TEST(Protocol, truncatedPacketsAreRejectedRatherThanReadPastTheEnd) {
    StateMessage sent;
    sent.numEnvs = 4;
    sent.obsDim = 16;
    sent.rewardDim = 4;
    sent.observations.assign(64, 1.0f);
    sent.rewardTerms.assign(16, 0.5f);
    sent.terminated.assign(4, 0);
    sent.truncated.assign(4, 0);
    sent.episodeStep.assign(4, 3);
    sent.finalMask = {0, 1, 0, 1};
    sent.finalObservations.assign(32, 2.0f);  // two rows of 16

    std::vector<uint8_t> bytes;
    encodeState(makeHeader(MessageType::State), sent, bytes);

    // Every possible truncation must fail cleanly, not just a convenient one.
    for (size_t cut = 0; cut < bytes.size(); ++cut) {
        StateMessage received;
        CHECK(!decodeState(bytes.data(), cut, received));
    }
    StateMessage complete;
    CHECK(decodeState(bytes.data(), bytes.size(), complete));
}

TEST(Protocol, aLyingLengthFieldCannotForceAHugeAllocation) {
    // A UDP endpoint accepts datagrams from anywhere. A count field claiming
    // 65535 environments in a 40-byte packet must be refused on the arithmetic,
    // before anything is resized.
    std::vector<uint8_t> bytes;
    StateMessage sent;
    sent.numEnvs = 1;
    sent.obsDim = 2;
    sent.rewardDim = 1;
    sent.observations = {1.0f, 2.0f};
    sent.rewardTerms = {1.0f};
    sent.terminated = {0};
    sent.truncated = {0};
    sent.episodeStep = {0};
    encodeState(makeHeader(MessageType::State), sent, bytes);

    bytes[kHeaderBytes + 0] = 0xFF;  // numEnvs = 65535
    bytes[kHeaderBytes + 1] = 0xFF;
    bytes[kHeaderBytes + 2] = 0xFF;  // obsDim = 65535
    bytes[kHeaderBytes + 3] = 0xFF;

    StateMessage received;
    CHECK(!decodeState(bytes.data(), bytes.size(), received));
}

TEST(Protocol, decodingTheWrongMessageTypeFails) {
    std::vector<uint8_t> bytes;
    HelloMessage hello;
    hello.numEnvs = 4;
    encodeHello(makeHeader(MessageType::Hello), hello, bytes);

    ActionMessage action;
    CHECK(!decodeAction(bytes.data(), bytes.size(), action));
    StateMessage state;
    CHECK(!decodeState(bytes.data(), bytes.size(), state));
}

TEST(Protocol, garbageIsAlwaysRejectedAndNeverCrashes) {
    Rng rng(9);
    for (int trial = 0; trial < 3000; ++trial) {
        std::vector<uint8_t> bytes(1 + rng.below(200));
        for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng.below(256));
        // Give a fraction of them a valid preamble, so the fuzz reaches the
        // payload decoders rather than bouncing off the magic check.
        if (trial % 3 == 0 && bytes.size() >= kHeaderBytes) {
            bytes[0] = 'A'; bytes[1] = 'I'; bytes[2] = 'B'; bytes[3] = 'F';
            bytes[4] = 1; bytes[5] = 0;
            bytes[6] = static_cast<uint8_t>(1 + rng.below(7));
            bytes[7] = 0;
        }
        HelloMessage hello;
        SpecMessage spec;
        ActionMessage action;
        StateMessage state;
        ResetMessage reset;
        ErrorMessage error;
        decodeHello(bytes.data(), bytes.size(), hello);
        decodeSpec(bytes.data(), bytes.size(), spec);
        decodeAction(bytes.data(), bytes.size(), action);
        decodeState(bytes.data(), bytes.size(), state);
        decodeReset(bytes.data(), bytes.size(), reset);
        decodeError(bytes.data(), bytes.size(), error);
    }
    CHECK(true);  // reaching here without a crash or a hang is the assertion
}

TEST(Protocol, oversizedInputIsRefused) {
    std::vector<uint8_t> huge(kMaxPacketBytes + 1, 0);
    huge[0] = 'A'; huge[1] = 'I'; huge[2] = 'B'; huge[3] = 'F';
    huge[4] = 1; huge[5] = 0;
    Header header;
    CHECK(!decodeHeader(huge.data(), huge.size(), header));
    CHECK(!decodeHeader(nullptr, 32, header));
    CHECK(!decodeHeader(huge.data(), 0, header));
}

// ---------------------------------------------------------------- sockets

TEST(Socket, aBoundSocketReportsItsPortAndRoundTripsADatagram) {
    UDPSocket server;
    CHECK(server.bind(0));  // let the OS choose
    CHECK(server.boundPort() != 0);

    UDPSocket client;
    CHECK(client.open());

    Endpoint destination;
    CHECK(UDPSocket::resolve("127.0.0.1", server.boundPort(), destination));

    const uint8_t payload[] = {1, 2, 3, 4, 5};
    CHECK(client.sendTo(destination, payload, sizeof(payload)));

    uint8_t buffer[64] = {};
    Endpoint from;
    const int received = server.receive(buffer, sizeof(buffer), from, 1000);
    CHECK(received == int(sizeof(payload)));
    CHECK(std::memcmp(buffer, payload, sizeof(payload)) == 0);
    CHECK(from.port != 0);

    // And back the other way, to the address the server learned from the packet.
    const uint8_t reply[] = {9, 8, 7};
    CHECK(server.sendTo(from, reply, sizeof(reply)));
    const int echoed = client.receive(buffer, sizeof(buffer), from, 1000);
    CHECK(echoed == int(sizeof(reply)));
}

TEST(Socket, receiveTimesOutWithoutBlockingForever) {
    UDPSocket socket;
    CHECK(socket.bind(0));
    uint8_t buffer[16];
    Endpoint from;
    CHECK(socket.receive(buffer, sizeof(buffer), from, 20) == 0);
    // A zero timeout must poll rather than block.
    CHECK(socket.receive(buffer, sizeof(buffer), from, 0) == 0);
}

TEST(Socket, operationsOnAClosedSocketFailQuietly) {
    UDPSocket socket;
    uint8_t buffer[16];
    Endpoint from;
    CHECK(!socket.isOpen());
    CHECK(socket.receive(buffer, sizeof(buffer), from, 0) == -1);
    Endpoint anywhere{0x7F000001u, 9999};
    CHECK(!socket.sendTo(anywhere, buffer, sizeof(buffer)));

    CHECK(socket.bind(0));
    CHECK(socket.isOpen());
    socket.close();
    CHECK(!socket.isOpen());
    CHECK(socket.receive(buffer, sizeof(buffer), from, 0) == -1);
}

TEST(Socket, endpointFormatting) {
    Endpoint endpoint{0x7F000001u, 51234};
    CHECK(endpoint.toString() == "127.0.0.1:51234");
    CHECK(endpoint.valid());
    CHECK(!Endpoint{}.valid());
}
