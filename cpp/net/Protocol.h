// Wire format for the C++/Python bridge.
//
// Everything is serialized explicitly, little-endian, one byte at a time. No
// struct is ever memcpy'd across the boundary: C++ padding rules and numpy dtype
// alignment are free to disagree, and a mismatch there is the kind of bug that
// shows up as a policy learning from garbage rather than as a crash.
//
// Payloads are struct-of-arrays - all observations, then all reward terms, then
// the flags - so the Python side can wrap each block in a single numpy view
// instead of unpacking per environment.
//
// The one thing assumed rather than encoded is IEEE-754 binary32 for floats,
// which holds on every platform either side of this bridge runs on.
//
// Full byte layout: docs/PROTOCOL.md
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aibf::net {

// The bytes 'A','I','B','F' read back as a little-endian uint32.
constexpr uint32_t kMagic = 0x46424941u;
constexpr uint16_t kVersion = 1;

// Largest datagram either side will send or accept. A 32-environment 3D
// observation is roughly 26 KB, comfortably inside the 65507-byte UDP payload
// limit; this cap keeps a corrupt length field from allocating wildly.
constexpr size_t kMaxPacketBytes = 60000;

enum class MessageType : uint16_t {
    Hello = 1,   // python -> c++  : request a spec, declare the environment count
    Spec = 2,    // c++ -> python  : dimensions, joint limits, names
    Action = 3,  // python -> c++  : actions plus per-environment reset flags
    State = 4,   // c++ -> python  : observations, reward terms, episode flags
    Reset = 5,   // python -> c++  : reset every environment
    Bye = 6,     // python -> c++  : client is going away
    Error = 7,   // c++ -> python  : a message could not be honoured
};

constexpr size_t kHeaderBytes = 16;

struct Header {
    uint32_t magic = kMagic;
    uint16_t version = kVersion;
    MessageType type = MessageType::Hello;
    // Identifies one run of the simulator. If the server restarts, the session
    // changes and the client re-handshakes instead of silently talking to a
    // process that no longer shares its state.
    uint32_t session = 0;
    // Index of the control step this message concerns. Lets both sides discard
    // stale datagrams and detect duplicates.
    uint32_t step = 0;
};

// ---------------------------------------------------------------- messages

struct HelloMessage {
    uint16_t numEnvs = 1;
    uint64_t seed = 0;
};

struct SpecMessage {
    uint16_t numEnvs = 0;
    uint16_t obsDim = 0;
    uint16_t actionDim = 0;
    uint16_t rewardDim = 0;
    float controlHz = 0;
    float physicsHz = 0;
    uint16_t maxEpisodeSteps = 0;
    std::vector<float> actionLower;  // actionDim entries, the joint limits
    std::vector<float> actionUpper;
    std::vector<std::string> actionNames;
    std::vector<std::string> rewardNames;
    std::vector<std::string> observationNames;
};

struct ActionMessage {
    uint16_t numEnvs = 0;
    uint16_t actionDim = 0;
    std::vector<uint8_t> resetMask;  // numEnvs entries
    std::vector<float> actions;      // numEnvs * actionDim
};

struct StateMessage {
    uint16_t numEnvs = 0;
    uint16_t obsDim = 0;
    uint16_t rewardDim = 0;
    // Post-reset for any environment that ended this step; everything below is
    // the terminal value for that episode.
    std::vector<float> observations;  // numEnvs * obsDim
    std::vector<float> rewardTerms;   // numEnvs * rewardDim
    std::vector<uint8_t> terminated;  // numEnvs
    std::vector<uint8_t> truncated;   // numEnvs
    std::vector<uint32_t> episodeStep;  // numEnvs

    // Which environments auto-reset this step, and the observation each of them
    // ended on. Dense: one row per set flag, in environment order. A truncated
    // episode needs this to bootstrap its value estimate from the state it was
    // actually cut off in.
    std::vector<uint8_t> finalMask;         // numEnvs
    std::vector<float> finalObservations;   // popcount(finalMask) * obsDim
};

struct ResetMessage {
    uint64_t seed = 0;
    uint8_t reseed = 0;  // non-zero re-seeds the environments from `seed`
};

struct ErrorMessage {
    std::string reason;
};

// ---------------------------------------------------------------- encode

void encodeHeader(const Header& header, std::vector<uint8_t>& out);
void encodeHello(const Header& header, const HelloMessage& message, std::vector<uint8_t>& out);
void encodeSpec(const Header& header, const SpecMessage& message, std::vector<uint8_t>& out);
void encodeAction(const Header& header, const ActionMessage& message, std::vector<uint8_t>& out);
void encodeState(const Header& header, const StateMessage& message, std::vector<uint8_t>& out);
void encodeReset(const Header& header, const ResetMessage& message, std::vector<uint8_t>& out);
void encodeError(const Header& header, const ErrorMessage& message, std::vector<uint8_t>& out);

// ---------------------------------------------------------------- decode
//
// Every decoder validates as it reads and returns false on a short, malformed or
// implausibly-sized packet rather than trusting a length field. A UDP endpoint
// accepts datagrams from anywhere, so a decoder that trusts its input is a
// remote crash waiting for a stray packet.

bool decodeHeader(const uint8_t* data, size_t size, Header& out);
bool decodeHello(const uint8_t* data, size_t size, HelloMessage& out);
bool decodeSpec(const uint8_t* data, size_t size, SpecMessage& out);
bool decodeAction(const uint8_t* data, size_t size, ActionMessage& out);
bool decodeState(const uint8_t* data, size_t size, StateMessage& out);
bool decodeReset(const uint8_t* data, size_t size, ResetMessage& out);
bool decodeError(const uint8_t* data, size_t size, ErrorMessage& out);

const char* messageTypeName(MessageType type);

}  // namespace aibf::net
