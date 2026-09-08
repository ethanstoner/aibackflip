#include "net/Protocol.h"

#include <cstring>

namespace aibf::net {

namespace {

// ---- writing ----

void putU8(std::vector<uint8_t>& out, uint8_t v) { out.push_back(v); }

void putU16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
}

void putU32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
}

void putU64(std::vector<uint8_t>& out, uint64_t v) {
    putU32(out, static_cast<uint32_t>(v & 0xFFFFFFFFull));
    putU32(out, static_cast<uint32_t>((v >> 32) & 0xFFFFFFFFull));
}

void putF32(std::vector<uint8_t>& out, float v) {
    // memcpy rather than a reinterpret_cast: type punning through a pointer is
    // undefined, and this compiles to the same single move.
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    putU32(out, bits);
}

void putString(std::vector<uint8_t>& out, const std::string& s) {
    const uint16_t length = static_cast<uint16_t>(s.size() > 0xFFFF ? 0xFFFF : s.size());
    putU16(out, length);
    out.insert(out.end(), s.begin(), s.begin() + length);
}

void putF32Array(std::vector<uint8_t>& out, const std::vector<float>& values) {
    for (const float v : values) putF32(out, v);
}

// Writes exactly `count` elements, zero-filling a short vector and ignoring a
// long one. The header declares these counts, so writing anything else produces
// a packet no decoder can read - a caller that forgets to size one block should
// not be able to emit garbage silently.
void putF32ArrayExact(std::vector<uint8_t>& out, const std::vector<float>& values, size_t count) {
    for (size_t i = 0; i < count; ++i) putF32(out, i < values.size() ? values[i] : 0.0f);
}

void putU8ArrayExact(std::vector<uint8_t>& out, const std::vector<uint8_t>& values, size_t count) {
    for (size_t i = 0; i < count; ++i) out.push_back(i < values.size() ? values[i] : uint8_t(0));
}

void putU32ArrayExact(std::vector<uint8_t>& out, const std::vector<uint32_t>& values,
                      size_t count) {
    for (size_t i = 0; i < count; ++i) putU32(out, i < values.size() ? values[i] : 0u);
}

// ---- reading ----

// Tracks a failure flag rather than throwing, so a decoder reads straight
// through and is checked once at the end.
class Reader {
public:
    Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    uint8_t u8() {
        if (pos_ + 1 > size_) return static_cast<uint8_t>(fail());
        return data_[pos_++];
    }
    uint16_t u16() {
        if (pos_ + 2 > size_) return static_cast<uint16_t>(fail());
        const uint16_t v = static_cast<uint16_t>(data_[pos_]) |
                           (static_cast<uint16_t>(data_[pos_ + 1]) << 8);
        pos_ += 2;
        return v;
    }
    uint32_t u32() {
        if (pos_ + 4 > size_) return fail();
        const uint32_t v = static_cast<uint32_t>(data_[pos_]) |
                           (static_cast<uint32_t>(data_[pos_ + 1]) << 8) |
                           (static_cast<uint32_t>(data_[pos_ + 2]) << 16) |
                           (static_cast<uint32_t>(data_[pos_ + 3]) << 24);
        pos_ += 4;
        return v;
    }
    uint64_t u64() {
        const uint64_t lo = u32();
        const uint64_t hi = u32();
        return lo | (hi << 32);
    }
    float f32() {
        const uint32_t bits = u32();
        float v = 0;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    std::string string() {
        const uint16_t length = u16();
        if (!ok_ || pos_ + length > size_) {
            fail();
            return std::string();
        }
        std::string s(reinterpret_cast<const char*>(data_ + pos_), length);
        pos_ += length;
        return s;
    }

    // Reads `count` floats, refusing up front if the buffer cannot hold them.
    // Without that check a corrupt count would reserve gigabytes before failing.
    bool f32Array(size_t count, std::vector<float>& out) {
        if (!ok_ || pos_ + count * 4 > size_) {
            fail();
            return false;
        }
        out.resize(count);
        for (size_t i = 0; i < count; ++i) out[i] = f32();
        return ok_;
    }
    bool u8Array(size_t count, std::vector<uint8_t>& out) {
        if (!ok_ || pos_ + count > size_) {
            fail();
            return false;
        }
        out.assign(data_ + pos_, data_ + pos_ + count);
        pos_ += count;
        return true;
    }
    bool u32Array(size_t count, std::vector<uint32_t>& out) {
        if (!ok_ || pos_ + count * 4 > size_) {
            fail();
            return false;
        }
        out.resize(count);
        for (size_t i = 0; i < count; ++i) out[i] = u32();
        return ok_;
    }

    void skipHeader() { pos_ = kHeaderBytes; }
    bool ok() const { return ok_; }

private:
    uint32_t fail() {
        ok_ = false;
        return 0;
    }

    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
    bool ok_ = true;
};

// Shared preamble check: magic, version and the declared type.
bool openPacket(const uint8_t* data, size_t size, MessageType expected, Reader& reader) {
    Header header;
    if (!decodeHeader(data, size, header)) return false;
    if (header.type != expected) return false;
    reader.skipHeader();
    return true;
}

}  // namespace

const char* messageTypeName(MessageType type) {
    switch (type) {
        case MessageType::Hello: return "HELLO";
        case MessageType::Spec: return "SPEC";
        case MessageType::Action: return "ACTION";
        case MessageType::State: return "STATE";
        case MessageType::Reset: return "RESET";
        case MessageType::Bye: return "BYE";
        case MessageType::Error: return "ERROR";
        default: return "UNKNOWN";
    }
}

// ---------------------------------------------------------------- encode

void encodeHeader(const Header& header, std::vector<uint8_t>& out) {
    putU32(out, kMagic);
    putU16(out, header.version);
    putU16(out, static_cast<uint16_t>(header.type));
    putU32(out, header.session);
    putU32(out, header.step);
}

void encodeHello(const Header& header, const HelloMessage& message, std::vector<uint8_t>& out) {
    out.clear();
    encodeHeader(header, out);
    putU16(out, message.numEnvs);
    putU64(out, message.seed);
}

void encodeSpec(const Header& header, const SpecMessage& message, std::vector<uint8_t>& out) {
    out.clear();
    encodeHeader(header, out);
    putU16(out, message.numEnvs);
    putU16(out, message.obsDim);
    putU16(out, message.actionDim);
    putU16(out, message.rewardDim);
    putF32(out, message.controlHz);
    putF32(out, message.physicsHz);
    putU16(out, message.maxEpisodeSteps);
    putF32Array(out, message.actionLower);
    putF32Array(out, message.actionUpper);

    putU16(out, static_cast<uint16_t>(message.actionNames.size()));
    for (const std::string& name : message.actionNames) putString(out, name);
    putU16(out, static_cast<uint16_t>(message.rewardNames.size()));
    for (const std::string& name : message.rewardNames) putString(out, name);
    putU16(out, static_cast<uint16_t>(message.observationNames.size()));
    for (const std::string& name : message.observationNames) putString(out, name);
}

void encodeAction(const Header& header, const ActionMessage& message, std::vector<uint8_t>& out) {
    out.clear();
    encodeHeader(header, out);
    putU16(out, message.numEnvs);
    putU16(out, message.actionDim);
    putU8ArrayExact(out, message.resetMask, message.numEnvs);
    putF32ArrayExact(out, message.actions,
                     static_cast<size_t>(message.numEnvs) * message.actionDim);
}

void encodeState(const Header& header, const StateMessage& message, std::vector<uint8_t>& out) {
    out.clear();
    encodeHeader(header, out);
    const size_t envs = message.numEnvs;
    putU16(out, message.numEnvs);
    putU16(out, message.obsDim);
    putU16(out, message.rewardDim);
    putF32ArrayExact(out, message.observations, envs * message.obsDim);
    putF32ArrayExact(out, message.rewardTerms, envs * message.rewardDim);
    putU8ArrayExact(out, message.terminated, envs);
    putU8ArrayExact(out, message.truncated, envs);
    putU32ArrayExact(out, message.episodeStep, envs);
    putU8ArrayExact(out, message.finalMask, envs);

    // The count is sent explicitly rather than derived from the mask so the
    // decoder can size the block without trusting a popcount it has not read.
    const size_t finalRows =
        message.obsDim > 0 ? message.finalObservations.size() / message.obsDim : 0;
    putU16(out, static_cast<uint16_t>(finalRows));
    putF32ArrayExact(out, message.finalObservations, finalRows * message.obsDim);
}

void encodeReset(const Header& header, const ResetMessage& message, std::vector<uint8_t>& out) {
    out.clear();
    encodeHeader(header, out);
    putU64(out, message.seed);
    putU8(out, message.reseed);
}

void encodeError(const Header& header, const ErrorMessage& message, std::vector<uint8_t>& out) {
    out.clear();
    encodeHeader(header, out);
    putString(out, message.reason);
}

// ---------------------------------------------------------------- decode

bool decodeHeader(const uint8_t* data, size_t size, Header& out) {
    if (!data || size < kHeaderBytes || size > kMaxPacketBytes) return false;
    Reader reader(data, size);
    out.magic = reader.u32();
    if (out.magic != kMagic) return false;
    out.version = reader.u16();
    if (out.version != kVersion) return false;
    out.type = static_cast<MessageType>(reader.u16());
    out.session = reader.u32();
    out.step = reader.u32();
    return reader.ok();
}

bool decodeHello(const uint8_t* data, size_t size, HelloMessage& out) {
    Reader reader(data, size);
    if (!openPacket(data, size, MessageType::Hello, reader)) return false;
    out.numEnvs = reader.u16();
    out.seed = reader.u64();
    return reader.ok();
}

bool decodeSpec(const uint8_t* data, size_t size, SpecMessage& out) {
    Reader reader(data, size);
    if (!openPacket(data, size, MessageType::Spec, reader)) return false;

    out.numEnvs = reader.u16();
    out.obsDim = reader.u16();
    out.actionDim = reader.u16();
    out.rewardDim = reader.u16();
    out.controlHz = reader.f32();
    out.physicsHz = reader.f32();
    out.maxEpisodeSteps = reader.u16();
    if (!reader.ok()) return false;

    if (!reader.f32Array(out.actionDim, out.actionLower)) return false;
    if (!reader.f32Array(out.actionDim, out.actionUpper)) return false;

    auto readNames = [&reader](std::vector<std::string>& names) {
        const uint16_t count = reader.u16();
        if (!reader.ok()) return false;
        names.clear();
        names.reserve(count);
        for (uint16_t i = 0; i < count; ++i) {
            names.push_back(reader.string());
            if (!reader.ok()) return false;
        }
        return true;
    };
    if (!readNames(out.actionNames)) return false;
    if (!readNames(out.rewardNames)) return false;
    if (!readNames(out.observationNames)) return false;
    return reader.ok();
}

bool decodeAction(const uint8_t* data, size_t size, ActionMessage& out) {
    Reader reader(data, size);
    if (!openPacket(data, size, MessageType::Action, reader)) return false;
    out.numEnvs = reader.u16();
    out.actionDim = reader.u16();
    if (!reader.ok()) return false;
    if (!reader.u8Array(out.numEnvs, out.resetMask)) return false;
    const size_t count = static_cast<size_t>(out.numEnvs) * static_cast<size_t>(out.actionDim);
    if (!reader.f32Array(count, out.actions)) return false;
    return reader.ok();
}

bool decodeState(const uint8_t* data, size_t size, StateMessage& out) {
    Reader reader(data, size);
    if (!openPacket(data, size, MessageType::State, reader)) return false;
    out.numEnvs = reader.u16();
    out.obsDim = reader.u16();
    out.rewardDim = reader.u16();
    if (!reader.ok()) return false;

    if (!reader.f32Array(static_cast<size_t>(out.numEnvs) * out.obsDim, out.observations)) {
        return false;
    }
    if (!reader.f32Array(static_cast<size_t>(out.numEnvs) * out.rewardDim, out.rewardTerms)) {
        return false;
    }
    if (!reader.u8Array(out.numEnvs, out.terminated)) return false;
    if (!reader.u8Array(out.numEnvs, out.truncated)) return false;
    if (!reader.u32Array(out.numEnvs, out.episodeStep)) return false;
    if (!reader.u8Array(out.numEnvs, out.finalMask)) return false;

    const uint16_t finalCount = reader.u16();
    if (!reader.ok()) return false;
    if (finalCount > out.numEnvs) return false;  // cannot exceed one per environment
    if (!reader.f32Array(static_cast<size_t>(finalCount) * out.obsDim, out.finalObservations)) {
        return false;
    }
    return reader.ok();
}

bool decodeReset(const uint8_t* data, size_t size, ResetMessage& out) {
    Reader reader(data, size);
    if (!openPacket(data, size, MessageType::Reset, reader)) return false;
    out.seed = reader.u64();
    out.reseed = reader.u8();
    return reader.ok();
}

bool decodeError(const uint8_t* data, size_t size, ErrorMessage& out) {
    Reader reader(data, size);
    if (!openPacket(data, size, MessageType::Error, reader)) return false;
    out.reason = reader.string();
    return reader.ok();
}

}  // namespace aibf::net
