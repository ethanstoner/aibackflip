// Thin UDP socket wrapper over Winsock / BSD sockets.
//
// Deliberately minimal: bind, send, receive-with-timeout. The reliability the
// bridge needs (duplicate suppression, retries, re-handshaking) lives one layer
// up in EnvServer and the Python client, where it can be expressed in terms of
// control steps rather than bytes.
#pragma once

#include <cstdint>
#include <string>

namespace aibf::net {

struct Endpoint {
    uint32_t address = 0;  // IPv4, host byte order
    uint16_t port = 0;

    bool operator==(const Endpoint& other) const {
        return address == other.address && port == other.port;
    }
    bool operator!=(const Endpoint& other) const { return !(*this == other); }
    bool valid() const { return port != 0; }
    std::string toString() const;
};

class UDPSocket {
public:
    UDPSocket() = default;
    ~UDPSocket();
    UDPSocket(const UDPSocket&) = delete;
    UDPSocket& operator=(const UDPSocket&) = delete;

    // Binds to a local port. Port 0 asks the OS to pick one, readable
    // afterwards via boundPort().
    bool bind(uint16_t port, const std::string& address = "127.0.0.1");
    // Unbound socket for a client that only ever initiates.
    bool open();
    void close();
    bool isOpen() const { return handle_ != kInvalid; }

    // Blocks up to timeoutMs (negative blocks indefinitely). Returns the number
    // of bytes received, 0 on timeout, and -1 on error.
    int receive(uint8_t* buffer, size_t capacity, Endpoint& from, int timeoutMs);
    bool sendTo(const Endpoint& to, const uint8_t* data, size_t size);

    // Enlarging the receive buffer matters at 32 environments: the default is
    // small enough that a burst of large datagrams can be dropped by the kernel
    // before the process ever sees them.
    bool setReceiveBufferBytes(int bytes);

    uint16_t boundPort() const { return boundPort_; }
    const std::string& lastError() const { return lastError_; }

    static bool resolve(const std::string& host, uint16_t port, Endpoint& out);

private:
#ifdef _WIN32
    using Handle = uintptr_t;
    static constexpr Handle kInvalid = ~static_cast<uintptr_t>(0);
#else
    using Handle = int;
    static constexpr Handle kInvalid = -1;
#endif

    bool create();
    void recordError(const char* what);

    Handle handle_ = kInvalid;
    uint16_t boundPort_ = 0;
    std::string lastError_;
};

}  // namespace aibf::net
