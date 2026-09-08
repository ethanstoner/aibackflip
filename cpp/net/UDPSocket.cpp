#include "net/UDPSocket.h"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace aibf::net {

namespace {

#ifdef _WIN32
// Winsock needs process-wide initialisation. Reference counted so that opening
// and closing sockets in any order cannot tear it down while one is still live.
struct WinsockGuard {
    WinsockGuard() {
        WSADATA data;
        ok = WSAStartup(MAKEWORD(2, 2), &data) == 0;
    }
    ~WinsockGuard() {
        if (ok) WSACleanup();
    }
    bool ok = false;
};

bool ensureWinsock() {
    static WinsockGuard guard;
    return guard.ok;
}

int lastSocketError() { return WSAGetLastError(); }
#else
bool ensureWinsock() { return true; }
int lastSocketError() { return errno; }
#endif

}  // namespace

std::string Endpoint::toString() const {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%u.%u.%u.%u:%u", (address >> 24) & 0xFFu,
                  (address >> 16) & 0xFFu, (address >> 8) & 0xFFu, address & 0xFFu, port);
    return buffer;
}

UDPSocket::~UDPSocket() { close(); }

void UDPSocket::recordError(const char* what) {
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "%s failed (error %d)", what, lastSocketError());
    lastError_ = buffer;
}

bool UDPSocket::create() {
    if (!ensureWinsock()) {
        lastError_ = "winsock initialisation failed";
        return false;
    }
    close();
#ifdef _WIN32
    handle_ = static_cast<Handle>(::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP));
#else
    handle_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
#endif
    if (handle_ == kInvalid) {
        recordError("socket");
        return false;
    }
    return true;
}

bool UDPSocket::open() {
    if (!create()) return false;
    boundPort_ = 0;
    setReceiveBufferBytes(4 * 1024 * 1024);
    return true;
}

bool UDPSocket::bind(uint16_t port, const std::string& address) {
    if (!create()) return false;

    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_port = htons(port);
    if (address.empty() || address == "0.0.0.0") {
        local.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (::inet_pton(AF_INET, address.c_str(), &local.sin_addr) != 1) {
        lastError_ = "could not parse bind address " + address;
        close();
        return false;
    }

    if (::bind(static_cast<
#ifdef _WIN32
                   SOCKET
#else
                   int
#endif
                   >(handle_),
               reinterpret_cast<sockaddr*>(&local), sizeof(local)) != 0) {
        recordError("bind");
        close();
        return false;
    }

    sockaddr_in actual{};
#ifdef _WIN32
    int length = static_cast<int>(sizeof(actual));
#else
    socklen_t length = sizeof(actual);
#endif
    if (::getsockname(static_cast<
#ifdef _WIN32
                          SOCKET
#else
                          int
#endif
                          >(handle_),
                      reinterpret_cast<sockaddr*>(&actual), &length) == 0) {
        boundPort_ = ntohs(actual.sin_port);
    } else {
        boundPort_ = port;
    }

    setReceiveBufferBytes(4 * 1024 * 1024);
    return true;
}

void UDPSocket::close() {
    if (handle_ == kInvalid) return;
#ifdef _WIN32
    ::closesocket(static_cast<SOCKET>(handle_));
#else
    ::close(handle_);
#endif
    handle_ = kInvalid;
    boundPort_ = 0;
}

bool UDPSocket::setReceiveBufferBytes(int bytes) {
    if (handle_ == kInvalid) return false;
#ifdef _WIN32
    const int result = ::setsockopt(static_cast<SOCKET>(handle_), SOL_SOCKET, SO_RCVBUF,
                                    reinterpret_cast<const char*>(&bytes), sizeof(bytes));
#else
    const int result = ::setsockopt(handle_, SOL_SOCKET, SO_RCVBUF, &bytes, sizeof(bytes));
#endif
    return result == 0;
}

int UDPSocket::receive(uint8_t* buffer, size_t capacity, Endpoint& from, int timeoutMs) {
    if (handle_ == kInvalid) return -1;

    if (timeoutMs >= 0) {
        fd_set readable;
        FD_ZERO(&readable);
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4127)  // FD_SET expands to a constant-condition loop
        FD_SET(static_cast<SOCKET>(handle_), &readable);
#pragma warning(pop)
#else
        FD_SET(handle_, &readable);
#endif
        timeval timeout{};
        timeout.tv_sec = timeoutMs / 1000;
        timeout.tv_usec = (timeoutMs % 1000) * 1000;
#ifdef _WIN32
        const int ready = ::select(0, &readable, nullptr, nullptr, &timeout);
#else
        const int ready = ::select(handle_ + 1, &readable, nullptr, nullptr, &timeout);
#endif
        if (ready == 0) return 0;  // timed out
        if (ready < 0) {
            recordError("select");
            return -1;
        }
    }

    sockaddr_in sender{};
#ifdef _WIN32
    int length = static_cast<int>(sizeof(sender));
    const int received = ::recvfrom(static_cast<SOCKET>(handle_),
                                   reinterpret_cast<char*>(buffer), static_cast<int>(capacity), 0,
                                   reinterpret_cast<sockaddr*>(&sender), &length);
#else
    socklen_t length = sizeof(sender);
    const int received =
        static_cast<int>(::recvfrom(handle_, buffer, capacity, 0,
                                    reinterpret_cast<sockaddr*>(&sender), &length));
#endif
    if (received < 0) {
#ifdef _WIN32
        // A previous send to a closed port makes the *next* recvfrom fail with
        // WSAECONNRESET on Windows, which is a UDP-specific quirk and not an
        // error worth reporting: the peer simply is not listening yet.
        if (lastSocketError() == WSAECONNRESET) return 0;
#endif
        recordError("recvfrom");
        return -1;
    }

    from.address = ntohl(sender.sin_addr.s_addr);
    from.port = ntohs(sender.sin_port);
    return received;
}

bool UDPSocket::sendTo(const Endpoint& to, const uint8_t* data, size_t size) {
    if (handle_ == kInvalid || !to.valid()) return false;

    sockaddr_in destination{};
    destination.sin_family = AF_INET;
    destination.sin_port = htons(to.port);
    destination.sin_addr.s_addr = htonl(to.address);

#ifdef _WIN32
    const int sent = ::sendto(static_cast<SOCKET>(handle_),
                              reinterpret_cast<const char*>(data), static_cast<int>(size), 0,
                              reinterpret_cast<const sockaddr*>(&destination),
                              static_cast<int>(sizeof(destination)));
#else
    const int sent = static_cast<int>(::sendto(handle_, data, size, 0,
                                               reinterpret_cast<const sockaddr*>(&destination),
                                               sizeof(destination)));
#endif
    if (sent < 0) {
        recordError("sendto");
        return false;
    }
    return static_cast<size_t>(sent) == size;
}

bool UDPSocket::resolve(const std::string& host, uint16_t port, Endpoint& out) {
    if (!ensureWinsock()) return false;
    in_addr address{};
    if (::inet_pton(AF_INET, host.c_str(), &address) == 1) {
        out.address = ntohl(address.s_addr);
        out.port = port;
        return true;
    }

    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* results = nullptr;
    if (::getaddrinfo(host.c_str(), nullptr, &hints, &results) != 0 || !results) return false;

    const sockaddr_in* resolved = reinterpret_cast<const sockaddr_in*>(results->ai_addr);
    out.address = ntohl(resolved->sin_addr.s_addr);
    out.port = port;
    ::freeaddrinfo(results);
    return true;
}

}  // namespace aibf::net
