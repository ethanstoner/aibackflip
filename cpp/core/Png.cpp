#include "core/Png.h"

#include <cstring>
#include <fstream>
#include <vector>

namespace aibf {

namespace {

uint32_t crcTableEntry(uint32_t n) {
    uint32_t c = n;
    for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    return c;
}

uint32_t crc32(const uint8_t* data, size_t length, uint32_t start = 0xFFFFFFFFu) {
    static const std::vector<uint32_t> table = [] {
        std::vector<uint32_t> t(256);
        for (uint32_t n = 0; n < 256; ++n) t[n] = crcTableEntry(n);
        return t;
    }();
    uint32_t c = start;
    for (size_t i = 0; i < length; ++i) {
        c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c;
}

uint32_t adler32(const uint8_t* data, size_t length) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < length; ++i) {
        a = (a + data[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

void pushBigEndian32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v >> 24));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v));
}

void pushChunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& payload) {
    pushBigEndian32(out, static_cast<uint32_t>(payload.size()));
    const size_t crcStart = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), payload.begin(), payload.end());
    const uint32_t crc = crc32(out.data() + crcStart, out.size() - crcStart) ^ 0xFFFFFFFFu;
    pushBigEndian32(out, crc);
}

}  // namespace

bool writePng(const std::string& path, int width, int height, int channels,
              const uint8_t* pixels, bool flipVertically) {
    if (!pixels || width <= 0 || height <= 0 || (channels != 3 && channels != 4)) return false;

    const size_t stride = static_cast<size_t>(width) * static_cast<size_t>(channels);

    // PNG scanlines are prefixed with a filter byte; filter 0 means "none".
    std::vector<uint8_t> raw;
    raw.reserve((stride + 1) * static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) {
        const int sourceRow = flipVertically ? (height - 1 - y) : y;
        raw.push_back(0);
        const uint8_t* row = pixels + static_cast<size_t>(sourceRow) * stride;
        raw.insert(raw.end(), row, row + stride);
    }

    // zlib stream: 0x78 0x01 (deflate, 32K window, no preset dictionary),
    // then stored blocks, then the adler32 of the uncompressed data.
    std::vector<uint8_t> zlib;
    zlib.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
    zlib.push_back(0x78);
    zlib.push_back(0x01);

    const size_t kMaxBlock = 65535;
    size_t offset = 0;
    if (raw.empty()) {
        zlib.push_back(0x01);
        zlib.push_back(0);
        zlib.push_back(0);
        zlib.push_back(0xFF);
        zlib.push_back(0xFF);
    }
    while (offset < raw.size()) {
        const size_t block = (raw.size() - offset < kMaxBlock) ? (raw.size() - offset) : kMaxBlock;
        const bool last = (offset + block) >= raw.size();
        zlib.push_back(last ? 1 : 0);
        // Stored-block length and its one's complement, little-endian.
        zlib.push_back(static_cast<uint8_t>(block & 0xFF));
        zlib.push_back(static_cast<uint8_t>((block >> 8) & 0xFF));
        zlib.push_back(static_cast<uint8_t>(~block & 0xFF));
        zlib.push_back(static_cast<uint8_t>((~block >> 8) & 0xFF));
        zlib.insert(zlib.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                    raw.begin() + static_cast<std::ptrdiff_t>(offset + block));
        offset += block;
    }
    pushBigEndian32(zlib, adler32(raw.data(), raw.size()));

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};

    std::vector<uint8_t> ihdr;
    pushBigEndian32(ihdr, static_cast<uint32_t>(width));
    pushBigEndian32(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(8);                                     // bit depth
    ihdr.push_back(channels == 4 ? uint8_t(6) : uint8_t(2));  // RGBA or RGB
    ihdr.push_back(0);                                     // deflate
    ihdr.push_back(0);                                     // adaptive filtering
    ihdr.push_back(0);                                     // no interlace
    pushChunk(png, "IHDR", ihdr);
    pushChunk(png, "IDAT", zlib);
    pushChunk(png, "IEND", {});

    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(png.data()),
               static_cast<std::streamsize>(png.size()));
    return file.good();
}

}  // namespace aibf
