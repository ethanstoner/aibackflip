// Minimal PNG writer.
//
// Screenshots are how rendered behaviour gets verified rather than asserted, so
// the simulator needs to be able to write one without an image library. The
// deflate stream uses stored (uncompressed) blocks: a valid, universally
// readable PNG at the cost of file size, which for a debug capture is free.
#pragma once

#include <cstdint>
#include <string>

namespace aibf {

// `channels` must be 3 (RGB) or 4 (RGBA). `pixels` is row-major, tightly
// packed. OpenGL hands back rows bottom-up, so flipVertically is the normal
// case for a screenshot.
bool writePng(const std::string& path, int width, int height, int channels,
              const uint8_t* pixels, bool flipVertically);

}  // namespace aibf
