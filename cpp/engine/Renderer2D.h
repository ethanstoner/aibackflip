// Immediate-style batched 2D renderer.
//
// Draw calls append vertices to CPU buffers that are uploaded once per frame:
// one draw for triangles, one for lines. The scene is a few thousand vertices,
// so batching is about keeping the call sites simple rather than about
// throughput. Nothing here is on the training path - headless runs never
// construct a Renderer2D.
#pragma once

#include <vector>

#include "core/Math.h"

namespace aibf {

struct Color {
    Real r = 1, g = 1, b = 1, a = 1;
    constexpr Color() = default;
    constexpr Color(Real r_, Real g_, Real b_, Real a_ = 1) : r(r_), g(g_), b(b_), a(a_) {}
    constexpr Color withAlpha(Real alpha) const { return Color(r, g, b, alpha); }
    constexpr Color scaled(Real k) const { return Color(r * k, g * k, b * k, a); }
};

// Arms, legs and torso get separate hues rather than shades of one colour. In a
// sagittal 2D figure the limbs project onto the same line and spend most of
// their time overlapping the torso; without distinct hues a pose is genuinely
// unreadable, which makes every later debugging session harder.
namespace palette {
constexpr Color background{Real(0.09), Real(0.10), Real(0.13)};
constexpr Color ground{Real(0.20), Real(0.22), Real(0.27)};
constexpr Color grid{Real(0.16), Real(0.17), Real(0.21)};
constexpr Color torso{Real(0.35), Real(0.40), Real(0.58)};
constexpr Color arm{Real(0.33), Real(0.78), Real(0.72)};
constexpr Color leg{Real(0.40), Real(0.60), Real(0.92)};
constexpr Color head{Real(0.90), Real(0.80), Real(0.55)};
constexpr Color foot{Real(0.95), Real(0.56), Real(0.30)};
constexpr Color hand{Real(0.55), Real(0.88), Real(0.80)};
constexpr Color joint{Real(0.95), Real(0.95), Real(0.98)};
constexpr Color contact{Real(0.98), Real(0.36), Real(0.36)};
constexpr Color com{Real(0.99), Real(0.85), Real(0.25)};
constexpr Color accent{Real(0.45), Real(0.85), Real(0.60)};
constexpr Color trail{Real(0.60), Real(0.75), Real(0.95)};
}  // namespace palette

// Orthographic 2D camera in world units.
struct Camera2D {
    Vec2 center{0, Real(1.0)};
    Real halfHeight = Real(1.4);  // metres visible above and below the centre
    Real aspect = Real(16) / Real(9);

    Real halfWidth() const { return halfHeight * aspect; }
    Mat4 viewProjection() const;
    Vec2 screenToWorld(const Vec2& pixel, int screenWidth, int screenHeight) const;
    Vec2 worldToScreen(const Vec2& world, int screenWidth, int screenHeight) const;
    // Scales a pixel distance into world units; used to keep debug markers a
    // constant on-screen size while zooming.
    Real pixelsToWorld(Real pixels, int screenHeight) const;
};

class Renderer2D {
public:
    ~Renderer2D();

    bool initialize();
    void shutdown();

    void beginFrame(const Camera2D& camera, const Color& clear);
    void endFrame();

    // Capsule with its axis along local +Y, matching RigidBody2D.
    void capsule(const Vec2& center, Real angle, Real radius, Real halfLength, const Color& color);
    void capsuleOutline(const Vec2& center, Real angle, Real radius, Real halfLength,
                        const Color& color);
    void circle(const Vec2& center, Real radius, const Color& color);
    void circleOutline(const Vec2& center, Real radius, const Color& color);
    void line(const Vec2& a, const Vec2& b, const Color& color);
    void thickLine(const Vec2& a, const Vec2& b, Real width, const Color& color);
    void quad(const Vec2& min, const Vec2& max, const Color& color);
    void triangle(const Vec2& a, const Vec2& b, const Vec2& c, const Color& color);
    void arrow(const Vec2& from, const Vec2& to, Real headSize, const Color& color);
    void cross(const Vec2& center, Real size, const Color& color);

    // Ground plane plus a metre grid, drawn to fill the visible area.
    void groundAndGrid(const Camera2D& camera, Real groundY, Real spacing);

private:
    struct Vertex {
        Real x, y;
        Real r, g, b, a;
    };

    void vertex(const Vec2& p, const Color& c, std::vector<Vertex>& into);
    void flush(std::vector<Vertex>& vertices, unsigned int primitive);

    unsigned int program_ = 0;
    unsigned int vao_ = 0;
    unsigned int vbo_ = 0;
    int viewProjectionLocation_ = -1;
    size_t bufferCapacity_ = 0;

    std::vector<Vertex> triangles_;
    std::vector<Vertex> lines_;
    Mat4 viewProjection_;
};

}  // namespace aibf
