#include "engine/Renderer2D.h"

#include <cmath>
#include <cstdio>

#include "engine/GL.h"

namespace aibf {

namespace {

constexpr int kCapSegments = 14;  // per hemispherical cap

const char* kVertexShader = R"(#version 330 core
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec4 aColor;
uniform mat4 uViewProjection;
out vec4 vColor;
void main() {
    vColor = aColor;
    gl_Position = uViewProjection * vec4(aPosition, 0.0, 1.0);
}
)";

const char* kFragmentShader = R"(#version 330 core
in vec4 vColor;
out vec4 fragColor;
void main() { fragColor = vColor; }
)";

unsigned int compile(unsigned int stage, const char* source, const char* label) {
    const unsigned int shader = gl::CreateShader(stage);
    gl::ShaderSource(shader, 1, &source, nullptr);
    gl::CompileShader(shader);
    GLint ok = 0;
    gl::GetShaderiv(shader, gl::kCompileStatus, &ok);
    if (!ok) {
        char log[1024] = {};
        gl::GetShaderInfoLog(shader, sizeof(log) - 1, nullptr, log);
        std::fprintf(stderr, "%s shader failed to compile:\n%s\n", label, log);
        gl::DeleteShader(shader);
        return 0;
    }
    return shader;
}

// Outline of a capsule whose axis runs along local +Y, in counter-clockwise
// order starting at the right of the top cap.
void capsuleOutlinePoints(const Vec2& center, Real angle, Real radius, Real halfLength,
                          Vec2* out) {
    int n = 0;
    for (int i = 0; i <= kCapSegments; ++i) {
        const Real t = kPi * Real(i) / Real(kCapSegments);
        out[n++] = center + rotate(Vec2(radius * std::cos(t), halfLength + radius * std::sin(t)),
                                   angle);
    }
    for (int i = 0; i <= kCapSegments; ++i) {
        const Real t = kPi + kPi * Real(i) / Real(kCapSegments);
        out[n++] = center + rotate(Vec2(radius * std::cos(t), -halfLength + radius * std::sin(t)),
                                   angle);
    }
}

constexpr int kOutlinePoints = 2 * (kCapSegments + 1);

}  // namespace

// ---------------------------------------------------------------- camera

Mat4 Camera2D::viewProjection() const {
    const Real hw = halfWidth();
    return orthographic(center.x - hw, center.x + hw, center.y - halfHeight,
                        center.y + halfHeight, Real(-1), Real(1));
}

Vec2 Camera2D::screenToWorld(const Vec2& pixel, int screenWidth, int screenHeight) const {
    if (screenWidth <= 0 || screenHeight <= 0) return center;
    const Real ndcX = Real(2) * pixel.x / Real(screenWidth) - Real(1);
    const Real ndcY = Real(1) - Real(2) * pixel.y / Real(screenHeight);
    return center + Vec2(ndcX * halfWidth(), ndcY * halfHeight);
}

Vec2 Camera2D::worldToScreen(const Vec2& world, int screenWidth, int screenHeight) const {
    const Vec2 offset = world - center;
    const Real ndcX = offset.x / halfWidth();
    const Real ndcY = offset.y / halfHeight;
    return Vec2((ndcX + Real(1)) * Real(0.5) * Real(screenWidth),
                (Real(1) - ndcY) * Real(0.5) * Real(screenHeight));
}

Real Camera2D::pixelsToWorld(Real pixels, int screenHeight) const {
    if (screenHeight <= 0) return 0;
    return pixels * (Real(2) * halfHeight) / Real(screenHeight);
}

// ---------------------------------------------------------------- lifecycle

Renderer2D::~Renderer2D() { shutdown(); }

bool Renderer2D::initialize() {
    const unsigned int vertexShader = compile(gl::kVertexShader, kVertexShader, "vertex");
    const unsigned int fragmentShader = compile(gl::kFragmentShader, kFragmentShader, "fragment");
    if (!vertexShader || !fragmentShader) return false;

    program_ = gl::CreateProgram();
    gl::AttachShader(program_, vertexShader);
    gl::AttachShader(program_, fragmentShader);
    gl::LinkProgram(program_);

    GLint ok = 0;
    gl::GetProgramiv(program_, gl::kLinkStatus, &ok);
    if (!ok) {
        char log[1024] = {};
        gl::GetProgramInfoLog(program_, sizeof(log) - 1, nullptr, log);
        std::fprintf(stderr, "shader program failed to link:\n%s\n", log);
        return false;
    }
    gl::DeleteShader(vertexShader);
    gl::DeleteShader(fragmentShader);

    viewProjectionLocation_ = gl::GetUniformLocation(program_, "uViewProjection");

    gl::GenVertexArrays(1, &vao_);
    gl::BindVertexArray(vao_);
    gl::GenBuffers(1, &vbo_);
    gl::BindBuffer(gl::kArrayBuffer, vbo_);
    gl::EnableVertexAttribArray(0);
    gl::VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                            reinterpret_cast<const void*>(offsetof(Vertex, x)));
    gl::EnableVertexAttribArray(1);
    gl::VertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                            reinterpret_cast<const void*>(offsetof(Vertex, r)));
    gl::BindVertexArray(0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return true;
}

void Renderer2D::shutdown() {
    if (vbo_) {
        gl::DeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    if (vao_) {
        gl::DeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    if (program_) {
        gl::DeleteProgram(program_);
        program_ = 0;
    }
}

void Renderer2D::beginFrame(const Camera2D& camera, const Color& clear) {
    viewProjection_ = camera.viewProjection();
    triangles_.clear();
    lines_.clear();
    glClearColor(clear.r, clear.g, clear.b, clear.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer2D::endFrame() {
    gl::UseProgram(program_);
    gl::UniformMatrix4fv(viewProjectionLocation_, 1, GL_FALSE, viewProjection_.data());
    gl::BindVertexArray(vao_);
    // Triangles first so debug lines and markers draw over the bodies.
    flush(triangles_, GL_TRIANGLES);
    flush(lines_, GL_LINES);
    gl::BindVertexArray(0);
    gl::UseProgram(0);
}

void Renderer2D::flush(std::vector<Vertex>& vertices, unsigned int primitive) {
    if (vertices.empty()) return;
    gl::BindBuffer(gl::kArrayBuffer, vbo_);
    const size_t bytes = vertices.size() * sizeof(Vertex);
    // Orphan and regrow rather than calling BufferData every frame at the same
    // size, which would make the driver reallocate for no reason.
    if (bytes > bufferCapacity_) {
        bufferCapacity_ = bytes * 2;
        gl::BufferData(gl::kArrayBuffer, static_cast<gl::GLsizeiptr>(bufferCapacity_), nullptr,
                       gl::kStreamDraw);
    }
    gl::BufferSubData(gl::kArrayBuffer, 0, static_cast<gl::GLsizeiptr>(bytes), vertices.data());
    glDrawArrays(primitive, 0, static_cast<GLsizei>(vertices.size()));
    vertices.clear();
}

void Renderer2D::vertex(const Vec2& p, const Color& c, std::vector<Vertex>& into) {
    into.push_back(Vertex{p.x, p.y, c.r, c.g, c.b, c.a});
}

// ---------------------------------------------------------------- shapes

void Renderer2D::triangle(const Vec2& a, const Vec2& b, const Vec2& c, const Color& color) {
    vertex(a, color, triangles_);
    vertex(b, color, triangles_);
    vertex(c, color, triangles_);
}

void Renderer2D::capsule(const Vec2& center, Real angle, Real radius, Real halfLength,
                         const Color& color) {
    Vec2 points[kOutlinePoints];
    capsuleOutlinePoints(center, angle, radius, halfLength, points);
    for (int i = 1; i + 1 < kOutlinePoints; ++i) {
        triangle(points[0], points[i], points[i + 1], color);
    }
}

void Renderer2D::capsuleOutline(const Vec2& center, Real angle, Real radius, Real halfLength,
                                const Color& color) {
    Vec2 points[kOutlinePoints];
    capsuleOutlinePoints(center, angle, radius, halfLength, points);
    for (int i = 0; i < kOutlinePoints; ++i) {
        line(points[i], points[(i + 1) % kOutlinePoints], color);
    }
}

void Renderer2D::circle(const Vec2& center, Real radius, const Color& color) {
    capsule(center, 0, radius, 0, color);
}

void Renderer2D::circleOutline(const Vec2& center, Real radius, const Color& color) {
    capsuleOutline(center, 0, radius, 0, color);
}

void Renderer2D::line(const Vec2& a, const Vec2& b, const Color& color) {
    vertex(a, color, lines_);
    vertex(b, color, lines_);
}

void Renderer2D::thickLine(const Vec2& a, const Vec2& b, Real width, const Color& color) {
    const Vec2 direction = b - a;
    if (lengthSq(direction) < Real(1e-12)) return;
    const Vec2 offset = normalize(perp(direction)) * (width * Real(0.5));
    triangle(a - offset, a + offset, b + offset, color);
    triangle(a - offset, b + offset, b - offset, color);
}

void Renderer2D::quad(const Vec2& min, const Vec2& max, const Color& color) {
    triangle(min, Vec2(max.x, min.y), max, color);
    triangle(min, max, Vec2(min.x, max.y), color);
}

void Renderer2D::arrow(const Vec2& from, const Vec2& to, Real headSize, const Color& color) {
    line(from, to, color);
    const Vec2 direction = to - from;
    if (lengthSq(direction) < Real(1e-12)) return;
    const Vec2 unit = normalize(direction);
    const Vec2 side = perp(unit) * (headSize * Real(0.5));
    const Vec2 base = to - unit * headSize;
    triangle(to, base + side, base - side, color);
}

void Renderer2D::cross(const Vec2& center, Real size, const Color& color) {
    line(center - Vec2(size, 0), center + Vec2(size, 0), color);
    line(center - Vec2(0, size), center + Vec2(0, size), color);
}

void Renderer2D::groundAndGrid(const Camera2D& camera, Real groundY, Real spacing) {
    const Real hw = camera.halfWidth();
    const Real left = camera.center.x - hw;
    const Real right = camera.center.x + hw;
    const Real top = camera.center.y + camera.halfHeight;
    const Real bottom = camera.center.y - camera.halfHeight;

    if (spacing > Real(0)) {
        const Real firstX = std::floor(left / spacing) * spacing;
        for (Real x = firstX; x <= right; x += spacing) {
            line(Vec2(x, bottom), Vec2(x, top), palette::grid);
        }
        const Real firstY = std::floor(bottom / spacing) * spacing;
        for (Real y = firstY; y <= top; y += spacing) {
            if (y < groundY) continue;  // grid only above the floor
            line(Vec2(left, y), Vec2(right, y), palette::grid);
        }
    }

    quad(Vec2(left, bottom), Vec2(right, groundY), palette::ground);
    line(Vec2(left, groundY), Vec2(right, groundY), palette::accent);
}

}  // namespace aibf
