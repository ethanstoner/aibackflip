#include "engine/Renderer3D.h"

#include <cmath>
#include <cstdio>

namespace aibf {

namespace {

constexpr int kRings = 10;    // per hemispherical cap
constexpr int kSegments = 20; // around the axis

const char* kCapsuleVertex = R"(#version 330 core
layout(location = 0) in vec3 aPosition;   // unit capsule: caps at +/-1 along y
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aCapSign;   // -1 lower cap, 0 middle, +1 upper cap

uniform mat4 uViewProjection;
uniform mat4 uModel;
uniform mat3 uNormalMatrix;
uniform float uRadius;
uniform float uHalfLength;

out vec3 vNormal;
out vec3 vWorld;

void main() {
    // Only the cylindrical middle is stretched. Scaling the whole mesh by
    // (r, h, r) would squash the hemispherical caps into ellipsoids, which is
    // what makes a naively drawn capsule look wrong at short half-lengths.
    vec3 local = aPosition * uRadius;
    local.y += aCapSign * uHalfLength;

    vec4 world = uModel * vec4(local, 1.0);
    vWorld = world.xyz;
    vNormal = normalize(uNormalMatrix * aNormal);
    gl_Position = uViewProjection * world;
}
)";

const char* kCapsuleFragment = R"(#version 330 core
in vec3 vNormal;
in vec3 vWorld;
uniform vec4 uColour;
uniform vec3 uLightDirection;
out vec4 fragColor;

void main() {
    vec3 n = normalize(vNormal);
    float key = max(dot(n, -normalize(uLightDirection)), 0.0);
    // A dim second light from below keeps limbs readable when the figure is
    // upside down, which for this project is most of the interesting footage.
    float fill = max(dot(n, vec3(0.0, 1.0, 0.0)), 0.0) * 0.25;
    float ambient = 0.35;
    vec3 lit = uColour.rgb * (ambient + key * 0.75 + fill);
    fragColor = vec4(lit, uColour.a);
}
)";

const char* kGroundVertex = R"(#version 330 core
layout(location = 0) in vec2 aPlane;
uniform mat4 uViewProjection;
uniform float uExtent;
out vec3 vWorld;
void main() {
    vWorld = vec3(aPlane.x * uExtent, 0.0, aPlane.y * uExtent);
    gl_Position = uViewProjection * vec4(vWorld, 1.0);
}
)";

const char* kGroundFragment = R"(#version 330 core
in vec3 vWorld;
uniform vec4 uColour;
uniform vec4 uGridColour;
uniform vec3 uEye;
out vec4 fragColor;

void main() {
    // Procedural half-metre grid, matching the 2D renderer's pitch so the two
    // views read as the same world. Drawn with derivatives so the lines stay
    // one pixel wide at any distance instead of aliasing into noise.
    vec2 coord = vWorld.xz * 2.0;
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
    float line = min(grid.x, grid.y);
    float strength = 1.0 - min(line, 1.0);

    // Fade the whole plane out with distance rather than ending in a hard edge.
    float fade = 1.0 - clamp(length(vWorld - uEye) / 40.0, 0.0, 1.0);
    vec3 colour = mix(uColour.rgb, uGridColour.rgb, strength * 0.6);
    fragColor = vec4(colour, uColour.a * fade);
}
)";

unsigned int compile(unsigned int stage, const char* source, const char* label,
                     std::string& error) {
    const unsigned int shader = gl::CreateShader(stage);
    gl::ShaderSource(shader, 1, &source, nullptr);
    gl::CompileShader(shader);
    GLint ok = 0;
    gl::GetShaderiv(shader, gl::kCompileStatus, &ok);
    if (!ok) {
        char log[1024] = {};
        gl::GetShaderInfoLog(shader, sizeof(log) - 1, nullptr, log);
        error = std::string(label) + " shader: " + log;
        gl::DeleteShader(shader);
        return 0;
    }
    return shader;
}

unsigned int link(const char* vertexSource, const char* fragmentSource, const char* label,
                  std::string& error) {
    const unsigned int vs = compile(gl::kVertexShader, vertexSource, label, error);
    if (!vs) return 0;
    const unsigned int fs = compile(gl::kFragmentShader, fragmentSource, label, error);
    if (!fs) {
        gl::DeleteShader(vs);
        return 0;
    }
    const unsigned int program = gl::CreateProgram();
    gl::AttachShader(program, vs);
    gl::AttachShader(program, fs);
    gl::LinkProgram(program);
    gl::DeleteShader(vs);
    gl::DeleteShader(fs);

    GLint ok = 0;
    gl::GetProgramiv(program, gl::kLinkStatus, &ok);
    if (!ok) {
        char log[1024] = {};
        gl::GetProgramInfoLog(program, sizeof(log) - 1, nullptr, log);
        error = std::string(label) + " program: " + log;
        gl::DeleteProgram(program);
        return 0;
    }
    return program;
}

struct CapsuleVertex {
    float px, py, pz;
    float nx, ny, nz;
    float capSign;
};

}  // namespace

void Renderer3D::buildCapsuleMesh() {
    std::vector<CapsuleVertex> vertices;
    std::vector<uint32_t> indices;

    // Rings from the bottom pole to the top pole. The two hemispheres carry a
    // cap sign of -1 and +1; the seam between them is duplicated so the middle
    // can be stretched without dragging the caps with it.
    const int totalRings = 2 * kRings + 2;
    for (int ring = 0; ring < totalRings; ++ring) {
        const bool upper = ring >= kRings + 1;
        const int localRing = upper ? ring - (kRings + 1) : ring;
        // Latitude from -90 degrees at the bottom pole to 0 at the seam.
        const Real t = Real(localRing) / Real(kRings);
        const Real latitude = upper ? (t * kHalfPi) : (t * kHalfPi - kHalfPi);
        const Real y = std::sin(latitude);
        const Real r = std::cos(latitude);
        const Real capSign = upper ? Real(1) : Real(-1);

        for (int segment = 0; segment <= kSegments; ++segment) {
            const Real angle = Real(segment) / Real(kSegments) * Real(2) * kPi;
            const Real x = std::cos(angle) * r;
            const Real z = std::sin(angle) * r;
            CapsuleVertex v;
            v.px = float(x);
            v.py = float(y);
            v.pz = float(z);
            // The normal of a capsule surface is the normal of the sphere the
            // point came from, which stays correct after the middle is
            // stretched precisely because the stretch is along y only.
            v.nx = float(x);
            v.ny = float(y);
            v.nz = float(z);
            v.capSign = float(capSign);
            vertices.push_back(v);
        }
    }

    const int stride = kSegments + 1;
    for (int ring = 0; ring + 1 < totalRings; ++ring) {
        for (int segment = 0; segment < kSegments; ++segment) {
            const uint32_t a = uint32_t(ring * stride + segment);
            const uint32_t b = uint32_t(a + 1);
            const uint32_t c = uint32_t((ring + 1) * stride + segment);
            const uint32_t d = uint32_t(c + 1);
            indices.push_back(a);
            indices.push_back(c);
            indices.push_back(b);
            indices.push_back(b);
            indices.push_back(c);
            indices.push_back(d);
        }
    }
    capsuleIndexCount_ = int(indices.size());

    gl::GenVertexArrays(1, &capsuleVao_);
    gl::BindVertexArray(capsuleVao_);

    gl::GenBuffers(1, &capsuleVbo_);
    gl::BindBuffer(gl::kArrayBuffer, capsuleVbo_);
    gl::BufferData(gl::kArrayBuffer, gl::GLsizeiptr(vertices.size() * sizeof(CapsuleVertex)),
                   vertices.data(), gl::kStaticDraw);

    gl::GenBuffers(1, &capsuleIbo_);
    gl::BindBuffer(gl::kElementArrayBuffer, capsuleIbo_);
    gl::BufferData(gl::kElementArrayBuffer, gl::GLsizeiptr(indices.size() * sizeof(uint32_t)),
                   indices.data(), gl::kStaticDraw);

    gl::EnableVertexAttribArray(0);
    gl::VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CapsuleVertex), (const void*)0);
    gl::EnableVertexAttribArray(1);
    gl::VertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CapsuleVertex),
                            (const void*)(3 * sizeof(float)));
    gl::EnableVertexAttribArray(2);
    gl::VertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(CapsuleVertex),
                            (const void*)(6 * sizeof(float)));
    gl::BindVertexArray(0);
}

bool Renderer3D::initialize() {
    capsuleProgram_ = link(kCapsuleVertex, kCapsuleFragment, "capsule", lastError_);
    if (!capsuleProgram_) return false;
    groundProgram_ = link(kGroundVertex, kGroundFragment, "ground", lastError_);
    if (!groundProgram_) return false;

    capsuleUniforms_.viewProjection = gl::GetUniformLocation(capsuleProgram_, "uViewProjection");
    capsuleUniforms_.model = gl::GetUniformLocation(capsuleProgram_, "uModel");
    capsuleUniforms_.normalMatrix = gl::GetUniformLocation(capsuleProgram_, "uNormalMatrix");
    capsuleUniforms_.colour = gl::GetUniformLocation(capsuleProgram_, "uColour");
    capsuleUniforms_.radius = gl::GetUniformLocation(capsuleProgram_, "uRadius");
    capsuleUniforms_.halfLength = gl::GetUniformLocation(capsuleProgram_, "uHalfLength");
    capsuleUniforms_.lightDirection = gl::GetUniformLocation(capsuleProgram_, "uLightDirection");

    groundUniforms_.viewProjection = gl::GetUniformLocation(groundProgram_, "uViewProjection");
    groundUniforms_.colour = gl::GetUniformLocation(groundProgram_, "uColour");
    groundUniforms_.gridColour = gl::GetUniformLocation(groundProgram_, "uGridColour");
    groundUniforms_.extent = gl::GetUniformLocation(groundProgram_, "uExtent");
    groundUniforms_.eye = gl::GetUniformLocation(groundProgram_, "uEye");

    buildCapsuleMesh();

    const float quad[] = {-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1};
    gl::GenVertexArrays(1, &groundVao_);
    gl::BindVertexArray(groundVao_);
    gl::GenBuffers(1, &groundVbo_);
    gl::BindBuffer(gl::kArrayBuffer, groundVbo_);
    gl::BufferData(gl::kArrayBuffer, sizeof(quad), quad, gl::kStaticDraw);
    gl::EnableVertexAttribArray(0);
    gl::VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (const void*)0);
    gl::BindVertexArray(0);

    return true;
}

void Renderer3D::shutdown() {
    if (capsuleVbo_) gl::DeleteBuffers(1, &capsuleVbo_);
    if (capsuleIbo_) gl::DeleteBuffers(1, &capsuleIbo_);
    if (groundVbo_) gl::DeleteBuffers(1, &groundVbo_);
    if (capsuleVao_) gl::DeleteVertexArrays(1, &capsuleVao_);
    if (groundVao_) gl::DeleteVertexArrays(1, &groundVao_);
    if (capsuleProgram_) gl::DeleteProgram(capsuleProgram_);
    if (groundProgram_) gl::DeleteProgram(groundProgram_);
    capsuleVbo_ = capsuleIbo_ = groundVbo_ = capsuleVao_ = groundVao_ = 0;
    capsuleProgram_ = groundProgram_ = 0;
}

void Renderer3D::beginFrame(const Camera3D& camera, const Vec4& clearColour) {
    viewProjection_ = camera.projection() * camera.view();
    eye_ = camera.eye();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(clearColour.x, clearColour.y, clearColour.z, clearColour.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer3D::capsule(const Vec3& position, const Quat& orientation, Real radius,
                         Real halfLength, const Vec4& colour) {
    const Mat4 model = Mat4::fromRotationTranslation(orientation, position);
    const Mat3 normalMatrix = toMat3(orientation);  // rotation only, so no inverse transpose

    gl::UseProgram(capsuleProgram_);
    gl::UniformMatrix4fv(capsuleUniforms_.viewProjection, 1, GL_FALSE, &viewProjection_.c0.x);
    gl::UniformMatrix4fv(capsuleUniforms_.model, 1, GL_FALSE, &model.c0.x);
    // Uploaded as a mat3 rather than reusing the 4x4 path: GL reads a mat3
    // uniform as nine tightly packed floats, and a 4x4 upload would feed the
    // shader the wrong columns without any error.
    const float normal3[9] = {float(normalMatrix.c0.x), float(normalMatrix.c0.y),
                              float(normalMatrix.c0.z), float(normalMatrix.c1.x),
                              float(normalMatrix.c1.y), float(normalMatrix.c1.z),
                              float(normalMatrix.c2.x), float(normalMatrix.c2.y),
                              float(normalMatrix.c2.z)};
    gl::UniformMatrix3fv(capsuleUniforms_.normalMatrix, 1, GL_FALSE, normal3);
    gl::Uniform4f(capsuleUniforms_.colour, colour.x, colour.y, colour.z, colour.w);
    gl::Uniform1f(capsuleUniforms_.radius, float(radius));
    gl::Uniform1f(capsuleUniforms_.halfLength, float(halfLength));
    gl::Uniform3f(capsuleUniforms_.lightDirection, -0.4f, -0.85f, -0.35f);

    gl::BindVertexArray(capsuleVao_);
    glDrawElements(GL_TRIANGLES, capsuleIndexCount_, GL_UNSIGNED_INT, nullptr);
    gl::BindVertexArray(0);
}

void Renderer3D::ground(Real extent, const Vec4& colour, const Vec4& gridColour) {
    gl::UseProgram(groundProgram_);
    gl::UniformMatrix4fv(groundUniforms_.viewProjection, 1, GL_FALSE, &viewProjection_.c0.x);
    gl::Uniform4f(groundUniforms_.colour, colour.x, colour.y, colour.z, colour.w);
    gl::Uniform4f(groundUniforms_.gridColour, gridColour.x, gridColour.y, gridColour.z,
                  gridColour.w);
    gl::Uniform1f(groundUniforms_.extent, float(extent));
    gl::Uniform3f(groundUniforms_.eye, float(eye_.x), float(eye_.y), float(eye_.z));

    // The ground is two-sided: seen from below during a flip, a back-face-culled
    // plane simply vanishes and the figure appears to float in the void.
    glDisable(GL_CULL_FACE);
    gl::BindVertexArray(groundVao_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    gl::BindVertexArray(0);
    glEnable(GL_CULL_FACE);
}

void Renderer3D::endFrame() {
    gl::UseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
}

}  // namespace aibf
