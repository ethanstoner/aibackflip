// Capsule renderer for the 3D humanoid.
//
// Every shape in the world is a capsule, so there is exactly one mesh: a unit
// capsule with a half-length of 1 and a radius of 1, generated once and drawn
// with a per-instance model matrix. Non-uniform scaling would shear the caps, so
// the shader is given the radius and half-length separately and stretches only
// the cylindrical middle, which keeps a stubby capsule looking like a capsule
// rather than an ellipsoid.
//
// The ground is one quad with the grid drawn procedurally in the fragment
// shader, matching the 2D renderer's grid so the two look like the same world.
#pragma once

#include <cstdint>
#include <vector>

#include "core/Math.h"
#include "engine/GL.h"

namespace aibf {

struct Camera3D {
    Vec3 target{0, Real(0.9), 0};
    Real distance = Real(4.5);
    Real yaw = Real(0.6);     // radians, around +Y
    Real pitch = Real(0.15);  // radians, above the horizon
    Real fovY = Real(0.9);
    Real aspect = Real(16) / Real(9);
    Real nearPlane = Real(0.05);
    Real farPlane = Real(200);

    Vec3 eye() const {
        const Real cp = std::cos(pitch);
        return target + Vec3(std::sin(yaw) * cp, std::sin(pitch), std::cos(yaw) * cp) * distance;
    }
    Mat4 view() const { return lookAt(eye(), target, Vec3(0, 1, 0)); }
    Mat4 projection() const { return perspective(fovY, aspect, nearPlane, farPlane); }
};

class Renderer3D {
public:
    bool initialize();
    void shutdown();

    void beginFrame(const Camera3D& camera, const Vec4& clearColour);
    // `halfLength` is along the capsule's local +Y, matching RigidBody3D.
    void capsule(const Vec3& position, const Quat& orientation, Real radius, Real halfLength,
                 const Vec4& colour);
    void ground(Real extent, const Vec4& colour, const Vec4& gridColour);
    void endFrame();

    const std::string& lastError() const { return lastError_; }

private:
    void buildCapsuleMesh();

    uint32_t capsuleVao_ = 0, capsuleVbo_ = 0, capsuleIbo_ = 0;
    int capsuleIndexCount_ = 0;
    uint32_t groundVao_ = 0, groundVbo_ = 0;
    uint32_t capsuleProgram_ = 0, groundProgram_ = 0;

    struct CapsuleUniforms {
        int viewProjection = -1;
        int model = -1;
        int normalMatrix = -1;
        int colour = -1;
        int radius = -1;
        int halfLength = -1;
        int lightDirection = -1;
    } capsuleUniforms_;

    struct GroundUniforms {
        int viewProjection = -1;
        int colour = -1;
        int gridColour = -1;
        int extent = -1;
        int eye = -1;
    } groundUniforms_;

    Mat4 viewProjection_ = Mat4::identity();
    Vec3 eye_;
    std::string lastError_;
};

}  // namespace aibf
