// Vector / matrix / quaternion math for the physics engine and renderer.
//
// Conventions:
//   - Real is float; switch the typedef to double to test whether an instability
//     is precision-related. Nothing else in the codebase assumes 32-bit.
//   - Matrices are column-major (Vec3 c0,c1,c2), matching what OpenGL wants
//     uploaded, so Mat3/Mat4 can be handed to glUniformMatrix* without transpose.
//   - 2D cross products follow the Box2D convention documented at each overload.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace aibf {

using Real = float;

constexpr Real kPi = Real(3.14159265358979323846);
constexpr Real kTwoPi = kPi * Real(2);
constexpr Real kHalfPi = kPi * Real(0.5);
constexpr Real kDeg2Rad = kPi / Real(180);
constexpr Real kRad2Deg = Real(180) / kPi;
constexpr Real kEpsilon = Real(1e-6);

inline Real clamp(Real v, Real lo, Real hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline Real lerp(Real a, Real b, Real t) { return a + (b - a) * t; }
inline Real sign(Real v) { return v < Real(0) ? Real(-1) : Real(1); }
inline Real sqr(Real v) { return v * v; }
inline bool isFinite(Real v) { return std::isfinite(v); }

// Wraps an angle into (-pi, pi]. Used everywhere joint error is computed so a
// joint near +/-pi does not produce a ~2pi error spike and a torque explosion.
inline Real wrapAngle(Real a) {
    a = std::fmod(a + kPi, kTwoPi);
    if (a < Real(0)) a += kTwoPi;
    return a - kPi;
}

// ---------------------------------------------------------------- Vec2

struct Vec2 {
    Real x = 0, y = 0;

    Vec2() = default;
    Vec2(Real x_, Real y_) : x(x_), y(y_) {}

    Vec2 operator-() const { return {-x, -y}; }
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(Real s) const { return {x * s, y * s}; }
    Vec2 operator/(Real s) const { return {x / s, y / s}; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }
    Vec2& operator*=(Real s) { x *= s; y *= s; return *this; }
    bool operator==(const Vec2& o) const { return x == o.x && y == o.y; }
};

inline Vec2 operator*(Real s, const Vec2& v) { return {v.x * s, v.y * s}; }
inline Real dot(const Vec2& a, const Vec2& b) { return a.x * b.x + a.y * b.y; }
inline Real lengthSq(const Vec2& v) { return v.x * v.x + v.y * v.y; }
inline Real length(const Vec2& v) { return std::sqrt(lengthSq(v)); }

inline Vec2 normalize(const Vec2& v) {
    Real len = length(v);
    return len > kEpsilon ? v / len : Vec2(0, 0);
}

// Scalar "z" of the 3D cross product of two in-plane vectors.
inline Real cross(const Vec2& a, const Vec2& b) { return a.x * b.y - a.y * b.x; }
// omega x r, for a scalar angular velocity: gives the velocity of a point at r.
inline Vec2 cross(Real s, const Vec2& v) { return {-s * v.y, s * v.x}; }
// r x omega, the mirrored overload.
inline Vec2 cross(const Vec2& v, Real s) { return {s * v.y, -s * v.x}; }

// Rotates counter-clockwise by `angle` radians.
inline Vec2 rotate(const Vec2& v, Real angle) {
    Real c = std::cos(angle), s = std::sin(angle);
    return {c * v.x - s * v.y, s * v.x + c * v.y};
}
inline Vec2 perp(const Vec2& v) { return {-v.y, v.x}; }
inline bool isFinite(const Vec2& v) { return std::isfinite(v.x) && std::isfinite(v.y); }

// ---------------------------------------------------------------- Vec3

struct Vec3 {
    Real x = 0, y = 0, z = 0;

    Vec3() = default;
    Vec3(Real x_, Real y_, Real z_) : x(x_), y(y_), z(z_) {}
    explicit Vec3(Real s) : x(s), y(s), z(s) {}

    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(Real s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(Real s) const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(Real s) { x *= s; y *= s; z *= s; return *this; }
    Real operator[](int i) const { return (&x)[i]; }
    Real& operator[](int i) { return (&x)[i]; }
};

inline Vec3 operator*(Real s, const Vec3& v) { return {v.x * s, v.y * s, v.z * s}; }
inline Real dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Real lengthSq(const Vec3& v) { return dot(v, v); }
inline Real length(const Vec3& v) { return std::sqrt(lengthSq(v)); }

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline Vec3 normalize(const Vec3& v) {
    Real len = length(v);
    return len > kEpsilon ? v / len : Vec3(0, 0, 0);
}
inline bool isFinite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// ---------------------------------------------------------------- Vec4

struct Vec4 {
    Real x = 0, y = 0, z = 0, w = 0;

    Vec4() = default;
    Vec4(Real x_, Real y_, Real z_, Real w_) : x(x_), y(y_), z(z_), w(w_) {}
    Vec4(const Vec3& v, Real w_) : x(v.x), y(v.y), z(v.z), w(w_) {}

    Vec4 operator+(const Vec4& o) const { return {x + o.x, y + o.y, z + o.z, w + o.w}; }
    Vec4 operator*(Real s) const { return {x * s, y * s, z * s, w * s}; }
    Vec3 xyz() const { return {x, y, z}; }
};

// ---------------------------------------------------------------- Mat2

// 2x2, column-major. Exists for the 2-DOF block solve inside revolute joints.
struct Mat2 {
    Vec2 c0{1, 0}, c1{0, 1};

    Mat2() = default;
    Mat2(const Vec2& a, const Vec2& b) : c0(a), c1(b) {}

    Vec2 operator*(const Vec2& v) const { return c0 * v.x + c1 * v.y; }
};

// Returns the inverse, or the zero matrix when singular. Callers treat a zero
// result as "this constraint contributes nothing this iteration" rather than
// dividing by a near-zero determinant and launching the body into orbit.
//
// C4723 is suppressed because MSVC constant-folds callers that pass a singular
// matrix (a*b - a*b collapses to a literal 0) and warns about the division
// without accounting for the guard immediately above it. Anything reaching the
// division has |det| >= 1e-12, so the reciprocal is finite by construction.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4723)
#endif
inline Mat2 inverse(const Mat2& m) {
    Real a = m.c0.x, b = m.c1.x, c = m.c0.y, d = m.c1.y;
    Real det = a * d - b * c;
    if (std::abs(det) < Real(1e-12)) return {Vec2(0, 0), Vec2(0, 0)};
    Real inv = Real(1) / det;
    return {Vec2(d * inv, -c * inv), Vec2(-b * inv, a * inv)};
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// ---------------------------------------------------------------- Mat3

struct Mat3 {
    Vec3 c0{1, 0, 0}, c1{0, 1, 0}, c2{0, 0, 1};

    Mat3() = default;
    Mat3(const Vec3& a, const Vec3& b, const Vec3& c) : c0(a), c1(b), c2(c) {}

    static Mat3 identity() { return {}; }
    static Mat3 zero() { return {Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0)}; }
    static Mat3 diagonal(Real a, Real b, Real c) {
        return {Vec3(a, 0, 0), Vec3(0, b, 0), Vec3(0, 0, c)};
    }

    Vec3 operator*(const Vec3& v) const { return c0 * v.x + c1 * v.y + c2 * v.z; }
    Mat3 operator*(const Mat3& m) const { return {*this * m.c0, *this * m.c1, *this * m.c2}; }
    Mat3 operator*(Real s) const { return {c0 * s, c1 * s, c2 * s}; }
    Mat3 operator+(const Mat3& m) const { return {c0 + m.c0, c1 + m.c1, c2 + m.c2}; }
    Mat3 operator-(const Mat3& m) const { return {c0 - m.c0, c1 - m.c1, c2 - m.c2}; }
};

inline Mat3 transpose(const Mat3& m) {
    return {Vec3(m.c0.x, m.c1.x, m.c2.x),
            Vec3(m.c0.y, m.c1.y, m.c2.y),
            Vec3(m.c0.z, m.c1.z, m.c2.z)};
}

// Skew-symmetric matrix such that skew(a) * b == cross(a, b).
inline Mat3 skew(const Vec3& v) {
    return {Vec3(0, v.z, -v.y), Vec3(-v.z, 0, v.x), Vec3(v.y, -v.x, 0)};
}

inline Real determinant(const Mat3& m) { return dot(m.c0, cross(m.c1, m.c2)); }

inline Mat3 inverse(const Mat3& m) {
    Vec3 r0 = cross(m.c1, m.c2);
    Vec3 r1 = cross(m.c2, m.c0);
    Vec3 r2 = cross(m.c0, m.c1);
    Real det = dot(m.c0, r0);
    if (std::abs(det) < Real(1e-16)) return Mat3::zero();
    Real inv = Real(1) / det;
    // rows of the adjugate become columns after the implicit transpose
    return {Vec3(r0.x, r1.x, r2.x) * inv,
            Vec3(r0.y, r1.y, r2.y) * inv,
            Vec3(r0.z, r1.z, r2.z) * inv};
}

// ---------------------------------------------------------------- Quat

// Unit quaternion, w is the scalar part. Orientation is stored as a quaternion
// everywhere in the 3D engine; Euler angles never take part in integration.
struct Quat {
    Real x = 0, y = 0, z = 0, w = 1;

    Quat() = default;
    Quat(Real x_, Real y_, Real z_, Real w_) : x(x_), y(y_), z(z_), w(w_) {}

    static Quat identity() { return {}; }

    static Quat fromAxisAngle(const Vec3& axis, Real angle) {
        Vec3 n = normalize(axis);
        Real h = angle * Real(0.5);
        Real s = std::sin(h);
        return {n.x * s, n.y * s, n.z * s, std::cos(h)};
    }

    Vec3 axis() const { return {x, y, z}; }

    // Hamilton product: (*this * o) applies o first, then *this.
    Quat operator*(const Quat& o) const {
        return {w * o.x + x * o.w + y * o.z - z * o.y,
                w * o.y - x * o.z + y * o.w + z * o.x,
                w * o.z + x * o.y - y * o.x + z * o.w,
                w * o.w - x * o.x - y * o.y - z * o.z};
    }
    Quat operator*(Real s) const { return {x * s, y * s, z * s, w * s}; }
    Quat operator+(const Quat& o) const { return {x + o.x, y + o.y, z + o.z, w + o.w}; }
    Quat operator-() const { return {-x, -y, -z, -w}; }
};

inline Quat conjugate(const Quat& q) { return {-q.x, -q.y, -q.z, q.w}; }
inline Real dot(const Quat& a, const Quat& b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline Real length(const Quat& q) { return std::sqrt(dot(q, q)); }

inline Quat normalize(const Quat& q) {
    Real len = length(q);
    return len > kEpsilon ? q * (Real(1) / len) : Quat::identity();
}

inline Vec3 rotate(const Quat& q, const Vec3& v) {
    Vec3 u = q.axis();
    return v + Real(2) * cross(u, cross(u, v) + v * q.w);
}
inline Vec3 rotateInverse(const Quat& q, const Vec3& v) { return rotate(conjugate(q), v); }

inline Mat3 toMat3(const Quat& q) {
    Real xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    Real xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    Real wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    return {Vec3(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)),
            Vec3(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)),
            Vec3(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy))};
}

// First-order quaternion integration followed by renormalization. The drift this
// introduces is bounded by the renormalize, which is why orientation never needs
// to be re-orthogonalized the way a rotation matrix would.
inline Quat integrate(const Quat& q, const Vec3& omega, Real dt) {
    Quat dq = Quat(omega.x, omega.y, omega.z, Real(0)) * q;
    return normalize(q + dq * (Real(0.5) * dt));
}

inline Quat slerp(const Quat& a, const Quat& b, Real t) {
    Real cosTheta = dot(a, b);
    Quat end = b;
    if (cosTheta < Real(0)) { end = -b; cosTheta = -cosTheta; }
    if (cosTheta > Real(0.9995)) {
        return normalize(Quat(lerp(a.x, end.x, t), lerp(a.y, end.y, t),
                              lerp(a.z, end.z, t), lerp(a.w, end.w, t)));
    }
    Real theta = std::acos(clamp(cosTheta, Real(-1), Real(1)));
    Real sinTheta = std::sin(theta);
    Real wa = std::sin((Real(1) - t) * theta) / sinTheta;
    Real wb = std::sin(t * theta) / sinTheta;
    return normalize(a * wa + end * wb);
}

// Shortest-arc rotation taking `a` onto `b`; the residual used by 3D joint
// constraints and by the imitation pose reward.
inline Quat rotationBetween(const Quat& a, const Quat& b) { return b * conjugate(a); }

// Angle of the rotation, in [0, pi].
inline Real angleOf(const Quat& q) {
    Real w = clamp(std::abs(q.w), Real(-1), Real(1));
    return Real(2) * std::acos(w);
}

inline bool isFinite(const Quat& q) {
    return std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z) && std::isfinite(q.w);
}

// ---------------------------------------------------------------- Mat4

// Column-major, ready for glUniformMatrix4fv with transpose=GL_FALSE.
struct Mat4 {
    Vec4 c0{1, 0, 0, 0}, c1{0, 1, 0, 0}, c2{0, 0, 1, 0}, c3{0, 0, 0, 1};

    Mat4() = default;
    Mat4(const Vec4& a, const Vec4& b, const Vec4& c, const Vec4& d)
        : c0(a), c1(b), c2(c), c3(d) {}

    static Mat4 identity() { return {}; }

    Vec4 operator*(const Vec4& v) const { return c0 * v.x + c1 * v.y + c2 * v.z + c3 * v.w; }
    Mat4 operator*(const Mat4& m) const {
        return {*this * m.c0, *this * m.c1, *this * m.c2, *this * m.c3};
    }

    const Real* data() const { return &c0.x; }

    static Mat4 translation(const Vec3& t) {
        Mat4 m;
        m.c3 = Vec4(t.x, t.y, t.z, 1);
        return m;
    }
    static Mat4 scale(const Vec3& s) {
        Mat4 m;
        m.c0 = Vec4(s.x, 0, 0, 0);
        m.c1 = Vec4(0, s.y, 0, 0);
        m.c2 = Vec4(0, 0, s.z, 0);
        return m;
    }
    static Mat4 fromMat3(const Mat3& r) {
        return {Vec4(r.c0, 0), Vec4(r.c1, 0), Vec4(r.c2, 0), Vec4(0, 0, 0, 1)};
    }
    // Rigid transform: rotate by q, then translate by t.
    static Mat4 fromRotationTranslation(const Quat& q, const Vec3& t) {
        Mat4 m = fromMat3(toMat3(q));
        m.c3 = Vec4(t.x, t.y, t.z, 1);
        return m;
    }
};

inline Mat4 orthographic(Real l, Real r, Real b, Real t, Real n, Real f) {
    Mat4 m;
    m.c0 = Vec4(Real(2) / (r - l), 0, 0, 0);
    m.c1 = Vec4(0, Real(2) / (t - b), 0, 0);
    m.c2 = Vec4(0, 0, Real(-2) / (f - n), 0);
    m.c3 = Vec4(-(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n), 1);
    return m;
}

inline Mat4 perspective(Real fovYRadians, Real aspect, Real zNear, Real zFar) {
    Real tanHalf = std::tan(fovYRadians * Real(0.5));
    Mat4 m;
    m.c0 = Vec4(Real(1) / (aspect * tanHalf), 0, 0, 0);
    m.c1 = Vec4(0, Real(1) / tanHalf, 0, 0);
    m.c2 = Vec4(0, 0, -(zFar + zNear) / (zFar - zNear), -1);
    m.c3 = Vec4(0, 0, Real(-2) * zFar * zNear / (zFar - zNear), 0);
    return m;
}

inline Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 m;
    m.c0 = Vec4(s.x, u.x, -f.x, 0);
    m.c1 = Vec4(s.y, u.y, -f.y, 0);
    m.c2 = Vec4(s.z, u.z, -f.z, 0);
    m.c3 = Vec4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1);
    return m;
}

}  // namespace aibf
