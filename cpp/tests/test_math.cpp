#include "core/Math.h"
#include "core/Rng.h"
#include "core/Test.h"

using namespace aibf;

// ------------------------------------------------------------------ Vec2

TEST(Vec2, dotAndLength) {
    Vec2 a(3, 4);
    CHECK_NEAR(length(a), 5.0, 1e-6);
    CHECK_NEAR(lengthSq(a), 25.0, 1e-6);
    CHECK_NEAR(dot(a, Vec2(1, 0)), 3.0, 1e-6);
    CHECK_NEAR(length(normalize(a)), 1.0, 1e-6);
    // normalize of a degenerate vector must not produce NaN
    CHECK(isFinite(normalize(Vec2(0, 0))));
}

TEST(Vec2, crossConventionsAgree) {
    // The two scalar/vector overloads must be exact negatives of one another,
    // or angular impulse and point velocity will disagree in sign and the
    // solver will pump energy instead of removing it.
    Vec2 r(0.3f, -0.7f);
    Real w = 2.5f;
    Vec2 a = cross(w, r);
    Vec2 b = cross(r, w);
    CHECK_VEC2_NEAR(a + b, 0.0, 0.0, 1e-6);
    CHECK_NEAR(cross(Vec2(1, 0), Vec2(0, 1)), 1.0, 1e-6);
}

TEST(Vec2, pointVelocityMatchesFiniteDifference) {
    // v_point = v + omega x r, checked against the derivative of the actual
    // rotated offset rather than against the same formula written twice.
    Vec2 r(0.4f, 0.1f);
    Real omega = 1.7f;
    Real dt = 1e-4f;
    Vec2 analytic = cross(omega, r);
    Vec2 numeric = (rotate(r, omega * dt) - r) / dt;
    CHECK_VEC2_NEAR(analytic - numeric, 0.0, 0.0, 1e-3);
}

TEST(Vec2, rotateIsOrthonormal) {
    Vec2 v(1.3f, -2.4f);
    Vec2 r = rotate(v, 0.9f);
    CHECK_NEAR(length(r), length(v), 1e-5);
    CHECK_NEAR(length(rotate(r, -0.9f) - v), 0.0, 1e-5);
    CHECK_NEAR(dot(v, perp(v)), 0.0, 1e-6);
}

TEST(Angle, wrapKeepsErrorsSmallAcrossThePiBoundary) {
    CHECK_NEAR(wrapAngle(0.0f), 0.0, 1e-6);
    CHECK_NEAR(wrapAngle(kTwoPi + 0.25f), 0.25, 1e-5);
    CHECK_NEAR(wrapAngle(-kTwoPi - 0.25f), -0.25, 1e-5);
    // a joint at +179 deg targeting -179 deg is 2 degrees away, not 358
    Real err = wrapAngle(Real(-179) * kDeg2Rad - Real(179) * kDeg2Rad);
    CHECK_NEAR(std::abs(err), Real(2) * kDeg2Rad, 1e-5);
}

// ------------------------------------------------------------------ Mat2 / Mat3

TEST(Mat2, inverseRoundTrips) {
    Mat2 m(Vec2(2, 1), Vec2(-1, 3));
    Mat2 inv = inverse(m);
    Vec2 v(0.7f, -1.2f);
    CHECK_NEAR(length(inv * (m * v) - v), 0.0, 1e-5);
}

TEST(Mat2, nearSingularInverseDegradesToZeroInsteadOfInfinity) {
    // Columns grow progressively more parallel. The guard has to engage before
    // 1/det overflows, because a revolute joint whose two anchored bodies line
    // up produces exactly this matrix.
    for (int i = 0; i < 20; ++i) {
        Real eps = std::pow(Real(10), Real(-i));
        Mat2 m(Vec2(1, 2), Vec2(2 + eps, 4));
        Vec2 r = inverse(m) * Vec2(1, 1);
        CHECK(isFinite(r));
    }
    // Exactly singular: identical columns, so the determinant cancels bit-for-bit.
    Rng rng(3);
    Real a = rng.uniform(1.0f, 2.0f), b = rng.uniform(1.0f, 2.0f);
    Mat2 duplicated(Vec2(a, b), Vec2(a, b));
    CHECK_NEAR(length(inverse(duplicated) * Vec2(1, 1)), 0.0, 1e-9);
}

TEST(Mat3, inverseRoundTrips) {
    Mat3 m(Vec3(2, 0.3f, -0.1f), Vec3(0.1f, 1.5f, 0.4f), Vec3(-0.2f, 0.7f, 3.0f));
    Mat3 inv = inverse(m);
    Vec3 v(0.9f, -1.4f, 2.2f);
    CHECK_NEAR(length(inv * (m * v) - v), 0.0, 1e-4);
}

TEST(Mat3, skewReproducesCrossProduct) {
    Vec3 a(0.3f, -1.1f, 2.0f), b(-0.7f, 0.5f, 1.3f);
    CHECK_NEAR(length(skew(a) * b - cross(a, b)), 0.0, 1e-5);
}

TEST(Mat3, transposeOfRotationIsItsInverse) {
    Quat q = Quat::fromAxisAngle(Vec3(0.3f, 1.0f, -0.4f), 0.8f);
    Mat3 r = toMat3(q);
    Mat3 shouldBeIdentity = transpose(r) * r;
    CHECK_NEAR(length(shouldBeIdentity.c0 - Vec3(1, 0, 0)), 0.0, 1e-5);
    CHECK_NEAR(length(shouldBeIdentity.c1 - Vec3(0, 1, 0)), 0.0, 1e-5);
    CHECK_NEAR(length(shouldBeIdentity.c2 - Vec3(0, 0, 1)), 0.0, 1e-5);
    CHECK_NEAR(determinant(r), 1.0, 1e-5);
}

// ------------------------------------------------------------------ Quat

TEST(Quat, rotateAgreesWithMatrixForm) {
    Quat q = Quat::fromAxisAngle(Vec3(0.2f, -0.8f, 0.5f), 1.1f);
    Vec3 v(1.3f, 0.4f, -2.1f);
    CHECK_NEAR(length(rotate(q, v) - toMat3(q) * v), 0.0, 1e-4);
}

TEST(Quat, rotationPreservesLengthAndInverts) {
    Quat q = Quat::fromAxisAngle(Vec3(1, 1, 0), 2.0f);
    Vec3 v(0.5f, -1.5f, 3.0f);
    Vec3 r = rotate(q, v);
    CHECK_NEAR(length(r), length(v), 1e-4);
    CHECK_NEAR(length(rotateInverse(q, r) - v), 0.0, 1e-4);
}

TEST(Quat, productAppliesRightHandSideFirst) {
    Quat a = Quat::fromAxisAngle(Vec3(0, 0, 1), kHalfPi);
    Quat b = Quat::fromAxisAngle(Vec3(1, 0, 0), kHalfPi);
    Vec3 v(0, 0, 1);
    CHECK_NEAR(length(rotate(a * b, v) - rotate(a, rotate(b, v))), 0.0, 1e-5);
}

TEST(Quat, axisAngleMatchesKnownRotation) {
    // 90 degrees about +z takes +x onto +y
    Quat q = Quat::fromAxisAngle(Vec3(0, 0, 1), kHalfPi);
    CHECK_VEC3_NEAR(rotate(q, Vec3(1, 0, 0)), 0.0, 1.0, 0.0, 1e-5);
    CHECK_NEAR(angleOf(q), kHalfPi, 1e-5);
}

TEST(Quat, integrationTracksConstantAngularVelocity) {
    // Spin at a constant rate for one second in 1/240 s steps and check the
    // accumulated angle. This is the actual integrator the 3D engine uses, so
    // the tolerance here is the drift budget per simulated second.
    const Real omega = 2.0f;
    const Real dt = 1.0f / 240.0f;
    Quat q = Quat::identity();
    for (int i = 0; i < 240; ++i) q = integrate(q, Vec3(0, 0, omega), dt);
    CHECK_NEAR(angleOf(q), omega, 1e-3);
    CHECK_NEAR(length(q), 1.0, 1e-5);
    CHECK_VEC3_NEAR(rotate(q, Vec3(1, 0, 0)), std::cos(omega), std::sin(omega), 0.0, 2e-3);
}

TEST(Quat, integrationStaysNormalizedUnderLongRuns) {
    Quat q = Quat::identity();
    Rng rng(7);
    for (int i = 0; i < 20000; ++i) {
        Vec3 omega(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5));
        q = integrate(q, omega, 1.0f / 240.0f);
    }
    CHECK(isFinite(q));
    CHECK_NEAR(length(q), 1.0, 1e-4);
}

TEST(Quat, slerpHitsItsEndpointsAndMidpoint) {
    Quat a = Quat::identity();
    Quat b = Quat::fromAxisAngle(Vec3(0, 1, 0), 1.2f);
    CHECK_NEAR(angleOf(rotationBetween(slerp(a, b, 0.0f), a)), 0.0, 1e-4);
    CHECK_NEAR(angleOf(rotationBetween(slerp(a, b, 1.0f), b)), 0.0, 1e-4);
    CHECK_NEAR(angleOf(slerp(a, b, 0.5f)), 0.6, 1e-4);
}

TEST(Quat, slerpTakesTheShortPathThroughTheDoubleCover) {
    // q and -q are the same rotation; slerp must not travel the long way round.
    Quat a = Quat::fromAxisAngle(Vec3(0, 1, 0), 0.2f);
    Quat b = -Quat::fromAxisAngle(Vec3(0, 1, 0), 0.4f);
    Quat mid = slerp(a, b, 0.5f);
    CHECK_NEAR(angleOf(rotationBetween(a, mid)), 0.1, 1e-3);
}

TEST(Quat, rotationBetweenIsTheResidual) {
    Quat a = Quat::fromAxisAngle(Vec3(0.3f, 0.5f, -0.2f), 0.7f);
    Quat b = Quat::fromAxisAngle(Vec3(-0.1f, 0.8f, 0.4f), 1.3f);
    Quat d = rotationBetween(a, b);
    CHECK_NEAR(angleOf(rotationBetween(d * a, b)), 0.0, 1e-4);
}

// ------------------------------------------------------------------ Mat4

TEST(Mat4, perspectiveMapsNearPlaneToMinusOne) {
    Real n = 0.1f, f = 100.0f;
    Mat4 p = perspective(60.0f * kDeg2Rad, 1.6f, n, f);
    Vec4 clipNear = p * Vec4(0, 0, -n, 1);
    CHECK_NEAR(clipNear.z / clipNear.w, -1.0, 1e-4);
    Vec4 clipFar = p * Vec4(0, 0, -f, 1);
    CHECK_NEAR(clipFar.z / clipFar.w, 1.0, 1e-4);
}

TEST(Mat4, lookAtPlacesTheTargetDownTheNegativeZAxis) {
    Mat4 v = lookAt(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0));
    Vec4 eyeSpace = v * Vec4(0, 0, 0, 1);
    CHECK_VEC3_NEAR(eyeSpace.xyz(), 0.0, 0.0, -5.0, 1e-5);
}

TEST(Mat4, orthographicMapsCornersToNdc) {
    Mat4 o = orthographic(-2, 2, -1, 1, -1, 1);
    Vec4 c = o * Vec4(2, 1, 0, 1);
    CHECK_VEC3_NEAR(c.xyz(), 1.0, 1.0, 0.0, 1e-5);
}

TEST(Mat4, rigidTransformComposesRotationThenTranslation) {
    Quat q = Quat::fromAxisAngle(Vec3(0, 0, 1), kHalfPi);
    Mat4 m = Mat4::fromRotationTranslation(q, Vec3(1, 2, 3));
    Vec4 p = m * Vec4(1, 0, 0, 1);
    CHECK_VEC3_NEAR(p.xyz(), 1.0, 3.0, 3.0, 1e-5);
}

// ------------------------------------------------------------------ Rng

TEST(Rng, sameSeedReproducesTheSameStream) {
    Rng a(12345), b(12345);
    for (int i = 0; i < 64; ++i) CHECK(a.nextU32() == b.nextU32());
}

TEST(Rng, differentSeedsDiverge) {
    Rng a(12345), b(12346);
    bool diverged = false;
    for (int i = 0; i < 64; ++i) {
        if (a.uniform() != b.uniform()) diverged = true;
    }
    CHECK(diverged);
}

TEST(Rng, differentStreamsDiverge) {
    Rng a(1, 11), b(1, 22);
    bool diverged = false;
    for (int i = 0; i < 64; ++i) {
        if (a.nextU32() != b.nextU32()) diverged = true;
    }
    CHECK(diverged);
}

TEST(Rng, uniformStaysInRange) {
    Rng rng(99);
    for (int i = 0; i < 20000; ++i) {
        Real u = rng.uniform();
        CHECK(u >= 0.0f && u < 1.0f);
        Real r = rng.uniform(-3.0f, 7.0f);
        CHECK(r >= -3.0f && r < 7.0f);
        CHECK(rng.below(5) < 5u);
    }
}

TEST(Rng, gaussianHasUnitMomentsAndNoNaNs) {
    Rng rng(2024);
    double sum = 0, sumSq = 0;
    const int n = 200000;
    for (int i = 0; i < n; ++i) {
        Real g = rng.gaussian();
        CHECK(std::isfinite(g));
        sum += g;
        sumSq += double(g) * g;
    }
    double mean = sum / n;
    double var = sumSq / n - mean * mean;
    CHECK_NEAR(mean, 0.0, 0.02);
    CHECK_NEAR(var, 1.0, 0.02);
}

TEST(Rng, unitSphereSamplesAreUnitLength) {
    Rng rng(5);
    for (int i = 0; i < 1000; ++i) {
        CHECK_NEAR(length(rng.uniformOnUnitSphere()), 1.0, 1e-4);
        CHECK(length(rng.uniformInUnitCircle()) <= 1.0f + 1e-5f);
    }
}
