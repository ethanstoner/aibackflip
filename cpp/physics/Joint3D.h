// 3D joints: a ball-and-socket with cone and twist limits, and a hinge.
//
// The stacking order from the 2D revolute joint carries over unchanged, and for
// the same reason. Every joint solves, in this order:
//
//   * the motor,
//   * the angular limits,
//   * the 3-DOF point constraint holding the anchors together.
//
// The point constraint goes last deliberately: whichever constraint is solved
// last gets the final say, and a limb detaching from its socket is a far worse
// failure than a motor missing its target by a few degrees.
//
// What is genuinely new in 3D:
//
//   * A shoulder has two different limits, not one. How far the arm has swung
//     away from its rest direction and how far it has twisted about its own
//     length are independent, and a single angle cannot express both. They are
//     separated with a swing-twist decomposition.
//   * The motor drives a relative *orientation*, so its error is a rotation
//     vector rather than a scalar, and its effective mass is a 3x3 matrix.
//   * A hinge has to actively forbid the two axes it does not turn about. In 2D
//     that was free, because there was only one axis to begin with.
#pragma once

#include <cstdint>

#include "core/Math.h"
#include "physics/RigidBody3D.h"
#include "physics/Solver2D.h"  // SolverConfig is dimension-agnostic

namespace aibf {

// Effective mass of a 3-DOF point constraint:
//   K = (1/mA + 1/mB) I  +  [rA]x^T IA^-1 [rA]x  +  [rB]x^T IB^-1 [rB]x
Mat3 pointEffectiveMass3D(const RigidBody3D& a, const RigidBody3D& b,
                          const Vec3& rA, const Vec3& rB);

// Effective mass for a 1-DOF angular constraint about a world axis.
Real angularMassAboutAxis(const RigidBody3D& a, const RigidBody3D& b, const Vec3& axis);

// Ball-and-socket. Used for shoulders and hips.
struct BallJoint3D {
    // ---- definition ----
    int32_t bodyA = -1;
    int32_t bodyB = -1;
    Vec3 localAnchorA;
    Vec3 localAnchorB;
    // Relative orientation (A to B) that counts as zero, so a figure authored in
    // a rest pose reads zero at every joint by construction.
    Quat referenceRotation = Quat::identity();

    // The limb's own long axis, in each body's frame. Swing is measured as the
    // angle between these two once the reference rotation is taken out.
    Vec3 localAxisA{0, 1, 0};
    Vec3 localAxisB{0, 1, 0};

    // ---- limits ----
    bool enableConeLimit = false;
    Real coneAngle = kPi;        // maximum swing away from the reference axis
    bool enableTwistLimit = false;
    Real lowerTwist = -kPi;
    Real upperTwist = kPi;

    // ---- motor ----
    bool enableMotor = false;
    Quat targetRotation = Quat::identity();  // desired relative orientation
    Real stiffness = 0;   // kp, N*m per radian
    Real damping = 0;     // kd, N*m per (radian/second)
    Real maxTorque = 0;   // hard clamp on |applied torque|

    // ---- solver scratch ----
    Vec3 rA, rB;
    Mat3 pointMassInv;
    Mat3 motorSoftMassInv;
    Vec3 motorBias;
    Real motorGamma = 0;
    Real motorMaxImpulse = 0;

    Vec3 coneAxis;         // world axis the cone limit acts about
    Real coneMass = 0;
    Real coneError = 0;    // C, positive means inside the limit

    Vec3 twistAxis;        // world axis the twist limit acts about
    Real twistMass = 0;
    Real twistValue = 0;

    Vec3 pointImpulse;     // accumulated, warm-started across substeps
    Vec3 motorImpulse;
    Real lowerTwistImpulse = 0;
    Real upperTwistImpulse = 0;
    Real coneImpulse = 0;

    // Relative rotation in the joint's own convention.
    Quat relativeRotation(const RigidBody3D& a, const RigidBody3D& b) const {
        return conjugate(a.orientation) * b.orientation * conjugate(referenceRotation);
    }
    Vec3 relativeAngularVelocity(const RigidBody3D& a, const RigidBody3D& b) const {
        return b.angularVelocity - a.angularVelocity;
    }

    // Angle between the two limb axes; the quantity the cone limit bounds.
    Real swingAngle(const RigidBody3D& a, const RigidBody3D& b) const;

    // Distance the two anchors have drifted apart, the number the exit criterion
    // is measured with.
    Real anchorError(const RigidBody3D& a, const RigidBody3D& b) const {
        return length(b.localToWorld(localAnchorB) - a.localToWorld(localAnchorA));
    }

    void resetAccumulators() {
        pointImpulse = Vec3(0, 0, 0);
        motorImpulse = Vec3(0, 0, 0);
        lowerTwistImpulse = upperTwistImpulse = coneImpulse = 0;
    }

    void prepare(const RigidBody3D& a, const RigidBody3D& b, const SolverConfig& cfg, Real dt);
    void warmStart(RigidBody3D& a, RigidBody3D& b);
    void solveVelocity(RigidBody3D& a, RigidBody3D& b, const SolverConfig& cfg, Real dt);
    Real solvePosition(RigidBody3D& a, RigidBody3D& b, const SolverConfig& cfg);
};

// Hinge. Used for knees, elbows and ankles: one rotational degree of freedom
// with a scalar limit, exactly like the 2D revolute joint, plus the two-axis
// angular lock that 2D got for free.
struct HingeJoint3D {
    // ---- definition ----
    int32_t bodyA = -1;
    int32_t bodyB = -1;
    Vec3 localAnchorA;
    Vec3 localAnchorB;
    Vec3 localHingeAxisA{0, 0, 1};
    Vec3 localHingeAxisB{0, 0, 1};
    Quat referenceRotation = Quat::identity();

    bool enableLimit = false;
    Real lowerAngle = -kPi;
    Real upperAngle = kPi;

    bool enableMotor = false;
    Real targetAngle = 0;
    Real stiffness = 0;
    Real damping = 0;
    Real maxTorque = 0;

    // ---- solver scratch ----
    Vec3 rA, rB;
    Mat3 pointMassInv;
    Vec3 hingeAxisWorld;
    // The two directions the hinge must not rotate about.
    Vec3 lockAxis1, lockAxis2;
    Mat2 lockMassInv;
    Real axialMass = 0;
    Real motorSoftMass = 0;
    Real motorBias = 0;
    Real motorGamma = 0;
    Real motorMaxImpulse = 0;

    Vec3 pointImpulse;
    Vec2 lockImpulse;
    Real motorImpulse = 0;
    Real lowerImpulse = 0;
    Real upperImpulse = 0;

    Real jointAngle(const RigidBody3D& a, const RigidBody3D& b) const;
    Real jointRate(const RigidBody3D& a, const RigidBody3D& b) const {
        return dot(b.angularVelocity - a.angularVelocity, a.localToWorldDir(localHingeAxisA));
    }
    Real anchorError(const RigidBody3D& a, const RigidBody3D& b) const {
        return length(b.localToWorld(localAnchorB) - a.localToWorld(localAnchorA));
    }
    // How far the hinge has bent out of its own plane. Zero for a healthy hinge;
    // a knee that starts bending sideways shows up here before it looks wrong.
    Real axisMisalignment(const RigidBody3D& a, const RigidBody3D& b) const;

    void resetAccumulators() {
        pointImpulse = Vec3(0, 0, 0);
        lockImpulse = Vec2(0, 0);
        motorImpulse = lowerImpulse = upperImpulse = 0;
    }

    void prepare(const RigidBody3D& a, const RigidBody3D& b, const SolverConfig& cfg, Real dt);
    void warmStart(RigidBody3D& a, RigidBody3D& b);
    void solveVelocity(RigidBody3D& a, RigidBody3D& b, const SolverConfig& cfg, Real dt);
    Real solvePosition(RigidBody3D& a, RigidBody3D& b, const SolverConfig& cfg);
};

}  // namespace aibf
