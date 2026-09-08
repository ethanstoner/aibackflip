// Revolute (pin) joint: two bodies share a point, with optional angle limits
// and an optional motor driving the relative angle towards a target.
//
// Three constraints stack on the same joint:
//   * a one-DOF motor,
//   * two one-sided angular constraints for the limits,
//   * a 2-DOF point constraint holding the anchors together.
// They are solved in that order every iteration. The point constraint goes last
// deliberately: whichever constraint is solved last gets the final say, and a
// limb detaching from its socket is a far worse failure than a motor missing
// its target by a few degrees.
#pragma once

#include <cstdint>

#include "core/Math.h"
#include "physics/RigidBody2D.h"
#include "physics/Solver2D.h"

namespace aibf {

struct RevoluteJoint2D {
    // ---- definition ----
    int32_t bodyA = -1;
    int32_t bodyB = -1;
    Vec2 localAnchorA;          // anchor in A's local frame
    Vec2 localAnchorB;          // anchor in B's local frame
    Real referenceAngle = 0;    // value of (angleB - angleA) that counts as zero

    bool enableLimit = false;
    Real lowerAngle = -kPi;
    Real upperAngle = kPi;

    // ---- motor (driven by the policy; see solveMotor for the formulation) ----
    bool enableMotor = false;
    Real targetAngle = 0;
    Real stiffness = 0;   // kp, N*m per radian
    Real damping = 0;     // kd, N*m per (radian/second)
    Real maxTorque = 0;   // hard clamp on |applied torque|

    // ---- solver scratch, valid between prepare() and the end of the substep ----
    Vec2 rA, rB;
    Mat2 pointMassInv;      // inverse of the 2x2 effective mass
    Real axialMass = 0;     // 1 / (invIa + invIb), or 0 if both are static
    Real motorSoftMass = 0;
    Real motorBias = 0;
    Real motorGamma = 0;
    Real motorMaxImpulse = 0;

    Vec2 pointImpulse;      // accumulated, warm-started across substeps
    Real lowerImpulse = 0;
    Real upperImpulse = 0;
    Real motorImpulse = 0;

    // Relative angle in the joint's own convention.
    Real relativeAngle(const RigidBody2D& a, const RigidBody2D& b) const {
        return b.angle - a.angle - referenceAngle;
    }
    Real relativeAngularVelocity(const RigidBody2D& a, const RigidBody2D& b) const {
        return b.angularVelocity - a.angularVelocity;
    }

    // Distance the two anchors have drifted apart. This is the number M1's exit
    // criterion is measured with.
    Real anchorError(const RigidBody2D& a, const RigidBody2D& b) const {
        return length(b.localToWorld(localAnchorB) - a.localToWorld(localAnchorA));
    }

    void resetAccumulators() {
        pointImpulse = Vec2(0, 0);
        lowerImpulse = upperImpulse = motorImpulse = 0;
    }

    void prepare(const RigidBody2D& a, const RigidBody2D& b, const SolverConfig& cfg, Real dt);
    void warmStart(RigidBody2D& a, RigidBody2D& b);
    void solveVelocity(RigidBody2D& a, RigidBody2D& b, const SolverConfig& cfg, Real dt);
    // Returns the remaining anchor error, so the caller can report convergence.
    Real solvePosition(RigidBody2D& a, RigidBody2D& b, const SolverConfig& cfg);
};

}  // namespace aibf
