#include "physics/Joint2D.h"

#include <algorithm>

namespace aibf {

namespace {

// Effective mass of the 2-DOF point constraint. Standard revolute-joint K
// matrix: the diagonal picks up each body's inverse mass plus the angular
// contribution of its lever arm, and the off-diagonal couples the two axes.
Mat2 pointEffectiveMass(const RigidBody2D& a, const RigidBody2D& b, const Vec2& rA,
                        const Vec2& rB) {
    const Real mA = a.invMass, mB = b.invMass;
    const Real iA = a.invInertia, iB = b.invInertia;
    const Real offDiagonal = -rA.y * rA.x * iA - rB.y * rB.x * iB;
    Mat2 k;
    k.c0.x = mA + mB + rA.y * rA.y * iA + rB.y * rB.y * iB;
    k.c0.y = offDiagonal;
    k.c1.x = offDiagonal;
    k.c1.y = mA + mB + rA.x * rA.x * iA + rB.x * rB.x * iB;
    return k;
}

}  // namespace

void RevoluteJoint2D::prepare(const RigidBody2D& a, const RigidBody2D& b, const SolverConfig& cfg,
                              Real dt) {
    (void)cfg;
    rA = a.localToWorldDir(localAnchorA);
    rB = b.localToWorldDir(localAnchorB);

    pointMassInv = inverse(pointEffectiveMass(a, b, rA, rB));

    const Real invAxial = a.invInertia + b.invInertia;
    axialMass = invAxial > Real(0) ? Real(1) / invAxial : Real(0);

    if (enableMotor) {
        // Implicit spring-damper. Solving `torque = kp*error - kd*rate` as a
        // soft velocity constraint rather than applying it as an explicit
        // external torque is what makes a stiff motor usable: the explicit form
        // is only stable while kp*dt^2 stays below the joint's inertia, which
        // rules out the torques a backflip needs.
        //
        //   gamma (CFM)  = 1 / (h*(kd + h*kp))
        //   bias         = kp/(kd + h*kp) * C
        //   softMass     = 1 / (invI_a + invI_b + gamma)
        const Real denom = damping + dt * stiffness;
        if (denom > Real(0)) {
            motorGamma = Real(1) / (dt * denom);
            const Real error = wrapAngle(relativeAngle(a, b) - targetAngle);
            motorBias = (stiffness / denom) * error;
        } else {
            motorGamma = 0;
            motorBias = 0;
        }
        const Real invSoft = invAxial + motorGamma;
        motorSoftMass = invSoft > Real(0) ? Real(1) / invSoft : Real(0);
        motorMaxImpulse = maxTorque * dt;
    } else {
        motorGamma = motorBias = motorSoftMass = motorMaxImpulse = 0;
    }
}

void RevoluteJoint2D::warmStart(RigidBody2D& a, RigidBody2D& b) {
    a.applyImpulse(-pointImpulse, rA);
    b.applyImpulse(pointImpulse, rB);

    const Real axial = motorImpulse + lowerImpulse - upperImpulse;
    a.applyAngularImpulse(-axial);
    b.applyAngularImpulse(axial);
}

void RevoluteJoint2D::solveVelocity(RigidBody2D& a, RigidBody2D& b, const SolverConfig& cfg,
                                    Real dt) {
    const Real invDt = Real(1) / dt;

    if (enableMotor && motorSoftMass > Real(0)) {
        const Real cdot = relativeAngularVelocity(a, b);
        Real impulse = -motorSoftMass * (cdot + motorBias + motorGamma * motorImpulse);
        const Real previous = motorImpulse;
        motorImpulse = clamp(previous + impulse, -motorMaxImpulse, motorMaxImpulse);
        impulse = motorImpulse - previous;
        a.applyAngularImpulse(-impulse);
        b.applyAngularImpulse(impulse);
    }

    if (enableLimit && axialMass > Real(0)) {
        const Real angleDelta = relativeAngle(a, b);

        // Lower limit. C > 0 means the limit is not yet reached, and the
        // speculative bias C/dt lets the joint travel exactly to the stop this
        // substep without passing through it.
        {
            const Real c = angleDelta - lowerAngle;
            const Real cdot = relativeAngularVelocity(a, b);
            const Real bias = (c > Real(0))
                                  ? c * invDt
                                  : cfg.limitBaumgarte * invDt *
                                        std::max(c, -cfg.maxAngularCorrection);
            Real impulse = -axialMass * (cdot + bias);
            const Real previous = lowerImpulse;
            lowerImpulse = std::max(previous + impulse, Real(0));
            impulse = lowerImpulse - previous;
            a.applyAngularImpulse(-impulse);
            b.applyAngularImpulse(impulse);
        }

        // Upper limit, same constraint with the sign of the axis flipped.
        {
            const Real c = upperAngle - angleDelta;
            const Real cdot = -relativeAngularVelocity(a, b);
            const Real bias = (c > Real(0))
                                  ? c * invDt
                                  : cfg.limitBaumgarte * invDt *
                                        std::max(c, -cfg.maxAngularCorrection);
            Real impulse = -axialMass * (cdot + bias);
            const Real previous = upperImpulse;
            upperImpulse = std::max(previous + impulse, Real(0));
            impulse = upperImpulse - previous;
            a.applyAngularImpulse(impulse);
            b.applyAngularImpulse(-impulse);
        }
    }

    // Point constraint last, so it has the final say on the anchors.
    {
        Vec2 cdot = b.velocityAtOffset(rB) - a.velocityAtOffset(rA);
        if (!cfg.useJointPositionSolver) {
            const Vec2 c = (b.position + rB) - (a.position + rA);
            cdot += c * (cfg.jointBaumgarte * invDt);
        }
        const Vec2 impulse = -(pointMassInv * cdot);
        pointImpulse += impulse;
        a.applyImpulse(-impulse, rA);
        b.applyImpulse(impulse, rB);
    }
}

Real RevoluteJoint2D::solvePosition(RigidBody2D& a, RigidBody2D& b, const SolverConfig& cfg) {
    // Recomputed from the current transforms rather than reusing the cached rA
    // and rB, because the position pass moves bodies as it goes.
    const Vec2 currentRa = a.localToWorldDir(localAnchorA);
    const Vec2 currentRb = b.localToWorldDir(localAnchorB);
    const Vec2 c = (b.position + currentRb) - (a.position + currentRa);
    const Real error = length(c);
    if (error < Real(1e-9)) return error;

    Vec2 impulse = -(inverse(pointEffectiveMass(a, b, currentRa, currentRb)) * c) *
                   cfg.jointRelaxation;

    const Real magnitude = length(impulse);
    if (magnitude > cfg.maxLinearCorrection) {
        impulse = impulse * (cfg.maxLinearCorrection / magnitude);
    }

    a.position -= impulse * a.invMass;
    a.angle -= a.invInertia * cross(currentRa, impulse);
    b.position += impulse * b.invMass;
    b.angle += b.invInertia * cross(currentRb, impulse);

    return error;
}

}  // namespace aibf
