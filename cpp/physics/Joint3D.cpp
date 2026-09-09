#include "physics/Joint3D.h"

#include <algorithm>

namespace aibf {

Mat3 pointEffectiveMass3D(const RigidBody3D& a, const RigidBody3D& b,
                          const Vec3& rA, const Vec3& rB) {
    const Real m = a.invMass + b.invMass;
    const Mat3 sA = skew(rA);
    const Mat3 sB = skew(rB);
    // [r]x^T I [r]x, written with the transpose rather than folding in the sign,
    // because the sign is exactly the thing that is easy to get backwards and
    // hard to notice: the wrong one still produces a symmetric matrix.
    return Mat3::diagonal(m, m, m) +
           transpose(sA) * a.invInertiaWorld * sA +
           transpose(sB) * b.invInertiaWorld * sB;
}

Real angularMassAboutAxis(const RigidBody3D& a, const RigidBody3D& b, const Vec3& axis) {
    const Real inv = dot(axis, a.invInertiaWorld * axis) + dot(axis, b.invInertiaWorld * axis);
    return inv > Real(0) ? Real(1) / inv : Real(0);
}

namespace {

// One-sided angular constraint about `axis`, shared by the cone and twist
// limits and by the hinge's own limits. `c` is the constraint function, positive
// when inside the limit; `rate` is its time derivative.
//
// Identical in shape to the 2D limit solve, including the speculative branch:
// when C > 0 the bias C/dt lets the joint travel exactly to the stop this
// substep without passing through it.
void solveAngularLimit(RigidBody3D& a, RigidBody3D& b, const Vec3& axis, Real c, Real rate,
                       Real mass, Real& accumulated, const SolverConfig& cfg, Real invDt) {
    if (mass <= Real(0)) return;
    const Real bias = (c > Real(0))
                          ? c * invDt
                          : cfg.limitBaumgarte * invDt * std::max(c, -cfg.maxAngularCorrection);
    Real impulse = -mass * (rate + bias);
    const Real previous = accumulated;
    accumulated = std::max(previous + impulse, Real(0));
    impulse = accumulated - previous;

    const Vec3 angular = axis * impulse;
    a.applyAngularImpulse(-angular);
    b.applyAngularImpulse(angular);
}

// Implicit spring-damper coefficients, the 3D form of the scalar version in
// Joint2D. Solving `torque = kp*error - kd*rate` as a soft velocity constraint
// rather than an explicit external torque is what makes a stiff motor usable:
// the explicit form is only stable while kp*dt^2 stays below the joint's
// inertia, which rules out the torques a backflip needs.
struct SoftConstraint {
    Real gamma = 0;
    Real biasScale = 0;
};

SoftConstraint softCoefficients(Real stiffness, Real damping, Real dt) {
    SoftConstraint out;
    const Real denom = damping + dt * stiffness;
    if (denom > Real(0)) {
        out.gamma = Real(1) / (dt * denom);
        out.biasScale = stiffness / denom;
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------- ball joint

Real BallJoint3D::swingAngle(const RigidBody3D& a, const RigidBody3D& b) const {
    const Vec3 axisA = a.localToWorldDir(rotate(referenceRotation, localAxisA));
    const Vec3 axisB = b.localToWorldDir(localAxisB);
    return std::acos(clamp(dot(normalize(axisA), normalize(axisB)), Real(-1), Real(1)));
}

void BallJoint3D::prepare(const RigidBody3D& a, const RigidBody3D& b, const SolverConfig& cfg,
                          Real dt) {
    (void)cfg;
    rA = a.localToWorldDir(localAnchorA);
    rB = b.localToWorldDir(localAnchorB);
    pointMassInv = inverse(pointEffectiveMass3D(a, b, rA, rB));

    if (enableMotor) {
        const SoftConstraint soft = softCoefficients(stiffness, damping, dt);
        motorGamma = soft.gamma;

        // The error is a rotation vector, so one constraint covers all three
        // axes and the shortest-arc choice inside rotationVector keeps the motor
        // from taking the long way round a nearly-complete turn.
        const Quat current = relativeRotation(a, b);
        const Quat error = conjugate(targetRotation) * current;
        motorBias = rotationVector(error) * soft.biasScale;

        const Mat3 invMassMatrix = a.invInertiaWorld + b.invInertiaWorld +
                                   Mat3::diagonal(motorGamma, motorGamma, motorGamma);
        motorSoftMassInv = inverse(invMassMatrix);
        motorMaxImpulse = maxTorque * dt;
    } else {
        motorSoftMassInv = Mat3::zero();
        motorBias = Vec3(0, 0, 0);
        motorGamma = 0;
        motorMaxImpulse = 0;
    }

    if (enableConeLimit) {
        const Vec3 axisA = normalize(a.localToWorldDir(rotate(referenceRotation, localAxisA)));
        const Vec3 axisB = normalize(b.localToWorldDir(localAxisB));
        const Vec3 perpendicular = cross(axisA, axisB);
        const Real sine = length(perpendicular);
        // Exactly aligned or exactly opposed: there is no unique rotation axis.
        // Aligned is deep inside the cone and needs no constraint; opposed is a
        // half turn, where any perpendicular reduces the angle equally.
        if (sine > Real(1e-5)) {
            coneAxis = perpendicular / sine;
        } else {
            Vec3 t1, t2;
            orthonormalBasis(axisA, t1, t2);
            coneAxis = t1;
        }
        coneError = coneAngle - swingAngle(a, b);
        coneMass = angularMassAboutAxis(a, b, coneAxis);
    } else {
        coneMass = 0;
        coneError = 0;
    }

    if (enableTwistLimit) {
        twistAxis = normalize(b.localToWorldDir(localAxisB));
        const Quat current = relativeRotation(a, b);
        twistValue = twistAngle(current, normalize(localAxisB));
        twistMass = angularMassAboutAxis(a, b, twistAxis);
    } else {
        twistMass = 0;
        twistValue = 0;
    }
}

void BallJoint3D::warmStart(RigidBody3D& a, RigidBody3D& b) {
    a.applyImpulse(-pointImpulse, rA);
    b.applyImpulse(pointImpulse, rB);

    Vec3 angular = motorImpulse;
    if (enableConeLimit) angular += coneAxis * -coneImpulse;
    if (enableTwistLimit) angular += twistAxis * (lowerTwistImpulse - upperTwistImpulse);
    a.applyAngularImpulse(-angular);
    b.applyAngularImpulse(angular);
}

void BallJoint3D::solveVelocity(RigidBody3D& a, RigidBody3D& b, const SolverConfig& cfg, Real dt) {
    const Real invDt = Real(1) / dt;

    if (enableMotor) {
        const Vec3 cdot = relativeAngularVelocity(a, b);
        Vec3 impulse = -(motorSoftMassInv * (cdot + motorBias + motorImpulse * motorGamma));
        Vec3 total = motorImpulse + impulse;
        // Clamped by magnitude rather than per axis, so the torque limit means
        // the same thing whichever way the joint happens to be oriented.
        const Real magnitude = length(total);
        if (motorMaxImpulse > Real(0) && magnitude > motorMaxImpulse) {
            total = total * (motorMaxImpulse / magnitude);
        }
        impulse = total - motorImpulse;
        motorImpulse = total;
        a.applyAngularImpulse(-impulse);
        b.applyAngularImpulse(impulse);
    }

    if (enableConeLimit && coneMass > Real(0)) {
        // Rotating B about +coneAxis increases the swing, so the rate that
        // reduces the constraint is the negative projection.
        const Real rate = -dot(relativeAngularVelocity(a, b), coneAxis);
        Real impulse = -coneMass * (rate + ((coneError > Real(0))
                                                ? coneError * invDt
                                                : cfg.limitBaumgarte * invDt *
                                                      std::max(coneError,
                                                               -cfg.maxAngularCorrection)));
        const Real previous = coneImpulse;
        coneImpulse = std::max(previous + impulse, Real(0));
        impulse = coneImpulse - previous;
        const Vec3 angular = coneAxis * -impulse;
        a.applyAngularImpulse(-angular);
        b.applyAngularImpulse(angular);
    }

    if (enableTwistLimit && twistMass > Real(0)) {
        const Real rate = dot(relativeAngularVelocity(a, b), twistAxis);
        solveAngularLimit(a, b, twistAxis, twistValue - lowerTwist, rate, twistMass,
                          lowerTwistImpulse, cfg, invDt);
        solveAngularLimit(a, b, -twistAxis, upperTwist - twistValue, -rate, twistMass,
                          upperTwistImpulse, cfg, invDt);
    }

    // Point constraint last, so it has the final say on the anchors.
    {
        Vec3 cdot = b.velocityAtOffset(rB) - a.velocityAtOffset(rA);
        if (!cfg.useJointPositionSolver) {
            const Vec3 c = (b.position + rB) - (a.position + rA);
            cdot += c * (cfg.jointBaumgarte * invDt);
        }
        const Vec3 impulse = -(pointMassInv * cdot);
        pointImpulse += impulse;
        a.applyImpulse(-impulse, rA);
        b.applyImpulse(impulse, rB);
    }
}

Real BallJoint3D::solvePosition(RigidBody3D& a, RigidBody3D& b, const SolverConfig& cfg) {
    // Recomputed from the current transforms rather than reusing rA and rB,
    // because the position pass moves bodies as it goes.
    const Vec3 currentRa = a.localToWorldDir(localAnchorA);
    const Vec3 currentRb = b.localToWorldDir(localAnchorB);
    const Vec3 c = (b.position + currentRb) - (a.position + currentRa);
    const Real error = length(c);
    if (error < Real(1e-9)) return error;

    Vec3 impulse = -(inverse(pointEffectiveMass3D(a, b, currentRa, currentRb)) * c) *
                   cfg.jointRelaxation;
    const Real magnitude = length(impulse);
    if (magnitude > cfg.maxLinearCorrection) {
        impulse = impulse * (cfg.maxLinearCorrection / magnitude);
    }

    a.position -= impulse * a.invMass;
    b.position += impulse * b.invMass;
    if (a.invMass > Real(0) || length(a.invInertiaWorld * Vec3(1, 1, 1)) > Real(0)) {
        a.orientation = integrateExact(a.orientation,
                                       a.invInertiaWorld * cross(currentRa, -impulse), Real(1));
        a.refreshInertiaWorld();
    }
    if (b.invMass > Real(0) || length(b.invInertiaWorld * Vec3(1, 1, 1)) > Real(0)) {
        b.orientation = integrateExact(b.orientation,
                                       b.invInertiaWorld * cross(currentRb, impulse), Real(1));
        b.refreshInertiaWorld();
    }

    return error;
}

// ---------------------------------------------------------------- hinge joint

Real HingeJoint3D::jointAngle(const RigidBody3D& a, const RigidBody3D& b) const {
    const Quat relative = conjugate(a.orientation) * b.orientation * conjugate(referenceRotation);
    return twistAngle(relative, normalize(localHingeAxisA));
}

Real HingeJoint3D::axisMisalignment(const RigidBody3D& a, const RigidBody3D& b) const {
    const Vec3 axisA = normalize(a.localToWorldDir(localHingeAxisA));
    const Vec3 axisB = normalize(b.localToWorldDir(localHingeAxisB));
    return std::acos(clamp(dot(axisA, axisB), Real(-1), Real(1)));
}

void HingeJoint3D::prepare(const RigidBody3D& a, const RigidBody3D& b, const SolverConfig& cfg,
                           Real dt) {
    (void)cfg;
    rA = a.localToWorldDir(localAnchorA);
    rB = b.localToWorldDir(localAnchorB);
    pointMassInv = inverse(pointEffectiveMass3D(a, b, rA, rB));

    hingeAxisWorld = normalize(a.localToWorldDir(localHingeAxisA));
    orthonormalBasis(hingeAxisWorld, lockAxis1, lockAxis2);

    // 2x2 effective mass for the two axes the hinge must not rotate about.
    const Mat3 invI = a.invInertiaWorld + b.invInertiaWorld;
    Mat2 k;
    k.c0.x = dot(lockAxis1, invI * lockAxis1);
    k.c0.y = dot(lockAxis2, invI * lockAxis1);
    k.c1.x = dot(lockAxis1, invI * lockAxis2);
    k.c1.y = dot(lockAxis2, invI * lockAxis2);
    lockMassInv = inverse(k);

    axialMass = angularMassAboutAxis(a, b, hingeAxisWorld);

    if (enableMotor) {
        const SoftConstraint soft = softCoefficients(stiffness, damping, dt);
        motorGamma = soft.gamma;
        motorBias = soft.biasScale * wrapAngle(jointAngle(a, b) - targetAngle);
        const Real invSoft = (axialMass > Real(0) ? Real(1) / axialMass : Real(0)) + motorGamma;
        motorSoftMass = invSoft > Real(0) ? Real(1) / invSoft : Real(0);
        motorMaxImpulse = maxTorque * dt;
    } else {
        motorGamma = motorBias = motorSoftMass = motorMaxImpulse = 0;
    }
}

void HingeJoint3D::warmStart(RigidBody3D& a, RigidBody3D& b) {
    a.applyImpulse(-pointImpulse, rA);
    b.applyImpulse(pointImpulse, rB);

    Vec3 angular = lockAxis1 * lockImpulse.x + lockAxis2 * lockImpulse.y +
                   hingeAxisWorld * (motorImpulse + lowerImpulse - upperImpulse);
    a.applyAngularImpulse(-angular);
    b.applyAngularImpulse(angular);
}

void HingeJoint3D::solveVelocity(RigidBody3D& a, RigidBody3D& b, const SolverConfig& cfg,
                                 Real dt) {
    const Real invDt = Real(1) / dt;

    if (enableMotor && motorSoftMass > Real(0)) {
        const Real cdot = jointRate(a, b);
        Real impulse = -motorSoftMass * (cdot + motorBias + motorGamma * motorImpulse);
        const Real previous = motorImpulse;
        motorImpulse = clamp(previous + impulse, -motorMaxImpulse, motorMaxImpulse);
        impulse = motorImpulse - previous;
        const Vec3 angular = hingeAxisWorld * impulse;
        a.applyAngularImpulse(-angular);
        b.applyAngularImpulse(angular);
    }

    if (enableLimit && axialMass > Real(0)) {
        const Real angle = jointAngle(a, b);
        const Real rate = jointRate(a, b);
        solveAngularLimit(a, b, hingeAxisWorld, angle - lowerAngle, rate, axialMass,
                          lowerImpulse, cfg, invDt);
        solveAngularLimit(a, b, -hingeAxisWorld, upperAngle - angle, -rate, axialMass,
                          upperImpulse, cfg, invDt);
    }

    // The two axes the hinge does not turn about. Without this a knee is a ball
    // joint that happens to have a limit on one axis, and the shin swings
    // sideways under load.
    {
        const Vec3 relative = b.angularVelocity - a.angularVelocity;
        const Vec2 cdot(dot(relative, lockAxis1), dot(relative, lockAxis2));
        const Vec2 lambda = -(lockMassInv * cdot);
        lockImpulse += lambda;
        const Vec3 angular = lockAxis1 * lambda.x + lockAxis2 * lambda.y;
        a.applyAngularImpulse(-angular);
        b.applyAngularImpulse(angular);
    }

    // Point constraint last.
    {
        Vec3 cdot = b.velocityAtOffset(rB) - a.velocityAtOffset(rA);
        if (!cfg.useJointPositionSolver) {
            const Vec3 c = (b.position + rB) - (a.position + rA);
            cdot += c * (cfg.jointBaumgarte * invDt);
        }
        const Vec3 impulse = -(pointMassInv * cdot);
        pointImpulse += impulse;
        a.applyImpulse(-impulse, rA);
        b.applyImpulse(impulse, rB);
    }
}

Real HingeJoint3D::solvePosition(RigidBody3D& a, RigidBody3D& b, const SolverConfig& cfg) {
    const Vec3 currentRa = a.localToWorldDir(localAnchorA);
    const Vec3 currentRb = b.localToWorldDir(localAnchorB);
    const Vec3 c = (b.position + currentRb) - (a.position + currentRa);
    const Real error = length(c);
    if (error < Real(1e-9)) return error;

    Vec3 impulse = -(inverse(pointEffectiveMass3D(a, b, currentRa, currentRb)) * c) *
                   cfg.jointRelaxation;
    const Real magnitude = length(impulse);
    if (magnitude > cfg.maxLinearCorrection) {
        impulse = impulse * (cfg.maxLinearCorrection / magnitude);
    }

    a.position -= impulse * a.invMass;
    b.position += impulse * b.invMass;
    a.orientation = integrateExact(a.orientation,
                                   a.invInertiaWorld * cross(currentRa, -impulse), Real(1));
    a.refreshInertiaWorld();
    b.orientation = integrateExact(b.orientation,
                                   b.invInertiaWorld * cross(currentRb, impulse), Real(1));
    b.refreshInertiaWorld();

    return error;
}

}  // namespace aibf
