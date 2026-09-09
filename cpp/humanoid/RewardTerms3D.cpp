#include "humanoid/RewardTerms3D.h"

#include <algorithm>
#include <cmath>

namespace aibf {

namespace {

// How upright a link is, compared against *its own* rest orientation rather
// than against world up. A foot rests horizontal, so scoring it against world up
// would permanently mark it as fallen over.
Real uprightness(const RigidBody3D& body, const Quat& restOrientation) {
    const Vec3 current = body.localToWorldDir(Vec3(0, 1, 0));
    const Vec3 rest = rotate(restOrientation, Vec3(0, 1, 0));
    return std::max(Real(0), dot(current, rest));
}

}  // namespace

void writeRewardTerms3D(const World3D& world, const Humanoid3D& figure,
                        const ObservationScales3D& scales, const Real* actions, int actionCount,
                        bool alive, Real* out) {
    for (int i = 0; i < kTermCount; ++i) out[i] = 0;

    out[kTermAlive] = alive ? Real(1) : Real(0);

    const Humanoid3DConfig& cfg = figure.config();
    const RigidBody3D& pelvis = figure.link(world, kPelvis);
    const RigidBody3D& chest = figure.link(world, kChest);
    const RigidBody3D& head = figure.link(world, kHead);

    const Real restPelvis = cfg.links[kPelvis].restPosition.y;
    const Real restHead = cfg.links[kHead].restPosition.y;

    out[kTermPelvisHeight] = clamp(pelvis.position.y / restPelvis, Real(0), Real(1));
    out[kTermHeadHeight] = clamp(head.position.y / restHead, Real(0), Real(1));
    out[kTermChestUpright] = uprightness(chest, cfg.links[kChest].restOrientation);
    out[kTermHeadUpright] = uprightness(head, cfg.links[kHead].restOrientation);

    // Balance in 3D is a two-dimensional problem: the centre of mass can leave
    // the support polygon sideways as easily as forwards, and the 2D figure
    // could not fall that way at all. Both horizontal axes count here.
    {
        const Vec3 com = figure.centerOfMass(world);
        Vec3 support(0, 0, 0);
        int contacts = 0;
        for (const int foot : {kFootL, kFootR}) {
            if (!figure.footInContact(world, foot)) continue;
            support += figure.link(world, foot).position;
            ++contacts;
        }
        if (contacts > 0) {
            support = support / Real(contacts);
            const Real dx = com.x - support.x;
            const Real dz = com.z - support.z;
            out[kTermComOverSupport] = std::exp(Real(-20) * (dx * dx + dz * dz));
        }
        // With no feet down there is no support polygon, so the term stays zero
        // rather than inventing a value from the last one that existed.
    }

    {
        int down = 0;
        if (figure.footInContact(world, kFootL)) ++down;
        if (figure.footInContact(world, kFootR)) ++down;
        out[kTermFootContact] = Real(down) * Real(0.5);
    }

    // Horizontal drift is the length of the pelvis's horizontal velocity, again
    // because there are now two ways to drift.
    out[kTermHorizontalDriftCost] =
        std::sqrt(pelvis.velocity.x * pelvis.velocity.x + pelvis.velocity.z * pelvis.velocity.z);
    out[kTermVerticalDriftCost] = std::abs(pelvis.velocity.y);
    out[kTermAngularDriftCost] = length(pelvis.angularVelocity);

    if (actions != nullptr && actionCount > 0) {
        Real sum = 0;
        for (int i = 0; i < actionCount; ++i) {
            const Real a = clamp(actions[i], Real(-1), Real(1));
            sum += a * a;
        }
        out[kTermActionCost] = sum / Real(actionCount);
    }

    // Torque as a fraction of each joint's own ceiling, averaged.
    //
    // Compared against `motorMaxImpulse` rather than against `maxTorque`,
    // because what the solver accumulates is an impulse, and the joint already
    // holds the matching ceiling (maxTorque * dt). Dividing an impulse by a
    // torque would produce a number that changed meaning with the timestep.
    //
    // Per joint rather than in newton metres, so one heavily loaded hip does
    // not dominate a sum over twelve joints of very different sizes.
    {
        Real sum = 0;
        int counted = 0;
        for (int i = 0; i < kJointCount; ++i) {
            const int32_t ball = figure.ballIndex(i);
            if (ball >= 0) {
                const BallJoint3D& joint = world.ballJoint(ball);
                if (!(joint.motorMaxImpulse > Real(0))) continue;
                sum += clamp(length(joint.motorImpulse) / joint.motorMaxImpulse, Real(0), Real(1));
            } else {
                const HingeJoint3D& joint = world.hingeJoint(figure.hingeIndex(i));
                if (!(joint.motorMaxImpulse > Real(0))) continue;
                sum += clamp(std::abs(joint.motorImpulse) / joint.motorMaxImpulse, Real(0),
                             Real(1));
            }
            ++counted;
        }
        out[kTermTorqueCost] = counted > 0 ? sum / Real(counted) : Real(0);
    }

    // Worst limit violation across every joint, in radians. Zero for a figure
    // inside its limits, which is the normal case.
    {
        Real worst = 0;
        for (int i = 0; i < kJointCount; ++i) {
            const JointConfig3D& jc = cfg.joints[static_cast<size_t>(i)];
            const int32_t ball = figure.ballIndex(i);
            if (ball >= 0) {
                const BallJoint3D& joint = world.ballJoint(ball);
                const Real swing =
                    joint.swingAngle(world.body(joint.bodyA), world.body(joint.bodyB));
                worst = std::max(worst, swing - jc.coneAngle);
                const Real twist = twistAngle(
                    joint.relativeRotation(world.body(joint.bodyA), world.body(joint.bodyB)),
                    normalize(joint.localAxisB));
                worst = std::max(worst, twist - jc.upperTwist);
                worst = std::max(worst, jc.lowerTwist - twist);
                continue;
            }
            const int32_t hinge = figure.hingeIndex(i);
            const HingeJoint3D& joint = world.hingeJoint(hinge);
            const Real angle = joint.jointAngle(world.body(joint.bodyA), world.body(joint.bodyB));
            worst = std::max(worst, angle - jc.upperLimit);
            worst = std::max(worst, jc.lowerLimit - angle);
        }
        out[kTermJointLimitCost] = std::max(Real(0), worst);
    }

    (void)scales;
}

}  // namespace aibf
