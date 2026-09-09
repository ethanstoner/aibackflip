#include "humanoid/RewardTerms.h"

#include <algorithm>
#include <cmath>

namespace aibf {

const std::vector<std::string>& rewardTermNames() {
    static const std::vector<std::string> kNames = {
        "alive",
        "pelvis_height",
        "head_height",
        "chest_upright",
        "head_upright",
        "com_over_support",
        "foot_contact",
        "horizontal_drift_cost",
        "vertical_drift_cost",
        "angular_drift_cost",
        "action_cost",
        "torque_cost",
        "joint_limit_cost",
        "pose_match",
        "joint_velocity_match",
        "end_effector_match",
        "root_match",
        "com_match",
    };
    static_assert(kTermCount == 18, "reward term names must match the enum");
    return kNames;
}

namespace {

// Horizontal centre of whatever is currently touching the ground. Returns false
// when nothing is, in which case "is the centre of mass over the feet" has no
// answer and the term is scored zero rather than guessed.
bool supportCenter(const World2D& world, const Humanoid2D& figure, Real& centerX) {
    Real minX = 0, maxX = 0;
    bool any = false;
    const int32_t footL = figure.bodyIndex(kFootL);
    const int32_t footR = figure.bodyIndex(kFootR);

    for (const Manifold& m : world.manifolds()) {
        if (m.bodyB != -1) continue;  // ground contacts only
        if (m.bodyA != footL && m.bodyA != footR) continue;
        for (int i = 0; i < m.pointCount; ++i) {
            if (m.points[i].separation > Real(0)) continue;  // speculative, not touching
            const Real x = m.points[i].position.x;
            if (!any) {
                minX = maxX = x;
                any = true;
            } else {
                minX = std::min(minX, x);
                maxX = std::max(maxX, x);
            }
        }
    }
    if (!any) return false;
    centerX = (minX + maxX) * Real(0.5);
    return true;
}

}  // namespace

void writeRewardTerms(const World2D& world, const Humanoid2D& figure,
                      const ObservationScales& scales, const Real* lastActions, Real physicsDt,
                      Real* out) {
    const Humanoid2DConfig& config = figure.config();
    const RigidBody2D& pelvis = figure.link(world, kPelvis);

    out[kTermAlive] = Real(1);
    out[kTermPelvisHeight] =
        clamp(pelvis.position.y / scales.pelvisRestHeight, Real(0), Real(1));
    out[kTermHeadHeight] =
        clamp(figure.headHeight(world) / scales.headRestHeight, Real(0), Real(1));
    out[kTermChestUpright] = std::max(Real(0), figure.uprightness(world, kChest));
    out[kTermHeadUpright] = std::max(Real(0), figure.uprightness(world, kHead));

    Real centerX = 0;
    if (supportCenter(world, figure, centerX)) {
        // Scaled by foot length, so "how far off centre" means the same thing
        // for a differently proportioned figure.
        const Real footLength =
            Real(2) * (config.links[kFootL].halfLength + config.links[kFootL].radius);
        const Real offset = std::abs(figure.centerOfMass(world).x - centerX) / footLength;
        out[kTermComOverSupport] = std::exp(-Real(2) * offset * offset);
    } else {
        out[kTermComOverSupport] = 0;
    }

    const int contacts = (figure.footContact(world, true) ? 1 : 0) +
                         (figure.footContact(world, false) ? 1 : 0);
    out[kTermFootContact] = Real(contacts) * Real(0.5);

    out[kTermHorizontalDriftCost] = std::abs(pelvis.velocity.x);
    out[kTermVerticalDriftCost] = std::abs(pelvis.velocity.y);
    out[kTermAngularDriftCost] = std::abs(pelvis.angularVelocity);

    Real actionCost = 0;
    if (lastActions) {
        for (int j = 0; j < kJointCount; ++j) {
            const Real a = clamp(lastActions[j], Real(-1), Real(1));
            actionCost += a * a;
        }
        actionCost /= Real(kJointCount);
    }
    out[kTermActionCost] = actionCost;

    // motorImpulse is accumulated over one substep, so dividing by the physics
    // timestep recovers the torque the motor actually applied.
    Real torqueCost = 0;
    for (int j = 0; j < kJointCount; ++j) {
        const Real ceiling = config.joints[static_cast<size_t>(j)].maxTorque;
        if (ceiling <= Real(0)) continue;
        const Real torque = std::abs(figure.joint(world, j).motorImpulse) / physicsDt;
        torqueCost += std::min(torque / ceiling, Real(1));
    }
    out[kTermTorqueCost] = torqueCost / Real(kJointCount);

    out[kTermJointLimitCost] = figure.worstLimitViolation(world);

    // Zero unless a reference motion fills them in; writeImitationTerms is a
    // separate call because only the imitation environment has the targets.
    for (int t = kTermPoseMatch; t < kTermCount; ++t) out[t] = Real(0);
}

Real poseTrackingError(const World2D& world, const Humanoid2D& figure,
                       const ImitationTargets& targets) {
    if (!targets.valid || targets.jointAngles.size() != static_cast<size_t>(kJointCount)) {
        return Real(0);
    }
    Real sum = 0;
    for (int j = 0; j < kJointCount; ++j) {
        const Real error =
            wrapAngle(figure.jointAngle(world, j) - targets.jointAngles[static_cast<size_t>(j)]);
        sum += error * error;
    }
    return std::sqrt(sum / Real(kJointCount));
}

void writeImitationTerms(const World2D& world, const Humanoid2D& figure,
                         const ImitationTargets& targets, const ImitationScales& scales,
                         const ObservationScales& observationScales, Real* out) {
    for (int t = kTermPoseMatch; t < kTermCount; ++t) out[t] = Real(0);
    if (!targets.valid) return;

    const RigidBody2D& pelvis = figure.link(world, kPelvis);

    // ---- pose ----
    Real poseError = 0;
    for (int j = 0; j < kJointCount; ++j) {
        // Wrapped, so a joint sitting near +/-pi does not read as a 2pi error.
        const Real error =
            wrapAngle(figure.jointAngle(world, j) - targets.jointAngles[static_cast<size_t>(j)]);
        poseError += error * error;
    }
    out[kTermPoseMatch] = std::exp(-scales.pose * poseError);

    // ---- joint velocity ----
    Real velocityError = 0;
    for (int j = 0; j < kJointCount; ++j) {
        const Real error =
            figure.jointVelocity(world, j) - targets.jointVelocities[static_cast<size_t>(j)];
        velocityError += error * error;
    }
    out[kTermJointVelocityMatch] = std::exp(-scales.jointVelocity * velocityError);

    // ---- end effectors ----
    //
    // Compared relative to the root. Absolute positions would fold the root
    // tracking error into this term as well, double-counting it and making the
    // two terms impossible to read apart in the logs.
    Real endEffectorError = 0;
    const size_t effectorCount =
        std::min(targets.endEffectors.size(), sizeof(kEndEffectors) / sizeof(kEndEffectors[0]));
    for (size_t e = 0; e < effectorCount; ++e) {
        const Vec2 actual = figure.link(world, kEndEffectors[e]).position - pelvis.position;
        endEffectorError += lengthSq(actual - targets.endEffectors[e]);
    }
    out[kTermEndEffectorMatch] = std::exp(-scales.endEffector * endEffectorError);

    // ---- root ----
    //
    // Height and orientation only. Horizontal position is deliberately excluded:
    // the observation hides absolute X, so demanding the figure be at a
    // particular X would be asking it to track something it cannot see.
    const Real heightError = pelvis.position.y - targets.rootHeight;
    const Real angleError = wrapAngle(pelvis.angle - targets.rootAngle);
    const Real rootError =
        heightError * heightError + scales.rootAngleWeight * angleError * angleError;
    out[kTermRootMatch] = std::exp(-scales.root * rootError);

    // ---- centre of mass ----
    const Vec2 comOffset = figure.centerOfMass(world) - pelvis.position;
    out[kTermComMatch] = std::exp(-scales.com * lengthSq(comOffset - targets.comOffset));

    (void)observationScales;
}

}  // namespace aibf
