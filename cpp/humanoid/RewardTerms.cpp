#include "humanoid/RewardTerms.h"

#include <algorithm>
#include <cmath>

namespace aibf {

const std::vector<std::string>& rewardTermNames() {
    static const std::vector<std::string> kNames = {
        "alive",         "pelvis_height",       "head_height",      "chest_upright",
        "head_upright",  "com_over_support",    "foot_contact",     "horizontal_drift_cost",
        "vertical_drift_cost", "angular_drift_cost", "action_cost",  "torque_cost",
        "joint_limit_cost",
    };
    static_assert(kTermCount == 13, "reward term names must match the enum");
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
}

}  // namespace aibf
