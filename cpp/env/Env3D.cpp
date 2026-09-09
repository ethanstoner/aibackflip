#include "env/Env3D.h"

#include <algorithm>
#include <cmath>

namespace aibf {

// ---------------------------------------------------------------- config

EnvConfig3D EnvConfig3D::fromJson(const Json& json) {
    EnvConfig3D config;
    if (!json.isObject()) return config;

    if (json.contains("humanoid")) config.humanoid = Humanoid3DConfig::fromJson(json["humanoid"]);

    config.physicsHz = json["physics_hz"].real(config.physicsHz);
    config.substepsPerControl = json["substeps_per_control"].integer(config.substepsPerControl);
    config.maxEpisodeSteps = json["max_episode_steps"].integer(config.maxEpisodeSteps);

    const Json& noise = json["reset_noise"];
    config.resetNoise.rootTilt = noise["root_tilt"].real(config.resetNoise.rootTilt);
    config.resetNoise.rootHeight = noise["root_height"].real(config.resetNoise.rootHeight);
    config.resetNoise.jointAngle = noise["joint_angle"].real(config.resetNoise.jointAngle);
    config.resetNoise.linearVelocity =
        noise["linear_velocity"].real(config.resetNoise.linearVelocity);
    config.resetNoise.angularVelocity =
        noise["angular_velocity"].real(config.resetNoise.angularVelocity);

    const Json& termination = json["termination"];
    config.terminatePelvisHeightRatio =
        termination["pelvis_height_ratio"].real(config.terminatePelvisHeightRatio);
    config.terminateHeadHeightRatio =
        termination["head_height_ratio"].real(config.terminateHeadHeightRatio);
    config.terminateChestUprightBelow =
        termination["chest_upright_below"].real(config.terminateChestUprightBelow);

    const Json& push = json["disturbance"];
    config.pushProbabilityPerStep =
        push["probability_per_step"].real(config.pushProbabilityPerStep);
    config.pushIntervalSteps = push["interval_steps"].integer(config.pushIntervalSteps);
    config.pushAtStep = push["at_step"].integer(config.pushAtStep);
    config.pushImpulseMin = push["impulse_min"].real(config.pushImpulseMin);
    config.pushImpulseMax = push["impulse_max"].real(config.pushImpulseMax);
    config.pushOffsetMax = push["offset_max"].real(config.pushOffsetMax);

    return config;
}

EnvConfig3D EnvConfig3D::loadFile(const std::string& path, std::string* error) {
    std::string parseError;
    const Json json = Json::parseFile(path, &parseError);
    if (!parseError.empty()) {
        if (error) *error = parseError;
        return defaults();
    }
    EnvConfig3D config = fromJson(json);
    if (error) *error = config.validate();
    return config;
}

Json EnvConfig3D::toJson() const {
    Json root = Json::object();
    root.set("humanoid", humanoid.toJson());
    root.set("physics_hz", Json(double(physicsHz)));
    root.set("substeps_per_control", Json(substepsPerControl));
    root.set("max_episode_steps", Json(maxEpisodeSteps));

    Json noise = Json::object();
    noise.set("root_tilt", Json(double(resetNoise.rootTilt)));
    noise.set("root_height", Json(double(resetNoise.rootHeight)));
    noise.set("joint_angle", Json(double(resetNoise.jointAngle)));
    noise.set("linear_velocity", Json(double(resetNoise.linearVelocity)));
    noise.set("angular_velocity", Json(double(resetNoise.angularVelocity)));
    root.set("reset_noise", std::move(noise));

    Json termination = Json::object();
    termination.set("pelvis_height_ratio", Json(double(terminatePelvisHeightRatio)));
    termination.set("head_height_ratio", Json(double(terminateHeadHeightRatio)));
    termination.set("chest_upright_below", Json(double(terminateChestUprightBelow)));
    root.set("termination", std::move(termination));

    Json push = Json::object();
    push.set("probability_per_step", Json(double(pushProbabilityPerStep)));
    push.set("interval_steps", Json(pushIntervalSteps));
    push.set("at_step", Json(pushAtStep));
    push.set("impulse_min", Json(double(pushImpulseMin)));
    push.set("impulse_max", Json(double(pushImpulseMax)));
    push.set("offset_max", Json(double(pushOffsetMax)));
    root.set("disturbance", std::move(push));

    return root;
}

std::string EnvConfig3D::validate() const {
    const std::string humanoidProblem = humanoid.validate();
    if (!humanoidProblem.empty()) return "humanoid: " + humanoidProblem;
    if (!(physicsHz > Real(0))) return "physics_hz must be positive";
    if (substepsPerControl < 1) return "substeps_per_control must be at least 1";
    if (maxEpisodeSteps < 1) return "max_episode_steps must be at least 1";
    if (pushProbabilityPerStep < Real(0) || pushProbabilityPerStep > Real(1)) {
        return "disturbance probability_per_step must lie in [0, 1]";
    }
    if (pushImpulseMax < pushImpulseMin) return "disturbance impulse range is inverted";
    // Env3D reports a fixed action dimension because the joint kinds are fixed
    // by the enum. A config that disagrees would silently misalign every action
    // after the first differing joint, which is the kind of failure that looks
    // like bad training rather than a bad config.
    if (humanoid.actionDim() != 6 * kBallJointDof + 6 * kHingeJointDof) {
        return "humanoid joint kinds do not match the fixed action layout";
    }
    return std::string();
}

// ---------------------------------------------------------------- env

void Env3D::initialize(const EnvConfig3D& config, uint64_t seed) {
    config_ = config;
    rng_ = Rng(seed);

    world_.clear();
    world_.addHalfSpace(HalfSpace3D{Vec3(0, 1, 0), Real(0), Real(1.0), Real(0)});
    figure_.build(world_, config_.humanoid, config_.spawnPosition);

    scales_.height = config_.humanoid.restHeight();
    lastActions_.assign(static_cast<size_t>(actionDim()), Real(0));

    reset();
}

void Env3D::reset() {
    episodeStep_ = 0;
    terminated_ = false;
    truncated_ = false;
    terminationReason_ = "";
    phase_ = 0;
    pushCount_ = 0;
    std::fill(lastActions_.begin(), lastActions_.end(), Real(0));

    // A small random tilt about a random *horizontal* axis. In 2D there was only
    // one axis to tilt about; here the figure can be nudged forwards, backwards
    // or sideways, and sideways is the direction it has no ankle roll to answer
    // with, so it is the interesting one.
    const Real tiltAngle = rng_.uniform(-config_.resetNoise.rootTilt, config_.resetNoise.rootTilt);
    const Real tiltDirection = rng_.uniform(Real(0), Real(2) * kPi);
    const Vec3 tiltAxis(std::cos(tiltDirection), 0, std::sin(tiltDirection));
    const Quat rootOrientation = Quat::fromAxisAngle(tiltAxis, tiltAngle);

    Vec3 spawn = config_.spawnPosition;
    spawn.y += rng_.uniform(-config_.resetNoise.rootHeight, config_.resetNoise.rootHeight);
    figure_.setPose(world_, spawn, rootOrientation);

    // Drop the figure so its lowest point rests on the ground, whatever the
    // tilt did to it.
    const Real lowest = figure_.lowestPoint(world_);
    for (int i = 0; i < kLinkCount; ++i) figure_.link(world_, i).position.y -= lowest;

    for (int i = 0; i < kLinkCount; ++i) {
        RigidBody3D& body = figure_.link(world_, i);
        body.velocity = Vec3(rng_.uniform(-1, 1), rng_.uniform(-1, 1), rng_.uniform(-1, 1)) *
                        config_.resetNoise.linearVelocity;
        body.angularVelocity =
            Vec3(rng_.uniform(-1, 1), rng_.uniform(-1, 1), rng_.uniform(-1, 1)) *
            config_.resetNoise.angularVelocity;
        body.clearForces();
        body.refreshInertiaWorld();
    }

    figure_.relaxToRestPose(world_);
    world_.clearContactCache();
    world_.resetStats();
    // Contacts have to exist before the first observation is read, and stepping
    // the world to get them would silently advance the physics and let the
    // motors' damping brake the velocities just set.
    world_.refreshContacts();
}

void Env3D::step(const Real* actions, int count) {
    if (done()) return;

    if (actions != nullptr) {
        const int n = std::min(count, static_cast<int>(lastActions_.size()));
        for (int i = 0; i < n; ++i) lastActions_[static_cast<size_t>(i)] = actions[i];
        figure_.setJointTargetsNormalized(world_, actions, count);
    }

    maybeDisturb();

    const Real dt = config_.physicsDt();
    for (int i = 0; i < config_.substepsPerControl; ++i) world_.step(dt);

    ++episodeStep_;
    evaluateTermination();
}

void Env3D::writeObservation(Real* out) const {
    writeObservation3D(world_, figure_, scales_, phase_, out);
}

void Env3D::writeRewardTerms(Real* out) const {
    writeRewardTerms3D(world_, figure_, scales_, lastActions_.data(),
                       static_cast<int>(lastActions_.size()), !terminated_, out);
}

void Env3D::evaluateTermination() {
    if (world_.stats().unstable) {
        terminated_ = true;
        terminationReason_ = "unstable";
        return;
    }
    for (int i = 0; i < kLinkCount; ++i) {
        if (!figure_.link(world_, i).isFinite()) {
            terminated_ = true;
            terminationReason_ = "non_finite";
            return;
        }
    }

    const Humanoid3DConfig& cfg = config_.humanoid;
    const Real pelvisHeight = figure_.link(world_, kPelvis).position.y;
    if (pelvisHeight < cfg.links[kPelvis].restPosition.y * config_.terminatePelvisHeightRatio) {
        terminated_ = true;
        terminationReason_ = "pelvis_low";
        return;
    }

    const Real headHeight = figure_.link(world_, kHead).position.y;
    if (headHeight < cfg.links[kHead].restPosition.y * config_.terminateHeadHeightRatio) {
        terminated_ = true;
        terminationReason_ = "head_low";
        return;
    }

    const RigidBody3D& chest = figure_.link(world_, kChest);
    const Vec3 chestUp = chest.localToWorldDir(Vec3(0, 1, 0));
    if (chestUp.y < config_.terminateChestUprightBelow) {
        terminated_ = true;
        terminationReason_ = "chest_fallen";
        return;
    }

    if (episodeStep_ >= config_.maxEpisodeSteps) {
        truncated_ = true;
        terminationReason_ = "time_limit";
    }
}

void Env3D::maybeDisturb() {
    bool shove = false;
    Real direction = 0;

    if (config_.pushAtStep > 0) {
        if (episodeStep_ == config_.pushAtStep) {
            shove = true;
            direction = (pushCount_ % 2 == 0) ? Real(1) : Real(-1);
        }
    } else if (config_.pushIntervalSteps > 0) {
        if (episodeStep_ > 0 && episodeStep_ % config_.pushIntervalSteps == 0) {
            shove = true;
            direction = (pushCount_ % 2 == 0) ? Real(1) : Real(-1);
        }
    } else if (config_.pushProbabilityPerStep > Real(0) &&
               rng_.chance(config_.pushProbabilityPerStep)) {
        shove = true;
        direction = rng_.chance(Real(0.5)) ? Real(1) : Real(-1);
    }

    if (!shove) return;
    const Real magnitude = rng_.uniform(config_.pushImpulseMin, config_.pushImpulseMax);
    const Real offset = config_.pushOffsetMax > Real(0)
                            ? rng_.uniform(-config_.pushOffsetMax, config_.pushOffsetMax)
                            : Real(0);
    // A 3D shove can come from any direction in the horizontal plane, not just
    // along one axis. Sideways is the interesting case: the figure has no ankle
    // roll, so it has to answer with its hips.
    const Real bearing = rng_.uniform(Real(0), Real(2) * kPi);
    const Vec3 impulse(std::cos(bearing) * magnitude * direction, 0,
                       std::sin(bearing) * magnitude * direction);
    push(impulse, offset);
    ++pushCount_;
}

void Env3D::push(const Vec3& impulse, Real offset) {
    RigidBody3D& pelvis = figure_.link(world_, kPelvis);
    const Vec3 arm = rotate(pelvis.orientation, Vec3(0, offset, 0));
    pelvis.applyImpulse(impulse, arm);
}

}  // namespace aibf
