#include "env/Env2D.h"

#include <algorithm>
#include <cmath>

namespace aibf {

// ---------------------------------------------------------------- config

EnvConfig EnvConfig::fromJson(const Json& json) {
    EnvConfig config;
    if (!json.isObject()) return config;

    if (json.contains("humanoid")) config.humanoid = Humanoid2DConfig::fromJson(json["humanoid"]);

    config.physicsHz = json["physics_hz"].real(config.physicsHz);
    config.substepsPerControl = json["substeps_per_control"].integer(config.substepsPerControl);
    config.maxEpisodeSteps = json["max_episode_steps"].integer(config.maxEpisodeSteps);
    config.spawnPosition = json["spawn_position"].vec2(config.spawnPosition);

    const Json& noise = json["reset_noise"];
    config.resetNoise.rootAngle = noise["root_angle"].real(config.resetNoise.rootAngle);
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
    config.pushImpulseMin = push["impulse_min"].real(config.pushImpulseMin);
    config.pushImpulseMax = push["impulse_max"].real(config.pushImpulseMax);

    return config;
}

EnvConfig EnvConfig::loadFile(const std::string& path, std::string* error) {
    std::string parseError;
    const Json json = Json::parseFile(path, &parseError);
    if (!parseError.empty()) {
        if (error) *error = parseError;
        return defaults();
    }
    EnvConfig config = fromJson(json);
    if (error) *error = config.validate();
    return config;
}

Json EnvConfig::toJson() const {
    Json root = Json::object();
    root.set("humanoid", humanoid.toJson());
    root.set("physics_hz", Json(double(physicsHz)));
    root.set("substeps_per_control", Json(substepsPerControl));
    root.set("max_episode_steps", Json(maxEpisodeSteps));

    Json spawn = Json::array();
    spawn.push(Json(double(spawnPosition.x)));
    spawn.push(Json(double(spawnPosition.y)));
    root.set("spawn_position", std::move(spawn));

    Json noise = Json::object();
    noise.set("root_angle", Json(double(resetNoise.rootAngle)));
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
    push.set("impulse_min", Json(double(pushImpulseMin)));
    push.set("impulse_max", Json(double(pushImpulseMax)));
    root.set("disturbance", std::move(push));

    return root;
}

std::string EnvConfig::validate() const {
    const std::string humanoidProblem = humanoid.validate();
    if (!humanoidProblem.empty()) return "humanoid: " + humanoidProblem;
    if (!(physicsHz > Real(0))) return "physics_hz must be positive";
    if (substepsPerControl < 1) return "substeps_per_control must be at least 1";
    if (maxEpisodeSteps < 1) return "max_episode_steps must be at least 1";
    if (pushProbabilityPerStep < Real(0) || pushProbabilityPerStep > Real(1)) {
        return "disturbance probability_per_step must lie in [0, 1]";
    }
    if (pushImpulseMax < pushImpulseMin) return "disturbance impulse range is inverted";
    return std::string();
}

// ---------------------------------------------------------------- env

void Env2D::initialize(const EnvConfig& config, uint64_t seed) {
    config_ = config;
    rng_.seedWith(seed);
    world_ = World2D{};
    world_.addHalfSpace(HalfSpace{Vec2(0, 1), 0, Real(1.0), Real(0)});
    figure_.build(world_, config_.humanoid);
    figure_.setMotorsEnabled(world_, true);
    scales_ = ObservationScales::fromConfig(config_.humanoid);
    lastActions_.assign(static_cast<size_t>(kJointCount), Real(0));
    reset();
}

void Env2D::reset() {
    figure_.reset(world_, config_.spawnPosition, config_.resetNoise, rng_);
    figure_.setMotorsEnabled(world_, true);
    std::fill(lastActions_.begin(), lastActions_.end(), Real(0));
    episodeStep_ = 0;
    terminated_ = false;
    truncated_ = false;
    terminationReason_ = "";
    phase_ = 0;

    // One settle step so the first observation reports real contact state
    // rather than "nothing is touching anything", which it would otherwise do
    // because no collision pass has run since the figure was placed.
    world_.step(config_.physicsDt());
}

void Env2D::step(const Real* actions, int count) {
    if (done()) return;

    const int n = std::min(count, static_cast<int>(kJointCount));
    for (int j = 0; j < n; ++j) {
        lastActions_[static_cast<size_t>(j)] = clamp(actions[j], Real(-1), Real(1));
    }
    figure_.applyNormalizedActions(world_, lastActions_.data(), n);

    maybeDisturb();

    const Real dt = config_.physicsDt();
    for (int s = 0; s < config_.substepsPerControl; ++s) world_.step(dt);

    ++episodeStep_;
    evaluateTermination();
}

void Env2D::maybeDisturb() {
    if (config_.pushProbabilityPerStep <= Real(0)) return;
    if (!rng_.chance(config_.pushProbabilityPerStep)) return;
    const Real magnitude = rng_.uniform(config_.pushImpulseMin, config_.pushImpulseMax);
    const Real direction = rng_.chance(Real(0.5)) ? Real(1) : Real(-1);
    push(Vec2(magnitude * direction, 0));
}

void Env2D::push(const Vec2& impulse) {
    figure_.link(world_, kPelvis).applyImpulse(impulse, Vec2(0, 0));
}

void Env2D::evaluateTermination() {
    // Numerical failure first: once a body is non-finite every other test below
    // is meaningless, and the episode has to end rather than poison the batch.
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

    const Real pelvisRatio =
        figure_.link(world_, kPelvis).position.y / scales_.pelvisRestHeight;
    if (pelvisRatio < config_.terminatePelvisHeightRatio) {
        terminated_ = true;
        terminationReason_ = "pelvis_low";
        return;
    }

    const Real headRatio = figure_.headHeight(world_) / scales_.headRestHeight;
    if (headRatio < config_.terminateHeadHeightRatio) {
        terminated_ = true;
        terminationReason_ = "head_low";
        return;
    }

    if (figure_.uprightness(world_, kChest) < config_.terminateChestUprightBelow) {
        terminated_ = true;
        terminationReason_ = "chest_fallen";
        return;
    }

    // Truncation is kept distinct from termination: hitting the time limit is
    // not a failure, and PPO has to bootstrap the value function through it
    // rather than treating the return as ending there.
    if (episodeStep_ >= config_.maxEpisodeSteps) {
        truncated_ = true;
        terminationReason_ = "time_limit";
    }
}

void Env2D::writeObservation(Real* out) const {
    aibf::writeObservation(world_, figure_, scales_, phase_, out);
}

void Env2D::writeRewardTerms(Real* out) const {
    aibf::writeRewardTerms(world_, figure_, scales_, lastActions_.data(), config_.physicsDt(), out);
    // The alive term has to reflect the episode's actual state, not a constant.
    out[kTermAlive] = terminated_ ? Real(0) : Real(1);
}

}  // namespace aibf
