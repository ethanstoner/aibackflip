#include "env/Env2D.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

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
    config.pushIntervalSteps = push["interval_steps"].integer(config.pushIntervalSteps);
    config.pushImpulseMin = push["impulse_min"].real(config.pushImpulseMin);
    config.pushImpulseMax = push["impulse_max"].real(config.pushImpulseMax);

    const Json& imitation = json["imitation"];
    if (imitation.isObject()) {
        ImitationSettings& settings = config.imitation;
        settings.motionPath = imitation["motion"].string(settings.motionPath);
        settings.enabled = imitation["enabled"].boolean(!settings.motionPath.empty());
        settings.referenceStateInit =
            imitation["reference_state_init"].boolean(settings.referenceStateInit);
        settings.earlyTerminationPoseError =
            imitation["early_termination_pose_error"].real(settings.earlyTerminationPoseError);
        settings.earlyTerminationRootError =
            imitation["early_termination_root_error"].real(settings.earlyTerminationRootError);
        settings.endEpisodeAtMotionEnd =
            imitation["end_episode_at_motion_end"].boolean(settings.endEpisodeAtMotionEnd);

        const Json& scales = imitation["scales"];
        settings.scales.pose = scales["pose"].real(settings.scales.pose);
        settings.scales.jointVelocity =
            scales["joint_velocity"].real(settings.scales.jointVelocity);
        settings.scales.endEffector = scales["end_effector"].real(settings.scales.endEffector);
        settings.scales.root = scales["root"].real(settings.scales.root);
        settings.scales.com = scales["com"].real(settings.scales.com);
        settings.scales.rootAngleWeight =
            scales["root_angle_weight"].real(settings.scales.rootAngleWeight);
    }

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
    push.set("interval_steps", Json(pushIntervalSteps));
    push.set("impulse_min", Json(double(pushImpulseMin)));
    push.set("impulse_max", Json(double(pushImpulseMax)));
    root.set("disturbance", std::move(push));

    Json imitationJson = Json::object();
    imitationJson.set("motion", Json(imitation.motionPath));
    imitationJson.set("enabled", Json(imitation.enabled));
    imitationJson.set("reference_state_init", Json(imitation.referenceStateInit));
    imitationJson.set("early_termination_pose_error",
                      Json(double(imitation.earlyTerminationPoseError)));
    imitationJson.set("early_termination_root_error",
                      Json(double(imitation.earlyTerminationRootError)));
    imitationJson.set("end_episode_at_motion_end", Json(imitation.endEpisodeAtMotionEnd));
    Json scalesJson = Json::object();
    scalesJson.set("pose", Json(double(imitation.scales.pose)));
    scalesJson.set("joint_velocity", Json(double(imitation.scales.jointVelocity)));
    scalesJson.set("end_effector", Json(double(imitation.scales.endEffector)));
    scalesJson.set("root", Json(double(imitation.scales.root)));
    scalesJson.set("com", Json(double(imitation.scales.com)));
    scalesJson.set("root_angle_weight", Json(double(imitation.scales.rootAngleWeight)));
    imitationJson.set("scales", std::move(scalesJson));
    root.set("imitation", std::move(imitationJson));

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

    motionLoaded_ = false;
    if (config_.imitation.enabled && !config_.imitation.motionPath.empty()) {
        std::string error;
        motion_ = Motion2D::loadFile(config_.imitation.motionPath, &error);
        motionLoaded_ = error.empty() && !motion_.empty();
        if (!motionLoaded_) {
            // Loud rather than silently falling back to a non-imitation task,
            // which would look like the reward simply never learning.
            std::fprintf(stderr, "imitation motion '%s' could not be used: %s\n",
                         config_.imitation.motionPath.c_str(),
                         error.empty() ? "clip is empty" : error.c_str());
        } else {
            const int clamped = motion_.clampToLimits(config_.humanoid);
            if (clamped > 0) {
                std::fprintf(stderr,
                             "note: %d joint angles in '%s' were outside the humanoid's limits "
                             "and have been clamped\n",
                             clamped, motion_.name.c_str());
            }
        }
    }

    reset();
}

void Env2D::reset() {
    // Reference State Initialization. Starting at a random phase is what makes
    // the back half of an acrobatic motion learnable at all.
    Real startPhase = 0;
    if (motionLoaded_ && config_.imitation.referenceStateInit) {
        startPhase = rng_.uniform(Real(0), Real(1));
    }

    if (motionLoaded_) {
        applyReferenceStateInit(startPhase);
    } else {
        figure_.reset(world_, config_.spawnPosition, config_.resetNoise, rng_);
    }

    figure_.setMotorsEnabled(world_, true);
    std::fill(lastActions_.begin(), lastActions_.end(), Real(0));
    episodeStep_ = 0;
    terminated_ = false;
    truncated_ = false;
    terminationReason_ = "";
    phase_ = startPhase;
    pushCount_ = 0;
    motionTime_ = motionLoaded_ ? motion_.timeAt(startPhase) : Real(0);

    // Populate contact state so the first observation does not report "nothing
    // is touching anything". This runs collision detection only: stepping the
    // world here would advance the physics by a substep, and the motors'
    // damping term would brake the velocities that reference state
    // initialization had just set - which for an episode starting mid-motion
    // discards the entire point of starting it there.
    world_.refreshContacts();
    updateImitationTargets();
}

void Env2D::applyReferenceStateInit(Real phase) {
    const MotionPose pose = motion_.samplePhase(phase);
    const MotionPose velocity = motion_.samplePhaseVelocity(phase);

    // Absolute X is invisible to the policy and irrelevant to the reward, so
    // the figure always starts at the spawn column regardless of where the clip
    // says the root is.
    const Vec2 rootPosition(config_.spawnPosition.x, pose.rootPosition.y);

    std::vector<Real> angles = pose.jointAngles;
    angles.resize(static_cast<size_t>(kJointCount), Real(0));
    std::vector<Real> jointVelocities = velocity.jointAngles;
    jointVelocities.resize(static_cast<size_t>(kJointCount), Real(0));

    // Perturbation on top of the reference, so the policy sees a spread of
    // states around the motion rather than exactly the motion.
    const ResetNoise& noise = config_.resetNoise;
    for (size_t j = 0; j < angles.size(); ++j) {
        const JointConfig& jc = config_.humanoid.joints[j];
        angles[j] = clamp(angles[j] + rng_.uniform(-noise.jointAngle, noise.jointAngle),
                          jc.lowerLimit, jc.upperLimit);
    }

    figure_.setPoseAndVelocity(
        world_, rootPosition + Vec2(0, rng_.uniform(-noise.rootHeight, noise.rootHeight)),
        pose.rootAngle + rng_.uniform(-noise.rootAngle, noise.rootAngle), angles.data(),
        velocity.rootPosition, velocity.rootAngle, jointVelocities.data());

    for (int j = 0; j < kJointCount; ++j) {
        RevoluteJoint2D& joint = figure_.joint(world_, j);
        joint.targetAngle = angles[static_cast<size_t>(j)];
        joint.resetAccumulators();
    }
    world_.clearContactCache();
    world_.resetStats();
}

void Env2D::updateImitationTargets() {
    targets_.valid = false;
    if (!motionLoaded_) return;

    const MotionPose pose = motion_.sample(motionTime_);
    const MotionPose velocity = motion_.sampleVelocity(motionTime_);

    targets_.jointAngles = pose.jointAngles;
    targets_.jointAngles.resize(static_cast<size_t>(kJointCount), Real(0));
    targets_.jointVelocities = velocity.jointAngles;
    targets_.jointVelocities.resize(static_cast<size_t>(kJointCount), Real(0));
    targets_.rootHeight = pose.rootPosition.y;
    targets_.rootAngle = pose.rootAngle;

    // End-effector and centre-of-mass targets come from forward kinematics on
    // the reference pose, expressed relative to the root.
    std::vector<Vec2> linkPositions(config_.humanoid.links.size());
    std::vector<Real> linkAngles(config_.humanoid.links.size());
    figure_.forwardKinematics(Vec2(0, pose.rootPosition.y), pose.rootAngle,
                              targets_.jointAngles.data(), linkPositions.data(),
                              linkAngles.data());

    const Vec2 root = linkPositions[kPelvis];
    targets_.endEffectors.clear();
    for (const int link : kEndEffectors) {
        targets_.endEffectors.push_back(linkPositions[static_cast<size_t>(link)] - root);
    }

    Vec2 weighted(0, 0);
    Real mass = 0;
    for (size_t i = 0; i < config_.humanoid.links.size(); ++i) {
        weighted += linkPositions[i] * config_.humanoid.links[i].mass;
        mass += config_.humanoid.links[i].mass;
    }
    targets_.comOffset = mass > Real(0) ? (weighted / mass) - root : Vec2(0, 0);
    targets_.valid = true;
}

Real Env2D::poseError() const {
    return targets_.valid ? poseTrackingError(world_, figure_, targets_) : Real(0);
}

Real Env2D::rootError() const {
    if (!targets_.valid) return Real(0);
    const RigidBody2D& pelvis = figure_.link(world_, kPelvis);
    const Real heightError = pelvis.position.y - targets_.rootHeight;
    const Real angleError = wrapAngle(pelvis.angle - targets_.rootAngle);
    // Same combination the root reward term uses, so the two agree about what
    // "off the reference" means.
    return std::sqrt(heightError * heightError +
                     config_.imitation.scales.rootAngleWeight * angleError * angleError);
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

    if (motionLoaded_) {
        motionTime_ += Real(config_.substepsPerControl) * dt;
        phase_ = motion_.phaseAt(motionTime_);
        updateImitationTargets();
    }

    evaluateTermination();
}

void Env2D::maybeDisturb() {
    bool shove = false;
    Real direction = 0;

    if (config_.pushIntervalSteps > 0) {
        // Measurement schedule: exactly on the beat, alternating side, so a
        // survival rate is attributable to the impulse magnitude and not to how
        // many shoves happened to land.
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
    push(Vec2(magnitude * direction, 0));
    ++pushCount_;
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

    if (motionLoaded_) {
        // Posture tests are meaningless while imitating: halfway through a
        // backflip the figure is upside down at knee height, which every one of
        // them would call a failure. Deviation from the reference is the only
        // thing that means failure here.
        if (config_.imitation.earlyTerminationPoseError > Real(0) &&
            poseError() > config_.imitation.earlyTerminationPoseError) {
            terminated_ = true;
            terminationReason_ = "lost_the_motion";
            return;
        }
        // Checked separately from the joint pose, because joint angles are
        // root-relative and a figure on the ground can hold the reference pose
        // perfectly while doing nothing the motion describes.
        if (config_.imitation.earlyTerminationRootError > Real(0) &&
            rootError() > config_.imitation.earlyTerminationRootError) {
            terminated_ = true;
            terminationReason_ = "root_off_reference";
            return;
        }
        if (config_.imitation.endEpisodeAtMotionEnd && !motion_.loop &&
            motionTime_ >= motion_.duration()) {
            // Reaching the end of the clip is a success, so it truncates rather
            // than terminates and the value function bootstraps through it.
            truncated_ = true;
            terminationReason_ = "motion_complete";
            return;
        }
    } else {
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
    if (targets_.valid) {
        writeImitationTerms(world_, figure_, targets_, config_.imitation.scales, scales_, out);
    }
    // The alive term has to reflect the episode's actual state, not a constant.
    out[kTermAlive] = terminated_ ? Real(0) : Real(1);
}

}  // namespace aibf
