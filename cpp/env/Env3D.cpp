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

    const Json& imitation = json["imitation"];
    if (imitation.isObject()) {
        ImitationSettings3D& settings = config.imitation;
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

    scales_ = ObservationScales3D::fromConfig(config_.humanoid);
    lastActions_.assign(static_cast<size_t>(actionDim()), Real(0));

    fkPositions_.assign(kLinkCount, Vec3(0, 0, 0));
    fkOrientations_.assign(kLinkCount, Quat::identity());
    fkVelocities_.assign(kLinkCount, Vec3(0, 0, 0));
    fkAngularVelocities_.assign(kLinkCount, Vec3(0, 0, 0));
    fkJointRates_.assign(kJointCount, Vec3(0, 0, 0));

    motionLoaded_ = false;
    if (config_.imitation.enabled && !config_.imitation.motionPath.empty()) {
        std::string error;
        motion_ = Motion3D::readFile(config_.imitation.motionPath, &error);
        // A missing or broken clip disables imitation rather than silently
        // scoring against an empty reference, which would make every tracking
        // term report a plausible number for a motion that does not exist.
        motionLoaded_ = error.empty() && !motion_.keyframes.empty();
        if (!motionLoaded_) {
            std::fprintf(stderr, "imitation disabled, could not load %s: %s\n",
                         config_.imitation.motionPath.c_str(),
                         error.empty() ? "empty clip" : error.c_str());
        } else {
            motion_.clampToLimits(config_.humanoid);
        }
    }
    targets_.jointRotations.assign(kJointCount, Quat::identity());
    targets_.jointRates.assign(kJointCount, Vec3(0, 0, 0));
    targets_.endEffectors.assign(sizeof(kEndEffectors) / sizeof(kEndEffectors[0]),
                                 Vec3(0, 0, 0));

    reset();
}

void Env3D::reset() {
    episodeStep_ = 0;
    terminated_ = false;
    truncated_ = false;
    terminationReason_ = "";
    phase_ = 0;
    motionTime_ = 0;
    pushCount_ = 0;
    std::fill(lastActions_.begin(), lastActions_.end(), Real(0));

    if (motionLoaded_) {
        const Real startPhase =
            config_.imitation.referenceStateInit ? rng_.uniform(Real(0), Real(1)) : Real(0);
        applyReferenceStateInit(startPhase);
        figure_.relaxToRestPose(world_);
        world_.clearContactCache();
        world_.resetStats();
        updateImitationTargets();
        world_.refreshContacts();
        return;
    }

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

    if (motionLoaded_) {
        motionTime_ += Real(1) / config_.controlHz();
        const Real duration = motion_.duration();
        if (duration > Real(0)) {
            phase_ = motion_.loop ? std::fmod(motionTime_ / duration, Real(1))
                                  : clamp(motionTime_ / duration, Real(0), Real(1));
        }
        updateImitationTargets();
    }

    evaluateTermination();
}

void Env3D::writeObservation(Real* out) const {
    writeObservation3D(world_, figure_, scales_, phase_, out);
}

void Env3D::writeRewardTerms(Real* out) const {
    writeRewardTerms3D(world_, figure_, scales_, lastActions_.data(),
                       static_cast<int>(lastActions_.size()), !terminated_, out);
    // Written second, because writeRewardTerms3D zeroes the whole vector first.
    writeImitationTerms3D(world_, figure_, targets_, config_.imitation.scales, out);
}

Real Env3D::poseError() const { return poseError3D(world_, figure_, targets_); }

Real Env3D::rootError() const {
    return rootError3D(world_, figure_, targets_, config_.imitation.scales.rootAngleWeight);
}

void Env3D::updateImitationTargets() {
    targets_.valid = false;
    if (!motionLoaded_) return;

    const MotionPose3D pose = motion_.sample(motionTime_);
    Vec3 rootVelocity, rootSpin;
    motion_.sampleVelocity(motionTime_, rootVelocity, rootSpin, fkJointRates_);

    figure_.forwardKinematics(world_, pose.rootPosition, pose.rootOrientation,
                              pose.jointRotations.data(), fkPositions_.data(),
                              fkOrientations_.data());

    targets_.jointRotations = pose.jointRotations;
    targets_.jointRates = fkJointRates_;
    targets_.rootHeight = pose.rootPosition.y;
    targets_.rootOrientation = pose.rootOrientation;

    // Relative to the root, so the term measures limb configuration rather than
    // where the figure happens to be.
    const size_t effectors = sizeof(kEndEffectors) / sizeof(kEndEffectors[0]);
    targets_.endEffectors.resize(effectors);
    for (size_t e = 0; e < effectors; ++e) {
        targets_.endEffectors[e] = fkPositions_[kEndEffectors[e]] - pose.rootPosition;
    }

    Vec3 weighted(0, 0, 0);
    Real totalMass = 0;
    for (int i = 0; i < kLinkCount; ++i) {
        const Real mass = config_.humanoid.links[static_cast<size_t>(i)].mass;
        weighted += fkPositions_[i] * mass;
        totalMass += mass;
    }
    targets_.comOffset =
        (totalMass > Real(0) ? weighted / totalMass : pose.rootPosition) - pose.rootPosition;

    targets_.valid = true;
}

void Env3D::applyReferenceStateInit(Real phase) {
    const Real duration = motion_.duration();
    motionTime_ = phase * duration;
    phase_ = phase;

    const MotionPose3D pose = motion_.sample(motionTime_);
    Vec3 rootVelocity, rootSpin;
    motion_.sampleVelocity(motionTime_, rootVelocity, rootSpin, fkJointRates_);

    figure_.forwardKinematics(world_, pose.rootPosition, pose.rootOrientation,
                              pose.jointRotations.data(), fkPositions_.data(),
                              fkOrientations_.data());
    figure_.forwardKinematicsVelocity(world_, rootVelocity, rootSpin, fkJointRates_.data(),
                                      fkPositions_.data(), fkOrientations_.data(),
                                      fkVelocities_.data(), fkAngularVelocities_.data());

    // Small noise on top, so a policy cannot memorise a finite set of starting
    // states. The reference pose itself is exact; only the perturbation is
    // random.
    for (int i = 0; i < kLinkCount; ++i) {
        fkVelocities_[i] +=
            Vec3(rng_.uniform(-1, 1), rng_.uniform(-1, 1), rng_.uniform(-1, 1)) *
            config_.resetNoise.linearVelocity;
        fkAngularVelocities_[i] +=
            Vec3(rng_.uniform(-1, 1), rng_.uniform(-1, 1), rng_.uniform(-1, 1)) *
            config_.resetNoise.angularVelocity;
    }

    figure_.applyPose(world_, fkPositions_.data(), fkOrientations_.data(), fkVelocities_.data(),
                      fkAngularVelocities_.data());
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

    // While imitating, the reference decides what counts as failure. The
    // postural tests are switched off: a backflip spends half its time upside
    // down with the head near the floor, which every one of them would call a
    // fall.
    if (motionLoaded_ && targets_.valid) {
        const ImitationSettings3D& settings = config_.imitation;
        if (settings.earlyTerminationPoseError > Real(0) &&
            poseError() > settings.earlyTerminationPoseError) {
            terminated_ = true;
            terminationReason_ = "lost_the_motion";
            return;
        }
        if (settings.earlyTerminationRootError > Real(0) &&
            rootError() > settings.earlyTerminationRootError) {
            terminated_ = true;
            terminationReason_ = "root_off_reference";
            return;
        }
        if (settings.endEpisodeAtMotionEnd && !motion_.loop &&
            motionTime_ >= motion_.duration()) {
            // Finishing the clip is a success, so it is truncation.
            truncated_ = true;
            terminationReason_ = "motion_complete";
            return;
        }
        if (episodeStep_ >= config_.maxEpisodeSteps) {
            truncated_ = true;
            terminationReason_ = "time_limit";
        }
        return;
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
