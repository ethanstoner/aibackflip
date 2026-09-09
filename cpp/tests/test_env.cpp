#include <cmath>
#include <filesystem>
#include <system_error>
#include <vector>

#include "core/Test.h"
#include "env/EnvBatch.h"
#include "net/EnvServer.h"

using namespace aibf;

namespace {

EnvConfig quickConfig(int maxSteps = 200) {
    EnvConfig config;
    config.maxEpisodeSteps = maxSteps;
    config.resetNoise.rootAngle = Real(0.05);
    config.resetNoise.jointAngle = Real(0.1);
    return config;
}

std::vector<Real> zeroActions(int count = 1) {
    return std::vector<Real>(static_cast<size_t>(count * Env2D::actionDim()), Real(0));
}

}  // namespace

// ---------------------------------------------------------------- config

TEST(EnvConfig, aDefaultConstructedConfigIsUsable) {
    // This is not a formality. Humanoid2DConfig's own default state has empty
    // link and joint vectors, so an EnvConfig that merely default-constructed
    // its humanoid produced a figure with no bodies, and the first index of
    // kPelvis walked off the end of an empty vector.
    const EnvConfig config;
    CHECK(config.validate().empty());
    CHECK(config.humanoid.links.size() == static_cast<size_t>(kLinkCount));
    CHECK(config.humanoid.joints.size() == static_cast<size_t>(kJointCount));
    CHECK(EnvConfig::defaults().validate().empty());
}

TEST(EnvConfig, roundTripsThroughJson) {
    EnvConfig original;
    original.maxEpisodeSteps = 777;
    original.substepsPerControl = 3;
    original.resetNoise.jointAngle = Real(0.33);
    original.pushProbabilityPerStep = Real(0.01);
    original.pushImpulseMax = Real(45);
    original.terminateChestUprightBelow = Real(0.4);

    std::string error;
    const Json json = Json::parse(original.toJson().dump(), &error);
    CHECK(error.empty());
    const EnvConfig restored = EnvConfig::fromJson(json);

    CHECK(restored.validate().empty());
    CHECK(restored.maxEpisodeSteps == 777);
    CHECK(restored.substepsPerControl == 3);
    CHECK_NEAR(restored.resetNoise.jointAngle, 0.33, 1e-5);
    CHECK_NEAR(restored.pushProbabilityPerStep, 0.01, 1e-6);
    CHECK_NEAR(restored.pushImpulseMax, 45.0, 1e-4);
    CHECK_NEAR(restored.terminateChestUprightBelow, 0.4, 1e-5);
    CHECK_NEAR(restored.humanoid.totalMass(), original.humanoid.totalMass(), 1e-3);
    CHECK_NEAR(restored.controlHz(), original.controlHz(), 1e-4);
}

TEST(EnvConfig, validationRejectsNonsense) {
    {
        EnvConfig config;
        config.substepsPerControl = 0;
        CHECK(!config.validate().empty());
    }
    {
        EnvConfig config;
        config.physicsHz = 0;
        CHECK(!config.validate().empty());
    }
    {
        EnvConfig config;
        config.pushImpulseMin = 50;
        config.pushImpulseMax = 10;
        CHECK(!config.validate().empty());
    }
    {
        EnvConfig config;
        config.humanoid.links[kChest].mass = 0;
        CHECK(!config.validate().empty());
    }
}

// ---------------------------------------------------------------- observations

TEST(Observation, theLayoutIsSelfConsistent) {
    // Names and dimension have to agree, or the SPEC handshake sends a map that
    // does not describe the vector it accompanies.
    const std::vector<std::string> names = ObservationLayout::fieldNames();
    CHECK(names.size() == static_cast<size_t>(ObservationLayout::kDimension));
    for (const std::string& name : names) CHECK(!name.empty());
}

TEST(Observation, everyValueIsFiniteInAndOutOfContact) {
    Env2D env;
    env.initialize(quickConfig(), 5);
    std::vector<Real> obs(Env2D::observationDim());

    env.writeObservation(obs.data());
    for (const Real v : obs) CHECK(std::isfinite(v));

    // Airborne, so nothing is touching and the support-centre logic has no data.
    env.figure().setPose(env.world(), Vec2(0, Real(3)), Real(0.7), nullptr);
    env.writeObservation(obs.data());
    for (const Real v : obs) CHECK(std::isfinite(v));
}

TEST(Observation, absoluteHorizontalPositionIsInvisible) {
    // The task is identical wherever the figure stands, so translating it must
    // not change the observation. If it does, the policy can learn to depend on
    // where the episode happened to spawn.
    //
    // The tolerance is derived rather than picked. Positions are float, so
    // computing an offset at x = 37.5 m loses about x*2^-23 of absolute
    // precision before the offset is divided by the body height - and that is
    // rounding, not a leak. Anything larger than a small multiple of that bound
    // means absolute position is genuinely reaching the observation.
    EnvConfig config = quickConfig();
    config.resetNoise = ResetNoise{};

    auto observationAt = [](const EnvConfig& cfg, Real x) {
        EnvConfig shifted = cfg;
        shifted.spawnPosition = Vec2(x, cfg.spawnPosition.y);
        Env2D env;
        env.initialize(shifted, 3);
        std::vector<Real> obs(Env2D::observationDim());
        env.writeObservation(obs.data());
        return obs;
    };

    const std::vector<Real> atOrigin = observationAt(config, Real(0));
    for (const Real x : {Real(5), Real(37.5), Real(200)}) {
        const std::vector<Real> shifted = observationAt(config, x);
        const Real floatEpsilon = Real(1.1920929e-7);
        const Real bound = std::max(Real(1e-6), Real(32) * x * floatEpsilon);
        for (int i = 0; i < Env2D::observationDim(); ++i) {
            CHECK(std::abs(atOrigin[i] - shifted[i]) <= bound);
        }
    }
}

TEST(Observation, heightIsExpressedRelativeToTheRestPose) {
    Env2D env;
    EnvConfig config = quickConfig();
    config.resetNoise = ResetNoise{};
    env.initialize(config, 1);
    std::vector<Real> obs(Env2D::observationDim());
    env.writeObservation(obs.data());
    // Standing at the rest pose reads as 1.
    CHECK_NEAR(obs[ObservationLayout::kPelvisHeight], 1.0, 0.02);
    CHECK_NEAR(obs[ObservationLayout::kHeadHeight], 1.0, 0.02);
}

TEST(Observation, orientationIsEncodedOnTheCircle) {
    Env2D env;
    env.initialize(quickConfig(), 1);
    std::vector<Real> obs(Env2D::observationDim());

    for (const Real angle : {Real(0), Real(1.0), Real(3.0), Real(-3.0), kPi}) {
        env.figure().setPose(env.world(), Vec2(0, 1), angle, nullptr);
        env.writeObservation(obs.data());
        const Real s = obs[ObservationLayout::kPelvisOrientation];
        const Real c = obs[ObservationLayout::kPelvisOrientation + 1];
        CHECK_NEAR(s * s + c * c, 1.0, 1e-4);
        CHECK_NEAR(std::atan2(s, c), wrapAngle(angle), 1e-4);
    }
}

TEST(Observation, phaseWrapsContinuously) {
    Env2D env;
    env.initialize(quickConfig(), 1);
    std::vector<Real> nearEnd(Env2D::observationDim());
    std::vector<Real> nearStart(Env2D::observationDim());

    env.setPhase(Real(0.999));
    env.writeObservation(nearEnd.data());
    env.setPhase(Real(0.001));
    env.writeObservation(nearStart.data());

    // Adjacent phases must be adjacent inputs; a raw scalar would jump by 1.
    const Real ds = nearEnd[ObservationLayout::kPhase] - nearStart[ObservationLayout::kPhase];
    const Real dc = nearEnd[ObservationLayout::kPhase + 1] - nearStart[ObservationLayout::kPhase + 1];
    CHECK(std::sqrt(ds * ds + dc * dc) < Real(0.02));
}

TEST(Observation, footContactFlagsTrackTheGround) {
    Env2D env;
    EnvConfig config = quickConfig();
    config.resetNoise = ResetNoise{};
    env.initialize(config, 1);
    std::vector<Real> obs(Env2D::observationDim());

    env.writeObservation(obs.data());
    CHECK_NEAR(obs[ObservationLayout::kFootContacts], 1.0, 1e-6);
    CHECK_NEAR(obs[ObservationLayout::kFootContacts + 1], 1.0, 1e-6);

    env.figure().setPose(env.world(), Vec2(0, Real(3)), 0, nullptr);
    env.world().step(Real(1) / Real(240));
    env.writeObservation(obs.data());
    CHECK_NEAR(obs[ObservationLayout::kFootContacts], 0.0, 1e-6);
    CHECK_NEAR(obs[ObservationLayout::kFootContacts + 1], 0.0, 1e-6);
}

// ---------------------------------------------------------------- rewards

TEST(RewardTerms, namesMatchTheEnum) {
    CHECK(rewardTermNames().size() == static_cast<size_t>(kTermCount));
    CHECK(rewardTermNames()[kTermAlive] == "alive");
    CHECK(rewardTermNames()[kTermTorqueCost] == "torque_cost");
}

TEST(RewardTerms, standingScoresHigherThanFallen) {
    // The single most important property of the reward: the state we want must
    // outscore the state we do not, on the terms that describe it.
    EnvConfig config = quickConfig();
    config.resetNoise = ResetNoise{};

    Env2D standing;
    standing.initialize(config, 1);
    std::vector<Real> upright(Env2D::rewardTermCount());
    standing.writeRewardTerms(upright.data());

    Env2D fallen;
    fallen.initialize(config, 1);
    fallen.figure().setMotorsEnabled(fallen.world(), false);
    for (int i = 0; i < 240 * 4; ++i) fallen.world().step(Real(1) / Real(240));
    std::vector<Real> collapsed(Env2D::rewardTermCount());
    fallen.writeRewardTerms(collapsed.data());

    CHECK(upright[kTermPelvisHeight] > collapsed[kTermPelvisHeight] + Real(0.3));
    CHECK(upright[kTermHeadHeight] > collapsed[kTermHeadHeight] + Real(0.3));
    CHECK(upright[kTermChestUpright] > collapsed[kTermChestUpright]);
}

TEST(RewardTerms, everyTermIsFiniteAndNonNegative) {
    Env2D env;
    env.initialize(quickConfig(), 11);
    std::vector<Real> terms(Env2D::rewardTermCount());
    Rng rng(4);
    std::vector<Real> actions(Env2D::actionDim());

    for (int step = 0; step < 600; ++step) {
        for (Real& a : actions) a = rng.uniform(Real(-1), Real(1));
        env.step(actions.data(), Env2D::actionDim());
        env.writeRewardTerms(terms.data());
        for (int t = 0; t < Env2D::rewardTermCount(); ++t) {
            CHECK(std::isfinite(terms[t]));
            CHECK(terms[t] >= Real(0));
        }
        if (env.done()) env.reset();
    }
}

TEST(RewardTerms, boundedTermsStayWithinTheirDocumentedRange) {
    Env2D env;
    env.initialize(quickConfig(), 12);
    std::vector<Real> terms(Env2D::rewardTermCount());
    Rng rng(8);
    std::vector<Real> actions(Env2D::actionDim());

    for (int step = 0; step < 900; ++step) {
        for (Real& a : actions) a = rng.uniform(Real(-2), Real(2));  // deliberately out of range
        env.step(actions.data(), Env2D::actionDim());
        env.writeRewardTerms(terms.data());
        CHECK(terms[kTermPelvisHeight] <= Real(1));
        CHECK(terms[kTermHeadHeight] <= Real(1));
        CHECK(terms[kTermChestUpright] <= Real(1));
        CHECK(terms[kTermHeadUpright] <= Real(1));
        CHECK(terms[kTermComOverSupport] <= Real(1));
        CHECK(terms[kTermFootContact] <= Real(1));
        CHECK(terms[kTermActionCost] <= Real(1));
        CHECK(terms[kTermTorqueCost] <= Real(1));
        if (env.done()) env.reset();
    }
}

TEST(RewardTerms, theAliveTermGoesToZeroOnlyOnTermination) {
    Env2D env;
    EnvConfig config = quickConfig(50);
    env.initialize(config, 2);
    std::vector<Real> terms(Env2D::rewardTermCount());
    env.writeRewardTerms(terms.data());
    CHECK_NEAR(terms[kTermAlive], 1.0, 1e-6);

    env.figure().setMotorsEnabled(env.world(), false);
    const std::vector<Real> actions = zeroActions();
    while (!env.done()) env.step(actions.data(), Env2D::actionDim());
    env.writeRewardTerms(terms.data());
    // Truncation is not death: only a real termination zeroes the alive bonus.
    CHECK_NEAR(terms[kTermAlive], env.terminated() ? 0.0 : 1.0, 1e-6);
}

// ---------------------------------------------------------------- episodes

TEST(Env, aFallingFigureTerminatesWithAReason) {
    Env2D env;
    env.initialize(quickConfig(2000), 21);
    env.figure().setMotorsEnabled(env.world(), false);
    const std::vector<Real> actions = zeroActions();

    while (!env.done()) env.step(actions.data(), Env2D::actionDim());

    CHECK(env.terminated());
    CHECK(!env.truncated());
    CHECK(std::string(env.terminationReason()) != "");
    CHECK(std::string(env.terminationReason()) != "time_limit");
    CHECK(env.episodeStep() < 2000);
}

TEST(Env, truncationIsDistinctFromTermination) {
    // PPO has to bootstrap through a time limit rather than treat it as a
    // failure, so the two cases cannot be collapsed into one flag.
    Env2D env;
    EnvConfig config = quickConfig(40);
    config.resetNoise = ResetNoise{};
    env.initialize(config, 1);
    const std::vector<Real> actions = zeroActions();

    while (!env.done()) env.step(actions.data(), Env2D::actionDim());

    CHECK(env.truncated());
    CHECK(!env.terminated());
    CHECK(env.episodeStep() == 40);
    CHECK(std::string(env.terminationReason()) == "time_limit");
}

TEST(Env, steppingAFinishedEpisodeDoesNothing) {
    Env2D env;
    env.initialize(quickConfig(20), 1);
    const std::vector<Real> actions = zeroActions();
    while (!env.done()) env.step(actions.data(), Env2D::actionDim());

    const int stepAtEnd = env.episodeStep();
    for (int i = 0; i < 10; ++i) env.step(actions.data(), Env2D::actionDim());
    CHECK(env.episodeStep() == stepAtEnd);
}

TEST(Env, resetClearsTerminationAndTheStepCounter) {
    Env2D env;
    env.initialize(quickConfig(30), 1);
    const std::vector<Real> actions = zeroActions();
    while (!env.done()) env.step(actions.data(), Env2D::actionDim());

    env.reset();
    CHECK(!env.done());
    CHECK(!env.terminated());
    CHECK(!env.truncated());
    CHECK(env.episodeStep() == 0);
}

TEST(Env, resetsDifferButAreReproducibleFromASeed) {
    auto firstObservation = [](uint64_t seed) {
        Env2D env;
        env.initialize(quickConfig(), seed);
        std::vector<Real> obs(Env2D::observationDim());
        env.writeObservation(obs.data());
        return obs;
    };
    const std::vector<Real> a = firstObservation(77);
    const std::vector<Real> b = firstObservation(77);
    const std::vector<Real> c = firstObservation(78);
    CHECK(a == b);
    CHECK(a != c);

    // Successive resets within one environment must also differ, or the reset
    // noise is not actually being consumed.
    Env2D env;
    env.initialize(quickConfig(), 5);
    std::vector<Real> first(Env2D::observationDim());
    std::vector<Real> second(Env2D::observationDim());
    env.writeObservation(first.data());
    env.reset();
    env.writeObservation(second.data());
    CHECK(first != second);
}

TEST(Env, aRandomPolicyNeverBreaksTheSimulation) {
    Env2D env;
    env.initialize(quickConfig(300), 33);
    Rng rng(33);
    std::vector<Real> actions(Env2D::actionDim());
    std::vector<Real> obs(Env2D::observationDim());
    int episodes = 0;

    for (int step = 0; step < 20000; ++step) {
        for (Real& a : actions) a = rng.uniform(Real(-1), Real(1));
        env.step(actions.data(), Env2D::actionDim());
        env.writeObservation(obs.data());
        for (const Real v : obs) CHECK(std::isfinite(v));
        if (env.done()) {
            ++episodes;
            CHECK(std::string(env.terminationReason()) != "unstable");
            CHECK(std::string(env.terminationReason()) != "non_finite");
            env.reset();
        }
    }
    CHECK(episodes > 10);  // it really did run many episodes, not one long one
}

// ---------------------------------------------------------------- imitation

namespace {

// A short clip that actually moves, so reference velocities are non-zero and a
// velocity comparison means something.
Motion2D testClip(const Humanoid2D& figure) {
    Motion2D motion;
    motion.name = "test_squat";
    motion.loop = true;
    for (int i = 0; i < 3; ++i) {
        MotionKeyframe key;
        key.time = Real(i) * Real(0.6);
        key.jointAngles.assign(kJointCount, Real(0));
        if (i == 1) {
            key.jointAngles[kHipL] = Real(0.9);
            key.jointAngles[kHipR] = Real(0.9);
            key.jointAngles[kKneeL] = Real(-1.5);
            key.jointAngles[kKneeR] = Real(-1.5);
            key.jointAngles[kAnkleL] = Real(0.4);
            key.jointAngles[kAnkleR] = Real(0.4);
        }
        key.rootPosition = Vec2(0, figure.groundedRootHeight(0, key.jointAngles.data()));
        motion.insertKeyframe(key);
    }
    return motion;
}

// Env2D loads a motion by path, so these have to live on disk. Written to the
// system temp directory rather than the working directory, so running the tests
// does not leave files wherever they happened to be run from.
std::string tempClipPath(const std::string& name) {
    std::error_code error;
    const std::filesystem::path directory = std::filesystem::temp_directory_path(error);
    return (error ? std::filesystem::path(".") : directory).string() + "/aibf_test_" + name +
           ".json";
}

std::string writeTestClip(const Humanoid2D& figure) {
    const std::string path = tempClipPath("clip");
    testClip(figure).writeFile(path);
    return path;
}

}  // namespace

TEST(Imitation, forwardKinematicsVelocityReproducesTheJointRates) {
    // The foundation reference state initialization stands on. If this is wrong,
    // an episode started mid-motion begins with velocities that do not belong to
    // the pose it was placed in.
    World2D world;
    Humanoid2D figure;
    figure.build(world, Humanoid2DConfig::defaults());

    Rng rng(5);
    Real angles[kJointCount];
    Real velocities[kJointCount];
    for (int j = 0; j < kJointCount; ++j) {
        const JointConfig& jc = Humanoid2DConfig::defaults().joints[static_cast<size_t>(j)];
        angles[j] = rng.uniform(jc.lowerLimit * Real(0.5), jc.upperLimit * Real(0.5));
        velocities[j] = rng.uniform(Real(-3), Real(3));
    }
    const Vec2 rootVelocity(Real(0.4), Real(-1.1));
    const Real rootAngularVelocity = Real(0.7);

    figure.setPoseAndVelocity(world, Vec2(0, Real(1.1)), Real(0.2), angles, rootVelocity,
                              rootAngularVelocity, velocities);

    CHECK_NEAR(figure.link(world, kPelvis).angularVelocity, rootAngularVelocity, 1e-5);
    CHECK_VEC2_NEAR(figure.link(world, kPelvis).velocity, rootVelocity.x, rootVelocity.y, 1e-5);
    for (int j = 0; j < kJointCount; ++j) {
        CHECK_NEAR(figure.jointVelocity(world, j), velocities[j], 1e-4);
        CHECK_NEAR(figure.jointAngle(world, j), angles[j], 1e-4);
    }
}

TEST(Imitation, resetLeavesTheFigureOnTheReferenceInBothPoseAndVelocity) {
    // This is the test that catches a reset which quietly advances the physics.
    // A settle step there is not harmless: the motors' damping term brakes the
    // velocities reference state initialization had just set, and an episode
    // starting mid-motion loses the entire point of starting there.
    World2D scratchWorld;
    Humanoid2D scratchFigure;
    scratchFigure.build(scratchWorld, Humanoid2DConfig::defaults());

    EnvConfig config;
    config.maxEpisodeSteps = 200;
    config.resetNoise = ResetNoise{};  // isolate the reference from the noise
    config.imitation.enabled = true;
    config.imitation.motionPath = writeTestClip(scratchFigure);
    config.imitation.referenceStateInit = true;

    Env2D env;
    env.initialize(config, 11);
    CHECK(env.hasMotion());

    Rng rng(3);
    for (int trial = 0; trial < 12; ++trial) {
        env.reset();
        const ImitationTargets& targets = env.imitationTargets();
        CHECK(targets.valid);

        // Pose lands on the reference.
        CHECK(env.poseError() < Real(0.02));

        // And so do the velocities, which is the part a settle step destroys.
        Real worstVelocityError = 0;
        for (int j = 0; j < kJointCount; ++j) {
            const Real error = std::abs(env.figure().jointVelocity(env.world(), j) -
                                        targets.jointVelocities[static_cast<size_t>(j)]);
            worstVelocityError = std::max(worstVelocityError, error);
        }
        CHECK(worstVelocityError < Real(0.05));
        (void)rng;
    }
}

TEST(Imitation, referenceStateInitStartsEpisodesAtDifferentPhases) {
    World2D scratchWorld;
    Humanoid2D scratchFigure;
    scratchFigure.build(scratchWorld, Humanoid2DConfig::defaults());

    EnvConfig config;
    config.imitation.enabled = true;
    config.imitation.motionPath = writeTestClip(scratchFigure);
    config.imitation.referenceStateInit = true;

    Env2D env;
    env.initialize(config, 7);
    Real lowest = Real(1), highest = Real(0);
    for (int i = 0; i < 40; ++i) {
        env.reset();
        lowest = std::min(lowest, env.phase());
        highest = std::max(highest, env.phase());
    }
    // Without this spread, the back half of an acrobatic motion never gets a
    // gradient until the front half is already solved.
    CHECK(lowest < Real(0.2));
    CHECK(highest > Real(0.8));

    // Turning it off pins every episode to the start.
    config.imitation.referenceStateInit = false;
    Env2D fixed;
    fixed.initialize(config, 7);
    for (int i = 0; i < 5; ++i) {
        fixed.reset();
        CHECK_NEAR(fixed.phase(), 0.0, 1e-6);
    }
}

TEST(Imitation, trackingTermsFallAsTheFigureDriftsFromTheReference) {
    World2D scratchWorld;
    Humanoid2D scratchFigure;
    scratchFigure.build(scratchWorld, Humanoid2DConfig::defaults());

    EnvConfig config;
    config.maxEpisodeSteps = 400;
    config.resetNoise = ResetNoise{};
    config.imitation.enabled = true;
    config.imitation.motionPath = writeTestClip(scratchFigure);
    config.imitation.earlyTerminationPoseError = 0;  // let it drift freely

    Env2D env;
    env.initialize(config, 2);
    env.reset();

    std::vector<Real> onReference(Env2D::rewardTermCount());
    env.writeRewardTerms(onReference.data());
    CHECK(onReference[kTermPoseMatch] > Real(0.9));
    CHECK(onReference[kTermRootMatch] > Real(0.9));
    CHECK(onReference[kTermComMatch] > Real(0.9));
    CHECK(onReference[kTermEndEffectorMatch] > Real(0.7));

    // A zero action commands the rest pose, so the figure abandons the squat.
    const std::vector<Real> zeros(Env2D::actionDim(), Real(0));
    for (int i = 0; i < 40; ++i) env.step(zeros.data(), Env2D::actionDim());

    std::vector<Real> drifted(Env2D::rewardTermCount());
    env.writeRewardTerms(drifted.data());
    CHECK(drifted[kTermPoseMatch] < onReference[kTermPoseMatch] - Real(0.2));
    CHECK(env.poseError() > Real(0.1));
}

TEST(Imitation, aLostMotionTerminatesAndFinishingItTruncates) {
    World2D scratchWorld;
    Humanoid2D scratchFigure;
    scratchFigure.build(scratchWorld, Humanoid2DConfig::defaults());

    // Losing the motion is a failure.
    {
        EnvConfig config;
        config.maxEpisodeSteps = 400;
        config.resetNoise = ResetNoise{};
        config.imitation.enabled = true;
        config.imitation.motionPath = writeTestClip(scratchFigure);
        config.imitation.referenceStateInit = false;
        config.imitation.earlyTerminationPoseError = Real(0.5);

        Env2D env;
        env.initialize(config, 4);
        const std::vector<Real> zeros(Env2D::actionDim(), Real(0));
        while (!env.done() && env.episodeStep() < 400) {
            env.step(zeros.data(), Env2D::actionDim());
        }
        CHECK(env.terminated());
        CHECK(std::string(env.terminationReason()) == "lost_the_motion");
    }

    // Reaching the end of a one-shot clip is a success, so it truncates and the
    // value function bootstraps through it rather than treating it as death.
    {
        Motion2D oneShot = testClip(scratchFigure);
        oneShot.loop = false;
        const std::string oneShotPath = tempClipPath("oneshot");
        oneShot.writeFile(oneShotPath);

        EnvConfig config;
        config.maxEpisodeSteps = 4000;
        config.resetNoise = ResetNoise{};
        config.imitation.enabled = true;
        config.imitation.motionPath = oneShotPath;
        config.imitation.referenceStateInit = false;
        config.imitation.earlyTerminationPoseError = 0;  // never lose it

        Env2D env;
        env.initialize(config, 4);
        const std::vector<Real> zeros(Env2D::actionDim(), Real(0));
        while (!env.done() && env.episodeStep() < 4000) {
            env.step(zeros.data(), Env2D::actionDim());
        }
        CHECK(env.truncated());
        CHECK(!env.terminated());
        CHECK(std::string(env.terminationReason()) == "motion_complete");
    }
}

TEST(Imitation, posturalTerminationsAreDisabledWhileImitating) {
    // Halfway through a backflip the figure is upside down at knee height, which
    // pelvis_low, head_low and chest_fallen would all call a failure. Only
    // deviation from the reference may end an imitation episode.
    World2D scratchWorld;
    Humanoid2D scratchFigure;
    scratchFigure.build(scratchWorld, Humanoid2DConfig::defaults());

    Motion2D inverted;
    inverted.name = "inverted";
    inverted.loop = true;
    for (int i = 0; i < 3; ++i) {
        MotionKeyframe key;
        key.time = Real(i) * Real(0.5);
        key.jointAngles.assign(kJointCount, Real(0));
        key.rootAngle = Real(i) * kPi;  // rotates through inversion
        key.rootPosition = Vec2(0, Real(0.6));
        inverted.insertKeyframe(key);
    }
    const std::string invertedPath = tempClipPath("inverted");
    inverted.writeFile(invertedPath);

    EnvConfig config;
    config.maxEpisodeSteps = 30;
    config.resetNoise = ResetNoise{};
    config.imitation.enabled = true;
    config.imitation.motionPath = invertedPath;
    config.imitation.referenceStateInit = false;
    config.imitation.earlyTerminationPoseError = 0;

    Env2D env;
    env.initialize(config, 1);
    const std::vector<Real> zeros(Env2D::actionDim(), Real(0));
    while (!env.done()) env.step(zeros.data(), Env2D::actionDim());

    // It ran to the time limit despite being low and inverted throughout.
    CHECK(env.truncated());
    CHECK(std::string(env.terminationReason()) != "pelvis_low");
    CHECK(std::string(env.terminationReason()) != "chest_fallen");
    CHECK(std::string(env.terminationReason()) != "head_low");
}

// ---------------------------------------------------------------- batch

TEST(EnvBatch, environmentsAreIndependentAndDifferentlySeeded) {
    EnvBatch batch;
    batch.initialize(quickConfig(), 8, 100);
    CHECK(batch.size() == 8);
    CHECK(batch.observations().size() ==
          static_cast<size_t>(8 * EnvBatch::observationDim()));

    // Different reset noise per environment means the observations differ.
    bool anyDifference = false;
    const int dim = EnvBatch::observationDim();
    for (int i = 0; i < dim; ++i) {
        if (batch.observations()[i] != batch.observations()[static_cast<size_t>(dim) + i]) {
            anyDifference = true;
        }
    }
    CHECK(anyDifference);
}

TEST(EnvBatch, aResetMaskForcesOnlyTheEnvironmentsItNames) {
    EnvBatch batch;
    batch.initialize(quickConfig(500), 4, 7);
    const std::vector<Real> actions(static_cast<size_t>(4 * EnvBatch::actionDim()), Real(0));

    for (int i = 0; i < 20; ++i) batch.step(actions.data(), nullptr);
    for (int i = 0; i < 4; ++i) CHECK(batch.env(i).episodeStep() == 20);

    const std::vector<uint8_t> mask = {0, 1, 0, 1};
    batch.step(actions.data(), mask.data());
    CHECK(batch.env(0).episodeStep() == 21);
    CHECK(batch.env(1).episodeStep() == 0);
    CHECK(batch.env(2).episodeStep() == 21);
    CHECK(batch.env(3).episodeStep() == 0);
    // A forced reset is not an episode ending, so it is not reported as final.
    for (int i = 0; i < 4; ++i) CHECK(batch.finalMask()[static_cast<size_t>(i)] == 0);
}

TEST(EnvBatch, capturedBlocksAreInEnvironmentOrder) {
    EnvBatch batch;
    batch.initialize(quickConfig(), 3, 11);
    const int obsDim = EnvBatch::observationDim();

    CHECK(batch.rewardTerms().size() ==
          static_cast<size_t>(3 * EnvBatch::rewardTermCount()));
    for (int i = 0; i < 3; ++i) {
        CHECK(batch.terminated()[static_cast<size_t>(i)] == 0);
        CHECK(batch.truncated()[static_cast<size_t>(i)] == 0);
        CHECK(batch.episodeStep()[static_cast<size_t>(i)] == 0);
        // Each environment's own observation must land in its own row.
        std::vector<Real> single(obsDim);
        batch.env(i).writeObservation(single.data());
        for (int k = 0; k < obsDim; ++k) {
            CHECK(batch.observations()[static_cast<size_t>(i) * obsDim + k] == single[k]);
        }
    }
}

TEST(EnvBatch, endedEpisodesAutoResetAndReportTheirFinalObservation) {
    // The property PPO depends on. When an episode ends the reported reward,
    // flags and length must describe the episode that ended, the observation
    // must be the fresh one, and the state it ended in must still be available.
    EnvBatch batch;
    EnvConfig config = quickConfig(25);
    config.resetNoise = ResetNoise{};
    batch.initialize(config, 2, 3);
    const std::vector<Real> actions(static_cast<size_t>(2 * EnvBatch::actionDim()), Real(0));
    const int obsDim = EnvBatch::observationDim();

    int stepsUntilDone = 0;
    while (batch.doneCount() == 0 && stepsUntilDone < 200) {
        batch.step(actions.data(), nullptr);
        ++stepsUntilDone;
    }

    CHECK(batch.doneCount() > 0);
    CHECK(batch.finalObservations().size() ==
          static_cast<size_t>(batch.doneCount()) * static_cast<size_t>(obsDim));

    for (int i = 0; i < 2; ++i) {
        if (!batch.finalMask()[static_cast<size_t>(i)]) continue;
        // Terminal length is reported, not the reset one.
        CHECK(batch.episodeStep()[static_cast<size_t>(i)] > 0);
        // The environment itself is already back at the start.
        CHECK(batch.env(i).episodeStep() == 0);
        CHECK(!batch.env(i).done());
        // The reported observation is the fresh one, not the terminal one.
        std::vector<Real> fresh(obsDim);
        batch.env(i).writeObservation(fresh.data());
        for (int k = 0; k < obsDim; ++k) {
            CHECK(batch.observations()[static_cast<size_t>(i) * obsDim + k] == fresh[k]);
        }
    }

    // Final observations are packed densely in environment order.
    int packed = 0;
    for (int i = 0; i < 2; ++i) {
        if (batch.finalMask()[static_cast<size_t>(i)]) ++packed;
    }
    CHECK(packed == batch.doneCount());
}

TEST(EnvBatch, aTruncatedEpisodeStillReportsWhereItWasCutOff) {
    // Truncation is the case that makes the final observation necessary: the
    // critic has to bootstrap from the state the episode was cut off in.
    EnvBatch batch;
    EnvConfig config = quickConfig(30);
    config.resetNoise = ResetNoise{};
    batch.initialize(config, 1, 1);
    const std::vector<Real> actions(static_cast<size_t>(EnvBatch::actionDim()), Real(0));

    for (int i = 0; i < 30; ++i) batch.step(actions.data(), nullptr);

    CHECK(batch.truncated()[0] == 1);
    CHECK(batch.terminated()[0] == 0);
    CHECK(batch.finalMask()[0] == 1);
    CHECK(batch.episodeStep()[0] == 30);
    CHECK(batch.finalObservations().size() ==
          static_cast<size_t>(EnvBatch::observationDim()));
    // A figure still standing at the time limit reports a healthy final height,
    // which is exactly the information a zeroed bootstrap would discard.
    CHECK(batch.finalObservations()[ObservationLayout::kPelvisHeight] > Real(0.7));
}

TEST(EnvBatch, aBatchOfRandomPoliciesRunsCleanly) {
    EnvBatch batch;
    batch.initialize(quickConfig(200), 16, 4242);
    Rng rng(1);
    std::vector<Real> actions(static_cast<size_t>(16 * EnvBatch::actionDim()));
    int episodes = 0;

    for (int step = 0; step < 1200; ++step) {
        for (Real& a : actions) a = rng.uniform(Real(-1), Real(1));
        batch.step(actions.data(), nullptr);
        for (const Real v : batch.observations()) CHECK(std::isfinite(v));
        for (const Real v : batch.finalObservations()) CHECK(std::isfinite(v));
        episodes += batch.doneCount();
    }
    CHECK(episodes > 40);
    // Auto-reset means no environment is ever left in a finished state.
    for (int i = 0; i < 16; ++i) CHECK(!batch.env(i).done());
}
