#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "core/Test.h"
#include "humanoid/Humanoid3D.h"
#include "humanoid/Observation3D.h"
#include "humanoid/RewardTerms3D.h"

using namespace aibf;

namespace {

constexpr Real kDt = Real(1) / Real(240);

struct Figure {
    World3D world;
    Humanoid3D figure;

    Figure() {
        world.addHalfSpace(HalfSpace3D{Vec3(0, 1, 0), Real(0), Real(1.0), Real(0)});
        figure.build(world, Humanoid3DConfig::defaults(), Vec3(0, Real(1.0), 0));
    }

    void run(int steps) {
        for (int i = 0; i < steps; ++i) world.step(kDt);
    }
};

}  // namespace

// ---------------------------------------------------------------- config

TEST(Humanoid3DConfig, theDefaultFigureIsWellFormed) {
    const Humanoid3DConfig cfg = Humanoid3DConfig::defaults();
    CHECK(cfg.validate().empty());
    CHECK(cfg.links.size() == static_cast<size_t>(kLinkCount));
    CHECK(cfg.joints.size() == static_cast<size_t>(kJointCount));
    // Same person as the 2D figure, so results can be compared.
    CHECK_NEAR(cfg.totalMass(), 69.0, 1.0);
    CHECK_NEAR(cfg.restHeight(), 1.64, 0.06);
}

TEST(Humanoid3DConfig, theActionVectorIsThreePerBallAndOnePerHinge) {
    const Humanoid3DConfig cfg = Humanoid3DConfig::defaults();
    // Six ball joints and six hinges: waist, neck, two shoulders, two hips are
    // three degrees of freedom each; elbows, knees and ankles are one.
    CHECK(cfg.actionDim() == 6 * 3 + 6 * 1);

    CHECK(cfg.actionOffset(kWaist) == 0);
    CHECK(cfg.actionOffset(kNeck) == 3);
    // Offsets must be strictly increasing and land exactly on the total.
    int previous = -1;
    for (int i = 0; i < kJointCount; ++i) {
        const int offset = cfg.actionOffset(i);
        CHECK(offset > previous);
        previous = offset;
    }
    CHECK(cfg.actionOffset(kJointCount - 1) + cfg.joints[kJointCount - 1].dof() ==
          cfg.actionDim());
}

TEST(Humanoid3DConfig, validationCatchesATypoedAnchor) {
    Humanoid3DConfig cfg = Humanoid3DConfig::defaults();
    CHECK(cfg.validate().empty());
    // A knee pivot placed at the shoulder. This is the mistake the rest-pose
    // authoring exists to make impossible to miss.
    cfg.joints[kKneeL].restAnchor = Vec3(0, Real(1.4), Real(0.18));
    CHECK(!cfg.validate().empty());
}

TEST(Humanoid3DConfig, theLimbsAreSeparatedInZ) {
    // The whole point of going to 3D. If left and right share a plane, the
    // figure is a 2D figure with extra coordinates and a cartwheel is still
    // impossible.
    const Humanoid3DConfig cfg = Humanoid3DConfig::defaults();
    CHECK(cfg.links[kUpperLegL].restPosition.z > Real(0.05));
    CHECK(cfg.links[kUpperLegR].restPosition.z < Real(-0.05));
    CHECK(cfg.links[kUpperArmL].restPosition.z > Real(0.1));
    CHECK(cfg.links[kUpperArmR].restPosition.z < Real(-0.1));
    // The spine stays on the centre line.
    CHECK_NEAR(cfg.links[kPelvis].restPosition.z, 0.0, 1e-6);
    CHECK_NEAR(cfg.links[kHead].restPosition.z, 0.0, 1e-6);
}

// ---------------------------------------------------------------- construction

TEST(Humanoid3D, everyJointReadsZeroInTheRestPose) {
    // The invariant the reference motions, the observation and the reward all
    // lean on. It is true by construction because the reference rotations are
    // derived from the rest pose rather than authored, and this is what proves
    // the derivation is right.
    Figure rig;

    for (int i = 0; i < kJointCount; ++i) {
        const int32_t ball = rig.figure.ballIndex(i);
        if (ball >= 0) {
            const BallJoint3D& joint = rig.world.ballJoint(ball);
            const Quat relative = joint.relativeRotation(rig.world.body(joint.bodyA),
                                                         rig.world.body(joint.bodyB));
            CHECK(length(rotationVector(relative)) < Real(1e-4));
            continue;
        }
        const int32_t hinge = rig.figure.hingeIndex(i);
        CHECK(hinge >= 0);
        const HingeJoint3D& joint = rig.world.hingeJoint(hinge);
        CHECK(std::abs(joint.jointAngle(rig.world.body(joint.bodyA),
                                        rig.world.body(joint.bodyB))) < Real(1e-4));
    }
}

TEST(Humanoid3D, theRestPoseStartsWithItsJointsAlreadyTogether) {
    // A figure whose anchors are apart at t=0 yanks itself into shape in the
    // first few steps, which looks like a solver problem and is not one.
    Figure rig;
    rig.world.step(kDt);
    CHECK(rig.world.stats().maxJointAnchorError < Real(0.001));
}

TEST(Humanoid3D, theFeetStartOnTheGround) {
    Figure rig;
    CHECK_NEAR(rig.figure.lowestPoint(rig.world), 0.0, 0.01);
}

// ---------------------------------------------------------------- dynamics

TEST(Humanoid3D, anUnactuatedFigureCollapsesWithoutComingApart) {
    // The 3D ragdoll check. With every motor off the figure has to fall over
    // like a body, not explode and not stay standing.
    Figure rig;
    for (int i = 0; i < kJointCount; ++i) {
        const int32_t ball = rig.figure.ballIndex(i);
        if (ball >= 0) rig.world.ballJoint(ball).enableMotor = false;
        const int32_t hinge = rig.figure.hingeIndex(i);
        if (hinge >= 0) rig.world.hingeJoint(hinge).enableMotor = false;
    }

    const Real startHeight = rig.figure.link(rig.world, kHead).position.y;
    Real worstAnchor = 0;
    for (int i = 0; i < 720; ++i) {
        rig.world.step(kDt);
        worstAnchor = std::max(worstAnchor, rig.world.stats().maxJointAnchorError);
    }

    CHECK(!rig.world.stats().unstable);
    CHECK(worstAnchor < Real(0.02));
    // It fell.
    CHECK(rig.figure.link(rig.world, kHead).position.y < startHeight - Real(0.4));
    // But it did not sink through the floor or fly away.
    CHECK(rig.figure.lowestPoint(rig.world) > Real(-0.05));
    CHECK(rig.figure.centerOfMass(rig.world).y < Real(1.0));
}

TEST(Humanoid3D, theMotorsHoldTheFigureUpForAWhile) {
    // Not a standing controller. The motors hold their rest targets, which is
    // enough to keep an undisturbed figure upright for a second or two, and that
    // is the check that the gains and the joint frames are not nonsense. Actual
    // balance is what PPO is for.
    Figure rig;
    rig.figure.relaxToRestPose(rig.world);
    rig.run(240);

    CHECK(!rig.world.stats().unstable);
    CHECK(rig.figure.link(rig.world, kHead).position.y > Real(1.2));
    CHECK(rig.world.stats().maxJointAnchorError < Real(0.01));
}

TEST(Humanoid3D, aBallJointTargetActuallyMovesTheLimb) {
    Figure rig;
    std::vector<Real> actions(static_cast<size_t>(rig.figure.actionDim()), Real(0));

    const Humanoid3DConfig& cfg = rig.figure.config();
    // Swing the left shoulder about x, which lifts the arm.
    actions[static_cast<size_t>(cfg.actionOffset(kShoulderL))] = Real(1);
    rig.figure.setJointTargetsNormalized(rig.world, actions.data(),
                                         static_cast<int>(actions.size()));

    const Vec3 before = rig.figure.link(rig.world, kLowerArmL).position;
    rig.run(180);
    const Vec3 after = rig.figure.link(rig.world, kLowerArmL).position;

    CHECK(!rig.world.stats().unstable);
    CHECK(length(after - before) > Real(0.15));
    // The right arm was not asked to move, so it should not have moved much.
    CHECK(std::abs(rig.figure.link(rig.world, kLowerArmR).position.z +
                   cfg.links[kLowerArmR].restPosition.z * Real(-1)) < Real(0.3));
}

TEST(Humanoid3D, aHingeTargetBendsTheKneeAndOnlyTheKnee) {
    Figure rig;
    std::vector<Real> actions(static_cast<size_t>(rig.figure.actionDim()), Real(0));
    const Humanoid3DConfig& cfg = rig.figure.config();

    // -1 maps onto the knee's lower limit, which is the direction a knee bends.
    actions[static_cast<size_t>(cfg.actionOffset(kKneeL))] = Real(-1);
    rig.figure.setJointTargetsNormalized(rig.world, actions.data(),
                                         static_cast<int>(actions.size()));

    const int32_t hinge = rig.figure.hingeIndex(kKneeL);
    rig.run(240);

    const HingeJoint3D& joint = rig.world.hingeJoint(hinge);
    const Real angle = joint.jointAngle(rig.world.body(joint.bodyA), rig.world.body(joint.bodyB));
    CHECK(!rig.world.stats().unstable);
    CHECK(angle < Real(-0.5));
    // And it stayed a hinge while doing it, rather than letting the shin swing
    // out sideways.
    CHECK(joint.axisMisalignment(rig.world.body(joint.bodyA), rig.world.body(joint.bodyB)) <
          Real(0.05));
}

TEST(Humanoid3D, theActionMappingIsSymmetricAboutTheRestPose) {
    // Zero action has to mean the rest pose exactly. A policy initialised near
    // zero would otherwise start by commanding a pose nobody chose.
    Figure rig;
    std::vector<Real> actions(static_cast<size_t>(rig.figure.actionDim()), Real(0));
    rig.figure.setJointTargetsNormalized(rig.world, actions.data(),
                                         static_cast<int>(actions.size()));

    for (int i = 0; i < kJointCount; ++i) {
        const int32_t ball = rig.figure.ballIndex(i);
        if (ball >= 0) {
            CHECK(length(rotationVector(rig.world.ballJoint(ball).targetRotation)) < Real(1e-5));
        }
        const int32_t hinge = rig.figure.hingeIndex(i);
        if (hinge >= 0) CHECK_NEAR(rig.world.hingeJoint(hinge).targetAngle, 0.0, 1e-6);
    }
}

TEST(Humanoid3D, noReachableActionAsksForAPoseTheJointWillRefuse) {
    // Swept over the corners and axes of the action cube rather than checked
    // against the config, because the question is what the *mapping* produces.
    //
    // The first version of this compared each joint's target range against its
    // cone and failed on four joints: with per-axis ranges sized so a single
    // axis reaches the cone, a diagonal action asks for sqrt(2) times as much.
    // That is not a stability problem, the limit refuses it, but it is action
    // resolution spent commanding poses that can never happen. The mapping now
    // clamps swing and twist separately, and this is the check that it does.
    Figure rig;
    const Humanoid3DConfig& cfg = rig.figure.config();
    std::vector<Real> actions(static_cast<size_t>(rig.figure.actionDim()), Real(0));

    const Real corners[3] = {Real(-1), Real(0), Real(1)};
    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
            for (int z = 0; z < 3; ++z) {
                for (int i = 0; i < kJointCount; ++i) {
                    const int offset = cfg.actionOffset(i);
                    if (cfg.joints[static_cast<size_t>(i)].kind == JointKind::Ball) {
                        actions[static_cast<size_t>(offset + 0)] = corners[x];
                        actions[static_cast<size_t>(offset + 1)] = corners[y];
                        actions[static_cast<size_t>(offset + 2)] = corners[z];
                    } else {
                        actions[static_cast<size_t>(offset)] = corners[x];
                    }
                }
                rig.figure.setJointTargetsNormalized(rig.world, actions.data(),
                                                     static_cast<int>(actions.size()));

                for (int i = 0; i < kJointCount; ++i) {
                    const JointConfig3D& jc = cfg.joints[static_cast<size_t>(i)];
                    const int32_t ball = rig.figure.ballIndex(i);
                    if (ball >= 0) {
                        const Quat target = rig.world.ballJoint(ball).targetRotation;
                        Quat swing, twist;
                        swingTwistDecomposition(target, Vec3(0, 1, 0), swing, twist);
                        CHECK(angleOf(swing) <= jc.coneAngle + Real(1e-3));
                        const Real twistValue = twistAngle(target, Vec3(0, 1, 0));
                        CHECK(twistValue >= jc.lowerTwist - Real(1e-3));
                        CHECK(twistValue <= jc.upperTwist + Real(1e-3));
                        continue;
                    }
                    const int32_t hinge = rig.figure.hingeIndex(i);
                    const Real target = rig.world.hingeJoint(hinge).targetAngle;
                    CHECK(target >= jc.lowerLimit - Real(1e-4));
                    CHECK(target <= jc.upperLimit + Real(1e-4));
                }
            }
        }
    }
}

TEST(Humanoid3D, aFigureDroppedFromAHeightLandsIntact) {
    Figure rig;
    rig.figure.setPose(rig.world, Vec3(0, Real(2.2), 0), Quat::identity());
    rig.figure.relaxToRestPose(rig.world);
    rig.world.clearContactCache();

    Real worstAnchor = 0;
    for (int i = 0; i < 720; ++i) {
        rig.world.step(kDt);
        worstAnchor = std::max(worstAnchor, rig.world.stats().maxJointAnchorError);
    }

    CHECK(!rig.world.stats().unstable);
    CHECK(worstAnchor < Real(0.03));
    CHECK(rig.figure.lowestPoint(rig.world) > Real(-0.05));
    CHECK(rig.world.stats().velocityClampEvents == 0);
}

TEST(Humanoid3D, aTumblingFigureInFreeFlightKeepsItsAngularMomentum) {
    // The property a backflip depends on. Once the feet leave the ground nothing
    // can change the figure's total angular momentum, so whatever spin it left
    // with is the spin it has to land with.
    World3D world;
    world.gravity = Vec3(0, 0, 0);
    Humanoid3D figure;
    figure.build(world, Humanoid3DConfig::defaults(), Vec3(0, Real(5.0), 0));
    figure.relaxToRestPose(world);

    for (int i = 0; i < kLinkCount; ++i) {
        figure.link(world, i).angularVelocity = Vec3(0, 0, Real(6));
    }

    const Vec3 initial = world.angularMomentum();
    for (int i = 0; i < 240; ++i) world.step(kDt);
    const Vec3 final = world.angularMomentum();

    CHECK(!world.stats().unstable);
    CHECK(length(initial) > Real(1));
    // One second of tumbling, within the truncation bound measured in
    // test_joints3d for a two-body pair.
    CHECK(length(final - initial) < length(initial) * Real(0.05));
}

TEST(Humanoid3D, standingReadsOneInTheHeightSlots) {
    // The same assertion the 2D suite makes, and it is here because the two
    // figures briefly disagreed. The 3D observation divided the pelvis by the
    // *whole body* height, so a perfectly upright figure read 0.607, and the
    // evaluation script, which prints that slot as "of rest height", reported a
    // working standing policy as a deep crouch sitting just above its own
    // termination threshold.
    //
    // Nothing was physically wrong. The reward and the termination both used the
    // pelvis's own rest height and were correct throughout. Only the observation
    // used a different convention, and one convention per repository is the
    // whole point.
    World3D world;
    world.addHalfSpace(HalfSpace3D{Vec3(0, 1, 0), Real(0), Real(1.0), Real(0)});
    Humanoid3D figure;
    const Humanoid3DConfig cfg = Humanoid3DConfig::defaults();
    figure.build(world, cfg, Vec3(0, Real(1.0), 0));

    const ObservationScales3D scales = ObservationScales3D::fromConfig(cfg);
    std::vector<Real> obs(static_cast<size_t>(ObservationLayout3D::kDimension));
    writeObservation3D(world, figure, scales, Real(0), obs.data());

    CHECK_NEAR(obs[ObservationLayout3D::kPelvisHeight], 1.0, 0.02);
    CHECK_NEAR(obs[ObservationLayout3D::kHeadHeight], 1.0, 0.02);

    // Offsets stay divided by the whole body, which is what makes them
    // comparable between links rather than each carrying its own scale.
    const int headOffset = ObservationLayout3D::kLinkOffsets + 3 * 1;  // chest is index 0
    CHECK(std::abs(obs[headOffset + 1]) < Real(0.5));
}

// ---------------------------------------------------------------- kinematics

TEST(Humanoid3D, forwardKinematicsAgreesWithTheSolverAtRest) {
    // FK is derived from the joint constants; the figure in the world was built
    // from the same rest pose. If the two disagree, one of them has the child
    // orientation composition backwards, and that is a mistake whose symptom is
    // a reference motion that looks subtly wrong rather than obviously broken.
    Figure rig;
    std::vector<Quat> rotations(kJointCount, Quat::identity());
    std::vector<Vec3> positions(kLinkCount);
    std::vector<Quat> orientations(kLinkCount);

    const RigidBody3D& pelvis = rig.figure.link(rig.world, kPelvis);
    rig.figure.forwardKinematics(rig.world, pelvis.position, pelvis.orientation, rotations.data(),
                                 positions.data(), orientations.data());

    for (int i = 0; i < kLinkCount; ++i) {
        const RigidBody3D& body = rig.figure.link(rig.world, i);
        CHECK(length(positions[i] - body.position) < Real(1e-4));
        CHECK(angleOf(orientations[i] * conjugate(body.orientation)) < Real(1e-3));
    }
}

TEST(Humanoid3D, forwardKinematicsMatchesTheSolverAfterTheJointsMove) {
    // The rest pose alone would pass even if the joint rotation were ignored
    // entirely, because every rotation there is the identity. This drives the
    // figure to a real pose, lets the solver settle it, then asks FK to
    // reproduce it from the joint values the solver ended up at.
    Figure rig;
    std::vector<Real> actions(static_cast<size_t>(rig.figure.actionDim()), Real(0));
    const Humanoid3DConfig& cfg = rig.figure.config();
    actions[static_cast<size_t>(cfg.actionOffset(kShoulderL))] = Real(0.7);
    actions[static_cast<size_t>(cfg.actionOffset(kKneeL))] = Real(-0.6);
    actions[static_cast<size_t>(cfg.actionOffset(kWaist)) + 2] = Real(0.4);
    rig.figure.setJointTargetsNormalized(rig.world, actions.data(),
                                         static_cast<int>(actions.size()));
    rig.run(240);

    std::vector<Quat> rotations(kJointCount);
    for (int j = 0; j < kJointCount; ++j) {
        rotations[static_cast<size_t>(j)] = jointRotation(rig.world, rig.figure, j);
    }

    std::vector<Vec3> positions(kLinkCount);
    std::vector<Quat> orientations(kLinkCount);
    const RigidBody3D& pelvis = rig.figure.link(rig.world, kPelvis);
    rig.figure.forwardKinematics(rig.world, pelvis.position, pelvis.orientation, rotations.data(),
                                 positions.data(), orientations.data());

    // Two centimetres at the worst link, not two millimetres, and the reason is
    // worth stating because it looks like a kinematics bug and is not.
    //
    // A hinge's state is a single number, so jointRotation() reports a pure
    // rotation about the hinge axis. The solver's actual relative rotation has a
    // small out-of-plane residual as well, because the two-axis lock is a
    // constraint rather than an exact kinematic reduction. Reading a hinge back
    // therefore loses that residual, and the left foot sits at the end of a
    // chain containing two of them, so the error accumulates to about 13 mm with
    // the joint anchors still together to 30 microns.
    //
    // This costs nothing in practice: forward kinematics is used to expand a
    // *reference clip*, whose hinges are pure axis rotations by construction, so
    // the lossy direction is never exercised.
    Real worstPosition = 0, worstAngle = 0;
    for (int i = 0; i < kLinkCount; ++i) {
        const RigidBody3D& body = rig.figure.link(rig.world, i);
        worstPosition = std::max(worstPosition, length(positions[i] - body.position));
        worstAngle = std::max(worstAngle,
                              angleOf(orientations[i] * conjugate(body.orientation)));
    }
    CHECK(worstPosition < Real(0.02));
    CHECK(worstAngle < Real(0.08));
    // The joints themselves are together, so the gap above is the hinge
    // reduction and not a solver failure.
    CHECK(rig.world.stats().maxJointAnchorError < Real(0.001));
}

TEST(Humanoid3D, forwardKinematicsRoundTripsExactlyOnItsOwnOutput) {
    // The exact version of the test above, over the path that is actually used:
    // a set of joint rotations goes in, the figure is placed at the result, and
    // reading it back reproduces the same links. No hinge reduction happens
    // anywhere in that loop.
    Figure rig;
    std::vector<Quat> rotations(kJointCount, Quat::identity());
    rotations[kShoulderL] = Quat::fromAxisAngle(normalize(Vec3(1, Real(0.3), 0)), Real(0.6));
    rotations[kHipR] = Quat::fromAxisAngle(normalize(Vec3(Real(0.2), 0, 1)), Real(0.5));
    rotations[kKneeL] = Quat::fromAxisAngle(Vec3(0, 0, 1), Real(-0.7));

    const Vec3 rootPosition(Real(0.4), Real(1.1), Real(-0.2));
    const Quat rootOrientation = Quat::fromAxisAngle(normalize(Vec3(1, 1, 0)), Real(0.35));

    std::vector<Vec3> positions(kLinkCount);
    std::vector<Quat> orientations(kLinkCount);
    rig.figure.forwardKinematics(rig.world, rootPosition, rootOrientation, rotations.data(),
                                 positions.data(), orientations.data());
    rig.figure.applyPose(rig.world, positions.data(), orientations.data(), nullptr, nullptr);

    std::vector<Vec3> again(kLinkCount);
    std::vector<Quat> againOrientations(kLinkCount);
    rig.figure.forwardKinematics(rig.world, rootPosition, rootOrientation, rotations.data(),
                                 again.data(), againOrientations.data());

    for (int i = 0; i < kLinkCount; ++i) {
        const RigidBody3D& body = rig.figure.link(rig.world, i);
        CHECK(length(again[i] - body.position) < Real(1e-4));
        CHECK(angleOf(againOrientations[i] * conjugate(body.orientation)) < Real(1e-3));
    }

    // And the placed figure's joints are satisfied, which is what makes this a
    // legal state to start an episode from rather than a pose the solver has to
    // repair.
    rig.world.refreshContacts();
    rig.world.step(Real(1) / Real(240));
    CHECK(rig.world.stats().maxJointAnchorError < Real(0.002));
}

TEST(Humanoid3D, velocityKinematicsMatchAFiniteDifference) {
    // Reference State Initialization stands on this. Starting an episode at the
    // apex of a flip with the right shape and zero velocity is a different state
    // entirely, so the velocities have to be right too.
    Figure rig;
    std::vector<Quat> rotations(kJointCount, Quat::identity());
    rotations[kShoulderL] = Quat::fromAxisAngle(Vec3(1, 0, 0), Real(0.4));
    rotations[kKneeL] = Quat::fromAxisAngle(Vec3(0, 0, 1), Real(-0.5));

    std::vector<Vec3> rates(kJointCount, Vec3(0, 0, 0));
    rates[kShoulderL] = Vec3(Real(1.2), 0, 0);
    rates[kKneeL] = Vec3(0, 0, Real(-0.9));

    const Vec3 rootPosition(0, Real(1.0), 0);
    const Quat rootOrientation = Quat::fromAxisAngle(Vec3(0, 0, 1), Real(0.2));
    const Vec3 rootVelocity(Real(0.3), Real(-0.4), Real(0.1));
    const Vec3 rootSpin(0, 0, Real(1.5));

    std::vector<Vec3> positions(kLinkCount);
    std::vector<Quat> orientations(kLinkCount);
    rig.figure.forwardKinematics(rig.world, rootPosition, rootOrientation, rotations.data(),
                                 positions.data(), orientations.data());

    std::vector<Vec3> velocities(kLinkCount);
    std::vector<Vec3> spins(kLinkCount);
    rig.figure.forwardKinematicsVelocity(rig.world, rootVelocity, rootSpin, rates.data(),
                                         positions.data(), orientations.data(), velocities.data(),
                                         spins.data());

    // Numerically differentiate the pose kinematics and compare.
    const Real h = Real(1e-4);
    std::vector<Quat> advanced(kJointCount);
    for (int j = 0; j < kJointCount; ++j) {
        advanced[static_cast<size_t>(j)] =
            quatFromRotationVector(rates[static_cast<size_t>(j)] * h) *
            rotations[static_cast<size_t>(j)];
    }
    std::vector<Vec3> nextPositions(kLinkCount);
    std::vector<Quat> nextOrientations(kLinkCount);
    rig.figure.forwardKinematics(rig.world, rootPosition + rootVelocity * h,
                                 integrateExact(rootOrientation, rootSpin, h), advanced.data(),
                                 nextPositions.data(), nextOrientations.data());

    for (int i = 0; i < kLinkCount; ++i) {
        const Vec3 numerical = (nextPositions[i] - positions[i]) / h;
        CHECK(length(numerical - velocities[i]) < Real(0.05));
    }
}

TEST(Humanoid3D, groundedRootHeightPutsTheLowestPointOnTheFloor) {
    // So an authored crouch actually reaches the floor rather than hovering or
    // starting the episode interpenetrating.
    Figure rig;
    std::vector<Quat> rotations(kJointCount, Quat::identity());
    rotations[kKneeL] = Quat::fromAxisAngle(Vec3(0, 0, 1), Real(-0.8));
    rotations[kKneeR] = Quat::fromAxisAngle(Vec3(0, 0, 1), Real(-0.8));

    const Quat upright = Quat::identity();
    const Real height = rig.figure.groundedRootHeight(rig.world, upright, rotations.data());

    std::vector<Vec3> positions(kLinkCount);
    std::vector<Quat> orientations(kLinkCount);
    rig.figure.forwardKinematics(rig.world, Vec3(0, height, 0), upright, rotations.data(),
                                 positions.data(), orientations.data());

    Real lowest = std::numeric_limits<Real>::max();
    for (int i = 0; i < kLinkCount; ++i) {
        const LinkConfig3D& lc = rig.figure.config().links[static_cast<size_t>(i)];
        const Vec3 axis = rotate(orientations[i], Vec3(0, lc.halfLength, 0));
        lowest = std::min(lowest, positions[i].y - std::abs(axis.y) - lc.radius);
    }
    CHECK_NEAR(lowest, 0.0, 1e-4);
}
