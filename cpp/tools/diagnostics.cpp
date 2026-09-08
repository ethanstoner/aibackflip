// Measurements that are more useful as numbers than as pass/fail assertions.
//
// The unit tests pin behaviour against thresholds; this reports the actual
// values behind those thresholds, so solver settings can be chosen from
// evidence and the numbers can be quoted in docs/PROGRESS.md.
//
//   aibf_diag            run everything
//   aibf_diag drift      run one section by name
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/Rng.h"
#include "humanoid/Humanoid2D.h"
#include "physics/World2D.h"

using namespace aibf;

namespace {

constexpr Real kDt = Real(1) / Real(240);

int stepsFor(Real seconds) { return static_cast<int>(std::lround(seconds / kDt)); }

struct Chain {
    World2D world;
    std::vector<int32_t> links;

    // A pinned chain of `count` links, yanked sideways so every joint is loaded.
    explicit Chain(int count, bool positionSolver, int velocityIterations = 10) {
        world.solver.useJointPositionSolver = positionSolver;
        world.solver.positionIterations = positionSolver ? 4 : 0;
        world.solver.velocityIterations = velocityIterations;

        RigidBody2D anchor;
        anchor.position = Vec2(0, 3);
        anchor.radius = Real(0.02);
        anchor.makeStatic();
        anchor.collisionGroup = 1;
        int32_t previous = world.addBody(anchor);

        const Real half = Real(0.25);
        Vec2 attach(0, 3);
        Vec2 previousAnchor(0, 0);
        for (int i = 0; i < count; ++i) {
            RigidBody2D body;
            body.position = attach - Vec2(0, half);
            body.setCapsule(Real(0.05), half, Real(400));
            body.collisionGroup = 1;
            const int32_t link = world.addBody(body);
            links.push_back(link);

            RevoluteJoint2D joint;
            joint.bodyA = previous;
            joint.bodyB = link;
            joint.localAnchorA = previousAnchor;
            joint.localAnchorB = Vec2(0, half);
            world.addJoint(joint);

            previous = link;
            previousAnchor = Vec2(0, -half);
            attach = attach - Vec2(0, 2 * half);
        }
    }

    Real runAndReportWorstAnchorError(Real seconds, Real yank) {
        world.body(links.back()).velocity = Vec2(yank, 0);
        Real worst = 0;
        const int steps = stepsFor(seconds);
        for (int i = 0; i < steps; ++i) {
            if (i % 240 == 0) world.body(links.back()).velocity += Vec2(yank, 0);
            world.step(kDt);
            worst = std::max(worst, world.stats().maxJointAnchorError);
        }
        return worst;
    }
};

void sectionJointDrift() {
    std::printf("\n== joint anchor drift (5-link chain, repeatedly yanked, 6 s) ==\n");
    std::printf("  %-34s %14s\n", "stabilisation", "worst error");
    {
        Chain chain(5, true);
        std::printf("  %-34s %11.3g m\n", "position solver (4 iterations)",
                    double(chain.runAndReportWorstAnchorError(Real(6), Real(12))));
    }
    {
        Chain chain(5, false);
        std::printf("  %-34s %11.3g m\n", "Baumgarte velocity bias only",
                    double(chain.runAndReportWorstAnchorError(Real(6), Real(12))));
    }

    std::printf("\n== anchor drift vs velocity iterations (position solver on) ==\n");
    std::printf("  %-34s %14s\n", "velocity iterations", "worst error");
    for (const int iterations : {2, 4, 6, 8, 10, 16, 24}) {
        Chain chain(5, true, iterations);
        std::printf("  %-34d %11.3g m\n", iterations,
                    double(chain.runAndReportWorstAnchorError(Real(6), Real(12))));
    }
}

void sectionEnergy() {
    std::printf("\n== energy drift, undamped pendulum, no contacts ==\n");
    World2D world;
    RigidBody2D anchor;
    anchor.position = Vec2(0, 2);
    anchor.radius = Real(0.02);
    anchor.makeStatic();
    anchor.collisionGroup = 1;
    const int32_t pivot = world.addBody(anchor);

    RigidBody2D rod;
    rod.position = Vec2(Real(0.5), 2);
    rod.angle = kHalfPi;
    rod.setCapsule(Real(0.05), Real(0.5), Real(100));
    rod.collisionGroup = 1;
    const int32_t bob = world.addBody(rod);

    RevoluteJoint2D joint;
    joint.bodyA = pivot;
    joint.bodyB = bob;
    joint.localAnchorB = Vec2(0, Real(0.5));
    world.addJoint(joint);

    const Real start = world.kineticEnergy() + world.potentialEnergy();
    std::printf("  %-34s %14s\n", "simulated time", "energy vs start");
    for (const Real seconds : {Real(1), Real(5), Real(10), Real(30), Real(60)}) {
        const int target = stepsFor(seconds);
        static int done = 0;
        for (; done < target; ++done) world.step(kDt);
        const Real now = world.kineticEnergy() + world.potentialEnergy();
        std::printf("  %-31.0f s %13.3f%%\n", double(seconds),
                    double((now - start) / std::abs(start) * Real(100)));
    }
}

void sectionResting() {
    std::printf("\n== humanoid at rest on the floor ==\n");
    World2D world;
    world.addHalfSpace(HalfSpace{Vec2(0, 1), 0, Real(1.0), Real(0)});
    Humanoid2D figure;
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    figure.build(world, config);
    figure.reset(world, config.links[kPelvis].restPosition);
    figure.setMotorsEnabled(world, true);

    std::printf("  %-34s %10s %10s %12s\n", "simulated time", "pelvis", "penetr.", "anchor err");
    int done = 0;
    for (const Real seconds : {Real(1), Real(5), Real(15), Real(30)}) {
        const int target = stepsFor(seconds);
        for (; done < target; ++done) world.step(kDt);
        std::printf("  %-31.0f s %9.4f m %8.2g m %10.2g m\n", double(seconds),
                    double(figure.link(world, kPelvis).position.y),
                    double(world.stats().maxPenetration),
                    double(world.stats().maxJointAnchorError));
    }
    std::printf("  note: this is a symmetric equilibrium held by pose motors, not balance.\n");
}

void sectionPerturbation() {
    std::printf("\n== response to a sideways shove (motors holding the rest pose) ==\n");
    std::printf("  %-14s %12s %12s %10s\n", "impulse", "pelvis @4s", "upright", "fell?");
    for (const Real impulse : {Real(0), Real(5), Real(15), Real(40), Real(90)}) {
        World2D world;
        world.addHalfSpace(HalfSpace{Vec2(0, 1), 0, Real(1.0), Real(0)});
        Humanoid2D figure;
        const Humanoid2DConfig config = Humanoid2DConfig::defaults();
        figure.build(world, config);
        figure.reset(world, config.links[kPelvis].restPosition);
        figure.setMotorsEnabled(world, true);

        for (int i = 0; i < stepsFor(Real(0.5)); ++i) world.step(kDt);
        figure.link(world, kPelvis).applyImpulse(Vec2(impulse, 0), Vec2(0, 0));
        for (int i = 0; i < stepsFor(Real(4)); ++i) world.step(kDt);

        const Real height = figure.link(world, kPelvis).position.y;
        std::printf("  %-11.0f Ns %10.3f m %12.2f %10s\n", double(impulse), double(height),
                    double(figure.uprightness(world, kChest)),
                    height < Real(0.6) ? "yes" : "no");
    }
    std::printf("  a pose-holding controller has no balance strategy; anything it survives\n"
                "  it survives by stiffness alone. M5 is where balance is actually learned.\n");
}

// Can each joint's motor actually drive its own subtree to a commanded angle
// against gravity, and does it need to saturate to do it? Gains that look
// plausible in aggregate can still leave one joint unable to lift its own limb,
// and that shows up during training as a policy that never uses it.
void sectionJointAuthority() {
    std::printf("\n== per-joint motor authority (pelvis pinned, gravity on, 2.5 s hold) ==\n");
    std::printf("  %-12s %8s %8s %8s %10s %10s %7s\n", "joint", "target", "reached", "error",
                "peak tau", "hold tau", "of max");

    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    Real worstError = 0;
    const char* worstJoint = "";

    for (int j = 0; j < kJointCount; ++j) {
        World2D world;
        world.addHalfSpace(HalfSpace{Vec2(0, 1), Real(-5), Real(1.0), Real(0)});
        Humanoid2D figure;
        figure.build(world, config);
        figure.reset(world, config.links[kPelvis].restPosition);
        figure.setMotorsEnabled(world, true);
        figure.link(world, kPelvis).makeStatic();

        // Command 70% of the way towards whichever limit is further from rest,
        // so the joint has to work rather than sit near its neutral pose.
        const JointConfig& jc = config.joints[static_cast<size_t>(j)];
        const Real reach = (std::abs(jc.upperLimit) > std::abs(jc.lowerLimit)) ? jc.upperLimit
                                                                              : jc.lowerLimit;
        const Real target = reach * Real(0.7);
        figure.setJointTarget(world, j, target);

        // Peak torque is dominated by the transient: a step command starts with
        // a huge kp*error and clamps immediately, so peaking at 100% says
        // nothing. What matters is the torque still needed once settled.
        const int total = stepsFor(Real(2.5));
        const int holdFrom = stepsFor(Real(2.0));
        Real peakTorque = 0;
        Real holdSum = 0;
        int holdSamples = 0;
        for (int i = 0; i < total; ++i) {
            world.step(kDt);
            const RevoluteJoint2D& rj = figure.joint(world, j);
            const Real torque = std::abs(rj.motorImpulse) / kDt;
            peakTorque = std::max(peakTorque, torque);
            if (i >= holdFrom) {
                holdSum += torque;
                ++holdSamples;
            }
        }
        const Real holdTorque = holdSamples > 0 ? holdSum / Real(holdSamples) : Real(0);

        const Real reached = figure.jointAngle(world, j);
        const Real error = std::abs(reached - target);
        if (error > worstError) {
            worstError = error;
            worstJoint = jc.name.c_str();
        }
        std::printf("  %-12s %+8.3f %+8.3f %8.4f %8.1f Nm %8.1f Nm %6.0f%%\n", jc.name.c_str(),
                    double(target), double(reached), double(error), double(peakTorque),
                    double(holdTorque), double(holdTorque / jc.maxTorque * Real(100)));
    }
    std::printf("  worst tracking error: %.4f rad (%s)\n", double(worstError), worstJoint);
    std::printf("  peak torque saturates on a step command by construction and is not a\n"
                "  problem. Hold torque is the diagnostic: near 100%% there means the joint is\n"
                "  out of authority even at rest, and the ceiling or the gain has to rise.\n");
}

void sectionThroughput() {
    std::printf("\n== headless throughput ==\n");

    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    for (const int worldCount : {1, 8, 25, 32}) {
        std::vector<World2D> worlds(static_cast<size_t>(worldCount));
        std::vector<Humanoid2D> figures(static_cast<size_t>(worldCount));
        Rng rng(7);
        for (int i = 0; i < worldCount; ++i) {
            worlds[static_cast<size_t>(i)].addHalfSpace(
                HalfSpace{Vec2(0, 1), 0, Real(1.0), Real(0)});
            figures[static_cast<size_t>(i)].build(worlds[static_cast<size_t>(i)], config);
            ResetNoise noise;
            noise.rootAngle = Real(0.1);
            noise.jointAngle = Real(0.2);
            figures[static_cast<size_t>(i)].reset(worlds[static_cast<size_t>(i)],
                                                  config.links[kPelvis].restPosition, noise, rng);
            figures[static_cast<size_t>(i)].setMotorsEnabled(worlds[static_cast<size_t>(i)], true);
        }

        const int substeps = 4;  // one 60 Hz control step
        const int controlSteps = 3000;
        const auto start = std::chrono::steady_clock::now();
        for (int step = 0; step < controlSteps; ++step) {
            for (int i = 0; i < worldCount; ++i) {
                for (int s = 0; s < substeps; ++s) worlds[static_cast<size_t>(i)].step(kDt);
            }
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        const double envSteps = double(controlSteps) * double(worldCount);
        std::printf("  %2d worlds: %9.0f control-steps/s  (%8.0f env-steps/s, %5.0fx real time)\n",
                    worldCount, double(controlSteps) / seconds, envSteps / seconds,
                    envSteps / seconds / 60.0 / double(worldCount));
    }
    std::printf("  single-threaded, one humanoid per world, 4 substeps per control step.\n");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    auto wanted = [&](const char* name) {
        return filter.empty() || std::strstr(name, filter.c_str()) != nullptr;
    };

    std::printf("aibackflip physics diagnostics (dt = 1/%.0f s)\n", double(1 / kDt));
    if (wanted("drift")) sectionJointDrift();
    if (wanted("energy")) sectionEnergy();
    if (wanted("resting")) sectionResting();
    if (wanted("authority")) sectionJointAuthority();
    if (wanted("perturbation")) sectionPerturbation();
    if (wanted("throughput")) sectionThroughput();
    std::printf("\n");
    return 0;
}
