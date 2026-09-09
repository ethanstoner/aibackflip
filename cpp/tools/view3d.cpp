// Plays a 3D reference motion, either kinematically or through the motors.
//
// Two modes, and the difference between them is the whole point:
//
//   --kinematic  the figure is placed at the reference pose every frame. This
//                shows what was authored, with no physics at all.
//   --preview    the motors chase the reference and physics decides the rest,
//                with no learning anywhere. If the PD preview cannot roughly
//                follow a clip, no policy is going to either, and finding that
//                out costs seconds instead of an hour of training.
//
//   aibf_view3d motions/backflip3d.json --preview --capture out/flip.png
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>

#include "engine/Renderer3D.h"
#include "engine/Window.h"
#include "humanoid/Humanoid3D.h"
#include "env/Env3D.h"
#include "motion/Motion3D.h"

using namespace aibf;

namespace {

Vec4 linkColour(int linkId) {
    switch (linkId) {
        case kHead: return Vec4{Real(0.89), Real(0.86), Real(0.63), Real(1)};
        case kChest:
        case kPelvis: return Vec4{Real(0.42), Real(0.47), Real(0.58), Real(1)};
        case kUpperArmL:
        case kLowerArmL:
        case kUpperArmR:
        case kLowerArmR: return Vec4{Real(0.50), Real(0.83), Real(0.76), Real(1)};
        case kFootL:
        case kFootR: return Vec4{Real(0.87), Real(0.55), Real(0.29), Real(1)};
        default: return Vec4{Real(0.43), Real(0.61), Real(0.91), Real(1)};
    }
}

// Drives the motors from the clip. Ball joints take the rotation directly;
// hinges take its component about their own axis, because that is the only part
// of it the joint can produce.
void driveMotors(World3D& world, const Humanoid3D& figure, const MotionPose3D& pose) {
    for (int j = 0; j < kJointCount; ++j) {
        const int32_t ball = figure.ballIndex(j);
        if (ball >= 0) {
            world.ballJoint(ball).targetRotation = pose.jointRotations[static_cast<size_t>(j)];
            continue;
        }
        HingeJoint3D& hinge = world.hingeJoint(figure.hingeIndex(j));
        hinge.targetAngle = twistAngle(pose.jointRotations[static_cast<size_t>(j)],
                                       normalize(hinge.localHingeAxisA));
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string path;
    std::string capturePath;
    // The env config the training run will use, so the preview exercises the
    // configured figure rather than the defaults. Motor gains in particular are
    // the whole question a preview is asked to answer.
    std::string envConfigPath;
    bool kinematic = false;
    int captureCount = 8;
    int width = 1280, height = 720;
    Real speed = 1;
    Real cameraYaw = Real(0.9);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--kinematic") kinematic = true;
        else if (arg == "--preview") kinematic = false;
        else if (arg == "--capture" && i + 1 < argc) capturePath = argv[++i];
        else if (arg == "--frames" && i + 1 < argc) captureCount = std::atoi(argv[++i]);
        else if (arg == "--width" && i + 1 < argc) width = std::atoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) height = std::atoi(argv[++i]);
        else if (arg == "--speed" && i + 1 < argc) speed = Real(std::atof(argv[++i]));
        else if (arg == "--yaw" && i + 1 < argc) cameraYaw = Real(std::atof(argv[++i]));
        else if (arg == "--config" && i + 1 < argc) envConfigPath = argv[++i];
        else if (arg[0] != '-') path = arg;
    }
    if (path.empty()) {
        std::fprintf(stderr, "usage: aibf_view3d <motion.json> [--kinematic|--preview]\n");
        return 2;
    }

    std::string error;
    Motion3D motion = Motion3D::readFile(path, &error);
    if (!error.empty()) {
        std::fprintf(stderr, "could not load %s: %s\n", path.c_str(), error.c_str());
        return 1;
    }

    Humanoid3DConfig config = Humanoid3DConfig::defaults();
    if (!envConfigPath.empty()) {
        std::string configError;
        const EnvConfig3D envConfig = EnvConfig3D::loadFile(envConfigPath, &configError);
        if (!configError.empty()) {
            std::fprintf(stderr, "config problem: %s\n", configError.c_str());
            return 2;
        }
        config = envConfig.humanoid;
        std::printf("using %s\n", envConfigPath.c_str());
    }
    World3D world;
    world.addHalfSpace(HalfSpace3D{Vec3(0, 1, 0), Real(0), Real(1.0), Real(0)});
    Humanoid3D figure;
    figure.build(world, config, config.links[kPelvis].restPosition);
    motion.clampToLimits(config);

    const bool capturing = !capturePath.empty();
    if (capturing) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    Window window;
    if (!window.create(width, height, "aibf_view3d")) return 1;
    if (capturing) glfwHideWindow(window.handle());

    Renderer3D renderer;
    if (!renderer.initialize()) {
        std::fprintf(stderr, "renderer failed: %s\n", renderer.lastError().c_str());
        return 1;
    }

    Camera3D camera;
    camera.yaw = cameraYaw;
    camera.distance = Real(5.5);

    std::vector<Vec3> positions(kLinkCount);
    std::vector<Quat> orientations(kLinkCount);
    std::vector<Vec3> velocities(kLinkCount);
    std::vector<Vec3> spins(kLinkCount);
    std::vector<Vec3> jointRates(kJointCount);

    // Place the figure on the first frame of the clip, so a preview starts from
    // the pose the motion starts from rather than from the rest pose.
    {
        const MotionPose3D start = motion.sample(0);
        Vec3 rootVelocity, rootSpin;
        motion.sampleVelocity(0, rootVelocity, rootSpin, jointRates);
        figure.forwardKinematics(world, start.rootPosition, start.rootOrientation,
                                 start.jointRotations.data(), positions.data(),
                                 orientations.data());
        figure.forwardKinematicsVelocity(world, rootVelocity, rootSpin, jointRates.data(),
                                         positions.data(), orientations.data(), velocities.data(),
                                         spins.data());
        figure.applyPose(world, positions.data(), orientations.data(), velocities.data(),
                         spins.data());
        world.clearContactCache();
        world.refreshContacts();
    }

    const Real dt = Real(1) / Real(240);
    const int substeps = 4;
    const Real controlDt = dt * Real(substeps);
    Real time = 0;
    int frame = 0;
    int captured = 0;
    const Real captureSpacing =
        captureCount > 1 ? motion.duration() / Real(captureCount - 1) : motion.duration();
    Real nextCapture = 0;

    Real peakRoot = 0, lowestPoint = Real(1e9);

    while (window.isOpen()) {
        window.pollEvents();
        if (window.input().keyPressed[GLFW_KEY_ESCAPE]) window.requestClose();
        camera.aspect = window.aspect();

        const MotionPose3D pose = motion.sample(time);

        if (kinematic) {
            Vec3 rootVelocity, rootSpin;
            motion.sampleVelocity(time, rootVelocity, rootSpin, jointRates);
            figure.forwardKinematics(world, pose.rootPosition, pose.rootOrientation,
                                     pose.jointRotations.data(), positions.data(),
                                     orientations.data());
            figure.applyPose(world, positions.data(), orientations.data(), nullptr, nullptr);
        } else {
            driveMotors(world, figure, pose);
            for (int i = 0; i < substeps; ++i) world.step(dt);
        }

        const RigidBody3D& pelvis = figure.link(world, kPelvis);
        peakRoot = std::max(peakRoot, pelvis.position.y);
        lowestPoint = std::min(lowestPoint, figure.lowestPoint(world));

        camera.target = camera.target + (Vec3(pelvis.position.x, Real(0.9), pelvis.position.z) -
                                         camera.target) * Real(0.1);

        renderer.beginFrame(camera, Vec4{Real(0.09), Real(0.10), Real(0.13), Real(1)});
        renderer.ground(Real(40), Vec4{Real(0.16), Real(0.17), Real(0.21), Real(1)},
                        Vec4{Real(0.28), Real(0.31), Real(0.38), Real(1)});
        for (int i = 0; i < kLinkCount; ++i) {
            const RigidBody3D& body = figure.link(world, i);
            renderer.capsule(body.position, body.orientation, body.radius, body.halfLength,
                             linkColour(i));
        }
        renderer.endFrame();

        if (capturing && time >= nextCapture && captured < captureCount) {
            std::string out = capturePath;
            const size_t dot = out.rfind('.');
            char suffix[16];
            std::snprintf(suffix, sizeof(suffix), "_%02d", captured);
            out = (dot == std::string::npos) ? out + suffix
                                             : out.substr(0, dot) + suffix + out.substr(dot);
            if (!window.saveScreenshot(out)) {
                std::fprintf(stderr, "failed to write %s\n", out.c_str());
                return 1;
            }
            std::printf("t=%.2f turns=%.2f pelvis=%.3f m  %s\n", double(time),
                        double(pose.rootTurns), double(pelvis.position.y), out.c_str());
            ++captured;
            nextCapture += captureSpacing;
        }

        window.present();

        time += controlDt * speed;
        ++frame;
        if (time > motion.duration()) {
            if (capturing) break;
            time = 0;
        }
    }

    std::printf("%s: %s, peak pelvis %.3f m, lowest point %.3f m\n", motion.name.c_str(),
                kinematic ? "kinematic" : "PD preview", double(peakRoot), double(lowestPoint));
    renderer.shutdown();
    return 0;
}
