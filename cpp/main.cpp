// Interactive 2D humanoid simulator.
//
// The physics loop is fixed-step and completely independent of the render rate:
// the frame time is accumulated and drained in whole 1/240 s substeps, with a
// cap so that a stall (dragging the window, a breakpoint) cannot produce a
// hundred catch-up steps and launch the figure. What runs here is the same
// World2D that headless training will step, which is the point.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>

#include "core/Rng.h"
#include "engine/DebugDraw.h"
#include "engine/Renderer2D.h"
#include "engine/Window.h"
#include "humanoid/Humanoid2D.h"
#include "physics/World2D.h"

using namespace aibf;

namespace {

constexpr Real kPhysicsHz = 240;
constexpr Real kPhysicsDt = Real(1) / kPhysicsHz;
constexpr Real kMaxFrameTime = Real(0.25);  // never simulate more than this per frame

struct Options {
    int width = 1280;
    int height = 720;
    bool motorsOn = false;
    std::string configPath;
    std::string dumpConfigPath;
    // Automated capture: simulate `captureTime` seconds, write a PNG, exit.
    // This is how rendered output gets checked without a person watching.
    std::string capturePath;
    Real captureTime = Real(2);
    int captureCount = 1;
    Real captureInterval = Real(0.5);
    bool hidden = false;
    uint64_t seed = 1;
};

void printUsage() {
    std::printf(
        "aibackflip - 2D humanoid simulator\n"
        "\n"
        "  --config <path>          humanoid config JSON (defaults are built in)\n"
        "  --dump-config <path>     write the effective config as JSON and exit\n"
        "  --motors                 start with the joint motors holding the rest pose\n"
        "  --width/--height <n>     window size\n"
        "  --seed <n>               RNG seed\n"
        "  --capture <path.png>     simulate, screenshot, and exit (no interaction)\n"
        "  --capture-time <sec>     simulated time before the first capture (default 2)\n"
        "  --capture-count <n>      number of captures (default 1)\n"
        "  --capture-interval <sec> simulated time between captures (default 0.5)\n"
        "  --hidden                 do not show the window (for capture on a build machine)\n"
        "  --check-gl               probe for a GL 3.3 core context and exit\n"
        "\n"
        "controls\n"
        "  space  pause          .  single step        r  reset\n"
        "  m      motors on/off  b  drop a ball        x  shove the pelvis\n"
        "  o      outlines       j  joint targets      v  velocities\n"
        "  c      contacts       t  trails             k  centre of mass\n"
        "  f      camera follow  p  screenshot\n"
        "  mouse  left-drag a limb, right-click to shove, wheel to zoom\n");
}

bool parseOptions(int argc, char** argv, Options& out, bool& probeOnly, bool& showHelp) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](Real& value) {
            if (i + 1 < argc) value = static_cast<Real>(std::atof(argv[++i]));
        };
        auto nextInt = [&](int& value) {
            if (i + 1 < argc) value = std::atoi(argv[++i]);
        };
        if (arg == "--help" || arg == "-h") { showHelp = true; return true; }
        else if (arg == "--check-gl") { probeOnly = true; }
        else if (arg == "--motors") { out.motorsOn = true; }
        else if (arg == "--hidden") { out.hidden = true; }
        else if (arg == "--width") { nextInt(out.width); }
        else if (arg == "--height") { nextInt(out.height); }
        else if (arg == "--seed" && i + 1 < argc) { out.seed = std::strtoull(argv[++i], nullptr, 10); }
        else if (arg == "--config" && i + 1 < argc) { out.configPath = argv[++i]; }
        else if (arg == "--dump-config" && i + 1 < argc) { out.dumpConfigPath = argv[++i]; }
        else if (arg == "--capture" && i + 1 < argc) { out.capturePath = argv[++i]; }
        else if (arg == "--capture-time") { next(out.captureTime); }
        else if (arg == "--capture-count") { nextInt(out.captureCount); }
        else if (arg == "--capture-interval") { next(out.captureInterval); }
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

int probeGl() {
    if (!glfwInit()) {
        const char* error = nullptr;
        glfwGetError(&error);
        std::printf("glfwInit failed: %s\n", error ? error : "unknown");
        return 1;
    }
    std::printf("GLFW %s\n", glfwGetVersionString());
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(64, 64, "probe", nullptr, nullptr);
    if (!window) {
        const char* error = nullptr;
        glfwGetError(&error);
        std::printf("no GL 3.3 core context: %s\n", error ? error : "unknown");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    std::printf("GL 3.3 core context OK\n");
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// Everything the interactive loop mutates, kept together so reset() is one call
// and cannot forget a field.
struct Simulation {
    World2D world;
    Humanoid2D figure;
    Humanoid2DConfig config;
    Rng rng;
    Trail headTrail{300};
    Trail comTrail{300};
    std::vector<int32_t> props;  // balls thrown at the figure
    Real elapsed = 0;

    void build(const Humanoid2DConfig& cfg, uint64_t seed) {
        config = cfg;
        rng.seedWith(seed);
        world.addHalfSpace(HalfSpace{Vec2(0, 1), 0, Real(1.0), Real(0)});
        world.addHalfSpace(HalfSpace{Vec2(1, 0), Real(-12), Real(0.5), Real(0.1)});
        world.addHalfSpace(HalfSpace{Vec2(-1, 0), Real(-12), Real(0.5), Real(0.1)});
        figure.build(world, config);
        reset();
    }

    void reset() {
        figure.reset(world, config.links[kPelvis].restPosition);
        for (const int32_t prop : props) {
            RigidBody2D& body = world.body(prop);
            body.position = Vec2(0, Real(-50));  // parked out of the way
            body.velocity = Vec2(0, 0);
            body.angularVelocity = 0;
        }
        headTrail.clear();
        comTrail.clear();
        elapsed = 0;
    }

    void dropBall(const Vec2& from, const Vec2& velocity) {
        RigidBody2D ball;
        ball.position = from;
        ball.setCapsuleWithMass(Real(0.12), 0, Real(4));
        ball.velocity = velocity;
        ball.friction = Real(0.6);
        ball.restitution = Real(0.35);
        ball.collisionGroup = 0;  // props do collide with the figure
        props.push_back(world.addBody(ball));
    }

    void step() {
        world.step(kPhysicsDt);
        elapsed += kPhysicsDt;
    }
};

}  // namespace

int main(int argc, char** argv) {
    Options options;
    bool probeOnly = false;
    bool showHelp = false;
    if (!parseOptions(argc, argv, options, probeOnly, showHelp)) return 2;
    if (showHelp) {
        printUsage();
        return 0;
    }
    if (probeOnly) return probeGl();

    Humanoid2DConfig config = Humanoid2DConfig::defaults();
    if (!options.configPath.empty()) {
        std::string error;
        config = Humanoid2DConfig::loadFile(options.configPath, &error);
        if (!error.empty()) {
            std::fprintf(stderr, "config problem: %s\n", error.c_str());
            return 2;
        }
        std::printf("loaded %s\n", options.configPath.c_str());
    }
    const std::string configProblem = config.validate();
    if (!configProblem.empty()) {
        std::fprintf(stderr, "invalid humanoid config: %s\n", configProblem.c_str());
        return 2;
    }

    if (!options.dumpConfigPath.empty()) {
        if (!config.toJson().writeFile(options.dumpConfigPath)) {
            std::fprintf(stderr, "could not write %s\n", options.dumpConfigPath.c_str());
            return 1;
        }
        std::printf("wrote %s\n", options.dumpConfigPath.c_str());
        return 0;
    }

    const bool capturing = !options.capturePath.empty();

    if (options.hidden || capturing) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    Window window;
    if (!window.create(options.width, options.height, "aibackflip - 2D humanoid")) return 1;
    // The hint has to be set before glfwCreateWindow, which Window::create does
    // internally, so re-apply visibility explicitly for the capture path.
    if (options.hidden || capturing) glfwHideWindow(window.handle());

    Renderer2D renderer;
    if (!renderer.initialize()) {
        std::fprintf(stderr, "renderer initialisation failed\n");
        return 1;
    }

    Simulation sim;
    sim.build(config, options.seed);
    sim.figure.setMotorsEnabled(sim.world, options.motorsOn);

    Camera2D camera;
    camera.center = Vec2(0, Real(0.9));
    camera.halfHeight = Real(1.3);

    DebugDrawOptions draw;
    bool paused = false;
    bool singleStep = false;
    bool motorsOn = options.motorsOn;
    bool followCamera = true;
    int screenshotIndex = 0;

    // Capture schedule, in simulated seconds.
    int capturesTaken = 0;
    Real nextCaptureAt = options.captureTime;

    double previousTime = glfwGetTime();
    Real accumulator = 0;
    int frame = 0;

    while (window.isOpen()) {
        window.pollEvents();
        const Input& input = window.input();

        camera.aspect = window.aspect();

        // ---- input ----
        if (!capturing) {
            if (input.keyPressed[GLFW_KEY_ESCAPE]) window.requestClose();
            if (input.keyPressed[GLFW_KEY_SPACE]) paused = !paused;
            if (input.keyPressed[GLFW_KEY_PERIOD]) singleStep = true;
            if (input.keyPressed[GLFW_KEY_R]) sim.reset();
            if (input.keyPressed[GLFW_KEY_M]) {
                motorsOn = !motorsOn;
                sim.figure.setMotorsEnabled(sim.world, motorsOn);
                sim.figure.holdRestPose(sim.world);
            }
            if (input.keyPressed[GLFW_KEY_O]) draw.outlines = !draw.outlines;
            if (input.keyPressed[GLFW_KEY_J]) draw.jointTargets = !draw.jointTargets;
            if (input.keyPressed[GLFW_KEY_V]) draw.velocities = !draw.velocities;
            if (input.keyPressed[GLFW_KEY_C]) draw.contacts = !draw.contacts;
            if (input.keyPressed[GLFW_KEY_T]) draw.trail = !draw.trail;
            if (input.keyPressed[GLFW_KEY_K]) draw.centerOfMass = !draw.centerOfMass;
            if (input.keyPressed[GLFW_KEY_F]) followCamera = !followCamera;
            if (input.keyPressed[GLFW_KEY_P]) {
                char path[128];
                std::snprintf(path, sizeof(path), "screenshot_%03d.png", screenshotIndex++);
                if (window.saveScreenshot(path)) std::printf("wrote %s\n", path);
            }
            if (input.keyPressed[GLFW_KEY_B]) {
                sim.dropBall(Vec2(Real(-3), Real(1.6)), Vec2(Real(7), Real(1)));
            }
            if (input.keyPressed[GLFW_KEY_X]) {
                RigidBody2D& pelvis = sim.figure.link(sim.world, kPelvis);
                pelvis.applyImpulse(Vec2(Real(60), 0), Vec2(0, 0));
            }

            if (input.scroll != Real(0)) {
                camera.halfHeight =
                    clamp(camera.halfHeight * std::pow(Real(0.9), input.scroll), Real(0.3),
                          Real(12));
            }

            const Vec2 worldMouse =
                camera.screenToWorld(input.mouseScreen, window.width(), window.height());
            if (input.mousePressed[GLFW_MOUSE_BUTTON_LEFT]) {
                const int32_t picked = sim.world.pickBody(worldMouse, Real(0.12));
                if (picked >= 0) sim.world.grab(picked, worldMouse);
            }
            if (input.mouseDown[GLFW_MOUSE_BUTTON_LEFT]) {
                sim.world.mouse().target = worldMouse;
            }
            if (input.mouseReleased[GLFW_MOUSE_BUTTON_LEFT]) sim.world.releaseGrab();
            if (input.mousePressed[GLFW_MOUSE_BUTTON_RIGHT]) {
                const int32_t picked = sim.world.pickBody(worldMouse, Real(0.2));
                if (picked >= 0) {
                    RigidBody2D& body = sim.world.body(picked);
                    body.applyImpulseAtPoint(Vec2(Real(40), Real(10)), worldMouse);
                }
            }
        }

        // ---- fixed-step physics ----
        const double now = glfwGetTime();
        Real frameTime = static_cast<Real>(now - previousTime);
        previousTime = now;
        // In capture mode the simulation advances by wall-clock-independent
        // steps, so a slow machine produces the same frame as a fast one.
        if (capturing) frameTime = kPhysicsDt * Real(8);
        frameTime = std::min(frameTime, kMaxFrameTime);

        if (!paused || singleStep) {
            accumulator += singleStep ? kPhysicsDt : frameTime;
            singleStep = false;
            int steps = 0;
            while (accumulator >= kPhysicsDt && steps < 64) {
                sim.step();
                ++steps;
                accumulator -= kPhysicsDt;
            }
            if (steps >= 64) accumulator = 0;  // gave up catching up; drop the debt
        }

        sim.headTrail.push(sim.figure.link(sim.world, kHead).position);
        sim.comTrail.push(sim.figure.centerOfMass(sim.world));

        if (followCamera) {
            const Vec2 target = sim.figure.centerOfMass(sim.world);
            camera.center.x += (target.x - camera.center.x) * Real(0.08);
            camera.center.y += (Real(0.9) - camera.center.y) * Real(0.02);
        }

        // ---- render ----
        renderer.beginFrame(camera, palette::background);
        renderer.groundAndGrid(camera, 0, Real(0.5));
        if (draw.trail) {
            drawTrail(renderer, sim.comTrail, palette::com);
            drawTrail(renderer, sim.headTrail, palette::trail);
        }
        for (const int32_t prop : sim.props) {
            const RigidBody2D& body = sim.world.body(prop);
            renderer.capsule(body.position, body.angle, body.radius, body.halfLength,
                             palette::accent);
        }
        drawHumanoid(renderer, sim.world, sim.figure, camera, draw);
        drawWorld(renderer, sim.world, camera, draw);
        drawMouseSpring(renderer, sim.world, camera);
        renderer.endFrame();

        // ---- capture ----
        if (capturing && sim.elapsed >= nextCaptureAt) {
            std::string path = options.capturePath;
            if (options.captureCount > 1) {
                const size_t dot = path.rfind('.');
                char suffix[16];
                std::snprintf(suffix, sizeof(suffix), "_%02d", capturesTaken);
                path = (dot == std::string::npos) ? path + suffix
                                                  : path.substr(0, dot) + suffix + path.substr(dot);
            }
            if (window.saveScreenshot(path)) {
                std::printf("captured %s at t=%.2fs  pelvis=%.3f  head=%.3f  anchor_err=%.2gm\n",
                            path.c_str(), double(sim.elapsed),
                            double(sim.figure.link(sim.world, kPelvis).position.y),
                            double(sim.figure.headHeight(sim.world)),
                            double(sim.world.stats().maxJointAnchorError));
            } else {
                std::fprintf(stderr, "failed to write %s\n", path.c_str());
                return 1;
            }
            ++capturesTaken;
            nextCaptureAt += options.captureInterval;
            if (capturesTaken >= options.captureCount) break;
        }

        window.present();

        if (++frame % 15 == 0 && !capturing) {
            char title[256];
            std::snprintf(title, sizeof(title),
                          "aibackflip | t=%.1fs | pelvis %.2fm | contacts %d | anchor err %.1e m | "
                          "motors %s%s",
                          double(sim.elapsed),
                          double(sim.figure.link(sim.world, kPelvis).position.y),
                          sim.world.stats().contactCount,
                          double(sim.world.stats().maxJointAnchorError), motorsOn ? "on" : "off",
                          paused ? " | PAUSED" : "");
            window.setTitle(title);
        }
    }

    if (sim.world.stats().unstable) {
        std::fprintf(stderr, "simulation went unstable\n");
        return 1;
    }
    return 0;
}
