// Environment host: serves a batch of humanoid environments to a Python trainer
// over UDP.
//
// One binary, two modes. `--headless` (the default) never touches OpenGL and is
// what training runs against. `--render` opens a window showing one environment
// while serving exactly the same batch, so what you watch is what is being
// trained rather than a separate replay path.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "env/EnvBatch.h"
#include "net/EnvServer.h"

#ifdef AIBF_WITH_RENDERER
#include <GLFW/glfw3.h>

#include "engine/DebugDraw.h"
#include "engine/Renderer2D.h"
#include "engine/Window.h"
#endif

using namespace aibf;

namespace {

struct Options {
    net::EnvServer::Options server;
    std::string configPath;
    std::string dumpConfigPath;
    bool render = false;
    // Serves the 3D figure instead of the 2D one. Same protocol, same batch,
    // same server: only the environment type differs, so Python needs no change
    // beyond reading the dimensions out of the SPEC handshake, which it already
    // does rather than hardcoding them.
    bool threeD = false;
    int renderEnv = 0;
    int width = 1280;
    int height = 720;
    double statusIntervalSeconds = 5;
    // Screenshots the rendered environment once the client has driven it for
    // this many control steps, then exits. Used to capture what a policy is
    // actually doing without a person watching.
    std::string capturePath;
    int captureAfterSteps = 120;
    int captureCount = 1;
    int captureIntervalSteps = 60;
};

void printUsage() {
    std::printf(
        "aibf_env - humanoid environment server\n"
        "\n"
        "  --port <n>          UDP port to bind (default 51234)\n"
        "  --bind <addr>       bind address (default 127.0.0.1)\n"
        "  --envs <n>          environments in the batch (default 25)\n"
        "  --seed <n>          base RNG seed (default 1)\n"
        "  --config <path>     environment config JSON\n"
        "  --dump-config <p>   write the effective config as JSON and exit\n"
        "  --headless          no window at all (default)\n"
        "  --render            open a window showing one environment while serving\n"
        "  --render-env <i>    which environment to draw (default 0)\n"
        "  --capture <path>    screenshot the rendered environment and exit\n"
        "  --capture-after <n> control steps to wait before the first capture\n"
        "  --capture-count <n> number of captures (default 1)\n"
        "  --capture-every <n> control steps between captures (default 60)\n"
        "  --quiet             suppress the periodic status line\n"
        "  --verbose           log connections and resets\n"
        "  --exit-on-bye       shut down when the client disconnects (default: keep\n"
        "                      serving, so a restarted trainer can reconnect)\n"
        "\n"
        "render-mode controls\n"
        "  x / z    shove the pelvis right / left (hold shift for a hard shove)\n"
        "  b        throw a ball at it\n"
        "  mouse    left-drag a limb\n"
        "  t c j v  trails, contacts, joint targets, velocities\n"
        "  wheel    zoom\n");
}

bool parseOptions(int argc, char** argv, Options& out, bool& showHelp) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto nextInt = [&](int& value) {
            if (i + 1 < argc) value = std::atoi(argv[++i]);
        };
        if (arg == "--help" || arg == "-h") { showHelp = true; return true; }
        else if (arg == "--headless") { out.render = false; }
        else if (arg == "--render") { out.render = true; }
        else if (arg == "--quiet") { out.statusIntervalSeconds = 0; }
        else if (arg == "--verbose") { out.server.verbose = true; }
        else if (arg == "--exit-on-bye") { out.server.exitOnBye = true; }
        else if (arg == "--capture-after") { nextInt(out.captureAfterSteps); }
        else if (arg == "--capture-count") { nextInt(out.captureCount); }
        else if (arg == "--capture-every") { nextInt(out.captureIntervalSteps); }
        else if (arg == "--capture" && i + 1 < argc) {
            out.capturePath = argv[++i];
            out.render = true;
        }
        else if (arg == "--3d") { out.threeD = true; }
        else if (arg == "--render-env") { nextInt(out.renderEnv); }
        else if (arg == "--width") { nextInt(out.width); }
        else if (arg == "--height") { nextInt(out.height); }
        else if (arg == "--envs") { nextInt(out.server.numEnvs); }
        else if (arg == "--port" && i + 1 < argc) {
            out.server.port = static_cast<uint16_t>(std::atoi(argv[++i]));
        } else if (arg == "--bind" && i + 1 < argc) {
            out.server.bindAddress = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            out.server.seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--config" && i + 1 < argc) {
            out.configPath = argv[++i];
        } else if (arg == "--dump-config" && i + 1 < argc) {
            out.dumpConfigPath = argv[++i];
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

template <typename Server>
void printStatus(const Server& server, double elapsedSeconds, uint64_t stepsAtLastPrint) {
    const auto& stats = server.stats();
    const double rate = elapsedSeconds > 0
                            ? double(stats.controlStepsServed - stepsAtLastPrint) / elapsedSeconds
                            : 0.0;
    std::printf(
        "step %-8u  %7.0f control-steps/s  %8.0f env-steps/s  episodes %-7llu  "
        "dup %llu  stale %llu  bad %llu\n",
        stats.step, rate, rate * double(server.batch().size()),
        static_cast<unsigned long long>(stats.episodesFinished),
        static_cast<unsigned long long>(stats.duplicateRequestsAnswered),
        static_cast<unsigned long long>(stats.stalePacketsDropped),
        static_cast<unsigned long long>(stats.malformedPacketsDropped));
    std::fflush(stdout);
}

template <typename Server>
int runHeadless(Server& server, const Options& options) {
    std::printf("serving %d environments on %s:%u (session %u)\n", server.batch().size(),
                options.server.bindAddress.c_str(), server.port(), server.session());
    std::printf("observation %d, action %d, reward terms %d\n", EnvBatch::observationDim(),
                EnvBatch::actionDim(), EnvBatch::rewardTermCount());
    std::fflush(stdout);

    auto lastPrint = std::chrono::steady_clock::now();
    uint64_t stepsAtLastPrint = 0;

    while (server.isRunning() && !server.shouldStop()) {
        server.poll(100);
        if (options.statusIntervalSeconds > 0) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - lastPrint).count();
            if (elapsed >= options.statusIntervalSeconds) {
                printStatus(server, elapsed, stepsAtLastPrint);
                stepsAtLastPrint = server.stats().controlStepsServed;
                lastPrint = now;
            }
        }
    }
    std::printf("stopping after %llu control steps\n",
                static_cast<unsigned long long>(server.stats().controlStepsServed));
    return 0;
}

#ifdef AIBF_WITH_RENDERER
int runRendered(net::EnvServer& server, const Options& options) {
    const bool capturing = !options.capturePath.empty();
    if (capturing) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    Window window;
    if (!window.create(options.width, options.height, "aibf_env - serving")) return 1;
    if (capturing) glfwHideWindow(window.handle());
    Renderer2D renderer;
    if (!renderer.initialize()) {
        std::fprintf(stderr, "renderer initialisation failed\n");
        return 1;
    }

    Camera2D camera;
    camera.center = Vec2(0, Real(0.9));
    camera.halfHeight = Real(1.3);
    DebugDrawOptions draw;
    draw.trail = false;

    const int index = clamp(Real(options.renderEnv), Real(0),
                            Real(server.batch().size() - 1)) > Real(0)
                          ? options.renderEnv
                          : 0;
    Trail comTrail(240);
    int frame = 0;
    int capturesTaken = 0;
    int nextCaptureStep = options.captureAfterSteps;

    // Props thrown at the figure. Created once and recycled, so a long session
    // does not accumulate bodies in the world.
    std::vector<int32_t> props;
    size_t nextProp = 0;

    while (window.isOpen() && server.isRunning() && !server.shouldStop()) {
        window.pollEvents();
        const Input& input = window.input();
        if (input.keyPressed[GLFW_KEY_ESCAPE]) window.requestClose();
        if (input.keyPressed[GLFW_KEY_T]) draw.trail = !draw.trail;
        if (input.keyPressed[GLFW_KEY_C]) draw.contacts = !draw.contacts;
        if (input.keyPressed[GLFW_KEY_J]) draw.jointTargets = !draw.jointTargets;
        if (input.keyPressed[GLFW_KEY_V]) draw.velocities = !draw.velocities;
        if (input.scroll != Real(0)) {
            camera.halfHeight =
                clamp(camera.halfHeight * std::pow(Real(0.9), input.scroll), Real(0.3), Real(12));
        }
        camera.aspect = window.aspect();

        // Serve for most of a frame, then draw. The trainer sets the pace; this
        // loop only has to avoid starving it while the window is open.
        server.poll(8, 256);

        Env2D& env = server.batch().env(index);

        // ---- disturbances, for watching a policy recover ----
        //
        // These act on the live environment the trainer is stepping, so what is
        // being poked is the same simulation being learned from.
        const Vec2 worldMouse =
            camera.screenToWorld(input.mouseScreen, window.width(), window.height());
        if (input.mousePressed[GLFW_MOUSE_BUTTON_LEFT]) {
            const int32_t picked = env.world().pickBody(worldMouse, Real(0.12));
            if (picked >= 0) env.world().grab(picked, worldMouse);
        }
        if (input.mouseDown[GLFW_MOUSE_BUTTON_LEFT]) env.world().mouse().target = worldMouse;
        if (input.mouseReleased[GLFW_MOUSE_BUTTON_LEFT]) env.world().releaseGrab();

        if (input.keyPressed[GLFW_KEY_X] || input.keyPressed[GLFW_KEY_Z]) {
            const Real direction = input.keyPressed[GLFW_KEY_X] ? Real(1) : Real(-1);
            const Real magnitude = input.shift ? Real(120) : Real(45);
            env.push(Vec2(direction * magnitude, 0));
            std::printf("pushed env %d with %.0f N s\n", index, double(direction * magnitude));
            std::fflush(stdout);
        }

        if (input.keyPressed[GLFW_KEY_B]) {
            const Vec2 from = env.figure().centerOfMass(env.world()) + Vec2(Real(-3), Real(0.8));
            const Vec2 velocity(Real(9), Real(0.5));
            if (props.size() < 6) {
                RigidBody2D ball;
                ball.setCapsuleWithMass(Real(0.12), 0, Real(4));
                ball.friction = Real(0.6);
                ball.restitution = Real(0.35);
                ball.collisionGroup = 0;  // props do collide with the figure
                ball.position = from;
                ball.velocity = velocity;
                props.push_back(env.world().addBody(ball));
            } else {
                RigidBody2D& ball = env.world().body(props[nextProp % props.size()]);
                ball.position = from;
                ball.velocity = velocity;
                ball.angularVelocity = 0;
                ++nextProp;
            }
        }
        const Vec2 com = env.figure().centerOfMass(env.world());
        comTrail.push(com);
        camera.center.x += (com.x - camera.center.x) * Real(0.08);

        renderer.beginFrame(camera, palette::background);
        renderer.groundAndGrid(camera, 0, Real(0.5));
        if (draw.trail) drawTrail(renderer, comTrail, palette::com);
        for (const int32_t prop : props) {
            const RigidBody2D& body = env.world().body(prop);
            renderer.capsule(body.position, body.angle, body.radius, body.halfLength,
                             palette::accent);
        }
        drawHumanoid(renderer, env.world(), env.figure(), camera, draw);
        drawWorld(renderer, env.world(), camera, draw);
        drawMouseSpring(renderer, env.world(), camera);
        renderer.endFrame();

        if (capturing && server.stats().step >= static_cast<uint32_t>(nextCaptureStep)) {
            std::string path = options.capturePath;
            if (options.captureCount > 1) {
                const size_t dot = path.rfind('.');
                char suffix[16];
                std::snprintf(suffix, sizeof(suffix), "_%02d", capturesTaken);
                path = (dot == std::string::npos) ? path + suffix
                                                  : path.substr(0, dot) + suffix + path.substr(dot);
            }
            if (!window.saveScreenshot(path)) {
                std::fprintf(stderr, "failed to write %s\n", path.c_str());
                return 1;
            }
            std::printf("captured %s at step %u (env %d, pelvis %.3f m, head %.3f m, "
                        "episode step %d, contacts %d)\n",
                        path.c_str(), server.stats().step, index,
                        double(env.figure().link(env.world(), kPelvis).position.y),
                        double(env.figure().headHeight(env.world())), env.episodeStep(),
                        (env.figure().footContact(env.world(), true) ? 1 : 0) +
                            (env.figure().footContact(env.world(), false) ? 1 : 0));
            std::fflush(stdout);
            ++capturesTaken;
            nextCaptureStep += options.captureIntervalSteps;
            if (capturesTaken >= options.captureCount) return 0;
        }

        window.present();

        if (++frame % 15 == 0) {
            char title[256];
            std::snprintf(title, sizeof(title),
                          "aibf_env | env %d/%d | step %u | ep step %d | pelvis %.2f | %s",
                          index, server.batch().size(), server.stats().step, env.episodeStep(),
                          double(env.figure().link(env.world(), kPelvis).position.y),
                          server.stats().clientKnown ? "client connected" : "waiting for client");
            window.setTitle(title);
        }
    }
    return 0;
}
#endif

}  // namespace

// The 3D path. Headless only: the renderer is still 2D, so a 3D session is
// trained and then inspected with the tools rather than watched live.
int run3D(const Options& options) {
    EnvConfig3D config = EnvConfig3D::defaults();
    if (!options.configPath.empty()) {
        std::string error;
        config = EnvConfig3D::loadFile(options.configPath, &error);
        if (!error.empty()) {
            std::fprintf(stderr, "config problem: %s\n", error.c_str());
            return 2;
        }
    }
    const std::string problem = config.validate();
    if (!problem.empty()) {
        std::fprintf(stderr, "invalid environment config: %s\n", problem.c_str());
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

    if (options.render) {
        std::fprintf(stderr, "the renderer is 2D only; --3d implies headless\n");
        return 2;
    }

    net::EnvServer3D server;
    if (!server.start(config, options.server)) {
        std::fprintf(stderr, "could not start the server: %s\n", server.lastError().c_str());
        return 1;
    }
    return runHeadless(server, options);
}

int main(int argc, char** argv) {
    Options options;
    bool showHelp = false;
    if (!parseOptions(argc, argv, options, showHelp)) return 2;
    if (showHelp) {
        printUsage();
        return 0;
    }

    if (options.threeD) return run3D(options);

    EnvConfig config = EnvConfig::defaults();
    if (!options.configPath.empty()) {
        std::string error;
        config = EnvConfig::loadFile(options.configPath, &error);
        if (!error.empty()) {
            std::fprintf(stderr, "config problem: %s\n", error.c_str());
            return 2;
        }
    }
    const std::string problem = config.validate();
    if (!problem.empty()) {
        std::fprintf(stderr, "invalid environment config: %s\n", problem.c_str());
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

    net::EnvServer server;
    if (!server.start(config, options.server)) {
        std::fprintf(stderr, "could not start the server: %s\n", server.lastError().c_str());
        return 1;
    }

#ifdef AIBF_WITH_RENDERER
    if (options.render) return runRendered(server, options);
#else
    if (options.render) {
        std::fprintf(stderr, "this build has no renderer; --render is unavailable\n");
        return 2;
    }
#endif
    return runHeadless(server, options);
}
