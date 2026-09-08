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
    int renderEnv = 0;
    int width = 1280;
    int height = 720;
    double statusIntervalSeconds = 5;
    // Screenshots the rendered environment once the client has driven it for
    // this many control steps, then exits. Used to capture what a policy is
    // actually doing without a person watching.
    std::string capturePath;
    int captureAfterSteps = 120;
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
        "  --quiet             suppress the periodic status line\n"
        "  --verbose           log connections and resets\n"
        "  --exit-on-bye       shut down when the client disconnects (default: keep\n"
        "                      serving, so a restarted trainer can reconnect)\n");
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
        else if (arg == "--capture" && i + 1 < argc) {
            out.capturePath = argv[++i];
            out.render = true;
        }
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

void printStatus(const net::EnvServer& server, double elapsedSeconds, uint64_t stepsAtLastPrint) {
    const net::EnvServer::Stats& stats = server.stats();
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

int runHeadless(net::EnvServer& server, const Options& options) {
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

    while (window.isOpen() && server.isRunning() && !server.shouldStop()) {
        window.pollEvents();
        if (window.input().keyPressed[GLFW_KEY_ESCAPE]) window.requestClose();
        if (window.input().keyPressed[GLFW_KEY_T]) draw.trail = !draw.trail;
        if (window.input().keyPressed[GLFW_KEY_C]) draw.contacts = !draw.contacts;
        if (window.input().scroll != Real(0)) {
            camera.halfHeight = clamp(camera.halfHeight * std::pow(Real(0.9), window.input().scroll),
                                      Real(0.3), Real(12));
        }
        camera.aspect = window.aspect();

        // Serve for most of a frame, then draw. The trainer sets the pace; this
        // loop only has to avoid starving it while the window is open.
        server.poll(8, 256);

        Env2D& env = server.batch().env(index);
        const Vec2 com = env.figure().centerOfMass(env.world());
        comTrail.push(com);
        camera.center.x += (com.x - camera.center.x) * Real(0.08);

        renderer.beginFrame(camera, palette::background);
        renderer.groundAndGrid(camera, 0, Real(0.5));
        if (draw.trail) drawTrail(renderer, comTrail, palette::com);
        drawHumanoid(renderer, env.world(), env.figure(), camera, draw);
        drawWorld(renderer, env.world(), camera, draw);
        renderer.endFrame();

        if (capturing && server.stats().step >= static_cast<uint32_t>(options.captureAfterSteps)) {
            if (!window.saveScreenshot(options.capturePath)) {
                std::fprintf(stderr, "failed to write %s\n", options.capturePath.c_str());
                return 1;
            }
            std::printf("captured %s after %u control steps (env %d, pelvis %.3f m, "
                        "episode step %d)\n",
                        options.capturePath.c_str(), server.stats().step, index,
                        double(env.figure().link(env.world(), kPelvis).position.y),
                        env.episodeStep());
            return 0;
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

int main(int argc, char** argv) {
    Options options;
    bool showHelp = false;
    if (!parseOptions(argc, argv, options, showHelp)) return 2;
    if (showHelp) {
        printUsage();
        return 0;
    }

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
