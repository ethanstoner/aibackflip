// The 3D render loop for aibf_env, kept out of envserver.cpp so the 2D path is
// not buried under a second one. Included only when AIBF_WITH_RENDERER is set.
//
// Draws whichever environment the trainer is currently stepping, so what is on
// screen is the same simulation being learned from rather than a replay.

namespace {

// The figure's limb palette, matching the 2D renderer so the two views read as
// the same character: pale head, mint arms, blue legs, orange feet.
Vec4 linkColour3D(int linkId) {
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
        default: return Vec4{Real(0.43), Real(0.61), Real(0.91), Real(1)};  // legs
    }
}

}  // namespace

int runRendered3D(net::EnvServer3D& server, const Options& options) {
    const bool capturing = !options.capturePath.empty();
    if (capturing) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    Window window;
    if (!window.create(options.width, options.height, "aibf_env - serving 3D")) return 1;
    if (capturing) glfwHideWindow(window.handle());

    Renderer3D renderer;
    if (!renderer.initialize()) {
        std::fprintf(stderr, "renderer initialisation failed: %s\n", renderer.lastError().c_str());
        return 1;
    }

    Camera3D camera;
    const int index = (options.renderEnv >= 0 && options.renderEnv < server.batch().size())
                          ? options.renderEnv
                          : 0;
    int capturesTaken = 0;
    int nextCaptureStep = options.captureAfterSteps;

    while (window.isOpen() && server.isRunning() && !server.shouldStop()) {
        window.pollEvents();
        const Input& input = window.input();
        if (input.keyPressed[GLFW_KEY_ESCAPE]) window.requestClose();
        if (input.scroll != Real(0)) {
            camera.distance =
                clamp(camera.distance * std::pow(Real(0.9), input.scroll), Real(1), Real(30));
        }
        // Orbit with the left mouse button. A 3D figure has a back, and a fixed
        // camera hides half of what it is doing.
        if (input.mouseDown[GLFW_MOUSE_BUTTON_LEFT]) {
            camera.yaw -= input.mouseDelta.x * Real(0.008);
            camera.pitch = clamp(camera.pitch + input.mouseDelta.y * Real(0.008), Real(-0.4),
                                 Real(1.3));
        }
        camera.aspect = window.aspect();

        // Serve for most of a frame, then draw. The trainer sets the pace; this
        // loop only has to avoid starving it while the window is open.
        server.poll(8, 256);

        Env3D& env = server.batch().env(index);

        if (input.keyPressed[GLFW_KEY_X] || input.keyPressed[GLFW_KEY_Z]) {
            const Real direction = input.keyPressed[GLFW_KEY_X] ? Real(1) : Real(-1);
            const Real magnitude = input.shift ? Real(120) : Real(45);
            // Off centre, so the shove carries torque and not just momentum.
            env.push(Vec3(direction * magnitude, 0, 0), Real(0.25));
            std::printf("pushed env %d with %.0f N s\n", index, double(direction * magnitude));
            std::fflush(stdout);
        }
        if (input.keyPressed[GLFW_KEY_C]) {
            // Sideways, the direction the figure has no ankle roll to answer.
            const Real magnitude = input.shift ? Real(120) : Real(45);
            env.push(Vec3(0, 0, magnitude), Real(0.25));
            std::printf("pushed env %d sideways with %.0f N s\n", index, double(magnitude));
            std::fflush(stdout);
        }

        const Vec3 com = env.figure().centerOfMass(env.world());
        camera.target = camera.target + (Vec3(com.x, Real(0.9), com.z) - camera.target) * Real(0.08);

        renderer.beginFrame(camera, Vec4{Real(0.09), Real(0.10), Real(0.13), Real(1)});
        renderer.ground(Real(40), Vec4{Real(0.16), Real(0.17), Real(0.21), Real(1)},
                        Vec4{Real(0.28), Real(0.31), Real(0.38), Real(1)});
        for (int i = 0; i < kLinkCount; ++i) {
            const RigidBody3D& body = env.figure().link(env.world(), i);
            renderer.capsule(body.position, body.orientation, body.radius, body.halfLength,
                             linkColour3D(i));
        }
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
                        "episode step %d)\n",
                        path.c_str(), server.stats().step, index,
                        double(env.figure().link(env.world(), kPelvis).position.y),
                        double(env.figure().link(env.world(), kHead).position.y),
                        env.episodeStep());
            std::fflush(stdout);
            ++capturesTaken;
            if (capturesTaken >= options.captureCount) break;
            nextCaptureStep += options.captureIntervalSteps;
        }

        window.present();
    }

    renderer.shutdown();
    return 0;
}
