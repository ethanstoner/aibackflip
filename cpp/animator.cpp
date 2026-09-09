// Reference motion editor.
//
//   aibf_animator motions/backflip.json
//   aibf_animator motions/new.json --new
//
// Authoring is keyframe-based: scrub to a time, insert a keyframe, and rotate
// joints until the pose looks right. Two things here are not decoration.
//
// `g` snaps the root height so the lowest point of the figure rests exactly on
// the floor. The root height of a ground-contact pose is a consequence of the
// joint angles, not a free parameter, and being a centimetre out produces a
// reference that floats or sinks - one no policy can ever track.
//
// `p` runs a physics preview: the same humanoid, driven by its PD motors towards
// the reference, with gravity and contacts on and the root completely free. If
// the preview cannot get near the reference, no amount of training will either.
// It is a lower bound rather than a verdict - a learned policy anticipates where
// a tracking controller only reacts - but a motion the preview cannot begin to
// follow is a motion worth fixing before spending an hour training against it.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>

#include "engine/DebugDraw.h"
#include "engine/Renderer2D.h"
#include "engine/Window.h"
#include "humanoid/Humanoid2D.h"
#include "motion/Motion2D.h"
#include "physics/World2D.h"

using namespace aibf;

namespace {

constexpr Real kPhysicsDt = Real(1) / Real(240);
constexpr int kSubstepsPerControl = 4;
constexpr Real kScrubStep = Real(1) / Real(30);
constexpr Real kKeyframeEpsilon = Real(1e-3);

void printUsage() {
    std::printf(
        "aibf_animator - reference motion editor\n"
        "\n"
        "  aibf_animator <motion.json> [--new] [--config humanoid.json]\n"
        "\n"
        "timeline\n"
        "  space          play / pause            home / end   jump to start / end\n"
        "  left / right   scrub one frame         , / .        previous / next keyframe\n"
        "\n"
        "editing (only on a keyframe; press k to make one here)\n"
        "  up / down      select joint            a / d        rotate it (shift = coarse)\n"
        "  w / s          root height             q / e        root angle\n"
        "  g              drop the root so the figure rests on the floor\n"
        "  k              insert a keyframe here  delete       remove this keyframe\n"
        "  0              zero the selected joint  shift+0     zero every joint\n"
        "\n"
        "clip\n"
        "  l   toggle looping        i   linear / hermite interpolation\n"
        "  [ ] shift this keyframe earlier / later\n"
        "\n"
        "preview and view\n"
        "  p   physics preview on/off    r   restart the preview\n"
        "  o   onion skin                t   trail\n"
        "  ctrl+s  save                  esc quit\n");
}

// The keyframe the playhead is sitting on, or -1 between keyframes. Editing is
// deliberately restricted to keyframes: silently creating one on every nudge
// would fill a clip with keys the author never meant to make.
int keyframeAt(const Motion2D& motion, Real time) {
    for (int i = 0; i < motion.frameCount(); ++i) {
        if (std::abs(motion.keyframes[static_cast<size_t>(i)].time - time) < kKeyframeEpsilon) {
            return i;
        }
    }
    return -1;
}

const char* jointName(const Humanoid2DConfig& config, int joint) {
    return config.joints[static_cast<size_t>(joint)].name.c_str();
}

// The reference figure: filled with the usual limb colours so a pose is
// actually readable, plus an outline so it stays distinguishable from the solid
// physics preview drawn underneath it. Outline-only is unreadable once the
// figure is inverted and tucked, which is exactly when it needs reading.
void drawReference(Renderer2D& renderer, const Humanoid2D& figure, const MotionPose& pose,
                   const Color& outline, Real alpha, bool filled) {
    const Humanoid2DConfig& config = figure.config();
    std::vector<Vec2> positions(config.links.size());
    std::vector<Real> angles(config.links.size());
    figure.forwardKinematics(pose.rootPosition, pose.rootAngle, pose.jointAngles.data(),
                             positions.data(), angles.data());

    if (filled) {
        static constexpr int kDrawOrder[] = {
            kUpperLegR, kLowerLegR, kFootR, kUpperArmR, kLowerArmR,
            kPelvis,    kChest,     kHead,
            kUpperLegL, kLowerLegL, kFootL, kUpperArmL, kLowerArmL,
        };
        for (const int i : kDrawOrder) {
            const size_t index = static_cast<size_t>(i);
            renderer.capsule(positions[index], angles[index], config.links[index].radius,
                             config.links[index].halfLength,
                             linkColor(i).withAlpha(alpha * Real(0.85)));
        }
    }
    for (size_t i = 0; i < config.links.size(); ++i) {
        renderer.capsuleOutline(positions[i], angles[i], config.links[i].radius,
                                config.links[i].halfLength, outline.withAlpha(alpha));
    }
}

// Timeline drawn in world space, pinned to the bottom of the view so it tracks
// the camera without needing a second projection.
void drawTimeline(Renderer2D& renderer, const Camera2D& camera, const Motion2D& motion,
                  Real time, int activeKeyframe) {
    if (motion.empty()) return;

    const Real halfWidth = camera.halfWidth();
    const Real left = camera.center.x - halfWidth * Real(0.9);
    const Real right = camera.center.x + halfWidth * Real(0.9);
    const Real y = camera.center.y - camera.halfHeight * Real(0.86);
    const Real tick = camera.halfHeight * Real(0.035);

    renderer.thickLine(Vec2(left, y), Vec2(right, y), tick * Real(0.25),
                       palette::grid.withAlpha(Real(0.9)));

    const Real start = motion.keyframes.front().time;
    const Real span = motion.duration() > Real(0) ? motion.duration() : Real(1);
    auto toX = [&](Real t) { return left + (right - left) * ((t - start) / span); };

    for (int i = 0; i < motion.frameCount(); ++i) {
        const Real x = toX(motion.keyframes[static_cast<size_t>(i)].time);
        const bool active = (i == activeKeyframe);
        const Color color = active ? palette::com : palette::accent;
        const Real height = active ? tick * Real(1.6) : tick;
        renderer.thickLine(Vec2(x, y - height), Vec2(x, y + height), tick * Real(0.3), color);
    }

    const Real playhead = toX(time);
    renderer.thickLine(Vec2(playhead, y - tick * Real(2.2)), Vec2(playhead, y + tick * Real(2.2)),
                       tick * Real(0.22), palette::contact);
}

struct Preview {
    World2D world;
    Humanoid2D figure;
    bool active = false;
    Real time = 0;

    void build(const Humanoid2DConfig& config) {
        world = World2D{};
        world.addHalfSpace(HalfSpace{Vec2(0, 1), 0, Real(1.0), Real(0)});
        figure.build(world, config);
        figure.setMotorsEnabled(world, true);
    }

    void restart(const Motion2D& motion, Real startTime) {
        const MotionPose pose = motion.sample(startTime);
        const MotionPose velocity = motion.sampleVelocity(startTime);
        figure.setPoseAndVelocity(world, pose.rootPosition, pose.rootAngle,
                                  pose.jointAngles.data(), velocity.rootPosition,
                                  velocity.rootAngle, velocity.jointAngles.data());
        for (int j = 0; j < kJointCount; ++j) {
            RevoluteJoint2D& joint = figure.joint(world, j);
            joint.targetAngle = pose.jointAngles[static_cast<size_t>(j)];
            joint.resetAccumulators();
        }
        world.clearContactCache();
        world.resetStats();
        time = startTime;
    }

    // One control step: retarget the motors at the reference, then simulate.
    void step(const Motion2D& motion) {
        const MotionPose pose = motion.sample(time);
        for (int j = 0; j < kJointCount; ++j) {
            figure.setJointTarget(world, j, pose.jointAngles[static_cast<size_t>(j)]);
        }
        for (int s = 0; s < kSubstepsPerControl; ++s) world.step(kPhysicsDt);
        time += Real(kSubstepsPerControl) * kPhysicsDt;
    }

    // RMS joint error against the reference: the number that says whether this
    // motion is trackable at all.
    Real trackingError(const Motion2D& motion) const {
        const MotionPose pose = motion.sample(time);
        Real sum = 0;
        for (int j = 0; j < kJointCount; ++j) {
            const Real error = wrapAngle(figure.jointAngle(world, j) -
                                         pose.jointAngles[static_cast<size_t>(j)]);
            sum += error * error;
        }
        return std::sqrt(sum / Real(kJointCount));
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string motionPath;
    std::string configPath;
    std::string capturePath;
    bool createNew = false;
    // Capture mode: scrub to a phase, optionally run the physics preview for a
    // while, screenshot and exit. Lets a motion be checked - and the preview's
    // tracking error measured - without a person at the keyboard.
    Real captureAt = Real(0.5);
    int capturePreviewSteps = 0;
    int captureCount = 1;
    Real captureStride = Real(0.2);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        }
        if (arg == "--new") {
            createNew = true;
        } else if (arg == "--config" && i + 1 < argc) {
            configPath = argv[++i];
        } else if (arg == "--capture" && i + 1 < argc) {
            capturePath = argv[++i];
        } else if (arg == "--at" && i + 1 < argc) {
            captureAt = static_cast<Real>(std::atof(argv[++i]));
        } else if (arg == "--capture-count" && i + 1 < argc) {
            captureCount = std::atoi(argv[++i]);
        } else if (arg == "--capture-stride" && i + 1 < argc) {
            captureStride = static_cast<Real>(std::atof(argv[++i]));
        } else if (arg == "--preview-steps" && i + 1 < argc) {
            capturePreviewSteps = std::atoi(argv[++i]);
        } else if (motionPath.empty()) {
            motionPath = arg;
        } else {
            std::fprintf(stderr, "unexpected argument: %s\n", arg.c_str());
            return 2;
        }
    }
    if (motionPath.empty()) {
        printUsage();
        return 2;
    }

    Humanoid2DConfig config = Humanoid2DConfig::defaults();
    if (!configPath.empty()) {
        std::string error;
        config = Humanoid2DConfig::loadFile(configPath, &error);
        if (!error.empty()) {
            std::fprintf(stderr, "config problem: %s\n", error.c_str());
            return 2;
        }
    }

    World2D referenceWorld;
    Humanoid2D figure;
    figure.build(referenceWorld, config);

    Motion2D motion;
    if (createNew) {
        motion.name = "motion";
        MotionKeyframe first;
        first.time = 0;
        first.jointAngles.assign(kJointCount, Real(0));
        first.rootPosition = Vec2(0, figure.groundedRootHeight(0, first.jointAngles.data()));
        motion.insertKeyframe(first);
        MotionKeyframe last = first;
        last.time = Real(1);
        motion.insertKeyframe(last);
        std::printf("new clip; will save to %s\n", motionPath.c_str());
    } else {
        std::string error;
        motion = Motion2D::loadFile(motionPath, &error);
        if (!error.empty()) {
            std::fprintf(stderr, "could not load %s: %s\n", motionPath.c_str(), error.c_str());
            return 2;
        }
        const int clamped = motion.clampToLimits(config);
        std::printf("loaded %s: %d keyframes, %.2fs, %s%s\n", motionPath.c_str(),
                    motion.frameCount(), double(motion.duration()),
                    motion.loop ? "looping" : "one-shot",
                    clamped ? "  (some angles were outside the joint limits and were clamped)"
                            : "");
    }

    const bool capturing = !capturePath.empty();
    if (capturing) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    Window window;
    if (!window.create(1280, 720, "aibf_animator")) return 1;
    if (capturing) glfwHideWindow(window.handle());
    Renderer2D renderer;
    if (!renderer.initialize()) {
        std::fprintf(stderr, "renderer initialisation failed\n");
        return 1;
    }

    Preview preview;
    preview.build(config);

    Camera2D camera;
    camera.center = Vec2(0, Real(0.95));
    camera.halfHeight = Real(1.5);

    Real time = 0;
    int selectedJoint = kHipL;
    bool playing = false;
    bool onionSkin = true;
    bool dirty = false;
    Trail rootTrail(300);
    int frame = 0;
    int capturesTaken = 0;
    double previousTime = glfwGetTime();

    if (capturing) {
        time = clamp(captureAt, Real(0), Real(1)) * motion.duration();
        if (capturePreviewSteps > 0) {
            preview.active = true;
            preview.restart(motion, time);
            for (int i = 0; i < capturePreviewSteps; ++i) {
                preview.step(motion);
                rootTrail.push(preview.figure.link(preview.world, kPelvis).position);
            }
            time = preview.time;
        }
        onionSkin = false;
    } else {
        printUsage();
    }

    while (window.isOpen()) {
        window.pollEvents();
        const Input& input = window.input();
        const Real coarse = input.shift ? Real(5) : Real(1);

        if (input.keyPressed[GLFW_KEY_ESCAPE]) window.requestClose();

        // ---- timeline ----
        if (input.keyPressed[GLFW_KEY_SPACE]) playing = !playing;
        if (input.keyPressed[GLFW_KEY_HOME]) { time = 0; playing = false; }
        if (input.keyPressed[GLFW_KEY_END]) { time = motion.duration(); playing = false; }
        if (input.keyPressed[GLFW_KEY_LEFT]) { time -= kScrubStep * coarse; playing = false; }
        if (input.keyPressed[GLFW_KEY_RIGHT]) { time += kScrubStep * coarse; playing = false; }
        if (input.keyPressed[GLFW_KEY_COMMA] || input.keyPressed[GLFW_KEY_PERIOD]) {
            const bool forward = input.keyPressed[GLFW_KEY_PERIOD];
            Real best = forward ? motion.duration() : Real(0);
            for (const MotionKeyframe& key : motion.keyframes) {
                if (forward && key.time > time + kKeyframeEpsilon) { best = key.time; break; }
                if (!forward && key.time < time - kKeyframeEpsilon) best = key.time;
            }
            time = best;
            playing = false;
        }

        if (playing) {
            const double now = glfwGetTime();
            time += static_cast<Real>(std::min(now - previousTime, 0.1));
            previousTime = now;
            if (time > motion.duration()) time = motion.loop ? Real(0) : motion.duration();
        } else {
            previousTime = glfwGetTime();
        }
        time = clamp(time, Real(0), std::max(motion.duration(), kScrubStep));

        int active = keyframeAt(motion, time);

        // ---- editing ----
        if (input.keyPressed[GLFW_KEY_UP]) {
            selectedJoint = (selectedJoint + kJointCount - 1) % kJointCount;
        }
        if (input.keyPressed[GLFW_KEY_DOWN]) {
            selectedJoint = (selectedJoint + 1) % kJointCount;
        }

        if (input.keyPressed[GLFW_KEY_K]) {
            // Sampled from the curve, so inserting a keyframe never changes the
            // pose that is already on screen.
            const MotionPose pose = motion.sample(time);
            MotionKeyframe key;
            key.time = time;
            key.rootPosition = pose.rootPosition;
            key.rootAngle = pose.rootAngle;
            key.jointAngles = pose.jointAngles;
            active = motion.insertKeyframe(key);
            dirty = true;
            std::printf("keyframe at %.3fs (%d total)\n", double(time), motion.frameCount());
        }
        if (input.keyPressed[GLFW_KEY_DELETE] && active >= 0 && motion.frameCount() > 2) {
            motion.removeKeyframe(active);
            active = -1;
            dirty = true;
            std::printf("removed keyframe; %d remain\n", motion.frameCount());
        }

        if (active >= 0) {
            MotionKeyframe& key = motion.keyframes[static_cast<size_t>(active)];
            const JointConfig& jc = config.joints[static_cast<size_t>(selectedJoint)];
            const Real nudge = Real(0.02) * coarse;
            bool edited = false;

            if (input.keyDown[GLFW_KEY_A] || input.keyDown[GLFW_KEY_D]) {
                const Real delta = (input.keyDown[GLFW_KEY_D] ? nudge : -nudge);
                Real& angle = key.jointAngles[static_cast<size_t>(selectedJoint)];
                angle = clamp(angle + delta, jc.lowerLimit, jc.upperLimit);
                edited = true;
            }
            if (input.keyDown[GLFW_KEY_W]) { key.rootPosition.y += nudge; edited = true; }
            if (input.keyDown[GLFW_KEY_S]) { key.rootPosition.y -= nudge; edited = true; }
            if (input.keyDown[GLFW_KEY_Q]) { key.rootAngle += nudge; edited = true; }
            if (input.keyDown[GLFW_KEY_E]) { key.rootAngle -= nudge; edited = true; }

            if (input.keyPressed[GLFW_KEY_G]) {
                key.rootPosition.y =
                    figure.groundedRootHeight(key.rootAngle, key.jointAngles.data());
                edited = true;
                std::printf("root dropped to %.4f m\n", double(key.rootPosition.y));
            }
            if (input.keyPressed[GLFW_KEY_0]) {
                if (input.shift) {
                    std::fill(key.jointAngles.begin(), key.jointAngles.end(), Real(0));
                } else {
                    key.jointAngles[static_cast<size_t>(selectedJoint)] = 0;
                }
                edited = true;
            }
            if (input.keyPressed[GLFW_KEY_LEFT_BRACKET] ||
                input.keyPressed[GLFW_KEY_RIGHT_BRACKET]) {
                const Real shift =
                    (input.keyPressed[GLFW_KEY_RIGHT_BRACKET] ? kScrubStep : -kScrubStep);
                const Real moved = key.time + shift;
                // Keyframes must stay strictly ordered, so a shift that would
                // cross a neighbour is refused rather than silently reordering.
                const bool clearBefore =
                    active == 0 || moved > motion.keyframes[static_cast<size_t>(active - 1)].time +
                                               kKeyframeEpsilon;
                const bool clearAfter =
                    active == motion.frameCount() - 1 ||
                    moved < motion.keyframes[static_cast<size_t>(active + 1)].time -
                                kKeyframeEpsilon;
                if (moved >= 0 && clearBefore && clearAfter) {
                    key.time = moved;
                    time = moved;
                    edited = true;
                }
            }
            if (edited) dirty = true;
        } else if (input.keyPressed[GLFW_KEY_A] || input.keyPressed[GLFW_KEY_D] ||
                   input.keyPressed[GLFW_KEY_G] || input.keyPressed[GLFW_KEY_W] ||
                   input.keyPressed[GLFW_KEY_S]) {
            std::printf("not on a keyframe - press k to make one at %.3fs\n", double(time));
        }

        // ---- clip settings ----
        if (input.keyPressed[GLFW_KEY_L]) {
            motion.loop = !motion.loop;
            dirty = true;
            std::printf("looping %s\n", motion.loop ? "on" : "off");
        }
        if (input.keyPressed[GLFW_KEY_I]) {
            motion.interpolation = motion.interpolation == MotionInterpolation::Hermite
                                       ? MotionInterpolation::Linear
                                       : MotionInterpolation::Hermite;
            dirty = true;
            std::printf("interpolation: %s\n",
                        motion.interpolation == MotionInterpolation::Linear ? "linear"
                                                                            : "hermite");
        }
        if (input.ctrl && input.keyPressed[GLFW_KEY_S]) {
            const std::string problem = motion.validate();
            if (!problem.empty()) {
                std::fprintf(stderr, "not saved - %s\n", problem.c_str());
            } else if (motion.writeFile(motionPath)) {
                dirty = false;
                std::printf("saved %s (%d keyframes, %.2fs)\n", motionPath.c_str(),
                            motion.frameCount(), double(motion.duration()));
            } else {
                std::fprintf(stderr, "could not write %s\n", motionPath.c_str());
            }
        }

        // ---- preview ----
        if (input.keyPressed[GLFW_KEY_P]) {
            preview.active = !preview.active;
            if (preview.active) preview.restart(motion, time);
            rootTrail.clear();
        }
        if (input.keyPressed[GLFW_KEY_R] && preview.active) {
            preview.restart(motion, time);
            rootTrail.clear();
        }
        if (input.keyPressed[GLFW_KEY_O]) onionSkin = !onionSkin;
        if (input.keyPressed[GLFW_KEY_T]) rootTrail.clear();

        if (preview.active && playing) {
            preview.step(motion);
            rootTrail.push(preview.figure.link(preview.world, kPelvis).position);
        }

        if (input.scroll != Real(0)) {
            camera.halfHeight =
                clamp(camera.halfHeight * std::pow(Real(0.9), input.scroll), Real(0.4), Real(8));
        }
        camera.aspect = window.aspect();

        // ---- draw ----
        const MotionPose pose = motion.sample(time);
        renderer.beginFrame(camera, palette::background);
        renderer.groundAndGrid(camera, 0, Real(0.5));

        if (onionSkin) {
            // The neighbouring keyframes, faintly, so a pose can be judged
            // against what it is coming from and going to.
            for (const MotionKeyframe& key : motion.keyframes) {
                if (std::abs(key.time - time) < kKeyframeEpsilon) continue;
                if (std::abs(key.time - time) > motion.duration() * Real(0.35)) continue;
                MotionPose ghost;
                ghost.rootPosition = key.rootPosition;
                ghost.rootAngle = key.rootAngle;
                ghost.jointAngles = key.jointAngles;
                drawReference(renderer, figure, ghost, palette::grid, Real(0.5), false);
            }
        }

        if (preview.active) {
            drawTrail(renderer, rootTrail, palette::com);
            DebugDrawOptions options;
            options.trail = false;
            drawHumanoid(renderer, preview.world, preview.figure, camera, options);
            drawWorld(renderer, preview.world, camera, options);
        }
        drawReference(renderer, figure, pose, active >= 0 ? palette::com : palette::joint,
                      preview.active ? Real(0.55) : Real(1), !preview.active);

        // The selected joint's pivot, so it is obvious which one a/d moves.
        {
            std::vector<Vec2> positions(config.links.size());
            std::vector<Real> angles(config.links.size());
            figure.forwardKinematics(pose.rootPosition, pose.rootAngle, pose.jointAngles.data(),
                                     positions.data(), angles.data());
            const JointConfig& jc = config.joints[static_cast<size_t>(selectedJoint)];
            const Vec2 pivot = positions[static_cast<size_t>(jc.child)];
            const Real marker = camera.pixelsToWorld(Real(7), 1080);
            renderer.circleOutline(pivot, marker, palette::contact);
        }

        drawTimeline(renderer, camera, motion, time, active);
        renderer.endFrame();

        if (capturing) {
            std::string path = capturePath;
            if (captureCount > 1) {
                const size_t dot = path.rfind('.');
                char suffix[16];
                std::snprintf(suffix, sizeof(suffix), "_%02d", capturesTaken);
                path = (dot == std::string::npos) ? path + suffix
                                                  : path.substr(0, dot) + suffix + path.substr(dot);
            }
            if (!window.saveScreenshot(path)) {
                std::fprintf(stderr, "could not write %s\n", path.c_str());
                return 1;
            }
            std::printf("captured %s  t=%.3fs phase=%.2f root=%.3f m %+.2f rad",
                        path.c_str(), double(time), double(motion.phaseAt(time)),
                        double(pose.rootPosition.y), double(pose.rootAngle));
            if (preview.active) {
                std::printf("  preview: pelvis %.3f m, tracking err %.3f rad",
                            double(preview.figure.link(preview.world, kPelvis).position.y),
                            double(preview.trackingError(motion)));
            }
            std::printf("\n");
            std::fflush(stdout);

            if (++capturesTaken >= captureCount) break;
            // Advance for the next frame of the strip.
            const Real target =
                clamp(captureAt + captureStride * Real(capturesTaken), Real(0), Real(1)) *
                motion.duration();
            if (preview.active) {
                while (preview.time < target) {
                    preview.step(motion);
                    rootTrail.push(preview.figure.link(preview.world, kPelvis).position);
                }
                time = preview.time;
            } else {
                time = target;
            }
            continue;
        }

        window.present();

        if (++frame % 10 == 0) {
            const Real angle = pose.jointAngles.empty()
                                   ? Real(0)
                                   : pose.jointAngles[static_cast<size_t>(selectedJoint)];
            char title[400];
            std::snprintf(title, sizeof(title),
                          "%s%s | t %.3f / %.2fs  phase %.2f | %s | key %s | [%s] %+.3f rad | "
                          "root %.3f m %+.2f rad%s",
                          dirty ? "*" : "", motionPath.c_str(), double(time),
                          double(motion.duration()), double(motion.phaseAt(time)),
                          motion.loop ? "loop" : "once",
                          active >= 0 ? std::to_string(active).c_str() : "-",
                          jointName(config, selectedJoint), double(angle),
                          double(pose.rootPosition.y), double(pose.rootAngle),
                          preview.active ? "" : "");
            std::string caption = title;
            if (preview.active) {
                char extra[96];
                std::snprintf(extra, sizeof(extra), " | preview err %.3f rad",
                              double(preview.trackingError(motion)));
                caption += extra;
            }
            window.setTitle(caption);
        }
    }

    if (dirty) {
        std::printf("\nunsaved changes were discarded (ctrl+s saves)\n");
    }
    return 0;
}
