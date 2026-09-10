// Generates starter reference motions.
//
// These are seeds for the animator, not finished work. Writing the JSON by hand
// is possible but the root height of a ground-contact pose is a *consequence*
// of the joint angles, not a free parameter, and getting it wrong by a
// centimetre produces a reference that floats or intersects the floor - one no
// policy can ever track. So the grounded keyframes have their height solved by
// forward kinematics here, and the airborne ones are given explicit heights
// chosen to sit on a plausible ballistic arc.
//
//   aibf_motions motions/
//
// It also reads clips back out, as a CSV of what the reference asks for at each
// phase:
//
//   aibf_motions --dump motions/backflip.json [--samples 512]
//
// That mode exists so the analysis scripts can grade a policy against the
// reference without a second implementation of Hermite sampling living in
// Python. A duplicated interpolator would be a place for the two sides to
// silently disagree, and the disagreement would look exactly like a tracking
// failure. Shelling out to the same sampler the reward uses removes that
// possibility rather than testing for it.
//
// Sign convention, derived in M2 and confirmed by rendering the squat pose:
// positive rotation is counter-clockwise, and for a figure facing +X that tips
// the head backwards. A backflip is therefore a *positive* root rotation
// through +2*pi; a forward roll is negative.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "humanoid/Humanoid2D.h"
#include "motion/Motion2D.h"
#include "physics/World2D.h"

using namespace aibf;

namespace {

// Joint order matches JointId: waist, neck, shoulder_l, elbow_l, shoulder_r,
// elbow_r, hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r.
struct KeySpec {
    Real time = 0;
    Real rootAngle = 0;
    // When grounded, the root height is solved so the lowest point touches y=0.
    // Otherwise `height` is used directly.
    bool grounded = true;
    Real height = 1;
    Real joints[kJointCount] = {};
};

// Takes doubles so the tables below can be written as plain decimals rather
// than a wall of `f` suffixes; the conversion happens once, here.
KeySpec pose(double time, double rootAngle, bool grounded, double height,
             std::initializer_list<double> joints) {
    KeySpec spec;
    spec.time = static_cast<Real>(time);
    spec.rootAngle = static_cast<Real>(rootAngle);
    spec.grounded = grounded;
    spec.height = static_cast<Real>(height);
    int i = 0;
    for (const double value : joints) {
        if (i < kJointCount) spec.joints[i++] = static_cast<Real>(value);
    }
    return spec;
}

Motion2D build(const Humanoid2D& figure, const std::string& name, bool loop,
               const std::vector<KeySpec>& specs) {
    Motion2D motion;
    motion.name = name;
    motion.loop = loop;
    motion.interpolation = MotionInterpolation::Hermite;

    for (const KeySpec& spec : specs) {
        MotionKeyframe frame;
        frame.time = spec.time;
        frame.rootAngle = spec.rootAngle;
        frame.jointAngles.assign(spec.joints, spec.joints + kJointCount);
        const Real height = spec.grounded
                                ? figure.groundedRootHeight(spec.rootAngle, spec.joints)
                                : spec.height;
        frame.rootPosition = Vec2(0, height);
        motion.insertKeyframe(frame);
    }
    return motion;
}

// ------------------------------------------------------------------ clips
//
// Shorthand for readability below. Every list is
//   waist neck  shL elL  shR elR   hipL kneeL ankL   hipR kneeR ankR

Motion2D armRaise(const Humanoid2D& figure) {
    return build(figure, "arm_raise", true,
                 {
                     pose(0.0, 0, true, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                     pose(0.7, 0, true, 0,
                          {0, 0.1, 2.6, 0.1, 2.6, 0.1, 0, 0, 0, 0, 0, 0}),
                     pose(1.4, 0, true, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                 });
}

Motion2D squat(const Humanoid2D& figure) {
    return build(figure, "squat", true,
                 {
                     pose(0.0, 0, true, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                     pose(0.7, 0, true, 0,
                          {-0.30, 0, 1.20, 0.20, 1.20, 0.20,
                           1.00, -1.70, 0.45, 1.00, -1.70, 0.45}),
                     pose(1.4, 0, true, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                 });
}

Motion2D jump(const Humanoid2D& figure) {
    // Airborne heights follow a ballistic arc: leaving the ground at t=0.40 and
    // landing at t=0.92 gives 0.52 s of flight, which for g = 9.81 corresponds
    // to a rise of about g*t^2/8 = 0.33 m above the take-off height.
    return build(figure, "jump", false,
                 {
                     pose(0.00, 0, true, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                     pose(0.25, 0, true, 0,
                          {-0.25, 0, -0.60, 0.10, -0.60, 0.10,
                           0.85, -1.45, 0.40, 0.85, -1.45, 0.40}),
                     pose(0.40, 0, true, 0,
                          {0.10, 0, 1.80, 0.05, 1.80, 0.05,
                           -0.20, -0.05, -0.55, -0.20, -0.05, -0.55}),
                     pose(0.58, 0, false, 1.33,
                          {0, 0, 2.20, 0.10, 2.20, 0.10,
                           0.55, -0.85, -0.10, 0.55, -0.85, -0.10}),
                     pose(0.78, 0, false, 1.18,
                          {0, 0, 1.40, 0.05, 1.40, 0.05,
                           0.10, -0.25, -0.30, 0.10, -0.25, -0.30}),
                     pose(0.92, 0, true, 0,
                          {-0.20, 0, 0.60, 0.10, 0.60, 0.10,
                           0.70, -1.20, 0.35, 0.70, -1.20, 0.35}),
                     pose(1.25, 0, true, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                 });
}

Motion2D backflip(const Humanoid2D& figure) {
    // Take-off at t=0.38, touchdown at t=1.12: 0.74 s of flight, which needs a
    // rise of roughly 0.34 m and a rotation rate of 2*pi/0.74 = 8.5 rad/s. Both
    // are demanding but not absurd, which is the point - a reference that is
    // physically impossible cannot be tracked no matter how long it trains.
    const Real kFullTurn = kTwoPi;
    return build(figure, "backflip", false,
                 {
                     // stand
                     pose(0.00, 0, true, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                     // crouch and load, arms swung back
                     pose(0.22, 0.10, true, 0,
                          {-0.30, 0, -0.90, 0.10, -0.90, 0.10,
                           0.95, -1.55, 0.45, 0.95, -1.55, 0.45}),
                     // extend and swing the arms up: the arm swing is what
                     // actually generates the backward angular momentum
                     pose(0.38, 0.35, true, 0,
                          {0.35, 0, 2.40, 0.05, 2.40, 0.05,
                           -0.35, -0.05, -0.70, -0.35, -0.05, -0.70}),
                     // airborne, tucking
                     pose(0.55, 1.60, false, 1.30,
                          {-0.55, -0.30, 2.10, 1.60, 2.10, 1.60,
                           1.75, -2.25, -0.20, 1.75, -2.25, -0.20}),
                     // peak, fully tucked and inverted
                     pose(0.75, 3.40, false, 1.46,
                          {-0.70, -0.40, 1.90, 2.30, 1.90, 2.30,
                           2.05, -2.50, -0.30, 2.05, -2.50, -0.30}),
                     // opening out of the tuck
                     pose(0.95, 5.10, false, 1.28,
                          {-0.35, -0.20, 1.80, 1.20, 1.80, 1.20,
                           1.10, -1.50, -0.20, 1.10, -1.50, -0.20}),
                     // extended, reaching for the ground
                     pose(1.08, 5.95, false, 1.08,
                          {0.05, 0, 1.50, 0.30, 1.50, 0.30,
                           0.35, -0.55, -0.30, 0.35, -0.55, -0.30}),
                     // touchdown, absorbing
                     pose(1.20, kFullTurn, true, 0,
                          {-0.25, 0, 0.80, 0.20, 0.80, 0.20,
                           0.80, -1.30, 0.40, 0.80, -1.30, 0.40}),
                     // stand back up
                     pose(1.50, kFullTurn, true, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                 });
}

Motion2D forwardRoll(const Humanoid2D& figure) {
    // A forward roll is the mirror case: negative root rotation, and unlike the
    // backflip it stays in contact with the ground throughout, so every
    // keyframe is grounded and the height follows from the pose.
    return build(figure, "forward_roll", false,
                 {
                     pose(0.00, 0, true, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                     pose(0.30, -0.45, true, 0,
                          {-0.70, -0.50, 1.90, 0.60, 1.90, 0.60,
                           1.10, -1.60, 0.45, 1.10, -1.60, 0.45}),
                     pose(0.60, -1.80, true, 0,
                          {-0.80, -0.60, 2.30, 2.00, 2.30, 2.00,
                           1.90, -2.40, 0.30, 1.90, -2.40, 0.30}),
                     pose(0.95, -3.50, true, 0,
                          {-0.80, -0.60, 2.20, 2.30, 2.20, 2.30,
                           2.05, -2.55, 0.20, 2.05, -2.55, 0.20}),
                     pose(1.30, -5.10, true, 0,
                          {-0.50, -0.30, 1.60, 1.40, 1.60, 1.40,
                           1.50, -1.90, 0.30, 1.50, -1.90, 0.30}),
                     pose(1.65, -kTwoPi, true, 0,
                          {-0.20, 0, 0.80, 0.30, 0.80, 0.30,
                           0.80, -1.20, 0.40, 0.80, -1.20, 0.40}),
                     pose(2.00, -kTwoPi, true, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                 });
}

bool write(Motion2D motion, const Humanoid2DConfig& config, const std::string& directory) {
    const int clamped = motion.clampToLimits(config);
    const std::string problem = motion.validate();
    if (!problem.empty()) {
        std::fprintf(stderr, "%s is invalid: %s\n", motion.name.c_str(), problem.c_str());
        return false;
    }
    const std::string path = directory + "/" + motion.name + ".json";
    if (!motion.writeFile(path)) {
        std::fprintf(stderr, "could not write %s\n", path.c_str());
        return false;
    }
    // Reported because clamping the keyframes does not answer it. Hermite
    // overshoots between them, so a clip whose every authored pose is legal can
    // still sweep past a limit mid-segment, and the solver will refuse to hold
    // that pose. It caps the tracking reward at a value no policy can beat.
    const Real excess = motion.worstSampledLimitExcess(config);
    std::printf("%-14s %d frames  %.2fs  %s%s%s", motion.name.c_str(), motion.frameCount(),
                double(motion.duration()), motion.loop ? "looping" : "one-shot",
                clamped ? "  (clamped keys: yes)" : "",
                excess > Real(1e-4) ? "" : "\n");
    if (excess > Real(1e-4)) {
        std::printf("  <- sampled clip leaves its joint limits by %.4f rad between keyframes\n",
                    double(excess));
    }
    return true;
}

// ------------------------------------------------------------------- dump
//
// The reference at `samples` evenly spaced phases, as CSV on stdout. The
// comment header carries the joint names and each joint's limits, so a reader
// never has to hardcode the joint order and can see at a glance whether the
// clip asks for a pose the figure is not allowed to hold.
//
// Phases are inclusive of both ends: sample i is i/(samples-1), so a
// one-shot clip's final pose is actually in the table rather than one step
// short of it.
int dump(const std::string& path, int samples, const Humanoid2DConfig& config) {
    std::string error;
    const Motion2D motion = Motion2D::loadFile(path, &error);
    if (!error.empty()) {
        std::fprintf(stderr, "%s: %s\n", path.c_str(), error.c_str());
        return 1;
    }
    if (samples < 2) {
        std::fprintf(stderr, "--samples must be at least 2\n");
        return 1;
    }

    const int joints = motion.jointCount;
    std::printf("# name=%s duration=%.9g loop=%d joints=%d frames=%d samples=%d\n",
                motion.name.c_str(), double(motion.duration()), motion.loop ? 1 : 0, joints,
                motion.frameCount(), samples);

    std::printf("# joint_names=");
    for (int j = 0; j < joints; ++j) {
        const char* name = j < static_cast<int>(config.joints.size())
                               ? config.joints[static_cast<size_t>(j)].name.c_str()
                               : "?";
        std::printf("%s%s", j ? "," : "", name);
    }
    std::printf("\n# joint_limits=");
    for (int j = 0; j < joints; ++j) {
        if (j >= static_cast<int>(config.joints.size())) break;
        const JointConfig& jc = config.joints[static_cast<size_t>(j)];
        std::printf("%s%.9g:%.9g", j ? "," : "", double(jc.lowerLimit), double(jc.upperLimit));
    }
    std::printf("\n");

    std::printf("phase,time,root_x,root_y,root_angle");
    for (int j = 0; j < joints; ++j) std::printf(",q%d", j);
    for (int j = 0; j < joints; ++j) std::printf(",dq%d", j);
    std::printf("\n");

    for (int i = 0; i < samples; ++i) {
        const Real phase = Real(i) / Real(samples - 1);
        const MotionPose pose = motion.samplePhase(phase);
        const MotionPose rate = motion.samplePhaseVelocity(phase);
        std::printf("%.9g,%.9g,%.9g,%.9g,%.9g", double(phase), double(motion.timeAt(phase)),
                    double(pose.rootPosition.x), double(pose.rootPosition.y),
                    double(pose.rootAngle));
        for (int j = 0; j < joints; ++j) {
            std::printf(",%.9g", double(pose.jointAngles[static_cast<size_t>(j)]));
        }
        for (int j = 0; j < joints; ++j) {
            std::printf(",%.9g", double(rate.jointAngles[static_cast<size_t>(j)]));
        }
        std::printf("\n");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dumpPath;
    int samples = 512;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dumpPath = argv[++i];
        } else if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            samples = std::atoi(argv[++i]);
        }
    }
    if (!dumpPath.empty()) {
        World2D probe;
        Humanoid2D unused;
        const Humanoid2DConfig config = Humanoid2DConfig::defaults();
        unused.build(probe, config);
        return dump(dumpPath, samples, config);
    }

    const std::string directory = argc > 1 ? argv[1] : "motions";

    // A world is needed only because Humanoid2D caches its kinematics at build
    // time; nothing here is simulated.
    World2D world;
    Humanoid2D figure;
    const Humanoid2DConfig config = Humanoid2DConfig::defaults();
    figure.build(world, config);

    std::printf("writing reference motions to %s/\n", directory.c_str());
    bool ok = true;
    ok &= write(armRaise(figure), config, directory);
    ok &= write(squat(figure), config, directory);
    ok &= write(jump(figure), config, directory);
    ok &= write(forwardRoll(figure), config, directory);
    ok &= write(backflip(figure), config, directory);

    std::printf(
        "\nThese are seeds for the animator, not finished motions. Load one with\n"
        "aibf_animator to refine it, and check the physics preview: a reference the\n"
        "figure cannot physically follow will never be tracked no matter how long\n"
        "it trains.\n");
    return ok ? 0 : 1;
}
