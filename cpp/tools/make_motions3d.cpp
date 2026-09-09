// Writes the 3D reference motions.
//
// These are authored, not captured. A clip says what the motion should look
// like; the policy has to find torques that produce something close to it, and
// where the reference is physically impossible the physics wins.
//
// Two things are handled here rather than left to the author:
//
//   * Root height is solved by forward kinematics, so a keyframed crouch
//     actually reaches the floor instead of hovering above it or starting
//     interpenetrating.
//   * The root turn count is carried alongside the orientation, because a
//     quaternion double-covers and cannot express "one full turn" on its own.
//
//   aibf_motions3d [--out motions]
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "humanoid/Humanoid3D.h"
#include "motion/Motion3D.h"

using namespace aibf;

namespace {

// The axis each acrobatic motion turns about. A backflip is sagittal, the plane
// the 2D figure lived in. A cartwheel is frontal, and has no 2D counterpart at
// all: a sagittal figure has no frontal plane to turn in, which is why the
// cartwheel waited for this milestone.
const Vec3 kSagittal(0, 0, 1);
const Vec3 kFrontal(1, 0, 0);

struct Rig {
    World3D world;
    Humanoid3D figure;
    Humanoid3DConfig config;

    Rig() {
        config = Humanoid3DConfig::defaults();
        figure.build(world, config, config.links[kPelvis].restPosition);
    }
};

// A keyframe whose root height is solved so the figure's lowest point rests on
// the ground. `airborne` overrides that with an explicit height, since a figure
// in flight is not touching anything.
MotionKeyframe3D groundedKey(const Rig& rig, Real time, const std::vector<Quat>& joints,
                             const Quat& rootOrientation, Real turns, Real z = 0,
                             Real airborne = -1) {
    MotionKeyframe3D key;
    key.time = time;
    key.rootOrientation = rootOrientation;
    key.rootTurns = turns;
    key.jointRotations = joints;
    const Real height =
        airborne >= Real(0)
            ? airborne
            : rig.figure.groundedRootHeight(rig.world, rootOrientation, joints.data());
    key.rootPosition = Vec3(0, height, z);
    return key;
}

std::vector<Quat> restPose() { return std::vector<Quat>(kJointCount, Quat::identity()); }

void setHinge(std::vector<Quat>& pose, int joint, const Vec3& axis, Real angle) {
    pose[static_cast<size_t>(joint)] = Quat::fromAxisAngle(axis, angle);
}

void setBall(std::vector<Quat>& pose, int joint, const Vec3& axis, Real angle) {
    pose[static_cast<size_t>(joint)] = Quat::fromAxisAngle(normalize(axis), angle);
}

// ---------------------------------------------------------------- backflip

Motion3D makeBackflip(const Rig& rig) {
    Motion3D motion;
    motion.name = "backflip3d";
    motion.loop = false;

    auto stand = []() { return restPose(); };

    auto crouch = []() {
        std::vector<Quat> pose = restPose();
        setBall(pose, kHipL, kSagittal, Real(0.9));
        setBall(pose, kHipR, kSagittal, Real(0.9));
        setHinge(pose, kKneeL, kSagittal, Real(-1.7));
        setHinge(pose, kKneeR, kSagittal, Real(-1.7));
        setHinge(pose, kAnkleL, kSagittal, Real(0.5));
        setHinge(pose, kAnkleR, kSagittal, Real(0.5));
        // Arms swung back, ready to throw upward into the launch.
        setBall(pose, kShoulderL, kSagittal, Real(-0.9));
        setBall(pose, kShoulderR, kSagittal, Real(-0.9));
        return pose;
    };

    auto launch = []() {
        std::vector<Quat> pose = restPose();
        setBall(pose, kHipL, kSagittal, Real(-0.2));
        setBall(pose, kHipR, kSagittal, Real(-0.2));
        setHinge(pose, kKneeL, kSagittal, Real(-0.05));
        setHinge(pose, kKneeR, kSagittal, Real(-0.05));
        setHinge(pose, kAnkleL, kSagittal, Real(-0.6));
        setHinge(pose, kAnkleR, kSagittal, Real(-0.6));
        // Arms thrown overhead, which is where a real backflip gets much of its
        // angular momentum.
        setBall(pose, kShoulderL, kSagittal, Real(1.7));
        setBall(pose, kShoulderR, kSagittal, Real(1.7));
        return pose;
    };

    auto tuck = []() {
        std::vector<Quat> pose = restPose();
        setBall(pose, kHipL, kSagittal, Real(1.25));
        setBall(pose, kHipR, kSagittal, Real(1.25));
        setHinge(pose, kKneeL, kSagittal, Real(-2.4));
        setHinge(pose, kKneeR, kSagittal, Real(-2.4));
        setHinge(pose, kAnkleL, kSagittal, Real(0.4));
        setHinge(pose, kAnkleR, kSagittal, Real(0.4));
        setBall(pose, kShoulderL, kSagittal, Real(0.8));
        setBall(pose, kShoulderR, kSagittal, Real(0.8));
        setHinge(pose, kElbowL, kSagittal, Real(2.2));
        setHinge(pose, kElbowR, kSagittal, Real(2.2));
        // Chin tucked, which a gymnast does and which also keeps the head away
        // from the ground on the way over.
        setBall(pose, kWaist, kSagittal, Real(0.5));
        setBall(pose, kNeck, kSagittal, Real(0.4));
        return pose;
    };

    auto extend = []() {
        std::vector<Quat> pose = restPose();
        setBall(pose, kHipL, kSagittal, Real(0.3));
        setBall(pose, kHipR, kSagittal, Real(0.3));
        setHinge(pose, kKneeL, kSagittal, Real(-0.5));
        setHinge(pose, kKneeR, kSagittal, Real(-0.5));
        setBall(pose, kShoulderL, kSagittal, Real(0.3));
        setBall(pose, kShoulderR, kSagittal, Real(0.3));
        return pose;
    };

    auto turn = [](Real fraction) { return Quat::fromAxisAngle(kSagittal, fraction * Real(2) * kPi); };

    // The airborne arc is computed, not authored.
    //
    // The first version of this clip picked apex heights by eye, and the result
    // was a reference that asked for 0.62 m of rise from a launch that produced
    // 1.9 m/s, which reaches 0.18 m. That is not a hard motion, it is an
    // impossible one, and a tracking reward would have spent the whole run
    // punishing the policy for failing to break ballistics. The PD preview
    // caught it: it peaked at 1.20 m against the 1.62 m the clip demanded.
    //
    // So the apex is chosen first and everything else follows from it. With
    // rise = v0^2 / 2g, a 0.45 m apex needs 2.97 m/s and gives 0.605 s of
    // flight, and the keyframes in between sit on the parabola that velocity
    // actually produces.
    const Real gravity = Real(9.81);
    const Real apexRise = Real(0.45);
    const Real launchSpeed = std::sqrt(Real(2) * gravity * apexRise);
    const Real flightTime = Real(2) * launchSpeed / gravity;

    const Real launchTime = Real(0.46);
    const Real launchHeight = Real(1.02);
    const Real landTime = launchTime + flightTime;
    // Travelling backwards, which is what makes it a backflip rather than a
    // back tuck in place.
    const Real driftSpeed = Real(-0.55);

    auto flightHeight = [&](Real t) {
        const Real u = t - launchTime;
        return launchHeight + launchSpeed * u - Real(0.5) * gravity * u * u;
    };
    auto flightZ = [&](Real t) { return driftSpeed * (t - launchTime); };
    auto flightTurns = [&](Real t) {
        // A full turn spread over the flight, starting and ending level.
        return clamp((t - launchTime) / flightTime, Real(0), Real(1));
    };

    motion.insertKeyframe(groundedKey(rig, Real(0.00), stand(), turn(0), 0));
    motion.insertKeyframe(groundedKey(rig, Real(0.30), crouch(), turn(0), 0));
    motion.insertKeyframe(groundedKey(rig, launchTime, launch(), turn(0), 0, 0, launchHeight));

    // Four samples along the parabola: rising, apex, falling, and just before
    // touchdown.
    for (const Real fraction : {Real(0.25), Real(0.5), Real(0.75), Real(0.95)}) {
        const Real t = launchTime + fraction * flightTime;
        const Real turns = flightTurns(t);
        // Tucked through the middle of the flight and opening out for the
        // landing, which is what actually controls the rotation rate.
        const std::vector<Quat> pose = (fraction < Real(0.8)) ? tuck() : extend();
        motion.insertKeyframe(
            groundedKey(rig, t, pose, turn(turns), turns, flightZ(t), flightHeight(t)));
    }

    motion.insertKeyframe(
        groundedKey(rig, landTime, crouch(), turn(1), 1, flightZ(landTime)));
    motion.insertKeyframe(
        groundedKey(rig, landTime + Real(0.25), stand(), turn(1), 1, flightZ(landTime)));
    return motion;
}

// ---------------------------------------------------------------- cartwheel

Motion3D makeCartwheel(const Rig& rig) {
    Motion3D motion;
    motion.name = "cartwheel";
    motion.loop = false;

    auto stand = []() {
        std::vector<Quat> pose = restPose();
        // Arms up and out to the sides, the way a cartwheel starts.
        setBall(pose, kShoulderL, kFrontal, Real(-1.5));
        setBall(pose, kShoulderR, kFrontal, Real(1.5));
        return pose;
    };

    auto lunge = []() {
        std::vector<Quat> pose = restPose();
        // Weight over the leading leg, torso tipping sideways towards the hand
        // that will reach the floor first.
        setBall(pose, kHipL, kFrontal, Real(-0.5));
        setBall(pose, kHipR, kFrontal, Real(0.25));
        setHinge(pose, kKneeL, kSagittal, Real(-0.7));
        setBall(pose, kWaist, kFrontal, Real(-0.45));
        setBall(pose, kShoulderL, kFrontal, Real(-2.0));
        setBall(pose, kShoulderR, kFrontal, Real(1.2));
        return pose;
    };

    auto split = []() {
        std::vector<Quat> pose = restPose();
        // Legs opened in the frontal plane. This is the shape that has no 2D
        // representation: both legs move, in opposite directions, out of the
        // sagittal plane.
        setBall(pose, kHipL, kFrontal, Real(-1.15));
        setBall(pose, kHipR, kFrontal, Real(1.15));
        setBall(pose, kShoulderL, kFrontal, Real(-1.7));
        setBall(pose, kShoulderR, kFrontal, Real(1.7));
        return pose;
    };

    auto recover = []() {
        std::vector<Quat> pose = restPose();
        setBall(pose, kHipL, kFrontal, Real(-0.6));
        setBall(pose, kHipR, kFrontal, Real(0.35));
        setHinge(pose, kKneeR, kSagittal, Real(-0.6));
        setBall(pose, kShoulderL, kFrontal, Real(-1.6));
        setBall(pose, kShoulderR, kFrontal, Real(1.6));
        return pose;
    };

    // Turning about the frontal axis, so the figure rolls sideways over its
    // hands rather than end over end.
    auto turn = [](Real fraction) { return Quat::fromAxisAngle(kFrontal, fraction * Real(2) * kPi); };

    motion.insertKeyframe(groundedKey(rig, Real(0.00), stand(), turn(0), 0, Real(0.0)));
    motion.insertKeyframe(groundedKey(rig, Real(0.35), lunge(), turn(Real(0.06)), Real(0.06),
                                      Real(0.25)));
    motion.insertKeyframe(groundedKey(rig, Real(0.65), split(), turn(Real(0.25)), Real(0.25),
                                      Real(0.55), Real(1.15)));
    motion.insertKeyframe(groundedKey(rig, Real(0.95), split(), turn(Real(0.50)), Real(0.50),
                                      Real(0.85), Real(1.05)));
    motion.insertKeyframe(groundedKey(rig, Real(1.25), split(), turn(Real(0.75)), Real(0.75),
                                      Real(1.15), Real(1.15)));
    motion.insertKeyframe(groundedKey(rig, Real(1.55), recover(), turn(Real(0.94)), Real(0.94),
                                      Real(1.40)));
    motion.insertKeyframe(groundedKey(rig, Real(1.80), stand(), turn(1), 1, Real(1.55)));
    return motion;
}

// ---------------------------------------------------------------- squat

// A simple sanity clip: no rotation, no flight, just a crouch and back up. If
// imitation cannot learn this one, nothing acrobatic is worth attempting.
Motion3D makeSquat(const Rig& rig) {
    Motion3D motion;
    motion.name = "squat3d";
    motion.loop = true;

    auto down = []() {
        std::vector<Quat> pose = restPose();
        setBall(pose, kHipL, kSagittal, Real(0.9));
        setBall(pose, kHipR, kSagittal, Real(0.9));
        setHinge(pose, kKneeL, kSagittal, Real(-1.5));
        setHinge(pose, kKneeR, kSagittal, Real(-1.5));
        setHinge(pose, kAnkleL, kSagittal, Real(0.45));
        setHinge(pose, kAnkleR, kSagittal, Real(0.45));
        setBall(pose, kShoulderL, kSagittal, Real(1.3));
        setBall(pose, kShoulderR, kSagittal, Real(1.3));
        return pose;
    };

    motion.insertKeyframe(groundedKey(rig, Real(0.0), restPose(), Quat::identity(), 0));
    motion.insertKeyframe(groundedKey(rig, Real(0.6), down(), Quat::identity(), 0));
    motion.insertKeyframe(groundedKey(rig, Real(1.2), restPose(), Quat::identity(), 0));
    return motion;
}

void report(const Rig& rig, const Motion3D& motion) {
    std::printf("  %-12s %5.2f s, %2zu keyframes, %s\n", motion.name.c_str(),
                double(motion.duration()), motion.keyframes.size(),
                motion.loop ? "looping" : "once");

    // Report the extremes the clip asks for, so an obviously impossible motion
    // is visible here rather than after an hour of training.
    Real lowest = Real(1e9), highest = -Real(1e9), turns = 0;
    for (Real t = 0; t <= motion.duration(); t += Real(0.02)) {
        const MotionPose3D pose = motion.sample(t);
        std::vector<Vec3> positions(kLinkCount);
        std::vector<Quat> orientations(kLinkCount);
        rig.figure.forwardKinematics(rig.world, pose.rootPosition, pose.rootOrientation,
                                     pose.jointRotations.data(), positions.data(),
                                     orientations.data());
        for (int i = 0; i < kLinkCount; ++i) {
            const LinkConfig3D& lc = rig.config.links[static_cast<size_t>(i)];
            const Vec3 axis = rotate(orientations[i], Vec3(0, lc.halfLength, 0));
            lowest = std::min(lowest, positions[i].y - std::abs(axis.y) - lc.radius);
        }
        highest = std::max(highest, pose.rootPosition.y);
        turns = pose.rootTurns;
    }
    std::printf("               root peak %.2f m, lowest point %.3f m, %.2f turns\n",
                double(highest), double(lowest), double(turns));
    if (lowest < Real(-0.05)) {
        std::printf("               WARNING: the clip passes %.3f m through the floor\n",
                    double(lowest));
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string outDirectory = "motions";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) outDirectory = argv[++i];
    }

    Rig rig;
    std::printf("3D reference motions, figure %.1f kg, %.2f m\n", double(rig.config.totalMass()),
                double(rig.config.restHeight()));

    const Motion3D motions[] = {makeSquat(rig), makeBackflip(rig), makeCartwheel(rig)};
    for (const Motion3D& motion : motions) {
        Motion3D clip = motion;
        const int clamped = clip.clampToLimits(rig.config);

        const std::string problem = clip.validate();
        if (!problem.empty()) {
            std::fprintf(stderr, "%s is not a usable clip: %s\n", clip.name.c_str(),
                         problem.c_str());
            return 1;
        }

        const std::string path = outDirectory + "/" + clip.name + ".json";
        if (!clip.writeFile(path)) {
            std::fprintf(stderr, "could not write %s\n", path.c_str());
            return 1;
        }
        report(rig, clip);
        if (clamped > 0) {
            std::printf("               %d joint values were outside the limits and were "
                        "clamped\n", clamped);
        }
        std::printf("               wrote %s\n", path.c_str());
    }
    return 0;
}
