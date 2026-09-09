#include "motion/Motion3D.h"

#include <algorithm>
#include <cmath>

namespace aibf {

namespace {

Json quatToJson(const Quat& q) {
    Json out = Json::array();
    out.push(Json(double(q.x)));
    out.push(Json(double(q.y)));
    out.push(Json(double(q.z)));
    out.push(Json(double(q.w)));
    return out;
}

Quat quatFromJson(const Json& json) {
    if (!json.isArray() || json.size() < 4) return Quat::identity();
    return normalize(Quat(Real(json[0].real(0)), Real(json[1].real(0)), Real(json[2].real(0)),
                          Real(json[3].real(1))));
}

}  // namespace

void Motion3D::insertKeyframe(const MotionKeyframe3D& key) {
    for (MotionKeyframe3D& existing : keyframes) {
        if (std::abs(existing.time - key.time) < Real(1e-6)) {
            existing = key;
            return;
        }
    }
    keyframes.push_back(key);
    std::sort(keyframes.begin(), keyframes.end(),
              [](const MotionKeyframe3D& a, const MotionKeyframe3D& b) { return a.time < b.time; });
}

MotionPose3D Motion3D::sample(Real time) const {
    MotionPose3D pose;
    pose.resize(static_cast<size_t>(jointCount));
    if (keyframes.empty()) return pose;
    if (keyframes.size() == 1) {
        pose.rootPosition = keyframes[0].rootPosition;
        pose.rootOrientation = keyframes[0].rootOrientation;
        pose.rootTurns = keyframes[0].rootTurns;
        pose.jointRotations = keyframes[0].jointRotations;
        return pose;
    }

    const Real total = duration();
    Real t = time;
    if (loop && total > Real(0)) {
        t = std::fmod(t, total);
        if (t < Real(0)) t += total;
    } else {
        t = clamp(t, keyframes.front().time, keyframes.back().time);
    }

    size_t next = 1;
    while (next < keyframes.size() && keyframes[next].time < t) ++next;
    const MotionKeyframe3D& a = keyframes[next - 1];
    const MotionKeyframe3D& b = keyframes[next];
    const Real span = b.time - a.time;
    const Real u = span > Real(1e-9) ? (t - a.time) / span : Real(0);

    pose.rootPosition = a.rootPosition + (b.rootPosition - a.rootPosition) * u;
    // Slerp rather than a component lerp: lerping moves at the wrong angular
    // rate through the middle of an arc, which is exactly where a tracking term
    // is looking hardest.
    pose.rootOrientation = slerp(a.rootOrientation, b.rootOrientation, u);
    pose.rootTurns = a.rootTurns + (b.rootTurns - a.rootTurns) * u;

    const size_t joints = std::min(a.jointRotations.size(), b.jointRotations.size());
    for (size_t i = 0; i < joints && i < pose.jointRotations.size(); ++i) {
        pose.jointRotations[i] = slerp(a.jointRotations[i], b.jointRotations[i], u);
    }
    return pose;
}

void Motion3D::sampleVelocity(Real time, Vec3& rootVelocity, Vec3& rootAngularVelocity,
                              std::vector<Vec3>& jointRates) const {
    jointRates.assign(static_cast<size_t>(jointCount), Vec3(0, 0, 0));
    rootVelocity = Vec3(0, 0, 0);
    rootAngularVelocity = Vec3(0, 0, 0);
    if (keyframes.size() < 2) return;

    // Central difference. The step is small relative to the clip but large
    // relative to float precision, which matters because the difference of two
    // nearly equal quaternions is where cancellation lives.
    const Real h = std::max(Real(1e-3), duration() * Real(1e-3));
    const MotionPose3D before = sample(time - h);
    const MotionPose3D after = sample(time + h);
    const Real inv = Real(1) / (Real(2) * h);

    rootVelocity = (after.rootPosition - before.rootPosition) * inv;
    rootAngularVelocity =
        rotationVector(after.rootOrientation * conjugate(before.rootOrientation)) * inv;

    const size_t joints = std::min(before.jointRotations.size(), after.jointRotations.size());
    for (size_t i = 0; i < joints && i < jointRates.size(); ++i) {
        jointRates[i] =
            rotationVector(after.jointRotations[i] * conjugate(before.jointRotations[i])) * inv;
    }
}

int Motion3D::clampToLimits(const Humanoid3DConfig& config) {
    int changed = 0;
    for (MotionKeyframe3D& key : keyframes) {
        for (size_t i = 0; i < key.jointRotations.size() && i < config.joints.size(); ++i) {
            const JointConfig3D& jc = config.joints[i];
            const Quat original = key.jointRotations[i];

            if (jc.kind == JointKind::Hinge) {
                // Reduced to a single angle about the hinge axis, clamped, and
                // rebuilt. Anything the clip asked for off that axis is not a
                // pose this joint has, so keeping it would be authoring a
                // motion the figure cannot perform.
                const Vec3 axis = normalize(jc.hingeAxis);
                const Real angle = clamp(twistAngle(original, axis), jc.lowerLimit, jc.upperLimit);
                key.jointRotations[i] = Quat::fromAxisAngle(axis, angle);
            } else {
                Quat swing, twist;
                swingTwistDecomposition(original, Vec3(0, 1, 0), swing, twist);
                const Real swingAngle = angleOf(swing);
                if (swingAngle > jc.coneAngle) {
                    const Vec3 swingVector = rotationVector(swing);
                    const Real magnitude = length(swingVector);
                    if (magnitude > kEpsilon) {
                        swing = quatFromRotationVector(swingVector *
                                                       (jc.coneAngle / magnitude));
                    }
                }
                const Real twistValue =
                    clamp(twistAngle(twist, Vec3(0, 1, 0)), jc.lowerTwist, jc.upperTwist);
                key.jointRotations[i] =
                    swing * Quat::fromAxisAngle(Vec3(0, 1, 0), twistValue);
            }

            if (angleOf(key.jointRotations[i] * conjugate(original)) > Real(1e-4)) ++changed;
        }
    }
    return changed;
}

std::string Motion3D::validate() const {
    if (keyframes.empty()) return "the clip has no keyframes";
    for (size_t i = 0; i < keyframes.size(); ++i) {
        if (keyframes[i].jointRotations.size() != static_cast<size_t>(jointCount)) {
            return "keyframe " + std::to_string(i) + " has the wrong joint count";
        }
        if (i > 0 && keyframes[i].time <= keyframes[i - 1].time) {
            return "keyframe " + std::to_string(i) + " is not after the one before it";
        }
        for (const Quat& q : keyframes[i].jointRotations) {
            if (!isFinite(q)) return "keyframe " + std::to_string(i) + " has a non-finite rotation";
        }
    }
    if (keyframes.front().time != Real(0)) return "the first keyframe must be at time 0";
    return std::string();
}

Json Motion3D::toJson() const {
    Json root = Json::object();
    root.set("name", Json(name));
    root.set("loop", Json(loop));
    root.set("joint_count", Json(jointCount));

    Json frames = Json::array();
    for (const MotionKeyframe3D& key : keyframes) {
        Json entry = Json::object();
        entry.set("time", Json(double(key.time)));

        Json position = Json::array();
        position.push(Json(double(key.rootPosition.x)));
        position.push(Json(double(key.rootPosition.y)));
        position.push(Json(double(key.rootPosition.z)));
        entry.set("root_position", std::move(position));
        entry.set("root_orientation", quatToJson(key.rootOrientation));
        entry.set("root_turns", Json(double(key.rootTurns)));

        Json rotations = Json::array();
        for (const Quat& q : key.jointRotations) rotations.push(quatToJson(q));
        entry.set("joints", std::move(rotations));

        frames.push(std::move(entry));
    }
    root.set("keyframes", std::move(frames));
    return root;
}

Motion3D Motion3D::fromJson(const Json& json) {
    Motion3D motion;
    if (!json.isObject()) return motion;
    motion.name = json["name"].string(motion.name);
    motion.loop = json["loop"].boolean(motion.loop);
    motion.jointCount = json["joint_count"].integer(motion.jointCount);

    const Json& frames = json["keyframes"];
    if (!frames.isArray()) return motion;
    for (int i = 0; i < frames.size(); ++i) {
        const Json& entry = frames[i];
        MotionKeyframe3D key;
        key.time = Real(entry["time"].real(0));

        const Json& position = entry["root_position"];
        if (position.isArray() && position.size() >= 3) {
            key.rootPosition = Vec3(Real(position[0].real(0)), Real(position[1].real(1)),
                                    Real(position[2].real(0)));
        }
        key.rootOrientation = quatFromJson(entry["root_orientation"]);
        key.rootTurns = Real(entry["root_turns"].real(0));

        const Json& rotations = entry["joints"];
        key.jointRotations.assign(static_cast<size_t>(motion.jointCount), Quat::identity());
        if (rotations.isArray()) {
            for (int j = 0; j < rotations.size() && j < motion.jointCount; ++j) {
                key.jointRotations[static_cast<size_t>(j)] = quatFromJson(rotations[j]);
            }
        }
        motion.keyframes.push_back(key);
    }
    std::sort(motion.keyframes.begin(), motion.keyframes.end(),
              [](const MotionKeyframe3D& a, const MotionKeyframe3D& b) { return a.time < b.time; });
    return motion;
}

bool Motion3D::writeFile(const std::string& path) const { return toJson().writeFile(path); }

Motion3D Motion3D::readFile(const std::string& path, std::string* error) {
    std::string parseError;
    const Json json = Json::parseFile(path, &parseError);
    if (!parseError.empty()) {
        if (error) *error = parseError;
        return Motion3D{};
    }
    Motion3D motion = fromJson(json);
    if (error) *error = motion.validate();
    return motion;
}

}  // namespace aibf
