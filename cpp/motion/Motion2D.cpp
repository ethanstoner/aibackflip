#include "motion/Motion2D.h"

#include <algorithm>
#include <cmath>

namespace aibf {

namespace {

// Cubic Hermite basis, and its derivative with respect to the local parameter.
struct Hermite {
    Real h00, h10, h01, h11;
    Real d00, d10, d01, d11;

    static Hermite at(Real t) {
        const Real t2 = t * t;
        const Real t3 = t2 * t;
        Hermite h;
        h.h00 = Real(2) * t3 - Real(3) * t2 + Real(1);
        h.h10 = t3 - Real(2) * t2 + t;
        h.h01 = Real(-2) * t3 + Real(3) * t2;
        h.h11 = t3 - t2;
        h.d00 = Real(6) * t2 - Real(6) * t;
        h.d10 = Real(3) * t2 - Real(4) * t + Real(1);
        h.d01 = Real(-6) * t2 + Real(6) * t;
        h.d11 = Real(3) * t2 - Real(2) * t;
        return h;
    }
};

}  // namespace

Real Motion2D::duration() const {
    if (keyframes.empty()) return 0;
    return std::max(Real(0), keyframes.back().time - keyframes.front().time);
}

Real Motion2D::phaseAt(Real time) const {
    const Real total = duration();
    if (total <= Real(0)) return 0;
    const Real start = keyframes.front().time;
    Real phase = (time - start) / total;
    if (loop) {
        phase -= std::floor(phase);
    } else {
        phase = clamp(phase, Real(0), Real(1));
    }
    return phase;
}

int Motion2D::insertKeyframe(const MotionKeyframe& frame) {
    MotionKeyframe copy = frame;
    copy.jointAngles.resize(static_cast<size_t>(jointCount), Real(0));

    for (size_t i = 0; i < keyframes.size(); ++i) {
        if (std::abs(keyframes[i].time - copy.time) < Real(1e-6)) {
            keyframes[i] = copy;  // authoring over an existing key replaces it
            return static_cast<int>(i);
        }
        if (keyframes[i].time > copy.time) {
            keyframes.insert(keyframes.begin() + static_cast<std::ptrdiff_t>(i), copy);
            return static_cast<int>(i);
        }
    }
    keyframes.push_back(copy);
    return static_cast<int>(keyframes.size()) - 1;
}

void Motion2D::removeKeyframe(int index) {
    if (index < 0 || index >= frameCount()) return;
    keyframes.erase(keyframes.begin() + index);
}

void Motion2D::sortByTime() {
    std::stable_sort(keyframes.begin(), keyframes.end(),
                     [](const MotionKeyframe& a, const MotionKeyframe& b) {
                         return a.time < b.time;
                     });
}

void Motion2D::retime(Real seconds) {
    if (keyframes.size() < 2 || seconds <= Real(0)) return;
    const Real start = keyframes.front().time;
    const Real span = duration();
    if (span <= Real(0)) return;
    const Real scale = seconds / span;
    for (MotionKeyframe& frame : keyframes) {
        frame.time = (frame.time - start) * scale;
    }
}

// ---------------------------------------------------------------- sampling

namespace {

// Finite-difference tangent at keyframe `i`, expressed per unit time so that
// uneven keyframe spacing does not change the shape of the curve.
template <typename Get>
Real tangent(const std::vector<MotionKeyframe>& frames, int i, bool loop, Get get) {
    const int n = static_cast<int>(frames.size());
    if (n < 2) return 0;

    int previous = i - 1;
    int next = i + 1;
    if (previous < 0) previous = loop ? n - 1 : 0;
    if (next >= n) next = loop ? 0 : n - 1;
    if (previous == next) return 0;

    Real span = frames[static_cast<size_t>(next)].time - frames[static_cast<size_t>(previous)].time;
    if (loop && span <= Real(0)) {
        // Wrapped across the loop point; the interval is one period long.
        span += frames.back().time - frames.front().time;
    }
    if (std::abs(span) < Real(1e-9)) return 0;
    return (get(frames[static_cast<size_t>(next)]) - get(frames[static_cast<size_t>(previous)])) /
           span;
}

}  // namespace

MotionPose Motion2D::sample(Real time) const {
    MotionPose pose;
    pose.resize(static_cast<size_t>(jointCount));
    if (keyframes.empty()) return pose;
    if (keyframes.size() == 1) {
        pose.rootPosition = keyframes[0].rootPosition;
        pose.rootAngle = keyframes[0].rootAngle;
        for (int j = 0; j < jointCount; ++j) {
            pose.jointAngles[static_cast<size_t>(j)] =
                j < static_cast<int>(keyframes[0].jointAngles.size())
                    ? keyframes[0].jointAngles[static_cast<size_t>(j)]
                    : Real(0);
        }
        return pose;
    }

    const Real start = keyframes.front().time;
    Real local = time;
    if (loop) {
        const Real span = duration();
        local = start + (time - start) - span * std::floor((time - start) / span);
    } else {
        local = clamp(time, start, keyframes.back().time);
    }

    // Segment containing `local`.
    size_t i = 0;
    while (i + 2 < keyframes.size() && keyframes[i + 1].time <= local) ++i;
    const MotionKeyframe& a = keyframes[i];
    const MotionKeyframe& b = keyframes[i + 1];
    const Real span = b.time - a.time;
    const Real t = span > Real(1e-9) ? (local - a.time) / span : Real(0);

    auto interpolate = [&](auto get) -> Real {
        const Real pa = get(a);
        const Real pb = get(b);
        if (interpolation == MotionInterpolation::Linear) return lerp(pa, pb, t);
        const Real ma = tangent(keyframes, static_cast<int>(i), loop, get) * span;
        const Real mb = tangent(keyframes, static_cast<int>(i + 1), loop, get) * span;
        const Hermite h = Hermite::at(t);
        return h.h00 * pa + h.h10 * ma + h.h01 * pb + h.h11 * mb;
    };

    pose.rootPosition.x = interpolate([](const MotionKeyframe& f) { return f.rootPosition.x; });
    pose.rootPosition.y = interpolate([](const MotionKeyframe& f) { return f.rootPosition.y; });
    // Never wrapped: a clip that rotates through a full turn has to be able to
    // say so, which is exactly what a flip needs.
    pose.rootAngle = interpolate([](const MotionKeyframe& f) { return f.rootAngle; });

    for (int j = 0; j < jointCount; ++j) {
        pose.jointAngles[static_cast<size_t>(j)] = interpolate([j](const MotionKeyframe& f) {
            return j < static_cast<int>(f.jointAngles.size())
                       ? f.jointAngles[static_cast<size_t>(j)]
                       : Real(0);
        });
    }
    return pose;
}

MotionPose Motion2D::sampleVelocity(Real time) const {
    MotionPose velocity;
    velocity.resize(static_cast<size_t>(jointCount));
    if (keyframes.size() < 2) return velocity;

    const Real start = keyframes.front().time;
    Real local = time;
    if (loop) {
        const Real span = duration();
        local = start + (time - start) - span * std::floor((time - start) / span);
    } else {
        local = clamp(time, start, keyframes.back().time);
    }

    size_t i = 0;
    while (i + 2 < keyframes.size() && keyframes[i + 1].time <= local) ++i;
    const MotionKeyframe& a = keyframes[i];
    const MotionKeyframe& b = keyframes[i + 1];
    const Real span = b.time - a.time;
    if (span <= Real(1e-9)) return velocity;
    const Real t = (local - a.time) / span;

    auto derivative = [&](auto get) -> Real {
        const Real pa = get(a);
        const Real pb = get(b);
        if (interpolation == MotionInterpolation::Linear) return (pb - pa) / span;
        const Real ma = tangent(keyframes, static_cast<int>(i), loop, get) * span;
        const Real mb = tangent(keyframes, static_cast<int>(i + 1), loop, get) * span;
        const Hermite h = Hermite::at(t);
        // d/dt = (d/du) * (du/dt), and du/dt is 1/span.
        return (h.d00 * pa + h.d10 * ma + h.d01 * pb + h.d11 * mb) / span;
    };

    velocity.rootPosition.x = derivative([](const MotionKeyframe& f) { return f.rootPosition.x; });
    velocity.rootPosition.y = derivative([](const MotionKeyframe& f) { return f.rootPosition.y; });
    velocity.rootAngle = derivative([](const MotionKeyframe& f) { return f.rootAngle; });
    for (int j = 0; j < jointCount; ++j) {
        velocity.jointAngles[static_cast<size_t>(j)] = derivative([j](const MotionKeyframe& f) {
            return j < static_cast<int>(f.jointAngles.size())
                       ? f.jointAngles[static_cast<size_t>(j)]
                       : Real(0);
        });
    }
    return velocity;
}

MotionPose Motion2D::samplePhase(Real phase) const {
    if (keyframes.empty()) return sample(0);
    return sample(keyframes.front().time + timeAt(phase));
}

MotionPose Motion2D::samplePhaseVelocity(Real phase) const {
    if (keyframes.empty()) return sampleVelocity(0);
    return sampleVelocity(keyframes.front().time + timeAt(phase));
}

// ---------------------------------------------------------------- json

Motion2D Motion2D::fromJson(const Json& json, std::string* error) {
    Motion2D motion;
    if (error) error->clear();
    if (!json.isObject()) {
        if (error) *error = "motion must be a JSON object";
        return motion;
    }

    motion.name = json["name"].string("motion");
    motion.loop = json["loop"].boolean(false);
    motion.jointCount = json["joint_count"].integer(kJointCount);
    const std::string mode = json["interpolation"].string("hermite");
    motion.interpolation =
        (mode == "linear") ? MotionInterpolation::Linear : MotionInterpolation::Hermite;

    const Json& frames = json["frames"];
    for (int i = 0; i < static_cast<int>(frames.size()); ++i) {
        const Json& entry = frames[i];
        MotionKeyframe frame;
        frame.time = entry["time"].real(0);
        frame.rootPosition = entry["root_position"].vec2(Vec2(0, 1));
        frame.rootAngle = entry["root_angle"].real(0);
        frame.jointAngles = entry["joints"].realArray();
        frame.jointAngles.resize(static_cast<size_t>(motion.jointCount), Real(0));
        motion.keyframes.push_back(frame);
    }
    motion.sortByTime();

    const std::string problem = motion.validate();
    if (!problem.empty() && error) *error = problem;
    return motion;
}

Motion2D Motion2D::loadFile(const std::string& path, std::string* error) {
    std::string parseError;
    const Json json = Json::parseFile(path, &parseError);
    if (!parseError.empty()) {
        if (error) *error = parseError;
        return Motion2D{};
    }
    return fromJson(json, error);
}

Json Motion2D::toJson() const {
    Json root = Json::object();
    root.set("name", Json(name));
    root.set("loop", Json(loop));
    root.set("joint_count", Json(jointCount));
    root.set("interpolation",
             Json(interpolation == MotionInterpolation::Linear ? "linear" : "hermite"));

    Json frames = Json::array();
    for (const MotionKeyframe& frame : keyframes) {
        Json entry = Json::object();
        entry.set("time", Json(double(frame.time)));
        Json position = Json::array();
        position.push(Json(double(frame.rootPosition.x)));
        position.push(Json(double(frame.rootPosition.y)));
        entry.set("root_position", std::move(position));
        entry.set("root_angle", Json(double(frame.rootAngle)));
        Json joints = Json::array();
        for (const Real angle : frame.jointAngles) joints.push(Json(double(angle)));
        entry.set("joints", std::move(joints));
        frames.push(std::move(entry));
    }
    root.set("frames", std::move(frames));
    return root;
}

bool Motion2D::writeFile(const std::string& path) const { return toJson().writeFile(path); }

// ---------------------------------------------------------------- validation

std::string Motion2D::validate() const {
    if (keyframes.empty()) return "motion has no keyframes";
    if (jointCount <= 0) return "motion has a non-positive joint count";

    for (size_t i = 0; i < keyframes.size(); ++i) {
        const MotionKeyframe& frame = keyframes[i];
        if (!std::isfinite(frame.time) || !isFinite(frame.rootPosition) ||
            !std::isfinite(frame.rootAngle)) {
            return "keyframe " + std::to_string(i) + " has a non-finite value";
        }
        if (frame.jointAngles.size() != static_cast<size_t>(jointCount)) {
            return "keyframe " + std::to_string(i) + " has " +
                   std::to_string(frame.jointAngles.size()) + " joint angles, expected " +
                   std::to_string(jointCount);
        }
        for (const Real angle : frame.jointAngles) {
            if (!std::isfinite(angle)) {
                return "keyframe " + std::to_string(i) + " has a non-finite joint angle";
            }
        }
        if (i > 0 && !(frame.time > keyframes[i - 1].time)) {
            return "keyframe " + std::to_string(i) + " does not advance in time";
        }
    }
    if (duration() <= Real(0)) return "motion has zero duration";
    return std::string();
}

int Motion2D::clampToLimits(const Humanoid2DConfig& config) {
    int clamped = 0;
    for (MotionKeyframe& frame : keyframes) {
        const size_t limit =
            std::min(frame.jointAngles.size(), config.joints.size());
        for (size_t j = 0; j < limit; ++j) {
            const JointConfig& jc = config.joints[j];
            const Real original = frame.jointAngles[j];
            const Real bounded = clamp(original, jc.lowerLimit, jc.upperLimit);
            if (bounded != original) {
                frame.jointAngles[j] = bounded;
                ++clamped;
            }
        }
    }
    return clamped;
}

}  // namespace aibf
