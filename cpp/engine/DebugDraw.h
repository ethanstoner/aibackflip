// Draws a physics world and a humanoid, with the overlays needed to see why the
// simulation is doing what it is doing. Shared by the interactive simulator, the
// animator, and trained-policy playback so all three show the same thing.
#pragma once

#include <deque>

#include "engine/Renderer2D.h"
#include "humanoid/Humanoid2D.h"
#include "physics/World2D.h"

namespace aibf {

struct DebugDrawOptions {
    bool bodies = true;
    bool outlines = false;
    bool joints = true;
    bool jointLimits = false;
    bool contacts = true;
    bool contactImpulses = false;
    bool centerOfMass = true;
    bool velocities = false;
    bool jointTargets = false;
    bool supportPolygon = true;
    bool trail = false;

    Real markerPixels = 5;
};

// Fixed-length history of a point, used for the motion trails acrobatics need.
class Trail {
public:
    explicit Trail(size_t capacity = 240) : capacity_(capacity) {}
    void push(const Vec2& p) {
        points_.push_back(p);
        while (points_.size() > capacity_) points_.pop_front();
    }
    void clear() { points_.clear(); }
    const std::deque<Vec2>& points() const { return points_; }

private:
    size_t capacity_;
    std::deque<Vec2> points_;
};

// Colour a link by which part of the body it is, with the right-hand limbs
// darkened. In 2D the left and right sides sit in the same plane, so without
// that they are indistinguishable.
Color linkColor(int linkId);

void drawWorld(Renderer2D& renderer, const World2D& world, const Camera2D& camera,
               const DebugDrawOptions& options);
void drawHumanoid(Renderer2D& renderer, const World2D& world, const Humanoid2D& figure,
                  const Camera2D& camera, const DebugDrawOptions& options);
void drawTrail(Renderer2D& renderer, const Trail& trail, const Color& color);
void drawMouseSpring(Renderer2D& renderer, const World2D& world, const Camera2D& camera);

}  // namespace aibf
