#include "engine/DebugDraw.h"

#include <algorithm>
#include <cmath>

namespace aibf {

namespace {

bool isRightSide(int linkId) {
    return linkId == kUpperArmR || linkId == kLowerArmR || linkId == kUpperLegR ||
           linkId == kLowerLegR || linkId == kFootR;
}

}  // namespace

Color linkColor(int linkId) {
    Color base;
    switch (linkId) {
        case kPelvis:
        case kChest: base = palette::torso; break;
        case kHead: base = palette::head; break;
        case kFootL:
        case kFootR: base = palette::foot; break;
        case kUpperArmL:
        case kUpperArmR: base = palette::arm; break;
        case kLowerArmL:
        case kLowerArmR: base = palette::hand; break;
        default: base = palette::leg; break;
    }
    // The near/far convention: right-side limbs render darker so a sagittal
    // pose is readable even when both legs project onto the same line.
    return isRightSide(linkId) ? base.scaled(Real(0.55)) : base;
}

void drawWorld(Renderer2D& renderer, const World2D& world, const Camera2D& camera,
               const DebugDrawOptions& options) {
    const Real marker = camera.pixelsToWorld(options.markerPixels, 1080);

    if (options.contacts) {
        for (const Manifold& m : world.manifolds()) {
            for (int i = 0; i < m.pointCount; ++i) {
                const ContactPoint& cp = m.points[i];
                // Speculative contacts have not touched yet; showing them at
                // half alpha distinguishes "about to hit" from "carrying load".
                const bool touching = cp.separation <= Real(0);
                renderer.circle(cp.position, marker * Real(0.8),
                                palette::contact.withAlpha(touching ? Real(1) : Real(0.35)));
                if (options.contactImpulses && cp.normalImpulse > Real(0)) {
                    const Real scale = Real(0.004);
                    renderer.arrow(cp.position, cp.position - m.normal * (cp.normalImpulse * scale),
                                   marker, palette::contact);
                }
            }
        }
    }

    if (options.velocities) {
        for (const RigidBody2D& body : world.bodies()) {
            if (body.isStatic) continue;
            renderer.arrow(body.position, body.position + body.velocity * Real(0.1), marker,
                           palette::accent.withAlpha(Real(0.8)));
        }
    }
}

void drawHumanoid(Renderer2D& renderer, const World2D& world, const Humanoid2D& figure,
                  const Camera2D& camera, const DebugDrawOptions& options) {
    const Humanoid2DConfig& config = figure.config();
    const Real marker = camera.pixelsToWorld(options.markerPixels, 1080);

    if (options.bodies) {
        // Explicit back-to-front order: far limbs, torso, near limbs. Index
        // order would put the arms behind the legs and bury them in the torso.
        static constexpr int kDrawOrder[] = {
            kUpperLegR, kLowerLegR, kFootR,
            kUpperArmR, kLowerArmR,
            kPelvis,    kChest,     kHead,
            kUpperLegL, kLowerLegL, kFootL,
            kUpperArmL, kLowerArmL,
        };
        static_assert(sizeof(kDrawOrder) / sizeof(kDrawOrder[0]) == kLinkCount,
                      "every link must appear exactly once in the draw order");
        for (const int i : kDrawOrder) {
            if (i >= static_cast<int>(config.links.size())) continue;
            const RigidBody2D& body = figure.link(world, i);
            renderer.capsule(body.position, body.angle, body.radius, body.halfLength,
                             linkColor(i));
        }
    }

    if (options.outlines) {
        for (int i = 0; i < static_cast<int>(config.links.size()); ++i) {
            const RigidBody2D& body = figure.link(world, i);
            renderer.capsuleOutline(body.position, body.angle, body.radius, body.halfLength,
                                    palette::joint.withAlpha(Real(0.25)));
        }
    }

    if (options.joints) {
        for (int i = 0; i < static_cast<int>(config.joints.size()); ++i) {
            const RevoluteJoint2D& rj = figure.joint(world, i);
            const Vec2 pivot = world.body(rj.bodyA).localToWorld(rj.localAnchorA);
            renderer.circle(pivot, marker * Real(0.55), palette::joint);
        }
    }

    if (options.jointTargets) {
        // A stub pointing where the motor is currently asking the child link to
        // be, next to where it actually is. The gap between them is the
        // tracking error, visible at a glance.
        for (int i = 0; i < static_cast<int>(config.joints.size()); ++i) {
            const RevoluteJoint2D& rj = figure.joint(world, i);
            if (!rj.enableMotor) continue;
            const RigidBody2D& parent = world.body(rj.bodyA);
            const RigidBody2D& child = world.body(rj.bodyB);
            const Vec2 pivot = parent.localToWorld(rj.localAnchorA);
            const Real targetAngle = parent.angle + rj.referenceAngle + rj.targetAngle;
            const Real reach = child.halfLength + child.radius;
            renderer.line(pivot, pivot + rotate(Vec2(0, -reach), targetAngle),
                          palette::com.withAlpha(Real(0.75)));
        }
    }

    if (options.centerOfMass) {
        const Vec2 com = figure.centerOfMass(world);
        renderer.circleOutline(com, marker * Real(1.4), palette::com);
        renderer.cross(com, marker * Real(1.4), palette::com);
        // Vertical projection of the centre of mass. Whether it lands inside
        // the feet is the whole story of a balance controller.
        renderer.line(com, Vec2(com.x, 0), palette::com.withAlpha(Real(0.4)));
    }

    if (options.supportPolygon) {
        // Horizontal extent of whatever is currently touching the ground.
        Real minX = Real(1e9), maxX = Real(-1e9);
        bool any = false;
        for (const Manifold& m : world.manifolds()) {
            if (m.bodyB != -1) continue;  // ground contacts only
            for (int i = 0; i < m.pointCount; ++i) {
                if (m.points[i].separation > Real(0)) continue;
                minX = std::min(minX, m.points[i].position.x);
                maxX = std::max(maxX, m.points[i].position.x);
                any = true;
            }
        }
        if (any) {
            const Real y = marker * Real(0.5);
            renderer.thickLine(Vec2(minX, y), Vec2(maxX, y), marker * Real(0.5),
                               palette::accent.withAlpha(Real(0.85)));
        }
    }
}

void drawTrail(Renderer2D& renderer, const Trail& trail, const Color& color) {
    const auto& points = trail.points();
    if (points.size() < 2) return;
    // Older samples fade out, so the direction of travel reads without arrows.
    for (size_t i = 1; i < points.size(); ++i) {
        const Real age = Real(i) / Real(points.size());
        renderer.line(points[i - 1], points[i], color.withAlpha(age * Real(0.9)));
    }
}

void drawMouseSpring(Renderer2D& renderer, const World2D& world, const Camera2D& camera) {
    const MouseSpring& mouse = world.mouse();
    if (!mouse.active || mouse.body < 0) return;
    const RigidBody2D& body = world.body(mouse.body);
    const Vec2 grabbed = body.localToWorld(mouse.localAnchor);
    const Real marker = camera.pixelsToWorld(Real(5), 1080);
    renderer.line(grabbed, mouse.target, palette::com);
    renderer.circleOutline(mouse.target, marker, palette::com);
    renderer.circle(grabbed, marker * Real(0.6), palette::com);
}

}  // namespace aibf
