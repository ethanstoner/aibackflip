// The 2D physics world: bodies, revolute joints, static half-spaces, and a
// sequential-impulse solver.
//
// step() advances exactly one fixed substep. Nothing in here knows about frame
// rate, rendering, or episodes - that lives a layer up, which is what lets a
// headless training run and a rendered playback share the same dynamics.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "physics/Collision2D.h"
#include "physics/Joint2D.h"
#include "physics/RigidBody2D.h"
#include "physics/Solver2D.h"

namespace aibf {

// Impulses carried from one substep to the next. Warm starting is the single
// biggest contributor to a humanoid standing still instead of slowly sinking:
// without it, ten velocity iterations have to rediscover the support forces
// from scratch every substep and never quite converge.
struct CachedImpulse {
    Real normal = 0;
    Real tangent = 0;
};

struct WorldStats {
    int contactCount = 0;
    int manifoldCount = 0;
    Real maxPenetration = 0;        // positive number, depth of the worst overlap
    Real maxJointAnchorError = 0;   // metres of drift on the worst joint
    Real maxSpeed = 0;
    int velocityClampEvents = 0;    // non-zero means something is going wrong
    bool unstable = false;          // a NaN or infinity reached a body
};

// Soft point constraint used for mouse dragging. Deliberately soft, so pulling
// on a limb propagates through the joint chain and drags the whole body instead
// of ripping one capsule out of its socket.
struct MouseSpring {
    bool active = false;
    int32_t body = -1;
    Vec2 localAnchor;
    Vec2 target;
    Real maxForce = 2000;
    Real stiffness = 1200;
    Real damping = 60;
    Vec2 accumulated;
};

class World2D {
public:
    World2D();

    Vec2 gravity{0, Real(-9.81)};
    SolverConfig solver;

    // Generous ceilings. These are diagnostics, not a stability crutch: if
    // clamping ever fires during training, WorldStats reports it rather than
    // quietly hiding a solver bug.
    Real maxLinearVelocity = 200;
    Real maxAngularVelocity = 200;

    int32_t addBody(const RigidBody2D& body);
    int32_t addJoint(const RevoluteJoint2D& joint);
    int32_t addHalfSpace(const HalfSpace& plane);

    void clear();

    RigidBody2D& body(int32_t i) { return bodies_[static_cast<size_t>(i)]; }
    const RigidBody2D& body(int32_t i) const { return bodies_[static_cast<size_t>(i)]; }
    RevoluteJoint2D& joint(int32_t i) { return joints_[static_cast<size_t>(i)]; }
    const RevoluteJoint2D& joint(int32_t i) const { return joints_[static_cast<size_t>(i)]; }
    const HalfSpace& halfSpace(int32_t i) const { return planes_[static_cast<size_t>(i)]; }

    int32_t bodyCount() const { return static_cast<int32_t>(bodies_.size()); }
    int32_t jointCount() const { return static_cast<int32_t>(joints_.size()); }
    int32_t halfSpaceCount() const { return static_cast<int32_t>(planes_.size()); }

    std::vector<RigidBody2D>& bodies() { return bodies_; }
    const std::vector<RigidBody2D>& bodies() const { return bodies_; }
    const std::vector<Manifold>& manifolds() const { return manifolds_; }

    // Advances one fixed substep.
    void step(Real dt);

    // True while any contact touches this body, computed from the manifolds of
    // the most recent step. Used for the foot-contact observation.
    bool hasContact(int32_t bodyIndex) const;
    Vec2 contactImpulseOn(int32_t bodyIndex) const;

    Vec2 centerOfMass() const;
    Vec2 centerOfMassVelocity() const;
    Real totalMass() const;
    Real kineticEnergy() const;
    Real potentialEnergy(Real datum = 0) const;

    MouseSpring& mouse() { return mouse_; }
    const MouseSpring& mouse() const { return mouse_; }
    void grab(int32_t bodyIndex, const Vec2& worldPoint);
    void releaseGrab() { mouse_.active = false; }
    // Nearest body whose capsule surface is within `radius` of the point, or -1.
    int32_t pickBody(const Vec2& worldPoint, Real radius = Real(0.05)) const;

    const WorldStats& stats() const { return stats_; }
    void resetStats() { stats_ = WorldStats{}; }

    // Drops every cached impulse. Call after teleporting bodies on reset, or the
    // solver warm-starts from forces that belonged to the previous episode.
    void clearContactCache();

private:
    void integrateVelocities(Real dt);
    void integratePositions(Real dt);
    void generateManifolds(std::vector<Manifold>& out) const;
    void restoreCachedImpulses();
    void prepareContacts(Real dt);
    void warmStart();
    void solveVelocities(Real dt);
    void solvePositions();
    void solveMouseSpring(Real dt);
    void storeImpulses();
    void checkForInstability();

    RigidBody2D& bodyRef(int32_t i) { return i < 0 ? worldBody_ : bodies_[static_cast<size_t>(i)]; }

    std::vector<RigidBody2D> bodies_;
    std::vector<RevoluteJoint2D> joints_;
    std::vector<HalfSpace> planes_;
    std::vector<Manifold> manifolds_;
    // Scratch for the position pass, kept as a member only to avoid reallocating
    // it on every iteration of every substep.
    mutable std::vector<Manifold> positionScratch_;

    std::unordered_map<uint64_t, CachedImpulse> impulseCache_;
    std::unordered_map<uint64_t, CachedImpulse> nextImpulseCache_;

    // Stands in for "the immovable world" on plane contacts, so the contact
    // solver has a single code path instead of a static special case.
    RigidBody2D worldBody_;
    MouseSpring mouse_;
    WorldStats stats_;
};

}  // namespace aibf
