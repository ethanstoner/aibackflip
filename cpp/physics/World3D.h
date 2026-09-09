// The 3D physics world: bodies, ball and hinge joints, static half-spaces, and
// the same sequential-impulse solver structure as World2D.
//
// step() advances exactly one fixed substep. Nothing in here knows about frame
// rate, rendering, or episodes, which is what lets a headless training run and a
// rendered playback share the same dynamics.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "physics/Collision3D.h"
#include "physics/Joint3D.h"
#include "physics/RigidBody3D.h"
#include "physics/Solver2D.h"

namespace aibf {

// Impulses carried from one substep to the next. Two tangents rather than one,
// because a 3D contact plane has two friction directions.
struct CachedImpulse3D {
    Real normal = 0;
    Real tangent1 = 0;
    Real tangent2 = 0;
};

struct WorldStats3D {
    int contactCount = 0;
    int manifoldCount = 0;
    Real maxPenetration = 0;
    Real maxJointAnchorError = 0;
    Real maxSpeed = 0;
    int velocityClampEvents = 0;
    bool unstable = false;
};

class World3D {
public:
    World3D();

    Vec3 gravity{0, Real(-9.81), 0};
    SolverConfig solver;

    Real maxLinearVelocity = 200;
    Real maxAngularVelocity = 200;

    int32_t addBody(const RigidBody3D& body);
    int32_t addBallJoint(const BallJoint3D& joint);
    int32_t addHingeJoint(const HingeJoint3D& joint);
    int32_t addHalfSpace(const HalfSpace3D& plane);

    void clear();

    RigidBody3D& body(int32_t i) { return bodies_[static_cast<size_t>(i)]; }
    const RigidBody3D& body(int32_t i) const { return bodies_[static_cast<size_t>(i)]; }
    BallJoint3D& ballJoint(int32_t i) { return balls_[static_cast<size_t>(i)]; }
    const BallJoint3D& ballJoint(int32_t i) const { return balls_[static_cast<size_t>(i)]; }
    HingeJoint3D& hingeJoint(int32_t i) { return hinges_[static_cast<size_t>(i)]; }
    const HingeJoint3D& hingeJoint(int32_t i) const { return hinges_[static_cast<size_t>(i)]; }
    const HalfSpace3D& halfSpace(int32_t i) const { return planes_[static_cast<size_t>(i)]; }

    int32_t bodyCount() const { return static_cast<int32_t>(bodies_.size()); }
    int32_t ballJointCount() const { return static_cast<int32_t>(balls_.size()); }
    int32_t hingeJointCount() const { return static_cast<int32_t>(hinges_.size()); }
    int32_t halfSpaceCount() const { return static_cast<int32_t>(planes_.size()); }

    std::vector<RigidBody3D>& bodies() { return bodies_; }
    const std::vector<RigidBody3D>& bodies() const { return bodies_; }
    const std::vector<Manifold3D>& manifolds() const { return manifolds_; }

    void step(Real dt);

    // Rebuilds contact manifolds without integrating or solving. Needed on reset
    // so contact state is populated before the first observation is read;
    // stepping the world to get it would silently advance the physics and let
    // the motors' damping brake whatever initial velocity was just set.
    void refreshContacts();

    bool hasContact(int32_t bodyIndex) const;

    Vec3 centerOfMass() const;
    Vec3 centerOfMassVelocity() const;
    Real totalMass() const;
    Real kineticEnergy() const;
    Real potentialEnergy(Real datum = 0) const;
    // Total angular momentum about the centre of mass. Conserved in free flight,
    // and the quantity a tumbling figure's behaviour is judged against.
    Vec3 angularMomentum() const;

    const WorldStats3D& stats() const { return stats_; }
    void resetStats() { stats_ = WorldStats3D{}; }

    // Drops every cached impulse. Call after teleporting bodies on reset, or the
    // solver warm-starts from forces belonging to the previous episode.
    void clearContactCache();

private:
    void integrateVelocities(Real dt);
    void integratePositions(Real dt);
    void generateManifolds(std::vector<Manifold3D>& out) const;
    void restoreCachedImpulses();
    void prepareContacts();
    void warmStart();
    void solveVelocities(Real dt);
    void solvePositions();
    void storeImpulses();
    void checkForInstability();

    RigidBody3D& bodyRef(int32_t i) { return i < 0 ? worldBody_ : bodies_[static_cast<size_t>(i)]; }
    const RigidBody3D& bodyRef(int32_t i) const {
        return i < 0 ? worldBody_ : bodies_[static_cast<size_t>(i)];
    }

    std::vector<RigidBody3D> bodies_;
    std::vector<BallJoint3D> balls_;
    std::vector<HingeJoint3D> hinges_;
    std::vector<HalfSpace3D> planes_;
    std::vector<Manifold3D> manifolds_;
    mutable std::vector<Manifold3D> positionScratch_;

    std::unordered_map<uint64_t, CachedImpulse3D> impulseCache_;
    std::unordered_map<uint64_t, CachedImpulse3D> nextImpulseCache_;

    // Stands in for "the immovable world" on plane contacts, so the contact
    // solver has one code path instead of a static special case.
    RigidBody3D worldBody_;
    WorldStats3D stats_;
};

}  // namespace aibf
