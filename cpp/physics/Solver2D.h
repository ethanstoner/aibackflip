// Solver tuning shared by joints and contacts.
#pragma once

#include "core/Math.h"

namespace aibf {

struct SolverConfig {
    // Sequential-impulse iteration counts. Ten velocity iterations is enough for
    // a 12-joint chain; below about six the humanoid's legs visibly stretch on
    // landing. The position pass needs far fewer because it converges directly
    // on the error rather than on its derivative.
    int velocityIterations = 10;
    int positionIterations = 4;

    // Fraction of remaining contact penetration removed per position iteration.
    Real contactBaumgarte = Real(0.2);

    // Joint anchors get the full correction each iteration - a 12-joint chain
    // has to converge in a handful of passes, and unlike contacts there is no
    // one-sided inequality for an over-correction to fight against.
    Real jointRelaxation = Real(1);

    // Bias rate for the one-sided angular limit constraints.
    Real limitBaumgarte = Real(0.2);

    // Caps on a single position correction, so one deep penetration cannot
    // teleport a body across the world.
    Real maxLinearCorrection = Real(0.2);
    Real maxAngularCorrection = Real(8) * kDeg2Rad;

    // Joint anchors can be stabilised either by a dedicated position pass
    // (default; removes drift without injecting energy) or by a Baumgarte bias
    // folded into the velocity solve. Both are implemented so the two can be
    // measured against each other rather than argued about - the numbers are in
    // docs/PROGRESS.md.
    bool useJointPositionSolver = true;
    Real jointBaumgarte = Real(0.2);

    // Relative normal speed below which a contact counts as resting and
    // restitution is suppressed. Without it a body at rest jitters forever.
    Real restitutionThreshold = Real(0.5);
};

}  // namespace aibf
