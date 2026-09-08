// PCG32. Deterministic across platforms and compilers, unlike <random>'s
// distributions, so a seed reproduces a training run exactly.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "core/Math.h"

namespace aibf {

class Rng {
public:
    explicit Rng(uint64_t seed = 0x853c49e6748fea9bULL, uint64_t stream = 0xda3e39cb94b95bdbULL) {
        seedWith(seed, stream);
    }

    void seedWith(uint64_t seed, uint64_t stream = 0xda3e39cb94b95bdbULL) {
        state_ = 0;
        inc_ = (stream << 1u) | 1u;
        nextU32();
        state_ += seed;
        nextU32();
        hasSpareGaussian_ = false;
    }

    uint32_t nextU32() {
        uint64_t old = state_;
        state_ = old * 6364136223846793005ULL + inc_;
        uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = static_cast<uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    // Uniform in [0, 1).
    Real uniform() { return Real(nextU32() >> 8) * Real(1.0 / 16777216.0); }
    Real uniform(Real lo, Real hi) { return lo + (hi - lo) * uniform(); }

    // Uniform integer in [0, bound), rejection-sampled so it stays unbiased.
    uint32_t below(uint32_t bound) {
        uint32_t threshold = (~bound + 1u) % bound;
        for (;;) {
            uint32_t r = nextU32();
            if (r >= threshold) return r % bound;
        }
    }

    bool chance(Real probability) { return uniform() < probability; }

    // Marsaglia polar method; caches the second variate.
    Real gaussian() {
        if (hasSpareGaussian_) {
            hasSpareGaussian_ = false;
            return spareGaussian_;
        }
        Real u, v, s;
        do {
            u = uniform(Real(-1), Real(1));
            v = uniform(Real(-1), Real(1));
            s = u * u + v * v;
        } while (s >= Real(1) || s == Real(0));
        Real f = std::sqrt(Real(-2) * std::log(s) / s);
        spareGaussian_ = v * f;
        hasSpareGaussian_ = true;
        return u * f;
    }

    Real gaussian(Real mean, Real stddev) { return mean + stddev * gaussian(); }

    Vec2 uniformInUnitCircle() {
        for (;;) {
            Vec2 p(uniform(Real(-1), Real(1)), uniform(Real(-1), Real(1)));
            if (lengthSq(p) <= Real(1)) return p;
        }
    }

    Vec3 uniformOnUnitSphere() {
        Real z = uniform(Real(-1), Real(1));
        Real theta = uniform(Real(0), kTwoPi);
        Real r = std::sqrt(std::max(Real(0), Real(1) - z * z));
        return {r * std::cos(theta), r * std::sin(theta), z};
    }

private:
    uint64_t state_ = 0;
    uint64_t inc_ = 0;
    Real spareGaussian_ = 0;
    bool hasSpareGaussian_ = false;
};

}  // namespace aibf
