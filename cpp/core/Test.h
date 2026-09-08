// Minimal self-registering test framework. Deliberately dependency-free: the
// physics tests need to run in CI and on a fresh clone without fetching anything.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace aibf::test {

struct Case {
    std::string suite;
    std::string name;
    std::function<void()> fn;
};

struct Failure {
    std::string message;
};

std::vector<Case>& registry();

struct Registrar {
    Registrar(const char* suite, const char* name, std::function<void()> fn) {
        registry().push_back({suite, name, std::move(fn)});
    }
};

[[noreturn]] void fail(const char* file, int line, const std::string& message);

// Runs every registered case whose "suite.name" contains `filter`
// (nullptr or empty runs everything). Returns the number of failures.
int runAll(const char* filter);

}  // namespace aibf::test

#define AIBF_CAT_(a, b) a##b
#define AIBF_CAT(a, b) AIBF_CAT_(a, b)

#define TEST(suite, name)                                                        \
    static void AIBF_CAT(aibf_test_, __LINE__)();                                \
    static ::aibf::test::Registrar AIBF_CAT(aibf_reg_, __LINE__)(                \
        #suite, #name, AIBF_CAT(aibf_test_, __LINE__));                          \
    static void AIBF_CAT(aibf_test_, __LINE__)()

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) ::aibf::test::fail(__FILE__, __LINE__, "CHECK failed: " #cond); \
    } while (0)

#define CHECK_NEAR(actual, expected, tol)                                        \
    do {                                                                         \
        double a_ = double(actual), e_ = double(expected), t_ = double(tol);      \
        if (!(std::abs(a_ - e_) <= t_)) {                                         \
            char buf_[256];                                                       \
            std::snprintf(buf_, sizeof(buf_),                                      \
                          "CHECK_NEAR failed: %s = %.9g, expected %.9g (tol %.3g)", \
                          #actual, a_, e_, t_);                                    \
            ::aibf::test::fail(__FILE__, __LINE__, buf_);                          \
        }                                                                         \
    } while (0)

#define CHECK_VEC2_NEAR(actual, ex, ey, tol)                                     \
    do {                                                                         \
        auto v_ = (actual);                                                       \
        CHECK_NEAR(v_.x, ex, tol);                                                \
        CHECK_NEAR(v_.y, ey, tol);                                                \
    } while (0)

#define CHECK_VEC3_NEAR(actual, ex, ey, ez, tol)                                 \
    do {                                                                         \
        auto v_ = (actual);                                                       \
        CHECK_NEAR(v_.x, ex, tol);                                                \
        CHECK_NEAR(v_.y, ey, tol);                                                \
        CHECK_NEAR(v_.z, ez, tol);                                                \
    } while (0)
