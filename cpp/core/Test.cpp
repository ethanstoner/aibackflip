#include "core/Test.h"

#include <algorithm>
#include <chrono>

namespace aibf::test {

std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

void fail(const char* file, int line, const std::string& message) {
    throw Failure{std::string(file) + ":" + std::to_string(line) + ": " + message};
}

int runAll(const char* filter) {
    std::string needle = filter ? filter : "";
    auto& cases = registry();
    std::stable_sort(cases.begin(), cases.end(), [](const Case& a, const Case& b) {
        return a.suite < b.suite;
    });

    int passed = 0, failed = 0, skipped = 0;
    std::string currentSuite;
    auto start = std::chrono::steady_clock::now();

    for (const Case& c : cases) {
        std::string full = c.suite + "." + c.name;
        if (!needle.empty() && full.find(needle) == std::string::npos) {
            ++skipped;
            continue;
        }
        if (c.suite != currentSuite) {
            currentSuite = c.suite;
            std::printf("\n[%s]\n", currentSuite.c_str());
        }
        try {
            c.fn();
            std::printf("  PASS  %s\n", c.name.c_str());
            ++passed;
        } catch (const Failure& f) {
            std::printf("  FAIL  %s\n        %s\n", c.name.c_str(), f.message.c_str());
            ++failed;
        } catch (const std::exception& e) {
            std::printf("  FAIL  %s\n        unexpected exception: %s\n", c.name.c_str(), e.what());
            ++failed;
        } catch (...) {
            std::printf("  FAIL  %s\n        unexpected non-standard exception\n", c.name.c_str());
            ++failed;
        }
    }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - start)
                  .count();
    std::printf("\n%d passed, %d failed", passed, failed);
    if (skipped) std::printf(", %d filtered out", skipped);
    std::printf("  (%lld ms)\n", static_cast<long long>(ms));
    return failed;
}

}  // namespace aibf::test
