// Entry point for the C++ test binary.
//   aibf_tests            run everything
//   aibf_tests Quat       run only cases whose "Suite.name" contains "Quat"
#include <cstdio>

#include "core/Test.h"

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    if (filter) std::printf("filter: %s\n", filter);
    return aibf::test::runAll(filter) == 0 ? 0 : 1;
}
