// Simulation host. Grows into the environment server across M1-M3; for now it
// reports the build configuration and can smoke-test that GLFW links and can
// reach a display, which is the part of the toolchain most likely to be broken.
#include <cstdio>
#include <cstring>

#include <GLFW/glfw3.h>

#include "core/Math.h"
#include "core/Rng.h"

namespace {

void printUsage() {
    std::printf(
        "aibackflip simulation host\n"
        "\n"
        "  --check-gl    initialise GLFW, report the version, and exit\n"
        "  --headless    no window (default once the env server lands)\n"
        "  --help        this message\n");
}

int checkGl() {
    if (!glfwInit()) {
        const char* err = nullptr;
        glfwGetError(&err);
        std::printf("glfwInit failed: %s\n", err ? err : "unknown");
        return 1;
    }
    int major = 0, minor = 0, rev = 0;
    glfwGetVersion(&major, &minor, &rev);
    std::printf("GLFW %d.%d.%d (%s)\n", major, minor, rev, glfwGetVersionString());

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(64, 64, "aibackflip probe", nullptr, nullptr);
    if (!window) {
        const char* err = nullptr;
        glfwGetError(&err);
        std::printf("no GL 3.3 core context: %s\n", err ? err : "unknown");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    std::printf("GL 3.3 core context OK\n");
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool headless = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            printUsage();
            return 0;
        }
        if (std::strcmp(argv[i], "--check-gl") == 0) return checkGl();
        if (std::strcmp(argv[i], "--headless") == 0) headless = true;
    }

    std::printf("aibackflip host (Real = %zu bytes, headless = %s)\n",
                sizeof(aibf::Real), headless ? "yes" : "no");
    std::printf("no simulation wired up yet; see docs/PROGRESS.md\n");
    return 0;
}
