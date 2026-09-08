#include "engine/Window.h"

#include <cstdio>
#include <vector>

#include "core/Png.h"
#include "engine/GL.h"

namespace aibf {

void Input::clearEdges() {
    for (int i = 0; i < kMaxKeys; ++i) {
        keyPressed[i] = false;
        keyReleased[i] = false;
    }
    for (int i = 0; i < kMaxButtons; ++i) {
        mousePressed[i] = false;
        mouseReleased[i] = false;
    }
    scroll = 0;
    mouseDelta = Vec2(0, 0);
}

Window::~Window() { destroy(); }

bool Window::create(int width, int height, const std::string& title, int samples) {
    if (!glfwInit()) {
        const char* error = nullptr;
        glfwGetError(&error);
        std::fprintf(stderr, "glfwInit failed: %s\n", error ? error : "unknown");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, samples);

    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        const char* error = nullptr;
        glfwGetError(&error);
        std::fprintf(stderr, "could not create a GL 3.3 core window: %s\n",
                     error ? error : "unknown");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    std::string missing;
    if (!gl::load(&missing)) {
        std::fprintf(stderr, "OpenGL entry point unavailable: %s\n", missing.c_str());
        destroy();
        return false;
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetKeyCallback(window_, keyCallback);
    glfwSetMouseButtonCallback(window_, mouseButtonCallback);
    glfwSetCursorPosCallback(window_, cursorCallback);
    glfwSetScrollCallback(window_, scrollCallback);
    glfwSetFramebufferSizeCallback(window_, resizeCallback);

    glfwGetFramebufferSize(window_, &width_, &height_);
    glViewport(0, 0, width_, height_);
    if (samples > 0) glEnable(gl::kMultisample);
    return true;
}

void Window::destroy() {
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
    }
}

bool Window::isOpen() const { return window_ && !glfwWindowShouldClose(window_); }

void Window::requestClose() {
    if (window_) glfwSetWindowShouldClose(window_, 1);
}

void Window::pollEvents() {
    input_.clearEdges();
    glfwPollEvents();
}

void Window::present() {
    if (window_) glfwSwapBuffers(window_);
}

void Window::setTitle(const std::string& title) {
    if (window_) glfwSetWindowTitle(window_, title.c_str());
}

bool Window::saveScreenshot(const std::string& path) const {
    if (!window_ || width_ <= 0 || height_ <= 0) return false;
    std::vector<uint8_t> pixels(static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3);
    // Rows arrive bottom-up and may not be 4-byte aligned at odd widths.
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    return writePng(path, width_, height_, 3, pixels.data(), true);
}

// ---------------------------------------------------------------- callbacks

void Window::keyCallback(GLFWwindow* window, int key, int, int action, int mods) {
    Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (!self || key < 0 || key >= Input::kMaxKeys) return;
    if (action == GLFW_PRESS) {
        self->input_.keyDown[key] = true;
        self->input_.keyPressed[key] = true;
    } else if (action == GLFW_RELEASE) {
        self->input_.keyDown[key] = false;
        self->input_.keyReleased[key] = true;
    }
    self->input_.shift = (mods & GLFW_MOD_SHIFT) != 0;
    self->input_.ctrl = (mods & GLFW_MOD_CONTROL) != 0;
    self->input_.alt = (mods & GLFW_MOD_ALT) != 0;
}

void Window::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (!self || button < 0 || button >= Input::kMaxButtons) return;
    if (action == GLFW_PRESS) {
        self->input_.mouseDown[button] = true;
        self->input_.mousePressed[button] = true;
    } else if (action == GLFW_RELEASE) {
        self->input_.mouseDown[button] = false;
        self->input_.mouseReleased[button] = true;
    }
    self->input_.shift = (mods & GLFW_MOD_SHIFT) != 0;
    self->input_.ctrl = (mods & GLFW_MOD_CONTROL) != 0;
}

void Window::cursorCallback(GLFWwindow* window, double x, double y) {
    Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (!self) return;
    const Vec2 position(static_cast<Real>(x), static_cast<Real>(y));
    // The first sample after the cursor enters would otherwise report a delta
    // the size of the window and fling whatever is being dragged.
    if (self->hasCursorSample_) {
        self->input_.mouseDelta += position - self->input_.mouseScreen;
    }
    self->input_.mouseScreen = position;
    self->hasCursorSample_ = true;
}

void Window::scrollCallback(GLFWwindow* window, double, double dy) {
    Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (self) self->input_.scroll += static_cast<Real>(dy);
}

void Window::resizeCallback(GLFWwindow* window, int width, int height) {
    Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (!self) return;
    self->width_ = width;
    self->height_ = height;
    glViewport(0, 0, width, height);
}

}  // namespace aibf
