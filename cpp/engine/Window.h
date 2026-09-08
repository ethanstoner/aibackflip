// GLFW window plus the input snapshot the tools read.
//
// Input is polled into a plain struct rather than exposed through callbacks so
// the simulation loop reads a consistent snapshot for the whole frame, and
// edge-triggered events ("pressed this frame") cannot be missed or seen twice.
#pragma once

#include <string>

#include "core/Math.h"

struct GLFWwindow;

namespace aibf {

struct Input {
    static constexpr int kMaxKeys = 512;
    static constexpr int kMaxButtons = 8;

    bool keyDown[kMaxKeys] = {};
    bool keyPressed[kMaxKeys] = {};   // went down during the last poll
    bool keyReleased[kMaxKeys] = {};

    Vec2 mouseScreen;                 // pixels, origin at the top left
    Vec2 mouseDelta;
    bool mouseDown[kMaxButtons] = {};
    bool mousePressed[kMaxButtons] = {};
    bool mouseReleased[kMaxButtons] = {};
    Real scroll = 0;

    bool shift = false;
    bool ctrl = false;
    bool alt = false;

    void clearEdges();
};

class Window {
public:
    ~Window();

    bool create(int width, int height, const std::string& title, int samples = 4);
    void destroy();

    bool isOpen() const;
    void requestClose();

    void pollEvents();
    void present();

    void setTitle(const std::string& title);

    int width() const { return width_; }
    int height() const { return height_; }
    Real aspect() const { return height_ > 0 ? Real(width_) / Real(height_) : Real(1); }

    const Input& input() const { return input_; }

    // Reads the front buffer back and writes a PNG. Used to verify rendered
    // output rather than assert it.
    bool saveScreenshot(const std::string& path) const;

    GLFWwindow* handle() const { return window_; }

private:
    static void keyCallback(GLFWwindow*, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow*, int button, int action, int mods);
    static void cursorCallback(GLFWwindow*, double x, double y);
    static void scrollCallback(GLFWwindow*, double dx, double dy);
    static void resizeCallback(GLFWwindow*, int width, int height);

    GLFWwindow* window_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    Input input_;
    bool hasCursorSample_ = false;
};

}  // namespace aibf
