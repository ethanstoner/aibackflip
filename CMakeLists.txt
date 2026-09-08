cmake_minimum_required(VERSION 3.20)
project(aibackflip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------- warnings
# No fast-math anywhere: NaN and infinity checks in the solver are load-bearing,
# and /fp:fast lets the compiler assume they cannot happen.
add_library(aibf_flags INTERFACE)
if(MSVC)
  target_compile_options(aibf_flags INTERFACE
    /W4 /permissive- /Zc:__cplusplus /utf-8 /MP /fp:precise)
  target_compile_definitions(aibf_flags INTERFACE
    _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
  target_compile_options(aibf_flags INTERFACE
    -Wall -Wextra -Wno-unused-parameter -ffp-contract=off)
endif()

# ---------------------------------------------------------------- deps
include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

find_package(OpenGL REQUIRED)

# ---------------------------------------------------------------- core
add_library(aibf_core STATIC
  cpp/core/Json.cpp
  cpp/core/Png.cpp
  cpp/core/Test.cpp
)
target_include_directories(aibf_core PUBLIC ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(aibf_core PUBLIC aibf_flags)

# ---------------------------------------------------------------- physics
add_library(aibf_physics STATIC
  cpp/physics/Collision2D.cpp
  cpp/physics/Joint2D.cpp
  cpp/physics/World2D.cpp
)
target_link_libraries(aibf_physics PUBLIC aibf_core)

# ---------------------------------------------------------------- humanoid
add_library(aibf_humanoid STATIC
  cpp/humanoid/HumanoidConfig.cpp
  cpp/humanoid/Humanoid2D.cpp
)
target_link_libraries(aibf_humanoid PUBLIC aibf_physics)

# ---------------------------------------------------------------- executables
# ---------------------------------------------------------------- rendering
# Kept separate from the simulation libraries so headless training never links
# against GLFW or OpenGL at all.
add_library(aibf_engine STATIC
  cpp/engine/GL.cpp
  cpp/engine/Window.cpp
  cpp/engine/Renderer2D.cpp
  cpp/engine/DebugDraw.cpp
)
target_link_libraries(aibf_engine PUBLIC aibf_humanoid glfw OpenGL::GL)

# ---------------------------------------------------------------- executables
add_executable(aibackflip cpp/main.cpp)
target_link_libraries(aibackflip PRIVATE aibf_engine)

# Reports measurements rather than assertions; links no rendering.
add_executable(aibf_diag cpp/tools/diagnostics.cpp)
target_link_libraries(aibf_diag PRIVATE aibf_humanoid)

add_executable(aibf_tests
  cpp/tests_main.cpp
  cpp/tests/test_math.cpp
  cpp/tests/test_json.cpp
  cpp/tests/test_collision.cpp
  cpp/tests/test_physics.cpp
  cpp/tests/test_joints.cpp
  cpp/tests/test_humanoid.cpp
  cpp/tests/test_motors.cpp
)
target_link_libraries(aibf_tests PRIVATE aibf_humanoid)

enable_testing()
add_test(NAME cpp_unit COMMAND aibf_tests)
