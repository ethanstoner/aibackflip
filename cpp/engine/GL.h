// Minimal OpenGL 3.3 core loader.
//
// Only the entry points the renderers actually use are declared. GL 1.1
// functions (glClear, glDrawArrays, glEnable, ...) are exported directly by
// opengl32 and are called through <GL/gl.h> as usual; everything from GL 1.5
// onwards has to be fetched at runtime, which is what this file does.
//
// A generated loader would work too, but it would add a code generator to the
// build to obtain a list this file can simply state.
#pragma once

#include <cstddef>
#include <string>

#include <GLFW/glfw3.h>  // pulls in <GL/gl.h> for the 1.1 types and functions

namespace aibf::gl {

using GLchar = char;
using GLsizeiptr = std::ptrdiff_t;
using GLintptr = std::ptrdiff_t;

// Constants absent from the 1.1 headers Windows ships.
constexpr GLenum kArrayBuffer = 0x8892;
constexpr GLenum kElementArrayBuffer = 0x8893;
constexpr GLenum kStreamDraw = 0x88E0;
constexpr GLenum kStaticDraw = 0x88E4;
constexpr GLenum kDynamicDraw = 0x88E8;
constexpr GLenum kFragmentShader = 0x8B30;
constexpr GLenum kVertexShader = 0x8B31;
constexpr GLenum kCompileStatus = 0x8B81;
constexpr GLenum kLinkStatus = 0x8B82;
constexpr GLenum kInfoLogLength = 0x8B84;
constexpr GLenum kMultisample = 0x809D;

using GenVertexArraysProc = void(APIENTRY*)(GLsizei, GLuint*);
using DeleteVertexArraysProc = void(APIENTRY*)(GLsizei, const GLuint*);
using BindVertexArrayProc = void(APIENTRY*)(GLuint);
using GenBuffersProc = void(APIENTRY*)(GLsizei, GLuint*);
using DeleteBuffersProc = void(APIENTRY*)(GLsizei, const GLuint*);
using BindBufferProc = void(APIENTRY*)(GLenum, GLuint);
using BufferDataProc = void(APIENTRY*)(GLenum, GLsizeiptr, const void*, GLenum);
using BufferSubDataProc = void(APIENTRY*)(GLenum, GLintptr, GLsizeiptr, const void*);
using EnableVertexAttribArrayProc = void(APIENTRY*)(GLuint);
using VertexAttribPointerProc = void(APIENTRY*)(GLuint, GLint, GLenum, GLboolean, GLsizei,
                                                const void*);
using CreateShaderProc = GLuint(APIENTRY*)(GLenum);
using ShaderSourceProc = void(APIENTRY*)(GLuint, GLsizei, const GLchar* const*, const GLint*);
using CompileShaderProc = void(APIENTRY*)(GLuint);
using GetShaderivProc = void(APIENTRY*)(GLuint, GLenum, GLint*);
using GetShaderInfoLogProc = void(APIENTRY*)(GLuint, GLsizei, GLsizei*, GLchar*);
using DeleteShaderProc = void(APIENTRY*)(GLuint);
using CreateProgramProc = GLuint(APIENTRY*)();
using AttachShaderProc = void(APIENTRY*)(GLuint, GLuint);
using LinkProgramProc = void(APIENTRY*)(GLuint);
using GetProgramivProc = void(APIENTRY*)(GLuint, GLenum, GLint*);
using GetProgramInfoLogProc = void(APIENTRY*)(GLuint, GLsizei, GLsizei*, GLchar*);
using UseProgramProc = void(APIENTRY*)(GLuint);
using DeleteProgramProc = void(APIENTRY*)(GLuint);
using GetUniformLocationProc = GLint(APIENTRY*)(GLuint, const GLchar*);
using UniformMatrix4fvProc = void(APIENTRY*)(GLint, GLsizei, GLboolean, const GLfloat*);
using UniformMatrix3fvProc = void(APIENTRY*)(GLint, GLsizei, GLboolean, const GLfloat*);
using Uniform4fProc = void(APIENTRY*)(GLint, GLfloat, GLfloat, GLfloat, GLfloat);
using Uniform3fProc = void(APIENTRY*)(GLint, GLfloat, GLfloat, GLfloat);
using Uniform1fProc = void(APIENTRY*)(GLint, GLfloat);
using Uniform1iProc = void(APIENTRY*)(GLint, GLint);

extern GenVertexArraysProc GenVertexArrays;
extern DeleteVertexArraysProc DeleteVertexArrays;
extern BindVertexArrayProc BindVertexArray;
extern GenBuffersProc GenBuffers;
extern DeleteBuffersProc DeleteBuffers;
extern BindBufferProc BindBuffer;
extern BufferDataProc BufferData;
extern BufferSubDataProc BufferSubData;
extern EnableVertexAttribArrayProc EnableVertexAttribArray;
extern VertexAttribPointerProc VertexAttribPointer;
extern CreateShaderProc CreateShader;
extern ShaderSourceProc ShaderSource;
extern CompileShaderProc CompileShader;
extern GetShaderivProc GetShaderiv;
extern GetShaderInfoLogProc GetShaderInfoLog;
extern DeleteShaderProc DeleteShader;
extern CreateProgramProc CreateProgram;
extern AttachShaderProc AttachShader;
extern LinkProgramProc LinkProgram;
extern GetProgramivProc GetProgramiv;
extern GetProgramInfoLogProc GetProgramInfoLog;
extern UseProgramProc UseProgram;
extern DeleteProgramProc DeleteProgram;
extern GetUniformLocationProc GetUniformLocation;
extern UniformMatrix4fvProc UniformMatrix4fv;
extern UniformMatrix3fvProc UniformMatrix3fv;
extern Uniform4fProc Uniform4f;
extern Uniform3fProc Uniform3f;
extern Uniform1fProc Uniform1f;
extern Uniform1iProc Uniform1i;

// Must be called with a current context. Returns false and fills `missing` with
// the name of the first entry point that could not be resolved.
bool load(std::string* missing = nullptr);

}  // namespace aibf::gl
