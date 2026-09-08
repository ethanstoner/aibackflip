#include "engine/GL.h"

namespace aibf::gl {

GenVertexArraysProc GenVertexArrays = nullptr;
DeleteVertexArraysProc DeleteVertexArrays = nullptr;
BindVertexArrayProc BindVertexArray = nullptr;
GenBuffersProc GenBuffers = nullptr;
DeleteBuffersProc DeleteBuffers = nullptr;
BindBufferProc BindBuffer = nullptr;
BufferDataProc BufferData = nullptr;
BufferSubDataProc BufferSubData = nullptr;
EnableVertexAttribArrayProc EnableVertexAttribArray = nullptr;
VertexAttribPointerProc VertexAttribPointer = nullptr;
CreateShaderProc CreateShader = nullptr;
ShaderSourceProc ShaderSource = nullptr;
CompileShaderProc CompileShader = nullptr;
GetShaderivProc GetShaderiv = nullptr;
GetShaderInfoLogProc GetShaderInfoLog = nullptr;
DeleteShaderProc DeleteShader = nullptr;
CreateProgramProc CreateProgram = nullptr;
AttachShaderProc AttachShader = nullptr;
LinkProgramProc LinkProgram = nullptr;
GetProgramivProc GetProgramiv = nullptr;
GetProgramInfoLogProc GetProgramInfoLog = nullptr;
UseProgramProc UseProgram = nullptr;
DeleteProgramProc DeleteProgram = nullptr;
GetUniformLocationProc GetUniformLocation = nullptr;
UniformMatrix4fvProc UniformMatrix4fv = nullptr;
Uniform4fProc Uniform4f = nullptr;
Uniform3fProc Uniform3f = nullptr;
Uniform1fProc Uniform1f = nullptr;
Uniform1iProc Uniform1i = nullptr;

namespace {

bool resolve(void*& slot, const char* name, std::string* missing) {
    void* address = reinterpret_cast<void*>(glfwGetProcAddress(name));
    if (!address) {
        if (missing && missing->empty()) *missing = name;
        return false;
    }
    slot = address;
    return true;
}

}  // namespace

bool load(std::string* missing) {
    if (missing) missing->clear();
    bool ok = true;

// Each entry point is resolved through the same path so a driver that is
// missing one of them names it rather than crashing on the first call.
#define AIBF_LOAD(fn, name) \
    ok &= resolve(reinterpret_cast<void*&>(fn), name, missing)

    AIBF_LOAD(GenVertexArrays, "glGenVertexArrays");
    AIBF_LOAD(DeleteVertexArrays, "glDeleteVertexArrays");
    AIBF_LOAD(BindVertexArray, "glBindVertexArray");
    AIBF_LOAD(GenBuffers, "glGenBuffers");
    AIBF_LOAD(DeleteBuffers, "glDeleteBuffers");
    AIBF_LOAD(BindBuffer, "glBindBuffer");
    AIBF_LOAD(BufferData, "glBufferData");
    AIBF_LOAD(BufferSubData, "glBufferSubData");
    AIBF_LOAD(EnableVertexAttribArray, "glEnableVertexAttribArray");
    AIBF_LOAD(VertexAttribPointer, "glVertexAttribPointer");
    AIBF_LOAD(CreateShader, "glCreateShader");
    AIBF_LOAD(ShaderSource, "glShaderSource");
    AIBF_LOAD(CompileShader, "glCompileShader");
    AIBF_LOAD(GetShaderiv, "glGetShaderiv");
    AIBF_LOAD(GetShaderInfoLog, "glGetShaderInfoLog");
    AIBF_LOAD(DeleteShader, "glDeleteShader");
    AIBF_LOAD(CreateProgram, "glCreateProgram");
    AIBF_LOAD(AttachShader, "glAttachShader");
    AIBF_LOAD(LinkProgram, "glLinkProgram");
    AIBF_LOAD(GetProgramiv, "glGetProgramiv");
    AIBF_LOAD(GetProgramInfoLog, "glGetProgramInfoLog");
    AIBF_LOAD(UseProgram, "glUseProgram");
    AIBF_LOAD(DeleteProgram, "glDeleteProgram");
    AIBF_LOAD(GetUniformLocation, "glGetUniformLocation");
    AIBF_LOAD(UniformMatrix4fv, "glUniformMatrix4fv");
    AIBF_LOAD(Uniform4f, "glUniform4f");
    AIBF_LOAD(Uniform3f, "glUniform3f");
    AIBF_LOAD(Uniform1f, "glUniform1f");
    AIBF_LOAD(Uniform1i, "glUniform1i");

#undef AIBF_LOAD
    return ok;
}

}  // namespace aibf::gl
