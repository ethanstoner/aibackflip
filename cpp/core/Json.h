// Small JSON reader/writer.
//
// Configs and reference motions both live in JSON, and pulling in a dependency
// for a format this small is not worth the build surface. Reads are total: a
// missing key yields a null value rather than throwing, so config loading can
// be written as a flat list of "read this, fall back to that" lines.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "core/Math.h"

namespace aibf {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() = default;
    explicit Json(bool v) : type_(Type::Bool), bool_(v) {}
    explicit Json(double v) : type_(Type::Number), number_(v) {}
    explicit Json(int v) : type_(Type::Number), number_(v) {}
    explicit Json(const char* v) : type_(Type::String), string_(v) {}
    explicit Json(std::string v) : type_(Type::String), string_(std::move(v)) {}

    static Json array() {
        Json j;
        j.type_ = Type::Array;
        return j;
    }
    static Json object() {
        Json j;
        j.type_ = Type::Object;
        return j;
    }

    // On failure returns a null Json and fills `error` with a message including
    // the byte offset.
    static Json parse(const std::string& text, std::string* error = nullptr);
    static Json parseFile(const std::string& path, std::string* error = nullptr);

    std::string dump(int indent = 2) const;
    bool writeFile(const std::string& path, int indent = 2) const;

    Type type() const { return type_; }
    bool isNull() const { return type_ == Type::Null; }
    bool isBool() const { return type_ == Type::Bool; }
    bool isNumber() const { return type_ == Type::Number; }
    bool isString() const { return type_ == Type::String; }
    bool isArray() const { return type_ == Type::Array; }
    bool isObject() const { return type_ == Type::Object; }

    // Reads with fallbacks. Nothing here throws or asserts; a config that is
    // missing a field takes the default and carries on.
    bool boolean(bool fallback = false) const { return isBool() ? bool_ : fallback; }
    double number(double fallback = 0) const { return isNumber() ? number_ : fallback; }
    Real real(Real fallback = 0) const {
        return isNumber() ? static_cast<Real>(number_) : fallback;
    }
    int integer(int fallback = 0) const {
        return isNumber() ? static_cast<int>(number_) : fallback;
    }
    const std::string& string(const std::string& fallback) const {
        return isString() ? string_ : fallback;
    }
    std::string string() const { return isString() ? string_ : std::string(); }

    size_t size() const;
    bool contains(const std::string& key) const;
    std::vector<std::string> keys() const;

    // Missing keys and out-of-range indices return a shared null value, so
    // chains like cfg["humanoid"]["torso"]["mass"].real(8.0f) are safe.
    //
    // Both are const-only, and the index overload takes int rather than size_t.
    // Both details matter: a literal 0 is a null pointer constant, so it
    // converts to const char* and therefore to std::string. With an
    // auto-vivifying non-const operator[](std::string) in the overload set,
    // `value[0]` on a non-const Json ties - the string overload wins on the
    // object argument, the index overload wins on the index - and the call is
    // ambiguous. Writing goes through set() instead, which sidesteps it.
    const Json& operator[](const std::string& key) const;
    const Json& operator[](int index) const;

    // Inserts or overwrites, promoting this value to an object if needed.
    // Returns a reference to the stored child so builders can chain.
    Json& set(const std::string& key, Json value);
    void push(Json value);

    // Convenience for the common "array of numbers" shapes.
    std::vector<Real> realArray() const;
    Vec2 vec2(const Vec2& fallback = Vec2(0, 0)) const;
    Vec3 vec3(const Vec3& fallback = Vec3(0, 0, 0)) const;

private:
    void dumpTo(std::string& out, int indent, int depth) const;

    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0;
    std::string string_;
    std::vector<Json> array_;
    // Ordered so a written file keeps a stable, diffable key order.
    std::map<std::string, Json> object_;
};

}  // namespace aibf
