#include "core/Json.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace aibf {

namespace {

const Json& nullValue() {
    static const Json kNull;
    return kNull;
}

class Parser {
public:
    Parser(const std::string& text) : text_(text) {}

    bool parse(Json& out) {
        skipWhitespace();
        if (!parseValue(out)) return false;
        skipWhitespace();
        if (pos_ != text_.size()) {
            setError("trailing characters after the top-level value");
            return false;
        }
        return true;
    }

    const std::string& error() const { return error_; }

private:
    void setError(const std::string& message) {
        if (error_.empty()) {
            error_ = "at byte " + std::to_string(pos_) + ": " + message;
        }
    }

    bool atEnd() const { return pos_ >= text_.size(); }
    char peek() const { return text_[pos_]; }

    void skipWhitespace() {
        while (!atEnd()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else if (c == '/' && pos_ + 1 < text_.size() && text_[pos_ + 1] == '/') {
                // Line comments are not standard JSON, but configs are read by
                // people and an un-commentable config invites a second file.
                while (!atEnd() && text_[pos_] != '\n') ++pos_;
            } else {
                break;
            }
        }
    }

    bool expect(char c) {
        if (atEnd() || text_[pos_] != c) {
            setError(std::string("expected '") + c + "'");
            return false;
        }
        ++pos_;
        return true;
    }

    bool literal(const char* word, Json value, Json& out) {
        const size_t len = std::char_traits<char>::length(word);
        if (text_.compare(pos_, len, word) != 0) return false;
        pos_ += len;
        out = std::move(value);
        return true;
    }

    bool parseValue(Json& out) {
        if (atEnd()) {
            setError("unexpected end of input");
            return false;
        }
        switch (peek()) {
            case '{': return parseObject(out);
            case '[': return parseArray(out);
            case '"': {
                std::string s;
                if (!parseString(s)) return false;
                out = Json(std::move(s));
                return true;
            }
            case 't':
                if (literal("true", Json(true), out)) return true;
                setError("invalid literal");
                return false;
            case 'f':
                if (literal("false", Json(false), out)) return true;
                setError("invalid literal");
                return false;
            case 'n':
                if (literal("null", Json(), out)) return true;
                setError("invalid literal");
                return false;
            default: return parseNumber(out);
        }
    }

    bool parseObject(Json& out) {
        if (!expect('{')) return false;
        out = Json::object();
        skipWhitespace();
        if (!atEnd() && peek() == '}') {
            ++pos_;
            return true;
        }
        for (;;) {
            skipWhitespace();
            std::string key;
            if (!parseString(key)) return false;
            skipWhitespace();
            if (!expect(':')) return false;
            skipWhitespace();
            Json value;
            if (!parseValue(value)) return false;
            out.set(key, std::move(value));
            skipWhitespace();
            if (atEnd()) {
                setError("unterminated object");
                return false;
            }
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            return expect('}');
        }
    }

    bool parseArray(Json& out) {
        if (!expect('[')) return false;
        out = Json::array();
        skipWhitespace();
        if (!atEnd() && peek() == ']') {
            ++pos_;
            return true;
        }
        for (;;) {
            skipWhitespace();
            Json value;
            if (!parseValue(value)) return false;
            out.push(std::move(value));
            skipWhitespace();
            if (atEnd()) {
                setError("unterminated array");
                return false;
            }
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            return expect(']');
        }
    }

    bool parseString(std::string& out) {
        if (!expect('"')) return false;
        out.clear();
        while (!atEnd()) {
            const char c = text_[pos_++];
            if (c == '"') return true;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (atEnd()) break;
            const char esc = text_[pos_++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    if (pos_ + 4 > text_.size()) {
                        setError("truncated \\u escape");
                        return false;
                    }
                    unsigned code = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char h = text_[pos_ + static_cast<size_t>(i)];
                        code <<= 4;
                        if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
                        else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
                        else {
                            setError("bad hex digit in \\u escape");
                            return false;
                        }
                    }
                    pos_ += 4;
                    appendUtf8(out, code);
                    break;
                }
                default:
                    setError("unknown escape sequence");
                    return false;
            }
        }
        setError("unterminated string");
        return false;
    }

    static void appendUtf8(std::string& out, unsigned code) {
        if (code < 0x80) {
            out.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }

    bool parseNumber(Json& out) {
        const size_t start = pos_;
        if (!atEnd() && (peek() == '-' || peek() == '+')) ++pos_;
        bool anyDigits = false;
        while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek()))) {
            ++pos_;
            anyDigits = true;
        }
        if (!atEnd() && peek() == '.') {
            ++pos_;
            while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
                anyDigits = true;
            }
        }
        if (anyDigits && !atEnd() && (peek() == 'e' || peek() == 'E')) {
            ++pos_;
            if (!atEnd() && (peek() == '-' || peek() == '+')) ++pos_;
            while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        }
        if (!anyDigits) {
            setError("expected a value");
            return false;
        }
        out = Json(std::strtod(text_.substr(start, pos_ - start).c_str(), nullptr));
        return true;
    }

    const std::string& text_;
    size_t pos_ = 0;
    std::string error_;
};

void escapeInto(std::string& out, const std::string& s) {
    out.push_back('"');
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

// Emits the shortest representation that reads back as the same value, and no
// trailing ".0" on integers.
//
// Without this, a config written from 32-bit Reals comes back as
// "-0.800000012" - correct, but noise in a file that people read and diff. The
// shortest form is found by trying increasing precision until the text parses
// back to the original value, at float precision when the value came from a
// float and at double precision otherwise.
void numberInto(std::string& out, double v) {
    if (!std::isfinite(v)) {
        out += "0";
        return;
    }
    if (v == static_cast<double>(static_cast<long long>(v)) && std::abs(v) < 1e15) {
        out += std::to_string(static_cast<long long>(v));
        return;
    }

    const bool floatExact = static_cast<double>(static_cast<float>(v)) == v;
    char buf[40];
    for (int precision = 6; precision <= 17; ++precision) {
        std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
        const double parsed = std::strtod(buf, nullptr);
        if (floatExact ? (static_cast<float>(parsed) == static_cast<float>(v)) : (parsed == v)) {
            out += buf;
            return;
        }
    }
    out += buf;
}

}  // namespace

Json Json::parse(const std::string& text, std::string* error) {
    Parser parser(text);
    Json result;
    if (!parser.parse(result)) {
        if (error) *error = parser.error();
        return Json();
    }
    if (error) error->clear();
    return result;
}

Json Json::parseFile(const std::string& path, std::string* error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (error) *error = "could not open " + path;
        return Json();
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    Json result = parse(text, error);
    if (error && !error->empty()) *error = path + " " + *error;
    return result;
}

size_t Json::size() const {
    if (isArray()) return array_.size();
    if (isObject()) return object_.size();
    return 0;
}

bool Json::contains(const std::string& key) const {
    return isObject() && object_.find(key) != object_.end();
}

std::vector<std::string> Json::keys() const {
    std::vector<std::string> result;
    if (isObject()) {
        result.reserve(object_.size());
        for (const auto& entry : object_) result.push_back(entry.first);
    }
    return result;
}

const Json& Json::operator[](const std::string& key) const {
    if (!isObject()) return nullValue();
    const auto it = object_.find(key);
    return it == object_.end() ? nullValue() : it->second;
}

const Json& Json::operator[](int index) const {
    if (!isArray() || index < 0 || static_cast<size_t>(index) >= array_.size()) return nullValue();
    return array_[static_cast<size_t>(index)];
}

Json& Json::set(const std::string& key, Json value) {
    if (!isObject()) {
        type_ = Type::Object;
        array_.clear();
    }
    Json& slot = object_[key];
    slot = std::move(value);
    return slot;
}

void Json::push(Json value) {
    if (!isArray()) {
        type_ = Type::Array;
        object_.clear();
    }
    array_.push_back(std::move(value));
}

std::vector<Real> Json::realArray() const {
    std::vector<Real> result;
    if (!isArray()) return result;
    result.reserve(array_.size());
    for (const Json& v : array_) result.push_back(v.real(0));
    return result;
}

Vec2 Json::vec2(const Vec2& fallback) const {
    if (!isArray() || array_.size() < 2) return fallback;
    return Vec2(array_[0].real(fallback.x), array_[1].real(fallback.y));
}

Vec3 Json::vec3(const Vec3& fallback) const {
    if (!isArray() || array_.size() < 3) return fallback;
    return Vec3(array_[0].real(fallback.x), array_[1].real(fallback.y), array_[2].real(fallback.z));
}

void Json::dumpTo(std::string& out, int indent, int depth) const {
    const bool pretty = indent > 0;
    const std::string pad = pretty ? std::string(static_cast<size_t>(indent * (depth + 1)), ' ') : "";
    const std::string padEnd = pretty ? std::string(static_cast<size_t>(indent * depth), ' ') : "";
    const char* newline = pretty ? "\n" : "";

    switch (type_) {
        case Type::Null: out += "null"; break;
        case Type::Bool: out += bool_ ? "true" : "false"; break;
        case Type::Number: numberInto(out, number_); break;
        case Type::String: escapeInto(out, string_); break;
        case Type::Array: {
            if (array_.empty()) {
                out += "[]";
                break;
            }
            // Arrays of plain numbers stay on one line. A motion file with one
            // joint angle per line is unreadable and enormous.
            bool allScalar = true;
            for (const Json& v : array_) {
                if (v.isArray() || v.isObject()) {
                    allScalar = false;
                    break;
                }
            }
            if (allScalar) {
                out += "[";
                for (size_t i = 0; i < array_.size(); ++i) {
                    if (i) out += ", ";
                    array_[i].dumpTo(out, 0, 0);
                }
                out += "]";
                break;
            }
            out += "[";
            out += newline;
            for (size_t i = 0; i < array_.size(); ++i) {
                out += pad;
                array_[i].dumpTo(out, indent, depth + 1);
                if (i + 1 < array_.size()) out += ",";
                out += newline;
            }
            out += padEnd;
            out += "]";
            break;
        }
        case Type::Object: {
            if (object_.empty()) {
                out += "{}";
                break;
            }
            out += "{";
            out += newline;
            size_t i = 0;
            for (const auto& entry : object_) {
                out += pad;
                escapeInto(out, entry.first);
                out += pretty ? ": " : ":";
                entry.second.dumpTo(out, indent, depth + 1);
                if (++i < object_.size()) out += ",";
                out += newline;
            }
            out += padEnd;
            out += "}";
            break;
        }
    }
}

std::string Json::dump(int indent) const {
    std::string out;
    dumpTo(out, indent, 0);
    return out;
}

bool Json::writeFile(const std::string& path, int indent) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file << dump(indent) << "\n";
    return file.good();
}

}  // namespace aibf
