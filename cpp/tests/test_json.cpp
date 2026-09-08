#include "core/Json.h"
#include "core/Rng.h"
#include "core/Test.h"

using namespace aibf;

TEST(Json, parsesScalars) {
    CHECK(Json::parse("true").boolean() == true);
    CHECK(Json::parse("false").boolean(true) == false);
    CHECK(Json::parse("null").isNull());
    CHECK_NEAR(Json::parse("3.5").number(), 3.5, 1e-12);
    CHECK_NEAR(Json::parse("-2e3").number(), -2000.0, 1e-9);
    CHECK(Json::parse("\"hello\"").string() == "hello");
}

TEST(Json, parsesNestedStructures) {
    std::string error;
    Json j = Json::parse(R"({
        "name": "humanoid",
        "gravity": [0, -9.81],
        "joints": [
            {"name": "knee", "limits": [-2.4, 0.05]},
            {"name": "hip",  "limits": [-1.2, 1.6]}
        ],
        "enabled": true
    })", &error);

    CHECK(error.empty());
    CHECK(j.isObject());
    CHECK(j["name"].string() == "humanoid");
    CHECK_NEAR(j["gravity"].vec2().y, -9.81, 1e-5);
    CHECK(j["joints"].size() == 2);
    CHECK(j["joints"][1]["name"].string() == "hip");
    CHECK_NEAR(j["joints"][0]["limits"][0].number(), -2.4, 1e-6);
    CHECK(j["enabled"].boolean());
}

TEST(Json, missingKeysReturnTheFallbackInsteadOfThrowing) {
    // Config loading depends on this being total: an older config file missing a
    // newly-added field has to take the default and keep going.
    Json j = Json::parse(R"({"a": 1})");
    CHECK(j["nope"].isNull());
    CHECK_NEAR(j["nope"].real(Real(7)), 7.0, 1e-6);
    CHECK(j["nope"]["deeper"]["deeper still"].isNull());
    CHECK_NEAR(j["nope"]["deeper"][3].real(Real(-1)), -1.0, 1e-6);
    CHECK(j["a"][2].isNull());  // indexing a number, not an array
}

TEST(Json, wrongTypeReadsFallBackRatherThanReinterpret) {
    Json j = Json::parse(R"({"count": "twelve", "flag": 3})");
    CHECK_NEAR(j["count"].real(Real(-1)), -1.0, 1e-6);
    CHECK(j["flag"].boolean(true) == true);
    CHECK(j["count"].string() == "twelve");
}

TEST(Json, handlesEscapesAndUnicode) {
    Json j = Json::parse(R"({"s": "a\"b\\c\ndé"})");
    const std::string s = j["s"].string();
    CHECK(s.find('"') != std::string::npos);
    CHECK(s.find('\\') != std::string::npos);
    CHECK(s.find('\n') != std::string::npos);
    CHECK(s.size() > 7);  // the two-byte UTF-8 encoding of the accented char
}

TEST(Json, reportsErrorsWithAPosition) {
    std::string error;
    Json j = Json::parse(R"({"a": )", &error);
    CHECK(j.isNull());
    CHECK(!error.empty());
    CHECK(error.find("byte") != std::string::npos);

    error.clear();
    Json bad = Json::parse("{\"a\": 1} trailing", &error);
    CHECK(bad.isNull());
    CHECK(!error.empty());
}

TEST(Json, lineCommentsAreAccepted) {
    // Configs are read and edited by hand; a format that cannot hold a note
    // about why a gain is 4000 pushes that note somewhere it will rot.
    std::string error;
    Json j = Json::parse(R"({
        // knee gains, raised after the landing test
        "kp": 4000,
        "kd": 400  // critically damped at the shank's inertia
    })", &error);
    CHECK(error.empty());
    CHECK_NEAR(j["kp"].real(), 4000.0, 1e-6);
    CHECK_NEAR(j["kd"].real(), 400.0, 1e-6);
}

TEST(Json, roundTripsThroughDump) {
    const char* source = R"({
        "joints": [{"limits": [-2.4, 0.05], "kp": 300.5}],
        "name": "test", "flag": false, "nothing": null
    })";
    Json original = Json::parse(source);
    std::string error;
    Json reparsed = Json::parse(original.dump(), &error);

    CHECK(error.empty());
    CHECK(reparsed["name"].string() == "test");
    CHECK(reparsed["flag"].boolean(true) == false);
    CHECK(reparsed["nothing"].isNull());
    CHECK_NEAR(reparsed["joints"][0]["kp"].number(), 300.5, 1e-9);
    CHECK_NEAR(reparsed["joints"][0]["limits"][0].number(), -2.4, 1e-9);
    // Dumping twice must be byte-identical, or motion files churn in diffs.
    CHECK(reparsed.dump() == original.dump());
}

TEST(Json, buildsDocumentsProgrammatically) {
    Json doc = Json::object();
    doc.set("name", Json("backflip"));
    doc.set("fps", Json(30));
    Json frames = Json::array();
    for (int i = 0; i < 3; ++i) {
        Json frame = Json::object();
        frame.set("t", Json(i * 0.1));
        Json pose = Json::array();
        pose.push(Json(0.5));
        pose.push(Json(-0.25));
        frame.set("pose", std::move(pose));
        frames.push(std::move(frame));
    }
    doc.set("frames", std::move(frames));

    std::string error;
    Json reparsed = Json::parse(doc.dump(), &error);
    CHECK(error.empty());
    CHECK(reparsed["frames"].size() == 3);
    CHECK_NEAR(reparsed["frames"][2]["t"].number(), 0.2, 1e-9);
    CHECK_NEAR(reparsed["frames"][1]["pose"][1].number(), -0.25, 1e-9);
    CHECK(reparsed["fps"].integer() == 30);
}

TEST(Json, integersDumpWithoutADecimalTail) {
    Json doc = Json::object();
    doc.set("frames", Json(120));
    CHECK(doc.dump(0) == "{\"frames\":120}");
}

TEST(Json, floatValuesDumpWithoutPrecisionNoise) {
    // Configs are written from 32-bit Reals. Widening -0.8f to double and
    // printing it at double precision gives "-0.800000012", which is correct and
    // unreadable; the writer has to emit the shortest form that reads back.
    Json doc = Json::object();
    doc.set("lower", Json(double(-0.8f)));
    doc.set("anchor", Json(double(1.08f)));
    doc.set("kp", Json(double(4000.0f)));
    CHECK(doc.dump(0) == "{\"anchor\":1.08,\"kp\":4000,\"lower\":-0.8}");
}

TEST(Json, shorteningNeverChangesTheValue) {
    // The shortening must be exact, not approximate: whatever is written has to
    // parse back to the same number.
    Rng rng(2);
    for (int i = 0; i < 2000; ++i) {
        const double original = double(Real(rng.uniform(-1000, 1000)));
        Json doc = Json::object();
        doc.set("v", Json(original));
        const double restored = Json::parse(doc.dump(0))["v"].number();
        CHECK(Real(restored) == Real(original));
    }
    // Genuine doubles keep full precision rather than being rounded to float.
    const double precise = 0.1234567890123456;
    Json doc = Json::object();
    doc.set("v", Json(precise));
    CHECK_NEAR(Json::parse(doc.dump(0))["v"].number(), precise, 1e-18);
}

TEST(Json, emptyContainersRoundTrip) {
    Json doc = Json::parse(R"({"a": [], "b": {}})");
    CHECK(doc["a"].isArray());
    CHECK(doc["a"].size() == 0);
    CHECK(doc["b"].isObject());
    CHECK(Json::parse(doc.dump())["b"].isObject());
}

TEST(Json, realArrayExtractsNumbers) {
    std::vector<Real> values = Json::parse("[1, 2.5, -3]").realArray();
    CHECK(values.size() == 3);
    CHECK_NEAR(values[1], 2.5, 1e-6);
    CHECK(Json::parse("42").realArray().empty());
}
