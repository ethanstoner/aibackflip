// Cross-language protocol check.
//
// Round-trip tests inside one language prove the codec is self-consistent, not
// that the two implementations agree. This tool closes that gap in both
// directions:
//
//   aibf_fixture write <dir>    C++ writes SPEC and STATE packets with known
//                               contents; the Python suite decodes and asserts.
//   aibf_fixture check <file>   C++ decodes an ACTION packet Python wrote and
//                               verifies the same known contents.
//
// The values below are deliberately awkward - negative zeros, denormals, values
// whose bytes differ under either endianness - so a layout or byte-order
// disagreement cannot pass by coincidence.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "net/Protocol.h"

using namespace aibf::net;

namespace {

bool writeFile(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    return file.good();
}

bool readFile(const std::string& path, std::vector<uint8_t>& bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const std::streamsize size = file.tellg();
    file.seekg(0);
    bytes.resize(static_cast<size_t>(size));
    return static_cast<bool>(
        file.read(reinterpret_cast<char*>(bytes.data()), size));
}

// Kept identical to python/tests/test_protocol_fixtures.py.
SpecMessage referenceSpec() {
    SpecMessage spec;
    spec.numEnvs = 3;
    spec.obsDim = 5;
    spec.actionDim = 2;
    spec.rewardDim = 2;
    spec.controlHz = 60.0f;
    spec.physicsHz = 240.0f;
    spec.maxEpisodeSteps = 1000;
    spec.actionLower = {-2.5f, -0.125f};
    spec.actionUpper = {0.75f, 3.0f};
    spec.actionNames = {"knee_l", "hip_l"};
    spec.rewardNames = {"alive", "torque_cost"};
    spec.observationNames = {"a", "b", "c", "d", "e"};
    return spec;
}

StateMessage referenceState() {
    StateMessage state;
    state.numEnvs = 3;
    state.obsDim = 5;
    state.rewardDim = 2;
    state.observations = {
        0.0f,   -0.0f,  0.5f,   -0.25f,  1.0f / 3.0f,
        1e-30f, -1e30f, 12345.6789f, -0.1f, 2.0f,
        3.0f,   4.0f,   5.0f,   6.0f,    7.0f,
    };
    state.rewardTerms = {1.0f, 0.25f, 0.0f, 0.5f, 1.0f, 0.125f};
    state.terminated = {0, 1, 0};
    state.truncated = {0, 0, 1};
    state.episodeStep = {7, 250, 4294967295u};
    state.finalMask = {0, 1, 1};
    state.finalObservations = {
        -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
        -6.0f, -7.0f, -8.0f, -9.0f, -10.0f,
    };
    return state;
}

int writeFixtures(const std::string& directory) {
    std::vector<uint8_t> bytes;

    Header specHeader;
    specHeader.type = MessageType::Spec;
    specHeader.session = 0xABCD1234u;
    specHeader.step = 0;
    encodeSpec(specHeader, referenceSpec(), bytes);
    if (!writeFile(directory + "/spec.bin", bytes)) {
        std::fprintf(stderr, "could not write %s/spec.bin\n", directory.c_str());
        return 1;
    }
    std::printf("wrote %s/spec.bin (%zu bytes)\n", directory.c_str(), bytes.size());

    Header stateHeader;
    stateHeader.type = MessageType::State;
    stateHeader.session = 0xABCD1234u;
    stateHeader.step = 42;
    encodeState(stateHeader, referenceState(), bytes);
    if (!writeFile(directory + "/state.bin", bytes)) {
        std::fprintf(stderr, "could not write %s/state.bin\n", directory.c_str());
        return 1;
    }
    std::printf("wrote %s/state.bin (%zu bytes)\n", directory.c_str(), bytes.size());
    return 0;
}

int checkAction(const std::string& path) {
    std::vector<uint8_t> bytes;
    if (!readFile(path, bytes)) {
        std::fprintf(stderr, "could not read %s\n", path.c_str());
        return 1;
    }

    Header header;
    if (!decodeHeader(bytes.data(), bytes.size(), header)) {
        std::fprintf(stderr, "FAIL: header did not decode\n");
        return 1;
    }
    if (header.type != MessageType::Action) {
        std::fprintf(stderr, "FAIL: expected ACTION, got %s\n", messageTypeName(header.type));
        return 1;
    }
    if (header.session != 0x0BADF00Du || header.step != 99) {
        std::fprintf(stderr, "FAIL: header session/step were %u/%u\n", header.session,
                     header.step);
        return 1;
    }

    ActionMessage action;
    if (!decodeAction(bytes.data(), bytes.size(), action)) {
        std::fprintf(stderr, "FAIL: action payload did not decode\n");
        return 1;
    }

    const std::vector<uint8_t> expectedMask = {1, 0, 1};
    const std::vector<float> expectedActions = {-1.0f, 0.0f, 1.0f, 0.5f, -0.5f, 0.25f};
    if (action.numEnvs != 3 || action.actionDim != 2) {
        std::fprintf(stderr, "FAIL: shape was %ux%u\n", action.numEnvs, action.actionDim);
        return 1;
    }
    if (action.resetMask != expectedMask) {
        std::fprintf(stderr, "FAIL: reset mask mismatch\n");
        return 1;
    }
    for (size_t i = 0; i < expectedActions.size(); ++i) {
        if (std::memcmp(&action.actions[i], &expectedActions[i], sizeof(float)) != 0) {
            std::fprintf(stderr, "FAIL: action[%zu] was %.9g, expected %.9g\n", i,
                         double(action.actions[i]), double(expectedActions[i]));
            return 1;
        }
    }

    std::printf("OK: the Python-encoded ACTION decoded exactly\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "write") == 0) return writeFixtures(argv[2]);
    if (argc >= 3 && std::strcmp(argv[1], "check") == 0) return checkAction(argv[2]);
    std::printf("usage:\n  aibf_fixture write <directory>\n  aibf_fixture check <action.bin>\n");
    return 2;
}
