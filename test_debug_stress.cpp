// test_debug_stress.cpp - stress-test debug mode with benchmark model
#include <stdio.h>
#include <stdlib.h>
#include <MNN/Interpreter.hpp>
#include <MNN/ErrorCode.hpp>

int main(int argc, char* argv[]) {
    const char* modelPath = nullptr;
    int repeat = 3;
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) repeat = atoi(argv[2]);
    if (!modelPath) {
        fprintf(stderr, "Usage: %s model.mnn [repeat]\n", argv[0]);
        return 1;
    }

    // Load model
    fprintf(stderr, "Loading model: %s\n", modelPath);
    auto* interpreter = MNN::Interpreter::createFromFile(modelPath);
    if (!interpreter) {
        fprintf(stderr, "FAIL: createFromFile\n");
        return 1;
    }

    // Create session with Session_Debug mode for callbacks
    interpreter->setSessionMode(MNN::Interpreter::Session_Debug);

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 2;

    auto* session = interpreter->createSession(config);
    if (!session) {
        fprintf(stderr, "FAIL: createSession\n");
        return 1;
    }
    fprintf(stderr, "Session created\n");

    // Get input tensor
    auto* input = interpreter->getSessionInput(session, nullptr);
    if (!input) {
        fprintf(stderr, "FAIL: getSessionInput\n");
        return 1;
    }

    // Print input shape
    fprintf(stderr, "Input shape: [");
    for (int i = 0; i < input->dimensions(); i++) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%d", input->length(i));
    }
    fprintf(stderr, "]\n");

    // Fill input with data
    if (input->host<void>() && input->size() > 0) {
        float* data = (float*)input->host<void>();
        int count = input->size() / sizeof(float);
        for (int i = 0; i < count; i++) data[i] = 0.5f;
    }

    // Test 1: Aggregate (runSession)
    fprintf(stderr, "\n--- Test 1: runSession (aggregate) ---\n");
    for (int r = 0; r < repeat; r++) {
        auto err = interpreter->runSession(session);
        if (err != MNN::NO_ERROR) {
            fprintf(stderr, "FAIL: runSession attempt %d returned error %d\n", r, err);
            break;
        }
    }
    fprintf(stderr, "OK: runSession completed %dx\n", repeat);

    // Get aggregate metrics
    float memory = 0, flops = 0;
    interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &memory);
    interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
    fprintf(stderr, "  MEMORY=%f MB  FLOPS=%f M\n", memory, flops);

    // Test 2: Per-node via callbacks (runSessionWithCallBackInfo)
    fprintf(stderr, "\n--- Test 2: runSessionWithCallBackInfo (per-node) ---\n");
    int beforeCount = 0, afterCount = 0;
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) -> bool {
        if (info) {
            fprintf(stderr, "  BEFORE: name=%s type=%s flops=%.2f\n",
                    info->name().c_str(), info->type().c_str(), info->flops());
        } else {
            fprintf(stderr, "  BEFORE: null info!\n");
        }
        beforeCount++;
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) -> bool {
        if (info) {
            fprintf(stderr, "  AFTER:  name=%s type=%s flops=%.2f\n",
                    info->name().c_str(), info->type().c_str(), info->flops());
        }
        afterCount++;
        return true;
    };

    auto err = interpreter->runSessionWithCallBackInfo(session, before, after);
    if (err != MNN::NO_ERROR) {
        fprintf(stderr, "FAIL: runSessionWithCallBackInfo returned error %d\n", err);
    } else {
        fprintf(stderr, "OK: callbacks fired: before=%d after=%d\n", beforeCount, afterCount);
    }

    // Test 3: Second aggregate run to check session not stuck
    fprintf(stderr, "\n--- Test 3: runSession again (check session not stuck) ---\n");
    err = interpreter->runSession(session);
    if (err != MNN::NO_ERROR) {
        fprintf(stderr, "FAIL: runSession after callbacks returned error %d\n", err);
    } else {
        fprintf(stderr, "OK\n");
    }

    interpreter->releaseSession(session);
    MNN::Interpreter::destroy(interpreter);
    fprintf(stderr, "\nDone.\n");
    return 0;
}
