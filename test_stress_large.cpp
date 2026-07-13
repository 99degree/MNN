// test_stress_large.cpp - stress with many ops, no Express/Flatbuffer needed
// Just reuses the existing converted model repeatedly in a loop to simulate
// very large op counts, OR run many sessions sequentially.
#include <stdio.h>
#include <stdlib.h>
#include <MNN/Interpreter.hpp>
#include <MNN/ErrorCode.hpp>

int main(int argc, char* argv[]) {
    const char* modelPath = nullptr;
    int numSessions = 1;
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) numSessions = atoi(argv[2]);
    if (!modelPath) {
        fprintf(stderr, "Usage: %s model.mnn [num_sessions]\n", argv[0]);
        return 1;
    }

    for (int s = 0; s < numSessions; s++) {
        fprintf(stderr, "\n=== Session %d/%d ===\n", s+1, numSessions);

        auto* interpreter = MNN::Interpreter::createFromFile(modelPath);
        if (!interpreter) {
            fprintf(stderr, "FAIL: createFromFile\n");
            continue;
        }

        // Session_Debug mode for callbacks
        interpreter->setSessionMode(MNN::Interpreter::Session_Debug);

        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_CPU;
        config.numThread = 2;

        auto* session = interpreter->createSession(config);
        if (!session) {
            fprintf(stderr, "FAIL: createSession\n");
            MNN::Interpreter::destroy(interpreter);
            continue;
        }

        // Get input and fill with data
        auto* input = interpreter->getSessionInput(session, nullptr);
        if (input && input->host<void>()) {
            float* data = (float*)input->host<void>();
            int count = input->size() / sizeof(float);
            for (int i = 0; i < count; i++) data[i] = 0.5f;
        }

        // Aggregate run
        auto err = interpreter->runSession(session);
        if (err != MNN::NO_ERROR) {
            fprintf(stderr, "FAIL: runSession error=%d\n", err);
        } else {
            float mem=0, flops=0;
            interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &mem);
            interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
            fprintf(stderr, "  aggregate OK  mem=%.3f flops=%.3f\n", mem, flops);
        }

        // Per-node run
        int beforeCount=0, afterCount=0;
        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& t, const MNN::OperatorInfo* info) -> bool {
            if (info) beforeCount++;
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& t, const MNN::OperatorInfo* info) -> bool {
            if (info) afterCount++;
            return true;
        };

        err = interpreter->runSessionWithCallBackInfo(session, before, after);
        if (err != MNN::NO_ERROR) {
            fprintf(stderr, "FAIL: runSessionWithCallBackInfo error=%d\n", err);
        } else {
            fprintf(stderr, "  callbacks OK  before=%d after=%d\n", beforeCount, afterCount);
        }

        // Check session not stuck
        err = interpreter->runSession(session);
        fprintf(stderr, "  second aggregate: %s\n", err==MNN::NO_ERROR ? "OK" : "FAIL");

        interpreter->releaseSession(session);
        MNN::Interpreter::destroy(interpreter);
    }

    fprintf(stderr, "\nDone.\n");
    return 0;
}
