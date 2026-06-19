// test_stress_bayer.cpp - stress with 4K Bayer (4096x4096) input + many session clones
#include <stdio.h>
#include <stdlib.h>
#include <MNN/Interpreter.hpp>
#include <MNN/ErrorCode.hpp>

int main(int argc, char* argv[]) {
    const char* modelPath = nullptr;
    int inputSize = 2048;  // start at 2K, scale up
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) inputSize = atoi(argv[2]);
    if (!modelPath) {
        fprintf(stderr, "Usage: %s model.mnn [input_size]\n", argv[0]);
        return 1;
    }

    auto* interpreter = MNN::Interpreter::createFromFile(modelPath);
    if (!interpreter) {
        fprintf(stderr, "FAIL: createFromFile\n");
        return 1;
    }

    // Session_Debug mode
    interpreter->setSessionMode(MNN::Interpreter::Session_Debug);

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 2;

    auto* session = interpreter->createSession(config);
    if (!session) {
        fprintf(stderr, "FAIL: createSession\n");
        return 1;
    }

    auto* input = interpreter->getSessionInput(session, nullptr);
    if (!input) {
        fprintf(stderr, "FAIL: getSessionInput\n");
        return 1;
    }

    // Get original shape
    fprintf(stderr, "Original input shape: [");
    for (int i = 0; i < input->dimensions(); i++) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%d", input->length(i));
    }
    fprintf(stderr, "]\n");

    // Resize to large Bayer image
    std::vector<int> newDims;
    if (input->dimensions() == 4) {
        // NCHW: batch=1, channel=1, height, width
        newDims = {1, 1, inputSize, inputSize};
    } else if (input->dimensions() == 3) {
        newDims = {1, inputSize, inputSize};
    } else if (input->dimensions() == 2) {
        newDims = {inputSize, inputSize};
    } else {
        newDims = std::vector<int>(input->dimensions(), inputSize);
    }
    
    fprintf(stderr, "Resizing to: [");
    for (int i = 0; i < newDims.size(); i++) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%d", newDims[i]);
    }
    fprintf(stderr, "]\n");

    interpreter->resizeTensor(input, newDims);
    interpreter->resizeSession(session);

    // Fill input with data
    if (input->host<void>() && input->size() > 0) {
        float* data = (float*)input->host<void>();
        int count = input->size() / sizeof(float);
        for (int i = 0; i < count; i++) data[i] = 0.5f;
        fprintf(stderr, "Input size: %.1f MB (%d floats)\n", 
                (float)input->size() / (1024*1024), count);
    }

    // Test 1: Aggregate
    fprintf(stderr, "\n--- Test 1: runSession (aggregate) ---\n");
    auto err = interpreter->runSession(session);
    if (err != MNN::NO_ERROR) {
        fprintf(stderr, "FAIL: runSession error=%d\n", err);
    } else {
        float mem=0, flops=0;
        interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &mem);
        interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
        fprintf(stderr, "OK: mem=%.1f MB flops=%.1f M\n", mem, flops);
    }

    // Test 2: Per-node (even if aggregate failed, try callbacks)
    fprintf(stderr, "\n--- Test 2: runSessionWithCallBackInfo ---\n");
    int beforeCount=0, afterCount=0;
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& t, const MNN::OperatorInfo* info) -> bool {
        beforeCount++;
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& t, const MNN::OperatorInfo* info) -> bool {
        afterCount++;
        return true;
    };

    err = interpreter->runSessionWithCallBackInfo(session, before, after);
    if (err != MNN::NO_ERROR) {
        fprintf(stderr, "FAIL: runSessionWithCallBackInfo error=%d\n", err);
    } else {
        fprintf(stderr, "OK: callbacks before=%d after=%d\n", beforeCount, afterCount);
    }

    // Test 3: Second aggregate
    fprintf(stderr, "\n--- Test 3: runSession again ---\n");
    err = interpreter->runSession(session);
    fprintf(stderr, "result=%d\n", err);

    interpreter->releaseSession(session);
    MNN::Interpreter::destroy(interpreter);
    fprintf(stderr, "\nDone.\n");
    return 0;
}
