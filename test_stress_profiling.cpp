// test_stress_profiling.cpp - stress-test per-node profiling with many ops
// Build: g++ test_stress_profiling.cpp -I./include -L./build_vk/OFF -lMNN -lpthread -std=c++11 -o test_stress_profiling
// Run: LD_LIBRARY_PATH=./build_vk/OFF ./test_stress_profiling

#include <stdio.h>
#include <stdlib.h>
#include <MNN/Interpreter.hpp>
#include <MNN/ErrorCode.hpp>
#include "MNN_generated.h"
#include <memory>
#include <vector>

static MNN::Interpreter* buildBigModel(int numOps) {
    std::unique_ptr<MNN::NetT> netT(new MNN::NetT);
    netT->tensorName.push_back("input");

    // Input (1, 1, 64, 64)
    {
        std::unique_ptr<MNN::OpT> input(new MNN::OpT);
        input->type = MNN::OpType_Input;
        auto param = new MNN::InputT();
        param->dims = {1, 1, 64, 64};
        param->dtype = MNN::DataType_DT_FLOAT;
        param->dformat = MNN::MNN_DATA_FORMAT_NCHW;
        input->main.type = MNN::OpParameter_Input;
        input->main.value = param;
        input->outputIndexes.push_back(0);
        input->name = "input";
        netT->oplists.emplace_back(std::move(input));
        netT->tensorName.push_back("input_out");
    }

    int currentOutput = 0;
    for (int i = 0; i < numOps; i++) {
        int inIdx = currentOutput;
        int outIdx = currentOutput + 1;
        // Conv 3x3
        {
            std::unique_ptr<MNN::OpT> conv(new MNN::OpT);
            conv->type = MNN::OpType_Convolution;
            conv->inputIndexes.push_back(inIdx);
            conv->outputIndexes.push_back(outIdx);
            auto conv2D = new MNN::Convolution2DT();
            conv2D->common.reset(new MNN::Convolution2DCommonT());
            conv2D->common->padX = 1;
            conv2D->common->padY = 1;
            conv2D->common->kernelX = 3;
            conv2D->common->kernelY = 3;
            conv2D->common->strideX = 1;
            conv2D->common->strideY = 1;
            conv2D->common->outputCount = 1;
            // Random weights (small)
            conv2D->weight = std::vector<float>(3*3*1*1, 0.1f);
            conv2D->bias   = std::vector<float>(1, 0.0f);
            conv->main.type = MNN::OpParameter_Convolution2D;
            conv->main.value = conv2D;
            char name[64];
            snprintf(name, sizeof(name), "conv_%d", i);
            conv->name = name;
            netT->oplists.emplace_back(std::move(conv));
            netT->tensorName.push_back(std::string("conv_out_") + std::to_string(i));
            currentOutput = outIdx;
            outIdx++;
        }
        // Relu
        {
            std::unique_ptr<MNN::OpT> relu(new MNN::OpT);
            relu->type = MNN::OpType_ReLU;
            relu->inputIndexes.push_back(currentOutput);
            relu->outputIndexes.push_back(outIdx);
            char name[64];
            snprintf(name, sizeof(name), "relu_%d", i);
            relu->name = name;
            netT->oplists.emplace_back(std::move(relu));
            netT->tensorName.push_back(std::string("relu_out_") + std::to_string(i));
            currentOutput = outIdx;
        }
    }

    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, netT.get());
    builder.Finish(offset);
    auto netPtr = MNN::Net::Get(builder.GetBufferPointer());
    auto interpreter = MNN::Interpreter::createFromBuffer(builder.GetBufferPointer(), builder.GetSize());
    return interpreter;
}

int main(int argc, char* argv[]) {
    int numOps = 5;
    if (argc > 1) numOps = atoi(argv[1]);

    fprintf(stderr, "Building model with %d Conv+Relu pairs (%d total ops)...\n", numOps, numOps*2 + 1);
    auto* interpreter = buildBigModel(numOps);
    if (!interpreter) {
        fprintf(stderr, "FAIL: buildBigModel returned null\n");
        return 1;
    }
    fprintf(stderr, "Interpreter created\n");

    // Set debug mode for callbacks
    interpreter->setSessionMode(MNN::Interpreter::Session_Debug);

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 2;

    auto* session = interpreter->createSession(config);
    if (!session) {
        fprintf(stderr, "FAIL: createSession returned null\n");
        return 1;
    }
    fprintf(stderr, "Session created\n");

    // Get input tensor and set data
    auto* input = interpreter->getSessionInput(session, nullptr);
    if (input) {
        auto nchw = std::vector<int>{1, 1, 64, 64};
        interpreter->resizeTensor(input, nchw);
        // Fill with data
        float* data = (float*)input->host<void>();
        for (int i = 0; i < 64*64; i++) data[i] = 0.5f;
    }

    // Run aggregate first
    fprintf(stderr, "\n--- Test 1: runSession (aggregate) ---\n");
    auto err = interpreter->runSession(session);
    if (err != MNN::NO_ERROR) {
        fprintf(stderr, "FAIL: runSession returned error %d\n", err);
    } else {
        fprintf(stderr, "OK: runSession succeeded\n");
        // Get metrics
        float memory = 0;
        bool ok = interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &memory);
        fprintf(stderr, "  getSessionInfo(MEMORY): %s, value=%f\n", ok ? "ok" : "fail", memory);
        float flops = 0;
        ok = interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
        fprintf(stderr, "  getSessionInfo(FLOPS): %s, value=%f\n", ok ? "ok" : "fail", flops);
    }

    // Now run with callbacks
    fprintf(stderr, "\n--- Test 2: runSessionWithCallBackInfo (per-node) ---\n");
    int beforeCount = 0, afterCount = 0;
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) -> bool {
        if (info) {
            fprintf(stderr, "  BEFORE: %s (type=%s, flops=%.2f)\n", info->name().c_str(), info->type().c_str(), info->flops());
        } else {
            fprintf(stderr, "  BEFORE: info=nullptr!\n");
        }
        beforeCount++;
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) -> bool {
        if (info) {
            fprintf(stderr, "  AFTER:  %s (type=%s, flops=%.2f)\n", info->name().c_str(), info->type().c_str(), info->flops());
        } else {
            fprintf(stderr, "  AFTER: info=nullptr!\n");
        }
        afterCount++;
        return true;
    };

    err = interpreter->runSessionWithCallBackInfo(session, before, after);
    if (err != MNN::NO_ERROR) {
        fprintf(stderr, "FAIL: runSessionWithCallBackInfo returned error %d\n", err);
    } else {
        fprintf(stderr, "OK: callbacks fired %d before, %d after\n", beforeCount, afterCount);
        if (beforeCount > 0 && afterCount > 0) {
            fprintf(stderr, "PASS: per-node profiling works\n");
        } else {
            fprintf(stderr, "WARN: callbacks fired but counts seem off (before=%d, after=%d)\n", beforeCount, afterCount);
        }
    }

    // Run again without callbacks to confirm session is still usable
    fprintf(stderr, "\n--- Test 3: runSession again (check session not stuck) ---\n");
    err = interpreter->runSession(session);
    if (err != MNN::NO_ERROR) {
        fprintf(stderr, "FAIL: second runSession returned error %d\n", err);
    } else {
        fprintf(stderr, "OK: second runSession succeeded\n");
    }

    interpreter->releaseSession(session);
    MNN::Interpreter::destroy(interpreter);
    fprintf(stderr, "\nDone.\n");
    return 0;
}
